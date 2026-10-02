"""Spelers volgen over frames heen (ByteTrack-achtig, met cameracompensatie).

Metafoor: elke track is een post-it op het scherm. Bij een nieuw frame schuiven we
eerst alle post-its mee met de camerabeweging, voorspellen we waar de speler naartoe
liep, en plakken we elke nieuwe detectie op de best overlappende post-it. Zekere
detecties krijgen eerst een kans, twijfelgevallen daarna (zo blijft een half verborgen
speler gevolgd). Een post-it die te lang niets vindt, valt eraf.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .calibration import apply_h


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None] - inter + 1e-9)


def greedy_match(score: np.ndarray, thresh: float) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    if score.size == 0:
        return pairs
    s = score.copy()
    while True:
        i, j = np.unravel_index(np.argmax(s), s.shape)
        if s[i, j] < thresh:
            return pairs
        pairs.append((int(i), int(j)))
        s[i, :] = -1
        s[:, j] = -1


@dataclass
class Track:
    id: int
    box: np.ndarray
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(4))
    hits: int = 1
    lost: int = 0

    def predicted(self) -> np.ndarray:
        return self.box + self.velocity


def _warp_box(H: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Box meeschuiven met de camera: voetpunt en hoogte via de homografie."""
    x1, y1, x2, y2 = box
    pts = apply_h(H, np.array([[(x1 + x2) / 2, y2], [(x1 + x2) / 2, y1]]))
    cx, foot_y = pts[0]
    top_y = pts[1][1]
    h = max(1.0, foot_y - top_y)
    w = (x2 - x1) * h / max(1.0, y2 - y1)
    return np.array([cx - w / 2, top_y, cx + w / 2, foot_y])


class Tracker:
    def __init__(self, fps: float, high: float = 0.4, low: float = 0.15,
                 match_iou: float = 0.2, lost_seconds: float = 2.0, min_hits: int = 3):
        self.high, self.low, self.match_iou, self.min_hits = high, low, match_iou, min_hits
        self.max_lost = max(1, int(round(lost_seconds * fps)))
        self.tracks: list[Track] = []
        self._next = 1

    def update(self, boxes: np.ndarray, scores: np.ndarray,
               H: np.ndarray | None = None) -> list[tuple[int, np.ndarray, float]]:
        """Geeft (track_id, box, score) voor bevestigde tracks die in dit frame gezien zijn."""
        boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        if H is not None and not np.allclose(H, np.eye(3)):
            for t in self.tracks:
                t.box = _warp_box(H, t.box)
        preds = np.array([t.predicted() for t in self.tracks]).reshape(-1, 4)

        out: list[tuple[int, np.ndarray, float]] = []
        unmatched_tracks = set(range(len(self.tracks)))
        hi = np.where(scores >= self.high)[0]
        lo = np.where((scores >= self.low) & (scores < self.high))[0]
        unmatched_hi = set(hi.tolist())

        for det_idx, thresh in ((hi, self.match_iou), (lo, 0.3)):
            tr = sorted(unmatched_tracks)
            if not tr or len(det_idx) == 0:
                continue
            m = iou_matrix(preds[tr], boxes[det_idx])
            for i, j in greedy_match(m, thresh):
                t = self.tracks[tr[i]]
                d = int(det_idx[j])
                new = boxes[d]
                t.velocity = 0.6 * t.velocity + 0.4 * (new - t.box) / (t.lost + 1)
                t.box, t.lost = new, 0
                t.hits += 1
                unmatched_tracks.discard(tr[i])
                unmatched_hi.discard(d)
                if t.hits >= self.min_hits:
                    out.append((t.id, new.copy(), float(scores[d])))

        for t_idx in unmatched_tracks:
            t = self.tracks[t_idx]
            t.lost += 1
            t.box = t.predicted()
            t.velocity *= 0.5
        for d in sorted(unmatched_hi):
            self.tracks.append(Track(self._next, boxes[d].copy()))
            if self.min_hits <= 1:
                out.append((self._next, boxes[d].copy(), float(scores[d])))
            self._next += 1
        self.tracks = [t for t in self.tracks
                       if t.lost <= self.max_lost and not (t.lost > 0 and t.hits < self.min_hits)]
        return out
