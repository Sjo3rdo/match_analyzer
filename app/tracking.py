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
    color: np.ndarray | None = None  # gemiddelde shirtkleur (Lab), voor herkenning

    def predicted(self) -> np.ndarray:
        return self.box + self.velocity


_LAB_W = np.array([0.5, 1.0, 1.0])


def match_scores(preds: np.ndarray, lost: np.ndarray, boxes: np.ndarray,
                 track_colors: list, det_colors: list) -> np.ndarray:
    """Hoe goed past elke detectie bij elke track (0..1)?

    Kleine, verre spelers verplaatsen zich per frame soms meer dan hun eigen breedte, dan
    overlappen de vakjes niet meer. Daarom kijken we vooral naar de afstand tussen de
    voetpunten, gemeten in spelerslengtes, en naar de shirtkleur.
    """
    if len(preds) == 0 or len(boxes) == 0:
        return np.zeros((len(preds), len(boxes)))
    iou = iou_matrix(preds, boxes)
    pf = np.stack([(preds[:, 0] + preds[:, 2]) / 2, preds[:, 3]], 1)
    bf = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]], 1)
    ph = np.maximum(1.0, preds[:, 3] - preds[:, 1])
    bh = np.maximum(1.0, boxes[:, 3] - boxes[:, 1])
    dist = np.linalg.norm(pf[:, None] - bf[None], axis=2)
    sigma = 0.35 * ph[:, None] * (1 + 0.5 * np.minimum(lost, 10))[:, None]
    geo = np.exp(-0.5 * (dist / sigma) ** 2)
    ratio = bh[None] / ph[:, None]
    size = np.exp(-0.5 * (np.log(ratio) / 0.25) ** 2)  # lengte mag niet ineens heel anders zijn
    score = np.maximum(iou, geo) * size
    for i, tc in enumerate(track_colors):
        if tc is None:
            continue
        for j, dc in enumerate(det_colors):
            if dc is not None:
                de = np.linalg.norm((tc - dc) * _LAB_W)
                score[i, j] *= 0.1 + 0.9 * np.exp(-0.5 * (de / 12) ** 2)
    return score


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
               H: np.ndarray | None = None, colors: list | None = None) -> list[tuple[int, np.ndarray, float]]:
        """Geeft (track_id, box, score) voor bevestigde tracks die in dit frame gezien zijn.

        colors: optioneel de shirtkleur (Lab) per detectie (of None), voor betere herkenning."""
        boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        colors = list(colors) if colors is not None else [None] * len(boxes)
        if H is not None and not np.allclose(H, np.eye(3)):
            for t in self.tracks:
                t.box = _warp_box(H, t.box)
        preds = np.array([t.predicted() for t in self.tracks]).reshape(-1, 4)

        out: list[tuple[int, np.ndarray, float]] = []
        unmatched_tracks = set(range(len(self.tracks)))
        hi = np.where(scores >= self.high)[0]
        lo = np.where((scores >= self.low) & (scores < self.high))[0]
        unmatched_hi = set(hi.tolist())

        lost = np.array([t.lost for t in self.tracks])
        for det_idx, thresh in ((hi, self.match_iou), (lo, 0.3)):
            tr = sorted(unmatched_tracks)
            if not tr or len(det_idx) == 0:
                continue
            m = match_scores(preds[tr], lost[tr], boxes[det_idx], [self.tracks[i].color for i in tr],
                             [colors[j] for j in det_idx])
            for i, j in greedy_match(m, thresh):
                t = self.tracks[tr[i]]
                d = int(det_idx[j])
                new = boxes[d]
                t.velocity = 0.6 * t.velocity + 0.4 * (new - t.box) / (t.lost + 1)
                t.box, t.lost = new, 0
                t.hits += 1
                if colors[d] is not None:
                    t.color = colors[d].astype(np.float64) if t.color is None else 0.85 * t.color + 0.15 * colors[d]
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
            self.tracks.append(Track(self._next, boxes[d].copy(),
                                     color=None if colors[d] is None else colors[d].astype(np.float64)))
            if self.min_hits <= 1:
                out.append((self._next, boxes[d].copy(), float(scores[d])))
            self._next += 1
        self.tracks = [t for t in self.tracks
                       if t.lost <= self.max_lost and not (t.lost > 0 and t.hits < self.min_hits)]
        return out


def stitch_tracks(tracklets: dict[int, dict], fps: float, max_gap_s: float = 3.0) -> dict[int, int]:
    """Aan elkaar plakken van stukjes track van dezelfde speler (offline, na het volgen).

    tracklets: id -> {"idx": frame-indices, "foot": voetpunten in cameragecorrigeerde
    coördinaten (n, 2), "h": spelerslengte in px (n,), "color": Lab of None}.
    Een stuk B volgt op A als B kort na A begint, ongeveer waar A naartoe liep, met dezelfde
    lengte en shirtkleur. Geeft id -> id van het eerste stuk van de keten.
    """
    info = {}
    for tid, tr in tracklets.items():
        idx, foot, h = np.asarray(tr["idx"]), np.asarray(tr["foot"], float), np.asarray(tr["h"], float)
        if len(idx) == 0:
            continue
        k = max(0, len(idx) - 6)
        dt = max(1, idx[-1] - idx[k])
        vel = (foot[-1] - foot[k]) / dt if len(idx) > 1 else np.zeros(2)
        vmax = 0.6 * h[-1]  # hoogstens ~0,6 lichaamslengte per frame (10 fps)
        sp = np.linalg.norm(vel)
        if sp > vmax:
            vel = vel * vmax / sp
        info[tid] = dict(s=idx[0], e=idx[-1], p0=foot[0], p1=foot[-1], h0=np.median(h[:5]), h1=np.median(h[-5:]),
                         v=vel, c=tr.get("color"))
    max_gap = max(1, int(round(max_gap_s * fps)))
    cands = []
    for a, A in info.items():
        for b, B in info.items():
            gap = B["s"] - A["e"]
            if a == b or gap <= 0 or gap > max_gap:
                continue
            ratio = B["h0"] / max(1.0, A["h1"])
            if not 0.67 < ratio < 1.5:
                continue
            de = 0.0
            if A["c"] is not None and B["c"] is not None:
                de = float(np.linalg.norm((np.asarray(A["c"]) - np.asarray(B["c"])) * _LAB_W))
                if de > 15:
                    continue
            hh = 0.5 * (A["h1"] + B["h0"])
            pred = A["p1"] + A["v"] * min(gap, fps)  # niet eindeloos doorrekenen
            gate = hh * (1.0 + 0.35 * gap * 10 / fps)
            dist = float(np.linalg.norm(B["p0"] - pred))
            if dist > gate:
                continue
            cands.append((dist / gate + 0.5 * de / 18 + 0.02 * gap, a, b))
    nxt, prev = {}, {}
    for _, a, b in sorted(cands):
        if a in nxt or b in prev:
            continue
        nxt[a], prev[b] = b, a
    root = {}
    for tid in info:
        r = tid
        while r in prev:
            r = prev[r]
        root[tid] = r
    return root
