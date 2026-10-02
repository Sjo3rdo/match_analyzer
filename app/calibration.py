"""Veldkalibratie voor een bewegende (handheld) camera.

Werkwijze, als metafoor: denk aan een doorzichtige plattegrond van het veld die je
over het beeld legt.

1. Op een paar *sleutelframes* klikt de gebruiker herkenbare punten aan (hoekvlag,
   16 m-lijn, middenstip ...). Daaruit volgt een homografie K: beeld -> veld (meters).
2. Tijdens de analyse meten we tussen opeenvolgende frames hoe de camera bewoog
   (draaien/zoomen) met optical flow op de achtergrond. Dat geeft per frame H_i:
   frame i-1 -> frame i.
3. Voor elk frame t "schuiven" we de plattegrond van het dichtstbijzijnde sleutelframe
   mee met die camerabeweging. Ligt t tussen twee sleutelframes, dan mengen we beide
   schattingen, zodat opgestapelde fouten (drift) beperkt blijven.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


def fit_homography(image_pts: np.ndarray, pitch_pts: np.ndarray) -> tuple[np.ndarray, float]:
    """Homografie beeld -> veld plus gemiddelde terugprojectiefout in meters."""
    image_pts = np.asarray(image_pts, dtype=np.float64).reshape(-1, 2)
    pitch_pts = np.asarray(pitch_pts, dtype=np.float64).reshape(-1, 2)
    if len(image_pts) < 4:
        raise ValueError("Minstens 4 punten nodig voor kalibratie")
    method = cv2.RANSAC if len(image_pts) >= 6 else 0
    K, _ = cv2.findHomography(image_pts, pitch_pts, method, 2.0)
    if K is None:
        raise ValueError("Kalibratie mislukt: punten liggen (bijna) op één lijn")
    err = float(np.mean(np.linalg.norm(apply_h(K, image_pts) - pitch_pts, axis=1)))
    return K / K[2, 2], err


def apply_h(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if len(pts) == 0:
        return pts
    hom = np.hstack([pts, np.ones((len(pts), 1))]) @ H.T
    with np.errstate(divide="ignore", invalid="ignore"):
        return hom[:, :2] / hom[:, 2:3]


class MotionEstimator:
    """Schat camerabeweging tussen frames uit achtergrondpunten (spelers gemaskeerd)."""

    def __init__(self, work_width: int = 960):
        self.work_width = work_width
        self.prev_gray: np.ndarray | None = None
        self.scale = 1.0

    def _prep(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        self.scale = min(1.0, self.work_width / w)
        small = cv2.resize(frame, None, fx=self.scale, fy=self.scale) if self.scale < 1 else frame
        return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    def step(self, frame: np.ndarray, boxes: np.ndarray | None = None) -> np.ndarray:
        """Geeft H (vorig frame -> dit frame) in originele pixelcoördinaten."""
        gray = self._prep(frame)
        prev, self.prev_gray = self.prev_gray, gray
        if prev is None:
            return np.eye(3)
        mask = np.full(prev.shape, 255, np.uint8)
        if boxes is not None:
            for x1, y1, x2, y2 in (np.asarray(boxes) * self.scale).astype(int):
                pad = int(0.15 * (y2 - y1))
                mask[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad] = 0
        p0 = cv2.goodFeaturesToTrack(prev, maxCorners=600, qualityLevel=0.01,
                                     minDistance=8, mask=mask)
        if p0 is None or len(p0) < 12:
            return np.eye(3)
        p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, gray, p0, None, winSize=(21, 21), maxLevel=4)
        good = st.reshape(-1) == 1
        if good.sum() < 12:
            return np.eye(3)
        H, inl = cv2.findHomography(p0[good], p1[good], cv2.RANSAC, 3.0)
        if H is None or inl.sum() < 10 or not _plausible(H):
            return np.eye(3)
        S = np.diag([self.scale, self.scale, 1.0])
        H = np.linalg.inv(S) @ H @ S
        return H / H[2, 2]


def _plausible(H: np.ndarray) -> bool:
    """Wijs extreme sprongen af (een handheld camera beweegt beperkt per frame)."""
    det = np.linalg.det(H[:2, :2])
    return 0.7 < det < 1.4 and abs(H[2, 0]) < 1e-2 and abs(H[2, 1]) < 1e-2


def cumulative(inter: np.ndarray) -> np.ndarray:
    """A_t: frame t -> frame 0 (referentie), uit H_i: frame i-1 -> frame i."""
    n = len(inter)
    A = np.empty((n, 3, 3))
    A[0] = np.eye(3)
    for i in range(1, n):
        M = A[i - 1] @ np.linalg.inv(inter[i])
        A[i] = M / M[2, 2]
    return A


@dataclass
class Keyframe:
    idx: int  # geanalyseerd-frame-index
    K: np.ndarray  # beeld (op dit frame) -> veld


class CameraModel:
    """Projecteert beeldpunten op het veld voor elk geanalyseerd frame."""

    def __init__(self, inter: np.ndarray, keyframes: list[Keyframe]):
        self.A = cumulative(inter) if len(inter) else np.zeros((0, 3, 3))
        self.keyframes = sorted(keyframes, key=lambda k: k.idx)
        # Per sleutelframe: veld <- referentieframe
        self._G = [k.K @ np.linalg.inv(self.A[k.idx]) for k in self.keyframes]
        self._kidx = np.array([k.idx for k in self.keyframes])

    @property
    def calibrated(self) -> bool:
        return len(self.keyframes) > 0

    def homographies(self, idx: int) -> list[tuple[np.ndarray, float]]:
        """(H beeld->veld, gewicht) voor frame idx; mengt de twee omliggende sleutelframes."""
        if not self.calibrated:
            return []
        pos = int(np.searchsorted(self._kidx, idx))
        if pos == 0:
            return [(self._G[0] @ self.A[idx], 1.0)]
        if pos >= len(self._kidx):
            return [(self._G[-1] @ self.A[idx], 1.0)]
        a, b = pos - 1, pos
        ia, ib = self._kidx[a], self._kidx[b]
        w = (ib - idx) / max(1, ib - ia)
        return [(self._G[a] @ self.A[idx], w), (self._G[b] @ self.A[idx], 1 - w)]

    def project(self, idx: int, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        hs = self.homographies(idx)
        if not hs:
            return np.full_like(pts, np.nan)
        out = np.zeros_like(pts)
        for H, w in hs:
            out += w * apply_h(H, pts)
        return out
