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

import math
from dataclasses import dataclass

import cv2
import numpy as np


def fit_homography(image_pts: np.ndarray, pitch_pts: np.ndarray) -> tuple[np.ndarray, float]:
    """Homografie beeld -> veld uit alleen punten, plus gemiddelde fout in meters."""
    return fit_calibration([{"img": a, "pitch": b} for a, b in zip(np.asarray(image_pts).reshape(-1, 2),
                                                                    np.asarray(pitch_pts).reshape(-1, 2))])


def _norm_matrix(pts: np.ndarray) -> np.ndarray:
    c = pts.mean(axis=0)
    d = np.mean(np.linalg.norm(pts - c, axis=1)) or 1.0
    k = np.sqrt(2) / d
    return np.array([[k, 0, -k * c[0]], [0, k, -k * c[1]], [0, 0, 1]])


def _line_coeffs(seg) -> np.ndarray:
    (x1, y1), (x2, y2) = seg
    l = np.array([y1 - y2, x2 - x1, x1 * y2 - x2 * y1], dtype=np.float64)
    return l / (np.hypot(l[0], l[1]) or 1.0)


def calibration_dof(points: list[dict]) -> int:
    """Hoeveel 'informatie' de kalibratie heeft: een punt telt 2, elke lijn hoogstens 2."""
    per_line: dict[str, int] = {}
    n = 0
    for p in points:
        if "pitch" in p and p["pitch"] is not None:
            n += 2
        elif p.get("line"):
            key = str(p["line"])
            per_line[key] = min(2, per_line.get(key, 0) + 1)
    return n + sum(per_line.values())


def calibration_residuals(K: np.ndarray, points: list[dict]) -> np.ndarray:
    """Fout per kalibratiepunt in meters (punt: afstand tot het veldpunt, lijnpunt: tot de lijn)."""
    out = []
    for p in points:
        q = apply_h(K, np.asarray(p["img"], float))[0]
        if "pitch" in p and p["pitch"] is not None:
            out.append(np.linalg.norm(q - np.asarray(p["pitch"], float)))
        else:
            l = _line_coeffs(p["line"])
            out.append(abs(l[0] * q[0] + l[1] * q[1] + l[2]))
    return np.array(out)


def fit_keyframe(kf: dict, camera: dict | None = None) -> tuple[np.ndarray, float]:
    """Kalibratie van een sleutelframe. Automatische sleutelframes bevatten exacte punten
    (een raster uit de lijnen-fit), daar is geen cameramodel of verfijning voor nodig."""
    if kf.get("auto"):
        return _fit_free(kf["points"], refine=False)
    return fit_calibration(kf["points"], camera=camera)


def fit_calibration(points: list[dict], camera: dict | None = None) -> tuple[np.ndarray, float]:
    """Kalibratie; met `camera` (voorkennis over positie/hoogte) via het cameramodel."""
    if camera is not None:
        try:
            K, _ = fit_camera(points, camera)
            err = float(np.mean(calibration_residuals(K, points)))
        except (ValueError, np.linalg.LinAlgError):
            if calibration_dof(points) < 8:
                raise
            K, err = None, np.inf
        if calibration_dof(points) >= 8:
            # genoeg aangeklikt: kies wat het beste bij de klikken past
            try:
                K2, err2 = _fit_free(points)
                if err2 < 0.5 * err:
                    return K2, err2
            except ValueError:
                pass
        if K is not None:
            return K, err
    return _fit_free(points)


def _fit_free(points: list[dict], refine: bool = True) -> tuple[np.ndarray, float]:
    """Homografie beeld -> veld uit punten én punten-op-een-lijn.

    points: [{"img": [x, y], "pitch": [X, Y]}]  (bekend veldpunt), of
            [{"img": [x, y], "line": [[X1, Y1], [X2, Y2]]}]  (ergens op die veldlijn).
    Een punt levert twee vergelijkingen, een punt-op-lijn één (l^T K x = 0). Samen lineair
    oplossen (genormaliseerd), daarna verfijnen op de echte fout in meters.
    """
    if calibration_dof(points) < 8:
        raise ValueError("Te weinig informatie: gebruik minstens 4 punten, of combineer punten "
                         "met lijnen (elke lijn telt mee met hoogstens 2 punten)")
    img = np.array([p["img"] for p in points], dtype=np.float64)
    world = [np.asarray(p["pitch"], float) for p in points if p.get("pitch") is not None]
    for p in points:
        if p.get("pitch") is None:
            world += [np.asarray(p["line"][0], float), np.asarray(p["line"][1], float)]
    Ti, Tp = _norm_matrix(img), _norm_matrix(np.array(world))
    Tp_inv_T = np.linalg.inv(Tp).T
    rows = []
    for p, (x, y) in zip(points, apply_h(Ti, img)):
        if p.get("pitch") is not None:
            X, Y = apply_h(Tp, np.asarray(p["pitch"], float))[0]
            rows.append([x, y, 1, 0, 0, 0, -X * x, -X * y, -X])
            rows.append([0, 0, 0, x, y, 1, -Y * x, -Y * y, -Y])
        else:
            a, b, c = Tp_inv_T @ _line_coeffs(p["line"])
            n = np.hypot(a, b) or 1.0
            a, b, c = a / n, b / n, c / n
            rows.append([a * x, a * y, a, b * x, b * y, b, c * x, c * y, c])
    A = np.array(rows)
    _, sv, vt = np.linalg.svd(A)
    if len(sv) >= 8 and sv[7] < 1e-9 * sv[0]:
        raise ValueError("Deze combinatie legt het veld nog niet vast (bijv. precies 2 punten + 2 lijnen, "
                         "of alles op één lijn). Voeg nog een punt of een andere lijn toe.")
    Kn = vt[-1].reshape(3, 3)
    K = np.linalg.inv(Tp) @ Kn @ Ti
    K = _refine(K / K[2, 2], points) if refine else K / K[2, 2]
    if not np.all(np.isfinite(K)):
        raise ValueError("Kalibratie mislukt")
    return K, float(np.mean(calibration_residuals(K, points)))


def _refine(K: np.ndarray, points: list[dict], iters: int = 15) -> np.ndarray:
    """Gauss-Newton op de fouten in meters (punten 2D, lijnpunten 1D)."""
    def res(h):
        Kh = np.append(h, 1.0).reshape(3, 3)
        r = []
        for p in points:
            q = apply_h(Kh, np.asarray(p["img"], float))[0]
            if p.get("pitch") is not None:
                r += list(q - np.asarray(p["pitch"], float))
            else:
                l = _line_coeffs(p["line"])
                r.append(l[0] * q[0] + l[1] * q[1] + l[2])
        return np.array(r)
    h = K.reshape(-1)[:8].copy()
    r = res(h)
    if not np.all(np.isfinite(r)):
        return K
    for _ in range(iters):
        J = np.empty((len(r), 8))
        for k in range(8):
            e = 1e-7 * max(1.0, abs(h[k]))
            hk = h.copy()
            hk[k] += e
            J[:, k] = (res(hk) - r) / e
        try:
            step = np.linalg.lstsq(J, -r, rcond=None)[0]
        except np.linalg.LinAlgError:
            break
        h_new = h + step
        r_new = res(h_new)
        if not np.all(np.isfinite(r_new)) or r_new @ r_new >= r @ r:
            break
        h, r = h_new, r_new
    return np.append(h, 1.0).reshape(3, 3)


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


# --- cameramodel: als bekend is waar de camera ongeveer stond ---------------------------------
#
# Een homografie heeft 8 onbekenden. Een echte camera boven een plat veld heeft er 7: positie
# (x, y), hoogte, draaien (yaw), kantelen (tilt), scheef houden (roll) en brandpuntsafstand
# (zoom). Weten we positie en hoogte ongeveer, en is roll klein, dan blijven er praktisch
# alleen draaien, kantelen en zoom over: dan is 1 punt + 1 lijn al genoeg.
#
# Coördinaten: X langs het veld, Y naar de onderste zijlijn, Z omlaag (rechtshandig), dus
# een camera op 1,6 m hoogte staat op Z = -1,6.

DEFAULT_HFOV_DEG = 64.0  # telefoon, hoofdlens (1x), video met stabilisatie


def camera_homography(params: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Veld -> beeld voor camera-parameters [x, y, h, yaw, tilt, roll, log_f]."""
    cx, cy, h, yaw, tilt, roll, logf = params
    W, Hh = size
    f = math.exp(logf)
    fwd = np.array([math.cos(yaw) * math.cos(tilt), math.sin(yaw) * math.cos(tilt), math.sin(tilt)])
    down = np.array([0.0, 0.0, 1.0])
    right = np.cross(down, fwd)
    right /= np.linalg.norm(right) or 1.0
    dn = np.cross(fwd, right)
    cr, sr = math.cos(roll), math.sin(roll)
    right, dn = cr * right + sr * dn, -sr * right + cr * dn
    R = np.stack([right, dn, fwd])
    C = np.array([cx, cy, -h])
    Kc = np.array([[f, 0, W / 2], [0, f, Hh / 2], [0, 0, 1.0]])
    H = Kc @ np.column_stack([R[:, 0], R[:, 1], -R @ C])
    return H / H[2, 2] if abs(H[2, 2]) > 1e-12 else H


def default_focal(width: int) -> float:
    return (width / 2) / math.tan(math.radians(DEFAULT_HFOV_DEG / 2))


def _cam_residuals(p: np.ndarray, points: list[dict], prior: dict, size, sigma_px: float = 4.0) -> np.ndarray:
    H = camera_homography(p, size)
    Hinv_T = np.linalg.inv(H).T
    r = []
    for q in points:
        x, y = q["img"]
        if q.get("pitch") is not None:
            X, Y = q["pitch"]
            v = H @ np.array([X, Y, 1.0])
            if v[2] <= 1e-6:  # achter de camera: zware straf
                r += [50.0, 50.0]
                continue
            r += [(v[0] / v[2] - x) / sigma_px, (v[1] / v[2] - y) / sigma_px]
        else:
            l = Hinv_T @ _line_coeffs(q["line"])
            n = math.hypot(l[0], l[1]) or 1.0
            r.append((l[0] * x + l[1] * y + l[2]) / n / sigma_px)
    sp = max(3.0, float(prior.get("sigma_pos", 5.0)))
    r += [(p[0] - prior["x"]) / sp, (p[1] - prior["y"]) / sp, (p[2] - prior["h"]) / 0.5,
          p[5] / math.radians(4), (p[6] - math.log(prior["f"])) / math.log(2.2)]
    return np.array(r)


def _lm(fun, p0: np.ndarray, iters: int = 60) -> np.ndarray:
    """Levenberg-Marquardt met numerieke afgeleiden (klein probleem, 7 parameters)."""
    p, r = p0.copy(), fun(p0)
    lam = 1e-2
    for _ in range(iters):
        J = np.empty((len(r), len(p)))
        for k in range(len(p)):
            e = 1e-6 * max(1.0, abs(p[k]))
            pk = p.copy()
            pk[k] += e
            J[:, k] = (fun(pk) - r) / e
        A, g = J.T @ J, J.T @ r
        improved = False
        for _ in range(8):
            try:
                step = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-9), -g)
            except np.linalg.LinAlgError:
                break
            r_new = fun(p + step)
            if np.all(np.isfinite(r_new)) and r_new @ r_new < r @ r:
                p, r, lam, improved = p + step, r_new, max(lam / 3, 1e-7), True
                break
            lam *= 4
        if not improved or np.linalg.norm(step) < 1e-9:
            break
    return p


def fit_camera(points: list[dict], prior: dict) -> tuple[np.ndarray, dict]:
    """Camera-fit met voorkennis. prior: {x, y, h, f, sigma_pos, width, height}.

    Geeft (homografie beeld -> veld, cameraparameters)."""
    size = (int(prior["width"]), int(prior["height"]))
    data_dof = calibration_dof(points)
    if data_dof < 3:
        raise ValueError("Met de camerapositie is minder nodig, maar nog wel minstens 1 punt + 1 lijn "
                         "(of 2 lijnen, of 2 punten)")
    fun = lambda p: _cam_residuals(p, points, prior, size)  # noqa: E731
    # Startwaarden: alle kijkrichtingen proberen, de beste verfijnen
    starts = []
    for yaw in np.radians(np.arange(0, 360, 10)):
        for tilt in np.radians([2, 5, 10, 18, 30, 45]):
            p0 = np.array([prior["x"], prior["y"], prior["h"], yaw, tilt, 0.0, math.log(prior["f"])])
            starts.append((float(np.sum(fun(p0) ** 2)), p0))
    starts.sort(key=lambda c: c[0])
    fits = [_lm(fun, p0) for _, p0 in starts[:4]]
    p = min(fits, key=lambda q: float(np.sum(fun(q) ** 2)))
    H = camera_homography(p, size)
    K = np.linalg.inv(H)
    K /= K[2, 2]
    cam = {"x": float(p[0]), "y": float(p[1]), "h": float(p[2]), "yaw_deg": math.degrees(p[3]) % 360,
           "tilt_deg": math.degrees(p[4]), "roll_deg": math.degrees(p[5]),
           "hfov_deg": math.degrees(2 * math.atan(size[0] / 2 / math.exp(p[6])))}
    return K, cam


def camera_prior(clip: dict) -> dict | None:
    """Voorkennis over de camera uit de clipgegevens (positie, hoogte), of None."""
    if clip.get("cam_x") is None or clip.get("cam_y") is None or not clip.get("width"):
        return None
    sigma = 3.0 if clip.get("cam_source") == "hand" else max(4.0, float(clip.get("gps_acc") or 8.0) + 3.0)
    return {"x": float(clip["cam_x"]), "y": float(clip["cam_y"]), "h": float(clip.get("cam_h") or 1.6),
            "f": default_focal(int(clip["width"])), "sigma_pos": sigma,
            "width": int(clip["width"]), "height": int(clip["height"])}
