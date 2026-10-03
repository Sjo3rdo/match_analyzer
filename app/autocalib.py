"""Automatisch bijstellen van de kalibratie aan de hand van de witte veldlijnen.

Metafoor: de kalibratie is een doorzichtig vel met de veldlijnen erop. De camerabeweging
schuift dat vel mee, maar na een tijdje gaat het een beetje scheef liggen (drift). Elke
seconde zoeken we in het beeld de echte witte lijnen en draaien/zoomen we het vel een klein
beetje bij tot de getekende lijnen er zo goed mogelijk op vallen. Dat resultaat wordt een
automatisch sleutelframe.

Werkwijze per beeld:
1. Lijnen vinden: dunne, lichte strepen op het gras (een "top-hat"-filter laat smalle lichte
   structuren over en negeert brede maaistroken); spelers worden weggemaskeerd.
2. Afstandskaart: per pixel de afstand tot de dichtstbijzijnde gevonden lijn.
3. Het veldmodel (punten om de halve meter op alle lijnen) projecteren met de voorspelde
   kalibratie en een kleine cameradraaiing + zoom zoeken waarbij die punten zo dicht mogelijk
   op gevonden lijnen vallen (Nelder-Mead, robuuste kosten).
4. Alleen accepteren als genoeg van het veldmodel echt op lijnen valt.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import cv2
import numpy as np

from . import pitch
from .calibration import (CameraModel, Keyframe, apply_h, camera_prior, cumulative, fit_calibration, fit_camera,
                          normalize_h)

log = logging.getLogger(__name__)

WORK_WIDTH = 960
PRECISION_MIN = 0.85  # deel van de gevonden lijnpixels dat op het model moet vallen
MAX_ROT_STEP_DEG = 0.8  # grootste toegestane correctie per bijstelling
MAX_ZOOM_STEP = 0.02
MIN_SPREAD = 0.08  # lijnrichtingen: 0 = alles evenwijdig, 1 = alle kanten op


def pitch_samples(step: float = 0.5, with_tangents: bool = False):
    """Punten op alle veldlijnen (meters)."""
    L, W, c = pitch.LENGTH, pitch.WIDTH, pitch.HALF_W
    segs = [((0, 0), (L, 0)), ((0, W), (L, W)), ((0, 0), (0, W)), ((L, 0), (L, W)), ((L / 2, 0), (L / 2, W))]
    for x0, s in ((0.0, 1), (L, -1)):
        for d, half in ((16.5, 20.16), (5.5, 9.16)):
            segs += [((x0, c - half), (x0 + s * d, c - half)), ((x0, c + half), (x0 + s * d, c + half)),
                     ((x0 + s * d, c - half), (x0 + s * d, c + half))]
    pts, tans = [], []
    for a, b in segs:
        a, b = np.array(a, float), np.array(b, float)
        n = max(2, int(np.linalg.norm(b - a) / step) + 1)
        pts.append(a + np.linspace(0, 1, n)[:, None] * (b - a))
        tans.append(np.tile((b - a) / np.linalg.norm(b - a), (n, 1)))
    r = 9.15
    for cx, a0, a1 in ((L / 2, 0, 2 * math.pi), (11, -math.acos(5.5 / r), math.acos(5.5 / r)),
                       (L - 11, math.pi - math.acos(5.5 / r), math.pi + math.acos(5.5 / r))):
        n = max(8, int(r * (a1 - a0) / step))
        t = np.linspace(a0, a1, n)
        pts.append(np.stack([cx + r * np.cos(t), c + r * np.sin(t)], 1))
        tans.append(np.stack([-np.sin(t), np.cos(t)], 1))
    if with_tangents:
        return np.vstack(pts), np.vstack(tans)
    return np.vstack(pts)


SAMPLES, TANGENTS = pitch_samples(with_tangents=True)
DENSE = pitch_samples(step=0.1)  # om de modellijnen als plaatje te tekenen


def detect_lines(frame: np.ndarray, boxes: np.ndarray | None = None, roi: np.ndarray | None = None) -> np.ndarray:
    """Masker (0/255, werkschaal) met witte veldlijnen.

    roi: optioneel masker (werkschaal) van waar het veld ongeveer ligt; daarbuiten negeren we
    alles (bomen, hekken, reclameborden, publiek)."""
    s = WORK_WIDTH / frame.shape[1]
    small = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.int16)
    L, A, B = lab[..., 0], lab[..., 1] - 128, lab[..., 2] - 128
    # 1. lichter dan de directe omgeving (smalle structuur: top-hat)
    th = cv2.morphologyEx(lab[..., 0].astype(np.uint8), cv2.MORPH_TOPHAT,
                          cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))).astype(np.int16)
    # 2. wit: weinig kleur (grassprieten in de zon zijn licht maar groen/geel)
    chroma = np.sqrt(A.astype(np.float32) ** 2 + B.astype(np.float32) ** 2)
    local = cv2.blur(lab[..., 0].astype(np.uint8), (25, 25)).astype(np.int16)
    mask = (th > 22) & (chroma < 22) & (L > local + 15)
    out = mask.astype(np.uint8) * 255
    if roi is not None:
        out &= roi
    if boxes is not None:
        for x1, y1, x2, y2 in (np.asarray(boxes).reshape(-1, 4) * s).astype(int):
            pad = int(0.1 * (y2 - y1)) + 2
            out[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad] = 0
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    # 3. alleen dunne, lange stukken (lijnen), geen vlekjes (shirts, schoenen, bal, glinstering).
    #    "Dun" = de dikste plek van het stuk is smal; "lang" = het stuk strekt zich ver uit.
    #    (Aan elkaar hangende lijnen vormen één netwerk; dat is dun én lang, dus blijft staan.)
    n, lab_cc, stats, _ = cv2.connectedComponentsWithStats(out, connectivity=8)
    half_width = cv2.distanceTransform(out, cv2.DIST_L2, 3)
    keep = np.zeros(n, bool)
    for k in range(1, n):
        x, y, w, h, area = stats[k]
        if area < 10:
            continue
        sub = lab_cc[y:y + h, x:x + w] == k
        thick = 2 * float(half_width[y:y + h, x:x + w][sub].max())
        extent = float(np.hypot(w, h))
        keep[k] = extent >= 15 and extent >= 5 * max(thick, 1.5)
    return np.where(keep[lab_cc], 255, 0).astype(np.uint8)


def centerlines(mask: np.ndarray) -> np.ndarray:
    """Hartlijnen (1 pixel breed) van de gevonden lijnen: morfologisch skelet.

    Brede lijnen vlak bij de camera zouden anders een "speelruimte" van een halve lijnbreedte
    geven waarbinnen het model ongemerkt kan verschuiven."""
    img = (mask > 0).astype(np.uint8)
    skel = np.zeros_like(img)
    k = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    for _ in range(40):
        if not img.any():
            break
        eroded = cv2.erode(img, k)
        skel |= img & ~cv2.dilate(eroded, k)
        img = eroded
    return skel * 255


def pitch_roi(H_work: np.ndarray, size: tuple[int, int], margin: float = 4.0) -> np.ndarray:
    """Masker van het (voorspelde) veld plus marge, in werkschaal."""
    W, Hh = size
    poly = np.array([[-margin, -margin], [pitch.LENGTH + margin, -margin], [pitch.LENGTH + margin, pitch.WIDTH + margin],
                     [-margin, pitch.WIDTH + margin]], float)
    # dicht bemonsteren zodat het stuk achter de camera netjes wordt afgekapt
    edge = np.vstack([a + np.linspace(0, 1, 60)[:, None] * (b - a) for a, b in zip(poly, np.roll(poly, -1, 0))])
    hom = np.hstack([edge, np.ones((len(edge), 1))]) @ H_work.T
    ok = hom[:, 2] > 1e-6
    img = hom[ok, :2] / hom[ok, 2:3]
    m = np.zeros((Hh, W), np.uint8)
    if len(img) >= 3:
        img = np.clip(img, -4 * W, 5 * W).astype(np.int32)
        cv2.fillPoly(m, [cv2.convexHull(img)], 255)
    return m


def _rot(rx: float, ry: float, rz: float) -> np.ndarray:
    cx, sx, cy, sy, cz, sz = math.cos(rx), math.sin(rx), math.cos(ry), math.sin(ry), math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def correction(params: np.ndarray, f: float, size: tuple[int, int]) -> np.ndarray:
    """Beeld -> beeld: kleine cameradraaiing (3 hoeken) + zoom, rond het beeldmidden."""
    rx, ry, rz, ls = params
    W, H = size
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
    K2 = np.array([[f * math.exp(ls), 0, W / 2], [0, f * math.exp(ls), H / 2], [0, 0, 1.0]])
    return K2 @ _rot(rx, ry, rz) @ np.linalg.inv(K)


def _nelder_mead(fun, x0: np.ndarray, step: np.ndarray, iters: int = 160) -> np.ndarray:
    n = len(x0)
    pts = [x0] + [x0 + np.eye(n)[i] * step[i] for i in range(n)]
    vals = [fun(p) for p in pts]
    for _ in range(iters):
        order = np.argsort(vals)
        pts, vals = [pts[i] for i in order], [vals[i] for i in order]
        c = np.mean(pts[:-1], axis=0)
        xr = c + (c - pts[-1])
        fr = fun(xr)
        if fr < vals[0]:
            xe = c + 2 * (c - pts[-1])
            fe = fun(xe)
            pts[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = xr, fr
        else:
            xc = c + 0.5 * (pts[-1] - c)
            fc = fun(xc)
            if fc < vals[-1]:
                pts[-1], vals[-1] = xc, fc
            else:
                pts = [pts[0] + 0.5 * (p - pts[0]) for p in pts]
                vals = [fun(p) for p in pts]
        if np.max(np.abs(np.array(vals) - vals[0])) < 1e-4:
            break
    return pts[int(np.argmin(vals))]


def _line_width_px(H: np.ndarray, pts: np.ndarray, width_m: float = 0.12) -> np.ndarray:
    """Hoe breed een veldlijn op deze plek in beeld is (pixels, kleinste richting)."""
    def proj(p):
        hom = np.hstack([p, np.ones((len(p), 1))]) @ H.T
        return hom[:, :2] / np.where(np.abs(hom[:, 2:3]) > 1e-9, hom[:, 2:3], 1e-9)
    base = proj(pts)
    dx = np.linalg.norm(proj(pts + [width_m, 0]) - base, axis=1)
    dy = np.linalg.norm(proj(pts + [0, width_m]) - base, axis=1)
    return np.minimum(dx, dy)


def camera_from_h(H: np.ndarray, cam: dict, size: tuple[int, int], p0: np.ndarray) -> np.ndarray:
    """Kijkrichting [yaw, tilt, roll, log_f] die bij homografie H past, met vaste camerapositie."""
    from .calibration import _lm, camera_homography

    xs, ys = np.meshgrid(np.linspace(-5, pitch.LENGTH + 5, 23), np.linspace(-5, pitch.WIDTH + 5, 15))
    world = np.stack([xs.ravel(), ys.ravel()], 1)
    hom = np.hstack([world, np.ones((len(world), 1))]) @ H.T
    ok = hom[:, 2] > 1e-6
    img = hom[:, :2] / np.where(ok, hom[:, 2], 1)[:, None]
    W, Hh = size
    ok &= (img[:, 0] > -0.2 * W) & (img[:, 0] < 1.2 * W) & (img[:, 1] > -0.2 * Hh) & (img[:, 1] < 1.2 * Hh)
    world, img = world[ok], img[ok]
    if len(world) < 6:
        return p0

    def res(q):
        Hq = camera_homography(np.array([cam["x"], cam["y"], cam["h"], *q]), size)
        return ((apply_h(Hq, world) - img) / 10.0).ravel()

    return _lm(res, np.asarray(p0, float), iters=30)


def refine(frame: np.ndarray, H_pred: np.ndarray, boxes: np.ndarray | None, f_full: float,
           debug: dict | None = None, camera: dict | None = None) -> tuple[np.ndarray, dict] | None:
    """Stel H (veld -> beeld, volle resolutie) bij op de lijnen in dit beeld. None = niet betrouwbaar.

    Zonder `camera`: een kleine extra draaiing + zoom bovenop de voorspelling (relatief).
    Met `camera` ({x, y, h, roll, log_f, q0}): de camera staat stil op een bekende plek en zoomt
    niet, dus we zoeken de absolute kijkrichting; zoom en scheefstand worden naar de waarden van
    het handmatige sleutelframe getrokken. Dan is ook één zichtbare lijn genoeg en kan er niets
    ongemerkt 'langs de lijn' wegglijden."""
    from .calibration import camera_homography

    full = (frame.shape[1], frame.shape[0])
    s = WORK_WIDTH / frame.shape[1]
    S = np.diag([s, s, 1.0])
    Hw = S @ H_pred
    size = (WORK_WIDTH, int(round(frame.shape[0] * s)))
    Wd, Hd = size
    lines = detect_lines(frame, boxes, roi=pitch_roi(Hw, size))
    if lines.sum() / 255 < 150:
        return None
    lines = centerlines(lines)
    dt = cv2.distanceTransform(255 - lines, cv2.DIST_L2, 3)
    det_y, det_x = np.nonzero(lines)
    # veldmodel-punten die nu in beeld liggen, voor de camera, en breed genoeg om te zien
    hom = np.hstack([SAMPLES, np.ones((len(SAMPLES), 1))]) @ Hw.T
    front = hom[:, 2] > 1e-6
    proj = hom[:, :2] / np.where(front, hom[:, 2], 1)[:, None]
    vis = front & (proj[:, 0] > 2) & (proj[:, 0] < Wd - 3) & (proj[:, 1] > 2) & (proj[:, 1] < Hd - 3)
    vis &= _line_width_px(Hw, SAMPLES) >= 1.2
    if vis.sum() < 60:
        return None
    world, world_tip = SAMPLES[vis], SAMPLES[vis] + 0.5 * TANGENTS[vis]
    f = f_full * s
    deg = math.radians(1.0)

    if camera is None:
        def H_of(p):
            return correction(p, f, size) @ Hw
        x0 = np.zeros(4)
        sig = np.array([math.radians(1.5)] * 3 + [0.02])
        center = x0
        steps = [np.array([deg, deg, deg, 0.03]), np.array([deg / 3] * 3 + [0.01]), np.array([deg / 10] * 3 + [0.003])]
    else:
        fixed = [camera["x"], camera["y"], camera["h"]]

        def H_of(p):
            return S @ camera_homography(np.array([*fixed, *p]), full)
        x0 = camera_from_h(H_pred, camera, full, camera["q0"])
        # trek scheefstand en zoom naar die van het handmatige sleutelframe, richting naar de voorspelling
        center = np.array([x0[0], x0[1], camera["roll"], camera["log_f"]])
        sig = np.array([math.radians(1.5), math.radians(1.5), math.radians(1.0), 0.02])
        steps = [np.array([deg, deg, deg, 0.03]), np.array([deg / 3] * 3 + [0.01]), np.array([deg / 10] * 3 + [0.003])]

    def project(Hm, pts):
        hh = np.hstack([pts, np.ones((len(pts), 1))]) @ Hm.T
        return hh[:, :2] / np.where(np.abs(hh[:, 2:3]) > 1e-9, hh[:, 2:3], 1e-9), hh[:, 2] > 1e-6

    def dists(p):
        q, fr = project(H_of(p), world)
        inside = fr & (q[:, 0] >= 0) & (q[:, 0] < Wd - 1) & (q[:, 1] >= 0) & (q[:, 1] < Hd - 1)
        d = np.full(len(q), 99.0, np.float32)
        qi = q[inside].astype(np.float32)
        if len(qi):
            d[inside] = cv2.remap(dt, qi[:, 0:1], qi[:, 1:2], cv2.INTER_LINEAR).ravel()
        return d, q

    def score(p, tau):
        d, _ = dists(p)
        return float(np.mean(np.minimum(d, tau))) / tau + 0.02 * float(np.sum(((p - center) / sig) ** 2))

    def precision(p, px=3.0):
        """Deel van de gevonden lijnpixels dat op een modellijn valt (gezien vanaf het beeld)."""
        q, fr = project(H_of(p), DENSE)
        q = q[fr]
        q = q[(q[:, 0] >= 0) & (q[:, 0] < Wd) & (q[:, 1] >= 0) & (q[:, 1] < Hd)].astype(np.int32)
        model = np.zeros((Hd, Wd), np.uint8)
        model[q[:, 1], q[:, 0]] = 255
        model = cv2.dilate(model, np.ones((3, 3), np.uint8))
        dm = cv2.distanceTransform(255 - model, cv2.DIST_L2, 3)
        return float((dm[det_y, det_x] < px).mean()) if len(det_x) else 0.0

    x = x0.copy()
    before = float((dists(x0)[0] < 2.5).mean())
    for tau, st in zip((25.0, 8.0, 3.0), steps):
        x = _nelder_mead(lambda p: score(p, tau), x, st)
    d, q = dists(x)
    hit = d < 2.5
    after = float(hit.mean())
    prec_before, prec = precision(x0), precision(x)
    # lijnrichtingen van de raakpunten (alleen relevant zonder bekende camerapositie)
    spread = 0.0
    if hit.sum() >= 10:
        qt, _ = project(H_of(x), world_tip)
        tv = qt[hit] - q[hit]
        tv /= np.linalg.norm(tv, axis=1, keepdims=True) + 1e-9
        nrm = np.stack([-tv[:, 1], tv[:, 0]], 1)
        ev = np.linalg.eigvalsh(nrm.T @ nrm / len(nrm))
        spread = float(ev[0] / max(ev[1], 1e-9))
    # Alleen evenwijdige lijnen in beeld (bijv. alleen de zijlijn): dan is niet te zien hoeveel de
    # camera langs die lijn gedraaid is. Overslaan; de camerabeweging overbrugt tot er weer
    # dwarslijnen (16-meter, middenlijn, doellijn, cirkel) in beeld zijn.
    one_dir = spread < MIN_SPREAD
    if camera is None:
        rot = float(np.linalg.norm(x[:3]))
        zoom_dev = abs(x[3])
        max_rot, max_zoom = math.radians(MAX_ROT_STEP_DEG), MAX_ZOOM_STEP
        ok_extra = True
    else:
        rot = float(np.hypot(x[0] - x0[0], x[1] - x0[1]))
        zoom_dev = abs(x[3] - x0[3])
        max_rot, max_zoom = math.radians(MAX_ROT_STEP_DEG), MAX_ZOOM_STEP
        # zoom en scheefstand mogen niet ver van het handmatige sleutelframe liggen
        ok_extra = abs(x[3] - camera["log_f"]) < 0.06 and abs(x[2] - camera["roll"]) < math.radians(3)
    info = {"before": round(before, 3), "after": round(after, 3), "n": int(len(world)),
            "precision": round(prec, 3), "precision_before": round(prec_before, 3),
            "rot_deg": round(math.degrees(rot), 2), "zoom": round(math.exp(zoom_dev), 3),
            "spread": round(spread, 3), "one_direction": bool(one_dir), "camera_mode": camera is not None}
    if debug is not None:
        debug.update(info)
    # Acceptatie: genoeg raak, gevonden lijnen vallen op het model, en alleen een kleine correctie
    # (één beeld kan dubbelzinnig zijn; omdat we elke seconde bijstellen is de echte correctie klein).
    if (one_dir or after < 0.30 or after * len(world) < 50 or prec < PRECISION_MIN or prec < prec_before - 0.02
            or zoom_dev > max_zoom or rot > max_rot or not ok_extra):
        return None
    H_new = np.linalg.inv(S) @ H_of(x)
    info["params"] = [round(float(v), 6) for v in x]
    return normalize_h(H_new), info


def keyframe_points(H: np.ndarray, size: tuple[int, int]) -> list[dict]:
    """Een raster van veldpunten dat in beeld ligt, als sleutelframe-punten (beeld <-> veld)."""
    xs, ys = np.meshgrid(np.linspace(0, pitch.LENGTH, 15), np.linspace(0, pitch.WIDTH, 9))
    world = np.stack([xs.ravel(), ys.ravel()], 1)
    hom = np.hstack([world, np.ones((len(world), 1))]) @ H.T
    ok = hom[:, 2] > 1e-6
    img = hom[:, :2] / np.where(ok, hom[:, 2], 1)[:, None]
    W, Hh = size
    ok &= (img[:, 0] >= 0) & (img[:, 0] <= W) & (img[:, 1] >= 0) & (img[:, 1] <= Hh)
    sel = np.flatnonzero(ok)
    if len(sel) < 4:
        return []
    if len(sel) > 12:
        sel = sel[np.linspace(0, len(sel) - 1, 12).astype(int)]
    return [{"name": "auto", "img": [round(float(img[i, 0]), 2), round(float(img[i, 1]), 2)],
             "pitch": [round(float(world[i, 0]), 3), round(float(world[i, 1]), 3)]} for i in sel]


class _FrameReader:
    """Leest frames op opgegeven tijden: vooruit achter elkaar (snel), terug via springen."""

    def __init__(self, path: Path):
        self.path = path
        self.cap = cv2.VideoCapture(str(path))
        self.pos = -1.0

    def at(self, t: float) -> np.ndarray | None:
        if t < self.pos or t - self.pos > 5.0:  # terug of ver vooruit: springen
            self.cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t - 1.0) * 1000)
        while True:
            if not self.cap.grab():
                return None
            self.pos = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if self.pos >= t - 0.02:
                ok, frame = self.cap.retrieve()
                return frame if ok else None

    def close(self):
        self.cap.release()


def _focal(clip: dict, manual: list[dict]) -> float:
    """Brandpuntsafstand (px) voor de correctie: uit het cameramodel als dat er is."""
    from .calibration import default_focal

    prior = camera_prior(clip)
    if prior is not None:
        for kf in manual:
            try:
                _, cam = fit_camera(kf["points"], prior)
                return (clip["width"] / 2) / math.tan(math.radians(cam["hfov_deg"]) / 2)
            except ValueError:
                continue
    return default_focal(int(clip["width"]))


def run_autocalib(store, clip_id: int, every_s: float = 1.0, progress=None) -> dict:
    """Leg elke `every_s` seconden een automatisch sleutelframe vast (waar de lijnen dat toelaten)."""
    from .storage import clip_dir

    clip = store.one("SELECT * FROM clips WHERE id = ?", (clip_id,))
    frames = store.all("SELECT idx, t FROM frames WHERE clip_id = ? ORDER BY idx", (clip_id,))
    if not frames:
        raise ValueError("Analyseer de video eerst")
    t = np.array([f["t"] for f in frames])
    inter = np.load(clip_dir(clip_id) / "motion.npy")
    A = cumulative(inter[:len(t)])
    store.run("DELETE FROM keyframes WHERE clip_id = ? AND auto = 1", (clip_id,))
    manual = [kf for kf in store.keyframes(clip_id) if not kf.get("auto")]
    prior = camera_prior(clip)
    accepted: dict[int, np.ndarray] = {}  # frame-index -> K (beeld -> veld)
    for kf in manual:
        try:
            K, _ = fit_calibration(kf["points"], camera=prior)
        except ValueError:
            continue
        accepted[int(np.argmin(np.abs(t - kf["t"])))] = K
    if not accepted:
        raise ValueError("Kalibreer eerst één sleutelframe met de hand")
    f_full = _focal(clip, manual)
    # Bekende camerapositie (aangeklikt of via GPS)? Dan bijstellen met het cameramodel: vaste plek,
    # zoom en scheefstand vastgehouden aan het handmatige sleutelframe.
    cam_mode = None
    params_of: dict[int, np.ndarray] = {}
    if prior is not None:
        for kf in manual:
            try:
                _, cam = fit_camera(kf["points"], prior)
            except ValueError:
                continue
            p = np.array(cam["params"])
            cam_mode = {"x": p[0], "y": p[1], "h": p[2], "roll": p[5], "log_f": p[6]}
            params_of[int(np.argmin(np.abs(t - kf["t"])))] = p[3:]
            break
    size = (int(clip["width"]), int(clip["height"]))
    fps = (len(t) - 1) / max(1e-6, t[-1] - t[0])
    step = max(1, int(round(every_s * fps)))
    first = min(accepted)
    targets_fwd = [i for i in range(first + step, len(t), step)]
    targets_bwd = [i for i in range(first - step, -1, -step)]
    total = len(targets_fwd) + len(targets_bwd)
    boxes_of: dict[int, list] = {}
    for r in store.all("SELECT idx, x1, y1, x2, y2 FROM detections WHERE clip_id = ?", (clip_id,)):
        boxes_of.setdefault(r["idx"], []).append([r["x1"], r["y1"], r["x2"], r["y2"]])

    reader = _FrameReader(Path(clip["path"]))
    done = ok_count = 0

    def _try(i: int, k: int) -> bool:
        G = accepted[k] @ np.linalg.inv(A[k]) @ A[i]  # beeld(i) -> veld, via de camerabeweging
        H_pred = np.linalg.inv(G)
        frame = reader.at(float(t[i]))
        if frame is None:
            return False
        camera = None
        if cam_mode is not None:
            near = min(params_of, key=lambda q: abs(q - i)) if params_of else None
            q0 = params_of[near] if near is not None else np.array([0.0, 0.2, cam_mode["roll"], cam_mode["log_f"]])
            camera = {**cam_mode, "q0": q0}
        res = refine(frame, normalize_h(H_pred), np.array(boxes_of.get(i, [])).reshape(-1, 4), f_full,
                     camera=camera)
        if res is None:
            return False
        H_new, info = res
        pts = keyframe_points(H_new, size)
        if len(pts) < 4:
            return False
        accepted[i] = np.linalg.inv(H_new)
        if camera is not None and "params" in info:
            params_of[i] = np.array(info["params"])
        store.run("INSERT INTO keyframes (clip_id, t, points, auto, score) VALUES (?,?,?,1,?)",
                  (clip_id, float(t[i]), json.dumps(pts), json.dumps(info)))
        return True
    try:
        for direction, targets in ((1, targets_fwd), (-1, targets_bwd)):
            for i in targets:
                done += 1
                if any(abs(i - k) < step / 2 for k in accepted):
                    continue
                # dichtstbijzijnde geaccepteerde sleutelframe aan de kant waar we vandaan komen
                ks = [k for k in accepted if (k <= i if direction > 0 else k >= i)] or list(accepted)
                k = min(ks, key=lambda q: abs(q - i))
                # lukt het niet, probeer dan een tussenmoment dichter bij het laatste goede sleutelframe
                # (kleinere afwijking); daarna verder vanaf daar
                tries = [i]
                if abs(i - k) > 2:
                    tries.append(k + (i - k) // 2)
                for j in tries:
                    if any(abs(j - q) < 2 for q in accepted):
                        continue
                    kk = min([q for q in accepted if (q <= j if direction > 0 else q >= j)] or list(accepted),
                             key=lambda q: abs(q - j))
                    if _try(j, kk):
                        ok_count += 1
                        if j == i:
                            break
                if progress and done % 5 == 0:
                    progress(done / max(1, total), ok_count)
    finally:
        reader.close()
    return {"tried": total, "accepted": ok_count}
