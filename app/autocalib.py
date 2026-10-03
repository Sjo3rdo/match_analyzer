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
    # 3. op gras: een veldlijn is geschilderd op het gras, dus rondom moet vooral gras liggen.
    #    Zo vallen een tegelpad, een wit hek, reclameborden en lucht tussen bomen af.
    #    (We kijken naar de pixels rond de lijn die zelf geen lijn zijn, zodat ook een brede lijn
    #    vlak voor de camera blijft staan.)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    green = (hsv[..., 0] >= 22) & (hsv[..., 0] <= 85) & (hsv[..., 1] >= 50) & (hsv[..., 2] >= 40)
    # bladeren zijn ook groen, maar veel donkerder dan het speelveld: gras = groen en ongeveer zo
    # licht als het meeste groen in de onderste helft van het beeld (daar ligt bijna altijd veld)
    lower = green[green.shape[0] // 2:]
    v_field = float(np.median(hsv[green.shape[0] // 2:][..., 2][lower])) if lower.any() else 0.0
    grass = (green & (hsv[..., 2] >= 0.55 * v_field)).astype(np.float32)
    not_line = (~mask).astype(np.float32)
    k = (25, 25)
    frac = cv2.blur(grass * not_line, k) / np.maximum(cv2.blur(not_line, k), 1e-3)
    mask &= frac >= 0.45
    out = mask.astype(np.uint8) * 255
    if roi is not None:
        out &= roi
    if boxes is not None:
        for x1, y1, x2, y2 in (np.asarray(boxes).reshape(-1, 4) * s).astype(int):
            pad = int(0.1 * (y2 - y1)) + 2
            out[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad] = 0
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    # 4. alleen dunne, lange stukken (lijnen), geen vlekjes (shirts, schoenen, bal, glinstering).
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
    # goedgekeurde automatische sleutelframes blijven staan en tellen mee als ankers
    store.run("DELETE FROM keyframes WHERE clip_id = ? AND auto = 1 AND COALESCE(accepted, 0) = 0", (clip_id,))
    manual = [kf for kf in store.keyframes(clip_id) if not kf.get("auto")]
    kept = [kf for kf in store.keyframes(clip_id) if kf.get("auto")]
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
    for kf in kept:
        try:
            accepted.setdefault(int(np.argmin(np.abs(t - kf["t"]))), fit_calibration(kf["points"])[0])
        except ValueError:
            continue
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


# --- automatisch voorstel: het veld zoeken zonder klikken ---------------------------------
#
# Metafoor: je staat op een bekende plek en draait langzaam rond met een plattegrond in je hand.
# Bij elke kijkrichting leg je de plattegrond over het beeld en tel je hoeveel getekende lijnen
# op echte witte lijnen vallen. De richting waarbij het best klopt is het voorstel. Daarna
# schuiven we nog een beetje met plek, zoom en scheefstand tot het precies past.

SAMPLES_COARSE = pitch_samples(step=1.0)


def _view_gain(H_work: np.ndarray, dt: np.ndarray, size: tuple[int, int], tau: float,
               samples: np.ndarray = SAMPLES) -> tuple[float, int, float]:
    """Hoe goed de zichtbare modellijnen op gevonden lijnen vallen: (winst, aantal zichtbaar, deel raak).

    Elk zichtbaar modelpunt levert +1 op als het precies op een lijn valt en -0,5 als er niets in
    de buurt is. Een kijkrichting waarin veel lijnen kloppen wint dus van een waarin toevallig een
    stukje lijn past. Lijnen die in beeld te dun zijn om te zien (ver weg) tellen niet mee."""
    Wd, Hd = size
    hom = np.hstack([samples, np.ones((len(samples), 1))]) @ H_work.T
    front = hom[:, 2] > 1e-6
    q = hom[:, :2] / np.where(front, hom[:, 2], 1)[:, None]
    vis = front & (q[:, 0] >= 0) & (q[:, 0] < Wd - 1) & (q[:, 1] >= 0) & (q[:, 1] < Hd - 1)
    if vis.any():
        vis[vis] = _line_width_px(H_work, samples[vis]) >= 1.5
    n = int(vis.sum())
    if n < 10:
        return -1e3, n, 0.0
    qi = q[vis].astype(np.float32)
    d = cv2.remap(dt, qi[:, 0:1], qi[:, 1:2], cv2.INTER_LINEAR).ravel()
    gain = float(np.sum(1.0 - 1.3 * np.minimum(d, tau) / tau))
    return gain, n, float((d < 2.5).mean())


def _precision(H_work: np.ndarray, det_xy: np.ndarray, size: tuple[int, int], px: float = 3.0) -> float:
    """Deel van de gevonden lijnpixels binnen het veld dat op een modellijn valt."""
    Wd, Hd = size
    if not len(det_xy):
        return 0.0
    roi = pitch_roi(H_work, size, margin=3.0)
    inside = roi[det_xy[:, 1], det_xy[:, 0]] > 0
    if inside.sum() < 30:
        return 0.0
    hom = np.hstack([DENSE, np.ones((len(DENSE), 1))]) @ H_work.T
    fr = hom[:, 2] > 1e-6
    q = hom[fr, :2] / hom[fr, 2:3]
    q = q[(q[:, 0] >= 0) & (q[:, 0] < Wd) & (q[:, 1] >= 0) & (q[:, 1] < Hd)].astype(np.int32)
    model = np.zeros((Hd, Wd), np.uint8)
    model[q[:, 1], q[:, 0]] = 255
    dm = cv2.distanceTransform(255 - cv2.dilate(model, np.ones((3, 3), np.uint8)), cv2.DIST_L2, 3)
    pts = det_xy[inside]
    return float((dm[pts[:, 1], pts[:, 0]] < px).mean())


def _batch_homographies(P: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """camera_homography voor veel parametersets tegelijk: (M, 7) -> (M, 3, 3)."""
    x, y, h, yaw, tilt, roll, logf = P.T
    W, Hh = size
    f = np.exp(logf)
    fwd = np.stack([np.cos(yaw) * np.cos(tilt), np.sin(yaw) * np.cos(tilt), np.sin(tilt)], 1)
    right = np.stack([-fwd[:, 1], fwd[:, 0], np.zeros(len(P))], 1)  # cross((0,0,1), fwd)
    right /= np.linalg.norm(right, axis=1, keepdims=True) + 1e-12
    dn = np.cross(fwd, right)
    cr, sr = np.cos(roll)[:, None], np.sin(roll)[:, None]
    right, dn = cr * right + sr * dn, -sr * right + cr * dn
    R = np.stack([right, dn, fwd], 1)  # (M, 3, 3) rijen
    C = np.stack([x, y, -h], 1)
    t = -np.einsum("mij,mj->mi", R, C)
    E = np.stack([R[:, :, 0], R[:, :, 1], t], 2)
    K = np.zeros((len(P), 3, 3))
    K[:, 0, 0] = f
    K[:, 1, 1] = f
    K[:, 0, 2] = W / 2
    K[:, 1, 2] = Hh / 2
    K[:, 2, 2] = 1
    Hs = K @ E
    return Hs / np.abs(Hs[:, 2:3, 2:3])


SAMPLES_C2 = pitch_samples(step=2.0)


def _coarse_scores(P: np.ndarray, full: tuple[int, int], s: float, dt: np.ndarray, size: tuple[int, int],
                   det: np.ndarray, tau_px: float = 25.0, tau_m: float = 1.5, keep: float = 0.15) -> np.ndarray:
    """Grove score voor veel camerastanden tegelijk: (modellijnen op gevonden lijnen) x (deel van
    de gevonden lijnen dat het model verklaart). Het tweede deel alleen voor de beste `keep`."""
    Wd, Hd = size
    Hs = (_batch_homographies(P, full) * np.array([s, s, 1.0])[None, :, None])
    X = np.vstack([SAMPLES_C2.T, np.ones(len(SAMPLES_C2))]).astype(np.float32)
    Hf = Hs.astype(np.float32)
    hh = Hf @ X  # (M, 3, N)
    z = hh[:, 2]
    front = z > 1e-6
    zs = np.where(front, z, 1.0)
    u, v = hh[:, 0] / zs, hh[:, 1] / zs
    vis = front & (u >= 0) & (u < Wd - 1) & (v >= 0) & (v < Hd - 1)
    # te dunne lijnen (ver weg) tellen niet mee: breedte van 12 cm in beeld, kleinste richting
    wmin = None
    for col in (0, 1):
        dh = Hf[:, :, col:col + 1] * 0.12  # verschuiving van 12 cm langs x of y
        z2 = z + dh[:, 2]
        z2 = np.where(np.abs(z2) > 1e-6, z2, 1e-6)
        wd = np.hypot((hh[:, 0] + dh[:, 0]) / z2 - u, (hh[:, 1] + dh[:, 1]) / z2 - v)
        wmin = wd if wmin is None else np.minimum(wmin, wd)
    vis &= wmin >= 1.5
    ui = np.where(vis, u, 0).astype(np.int32)
    vi = np.where(vis, v, 0).astype(np.int32)
    d = np.minimum(dt[vi, ui], tau_px)
    gain = np.where(vis, 1.0 - 1.3 * d / tau_px, 0.0).sum(1)
    gain[vis.sum(1) < 8] = 0.0
    out = np.zeros(len(P))
    top = np.flatnonzero(gain > 0)
    if len(top) > keep * len(P):
        top = top[np.argsort(-gain[top])[:max(1, int(keep * len(P)))]]
    if len(top):
        out[top] = gain[top] * np.maximum(_batch_explained(Hs[top], det, tau_m), 0.0)
    return out


_FIELD_RES, _FIELD_MARGIN = 0.1, 12.0
_field_dt: np.ndarray | None = None


def _field_distance() -> np.ndarray:
    """Bovenaanzicht: per 10 cm de afstand (m) tot de dichtstbijzijnde veldlijn."""
    global _field_dt
    if _field_dt is None:
        w = int((pitch.LENGTH + 2 * _FIELD_MARGIN) / _FIELD_RES)
        h = int((pitch.WIDTH + 2 * _FIELD_MARGIN) / _FIELD_RES)
        img = np.full((h, w), 255, np.uint8)
        q = ((DENSE + _FIELD_MARGIN) / _FIELD_RES).astype(int)
        img[np.clip(q[:, 1], 0, h - 1), np.clip(q[:, 0], 0, w - 1)] = 0
        _field_dt = cv2.distanceTransform(img, cv2.DIST_L2, 3) * _FIELD_RES
    return _field_dt


def _batch_explained(Hs_work: np.ndarray, det: np.ndarray, tau_m: float) -> np.ndarray:
    """Hoeveel van de gevonden lijnpixels door het veldmodel verklaard worden, per camerastand.

    Elk gevonden pixel wordt teruggeprojecteerd op het veld; ligt het op een veldlijn dan +1,
    ver ernaast -0,3. Zo verliest een stand die maar een paar lijnen 'gebruikt'."""
    fd = _field_distance()
    fh, fw = fd.shape
    Ginv = np.linalg.inv(Hs_work)  # beeld -> veld
    X = np.vstack([det.T, np.ones(len(det))])
    hh = Ginv @ X  # (M, 3, K)
    z = hh[:, 2]
    # H (X/z) = x/z, dus het veldpunt ligt vóór de camera (w > 0) precies als z > 0
    fwd = z > 1e-12
    zs = np.where(fwd, z, 1.0)
    fx = (hh[:, 0] / zs + _FIELD_MARGIN) / _FIELD_RES
    fy = (hh[:, 1] / zs + _FIELD_MARGIN) / _FIELD_RES
    on = fwd & (fx >= 0) & (fx < fw) & (fy >= 0) & (fy < fh)
    d = fd[np.where(on, fy, 0).astype(np.int32), np.where(on, fx, 0).astype(np.int32)]
    v = np.where(on, 1.0 - 1.3 * np.minimum(d, tau_m) / tau_m, -0.3)
    return v.mean(1)


def propose(frame: np.ndarray, prior: dict, boxes: np.ndarray | None = None,
            debug: dict | None = None) -> tuple[np.ndarray, dict] | None:
    """Zoek de kalibratie (veld -> beeld, volle resolutie) bij een bekende camerapositie.

    prior: camera_prior(clip). None als er geen overtuigende plek gevonden wordt."""
    from .calibration import camera_homography

    full = (frame.shape[1], frame.shape[0])
    s = WORK_WIDTH / frame.shape[1]
    S = np.diag([s, s, 1.0])
    size = (WORK_WIDTH, int(round(frame.shape[0] * s)))
    lines = detect_lines(frame, boxes)
    if lines.sum() / 255 < 150:
        return None
    lines = centerlines(lines)
    dt = cv2.distanceTransform(255 - lines, cv2.DIST_L2, 3)
    det_y, det_x = np.nonzero(lines)
    det_xy = np.stack([det_x, det_y], 1)
    cx, cy, ch = prior["x"], prior["y"], prior["h"]
    sp = max(3.0, float(prior.get("sigma_pos", 5.0)))

    def H_work(p):  # p = [x, y, h, yaw, tilt, roll, log_f]
        return S @ camera_homography(p, full)

    rng = np.random.default_rng(0)
    det_sub = det_xy[rng.choice(len(det_xy), min(400, len(det_xy)), replace=False)].astype(float)
    det_c = det_sub[:200]

    # 1. grof: vanaf elke plek binnen de onzekerheid alle kijkrichtingen naar het veld en een paar
    #    zoomstanden. De plek doet er veel toe: 2 m naast de echte plek klopt de zijlijn vlak voor je al niet meer.
    log_fs = [math.log(prior["f"]) + d for d in (-0.25, 0.0, 0.3)]  # iets uitgezoomd .. ingezoomd
    tilts = np.radians([1.5, 3, 5, 7.5, 10, 13, 17, 22, 28, 35])
    pstep = max(1.0, sp / 4)
    offs = np.arange(-1.5 * sp, 1.5 * sp + 1e-6, pstep)
    corners = np.array([[0, 0], [pitch.LENGTH, 0], [pitch.LENGTH, pitch.WIDTH], [0, pitch.WIDTH], [pitch.LENGTH / 2, pitch.WIDTH / 2]])
    pool = []
    for dx in offs:
        for dy in offs:
            px, py = cx + dx, cy + dy
            ang = np.degrees(np.arctan2(corners[:, 1] - py, corners[:, 0] - px))
            if 0 <= px <= pitch.LENGTH and 0 <= py <= pitch.WIDTH:
                yaws = np.arange(0, 360, 2.0)
            else:  # kijkrichtingen tussen de uiterste hoekpunten (plus wat marge)
                rel = (ang - ang[4] + 180) % 360 - 180
                yaws = ang[4] + np.arange(rel.min() - 25, rel.max() + 25.1, 2.0)
            Y, T, F = np.meshgrid(np.radians(yaws), tilts, log_fs, indexing="ij")
            P = np.stack([np.full(Y.size, px), np.full(Y.size, py), np.full(Y.size, ch), Y.ravel(), T.ravel(),
                          np.zeros(Y.size), F.ravel()], 1)
            sc = _coarse_scores(P, full, s, dt, size, det_c)
            for i in np.argsort(-sc)[:3]:
                if sc[i] > 0:
                    pool.append((float(sc[i]), P[i]))
    if not pool:
        return None
    pool.sort(key=lambda c: -c[0])
    cand = pool

    # 2. de beste kandidaten verfijnen: richting, zoom, scheefstand en (binnen de onzekerheid) de plek
    lf0 = math.log(prior["f"])
    lf_lo, lf_hi = math.log(full[0] / 2 / math.tan(math.radians(50))), math.log(full[0] / 2 / math.tan(math.radians(17.5)))

    def objective(q, tau):
        # harde grenzen: plek binnen de onzekerheid, camera kijkt omlaag, normale zoom, bijna recht
        if (abs(q[0]) > 2 * sp or abs(q[1]) > 2 * sp or not math.radians(0.5) < q[3] < math.radians(60)
                or abs(q[4]) > math.radians(8) or not lf_lo < q[5] < lf_hi):
            return 1e6
        p = np.array([cx + q[0], cy + q[1], ch, *q[2:]])
        Hw = H_work(p)
        g, n, _ = _view_gain(Hw, dt, size, tau)
        e = float(_batch_explained(Hw[None], det_sub, tau_m=tau / 16.0)[0])
        pri = (q[0] / sp) ** 2 + (q[1] / sp) ** 2 + (q[4] / math.radians(4)) ** 2 + ((q[5] - lf0) / 0.5) ** 2
        return -max(g, 0.0) / 100.0 * max(e, 0.0) + 0.02 * pri

    deg = math.radians(1.0)
    results = []
    seen = []
    for c0, p in cand[:60]:
        if any(abs((p[3] - o[3] + math.pi) % (2 * math.pi) - math.pi) < math.radians(5) and abs(p[4] - o[4]) < math.radians(4)
               and abs(p[6] - o[6]) < 0.1 and math.hypot(p[0] - o[0], p[1] - o[1]) < 2.5 for o in seen):
            continue  # vrijwel dezelfde kandidaat
        seen.append(p)
        if len(seen) > 8:
            break
        q = np.array([p[0] - cx, p[1] - cy, p[3], p[4], 0.0, p[6]])
        for tau, st in ((25.0, [2, 2, 2 * deg, deg, deg, 0.05]), (8.0, [1, 1, deg / 2, deg / 3, deg / 2, 0.02]),
                        (3.0, [0.4, 0.4, deg / 6, deg / 10, deg / 6, 0.006])):
            q = _nelder_mead(lambda v: objective(v, tau), q, np.array(st), iters=120)
        p_new = np.array([cx + q[0], cy + q[1], ch, *q[2:]])
        Hw = H_work(p_new)
        gain, n, hit = _view_gain(Hw, dt, size, 3.0)
        prec = _precision(Hw, det_xy, size)
        results.append({"p": p_new, "hit": hit, "precision": prec, "n": n, "gain": gain, "score": gain * prec ** 6})
    best = max(results, key=lambda r: r["score"])
    # Twijfel: een wezenlijk andere camerastand past bijna even goed (bijv. alleen de zijlijn in beeld)
    Hb = camera_homography(best["p"], full)
    xs, ys = np.meshgrid(np.linspace(0, pitch.LENGTH, 15), np.linspace(0, pitch.WIDTH, 9))
    grid_w = np.stack([xs.ravel(), ys.ravel()], 1)
    gb = apply_h(Hb, grid_w)
    in_view = np.isfinite(gb).all(1) & (gb[:, 0] >= 0) & (gb[:, 0] <= full[0]) & (gb[:, 1] >= 0) & (gb[:, 1] <= full[1])
    ambiguous = False
    for r in results:
        if r is best or r["score"] < 0.85 * best["score"] or not in_view.sum():
            continue
        diff = np.linalg.norm(apply_h(camera_homography(r["p"], full), grid_w[in_view]) - gb[in_view], axis=1)
        if np.median(diff) > 0.03 * full[0]:
            ambiguous = True
    info = {"hit": round(best["hit"], 3), "precision": round(best["precision"], 3), "n": best["n"], "ambiguous": ambiguous,
            "camera": {"x": round(float(best["p"][0]), 1), "y": round(float(best["p"][1]), 1),
                       "yaw_deg": round(math.degrees(best["p"][3]) % 360, 1), "tilt_deg": round(math.degrees(best["p"][4]), 1),
                       "roll_deg": round(math.degrees(best["p"][5]), 1),
                       "hfov_deg": round(math.degrees(2 * math.atan(full[0] / 2 / math.exp(best["p"][6]))), 1)}}
    if debug is not None:
        debug.update(info, candidates=[{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k != "p"}
                                       for r in results])
    if best["hit"] < 0.7 or best["precision"] < 0.8 or best["hit"] * best["n"] < 30:
        return None
    return camera_homography(best["p"], full), info
