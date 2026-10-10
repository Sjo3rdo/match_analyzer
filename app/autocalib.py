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
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from . import pitch
from .calibration import (CameraModel, Keyframe, apply_h, camera_prior, cumulative, fit_calibration, fit_camera, kf_camera,
                          normalize_h)

log = logging.getLogger(__name__)

WORK_WIDTH = 960
PRECISION_MIN = 0.85  # deel van de gevonden lijnpixels dat op het model moet vallen
MAX_ROT_STEP_DEG = 0.8  # grootste toegestane correctie per bijstelling
MAX_ZOOM_STEP = 0.02
MIN_SPREAD = 0.08  # lijnrichtingen: 0 = alles evenwijdig, 1 = alle kanten op


def pitch_samples(step: float = 0.5, with_tangents: bool = False, geom: pitch.Geometry = pitch.DEFAULT):
    """Punten op alle veldlijnen (meters)."""
    L, W, c = geom.length, geom.width, geom.half_w
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


class _Model:
    """Veldlijnen als punten, per veldmaat één keer uitgerekend."""

    def __init__(self, geom: pitch.Geometry):
        self.geom = geom
        self.samples, self.tangents = pitch_samples(with_tangents=True, geom=geom)
        self.dense = pitch_samples(step=0.1, geom=geom)  # om de modellijnen als plaatje te tekenen
        self.coarse = pitch_samples(step=1.0, geom=geom)
        self.c2, self.c2_tan = pitch_samples(step=2.0, with_tangents=True, geom=geom)
        self._field_dt = None


@lru_cache(maxsize=8)
def model(geom: pitch.Geometry = pitch.DEFAULT) -> _Model:
    return _Model(geom)


SAMPLES, TANGENTS = model().samples, model().tangents
DENSE = model().dense


# Extra lijnzoeker voor lastige beelden (versleten lijnen, fel zonlicht, korrelig beeld): zie
# ridge_lines. Trager, dus alleen als het gewone zoeken het veld niet vindt.
RIDGE_L, RIDGE_C, RIDGE_LOC = 12.0, 4.0, 8  # lichter dan ernaast, grijzer dan ernaast, lichter dan omgeving
_LINE_KERNELS: dict = {}


def _line_kernels(n_angles: int, length: int):
    """Per richting: (kernel voor het gemiddelde langs een kort lijnstukje, eenheidsnormaal)."""
    key = (n_angles, length)
    if key not in _LINE_KERNELS:
        r = length // 2 + 1
        out = []
        for k in range(n_angles):
            a = np.pi * k / n_angles
            d = np.array([np.cos(a), np.sin(a)])
            img = np.zeros((2 * r + 1, 2 * r + 1), np.float32)
            p, q = r + d * length / 2, r - d * length / 2
            cv2.line(img, tuple(np.round(p * 16).astype(int)), tuple(np.round(q * 16).astype(int)), 1.0, 1,
                     cv2.LINE_AA, shift=4)
            out.append((img / img.sum(), np.array([-d[1], d[0]])))
        _LINE_KERNELS[key] = out
    return _LINE_KERNELS[key]


def _shifted(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """out(x, y) = img(x + dx, y + dy), met tussenwaarden."""
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_REPLICATE)


def ridge_lines(lum: np.ndarray, chroma: np.ndarray, n_angles: int = 12, length: int = 15,
                side: int = 3, min_ridge: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per pixel: hoeveel lichter een kort lijnstukje door dat pixel is dan twee evenwijdige stukjes
    ernaast, in de richting waarin dat het meest is, en hoeveel grijzer (minder kleur) het daar is.
    Geeft (lichter, grijzer, richting) terug; 'grijzer' alleen waar lichter > min_ridge.

    Metafoor: in plaats van naar één steentje in een stoeprand te kijken, kijk je naar een stukje
    stoeprand van een paar stappen lang. Een versleten of korrelige lijn valt per pixel weg in de ruis,
    maar gemiddeld over een stukje lijn steekt hij er duidelijk bovenuit; losse lichte sprieten niet."""
    best = np.full(lum.shape, -1e9, np.float32)
    angle = np.zeros(lum.shape, np.uint8)
    kernels = _line_kernels(n_angles, length)
    for k, (kern, (nx, ny)) in enumerate(kernels):
        mid = cv2.filter2D(lum, -1, kern)  # gemiddelde langs het lijnstukje
        # hetzelfde stukje, `side` pixels naar weerskanten verschoven (evenwijdig ernaast);
        # lichter dan béide kanten: een schaduwrand (licht naast donker) telt dan niet
        ridge = np.minimum(mid - _shifted(mid, side * nx, side * ny), mid - _shifted(mid, -side * nx, -side * ny))
        better = ridge > best
        best[better] = ridge[better]
        angle[better] = k
    # Grijzer dan ernaast: alleen uitrekenen waar het op een lijn lijkt (een klein deel van het beeld)
    grey = np.zeros(lum.shape, np.float32)
    ys, xs = np.nonzero(best > min_ridge)
    if len(ys):
        h, w = lum.shape
        t = np.arange(length, dtype=np.float32) - (length - 1) / 2
        dirs = np.array([[np.cos(np.pi * k / n_angles), np.sin(np.pi * k / n_angles)] for k in range(n_angles)], np.float32)
        d = dirs[angle[ys, xs]]
        n = np.stack([-d[:, 1], d[:, 0]], 1)

        def mean_along(off):
            px = xs[:, None] + off * n[:, :1] + t[None] * d[:, :1]
            py = ys[:, None] + off * n[:, 1:] + t[None] * d[:, 1:]
            px = np.clip(np.rint(px), 0, w - 1).astype(np.intp)
            py = np.clip(np.rint(py), 0, h - 1).astype(np.intp)
            return chroma[py, px].mean(1)
        grey[ys, xs] = 0.5 * (mean_along(side) + mean_along(-side)) - mean_along(0)
    return best, grey, angle


def _long_straight(cand: np.ndarray, min_len: int = 40) -> np.ndarray:
    """Alleen de pixels die op een lang, recht stuk liggen (losse sprieten en kluitjes vallen af)."""
    segs = cv2.HoughLinesP(cand.astype(np.uint8), 1, np.pi / 180, threshold=25,
                           minLineLength=min_len, maxLineGap=6)
    on = np.zeros(cand.shape, np.uint8)
    if segs is not None:
        for x1, y1, x2, y2 in segs.reshape(-1, 4):
            cv2.line(on, (int(x1), int(y1)), (int(x2), int(y2)), 1, 3)
    return on > 0


def _bridge_gaps(mask: np.ndarray, angle: np.ndarray, allowed: np.ndarray, n_angles: int = 12,
                 length: int = 9) -> np.ndarray:
    """Kleine gaatjes in een lijn dichten, alleen in de richting van de lijn zelf (en alleen waar het
    beeld daar ook op een lijn lijkt), zodat een onderbroken lijn weer één lang stuk wordt."""
    out = mask.copy()
    r = length // 2
    for k in range(n_angles):
        sub = np.where((mask > 0) & (angle == k), 255, 0).astype(np.uint8)
        if not sub.any():
            continue
        a = np.pi * k / n_angles
        se = np.zeros((2 * r + 1, 2 * r + 1), np.uint8)
        d = np.array([np.cos(a), np.sin(a)]) * r
        cv2.line(se, tuple(np.round(r - d).astype(int)), tuple(np.round(r + d).astype(int)), 1, 1)
        out |= cv2.morphologyEx(sub, cv2.MORPH_CLOSE, se) & allowed
    return out


def detect_lines(frame: np.ndarray, boxes: np.ndarray | None = None, roi: np.ndarray | None = None,
                 enhance: bool = False) -> np.ndarray:
    """Masker (0/255, werkschaal) met witte veldlijnen.

    roi: optioneel masker (werkschaal) van waar het veld ongeveer ligt; daarbuiten negeren we
    alles (bomen, hekken, reclameborden, publiek).
    enhance: ook zwakke, versleten of korrelige lijnen zoeken (trager; voor lastige beelden)."""
    s = WORK_WIDTH / frame.shape[1]
    # Heeft de app zelf geleerd veldlijnen te zien (knop "Het veld herkennen" bij Trainen), dan
    # gebruiken we dat netwerk; de vorm-controles hieronder blijven gelden.
    from .learning import learned_lines

    learned = learned_lines(frame)
    if learned is not None:
        out = learned
        if roi is not None:
            out &= roi
        if boxes is not None:
            for x1, y1, x2, y2 in (np.asarray(boxes).reshape(-1, 4) * s).astype(int):
                pad = int(0.1 * (y2 - y1)) + 2
                out[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad] = 0
        return _line_components(out)[0]
    small = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.int16)
    L, A, B = lab[..., 0], lab[..., 1] - 128, lab[..., 2] - 128
    # 1. lichter dan de directe omgeving (smalle structuur: top-hat)
    th = cv2.morphologyEx(lab[..., 0].astype(np.uint8), cv2.MORPH_TOPHAT,
                          cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))).astype(np.int16)
    # 2. wit: weinig kleur (grassprieten in de zon zijn licht maar groen/geel)
    #    Een dunne of wat versleten lijn in de zon is niet helemaal wit maar licht geelgroen: dan telt
    #    hij ook, als hij duidelijk minder kleur heeft dan het gras eromheen (verf maakt het gras
    #    "grijzer"; een zonnig grassprietje wordt juist lichter in dezelfde groene kleur).
    chroma = np.sqrt(A.astype(np.float32) ** 2 + B.astype(np.float32) ** 2)
    local = cv2.blur(lab[..., 0].astype(np.uint8), (25, 25)).astype(np.int16)
    bg_a = cv2.blur(A.astype(np.float32), (25, 25))
    bg_b = cv2.blur(B.astype(np.float32), (25, 25))
    bg_c = np.sqrt(bg_a ** 2 + bg_b ** 2)
    to_grey = -((A - bg_a) * bg_a + (B - bg_b) * bg_b) / np.maximum(bg_c, 1.0)  # > 0: minder kleur
    white = (chroma < 22) | ((chroma < 40) & (to_grey >= 3.0) & (chroma <= 0.9 * bg_c))
    # Op korrelig kunstgras in de zon is de grond zelf vol lichte, kleurloze puntjes; dan moet een
    # lijn duidelijk feller zijn dan wat daar gewoon is. (Mediaan: de lijn zelf telt niet mee. Alleen
    # kleurloze puntjes: op echt gras zijn de lichte sprieten groen en houden we de lage drempel.)
    texture = cv2.medianBlur(np.clip(th * white, 0, 255).astype(np.uint8), 31).astype(np.float32)
    mask = (th > np.maximum(22, 2.2 * texture)) & white & (L > local + 15)
    if enhance:
        # Lastig beeld: ook zwakke lijnen zoeken door over een stukje lijn te middelen. Alleen wat op
        # een lang, recht stuk ligt telt mee, zodat losse lichte sprieten geen lijn worden.
        rl, rg, ang = ridge_lines(L.astype(np.float32), chroma, min_ridge=RIDGE_L)
        cand = (rl > RIDGE_L) & (rg > RIDGE_C) & (L > local + RIDGE_LOC) & ~mask
        mask |= cand & _long_straight(cand)
    # 3. op gras: een veldlijn is geschilderd op het gras, dus rondom moet vooral gras liggen.
    #    Zo vallen een tegelpad, een wit hek, reclameborden en lucht tussen bomen af.
    #    (We kijken naar de pixels rond de lijn die zelf geen lijn zijn, zodat ook een brede lijn
    #    vlak voor de camera blijft staan.)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    green = (hsv[..., 0] >= 22) & (hsv[..., 0] <= 85) & (hsv[..., 1] >= 30) & (hsv[..., 2] >= 40)  # ook dof kunstgras
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
    if enhance:
        out = _bridge_gaps(out, ang, np.where(rl > 0.5 * RIDGE_L, 255, 0).astype(np.uint8))
    # 4. alleen dunne, lange stukken (lijnen), geen vlekjes (shirts, schoenen, bal, glinstering).
    #    "Dun" = de dikste plek van het stuk is smal; "lang" = het stuk strekt zich ver uit.
    #    (Aan elkaar hangende lijnen vormen één netwerk; dat is dun én lang, dus blijft staan.)
    #    Een netwerk van een paar lijnen (hoek van het strafschopgebied) heeft een hartlijn van
    #    hooguit een paar keer de lengte van het stuk; de korrels van kunstgras of glinsterend gras
    #    vormen een fijnmazig netwerk dat een vlak vult, met een veel langere hartlijn.
    #    Hangt een echte (dikkere) lijn aan zo'n korrelveld vast, dan pellen we de dunne korrels
    #    eraf en kijken we nog eens.
    keep, rest = _line_components(out)
    if rest.any():
        peeled = cv2.morphologyEx(rest, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        keep2, _ = _line_components(peeled, max_ratio=1.6)
        keep = keep | keep2
    keep = _grass_both_sides(keep, lab)
    # Droge, lichte plekken in het gras hebben dezelfde kleur als een lijn in de zon, maar zijn vlekken.
    return _thin_if_tinted(keep, chroma < 22)


def _thin_if_tinted(mask: np.ndarray, pure_white: np.ndarray, max_thick: float = 2.0) -> np.ndarray:
    """Stukken die vooral uit getinte pixels bestaan (licht, maar niet echt wit) alleen houden als
    ze dun zijn (hooguit max_thick pixels) of een lange, rechte streep over een groot deel van het
    beeld vormen (de lijn vlak voor je in fel zonlicht)."""
    n, lab_cc, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n < 2:
        return mask
    area = np.bincount(lab_cc.ravel(), minlength=n)
    white = np.bincount(lab_cc.ravel(), weights=pure_white.ravel().astype(float), minlength=n)
    skel = np.bincount(lab_cc[centerlines(mask) > 0], minlength=n)
    extent = np.hypot(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT])
    long_straight = (extent >= 0.3 * mask.shape[1]) & (skel <= 1.6 * extent)  # rafelige rand geeft zijtakjes
    keep = (white >= 0.5 * area) | (area <= max_thick * np.maximum(skel, 1)) | long_straight
    keep[0] = False
    return np.where(keep[lab_cc], 255, 0).astype(np.uint8)


def _shift(a: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Waarde van (y - dy, x - dx) op plek (y, x); buiten beeld 0."""
    out = np.zeros_like(a)
    h, w = a.shape[:2]
    out[max(dy, 0):h + min(dy, 0), max(dx, 0):w + min(dx, 0)] = a[max(-dy, 0):h - max(dy, 0), max(-dx, 0):w - max(dx, 0)]
    return out


def _grass_both_sides(mask: np.ndarray, lab: np.ndarray, offset: int = 8) -> np.ndarray:
    """Alleen stukken met aan beide kanten hetzelfde gras.

    Een geschilderde lijn ligt midden in het gras: aan weerskanten (dwars op de lijn) is de kleur
    vrijwel gelijk. Hooguit is het gras erachter wat lichter, want verder weg. De rij reclameborden
    langs de overkant heeft onder zich gras en boven zich iets wat veel donkerder is (bomen, hek) of
    een andere tint heeft (lucht, de bovenrand van het bord), ook als dat toevallig ook groen is.
    We vergelijken de gemiddelde kleur (Lab) van de omgeving aan weerskanten, zonder de lijnpixels
    zelf; per stuk telt de mediaan."""
    n, lab_cc, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n < 2:
        return mask
    labf = lab.astype(np.float32)
    nl = (mask == 0).astype(np.float32)
    o = offset

    def side(kw, kh):
        den = np.maximum(cv2.blur(nl, (kw, kh)), 1e-3)
        return np.stack([cv2.blur(labf[..., i] * nl, (kw, kh)) / den for i in range(3)], -1)

    v, hz = side(25, 2 * o - 1), side(2 * o - 1, 25)
    d_v = _shift(v, o, 0) - _shift(v, -o, 0)  # boven min onder
    d_h = _shift(hz, 0, o) - _shift(hz, 0, -o)  # links min rechts
    # dwars op het stuk kijken: bij een liggend stuk boven/onder, bij een staand stuk links/rechts
    flat = stats[:, cv2.CC_STAT_WIDTH] >= stats[:, cv2.CC_STAT_HEIGHT]
    d = np.where(flat[lab_cc][..., None], d_v, d_h)
    on = lab_cc > 0
    labels, vals = lab_cc[on], d[on]
    order = np.argsort(labels, kind="stable")
    labels, vals = labels[order], vals[order]
    bounds = np.flatnonzero(np.diff(labels)) + 1
    keep = np.zeros(n, bool)
    for grp_l, grp_v in zip(np.split(labels, bounds), np.split(vals, bounds)):
        k = grp_l[0]
        dl, da, db = np.median(grp_v, axis=0)
        same_tint = math.hypot(da, db) < 12  # (OpenCV-schaal: 0..255)
        keep[k] = same_tint and (dl > -15 if flat[k] else abs(dl) < 30)
    return np.where(keep[lab_cc], 255, 0).astype(np.uint8)


def _line_components(mask: np.ndarray, max_ratio: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(stukken die op lijnen lijken, stukken die afvielen omdat ze een vlak vullen).

    Lengte van de hartlijn gedeeld door de omvang van het stuk: een recht stuk lijn ~1, een hoek of
    T-kruising ~1,5, de middencirkel of een heel strafschopgebied ~2,5; kronkels van gras ~2-4.
    Kleine stukken echte lijn zijn vrijwel recht, grote netwerken mogen meer."""
    n, lab_cc, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    half_width = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    skel_len = np.bincount(lab_cc[centerlines(mask) > 0], minlength=n)
    keep = np.zeros(n, bool)
    dense = np.zeros(n, bool)
    for k in range(1, n):
        x, y, w, h, area = stats[k]
        if area < 10:
            continue
        sub = lab_cc[y:y + h, x:x + w] == k
        thick = 2 * float(half_width[y:y + h, x:x + w][sub].max())
        extent = float(np.hypot(w, h))
        line_like = extent >= 15 and extent >= 5 * max(thick, 1.5)
        ratio = skel_len[k] / max(extent, 1.0)
        limit = max_ratio if max_ratio is not None else (1.6 if extent < 80 else 2.6)
        keep[k] = line_like and ratio <= limit
        dense[k] = extent >= 15 and ratio > 4
    return (np.where(keep[lab_cc], 255, 0).astype(np.uint8),
            np.where(dense[lab_cc], 255, 0).astype(np.uint8))


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


def pitch_roi(H_work: np.ndarray, size: tuple[int, int], margin: float = 4.0,
              geom: pitch.Geometry = pitch.DEFAULT) -> np.ndarray:
    """Masker van het (voorspelde) veld plus marge, in werkschaal."""
    W, Hh = size
    L, Wp = geom.length, geom.width
    poly = np.array([[-margin, -margin], [L + margin, -margin], [L + margin, Wp + margin], [-margin, Wp + margin]], float)
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


def camera_from_h(H: np.ndarray, cam: dict, size: tuple[int, int], p0: np.ndarray,
                  geom: pitch.Geometry = pitch.DEFAULT) -> np.ndarray:
    """Kijkrichting [yaw, tilt, roll, log_f] die bij homografie H past, met vaste camerapositie."""
    from .calibration import _lm, camera_homography

    xs, ys = np.meshgrid(np.linspace(-5, geom.length + 5, 23), np.linspace(-5, geom.width + 5, 15))
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


CAM_SEARCH_DEG = 5.0  # zo ver mag een voorspelling (bijv. doorgegeven via beeldherkenning) ernaast zitten
MIN_FIXED_PX = 1.0  # voorstel: zoveel pixels per graad moeten de lijnen minstens verschuiven (gemeten op 9 handmomenten:
                    # goede voorstellen 1,7-1,9, foute op vage graslijnen 0-0,5)
MIN_SENS_PX_PER_DEG = 3.0  # de lijnen moeten minstens zoveel pixels per graad verschuiven bij draaien


def _goal_yaw(frame: np.ndarray, size: tuple[int, int], S: np.ndarray, full: tuple[int, int],
              geom: pitch.Geometry, to_params, q: np.ndarray, fit_at, debug: dict | None = None) -> dict | None:
    """Draai tot 4 graden om stand q en kijk waar de doelpalen en de lat op een wit frame vallen.
    Niet verder: op veel sportparken staan meer doelen (pupillendoeltjes, het veld ernaast), en een
    ander doel mag niet "gevonden" worden.
    fit_at(yaw): de stand met die draairichting waarbij de lijnen weer precies passen (kantelen en
    scheefstand compenseren); zo blijft de zijlijn op zijn plek en verschuift alleen het doel.
    Geeft {"q", "dyaw_deg", "gain"} bij een duidelijke piek, anders None (geen doel, of niet te zien)."""
    from .calibration import camera_projection
    goal_dt = goal_evidence(cv2.resize(frame, size, interpolation=cv2.INTER_AREA))
    dyaws = np.radians(np.arange(-4.0, 4.01, 0.25))
    qs = [fit_at(q[0] + d) for d in dyaws]
    Ps = np.stack([S @ camera_projection(to_params(v), full) for v in qs])
    gg = _goal_gain(Ps, size, goal_dt, geom, tau=3.0)
    if debug is not None:
        debug["goal_scan"] = [round(float(v), 1) for v in gg]
    # alleen een piek binnen het zoekgebied telt: op de rand kan het ook iets ernaast zijn (een
    # lichtmast, het andere doel)
    inner = gg.copy()
    inner[:1] = inner[-1:] = -1.0
    k, base = int(np.argmax(inner)), float(np.median(gg))
    if gg[k] >= 8 and gg[k] - base >= 8 and gg[k] >= 1.6 * base and gg[k] >= 0.8 * gg.max():  # een duidelijke piek
        # fijner rond de piek
        fine = np.radians(np.arange(-0.2, 0.21, 0.05)) + dyaws[k]
        qf = [fit_at(q[0] + d) for d in fine]
        gf = _goal_gain(np.stack([S @ camera_projection(to_params(v), full) for v in qf]), size, goal_dt, geom, tau=3.0)
        j = int(np.argmax(gf))
        return {"gain": round(float(gf[j]), 1), "dyaw_deg": round(math.degrees(fine[j]), 2), "q": qf[j]}
    return None


def refine_camera(frame: np.ndarray, H_pred: np.ndarray, boxes: np.ndarray | None, camera: dict,
                  geom: pitch.Geometry = pitch.DEFAULT, enhance: bool = False,
                  debug: dict | None = None) -> tuple[np.ndarray, dict] | None:
    """camera["search"]: ook ver (tot CAM_SEARCH_DEG) om de voorspelling heen zoeken en op het doel
    richten; voor een kalibratie die via de omgeving is doorgegeven. Zonder (tijdens het volgen, de
    voorspelling ligt dan al dichtbij) is het veel sneller."""
    """Kalibratie bijstellen op de lijnen als bekend is waar de camera staat (plek, hoogte, zoom).

    Stap voor stap, zoals je een overtrekvel op een foto legt:
    1. grof: de kijkrichting tot CAM_SEARCH_DEG graden om de voorspelling heen proberen (draaien en
       kantelen) en kijken waar de getekende lijnen op witte lijnen vallen;
    2. precies: kleinste kwadraten op de lijnpixels (_lsq_lines); zoom en scheefstand blijven dicht
       bij die van het handmatige sleutelframe;
    3. draairichting: langs één zijlijn draaien is nauwelijks te zien (de lijn schuift langs zichzelf).
       Staat er een doel in beeld, dan richten we daarop; anders houden we die van de voorspelling en
       corrigeren we alleen op/neer en scheefstand (de lijn 'ver boven de zijlijn' verdwijnt dan wel);
    4. keuring: de gevonden lijnen moeten op het model vallen, en op/neer moet echt vastliggen.
    None als het niet betrouwbaar is."""
    from .calibration import camera_homography

    full = (frame.shape[1], frame.shape[0])
    s = WORK_WIDTH / frame.shape[1]
    S = np.diag([s, s, 1.0])
    size = (WORK_WIDTH, int(round(frame.shape[0] * s)))
    fixed = [camera["x"], camera["y"], camera["h"]]

    def to_params(q):
        return np.array([*fixed, *q])

    def Hw_of(q):
        return S @ camera_homography(to_params(q), full)

    q0 = camera_from_h(H_pred, camera, full, camera["q0"], geom)
    mdl = model(geom)
    lines = detect_lines(frame, boxes, roi=pitch_roi(Hw_of(q0), size, margin=10.0, geom=geom), enhance=enhance)
    if lines.sum() / 255 < 100:
        return None
    lines = centerlines(lines)
    dt = cv2.distanceTransform(255 - lines, cv2.DIST_L2, 3)
    det_y, det_x = np.nonzero(lines)
    det_xy = np.stack([det_x, det_y], 1)

    # 1. grof: draaien en kantelen rond de voorspelling
    search = bool(camera.get("search"))
    starts = []
    for dyaw in (np.arange(-CAM_SEARCH_DEG, CAM_SEARCH_DEG + 1e-6, 0.5) if search else ()):
        for dtilt in np.arange(-2.5, 2.51, 0.5):
            q = q0 + np.radians([dyaw, dtilt, 0.0, 0.0]) * np.array([1, 1, 0, 0])
            g, n, _ = _view_gain(Hw_of(q), dt, size, 12.0, mdl.samples, mdl.tangents)
            starts.append((g, q))
    starts.sort(key=lambda c: -c[0])
    picked = [q0]
    for g, q in starts:
        if len(picked) >= 4:
            break
        if all(np.degrees(np.abs(q[:2] - o[:2])).max() > 1.2 for o in picked):
            picked.append(q)

    # 2. precies, vanaf elk startpunt; de stand met de meeste raak-pixels wint
    roll0, lf0 = camera["roll"], camera["log_f"]

    def prior_res(q):  # kijkrichting: als de lijnen hem niet vastleggen, die van de voorspelling houden
        return 2.0 * np.array([(q[0] - q0[0]) / math.radians(1.5), (q[2] - roll0) / math.radians(1.0),
                               (q[3] - lf0) / 0.01])  # zoom: die van het handmatige sleutelframe

    best = None
    for qs in picked:
        q, info = _lsq_lines(det_xy, S, full, geom, to_params, qs, prior_res, gates=(12.0, 6.0, 3.0))
        Hw = Hw_of(q)
        prec = _precision(Hw, det_xy, size, geom=geom, cam=np.array(fixed[:2]))
        cand = {"q": q, "n": info["n"], "J": info["J"], "sv": info["sv"], "null": info["null"], "precision": prec,
                "score": info["n"] * prec ** 4}
        if best is None or cand["score"] > best["score"]:
            best = cand
    q = best["q"]
    # pixels per graad: [draaien en kantelen samen (de zwakste combinatie), kantelen alleen]
    sens = np.array([best["sv"][-1], best["J"][1]]) * math.radians(1.0)
    # 3. de draairichting. Een doel in beeld legt die het best vast (palen en lat op het witte frame);
    #    anders de lijnen, als die genoeg verschuiven bij draaien; anders die van de voorspelling houden.
    null = best["null"] / best["null"][0] if abs(best["null"][0]) > 0.3 else np.array([1.0, 0.0, 0.0])

    def fit_at(yaw):  # draairichting vast; kantelen, scheefstand en zoom passend op de lijnen
        def pr(v):
            return 2.0 * np.array([(v[0] - yaw) / math.radians(0.02), (v[2] - roll0) / math.radians(3.0),
                                   (v[3] - lf0) / 0.01])
        start = q + np.array([*((yaw - q[0]) * null), 0.0])  # langs de richting waarin de lijn blijft liggen
        return _lsq_lines(det_xy, S, full, geom, to_params, start, pr, gates=(8.0, 3.0), max_pts=300, iters=6)[0]

    goal_fix = _goal_yaw(frame, size, S, full, geom, to_params, np.array([q0[0], *q[1:]]), fit_at, debug) if search else None
    keep = None
    if goal_fix is not None:
        keep, sigma, gates = goal_fix.pop("q"), 0.3, (6.0, 3.0)
    elif sens[0] < MIN_SENS_PX_PER_DEG:  # alleen de zijlijn: op/neer, scheefstand en zoom bijstellen
        keep, sigma, gates = np.array([q0[0], q[1], q[2], q[3]]), 0.5, (8.0, 4.0, 3.0)
    if keep is not None:
        def prior_keep(v):
            return 2.0 * np.array([(v[0] - keep[0]) / math.radians(sigma), (v[2] - roll0) / math.radians(1.0),
                                   (v[3] - lf0) / 0.01])
        q, _ = _lsq_lines(det_xy, S, full, geom, to_params, keep, prior_keep, gates=gates)
    Hw = Hw_of(q)
    gain, n, hit = _view_gain(Hw, dt, size, 3.0, mdl.samples, mdl.tangents)
    rot = float(np.degrees(np.hypot(*(q[:2] - q0[:2]))))
    info = {"after": round(hit, 3), "n": int(best["n"]), "precision": round(best["precision"], 3),
            "rot_deg": round(rot, 2), "zoom": round(math.exp(abs(q[3] - lf0)), 3),
            "sens_px_per_deg": [round(float(v), 1) for v in sens], "camera_mode": True, "methode": "lijnen-camera"}
    if debug is not None:
        debug.update(info)
    # Veel lijnpixels precies op het model is sterk bewijs, ook als er daarnaast wat rommel (schittering
    # op kunstgras, schaduwranden) als lijn is aangezien
    clean = best["precision"] >= PRECISION_MIN or (best["n"] >= 300 and best["precision"] >= 0.6)
    info["yaw_uncertain"] = bool(sens[0] < MIN_SENS_PX_PER_DEG and goal_fix is None)  # alleen op/neer en scheefstand
    if goal_fix:
        info["doel"] = goal_fix
    if (best["n"] < 60 or not clean or sens[1] < MIN_SENS_PX_PER_DEG
            or rot > CAM_SEARCH_DEG + 1.0 or abs(q[3] - lf0) > 0.03 or abs(q[2] - roll0) > math.radians(3)):
        return None
    info["params"] = [round(float(v), 6) for v in q]
    return normalize_h(np.linalg.inv(S) @ Hw), info


def refine(frame: np.ndarray, H_pred: np.ndarray, boxes: np.ndarray | None, f_full: float,
           debug: dict | None = None, camera: dict | None = None,
           geom: pitch.Geometry = pitch.DEFAULT, enhance: bool = False) -> tuple[np.ndarray, dict] | None:
    """Stel H (veld -> beeld, volle resolutie) bij op de lijnen in dit beeld. None = niet betrouwbaar.

    Zonder `camera`: een kleine extra draaiing + zoom bovenop de voorspelling (relatief).
    Met `camera` ({x, y, h, roll, log_f, q0}): de camera staat stil op een bekende plek en zoomt
    niet, dus we zoeken de absolute kijkrichting; zoom en scheefstand worden naar de waarden van
    het handmatige sleutelframe getrokken. Dan is ook één zichtbare lijn genoeg en kan er niets
    ongemerkt 'langs de lijn' wegglijden."""
    from .calibration import camera_homography

    if camera is not None:
        return refine_camera(frame, H_pred, boxes, camera, geom=geom, enhance=enhance, debug=debug)
    full = (frame.shape[1], frame.shape[0])
    s = WORK_WIDTH / frame.shape[1]
    S = np.diag([s, s, 1.0])
    Hw = S @ H_pred
    size = (WORK_WIDTH, int(round(frame.shape[0] * s)))
    Wd, Hd = size
    mdl = model(geom)
    SAMPLES, TANGENTS, DENSE = mdl.samples, mdl.tangents, mdl.dense
    lines = detect_lines(frame, boxes, roi=pitch_roi(Hw, size, geom=geom), enhance=enhance)
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
        x0 = camera_from_h(H_pred, camera, full, camera["q0"], geom)
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


def keyframe_points(H: np.ndarray, size: tuple[int, int], geom: pitch.Geometry = pitch.DEFAULT) -> list[dict]:
    """Een raster van veldpunten dat in beeld ligt, als sleutelframe-punten (beeld <-> veld)."""
    xs, ys = np.meshgrid(np.linspace(0, geom.length, 15), np.linspace(0, geom.width, 9))
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
                _, cam = fit_camera(kf["points"], kf_camera(prior, kf))
                return (clip["width"] / 2) / math.tan(math.radians(cam["hfov_deg"]) / 2)
            except ValueError:
                continue
    return default_focal(int(clip["width"]))


def run_autocalib(store, clip_id: int, every_s: float = 1.0, progress=None) -> dict:
    """Leg elke `every_s` seconden een automatisch sleutelframe vast (waar de lijnen dat toelaten)."""
    from .storage import clip_dir

    clip = store.one("SELECT * FROM clips WHERE id = ?", (clip_id,))
    geom = pitch.of_match(store.one("SELECT * FROM matches WHERE id = ?", (clip["match_id"],)))
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
            K, _ = fit_calibration(kf["points"], camera=kf_camera(prior, kf))
        except ValueError:
            continue
        accepted[int(np.argmin(np.abs(t - kf["t"])))] = K
    if not accepted:
        raise ValueError("Leg eerst zelf één ijkmoment vast (Kalibratie)")
    for kf in kept:
        try:
            accepted.setdefault(int(np.argmin(np.abs(t - kf["t"]))), fit_calibration(kf["points"])[0])
        except ValueError:
            continue
    if not accepted:
        raise ValueError("Leg eerst zelf één ijkmoment vast (Kalibratie)")
    f_full = _focal(clip, manual)
    # Bekende camerapositie (aangeklikt of via GPS)? Dan bijstellen met het cameramodel: vaste plek,
    # zoom en scheefstand vastgehouden aan het handmatige sleutelframe.
    cam_mode = None
    params_of: dict[int, np.ndarray] = {}
    if prior is not None:
        for kf in manual:
            try:
                _, cam = fit_camera(kf["points"], kf_camera(prior, kf))
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
        args = (frame, normalize_h(H_pred), np.array(boxes_of.get(i, [])).reshape(-1, 4), f_full)
        res = refine(*args, camera=camera, geom=geom)
        if res is None:  # lastig beeld: nog eens, en dan ook naar zwakke lijnen zoeken
            res = refine(*args, camera=camera, geom=geom, enhance=True)
        if res is None:
            return False
        H_new, info = res
        pts = keyframe_points(H_new, size, geom)
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

def _image_length(H: np.ndarray, pts: np.ndarray, tans: np.ndarray, step: float) -> np.ndarray:
    """Hoe lang het stuk lijn dat een modelpunt voorstelt (step meter) in beeld is (pixels)."""
    def proj(p):
        hom = np.hstack([p, np.ones((len(p), 1))]) @ H.T
        return hom[:, :2] / np.where(np.abs(hom[:, 2:3]) > 1e-9, hom[:, 2:3], 1e-9)
    return np.linalg.norm(proj(pts + 0.5 * step * tans) - proj(pts - 0.5 * step * tans), axis=1)


SEG_PX, SEG_MAX = 5.0, 3.0  # een modelpunt telt per 5 pixel lijn in beeld, hooguit 8 keer


def _view_gain(H_work: np.ndarray, dt: np.ndarray, size: tuple[int, int], tau: float,
               samples: np.ndarray = SAMPLES, tangents: np.ndarray | None = None,
               step: float = 0.5) -> tuple[float, int, float]:
    """Hoe goed de zichtbare modellijnen op gevonden lijnen vallen: (winst, aantal zichtbaar, deel raak).

    Elk zichtbaar modelpunt levert +1 op als het precies op een lijn valt en -0,3 als er niets in
    de buurt is. Een kijkrichting waarin veel lijnen kloppen wint dus van een waarin toevallig een
    stukje lijn past. Lijnen die in beeld te dun zijn om te zien (ver weg) tellen niet mee.
    tangents: dan telt elk punt naar de lengte van zijn stuk lijn in beeld (de zijlijn vlak voor je
    loopt over het hele beeld en weegt dus zwaar, ook al is het maar een paar meter veld)."""
    Wd, Hd = size
    hom = np.hstack([samples, np.ones((len(samples), 1))]) @ H_work.T
    front = hom[:, 2] > 1e-6
    q = hom[:, :2] / np.where(front, hom[:, 2], 1)[:, None]
    vis = front & (q[:, 0] >= 0) & (q[:, 0] < Wd - 1) & (q[:, 1] >= 0) & (q[:, 1] < Hd - 1)
    wpen = np.zeros(len(samples))
    if vis.any():
        w = _line_width_px(H_work, samples[vis])
        wpen[vis] = _miss_weight(w)
        vis[vis] = w >= 1.5
    n = int(vis.sum())
    if n < 10:
        return -1e3, n, 0.0
    qi = q[vis].astype(np.float32)
    d = cv2.remap(dt, qi[:, 0:1], qi[:, 1:2], cv2.INTER_LINEAR).ravel()
    v = 1.0 - 1.3 * np.minimum(d, tau) / tau
    v = np.where(v < 0, v * wpen[vis], v)
    if tangents is None:
        return float(np.sum(v)), n, float((d < 2.5).mean())
    wt = np.clip(_image_length(H_work, samples[vis], tangents[vis], step) / SEG_PX, 1.0, SEG_MAX)
    return float(np.sum(v * wt)), n, float(np.sum((d < 2.5) * wt) / np.sum(wt))


def _miss_weight(width_px: np.ndarray) -> np.ndarray:
    """Hoe zwaar een niet gevonden stuk modellijn telt, naar hoe breed de lijn in beeld is. Een
    lijn vlak voor je (breed) zie je altijd; een verre, dunne lijn valt in tegenlicht of tegen witte
    reclameborden vaak weg. Zo wint niet de stand die 'zo min mogelijk lijnen in beeld' voorspelt."""
    return np.clip((width_px - 1.5) / 3.0, 0.0, 1.0)


def _near_side(fx: np.ndarray, fy: np.ndarray, cam: np.ndarray | None, geom: pitch.Geometry,
               margin: float = 0.5) -> np.ndarray:
    """Ligt dit veldpunt buiten het veld, aan de kant van de camera (tussen jou en de lijn)?

    Daar staan vaak lijnen die niet bij het veld horen: de coachzone voor de dug-out, de lijnen van
    een pupillenveldje, de rand van het kunstgras. Die zeggen niets over of de kalibratie klopt.
    cam: (x, y) of (M, 2) camerastanden; None = niets uitsluiten."""
    if cam is None:
        return np.zeros(np.shape(fx), bool)
    cam = np.asarray(cam, float).reshape(-1, 2)
    cx, cy = cam[:, 0:1], cam[:, 1:2]
    if np.ndim(fx) == 1:
        cx, cy = cx[0], cy[0]
    L, W = geom.length, geom.width
    return (((cy > W) & (fy > W + margin)) | ((cy < 0) & (fy < -margin))
            | ((cx > L) & (fx > L + margin)) | ((cx < 0) & (fx < -margin)))


def _precision(H_work: np.ndarray, det_xy: np.ndarray, size: tuple[int, int], px: float = 3.0,
               geom: pitch.Geometry = pitch.DEFAULT, cam: np.ndarray | None = None) -> float:
    """Deel van de gevonden lijnpixels binnen het veld dat op een modellijn valt.
    cam: camerapositie; lijnen tussen de camera en het veld tellen dan niet mee."""
    Wd, Hd = size
    if not len(det_xy):
        return 0.0
    DENSE = model(geom).dense
    roi = pitch_roi(H_work, size, margin=3.0, geom=geom)
    inside = roi[det_xy[:, 1], det_xy[:, 0]] > 0
    if cam is not None and inside.any():
        f = apply_h(np.linalg.inv(H_work), det_xy[inside].astype(float))
        near = _near_side(f[:, 0], f[:, 1], cam, geom)
        inside[np.flatnonzero(inside)[near]] = False
    if inside.sum() < 30:
        return 0.0
    canvas = np.zeros((Hd, Wd), np.uint8)
    cv2.polylines(canvas, _model_polylines(H_work, size, DENSE), False, 255, 1)
    dm = cv2.distanceTransform(255 - cv2.dilate(canvas, np.ones((3, 3), np.uint8)), cv2.DIST_L2, 3)
    pts = det_xy[inside]
    return float((dm[pts[:, 1], pts[:, 0]] < px).mean())


# --- het doel als herkenningspunt -----------------------------------------------------------------
# Een doel is een wit frame van vaste maat (7,32 x 2,44 m) dat vaak nog goed te zien is als de lijnen
# in het tegenlicht wegvallen. Voor elke camerastand weten we waar de palen en de lat in beeld horen;
# staan daar dunne witte staanders en een witte ligger, dan klopt die stand.

GOAL_STEP = 0.25


@lru_cache(maxsize=8)
def goal_samples(geom: pitch.Geometry = pitch.DEFAULT) -> tuple[np.ndarray, np.ndarray]:
    """Punten (x, y, hoogte) op beide doelen en per punt de soort: 0 = paal (staand), 1 = lat (liggend).
    De voet van de palen laten we weg: daar staan vaak keepers, spelers of reclameborden voor."""
    c, half, top = geom.width / 2, 3.66, pitch.GOAL_HEIGHT + 0.06  # midden van de lat
    pts, kind = [], []
    for x0 in (0.0, geom.length):
        for y in (c - half - 0.06, c + half + 0.06):
            for z in np.arange(0.5, top + 1e-6, GOAL_STEP):
                pts.append((x0, y, z)); kind.append(0)
        for y in np.arange(c - half, c + half + 1e-6, GOAL_STEP):
            pts.append((x0, y, top)); kind.append(1)
    return np.array(pts, float), np.array(kind)


def goal_evidence(small: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Afstandskaarten (werkschaal) tot dunne witte staanders (palen) en liggers (lat)."""
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    L = lab[..., 0]
    chroma = np.hypot(lab[..., 1].astype(np.float32) - 128, lab[..., 2].astype(np.float32) - 128)
    white = (chroma < 22) & (L > 165)  # fel wit (geen grijze lucht tussen bladeren, geen geel gras)
    out = []
    for k_open, k_long in (((9, 1), (1, 11)), ((1, 9), (11, 1))):  # staand: smal in x; liggend: smal in y
        th = cv2.morphologyEx(L, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, k_open))
        m = ((th > 30) & white).astype(np.uint8) * 255
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, k_long))  # echt een streep
        out.append(cv2.distanceTransform(255 - m, cv2.DIST_L2, 3))
    return out[0], out[1]


def _goal_gain(P: np.ndarray, size: tuple[int, int], dts: tuple[np.ndarray, np.ndarray],
               geom: pitch.Geometry, tau: float = 3.0) -> np.ndarray:
    """Hoe goed de doelen op witte palen en latten vallen, per camerastand. P: (M, 3, 4) in werkschaal.
    Alleen winst (geen straf): een doel dat wegvalt tegen witte borden of achter de keeper, of
    dat buiten beeld ligt, kost niets."""
    Wd, Hd = size
    pts, kind = goal_samples(geom)
    X = np.vstack([pts[:, 0], pts[:, 1], -pts[:, 2], np.ones(len(pts))])  # hoogte z -> Z = -z
    hh = P @ X  # (M, 3, N)
    z = hh[:, 2]
    front = z > 1e-6
    zs = np.where(front, z, 1.0)
    u, v = hh[:, 0] / zs, hh[:, 1] / zs
    vis = front & (u >= 0) & (u < Wd - 1) & (v >= 0) & (v < Hd - 1)
    ui, vi = np.where(vis, u, 0).astype(np.int32), np.where(vis, v, 0).astype(np.int32)
    d = np.where(kind[None, :] == 0, dts[0][vi, ui], dts[1][vi, ui])
    g = np.where(vis, np.maximum(0.0, 1.0 - np.minimum(d, tau) / tau), 0.0)
    return g.sum(1)


def _model_polylines(H_work: np.ndarray, size: tuple[int, int], dense: np.ndarray) -> list[np.ndarray]:
    """De modellijnen als doorgetrokken lijnstukken in beeld. Losse punten om de 10 cm liggen vlak
    voor de camera tientallen pixels uit elkaar; dan zou een goed passende zijlijn tussen de puntjes
    door 'naast het model' vallen."""
    Wd, Hd = size
    hom = np.hstack([dense, np.ones((len(dense), 1))]) @ H_work.T
    fr = hom[:, 2] > 1e-6
    q = hom[:, :2] / np.where(fr, hom[:, 2], 1.0)[:, None]
    ok = fr & (np.abs(q) < 8 * max(Wd, Hd)).all(1)
    # opeenvolgende punten van hetzelfde lijnstuk (10 cm uit elkaar) verbinden
    step_ok = ok[:-1] & ok[1:] & (np.linalg.norm(np.diff(dense, axis=0), axis=1) < 0.15)
    out, start = [], None
    for i, good in enumerate(step_ok):
        if good and start is None:
            start = i
        elif not good and start is not None:
            out.append(np.round(q[start:i + 1]).astype(np.int32))
            start = None
    if start is not None:
        out.append(np.round(q[start:]).astype(np.int32))
    return out


def _batch_homographies(P: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """camera_homography voor veel parametersets tegelijk: (M, 7) -> (M, 3, 3)."""
    return _batch_projections(P, size)[:, :, [0, 1, 3]]


def _batch_projections(P: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """camera_projection voor veel parametersets tegelijk: (M, 7) -> (M, 3, 4), zelfde schaal als
    _batch_homographies (een punt op hoogte z heeft Z = -z)."""
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
    E = np.concatenate([R, t[:, :, None]], 2)
    K = np.zeros((len(P), 3, 3))
    K[:, 0, 0] = f
    K[:, 1, 1] = f
    K[:, 0, 2] = W / 2
    K[:, 1, 2] = Hh / 2
    K[:, 2, 2] = 1
    Ps = K @ E
    return Ps / np.abs(Ps[:, 2:3, 3:4])


def _coarse_scores(P: np.ndarray, full: tuple[int, int], s: float, dt: np.ndarray, size: tuple[int, int],
                   det: np.ndarray, tau_px: float = 25.0, tau_m: float = 1.5, keep: float = 0.15,
                   geom: pitch.Geometry = pitch.DEFAULT, goal_dt: tuple | None = None) -> np.ndarray:
    """Grove score voor veel camerastanden tegelijk: (modellijnen op gevonden lijnen) x (deel van
    de gevonden lijnen dat het model verklaart). Het tweede deel alleen voor de beste `keep`."""
    Wd, Hd = size
    Hs = (_batch_homographies(P, full) * np.array([s, s, 1.0])[None, :, None])
    c2, c2t = model(geom).c2, model(geom).c2_tan
    X = np.vstack([c2.T, np.ones(len(c2))]).astype(np.float32)
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
    val = 1.0 - 1.3 * d / tau_px
    val = np.where(val < 0, val * _miss_weight(wmin), val)
    # elk punt staat voor 2 m lijn: tel het naar de lengte daarvan in beeld (zie _view_gain)
    Tt = np.vstack([c2t.T, np.zeros(len(c2t))]).astype(np.float32)
    ht = Hf @ Tt  # richting van de lijn
    du = (ht[:, 0] - u * ht[:, 2]) / zs
    dv = (ht[:, 1] - v * ht[:, 2]) / zs
    seg = np.hypot(du, dv) * 2.0
    wt = np.clip(seg / (SEG_PX * 4), 1.0, SEG_MAX)
    gain = np.where(vis, val * wt, 0.0).sum(1)
    gain[vis.sum(1) < 8] = 0.0
    out = np.zeros(len(P))
    top = np.flatnonzero(gain > 0)
    if len(top) > keep * len(P):
        top = top[np.argsort(-gain[top])[:max(1, int(keep * len(P)))]]
    if len(top):
        g = gain[top]
        if goal_dt is not None:  # palen en lat op witte staanders en een ligger: extra winst
            Ps = _batch_projections(P[top], full) * np.array([s, s, 1.0])[None, :, None]
            g = g + _goal_gain(Ps, size, goal_dt, geom, tau=6.0)
        out[top] = g * np.maximum(_batch_explained(Hs[top], det, tau_m, geom, cam=P[top, :2]), 0.0)
    return out


_FIELD_RES, _FIELD_MARGIN = 0.1, 12.0


def _field_distance(geom: pitch.Geometry = pitch.DEFAULT) -> np.ndarray:
    """Bovenaanzicht: per 10 cm de afstand (m) tot de dichtstbijzijnde veldlijn."""
    m = model(geom)
    if m._field_dt is None:
        w = int((geom.length + 2 * _FIELD_MARGIN) / _FIELD_RES)
        h = int((geom.width + 2 * _FIELD_MARGIN) / _FIELD_RES)
        img = np.full((h, w), 255, np.uint8)
        q = ((m.dense + _FIELD_MARGIN) / _FIELD_RES).astype(int)
        img[np.clip(q[:, 1], 0, h - 1), np.clip(q[:, 0], 0, w - 1)] = 0
        m._field_dt = cv2.distanceTransform(img, cv2.DIST_L2, 3) * _FIELD_RES
    return m._field_dt


def _batch_explained(Hs_work: np.ndarray, det: np.ndarray, tau_m: float,
                     geom: pitch.Geometry = pitch.DEFAULT, cam: np.ndarray | None = None) -> np.ndarray:
    """Hoeveel van de gevonden lijnpixels door het veldmodel verklaard worden, per camerastand.

    Elk gevonden pixel wordt teruggeprojecteerd op het veld; ligt het op een veldlijn dan +1,
    ver ernaast -0,3. Zo verliest een stand die maar een paar lijnen 'gebruikt'."""
    fd = _field_distance(geom)
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
    if cam is None:
        return v.mean(1)
    # lijnen tussen de camera en het veld (coachzone, pupillenveldje) niet meetellen
    skip = fwd & _near_side(hh[:, 0] / zs, hh[:, 1] / zs, cam, geom)
    n = np.maximum((~skip).sum(1), 1)
    return np.where(skip, 0.0, v).sum(1) / n


@lru_cache(maxsize=8)
def _segments(geom: pitch.Geometry = pitch.DEFAULT) -> tuple[np.ndarray, np.ndarray]:
    """Alle veldlijnen als rechte stukjes (begin, eind) in meters; cirkels en bogen in stukjes van ~0,5 m."""
    a, b = [], []
    for p0, p1 in geom.lines.values():
        a.append(p0); b.append(p1)
    pts = pitch_samples(step=0.5, geom=geom)
    # de bogen: opeenvolgende punten die niet op een rechte lijn liggen (cirkel + penaltybogen)
    L, W, c = geom.length, geom.width, geom.half_w
    for cx, a0, a1 in ((L / 2, 0, 2 * math.pi), (11, -math.acos(5.5 / 9.15), math.acos(5.5 / 9.15)),
                       (L - 11, math.pi - math.acos(5.5 / 9.15), math.pi + math.acos(5.5 / 9.15))):
        t = np.linspace(a0, a1, max(4, int(9.15 * (a1 - a0) / 0.5)))
        q = np.stack([cx + 9.15 * np.cos(t), c + 9.15 * np.sin(t)], 1)
        a += list(q[:-1]); b += list(q[1:])
    return np.array(a, float), np.array(b, float)


def _lsq_lines(det_xy: np.ndarray, S: np.ndarray, full: tuple[int, int], geom: pitch.Geometry,
               to_params, q0: np.ndarray, prior_res, gates=(5.0, 3.0, 3.0), max_pts: int = 1500,
               iters: int = 30) -> tuple[np.ndarray, dict]:
    """Kleinste kwadraten op de lijnpixels: elk gevonden lijnpixel trekt de dichtstbijzijnde modellijn
    naar zich toe (afstand in beeldpixels, werkschaal), zoals je een overtrekvel precies op een foto
    legt. Alleen pixels die al binnen `gate` pixels van een modellijn liggen tellen mee (eerst ruim,
    dan steeds strenger); lijnen tussen de camera en het veld niet.

    to_params(q) -> cameraparameters [x, y, h, yaw, tilt, roll, log_f]; prior_res(q) -> extra residuen.
    Geeft (q, info) met info["n"] (meetellende pixels) en info["J"] (gevoeligheid per parameter)."""
    from .calibration import _lm, camera_homography

    A, B = _segments(geom)
    D = B - A
    DD = np.maximum((D ** 2).sum(1), 1e-12)
    rng = np.random.default_rng(1)
    det = det_xy[rng.choice(len(det_xy), min(max_pts, len(det_xy)), replace=False)].astype(float)

    def assoc(pp, gate):
        Hw = S @ camera_homography(pp, full)
        f = apply_h(np.linalg.inv(Hw), det)
        ok = np.isfinite(f).all(1)
        ok &= ~_near_side(f[:, 0], f[:, 1], pp[:2], geom)
        idx = np.flatnonzero(ok)
        f = f[idx]
        t = np.clip(((f[:, None, :] - A[None]) * D[None]).sum(2) / DD[None], 0, 1)  # (N, K)
        c = A[None] + t[..., None] * D[None]
        k = np.argmin(((f[:, None, :] - c) ** 2).sum(2), axis=1)
        cp = c[np.arange(len(idx)), k]
        d = np.linalg.norm(apply_h(Hw, cp) - det[idx], axis=1)
        keep = np.isfinite(d) & (d < gate)
        return idx[keep], cp[keep], k[keep]

    def line_res(q, idx, cp, k):
        Hw = S @ camera_homography(to_params(q), full)
        u = D[k] / np.sqrt(DD[k])[:, None]
        p1, p2 = apply_h(Hw, cp - u), apply_h(Hw, cp + u)  # 1 m naar weerszijden langs de lijn
        dvec = p2 - p1
        n = np.stack([-dvec[:, 1], dvec[:, 0]], 1) / np.maximum(np.linalg.norm(dvec, axis=1), 1e-9)[:, None]
        r = ((det[idx] - p1) * n).sum(1)
        return np.where(np.isfinite(r), r, gates[-1])

    q = np.asarray(q0, float)
    idx = np.zeros(0, int)
    for gate in gates:
        idx, cp, k = assoc(to_params(q), gate)
        if len(idx) < 30:
            return q, {"n": int(len(idx)), "J": np.zeros(len(q)), "sv": np.zeros(min(3, len(q))),
                       "null": np.eye(min(3, len(q)))[0], "normal": np.zeros((len(q), len(q)))}
        q = _lm(lambda v: np.concatenate([line_res(v, idx, cp, k), prior_res(v)]), q, iters=iters)
    idx, cp, k = assoc(to_params(q), gates[-1])
    # gevoeligheid: hoeveel pixels verschuiven de lijnen per eenheid van elke parameter (J), en van de
    # eerste drie samen (sv): kunnen die elkaar opheffen (draaien + kantelen + scheefstand geeft
    # dezelfde lijn), dan is de kleinste singuliere waarde klein, ook al verschuift elk apart wel
    J, sv = np.zeros(len(q)), np.zeros(min(3, len(q)))
    if len(idx) >= 10:
        r0 = line_res(q, idx, cp, k)
        cols = []
        for j in range(len(q)):
            e = np.zeros(len(q)); e[j] = 1e-4
            cols.append((line_res(q + e, idx, cp, k) - r0) / 1e-4)
        M = np.stack(cols, 1)
        J = np.sqrt(np.mean(M ** 2, axis=0))
        normal = M.T @ M / len(idx)
        m = min(3, len(q))
        _, sv, vt = np.linalg.svd(M[:, :m] / math.sqrt(len(idx)), full_matrices=False)
        null = vt[-1]  # richting (eerste drie parameters) waarin de lijnen het minst verschuiven
    else:
        null = np.eye(min(3, len(q)))[0]
        normal = np.zeros((len(q), len(q)))
    return q, {"n": int(len(idx)), "J": J, "sv": sv, "null": null, "normal": normal}


def _polish(p: np.ndarray, det_xy: np.ndarray, S: np.ndarray, full: tuple[int, int], size: tuple[int, int],
            geom: pitch.Geometry, sp: float, p0: np.ndarray, lf0: float) -> tuple[np.ndarray, dict]:
    """Laatste, precieze bijstelling van een voorstel (plek binnen de onzekerheid, zoom en
    scheefstand dicht bij wat we al wisten). Zie _lsq_lines.

    Geeft ook hoe vast de lijnen de stand leggen: "fixed_px" = hoeveel pixels de lijnen minstens
    verschuiven bij de zwakste combinatie van 1 graad draaien/kantelen/scheef. Alleen een zijlijn in
    beeld: dan kun je langs die lijn draaien zonder dat het opvalt (klein)."""
    h = p[2]

    def to_params(q):
        return np.array([q[0], q[1], h, q[2], q[3], q[4], q[5]])

    def prior_res(q):
        return 2.0 * np.array([(q[0] - p0[0]) / sp, (q[1] - p0[1]) / sp, q[4] / math.radians(4), (q[5] - lf0) / 0.5])

    q, info = _lsq_lines(det_xy, S, full, geom, to_params, np.array([p[0], p[1], *p[3:]]), prior_res)
    # alleen draaien/kantelen/scheef (de plek komt van de luchtfoto of GPS: die leggen de lijnen
    # vanaf een lage camera nauwelijks vast)
    D = np.diag([math.radians(1.0)] * 3)  # per graad
    A = D @ info["normal"][2:5, 2:5] @ D
    fixed = float(np.sqrt(max(np.linalg.eigvalsh(A)[0], 0.0))) if np.isfinite(A).all() else 0.0
    return to_params(q), {"n": info["n"], "fixed_px": round(fixed, 2)}


def propose(frame: np.ndarray, prior: dict, boxes: np.ndarray | None = None,
            debug: dict | None = None, geom: pitch.Geometry = pitch.DEFAULT,
            enhance: bool = False) -> tuple[np.ndarray, dict] | None:
    """Zoek de kalibratie (veld -> beeld, volle resolutie) bij een bekende camerapositie.

    prior: camera_prior(clip). None als er geen overtuigende plek gevonden wordt.
    enhance: ook naar zwakke lijnen zoeken (zie detect_lines)."""
    from .calibration import camera_homography

    full = (frame.shape[1], frame.shape[0])
    s = WORK_WIDTH / frame.shape[1]
    S = np.diag([s, s, 1.0])
    size = (WORK_WIDTH, int(round(frame.shape[0] * s)))
    lines = detect_lines(frame, boxes, enhance=enhance)
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

    from .calibration import camera_projection
    goal_dt = goal_evidence(cv2.resize(frame, size, interpolation=cv2.INTER_AREA))

    def goal(p):  # winst van de doelen (palen + lat) voor één camerastand
        return float(_goal_gain((S @ camera_projection(p, full))[None], size, goal_dt, geom)[0])

    rng = np.random.default_rng(0)
    det_sub = det_xy[rng.choice(len(det_xy), min(400, len(det_xy)), replace=False)].astype(float)
    det_c = det_sub[:200]

    # 1. grof: vanaf elke plek binnen de onzekerheid alle kijkrichtingen naar het veld en een paar
    #    zoomstanden. De plek doet er veel toe: 2 m naast de echte plek klopt de zijlijn vlak voor je al niet meer.
    log_fs = [math.log(prior["f"]) + d for d in (-0.25, 0.0, 0.3)]  # iets uitgezoomd .. ingezoomd
    # laag langs de lijn (telefoon op 1 à 1,5 m) kijk je naar de overkant vaak net iets omhoog
    tilts = np.radians([-4, -2.5, -1, 0.5, 2, 3.5, 5, 6.5, 8, 10, 12, 14, 16.5, 19, 22, 26, 30, 35])
    pstep = max(1.0, sp / 4)
    offs = np.arange(-1.5 * sp, 1.5 * sp + 1e-6, pstep)
    L, Wp = geom.length, geom.width
    samples, tangents = model(geom).samples, model(geom).tangents
    corners = np.array([[0, 0], [L, 0], [L, Wp], [0, Wp], [L / 2, Wp / 2]])
    pool = []
    for dx in offs:
        for dy in offs:
            px, py = cx + dx, cy + dy
            ang = np.degrees(np.arctan2(corners[:, 1] - py, corners[:, 0] - px))
            if prior.get("yaw") is not None:  # zelf aangegeven kijkrichting: alleen daaromheen zoeken
                yaws = math.degrees(prior["yaw"]) + np.arange(-35, 35.1, 2.0)
            elif 0 <= px <= L and 0 <= py <= Wp:
                yaws = np.arange(0, 360, 2.0)
            else:  # kijkrichtingen tussen de uiterste hoekpunten (plus wat marge)
                rel = (ang - ang[4] + 180) % 360 - 180
                yaws = ang[4] + np.arange(rel.min() - 25, rel.max() + 25.1, 2.0)
            Y, T, F = np.meshgrid(np.radians(yaws), tilts, log_fs, indexing="ij")
            P = np.stack([np.full(Y.size, px), np.full(Y.size, py), np.full(Y.size, ch), Y.ravel(), T.ravel(),
                          np.zeros(Y.size), F.ravel()], 1)
            sc = _coarse_scores(P, full, s, dt, size, det_c, geom=geom, goal_dt=goal_dt)
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
        if (abs(q[0]) > 2 * sp or abs(q[1]) > 2 * sp or not math.radians(-8) < q[3] < math.radians(60)
                or abs(q[4]) > math.radians(8) or not lf_lo < q[5] < lf_hi):
            return 1e6
        p = np.array([cx + q[0], cy + q[1], ch, *q[2:]])
        Hw = H_work(p)
        g, n, _ = _view_gain(Hw, dt, size, tau, samples, tangents)
        g += goal(p)
        e = float(_batch_explained(Hw[None], det_sub, tau_m=tau / 16.0, geom=geom, cam=p[:2])[0])
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
        gain, n, hit = _view_gain(Hw, dt, size, 3.0, samples, tangents)
        gg = goal(p_new)
        prec = _precision(Hw, det_xy, size, geom=geom, cam=p_new[:2])
        results.append({"p": p_new, "hit": hit, "precision": prec, "n": n, "gain": gain, "goal": round(gg, 1),
                        "score": (gain + gg) * prec ** 6})
    best = max(results, key=lambda r: r["score"])
    # laatste precieze bijstelling op de lijnpixels; alleen houden als het beter past
    p_pol, pol = _polish(best["p"], det_xy, S, full, size, geom, sp, np.array([cx, cy]), lf0)
    Hw = H_work(p_pol)
    gain, n, hit = _view_gain(Hw, dt, size, 3.0, samples, tangents)
    prec = _precision(Hw, det_xy, size, geom=geom, cam=p_pol[:2])
    gg = goal(p_pol)
    if debug is not None:
        debug["polish"] = {"before": [round(float(v), 4) for v in best["p"]], "after": [round(float(v), 4) for v in p_pol],
                           "score_before": round(best["score"], 2), "score_after": round((gain + gg) * prec ** 6, 2)}
    if (gain + gg) * prec ** 6 >= 0.95 * best["score"]:  # de score is grof; kleinste kwadraten is preciezer
        best = {**best, "p": p_pol, "hit": hit, "precision": prec, "n": n, "gain": gain, "goal": round(gg, 1),
                "score": (gain + gg) * prec ** 6, "polished": True}
    best = {**best, "lines_n": pol["n"], "fixed_px": pol["fixed_px"]}
    # Twijfel: een wezenlijk andere camerastand past bijna even goed (bijv. alleen de zijlijn in beeld)
    Hb = camera_homography(best["p"], full)
    xs, ys = np.meshgrid(np.linspace(0, L, 15), np.linspace(0, Wp, 9))
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
            "goal": best["goal"], "lines_n": best.get("lines_n"), "fixed_px": best.get("fixed_px"),
            "camera": {"x": round(float(best["p"][0]), 1), "y": round(float(best["p"][1]), 1),
                       "yaw_deg": round(math.degrees(best["p"][3]) % 360, 1), "tilt_deg": round(math.degrees(best["p"][4]), 1),
                       "roll_deg": round(math.degrees(best["p"][5]), 1),
                       "hfov_deg": round(math.degrees(2 * math.atan(full[0] / 2 / math.exp(best["p"][6]))), 1)}}
    if debug is not None:
        debug["H"] = camera_homography(best["p"], full)
        debug.update(info, candidates=[{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k != "p"}
                                       for r in results])
    # Goed genoeg: (a) de meeste modellijnen in beeld vallen op gevonden lijnen en andersom, of
    # (b) heel veel lijnpixels liggen precies op het model en die lijnen leggen de stand vast
    # (niet alleen een zijlijn waarlangs je ongemerkt kunt schuiven). (b) helpt bij tegenlicht: dan
    # zijn de verre lijnen weg, maar de lijnen dichtbij zijn scherp.
    by_cover = best["hit"] >= 0.7 and best["precision"] >= 0.8 and best["hit"] * best["n"] >= 30
    by_lines = ((best.get("lines_n") or 0) >= 250 and best["precision"] >= 0.85
                and (best.get("fixed_px") or 0.0) >= MIN_FIXED_PX)
    info["accepted_by"] = "dekking" if by_cover else ("lijnen" if by_lines else None)
    if not (by_cover or by_lines):
        return None
    return camera_homography(best["p"], full), info
