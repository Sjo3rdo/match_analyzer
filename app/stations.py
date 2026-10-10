"""Standplaatsen: één keer kalibreren per plek, de rest van de video's automatisch.

Metafoor: een fotograaf op een statief. Zolang je op dezelfde plek staat, zijn positie, hoogte en
zoom bij elke video gelijk; alleen de kijkrichting verschilt. Is die plek één keer vastgelegd (jij
kalibreert één video), dan zoekt de app voor de andere video's alleen nog de kijkrichting:

1. Omgeving herkennen: bomen, huizen, reclameborden en het hek staan vanaf dezelfde plek altijd op
   dezelfde plek. Het nieuwe beeld wordt als een puzzelstuk tegen een al gekalibreerd beeld gelegd
   (zoals een panoramafoto aan elkaar wordt gezet). Omdat de camera alleen draait, past dat precies.
2. Lukt dat niet: de veldlijnen zoeken vanaf de bekende plek (zoals "Zoek het veld automatisch").

Alles wat de app zo voorstelt, controleer jij in één overzicht.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import cv2
import numpy as np

from . import geo, pitch
from .calibration import (apply_h, camera_from_homography, camera_homography, default_focal, fit_calibration,
                          fit_camera, camera_prior, kf_camera, mirrored, normalize_h, orient_by_points)
from .storage import Store

log = logging.getLogger(__name__)

SMALL_W = 960  # werkbreedte voor het vergelijken van beelden
MIN_INLIERS = 30
MAX_POS_DIFF_M = 10.0  # een voorstel moet een camera op (bijna) de standplaats opleveren
MAX_H_DIFF_M = 2.5
REF_STEP_S = 1.0  # om de hoeveel seconden een referentiebeeld in het panorama
MAX_REFS = 160
# Verder dan dit gedraaid t.o.v. een ijkmoment: het panorama hangt dan sterk af van de exacte zoom van de
# lens (op een paar procent na); zo'n voorstel krijgt een waarschuwing.
TRUST_DEG = 45.0
MAX_MODEL_ERR_PX = 25.0  # zo ver mag de omgevings-kalibratie van een echte camera op de standplaats afwijken
QUERY_STEP_S = 2.0  # om de hoeveel seconden een te kalibreren video bekeken wordt
MAX_QUERY = 25


# --- 1. groeperen --------------------------------------------------------------------------

def group(store: Store, match_id: int) -> list[dict]:
    """Video's per standplaats, per helft (je staat vaak per helft ergens anders): GPS-positie (met de
    nauwkeurigheid die de telefoon erbij opgeeft) en opnametijd. Twee video's horen bij dezelfde plek
    als ze dichter bij elkaar liggen dan hun onnauwkeurigheid samen (minstens 6 m). Video's zonder GPS
    gaan naar de plek van de video die er qua opnametijd het dichtst bij ligt."""
    clips = store.all("SELECT * FROM clips WHERE match_id = ? ORDER BY COALESCE(rec_start, 1e18), order_idx, id",
                      (match_id,))
    periods = sorted({int(c.get("period") or 1) for c in clips})
    out = []
    for per in periods:
        mine = [c for c in clips if int(c.get("period") or 1) == per]
        groups = _group_clips(mine)
        name = f"{per}e helft" if per <= 2 else f"Verlenging {per - 2}"
        many = len([g for g in groups if not g.get("no_gps")]) > 1
        for k, g in enumerate(groups):
            members = sorted(g["clips"], key=lambda c: (c.get("rec_start") or 1e18, c["order_idx"]))
            label = (f"{name} – zonder GPS" if g.get("no_gps") else f"{name} – standplaats {k + 1}" if many else name)
            out.append({"id": len(out) + 1, "period": per, "label": label,
                        "lat": g["lat"], "lon": g["lon"], "acc_m": round(g["acc"], 1) if g["acc"] else None,
                        "precision_m": round(1 / math.sqrt(sum(g["_w"])), 1) if g.get("_w") else None,
                        "clips": members, "no_gps": bool(g.get("no_gps"))})
    return out


def _group_clips(clips: list[dict]) -> list[dict]:
    groups: list[dict] = []
    for c in clips:
        if c.get("gps_lat") is None:
            continue
        acc = max(2.0, float(c.get("gps_acc") or 10.0))
        best, best_d = None, None
        for g in groups:
            d = geo.distance_m(c["gps_lat"], c["gps_lon"], g["lat"], g["lon"])
            if d <= max(6.0, 1.5 * math.hypot(acc, g["acc"])) and (best is None or d < best_d):
                best, best_d = g, d
        if best is None:
            best = {"lat": c["gps_lat"], "lon": c["gps_lon"], "acc": acc, "_w": [], "_ll": [], "_acc": [], "clips": []}
            groups.append(best)
        w = 1.0 / acc ** 2  # precieze metingen tellen zwaarder
        best["_w"].append(w)
        best["_ll"].append((c["gps_lat"], c["gps_lon"]))
        best["_acc"].append(acc)
        W = np.array(best["_w"])
        ll = np.array(best["_ll"])
        best["lat"], best["lon"] = float((W[:, None] * ll).sum(0)[0] / W.sum()), float((W[:, None] * ll).sum(0)[1] / W.sum())
        best["acc"] = float(np.median(best["_acc"]))
        best["clips"].append(c)
    timed = [(c.get("rec_start"), g) for g in groups for c in g["clips"] if c.get("rec_start") is not None]
    loose = []
    for c in clips:
        if c.get("gps_lat") is not None:
            continue
        near = [(abs(ts - c["rec_start"]), g) for ts, g in timed] if c.get("rec_start") is not None else []
        near = [x for x in near if x[0] <= 20 * 60]
        if near:
            min(near, key=lambda x: x[0])[1]["clips"].append(c)
        elif len(groups) == 1:  # maar één plek in deze helft: daar hoort hij vast bij
            groups[0]["clips"].append(c)
        else:
            loose.append(c)
    if loose:
        groups.append({"lat": None, "lon": None, "acc": None, "clips": loose, "no_gps": True})
    return groups


def best_clip(store: Store, station: dict, max_candidates: int = 12) -> dict | None:
    """De video van deze standplaats met het meeste veld (veldlijnen) in beeld: die is het makkelijkst
    om zelf te kalibreren. De score wordt per video bewaard."""
    from .autocalib import detect_lines
    from .pipeline import read_frame
    cands = [c for c in station["clips"] if c.get("path") and Path(c["path"]).exists()]
    if not cands:
        return None
    if len(cands) > max_candidates:
        cands = [cands[i] for i in np.linspace(0, len(cands) - 1, max_candidates).astype(int)]
    best, best_s = None, -1.0
    for c in cands:
        score = c.get("line_score")
        if score is None:
            try:
                frame, _ = read_frame(Path(c["path"]), float(c.get("duration") or 2) / 2)
                score = float((detect_lines(frame) > 0).sum())
            except (ValueError, OSError):
                score = 0.0
            store.run("UPDATE clips SET line_score = ? WHERE id = ?", (score, c["id"]))
        if score > best_s:
            best, best_s = c, score
    return best


# --- 2. de camera van de standplaats -----------------------------------------------------------

def _manual_keyframes(store: Store, clip_id: int) -> list[dict]:
    return [k for k in store.keyframes(clip_id) if not k.get("auto") and k.get("source") != "standplaats"]


def clip_camera(store: Store, clip: dict) -> np.ndarray | None:
    """Camera [x, y, h, yaw, tilt, roll, log_f] uit de eigen (handmatige) kalibratie van een video."""
    size = (int(clip["width"] or 1920), int(clip["height"] or 1080))
    prior = camera_prior(clip)
    for kf in _manual_keyframes(store, clip["id"]):
        try:
            if prior is not None:
                return np.array(fit_camera(kf["points"], kf_camera(prior, kf))[1]["params"])
            K, _ = fit_calibration(kf["points"])
        except (ValueError, np.linalg.LinAlgError):
            continue
        r = camera_from_homography(K, size)
        if r is not None and r[1] < 8.0:
            return r[0]
    return None


def anchor(store: Store, station: dict) -> tuple[dict, np.ndarray] | None:
    """De video van deze standplaats die je zelf hebt gekalibreerd (met de camera die eruit volgt).
    Zijn er meer, dan de best vastgelegde (meeste punten): daar liggen plek, hoogte en zoom het
    zekerst vast. Twee doelpalen alleen laten bijv. de zoom vrij wankel."""
    from .calibration import calibration_dof
    best = None
    for c in station["clips"]:
        if c["status"] == "klaar" or c.get("width"):
            kfs = _manual_keyframes(store, c["id"])
            if not kfs:
                continue
            p = clip_camera(store, c)
            if p is None:
                continue
            dof = max(calibration_dof(k["points"], elevated=True) for k in kfs)
            if best is None or dof > best[0]:
                best = (dof, c, p)
    return None if best is None else (best[1], best[2])


# --- 3. omgeving herkennen -------------------------------------------------------------------

class _Ref:
    def __init__(self, clip: dict, t: float, K: np.ndarray, frame: np.ndarray):
        self.clip, self.t, self.K = clip, t, K
        self.hops, self.turn = 0, 0.0  # afstand tot het ijkmoment waar de kalibratie vandaan komt
        self.s = SMALL_W / frame.shape[1]
        gray = cv2.cvtColor(cv2.resize(frame, None, fx=self.s, fy=self.s, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        self.kp, self.des = _sift().detectAndCompute(gray, None)


_SIFT = None


def _sift():
    global _SIFT
    if _SIFT is None:
        _SIFT = cv2.SIFT_create(nfeatures=3000)
    return _SIFT


def _sample_frames(path: Path, times: list[float]):
    """Beelden op (ongeveer) de gevraagde tijden, in één keer vooruit lezen (sneller dan steeds springen)."""
    times = sorted(times)
    if not times:
        return
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, times[0] - 0.5) * 1000)
    k = 0
    while k < len(times) and cap.grab():
        ft = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        if ft + 1e-3 >= times[k]:
            ok, frame = cap.retrieve()
            if ok:
                yield ft, frame
            while k < len(times) and times[k] <= ft + 1e-3:
                k += 1
    cap.release()


def _pair(a: "_Ref | tuple", b: "_Ref") -> tuple[np.ndarray, int] | None:
    """Homografie beeld a -> beeld b (volle resolutie) via de omgeving, met het aantal inliers."""
    kp_a, des_a, s_a = (a.kp, a.des, a.s) if isinstance(a, _Ref) else a
    if des_a is None or b.des is None or len(kp_a) < MIN_INLIERS or len(b.kp) < MIN_INLIERS:
        return None
    pairs = _matcher().knnMatch(des_a, b.des, k=2)
    good = [m for m, *rest in pairs if rest and m.distance < 0.75 * rest[0].distance]
    if len(good) < MIN_INLIERS:
        return None
    src = np.float32([kp_a[m.queryIdx].pt for m in good])
    dst = np.float32([b.kp[m.trainIdx].pt for m in good])
    Hs, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if Hs is None:
        return None
    n_in = int(mask.sum())
    if n_in < MIN_INLIERS or n_in < 0.25 * len(good):
        return None
    Sa, Sb = np.diag([s_a, s_a, 1.0]), np.diag([b.s, b.s, 1.0])
    return np.linalg.inv(Sb) @ Hs @ Sa, n_in


_MATCHER = None


def _matcher():
    global _MATCHER
    if _MATCHER is None:
        _MATCHER = cv2.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=48))
    return _MATCHER


def chain(clip: dict, t0: float, K0: np.ndarray, step: float = REF_STEP_S) -> list[_Ref]:
    """Panorama van één video: vanaf een gekalibreerd moment (t0, K0) elke `step` seconden een
    referentiebeeld, door de hele video heen. Elk beeld wordt tegen zijn buurman gelegd (de omgeving
    draait mee), zo krijgt elk beeld zijn eigen kalibratie. Metafoor: een rij dominostenen; vanaf het
    ijkmoment geeft elke steen zijn kalibratie door aan de volgende. Breekt de rij (te weinig
    overeenkomst, bijv. bij een snelle zwaai), dan stopt die kant."""
    dur = float(clip.get("duration") or 0)
    times = sorted({round(float(t), 3) for t in np.arange(t0 % step, dur, step)} | {round(t0, 3)})
    refs = []
    for ft, frame in _sample_frames(Path(clip["path"]), times):
        refs.append(_Ref(clip, ft, None, frame))
    if not refs:
        return []
    i0 = int(np.argmin([abs(r.t - t0) for r in refs]))
    refs[i0].K, refs[i0].hops = K0, 0
    for direction in (1, -1):
        i = i0
        while 0 <= i + direction < len(refs):
            nxt = refs[i + direction]
            m = _pair(nxt, refs[i])
            if m is None or m[1] < 2 * MIN_INLIERS:
                break
            nxt.K = refs[i].K @ m[0]
            # hoe ver van het ijkmoment: aantal schakels en hoeveel er gedraaid is (in beeldbreedtes)
            nxt.hops = refs[i].hops + 1
            i += direction
    y0 = _yaw(K0, clip)
    out = [r for r in refs if r.K is not None and r.des is not None and len(r.kp) >= MIN_INLIERS]
    for r in out:  # netto gedraaid t.o.v. het ijkmoment (graden): hoe verder, hoe onzekerder
        y = _yaw(r.K, clip)
        r.turn = 0.0 if y is None or y0 is None else abs((y - y0 + 180) % 360 - 180)
    return out


def _yaw(K: np.ndarray, clip: dict) -> float | None:
    """Kijkrichting (graden op de tekening) van kalibratie K: richting van het midden-onder van het beeld."""
    w, hh = float(clip.get("width") or 1920), float(clip.get("height") or 1080)
    a, b = apply_h(K, np.array([[w / 2, 0.98 * hh], [w / 2, 0.85 * hh]]))
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))) or np.linalg.norm(b - a) < 1e-6:
        return None
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))


def references(store: Store, station: dict, limit: int = MAX_REFS) -> list[_Ref]:
    """Gekalibreerde beelden van deze standplaats: rond jouw eigen ijkmomenten en goedgekeurde
    voorstellen een heel panorama (zie `chain`)."""
    from .calibration import fit_keyframe
    out = []
    for c in station["clips"]:
        own = _manual_keyframes(store, c["id"])
        approved = [k for k in store.keyframes(c["id"]) if k.get("source") == "standplaats"
                    and c.get("calib_review") == "goedgekeurd"]
        for kf in (own + approved)[:3]:
            try:
                K, _ = fit_keyframe(kf, camera=camera_prior(c))
            except (ValueError, np.linalg.LinAlgError):
                continue
            out += chain(c, float(kf["t"]), K)
    return _thin(_relay(_shortest(out)), limit)


def _relay(refs: list[_Ref], min_gain: float = 10.0) -> list[_Ref]:
    """Neem de kalibratie van een referentiebeeld over via een dichterbij liggend ijkmoment uit een
    andere video, als die er is. Metafoor: een routeplanner die niet per se via je eigen straat
    hoeft. Zo hangen bijv. de late beelden van een lange zwaai niet meer aan een lange ketting als je
    elders zelf een video kalibreerde die die kant op kijkt."""
    far = sorted((r for r in refs if r.turn > TRUST_DEG / 2), key=lambda r: -r.turn)
    for r in far:
        best = None
        for q in refs:
            if q.clip["id"] == r.clip["id"] or q.turn > r.turn - min_gain:
                continue
            m = _pair(r, q)  # beeld r -> beeld q
            if m is None or m[1] < 2 * MIN_INLIERS:
                continue
            K = q.K @ m[0]
            yq, yr = _yaw(q.K, q.clip), _yaw(K, r.clip)
            turn = q.turn + (0.0 if yq is None or yr is None else abs((yr - yq + 180) % 360 - 180))
            if turn < r.turn - min_gain and (best is None or turn < best[0]):
                best = (turn, K)
        if best is not None:
            r.turn, r.K = best
    return refs


def _shortest(refs: list[_Ref]) -> list[_Ref]:
    """Hetzelfde beeld via meerdere ijkmomenten: houd de kortste ketting (die is het nauwkeurigst)."""
    best: dict[tuple, _Ref] = {}
    for r in refs:
        key = (r.clip["id"], round(r.t, 1))
        if key not in best or r.turn < best[key].turn:
            best[key] = r
    return sorted(best.values(), key=lambda r: (r.clip["id"], r.t))


def _thin(refs: list[_Ref], limit: int) -> list[_Ref]:
    if len(refs) <= limit:
        return refs
    return [refs[i] for i in np.linspace(0, len(refs) - 1, limit).astype(int)]


def match_environment(frame: np.ndarray, refs: list[_Ref]) -> tuple[np.ndarray, dict] | None:
    """Leg dit beeld tegen de gekalibreerde beelden: (K beeld -> veld, info) of None."""
    if not refs:
        return None
    s = SMALL_W / frame.shape[1]
    gray = cv2.cvtColor(cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    kp, des = _sift().detectAndCompute(gray, None)
    if des is None or len(kp) < MIN_INLIERS:
        return None
    found = []
    for r in refs:
        m = _pair((kp, des, s), r)
        if m is not None:
            found.append((r.K @ m[0], m[1], r))
    if not found:
        return None
    top = max(f[1] for f in found)
    # bijna even goede overeenkomst: kies het referentiebeeld dat het dichtst bij een ijkmoment ligt
    K, n_in, r = min((f for f in found if f[1] >= 0.6 * top), key=lambda f: (f[2].turn, -f[1]))
    return K, {"inliers": n_in, "ref_clip": r.clip["id"], "ref_t": round(r.t, 2), "zwaai": round(r.turn, 1),
               "onzeker": bool(r.turn > TRUST_DEG)}


def plausible(K: np.ndarray, cam: np.ndarray, size: tuple[int, int], pts_img) -> tuple[bool, np.ndarray | None]:
    """Past dit bij een camera op de standplaats (niet gespiegeld, zelfde plek en hoogte)?

    De plek is bekend; we zoeken alleen kijkrichting en zoom die het best bij K passen en kijken hoe
    goed dat past. (Vrij terugrekenen van de plek is bij een lage camera te wankel: dan werd een goed
    voorstel weggegooid.) Geeft (ok, camera [x, y, h, yaw, tilt, roll, log_f])."""
    from .autocalib import camera_from_h
    K = orient_by_points(normalize_h(K), pts_img)
    if mirrored(K):
        return False, None
    H = normalize_h(np.linalg.inv(K))  # veld -> beeld
    W, Hh = size
    cam_d = {"x": float(cam[0]), "y": float(cam[1]), "h": float(cam[2])}
    # beginrichting: waar de onderkant van het beeld het veld raakt
    g = apply_h(K, np.array([[W / 2, 0.92 * Hh]]))[0]
    yaw0 = math.atan2(g[1] - cam[1], g[0] - cam[0]) if np.all(np.isfinite(g)) else float(cam[3])
    best = None
    for dy in (0.0, -0.35, 0.35):
        q = camera_from_h(H, cam_d, size, np.array([yaw0 + dy, 0.0, 0.0, cam[6]]))
        err = _model_err(H, np.array([cam[0], cam[1], cam[2], *q]), size)
        if best is None or err < best[0]:
            best = (err, q)
    err, q = best
    p = np.array([cam[0], cam[1], cam[2], *q])
    zoom = math.exp(q[3] - cam[6])
    return bool(err <= MAX_MODEL_ERR_PX and 0.4 <= zoom <= 6.0 and abs(q[2]) < math.radians(15)), p


def _model_err(H: np.ndarray, p: np.ndarray, size: tuple[int, int]) -> float:
    """Gemiddelde afstand (px) tussen H en de camera p, over het veld dat in beeld is."""
    xs, ys = np.meshgrid(np.linspace(-5, 110, 24), np.linspace(-5, 73, 17))
    world = np.stack([xs.ravel(), ys.ravel()], 1)
    a = np.hstack([world, np.ones((len(world), 1))]) @ H.T
    b = np.hstack([world, np.ones((len(world), 1))]) @ camera_homography(p, size).T
    ok = (a[:, 2] > 1e-6) & (b[:, 2] > 1e-6)
    a, b = a[ok, :2] / a[ok, 2:], b[ok, :2] / b[ok, 2:]
    W, Hh = size
    vis = (a[:, 0] >= 0) & (a[:, 0] <= W) & (a[:, 1] >= 0) & (a[:, 1] <= Hh)
    if vis.sum() < 6:
        return float("inf")
    return float(np.mean(np.linalg.norm(a[vis] - b[vis], axis=1)))


# --- 4. alles bij elkaar: een video automatisch kalibreren ---------------------------------------

def calibrate_clip(store: Store, clip: dict, cam: np.ndarray, refs: list[_Ref], geom: pitch.Geometry,
                   worker=None, lines: bool = True) -> dict:
    """Voorstel voor één video. Geeft {"ok", "method", "t", "H"}; slaat bij succes een sleutelframe op.

    Eerst de omgeving: elke paar seconden een beeld van deze video tegen het panorama leggen; het
    beeld met de meeste overeenkomst wint. Lukt dat niet en `lines`: de veldlijnen zoeken."""
    from .autocalib import keyframe_points, propose, refine
    size = (int(clip["width"]), int(clip["height"]))
    frames = store.all("SELECT t FROM frames WHERE clip_id = ? ORDER BY idx", (clip["id"],))
    if not frames:
        return {"ok": False, "reason": "nog niet geanalyseerd"}
    ft_all = np.array([f["t"] for f in frames])
    n_q = int(min(MAX_QUERY, max(3, (ft_all[-1] - ft_all[0]) // QUERY_STEP_S + 1)))
    want = [float(ft_all[int(round(q * (len(ft_all) - 1)))]) for q in np.linspace(0.05, 0.95, n_q)]
    f_full = math.exp(cam[6])
    cam_dict = {"x": float(cam[0]), "y": float(cam[1]), "h": float(cam[2]), "roll": float(cam[5]), "log_f": float(cam[6])}
    corners = [[0.25 * size[0], 0.75 * size[1]], [0.75 * size[0], 0.75 * size[1]], [0.5 * size[0], 0.6 * size[1]]]
    found = None
    cands = []
    shots = {}
    for ft, frame in _sample_frames(Path(clip["path"]), want):
        shots[ft] = frame
        m = match_environment(frame, refs)
        if m is not None:
            cands.append((m[1]["inliers"], ft, m))
    # elk moment dat klopt wordt een ijkmoment: dan hoeft de gemeten camerabeweging nooit ver te
    # overbruggen (bij zwenken raakt die anders het spoor kwijt)
    extra = []
    for n_in, ft, (K, info) in sorted(cands, key=lambda c: -c[0]):
        ok, p = plausible(K, cam, size, corners)
        if not ok:
            continue
        H = normalize_h(np.linalg.inv(K))  # veld -> beeld
        res = refine(shots[ft], H, None, f_full, camera={**cam_dict, "q0": p[3:]}, geom=geom)
        if res is not None:
            H = res[0]
            info = {**info, "lijnen": True}
        if found is None:
            found = (ft, H, "omgeving", info)
        else:
            extra.append((ft, H, info))
    if found is None and lines:
        from .pipeline import read_frame
        for q in (0.5, 0.2, 0.8):
            try:
                frame, ft = read_frame(Path(clip["path"]), float(ft_all[int(q * (len(ft_all) - 1))]))
            except ValueError:
                continue
            prior = {"x": float(cam[0]), "y": float(cam[1]), "h": float(cam[2]), "f": f_full, "sigma_pos": 2.0,
                     "width": size[0], "height": size[1]}
            for enhance in (False, True):
                res = propose(frame, prior, None, geom=geom, enhance=enhance)
                if res is not None and not res[1].get("ambiguous"):
                    found = (ft, res[0], "lijnen", {k: v for k, v in res[1].items() if k != "camera"})
                    break
            if found:
                break
    if found is None:
        if lines:
            store.run("UPDATE clips SET calib_review = 'mislukt', calib_method = NULL WHERE id = ?", (clip["id"],))
        return {"ok": False, "reason": "omgeving en lijnen niet gevonden"}
    ft, H, method, info = found
    pts = keyframe_points(H, size, geom)
    if len(pts) < 4:
        if lines:
            store.run("UPDATE clips SET calib_review = 'mislukt', calib_method = NULL WHERE id = ?", (clip["id"],))
        return {"ok": False, "reason": "te weinig veld in beeld"}
    store.run("INSERT INTO keyframes (clip_id, t, points, auto, source, score) VALUES (?,?,?,0,'standplaats',?)",
              (clip["id"], float(ft), json.dumps(pts), json.dumps({"methode": method, **info})))
    if method == "omgeving":
        _add_moments(store, clip, extra, size, geom)
    store.run("UPDATE clips SET calib_review = 'voorstel', calib_method = ?, cam_x = ?, cam_y = ?, cam_h = ?, cam_f = ?, "
              "cam_source = CASE WHEN cam_source = 'hand' THEN cam_source ELSE 'standplaats' END WHERE id = ?",
              (method, float(cam[0]), float(cam[1]), float(cam[2]), f_full, clip["id"]))
    if worker is not None:
        worker.submit_autocalib(clip["id"])
    return {"ok": True, "method": method, "t": ft, "H": H, "turn": float(info.get("zwaai", 0.0))}


def _add_moments(store: Store, clip: dict, moments, size, geom: pitch.Geometry) -> int:
    """Extra ijkmomenten (t, H veld -> beeld, info) via de omgeving, als standplaats-voorstel."""
    from .autocalib import keyframe_points
    n = 0
    for ft, H, info in moments:
        pts = keyframe_points(H, size, geom)
        if len(pts) >= 4:
            store.run("INSERT INTO keyframes (clip_id, t, points, auto, source, score) VALUES (?,?,?,0,'standplaats',?)",
                      (clip["id"], float(ft), json.dumps(pts), json.dumps({"methode": "omgeving", **info})))
            n += 1
    return n


def anchor_moments(store: Store, a: tuple[dict, np.ndarray], refs: list[_Ref], geom: pitch.Geometry) -> int:
    """Ook de zelf gekalibreerde video krijgt elke paar seconden een ijkmoment uit het panorama, zodat
    zwenken daar net zo goed gevolgd wordt."""
    clip = a[0]
    store.run("DELETE FROM keyframes WHERE clip_id = ? AND source = 'standplaats'", (clip["id"],))
    own = [float(k["t"]) for k in _manual_keyframes(store, clip["id"])]
    size = (int(clip["width"]), int(clip["height"]))
    moments, last = [], -1e9
    for r in sorted((r for r in refs if r.clip["id"] == clip["id"]), key=lambda r: r.t):
        if r.t - last < QUERY_STEP_S or min((abs(r.t - t) for t in own), default=1e9) < QUERY_STEP_S / 2:
            continue
        moments.append((r.t, normalize_h(np.linalg.inv(r.K)), {"ref_clip": clip["id"], "ref_t": round(r.t, 2)}))
        last = r.t
    return _add_moments(store, clip, moments, size, geom)


def run(store: Store, match_id: int, progress=None, worker=None) -> dict:
    """Alle video's zonder eigen kalibratie automatisch kalibreren, per standplaats.

    In rondes: elke gelukte video wordt zelf een stuk panorama, zodat video's die niets met het
    ijkmoment gemeen hebben het via een andere video alsnog vinden. Pas als de omgeving niets meer
    oplevert, zoekt de app (trager) naar veldlijnen."""
    from . import analytics
    match = store.one("SELECT * FROM matches WHERE id = ?", (match_id,))
    geom = pitch.of_match(match)
    stations = group(store, match_id)
    todo = []
    for st in stations:
        a = anchor(store, st)
        for c in st["clips"]:
            if _manual_keyframes(store, c["id"]) or c.get("calib_review") == "goedgekeurd":
                continue
            todo.append((st, a, c))
    summary = {"ok": 0, "mislukt": 0, "geen_anker": 0, "niet_geanalyseerd": 0}
    pending = []
    for st, a, c in todo:
        if c["status"] != "klaar":
            summary["niet_geanalyseerd"] += 1
        elif a is None:
            summary["geen_anker"] += 1
        else:
            store.run("DELETE FROM keyframes WHERE clip_id = ? AND source = 'standplaats'", (c["id"],))
            pending.append((st, a, c))
    refs_of: dict[int, list] = {}
    steps, total = 0, max(1, 2 * len(pending))

    def attempt(st, a, c, lines):
        nonlocal steps
        steps += 1
        if progress:
            progress(min(0.99, steps / total), f"{c['filename']} ({'lijnen' if lines else 'omgeving'})")
        if st["id"] not in refs_of:
            refs_of[st["id"]] = references(store, st)
        try:
            r = calibrate_clip(store, c, a[1], refs_of[st["id"]], geom, worker=worker, lines=lines)
        except Exception:  # noqa: BLE001
            log.exception("Standplaats-kalibratie van %s", c["filename"])
            r = {"ok": False}
        if r.get("ok"):
            summary["ok"] += 1
            analytics.invalidate(clip_id=c["id"])
            # deze video doet voortaan mee als stuk panorama
            new = chain(c, r["t"], np.linalg.inv(r["H"]))
            for x in new:  # de ketting erft de afstand van het voorstel zelf
                x.turn += r.get("turn", 0.0)
            refs_of[st["id"]] = _thin(_shortest(refs_of[st["id"]] + new), MAX_REFS)
        return r.get("ok")

    for st in stations:  # de zelf gekalibreerde video's: ook elke paar seconden een ijkmoment
        a = anchor(store, st)
        if a is not None and any(item[0]["id"] == st["id"] for item in pending):
            refs_of[st["id"]] = references(store, st)
            for c in st["clips"]:
                if not _manual_keyframes(store, c["id"]):
                    continue
                if anchor_moments(store, (c, a[1]), refs_of[st["id"]], geom) and worker is not None:
                    worker.submit_autocalib(c["id"])
                analytics.invalidate(clip_id=c["id"])
    progress_made = True
    while pending and progress_made:  # rondes via de omgeving
        progress_made = False
        for item in list(pending):
            if attempt(*item, lines=False):
                pending.remove(item)
                progress_made = True
    for item in pending:  # wat overblijft: veldlijnen zoeken
        if not attempt(*item, lines=True):
            summary["mislukt"] += 1
            analytics.invalidate(clip_id=item[2]["id"])
    return summary
