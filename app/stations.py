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
    """De video van deze standplaats die je zelf hebt gekalibreerd (met de camera die eruit volgt)."""
    for c in station["clips"]:
        if c["status"] == "klaar" or c.get("width"):
            p = clip_camera(store, c)
            if p is not None:
                return c, p
    return None


# --- 3. omgeving herkennen -------------------------------------------------------------------

class _Ref:
    def __init__(self, clip: dict, t: float, K: np.ndarray, frame: np.ndarray):
        self.clip, self.t, self.K = clip, t, K
        self.s = SMALL_W / frame.shape[1]
        gray = cv2.cvtColor(cv2.resize(frame, None, fx=self.s, fy=self.s, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        self.kp, self.des = _sift().detectAndCompute(gray, None)


_SIFT = None


def _sift():
    global _SIFT
    if _SIFT is None:
        _SIFT = cv2.SIFT_create(nfeatures=3000)
    return _SIFT


def references(store: Store, station: dict, limit: int = 10) -> list[_Ref]:
    """Gekalibreerde beelden van deze standplaats: jouw eigen sleutelframes en goedgekeurde voorstellen."""
    from .calibration import fit_keyframe
    from .pipeline import read_frame
    cands = []
    for c in station["clips"]:
        own = _manual_keyframes(store, c["id"])
        approved = [k for k in store.keyframes(c["id"]) if k.get("source") == "standplaats"
                    and c.get("calib_review") == "goedgekeurd"]
        for kf in own + approved:
            cands.append((c, kf))
    if len(cands) > limit:
        cands = [cands[i] for i in np.linspace(0, len(cands) - 1, limit).astype(int)]
    refs = []
    for c, kf in cands:
        try:
            K, _ = fit_keyframe(kf, camera=camera_prior(c))
            frame, ft = read_frame(Path(c["path"]), kf["t"])
        except (ValueError, OSError, np.linalg.LinAlgError):
            continue
        r = _Ref(c, ft, K, frame)
        if r.des is not None and len(r.kp) >= MIN_INLIERS:
            refs.append(r)
    return refs


def match_environment(frame: np.ndarray, refs: list[_Ref]) -> tuple[np.ndarray, dict] | None:
    """Leg dit beeld tegen de gekalibreerde beelden: (K beeld -> veld, info) of None."""
    if not refs:
        return None
    s = SMALL_W / frame.shape[1]
    gray = cv2.cvtColor(cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    kp, des = _sift().detectAndCompute(gray, None)
    if des is None or len(kp) < MIN_INLIERS:
        return None
    bf = cv2.BFMatcher(cv2.NORM_L2)
    best = None
    for r in refs:
        pairs = bf.knnMatch(des, r.des, k=2)
        good = [m for m, *rest in pairs if rest and m.distance < 0.75 * rest[0].distance]
        if len(good) < MIN_INLIERS:
            continue
        src = np.float32([kp[m.queryIdx].pt for m in good])
        dst = np.float32([r.kp[m.trainIdx].pt for m in good])
        Hs, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        if Hs is None:
            continue
        n_in = int(mask.sum())
        if n_in < MIN_INLIERS or n_in < 0.25 * len(good):
            continue
        if best is None or n_in > best[1]:
            Sd, Sr = np.diag([s, s, 1.0]), np.diag([r.s, r.s, 1.0])
            H_full = np.linalg.inv(Sr) @ Hs @ Sd  # dit beeld -> referentiebeeld (volle resolutie)
            best = (r.K @ H_full, n_in, r)
    if best is None:
        return None
    K, n_in, r = best
    return K, {"inliers": n_in, "ref_clip": r.clip["id"], "ref_t": round(r.t, 2)}


def plausible(K: np.ndarray, cam: np.ndarray, size: tuple[int, int], pts_img) -> tuple[bool, np.ndarray | None]:
    """Levert dit een camera op de standplaats op (zelfde plek en hoogte, niet gespiegeld)?"""
    K = orient_by_points(normalize_h(K), pts_img)
    if mirrored(K):
        return False, None
    r = camera_from_homography(K, size)
    if r is None or r[1] > 12.0:
        return False, None
    p = r[0]
    ok = math.hypot(p[0] - cam[0], p[1] - cam[1]) <= MAX_POS_DIFF_M and abs(p[2] - cam[2]) <= MAX_H_DIFF_M
    return ok, p


# --- 4. alles bij elkaar: een video automatisch kalibreren ---------------------------------------

def calibrate_clip(store: Store, clip: dict, cam: np.ndarray, refs: list[_Ref], geom: pitch.Geometry,
                   worker=None) -> dict:
    """Voorstel voor één video. Geeft {"ok", "method", ...}; slaat bij succes een sleutelframe op."""
    from .autocalib import keyframe_points, propose, refine
    from .pipeline import read_frame
    size = (int(clip["width"]), int(clip["height"]))
    frames = store.all("SELECT t FROM frames WHERE clip_id = ? ORDER BY idx", (clip["id"],))
    if not frames:
        return {"ok": False, "reason": "nog niet geanalyseerd"}
    ts = [frames[int(q * (len(frames) - 1))]["t"] for q in (0.5, 0.2, 0.8)]
    f_full = math.exp(cam[6])
    cam_dict = {"x": float(cam[0]), "y": float(cam[1]), "h": float(cam[2]), "roll": float(cam[5]), "log_f": float(cam[6])}
    prior = {"x": float(cam[0]), "y": float(cam[1]), "h": float(cam[2]), "f": f_full, "sigma_pos": 2.0,
             "width": size[0], "height": size[1]}
    corners = [[0.25 * size[0], 0.75 * size[1]], [0.75 * size[0], 0.75 * size[1]], [0.5 * size[0], 0.6 * size[1]]]
    found = None
    for t in ts:
        try:
            frame, ft = read_frame(Path(clip["path"]), t)
        except ValueError:
            continue
        m = match_environment(frame, refs)
        if m is not None:
            K, info = m
            ok, p = plausible(K, cam, size, corners)
            if ok:
                H = normalize_h(np.linalg.inv(K))  # veld -> beeld
                res = refine(frame, H, None, f_full, camera={**cam_dict, "q0": p[3:]}, geom=geom)
                if res is not None:
                    H = res[0]
                    info["lijnen"] = True
                found = (ft, H, "omgeving", info)
                break
        for enhance in (False, True):
            res = propose(frame, prior, None, geom=geom, enhance=enhance)
            if res is not None and not res[1].get("ambiguous"):
                found = (ft, res[0], "lijnen", {k: v for k, v in res[1].items() if k != "camera"})
                break
        if found:
            break
    if found is None:
        store.run("UPDATE clips SET calib_review = 'mislukt', calib_method = NULL WHERE id = ?", (clip["id"],))
        return {"ok": False, "reason": "omgeving en lijnen niet gevonden"}
    ft, H, method, info = found
    pts = keyframe_points(H, size, geom)
    if len(pts) < 4:
        store.run("UPDATE clips SET calib_review = 'mislukt', calib_method = NULL WHERE id = ?", (clip["id"],))
        return {"ok": False, "reason": "te weinig veld in beeld"}
    store.run("INSERT INTO keyframes (clip_id, t, points, auto, source, score) VALUES (?,?,?,0,'standplaats',?)",
              (clip["id"], float(ft), json.dumps(pts), json.dumps({"methode": method, **info})))
    store.run("UPDATE clips SET calib_review = 'voorstel', calib_method = ?, cam_x = ?, cam_y = ?, cam_h = ?, cam_f = ?, "
              "cam_source = CASE WHEN cam_source = 'hand' THEN cam_source ELSE 'standplaats' END WHERE id = ?",
              (method, float(cam[0]), float(cam[1]), float(cam[2]), f_full, clip["id"]))
    if worker is not None:
        worker.submit_autocalib(clip["id"])
    return {"ok": True, "method": method, "t": ft}


def run(store: Store, match_id: int, progress=None, worker=None) -> dict:
    """Alle video's zonder eigen kalibratie automatisch kalibreren, per standplaats."""
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
    done = ok = 0
    refs_of: dict[int, list] = {}
    summary = {"ok": 0, "mislukt": 0, "geen_anker": 0, "niet_geanalyseerd": 0}
    for st, a, c in todo:
        done += 1
        if progress:
            progress(done / max(1, len(todo)), f"{c['filename']} ({done}/{len(todo)})")
        if c["status"] != "klaar":
            summary["niet_geanalyseerd"] += 1
            continue
        if a is None:
            summary["geen_anker"] += 1
            continue
        store.run("DELETE FROM keyframes WHERE clip_id = ? AND source = 'standplaats'", (c["id"],))
        if st["id"] not in refs_of:
            refs_of[st["id"]] = references(store, st)
        try:
            r = calibrate_clip(store, c, a[1], refs_of[st["id"]], geom, worker=worker)
        except Exception:  # noqa: BLE001
            log.exception("Standplaats-kalibratie van %s", c["filename"])
            r = {"ok": False}
        summary["ok" if r.get("ok") else "mislukt"] += 1
        ok += bool(r.get("ok"))
        analytics.invalidate(clip_id=c["id"])
    return summary
