"""Webserver: REST-API + de interface (static/). Start met `python -m app` of ./run.sh."""
from __future__ import annotations

import json
import logging
import mimetypes
import shutil
import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import Body, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import analytics, clips as clip_export, config, geo, pitch, render
from .calibration import camera_prior, fit_calibration, fit_camera, normalize_h
from .pipeline import Worker, probe, read_frame
from .storage import Store, clip_dir

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
config.ensure_dirs()

store = Store()
worker: Worker | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global worker
    worker = Worker(store)
    # Clips die bij afsluiten nog bezig waren, opnieuw in de wachtrij zetten
    for c in store.all("SELECT id FROM clips WHERE status IN ('wachtrij', 'preview', 'analyse')"):
        worker.submit(c["id"])
    for c in store.all("SELECT id FROM clips WHERE calib_status IN ('wachtrij', 'bezig') AND status = 'klaar'"):
        worker.submit_autocalib(c["id"])
    # Video's van een oudere versie: teams opnieuw indelen met de verbeterde methode
    # (handmatige correcties blijven staan; opnieuw analyseren is niet nodig)
    for c in store.all("SELECT id FROM clips WHERE status = 'klaar' AND COALESCE(analysis_version, 0) = 2"):
        worker.submit_teams(c["id"])
    yield


app = FastAPI(title="Match Analyzer", lifespan=lifespan)


def _get(table: str, id_: int) -> dict:
    row = store.one(f"SELECT * FROM {table} WHERE id = ?", (id_,))
    if row is None:
        raise HTTPException(404, "Niet gevonden")
    return row


def _update(table: str, id_: int, data: dict, allowed: set[str], geometry: set[str] = frozenset()) -> dict:
    """Velden bijwerken. Velden in `geometry` raken de projectie naar het veld (zware herberekening),
    de rest alleen namen/koppelingen (licht)."""
    fields = {k: v for k, v in data.items() if k in allowed}
    if fields:
        sets = ", ".join(f"{k} = ?" for k in fields)
        store.run(f"UPDATE {table} SET {sets} WHERE id = ?", (*fields.values(), id_))
        analytics.invalidate(geometry=bool(set(fields) & set(geometry)))
    return _get(table, id_)


@app.get("/api/version")
def get_version():
    """Versie en git-commit, zodat je kunt zien of je de nieuwste versie draait."""
    from . import __version__

    commit = None
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).resolve().parent.parent,
                                capture_output=True, text=True, timeout=3).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        pass
    return {"version": __version__, "commit": commit}


# --- veld --------------------------------------------------------------------------------

@app.get("/api/pitch")
def get_pitch():
    return {"length": pitch.LENGTH, "width": pitch.WIDTH,
            "landmarks": [{"name": k, "x": v[0], "y": v[1]} for k, v in pitch.LANDMARKS.items()],
            "lines": [{"name": k, "from": list(a), "to": list(b)} for k, (a, b) in pitch.LINES.items()]}


# --- wedstrijden -------------------------------------------------------------------------

@app.get("/api/matches")
def list_matches():
    return store.all("SELECT m.*, (SELECT COUNT(*) FROM clips c WHERE c.match_id = m.id) AS n_clips "
                     "FROM matches m ORDER BY COALESCE(date, created_at) DESC")


@app.post("/api/matches")
def create_match(data: dict = Body(...)):
    if not data.get("name"):
        raise HTTPException(400, "Naam is verplicht")
    mid = store.run("INSERT INTO matches (name, date, team0_name, team1_name) VALUES (?,?,?,?)",
                    (data["name"], data.get("date"), data.get("team0_name") or "Thuis",
                     data.get("team1_name") or "Uit"))
    return _get("matches", mid)


@app.get("/api/matches/{match_id}")
def get_match(match_id: int):
    m = _get("matches", match_id)
    m["clips"] = store.all("SELECT * FROM clips WHERE match_id = ? ORDER BY order_idx, id", (match_id,))
    for c in m["clips"]:
        kfs = store.keyframes(c["id"])
        c["n_keyframes"] = len(kfs)
        c["n_manual_keyframes"] = sum(1 for k in kfs if not k.get("auto"))
    m["team_colors"] = _team_colors(match_id)
    m["n_assigned"] = store.one("SELECT COUNT(*) AS n FROM tracks t JOIN clips c ON c.id = t.clip_id "
                                "WHERE c.match_id = ? AND t.player_id IS NOT NULL", (match_id,))["n"]
    m["players"] = store.all("SELECT * FROM players WHERE match_id = ? ORDER BY team, "
                             "CAST(number AS INTEGER), name", (match_id,))
    return m


def _team_colors(match_id: int) -> list[str | None]:
    """Gemiddelde shirtkleur per team (voor de kleuren in de interface)."""
    out: list[str | None] = []
    for team in (0, 1):
        rows = store.all("SELECT t.color, t.n_frames FROM tracks t JOIN clips c ON c.id = t.clip_id "
                         "WHERE c.match_id = ? AND t.team = ? AND t.color IS NOT NULL", (match_id, team))
        if not rows:
            out.append(None)
            continue
        rgb = [[int(r["color"][i:i + 2], 16) for i in (1, 3, 5)] for r in rows]
        weights = [r["n_frames"] or 1 for r in rows]
        avg = [round(sum(c[k] * w for c, w in zip(rgb, weights)) / sum(weights)) for k in range(3)]
        out.append("#" + "".join(f"{v:02x}" for v in avg))
    return out


@app.patch("/api/matches/{match_id}")
def update_match(match_id: int, data: dict = Body(...)):
    return _update("matches", match_id, data, {"name", "date", "team0_name", "team1_name"})


@app.delete("/api/matches/{match_id}")
def delete_match(match_id: int):
    for c in store.all("SELECT id FROM clips WHERE match_id = ?", (match_id,)):
        shutil.rmtree(config.CLIPS_DIR / str(c["id"]), ignore_errors=True)
    store.run("DELETE FROM matches WHERE id = ?", (match_id,))
    analytics.invalidate()
    return {"ok": True}


# --- clips -------------------------------------------------------------------------------

@app.post("/api/matches/{match_id}/clips")
def upload_clips(match_id: int, files: list[UploadFile] = File(...)):
    _get("matches", match_id)
    n = store.one("SELECT COUNT(*) AS n FROM clips WHERE match_id = ?", (match_id,))["n"]
    created = []
    for i, f in enumerate(files):
        cid = store.run("INSERT INTO clips (match_id, filename, path, order_idx) VALUES (?,?,?,?)",
                        (match_id, f.filename, "", n + i))
        dst = clip_dir(cid) / ("source" + (Path(f.filename).suffix.lower() or ".mp4"))
        with dst.open("wb") as out:
            shutil.copyfileobj(f.file, out, length=8 * 1024 * 1024)
        try:
            info = probe(dst)
        except ValueError as e:
            store.run("DELETE FROM clips WHERE id = ?", (cid,))
            shutil.rmtree(dst.parent, ignore_errors=True)
            raise HTTPException(400, str(e)) from e
        _store_probe(cid, dst, info)
        created.append(_get("clips", cid))
    return created


def _store_probe(cid: int, path: Path, info: dict, inherit: dict | None = None) -> None:
    """Videogegevens opslaan; bij knippen GPS en camerapositie van het origineel overnemen."""
    extra = {k: info.get(k) for k in ("gps_lat", "gps_lon", "gps_acc", "device")}
    if inherit:
        for k in ("gps_lat", "gps_lon", "gps_acc", "device", "cam_x", "cam_y", "cam_h", "cam_source"):
            if extra.get(k) is None:
                extra[k] = inherit.get(k)
    sets = ", ".join(f"{k} = ?" for k in extra)
    store.run(f"UPDATE clips SET path = ?, fps = ?, width = ?, height = ?, duration = ?, {sets} WHERE id = ?",
              (str(path), info["fps"], info["width"], info["height"], info["duration"], *extra.values(), cid))


@app.patch("/api/clips/{clip_id}")
def update_clip(clip_id: int, data: dict = Body(...)):
    return _update("clips", clip_id, data, {"order_idx", "period", "start_minute"})


@app.delete("/api/clips/{clip_id}")
def delete_clip(clip_id: int):
    if _get("clips", clip_id)["status"] in ("wachtrij", "preview", "analyse"):
        raise HTTPException(409, "Deze video wordt nog geanalyseerd; verwijder hem als de analyse klaar is")
    store.run("DELETE FROM clips WHERE id = ?", (clip_id,))
    shutil.rmtree(config.CLIPS_DIR / str(clip_id), ignore_errors=True)
    analytics.invalidate()
    return {"ok": True}


@app.post("/api/clips/{clip_id}/process")
def process(clip_id: int):
    c = _get("clips", clip_id)
    if c["status"] in ("wachtrij", "preview", "analyse"):
        raise HTTPException(409, "Clip wordt al verwerkt")
    analytics.invalidate()
    worker.submit(clip_id)
    return _get("clips", clip_id)


def _range_response(path: Path, request: Request) -> Response:
    size = path.stat().st_size
    ctype = mimetypes.guess_type(path.name)[0] or "video/mp4"
    rng = request.headers.get("range")
    if not rng:
        return FileResponse(path, media_type=ctype, headers={"Accept-Ranges": "bytes"})
    start_s, _, end_s = rng.replace("bytes=", "").partition("-")
    start = int(start_s) if start_s else 0
    end = min(int(end_s) if end_s else size - 1, size - 1)
    if start >= size:
        return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})

    def body():
        with path.open("rb") as f:
            f.seek(start)
            left = end - start + 1
            while left > 0:
                chunk = f.read(min(1024 * 1024, left))
                if not chunk:
                    break
                left -= len(chunk)
                yield chunk

    return StreamingResponse(body(), status_code=206, media_type=ctype, headers={
        "Content-Range": f"bytes {start}-{end}/{size}", "Accept-Ranges": "bytes",
        "Content-Length": str(end - start + 1)})


@app.get("/api/clips/{clip_id}/video")
def clip_video(clip_id: int, request: Request):
    c = _get("clips", clip_id)
    preview = clip_dir(clip_id) / "preview.mp4"
    return _range_response(preview if preview.exists() else Path(c["path"]), request)


def _snap_time(clip_id: int, t: float) -> float:
    """Tijd afronden op het dichtstbijzijnde geanalyseerde frame (als de clip geanalyseerd is),
    zodat een sleutelframe precies op een frame van de camerabeweging valt."""
    r = store.one("SELECT t FROM frames WHERE clip_id = ? ORDER BY ABS(t - ?) LIMIT 1", (clip_id, t))
    return r["t"] if r else t


@app.get("/api/clips/{clip_id}/frame")
def clip_frame(clip_id: int, t: float = 0.0):
    c = _get("clips", clip_id)
    try:
        frame, ft = read_frame(Path(c["path"]), _snap_time(clip_id, t))
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return Response(buf.tobytes(), media_type="image/jpeg",
                    headers={"X-Frame-Time": f"{ft:.4f}", "Access-Control-Expose-Headers": "X-Frame-Time"})


# --- kalibratie --------------------------------------------------------------------------

@app.get("/api/clips/{clip_id}/keyframes")
def get_keyframes(clip_id: int):
    out = []
    prior = camera_prior(_get("clips", clip_id))
    for kf in store.keyframes(clip_id):
        if kf.get("auto"):
            kf["score"] = json.loads(kf["score"]) if kf.get("score") else None
            kf["error_m"] = None
        else:
            kf["error_m"] = _calib_error(kf["points"], prior)
        out.append(kf)
    return out


def _calib_error(points: list[dict], prior: dict | None = None) -> float | None:
    try:
        return round(fit_calibration(points, camera=prior)[1], 2)
    except ValueError:
        return None


@app.post("/api/clips/{clip_id}/keyframes")
def save_keyframe(clip_id: int, data: dict = Body(...)):
    """data: {t, points: [{name, img: [x, y], pitch: [x, y]}], id?}"""
    clip = _get("clips", clip_id)
    points = data.get("points") or []
    try:
        _, err = fit_calibration(points, camera=camera_prior(clip))
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    data["t"] = _snap_time(clip_id, float(data["t"]))
    if data.get("id"):
        store.run("UPDATE keyframes SET t = ?, points = ? WHERE id = ? AND clip_id = ?",
                  (data["t"], json.dumps(points), data["id"], clip_id))
        kid = data["id"]
    else:
        kid = store.run("INSERT INTO keyframes (clip_id, t, points) VALUES (?,?,?)",
                        (clip_id, data["t"], json.dumps(points)))
    analytics.invalidate(clip_id=clip_id)
    _start_autocalib(clip)
    return {"id": kid, "t": data["t"], "error_m": round(err, 2)}


@app.post("/api/clips/{clip_id}/calibrate-preview")
def calibrate_preview(clip_id: int, data: dict = Body(...)):
    """Live voorbeeld tijdens het klikken (met cameramodel): veld -> beeld, fout en camera."""
    clip = _get("clips", clip_id)
    points = data.get("points") or []
    prior = camera_prior(clip)
    try:
        K, err = fit_calibration(points, camera=prior)
    except (ValueError, np.linalg.LinAlgError) as e:
        return {"ok": False, "message": str(e)}
    cam = None
    if prior is not None:
        try:
            cam = {k: round(v, 1) for k, v in fit_camera(points, prior)[1].items() if k != "params"}
        except ValueError:
            pass
    H = np.linalg.inv(K)
    return {"ok": True, "H": normalize_h(H).tolist(), "error_m": round(err, 2), "camera": cam}


@app.patch("/api/clips/{clip_id}/camera")
def set_camera(clip_id: int, data: dict = Body(...)):
    """Waar stond de camera? data: {x, y, h, source: 'hand'|'gps'} (x/y null = wissen)."""
    _get("clips", clip_id)
    fields = {}
    for k in ("x", "y", "h"):
        if k in data:
            fields[f"cam_{k}"] = None if data[k] is None else float(data[k])
    if "source" in data:
        fields["cam_source"] = data["source"]
    if fields:
        sets = ", ".join(f"{k} = ?" for k in fields)
        store.run(f"UPDATE clips SET {sets} WHERE id = ?", (*fields.values(), clip_id))
        analytics.invalidate(clip_id=clip_id)
    return _get("clips", clip_id)


@app.post("/api/clips/{clip_id}/camera/gps")
def camera_from_gps(clip_id: int):
    """Zoek het voetbalveld bij de GPS-positie van de video (OpenStreetMap) en zet de camera daar."""
    c = _get("clips", clip_id)
    if c.get("gps_lat") is None:
        raise HTTPException(400, "Deze video bevat geen GPS-positie")
    try:
        polys = geo.query_pitches(c["gps_lat"], c["gps_lon"])
    except Exception as e:  # noqa: BLE001  (geen internet, server druk, ...)
        raise HTTPException(502, f"OpenStreetMap niet bereikbaar ({e}). Klik je positie dan zelf aan.") from e
    try:
        pos = geo.camera_on_pitch(c["gps_lat"], c["gps_lon"], polys)
    except ValueError as e:
        raise HTTPException(404, f"{e}. Klik je positie zelf aan op de veldtekening.") from e
    store.run("UPDATE clips SET cam_x = ?, cam_y = ?, cam_h = COALESCE(cam_h, 1.6), cam_source = 'gps' WHERE id = ?",
              (pos["x"], pos["y"], clip_id))
    analytics.invalidate(clip_id=clip_id)
    return {**pos, "clip": _get("clips", clip_id)}


@app.get("/api/clips/{clip_id}/propose")
def propose_calibration(clip_id: int, t: float = 0.0):
    """Automatisch voorstel: zoek het veld in dit beeld vanaf de bekende camerapositie."""
    from .autocalib import propose

    clip = _get("clips", clip_id)
    prior = camera_prior(clip)
    if prior is None:
        raise HTTPException(400, "Stel eerst in waar je stond (📍 of via GPS); dan kan de app het veld zelf zoeken")
    t = _snap_time(clip_id, t)
    try:
        frame, ft = read_frame(Path(clip["path"]), t)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    boxes = None
    fr = store.one("SELECT idx FROM frames WHERE clip_id = ? ORDER BY ABS(t - ?) LIMIT 1", (clip_id, ft))
    if fr:
        rows = store.all("SELECT x1, y1, x2, y2 FROM detections WHERE clip_id = ? AND idx = ?", (clip_id, fr["idx"]))
        boxes = np.array([[r["x1"], r["y1"], r["x2"], r["y2"]] for r in rows]).reshape(-1, 4)
    res = propose(frame, prior, boxes)
    if res is None:
        return {"ok": False, "t": ft, "message": "Geen overtuigend voorstel gevonden in dit beeld. Kies een moment met meer "
                                                  "veldlijnen in beeld, of klik zelf 1 punt + 1 lijn aan."}
    H, info = res
    W, Hh = int(clip["width"]), int(clip["height"])
    to_img = analytics.h_to_image(H)
    # veldpunten in beeld, plus die net buiten beeld (die tellen gewoon mee en maken de kalibratie
    # stevig), plus punten op de lijnen die in beeld zijn
    points = _spread(analytics.image_landmarks(to_img, W, Hh, line_points=False, margin=0.02), 8)
    if len(points) < 6:
        names = {p["name"] for p in points}
        extra = [p for p in analytics.image_landmarks(to_img, W, Hh, line_points=False, margin=0.6) if p["name"] not in names]
        points += _spread(extra, 6 - len(points))
    if len(points) < 4:
        points += [p for p in analytics.image_landmarks(to_img, W, Hh, margin=0.0) if p.get("line")][:6]
    return {"ok": True, "t": ft, "points": points, "H": normalize_h(H).tolist(), "info": info}


def _spread(points: list[dict], n: int) -> list[dict]:
    """Hoogstens n punten, zo ver mogelijk uit elkaar in beeld (verste-punt-keuze)."""
    if len(points) <= n:
        return points
    xy = np.array([p["img"] for p in points], float)
    chosen = [int(np.argmax(np.linalg.norm(xy - xy.mean(0), axis=1)))]
    d = np.linalg.norm(xy - xy[chosen[0]], axis=1)
    while len(chosen) < n:
        k = int(np.argmax(d))
        chosen.append(k)
        d = np.minimum(d, np.linalg.norm(xy - xy[k], axis=1))
    return [points[i] for i in sorted(chosen)]


@app.post("/api/clips/{clip_id}/warp")
def warp_points(clip_id: int, data: dict = Body(...)):
    """Beeldpunten van moment from_t verplaatsen naar moment to_t (ze bewegen mee met het veld).

    data: {from_t, to_t, points: [[x, y], ...]}"""
    clip = _get("clips", clip_id)
    pts = np.array(data.get("points") or [], float).reshape(-1, 2)
    t0, t1 = float(data["from_t"]), float(data["to_t"])
    if not len(pts):
        return {"ok": True, "points": []}
    out = analytics.warp_points(store, clip_id, t0, t1, pts)
    method = "camerabeweging"
    if out is None:  # niet geanalyseerd: de twee beelden direct met elkaar vergelijken
        try:
            out = _match_frames(Path(clip["path"]), t0, t1, pts)
            method = "beeldvergelijking"
        except ValueError as e:
            return {"ok": False, "message": str(e)}
    return {"ok": True, "method": method, "points": [[round(float(x), 1), round(float(y), 1)] for x, y in out]}


def _match_frames(path: Path, t0: float, t1: float, pts: np.ndarray) -> np.ndarray:
    """Camerabeweging tussen twee momenten zonder analyse: de beelden ertussen in stapjes van 0,1 s
    volgen (zoals bij de analyse) en de stapjes aan elkaar rijgen."""
    from .calibration import MotionEstimator

    lo, hi = min(t0, t1), max(t0, t1)
    if hi - lo > 30:
        raise ValueError("Te ver uit elkaar om de camerabeweging te volgen; analyseer de video eerst")
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, lo - 1.0) * 1000)
    motion = MotionEstimator()
    M = np.eye(3)  # beeld(lo) -> beeld(huidig)
    nxt, started = lo, False
    try:
        while cap.grab():
            pos = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if pos < nxt - 0.02:
                continue
            ok, frame = cap.retrieve()
            if not ok:
                break
            step = motion.step(frame)
            if started:
                M = step @ M
            started = True
            if pos >= hi - 0.02:
                break
            nxt = min(hi, pos + 0.1)
    finally:
        cap.release()
    if not started:
        raise ValueError("Beelden konden niet gelezen worden")
    if t1 < t0:
        M = np.linalg.inv(M)
    return cv2.perspectiveTransform(pts.reshape(-1, 1, 2).astype(np.float64), M).reshape(-1, 2)


@app.post("/api/clips/{clip_id}/autocalib/accept")
def accept_autocalib(clip_id: int, data: dict = Body(default={})):
    """Automatische sleutelframes goedkeuren (data.ids, of allemaal). Goedgekeurde blijven bij
    opnieuw bijstellen staan."""
    ids = data.get("ids")
    if ids:
        q = ",".join("?" * len(ids))
        store.run(f"UPDATE keyframes SET accepted = 1 WHERE clip_id = ? AND auto = 1 AND id IN ({q})", (clip_id, *ids))
    else:
        store.run("UPDATE keyframes SET accepted = 1 WHERE clip_id = ? AND auto = 1", (clip_id,))
    return {"ok": True}


@app.get("/api/clips/{clip_id}/predict")
def predict(clip_id: int, t: float = 0.0):
    return analytics.predict_landmarks(store, clip_id, t)


@app.delete("/api/keyframes/{kf_id}")
def delete_keyframe(kf_id: int):
    kf = store.one("SELECT * FROM keyframes WHERE id = ?", (kf_id,))
    store.run("DELETE FROM keyframes WHERE id = ?", (kf_id,))
    if kf and not kf["auto"]:
        if store.one("SELECT 1 AS x FROM keyframes WHERE clip_id = ? AND auto = 0", (kf["clip_id"],)):
            _start_autocalib(_get("clips", kf["clip_id"]))
        else:  # geen handmatige kalibratie meer: automatische vervallen ook
            store.run("DELETE FROM keyframes WHERE clip_id = ? AND auto = 1", (kf["clip_id"],))
    analytics.invalidate()
    return {"ok": True}


def _start_autocalib(clip: dict) -> None:
    if worker is not None and clip["status"] == "klaar":
        worker.submit_autocalib(clip["id"])


@app.post("/api/clips/{clip_id}/autocalib")
def start_autocalib(clip_id: int):
    """Kalibratie automatisch bijstellen over de hele video (op basis van de veldlijnen)."""
    c = _get("clips", clip_id)
    if c["status"] != "klaar":
        raise HTTPException(409, "Analyseer de video eerst")
    if not store.one("SELECT 1 AS x FROM keyframes WHERE clip_id = ? AND auto = 0", (clip_id,)):
        raise HTTPException(400, "Kalibreer eerst één sleutelframe met de hand")
    worker.submit_autocalib(clip_id)
    return _get("clips", clip_id)


@app.delete("/api/clips/{clip_id}/autocalib")
def clear_autocalib(clip_id: int):
    store.run("DELETE FROM keyframes WHERE clip_id = ? AND auto = 1", (clip_id,))
    store.run("UPDATE clips SET calib_status = NULL, calib_message = NULL WHERE id = ?", (clip_id,))
    analytics.invalidate()
    return {"ok": True}


# --- tracks en spelers -------------------------------------------------------------------

@app.get("/api/clips/{clip_id}/tracks")
def get_tracks(clip_id: int):
    rows = store.all("SELECT * FROM tracks WHERE clip_id = ? ORDER BY n_frames DESC", (clip_id,))
    d = analytics.load_clip(store, clip_id)
    for r in rows:
        r["valid"] = d is None or r["track_id"] in d.valid_tracks
    return rows


@app.get("/api/clips/{clip_id}/thumb/{track_id}")
def track_thumb(clip_id: int, track_id: int):
    p = clip_dir(clip_id) / "thumbs" / f"{track_id}.jpg"
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p, media_type="image/jpeg")


@app.post("/api/clips/{clip_id}/swap-teams")
def swap_teams(clip_id: int, data: dict = Body(default={})):
    """Thuis en Uit omwisselen: voor deze video, of (all) voor alle video's van de wedstrijd."""
    c = _get("clips", clip_id)
    ids = [r["id"] for r in store.all("SELECT id FROM clips WHERE match_id = ?", (c["match_id"],))] \
        if data.get("all") else [clip_id]
    q = ",".join("?" * len(ids))
    swap = "CASE {0} WHEN 0 THEN 1 WHEN 1 THEN 0 ELSE {0} END"
    store.run(f"UPDATE tracks SET team = {swap.format('team')}, team_auto = {swap.format('team_auto')} "
              f"WHERE clip_id IN ({q})", tuple(ids))
    if data.get("all"):
        m = store.one("SELECT * FROM matches WHERE id = ?", (c["match_id"],)) or {}
        if "team0_color" in m:
            store.run("UPDATE matches SET team0_color = ?, team1_color = ? WHERE id = ?",
                      (m.get("team1_color"), m.get("team0_color"), c["match_id"]))
    analytics.invalidate(geometry=False)
    return {"ok": True, "clips": ids}


@app.post("/api/clips/{clip_id}/reassign-teams")
def reassign_teams(clip_id: int):
    """Teams van deze video opnieuw automatisch indelen (handmatige correcties blijven staan)."""
    from .pipeline import assign_clip_teams

    c = _get("clips", clip_id)
    if c["status"] != "klaar":
        raise HTTPException(409, "Analyseer de video eerst")
    res = assign_clip_teams(store, clip_id, keep_manual=True)
    analytics.invalidate(geometry=False)
    return res


@app.patch("/api/clips/{clip_id}/tracks/{track_id}")
def update_track(clip_id: int, track_id: int, data: dict = Body(...)):
    fields = {k: data[k] for k in ("team", "player_id") if k in data}
    if not fields:
        raise HTTPException(400, "Niets te wijzigen")
    sets = ", ".join(f"{k} = ?" for k in fields)
    store.run(f"UPDATE tracks SET {sets} WHERE clip_id = ? AND track_id = ?",
              (*fields.values(), clip_id, track_id))
    analytics.invalidate(geometry=False)
    return store.one("SELECT * FROM tracks WHERE clip_id = ? AND track_id = ?", (clip_id, track_id))


@app.post("/api/matches/{match_id}/players")
def create_player(match_id: int, data: dict = Body(...)):
    if not data.get("name"):
        raise HTTPException(400, "Naam is verplicht")
    pid = store.run("INSERT INTO players (match_id, name, number, team) VALUES (?,?,?,?)",
                    (match_id, data["name"], data.get("number"), int(data.get("team", 0))))
    analytics.invalidate(geometry=False)
    return _get("players", pid)


@app.patch("/api/players/{player_id}")
def update_player(player_id: int, data: dict = Body(...)):
    return _update("players", player_id, data, {"name", "number", "team"})


@app.delete("/api/players/{player_id}")
def delete_player(player_id: int):
    store.run("DELETE FROM players WHERE id = ?", (player_id,))
    analytics.invalidate(geometry=False)
    return {"ok": True}


@app.post("/api/matches/{match_id}/auto-assign")
def auto_assign(match_id: int, data: dict = Body(default={})):
    """Koppel tracks aan spelers op basis van gelezen rugnummer + team (alleen nog niet gekoppelde)."""
    min_conf = float(data.get("min_conf", 0.5))
    players = store.all("SELECT * FROM players WHERE match_id = ? AND number IS NOT NULL", (match_id,))
    by_key = {(str(p["number"]).strip(), p["team"]): p["id"] for p in players}
    n = 0
    for tr in store.all("SELECT t.* FROM tracks t JOIN clips c ON c.id = t.clip_id WHERE c.match_id = ? "
                        "AND t.player_id IS NULL AND t.jersey_guess IS NOT NULL AND t.jersey_conf >= ?",
                        (match_id, min_conf)):
        pid = by_key.get((tr["jersey_guess"], tr["team"]))
        if pid:
            store.run("UPDATE tracks SET player_id = ? WHERE clip_id = ? AND track_id = ?",
                      (pid, tr["clip_id"], tr["track_id"]))
            n += 1
    analytics.invalidate(geometry=False)
    return {"assigned": n}


# --- analyse -----------------------------------------------------------------------------

@app.get("/api/matches/{match_id}/stats")
def match_stats(match_id: int):
    _get("matches", match_id)
    return analytics.match_stats(store, match_id)


@app.get("/api/clips/{clip_id}/positions")
def clip_positions(clip_id: int, t0: float = 0, t1: float = 60):
    return analytics.clip_positions(store, clip_id, t0, t1)


@app.get("/api/clips/{clip_id}/boxes")
def clip_boxes(clip_id: int, t0: float = 0, t1: float = 10):
    return analytics.clip_boxes(store, clip_id, t0, t1)


# --- clips (momenten), knippen en export -----------------------------------------------

def _moment_payload(data: dict) -> dict:
    out = {}
    for k in ("start", "end"):
        if k in data:
            out[k] = max(0.0, float(data[k]))
    for k in ("label", "comment"):
        if k in data:
            out[k] = data[k]
    if "spotlight_player_id" in data:
        out["spotlight_player_id"] = int(data["spotlight_player_id"]) if data["spotlight_player_id"] else None
    if "players" in data:
        out["players"] = json.dumps([int(p) for p in data["players"] or []])
    if "drawings" in data:
        out["drawings"] = json.dumps(data["drawings"] or [])
    return out


@app.get("/api/matches/{match_id}/moments")
def get_moments(match_id: int):
    return store.moments(match_id)


@app.post("/api/matches/{match_id}/moments")
def create_moment(match_id: int, data: dict = Body(...)):
    c = _get("clips", int(data["clip_id"]))
    fields = {"label": "Moment", "players": "[]", "drawings": "[]", **_moment_payload(data)}
    fields.setdefault("start", 0.0)
    fields.setdefault("end", fields["start"] + 10)
    if c["duration"]:
        fields["end"] = min(fields["end"], c["duration"])
    if fields["end"] - fields["start"] < 0.5:
        raise HTTPException(400, "Een clip moet minstens een halve seconde duren")
    cols = ["match_id", "clip_id", *fields]
    mid = store.run(f"INSERT INTO moments ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                    (match_id, c["id"], *fields.values()))
    return store.moment(mid)


@app.patch("/api/moments/{moment_id}")
def update_moment(moment_id: int, data: dict = Body(...)):
    m = store.moment(moment_id)
    if m is None:
        raise HTTPException(404, "Niet gevonden")
    fields = _moment_payload(data)
    start, end = fields.get("start", m["start"]), fields.get("end", m["end"])
    if end - start < 0.5:
        raise HTTPException(400, "Een clip moet minstens een halve seconde duren")
    if fields:
        sets = ", ".join(f"{k} = ?" for k in fields)
        store.run(f"UPDATE moments SET {sets} WHERE id = ?", (*fields.values(), moment_id))
    return store.moment(moment_id)


@app.delete("/api/moments/{moment_id}")
def delete_moment(moment_id: int):
    store.run("DELETE FROM moments WHERE id = ?", (moment_id,))
    return {"ok": True}


@app.post("/api/clips/{clip_id}/split")
def split_clip(clip_id: int, data: dict = Body(...)):
    """Knip een video in delen. data: {segments: [{start, end, name?, period?, start_minute?}],
    delete_original: bool}. Elk deel wordt een nieuwe video in dezelfde wedstrijd."""
    c = _get("clips", clip_id)
    if c["status"] in ("wachtrij", "preview", "analyse"):
        raise HTTPException(409, "Wacht tot de analyse klaar is")
    segments = sorted(data.get("segments") or [], key=lambda s: float(s["start"]))
    if not segments:
        raise HTTPException(400, "Geen delen gekozen")
    src = Path(c["path"])
    stem = Path(c["filename"]).stem
    created = []
    later = store.all("SELECT id, order_idx FROM clips WHERE match_id = ? AND order_idx > ?",
                      (c["match_id"], c["order_idx"]))
    for row in later:  # ruimte maken in de volgorde
        store.run("UPDATE clips SET order_idx = ? WHERE id = ?", (row["order_idx"] + len(segments), row["id"]))
    for i, seg in enumerate(segments):
        name = (seg.get("name") or f"deel {i + 1}").strip()
        filename = f"{stem} - {name}{src.suffix}"
        cid = store.run("INSERT INTO clips (match_id, filename, path, order_idx, period, start_minute) "
                        "VALUES (?,?,?,?,?,?)",
                        (c["match_id"], filename, "", c["order_idx"] + 1 + i, int(seg.get("period") or c["period"]),
                         float(seg.get("start_minute") or 0)))
        dst = clip_dir(cid) / ("source" + src.suffix.lower())
        try:
            clip_export.cut_copy(src, float(seg["start"]), float(seg["end"]), dst)
            info = probe(dst)
        except (RuntimeError, ValueError) as e:
            store.run("DELETE FROM clips WHERE id = ?", (cid,))
            shutil.rmtree(dst.parent, ignore_errors=True)
            raise HTTPException(400, str(e)) from e
        _store_probe(cid, dst, info, inherit=c)
        created.append(_get("clips", cid))
    if data.get("delete_original"):
        delete_clip(clip_id)
    analytics.invalidate()
    return created


@app.post("/api/matches/{match_id}/export")
def export(match_id: int, data: dict = Body(...)):
    """data: {name, moment_ids: [...], mode: 'reel' | 'zip'}"""
    ids = [int(i) for i in data.get("moment_ids") or []]
    if not ids:
        raise HTTPException(400, "Geen clips gekozen")
    try:
        out = render.export_moments(store, ids, data.get("name") or "clips", data.get("mode") or "reel")
    except Exception as e:  # noqa: BLE001
        logging.exception("Export mislukt")
        raise HTTPException(500, f"Exporteren mislukt: {e}") from e
    return {"file": out.name, "url": f"/api/exports/{out.name}"}


@app.get("/api/exports/{name}")
def get_export(name: str):
    p = _export_path(name)
    media = "application/zip" if p.suffix == ".zip" else "video/mp4"
    return FileResponse(p, media_type=media, filename=name)


def _export_path(name: str) -> Path:
    p = (config.EXPORTS_DIR / name).resolve()
    if p.parent != config.EXPORTS_DIR.resolve() or not p.exists():
        raise HTTPException(404)
    return p


@app.post("/api/exports/{name}/reveal")
def reveal_export(name: str):
    """Toon het bestand in de Finder (alleen op de Mac waar de app draait)."""
    p = _export_path(name)
    if sys.platform == "darwin":
        subprocess.run(["open", "-R", str(p)], check=False)
        return {"ok": True}
    return {"ok": False, "path": str(p)}


class NoCacheStatic(StaticFiles):
    """De interface altijd vers laden, zodat de browser na een update geen oude JavaScript gebruikt."""

    async def get_response(self, path, scope):
        resp = await super().get_response(path, scope)
        resp.headers["Cache-Control"] = "no-cache, must-revalidate"
        return resp


app.mount("/", NoCacheStatic(directory=config.STATIC_DIR, html=True), name="static")
