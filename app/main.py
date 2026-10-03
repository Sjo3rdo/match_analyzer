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
from fastapi import Body, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import analytics, clips as clip_export, config, pitch, render
from .calibration import fit_calibration
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
    yield


app = FastAPI(title="Match Analyzer", lifespan=lifespan)


def _get(table: str, id_: int) -> dict:
    row = store.one(f"SELECT * FROM {table} WHERE id = ?", (id_,))
    if row is None:
        raise HTTPException(404, "Niet gevonden")
    return row


def _update(table: str, id_: int, data: dict, allowed: set[str]) -> dict:
    fields = {k: v for k, v in data.items() if k in allowed}
    if fields:
        sets = ", ".join(f"{k} = ?" for k in fields)
        store.run(f"UPDATE {table} SET {sets} WHERE id = ?", (*fields.values(), id_))
        analytics.invalidate()
    return _get(table, id_)


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
        c["n_keyframes"] = len(store.keyframes(c["id"]))
    m["team_colors"] = _team_colors(match_id)
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
        store.run("UPDATE clips SET path = ?, fps = ?, width = ?, height = ?, duration = ? WHERE id = ?",
                  (str(dst), info["fps"], info["width"], info["height"], info["duration"], cid))
        created.append(_get("clips", cid))
    return created


@app.patch("/api/clips/{clip_id}")
def update_clip(clip_id: int, data: dict = Body(...)):
    return _update("clips", clip_id, data, {"order_idx", "period", "start_minute"})


@app.delete("/api/clips/{clip_id}")
def delete_clip(clip_id: int):
    _get("clips", clip_id)
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
    for kf in store.keyframes(clip_id):
        kf["error_m"] = _calib_error(kf["points"])
        out.append(kf)
    return out


def _calib_error(points: list[dict]) -> float | None:
    try:
        return round(fit_calibration(points)[1], 2)
    except ValueError:
        return None


@app.post("/api/clips/{clip_id}/keyframes")
def save_keyframe(clip_id: int, data: dict = Body(...)):
    """data: {t, points: [{name, img: [x, y], pitch: [x, y]}], id?}"""
    _get("clips", clip_id)
    points = data.get("points") or []
    try:
        _, err = fit_calibration(points)
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
    analytics.invalidate()
    return {"id": kid, "t": data["t"], "error_m": round(err, 2)}


@app.get("/api/clips/{clip_id}/predict")
def predict(clip_id: int, t: float = 0.0):
    return analytics.predict_landmarks(store, clip_id, t)


@app.delete("/api/keyframes/{kf_id}")
def delete_keyframe(kf_id: int):
    store.run("DELETE FROM keyframes WHERE id = ?", (kf_id,))
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


@app.patch("/api/clips/{clip_id}/tracks/{track_id}")
def update_track(clip_id: int, track_id: int, data: dict = Body(...)):
    fields = {k: data[k] for k in ("team", "player_id") if k in data}
    if not fields:
        raise HTTPException(400, "Niets te wijzigen")
    sets = ", ".join(f"{k} = ?" for k in fields)
    store.run(f"UPDATE tracks SET {sets} WHERE clip_id = ? AND track_id = ?",
              (*fields.values(), clip_id, track_id))
    analytics.invalidate()
    return store.one("SELECT * FROM tracks WHERE clip_id = ? AND track_id = ?", (clip_id, track_id))


@app.post("/api/matches/{match_id}/players")
def create_player(match_id: int, data: dict = Body(...)):
    if not data.get("name"):
        raise HTTPException(400, "Naam is verplicht")
    pid = store.run("INSERT INTO players (match_id, name, number, team) VALUES (?,?,?,?)",
                    (match_id, data["name"], data.get("number"), int(data.get("team", 0))))
    analytics.invalidate()
    return _get("players", pid)


@app.patch("/api/players/{player_id}")
def update_player(player_id: int, data: dict = Body(...)):
    return _update("players", player_id, data, {"name", "number", "team"})


@app.delete("/api/players/{player_id}")
def delete_player(player_id: int):
    store.run("DELETE FROM players WHERE id = ?", (player_id,))
    analytics.invalidate()
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
    analytics.invalidate()
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
        store.run("UPDATE clips SET path = ?, fps = ?, width = ?, height = ?, duration = ? WHERE id = ?",
                  (str(dst), info["fps"], info["width"], info["height"], info["duration"], cid))
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


app.mount("/", StaticFiles(directory=config.STATIC_DIR, html=True), name="static")
