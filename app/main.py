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

from . import analytics, clips as clip_export, config, geo, learning, pitch, render, shots, teams, training
from .calibration import camera_prior, fit_calibration, fit_camera, normalize_h, suspect_point, kf_camera
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
def get_pitch(match_id: int | None = None):
    """Veldmodel (punten en lijnen) met de veldmaten van de wedstrijd (standaard 105 x 68)."""
    g = pitch.of_match(store.one("SELECT * FROM matches WHERE id = ?", (match_id,)) if match_id else None)
    return {"length": g.length, "width": g.width,
            "landmarks": [{"name": k, "x": v[0], "y": v[1]} for k, v in g.landmarks.items()],
            "lines": [{"name": k, "from": list(a), "to": list(b)} for k, (a, b) in g.lines.items()],
            "elevated": [{"name": k, "x": v[0], "y": v[1], "h": v[2]} for k, v in g.elevated.items()],
            "elevated_lines": [{"name": k, "from": list(a), "to": list(b)} for k, (a, b) in g.elevated_lines.items()]}


# --- wedstrijden -------------------------------------------------------------------------

@app.get("/api/matches")
def list_matches():
    rows = store.all("SELECT m.*, (SELECT COUNT(*) FROM clips c WHERE c.match_id = m.id) AS n_clips "
                     "FROM matches m ORDER BY COALESCE(date, created_at) DESC")
    for m in rows:
        m["score"] = shots.score(store, m["id"])
    return rows


@app.post("/api/matches")
def create_match(data: dict = Body(...)):
    if not data.get("name"):
        raise HTTPException(400, "Naam is verplicht")
    mid = store.run("INSERT INTO matches (name, date, team0_name, team1_name) VALUES (?,?,?,?)",
                    (data["name"], data.get("date"), data.get("team0_name") or "Thuis",
                     data.get("team1_name") or "Uit"))
    # Vaste selectie overnemen: expliciet gekozen, of een opgeslagen team met dezelfde naam
    for team in (0, 1):
        sq = data.get(f"team{team}_squad")
        if not sq and data.get(f"team{team}_name"):
            row = store.one("SELECT id FROM squads WHERE lower(name) = lower(?) ORDER BY id DESC LIMIT 1",
                            (data[f"team{team}_name"].strip(),))
            sq = row and row["id"]
        if sq:
            _load_squad(mid, team, int(sq))
    return _get("matches", mid)


# --- vaste selecties (teams die je bij elke wedstrijd weer gebruikt) -----------------------

def _squad(squad_id: int) -> dict:
    sq = _get("squads", squad_id)
    sq["players"] = store.all("SELECT name, number FROM squad_players WHERE squad_id = ? "
                              "ORDER BY CAST(number AS INTEGER), name", (squad_id,))
    return sq


@app.get("/api/squads")
def list_squads():
    return [_squad(r["id"]) for r in store.all("SELECT id FROM squads ORDER BY lower(name)")]


@app.delete("/api/squads/{squad_id}")
def delete_squad(squad_id: int):
    store.run("DELETE FROM squads WHERE id = ?", (squad_id,))
    return {"ok": True}


@app.post("/api/matches/{match_id}/save-squad")
def save_squad(match_id: int, data: dict = Body(...)):
    """De selectie van team 0 of 1 bewaren als vaste selectie (zelfde naam = bijwerken)."""
    m = _get("matches", match_id)
    team = int(data.get("team", 0))
    name = (data.get("name") or m[f"team{team}_name"] or "").strip()
    if not name:
        raise HTTPException(400, "Geef het team een naam")
    players = store.all("SELECT name, number FROM players WHERE match_id = ? AND team = ?", (match_id, team))
    color = _team_colors(match_id)[team] or m.get(f"team{team}_color")
    row = store.one("SELECT id FROM squads WHERE lower(name) = lower(?)", (name,))
    with store.tx() as c:
        if row:
            sid = row["id"]
            c.execute("UPDATE squads SET name = ?, color = COALESCE(?, color) WHERE id = ?", (name, color, sid))
            c.execute("DELETE FROM squad_players WHERE squad_id = ?", (sid,))
        else:
            sid = c.execute("INSERT INTO squads (name, color) VALUES (?, ?)", (name, color)).lastrowid
        c.executemany("INSERT INTO squad_players (squad_id, name, number) VALUES (?,?,?)",
                      [(sid, p["name"], p["number"]) for p in players])
    store.run(f"UPDATE matches SET team{team}_squad = ? WHERE id = ?", (sid, match_id))
    return _squad(sid)


@app.post("/api/matches/{match_id}/load-squad")
def load_squad(match_id: int, data: dict = Body(...)):
    """Spelers van een vaste selectie (of van een eerdere wedstrijd) overnemen in team 0 of 1."""
    _get("matches", match_id)
    team = int(data.get("team", 0))
    if data.get("from_match"):
        src = _get("matches", int(data["from_match"]))
        src_team = int(data.get("from_team", team))
        players = store.all("SELECT name, number FROM players WHERE match_id = ? AND team = ?", (src["id"], src_team))
        n = _add_players(match_id, team, players)
        store.run(f"UPDATE matches SET team{team}_name = ? WHERE id = ?", (src[f"team{src_team}_name"], match_id))
    else:
        n = _load_squad(match_id, team, int(data["squad_id"]))
    analytics.invalidate(geometry=False)
    return {"added": n, "match": get_match(match_id)}


def _load_squad(match_id: int, team: int, squad_id: int) -> int:
    sq = _squad(squad_id)
    n = _add_players(match_id, team, sq["players"])
    store.run(f"UPDATE matches SET team{team}_name = ?, team{team}_squad = ?, "
              f"team{team}_color = COALESCE(team{team}_color, ?) WHERE id = ?", (sq["name"], squad_id, sq["color"], match_id))
    return n


def _add_players(match_id: int, team: int, players: list[dict]) -> int:
    """Spelers toevoegen die er nog niet zijn (zelfde naam of rugnummer in dit team = al aanwezig)."""
    have = store.all("SELECT name, number FROM players WHERE match_id = ? AND team = ?", (match_id, team))
    names = {(p["name"] or "").strip().lower() for p in have}
    numbers = {str(p["number"]).strip() for p in have if p["number"]}
    n = 0
    for p in players:
        nm, nr = (p["name"] or "").strip(), (str(p["number"]).strip() if p.get("number") else None)
        if not nm or nm.lower() in names or (nr and nr in numbers):
            continue
        store.run("INSERT INTO players (match_id, name, number, team) VALUES (?,?,?,?)", (match_id, nm, nr, team))
        names.add(nm.lower())
        if nr:
            numbers.add(nr)
        n += 1
    return n


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
    m["score"] = shots.score(store, match_id)
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
    if "pitch_length" in data or "pitch_width" in data:
        set_pitch_size(match_id, data.get("pitch_length"), data.get("pitch_width"))
    if "half_length" in data and data["half_length"] is not None:
        data = {**data, "half_length": int(data["half_length"])}
    for k in ("wizard_step", "wizard_done"):
        if k in data and data[k] is not None:
            data = {**data, k: int(data[k])}
    return _update("matches", match_id, data, {"name", "date", "team0_name", "team1_name", "team0_color", "team1_color",
                                                "half_length", "wizard_step", "wizard_done"})


def set_pitch_size(match_id: int, length: float | None, width: float | None) -> pitch.Geometry:
    """Veldmaten wijzigen. Kalibratiepunten (op naam) en camerapositie schuiven mee; de automatisch
    bijgestelde sleutelframes worden opnieuw gemaakt."""
    m = _get("matches", match_id)
    old = pitch.of_match(m)
    try:
        new = pitch.geometry(float(length) if length else None, float(width) if width else None)
    except (TypeError, ValueError) as e:
        raise HTTPException(400, "Ongeldige veldmaten") from e
    if not (40 <= new.length <= 130 and 25 <= new.width <= 100 and new.length > new.width):
        raise HTTPException(400, "Veldmaten kloppen niet: lengte 40-130 m, breedte 25-100 m, lengte groter dan breedte")
    if new == old:
        return new
    store.run("UPDATE matches SET pitch_length = ?, pitch_width = ? WHERE id = ?", (new.length, new.width, match_id))
    _remember_venue(match_id)
    for c in store.all("SELECT * FROM clips WHERE match_id = ?", (match_id,)):
        for kf in store.keyframes(c["id"]):
            if kf.get("auto"):
                store.run("DELETE FROM keyframes WHERE id = ?", (kf["id"],))
            else:
                store.run("UPDATE keyframes SET points = ? WHERE id = ?",
                          (json.dumps(pitch.remap_points(kf["points"], new)), kf["id"]))
        if c.get("cam_x") is not None and c.get("cam_y") is not None:
            x = c["cam_x"] * new.length / old.length
            y = c["cam_y"]
            y = new.width + (y - old.width) if y > old.width else (y if y < 0 else y * new.width / old.width)
            store.run("UPDATE clips SET cam_x = ?, cam_y = ? WHERE id = ?", (round(x, 1), round(y, 1), c["id"]))
    analytics.invalidate()
    for c in store.all("SELECT * FROM clips WHERE match_id = ?", (match_id,)):
        if store.one("SELECT 1 AS x FROM keyframes WHERE clip_id = ? AND auto = 0", (c["id"],)):
            _start_autocalib(c)
    return new


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
        learned = _learn_for_new_clip(match_id, cid)
        created.append({**_get("clips", cid), "learned": learned})
    return created


# --- onthouden wat de gebruiker eerder heeft ingesteld ----------------------------------------
# Zoals een trainer die een veld al kent: de maten van een veld (per GPS-plek) en waar je stond
# (per video vanaf dezelfde plek) hoeven maar één keer ingesteld te worden.

VENUE_RADIUS_M = 150.0
SAME_SPOT_M = 8.0


def _gps_of_match(match_id: int) -> tuple[float, float] | None:
    rows = store.all("SELECT gps_lat, gps_lon FROM clips WHERE match_id = ? AND gps_lat IS NOT NULL", (match_id,))
    if not rows:
        return None
    return float(np.median([r["gps_lat"] for r in rows])), float(np.median([r["gps_lon"] for r in rows]))


def _remember_venue(match_id: int) -> None:
    """Veldmaten van deze wedstrijd onthouden voor deze plek (GPS)."""
    m = store.one("SELECT pitch_length, pitch_width FROM matches WHERE id = ?", (match_id,))
    pos = _gps_of_match(match_id)
    if not m or pos is None or m["pitch_length"] is None:
        return
    for v in store.all("SELECT * FROM venues"):
        if geo.distance_m(pos[0], pos[1], v["lat"], v["lon"]) < VENUE_RADIUS_M:
            store.run("UPDATE venues SET pitch_length = ?, pitch_width = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                      (m["pitch_length"], m["pitch_width"], v["id"]))
            return
    store.run("INSERT INTO venues (lat, lon, pitch_length, pitch_width) VALUES (?,?,?,?)",
              (pos[0], pos[1], m["pitch_length"], m["pitch_width"]))


def _learn_for_new_clip(match_id: int, clip_id: int) -> list[str]:
    """Bij een nieuwe video overnemen wat al bekend is. Geeft uitleg terug voor de gebruiker."""
    out = []
    c = _get("clips", clip_id)
    if c.get("gps_lat") is None:
        return out
    m = _get("matches", match_id)
    if m.get("pitch_length") is None:  # veldmaten nog niet ingesteld: kennen we dit veld?
        best = None
        for v in store.all("SELECT * FROM venues WHERE pitch_length IS NOT NULL"):
            d = geo.distance_m(c["gps_lat"], c["gps_lon"], v["lat"], v["lon"])
            if d < VENUE_RADIUS_M and (best is None or d < best[0]):
                best = (d, v)
        if best:
            g = set_pitch_size(match_id, best[1]["pitch_length"], best[1]["pitch_width"])
            out.append(f"Veldmaten {g.length:g} × {g.width:g} m overgenomen: je hebt eerder op dit veld gefilmd")
    if _share_camera(match_id):
        out.append("Je positie is overgenomen van een andere video die je vanaf dezelfde plek filmde")
    return out


def _share_camera(match_id: int) -> int:
    """Video's zonder camerapositie krijgen de positie van een video die vanaf (bijna) dezelfde
    GPS-plek is gefilmd en waar je je plek zelf hebt aangeklikt."""
    clips = store.all("SELECT * FROM clips WHERE match_id = ?", (match_id,))
    known = [c for c in clips if c["cam_source"] == "hand" and c["gps_lat"] is not None and c["cam_x"] is not None]
    n = 0
    for c in clips:
        if c["cam_x"] is not None or c["gps_lat"] is None:
            continue
        near = [(geo.distance_m(c["gps_lat"], c["gps_lon"], k["gps_lat"], k["gps_lon"]), k) for k in known]
        near = [x for x in near if x[0] < SAME_SPOT_M]
        if near:
            k = min(near, key=lambda x: x[0])[1]
            store.run("UPDATE clips SET cam_x = ?, cam_y = ?, cam_h = ?, cam_source = 'kopie' WHERE id = ?",
                      (k["cam_x"], k["cam_y"], k["cam_h"], c["id"]))
            analytics.invalidate(clip_id=c["id"])
            n += 1
    return n


def _store_probe(cid: int, path: Path, info: dict, inherit: dict | None = None) -> None:
    """Videogegevens opslaan; bij knippen GPS en camerapositie van het origineel overnemen."""
    extra = {k: info.get(k) for k in ("gps_lat", "gps_lon", "gps_acc", "device", "rec_start")}
    if inherit:
        for k in ("gps_lat", "gps_lon", "gps_acc", "device", "cam_x", "cam_y", "cam_h", "cam_source"):
            if extra.get(k) is None:
                extra[k] = inherit.get(k)
    sets = ", ".join(f"{k} = ?" for k in extra)
    store.run(f"UPDATE clips SET path = ?, fps = ?, width = ?, height = ?, duration = ?, {sets} WHERE id = ?",
              (str(path), info["fps"], info["width"], info["height"], info["duration"], *extra.values(), cid))


@app.patch("/api/clips/{clip_id}")
def update_clip(clip_id: int, data: dict = Body(...)):
    if "flip" in data and data["flip"] is not None:
        data = {**data, "flip": 1 if data["flip"] else 0}
    if "train_ok" in data:
        data = {**data, "train_ok": 1 if data["train_ok"] else 0}
    return _update("clips", clip_id, data, {"order_idx", "period", "start_minute", "flip", "train_ok"})


@app.delete("/api/clips/{clip_id}")
def delete_clip(clip_id: int):
    if _get("clips", clip_id)["status"] in ("wachtrij", "preview", "analyse"):
        raise HTTPException(409, "Deze video wordt nog geanalyseerd; verwijder hem als de analyse klaar is")
    store.run("DELETE FROM clips WHERE id = ?", (clip_id,))
    shutil.rmtree(config.CLIPS_DIR / str(clip_id), ignore_errors=True)
    analytics.invalidate()
    return {"ok": True}


@app.post("/api/clips/{clip_id}/process")
def process(clip_id: int, data: dict = Body(default={})):
    """Analyse starten. data.mode: 'nauwkeurig' (standaard) of 'snel'."""
    c = _get("clips", clip_id)
    if c["status"] in ("wachtrij", "preview", "analyse"):
        raise HTTPException(409, "Clip wordt al verwerkt")
    if data.get("mode") in ("nauwkeurig", "snel"):
        store.run("UPDATE clips SET analysis_mode = ? WHERE id = ?", (data["mode"], clip_id))
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
def clip_frame(clip_id: int, t: float = 0.0, w: int | None = None):
    c = _get("clips", clip_id)
    try:
        frame, ft = read_frame(Path(c["path"]), _snap_time(clip_id, t))
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    if w and w < frame.shape[1]:  # klein plaatje (voorvertoning)
        frame = cv2.resize(frame, None, fx=w / frame.shape[1], fy=w / frame.shape[1], interpolation=cv2.INTER_AREA)
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
            try:
                kf["error_m"] = round(fit_calibration(kf["points"], camera=kf_camera(prior, kf))[1], 2)
            except ValueError as e:  # bijv. een oude kalibratie die gespiegeld bleek: zeg wat er mis is
                kf["error_m"], kf["problem"] = None, str(e)
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
    yaw = clip.get("cam_yaw")  # aangegeven kijkrichting: hoort bij dít moment
    try:
        _, err = fit_calibration(points, camera=camera_prior(clip, with_yaw=True))
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    data["t"] = _snap_time(clip_id, float(data["t"]))
    if data.get("id"):
        store.run("UPDATE keyframes SET t = ?, points = ?, source = NULL, yaw = COALESCE(?, yaw) WHERE id = ? AND clip_id = ?",
                  (data["t"], json.dumps(points), yaw, data["id"], clip_id))  # zelf bijgesteld: nu jouw eigen kalibratie
        kid = data["id"]
    else:
        kid = store.run("INSERT INTO keyframes (clip_id, t, points, yaw) VALUES (?,?,?,?)",
                        (clip_id, data["t"], json.dumps(points), yaw))
    # opgeslagen bij het ijkmoment; voor een volgend moment (na zwenken) geef je hem zo nodig opnieuw aan
    store.run("UPDATE clips SET calib_review = NULL, cam_yaw = NULL WHERE id = ?", (clip_id,))
    analytics.invalidate(clip_id=clip_id)
    _start_autocalib(clip)
    return {"id": kid, "t": data["t"], "error_m": round(err, 2)}


@app.post("/api/clips/{clip_id}/calibrate-preview")
def calibrate_preview(clip_id: int, data: dict = Body(...)):
    """Live voorbeeld tijdens het klikken (met cameramodel): veld -> beeld, fout en camera."""
    clip = _get("clips", clip_id)
    points = data.get("points") or []
    prior = camera_prior(clip, with_yaw=True)
    try:
        K, err = fit_calibration(points, camera=prior)
    except (ValueError, np.linalg.LinAlgError) as e:
        return {"ok": False, "message": str(e)}
    cam, goals = None, []
    if prior is not None:
        try:
            full = fit_camera(points, prior)[1]
            cam = {k: round(v, 1) for k, v in full.items() if k != "params"}
            if err >= 1:  # past niet goed: zoek de klik die niet bij de rest past
                cam["suspect"] = suspect_point(full["params"], points, prior,
                                               (int(clip["width"]), int(clip["height"])))
            goals = _goal_outlines(full["params"], (int(clip["width"]), int(clip["height"])),
                                   pitch.of_match(_get("matches", clip["match_id"])))
        except ValueError:
            pass
    H = np.linalg.inv(K)
    return {"ok": True, "H": normalize_h(H).tolist(), "error_m": round(err, 2), "camera": cam, "goals": goals}


def _goal_outlines(params, size, geom) -> list[list[list[float]]]:
    """De doelen (palen + lat) in beeld volgens het cameramodel, om de kalibratie te controleren."""
    from .calibration import camera_projection, project_3d

    P = camera_projection(np.asarray(params, float), size)
    out = []
    for side in ("links", "rechts"):
        a, b = geom.elevated_lines[f"Lat {side}"]
        frame = [(a[0], a[1], 0.0), a, b, (b[0], b[1], 0.0)]
        img = project_3d(P, frame)
        if np.isfinite(img).all():
            out.append([[round(float(x), 1), round(float(y), 1)] for x, y in img])
    return out


@app.patch("/api/clips/{clip_id}/camera")
def set_camera(clip_id: int, data: dict = Body(...)):
    """Waar stond de camera? data: {x, y, h, yaw, source: 'hand'|'gps'} (x/y null = wissen).
    yaw: kijkrichting in graden op de veldtekening (0 = naar rechts, 90 = naar onderen), null = onbekend."""
    _get("clips", clip_id)
    fields = {}
    for k in ("x", "y", "h", "yaw"):
        if k in data:
            fields[f"cam_{k}"] = None if data[k] is None else float(data[k])
    if "x" in data and data["x"] is None:
        fields["cam_yaw"] = None  # plek gewist: kijkrichting ook
    if "source" in data:
        fields["cam_source"] = data["source"]
    if fields:
        sets = ", ".join(f"{k} = ?" for k in fields)
        store.run(f"UPDATE clips SET {sets} WHERE id = ?", (*fields.values(), clip_id))
        analytics.invalidate(clip_id=clip_id)
        if data.get("source") == "hand":
            _share_camera(_get("clips", clip_id)["match_id"])
    return _get("clips", clip_id)


@app.post("/api/clips/{clip_id}/camera/gps")
def camera_from_gps(clip_id: int, data: dict = Body(default={})):
    """Zoek het voetbalveld bij de GPS-positie van de video (OpenStreetMap) en zet de camera daar.

    data.adopt_size: ook de veldmaten uit OpenStreetMap overnemen voor deze wedstrijd."""
    c = _get("clips", clip_id)
    if c.get("gps_lat") is None:
        raise HTTPException(400, "Deze video bevat geen GPS-positie")
    try:
        polys = geo.query_pitches(c["gps_lat"], c["gps_lon"])
    except geo.OsmError as e:
        raise HTTPException(502, f"OpenStreetMap niet bereikbaar: {e}. Of klik je positie zelf aan.") from e
    except Exception as e:  # noqa: BLE001
        logging.exception("OpenStreetMap")
        raise HTTPException(502, f"OpenStreetMap niet bereikbaar ({e}). Klik je positie dan zelf aan.") from e
    match = _get("matches", c["match_id"])
    try:
        pos = geo.camera_on_pitch(c["gps_lat"], c["gps_lon"], polys, pitch.of_match(match))
        if data.get("adopt_size"):
            g = set_pitch_size(c["match_id"], pos["pitch_length"], pos["pitch_width"])
            pos = geo.camera_on_pitch(c["gps_lat"], c["gps_lon"], polys, g)
    except ValueError as e:
        raise HTTPException(404, f"{e}. Klik je positie zelf aan op de veldtekening.") from e
    store.run("UPDATE clips SET cam_x = ?, cam_y = ?, cam_h = COALESCE(cam_h, 1.6), cam_source = 'gps' WHERE id = ?",
              (pos["x"], pos["y"], clip_id))
    analytics.invalidate(clip_id=clip_id)
    g = pitch.of_match(_get("matches", c["match_id"]))
    return {**pos, "clip": _get("clips", clip_id), "match_pitch": {"length": g.length, "width": g.width}}


@app.get("/api/clips/{clip_id}/propose")
def propose_calibration(clip_id: int, t: float = 0.0):
    """Automatisch voorstel: zoek het veld in dit beeld vanaf de bekende camerapositie."""
    from .autocalib import propose

    clip = _get("clips", clip_id)
    prior = camera_prior(clip, with_yaw=True)
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
    geom = pitch.of_match(_get("matches", clip["match_id"]))
    res = propose(frame, prior, boxes, geom=geom)
    if res is None:  # lastig beeld (versleten lijnen, fel zonlicht): ook naar zwakke lijnen zoeken
        res = propose(frame, prior, boxes, geom=geom, enhance=True)
    if res is None:
        return {"ok": False, "t": ft, "message": "Geen overtuigend voorstel gevonden in dit beeld. Kies een moment met meer "
                                                  "veldlijnen in beeld, of klik zelf 1 punt + 1 lijn aan."}
    H, info = res
    W, Hh = int(clip["width"]), int(clip["height"])
    to_img = analytics.h_to_image(H)
    # veldpunten in beeld, plus die net buiten beeld (die tellen gewoon mee en maken de kalibratie
    # stevig), plus punten op de lijnen die in beeld zijn
    points = _spread(analytics.image_landmarks(to_img, W, Hh, line_points=False, margin=0.02, geom=geom), 8)
    if len(points) < 6:
        names = {p["name"] for p in points}
        extra = [p for p in analytics.image_landmarks(to_img, W, Hh, line_points=False, margin=0.6, geom=geom)
                 if p["name"] not in names]
        points += _spread(extra, 6 - len(points))
    if len(points) < 4:
        points += [p for p in analytics.image_landmarks(to_img, W, Hh, margin=0.0, geom=geom) if p.get("line")][:6]
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
        raise HTTPException(400, "Leg eerst zelf één ijkmoment vast (Kalibratie)")
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
        r["calib_suspect"] = bool(d is not None and d.calib_suspect)
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
    _learn_team_colors(clip_id)
    return res


@app.patch("/api/clips/{clip_id}/tracks/{track_id}")
def update_track(clip_id: int, track_id: int, data: dict = Body(...)):
    """Team of speler van een track aanpassen. team 3 = toeschouwer: telt nergens meer mee; het
    antwoord bevat dan ook vergelijkbare personen ("similar") om in één keer mee weg te halen."""
    fields = {k: data[k] for k in ("team", "player_id") if k in data}
    if not fields:
        raise HTTPException(400, "Niets te wijzigen")
    if fields.get("team") == teams.TEAM_SPECTATOR:
        fields["player_id"] = None
    sets = ", ".join(f"{k} = ?" for k in fields)
    store.run(f"UPDATE tracks SET {sets} WHERE clip_id = ? AND track_id = ?",
              (*fields.values(), clip_id, track_id))
    analytics.invalidate(geometry=False)
    out = store.one("SELECT * FROM tracks WHERE clip_id = ? AND track_id = ?", (clip_id, track_id))
    if "team" in fields:
        _learn_team_colors(clip_id)
    if fields.get("team") == teams.TEAM_SPECTATOR:
        out["similar"] = analytics.similar_spectators(store, clip_id, track_id)
    return out


@app.post("/api/tracks/spectators")
def mark_spectators(data: dict = Body(...)):
    """Meerdere tracks tegelijk als toeschouwer aanwijzen (of terugzetten). data: {items: [{clip_id,
    track_id}], undo?: bool}. Terugzetten geeft ze weer het automatisch bepaalde team."""
    items = [(int(i["clip_id"]), int(i["track_id"])) for i in data.get("items") or []]
    with store.tx() as c:
        for cid, tid in items:
            if data.get("undo"):
                c.execute("UPDATE tracks SET team = team_auto WHERE clip_id = ? AND track_id = ? AND team = ?",
                          (cid, tid, teams.TEAM_SPECTATOR))
            else:
                c.execute("UPDATE tracks SET team = ?, player_id = NULL WHERE clip_id = ? AND track_id = ?",
                          (teams.TEAM_SPECTATOR, cid, tid))
    analytics.invalidate(geometry=False)
    return {"ok": True, "n": len(items)}


def _learn_team_colors(clip_id: int) -> None:
    """Leren van correcties: de teamkleuren van de wedstrijd (en van de vaste selectie, als die
    gekoppeld is) volgen wat de gebruiker heeft ingedeeld. Nieuwe video's van deze wedstrijd en
    volgende wedstrijden met dezelfde selectie beginnen dan met de goede kleuren."""
    c = store.one("SELECT match_id FROM clips WHERE id = ?", (clip_id,))
    if not c:
        return
    m = _get("matches", c["match_id"])
    colors = _team_colors(c["match_id"])
    for team in (0, 1):
        if not colors[team]:
            continue
        store.run(f"UPDATE matches SET team{team}_color = ? WHERE id = ?", (colors[team], m["id"]))
        if m.get(f"team{team}_squad"):
            store.run("UPDATE squads SET color = ? WHERE id = ?", (colors[team], m[f"team{team}_squad"]))


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


@app.get("/api/players/{player_id}/suggestions")
def player_suggestions(player_id: int, clip_id: int | None = None, limit: int = 8):
    """Koppel-assistent: tracks die waarschijnlijk ook bij deze speler horen."""
    _get("players", player_id)
    return analytics.suggest_tracks(store, player_id, clip_id, min(30, max(1, limit)))


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


@app.get("/api/matches/{match_id}/highlights")
def match_highlights(match_id: int):
    """Mogelijke hoogtepunten uit het geluid (gejuich, fluitsignalen), per video."""
    from . import audio
    from .pipeline import ffmpeg_exe

    out = []
    for c in store.all("SELECT * FROM clips WHERE match_id = ? AND status = 'klaar' ORDER BY order_idx, id", (match_id,)):
        try:
            for e in audio.clip_events(c, clip_dir(c["id"]), ffmpeg_exe()):
                out.append({**e, "clip_id": c["id"]})
        except Exception:  # noqa: BLE001
            logging.exception("Geluid van clip %s", c["id"])
    return out


# --- schoten en goals --------------------------------------------------------------------

_SHOT_FIELDS = {"t", "t_end", "status", "goal", "on_target", "team", "player_id", "x", "y", "goal_x", "auto"}


def _shot_payload(data: dict) -> dict:
    out = {k: data[k] for k in _SHOT_FIELDS if k in data}
    for k in ("goal", "on_target", "auto"):
        if k in out and out[k] is not None:
            out[k] = 1 if out[k] else 0
    if out.get("status") not in (None, "bevestigd", "afgewezen"):
        raise HTTPException(400, "Onbekende status")
    if out.get("goal"):
        out["on_target"] = 1  # een goal is altijd op doel
    return out


@app.get("/api/matches/{match_id}/shots")
def get_shots(match_id: int):
    _get("matches", match_id)
    return shots.overview(store, match_id)


@app.post("/api/matches/{match_id}/shots")
def add_shot(match_id: int, data: dict = Body(...)):
    """Een schot of goal opslaan: een bevestigd (of afgewezen) voorstel, of zelf toegevoegd in de video."""
    c = _get("clips", int(data["clip_id"]))
    if c["match_id"] != match_id:
        raise HTTPException(400, "Video hoort niet bij deze wedstrijd")
    fields = {"status": "bevestigd", **_shot_payload(data)}
    if "t" not in fields:
        raise HTTPException(400, "Tijd ontbreekt")
    if fields.get("x") is None and fields["status"] == "bevestigd":  # zelf toegevoegd: waar ongeveer?
        where = analytics.position_at(store, c["id"], float(fields["t"]), fields.get("player_id"))
        if where is not None:
            fields["x"], fields["y"] = where
    cols = ["match_id", "clip_id", *fields]
    sid = store.run(f"INSERT INTO shots ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                    (match_id, c["id"], *fields.values()))
    return _get("shots", sid)


@app.patch("/api/shots/{shot_id}")
def update_shot(shot_id: int, data: dict = Body(...)):
    _get("shots", shot_id)
    fields = _shot_payload(data)
    if "goal" in fields and not fields["goal"] and "on_target" not in data:
        fields.pop("on_target", None)
    if fields:
        sets = ", ".join(f"{k} = ?" for k in fields)
        store.run(f"UPDATE shots SET {sets} WHERE id = ?", (*fields.values(), shot_id))
    return _get("shots", shot_id)


@app.delete("/api/shots/{shot_id}")
def delete_shot(shot_id: int):
    store.run("DELETE FROM shots WHERE id = ?", (shot_id,))
    return {"ok": True}


@app.get("/api/clips/{clip_id}/ball")
def get_ball(clip_id: int, t0: float = 0.0, t1: float = 1e9):
    return analytics.clip_ball(store, clip_id, t0, t1)


@app.post("/api/clips/{clip_id}/ball")
def set_ball(clip_id: int, data: dict = Body(...)):
    """De bal zelf aanwijzen: data {t, x, y} in beeldpixels, of {t, x: null} = hier is geen bal.
    data {t, clear: true} haalt een eigen aanwijzing weer weg."""
    _get("clips", clip_id)
    t = _snap_time(clip_id, float(data["t"]))
    if data.get("clear"):
        store.run("DELETE FROM ball_manual WHERE clip_id = ? AND ABS(t - ?) < 0.02", (clip_id, t))
    else:
        x, y = data.get("x"), data.get("y")
        store.run("INSERT OR REPLACE INTO ball_manual (clip_id, t, x, y) VALUES (?,?,?,?)",
                  (clip_id, t, None if x is None else float(x), None if y is None else float(y)))
    analytics.invalidate(clip_id=clip_id)
    return {"ok": True, "t": t}


# --- de app slimmer maken (trainen op je eigen beelden) ------------------------------------------

trainer = training.Trainer()


def _after_training(kind: str, res: dict) -> None:
    if kind == "field":
        learning._field_cache.clear()
    analytics.invalidate(geometry=kind == "field")


trainer.on_done.append(_after_training)


@app.get("/api/training")
def training_overview():
    return {**training.overview(store), "status": trainer.status()}


@app.get("/api/training/status")
def training_status():
    return trainer.status()


@app.post("/api/training/start")
def training_start(data: dict = Body(...)):
    """data: {kind: 'detector'|'field'|'players', quick?: bool, match_id?, player_id?}"""
    kind = data.get("kind")
    if kind not in ("detector", "field", "players"):
        raise HTTPException(400, "Onbekende soort training")
    opts = {"quick": bool(data.get("quick", True))}
    if kind == "players":
        if not data.get("match_id"):
            raise HTTPException(400, "Kies een wedstrijd")
        opts["match_id"] = int(data["match_id"])
        if data.get("player_id"):
            opts["player_id"] = int(data["player_id"])
    try:
        return trainer.start(kind, opts)
    except RuntimeError as e:
        raise HTTPException(409, str(e)) from e


@app.post("/api/training/stop")
def training_stop():
    trainer.stop()
    return {"ok": True}


@app.post("/api/training/{kind}/activate")
def training_activate(kind: str, data: dict = Body(default={})):
    """Een getrainde versie gebruiken (file), of terug naar het standaardmodel (file: null)."""
    if kind not in learning.KINDS:
        raise HTTPException(400, "Onbekende soort")
    try:
        learning.set_active(kind, data.get("file"))
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    _after_training(kind, {})
    return {"ok": True, "active": learning.registry()[kind]["active"]}


@app.get("/api/matches/{match_id}/recognize")
def recognize_players(match_id: int):
    """Ongekoppelde stukken die volgens de spelerprofielen bij een speler horen."""
    _get("matches", match_id)
    return learning.recognize(store, match_id)


@app.get("/api/matches/{match_id}/profiles")
def match_profiles(match_id: int):
    """Welke spelers van deze wedstrijd al een profiel hebben."""
    out = {}
    for p in store.all("SELECT * FROM players WHERE match_id = ?", (match_id,)):
        out[p["id"]] = learning.profile_of(store, p) is not None
    return out


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
    if "spotlights" in data:  # spotlight op een of meer spelers
        ids = list(dict.fromkeys(int(p) for p in data["spotlights"] or []))
        out["spotlights"] = json.dumps(ids)
        out["spotlight_player_id"] = ids[0] if ids else None
    elif "spotlight_player_id" in data:
        pid = int(data["spotlight_player_id"]) if data["spotlight_player_id"] else None
        out["spotlight_player_id"] = pid
        out["spotlights"] = json.dumps([pid] if pid else [])
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


@app.get("/api/matches/{match_id}/order")
def order_proposal(match_id: int, half_length: int | None = None):
    """Voorstel voor volgorde, helft en beginminuut van de video's, uit de opnametijd."""
    from . import order
    _get("matches", match_id)
    return order.propose(store, match_id, half_length)


@app.post("/api/matches/{match_id}/order")
def order_apply(match_id: int, data: dict = Body(...)):
    from . import order
    _get("matches", match_id)
    if data.get("half_length"):
        store.run("UPDATE matches SET half_length = ? WHERE id = ?", (int(data["half_length"]), match_id))
    n = order.apply(store, match_id, data.get("items") or [])
    analytics.invalidate()
    return {"updated": n}


# --- standplaatsen: één keer kalibreren per plek, de rest automatisch ------------------------

@app.get("/api/matches/{match_id}/stations")
def get_stations(match_id: int):
    from . import stations
    m = _get("matches", match_id)
    out = []
    for st in stations.group(store, match_id):
        a = stations.anchor(store, st)
        clips = []
        for c in st["clips"]:
            kfs = store.keyframes(c["id"])
            own = [k for k in kfs if not k.get("auto") and k.get("source") != "standplaats"]
            prop = [k for k in kfs if k.get("source") == "standplaats"]
            state = ("eigen" if own else c.get("calib_review") if prop or c.get("calib_review") in ("mislukt", "afgekeurd")
                     else "niet_geanalyseerd" if c["status"] != "klaar" else "open")
            scores = [k["score"] if isinstance(k.get("score"), dict) else json.loads(k.get("score") or "{}") for k in prop]
            zwaai = [float(x["zwaai"]) for x in scores if x.get("zwaai") is not None]
            clips.append({"id": c["id"], "filename": c["filename"], "status": c["status"], "state": state,
                          "uncertain": bool(zwaai) and min(zwaai) > stations.TRUST_DEG,
                          "turn_deg": round(min(zwaai)) if zwaai else None,
                          "method": c.get("calib_method"), "gps_acc": c.get("gps_acc"), "rec_start": c.get("rec_start"),
                          "period": c.get("period"), "start_minute": c.get("start_minute"),
                          "has_calibration": bool(own or prop), "calib_status": c.get("calib_status")})
        cam = None
        if a is not None:
            p = a[1]
            cam = {"clip_id": a[0]["id"], "filename": a[0]["filename"], "x": round(float(p[0]), 1), "y": round(float(p[1]), 1),
                   "h": round(float(p[2]), 1)}
        out.append({"id": st["id"], "label": st["label"], "period": st["period"], "acc_m": st["acc_m"], "precision_m": st["precision_m"],
                    "no_gps": st["no_gps"], "anchor": cam, "clips": clips})
    return {"stations": out, "job": {"status": m.get("stations_status"), "progress": m.get("stations_progress"),
                                     "message": m.get("stations_message")}}


@app.get("/api/matches/{match_id}/stations/{station_id}/best")
def station_best_clip(match_id: int, station_id: int):
    """De video van deze standplaats met het meeste veld in beeld (om zelf te kalibreren)."""
    from . import stations
    st = next((x for x in stations.group(store, match_id) if x["id"] == station_id), None)
    if st is None:
        raise HTTPException(404, "Standplaats niet gevonden")
    c = stations.best_clip(store, st)
    if c is None:
        raise HTTPException(404, "Geen video gevonden")
    return _get("clips", c["id"])


@app.get("/api/matches/{match_id}/timeline")
def timeline_guess(match_id: int):
    from . import order
    _get("matches", match_id)
    return order.timeline_guess(store, match_id)


@app.post("/api/matches/{match_id}/timeline")
def timeline_apply(match_id: int, data: dict = Body(...)):
    """data: {kickoff, second_clip, kickoff2, half_length} (tijden in seconden sinds 1970)."""
    from . import order
    _get("matches", match_id)
    num = lambda k: float(data[k]) if data.get(k) is not None else None  # noqa: E731
    n = order.apply_timeline(store, match_id, num("kickoff"), int(data["second_clip"]) if data.get("second_clip") else None,
                             num("kickoff2"), int(data.get("half_length") or 45))
    analytics.invalidate()
    return {"updated": n}


@app.post("/api/matches/{match_id}/stations/run")
def run_stations(match_id: int):
    _get("matches", match_id)
    if worker is None:
        raise HTTPException(503, "De achtergrondverwerking draait niet")
    worker.submit_stations(match_id)
    return {"ok": True}


@app.post("/api/clips/{clip_id}/calib-review")
def calib_review(clip_id: int, data: dict = Body(...)):
    """Een automatisch voorgestelde kalibratie goedkeuren of afkeuren."""
    c = _get("clips", clip_id)
    status = data.get("status")
    if status not in ("goedgekeurd", "afgekeurd"):
        raise HTTPException(400, "Kies goedgekeurd of afgekeurd")
    if status == "afgekeurd":
        store.run("DELETE FROM keyframes WHERE clip_id = ? AND (source = 'standplaats' OR (auto = 1 AND COALESCE(accepted, 0) = 0))",
                  (clip_id,))
    store.run("UPDATE clips SET calib_review = ? WHERE id = ?", (status, clip_id))
    analytics.invalidate(clip_id=c["id"])
    return {"ok": True}


@app.get("/api/clips/{clip_id}/calib-thumb")
def calib_thumb(clip_id: int, w: int = 480):
    """Klein beeld met de veldlijnen erover zoals de kalibratie ze legt (om snel te controleren)."""
    from .autocalib import pitch_samples
    from .calibration import fit_keyframe
    c = _get("clips", clip_id)
    kfs = [k for k in store.keyframes(clip_id) if not k.get("auto")]
    if not kfs:
        raise HTTPException(404, "Nog niet gekalibreerd")
    kf = kfs[0]
    try:
        K, _ = fit_keyframe(kf, camera=camera_prior(c))
        frame, _ = read_frame(Path(c["path"]), kf["t"])
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    geom = pitch.of_match(_get("matches", c["match_id"]))
    G = np.linalg.inv(K)
    pts = pitch_samples(step=0.2, geom=geom)
    hom = np.hstack([pts, np.ones((len(pts), 1))]) @ G.T
    front = hom[:, 2] > 1e-9
    img = hom[front, :2] / hom[front, 2:3]
    Hh, W = frame.shape[:2]
    inside = (img[:, 0] >= 0) & (img[:, 0] < W) & (img[:, 1] >= 0) & (img[:, 1] < Hh)
    r = max(2, W // 640)
    for x, y in img[inside].astype(int):
        cv2.circle(frame, (int(x), int(y)), r, (0, 214, 255), -1, cv2.LINE_AA)
    s = min(1.0, w / W)
    small = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return Response(buf.tobytes(), media_type="image/jpeg", headers={"Cache-Control": "no-store"})


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
        if c.get("rec_start") is not None:  # het deel begon zoveel later dan de opname
            store.run("UPDATE clips SET rec_start = ? WHERE id = ?", (c["rec_start"] + float(seg["start"]), cid))
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
