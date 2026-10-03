"""Volledige keten met een synthetische video en een nep-detector (geen YOLO nodig)."""
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import os

import cv2
import numpy as np
import pytest

from app.detection import FrameDetections

PX_PER_M, OX, OY = 11.0, 60.0, 20.0  # veld -> beeld (bovenaanzicht)
FPS, SECONDS = 20, 6


def to_img(x, y):
    return OX + x * PX_PER_M, OY + y * PX_PER_M


def player_pos(k, t):
    # speler 1 loopt 6 m/s naar rechts, speler 2 staat stil, speler 3 (ander team) loopt 3 m/s omlaag
    return [(20 + 6 * t, 30), (50, 20), (70, 10 + 3 * t)][k]


def boxes_at(t):
    out = []
    for k in range(3):
        x, y = to_img(*player_pos(k, t))
        out.append([x - 8, y - 40, x + 8, y])
    return np.array(out, float)


def make_video(path):
    rng = np.random.default_rng(0)
    bg = rng.integers(60, 120, (720, 1280, 3), dtype=np.uint8)
    bg[..., 1] = np.clip(bg[..., 1].astype(int) + 60, 0, 255)
    x0, y0 = map(int, to_img(0, 0))
    x1, y1 = map(int, to_img(105, 68))
    cv2.rectangle(bg, (x0, y0), (x1, y1), (255, 255, 255), 2)
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (1280, 720))
    colors = [(0, 0, 220), (0, 0, 220), (220, 0, 0)]  # BGR: rood, rood, blauw
    for i in range(FPS * SECONDS):
        f = bg.copy()
        for b, c in zip(boxes_at(i / FPS), colors):
            cv2.rectangle(f, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), c, -1)
        w.write(f)
    w.release()


class FakeDetector:
    def __init__(self):
        self.calls = 0

    def __call__(self, frame):
        # pipeline analyseert iedere 2e frame (20 fps -> 10 fps)
        t = self.calls * 2 / FPS
        self.calls += 1
        b = boxes_at(t)
        bx, by = to_img(*player_pos(0, t))
        return FrameDetections(b, np.full(3, 0.9), (bx + 4, by - 2, 0.5))


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("MATCH_ANALYZER_DATA", str(tmp_path))
    from app import config
    importlib.reload(config)
    from app import storage, analytics, pipeline, main
    for m in (storage, analytics, pipeline, main):
        importlib.reload(m)
    from fastapi.testclient import TestClient
    return main, TestClient(main.app), tmp_path


def test_full_flow(env):
    main, client, tmp = env
    video = tmp / "wedstrijd.mp4"
    make_video(video)

    m = client.post("/api/matches", json={"name": "Test", "team0_name": "Rood", "team1_name": "Blauw"}).json()
    with video.open("rb") as f:
        r = client.post(f"/api/matches/{m['id']}/clips", files=[("files", ("wedstrijd.mp4", f, "video/mp4"))])
    assert r.status_code == 200, r.text
    clip = r.json()[0]
    assert abs(clip["duration"] - SECONDS) < 0.2

    corners = [(0, 0), (105, 0), (105, 68), (0, 68), (52.5, 34)]
    pts = [{"name": str(c), "img": list(to_img(*c)), "pitch": list(c)} for c in corners]
    r = client.post(f"/api/clips/{clip['id']}/keyframes", json={"t": 0, "points": pts})
    assert r.status_code == 200 and r.json()["error_m"] < 0.01

    from app import pipeline
    pipeline.process_clip(main.store, clip["id"], detector=FakeDetector())
    assert client.get(f"/api/matches/{m['id']}").json()["clips"][0]["status"] == "klaar"

    tracks = client.get(f"/api/clips/{clip['id']}/tracks").json()
    assert len(tracks) == 3 and all(t["valid"] for t in tracks)
    teams = {t["track_id"]: t["team"] for t in tracks}
    assert len(set(teams.values())) == 2  # rood vs blauw

    # speler koppelen
    runner = min(tracks, key=lambda t: t["track_id"])
    p = client.post(f"/api/matches/{m['id']}/players", json={"name": "Loper", "number": "9", "team": runner["team"]}).json()
    client.patch(f"/api/clips/{clip['id']}/tracks/{runner['track_id']}", json={"player_id": p["id"]})

    stats = client.get(f"/api/matches/{m['id']}/stats").json()
    loper = next(r for r in stats["players"] if r["player_id"] == p["id"])
    expected = 6 * (SECONDS - 0.1)
    assert abs(loper["distance_m"] - expected) < 0.1 * expected
    assert abs(loper["max_speed_kmh"] - 21.6) < 2
    assert stats["teams"][runner["team"]]["possession_pct"] == 100.0

    pos = client.get(f"/api/clips/{clip['id']}/positions?t0=0&t1=2").json()
    assert pos["calibrated"] and len(pos["frames"]) >= 15
    assert client.get(f"/api/clips/{clip['id']}/predict?t=3").json()
    # gezette punten bewegen mee met het veld (vaste camera: blijven staan)
    w = client.post(f"/api/clips/{clip['id']}/warp", json={"from_t": 0, "to_t": 3, "points": [[100, 200]]}).json()
    assert w["ok"] and abs(w["points"][0][0] - 100) < 2 and abs(w["points"][0][1] - 200) < 2
    assert client.get(f"/api/matches/{m['id']}/highlights").json() == []  # nepvideo zonder geluid
    # zonder camerapositie geen automatisch voorstel
    assert client.get(f"/api/clips/{clip['id']}/propose?t=1").status_code == 400
    assert client.post(f"/api/clips/{clip['id']}/autocalib/accept", json={}).json()["ok"]
    mm = client.get(f"/api/matches/{m['id']}").json()
    assert mm["clips"][0]["n_manual_keyframes"] == 1 and mm["n_assigned"] == 1

    frame = client.get(f"/api/clips/{clip['id']}/frame?t=1")
    assert frame.headers["content-type"] == "image/jpeg"
    vid = client.get(f"/api/clips/{clip['id']}/video", headers={"Range": "bytes=0-99"})
    assert vid.status_code == 206 and len(vid.content) == 100

    # clip (moment) met spotlight en tekening, exporteren als losse mp4 en als reel/zip
    mo = client.post(f"/api/matches/{m['id']}/moments", json={
        "clip_id": clip["id"], "start": 1.0, "end": 3.0, "label": "Sprint", "players": [p["id"]],
        "spotlight_player_id": p["id"],
        "drawings": [{"t": 2.0, "duration": 1.5, "shapes": [
            {"type": "arrow", "from": [0.2, 0.2], "to": [0.5, 0.5], "color": "#ff0000"},
            {"type": "circle", "center": [0.5, 0.5], "radius": 0.05},
            {"type": "free", "points": [[0.1, 0.1], [0.2, 0.15], [0.3, 0.1]]},
            {"type": "text", "at": [0.6, 0.2], "text": "Goed gelopen!"}]}]}).json()
    assert mo["players"] == [p["id"]] and len(mo["drawings"]) == 1
    mo2 = client.post(f"/api/matches/{m['id']}/moments", json={"clip_id": clip["id"], "start": 4, "end": 5.5}).json()
    assert client.patch(f"/api/moments/{mo2['id']}", json={"end": 4.1}).status_code == 400  # te kort
    assert len(client.get(f"/api/matches/{m['id']}/moments").json()) == 2

    r = client.post(f"/api/matches/{m['id']}/export", json={"name": "loper", "moment_ids": [mo["id"]]})
    assert r.status_code == 200, r.text
    out = tmp / "exports" / r.json()["file"]
    assert abs(_duration(out) - (2.0 + 1.5)) < 0.25  # clip + bevroren beeld
    r = client.post(f"/api/matches/{m['id']}/export", json={"name": "reel", "moment_ids": [mo["id"], mo2["id"]]})
    assert r.status_code == 200, r.text
    assert abs(_duration(tmp / "exports" / r.json()["file"]) - (3.5 + 1.5)) < 0.4
    r = client.post(f"/api/matches/{m['id']}/export", json={"name": "los", "moment_ids": [mo["id"], mo2["id"]], "mode": "zip"})
    assert r.status_code == 200 and r.json()["file"].endswith(".zip")
    assert client.get(r.json()["url"]).status_code == 200


def _duration(path):
    cap = cv2.VideoCapture(str(path))
    n, fps = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return n / fps


def test_split_video(env):
    main, client, tmp = env
    video = tmp / "lang.mp4"
    make_video(video)
    m = client.post("/api/matches", json={"name": "Knip"}).json()
    with video.open("rb") as f:
        clip = client.post(f"/api/matches/{m['id']}/clips", files=[("files", ("lang.mp4", f, "video/mp4"))]).json()[0]
    r = client.post(f"/api/clips/{clip['id']}/split", json={"delete_original": True, "segments": [
        {"start": 0.5, "end": 2.5, "name": "1e helft", "period": 1, "start_minute": 0},
        {"start": 3.0, "end": 5.5, "name": "2e helft", "period": 2, "start_minute": 45}]})
    assert r.status_code == 200, r.text
    parts = r.json()
    assert [c["period"] for c in parts] == [1, 2]
    assert abs(parts[0]["duration"] - 2.0) < 0.6 and abs(parts[1]["duration"] - 2.5) < 0.6
    clips = client.get(f"/api/matches/{m['id']}").json()["clips"]
    assert [c["filename"] for c in clips] == ["lang - 1e helft.mp4", "lang - 2e helft.mp4"]


def test_camera_position_via_gps_and_minimal_calibration(env, monkeypatch):
    import math
    from test_geo import _pitch_polygon
    from app import geo
    from app.calibration import apply_h, camera_homography, default_focal
    main, client, tmp = env
    video = tmp / "zijlijn.mp4"
    make_video(video)
    m = client.post("/api/matches", json={"name": "GPS"}).json()
    with video.open("rb") as f:
        clip = client.post(f"/api/matches/{m['id']}/clips", files=[("files", ("zijlijn.mp4", f, "video/mp4"))]).json()[0]
    assert client.post(f"/api/clips/{clip['id']}/camera/gps").status_code == 400  # geen GPS in deze video

    # GPS toevoegen alsof de iPhone het had opgeslagen: 6 m buiten de zijlijn, 10 m rechts van het midden
    lat0, lon0 = 53.1425, 6.3756
    poly, u, v = _pitch_polygon(lat0, lon0, 25, length=105, width=68)
    cam = 10 * u + 40 * v
    r = 6371000.0
    main.store.run("UPDATE clips SET gps_lat = ?, gps_lon = ?, gps_acc = 2 WHERE id = ?",
                   (lat0 + math.degrees(cam[1] / r), lon0 + math.degrees(cam[0] / (r * math.cos(math.radians(lat0)))), clip["id"]))
    monkeypatch.setattr(geo, "query_pitches", lambda lat, lon, **kw: [poly])
    pos = client.post(f"/api/clips/{clip['id']}/camera/gps").json()
    assert abs(pos["y"] - 74) < 0.5 and abs(abs(pos["x"] - 52.5) - 10) < 0.5
    client.patch(f"/api/clips/{clip['id']}/camera", json={"h": 1.7})

    # beeld van een camera op die plek: 1 punt + 1 lijn moet nu genoeg zijn
    c = client.get(f"/api/matches/{m['id']}").json()["clips"][0]
    true = np.array([c["cam_x"], c["cam_y"], 1.7, math.radians(-60), math.radians(4), 0.0, math.log(default_focal(c["width"]))])
    H = camera_homography(true, (c["width"], c["height"]))
    pts = [{"name": "Strafschopstip rechts", "img": apply_h(H, np.array([[94.0, 34.0]]))[0].tolist(), "pitch": [94.0, 34.0]}]
    for s in (0.6, 0.8):
        w = np.array([s * 105, 68.0])
        pts.append({"name": "Zijlijn onder", "img": apply_h(H, w[None])[0].tolist(), "line": [[0, 68], [105, 68]]})
    prev = client.post(f"/api/clips/{clip['id']}/calibrate-preview", json={"points": pts}).json()
    assert prev["ok"] and prev["error_m"] < 0.5 and prev["camera"]["h"] > 1
    assert client.post(f"/api/clips/{clip['id']}/keyframes", json={"t": 0, "points": pts}).status_code == 200


def test_pitch_size_moves_calibration(env):
    """Andere veldmaten: punten (op naam) en camerapositie schuiven mee."""
    main, client, tmp = env
    m = client.post("/api/matches", json={"name": "Klein veld"}).json()
    cid = main.store.run("INSERT INTO clips (match_id, filename, path, width, height, cam_x, cam_y, cam_h, cam_source) "
                         "VALUES (?, 'v.mp4', '', 1920, 1080, 52.5, 72, 1.6, 'hand')", (m["id"],))
    pts = [{"name": "Hoekvlag rechtsonder", "img": [100, 200], "pitch": [105, 68]},
           {"name": "Middenstip", "img": [300, 200], "pitch": [52.5, 34]},
           {"name": "Zijlijn onder", "img": [500, 300], "line": [[0, 68], [105, 68]]}]
    import json
    main.store.run("INSERT INTO keyframes (clip_id, t, points) VALUES (?, 0, ?)", (cid, json.dumps(pts)))
    assert client.patch(f"/api/matches/{m['id']}", json={"pitch_length": 30, "pitch_width": 20}).status_code == 400
    r = client.patch(f"/api/matches/{m['id']}", json={"pitch_length": 100, "pitch_width": 64})
    assert r.status_code == 200 and r.json()["pitch_length"] == 100
    kf = main.store.keyframes(cid)[0]["points"]
    assert kf[0]["pitch"] == [100, 64] and kf[1]["pitch"] == [50, 32] and kf[2]["line"] == [[0, 64], [100, 64]]
    c = client.get(f"/api/matches/{m['id']}").json()["clips"][0]
    assert c["cam_x"] == 50 and c["cam_y"] == 68  # 4 m buiten de zijlijn blijft 4 m buiten de zijlijn
    p = client.get(f"/api/pitch?match_id={m['id']}").json()
    assert p["length"] == 100 and {"name": "Doelpaal rechts boven", "x": 100, "y": 28.34} in p["landmarks"]


def test_squads_reused_between_matches(env):
    main, client, tmp = env
    m1 = client.post("/api/matches", json={"name": "Week 1", "team0_name": "JO17-1"}).json()
    for nr, nm in (("1", "Kees"), ("9", "Ali"), ("10", "Sam")):
        client.post(f"/api/matches/{m1['id']}/players", json={"name": nm, "number": nr, "team": 0})
    sq = client.post(f"/api/matches/{m1['id']}/save-squad", json={"team": 0}).json()
    assert sq["name"] == "JO17-1" and len(sq["players"]) == 3
    # nieuwe wedstrijd met dezelfde teamnaam: spelers staan er meteen in (aan de kant van team 1)
    m2 = client.post("/api/matches", json={"name": "Week 2", "team0_name": "VV Ander", "team1_name": "jo17-1"}).json()
    ps = client.get(f"/api/matches/{m2['id']}").json()["players"]
    assert sorted(p["name"] for p in ps if p["team"] == 1) == ["Ali", "Kees", "Sam"]
    # nogmaals laden voegt geen dubbele toe; uit een eerdere wedstrijd overnemen kan ook
    r = client.post(f"/api/matches/{m2['id']}/load-squad", json={"team": 1, "squad_id": sq["id"]}).json()
    assert r["added"] == 0
    m3 = client.post("/api/matches", json={"name": "Week 3"}).json()
    r = client.post(f"/api/matches/{m3['id']}/load-squad", json={"team": 0, "from_match": m1["id"], "from_team": 0}).json()
    assert r["added"] == 3 and r["match"]["team0_name"] == "JO17-1"


def test_link_assistant_suggests_continuation(env):
    """Na het koppelen van een stuk stelt de assistent het stuk voor dat er logisch op volgt."""
    main, client, tmp = env
    import json
    m = client.post("/api/matches", json={"name": "Assistent"}).json()
    mid = m["id"]
    st = main.store
    cid = st.run("INSERT INTO clips (match_id, filename, path, width, height, status) VALUES (?, 'v', '', 1280, 720, 'klaar')", (mid,))
    from app.storage import clip_dir
    n = 100
    st.tx  # noqa
    for i in range(n):
        st.run("INSERT INTO frames VALUES (?,?,?)", (cid, i, i / 10))
    np.save(clip_dir(cid) / "motion.npy", np.tile(np.eye(3), (n, 1, 1)))
    # speler A loopt naar rechts: track 1 (0-3 s), dan weg, dan track 2 (3,5-9 s) verder op dezelfde lijn.
    # Teamgenoot B (track 3) loopt ergens anders.
    def box(x, y):
        return (x - 8, y - 40, x + 8, y)
    rows = []
    for i in range(n):
        t = i / 10
        if t <= 3.0:
            rows.append((cid, i, 1, *box(100 + 40 * t, 400), 0.9))
        if t >= 3.5:
            rows.append((cid, i, 2, *box(100 + 40 * t, 400), 0.9))
        rows.append((cid, i, 3, *box(900 - 30 * t, 200), 0.9))
    with st.tx() as c:
        c.executemany("INSERT INTO detections VALUES (?,?,?,?,?,?,?,?)", rows)
        for tid in (1, 2, 3):
            c.execute("INSERT INTO tracks (clip_id, track_id, n_frames, team, team_auto, color) VALUES (?,?,?,?,?,?)",
                      (cid, tid, 40, 0, 0, "#d03030"))
    pa = client.post(f"/api/matches/{mid}/players", json={"name": "A", "team": 0}).json()
    client.patch(f"/api/clips/{cid}/tracks/1", json={"player_id": pa["id"]})
    sug = client.get(f"/api/players/{pa['id']}/suggestions?clip_id={cid}").json()
    assert sug and sug[0]["track_id"] == 2, sug
    assert all(s["track_id"] != 1 for s in sug)


def _synthetic_clip(st, mid, tracks, n=60):
    """Een geanalyseerde video zonder echte beelden: tracks = {tid: (functie t -> (x, y) voeten, kleur)}."""
    from app.storage import clip_dir
    cid = st.run("INSERT INTO clips (match_id, filename, path, width, height, status) VALUES (?, 'v', '', 1280, 720, 'klaar')", (mid,))
    np.save(clip_dir(cid) / "motion.npy", np.tile(np.eye(3), (n, 1, 1)))
    rows = []
    with st.tx() as c:
        for i in range(n):
            c.execute("INSERT INTO frames VALUES (?,?,?)", (cid, i, i / 10))
            for tid, (f, _) in tracks.items():
                x, y = f(i / 10)
                rows.append((cid, i, tid, x - 8, y - 40, x + 8, y, 0.9))
        c.executemany("INSERT INTO detections VALUES (?,?,?,?,?,?,?,?)", rows)
        for tid, (_, col) in tracks.items():
            c.execute("INSERT INTO tracks (clip_id, track_id, n_frames, t_start, t_end, team, team_auto, color) "
                      "VALUES (?,?,?,0,?,0,0,?)", (cid, tid, n, (n - 1) / 10, col))
    return cid


def test_spectator_removed_with_similar_ones(env):
    """Een toeschouwer aanwijzen: hij telt niet meer mee, en de buurman op dezelfde plek met dezelfde
    jas wordt voorgesteld om mee weg te halen; een speler met dezelfde kleur die rondloopt niet."""
    main, client, tmp = env
    mid = client.post("/api/matches", json={"name": "Publiek"}).json()["id"]
    cid = _synthetic_clip(main.store, mid, {
        1: (lambda t: (300, 650), "#303040"),          # toeschouwer, staat stil
        2: (lambda t: (320, 652), "#323242"),          # buurman, staat stil, zelfde jas
        3: (lambda t: (300 + 60 * t, 400), "#303040"),  # speler in donker shirt, loopt
        4: (lambda t: (900, 650), "#d03030"),          # iemand anders, ander shirt
    })
    # zonder kalibratie zou de app stilstaanders al weglaten; maak ze "geldig" om het aanwijzen te testen
    from app import analytics
    orig = analytics._valid_tracks
    analytics._valid_tracks = lambda d, **k: set(d.tracks) or {1, 2, 3, 4}
    try:
        analytics.invalidate()
        r = client.patch(f"/api/clips/{cid}/tracks/1", json={"team": 3}).json()
        assert [s["track_id"] for s in r["similar"]] == [2], r["similar"]
        tracks = {t["track_id"]: t for t in client.get(f"/api/clips/{cid}/tracks").json()}
        assert not tracks[1]["valid"] and tracks[2]["valid"]
        client.post("/api/tracks/spectators", json={"items": r["similar"]})
        tracks = {t["track_id"]: t for t in client.get(f"/api/clips/{cid}/tracks").json()}
        assert not tracks[2]["valid"] and tracks[3]["valid"] and tracks[2]["team"] == 3
        client.post("/api/tracks/spectators", json={"items": r["similar"], "undo": True})
        assert {t["track_id"]: t for t in client.get(f"/api/clips/{cid}/tracks").json()}[2]["valid"]
    finally:
        analytics._valid_tracks = orig


def test_corrections_teach_team_colors(env):
    """Zelf een paar tracks indelen: 'Opnieuw indelen' gebruikt die als voorbeeld, en de kleuren
    gaan mee naar de vaste selectie."""
    main, client, tmp = env
    mid = client.post("/api/matches", json={"name": "Leren", "team0_name": "Wit"}).json()["id"]
    cols = {1: "#e8e8e8", 2: "#e0e0e4", 3: "#2a3a8a", 4: "#24348a", 5: "#dcdce0", 6: "#30409a"}
    cid = _synthetic_clip(main.store, mid, {tid: (lambda t, k=tid: (100 * k + 40 * t, 300 + 10 * k), c) for tid, c in cols.items()})
    sq = client.post(f"/api/matches/{mid}/save-squad", json={"team": 0}).json()
    # de gebruiker zet wit in team 0 en blauw in team 1 (de automatische indeling had alles in 0)
    client.patch(f"/api/clips/{cid}/tracks/1", json={"team": 0})
    main.store.run("UPDATE tracks SET team_auto = 1 WHERE clip_id = ? AND track_id = 1", (cid,))  # = handmatig
    client.patch(f"/api/clips/{cid}/tracks/3", json={"team": 1})
    client.post(f"/api/clips/{cid}/reassign-teams")
    t = {r["track_id"]: r["team"] for r in client.get(f"/api/clips/{cid}/tracks").json()}
    assert t[1] == t[2] == t[5] == 0 and t[3] == t[4] == t[6] == 1, t
    sq2 = next(s for s in client.get("/api/squads").json() if s["id"] == sq["id"])
    assert sq2["color"] and int(sq2["color"][1:3], 16) > 180  # wit-achtig geleerd


def test_venue_and_camera_position_remembered(env):
    main, client, tmp = env
    st = main.store
    m1 = client.post("/api/matches", json={"name": "Thuis 1"}).json()["id"]
    c1 = st.run("INSERT INTO clips (match_id, filename, path, width, height, gps_lat, gps_lon) "
                "VALUES (?, 'a', '', 1920, 1080, 53.2493, 6.3917)", (m1,))
    client.patch(f"/api/matches/{m1}", json={"pitch_length": 100, "pitch_width": 64})
    # tweede video vanaf dezelfde plek: positie zelf aangeklikt in de eerste, overgenomen in de tweede
    c2 = st.run("INSERT INTO clips (match_id, filename, path, width, height, gps_lat, gps_lon) "
                "VALUES (?, 'b', '', 1920, 1080, 53.24932, 6.39172)", (m1,))
    client.patch(f"/api/clips/{c1}/camera", json={"x": 40, "y": 70, "h": 1.6, "source": "hand"})
    c2d = next(c for c in client.get(f"/api/matches/{m1}").json()["clips"] if c["id"] == c2)
    assert c2d["cam_x"] == 40 and c2d["cam_source"] == "kopie"
    # volgende wedstrijd op hetzelfde veld: veldmaten worden overgenomen bij het uploaden
    m2 = client.post("/api/matches", json={"name": "Thuis 2"}).json()["id"]
    c3 = st.run("INSERT INTO clips (match_id, filename, path, width, height, gps_lat, gps_lon) "
                "VALUES (?, 'c', '', 1920, 1080, 53.2494, 6.3919)", (m2,))
    learned = main._learn_for_new_clip(m2, c3)
    assert learned and client.get(f"/api/matches/{m2}").json()["pitch_length"] == 100
    m3 = client.post("/api/matches", json={"name": "Uit"}).json()["id"]
    c4 = st.run("INSERT INTO clips (match_id, filename, path, width, height, gps_lat, gps_lon) "
                "VALUES (?, 'd', '', 1920, 1080, 52.0, 5.0)", (m3,))
    assert main._learn_for_new_clip(m3, c4) == []


def test_manual_ball_and_gap_filling(env):
    main, client, tmp = env
    mid = client.post("/api/matches", json={"name": "Bal"}).json()["id"]
    cid = _synthetic_clip(main.store, mid, {1: (lambda t: (200, 400), "#d03030")}, n=30)
    with main.store.tx() as c:  # bal gevonden in beeld 0-4 en 9-12, ertussen niet
        for i in list(range(0, 5)) + list(range(9, 13)):
            c.execute("INSERT INTO ball VALUES (?,?,?,?,?)", (cid, i, 100 + 10 * i, 300, 0.5))
    corners = [(0, 0), (105, 0), (105, 68), (0, 68)]
    pts = [{"name": str(k), "img": [100 + k[0] * 10, 100 + k[1] * 8], "pitch": list(k)} for k in corners]
    assert client.post(f"/api/clips/{cid}/keyframes", json={"t": 0, "points": pts}).status_code == 200
    b = client.get(f"/api/clips/{cid}/ball").json()
    kinds = {round(x["t"], 1): x["kind"] for x in b}
    assert kinds[0.6] == "geschat" and kinds[0.0] == "gevonden"
    client.post(f"/api/clips/{cid}/ball", json={"t": 2.0, "x": 500, "y": 350})
    client.post(f"/api/clips/{cid}/ball", json={"t": 0.2, "x": None, "y": None})  # daar was geen bal
    b = {round(x["t"], 1): x for x in client.get(f"/api/clips/{cid}/ball").json()}
    assert b[2.0]["kind"] == "hand" and b[2.0]["x"] == 500 and 0.2 not in b
