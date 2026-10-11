"""Alle video's van een wedstrijd in één keer verwijderen."""
import importlib
import json

import pytest


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("MATCH_ANALYZER_DATA", str(tmp_path))
    from app import config
    importlib.reload(config)
    from app import storage, analytics, pipeline, stations, main
    for m in (storage, analytics, pipeline, stations, main):
        importlib.reload(m)
    from fastapi.testclient import TestClient
    return main, TestClient(main.app)


def _clip(st, mid, name, **kw):
    cols = {"match_id": mid, "filename": name, "path": "", "order_idx": 0, "status": "klaar", "width": 1280,
            "height": 720, "duration": 3.0, **kw}
    return st.run(f"INSERT INTO clips ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})", tuple(cols.values()))


def test_delete_all_videos_keeps_the_match_and_a_video_being_analysed(env):
    main, client = env
    st = main.store
    mid = client.post("/api/matches", json={"name": "Opruimen"}).json()["id"]
    other = client.post("/api/matches", json={"name": "Andere"}).json()["id"]
    a = _clip(st, mid, "a.mov")
    _clip(st, mid, "b.mov", status="wachtrij")
    _clip(st, mid, "c.mov", status="analyse")
    keep = _clip(st, other, "d.mov")
    st.run("INSERT INTO keyframes (clip_id, t, points) VALUES (?,?,?)", (a, 0.0, json.dumps([])))
    (main.config.CLIPS_DIR / str(a)).mkdir(parents=True, exist_ok=True)
    client.patch(f"/api/matches/{mid}", json={"pitch_length": 100, "pitch_width": 64})

    r = client.delete(f"/api/matches/{mid}/clips")
    assert r.status_code == 200, r.text
    assert r.json() == {"deleted": 2, "kept": ["c.mov"]}
    m = client.get(f"/api/matches/{mid}").json()
    assert [c["filename"] for c in m["clips"]] == ["c.mov"]
    assert (m["pitch_length"], m["pitch_width"]) == (100, 64)  # de wedstrijd zelf blijft
    assert not st.all("SELECT * FROM keyframes WHERE clip_id = ?", (a,))
    assert not (main.config.CLIPS_DIR / str(a)).exists()
    assert [c["id"] for c in client.get(f"/api/matches/{other}").json()["clips"]] == [keep]


def test_delete_all_videos_waits_for_station_calibration(env):
    main, client = env
    mid = client.post("/api/matches", json={"name": "Bezig"}).json()["id"]
    _clip(main.store, mid, "a.mov")
    main.store.run("UPDATE matches SET stations_status = 'bezig' WHERE id = ?", (mid,))
    r = client.delete(f"/api/matches/{mid}/clips")
    assert r.status_code == 409 and "Wacht" in r.json()["detail"]


def test_reset_calibration_wipes_everything_of_this_match_but_keeps_the_videos(env):
    main, client = env
    st = main.store
    lat, lon = 53.2499, 6.3917
    mid = client.post("/api/matches", json={"name": "Opnieuw"}).json()["id"]
    a = _clip(st, mid, "a.mov", gps_lat=lat, gps_lon=lon, cam_x=60.0, cam_y=70.0, cam_h=1.2, cam_source="hand",
              calib_review="goedgekeurd", duration=10.0)
    b = _clip(st, mid, "b.mov", gps_lat=lat, gps_lon=lon, cam_x=61.0, cam_y=70.0, cam_source="standplaats")
    st.run("INSERT INTO keyframes (clip_id, t, points) VALUES (?,?,?)", (a, 0.0, json.dumps([])))
    st.run("INSERT INTO keyframes (clip_id, t, points, source) VALUES (?,?,?, 'standplaats')", (b, 1.0, json.dumps([])))
    st.run("UPDATE matches SET pitch_length = 100, pitch_width = 64, pitch_corners = '[[1,2]]' WHERE id = ?", (mid,))
    st.run("INSERT INTO venues (lat, lon, pitch_length, pitch_width) VALUES (?,?,?,?)", (lat, lon, 100, 64))
    st.run("INSERT INTO venues (lat, lon, pitch_length, pitch_width) VALUES (?,?,?,?)", (52.0, 5.0, 90, 60))  # ander veld

    r = client.post(f"/api/matches/{mid}/calibration/reset", json={})
    assert r.status_code == 200, r.text
    m = client.get(f"/api/matches/{mid}").json()
    assert len(m["clips"]) == 2 and m["pitch_length"] is None and m["pitch_corners"] is None
    for c in m["clips"]:
        assert c["cam_x"] is None and c["cam_source"] is None and c["calib_review"] is None
    assert not st.all("SELECT * FROM keyframes")
    assert [v["lat"] for v in st.all("SELECT * FROM venues")] == [52.0]  # alleen dit veld vergeten


def test_jobs_and_confidence_tell_what_happens(env):
    main, client = env
    st = main.store
    mid = client.post("/api/matches", json={"name": "Status"}).json()["id"]
    a = _clip(st, mid, "a.mov", calib_status="bezig", calib_progress=0.4)
    _clip(st, mid, "b.mov", status="analyse", progress=0.5)
    st.run("INSERT INTO keyframes (clip_id, t, points, source, score) VALUES (?,?,?, 'standplaats', ?)",
           (a, 1.5, json.dumps([]), json.dumps({"methode": "omgeving", "lijnen": True, "zwaai": 10})))
    j = client.get(f"/api/matches/{mid}/jobs").json()
    assert j["busy"] and j["autocalib"]["running"][0]["filename"] == "a.mov" and j["analysis"]["running"][0]["progress"] == 0.5
    conf = client.get(f"/api/matches/{mid}/confidence").json()
    assert conf[str(a)]["pct"] == 90 and "veldlijnen" in conf[str(a)]["why"]
    assert client.post(f"/api/matches/{mid}/calibration/reset", json={}).status_code == 409  # eerst laten afmaken
