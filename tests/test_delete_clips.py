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
