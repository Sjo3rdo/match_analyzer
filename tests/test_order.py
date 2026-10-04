"""Volgorde en verloop van de wedstrijd uit de opnametijd van de video's."""
import importlib

import pytest


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("MATCH_ANALYZER_DATA", str(tmp_path))
    from app import config
    importlib.reload(config)
    from app import storage, analytics, pipeline, main
    for m in (storage, analytics, pipeline, main):
        importlib.reload(m)
    from fastapi.testclient import TestClient
    return main, TestClient(main.app)


def test_parse_iphone_and_ffmpeg_times():
    from app.geo import parse_time
    a = parse_time("2026-10-02T14:03:11+0200")  # iPhone (com.apple.quicktime.creationdate)
    b = parse_time("2026-10-02T12:03:11.000000Z")  # ffmpeg creation_time
    assert a == b and parse_time("onzin") is None


def test_order_from_recording_time_finds_half_time(env):
    main, client = env
    st = main.store
    mid = client.post("/api/matches", json={"name": "Verloop"}).json()["id"]
    t0 = 1_790_000_000.0
    # (bestandsnaam, start t.o.v. de eerste opname in s, duur in s); geüpload in een rommelige volgorde
    clips = [("c.mov", 20 * 60, 60), ("a.mov", 0, 90), ("e.mov", 62 * 60, 30), ("b.mov", 5 * 60, 40),
             ("d.mov", 55 * 60, 45), ("x.mov", None, 10)]
    for i, (name, off, dur) in enumerate(clips):
        st.run("INSERT INTO clips (match_id, filename, path, order_idx, duration, rec_start, status) VALUES (?,?,?,?,?,?,'klaar')",
               (mid, name, "", i, dur, None if off is None else t0 + off))
    r = client.get(f"/api/matches/{mid}/order?half_length=40").json()
    got = [(it["filename"], it["period"], it["start_minute"]) for it in r["items"]]
    # rust tussen c (20') en d (55'); de 2e helft begint bij 40'
    assert got == [("a.mov", 1, 0.0), ("b.mov", 1, 5.0), ("c.mov", 1, 20.0), ("d.mov", 2, 40.0), ("e.mov", 2, 47.0),
                   ("x.mov", 1, 0.0)]
    assert r["n_without_time"] == 1 and r["break_minutes"] == 34
    assert client.post(f"/api/matches/{mid}/order", json={"items": r["items"], "half_length": 40}).json()["updated"] == 6
    m = client.get(f"/api/matches/{mid}").json()
    assert [c["filename"] for c in m["clips"]] == ["a.mov", "b.mov", "c.mov", "d.mov", "e.mov", "x.mov"]
    assert m["half_length"] == 40 and m["clips"][3]["period"] == 2 and m["clips"][4]["start_minute"] == 47.0
