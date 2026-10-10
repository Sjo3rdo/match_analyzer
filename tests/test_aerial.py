"""Luchtfoto: hoeken van het veld aanklikken -> veldmaten, je plek per video, en onthouden per veld."""
import importlib
import math

import pytest

from test_geo import _ll, _pitch_polygon


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


def test_corners_set_size_and_place_every_video_and_are_remembered(env):
    main, client = env
    st = main.store
    lat0, lon0 = 53.2499, 6.3917
    poly, u, v = _pitch_polygon(lat0, lon0, 25, 99.0, 63.0)
    corners = [list(c) for c in poly[:4]]
    mid = client.post("/api/matches", json={"name": "Luchtfoto"}).json()["id"]
    a_ll = _ll(lat0, lon0, -10 * u + 34 * v)  # 2,5 m buiten de zijlijn, 10 m van het midden
    a = _clip(st, mid, "a", gps_lat=a_ll[0], gps_lon=a_ll[1], gps_acc=5)
    b = _clip(st, mid, "b")  # geen GPS
    info = client.get(f"/api/matches/{mid}/aerial").json()
    assert info["corners"] is None and info["center"] == pytest.approx(a_ll) and "{z}" in info["tiles"]

    r = client.post(f"/api/matches/{mid}/pitch-corners", json={"corners": [corners[i] for i in (2, 0, 3, 1)]})
    assert r.status_code == 200, r.text
    j = r.json()
    assert (j["length"], j["width"], j["placed"]) == (99.0, 63.0, 1)
    ca, cb = main._get("clips", a), main._get("clips", b)
    assert ca["cam_x"] == pytest.approx(99 / 2 + 10, abs=0.3) and ca["cam_y"] == pytest.approx(65.5, abs=0.3)
    assert ca["cam_source"] == "gps" and cb["cam_x"] is None

    # zelf aanklikken op de luchtfoto: 1 m verder van de zijlijn
    me = _ll(lat0, lon0, -10 * u + 35 * v)
    c = client.post(f"/api/clips/{a}/camera/latlon", json={"lat": me[0], "lon": me[1]}).json()
    assert c["cam_source"] == "hand" and c["cam_y"] == pytest.approx(66.5, abs=0.3)

    # een volgende wedstrijd op hetzelfde veld: veld en plek komen vanzelf
    mid2 = client.post("/api/matches", json={"name": "Thuis 2"}).json()["id"]
    d_ll = _ll(lat0, lon0, 30 * u + 36 * v)
    d = _clip(st, mid2, "d", gps_lat=d_ll[0], gps_lon=d_ll[1], gps_acc=5)
    assert client.get(f"/api/matches/{mid2}/aerial").json()["known"] == {"length": 99.0, "width": 63.0}
    msgs = main._learn_for_new_clip(mid2, d)
    assert any("luchtfoto" in m for m in msgs), msgs
    m2, cd = main._get("matches", mid2), main._get("clips", d)
    assert (m2["pitch_length"], m2["pitch_width"]) == (99.0, 63.0)
    assert cd["cam_x"] == pytest.approx(99 / 2 - 30, abs=0.3) and cd["cam_y"] == pytest.approx(67.5, abs=0.3)


def test_wrong_corners_are_explained(env):
    main, client = env
    mid = client.post("/api/matches", json={"name": "Fout"}).json()["id"]
    r = client.post(f"/api/matches/{mid}/pitch-corners", json={"corners": [[53, 6], [53, 6.0001], [53.0001, 6]]})
    assert r.status_code == 400 and "4 hoeken" in r.json()["detail"]
    r = client.post(f"/api/clips/999/camera/latlon", json={"lat": 53, "lon": 6})
    assert r.status_code == 404
