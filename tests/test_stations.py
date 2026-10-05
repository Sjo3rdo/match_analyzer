"""Standplaatsen: groeperen op GPS en tijd, en een video automatisch kalibreren vanaf dezelfde plek."""
import importlib
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("MATCH_ANALYZER_DATA", str(tmp_path))
    from app import config
    importlib.reload(config)
    from app import storage, analytics, pipeline, stations, main
    for m in (storage, analytics, pipeline, stations, main):
        importlib.reload(m)
    from fastapi.testclient import TestClient
    return main, TestClient(main.app), tmp_path


def _clip(st, mid, name, **kw):
    cols = {"match_id": mid, "filename": name, "path": "", "order_idx": 0, "status": "klaar", "width": 1280,
            "height": 720, "duration": 3.0, **kw}
    return st.run(f"INSERT INTO clips ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})", tuple(cols.values()))


def test_group_by_gps_accuracy_and_time(env):
    main, client, _ = env
    st = main.store
    mid = client.post("/api/matches", json={"name": "Groepen"}).json()["id"]
    lat0, lon0, t0 = 53.0, 6.0, 1_790_000_000.0
    dlat = 1 / 111_320  # 1 m naar het noorden
    a = _clip(st, mid, "a", gps_lat=lat0, gps_lon=lon0, gps_acc=3, rec_start=t0)
    b = _clip(st, mid, "b", gps_lat=lat0 + 9 * dlat, gps_lon=lon0, gps_acc=12, rec_start=t0 + 60)  # onnauwkeurig: zelfde plek
    c = _clip(st, mid, "c", gps_lat=lat0 + 60 * dlat, gps_lon=lon0, gps_acc=4, rec_start=t0 + 3000)  # 2e helft, andere kant
    d = _clip(st, mid, "d", rec_start=t0 + 3100)  # geen GPS: hoort bij wat er qua tijd vlakbij is
    e = _clip(st, mid, "e")  # niets bekend
    from app import stations
    groups = stations.group(st, mid)
    ids = [[x["id"] for x in g["clips"]] for g in groups]
    assert ids == [[a, b], [c, d], [e]] and groups[2]["no_gps"]
    assert groups[0]["precision_m"] < 3  # gewogen gemiddelde: preciezer dan elke meting los


def _world():
    """Bovenaanzicht van het veld plus omgeving (borden, struiken, paden) op de grond, 4 px per meter."""
    rng = np.random.default_rng(5)
    s, m = 4.0, 40.0
    W, H = int((105 + 2 * m) * s), int((68 + 2 * m) * s)
    img = np.full((H, W, 3), (60, 140, 70), np.uint8)
    for _ in range(900):  # 'omgeving': vlakjes en vormen in allerlei kleuren
        x, y = int(rng.integers(0, W)), int(rng.integers(0, H))
        col = tuple(int(v) for v in rng.integers(20, 255, 3))
        if rng.random() < 0.5:
            cv2.rectangle(img, (x, y), (x + int(rng.integers(4, 30)), y + int(rng.integers(4, 30))), col, -1)
        else:
            cv2.circle(img, (x, y), int(rng.integers(3, 14)), col, -1)
    pitch_rect = (int(m * s), int(m * s), int((m + 105) * s), int((m + 68) * s))
    img[pitch_rect[1]:pitch_rect[3], pitch_rect[0]:pitch_rect[2]] = (60, 145, 70)
    n = cv2.GaussianBlur(rng.integers(0, 30, (H, W), dtype=np.uint8), (3, 3), 0)
    img = cv2.add(img, cv2.merge([n, n, n]))
    from app import autocalib as ac
    for x, y in ac.pitch_samples(step=0.1):
        cv2.circle(img, (int((x + m) * s), int((y + m) * s)), 1, (240, 240, 240), -1)
    T = np.array([[s, 0, m * s], [0, s, m * s], [0, 0, 1.0]])  # veld (m) -> textuur (px)
    return img, T


def _background():
    """Rondom (360 graden) wat je boven de horizon ziet: bomen, huizen, reclameborden."""
    rng = np.random.default_rng(9)
    bg = np.zeros((600, 3600, 3), np.uint8)
    bg[:] = (235, 200, 160)  # lucht
    for _ in range(700):
        x, w = int(rng.integers(0, 3600)), int(rng.integers(8, 60))
        top = int(rng.integers(150, 560))
        col = tuple(int(v) for v in rng.integers(20, 220, 3))
        if rng.random() < 0.6:
            cv2.ellipse(bg, (x, top), (w, int(rng.integers(10, 50))), 0, 0, 360, col, -1)  # boom
        else:
            cv2.rectangle(bg, (x, top), (x + w, 600), col, -1)  # huis, bord
    n = rng.integers(0, 25, bg.shape[:2], dtype=np.uint8)
    return cv2.add(bg, cv2.merge([n, n, n]))


def _render(params, img, T, bg):
    """Beeld zoals een echte camera het ziet: de grond (veld + omgeving) en daarboven de horizon."""
    from app.calibration import camera_projection
    W, H = 1280, 720
    f = math.exp(params[6])
    Kc = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
    P = camera_projection(params, (W, H))
    M = np.linalg.inv(Kc) @ P[:, :3]
    R = M / np.linalg.norm(M[0])
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    rays = np.stack([uu, vv, np.ones_like(uu)], -1).reshape(-1, 3).astype(np.float64) @ np.linalg.inv(Kc).T @ R
    C = np.array([params[0], params[1], -params[2]])
    with np.errstate(divide="ignore", invalid="ignore"):
        s = -C[2] / rays[:, 2]
    ground = (rays[:, 2] > 0) & (s < 250)
    gp = C[:2] + s[:, None] * rays[:, :2]
    tx = (gp @ T[:2, :2].T + T[:2, 2]).astype(np.float32)
    az = np.arctan2(rays[:, 1], rays[:, 0])
    el = np.arctan2(-rays[:, 2], np.hypot(rays[:, 0], rays[:, 1]))
    bx = ((az + np.pi) / (2 * np.pi) * 3600).astype(np.float32)
    by = np.clip((0.45 - el) / 0.5 * 600, 0, 599).astype(np.float32)
    mx = np.where(ground, tx[:, 0], -1).reshape(H, W).astype(np.float32)
    my = np.where(ground, tx[:, 1], -1).reshape(H, W).astype(np.float32)
    out = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderValue=(60, 140, 70))
    sky = cv2.remap(bg, bx.reshape(H, W), by.reshape(H, W), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    return np.where(ground.reshape(H, W, 1), out, sky)


def _video(path, frame, n=30):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (1280, 720))
    for _ in range(n):
        vw.write(frame)
    vw.release()


def test_second_video_from_same_spot_is_calibrated_automatically(env):
    main, client, tmp = env
    st = main.store
    from app import autocalib, stations
    from app.calibration import apply_h, camera_homography, default_focal, fit_keyframe
    mid = client.post("/api/matches", json={"name": "Statief"}).json()["id"]
    img, T = _world()
    bg = _background()
    f = default_focal(1280)

    def cam(yaw):
        return np.array([40.0, 76.0, 3.0, math.radians(yaw), math.radians(9), 0.0, math.log(f)])
    G_a, G_b = camera_homography(cam(-60), (1280, 720)), camera_homography(cam(-80), (1280, 720))
    ids = []
    for name, yaw in (("a.mp4", -60), ("b.mp4", -80)):
        p = tmp / name
        _video(p, _render(cam(yaw), img, T, bg))
        cid = _clip(st, mid, name, path=str(p), gps_lat=53.0, gps_lon=6.0, gps_acc=5)
        with st.tx() as c:
            for i in range(30):
                c.execute("INSERT INTO frames VALUES (?,?,?)", (cid, i, i / 10))
        ids.append(cid)
    # video a: zelf gekalibreerd (zonder positie in te stellen)
    pts = autocalib.keyframe_points(G_a, (1280, 720))
    st.run("INSERT INTO keyframes (clip_id, t, points) VALUES (?,?,?)", (ids[0], 1.5, json.dumps(pts)))
    summary = stations.run(st, mid)
    assert summary["ok"] == 1, summary
    b = st.one("SELECT * FROM clips WHERE id = ?", (ids[1],))
    assert b["calib_review"] == "voorstel" and b["calib_method"] == "omgeving"
    assert abs(b["cam_x"] - 40) < 1.5 and abs(b["cam_y"] - 76) < 1.5 and abs(b["cam_h"] - 3) < 0.7
    kf = [k for k in st.keyframes(ids[1]) if k["source"] == "standplaats"][0]
    K, _ = fit_keyframe(kf)
    probe = np.array([[300.0, 600.0], [900.0, 500.0], [640.0, 650.0]])
    truth = apply_h(np.linalg.inv(G_b), probe)
    assert np.max(np.linalg.norm(apply_h(K, probe) - truth, axis=1)) < 1.0  # minder dan 1 m ernaast
    # overzicht en controle via de API
    r = client.get(f"/api/matches/{mid}/stations").json()
    states = {c["id"]: c["state"] for c in r["stations"][0]["clips"]}
    assert states == {ids[0]: "eigen", ids[1]: "voorstel"} and r["stations"][0]["anchor"]["clip_id"] == ids[0]
    assert client.get(f"/api/clips/{ids[1]}/calib-thumb").headers["content-type"] == "image/jpeg"
    client.post(f"/api/clips/{ids[1]}/calib-review", json={"status": "afgekeurd"})
    assert not [k for k in st.keyframes(ids[1]) if k["source"] == "standplaats"]
