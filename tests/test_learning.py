"""Leren van eigen beelden: voorbeelden kiezen, dataset schrijven, veldlijnen-labels, spelerprofielen
en het trainen in een apart proces (zonder echt YOLO te trainen)."""
import importlib
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_end_to_end import FakeDetector, make_video, to_img  # noqa: E402


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("MATCH_ANALYZER_DATA", str(tmp_path))
    from app import config
    importlib.reload(config)
    from app import storage, analytics, pipeline, learning, main
    for m in (storage, analytics, pipeline, learning, main):
        importlib.reload(m)
    from fastapi.testclient import TestClient
    return main, TestClient(main.app), tmp_path


def _analyzed_clip(main, client, tmp):
    video = tmp / "w.mp4"
    make_video(video)
    m = client.post("/api/matches", json={"name": "Leer"}).json()
    with video.open("rb") as f:
        clip = client.post(f"/api/matches/{m['id']}/clips", files=[("files", ("w.mp4", f, "video/mp4"))]).json()[0]
    corners = [(0, 0), (105, 0), (105, 68), (0, 68), (52.5, 34)]
    pts = [{"name": str(c), "img": list(to_img(*c)), "pitch": list(c)} for c in corners]
    client.post(f"/api/clips/{clip['id']}/keyframes", json={"t": 0, "points": pts})
    from app import pipeline
    pipeline.process_clip(main.store, clip["id"], detector=FakeDetector())
    return m, clip


def test_examples_prefer_hard_balls_and_skip_unknown(env):
    main, client, tmp = env
    from app import learning
    m, clip = _analyzed_clip(main, client, tmp)
    cid = clip["id"]
    assert learning.detector_examples(main.store) == []  # nog niet klaargezet
    client.patch(f"/api/clips/{cid}", json={"train_ok": True})
    st = main.store
    st.run("UPDATE ball SET src = 'zoom' WHERE clip_id = ? AND idx < 5", (cid,))
    st.run("DELETE FROM ball WHERE clip_id = ? AND idx BETWEEN 20 AND 29", (cid,))  # daar onbekend
    client.post(f"/api/clips/{cid}/ball", json={"t": 2.2, "x": 300, "y": 300})  # zelf aangewezen
    client.post(f"/api/clips/{cid}/ball", json={"t": 2.6, "x": None, "y": None})  # geen bal
    ex = learning.detector_examples(st)
    idx = {e["idx"]: e for e in ex}
    assert all(i in idx for i in range(5)) and idx[0]["hard"]       # ingezoomd gevonden: altijd mee
    assert 22 in idx and idx[22]["ball"][:2] == [300, 300]          # aangewezen
    assert 26 in idx and idx[26]["ball"] is None                    # 'geen bal' is ook een les
    assert not any(20 <= i <= 29 and i not in (22, 26) for i in idx)  # onbekend: niet gebruiken
    d = idx[4]["ball"][2]
    assert 4 < d < 8  # 1/8 van een speler van 40 px

    names = {i: f"c{i}" for i in range(80)}
    yaml = learning.write_detector_dataset(ex, tmp / "ds", names)
    labels = list((tmp / "ds" / "labels").rglob("*.txt"))
    assert len(labels) == len(ex) and yaml.exists()
    lines = (tmp / "ds" / "labels" / ("val" if learning._is_val(idx[4]) else "train") / f"{cid}_000004.txt").read_text().split("\n")
    assert sum(ln.startswith("0 ") for ln in lines) == 3 and sum(ln.startswith("32 ") for ln in lines) == 1


def test_field_labels_follow_calibration(env):
    main, client, tmp = env
    from app import learning, pitch
    # veld -> beeld zoals in de testvideo: 11 px per meter, vanaf (60, 20)
    H = np.array([[11.0, 0, 60], [0, 11.0, 20], [0, 0, 1]])
    mask = learning.line_mask(H, (640, 360), (1280, 720), pitch.DEFAULT)
    # bovenste zijlijn ligt op y = 20 px (volle resolutie) = 10 px op halve schaal
    assert mask[10, 100:300].mean() > 0.9 and mask[60, 100:300].mean() < 0.2
    p, r, f = learning.line_f1(mask, mask)
    assert f == 1.0
    assert learning.line_f1(np.zeros_like(mask), mask)[2] == 0.0


def test_recognize_uses_profiles(env):
    main, client, tmp = env
    from app import analytics, learning
    st = main.store
    mid = client.post("/api/matches", json={"name": "Profiel"}).json()["id"]
    cid = st.run("INSERT INTO clips (match_id, filename, path, width, height, status) VALUES (?, 'v', '', 1280, 720, 'klaar')", (mid,))
    from app.storage import clip_dir
    np.save(clip_dir(cid) / "motion.npy", np.tile(np.eye(3), (40, 1, 1)))
    with st.tx() as c:
        for i in range(40):
            c.execute("INSERT INTO frames VALUES (?,?,?)", (cid, i, i / 10))
            for tid, x in ((1, 100), (2, 400), (3, 700)):
                c.execute("INSERT INTO detections VALUES (?,?,?,?,?,?,?,?)", (cid, i, tid, x + 4 * i, 300, x + 4 * i + 16, 340, 0.9))
        for tid in (1, 2, 3):
            c.execute("INSERT INTO tracks (clip_id, track_id, n_frames, t_start, t_end, team, team_auto, color) VALUES (?,?,40,0,3.9,0,0,'#d03030')", (cid, tid))
    monkey = analytics._valid_tracks
    analytics._valid_tracks = lambda d, **k: {int(x) for x in np.unique(d.track)}
    analytics.invalidate()
    ali = client.post(f"/api/matches/{mid}/players", json={"name": "Ali", "team": 0}).json()
    bo = client.post(f"/api/matches/{mid}/players", json={"name": "Bo", "team": 0}).json()
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=64), rng.normal(size=64)
    unit = lambda v: (v / np.linalg.norm(v)).astype(np.float32)  # noqa: E731
    learning.save_profile(st, ali, unit(a), 5)
    learning.save_profile(st, bo, unit(b), 5)
    for tid, v in ((2, a + 0.3 * rng.normal(size=64)), (3, rng.normal(size=64))):
        st.run("INSERT INTO track_embeds VALUES (?,?,?)", (cid, tid, learning.to_blob(unit(v))))
    r = client.get(f"/api/matches/{mid}/recognize").json()
    assert [(x["track_id"], x["player_id"]) for x in r] == [(2, ali["id"])]  # 3 lijkt op niemand duidelijk
    # Ali al in beeld op hetzelfde moment (track 1 gekoppeld): dan kan track 2 hem niet zijn
    client.patch(f"/api/clips/{cid}/tracks/1", json={"player_id": ali["id"]})
    assert client.get(f"/api/matches/{mid}/recognize").json() == []
    assert client.get(f"/api/matches/{mid}/profiles").json() == {str(ali["id"]): True, str(bo["id"]): True}
    analytics._valid_tracks = monkey


def test_training_runs_in_separate_process(env):
    main, client, tmp = env
    mid = client.post("/api/matches", json={"name": "Leeg"}).json()["id"]
    ov = client.get("/api/training").json()
    assert ov["detector"]["examples"] == 0 and not ov["status"]["busy"]
    r = client.post("/api/training/start", json={"kind": "players", "match_id": mid})
    assert r.status_code == 200, r.text
    for _ in range(300):
        st = client.get("/api/training/status").json()
        if not st["busy"] and st.get("last"):
            break
        time.sleep(0.2)
    assert st["last"]["ok"] is False and "Koppel eerst" in st["last"]["message"]
    assert client.post("/api/training/start", json={"kind": "onzin"}).status_code == 400
    assert client.post("/api/training/detector/activate", json={"file": "bestaat-niet.pt"}).status_code == 400
    assert client.post("/api/training/detector/activate", json={"file": None}).json()["active"] is None
