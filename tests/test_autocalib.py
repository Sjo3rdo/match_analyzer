"""Automatisch bijstellen: een nagemaakte zwenkende video met opzettelijke drift moet weer
op de veldlijnen 'klikken', met maar één handmatig sleutelframe aan het begin."""
import math

import cv2
import numpy as np

from app import autocalib as ac
from app.calibration import CameraModel, Keyframe, apply_h, camera_homography, cumulative, default_focal

W, H, FPS, SECONDS = 960, 540, 20, 8
PX_PER_M, MARGIN = 8.0, 15.0


def _texture():
    """Bovenaanzicht van het veld: gras met ruis en maaistroken, witte lijnen."""
    rng = np.random.default_rng(3)
    tw, th = int((105 + 2 * MARGIN) * PX_PER_M), int((68 + 2 * MARGIN) * PX_PER_M)
    img = np.zeros((th, tw, 3), np.uint8)
    img[:] = (55, 140, 65)
    for x0 in range(0, 105, 10):
        a = int((x0 + MARGIN) * PX_PER_M)
        img[:, a:a + int(5 * PX_PER_M)] = (65, 158, 75)
    noise = rng.integers(0, 40, (th, tw), dtype=np.uint8)
    noise = cv2.GaussianBlur(noise, (3, 3), 0)
    img = cv2.add(img, cv2.merge([noise, noise, noise]))
    for x, y in ac.pitch_samples(step=0.05):
        cv2.circle(img, (int((x + MARGIN) * PX_PER_M), int((y + MARGIN) * PX_PER_M)), 1, (240, 240, 240), -1)
    return img


def _camera(t):
    f = default_focal(W)
    yaw = math.radians(-120 + 12 * t)  # zwenkt ~96° in 8 s
    return np.array([52.5, 85.0, 6.0, yaw, math.radians(13), 0.0, math.log(f)])


def _frame(tex, Hc):
    T = np.array([[PX_PER_M, 0, MARGIN * PX_PER_M], [0, PX_PER_M, MARGIN * PX_PER_M], [0, 0, 1.0]])
    M = Hc @ np.linalg.inv(T)  # textuur -> beeld
    out = cv2.warpPerspective(tex, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=(70, 95, 80))
    return out


def test_autocalib_removes_drift(tmp_path):
    from app import pipeline
    from app.detection import FrameDetections
    from app.storage import Store, clip_dir
    import json

    tex = _texture()
    path = tmp_path / "zwenk.mp4"
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for i in range(FPS * SECONDS):
        wr.write(_frame(tex, camera_homography(_camera(i / FPS), (W, H))))
    wr.release()

    store = Store(tmp_path / "db.sqlite")
    import app.config as config
    config.CLIPS_DIR = tmp_path / "clips"
    mid = store.run("INSERT INTO matches (name) VALUES ('x')")
    info = pipeline.probe(path)
    cid = store.run("INSERT INTO clips (match_id, filename, path, fps, width, height, duration) VALUES (?,?,?,?,?,?,?)",
                    (mid, "zwenk.mp4", str(path), info["fps"], W, H, info["duration"]))
    (clip_dir(cid) / "preview.mp4").write_bytes(b"x")

    class NoPlayers:
        def __call__(self, frame):
            return FrameDetections(np.zeros((0, 4)), np.zeros(0), None)

    pipeline.process_clip(store, cid, detector=NoPlayers())
    frames = store.all("SELECT idx, t FROM frames WHERE clip_id = ? ORDER BY idx", (cid,))
    t = np.array([f["t"] for f in frames])

    # sabotage: per stap een kleine extra draaiing in de camerabeweging (drift). 0,04° per
    # geanalyseerd frame = 0,4°/s; ruim meer dan gemeten op echte clips (~0,05-0,1°/s).
    inter = np.load(clip_dir(cid) / "motion.npy")
    f = default_focal(W)
    D = ac.correction(np.array([0.0, math.radians(0.04), 0.0, 0.0005]), f, (W, H))
    inter[1:] = np.array([D @ h for h in inter[1:]])
    np.save(clip_dir(cid) / "motion.npy", inter)

    # één handmatig sleutelframe aan het begin (exact)
    H0 = camera_homography(_camera(t[0]), (W, H))
    pts = ac.keyframe_points(H0, (W, H))
    store.run("INSERT INTO keyframes (clip_id, t, points) VALUES (?,?,?)", (cid, float(t[0]), json.dumps(pts)))

    def errors(keyframes):
        cam = CameraModel(inter[:len(t)], keyframes)
        out = []
        for i in range(0, len(t), 5):
            Ht = camera_homography(_camera(t[i]), (W, H))
            img = apply_h(Ht, ac.SAMPLES)
            ok = (img[:, 0] > 0) & (img[:, 0] < W) & (img[:, 1] > 0) & (img[:, 1] < H)
            got = np.zeros_like(img[ok])
            for G, w in cam.homographies(i):
                got += w * apply_h(np.linalg.inv(G), ac.SAMPLES[ok])
            out.append(np.median(np.linalg.norm(got - img[ok], axis=1)))
        return np.array(out)

    K0 = np.linalg.inv(H0)
    before = errors([Keyframe(0, K0)])
    res = ac.run_autocalib(store, cid)
    assert res["accepted"] >= 0.6 * res["tried"], res
    kfs = [Keyframe(0, K0)]
    for kf in store.keyframes(cid):
        if kf["auto"]:
            from app.calibration import fit_keyframe
            kfs.append(Keyframe(int(np.argmin(np.abs(t - kf["t"]))), fit_keyframe(kf)[0]))
    after = errors(kfs)
    # zonder bijstellen loopt de fout op tot tientallen pixels; met bijstellen blijft hij klein
    assert before[-1] > 25, before
    assert np.median(after) < 3 and after.max() < 8, (before.round(1), after.round(1))
