"""Automatisch bijstellen: een nagemaakte zwenkende video met opzettelijke drift moet weer
op de veldlijnen 'klikken', met maar één handmatig sleutelframe aan het begin."""
import math

import cv2
import numpy as np

from app import autocalib as ac
from app.calibration import CameraModel, Keyframe, apply_h, camera_homography, cumulative, default_focal

W, H, FPS, SECONDS = 960, 540, 20, 10
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
    yaw = math.radians(-130 + 8 * t)  # zwenkt 80° in 10 s: van het linker strafschopgebied tot voorbij het midden
    return np.array([52.5, 85.0, 6.0, yaw, math.radians(13), 0.0, math.log(f)])


def _frame(tex, Hc):
    T = np.array([[PX_PER_M, 0, MARGIN * PX_PER_M], [0, PX_PER_M, MARGIN * PX_PER_M], [0, 0, 1.0]])
    M = Hc @ np.linalg.inv(T)  # textuur -> beeld
    out = cv2.warpPerspective(tex, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=(70, 95, 80))
    return out


def _run_drift_test(tmp_path, with_camera: bool):
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
    if with_camera:  # camerapositie bekend (aangeklikt of via GPS), 2 m naast de echte plek
        cam = _camera(0)
        store.run("UPDATE clips SET cam_x = ?, cam_y = ?, cam_h = ?, cam_source = 'hand' WHERE id = ?",
                  (cam[0] + 1.5, cam[1] - 1.5, cam[2], cid))
    frames = store.all("SELECT idx, t FROM frames WHERE clip_id = ? ORDER BY idx", (cid,))
    t = np.array([f["t"] for f in frames])

    # Camerabeweging: de echte (uit de nagemaakte camera) plus een sluipende fout van 0,02° en
    # 0,05% zoom per geanalyseerd frame (0,2°/s). Dat is meer dan gemeten op echte clips; zonder
    # bijstellen loopt de kalibratie daardoor binnen enkele seconden tientallen pixels weg.
    f = default_focal(W)
    D = ac.correction(np.array([0.0, math.radians(0.02), 0.0, 0.0005]), f, (W, H))
    Hs = [camera_homography(_camera(ti), (W, H)) for ti in t]
    inter = np.array([np.eye(3)] + [D @ Hs[i] @ np.linalg.inv(Hs[i - 1]) for i in range(1, len(t))])
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
    kfs = [Keyframe(0, K0)]
    for kf in store.keyframes(cid):
        if kf["auto"]:
            from app.calibration import fit_keyframe
            kfs.append(Keyframe(int(np.argmin(np.abs(t - kf["t"]))), fit_keyframe(kf)[0]))
    after = errors(kfs)
    return before, after, res

def test_autocalib_removes_drift_with_camera_position(tmp_path):
    before, after, res = _run_drift_test(tmp_path, with_camera=True)
    assert res["accepted"] >= 0.5 * res["tried"], res
    assert before[-1] > 20, before  # zonder bijstellen: tientallen pixels weg
    # (pixels in een beeld van 57 graden breed; in de oude 64 graden was de grens 6 px)
    assert np.median(after) < 3 and after.max() < 6.8, (before.round(1), after.round(1))


def test_autocalib_without_camera_position_never_worse(tmp_path):
    """Zonder camerapositie is niet alles te bepalen, maar het mag nooit slechter worden."""
    before, after, res = _run_drift_test(tmp_path, with_camera=False)
    assert np.all(after <= before + 1.0), (before.round(1), after.round(1))
    assert np.median(after) < 0.7 * np.median(before)


def test_propose_finds_pitch_from_camera_position():
    """Automatisch voorstel: alleen de aangeklikte camerapositie is bekend (2 m ernaast), geen klikken."""
    tex = _texture()
    for ti in (3.0, 9.0):
        cam = _camera(ti)
        Hc = camera_homography(cam, (W, H))
        prior = {"x": cam[0] + 1.5, "y": cam[1] - 1.5, "h": cam[2], "f": default_focal(W), "sigma_pos": 3.0,
                 "width": W, "height": H}
        res = ac.propose(_frame(tex, Hc), prior)
        assert res is not None, ti
        Hp, info = res
        img = apply_h(Hc, ac.SAMPLES)
        ok = (img[:, 0] > 0) & (img[:, 0] < W) & (img[:, 1] > 0) & (img[:, 1] < H)
        err = np.linalg.norm(apply_h(Hp, ac.SAMPLES[ok]) - img[ok], axis=1)
        assert np.median(err) < 8, (ti, np.median(err), info)


def test_propose_refuses_without_lines():
    rng = np.random.default_rng(1)
    grass = np.full((H, W, 3), (55, 140, 65), np.uint8)
    grass = cv2.add(grass, rng.integers(0, 30, (H, W, 3), dtype=np.uint8))
    prior = {"x": 52.5, "y": 85.0, "h": 6.0, "f": default_focal(W), "sigma_pos": 8.0, "width": W, "height": H}
    assert ac.propose(grass, prior) is None


def test_detect_lines_sunny_tinted_line_but_not_the_boards():
    """Zonnig gras: een verre lijn is licht geelgroen in plaats van wit (telt wel), en achter het
    veld staat een rij witte reclameborden met bomen erboven (telt niet)."""
    rng = np.random.default_rng(5)
    img = np.zeros((1080, 1920, 3), np.uint8)
    img[:400] = (60, 90, 50)  # bomen (donkergroen)
    img[400:424] = (235, 235, 235)  # reclameborden
    img[424:] = (70, 170, 120)  # gras in de zon (BGR: geelgroen)
    noise = cv2.resize(rng.integers(0, 25, (540, 960), dtype=np.uint8), (1920, 1080))
    img = cv2.add(img, cv2.merge([noise, noise, noise]))
    cv2.line(img, (0, 840), (1919, 760), (165, 215, 190), 3)  # verre lijn: dun, lichter en grijzer dan het gras
    m = ac.detect_lines(img)
    assert m[370:430].sum() / 255 > 500  # de lijn (werkschaal: halve resolutie)
    assert m[195:216].sum() / 255 < 50  # de borden


def test_enhanced_line_search_finds_worn_lines_but_not_grass():
    """Een versleten, vage lijn (de helft van de verf weg) valt per pixel weg in het gras; de extra
    lijnzoeker middelt over een stukje lijn en vindt hem wel. Gewoon korrelig gras blijft gras."""
    rng = np.random.default_rng(0)

    def grass():
        img = np.zeros((1080, 1920, 3), np.uint8)
        img[:] = (50, 150, 70)
        n = cv2.GaussianBlur(rng.integers(0, 60, (1080, 1920), dtype=np.uint8), (3, 3), 0)
        return cv2.add(img, cv2.merge([n, n, n]))
    g = grass()
    line = np.zeros(g.shape[:2], np.uint8)
    cv2.line(line, (100, 700), (1800, 560), 255, 3)
    a = ((line > 0) & (rng.random(g.shape[:2]) > 0.5))[..., None] * 0.55
    img = (g * (1 - a) + np.array([225, 230, 228]) * a).astype(np.uint8)
    gt = cv2.resize(line, (960, 540)) > 0
    far = ~cv2.dilate(gt.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)

    def found(m):
        near = cv2.dilate((m > 0).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        return near[gt].mean()
    base, enh = ac.detect_lines(img), ac.detect_lines(img, enhance=True)
    assert found(enh) > 0.8 and found(enh) > found(base) + 0.3
    assert ((enh > 0) & far).sum() == 0
    assert (ac.detect_lines(grass(), enhance=True) > 0).sum() == 0


def _low_camera(yaw_deg=215.0, tilt_deg=1.0):
    """Telefoon op 1,2 m, 4 m buiten de zijlijn, kijkt schuin over het veld (zoals langs de lijn)."""
    return np.array([60.0, 72.0, 1.2, math.radians(yaw_deg), math.radians(tilt_deg), 0.0, math.log(default_focal(W))])


def _with_goals(frame, cam):
    """Bomen boven de horizon en witte doelen (palen + lat) in het beeld tekenen."""
    from app.calibration import camera_projection, project_3d
    P = camera_projection(cam, (W, H))
    out = frame.copy()
    far = project_3d(P, np.array([[x, -15.0, 0.0] for x in np.linspace(-15, 120, 60)]))  # rand achter het veld
    far = far[np.isfinite(far).all(1)]
    if len(far):
        top = int(np.clip(np.nanmin(far[:, 1]), 0, H))
        out[:top] = (40, 75, 45)
    for x0 in (0.0, 105.0):
        a, b = (x0, 34 - 3.66), (x0, 34 + 3.66)
        frame3 = [(*a, 0.0), (*a, 2.44), (*b, 2.44), (*b, 0.0)]
        q = project_3d(P, np.array(frame3))
        if np.isfinite(q).all():
            cv2.polylines(out, [np.round(q).astype(np.int32)], False, (250, 250, 250), 2)
    return out


def _sample_err(Ha, Hb):
    img = apply_h(Ha, ac.SAMPLES)
    ok = (img[:, 0] > 0) & (img[:, 0] < W) & (img[:, 1] > 0) & (img[:, 1] < H)
    return float(np.median(np.linalg.norm(apply_h(Hb, ac.SAMPLES[ok]) - img[ok], axis=1)))


def test_known_position_puts_the_line_back_on_the_sideline():
    """Doorgegeven kalibratie (via de omgeving) zit 1,5 graad te hoog en 1 graad gedraaid: de gele lijn
    ligt dan ver boven de zijlijn. Met de bekende plek moet bijstellen hem terugleggen, ook als de
    fout veel groter is dan de kleine stapjes van het volgen."""
    tex = _texture()
    cam = _low_camera()
    Hc = camera_homography(cam, (W, H))
    frame = _frame(tex, Hc)
    bad = cam.copy()
    bad[3] += math.radians(1.0)
    bad[4] += math.radians(1.5)
    H_bad = camera_homography(bad, (W, H))
    assert _sample_err(Hc, H_bad) > 15
    camera = {"x": cam[0], "y": cam[1], "h": cam[2], "roll": 0.0, "log_f": cam[6], "q0": bad[3:], "search": True}
    res = ac.refine(frame, H_bad, None, math.exp(cam[6]), camera=camera)
    assert res is not None
    assert _sample_err(Hc, res[0]) < 3, res[1]


def test_goal_fixes_the_view_direction_along_a_lone_sideline():
    """Alleen de zijlijn vlak voor je en een doel in beeld: langs de zijlijn draaien zie je aan de
    lijn nauwelijks, maar aan het doel wel."""
    tex = np.full_like(_texture(), (60, 140, 70))  # alleen de zijlijn aan jouw kant (de rest in tegenlicht)
    y = int((68 + MARGIN) * PX_PER_M)
    cv2.line(tex, (int(MARGIN * PX_PER_M), y), (int((105 + MARGIN) * PX_PER_M), y), (240, 240, 240), 2)
    cam = _low_camera(yaw_deg=200.0, tilt_deg=0.5)
    Hc = camera_homography(cam, (W, H))
    bad = cam.copy()
    bad[3] += math.radians(2.0)
    camera = {"x": cam[0], "y": cam[1], "h": cam[2], "roll": 0.0, "log_f": cam[6], "q0": bad[3:], "search": True}
    turned = {}
    for goals in (False, True):
        frame = _frame(tex, Hc)
        frame = _with_goals(frame, cam) if goals else frame
        dbg = {}
        res = ac.refine(frame, camera_homography(bad, (W, H)), None, math.exp(cam[6]), camera=camera, debug=dbg)
        assert res is not None, dbg
        turned[goals] = math.degrees(res[1]["params"][0] - cam[3])
    assert abs(turned[False] - 2.0) < 0.3, turned  # zonder doel: de draairichting van de voorspelling houden
    assert abs(turned[True]) < 0.4, turned  # met doel: rechtgezet


def test_known_position_refuses_a_frame_without_lines():
    tex = np.full_like(_texture(), (60, 140, 70))
    cam = _low_camera()
    Hc = camera_homography(cam, (W, H))
    camera = {"x": cam[0], "y": cam[1], "h": cam[2], "roll": 0.0, "log_f": cam[6], "q0": cam[3:]}
    assert ac.refine(_frame(tex, Hc), Hc, None, math.exp(cam[6]), camera=camera) is None
