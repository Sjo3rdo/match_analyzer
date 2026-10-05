import numpy as np

from app.calibration import CameraModel, Keyframe, apply_h, cumulative, fit_homography
from app.pitch import LANDMARKS


def _true_h():
    # beeld -> veld: een willekeurige maar realistische perspectief-transformatie
    return np.array([[0.05, 0.01, -5.0], [0.002, 0.09, -20.0], [0.0, 0.0004, 1.0]])


def test_fit_homography_recovers_mapping():
    H = _true_h()
    img = np.array([[100, 400], [1800, 420], [300, 900], [1600, 950], [960, 600], [500, 700]], float)
    world = apply_h(H, img)
    K, err = fit_homography(img, world)
    assert err < 1e-6
    assert np.allclose(apply_h(K, [[700, 500]]), apply_h(H, [[700, 500]]), atol=1e-6)


def test_landmarks_inside_pitch():
    for x, y in LANDMARKS.values():
        assert 0 <= x <= 105 and 0 <= y <= 68


def test_camera_model_follows_pan():
    # camera schuift 10 px per frame naar rechts -> beeldinhoud schuift 10 px naar links
    n = 20
    inter = np.tile(np.eye(3), (n, 1, 1))
    inter[1:, 0, 2] = -10
    K0 = _true_h()
    cam = CameraModel(inter, [Keyframe(0, K0)])
    # een vast punt op het veld dat in frame 0 op (800, 600) staat, staat in frame 5 op (750, 600)
    world0 = cam.project(0, [[800, 600]])
    world5 = cam.project(5, [[750, 600]])
    assert np.allclose(world0, world5, atol=1e-6)


def test_camera_model_blends_keyframes():
    n = 11
    inter = np.tile(np.eye(3), (n, 1, 1))
    K_a = _true_h()
    K_b = _true_h().copy()
    K_b[0, 2] += 1.0  # tweede sleutelframe 1 m verschoven
    cam = CameraModel(inter, [Keyframe(0, K_a), Keyframe(10, K_b)])
    p0 = cam.project(0, [[900, 600]])[0]
    p5 = cam.project(5, [[900, 600]])[0]
    p10 = cam.project(10, [[900, 600]])[0]
    assert np.isclose(p5[0], (p0[0] + p10[0]) / 2, atol=1e-3)


def test_cumulative_identity():
    A = cumulative(np.tile(np.eye(3), (5, 1, 1)))
    assert np.allclose(A, np.eye(3))


def _line_item(H, seg, s):
    """Beeldpunt dat op veldlijn seg ligt (fractie s), via de inverse van H."""
    a, b = np.array(seg[0], float), np.array(seg[1], float)
    w = (1 - s) * a + s * b
    return {"img": apply_h(np.linalg.inv(H), w[None])[0].tolist(), "line": [list(seg[0]), list(seg[1])]}


def test_calibration_with_lines_from_sideline():
    import pytest
    """Zijlijn-situatie: maar 2 echte punten in beeld, aangevuld met 3 lijnen.
    (Precies 2 punten + 2 lijnen ligt wiskundig nooit vast; dat test de volgende test.)"""
    from app.calibration import fit_calibration
    from app.pitch import LINES
    H = _true_h()
    pts = [{"img": apply_h(np.linalg.inv(H), np.array([[94.0, 34.0]]))[0].tolist(), "pitch": [94.0, 34.0]},  # strafschopstip
           {"img": apply_h(np.linalg.inv(H), np.array([[105.0, 30.34]]))[0].tolist(), "pitch": [105.0, 30.34]}]  # doelpaal
    for name, fr in (("Zijlijn onder", (0.6, 0.9)), ("16-meterlijn rechts (voorkant)", (0.2, 0.8)),
                     ("Doellijn rechts", (0.3, 0.6))):
        pts += [_line_item(H, LINES[name], f) for f in fr]
    K, err = fit_calibration(pts)
    with pytest.raises(ValueError):  # 2 punten + 2 lijnen: niet eenduidig
        fit_calibration(pts[:6])
    assert err < 1e-4
    probe = np.array([[700.0, 500.0], [1500.0, 650.0]])
    assert np.allclose(apply_h(K, probe), apply_h(H, probe), atol=1e-3)


def test_calibration_rejects_too_little_information():
    import pytest
    from app.calibration import fit_calibration
    from app.pitch import LINES
    H = _true_h()
    pts = [_line_item(H, LINES["Zijlijn onder"], f) for f in (0.1, 0.3, 0.5, 0.7, 0.9)]  # 5 punten, één lijn
    with pytest.raises(ValueError):
        fit_calibration(pts)


def _sideline_camera(zoom=1.0):
    from app.calibration import camera_homography, default_focal
    import math
    size = (1920, 1080)
    true = np.array([60.0, 74.0, 1.7, math.radians(-50), math.radians(4), math.radians(1.0),
                     math.log(default_focal(1920) * zoom)])
    return size, true, camera_homography(true, size)


def _items(H, spec):
    from app.pitch import LINES
    out = []
    for kind, val in spec:
        if kind == "pt":
            out.append({"img": apply_h(H, np.array([val], float))[0].tolist(), "pitch": list(val)})
        else:
            name, s = val
            a, b = np.array(LINES[name][0]), np.array(LINES[name][1])
            out.append({"img": apply_h(H, ((1 - s) * a + s * b)[None])[0].tolist(), "line": [list(a), list(b)]})
    return out


def _probe_error(H, K):
    probe = np.array([[x, y] for x in range(62, 106, 6) for y in range(17, 68, 6)], float)
    img = apply_h(H, probe)
    vis = (img[:, 0] > 0) & (img[:, 0] < 1920) & (img[:, 1] > 0) & (img[:, 1] < 1080)
    return np.linalg.norm(apply_h(K, img[vis]) - probe[vis], axis=1)


PRIOR = {"x": 63.0, "y": 77.0, "h": 1.6, "sigma_pos": 5.0, "width": 1920, "height": 1080}


def test_camera_fit_point_and_line_when_not_zoomed():
    """Zijlijn, ooghoogte, positie 4 m verkeerd geschat, niet ingezoomd: 1 punt + 1 lijn is genoeg."""
    import pytest
    from app.calibration import default_focal, fit_calibration
    size, true, H = _sideline_camera()
    pts = _items(H, [("pt", (94.0, 34.0)), ("line", ("Zijlijn onder", 0.55)), ("line", ("Zijlijn onder", 0.85))])
    K, err = fit_calibration(pts, camera={**PRIOR, "f": default_focal(1920)})
    e = _probe_error(H, K)
    assert np.median(e) < 1.5 and e.max() < 5  # verste punten (±65 m op ooghoogte) zijn het minst nauwkeurig
    with pytest.raises(ValueError):  # zonder positie lukt dit niet
        fit_calibration(pts)


def test_camera_fit_estimates_zoom_with_more_clicks():
    """Ingezoomd (1,1x): met 2 punten + 1 lijn rekent de app de zoom zelf uit."""
    from app.calibration import default_focal, fit_calibration
    size, true, H = _sideline_camera(zoom=1.1)
    pts = _items(H, [("pt", (94.0, 34.0)), ("pt", (105.0, 37.66)),
                     ("line", ("Zijlijn onder", 0.55)), ("line", ("Zijlijn onder", 0.85))])
    K, err = fit_calibration(pts, camera={**PRIOR, "f": default_focal(1920)})
    e = _probe_error(H, K)
    assert np.median(e) < 1.2 and e.max() < 5


def test_front_back_sign_when_pitch_corner_is_behind_camera():
    """Camera langs de zijlijn die naar rechts kijkt: hoek (0, 0) ligt achter de camera. Punten in
    beeld moeten dan nog steeds als 'vóór de camera' (w > 0) herkend worden."""
    import math
    from app.calibration import camera_homography, default_focal, fit_calibration, fit_camera
    from app.autocalib import keyframe_points
    size = (1920, 1080)
    p = np.array([60.0, 76.0, 1.8, math.radians(-30), math.radians(5), 0.0, math.log(default_focal(1920))])
    H = camera_homography(p, size)
    assert (H @ np.array([0.0, 0.0, 1.0]))[2] < 0  # hoek (0,0) ligt echt achter de camera
    pts = keyframe_points(H, size)
    assert len(pts) >= 4
    for K in (fit_calibration(pts)[0],
              fit_camera(pts, {"x": 60, "y": 76, "h": 1.8, "f": default_focal(1920), "sigma_pos": 3,
                               "width": 1920, "height": 1080})[0]):
        w = np.array([q["img"] + [1.0] for q in pts]) @ K[2]
        assert np.all(w > 0)
        Hb = np.linalg.inv(K)
        assert (Hb @ np.array([pts[0]["pitch"][0], pts[0]["pitch"][1], 1.0]))[2] > 0


def test_goal_post_tops_and_crossbar_help_the_camera_fit():
    """Ingezoomd op het doel, positie 3-4 m verkeerd geschat: met alleen de voeten van de palen
    blijft de app dicht bij die verkeerde plek; de bovenkanten en de lat zeggen hoe ver weg en hoe
    ingezoomd het doel is, en verbeteren zo de plek en de kalibratie."""
    import math
    from app import pitch
    from app.calibration import camera_projection, default_focal, fit_calibration, fit_camera, project_3d
    size, true, H = _sideline_camera(zoom=1.25)
    P = camera_projection(true, size)
    g = pitch.DEFAULT
    feet = _items(H, [("pt", pitch.LANDMARKS["Doelpaal rechts boven"]), ("pt", pitch.LANDMARKS["Doelpaal rechts onder"])])
    tops = [{"name": n, "img": project_3d(P, g.elevated[n])[0].tolist(), "pitch3": list(g.elevated[n])}
            for n in ("Doelpaal rechts boven, bovenkant", "Doelpaal rechts onder, bovenkant")]
    a, b = np.array(g.elevated_lines["Lat rechts"])
    tops.append({"name": "Lat rechts", "img": project_3d(P, 0.3 * a + 0.7 * b)[0].tolist(), "line3": [list(a), list(b)]})
    prior = {**PRIOR, "f": default_focal(1920)}

    def pos_err(pts):
        cam = fit_camera(pts, prior)[1]
        return math.hypot(cam["x"] - true[0], cam["y"] - true[1])

    assert pos_err(feet + tops) < 0.5 * pos_err(feet)
    near = np.array([[x, y] for x in range(90, 106, 3) for y in range(20, 50, 4)], float)
    K, _ = fit_calibration(feet + tops, camera=prior)
    assert np.median(np.linalg.norm(apply_h(K, apply_h(H, near)) - near, axis=1)) < 1.0


def test_elevated_points_need_camera_position():
    import pytest
    from app import pitch
    from app.calibration import fit_calibration
    H = _true_h()
    pts = _items(np.linalg.inv(H), [("pt", (105.0, 30.34)), ("pt", (105.0, 37.66)), ("pt", (94.0, 34.0))])
    pts += [{"name": "x", "img": [500.0, 300.0], "pitch3": list(pitch.DEFAULT.elevated["Doelpaal rechts boven, bovenkant"])}]
    with pytest.raises(ValueError, match="positie"):
        fit_calibration(pts)


def test_zoomed_clicks_count_more_and_fit_in_pixels():
    """Punten die ingezoomd zijn aangeklikt (z), liggen preciezer: de kalibratie moet daar vlak
    langs gaan, ook als de andere klikken wat slordig zijn."""
    from app.calibration import fit_calibration
    size, true, Himg = _sideline_camera()
    rng = np.random.default_rng(4)
    world = np.array([[94.0, 34.0], [105, 13.84], [88.5, 13.84], [105, 54.16], [88.5, 54.16], [80, 68], [100, 68], [94, 68]])
    img = apply_h(Himg, world)
    noisy = img + rng.normal(scale=6.0, size=img.shape)
    precise = [5, 6]  # de zijlijnpunten dichtbij, ingezoomd aangeklikt
    noisy[precise] = img[precise] + rng.normal(scale=0.5, size=(2, 2))

    def err_at(K, k):
        G = np.linalg.inv(K)
        return float(np.linalg.norm(apply_h(G, world[k:k + 1])[0] - img[k]))

    pts = [{"img": noisy[k].tolist(), "pitch": world[k].tolist()} for k in range(len(world))]
    K0, _ = fit_calibration(pts)
    for k in precise:
        pts[k]["z"] = 4
    K1, _ = fit_calibration(pts)
    assert max(err_at(K1, k) for k in precise) < max(err_at(K0, k) for k in precise)
    assert max(err_at(K1, k) for k in precise) < 2.5


def test_mirrored_field_is_never_chosen():
    """Een camera ziet het veld nooit in spiegelbeeld. Middenlijn, zijlijn en middenstip vallen bij
    dubbelvouwen op de middenlijn op zichzelf: dan kiest de app de echte (niet gespiegelde) kant."""
    import math
    import pytest
    from app import calibration as C
    W, Hh = 1920, 1080
    f = C.default_focal(W)
    cam, look = (45.0, 72.0, 1.7), (65.0, 45.0)  # langs de onderste zijlijn, kijkend naar de rechterhelft
    yaw = math.atan2(look[1] - cam[1], look[0] - cam[0])
    tilt = math.atan2(cam[2], math.hypot(look[0] - cam[0], look[1] - cam[1]))
    G = C.camera_homography(np.array([*cam, yaw, tilt, 0, math.log(f)]), (W, Hh))
    K_true = C.orient_by_points(C.normalize_h(np.linalg.inv(G)), apply_h(G, np.array([look])))
    assert not C.mirrored(K_true)
    img = lambda X: apply_h(G, np.array([X], float))[0].tolist()  # noqa: E731
    pts = [{"img": img((x, 68)), "line": [[0, 68], [105, 68]]} for x in (58, 66, 75, 85)]
    pts += [{"img": img((52.5, y)), "line": [[52.5, 0], [52.5, 68]]} for y in (25, 40, 55, 62)]
    pts += [{"img": img((52.5, 68)), "pitch": [52.5, 68]}, {"img": img((52.5, 34)), "pitch": [52.5, 34]}]
    M = np.array([[-1.0, 0, 105], [0, 1, 0], [0, 0, 1]])  # spiegelen in de middenlijn
    K_mirror = C.orient_by_points(C.normalize_h(M @ K_true), [p["img"] for p in pts])
    assert C.mirrored(K_mirror)
    assert np.mean(C.calibration_residuals(K_mirror, pts)) < 1e-6  # past net zo goed bij de klikken
    K, err = C._unmirror(K_mirror, 0.0, pts, refine=True)
    probe = np.array([img((75, 60))])
    assert not C.mirrored(K) and np.allclose(apply_h(K, probe), [[75, 60]], atol=0.05)
    # Alles links-rechts verwisseld aangeklikt (bijv. het verkeerde doel): geen echte camera kan dit zien
    swapped = [{"img": img((x, 68)), "pitch": [105 - x, 68]} for x in (58, 75, 90)]
    swapped += [{"img": img((88.5, 54.16)), "pitch": [16.5, 54.16]}, {"img": img((94, 34)), "pitch": [11, 34]}]
    with pytest.raises(ValueError, match="gespiegeld"):
        C.fit_calibration(swapped)


def test_swapped_goal_post_is_the_suspect_point():
    """Verkeerde paal gekoppeld: zonder die klik klopt de rest, dus die klik is de verdachte."""
    from app import pitch
    from app.calibration import default_focal, fit_camera, suspect_point
    size, true, H = _sideline_camera(zoom=1.25)
    L = pitch.LANDMARKS
    names = ["Doelpaal rechts boven", "Strafschopgebied rechts hoek boven", "Strafschopgebied rechts hoek onder",
             "Doelgebied rechts hoek onder", "Strafschopstip rechts", "Doelgebied rechts hoek boven"]
    pts = _items(H, [("pt", L[n]) for n in names])
    prior = {**PRIOR, "f": default_focal(1920)}
    assert suspect_point(fit_camera(pts, prior)[1]["params"], pts, prior, size) is None
    bad = [dict(p) for p in pts]
    bad[0]["pitch"] = list(L["Doelpaal rechts onder"])  # linker- en rechterpaal verwisseld
    assert suspect_point(fit_camera(bad, prior)[1]["params"], bad, prior, size) == 0


def test_own_view_direction_steers_the_camera_fit():
    """Kijkrichting zelf aangegeven (kompas): de fit zoekt alleen in die richting en blijft er dichtbij."""
    import math
    from app.calibration import default_focal, fit_camera, kf_camera
    size, true, H = _sideline_camera()
    pts = _items(H, [("pt", (88.5, 13.84)), ("ln", ("Zijlijn onder", 0.7))])
    prior = {**PRIOR, "f": default_focal(1920)}
    wrong = {**prior, "yaw": true[3] + math.radians(150)}  # bewust de verkeerde kant op
    right = kf_camera(prior, {"yaw": math.degrees(true[3]) + 8})
    yaw = lambda cam: (cam["yaw_deg"] - math.degrees(true[3]) + 180) % 360 - 180  # noqa: E731
    assert abs(yaw(fit_camera(pts, right)[1])) < 6
    assert abs(yaw(fit_camera(pts, wrong)[1])) > 60  # het kompas weegt echt mee
    assert kf_camera(prior, {"yaw": None}) is prior
