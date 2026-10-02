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
