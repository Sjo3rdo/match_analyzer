"""De balvolger: kiest de kandidaat die bij het spoor past en zoomt in waar de bal moet zijn."""
from app.ball import BallTracker, Candidate


def test_follows_track_and_ignores_second_ball():
    tr = BallTracker(1920, 1080)
    for i in range(5):
        c = tr.choose([Candidate(500 + 20 * i, 600, 0.3, 10), Candidate(1500, 900, 0.35, 10)][: 1 if i == 0 else 2])
        tr.update(c)
        assert c.x == 500 + 20 * i  # niet de reservebal rechtsonder, ook al is die zekerder
    # bal even weg: verwachte plek schuift door, uitsnede ligt eromheen
    tr.update(None)
    x0, y0, x1, y1 = tr.crop_box()
    assert x0 < 620 < x1 and y0 < 600 < y1 and x1 - x0 == 640
    # een zwakke kandidaat precies op het spoor telt wel, eentje ver weg niet
    assert tr.choose([Candidate(1200, 200, 0.1, 10)]) is None
    assert tr.choose([Candidate(622, 601, 0.06, 10)]).x == 622


def test_without_track_needs_confident_detection_and_scans_when_lost():
    tr = BallTracker(1920, 1080)
    assert tr.choose([Candidate(100, 100, 0.08, 8)]) is None
    assert tr.choose([Candidate(100, 100, 0.3, 8)]) is not None
    assert tr.crop_box() is None
    boxes = tr.scan_boxes(frame_no=10)
    assert boxes and all(b[3] - b[1] == 640 for b in boxes) and max(b[2] for b in boxes) == 1920
    assert tr.scan_boxes(frame_no=11) == []  # niet elk beeld


def test_static_logo_on_boards_is_dropped_but_moving_ball_kept():
    import numpy as np
    from app.ball import static_runs
    n = 60
    # camera zwenkt 5 px per beeld naar rechts: de achtergrond schuift 5 px naar links
    inter = np.tile(np.eye(3), (n, 1, 1))
    inter[1:, 0, 2] = -5
    from app.calibration import cumulative
    A = cumulative(inter)
    idx = np.arange(0, n, 2)
    logos = np.array([[800 - 5 * i + (22 if k % 3 == 0 else 0), 560] for k, i in enumerate(idx)], float)  # twee logo's
    conf = np.full(len(idx), 0.08)
    assert static_runs(idx, logos, conf, A).mean() > 0.9
    ball = np.array([[300 + 9 * i, 600 - 2 * i] for i in idx], float)  # rolt over het veld
    assert not static_runs(idx, ball, conf, A).any()
    assert not static_runs(idx, logos, np.full(len(idx), 0.6), A).any()  # zeker van: stilliggende bal
