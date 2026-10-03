import numpy as np

from app.tracking import Tracker


def test_ids_stable_for_moving_players():
    tr = Tracker(fps=10, min_hits=1)
    ids = []
    for i in range(30):
        boxes = np.array([[100 + 3 * i, 100, 140 + 3 * i, 200], [600 - 3 * i, 300, 640 - 3 * i, 400]])
        out = tr.update(boxes, np.array([0.9, 0.9]))
        ids.append(tuple(sorted(t for t, _, _ in out)))
    assert len(set(ids)) == 1 and len(ids[0]) == 2


def test_camera_pan_compensation():
    tr = Tracker(fps=10, min_hits=1)
    tr.update(np.array([[100, 100, 130, 180]]), np.array([0.9]))
    first = tr.tracks[0].id
    # camera draait: alles schuift 60 px (meer dan de box breed is)
    H = np.eye(3)
    H[0, 2] = -60
    out = tr.update(np.array([[40, 100, 70, 180]]), np.array([0.9]), H)
    assert [t for t, _, _ in out] == [first]


def test_track_survives_short_occlusion():
    tr = Tracker(fps=10, min_hits=1, lost_seconds=1.0)
    tr.update(np.array([[100, 100, 130, 180]]), np.array([0.9]))
    tid = tr.tracks[0].id
    for _ in range(5):
        tr.update(np.zeros((0, 4)), np.zeros(0))
    out = tr.update(np.array([[102, 100, 132, 180]]), np.array([0.9]))
    assert out[0][0] == tid


def _tracklet(start, n, x0, vx, color):
    idx = np.arange(start, start + n)
    foot = np.stack([x0 + vx * (idx - start), np.full(n, 500.0)], 1)
    return {"idx": idx, "foot": foot, "h": np.full(n, 50.0), "color": np.array(color, float)}


def test_stitch_joins_same_player_after_gap():
    from app.tracking import stitch_tracks
    yellow, green = [200, 121, 168], [135, 115, 136]
    tr = {1: _tracklet(0, 20, 100, 5, yellow),      # loopt naar rechts, stopt bij x=195
          2: _tracklet(24, 20, 215, 5, yellow),     # verschijnt 4 frames later verderop: zelfde speler
          3: _tracklet(24, 20, 220, 5, green)}      # zelfde plek, ander shirt: niet koppelen
    root = stitch_tracks(tr, fps=10)
    assert root[2] == 1 and root[3] == 3


def test_color_prevents_id_switch():
    from app.tracking import Tracker
    yellow, red = np.array([200, 121, 168.0]), np.array([150, 165, 145.0])
    tr = Tracker(fps=10, min_hits=1)
    tr.update(np.array([[100, 100, 120, 150]]), np.array([0.9]), colors=[yellow])
    tid = tr.tracks[0].id
    # volgende frame: scheidsrechter precies op de voorspelde plek, de speler iets verderop
    out = tr.update(np.array([[100, 100, 120, 150], [112, 100, 132, 150]]), np.array([0.9, 0.9]), colors=[red, yellow])
    got = {t: b[0] for t, b, _ in out}
    assert got[tid] == 112
