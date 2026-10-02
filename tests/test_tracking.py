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
