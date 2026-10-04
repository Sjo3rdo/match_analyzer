import numpy as np

from app import analytics
from app.analytics import ClipData, movement_stats, passes, possession_segments, smooth_series


def test_distance_and_sprint():
    t = np.arange(0, 10, 0.1)
    x = 8.0 * t  # 8 m/s = 28,8 km/u
    segs = smooth_series(t, np.stack([x + 10, np.full_like(x, 30)], axis=1))
    mv = movement_stats(segs)
    assert abs(mv["distance_m"] - 8.0 * 9.9) < 1.0
    assert abs(mv["max_speed_ms"] - 8.0) < 0.2
    assert len(mv["sprints"]) == 1


def test_gaps_split_segments_and_noise_rejected():
    t = np.r_[np.arange(0, 2, 0.1), np.arange(10, 12, 0.1)]
    xy = np.stack([np.r_[np.full(20, 10.0), np.full(20, 60.0)], np.full(40, 30.0)], axis=1)
    mv = movement_stats(smooth_series(t, xy))
    assert mv["distance_m"] < 0.5  # de sprong van 50 m over het gat telt niet


def _clip_with_ball():
    # 3 spelers: A en B (team 0), C (team 1). Bal gaat A -> B -> C.
    fps = 10
    n = 60
    t = np.arange(n) / fps
    pos = {1: (20, 30), 2: (40, 30), 3: (60, 30)}
    idx, track, xy = [], [], []
    for i in range(n):
        for tid, p in pos.items():
            idx.append(i), track.append(tid), xy.append(p)
    ball = [pos[1]] * 20 + [pos[2]] * 20 + [pos[3]] * 20
    d = ClipData(clip={"id": 1}, t=t, fps=fps, calibrated=True, idx=np.array(idx), track=np.array(track),
                 boxes=np.zeros((len(idx), 4)), xy=np.array(xy, float), ball_idx=np.arange(n),
                 ball_xy=np.array(ball, float), ball_img=np.zeros((n, 2)))
    d.entity = {1: "p1", 2: "p2", 3: "p3"}
    d.team = {1: 0, 2: 0, 3: 1}
    d.valid_tracks = {1, 2, 3}
    return d


def test_possession_and_passes():
    segs = possession_segments(_clip_with_ball())
    assert [s.entity for s in segs] == ["p1", "p2", "p3"]
    ps = passes(segs)
    assert [(p["from"], p["to"], p["success"]) for p in ps] == [("p1", "p2", True), ("p2", "p3", False)]


def test_heatmap_shape():
    hm = analytics.heatmap(np.array([[52.5, 34.0]] * 10), fps=10)
    assert len(hm) == 14 and len(hm[0]) == 21
    assert abs(sum(map(sum, hm)) - 1.0) < 1e-6


def _people_clip(calibrated: bool):
    """Twee tracks: een speler die rondloopt en een toeschouwer die stilstaat (camera zwenkt niet)."""
    from app.calibration import CameraModel
    fps, n = 10, 100
    t = np.arange(n) / fps
    idx, track, boxes, xy = [], [], [], []
    for i in range(n):
        idx += [i, i]
        track += [1, 2]
        px = 300 + 4 * i  # speler loopt 400 px
        boxes += [[px - 10, 300, px + 10, 360], [800, 500, 820, 560]]
        xy += [[30 + 0.4 * i, 30], [60, 70]]  # toeschouwer: net buiten de zijlijn
    d = ClipData(clip={"id": 1, "height": 1080}, t=t, fps=fps, calibrated=calibrated, idx=np.array(idx),
                 track=np.array(track), boxes=np.array(boxes, float), xy=np.array(xy, float),
                 ball_idx=np.zeros(0, int), ball_xy=np.zeros((0, 2)), ball_img=np.zeros((0, 2)),
                 camera=CameraModel(np.tile(np.eye(3), (n, 1, 1)), []))
    return d


def test_spectators_filtered():
    assert analytics._valid_tracks(_people_clip(False)) == {1}
    d = _people_clip(True)
    d.xy[d.track == 2] = [60, 66.5]  # op het veld volgens de kalibratie, maar stil langs de zijlijn
    assert analytics._valid_tracks(d) == {1}
    d.xy[d.track == 2] = [2, 34]  # stilstaande keeper op de doellijn blijft
    assert analytics._valid_tracks(d) == {1, 2}


def test_people_cut_off_at_bottom_filtered():
    d = _people_clip(False)
    d.boxes[d.track == 2] = [800, 900, 900, 1074]  # zo eindigt een kader van YOLO aan de rand (net boven 1080)
    d.boxes[d.track == 2, 0] += np.arange(100) * 5  # beweegt wel, maar voeten buiten beeld
    d.boxes[d.track == 2, 2] += np.arange(100) * 5
    assert analytics._valid_tracks(d) == {1}


def test_ball_outside_pitch_and_jumps_removed():
    from app import pitch
    t = np.arange(10) / 10
    xy = np.array([[50, 30], [50.5, 30], [51, 30], [90, 60], [52, 30], [52.5, 30], [53, 30],
                   [53.5, 80], [54, 30], [54.5, 30]], float)  # 3 = losse uitschieter, 7 = buiten het veld
    out = analytics.clean_ball(t, xy, pitch.DEFAULT)
    assert np.isnan(out[3]).all() and np.isnan(out[7]).all()
    assert not np.isnan(out[[0, 1, 2, 4, 5, 6, 8, 9]]).any()


def test_heatmap_counts_seconds_per_video_fps():
    pts = np.array([[52.5, 34.0]] * 25)
    assert abs(sum(map(sum, analytics.heatmap(pts, 12.5))) - 2.0) < 1e-6  # 25 beelden bij 12,5 per seconde


def test_bad_calibration_does_not_hide_everyone():
    """Staat (bijna) iedereen die rondloopt buiten het veld, dan klopt de kalibratie niet: niet wegfilteren."""
    from app.calibration import CameraModel
    fps, n, k = 10, 100, 6
    idx, track, boxes, xy = [], [], [], []
    for i in range(n):
        for j in range(k):
            idx.append(i)
            track.append(j + 1)
            px = 100 + 150 * j + 3 * i
            boxes.append([px - 10, 300, px + 10, 360])
            xy.append([300 + 0.3 * i, 300 + 10 * j])  # ver buiten het veld
    d = ClipData(clip={"id": 1, "height": 1080}, t=np.arange(n) / fps, fps=fps, calibrated=True, idx=np.array(idx),
                 track=np.array(track), boxes=np.array(boxes, float), xy=np.array(xy, float),
                 ball_idx=np.zeros(0, int), ball_xy=np.zeros((0, 2)), ball_img=np.zeros((0, 2)),
                 camera=CameraModel(np.tile(np.eye(3), (n, 1, 1)), []))
    assert analytics._valid_tracks(d) == set(range(1, k + 1))
    assert d.calib_suspect
    assert not _people_clip(True).calib_suspect


def _shot_clip(vy: float = 0.0, speed: float = 20.0):
    """Speler 1 dribbelt met de bal naar het rechterdoel en schiet op t = 2 s vanaf (85, 34)."""
    from app.calibration import CameraModel
    fps, n = 10, 50
    t = np.arange(n) / fps
    idx, track, boxes, xy = [], [], [], []
    ball_idx, ball_xy = [], []
    for i in range(n):
        px = 75 + 5 * min(t[i], 2.0)  # loopt naar (85, 34) en blijft daar
        idx.append(i)
        track.append(1)
        boxes.append([400, 300, 420, 360])
        xy.append([px, 34.0])
        if t[i] <= 2.0:
            ball_idx.append(i)
            ball_xy.append([px + 0.5, 34.0])
        elif t[i] <= 2.0 + 19 / speed:  # schot: tot vlak voor de doellijn, daarna kwijt (in het net)
            ball_idx.append(i)
            ball_xy.append([85.5 + speed * (t[i] - 2.0), 34.0 + vy * (t[i] - 2.0)])
    d = ClipData(clip={"id": 1, "height": 1080}, t=t, fps=fps, calibrated=True, idx=np.array(idx),
                 track=np.array(track), boxes=np.array(boxes, float), xy=np.array(xy, float),
                 ball_idx=np.array(ball_idx), ball_xy=np.array(ball_xy, float), ball_img=np.zeros((len(ball_idx), 2)),
                 camera=CameraModel(np.tile(np.eye(3), (n, 1, 1)), []))
    d.valid_tracks = {1}
    d.entity = {1: "p7"}
    d.team = {1: 0}
    return d


def test_shot_on_target_with_shooter_and_goal_chance():
    shots = analytics.detect_shots(_shot_clip())
    assert len(shots) == 1
    s = shots[0]
    assert abs(s["t"] - 2.0) < 0.15 and s["goal_x"] == 105 and s["on_target"] and s["speed_kmh"] >= 70
    assert s["entity"] == "p7" and s["team"] == 0 and (s["ball_at_line"] or s["lost_near_goal"])
    cheer = [{"kind": "gejuich", "t": 3.5}]
    assert analytics.goal_chance(s, cheer) == "waarschijnlijk"
    assert analytics.goal_chance({**s, "on_target": False}, cheer) is None
    # schuin naast het doel: wel een schot richting doel, niet op doel
    wide = analytics.detect_shots(_shot_clip(vy=8.0))
    assert len(wide) == 1 and not wide[0]["on_target"]
    # rustig rollen (een pass) is geen schot
    assert analytics.detect_shots(_shot_clip(speed=6.0)) == []
