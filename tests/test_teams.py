import numpy as np

from app.teams import TEAM_OTHER, assign_teams, shirt_color


def test_assign_teams_splits_colors():
    rng = np.random.default_rng(1)
    red = rng.normal([140, 190, 160], 3, (11, 3))
    blue = rng.normal([80, 140, 90], 3, (11, 3))
    ref = rng.normal([20, 128, 128], 3, (1, 3))
    labels = assign_teams(np.vstack([red, blue, ref]), np.r_[np.full(22, 100), 60])
    assert len(set(labels[:11])) == 1 and len(set(labels[11:22])) == 1
    assert labels[0] != labels[11]
    assert labels[22] == TEAM_OTHER


def test_shirt_color_ignores_grass():
    frame = np.zeros((200, 100, 3), np.uint8)
    frame[:] = (40, 160, 40)  # gras (BGR)
    frame[40:100, 30:70] = (0, 0, 220)  # rood shirt
    c = shirt_color(frame, np.array([10, 20, 90, 190]))
    assert c is not None and c[1] > 150  # Lab a* hoog = rood
