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


def test_sun_and_shade_stay_one_team():
    """Hetzelfde shirt in zon en schaduw verschilt vooral in helderheid: dat zijn geen twee teams."""
    rng = np.random.default_rng(2)
    green_sun = rng.normal([220, 112, 135], 3, (6, 3))
    green_shade = rng.normal([140, 106, 133], 3, (6, 3))
    blue = rng.normal([90, 133, 102], 3, (10, 3))
    labels = assign_teams(np.vstack([green_sun, green_shade, blue]), np.full(22, 50))
    assert len(set(labels[:12])) == 1 and labels[0] in (0, 1)
    assert len(set(labels[12:])) == 1 and labels[12] != labels[0]


def test_spectators_do_not_form_a_team():
    rng = np.random.default_rng(3)
    yellow = rng.normal([215, 125, 170], 3, (10, 3))
    green = rng.normal([200, 110, 135], 3, (10, 3))
    jackets = rng.normal([60, 128, 126], 4, (8, 3))  # stilstaand publiek in donkere jassen, lang in beeld
    w = np.r_[np.full(20, 40), np.full(8, 300)]
    exclude = np.r_[np.zeros(20, bool), np.ones(8, bool)]
    labels = assign_teams(np.vstack([yellow, green, jackets]), w, exclude=exclude)
    assert len(set(labels[:10])) == 1 and len(set(labels[10:20])) == 1 and labels[0] != labels[10]
    assert all(lab == TEAM_OTHER for lab in labels[20:])


def test_teams_aligned_between_videos(tmp_path):
    """Een tweede video krijgt dezelfde teamnummers als de eerste (welke kleur 'Thuis' is)."""
    from app import config
    from app.pipeline import assign_clip_teams
    from app.storage import Store
    from app.teams import lab_to_hex

    config.CLIPS_DIR = tmp_path / "clips"
    store = Store(tmp_path / "db.sqlite")
    mid = store.run("INSERT INTO matches (name) VALUES ('x')")
    red, blue = np.array([130, 190, 160.0]), np.array([80, 140, 90.0])
    rng = np.random.default_rng(4)
    clips = []
    for _ in range(2):
        cid = store.run("INSERT INTO clips (match_id, filename, path, status) VALUES (?, 'v', '', 'klaar')", (mid,))
        clips.append(cid)
        for i in range(18):
            c = (red if i % 2 else blue) + rng.normal(0, 2, 3)
            store.run("INSERT INTO tracks (clip_id, track_id, n_frames, team, team_auto, color) VALUES (?,?,?,?,?,?)",
                      (cid, i + 1, 50, -1, -1, lab_to_hex(c)))
    a, b = clips
    for red_team in (0, 1):  # video 1 met de hand op 'rood = red_team' zetten; video 2 moet volgen
        store.run("UPDATE tracks SET team = CASE WHEN track_id % 2 = 1 THEN ? ELSE ? END WHERE clip_id = ?",
                  (red_team, 1 - red_team, a))
        assign_clip_teams(store, b, keep_manual=False)
        got = {r["track_id"] % 2: r["team"] for r in store.all("SELECT track_id, team FROM tracks WHERE clip_id = ?", (b,))}
        assert got[1] == red_team and got[0] == 1 - red_team


def test_bright_shirt_far_from_mean_but_clearly_one_team():
    from app.teams import hex_to_lab
    yellow = [hex_to_lab(c) for c in ("#e0c678", "#eacf7f", "#eccf75", "#ecd385", "#d5bc71")]
    green = [hex_to_lab(c) for c in ("#3e7967", "#659c82", "#538872", "#628c78", "#73ae8e")]
    colors = np.array(yellow + green + [hex_to_lab("#f7d564"), hex_to_lab("#c9545b")])
    labels = assign_teams(colors, np.full(len(colors), 100))
    assert labels[-2] == labels[0]  # fel geel in de zon: toch het gele team
    assert labels[-1] == TEAM_OTHER  # rode scheidsrechter blijft overig
