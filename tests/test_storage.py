from app.storage import Store


def test_old_markers_become_moments(tmp_path):
    db = tmp_path / "t.db"
    s = Store(db)
    mid = s.run("INSERT INTO matches (name) VALUES ('x')")
    cid = s.run("INSERT INTO clips (match_id, filename, path) VALUES (?, 'a.mp4', '')", (mid,))
    pid = s.run("INSERT INTO players (match_id, name) VALUES (?, 'Jan')", (mid,))
    s.run("INSERT INTO markers (match_id, clip_id, t, label, player_id) VALUES (?,?,?,?,?)", (mid, cid, 20.0, "Kans", pid))
    s.conn.close()
    s = Store(db)  # opnieuw openen = migratie
    (m,) = s.moments(mid)
    assert (m["start"], m["end"], m["label"], m["players"], m["spotlight_player_id"]) == (14.0, 24.0, "Kans", [pid], pid)
    assert s.all("SELECT * FROM markers") == []
