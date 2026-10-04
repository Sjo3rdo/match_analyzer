"""Opslag in SQLite. Zware per-frame data (camerabeweging) staat als .npy naast de clip."""
from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS matches (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    date TEXT,
    team0_name TEXT DEFAULT 'Thuis',
    team1_name TEXT DEFAULT 'Uit',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS clips (
    id INTEGER PRIMARY KEY,
    match_id INTEGER NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    filename TEXT NOT NULL,
    path TEXT NOT NULL,
    order_idx INTEGER DEFAULT 0,
    period INTEGER DEFAULT 1,
    start_minute REAL DEFAULT 0,
    fps REAL, width INTEGER, height INTEGER, duration REAL,
    status TEXT DEFAULT 'nieuw',
    analysis_version INTEGER DEFAULT 0,
    progress REAL DEFAULT 0,
    message TEXT
);
CREATE TABLE IF NOT EXISTS frames (
    clip_id INTEGER NOT NULL REFERENCES clips(id) ON DELETE CASCADE,
    idx INTEGER NOT NULL,
    t REAL NOT NULL,
    PRIMARY KEY (clip_id, idx)
);
CREATE TABLE IF NOT EXISTS detections (
    clip_id INTEGER NOT NULL REFERENCES clips(id) ON DELETE CASCADE,
    idx INTEGER NOT NULL,
    track_id INTEGER NOT NULL,
    x1 REAL, y1 REAL, x2 REAL, y2 REAL, conf REAL
);
CREATE INDEX IF NOT EXISTS det_clip_idx ON detections(clip_id, idx);
CREATE INDEX IF NOT EXISTS det_clip_track ON detections(clip_id, track_id);
CREATE TABLE IF NOT EXISTS ball (
    clip_id INTEGER NOT NULL REFERENCES clips(id) ON DELETE CASCADE,
    idx INTEGER NOT NULL,
    x REAL, y REAL, conf REAL, src TEXT,
    PRIMARY KEY (clip_id, idx)
);
CREATE TABLE IF NOT EXISTS tracks (
    clip_id INTEGER NOT NULL REFERENCES clips(id) ON DELETE CASCADE,
    track_id INTEGER NOT NULL,
    n_frames INTEGER,
    t_start REAL, t_end REAL,
    color TEXT,
    team INTEGER DEFAULT -1,
    team_auto INTEGER DEFAULT -1,
    jersey_guess TEXT,
    jersey_conf REAL,
    player_id INTEGER REFERENCES players(id) ON DELETE SET NULL,
    PRIMARY KEY (clip_id, track_id)
);
CREATE TABLE IF NOT EXISTS players (
    id INTEGER PRIMARY KEY,
    match_id INTEGER NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    number TEXT,
    team INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS keyframes (
    id INTEGER PRIMARY KEY,
    clip_id INTEGER NOT NULL REFERENCES clips(id) ON DELETE CASCADE,
    t REAL NOT NULL,
    points TEXT NOT NULL,
    auto INTEGER DEFAULT 0,
    score TEXT,
    accepted INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS moments (
    id INTEGER PRIMARY KEY,
    match_id INTEGER NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    clip_id INTEGER NOT NULL REFERENCES clips(id) ON DELETE CASCADE,
    start REAL NOT NULL,
    end REAL NOT NULL,
    label TEXT DEFAULT 'Moment',
    comment TEXT,
    players TEXT DEFAULT '[]',
    spotlight_player_id INTEGER REFERENCES players(id) ON DELETE SET NULL,
    spotlights TEXT,
    drawings TEXT DEFAULT '[]',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS shots (
    id INTEGER PRIMARY KEY,
    match_id INTEGER NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    clip_id INTEGER NOT NULL REFERENCES clips(id) ON DELETE CASCADE,
    t REAL NOT NULL,
    t_end REAL,
    status TEXT NOT NULL DEFAULT 'bevestigd',  -- 'bevestigd' of 'afgewezen' (voorstel dat niet klopte)
    goal INTEGER DEFAULT 0,
    on_target INTEGER,
    team INTEGER,
    player_id INTEGER REFERENCES players(id) ON DELETE SET NULL,
    x REAL,
    y REAL,
    goal_x REAL,
    auto INTEGER DEFAULT 0,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS squads (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    color TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS squad_players (
    id INTEGER PRIMARY KEY,
    squad_id INTEGER NOT NULL REFERENCES squads(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    number TEXT
);
CREATE TABLE IF NOT EXISTS venues (
    id INTEGER PRIMARY KEY,
    lat REAL NOT NULL,
    lon REAL NOT NULL,
    pitch_length REAL,
    pitch_width REAL,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS ball_manual (
    clip_id INTEGER NOT NULL REFERENCES clips(id) ON DELETE CASCADE,
    t REAL NOT NULL,
    x REAL,
    y REAL,
    PRIMARY KEY (clip_id, t)
);
CREATE TABLE IF NOT EXISTS player_profiles (
    id INTEGER PRIMARY KEY,
    squad_id INTEGER REFERENCES squads(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    match_player_id INTEGER REFERENCES players(id) ON DELETE CASCADE,
    emb BLOB NOT NULL,
    n INTEGER DEFAULT 0,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS track_embeds (
    clip_id INTEGER NOT NULL REFERENCES clips(id) ON DELETE CASCADE,
    track_id INTEGER NOT NULL,
    emb BLOB NOT NULL,
    PRIMARY KEY (clip_id, track_id)
);
CREATE TABLE IF NOT EXISTS markers (
    id INTEGER PRIMARY KEY,
    match_id INTEGER NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
    clip_id INTEGER NOT NULL REFERENCES clips(id) ON DELETE CASCADE,
    t REAL NOT NULL,
    label TEXT,
    player_id INTEGER REFERENCES players(id) ON DELETE SET NULL
);
"""

_lock = threading.RLock()


def connect(path: Path | None = None) -> sqlite3.Connection:
    path = path or config.DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


class Store:
    """Kleine wrapper; één connectie, beschermd met een lock (worker + API)."""

    def __init__(self, path: Path | None = None):
        self.conn = connect(path)
        self.conn.executescript(SCHEMA)
        self._migrate()

    def _migrate(self) -> None:
        cols = {r["name"] for r in self.all("PRAGMA table_info(clips)")}
        if "analysis_version" not in cols:
            self.run("ALTER TABLE clips ADD COLUMN analysis_version INTEGER DEFAULT 0")
        kcols = {r["name"] for r in self.all("PRAGMA table_info(keyframes)")}
        if "auto" not in kcols:
            self.run("ALTER TABLE keyframes ADD COLUMN auto INTEGER DEFAULT 0")
            self.run("ALTER TABLE keyframes ADD COLUMN score TEXT")
        if "accepted" not in kcols:  # automatisch sleutelframe door de gebruiker goedgekeurd
            self.run("ALTER TABLE keyframes ADD COLUMN accepted INTEGER DEFAULT 0")
        # status van het automatisch bijstellen van de kalibratie
        for col, typ in (("calib_status", "TEXT"), ("calib_progress", "REAL"), ("calib_message", "TEXT")):
            if col not in cols:
                self.run(f"ALTER TABLE clips ADD COLUMN {col} {typ}")
        # GPS uit de video en de (geschatte) camerapositie op het veld
        for col, typ in (("gps_lat", "REAL"), ("gps_lon", "REAL"), ("gps_acc", "REAL"), ("device", "TEXT"),
                         ("cam_x", "REAL"), ("cam_y", "REAL"), ("cam_h", "REAL"), ("cam_source", "TEXT")):
            if col not in cols:
                self.run(f"ALTER TABLE clips ADD COLUMN {col} {typ}")
        # Analysekeuze (nauwkeurig/snel) en speelrichting per video (NULL = automatisch: 2e helft omdraaien)
        for col, typ in (("analysis_mode", "TEXT"), ("flip", "INTEGER")):
            if col not in cols:
                self.run(f"ALTER TABLE clips ADD COLUMN {col} {typ}")
        # Begin van de opname (seconden sinds 1970), uit de metadata van de video
        if "rec_start" not in cols:
            self.run("ALTER TABLE clips ADD COLUMN rec_start REAL")
        # Door de gebruiker goedgekeurd om de app mee te trainen
        if "train_ok" not in cols:
            self.run("ALTER TABLE clips ADD COLUMN train_ok INTEGER DEFAULT 0")
        # Spotlight op meer spelers tegelijk (JSON-lijst); spotlight_player_id blijft de eerste
        if "spotlights" not in {r["name"] for r in self.all("PRAGMA table_info(moments)")}:
            self.run("ALTER TABLE moments ADD COLUMN spotlights TEXT")
        bcols = {r["name"] for r in self.all("PRAGMA table_info(ball)")}
        if "src" not in bcols:  # hoe de bal gevonden is: 'det' (gewoon beeld), 'zoom' (ingezoomd), 'scan'
            self.run("ALTER TABLE ball ADD COLUMN src TEXT")
        mcols = {r["name"] for r in self.all("PRAGMA table_info(matches)")}
        # Veldmaten per wedstrijd, en de shirtkleur per team (om video's gelijk te trekken)
        for col, typ in (("pitch_length", "REAL"), ("pitch_width", "REAL"), ("team0_color", "TEXT"),
                         ("team1_color", "TEXT"), ("team0_squad", "INTEGER"), ("team1_squad", "INTEGER"),
                         ("half_length", "INTEGER")):
            if col not in mcols:
                self.run(f"ALTER TABLE matches ADD COLUMN {col} {typ}")
        # Oude 'markers' (één tijdstip) worden clips (begin + eind), zoals in een video-editor
        with self.tx() as c:
            for m in c.execute("SELECT * FROM markers").fetchall():
                players = json.dumps([m["player_id"]] if m["player_id"] else [])
                c.execute("INSERT INTO moments (match_id, clip_id, start, end, label, players, spotlight_player_id) "
                          "VALUES (?,?,?,?,?,?,?)", (m["match_id"], m["clip_id"], max(0.0, m["t"] - 6), m["t"] + 4,
                                                     m["label"], players, m["player_id"]))
            c.execute("DELETE FROM markers")

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        with _lock:
            self.conn.execute("BEGIN")
            try:
                yield self.conn
                self.conn.execute("COMMIT")
            except Exception:
                self.conn.execute("ROLLBACK")
                raise

    def all(self, sql: str, args: tuple = ()) -> list[dict[str, Any]]:
        with _lock:
            return [dict(r) for r in self.conn.execute(sql, args).fetchall()]

    def one(self, sql: str, args: tuple = ()) -> dict[str, Any] | None:
        with _lock:
            r = self.conn.execute(sql, args).fetchone()
            return dict(r) if r else None

    def rows(self, sql: str, args: tuple = ()) -> list[tuple]:
        """Rijen als gewone tuples: veel sneller dan dicts bij grote aantallen (detecties)."""
        with _lock:
            cur = self.conn.cursor()
            cur.row_factory = None
            try:
                return cur.execute(sql, args).fetchall()
            finally:
                cur.close()

    def run(self, sql: str, args: tuple = ()) -> int:
        with _lock:
            cur = self.conn.execute(sql, args)
            return cur.lastrowid

    # --- clips -----------------------------------------------------------
    def set_clip_status(self, clip_id: int, status: str, progress: float | None = None,
                        message: str | None = None) -> None:
        sets, args = ["status = ?"], [status]
        if progress is not None:
            sets.append("progress = ?")
            args.append(progress)
        if message is not None:
            sets.append("message = ?")
            args.append(message)
        self.run(f"UPDATE clips SET {', '.join(sets)} WHERE id = ?", (*args, clip_id))

    def clear_clip_results(self, clip_id: int) -> None:
        with self.tx() as c:
            for table in ("frames", "detections", "ball", "tracks", "track_embeds"):
                c.execute(f"DELETE FROM {table} WHERE clip_id = ?", (clip_id,))

    # --- kalibratie ------------------------------------------------------
    def moment(self, moment_id: int) -> dict[str, Any] | None:
        m = self.one("SELECT * FROM moments WHERE id = ?", (moment_id,))
        if m:
            _parse_moment(m)
        return m

    def moments(self, match_id: int) -> list[dict[str, Any]]:
        rows = self.all("SELECT m.* FROM moments m JOIN clips c ON c.id = m.clip_id WHERE m.match_id = ? "
                        "ORDER BY c.order_idx, c.id, m.start", (match_id,))
        for m in rows:
            _parse_moment(m)
        return rows

    def keyframes(self, clip_id: int) -> list[dict[str, Any]]:
        rows = self.all("SELECT * FROM keyframes WHERE clip_id = ? ORDER BY t", (clip_id,))
        for r in rows:
            r["points"] = json.loads(r["points"])
        return rows


def clip_dir(clip_id: int) -> Path:
    d = config.CLIPS_DIR / str(clip_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _parse_moment(m: dict[str, Any]) -> None:
    m["players"] = json.loads(m["players"] or "[]")
    m["drawings"] = json.loads(m["drawings"] or "[]")
    spots = json.loads(m.get("spotlights") or "null")
    if spots is None:  # clips van vóór de spotlight op meer spelers
        spots = [m["spotlight_player_id"]] if m.get("spotlight_player_id") else []
    m["spotlights"] = spots
