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
    x REAL, y REAL, conf REAL,
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
    points TEXT NOT NULL
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
            for table in ("frames", "detections", "ball", "tracks"):
                c.execute(f"DELETE FROM {table} WHERE clip_id = ?", (clip_id,))

    # --- kalibratie ------------------------------------------------------
    def keyframes(self, clip_id: int) -> list[dict[str, Any]]:
        rows = self.all("SELECT * FROM keyframes WHERE clip_id = ? ORDER BY t", (clip_id,))
        for r in rows:
            r["points"] = json.loads(r["points"])
        return rows


def clip_dir(clip_id: int) -> Path:
    d = config.CLIPS_DIR / str(clip_id)
    d.mkdir(parents=True, exist_ok=True)
    return d
