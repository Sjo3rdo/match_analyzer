"""Verwerking van een clip: preview maken, detecteren, volgen, camerabeweging, teams.

Draait in één achtergrondthread (één clip tegelijk), zodat de laptop bruikbaar blijft.
De kalibratie zit hier bewust *niet* in: die wordt pas bij de analyse toegepast, zodat
je sleutelframes kunt toevoegen of verbeteren zonder alles opnieuw te verwerken.
"""
from __future__ import annotations

import logging
import queue
import subprocess
import threading
import traceback
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from . import config, jersey, teams
from .calibration import MotionEstimator
from .storage import Store, clip_dir
from .tracking import Tracker

log = logging.getLogger(__name__)

MAX_COLOR_SAMPLES = 40
OCR_CROPS = 6


def ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # pragma: no cover
        return "ffmpeg"


def probe(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Kan video niet openen: {path.name}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    info = {"fps": fps, "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), "duration": n / fps if fps else 0}
    cap.release()
    return info


def make_preview(src: Path, dst: Path) -> None:
    """H.264-versie (720p) die elke browser afspeelt, ook als het origineel HEVC is."""
    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(src),
           "-vf", "scale=-2:'min(720,ih)'", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
           "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", str(dst)]
    subprocess.run(cmd, check=True)


def read_frame(path: Path, t: float) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError("Frame niet leesbaar")
    return frame


class _TrackStats:
    def __init__(self):
        self.n = 0
        self.t0 = self.t1 = 0.0
        self.colors: list[np.ndarray] = []
        self.best: list[tuple[float, np.ndarray]] = []  # (hoogte, uitsnede)

    def add(self, t: float, frame: np.ndarray, box: np.ndarray, sample: bool) -> None:
        if self.n == 0:
            self.t0 = t
        self.n += 1
        self.t1 = t
        if sample and len(self.colors) < MAX_COLOR_SAMPLES:
            c = teams.shirt_color(frame, box)
            if c is not None:
                self.colors.append(c)
        h = box[3] - box[1]
        if len(self.best) < OCR_CROPS or h > self.best[-1][0]:
            x1, y1, x2, y2 = [int(v) for v in box]
            crop = frame[max(0, y1):y2, max(0, x1):x2].copy()
            if crop.size:
                self.best.append((h, crop))
                self.best.sort(key=lambda p: -p[0])
                del self.best[OCR_CROPS:]


def process_clip(store: Store, clip_id: int, detector=None) -> None:
    clip = store.one("SELECT * FROM clips WHERE id = ?", (clip_id,))
    if clip is None:
        return
    src = Path(clip["path"])
    out_dir = clip_dir(clip_id)
    store.clear_clip_results(clip_id)

    store.set_clip_status(clip_id, "preview", 0.0, "Preview-video maken")
    preview = out_dir / "preview.mp4"
    if not preview.exists():
        make_preview(src, preview)

    if detector is None:
        from .detection import get_detector

        store.set_clip_status(clip_id, "analyse", 0.02, "Model laden")
        detector = get_detector()

    cap = cv2.VideoCapture(str(src))
    fps = clip["fps"] or cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    stride = max(1, int(round(fps / config.TARGET_FPS)))
    eff_fps = fps / stride
    tracker = Tracker(eff_fps)
    motion = MotionEstimator()
    stats: dict[int, _TrackStats] = defaultdict(_TrackStats)
    inter: list[np.ndarray] = []
    frame_rows, det_rows, ball_rows = [], [], []

    def flush():
        with store.tx() as c:
            c.executemany("INSERT INTO frames VALUES (?,?,?)", frame_rows)
            c.executemany("INSERT INTO detections VALUES (?,?,?,?,?,?,?,?)", det_rows)
            c.executemany("INSERT INTO ball VALUES (?,?,?,?,?)", ball_rows)
        frame_rows.clear(), det_rows.clear(), ball_rows.clear()

    frame_no, idx = 0, 0
    while True:
        if frame_no % stride:
            if not cap.grab():
                break
            frame_no += 1
            continue
        ok, frame = cap.read()
        if not ok:
            break
        t = frame_no / fps
        det = detector(frame)
        H = motion.step(frame, det.boxes)
        inter.append(H)
        tracked = tracker.update(det.boxes, det.scores, H if idx else None)
        frame_rows.append((clip_id, idx, t))
        for tid, box, score in tracked:
            det_rows.append((clip_id, idx, tid, *map(float, box), score))
            stats[tid].add(t, frame, box, sample=idx % 3 == 0)
        if det.ball:
            ball_rows.append((clip_id, idx, *map(float, det.ball)))
        idx += 1
        frame_no += 1
        if idx % 50 == 0:
            flush()
            store.set_clip_status(clip_id, "analyse", 0.05 + 0.9 * frame_no / total,
                                  f"Frame {frame_no} van {total}")
    cap.release()
    flush()
    np.save(out_dir / "motion.npy", np.array(inter) if inter else np.zeros((0, 3, 3)))

    store.set_clip_status(clip_id, "analyse", 0.96, "Teams en rugnummers bepalen")
    _finish_tracks(store, clip_id, stats, out_dir)
    store.set_clip_status(clip_id, "klaar", 1.0, f"{idx} frames geanalyseerd")


def _finish_tracks(store: Store, clip_id: int, stats: dict[int, _TrackStats], out_dir: Path) -> None:
    thumbs = out_dir / "thumbs"
    thumbs.mkdir(exist_ok=True)
    ids = [tid for tid, s in stats.items() if s.colors]
    colors = np.array([np.median(stats[t].colors, axis=0) for t in ids]) if ids else np.zeros((0, 3))
    weights = np.array([stats[t].n for t in ids])
    team_of = dict(zip(ids, teams.assign_teams(colors, weights).tolist()))
    color_of = {t: teams.lab_to_hex(c) for t, c in zip(ids, colors)}
    use_ocr = jersey.available()
    rows = []
    for tid, s in stats.items():
        if s.best:
            cv2.imwrite(str(thumbs / f"{tid}.jpg"), s.best[0][1])
        number, nconf = jersey.read_number([c for _, c in s.best]) if use_ocr and s.n >= 10 else (None, 0.0)
        team = team_of.get(tid, teams.TEAM_UNKNOWN)
        rows.append((clip_id, tid, s.n, s.t0, s.t1, color_of.get(tid), team, team, number, nconf))
    with store.tx() as c:
        c.executemany("INSERT INTO tracks (clip_id, track_id, n_frames, t_start, t_end, color, team, "
                      "team_auto, jersey_guess, jersey_conf) VALUES (?,?,?,?,?,?,?,?,?,?)", rows)


class Worker:
    """Eén achtergrondthread die clips in volgorde verwerkt."""

    def __init__(self, store: Store):
        self.store = store
        self.q: queue.Queue[int] = queue.Queue()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def submit(self, clip_id: int) -> None:
        self.store.set_clip_status(clip_id, "wachtrij", 0.0, "In de wachtrij")
        self.q.put(clip_id)

    def _run(self) -> None:
        while True:
            clip_id = self.q.get()
            try:
                process_clip(self.store, clip_id)
            except Exception as e:  # noqa: BLE001
                log.error("Verwerking clip %s mislukt\n%s", clip_id, traceback.format_exc())
                self.store.set_clip_status(clip_id, "fout", None, str(e))
