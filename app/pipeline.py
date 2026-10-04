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
import time
import traceback
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from . import config, jersey, teams
from .calibration import MotionEstimator, apply_h, cumulative
from .storage import Store, clip_dir
from .tracking import Tracker, stationary_ids, stitch_tracks

log = logging.getLogger(__name__)

MAX_COLOR_SAMPLES = 40
OCR_CROPS = 6


def ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # pragma: no cover
        return "ffmpeg"


# Versie van de analyse; clips die met een oudere versie zijn verwerkt, krijgen in de
# interface het advies om opnieuw te analyseren.
ANALYSIS_VERSION = 4  # 3: nieuwe teamindeling (zon/schaduw, toeschouwers, gelijk tussen video's); 4: balvolger
TEAMS_VERSION = 3  # wat alleen opnieuw teams indelen oplevert (zonder opnieuw te analyseren)


def probe(path: Path) -> dict:
    """Videogegevens. fps en duur komen van ffmpeg: OpenCV leest bij iPhone-video's
    (variabele framerate) een verkeerde fps uit de container, bijv. 28,7 i.p.v. 30."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Kan video niet openen: {path.name}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    info = {"fps": fps, "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), "duration": n / fps if fps else 0}
    cap.release()
    try:
        from .geo import read_video_metadata

        info.update(read_video_metadata(path, ffmpeg_exe()))
    except Exception:  # noqa: BLE001  (geen GPS is geen probleem)
        pass
    try:
        import imageio_ffmpeg

        gen = imageio_ffmpeg.read_frames(str(path))
        meta = next(gen)
        gen.close()
        if meta.get("fps"):
            info["fps"] = float(meta["fps"])
        if meta.get("duration"):
            info["duration"] = float(meta["duration"])
    except Exception:  # noqa: BLE001  (dan maar de OpenCV-waarden)
        pass
    return info


def make_preview(src: Path, dst: Path, duration: float | None = None, progress=None) -> None:
    """H.264-versie (720p) die elke browser afspeelt, ook als het origineel HEVC is.

    progress(fractie) wordt tussendoor aangeroepen (ffmpeg meldt hoe ver hij is)."""
    tmp = dst.with_suffix(".tmp.mp4")
    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error", "-nostats", "-progress", "pipe:1", "-i", str(src),
           "-vf", "scale=-2:'min(720,ih)'", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
           "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", str(tmp)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    last = 0.0
    for line in proc.stdout:
        key, _, val = line.strip().partition("=")
        if key in ("out_time_us", "out_time_ms") and duration and progress and val.isdigit():
            frac = min(1.0, int(val) / 1e6 / duration)  # (ffmpeg geeft bij beide microseconden)
            if frac - last >= 0.02:
                last = frac
                progress(frac)
    err = proc.stderr.read()
    if proc.wait() != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Preview maken mislukt: {err[-300:]}")
    tmp.replace(dst)  # pas als hij af is: een half bestand blijft nooit staan


def _eta(seconds: float) -> str:
    if seconds < 90:
        return "nog minder dan 2 min"
    if seconds < 3600:
        return f"nog ca. {round(seconds / 60)} min"
    return f"nog ca. {seconds / 3600:.1f} uur".replace(".", ",")


def read_frame(path: Path, t: float) -> tuple[np.ndarray, float]:
    """Exact het frame op tijd t (s), plus de echte tijd van dat frame.

    Direct springen met OpenCV komt bij HEVC een paar frames te vroeg uit. Daarom springen
    we iets ervoor en lezen we vooruit tot het juiste frame, met dezelfde decoder als de
    analyse (zodat beeld en analyse exact overeenkomen)."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t - 1.0) * 1000)
    best, best_t = None, None
    for _ in range(int(3 * fps) + 10):
        if not cap.grab():
            break
        ft = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        if best is not None and ft > t + 0.5 / fps:
            break
        ok, frame = cap.retrieve()
        if ok:
            best, best_t = frame, ft
        if ft >= t - 0.5 / fps:
            break
    cap.release()
    if best is None:
        raise ValueError("Frame niet leesbaar")
    return best, best_t


class _TrackStats:
    def __init__(self):
        self.n = 0
        self.t0 = self.t1 = 0.0
        self.colors: list[np.ndarray] = []
        self.best: list[tuple[float, np.ndarray]] = []  # (hoogte, uitsnede)
        self.idx: list[int] = []  # voor het achteraf aan elkaar plakken van tracks
        self.foot: list[tuple[float, float]] = []
        self.h: list[float] = []

    def merge(self, other: "_TrackStats") -> None:
        self.n += other.n
        self.t0, self.t1 = min(self.t0, other.t0), max(self.t1, other.t1)
        self.colors += other.colors
        self.best = sorted(self.best + other.best, key=lambda p: -p[0])[:OCR_CROPS]

    def add(self, t: float, frame: np.ndarray, box: np.ndarray, color: np.ndarray | None, idx: int = 0) -> None:
        if self.n == 0:
            self.t0 = t
        self.n += 1
        self.t1 = t
        self.idx.append(idx)
        self.foot.append(((box[0] + box[2]) / 2, box[3]))
        self.h.append(box[3] - box[1])
        if color is not None:
            self.colors.append(color)
            if len(self.colors) > 2 * MAX_COLOR_SAMPLES:  # geheugen beperken, spreiding houden
                self.colors = self.colors[::2]
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
        make_preview(src, preview, clip.get("duration"),
                     lambda f: store.set_clip_status(clip_id, "preview", f, f"Preview-video maken ({round(100 * f)}%)"))

    if detector is None:
        from .detection import get_detector

        store.set_clip_status(clip_id, "analyse", 0.02, "Model laden")
        detector = get_detector(clip.get("analysis_mode") or "nauwkeurig")

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
            c.executemany("INSERT INTO ball (clip_id, idx, x, y, conf, src) VALUES (?,?,?,?,?,?)", ball_rows)
        frame_rows.clear(), det_rows.clear(), ball_rows.clear()

    # Een lopende band met drie werkers die tegelijk bezig zijn:
    # 1. de lezer pakt de video uit (processor),
    # 2. de detector zoekt spelers en bal, een paar beelden tegelijk (grafische chip),
    # 3. de verwerker meet de camerabeweging, shirtkleuren en volgt de spelers (processor).
    # Zo wacht de grafische chip niet op het uitpakken en andersom.
    DONE = object()
    q_frames: queue.Queue = queue.Queue(maxsize=4)
    q_dets: queue.Queue = queue.Queue(maxsize=4)
    errors: list[BaseException] = []
    stop = threading.Event()
    counter = {"frame_no": 0}

    def put(q, item) -> bool:
        while not stop.is_set():
            try:
                q.put(item, timeout=0.2)
                return True
            except queue.Full:
                continue
        return False

    def reader():
        try:
            frame_no = 0
            while not stop.is_set():
                if frame_no % stride:
                    if not cap.grab():
                        break
                    frame_no += 1
                    continue
                ok, frame = cap.read()
                if not ok:
                    break
                t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # echte tijdstempel (variabele framerate!)
                frame_no += 1
                if not put(q_frames, (frame, t, frame_no)):
                    break
        except BaseException as e:  # noqa: BLE001
            errors.append(e)
        finally:
            put(q_frames, DONE)

    def worker_post():
        idx = 0
        started = last_report = time.time()
        try:
            while True:
                item = q_dets.get()
                if item is DONE:
                    break
                frame, t, frame_no, det = item
                H = motion.step(frame, det.boxes)
                inter.append(H)
                colors = [teams.shirt_color(frame, b) for b in det.boxes]
                color_of = {np.asarray(b, np.float64).tobytes(): c for b, c in zip(det.boxes, colors)}
                tracked = tracker.update(det.boxes, det.scores, H if idx else None, colors)
                frame_rows.append((clip_id, idx, t))
                for tid, box, score in tracked:
                    det_rows.append((clip_id, idx, tid, *map(float, box), score))
                    stats[tid].add(t, frame, box, color_of.get(np.asarray(box, np.float64).tobytes()), idx)
                if det.ball:
                    ball_rows.append((clip_id, idx, *map(float, det.ball[:3]), det.ball[3] if len(det.ball) > 3 else 'det'))
                idx += 1
                counter["frame_no"] = frame_no
                if idx % 50 == 0:
                    flush()
                if idx == 1 or idx % 50 == 0 or time.time() - last_report > 3:
                    last_report = time.time()
                    frac = min(1.0, frame_no / total)
                    eta = (time.time() - started) / frac * (1 - frac) if frac > 0.02 else None
                    store.set_clip_status(clip_id, "analyse", 0.05 + 0.9 * frac,
                                          f"Frame {frame_no} van {total}" + (f" · {_eta(eta)}" if eta is not None else ""))
        except BaseException as e:  # noqa: BLE001
            errors.append(e)
            stop.set()
            while True:  # de band leeg laten lopen, zodat de andere werkers niet vastlopen
                try:
                    if q_dets.get(timeout=1) is DONE:
                        break
                except queue.Empty:
                    break
        counter["idx"] = idx

    # De bal volgen: kandidaten kiezen die bij het spoor passen, en inzoomen als hij kwijt is
    from .ball import BallTracker

    tracker_ball = BallTracker(int(clip.get("width") or cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920),
                               int(clip.get("height") or cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080))
    zoom = getattr(detector, "detect_balls", None)

    def follow_ball(frame, frame_no, det):
        cands = getattr(det, "balls", None)
        if cands is None:  # (eenvoudige detector: alleen de beste bal)
            return
        pick, src = tracker_ball.choose(cands), "det"
        if pick is None and zoom is not None:
            crop = tracker_ball.crop_box()
            boxes = [crop] if crop else tracker_ball.scan_boxes(frame_no)
            if boxes:
                pick, src = tracker_ball.choose(zoom(frame, boxes)), "zoom" if crop else "scan"
        tracker_ball.update(pick)
        det.ball = (pick.x, pick.y, pick.conf, src) if pick is not None else None

    batch_size = getattr(detector, "batch_size", 1)
    detect_many = getattr(detector, "detect_batch", None)
    threads = [threading.Thread(target=reader, daemon=True), threading.Thread(target=worker_post, daemon=True)]
    for th in threads:
        th.start()
    try:
        ended = False
        while not ended and not stop.is_set():
            batch = []
            while len(batch) < batch_size:
                item = q_frames.get()
                if item is DONE:
                    ended = True
                    break
                batch.append(item)
            if not batch:
                break
            dets = detect_many([b[0] for b in batch]) if detect_many and len(batch) > 1 else [detector(b[0]) for b in batch]
            for (frame, t, frame_no), det in zip(batch, dets):
                follow_ball(frame, frame_no, det)
                if not put(q_dets, (frame, t, frame_no, det)):
                    break
    except BaseException as e:  # noqa: BLE001
        errors.append(e)
        stop.set()
    finally:
        try:  # einde van de band melden aan de verwerker
            q_dets.put(DONE, timeout=60)
        except queue.Full:
            stop.set()
        for th in threads:
            th.join(timeout=60)
    if errors:
        cap.release()
        raise errors[0]
    idx = counter.get("idx", 0)
    cap.release()
    flush()
    np.save(out_dir / "motion.npy", np.array(inter) if inter else np.zeros((0, 3, 3)))
    _drop_static_balls(store, clip_id, inter)

    store.set_clip_status(clip_id, "analyse", 0.95, "Stukjes van dezelfde speler aan elkaar plakken")
    _stitch(store, clip_id, stats, inter, eff_fps)
    store.set_clip_status(clip_id, "analyse", 0.96, "Teams en rugnummers bepalen")
    _finish_tracks(store, clip_id, stats, out_dir)
    store.set_clip_status(clip_id, "analyse", 0.98, "Geluid beluisteren (gejuich, fluitsignalen)")
    try:
        from . import audio

        (out_dir / "audio_events.json").unlink(missing_ok=True)
        audio.clip_events(clip, out_dir, ffmpeg_exe())
    except Exception:  # noqa: BLE001  (geen geluid of iets vreemds: geen hoogtepunten, wel een analyse)
        log.warning("Geluid van clip %s niet geanalyseerd\n%s", clip_id, traceback.format_exc())
    store.run("UPDATE clips SET analysis_version = ? WHERE id = ?", (ANALYSIS_VERSION, clip_id))
    store.set_clip_status(clip_id, "klaar", 1.0, f"{idx} frames geanalyseerd")


def _stitch(store: Store, clip_id: int, stats: dict[int, _TrackStats], inter: list, fps: float) -> None:
    """Tracks die bij dezelfde speler horen samenvoegen (in cameragecorrigeerde coördinaten)."""
    if not stats or not inter:
        return
    A = cumulative(np.array(inter))
    tracklets = {}
    for tid, s in stats.items():
        idx = np.array(s.idx)
        foot = np.array(s.foot, float).reshape(-1, 2)
        ref = np.array([apply_h(A[i], f[None])[0] for i, f in zip(idx, foot)]).reshape(-1, 2)
        tracklets[tid] = {"idx": idx, "foot": ref, "h": np.array(s.h),
                          "color": np.median(s.colors, axis=0) if s.colors else None}
    root = stitch_tracks(tracklets, fps)
    moves = [(r, clip_id, t) for t, r in root.items() if r != t]
    if not moves:
        return
    with store.tx() as c:
        c.executemany("UPDATE detections SET track_id = ? WHERE clip_id = ? AND track_id = ?", moves)
    for t, r in sorted(root.items(), key=lambda kv: stats[kv[0]].t0):
        if r != t:
            stats[r].merge(stats.pop(t))


def _finish_tracks(store: Store, clip_id: int, stats: dict[int, _TrackStats], out_dir: Path) -> None:
    thumbs = out_dir / "thumbs"
    thumbs.mkdir(exist_ok=True)
    ids = [tid for tid, s in stats.items() if s.colors]
    colors = np.array([np.median(stats[t].colors, axis=0) for t in ids]) if ids else np.zeros((0, 3))
    color_of = {t: teams.lab_to_hex(c) for t, c in zip(ids, colors)}
    use_ocr = jersey.available()
    rows = []
    for tid, s in stats.items():
        if s.best:
            cv2.imwrite(str(thumbs / f"{tid}.jpg"), s.best[0][1])
        number, nconf = jersey.read_number([c for _, c in s.best]) if use_ocr and s.n >= 10 else (None, 0.0)
        rows.append((clip_id, tid, s.n, s.t0, s.t1, color_of.get(tid), teams.TEAM_UNKNOWN, teams.TEAM_UNKNOWN,
                     number, nconf))
    with store.tx() as c:
        c.executemany("INSERT INTO tracks (clip_id, track_id, n_frames, t_start, t_end, color, team, "
                      "team_auto, jersey_guess, jersey_conf) VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
    assign_clip_teams(store, clip_id, keep_manual=False)


def _drop_static_balls(store: Store, clip_id: int, inter: list) -> None:
    """'Ballen' die eigenlijk een vast ding in de achtergrond zijn weggooien (zie ball.static_runs)."""
    from .ball import static_runs

    rows = store.rows("SELECT idx, x, y, conf FROM ball WHERE clip_id = ? ORDER BY idx", (clip_id,))
    if not rows or not inter:
        return
    arr = np.array(rows, float)
    drop = static_runs(arr[:, 0].astype(int), arr[:, 1:3], arr[:, 3], cumulative(np.array(inter)))
    if drop.any():
        with store.tx() as c:
            c.executemany("DELETE FROM ball WHERE clip_id = ? AND idx = ?", [(clip_id, int(i)) for i in arr[drop, 0]])


def assign_clip_teams(store: Store, clip_id: int, keep_manual: bool = True) -> dict:
    """Teams van een video (opnieuw) bepalen uit de shirtkleuren, en gelijktrekken met de andere
    video's van dezelfde wedstrijd (anders kan 'Thuis' in de 2e helft ineens 'Uit' heten).

    keep_manual: tracks waarvan de gebruiker het team zelf heeft aangepast, blijven zo."""
    tracks = store.all("SELECT track_id, color, n_frames, team, team_auto FROM tracks WHERE clip_id = ?", (clip_id,))
    clip = store.one("SELECT * FROM clips WHERE id = ?", (clip_id,))
    if not tracks or clip is None:
        return {"swapped": False}
    t = np.array([r[0] for r in store.rows("SELECT t FROM frames WHERE clip_id = ? ORDER BY idx", (clip_id,))])
    det = np.array(store.rows("SELECT idx, track_id, x1, y1, x2, y2 FROM detections WHERE clip_id = ?", (clip_id,)),
                   dtype=np.float64).reshape(-1, 6)
    still: set[int] = set()
    motion = clip_dir(clip_id) / "motion.npy"
    if len(det) and len(t) and motion.exists():
        A = cumulative(np.load(motion)[:len(t)])
        if len(A) == len(t):
            feet = np.stack([(det[:, 2] + det[:, 4]) / 2, det[:, 5]], 1)
            still = stationary_ids(t, det[:, 0].astype(int), det[:, 1].astype(int), feet, det[:, 5] - det[:, 3], A)
    have = [r for r in tracks if teams.hex_to_lab(r["color"]) is not None]
    labels: dict[int, int] = {}
    swapped = False
    if have:
        lab = np.array([teams.hex_to_lab(r["color"]) for r in have])
        w = np.array([r["n_frames"] or 1 for r in have], float)
        # Leren van correcties: wat de gebruiker zelf in een team zette, bepaalt de teamkleuren.
        # Aangewezen toeschouwers doen niet mee.
        manual = [keep_manual and r["team"] != r["team_auto"] for r in have]
        anchors = np.array([r["team"] if m and r["team"] in (0, 1) else -1 for r, m in zip(have, manual)])
        exclude = np.array([r["track_id"] in still or (m and r["team"] == teams.TEAM_SPECTATOR) for r, m in zip(have, manual)])
        new = teams.assign_teams(lab, w, exclude=exclude, anchors=anchors)
        anchored = all(np.any(anchors == t) for t in (0, 1))
        ref, from_clips = _reference_team_colors(store, clip["match_id"], exclude_clip=clip_id)
        mine = teams.team_centers(lab, w, new)
        if not anchored and all(c is not None for c in ref) and all(c is not None for c in mine):
            same = teams.color_distance(mine[0], ref[0]) + teams.color_distance(mine[1], ref[1])
            cross = teams.color_distance(mine[0], ref[1]) + teams.color_distance(mine[1], ref[0])
            # Andere video's van deze wedstrijd: zelfde shirts, dus gewoon het beste kiezen. Opgeslagen
            # teamkleuren (vorige wedstrijd) alleen bij een duidelijk verschil: het kan een uittenue zijn.
            if cross < (same if from_clips else 0.7 * same):
                new = np.where(new == 0, 1, np.where(new == 1, 0, new))
                swapped = True
        labels = {r["track_id"]: int(v) for r, v in zip(have, new)}
    rows = []
    for r in tracks:
        auto = labels.get(r["track_id"], teams.TEAM_UNKNOWN)
        manual = keep_manual and r["team"] != r["team_auto"]
        rows.append((auto, r["team"] if manual else auto, clip_id, r["track_id"]))
    with store.tx() as c:
        c.executemany("UPDATE tracks SET team_auto = ?, team = ? WHERE clip_id = ? AND track_id = ?", rows)
    return {"swapped": swapped, "stationary": len(still)}


def _reference_team_colors(store: Store, match_id: int, exclude_clip: int) -> tuple[list, bool]:
    """Kleur van team 0 en 1 volgens de andere video's van de wedstrijd (True), of anders de
    opgeslagen teamkleuren van de wedstrijd (False)."""
    rows = store.all(
        "SELECT t.color, t.n_frames, COALESCE(p.team, t.team) AS team FROM tracks t "
        "JOIN clips c ON c.id = t.clip_id LEFT JOIN players p ON p.id = t.player_id "
        "WHERE c.match_id = ? AND c.id != ? AND c.status = 'klaar' AND t.color IS NOT NULL", (match_id, exclude_clip))
    out: list[np.ndarray | None] = [None, None]
    for team in (0, 1):
        sel = [(teams.hex_to_lab(r["color"]), r["n_frames"] or 1) for r in rows if r["team"] == team]
        sel = [(c, w) for c, w in sel if c is not None]
        if sel:
            out[team] = np.average([c for c, _ in sel], axis=0, weights=[w for _, w in sel])
    if out[0] is not None and out[1] is not None:
        return out, True
    m = store.one("SELECT * FROM matches WHERE id = ?", (match_id,)) or {}  # dan de opgeslagen teamkleuren
    return [teams.hex_to_lab(m.get("team0_color")), teams.hex_to_lab(m.get("team1_color"))], False


class Worker:
    """Eén achtergrondthread die taken in volgorde uitvoert: analyseren en kalibratie bijstellen."""

    def __init__(self, store: Store):
        self.store = store
        self.q: queue.Queue[tuple[str, int]] = queue.Queue()
        self._pending: set[tuple[str, int]] = set()
        self._lock = threading.Lock()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _put(self, job: tuple[str, int]) -> bool:
        with self._lock:
            if job in self._pending:
                return False
            self._pending.add(job)
        self.q.put(job)
        return True

    def submit(self, clip_id: int) -> None:
        self.store.set_clip_status(clip_id, "wachtrij", 0.0, "In de wachtrij")
        self._put(("process", clip_id))

    def submit_teams(self, clip_id: int) -> None:
        """Teams opnieuw indelen met de huidige methode (zonder opnieuw te analyseren)."""
        self._put(("teams", clip_id))

    def submit_autocalib(self, clip_id: int) -> None:
        if self._put(("autocalib", clip_id)):
            self.store.run("UPDATE clips SET calib_status = 'wachtrij', calib_progress = 0, "
                           "calib_message = 'In de wachtrij' WHERE id = ?", (clip_id,))

    def _run(self) -> None:
        while True:
            kind, clip_id = self.q.get()
            with self._lock:
                self._pending.discard((kind, clip_id))
            if kind == "process":
                try:
                    process_clip(self.store, clip_id)
                except Exception as e:  # noqa: BLE001
                    log.error("Verwerking clip %s mislukt\n%s", clip_id, traceback.format_exc())
                    self.store.set_clip_status(clip_id, "fout", None, str(e))
                    continue
                from . import analytics

                analytics.invalidate(clip_id=clip_id)
                # bestaande handmatige kalibratie automatisch doortrekken over de hele video
                if self.store.one("SELECT 1 AS x FROM keyframes WHERE clip_id = ? AND auto = 0", (clip_id,)):
                    self.submit_autocalib(clip_id)
            elif kind == "teams":
                from . import analytics

                try:
                    assign_clip_teams(self.store, clip_id, keep_manual=True)
                    self.store.run("UPDATE clips SET analysis_version = MAX(COALESCE(analysis_version, 0), ?) WHERE id = ?",
                                   (TEAMS_VERSION, clip_id))
                except Exception:  # noqa: BLE001
                    log.error("Teams indelen clip %s mislukt\n%s", clip_id, traceback.format_exc())
                analytics.invalidate(geometry=False)
            else:
                self._autocalib(clip_id)

    def _autocalib(self, clip_id: int) -> None:
        from . import analytics
        from .autocalib import run_autocalib

        def progress(frac, n):
            self.store.run("UPDATE clips SET calib_status = 'bezig', calib_progress = ?, calib_message = ? WHERE id = ?",
                           (frac, f"{n} automatische sleutelframes", clip_id))
        try:
            progress(0.0, 0)
            res = run_autocalib(self.store, clip_id, progress=progress)
            self.store.run("UPDATE clips SET calib_status = 'klaar', calib_progress = 1, calib_message = ? WHERE id = ?",
                           (f"{res['accepted']} van {res['tried']} momenten automatisch bijgesteld", clip_id))
        except Exception as e:  # noqa: BLE001
            log.error("Automatisch bijstellen clip %s mislukt\n%s", clip_id, traceback.format_exc())
            self.store.run("UPDATE clips SET calib_status = 'fout', calib_message = ? WHERE id = ?", (str(e), clip_id))
        analytics.invalidate(clip_id=clip_id)
