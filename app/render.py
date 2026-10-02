"""Clips renderen voor export en delen: spotlight op een speler en tekeningen met bevroren beeld.

Werkwijze, als een filmmontage:
1. We lezen de originele video beeld voor beeld (exacte tijden, ook bij iPhone-video's).
2. Op elk beeld tekenen we eventueel de spotlight: een ring onder de voeten van de speler
   met zijn naam erboven. De positie komt uit de analyse en wordt tussen de geanalyseerde
   frames (10 per seconde) vloeiend geïnterpoleerd.
3. Bij een tekening "bevriezen" we het beeld: hetzelfde beeld met de tekening erop wordt
   een paar seconden herhaald. In het geluid voegen we op dat moment stilte in.
4. Alles gaat via een pijp naar ffmpeg, dat er een standaard-mp4 (1080p, 30 fps) van maakt
   die je overal kunt afspelen en delen.
"""
from __future__ import annotations

import subprocess
import tempfile
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np

from . import config
from .pipeline import ffmpeg_exe
from .storage import Store

OUT_W, OUT_H, OUT_FPS = 1920, 1080, 30
SPOT_COLOR = (0, 214, 255)  # BGR geel


def has_audio(path: Path) -> bool:
    r = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", str(path)], capture_output=True, text=True)
    return "Audio:" in r.stderr


def _hex_to_bgr(color: str) -> tuple[int, int, int]:
    c = (color or "#ffd600").lstrip("#")
    if len(c) != 6:
        return (0, 214, 255)
    return (int(c[4:6], 16), int(c[2:4], 16), int(c[0:2], 16))


# --- spotlight ---------------------------------------------------------------------------

class SpotlightTrack:
    """Box van één speler op willekeurige tijd t, geïnterpoleerd uit de analyse."""

    def __init__(self, rows: list[dict]):
        rows = sorted(rows, key=lambda r: r["t"])
        self.t = np.array([r["t"] for r in rows], dtype=np.float64)
        self.boxes = np.array([[r["x1"], r["y1"], r["x2"], r["y2"]] for r in rows], dtype=np.float64).reshape(-1, 4)

    def box_at(self, t: float, max_gap: float = 0.6) -> np.ndarray | None:
        if len(self.t) == 0:
            return None
        i = int(np.searchsorted(self.t, t))
        if 0 < i < len(self.t) and self.t[i] - self.t[i - 1] <= max_gap:
            a, b = self.t[i - 1], self.t[i]
            w = (t - a) / max(1e-6, b - a)
            return (1 - w) * self.boxes[i - 1] + w * self.boxes[i]
        j = min(range(max(0, i - 1), min(len(self.t), i + 1)), key=lambda k: abs(self.t[k] - t))
        return self.boxes[j] if abs(self.t[j] - t) <= 0.15 else None


def load_spotlight(store: Store, clip_id: int, player_id: int, t0: float, t1: float) -> SpotlightTrack:
    rows = store.all(
        "SELECT f.t, d.x1, d.y1, d.x2, d.y2 FROM detections d "
        "JOIN frames f ON f.clip_id = d.clip_id AND f.idx = d.idx "
        "JOIN tracks tr ON tr.clip_id = d.clip_id AND tr.track_id = d.track_id "
        "WHERE d.clip_id = ? AND tr.player_id = ? AND f.t BETWEEN ? AND ? ORDER BY f.t",
        (clip_id, player_id, t0 - 1, t1 + 1))
    # bij twee tracks van dezelfde speler op hetzelfde moment: de grootste box
    best: dict[float, dict] = {}
    for r in rows:
        h = r["y2"] - r["y1"]
        if r["t"] not in best or h > best[r["t"]]["y2"] - best[r["t"]]["y1"]:
            best[r["t"]] = r
    return SpotlightTrack(list(best.values()))


def draw_spotlight(frame: np.ndarray, box: np.ndarray, label: str) -> None:
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    cx, foot = int((x1 + x2) / 2), int(y2)
    ax, ay = max(12, int(0.75 * w)), max(5, int(0.22 * w))
    overlay = frame.copy()
    cv2.ellipse(overlay, (cx, foot), (ax, ay), 0, 0, 360, SPOT_COLOR, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
    thick = max(2, int(frame.shape[1] / 600))
    cv2.ellipse(frame, (cx, foot), (ax, ay), 0, 0, 360, SPOT_COLOR, thick, cv2.LINE_AA)
    if label:
        scale = max(0.6, frame.shape[1] / 2400)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        tx, ty = int(cx - tw / 2), int(y1 - 0.25 * h - 8)
        ty = max(th + 8, ty)
        cv2.rectangle(frame, (tx - 8, ty - th - 8), (tx + tw + 8, ty + 8), (20, 20, 20), -1)
        cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, scale, SPOT_COLOR, 2, cv2.LINE_AA)
        # pijltje naar de speler
        cv2.line(frame, (cx, ty + 8), (cx, int(y1 - 4)), SPOT_COLOR, thick, cv2.LINE_AA)


# --- tekeningen --------------------------------------------------------------------------

def draw_shapes(frame: np.ndarray, shapes: list[dict]) -> None:
    """Vormen in genormaliseerde coördinaten (0..1 van breedte/hoogte van het bronbeeld)."""
    H, W = frame.shape[:2]
    thick = max(3, int(W / 300))
    P = lambda p: (int(p[0] * W), int(p[1] * H))  # noqa: E731
    for s in shapes:
        col = _hex_to_bgr(s.get("color"))
        kind = s.get("type")
        if kind == "arrow":
            a, b = P(s["from"]), P(s["to"])
            length = max(1.0, np.hypot(b[0] - a[0], b[1] - a[1]))
            cv2.arrowedLine(frame, a, b, col, thick, cv2.LINE_AA, tipLength=min(0.4, 30 * thick / 3 / length))
        elif kind == "line":
            cv2.line(frame, P(s["from"]), P(s["to"]), col, thick, cv2.LINE_AA)
        elif kind == "circle":
            c = P(s["center"])
            rx = max(4, int(s["radius"] * W))
            cv2.ellipse(frame, c, (rx, max(3, int(rx * s.get("ratio", 1.0)))), 0, 0, 360, col, thick, cv2.LINE_AA)
        elif kind == "free":
            pts = np.array([P(p) for p in s.get("points", [])], np.int32)
            if len(pts) > 1:
                cv2.polylines(frame, [pts], False, col, thick, cv2.LINE_AA)
        elif kind == "text" and s.get("text"):
            scale = max(0.8, W / 1400)
            x, y = P(s["at"])
            (tw, th), _ = cv2.getTextSize(s["text"], cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
            cv2.rectangle(frame, (x - 10, y - th - 10), (x + tw + 10, y + 10), (20, 20, 20), -1)
            cv2.putText(frame, s["text"], (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, col, 2, cv2.LINE_AA)


# --- renderen ----------------------------------------------------------------------------

def _letterbox(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    s = min(OUT_W / w, OUT_H / h)
    nw, nh = int(round(w * s)), int(round(h * s))
    small = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    if (nw, nh) == (OUT_W, OUT_H):
        return small
    out = np.zeros((OUT_H, OUT_W, 3), np.uint8)
    y0, x0 = (OUT_H - nh) // 2, (OUT_W - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = small
    return out


def _frames(path: Path, start: float, end: float):
    """Bronbeelden met echte tijden in [start, end]."""
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, start - 1.0) * 1000)
    try:
        while True:
            if not cap.grab():
                return
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if t > end:
                return
            if t < start - 0.04:
                continue
            ok, frame = cap.retrieve()
            if ok:
                yield t, frame
    finally:
        cap.release()


def _audio_filter(length: float, freezes: list[tuple[float, float]], src_audio: bool) -> tuple[list[str], str]:
    """ffmpeg-invoer en filter: bronaudio met stilte op de bevroren momenten."""
    fmt = "aformat=sample_rates=48000:channel_layouts=stereo"
    if not src_audio:
        total = length + sum(d for _, d in freezes)
        return [], f"anullsrc=r=48000:cl=stereo,atrim=duration={total:.3f}[aout]"
    parts, labels, prev = [], [], 0.0
    for i, (at, dur) in enumerate(freezes):
        parts.append(f"[1:a]atrim=start={prev:.3f}:end={at:.3f},asetpts=PTS-STARTPTS,{fmt}[p{i}]")
        parts.append(f"anullsrc=r=48000:cl=stereo,atrim=duration={dur:.3f},{fmt}[s{i}]")
        labels += [f"[p{i}]", f"[s{i}]"]
        prev = at
    parts.append(f"[1:a]atrim=start={prev:.3f},asetpts=PTS-STARTPTS,{fmt},apad=whole_dur={length - prev:.3f}[pe]")
    labels.append("[pe]")
    parts.append(f"{''.join(labels)}concat=n={len(labels)}:v=0:a=1[aout]")
    return [], ";".join(parts)


def render_moment(store: Store, moment: dict, dst: Path) -> Path:
    clip = store.one("SELECT * FROM clips WHERE id = ?", (moment["clip_id"],))
    src = Path(clip["path"])
    start, end = float(moment["start"]), float(moment["end"])
    length = max(0.5, end - start)

    spot, label = None, ""
    pid = moment.get("spotlight_player_id")
    if pid:
        p = store.one("SELECT * FROM players WHERE id = ?", (pid,))
        if p:
            spot = load_spotlight(store, clip["id"], pid, start, end)
            label = f"{p['number']} {p['name']}" if p["number"] else p["name"]
    drawings = sorted(moment.get("drawings") or [], key=lambda d: d["t"])
    freezes = [(max(0.0, min(length, d["t"] - start)), float(d.get("duration", 4))) for d in drawings
               if start <= d["t"] <= end]
    freezes.sort()

    src_audio = has_audio(src)
    _, afilter = _audio_filter(length, freezes, src_audio)
    audio_in = ["-ss", f"{start:.3f}", "-t", f"{length:.3f}", "-i", str(src)] if src_audio else []
    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{OUT_W}x{OUT_H}", "-r", str(OUT_FPS), "-i", "-",
           *audio_in, "-filter_complex", afilter, "-map", "0:v", "-map", "[aout]",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(dst)]
    with tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=err)
        try:
            _write_frames(proc, src, start, end, length, spot, label, drawings)
        finally:
            proc.stdin.close()
            code = proc.wait()
        if code != 0:
            err.seek(0)
            raise RuntimeError(err.read().decode(errors="replace")[-800:])
    return dst


def _write_frames(proc, src, start, end, length, spot, label, drawings) -> None:
    pending = [d for d in sorted(drawings, key=lambda d: d["t"]) if start <= d["t"] <= end]
    frames = _frames(src, start, end)
    nxt = next(frames, None)
    if nxt is None:
        raise RuntimeError("Geen beelden gevonden in dit stuk van de video")
    cur = nxt
    nxt = next(frames, None)
    n_out = int(round(length * OUT_FPS))
    for k in range(n_out):
        target = start + k / OUT_FPS
        while nxt is not None and nxt[0] <= target + 1e-3:
            cur, nxt = nxt, next(frames, None)
        t, frame = cur
        img = frame.copy()
        if spot is not None:
            box = spot.box_at(target)
            if box is not None:
                draw_spotlight(img, box, label)
        out = _letterbox(img)
        proc.stdin.write(out.tobytes())
        # bevroren beeld met tekening
        while pending and pending[0]["t"] <= target + 0.5 / OUT_FPS:
            d = pending.pop(0)
            frozen = img.copy()
            draw_shapes(frozen, d.get("shapes", []))
            data = _letterbox(frozen).tobytes()
            for _ in range(int(round(float(d.get("duration", 4)) * OUT_FPS))):
                proc.stdin.write(data)


def safe_name(name: str) -> str:
    import re

    return re.sub(r"[^\w\-]+", "_", name).strip("_")[:60] or "clip"


def export_moments(store: Store, moment_ids: list[int], name: str, mode: str = "reel") -> Path:
    """mode 'reel': alles achter elkaar in één mp4. mode 'zip': elke clip als los bestand in een zip."""
    config.EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    moments = [m for m in (store.moment(i) for i in moment_ids) if m]
    if not moments:
        raise ValueError("Geen clips gekozen")
    if len(moments) == 1 and mode != "zip":
        return render_moment(store, moments[0], config.EXPORTS_DIR / f"{safe_name(name)}-{stamp}.mp4")
    with tempfile.TemporaryDirectory() as tmp:
        parts = []
        for i, m in enumerate(moments):
            parts.append(render_moment(store, m, Path(tmp) / f"{i + 1:02d}-{safe_name(m['label'] or 'clip')}.mp4"))
        if mode == "zip":
            dst = config.EXPORTS_DIR / f"{safe_name(name)}-{stamp}.zip"
            with zipfile.ZipFile(dst, "w", zipfile.ZIP_STORED) as z:
                for p in parts:
                    z.write(p, p.name)
            return dst
        dst = config.EXPORTS_DIR / f"{safe_name(name)}-{stamp}.mp4"
        lst = Path(tmp) / "list.txt"
        lst.write_text("".join(f"file '{p}'\n" for p in parts))
        subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
                        "-c", "copy", "-movflags", "+faststart", str(dst)], check=True)
        return dst
