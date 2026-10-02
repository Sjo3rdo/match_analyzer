"""Highlights exporteren met ffmpeg: losse fragmenten of één samengevoegde reel."""
from __future__ import annotations

import re
import subprocess
import tempfile
import time
from pathlib import Path

from . import config
from .pipeline import ffmpeg_exe


def _safe(name: str) -> str:
    return re.sub(r"[^\w\-]+", "_", name).strip("_")[:60] or "clip"


# Vast formaat, zodat fragmenten uit verschillende telefoonvideo's samengevoegd kunnen worden
_FORMAT = ("scale=1920:1080:force_original_aspect_ratio=decrease,"
           "pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30")


def cut(src: Path, start: float, end: float, dst: Path, audio: bool = True) -> Path:
    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error", "-ss", f"{max(0, start):.2f}", "-i", str(src),
           "-t", f"{max(0.5, end - start):.2f}", "-vf", _FORMAT,
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
           *(["-c:a", "aac"] if audio else ["-an"]), "-movflags", "+faststart", str(dst)]
    subprocess.run(cmd, check=True)
    return dst


def export(items: list[dict], name: str) -> Path:
    """items: [{"path", "start", "end"}]. Eén item -> los fragment, meer -> samengevoegd."""
    config.EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    dst = config.EXPORTS_DIR / f"{_safe(name)}-{stamp}.mp4"
    if len(items) == 1:
        it = items[0]
        return cut(Path(it["path"]), it["start"], it["end"], dst)
    with tempfile.TemporaryDirectory() as tmp:
        parts = []
        for i, it in enumerate(items):
            parts.append(cut(Path(it["path"]), it["start"], it["end"], Path(tmp) / f"{i:03d}.mp4",
                             audio=False))
        lst = Path(tmp) / "list.txt"
        lst.write_text("".join(f"file '{p}'\n" for p in parts))
        subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                        "-i", str(lst), "-c", "copy", "-movflags", "+faststart", str(dst)], check=True)
    return dst
