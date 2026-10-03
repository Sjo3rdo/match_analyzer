"""Een lange video knippen in delen (bijv. 1e en 2e helft, warming-up eruit) vóór de analyse.

We kopiëren de videostroom zonder opnieuw te coderen ("stream copy"): razendsnel, ook bij
bestanden van vele GB, en zonder kwaliteitsverlies. Nadeel: een knip kan alleen op een
sleutelbeeld van de video beginnen, dus het begin kan tot ~1 seconde eerder liggen dan gekozen.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from .pipeline import ffmpeg_exe


def cut_copy(src: Path, start: float, end: float, dst: Path) -> Path:
    if end <= start:
        raise ValueError("Het einde moet na het begin liggen")
    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error", "-ss", f"{max(0.0, start):.3f}", "-i", str(src),
           "-t", f"{end - start:.3f}", "-map", "0:v:0", "-map", "0:a:0?", "-c", "copy",
           "-avoid_negative_ts", "make_zero", "-movflags", "+faststart", str(dst)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not dst.exists() or dst.stat().st_size == 0:
        raise RuntimeError(f"Knippen mislukt: {r.stderr[-500:]}")
    return dst
