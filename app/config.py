"""Centrale instellingen en paden."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("MATCH_ANALYZER_DATA", ROOT / "data"))
DB_PATH = DATA_DIR / "match_analyzer.db"
CLIPS_DIR = DATA_DIR / "clips"
EXPORTS_DIR = DATA_DIR / "exports"
STATIC_DIR = ROOT / "static"

# Detectie
# Wordt bij eerste gebruik automatisch gedownload. yolo11m = goede balans snelheid/nauwkeurigheid
# op een MacBook; kies yolo11l/x voor meer precisie of yolo11s/n voor snelheid.
YOLO_MODEL = os.environ.get("MATCH_ANALYZER_MODEL", "yolo11m.pt")
MODELS_DIR = DATA_DIR / "models"
BALL_CONF = 0.10
PERSON_CONF = 0.10  # (het model zoekt lager, voor de bal)
DETECT_IMGSZ = 1280  # groot, zodat verre spelers en de bal nog gevonden worden
TARGET_FPS = 10.0  # aantal geanalyseerde frames per seconde video
# Snelle analyse: kleiner model op een kleiner beeld. Ongeveer 2x zo snel, maar mist vaker
# spelers ver weg en de bal.
FAST_MODEL = os.environ.get("MATCH_ANALYZER_FAST_MODEL", "yolo11s.pt")
FAST_IMGSZ = 960

# Analyse
MAX_SPEED_MS = 11.0  # alles sneller is een meetfout (~40 km/u)
SPRINT_SPEED_MS = 25 / 3.6
SPRINT_MIN_S = 1.0
POSSESSION_RADIUS_M = 2.0
POSSESSION_MIN_FRAMES = 2
PASS_MAX_GAP_S = 5.0
HEATMAP_BINS = (21, 14)  # cellen van 5 x ~4,9 m


def ensure_dirs() -> None:
    for d in (DATA_DIR, CLIPS_DIR, EXPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
