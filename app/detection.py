"""Spelers en bal detecteren met YOLO (Ultralytics), op de GPU van de Mac (MPS) indien mogelijk."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from . import config

PERSON, BALL = 0, 32  # COCO-klassen


def best_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _no_telemetry() -> None:
    """Ultralytics stuurt standaard anonieme gebruiksgegevens naar Google Analytics. Alles blijft hier
    op je eigen computer, dus dat zetten we uit (alleen voor deze app, je instellingen blijven gelijk)."""
    try:
        from ultralytics.utils import events

        events.events.enabled = False
    except Exception:  # noqa: BLE001  (andere versie van Ultralytics)
        pass


BALL_MIN_CONF = 0.05  # zo laag zoeken naar de bal; de balvolger beslist (zie ball.py)


def _candidates(xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray) -> list:
    from .ball import Candidate

    k = np.flatnonzero(cls == BALL)
    return [Candidate(float((xyxy[i, 0] + xyxy[i, 2]) / 2), float((xyxy[i, 1] + xyxy[i, 3]) / 2), float(conf[i]),
                      float(max(xyxy[i, 2] - xyxy[i, 0], xyxy[i, 3] - xyxy[i, 1]))) for i in k]


@dataclass
class FrameDetections:
    boxes: np.ndarray  # (n, 4) xyxy personen
    scores: np.ndarray  # (n,)
    ball: tuple[float, float, float] | None  # (x, y, conf) middelpunt bal
    balls: list = None  # alle balkandidaten (ball.Candidate), ook twijfelachtige


MODES = {
    # naam: (model, beeldgrootte voor de detectie)
    "nauwkeurig": (config.YOLO_MODEL, config.DETECT_IMGSZ),
    "snel": (config.FAST_MODEL, config.FAST_IMGSZ),
}


class Detector:
    batch_size = 4  # een paar beelden tegelijk: de grafische chip werkt dan efficiënter

    def __init__(self, model_name: str = config.YOLO_MODEL, imgsz: int = config.DETECT_IMGSZ):
        from pathlib import Path

        from ultralytics import YOLO

        _no_telemetry()
        path = Path(model_name)
        if not path.is_absolute() and not path.exists():
            config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
            path = config.MODELS_DIR / path.name
        try:
            self.model = YOLO(str(path))
        except Exception as e:  # noqa: BLE001  (meestal: geen internet bij de eerste keer)
            raise RuntimeError(f"Het detectiemodel ({path.name}) kon niet worden geladen of gedownload. "
                               f"Controleer je internetverbinding en probeer het opnieuw. ({e})") from e
        self.imgsz = imgsz
        self.device = best_device()

    def _convert(self, res) -> FrameDetections:
        b = res.boxes
        xyxy = b.xyxy.cpu().numpy() if len(b) else np.zeros((0, 4))
        conf = b.conf.cpu().numpy() if len(b) else np.zeros(0)
        cls = b.cls.cpu().numpy().astype(int) if len(b) else np.zeros(0, int)
        people = (cls == PERSON) & (conf >= config.PERSON_CONF)
        ball = None
        balls = np.where((cls == BALL) & (conf >= config.BALL_CONF))[0]
        if len(balls):
            k = balls[np.argmax(conf[balls])]
            x1, y1, x2, y2 = xyxy[k]
            ball = ((x1 + x2) / 2, (y1 + y2) / 2, float(conf[k]))
        return FrameDetections(xyxy[people], conf[people], ball, _candidates(xyxy, conf, cls))

    def detect_batch(self, frames: list[np.ndarray]) -> list[FrameDetections]:
        res = self.model.predict(frames, imgsz=self.imgsz, conf=BALL_MIN_CONF,
                                 classes=[PERSON, BALL], device=self.device, verbose=False)
        return [self._convert(r) for r in res]

    def detect_balls(self, frame: np.ndarray, boxes: list[tuple[int, int, int, int]], imgsz: int = 640) -> list:
        """Balkandidaten in uitsneden van het beeld op volle resolutie (inzoomen), in beeldpixels."""
        if not boxes:
            return []
        crops = [frame[y0:y1, x0:x1] for x0, y0, x1, y1 in boxes]
        res = self.model.predict(crops, imgsz=imgsz, conf=BALL_MIN_CONF, classes=[BALL], device=self.device, verbose=False)
        out = []
        for (x0, y0, _, _), r in zip(boxes, res):
            b = r.boxes
            if not len(b):
                continue
            xyxy = b.xyxy.cpu().numpy() + np.array([x0, y0, x0, y0])
            out += _candidates(xyxy, b.conf.cpu().numpy(), b.cls.cpu().numpy().astype(int))
        return out

    def __call__(self, frame: np.ndarray) -> FrameDetections:
        return self.detect_batch([frame])[0]


@lru_cache(maxsize=2)
def get_detector(mode: str = "nauwkeurig") -> Detector:
    model, imgsz = MODES.get(mode, MODES["nauwkeurig"])
    return Detector(model, imgsz)
