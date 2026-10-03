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


@dataclass
class FrameDetections:
    boxes: np.ndarray  # (n, 4) xyxy personen
    scores: np.ndarray  # (n,)
    ball: tuple[float, float, float] | None  # (x, y, conf) middelpunt bal


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
        people = cls == PERSON
        ball = None
        balls = np.where((cls == BALL) & (conf >= config.BALL_CONF))[0]
        if len(balls):
            k = balls[np.argmax(conf[balls])]
            x1, y1, x2, y2 = xyxy[k]
            ball = ((x1 + x2) / 2, (y1 + y2) / 2, float(conf[k]))
        return FrameDetections(xyxy[people], conf[people], ball)

    def detect_batch(self, frames: list[np.ndarray]) -> list[FrameDetections]:
        res = self.model.predict(frames, imgsz=self.imgsz, conf=min(config.BALL_CONF, 0.1),
                                 classes=[PERSON, BALL], device=self.device, verbose=False)
        return [self._convert(r) for r in res]

    def __call__(self, frame: np.ndarray) -> FrameDetections:
        return self.detect_batch([frame])[0]


@lru_cache(maxsize=2)
def get_detector(mode: str = "nauwkeurig") -> Detector:
    model, imgsz = MODES.get(mode, MODES["nauwkeurig"])
    return Detector(model, imgsz)
