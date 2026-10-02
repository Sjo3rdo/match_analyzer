"""Optioneel: rugnummers lezen met EasyOCR (pip install -r requirements-ocr.txt).

Telefoonbeelden vanaf de zijlijn zijn vaak te klein of onscherp; daarom stemmen we
over meerdere uitsneden per track en tonen we het resultaat alleen als *suggestie*.
"""
from __future__ import annotations

from collections import Counter
from functools import lru_cache

import cv2
import numpy as np


@lru_cache(maxsize=1)
def _reader():
    try:
        import easyocr  # type: ignore
    except ImportError:
        return None
    return easyocr.Reader(["en"], gpu=False, verbose=False)


def available() -> bool:
    return _reader() is not None


def read_number(crops: list[np.ndarray]) -> tuple[str | None, float]:
    """Meest gestemde nummer en het aandeel stemmen (0..1)."""
    reader = _reader()
    if reader is None or not crops:
        return None, 0.0
    votes: Counter[str] = Counter()
    for crop in crops:
        h = crop.shape[0]
        torso = crop[int(0.1 * h):int(0.6 * h)]
        if torso.size == 0:
            continue
        torso = cv2.resize(torso, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        for _, text, conf in reader.readtext(torso, allowlist="0123456789"):
            if text and len(text) <= 2 and conf > 0.4 and text != "0":
                votes[text.lstrip("0") or text] += conf
    if not votes:
        return None, 0.0
    number, score = votes.most_common(1)[0]
    return number, float(score / sum(votes.values()))
