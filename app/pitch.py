"""Veldmodel (11 tegen 11, 105 x 68 m) met herkenbare punten voor kalibratie.

Coördinaten in meters: x loopt van de linker doellijn (0) naar de rechter (105),
y van de bovenste zijlijn (0) naar de onderste (68).
"""
from __future__ import annotations

import math

LENGTH = 105.0
WIDTH = 68.0
HALF_W = WIDTH / 2

_PA_HALF = 20.16  # halve breedte strafschopgebied
_GA_HALF = 9.16  # halve breedte doelgebied
_GOAL_HALF = 3.66
_CIRCLE_R = 9.15
_ARC_DY = math.sqrt(_CIRCLE_R**2 - 5.5**2)  # snijpunt penaltyboog met 16 m-lijn


def _landmarks() -> dict[str, tuple[float, float]]:
    pts: dict[str, tuple[float, float]] = {
        "Hoekvlag linksboven": (0, 0),
        "Hoekvlag rechtsboven": (LENGTH, 0),
        "Hoekvlag linksonder": (0, WIDTH),
        "Hoekvlag rechtsonder": (LENGTH, WIDTH),
        "Middenlijn boven": (LENGTH / 2, 0),
        "Middenlijn onder": (LENGTH / 2, WIDTH),
        "Middenstip": (LENGTH / 2, HALF_W),
        "Middencirkel boven": (LENGTH / 2, HALF_W - _CIRCLE_R),
        "Middencirkel onder": (LENGTH / 2, HALF_W + _CIRCLE_R),
        "Middencirkel links": (LENGTH / 2 - _CIRCLE_R, HALF_W),
        "Middencirkel rechts": (LENGTH / 2 + _CIRCLE_R, HALF_W),
    }
    for side, x0, sign in (("links", 0.0, 1), ("rechts", LENGTH, -1)):
        pts[f"Strafschopgebied {side} hoek boven"] = (x0 + sign * 16.5, HALF_W - _PA_HALF)
        pts[f"Strafschopgebied {side} hoek onder"] = (x0 + sign * 16.5, HALF_W + _PA_HALF)
        pts[f"Strafschopgebied {side} doellijn boven"] = (x0, HALF_W - _PA_HALF)
        pts[f"Strafschopgebied {side} doellijn onder"] = (x0, HALF_W + _PA_HALF)
        pts[f"Doelgebied {side} hoek boven"] = (x0 + sign * 5.5, HALF_W - _GA_HALF)
        pts[f"Doelgebied {side} hoek onder"] = (x0 + sign * 5.5, HALF_W + _GA_HALF)
        pts[f"Doelgebied {side} doellijn boven"] = (x0, HALF_W - _GA_HALF)
        pts[f"Doelgebied {side} doellijn onder"] = (x0, HALF_W + _GA_HALF)
        pts[f"Doelpaal {side} boven"] = (x0, HALF_W - _GOAL_HALF)
        pts[f"Doelpaal {side} onder"] = (x0, HALF_W + _GOAL_HALF)
        pts[f"Strafschopstip {side}"] = (x0 + sign * 11, HALF_W)
        pts[f"Penaltyboog {side} boven"] = (x0 + sign * 16.5, HALF_W - _ARC_DY)
        pts[f"Penaltyboog {side} onder"] = (x0 + sign * 16.5, HALF_W + _ARC_DY)
    return {k: (round(v[0], 3), round(v[1], 3)) for k, v in pts.items()}


LANDMARKS = _landmarks()


def on_pitch(x: float, y: float, margin: float = 3.0) -> bool:
    return -margin <= x <= LENGTH + margin and -margin <= y <= WIDTH + margin
