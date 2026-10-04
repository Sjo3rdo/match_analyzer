"""Veldmodel (11 tegen 11) met herkenbare punten voor kalibratie.

Coördinaten in meters: x loopt van de linker doellijn (0) naar de rechter (lengte),
y van de bovenste zijlijn (0) naar de onderste (breedte).

Lengte en breedte verschillen per veld (amateurvelden zijn vaak kleiner dan 105 x 68). De
vlakken rond het doel (strafschopgebied, doelgebied, stip) en de middencirkel hebben wel altijd
dezelfde maten; die schuiven gewoon mee met de doellijnen en het midden.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache

DEFAULT_LENGTH = 105.0
DEFAULT_WIDTH = 68.0

_PA_HALF = 20.16  # halve breedte strafschopgebied
_GA_HALF = 9.16  # halve breedte doelgebied
_GOAL_HALF = 3.66
_CIRCLE_R = 9.15
GOAL_HEIGHT = 2.44  # onderkant van de lat
_ARC_DY = math.sqrt(_CIRCLE_R**2 - 5.5**2)  # snijpunt penaltyboog met 16 m-lijn


@dataclass(frozen=True)
class Geometry:
    length: float = DEFAULT_LENGTH
    width: float = DEFAULT_WIDTH
    landmarks: dict = field(init=False, repr=False, compare=False, hash=False)
    lines: dict = field(init=False, repr=False, compare=False, hash=False)
    elevated: dict = field(init=False, repr=False, compare=False, hash=False)
    elevated_lines: dict = field(init=False, repr=False, compare=False, hash=False)

    def __post_init__(self):
        object.__setattr__(self, "landmarks", _landmarks(self.length, self.width))
        object.__setattr__(self, "lines", _lines(self.length, self.width))
        elevated, elevated_lines = _goal_frames(self.length, self.width)
        object.__setattr__(self, "elevated", elevated)
        object.__setattr__(self, "elevated_lines", elevated_lines)

    @property
    def half_w(self) -> float:
        return self.width / 2

    def on_pitch(self, x: float, y: float, margin: float = 3.0) -> bool:
        return -margin <= x <= self.length + margin and -margin <= y <= self.width + margin


@lru_cache(maxsize=32)
def geometry(length: float | None = None, width: float | None = None) -> Geometry:
    return Geometry(round(float(length or DEFAULT_LENGTH), 2), round(float(width or DEFAULT_WIDTH), 2))


def of_match(match: dict | None) -> Geometry:
    """Veldmaten van een wedstrijd (of de standaard 105 x 68)."""
    match = match or {}
    return geometry(match.get("pitch_length"), match.get("pitch_width"))


def _landmarks(L: float, W: float) -> dict[str, tuple[float, float]]:
    c = W / 2
    pts: dict[str, tuple[float, float]] = {
        "Hoekvlag linksboven": (0, 0),
        "Hoekvlag rechtsboven": (L, 0),
        "Hoekvlag linksonder": (0, W),
        "Hoekvlag rechtsonder": (L, W),
        "Middenlijn boven": (L / 2, 0),
        "Middenlijn onder": (L / 2, W),
        "Middenstip": (L / 2, c),
        "Middencirkel boven": (L / 2, c - _CIRCLE_R),
        "Middencirkel onder": (L / 2, c + _CIRCLE_R),
        "Middencirkel links": (L / 2 - _CIRCLE_R, c),
        "Middencirkel rechts": (L / 2 + _CIRCLE_R, c),
    }
    for side, x0, sign in (("links", 0.0, 1), ("rechts", L, -1)):
        pts[f"Strafschopgebied {side} hoek boven"] = (x0 + sign * 16.5, c - _PA_HALF)
        pts[f"Strafschopgebied {side} hoek onder"] = (x0 + sign * 16.5, c + _PA_HALF)
        pts[f"Strafschopgebied {side} doellijn boven"] = (x0, c - _PA_HALF)
        pts[f"Strafschopgebied {side} doellijn onder"] = (x0, c + _PA_HALF)
        pts[f"Doelgebied {side} hoek boven"] = (x0 + sign * 5.5, c - _GA_HALF)
        pts[f"Doelgebied {side} hoek onder"] = (x0 + sign * 5.5, c + _GA_HALF)
        pts[f"Doelgebied {side} doellijn boven"] = (x0, c - _GA_HALF)
        pts[f"Doelgebied {side} doellijn onder"] = (x0, c + _GA_HALF)
        pts[f"Doelpaal {side} boven"] = (x0, c - _GOAL_HALF)
        pts[f"Doelpaal {side} onder"] = (x0, c + _GOAL_HALF)
        pts[f"Strafschopstip {side}"] = (x0 + sign * 11, c)
        pts[f"Penaltyboog {side} boven"] = (x0 + sign * 16.5, c - _ARC_DY)
        pts[f"Penaltyboog {side} onder"] = (x0 + sign * 16.5, c + _ARC_DY)
    return {k: (round(v[0], 3), round(v[1], 3)) for k, v in pts.items()}


def _lines(L: float, W: float) -> dict[str, tuple[tuple[float, float], tuple[float, float]]]:
    """Rechte veldlijnen (begin- en eindpunt in meters). Handig bij beelden vanaf de zijlijn,
    waar je vaak een lijn wel ziet maar geen hoekpunt."""
    c = W / 2
    lines = {
        "Zijlijn boven": ((0, 0), (L, 0)),
        "Zijlijn onder": ((0, W), (L, W)),
        "Middenlijn": ((L / 2, 0), (L / 2, W)),
        "Doellijn links": ((0, 0), (0, W)),
        "Doellijn rechts": ((L, 0), (L, W)),
    }
    for side, x0, sign in (("links", 0.0, 1), ("rechts", L, -1)):
        xp, xg = x0 + sign * 16.5, x0 + sign * 5.5
        lines[f"16-meterlijn {side} (voorkant)"] = ((xp, c - _PA_HALF), (xp, c + _PA_HALF))
        lines[f"16-meterlijn {side} zijkant boven"] = ((x0, c - _PA_HALF), (xp, c - _PA_HALF))
        lines[f"16-meterlijn {side} zijkant onder"] = ((x0, c + _PA_HALF), (xp, c + _PA_HALF))
        lines[f"5-meterlijn {side} (voorkant)"] = ((xg, c - _GA_HALF), (xg, c + _GA_HALF))
        lines[f"5-meterlijn {side} zijkant boven"] = ((x0, c - _GA_HALF), (xg, c - _GA_HALF))
        lines[f"5-meterlijn {side} zijkant onder"] = ((x0, c + _GA_HALF), (xg, c + _GA_HALF))
    return {k: (tuple(map(float, a)), tuple(map(float, b))) for k, (a, b) in lines.items()}


def _goal_frames(L: float, W: float):
    """Punten en lijnen in de lucht: de bovenkant van de doelpalen en de lat (x, y, hoogte).
    De voet van een paal is het gewone punt "Doelpaal ... boven/onder" (boven/onder = in de
    veldtekening)."""
    c = W / 2
    pts, lines = {}, {}
    for side, x0 in (("links", 0.0), ("rechts", L)):
        a = (x0, round(c - _GOAL_HALF, 3), GOAL_HEIGHT)
        b = (x0, round(c + _GOAL_HALF, 3), GOAL_HEIGHT)
        pts[f"Doelpaal {side} boven, bovenkant"] = a
        pts[f"Doelpaal {side} onder, bovenkant"] = b
        lines[f"Lat {side}"] = (a, b)
    return pts, lines


def remap_points(points: list[dict], new: Geometry) -> list[dict]:
    """Kalibratiepunten (op naam) naar de veldcoördinaten van andere veldmaten omzetten."""
    out = []
    for p in points:
        q = dict(p)
        if q.get("pitch") is not None and q.get("name") in new.landmarks:
            q["pitch"] = list(new.landmarks[q["name"]])
        elif q.get("line") and q.get("name") in new.lines:
            a, b = new.lines[q["name"]]
            q["line"] = [list(a), list(b)]
        elif q.get("pitch3") is not None and q.get("name") in new.elevated:
            q["pitch3"] = list(new.elevated[q["name"]])
        elif q.get("line3") and q.get("name") in new.elevated_lines:
            a, b = new.elevated_lines[q["name"]]
            q["line3"] = [list(a), list(b)]
        out.append(q)
    return out


# Standaardveld, voor code die (nog) geen wedstrijd kent
DEFAULT = geometry()
LENGTH = DEFAULT.length
WIDTH = DEFAULT.width
HALF_W = WIDTH / 2
LANDMARKS = DEFAULT.landmarks
LINES = DEFAULT.lines


def on_pitch(x: float, y: float, margin: float = 3.0) -> bool:
    return DEFAULT.on_pitch(x, y, margin)
