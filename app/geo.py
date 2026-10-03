"""GPS uit de video en het voetbalveld uit OpenStreetMap: waar stond de camera op het veld?

Stap voor stap:
1. Een iPhone (en de meeste Android-toestellen) slaat bij een video de GPS-positie op.
2. Voetbalvelden zijn in OpenStreetMap ingetekend als rechthoek ("leisure=pitch").
3. We zoeken het veld bij die positie, leggen er een lokaal assenstelsel in meters op en
   rekenen de camerapositie om naar veldcoördinaten (0..105 x 0..68).

Afspraak: de camera staat aan de kant van de 'onderste' zijlijn (y = 68), en x loopt van links
naar rechts zoals jij het veld zag. Zo komt de veldtekening overeen met je beeld.

Let op privacy: alleen de coördinaat wordt naar OpenStreetMap gestuurd, en alleen als de
gebruiker daarom vraagt.
"""
from __future__ import annotations

import json
import math
import re
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from . import pitch

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
_ISO6709 = re.compile(r"([+-]\d+(?:\.\d+)?)([+-]\d+(?:\.\d+)?)([+-]\d+(?:\.\d+)?)?")


def parse_iso6709(text: str) -> tuple[float, float] | None:
    m = _ISO6709.search(text or "")
    if not m:
        return None
    lat, lon = float(m.group(1)), float(m.group(2))
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    return lat, lon


def read_video_metadata(path: Path, ffmpeg: str) -> dict:
    """GPS (lat, lon, nauwkeurigheid in m) en toesteltype uit de containermetadata."""
    r = subprocess.run([ffmpeg, "-hide_banner", "-i", str(path)], capture_output=True, text=True)
    meta: dict = {}
    for line in r.stderr.splitlines():
        key, _, val = line.strip().partition(":")
        key, val = key.strip().lower(), val.strip()
        if key.endswith("location.iso6709") or key in ("location", "location-eng"):
            pos = parse_iso6709(val)
            if pos and "gps_lat" not in meta:
                meta["gps_lat"], meta["gps_lon"] = pos
        elif key.endswith("location.accuracy.horizontal"):
            try:
                meta["gps_acc"] = float(val)
            except ValueError:
                pass
        elif key.endswith("quicktime.model") or key == "model":
            meta.setdefault("device", val)
    return meta


def to_local(lat: float, lon: float, lat0: float, lon0: float) -> np.ndarray:
    """Graden -> meters (oost, noord) rond (lat0, lon0). Op veldschaal nauwkeurig genoeg."""
    r = 6371000.0
    return np.array([math.radians(lon - lon0) * r * math.cos(math.radians(lat0)),
                     math.radians(lat - lat0) * r])


def query_pitches(lat: float, lon: float, radius: int = 300, timeout: int = 25) -> list[list[tuple[float, float]]]:
    """Omtrek (lat, lon) van voetbalvelden in de buurt, uit OpenStreetMap (Overpass)."""
    q = (f"[out:json][timeout:{timeout}];"
         f"(way(around:{radius},{lat},{lon})[\"leisure\"=\"pitch\"];);out geom;")
    req = urllib.request.Request(OVERPASS_URL, data=urllib.parse.urlencode({"data": q}).encode(),
                                 headers={"User-Agent": "match-analyzer/1.0"})
    with urllib.request.urlopen(req, timeout=timeout + 5) as resp:
        data = json.load(resp)
    out = []
    for el in data.get("elements", []):
        sport = (el.get("tags") or {}).get("sport", "soccer")
        geom = el.get("geometry") or []
        if "soccer" in sport and len(geom) >= 4:
            out.append([(g["lat"], g["lon"]) for g in geom])
    return out


def camera_on_pitch(lat: float, lon: float, polygons: list[list[tuple[float, float]]]) -> dict:
    """Kies het veld bij de camera en geef de camerapositie in veldcoördinaten."""
    best = None
    for poly in polygons:
        pts = np.array([to_local(a, b, lat, lon) for a, b in poly], np.float32)  # camera = (0, 0)
        (cx, cy), (w, h), ang = cv2.minAreaRect(pts)
        length, width = max(w, h), min(w, h)
        if length < 40 or width < 25:  # trainingsveldjes, pannakooien
            continue
        dist = abs(cv2.pointPolygonTest(pts.reshape(-1, 1, 2), (0.0, 0.0), True))
        inside = cv2.pointPolygonTest(pts.reshape(-1, 1, 2), (0.0, 0.0), False) >= 0
        score = 0.0 if inside else dist
        if best is None or score < best[0]:
            best = (score, np.array([cx, cy]), length, width, ang, w >= h)
    if best is None:
        raise ValueError("Geen voetbalveld gevonden bij de GPS-positie van deze video")
    _, center, length, width, ang, w_is_long = best
    a = math.radians(ang)
    e1, e2 = np.array([math.cos(a), math.sin(a)]), np.array([-math.sin(a), math.cos(a)])
    u, v = (e1, e2) if w_is_long else (e2, e1)  # u: lengterichting, v: breedterichting
    rel = -center  # camera t.o.v. het middelpunt
    if rel @ v < 0:  # v wijst naar de kant van de camera (de 'onderste' zijlijn)
        v = -v
    right = np.array([-v[1], v[0]])  # rechts, gezien vanaf de camera die naar het veld kijkt (-v)
    if u @ right < 0:
        u = -u
    x = pitch.LENGTH / 2 + (rel @ u) * pitch.LENGTH / length
    y = pitch.WIDTH / 2 + (rel @ v) * pitch.WIDTH / width
    return {"x": round(float(x), 1), "y": round(float(y), 1), "pitch_length": round(float(length), 1),
            "pitch_width": round(float(width), 1), "distance_to_pitch": round(float(best[0]), 1)}
