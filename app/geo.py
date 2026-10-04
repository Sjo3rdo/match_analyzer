"""GPS uit de video en het voetbalveld uit OpenStreetMap: waar stond de camera op het veld?

Stap voor stap:
1. Een iPhone (en de meeste Android-toestellen) slaat bij een video de GPS-positie op.
2. Voetbalvelden zijn in OpenStreetMap ingetekend als rechthoek ("leisure=pitch").
3. We zoeken het veld bij die positie, leggen er een lokaal assenstelsel in meters op en
   rekenen de camerapositie om naar veldcoördinaten (0..lengte x 0..breedte).

Afspraak: de camera staat aan de kant van de 'onderste' zijlijn (y = 68), en x loopt van links
naar rechts zoals jij het veld zag. Zo komt de veldtekening overeen met je beeld.

Let op privacy: alleen de coördinaat wordt naar OpenStreetMap gestuurd, en alleen als de
gebruiker daarom vraagt.
"""
from __future__ import annotations

import json
import logging
import math
import re
import ssl
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from . import pitch

# De gratis OpenStreetMap-zoekservers (Overpass). Is de eerste druk of weigert hij, dan de volgende.
OVERPASS_URLS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
)
USER_AGENT = "match-analyzer/0.13 (lokale voetbalanalyse; https://github.com/sjo3rdo/match_analyzer)"


class OsmError(RuntimeError):
    """OpenStreetMap niet bereikbaar; de tekst zegt in gewone woorden waarom."""


def _ssl_context() -> ssl.SSLContext:
    """Beveiligde verbinding met de certificaten van certifi. Python op de Mac heeft soms geen eigen
    certificaten (dan faalt elke https-verbinding met CERTIFICATE_VERIFY_FAILED)."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:  # noqa: BLE001
        return ssl.create_default_context()


def _overpass(query: str, timeout: int) -> dict:
    body = urllib.parse.urlencode({"data": query}).encode()
    ctx = _ssl_context()
    problems = []
    for url in OVERPASS_URLS:
        req = urllib.request.Request(url, data=body, headers={
            "User-Agent": USER_AGENT, "Accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded"})
        try:
            with urllib.request.urlopen(req, timeout=timeout + 5, context=ctx) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:  # server bereikt, maar hij wil nu niet (druk, geweigerd)
            problems.append(("http", e.code))
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
            reason = getattr(e, "reason", e)
            problems.append(("ssl" if isinstance(reason, ssl.SSLError) or "CERTIFICATE" in str(e) else "net", str(reason)))
        logging.warning("OpenStreetMap via %s mislukt: %s", url, problems[-1])
    kinds = {k for k, _ in problems}
    if "http" in kinds:
        codes = sorted({str(c) for k, c in problems if k == "http"})
        raise OsmError(f"de OpenStreetMap-servers zijn nu druk of weigeren het verzoek (code {', '.join(codes)}); "
                       "probeer het over een paar minuten opnieuw")
    if kinds == {"ssl"}:
        raise OsmError("de beveiligde verbinding wordt niet vertrouwd (certificaten van Python ontbreken). "
                       "Installeer ze met: .venv/bin/pip install --upgrade certifi")
    raise OsmError(f"geen verbinding met internet? ({problems[0][1] if problems else 'onbekend'})")
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
        elif key.endswith("quicktime.creationdate"):  # iPhone: begin van de opname, met tijdzone
            ts = parse_time(val)
            if ts is not None:
                meta["rec_start"] = ts
        elif key == "creation_time" and "rec_start" not in meta:
            ts = parse_time(val)
            if ts is not None:
                meta["rec_start_fallback"] = ts
    if "rec_start" not in meta and "rec_start_fallback" in meta:
        meta["rec_start"] = meta["rec_start_fallback"]
    meta.pop("rec_start_fallback", None)
    return meta


def parse_time(val: str) -> float | None:
    """'2026-10-02T14:03:11+0200' of '2026-10-02T12:03:11.000000Z' -> seconden sinds 1970 (UTC)."""
    from datetime import datetime, timezone
    v = val.strip().replace("Z", "+00:00")
    if len(v) >= 5 and v[-5] in "+-" and v[-3] != ":":  # +0200 -> +02:00
        v = v[:-2] + ":" + v[-2:]
    try:
        dt = datetime.fromisoformat(v)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def to_local(lat: float, lon: float, lat0: float, lon0: float) -> np.ndarray:
    """Graden -> meters (oost, noord) rond (lat0, lon0). Op veldschaal nauwkeurig genoeg."""
    r = 6371000.0
    return np.array([math.radians(lon - lon0) * r * math.cos(math.radians(lat0)),
                     math.radians(lat - lat0) * r])


def query_pitches(lat: float, lon: float, radius: int = 300, timeout: int = 25) -> list[list[tuple[float, float]]]:
    """Omtrek (lat, lon) van voetbalvelden in de buurt, uit OpenStreetMap (Overpass)."""
    q = (f"[out:json][timeout:{timeout}];"
         f"(way(around:{radius},{lat},{lon})[\"leisure\"=\"pitch\"];);out geom;")
    data = _overpass(q, timeout)
    out = []
    for el in data.get("elements", []):
        sport = (el.get("tags") or {}).get("sport", "soccer")
        geom = el.get("geometry") or []
        if "soccer" in sport and len(geom) >= 4:
            out.append([(g["lat"], g["lon"]) for g in geom])
    return out


def camera_on_pitch(lat: float, lon: float, polygons: list[list[tuple[float, float]]],
                    geom: pitch.Geometry = pitch.DEFAULT) -> dict:
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
    # naar de veldmaten van de wedstrijd (zijn die gelijk aan het OpenStreetMap-veld, dan 1-op-1)
    x = geom.length / 2 + (rel @ u) * geom.length / length
    y = geom.width / 2 + (rel @ v) * geom.width / width
    return {"x": round(float(x), 1), "y": round(float(y), 1), "pitch_length": round(float(length), 1),
            "pitch_width": round(float(width), 1), "distance_to_pitch": round(float(best[0]), 1)}


def distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Afstand in meters tussen twee GPS-posities (kleine afstanden, platte benadering)."""
    r = 6371000.0
    dx = math.radians(lon2 - lon1) * r * math.cos(math.radians((lat1 + lat2) / 2))
    dy = math.radians(lat2 - lat1) * r
    return math.hypot(dx, dy)
