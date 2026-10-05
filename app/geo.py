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
USER_AGENT = "match-analyzer/0.18 (lokale voetbalanalyse; https://github.com/sjo3rdo/match_analyzer)"
OSM_API = "https://api.openstreetmap.org/api/0.6/map"


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


def _problem(e: Exception) -> tuple[str, str]:
    """Wat ging er mis, in een paar woorden: ('http', code), ('ssl', ...) of ('net', ...)."""
    if isinstance(e, urllib.error.HTTPError):  # server bereikt, maar hij wil nu niet (druk, geweigerd)
        return ("http", str(e.code))
    reason = getattr(e, "reason", e)
    return ("ssl" if isinstance(reason, ssl.SSLError) or "CERTIFICATE" in str(e) else "net", str(reason))


def _get(url: str, data: bytes | None, timeout: float) -> bytes:
    headers = {"User-Agent": USER_AGENT}
    if data is not None:
        headers["Content-Type"] = "application/x-www-form-urlencoded"
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout, context=_ssl_context()) as resp:
        return resp.read()


def _overpass(query: str, timeout: int, problems: list) -> dict | None:
    """Alle Overpass-servers tegelijk vragen; de eerste die antwoordt wint (niet drie keer wachten)."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    body = urllib.parse.urlencode({"data": query}).encode()
    pool = ThreadPoolExecutor(max_workers=len(OVERPASS_URLS))
    futs = {pool.submit(_get, url, body, timeout + 5): url for url in OVERPASS_URLS}
    try:
        for fut in as_completed(futs):
            try:
                return json.loads(fut.result())
            except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
                problems.append(_problem(e))
                logging.warning("OpenStreetMap via %s mislukt: %s", futs[fut], problems[-1])
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    return None


def _osm_api_pitches(bbox: tuple[float, float, float, float], timeout: float) -> list[list[tuple[float, float]]]:
    """Terugval: het kaartje van een klein vierkant direct van openstreetmap.org (een andere dienst dan
    de zoekservers). bbox = (zuid, west, noord, oost)."""
    import xml.etree.ElementTree as ET
    s_, w_, n_, e_ = bbox
    raw = _get(f"{OSM_API}?bbox={w_:.6f},{s_:.6f},{e_:.6f},{n_:.6f}", None, timeout)
    root = ET.fromstring(raw)
    nodes = {nd.get("id"): (float(nd.get("lat")), float(nd.get("lon"))) for nd in root.iter("node")}
    out = []
    for way in root.iter("way"):
        tags = {t.get("k"): t.get("v") for t in way.iter("tag")}
        if tags.get("leisure") != "pitch":
            continue
        if "soccer" not in tags.get("sport", "soccer"):
            continue
        pts = [nodes[r.get("ref")] for r in way.iter("nd") if r.get("ref") in nodes]
        if len(pts) >= 4:
            out.append(pts)
    return out


def _osm_error(problems: list) -> OsmError:
    kinds = {k for k, _ in problems}
    if kinds == {"ssl"}:
        return OsmError("de beveiligde verbinding wordt niet vertrouwd (certificaten van Python ontbreken). "
                        "Installeer ze met: .venv/bin/pip install --upgrade certifi")
    if "http" in kinds or any("timed out" in d.lower() for _, d in problems):
        codes = sorted({d for k, d in problems if k == "http"})
        return OsmError("de OpenStreetMap-servers zijn nu te druk en antwoorden niet op tijd"
                        + (f" (code {', '.join(codes)})" if codes else "") + "; probeer het over een paar minuten opnieuw")
    return OsmError(f"geen verbinding met internet? ({problems[0][1] if problems else 'onbekend'})")


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


def query_pitches(lat: float, lon: float, radius: int = 300, timeout: int = 20) -> list[list[tuple[float, float]]]:
    """Omtrek (lat, lon) van voetbalvelden in de buurt, uit OpenStreetMap.

    Eerst de zoekservers (Overpass, allemaal tegelijk), met een lichte vraag: alleen velden in een
    klein vierkant rond je plek. Antwoordt geen enkele op tijd, dan het kaartje van dat vierkant
    rechtstreeks van openstreetmap.org. Wat gevonden is, onthoudt de app voor die plek."""
    from . import config
    key = f"{lat:.4f},{lon:.4f},{radius}"
    cache_path = config.DATA_DIR / "osm_cache.json"
    try:
        cache = json.loads(cache_path.read_text())
    except (OSError, ValueError):
        cache = {}
    if key in cache:
        return [[tuple(p) for p in poly] for poly in cache[key]]
    dlat = radius / 111_320
    dlon = radius / (111_320 * max(0.2, math.cos(math.radians(lat))))
    bbox = (lat - dlat, lon - dlon, lat + dlat, lon + dlon)  # zuid, west, noord, oost
    problems: list = []
    q = (f"[out:json][timeout:{timeout}];"
         f"way[\"leisure\"=\"pitch\"]({bbox[0]:.6f},{bbox[1]:.6f},{bbox[2]:.6f},{bbox[3]:.6f});out geom;")
    data = _overpass(q, timeout, problems)
    if data is not None:
        out = []
        for el in data.get("elements", []):
            sport = (el.get("tags") or {}).get("sport", "soccer")
            geom = el.get("geometry") or []
            if "soccer" in sport and len(geom) >= 4:
                out.append([(g["lat"], g["lon"]) for g in geom])
    else:
        try:
            out = _osm_api_pitches(bbox, timeout + 10)
        except Exception as e:  # noqa: BLE001
            problems.append(_problem(e) if isinstance(e, OSError) else ("net", str(e)))
            logging.warning("OpenStreetMap via %s mislukt: %s", OSM_API, problems[-1])
            raise _osm_error(problems) from e
    if out:
        cache[key] = out
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(cache))
        except OSError:
            pass
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
