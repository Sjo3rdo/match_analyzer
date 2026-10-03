"""Hoogtepunten vinden in het geluid: gejuich en fluitsignalen.

Zoals Veo hoogtepunten voorstelt, luisteren we hier naar de video:
- Gejuich: het wordt ineens een stuk luider dan wat in die fase van de wedstrijd normaal is, en
  dat houdt even aan (minstens anderhalve seconde; een losse roep van een speler is korter). We meten alleen het stemgebied (300 - 4000 Hz), zodat
  windgeruis in de microfoon van de telefoon niet als gejuich telt. Het moment zelf (de kans of
  de goal) zit meestal net vóór het gejuich; een clip begint daarom een stuk eerder.
- Fluitsignaal: een fluit is een heldere, hoge toon (ongeveer 2 - 4,5 kHz). In dat gebied staat dan
  één duidelijke piek die veel sterker is dan de rest, een tiende tot een paar seconden lang.

Het zijn suggesties: de gebruiker kijkt of het echt iets was en maakt er met één klik een clip van.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

SR = 16000
HOP = SR // 10  # 0,1 s per stap
NFFT = 2048
VERSION = 2


def extract_audio(path: Path, ffmpeg: str) -> np.ndarray | None:
    """Geluid als mono 16 kHz (float32, -1..1); None als de video geen geluid heeft."""
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(SR),
           "-f", "s16le", "-"]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0 or not r.stdout:
        return None
    return np.frombuffer(r.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def _features(x: np.ndarray, chunk: int = 600) -> dict[str, np.ndarray]:
    """Per 0,1 s: luidheid in het stemgebied (dB), hoe 'helder' dat geluid is (deel boven 1 kHz),
    hoeveel er onder de 300 Hz zit (wind, gestommel), en voor het fluitgebied het aandeel van de
    energie en hoe 'toonachtig' het is (sterkste piek t.o.v. het gemiddelde in dat gebied)."""
    n = len(x) // HOP
    freqs = np.fft.rfftfreq(NFFT, 1 / SR)
    voice = (freqs >= 300) & (freqs <= 4000)
    high = (freqs >= 1000) & (freqs <= 4000)
    low = freqs < 300
    whistle = (freqs >= 2000) & (freqs <= 4500)
    total = freqs >= 150
    win = np.hanning(NFFT).astype(np.float32)
    pad = np.pad(x, (NFFT // 2, NFFT))
    level = np.empty(n, np.float32)
    bright = np.empty(n, np.float32)
    lowf = np.empty(n, np.float32)
    w_frac = np.empty(n, np.float32)
    w_peak = np.empty(n, np.float32)
    for a in range(0, n, chunk):  # in stukken, zodat een hele wedstrijd weinig geheugen kost
        idx = np.arange(a, min(n, a + chunk))
        frames = np.stack([pad[i * HOP:i * HOP + NFFT] for i in idx]) * win
        p = np.abs(np.fft.rfft(frames, axis=1)) ** 2
        ev = p[:, voice].sum(1)
        level[idx] = 10 * np.log10(ev + 1e-9)
        bright[idx] = p[:, high].sum(1) / (ev + 1e-12)
        lowf[idx] = p[:, low].sum(1) / (p.sum(1) + 1e-12)
        wb = p[:, whistle]
        w_frac[idx] = wb.sum(1) / (p[:, total].sum(1) + 1e-9)
        w_peak[idx] = wb.max(1) / (wb.mean(1) + 1e-12)
    return {"level": level, "bright": bright, "low": lowf, "w_frac": w_frac, "w_peak": w_peak}


def _background(level: np.ndarray, window_s: float = 30.0) -> np.ndarray:
    """Wat 'normaal' is in deze fase: lopende mediaan over een halve minuut."""
    block = 10  # per seconde
    nb = max(1, len(level) // block)
    blocks = np.array([np.median(level[i * block:(i + 1) * block]) for i in range(nb)])
    half = int(window_s / 2)
    med = np.array([np.median(blocks[max(0, i - half):i + half + 1]) for i in range(nb)])
    return np.interp(np.arange(len(level)) / block, np.arange(nb) + 0.5, med)


def _groups(mask: np.ndarray, max_gap: int = 3) -> list[tuple[int, int]]:
    """Aaneengesloten stukken (start, eind) waar mask waar is; korte gaatjes worden overbrugd."""
    out: list[tuple[int, int]] = []
    on = np.flatnonzero(mask)
    if not len(on):
        return out
    start = prev = on[0]
    for i in on[1:]:
        if i - prev > max_gap:
            out.append((start, prev))
            start = i
        prev = i
    out.append((start, prev))
    return out


def find_events(x: np.ndarray) -> list[dict]:
    """Gejuich en fluitsignalen in een geluidsfragment (16 kHz mono)."""
    if x is None or len(x) < SR:
        return []
    f = _features(x)
    level, w_frac, w_peak = f["level"], f["w_frac"], f["w_peak"]
    bg = _background(level)
    excess = level - bg
    events = []
    # fluitsignaal: heldere hoge toon die boven het achtergrondgeluid uitkomt
    tonal = (w_frac > 0.4) & (w_peak > 25.0) & (excess > -3.0)
    # gejuich: minstens 8 dB boven normaal, minstens 1 s, niet vooral een fluit, en klinkend als
    # een joelende groep (ook hoge tonen) in plaats van wind in de microfoon of één zware stem vlakbij
    for a, b in _groups(excess > 8.0):
        dur = (b - a + 1) / 10
        if dur < 1.5 or tonal[a:b + 1].mean() > 0.5:  # korter: één roep ("hier!"), geen gejuich
            continue
        if np.mean(f["bright"][a:b + 1]) < 0.12 or np.mean(f["low"][a:b + 1]) > 0.6:
            continue
        score = float(np.mean(excess[a:b + 1]) * np.sqrt(min(dur, 6.0)))
        events.append({"kind": "gejuich", "t": round(a / 10, 1), "t_end": round((b + 1) / 10, 1),
                       "score": round(float(score), 1)})
    for a, b in _groups(tonal, max_gap=1):
        dur = (b - a + 1) / 10
        if not 0.2 <= dur <= 4.0:
            continue
        events.append({"kind": "fluitsignaal", "t": round(a / 10, 1), "t_end": round((b + 1) / 10, 1),
                       "score": round(float(np.mean(w_peak[a:b + 1]) / 10 * min(dur, 2.0)), 1)})
    return sorted(_strongest_per_window(events), key=lambda e: e["t"])


def _strongest_per_window(events: list[dict], window: float = 30.0) -> list[dict]:
    """Per soort hooguit één moment per halve minuut: het sterkste. Bij een hele wedstrijd blijft
    de lijst zo te overzien, en een doelpunt geeft vaak een paar golven gejuich vlak na elkaar."""
    kept: list[dict] = []
    for e in sorted(events, key=lambda e: -e["score"]):
        if not any(k["kind"] == e["kind"] and abs(k["t"] - e["t"]) < window for k in kept):
            kept.append(e)
    return kept


def clip_events(clip: dict, out_dir: Path, ffmpeg: str) -> list[dict]:
    """Geluidsmomenten van een video (eenmalig berekend en bewaard naast de video)."""
    cache = out_dir / "audio_events.json"
    if cache.exists():
        try:
            data = json.loads(cache.read_text())
            if data.get("version") == VERSION:
                return data["events"]
        except (ValueError, KeyError):
            pass
    x = extract_audio(Path(clip["path"]), ffmpeg)
    events = find_events(x) if x is not None else []
    cache.write_text(json.dumps({"version": VERSION, "events": events}))
    return events
