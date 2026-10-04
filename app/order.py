"""Volgorde van de video's en het verloop van de wedstrijd, uit de opnametijd.

Metafoor: losse foto's van een dag op volgorde leggen met de klok van de camera. Waar een lang gat
zit (de rust), begint de 2e helft. De eerste video van een helft begint op de aftrap van die helft
(0' of de speeltijd van één helft); de rest telt daar vanaf door."""
from __future__ import annotations

import logging
from pathlib import Path

from .storage import Store

BREAK_MIN_S = 8 * 60  # een gat van minstens zoveel tussen twee video's = de rust
HALF_LENGTHS = (30, 40, 45)


def ensure_rec_start(store: Store, clip: dict) -> float | None:
    """Opnametijd van een video; oudere video's (van vóór deze versie) lezen we nu alsnog uit."""
    if clip.get("rec_start") is not None:
        return clip["rec_start"]
    if not clip.get("path") or not Path(clip["path"]).exists():
        return None
    from .geo import read_video_metadata
    from .pipeline import ffmpeg_exe
    try:
        ts = read_video_metadata(Path(clip["path"]), ffmpeg_exe()).get("rec_start")
    except Exception:  # noqa: BLE001
        logging.exception("Metadata van %s", clip["path"])
        return None
    if ts is not None:
        store.run("UPDATE clips SET rec_start = ? WHERE id = ?", (ts, clip["id"]))
    return ts


def propose(store: Store, match_id: int, half_length: int | None = None) -> dict:
    """Voorstel: per video de volgorde, helft en beginminuut. Video's zonder opnametijd blijven achteraan
    in hun huidige volgorde (en houden helft en minuut)."""
    match = store.one("SELECT * FROM matches WHERE id = ?", (match_id,))
    half = int(half_length or (match or {}).get("half_length") or 45)
    clips = store.all("SELECT * FROM clips WHERE match_id = ? ORDER BY order_idx, id", (match_id,))
    timed = [(ensure_rec_start(store, c), c) for c in clips]
    known = sorted([(ts, c) for ts, c in timed if ts is not None], key=lambda x: x[0])
    unknown = [c for ts, c in timed if ts is None]
    # de rust: het grootste gat (van einde video tot begin volgende) van minstens 8 minuten
    brk, brk_gap = None, 0.0
    for k in range(1, len(known)):
        prev_ts, prev = known[k - 1]
        gap = known[k][0] - (prev_ts + float(prev.get("duration") or 0))
        if gap >= BREAK_MIN_S and gap > brk_gap:
            brk, brk_gap = k, gap
    items = []
    first1 = known[0][0] if known else None
    first2 = known[brk][0] if brk is not None else None
    for k, (ts, c) in enumerate(known):
        second = brk is not None and k >= brk
        start = (ts - first2) / 60 + half if second else (ts - first1) / 60
        items.append({"clip_id": c["id"], "filename": c["filename"], "rec_start": ts, "order_idx": k,
                      "period": 2 if second else 1, "start_minute": round(start, 1),
                      "old": {"order_idx": c["order_idx"], "period": c["period"], "start_minute": c["start_minute"]}})
    for j, c in enumerate(unknown):
        items.append({"clip_id": c["id"], "filename": c["filename"], "rec_start": None, "order_idx": len(known) + j,
                      "period": c["period"], "start_minute": c["start_minute"],
                      "old": {"order_idx": c["order_idx"], "period": c["period"], "start_minute": c["start_minute"]}})
    return {"half_length": half, "items": items, "break_after": (known[brk - 1][1]["id"] if brk is not None else None),
            "break_minutes": round(brk_gap / 60) if brk is not None else None, "n_without_time": len(unknown)}


def apply(store: Store, match_id: int, items: list[dict]) -> int:
    n = 0
    for it in items:
        c = store.one("SELECT id FROM clips WHERE id = ? AND match_id = ?", (int(it["clip_id"]), match_id))
        if c is None:
            continue
        store.run("UPDATE clips SET order_idx = ?, period = ?, start_minute = ? WHERE id = ?",
                  (int(it["order_idx"]), int(it["period"]), float(it["start_minute"]), c["id"]))
        n += 1
    return n
