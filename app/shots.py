"""Schoten en goals: voorstellen uit het balspoor en het geluid, en wat de gebruiker bevestigde.

Een voorstel wordt pas een schot of goal als jij het bevestigt (of zelf toevoegt). Bevestigde
schoten tellen mee in de statistieken, de schotenkaart en de stand."""
from __future__ import annotations

import logging

from . import analytics, pitch
from .storage import Store, clip_dir

SAME_SHOT_S = 1.5  # een voorstel en een opgeslagen schot binnen deze tijd zijn hetzelfde


def _sounds(clip: dict) -> list[dict]:
    from . import audio
    from .pipeline import ffmpeg_exe
    try:
        return audio.clip_events(clip, clip_dir(clip["id"]), ffmpeg_exe())
    except Exception:  # noqa: BLE001  (geen geluid: dan alleen op de bal)
        logging.exception("Geluid van clip %s", clip["id"])
        return []


def _map_xy(clip: dict, geom: pitch.Geometry, x, y):
    """Plek op de schotenkaart: 2e helft gespiegeld, zodat beide helften dezelfde kant op spelen."""
    if x is None or y is None:
        return None
    if analytics.flipped(clip):
        return [round(geom.length - x, 1), round(geom.width - y, 1)]
    return [round(x, 1), round(y, 1)]


def _map_goal_x(clip: dict, geom: pitch.Geometry, gx):
    if gx is None:
        return None
    return round(geom.length - gx, 1) if analytics.flipped(clip) else gx


def overview(store: Store, match_id: int) -> dict:
    match = store.one("SELECT * FROM matches WHERE id = ?", (match_id,))
    geom = pitch.of_match(match)
    clips = {c["id"]: c for c in store.all("SELECT * FROM clips WHERE match_id = ? ORDER BY order_idx, id", (match_id,))}
    players = {p["id"]: p for p in store.all("SELECT * FROM players WHERE match_id = ?", (match_id,))}
    rows = store.all("SELECT * FROM shots WHERE match_id = ? ORDER BY clip_id, t", (match_id,))

    def minute(c, t):
        return round((c.get("start_minute") or 0) + t / 60, 2)

    # voorstellen
    suggestions = []
    for c in clips.values():
        if c["status"] != "klaar":
            continue
        d = analytics.load_clip(store, c["id"])
        if d is None or not d.calibrated:
            continue
        found = analytics.detect_shots(d)
        if not found:
            continue
        sounds = _sounds(c)
        done = [r["t"] for r in rows if r["clip_id"] == c["id"]]
        for sh in found:
            if any(abs(sh["t"] - t) < SAME_SHOT_S for t in done):
                continue
            ent = sh.get("entity") or ""
            pid = int(ent[1:]) if ent.startswith("p") and int(ent[1:]) in players else None
            suggestions.append({**sh, "clip_id": c["id"], "minute": minute(c, sh["t"]), "player_id": pid,
                                "goal_chance": analytics.goal_chance(sh, sounds),
                                "map": _map_xy(c, geom, sh["x"], sh["y"]), "map_goal_x": _map_goal_x(c, geom, sh["goal_x"])})
    suggestions.sort(key=lambda s: ({"waarschijnlijk": 0, "mogelijk": 1}.get(s["goal_chance"], 2), s["clip_id"], s["t"]))

    # bevestigd
    shots = []
    for r in rows:
        if r["status"] != "bevestigd" or r["clip_id"] not in clips:
            continue
        c = clips[r["clip_id"]]
        p = players.get(r["player_id"])
        team = p["team"] if p and r["team"] is None else r["team"]
        shots.append({**r, "team": team, "goal": bool(r["goal"]), "on_target": bool(r["on_target"] or r["goal"]),
                      "minute": minute(c, r["t"]), "period": c.get("period") or 1,
                      "map": _map_xy(c, geom, r["x"], r["y"]), "map_goal_x": _map_goal_x(c, geom, r["goal_x"])})
    shots.sort(key=lambda s: (s["period"], s["minute"]))

    # stand en tijdlijn: de goals op volgorde van de wedstrijd
    score = [0, 0]
    timeline = []
    for s in shots:
        if s["goal"] and s["team"] in (0, 1):
            score[s["team"]] += 1
            timeline.append({"id": s["id"], "minute": s["minute"], "team": s["team"], "player_id": s["player_id"],
                             "clip_id": s["clip_id"], "t": s["t"], "score": list(score)})
    teams = []
    for team in (0, 1):
        mine = [s for s in shots if s["team"] == team]
        teams.append({"team": team, "shots": len(mine), "on_target": sum(s["on_target"] for s in mine),
                      "goals": sum(s["goal"] for s in mine)})
    per_player: dict[int, dict] = {}
    for s in shots:
        if s["player_id"]:
            q = per_player.setdefault(s["player_id"], {"shots": 0, "on_target": 0, "goals": 0})
            q["shots"] += 1
            q["on_target"] += int(s["on_target"])
            q["goals"] += int(s["goal"])
    return {"suggestions": suggestions, "shots": shots, "score": score, "timeline": timeline, "teams": teams,
            "players": {str(k): v for k, v in per_player.items()}}


def score(store: Store, match_id: int) -> list[int] | None:
    """Stand uit de bevestigde goals (None als er nog geen goal is ingevoerd)."""
    rows = store.all("SELECT s.team, s.player_id, p.team AS pteam FROM shots s LEFT JOIN players p ON p.id = s.player_id "
                     "WHERE s.match_id = ? AND s.status = 'bevestigd' AND s.goal = 1", (match_id,))
    if not rows:
        return None
    out = [0, 0]
    for r in rows:
        team = r["team"] if r["team"] is not None else r["pteam"]
        if team in (0, 1):
            out[team] += 1
    return out
