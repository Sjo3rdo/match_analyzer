"""Van ruwe detecties naar voetbalstatistieken.

Stap voor stap:
1. Voetpunt van elke speler (onderkant midden van de box) -> veldcoördinaten (meters)
   via het cameramodel uit de kalibratie.
2. Tracks die vooral buiten het veld staan (publiek, wissels) vallen af.
3. Tracks worden gekoppeld aan spelers ("entiteiten"); een speler kan meerdere tracks
   hebben, want volgen breekt af bij bijv. een botsing of als de camera wegdraait.
4. Per speler: afstand, topsnelheid, sprints, heatmap, gemiddelde positie.
5. Balbezit: de speler die het dichtst bij de bal staat (binnen 2 m) en dat een paar
   frames volhoudt. Wisselt bezit binnen 5 s naar een ploeggenoot = pass, naar de
   tegenstander = balverlies.
"""
from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import config, pitch
from .calibration import CameraModel, Keyframe, apply_h, camera_prior, fit_keyframe
from .storage import Store, clip_dir
from .teams import TEAM_OTHER

_cache: dict[tuple[int, int], "ClipData"] = {}
_cache_lock = threading.Lock()
_version = 0


def invalidate() -> None:
    """Aanroepen na elke wijziging die de analyse beïnvloedt."""
    global _version
    with _cache_lock:
        _version += 1
        _cache.clear()


@dataclass
class ClipData:
    clip: dict
    t: np.ndarray  # tijd (s) per geanalyseerd frame
    fps: float  # geanalyseerde frames per seconde
    calibrated: bool
    # detecties (rijen)
    idx: np.ndarray
    track: np.ndarray
    boxes: np.ndarray
    xy: np.ndarray  # veldpositie per detectie
    # bal
    ball_idx: np.ndarray
    ball_xy: np.ndarray
    ball_img: np.ndarray
    camera: CameraModel | None = None
    tracks: dict[int, dict] = field(default_factory=dict)
    entity: dict[int, str] = field(default_factory=dict)  # track -> entiteit
    team: dict[int, int] = field(default_factory=dict)  # track -> team
    valid_tracks: set[int] = field(default_factory=set)


def entity_key(clip_id: int, track_id: int, player_id: int | None) -> str:
    return f"p{player_id}" if player_id else f"c{clip_id}t{track_id}"


def load_clip(store: Store, clip_id: int) -> ClipData | None:
    with _cache_lock:
        cached = _cache.get((clip_id, _version))
    if cached is not None:
        return cached
    clip = store.one("SELECT * FROM clips WHERE id = ?", (clip_id,))
    if clip is None or clip["status"] != "klaar":
        return None
    frames = store.all("SELECT idx, t FROM frames WHERE clip_id = ? ORDER BY idx", (clip_id,))
    t = np.array([f["t"] for f in frames], dtype=np.float64)
    if len(t) == 0:
        return None
    fps = (len(t) - 1) / (t[-1] - t[0]) if len(t) > 1 and t[-1] > t[0] else config.TARGET_FPS

    motion_path = clip_dir(clip_id) / "motion.npy"
    inter = np.load(motion_path) if Path(motion_path).exists() else np.tile(np.eye(3), (len(t), 1, 1))
    kfs = []
    prior = camera_prior(clip)
    for kf in store.keyframes(clip_id):
        try:
            K, _ = fit_keyframe(kf, camera=prior)
        except ValueError:
            continue
        kfs.append(Keyframe(int(np.argmin(np.abs(t - kf["t"]))), K))
    cam = CameraModel(inter[:len(t)], kfs)

    rows = store.all("SELECT idx, track_id, x1, y1, x2, y2 FROM detections WHERE clip_id = ? "
                     "ORDER BY idx", (clip_id,))
    idx = np.array([r["idx"] for r in rows], dtype=int)
    track = np.array([r["track_id"] for r in rows], dtype=int)
    boxes = np.array([[r["x1"], r["y1"], r["x2"], r["y2"]] for r in rows]).reshape(-1, 4)
    feet = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]], axis=1) if len(rows) else np.zeros((0, 2))
    xy = _project_rows(cam, idx, feet)

    brows = store.all("SELECT idx, x, y FROM ball WHERE clip_id = ? ORDER BY idx", (clip_id,))
    ball_idx = np.array([r["idx"] for r in brows], dtype=int)
    ball_img = np.array([[r["x"], r["y"]] for r in brows]).reshape(-1, 2)
    ball_xy = _project_rows(cam, ball_idx, ball_img)

    data = ClipData(clip, t, fps, cam.calibrated, idx, track, boxes, xy, ball_idx, ball_xy, ball_img,
                    camera=cam)
    players = {p["id"]: p for p in store.all(
        "SELECT * FROM players WHERE match_id = ?", (clip["match_id"],))}
    for tr in store.all("SELECT * FROM tracks WHERE clip_id = ?", (clip_id,)):
        tid = tr["track_id"]
        data.tracks[tid] = tr
        pid = tr["player_id"]
        data.entity[tid] = entity_key(clip_id, tid, pid)
        data.team[tid] = players[pid]["team"] if pid in players else tr["team"]
    data.valid_tracks = _valid_tracks(data)
    with _cache_lock:
        _cache[(clip_id, _version)] = data
    return data


def _project_rows(cam: CameraModel, idx: np.ndarray, pts: np.ndarray) -> np.ndarray:
    out = np.full((len(idx), 2), np.nan)
    if not cam.calibrated or len(idx) == 0:
        return out
    order = np.argsort(idx, kind="stable")
    bounds = np.flatnonzero(np.diff(idx[order])) + 1
    for group in np.split(order, bounds):
        out[group] = cam.project(int(idx[group[0]]), pts[group])
    return out


def _valid_tracks(d: ClipData, min_frames: int = 10, min_on_pitch: float = 0.6) -> set[int]:
    """Tracks die echt spelers op het veld zijn (geen toeschouwers, wissels of mensen vlak voor de camera).

    - Kort in beeld (< min_frames): weg.
    - Voeten meestal onder de rand van het beeld: iemand vlak voor de camera, niet op het veld.
    - Staat de hele tijd op dezelfde plek (gemeten tegen de achtergrond, dus los van het zwenken):
      een toeschouwer. Na kalibratie geldt dat alleen langs de zijlijnen of buiten het veld, zodat
      een keeper die even stilstaat blijft meetellen.
    - Na kalibratie: meestal buiten het veld."""
    valid = set()
    if not len(d.track):
        return valid
    H_img = float(d.clip.get("height") or 0)
    feet = np.stack([(d.boxes[:, 0] + d.boxes[:, 2]) / 2, d.boxes[:, 3], np.ones(len(d.boxes))], 1)
    A = d.camera.A if d.camera is not None and len(d.camera.A) else None
    if A is not None:  # voetpunt in het referentiebeeld (camerabeweging eruit)
        ref = np.einsum("nij,nj->ni", A[np.clip(d.idx, 0, len(A) - 1)], feet)
        ref = ref[:, :2] / np.where(np.abs(ref[:, 2:3]) > 1e-9, ref[:, 2:3], 1e-9)
    else:
        ref = feet[:, :2]
    height = d.boxes[:, 3] - d.boxes[:, 1]
    for tid in np.unique(d.track):
        m = d.track == tid
        if m.sum() < min_frames:
            continue
        if H_img and (d.boxes[m, 3] >= H_img - 3).mean() > 0.5:
            continue
        span = d.t[d.idx[m]].max() - d.t[d.idx[m]].min()
        r = ref[m]
        dev = np.linalg.norm(r - np.median(r, axis=0), axis=1)
        stationary = span >= 6.0 and np.percentile(dev, 90) < 0.5 * max(1.0, float(np.median(height[m])))
        if d.calibrated:
            x, y = d.xy[m, 0], d.xy[m, 1]
            ok = (x >= -1.5) & (x <= pitch.LENGTH + 1.5) & (y >= -1.5) & (y <= pitch.WIDTH + 1.5)
            if ok.mean() < min_on_pitch:
                continue
            my = float(np.nanmedian(y)) if np.isfinite(y).any() else 0.0
            if stationary and (my < 4.0 or my > pitch.WIDTH - 4.0):
                continue
        elif stationary:
            continue
        valid.add(int(tid))
    return valid


# --- bewegingsstatistieken ---------------------------------------------------------------

def smooth_series(t: np.ndarray, xy: np.ndarray, max_gap: float = 0.6, window: int = 5
                  ) -> list[tuple[np.ndarray, np.ndarray]]:
    """Knip een reeks op bij gaten en strijk elk stuk glad (voortschrijdend gemiddelde)."""
    keep = ~np.isnan(xy).any(axis=1)
    t, xy = t[keep], xy[keep]
    if len(t) == 0:
        return []
    cuts = np.flatnonzero(np.diff(t) > max_gap) + 1
    segments = []
    for ts, ps in zip(np.split(t, cuts), np.split(xy, cuts)):
        if len(ts) < 2:
            continue
        w = min(window, len(ts))
        kernel = np.ones(w) / w
        pad = w // 2
        sm = np.stack([np.convolve(np.pad(ps[:, k], (pad, w - 1 - pad), mode="edge"), kernel, "valid")
                       for k in range(2)], axis=1)
        segments.append((ts, sm))
    return segments


def movement_stats(segments: list[tuple[np.ndarray, np.ndarray]]) -> dict:
    dist, max_speed, seconds = 0.0, 0.0, 0.0
    sprints: list[tuple[float, float, float]] = []  # (start, eind, topsnelheid)
    for ts, ps in segments:
        dt = np.diff(ts)
        step = np.linalg.norm(np.diff(ps, axis=0), axis=1)
        speed = step / np.maximum(dt, 1e-6)
        ok = speed <= config.MAX_SPEED_MS
        dist += float(step[ok].sum())
        seconds += float(ts[-1] - ts[0])
        if ok.any():
            max_speed = max(max_speed, float(speed[ok].max()))
        fast = ok & (speed >= config.SPRINT_SPEED_MS)
        start = None
        for i, f in enumerate(np.append(fast, False)):
            if f and start is None:
                start = i
            elif not f and start is not None:
                t0, t1 = ts[start], ts[i]
                if t1 - t0 >= config.SPRINT_MIN_S:
                    sprints.append((float(t0), float(t1), float(speed[start:i].max())))
                start = None
    return {"distance_m": dist, "max_speed_ms": max_speed, "seconds": seconds, "sprints": sprints}


def heatmap(points: np.ndarray, fps: float) -> list[list[float]]:
    """Seconden per veldcel (rijen = y, kolommen = x)."""
    nx, ny = config.HEATMAP_BINS
    if len(points) == 0:
        return np.zeros((ny, nx)).tolist()
    h, _, _ = np.histogram2d(points[:, 1], points[:, 0], bins=[ny, nx],
                             range=[[0, pitch.WIDTH], [0, pitch.LENGTH]])
    return (h / fps).round(2).tolist()


# --- balbezit en passes ------------------------------------------------------------------

@dataclass
class Possession:
    entity: str
    team: int
    track: int
    t0: float
    t1: float


def possession_segments(d: ClipData) -> list[Possession]:
    if not d.calibrated or len(d.ball_idx) == 0:
        return []
    by_idx: dict[int, list[int]] = defaultdict(list)
    for row in range(len(d.idx)):
        if int(d.track[row]) in d.valid_tracks:
            by_idx[int(d.idx[row])].append(row)
    owners: list[tuple[int, int | None]] = []  # (frame, track)
    for bi, bxy in zip(d.ball_idx, d.ball_xy):
        rows = by_idx.get(int(bi), [])
        if not rows or np.isnan(bxy).any():
            owners.append((int(bi), None))
            continue
        dist = np.linalg.norm(d.xy[rows] - bxy, axis=1)
        k = int(np.nanargmin(dist)) if not np.isnan(dist).all() else -1
        owners.append((int(bi), int(d.track[rows[k]]) if k >= 0 and dist[k] <= config.POSSESSION_RADIUS_M else None))

    segs: list[Possession] = []
    run_track, run_start, run_len, prev_idx = None, 0, 0, -10
    max_skip = max(1, int(round(0.3 * d.fps)))

    def close(end_idx: int):
        if run_track is not None and run_len >= config.POSSESSION_MIN_FRAMES:
            tid = run_track
            segs.append(Possession(d.entity.get(tid, f"t{tid}"), d.team.get(tid, -1), tid,
                                   float(d.t[run_start]), float(d.t[end_idx])))

    for fi, tid in owners + [(10**9, None)]:
        contiguous = fi - prev_idx <= max_skip
        if tid is not None and tid == run_track and contiguous:
            run_len += 1
        else:
            close(min(prev_idx, len(d.t) - 1))
            run_track, run_start, run_len = (tid, fi, 1) if tid is not None else (None, 0, 0)
        prev_idx = fi
    # aaneengesloten stukken van dezelfde speler samenvoegen
    merged: list[Possession] = []
    for s in segs:
        if merged and merged[-1].entity == s.entity and s.t0 - merged[-1].t1 <= 1.0:
            merged[-1].t1 = s.t1
        else:
            merged.append(s)
    return merged


def passes(segs: list[Possession]) -> list[dict]:
    """Opeenvolgende bezitters: zelfde team = pass, ander team = balverlies."""
    out = []
    real = [s for s in segs if s.team in (0, 1)]
    for a, b in zip(real, real[1:]):
        if a.entity == b.entity or b.t0 - a.t1 > config.PASS_MAX_GAP_S:
            continue
        out.append({"t": a.t1, "t_end": b.t0, "from": a.entity, "to": b.entity,
                    "team": a.team, "success": a.team == b.team})
    return out


# --- wedstrijdniveau ---------------------------------------------------------------------

def match_stats(store: Store, match_id: int) -> dict:
    match = store.one("SELECT * FROM matches WHERE id = ?", (match_id,))
    players = {f"p{p['id']}": p for p in store.all("SELECT * FROM players WHERE match_id = ?", (match_id,))}
    clips = store.all("SELECT * FROM clips WHERE match_id = ? ORDER BY order_idx, id", (match_id,))

    seg_by_entity: dict[str, list] = defaultdict(list)
    pts_by_entity: dict[str, list] = defaultdict(list)
    sec_by_entity: dict[str, float] = defaultdict(float)
    team_of: dict[str, int] = {}
    label_of: dict[str, str] = {}
    events: list[dict] = []
    poss_time = [0.0, 0.0]
    all_passes: list[dict] = []
    touches: dict[str, int] = defaultdict(int)
    status = []

    for clip in clips:
        d = load_clip(store, clip["id"])
        status.append({"clip_id": clip["id"], "filename": clip["filename"], "status": clip["status"],
                       "calibrated": bool(d and d.calibrated)})
        if d is None or not d.calibrated:
            continue
        for tid in d.valid_tracks:
            ent = d.entity[tid] if tid in d.entity else entity_key(clip["id"], tid, None)
            if ent not in players and d.team.get(tid, -1) == TEAM_OTHER:
                continue
            m = d.track == tid
            ts, xy = d.t[d.idx[m]], d.xy[m]
            seg_by_entity[ent].extend(smooth_series(ts, xy))
            on = ~np.isnan(xy).any(axis=1)
            pts_by_entity[ent].append(xy[on])
            sec_by_entity[ent] += on.sum() / d.fps
            team_of.setdefault(ent, d.team.get(tid, -1))
            label_of.setdefault(ent, f"Track {tid} ({clip['filename']})")
            for s in movement_stats(smooth_series(ts, xy))["sprints"]:
                events.append({"kind": "sprint", "clip_id": clip["id"], "t": s[0], "t_end": s[1],
                               "entity": ent, "value": round(s[2] * 3.6, 1)})
        segs = possession_segments(d)
        for s in segs:
            touches[s.entity] += 1
            if s.team in (0, 1):
                poss_time[s.team] += s.t1 - s.t0 + 1 / d.fps
        for p in passes(segs):
            p["clip_id"] = clip["id"]
            all_passes.append(p)
            events.append({"kind": "pass" if p["success"] else "balverlies", "clip_id": clip["id"],
                           "t": p["t"], "t_end": p["t_end"], "entity": p["from"], "to": p["to"]})

    rows = []
    for ent, segs in seg_by_entity.items():
        mv = movement_stats(segs)
        pts = np.concatenate(pts_by_entity[ent]) if pts_by_entity[ent] else np.zeros((0, 2))
        p = players.get(ent)
        fps = config.TARGET_FPS
        rows.append({
            "entity": ent,
            "player_id": p["id"] if p else None,
            "name": p["name"] if p else label_of.get(ent, ent),
            "number": p["number"] if p else None,
            "team": p["team"] if p else team_of.get(ent, -1),
            "minutes": round(sec_by_entity[ent] / 60, 1),
            "distance_m": round(mv["distance_m"]),
            "max_speed_kmh": round(mv["max_speed_ms"] * 3.6, 1),
            "sprints": len(mv["sprints"]),
            "avg_pos": pts.mean(axis=0).round(1).tolist() if len(pts) else None,
            "heatmap": heatmap(pts, fps),
            "passes": sum(1 for x in all_passes if x["from"] == ent and x["success"]),
            "passes_failed": sum(1 for x in all_passes if x["from"] == ent and not x["success"]),
            "possessions": touches.get(ent, 0),
        })
    rows.sort(key=lambda r: (r["player_id"] is None, r["team"], -r["minutes"]))

    total_poss = sum(poss_time)
    team_rows = []
    for team in (0, 1):
        tp = [x for x in all_passes if x["team"] == team]
        ok = sum(1 for x in tp if x["success"])
        team_rows.append({
            "team": team,
            "name": match[f"team{team}_name"] if match else f"Team {team}",
            "possession_pct": round(100 * poss_time[team] / total_poss, 1) if total_poss else None,
            "passes": ok,
            "pass_accuracy_pct": round(100 * ok / len(tp), 1) if tp else None,
        })

    network = defaultdict(int)
    for x in all_passes:
        if x["success"]:
            network[(x["from"], x["to"])] += 1
    return {
        "clips": status,
        "players": rows,
        "teams": team_rows,
        "events": sorted(events, key=lambda e: (e["clip_id"], e["t"])),
        "pass_network": [{"from": a, "to": b, "count": n} for (a, b), n in network.items()],
    }


def clip_positions(store: Store, clip_id: int, t0: float, t1: float) -> dict:
    """Posities per frame voor de 2D-minimap naast de video."""
    d = load_clip(store, clip_id)
    if d is None:
        return {"calibrated": False, "frames": []}
    lo, hi = np.searchsorted(d.t, t0), np.searchsorted(d.t, t1, side="right")
    frames = []
    sel = (d.idx >= lo) & (d.idx < hi)
    rows_by_idx: dict[int, list[int]] = defaultdict(list)
    for r in np.flatnonzero(sel):
        rows_by_idx[int(d.idx[r])].append(int(r))
    ball = {int(i): xy for i, xy in zip(d.ball_idx, d.ball_xy)}
    for i in range(lo, hi):
        ps = []
        for r in rows_by_idx.get(i, []):
            tid = int(d.track[r])
            if tid not in d.valid_tracks or np.isnan(d.xy[r]).any():
                continue
            ps.append([round(float(d.xy[r, 0]), 2), round(float(d.xy[r, 1]), 2), d.team.get(tid, -1),
                       d.entity.get(tid), tid])
        b = ball.get(i)
        frames.append({"t": round(float(d.t[i]), 3), "p": ps,
                       "b": None if b is None or np.isnan(b).any() else [round(float(b[0]), 2), round(float(b[1]), 2)]})
    return {"calibrated": d.calibrated, "frames": frames}


def _to_image(d: ClipData, i: int, world: np.ndarray) -> np.ndarray:
    """Veldpunten -> beeldpunten op geanalyseerd frame i (NaN als achter de camera)."""
    img = np.zeros_like(world)
    for H, w in d.camera.homographies(i):
        hom = np.hstack([world, np.ones((len(world), 1))]) @ np.linalg.inv(H).T
        hom[hom[:, 2] <= 0, 2] = np.nan
        img += w * hom[:, :2] / hom[:, 2:3]
    return img


def image_landmarks(to_image, W: int, Hh: int, line_points: bool = True, margin: float = 0.02) -> list[dict]:
    """Veldpunten en -lijnen die in beeld liggen, als kalibratiepunten (beeld <-> veld).

    to_image: functie veldpunten (n, 2) -> beeldpunten (n, 2), NaN als achter de camera.
    Lijnen komen terug als twee punten-op-de-lijn, zodat ze met één klik over te nemen zijn
    (handig bij beelden vanaf de zijlijn)."""
    inside = lambda x, y: np.isfinite(x) & np.isfinite(y) & (x >= -margin * W) & (x <= (1 + margin) * W) \
        & (y >= -margin * Hh) & (y <= (1 + margin) * Hh)  # noqa: E731
    names = list(pitch.LANDMARKS)
    world = np.array([pitch.LANDMARKS[n] for n in names], dtype=np.float64)
    img = to_image(world)
    out = []
    for n, (x, y), wp in zip(names, img, world):
        if inside(x, y):
            out.append({"name": n, "img": [round(float(x), 1), round(float(y), 1)], "pitch": wp.tolist()})
    if not line_points:
        return out
    for n, (a, b) in pitch.LINES.items():
        s = np.linspace(0, 1, 41)[:, None]
        pts = to_image((1 - s) * np.array(a) + s * np.array(b))
        ok = np.flatnonzero(inside(pts[:, 0], pts[:, 1]))
        if len(ok) < 4:
            continue
        for k in (ok[len(ok) // 4], ok[(3 * len(ok)) // 4]):
            out.append({"name": n, "img": [round(float(pts[k, 0]), 1), round(float(pts[k, 1]), 1)],
                        "line": [list(a), list(b)]})
    return out


def h_to_image(H: np.ndarray):
    """to_image-functie voor een homografie veld -> beeld."""
    def f(world):
        hom = np.hstack([world, np.ones((len(world), 1))]) @ H.T
        hom[hom[:, 2] <= 0, 2] = np.nan
        return hom[:, :2] / hom[:, 2:3]
    return f


def predict_landmarks(store: Store, clip_id: int, t: float) -> list[dict]:
    """Waar de veldpunten en -lijnen volgens het cameramodel in beeld liggen op tijdstip t."""
    d = load_clip(store, clip_id)
    if d is None or not d.calibrated:
        return []
    i = int(np.argmin(np.abs(d.t - t)))
    return image_landmarks(lambda world: _to_image(d, i, world), d.clip["width"], d.clip["height"])


def warp_points(store: Store, clip_id: int, t_from: float, t_to: float, pts: np.ndarray) -> np.ndarray | None:
    """Beeldpunten op tijdstip t_from verplaatsen naar waar dezelfde plek op t_to in beeld is,
    via de gemeten camerabeweging. None als de video nog niet geanalyseerd is."""
    d = load_clip(store, clip_id)
    if d is None or len(d.t) == 0:
        return None
    i, j = int(np.argmin(np.abs(d.t - t_from))), int(np.argmin(np.abs(d.t - t_to)))
    A = d.camera.A
    M = np.linalg.inv(A[j]) @ A[i]  # beeld(i) -> referentie -> beeld(j)
    return apply_h(M, np.asarray(pts, float).reshape(-1, 2))


def clip_boxes(store: Store, clip_id: int, t0: float, t1: float) -> list[dict]:
    """Boxen in beeldcoördinaten (voor aanklikken van spelers in de video)."""
    d = load_clip(store, clip_id)
    if d is None:
        return []
    lo, hi = np.searchsorted(d.t, t0), np.searchsorted(d.t, t1, side="right")
    out = []
    for r in np.flatnonzero((d.idx >= lo) & (d.idx < hi)):
        tid = int(d.track[r])
        out.append({"t": round(float(d.t[d.idx[r]]), 3), "track": tid, "box": d.boxes[r].round(1).tolist(),
                    "entity": d.entity.get(tid), "team": d.team.get(tid, -1),
                    "valid": tid in d.valid_tracks})
    return out
