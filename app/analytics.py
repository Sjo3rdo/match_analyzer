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
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import config, pitch
from .calibration import CameraModel, Keyframe, apply_h, camera_prior, fit_keyframe
from .storage import Store, clip_dir
from .teams import TEAM_OTHER, TEAM_SPECTATOR
from .tracking import still_fraction

# Twee lagen cache, zoals een kast met twee planken:
# - onder de zware spullen (detecties ingelezen en naar het veld geprojecteerd). Die veranderen
#   alleen als de analyse, de kalibratie, de camerapositie of de veldmaten veranderen;
# - boven de lichte spullen (welke track bij welke speler en welk team hoort). Die veranderen
#   bij elke klik in "Spelers", maar zijn in een oogwenk opnieuw gemaakt.
_lock = threading.RLock()
_geo: dict[int, tuple[tuple[int, int], "ClipGeo"]] = {}
_geo_epoch = 0
_clip_epoch: dict[int, int] = defaultdict(int)
_meta_epoch = 0
_data: OrderedDict = OrderedDict()
_stats: OrderedDict = OrderedDict()


def invalidate(geometry: bool = True, clip_id: int | None = None) -> None:
    """Aanroepen na elke wijziging.

    geometry=True: analyse, kalibratie, camerapositie of veldmaten veranderd (zwaar).
    geometry=False: alleen koppelingen, teams, namen of de speelrichting (licht).
    clip_id: alleen die video (anders alles)."""
    global _geo_epoch, _meta_epoch
    with _lock:
        _meta_epoch += 1
        if geometry:
            if clip_id is None:
                _geo_epoch += 1
                _geo.clear()
            else:
                _clip_epoch[clip_id] += 1
                _geo.pop(clip_id, None)
        _data.clear()
        _stats.clear()


def _epoch(clip_id: int) -> tuple[int, int]:
    return (_geo_epoch, _clip_epoch[clip_id])


@dataclass
class ClipData:
    clip: dict
    t: np.ndarray  # tijd (s) per geanalyseerd frame
    fps: float  # geanalyseerde frames per seconde
    calibrated: bool
    # detecties (rijen, gesorteerd op frame)
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
    calib_suspect: bool = False  # de kalibratie zet bijna iedereen buiten het veld
    groups: dict[int, np.ndarray] | None = None  # track -> rijnummers (op volgorde van tijd)
    geom: pitch.Geometry = pitch.DEFAULT
    stationary: dict[int, float] = field(default_factory=dict)  # track -> deel van de tijd stil

    def rows_of(self, tid: int) -> np.ndarray:
        if self.groups is None:
            self.groups = group_rows(self.track)
        return self.groups.get(int(tid), np.zeros(0, int))


def entity_key(clip_id: int, track_id: int, player_id: int | None) -> str:
    return f"p{player_id}" if player_id else f"c{clip_id}t{track_id}"


def group_rows(track: np.ndarray) -> dict[int, np.ndarray]:
    """Rijnummers per track in één keer (in plaats van per track de hele tabel te doorzoeken)."""
    if len(track) == 0:
        return {}
    order = np.argsort(track, kind="stable")  # stabiel: binnen een track blijft de tijdsvolgorde
    cuts = np.flatnonzero(np.diff(track[order])) + 1
    return {int(track[g[0]]): g for g in np.split(order, cuts)}


def flipped(clip: dict) -> bool:
    """Speelrichting omdraaien voor de statistieken? Standaard bij de 2e helft."""
    f = clip.get("flip")
    return bool(f) if f is not None else clip.get("period") == 2


def _load_geo(store: Store, clip_id: int) -> ClipData | None:
    """Het zware deel: detecties inlezen, naar het veld projecteren, toeschouwers herkennen."""
    with _lock:
        hit = _geo.get(clip_id)
        if hit is not None and hit[0] == _epoch(clip_id):
            return hit[1]
        epoch = _epoch(clip_id)
    clip = store.one("SELECT * FROM clips WHERE id = ?", (clip_id,))
    if clip is None or clip["status"] != "klaar":
        return None
    match = store.one("SELECT * FROM matches WHERE id = ?", (clip["match_id"],))
    geom = pitch.of_match(match)
    frames = store.rows("SELECT t FROM frames WHERE clip_id = ? ORDER BY idx", (clip_id,))
    t = np.array([f[0] for f in frames], dtype=np.float64)
    if len(t) == 0:
        return None
    fps = (len(t) - 1) / (t[-1] - t[0]) if len(t) > 1 and t[-1] > t[0] else config.TARGET_FPS

    motion_path = clip_dir(clip_id) / "motion.npy"
    inter = np.load(motion_path) if Path(motion_path).exists() else np.tile(np.eye(3), (len(t), 1, 1))
    if len(inter) < len(t):
        inter = np.concatenate([inter, np.tile(np.eye(3), (len(t) - len(inter), 1, 1))])
    kfs = []
    prior = camera_prior(clip)
    for kf in store.keyframes(clip_id):
        try:
            K, _ = fit_keyframe(kf, camera=prior)
        except ValueError:
            continue
        kfs.append(Keyframe(int(np.argmin(np.abs(t - kf["t"]))), K))
    cam = CameraModel(inter[:len(t)], kfs)

    rows = store.rows("SELECT idx, track_id, x1, y1, x2, y2 FROM detections WHERE clip_id = ? ORDER BY idx",
                      (clip_id,))
    arr = np.array(rows, dtype=np.float64).reshape(-1, 6)
    idx, track, boxes = arr[:, 0].astype(int), arr[:, 1].astype(int), arr[:, 2:6]
    feet = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]], axis=1)
    xy = cam.project_rows(idx, feet)

    ball_idx, ball_img, manual, no_ball = _ball_rows(store, clip_id, t)
    ball_xy = clean_ball(t[ball_idx] if len(ball_idx) else np.zeros(0), cam.project_rows(ball_idx, ball_img),
                         geom, keep=manual) if cam.calibrated else np.full((len(ball_idx), 2), np.nan)
    ball_idx, ball_xy, ball_img = fill_ball_gaps(t, ball_idx, ball_xy, ball_img, blocked=no_ball)

    d = ClipData(clip, t, fps, cam.calibrated, idx, track, boxes, xy, ball_idx, ball_xy, ball_img,
                 camera=cam, groups=group_rows(track), geom=geom)
    d.stationary = stationary_tracks(d)
    d.valid_tracks = _valid_tracks(d)
    with _lock:
        if _epoch(clip_id) == epoch:
            _geo[clip_id] = (epoch, d)
    return d


def load_clip(store: Store, clip_id: int) -> ClipData | None:
    """Alle gegevens van een geanalyseerde video, inclusief koppelingen aan spelers en teams."""
    geo = _load_geo(store, clip_id)
    if geo is None:
        return None
    key = (clip_id, _epoch(clip_id), _meta_epoch)
    with _lock:
        hit = _data.get(key)
        if hit is not None:
            return hit
    clip = store.one("SELECT * FROM clips WHERE id = ?", (clip_id,)) or geo.clip
    d = ClipData(clip, geo.t, geo.fps, geo.calibrated, geo.idx, geo.track, geo.boxes, geo.xy,
                 geo.ball_idx, geo.ball_xy, geo.ball_img, camera=geo.camera, valid_tracks=geo.valid_tracks,
                 groups=geo.groups, geom=geo.geom, stationary=geo.stationary, calib_suspect=geo.calib_suspect)
    players = {p["id"]: p for p in store.all("SELECT * FROM players WHERE match_id = ?", (clip["match_id"],))}
    for tr in store.all("SELECT * FROM tracks WHERE clip_id = ?", (clip_id,)):
        tid = tr["track_id"]
        d.tracks[tid] = tr
        pid = tr["player_id"]
        d.entity[tid] = entity_key(clip_id, tid, pid)
        d.team[tid] = players[pid]["team"] if pid in players else tr["team"]
    # door de gebruiker aangewezen toeschouwers tellen nergens mee
    spectators = {tid for tid, tr in d.tracks.items() if tr["team"] == TEAM_SPECTATOR and tr["player_id"] is None}
    if spectators:
        d.valid_tracks = set(geo.valid_tracks) - spectators
    with _lock:
        _data[key] = d
        while len(_data) > 16:
            _data.popitem(last=False)
    return d


def _project_rows(cam: CameraModel, idx: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return cam.project_rows(idx, pts)


def _ball_rows(store: Store, clip_id: int, t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, set[int]]:
    """Balposities per beeld (beeldpixels): wat de analyse vond, met daaroverheen wat de gebruiker
    zelf aanwees ("hier is de bal" of "hier is geen bal"). Ook: welke rijen met de hand zijn gezet,
    en in welke beelden volgens de gebruiker geen bal is."""
    found = {int(i): (x, y) for i, x, y in store.rows("SELECT idx, x, y FROM ball WHERE clip_id = ?", (clip_id,))}
    manual: set[int] = set()
    no_ball: set[int] = set()
    if len(t):
        for tm, x, y in store.rows("SELECT t, x, y FROM ball_manual WHERE clip_id = ?", (clip_id,)):
            i = int(np.argmin(np.abs(t - tm)))
            if abs(t[i] - tm) > 0.08:
                continue
            if x is None or y is None:
                found.pop(i, None)
                manual.discard(i)
                no_ball.add(i)
            else:
                found[i] = (x, y)
                manual.add(i)
    idx = np.array(sorted(found), dtype=int)
    img = np.array([found[i] for i in idx], dtype=np.float64).reshape(-1, 2)
    return idx, img, np.isin(idx, list(manual)), no_ball


def fill_ball_gaps(t: np.ndarray, idx: np.ndarray, xy: np.ndarray, img: np.ndarray, max_gap: float = 1.0,
                   max_speed: float = 35.0, blocked: set[int] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Korte gaten in het balspoor opvullen (de bal achter een speler, of even niet gezien):
    rechte lijn tussen de laatste plek ervoor en de eerste erna, als dat met een haalbare snelheid kan."""
    ok = np.flatnonzero(~np.isnan(xy).any(axis=1)) if len(xy) else np.zeros(0, int)
    if len(ok) < 2:
        return idx, xy, img
    add_i, add_xy, add_img = [], [], []
    for a, b in zip(ok[:-1], ok[1:]):
        i0, i1 = int(idx[a]), int(idx[b])
        if i1 - i0 < 2:
            continue
        dt = t[i1] - t[i0]
        if dt > max_gap or np.linalg.norm(xy[b] - xy[a]) / max(dt, 1e-3) > max_speed:
            continue
        have = set(idx[a:b + 1].tolist()) | (blocked or set())  # (ook: waar de gebruiker zei: geen bal)
        for i in range(i0 + 1, i1):
            if i in have:
                continue
            f = (t[i] - t[i0]) / max(dt, 1e-6)
            add_i.append(i)
            add_xy.append(xy[a] + f * (xy[b] - xy[a]))
            add_img.append(img[a] + f * (img[b] - img[a]))
    if not add_i:
        return idx, xy, img
    all_i = np.concatenate([idx, add_i])
    order = np.argsort(all_i, kind="stable")
    return (all_i[order], np.vstack([xy, add_xy])[order], np.vstack([img, add_img])[order])


def clean_ball(t: np.ndarray, xy: np.ndarray, geom: pitch.Geometry, margin: float = 2.0,
               max_speed: float = 40.0, keep: np.ndarray | None = None) -> np.ndarray:
    """Onmogelijke balposities weggooien (NaN):
    - buiten het veld: een reservebal of pion langs de lijn, of een hoge bal (de projectie gaat
      ervan uit dat de bal op de grond ligt, dus een bal in de lucht komt ver buiten het veld uit);
    - losse uitschieters: een 'bal' die ineens 30 m verderop ligt en direct weer terug is."""
    xy = np.array(xy, dtype=np.float64, copy=True)
    orig = xy.copy()
    if len(xy) == 0:
        return xy
    x, y = xy[:, 0], xy[:, 1]
    off = ~((x >= -margin) & (x <= geom.length + margin) & (y >= -margin) & (y <= geom.width + margin))
    xy[off] = np.nan
    ok = np.flatnonzero(~np.isnan(xy).any(axis=1))
    for k in range(1, len(ok) - 1):
        a, b, c = ok[k - 1], ok[k], ok[k + 1]
        dt1, dt2 = max(1e-3, t[b] - t[a]), max(1e-3, t[c] - t[b])
        if dt1 > 1.0 or dt2 > 1.0:
            continue
        v1 = np.linalg.norm(xy[b] - xy[a]) / dt1
        v2 = np.linalg.norm(xy[c] - xy[b]) / dt2
        if v1 > max_speed and v2 > max_speed and np.linalg.norm(xy[c] - xy[a]) / (dt1 + dt2) < max_speed:
            xy[b] = np.nan
    if keep is not None and len(keep):  # met de hand aangewezen: altijd houden
        xy[keep] = orig[keep]
    return xy


def stationary_tracks(d: ClipData) -> dict[int, float]:
    """Per track welk deel van de tijd hij stilstaat t.o.v. de achtergrond (zie tracking.still_fraction)."""
    if d.camera is None or len(d.camera.A) == 0 or len(d.track) == 0:
        return {}
    feet = np.stack([(d.boxes[:, 0] + d.boxes[:, 2]) / 2, d.boxes[:, 3]], 1)
    return still_fraction(d.t, d.idx, d.track, feet, d.boxes[:, 3] - d.boxes[:, 1], d.camera.A)


def _on_pitch_frac(d: ClipData, rows: np.ndarray, margin: float = 1.5) -> float:
    x, y = d.xy[rows, 0], d.xy[rows, 1]
    ok = (x >= -margin) & (x <= d.geom.length + margin) & (y >= -margin) & (y <= d.geom.width + margin)
    return float(ok.mean()) if len(ok) else 0.0


def _valid_tracks(d: ClipData, min_frames: int = 10, min_on_pitch: float = 0.6) -> set[int]:
    """Tracks die echt spelers op het veld zijn (geen toeschouwers, wissels of mensen vlak voor de camera).

    - Kort in beeld (< min_frames): weg.
    - Voeten meestal onder de rand van het beeld: iemand vlak voor de camera, niet op het veld.
    - Staat stil ten opzichte van de achtergrond (stationary_tracks): een toeschouwer. Na
      kalibratie geldt dat alleen langs de zijlijnen of buiten het veld, zodat een keeper die even
      stilstaat blijft meetellen.
    - Na kalibratie: meestal buiten het veld."""
    valid: set[int] = set()
    if not len(d.track):
        return valid
    H_img = float(d.clip.get("height") or 0)
    still = d.stationary if d.stationary else stationary_tracks(d)
    # Zonder kalibratie weten we niet waar het veld is: wie meestal stilstaat is waarschijnlijk publiek.
    # Met kalibratie geldt het alleen langs de zijlijn, en strenger (een buitenspeler die daar
    # wacht, moet blijven meetellen).
    W = d.geom.width
    groups = d.groups if d.groups is not None else group_rows(d.track)
    calibrated = d.calibrated
    if calibrated:
        # Klopt de kalibratie wel? Als bijna iedereen die rondloopt volgens de kalibratie buiten het
        # veld staat, ligt dat aan de kalibratie, niet aan de spelers. Dan negeren we de kalibratie
        # hier (anders verdwijnt iedereen) en melden we het.
        moving = [rows for tid, rows in groups.items() if len(rows) >= 2 * min_frames and still.get(tid, 0.0) < 0.5]
        if len(moving) >= 5:
            inside = [_on_pitch_frac(d, rows) >= min_on_pitch for rows in moving]
            if np.mean(inside) < 0.35:
                calibrated = False
                d.calib_suspect = True
    for tid, rows in groups.items():
        if len(rows) < min_frames:
            continue
        if H_img and (d.boxes[rows, 3] >= 0.99 * H_img).mean() > 0.5:  # kaders aan de rand eindigen net erboven
            continue
        frac = still.get(tid, 0.0)
        st = frac >= (0.85 if calibrated else 0.6)
        if calibrated:
            x, y = d.xy[rows, 0], d.xy[rows, 1]
            if _on_pitch_frac(d, rows) < min_on_pitch:
                continue
            my = float(np.nanmedian(y)) if np.isfinite(y).any() else 0.0
            if st and (my < 4.0 or my > W - 4.0):
                continue
        elif st:
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


def heatmap(points: np.ndarray, fps: float | np.ndarray, geom: pitch.Geometry = pitch.DEFAULT) -> list[list[float]]:
    """Seconden per veldcel (rijen = y, kolommen = x).

    fps: geanalyseerde frames per seconde, of per punt (video's kunnen verschillen: bij 25 fps
    analyseren we 12,5 beelden per seconde, bij 30 fps 10)."""
    nx, ny = config.HEATMAP_BINS
    if len(points) == 0:
        return np.zeros((ny, nx)).tolist()
    w = 1.0 / np.broadcast_to(np.asarray(fps, dtype=np.float64), (len(points),))
    h, _, _ = np.histogram2d(points[:, 1], points[:, 0], bins=[ny, nx],
                             range=[[0, geom.width], [0, geom.length]], weights=w)
    return h.round(2).tolist()


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
    vrows = np.flatnonzero(np.isin(d.track, np.fromiter(d.valid_tracks, int, len(d.valid_tracks)))
                           & ~np.isnan(d.xy).any(axis=1))
    vrows = vrows[np.argsort(d.idx[vrows], kind="stable")]
    vidx = d.idx[vrows]
    owners: list[tuple[int, int | None]] = []  # (frame, track)
    for bi, bxy in zip(d.ball_idx, d.ball_xy):
        lo, hi = np.searchsorted(vidx, bi), np.searchsorted(vidx, bi, side="right")
        rows = vrows[lo:hi]
        if not len(rows) or np.isnan(bxy).any():
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
    """Statistieken van de hele wedstrijd (alle video's). Bewaard tot er iets verandert."""
    key = (match_id, _geo_epoch, _meta_epoch)
    with _lock:
        if key in _stats:
            return _stats[key]
    out = _match_stats(store, match_id)
    with _lock:
        _stats[key] = out
        while len(_stats) > 4:
            _stats.popitem(last=False)
    return out


def _match_stats(store: Store, match_id: int) -> dict:
    match = store.one("SELECT * FROM matches WHERE id = ?", (match_id,))
    geom = pitch.of_match(match)
    players = {f"p{p['id']}": p for p in store.all("SELECT * FROM players WHERE match_id = ?", (match_id,))}
    clips = store.all("SELECT * FROM clips WHERE match_id = ? ORDER BY order_idx, id", (match_id,))

    seg_by_entity: dict[str, list] = defaultdict(list)
    pts_by_entity: dict[str, list] = defaultdict(list)
    fps_by_entity: dict[str, list] = defaultdict(list)
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
        flip = flipped(clip)  # 2e helft: zelfde speelrichting als de 1e (alleen voor posities)
        for tid in sorted(d.valid_tracks):
            ent = d.entity[tid] if tid in d.entity else entity_key(clip["id"], tid, None)
            if ent not in players and d.team.get(tid, -1) == TEAM_OTHER:
                continue
            rows = d.rows_of(tid)
            ts, xy = d.t[d.idx[rows]], d.xy[rows]
            seg_by_entity[ent].extend(smooth_series(ts, xy))
            on = ~np.isnan(xy).any(axis=1)
            pos = xy[on]
            if flip:
                pos = np.stack([geom.length - pos[:, 0], geom.width - pos[:, 1]], axis=1)
            pts_by_entity[ent].append(pos)
            fps_by_entity[ent].append(np.full(len(pos), d.fps))
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

    passes_ok, passes_bad = defaultdict(int), defaultdict(int)
    for x in all_passes:
        (passes_ok if x["success"] else passes_bad)[x["from"]] += 1
    rows = []
    for ent, segs in seg_by_entity.items():
        mv = movement_stats(segs)
        pts = np.concatenate(pts_by_entity[ent]) if pts_by_entity[ent] else np.zeros((0, 2))
        fps = np.concatenate(fps_by_entity[ent]) if fps_by_entity[ent] else np.zeros(0)
        p = players.get(ent)
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
            "heatmap": heatmap(pts, fps, geom),
            "passes": passes_ok.get(ent, 0),
            "passes_failed": passes_bad.get(ent, 0),
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
        "pitch": {"length": geom.length, "width": geom.width},
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
    r0, r1 = np.searchsorted(d.idx, lo), np.searchsorted(d.idx, hi)  # rijen staan op volgorde van frame
    rows_by_idx: dict[int, list[int]] = defaultdict(list)
    for r in range(r0, r1):
        rows_by_idx[int(d.idx[r])].append(r)
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


def image_landmarks(to_image, W: int, Hh: int, line_points: bool = True, margin: float = 0.02,
                    geom: pitch.Geometry = pitch.DEFAULT) -> list[dict]:
    """Veldpunten en -lijnen die in beeld liggen, als kalibratiepunten (beeld <-> veld).

    to_image: functie veldpunten (n, 2) -> beeldpunten (n, 2), NaN als achter de camera.
    Lijnen komen terug als twee punten-op-de-lijn, zodat ze met één klik over te nemen zijn
    (handig bij beelden vanaf de zijlijn)."""
    inside = lambda x, y: np.isfinite(x) & np.isfinite(y) & (x >= -margin * W) & (x <= (1 + margin) * W) \
        & (y >= -margin * Hh) & (y <= (1 + margin) * Hh)  # noqa: E731
    names = list(geom.landmarks)
    world = np.array([geom.landmarks[n] for n in names], dtype=np.float64)
    img = to_image(world)
    out = []
    for n, (x, y), wp in zip(names, img, world):
        if inside(x, y):
            out.append({"name": n, "img": [round(float(x), 1), round(float(y), 1)], "pitch": wp.tolist()})
    if not line_points:
        return out
    for n, (a, b) in geom.lines.items():
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
    return image_landmarks(lambda world: _to_image(d, i, world), d.clip["width"], d.clip["height"], geom=d.geom)


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
    for r in range(np.searchsorted(d.idx, lo), np.searchsorted(d.idx, hi)):
        tid = int(d.track[r])
        out.append({"t": round(float(d.t[d.idx[r]]), 3), "track": tid, "box": d.boxes[r].round(1).tolist(),
                    "entity": d.entity.get(tid), "team": d.team.get(tid, -1),
                    "valid": tid in d.valid_tracks})
    return out


# --- koppel-assistent ----------------------------------------------------------------------
#
# Metafoor: een speler laat een spoor achter dat steeds even onderbroken wordt (botsing, camera
# draait weg). Heb je één stuk spoor herkend, dan zoeken we stukken die er logisch op aansluiten:
# ze zijn niet tegelijk in beeld, beginnen ongeveer waar het vorige stuk ophield (rekening houdend
# met hoe ver iemand in die tijd kan rennen) en hebben hetzelfde shirt. Een stuk dat beter aansluit
# op een teamgenoot die al gekoppeld is, zakt in de lijst.

def _track_summary(d: ClipData, tid: int) -> dict | None:
    rows = d.rows_of(tid)
    if len(rows) == 0:
        return None
    tt = d.t[d.idx[rows]]
    feet = np.stack([(d.boxes[rows, 0] + d.boxes[rows, 2]) / 2, d.boxes[rows, 3]], 1)
    xy = d.xy[rows]
    ok = ~np.isnan(xy).any(axis=1)
    return {"tid": tid, "t0": float(tt[0]), "t1": float(tt[-1]), "i0": int(d.idx[rows[0]]), "i1": int(d.idx[rows[-1]]),
            "f0": feet[0], "f1": feet[-1], "h0": float(np.median(d.boxes[rows[:5], 3] - d.boxes[rows[:5], 1])),
            "h1": float(np.median(d.boxes[rows[-5:], 3] - d.boxes[rows[-5:], 1])),
            "p0": xy[ok][0] if ok.any() else None, "p1": xy[ok][-1] if ok.any() else None, "n": len(rows)}


def _continuity(d: ClipData, a: dict, b: dict) -> tuple[float, float, float | None]:
    """Hoe logisch b op a volgt (a eindigt vóór b begint): (kans 0..1, gat in s, afstand in m of None)."""
    gap = b["t0"] - a["t1"]
    if gap < -0.3:
        return 0.0, gap, None
    gap = max(gap, 0.0)
    if d.calibrated and a["p1"] is not None and b["p0"] is not None:
        dist = float(np.linalg.norm(b["p0"] - a["p1"]))
        reach = 2.0 + 6.0 * gap  # wat een speler in die tijd ongeveer kan afleggen (plus meetfout)
        p = float(np.exp(-0.5 * (max(0.0, dist - 0.5 * reach) / (0.5 * reach)) ** 2))
    else:
        # zonder kalibratie: in het beeld, met de camerabeweging eruit, in lichaamslengtes
        A = d.camera.A
        M = np.linalg.inv(A[b["i0"]]) @ A[a["i1"]]
        q = M @ np.array([a["f1"][0], a["f1"][1], 1.0])
        q = q[:2] / q[2] if abs(q[2]) > 1e-9 else np.array([np.inf, np.inf])
        h = max(1.0, 0.5 * (a["h1"] + b["h0"]))
        dist = None
        moved = float(np.linalg.norm(q - b["f0"])) / h
        reach = 1.0 + 3.5 * gap
        p = float(np.exp(-0.5 * (max(0.0, moved - 0.5 * reach) / (0.5 * reach)) ** 2))
        ratio = b["h0"] / max(1.0, a["h1"])
        if gap < 3 and not 0.6 < ratio < 1.6:  # ineens veel groter of kleiner: iemand anders
            p *= 0.3
    return p * float(np.exp(-gap / 40.0)), gap, dist


def suggest_tracks(store: Store, player_id: int, clip_id: int | None = None, limit: int = 8) -> list[dict]:
    """Tracks die waarschijnlijk ook bij deze speler horen, best passend eerst."""
    from .teams import color_distance, hex_to_lab

    from . import learning

    p = store.one("SELECT * FROM players WHERE id = ?", (player_id,))
    if p is None:
        return []
    profile = learning.profile_of(store, p)
    center = learning.team_centers(store, p["match_id"]).get(p["team"]) if profile is not None else None
    linked = store.all("SELECT t.* FROM tracks t JOIN clips c ON c.id = t.clip_id WHERE c.match_id = ? "
                       "AND t.player_id = ?", (p["match_id"], player_id))
    cols = [(hex_to_lab(r["color"]), r["n_frames"] or 1) for r in linked]
    cols = [(c, w) for c, w in cols if c is not None]
    my_color = np.average([c for c, _ in cols], axis=0, weights=[w for _, w in cols]) if cols else None
    if clip_id is not None:
        clip_ids = [clip_id]
    else:
        clip_ids = sorted({r["clip_id"] for r in linked}) or [
            r["id"] for r in store.all("SELECT id FROM clips WHERE match_id = ? AND status = 'klaar'", (p["match_id"],))]
    out = []
    for cid in clip_ids:
        d = load_clip(store, cid)
        if d is None:
            continue
        mine = [s for s in (_track_summary(d, r["track_id"]) for r in linked if r["clip_id"] == cid) if s]
        others: dict[int, list[dict]] = defaultdict(list)  # al gekoppelde teamgenoten
        for tid, tr in d.tracks.items():
            if tr["player_id"] and tr["player_id"] != player_id:
                s = _track_summary(d, tid)
                if s:
                    others[tr["player_id"]].append(s)
        for tid, tr in d.tracks.items():
            if tr["player_id"] or tid not in d.valid_tracks or (tr["n_frames"] or 0) < 5:
                continue
            team = d.team.get(tid, -1)
            if team not in (p["team"], -1):
                continue
            c = _track_summary(d, tid)
            if c is None:
                continue
            # tegelijk in beeld met een stuk van deze speler: dat kan hij niet zijn
            if any(min(c["t1"], m["t1"]) - max(c["t0"], m["t0"]) > 0.3 for m in mine):
                continue

            def cont(segs):
                best = (0.0, None, None)
                for m in segs:
                    for a, b in ((m, c), (c, m)):
                        if a["t1"] <= b["t0"] + 0.3:
                            pr, gap, dist = _continuity(d, a, b)
                            if pr > best[0]:
                                best = (pr, gap, dist)
                return best
            pc, gap, dist = cont(mine) if mine else (0.3, None, None)
            rival = max((cont(segs)[0] for segs in others.values()), default=0.0)
            score = 0.3 + 0.7 * pc
            if rival > pc + 0.2:
                score *= 0.5
            col = hex_to_lab(tr["color"])
            de = color_distance(col, my_color) if col is not None and my_color is not None else None
            if de is not None:
                score *= 0.15 + 0.85 * float(np.exp(-0.5 * (de / 12.0) ** 2))
            look = None
            if profile is not None:
                emb = learning.track_embedding(store, cid, tid)
                if emb is not None:
                    look = learning.similarity(emb, profile, center)
                    score *= 0.5 + 1.0 * float(np.clip((look + 0.1) / 0.6, 0, 1))  # lijkt hij op zijn profiel?
            if tr.get("jersey_guess") and p.get("number") and (tr.get("jersey_conf") or 0) >= 0.4:
                score *= 1.4 if str(tr["jersey_guess"]) == str(p["number"]).strip() else 0.3
            reason = []
            if gap is not None:
                reason.append(f"sluit aan {gap:.1f} s {'later' if c['t0'] >= max(m['t1'] for m in mine) - 0.3 else 'eerder'}"
                              .replace(".", ","))
                if dist is not None:
                    reason.append(f"{dist:.0f} m verderop")
            if de is not None:
                reason.append("zelfde shirt" if de < 8 else "shirt lijkt erop" if de < 16 else "ander shirt?")
            if rival > pc + 0.2:
                reason.append("past ook bij een teamgenoot")
            if look is not None:
                reason.append("lijkt op zijn profiel" if look >= 0.35 else "lijkt niet op zijn profiel" if look < 0.05 else "")
                reason = [r for r in reason if r]
            out.append({"clip_id": cid, "track_id": tid, "t_start": round(c["t0"], 2), "t_end": round(c["t1"], 2),
                        "n_frames": c["n"], "score": round(min(1.0, score), 3), "reason": ", ".join(reason)})
    out.sort(key=lambda r: -r["score"])
    return out[:limit]


# --- toeschouwers aanwijzen -----------------------------------------------------------------

def _anchored_feet(d: ClipData, rows: np.ndarray) -> np.ndarray | None:
    """Voetpunten omgerekend naar het eerste beeld van de video (de camerabeweging eruit): wie
    stilstaat langs de lijn, staat dan steeds op dezelfde plek."""
    if d.camera is None or len(d.camera.A) == 0:
        return None
    A = d.camera.A[d.idx[rows]]
    feet = np.stack([(d.boxes[rows, 0] + d.boxes[rows, 2]) / 2, d.boxes[rows, 3], np.ones(len(rows))], 1)
    p = np.einsum("nij,nj->ni", A, feet)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = p[:, :2] / p[:, 2:3]
    return p


def similar_spectators(store: Store, clip_id: int, track_id: int, limit: int = 40) -> list[dict]:
    """Tracks in deze wedstrijd die waarschijnlijk dezelfde toeschouwer zijn (of vlak naast hem staan):
    zelfde soort kleding en op dezelfde plek (in deze video: zelfde plek ten opzichte van de
    achtergrond; in andere video's: zelfde plek op het veld, als beide gekalibreerd zijn)."""
    from .teams import color_distance, hex_to_lab

    d = load_clip(store, clip_id)
    if d is None or track_id not in d.tracks:
        return []
    ref_rows = d.rows_of(track_id)
    ref_col = hex_to_lab(d.tracks[track_id]["color"])
    if len(ref_rows) == 0 or ref_col is None:
        return []
    ref_h = float(np.median(d.boxes[ref_rows, 3] - d.boxes[ref_rows, 1]))
    ref_img = _anchored_feet(d, ref_rows)
    ref_img = np.nanmedian(ref_img, axis=0) if ref_img is not None else None
    ref_xy = np.nanmedian(d.xy[ref_rows], axis=0) if d.calibrated else None
    clip = store.one("SELECT match_id FROM clips WHERE id = ?", (clip_id,))
    out = []
    for c in store.all("SELECT id FROM clips WHERE match_id = ? AND status = 'klaar'", (clip["match_id"],)):
        cd = d if c["id"] == clip_id else load_clip(store, c["id"])
        if cd is None:
            continue
        for tid, tr in cd.tracks.items():
            if (c["id"], tid) == (clip_id, track_id) or tr["player_id"] or tr["team"] == TEAM_SPECTATOR:
                continue
            if tid not in cd.valid_tracks:
                continue  # telt toch al niet mee
            col = hex_to_lab(tr["color"])
            if col is None or color_distance(col, ref_col) > 14:
                continue
            rows = cd.rows_of(tid)
            if len(rows) == 0:
                continue
            dist = None
            if cd is d and ref_img is not None:
                p = _anchored_feet(cd, rows)
                if p is not None and np.isfinite(p).any():
                    dist = float(np.linalg.norm(np.nanmedian(p, axis=0) - ref_img)) / max(ref_h, 1.0)
                    if dist > 1.5:
                        continue
                    dist_txt = "zelfde plek in beeld"
            if dist is None:
                if ref_xy is None or not cd.calibrated or not np.isfinite(ref_xy).all():
                    continue
                xy = np.nanmedian(cd.xy[rows], axis=0)
                if not np.isfinite(xy).all() or np.linalg.norm(xy - ref_xy) > 4.0:
                    continue
                dist = float(np.linalg.norm(xy - ref_xy)) / 4.0
                dist_txt = f"{np.linalg.norm(xy - ref_xy):.0f} m van de aangewezen plek"
            still = cd.stationary.get(tid, 0.0) if cd.stationary else 0.0
            if still < 0.4:
                continue  # loopt rond: waarschijnlijk toch een speler
            out.append({"clip_id": c["id"], "track_id": tid, "t_start": round(float(tr["t_start"] or 0), 2),
                        "t_end": round(float(tr["t_end"] or 0), 2), "reason": dist_txt,
                        "score": round(float(still) * float(np.exp(-dist)), 3)})
    out.sort(key=lambda r: -r["score"])
    return out[:limit]


def clip_ball(store: Store, clip_id: int, t0: float, t1: float) -> list[dict]:
    """De bal in beeldpixels tussen t0 en t1 (om in de video te tekenen en te controleren).
    kind: 'gevonden' (door de analyse), 'hand' (door de gebruiker) of 'geschat' (gat opgevuld)."""
    d = load_clip(store, clip_id)
    if d is None or not len(d.ball_idx):
        return []
    found = {int(i) for (i,) in store.rows("SELECT idx FROM ball WHERE clip_id = ?", (clip_id,))}
    manual = {int(np.argmin(np.abs(d.t - tm))) for (tm,) in store.rows(
        "SELECT t FROM ball_manual WHERE clip_id = ? AND x IS NOT NULL", (clip_id,))}
    lo, hi = np.searchsorted(d.t, t0), np.searchsorted(d.t, t1, side="right")
    out = []
    for k in range(np.searchsorted(d.ball_idx, lo), np.searchsorted(d.ball_idx, hi)):
        i = int(d.ball_idx[k])
        x, y = d.ball_img[k]
        if not np.isfinite([x, y]).all():
            continue
        out.append({"t": round(float(d.t[i]), 3), "x": round(float(x), 1), "y": round(float(y), 1),
                    "kind": "hand" if i in manual else "gevonden" if i in found else "geschat"})
    return out
