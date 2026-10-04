"""De app slimmer maken met jouw eigen beelden (alles blijft op je laptop).

Metafoor: een stagiair die meekijkt. Elke keer dat jij iets verbetert (de bal aanwijzen, een
toeschouwer weghalen, het veld goed leggen, een speler koppelen), schrijft hij dat op. Pas als jij
zegt "ga maar studeren" (de knop), gaat hij met die aantekeningen oefenen. Daarna vergelijken we
zijn nieuwe kennis met de oude op beelden die hij niet heeft gezien; alleen als hij er echt beter
van is geworden, gebruikt de app voortaan het nieuwe model. Terug naar het oude kan altijd.

Drie soorten oefenen:
1. Spelers en bal (detector): het YOLO-model bijtrainen op beelden van video's die jij hebt
   goedgekeurd. Vooral de bal: wat de app alleen ingezoomd vond of wat jij aanwees, leert hij zo
   ook in het gewone beeld te zien.
2. Het veld (veldlijnen): een klein netwerk leert uit goed gekalibreerde beelden welke pixels
   veldlijn zijn, ook op velden met slechte of halfverdwenen lijnen.
3. Spelers herkennen: per speler een "uiterlijk-profiel" (houding, haar, schoenen, kleur) uit zijn
   gekoppelde stukken video, bewaard bij de vaste selectie, zodat de app hem in een volgende
   wedstrijd zelf herkent.
"""
from __future__ import annotations

import json
import math
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

from . import config, pitch
from .storage import Store

REGISTRY = "registry.json"
KINDS = ("detector", "field")


# --- modelversies -------------------------------------------------------------------------

def custom_dir() -> Path:
    d = config.MODELS_DIR / "custom"
    d.mkdir(parents=True, exist_ok=True)
    return d


def registry() -> dict:
    p = config.MODELS_DIR / REGISTRY
    try:
        r = json.loads(p.read_text())
    except (OSError, ValueError):
        r = {}
    for k in KINDS:
        r.setdefault(k, {"active": None, "versions": []})
    return r


def save_registry(r: dict) -> None:
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    p = config.MODELS_DIR / REGISTRY
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(r, indent=1))
    tmp.replace(p)


def active_model(kind: str) -> Path | None:
    name = registry()[kind].get("active")
    if not name:
        return None
    p = custom_dir() / name
    return p if p.exists() else None


def set_active(kind: str, name: str | None) -> None:
    r = registry()
    if name is not None and not any(v["file"] == name for v in r[kind]["versions"]):
        raise ValueError("Onbekende versie")
    r[kind]["active"] = name
    save_registry(r)


def add_version(kind: str, file: str, info: dict, activate: bool) -> None:
    r = registry()
    r[kind]["versions"].append({"file": file, "created": time.strftime("%Y-%m-%d %H:%M"), **info})
    if activate:
        r[kind]["active"] = file
    save_registry(r)


def device() -> str:
    from .detection import best_device
    return best_device()


# --- trainingsdata verzamelen ---------------------------------------------------------------

def approved_clips(store: Store) -> list[dict]:
    """Video's die de gebruiker heeft goedgekeurd voor training (en die geanalyseerd zijn)."""
    return store.all("SELECT * FROM clips WHERE train_ok = 1 AND status = 'klaar' ORDER BY id")


def _ball_diameter(boxes: np.ndarray, y: float) -> float:
    """Doorsnede van de bal in pixels, geschat uit de spelers op dezelfde hoogte in beeld
    (een bal is ongeveer 1/8 van een speler)."""
    if not len(boxes):
        return 10.0
    feet = boxes[:, 3]
    k = int(np.argmin(np.abs(feet - y)))
    return float(np.clip(0.125 * (boxes[k, 3] - boxes[k, 1]), 5.0, 60.0))


def detector_examples(store: Store, max_images: int = 800) -> list[dict]:
    """Beelden met labels voor het bijtrainen van de detector.

    We nemen alleen beelden waarvan we weten waar de bal is (gevonden, ingezoomd gevonden of
    door jou aangewezen) of waarvan jij zei dat er geen bal is. Een beeld waarin de bal er wél is
    maar niet gevonden werd, zou het model leren de bal níet te zien; die laten we weg.
    Moeilijke voorbeelden (alleen ingezoomd gevonden, of door jou aangewezen) gaan voor."""
    out: list[dict] = []
    for clip in approved_clips(store):
        cid = clip["id"]
        t = {i: tt for i, tt in store.rows("SELECT idx, t FROM frames WHERE clip_id = ?", (cid,))}
        if not t:
            continue
        times = np.array([t[i] for i in sorted(t)])
        ball = {int(i): (x, y, src or "det") for i, x, y, src in
                store.rows("SELECT idx, x, y, src FROM ball WHERE clip_id = ?", (cid,))}
        no_ball: set[int] = set()
        for tm, x, y in store.rows("SELECT t, x, y FROM ball_manual WHERE clip_id = ?", (cid,)):
            i = int(np.argmin(np.abs(times - tm)))
            if abs(times[i] - tm) > 0.08:
                continue
            if x is None:
                ball.pop(i, None)
                no_ball.add(i)
            else:
                ball[i] = (x, y, "hand")
        hard = [i for i, b in ball.items() if b[2] in ("hand", "zoom", "scan")]
        easy = sorted(i for i, b in ball.items() if b[2] == "det")[::4]  # gewone treffers: om de zoveel
        chosen = sorted(set(hard) | set(easy) | no_ball)
        if len(chosen) > 400:  # per video begrenzen, verspreid over de video
            keep = set(hard[:200]) | set(no_ball)
            rest = [i for i in chosen if i not in keep]
            chosen = sorted(keep | set(rest[:: max(1, len(rest) // max(1, 400 - len(keep)))]))
        dets: dict[int, list] = {}
        if chosen:
            q = ",".join("?" * len(chosen))
            for i, x1, y1, x2, y2 in store.rows(
                    f"SELECT idx, x1, y1, x2, y2 FROM detections WHERE clip_id = ? AND idx IN ({q})", (cid, *chosen)):
                dets.setdefault(int(i), []).append((x1, y1, x2, y2))
        for i in chosen:
            boxes = np.array(dets.get(i, []), float).reshape(-1, 4)
            b = ball.get(i)
            out.append({"clip_id": cid, "path": clip["path"], "t": float(t[i]), "idx": int(i),
                        "w": int(clip["width"]), "h": int(clip["height"]), "persons": boxes.tolist(),
                        "ball": None if b is None else [b[0], b[1], _ball_diameter(boxes, b[1])],
                        "hard": b is not None and b[2] != "det"})
    if len(out) > max_images:
        hard = [e for e in out if e["hard"]]
        rest = [e for e in out if not e["hard"]]
        n_rest = max(0, max_images - len(hard))
        out = hard[:max_images] + rest[:: max(1, len(rest) // max(1, n_rest))][:n_rest]
    return out


def _is_val(e: dict) -> bool:
    """Ongeveer 1 op de 7 stukjes van 10 seconden houden we apart om te controleren (niet om op
    te oefenen), zodat het model zich niet kan 'inprenten' wat er vlak daarnaast gebeurt."""
    return (int(e["t"] // 10) + e["clip_id"]) % 7 == 3


def read_frames(examples: list[dict], progress=None):
    """(voorbeeld, beeld) per voorbeeld, per video op volgorde van tijd gelezen."""
    by_clip: dict[str, list[dict]] = {}
    for e in examples:
        by_clip.setdefault(e["path"], []).append(e)
    done = 0
    for path, items in by_clip.items():
        cap = cv2.VideoCapture(path)
        try:
            for e in sorted(items, key=lambda x: x["t"]):
                cap.set(cv2.CAP_PROP_POS_MSEC, e["t"] * 1000)
                ok, frame = cap.read()
                done += 1
                if progress:
                    progress(done / len(examples))
                if ok:
                    yield e, frame
        finally:
            cap.release()


def write_detector_dataset(examples: list[dict], root: Path, names: dict, progress=None) -> Path:
    """YOLO-dataset op schijf: images/{train,val} en labels/{train,val} plus data.yaml."""
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    from .detection import BALL, PERSON

    n = 0
    for e, frame in read_frames(examples, progress):
        split = "val" if _is_val(e) else "train"
        H, W = frame.shape[:2]
        name = f"{e['clip_id']}_{e['idx']:06d}"
        cv2.imwrite(str(root / "images" / split / f"{name}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        lines = []
        for x1, y1, x2, y2 in e["persons"]:
            x1, x2 = max(0.0, x1), min(W - 1.0, x2)
            y1, y2 = max(0.0, y1), min(H - 1.0, y2)
            if x2 - x1 > 2 and y2 - y1 > 4:
                lines.append(f"{PERSON} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
        if e["ball"]:
            x, y, d = e["ball"]
            lines.append(f"{BALL} {x / W:.6f} {y / H:.6f} {d / W:.6f} {d / H:.6f}")
        (root / "labels" / split / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        n += 1
    yaml = root / "data.yaml"
    yaml.write_text(f"path: {root}\ntrain: images/train\nval: images/val\nnames:\n" +
                    "".join(f"  {k}: {json.dumps(v)}\n" for k, v in sorted(names.items())))
    return yaml


# --- het veld: een klein netwerk dat veldlijnen leert zien -------------------------------------

FIELD_SIZE = (960, 544)  # (breedte, hoogte) waarop het netwerk werkt (zelfde schaal als de lijnherkenning)


def field_examples(store: Store, max_frames: int = 600) -> list[dict]:
    """Beelden met een betrouwbare kalibratie: momenten met een eigen sleutelframe of een
    automatisch sleutelframe dat goed op de lijnen paste (of door jou is goedgekeurd)."""
    from . import analytics

    out = []
    for clip in approved_clips(store):
        d = analytics.load_clip(store, clip["id"])
        if d is None or not d.calibrated or not len(d.t):
            continue
        kfs = store.keyframes(clip["id"])
        times = []
        for k in kfs:
            ok = not k.get("auto") or k.get("accepted")
            if not ok and k.get("score"):
                try:
                    ok = json.loads(k["score"]).get("precision", 0) >= 0.9
                except (ValueError, AttributeError):
                    ok = False
            if ok:
                times.append(float(k["t"]))
        # rond elk betrouwbaar moment ook de beelden tot 2 s ervoor en erna: de camerabeweging houdt
        # de kalibratie daar nog goed vast
        near = sorted({int(np.argmin(np.abs(d.t - (tm + dt)))) for tm in times for dt in np.arange(-2.0, 2.01, 0.5)})
        for i in near:
            hs = d.camera.homographies(i)
            if not hs:
                continue
            Himg = max(hs, key=lambda hw: hw[1])[0]  # beeld -> veld
            rows = np.flatnonzero(d.idx == i)
            out.append({"clip_id": clip["id"], "path": clip["path"], "t": float(d.t[i]), "idx": i,
                        "H": np.linalg.inv(Himg).tolist(), "boxes": d.boxes[rows].tolist(),
                        "w": int(clip["width"]), "h": int(clip["height"]), "geom": [d.geom.length, d.geom.width]})
    if len(out) > max_frames:
        out = out[:: math.ceil(len(out) / max_frames)]
    return out


def line_mask(H_pitch_to_img: np.ndarray, size: tuple[int, int], frame_size: tuple[int, int],
              geom: pitch.Geometry) -> np.ndarray:
    """Waar de veldlijnen volgens de kalibratie in beeld liggen (0/1), op netwerkschaal."""
    from .autocalib import _line_width_px, model

    W, Hh = size
    s = np.diag([W / frame_size[0], Hh / frame_size[1], 1.0])
    Hs = s @ np.asarray(H_pitch_to_img, float)
    pts = model(geom).dense
    hom = np.hstack([pts, np.ones((len(pts), 1))]) @ Hs.T
    front = hom[:, 2] > 1e-6
    img = hom[:, :2] / np.where(front[:, None], hom[:, 2:3], 1.0)
    width = _line_width_px(Hs, pts)
    mask = np.zeros((Hh, W), np.uint8)
    inside = front & (img[:, 0] > -20) & (img[:, 0] < W + 20) & (img[:, 1] > -20) & (img[:, 1] < Hh + 20)
    for (x, y), wpx in zip(img[inside], width[inside]):
        r = int(np.clip(round(wpx / 2), 1, 4))
        cv2.circle(mask, (int(round(x)), int(round(y))), r, 1, -1)
    return mask


def field_net():
    """Een klein U-Net: beeld in, per pixel de kans op veldlijn uit."""
    import torch
    from torch import nn

    def block(a, b):
        return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(inplace=True),
                             nn.Conv2d(b, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(inplace=True))

    class Net(nn.Module):
        def __init__(self, c=(16, 32, 64, 96)):
            super().__init__()
            self.d1, self.d2, self.d3, self.d4 = block(3, c[0]), block(c[0], c[1]), block(c[1], c[2]), block(c[2], c[3])
            self.pool = nn.MaxPool2d(2)
            self.u3, self.u2, self.u1 = block(c[3] + c[2], c[2]), block(c[2] + c[1], c[1]), block(c[1] + c[0], c[0])
            self.out = nn.Conv2d(c[0], 1, 1)

        def forward(self, x):
            up = lambda a, b: torch.cat([nn.functional.interpolate(a, size=b.shape[-2:], mode="bilinear",  # noqa: E731
                                                                    align_corners=False), b], 1)
            x1 = self.d1(x)
            x2 = self.d2(self.pool(x1))
            x3 = self.d3(self.pool(x2))
            x4 = self.d4(self.pool(x3))
            y = self.u3(up(x4, x3))
            y = self.u2(up(y, x2))
            y = self.u1(up(y, x1))
            return self.out(y)

    return Net()


def to_tensor(frame: np.ndarray):
    import torch
    small = cv2.resize(frame, FIELD_SIZE, interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255.0
    return (x - 0.5) / 0.25


_field_cache: dict = {}


def learned_lines(frame: np.ndarray, threshold: float = 0.5) -> np.ndarray | None:
    """Veldlijnen volgens het geleerde netwerk (0/255, schaal 960 breed), of None als er (nog)
    geen geleerd veldmodel actief is."""
    p = active_model("field")
    if p is None:
        return None
    import torch

    key = (str(p), p.stat().st_mtime)
    if _field_cache.get("key") != key:
        net = field_net()
        net.load_state_dict(torch.load(p, map_location="cpu"))
        net.eval()
        dev = device()
        try:
            net.to(dev)
        except Exception:  # noqa: BLE001
            dev = "cpu"
        _field_cache.update(key=key, net=net, dev=dev)
    net, dev = _field_cache["net"], _field_cache["dev"]
    with torch.no_grad():
        prob = torch.sigmoid(net(to_tensor(frame)[None].to(dev)))[0, 0].cpu().numpy()
    s = 960 / frame.shape[1]
    h = int(round(frame.shape[0] * s))
    prob = cv2.resize(prob, (960, h), interpolation=cv2.INTER_LINEAR)
    return (prob > threshold).astype(np.uint8) * 255


def line_f1(pred: np.ndarray, truth: np.ndarray, tol: int = 2) -> tuple[float, float, float]:
    """(precisie, vangst, F1) van gevonden lijnpixels, met een paar pixels speling."""
    k = np.ones((2 * tol + 1, 2 * tol + 1), np.uint8)
    p, t = pred > 0, truth > 0
    if not p.any() and not t.any():
        return 1.0, 1.0, 1.0
    prec = float((p & (cv2.dilate(t.astype(np.uint8), k) > 0)).sum() / max(1, p.sum()))
    rec = float((t & (cv2.dilate(p.astype(np.uint8), k) > 0)).sum() / max(1, t.sum()))
    return prec, rec, 2 * prec * rec / max(1e-9, prec + rec)


# --- spelers herkennen -----------------------------------------------------------------------

EMB_MODEL = "yolo11n-cls.pt"
_emb_cache: dict = {}


def _embedder():
    if "m" not in _emb_cache:
        from ultralytics import YOLO

        from .detection import _no_telemetry
        _no_telemetry()
        path = config.MODELS_DIR / EMB_MODEL
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        _emb_cache["m"] = YOLO(str(path) if path.exists() else EMB_MODEL)
        if not path.exists():
            src = Path(EMB_MODEL)
            if src.exists():
                shutil.move(str(src), path)
    return _emb_cache["m"]


def _color_parts(crop: np.ndarray) -> np.ndarray:
    """Kleur van romp, broek en sokken/schoenen (HSV-histogrammen): wat het algemene netwerk mist."""
    h = crop.shape[0]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    feats = []
    for a, b in ((0.0, 0.2), (0.2, 0.55), (0.55, 0.8), (0.8, 1.0)):  # hoofd, romp, broek, sokken/schoenen
        part = hsv[int(a * h):max(int(a * h) + 1, int(b * h))]
        hist = cv2.calcHist([part], [0, 1], None, [12, 4], [0, 180, 0, 256]).ravel()
        feats.append(hist / max(1.0, hist.sum()))
    return np.concatenate(feats)


def appearance(crops: list[np.ndarray]) -> np.ndarray | None:
    """Uiterlijk van een persoon (gemiddeld over een paar uitsneden): één genormaliseerde vector."""
    crops = [c for c in crops if c is not None and c.size and c.shape[0] >= 24 and c.shape[1] >= 8]
    if not crops:
        return None
    m = _embedder()
    sized = [cv2.resize(c, (64, 128), interpolation=cv2.INTER_AREA) for c in crops]
    embs = m.embed(sized, imgsz=128, verbose=False, device=device())
    deep = np.mean([e.cpu().numpy() for e in embs], axis=0)
    deep = deep / (np.linalg.norm(deep) or 1.0)
    col = np.mean([_color_parts(c) for c in sized], axis=0)
    col = col / (np.linalg.norm(col) or 1.0)
    v = np.concatenate([deep, col]).astype(np.float32)  # (gewicht 1:1 werkte het best op echte clips)
    return v / (np.linalg.norm(v) or 1.0)


def similarity(a: np.ndarray, b: np.ndarray, center: np.ndarray | None = None) -> float:
    """Hoe erg twee uiterlijken op elkaar lijken (-1..1). Met center (het gemiddelde uiterlijk van
    het team in deze wedstrijd) tellen alleen de verschillen tussen teamgenoten: iedereen heeft
    hetzelfde shirt, dus dat zegt niets."""
    if center is not None:
        a, b = a - center, b - center
        a, b = a / (np.linalg.norm(a) or 1.0), b / (np.linalg.norm(b) or 1.0)
    return float(np.dot(a, b))


def team_centers(store: Store, match_id: int) -> dict[int, np.ndarray]:
    """Gemiddeld uiterlijk per team in deze wedstrijd (uit de bekeken stukken)."""
    rows = store.all("SELECT e.emb, COALESCE(p.team, t.team) AS team FROM track_embeds e "
                     "JOIN tracks t ON t.clip_id = e.clip_id AND t.track_id = e.track_id JOIN clips c ON c.id = t.clip_id "
                     "LEFT JOIN players p ON p.id = t.player_id WHERE c.match_id = ?", (match_id,))
    out = {}
    for team in (0, 1):
        vs = [from_blob(r["emb"]) for r in rows if r["team"] == team]
        if len(vs) >= 3:
            out[team] = np.mean(vs, axis=0)
    return out


def to_blob(v: np.ndarray) -> bytes:
    return np.asarray(v, np.float32).tobytes()


def from_blob(b: bytes) -> np.ndarray:
    return np.frombuffer(b, np.float32)


def profile_key(store: Store, player: dict) -> tuple[int | None, str, int | None]:
    """Waar het profiel van een speler bewaard wordt: bij de vaste selectie (dan herkent de app hem
    ook in volgende wedstrijden) of anders alleen bij deze wedstrijd."""
    m = store.one("SELECT * FROM matches WHERE id = ?", (player["match_id"],)) or {}
    sq = m.get(f"team{player['team']}_squad") if player.get("team") in (0, 1) else None
    if sq:
        return int(sq), player["name"].strip().lower(), None
    return None, player["name"].strip().lower(), int(player["id"])


def profile_of(store: Store, player: dict) -> np.ndarray | None:
    sq, name, pid = profile_key(store, player)
    row = (store.one("SELECT emb FROM player_profiles WHERE squad_id = ? AND name = ?", (sq, name)) if sq
           else store.one("SELECT emb FROM player_profiles WHERE match_player_id = ?", (pid,)))
    return from_blob(row["emb"]) if row else None


def save_profile(store: Store, player: dict, emb: np.ndarray, n: int) -> None:
    sq, name, pid = profile_key(store, player)
    if sq:
        store.run("DELETE FROM player_profiles WHERE squad_id = ? AND name = ?", (sq, name))
    else:
        store.run("DELETE FROM player_profiles WHERE match_player_id = ?", (pid,))
    store.run("INSERT INTO player_profiles (squad_id, name, match_player_id, emb, n) VALUES (?,?,?,?,?)",
              (sq, name, pid, to_blob(emb), n))


def track_embedding(store: Store, clip_id: int, track_id: int) -> np.ndarray | None:
    row = store.one("SELECT emb FROM track_embeds WHERE clip_id = ? AND track_id = ?", (clip_id, track_id))
    return from_blob(row["emb"]) if row else None


def recognize(store: Store, match_id: int, min_score: float = 0.35, margin: float = 0.12) -> list[dict]:
    """Ongekoppelde tracks die volgens de spelerprofielen bij een speler horen (beste eerst).

    Per track vergelijken we het uiterlijk met de profielen van de spelers van zijn team. Hij
    moet duidelijk het meest lijken op één speler (een marge boven de nummer twee)."""
    from . import analytics

    players = store.all("SELECT * FROM players WHERE match_id = ?", (match_id,))
    profiles = [(p, profile_of(store, p)) for p in players]
    profiles = [(p, e) for p, e in profiles if e is not None]
    if not profiles:
        return []
    centers = team_centers(store, match_id)
    out = []
    for clip in store.all("SELECT id FROM clips WHERE match_id = ? AND status = 'klaar'", (match_id,)):
        d = analytics.load_clip(store, clip["id"])
        if d is None:
            continue
        embs = {tid: from_blob(b) for tid, b in store.rows("SELECT track_id, emb FROM track_embeds WHERE clip_id = ?",
                                                             (clip["id"],))}
        linked_times: dict[int, list[tuple[float, float]]] = {}
        for tid, tr in d.tracks.items():
            if tr["player_id"]:
                linked_times.setdefault(tr["player_id"], []).append((tr["t_start"] or 0, tr["t_end"] or 0))
        for tid, tr in d.tracks.items():
            if tr["player_id"] or tid not in d.valid_tracks or tid not in embs:
                continue
            team = d.team.get(tid, -1)
            if team not in (0, 1, -1):
                continue  # scheidsrechter, keeper in ander tenue, toeschouwer
            cands = [(p, similarity(embs[tid], e, centers.get(p["team"]))) for p, e in profiles if team == -1 or p["team"] == team]
            # wie op hetzelfde moment al ergens anders in beeld is, kan het niet zijn
            cands = [(p, s) for p, s in cands if not any(
                min(tr["t_end"] or 0, b) - max(tr["t_start"] or 0, a) > 0.3 for a, b in linked_times.get(p["id"], []))]
            if not cands:
                continue
            cands.sort(key=lambda c: -c[1])
            best, s1 = cands[0]
            s2 = cands[1][1] if len(cands) > 1 else 0.0
            if s1 >= min_score and s1 - s2 >= margin:
                out.append({"clip_id": clip["id"], "track_id": tid, "player_id": best["id"], "score": round(s1, 3),
                            "margin": round(s1 - s2, 3), "t_start": round(float(tr["t_start"] or 0), 2),
                            "t_end": round(float(tr["t_end"] or 0), 2)})
    out.sort(key=lambda r: (-r["margin"], -r["score"]))
    return out
