"""Trainen in een apart proces (zodat de app gewoon blijft werken en je het kunt stoppen).

Gebruik: python -m app.train <soort> <map> '<opties als json>'
soort: detector | field | players

Het proces schrijft zijn voortgang naar <map>/progress.json en het resultaat naar <map>/result.json.
"""
from __future__ import annotations

import json
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from . import config, learning
from .storage import Store


class Progress:
    """Voortgang bijhouden: welke stap, hoe ver, en hoe lang nog (geschat uit het tempo tot nu)."""

    def __init__(self, run: Path, steps: list[tuple[str, float]], expected_s: float | None = None):
        self.run, self.steps = run, steps  # [(naam, gewicht)]
        self.total = sum(w for _, w in steps)
        self.started = self.step_started = time.time()
        self.step = 0
        self.expected = expected_s  # verwachte totale duur (uit de planning), voor een eerlijke schatting vooraf

    def __call__(self, frac: float, message: str = "") -> None:
        frac = min(1.0, max(0.0, frac))
        done = sum(w for _, w in self.steps[:self.step]) + self.steps[self.step][1] * frac
        f = done / self.total
        el = time.time() - self.started
        # resterende tijd: binnen de huidige stap uit het tempo tot nu, de stappen daarna uit de planning
        later_w = sum(w for _, w in self.steps[self.step + 1:]) / self.total
        step_el = time.time() - self.step_started
        if frac > 0.05:
            rest = step_el / frac * (1 - frac)
        elif self.expected:
            rest = self.expected * self.steps[self.step][1] / self.total * (1 - frac)
        else:
            rest = None
        later = self.expected * later_w if self.expected else (el / f * later_w if f > 0.03 else None)
        eta = None if rest is None or later is None else rest + later
        data = {"phase": self.steps[self.step][0], "frac": round(f, 4), "eta_s": None if eta is None else round(eta),
                "message": message, "elapsed_s": round(el), "updated": time.time()}
        tmp = self.run / "progress.tmp"
        tmp.write_text(json.dumps(data))
        tmp.replace(self.run / "progress.json")

    def next(self, message: str = "") -> None:
        self.step = min(self.step + 1, len(self.steps) - 1)
        self.step_started = time.time()
        self(0.0, message)


def result(run: Path, data: dict) -> None:
    (run / "result.json").write_text(json.dumps(data, indent=1))


# --- spelers en bal ---------------------------------------------------------------------------

SEC_PER_IMAGE = {"mps": 0.55, "cuda": 0.15, "cpu": 4.0}  # één keer leren van één beeld (1280 px)


def detector_plan(n_images: int, quick: bool) -> dict:
    epochs = 8 if quick else 20
    dev = learning.device()
    est = n_images * 0.85 * epochs * SEC_PER_IMAGE.get(dev, 4.0) + n_images * 1.5 + 120
    return {"epochs": epochs, "device": dev, "minutes": max(1, round(est / 60))}


def _class_ap(metrics, cls: int) -> float | None:
    box = metrics.box
    idx = list(getattr(box, "ap_class_index", []))
    if cls not in idx:
        return None
    return float(box.ap50[idx.index(cls)])


def train_detector(store: Store, run: Path, opts: dict) -> dict:
    from ultralytics import YOLO

    from .detection import BALL, PERSON, _no_telemetry

    _no_telemetry()
    prog = Progress(run, [("Voorbeelden verzamelen", 0.03), ("Beelden klaarzetten", 0.12),
                          ("Oefenen", 0.75), ("Vergelijken met het oude model", 0.10)])
    prog(0, "Welke beelden kunnen we gebruiken?")
    ex = learning.detector_examples(store, max_images=int(opts.get("max_images", 800)))
    n_val = sum(1 for e in ex if learning._is_val(e))
    if len(ex) < int(opts.get("min_examples", 30)) or n_val < 2:
        return {"ok": False, "message": f"Te weinig voorbeelden ({len(ex)}). Keur meer video's goed voor training, "
                                        "of wijs in 'Video + minimap' vaker de bal aan."}
    base_path = opts.get("base") or learning.active_model("detector") or (config.MODELS_DIR / Path(config.YOLO_MODEL).name)
    base = YOLO(str(base_path) if Path(base_path).exists() else config.YOLO_MODEL)
    prog.next(f"{len(ex)} beelden uit je video's")
    data_dir = run / "data"
    yaml = learning.write_detector_dataset(ex, data_dir, base.names, progress=lambda f: prog(f, "Beelden klaarzetten"))
    plan = detector_plan(len(ex), bool(opts.get("quick", True)))
    plan["epochs"] = int(opts.get("epochs", plan["epochs"]))
    imgsz = int(opts.get("imgsz", config.DETECT_IMGSZ))
    batch = int(opts.get("batch", 2 if plan["device"] != "cuda" else 8))

    prog.expected = plan["minutes"] * 60
    prog.next("Oefenen begint")
    model = YOLO(str(base_path) if Path(base_path).exists() else config.YOLO_MODEL)
    n_batches = max(1, int(len(ex) * 0.85 / batch))
    state = {"epoch": 0, "batch": 0}

    def on_batch(trainer):
        state["batch"] += 1
        f = (state["epoch"] + min(1.0, state["batch"] / n_batches)) / plan["epochs"]
        prog(f, f"Ronde {state['epoch'] + 1} van {plan['epochs']}")

    def on_epoch(trainer):
        state["epoch"] += 1
        state["batch"] = 0

    model.add_callback("on_train_batch_end", on_batch)
    model.add_callback("on_train_epoch_end", on_epoch)
    model.train(data=str(yaml), epochs=plan["epochs"], imgsz=imgsz, batch=batch, device=plan["device"],
                project=str(run), name="yolo", exist_ok=True, workers=0, plots=False, verbose=False,
                freeze=10,  # alleen de bovenste lagen bijstellen: snel, en het model vergeet niet wat het al kon
                lr0=0.002, warmup_epochs=1, mosaic=0.0, scale=0.25, degrees=0.0, fliplr=0.5,
                hsv_h=0.01, hsv_s=0.4, hsv_v=0.3, patience=max(3, plan["epochs"] // 3), val=False, amp=False)
    best = run / "yolo" / "weights" / "last.pt"
    if not best.exists():
        return {"ok": False, "message": "Het oefenen gaf geen model (zie log)"}

    prog.next("Het nieuwe en het oude model op dezelfde controlebeelden")
    kw = dict(data=str(yaml), imgsz=imgsz, batch=batch, device=plan["device"], plots=False, verbose=False,
              classes=[PERSON, BALL], split="val", workers=0)
    m_old = base.val(**kw)
    prog(0.5, "Nog even vergelijken")
    m_new = YOLO(str(best)).val(**kw)
    old = {"ball": _class_ap(m_old, BALL), "person": _class_ap(m_old, PERSON)}
    new = {"ball": _class_ap(m_new, BALL), "person": _class_ap(m_new, PERSON)}
    nb, ob, np_, op = new["ball"] or 0, old["ball"] or 0, new["person"] or 0, old["person"] or 0
    # beter = op één punt duidelijk vooruit en op het andere niet achteruit
    better = (nb > ob + 0.01 and np_ >= op - 0.02) or (np_ > op + 0.02 and nb >= ob - 0.01)
    name = f"detector-{time.strftime('%Y%m%d-%H%M')}.pt"
    shutil.copy(best, learning.custom_dir() / name)
    learning.add_version("detector", name, {"metrics": new, "previous": old, "n_images": len(ex),
                                            "n_val": n_val, "better": better}, activate=better)
    pct = lambda v: "–" if v is None else f"{round(100 * v)}%"  # noqa: E731
    msg = (f"Bal: {pct(old['ball'])} → {pct(new['ball'])}, spelers: {pct(old['person'])} → {pct(new['person'])} "
           f"(gevonden op {n_val} controlebeelden). ")
    msg += "Het nieuwe model wordt vanaf nu gebruikt." if better else \
        "Het nieuwe model is niet duidelijk beter; de app blijft het oude gebruiken (je kunt het wel zelf kiezen)."
    shutil.rmtree(data_dir, ignore_errors=True)
    return {"ok": True, "better": better, "file": name, "metrics": new, "previous": old, "message": msg}


# --- het veld ----------------------------------------------------------------------------------

def field_plan(n_frames: int, quick: bool) -> dict:
    epochs = 15 if quick else 40
    dev = learning.device()
    sec = {"mps": 0.12, "cuda": 0.03, "cpu": 1.2}.get(dev, 1.2)
    return {"epochs": epochs, "device": dev, "minutes": max(1, round((n_frames * 0.85 * epochs * sec + n_frames * 0.6 + 60) / 60))}


def train_field(store: Store, run: Path, opts: dict) -> dict:
    import cv2
    import torch

    from . import pitch
    from .autocalib import detect_lines

    prog = Progress(run, [("Voorbeelden verzamelen", 0.05), ("Beelden klaarzetten", 0.15), ("Oefenen", 0.7),
                          ("Vergelijken met de gewone lijnherkenning", 0.10)])
    prog(0, "Welke beelden zijn goed gekalibreerd?")
    ex = learning.field_examples(store, max_frames=int(opts.get("max_frames", 600)))
    if len(ex) < int(opts.get("min_examples", 20)):
        return {"ok": False, "message": f"Te weinig goed gekalibreerde beelden ({len(ex)}). Kalibreer en keur meer video's goed."}
    prog.next(f"{len(ex)} beelden")
    X, Y, M, val, frames = [], [], [], [], []
    for k, (e, frame) in enumerate(learning.read_frames(ex, lambda f: prog(f, "Beelden klaarzetten"))):
        geom = pitch.geometry(*e["geom"])
        mask = learning.line_mask(np.array(e["H"]), learning.FIELD_SIZE, (e["w"], e["h"]), geom)
        ignore = np.ones_like(mask, np.float32)  # spelers: daar weten we het niet zeker
        sx, sy = learning.FIELD_SIZE[0] / e["w"], learning.FIELD_SIZE[1] / e["h"]
        for x1, y1, x2, y2 in e["boxes"]:
            ignore[max(0, int(y1 * sy) - 2):int(y2 * sy) + 3, max(0, int(x1 * sx) - 2):int(x2 * sx) + 3] = 0
        X.append(learning.to_tensor(frame))
        Y.append(torch.from_numpy(mask.astype(np.float32)))
        M.append(torch.from_numpy(ignore))
        is_val = learning._is_val(e)
        val.append(is_val)
        if is_val:
            frames.append((frame, mask, ignore))
    tr = [i for i, v in enumerate(val) if not v]
    vl = [i for i, v in enumerate(val) if v]
    if len(vl) < 1 or len(tr) < 3:
        return {"ok": False, "message": "Te weinig beelden om te oefenen en te controleren"}
    plan = field_plan(len(ex), bool(opts.get("quick", True)))
    plan["epochs"] = int(opts.get("epochs", plan["epochs"]))
    dev = plan["device"]
    net = learning.field_net().to(dev)
    prev = learning.active_model("field")
    if prev is not None:  # verder leren vanaf het vorige veldmodel
        try:
            net.load_state_dict(torch.load(prev, map_location="cpu"))
        except Exception:  # noqa: BLE001
            pass
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    pos_w = torch.tensor(8.0, device=dev)  # lijnen zijn maar een klein deel van het beeld
    bs = 4
    prog.expected = plan["minutes"] * 60
    prog.next("Oefenen begint")
    rng = np.random.default_rng(0)
    for ep in range(plan["epochs"]):
        net.train()
        order = rng.permutation(tr)
        for b in range(0, len(order), bs):
            ids = order[b:b + bs]
            x = torch.stack([X[i] for i in ids]).to(dev)
            if rng.random() < 0.5:  # spiegelen
                x = torch.flip(x, [3])
                y = torch.flip(torch.stack([Y[i] for i in ids]), [2]).to(dev)
                m = torch.flip(torch.stack([M[i] for i in ids]), [2]).to(dev)
            else:
                y = torch.stack([Y[i] for i in ids]).to(dev)
                m = torch.stack([M[i] for i in ids]).to(dev)
            out = net(x)[:, 0]
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(out, y, pos_weight=pos_w, reduction="none") * m).mean()
            p = torch.sigmoid(out) * m
            dice = 1 - (2 * (p * y).sum() + 1) / (p.sum() + (y * m).sum() + 1)
            opt.zero_grad()
            (loss + dice).backward()
            opt.step()
            prog((ep + min(1.0, (b + bs) / len(order))) / plan["epochs"], f"Ronde {ep + 1} van {plan['epochs']}")
    prog.next("Vergelijken")
    name = f"field-{time.strftime('%Y%m%d-%H%M')}.pt"
    tmp = run / name
    torch.save(net.cpu().state_dict(), tmp)
    net.eval()
    f_new, f_old = [], []
    with torch.no_grad():
        for k, (frame, mask, ignore) in enumerate(frames):
            prob = torch.sigmoid(net(learning.to_tensor(frame)[None]))[0, 0].numpy()
            pred = (prob > 0.5).astype(np.uint8)
            classic = cv2.resize(detect_lines(frame), learning.FIELD_SIZE, interpolation=cv2.INTER_NEAREST) > 0
            ign = ignore > 0
            f_new.append(learning.line_f1(pred * ign, mask * ign)[2])
            f_old.append(learning.line_f1(classic * ign, mask * ign)[2])
            prog((k + 1) / len(frames), "Vergelijken")
    new, old = float(np.mean(f_new)), float(np.mean(f_old))
    better = new > old + 0.02
    shutil.move(str(tmp), learning.custom_dir() / name)
    learning.add_version("field", name, {"metrics": {"f1": round(new, 3)}, "previous": {"f1": round(old, 3)},
                                         "n_images": len(ex), "n_val": len(vl), "better": better}, activate=better)
    msg = (f"Veldlijnen goed herkend: {round(100 * old)}% → {round(100 * new)}% (op {len(vl)} controlebeelden). ")
    msg += "Het automatisch kalibreren gebruikt vanaf nu het geleerde veldmodel." if better else \
        "Niet duidelijk beter dan de gewone lijnherkenning; die blijft in gebruik."
    return {"ok": True, "better": better, "file": name, "message": msg}


# --- spelers herkennen ---------------------------------------------------------------------------

def players_plan(store: Store, match_id: int) -> dict:
    n = store.one("SELECT COALESCE(SUM(duration), 0) AS s FROM clips WHERE match_id = ? AND status = 'klaar'", (match_id,))["s"]
    sec = {"mps": 0.08, "cuda": 0.05, "cpu": 0.25}.get(learning.device(), 0.25)
    return {"minutes": max(1, round((n * 2 * sec + n * 0.15 + 20) / 60))}


def embed_tracks(store: Store, clip: dict, prog, only: set[int] | None = None, per_track: int = 6) -> int:
    """Per track een uiterlijk-vector uit een paar uitsneden, verspreid over de tijd dat hij in beeld is."""
    import cv2

    tracks = store.all("SELECT track_id, n_frames FROM tracks WHERE clip_id = ? AND n_frames >= 8", (clip["id"],))
    want = {r["track_id"] for r in tracks if only is None or r["track_id"] in only}
    if not want:
        return 0
    det = np.array(store.rows("SELECT idx, track_id, x1, y1, x2, y2 FROM detections WHERE clip_id = ?", (clip["id"],)),
                   float).reshape(-1, 6)
    det = det[np.isin(det[:, 1], list(want))]
    t = {i: tt for i, tt in store.rows("SELECT idx, t FROM frames WHERE clip_id = ?", (clip["id"],))}
    pick: dict[int, list[np.ndarray]] = {}
    for tid in want:
        rows = det[det[:, 1] == tid]
        if not len(rows):
            continue
        h = rows[:, 5] - rows[:, 3]
        rows = rows[h >= max(24.0, 0.5 * np.median(h))]  # niet de kleine (verre) stukjes
        for r in rows[np.linspace(0, len(rows) - 1, min(per_track, len(rows))).astype(int)]:
            pick.setdefault(int(r[0]), []).append(r)
    crops: dict[int, list[np.ndarray]] = {}
    cap = cv2.VideoCapture(clip["path"])
    try:
        frames = sorted(pick)
        for k, i in enumerate(frames):
            cap.set(cv2.CAP_PROP_POS_MSEC, t.get(i, 0.0) * 1000)
            ok, frame = cap.read()
            if ok:
                for r in pick[i]:
                    x1, y1, x2, y2 = map(int, r[2:6])
                    c = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
                    if c.size:
                        crops.setdefault(int(r[1]), []).append(c)
            if k % 20 == 0:
                prog(k / max(1, len(frames)), f"{clip['filename']}: uitsneden bekijken")
    finally:
        cap.release()
    n = 0
    for tid, cs in crops.items():
        v = learning.appearance(cs)
        if v is not None:
            store.run("INSERT OR REPLACE INTO track_embeds (clip_id, track_id, emb) VALUES (?,?,?)",
                      (clip["id"], tid, learning.to_blob(v)))
            n += 1
    return n


def train_players(store: Store, run: Path, opts: dict) -> dict:
    match_id = int(opts["match_id"])
    only = opts.get("player_id")
    clips = store.all("SELECT * FROM clips WHERE match_id = ? AND status = 'klaar' ORDER BY order_idx, id", (match_id,))
    prog = Progress(run, [(f"Video {k + 1}", 1.0) for k in range(len(clips))] + [("Profielen maken", 0.2)],
                    expected_s=players_plan(store, match_id)["minutes"] * 60)
    have = {(c, t) for c, t in store.rows("SELECT clip_id, track_id FROM track_embeds WHERE clip_id IN "
                                            "(SELECT id FROM clips WHERE match_id = ?)", (match_id,))}
    for k, clip in enumerate(clips):
        if k:
            prog.next()
        todo = {t for (t,) in store.rows("SELECT track_id FROM tracks WHERE clip_id = ?", (clip["id"],))
                if (clip["id"], t) not in have}
        embed_tracks(store, clip, prog, only=todo)
    prog.next("Profielen maken")
    players = store.all("SELECT * FROM players WHERE match_id = ?" + (" AND id = ?" if only else ""),
                        (match_id, int(only)) if only else (match_id,))
    made = []
    for p in players:
        # alle gekoppelde stukken van deze speler, ook uit eerdere wedstrijden met dezelfde selectie
        sq, name, _ = learning.profile_key(store, p)
        same = [p["id"]] + ([r["id"] for r in store.all(
            "SELECT pl.id FROM players pl JOIN matches m ON m.id = pl.match_id WHERE lower(trim(pl.name)) = ? AND "
            "((pl.team = 0 AND m.team0_squad = ?) OR (pl.team = 1 AND m.team1_squad = ?)) AND pl.id != ?",
            (name, sq, sq, p["id"]))] if sq else [])
        q = ",".join("?" * len(same))
        rows = store.all(f"SELECT e.emb, t.n_frames FROM tracks t JOIN track_embeds e ON e.clip_id = t.clip_id "
                         f"AND e.track_id = t.track_id WHERE t.player_id IN ({q})", tuple(same))
        if not rows:
            continue
        w = np.array([min(r["n_frames"] or 1, 200) for r in rows], float)
        v = np.average([learning.from_blob(r["emb"]) for r in rows], axis=0, weights=w)
        v = v / (np.linalg.norm(v) or 1.0)
        learning.save_profile(store, p, v, len(rows))
        made.append(p["name"])
    prog(1.0, "Klaar")
    found = learning.recognize(store, match_id)
    if only:
        found = [r for r in found if r["player_id"] == int(only)]
    if not made:
        return {"ok": False, "message": "Koppel eerst een paar stukken (tracks) aan de speler(s); daarvan leert de app hoe ze eruitzien."}
    return {"ok": True, "message": f"Profiel gemaakt voor {', '.join(made[:6])}{' en anderen' if len(made) > 6 else ''}. "
                                   f"{len(found)} ongekoppelde stukken lijken op een speler.", "n_found": len(found)}


JOBS = {"detector": train_detector, "field": train_field, "players": train_players}


def main(argv: list[str]) -> int:
    kind, run = argv[0], Path(argv[1])
    opts = json.loads(argv[2]) if len(argv) > 2 else {}
    run.mkdir(parents=True, exist_ok=True)
    store = Store(config.DB_PATH)
    try:
        res = JOBS[kind](store, run, opts)
    except Exception as e:  # noqa: BLE001
        (run / "error.txt").write_text(traceback.format_exc())
        res = {"ok": False, "message": f"Er ging iets mis: {e}"}
    result(run, res)
    return 0 if res.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
