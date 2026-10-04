"""Trainen starten, volgen en stoppen vanuit de app (het trainen zelf draait in app/train.py,
in een apart proces)."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from . import config, learning

LABELS = {"detector": "Spelers en bal herkennen", "field": "Het veld herkennen", "players": "Spelers herkennen"}


class Trainer:
    """Er traint steeds hoogstens één ding tegelijk (het kost veel rekenkracht)."""

    def __init__(self):
        self.proc: subprocess.Popen | None = None
        self.run: Path | None = None
        self.kind: str | None = None
        self.opts: dict = {}
        self.started = 0.0
        self.last: dict | None = None  # resultaat van de vorige training
        self.on_done = []  # functies die na afloop worden aangeroepen (bijv. caches legen)
        self._lock = threading.Lock()

    def busy(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, kind: str, opts: dict) -> dict:
        with self._lock:
            if self.busy():
                raise RuntimeError(f"Er wordt al getraind: {LABELS.get(self.kind, self.kind)}")
            run = config.DATA_DIR / "training" / f"{kind}-{time.strftime('%Y%m%d-%H%M%S')}"
            run.mkdir(parents=True, exist_ok=True)
            env = {**os.environ, "MATCH_ANALYZER_DATA": str(config.DATA_DIR)}
            log = (run / "log.txt").open("w")
            self.proc = subprocess.Popen([sys.executable, "-m", "app.train", kind, str(run), json.dumps(opts)],
                                         cwd=str(config.ROOT), env=env, stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True)
            self.run, self.kind, self.opts, self.started, self.last = run, kind, opts, time.time(), None
            threading.Thread(target=self._wait, args=(self.proc, run, kind), daemon=True).start()
            return self.status()

    def _wait(self, proc: subprocess.Popen, run: Path, kind: str) -> None:
        proc.wait()
        res = _read(run / "result.json") or {"ok": False, "message": "Gestopt"}
        self.last = {**res, "kind": kind, "finished": time.time()}
        for f in self.on_done:
            try:
                f(kind, res)
            except Exception:  # noqa: BLE001
                pass

    def stop(self) -> None:
        if self.busy():
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                self.proc.terminate()

    def status(self) -> dict:
        if self.busy():
            p = _read(self.run / "progress.json") or {"phase": "Starten", "frac": 0.0}
            return {"busy": True, "kind": self.kind, "label": LABELS.get(self.kind), "opts": self.opts, **p}
        return {"busy": False, "last": self.last}


def _read(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def overview(store) -> dict:
    """Wat er klaarstaat om van te leren, hoe lang het ongeveer duurt, en welke modellen er zijn."""
    from .train import detector_plan, field_plan, players_plan

    clips = learning.approved_clips(store)
    n_det = len(learning.detector_examples(store)) if clips else 0
    n_field = len(learning.field_examples(store)) if clips else 0
    reg = learning.registry()
    out = {"approved": [{"id": c["id"], "filename": c["filename"], "match_id": c["match_id"]} for c in clips],
           "device": learning.device()}
    for kind, n, plan in (("detector", n_det, detector_plan), ("field", n_field, field_plan)):
        out[kind] = {"examples": n, "quick": plan(n, True), "thorough": plan(n, False),
                     "active": reg[kind]["active"], "versions": list(reversed(reg[kind]["versions"]))}
    out["matches"] = [{"id": m["id"], "name": m["name"], "n_linked": m["n_linked"], **players_plan(store, m["id"])}
                      for m in store.all("SELECT m.id, m.name, (SELECT COUNT(*) FROM tracks t JOIN clips c ON c.id = t.clip_id "
                                         "WHERE c.match_id = m.id AND t.player_id IS NOT NULL) AS n_linked "
                                         "FROM matches m ORDER BY COALESCE(m.date, m.created_at) DESC")]
    return out
