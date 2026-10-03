"""De bal volgen tijdens de analyse.

Een bal is klein (vaak maar 6-15 pixels), beweegt snel en verdwijnt achter spelers. Het
detectiemodel ziet hem daardoor lang niet in elk beeld. Metafoor: een vogelaar die een vogel even
kwijt is, kijkt met de verrekijker eerst op de plek waar de vogel heen vloog, en pas als hij hem
dan nog niet ziet, zoekt hij de hele lucht af. Zo doen wij het ook:

1. Per beeld kijken we naar alle kandidaten die het model vindt (ook twijfelachtige), en kiezen we
   de kandidaat die past bij waar de bal net was en heen ging; een losse bal ergens anders (een
   reservebal, een witte schoen) telt dan niet.
2. Ziet het model de bal niet: inzoomen op de plek waar hij nu zou moeten zijn (uitsnede op volle
   resolutie: de bal is daar anderhalf tot twee keer zo groot als in het gewone detectiebeeld).
3. Lang kwijt: af en toe het speelveld in stukken op volle resolutie afzoeken.

Wat de gebruiker zelf aanwijst (video-scherm: "⚽ Bal aanwijzen") gaat altijd voor.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CROP = 640  # uitsnede rond de verwachte plek (pixels op volle resolutie)
MIN_CONF = 0.05  # zo laag mag een kandidaat zijn als hij precies op het spoor ligt
NEW_CONF = 0.12  # zonder spoor moet het model zekerder zijn
MAX_LOST = 25  # zo lang (beelden) blijven we inzoomen op de verwachte plek
SCAN_EVERY = 5  # bal lang kwijt: elk zoveelste beeld het veld afzoeken


@dataclass
class Candidate:
    x: float
    y: float
    conf: float
    size: float  # diameter in pixels


class BallTracker:
    """Houdt bij waar de bal was en heen gaat (in beeldpixels)."""

    def __init__(self, width: int, height: int):
        self.W, self.H = width, height
        self.pos: np.ndarray | None = None
        self.vel = np.zeros(2)
        self.size = 12.0
        self.lost = 10**6  # beelden sinds de bal voor het laatst gezien is
        self.n_seen = 0

    def predicted(self) -> np.ndarray | None:
        if self.pos is None or self.lost > MAX_LOST:
            return None
        k = min(self.lost + 1, 8)  # niet eindeloos doorrekenen: een bal remt af of wordt geraakt
        p = self.pos + self.vel * k
        return np.clip(p, [0, 0], [self.W - 1, self.H - 1])

    def gate(self) -> float:
        """Zoekstraal rond de verwachte plek: groeit hoe langer de bal weg is."""
        return 40.0 + 6.0 * self.size ** 0.5 + 30.0 * min(self.lost, 10)

    def choose(self, cands: list[Candidate]) -> Candidate | None:
        if not cands:
            return None
        p = self.predicted()
        if p is None:
            best = max(cands, key=lambda c: c.conf)
            return best if best.conf >= NEW_CONF else None
        r = self.gate()
        scored = []
        for c in cands:
            d = float(np.hypot(c.x - p[0], c.y - p[1]))
            if d <= r:
                scored.append((c.conf * np.exp(-d / r), c))
            elif c.conf >= 0.45:  # heel zeker ergens anders: de bal is weggeschoten of we zaten fout
                scored.append((0.5 * c.conf * np.exp(-d / (4 * r)), c))
        if not scored:
            return None
        return max(scored, key=lambda s: s[0])[1]

    def update(self, c: Candidate | None) -> None:
        if c is None:
            self.lost += 1
            if self.lost > MAX_LOST:
                self.vel[:] = 0
            return
        p = np.array([c.x, c.y])
        if self.pos is not None and self.lost <= 5:
            v = (p - self.pos) / (self.lost + 1)
            self.vel = 0.5 * self.vel + 0.5 * v
        else:
            self.vel[:] = 0
        self.pos = p
        self.size = 0.7 * self.size + 0.3 * max(4.0, c.size)
        self.lost = 0
        self.n_seen += 1

    def crop_box(self) -> tuple[int, int, int, int] | None:
        """Uitsnede (x0, y0, x1, y1) rond de verwachte plek, of None."""
        p = self.predicted()
        if p is None:
            return None
        s = min(CROP, self.W, self.H)
        x0 = int(np.clip(p[0] - s / 2, 0, self.W - s))
        y0 = int(np.clip(p[1] - s / 2, 0, self.H - s))
        return x0, y0, x0 + s, y0 + s

    def scan_boxes(self, frame_no: int) -> list[tuple[int, int, int, int]]:
        """Bal lang kwijt: af en toe het beeld in vakken afzoeken (de bovenste rand is meestal lucht
        of publiek; die slaan we over)."""
        if self.lost <= MAX_LOST or frame_no % SCAN_EVERY:
            return []
        s = min(CROP, self.H)
        y0 = max(0, int(self.H * 0.25))
        rows = sorted({min(y0, self.H - s), self.H - s})
        n = int(np.ceil(self.W / s))
        xs = np.linspace(0, self.W - s, n).astype(int) if n > 1 else [0]
        return [(int(x), int(y), int(x) + s, int(y) + s) for y in rows for x in xs]


def static_runs(idx: np.ndarray, pos: np.ndarray, conf: np.ndarray, A: np.ndarray, size: float = 12.0,
                min_count: int = 8, min_span: int = 20, max_gap: int = 12, max_conf: float = 0.35) -> np.ndarray:
    """Welke balposities eigenlijk een vast ding in de achtergrond zijn (een rond logo op een
    reclamebord, een witte paal): ze liggen seconden lang op dezelfde plek ten opzichte van de
    achtergrond, terwijl de camera beweegt, en het model is er niet zeker van. Een bal die stil
    ligt (vrije trap) ziet het model meestal wel zeker; die blijft staan.

    idx: beeldnummers (oplopend), pos: (n, 2) beeldpixels, A: beeld -> eerste beeld (cumulatief).
    min_span/max_gap in beelden. Geeft een masker: True = weggooien."""
    n = len(idx)
    out = np.zeros(n, bool)
    if n < min_count or len(A) == 0:
        return out
    try:
        Ainv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return out

    def warp(i_from, i_to, pt):  # beeld(i_from) -> beeld(i_to), via de camerabeweging
        q = Ainv[i_to] @ A[i_from] @ np.array([pt[0], pt[1], 1.0])
        return q[:2] / q[2] if abs(q[2]) > 1e-9 else np.array([np.nan, np.nan])

    # per positie: ligt hij (via de camerabeweging teruggerekend) op een plek in de achtergrond waar
    # kort daarvoor ook al een "bal" lag? Een rij gelijke logo's geeft zo ook steeds herhalingen.
    tol = max(4.0, 0.6 * size)
    static = np.zeros(n, bool)
    for k in range(n):
        if idx[k] >= len(A):
            continue
        for j in range(k - 1, -1, -1):
            if idx[k] - idx[j] > 30:
                break
            if idx[j] < len(A) and idx[k] - idx[j] >= 2 and np.linalg.norm(warp(idx[j], idx[k], pos[j]) - pos[k]) < tol:
                static[k] = True
                break
    # stukken waarin bijna alles zo'n herhaling is, lang genoeg en met weinig zekerheid: weg
    start = 0
    for k in range(1, n + 1):
        if k < n and idx[k] - idx[k - 1] <= max_gap:
            continue
        seg = np.arange(start, k)
        # binnen het stuk de langste reeks waar (bijna) alles stil ligt
        run_start = None
        for m in list(seg) + [None]:
            ok = m is not None and (static[m] or (m + 1 < k and static[m + 1]))
            if ok and run_start is None:
                run_start = m
            elif not ok and run_start is not None:
                r = np.arange(run_start, m if m is not None else k)
                if (len(r) >= min_count and idx[r[-1]] - idx[r[0]] >= min_span
                        and float(np.mean(conf[r])) < max_conf):
                    out[r] = True
                run_start = None
        start = k
    return out
