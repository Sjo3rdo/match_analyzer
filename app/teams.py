"""Teamindeling op shirtkleur.

Per detectie nemen we de romp (bovenste helft van de box), gooien groene graspixels weg
en bewaren de mediaankleur in Lab-kleurruimte. Per track middelen we die kleuren, en
k-means met k=3 groepeert de tracks: de twee grootste groepen zijn de teams, de rest
(scheidsrechter, keepers) heet 'overig'. De gebruiker kan alles corrigeren.
"""
from __future__ import annotations

import cv2
import numpy as np

TEAM_OTHER = 2
TEAM_SPECTATOR = 3  # door de gebruiker aangewezen als toeschouwer: telt nergens mee
TEAM_UNKNOWN = -1


def _surroundings(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    """Pixels rondom de speler: links en rechts op romphoogte, en de grond onder de voeten."""
    H, W = frame.shape[:2]
    h, w = y2 - y1, x2 - x1
    r0, r1 = y1 + int(0.15 * h), y1 + int(0.5 * h)
    pad = max(3, int(0.6 * w))
    parts = [frame[max(0, r0):r1, max(0, x1 - pad):max(0, x1 - 2)],
             frame[max(0, r0):r1, min(W, x2 + 2):min(W, x2 + pad)],
             frame[min(H, y2 + 1):min(H, y2 + max(3, int(0.2 * h))), max(0, x1 - pad // 2):min(W, x2 + pad // 2)]]
    pix = [p.reshape(-1, 3) for p in parts if p.size]
    return np.vstack(pix) if pix else np.zeros((0, 3), np.uint8)


def shirt_color(frame: np.ndarray, box: np.ndarray) -> np.ndarray | None:
    """Mediaankleur (Lab) van het shirt, of None als er te weinig bruikbare pixels zijn.

    We houden alleen rompixels over die duidelijk afwijken van de directe omgeving van de
    speler (gras, bomen, hek, lucht). Een vaste "groen = gras"-regel werkt niet: dan
    verdwijnen groene shirts, en gele shirts lijken soms sprekend op zonbeschenen gras.
    """
    x1, y1, x2, y2 = [int(v) for v in box]
    h, w = y2 - y1, x2 - x1
    if h < 12 or w < 5:  # ook verre spelers (ca. 15 px hoog) krijgen zo een kleur
        return None
    crop = frame[max(0, y1 + int(0.15 * h)):max(0, y1 + int(0.5 * h)),
                 max(0, x1 + int(0.15 * w)):max(0, x2 - int(0.15 * w))]
    if crop.size == 0:
        return None
    n_px = crop.shape[0] * crop.shape[1]
    if n_px > 160:  # grote (dichtbije) spelers: verkleinen, dat scheelt veel rekenwerk
        f = np.sqrt(160 / n_px)
        crop = cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    around = _surroundings(frame, x1, y1, x2, y2)
    if len(around) >= 10:
        if len(around) > 150:
            around = around[np.linspace(0, len(around) - 1, 150).astype(int)]
        bg = cv2.cvtColor(around.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
        wts = np.array([0.5, 1.0, 1.0], np.float32)  # helderheid telt minder (schaduw/zon)
        d2 = (((lab[:, None, :] - bg[None]) * wts) ** 2).sum(-1).min(axis=1)
        lab = lab[d2 > 81]  # afstand > 9
    if len(lab) < max(8, 0.1 * crop.shape[0] * crop.shape[1]):
        return None
    return np.median(lab, axis=0).astype(np.float32)


def lab_to_hex(lab: np.ndarray) -> str:
    px = np.uint8([[np.clip(lab, 0, 255)]])
    b, g, r = cv2.cvtColor(px, cv2.COLOR_LAB2BGR)[0, 0]
    return f"#{r:02x}{g:02x}{b:02x}"


def kmeans(x: np.ndarray, k: int, weights: np.ndarray | None = None, iters: int = 50,
           seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Gewogen k-means (k-means++ start). Geeft (labels, centra)."""
    rng = np.random.default_rng(seed)
    n = len(x)
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=np.float64)
    centers = [x[rng.choice(n, p=w / w.sum())]]
    for _ in range(1, k):
        d = np.min([np.sum((x - c) ** 2, axis=1) for c in centers], axis=0) * w
        centers.append(x[rng.choice(n, p=d / d.sum())] if d.sum() > 0 else x[rng.integers(n)])
    c = np.array(centers, dtype=np.float64)
    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        labels = np.argmin(((x[:, None, :] - c[None]) ** 2).sum(-1), axis=1)
        new = np.array([np.average(x[labels == j], axis=0, weights=w[labels == j])
                        if np.any(labels == j) else c[j] for j in range(k)])
        if np.allclose(new, c):
            break
        c = new
    return labels, c


# Lab-weging. OpenCV-Lab heeft helderheid op een schaal van 0..255 en kleur rond 128; de
# helderheid verschilt binnen één team enorm (zon tegenover schaduw: een donkerblauw shirt in de
# zon is bijna grijs), de kleurtint veel minder. Daarom telt helderheid maar licht mee: genoeg om
# wit en zwart uit elkaar te houden, niet zoveel dat zon en schaduw twee 'teams' worden.
_W = np.array([0.15, 1.0, 1.0])
JOIN = 0.4  # hoe dicht een kleur bij een team moet liggen (deel van de afstand tussen de teams)


def hex_to_lab(color: str | None) -> np.ndarray | None:
    """'#rrggbb' -> Lab (OpenCV-schaal); terug van lab_to_hex."""
    if not color or len(color) != 7:
        return None
    bgr = np.uint8([[[int(color[5:7], 16), int(color[3:5], 16), int(color[1:3], 16)]]])
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[0, 0].astype(np.float64)


def color_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm((np.asarray(a, float) - np.asarray(b, float)) * _W))


def team_centers(colors: np.ndarray, weights: np.ndarray, labels: np.ndarray) -> list[np.ndarray | None]:
    """Gewogen gemiddelde kleur van team 0 en team 1 (None als een team leeg is)."""
    out = []
    for team in (0, 1):
        m = labels == team
        out.append(np.average(colors[m], axis=0, weights=weights[m]) if m.any() and weights[m].sum() > 0 else None)
    return out


def assign_teams(colors: np.ndarray, n_frames: np.ndarray, exclude: np.ndarray | None = None,
                 anchors: np.ndarray | None = None) -> np.ndarray:
    """Teamlabel per track: 0, 1 of TEAM_OTHER.

    1. Toeschouwers (`exclude`, bijv. wie stilstaat) doen niet mee: anders vormen donkere jassen
       langs de lijn een eigen 'team'.
    2. k-means met meer groepen dan nodig (twee teams, maar ook zon/schaduw-varianten,
       scheidsrechter, keepers).
    3. De teams zijn het paar groepen dat groot is én duidelijk van kleur verschilt (twee
       tinten van hetzelfde groene shirt zijn geen twee teams).
    4. Elke track gaat naar het team waar zijn kleur het dichtst bij ligt, als die duidelijk
       genoeg lijkt; anders 'overig' (scheidsrechter, keeper, publiek).

    anchors: per track het team dat de gebruiker zelf heeft gekozen (0/1), of -1. Heeft de
    gebruiker in beide teams iemand ingedeeld, dan zijn dat de teamkleuren (leren van correcties)
    in plaats van wat k-means vindt.
    """
    colors = np.asarray(colors, dtype=np.float64).reshape(-1, 3)
    n = len(colors)
    out = np.full(n, TEAM_OTHER, dtype=int)
    if n == 0:
        return np.zeros(0, dtype=int)
    use = np.ones(n, bool) if exclude is None else ~np.asarray(exclude, bool)
    if use.sum() < 3:
        out[use] = 0
        return out
    w = np.asarray(n_frames, dtype=np.float64)[use]
    w = np.where(w > 0, w, 1.0)
    x = colors[use] * _W
    anc = np.full(n, -1) if anchors is None else np.asarray(anchors, int)
    if all(np.any(anc == t) for t in (0, 1)):
        aw = np.where(np.asarray(n_frames, float) > 0, np.asarray(n_frames, float), 1.0)
        team_c = np.array([np.average(colors[anc == t] * _W, axis=0, weights=aw[anc == t]) for t in (0, 1)])
        return _assign_to(colors, use, team_c, x, w, anc)
    k = min(6, len(x))
    best = None
    for seed in range(10):  # meerdere starts: k-means kan in een slecht lokaal optimum belanden
        labels, centers = kmeans(x, k, weights=w, seed=seed)
        cost = float((w * ((x - centers[labels]) ** 2).sum(1)).sum())
        if best is None or cost < best[0]:
            best = (cost, labels, centers)
    _, labels, c = best
    size = np.array([w[labels == j].sum() for j in range(k)])
    pair, score = (0, 1), -1.0
    for i in range(k):
        for j in range(i + 1, k):
            s = min(size[i], size[j]) * min(1.0, np.linalg.norm(c[i] - c[j]) / 30.0)
            if s > score:
                pair, score = (i, j), s
    team_c = c[list(pair)]
    if size[pair[1]] > size[pair[0]]:  # team 0 = de grootste groep
        team_c = team_c[::-1]
    return _assign_to(colors, use, team_c, x, w, anc)


def _assign_to(colors, use, team_c, x, w, anc) -> np.ndarray:
    """Teamcentra verfijnen en elke track bij het dichtstbijzijnde team indelen (of 'overig')."""
    # De centra zijn nu die van twee (deel)groepen; verfijn ze met alle tracks die er duidelijk
    # bij horen, zodat een team dat in zon en schaduw is gesplitst één gemiddelde kleur krijgt.
    for _ in range(2):
        sep = np.linalg.norm(team_c[0] - team_c[1])
        d = np.linalg.norm(x[:, None, :] - team_c[None], axis=2)
        lab = np.where(d.min(axis=1) < JOIN * sep, np.argmin(d, axis=1), -1)
        lab = np.where(anc[use] >= 0, anc[use], lab)  # wat de gebruiker koos telt altijd mee
        new_c = [np.average(x[lab == t], axis=0, weights=w[lab == t]) if np.any(lab == t) else team_c[t]
                 for t in (0, 1)]
        team_c = np.array(new_c)
    sep = np.linalg.norm(team_c[0] - team_c[1])
    # Wie stilstond telde niet mee voor de teamkleuren, maar kan best een speler zijn (een keeper,
    # een verdediger die even wacht): die hoort bij een team als zijn shirt er sprekend op lijkt.
    d = np.linalg.norm((colors * _W)[:, None, :] - team_c[None], axis=2)
    near, far = d.min(axis=1), d.max(axis=1)
    limit = np.where(use, JOIN, 0.35) * sep
    # Iets verder van het teamgemiddelde (fel zonlicht, ver weg) maar overduidelijk niet het andere
    # team: hoort er ook bij. (Een scheidsrechter in het zwart zit dichter bij het donkerste team,
    # maar niet drie keer zo dicht als bij het andere.)
    clear = (near < 0.45 * sep) & (far > 3.0 * near)
    return np.where((near < limit) | clear, np.argmin(d, axis=1), TEAM_OTHER).astype(int)
