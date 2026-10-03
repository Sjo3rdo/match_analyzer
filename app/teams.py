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
    if h < 20 or w < 8:
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


_W = np.array([0.5, 1.0, 1.0])  # Lab-weging: helderheid telt minder (zon/schaduw)


def assign_teams(colors: np.ndarray, n_frames: np.ndarray) -> np.ndarray:
    """Teamlabel per track: 0, 1 of TEAM_OTHER.

    k-means met meer groepen dan nodig (teams, scheidsrechter, keepers, publiek). De twee
    groepen met de meeste speeltijd zijn de teams; een track hoort bij het dichtstbijzijnde
    team als zijn kleur daar duidelijk genoeg op lijkt, anders is hij 'overig'.
    """
    colors = np.asarray(colors, dtype=np.float64)
    n = len(colors)
    if n == 0:
        return np.zeros(0, dtype=int)
    if n < 3:
        return np.zeros(n, dtype=int)
    w = np.asarray(n_frames, dtype=np.float64)
    x = colors * _W
    k = min(5, n)
    best, best_cost, best_c = None, np.inf, None
    for seed in range(10):  # meerdere starts: k-means kan in een slecht lokaal optimum belanden
        labels, centers = kmeans(x, k, weights=w, seed=seed)
        cost = float((w * ((x - centers[labels]) ** 2).sum(1)).sum())
        if cost < best_cost:
            best, best_cost, best_c = labels, cost, centers
    size = np.array([w[best == j].sum() for j in range(k)])
    t0, t1 = np.argsort(-size)[:2]
    team_c = best_c[[t0, t1]]
    sep = np.linalg.norm(team_c[0] - team_c[1])
    # overige groepen: bij een team als ze er duidelijk op lijken (bijv. hetzelfde shirt in de
    # schaduw), anders scheidsrechter/keeper/publiek
    mapping = {}
    for j in range(k):
        dj = np.linalg.norm(best_c[j] - team_c, axis=1)
        mapping[j] = int(np.argmin(dj)) if dj.min() < 0.45 * sep else TEAM_OTHER
    mapping[t0], mapping[t1] = 0, 1
    return np.array([mapping[l] for l in best])
