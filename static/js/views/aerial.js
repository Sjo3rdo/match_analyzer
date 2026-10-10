// Luchtfoto van PDOK: de 4 hoekvlaggen aanklikken (-> veldmaten en ligging), daarna je eigen plek.
// Net een plattegrond op tafel: je wijst de hoeken van het veld aan en zet je pion waar je stond.
// De browser haalt alleen de kaartstukjes rond het veld op bij PDOK; je video's blijven op je computer.
import { api, h, toast } from '../util.js';
import { fitHomography, apply } from '../homography.js';
import { pitchPolylines, setPitchSize, L as PL, W as PW } from '../pitch.js';

const TILE = 256, MIN_Z = 15, MAX_Z = 21;
const worldSize = z => TILE * 2 ** z;
const toWorld = ([lat, lon], z) => {
  const s = Math.sin(lat * Math.PI / 180);
  return [(lon + 180) / 360 * worldSize(z), (0.5 - Math.log((1 + s) / (1 - s)) / (4 * Math.PI)) * worldSize(z)];
};
const toLatLon = ([x, y], z) => {
  const n = Math.PI - 2 * Math.PI * y / worldSize(z);
  return [180 / Math.PI * Math.atan(Math.sinh(n)), x / worldSize(z) * 360 - 180];
};
const metersPerPx = (lat, z) => 156543.03392 * Math.cos(lat * Math.PI / 180) / 2 ** z;

// Zelfde volgorde als de server (geo.order_corners): linksonder, rechtsonder, rechtsboven, linksboven,
// gezien vanaf jouw kant. Voor de voorvertoning; bij opslaan rekent de server het opnieuw uit.
function orderCorners(corners, near) {
  const z = 20, w = corners.map(c => toWorld(c, z));
  const cx = w.reduce((s, p) => s + p[0], 0) / 4, cy = w.reduce((s, p) => s + p[1], 0) / 4;
  const loc = w.map(([x, y]) => [x - cx, cy - y]);  // oost, noord
  const order = [0, 1, 2, 3].sort((a, b) => Math.atan2(loc[a][1], loc[a][0]) - Math.atan2(loc[b][1], loc[b][0]));
  const pts = order.map(i => loc[i]);
  const side = i => Math.hypot(pts[(i + 1) % 4][0] - pts[i][0], pts[(i + 1) % 4][1] - pts[i][1]);
  const longs = side(0) + side(2) >= side(1) + side(3) ? [0, 2] : [1, 3];
  let me;
  if (near) { const [x, y] = toWorld(near, z); me = [x - cx, cy - y]; } else me = [(loc[0][0] + loc[1][0]) / 2, (loc[0][1] + loc[1][1]) / 2];
  const mid = i => [(pts[i][0] + pts[(i + 1) % 4][0]) / 2, (pts[i][1] + pts[(i + 1) % 4][1]) / 2];
  const i = longs.reduce((b, k) => (Math.hypot(mid(k)[0] - me[0], mid(k)[1] - me[1]) < Math.hypot(mid(b)[0] - me[0], mid(b)[1] - me[1]) ? k : b));
  const ordered = [0, 1, 2, 3].map(k => corners[order[(i + k) % 4]]);
  const m = metersPerPx(ordered[0][0], z), d = (a, b) => {
    const [ax, ay] = toWorld(ordered[a], z), [bx, by] = toWorld(ordered[b], z); return Math.hypot(ax - bx, ay - by) * m;
  };
  return { ordered, length: (d(0, 1) + d(3, 2)) / 2, width: (d(0, 3) + d(1, 2)) / 2 };
}

export function aerialPanel({ match, clip = null, onCorners = () => {}, onCamera = () => {} }) {
  const canvas = h('canvas', { style: { width: '100%', height: '440px', display: 'block', borderRadius: '10px', cursor: 'crosshair', background: '#1d2622', touchAction: 'none' } });
  const status = h('div', { style: { marginTop: '8px' } });
  const el = h('div', { className: 'aerial' },
    h('div', { style: { position: 'relative' } }, canvas,
      h('div', { style: { position: 'absolute', right: '8px', top: '8px', display: 'flex', flexDirection: 'column', gap: '4px' } },
        h('button', { className: 'small', title: 'Inzoomen', onclick: () => zoomAt(cssW / 2, cssH / 2, 1) }, '+'),
        h('button', { className: 'small', title: 'Uitzoomen', onclick: () => zoomAt(cssW / 2, cssH / 2, -1) }, '−')),
      h('div', { className: 'small', style: { position: 'absolute', left: '8px', bottom: '6px', color: '#fff', textShadow: '0 1px 2px #000' } },
        'Luchtfoto: PDOK / Beeldmateriaal Nederland')),
    status);

  let info = null, z = 19, center = null;  // center: wereldpixels op zoom z
  let corners = [], saved = false, known = null, size = null, mode = 'hoeken';
  let cssW = 0, cssH = 0, down = null, dragCorner = -1, alive = true;
  const tiles = new Map();

  function tile(zz, x, y) {
    const key = `${zz}/${x}/${y}`;
    let t = tiles.get(key);
    if (!t) {
      t = { img: new Image(), ok: false, failed: false };
      t.img.onload = () => { t.ok = true; draw(); };
      t.img.onerror = () => { t.failed = true; draw(); };
      t.img.src = info.tiles.replace('{z}', zz).replace('{x}', x).replace('{y}', y);
      tiles.set(key, t);
    }
    return t;
  }
  const toScreen = ll => { const [x, y] = toWorld(ll, z); return [x - center[0] + cssW / 2, y - center[1] + cssH / 2]; };
  const fromScreen = (sx, sy) => toLatLon([sx - cssW / 2 + center[0], sy - cssH / 2 + center[1]], z);
  const gps = () => (clip?.gps_lat != null ? [clip.gps_lat, clip.gps_lon] : info?.center);

  // veld -> scherm, als de 4 hoeken er zijn (voor de lijnen en je plek)
  function pitchToScreen() {
    if (corners.length !== 4) return null;
    const o = saved ? { ordered: corners, ...size } : orderCorners(corners, info.center);
    const L = o.length, W = o.width;
    const H = fitHomography([[0, W], [L, W], [L, 0], [0, 0]], o.ordered.map(toScreen));
    return H ? { H, L, W, ordered: o.ordered } : null;
  }

  function draw() {
    if (!alive || !info || !center) return;
    const dpr = window.devicePixelRatio || 1;
    cssW = canvas.clientWidth || 700; cssH = canvas.clientHeight || 440;
    if (canvas.width !== Math.round(cssW * dpr)) { canvas.width = Math.round(cssW * dpr); canvas.height = Math.round(cssH * dpr); }
    const c = canvas.getContext('2d');
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    c.fillStyle = '#1d2622'; c.fillRect(0, 0, cssW, cssH);
    const x0 = Math.floor((center[0] - cssW / 2) / TILE), x1 = Math.floor((center[0] + cssW / 2) / TILE);
    const y0 = Math.floor((center[1] - cssH / 2) / TILE), y1 = Math.floor((center[1] + cssH / 2) / TILE);
    for (let tx = x0; tx <= x1; tx++) for (let ty = y0; ty <= y1; ty++) {
      const sx = tx * TILE - center[0] + cssW / 2, sy = ty * TILE - center[1] + cssH / 2;
      const t = tile(z, tx, ty);
      if (t.ok) { c.drawImage(t.img, sx, sy, TILE, TILE); continue; }
      for (let up = 1; up <= 3; up++) {  // nog niet binnen (of bestaat niet zo scherp): grover stuk uitvergroten
        const k = 2 ** up, px = Math.floor(tx / k), py = Math.floor(ty / k);
        const p = tiles.get(`${z - up}/${px}/${py}`) ?? (t.failed ? tile(z - up, px, py) : null);
        if (p?.ok) {
          const part = TILE / k, ox = (tx % k) * part, oy = (ty % k) * part;
          c.drawImage(p.img, ox, oy, part, part, sx, sy, TILE, TILE); break;
        }
      }
    }
    const P = pitchToScreen();
    if (P) {  // de veldlijnen zoals ze bij deze hoeken horen: liggen ze op de lijnen van de foto?
      const was = [PL, PW];  // de veldtekening elders in de app niet veranderen
      setPitchSize(P.L, P.W); const lines = pitchPolylines(); setPitchSize(...was);
      c.save(); c.strokeStyle = saved ? 'rgba(255,255,255,.9)' : 'rgba(255,214,0,.95)'; c.lineWidth = 1.5;
      for (const line of lines) {
        c.beginPath(); line.forEach((p, i) => { const [x, y] = apply(P.H, p); i ? c.lineTo(x, y) : c.moveTo(x, y); }); c.stroke();
      }
      c.restore();
    } else if (corners.length > 1) {
      c.save(); c.strokeStyle = '#ffd600'; c.lineWidth = 2; c.setLineDash([6, 4]); c.beginPath();
      corners.forEach((ll, i) => { const [x, y] = toScreen(ll); i ? c.lineTo(x, y) : c.moveTo(x, y); }); c.stroke(); c.restore();
    }
    if (mode === 'hoeken') corners.forEach((ll, i) => {
      const [x, y] = toScreen(ll);
      c.beginPath(); c.arc(x, y, 9, 0, 2 * Math.PI); c.fillStyle = 'rgba(255,214,0,.9)'; c.fill();
      c.lineWidth = 2; c.strokeStyle = '#000'; c.stroke();
      c.fillStyle = '#000'; c.font = 'bold 11px system-ui'; c.textAlign = 'center'; c.textBaseline = 'middle'; c.fillText(String(i + 1), x, y + 0.5);
    });
    const g = gps();
    if (g && clip) {  // waar de GPS van deze video je zet, met de onzekerheid
      const [x, y] = toScreen(g), r = Math.max(5, clip.gps_acc || 8) / metersPerPx(g[0], z);
      c.beginPath(); c.arc(x, y, r, 0, 2 * Math.PI); c.fillStyle = 'rgba(77,181,255,.18)'; c.fill();
      c.strokeStyle = 'rgba(77,181,255,.8)'; c.lineWidth = 1; c.stroke();
      c.beginPath(); c.arc(x, y, 4, 0, 2 * Math.PI); c.fillStyle = '#4db5ff'; c.fill();
    }
    if (P && saved && clip?.cam_x != null) {
      const [x, y] = apply(P.H, [clip.cam_x, clip.cam_y]);
      c.font = '22px system-ui'; c.textAlign = 'center'; c.textBaseline = 'bottom'; c.fillText('📍', x, y + 3);
    }
  }

  function zoomAt(sx, sy, dz) {
    const nz = Math.min(MAX_Z, Math.max(MIN_Z, z + dz));
    if (nz === z) return;
    const ll = fromScreen(sx, sy), w = toWorld(ll, nz);
    center = [w[0] - (sx - cssW / 2), w[1] - (sy - cssH / 2)]; z = nz; draw();
  }
  function pos(e) { const r = canvas.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; }
  canvas.addEventListener('wheel', e => { e.preventDefault(); const [x, y] = pos(e); zoomAt(x, y, e.deltaY < 0 ? 1 : -1); }, { passive: false });
  canvas.addEventListener('pointerdown', e => {
    if (!info) return;
    const [x, y] = pos(e);
    dragCorner = mode === 'hoeken' ? corners.findIndex(ll => { const [cx, cy] = toScreen(ll); return Math.hypot(cx - x, cy - y) < 12; }) : -1;
    down = { x, y, cx: center[0], cy: center[1], moved: false };
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener('pointermove', e => {
    if (!down) return;
    const [x, y] = pos(e);
    if (Math.hypot(x - down.x, y - down.y) > 4) down.moved = true;
    if (!down.moved) return;
    if (dragCorner >= 0) { corners[dragCorner] = fromScreen(x, y); if (saved) saved = false; draw(); renderStatus(); return; }
    center = [down.cx - (x - down.x), down.cy - (y - down.y)]; draw();
  });
  canvas.addEventListener('pointerup', async e => {
    if (!down) return;
    const [x, y] = pos(e), wasClick = !down.moved, dc = dragCorner;
    down = null; dragCorner = -1;
    if (!wasClick || dc >= 0) return;
    if (mode === 'hoeken') {
      if (corners.length >= 4) return toast('Er staan al 4 hoeken. Sleep een hoek om hem te verplaatsen, of begin opnieuw.');
      corners.push(fromScreen(x, y)); known = null; draw(); renderStatus();
    } else if (mode === 'plek' && clip) {
      const [lat, lon] = fromScreen(x, y);
      const c = await api(`/clips/${clip.id}/camera/latlon`, { json: { lat, lon } }).catch(() => null);
      if (!c) return;
      Object.assign(clip, c); draw(); renderStatus(); onCamera(clip);
    }
  });
  const onResize = () => draw();
  window.addEventListener('resize', onResize);

  async function saveCorners() {
    const r = await api(`/matches/${match.id}/pitch-corners`, { json: { corners } }).catch(() => null);
    if (!r) return;
    corners = r.corners; size = { length: r.length, width: r.width }; saved = true; known = null; mode = clip ? 'plek' : 'hoeken';
    if (clip) {  // de server rekende je plek uit de GPS uit
      const fresh = (await api(`/matches/${match.id}/aerial`).catch(() => null))?.clips?.find(x => x.id === clip.id);
      if (fresh) Object.assign(clip, fresh);
    }
    setPitchSize(r.length, r.width);
    toast(`Veld vastgelegd: ${r.length} × ${r.width} m` + (r.placed ? ` · plek van ${r.placed} video('s) uitgerekend` : ''));
    draw(); renderStatus(); onCorners(r);
  }

  function renderStatus() {
    const btn = (label, onclick, primary = false) => h('button', { className: primary ? 'primary' : '', onclick }, label);
    const restart = btn('Opnieuw aanklikken', () => { corners = []; saved = false; known = null; mode = 'hoeken'; draw(); renderStatus(); });
    let body;
    if (known && corners.length === 4 && !saved) {
      body = [h('b', {}, `Dit veld ken je al: ${known.length} × ${known.width} m. `), 'Liggen de gele lijnen op de lijnen van de foto?',
        h('div', { className: 'row', style: { marginTop: '6px' } }, btn('✓ Klopt, gebruik dit veld', saveCorners, true), restart)];
    } else if (saved && mode === 'plek') {
      body = [h('b', {}, `✓ Veld vastgelegd: ${size.length} × ${size.width} m. `),
        clip ? (clip.cam_source === 'hand' ? '📍 Je plek staat op de foto. Klik ergens anders om hem te verplaatsen.'
          : 'De blauwe stip is je plek volgens de GPS van de video (de cirkel is de onzekerheid). Klik op de foto waar je precies stond.') : '',
        h('div', { className: 'row', style: { marginTop: '6px' } }, btn('Hoeken aanpassen', () => { mode = 'hoeken'; draw(); renderStatus(); }))];
    } else if (corners.length < 4) {
      body = [h('b', {}, corners.length ? `Nog ${4 - corners.length} hoek${corners.length === 3 ? '' : 'en'}… ` : 'Klik de 4 hoekvlaggen van het veld aan. '),
        info.center ? 'De volgorde maakt niet uit. ' : 'Begin met de 2 hoeken aan jouw kant (eerst links, dan rechts). ',
        'Zoom in met het scrollwiel of +, sleep om te schuiven.',
        corners.length ? h('div', { className: 'row', style: { marginTop: '6px' } }, restart) : null];
    } else {
      const o = orderCorners(corners, info.center), bad = !(o.length >= 40 && o.length <= 130 && o.width >= 25 && o.width <= 100);
      body = [h('b', {}, `Veld: ${o.length.toFixed(1)} × ${o.width.toFixed(1)} m. `),
        bad ? 'Dat lijkt geen voetbalveld; sleep de hoeken naar de hoekvlaggen.' : 'Liggen de gele lijnen op de lijnen van de foto? Sleep anders een hoek een beetje bij.',
        h('div', { className: 'row', style: { marginTop: '6px' } },
          h('button', { className: 'primary', disabled: bad, onclick: saveCorners }, saved ? '✓ Opgeslagen' : '✓ Veld opslaan'),
          saved && clip ? btn('Klaar met hoeken', () => { mode = 'plek'; draw(); renderStatus(); }) : null, restart)];
    }
    status.replaceChildren(h('div', { className: 'hint' }, ...body),
      h('div', { className: 'small muted', style: { marginTop: '4px' } },
        'De luchtfoto komt van PDOK (gratis, van de overheid). Je browser haalt alleen de stukjes foto rond dit veld op; je video\'s blijven op je computer.'));
  }

  (async () => {
    info = await api(`/matches/${match.id}/aerial`).catch(() => null);
    if (!info) return;
    // veld al vastgelegd en een video gekozen: begin bij je plek (dan zet je die zo goed); anders bij het veld
    const mid = info.corners && [info.corners.reduce((s, c) => s + c[0], 0) / 4, info.corners.reduce((s, c) => s + c[1], 0) / 4];
    const start = info.saved && clip?.gps_lat != null ? gps() : mid || gps();
    if (!start) {
      status.replaceChildren(h('div', { className: 'hint warn-hint' }, 'Deze video\'s bevatten geen GPS-positie, dus de app weet niet welk veld het is. Klik je plek dan op de veldtekening aan.'));
      canvas.style.display = 'none';
      return;
    }
    if (info.corners) {
      corners = info.corners; saved = info.saved; known = info.known;
      size = saved ? info.pitch : known;
      mode = saved && clip ? 'plek' : 'hoeken';
    }
    z = info.corners ? 19 : 18;  // nog geen hoeken: iets verder weg, zodat het hele veld naast je plek in beeld past
    center = toWorld(start, z);
    requestAnimationFrame(() => { draw(); renderStatus(); });
  })();

  return { el, destroy() { alive = false; window.removeEventListener('resize', onResize); } };
}
