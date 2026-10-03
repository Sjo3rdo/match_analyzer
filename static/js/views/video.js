// Video met spelersboxen + 2D-minimap (bovenaanzicht), momenten markeren en spelers aanklikken.
import { api, h, toast, fmtTime, matchMinute, TEAM_COLORS, teamName, playerLabel, labelList, teamOptions, markSpectator } from '../util.js';
import { PitchView } from '../pitch.js';

export async function render(root, ctx) {
  const { match } = ctx;
  const clips = match.clips.filter(c => c.status === 'klaar');
  if (!clips.length) {
    root.append(h('div', { className: 'panel empty' }, 'Nog geen geanalyseerde video\'s.'));
    return;
  }
  let clip = clips.find(c => c.id === Number(ctx.params.get('clip'))) || clips[0];
  let selectedTrack = ctx.params.get('track') ? Number(ctx.params.get('track')) : null;
  let showBoxes = true, placingBall = false;
  const playerById = new Map(match.players.map(p => [`p${p.id}`, p]));

  const video = h('video', { controls: true, preload: 'auto', playsInline: true });
  const overlay = h('canvas');
  const mini = h('canvas');
  const side = h('div', { className: 'panel' });
  const eventsBox = h('div', { className: 'list' });
  const label = h('input', { placeholder: 'Label (bijv. Goal, Kans, Redding)', list: 'moment-labels', style: { flex: 1 } });
  const ballBtn = h('button', { title: 'Pauzeer, klik hierop en klik daarna op de bal in de video. De app gebruikt dat voor balbezit en passes.',
    onclick: () => { placingBall = !placingBall; video.pause(); ballBtn.className = placingBall ? 'primary' : ''; ballBtn.textContent = placingBall ? 'Klik nu op de bal…' : '⚽ Bal aanwijzen'; } },
    '⚽ Bal aanwijzen');
  const markerPlayer = h('select', {}, h('option', { value: '' }, '– speler (optioneel) –'),
    match.players.map(p => h('option', { value: p.id }, playerLabel(p))));

  root.append(
    h('div', { className: 'row', style: { marginBottom: '12px' } }, 'Video:',
      h('select', { onchange: e => { clip = clips.find(c => c.id === Number(e.target.value)); ctx.setParam('clip', clip.id); loadClip(0); } },
        clips.map(c => h('option', { value: c.id, selected: c.id === clip.id }, c.filename))),
      h('label', {}, h('input', { type: 'checkbox', checked: true, onchange: e => { showBoxes = e.target.checked; draw(); } }), ' spelersboxen tonen'),
      ballBtn,
      h('button', { title: 'Op dit moment is er geen bal te zien (de app zag iets anders aan voor de bal)', onclick: () => setBall(null) }, '🚫 Geen bal hier'),
      h('span', { className: 'small muted' }, 'Bal: ⭕ gevonden · 🟡 geschat · 🟢 door jou aangewezen')),
    h('div', { className: 'grid2' },
      h('div', {},
        h('div', { className: 'video-wrap' }, video, overlay),
        h('div', { className: 'panel', style: { marginTop: '12px' } },
          h('div', { className: 'row' }, label, markerPlayer,
            h('button', { className: 'primary', onclick: addMoment, title: 'Maakt een clip van 6 s vóór tot 4 s na dit moment' }, '✂️ Maak clip')),
          h('div', { className: 'small muted', style: { marginTop: '6px' } },
            'Pauzeer bij een kans of goal, geef een label en klik "Maak clip" (6 s ervoor tot 4 s erna). Je clips staan hieronder bij ',
            '"Momenten in deze video" en bij ', h('a', { href: `#/match/${match.id}/momenten` }, 'Clips & delen'),
            ', waar je ze kunt bijknippen, tekenen en delen.'),
          labelList())),
      h('div', {},
        h('div', { className: 'panel' }, h('h3', {}, 'Minimap'), mini,
          h('div', { className: 'small muted', style: { marginTop: '6px' } },
            h('span', { className: 'dot', style: { background: TEAM_COLORS[0] } }), match.team0_name, '  ',
            h('span', { className: 'dot', style: { background: TEAM_COLORS[1] } }), match.team1_name, '  ',
            h('span', { className: 'dot', style: { background: '#fff' } }), 'bal')),
        side,
        h('div', { className: 'panel' }, h('h3', {}, 'Momenten in deze video'), eventsBox))));

  const pv = new PitchView(mini);
  requestAnimationFrame(() => { pv.resize(); draw(); });

  // --- data in tijdvensters ophalen -------------------------------------------------------
  const posCache = new Map(), boxCache = new Map(), ballCache = new Map();
  async function windowed(cache, kind, size, t) {
    const k = Math.floor(t / size);
    if (!cache.has(k)) {
      cache.set(k, api(`/clips/${clip.id}/${kind}?t0=${k * size}&t1=${(k + 1) * size}`));
      if (cache.size > 12) cache.delete(cache.keys().next().value);
    }
    return cache.get(k);
  }
  function nearestFrame(frames, t) {
    let lo = 0, hi = frames.length - 1, best = null;
    while (lo <= hi) { const m = (lo + hi) >> 1; if (frames[m].t <= t) { best = frames[m]; lo = m + 1; } else hi = m - 1; }
    return best && t - best.t < 0.5 ? best : null;
  }

  let lastBoxes = [];
  async function draw() {
    const t = video.currentTime;
    const [pos, boxes, balls] = await Promise.all([windowed(posCache, 'positions', 30, t), windowed(boxCache, 'boxes', 10, t),
      windowed(ballCache, 'ball', 30, t)]);
    // minimap
    pv.draw();
    const fr = nearestFrame(pos.frames || [], t);
    if (!pos.calibrated) {
      const c = pv.ctx; c.fillStyle = 'rgba(0,0,0,.5)'; c.fillRect(0, 0, pv.cssW, pv.cssH);
      c.fillStyle = '#fff'; c.font = '14px sans-serif'; c.textAlign = 'center';
      c.fillText('Kalibreer deze video eerst (tabblad Kalibratie)', pv.cssW / 2, pv.cssH / 2);
    } else if (fr) {
      for (const [x, y, team, ent, tid] of fr.p) {
        const p = playerById.get(ent);
        pv.dot(x, y, TEAM_COLORS[team] || '#666', tid === selectedTrack ? 8 : 6, p?.number || null);
      }
      if (fr.b) pv.dot(fr.b[0], fr.b[1], '#fff', 3.5);
    }
    // boxen over de video
    // Het canvas laat onderin ruimte vrij voor de videoknoppen; coördinaten beginnen linksboven de video.
    const r = video.getBoundingClientRect(), ro = overlay.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
    overlay.width = ro.width * dpr; overlay.height = ro.height * dpr;
    const c = overlay.getContext('2d');
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    c.clearRect(0, 0, ro.width, ro.height);
    const sx = r.width / clip.width, sy = sx;
    const yOff = (r.height - clip.height * sy) / 2;
    let nearestT = null;
    for (const b of boxes) if (b.t <= t && t - b.t < 0.5 && (nearestT === null || b.t > nearestT)) nearestT = b.t;
    lastBoxes = boxes.filter(b => b.t === nearestT).map(b => ({ ...b, px: [b.box[0] * sx, b.box[1] * sy + yOff, b.box[2] * sx, b.box[3] * sy + yOff] }));
    lastScale = { sx, sy, yOff };
    const ball = nearestFrame(balls || [], t);
    if (ball && t - ball.t < 0.15) {  // de bal: gevonden (wit), geschat (geel) of door jou aangewezen (groen)
      const col = { hand: '#30d158', geschat: '#ffd600' }[ball.kind] || '#fff';
      c.beginPath(); c.arc(ball.x * sx, ball.y * sy + yOff, 9, 0, 2 * Math.PI);
      c.strokeStyle = col; c.lineWidth = 2; c.setLineDash(ball.kind === 'geschat' ? [3, 3] : []); c.stroke(); c.setLineDash([]);
    }
    if (!showBoxes) return;
    for (const b of lastBoxes) {
      const [x1, y1, x2, y2] = b.px;
      const sel = b.track === selectedTrack;
      c.strokeStyle = sel ? '#ffd600' : b.valid ? (TEAM_COLORS[b.team] || '#aaa') : 'rgba(200,200,200,.4)';
      c.lineWidth = sel ? 3 : 1.5;
      c.strokeRect(x1, y1, x2 - x1, y2 - y1);
      const p = playerById.get(b.entity);
      if (p || sel) {
        c.font = '600 12px sans-serif'; c.fillStyle = c.strokeStyle;
        c.fillText(p ? playerLabel(p) : `#${b.track}`, x1, y1 - 4);
      }
    }
  }

  let raf;
  const loop = () => { draw(); if (!video.paused) raf = requestAnimationFrame(() => setTimeout(loop, 60)); };
  video.addEventListener('play', loop);
  video.addEventListener('seeked', draw);
  video.addEventListener('loadeddata', draw);

  let lastScale = null;
  async function setBall(px) {
    video.pause();
    const t = video.currentTime;
    const body = px ? { t, x: px[0], y: px[1] } : { t, x: null, y: null };
    await api(`/clips/${clip.id}/ball`, { json: body });
    ballCache.clear(); posCache.clear();
    toast(px ? 'Bal aangewezen. Speel een stukje verder en wijs hem weer aan waar de app hem mist.' : 'Opgeslagen: hier geen bal');
    draw();
  }
  overlay.addEventListener('click', e => {
    const r = overlay.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
    if (placingBall && lastScale) {
      placingBall = false; ballBtn.className = ''; ballBtn.textContent = '⚽ Bal aanwijzen';
      setBall([x / lastScale.sx, (y - lastScale.yOff) / lastScale.sy]);
      return;
    }
    const hit = lastBoxes.filter(b => x >= b.px[0] && x <= b.px[2] && y >= b.px[1] && y <= b.px[3])
      .sort((a, b) => (a.px[2] - a.px[0]) - (b.px[2] - b.px[0]))[0];
    if (!hit) { video.paused ? video.play() : video.pause(); return; }
    video.pause();
    selectedTrack = hit.track;
    drawSide();
    draw();
  });

  async function drawSide() {
    if (selectedTrack === null) {
      side.replaceChildren(h('h3', {}, 'Speler aanklikken'),
        h('div', { className: 'muted small' }, 'Pauzeer en klik op een speler in de video om hem aan een naam te koppelen.'));
      return;
    }
    const tracks = await api(`/clips/${clip.id}/tracks`);
    const t = tracks.find(x => x.track_id === selectedTrack);
    if (!t) return;
    const patch = async data => {
      await api(`/clips/${clip.id}/tracks/${t.track_id}`, { method: 'PATCH', json: data });
      Object.assign(t, data);
      if ('player_id' in data) {
        // speler-label direct bijwerken
        for (const cache of [posCache, boxCache]) cache.clear();
      }
      toast('Opgeslagen'); draw();
    };
    side.replaceChildren(
      h('h3', {}, `Track #${t.track_id}`),
      h('div', { className: 'row' },
        h('img', { src: `/api/clips/${clip.id}/thumb/${t.track_id}`, style: { height: '90px', borderRadius: '6px' } }),
        h('div', { className: 'small' },
          h('div', {}, `In beeld: ${fmtTime(t.t_start)} – ${fmtTime(t.t_end)}`),
          t.jersey_guess ? h('div', {}, `Rugnummer gelezen: ${t.jersey_guess}`) : null,
          h('div', {}, h('span', { className: 'swatch', style: { background: t.color || '#ccc' } }), ' shirtkleur'))),
      h('div', { className: 'row', style: { marginTop: '8px' } },
        h('select', { onchange: e => {
          const v = Number(e.target.value);
          if (v !== 3) return patch({ team: v });
          t.team = 3;
          const box = h('div');
          side.append(box);
          markSpectator(clip.id, t.track_id, box, () => { for (const cache of [posCache, boxCache]) cache.clear(); draw(); });
          for (const cache of [posCache, boxCache]) cache.clear(); draw();
        } }, teamOptions(match, t.team)),
        h('select', { onchange: e => patch({ player_id: e.target.value ? Number(e.target.value) : null }) },
          h('option', { value: '' }, '– speler –'),
          match.players.map(p => h('option', { value: p.id, selected: t.player_id === p.id }, `${playerLabel(p)} (${teamName(match, p.team)})`)))),
      t.player_id ? h('div', { style: { marginTop: '8px' } }, h('a', { href: `#/match/${match.id}/spelers?clip=${clip.id}&assist=${t.player_id}` },
        '🔍 Zoek meer tracks van deze speler (koppel-assistent)')) : null,
      match.players.length ? null : h('div', { className: 'small muted' }, 'Voeg eerst spelers toe bij "Spelers".'));
  }

  async function addMoment() {
    const t = video.currentTime, pid = markerPlayer.value ? Number(markerPlayer.value) : null;
    const m = await api(`/matches/${match.id}/moments`, { json: { clip_id: clip.id, start: Math.max(0, t - 6), end: t + 4,
      label: label.value || 'Moment', players: pid ? [pid] : [], spotlight_player_id: pid } });
    label.value = '';
    toast('✓ Clip gemaakt. Hij staat bij "Momenten in deze video" en bij "Clips & delen".');
    loadEvents();
    return m;
  }

  async function loadEvents() {
    const [moments, stats, hl] = await Promise.all([api(`/matches/${match.id}/moments`), api(`/matches/${match.id}/stats`),
      api(`/matches/${match.id}/highlights`).catch(() => [])]);
    const name = ent => playerLabel(playerById.get(ent)) || 'onbekend';
    const items = [
      ...moments.filter(m => m.clip_id === clip.id).map(m => ({ t: m.start, text: `🎬 ${m.label}${m.players.length ? ' – ' + m.players.map(id => name('p' + id)).join(', ') : ''}`,
        href: `#/match/${match.id}/momenten?moment=${m.id}` })),
      ...stats.events.filter(e => e.clip_id === clip.id && (e.kind !== 'sprint' || playerById.has(e.entity))).map(e => ({
        t: e.t, text: e.kind === 'sprint' ? `⚡ Sprint ${name(e.entity)} (${e.value} km/u)`
          : e.kind === 'pass' ? `➡️ Pass ${name(e.entity)} → ${name(e.to)}` : `✖️ Balverlies ${name(e.entity)}` })),
      ...hl.filter(e => e.clip_id === clip.id).map(e => ({ t: Math.max(0, e.t - 6),
        text: e.kind === 'gejuich' ? '📣 Gejuich (mogelijk een kans of goal)' : '🔔 Fluitsignaal' })),
    ].sort((a, b) => a.t - b.t);
    eventsBox.replaceChildren(...(items.length ? items.map(it => h('div', { className: 'list-item', onclick: () => { video.currentTime = Math.max(0, it.t - 2); video.play(); } },
      h('b', {}, matchMinute(clip, it.t)), h('span', { className: 'muted small' }, fmtTime(it.t)), h('span', { style: { flex: 1 } }, it.text),
      it.href ? h('a', { href: it.href, onclick: e => e.stopPropagation() }, 'bewerk') : null))
      : [h('div', { className: 'muted small' }, 'Nog geen momenten.')]));
  }

  function loadClip(t0) {
    posCache.clear(); boxCache.clear(); ballCache.clear();
    video.src = `/api/clips/${clip.id}/video`;
    video.addEventListener('loadedmetadata', () => { video.currentTime = t0 || 0; }, { once: true });
    drawSide(); loadEvents();
  }
  loadClip(Number(ctx.params.get('t')) || 0);
  const onResize = () => { pv.resize(); draw(); };
  window.addEventListener('resize', onResize);
  return () => { cancelAnimationFrame(raf); video.pause(); video.removeAttribute('src'); window.removeEventListener('resize', onResize); };
}
