// Clips & delen: clips maken en inkorten, spelers taggen, tekenen, spotlight, afspeellijst per speler,
// exporteren en delen. Werkt zoals de Veo-editor, maar dan lokaal op je laptop.
import { api, h, toast, fmtTime, matchMinute, teamName, playerLabel, labelList, exportPanel } from '../util.js';

const COLORS = ['#ffd600', '#ff3b30', '#ffffff', '#34c759', '#0a84ff'];
const TOOLS = [['arrow', '➚ Pijl'], ['line', '╱ Lijn'], ['circle', '◯ Cirkel'], ['free', '✎ Vrij'], ['text', 'T Tekst']];
const KIND = { sprint: '⚡ Sprint', pass: '➡️ Pass', balverlies: '✖️ Balverlies', gejuich: '📣 Gejuich', fluitsignaal: '🔔 Fluitsignaal' };

export async function render(root, ctx) {
  const { match } = ctx;
  if (!match.clips.length) {
    root.append(h('div', { className: 'panel empty' }, 'Upload eerst een video bij "1. Video\'s".'));
    return;
  }
  const clipById = new Map(match.clips.map(c => [c.id, c]));
  const playerById = new Map(match.players.map(p => [p.id, p]));
  let moments = await api(`/matches/${match.id}/moments`);
  let current = moments.find(m => m.id === Number(ctx.params.get('moment'))) || moments[0] || null;
  let filterPlayer = '', filterLabel = '';
  const selected = new Set();
  let playlist = null, rangePlaying = false, drawMode = null, freezeTimer = null;
  let spot = null; // { t: [], boxes: [] } voor de spotlight-voorvertoning
  let freezeShapes = null, pinnedShapes = null; // tekening die nu getoond wordt
  let events = [];

  // --- elementen ----------------------------------------------------------------------
  const video = h('video', { controls: true, preload: 'auto', playsInline: true });
  const overlay = h('canvas');
  const wrap = h('div', { className: 'video-wrap' }, video, overlay);
  const editor = h('div');
  const listBox = h('div', { className: 'list', style: { maxHeight: '360px' } });
  const suggestBox = h('div', { className: 'list', style: { maxHeight: '260px' } });
  const exp = exportPanel();
  const playerFilter = h('select', { onchange: e => { filterPlayer = e.target.value; drawList(); drawSuggestions(); } },
    h('option', { value: '' }, 'Alle spelers'), match.players.map(p => h('option', { value: p.id }, playerLabel(p))));
  const labelFilter = h('input', { placeholder: 'Filter op label', list: 'moment-labels', style: { width: '150px' },
    oninput: e => { filterLabel = e.target.value.trim().toLowerCase(); drawList(); } });

  root.append(
    labelList(),
    h('div', { className: 'hint' },
      'Maak clips zoals in Veo: kies een moment (hier, bij "Video + minimap" of uit de suggesties), stel begin en eind in, ' +
      'tag spelers, zet een spotlight op een speler en teken op het beeld. Exporteer daarna één clip, een reel of losse ' +
      'bestanden en deel ze direct via AirDrop, WhatsApp, Berichten of Mail.'),
    h('div', { className: 'panel row' }, playerFilter, labelFilter,
      h('button', { onclick: playPlaylist }, '▶ Afspeellijst'),
      h('span', { style: { flex: 1 } }),
      h('button', { onclick: () => { visible().forEach(m => selected.add(m.id)); drawList(); } }, 'Selecteer zichtbare'),
      h('button', { onclick: () => { selected.clear(); drawList(); } }, 'Niets'),
      h('button', { className: 'primary', onclick: () => exportSelection('reel') }, '🎬 Exporteer als één video'),
      h('button', { onclick: () => exportSelection('zip') }, '🗂️ Losse bestanden (zip)')),
    h('div', { className: 'grid2' },
      h('div', {}, h('div', { className: 'panel' }, wrap, editor)),
      h('div', {},
        h('div', { className: 'panel' }, h('div', { className: 'row', style: { justifyContent: 'space-between' } },
          h('h3', {}, 'Clips'), h('button', { onclick: newClipHere }, '+ Nieuwe clip')), listBox),
        h('div', { className: 'panel' }, h('h3', {}, 'Exporteren & delen'), exp.el,
          h('div', { className: 'small muted' }, 'Kies clips met de vinkjes, of exporteer de geopende clip.')),
        h('div', { className: 'panel' }, h('h3', {}, 'Suggesties (automatisch gevonden)'), suggestBox))));

  // --- lijst ---------------------------------------------------------------------------
  function visible() {
    return moments.filter(m => (!filterPlayer || m.players.includes(Number(filterPlayer)) || m.spotlight_player_id === Number(filterPlayer))
      && (!filterLabel || (m.label || '').toLowerCase().includes(filterLabel)));
  }
  function names(ids) { return ids.map(id => playerLabel(playerById.get(id))).filter(Boolean).join(', '); }
  function drawList() {
    const v = visible();
    listBox.replaceChildren(...(v.length ? v.map(m => {
      const c = clipById.get(m.clip_id);
      return h('div', { className: `list-item ${current?.id === m.id ? 'selected' : ''}`, onclick: () => select(m) },
        h('input', { type: 'checkbox', checked: selected.has(m.id), onclick: e => e.stopPropagation(),
          onchange: e => { e.target.checked ? selected.add(m.id) : selected.delete(m.id); } }),
        h('b', {}, matchMinute(c, m.start)),
        h('span', { style: { flex: 1 } }, m.label || 'Moment', ' ', h('span', { className: 'muted small' }, names(m.players))),
        m.spotlight_player_id ? h('span', { title: 'Spotlight' }, '🔦') : null,
        m.drawings.length ? h('span', { title: 'Tekeningen' }, `✏️${m.drawings.length}`) : null,
        h('span', { className: 'muted small' }, `${Math.round(m.end - m.start)}s`));
    }) : [h('div', { className: 'muted small' }, 'Nog geen clips. Maak er een met "+ Nieuwe clip" of via een suggestie.')]));
  }

  async function loadSuggestions() {
    const [st, hl] = await Promise.all([api(`/matches/${match.id}/stats`).catch(() => ({ events: [] })),
      api(`/matches/${match.id}/highlights`).catch(() => [])]);
    // Gejuich bovenaan (sterkste eerst): dat zijn de kansen en goals; daarna de rest op tijd
    const loud = hl.filter(e => e.kind === 'gejuich').sort((a, b) => b.score - a.score);
    events = [...loud, ...[...st.events, ...hl.filter(e => e.kind !== 'gejuich')]
      .sort((a, b) => a.clip_id - b.clip_id || a.t - b.t)];
    drawSuggestions();
  }
  function drawSuggestions() {
    const pid = filterPlayer ? `p${filterPlayer}` : null;
    const ev = events.filter(e => clipById.has(e.clip_id) && (!pid || e.entity === pid || e.to === pid)
      && (e.kind !== 'sprint' || String(e.entity).startsWith('p'))).slice(0, 80);
    const label = ent => (String(ent || '').startsWith('p') ? playerLabel(playerById.get(Number(ent.slice(1)))) : '') || 'onbekend';
    suggestBox.replaceChildren(...(ev.length ? ev.map(e => h('div', { className: 'list-item' },
      h('b', {}, matchMinute(clipById.get(e.clip_id), e.t)), h('span', {}, KIND[e.kind] || e.kind),
      e.kind === 'gejuich' || e.kind === 'fluitsignaal'
        ? h('span', { style: { flex: 1 }, className: 'small muted' }, e.kind === 'gejuich' ? `uit het geluid, sterkte ${Math.round(e.score)}` : 'uit het geluid')
        : h('span', { style: { flex: 1 }, className: 'small' }, label(e.entity), e.to ? ` → ${label(e.to)}` : '', e.value ? ` (${e.value} km/u)` : ''),
      h('button', { onclick: () => fromEvent(e) }, '+ clip'))) : [h('div', { className: 'muted small' }, 'Geen suggesties (analyseer en kalibreer eerst, en koppel spelers).')]));
  }
  async function fromEvent(e) {
    if (e.kind === 'gejuich' || e.kind === 'fluitsignaal') {
      // het moment zelf zit vóór het gejuich of het fluitsignaal
      const m = await api(`/matches/${match.id}/moments`, { json: { clip_id: e.clip_id, start: Math.max(0, e.t - 10),
        end: e.t_end + 3, label: e.kind === 'gejuich' ? 'Kans' : 'Fluitsignaal', players: [] } });
      moments.push(m); select(m); toast('Clip gemaakt: kijk of het klopt en pas het label aan');
      return;
    }
    const pid = String(e.entity || '').startsWith('p') ? Number(e.entity.slice(1)) : null;
    const players = [pid, String(e.to || '').startsWith('p') ? Number(e.to.slice(1)) : null].filter(Boolean);
    const m = await api(`/matches/${match.id}/moments`, { json: { clip_id: e.clip_id, start: Math.max(0, e.t - 4),
      end: (e.t_end || e.t) + 3, label: { sprint: 'Sprint', pass: 'Pass', balverlies: 'Balverlies' }[e.kind] || 'Moment',
      players: [...new Set(players)], spotlight_player_id: pid } });
    moments.push(m); select(m); toast('Clip gemaakt');
  }
  async function newClipHere() {
    const clipId = current?.clip_id || match.clips[0].id;
    const t = clipId === current?.clip_id ? video.currentTime : 0;
    const m = await api(`/matches/${match.id}/moments`, { json: { clip_id: clipId, start: Math.max(0, t - 6), end: t + 4, label: 'Moment' } });
    moments.push(m); select(m);
  }

  // --- editor --------------------------------------------------------------------------
  // Wijzigingen verzamelen en na 0,3 s in één keer opslaan. Alle velden die intussen veranderd zijn
  // gaan mee (eerder ging bij snel achter elkaar wijzigen van verschillende velden de eerste verloren).
  let saveTimer, pending = {}, pendingFor = null;
  function flushSave() {
    clearTimeout(saveTimer);
    if (!pendingFor || !Object.keys(pending).length) return Promise.resolve();
    const m = pendingFor, fields = pending;
    pending = {}; pendingFor = null;
    return api(`/moments/${m.id}`, { method: 'PATCH', json: fields })
      .then(r => { for (const [k, v] of Object.entries(r)) if (!(pendingFor === m && k in pending)) m[k] = v; })
      .catch(() => {}).finally(drawList);
  }
  function save(fields) {
    if (pendingFor && pendingFor !== current) flushSave();
    Object.assign(current, fields);
    pendingFor = current;
    Object.assign(pending, fields);
    clearTimeout(saveTimer);
    saveTimer = setTimeout(flushSave, 300);
    drawList();
  }

  function select(m) {
    flushSave();
    current = m; stopDraw(); playlist = playlist && playlist.includes(m.id) ? playlist : null;
    ctx.setParam('moment', m.id);
    const c = clipById.get(m.clip_id);
    const url = `/api/clips/${c.id}/video`;
    if (!video.src.endsWith(url)) video.src = url;
    const seek = () => { video.currentTime = m.start; drawOverlay(); };
    video.readyState >= 1 ? seek() : video.addEventListener('loadedmetadata', seek, { once: true });
    loadSpot(); drawEditor(); drawList();
  }

  function timeCtl(label, key) {
    const shown = h('b', { style: { minWidth: '52px', display: 'inline-block', textAlign: 'center' } }, fmtTime(current[key]));
    const set = v => {
      const c = clipById.get(current.clip_id);
      v = Math.max(0, Math.min(c.duration || 1e9, v));
      if (key === 'start' && current.end - v < 0.5) return toast('Begin moet vóór het eind liggen');
      if (key === 'end' && v - current.start < 0.5) return toast('Eind moet na het begin liggen');
      save({ [key]: Math.round(v * 10) / 10 }); shown.textContent = fmtTime(current[key]); dur.textContent = durText();
      video.currentTime = current[key];
    };
    return h('div', { className: 'row' }, h('span', { style: { width: '44px' } }, label),
      h('button', { onclick: () => set(current[key] - 1) }, '−1s'), shown, h('button', { onclick: () => set(current[key] + 1) }, '+1s'),
      h('button', { onclick: () => set(video.currentTime), title: 'Zet op de huidige videotijd' }, '⤓ hier'));
  }
  const dur = h('span', { className: 'muted' });
  const durText = () => `duur ${Math.round((current.end - current.start) * 10) / 10} s`;

  function drawEditor() {
    if (!current) { editor.replaceChildren(h('div', { className: 'empty' }, 'Kies of maak een clip.')); return; }
    const m = current, c = clipById.get(m.clip_id);
    dur.textContent = durText();
    const chips = [0, 1].map(team => h('div', { className: 'row small', style: { gap: '6px', marginTop: '4px' } },
      h('span', { className: 'muted', style: { width: '90px' } }, teamName(match, team)),
      match.players.filter(p => p.team === team).map(p => h('label', { className: `chip ${m.players.includes(p.id) ? 'on' : ''}` },
        h('input', { type: 'checkbox', checked: m.players.includes(p.id), onchange: e => {
          const ps = new Set(current.players); e.target.checked ? ps.add(p.id) : ps.delete(p.id);
          save({ players: [...ps] }); e.target.parentElement.classList.toggle('on', e.target.checked);
        } }), playerLabel(p)))));
    editor.replaceChildren(
      h('div', { className: 'row', style: { marginTop: '10px', justifyContent: 'space-between' } },
        h('div', { className: 'muted small' }, `${c.filename} · ${matchMinute(c, m.start)}`, playlist ? ` · afspeellijst ${playlist.indexOf(m.id) + 1}/${playlist.length}` : ''),
        h('div', { className: 'row' },
          h('button', { className: 'primary', onclick: playRange }, '▶ Speel clip'),
          h('button', { onclick: startDraw }, '✏️ Teken op dit beeld'))),
      h('div', { className: 'row', style: { marginTop: '8px', gap: '20px' } }, timeCtl('Begin', 'start'), timeCtl('Eind', 'end'), dur),
      h('div', { className: 'row', style: { marginTop: '10px' } },
        h('input', { value: m.label || '', list: 'moment-labels', placeholder: 'Label', style: { width: '180px' }, oninput: e => save({ label: e.target.value }) }),
        h('label', {}, '🔦 Spotlight: ', h('select', { onchange: e => { save({ spotlight_player_id: e.target.value ? Number(e.target.value) : null }); loadSpot(); } },
          h('option', { value: '' }, 'geen'), match.players.map(p => h('option', { value: p.id, selected: m.spotlight_player_id === p.id }, playerLabel(p))))),
        c.status !== 'klaar' ? h('span', { className: 'small muted' }, '(spotlight werkt na analyse en koppelen)') : null),
      match.players.length ? h('div', { style: { marginTop: '8px' } }, h('div', { className: 'small muted' }, 'Spelers in deze clip:'), chips)
        : h('div', { className: 'small muted', style: { marginTop: '8px' } }, 'Voeg spelers toe bij "3. Spelers" om ze te taggen.'),
      h('textarea', { placeholder: 'Opmerking (bijv. "let op de loopactie van de spits")', rows: 2, style: { width: '100%', marginTop: '8px' },
        oninput: e => save({ comment: e.target.value }) }, m.comment || ''),
      h('div', { style: { marginTop: '8px' } }, h('div', { className: 'small muted' }, 'Tekeningen (beeld bevriest in de export):'),
        m.drawings.length ? m.drawings.map((d, i) => h('div', { className: 'list-item', onclick: () => { video.pause(); rangePlaying = false; showDrawing(d); } },
          h('b', {}, fmtTime(d.t)), h('span', { style: { flex: 1 } }, `${d.shapes.length} vorm(en), ${d.duration}s stil`),
          h('button', { className: 'danger', onclick: e => { e.stopPropagation(); save({ drawings: current.drawings.filter((_, k) => k !== i) }); drawEditor(); } }, '×')))
          : h('div', { className: 'small muted' }, 'Nog geen. Pauzeer op het juiste beeld en klik "Teken op dit beeld".')),
      h('div', { className: 'row', style: { marginTop: '12px' } },
        h('button', { className: 'primary', onclick: () => exp.run(match.id, { moment_ids: [m.id], name: `${m.label || 'clip'}-${matchMinute(c, m.start)}` }, m.label) }, '📤 Exporteer & deel deze clip'),
        h('span', { style: { flex: 1 } }),
        h('button', { className: 'danger', onclick: async () => {
          if (!confirm('Deze clip verwijderen?')) return;
          if (pendingFor === m) { pending = {}; pendingFor = null; clearTimeout(saveTimer); }
          await api(`/moments/${m.id}`, { method: 'DELETE' });
          moments = moments.filter(x => x.id !== m.id); current = null;
          visible()[0] ? select(visible()[0]) : (drawEditor(), drawList());
        } }, 'Verwijder clip')));
  }

  // --- afspelen: bereik, afspeellijst, tekeningen ----------------------------------------
  let lastT = 0;
  function playRange() {
    if (!current) return;
    stopDraw(); rangePlaying = true; lastT = current.start - 0.01;
    video.currentTime = current.start; video.play();
  }
  function playPlaylist() {
    const v = visible();
    if (!v.length) return toast('Geen clips in deze selectie');
    playlist = v.map(m => m.id);
    select(v[0]);
    const go = () => playRange();
    video.readyState >= 1 ? setTimeout(go, 100) : video.addEventListener('loadeddata', go, { once: true });
  }
  function tick() {
    if (current && rangePlaying && !video.paused) {
      const t = video.currentTime;
      const d = current.drawings.find(x => x.t > lastT && x.t <= t);
      lastT = t;
      if (d) {
        freezeShapes = d.shapes;
        freezeTimer = setTimeout(() => { freezeTimer = null; freezeShapes = null; if (rangePlaying) video.play(); }, d.duration * 1000);
        video.pause();
      } else if (t >= current.end) {
        video.pause(); rangePlaying = false;
        if (playlist) {
          const i = playlist.indexOf(current.id);
          const next = moments.find(m => m.id === playlist[i + 1]);
          if (next) {
            select(next);
            const go = () => playRange();
            video.readyState >= 1 ? setTimeout(go, 150) : video.addEventListener('loadeddata', go, { once: true });
          } else { playlist = null; drawEditor(); toast('Einde afspeellijst'); }
        }
      }
    }
    if (!drawMode) drawOverlay(freezeShapes || pinnedShapes);
    raf = requestAnimationFrame(tick);
  }
  let raf = requestAnimationFrame(tick);
  video.addEventListener('pause', () => { if (!freezeTimer) rangePlaying = rangePlaying && false; });
  video.addEventListener('seeking', () => { lastT = video.currentTime; pinnedShapes = null; });
  video.addEventListener('play', () => { pinnedShapes = null; });

  // --- overlay: spotlight en tekeningen ----------------------------------------------
  async function loadSpot() {
    spot = null;
    const m = current;
    if (!m?.spotlight_player_id || clipById.get(m.clip_id).status !== 'klaar') return;
    const boxes = await api(`/clips/${m.clip_id}/boxes?t0=${m.start - 1}&t1=${m.end + 1}`);
    const mine = boxes.filter(b => b.entity === `p${m.spotlight_player_id}`).sort((a, b) => a.t - b.t);
    if (m === current) spot = { t: mine.map(b => b.t), boxes: mine.map(b => b.box) };
  }
  function spotBox(t) {
    if (!spot || !spot.t.length) return null;
    let i = spot.t.findIndex(x => x > t);
    if (i === -1) i = spot.t.length;
    if (i > 0 && i < spot.t.length && spot.t[i] - spot.t[i - 1] <= 0.6) {
      const w = (t - spot.t[i - 1]) / (spot.t[i] - spot.t[i - 1]);
      return spot.boxes[i].map((v, k) => (1 - w) * spot.boxes[i - 1][k] + w * v);
    }
    const j = i > 0 && (i === spot.t.length || t - spot.t[i - 1] < spot.t[i] - t) ? i - 1 : i;
    return Math.abs(spot.t[j] - t) <= 0.15 ? spot.boxes[j] : null;
  }
  function sizeOverlay() {
    const r = overlay.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
    overlay.width = r.width * dpr; overlay.height = r.height * dpr;
    const c = overlay.getContext('2d'); c.setTransform(dpr, 0, 0, dpr, 0, 0); c.clearRect(0, 0, r.width, r.height);
    const v = video.getBoundingClientRect();
    return { c, W: v.width, H: v.height };
  }
  function drawOverlay(shapes = null) {
    if (!current) return;
    const { c, W, H } = sizeOverlay();
    const clip = clipById.get(current.clip_id);
    const b = spotBox(video.currentTime);
    if (b && clip.width) {
      const s = W / clip.width;
      const [x1, y1, x2, y2] = b.map(v => v * s);
      const cx = (x1 + x2) / 2, w = x2 - x1;
      c.beginPath(); c.ellipse(cx, y2, Math.max(8, 0.75 * w), Math.max(4, 0.22 * w), 0, 0, 2 * Math.PI);
      c.fillStyle = 'rgba(255,214,0,.3)'; c.fill(); c.strokeStyle = '#ffd600'; c.lineWidth = 2.5; c.stroke();
      const p = playerById.get(current.spotlight_player_id);
      if (p) { c.font = '600 13px sans-serif'; c.textAlign = 'center'; c.fillStyle = '#ffd600'; c.fillText(playerLabel(p), cx, y1 - 8); }
    }
    if (shapes) drawShapes(c, shapes, W, H);
  }
  function showDrawing(d) {
    // na het verspringen van de video de tekening laten staan
    video.addEventListener('seeked', () => { pinnedShapes = d.shapes; }, { once: true });
    video.currentTime = d.t;
  }

  function drawShapes(c, shapes, W, H) {
    const lw = Math.max(3, W / 300);
    for (const s of shapes) {
      c.strokeStyle = c.fillStyle = s.color || '#ffd600'; c.lineWidth = lw; c.lineCap = 'round'; c.lineJoin = 'round';
      const P = p => [p[0] * W, p[1] * H];
      if (s.type === 'arrow' || s.type === 'line') {
        const [ax, ay] = P(s.from), [bx, by] = P(s.to);
        c.beginPath(); c.moveTo(ax, ay); c.lineTo(bx, by); c.stroke();
        if (s.type === 'arrow') {
          const a = Math.atan2(by - ay, bx - ax), L = Math.min(lw * 5, Math.hypot(bx - ax, by - ay) * 0.4);
          c.beginPath(); c.moveTo(bx, by); c.lineTo(bx - L * Math.cos(a - 0.45), by - L * Math.sin(a - 0.45));
          c.moveTo(bx, by); c.lineTo(bx - L * Math.cos(a + 0.45), by - L * Math.sin(a + 0.45)); c.stroke();
        }
      } else if (s.type === 'circle') {
        const [cx, cy] = P(s.center), rx = s.radius * W;
        c.beginPath(); c.ellipse(cx, cy, Math.max(2, rx), Math.max(2, rx * (s.ratio || 1)), 0, 0, 2 * Math.PI); c.stroke();
      } else if (s.type === 'free' && s.points.length > 1) {
        c.beginPath(); s.points.forEach((p, i) => i ? c.lineTo(...P(p)) : c.moveTo(...P(p))); c.stroke();
      } else if (s.type === 'text' && s.text) {
        const [x, y] = P(s.at), fs = Math.max(14, W / 45);
        c.font = `600 ${fs}px sans-serif`; c.textAlign = 'left';
        const tw = c.measureText(s.text).width;
        c.fillStyle = 'rgba(20,20,20,.85)'; c.fillRect(x - 8, y - fs - 4, tw + 16, fs + 14);
        c.fillStyle = s.color || '#ffd600'; c.fillText(s.text, x, y);
      }
    }
  }

  // --- tekenmodus ------------------------------------------------------------------------
  const drawBar = h('div', { className: 'panel row', style: { display: 'none', marginTop: '8px' } });
  wrap.after(drawBar);
  function startDraw() {
    if (!current) return;
    video.pause(); rangePlaying = false;
    const t = video.currentTime;
    if (t < current.start || t > current.end) return toast('Ga eerst naar een beeld binnen de clip');
    drawMode = { t, shapes: [], tool: 'arrow', color: COLORS[0], duration: 4, temp: null };
    video.controls = false; overlay.classList.add('full'); overlay.style.cursor = 'crosshair';
    drawToolbar(); redrawDraw();
  }
  function stopDraw() {
    if (!drawMode) return;
    drawMode = null; video.controls = true; overlay.classList.remove('full'); overlay.style.cursor = '';
    drawBar.style.display = 'none'; drawOverlay();
  }
  function drawToolbar() {
    const d = drawMode;
    drawBar.style.display = 'flex';
    drawBar.replaceChildren(
      ...TOOLS.map(([k, l]) => h('button', { className: d.tool === k ? 'primary' : '', onclick: () => { d.tool = k; drawToolbar(); } }, l)),
      ...COLORS.map(col => h('button', { title: col, onclick: () => { d.color = col; drawToolbar(); },
        style: { background: col, width: '28px', height: '28px', padding: 0, outline: d.color === col ? '3px solid var(--accent)' : '' } })),
      h('button', { onclick: () => { d.shapes.pop(); redrawDraw(); } }, '↶ Ongedaan'),
      h('label', {}, 'Stilstaan ', h('input', { type: 'number', min: 1, max: 15, value: d.duration, style: { width: '56px' },
        onchange: e => { d.duration = Math.max(1, Number(e.target.value) || 4); } }), ' s'),
      h('button', { className: 'primary', onclick: () => {
        if (!d.shapes.length) return toast('Teken eerst iets');
        save({ drawings: [...current.drawings, { t: Math.round(d.t * 100) / 100, duration: d.duration, shapes: d.shapes }] });
        stopDraw(); drawEditor(); toast('Tekening opgeslagen');
      } }, 'Opslaan'),
      h('button', { onclick: stopDraw }, 'Annuleren'));
  }
  function redrawDraw() {
    if (!drawMode) return;
    const { c, W, H } = sizeOverlay();
    drawShapes(c, drawMode.temp ? [...drawMode.shapes, drawMode.temp] : drawMode.shapes, W, H);
  }
  function norm(e) {
    const r = video.getBoundingClientRect();
    return [Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)), Math.min(1, Math.max(0, (e.clientY - r.top) / r.height))];
  }
  overlay.addEventListener('pointerdown', e => {
    if (!drawMode) return;
    const p = norm(e), d = drawMode;
    if (d.tool === 'text') {
      const text = prompt('Tekst:');
      if (text) { d.shapes.push({ type: 'text', at: p, text, color: d.color }); redrawDraw(); }
      return;
    }
    overlay.setPointerCapture(e.pointerId);
    d.temp = d.tool === 'free' ? { type: 'free', points: [p], color: d.color }
      : d.tool === 'circle' ? { type: 'circle', center: p, radius: 0, ratio: 1, color: d.color, _start: p }
      : { type: d.tool, from: p, to: p, color: d.color };
  });
  overlay.addEventListener('pointermove', e => {
    const d = drawMode;
    if (!d?.temp) return;
    const p = norm(e), r = video.getBoundingClientRect();
    if (d.temp.type === 'free') d.temp.points.push(p);
    else if (d.temp.type === 'circle') {
      const dx = (p[0] - d.temp._start[0]), dy = (p[1] - d.temp._start[1]) * r.height / r.width;
      d.temp.radius = Math.hypot(dx, dy); d.temp.ratio = 1;
    } else d.temp.to = p;
    redrawDraw();
  });
  overlay.addEventListener('pointerup', () => {
    const d = drawMode;
    if (!d?.temp) return;
    const s = d.temp; delete s._start; d.temp = null;
    const big = s.type === 'free' ? s.points.length > 2 : s.type === 'circle' ? s.radius > 0.005
      : Math.hypot(s.to[0] - s.from[0], s.to[1] - s.from[1]) > 0.01;
    if (big) d.shapes.push(s);
    redrawDraw();
  });

  // --- export ------------------------------------------------------------------------
  function exportSelection(mode) {
    const ids = moments.filter(m => selected.has(m.id)).map(m => m.id);
    if (!ids.length) return toast('Vink eerst clips aan (of gebruik "Selecteer zichtbare")');
    const p = filterPlayer ? playerById.get(Number(filterPlayer)) : null;
    exp.run(match.id, { moment_ids: ids, mode, name: p ? `clips-${playerLabel(p)}` : `clips-${match.name}` },
      p ? `Clips van ${playerLabel(p)}` : match.name);
  }

  drawList(); drawEditor();
  if (current) select(current);
  loadSuggestions();
  const onResize = () => (drawMode ? redrawDraw() : drawOverlay());
  window.addEventListener('resize', onResize);
  return () => { flushSave(); cancelAnimationFrame(raf); clearTimeout(freezeTimer); video.pause(); video.removeAttribute('src'); window.removeEventListener('resize', onResize); };
}
