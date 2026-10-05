// Begeleide route voor een nieuwe wedstrijd: één vraag per scherm, van uploaden tot spelers.
import { api, h, toast, fmtTime, teamName, TEAM_COLORS } from '../util.js';
import { PitchView, setPitchSize, pitchPolylines } from '../pitch.js';
import * as calibrate from './calibrate.js';
import { stationsPanel } from './stations.js';
import { heightField } from './camheight.js';

const STEPS = [
  ['wedstrijd', 'Wedstrijd'], ['videos', "Video's"], ['tijdlijn', 'Tijdlijn'], ['analyse', 'Analyseren'],
  ['kalibratie', 'Veld vastleggen'], ['controle', 'Controleren'], ['spelers', 'Teams & spelers'],
];
const BUSY = ['wachtrij', 'preview', 'analyse'];

const clock = ts => (ts ? new Date(ts * 1000).toLocaleTimeString('nl-NL', { hour: '2-digit', minute: '2-digit', second: '2-digit' }) : '–');
// HH:MM(:SS) op dezelfde dag als ref (seconden sinds 1970) -> seconden sinds 1970
function atTime(ref, hhmm) {
  if (!hhmm) return null;
  const d = new Date(ref * 1000);
  const [hh, mm, ss] = hhmm.split(':').map(Number);
  d.setHours(hh, mm, ss || 0, 0);
  return d.getTime() / 1000;
}
const timeValue = ts => (ts ? new Date(ts * 1000).toTimeString().slice(0, 8) : '');

export async function render(root, ctx) {
  let match = ctx.match;
  let step = Math.min(STEPS.length, Math.max(1, Number(ctx.params.get('stap')) || match.wizard_step || 1));
  let cleanupStep = () => {};
  const stepper = h('div', { className: 'wizard-steps' });
  const card = h('div', { className: 'panel wizard-card' });
  const nav = h('div', { className: 'wizard-nav' });
  root.append(stepper, card, nav);

  const reload = async () => { match = await api(`/matches/${match.id}`); return match; };
  async function go(n) {
    cleanupStep(); cleanupStep = () => {};
    step = Math.min(STEPS.length, Math.max(1, n));
    ctx.setParam('stap', step);
    if (step > (match.wizard_step || 0)) api(`/matches/${match.id}`, { method: 'PATCH', json: { wizard_step: step } }).catch(() => {});
    match.wizard_step = Math.max(match.wizard_step || 0, step);
    await reload();
    drawStepper();
    card.replaceChildren(); nav.replaceChildren();
    card.style.maxWidth = step === 5 || step === 6 ? 'none' : '';  // kalibreren en controleren: volle breedte
    window.scrollTo(0, 0);
    await STEP_FN[step - 1]();
  }
  function drawStepper() {
    stepper.replaceChildren(...STEPS.map(([, label], i) => h('div', {
      className: `ws ${i + 1 === step ? 'cur' : ''} ${i + 1 < (match.wizard_step || 1) && i + 1 !== step ? 'done' : ''}`,
      onclick: () => i + 1 <= (match.wizard_step || 1) && go(i + 1),
    }, h('span', { className: 'n' }, i + 1 < (match.wizard_step || 1) && i + 1 !== step ? '✓' : i + 1), label)));
  }
  function navButtons({ next = 'Volgende →', onNext = () => go(step + 1), disabled = false, note = null } = {}) {
    nav.replaceChildren(
      step > 1 ? h('button', { onclick: () => go(step - 1) }, '← Terug') : h('span'),
      h('div', { className: 'row' }, note ? h('span', { className: 'small muted' }, note) : null,
        h('button', { className: 'primary', disabled, onclick: onNext }, next)));
  }
  const add = (...items) => card.append(...items.flat().filter(x => x !== null && x !== undefined && x !== false));
  const title = (t, lead) => [h('div', { className: 'eyebrow' }, `Stap ${step} van ${STEPS.length}`), h('h2', {}, t), lead ? h('p', { className: 'lead' }, lead) : null];

  // --- 1. wedstrijd ---------------------------------------------------------------------
  async function stepMatch() {
    const squads = await api('/squads').catch(() => []);
    const name = h('input', { value: match.name, style: { width: '100%' } });
    const date = h('input', { type: 'date', value: match.date || '' });
    const t0 = h('input', { value: match.team0_name, list: 'wz-squads' });
    const t1 = h('input', { value: match.team1_name, list: 'wz-squads' });
    const half = h('select', {}, [30, 35, 40, 45].map(m => h('option', { value: m, selected: (match.half_length || 45) === m }, `${m} minuten`)));
    add(...title('Over de wedstrijd', 'Een paar gegevens vooraf. Kies bij een team een opgeslagen naam, dan staan de spelers er straks meteen in.'),
      h('datalist', { id: 'wz-squads' }, squads.map(q => h('option', { value: q.name }))),
      h('div', { className: 'form-grid' },
        h('label', {}, 'Naam', name), h('label', {}, 'Datum', date),
        h('label', {}, 'Thuisteam', t0), h('label', {}, 'Uitteam', t1),
        h('label', {}, 'Speeltijd per helft', half)));
    navButtons({ onNext: async () => {
      await api(`/matches/${match.id}`, { method: 'PATCH', json: { name: name.value.trim() || match.name, date: date.value || null,
        team0_name: t0.value.trim() || 'Thuis', team1_name: t1.value.trim() || 'Uit', half_length: Number(half.value) } });
      for (const [team, inp] of [[0, t0], [1, t1]]) {  // opgeslagen selectie? meteen overnemen
        const sq = squads.find(q => q.name.toLowerCase() === inp.value.trim().toLowerCase());
        if (sq) await api(`/matches/${match.id}/load-squad`, { json: { team, squad_id: sq.id } }).catch(() => {});
      }
      go(step + 1);
    } });
  }

  // --- 2. video's -------------------------------------------------------------------------
  async function stepVideos() {
    const list = h('div', { className: 'list' });
    const status = h('div', { className: 'small muted' });
    const input = h('input', { type: 'file', accept: 'video/*', multiple: true, style: { display: 'none' } });
    const zone = h('div', { className: 'dropzone', onclick: () => input.click() },
      h('div', { style: { fontSize: '30px' } }, '🎬'), h('b', {}, 'Sleep al je video\'s van deze wedstrijd hierheen'),
      h('div', {}, 'of klik om ze te kiezen. Alles in één keer mag (ook 60 stuks).'));
    const drawList = () => {
      list.replaceChildren(...match.clips.map(c => h('div', { className: 'list-item' }, h('span', { style: { flex: 1 } }, c.filename),
        h('span', { className: 'small muted' }, fmtTime(c.duration)),
        h('span', { className: 'small muted' }, c.rec_start ? `gefilmd ${clock(c.rec_start)}` : 'geen opnametijd'),
        c.gps_lat != null ? h('span', { className: 'badge ok', title: 'Bevat de GPS-positie' }, 'GPS') : null)));
      navButtons({ disabled: !match.clips.length, note: match.clips.length ? `${match.clips.length} video('s)` : 'Upload eerst minstens één video' });
    };
    const upload = files => {
      if (!files.length) return;
      const fd = new FormData();
      for (const f of files) fd.append('files', f);
      const xhr = new XMLHttpRequest();
      xhr.open('POST', `/api/matches/${match.id}/clips`);
      xhr.upload.onprogress = e => { status.replaceChildren(h('div', { className: 'progress' }, h('div', { style: { width: `${Math.round(100 * e.loaded / e.total)}%` } })),
        `Uploaden… ${Math.round(100 * e.loaded / e.total)}%`); };
      xhr.onload = async () => {
        if (xhr.status >= 400) { toast(JSON.parse(xhr.responseText).detail || 'Upload mislukt', true); status.textContent = ''; return; }
        status.textContent = `${files.length} video('s) toegevoegd.`;
        await reload(); drawList();
      };
      xhr.send(fd);
    };
    input.addEventListener('change', () => upload([...input.files]));
    zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('over'); });
    zone.addEventListener('dragleave', () => zone.classList.remove('over'));
    zone.addEventListener('drop', e => { e.preventDefault(); zone.classList.remove('over'); upload([...e.dataTransfer.files].filter(f => f.type.startsWith('video') || /\.(mov|mp4|m4v)$/i.test(f.name))); });
    add(...title("Video's toevoegen", 'Voeg alle video\'s van deze wedstrijd toe, in willekeurige volgorde: de app zet ze straks zelf op volgorde.'),
      zone, input, status, h('div', { style: { marginTop: '12px' } }, list));
    drawList();
  }

  // --- 3. tijdlijn ------------------------------------------------------------------------
  async function stepTimeline() {
    const g = await api(`/matches/${match.id}/timeline`);
    if (!g.clips.length) {
      add(...title('Volgorde en helften', 'Deze video\'s bevatten geen opnametijd, dus de app kan de volgorde niet zelf bepalen. Je kunt helft en beginminuut later per video invullen bij "Video\'s".'));
      return navButtons();
    }
    const ref = g.clips[0].rec_start;
    const kick = h('input', { type: 'time', step: 1, value: timeValue(g.kickoff) });
    let second = g.second_clip;
    const kick2 = h('input', { type: 'time', step: 1, value: timeValue(g.kickoff2) });
    const preview = h('div');
    const choose = h('select', { onchange: e => { second = e.target.value ? Number(e.target.value) : null;
      const c = g.clips.find(x => x.id === second); if (c) kick2.value = timeValue(c.rec_start); drawSecond(); } },
      h('option', { value: '' }, 'Er is geen 2e helft (of die is niet gefilmd)'),
      g.clips.map(c => h('option', { value: c.id, selected: c.id === second }, `${clock(c.rec_start)} – ${c.filename}`)));
    function drawSecond() {
      const c = g.clips.find(x => x.id === second);
      preview.replaceChildren(...(c ? [
        h('img', { className: 'wizard-thumb', src: `/api/clips/${c.id}/frame?t=1&w=640`, alt: c.filename }),
        h('div', { className: 'small muted', style: { margin: '6px 0' } }, `${c.filename} · gefilmd om ${clock(c.rec_start)}`,
          c.id === g.second_clip && g.gap_minutes ? ` · daarvoor ${g.gap_minutes} minuten niets gefilmd (de rust?)` : ''),
        h('label', { className: 'row' }, 'Hoe laat begon de 2e helft?', kick2)] : [h('div', { className: 'small muted' }, 'Alle video\'s horen dan bij de 1e helft.')]));
    }
    add(...title('Wanneer begon de wedstrijd?',
      'Met de opnametijd van je video\'s zet de app alles op volgorde en rekent hij voor elke video de wedstrijdminuut uit.'),
      h('label', { className: 'row' }, h('b', {}, 'Hoe laat was de aftrap?'), kick,
        h('span', { className: 'small muted' }, `(je eerste video begon om ${clock(g.clips[0].rec_start)})`)),
      h('h3', { style: { marginTop: '18px' } }, g.second_clip ? 'Begint hier de 2e helft?' : 'Is er een 2e helft gefilmd?'),
      g.second_clip ? h('p', { className: 'small muted' }, 'De app denkt dat dit de eerste video van de 2e helft is, omdat er daarvoor het langst niets is gefilmd. Klopt het niet? Kies de goede video.') : null,
      h('div', { className: 'row' }, 'Eerste video van de 2e helft:', choose), preview,
      g.n_without_time ? h('p', { className: 'small muted' }, `${g.n_without_time} video('s) zonder opnametijd komen achteraan; die kun je later bij "Video's" zelf plaatsen.`) : null);
    drawSecond();
    navButtons({ next: '✓ Klopt, verder →', onNext: async () => {
      await api(`/matches/${match.id}/timeline`, { json: { kickoff: atTime(ref, kick.value), second_clip: second,
        kickoff2: second ? atTime(g.clips.find(x => x.id === second).rec_start, kick2.value) : null, half_length: match.half_length || 45 } });
      toast('Volgorde, helften en minuten ingesteld'); go(step + 1);
    } });
  }

  // --- 4. analyseren ----------------------------------------------------------------------
  async function stepAnalyse() {
    const body = h('div');
    const mode = h('select', {}, h('option', { value: 'nauwkeurig' }, 'Nauwkeurig: vindt ook spelers ver weg en de bal het best'),
      h('option', { value: 'snel' }, 'Snel: ongeveer twee keer zo snel, mist vaker verre spelers en de bal'));
    add(...title('Video\'s analyseren', 'De app zoekt in elke video naar spelers en de bal. Dat duurt ongeveer even lang als de video\'s zelf. Je hoeft niet te wachten: ga gerust al door naar de volgende stap.'), body);
    let timer;
    const draw = async () => {
      await reload();
      const clips = match.clips, done = clips.filter(c => c.status === 'klaar').length;
      const busy = clips.filter(c => BUSY.includes(c.status)), todo = clips.filter(c => c.status !== 'klaar' && !BUSY.includes(c.status));
      const frac = clips.length ? clips.reduce((s, c) => s + (c.status === 'klaar' ? 1 : BUSY.includes(c.status) ? (c.progress || 0) : 0), 0) / clips.length : 0;
      body.replaceChildren(
        todo.length ? h('div', { className: 'row' }, mode, h('button', { className: 'primary', onclick: async () => {
          for (const c of todo) await api(`/clips/${c.id}/process`, { json: { mode: mode.value } });
          toast(`${todo.length} video('s) in de wachtrij`); draw();
        } }, `▶ Start de analyse van ${todo.length} video('s)`)) : null,
        h('div', { className: 'progress big', style: { marginTop: '14px' } }, h('div', { style: { width: `${Math.round(100 * frac)}%` } })),
        h('div', { className: 'small muted', style: { marginTop: '6px' } }, `${done} van ${clips.length} klaar`,
          busy.length ? ` · bezig: ${busy.find(c => c.status === 'analyse')?.filename || busy[0].filename}` : ''));
      navButtons({ note: done < clips.length ? 'De analyse loopt door terwijl je verder gaat' : '✓ Alles geanalyseerd' });
      clearTimeout(timer);
      if (busy.length) timer = setTimeout(draw, 3000);
    };
    cleanupStep = () => clearTimeout(timer);
    await draw();
  }

  // --- 5. veld vastleggen per standplaats: a) waar stond je  b) plek verbeteren  c) veld intekenen  d) herkenningspunten
  async function stepCalibrate() {
    const data = await api(`/matches/${match.id}/stations`);
    const need = data.stations.filter(s => !s.anchor && !s.no_gps);
    const noGps = data.stations.filter(s => s.no_gps && !s.anchor);
    if (!need.length) {
      add(...title('Het veld is vastgelegd', 'Voor elke standplaats is één video gekalibreerd. In de volgende stap doet de app de rest.'),
        h('div', {}, data.stations.filter(s => s.anchor).map(s => h('div', { className: 'list-item' }, h('b', {}, s.label),
          h('span', { className: 'small muted' }, `${s.clips.length} video('s)`), h('span', { className: 'badge ok' }, `✓ via ${s.anchor.filename}`)))),
        noGps.length ? h('p', { className: 'small muted' }, `${noGps.reduce((n, s) => n + s.clips.length, 0)} video('s) zonder GPS-positie kon de app niet aan een plek koppelen; die kalibreer je later zelf bij "Kalibratie".`) : null);
      return navButtons();
    }
    const st = need[0], total = data.stations.filter(s => !s.no_gps).length, nr = total - need.length + 1;
    add(...title(`Veld vastleggen: ${st.label}`, `Standplaats ${nr} van ${total} · ${st.clips.length} video('s). De app zoekt de video met het meeste veld in beeld…`));
    navButtons({ next: 'Deze standplaats overslaan →', onNext: () => go(step + 1), note: 'Sla over als geen enkele video bruikbaar is' });
    let clip = await api(`/matches/${match.id}/stations/${st.id}/best`).catch(() => null);
    if (!clip) { add(h('p', {}, 'Geen bruikbare video gevonden.')); return; }

    const SUBS = [['plek', 'Waar stond je?'], ['verbeter', 'Plek verbeteren'], ['voorstel', 'Veld intekenen'], ['punten', 'Herkenningspunten']];
    let sub = clip.cam_x != null ? 'verbeter' : 'plek', osm = null, gpsMsg = null, subCleanup = () => {};
    const body = h('div');
    const subBar = h('div', { className: 'wizard-steps small' });
    const pick = h('select', { onchange: async e => {
      clip = (await reload()).clips.find(c => c.id === Number(e.target.value)) || clip; show(clip.cam_x != null ? 'verbeter' : 'plek');
    } },
      st.clips.map(c => h('option', { value: c.id, selected: c.id === clip.id }, c.filename)));
    card.replaceChildren(); add(...title(`Veld vastleggen: ${st.label}`,
      `Standplaats ${nr} van ${total} · ${st.clips.length} video('s). Je legt je plek één keer vast met één video; de app doet de rest van deze standplaats daarna zelf.`),
      h('div', { className: 'row small' }, 'Video:', pick, h('span', { className: 'muted' }, '(de app koos die met het meeste veld in beeld)')),
      subBar, body);
    cleanupStep = () => subCleanup();

    const setCam = async data => { const c = await api(`/clips/${clip.id}/camera`, { method: 'PATCH', json: data }); Object.assign(clip, c); };

    function show(s) {
      subCleanup(); subCleanup = () => {};
      sub = s;
      const idx = SUBS.findIndex(x => x[0] === s);
      subBar.replaceChildren(...SUBS.map(([k, label], i) => h('div', { className: `ws ${k === s ? 'cur' : ''} ${i < idx ? 'done' : ''}`,
        onclick: () => (i <= idx || (i === 1 && clip.cam_x != null)) && show(k) },
        h('span', { className: 'n' }, i < idx ? '✓' : 'abcd'[i]), label)));
      body.replaceChildren();
      ({ plek: subPlace, verbeter: subRefine, voorstel: subPropose, punten: subPoints })[s]();
    }

    // een veldtekening waarop je je plek ziet (en aanklikt of versleept)
    function pitchPicker({ onPick, drag = false, acc = null, dir = null }) {
      const canvas = h('canvas', { style: { cursor: 'crosshair', borderRadius: '10px', maxWidth: '760px' } });
      const pv = new PitchView(canvas, { margin: 14 });
      let dragging = false;
      const draw = () => {
        pv.resize(); pv.draw();
        if (clip.cam_x != null) {
          if (acc) {  // onzekerheid van de GPS-positie
            const c = pv.ctx, [px, py] = pv.toPx(clip.cam_x, clip.cam_y);
            c.beginPath(); c.arc(px, py, acc * pv.scale, 0, 2 * Math.PI); c.fillStyle = 'rgba(255,214,0,.18)'; c.fill();
          }
          const yaw = aim ? Math.atan2(aim[1] - clip.cam_y, aim[0] - clip.cam_x) * 180 / Math.PI : clip.cam_yaw;
          if (yaw != null) {  // kijkrichting: gestippelde pijl
            const c = pv.ctx, a = yaw * Math.PI / 180, [x0, y0] = pv.toPx(clip.cam_x, clip.cam_y);
            const [x1, y1] = pv.toPx(clip.cam_x + 28 * Math.cos(a), clip.cam_y + 28 * Math.sin(a)), b = Math.atan2(y1 - y0, x1 - x0);
            c.save(); c.strokeStyle = c.fillStyle = '#4db5ff'; c.lineWidth = 3; c.setLineDash([7, 5]);
            c.beginPath(); c.moveTo(x0, y0); c.lineTo(x1, y1); c.stroke(); c.setLineDash([]);
            c.beginPath(); c.moveTo(x1, y1); c.lineTo(x1 - 12 * Math.cos(b - .45), y1 - 12 * Math.sin(b - .45));
            c.lineTo(x1 - 12 * Math.cos(b + .45), y1 - 12 * Math.sin(b + .45)); c.closePath(); c.fill(); c.restore();
          }
          pv.dot(clip.cam_x, clip.cam_y, '#ffd600', 9, '📍');
        }
      };
      let aim = null;  // kijkrichting die je nu aanwijst (klikken of slepen)
      const at = e => { const r = canvas.getBoundingClientRect(); return pv.toM(e.clientX - r.left, e.clientY - r.top); };
      canvas.addEventListener('mousedown', e => {
        const [x, y] = at(e);
        if (dir?.active() && clip.cam_x != null) { aim = [x, y]; draw(); return; }
        if (drag && clip.cam_x != null && Math.hypot(x - clip.cam_x, y - clip.cam_y) < 6) { dragging = true; return; }
        clip.cam_x = x; clip.cam_y = y; draw(); onPick(x, y);
      });
      const move = e => {
        if (aim) { aim = at(e); draw(); return; }
        if (!dragging) return; const [x, y] = at(e); clip.cam_x = x; clip.cam_y = y; draw();
      };
      const up = () => {
        if (aim) {
          const [ax, ay] = aim; aim = null;
          if (Math.hypot(ax - clip.cam_x, ay - clip.cam_y) > 3) dir.onAim(Math.round(Math.atan2(ay - clip.cam_y, ax - clip.cam_x) * 180 / Math.PI));
          draw(); return;
        }
        if (dragging) { dragging = false; onPick(clip.cam_x, clip.cam_y); }
      };
      window.addEventListener('mousemove', move); window.addEventListener('mouseup', up);
      subCleanup = () => { window.removeEventListener('mousemove', move); window.removeEventListener('mouseup', up); };
      requestAnimationFrame(draw);
      return { el: canvas, draw };
    }

    // a) waar stond je?
    function subPlace() {
      const status = h('div');
      const picker = pitchPicker({ onPick: async (x, y) => {
        await setCam({ x, y, h: clip.cam_h || 1.6, source: 'hand' });
        status.replaceChildren(h('div', { className: 'hint' }, '📍 Je plek staat op de tekening.'));
        nextBtn.disabled = false;
      } });
      const nextBtn = h('button', { className: 'primary', disabled: clip.cam_x == null, onclick: () => show('verbeter') }, 'Volgende: plek verbeteren →');
      const gpsBtn = clip.gps_lat != null ? h('button', { className: 'primary', onclick: async () => {
        gpsBtn.disabled = true; gpsBtn.textContent = '📡 Zoeken in OpenStreetMap…';
        status.replaceChildren(h('div', { className: 'small muted' }, 'De app stuurt alleen de GPS-positie van deze video naar OpenStreetMap en zoekt het voetbalveld dat daar ligt.'));
        try {
          const r = await fetch(`/api/clips/${clip.id}/camera/gps`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' });
          const j = await r.json();
          if (!r.ok) throw new Error(j.detail || r.statusText);
          Object.assign(clip, j.clip);
          osm = [j.pitch_length, j.pitch_width];
          gpsMsg = `Gevonden: een veld van ${j.pitch_length} × ${j.pitch_width} m. Je plek staat op de tekening (GPS ± ${Math.max(5, Math.round(clip.gps_acc || 8))} m; de gele cirkel).`;
          status.replaceChildren(h('div', { className: 'hint' }, '✓ ', gpsMsg));
          picker.draw(); nextBtn.disabled = false;
        } catch (err) {
          status.replaceChildren(h('div', { className: 'hint warn-hint' }, h('b', {}, 'Zoeken via GPS lukte niet. '), String(err.message || err),
            h('div', { style: { marginTop: '4px' } }, 'Geen probleem: klik hieronder zelf op de tekening waar je stond.')));
        } finally { gpsBtn.disabled = false; gpsBtn.textContent = '📡 Zoek mijn plek via GPS'; }
      } }, '📡 Zoek mijn plek via GPS') : null;
      body.append(
        h('p', { className: 'lead' }, clip.gps_lat != null
          ? 'Je video weet ongeveer waar je stond. Klik op de knop: de app zoekt het voetbalveld op en zet je plek op de tekening. Lukt dat niet, klik dan zelf op de tekening.'
          : 'Deze video bevat geen GPS-positie. Klik op de tekening waar je ongeveer stond (naast het veld mag ook).'),
        h('div', { className: 'row', style: { marginBottom: '10px' } }, gpsBtn), status, picker.el,
        h('div', { className: 'row', style: { marginTop: '12px' } }, nextBtn));
      if (clip.cam_x != null) status.replaceChildren(h('div', { className: 'hint' }, gpsMsg || '📍 Je plek staat al op de tekening.'));
    }

    // b) plek verbeteren en extra gegevens
    function subRefine() {
      const saved = h('span', { className: 'small muted' });
      let aiming = false;
      const dirBtn = h('button', { onclick: () => { aiming = !aiming; drawDir(); } });
      const dirClear = h('button', { onclick: async () => { await setCam({ yaw: null }); drawDir(); picker.draw(); } }, 'Wissen');
      const drawDir = () => {
        dirBtn.className = aiming ? 'primary' : '';
        dirBtn.textContent = aiming ? 'Klik of sleep nu op het veld naar waar je keek…' : '🧭 Geef aan waar je naartoe keek';
        dirClear.style.display = clip.cam_yaw != null ? '' : 'none';
      };
      const picker = pitchPicker({ drag: true, acc: clip.cam_source === 'gps' ? Math.max(5, clip.gps_acc || 8) : null, onPick: async (x, y) => {
        await setCam({ x, y, source: 'hand' }); saved.textContent = '✓ opgeslagen';
      }, dir: { active: () => aiming, onAim: async yaw => {
        await setCam({ yaw }); aiming = false; drawDir(); picker.draw(); saved.textContent = '✓ kijkrichting opgeslagen';
      } } });
      drawDir();
      const height = heightField(clip.cam_h || 1.6, async v => { await setCam({ h: v }); saved.textContent = '✓ opgeslagen'; });
      const len = h('input', { type: 'number', min: 50, max: 120, step: 0.5, value: match.pitch_length || 105, style: { width: '80px' } });
      const wid = h('input', { type: 'number', min: 30, max: 90, step: 0.5, value: match.pitch_width || 68, style: { width: '80px' } });
      const saveSize = async () => {
        await api(`/matches/${match.id}`, { method: 'PATCH', json: { pitch_length: Number(len.value), pitch_width: Number(wid.value) } });
        await reload(); setPitchSize(match.pitch_length, match.pitch_width); picker.draw(); saved.textContent = '✓ veldmaten opgeslagen';
      };
      len.addEventListener('change', saveSize); wid.addEventListener('change', saveSize);
      const osmDiff = osm && (Math.abs(osm[0] - (match.pitch_length || 105)) > 1 || Math.abs(osm[1] - (match.pitch_width || 68)) > 1);
      body.append(
        h('p', { className: 'lead' }, 'Klopt de plek? Sleep de 📍 naar waar je echt stond: hoe preciezer, hoe beter de app straks het veld vindt. Geef ook aan waar je naartoe keek, hoe hoog je de telefoon hield en hoe groot het veld is.'),
        picker.el,
        h('div', { style: { marginTop: '10px' } },
          h('div', { style: { marginBottom: '4px' } }, 'Waar keek je naartoe? (niet verplicht, maar het helpt de app de goede kant op te zoeken)'),
          h('div', { className: 'row' }, dirBtn, dirClear),
          h('div', { className: 'small muted', style: { marginTop: '4px' } }, 'Klik op het veld wat ongeveer in het midden van je beeld stond, of sleep een pijl die kant op. Zwenkte je veel? Kies dan het midden van wat je filmde.')),
        h('div', { className: 'form-grid', style: { marginTop: '12px' } },
          h('div', {}, h('div', { style: { marginBottom: '4px' } }, 'Hoe hoog hield je de telefoon?'), height),
          h('label', {}, 'Veldmaten (lengte × breedte, meter)', h('div', { className: 'row' }, len, '×', wid)),
          osmDiff ? h('div', { className: 'hint' }, `OpenStreetMap zegt ${osm[0]} × ${osm[1]} m. `,
            h('button', { className: 'small', onclick: () => { len.value = osm[0]; wid.value = osm[1]; saveSize(); } }, 'Overnemen')) : null),
        h('div', { className: 'small muted' }, 'Amateurvelden zijn vaak kleiner dan 105 × 68 m. Weet je het niet, laat het dan zo.'),
        h('div', { className: 'row', style: { marginTop: '12px' } },
          h('button', { onclick: () => show('plek') }, '← Waar stond je'),
          h('button', { className: 'primary', onclick: () => show('voorstel') }, 'Volgende: veld intekenen →'), saved));
    }

    // c) de app tekent het veld in; jij zegt of het klopt
    function subPropose() {
      const moments = [0.5, 0.25, 0.75, 0.1, 0.9].map(q => Math.round(q * (clip.duration || 10) * 10) / 10);
      let k = 0, result = null;
      const canvas = h('canvas', { style: { width: '100%', maxWidth: '960px', borderRadius: '10px', background: '#000' } });
      const msg = h('div');
      const buttons = h('div', { className: 'row', style: { marginTop: '10px' } });
      body.append(h('p', { className: 'lead' }, 'De app zoekt nu zelf de witte lijnen en legt de veldtekening erop, vanaf jouw plek. Vallen de gele lijnen op de witte lijnen in het beeld?'),
        msg, canvas, buttons);
      const tryNext = async () => {
        const t = moments[k % moments.length]; k++;
        msg.replaceChildren(h('div', { className: 'small muted' }, `⏳ Zoeken op ${fmtTime(t)}… (± 10 seconden)`)); buttons.replaceChildren();
        result = await api(`/clips/${clip.id}/propose?t=${t}`).catch(e => ({ ok: false, message: e.message }));
        const frameT = result.t ?? t;
        const img = new Image();
        img.onload = () => {
          const W = 960, s = W / (clip.width || img.width);
          canvas.width = W; canvas.height = Math.round((clip.height || img.height) * s);
          const c = canvas.getContext('2d'); c.drawImage(img, 0, 0, canvas.width, canvas.height);
          if (result.ok) {
            const H = result.H;
            c.strokeStyle = '#ffd600'; c.lineWidth = 2.5;
            for (const pl of pitchPolylines()) {
              c.beginPath(); let pen = false;
              for (const [X, Y] of pl) {
                const w = H[2][0] * X + H[2][1] * Y + H[2][2];
                if (w <= 1e-6) { pen = false; continue; }
                const u = (H[0][0] * X + H[0][1] * Y + H[0][2]) / w * s, v = (H[1][0] * X + H[1][1] * Y + H[1][2]) / w * s;
                if (Math.abs(u) > 4 * W || Math.abs(v) > 4 * W) { pen = false; continue; }
                pen ? c.lineTo(u, v) : c.moveTo(u, v); pen = true;
              }
              c.stroke();
            }
          }
        };
        img.src = `/api/clips/${clip.id}/frame?t=${frameT}&w=960`;
        if (result.ok) {
          msg.replaceChildren(h('div', { className: 'hint' }, result.info?.ambiguous
            ? 'Er zijn hier weinig lijnen te zien; kijk extra goed of het klopt.' : 'Gevonden. Liggen de gele lijnen op de witte lijnen?'));
          buttons.replaceChildren(
            h('button', { className: 'primary', onclick: async () => {
              await api(`/clips/${clip.id}/keyframes`, { json: { t: result.t, points: result.points } });
              toast(`✓ ${st.label} vastgelegd`); go(step);
            } }, '✓ Ja, dit klopt'),
            h('button', { onclick: tryNext }, '↻ Ander moment proberen'),
            h('button', { onclick: () => show('punten') }, '✗ Klopt niet: zelf herkenningspunten aanklikken'));
        } else {
          msg.replaceChildren(h('div', { className: 'hint warn-hint' }, 'Op dit moment vond de app het veld niet zelf. ', result.message || ''));
          buttons.replaceChildren(
            k < moments.length ? h('button', { className: 'primary', onclick: tryNext }, '↻ Ander moment proberen') : null,
            h('button', { className: k < moments.length ? '' : 'primary', onclick: () => show('punten') }, 'Zelf herkenningspunten aanklikken'),
            h('button', { onclick: () => show('verbeter') }, '← Plek verbeteren'));
        }
      };
      tryNext();
    }

    // d) zelf herkenningspunten aanklikken (met uitleg per klik)
    async function subPoints() {
      const holder = h('div');
      body.append(h('p', { className: 'lead' }, 'Klik in het beeld op een punt dat je herkent (hoekvlag, hoek van het strafschopgebied, middenstip, doelpaal) en daarna op hetzelfde punt in de veldtekening. Omdat je plek bekend is, zijn 1 punt en 1 lijn vaak al genoeg. De balk hieronder zegt steeds wat de volgende klik is.'),
        h('button', { className: 'small', onclick: () => show('voorstel') }, '← Toch het voorstel van de app'), holder);
      const m = await reload();
      const r = await calibrate.render(holder, {
        match: m, params: new URLSearchParams({ clip: clip.id }), setParam() {}, refreshSteps() {},
        wizard: { noAutoPropose: true, onSaved: () => { toast(`✓ ${st.label} vastgelegd`); go(step); } },
      });
      subCleanup = typeof r === 'function' ? r : () => {};
    }

    show(sub);
  }

  // --- 6. de rest automatisch, en controleren ----------------------------------------------------
  async function stepCheck() {
    const info = h('div');
    const panel = stationsPanel(match, id => { location.hash = `#/match/${match.id}/kalibratie?clip=${id}`; });
    add(...title('De rest automatisch, en controleren',
      'De app legt elke andere video als een puzzelstuk tegen je gekalibreerde beelden (bomen, huizen, borden staan vanaf dezelfde plek altijd op dezelfde plek). Kijk daarna de plaatjes langs: liggen de gele lijnen op de witte lijnen? Klik ✓ of ✗. Een ✗ of ⚠ los je op met ✎.'),
      info, panel.el);
    let timer, started = false;
    const tick = async () => {
      await reload();
      const clips = match.clips, done = clips.filter(c => c.status === 'klaar').length;
      const data = await api(`/matches/${match.id}/stations`);
      const busyJob = ['wachtrij', 'bezig'].includes(data.job.status);
      const open = data.stations.flatMap(s => s.clips).filter(c => c.state === 'open' && s_anchor(data, c));
      if (done < clips.length) info.replaceChildren(h('p', { className: 'small warn' }, `Wacht nog op de analyse (${done} van ${clips.length} klaar). Zodra die klaar is, begint de app hier vanzelf.`));
      else info.replaceChildren();
      if (!started && !busyJob && open.length && done === clips.length) {
        started = true;
        await api(`/matches/${match.id}/stations/run`, { json: {} });
        panel.refresh();
      }
      const all = data.stations.flatMap(s => s.clips);
      const left = all.filter(c => ['voorstel', 'mislukt', 'afgekeurd', 'open'].includes(c.state)).length;
      navButtons({ note: left ? `Nog ${left} video('s) te controleren of te doen; dat kan ook later` : '✓ Alles gecontroleerd' });
      clearTimeout(timer);
      if (done < clips.length || busyJob) timer = setTimeout(tick, 4000);
    };
    const s_anchor = (data, c) => data.stations.some(s => s.anchor && s.clips.some(x => x.id === c.id));
    cleanupStep = () => { clearTimeout(timer); panel.cleanup(); };
    await tick();
  }

  // --- 7. teams en spelers -------------------------------------------------------------------
  async function stepPlayers() {
    const squads = await api('/squads').catch(() => []);
    const analyzed = match.clips.filter(c => c.status === 'klaar');
    const teams = h('div', { className: 'grid-cols' });
    const drawTeams = () => teams.replaceChildren(...[0, 1].map(team => {
      const mine = match.players.filter(p => p.team === team);
      const nr = h('input', { placeholder: 'Nr', style: { width: '56px' } });
      const nm = h('input', { placeholder: 'Naam speler' });
      const add = async () => {
        if (!nm.value.trim()) return nm.focus();
        await api(`/matches/${match.id}/players`, { json: { name: nm.value.trim(), number: nr.value.trim() || null, team } });
        await reload(); drawTeams(); setTimeout(() => teams.querySelectorAll('input')[team * 2]?.focus());
      };
      nm.addEventListener('keydown', e => e.key === 'Enter' && add());
      return h('div', { className: 'panel' },
        h('h3', {}, h('span', { className: 'swatch', style: { background: (match.team_colors || [])[team] || TEAM_COLORS[team] } }), ' ', teamName(match, team)),
        h('div', { className: 'small muted' }, (match.team_colors || [])[team] ? 'Shirtkleur zoals de app hem zag' : 'Shirtkleur nog onbekend (na de analyse)'),
        squads.length ? h('select', { style: { marginTop: '8px' }, onchange: async e => {
          if (!e.target.value) return;
          const r = await api(`/matches/${match.id}/load-squad`, { json: { team, squad_id: Number(e.target.value) } });
          toast(`${r.added} speler(s) overgenomen`); await reload(); drawTeams();
        } }, h('option', { value: '' }, '📋 Vaste selectie overnemen…'), squads.map(q => h('option', { value: q.id }, `${q.name} (${q.players.length})`))) : null,
        h('div', { className: 'small', style: { margin: '8px 0' } }, mine.length ? mine.map(p => `${p.number ? '#' + p.number + ' ' : ''}${p.name}`).join(', ') : 'Nog geen spelers.'),
        h('div', { className: 'row' }, nr, nm, h('button', { onclick: add }, '+')));
    }));
    add(...title('Teams en spelers',
      'Klopt de kleur bij de juiste ploeg? Zo niet, wissel ze om. Voer daarna de spelers in (met rugnummer), of neem een vaste selectie over.'),
      analyzed.length ? h('div', { className: 'row', style: { marginBottom: '10px' } },
        h('button', { onclick: async () => {
          await api(`/clips/${analyzed[0].id}/swap-teams`, { json: { all: true } });
          await reload(); drawTeams(); toast('Teams omgewisseld');
        } }, '⇄ De kleuren horen bij de andere ploeg: omwisselen')) : null,
      teams);
    drawTeams();
    navButtons({ next: '✓ Klaar', onNext: async () => {
      let msg = 'Klaar! Koppel nu de spelers aan wat de app zag.';
      if (match.players.length && analyzed.length) {
        const r = await api(`/matches/${match.id}/auto-assign`, { json: {} }).catch(() => null);
        if (r?.assigned) msg = `Klaar! ${r.assigned} stuk(ken) via het rugnummer al gekoppeld; koppel de rest bij Spelers.`;
      }
      await api(`/matches/${match.id}`, { method: 'PATCH', json: { wizard_done: 1, wizard_step: STEPS.length } });
      toast(msg); location.hash = `#/match/${match.id}/spelers`;
    } });
  }

  const STEP_FN = [stepMatch, stepVideos, stepTimeline, stepAnalyse, stepCalibrate, stepCheck, stepPlayers];
  await go(step);
  return () => cleanupStep();
}
