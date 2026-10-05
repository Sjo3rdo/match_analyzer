// Begeleide route voor een nieuwe wedstrijd: één vraag per scherm, van uploaden tot spelers.
import { api, h, toast, fmtTime, teamName, TEAM_COLORS } from '../util.js';
import * as calibrate from './calibrate.js';
import { stationsPanel } from './stations.js';

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

  // --- 5. veld vastleggen per standplaats -----------------------------------------------------
  async function stepCalibrate() {
    const data = await api(`/matches/${match.id}/stations`);
    const need = data.stations.filter(s => !s.anchor && !s.no_gps);
    const noGps = data.stations.filter(s => s.no_gps && !s.anchor);
    if (!need.length) {
      add(...title('Het veld is vastgelegd', 'Voor elke standplaats heb je één video gekalibreerd. In de volgende stap doet de app de rest.'),
        h('div', {}, data.stations.filter(s => s.anchor).map(s => h('div', { className: 'list-item' }, h('b', {}, s.label),
          h('span', { className: 'small muted' }, `${s.clips.length} video('s)`), h('span', { className: 'badge ok' }, `✓ via ${s.anchor.filename}`)))),
        noGps.length ? h('p', { className: 'small muted' }, `${noGps.reduce((n, s) => n + s.clips.length, 0)} video('s) zonder GPS-positie kon de app niet aan een plek koppelen; die kalibreer je later zelf bij "Kalibratie".`) : null);
      return navButtons();
    }
    const st = need[0], total = data.stations.filter(s => !s.no_gps).length, nr = total - need.length + 1;
    add(...title(`Veld vastleggen: ${st.label}`,
      `Standplaats ${nr} van ${total} · ${st.clips.length} video('s). Zie het als een fotograaf op een statief: leg je plek één keer vast met één video, dan doet de app de rest van deze standplaats zelf.`),
      h('div', { className: 'small muted' }, 'De app zoekt de video met het meeste veld in beeld…'));
    navButtons({ next: 'Deze standplaats overslaan →', onNext: () => go(step + 1), note: 'Sla over als geen enkele video bruikbaar is' });
    const best = await api(`/matches/${match.id}/stations/${st.id}/best`).catch(() => null);
    if (!best) { add(h('p', {}, 'Geen bruikbare video gevonden.')); return; }
    const other = h('select', { onchange: e => open(Number(e.target.value)) },
      st.clips.map(c => h('option', { value: c.id, selected: c.id === best.id }, c.filename)));
    const holder = h('div');
    card.replaceChildren(); add(...title(`Veld vastleggen: ${st.label}`,
      `Standplaats ${nr} van ${total} · ${st.clips.length} video('s). Leg je plek één keer vast met deze video; de app doet de rest van deze standplaats daarna zelf.`),
      h('div', { className: 'row small' }, 'Video:', other, h('span', { className: 'muted' }, '(de app koos die met het meeste veld in beeld; kies gerust een andere)')),
      holder);
    async function open(id) {
      cleanupStep();
      holder.replaceChildren();
      const m = await reload();
      const sub = await calibrate.render(holder, {
        match: m, params: new URLSearchParams({ clip: id }), setParam() {}, refreshSteps() {},
        wizard: { onSaved: () => { toast(`✓ ${st.label} vastgelegd`); go(step); } },
      });
      cleanupStep = typeof sub === 'function' ? sub : () => {};
    }
    await open(best.id);
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
