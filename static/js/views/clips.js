// Stap 1: video's uploaden, ordenen en laten analyseren.
import { api, h, fmtTime, toast, deleteAllClips } from '../util.js';

const BUSY = ['wachtrij', 'preview', 'analyse'];

export async function render(root, ctx) {
  const { match } = ctx;
  const panel = h('div', { className: 'panel' });
  const splitPanel = h('div');
  const orderPanel = h('div');
  let splitCleanup = () => {};
  const fileInput = h('input', { type: 'file', accept: 'video/*', multiple: true });
  const status = h('span', { className: 'muted' });
  const upload = async () => {
    if (!fileInput.files.length) return toast('Kies eerst één of meer video\'s');
    const fd = new FormData();
    for (const f of fileInput.files) fd.append('files', f);
    status.textContent = 'Uploaden...';
    const xhr = new XMLHttpRequest();
    xhr.open('POST', `/api/matches/${match.id}/clips`);
    xhr.upload.onprogress = e => { status.textContent = `Uploaden... ${Math.round(100 * e.loaded / e.total)}%`; };
    xhr.onload = () => {
      if (xhr.status >= 400) { toast(JSON.parse(xhr.responseText).detail || 'Upload mislukt', true); status.textContent = ''; return; }
      const learned = [...new Set(JSON.parse(xhr.responseText).flatMap(c => c.learned || []))];
      toast(learned.length ? `Geüpload. ${learned.join('. ')}.` : 'Geüpload'); ctx.reload();
    };
    xhr.send(fd);
  };
  root.append(
    h('div', { className: 'hint' },
      'Upload alle video\'s van deze wedstrijd (bijv. 1e en 2e helft, of losse stukken). Geef per video aan welke helft het is en ' +
      'op welke wedstrijdminuut hij begint. Klik daarna op "Analyseer": de app maakt een afspeelbare kopie en zoekt in ieder frame ' +
      'naar spelers en de bal. Dat kan op een MacBook ongeveer even lang duren als de video zelf.'),
    h('div', { className: 'panel' }, h('h2', {}, 'Video\'s toevoegen'),
      h('div', { className: 'row' }, fileInput, h('button', { className: 'primary', onclick: upload }, 'Uploaden'), status)),
    panel, orderPanel, splitPanel);

  let timer, wasBusy = null;
  const draw = async () => {
    const m = await api(`/matches/${match.id}`);
    const clips = m.clips;
    const todo = clips.filter(c => c.status !== 'klaar' && !BUSY.includes(c.status));
    panel.replaceChildren(h('div', { className: 'row', style: { justifyContent: 'space-between' } },
      h('h2', {}, 'Video\'s van deze wedstrijd'),
      clips.length ? h('div', { className: 'row' },
        h('button', { onclick: openOrder, title: 'Sorteer de video\'s op het moment van filmen en stel helft en beginminuut voor' },
          '🕒 Volgorde uit opnametijd'),
        h('button', { className: 'primary', disabled: !todo.length, onclick: () => analyzeAll(todo),
          title: 'Zet alle video\'s die nog niet geanalyseerd zijn in de wachtrij (ze worden één voor één gedaan)' },
          `▶ Analyseer alles${todo.length ? ` (${todo.length})` : ''}`),
        h('button', { className: 'danger', title: 'Alle video\'s van deze wedstrijd in één keer verwijderen',
          onclick: async () => { if (await deleteAllClips(match, clips.length)) { draw(); ctx.refreshSteps(); } } },
          '🗑 Alle video\'s verwijderen')) : null));
    if (!clips.length) { panel.append(h('div', { className: 'empty' }, 'Nog geen video\'s.')); return; }
    const patch = (c, data) => api(`/clips/${c.id}`, { method: 'PATCH', json: data });
    const move = async (i, dir) => {
      const j = i + dir; if (j < 0 || j >= clips.length) return;
      const order = clips.map(c => c);
      [order[i], order[j]] = [order[j], order[i]];
      await Promise.all(order.map((c, k) => patch(c, { order_idx: k })));
      draw();
    };
    panel.append(h('table', {},
      h('tr', {}, ['', 'Bestand', 'Duur', 'Helft', 'Start (min)',
        h('span', { title: 'Speelrichting omdraaien voor de statistieken, zodat beide helften dezelfde kant op spelen' }, 'Richting ⇄'),
        'Status', 'Kalibratie', ''].map(t => h('th', {}, t))),
      clips.map((c, i) => {
        const busy = BUSY.includes(c.status);
        const badge = c.status === 'klaar' ? 'ok' : c.status === 'fout' ? 'err' : busy ? 'busy' : '';
        return h('tr', {},
          h('td', {}, h('button', { onclick: () => move(i, -1), title: 'Omhoog' }, '↑'), h('button', { onclick: () => move(i, 1), title: 'Omlaag' }, '↓')),
          h('td', {}, c.filename, h('div', { className: 'muted small' }, `${c.width}×${c.height} · ${Math.round(c.fps)} fps`)),
          h('td', {}, fmtTime(c.duration)),
          h('td', {}, h('select', { onchange: async e => { await patch(c, { period: Number(e.target.value) }); if (c.flip == null) draw(); } },
            [1, 2, 3, 4].map(p => h('option', { value: p, selected: c.period === p }, p <= 2 ? `${p}e helft` : `Verlenging ${p - 2}`)))),
          h('td', {}, h('input', { type: 'number', min: 0, step: 1, value: c.start_minute, style: { width: '70px' },
            onchange: e => patch(c, { start_minute: Number(e.target.value) }) })),
          h('td', {}, h('label', { className: 'small', title: 'Omdraaien: de teams wisselen in de rust van kant. Zo spelen beide helften in heatmaps en teamvorm dezelfde kant op. Vink uit als je in de rust zelf naar de andere kant van het veld bent gelopen.' },
            h('input', { type: 'checkbox', checked: c.flip != null ? !!c.flip : c.period === 2,
              onchange: e => { c.flip = e.target.checked ? 1 : 0; patch(c, { flip: c.flip }); } }), ' omdraaien')),
          h('td', {}, h('span', { className: `badge ${badge}` }, c.status),
            c.status === 'klaar' && (c.analysis_version || 0) < 2
              ? h('div', { className: 'badge err', title: 'Deze video is geanalyseerd met een oudere versie die de tijden van iPhone-video\'s verkeerd las. Klik op "Opnieuw".' }, 'opnieuw analyseren aanbevolen')
              : c.status === 'klaar' && (c.analysis_version || 0) < 4
                ? h('div', { className: 'badge busy', title: 'De app volgt de bal nu beter (inzoomen als hij kwijt is). Klik op "Opnieuw" om dat ook voor deze video te gebruiken. Je kalibratie blijft bewaard; spelers koppelen moet je daarna opnieuw doen.' }, 'betere baldetectie: opnieuw analyseren') : null,
            busy ? h('div', { className: 'progress', title: c.message }, h('div', { style: { width: `${Math.round(100 * c.progress)}%` } })) : null,
            h('div', { className: 'muted small' }, c.message || '')),
          h('td', {}, c.n_keyframes ? h('span', { className: 'badge ok' }, `${c.n_keyframes} ijkmoment(s)`)
            : h('a', { href: `#/match/${match.id}/kalibratie?clip=${c.id}` }, 'Kalibreren')),
          h('td', {},
            h('select', { disabled: busy, title: 'Nauwkeurig: vindt ook spelers ver weg en de bal het best. Snel: ongeveer 2x zo snel, mist vaker verre spelers en de bal.',
              onchange: e => { c.analysis_mode = e.target.value; } },
              [['nauwkeurig', 'Nauwkeurig'], ['snel', 'Snel']].map(([v, l]) => h('option', { value: v, selected: (c.analysis_mode || 'nauwkeurig') === v }, l))), ' ',
            h('button', { className: busy ? '' : 'primary', disabled: busy, onclick: async () => {
              if (c.status === 'klaar' && !confirm(`${c.filename} opnieuw analyseren? De kalibratie blijft bewaard, maar gekoppelde spelers, ` +
                'teamcorrecties en aangewezen toeschouwers in deze video moet je daarna opnieuw doen.')) return;
              await api(`/clips/${c.id}/process`, { json: { mode: c.analysis_mode || 'nauwkeurig' } }); draw(); ctx.refreshSteps();
            } },
              c.status === 'klaar' ? 'Opnieuw' : 'Analyseer'), ' ',
            h('button', { disabled: busy, onclick: () => openSplit(c), title: 'Lange video in delen knippen, of warming-up/rust eruit halen' }, '✂️ Knippen'), ' ',
            h('button', { className: 'danger', disabled: busy, title: busy ? 'Wacht tot de analyse klaar is' : '', onclick: async () => {
              if (!confirm(`${c.filename} verwijderen?`)) return;
              await api(`/clips/${c.id}`, { method: 'DELETE' }); draw(); ctx.refreshSteps();
            } }, 'Verwijder')));
      })));
    clearTimeout(timer);
    const busyNow = clips.some(c => BUSY.includes(c.status));
    if (wasBusy && !busyNow) {
      ctx.refreshSteps();
      if (clips.every(c => c.status === 'klaar')) toast('Analyse klaar. Volgende stap: kalibreren (tabblad met het gele bolletje).');
    }
    wasBusy = busyNow;
    if (busyNow) timer = setTimeout(draw, 2000);
  };
  async function analyzeAll(todo) {
    if (!confirm(`${todo.length} video('s) analyseren? Ze gaan in de wachtrij en worden één voor één gedaan, elk met de keuze ` +
      'Nauwkeurig/Snel die erbij staat. Je kunt de app intussen gewoon gebruiken.')) return;
    for (const c of todo) await api(`/clips/${c.id}/process`, { json: { mode: c.analysis_mode || 'nauwkeurig' } });
    toast(`${todo.length} video('s) in de wachtrij`); draw(); ctx.refreshSteps();
  }

  // --- volgorde en verloop uit de opnametijd -----------------------------------------------
  async function openOrder(halfLength) {
    const p = await api(`/matches/${match.id}/order${typeof halfLength === 'number' ? `?half_length=${halfLength}` : ''}`);
    const fmt = ts => (ts ? new Date(ts * 1000).toLocaleString('nl-NL', { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit', second: '2-digit' }) : 'onbekend');
    const half = h('select', { onchange: e => openOrder(Number(e.target.value)) },
      [30, 40, 45].map(m => h('option', { value: m, selected: p.half_length === m }, `${m} minuten`)));
    const rows = p.items.map(it => {
      const changed = it.old.order_idx !== it.order_idx || it.old.period !== it.period || Math.abs((it.old.start_minute || 0) - it.start_minute) > 0.05;
      return h('tr', { className: changed ? 'changed' : '' },
        h('td', {}, it.order_idx + 1), h('td', {}, it.filename), h('td', { className: 'small' }, fmt(it.rec_start)),
        h('td', {}, h('select', { onchange: e => { it.period = Number(e.target.value); } },
          [1, 2, 3, 4].map(q => h('option', { value: q, selected: it.period === q }, q <= 2 ? `${q}e helft` : `Verlenging ${q - 2}`)))),
        h('td', {}, h('input', { type: 'number', min: 0, step: 0.1, value: it.start_minute, style: { width: '80px' },
          onchange: e => { it.start_minute = Number(e.target.value); } })),
        h('td', { className: 'small muted' }, changed ? `was ${it.old.order_idx + 1} · ${it.old.period}e · ${it.old.start_minute}'` : 'ongewijzigd'),
      );
    });
    const breakIdx = p.items.findIndex(it => it.clip_id === p.break_after);
    if (breakIdx >= 0) rows.splice(breakIdx + 1, 0, h('tr', {}, h('td', { colSpan: 6, className: 'small rust' },
      `☕ Rust: ${p.break_minutes} minuten niet gefilmd. Hierna begint de 2e helft.`)));
    orderPanel.replaceChildren(h('div', { className: 'panel' },
      h('div', { className: 'row', style: { justifyContent: 'space-between' } }, h('h2', {}, 'Volgorde uit de opnametijd'),
        h('label', {}, 'Speeltijd per helft: ', half)),
      h('div', { className: 'small muted' },
        'Voorstel: de video\'s op volgorde van filmen. Een gat van 8 minuten of meer is de rust. De eerste video van een helft ',
        'begint op de aftrap (0\' of de speeltijd van één helft); de rest telt daar vanaf door. Begon je later of eerder met ',
        'filmen? Pas de minuten aan voordat je het overneemt.',
        p.n_without_time ? ` ${p.n_without_time} video('s) zonder opnametijd staan achteraan en houden hun helft en minuut.` : ''),
      h('table', {}, h('tr', {}, ['#', 'Bestand', 'Gefilmd', 'Helft', 'Start (min)', ''].map(t => h('th', {}, t))), rows),
      h('div', { className: 'row', style: { marginTop: '10px' } },
        h('button', { className: 'primary', onclick: async () => {
          await api(`/matches/${match.id}/order`, { json: { items: p.items, half_length: p.half_length } });
          orderPanel.replaceChildren(); toast('Volgorde, helft en minuten overgenomen'); draw();
        } }, '✓ Overnemen'),
        h('button', { onclick: () => orderPanel.replaceChildren() }, 'Annuleren'))));
    orderPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  await draw();
  // --- knippen vóór analyse ---------------------------------------------------------------
  function openSplit(c) {
    splitCleanup();
    let segs = [], pendingStart = null;
    const video = h('video', { controls: true, preload: 'auto', playsInline: true, src: `/api/clips/${c.id}/video`, style: { width: '100%', borderRadius: '8px', background: '#000' } });
    const timeline = h('div', { className: 'timeline', title: 'Klik om naar dat moment te gaan' });
    const list = h('div');
    const status = h('span', { className: 'muted small' });
    const delOrig = h('input', { type: 'checkbox' });
    const dur = () => video.duration || c.duration || 1;
    timeline.addEventListener('click', e => {
      const r = timeline.getBoundingClientRect();
      video.currentTime = dur() * (e.clientX - r.left) / r.width;
    });
    const drawTimeline = () => {
      timeline.replaceChildren(
        ...segs.map(s => h('div', { className: 'seg', style: { left: `${100 * s.start / dur()}%`, width: `${100 * (s.end - s.start) / dur()}%` } }, s.name)),
        pendingStart !== null ? h('div', { className: 'seg', style: { left: `${100 * pendingStart / dur()}%`, width: `${100 * Math.max(0, video.currentTime - pendingStart) / dur()}%`, opacity: .4 } }) : null,
        h('div', { className: 'head', style: { left: `${100 * video.currentTime / dur()}%` } }));
      status.textContent = pendingStart !== null ? `Begin gezet op ${fmtTime(pendingStart)} – ga naar het einde en klik "Einde deel"` : '';
    };
    const addSeg = (start, end) => {
      if (end - start < 1) return toast('Een deel moet minstens 1 seconde duren');
      const n = segs.length;
      segs.push({ start, end, name: n === 0 ? '1e helft' : n === 1 ? '2e helft' : `deel ${n + 1}`, period: Math.min(n + 1, 4), start_minute: n === 1 ? 45 : 0 });
      segs.sort((a, b) => a.start - b.start); drawList(); drawTimeline();
    };
    const drawList = () => list.replaceChildren(...segs.map((s, i) => h('div', { className: 'row', style: { margin: '6px 0' } },
      h('b', {}, `${fmtTime(s.start)} – ${fmtTime(s.end)}`),
      h('input', { value: s.name, style: { width: '140px' }, oninput: e => { s.name = e.target.value; drawTimeline(); } }),
      h('select', { onchange: e => { s.period = Number(e.target.value); } },
        [1, 2, 3, 4].map(p => h('option', { value: p, selected: s.period === p }, p <= 2 ? `${p}e helft` : `Verlenging ${p - 2}`))),
      h('label', {}, 'start min ', h('input', { type: 'number', min: 0, value: s.start_minute, style: { width: '64px' }, onchange: e => { s.start_minute = Number(e.target.value); } })),
      h('button', { onclick: () => { video.currentTime = s.start; video.play(); } }, '▶'),
      h('button', { className: 'danger', onclick: () => { segs.splice(i, 1); drawList(); drawTimeline(); } }, '×'))));
    video.addEventListener('timeupdate', drawTimeline);
    video.addEventListener('loadedmetadata', drawTimeline);
    splitPanel.replaceChildren(h('div', { className: 'panel' },
      h('div', { className: 'row', style: { justifyContent: 'space-between' } }, h('h2', {}, `✂️ Knippen: ${c.filename}`),
        h('button', { onclick: splitCleanup }, 'Sluiten')),
      h('div', { className: 'hint' }, 'Speel af of klik op de tijdlijn. Klik "Begin deel" bij de aftrap en "Einde deel" bij het eindsignaal; herhaal voor de 2e helft. ' +
        'Alles buiten de groene delen (warming-up, rust) valt weg. Knippen kopieert de video zonder kwaliteitsverlies en is binnen enkele seconden klaar; ' +
        'een deel kan tot ~1 s eerder beginnen dan gekozen.' + (c.status === 'klaar' ? ' Let op: de delen moeten daarna opnieuw geanalyseerd worden.' : '')),
      video, timeline,
      h('div', { className: 'row' },
        h('button', { onclick: () => { pendingStart = video.currentTime; drawTimeline(); } }, '⏵ Begin deel'),
        h('button', { onclick: () => { if (pendingStart === null) return toast('Klik eerst "Begin deel"'); addSeg(pendingStart, video.currentTime); pendingStart = null; drawTimeline(); } }, '⏹ Einde deel'),
        h('button', { onclick: () => { const t = video.currentTime; segs = []; addSeg(0, t); addSeg(t, dur()); }, title: 'Twee delen: tot hier en vanaf hier' }, '⫽ Splits hier in tweeën'),
        status),
      list,
      h('div', { className: 'row', style: { marginTop: '10px' } },
        h('label', {}, delOrig, ' originele video daarna verwijderen (bespaart schijfruimte)'),
        h('span', { style: { flex: 1 } }),
        h('button', { className: 'primary', onclick: async e => {
          if (!segs.length) return toast('Voeg eerst minstens één deel toe');
          e.target.disabled = true; e.target.textContent = 'Bezig met knippen...';
          try {
            const made = await api(`/clips/${c.id}/split`, { json: { segments: segs, delete_original: delOrig.checked } });
            toast(`${made.length} nieuwe video('s) gemaakt`); splitCleanup(); draw();
          } finally { e.target.disabled = false; e.target.textContent = '✂️ Knippen'; }
        } }, '✂️ Knippen'))));
    splitPanel.scrollIntoView({ behavior: 'smooth' });
    splitCleanup = () => { video.pause(); video.removeAttribute('src'); splitPanel.replaceChildren(); splitCleanup = () => {}; };
  }

  return () => { clearTimeout(timer); splitCleanup(); };
}
