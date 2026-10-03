// Stap 1: video's uploaden, ordenen en laten analyseren.
import { api, h, fmtTime, toast } from '../util.js';

const BUSY = ['wachtrij', 'preview', 'analyse'];

export async function render(root, ctx) {
  const { match } = ctx;
  const panel = h('div', { className: 'panel' });
  const splitPanel = h('div');
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
      toast('Geüpload'); ctx.reload();
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
    panel, splitPanel);

  let timer, wasBusy = null;
  const draw = async () => {
    const m = await api(`/matches/${match.id}`);
    const clips = m.clips;
    panel.replaceChildren(h('h2', {}, 'Video\'s van deze wedstrijd'));
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
      h('tr', {}, ['', 'Bestand', 'Duur', 'Helft', 'Start (min)', 'Status', 'Kalibratie', ''].map(t => h('th', {}, t))),
      clips.map((c, i) => {
        const busy = BUSY.includes(c.status);
        const badge = c.status === 'klaar' ? 'ok' : c.status === 'fout' ? 'err' : busy ? 'busy' : '';
        return h('tr', {},
          h('td', {}, h('button', { onclick: () => move(i, -1), title: 'Omhoog' }, '↑'), h('button', { onclick: () => move(i, 1), title: 'Omlaag' }, '↓')),
          h('td', {}, c.filename, h('div', { className: 'muted small' }, `${c.width}×${c.height} · ${Math.round(c.fps)} fps`)),
          h('td', {}, fmtTime(c.duration)),
          h('td', {}, h('select', { onchange: e => patch(c, { period: Number(e.target.value) }) },
            [1, 2, 3, 4].map(p => h('option', { value: p, selected: c.period === p }, p <= 2 ? `${p}e helft` : `Verlenging ${p - 2}`)))),
          h('td', {}, h('input', { type: 'number', min: 0, step: 1, value: c.start_minute, style: { width: '70px' },
            onchange: e => patch(c, { start_minute: Number(e.target.value) }) })),
          h('td', {}, h('span', { className: `badge ${badge}` }, c.status),
            c.status === 'klaar' && (c.analysis_version || 0) < 2
              ? h('div', { className: 'badge err', title: 'Deze video is geanalyseerd met een oudere versie die de tijden van iPhone-video\'s verkeerd las. Klik op "Opnieuw".' }, 'opnieuw analyseren aanbevolen') : null,
            busy ? h('div', { className: 'progress', title: c.message }, h('div', { style: { width: `${Math.round(100 * c.progress)}%` } })) : null,
            h('div', { className: 'muted small' }, c.message || '')),
          h('td', {}, c.n_keyframes ? h('span', { className: 'badge ok' }, `${c.n_keyframes} sleutelframe(s)`)
            : h('a', { href: `#/match/${match.id}/kalibratie?clip=${c.id}` }, 'Kalibreren')),
          h('td', {},
            h('button', { className: busy ? '' : 'primary', disabled: busy, onclick: async () => { await api(`/clips/${c.id}/process`, { method: 'POST' }); draw(); ctx.refreshSteps(); } },
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
      if (clips.every(c => c.status === 'klaar')) toast('Analyse klaar. Volgende stap: kalibreren (knop bovenaan).');
    }
    wasBusy = busyNow;
    if (busyNow) timer = setTimeout(draw, 2000);
  };
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
