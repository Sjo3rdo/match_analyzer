// Stap 1: video's uploaden, ordenen en laten analyseren.
import { api, h, fmtTime, toast } from '../util.js';

const BUSY = ['wachtrij', 'preview', 'analyse'];

export async function render(root, ctx) {
  const { match } = ctx;
  const panel = h('div', { className: 'panel' });
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
    panel);

  let timer;
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
            busy ? h('div', { className: 'progress', title: c.message }, h('div', { style: { width: `${Math.round(100 * c.progress)}%` } })) : null,
            h('div', { className: 'muted small' }, c.message || '')),
          h('td', {}, c.n_keyframes ? h('span', { className: 'badge ok' }, `${c.n_keyframes} sleutelframe(s)`)
            : h('a', { href: `#/match/${match.id}/kalibratie?clip=${c.id}` }, 'Kalibreren')),
          h('td', {},
            h('button', { className: busy ? '' : 'primary', disabled: busy, onclick: async () => { await api(`/clips/${c.id}/process`, { method: 'POST' }); draw(); } },
              c.status === 'klaar' ? 'Opnieuw' : 'Analyseer'), ' ',
            h('button', { className: 'danger', onclick: async () => {
              if (!confirm(`${c.filename} verwijderen?`)) return;
              await api(`/clips/${c.id}`, { method: 'DELETE' }); draw();
            } }, 'Verwijder')));
      })));
    clearTimeout(timer);
    if (clips.some(c => BUSY.includes(c.status))) timer = setTimeout(draw, 2000);
  };
  await draw();
  return () => clearTimeout(timer);
}
