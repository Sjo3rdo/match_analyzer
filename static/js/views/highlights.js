// Highlights: gemarkeerde momenten en automatisch gevonden acties exporteren als video.
import { api, h, toast, fmtTime, matchMinute, playerLabel } from '../util.js';

const KINDS = { marker: '📌 Gemarkeerd', sprint: '⚡ Sprint', pass: '➡️ Pass', balverlies: '✖️ Balverlies' };

export async function render(root, ctx) {
  const { match } = ctx;
  const clipById = new Map(match.clips.map(c => [c.id, c]));
  const playerByEnt = new Map(match.players.map(p => [`p${p.id}`, p]));
  const [markers, stats] = await Promise.all([api(`/matches/${match.id}/markers`), api(`/matches/${match.id}/stats`)]);

  const items = [
    ...markers.map(m => ({ id: `m${m.id}`, kind: 'marker', clip_id: m.clip_id, t: m.t, t_end: m.t, entity: m.player_id ? `p${m.player_id}` : null, text: m.label, markerId: m.id })),
    ...stats.events.map((e, i) => ({ id: `e${i}`, kind: e.kind, clip_id: e.clip_id, t: e.t, t_end: e.t_end, entity: e.entity, to: e.to,
      text: e.kind === 'sprint' ? `${e.value} km/u` : e.kind === 'pass' ? `naar ${playerLabel(playerByEnt.get(e.to)) || 'onbekend'}` : '' })),
  ].filter(it => clipById.has(it.clip_id))
   .sort((a, b) => (clipById.get(a.clip_id).order_idx - clipById.get(b.clip_id).order_idx) || a.t - b.t);

  let kinds = new Set(['marker', 'sprint']), player = '', chosen = new Set();
  const before = h('input', { type: 'number', value: 6, min: 0, style: { width: '60px' } });
  const after = h('input', { type: 'number', value: 4, min: 0, style: { width: '60px' } });
  const list = h('div', { className: 'list', style: { maxHeight: '520px' } });
  const result = h('div');
  const exportBtn = h('button', { className: 'primary', onclick: doExport }, 'Exporteer selectie');

  root.append(
    h('div', { className: 'hint' }, 'Kies momenten en exporteer ze als één video (highlight-reel) of als los fragment. ' +
      'Kies een speler om een persoonlijke reel te maken: alle momenten van die speler worden dan geselecteerd.'),
    h('div', { className: 'panel' },
      h('div', { className: 'row' },
        Object.entries(KINDS).map(([k, l]) => h('label', {}, h('input', { type: 'checkbox', checked: kinds.has(k),
          onchange: e => { e.target.checked ? kinds.add(k) : kinds.delete(k); draw(); } }), ' ', l)),
        h('select', { onchange: e => { player = e.target.value; chosen = new Set(visible().map(it => it.id)); draw(); } },
          h('option', { value: '' }, 'Alle spelers'),
          match.players.map(p => h('option', { value: `p${p.id}` }, playerLabel(p))))),
      h('div', { className: 'row', style: { marginTop: '10px' } },
        'Seconden ervoor', before, 'erna', after,
        h('button', { onclick: () => { chosen = new Set(visible().map(it => it.id)); draw(); } }, 'Alles selecteren'),
        h('button', { onclick: () => { chosen.clear(); draw(); } }, 'Niets'),
        exportBtn)),
    h('div', { className: 'grid2' }, h('div', { className: 'panel' }, list), h('div', { className: 'panel' }, h('h3', {}, 'Export'), result)));

  function visible() {
    return items.filter(it => kinds.has(it.kind) && (!player || it.entity === player || it.to === player));
  }
  function draw() {
    const v = visible();
    list.replaceChildren(...(v.length ? v.map(it => {
      const clip = clipById.get(it.clip_id);
      return h('label', { className: 'list-item' },
        h('input', { type: 'checkbox', checked: chosen.has(it.id), onchange: e => { e.target.checked ? chosen.add(it.id) : chosen.delete(it.id); exportBtn.textContent = `Exporteer selectie (${chosen.size})`; } }),
        h('b', {}, matchMinute(clip, it.t)),
        h('span', {}, KINDS[it.kind]),
        h('span', { style: { flex: 1 } }, playerLabel(playerByEnt.get(it.entity)), ' ', h('span', { className: 'muted' }, it.text)),
        h('a', { href: `#/match/${match.id}/video?clip=${it.clip_id}&t=${Math.max(0, it.t - 3)}`, onclick: e => e.stopPropagation() }, 'bekijk'),
        it.markerId ? h('button', { className: 'danger', onclick: async e => {
          e.preventDefault();
          await api(`/markers/${it.markerId}`, { method: 'DELETE' });
          items.splice(items.indexOf(it), 1); draw();
        } }, '×') : null);
    }) : [h('div', { className: 'muted small' }, 'Geen momenten voor deze filter. Markeer momenten in "Video + minimap".')]));
    exportBtn.textContent = `Exporteer selectie (${chosen.size})`;
  }

  async function doExport() {
    const sel = items.filter(it => chosen.has(it.id));
    if (!sel.length) return toast('Selecteer eerst momenten');
    const b = Number(before.value), a = Number(after.value);
    // overlappende fragmenten in dezelfde video samenvoegen
    const parts = [];
    for (const it of sel) {
      const start = Math.max(0, it.t - b), end = Math.max(it.t, it.t_end) + a, last = parts[parts.length - 1];
      if (last && last.clip_id === it.clip_id && start <= last.end) last.end = Math.max(last.end, end);
      else parts.push({ clip_id: it.clip_id, start, end });
    }
    const name = player ? `highlights-${playerLabel(playerByEnt.get(player))}` : `highlights-${match.name}`;
    exportBtn.disabled = true;
    result.replaceChildren(h('div', { className: 'muted' }, `Bezig met ${parts.length} fragment(en) exporteren...`));
    try {
      const r = await api(`/matches/${match.id}/export`, { json: { name, items: parts } });
      result.replaceChildren(h('video', { src: r.url, controls: true, style: { width: '100%', borderRadius: '8px' } }),
        h('p', {}, h('a', { className: 'btn primary', href: r.url, download: r.file }, 'Download ' + r.file)));
    } catch { result.replaceChildren(h('div', { className: 'badge err' }, 'Export mislukt')); }
    exportBtn.disabled = false;
  }
  draw();
}
