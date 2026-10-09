// Standplaatsen: één video per plek zelf kalibreren, de rest laat de app voorstellen; hier controleer je alles.
import { api, h, toast } from '../util.js';

const STATE = {
  eigen: ['ok', '✓ zelf gekalibreerd'],
  goedgekeurd: ['ok', '✓ goedgekeurd'],
  voorstel: ['busy', 'voorstel: controleer'],
  mislukt: ['err', '⚠ niet gelukt'],
  afgekeurd: ['err', '✗ afgekeurd'],
  open: ['', 'nog niet gekalibreerd'],
  niet_geanalyseerd: ['', 'eerst analyseren'],
};

// openClip(id): open die video in de kalibratie hieronder
export function stationsPanel(match, openClip) {
  const box = h('div', { className: 'panel stations' });
  let timer = null, open = match.clips.length > 3, stamp = Date.now();

  async function draw() {
    const data = await api(`/matches/${match.id}/stations`);
    const job = data.job || {};
    const busy = job.status === 'wachtrij' || job.status === 'bezig';
    const all = data.stations.flatMap(s => s.clips);
    const n = k => all.filter(c => c.state === k).length;
    const head = h('div', { className: 'row', style: { justifyContent: 'space-between', cursor: 'pointer' },
      onclick: () => { open = !open; draw(); } },
      h('h2', {}, `${open ? '▾' : '▸'} Alle video's in één keer`),
      h('span', { className: 'small muted' },
        `${n('eigen') + n('goedgekeurd')} klaar · ${n('voorstel')} te controleren · ${n('mislukt') + n('afgekeurd') + n('open')} nog te doen`));
    if (!open) { box.replaceChildren(head); return schedule(busy); }
    const noAnchor = data.stations.filter(s => !s.anchor && s.clips.some(c => c.state !== 'niet_geanalyseerd'));
    box.replaceChildren(head,
      h('div', { className: 'small muted' },
        'Zie het als een fotograaf op een statief: vanaf dezelfde plek zijn positie, hoogte en zoom gelijk, alleen de kijkrichting ',
        'verschilt. Kalibreer per standplaats één video zelf; de app legt de andere video\'s daarna als puzzelstukjes tegen de ',
        'omgeving (bomen, huizen, borden) of zoekt de veldlijnen. Daarna loop je hieronder de plaatjes langs: liggen de gele ',
        'lijnen op de witte lijnen? ✓ of ✗.'),
      h('div', { className: 'row', style: { margin: '10px 0' } },
        h('button', { className: 'primary', disabled: busy, onclick: async () => {
          await api(`/matches/${match.id}/stations/run`, { json: {} }); toast('Gestart'); draw();
        } }, '🤖 Kalibreer de rest automatisch'),
        busy ? h('div', { className: 'progress', style: { width: '220px' } }, h('div', { style: { width: `${Math.round(100 * (job.progress || 0))}%` } })) : null,
        h('span', { className: 'small muted' }, job.message || ''),
        noAnchor.length ? h('span', { className: 'small warn' }, `${noAnchor.length} standplaats(en) wachten op één eigen kalibratie`) : null),
      ...data.stations.map(st => station(st)));
    schedule(busy);
  }

  function station(st) {
    const todo = st.clips.filter(c => c.state === 'voorstel');
    const pick = st.clips.find(c => c.state !== 'niet_geanalyseerd') || st.clips[0];
    return h('div', { className: 'station' },
      h('div', { className: 'row', style: { justifyContent: 'space-between' } },
        h('div', {}, h('b', {}, st.label), ' ',
          h('span', { className: 'small muted' }, `${st.clips.length} video('s)`,
            st.acc_m ? ` · GPS ± ${st.acc_m} m${st.precision_m ? `, samen ± ${st.precision_m} m` : ''}` : '')),
        h('div', { className: 'row small' },
          st.anchor
            ? h('span', { className: 'badge ok', title: `Plek volgt uit je kalibratie van ${st.anchor.filename}` },
              `plek: x ${st.anchor.x} · y ${st.anchor.y} · ${st.anchor.h} m hoog`)
            : h('span', {}, h('span', { className: 'small warn' }, 'Kalibreer één video hiervan zelf '),
              h('button', { className: 'small', onclick: () => openClip(pick.id) }, '✎ ' + pick.filename)),
          todo.length ? h('button', { className: 'small', onclick: async () => {
            for (const c of todo) await api(`/clips/${c.id}/calib-review`, { json: { status: 'goedgekeurd' } });
            toast(`${todo.length} goedgekeurd`); draw();
          } }, `✓ Alle ${todo.length} goedkeuren`) : null)),
      h('div', { className: 'thumbs' }, st.clips.map(c => card(c))));
  }

  function card(c) {
    const [cls, text] = STATE[c.state] || ['', c.state];
    const img = c.has_calibration
      ? h('a', { href: `/api/clips/${c.id}/calib-thumb?w=1600&v=${stamp}`, target: '_blank', title: 'Groot bekijken' },
        h('img', { src: `/api/clips/${c.id}/calib-thumb?w=360&v=${stamp}`, loading: 'lazy', alt: c.filename }))
      : h('div', { className: 'thumb-empty' }, c.state === 'niet_geanalyseerd' ? 'nog niet geanalyseerd' : 'geen kalibratie');
    const review = async status => {
      await api(`/clips/${c.id}/calib-review`, { json: { status } });
      stamp = Date.now(); draw();
    };
    return h('div', { className: `thumb ${c.state}` },
      img,
      h('div', { className: 'row small', style: { justifyContent: 'space-between', marginTop: '4px' } },
        h('span', { className: 'muted', title: c.filename, style: { overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: '150px' } }, c.filename),
        h('span', { className: `badge ${cls}` }, text)),
      c.method && c.state === 'voorstel' ? h('div', { className: 'small muted' }, c.method === 'omgeving' ? 'via de omgeving' : 'via de veldlijnen') : null,
      c.uncertain && c.state === 'voorstel' ? h('div', { className: 'small warn', title: 'Het panorama is vanaf je ijkmoment ver doorgedraaid; kleine afwijkingen in de zoom tellen dan op.' },
        `Onzeker: ${c.turn_deg}° weggedraaid van je ijkmoment. Controleer goed, of kalibreer een video die deze kant op kijkt zelf.`) : null,
      h('div', { className: 'row small', style: { marginTop: '4px' } },
        c.state === 'voorstel' ? h('button', { className: 'small primary', onclick: () => review('goedgekeurd'), title: 'De gele lijnen liggen op de witte' }, '✓') : null,
        c.state === 'voorstel' ? h('button', { className: 'small danger', onclick: () => review('afgekeurd'), title: 'Klopt niet: weghalen' }, '✗') : null,
        h('button', { className: 'small', onclick: () => openClip(c.id), title: 'Openen om zelf te kalibreren of bij te stellen' }, '✎')));
  }

  function schedule(busy) {
    clearTimeout(timer);
    if (busy) timer = setTimeout(() => { stamp = Date.now(); draw(); }, 3000);
  }

  draw();
  return { el: box, refresh: () => { stamp = Date.now(); draw(); }, cleanup: () => clearTimeout(timer) };
}
