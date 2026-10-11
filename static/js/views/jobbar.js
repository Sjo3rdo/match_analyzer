// Statusbalk: wat doet de app nu op de achtergrond, hoe lang nog, en wat kun jij nu het best doen?
// Zoals het bord bij de bakker: "uw bestelling is bijna klaar, u kunt intussen verder winkelen".
import { api, h, toast } from '../util.js';

const mins = s => (s < 90 ? 'minder dan 2 min' : `± ${Math.round(s / 60)} min`);

// opts: { wizard, openClip(id), openStations(), onDone() }
export function jobBar(match, opts = {}) {
  const el = h('div', { className: 'jobbar' });
  let timer = null, wasBusy = null, alive = true, nextCache = null, nextAt = 0;

  async function nextStep(force) {
    if (!force && nextCache && Date.now() - nextAt < 15000) return nextCache;
    const data = await api(`/matches/${match.id}/stations`).catch(() => null);
    if (!data) return null;
    const usable = data.stations.filter(s => !s.no_gps && s.clips.some(c => c.state !== 'niet_geanalyseerd'));
    const all = data.stations.flatMap(s => s.clips);
    const count = k => all.filter(c => c.state === k).length;
    const noAnchor = usable.filter(s => !s.anchor);
    const confs = all.map(c => c.confidence?.pct).filter(v => v != null);
    const avg = confs.length ? Math.round(confs.reduce((a, b) => a + b, 0) / confs.length) : null;
    let out;
    if (usable.length && noAnchor.length === usable.length) {  // eerste keer: nog niets zelf gekalibreerd
      out = { text: [h('b', {}, 'Begin hier: '),
        `kalibreer per standplaats één video zelf (de app vond ${usable.length} plek${usable.length > 1 ? 'ken' : ''} waar je filmde); de rest doet de app.`],
        title: 'Verwachting: vanaf dezelfde plek en dezelfde kant op kijkend meestal binnen een meter. Kijk je ver de andere kant op ' +
          '(meer dan 45°), dan vraagt de app om één extra ijkmoment aan die kant.',
        station: noAnchor[0] };
    } else if (noAnchor.length) {
      out = { text: [h('b', {}, `Nog ${noAnchor.length} standplaats${noAnchor.length > 1 ? 'en' : ''} zonder eigen ijkmoment. `),
        `Volgende: ${noAnchor[0].label}.`], station: noAnchor[0] };
    } else {
      const unsure = all.filter(c => c.state === 'voorstel' && c.uncertain).sort((a, b) => (b.turn_deg || 0) - (a.turn_deg || 0));
      if (unsure.length) out = { text: [h('b', {}, `Kalibreer ook ${unsure[0].filename} zelf. `),
        `Die kijkt ${unsure[0].turn_deg}° een andere kant op dan je ijkmoment; één ijkmoment daar maakt die kant net zo nauwkeurig.`],
        clip: unsure[0] };
      else if (count('voorstel')) out = { text: [h('b', {}, `${count('voorstel')} voorstel${count('voorstel') > 1 ? 'len' : ''} klaar om te controleren`),
        avg != null ? ` (gemiddeld ${avg}% zeker).` : '.', ' Liggen de gele lijnen op de witte? Keur ze goed.'], review: true };
      else if (count('mislukt') || count('open')) out = { text: [h('b', {}, `${count('mislukt') + count('open')} video('s) nog zonder kalibratie. `),
        'Kalibreer er één zelf, of start het automatisch kalibreren opnieuw.'], clip: all.find(c => c.state === 'mislukt' || c.state === 'open') };
      else out = { text: [h('b', {}, '✓ Alle video\'s zijn gekalibreerd'), avg != null ? ` (gemiddeld ${avg}% zeker).` : '.'], done: true };
    }
    nextCache = out; nextAt = Date.now();
    return out;
  }

  async function draw() {
    if (!alive) return;
    const j = await api(`/matches/${match.id}/jobs`).catch(() => null);
    if (!j || !alive) return schedule(false);
    const parts = [];
    if (j.analysis.running.length || j.analysis.queued) {
      const r = j.analysis.running[0];
      parts.push(`Analyseren: ${j.analysis.done} van ${j.analysis.total} klaar${r ? ` · nu ${r.filename} (${Math.round(100 * r.progress)}%)` : ''}`);
    }
    if (j.stations.status === 'bezig' || j.stations.status === 'wachtrij') {
      const p = Math.round(100 * (j.stations.progress || 0));
      parts.push(j.stations.status === 'wachtrij' ? 'De andere video\'s kalibreren: begint zodra het bijstellen klaar is'
        : `De andere video's kalibreren: ${j.stations.message || p + '%'}${j.stations.eta_s != null ? ` · nog ${mins(j.stations.eta_s)}` : ''}`);
    }
    const ac = j.autocalib.running.length + j.autocalib.queued, r1 = j.autocalib.running[0];
    if (ac) parts.push(`Bijstellen op de veldlijnen: ${r1 ? `${r1.filename} ${Math.round(100 * r1.progress)}%` : ''}` +
      `${ac - (r1 ? 1 : 0) > 0 ? `${r1 ? ', daarna nog' : ''} ${ac - (r1 ? 1 : 0)} video('s)` : ''}`);
    if (j.busy) {
      const p = j.stations.status === 'bezig' ? j.stations.progress || 0 : null;
      el.className = 'jobbar busy';
      el.replaceChildren(...[h('span', { className: 'pulse' }),
        h('div', { style: { flex: 1 } }, h('div', {}, parts.join(' · ')),
          h('div', { className: 'small muted' }, 'Dit gaat op de achtergrond door. Je kunt gewoon verder kalibreren of iets anders doen.')),
        p != null ? h('div', { className: 'progress', style: { width: '160px' } }, h('div', { style: { width: `${Math.round(100 * p)}%` } })) : null]
        .filter(Boolean));
    } else if (!opts.wizard) {
      const n = await nextStep(wasBusy === true);
      if (!alive) return;
      el.className = `jobbar ${n?.done ? 'done' : ''}`;
      let btn = null;
      if (n?.station) btn = h('button', { className: 'primary small', onclick: async () => {
        const c = await api(`/matches/${match.id}/stations/${n.station.id}/best`).catch(() => null);
        if (c) opts.openClip?.(c.id);
      } }, `✎ Kalibreer ${n.station.label}`);
      else if (n?.clip) btn = h('button', { className: 'primary small', onclick: () => opts.openClip?.(n.clip.id) }, `✎ ${n.clip.filename}`);
      else if (n?.review) btn = h('button', { className: 'primary small', onclick: () => opts.openStations?.() }, 'Bekijk en keur goed');
      else if (n?.done) btn = h('a', { className: 'btn primary small', href: `#/match/${match.id}/spelers` }, 'Verder: spelers →');
      el.replaceChildren(...[h('div', { style: { flex: 1 }, title: n?.title || '' }, ...(n ? n.text : ['Klaar.']),
        n?.title ? h('span', { className: 'muted' }, ' ⓘ') : null), btn].filter(Boolean));
    } else {
      el.className = 'jobbar'; el.replaceChildren();
    }
    if (wasBusy && !j.busy) {
      toast(j.stations.message ? `Klaar met kalibreren: ${j.stations.message}` : 'Klaar op de achtergrond');
      opts.onDone?.();
    }
    wasBusy = j.busy;
    opts.onDraw?.();
    schedule(j.busy);
  }

  function schedule(busy) {
    clearTimeout(timer);
    timer = setTimeout(draw, busy ? 2500 : 15000);
  }

  draw();
  return { el, refresh: () => { nextCache = null; draw(); }, cleanup: () => { alive = false; clearTimeout(timer); } };
}
