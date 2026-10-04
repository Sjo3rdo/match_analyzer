// De app slimmer maken: trainen op je eigen beelden, wanneer het jou uitkomt.
import { api, h, toast } from '../util.js';

const fmtMin = m => (m >= 90 ? `± ${Math.round(m / 60 * 10) / 10} uur` : `± ${m} min`);
const fmtEta = s => (s == null ? 'tijd wordt geschat…' : s >= 3600 ? `nog ± ${Math.round(s / 360) / 10} uur`
  : s >= 90 ? `nog ± ${Math.round(s / 60)} min` : 'bijna klaar');
const pct = v => (v == null ? '–' : `${Math.round(100 * v)}%`);

export async function render(root) {
  const statusBox = h('div');
  const cards = h('div', { className: 'train-grid' });
  root.append(
    h('div', { className: 'page-head' },
      h('div', {}, h('div', { className: 'eyebrow' }, 'Zelflerend'), h('h1', {}, 'De app slimmer maken')),
      h('a', { className: 'btn', href: '#/' }, '← Wedstrijden')),
    h('div', { className: 'hint' },
      'Zie de app als een stagiair die meekijkt: alles wat jij verbetert (het veld goed leggen, de bal aanwijzen, een toeschouwer ',
      'weghalen, spelers koppelen) schrijft hij op. Pas als jij op een knop drukt, gaat hij daarmee oefenen. Daarna vergelijkt de app ',
      'het nieuwe model met het oude op beelden die het niet heeft gezien, en gebruikt het nieuwe alleen als het echt beter is. ',
      'Terug kan altijd. Alles blijft op je eigen laptop. Tip: laat de laptop aan de lader en zet hem niet in slaap.'),
    statusBox, cards);

  let data = null, timer = null;
  async function load() {
    data = await api('/training');
    draw();
  }

  function draw() {
    const st = data.status;
    statusBox.replaceChildren(...[st.busy ? h('div', { className: 'panel train-status' },
      h('div', { className: 'row', style: { justifyContent: 'space-between' } },
        h('div', {}, h('div', { className: 'eyebrow' }, 'Bezig met trainen'), h('h2', {}, st.label)),
        h('button', { className: 'danger', onclick: async () => {
          if (!confirm('Trainen stoppen? Wat tot nu toe gedaan is, gaat verloren; het oude model blijft in gebruik.')) return;
          await api('/training/stop', { json: {} }); toast('Gestopt'); setTimeout(load, 1500);
        } }, '■ Stoppen')),
      h('div', { className: 'progress big' }, h('div', { style: { width: `${Math.round(100 * (st.frac || 0))}%` } })),
      h('div', { className: 'row small muted', style: { justifyContent: 'space-between', marginTop: '6px' } },
        h('span', {}, `${st.phase || ''}${st.message ? ' · ' + st.message : ''}`),
        h('span', {}, `${Math.round(100 * (st.frac || 0))}% · ${fmtEta(st.eta_s)}`)))
      : st.last ? h('div', { className: `panel train-result ${st.last.ok ? (st.last.better === false ? 'neutral' : 'good') : 'bad'}` },
        h('b', {}, st.last.ok ? 'Training klaar' : 'Training niet gelukt'), h('div', {}, st.last.message || '')) : null].filter(Boolean));

    const busy = st.busy;
    const approved = data.approved;
    const modelCard = (kind, title, what, unit) => {
      const k = data[kind];
      const enough = k.examples >= (kind === 'detector' ? 30 : 20);
      return h('div', { className: 'panel train-card' },
        h('div', { className: 'card-title' }, h('h2', {}, title), h('span', { className: `badge ${k.active ? 'ok' : ''}` }, k.active ? 'eigen model actief' : 'standaard')),
        h('div', { className: 'small muted' }, what),
        h('div', { className: 'stat-row' },
          h('div', { className: 'stat' }, h('b', {}, approved.length), "video's klaargezet"),
          h('div', { className: 'stat' }, h('b', {}, k.examples), unit)),
        enough ? null : h('div', { className: 'small warn' }, approved.length
          ? `Nog te weinig voorbeelden (minimaal ${kind === 'detector' ? 30 : 20}). Zet meer video's klaar bij Kalibratie.`
          : 'Zet eerst video\'s klaar: open een wedstrijd → Kalibratie → "✓ Gebruik voor training".'),
        h('div', { className: 'row', style: { marginTop: '12px' } },
          h('button', { className: 'primary', disabled: busy || !enough, onclick: () => start(kind, true, k.quick.minutes) },
            `▶ Snel trainen (${fmtMin(k.quick.minutes)})`),
          h('button', { disabled: busy || !enough, onclick: () => start(kind, false, k.thorough.minutes) },
            `Grondig (${fmtMin(k.thorough.minutes)})`)),
        k.versions.length ? h('div', { className: 'versions' },
          h('div', { className: 'eyebrow', style: { marginTop: '14px' } }, 'Eerdere trainingen'),
          ...k.versions.slice(0, 5).map(v => h('div', { className: 'list-item' },
            h('span', { style: { flex: 1 } }, h('b', {}, v.created), ' ',
              h('span', { className: 'small muted' }, kind === 'detector'
                ? `bal ${pct(v.previous?.ball)} → ${pct(v.metrics?.ball)}, spelers ${pct(v.previous?.person)} → ${pct(v.metrics?.person)}`
                : `veldlijnen ${pct(v.previous?.f1)} → ${pct(v.metrics?.f1)}`)),
            k.active === v.file ? h('span', { className: 'badge ok' }, 'in gebruik')
              : h('button', { className: 'small', onclick: () => activate(kind, v.file) }, 'Gebruik deze'))),
          k.active ? h('button', { className: 'small', style: { marginTop: '6px' }, onclick: () => activate(kind, null) }, '↺ Terug naar standaard') : null) : null);
    };

    const matchSel = h('select', {}, data.matches.map(m => h('option', { value: m.id }, `${m.name} (${m.n_linked} gekoppeld)`)));
    const playersCard = h('div', { className: 'panel train-card' },
      h('div', { className: 'card-title' }, h('h2', {}, 'Spelers herkennen')),
      h('div', { className: 'small muted' },
        'Per speler leert de app hoe hij eruitziet (houding, haar, schoenen, kleur) uit de stukken die je aan hem hebt gekoppeld. ',
        'Het profiel wordt bewaard bij je vaste selectie, zodat de app hem in een volgende wedstrijd zelf voorstelt. ',
        'Per speler kan het ook: klik op 🧠 achter zijn naam bij Spelers.'),
      data.matches.length ? h('div', { className: 'row', style: { marginTop: '12px' } }, matchSel,
        h('button', { className: 'primary', disabled: busy, onclick: () => {
          const m = data.matches.find(x => x.id === Number(matchSel.value));
          if (!m.n_linked) return toast('Koppel eerst een paar stukken aan spelers in deze wedstrijd', true);
          start('players', true, m.minutes, { match_id: m.id });
        } }, '▶ Leer deze spelers herkennen')) : h('div', { className: 'small muted' }, 'Nog geen wedstrijden.'));

    cards.replaceChildren(
      modelCard('detector', 'Spelers en bal herkennen', 'Het detectiemodel oefent op je klaargezette video\'s. Vooral de bal: wat de app alleen ingezoomd vond of wat jij aanwees, leert hij in het gewone beeld te zien.', 'beelden om van te leren'),
      modelCard('field', 'Het veld herkennen', 'Een klein netwerk leert uit goed gekalibreerde beelden welke pixels veldlijn zijn, zodat het automatisch kalibreren ook lukt op velden met slechte of halfverdwenen lijnen.', 'gekalibreerde beelden'),
      playersCard);
    clearTimeout(timer);
    if (busy) timer = setTimeout(async () => { data.status = await api('/training/status'); draw(); if (!data.status.busy) load(); }, 2500);
  }

  async function start(kind, quick, minutes, extra = {}) {
    const hint = data.device === 'cpu' ? ' Deze computer heeft geen grafische chip die de app kan gebruiken; het kan veel langer duren.' : '';
    if (!confirm(`Trainen duurt ${fmtMin(minutes)}. Je kunt de app intussen gewoon gebruiken (iets trager) en het trainen altijd stoppen.${hint} Nu starten?`)) return;
    await api('/training/start', { json: { kind, quick, ...extra } });
    toast('Trainen gestart');
    load();
  }
  async function activate(kind, file) {
    await api(`/training/${kind}/activate`, { json: { file } });
    toast(file ? 'Dit model wordt nu gebruikt' : 'Terug naar het standaardmodel');
    load();
  }

  await load();
  return () => clearTimeout(timer);
}

// Een klein label rechtsboven dat laat zien dat er getraind wordt (op elke pagina)
export function startIndicator(el) {
  const tick = async () => {
    try {
      const st = await api('/training/status');
      el.replaceChildren(...(st.busy ? [h('a', { href: '#/trainen', className: 'train-pill' },
        h('span', { className: 'pulse' }), `🧠 ${Math.round(100 * (st.frac || 0))}%`)]
        : [h('a', { href: '#/trainen', className: 'train-link' }, '🧠 Trainen')]));
      setTimeout(tick, st.busy ? 3000 : 15000);
    } catch { setTimeout(tick, 15000); }
  };
  tick();
}
