// Kleine hulpfuncties: API-aanroepen, DOM bouwen, meldingen.
export async function api(path, opts = {}) {
  const init = { ...opts };
  if (opts.json !== undefined) {
    init.method = init.method || 'POST';
    init.headers = { 'Content-Type': 'application/json' };
    init.body = JSON.stringify(opts.json);
  }
  const res = await fetch('/api' + path, init);
  if (!res.ok) {
    let msg = res.statusText;
    try { msg = (await res.json()).detail || msg; } catch {}
    toast(msg, true);
    throw new Error(msg);
  }
  return res.headers.get('content-type')?.includes('json') ? res.json() : res;
}

export function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === undefined || v === null || v === false) continue;
    if (k.startsWith('on')) el.addEventListener(k.slice(2).toLowerCase(), v);
    else if (k === 'style' && typeof v === 'object') Object.assign(el.style, v);
    else if (k in el && k !== 'list') el[k] = v;
    else el.setAttribute(k, v);
  }
  for (const c of children.flat()) {
    if (c === null || c === undefined || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

let toastTimer;
export function toast(msg, error = false) {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.style.background = error ? 'var(--danger)' : '';
  el.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove('show'), 3500);
}

export function fmtTime(s) {
  if (s == null || isNaN(s)) return '–';
  const m = Math.floor(s / 60), sec = Math.floor(s % 60);
  return `${m}:${String(sec).padStart(2, '0')}`;
}

export function matchMinute(clip, t) {
  return `${Math.floor((clip.start_minute || 0) + t / 60)}'`;
}

export const TEAM_COLORS = ['#2f6fdb', '#e0472f', '#9a9a9a'];
export function teamName(match, team) {
  if (team === 0) return match.team0_name;
  if (team === 1) return match.team1_name;
  if (team === 2) return 'Overig (scheids/keeper)';
  if (team === 3) return 'Toeschouwer (telt niet mee)';
  return 'Onbekend';
}

// Keuzelijst voor het team van een track (met de optie om iemand als toeschouwer weg te halen)
export function teamOptions(match, current) {
  return [[0, teamName(match, 0)], [1, teamName(match, 1)], [2, 'Overig (scheids/keeper)'],
    [3, '🚫 Toeschouwer (weghalen)'], [-1, 'Onbekend']].map(([v, l]) => h('option', { value: v, selected: current === v }, l));
}

// Een track als toeschouwer weghalen; daarna vergelijkbare personen (zelfde plek, zelfde kleding)
// in één keer mee laten weghalen. box: element waarin de vraag komt; done: na afloop.
export async function markSpectator(clipId, trackId, box, done) {
  const r = await api(`/clips/${clipId}/tracks/${trackId}`, { method: 'PATCH', json: { team: 3 } });
  toast('Weggehaald: telt nergens meer mee');
  const sim = r.similar || [];
  if (!sim.length || !box) { done?.(); return; }
  const close = () => { box.replaceChildren(); done?.(); };
  box.replaceChildren(h('div', { className: 'hint', style: { marginBottom: '12px' } },
    h('b', {}, `Nog ${sim.length} vergelijkbare ${sim.length === 1 ? 'persoon' : 'personen'} gevonden`),
    h('div', { className: 'small muted', style: { margin: '4px 0 8px' } },
      'Zelfde soort kleding en op dezelfde plek (ook in je andere video\'s van deze wedstrijd). Ook weghalen?'),
    h('div', { className: 'row', style: { gap: '6px', marginBottom: '8px' } }, sim.slice(0, 16).map(s =>
      h('img', { src: `/api/clips/${s.clip_id}/thumb/${s.track_id}`, title: s.reason, style: { height: '70px', borderRadius: '4px' } }))),
    h('div', { className: 'row' },
      h('button', { className: 'primary', onclick: async () => {
        await api('/tracks/spectators', { json: { items: sim } });
        toast(`${sim.length} extra weggehaald`); close();
      } }, `Ja, alle ${sim.length} weghalen`),
      h('button', { onclick: close }, 'Nee'))));
}

export function playerLabel(p) {
  return p ? `${p.number ? '#' + p.number + ' ' : ''}${p.name}` : '';
}

export function select(options, value, onchange, attrs = {}) {
  const s = h('select', { ...attrs, onchange: e => onchange(e.target.value) },
    options.map(([v, label]) => h('option', { value: v, selected: String(v) === String(value ?? '') }, label)));
  return s;
}

export const MOMENT_LABELS = ['Goal', 'Kans', 'Schot', 'Redding', 'Assist', 'Duel', 'Opbouw', 'Pressing', 'Omschakeling', 'Standaardsituatie', 'Fout', 'Leermoment'];
export function labelList() {
  return h('datalist', { id: 'moment-labels' }, MOMENT_LABELS.map(l => h('option', { value: l })));
}

// Delen via de deelknop van de Mac (Safari): AirDrop, Berichten, Mail, WhatsApp, ...
// Het bestand wordt eerst opgehaald; delen zelf moet daarna met een nieuwe klik (browsereis).
export async function prepareShare(url, filename) {
  const blob = await (await fetch(url)).blob();
  return new File([blob], filename, { type: blob.type || 'video/mp4' });
}
export function canShareFiles(file) {
  try { return !!(navigator.canShare && navigator.canShare({ files: [file] })); } catch { return false; }
}
export async function shareFile(file, title) {
  try {
    await navigator.share({ files: [file], title });
  } catch (e) {
    if (e.name !== 'AbortError') toast('Delen lukte niet: ' + e.message, true);
  }
}

// Exportpaneel: maakt de video, en toont daarna Download / Deel / Toon in Finder
export function exportPanel() {
  const box = h('div');
  async function run(matchId, body, title) {
    box.replaceChildren(h('div', { className: 'muted' }, '⏳ Video maken... (een paar seconden per clip)'));
    let r;
    try { r = await api(`/matches/${matchId}/export`, { json: body }); }
    catch { box.replaceChildren(h('span', { className: 'badge err' }, 'Exporteren mislukt')); return; }
    const isZip = r.file.endsWith('.zip');
    const shareBtn = h('button', { className: 'primary', disabled: true }, 'Delen voorbereiden...');
    box.replaceChildren(
      isZip ? null : h('video', { src: r.url, controls: true, playsInline: true, style: { width: '100%', borderRadius: '8px' } }),
      h('div', { className: 'row', style: { marginTop: '8px' } },
        shareBtn,
        h('a', { className: 'btn', href: r.url, download: r.file }, '⬇️ Download'),
        h('button', { onclick: () => api(`/exports/${r.file}/reveal`, { method: 'POST' }) }, 'Toon in Finder')),
      h('div', { className: 'small muted' }, r.file));
    try {
      const file = await prepareShare(r.url, r.file);
      if (canShareFiles(file)) {
        shareBtn.disabled = false;
        shareBtn.textContent = '📤 Deel (AirDrop, WhatsApp, Mail...)';
        shareBtn.onclick = () => shareFile(file, title);
      } else {
        shareBtn.textContent = 'Delen kan niet in deze browser (gebruik Safari)';
      }
    } catch { shareBtn.textContent = 'Delen niet beschikbaar'; }
  }
  return { el: box, run };
}
