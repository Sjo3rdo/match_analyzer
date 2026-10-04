// Router: #/ (wedstrijden) en #/match/<id>/<tab>[?clip=<id>]
import { api, h, TEAM_COLORS } from './util.js';
import { setPitchSize } from './pitch.js';
import * as matches from './views/matches.js';
import * as clips from './views/clips.js';
import * as calibrate from './views/calibrate.js';
import * as players from './views/players.js';
import * as video from './views/video.js';
import * as stats from './views/stats.js';
import * as moments from './views/moments.js';
import * as training from './views/training.js';

// Tabbladen; de eerste drie zijn de stappen die je doorloopt (met een ✓ als ze klaar zijn en een
// geel bolletje bij de volgende stap), de rest is om te bekijken.
const TABS = [
  ['clips', 'Video\'s', clips],
  ['kalibratie', 'Kalibratie', calibrate],
  ['spelers', 'Spelers', players],
  ['video', 'Video + minimap', video],
  ['statistieken', 'Statistieken', stats],
  ['momenten', 'Clips & delen', moments],
];

// Stappen van een wedstrijd: wat is klaar en wat is de volgende stap?
function steps(match) {
  const clips = match.clips;
  const analyzed = clips.filter(c => c.status === 'klaar');
  const busy = clips.some(c => ['wachtrij', 'preview', 'analyse'].includes(c.status));
  const videos = clips.length > 0 && analyzed.length === clips.length;
  const calib = analyzed.length > 0 && analyzed.every(c => (c.n_manual_keyframes ?? c.n_keyframes) > 0);
  const players = match.players.length > 0 && match.n_assigned > 0;
  return {
    clips: { done: videos, todo: !clips.length ? 'Upload je video\'s' : busy ? 'Even wachten: de analyse loopt' : 'Klik op "Analyseer" bij elke video' },
    kalibratie: { done: calib, todo: 'Leg het veld op het beeld' },
    spelers: { done: players, todo: 'Voer de selectie in en koppel spelers' },
  };
}

function renderTabs(el, match, tab) {
  const st = steps(match);
  const next = ['clips', 'kalibratie', 'spelers'].find(k => !st[k].done);
  el.replaceChildren(...TABS.map(([key, label], i) => {
    const s = st[key];
    const mark = !s ? null : s.done ? h('span', { className: 'tabmark done' }, '✓')
      : h('span', { className: `tabmark ${key === next ? 'next' : ''}` }, i + 1);
    return h('a', {
      href: `#/match/${match.id}/${key}`, className: `${key === tab ? 'active' : ''} ${key === next ? 'next' : ''}`,
      title: s ? (s.done ? 'Klaar' : key === next ? `Volgende stap: ${s.todo}` : s.todo) : '',
    }, mark, label);
  }));
}

let cleanup = null;
const DEFAULT_COLORS = [...TEAM_COLORS];

async function route() {
  if (cleanup) { try { cleanup(); } catch {} cleanup = null; }
  const view = document.getElementById('view');
  const tabs = document.getElementById('tabs');
  view.replaceChildren();
  tabs.replaceChildren();
  const [path, query] = location.hash.slice(1).split('?');
  const params = new URLSearchParams(query || '');
  const parts = path.split('/').filter(Boolean);
  if (parts[0] === 'trainen') {
    cleanup = await training.render(view);
    return;
  }
  if (parts[0] !== 'match') {
    cleanup = await matches.render(view);
    return;
  }
  const matchId = Number(parts[1]);
  const tab = parts[2] || 'clips';
  const match = await api(`/matches/${matchId}`);
  setPitchSize(match.pitch_length, match.pitch_width);
  // Teamkleuren in de interface = gemiddelde shirtkleur uit de video (indien bekend)
  TEAM_COLORS.splice(0, 2, ...match.team_colors.map((c, i) => c || DEFAULT_COLORS[i]));
  renderTabs(tabs, match, tab);
  const ctx = {
    match, params,
    reload: () => route(),
    async refreshSteps() { renderTabs(tabs, await api(`/matches/${matchId}`), tab); },
    setParam(k, v) {
      const p = new URLSearchParams(params); p.set(k, v);
      history.replaceState(null, '', `#/match/${matchId}/${tab}?${p}`);
    },
  };
  const mod = (TABS.find(t => t[0] === tab) || TABS[0])[2];
  view.append(h('div', { className: 'match-head' },
    h('div', {}, h('div', { className: 'eyebrow' }, 'Wedstrijd'), h('h1', {}, match.name)),
    h('span', { className: 'vs' }, h('span', { className: 'dot', style: { background: TEAM_COLORS[0] } }), match.team0_name,
      match.score ? h('b', { className: 'score', title: 'Stand uit de bevestigde goals' }, ` ${match.score[0]} – ${match.score[1]} `) : ' – ',
      h('span', { className: 'dot', style: { background: TEAM_COLORS[1], marginLeft: '4px' } }), match.team1_name),
    match.date ? h('span', { className: 'date' }, match.date) : null));
  cleanup = await mod.render(view, ctx);
}

api('/version').then(v => {
  document.getElementById('version').textContent = `versie ${v.version}${v.commit ? ' · ' + v.commit : ''}`;
}).catch(() => {});

training.startIndicator(document.getElementById('train-indicator'));

// licht/donker (standaard donker); de keuze onthouden we in deze browser
const themeBtn = document.getElementById('theme-toggle');
function showTheme() {
  const light = document.documentElement.dataset.theme === 'light';
  themeBtn.textContent = light ? '☾' : '☀︎';
  themeBtn.title = light ? 'Donker thema' : 'Licht thema';
}
themeBtn.addEventListener('click', () => {
  const next = document.documentElement.dataset.theme === 'light' ? 'dark' : 'light';
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem('theme', next); } catch {}
  showTheme();
  window.dispatchEvent(new Event('resize'));  // canvassen (veld, minimap) opnieuw tekenen
});
showTheme();
window.addEventListener('hashchange', route);
route();
