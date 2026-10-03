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

const TABS = [
  ['clips', '1. Video\'s', clips],
  ['kalibratie', '2. Kalibratie', calibrate],
  ['spelers', '3. Spelers', players],
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
  return [
    { key: 'clips', label: 'Video\'s', done: videos,
      todo: !clips.length ? 'Upload je video\'s' : busy ? 'Even wachten: de analyse loopt' : 'Klik op "Analyseer" bij elke video' },
    { key: 'kalibratie', label: 'Kalibratie', done: calib, todo: 'Leg het veld op het beeld' },
    { key: 'spelers', label: 'Spelers', done: players, todo: 'Voer de selectie in en koppel spelers' },
    { key: 'statistieken', label: 'Bekijken', done: false, todo: 'Bekijk statistieken, video en clips' },
  ];
}

function renderSteps(el, match, tab) {
  const list = steps(match);
  const next = list.find(s => !s.done) || list[list.length - 1];
  const viewTabs = ['statistieken', 'video', 'momenten'];
  const here = viewTabs.includes(tab) ? 'statistieken' : tab;
  el.replaceChildren(
    h('div', { className: 'steps' }, list.flatMap((s, i) => [
      i ? h('span', { className: 'muted' }, '›') : null,
      h('a', {
        href: `#/match/${match.id}/${s.key}`,
        className: `step ${s.done ? 'done' : ''} ${s.key === here ? 'here' : ''} ${s === next ? 'next' : ''}`,
      }, h('span', { className: 'num' }, s.done ? '✓' : i + 1), s.label)]).filter(Boolean)),
    next.key !== here
      ? h('a', { className: 'btn primary', href: `#/match/${match.id}/${next.key}` }, `Volgende stap: ${next.label.toLowerCase()} →`)
      : h('span', { className: 'muted small' }, next.done ? '' : `Nu: ${next.todo}`));
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
  if (parts[0] !== 'match') {
    cleanup = await matches.render(view);
    return;
  }
  const matchId = Number(parts[1]);
  const tab = parts[2] || 'clips';
  for (const [key, label] of TABS) {
    tabs.append(h('a', { href: `#/match/${matchId}/${key}`, className: key === tab ? 'active' : '' }, label));
  }
  const match = await api(`/matches/${matchId}`);
  setPitchSize(match.pitch_length, match.pitch_width);
  // Teamkleuren in de interface = gemiddelde shirtkleur uit de video (indien bekend)
  TEAM_COLORS.splice(0, 2, ...match.team_colors.map((c, i) => c || DEFAULT_COLORS[i]));
  const stepBar = h('div', { className: 'stepbar' });
  renderSteps(stepBar, match, tab);
  const ctx = {
    match, params,
    reload: () => route(),
    async refreshSteps() { renderSteps(stepBar, await api(`/matches/${matchId}`), tab); },
    setParam(k, v) {
      const p = new URLSearchParams(params); p.set(k, v);
      history.replaceState(null, '', `#/match/${matchId}/${tab}?${p}`);
    },
  };
  const mod = (TABS.find(t => t[0] === tab) || TABS[0])[2];
  view.append(h('h1', {}, match.name, ' ', h('span', { className: 'muted small' },
    `${match.team0_name} – ${match.team1_name}${match.date ? ' · ' + match.date : ''}`)), stepBar);
  cleanup = await mod.render(view, ctx);
}

api('/version').then(v => {
  document.getElementById('version').textContent = `versie ${v.version}${v.commit ? ' · ' + v.commit : ''}`;
}).catch(() => {});

window.addEventListener('hashchange', route);
route();
