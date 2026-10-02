// Router: #/ (wedstrijden) en #/match/<id>/<tab>[?clip=<id>]
import { api, h, TEAM_COLORS } from './util.js';
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
  // Teamkleuren in de interface = gemiddelde shirtkleur uit de video (indien bekend)
  TEAM_COLORS.splice(0, 2, ...match.team_colors.map((c, i) => c || DEFAULT_COLORS[i]));
  const ctx = {
    match, params,
    reload: () => route(),
    setParam(k, v) {
      const p = new URLSearchParams(params); p.set(k, v);
      history.replaceState(null, '', `#/match/${matchId}/${tab}?${p}`);
    },
  };
  const mod = (TABS.find(t => t[0] === tab) || TABS[0])[2];
  view.append(h('h1', {}, match.name, ' ', h('span', { className: 'muted small' },
    `${match.team0_name} – ${match.team1_name}${match.date ? ' · ' + match.date : ''}`)));
  cleanup = await mod.render(view, ctx);
}

window.addEventListener('hashchange', route);
route();
