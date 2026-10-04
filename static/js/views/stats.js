// Statistieken: teams, spelers, heatmap, teamvorm en passnetwerk.
import { api, h, toast, TEAM_COLORS, teamName, playerLabel } from '../util.js';
import { PitchView } from '../pitch.js';

export async function render(root, ctx) {
  const { match } = ctx;
  const [stats, shots] = await Promise.all([api(`/matches/${match.id}/stats`),
    api(`/matches/${match.id}/shots`).catch(() => ({ shots: [], timeline: [], teams: [{}, {}], players: {}, score: [0, 0] }))]);
  const playerById = new Map(match.players.map(p => [p.id, p]));
  for (const p of stats.players) {
    const q = shots.players[String(p.player_id)] || {};
    p.shots = q.shots || 0; p.goals = q.goals || 0;
  }
  let onlyNamed = stats.players.some(p => p.player_id);
  let sortKey = null, sortDir = -1, selected = null, networkTeam = 0;

  const warnings = stats.clips.filter(c => c.status !== 'klaar' || !c.calibrated);
  if (warnings.length) root.append(h('div', { className: 'hint' },
    'Niet meegeteld: ', warnings.map(c => `${c.filename} (${c.status !== 'klaar' ? 'nog niet geanalyseerd' : 'niet gekalibreerd'})`).join(', '),
    '. Statistieken in meters vereisen kalibratie.'));
  if (!stats.players.length) {
    root.append(h('div', { className: 'panel empty' }, 'Nog geen statistieken. Analyseer en kalibreer eerst minstens één video.'));
    return;
  }

  root.append(h('div', { className: 'grid-cols' }, stats.teams.map(t => h('div', { className: 'panel' },
    h('h2', {}, h('span', { className: 'dot', style: { background: TEAM_COLORS[t.team] } }), t.name),
    h('div', {},
      h('span', { className: 'stat' }, h('b', {}, t.possession_pct != null ? `${t.possession_pct}%` : '–'), 'balbezit'),
      h('span', { className: 'stat' }, h('b', {}, t.passes), 'geslaagde passes'),
      h('span', { className: 'stat' }, h('b', {}, t.pass_accuracy_pct != null ? `${t.pass_accuracy_pct}%` : '–'), 'passnauwkeurigheid'),
      h('span', { className: 'stat' }, h('b', {}, (sumTeam(t.team, 'distance_m') / 1000).toFixed(1) + ' km'), 'totale afstand'),
      h('span', { className: 'stat' }, h('b', {}, shots.teams[t.team]?.shots ?? 0), 'schoten'),
      h('span', { className: 'stat' }, h('b', {}, shots.teams[t.team]?.on_target ?? 0), 'op doel'),
      h('span', { className: 'stat' }, h('b', {}, shots.teams[t.team]?.goals ?? 0), 'goals'))))));

  // --- schoten en goals: stand, tijdlijn en schotenkaart ---------------------------------
  const shotCanvas = h('canvas');
  const minute = m => `${Math.floor(m)}'`;
  const who = id => playerLabel(playerById.get(id)) || 'onbekend';
  async function goalClip(g) {
    const m = await api(`/matches/${match.id}/moments`, { json: { clip_id: g.clip_id, start: Math.max(0, g.t - 6), end: g.t + 5,
      label: 'Goal', players: g.player_id ? [g.player_id] : [], spotlights: g.player_id ? [g.player_id] : [] } });
    toast('Clip gemaakt'); location.hash = `#/match/${match.id}/momenten?moment=${m.id}`;
  }
  root.append(h('div', { className: 'grid-cols' },
    h('div', { className: 'panel' },
      h('h3', {}, 'Stand'),
      h('div', { className: 'scoreline' },
        h('span', {}, h('span', { className: 'dot', style: { background: TEAM_COLORS[0] } }), match.team0_name),
        h('b', {}, `${shots.score[0]} – ${shots.score[1]}`),
        h('span', {}, match.team1_name, h('span', { className: 'dot', style: { background: TEAM_COLORS[1], marginLeft: '6px' } }))),
      shots.timeline.length ? h('div', { className: 'list' }, shots.timeline.map(g => h('div', { className: 'list-item' },
        h('b', {}, minute(g.minute)), h('span', { className: 'dot', style: { background: TEAM_COLORS[g.team] } }),
        h('span', { style: { flex: 1 } }, `⚽ ${who(g.player_id)}`), h('b', {}, `${g.score[0]} – ${g.score[1]}`),
        h('button', { className: 'small', onclick: () => goalClip(g) }, '🎬 clip'))))
        : h('div', { className: 'small muted' }, 'Nog geen goals bevestigd. Bevestig ze bij "Clips & delen" (voorstellen) of voeg ze toe bij "Video + minimap".')),
    h('div', { className: 'panel' }, h('h3', {}, 'Schotenkaart'), shotCanvas,
      h('div', { className: 'small muted', style: { marginTop: '6px' } },
        '● op doel · ○ naast · groot met ring = goal. 2e helft gespiegeld, net als de heatmaps.'))));
  const shotPv = new PitchView(shotCanvas);
  function drawShots() {
    shotPv.resize(); shotPv.draw();
    const c = shotPv.ctx;
    for (const s of shots.shots) {
      if (!s.map) continue;
      const col = TEAM_COLORS[s.team] || '#999';
      const [px, py] = shotPv.toPx(...s.map);
      if (s.map_goal_x != null) {  // lijntje richting doel
        const [gx, gy] = shotPv.toPx(s.map_goal_x, stats.pitch.width / 2);
        c.beginPath(); c.moveTo(px, py); c.lineTo(gx, gy); c.strokeStyle = 'rgba(255,255,255,.25)'; c.lineWidth = 1; c.stroke();
      }
      c.beginPath(); c.arc(px, py, s.goal ? 8 : 5, 0, 2 * Math.PI);
      if (s.on_target) { c.fillStyle = col; c.fill(); }
      c.strokeStyle = s.goal ? '#fff' : col; c.lineWidth = s.goal ? 3 : 2; c.stroke();
    }
    if (!shots.shots.some(s => s.map)) {
      c.fillStyle = '#fff'; c.font = '14px sans-serif'; c.textAlign = 'center';
      c.fillText('Nog geen bevestigde schoten', shotPv.cssW / 2, shotPv.cssH / 2);
    }
  }

  function sumTeam(team, key) {
    return stats.players.filter(p => p.team === team && (!onlyNamed || p.player_id)).reduce((s, p) => s + (p[key] || 0), 0);
  }

  const tablePanel = h('div', { className: 'panel' });
  const heatCanvas = h('canvas'), shapeCanvas = h('canvas'), netCanvas = h('canvas');
  const heatTitle = h('h3', {}, 'Heatmap');
  root.append(tablePanel, h('div', { className: 'grid-cols' },
    h('div', { className: 'panel' }, heatTitle, heatCanvas, h('div', { className: 'small muted', style: { marginTop: '6px' } },
      'Klik op een speler in de tabel. Video\'s met "Richting ⇄ omdraaien" (standaard de 2e helft) zijn gespiegeld, zodat beide helften dezelfde kant op spelen.')),
    h('div', { className: 'panel' }, h('h3', {}, 'Teamvorm (gemiddelde posities)'), shapeCanvas)),
  h('div', { className: 'panel' }, h('div', { className: 'row' }, h('h3', {}, 'Passnetwerk'),
    h('select', { onchange: e => { networkTeam = Number(e.target.value); drawNetwork(); } },
      [0, 1].map(t => h('option', { value: t }, teamName(match, t))))),
  h('div', { style: { maxWidth: '900px' } }, netCanvas)));

  const COLS = [
    ['name', 'Speler'], ['team', 'Team'], ['minutes', 'Minuten'], ['distance_m', 'Afstand'],
    ['max_speed_kmh', 'Topsnelheid'], ['sprints', 'Sprints'], ['passes', 'Passes'], ['passes_failed', 'Balverlies'],
    ['possessions', 'Balcontacten'], ['shots', 'Schoten'], ['goals', 'Goals'],
  ];
  function rows() {
    let r = stats.players.filter(p => !onlyNamed || p.player_id);
    if (sortKey) r = [...r].sort((a, b) => (a[sortKey] > b[sortKey] ? 1 : a[sortKey] < b[sortKey] ? -1 : 0) * sortDir);
    return r;
  }
  function drawTable() {
    tablePanel.replaceChildren(
      h('div', { className: 'row', style: { justifyContent: 'space-between' } }, h('h2', {}, 'Spelers'),
        h('div', { className: 'row' },
          h('label', {}, h('input', { type: 'checkbox', checked: onlyNamed, onchange: e => { onlyNamed = e.target.checked; drawTable(); drawShape(); } }),
            ' alleen gekoppelde spelers'),
          h('button', { onclick: exportCsv }, 'Exporteer CSV'))),
      h('div', { style: { overflowX: 'auto' } }, h('table', {},
        h('tr', {}, COLS.map(([k, l]) => h('th', { className: 'sortable', onclick: () => { sortDir = sortKey === k ? -sortDir : -1; sortKey = k; drawTable(); } },
          l, sortKey === k ? (sortDir < 0 ? ' ↓' : ' ↑') : ''))),
        rows().map(p => h('tr', { className: `clickable ${selected === p.entity ? 'selected' : ''}`, onclick: () => { selected = p.entity; drawTable(); drawHeat(); } },
          h('td', {}, p.number ? h('b', {}, `#${p.number} `) : null, p.name),
          h('td', {}, h('span', { className: 'dot', style: { background: TEAM_COLORS[p.team] || '#999' } }), teamName(match, p.team)),
          h('td', {}, p.minutes),
          h('td', {}, `${(p.distance_m / 1000).toFixed(2)} km`),
          h('td', {}, `${p.max_speed_kmh} km/u`),
          h('td', {}, p.sprints), h('td', {}, p.passes), h('td', {}, p.passes_failed), h('td', {}, p.possessions),
          h('td', {}, p.shots), h('td', {}, p.goals))))));
  }

  const heatPv = new PitchView(heatCanvas), shapePv = new PitchView(shapeCanvas), netPv = new PitchView(netCanvas);
  function drawHeat() {
    heatPv.resize(); heatPv.draw();
    const p = stats.players.find(x => x.entity === selected) || rows()[0];
    if (!p) return;
    selected = p.entity;
    heatTitle.textContent = `Heatmap – ${p.name}`;
    heatPv.heat(p.heatmap);
    if (p.avg_pos) heatPv.dot(p.avg_pos[0], p.avg_pos[1], TEAM_COLORS[p.team] || '#999', 7, p.number || '');
  }
  function drawShape() {
    shapePv.resize(); shapePv.draw();
    for (const p of rows()) if (p.avg_pos && p.team >= 0 && p.team <= 1)
      shapePv.dot(p.avg_pos[0], p.avg_pos[1], TEAM_COLORS[p.team], p.player_id ? 9 : 4, p.number || '');
  }
  function drawNetwork() {
    netPv.resize(); netPv.draw();
    const byEnt = new Map(stats.players.map(p => [p.entity, p]));
    const edges = stats.pass_network.filter(e => byEnt.get(e.from)?.team === networkTeam);
    const max = Math.max(1, ...edges.map(e => e.count));
    const c = netPv.ctx;
    for (const e of edges) {
      const a = byEnt.get(e.from), b = byEnt.get(e.to);
      if (!a?.avg_pos || !b?.avg_pos) continue;
      c.beginPath(); c.moveTo(...netPv.toPx(...a.avg_pos)); c.lineTo(...netPv.toPx(...b.avg_pos));
      c.strokeStyle = `rgba(255,255,255,${0.25 + 0.65 * e.count / max})`; c.lineWidth = 1 + 7 * e.count / max; c.stroke();
    }
    const involved = new Set(edges.flatMap(e => [e.from, e.to]));
    for (const ent of involved) {
      const p = byEnt.get(ent);
      if (p?.avg_pos) netPv.dot(p.avg_pos[0], p.avg_pos[1], TEAM_COLORS[p.team] || '#999', 10, p.number || '');
    }
    if (!edges.length) { c.fillStyle = '#fff'; c.font = '14px sans-serif'; c.textAlign = 'center'; c.fillText('Nog geen passes gedetecteerd', netPv.cssW / 2, netPv.cssH / 2); }
  }
  function exportCsv() {
    const head = ['speler', 'rugnummer', 'team', 'minuten', 'afstand_m', 'topsnelheid_kmh', 'sprints', 'passes', 'balverlies', 'balcontacten', 'schoten', 'goals'];
    const lines = [head.join(';'), ...rows().map(p => [p.name, p.number || '', teamName(match, p.team), p.minutes, p.distance_m,
      p.max_speed_kmh, p.sprints, p.passes, p.passes_failed, p.possessions, p.shots, p.goals].join(';'))];
    // \ufeff vooraan: dan leest Excel de letters met accenten (é, ë) goed
    const a = h('a', { href: URL.createObjectURL(new Blob(['\ufeff' + lines.join('\n')], { type: 'text/csv;charset=utf-8' })), download: `${match.name}.csv` });
    a.click();
  }

  drawTable();
  requestAnimationFrame(() => { drawHeat(); drawShape(); drawNetwork(); drawShots(); });
  const onResize = () => { drawHeat(); drawShape(); drawNetwork(); drawShots(); };
  window.addEventListener('resize', onResize);
  return () => window.removeEventListener('resize', onResize);
}
