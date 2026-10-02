// Stap 3: wie is wie? Selectie invoeren en gevolgde "tracks" aan spelers koppelen.
import { api, h, toast, fmtTime, TEAM_COLORS, teamName, playerLabel } from '../util.js';

export async function render(root, ctx) {
  const { match } = ctx;
  let players = match.players;
  const clips = match.clips.filter(c => c.status === 'klaar');

  const rosterPanel = h('div', { className: 'grid-cols' });
  const trackPanel = h('div', { className: 'panel' });
  root.append(
    h('div', { className: 'hint' },
      'De app volgt iedereen op het veld als losse "tracks" en deelt ze op shirtkleur in teams in. Een speler krijgt meestal ' +
      'meerdere tracks (het volgen breekt af bij botsingen of als de camera wegdraait). Koppel hieronder tracks aan spelers. ' +
      'Snelste manier: voer rugnummers in en klik "Automatisch koppelen" (werkt als rugnummers leesbaar waren), ' +
      'en koppel de rest hier of door in "Video + minimap" op een speler te klikken.'),
    h('div', { className: 'panel' }, h('h2', {}, 'Selectie'), rosterPanel),
    trackPanel);

  function drawRoster() {
    rosterPanel.replaceChildren(...[0, 1].map(team => {
      const num = h('input', { placeholder: 'Nr', style: { width: '56px' } });
      const name = h('input', { placeholder: 'Naam' });
      const add = async () => {
        if (!name.value.trim()) return name.focus();
        await api(`/matches/${match.id}/players`, { json: { name: name.value.trim(), number: num.value.trim() || null, team } });
        players = (await api(`/matches/${match.id}`)).players;
        drawRoster(); drawTracks();
        setTimeout(() => rosterPanel.querySelectorAll('input')[team * 2]?.focus());
      };
      name.addEventListener('keydown', e => e.key === 'Enter' && add());
      return h('div', {},
        h('h3', {}, h('span', { className: 'dot', style: { background: TEAM_COLORS[team] } }), teamName(match, team)),
        h('table', {}, players.filter(p => p.team === team).map(p => h('tr', {},
          h('td', { style: { width: '60px' } }, h('input', { value: p.number || '', style: { width: '56px' },
            onchange: e => api(`/players/${p.id}`, { method: 'PATCH', json: { number: e.target.value || null } }) })),
          h('td', {}, h('input', { value: p.name, onchange: e => api(`/players/${p.id}`, { method: 'PATCH', json: { name: e.target.value } }) })),
          h('td', {}, h('button', { className: 'danger', onclick: async () => {
            await api(`/players/${p.id}`, { method: 'DELETE' });
            players = players.filter(q => q.id !== p.id); drawRoster(); drawTracks();
          } }, '×'))))),
        h('div', { className: 'row', style: { marginTop: '8px' } }, num, name, h('button', { onclick: add }, 'Toevoegen')));
    }));
  }

  let clipId = clips[0]?.id, onlyOpen = true, onlyPitch = true, limit = 48;
  async function drawTracks() {
    trackPanel.replaceChildren(h('h2', {}, 'Tracks koppelen'));
    if (!clips.length) { trackPanel.append(h('div', { className: 'empty' }, 'Nog geen geanalyseerde video\'s.')); return; }
    const tracks = await api(`/clips/${clipId}/tracks`);
    const shown = tracks.filter(t => (!onlyOpen || !t.player_id) && (!onlyPitch || t.valid));
    const clip = clips.find(c => c.id === clipId);
    trackPanel.append(h('div', { className: 'row', style: { marginBottom: '12px' } },
      'Video:', h('select', { onchange: e => { clipId = Number(e.target.value); limit = 48; drawTracks(); } },
        clips.map(c => h('option', { value: c.id, selected: c.id === clipId }, c.filename))),
      h('label', {}, h('input', { type: 'checkbox', checked: onlyOpen, onchange: e => { onlyOpen = e.target.checked; drawTracks(); } }), ' alleen nog niet gekoppeld'),
      h('label', {}, h('input', { type: 'checkbox', checked: onlyPitch, onchange: e => { onlyPitch = e.target.checked; drawTracks(); } }), ' alleen op het veld'),
      h('button', { className: 'primary', onclick: async () => {
        const r = await api(`/matches/${match.id}/auto-assign`, { json: {} });
        toast(`${r.assigned} track(s) automatisch gekoppeld via rugnummer`); drawTracks();
      } }, 'Automatisch koppelen (rugnummer)'),
      h('span', { className: 'muted small' }, `${shown.length} van ${tracks.length} tracks`)));
    trackPanel.append(h('div', { className: 'cards' }, shown.slice(0, limit).map(t => card(t, clip))));
    if (shown.length > limit) trackPanel.append(h('div', { style: { marginTop: '10px' } },
      h('button', { onclick: () => { limit += 48; drawTracks(); } }, 'Meer tonen')));
  }

  function card(t, clip) {
    const patch = data => api(`/clips/${t.clip_id}/tracks/${t.track_id}`, { method: 'PATCH', json: data });
    const teamPlayers = players.filter(p => t.team < 0 || t.team > 1 || p.team === t.team);
    return h('div', { className: 'card' },
      h('img', { src: `/api/clips/${t.clip_id}/thumb/${t.track_id}`, loading: 'lazy', title: 'Klik om in de video te bekijken',
        style: { cursor: 'pointer' }, onclick: () => location.hash = `#/match/${match.id}/video?clip=${t.clip_id}&t=${t.t_start}&track=${t.track_id}` }),
      h('div', { className: 'row small', style: { justifyContent: 'space-between' } },
        h('span', {}, h('span', { className: 'swatch', style: { background: t.color || '#ccc' } }), ` #${t.track_id}`),
        h('span', { className: 'muted' }, `${fmtTime(t.t_start)}–${fmtTime(t.t_end)}`)),
      t.jersey_guess ? h('div', { className: 'small' }, `Rugnummer? ${t.jersey_guess} (${Math.round(100 * t.jersey_conf)}%)`) : null,
      h('select', { onchange: async e => { await patch({ team: Number(e.target.value) }); t.team = Number(e.target.value); } },
        [[0, teamName(match, 0)], [1, teamName(match, 1)], [2, 'Overig (scheids/keeper)'], [-1, 'Onbekend']].map(([v, l]) =>
          h('option', { value: v, selected: t.team === v }, l))),
      h('select', { onchange: async e => {
        const pid = e.target.value ? Number(e.target.value) : null;
        await patch({ player_id: pid }); t.player_id = pid;
        if (pid && onlyOpen) e.target.closest('.card').remove();
      } },
        h('option', { value: '' }, '– speler kiezen –'),
        teamPlayers.map(p => h('option', { value: p.id, selected: t.player_id === p.id }, `${playerLabel(p)} (${teamName(match, p.team)})`))));
  }

  drawRoster();
  await drawTracks();
}
