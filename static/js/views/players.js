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
    h('div', { className: 'panel' },
      h('div', { className: 'row', style: { justifyContent: 'space-between' } }, h('h2', {}, 'Selectie'),
        h('button', { title: 'Als de app de teams andersom heeft genoemd (bijv. jouw team staat bij "Uit")', onclick: async () => {
          if (!clips.length) return toast('Nog geen geanalyseerde video\'s');
          await api(`/clips/${clips[0].id}/swap-teams`, { json: { all: true } });
          toast(`${teamName(match, 0)} en ${teamName(match, 1)} omgewisseld in alle video's`); drawTracks();
        } }, '⇄ Teams omwisselen')),
      rosterPanel),
    trackPanel);
  const squads = await api('/squads').catch(() => []);
  const otherMatches = (await api('/matches').catch(() => [])).filter(m => m.id !== match.id);

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
      const nameInput = h('input', { value: teamName(match, team), style: { fontWeight: 600, width: '200px' },
        title: 'Naam van het team', onchange: async e => {
          const v = e.target.value.trim() || (team ? 'Uit' : 'Thuis');
          await api(`/matches/${match.id}`, { method: 'PATCH', json: { [`team${team}_name`]: v } });
          match[`team${team}_name`] = v; drawTracks();
        } });
      const loader = h('select', { onchange: async e => {
        const v = e.target.value; e.target.value = '';
        if (!v) return;
        const [kind, id, fromTeam] = v.split(':');
        const r = await api(`/matches/${match.id}/load-squad`, { json: kind === 's'
          ? { team, squad_id: Number(id) } : { team, from_match: Number(id), from_team: Number(fromTeam) } });
        Object.assign(match, r.match); players = r.match.players;
        toast(`${r.added} speler(s) overgenomen`); drawRoster(); drawTracks();
      } },
        h('option', { value: '' }, '📋 Selectie overnemen…'),
        squads.length ? h('optgroup', { label: 'Vaste selecties' },
          squads.map(q => h('option', { value: `s:${q.id}` }, `${q.name} (${q.players.length} spelers)`))) : null,
        otherMatches.length ? h('optgroup', { label: 'Eerdere wedstrijden' },
          otherMatches.flatMap(m => [0, 1].map(t => h('option', { value: `m:${m.id}:${t}` }, `${m.name} – ${m[`team${t}_name`]}`)))) : null);
      return h('div', {},
        h('div', { className: 'row', style: { marginBottom: '6px' } },
          h('span', { className: 'dot', style: { background: TEAM_COLORS[team] } }), nameInput),
        h('div', { className: 'row small', style: { marginBottom: '8px' } }, loader,
          h('button', { title: 'Bewaar deze spelers onder de teamnaam, om ze bij een volgende wedstrijd over te nemen', onclick: async () => {
            const sq = await api(`/matches/${match.id}/save-squad`, { json: { team } });
            if (!squads.some(q => q.id === sq.id)) squads.push(sq);
            toast(`Selectie "${sq.name}" bewaard (${sq.players.length} spelers)`);
          } }, '💾 Bewaar als vaste selectie')),
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
      h('span', { className: 'muted small' }, `${shown.length} van ${tracks.length} tracks`),
      h('span', { style: { flex: 1 } }),
      h('button', { className: 'small', title: 'Alleen in deze video de teams omwisselen (als de app ze hier andersom heeft dan in de andere video\'s)', onclick: async () => {
        await api(`/clips/${clipId}/swap-teams`, { json: {} }); toast('Teams omgewisseld in deze video'); drawTracks();
      } }, '⇄ Alleen deze video'),
      h('button', { className: 'small', title: 'Teams opnieuw automatisch bepalen op shirtkleur (wat je zelf hebt aangepast blijft staan)', onclick: async () => {
        await api(`/clips/${clipId}/reassign-teams`, { method: 'POST' }); toast('Teams opnieuw ingedeeld'); drawTracks();
      } }, '↻ Opnieuw indelen')));
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
