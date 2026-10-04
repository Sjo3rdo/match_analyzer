// Stap 3: wie is wie? Selectie invoeren en gevolgde "tracks" aan spelers koppelen.
import { api, h, toast, fmtTime, TEAM_COLORS, teamName, playerLabel, teamOptions, markSpectator } from '../util.js';

export async function render(root, ctx) {
  const { match } = ctx;
  let players = match.players;
  const clips = match.clips.filter(c => c.status === 'klaar');

  const rosterPanel = h('div', { className: 'grid-cols' });
  const trackPanel = h('div', { className: 'panel' });
  const assistBox = h('div');
  let assistFor = null, hidden = new Set();
  root.append(
    h('div', { className: 'hint' },
      'De app volgt iedereen op het veld als losse "tracks" en deelt ze op shirtkleur in teams in. Een speler krijgt meestal ' +
      'meerdere tracks (het volgen breekt af bij botsingen of als de camera wegdraait). Koppel hieronder tracks aan spelers. ' +
      'Snelste manier: voer rugnummers in en klik "Automatisch koppelen" (werkt als rugnummers leesbaar waren), ' +
      'en koppel de rest hier of door in "Video + minimap" op een speler te klikken. Staat er een toeschouwer tussen, klik dan 🚫: ' +
      'die telt dan nergens meer mee, en de app biedt aan vergelijkbare personen op dezelfde plek mee weg te halen. ' +
      'Zet je een paar spelers zelf in het goede team, dan gebruikt "↻ Opnieuw indelen" die als voorbeeld voor de rest.'),
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
  let profiles = await api(`/matches/${match.id}/profiles`).catch(() => ({}));
  const recogBox = h('div');
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
          h('td', {}, h('button', { title: 'Koppel-assistent: zoek tracks van deze speler', onclick: () => openAssist(p.id) }, '🔍'), ' ',
            h('button', { title: profiles[p.id] ? 'De app kent deze speler al. Opnieuw leren (met alles wat nu gekoppeld is)'
              : 'Leer de app deze speler herkennen (uit de stukken die aan hem gekoppeld zijn)',
              className: profiles[p.id] ? 'learned' : '', onclick: () => learnPlayer(p) }, '🧠'), ' ',
            h('button', { className: 'danger', onclick: async () => {
            await api(`/players/${p.id}`, { method: 'DELETE' });
            players = players.filter(q => q.id !== p.id); drawRoster(); drawTracks();
          } }, '×'))))),
        h('div', { className: 'row', style: { marginTop: '8px' } }, num, name, h('button', { onclick: add }, 'Toevoegen')));
    }));
  }

  let clipId = clips.find(c => c.id === Number(ctx.params.get('clip')))?.id || clips[0]?.id;
  let onlyOpen = true, onlyPitch = true, limit = 48;
  async function drawTracks() {
    trackPanel.replaceChildren(h('h2', {}, 'Tracks koppelen'), recogBox, assistBox);
    if (!clips.length) { trackPanel.append(h('div', { className: 'empty' }, 'Nog geen geanalyseerde video\'s.')); return; }
    const tracks = await api(`/clips/${clipId}/tracks`);
    const shown = tracks.filter(t => (!onlyOpen || !t.player_id) && (!onlyPitch || t.valid));
    const clip = clips.find(c => c.id === clipId);
    trackPanel.append(h('div', { className: 'row', style: { marginBottom: '12px' } },
      'Video:', h('select', { onchange: e => { clipId = Number(e.target.value); limit = 48; drawTracks(); if (assistFor) openAssist(assistFor); } },
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
      h('button', { className: 'small', title: 'Teams opnieuw automatisch bepalen op shirtkleur. Wat je zelf hebt aangepast blijft staan en dient als voorbeeld voor de rest.', onclick: async () => {
        await api(`/clips/${clipId}/reassign-teams`, { method: 'POST' }); toast('Teams opnieuw ingedeeld'); drawTracks();
      } }, '↻ Opnieuw indelen')));
    trackPanel.append(h('div', { className: 'cards' }, shown.slice(0, limit).map(t => card(t, clip))));
    drawRecognize();
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
        h('span', { className: 'muted' }, `${fmtTime(t.t_start)}–${fmtTime(t.t_end)}`),
        t.team === 3 ? null : h('button', { className: 'small', title: 'Dit is een toeschouwer: weghalen (telt dan nergens meer mee)',
          style: { padding: '1px 6px' }, onclick: e => spectator(t, e.target.closest('.card')) }, '🚫')),
      t.jersey_guess ? h('div', { className: 'small' }, `Rugnummer? ${t.jersey_guess} (${Math.round(100 * t.jersey_conf)}%)`) : null,
      h('select', { onchange: async e => {
        const v = Number(e.target.value);
        if (v === 3) return spectator(t, e.target.closest('.card'));
        await patch({ team: v }); t.team = v;
      } }, teamOptions(match, t.team)),
      h('select', { onchange: async e => {
        const pid = e.target.value ? Number(e.target.value) : null;
        await patch({ player_id: pid }); t.player_id = pid;
        if (pid && onlyOpen) e.target.closest('.card').remove();
        if (pid) openAssist(pid);
      } },
        h('option', { value: '' }, '– speler kiezen –'),
        teamPlayers.map(p => h('option', { value: p.id, selected: t.player_id === p.id }, `${playerLabel(p)} (${teamName(match, p.team)})`))));
  }

  function spectator(t, cardEl) {
    t.team = 3;
    if (onlyPitch) cardEl?.remove();
    markSpectator(t.clip_id, t.track_id, assistBox, () => drawTracks());
  }

  // --- spelers herkennen met hun profiel -----------------------------------------------------
  async function learnPlayer(p) {
    const linked = (await api(`/matches/${match.id}`)).n_assigned;
    if (!linked) return toast(`Koppel eerst een paar stukken aan ${playerLabel(p)}; daarvan leert de app hoe hij eruitziet`, true);
    const ov = await api('/training').catch(() => null);
    const est = ov?.matches.find(m => m.id === match.id)?.minutes || 1;
    if (!confirm(`De app bekijkt alle stukken in de video's van deze wedstrijd om ${playerLabel(p)} te leren herkennen. ` +
      `Dat duurt ± ${est} min (de eerste keer per wedstrijd; daarna sneller). Nu starten?`)) return;
    await api('/training/start', { json: { kind: 'players', match_id: match.id, player_id: p.id } });
    toast('Bezig met leren. Voortgang rechtsboven; als het klaar is, verschijnen hier de voorstellen.');
    waitForTraining();
  }
  let waitTimer = null;
  async function waitForTraining() {
    clearTimeout(waitTimer);
    const st = await api('/training/status').catch(() => null);
    if (st?.busy) { waitTimer = setTimeout(waitForTraining, 3000); return; }
    if (st?.last) toast(st.last.message || 'Klaar', !st.last.ok);
    profiles = await api(`/matches/${match.id}/profiles`).catch(() => ({}));
    drawRoster(); drawRecognize();
  }
  async function drawRecognize() {
    const list = await api(`/matches/${match.id}/recognize`).catch(() => []);
    const here = list.filter(r => r.clip_id === clipId);
    if (!list.length) { recogBox.replaceChildren(); return; }
    const byId = new Map(players.map(p => [p.id, p]));
    recogBox.replaceChildren(h('div', { className: 'hint recog' },
      h('div', { className: 'row', style: { justifyContent: 'space-between' } },
        h('b', {}, `🧠 Herkend: ${list.length} stuk(ken) lijken op een speler`, here.length < list.length ? ` (${here.length} in deze video)` : ''),
        here.length ? h('button', { className: 'primary small', onclick: async () => {
          if (!confirm(`${here.length} voorstel(len) in deze video in één keer koppelen? Controleer ze eerst even hieronder.`)) return;
          for (const r of here) await api(`/clips/${r.clip_id}/tracks/${r.track_id}`, { method: 'PATCH', json: { player_id: r.player_id } });
          toast(`${here.length} gekoppeld`); drawTracks();
        } }, `✓ Alle ${here.length} koppelen`) : null),
      here.length ? h('div', { className: 'cards', style: { marginTop: '8px' } }, here.slice(0, 24).map(r => h('div', { className: 'card' },
        h('img', { src: `/api/clips/${r.clip_id}/thumb/${r.track_id}`, loading: 'lazy' }),
        h('div', { className: 'small' }, h('b', {}, playerLabel(byId.get(r.player_id))), ` · ${Math.round(100 * r.score)}% gelijk`),
        h('div', { className: 'row' },
          h('button', { className: 'primary small', onclick: async () => {
            await api(`/clips/${r.clip_id}/tracks/${r.track_id}`, { method: 'PATCH', json: { player_id: r.player_id } });
            toast('Gekoppeld'); drawTracks();
          } }, '✓'),
          h('button', { className: 'small', title: 'Niet deze speler', onclick: e => e.target.closest('.card').remove() }, '✗')))))
        : h('div', { className: 'small muted' }, 'Kies een andere video hierboven om de voorstellen te zien.')));
  }

  // --- koppel-assistent ---------------------------------------------------------------------
  // Na het koppelen van één track stelt de app tracks voor die er logisch op aansluiten:
  // niet tegelijk in beeld, beginnend waar het vorige stuk ophield, met hetzelfde shirt.
  async function openAssist(pid) {
    if (assistFor !== pid) hidden = new Set();
    assistFor = pid;
    const p = players.find(x => x.id === pid);
    const list = (await api(`/players/${pid}/suggestions?clip_id=${clipId}&limit=8`)).filter(r => !hidden.has(r.track_id));
    const label = sc => sc >= 0.6 ? ['waarschijnlijk', 'ok'] : sc >= 0.35 ? ['mogelijk', 'busy'] : ['twijfel', ''];
    assistBox.replaceChildren(h('div', { className: 'hint', style: { marginBottom: '12px' } },
      h('div', { className: 'row', style: { justifyContent: 'space-between' } },
        h('b', {}, `🔍 Koppel-assistent: waarschijnlijk ook ${playerLabel(p)}`),
        h('button', { className: 'small', onclick: () => { assistFor = null; assistBox.replaceChildren(); } }, 'Sluiten')),
      list.length ? h('div', { className: 'small muted', style: { margin: '4px 0 8px' } },
        'Klopt het? Klik ✓ om te koppelen; daarna zoekt de app verder vanaf dat stuk. Klik op het plaatje om het in de video te zien.') : null,
      list.length ? h('div', { className: 'cards' }, list.map(r => {
        const [txt, cls] = label(r.score);
        return h('div', { className: 'card' },
          h('img', { src: `/api/clips/${r.clip_id}/thumb/${r.track_id}`, loading: 'lazy', style: { cursor: 'pointer' },
            onclick: () => location.hash = `#/match/${match.id}/video?clip=${r.clip_id}&t=${r.t_start}&track=${r.track_id}` }),
          h('div', { className: 'row small', style: { justifyContent: 'space-between' } },
            h('span', {}, `#${r.track_id}`), h('span', { className: 'muted' }, `${fmtTime(r.t_start)}–${fmtTime(r.t_end)}`)),
          h('div', { className: 'small' }, h('span', { className: `badge ${cls}` }, `${txt} ${Math.round(100 * r.score)}%`)),
          h('div', { className: 'small muted' }, r.reason),
          h('div', { className: 'row' },
            h('button', { className: 'primary', onclick: async () => {
              await api(`/clips/${r.clip_id}/tracks/${r.track_id}`, { method: 'PATCH', json: { player_id: pid } });
              toast(`Gekoppeld aan ${playerLabel(p)}`); await drawTracks(); openAssist(pid);
            } }, '✓ Koppel'),
            h('button', { title: 'Niet deze speler', onclick: () => { hidden.add(r.track_id); openAssist(pid); } }, '✗')));
      })) : h('div', { className: 'small muted' }, 'Geen goede kandidaten (meer) in deze video. Koppel een volgend stuk met de hand, dan zoekt de app verder.')));
  }

  drawRoster();
  await drawTracks();
  api('/training/status').then(st => { if (st.busy && st.kind === 'players') waitForTraining(); }).catch(() => {});
  const assistParam = Number(ctx.params.get('assist'));
  if (assistParam && players.some(p => p.id === assistParam)) openAssist(assistParam);
}
