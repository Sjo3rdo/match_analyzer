import { api, h } from '../util.js';

export async function render(root) {
  const list = await api('/matches');
  const squads = await api('/squads').catch(() => []);
  const name = h('input', { placeholder: 'Bijv. JO17-1 – VV Voorbeeld', style: { minWidth: '260px', flex: 1 } });
  const date = h('input', { type: 'date', value: new Date().toISOString().slice(0, 10) });
  // Teamnaam kiezen uit de vaste selecties: dan worden de spelers meteen overgenomen
  const t0 = h('input', { placeholder: 'Thuisteam', value: 'Thuis', list: 'squad-names' });
  const t1 = h('input', { placeholder: 'Uitteam', value: 'Uit', list: 'squad-names' });
  const create = async () => {
    if (!name.value.trim()) return name.focus();
    const m = await api('/matches', { json: { name: name.value, date: date.value, team0_name: t0.value, team1_name: t1.value } });
    location.hash = `#/match/${m.id}/clips`;
  };
  name.addEventListener('keydown', e => e.key === 'Enter' && create());
  root.append(
    h('div', { className: 'hero' },
      h('div', {}, h('div', { className: 'eyebrow' }, 'Overzicht'), h('h1', {}, 'Wedstrijden'),
        h('p', {}, 'Upload je telefoonvideo\'s, leg het veld erop en zie per speler wat er gebeurde. Alles blijft op je eigen laptop.'))),
    h('div', { className: 'panel' },
      h('h3', {}, 'Nieuwe wedstrijd'),
      h('datalist', { id: 'squad-names' }, squads.map(q => h('option', { value: q.name }))),
      h('div', { className: 'row' }, name, date, t0, h('span', { className: 'muted' }, '–'), t1,
        h('button', { className: 'primary', onclick: create }, '+ Aanmaken')),
      squads.length ? h('div', { className: 'small muted', style: { marginTop: '8px' } },
        `Kies bij een team een opgeslagen naam (${squads.map(q => q.name).join(', ')}): dan staan de spelers er meteen in.`) : null),
    list.length ? h('div', { className: 'match-grid' }, list.map(m => h('div', { className: 'match-card', onclick: () => location.hash = `#/match/${m.id}/clips` },
      h('div', { className: 'eyebrow' }, m.date || 'Zonder datum'),
      h('div', { className: 'teams' }, m.name),
      h('div', { className: 'meta' }, h('span', {}, m.score ? `${m.team0_name} ${m.score[0]} – ${m.score[1]} ${m.team1_name}` : `${m.team0_name} – ${m.team1_name}`), h('span', {}, `🎬 ${m.n_clips} video${m.n_clips === 1 ? '' : "'s"}`)),
      h('button', { className: 'danger small del', title: 'Verwijderen', onclick: async e => {
        e.stopPropagation();
        if (!confirm(`"${m.name}" en alle bijbehorende video's verwijderen?`)) return;
        await api(`/matches/${m.id}`, { method: 'DELETE' });
        location.reload();
      } }, 'Verwijderen'))))
      : h('div', { className: 'panel empty' }, 'Nog geen wedstrijden. Maak er hierboven een aan.'));
}
