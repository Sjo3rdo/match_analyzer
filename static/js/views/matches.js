import { api, h } from '../util.js';

export async function render(root) {
  const list = await api('/matches');
  const squads = await api('/squads').catch(() => []);
  const name = h('input', { placeholder: 'Bijv. JO17-1 – VV Voorbeeld', size: 30 });
  const date = h('input', { type: 'date', value: new Date().toISOString().slice(0, 10) });
  // Teamnaam kiezen uit de vaste selecties: dan worden de spelers meteen overgenomen
  const t0 = h('input', { placeholder: 'Thuisteam', value: 'Thuis', list: 'squad-names' });
  const t1 = h('input', { placeholder: 'Uitteam', value: 'Uit', list: 'squad-names' });
  const create = async () => {
    if (!name.value.trim()) return name.focus();
    const m = await api('/matches', { json: { name: name.value, date: date.value, team0_name: t0.value, team1_name: t1.value } });
    location.hash = `#/match/${m.id}/clips`;
  };
  root.append(
    h('h1', {}, 'Wedstrijden'),
    h('div', { className: 'panel' },
      h('h2', {}, 'Nieuwe wedstrijd'),
      h('datalist', { id: 'squad-names' }, squads.map(q => h('option', { value: q.name }))),
      h('div', { className: 'row' }, name, date, t0, t1, h('button', { className: 'primary', onclick: create }, 'Aanmaken')),
      squads.length ? h('div', { className: 'small muted', style: { marginTop: '6px' } },
        `Kies bij een team een opgeslagen naam (${squads.map(q => q.name).join(', ')}): dan staan de spelers er meteen in.`) : null),
    h('div', { className: 'panel' },
      list.length ? h('table', {},
        h('tr', {}, h('th', {}, 'Wedstrijd'), h('th', {}, 'Datum'), h('th', {}, 'Teams'), h('th', {}, "Video's"), h('th')),
        list.map(m => h('tr', { className: 'clickable', onclick: () => location.hash = `#/match/${m.id}/clips` },
          h('td', {}, h('b', {}, m.name)), h('td', {}, m.date || ''), h('td', {}, `${m.team0_name} – ${m.team1_name}`),
          h('td', {}, m.n_clips),
          h('td', {}, h('button', { className: 'danger', onclick: async e => {
            e.stopPropagation();
            if (!confirm(`"${m.name}" en alle bijbehorende video's verwijderen?`)) return;
            await api(`/matches/${m.id}`, { method: 'DELETE' });
            location.reload();
          } }, 'Verwijderen')))))
        : h('div', { className: 'empty' }, 'Nog geen wedstrijden. Maak er hierboven een aan.')));
}
