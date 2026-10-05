// Hoe hoog hield je de telefoon? Vrij in te vullen, met snelkeuzes. 1,6 m is ooghoogte van een
// staande volwassene: daar houd je een telefoon meestal als je langs de lijn filmt.
import { h } from '../util.js';

export const HEIGHTS = [[1.6, 'Staand (ooghoogte)'], [2.1, 'Boven je hoofd'], [2.5, 'Bankje / heuvel'], [4, 'Tribune'], [6, 'Hoge tribune']];

export function heightField(value, onChange, estimate = null) {
  const v0 = Number(value) || 1.6;
  const input = h('input', { type: 'number', min: 0.5, max: 30, step: 0.1, value: v0.toFixed(1), style: { width: '72px' },
    onchange: () => set(Number(input.value)) });
  const presets = h('div', { className: 'row', style: { gap: '4px', flexWrap: 'wrap' } });
  function set(v) {
    if (!(v >= 0.5 && v <= 30)) { input.value = v0.toFixed(1); return; }
    v = Math.round(v * 10) / 10;
    input.value = v.toFixed(1); drawPresets(v); onChange(v);
  }
  function drawPresets(v) {
    presets.replaceChildren(...HEIGHTS.map(([hv, l]) => h('button', { className: `small${Math.abs(hv - v) < 0.05 ? ' primary' : ''}`,
      onclick: () => set(hv) }, `${l} ${String(hv).replace('.', ',')} m`)));
  }
  drawPresets(v0);
  return h('div', {},
    h('div', { className: 'row', style: { gap: '6px' } }, input, h('span', {}, 'meter boven het gras'),
      estimate != null && Math.abs(estimate - v0) >= 0.3
        ? h('button', { className: 'small', title: 'Hoogte die de app uit je klikken berekent', onclick: () => set(estimate) },
          `Neem ${String(estimate).replace('.', ',')} m over (berekend)`) : null),
    presets,
    h('div', { className: 'small muted', style: { marginTop: '4px' } },
      'Bedoeld is de hoogte van de telefoon, niet van jezelf. Staand langs de lijn is dat ooghoogte (± 1,6 m). Op een tribune: tel de hoogte van je rij erbij op.'));
}
