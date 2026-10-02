// Stap 2: het veld "vastpinnen" op het beeld. Klik een punt in het beeld en daarna het
// bijbehorende punt op de veldtekening (of andersom). Minstens 4, liefst 6+ punten.
import { api, h, fmtTime, toast } from '../util.js';
import { PitchView, pitchPolylines } from '../pitch.js';
import { fitHomography, apply } from '../homography.js';

export async function render(root, ctx) {
  const { match } = ctx;
  const clips = match.clips;
  if (!clips.length) {
    root.append(h('div', { className: 'panel empty' }, 'Upload eerst een video.'));
    return;
  }
  const pitchInfo = await api('/pitch');
  let clip = clips.find(c => c.id === Number(ctx.params.get('clip'))) || clips[0];
  let t = 0, img = null, pairs = [], pendingImg = null, pendingPitch = null, editingId = null;
  let predicted = [], showPred = true, drag = null, keyframes = [];

  const frameCanvas = h('canvas');
  const pitchCanvas = h('canvas');
  const slider = h('input', { type: 'range', min: 0, step: 0.1, style: { flex: 1 } });
  const timeLabel = h('span', { className: 'muted' });
  const errLabel = h('span');
  const pairList = h('div', { className: 'list' });
  const kfList = h('div');
  const clipSel = h('select', { onchange: e => { clip = clips.find(c => c.id === Number(e.target.value)); ctx.setParam('clip', clip.id); loadClip(); } },
    clips.map(c => h('option', { value: c.id, selected: c.id === clip.id }, c.filename)));
  const predToggle = h('label', { className: 'small' }, h('input', { type: 'checkbox', checked: true,
    onchange: e => { showPred = e.target.checked; drawFrame(); } }), ' toon voorspelling (geel)');

  root.append(
    h('div', { className: 'hint' },
      'Zo werkt het: 1) Kies met de schuif een moment. 2) Klik in het beeld op een herkenbaar punt (hoek strafschopgebied, middenstip, ' +
      'snijpunt middenlijn/zijlijn...). 3) Klik op hetzelfde punt in de veldtekening rechts. Herhaal tot je minstens 4 punten hebt, ' +
      'verspreid over het beeld. De witte lijnen laten zien of het klopt. Omdat je camera beweegt: voeg elke 30–60 seconden en na ' +
      'flinke zwenks of zooms een nieuw sleutelframe toe. Na de analyse voorspelt de app de punten (geel) en hoef je ze alleen bij te schuiven.'),
    h('div', { className: 'row', style: { marginBottom: '12px' } }, 'Video:', clipSel),
    h('div', { className: 'grid2' },
      h('div', { className: 'panel' },
        h('div', { className: 'row', style: { marginBottom: '8px' } }, slider, timeLabel),
        h('div', { className: 'frame-wrap' }, frameCanvas),
        h('div', { className: 'row small muted', style: { marginTop: '6px' } },
          'Sleep een punt om het te verschuiven. Rechtsklik op een punt verwijdert het.', predToggle)),
      h('div', {},
        h('div', { className: 'panel' }, h('h3', {}, 'Veld (klik op een punt)'), pitchCanvas),
        h('div', { className: 'panel' },
          h('div', { className: 'row', style: { justifyContent: 'space-between' } }, h('h3', {}, 'Punten in dit sleutelframe'), errLabel),
          pairList,
          h('div', { className: 'row', style: { marginTop: '10px' } },
            h('button', { className: 'primary', onclick: save }, 'Sleutelframe opslaan'),
            h('button', { onclick: usePredicted }, 'Voorspelde punten overnemen'),
            h('button', { onclick: () => { pairs = []; editingId = null; redraw(); } }, 'Leegmaken'))),
        h('div', { className: 'panel' }, h('h3', {}, 'Sleutelframes van deze video'), kfList))));

  const pv = new PitchView(pitchCanvas);
  requestAnimationFrame(() => { pv.resize(); drawPitch(); });

  function imgCoords(e) {
    const r = frameCanvas.getBoundingClientRect();
    return [(e.clientX - r.left) * frameCanvas.width / r.width, (e.clientY - r.top) * frameCanvas.height / r.height];
  }
  const pxPerCss = () => frameCanvas.width / frameCanvas.getBoundingClientRect().width;

  frameCanvas.addEventListener('mousedown', e => {
    if (!img || e.button !== 0) return;
    const p = imgCoords(e), tol = 12 * pxPerCss();
    const hit = pairs.findIndex(q => Math.hypot(q.img[0] - p[0], q.img[1] - p[1]) < tol);
    if (hit >= 0) { drag = hit; return; }
    if (pendingPitch) {
      addPair(pendingPitch, p); pendingPitch = null;
    } else {
      pendingImg = p;
    }
    redraw();
  });
  frameCanvas.addEventListener('mousemove', e => {
    if (drag === null) return;
    pairs[drag].img = imgCoords(e);
    drawFrame();
  });
  window.addEventListener('mouseup', onUp);
  function onUp() { if (drag !== null) { drag = null; redraw(); } }
  frameCanvas.addEventListener('contextmenu', e => {
    e.preventDefault();
    const p = imgCoords(e), tol = 12 * pxPerCss();
    const hit = pairs.findIndex(q => Math.hypot(q.img[0] - p[0], q.img[1] - p[1]) < tol);
    if (hit >= 0) { pairs.splice(hit, 1); redraw(); }
  });
  pitchCanvas.addEventListener('click', e => {
    const r = pitchCanvas.getBoundingClientRect();
    const [mx, my] = pv.toM(e.clientX - r.left, e.clientY - r.top);
    let best = null, bd = 4;
    for (const lm of pitchInfo.landmarks) {
      const d = Math.hypot(lm.x - mx, lm.y - my);
      if (d < bd) { bd = d; best = lm; }
    }
    if (!best) return toast('Klik dichter bij een wit punt op het veld');
    if (pendingImg) { addPair(best, pendingImg); pendingImg = null; }
    else pendingPitch = best;
    redraw();
  });

  function addPair(lm, p) {
    pairs = pairs.filter(q => q.name !== lm.name);
    pairs.push({ name: lm.name, img: p, pitch: [lm.x, lm.y] });
  }

  function currentH() {
    if (pairs.length < 4) return null;
    return fitHomography(pairs.map(p => p.pitch), pairs.map(p => p.img));
  }

  function drawFrame() {
    const c = frameCanvas.getContext('2d');
    if (!img) return;
    c.drawImage(img, 0, 0);
    const lw = 2.5 * pxPerCss();
    const drawLines = (H, color, dash) => {
      c.save(); c.strokeStyle = color; c.lineWidth = lw; c.setLineDash(dash);
      for (const line of pitchPolylines()) {
        c.beginPath(); let started = false;
        for (const pt of line) {
          const [x, y, w] = apply(H, pt);
          if (w <= 0) { started = false; continue; }
          if (!started) { c.moveTo(x, y); started = true; } else c.lineTo(x, y);
        }
        c.stroke();
      }
      c.restore();
    };
    if (showPred && predicted.length >= 4) {
      const Hp = fitHomography(predicted.map(p => p.pitch), predicted.map(p => p.img));
      if (Hp) drawLines(Hp, 'rgba(255,214,0,.85)', [8 * pxPerCss(), 6 * pxPerCss()]);
    }
    const H = currentH();
    if (H) drawLines(H, 'rgba(255,255,255,.9)', []);
    const r = 7 * pxPerCss();
    pairs.forEach((p, i) => {
      c.beginPath(); c.arc(p.img[0], p.img[1], r, 0, 2 * Math.PI);
      c.fillStyle = '#ff2d55'; c.fill(); c.lineWidth = lw * .6; c.strokeStyle = '#fff'; c.stroke();
      c.fillStyle = '#fff'; c.font = `bold ${r * 1.5}px sans-serif`; c.fillText(String(i + 1), p.img[0] + r * 1.3, p.img[1] - r);
    });
    if (pendingImg) {
      c.beginPath(); c.arc(pendingImg[0], pendingImg[1], r, 0, 2 * Math.PI);
      c.strokeStyle = '#ffd600'; c.lineWidth = lw; c.stroke();
    }
  }

  function drawPitch() {
    pv.draw();
    const used = new Map(pairs.map((p, i) => [p.name, i + 1]));
    for (const lm of pitchInfo.landmarks) {
      const n = used.get(lm.name);
      if (n) pv.dot(lm.x, lm.y, '#ff2d55', 7, String(n));
      else pv.dot(lm.x, lm.y, pendingPitch?.name === lm.name ? '#ffd600' : 'rgba(255,255,255,.9)', 3.5);
    }
  }

  function drawPairs() {
    pairList.replaceChildren(...pairs.map((p, i) => h('div', { className: 'list-item' },
      h('b', {}, i + 1), h('span', { style: { flex: 1 } }, p.name),
      h('button', { onclick: () => { pairs.splice(i, 1); redraw(); } }, '×'))));
    if (!pairs.length) pairList.append(h('div', { className: 'muted small' },
      pendingImg ? 'Klik nu het bijbehorende punt op het veld.' : pendingPitch ? `Klik nu "${pendingPitch.name}" in het beeld.` : 'Nog geen punten.'));
    if (pairs.length >= 4) {
      const H = fitHomography(pairs.map(p => p.img), pairs.map(p => p.pitch));
      const err = H ? pairs.reduce((s, p) => { const q = apply(H, p.img); return s + Math.hypot(q[0] - p.pitch[0], q[1] - p.pitch[1]); }, 0) / pairs.length : NaN;
      errLabel.replaceChildren(h('span', { className: `badge ${err < 1 ? 'ok' : 'err'}` }, `afwijking ${err.toFixed(2)} m`));
    } else errLabel.replaceChildren(h('span', { className: 'muted small' }, `${pairs.length}/4 punten`));
  }

  function redraw() { drawFrame(); drawPitch(); drawPairs(); }

  async function loadFrame() {
    timeLabel.textContent = `${fmtTime(t)} / ${fmtTime(clip.duration)}`;
    const im = new Image();
    im.src = `/api/clips/${clip.id}/frame?t=${t.toFixed(2)}`;
    await im.decode().catch(() => toast('Frame kon niet geladen worden', true));
    img = im;
    frameCanvas.width = im.naturalWidth; frameCanvas.height = im.naturalHeight;
    predicted = clip.status === 'klaar' ? await api(`/clips/${clip.id}/predict?t=${t}`) : [];
    redraw();
  }

  async function loadKeyframes() {
    keyframes = await api(`/clips/${clip.id}/keyframes`);
    kfList.replaceChildren(...(keyframes.length ? keyframes.map(kf => h('div', { className: 'list-item', onclick: () => {
      t = kf.t; slider.value = t; pairs = kf.points.map(p => ({ ...p })); editingId = kf.id; loadFrame();
    } },
      h('b', {}, fmtTime(kf.t)), h('span', { style: { flex: 1 } }, `${kf.points.length} punten`),
      kf.error_m != null ? h('span', { className: `badge ${kf.error_m < 1 ? 'ok' : 'err'}` }, `${kf.error_m} m`) : null,
      editingId === kf.id ? h('span', { className: 'badge busy' }, 'bewerken') : null,
      h('button', { className: 'danger', onclick: async e => {
        e.stopPropagation();
        await api(`/keyframes/${kf.id}`, { method: 'DELETE' });
        if (editingId === kf.id) editingId = null;
        loadKeyframes();
      } }, '×'))) : [h('div', { className: 'muted small' }, 'Nog geen sleutelframes. Zonder kalibratie kan de app geen meters berekenen.')]));
  }

  async function save() {
    if (pairs.length < 4) return toast('Minstens 4 punten nodig');
    const res = await api(`/clips/${clip.id}/keyframes`, { json: { id: editingId, t, points: pairs } });
    editingId = res.id;
    toast(`Sleutelframe opgeslagen (afwijking ${res.error_m} m)`);
    await loadKeyframes();
    predicted = clip.status === 'klaar' ? await api(`/clips/${clip.id}/predict?t=${t}`) : [];
    drawFrame();
  }

  function usePredicted() {
    if (!predicted.length) return toast(clip.status === 'klaar' ? 'Nog geen voorspelling: sla eerst een sleutelframe op' : 'Voorspelling kan pas na de analyse');
    pairs = predicted.map(p => ({ ...p, img: [...p.img] }));
    editingId = null;
    redraw();
    toast('Punten overgenomen. Sleep ze naar de juiste plek en sla op.');
  }

  let debounce;
  slider.addEventListener('input', () => {
    t = Number(slider.value);
    timeLabel.textContent = `${fmtTime(t)} / ${fmtTime(clip.duration)}`;
    clearTimeout(debounce);
    debounce = setTimeout(() => {
      // Bij een nieuw moment begin je een nieuw sleutelframe
      if (editingId) { editingId = null; pairs = []; }
      loadFrame(); loadKeyframes();
    }, 250);
  });

  async function loadClip() {
    slider.max = clip.duration || 0; t = 0; slider.value = 0; pairs = []; editingId = null;
    await loadKeyframes();
    if (keyframes.length) { const kf = keyframes[0]; t = kf.t; slider.value = t; pairs = kf.points.map(p => ({ ...p })); editingId = kf.id; }
    await loadFrame();
  }
  await loadClip();
  const onResize = () => { pv.resize(); drawPitch(); };
  window.addEventListener('resize', onResize);
  return () => { window.removeEventListener('mouseup', onUp); window.removeEventListener('resize', onResize); };
}
