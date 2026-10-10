// Stap 2: het veld "vastpinnen" op het beeld. Klik een punt in het beeld en daarna het
// bijbehorende punt (of de lijn) op de veldtekening, of andersom. Weet de app waar je stond
// (aangeklikt of via GPS), dan rekent hij met een cameramodel en is 1 punt + 1 lijn genoeg.
import { api, h, fmtTime, toast } from '../util.js';
import { PitchView, pitchPolylines, setPitchSize } from '../pitch.js';
import { apply, inv, fitCalibration, calibrationInfo, calibrationError } from '../homography.js';
import { stationsPanel } from './stations.js';
import { heightField } from './camheight.js';

export async function render(root, ctx) {
  const { match } = ctx;
  const clips = match.clips;
  if (!clips.length) {
    root.append(h('div', { className: 'panel empty' }, 'Upload eerst een video.'));
    return;
  }
  const pitchInfo = await api(`/pitch?match_id=${match.id}`);
  setPitchSize(pitchInfo.length, pitchInfo.width);  // tekening en punten altijd op dezelfde veldmaten
  let clip = clips.find(c => c.id === Number(ctx.params.get('clip'))) || clips[0];
  let t = 0, img = null, pairs = [], pendingImg = null, pendingPitch = null, editingId = null;
  let predicted = [], showPred = true, drag = null, keyframes = [];
  let fit = { H: null, err: null, cam: null, msg: '' }, placingCam = false, osmSize = null;
  let placingDir = false, dirDrag = null, skipClick = false;
  let touched = null, trigger = null;  // laatst gezette/versleepte klik; de klik waarna het rood werd  // kijkrichting aangeven (klikken of slepen)
  let frameT = null;            // tijd van het beeld dat nu getoond wordt
  let proposal = null;          // automatisch voorstel dat nog bevestigd moet worden
  let proposing = false;        // de app zoekt nu het veld
  let gpsError = null;          // waarom zoeken via GPS niet lukte (blijft in beeld)
  let goalSide = null;          // geklikt bij dit doel: kies voet/bovenkant/lat
  let pendingZoom = 1;          // zoomfactor bij het aanklikken van het beeldpunt dat nog wacht
  let view = { s: 1, ox: 0, oy: 0 }, pan = null;  // inzoomen op het beeld
  const hasCam = () => clip.cam_x != null && clip.cam_y != null;

  const frameCanvas = h('canvas');
  const pitchCanvas = h('canvas');
  const slider = h('input', { type: 'range', min: 0, step: 0.1, style: { flex: 1 } });
  const timeLabel = h('span', { className: 'muted' });
  const errLabel = h('span');
  const pairList = h('div', { className: 'list' });
  const fitWarn = h('div');
  const camPanel = h('div', { className: 'panel' });
  const kfList = h('div');
  const proposeBox = h('div');
  const saveBtn = h('button', { className: 'primary', onclick: save }, 'Ijkmoment opslaan');
  const zoomLabel = h('span', { className: 'muted small' });
  const chooser = h('div', { className: 'small', style: { marginTop: '8px' } });  // keuze bij het doel / uit de lijst
  const clipSel = h('select', { onchange: e => { clip = clips.find(c => c.id === Number(e.target.value)); ctx.setParam('clip', clip.id); loadClip(); } },
    clips.map(c => h('option', { value: c.id, selected: c.id === clip.id }, c.filename)));
  const predToggle = h('label', { className: 'small' }, h('input', { type: 'checkbox', checked: true,
    onchange: e => { showPred = e.target.checked; drawFrame(); } }), ' toon voorspelling (geel)');

  // Begeleide modus (wizard): één video, een coach die zegt wat de volgende klik is, geen overzicht/uitleg.
  const wiz = ctx.wizard || null;
  const coach = h('div', { className: 'coach' });
  const stations = wiz ? { el: null, refresh() {}, cleanup() {} } : stationsPanel(match, id => {
    clip = clips.find(c => c.id === id) || clip; clipSel.value = clip.id; ctx.setParam('clip', clip.id); loadClip();
    clipSel.scrollIntoView({ behavior: 'smooth', block: 'start' });
  });
  root.append(...[
    wiz ? coach : null,
    wiz ? null : h('div', { className: 'hint' },
      'Tip: stel eerst in waar je stond (📍 hieronder, of via GPS als je video dat heeft). Dan is 1 punt + 1 lijn al genoeg. ' +
      'Zo werkt het: 1) Kies met de schuif een moment. 2) Klik in het beeld op een herkenbaar punt (hoek strafschopgebied, ' +
      'strafschopstip, doelpaal, hoekvlag...) of op een plek ergens op een veldlijn. 3) Klik hetzelfde punt of dezelfde lijn aan ' +
      'in de veldtekening rechts (lijnen worden rood als je erop klikt), of kies het onder de tekening uit de lijst. Klik je bij een doel, ' +
      'dan kies je of het de voet of de bovenkant van een paal is, of de lat. Een punt telt 2, een lijn telt mee met hoogstens 2 punten; ' +
      'je hebt samen 8 nodig, bijv. 4 punten, of 1 punt + 3 lijnen (handig vanaf de zijlijn: zijlijn, 16-meterlijn, doellijn). ' +
      'De witte lijnen laten zien of het klopt. Eén ijkmoment is genoeg: daarna legt de app het veld elke seconde zelf ' +
      'opnieuw op de witte lijnen (🤖 hieronder). De gele lijnen tonen op elk moment hoe goed het past; past het ergens niet, ' +
      'zet daar dan een extra ijkmoment.'),
    stations.el,
    wiz ? null : h('div', { className: 'row', style: { marginBottom: '12px' } }, 'Video:', clipSel),
    h('div', { className: 'grid2' },
      h('div', { className: 'panel' },
        h('div', { className: 'row', style: { marginBottom: '8px' } }, slider, timeLabel),
        h('div', { className: 'frame-wrap' }, frameCanvas),
        h('div', { className: 'row small', style: { marginTop: '6px' } },
          h('button', { title: 'Inzoomen', onclick: () => zoomBy(1.6) }, '🔍＋'),
          h('button', { title: 'Uitzoomen', onclick: () => zoomBy(1 / 1.6) }, '🔍－'),
          h('button', { onclick: () => { view = { s: 1, ox: 0, oy: 0 }; drawFrame(); } }, 'Passend'), zoomLabel,
          h('span', { style: { flex: 1 } }), predToggle),
        h('div', { className: 'small muted', style: { marginTop: '4px' } },
          'Inzoomen: knijp op het trackpad (of ⌥ + scrollen). Verschuiven: met twee vingers vegen, of Shift + slepen. ' +
          'Sleep een punt om het te verschuiven; rechtsklik verwijdert het. Punten die je zet schuiven mee met het veld als je ' +
          'naar een ander moment gaat, zodat je bijv. de verre hoekvlag op een later moment kunt toevoegen.')),
      h('div', {},
        h('div', { className: 'panel' }, h('h3', {}, 'Veld (klik op een punt of lijn)'), pitchCanvas, chooser),
        camPanel,
        h('div', { className: 'panel' },
          h('div', { className: 'row', style: { justifyContent: 'space-between' } }, h('h3', {}, 'Punten in dit ijkmoment'), errLabel),
          fitWarn,
          proposeBox,
          pairList,
          h('div', { className: 'row', style: { marginTop: '10px' } },
            saveBtn,
            h('button', { onclick: usePredicted }, 'Voorspelde punten overnemen'),
            h('button', { onclick: () => { pairs = []; editingId = null; proposal = null; redraw(); } }, 'Leegmaken'))),
        wiz ? null : h('div', { className: 'panel' }, h('h3', {}, 'Ijkmomenten van deze video'), kfList)))].filter(Boolean));

  const pv = new PitchView(pitchCanvas, { margin: 14 });  // ruimte om je eigen plek naast het veld aan te klikken
  requestAnimationFrame(() => { pv.resize(); drawPitch(); });
  drawChooser();

  function canvasCoords(e) {
    const r = frameCanvas.getBoundingClientRect();
    return [(e.clientX - r.left) * frameCanvas.width / r.width, (e.clientY - r.top) * frameCanvas.height / r.height];
  }
  function imgCoords(e) {
    const [cx, cy] = canvasCoords(e);
    return [cx / view.s + view.ox, cy / view.s + view.oy];
  }
  // schermpixels -> beeldpixels (zodat stippen en lijnen even groot blijven als je inzoomt)
  const pxPerCss = () => frameCanvas.width / (frameCanvas.getBoundingClientRect().width || frameCanvas.width) / view.s;

  // --- inzoomen en verschuiven ---------------------------------------------------------
  function clampView() {
    const W = frameCanvas.width, Hh = frameCanvas.height;
    view.s = Math.max(1, Math.min(8, view.s));
    view.ox = Math.max(0, Math.min(W - W / view.s, view.ox));
    view.oy = Math.max(0, Math.min(Hh - Hh / view.s, view.oy));
  }
  function zoomAt(cx, cy, k) {
    const ix = cx / view.s + view.ox, iy = cy / view.s + view.oy;
    view.s *= k; view.s = Math.max(1, Math.min(8, view.s));
    view.ox = ix - cx / view.s; view.oy = iy - cy / view.s;
    clampView(); drawFrame();
  }
  function zoomBy(k) { zoomAt(frameCanvas.width / 2, frameCanvas.height / 2, k); }
  frameCanvas.addEventListener('wheel', e => {
    if (!img) return;
    const zoom = e.ctrlKey || e.altKey || e.metaKey;  // knijpen op het trackpad geeft ctrlKey
    if (!zoom && view.s <= 1) return;  // niet ingezoomd: gewoon de pagina scrollen
    e.preventDefault();
    if (zoom) {
      const [cx, cy] = canvasCoords(e);
      zoomAt(cx, cy, Math.exp(-e.deltaY * (e.ctrlKey ? 0.01 : 0.002)));
    } else {
      const k = pxPerCss();
      view.ox += e.deltaX * k; view.oy += e.deltaY * k;
      clampView(); drawFrame();
    }
  }, { passive: false });

  frameCanvas.addEventListener('mousedown', e => {
    if (!img) return;
    if (e.button === 1 || (e.button === 0 && e.shiftKey)) {  // verschuiven
      e.preventDefault();
      pan = { x: e.clientX, y: e.clientY, ox: view.ox, oy: view.oy };
      return;
    }
    if (e.button !== 0) return;
    const p = imgCoords(e), tol = 12 * pxPerCss();
    const hit = pairs.findIndex(q => Math.hypot(q.img[0] - p[0], q.img[1] - p[1]) < tol);
    if (hit >= 0) { drag = hit; return; }
    if (pendingPitch) {
      addPair(pendingPitch, p, view.s); pendingPitch = null;
    } else {
      pendingImg = p; pendingZoom = view.s;
    }
    redraw();
  });
  window.addEventListener('mousemove', onPan);
  function onPan(e) {
    if (!pan) return;
    const r = frameCanvas.getBoundingClientRect(), k = frameCanvas.width / r.width / view.s;
    view.ox = pan.ox - (e.clientX - pan.x) * k; view.oy = pan.oy - (e.clientY - pan.y) * k;
    clampView(); drawFrame();
  }
  frameCanvas.addEventListener('mousemove', e => {
    if (drag === null) return;
    pairs[drag].img = imgCoords(e);
    pairs[drag].z = Math.max(pairs[drag].z || 1, view.s);  // ingezoomd verschoven: precies
    drawFrame(); scheduleFit();
  });
  window.addEventListener('mouseup', onUp);
  function onUp() { pan = null; if (drag !== null) { touched = pairs[drag]; drag = null; redraw(); } }
  frameCanvas.addEventListener('contextmenu', e => {
    e.preventDefault();
    const p = imgCoords(e), tol = 12 * pxPerCss();
    const hit = pairs.findIndex(q => Math.hypot(q.img[0] - p[0], q.img[1] - p[1]) < tol);
    if (hit >= 0) { pairs.splice(hit, 1); redraw(); }
  });
  pitchCanvas.addEventListener('click', async e => {
    const r = pitchCanvas.getBoundingClientRect();
    const [mx, my] = pv.toM(e.clientX - r.left, e.clientY - r.top);
    if (skipClick) { skipClick = false; return; }
    if (placingDir && hasCam()) { placingDir = false; await setYaw(mx, my); return; }
    if (placingCam) {
      placingCam = false;
      await setCamera({ x: Math.round(mx * 10) / 10, y: Math.round(my * 10) / 10, h: clip.cam_h || 1.6, source: 'hand' });
      return;
    }
    let best = null, bd = 2.5;
    for (const lm of pitchInfo.landmarks) {
      const d = Math.hypot(lm.x - mx, lm.y - my);
      if (d < bd) { bd = d; best = lm; }
    }
    if (!best) {  // geen punt in de buurt: dan een lijn?
      let ld = 2;
      for (const ln of pitchInfo.lines) {
        const d = segDist([mx, my], ln.from, ln.to);
        if (d < ld) { ld = d; best = { name: ln.name, line: [ln.from, ln.to] }; }
      }
    }
    // bij een doel: kiezen tussen de voet en de bovenkant van een paal, de lat of de doellijn
    const side = [['links', 0], ['rechts', pitchInfo.length]].find(([, x0]) =>
      Math.abs(mx - x0) < 4 && Math.abs(my - pitchInfo.width / 2) < 6);
    if (side) { goalChooser(side[0]); return; }
    if (!best) return toast('Klik dichter bij een wit punt of een lijn op het veld, of kies hieronder uit de lijst');
    pick(best);
  });

  // kijkrichting: sleep vanaf je blauwe stip, of klik op 🧭 en dan op het veld
  pitchCanvas.addEventListener('mousedown', e => {
    if (!hasCam() || placingCam) return;
    const r = pitchCanvas.getBoundingClientRect();
    const [mx, my] = pv.toM(e.clientX - r.left, e.clientY - r.top);
    const cx = fit.cam ? fit.cam.x : clip.cam_x, cy = fit.cam ? fit.cam.y : clip.cam_y;
    if (Math.hypot(mx - cx, my - cy) < 3.5) { dirDrag = [mx, my]; e.preventDefault(); }
  });
  const dirMove = e => {
    if (!dirDrag) return;
    const r = pitchCanvas.getBoundingClientRect();
    dirDrag = pv.toM(e.clientX - r.left, e.clientY - r.top); drawPitch();
  };
  const dirUp = async () => {
    if (!dirDrag) return;
    const [mx, my] = dirDrag; dirDrag = null;
    if (Math.hypot(mx - clip.cam_x, my - clip.cam_y) > 4) { skipClick = true; await setYaw(mx, my); }
    else drawPitch();
  };
  window.addEventListener('mousemove', dirMove); window.addEventListener('mouseup', dirUp);
  async function setYaw(mx, my) {
    const yaw = Math.round(Math.atan2(my - clip.cam_y, mx - clip.cam_x) * 180 / Math.PI);
    await setCamera({ yaw });  // zoekt zelf opnieuw als er nog geen ijkmoment is
    if (!pairs.length && !proposing && keyframes.some(k => !k.auto)) runPropose();  // ook dan: met kompas opnieuw zoeken
  }

  function pick(lm) {
    if (pendingImg) { addPair(lm, pendingImg, pendingZoom); pendingImg = null; }
    else pendingPitch = lm;
    drawChooser();
    redraw();
  }

  // Bij het doel liggen de palen vlak naast elkaar; daarom kies je hier met knoppen. Links en rechts
  // zoals jij het doel zag vanaf je plek (de app rekent om naar de tekening). Bovenkant = waar de lat zit.
  function goalChooser(side) { goalSide = side; drawChooser(); }
  // welke paal ("boven"/"onder" in de tekening) zag de filmer links? Zonder plek: vanaf de onderste zijlijn
  function leftPost(side) {
    const gx = side === 'links' ? 0 : pitchInfo.length, gy = pitchInfo.width / 2;
    const cx = hasCam() ? clip.cam_x : pitchInfo.length / 2, cy = hasCam() ? clip.cam_y : pitchInfo.width + 5;
    const dx = gx - cx, dy = gy - cy;
    // y wijst omlaag in de tekening: een negatief kruisproduct = links van de kijkrichting
    const cross = (y, x) => dx * (y - cy) - dy * (x - cx);
    return cross(gy - 3.66, gx) < cross(gy + 3.66, gx) ? 'boven' : 'onder';
  }
  // "Doelpaal links onder, bovenkant" -> "Doel links: linkerpaal, bovenkant" (vanuit de filmer)
  function label(name) {
    const m = /^Doelpaal (links|rechts) (boven|onder)(, bovenkant)?$/.exec(name || '');
    if (!m) return name;
    return `Doel ${m[1]}: ${leftPost(m[1]) === m[2] ? 'linkerpaal' : 'rechterpaal'}, ${m[3] ? 'bovenkant' : 'voet'}`;
  }
  function drawChooser() {
    const named = n => pitchInfo.landmarks.find(l => l.name === n);
    const up = n => pitchInfo.elevated.find(l => l.name === n);
    const btn = (label, lm, title) => lm ? h('button', { className: pendingPitch?.name === lm.name ? 'primary' : '', title,
      onclick: () => { goalSide = null; pick(lm); } }, label) : null;
    const rows = [];
    if (goalSide) {
      const s = goalSide, L = leftPost(s), R = L === 'boven' ? 'onder' : 'boven';
      const lat = pitchInfo.elevated_lines.find(l => l.name === `Lat ${s}`);
      const goalLine = pitchInfo.lines.find(l => l.name === `Doellijn ${s}`);
      const post = (paal, wie) => h('div', { className: 'post-col' }, h('b', {}, wie),
        btn('⬆ Bovenkant', up(`Doelpaal ${s} ${paal}, bovenkant`), `Bovenkant van de ${wie.toLowerCase()}, waar de lat begint (2,44 m)`),
        btn('⬇ Onderkant (voet)', named(`Doelpaal ${s} ${paal}`), `Waar de ${wie.toLowerCase()} de grond raakt`));
      rows.push(h('div', { className: 'hint', style: { margin: '0 0 6px' } },
        h('b', {}, `Doel ${s}: wat klik je aan in het beeld?`),
        h('div', { className: 'muted' }, `Links en rechts zoals jij het doel zag ${hasCam() ? 'vanaf je plek' : 'vanaf de zijlijn onderaan de tekening'}.`),
        h('div', { className: 'row', style: { marginTop: '8px', gap: '14px', alignItems: 'flex-start' } },
          post(L, 'Linkerpaal'), post(R, 'Rechterpaal'),
          h('div', { className: 'post-col' }, h('b', {}, 'Of'),
            lat ? btn('Ergens op de lat', { name: lat.name, line3: [lat.from, lat.to] }, 'Een plek op de lat (2,44 m hoog)') : null,
            goalLine ? btn('Ergens op de doellijn', { name: goalLine.name, line: [goalLine.from, goalLine.to] }) : null,
            h('button', { onclick: () => { goalSide = null; drawChooser(); } }, 'Annuleren'))),
        hasCam() ? null : h('div', { className: 'muted', style: { marginTop: '4px' } },
          'De bovenkant en de lat tellen mee zodra de app weet waar je stond (📍 hieronder).')));
    }
    // alles ook uit een lijst te kiezen, bijvoorbeeld een zijlijn als je verder geen vast punt ziet
    const opt = (v, text) => h('option', { value: v, selected: pendingPitch && v.endsWith('|' + pendingPitch.name) }, text);
    const list = h('select', { onchange: e => {
      const [kind, name] = e.target.value.split('|');
      e.target.value = '';
      if (kind === 'p') { const l = named(name); pick(l); }
      else if (kind === 'l') { const l = pitchInfo.lines.find(x => x.name === name); pick({ name, line: [l.from, l.to] }); }
      else if (kind === 'u') pick(up(name));
      else if (kind === 'ul') { const l = pitchInfo.elevated_lines.find(x => x.name === name); pick({ name, line3: [l.from, l.to] }); }
    } },
      h('option', { value: '' }, 'Of kies een punt of lijn uit de lijst…'),
      h('optgroup', { label: 'Lijnen (een plek ergens op de lijn)' }, pitchInfo.lines.map(l => opt(`l|${l.name}`, l.name))),
      h('optgroup', { label: 'Punten op de grond' }, pitchInfo.landmarks.map(l => opt(`p|${l.name}`, label(l.name)))),
      h('optgroup', { label: 'Doel, in de lucht (2,44 m)' }, [
        ...pitchInfo.elevated.map(l => opt(`u|${l.name}`, label(l.name))),
        ...pitchInfo.elevated_lines.map(l => opt(`ul|${l.name}`, `${l.name} (ergens op de lat)`))]));
    rows.push(h('div', { className: 'row' }, list,
      h('span', { className: 'muted' }, 'Zie je alleen een zijlijn? Kies "Zijlijn onder" (de kant waar jij staat als je onderaan de tekening staat) en klik er een paar plekken op aan.')));
    chooser.replaceChildren(...rows);
  }

  // z = hoe ver je was ingezoomd bij het klikken: zo'n klik is preciezer en telt zwaarder mee
  function addPair(lm, p, z = 1) {
    z = Math.round(Math.min(4, Math.max(1, z)) * 10) / 10;
    if (lm.line || lm.line3) {  // meerdere punten op dezelfde lijn mogen
      pairs.push(lm.line ? { name: lm.name, img: p, line: lm.line, z } : { name: lm.name, img: p, line3: lm.line3, z });
      touched = pairs[pairs.length - 1];
      return;
    }
    pairs = pairs.filter(q => q.name !== lm.name);
    pairs.push(lm.h != null ? { name: lm.name, img: p, pitch3: [lm.x, lm.y, lm.h], z } : { name: lm.name, img: p, pitch: [lm.x, lm.y], z });
    touched = pairs[pairs.length - 1];
  }

  function segDist(p, a, b) {
    const dx = b[0] - a[0], dy = b[1] - a[1];
    const t = Math.max(0, Math.min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / (dx * dx + dy * dy || 1)));
    return Math.hypot(p[0] - a[0] - t * dx, p[1] - a[1] - t * dy);
  }

  // veld -> beeld (voor het tekenen van de lijnen over het beeld)
  function toImageH(list) {
    const G = fitCalibration(list);
    return G ? inv(G) : null;
  }

  // Wat is nodig? Met camerapositie rekent de server met een cameramodel (1 punt + 1 lijn is genoeg)
  function needInfo() {
    const info = calibrationInfo(pairs);
    const up = pairs.filter(p => p.pitch3 || p.line3);
    if (!hasCam()) {
      return up.length && !info.ok ? { ...info, hint: `${info.hint}; de bovenkant van een paal en de lat tellen pas mee als je je positie instelt (📍)` } : info;
    }
    // met camerapositie tellen ook de punten in de lucht mee
    const upLines = new Set(up.filter(p => p.line3).map(p => p.name));
    const dof = info.dof + 2 * up.filter(p => p.pitch3).length + Math.min(2 * upLines.size, up.filter(p => p.line3).length);
    const ok = dof >= 3;
    return { ...info, dof, ok, hint: ok ? '' : `nog ${3 - dof} nodig (met je positie: 1 punt + 1 lijn is genoeg)` };
  }

  let fitTimer, fitSeq = 0;
  function scheduleFit() {
    clearTimeout(fitTimer);
    if (!hasCam()) {
      // meteen een snelle schatting tekenen; de server legt de lijnen daarna precies door je punten
      const info = calibrationInfo(pairs);
      const G = info.ok ? fitCalibration(pairs) : null;
      fit = { H: G ? inv(G) : null, err: G ? calibrationError(G, pairs) : null, cam: null, msg: needInfo().hint };
      drawFrame(); drawErr();
      if (!info.ok) return;
    }
    fitTimer = setTimeout(async () => {
      const seq = ++fitSeq, info = needInfo();
      if (!info.ok) { fit = { H: null, err: null, cam: null, msg: info.hint }; drawFrame(); drawErr(); drawCamPanel(); return; }
      const r = await api(`/clips/${clip.id}/calibrate-preview`, { json: { points: pairs } }).catch(() => null);
      if (seq !== fitSeq) return;
      const wasBad = badFit();
      fit = r && r.ok ? { H: r.H, err: r.error_m, errPx: r.error_px, badPx: r.bad_px, cam: r.camera, goals: r.goals || [], msg: '', without: r.without || null, withoutOk: r.without_ok }
        : { H: null, err: null, cam: null, msg: r?.message || 'mislukt' };
      if (!badFit()) trigger = null;
      else if (!wasBad || !pairs.includes(trigger)) trigger = pairs.includes(touched) ? touched : null;
      drawFrame(); drawErr(); drawCamPanel(); drawPitch();
    }, 150);
  }

  async function setCamera(data) {
    const c = await api(`/clips/${clip.id}/camera`, { method: 'PATCH', json: data });
    Object.assign(clip, c);
    drawCamPanel(); drawPitch(); scheduleFit(); await loadKeyframes();
    proposeIfEmpty();
  }
  // nog geen eigen ijkmoment en geen punten gezet: dan meteen een voorstel doen
  function proposeIfEmpty() {
    if (hasCam() && !pairs.length && !keyframes.some(k => !k.auto)) runPropose();
    else drawProposal();
  }

  function drawCamPanel() {
    drawCoach();
    const hNow = clip.cam_h || 1.6;
    camPanel.replaceChildren(...[
      h('h3', {}, '📍 Waar stond je bij het filmen?'),
      h('div', { className: 'small', style: { marginBottom: '8px' } },
        hasCam()
          ? [`Op ${clip.cam_x.toFixed(0)} m langs het veld, ${clip.cam_y > pitchInfo.width ? (clip.cam_y - pitchInfo.width).toFixed(0) + ' m buiten de zijlijn' : clip.cam_y < 0 ? (-clip.cam_y).toFixed(0) + ' m achter de verre zijlijn' : 'op het veld'}`,
             clip.cam_source === 'gps' ? ` (via GPS${clip.gps_acc ? ', ±' + Math.max(5, Math.round(clip.gps_acc)) + ' m' : ''} – klik gerust zelf preciezer)`
               : clip.cam_source === 'kopie' ? ' (overgenomen van een video vanaf dezelfde plek – klik gerust zelf preciezer)' : ' (aangeklikt)',
             fit.cam ? h('div', { className: 'muted' }, `Geschat uit je klikken: ${fit.cam.h} m hoog, kijkhoek ${fit.cam.hfov_deg}°`) : null]
          : 'Nog niet ingesteld. Weet de app waar je stond, dan is 1 punt + 1 lijn al genoeg om te kalibreren.'),
      h('div', { className: 'row' },
        h('button', { className: placingCam ? 'primary' : '', onclick: () => { placingCam = !placingCam; drawCamPanel(); } },
          placingCam ? 'Klik nu op de veldtekening…' : '📍 Klik waar je stond'),
        clip.gps_lat != null ? h('button', { 'data-gps': '1', title: 'Stuurt alleen de GPS-positie van deze video naar OpenStreetMap om het veld te vinden', onclick: async e => {
          e.target.disabled = true; e.target.textContent = 'Zoeken...'; gpsError = null;
          try {
            const r = await api(`/clips/${clip.id}/camera/gps`, { method: 'POST' }).catch(err => { gpsError = err.message || String(err); return null; });
            if (!r) return;
            Object.assign(clip, r.clip);
            toast(`Veld gevonden (${r.pitch_length} × ${r.pitch_width} m). Je positie staat op de tekening.`);
            osmSize = (Math.abs(r.pitch_length - pitchInfo.length) > 1 || Math.abs(r.pitch_width - pitchInfo.width) > 1)
              ? [r.pitch_length, r.pitch_width] : null;
            drawPitch(); scheduleFit(); await loadKeyframes();
            proposeIfEmpty();
          } finally { drawCamPanel(); }
        } }, '📡 Zoek via GPS') : null,
        hasCam() ? h('button', { onclick: () => setCamera({ x: null, y: null, source: null }) }, 'Wissen') : null),
      hasCam() ? h('div', { style: { marginTop: '10px' } },
        h('div', { className: 'small', style: { marginBottom: '4px' } }, 'Waar keek je op dit moment naartoe? (helpt de app, vooral met weinig punten)'),
        h('div', { className: 'row' },
          h('button', { className: placingDir ? 'primary' : '', onclick: () => { placingDir = !placingDir; placingCam = false; drawCamPanel(); } },
            placingDir ? 'Klik nu op het veld waar het midden van je beeld is…' : '🧭 Geef kijkrichting aan'),
          clip.cam_yaw != null ? h('button', { onclick: () => setCamera({ yaw: null }) }, 'Wissen') : null,
          h('span', { className: 'small muted' }, clip.cam_yaw != null ? '✓ aangegeven (blauwe pijl)' : 'of sleep vanaf je blauwe stip')),
        h('div', { className: 'small muted', style: { marginTop: '4px' } },
          'Wordt bij het opslaan bewaard bij dit ijkmoment. Zwenk je later in de video, geef hem dan voor dat moment opnieuw aan.')) : null,
      h('div', { style: { marginTop: '10px' } }, h('div', { className: 'small', style: { marginBottom: '4px' } }, 'Hoe hoog hield je de telefoon?'),
        heightField(hNow, v => hasCam() ? setCamera({ h: v }) : (clip.cam_h = v), fit.cam ? fit.cam.h : null)),
      gpsError ? h('div', { className: 'hint warn-hint', style: { marginTop: '8px' } }, h('b', {}, 'Zoeken via GPS lukte niet. '), gpsError,
        h('div', {}, 'Klik dan zelf op "📍 Klik waar je stond" en daarna op de veldtekening.')) : null,
      clip.gps_lat != null && !hasCam() ? h('div', { className: 'small muted', style: { marginTop: '6px' } },
        'Deze video bevat een GPS-positie. "Zoek via GPS" zoekt het veld op in OpenStreetMap (alleen de coördinaat wordt verstuurd).') : null,
      osmSize ? h('div', { className: 'hint', style: { marginTop: '8px' } },
        `Volgens OpenStreetMap is dit veld ${osmSize[0]} × ${osmSize[1]} m (de app rekent nu met ${pitchInfo.length} × ${pitchInfo.width} m). `,
        h('button', { className: 'primary', onclick: async () => {
          await api(`/clips/${clip.id}/camera/gps`, { json: { adopt_size: true } });
          toast('Veldmaten overgenomen'); ctx.reload();
        } }, 'Overnemen')) : null,
      sizeRow(),
    ].filter(Boolean));
  }

  // Veldmaten: amateurvelden zijn vaak kleiner dan 105 x 68 m
  function sizeRow() {
    const len = h('input', { type: 'number', min: 40, max: 130, step: 0.5, value: pitchInfo.length, style: { width: '72px' } });
    const wid = h('input', { type: 'number', min: 25, max: 100, step: 0.5, value: pitchInfo.width, style: { width: '64px' } });
    return h('div', { style: { marginTop: '10px', paddingTop: '8px', borderTop: '1px solid var(--border)' } },
      h('div', { className: 'row small' }, h('b', {}, 'Veldmaten'), len, '×', wid, 'm',
        h('button', { onclick: async () => {
          if (Number(len.value) === pitchInfo.length && Number(wid.value) === pitchInfo.width) return;
          if (!confirm('Veldmaten wijzigen? Je kalibratiepunten schuiven mee; het automatisch bijstellen begint opnieuw.')) return;
          await api(`/matches/${match.id}`, { method: 'PATCH', json: { pitch_length: Number(len.value), pitch_width: Number(wid.value) } });
          toast('Veldmaten opgeslagen'); ctx.reload();
        } }, 'Opslaan')),
      h('div', { className: 'small muted', style: { marginTop: '4px' } },
        'Standaard 105 × 68 m. Amateurvelden zijn vaak kleiner (bijv. 100 × 64); met de echte maten kloppen afstanden en posities beter. ',
        'Het strafschopgebied en de middencirkel zijn altijd even groot.'));
  }

  function drawFrame() {
    const c = frameCanvas.getContext('2d');
    if (!img) return;
    c.setTransform(1, 0, 0, 1, 0, 0);
    c.fillStyle = '#000'; c.fillRect(0, 0, frameCanvas.width, frameCanvas.height);
    c.setTransform(view.s, 0, 0, view.s, -view.ox * view.s, -view.oy * view.s);
    c.drawImage(img, 0, 0);
    zoomLabel.textContent = view.s > 1.01 ? `${view.s.toFixed(1)}×` : '';
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
    if (showPred && predicted.length) {
      const Hp = toImageH(predicted);
      if (Hp) drawLines(Hp, 'rgba(255,214,0,.85)', [8 * pxPerCss(), 6 * pxPerCss()]);
    }
    const bad = badFit(), col = bad ? 'rgba(255,90,95,.95)' : 'rgba(255,255,255,.9)', dash = bad ? [10 * pxPerCss(), 7 * pxPerCss()] : [];
    if (fit.H) drawLines(fit.H, col, dash);
    for (const g of fit.goals || []) {  // de doelen volgens het cameramodel (palen en lat)
      c.save(); c.strokeStyle = col; c.setLineDash(dash); c.lineWidth = lw; c.beginPath();
      g.forEach(([x, y], i) => (i ? c.lineTo(x, y) : c.moveTo(x, y))); c.stroke(); c.restore();
    }
    const r = 7 * pxPerCss();
    for (const i of badFit() ? suspect() : []) {  // botsende punten: gele ring in het beeld
      const q = pairs[i]?.img;
      if (!q) continue;
      c.save(); c.beginPath(); c.arc(q[0], q[1], r * 2.2, 0, 2 * Math.PI);
      c.strokeStyle = '#ffc53d'; c.lineWidth = lw * 1.2; c.stroke(); c.restore();
    }
    pairs.forEach((p, i) => {
      c.beginPath(); c.arc(p.img[0], p.img[1], r, 0, 2 * Math.PI);
      c.fillStyle = '#ff2d55'; c.fill(); c.lineWidth = lw * .6; c.strokeStyle = '#fff'; c.stroke();
      c.fillStyle = '#fff'; c.font = `bold ${r * 1.5}px sans-serif`; c.fillText(String(i + 1), p.img[0] + r * 1.3, p.img[1] - r);
    });
    if (pendingImg) {
      c.beginPath(); c.arc(pendingImg[0], pendingImg[1], r, 0, 2 * Math.PI);
      c.strokeStyle = '#ffd600'; c.lineWidth = lw; c.stroke();
    }
    const tip = nextTip();
    if (tip?.at) {  // voorgesteld volgend punt: gele ring waar de app het verwacht
      c.save(); c.beginPath(); c.arc(tip.at[0], tip.at[1], r * 2.4, 0, 2 * Math.PI);
      c.strokeStyle = '#ffd600'; c.lineWidth = lw; c.setLineDash([5 * pxPerCss(), 4 * pxPerCss()]); c.stroke(); c.restore();
    }
  }

  function drawPitch() {
    pv.draw();
    const c = pv.ctx;
    const usedLines = new Set(pairs.filter(p => p.line).map(p => p.name));
    for (const ln of pitchInfo.lines) {
      const on = usedLines.has(ln.name), pend = pendingPitch?.name === ln.name;
      if (!on && !pend) continue;
      c.save(); c.strokeStyle = pend ? '#ffd600' : '#ff2d55'; c.lineWidth = 4;
      c.beginPath(); c.moveTo(...pv.toPx(...ln.from)); c.lineTo(...pv.toPx(...ln.to)); c.stroke(); c.restore();
    }
    const used = new Map(pairs.map((p, i) => [p.name, i + 1]).filter(([n]) => !usedLines.has(n)));
    for (const lm of pitchInfo.landmarks) {
      const n = used.get(lm.name);
      if (n) pv.dot(lm.x, lm.y, '#ff2d55', 7, String(n));
      else pv.dot(lm.x, lm.y, pendingPitch?.name === lm.name ? '#ffd600' : 'rgba(255,255,255,.9)', 3.5);
    }
    // punten in de lucht (bovenkant paal, lat): net achter het doel getekend, met een streepje naar de paal
    pairs.forEach((p, i) => {
      const pts = p.pitch3 ? [p.pitch3] : p.line3 || null;
      if (!pts) return;
      const out = pts[0][0] < pitchInfo.length / 2 ? -3.2 : 3.2;
      c.save(); c.strokeStyle = '#ff2d55'; c.lineWidth = 2;
      if (p.line3) {
        c.lineWidth = 4; c.beginPath(); c.moveTo(...pv.toPx(pts[0][0] + out / 2, pts[0][1])); c.lineTo(...pv.toPx(pts[1][0] + out / 2, pts[1][1])); c.stroke();
        c.restore(); pv.dot(pts[0][0] + out, (pts[0][1] + pts[1][1]) / 2, '#ff2d55', 7, String(i + 1));
      } else {
        c.beginPath(); c.moveTo(...pv.toPx(pts[0][0], pts[0][1])); c.lineTo(...pv.toPx(pts[0][0] + out, pts[0][1])); c.stroke();
        c.restore(); pv.dot(pts[0][0] + out, pts[0][1], '#ff2d55', 7, String(i + 1));
        c.save(); c.font = '600 10px sans-serif'; c.fillStyle = '#fff'; c.textAlign = 'center';
        c.fillText('↑', ...pv.toPx(pts[0][0] + out * 1.75, pts[0][1] + 0.5)); c.restore();
      }
    });
    if (hasCam()) {  // aangegeven kijkrichting: gestippelde pijl
      const yaw = dirDrag ? Math.atan2(dirDrag[1] - clip.cam_y, dirDrag[0] - clip.cam_x) * 180 / Math.PI : clip.cam_yaw;
      if (yaw != null) {
        const a = yaw * Math.PI / 180, L = 28, [x0, y0] = [clip.cam_x, clip.cam_y], [x1, y1] = [x0 + L * Math.cos(a), y0 + L * Math.sin(a)];
        c.save(); c.strokeStyle = '#4db5ff'; c.fillStyle = '#4db5ff'; c.lineWidth = 3; c.setLineDash([7, 5]);
        c.beginPath(); c.moveTo(...pv.toPx(x0, y0)); c.lineTo(...pv.toPx(x1, y1)); c.stroke(); c.setLineDash([]);
        const [hx, hy] = pv.toPx(x1, y1), b = Math.atan2(...[pv.toPx(x1, y1)[1] - pv.toPx(x0, y0)[1], pv.toPx(x1, y1)[0] - pv.toPx(x0, y0)[0]]);
        c.beginPath(); c.moveTo(hx, hy); c.lineTo(hx - 12 * Math.cos(b - .45), hy - 12 * Math.sin(b - .45));
        c.lineTo(hx - 12 * Math.cos(b + .45), hy - 12 * Math.sin(b + .45)); c.closePath(); c.fill(); c.restore();
      }
    }
    if (hasCam()) {  // jouw plek + kijkrichting (als die al bekend is)
      const cx = fit.cam ? fit.cam.x : clip.cam_x, cy = fit.cam ? fit.cam.y : clip.cam_y;
      if (fit.cam) {
        const a = fit.cam.yaw_deg * Math.PI / 180, half = fit.cam.hfov_deg * Math.PI / 360;
        c.save(); c.fillStyle = badFit() ? 'rgba(255,90,95,.16)' : 'rgba(10,132,255,.18)'; c.beginPath(); c.moveTo(...pv.toPx(cx, cy));
        c.lineTo(...pv.toPx(cx + 60 * Math.cos(a - half), cy + 60 * Math.sin(a - half)));
        c.lineTo(...pv.toPx(cx + 60 * Math.cos(a + half), cy + 60 * Math.sin(a + half))); c.closePath(); c.fill(); c.restore();
      }
      pv.dot(cx, cy, '#0a84ff', 7);
      c.save(); c.font = '600 12px sans-serif'; c.fillStyle = '#fff'; c.textAlign = 'center';
      c.fillText('jij', ...pv.toPx(cx, cy + 4.5)); c.restore();
    }
  }

  function drawPairs() {
    const bad = suspect();
    pairList.replaceChildren(...pairs.map((p, i) => h('div', { className: `list-item${bad.includes(i) ? ' suspect' : ''}`, title: bad.includes(i) ? 'Dit punt past niet bij de rest' : '' },
      h('b', {}, i + 1), h('span', { style: { flex: 1 } }, p.line ? `op lijn: ${p.name}` : p.line3 ? `op de lat: ${p.name}` : label(p.name),
        offScreen(p.img) ? h('span', { className: 'muted small', title: 'Dit punt ligt nu buiten beeld (gezet op een ander moment). Het telt gewoon mee.' }, ' · buiten beeld') : null),
      h('button', { onclick: () => removePair(i) }, '×'))));
    const tip = badFit() ? null : nextTip();
    fitWarn.replaceChildren(badFitHint() || (tip ? h('div', { className: 'hint', style: { margin: '8px 0' } }, h('b', {}, 'Tip: '), tip.text) : ''));
    if (!pairs.length) pairList.append(h('div', { className: 'muted small' },
      pendingImg ? 'Klik nu het bijbehorende punt op het veld.' : pendingPitch ? `Klik nu ${pendingPitch.line || pendingPitch.line3 ? 'een plek op ' : ''}"${label(pendingPitch.name)}" in het beeld.` : 'Nog geen punten.'));
  }

  // Past één klik niet bij de rest? Dan trekt die de hele kalibratie scheef (ook de kijkrichting).
  // De server zoekt de klik zonder welke de rest wél klopt. Index, of -1 (niets verdachts of onduidelijk).
  // Oordeel in beeldpixels: in de verte is 1 pixel al meters, dus een nette klik op een verre hoekvlag
  // mag niet als 'fout' tellen. Zonder pixelmaat (snelle schatting in de browser) nog niet oordelen.
  function badFit() { return !!(fit.H && fit.errPx != null && fit.errPx > (fit.badPx || 12)); }
  const errText = () => `${fit.errPx.toFixed(0)} pixels${fit.err != null ? ` (≈ ${fit.err.toFixed(1)} m)` : ''}`;

  // Welk punt helpt nu het meest? Zoals een plank vastpakken: aan de uiteinden, niet met de handen
  // naast elkaar. De app kiest het herkenbare punt in beeld dat het verst van je klikken ligt.
  function nextTip() {
    if (!img || !fit.H || badFit() || !pairs.length || pairs.length >= 6 || pendingImg || pendingPitch) return null;
    const W = frameCanvas.width, Hh = frameCanvas.height;
    const pts = pairs.map(p => p.img).filter(q => !offScreen(q));
    if (!pts.length) return null;
    const used = new Set(pairs.map(p => p.name));
    let best = null;
    for (const lm of pitchInfo.landmarks) {
      if (used.has(lm.name)) continue;
      const [u, v, w] = apply(fit.H, [lm.x, lm.y]);
      if (!(w > 0) || u < 0.04 * W || u > 0.96 * W || v < 0.04 * Hh || v > 0.96 * Hh) continue;
      const d = Math.min(...pts.map(q => Math.hypot(q[0] - u, q[1] - v)));
      if (!best || d > best.d) best = { d, lm, at: [u, v] };
    }
    if (best && best.d > 0.3 * W) return { text: `Klik ook "${label(best.lm.name)}" aan (gele ring in het beeld): ver van je andere punten, dus het maakt de kalibratie stabieler.`, at: best.at, lm: best.lm };
    const ys = pts.map(q => q[1]);
    if (Math.max(...ys) < 0.55 * Hh) return { text: 'Klik ook iets dichtbij aan, onderin je beeld, bijvoorbeeld een plek op de zijlijn voor je.' };
    if (Math.min(...ys) > 0.45 * Hh) return { text: 'Klik ook iets ver weg aan, bovenin je beeld: de verre zijlijn, een hoekvlag of een doel.' };
    return null;
  }

  // Welke klik veroorzaakt het rood, en met welke andere klik botst hij? Per klik weet de server of
  // de rest zonder die klik wél klopt (fit.without). Zo vind je de getuige met het afwijkende verhaal.
  function conflict() {
    if (!badFit() || !fit.without || fit.without.length !== pairs.length) return null;
    const ok = fit.without.map((v, i) => [v, i]).filter(([v]) => v != null && v <= fit.withoutOk).sort((a, b) => a[0] - b[0]).map(([, i]) => i);
    const t = trigger ? pairs.indexOf(trigger) : -1;
    const nm = i => `punt ${i + 1} (${label(pairs[i].name)})`;
    const since = t >= 0 ? `Sinds ${nm(t)} klopt het niet meer: ` : '';
    if (t >= 0 && ok.includes(t)) {
      const other = ok.find(i => i !== t);
      if (other != null) return { text: `${since}${nm(t)} en ${nm(other)} spreken elkaar tegen. Eén van beide staat verkeerd.`, remove: [t, other], mark: [t, other] };
      return { text: `${since}dat punt past niet bij de andere punten. Staat het op de goede plek, en is het goed gekoppeld?`, remove: [t], mark: [t] };
    }
    if (ok.length === 1) return { text: `${since}${nm(ok[0])} past niet bij de rest${t >= 0 ? ` (o.a. niet bij punt ${t + 1})` : ''}.`, remove: [ok[0]], mark: [ok[0]] };
    if (ok.length >= 2) return { text: `${since}${nm(ok[0])} of ${nm(ok[1])} staat verkeerd; zonder één van beide klopt de rest wel.`, remove: [ok[0], ok[1]], mark: [ok[0], ok[1]] };
    return { text: `${since}geen enkel los punt is de boosdoener; waarschijnlijk staan er meer verkeerd${t >= 0 ? `. Begin bij punt ${t + 1}` : ''}.`, remove: t >= 0 ? [t] : [], mark: t >= 0 ? [t] : [] };
  }
  function suspect() { return conflict()?.mark || []; }
  function removePair(i) {
    if (pairs[i] === touched) touched = null;
    pairs.splice(i, 1); redraw();
  }
  function badFitHint() {
    if (!badFit()) return null;
    const c = conflict();
    return h('div', { className: 'hint warn-hint', style: { margin: '8px 0' } },
      h('b', {}, `De punten passen niet bij elkaar (gemiddeld ${errText()} ernaast). `),
      c ? c.text + ' ' : '',
      'Haal het weg of sleep het naar de goede plek. Vaak gaat het om linker- en rechterpaal die verwisseld zijn.',
      c && c.remove.length ? h('div', { className: 'row', style: { marginTop: '6px', gap: '6px' } },
        c.remove.map(i => h('button', { className: 'small', onclick: () => removePair(i) }, `Punt ${i + 1} weghalen`))) : null);
  }

  // Wat moet je nu doen? (alleen in de wizard)
  function drawCoach() {
    if (!wiz) return;
    const info = needInfo();
    let step, text, extra = null;
    if (proposing) [step, text] = ['…', 'De app zoekt het veld vanaf jouw plek. Even geduld (± 10 seconden).'];
    else if (!hasCam() && !pairs.length) {
      [step, text] = ['1', 'Waar stond je? Klik hieronder op 📡 Zoek via GPS, of klik op de veldtekening rechts op de plek waar je stond.'];
      extra = clip.gps_lat != null ? h('button', { className: 'primary', onclick: () => document.querySelector('[data-gps]')?.click() }, '📡 Zoek mijn plek via GPS') : null;
    } else if (proposal && !proposal.failed) {
      [step, text] = ['2', 'Voorstel van de app. Vallen de witte lijnen op de witte lijnen in het beeld? Dan klopt het. Zo niet, sleep een punt naar de goede plek.'];
      extra = h('button', { className: 'primary', onclick: save }, '✓ Klopt – opslaan');
    } else if (pendingImg) [step, text] = ['2', 'Klik nu hetzelfde punt (of dezelfde lijn) aan in de veldtekening rechts, of kies het in de lijst eronder.'];
    else if (pendingPitch) [step, text] = ['2', 'Klik nu dat punt in het beeld links aan.'];
    else if (!info.ok) [step, text] = ['2', `Klik in het beeld op iets wat je herkent: een hoek van het strafschopgebied, de middenstip, een doelpaal, of een plek op een veldlijn. ${info.hint ? '(' + info.hint + ')' : ''} Zoom in voor precisie.`];
    else if (badFit()) {
      const c = conflict();
      [step, text] = ['!', `De punten passen nog niet bij elkaar (${errText()} ernaast). ` +
        (c ? c.text + ' Haal het weg (×) of sleep het naar de goede plek.' : 'Controleer de punten in de lijst: is er één verkeerd gekoppeld?')];
      if (c && c.remove.length) extra = h('div', { className: 'row', style: { gap: '6px' } },
        c.remove.map(i => h('button', { className: 'small', onclick: () => removePair(i) }, `Punt ${i + 1} weghalen`)));
    } else {
      const tip = nextTip();
      [step, text] = ['3', 'Liggen de witte lijnen nu op het veld? Sla dan op. Nog niet goed? Sleep een punt of klik er een bij.' + (tip ? ' Tip: ' + tip.text : '')];
      extra = h('button', { className: 'primary', onclick: save }, '✓ Opslaan');
    }
    if (proposal?.failed && !proposing) text = `De app vond het veld niet zelf (${proposal.failed}). ` + text;
    coach.classList.toggle('bad', step === '!');
    coach.replaceChildren(...[h('span', { className: 'coach-step' }, step), h('div', { style: { flex: 1 } }, text), extra].filter(Boolean));
  }

  function drawErr() {
    drawCoach(); drawPairs();
    const info = needInfo(), need = hasCam() ? 3 : 8;
    if (fit.H && fit.errPx != null) {
      errLabel.replaceChildren(h('span', { className: `badge ${badFit() ? 'err' : 'ok'}`,
        title: 'Hoe ver je kliks gemiddeld naast de kalibratie liggen, in beeldpixels (en ongeveer in meters op het veld; in de verte is 1 pixel al meters)' },
        `afwijking ${errText()}`));
    } else if (fit.H && fit.err != null) {
      errLabel.replaceChildren(h('span', { className: 'badge' }, `afwijking ≈ ${fit.err.toFixed(1)} m`));
    } else errLabel.replaceChildren(h('span', { className: 'muted small' },
      `${Math.min(info.dof, need)}/${need} · ${fit.msg || info.hint || 'niet eenduidig'}`));
  }

  function offScreen([x, y]) { return !img || x < 0 || y < 0 || x > frameCanvas.width || y > frameCanvas.height; }

  function redraw() { drawFrame(); drawPitch(); drawPairs(); drawProposal(); scheduleFit(); }

  // Gezette punten schuiven mee met het veld als je naar een ander moment gaat (via de gemeten
  // camerabeweging, of door de twee beelden te vergelijken als de video nog niet geanalyseerd is).
  async function warpPairs(fromT, toT) {
    const list = [...pairs.map(p => p.img), ...(pendingImg ? [pendingImg] : [])];
    if (!list.length || fromT == null || Math.abs(fromT - toT) < 1e-3) return;
    const r = await api(`/clips/${clip.id}/warp`, { json: { from_t: fromT, to_t: toT, points: list } }).catch(() => null);
    if (!r || !r.ok) {
      toast(`Punten konden niet meebewegen: ${r?.message || 'fout'}. Ze zijn gewist.`, true);
      pairs = []; pendingImg = null; editingId = null; proposal = null;
      return;
    }
    pairs.forEach((p, i) => { p.img = r.points[i]; });
    if (pendingImg) pendingImg = r.points[pairs.length];
    if (editingId) {  // het opgeslagen ijkmoment blijft zoals het was; opslaan maakt een nieuw
      editingId = null;
      toast('De punten zijn meegeschoven met het veld. Opslaan maakt een nieuw ijkmoment op dit moment.');
    }
  }

  async function loadFrame({ warp = false } = {}) {
    // De server rondt af op het dichtstbijzijnde geanalyseerde frame en geeft de exacte tijd terug
    const res = await fetch(`/api/clips/${clip.id}/frame?t=${t.toFixed(3)}`);
    if (!res.ok) return toast('Frame kon niet geladen worden', true);
    const ft = Number(res.headers.get('X-Frame-Time'));
    if (!isNaN(ft)) { t = ft; slider.value = t; }
    if (warp) await warpPairs(frameT, t);
    frameT = t;
    timeLabel.textContent = `${fmtTime(t)} / ${fmtTime(clip.duration)}`;
    const im = new Image();
    const url = URL.createObjectURL(await res.blob());
    im.src = url;
    await im.decode().catch(() => toast('Frame kon niet geladen worden', true));
    URL.revokeObjectURL(url);
    img = im;
    if (frameCanvas.width !== im.naturalWidth || frameCanvas.height !== im.naturalHeight) {
      frameCanvas.width = im.naturalWidth; frameCanvas.height = im.naturalHeight;
      view = { s: 1, ox: 0, oy: 0 };
    }
    redraw();  // punten meteen tonen; de gele voorspelling kan even duren
    const forT = t;
    const pred = clip.status === 'klaar' ? await api(`/clips/${clip.id}/predict?t=${t}`).catch(() => []) : [];
    if (forT === t) { predicted = pred; drawFrame(); }
  }

  async function loadKeyframes() {
    keyframes = await api(`/clips/${clip.id}/keyframes`);
    const manual = keyframes.filter(k => !k.auto), autos = keyframes.filter(k => k.auto);
    const rows = manual.map(kf => h('div', { className: 'list-item', onclick: () => {
      t = kf.t; slider.value = t; pairs = kf.points.map(p => ({ ...p })); editingId = kf.id; proposal = null; loadFrame();
    } },
      h('b', {}, fmtTime(kf.t)), h('span', { style: { flex: 1 } }, `${kf.points.length} punten`),
      kf.error_px != null ? h('span', { className: `badge ${kf.error_px > (kf.bad_px || 12) ? 'err' : 'ok'}`, title: `≈ ${kf.error_m} m op het veld` }, `${kf.error_px} px`)
        : kf.error_m != null ? h('span', { className: 'badge' }, `${kf.error_m} m`)
        : kf.problem ? h('span', { className: 'badge err', title: kf.problem }, '⚠ klopt niet') : null,
      editingId === kf.id ? h('span', { className: 'badge busy' }, 'bewerken') : null,
      h('button', { className: 'danger', onclick: async e => {
        e.stopPropagation();
        await api(`/keyframes/${kf.id}`, { method: 'DELETE' });
        if (editingId === kf.id) editingId = null;
        loadKeyframes();
      } }, '×')));
    kfList.replaceChildren(...(rows.length ? rows : [h('div', { className: 'muted small' },
      'Nog geen ijkmomenten. Kalibreer er één; daarna stelt de app de rest van de video automatisch bij.')]), autoPanel(autos, manual),
      trainPanel(manual) || '');
  }

  // Klopt alles? Dan mag de app van deze video leren (pas trainen als jij dat kiest, bij Trainen).
  function trainPanel(manual) {
    if (clip.status !== 'klaar' || !manual.length) return null;
    const on = !!clip.train_ok;
    return h('div', { className: 'train-box' },
      h('div', { className: 'row', style: { justifyContent: 'space-between' } },
        h('div', {}, h('b', {}, on ? '🧠 Klaargezet om van te leren' : '🧠 Laat de app hiervan leren'),
          h('div', { className: 'small muted' }, on
            ? 'Bij de volgende training leert de app van deze video: het veld, de spelers en de bal (ook wat jij hebt aangewezen).'
            : 'Klopt de kalibratie in de hele video (gele lijnen op de witte lijnen) en heb je de bal en toeschouwers waar nodig verbeterd? Zet de video dan klaar. Het trainen zelf start pas als jij dat kiest.')),
        h('button', { className: on ? '' : 'primary', onclick: async () => {
          const c = await api(`/clips/${clip.id}`, { method: 'PATCH', json: { train_ok: !on } });
          Object.assign(clip, c);
          toast(on ? 'Niet meer klaargezet' : 'Klaargezet. Start het trainen wanneer het jou uitkomt via 🧠 Trainen (rechtsboven).');
          loadKeyframes();
        } }, on ? 'Toch niet' : '✓ Gebruik voor training')));
  }

  // Automatisch bijstellen: status, aantal automatische ijkmomenten, opnieuw/wissen
  let calibTimer;
  function autoPanel(autos, manual) {
    const st = clip.calib_status, busy = st === 'wachtrij' || st === 'bezig';
    const open = autos.filter(k => !k.accepted).length;
    const fmtScore = k => k.score?.precision != null ? `${Math.round(100 * k.score.precision)}%` : '';
    const jump = k => { t = k.t; slider.value = t; loadFrame({ warp: true }); loadKeyframes(); };
    const review = autos.length && !busy ? h('div', {},
      h('div', { className: 'small muted', style: { margin: '6px 0' } },
        'Controleer ze gerust: klik op een tijd om dat moment te zien. De gele stippellijnen tonen hoe het veld daar ligt. ',
        'Klopt het? Dan ✓. Klopt het niet? ✗ verwijdert hem, ✎ zet de punten klaar om zelf bij te stellen.'),
      h('div', { className: 'list', style: { maxHeight: '220px', overflowY: 'auto' } }, autos.map(k => h('div', {
        className: 'list-item', style: Math.abs(k.t - t) < 0.05 ? { outline: '2px solid var(--accent, #0a84ff)' } : {}, onclick: () => jump(k) },
        h('b', {}, fmtTime(k.t)),
        h('span', { style: { flex: 1 }, className: 'small muted', title: 'Deel van de gevonden witte lijnen dat op de veldtekening valt' }, fmtScore(k)),
        k.accepted ? h('span', { className: 'badge ok' }, '✓ goed') : null,
        k.accepted ? null : h('button', { title: 'Klopt', onclick: async e => {
          e.stopPropagation(); await api(`/clips/${clip.id}/autocalib/accept`, { json: { ids: [k.id] } }); loadKeyframes();
        } }, '✓'),
        h('button', { title: 'Zelf bijstellen', onclick: async e => {
          e.stopPropagation();
          t = k.t; slider.value = t; pairs = []; editingId = null; proposal = null;
          await loadFrame(); usePredicted();
        } }, '✎'),
        h('button', { className: 'danger', title: 'Klopt niet: verwijderen', onclick: async e => {
          e.stopPropagation(); await api(`/keyframes/${k.id}`, { method: 'DELETE' }); loadKeyframes();
          predicted = await api(`/clips/${clip.id}/predict?t=${t}`); drawFrame();
        } }, '✗')))),
      open ? h('div', { className: 'row', style: { marginTop: '6px' } },
        h('button', { className: 'primary', onclick: async () => {
          await api(`/clips/${clip.id}/autocalib/accept`, { json: {} }); toast(`${open} automatische ijkmomenten goedgekeurd`); loadKeyframes();
        } }, `✓ Alles accepteren (${open})`)) : null) : null;
    const box = h('div', { style: { marginTop: '10px', paddingTop: '8px', borderTop: '1px solid var(--border)' } },
      h('div', { className: 'row', style: { justifyContent: 'space-between' } },
        h('b', {}, '🤖 Automatisch bijgesteld'),
        h('span', { className: `badge ${busy ? 'busy' : st === 'fout' ? 'err' : autos.length ? 'ok' : ''}` },
          busy ? `bezig ${Math.round(100 * (clip.calib_progress || 0))}%` : `${autos.length} ijkmomenten`)),
      h('div', { className: 'small muted', style: { margin: '4px 0' } },
        st === 'fout' ? clip.calib_message
          : busy ? 'De app zoekt elke seconde de witte lijnen en legt het veld er opnieuw op.'
          : clip.status !== 'klaar' ? 'Start na de analyse vanzelf.'
          : !manual.length ? 'Start vanzelf zodra je één ijkmoment hebt opgeslagen.'
          : (clip.calib_message || '') + (hasCam() ? '' : ' Tip: stel in waar je stond; dan werkt het bijstellen veel nauwkeuriger.')),
      review,
      manual.length && clip.status === 'klaar' ? h('div', { className: 'row', style: { marginTop: '6px' } },
        h('button', { disabled: busy, onclick: async () => {
          await api(`/clips/${clip.id}/autocalib`, { method: 'POST' }); refreshClip();
        }, title: 'Goedgekeurde ijkmomenten blijven staan' }, 'Opnieuw bijstellen'),
        autos.length ? h('button', { disabled: busy, onclick: async () => {
          if (!confirm('Alle automatische ijkmomenten wissen, ook de goedgekeurde?')) return;
          await api(`/clips/${clip.id}/autocalib`, { method: 'DELETE' }); await refreshClip(); loadKeyframes();
        } }, 'Wis automatische') : null) : null);
    clearTimeout(calibTimer);
    if (busy) calibTimer = setTimeout(refreshClip, 2000);
    return box;
  }

  async function refreshClip() {
    const m = await api(`/matches/${ctx.match.id}`);
    const fresh = m.clips.find(c => c.id === clip.id);
    if (!fresh) return;
    const wasBusy = clip.calib_status === 'wachtrij' || clip.calib_status === 'bezig';
    Object.assign(clip, fresh);
    const nowBusy = clip.calib_status === 'wachtrij' || clip.calib_status === 'bezig';
    await loadKeyframes();
    if (wasBusy && !nowBusy) ctx.refreshSteps?.();
    if (wasBusy && !nowBusy && clip.status === 'klaar') {  // klaar: voorspelling (geel) bijwerken
      predicted = await api(`/clips/${clip.id}/predict?t=${t}`);
      drawFrame();
    }
  }

  async function save() {
    const info = needInfo();
    if (!info.ok) return toast(`Nog niet genoeg: ${info.hint}`);
    const res = await api(`/clips/${clip.id}/keyframes`, { json: { id: editingId, t, points: pairs } });
    editingId = res.id; proposal = null; drawProposal();
    clip.cam_yaw = null;  // bewaard bij dit ijkmoment
    ctx.refreshSteps?.();
    t = res.t;
    toast(`Ijkmoment opgeslagen (afwijking ${res.error_px != null ? res.error_px + ' px' : res.error_m + ' m'})${clip.status === 'klaar' ? ' – de app stelt nu de rest van de video automatisch bij' : ''}`);
    setTimeout(refreshClip, 500);
    await loadKeyframes(); stations.refresh();
    if (wiz) return wiz.onSaved(clip);
    predicted = clip.status === 'klaar' ? await api(`/clips/${clip.id}/predict?t=${t}`) : [];
    drawFrame();
  }

  function usePredicted() {
    if (!predicted.length) return toast(clip.status === 'klaar' ? 'Nog geen voorspelling: sla eerst een ijkmoment op' : 'Voorspelling kan pas na de analyse');
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
    debounce = setTimeout(() => { loadFrame({ warp: true }); loadKeyframes(); }, 250);
  });

  async function loadClip() {
    placingCam = false; drawCamPanel();
    slider.max = clip.duration || 0; t = 0; slider.value = 0; pairs = []; editingId = null; proposal = null; frameT = null;
    view = { s: 1, ox: 0, oy: 0 };
    await loadKeyframes();
    const manual = keyframes.filter(k => !k.auto);
    if (manual.length) { const kf = manual[0]; t = kf.t; slider.value = t; pairs = kf.points.map(p => ({ ...p })); editingId = kf.id; }
    await loadFrame();
    // nog niets gekalibreerd: de app doet zelf een voorstel (in de wizard is dat al geprobeerd)
    if (!manual.length && hasCam() && !wiz?.noAutoPropose) runPropose();
  }

  // --- automatisch voorstel ---------------------------------------------------------------
  async function runPropose() {
    if (proposing) return;
    if (!hasCam()) return toast('Stel eerst in waar je stond (📍 of via GPS); dan kan de app het veld zelf zoeken');
    proposing = true; drawProposal();
    try {
      const r = await api(`/clips/${clip.id}/propose?t=${t.toFixed(3)}`);
      if (Math.abs(r.t - t) > 0.05) return;  // intussen naar een ander moment gegaan
      if (!r.ok) { proposal = { failed: r.message }; return; }
      pairs = r.points.map(p => ({ ...p, img: [...p.img] }));
      editingId = null; pendingImg = null; pendingPitch = null;
      proposal = { info: r.info };
    } catch { proposal = null; } finally { proposing = false; redraw(); }
  }

  function drawProposal() {
    drawCoach();
    saveBtn.textContent = proposal && !proposal.failed ? '✓ Klopt – opslaan' : 'Ijkmoment opslaan';
    const manualCount = keyframes.filter(k => !k.auto).length;
    const btn = h('button', { disabled: proposing || !hasCam(), title: hasCam() ? '' : 'Stel eerst in waar je stond',
      onclick: runPropose }, proposing ? '⏳ Veld zoeken… (± 10 s)' : '🤖 Zoek het veld automatisch');
    let body;
    if (proposing) body = h('div', { className: 'small muted' }, 'De app zoekt de witte lijnen en draait in gedachten rond vanaf jouw plek tot de veldtekening erop past.');
    else if (proposal?.failed) body = h('div', { className: 'small' }, proposal.failed);
    else if (proposal) body = h('div', { className: 'small' },
      h('b', {}, 'Voorstel van de app. '), 'Vallen de witte lijnen op het veld? Sleep punten bij waar nodig (zoom gerust in) en klik ',
      h('b', {}, '✓ Klopt'), '. Daarna stelt de app de rest van de video zelf bij.',
      proposal.info.ambiguous
        ? h('div', { style: { marginTop: '4px' } }, h('span', { className: 'badge err' }, 'twijfel'),
            ' Er zijn hier weinig lijnen te zien, dus ook een andere stand past bijna even goed. Controleer extra goed, ',
            'of schuif naar een moment met meer lijnen in beeld (16-meter, middenlijn, cirkel) en zoek opnieuw.')
        : h('div', { className: 'muted' }, `Pasvorm: ${Math.round(100 * proposal.info.precision)}% van de gevonden lijnen valt op het model.`));
    else if (!hasCam()) body = manualCount ? null : h('div', { className: 'small muted' },
      'Tip: stel hieronder in waar je stond. Dan kan de app het veld zelf zoeken en hoef je alleen te controleren.');
    else body = manualCount ? null : h('div', { className: 'small muted' }, 'Laat de app het veld zoeken, of klik zelf punten aan.');
    proposeBox.replaceChildren(h('div', { className: proposal && !proposal.failed ? 'hint' : '', style: { margin: '6px 0' } },
      ...[h('div', { className: 'row' }, btn), body].filter(Boolean)));
  }
  await loadClip();
  const onResize = () => { pv.resize(); drawPitch(); };
  window.addEventListener('resize', onResize);
  return () => {
    stations.cleanup();
    clearTimeout(calibTimer); window.removeEventListener('mouseup', onUp); window.removeEventListener('mousemove', onPan);
    window.removeEventListener('resize', onResize);
  };
}
