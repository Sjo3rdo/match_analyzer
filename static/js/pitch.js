// Het veld tekenen op een canvas. Meters -> pixels met een eenvoudige schaal.
export const L = 105, W = 68;

export class PitchView {
  constructor(canvas, { margin = 4 } = {}) {
    this.canvas = canvas;
    canvas.style.width = '100%';
    this.margin = margin;
    this.ctx = canvas.getContext('2d');
    this.resize();
  }
  resize(width) {
    const w = width || this.canvas.clientWidth || this.canvas.parentElement?.clientWidth || 600;
    const dpr = window.devicePixelRatio || 1;
    this.cssW = w;
    this.cssH = w * (W + 2 * this.margin) / (L + 2 * this.margin);
    this.canvas.width = Math.round(this.cssW * dpr);
    this.canvas.height = Math.round(this.cssH * dpr);
    this.canvas.style.height = this.cssH + 'px';
    this.scale = this.cssW / (L + 2 * this.margin);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  toPx(x, y) { return [(x + this.margin) * this.scale, (y + this.margin) * this.scale]; }
  toM(px, py) { return [px / this.scale - this.margin, py / this.scale - this.margin]; }

  draw() {
    const c = this.ctx, s = this.scale;
    const css = getComputedStyle(document.documentElement);
    c.fillStyle = css.getPropertyValue('--pitch').trim() || '#2e7d4a';
    c.fillRect(0, 0, this.cssW, this.cssH);
    c.strokeStyle = 'rgba(255,255,255,.85)';
    c.lineWidth = 1.5;
    const rect = (x, y, w, hh) => { const [px, py] = this.toPx(x, y); c.strokeRect(px, py, w * s, hh * s); };
    const line = (x1, y1, x2, y2) => { c.beginPath(); c.moveTo(...this.toPx(x1, y1)); c.lineTo(...this.toPx(x2, y2)); c.stroke(); };
    const arc = (x, y, r, a0 = 0, a1 = 2 * Math.PI) => { c.beginPath(); c.arc(...this.toPx(x, y), r * s, a0, a1); c.stroke(); };
    const spot = (x, y) => { c.beginPath(); c.arc(...this.toPx(x, y), 2, 0, 2 * Math.PI); c.fillStyle = '#fff'; c.fill(); };
    rect(0, 0, L, W);
    line(L / 2, 0, L / 2, W);
    arc(L / 2, W / 2, 9.15); spot(L / 2, W / 2);
    for (const [x0, sgn] of [[0, 1], [L, -1]]) {
      rect(Math.min(x0, x0 + sgn * 16.5), W / 2 - 20.16, 16.5, 40.32);
      rect(Math.min(x0, x0 + sgn * 5.5), W / 2 - 9.16, 5.5, 18.32);
      spot(x0 + sgn * 11, W / 2);
      const a = Math.acos(5.5 / 9.15);
      if (sgn > 0) arc(11, W / 2, 9.15, -a, a); else arc(L - 11, W / 2, 9.15, Math.PI - a, Math.PI + a);
      rect(Math.min(x0, x0 - sgn * 1.5), W / 2 - 3.66, 1.5, 7.32);
    }
  }

  dot(x, y, color, r = 5, label = null) {
    const c = this.ctx, [px, py] = this.toPx(x, y);
    c.beginPath(); c.arc(px, py, r, 0, 2 * Math.PI);
    c.fillStyle = color; c.fill();
    c.lineWidth = 1.2; c.strokeStyle = 'rgba(0,0,0,.6)'; c.stroke();
    if (label) {
      c.font = `600 ${Math.max(9, r * 1.6)}px sans-serif`;
      c.fillStyle = '#fff'; c.textAlign = 'center'; c.textBaseline = 'middle';
      c.fillText(label, px, py + 0.5);
    }
  }

  heat(grid, color = [255, 80, 0]) {
    // grid: rijen = y, kolommen = x (seconden per cel)
    const ny = grid.length, nx = grid[0].length;
    const max = Math.max(...grid.flat(), 1e-9);
    const off = document.createElement('canvas');
    off.width = nx; off.height = ny;
    const oc = off.getContext('2d'), img = oc.createImageData(nx, ny);
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const v = Math.sqrt(grid[j][i] / max), k = 4 * (j * nx + i);
      img.data.set([color[0], color[1] * (1 - v) + 20 * v, color[2], Math.round(220 * v)], k);
    }
    oc.putImageData(img, 0, 0);
    const [x0, y0] = this.toPx(0, 0);
    this.ctx.save();
    this.ctx.imageSmoothingEnabled = true;
    this.ctx.filter = 'blur(6px)';
    this.ctx.drawImage(off, x0, y0, L * this.scale, W * this.scale);
    this.ctx.restore();
  }
}

// Lijnen van het veld als reeksen punten (meters), voor de controle-overlay bij kalibratie.
export function pitchPolylines() {
  const lines = [];
  // Rechte stukken opdelen in stappen van 1 m: ligt een uiteinde achter de camera, dan wordt het
  // zichtbare deel toch getekend (anders valt het hele stuk weg).
  const seg = (pts) => {
    const dense = [pts[0]];
    for (let i = 1; i < pts.length; i++) {
      const [x0, y0] = pts[i - 1], [x1, y1] = pts[i];
      const n = Math.max(1, Math.ceil(Math.hypot(x1 - x0, y1 - y0)));
      for (let k = 1; k <= n; k++) dense.push([x0 + (x1 - x0) * k / n, y0 + (y1 - y0) * k / n]);
    }
    lines.push(dense);
  };
  const rect = (x, y, w, hh) => seg([[x, y], [x + w, y], [x + w, y + hh], [x, y + hh], [x, y]]);
  const circle = (cx, cy, r, a0 = 0, a1 = 2 * Math.PI, n = 48) => {
    const pts = [];
    for (let i = 0; i <= n; i++) { const a = a0 + (a1 - a0) * i / n; pts.push([cx + r * Math.cos(a), cy + r * Math.sin(a)]); }
    seg(pts);
  };
  rect(0, 0, L, W);
  seg([[L / 2, 0], [L / 2, W]]);
  circle(L / 2, W / 2, 9.15);
  rect(0, W / 2 - 20.16, 16.5, 40.32); rect(L - 16.5, W / 2 - 20.16, 16.5, 40.32);
  rect(0, W / 2 - 9.16, 5.5, 18.32); rect(L - 5.5, W / 2 - 9.16, 5.5, 18.32);
  const a = Math.acos(5.5 / 9.15);
  circle(11, W / 2, 9.15, -a, a, 16); circle(L - 11, W / 2, 9.15, Math.PI - a, Math.PI + a, 16);
  return lines;
}
