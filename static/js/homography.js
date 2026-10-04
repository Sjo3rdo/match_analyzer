// Homografie (3x3) uit >= 4 puntparen met kleinste kwadraten (h33 = 1).
export function fitHomography(src, dst) {
  const n = src.length;
  if (n < 4) return null;
  // Normaliseren voor numerieke stabiliteit
  const norm = pts => {
    const cx = pts.reduce((s, p) => s + p[0], 0) / pts.length, cy = pts.reduce((s, p) => s + p[1], 0) / pts.length;
    const d = pts.reduce((s, p) => s + Math.hypot(p[0] - cx, p[1] - cy), 0) / pts.length || 1;
    const k = Math.SQRT2 / d;
    return [[k, 0, -k * cx], [0, k, -k * cy], [0, 0, 1]];
  };
  const Ts = norm(src), Td = norm(dst);
  const s = src.map(p => apply(Ts, p)), d = dst.map(p => apply(Td, p));
  const A = [], b = [];
  for (let i = 0; i < n; i++) {
    const [x, y] = s[i], [u, v] = d[i];
    A.push([x, y, 1, 0, 0, 0, -u * x, -u * y]); b.push(u);
    A.push([0, 0, 0, x, y, 1, -v * x, -v * y]); b.push(v);
  }
  const AtA = Array.from({ length: 8 }, (_, i) => Array.from({ length: 8 }, (_, j) => A.reduce((acc, r) => acc + r[i] * r[j], 0)));
  const Atb = Array.from({ length: 8 }, (_, i) => A.reduce((acc, r, k) => acc + r[i] * b[k], 0));
  const hv = solve(AtA, Atb);
  if (!hv) return null;
  const Hn = [[hv[0], hv[1], hv[2]], [hv[3], hv[4], hv[5]], [hv[6], hv[7], 1]];
  const H = mul(inv(Td), mul(Hn, Ts));
  return H.map(r => r.map(v => v / H[2][2]));
}

export function apply(H, [x, y]) {
  const w = H[2][0] * x + H[2][1] * y + H[2][2];
  return [(H[0][0] * x + H[0][1] * y + H[0][2]) / w, (H[1][0] * x + H[1][1] * y + H[1][2]) / w, w];
}

function mul(A, B) {
  return A.map((r, i) => B[0].map((_, j) => r.reduce((s, _, k) => s + A[i][k] * B[k][j], 0)));
}

export function inv(m) {
  const [[a, b, c], [d, e, f], [g, hh, i]] = m;
  const A = e * i - f * hh, B = -(d * i - f * g), C = d * hh - e * g;
  const det = a * A + b * B + c * C;
  return [[A / det, -(b * i - c * hh) / det, (b * f - c * e) / det],
          [B / det, (a * i - c * g) / det, -(a * f - c * d) / det],
          [C / det, -(a * hh - b * g) / det, (a * e - b * d) / det]];
}

function solve(M, v) {
  const n = v.length, A = M.map((r, i) => [...r, v[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(A[r][c]) > Math.abs(A[p][c])) p = r;
    if (Math.abs(A[p][c]) < 1e-12) return null;
    [A[c], A[p]] = [A[p], A[c]];
    for (let r = 0; r < n; r++) if (r !== c) {
      const f = A[r][c] / A[c][c];
      for (let k = c; k <= n; k++) A[r][k] -= f * A[c][k];
    }
  }
  return A.map((r, i) => r[n] / r[i]);
}

// --- kalibratie met punten én punten-op-een-lijn (beeld -> veld) ----------------------------
// pairs: [{img: [x, y], pitch: [X, Y]}] of [{img: [x, y], line: [[X1, Y1], [X2, Y2]]}]
function lineCoeffs([[x1, y1], [x2, y2]]) {
  const a = y1 - y2, b = x2 - x1, c = x1 * y2 - x2 * y1, n = Math.hypot(a, b) || 1;
  return [a / n, b / n, c / n];
}

// Hoeveel informatie: punt = 2, elke lijn hoogstens 2 (meer punten op dezelfde lijn helpen niet)
// Alleen punten en lijnen op de grond: punten in de lucht (bovenkant paal, lat) kan alleen het
// cameramodel op de server gebruiken.
const ground = pairs => pairs.filter(p => p.pitch || p.line);

export function calibrationInfo(pairs) {
  pairs = ground(pairs);
  const lines = new Map();
  let points = 0;
  for (const p of pairs) {
    if (p.pitch) points++;
    else { const k = JSON.stringify(p.line); lines.set(k, Math.min(2, (lines.get(k) || 0) + 1)); }
  }
  const dof = 2 * points + [...lines.values()].reduce((s, v) => s + v, 0);
  const ok = dof >= 8 && !(points === 2 && lines.size === 2 && dof === 8);
  let hint = '';
  if (dof < 8) hint = `nog ${8 - dof} nodig (punt telt 2, lijn telt max 2)`;
  else if (!ok) hint = '2 punten + 2 lijnen ligt niet vast: voeg nog een punt of lijn toe';
  return { dof, ok, hint, points, lines: lines.size };
}

export function fitCalibration(pairs) {
  pairs = ground(pairs);
  if (!calibrationInfo(pairs).ok) return null;
  const imgs = pairs.map(p => p.img);
  const world = pairs.flatMap(p => (p.pitch ? [p.pitch] : p.line));
  const norm = pts => {
    const cx = pts.reduce((s, p) => s + p[0], 0) / pts.length, cy = pts.reduce((s, p) => s + p[1], 0) / pts.length;
    const d = pts.reduce((s, p) => s + Math.hypot(p[0] - cx, p[1] - cy), 0) / pts.length || 1;
    const k = Math.SQRT2 / d;
    return [[k, 0, -k * cx], [0, k, -k * cy], [0, 0, 1]];
  };
  const Ti = norm(imgs), Tp = norm(world), TpInvT = transpose(inv(Tp));
  const A = [], b = [];
  pairs.forEach(p => {
    const [x, y] = apply(Ti, p.img);
    if (p.pitch) {
      const [X, Y] = apply(Tp, p.pitch);
      A.push([x, y, 1, 0, 0, 0, -X * x, -X * y]); b.push(X);
      A.push([0, 0, 0, x, y, 1, -Y * x, -Y * y]); b.push(Y);
    } else {
      const l = lineCoeffs(p.line);
      let [la, lb, lc] = TpInvT.map(r => r[0] * l[0] + r[1] * l[1] + r[2] * l[2]);
      const n = Math.hypot(la, lb) || 1; la /= n; lb /= n; lc /= n;
      A.push([la * x, la * y, la, lb * x, lb * y, lb, lc * x, lc * y]); b.push(-lc);
    }
  });
  const AtA = Array.from({ length: 8 }, (_, i) => Array.from({ length: 8 }, (_, j) => A.reduce((s, r) => s + r[i] * r[j], 0)));
  const Atb = Array.from({ length: 8 }, (_, i) => A.reduce((s, r, k) => s + r[i] * b[k], 0));
  const h = solve(AtA, Atb);
  if (!h) return null;
  const Gn = [[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], 1]];
  const G = mul(inv(Tp), mul(Gn, Ti));
  if (!G.flat().every(Number.isFinite)) return null;
  // Schalen zonder het teken om te draaien: het teken van w zegt of iets vóór of achter de camera
  // ligt. Aangeklikte punten liggen op het veld, vóór de camera: die moeten w > 0 krijgen.
  const d = Math.abs(G[2][2]) > 1e-12 ? Math.abs(G[2][2]) : 1;
  let Gs = G.map(r => r.map(v => v / d));
  const votes = pairs.reduce((s, p) => s + Math.sign(Gs[2][0] * p.img[0] + Gs[2][1] * p.img[1] + Gs[2][2]), 0);
  if (votes < 0) Gs = Gs.map(r => r.map(v => -v));
  // Een echte camera ziet het veld nooit gespiegeld (determinant > 0). Gespiegeld: laat de server
  // het oplossen (die kiest de echte kant, of zegt wat er niet klopt).
  const [[g0, g1, g2], [g3, g4, g5], [g6, g7, g8]] = Gs;
  if (g0 * (g4 * g8 - g5 * g7) - g1 * (g3 * g8 - g5 * g6) + g2 * (g3 * g7 - g4 * g6) <= 0) return null;
  return Gs;
}

// Gemiddelde fout in meters (punt: afstand, lijnpunt: afstand tot de lijn)
export function calibrationError(G, pairs) {
  pairs = ground(pairs);
  const e = pairs.map(p => {
    const q = apply(G, p.img);
    if (p.pitch) return Math.hypot(q[0] - p.pitch[0], q[1] - p.pitch[1]);
    const [a, b, c] = lineCoeffs(p.line);
    return Math.abs(a * q[0] + b * q[1] + c);
  });
  return e.reduce((s, v) => s + v, 0) / e.length;
}

function transpose(m) { return m[0].map((_, j) => m.map(r => r[j])); }
