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
