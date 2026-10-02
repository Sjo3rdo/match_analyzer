// Kleine hulpfuncties: API-aanroepen, DOM bouwen, meldingen.
export async function api(path, opts = {}) {
  const init = { ...opts };
  if (opts.json !== undefined) {
    init.method = init.method || 'POST';
    init.headers = { 'Content-Type': 'application/json' };
    init.body = JSON.stringify(opts.json);
  }
  const res = await fetch('/api' + path, init);
  if (!res.ok) {
    let msg = res.statusText;
    try { msg = (await res.json()).detail || msg; } catch {}
    toast(msg, true);
    throw new Error(msg);
  }
  return res.headers.get('content-type')?.includes('json') ? res.json() : res;
}

export function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === undefined || v === null || v === false) continue;
    if (k.startsWith('on')) el.addEventListener(k.slice(2).toLowerCase(), v);
    else if (k === 'style' && typeof v === 'object') Object.assign(el.style, v);
    else if (k in el && k !== 'list') el[k] = v;
    else el.setAttribute(k, v);
  }
  for (const c of children.flat()) {
    if (c === null || c === undefined || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

let toastTimer;
export function toast(msg, error = false) {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.style.background = error ? 'var(--danger)' : '';
  el.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove('show'), 3500);
}

export function fmtTime(s) {
  if (s == null || isNaN(s)) return '–';
  const m = Math.floor(s / 60), sec = Math.floor(s % 60);
  return `${m}:${String(sec).padStart(2, '0')}`;
}

export function matchMinute(clip, t) {
  return `${Math.floor((clip.start_minute || 0) + t / 60)}'`;
}

export const TEAM_COLORS = ['#2f6fdb', '#e0472f', '#9a9a9a'];
export function teamName(match, team) {
  if (team === 0) return match.team0_name;
  if (team === 1) return match.team1_name;
  if (team === 2) return 'Overig (scheids/keeper)';
  return 'Onbekend';
}

export function playerLabel(p) {
  return p ? `${p.number ? '#' + p.number + ' ' : ''}${p.name}` : '';
}

export function select(options, value, onchange, attrs = {}) {
  const s = h('select', { ...attrs, onchange: e => onchange(e.target.value) },
    options.map(([v, label]) => h('option', { value: v, selected: String(v) === String(value ?? '') }, label)));
  return s;
}
