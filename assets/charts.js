/* Clutch IQ — tiny SVG chart kit (line, bars, matrix) with hover tooltips.
   Recessive axes, 2px lines, direct end-labels for <= 4 series + legend. */
(function (root) {
  'use strict';
  const SERIES = ['#0aa5c4', '#c43fb0', '#2f9e5e', '#7f6ff0', '#c98500', '#3d78e0', '#e0574a'];
  const tipEl = () => document.getElementById('tip');
  function showTip(html, ev) {
    const t = tipEl(); if (!t) return; t.innerHTML = html; t.style.display = 'block';
    const pad = 14; let x = ev.clientX + pad, y = ev.clientY + pad;
    const r = t.getBoundingClientRect();
    if (x + r.width > window.innerWidth - 8) x = ev.clientX - r.width - pad;
    if (y + r.height > window.innerHeight - 8) y = ev.clientY - r.height - pad;
    t.style.left = x + 'px'; t.style.top = y + 'px';
  }
  function hideTip() { const t = tipEl(); if (t) t.style.display = 'none'; }
  const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

  function niceTicks(lo, hi, n = 5) {
    if (!isFinite(lo) || !isFinite(hi)) return [0, 1];
    if (hi === lo) { hi = lo + 1; lo = lo - 1; }
    const span = hi - lo; const step0 = span / n; const mag = Math.pow(10, Math.floor(Math.log10(step0)));
    const err = step0 / mag; const step = (err >= 7.5 ? 10 : err >= 3.5 ? 5 : err >= 1.5 ? 2 : 1) * mag;
    const out = []; for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(+v.toPrecision(12));
    return out;
  }
  function fmtNum(v) {
    if (v === null || v === undefined || !isFinite(v)) return '—';
    const a = Math.abs(v);
    if (a >= 10000) return (v / 1000).toFixed(0) + 'k';
    if (a >= 1000) return (v / 1000).toFixed(1) + 'k';
    if (a >= 100) return v.toFixed(0);
    if (a >= 10) return v.toFixed(1);
    if (a >= 1) return v.toFixed(2);
    if (a === 0) return '0';
    if (a >= 0.01) return v.toFixed(3);
    return v.toExponential(1);
  }

  /** Multi-series line chart. opts.series: [{name, color, points:[[x,y]], dash}] */
  function line(el, opts) {
    const W = Math.max(300, Math.round(el.clientWidth || 640)), H = opts.height || 260, m = { l: 52, r: opts.series.length <= 4 && opts.directLabels !== false ? 92 : 18, t: 14, b: 38 };
    const all = opts.series.flatMap(s => s.points.filter(p => p[1] !== null && p[1] !== undefined && isFinite(p[1])));
    if (!all.length) { el.innerHTML = '<div class="empty">No data yet.</div>'; return; }
    const lx = opts.logX, ly = opts.logY;
    const tx = v => (lx ? Math.log10(Math.max(v, 1e-12)) : v), ty = v => (ly ? Math.log10(Math.max(v, 1e-14)) : v);
    let x0 = Math.min(...all.map(p => tx(p[0]))), x1 = Math.max(...all.map(p => tx(p[0])));
    let ys = all.map(p => ty(p[1])); (opts.refLines || []).forEach(r => ys.push(ty(r.y)));
    let y0 = opts.yMin !== undefined ? ty(opts.yMin) : Math.min(...ys), y1 = opts.yMax !== undefined ? ty(opts.yMax) : Math.max(...ys);
    if (!ly && opts.yMin === undefined && y0 > 0 && opts.zero) y0 = 0;
    if (y1 === y0) { y1 += 1; y0 -= 1; }
    const padY = (y1 - y0) * 0.06; if (opts.yMin === undefined) y0 -= padY; if (opts.yMax === undefined) y1 += padY;
    if (x1 === x0) x1 = x0 + 1;
    const X = v => m.l + (tx(v) - x0) / (x1 - x0) * (W - m.l - m.r), Y = v => H - m.b - (ty(v) - y0) / (y1 - y0) * (H - m.t - m.b);
    let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(opts.aria || opts.yLabel || 'chart')}">`;
    // grid + axes
    const yt = ly ? logTicks(y0, y1) : niceTicks(y0, y1, 5).filter(v => v >= y0 && v <= y1);
    svg += '<g class="grid">' + yt.map(v => `<line x1="${m.l}" x2="${W - m.r}" y1="${Y(ly ? Math.pow(10, v) : v)}" y2="${Y(ly ? Math.pow(10, v) : v)}"/>`).join('') + '</g>';
    svg += '<g class="axis">' + yt.map(v => `<text x="${m.l - 6}" y="${Y(ly ? Math.pow(10, v) : v) + 3}" text-anchor="end">${ly ? fmtNum(Math.pow(10, v)) : (opts.fmtY || fmtNum)(v)}</text>`).join('');
    let xt = lx ? logTicks(x0, x1).map(v => Math.pow(10, v)) : niceTicks(x0, x1, 6).filter(v => v >= x0 && v <= x1);
    { const seen = new Set(); xt = xt.filter(v => { const l = (opts.fmtX || fmtNum)(v); if (seen.has(l)) return false; seen.add(l); return true; }); }
    svg += xt.map(v => `<text x="${X(v)}" y="${H - m.b + 15}" text-anchor="middle">${(opts.fmtX || fmtNum)(v)}</text>`).join('');
    svg += `<line x1="${m.l}" x2="${W - m.r}" y1="${H - m.b}" y2="${H - m.b}"/>`;
    if (opts.xLabel) svg += `<text class="ttl" x="${(m.l + W - m.r) / 2}" y="${H - 4}" text-anchor="middle">${esc(opts.xLabel)}</text>`;
    if (opts.yLabel) svg += `<text class="ttl" transform="translate(12 ${(m.t + H - m.b) / 2}) rotate(-90)" text-anchor="middle">${esc(opts.yLabel)}</text>`;
    svg += '</g>';
    (opts.refLines || []).forEach(r => { svg += `<line x1="${m.l}" x2="${W - m.r}" y1="${Y(r.y)}" y2="${Y(r.y)}" stroke="${r.color || '#a9b6c6'}" stroke-dasharray="5 4" stroke-width="1.2"/><text x="${m.l + 6}" y="${Y(r.y) - 5}" text-anchor="start" font-size="10.5" fill="${r.color || '#a9b6c6'}">${esc(r.label || '')}</text>`; });
    // bands
    opts.series.forEach((s, i) => {
      if (!s.band) return; const c = s.color || SERIES[i % SERIES.length];
      const up = s.band.map(p => `${X(p[0])},${Y(p[2])}`), dn = s.band.slice().reverse().map(p => `${X(p[0])},${Y(p[1])}`);
      svg += `<polygon points="${up.concat(dn).join(' ')}" fill="${c}" opacity=".14"/>`;
    });
    const labels = [];
    opts.series.forEach((s, i) => {
      const c = s.color || SERIES[i % SERIES.length]; const pts = s.points.filter(p => p[1] !== null && isFinite(p[1]));
      if (!pts.length) return;
      const d = pts.map((p, j) => `${j ? 'L' : 'M'}${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join('');
      svg += `<path d="${d}" fill="none" stroke="${c}" stroke-width="${s.width || 2}" stroke-linejoin="round" stroke-linecap="round" ${s.dash ? `stroke-dasharray="${s.dash}"` : ''}/>`;
      if (s.markers) pts.forEach(p => { svg += `<circle cx="${X(p[0])}" cy="${Y(p[1])}" r="3.2" fill="${c}" stroke="#0a0f16" stroke-width="1.5"/>`; });
      const last = pts[pts.length - 1]; labels.push({ y: Y(last[1]), x: X(last[0]), name: s.name, c });
    });
    if (opts.series.length <= 4 && opts.directLabels !== false) {
      labels.sort((a, b) => a.y - b.y); for (let i = 1; i < labels.length; i++) if (labels[i].y - labels[i - 1].y < 12) labels[i].y = labels[i - 1].y + 12;
      labels.forEach(l => { svg += `<text class="lab" x="${W - m.r + 6}" y="${l.y + 3.5}" fill="${l.c}">${esc(trunc(l.name, 14))}</text>`; });
    }
    svg += `<line class="xh" x1="0" x2="0" y1="${m.t}" y2="${H - m.b}" stroke="#a9b6c6" stroke-width="1" opacity="0"/>`;
    svg += `<rect class="hit" x="${m.l}" y="${m.t}" width="${W - m.l - m.r}" height="${H - m.t - m.b}" fill="transparent"/></svg>`;
    let legend = '';
    if (opts.series.length >= 2) legend = '<div class="legend" style="margin-top:6px">' + opts.series.map((s, i) => `<span><i style="background:${s.color || SERIES[i % SERIES.length]}"></i>${esc(s.name)}</span>`).join('') + '</div>';
    el.innerHTML = `<div class="chart">${svg}</div>${legend}`;
    const svgEl = el.querySelector('svg'), hit = svgEl.querySelector('.hit'), xh = svgEl.querySelector('.xh');
    hit.addEventListener('mousemove', ev => {
      const r = svgEl.getBoundingClientRect(); const vx = (ev.clientX - r.left) / r.width * W;
      const xv = x0 + (vx - m.l) / (W - m.l - m.r) * (x1 - x0);
      const rows = opts.series.map((s, i) => {
        const pts = s.points.filter(p => p[1] !== null && isFinite(p[1])); if (!pts.length) return null;
        let best = pts[0]; for (const p of pts) if (Math.abs(tx(p[0]) - xv) < Math.abs(tx(best[0]) - xv)) best = p; return { s, p: best, c: s.color || SERIES[i % SERIES.length] };
      }).filter(Boolean);
      if (!rows.length) return; xh.setAttribute('x1', X(rows[0].p[0])); xh.setAttribute('x2', X(rows[0].p[0])); xh.setAttribute('opacity', '.5');
      showTip(`<b>${esc(opts.xLabel || 'x')}: ${(opts.fmtX || fmtNum)(rows[0].p[0])}</b>` + rows.map(r => `<div class="r"><span><i class="sw" style="background:${r.c}"></i>${esc(r.s.name)}</span><span>${(opts.fmtTip || fmtNum)(r.p[1])}</span></div>`).join(''), ev);
    });
    hit.addEventListener('mouseleave', () => { hideTip(); xh.setAttribute('opacity', '0'); });
  }
  function logTicks(a, b) { const out = []; for (let e = Math.ceil(a); e <= Math.floor(b); e++) out.push(e); if (out.length < 2) return [a, b]; return out; }
  const trunc = (s, n) => (String(s).length > n ? String(s).slice(0, n - 1) + '…' : String(s));

  /** Grouped vertical bars. opts: {categories, series:[{name,color,values,err}], yLabel, fmt, height, zero} */
  function bars(el, opts) {
    const W = Math.max(300, Math.round(el.clientWidth || 640)), H = opts.height || 240, m = { l: 52, r: 12, t: 12, b: opts.rotate ? 70 : 42 };
    const cats = opts.categories, S = opts.series; const vals = S.flatMap(s => s.values.map((v, i) => v + (s.err ? s.err[i] || 0 : 0)));
    const vmin = Math.min(0, ...S.flatMap(s => s.values.map((v, i) => v - (s.err ? s.err[i] || 0 : 0))));
    let vmax = Math.max(...vals); if (vmax <= vmin) vmax = vmin + 1;
    const ticks = niceTicks(vmin, vmax, 5); if (ticks.length > 1 && ticks[ticks.length - 1] < vmax - 1e-9) ticks.push(+(ticks[ticks.length - 1] + (ticks[1] - ticks[0])).toPrecision(12)); const lo = Math.min(vmin, ticks[0]), hi = Math.max(vmax, ticks[ticks.length - 1]);
    const Y = v => H - m.b - (v - lo) / (hi - lo) * (H - m.t - m.b);
    const bw = (W - m.l - m.r) / cats.length; const inner = Math.min(46, bw * 0.78 / S.length);
    let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(opts.yLabel || 'bar chart')}"><g class="grid">${ticks.map(t => `<line x1="${m.l}" x2="${W - m.r}" y1="${Y(t)}" y2="${Y(t)}"/>`).join('')}</g><g class="axis">${ticks.map(t => `<text x="${m.l - 6}" y="${Y(t) + 3}" text-anchor="end">${(opts.fmt || fmtNum)(t)}</text>`).join('')}`;
    cats.forEach((c, i) => { const cx = m.l + bw * (i + 0.5); svg += opts.rotate ? `<text transform="translate(${cx} ${H - m.b + 10}) rotate(-35)" text-anchor="end">${esc(trunc(c, 18))}</text>` : `<text x="${cx}" y="${H - m.b + 15}" text-anchor="middle">${esc(trunc(c, Math.max(6, Math.floor(bw / 6.5))))}</text>`; });
    if (opts.yLabel) svg += `<text class="ttl" transform="translate(12 ${(m.t + H - m.b) / 2}) rotate(-90)" text-anchor="middle">${esc(opts.yLabel)}</text>`;
    svg += `<line x1="${m.l}" x2="${W - m.r}" y1="${Y(0)}" y2="${Y(0)}"/></g>`;
    cats.forEach((c, i) => {
      S.forEach((s, j) => {
        const v = s.values[i]; if (v === null || v === undefined) return; const col = (s.colors && s.colors[i]) || s.color || SERIES[j % SERIES.length];
        const x = m.l + bw * i + (bw - inner * S.length - 2 * (S.length - 1)) / 2 + j * (inner + 2); const y = Math.min(Y(v), Y(0)), h = Math.max(1, Math.abs(Y(v) - Y(0)));
        svg += `<path class="bar" data-i="${i}" data-j="${j}" d="${roundTop(x, y, inner, h, v >= 0)}" fill="${col}"/>`;
        if (s.err && s.err[i]) { const cx = x + inner / 2; svg += `<line x1="${cx}" x2="${cx}" y1="${Y(v - s.err[i])}" y2="${Y(v + s.err[i])}" stroke="#e7eef6" stroke-width="1.2" opacity=".8"/><line x1="${cx - 4}" x2="${cx + 4}" y1="${Y(v + s.err[i])}" y2="${Y(v + s.err[i])}" stroke="#e7eef6" opacity=".8"/><line x1="${cx - 4}" x2="${cx + 4}" y1="${Y(v - s.err[i])}" y2="${Y(v - s.err[i])}" stroke="#e7eef6" opacity=".8"/>`; }
        if (opts.valueLabels) svg += `<text x="${x + inner / 2}" y="${(v >= 0 ? Y(v) : Y(v) + 12) - 4}" text-anchor="middle" font-size="10" fill="#a9b6c6" font-family="IBM Plex Mono">${(opts.fmt || fmtNum)(v)}</text>`;
      });
    });
    (opts.refLines || []).forEach(r => { svg += `<line x1="${m.l}" x2="${W - m.r}" y1="${Y(r.y)}" y2="${Y(r.y)}" stroke="${r.color || '#a9b6c6'}" stroke-dasharray="5 4"/><text x="${W - m.r - 2}" y="${Y(r.y) - 4}" text-anchor="end" font-size="10.5" fill="${r.color || '#a9b6c6'}">${esc(r.label || '')}</text>`; });
    svg += '</svg>';
    const legend = S.length >= 2 ? '<div class="legend" style="margin-top:6px">' + S.map((s, j) => `<span><i style="background:${s.color || SERIES[j % SERIES.length]}"></i>${esc(s.name)}</span>`).join('') + '</div>' : '';
    el.innerHTML = `<div class="chart">${svg}</div>${legend}`;
    el.querySelectorAll('.bar').forEach(b => {
      b.addEventListener('mousemove', ev => { const i = +b.dataset.i, j = +b.dataset.j, s = S[j]; showTip(`<b>${esc(cats[i])}</b><div class="r"><span>${esc(s.name)}</span><span>${(opts.fmtTip || opts.fmt || fmtNum)(s.values[i])}${s.err && s.err[i] ? ' ± ' + fmtNum(s.err[i]) : ''}</span></div>${opts.extraTip ? opts.extraTip(i, j) : ''}`, ev); });
      b.addEventListener('mouseleave', hideTip);
    });
  }
  function roundTop(x, y, w, h, up) {
    const r = Math.min(4, w / 2, h);
    if (up) return `M${x},${y + h}V${y + r}Q${x},${y} ${x + r},${y}H${x + w - r}Q${x + w},${y} ${x + w},${y + r}V${y + h}Z`;
    return `M${x},${y}V${y + h - r}Q${x},${y + h} ${x + r},${y + h}H${x + w - r}Q${x + w},${y + h} ${x + w},${y + h - r}V${y}Z`;
  }

  /** Square matrix heatmap (e.g. policy agreement). */
  function matrix(el, opts) {
    const n = opts.rows.length, cell = 1;
    const col = v => { const t = Math.max(0, Math.min(1, (v - (opts.min || 0)) / ((opts.max || 1) - (opts.min || 0)))); return seqColor(t); };
    let h = `<div style="overflow-x:auto"><div style="display:grid;grid-template-columns:150px repeat(${opts.cols.length},minmax(36px,1fr));gap:2px;font-size:11px;min-width:${150 + 40 * opts.cols.length}px">`;
    h += '<div></div>' + opts.cols.map(c => `<div class="faint" style="text-align:center;padding:2px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${esc(c)}">${esc(trunc(c, 10))}</div>`).join('');
    opts.rows.forEach((r, i) => {
      h += `<div class="dim" style="padding:4px 6px 4px 0;text-align:right;white-space:nowrap;overflow:hidden;text-overflow:ellipsis" title="${esc(r)}">${esc(r)}</div>`;
      opts.values[i].forEach((v, j) => { const c = col(v); h += `<div class="mx" data-i="${i}" data-j="${j}" style="background:${c};color:${lum(c) > .45 ? '#03121a' : '#e7eef6'};text-align:center;padding:6px 2px;border-radius:4px;font-family:var(--f-mono)">${(opts.fmt || (x => Math.round(x * 100)))(v)}</div>`; });
    });
    el.innerHTML = h + '</div></div>';
    el.querySelectorAll('.mx').forEach(c => { c.addEventListener('mousemove', ev => { const i = +c.dataset.i, j = +c.dataset.j; showTip(`<b>${esc(opts.rows[i])} vs ${esc(opts.cols[j])}</b><div class="r"><span>${esc(opts.label || 'value')}</span><span>${(opts.fmtTip || fmtNum)(opts.values[i][j])}</span></div>`, ev); }); c.addEventListener('mouseleave', hideTip); });
  }
  // sequential cyan ramp (surface -> bright) and diverging cyan <-> coral via grey
  const RAMP = ['#0f1e27', '#0b3442', '#0a4e60', '#08697f', '#06869e', '#14a5bd', '#47c4d8', '#8fdeea'];
  function seqColor(t) { t = Math.max(0, Math.min(1, t)); const x = t * (RAMP.length - 1), i = Math.floor(x), f = x - i; return mix(RAMP[i], RAMP[Math.min(RAMP.length - 1, i + 1)], f); }
  function divColor(t) { // t in [-1,1]
    t = Math.max(-1, Math.min(1, t)); const mid = '#2b3340';
    return t >= 0 ? mix(mid, '#e0574a', t) : mix(mid, '#0aa5c4', -t);
  }
  function mix(a, b, f) { const pa = hex(a), pb = hex(b); return '#' + pa.map((v, i) => Math.round(v + (pb[i] - v) * f).toString(16).padStart(2, '0')).join(''); }
  function hex(h) { h = h.replace('#', ''); return [0, 2, 4].map(i => parseInt(h.slice(i, i + 2), 16)); }
  function lum(h) { const [r, g, b] = hex(h).map(v => v / 255); return 0.2126 * r + 0.7152 * g + 0.0722 * b; }

  root.Charts = { line, bars, matrix, showTip, hideTip, fmtNum, seqColor, divColor, lum, SERIES, esc, niceTicks };
})(window);
