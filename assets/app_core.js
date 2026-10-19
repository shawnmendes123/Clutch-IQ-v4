/* Clutch IQ — app shell: engine routing, navigation, viva mode, shared UI parts. */
(function (root) {
  'use strict';
  const RL = root.ClutchRL, C = root.Charts, esc = C.esc;

  const App = root.App = {
    engine: 'browser', base: '/api', screens: {}, viva: {}, data: {}, rendered: {}, current: 'overview',
  };

  // ------------------------------------------------------------ helpers
  const $ = (sel, el = document) => el.querySelector(sel);
  const $$ = (sel, el = document) => Array.from(el.querySelectorAll(sel));
  App.$ = $; App.$$ = $$;
  App.f = (v, d = 3) => (v === null || v === undefined || (typeof v === 'number' && !isFinite(v))) ? '—' : (typeof v === 'number' ? v.toFixed(d) : String(v));
  App.pct = (v, d = 1) => (v === null || v === undefined) ? '—' : (v * 100).toFixed(d) + '%';
  App.sgn = (v, d = 3) => (v >= 0 ? '+' : '') + v.toFixed(d);
  App.toast = (msg, ms = 3200) => { const t = $('#toast'); t.textContent = msg; t.style.display = 'block'; clearTimeout(App._tt); App._tt = setTimeout(() => { t.style.display = 'none'; }, ms); };
  App.sleep = ms => new Promise(r => setTimeout(r, ms));

  const PROV = { 'REAL DATA': 'real', SIMULATED: 'sim', 'MODEL ESTIMATE': 'model', 'DERIVED FEATURE': 'derived', 'FALLBACK/PROXY': 'proxy', LOGGED: 'real' };
  App.prov = (p) => {
    const key = Object.keys(PROV).find(k => String(p).toUpperCase().startsWith(k)) || 'SIMULATED';
    const label = { real: 'Real data', sim: 'Simulated', model: 'Model estimate', derived: 'Derived feature', proxy: 'Fallback / proxy' }[PROV[key]];
    return `<span class="prov ${PROV[key]}" title="${esc(p)}">${label}</span>`;
  };
  App.act = a => `<span class="act a${a}">${['▼', '●', '▲'][a]} ${RL.ACTION_NAMES[a]}</span>`;
  App.stateChip = s => { if (s === RL.TERMINAL) return '<span class="statechip"><span>END</span></span>'; const [f, t, k] = RL.decode(s); return `<span class="statechip" title="state ${s} · key ${RL.stateKey(s)}"><span>${RL.FATIGUE_LEVELS[f]}</span><span>${RL.TILT_LEVELS[t]}</span><span>${RL.STREAK_LEVELS[k]}</span></span>`; };
  App.kpi = (v, k, d = '', cls = '') => `<div class="kpi"><div class="v ${cls}">${v}</div><div class="k">${k}</div>${d ? `<div class="d">${d}</div>` : ''}</div>`;
  App.header = (title, lede, units = []) => `<div class="sh"><div><h1>${title}</h1><p>${lede}</p></div><div class="units">${units.map(u => `<span class="unit ${u[0]}">${u[1]}</span>`).join('')}</div></div><div class="viva-card" data-viva></div>`;
  App.table = (cols, rows, opts = {}) => `<div class="tbl-wrap ${opts.scroll ? 'scroll-y' : ''}"><table class="t"><thead><tr>${cols.map(c => `<th class="${c.n ? 'n' : ''}">${c.h}</th>`).join('')}</tr></thead><tbody>${rows.map((r, i) => `<tr class="${opts.rowClass ? opts.rowClass(r, i) : ''}">${cols.map(c => `<td class="${c.n ? 'n' : ''}">${c.f ? c.f(r, i) : esc(r[c.k])}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
  App.busy = (el, on) => { if (el) el.classList.toggle('busy', on); };
  App.err = e => `<div class="note err">${esc(e && (e.detail || e.message) || e)}</div>`;

  // ------------------------------------------------------------ engine routing
  const BACKEND_ONLY = [/^\/upload/, /^\/schema/, /^\/engineer/, /^\/players\/(?!synthetic)/, /^\/matchmake/, /^\/forecast/, /^\/report/, /^\/tournament/, /^\/rl\/episodes/, /^\/rl\/mc\/real/];
  App.backendOnly = path => BACKEND_ONLY.some(r => r.test(path));
  App.api = async function (method, path, body) {
    if (App.engine === 'python' || App.backendOnly(path)) {
      if (App.engine !== 'python') { const e = new Error('This feature needs the Python backend (it works on uploaded files). Open "Backend settings" to connect.'); e.status = 0; throw e; }
      const opt = { method };
      if (body instanceof FormData) opt.body = body;
      else if (body !== undefined && method !== 'GET') { opt.body = JSON.stringify(body); opt.headers = { 'Content-Type': 'application/json' }; }
      let res;
      try { res = await fetch(App.base + path, opt); }
      catch (e) {
        if (App.backendOnly(path)) throw new Error('Backend unreachable — ' + e.message);
        App.setEngine('browser', 'Backend unreachable — switched to the in-browser engine.');
        return App.api(method, path, body);
      }
      const data = await res.json().catch(() => ({ detail: res.statusText }));
      if (!res.ok) { const e = new Error(typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail)); e.status = res.status; e.detail = e.message; throw e; }
      return data;
    }
    await App.sleep(15);   // let the spinner paint
    return RL.call(method, path, body || {});
  };
  App.setEngine = function (eng, msg) {
    App.engine = eng; const b = $('#engineBadge');
    b.classList.toggle('py', eng === 'python');
    $('#engineText').innerHTML = eng === 'python' ? `<b>Python backend</b> · ${esc(App.base)}` : '<b>Browser engine</b> · offline mirror';
    if (msg) App.toast(msg);
    document.dispatchEvent(new CustomEvent('engine-change'));
  };
  App.probe = async function (base) {
    try { const r = await fetch(base + '/health', { cache: 'no-store' }); if (!r.ok) return false; const d = await r.json(); return d && d.status === 'ok' && d.rl_endpoints; } catch (e) { return false; }
  };

  // ------------------------------------------------------------ navigation
  App.go = function (id, push = true) {
    if (!document.getElementById(id)) id = 'overview';
    $$('.screen').forEach(s => s.classList.toggle('on', s.id === id));
    $$('#nav button').forEach(b => { if (b.dataset.go === id) b.setAttribute('aria-current', 'page'); else b.removeAttribute('aria-current'); });
    App.current = id; document.body.classList.remove('nav-open');
    if (push) history.replaceState(null, '', '#' + id);
    const scr = App.screens[id];
    if (scr && !App.rendered[id]) { App.rendered[id] = true; scr.render(document.getElementById(id)); }
    else if (scr && scr.onShow) scr.onShow(document.getElementById(id));
    App.renderViva(id); window.scrollTo({ top: 0 });
  };

  // ------------------------------------------------------------ viva mode
  App.renderViva = function (id) {
    const el = document.querySelector(`#${id} [data-viva]`); const fn = App.viva[id]; if (!el || !fn) return;
    let v; try { v = fn(App.data[id] || {}); } catch (e) { v = null; }
    if (!v) { el.innerHTML = ''; return; }
    el.innerHTML = `<div class="vh"><span class="unit ${v.unitClass || 'ux'}">${esc(v.unit)}</span><strong>${esc(v.concept)}</strong>${v.prov ? App.prov(v.prov) : ''}</div>
      <dl class="viva-grid">
        <div><dt>Why it is used here</dt><dd>${v.why}</dd></div>
        <div><dt>Inputs</dt><dd>${v.inputs}</dd></div>
        <div><dt>Outputs</dt><dd>${v.outputs}</dd></div>
        <div class="wide"><dt>Mathematical formulation</dt><dd><div class="eq wrap">${v.formula}</div></dd></div>
        <div class="wide"><dt>Worked example from the current run</dt><dd>${v.example || '<span class="faint">Run the computation on this screen to populate a live example.</span>'}</dd></div>
        <div><dt>Assumptions</dt><dd>${v.assumptions}</dd></div>
        <div class="wide" style="grid-column:span 2"><dt>Where it lives in the source</dt><dd><code>${v.code}</code></dd></div>
      </dl>`;
  };
  App.setData = function (id, patch) { App.data[id] = Object.assign(App.data[id] || {}, patch); if (App.current === id) App.renderViva(id); };

  // ------------------------------------------------------------ policy map (signature component)
  const ACOL = ['#0aa5c4', '#4b5b70', '#c43fb0'];
  App.ACOL = ACOL;
  /**
   * opts: {actions?, values?, mode:'policy'|'value'|'diverging'|'count', selected, diffWith, changed:Set,
   *        onSelect(s), tip(s) -> html, vmin, vmax, labels: fn(s)->small text}
   */
  App.policyMap = function (el, opts) {
    const vals = opts.values; let vmin = opts.vmin, vmax = opts.vmax;
    if (vals && (vmin === undefined || vmax === undefined)) { const v = vals.slice(0, 48).filter(x => x !== null && isFinite(x)); vmin = vmin === undefined ? Math.min(...v) : vmin; vmax = vmax === undefined ? Math.max(...v) : vmax; }
    const color = s => {
      if (opts.mode === 'policy' || !vals) return ACOL[opts.actions[s]];
      const v = vals[s]; if (v === null || !isFinite(v)) return '#151d29';
      if (opts.mode === 'diverging') { const m = Math.max(Math.abs(vmin), Math.abs(vmax)) || 1; return C.divColor(v / m); }
      return C.seqColor(vmax > vmin ? (v - vmin) / (vmax - vmin) : 0.5);
    };
    let h = '<div class="pmap">';
    for (let f = 0; f < 4; f++) {
      h += `<div class="pblock"><h5><span>${RL.FATIGUE_LEVELS[f]}</span><span class="faint">fatigue ${f}</span></h5><div class="pgrid"><div></div>${['Losing', 'Neutral', 'Winning'].map(x => `<div class="ch">${x}</div>`).join('')}`;
      for (let t = 0; t < 4; t++) {
        h += `<div class="rh">${RL.TILT_LEVELS[t].toLowerCase()}</div>`;
        for (let k = 0; k < 3; k++) {
          const s = RL.encode(f, t, k); const bg = color(s);
          const txtCol = (opts.mode === 'policy' || !vals) ? '#fff' : (C.lum(bg) > 0.45 ? '#03121a' : '#e7eef6');
          let main = '', small = '';
          if (opts.actions && (opts.mode === 'policy' || !vals)) main = ['▼', '●', '▲'][opts.actions[s]];
          if (vals && opts.mode !== 'policy') main = opts.fmt ? opts.fmt(vals[s]) : C.fmtNum(vals[s]);
          if (opts.actions && vals && opts.mode !== 'policy') small = RL.ACTION_NAMES[opts.actions[s]][0];
          if (opts.labels) small = opts.labels(s);
          const cls = ['cell'];
          if (opts.selected === s) cls.push('sel');
          if (opts.diffWith && opts.actions && opts.diffWith[s] !== opts.actions[s]) cls.push('diff');
          if (opts.changed && opts.changed.has(s)) cls.push('changed');
          h += `<div class="${cls.join(' ')}" data-s="${s}" tabindex="0" role="button" aria-label="${esc(RL.stateLabel(s))}" style="background:${bg};color:${txtCol}">${main}${small ? `<small>${small}</small>` : ''}</div>`;
        }
      }
      h += '</div></div>';
    }
    h += '</div>';
    if (opts.mode === 'policy' || !vals) h += `<div class="legend" style="margin-top:8px"><span><i style="background:${ACOL[0]}"></i>▼ LOWER difficulty</span><span><i style="background:${ACOL[1]}"></i>● KEEP</span><span><i style="background:${ACOL[2]}"></i>▲ RAISE difficulty</span>${opts.diffWith ? '<span><i style="background:var(--amber);border-radius:50%"></i>differs from reference</span>' : ''}${opts.changed ? '<span><i style="box-shadow:0 0 0 2px var(--amber) inset;background:transparent"></i>changed this iteration</span>' : ''}</div>`;
    else if (opts.mode === 'diverging') h += `<div class="legend" style="margin-top:8px"><span>${C.fmtNum(-Math.max(Math.abs(vmin), Math.abs(vmax)))}</span><span class="ramp" style="background:linear-gradient(90deg,#0aa5c4,#2b3340,#e0574a)"></span><span>${C.fmtNum(Math.max(Math.abs(vmin), Math.abs(vmax)))}</span><span class="faint">${esc(opts.legend || '')}</span></div>`;
    else h += `<div class="legend" style="margin-top:8px"><span>${C.fmtNum(vmin)}</span><span class="ramp" style="background:linear-gradient(90deg,${C.seqColor(0)},${C.seqColor(.5)},${C.seqColor(1)})"></span><span>${C.fmtNum(vmax)}</span><span class="faint">${esc(opts.legend || '')}</span></div>`;
    el.innerHTML = h;
    $$('.cell', el).forEach(c => {
      const s = +c.dataset.s;
      c.addEventListener('mousemove', ev => C.showTip(`<b>${esc(RL.stateLabel(s))}</b><div class="faint" style="font-size:11px">state ${s} · key ${RL.stateKey(s)}</div>${opts.tip ? opts.tip(s) : ''}`, ev));
      c.addEventListener('mouseleave', C.hideTip);
      if (opts.onSelect) { c.addEventListener('click', () => opts.onSelect(s)); c.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); opts.onSelect(s); } }); }
    });
  };

  App.policyOptions = (sel = 'dp_optimal', exclude = []) => [['dp_optimal', 'DP optimal π*'], ['q_learning', 'Q-learning (production)'], ['heuristic', 'Manual heuristic'], ['keep', 'Always KEEP'], ['lower', 'Always LOWER'], ['raise', 'Always RAISE'], ['random', 'Uniform random']]
    .filter(o => !exclude.includes(o[0])).map(([v, l]) => `<option value="${v}" ${v === sel ? 'selected' : ''}>${l}</option>`).join('');

  App.slider = (id, label, min, max, step, val, fmt = v => v) => `<div class="field"><label for="${id}">${label}<span class="val" id="${id}-v">${fmt(val)}</span></label><input type="range" id="${id}" min="${min}" max="${max}" step="${step}" value="${val}"></div>`;
  App.bindSlider = (root, id, fmt = v => v, onInput) => { const el = $('#' + id, root); const out = $('#' + id + '-v', root); el.addEventListener('input', () => { out.textContent = fmt(+el.value); if (onInput) onInput(+el.value); }); return el; };
  App.num = (id, label, val, min, max, step = 1) => `<div class="field"><label for="${id}">${label}</label><input type="number" id="${id}" value="${val}" min="${min}" max="${max}" step="${step}"></div>`;
  App.sel = (id, label, optionsHtml) => `<div class="field"><label for="${id}">${label}</label><select id="${id}">${optionsHtml}</select></div>`;
  App.opts = (pairs, sel) => pairs.map(([v, l]) => `<option value="${v}" ${String(v) === String(sel) ? 'selected' : ''}>${l}</option>`).join('');
  App.val = (root, id) => { const el = $('#' + id, root); return el.type === 'number' || el.type === 'range' ? +el.value : (el.type === 'checkbox' ? el.checked : el.value); };
  App.run = async function (btn, fn) { const old = btn.innerHTML; btn.disabled = true; btn.innerHTML = '<span class="spin"></span> Running'; try { await fn(); } catch (e) { console.error(e); App.toast(e.detail || e.message || String(e), 5000); } finally { btn.disabled = false; btn.innerHTML = old; } };
  App.engineNote = () => `<span class="faint" style="font-size:11.5px">computed by ${App.engine === 'python' ? 'Python backend' : 'browser engine'}</span>`;

  // ------------------------------------------------------------ boot
  App.boot = async function () {
    $$('#nav button').forEach(b => b.addEventListener('click', () => App.go(b.dataset.go)));
    $('#menuBtn').addEventListener('click', () => document.body.classList.toggle('nav-open'));
    $('#vivaToggle').addEventListener('change', e => { document.body.classList.toggle('viva', e.target.checked); try { localStorage.setItem('ciq-viva', e.target.checked ? '1' : '0'); } catch (x) { /* ignore */ } App.renderViva(App.current); });
    try { if (localStorage.getItem('ciq-viva') === '1') { $('#vivaToggle').checked = true; document.body.classList.add('viva'); } } catch (x) { /* ignore */ }
    const panel = $('#enginePanel');
    $('#engineBtn').addEventListener('click', () => panel.classList.toggle('hidden'));
    $('#engineClose').addEventListener('click', () => panel.classList.add('hidden'));
    $('#engineOffline').addEventListener('click', () => { App.setEngine('browser', 'Using the in-browser engine.'); panel.classList.add('hidden'); });
    $('#engineConnect').addEventListener('click', async () => {
      const base = $('#backendUrl').value.replace(/\/$/, ''); $('#engineMsg').textContent = 'Connecting…';
      if (await App.probe(base)) { App.base = base; try { localStorage.setItem('ciq-base', base); } catch (x) { /* ignore */ } App.setEngine('python', 'Connected to the Python backend.'); $('#engineMsg').textContent = 'Connected.'; panel.classList.add('hidden'); }
      else $('#engineMsg').innerHTML = `<span class="bad">Could not reach ${esc(base)}/health.</span> Start it with <span class="kbd">uvicorn backend_app:app --port 8000</span> and use <span class="kbd">http://localhost:8000/api</span>.`;
    });
    let saved = null; try { saved = localStorage.getItem('ciq-base'); } catch (x) { /* ignore */ }
    const configured = (window.CLUTCHIQ_API_BASE || '').replace(/\/+$/, '');
    const candidates = [saved, configured, location.protocol === 'file:' ? 'http://localhost:8000/api' : '/api'].filter((x, i, a) => x && a.indexOf(x) === i);
    App.setEngine('browser');
    for (const c of candidates) { if (await App.probe(c)) { App.base = c; $('#backendUrl').value = c; App.setEngine('python'); break; } }
    if (App.engine !== 'python') $('#backendUrl').value = candidates[candidates.length - 1];
    App.go((location.hash || '#overview').slice(1), false);
  };
})(window);
