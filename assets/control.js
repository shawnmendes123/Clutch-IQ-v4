/* Clutch IQ — RL control room: Live session (U1), Q-learning (TD), MDP explorer (U3), Dynamic programming (U4). */
(function (root) {
  'use strict';
  const App = root.App, RL = root.ClutchRL, C = root.Charts, esc = C.esc, $ = App.$, $$ = App.$$;
  const f = App.f;

  App.miniMap = function (el, actions, diffWith, caption) {
    let h = '<div class="mini">';
    for (let fb = 0; fb < 4; fb++) {
      h += '<div class="mb">';
      for (let t = 0; t < 4; t++) for (let k = 0; k < 3; k++) { const s = RL.encode(fb, t, k); h += `<div class="mc ${diffWith && diffWith[s] !== actions[s] ? 'diff' : ''}" data-s="${s}" style="background:${App.ACOL[actions[s]]}"></div>`; }
      h += '</div>';
    }
    el.innerHTML = h + '</div>' + (caption ? `<div class="mini-cap">${caption}</div>` : '');
    $$('.mc', el).forEach(c => { const s = +c.dataset.s; c.addEventListener('mousemove', ev => C.showTip(`<b>${esc(RL.stateLabel(s))}</b><div class="r"><span>action</span><span>${RL.ACTION_NAMES[actions[s]]}</span></div>${diffWith ? `<div class="r"><span>reference</span><span>${RL.ACTION_NAMES[diffWith[s]]}</span></div>` : ''}`, ev)); c.addEventListener('mouseleave', C.hideTip); });
  };
  App.tabs = function (el, names, onPick, initial = 0) {
    el.innerHTML = names.map((n, i) => `<button role="tab" aria-selected="${i === initial}" data-i="${i}">${n}</button>`).join('');
    $$('button', el).forEach(b => b.addEventListener('click', () => { $$('button', el).forEach(x => x.setAttribute('aria-selected', String(x === b))); onPick(+b.dataset.i); }));
  };
  App.backupHtml = function (bd, mode, V_label = 'V') {
    // mode: 'expectation' | 'optimality'
    const g = bd.gamma; const pa = bd.per_action;
    const rows = pa.map(p => {
      const terms = p.terms.map((t, j) => `<tr><td>${j === 0 ? App.act(p.action) : ''}</td><td class="mono">${esc(t.label)}</td><td class="n">${f(t.p, 4)}</td><td class="n">${f(t.r, 3)}</td><td class="n">${f(t.V_next, 3)}</td><td class="n" title="${f(t.p, 4)} × (${f(t.r, 3)} + ${g} × ${f(t.V_next, 3)})">${f(t.contribution, 4)}</td></tr>`).join('');
      const other = p.other_terms.count ? `<tr><td></td><td class="faint">+${p.other_terms.count} smaller</td><td class="n">${f(p.other_terms.p, 4)}</td><td></td><td></td><td class="n">${f(p.other_terms.contribution, 4)}</td></tr>` : '';
      return terms + other + `<tr class="sel"><td colspan="5" class="n">q(s, ${p.action_name}) = Σ p·[r + γV(s′)]</td><td class="n"><b class="cy">${f(p.q, 4)}</b></td></tr>`;
    }).join('');
    let concl;
    if (mode === 'expectation') {
      const pi = bd.expectation.pi; concl = `V<sub>π</sub>(s) = Σ<sub>a</sub> π(a|s)·q<sub>π</sub>(s,a) = ${pa.map(p => `${f(pi[p.action], 3)}×${f(p.q, 3)}`).join(' + ')} = <span class="res">${f(bd.expectation.value, 4)}</span>`;
    } else concl = `V*(s) = max<sub>a</sub> q*(s,a) = max(${pa.map(p => f(p.q, 3)).join(', ')}) = <span class="res">${f(bd.optimality.max_q, 4)}</span>  →  greedy action ${App.act(bd.optimality.argmax)}`;
    return `<div class="tbl-wrap scroll-y" style="max-height:330px"><table class="t"><thead><tr><th>a</th><th>s′</th><th class="n">p(s′|s,a)</th><th class="n">r(s,a,s′)</th><th class="n">${V_label}(s′)</th><th class="n">p·[r + γ${V_label}(s′)]</th></tr></thead><tbody>${rows}</tbody></table></div><div class="eq" style="margin-top:10px"><span class="calc">${concl}</span></div>`;
  };

  // shared policy fetch (works on both engines)
  App.getPolicy = async function (name, gamma = 0.9) {
    if (name === 'q_learning') { const q = await App.api('GET', `/rl/q/table?gamma=${gamma}`); return { actions: q.actions, Q: q.Q, matrix: RL.deterministicPolicy(q.actions), label: 'Q-learning (production table)' }; }
    if (name === 'dp_optimal') { const v = await App.api('POST', '/rl/dp/value-iteration', { gamma, theta: 1e-8, max_iter: 3000, snapshots: 0 }); return { actions: v.actions, Q: v.Q, matrix: RL.deterministicPolicy(v.actions), label: `DP optimal π* (γ=${gamma})` }; }
    const m = RL.resolvePolicy(name); return { actions: RL.policyActions(m).slice(0, 48), Q: null, matrix: m, label: { heuristic: 'Manual heuristic', keep: 'Always KEEP', lower: 'Always LOWER', raise: 'Always RAISE', random: 'Uniform random' }[name] };
  };

  // =================================================================== LIVE SESSION (Unit 1)
  App.screens['ctl-session'] = {
    render(el) {
      el.innerHTML = `${App.header('Live session', 'Shadow one simulated session match by match. Every transition shows the state, the action, the immediate reward and how much it contributes to the discounted return — the RL interaction loop, step by step.', [['u1', 'Unit 1 · RL loop'], ['u1', 'Discounted return']])}
      <div class="grid g-side">
        <div class="panel ctl">
          <h2>Session setup</h2>
          <div id="sPlayer" class="note"></div>
          <div class="grid g3" style="gap:8px">
            ${App.sel('sF', 'Fatigue', App.opts(RL.FATIGUE_LEVELS.map((x, i) => [i, x]), 0))}${App.sel('sT', 'Tilt', App.opts(RL.TILT_LEVELS.map((x, i) => [i, x]), 0))}${App.sel('sK', 'Streak', App.opts(RL.STREAK_LEVELS.map((x, i) => [i, x]), 1))}
          </div>
          ${App.sel('sPol', 'Controller policy π', App.policyOptions('q_learning'))}
          <div class="row">${App.num('sSeed', 'Episode seed', 7, 0, 999999)}<button class="btn sm" id="sNewSeed" style="align-self:flex-end">New seed</button></div>
          ${App.slider('sGamma', 'Discount γ for the return', 0, 1, 0.01, 0.9, v => v.toFixed(2))}
          <div class="row"><button class="btn primary" id="sStep">Play one match</button><button class="btn" id="sAuto">Auto-play</button><button class="btn" id="sReset">Reset</button></div>
          <p class="sub" style="margin:0">Same seed + same start state + same policy = the same session, on either engine (or via <span class="mono">POST /api/rl/trajectory</span>).</p>
        </div>
        <div class="stack">
          <div class="grid g2">
            <div class="panel hot"><div class="row between"><h2>State S<sub>t</sub> ${App.prov('DERIVED FEATURE')}</h2><span class="mono faint" id="sT_">t = 0</span></div><div id="sGauges" class="stack" style="margin-top:10px"></div></div>
            <div class="panel"><div class="row between"><h2>Agent decision A<sub>t</sub></h2><span id="sPolLbl" class="faint" style="font-size:12px"></span></div><div class="decision" id="sDecision" style="margin:8px 0 10px">—</div><div id="sQ"></div></div>
          </div>
          <div class="panel"><h2>What the simulator expects for each action ${App.prov('SIMULATED')}</h2><p class="sub">Closed-form quantities of the session simulator for the current state — this is what an agent would have to learn; the RL algorithms never read it directly.</p><div id="sOutlook"></div></div>
          <div class="panel"><h2>Last reward R<sub>t+1</sub></h2><div id="sReward" class="empty">Play a match to see the reward decomposition.</div></div>
        </div>
      </div>
      <div class="panel" style="margin-top:16px"><div class="row between"><h2>Trajectory S₀ → A₀ → R₁ → S₁ → …</h2><span id="sSummary" class="dim"></span></div><div class="tape" id="sTape"></div><div id="sTable"></div></div>
      <div class="grid g2" style="margin-top:16px">
        <div class="panel"><h2>Same episode, different γ</h2><p class="sub">G₀ = Σ γ<sup>t</sup> R<sub>t+1</sub> for this exact reward sequence. γ = 0 keeps only the first reward (myopic); γ → 1 weighs the whole session (far-sighted).</p><div id="sGcurve"></div></div>
        <div class="panel"><h2>Cumulative reward vs discounted return</h2><p class="sub">Undiscounted sum Σ R and discounted Σ γ<sup>t</sup>R along the session.</p><div id="sCum"></div></div>
      </div>`;
      const seedEl = $('#sSeed', el);
      $('#sNewSeed', el).addEventListener('click', () => { seedEl.value = Math.floor(Math.random() * 99999); this.reset(el); });
      App.bindSlider(el, 'sGamma', v => v.toFixed(2), () => this.drawTraj(el));
      ['sF', 'sT', 'sK', 'sPol', 'sSeed'].forEach(id => $('#' + id, el).addEventListener('change', () => { if (id !== 'sPol' && id !== 'sSeed') App.data['ctl-session'].player = null; this.reset(el); }));
      $('#sStep', el).addEventListener('click', () => this.step(el));
      $('#sReset', el).addEventListener('click', () => this.reset(el));
      $('#sAuto', el).addEventListener('click', e => { if (this._auto) { clearInterval(this._auto); this._auto = null; e.currentTarget.textContent = 'Auto-play'; } else { e.currentTarget.textContent = 'Pause'; this._auto = setInterval(() => { if (!this.step(el)) { clearInterval(this._auto); this._auto = null; $('#sAuto', el).textContent = 'Auto-play'; } }, 900); } });
      document.addEventListener('engine-change', () => { if (App.rendered['ctl-session']) this.reset(el); });
      this.applyPlayer(el); this.reset(el);
    },
    onShow(el) { if (this.applyPlayer(el)) this.reset(el); },
    applyPlayer(el) {
      const d = App.data['ctl-session'] || (App.data['ctl-session'] = {}); const p = d.player;
      if (p && p !== this._lastPlayer) {
        this._lastPlayer = p; $('#sF', el).value = RL.bucketFatigue(p.fatigue_index); $('#sT', el).value = RL.bucketTilt(p.tilt_score); $('#sK', el).value = RL.bucketStreak(p.win_streak);
        $('#sPlayer', el).innerHTML = `Shadowing <b>${esc(p.name)}</b> (${esc(p.archetype)}) — fatigue ${f(p.fatigue_index, 2)}, tilt ${f(p.tilt_score, 2)}, streak ${p.win_streak} ${App.prov(p.source || 'SIMULATED')}`;
        return true;
      }
      if (!p) $('#sPlayer', el).innerHTML = 'Pick a start state, or click a player in the <a href="#ops-queue">live queue</a>.';
      return false;
    },
    async reset(el) {
      if (this._auto) { clearInterval(this._auto); this._auto = null; $('#sAuto', el).textContent = 'Auto-play'; }
      const d = App.data['ctl-session']; const pol = $('#sPol', el).value; d.pol = await App.getPolicy(pol, 0.9);
      d.s0 = RL.encode(+$('#sF', el).value, +$('#sT', el).value, +$('#sK', el).value); d.s = d.s0; d.seed = +$('#sSeed', el).value; d.steps = []; d.done = false;
      d.envRng = new RL.Mulberry32(RL.deriveSeed(d.seed, 0)); d.actRng = new RL.Mulberry32(RL.deriveSeed(d.seed, 1));
      $('#sPolLbl', el).textContent = d.pol.label; $('#sReward', el).className = 'empty'; $('#sReward', el).textContent = 'Play a match to see the reward decomposition.';
      this.drawState(el); this.drawTraj(el); App.renderViva('ctl-session');
    },
    step(el) {
      const d = App.data['ctl-session']; if (d.done) { App.toast('Session over — press Reset or pick a new seed.'); return false; }
      const row = d.pol.matrix[d.s]; const a = d.actRng.sampleDiscrete(row);
      const [s2, r, done, info] = RL.envStep(d.s, a, d.envRng);
      d.steps.push({ s: d.s, a, r, s2, done, info, pi: row.slice() }); d.s = s2; d.done = done;
      this.drawReward(el, d.steps[d.steps.length - 1]); this.drawState(el); this.drawTraj(el); App.renderViva('ctl-session');
      return !done;
    },
    drawState(el) {
      const d = App.data['ctl-session']; const s = d.s; $('#sT_', el).textContent = `t = ${d.steps.length}`;
      if (s === RL.TERMINAL) { $('#sGauges', el).innerHTML = `<div class="decision bad" style="font-size:22px">SESSION OVER</div><p class="sub">Terminal state reached: ${esc(d.steps[d.steps.length - 1].info.rage_quit ? 'rage quit' : d.steps[d.steps.length - 1].info.natural_end ? 'natural end (exhausted)' : 'boredom quit')}. V(terminal) = 0.</p>`; $('#sDecision', el).textContent = '—'; $('#sQ', el).innerHTML = ''; $('#sOutlook', el).innerHTML = ''; return; }
      const [fb, tb, kb] = RL.decode(s);
      const g = (lab, val, max, names, col) => `<div class="gauge"><span>${lab}</span><div class="tr"><i style="width:${(val + 1) / max * 100}%;background:${col}"></i>${[1, 2, 3].slice(0, max - 1).map(x => `<b style="left:${x / max * 100}%"></b>`).join('')}</div><span class="mono">${names[val]}</span></div>`;
      $('#sGauges', el).innerHTML = g('Fatigue', fb, 4, RL.FATIGUE_LEVELS, 'var(--cyan)') + g('Tilt', tb, 4, RL.TILT_LEVELS, 'var(--amber)') + g('Streak', kb, 3, RL.STREAK_LEVELS, 'var(--magenta)') + `<div class="row" style="margin-top:4px">${App.stateChip(s)}<span class="faint mono">s = ${s}, key ${RL.stateKey(s)}</span></div>`;
      const pi = d.pol.matrix[s]; const best = pi.indexOf(Math.max(...pi));
      $('#sDecision', el).innerHTML = `<span style="color:${['#22c3e0', '#a9b6c6', '#e55bd0'][best]}">${['▼', '●', '▲'][best]} ${RL.ACTION_NAMES[best]}</span>`;
      const q = d.pol.Q ? d.pol.Q[s] : null; const mx = q ? Math.max(...q.map(Math.abs)) || 1 : 1;
      $('#sQ', el).innerHTML = (q ? '<div class="lbl">Q(s, a) — the agent\'s estimate of the return after each action</div>' + [0, 1, 2].map(a => `<div class="qbar"><span>${RL.ACTION_NAMES[a]}</span><div class="tr"><i style="width:${Math.abs(q[a]) / mx * 100}%;background:${App.ACOL[a]};opacity:${a === best ? 1 : .55}"></i></div><span class="mono">${f(q[a], 3)}</span></div>`).join('') : '') + `<div class="lbl" style="margin-top:8px">π(a|s) = ${pi.map((p, a) => `${RL.ACTION_NAMES[a]} ${f(p, 2)}`).join(' · ')}</div>`;
      const sm = [0, 1, 2].map(a => RL.stepModel(s, a));
      $('#sOutlook', el).innerHTML = App.table([{ h: 'Action', f: r => App.act(r.a) }, { h: 'P(win)', n: 1, f: r => f(r.m.p_win, 2) }, { h: 'Closeness C', n: 1, f: r => f(r.m.closeness, 2) }, { h: 'P(rage quit)', n: 1, f: r => App.pct(r.m.rage_prob, 1) }, { h: 'P(boredom quit)', n: 1, f: r => App.pct(r.m.boredom_prob, 1) }, { h: 'P(fatigue ↑)', n: 1, f: r => f(r.m.fatigue_up, 2) }, { h: 'E[R<sub>t+1</sub>]', n: 1, f: r => f(r.m.expected_reward_immediate, 3) }], sm.map((m, a) => ({ a, m })), { rowClass: r => r.a === best ? 'sel' : '' });
    },
    drawReward(el, st) {
      const C_ = st.info.closeness, cont = st.info.rage_quit || (st.done && !st.info.natural_end) ? 0 : 1, rage = st.info.rage_quit ? 1 : 0;
      $('#sReward', el).className = '';
      $('#sReward', el).innerHTML = `<div class="eq"><span class="calc">R = w₁·(match completed) + w₂·C − w₃·(rage quit) = 1.0×${cont} + 1.2×${f(C_, 3)} − 3.0×${rage} = <span class="res ${st.r < 0 ? 'bad' : ''}">${f(st.r, 3)}</span></span></div><p class="sub" style="margin-top:8px">${st.info.win ? 'Won' : 'Lost'} the match (P(win) was ${f(st.info.p_win, 2)}). ${st.info.rage_quit ? '<span class="bad">Rage quit — terminal, penalised by w₃.</span>' : st.info.natural_end ? '<span class="good">Exhausted player logged off — natural terminal state, no penalty.</span>' : st.done ? '<span class="warn">One-sided win, the player left out of boredom — terminal, no continuation reward.</span>' : 'Session continues.'}</p>`;
    },
    drawTraj(el) {
      const d = App.data['ctl-session']; if (!d || !d.steps) return; const g = +$('#sGamma', el).value; const st = d.steps;
      const R = st.map(x => x.r); const G = RL.returnsFromRewards(R, g);
      $('#sTape', el).innerHTML = st.length ? st.map((x, t) => `<div class="tstep ${x.info.rage_quit ? 'end' : ''} ${x.info.natural_end ? 'nat' : ''}"><div class="t">t = ${t}</div><div class="s">${RL.stateShort(x.s)}</div>${App.act(x.a)}<div class="r ${x.r < 0 ? 'bad' : 'good'}" style="margin-top:4px">R = ${f(x.r, 2)}</div><div class="faint mono" style="font-size:10.5px">γ<sup>${t}</sup>R = ${f(Math.pow(g, t) * x.r, 3)}</div></div><span class="tarrow">→</span>`).join('') + `<div class="tstep"><div class="t">t = ${st.length}</div><div class="s">${RL.stateShort(d.s)}</div><div class="faint">${d.done ? 'terminal' : 'next state'}</div></div>` : '<div class="empty" style="flex:1">Press “Play one match”.</div>';
      const tape = $('#sTape', el); tape.scrollLeft = tape.scrollWidth;
      let cum = 0, dc = 0;
      const rows = st.map((x, t) => { cum += x.r; dc += Math.pow(g, t) * x.r; return { t, x, cum, dc, G: G[t] }; });
      $('#sTable', el).innerHTML = st.length ? App.table([{ h: 't', n: 1, f: r => r.t }, { h: 'S<sub>t</sub>', f: r => `<span class="mono">${RL.stateShort(r.x.s)}</span>` }, { h: 'A<sub>t</sub>', f: r => App.act(r.x.a) }, { h: 'π(A<sub>t</sub>|S<sub>t</sub>)', n: 1, f: r => f(r.x.pi[r.x.a], 2) }, { h: 'R<sub>t+1</sub>', n: 1, f: r => f(r.x.r, 3) }, { h: 'S<sub>t+1</sub>', f: r => `<span class="mono">${RL.stateShort(r.x.s2)}</span>${r.x.done ? ' <span class="tag">terminal</span>' : ''}` }, { h: 'γ<sup>t</sup>', n: 1, f: r => f(Math.pow(g, r.t), 4) }, { h: 'γ<sup>t</sup>R<sub>t+1</sub>', n: 1, f: r => f(Math.pow(g, r.t) * r.x.r, 4) }, { h: 'Σ R (cumulative)', n: 1, f: r => f(r.cum, 3) }, { h: 'G<sub>t</sub> = R<sub>t+1</sub> + γG<sub>t+1</sub>', n: 1, f: r => d.done ? `<b>${f(r.G, 4)}</b>` : '<span class="faint" title="needs the rest of the episode">pending</span>' }], rows, { scroll: true }) : '';
      const ended = d.done ? (st[st.length - 1].info.rage_quit ? 'rage quit' : st[st.length - 1].info.natural_end ? 'natural end' : 'boredom quit') : 'in progress';
      $('#sSummary', el).innerHTML = st.length ? `${st.length} matches · ${ended} · Σ R = <span class="mono">${f(cum, 3)}</span> · ${d.done ? `G₀ = <span class="mono cy">${f(G[0], 4)}</span>` : `Σγ<sup>t</sup>R so far = <span class="mono">${f(dc, 3)}</span>`}` : '';
      const gs = Array.from({ length: 51 }, (_, i) => i / 50);
      if (st.length) C.line($('#sGcurve', el), { series: [{ name: 'G₀(γ)', points: gs.map(x => [x, RL.discountedReturn(R, x)]) }], xLabel: 'γ', yLabel: 'G₀', refLines: [{ y: RL.discountedReturn(R, g), label: `current γ=${g.toFixed(2)}`, color: '#ff8de0' }], height: 210, fmtX: v => v.toFixed(1) });
      else $('#sGcurve', el).innerHTML = '<div class="empty">Play a few matches.</div>';
      if (st.length) C.line($('#sCum', el), { series: [{ name: 'Σ R', points: rows.map(r => [r.t, r.cum]), markers: true }, { name: 'Σ γᵗR', points: rows.map(r => [r.t, r.dc]), markers: true }], xLabel: 't (match)', yLabel: 'reward', height: 210, fmtX: v => v.toFixed(0) });
      else $('#sCum', el).innerHTML = '';
      App.setData('ctl-session', {});
    },
  };
  App.viva['ctl-session'] = d => {
    const st = d.steps || []; const last = st[st.length - 1];
    let ex = null;
    if (st.length) {
      const g = +(document.getElementById('sGamma') || { value: 0.9 }).value; const R = st.map(x => x.r); const G = RL.returnsFromRewards(R, g);
      ex = `<span class="calc">Last transition: S=${RL.stateShort(last.s)}, A=${RL.ACTION_NAMES[last.a]}, R=${f(last.r, 3)}, S′=${RL.stateShort(last.s2)}.<br>${d.done ? `G₀ = ${R.slice(0, 5).map((r, t) => `${f(Math.pow(g, t), 3)}×${f(r, 2)}`).join(' + ')}${R.length > 5 ? ' + …' : ''} = <span class="res">${f(G[0], 4)}</span>  (γ = ${g})` : `G₀ cannot be computed yet: it needs the rewards still to come (delayed reward).`}</span>`;
    }
    return { unit: 'Unit 1', unitClass: 'u1', concept: 'Interaction loop, immediate vs delayed reward, discounted return', prov: 'SIMULATED',
      why: 'Lobby difficulty affects tilt and fatigue several matches later, so the controller must value delayed consequences — exactly what the return G<sub>t</sub> captures.',
      inputs: 'Start state (from a queue player or chosen), policy π, seed, γ.', outputs: 'Trajectory (S<sub>t</sub>, A<sub>t</sub>, R<sub>t+1</sub>, S<sub>t+1</sub>, done), Σ R, G<sub>t</sub> for every t.',
      formula: 'G<sub>t</sub> = R<sub>t+1</sub> + γR<sub>t+2</sub> + γ²R<sub>t+3</sub> + … = R<sub>t+1</sub> + γG<sub>t+1</sub>,  G<sub>T</sub> = 0;   A<sub>t</sub> ~ π(·|S<sub>t</sub>)',
      example: ex, assumptions: 'Session dynamics come from the simulator; the controller sees only the 48 bucketed states.', code: 'rl_core.env_step · rl_evaluation.trajectory · assets/control.js (Live session)' };
  };

  // =================================================================== Q-LEARNING
  App.screens['ctl-q'] = {
    render(el) {
      el.innerHTML = `${App.header('Tabular Q-learning', 'The original Clutch IQ optimiser, now named correctly: a 48×3 table learned by off-policy TD control. v3 called it a “DQN”; there is no neural network, replay buffer or target network here.', [['u1', 'Model-free'], ['ux', 'TD control'], ['u2', 'ε-greedy']])}
      <div class="tabs" id="qTabs" role="tablist"></div>
      <div id="qPane0">
        <div class="grid g-side-r">
          <div class="panel"><div class="row between"><h2>Production policy (greedy w.r.t. Q) ${App.prov('SIMULATED')}</h2><span id="qMeta" class="faint" style="font-size:12px"></span></div><p class="sub">Amber dots mark states where Q-learning disagrees with the DP optimum. Click a state to explain one TD update there.</p><div id="qMap"></div></div>
          <div class="stack"><div class="panel"><h2>Agreement with DP</h2><div class="kpis" id="qKpis"></div><h3>Disagreements, largest value gap first</h3><div id="qDis"></div></div></div>
        </div>
        <div class="panel" style="margin-top:16px"><h2>One TD update, with real numbers</h2>
          <div class="grid g-side"><div class="ctl">
            <div id="qTdState"></div>
            ${App.sel('qTdA', 'Action A<sub>t</sub>', App.opts([[0, 'LOWER'], [1, 'KEEP'], [2, 'RAISE']], 1))}
            <div class="grid g2" style="gap:8px">${App.num('qTdAlpha', 'α (step size)', 0.05, 0.001, 1, 0.01)}${App.num('qTdGamma', 'γ', 0.9, 0, 1, 0.01)}</div>
            ${App.num('qTdSeed', 'Transition seed', 1, 0, 99999)}
            <button class="btn primary" id="qTdRun">Sample a transition &amp; update</button></div>
            <div id="qTdOut" class="empty">Choose a state on the map above (default: TIRED · TILTED · LOSING) and sample a transition.</div></div></div>
        <div class="panel" style="margin-top:16px"><h2>Q-table</h2><div id="qTable"></div></div>
      </div>
      <div id="qPane1" class="hidden">
        <div class="grid g-side">
          <div class="panel ctl"><h2>Training lab</h2><p class="sub">Trains a fresh table. The production policy is never overwritten unless you ask for it explicitly (Python backend only).</p>
            <div class="grid g2" style="gap:8px">${App.num('qEp', 'Episodes', 30000, 100, 100000, 100)}${App.num('qAlpha', 'α', 0.05, 0.001, 1, 0.005)}${App.num('qGamma', 'γ', 0.9, 0, 1, 0.01)}${App.num('qSeed', 'Seed', 7, 0, 999999)}${App.num('qE0', 'ε start', 1.0, 0, 1, 0.05)}${App.num('qDecay', 'ε decay / episode', 0.9998, 0.9, 1, 0.0001)}${App.num('qEmin', 'ε minimum', 0.05, 0, 1, 0.01)}${App.num('qWatch', 'Watch state (0–47)', 42, 0, 47)}</div>
            <button class="btn primary" id="qTrain">Train fresh table</button>
            <label class="checks"><label><input type="checkbox" id="qSave"> Save as production policy (backend)</label></label>
          </div>
          <div class="stack"><div class="kpis" id="qTKpis"></div><div class="grid g2"><div class="panel"><h2>Greedy-policy value J during training</h2><p class="sub">J = Σ d₀(s)·V<sup>π<sub>greedy</sub></sup>(s), evaluated on the model only for monitoring.</p><div id="qcJ"></div></div><div class="panel"><h2>Agreement with the DP policy</h2><div id="qcA"></div></div><div class="panel"><h2>Mean |TD error| per step</h2><div id="qcTD"></div></div><div class="panel"><h2>Exploration rate ε</h2><div id="qcE"></div></div></div></div>
        </div>
        <div class="grid g2" style="margin-top:16px"><div class="panel"><h2>Learned policy vs DP</h2><div id="qLMap"></div></div><div class="panel"><h2>Q(s,a) at the watched state, update by update</h2><div id="qWatchChart"></div><div id="qWatchTbl" style="margin-top:10px"></div></div></div>
      </div>`;
      App.tabs($('#qTabs', el), ['Production table & TD update', 'Training lab'], i => { $('#qPane0', el).classList.toggle('hidden', i !== 0); $('#qPane1', el).classList.toggle('hidden', i !== 1); if (i === 1 && !this._trained) $('#qTrain', el).click(); });
      this.tdState = RL.encode(2, 2, 0);
      $('#qTdRun', el).addEventListener('click', e => App.run(e.currentTarget, () => this.td(el)));
      $('#qTrain', el).addEventListener('click', e => App.run(e.currentTarget, () => this.train(el)));
      this.load(el); document.addEventListener('engine-change', () => { if (App.rendered['ctl-q']) this.load(el); });
    },
    async load(el) {
      const q = await App.api('GET', '/rl/q/table?gamma=0.9'); this.q = q; App.setData('ctl-q', { table: q });
      $('#qMeta', el).textContent = q.meta && q.meta.params ? `${q.meta.params.episodes} episodes · α=${q.meta.params.alpha} · seed ${q.meta.params.seed}` : '';
      this.drawMap(el);
      $('#qKpis', el).innerHTML = App.kpi(App.pct(q.agreement_with_dp, 1), 'states matching DP π*') + App.kpi(q.disagreements.length, 'states that differ') + App.kpi(q.disagreements.length ? f(Math.max(...q.disagreements.map(x => x.q_gap)), 3) : '0', 'largest q*-gap');
      $('#qDis', el).innerHTML = q.disagreements.length ? App.table([{ h: 'State', f: r => `<span class="mono" title="${esc(r.label)}">${r.short}</span>` }, { h: 'Q-learn', f: r => App.act(RL.ACTION_NAMES.indexOf(r.action)) }, { h: 'DP', f: r => App.act(RL.ACTION_NAMES.indexOf(r.reference_action)) }, { h: 'q*-gap', n: 1, f: r => f(r.q_gap, 3) }], q.disagreements, { scroll: true }) + '<p class="sub" style="margin-top:8px">q*-gap = q*(s, a<sub>DP</sub>) − q*(s, a<sub>Q</sub>): the value lost in that state. Small gaps are near-ties where sampling noise decides; large gaps are states the learner has not yet estimated well (usually rarely visited).</p>' : '<div class="note ok">Identical to the DP policy.</div>';
      $('#qTable', el).innerHTML = App.table([{ h: 's', n: 1, f: (r, i) => i }, { h: 'key', f: (r, i) => `<span class="mono">${q.keys[i]}</span>` }, { h: 'State', f: (r, i) => esc(RL.stateLabel(i)) }, { h: 'Q(s, LOWER)', n: 1, f: r => f(r[0], 4) }, { h: 'Q(s, KEEP)', n: 1, f: r => f(r[1], 4) }, { h: 'Q(s, RAISE)', n: 1, f: r => f(r[2], 4) }, { h: 'greedy', f: (r, i) => App.act(q.actions[i]) }, { h: 'DP', f: (r, i) => App.act(q.dp_actions[i]) }], q.Q, { scroll: true, rowClass: (r, i) => q.actions[i] !== q.dp_actions[i] ? 'sel' : '' });
      this.drawTdState(el);
      if (!this._tdDone) { this._tdDone = true; $('#qTdRun', el).click(); }
    },
    drawMap(el) { const q = this.q; App.policyMap($('#qMap', el), { actions: q.actions, mode: 'policy', diffWith: q.dp_actions, selected: this.tdState, onSelect: s => { this.tdState = s; this.drawMap(el); this.drawTdState(el); }, tip: s => [0, 1, 2].map(a => `<div class="r"><span>Q(s,${RL.ACTION_NAMES[a]})</span><span>${f(q.Q[s][a], 3)}</span></div>`).join('') + `<div class="r"><span>DP action</span><span>${RL.ACTION_NAMES[q.dp_actions[s]]}</span></div>` }); },
    drawTdState(el) { $('#qTdState', el).innerHTML = `<div class="lbl">State S<sub>t</sub></div>${App.stateChip(this.tdState)}`; },
    async td(el) {
      const s = this.tdState, a = +$('#qTdA', el).value, alpha = +$('#qTdAlpha', el).value, gamma = +$('#qTdGamma', el).value, seed = +$('#qTdSeed', el).value;
      const rng = new RL.Mulberry32(RL.deriveSeed(seed, s * 3 + a)); const [s2, r, done, info] = RL.envStep(s, a, rng);
      const u = await App.api('POST', '/rl/q/td-step', { state: s, action: a, reward: r, next_state: s2, done, alpha, gamma });
      const q2 = s2 === RL.TERMINAL ? [0, 0, 0] : this.q.Q[s2];
      $('#qTdOut', el).className = '';
      $('#qTdOut', el).innerHTML = `<div class="row" style="margin-bottom:8px">${App.stateChip(s)} ${App.act(a)} <span class="mono">→ R = ${f(r, 3)}</span> → ${App.stateChip(s2)} ${App.prov('SIMULATED')}</div>
        <div class="eq"><div class="calc">
        TD target = R + γ·max<sub>a′</sub> Q(S′, a′) ${done ? '<span class="d">(S′ terminal ⇒ no bootstrap)</span>' : ''}<br>
        &nbsp;&nbsp;= ${f(r, 4)} + ${gamma}×max(${q2.map(x => f(x, 3)).join(', ')}) = ${f(r, 4)} + ${gamma}×${f(u.max_next_q, 4)} = <span class="res">${f(u.td_target, 4)}</span><br>
        TD error δ = target − Q(S, A) = ${f(u.td_target, 4)} − ${f(u.old_q, 4)} = <span class="res">${f(u.td_error, 4)}</span><br>
        Q(S, A) ← Q(S, A) + α·δ = ${f(u.old_q, 4)} + ${alpha}×${f(u.td_error, 4)} = <span class="res">${f(u.new_q, 4)}</span></div></div>
        <p class="sub" style="margin-top:8px">${info.win ? 'Won' : 'Lost'} the sampled match. The update is computed on a copy — the production table is unchanged.</p>`;
      App.setData('ctl-q', { td: u });
    },
    async train(el) {
      const body = { episodes: App.val(el, 'qEp'), alpha: App.val(el, 'qAlpha'), gamma: App.val(el, 'qGamma'), seed: App.val(el, 'qSeed'), eps_start: App.val(el, 'qE0'), eps_decay: App.val(el, 'qDecay'), eps_min: App.val(el, 'qEmin'), watch_state: App.val(el, 'qWatch'), save_as_default: $('#qSave', el).checked };
      if (body.save_as_default && App.engine !== 'python') throw new Error('Saving a production policy needs the Python backend.');
      const r = await App.api('POST', '/rl/q/train', body); this._trained = true; App.setData('ctl-q', { train: r });
      $('#qTKpis', el).innerHTML = App.kpi(App.pct(r.agreement_with_dp, 1), 'agreement with DP') + App.kpi(f(r.J_learned, 3), 'J of learned policy', `J* = ${f(r.J_dp_optimal, 3)}`) + App.kpi(f(r.seconds, 2) + ' s', 'training time') + App.kpi(`${(r.explore_steps / 1000).toFixed(1)}k / ${(r.exploit_steps / 1000).toFixed(1)}k`, 'explore / exploit steps') + App.kpi(r.saved_as_default ? 'saved' : 'unchanged', 'production policy', '', r.saved_as_default ? 'warn' : '');
      const xs = r.curve.map(c => c.episodes);
      C.line($('#qcJ', el), { series: [{ name: 'J(greedy)', points: r.curve.map(c => [c.episodes, c.greedy_policy_value]) }], refLines: [{ y: r.J_dp_optimal, label: 'J* (DP)', color: '#3fdf8f' }], xLabel: 'episodes', yLabel: 'J', height: 200 });
      C.line($('#qcA', el), { series: [{ name: 'agreement', points: r.curve.map(c => [c.episodes, c.agreement_with_reference]) }], xLabel: 'episodes', yLabel: 'fraction of states', yMin: 0, yMax: 1, height: 200, fmtY: v => Math.round(v * 100) + '%', fmtTip: v => App.pct(v, 1) });
      C.line($('#qcTD', el), { series: [{ name: '|δ|', points: r.curve.map(c => [c.episodes, c.mean_abs_td_error]) }], xLabel: 'episodes', yLabel: 'mean |TD error|', height: 200 });
      C.line($('#qcE', el), { series: [{ name: 'ε', points: r.curve.map(c => [c.episodes, c.epsilon]) }], xLabel: 'episodes', yLabel: 'ε', yMin: 0, yMax: 1, height: 200 });
      App.policyMap($('#qLMap', el), { actions: r.actions, mode: 'policy', diffWith: r.dp_actions, tip: s => [0, 1, 2].map(a => `<div class="r"><span>Q(s,${RL.ACTION_NAMES[a]})</span><span>${f(r.Q[s][a], 3)}</span></div>`).join('') + `<div class="r"><span>visits</span><span>${r.visits[s].join(' / ')}</span></div>` });
      const w = r.watch_log; const qa = [[], [], []]; const cur = [0, 0, 0];
      w.forEach((u, i) => { cur[u.action] = u.new_q; qa.forEach((arr, a) => arr.push([i + 1, cur[a]])); });
      C.line($('#qWatchChart', el), { series: [0, 1, 2].map(a => ({ name: 'Q(s,' + RL.ACTION_NAMES[a] + ')', points: qa[a], color: App.ACOL[a] === '#4b5b70' ? '#a9b6c6' : App.ACOL[a] })), xLabel: `update # at state ${body.watch_state} (${RL.stateShort(body.watch_state)})`, yLabel: 'Q', height: 210 });
      $('#qWatchTbl', el).innerHTML = App.table([{ h: 'ep', n: 1, f: u => u.episode }, { h: 'A', f: u => App.act(u.action) }, { h: 'R', n: 1, f: u => f(u.reward, 3) }, { h: 'S′', f: u => `<span class="mono">${u.next_label}</span>` }, { h: 'old Q', n: 1, f: u => f(u.old_q, 4) }, { h: 'target', n: 1, f: u => f(u.td_target, 4) }, { h: 'δ', n: 1, f: u => f(u.td_error, 4) }, { h: 'new Q', n: 1, f: u => f(u.new_q, 4) }], w.slice(0, 20), { scroll: true });
    },
  };
  App.viva['ctl-q'] = d => {
    const u = d.td || (d.train && d.train.first_update);
    return { unit: 'Unit 1 · model-free', unitClass: 'u1', concept: 'Tabular Q-learning (off-policy TD control)', prov: 'SIMULATED',
      why: 'Learns the session policy from experience without knowing p(s′|s,a) — the model-free baseline that DP and MC are compared against.',
      inputs: 'Transitions (S, A, R, S′, done) generated by ε-greedy behaviour; α, γ, ε schedule, seed.', outputs: 'Q(s,a) table (48×3), greedy policy, learning curves.',
      formula: 'Q(S,A) ← Q(S,A) + α [ R + γ·max<sub>a′</sub>Q(S′,a′) − Q(S,A) ]   (target = R if S′ terminal);  behaviour: ε-greedy, ε ← max(ε<sub>min</sub>, ε·decay)',
      example: u ? `<span class="calc">S=${u.state_label}, A=${u.action_name}, R=${f(u.reward, 3)}, S′=${u.next_label}: target = ${f(u.reward, 3)} + ${u.gamma}×${f(u.max_next_q, 3)} = ${f(u.td_target, 4)}; δ = ${f(u.td_target, 4)} − ${f(u.old_q, 4)} = ${f(u.td_error, 4)}; new Q = ${f(u.new_q, 4)}</span>` : null,
      assumptions: 'Constant α (not a decaying schedule), so the table keeps fluctuating slightly; exploring starts from all 48 states for coverage.', code: 'q_learning.py: train, td_update · rl_optimizer.py (compat facade)' };
  };

  // =================================================================== MDP EXPLORER (Unit 3)
  App.screens['ctl-mdp'] = {
    render(el) {
      el.innerHTML = `${App.header('MDP explorer', 'The session controller as an explicit finite MDP ⟨S, A, P, R, γ⟩ with 48 states and 3 actions. Pick any state to see its transition probabilities, expected rewards, V and Q, and the Bellman backup computed with the program\'s actual numbers.', [['u3', 'Unit 3 · MDP'], ['u3', 'Bellman equations'], ['u3', 'Markov chain']])}
      <div class="grid g-side">
        <div class="stack">
          <div class="panel ctl"><h2>Model & policy</h2>
            ${App.sel('mPol', 'Policy π being evaluated', App.policyOptions('heuristic'))}
            ${App.slider('mEps', 'Soften π to ε-soft, ε', 0, 1, 0.05, 0, v => v.toFixed(2))}
            ${App.slider('mGamma', 'Discount γ', 0, 0.99, 0.01, 0.9, v => v.toFixed(2))}
            <div class="grid g2" style="gap:8px">${App.sel('mSamples', 'Samples per (s,a)', App.opts([[300, '300'], [1000, '1000'], [3000, '3000']], 1000))}${App.num('mSeed', 'Model seed', 7, 0, 99999)}</div>
          </div>
        </div>
        <div class="stack">
          <div class="panel"><h2>Choose a state <span class="faint" style="font-size:12px;font-family:var(--f-body)">colours show π(s); click any cell</span></h2><div id="mMap"></div></div>
          <div class="panel"><h2>Simulation-estimated transition model ${App.prov('MODEL ESTIMATE')}</h2><p class="sub">p̂(s′|s,a) = N(s,a,s′)/N(s,a) from rollouts of the session simulator — not from real players.</p><div id="mQual"></div></div>
        </div>
      </div>
      <div id="mDetail" style="margin-top:16px"></div>`;
      const go = () => this.load(el);
      ['mPol', 'mSamples', 'mSeed'].forEach(id => $('#' + id, el).addEventListener('change', go));
      App.bindSlider(el, 'mEps', v => v.toFixed(2)); App.bindSlider(el, 'mGamma', v => v.toFixed(2));
      ['mEps', 'mGamma'].forEach(id => $('#' + id, el).addEventListener('change', go));
      this.state = (App.data['ctl-mdp'] || {}).state ?? RL.encode(2, 2, 0);
      this.action = 0; go();
      document.addEventListener('engine-change', () => { if (App.rendered['ctl-mdp']) go(); });
    },
    onShow(el) { const s = (App.data['ctl-mdp'] || {}).state; if (s !== undefined && s !== this.state) { this.state = s; this.load(el); } },
    q(el) { return `gamma=${App.val(el, 'mGamma')}&samples=${App.val(el, 'mSamples')}&seed=${App.val(el, 'mSeed')}&policy=${App.val(el, 'mPol')}&policy_epsilon=${App.val(el, 'mEps')}&top_k=6`; },
    async load(el) {
      App.busy($('#mDetail', el), true);
      const d = await App.api('GET', `/rl/mdp/state/${this.state}?${this.q(el)}`);
      if (!this._mdp || this._mdpKey !== `${App.val(el, 'mSamples')}|${App.val(el, 'mSeed')}|${App.engine}`) { this._mdp = await App.api('GET', `/rl/mdp?samples=${App.val(el, 'mSamples')}&seed=${App.val(el, 'mSeed')}&include_matrices=false`); this._mdpKey = `${App.val(el, 'mSamples')}|${App.val(el, 'mSeed')}|${App.engine}`; }
      const mq = this._mdp;
      $('#mQual', el).innerHTML = `<div class="kpis">${App.kpi((mq.meta.total_simulated_transitions / 1000).toFixed(0) + 'k', 'simulated transitions')}${App.kpi(f(mq.quality_vs_simulator.max_abs_transition_error, 3), 'max |p̂ − p|', `s.e. bound ${f(mq.quality_vs_simulator.worst_case_std_error, 3)}`)}${App.kpi(f(mq.quality_vs_simulator.nonzero_transitions_per_pair_mean, 1), 'successors per (s,a)')}${App.kpi(mq.check.rows_sum_to_one ? 'yes' : 'NO', 'Σ<sub>s′</sub> p̂ = 1')}</div><p class="sub" style="margin-top:8px">Error is measured against the simulator's own exact law, which exists only because we wrote the simulator — real player dynamics would have no such reference.</p>`;
      App.policyMap($('#mMap', el), { actions: d.policy_actions, mode: 'policy', selected: this.state, onSelect: s => { this.state = s; App.setData('ctl-mdp', { state: s }); this.load(el); }, tip: s => `<div class="r"><span>V<sub>π</sub>(s)</span><span>${f(d.V_pi_all[s], 3)}</span></div><div class="r"><span>V*(s)</span><span>${f(d.V_star_all[s], 3)}</span></div><div class="r"><span>π*(s)</span><span>${RL.ACTION_NAMES[d.optimal_actions[s]]}</span></div>` });
      App.data['ctl-mdp'] = Object.assign(App.data['ctl-mdp'] || {}, { detail: d, state: this.state });
      this.drawDetail(el, d); App.busy($('#mDetail', el), false); App.renderViva('ctl-mdp');
    },
    drawDetail(el, d) {
      const s = this.state, pa = d.per_action;
      const card = p => `<div class="acard ${p.action === d.best_action_index ? 'best' : ''}"><div class="row between">${App.act(p.action)}${p.action === d.best_action_index ? '<span class="tag">best under π*</span>' : ''}</div><div class="kv" style="margin-top:8px"><span>π(a|s)</span><span>${f(p.pi, 3)}</span><span>r(s,a) = E[R]</span><span>${f(p.expected_reward, 4)}</span><span>Q<sub>π</sub>(s,a)</span><span>${f(p.q_pi, 4)}</span><span>Q*(s,a)</span><span>${f(p.q_star, 4)}</span><span>P(session ends)</span><span>${f(p.terminal_probability, 3)}</span><span>sim. P(win)</span><span>${f(p.simulator.p_win, 2)}</span><span>sim. closeness</span><span>${f(p.simulator.closeness, 2)}</span></div></div>`;
      $('#mDetail', el).innerHTML = `
        <div class="panel hot"><div class="row between"><div class="row">${App.stateChip(s)}<span class="mono faint">s = ${s} · key ${d.state.key}</span></div><div class="row">${App.kpi(f(d.V_pi, 4), 'V<sub>π</sub>(s)')}${App.kpi(f(d.V_star, 4), 'V*(s)')}${App.kpi(App.act(d.best_action_index), 'π*(s)')}</div></div>
        <div class="grid g3" style="margin-top:12px">${pa.map(card).join('')}</div></div>
        <div class="grid g2" style="margin-top:16px">
          <div class="panel"><div class="row between"><h2>Transitions p(s′|s,a)</h2><div class="seg" id="mActSeg">${[0, 1, 2].map(a => `<button aria-pressed="${a === this.action}" data-a="${a}">${RL.ACTION_NAMES[a]}</button>`).join('')}</div></div><div id="mGraph"></div><div id="mTrans"></div></div>
          <div class="panel"><h2>Bellman expectation backup for V<sub>π</sub>(s)</h2><p class="sub">V<sub>π</sub>(s) = Σ<sub>a</sub> π(a|s) Σ<sub>s′</sub> p(s′|s,a)[r(s,a,s′) + γV<sub>π</sub>(s′)] with γ = ${d.gamma}. Every number below comes from the model and the converged V<sub>π</sub>.</p>${App.backupHtml(d.bellman_expectation, 'expectation', 'V<sub>π</sub>')}</div>
        </div>
        <div class="grid g2" style="margin-top:16px">
          <div class="panel"><h2>Bellman optimality backup for V*(s)</h2><p class="sub">V*(s) = max<sub>a</sub> Σ<sub>s′</sub> p(s′|s,a)[r + γV*(s′)]; the arg-max is the optimal action.</p>${App.backupHtml(d.bellman_optimality, 'optimality', 'V*')}</div>
          <div class="panel"><h2>The Markov chain induced by π</h2><p class="sub">Fixing π turns the MDP into a Markov chain P<sub>π</sub>(s′|s) = Σ<sub>a</sub>π(a|s)p(s′|s,a). Below: probability the session is still alive after t matches, starting from the session-start distribution.</p><div id="mChain"></div></div>
        </div>`;
      $$('#mActSeg button', el).forEach(b => b.addEventListener('click', () => { this.action = +b.dataset.a; $$('#mActSeg button', el).forEach(x => x.setAttribute('aria-pressed', String(x === b))); this.drawTrans(el, d); }));
      this.drawTrans(el, d); this.drawChain(el);
    },
    drawTrans(el, d) {
      const p = d.per_action[this.action]; const ns = p.next_states;
      const W = 560, rowH = 34, H = Math.max(120, ns.length * rowH + 16), cx = 80, cy = H / 2, tx = 330;
      let svg = `<svg viewBox="0 0 ${W} ${H}" class="graph" role="img" aria-label="Transition graph"><circle cx="${cx}" cy="${cy}" r="30" fill="#0e141e" stroke="#22d8f5" stroke-width="1.5"/><text x="${cx}" y="${cy + 4}" text-anchor="middle" style="fill:#e7eef6;font-weight:600">${RL.stateShort(this.state)}</text><text x="${cx}" y="${cy + 46}" text-anchor="middle" style="fill:${App.ACOL[this.action] === '#4b5b70' ? '#a9b6c6' : App.ACOL[this.action]}">${RL.ACTION_NAMES[this.action]}</text>`;
      ns.forEach((n, i) => {
        const y = 8 + rowH * i + rowH / 2; const end = n.short === 'END', other = n.next_state === null; const col = end ? '#ff4d6a' : other ? '#6c7b8e' : '#2f9e5e';
        svg += `<path d="M${cx + 30},${cy} C${cx + 140},${cy} ${tx - 120},${y} ${tx},${y}" fill="none" stroke="${col}" stroke-width="${Math.max(1, n.p * 16)}" opacity=".7"/>`;
        svg += `<rect x="${tx}" y="${y - 12}" width="120" height="24" rx="6" fill="#0e141e" stroke="${col}"/><text x="${tx + 60}" y="${y + 4}" text-anchor="middle" style="fill:#e7eef6">${esc(other ? n.label : n.short)}</text><text x="${tx + 132}" y="${y + 4}" style="fill:${col}">p = ${f(n.p, 3)}</text>`;
      });
      $('#mGraph', el).innerHTML = svg + '</svg>';
      $('#mTrans', el).innerHTML = App.table([{ h: 'S′', f: n => n.next_state === null ? `<span class="faint">${esc(n.label)}</span>` : `<span class="mono">${n.short}</span> <span class="faint">${esc(n.label)}</span>` }, { h: 'p̂(s′|s,a)', n: 1, f: n => `<div class="row" style="justify-content:flex-end;gap:6px"><div class="pbar"><i style="width:${n.p * 100}%"></i></div>${f(n.p, 4)}</div>` }, { h: 'r̂(s,a,s′)', n: 1, f: n => n.r === null ? '' : f(n.r, 3) }, { h: 'count', n: 1, f: n => n.count ?? '' }], ns);
    },
    async drawChain(el) {
      const c = await App.api('GET', `/rl/mdp/markov-chain?policy=${App.val(el, 'mPol')}&gamma=${App.val(el, 'mGamma')}&samples=${App.val(el, 'mSamples')}&seed=${App.val(el, 'mSeed')}&horizon=25`);
      C.line($('#mChain', el), { series: [{ name: 'P(still playing)', points: c.prob_session_alive.map((p, t) => [t, p]), markers: true }], xLabel: 'matches played t', yLabel: 'P(session alive)', yMin: 0, yMax: 1, height: 210, fmtX: v => v.toFixed(0), fmtTip: v => App.pct(v, 1) });
      $('#mChain', el).insertAdjacentHTML('beforeend', `<p class="sub" style="margin-top:6px">Expected matches within 25: <span class="mono">${f(c.expected_matches_within_horizon, 2)}</span> · closed-form MRP value at this state v = (I − γP<sub>π</sub>)⁻¹R<sub>π</sub>: <span class="mono">${f(c.V_closed_form[this.state], 4)}</span></p>`);
    },
  };
  App.viva['ctl-mdp'] = d => {
    const det = d.detail; let ex = null;
    if (det) { const b = det.bellman_expectation; ex = `<span class="calc">State ${det.state.short} (${esc(det.state.label)}), γ = ${det.gamma}: ${b.per_action.map(p => `q<sub>π</sub>(s,${p.action_name}) = ${f(p.q, 3)}`).join(', ')};  V<sub>π</sub>(s) = ${b.per_action.map(p => `${f(b.expectation.pi[p.action], 2)}×${f(p.q, 3)}`).join(' + ')} = <span class="res">${f(b.expectation.value, 4)}</span>;  V*(s) = ${f(det.V_star, 4)} via ${det.best_action}.</span>`; }
    return { unit: 'Unit 3', unitClass: 'u3', concept: 'Finite MDP, value functions and Bellman equations', prov: 'MODEL ESTIMATE',
      why: 'Writing the controller as ⟨S, A, P, R, γ⟩ makes the problem solvable by planning (Unit 4) and gives a ground truth for learning methods (Unit 5, Q-learning).',
      inputs: 'Simulation-estimated p̂(s′|s,a), r̂(s,a,s′); a policy π; γ.', outputs: 'V<sub>π</sub>(s), Q<sub>π</sub>(s,a), V*(s), Q*(s,a), π*(s), P<sub>π</sub>.',
      formula: 'V<sub>π</sub>(s) = Σ<sub>a</sub> π(a|s) Σ<sub>s′</sub> p(s′|s,a)[r(s,a,s′) + γV<sub>π</sub>(s′)];   Q<sub>π</sub>(s,a) = Σ<sub>s′</sub> p(s′|s,a)[r + γV<sub>π</sub>(s′)];   V*(s) = max<sub>a</sub> Q*(s,a)',
      example: ex, assumptions: 'Markov property holds by construction of the simulator in the bucketed state; the model is estimated, not observed from players.', code: 'mdp_model.py: estimate_model, FiniteMDP · dynamic_programming.backup_breakdown · rl_api /rl/mdp/state' };
  };

  // =================================================================== DYNAMIC PROGRAMMING (Unit 4)
  App.screens['ctl-dp'] = {
    render(el) {
      el.innerHTML = `${App.header('Dynamic programming lab', 'Model-based planning on the 48-state session MDP: iterative policy evaluation, policy improvement, policy iteration and value iteration, with every sweep recorded.', [['u4', 'Unit 4 · DP'], ['u3', 'Bellman optimality'], ['u1', 'γ']])}
      <div class="panel ctl" style="margin-bottom:16px"><div class="grid g3" style="grid-template-columns:repeat(5,minmax(0,1fr))">
        ${App.sel('dGamma', 'γ', App.opts([[0, '0'], [0.5, '0.5'], [0.7, '0.7'], [0.9, '0.9'], [0.95, '0.95'], [0.99, '0.99']], 0.9))}
        ${App.sel('dTheta', 'θ (stop when max Δ < θ)', App.opts([[1e-2, '1e-2'], [1e-4, '1e-4'], [1e-6, '1e-6'], [1e-8, '1e-8']], 1e-6))}
        ${App.num('dMax', 'Max iterations', 1000, 1, 20000)}
        ${App.sel('dSamples', 'Model samples/(s,a)', App.opts([[300, '300'], [1000, '1000'], [3000, '3000']], 1000))}
        ${App.num('dSeed', 'Model seed', 7, 0, 99999)}</div></div>
      <div class="tabs" id="dTabs" role="tablist"></div>
      <div id="dPane"></div>`;
      this.explain = RL.encode(3, 2, 0);
      App.tabs($('#dTabs', el), ['Policy evaluation', 'Policy iteration', 'Value iteration', 'γ sweep'], i => this.show(el, i), 2);
      ['dGamma', 'dTheta', 'dMax', 'dSamples', 'dSeed'].forEach(id => $('#' + id, el).addEventListener('change', () => this.show(el, this.tab)));
      this.show(el, 2);
      document.addEventListener('engine-change', () => { if (App.rendered['ctl-dp']) this.show(el, this.tab); });
    },
    body(el, extra = {}) { return Object.assign({ gamma: +App.val(el, 'dGamma'), theta: +App.val(el, 'dTheta'), max_iter: App.val(el, 'dMax'), samples: +App.val(el, 'dSamples'), seed: App.val(el, 'dSeed'), snapshots: 60, explain_state: this.explain }, extra); },
    async show(el, i) {
      this.tab = i; const p = $('#dPane', el);
      if (i === 0) {
        p.innerHTML = `<div class="grid g-side"><div class="panel ctl"><h2>Iterative policy evaluation</h2><p class="sub">Computes V<sub>π</sub> for a fixed policy by repeated Bellman expectation backups (synchronous sweeps).</p>${App.sel('dPol', 'Policy π', App.policyOptions('random'))}<button class="btn primary" id="dRun">Evaluate π</button></div>
          <div class="stack"><div id="dPeK" class="kpis"></div><div class="panel"><h2>max<sub>s</sub>|V<sub>k+1</sub>(s) − V<sub>k</sub>(s)| per sweep</h2><div id="dPeC"></div></div></div></div>
          <div class="grid g2" style="margin-top:16px"><div class="panel"><h2>V<sub>k</sub> after sweep <span id="dSnapK" class="mono">—</span></h2>${App.slider('dSnap', 'Sweep', 1, 60, 1, 60, v => v)}<div id="dSnapMap"></div></div><div class="panel"><h2>Backup at the selected state</h2><div id="dPeB"></div></div></div>`;
        $('#dRun', p).addEventListener('click', e => App.run(e.currentTarget, () => this.pe(el)));
        $('#dPol', p).addEventListener('change', () => $('#dRun', p).click()); $('#dRun', p).click();
      } else if (i === 1) {
        p.innerHTML = `<div class="grid g-side"><div class="panel ctl"><h2>Policy iteration</h2><p class="sub">Start from KEEP everywhere; alternate full policy evaluation and greedy improvement until no state changes (policy-stable).</p><button class="btn primary" id="dRun">Run policy iteration</button></div><div class="stack"><div id="dPiK" class="kpis"></div><div class="panel"><h2>Iterations</h2><div id="dPiT"></div></div></div></div>
          <div class="panel" style="margin-top:16px"><div class="row between"><h2>Policy entering iteration <span id="dPiIt" class="mono"></span></h2><span class="faint" style="font-size:12px">outlined cells change in the next improvement step</span></div>${App.slider('dPiS', 'Iteration', 1, 1, 1, 1, v => v)}<div id="dPiMap"></div></div>`;
        $('#dRun', p).addEventListener('click', e => App.run(e.currentTarget, () => this.pi(el))); $('#dRun', p).click();
      } else if (i === 2) {
        p.innerHTML = `<div class="grid g-side"><div class="panel ctl"><h2>Value iteration</h2><p class="sub">One Bellman optimality backup per sweep: V<sub>k+1</sub>(s) = max<sub>a</sub> Σ p[r + γV<sub>k</sub>(s′)]. Then π*(s) = argmax.</p><button class="btn primary" id="dRun">Run value iteration</button></div>
          <div class="kpis" id="dViK" style="align-content:start"></div></div>
          <div class="grid g2" style="margin-top:16px"><div class="panel"><h2>Convergence: max Δ per sweep</h2><div id="dViC"></div></div><div class="panel"><h2>Contraction ratio Δ<sub>k+1</sub>/Δ<sub>k</sub></h2><p class="sub">Banach fixed-point view: the Bellman optimality operator is a γ-contraction, so the ratio never exceeds γ and {V<sub>k</sub>} is a Cauchy sequence.</p><div id="dViR"></div></div></div>
          <div class="grid g2" style="margin-top:16px"><div class="panel"><h2>Optimal policy π* ${App.prov('MODEL ESTIMATE')}</h2><p class="sub">Amber dots: Q-learning picks a different action. Click a cell to see its optimality backup.</p><div id="dViMap"></div></div><div class="panel"><h2>V*(s)</h2><div id="dViV"></div></div></div>
          <div class="grid g2" style="margin-top:16px"><div class="panel"><h2>Bellman optimality backup at <span id="dViS" class="mono"></span></h2><div id="dViB"></div></div><div class="panel"><h2>DP (model-based) vs Q-learning (model-free)</h2><div id="dViCmp"></div></div></div>`;
        $('#dRun', p).addEventListener('click', e => App.run(e.currentTarget, () => this.vi(el))); $('#dRun', p).click();
      } else {
        p.innerHTML = `<div class="panel"><div class="row between"><h2>How the discount factor changes π*</h2><button class="btn primary" id="dRun">Run γ sweep</button></div><p class="sub">Value iteration for each γ on the same model; rates measured on the same 300 seeded session-start episodes.</p><div id="dGs" class="grid" style="grid-template-columns:repeat(4,minmax(0,1fr))"></div></div>
          <div class="grid g2" style="margin-top:16px"><div class="panel"><h2>Actions chosen by π* (count of 48 states)</h2><div id="dGsC"></div></div><div class="panel"><h2>Simulated rage-quit rate under π*</h2><div id="dGsR"></div></div></div><div class="panel" style="margin-top:16px"><div id="dGsT"></div></div>`;
        $('#dRun', p).addEventListener('click', e => App.run(e.currentTarget, () => this.sweep(el))); $('#dRun', p).click();
      }
    },
    async pe(el) {
      const r = await App.api('POST', '/rl/dp/policy-evaluation', this.body(el, { policy: App.val(el, 'dPol') })); App.setData('ctl-dp', { pe: r, mode: 'pe' });
      $('#dPeK', el).innerHTML = App.kpi(r.iterations, 'sweeps', r.converged ? 'converged' : 'hit max iterations', r.converged ? 'good' : 'warn') + App.kpi(f(r.max_abs_diff_vs_closed_form, 8), 'max |V − (I−γP<sub>π</sub>)⁻¹R<sub>π</sub>|', 'vs closed-form MRP') + App.kpi(f(r.bellman_residual, 8), 'Bellman residual') + App.kpi((r.compute.total_operations / 1e6).toFixed(2) + 'M', 'multiply-adds', `${f(r.seconds * 1000, 1)} ms`);
      C.line($('#dPeC', el), { series: [{ name: 'max Δ', points: r.deltas.map((d, k) => [k + 1, Math.max(d, 1e-14)]) }], logY: true, xLabel: 'sweep k', yLabel: 'max Δ (log)', refLines: [{ y: +App.val(el, 'dTheta'), label: 'θ', color: '#ffb020' }], height: 220, fmtX: v => v.toFixed(0) });
      const sl = $('#dSnap', el); sl.max = r.snapshots.length; sl.value = r.snapshots.length; const vmin = Math.min(...r.V), vmax = Math.max(...r.V);
      const drawSnap = () => { const k = +sl.value; $('#dSnapK', el).textContent = k; $('#dSnap-v', el).textContent = k; App.policyMap($('#dSnapMap', el), { values: r.snapshots[k - 1], mode: 'value', vmin, vmax, legend: 'V_k(s)', selected: this.explain, onSelect: s => { this.explain = s; this.pe(el); } }); };
      sl.oninput = drawSnap; drawSnap();
      $('#dPeB', el).innerHTML = `<div class="row" style="margin-bottom:8px">${App.stateChip(this.explain)}</div>` + App.backupHtml(r.backup_example, 'expectation', 'V<sub>π</sub>');
    },
    async pi(el) {
      const r = await App.api('POST', '/rl/dp/policy-iteration', this.body(el)); App.setData('ctl-dp', { pi: r, mode: 'pi' });
      $('#dPiK', el).innerHTML = App.kpi(r.improvement_steps, 'improvement steps') + App.kpi(r.evaluation_sweeps_total, 'evaluation sweeps') + App.kpi(r.policy_stable ? 'yes' : 'no', 'policy stable', '', r.policy_stable ? 'good' : 'warn') + App.kpi(f(r.seconds * 1000, 1) + ' ms', 'time');
      $('#dPiT', el).innerHTML = App.table([{ h: 'Iteration', n: 1, f: h => h.iteration }, { h: 'Evaluation sweeps', n: 1, f: h => h.evaluation_sweeps }, { h: 'States changed by improvement', n: 1, f: h => h.n_changed }, { h: 'Mean V<sub>π</sub>', n: 1, f: h => f(h.V.reduce((a, b) => a + b, 0) / 48, 3) }, { h: 'Policy (L/K/R)', f: h => { const c = [0, 0, 0]; h.policy.forEach(a => c[a]++); return c.join(' / '); } }], r.history);
      const sl = $('#dPiS', el); sl.max = r.history.length; sl.value = 1;
      const draw = () => { const h = r.history[+sl.value - 1]; $('#dPiIt', el).textContent = h.iteration; $('#dPiS-v', el).textContent = sl.value; App.policyMap($('#dPiMap', el), { actions: h.policy, mode: 'policy', changed: new Set(h.changed_states), tip: s => `<div class="r"><span>V<sub>π</sub>(s)</span><span>${f(h.V[s], 3)}</span></div>` }); };
      sl.oninput = draw; draw();
    },
    async vi(el) {
      const r = await App.api('POST', '/rl/dp/value-iteration', this.body(el)); App.setData('ctl-dp', { vi: r, mode: 'vi' }); const g = +App.val(el, 'dGamma');
      const c = [0, 0, 0]; r.actions.forEach(a => c[a]++);
      $('#dViK', el).innerHTML = App.kpi(r.iterations, 'sweeps to θ', r.converged ? 'converged' : 'not converged', r.converged ? 'good' : 'warn') + App.kpi(f(r.max_contraction_ratio, 4), 'max Δ ratio', `≤ γ = ${g}`) + App.kpi(f(r.bellman_residual, 8), 'Bellman residual') + App.kpi(`${c[0]}/${c[1]}/${c[2]}`, 'LOWER / KEEP / RAISE') + App.kpi(f(r.seconds * 1000, 1) + ' ms', 'time', `${(r.compute.total_operations / 1e6).toFixed(2)}M ops`);
      C.line($('#dViC', el), { series: [{ name: 'max Δ', points: r.deltas.map((d, k) => [k + 1, Math.max(d, 1e-14)]) }], logY: true, xLabel: 'sweep k', yLabel: 'max Δ (log)', refLines: [{ y: +App.val(el, 'dTheta'), label: 'θ', color: '#ffb020' }], height: 220, fmtX: v => v.toFixed(0) });
      C.line($('#dViR', el), { series: [{ name: 'Δₖ₊₁/Δₖ', points: r.contraction_ratios.map((x, k) => [k + 1, x]) }], refLines: [{ y: g, label: `γ = ${g}`, color: '#ff8de0' }], xLabel: 'sweep k', yLabel: 'ratio', yMin: 0, yMax: Math.max(1, ...r.contraction_ratios), height: 220, fmtX: v => v.toFixed(0) });
      const sel = s => { this.explain = s; this.vi(el); };
      App.policyMap($('#dViMap', el), { actions: r.actions, mode: 'policy', diffWith: r.q_learning_actions, selected: this.explain, onSelect: sel, tip: s => [0, 1, 2].map(a => `<div class="r"><span>Q*(s,${RL.ACTION_NAMES[a]})</span><span>${f(r.Q[s][a], 3)}</span></div>`).join('') });
      App.policyMap($('#dViV', el), { values: r.V, actions: r.actions, mode: 'value', legend: 'V*(s)', selected: this.explain, onSelect: sel });
      $('#dViS', el).textContent = RL.stateShort(this.explain);
      $('#dViB', el).innerHTML = App.backupHtml(r.backup_example, 'optimality', 'V*');
      $('#dViCmp', el).innerHTML = `<div class="kpis">${App.kpi(App.pct(r.agreement_with_q_learning, 1), 'policy agreement')}${App.kpi(r.disagreements_with_q_learning.length, 'states that differ')}</div>
        <div class="grid g2" style="margin-top:12px;font-size:12.5px"><div><b>DP — model-based</b><p class="dim">Needs p(s′|s,a) and r(s,a,s′); plans by sweeping all 48 states; exact up to model error; milliseconds.</p></div><div><b>Q-learning — model-free</b><p class="dim">Needs only sampled transitions; learns from ~180k simulated steps (30,000 episodes); noisy in near-tied or rarely visited states.</p></div></div>
        ${r.disagreements_with_q_learning.length ? App.table([{ h: 'State', f: d => `<span class="mono">${d.short}</span>` }, { h: 'DP', f: d => App.act(RL.ACTION_NAMES.indexOf(d.reference_action)) }, { h: 'Q-learning', f: d => App.act(RL.ACTION_NAMES.indexOf(d.action)) }, { h: 'value lost q*(DP) − q*(Q)', n: 1, f: d => f(d.q_gap, 4) }], r.disagreements_with_q_learning, { scroll: true }) : ''}`;
    },
    async sweep(el) {
      const gs = [0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99];
      const r = await App.api('POST', '/rl/dp/gamma-sweep', { gammas: gs, samples: +App.val(el, 'dSamples'), seed: App.val(el, 'dSeed') }); App.setData('ctl-dp', { sweep: r, mode: 'sweep' });
      const box = $('#dGs', el); box.innerHTML = r.rows.map((x, i) => `<div><div id="gsm${i}"></div></div>`).join('');
      r.rows.forEach((x, i) => App.miniMap($('#gsm' + i, el), x.actions, i ? r.rows[i - 1].actions : null, `<span>γ = ${x.gamma}</span><span class="mono">${x.counts.join('/')}</span>`));
      C.bars($('#dGsC', el), { categories: r.rows.map(x => 'γ=' + x.gamma), series: [0, 1, 2].map(a => ({ name: RL.ACTION_NAMES[a], values: r.rows.map(x => x.counts[a]), color: a === 1 ? '#7d8da3' : App.ACOL[a] })), yLabel: 'states', height: 220, fmt: v => v.toFixed(0) });
      C.line($('#dGsR', el), { series: [{ name: 'rage-quit rate', points: r.rows.map(x => [x.gamma, x.sim_rage_quit_rate]), markers: true }], xLabel: 'γ', yLabel: 'rate', height: 220, fmtY: v => Math.round(v * 100) + '%', fmtTip: v => App.pct(v, 1) });
      $('#dGsT', el).innerHTML = App.table([{ h: 'γ', n: 1, f: x => x.gamma }, { h: 'LOWER / KEEP / RAISE', f: x => x.counts.join(' / ') }, { h: 'sweeps', n: 1, f: x => x.iterations }, { h: 'V* from session start', n: 1, f: x => f(x.V_session_start, 3) }, { h: 'sim. session length', n: 1, f: x => f(x.sim_session_length, 2) }, { h: 'sim. rage quit', n: 1, f: x => App.pct(x.sim_rage_quit_rate, 1) }, { h: 'sim. close-match rate', n: 1, f: x => App.pct(x.sim_close_match_rate, 1) }], r.rows) + '<p class="sub" style="margin-top:8px">Amber outlines in the small maps mark states whose action changed from the previous γ. A myopic controller (small γ) chases close matches now; a far-sighted one lowers difficulty more often to protect tilt and keep the session alive.</p>';
    },
  };
  App.viva['ctl-dp'] = d => {
    let ex = null;
    const b = (d.vi && d.vi.backup_example) || (d.pe && d.pe.backup_example);
    if (d.mode === 'vi' && d.vi) ex = `<span class="calc">State ${RL.stateShort(b.state)} after ${d.vi.iterations} sweeps: ${b.per_action.map(p => `q*(s,${p.action_name}) = r̄ ${f(p.expected_reward, 3)} + γ·E[V*(s′)] ${f(p.expected_next_value, 3)}×${b.gamma} = ${f(p.q, 4)}`).join(';  ')}  ⇒ V*(s) = <span class="res">${f(b.optimality.max_q, 4)}</span>, π*(s) = ${RL.ACTION_NAMES[b.optimality.argmax]}</span>`;
    else if (d.mode === 'pe' && d.pe) ex = `<span class="calc">V<sub>π</sub>(${RL.stateShort(b.state)}) = ${b.per_action.map(p => `${f(b.expectation.pi[p.action], 2)}×${f(p.q, 3)}`).join(' + ')} = <span class="res">${f(b.expectation.value, 4)}</span>; converged in ${d.pe.iterations} sweeps; closed-form gap ${f(d.pe.max_abs_diff_vs_closed_form, 8)}</span>`;
    else if (d.mode === 'pi' && d.pi) ex = `<span class="calc">${d.pi.history.map(h => `iter ${h.iteration}: ${h.evaluation_sweeps} eval sweeps, ${h.n_changed} states improved`).join('; ')} ⇒ policy-stable.</span>`;
    else if (d.sweep) ex = `<span class="calc">${d.sweep.rows.map(x => `γ=${x.gamma}: ${x.counts.join('/')}`).join(' · ')}</span>`;
    return { unit: 'Unit 4', unitClass: 'u4', concept: d.mode === 'pe' ? 'Iterative policy evaluation' : d.mode === 'pi' ? 'Policy iteration' : 'Value iteration', prov: 'MODEL ESTIMATE',
      why: 'Find the optimal session-control policy when the transition model is known — here, the simulation-estimated model.',
      inputs: 'p̂(s′|s,a), r̂(s,a,s′), γ, θ, max iterations.', outputs: 'V<sub>π</sub> or V*, π*, max-Δ per sweep, contraction ratios, operation counts.',
      formula: 'Evaluation: V<sub>k+1</sub>(s) = Σ<sub>a</sub>π(a|s)Σ<sub>s′</sub>p(s′|s,a)[r + γV<sub>k</sub>(s′)];  Improvement: π′(s) = argmax<sub>a</sub>Σ<sub>s′</sub>p[r + γV<sub>π</sub>(s′)];  Value iteration: V<sub>k+1</sub>(s) = max<sub>a</sub>Σ<sub>s′</sub>p[r + γV<sub>k</sub>(s′)]; stop when max<sub>s</sub>|V<sub>k+1</sub> − V<sub>k</sub>| < θ',
      example: ex, assumptions: 'Full knowledge of the (estimated) model; synchronous sweeps; terminal value fixed at 0.', code: 'dynamic_programming.py: policy_evaluation, policy_improvement, policy_iteration, value_iteration' };
  };
})(window);
