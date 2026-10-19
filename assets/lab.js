/* Clutch IQ — RL lab: Bandit arena (U2), Monte Carlo lab (U5), Algorithm comparison. */
(function (root) {
  'use strict';
  const App = root.App, RL = root.ClutchRL, C = root.Charts, esc = C.esc, $ = App.$, $$ = App.$$;
  const f = App.f;
  const ALG = { random: 'Explore only', greedy: 'Greedy', explore_first: 'Fixed exploration', epsilon_greedy: 'ε-greedy', optimistic: 'Optimistic greedy', ucb: 'UCB' };
  const ALG_COL = { greedy: '#0aa5c4', epsilon_greedy: '#c43fb0', optimistic: '#2f9e5e', ucb: '#7f6ff0', explore_first: '#c98500', random: '#3d78e0' };

  // =================================================================== BANDIT ARENA (Unit 2)
  App.screens['lab-bandit'] = {
    render(el) {
      const tArms = (App.data['lab-bandit'] || {}).arms;
      el.innerHTML = `${App.header('Bandit arena', 'Choosing the map for one match is a one-step decision: a multi-armed bandit. Each map is an arm; each match returns a satisfaction reward. Compare how exploration strategies trade information for reward on the same seeded environments.', [['u2', 'Unit 2 · bandits'], ['u1', 'Immediate RL']])}
      <div class="note" style="margin-bottom:16px">Why not the MDP here? A map pick does not change the player's next state the way lobby difficulty changes tilt and fatigue, so there is no delayed consequence to plan for — the bandit maximises E[R | A = map] directly. Bandit results are never placed on the sequential-RL leaderboard.</div>
      <div class="grid g-side">
        <div class="panel ctl"><h2>Arena setup</h2>
          ${App.sel('bSrc', 'Arms', App.opts([['sample_maps', 'Sample map pool · demo roster'], ...(tArms ? [['tournament', 'Tournament match ' + esc(tArms.label || '')]] : []), ['custom', 'Custom arm means']], tArms ? 'tournament' : 'sample_maps'))}
          <div id="bCustom" class="field hidden"><label for="bMeans">Arm means q*(a) in [0,1], comma-separated</label><input type="text" id="bMeans" value="0.3, 0.5, 0.7, 0.45, 0.6"></div>
          ${App.slider('bK', 'Number of arms (maps)', 2, 20, 1, 10, v => v)}
          <div><div class="lbl" style="margin-bottom:4px">Algorithms (same seeds for all)</div><div class="checks">${Object.entries(ALG).map(([k, n]) => `<label><input type="checkbox" value="${k}" ${['greedy', 'epsilon_greedy', 'optimistic', 'ucb'].includes(k) ? 'checked' : ''}> ${n}</label>`).join('')}</div></div>
          <div class="grid g2" style="gap:8px">${App.num('bEps', 'ε', 0.1, 0, 1, 0.01)}${App.num('bC', 'UCB c', 1.414, 0, 10, 0.1)}${App.num('bQ0', 'Optimistic Q₀', 1.0, -5, 10, 0.1)}${App.num('bExp', 'Exploration rounds', 50, 0, 5000)}${App.num('bRounds', 'Rounds T', 1000, 10, 10000, 10)}${App.num('bRuns', 'Runs (averaged)', 50, 1, 500)}${App.num('bSeed', 'Seed', 1, 0, 999999)}${App.num('bAlpha', 'Constant α (0 = 1/n)', 0, 0, 1, 0.01)}</div>
          <div class="grid g2" style="gap:8px">${App.sel('bModel', 'Reward model', App.opts([['bernoulli', 'Bernoulli thumbs-up'], ['gaussian', 'Gaussian quality score']], 'bernoulli'))}${App.sel('bNs', 'Stationarity', App.opts([['none', 'Stationary'], ['drift', 'Meta drift (random walk)'], ['patch', 'Balance patch at T/2']], 'none'))}</div>
          <button class="btn primary" id="bRun">Run arena</button>
        </div>
        <div class="stack">
          <div class="panel"><div class="row between"><h2>Arms ${App.prov('DERIVED FEATURE')}</h2><span id="bArmSrc" class="faint" style="font-size:12px"></span></div><p class="sub">q*(map) is derived from the tournament builder's map-fit score; each pull's reward is simulated around it. Agents never see q*.</p><div id="bArms"></div></div>
          <div class="kpis" id="bK2"></div>
        </div>
      </div>
      <div class="grid g2" style="margin-top:16px">
        <div class="panel"><h2>Mean cumulative regret Σ (q*<sub>max</sub> − q*(A<sub>t</sub>))</h2><div id="bcReg"></div></div>
        <div class="panel"><h2>% optimal arm selected</h2><div id="bcOpt"></div></div>
        <div class="panel"><h2>Mean cumulative reward</h2><div id="bcRew"></div></div>
        <div class="panel"><h2>Exploration rate</h2><p class="sub">Share of rounds that were explorations (random pick, forced round-robin, or a UCB bonus overriding the greedy arm).</p><div id="bcExp"></div></div>
      </div>
      <div class="panel" style="margin-top:16px"><h2>Summary ${App.prov('SIMULATED')}</h2><div id="bSum"></div><div id="bRec" style="margin-top:12px"></div></div>
      <div class="panel" style="margin-top:16px"><div class="row between"><h2>Inside one run</h2>${App.sel('bAlg', '', '')}</div>
        <div class="grid g2"><div><h3>Estimated Q<sub>T</sub>(a) vs true q*(a)</h3><div id="biQ"></div></div><div><h3>Pulls N<sub>T</sub>(a)</h3><div id="biN"></div></div></div>
        <h3>Selected arm, first rounds <span class="faint" style="font-weight:400">(brighter = better arm; white = optimal)</span></h3><div id="biStrip"></div>
        <div class="grid g2" style="margin-top:12px"><div><h3>Incremental update at round <span id="biT" class="mono"></span></h3>${App.slider('biSl', 'Round t', 1, 400, 1, 1, v => v)}<div id="biUpd"></div></div><div><h3>UCB decision at the last round</h3><div id="biUcb"></div></div></div>
      </div>`;
      const src = $('#bSrc', el); const syncSrc = () => { $('#bCustom', el).classList.toggle('hidden', src.value !== 'custom'); };
      src.addEventListener('change', syncSrc); syncSrc(); App.bindSlider(el, 'bK', v => v);
      $('#bRun', el).addEventListener('click', e => App.run(e.currentTarget, () => this.run(el)));
      $('#bRun', el).click();
      document.addEventListener('engine-change', () => { if (App.rendered['lab-bandit']) $('#bRun', el).click(); });
    },
    async run(el) {
      const algs = $$('.checks input:checked', el).map(i => i.value); if (!algs.length) throw new Error('Pick at least one algorithm.');
      const src = App.val(el, 'bSrc'); let arms = { source: src, k: App.val(el, 'bK') };
      if (src === 'custom') { const means = App.val(el, 'bMeans').split(',').map(x => parseFloat(x)).filter(x => isFinite(x)); if (means.length < 2 || means.some(m => m < 0 || m > 1)) throw new Error('Enter at least two arm means between 0 and 1.'); arms.means = means; }
      if (src === 'tournament') { const t = App.data['lab-bandit'].arms; arms = Object.assign({}, t, { k: App.val(el, 'bK') }); if (App.engine === 'python') { delete arms.map_pool; delete arms.roster_archetypes; } }
      const alpha = App.val(el, 'bAlpha');
      const body = { algorithms: algs, rounds: App.val(el, 'bRounds'), runs: App.val(el, 'bRuns'), seed: App.val(el, 'bSeed'), epsilon: App.val(el, 'bEps'), c: App.val(el, 'bC'), q0: App.val(el, 'bQ0'), alpha: alpha > 0 ? alpha : null, explore_rounds: App.val(el, 'bExp'), reward_model: App.val(el, 'bModel'), sigma: 0.1, nonstationary: App.val(el, 'bNs'), drift_sigma: 0.01, arms };
      const r = await App.api('POST', '/bandit/run', body); this.r = r; App.setData('lab-bandit', { run: r, body });
      const best = r.means.indexOf(Math.max(...r.means));
      $('#bArmSrc', el).textContent = r.roster_source;
      $('#bArms', el).innerHTML = App.table([{ h: '#', n: 1, f: (a, i) => i + 1 }, { h: 'Map (arm)', f: (a, i) => `${esc(a.map_name)}${i === best ? ' <span class="tag">optimal</span>' : ''}` }, { h: 'Fit score', n: 1, f: a => a.fit_score === null ? '—' : f(a.fit_score, 3) }, { h: 'q*(a)', n: 1, f: a => `<div class="row" style="justify-content:flex-end;gap:6px"><div class="pbar"><i style="width:${a.q_star * 100}%"></i></div>${f(a.q_star, 3)}</div>` }], r.arms, { scroll: true });
      const res = r.results; const keys = Object.keys(res);
      const winner = keys.reduce((a, b) => res[a].final_regret < res[b].final_regret ? a : b);
      $('#bK2', el).innerHTML = App.kpi(ALG[winner], 'lowest regret', `${f(res[winner].final_regret, 1)} after ${r.rounds} rounds`) + App.kpi(`${r.runs} × ${r.rounds}`, 'runs × rounds', `seed ${r.seed}`) + App.kpi(f(r.seconds, 2) + ' s', 'compute', App.engine === 'python' ? 'Python' : 'browser');
      const ser = (fn) => keys.map(k => ({ name: ALG[k], color: ALG_COL[k], points: res[k].rounds_index.map((t, i) => [t, fn(res[k], i)]) }));
      C.line($('#bcReg', el), { series: ser((x, i) => x.mean_cumulative_regret[i]), xLabel: 'round t', yLabel: 'regret', height: 230 });
      C.line($('#bcOpt', el), { series: ser((x, i) => x.pct_optimal[i]), xLabel: 'round t', yLabel: '% optimal', yMin: 0, yMax: 1, height: 230, fmtY: v => Math.round(v * 100) + '%', fmtTip: v => App.pct(v, 1) });
      C.line($('#bcRew', el), { series: ser((x, i) => x.mean_cumulative_reward[i]), xLabel: 'round t', yLabel: 'Σ reward', height: 230 });
      C.line($('#bcExp', el), { series: ser((x, i) => x.explore_rate[i]), xLabel: 'round t', yLabel: 'explore rate', yMin: 0, yMax: 1, height: 230, fmtY: v => Math.round(v * 100) + '%', fmtTip: v => App.pct(v, 1) });
      $('#bSum', el).innerHTML = App.table([{ h: 'Algorithm', f: k => `<i style="display:inline-block;width:10px;height:10px;border-radius:2px;background:${ALG_COL[k]};margin-right:6px"></i>${ALG[k]}` }, { h: 'Expected regret', n: 1, f: k => f(res[k].final_regret, 1) }, { h: 'Total reward ± 95% CI', n: 1, f: k => `${f(res[k].final_reward, 1)} ± ${f(res[k].final_reward_ci95, 1)}` }, { h: '% optimal (last 10%)', n: 1, f: k => App.pct(res[k].final_pct_optimal, 1) }, { h: 'Explore / exploit (run 1)', n: 1, f: k => `${res[k].example_run.n_explore} / ${res[k].example_run.n_exploit}` }, { h: 'Realised regret T·q*max − ΣR (run 1)', n: 1, f: k => res[k].example_run.realised_regret === null ? 'n/a' : f(res[k].example_run.realised_regret, 1) }], keys, { rowClass: k => k === winner ? 'best' : '' });
      $('#bRec', el).innerHTML = (res.ucb ? `<p class="sub">UCB's exploration bonus c·√(ln t/N) is on the reward scale: with rewards in [0, 1], c = √2 (UCB1, as in the notes' examples) explores heavily; try c ≈ 0.3–0.5 to see it overtake ε-greedy.</p>` : '') + `<div class="note ok">Rule-based recommendation (highest fit): <b>${esc(r.rule_based_map)}</b> · bandit recommendation (greedy on learned Q of ${ALG[r.best_algorithm]}, run 1): <b>${esc(r.bandit_recommended_map)}</b>${r.rule_based_map === r.bandit_recommended_map ? ' — they agree.' : ' — the learner has not separated the top arms yet; more rounds or exploration would.'}</div>`;
      const sel = $('#bAlg', el); sel.innerHTML = keys.map(k => `<option value="${k}">${ALG[k]}</option>`).join(''); sel.onchange = () => this.inspect(el); this.inspect(el);
    },
    inspect(el) {
      const r = this.r, k = $('#bAlg', el).value, ex = r.results[k].example_run, means = ex.final_means, base = r.means;
      const cats = r.arms.map(a => a.map_name);
      C.bars($('#biQ', el), { categories: cats, series: [{ name: 'Q_T(a)', values: ex.Q, color: ALG_COL[k] }, { name: 'q*(a)', values: means, color: '#5b6b80' }], height: 220, rotate: cats.length > 6, fmt: v => v.toFixed(2) });
      C.bars($('#biN', el), { categories: cats, series: [{ name: 'N_T(a)', values: ex.N, color: ALG_COL[k] }], height: 220, rotate: cats.length > 6, fmt: v => v.toFixed(0) });
      const order = base.map((m, i) => [m, i]).sort((a, b) => a[0] - b[0]).map(x => x[1]); const rank = []; order.forEach((arm, j) => { rank[arm] = j / Math.max(1, order.length - 1); });
      const bi = base.indexOf(Math.max(...base));
      $('#biStrip', el).innerHTML = `<div class="strip">${ex.arms.map((a, t) => `<i title="t=${t + 1}: ${esc(cats[a])} → R=${ex.rewards[t].toFixed(2)}" style="background:${a === bi ? '#ffffff' : C.seqColor(0.15 + 0.75 * rank[a])}"></i>`).join('')}</div><div class="mini-cap"><span>t = 1</span><span>t = ${ex.arms.length}</span></div>`;
      const sl = $('#biSl', el); sl.max = ex.arms.length; if (+sl.value > ex.arms.length) sl.value = 1;
      const body = App.data['lab-bandit'].body; const alpha = body.alpha; const init = k === 'optimistic' ? body.q0 : 0;
      const draw = () => {
        const t = +sl.value; $('#biT', el).textContent = t; $('#biSl-v', el).textContent = t;
        const Q = cats.map(() => init), N = cats.map(() => 0); let last = null;
        for (let i = 0; i < t; i++) { const a = ex.arms[i], R = ex.rewards[i]; N[a]++; const step = alpha || 1 / N[a]; const old = Q[a]; Q[a] = old + step * (R - old); last = { a, R, old, step, n: N[a], nw: Q[a] }; }
        $('#biUpd', el).innerHTML = `<div class="eq"><span class="calc">A<sub>${t}</sub> = ${esc(cats[last.a])}, R<sub>${t}</sub> = ${f(last.R, 3)}, n = N(a) = ${last.n}<br>Q<sub>n+1</sub> = Q<sub>n</sub> + ${alpha ? 'α' : '(1/n)'}·(R − Q<sub>n</sub>) = ${f(last.old, 4)} + ${alpha ? alpha : '1/' + last.n}×(${f(last.R, 3)} − ${f(last.old, 4)}) = <span class="res">${f(last.nw, 4)}</span></span></div><p class="sub" style="margin-top:6px">${alpha ? 'Constant step: an exponential recency-weighted average (weights α(1−α)<sup>n−i</sup>) that tracks non-stationary arms.' : 'Sample average: identical to the mean of all rewards seen from this arm, computed in O(1) memory.'}</p>`;
      };
      sl.oninput = draw; draw();
      const u = ex.ucb_last;
      $('#biUcb', el).innerHTML = u ? `<div class="eq" style="margin-bottom:8px"><span class="calc">A<sub>t</sub> = argmax<sub>a</sub> [Q<sub>t</sub>(a) + c·√(ln t / N<sub>t</sub>(a))],  t = ${u.t}, c = ${body.c}</span></div>` + App.table([{ h: 'Arm', f: (x, i) => esc(cats[i]) + (i === u.chosen ? ' <span class="tag">chosen</span>' : '') }, { h: 'Q<sub>t</sub>(a)', n: 1, f: (x, i) => f(u.Q[i], 3) }, { h: 'N<sub>t</sub>(a)', n: 1, f: (x, i) => u.N[i] }, { h: 'bonus', n: 1, f: (x, i) => u.bonus[i] === null ? '∞' : f(u.bonus[i], 3) }, { h: 'score', n: 1, f: (x, i) => u.scores[i] === null ? '∞' : f(u.scores[i], 3) }], cats, { scroll: true, rowClass: (x, i) => i === u.chosen ? 'sel' : '' }) : '<div class="empty">Include UCB in the run and select it above.</div>';
    },
  };
  App.viva['lab-bandit'] = d => {
    let ex = null; const r = d.run;
    if (r && r.results.ucb && r.results.ucb.example_run.ucb_last) { const u = r.results.ucb.example_run.ucb_last; const i = u.chosen; ex = `<span class="calc">UCB at t=${u.t}: chosen arm ${esc(r.arms[i].map_name)} with Q=${f(u.Q[i], 3)} + c√(ln ${u.t}/${u.N[i]}) = ${f(u.Q[i], 3)} + ${f(u.bonus[i], 3)} = ${f(u.scores[i], 3)}.  ε-greedy expected regret ${r.results.epsilon_greedy ? f(r.results.epsilon_greedy.final_regret, 1) : '—'} vs greedy ${r.results.greedy ? f(r.results.greedy.final_regret, 1) : '—'}.</span>`; }
    else if (r) { const k = Object.keys(r.results)[0]; ex = `<span class="calc">${ALG[k]}: expected regret ${f(r.results[k].final_regret, 1)} after ${r.rounds} rounds.</span>`; }
    return { unit: 'Unit 2', unitClass: 'u2', concept: 'Multi-armed bandits: exploration vs exploitation', prov: 'SIMULATED',
      why: 'Map choice for one match is immediate RL: the reward is observed right after the action and the action does not change the next state.',
      inputs: 'Arms (maps) with hidden q*(a); algorithm and its parameters; seed.', outputs: 'Q<sub>t</sub>(a), N<sub>t</sub>(a), selected arms, cumulative reward and regret, % optimal.',
      formula: 'Q<sub>n+1</sub> = Q<sub>n</sub> + (1/n)(R<sub>n</sub> − Q<sub>n</sub>)  [or α];  ε-greedy: random arm w.p. ε, else argmax Q;  UCB: argmax Q<sub>t</sub>(a) + c√(ln t/N<sub>t</sub>(a));  regret = Σ<sub>t</sub> (max<sub>a</sub>q*(a) − q*(A<sub>t</sub>))',
      example: ex, assumptions: 'Satisfaction is Bernoulli(q*) per match; arm means come from the rule-based fit score (DERIVED); one roster per arena.', code: 'bandits.py: run_bandit, run_arena, map_arms, ucb_scores' };
  };

  // =================================================================== MONTE CARLO LAB (Unit 5)
  App.screens['lab-mc'] = {
    render(el) {
      el.innerHTML = `${App.header('Monte Carlo lab', 'Model-free learning from complete sessions: no transition probabilities are used. Estimate values by averaging returns, learn a policy by MC control, and estimate a new matchmaking policy\'s value from sessions logged under the old one.', [['u5', 'Unit 5 · MC prediction'], ['u5', 'MC control'], ['u5', 'Importance sampling']])}
      <div class="tabs" id="mcTabs" role="tablist"></div><div id="mcPane"></div>`;
      App.tabs($('#mcTabs', el), ['Prediction', 'Control', 'Off-policy evaluation'], i => this.show(el, i));
      this.sel = RL.encode(1, 1, 1); this.show(el, 0);
      document.addEventListener('engine-change', () => { if (App.rendered['lab-mc']) this.show(el, this.tab); });
    },
    show(el, i) {
      this.tab = i; const p = $('#mcPane', el); const logs = (App.data['lab-mc'] || {}).episodes_id;
      if (i === 0) {
        p.innerHTML = `<div class="grid g-side"><div class="panel ctl"><h2>MC prediction of V<sub>π</sub></h2>
          ${App.sel('pPol', 'Policy π to evaluate', App.policyOptions('heuristic'))}${App.slider('pEps', 'ε-soften π', 0, 1, 0.05, 0, v => v.toFixed(2))}
          <div class="grid g2" style="gap:8px">${App.sel('pN', 'Episodes', App.opts([[200, '200'], [1000, '1,000'], [3000, '3,000'], [10000, '10,000'], [20000, '20,000']], 3000))}${App.num('pSeed', 'Seed', 11, 0, 999999)}${App.sel('pGamma', 'γ', App.opts([[0.5, '0.5'], [0.9, '0.9'], [0.99, '0.99'], [1, '1.0']], 0.9))}${App.sel('pStart', 'Start states', App.opts([['uniform', 'all 48 (exploring)'], ['session_start', 'session start']], 'uniform'))}</div>
          <button class="btn primary" id="pRun">Run MC prediction</button>${logs ? `<button class="btn" id="pLogs">Use uploaded logs (${esc(logs)})</button>` : ''}</div>
          <div class="stack"><div class="kpis" id="pK"></div><div class="panel"><h2>Error vs DP as episodes accumulate ${App.prov('SIMULATED')}</h2><p class="sub">MAE between MC estimates and V<sub>π</sub> from DP policy evaluation on the estimated model, over visited states.</p><div id="pC"></div></div></div></div>
          <div class="grid g2" style="margin-top:16px"><div class="panel"><h2>First-visit V(s)</h2><div id="pV"></div></div><div class="panel"><h2>|V<sub>MC</sub> − V<sub>DP</sub>|</h2><div id="pE"></div></div></div>
          <div class="grid g2" style="margin-top:16px"><div class="panel"><h2>Visit counts N(s) — first-visit</h2><div id="pNmap"></div></div><div class="panel"><h2>One complete episode and its returns</h2><p class="sub">Returns are computed backwards from the terminal state: G<sub>t</sub> = R<sub>t+1</sub> + γG<sub>t+1</sub>, G<sub>T</sub> = 0. First-visit uses only the first G<sub>t</sub> of each state per episode; every-visit uses all.</p><div id="pEp"></div></div></div>`;
        $('#pRun', p).addEventListener('click', e => App.run(e.currentTarget, () => this.predict(el)));
        if (logs) $('#pLogs', p).addEventListener('click', e => App.run(e.currentTarget, () => this.predictLogs(el)));
        App.bindSlider(el, 'pEps', v => v.toFixed(2)); $('#pRun', p).click();
      } else if (i === 1) {
        p.innerHTML = `<div class="grid g-side"><div class="panel ctl"><h2>Monte Carlo control</h2>
          <div class="seg" id="cVar"><button aria-pressed="true" data-v="on_policy">On-policy ε-soft</button><button aria-pressed="false" data-v="es">Exploring starts</button><button aria-pressed="false" data-v="off_policy">Off-policy (weighted IS)</button></div>
          <div class="grid g2" style="gap:8px">${App.sel('cN', 'Episodes', App.opts([[2000, '2,000'], [5000, '5,000'], [10000, '10,000'], [20000, '20,000'], [40000, '40,000']], 10000))}${App.num('cSeed', 'Seed', 7, 0, 999999)}${App.num('cEps', 'ε (target ε-soft)', 0.1, 0.01, 1, 0.01)}${App.num('cBeh', 'Behaviour ε (off-policy)', 0.3, 0.01, 1, 0.01)}</div>
          <button class="btn primary" id="cRun">Run MC control</button><p class="sub" id="cDesc"></p></div>
          <div class="stack"><div class="kpis" id="cK"></div><div class="grid g2"><div class="panel"><h2>Greedy policy value J</h2><div id="cJ"></div></div><div class="panel"><h2>Agreement with DP π*</h2><div id="cA"></div></div></div></div></div>
          <div class="grid g2" style="margin-top:16px"><div class="panel"><h2>Learned greedy policy (amber = differs from DP)</h2><div id="cMap"></div></div><div class="panel"><h2>Q(s,·) and visit counts at <span id="cS" class="mono"></span></h2><div id="cQ"></div><h3>Disagreements with DP</h3><div id="cD"></div></div></div>`;
        const desc = { on_policy: 'Generalised policy iteration: after each episode, average first-visit returns into Q(s,a) and make the policy ε-greedy w.r.t. Q. Learns the best ε-soft policy.', es: 'Every episode starts from a random (state, action) pair — possible because the simulator can be reset anywhere — then follows the greedy policy.', off_policy: 'Behaviour b is ε-soft and explores; the target π is greedy. Returns are re-weighted by W = Π 1/b(A|S) and the update stops at the first non-greedy action (learning only from episode tails).' };
        this.variant = 'on_policy'; $('#cDesc', p).textContent = desc.on_policy;
        $$('#cVar button', p).forEach(b => b.addEventListener('click', () => { this.variant = b.dataset.v; $$('#cVar button', p).forEach(x => x.setAttribute('aria-pressed', String(x === b))); $('#cDesc', p).textContent = desc[this.variant]; $('#cRun', p).click(); }));
        $('#cRun', p).addEventListener('click', e => App.run(e.currentTarget, () => this.control(el))); $('#cRun', p).click();
      } else {
        p.innerHTML = `<div class="grid g-side"><div class="panel ctl"><h2>Off-policy evaluation</h2><p class="sub">“Would a new matchmaking policy be better?” — answered from sessions logged under the current policy, before deploying anything.</p>
          ${App.sel('oPre', 'Scenario', App.opts([['small', 'Small change (works well)'], ['large', 'Deploy DP-optimal (large change)'], ['random', 'From uniform-random logs'], ['custom', 'Custom']], 'small'))}
          <div class="grid g2" style="gap:8px">${App.sel('oT', 'Target π (new)', App.policyOptions('heuristic'))}${App.num('oTe', 'Target ε', 0.1, 0, 1, 0.05)}${App.sel('oB', 'Behaviour b (logged)', App.policyOptions('heuristic'))}${App.num('oBe', 'Behaviour ε', 0.3, 0, 1, 0.05)}${App.sel('oN', 'Logged episodes', App.opts([[500, '500'], [2000, '2,000'], [5000, '5,000'], [10000, '10,000'], [30000, '30,000']], 5000))}${App.num('oSeed', 'Seed', 4, 0, 999999)}</div>
          <button class="btn primary" id="oRun">Estimate target value</button>${logs ? `<label class="checks"><label><input type="checkbox" id="oLogs"> use uploaded logs (${esc(logs)})</label></label>` : ''}</div>
          <div class="stack"><div class="panel"><h2>Estimates of J(π) = E[G₀] from the session-start distribution</h2><div id="oTbl"></div></div><div class="kpis" id="oK"></div></div></div>
          <div class="grid g2" style="margin-top:16px"><div class="panel"><h2>Estimators as logs accumulate</h2><div id="oC"></div></div><div class="panel"><h2>Importance ratios ρ<sub>0:T−1</sub> (log bins)</h2><div id="oH"></div></div></div>
          <div class="panel" style="margin-top:16px"><h2>One logged episode, step by step</h2><div id="oEp"></div></div>`;
        const pre = { small: ['heuristic', 0.1, 'heuristic', 0.3], large: ['dp_optimal', 0.1, 'heuristic', 0.3], random: ['heuristic', 0.1, 'random', 0] };
        const apply = () => { const v = App.val(el, 'oPre'); if (pre[v]) { $('#oT', el).value = pre[v][0]; $('#oTe', el).value = pre[v][1]; $('#oB', el).value = pre[v][2]; $('#oBe', el).value = pre[v][3]; } };
        $('#oPre', p).addEventListener('change', () => { apply(); $('#oRun', p).click(); });
        ['oT', 'oTe', 'oB', 'oBe'].forEach(id => $('#' + id, p).addEventListener('change', () => { $('#oPre', el).value = 'custom'; }));
        apply(); $('#oRun', p).addEventListener('click', e => App.run(e.currentTarget, () => this.offp(el))); $('#oRun', p).click();
      }
    },
    async predict(el) {
      const body = { policy: App.val(el, 'pPol'), policy_epsilon: App.val(el, 'pEps'), episodes: +App.val(el, 'pN'), gamma: +App.val(el, 'pGamma'), seed: App.val(el, 'pSeed'), start_dist: App.val(el, 'pStart') };
      const r = await App.api('POST', '/rl/mc/predict', body); App.setData('lab-mc', { pred: r, mode: 'pred' }); const last = r.curve[r.curve.length - 1];
      $('#pK', el).innerHTML = App.kpi(r.n_episodes.toLocaleString(), 'complete episodes') + App.kpi(f(r.mean_return, 3) + ' ± ' + f(r.std_return, 2), 'mean G₀ ± sd') + App.kpi(f(r.mean_length, 2), 'mean episode length') + App.kpi(f(last.mae_first_visit, 3) + ' / ' + f(last.mae_every_visit, 3), 'MAE first / every visit') + App.kpi(App.pct(last.coverage, 0), 'states visited') + App.kpi(f(r.seconds, 2) + ' s', 'compute');
      C.line($('#pC', el), { series: [{ name: 'first-visit', points: r.curve.map(c => [c.episodes, c.mae_first_visit]), markers: true }, { name: 'every-visit', points: r.curve.map(c => [c.episodes, c.mae_every_visit]), markers: true }], logX: true, xLabel: 'episodes (log)', yLabel: 'MAE vs DP', height: 230, zero: true });
      const err = r.V_first_visit.map((v, i) => r.N_first_visit[i] > 0 ? Math.abs(v - r.V_dp_reference[i]) : null);
      App.policyMap($('#pV', el), { values: r.V_first_visit.map((v, i) => r.N_first_visit[i] > 0 ? v : null), mode: 'value', legend: 'first-visit V(s)', tip: s => `<div class="r"><span>first-visit</span><span>${f(r.V_first_visit[s], 3)}</span></div><div class="r"><span>every-visit</span><span>${f(r.V_every_visit[s], 3)}</span></div><div class="r"><span>DP V<sub>π</sub></span><span>${f(r.V_dp_reference[s], 3)}</span></div>` });
      App.policyMap($('#pE', el), { values: err, mode: 'value', vmin: 0, legend: 'absolute error', fmt: v => v === null ? '' : v.toFixed(2) });
      App.policyMap($('#pNmap', el), { values: r.N_first_visit, mode: 'value', vmin: 0, legend: 'first visits', fmt: v => v >= 10000 ? C.fmtNum(v) : String(Math.round(v)) });
      const seenS = new Set(); const epRows = r.sample_episode.map(x => { const first = !seenS.has(x.state); seenS.add(x.state); return Object.assign({ first }, x); });
      $('#pEp', el).innerHTML = App.table([{ h: 't', n: 1, f: x => x.t }, { h: 'S<sub>t</sub>', f: x => `<span class="mono">${x.state_label}</span>` }, { h: 'A<sub>t</sub>', f: x => App.act(x.action) }, { h: 'R<sub>t+1</sub>', n: 1, f: x => f(x.reward, 3) }, { h: 'G<sub>t</sub>', n: 1, f: x => f(x.G_t, 4) }, { h: 'Visit', f: x => x.first ? '<span class="good">first</span>' : '<span class="faint">repeat</span>' }], epRows, { scroll: true });
    },
    async predictLogs(el) {
      const id = App.data['lab-mc'].episodes_id; const r = await App.api('POST', '/rl/mc/predict', { episodes_id: id, gamma: +App.val(el, 'pGamma') });
      $('#pK', el).innerHTML = App.kpi(r.n_episodes, 'logged episodes') + App.kpi(App.prov(r.provenance), 'provenance');
      App.policyMap($('#pV', el), { values: r.V_first_visit.map((v, i) => r.N_first_visit[i] > 0 ? v : null), mode: 'value', legend: 'first-visit V(s) from logs' });
      App.policyMap($('#pNmap', el), { values: r.N_first_visit, mode: 'value', vmin: 0, legend: 'first visits' });
      $('#pC', el).innerHTML = '<div class="note">Logged episodes have no DP reference curve — the true dynamics behind them are unknown.</div>'; $('#pE', el).innerHTML = ''; $('#pEp', el).innerHTML = '';
    },
    async control(el) {
      const body = { variant: this.variant, episodes: +App.val(el, 'cN'), gamma: 0.9, epsilon: App.val(el, 'cEps'), behaviour_epsilon: App.val(el, 'cBeh'), seed: App.val(el, 'cSeed') };
      const r = await App.api('POST', '/rl/mc/control', body); this.ctl = r; App.setData('lab-mc', { ctl: r, mode: 'ctl' });
      $('#cK', el).innerHTML = App.kpi(App.pct(r.agreement_with_dp, 1), 'agreement with DP') + App.kpi(f(r.J_learned, 3), 'J of learned policy', `J* = ${f(r.J_dp_optimal, 3)}`) + App.kpi(f(r.q_mae_vs_dp, 3), 'mean |Q − Q*|') + App.kpi(r.unvisited_pairs, 'unvisited (s,a) pairs') + (r.transitions_used !== null && r.transitions_used !== undefined ? App.kpi(r.transitions_used.toLocaleString(), 'transitions used (tails)') : '') + App.kpi(f(r.seconds, 2) + ' s', 'compute');
      C.line($('#cJ', el), { series: [{ name: 'J(greedy)', points: r.history.map(h => [h.episodes, h.greedy_policy_value]) }], refLines: [{ y: r.J_dp_optimal, label: 'J* (DP)', color: '#3fdf8f' }], xLabel: 'episodes', yLabel: 'J', height: 210 });
      C.line($('#cA', el), { series: [{ name: 'agreement', points: r.history.map(h => [h.episodes, h.agreement_with_reference]) }], xLabel: 'episodes', yLabel: 'agreement', yMin: 0, yMax: 1, height: 210, fmtY: v => Math.round(v * 100) + '%', fmtTip: v => App.pct(v, 1) });
      this.drawCtlState(el);
      $('#cD', el).innerHTML = r.disagreements.length ? App.table([{ h: 'State', f: d => `<span class="mono">${d.short}</span>` }, { h: 'MC', f: d => App.act(RL.ACTION_NAMES.indexOf(d.action)) }, { h: 'DP', f: d => App.act(RL.ACTION_NAMES.indexOf(d.reference_action)) }, { h: 'q*-gap', n: 1, f: d => f(d.q_gap, 4) }], r.disagreements, { scroll: true }) : '<div class="note ok">Identical to π*.</div>';
    },
    drawCtlState(el) {
      const r = this.ctl, s = this.sel;
      App.policyMap($('#cMap', el), { actions: r.actions, mode: 'policy', diffWith: r.dp_actions, selected: s, onSelect: x => { this.sel = x; this.drawCtlState(el); }, tip: x => [0, 1, 2].map(a => `<div class="r"><span>Q(s,${RL.ACTION_NAMES[a]})</span><span>${f(r.Q[x][a], 3)} (N=${r.N[x][a]})</span></div>`).join('') });
      $('#cS', el).textContent = RL.stateShort(s);
      $('#cQ', el).innerHTML = App.table([{ h: 'a', f: a => App.act(a) }, { h: 'Q<sub>MC</sub>(s,a)', n: 1, f: a => f(r.Q[s][a], 4) }, { h: r.variant === 'off_policy' ? 'updates' : 'first visits N(s,a)', n: 1, f: a => r.N[s][a] }, { h: 'greedy', f: a => a === r.actions[s] ? '✓' : '' }, { h: 'DP π*', f: a => a === r.dp_actions[s] ? '✓' : '' }], [0, 1, 2]);
    },
    async offp(el) {
      const useLogs = $('#oLogs', el) && $('#oLogs', el).checked;
      const body = { target: App.val(el, 'oT'), target_epsilon: App.val(el, 'oTe'), behaviour: App.val(el, 'oB'), behaviour_epsilon: App.val(el, 'oBe'), episodes: +App.val(el, 'oN'), gamma: 0.9, seed: App.val(el, 'oSeed'), start_dist: 'session_start' };
      if (useLogs) body.episodes_id = App.data['lab-mc'].episodes_id;
      const r = await App.api('POST', '/rl/mc/off-policy', body); App.setData('lab-mc', { off: r, mode: 'off' });
      const truth = r.reference_J_target_dp, mc = r.on_policy_mc_J_target;
      const rows = [['Ordinary IS', r.J_ordinary_is, 'unbiased, high variance', 'Unit 5'], ['Weighted IS', r.J_weighted_is, 'biased, consistent, lower variance', 'Unit 5'], ['Per-decision IS', r.J_per_decision_is, 'R<sub>k+1</sub> weighted by ρ<sub>0:k</sub> only', 'truncated-return IS'], ['Discounting-aware OIS', r.J_discounting_aware_ois, 'flat partial returns Ḡ<sub>0:h</sub>', 'truncated-return IS'], ['Discounting-aware WIS', r.J_discounting_aware_wis, 'weighted version', 'truncated-return IS'], ['Behaviour policy (logged average)', r.J_behaviour_on_policy, 'what the old policy actually earned', ''], ['Truth: DP evaluation of π', truth, 'model-based reference', ''], ['Truth: deploy π (on-policy MC)', mc.mean, `± ${f(mc.ci95, 3)} (95% CI)`, '']];
      $('#oTbl', el).innerHTML = App.table([{ h: 'Estimator', f: x => x[0] }, { h: 'Ĵ(π)', n: 1, f: x => `<b>${f(x[1], 3)}</b>` }, { h: 'error vs DP', n: 1, f: x => x[0].startsWith('Truth') || x[0].startsWith('Behaviour') ? '' : f(x[1] - truth, 3) }, { h: 'Notes', f: x => `<span class="faint">${x[2]}</span>` }], rows, { rowClass: x => x[0].startsWith('Truth') ? 'best' : '' });
      $('#oK', el).innerHTML = App.kpi(f(r.rho_stats.effective_sample_size, 0), 'effective sample size', `of ${r.n_episodes} episodes`) + App.kpi(App.pct(r.rho_stats.fraction_nonzero, 1), 'episodes with ρ > 0') + App.kpi(f(r.rho_stats.max, 1), 'largest ρ') + App.kpi(App.pct(r.policy_agreement, 0), 'π vs b agreement') + App.kpi(r.reliability === 'OK' ? 'reliable' : 'unreliable', 'diagnostic', r.coverage.satisfied ? 'coverage ✓' : 'coverage ✗', r.reliability === 'OK' ? 'good' : 'warn');
      C.line($('#oC', el), { series: [{ name: 'ordinary IS', points: r.curve.map(c => [c.episodes, c.ordinary_is]) }, { name: 'weighted IS', points: r.curve.map(c => [c.episodes, c.weighted_is]) }, { name: 'per-decision IS', points: r.curve.map(c => [c.episodes, c.per_decision_is]) }], refLines: [{ y: truth, label: 'truth (DP)', color: '#3fdf8f' }, { y: r.J_behaviour_on_policy, label: 'behaviour', color: '#6c7b8e' }], logX: true, xLabel: 'logged episodes (log)', yLabel: 'Ĵ(π)', height: 240 });
      const h = r.rho_stats.histogram;
      if (h.counts && h.counts.length) C.bars($('#oH', el), { categories: ['0'].concat(h.counts.map((c, i) => C.fmtNum(Math.sqrt(h.edges[i] * h.edges[i + 1])))), series: [{ name: 'episodes', values: [h.zero].concat(h.counts), colors: ['#5b6b80'].concat(h.counts.map(() => '#7f6ff0')) }], yLabel: 'episodes', height: 230, fmt: v => v.toFixed(0) });
      else $('#oH', el).innerHTML = '<div class="empty">All ratios are zero.</div>';
      $('#oH', el).insertAdjacentHTML('beforeend', `<p class="sub" style="margin-top:6px">${esc(r.reliability)}</p>`);
      const ep = r.sample_episodes.find(e => e.rho > 0) || r.sample_episodes[0]; let cum = 1;
      $('#oEp', el).innerHTML = `<p class="sub">Episode ${ep.episode}: G₀ = ${f(ep.G0, 3)}, ρ<sub>0:T−1</sub> = ${f(ep.rho, 4)}. The transition probabilities cancel; only π and b remain.</p>` + App.table([{ h: 't', n: 1, f: x => x.t }, { h: 'S<sub>t</sub>', f: x => `<span class="mono">${x.state}</span>` }, { h: 'A<sub>t</sub>', f: x => App.act(RL.ACTION_NAMES.indexOf(x.action)) }, { h: 'π(A|S)', n: 1, f: x => f(x.pi, 4) }, { h: 'b(A|S)', n: 1, f: x => f(x.b, 4) }, { h: 'π/b', n: 1, f: x => f(x.ratio, 4) }, { h: 'Π so far', n: 1, f: x => { cum *= x.ratio; return f(cum, 4); } }, { h: 'R<sub>t+1</sub>', n: 1, f: x => f(x.reward, 3) }], ep.steps);
    },
  };
  App.viva['lab-mc'] = d => {
    let ex = null, concept = 'Monte Carlo prediction (first-visit / every-visit)';
    if (d.mode === 'pred' && d.pred) { const e = d.pred.sample_episode; ex = `<span class="calc">Episode with ${e.length} steps: G<sub>${e.length - 1}</sub> = ${f(e[e.length - 1].reward, 3)}; ${e.length > 1 ? `G<sub>${e.length - 2}</sub> = ${f(e[e.length - 2].reward, 3)} + γ×${f(e[e.length - 1].G_t, 3)} = ${f(e[e.length - 2].G_t, 3)}; … ` : ''}G<sub>0</sub> = ${f(e[0].G_t, 4)}. After ${d.pred.n_episodes} episodes, MAE(first-visit) = ${f(d.pred.curve[d.pred.curve.length - 1].mae_first_visit, 3)}.</span>`; }
    if (d.mode === 'ctl' && d.ctl) { concept = 'Monte Carlo control'; ex = `<span class="calc">${d.ctl.variant}: ${d.ctl.history[d.ctl.history.length - 1].episodes} episodes → agreement with DP ${App.pct(d.ctl.agreement_with_dp, 1)}, J = ${f(d.ctl.J_learned, 3)} vs J* = ${f(d.ctl.J_dp_optimal, 3)}.</span>`; }
    if (d.mode === 'off' && d.off) { concept = 'Off-policy prediction by importance sampling'; const ep = d.off.sample_episodes.find(e => e.rho > 0) || d.off.sample_episodes[0]; ex = `<span class="calc">ρ = ${ep.steps.slice(0, 4).map(s => `${f(s.pi, 3)}/${f(s.b, 3)}`).join(' × ')}${ep.steps.length > 4 ? ' × …' : ''} = ${f(ep.rho, 4)};  OIS = Σρ·G / n = ${f(d.off.J_ordinary_is, 3)};  WIS = Σρ·G / Σρ = ${f(d.off.J_weighted_is, 3)};  truth = ${f(d.off.reference_J_target_dp, 3)}.</span>`; }
    return { unit: 'Unit 5', unitClass: 'u5', concept, prov: 'SIMULATED',
      why: 'MC learns from complete sessions without a model; importance sampling evaluates a candidate matchmaking policy from logs of the current one — before deployment.',
      inputs: 'Complete episodes (S, A, R, S′, done); γ; for off-policy: π(a|s) and b(a|s).', outputs: 'V(s), Q(s,a), visit counts, learned policy, IS estimates with effective sample size.',
      formula: 'V(s) = (1/N(s)) Σ G<sub>t</sub> over (first) visits;  MC control: Q(s,a) ← avg(Returns), π ← ε-greedy(Q);  ρ<sub>t:T−1</sub> = Π<sub>k</sub> π(A<sub>k</sub>|S<sub>k</sub>)/b(A<sub>k</sub>|S<sub>k</sub>);  OIS: Σρ G/|T(s)|;  WIS: Σρ G/Σρ',
      example: ex, assumptions: 'Episodes always terminate; coverage π(a|s) > 0 ⇒ b(a|s) > 0 is checked; uploaded match histories have no actions, so they are used for prediction only.', code: 'monte_carlo.py: mc_prediction_run, mc_control, off_policy_prediction' };
  };

  // =================================================================== ALGORITHM COMPARISON
  App.screens['lab-compare'] = {
    render(el) {
      el.innerHTML = `${App.header('Algorithm comparison arena', 'Every sequential policy is evaluated on the same seeded simulated sessions (common random numbers), so differences are not luck of the draw. Bandit algorithms are deliberately absent: they solve a different, one-step problem.', [['u4', 'DP'], ['u5', 'MC control'], ['ux', 'Q-learning'], ['u1', 'model-based vs model-free']])}
      <div class="panel ctl" style="margin-bottom:16px"><div class="grid" style="grid-template-columns:repeat(auto-fit,minmax(150px,1fr))">
        ${App.sel('xN', 'Evaluation episodes', App.opts([[100, '100'], [500, '500'], [1000, '1,000'], [2000, '2,000']], 500))}${App.num('xSeed', 'Evaluation seed', 2026, 0, 999999)}${App.sel('xStart', 'Start states', App.opts([['session_start', 'session start'], ['uniform', 'all 48 uniformly']], 'session_start'))}${App.sel('xMc', 'MC control episodes', App.opts([[5000, '5,000'], [10000, '10,000'], [20000, '20,000']], 10000))}${App.sel('xQ', 'Q-learning episodes', App.opts([[10000, '10,000'], [30000, '30,000 (production)'], [60000, '60,000']], 30000))}${App.sel('xStab', 'Stability runs', App.opts([[1, '1 (off)'], [2, '2'], [3, '3']], 1))}
      </div><div class="row" style="margin-top:10px"><button class="btn primary" id="xRun">Run comparison</button><span class="faint" style="font-size:12px">γ = 0.9 · all learners use the same simulator; DP uses the 1000-sample estimated model.</span></div></div>
      <div class="panel"><h2>Leaderboard ${App.prov('SIMULATED')}</h2><div id="xTbl"></div><p class="sub" style="margin-top:8px">± values are 95% confidence half-widths (1.96·sd/√n). “Δ vs DP” is the paired difference in discounted return on identical episodes — much tighter than comparing two independent means.</p></div>
      <div class="grid g2" style="margin-top:16px"><div class="panel"><h2>Discounted return per session</h2><div id="xG"></div></div><div class="panel"><h2>Rage-quit rate</h2><div id="xR"></div></div><div class="panel"><h2>Session length (matches)</h2><div id="xL"></div></div><div class="panel"><h2>Close-match rate (C ≥ 0.8)</h2><div id="xC"></div></div></div>
      <div class="grid g2" style="margin-top:16px"><div class="panel"><h2>Policy agreement matrix (% of 48 states)</h2><div id="xM"></div></div><div class="panel"><h2>Policies side by side</h2><div id="xP" class="grid g2"></div></div></div>
      <div class="panel" style="margin-top:16px"><h2>Why do the algorithms produce different policies?</h2><div id="xWhy"></div></div>`;
      $('#xRun', el).addEventListener('click', e => App.run(e.currentTarget, () => this.run(el))); $('#xRun', el).click();
      document.addEventListener('engine-change', () => { if (App.rendered['lab-compare']) $('#xRun', el).click(); });
    },
    async run(el) {
      const body = { n_episodes: +App.val(el, 'xN'), seed: App.val(el, 'xSeed'), gamma: 0.9, start_dist: App.val(el, 'xStart'), mc_episodes: +App.val(el, 'xMc'), mc_epsilon: 0.1, q_episodes: +App.val(el, 'xQ'), stability_runs: +App.val(el, 'xStab'), samples: 1000 };
      const r = await App.api('POST', '/rl/compare', body); App.setData('lab-compare', { cmp: r });
      const P = r.policies; const dp = P.find(p => p.key === 'dp_value_iteration');
      const ev = (p, k, pct) => `${pct ? App.pct(p.evaluation[k].mean, 1) : f(p.evaluation[k].mean, 3)} <span class="faint">± ${pct ? App.pct(p.evaluation[k].ci95, 1) : f(p.evaluation[k].ci95, 3)}</span>`;
      $('#xTbl', el).innerHTML = App.table([{ h: 'Policy', f: p => `<div style="min-width:210px" title="${esc(p.notes)}">${esc(p.name)}<div class="faint" style="font-size:11px">${esc(p.family)}</div></div>` }, { h: 'Discounted return', n: 1, f: p => ev(p, 'discounted_return') }, { h: 'Δ vs DP (paired)', n: 1, f: p => p.key === 'dp_value_iteration' ? '—' : `${App.sgn(p.paired_diff_vs_dp.mean, 3)} <span class="faint">± ${f(p.paired_diff_vs_dp.ci95, 3)}</span>` }, { h: 'Σ reward', n: 1, f: p => ev(p, 'episodic_return') }, { h: 'Session length', n: 1, f: p => ev(p, 'session_length') }, { h: 'Rage quit', n: 1, f: p => ev(p, 'rage_quit_rate', true) }, { h: 'Close match', n: 1, f: p => ev(p, 'close_match_rate', true) }, { h: 'Agree w/ DP', n: 1, f: p => App.pct(p.agreement_with_dp, 0) }, { h: 'Model J', n: 1, f: p => f(p.model_value_J, 3) }, { h: 'Stability', n: 1, f: p => p.stability === null || p.stability === undefined ? '—' : App.pct(p.stability, 0) }, { h: 'Compute', n: 1, f: p => f(p.compute_seconds, 2) + ' s' }, { h: 'Convergence', f: p => `<span class="faint">${esc(p.convergence)}</span>` }], P, { rowClass: p => p.key === 'dp_value_iteration' ? 'best' : '' });
      const cats = P.map(p => p.name.replace('Dynamic Programming — ', 'DP ').replace(' (v3-style rank-only matchmaking)', '').replace('Monte Carlo Control', 'MC control').replace('Manual rule (designer heuristic)', 'Heuristic').replace(' baseline', '').replace('Tabular ', ''));
      const col = P.map(p => p.family === 'model-based' ? '#0aa5c4' : p.family === 'model-free' ? '#c43fb0' : '#5b6b80');
      const bar = (id, k, pct) => C.bars($(id, el), { categories: cats, series: [{ name: k, values: P.map(p => p.evaluation[k].mean), err: P.map(p => p.evaluation[k].ci95), colors: col }], height: 230, rotate: true, fmt: pct ? (v => Math.round(v * 100) + '%') : undefined, fmtTip: pct ? (v => App.pct(v, 1)) : undefined });
      bar('#xG', 'discounted_return'); bar('#xR', 'rage_quit_rate', true); bar('#xL', 'session_length'); bar('#xC', 'close_match_rate', true);
      $('#xG', el).insertAdjacentHTML('beforeend', '<div class="legend" style="margin-top:4px"><span><i style="background:#0aa5c4"></i>model-based</span><span><i style="background:#c43fb0"></i>model-free</span><span><i style="background:#5b6b80"></i>baseline</span></div>');
      C.matrix($('#xM', el), { rows: cats, cols: cats, values: r.agreement_matrix.values, label: 'agreement', fmtTip: v => App.pct(v, 1) });
      $('#xP', el).innerHTML = P.filter(p => p.key !== 'random').map(p => `<div><div id="xp_${p.key}"></div></div>`).join('');
      P.filter(p => p.key !== 'random').forEach(p => App.miniMap($('#xp_' + p.key, el), p.actions, dp.actions, `<span>${esc(cats[P.indexOf(p)])}</span><span class="mono">${App.pct(p.agreement_with_dp, 0)}</span>`));
      const q = P.find(p => p.key === 'q_learning'), m = P.find(p => p.key === 'mc_control');
      const smallGap = arr => arr.filter(d => d.q_gap < 0.05).length;
      $('#xWhy', el).innerHTML = `<div class="grid g3" style="font-size:13px">
        <div><h3>Different information</h3><p class="dim">DP plans with the estimated transition model p̂(s′|s,a); MC and Q-learning only see sampled sessions. Model error (≈ ${'±'}1.6 pp per probability at 1000 samples) and sampling noise are different error sources.</p></div>
        <div><h3>Near-ties and rare states</h3><p class="dim">Of Q-learning's ${q ? q.disagreements.length : 0} disagreements with DP, ${q ? smallGap(q.disagreements) : 0} have a q*-gap below 0.05 (near-ties: either action is almost equally good). The larger gaps sit in states the learner rarely visits, where its estimates are still noisy. Net cost: model J ${q ? f(q.model_value_J, 3) : '—'} vs ${f(dp.model_value_J, 3)} (${q ? App.pct(q.model_value_J / dp.model_value_J - 1, 1) : '—'}).</p></div>
        <div><h3>Different targets</h3><p class="dim">On-policy MC control learns the best <i>ε-soft</i> policy (it keeps exploring), Q-learning learns the greedy optimum but with a constant step size; DP is exact for the model it is given. Off-policy MC learns only from episode tails and converges slowest.${m ? ` MC control here agrees with DP on ${App.pct(m.agreement_with_dp, 0)} of states.` : ''}</p></div></div>`;
    },
  };
  App.viva['lab-compare'] = d => {
    let ex = null; const r = d.cmp;
    if (r) { const dp = r.policies.find(p => p.key === 'dp_value_iteration'), q = r.policies.find(p => p.key === 'q_learning'), k = r.policies.find(p => p.key === 'keep'); ex = `<span class="calc">n = ${r.settings.n_episodes} identical episodes: DP ${f(dp.evaluation.discounted_return.mean, 3)}, Q-learning ${f(q.evaluation.discounted_return.mean, 3)} (paired Δ ${App.sgn(q.paired_diff_vs_dp.mean, 3)} ± ${f(q.paired_diff_vs_dp.ci95, 3)}), always-KEEP ${f(k.evaluation.discounted_return.mean, 3)}; rage quits ${App.pct(k.evaluation.rage_quit_rate.mean, 1)} → ${App.pct(dp.evaluation.rage_quit_rate.mean, 1)}.</span>`; }
    return { unit: 'Units 1, 4, 5', unitClass: 'u4', concept: 'Model-based vs model-free, evaluated fairly', prov: 'SIMULATED',
      why: 'A capstone claim needs evidence: identical environment, seeds and metrics for every sequential algorithm, with uncertainty reported.',
      inputs: 'Policies from DP (VI, PI), MC control, Q-learning, baselines; evaluation seed; n episodes.', outputs: 'Mean ± 95% CI of return, length, rage-quit and close-match rates; paired differences; agreement; stability; compute time.',
      formula: 'Ĵ(π) = (1/n) Σ<sub>i</sub> Σ<sub>t</sub> γ<sup>t</sup>R<sub>t+1</sub><sup>(i)</sup>;   CI = ±1.96·s/√n;   paired Δ<sub>i</sub> = G<sub>0</sub><sup>(i)</sup>(π) − G<sub>0</sub><sup>(i)</sup>(π*) on the same seed i',
      example: ex, assumptions: 'All numbers come from the simulator; stability = agreement between policies learned with different seeds.', code: 'rl_evaluation.py: compare_policies, evaluate_policy' };
  };
})(window);
