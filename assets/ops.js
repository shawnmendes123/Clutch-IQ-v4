/* Clutch IQ — Mission control + Operations screens (v3 features preserved). */
(function (root) {
  'use strict';
  const App = root.App, RL = root.ClutchRL, C = root.Charts, esc = C.esc, $ = App.$, $$ = App.$$;

  // =================================================================== shared player model (mirrors models.py / gnn_chemistry.py)
  const ARCHETYPES = { 'Aggressive Entry': [0.90, 0.30, 0.40, 0.15, 0.55], 'Passive Support': [0.15, 0.55, 0.85, 0.90, 0.45], 'Objective Anchor': [0.35, 0.70, 0.55, 0.40, 0.95], 'Flex Rotator': [0.55, 0.60, 0.60, 0.50, 0.60], 'Sniper/Isolation': [0.70, 0.85, 0.20, 0.10, 0.35] };
  const NAMES = ['Vexen', 'Ryza', 'Kael', 'Nova', 'Zeph', 'Ashra', 'Tinker', 'Ori', 'Quill', 'Draven', 'Mira', 'Sable', 'Fenix', 'Lyric', 'Bram', 'Ixel', 'Corvo', 'Wynn', 'Talon', 'Juno', 'Rook', 'Nyx', 'Ember', 'Skye', 'Halo', 'Grim', 'Petal', 'Vantage', 'Onyx', 'Riven'];
  const CL = ['#0aa5c4', '#c43fb0', '#2f9e5e', '#7f6ff0', '#c98500'];
  const clamp = (v, lo = 0, hi = 1) => Math.max(lo, Math.min(hi, v));
  function randomPlayer(r) {
    const u = (a, b) => a + (b - a) * r.random(), ri = (a, b) => a + Math.floor(r.random() * (b - a + 1)), ch = arr => arr[Math.floor(r.random() * arr.length)];
    const arch = ch(Object.keys(ARCHETYPES)); const vec = ARCHETYPES[arch].map(v => clamp(v + u(-0.12, 0.12)));
    const streak = ch([ri(-6, -1), ri(1, 6), 0]); const mins = u(0, 240); const fatigue = clamp(mins / 240 + u(-0.1, 0.1)); const tilt = clamp((Math.max(0, -streak) / 6) * 0.8 + u(0, 0.2));
    return { id: Math.floor(r.random() * 1e9).toString(36), name: ch(NAMES) + ri(10, 99), mmr: ri(900, 2600), archetype: arch, playstyle_vector: vec, win_streak: streak, reaction_drift_ms: fatigue * u(20, 90), comms_freq: clamp(u(0.1, 0.9) - tilt * 0.3), session_minutes: mins, tilt_score: tilt, fatigue_index: fatigue, source: 'SIMULATED' };
  }
  const fv = p => [...p.playstyle_vector, p.tilt_score, p.fatigue_index, p.mmr / 3000];
  const dist = (a, b) => Math.sqrt(a.reduce((s, v, i) => s + (v - b[i]) ** 2, 0));
  function embed(players, hops = 2) {
    const X = players.map(fv); const nb = X.map((xi, i) => X.map((xj, j) => [j, i === j ? Infinity : dist(xi, xj)]).sort((a, b) => a[1] - b[1]).slice(0, 5).map(e => e[0]));
    let H = X; for (let h = 0; h < hops; h++) H = H.map((hi, i) => hi.map((v, d) => 0.5 * v + 0.5 * nb[i].reduce((s, n) => s + H[n][d], 0) / nb[i].length)); return H;
  }
  function kmeans(X, k, seed) {
    const r = new RL.Mulberry32(seed); const used = new Set(); let cent = [];
    while (cent.length < k) { const i = Math.floor(r.random() * X.length); if (!used.has(i)) { used.add(i); cent.push([...X[i]]); } }
    let lab = []; for (let it = 0; it < 25; it++) { lab = X.map(x => { let b = 0, bd = Infinity; cent.forEach((c, ci) => { const d = dist(x, c); if (d < bd) { bd = d; b = ci; } }); return b; }); const sums = Array.from({ length: k }, () => new Array(X[0].length).fill(0)), cnt = new Array(k).fill(0); X.forEach((x, i) => { cnt[lab[i]]++; sums[lab[i]] = sums[lab[i]].map((v, d) => v + x[d]); }); cent = sums.map((s, ci) => cnt[ci] ? s.map(v => v / cnt[ci]) : cent[ci]); }
    return lab;
  }
  function chemistry(team) {
    if (team.length < 2) return 1; const dims = team[0].playstyle_vector.length; let rd = 0;
    for (let d = 0; d < dims; d++) { const v = team.map(p => p.playstyle_vector[d]); const m = v.reduce((a, b) => a + b, 0) / v.length; rd += Math.sqrt(v.reduce((a, b) => a + (b - m) ** 2, 0) / v.length); }
    rd /= dims; const tp = Math.max(0, team.filter(p => p.tilt_score > 0.6).length - 1) * 0.15; const mm = team.map(p => p.mmr);
    return clamp(0.5 + rd - tp - (Math.max(...mm) - Math.min(...mm)) / 3000 * 0.3);
  }
  function slate(players, size = 10) {
    const pool = [...players].sort((a, b) => b.mmr - a.mmr).slice(0, size); let A = [], B = [];
    pool.forEach((p, i) => { const even = Math.floor(i / 2) % 2 === 0; if (even) (i % 2 === 0 ? A : B).push(p); else (i % 2 === 0 ? B : A).push(p); });
    let bA = [...A], bB = [...B], best = chemistry(bA) + chemistry(bB);
    for (let i = 0; i < A.length; i++) for (let j = 0; j < B.length; j++) { const tA = [...A], tB = [...B];[tA[i], tB[j]] = [tB[j], tA[i]]; const gap = Math.abs(tA.reduce((s, p) => s + p.mmr, 0) - tB.reduce((s, p) => s + p.mmr, 0)); if (gap > 400) continue; const sc = chemistry(tA) + chemistry(tB); if (sc > best) { best = sc; bA = tA; bB = tB; } }
    const gap = Math.abs(bA.reduce((s, p) => s + p.mmr, 0) - bB.reduce((s, p) => s + p.mmr, 0));
    return { team_a: bA, team_b: bB, team_a_chemistry: chemistry(bA), team_b_chemistry: chemistry(bB), mmr_gap: gap, predicted_close_match: 1 - Math.min(1, gap / 1000) };
  }
  const Q = App.queue = { pool: [], clusters: [], labels: null, seed: 101, source: 'SIMULATED', datasetId: null };
  Q.refreshSim = function () { const r = new RL.Mulberry32(Q.seed); Q.pool = Array.from({ length: 40 }, () => randomPlayer(r)); Q.clusters = kmeans(embed(Q.pool), 5, Q.seed); Q.labels = null; Q.source = 'SIMULATED'; };
  Q.refreshSim();

  // =================================================================== MISSION CONTROL
  App.screens.overview = {
    render(el) {
      el.innerHTML = `${App.header('Matchmaking that reads the player — and learns how to', 'Clutch IQ balances teams on skill and chemistry, then a reinforcement-learning controller adjusts each player\'s lobby difficulty from their fatigue, tilt and streak. The same session environment is studied here through bandits, a finite MDP, dynamic programming, Monte Carlo and Q-learning.', [['u1', 'Unit 1'], ['u2', 'Unit 2'], ['u3', 'Unit 3'], ['u4', 'Unit 4'], ['u5', 'Unit 5']])}
      <div class="grid g-side-r">
        <div class="panel hot">
          <h2>The RL interaction loop, running now ${App.prov('SIMULATED')}</h2>
          <p class="sub">A real simulated session under the DP-optimal policy π*. Each tick is one match: the agent observes S<sub>t</sub>, picks A<sub>t</sub>, the environment returns R<sub>t+1</sub> and S<sub>t+1</sub>.</p>
          <div id="loopSvg"></div>
        </div>
        <div class="panel">
          <h2>This tick</h2>
          <div id="loopRead" class="stack"></div>
          <div class="row" style="margin-top:14px"><button class="btn primary" data-jump="ctl-session">Open live session</button><button class="btn" data-jump="acad-formulation">See the formal MDP</button></div>
        </div>
      </div>
      <div class="panel" style="margin-top:16px">
        <div class="row between"><h2>Optimal session-control policy π* across all 48 states ${App.prov('MODEL ESTIMATE')}</h2><span id="ovEngine"></span></div>
        <p class="sub">Value iteration on the simulation-estimated transition model, γ = 0.9. Rows: tilt; columns: streak; one block per fatigue level. Hover a cell for V*(s) and Q*(s,·); click to open it in the MDP explorer.</p>
        <div id="ovMap"></div>
      </div>
      <div class="grid g3" style="margin-top:16px" id="ovTiles"></div>`;
      $$('[data-jump]', el).forEach(b => b.addEventListener('click', () => App.go(b.dataset.jump)));
      this.load(el);
      document.addEventListener('engine-change', () => { if (App.current === 'overview') this.load(el); });
    },
    async load(el) {
      $('#ovEngine', el).innerHTML = App.engineNote();
      const vi = await App.api('POST', '/rl/dp/value-iteration', { gamma: 0.9, theta: 1e-8, max_iter: 2000, snapshots: 0 });
      const qt = await App.api('GET', '/rl/q/table?gamma=0.9');
      const nums = await App.api('GET', '/academic/numericals');
      App.data.overview = { vi };
      App.policyMap($('#ovMap', el), { actions: vi.actions, mode: 'policy', diffWith: qt.actions, tip: s => `<div class="r"><span>V*(s)</span><span>${App.f(vi.V[s])}</span></div>${[0, 1, 2].map(a => `<div class="r"><span>Q*(s,${RL.ACTION_NAMES[a]})</span><span>${App.f(vi.Q[s][a])}</span></div>`).join('')}<div class="r"><span>Q-learning picks</span><span>${RL.ACTION_NAMES[qt.actions[s]]}</span></div>`, onSelect: s => { App.setData('ctl-mdp', { state: s }); App.go('ctl-mdp'); } });
      const counts = [0, 0, 0]; vi.actions.forEach(a => counts[a]++);
      const exp = (root.CLUTCH_STATIC.experiments || {}).experiments || [];
      const e6 = exp.find(e => e.id === 'E6'), e7 = exp.find(e => e.id === 'E7');
      const dpRow = e6 && e6.rows.find(r => r.policy.startsWith('Dynamic Programming — Value')), keepRow = e6 && e6.rows.find(r => r.policy.startsWith('Always KEEP'));
      const tiles = [
        ['ctl-dp', 'u4', 'Dynamic programming', `${vi.iterations} sweeps`, `Value iteration converged (contraction ratio ≤ ${App.f(vi.max_contraction_ratio, 3)} < γ). π* uses ${counts[0]} LOWER / ${counts[1]} KEEP / ${counts[2]} RAISE.`],
        ['ctl-q', 'ux', 'Tabular Q-learning', App.pct(qt.agreement_with_dp, 0), 'of states where the model-free Q-learner picks the same action as DP; the Q-learning screen lists what each disagreement costs.'],
        ['lab-compare', 'u1', 'Rage quits, KEEP vs π*', dpRow ? `${App.pct(keepRow.rage_quit_rate.mean, 0)} → ${App.pct(dpRow.rage_quit_rate.mean, 0)}` : '—', 'Sessions ending in a rage quit under rank-only matchmaking vs the RL policy (1000 seeded simulated sessions).'],
        ['lab-bandit', 'u2', 'Bandit arena', e7 ? `${App.f(e7.stationary.find(r => r.key === 'greedy').final_regret, 0)} → ${App.f(e7.stationary.find(r => r.key === 'optimistic').final_regret, 0)}` : '—', 'Cumulative regret after 1000 map picks: greedy vs optimistic greedy (200 runs).'],
        ['lab-mc', 'u5', 'Monte Carlo lab', 'MAE ↓ with data', 'First- and every-visit estimates converge to the DP values as complete episodes accumulate; importance sampling evaluates new policies off-line.'],
        ['acad-numericals', 'u3', 'Lecture-note numericals', `${nums.matched}/${nums.checked}`, 'worked examples from the uploaded notes reproduced exactly by the same code that runs Clutch IQ.'],
      ];
      $('#ovTiles', el).innerHTML = tiles.map(([go, u, t, v, d]) => `<div class="panel" role="button" tabindex="0" data-go2="${go}" style="cursor:pointer"><div class="row between"><h2>${t}</h2><span class="unit ${u}">${{ u1: 'Unit 1', u2: 'Unit 2', u3: 'Unit 3', u4: 'Unit 4', u5: 'Unit 5', ux: 'TD' }[u]}</span></div><div class="kpi" style="background:none;border:0;padding:6px 0"><div class="v">${v}</div></div><p class="sub" style="margin:0">${d}</p></div>`).join('');
      $$('[data-go2]', el).forEach(t => { t.addEventListener('click', () => App.go(t.dataset.go2)); t.addEventListener('keydown', e => { if (e.key === 'Enter') App.go(t.dataset.go2); }); });
      this.startLoop(el, vi);
    },
    startLoop(el, vi) {
      clearInterval(this._timer); const pi = RL.deterministicPolicy(vi.actions); let seed = 40, ep = null, i = 0, G = 0;
      const svgBox = $('#loopSvg', el), read = $('#loopRead', el);
      const draw = (tr) => {
        const s = tr ? tr.state : 0, a = tr ? tr.action : 1, r = tr ? tr.reward : 0, s2 = tr ? tr.next_state : 0;
        svgBox.innerHTML = `<svg viewBox="0 0 620 250" class="graph" role="img" aria-label="Agent environment loop">
          <defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0L10,5L0,10z" fill="#a9b6c6"/></marker></defs>
          <rect x="20" y="70" width="190" height="110" rx="12" fill="#0e141e" stroke="#22d8f5"/><text x="115" y="104" text-anchor="middle" style="font-family:'Chakra Petch';font-size:17px;fill:#e7eef6;font-weight:600">Agent</text><text x="115" y="126" text-anchor="middle">Clutch IQ session controller</text><text x="115" y="148" text-anchor="middle" style="fill:#22d8f5">π*(a|s) — DP optimal</text>
          <rect x="410" y="70" width="190" height="110" rx="12" fill="#0e141e" stroke="#ff3dc8"/><text x="505" y="104" text-anchor="middle" style="font-family:'Chakra Petch';font-size:17px;fill:#e7eef6;font-weight:600">Environment</text><text x="505" y="126" text-anchor="middle">player's play session</text><text x="505" y="148" text-anchor="middle" style="fill:#ff8de0">simulated p(s′, r | s, a)</text>
          <path d="M210,95 C300,40 320,40 410,95" fill="none" stroke="#a9b6c6" stroke-width="1.5" marker-end="url(#ah)"/><text x="310" y="40" text-anchor="middle">action A<tspan dy="3" font-size="9">t</tspan><tspan dy="-3"> = </tspan><tspan style="fill:${App.ACOL[a]};font-weight:600">${RL.ACTION_NAMES[a]}</tspan></text>
          <path d="M410,150 C320,215 300,215 210,150" fill="none" stroke="#a9b6c6" stroke-width="1.5" marker-end="url(#ah)"/><text x="310" y="226" text-anchor="middle">reward R<tspan dy="3" font-size="9">t+1</tspan><tspan dy="-3"> = </tspan><tspan style="fill:${r < 0 ? '#ff4d6a' : '#3fdf8f'};font-weight:600">${r.toFixed(2)}</tspan>  ·  state S<tspan dy="3" font-size="9">t+1</tspan><tspan dy="-3"> = ${RL.stateShort(s2)}</tspan></text>
          <text x="310" y="128" text-anchor="middle" style="fill:#6c7b8e">t = ${i}</text>
        </svg>`;
        read.innerHTML = tr ? `<div><div class="lbl">State S<sub>t</sub></div>${App.stateChip(s)}</div><div><div class="lbl">Action A<sub>t</sub> ~ π*(·|S<sub>t</sub>)</div>${App.act(a)}</div><div><div class="lbl">Reward R<sub>t+1</sub></div><span class="num ${r < 0 ? 'bad' : 'good'}">${r.toFixed(3)}</span> <span class="faint">${tr.info.rage_quit ? 'rage quit' : tr.info.natural_end ? 'natural end' : tr.done ? 'boredom quit' : 'match completed'} · closeness ${App.f(tr.info.closeness, 2)}</span></div><div><div class="lbl">Discounted return so far Σγ<sup>k</sup>R</div><span class="num cy">${G.toFixed(3)}</span></div>` : '';
      };
      const tick = () => {
        if (!ep || i >= ep.transitions.length) { seed++; ep = RL.generateEpisode(pi, seed, { startDist: 'session_start', recordInfo: true }); i = 0; G = 0; }
        const tr = ep.transitions[i]; G += Math.pow(0.9, i) * tr.reward; draw(tr); i++;
      };
      tick(); const reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
      if (!reduce) this._timer = setInterval(() => { if (App.current === 'overview' && !document.hidden) tick(); }, 1700);
    },
  };
  App.viva.overview = () => ({ unit: 'Unit 1', unitClass: 'u1', concept: 'Agent, environment, state, action, reward, policy', prov: 'SIMULATED',
    why: 'The matchmaking session is a sequential decision problem: today\'s lobby difficulty changes tomorrow\'s tilt, fatigue and the chance of a rage quit.',
    inputs: 'S<sub>t</sub> = (fatigue, tilt, streak) buckets; policy π*(a|s).', outputs: 'A<sub>t</sub> ∈ {LOWER, KEEP, RAISE}; R<sub>t+1</sub>; S<sub>t+1</sub>.',
    formula: 'S<sub>0</sub>, A<sub>0</sub>, R<sub>1</sub>, S<sub>1</sub>, A<sub>1</sub>, R<sub>2</sub>, …   with   R = 1.0·(match completed) + 1.2·C − 3.0·(rage quit),   C = 1 − 2|p<sub>win</sub> − ½|',
    example: 'The loop above is a live seeded episode; the panel on the right shows S<sub>t</sub>, A<sub>t</sub>, R<sub>t+1</sub> and the running discounted return.',
    assumptions: 'Player behaviour is simulated (rl_core.SessionEnv); the state is the bucketed (fatigue, tilt, streak).', code: 'rl_core.py: env_step, spec · assets/rl_engine.js: envStep' });

  // =================================================================== DATA & UPLOAD
  App.screens['ops-data'] = {
    render(el) {
      el.innerHTML = `${App.header('Data & upload', 'Run on simulated players, or upload any competitive-game match history. The schema detector maps columns automatically and every engineered feature is labelled with where it came from.', [['u1', 'Unit 1 · state features'], ['u5', 'Unit 5 · real-data MC']])}
      <div class="grid g2">
        <div class="panel">
          <h2>Match-history CSV ${App.prov('REAL DATA')}</h2>
          <p class="sub">Kills, deaths, score, result, date and player columns are detected by name and statistics. Needs the Python backend.</p>
          <div class="row"><input type="file" id="csvFile" accept=".csv"><button class="btn primary" id="btnUpload">Upload &amp; detect schema</button></div>
          <div id="schemaOut" style="margin-top:14px"></div>
        </div>
        <div class="panel">
          <h2>What is real and what is simulated</h2>
          ${App.table([{ h: 'Signal', k: 'a' }, { h: 'Source', f: r => r.b }], [
            ['Kills, deaths, score, win/loss, dates', App.prov('REAL DATA')], ['KD ratio, streak, fatigue index, tilt score, sessions', App.prov('DERIVED FEATURE')],
            ['Match closeness without scorelines; rage-quit flag', App.prov('FALLBACK/PROXY')], ['Player reactions to a lobby-difficulty action', App.prov('SIMULATED')],
            ['Transition probabilities p(s′|s,a)', App.prov('MODEL ESTIMATE')], ['Synthetic queue in simulation mode', App.prov('SIMULATED')]].map(r => ({ a: r[0], b: r[1] })), {})}
        </div>
      </div>
      <div class="grid g2" style="margin-top:16px">
        <div class="panel">
          <h2>Monte Carlo prediction on your uploaded sessions ${App.prov('REAL DATA')}</h2>
          <p class="sub">Each (player, session) becomes one episode of bucketed states and rewards. Real match histories contain no matchmaking actions, so only prediction runs here — no control and no importance sampling.</p>
          <button class="btn" id="btnRealMc">Estimate V(s) from uploaded sessions</button>
          <div id="realMcOut" style="margin-top:12px"></div>
        </div>
        <div class="panel">
          <h2>Logged RL sessions (state, action, reward)</h2>
          <p class="sub">Upload logs with <span class="mono">episode_id, step, fatigue, tilt, streak, action, reward[, behaviour_prob]</span> to run MC prediction and — only if behaviour probabilities are present — off-policy evaluation. <a id="dlLogs" href="#">Download a clearly-labelled simulated sample</a>.</p>
          <div class="row"><input type="file" id="logFile" accept=".csv"><button class="btn" id="btnLogUpload">Upload logs</button></div>
          <div id="logOut" style="margin-top:12px"></div>
        </div>
      </div>`;
      $('#btnUpload', el).addEventListener('click', e => App.run(e.currentTarget, () => this.upload(el)));
      $('#btnRealMc', el).addEventListener('click', e => App.run(e.currentTarget, () => this.realMc(el)));
      $('#btnLogUpload', el).addEventListener('click', e => App.run(e.currentTarget, () => this.logs(el)));
      $('#dlLogs', el).addEventListener('click', async ev => { ev.preventDefault(); if (App.engine !== 'python') { App.toast('The sample is generated by the backend — connect it first.'); return; } location.href = App.base + '/rl/episodes/sample'; });
    },
    async upload(el) {
      const f = $('#csvFile', el).files[0]; if (!f) throw new Error('Choose a CSV file first.');
      const fd = new FormData(); fd.append('file', f);
      const d = await App.api('POST', '/upload', fd); Q.datasetId = d.dataset_id;
      const rows = d.schema_report.rows.filter(r => r.matched_column);
      $('#schemaOut', el).innerHTML = `<div class="note ok">${esc(d.filename)} · ${d.row_count} rows · ${rows.length} of ${d.schema_report.rows.length} roles detected</div><div class="row" style="margin:8px 0">${rows.map(r => `<span class="tag">${esc(r.role)} → <b>${esc(r.matched_column)}</b></span>`).join('')}</div><div id="engOut"><span class="spin"></span> Engineering features…</div>`;
      const e = await App.api('POST', `/engineer/${d.dataset_id}`);
      const notes = e.engineering_notes.map(n => `<li>${/FALLBACK|PROXY|approximated|fallback/.test(n) ? App.prov('FALLBACK/PROXY') : App.prov('DERIVED FEATURE')} ${esc(n)}</li>`).join('');
      $('#engOut', el).innerHTML = `<div class="note ok">Engineered ${e.player_count} player(s) and ${Object.keys(e.cluster_labels).length} labelled chemistry clusters.</div><ul style="margin:8px 0 0 18px;font-size:12.5px;display:grid;gap:4px">${notes}</ul><div class="row" style="margin-top:10px"><button class="btn sm" id="toQueue">Open these players in the live queue</button></div>`;
      const p = await App.api('GET', `/players/${d.dataset_id}`);
      Q.pool = p.players.map(x => Object.assign(x, { source: 'REAL DATA' })); Q.clusters = p.chemistry_clusters; Q.labels = p.cluster_labels; Q.source = 'REAL DATA';
      $('#toQueue', el).addEventListener('click', () => App.go('ops-queue'));
      App.rendered['ops-queue'] = false; App.rendered['ops-burnout'] = false;
    },
    async realMc(el) {
      if (!Q.datasetId) throw new Error('Upload a match-history CSV first.');
      const r = await App.api('GET', `/rl/mc/real/${Q.datasetId}?gamma=0.9`);
      const out = $('#realMcOut', el);
      out.innerHTML = `<div class="kpis">${App.kpi(r.n_episodes, 'sessions (episodes)')}${App.kpi(App.f(r.mean_length, 1), 'mean matches / session')}${App.kpi(r.states_observed + '/48', 'states observed')}${App.kpi(App.f(r.mean_return, 2), 'mean return G₀')}</div><div id="realMap" style="margin-top:12px"></div><ul style="margin:10px 0 0 18px;font-size:12px;color:var(--ink-2)">${r.notes.map(n => `<li>${esc(n)}</li>`).join('')}</ul>`;
      App.policyMap($('#realMap', out), { values: r.V_first_visit.map((v, i) => r.N_first_visit[i] > 0 ? v : null), mode: 'value', legend: 'first-visit V(s), blank = never observed', tip: s => `<div class="r"><span>first-visit V</span><span>${App.f(r.V_first_visit[s])}</span></div><div class="r"><span>every-visit V</span><span>${App.f(r.V_every_visit[s])}</span></div><div class="r"><span>visits</span><span>${r.N_every_visit[s]}</span></div>` });
    },
    async logs(el) {
      const f = $('#logFile', el).files[0]; if (!f) throw new Error('Choose a log CSV first.');
      const fd = new FormData(); fd.append('file', f); const r = await App.api('POST', '/rl/episodes/upload', fd);
      App.setData('lab-mc', { episodes_id: r.episodes_id, episodes_meta: r });
      $('#logOut', el).innerHTML = `<div class="note ${r.has_behaviour_probs ? 'ok' : 'warn'}">${r.n_episodes} episodes · ${r.n_steps} steps · ${esc(r.provenance)}<br>Enabled: ${r.capabilities.join(', ')}${r.warning ? '<br>' + esc(r.warning) : ''}</div><button class="btn sm" style="margin-top:8px" id="toMc">Use these logs in the Monte Carlo lab</button>`;
      $('#toMc', el).addEventListener('click', () => App.go('lab-mc'));
    },
  };
  App.viva['ops-data'] = () => ({ unit: 'Unit 1 / Unit 5', unitClass: 'u5', concept: 'State features from real data, and MC prediction on real sessions', prov: 'REAL DATA',
    why: 'The RL state must be computable from what a real match history contains; MC prediction is the only RL method that is valid on logs without actions.',
    inputs: 'Uploaded CSV (kills, deaths, result, date, player).', outputs: 'Bucketed states, per-match rewards, first/every-visit V(s).',
    formula: 'fatigue = (match index − 1)/matches in session;  tilt = 0.6·loss-streak/6 + 0.4·performance drop;  V(s) = mean of G<sub>t</sub> over visits to s',
    example: null, assumptions: 'Closeness = 0.5 and rage quit = proxy unless the file has scorelines/quit logs (labelled FALLBACK/PROXY).', code: 'feature_engine.py · monte_carlo.episodes_from_engineered · backend_app.mc_real' });

  // =================================================================== QUEUE & MATCHMAKING
  App.screens['ops-queue'] = {
    render(el) {
      el.innerHTML = `${App.header('Live queue & matchmaking', 'Players are embedded with a GraphSAGE-style neighbour aggregation and clustered by chemistry; a balanced 10-player lobby is drafted by MMR snake order plus a chemistry swap search.', [['ux', 'Ops']])}
      <div class="panel"><div class="row between"><h2>Queue <span id="qProv"></span></h2><div class="row"><button class="btn" id="qRefresh">New simulated queue</button><button class="btn primary" id="qMatch">Assemble balanced match</button></div></div>
      <p class="sub">Click a player to shadow them in the live session — their fatigue, tilt and streak become the starting RL state.</p>
      <div class="legend" id="qLegend" style="margin-bottom:10px"></div><div class="qgrid" id="qGrid"></div></div>
      <div class="panel hidden" id="qMatchPanel" style="margin-top:16px"><h2>Match result</h2><div class="grid g2" id="qTeams"></div><div class="kpis" id="qKpis" style="margin-top:12px"></div></div>`;
      $('#qRefresh', el).addEventListener('click', () => { Q.seed++; Q.refreshSim(); this.draw(el); $('#qMatchPanel', el).classList.add('hidden'); });
      $('#qMatch', el).addEventListener('click', () => this.match(el));
      this.draw(el);
    },
    onShow(el) { this.draw(el); },
    draw(el) {
      $('#qProv', el).innerHTML = App.prov(Q.source);
      $('#qLegend', el).innerHTML = Q.labels ? Object.entries(Q.labels).map(([c, m]) => `<span><i style="background:${CL[c % 5]}"></i>${esc(m.label)} · ${m.size}</span>`).join('') : [0, 1, 2, 3, 4].map(c => `<span><i style="background:${CL[c]}"></i>cluster ${c + 1}</span>`).join('');
      $('#qGrid', el).innerHTML = Q.pool.map((p, i) => { const c = CL[(Q.clusters[i] || 0) % 5]; const s = RL.stateFromFeatures(p.fatigue_index, p.tilt_score, p.win_streak); return `<div class="pcard" data-i="${i}" tabindex="0" role="button"><div class="nm"><span>${esc(p.name)}</span><i style="width:9px;height:9px;border-radius:50%;background:${c};display:inline-block"></i></div><div class="faint">${esc(p.archetype)} · ${p.mmr} MMR</div><div style="margin-top:6px;font-size:11px" class="dim">tilt ${App.f(p.tilt_score, 2)} · fatigue ${App.f(p.fatigue_index, 2)} · streak ${p.win_streak > 0 ? '+' : ''}${p.win_streak}</div><div class="bar"><i style="width:${p.tilt_score * 100}%;background:var(--amber)"></i></div><div class="bar"><i style="width:${p.fatigue_index * 100}%;background:var(--cyan)"></i></div><div class="faint mono" style="font-size:10.5px;margin-top:5px">RL state ${RL.stateShort(s)}</div></div>`; }).join('');
      $$('.pcard', el).forEach(c => { const go = () => { App.setData('ctl-session', { player: Q.pool[+c.dataset.i] }); App.go('ctl-session'); }; c.addEventListener('click', go); c.addEventListener('keydown', e => { if (e.key === 'Enter') go(); }); });
    },
    async match(el) {
      let r;
      if (Q.source === 'REAL DATA' && App.engine === 'python' && Q.datasetId) r = await App.api('POST', `/matchmake/${Q.datasetId}`, { slate_size: 10 });
      else r = slate(Q.pool, 10);
      const team = (t, cls, name) => `<div class="team ${cls}"><h3 style="margin-top:0">${name} · chemistry ${App.f(cls ? r.team_b_chemistry : r.team_a_chemistry, 2)}</h3>${t.map(p => `<div class="row between" style="padding:5px 0;border-bottom:1px dashed var(--line)"><span>${esc(p.name)}</span><span class="faint">${esc(p.archetype)} · ${p.mmr}</span></div>`).join('')}</div>`;
      $('#qTeams', el).innerHTML = team(r.team_a, '', 'Team Alpha') + team(r.team_b, 'b', 'Team Omega');
      $('#qKpis', el).innerHTML = App.kpi(r.mmr_gap, 'MMR gap (team totals)') + App.kpi(App.pct(r.predicted_close_match, 0), 'predicted close match', '1 − gap/1000') + App.kpi(App.f(r.team_a_chemistry, 2), 'Alpha chemistry') + App.kpi(App.f(r.team_b_chemistry, 2), 'Omega chemistry');
      $('#qMatchPanel', el).classList.remove('hidden');
    },
  };
  App.viva['ops-queue'] = () => ({ unit: 'Operations', unitClass: 'ux', concept: 'Chemistry clustering and balanced drafting (not RL)', prov: 'DERIVED FEATURE',
    why: 'Shows the matchmaking product the RL controller sits inside. Clustering and drafting are unsupervised/heuristic, so they are not claimed as RL.',
    inputs: 'Playstyle vector, tilt, fatigue, MMR.', outputs: 'Clusters, two 5-player teams, chemistry, closeness.',
    formula: 'h<sub>i</sub> ← ½h<sub>i</sub> + ½·mean<sub>j∈N(i)</sub> h<sub>j</sub> (2 hops) → k-means;  chemistry = 0.5 + role diversity − tilt penalty − 0.3·MMR spread',
    example: null, assumptions: 'k = 5 neighbours, 5 clusters; swap search keeps the MMR gap ≤ 400.', code: 'gnn_chemistry.py · assets/ops.js' });

  // =================================================================== TOURNAMENT
  const SAMPLE_PLAYERS = ['Vexen', 'Ryza', 'Kael', 'Nova', 'Zeph', 'Ashra', 'Tinker', 'Ori', 'Quill', 'Draven', 'Mira', 'Sable', 'Fenix', 'Lyric', 'Bram', 'Ixel', 'Corvo', 'Wynn', 'Talon', 'Juno', 'Rook', 'Nyx', 'Ember', 'Skye', 'Halo', 'Grim', 'Petal', 'Vantage', 'Onyx', 'Riven', 'Marz', 'Kite', 'Solace', 'Vega', 'Drift', 'Choc', 'Umbra', 'Zane', 'Frostt', 'Hollow', 'Pyre', 'Wisp', 'Cinder', 'Blitz', 'Marrow', 'Aster', 'Vale', 'Rune', 'Slate', 'Dune', 'Echo', 'Faust', 'Glint', 'Harrow', 'Ilex', 'Jinx'];
  const STYLES = ['Aggressive Entry', 'Passive Support', 'Objective Anchor', 'Flex Rotator', 'Sniper/Isolation', 'Rusher', 'Anchor', 'Support', 'Sniper', 'All-rounder'];
  function sampleMapsCsv() { const m = root.CLUTCH_STATIC.sample_maps; return 'rank,map_name,size,layout,best_modes,recommended_playstyle,sniper_friendly,description\n' + m.map(x => [x.rank, x.map_name, x.size, x.layout, `"${x.best_modes}"`, x.recommended_playstyle, x.sniper_friendly ? 'Yes' : 'No', `"${x.description.replace(/"/g, "'")}"`].join(',')).join('\n') + '\n'; }
  function samplePlayersCsv() { let seed = 7; const rng = () => { seed = (seed * 9301 + 49297) % 233280; return seed / 233280; }; return 'name,playstyle\n' + SAMPLE_PLAYERS.map(n => `${n},${STYLES[Math.floor(rng() * STYLES.length)]}`).join('\n') + '\n'; }
  App.screens['ops-tournament'] = {
    render(el) {
      el.innerHTML = `${App.header('Tournament', 'Upload a maps CSV and a players CSV: the engine snake-drafts balanced 5-player teams, schedules closest-skill matches and recommends a map for each. Each match can then be handed to the Bandit Arena to learn the map choice from feedback.', [['ux', 'Ops'], ['u2', 'Unit 2 hand-off']])}
      <div class="panel"><div class="row"><div class="field"><label>maps.csv</label><input type="file" id="tMaps" accept=".csv"></div><div class="field"><label>players.csv</label><input type="file" id="tPlayers" accept=".csv"></div><button class="btn" id="tSample">Use sample CSVs</button><button class="btn primary" id="tBuild">Build tournament</button></div><div id="tStatus" class="sub" style="margin-top:10px"></div></div>
      <div id="tOut" style="margin-top:16px"></div>`;
      $('#tSample', el).addEventListener('click', () => { const dt1 = new DataTransfer(); dt1.items.add(new File([sampleMapsCsv()], 'sample_maps.csv', { type: 'text/csv' })); $('#tMaps', el).files = dt1.files; const dt2 = new DataTransfer(); dt2.items.add(new File([samplePlayersCsv()], 'sample_players.csv', { type: 'text/csv' })); $('#tPlayers', el).files = dt2.files; $('#tStatus', el).textContent = 'Sample maps.csv (20 maps) and players.csv (56 players) loaded.'; });
      $('#tBuild', el).addEventListener('click', e => App.run(e.currentTarget, () => this.build(el)));
    },
    async build(el) {
      const m = $('#tMaps', el).files[0], p = $('#tPlayers', el).files[0]; if (!m || !p) throw new Error('Choose (or load the sample) maps.csv and players.csv first.');
      const fd = new FormData(); fd.append('maps_file', m); fd.append('players_file', p);
      const d = await App.api('POST', '/tournament/build', fd); App.data['ops-tournament'] = { t: d };
      $('#tStatus', el).innerHTML = `<span class="good">Built ${esc(d.tournament_id)}</span> — ${d.player_count} players → ${d.team_count} teams → ${d.match_count} matches${d.benched_players.length ? ` (benched: ${esc(d.benched_players.join(', '))})` : ''}`;
      const avgChem = d.teams.reduce((s, t) => s + t.chemistry_score, 0) / d.teams.length, avgR = d.matches.reduce((s, x) => s + x.reward, 0) / d.matches.length;
      $('#tOut', el).innerHTML = `<div class="kpis">${App.kpi(d.player_count, 'players drafted')}${App.kpi(d.team_count, 'teams')}${App.kpi(d.match_count, 'matches')}${App.kpi(App.f(avgChem, 3), 'avg chemistry')}${App.kpi(App.f(avgR, 3), 'avg projected reward')}</div>
        <div class="panel" style="margin-top:16px"><h2>Provenance</h2>${Object.entries(d.provenance || {}).map(([k, v]) => `<div class="note" style="margin-top:6px"><b>${esc(k)}</b>: ${esc(v)}</div>`).join('')}</div>
        <div class="panel" style="margin-top:16px"><h2>Scheduled matches</h2>${App.table([{ h: 'Match', k: 'match_id' }, { h: 'Teams', f: r => `${r.team_a} vs ${r.team_b}` }, { h: 'MMR gap', n: 1, f: r => r.mmr_gap }, { h: 'Closeness', n: 1, f: r => App.f(r.predicted_closeness, 3) }, { h: 'Chemistry', n: 1, f: r => App.f(r.avg_chemistry, 3) }, { h: 'Rule-based map', f: r => `<b>${esc(r.recommended_map.map_name)}</b> <span class="faint">fit ${r.recommended_map.fit_score}</span>` }, { h: 'Reward', n: 1, f: r => App.f(r.reward, 3) }, { h: 'Rage risk', f: r => r.rage_quit_risk ? '<span class="bad">elevated</span>' : '<span class="faint">low</span>' }, { h: '', f: r => `<button class="btn sm" data-bandit="${r.match_id}">Learn map with a bandit</button>` }], d.matches)}</div>
        <div class="panel" style="margin-top:16px"><h2>Drafted teams</h2><div class="grid g3">${d.teams.map(t => `<div class="team"><h3 style="margin-top:0">${t.team_id} · ${esc(t.dominant_playstyle)}</h3><div class="faint">avg MMR ${t.avg_mmr} · chemistry ${t.chemistry_score}</div><div style="margin-top:6px;font-size:12.5px">${t.players.map(esc).join(', ')}</div></div>`).join('')}</div></div>`;
      $$('[data-bandit]', el).forEach(b => b.addEventListener('click', () => { const match = d.matches.find(x => x.match_id === b.dataset.bandit); App.setData('lab-bandit', { arms: { source: 'tournament', tournament_id: d.tournament_id, match_id: match.match_id, k: 10, roster_archetypes: match.roster_archetypes, map_pool: d.map_pool, label: `${d.tournament_id} · ${match.match_id}` } }); App.rendered['lab-bandit'] = false; App.go('lab-bandit'); }));
    },
  };
  App.viva['ops-tournament'] = () => ({ unit: 'Unit 2 hand-off', unitClass: 'u2', concept: 'Rule-based map recommendation → learned with a bandit', prov: 'DERIVED FEATURE',
    why: 'The rule ranks maps by a fit score; a bandit learns which map actually satisfies players from match-by-match feedback.', inputs: 'Roster archetypes, map metadata.', outputs: 'Fit score per map; arm means q*(map) for the Bandit Arena.',
    formula: 'fit = 2·[map style = dominant style] + 1.5·[sniper-friendly ∧ sniper share > 25%] + 0.5·[¬sniper-friendly ∧ share ≤ 25%] + (21 − rank)/40;   q*(map) = 0.25 + 0.6·(fit − min)/(max − min)',
    example: null, assumptions: 'Satisfaction is monotone in fit (DERIVED FEATURE); per-match feedback is simulated.', code: 'tournament_builder.map_fit_score · bandits.map_arms' });

  // =================================================================== BURNOUT
  App.screens['ops-burnout'] = {
    render(el) {
      el.innerHTML = `${App.header('Burnout forecast & coach report', 'For a player from an uploaded dataset: a transparent linear trend of tilt across their matches and an offline, template-based coaching note.', [['ux', 'Ops']])}
      <div class="panel"><div class="row"><select id="bPlayer" style="max-width:320px"></select><button class="btn primary" id="bRun">Run forecast + report</button></div><p class="sub" id="bHint" style="margin-top:8px"></p></div>
      <div class="grid g2" style="margin-top:16px"><div class="panel"><h2>Burnout forecast ${App.prov('DERIVED FEATURE')}</h2><div id="bF" class="empty">Run a forecast.</div></div><div class="panel"><h2>Coach report</h2><div id="bR" class="empty">Run a forecast.</div></div></div>`;
      this.fill(el); $('#bRun', el).addEventListener('click', e => App.run(e.currentTarget, () => this.run(el)));
    },
    onShow(el) { this.fill(el); },
    fill(el) {
      const real = Q.source === 'REAL DATA' && Q.datasetId;
      $('#bPlayer', el).innerHTML = real ? Q.pool.map(p => `<option value="${esc(p.id)}">${esc(p.name)} — ${esc(p.archetype)}</option>`).join('') : '<option>Upload a dataset first</option>';
      $('#bHint', el).textContent = real ? '' : 'Forecasts need a real player history: upload a match-history CSV on the Data screen (Python backend).';
    },
    async run(el) {
      if (!Q.datasetId) throw new Error('Upload a match-history CSV first (Data & upload).');
      const pid = $('#bPlayer', el).value; const f = await App.api('GET', `/forecast/${Q.datasetId}/${encodeURIComponent(pid)}`);
      $('#bF', el).className = ''; $('#bF', el).innerHTML = f.status === 'ok' ? `<div class="kpis">${App.kpi(f.risk_level.toUpperCase(), 'risk level', '', f.risk_level === 'critical' ? 'bad' : f.risk_level === 'elevated' ? 'warn' : 'good')}${App.kpi(App.sgn(f.tilt_slope_per_session, 4), 'tilt slope / match')}${App.kpi(f.sessions_until_danger_zone ?? '—', 'matches to danger zone')}</div><div id="bChart" style="margin-top:10px"></div><p class="note" style="margin-top:8px">${esc(f.recommendation)}</p>` : `<div class="note warn">${esc(f.message)}</div>`;
      if (f.status === 'ok') C.line($('#bChart', el), { series: [{ name: 'projected tilt', points: f.projected_tilt_next_5_sessions.map((v, i) => [i + 1, v]), markers: true }], xLabel: 'matches ahead', yLabel: 'tilt', yMin: 0, yMax: 1, refLines: [{ y: 0.65, label: 'danger 0.65', color: '#ff4d6a' }], height: 200 });
      const r = await App.api('GET', `/report/${Q.datasetId}/${encodeURIComponent(pid)}`);
      $('#bR', el).className = ''; $('#bR', el).innerHTML = `<div style="font-size:13px;line-height:1.7">${esc(r.markdown).replace(/\*\*(.*?)\*\*/g, '<b>$1</b>').replace(/^## .*$/m, '').replace(/\n\n/g, '<br><br>').replace(/\n/g, '<br>')}</div>`;
    },
  };
  App.viva['ops-burnout'] = () => ({ unit: 'Operations', unitClass: 'ux', concept: 'Burnout trend (regression, not RL)', prov: 'DERIVED FEATURE', why: 'Gives the coach a forward-looking signal; deliberately a transparent linear model.', inputs: 'Per-match tilt and fatigue of one player.', outputs: 'Slope, matches to danger threshold, recommendation.', formula: 'tilt ≈ β₀ + β₁·index (least squares);  matches to danger = ⌈(0.65 − tilt<sub>now</sub>)/β₁⌉', example: null, assumptions: 'Short-horizon linear trend only.', code: 'burnout_forecast.py · coach_report.py' });
})(window);
