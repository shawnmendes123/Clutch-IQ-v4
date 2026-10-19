/* =========================================================================
   Clutch IQ — in-browser RL engine (offline mirror of the Python backend)
   -------------------------------------------------------------------------
   A faithful port of rl_core.py, mdp_model.py, dynamic_programming.py,
   monte_carlo.py, q_learning.py, rl_evaluation.py and bandits.py.

   It uses the same Mulberry32 PRNG and the same random-draw order as the
   Python code, so for the same seed it produces the same episodes, the same
   estimated model and the same learned tables (up to floating-point
   summation order). tests/test_js_parity.py checks this.

   ClutchRL.call(method, path, body) emulates the FastAPI endpoints with the
   same JSON shapes, so the UI renders identically with or without a server.
   ========================================================================= */
(function (root) {
  'use strict';

  // ----------------------------------------------------------------- PRNG
  const M32 = 0xFFFFFFFF;
  function deriveSeed(seed, k) {
    return (Math.imul((seed >>> 0) ^ 0x9E3779B9, 0x85EBCA6B) ^ Math.imul((k + 1) >>> 0, 0xC2B2AE35)) >>> 0;
  }
  class Mulberry32 {
    constructor(seed) { this.a = seed >>> 0; }
    random() {
      this.a = (this.a + 0x6D2B79F5) >>> 0;
      const a = this.a;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    }
    randint(lo, hi) { return lo + Math.floor(this.random() * (hi - lo + 1)); }
    choiceIndex(n) { return Math.floor(this.random() * n); }
    normal(mu = 0, sigma = 1) {
      let u1 = this.random(); const u2 = this.random();
      if (u1 < 1e-12) u1 = 1e-12;
      return mu + sigma * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }
    sampleDiscrete(probs) {
      const u = this.random(); let acc = 0;
      for (let i = 0; i < probs.length; i++) { acc += probs[i]; if (u < acc) return i; }
      return probs.length - 1;
    }
  }

  // --------------------------------------------------------- state space
  const FATIGUE_LEVELS = ['FRESH', 'WARM', 'TIRED', 'EXHAUSTED'];
  const TILT_LEVELS = ['CALM', 'UNEASY', 'TILTED', 'VOLATILE'];
  const STREAK_LEVELS = ['LOSING', 'NEUTRAL', 'WINNING'];
  const NF = 4, NT = 4, NK = 3, NS = 48, TERMINAL = 48, NS1 = 49, NA = 3;
  const ACTION_NAMES = ['LOWER', 'KEEP', 'RAISE'];
  const ACTION_LABELS = ['Lower Difficulty', 'Keep Steady', 'Raise Difficulty'];
  const ACTION_DELTA = [-1, 0, 1];
  const W1 = 1.0, W2 = 1.2, W3 = 3.0;

  const bucketFatigue = f => Math.max(0, Math.min(3, Math.floor(f * 4)));
  const bucketTilt = t => Math.max(0, Math.min(3, Math.floor(t * 4)));
  const bucketStreak = k => (k <= -2 ? 0 : (k >= 2 ? 2 : 1));
  const encode = (f, t, k) => f * (NT * NK) + t * NK + k;
  const decode = s => (s === TERMINAL ? [-1, -1, -1] : [Math.floor(s / (NT * NK)), Math.floor(s / NK) % NT, s % NK]);
  const stateFromFeatures = (f, t, k) => encode(bucketFatigue(f), bucketTilt(t), bucketStreak(k));
  function stateKey(s) { const [f, t, k] = decode(s); return `${f}-${t}-${k}`; }
  function keyToState(key) { const [f, t, k] = key.split('-').map(Number); return encode(f, t, k); }
  function stateLabel(s) { if (s === TERMINAL) return 'END'; const [f, t, k] = decode(s); return `${FATIGUE_LEVELS[f]} · ${TILT_LEVELS[t]} · ${STREAK_LEVELS[k]}`; }
  function stateShort(s) { if (s === TERMINAL) return 'END'; const [f, t, k] = decode(s); return `F${f}T${t}${'LNW'[k]}`; }
  function stateInfo(s) {
    if (s === TERMINAL) return { index: s, key: 'END', label: 'END (session over)', terminal: true };
    const [f, t, k] = decode(s);
    return { index: s, key: `${f}-${t}-${k}`, short: stateShort(s), label: stateLabel(s), fatigue_bucket: f, tilt_bucket: t, streak_bucket: k,
      fatigue: FATIGUE_LEVELS[f], tilt: TILT_LEVELS[t], streak: STREAK_LEVELS[k], fatigue_range: [f / 4, (f + 1) / 4],
      tilt_range: [t / 4, (t + 1) / 4], streak_range: ['<= -2', '-1..+1', '>= +2'][k], terminal: false };
  }

  // -------------------------------------------------------------- reward
  const rewardFunction = (len, close, rage) => W1 * len + W2 * close - W3 * (rage ? 1 : 0);

  // ----------------------------------------------------- simulator params
  const ENV_PARAMS = {
    p_win_base: 0.5, streak_w: 0.20, fatigue_w: 0.03, tilt_w: 0.06, action_w: 0.15, p_win_min: 0.05, p_win_max: 0.95,
    rage_base: 0.005, rage_tilt: 0.025, rage_fatigue: 0.01, rage_losing_streak: 0.02, rage_raise: 0.02, rage_stomp: 0.04,
    stomp_threshold: 0.30, boredom_threshold: 0.70, boredom_base: 0.06, boredom_hot: 0.04,
    tilt_up_loss: 0.30, tilt_up_stomp_bonus: 0.25, tilt_down_win: 0.35, tilt_down_close_bonus: 0.20, close_threshold: 0.80,
    streak_step: 0.50, fatigue_up: [0.33, 0.33, 0.36],
  };
  const P_ = ENV_PARAMS;
  function pWin(f, t, k, a) {
    const p = P_.p_win_base + P_.streak_w * (k - 1) - P_.fatigue_w * f - P_.tilt_w * t - P_.action_w * ACTION_DELTA[a];
    return Math.min(P_.p_win_max, Math.max(P_.p_win_min, p));
  }
  const closeness = pw => 1 - 2 * Math.abs(pw - 0.5);
  function rageProbAfterLoss(f, t, k, a, pw) {
    const q = P_.rage_base + P_.rage_tilt * t + P_.rage_fatigue * f + (k === 0 ? P_.rage_losing_streak : 0)
      + (a === 2 ? P_.rage_raise : 0) + (pw < P_.stomp_threshold ? P_.rage_stomp : 0);
    return Math.min(0.95, Math.max(0, q));
  }
  function boredomProbAfterWin(f, t, k, a, pw) {
    if (pw < P_.boredom_threshold) return 0;
    return P_.boredom_base + (k === 2 ? P_.boredom_hot : 0);
  }
  function stepModel(s, a) {
    const [f, t, k] = decode(s); const pw = pWin(f, t, k, a); const C = closeness(pw);
    const q = rageProbAfterLoss(f, t, k, a, pw); const b = boredomProbAfterWin(f, t, k, a, pw);
    return { p_win: pw, closeness: C, rage_given_loss: q, boredom_given_win: b, rage_prob: (1 - pw) * q, boredom_prob: pw * b,
      fatigue_up: P_.fatigue_up[a],
      expected_reward_immediate: (1 - pw) * q * rewardFunction(0, C, true) + pw * b * rewardFunction(0, C, false)
        + (1 - (1 - pw) * q - pw * b) * rewardFunction(1, C, false) };
  }

  /** One transition. Draws exactly five uniforms (fixed order). */
  function envStep(s, a, rng) {
    const [f, t, k] = decode(s);
    const pw = pWin(f, t, k, a); const C = closeness(pw);
    const uWin = rng.random(), uRage = rng.random(), uTilt = rng.random(), uStreak = rng.random(), uFat = rng.random();
    const win = uWin < pw;
    const rp = (1 - pw) * rageProbAfterLoss(f, t, k, a, pw);
    if (win) {
      if (uRage < boredomProbAfterWin(f, t, k, a, pw))
        return [TERMINAL, rewardFunction(0, C, false), true, { win: true, rage_quit: false, natural_end: false, p_win: pw, closeness: C, rage_prob: rp, boredom_quit: true }];
    } else if (uRage < rageProbAfterLoss(f, t, k, a, pw)) {
      return [TERMINAL, rewardFunction(0, C, true), true, { win: false, rage_quit: true, natural_end: false, p_win: pw, closeness: C, rage_prob: rp, boredom_quit: false }];
    }
    let t2, k2;
    if (win) {
      const pDown = P_.tilt_down_win + (C >= P_.close_threshold ? P_.tilt_down_close_bonus : 0);
      t2 = (t > 0 && uTilt < pDown) ? t - 1 : t;
      k2 = k === 0 ? 1 : ((k === 1 && uStreak < P_.streak_step) ? 2 : k);
    } else {
      const pUp = P_.tilt_up_loss + (pw < P_.stomp_threshold ? P_.tilt_up_stomp_bonus : 0);
      t2 = uTilt < pUp ? Math.min(3, t + 1) : t;
      k2 = k === 2 ? 1 : ((k === 1 && uStreak < P_.streak_step) ? 0 : k);
    }
    const r = rewardFunction(1, C, false);
    const info = { win, rage_quit: false, natural_end: false, p_win: pw, closeness: C, rage_prob: rp, boredom_quit: false };
    let f2 = f;
    if (uFat < P_.fatigue_up[a]) {
      if (f === 3) { info.natural_end = true; return [TERMINAL, r, true, info]; }
      f2 = f + 1;
    }
    return [encode(f2, t2, k2), r, false, info];
  }

  function exactTransition(s, a) {
    const [f, t, k] = decode(s); const pw = pWin(f, t, k, a); const C = closeness(pw); const q = rageProbAfterLoss(f, t, k, a, pw);
    const probs = new Map(), mass = new Map();
    const add = (s2, p, r) => { if (p <= 0) return; probs.set(s2, (probs.get(s2) || 0) + p); mass.set(s2, (mass.get(s2) || 0) + p * r); };
    const rRage = rewardFunction(0, C, true), rBored = rewardFunction(0, C, false), rOk = rewardFunction(1, C, false);
    const pf = P_.fatigue_up[a];
    for (const [win, pOut] of [[true, pw], [false, 1 - pw]]) {
      if (pOut <= 0) continue;
      let pQuit, tiltOut, streakOut;
      if (win) { pQuit = boredomProbAfterWin(f, t, k, a, pw); add(TERMINAL, pOut * pQuit, rBored); }
      else { pQuit = q; add(TERMINAL, pOut * pQuit, rRage); }
      const pCont = pOut * (1 - pQuit);
      if (win) {
        const pDown = P_.tilt_down_win + (C >= P_.close_threshold ? P_.tilt_down_close_bonus : 0);
        tiltOut = t > 0 ? [[t - 1, pDown], [t, 1 - pDown]] : [[t, 1]];
        streakOut = k === 0 ? [[1, 1]] : (k === 1 ? [[2, P_.streak_step], [1, 1 - P_.streak_step]] : [[2, 1]]);
      } else {
        const pUp = P_.tilt_up_loss + (pw < P_.stomp_threshold ? P_.tilt_up_stomp_bonus : 0);
        tiltOut = t < 3 ? [[Math.min(3, t + 1), pUp], [t, 1 - pUp]] : [[t, 1]];
        streakOut = k === 2 ? [[1, 1]] : (k === 1 ? [[0, P_.streak_step], [1, 1 - P_.streak_step]] : [[0, 1]]);
      }
      for (const [t2, pt] of tiltOut) for (const [k2, pk] of streakOut) {
        const base = pCont * pt * pk;
        if (f === 3) add(TERMINAL, base * pf, rOk); else add(encode(f + 1, t2, k2), base * pf, rOk);
        add(encode(f, t2, k2), base * (1 - pf), rOk);
      }
    }
    const rewards = new Map(); for (const [s2, p] of probs) rewards.set(s2, mass.get(s2) / p);
    return [probs, rewards];
  }

  // ---------------------------------------------------- start distributions
  function startDistribution(name = 'uniform') {
    if (name === 'uniform') return new Array(NS).fill(1 / NS);
    const pf = [0.70, 0.30, 0, 0], pt = [0.50, 0.30, 0.15, 0.05], pk = [0.20, 0.60, 0.20];
    const d = new Array(NS).fill(0);
    for (let f = 0; f < NF; f++) for (let t = 0; t < NT; t++) for (let k = 0; k < NK; k++) d[encode(f, t, k)] = pf[f] * pt[t] * pk[k];
    return d;
  }
  const sampleStartState = (rng, name = 'uniform') => (name === 'uniform' ? rng.choiceIndex(NS) : rng.sampleDiscrete(startDistribution(name)));

  // ------------------------------------------------------------- policies
  const uniformPolicy = () => Array.from({ length: NS1 }, () => [1 / NA, 1 / NA, 1 / NA]);
  const constantPolicy = a => Array.from({ length: NS1 }, () => [0, 1, 2].map(b => (b === a ? 1 : 0)));
  const deterministicPolicy = acts => Array.from({ length: NS1 }, (_, s) => { const a = s < acts.length ? acts[s] : 1; return [0, 1, 2].map(b => (b === a ? 1 : 0)); });
  function heuristicActions() {
    const acts = [];
    for (let s = 0; s < NS; s++) { const [f, t, k] = decode(s); acts.push((t >= 2 || k === 0) ? 0 : ((t === 0 && k === 2 && f <= 1) ? 2 : 1)); }
    return acts;
  }
  const heuristicPolicy = () => deterministicPolicy(heuristicActions());
  const epsilonSoft = (greedy, eps) => Array.from({ length: NS1 }, (_, s) => { const g = s < greedy.length ? greedy[s] : 1; return [0, 1, 2].map(a => (a === g ? 1 - eps + eps / NA : eps / NA)); });
  function argmaxRow(row) { let best = 0; for (let a = 1; a < row.length; a++) if (row[a] > row[best] + 1e-12) best = a; return best; }
  const greedyFromQ = Q => Q.map(argmaxRow);
  function policyActions(pi) { return pi.map(row => { let best = 0; for (let a = 1; a < NA; a++) if (row[a] > row[best]) best = a; return best; }); }

  // -------------------------------------------------------------- episodes
  function returnsFromRewards(rewards, gamma) { let G = 0; const out = new Array(rewards.length); for (let i = rewards.length - 1; i >= 0; i--) { G = rewards[i] + gamma * G; out[i] = G; } return out; }
  function discountedReturn(rewards, gamma) { let G = 0; for (let i = rewards.length - 1; i >= 0; i--) G = rewards[i] + gamma * G; return G; }
  const MAX_EPISODE_STEPS = 200;
  function generateEpisode(policy, seed, opts = {}) {
    const envRng = new Mulberry32(deriveSeed(seed, 0)); const actRng = new Mulberry32(deriveSeed(seed, 1));
    let s = (opts.startState !== undefined && opts.startState !== null) ? opts.startState : sampleStartState(envRng, opts.startDist || 'uniform');
    const trans = [];
    for (let step = 0; step < MAX_EPISODE_STEPS; step++) {
      let a, bprob;
      if (step === 0 && opts.firstAction !== undefined && opts.firstAction !== null) { a = opts.firstAction; bprob = null; }
      else { const row = policy[s]; a = actRng.sampleDiscrete(row); bprob = row[a]; }
      const [s2, r, done, info] = envStep(s, a, envRng);
      const extra = opts.recordInfo ? { win: info.win, rage_quit: info.rage_quit, natural_end: info.natural_end, p_win: +info.p_win.toFixed(4), closeness: +info.closeness.toFixed(4) } : {};
      trans.push({ state: s, action: a, reward: r, next_state: s2, done, behaviour_prob: bprob, info: extra });
      s = s2;
      if (done) break;
    }
    return { transitions: trans, source: 'SIMULATED', meta: { seed } };
  }

  // =================================================================== MDP
  class FiniteMDP {
    constructor(P, R, terminal, gamma = 0.9, names = null, actionNames = null, meta = {}) {
      this.P = P; this.R = R; this.terminal = terminal; this.gamma = gamma;
      this.nS = P.length; this.nA = P[0].length;
      this.stateNames = names || P.map((_, i) => String(i)); this.actionNames = actionNames || P[0].map((_, i) => String(i));
      this.meta = meta;
      for (let s = 0; s < this.nS; s++) if (terminal[s]) for (let a = 0; a < this.nA; a++) { this.P[s][a] = new Array(this.nS).fill(0); this.P[s][a][s] = 1; this.R[s][a] = new Array(this.nS).fill(0); }
      this._rsa = null;
    }
    get r_sa() {
      if (!this._rsa) this._rsa = this.P.map((rowA, s) => rowA.map((row, a) => { let x = 0; for (let j = 0; j < this.nS; j++) x += row[j] * this.R[s][a][j]; return x; }));
      return this._rsa;
    }
    withGamma(g) { const m = Object.create(FiniteMDP.prototype); Object.assign(m, this); m.gamma = g; return m; }
    qFromV(V, g) {
      g = g === undefined ? this.gamma : g;
      const Q = new Array(this.nS);
      for (let s = 0; s < this.nS; s++) {
        Q[s] = new Array(this.nA);
        for (let a = 0; a < this.nA; a++) { const p = this.P[s][a], r = this.R[s][a]; let x = 0; for (let j = 0; j < this.nS; j++) if (p[j] !== 0) x += p[j] * (r[j] + g * V[j]); Q[s][a] = x; }
      }
      return Q;
    }
    policyMarkovChain(pi) {
      return this.P.map((rowA, s) => { const out = new Array(this.nS).fill(0); for (let a = 0; a < this.nA; a++) { const w = pi[s][a]; if (w) for (let j = 0; j < this.nS; j++) out[j] += w * rowA[a][j]; } return out; });
    }
    policyMRP(pi) { const rsa = this.r_sa; return [this.policyMarkovChain(pi), rsa.map((row, s) => row.reduce((x, v, a) => x + pi[s][a] * v, 0))]; }
    mrpClosedForm(pi, g) {
      g = g === undefined ? this.gamma : g;
      const [Ppi, Rpi] = this.policyMRP(pi);
      const idx = []; for (let s = 0; s < this.nS; s++) if (!this.terminal[s]) idx.push(s);
      const n = idx.length; const A = idx.map((s, i) => idx.map((s2, j) => (i === j ? 1 : 0) - g * Ppi[s][s2])); const b = idx.map(s => Rpi[s]);
      const x = solveLinear(A, b); const v = new Array(this.nS).fill(0); idx.forEach((s, i) => { v[s] = x[i]; }); return v;
    }
    sequenceProbability(states, pi, start) { const P = this.policyMarkovChain(pi); let p = start ? start[states[0]] : 1; for (let i = 1; i < states.length; i++) p *= P[states[i - 1]][states[i]]; return p; }
    check() { let mx = 0; for (const rowA of this.P) for (const row of rowA) mx = Math.max(mx, Math.abs(row.reduce((x, y) => x + y, 0) - 1)); return { rows_sum_to_one: mx < 1e-9, max_row_error: mx }; }
  }
  function solveLinear(A, b) { // Gaussian elimination with partial pivoting
    const n = b.length; const M = A.map((row, i) => row.slice().concat([b[i]]));
    for (let c = 0; c < n; c++) {
      let p = c; for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
      [M[c], M[p]] = [M[p], M[c]];
      for (let r = c + 1; r < n; r++) { const f = M[r][c] / M[c][c]; if (f) for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k]; }
    }
    const x = new Array(n).fill(0);
    for (let r = n - 1; r >= 0; r--) { let s = M[r][n]; for (let k = r + 1; k < n; k++) s -= M[r][k] * x[k]; x[r] = s / M[r][r]; }
    return x;
  }
  const terminalMask = () => Array.from({ length: NS1 }, (_, i) => i === TERMINAL);
  const shortNames = () => Array.from({ length: NS1 }, (_, i) => stateShort(i));
  const zeros3 = () => Array.from({ length: NS1 }, () => Array.from({ length: NA }, () => new Array(NS1).fill(0)));

  const _modelCache = new Map();
  function estimateModel(samples = 1000, seed = 7) {
    const key = `${samples}|${seed}`; if (_modelCache.has(key)) return _modelCache.get(key);
    const t0 = now();
    const counts = zeros3(), rsum = zeros3();
    for (let s = 0; s < NS; s++) for (let a = 0; a < NA; a++) {
      const rng = new Mulberry32(deriveSeed(seed, 1000 + s * NA + a));
      const c = counts[s][a], rs = rsum[s][a];
      for (let i = 0; i < samples; i++) { const [s2, r] = envStep(s, a, rng); c[s2] += 1; rs[s2] += r; }
    }
    const P = zeros3(), R = zeros3();
    for (let s = 0; s < NS1; s++) for (let a = 0; a < NA; a++) {
      const tot = counts[s][a].reduce((x, y) => x + y, 0) || 1;
      for (let j = 0; j < NS1; j++) { P[s][a][j] = counts[s][a][j] / tot; R[s][a][j] = counts[s][a][j] > 0 ? rsum[s][a][j] / counts[s][a][j] : 0; }
    }
    const m = new FiniteMDP(P, R, terminalMask(), 0.9, shortNames(), ACTION_NAMES.slice(),
      { kind: 'SIMULATION-ESTIMATED TRANSITION MODEL', provenance: 'MODEL ESTIMATE', samples_per_pair: samples, seed,
        total_simulated_transitions: samples * NS * NA, build_seconds: (now() - t0) / 1000 });
    m.counts = counts;
    _modelCache.set(key, m);
    return m;
  }
  let _exact = null;
  function exactModel() {
    if (_exact) return _exact;
    const P = zeros3(), R = zeros3();
    for (let s = 0; s < NS; s++) for (let a = 0; a < NA; a++) { const [pr, rw] = exactTransition(s, a); for (const [s2, p] of pr) { P[s][a][s2] = p; R[s][a][s2] = rw.get(s2); } }
    _exact = new FiniteMDP(P, R, terminalMask(), 0.9, shortNames(), ACTION_NAMES.slice(), { kind: 'SIMULATOR GROUND TRUTH (validation only)', provenance: 'SIMULATED' });
    return _exact;
  }
  function getModel(samples = 1000, seed = 7, gamma = 0.9) { const base = estimateModel(samples, seed); const m = base.withGamma(gamma); m.counts = base.counts; return m; }
  function modelQuality(m) {
    const ex = exactModel(); let mx = 0, sum = 0, n = 0, rmx = 0, nz = 0;
    const er = ex.r_sa, mr = m.r_sa;
    for (let s = 0; s < NS; s++) for (let a = 0; a < NA; a++) {
      for (let j = 0; j < NS1; j++) { const d = Math.abs(m.P[s][a][j] - ex.P[s][a][j]); mx = Math.max(mx, d); if (ex.P[s][a][j] > 0) { sum += d; n++; } if (m.P[s][a][j] > 0) nz++; }
      rmx = Math.max(rmx, Math.abs(mr[s][a] - er[s][a]));
    }
    const ns = m.meta.samples_per_pair || 1;
    return { max_abs_transition_error: mx, mean_abs_transition_error: sum / n, max_expected_reward_error: rmx,
      worst_case_std_error: Math.sqrt(0.25 / ns), nonzero_transitions_per_pair_mean: nz / (NS * NA), rows_sum_to_one: m.check().rows_sum_to_one };
  }
  function topTransitions(m, s, a, k = 6) {
    const row = m.P[s][a]; const order = row.map((p, j) => [p, j]).sort((x, y) => y[0] - x[0] || x[1] - y[1]).map(x => x[1]);
    const out = [];
    for (const s2 of order.slice(0, k)) { if (row[s2] <= 0) break; const it = { next_state: s2, label: stateLabel(s2), short: stateShort(s2), p: row[s2], r: m.R[s][a][s2] }; if (m.counts) it.count = m.counts[s][a][s2]; out.push(it); }
    const restIdx = order.slice(k); const rest = restIdx.reduce((x, j) => x + row[j], 0);
    if (rest > 1e-12) out.push({ next_state: null, label: `${restIdx.filter(j => row[j] > 0).length} other states`, short: 'other', p: rest, r: null });
    return out;
  }

  // ===================================================== dynamic programming
  function asPolicy(pi, nS, nA) { if (typeof pi[0] === 'number') return pi.map(a => Array.from({ length: nA }, (_, b) => (b === a ? 1 : 0))); return pi; }
  function expectationBackup(mdp, V, pi, g) { const Q = mdp.qFromV(V, g); return Q.map((row, s) => (mdp.terminal[s] ? 0 : row.reduce((x, q, a) => x + pi[s][a] * q, 0))); }
  function policyEvaluation(mdp, pi, g, theta = 1e-6, maxIter = 1000, V0 = null, inPlace = false, recordSnapshots = 0) {
    g = g === undefined || g === null ? mdp.gamma : g; pi = asPolicy(pi, mdp.nS, mdp.nA);
    let V = V0 ? V0.slice() : new Array(mdp.nS).fill(0);
    const deltas = [], snaps = []; const t0 = now(); let converged = false, it = 0;
    for (it = 1; it <= maxIter; it++) {
      let delta = 0;
      if (inPlace) {
        for (let s = 0; s < mdp.nS; s++) { if (mdp.terminal[s]) continue; const v = V[s]; let x = 0; for (let a = 0; a < mdp.nA; a++) { if (!pi[s][a]) continue; const p = mdp.P[s][a], r = mdp.R[s][a]; let q = 0; for (let j = 0; j < mdp.nS; j++) if (p[j]) q += p[j] * (r[j] + g * V[j]); x += pi[s][a] * q; } V[s] = x; delta = Math.max(delta, Math.abs(v - x)); }
      } else {
        const nv = expectationBackup(mdp, V, pi, g); for (let s = 0; s < mdp.nS; s++) delta = Math.max(delta, Math.abs(nv[s] - V[s])); V = nv;
      }
      deltas.push(delta); if (it <= recordSnapshots) snaps.push(V.slice());
      if (delta < theta) { converged = true; break; }
    }
    return { V, iterations: Math.min(it, maxIter), deltas, converged, snapshots: snaps, seconds: (now() - t0) / 1000, gamma: g, theta };
  }
  function greedyPolicy(mdp, V, g, oldActions = null, tol = 1e-9) {
    const Q = mdp.qFromV(V, g);
    const best = Q.map(row => { let b = 0; for (let a = 1; a < row.length; a++) if (row[a] > row[b]) b = a; return b; });
    if (oldActions) for (let s = 0; s < mdp.nS; s++) if (Q[s][oldActions[s]] >= Q[s][best[s]] - tol) best[s] = oldActions[s];
    return [best, Q];
  }
  function policyImprovement(mdp, V, acts, g) {
    const [nw, Q] = greedyPolicy(mdp, V, g, acts);
    const changed = []; for (let s = 0; s < mdp.nS; s++) if (nw[s] !== acts[s] && !mdp.terminal[s]) changed.push(s);
    return { actions: nw, Q, changed_states: changed, policy_stable: changed.length === 0 };
  }
  function policyIteration(mdp, g, theta = 1e-6, maxIter = 100, evalMaxIter = 1000, initial = null) {
    g = g === undefined || g === null ? mdp.gamma : g;
    let acts = initial ? initial.slice() : new Array(mdp.nS).fill(1); let V = new Array(mdp.nS).fill(0);
    const history = []; const t0 = now(); let total = 0, stable = false;
    for (let k = 1; k <= maxIter; k++) {
      const e = policyEvaluation(mdp, acts, g, theta, evalMaxIter, V); V = e.V; total += e.iterations;
      const imp = policyImprovement(mdp, V, acts, g);
      history.push({ iteration: k, evaluation_sweeps: e.iterations, final_delta: e.deltas[e.deltas.length - 1], changed_states: imp.changed_states, n_changed: imp.changed_states.length, policy: acts.slice(), V: V.slice() });
      acts = imp.actions; if (imp.policy_stable) { stable = true; break; }
    }
    return { V, Q: mdp.qFromV(V, g), actions: acts, history, policy_stable: stable, improvement_steps: history.length, evaluation_sweeps_total: total, seconds: (now() - t0) / 1000, gamma: g, theta };
  }
  function valueIteration(mdp, g, theta = 1e-6, maxIter = 2000, recordSnapshots = 0, inPlace = false) {
    g = g === undefined || g === null ? mdp.gamma : g;
    let V = new Array(mdp.nS).fill(0); const deltas = [], snaps = []; const t0 = now(); let converged = false, it = 0;
    for (it = 1; it <= maxIter; it++) {
      let delta = 0;
      if (inPlace) {
        for (let s = 0; s < mdp.nS; s++) { if (mdp.terminal[s]) continue; const v = V[s]; let best = -Infinity; for (let a = 0; a < mdp.nA; a++) { const p = mdp.P[s][a], r = mdp.R[s][a]; let q = 0; for (let j = 0; j < mdp.nS; j++) if (p[j]) q += p[j] * (r[j] + g * V[j]); if (q > best) best = q; } V[s] = best; delta = Math.max(delta, Math.abs(v - best)); }
      } else {
        const Q = mdp.qFromV(V, g); const nv = Q.map((row, s) => (mdp.terminal[s] ? 0 : Math.max(...row)));
        for (let s = 0; s < mdp.nS; s++) delta = Math.max(delta, Math.abs(nv[s] - V[s])); V = nv;
      }
      deltas.push(delta); if (it <= recordSnapshots) snaps.push(V.slice());
      if (delta < theta) { converged = true; break; }
    }
    const [actions, Q] = greedyPolicy(mdp, V, g);
    const ratios = []; for (let i = 0; i + 1 < deltas.length; i++) if (deltas[i] > 0) ratios.push(deltas[i + 1] / deltas[i]);
    return { V, Q, actions, iterations: Math.min(it, maxIter), deltas, converged, contraction_ratios: ratios, max_contraction_ratio: ratios.length ? Math.max(...ratios) : 0, snapshots: snaps, seconds: (now() - t0) / 1000, gamma: g, theta };
  }
  function bellmanResidual(mdp, V, g, pi = null) {
    let TV; if (pi === null) TV = mdp.qFromV(V, g).map(r => Math.max(...r)); else TV = expectationBackup(mdp, V, asPolicy(pi, mdp.nS, mdp.nA), g);
    let mx = 0; for (let s = 0; s < mdp.nS; s++) { if (mdp.terminal[s]) TV[s] = 0; mx = Math.max(mx, Math.abs(TV[s] - V[s])); } return mx;
  }
  function backupBreakdown(mdp, s, V, g, pi = null, topK = 6, labels = null) {
    labels = labels || mdp.stateNames;
    const per = [];
    for (let a = 0; a < mdp.nA; a++) {
      const row = mdp.P[s][a];
      const order = row.map((p, j) => [p, j]).filter(x => x[0] > 0).sort((x, y) => y[0] - x[0] || x[1] - y[1]).map(x => x[1]);
      const terms = order.map(s2 => ({ next_state: s2, label: labels[s2], p: row[s2], r: mdp.R[s][a][s2], V_next: V[s2], contribution: row[s2] * (mdp.R[s][a][s2] + g * V[s2]) }));
      const q = terms.reduce((x, t) => x + t.contribution, 0); const rest = terms.slice(topK);
      per.push({ action: a, action_name: mdp.actionNames[a], q, expected_reward: mdp.r_sa[s][a], expected_next_value: row.reduce((x, p, j) => x + p * V[j], 0),
        terms: terms.slice(0, topK), other_terms: { count: rest.length, p: rest.reduce((x, t) => x + t.p, 0), contribution: rest.reduce((x, t) => x + t.contribution, 0) } });
    }
    const qs = per.map(p => p.q); let am = 0; for (let a = 1; a < qs.length; a++) if (qs[a] > qs[am]) am = a;
    const out = { state: s, gamma: g, per_action: per, optimality: { max_q: Math.max(...qs), argmax: am } };
    if (pi) { pi = asPolicy(pi, mdp.nS, mdp.nA); out.expectation = { pi: pi[s].slice(), value: qs.reduce((x, q, a) => x + pi[s][a] * q, 0) }; }
    return out;
  }

  // =========================================================== Monte Carlo
  function sar(ep) {
    if (ep.transitions) return [ep.transitions.map(t => t.state), ep.transitions.map(t => t.action), ep.transitions.map(t => t.reward)];
    if (ep.states) return [ep.states, ep.actions || ep.states.map(() => null), ep.rewards];
    return [ep[0], ep[0].map(() => null), ep[1]];
  }
  function firstVisitPrediction(eps, g, n) {
    const N = new Array(n).fill(0), Gs = new Array(n).fill(0);
    for (const ep of eps) { const [S, , R] = sar(ep); const G = returnsFromRewards(R, g); const seen = new Set(); S.forEach((s, t) => { if (seen.has(s)) return; seen.add(s); N[s]++; Gs[s] += G[t]; }); }
    return { V: Gs.map((x, i) => (N[i] > 0 ? x / N[i] : 0)), N };
  }
  function everyVisitPrediction(eps, g, n) {
    const N = new Array(n).fill(0), Gs = new Array(n).fill(0);
    for (const ep of eps) { const [S, , R] = sar(ep); const G = returnsFromRewards(R, g); S.forEach((s, t) => { N[s]++; Gs[s] += G[t]; }); }
    return { V: Gs.map((x, i) => (N[i] > 0 ? x / N[i] : 0)), N };
  }
  function episodeTable(ep, g) {
    const [S, A, R] = sar(ep); const G = returnsFromRewards(R, g);
    return S.map((s, t) => { const tr = ep.transitions[t]; return { t, state: s, state_label: stateShort(s), action: A[t], action_name: A[t] === null ? null : ACTION_NAMES[A[t]], reward: +R[t].toFixed(4), next_state: tr.next_state, done: tr.done, G_t: +G[t].toFixed(4), discount: +Math.pow(g, t).toFixed(6), info: tr.info }; });
  }
  const DEFAULT_CHECKPOINTS = [10, 25, 50, 100, 250, 500, 1000, 2000, 5000, 10000, 20000];
  function maeMask(V, ref, mask) { let s = 0, n = 0; for (let i = 0; i < NS; i++) if (mask[i]) { s += Math.abs(V[i] - ref[i]); n++; } return n ? s / n : null; }
  function mcPredictionRun(policy, nEp, g, seed, startDist = 'uniform', refV = null, checkpoints = null) {
    const t0 = now();
    const cps = new Set((checkpoints || DEFAULT_CHECKPOINTS).filter(c => c <= nEp).concat([nEp]));
    const Nf = new Array(NS1).fill(0), Gf = new Array(NS1).fill(0), Ne = new Array(NS1).fill(0), Ge = new Array(NS1).fill(0);
    const curve = [], lengths = [], rets = []; let sample = null;
    for (let i = 0; i < nEp; i++) {
      const ep = generateEpisode(policy, deriveSeed(seed, i), { startDist, recordInfo: i === 0 }); if (i === 0) sample = ep;
      const [S, , R] = sar(ep); const G = returnsFromRewards(R, g); lengths.push(S.length); rets.push(G[0]);
      const seen = new Set();
      S.forEach((s, t) => { Ne[s]++; Ge[s] += G[t]; if (!seen.has(s)) { seen.add(s); Nf[s]++; Gf[s] += G[t]; } });
      if (cps.has(i + 1)) {
        const Vf = Gf.map((x, j) => (Nf[j] > 0 ? x / Nf[j] : 0)), Ve = Ge.map((x, j) => (Ne[j] > 0 ? x / Ne[j] : 0));
        const mask = Nf.slice(0, NS).map(x => x > 0);
        const row = { episodes: i + 1, coverage: mask.filter(Boolean).length / NS };
        if (refV) { row.mae_first_visit = maeMask(Vf, refV, mask); row.mae_every_visit = maeMask(Ve, refV, mask); }
        curve.push(row);
      }
    }
    const Vf = Gf.map((x, j) => (Nf[j] > 0 ? x / Nf[j] : 0)), Ve = Ge.map((x, j) => (Ne[j] > 0 ? x / Ne[j] : 0));
    return { V_first_visit: Vf, V_every_visit: Ve, N_first_visit: Nf, N_every_visit: Ne, curve, sample_episode: sample ? episodeTable(sample, g) : [],
      mean_return: mean(rets), std_return: std(rets, 0), mean_length: mean(lengths), n_episodes: nEp, seconds: (now() - t0) / 1000, provenance: 'SIMULATED' };
  }
  function policyValueUnderModel(acts, model, g, startDist) {
    const full = acts.concat(new Array(model.nS - acts.length).fill(1));
    const e = policyEvaluation(model, full, g, 1e-6, 500); const d0 = startDistribution(startDist).concat([0]);
    return [d0.reduce((x, p, i) => x + p * e.V[i], 0), e.V];
  }
  function argmaxFirst(row) { let b = 0; for (let a = 1; a < row.length; a++) if (row[a] > row[b]) b = a; return b; }
  function mcControl(variant = 'on_policy', nEp = 5000, g = 0.9, eps = 0.1, seed = 7, startDist = 'uniform', refActions = null, model = null, nCheck = 20, behEps = 0.3) {
    const t0 = now();
    const Q = Array.from({ length: NS1 }, () => [0, 0, 0]), N = Array.from({ length: NS1 }, () => [0, 0, 0]), C = Array.from({ length: NS1 }, () => [0, 0, 0]);
    const greedy = new Array(NS1).fill(1); const every = Math.max(1, Math.floor(nEp / nCheck));
    const history = [], rets = []; const rngEs = new Mulberry32(deriveSeed(seed, 777)); let used = 0;
    for (let i = 0; i < nEp; i++) {
      const es = deriveSeed(seed, i); let ep, bpol = null;
      if (variant === 'es') { const s0 = rngEs.choiceIndex(NS); const a0 = rngEs.choiceIndex(NA); ep = generateEpisode(deterministicPolicy(greedy), es, { startState: s0, firstAction: a0 }); }
      else if (variant === 'on_policy') ep = generateEpisode(epsilonSoft(greedy, eps), es, { startDist });
      else if (variant === 'off_policy') { bpol = epsilonSoft(greedy, behEps); ep = generateEpisode(bpol, es, { startDist }); }
      else throw new Error('variant must be es | on_policy | off_policy');
      const [S, A, R] = sar(ep); rets.push(discountedReturn(R, g));
      if (variant !== 'off_policy') {
        const G = returnsFromRewards(R, g); const first = new Map();
        S.forEach((s, t) => { const k = s * 3 + A[t]; if (!first.has(k)) first.set(k, t); });
        for (const [k, t] of first) { const s = Math.floor(k / 3), a = k % 3; N[s][a]++; Q[s][a] += (G[t] - Q[s][a]) / N[s][a]; greedy[s] = argmaxFirst(Q[s]); }
      } else {
        let G = 0, W = 1;
        for (let t = S.length - 1; t >= 0; t--) {
          const s = S[t], a = A[t]; G = g * G + R[t]; C[s][a] += W; N[s][a]++; Q[s][a] += (W / C[s][a]) * (G - Q[s][a]); greedy[s] = argmaxFirst(Q[s]); used++;
          if (a !== greedy[s]) break; W = W / bpol[s][a];
        }
      }
      if ((i + 1) % every === 0 || i + 1 === nEp) {
        const row = { episodes: i + 1, mean_return_window: mean(rets.slice(-every)) };
        if (refActions) row.agreement_with_reference = agreement(greedy, refActions);
        if (model) row.greedy_policy_value = policyValueUnderModel(greedy.slice(0, NS), model, g, 'session_start')[0];
        history.push(row);
      }
    }
    let unvisited = 0; for (let s = 0; s < NS; s++) for (let a = 0; a < NA; a++) if (N[s][a] === 0) unvisited++;
    const out = { variant, Q, N, actions: greedy.slice(0, NS), history, n_episodes: nEp, gamma: g, epsilon: eps, behaviour_epsilon: variant === 'off_policy' ? behEps : null,
      seconds: (now() - t0) / 1000, provenance: 'SIMULATED', unvisited_pairs: unvisited };
    if (variant === 'off_policy') { out.cumulative_weights = C; out.transitions_used = used; }
    return out;
  }
  function coverageCheck(target, beh) { const bad = []; for (let s = 0; s < NS; s++) for (let a = 0; a < NA; a++) if (target[s][a] > 0 && beh[s][a] <= 0) bad.push({ state: s, action: a }); return { satisfied: bad.length === 0, violations: bad.slice(0, 20), n_violations: bad.length }; }
  function offPolicyPrediction(target, beh, nEp, g, seed, startDist = 'session_start', refValue = null, refV = null, checkpoints = null) {
    const t0 = now(); const cov = coverageCheck(target, beh);
    if (!cov.satisfied) return { error: 'coverage assumption violated: target takes actions the behaviour policy never takes', coverage: cov };
    const num = new Array(NS1).fill(0), den = new Array(NS1).fill(0), cnt = new Array(NS1).fill(0);
    const rhos = [], G0s = [], wG0 = [], pdis = [], daN = [], daD = [];
    const cps = new Set((checkpoints || DEFAULT_CHECKPOINTS).filter(c => c <= nEp).concat([nEp]));
    const curve = [], samples = [];
    for (let i = 0; i < nEp; i++) {
      const ep = generateEpisode(beh, deriveSeed(seed, i), { startDist }); const [S, A, R] = sar(ep); const G = returnsFromRewards(R, g); const T = S.length;
      const rhoT = new Array(T).fill(0); let acc = 1; for (let t = T - 1; t >= 0; t--) { acc *= target[S[t]][A[t]] / beh[S[t]][A[t]]; rhoT[t] = acc; }
      const seen = new Set(); S.forEach((s, t) => { if (seen.has(s)) return; seen.add(s); num[s] += rhoT[t] * G[t]; den[s] += rhoT[t]; cnt[s]++; });
      rhos.push(rhoT[0]); G0s.push(G[0]); wG0.push(rhoT[0] * G[0]);
      const pref = []; acc = 1; for (let k = 0; k < T; k++) { acc *= target[S[k]][A[k]] / beh[S[k]][A[k]]; pref.push(acc); }
      let pd = 0; for (let k = 0; k < T; k++) pd += Math.pow(g, k) * pref[k] * R[k]; pdis.push(pd);
      const flat = []; let cs = 0; for (const r of R) { cs += r; flat.push(cs); }
      let n_ = 0, d_ = 0; for (let h = 1; h < T; h++) { const w = (1 - g) * Math.pow(g, h - 1) * pref[h - 1]; n_ += w * flat[h - 1]; d_ += w; }
      const w = Math.pow(g, T - 1) * pref[T - 1]; n_ += w * flat[T - 1]; d_ += w; daN.push(n_); daD.push(d_);
      if (i < 6) samples.push({ episode: i, length: T, G0: +G[0].toFixed(4), rho: rhoT[0], steps: S.slice(0, 12).map((s, t) => ({ t, state: stateShort(s), action: ACTION_NAMES[A[t]], pi: target[s][A[t]], b: beh[s][A[t]], ratio: target[s][A[t]] / beh[s][A[t]], reward: +R[t].toFixed(4) })) });
      if (cps.has(i + 1)) {
        const n = i + 1, wsum = sum(rhos), dd = sum(daD);
        curve.push({ episodes: n, ordinary_is: sum(wG0) / n, weighted_is: wsum > 0 ? sum(wG0) / wsum : null, per_decision_is: mean(pdis), discounting_aware_ois: mean(daN), discounting_aware_wis: dd > 0 ? sum(daN) / dd : null, behaviour_mean_return: mean(G0s) });
      }
    }
    const sq = rhos.reduce((x, r) => x + r * r, 0); const ess = sq > 0 ? (sum(rhos) ** 2) / sq : 0; const last = curve[curve.length - 1];
    const Vo = num.map((x, i) => (cnt[i] > 0 ? x / cnt[i] : 0)), Vw = num.map((x, i) => (den[i] > 0 ? x / den[i] : 0));
    const out = { J_ordinary_is: last.ordinary_is, J_weighted_is: last.weighted_is, J_per_decision_is: last.per_decision_is, J_discounting_aware_ois: last.discounting_aware_ois, J_discounting_aware_wis: last.discounting_aware_wis,
      reliability: (ess >= 0.05 * nEp && ess >= 100) ? 'OK' : 'LOW — effective sample size is small; importance weights are heavy-tailed, treat the estimate with caution',
      J_behaviour_on_policy: mean(G0s), reference_J_target: refValue, curve, V_ordinary_is: Vo, V_weighted_is: Vw, visits: cnt,
      rho_stats: { mean: mean(rhos), max: rhos.reduce((m, r) => (r > m ? r : m), -Infinity), fraction_nonzero: rhos.filter(r => r > 0).length / rhos.length, effective_sample_size: ess, histogram: histLog(rhos) },
      coverage: cov, sample_episodes: samples, n_episodes: nEp, seconds: (now() - t0) / 1000, provenance: 'SIMULATED' };
    if (refV) { out.mae_ordinary_is = maeMask(Vo, refV, cnt.slice(0, NS).map(x => x > 0)); out.mae_weighted_is = maeMask(Vw, refV, den.slice(0, NS).map(x => x > 0)); }
    return out;
  }
  function histLog(x, bins = 12) {
    const nz = x.filter(v => v > 0); const zero = x.length - nz.length; if (!nz.length) return { zero, edges: [], counts: [] };
    let lo = Math.log10(nz.reduce((m, v) => (v < m ? v : m), Infinity)), hi = Math.log10(nz.reduce((m, v) => (v > m ? v : m), -Infinity)); if (hi - lo < 1e-9) hi = lo + 1;
    const edges = Array.from({ length: bins + 1 }, (_, i) => Math.pow(10, lo + (hi - lo) * i / bins)); const counts = new Array(bins).fill(0);
    for (const v of nz) { let b = Math.floor((Math.log10(v) - lo) / (hi - lo) * bins); if (b >= bins) b = bins - 1; if (b < 0) b = 0; counts[b]++; }
    return { zero, edges, counts, log_scale: true };
  }

  // ============================================================ Q-learning
  const Q_DEFAULT = { episodes: 30000, alpha: 0.05, gamma: 0.9, eps_start: 1.0, eps_decay: 0.9998, eps_min: 0.05, seed: 7, start_dist: 'uniform' };
  function epsilonGreedy(Qs, eps, rng) { if (rng.random() < eps) return [rng.choiceIndex(NA), true]; return [argmaxRow(Qs), false]; }
  function tdUpdate(Q, s, a, r, s2, done, alpha, g) {
    const old = Q[s][a]; const maxNext = done ? 0 : Math.max(...Q[s2]); const target = r + (done ? 0 : g * maxNext); const err = target - old; Q[s][a] = old + alpha * err;
    return { state: s, state_label: stateShort(s), action: a, action_name: ACTION_NAMES[a], reward: r, next_state: s2, next_label: stateShort(s2), done, old_q: old, max_next_q: maxNext, td_target: target, td_error: err, alpha, gamma: g, new_q: Q[s][a] };
  }
  function qTrain(p = {}) {
    p = Object.assign({}, Q_DEFAULT, p);
    const { episodes, alpha, gamma, eps_start, eps_decay, eps_min, seed } = p; const startDist = p.start_dist || 'uniform';
    const refActions = p.reference_actions || null, model = p.model || null, nCheck = p.n_checkpoints || 40, tdLast = p.td_log_last || 30, watch = (p.watch_state === undefined ? null : p.watch_state);
    const t0 = now(); const Q = Array.from({ length: NS1 }, () => [0, 0, 0]); const visits = Array.from({ length: NS1 }, () => [0, 0, 0]);
    let eps = eps_start; const every = Math.max(1, Math.floor(episodes / nCheck));
    const curve = [], tdLog = [], watchLog = [], er = [], ed = [], el = [], ta = []; let explore = 0, exploit = 0, firstUpdate = null;
    for (let ep = 0; ep < episodes; ep++) {
      const es = deriveSeed(seed, ep); const envRng = new Mulberry32(deriveSeed(es, 0)), actRng = new Mulberry32(deriveSeed(es, 1));
      let s = sampleStartState(envRng, startDist); let G = 0, Gd = 0, disc = 1, n = 0, abs = 0;
      for (let step = 0; step < MAX_EPISODE_STEPS; step++) {
        const [a, ex] = epsilonGreedy(Q[s], eps, actRng); if (ex) explore++; else exploit++;
        const [s2, r, done] = envStep(s, a, envRng); const u = tdUpdate(Q, s, a, r, s2, done, alpha, gamma); visits[s][a]++;
        if (!firstUpdate) firstUpdate = Object.assign({}, u, { episode: ep, epsilon: eps });
        if (watch !== null && s === watch && watchLog.length < 200) watchLog.push(Object.assign({}, u, { episode: ep }));
        abs += Math.abs(u.td_error); G += r; Gd += disc * r; disc *= gamma; n++; s = s2; if (done) break;
      }
      er.push(G); ed.push(Gd); el.push(n); ta.push(abs / Math.max(1, n));
      if ((ep + 1) % every === 0 || ep + 1 === episodes) {
        const greedy = greedyFromQ(Q.slice(0, NS));
        const row = { episodes: ep + 1, epsilon: eps, mean_return: mean(er.slice(-every)), mean_discounted_return: mean(ed.slice(-every)), mean_length: mean(el.slice(-every)), mean_abs_td_error: mean(ta.slice(-every)) };
        if (refActions) row.agreement_with_reference = agreement(greedy, refActions);
        if (model) row.greedy_policy_value = policyValueUnderModel(greedy, model, gamma, 'session_start')[0];
        curve.push(row);
      }
      eps = Math.max(eps_min, eps * eps_decay);
    }
    const rQ = Q.map(r => r.slice()); const envRng = new Mulberry32(deriveSeed(deriveSeed(seed, episodes + 1), 0)); let s = sampleStartState(envRng, 'session_start');
    for (let i = 0; i < tdLast; i++) { const a = argmaxRow(rQ[s]); const [s2, r, done] = envStep(s, a, envRng); tdLog.push(tdUpdate(rQ, s, a, r, s2, done, alpha, gamma)); s = s2; if (done) s = sampleStartState(envRng, 'session_start'); }
    return { Q, actions: greedyFromQ(Q.slice(0, NS)), curve, td_log: tdLog, first_update: firstUpdate, watch_log: watchLog, explore_steps: explore, exploit_steps: exploit, final_epsilon: eps,
      params: { episodes, alpha, gamma, eps_start, eps_decay, eps_min, seed, start_dist: startDist }, seconds: (now() - t0) / 1000, provenance: 'SIMULATED', visits };
  }
  let PRODUCTION_Q = null, PRODUCTION_META = null;
  function loadProduction() {
    if (PRODUCTION_Q) return [PRODUCTION_Q, PRODUCTION_META];
    const D = root.CLUTCH_STATIC || {};
    if (D.q_table) {
      PRODUCTION_Q = Array.from({ length: NS1 }, () => [0, 0, 0]);
      for (const [k, v] of Object.entries(D.q_table)) PRODUCTION_Q[keyToState(k)] = v.slice();
      PRODUCTION_META = D.q_meta || { algorithm: 'Tabular Q-learning' };
    } else { const r = qTrain(Q_DEFAULT); PRODUCTION_Q = r.Q; PRODUCTION_META = { algorithm: 'Tabular Q-learning', params: r.params, note: 'trained in-browser' }; }
    return [PRODUCTION_Q, PRODUCTION_META];
  }

  // ============================================================ evaluation
  function mean(x) { return x.length ? x.reduce((a, b) => a + b, 0) / x.length : 0; }
  function sum(x) { return x.reduce((a, b) => a + b, 0); }
  function std(x, ddof = 1) { if (x.length <= ddof) return 0; const m = mean(x); return Math.sqrt(x.reduce((a, b) => a + (b - m) * (b - m), 0) / (x.length - ddof)); }
  const ci95 = x => (x.length > 1 ? 1.96 * std(x, 1) / Math.sqrt(x.length) : 0);
  const summ = x => ({ mean: mean(x), std: x.length > 1 ? std(x, 1) : 0, ci95: ci95(x) });
  function agreement(a, b) { let n = 0; for (let s = 0; s < NS; s++) if (a[s] === b[s]) n++; return n / NS; }
  function disagreementAnalysis(acts, ref, Qref) {
    const out = []; for (let s = 0; s < NS; s++) { const a = acts[s], r = ref[s]; if (a !== r) out.push({ state: s, label: stateLabel(s), short: stateShort(s), action: ACTION_NAMES[a], reference_action: ACTION_NAMES[r], q_gap: Qref[s][r] - Qref[s][a] }); }
    return out.sort((x, y) => y.q_gap - x.q_gap);
  }
  function evaluatePolicy(policy, n = 1000, seed = 2026, g = 0.9, startDist = 'session_start', keepRaw = false) {
    const ret = [], dret = [], len = [], close = [], rage = [], bored = [], natural = [], mc = [];
    for (let i = 0; i < n; i++) {
      const ep = generateEpisode(policy, deriveSeed(seed, i), { startDist, recordInfo: true }); const R = ep.transitions.map(t => t.reward);
      ret.push(sum(R)); dret.push(discountedReturn(R, g)); len.push(R.length);
      const cs = ep.transitions.map(t => t.info.closeness); close.push(mean(cs.map(c => (c >= 0.8 ? 1 : 0)))); mc.push(mean(cs));
      const last = ep.transitions[ep.transitions.length - 1].info;
      rage.push(last.rage_quit ? 1 : 0); natural.push(last.natural_end ? 1 : 0); bored.push(!last.rage_quit && !last.natural_end ? 1 : 0);
    }
    const out = { n_episodes: n, seed, gamma: g, start_dist: startDist, episodic_return: summ(ret), discounted_return: summ(dret), session_length: summ(len), close_match_rate: summ(close), mean_closeness: summ(mc), rage_quit_rate: summ(rage), boredom_quit_rate: summ(bored), natural_end_rate: summ(natural), provenance: 'SIMULATED' };
    if (keepRaw) out._raw_discounted = dret;
    return out;
  }
  function modelValue(model, acts, g, startDist = 'session_start') { const full = acts.slice(0, NS).concat([1]); const V = policyEvaluation(model, full, g, 1e-8, 2000).V; const d0 = startDistribution(startDist).concat([0]); return d0.reduce((x, p, i) => x + p * V[i], 0); }
  function trajectory(policy, seed, g, startState = null, startDist = 'session_start', Q = null) {
    const ep = generateEpisode(policy, seed, { startState, startDist, recordInfo: true }); const R = ep.transitions.map(t => t.reward); const Gt = returnsFromRewards(R, g);
    let cum = 0, dcum = 0;
    const steps = ep.transitions.map((tr, t) => { cum += tr.reward; const c = Math.pow(g, t) * tr.reward; dcum += c; const row = { t, state: tr.state, state_label: stateLabel(tr.state), state_short: stateShort(tr.state), action: tr.action, action_name: ACTION_NAMES[tr.action], pi: policy[tr.state], reward: tr.reward, next_state: tr.next_state, next_label: stateLabel(tr.next_state), done: tr.done, discount: Math.pow(g, t), discounted_contribution: c, cumulative_reward: cum, discounted_cumulative: dcum, G_t: Gt[t], info: tr.info }; if (Q) row.q = Q[tr.state].slice(); return row; });
    const last = ep.transitions[ep.transitions.length - 1].info;
    return { steps, G0: Gt.length ? Gt[0] : 0, total_reward: cum, length: steps.length, terminated_by: last.rage_quit ? 'rage quit' : (last.natural_end ? 'natural end (exhausted)' : 'boredom quit'), gamma: g, seed, provenance: 'SIMULATED' };
  }
  function stability(lists) { if (lists.length < 2) return 1; const v = []; for (let i = 0; i < lists.length; i++) for (let j = i + 1; j < lists.length; j++) v.push(agreement(lists[i], lists[j])); return mean(v); }
  function comparePolicies(o = {}) {
    const n = o.n_episodes || 500, seed = o.seed === undefined ? 2026 : o.seed, g = o.gamma === undefined ? 0.9 : o.gamma, sd = o.start_dist || 'session_start';
    const samples = o.samples || 1000, mcEp = o.mc_episodes || 10000, mcEps = o.mc_epsilon === undefined ? 0.1 : o.mc_epsilon, qEp = o.q_episodes || 30000, runs = o.stability_runs || 1;
    const model = getModel(samples, 7, g); const tModel = model.meta.build_seconds || 0; const vi = valueIteration(model, g, 1e-8); const ref = vi.actions.slice(0, NS);
    const entries = [];
    const add = (key, name, family, actions, policy, seconds, conv, stab, prov, notes) => entries.push({ key, name, family, actions: actions.slice(0, NS), policy_matrix: policy, compute_seconds: seconds, convergence: conv, stability: stab, provenance: prov, notes });
    add('random', 'Random baseline', 'baseline', policyActions(uniformPolicy()), uniformPolicy(), 0, 'n/a', null, 'SIMULATED', 'uniform pi(a|s)=1/3 (stochastic)');
    add('keep', 'Always KEEP (v3-style rank-only matchmaking)', 'baseline', new Array(NS).fill(1), constantPolicy(1), 0, 'n/a', 1, 'SIMULATED', 'ignores fatigue/tilt/streak');
    const hp = heuristicPolicy(); add('heuristic', 'Manual rule (designer heuristic)', 'baseline', policyActions(hp), hp, 0, 'n/a', 1, 'SIMULATED', 'LOWER if tilted or losing; RAISE if calm, winning and fresh');
    const vstab = runs > 1 ? stability(Array.from({ length: runs }, (_, j) => valueIteration(getModel(samples, 7 + j, g), g, 1e-8).actions.slice(0, NS))) : null;
    add('dp_value_iteration', 'Dynamic Programming — Value Iteration', 'model-based', ref, deterministicPolicy(ref), tModel + vi.seconds, `${vi.iterations} sweeps to theta=1e-8`, vstab, 'MODEL ESTIMATE', `plans on the simulation-estimated model (${samples} samples/pair); time includes model estimation`);
    const pir = policyIteration(model, g, 1e-8);
    add('dp_policy_iteration', 'Dynamic Programming — Policy Iteration', 'model-based', pir.actions.slice(0, NS), deterministicPolicy(pir.actions), tModel + pir.seconds, `${pir.improvement_steps} improvements / ${pir.evaluation_sweeps_total} eval sweeps`, null, 'MODEL ESTIMATE', 'should match value iteration up to exact ties');
    const mcr = Array.from({ length: runs }, (_, j) => mcControl('on_policy', mcEp, g, mcEps, 7 + j));
    add('mc_control', `Monte Carlo Control (on-policy, eps=${mcEps})`, 'model-free', mcr[0].actions, deterministicPolicy(mcr[0].actions), mcr[0].seconds, `${mcEp} episodes`, runs > 1 ? stability(mcr.map(r => r.actions)) : null, 'SIMULATED', 'greedy policy w.r.t. learned Q; learned from complete episodes');
    const qp = Object.assign({}, Q_DEFAULT, { episodes: qEp, gamma: g }); const qr = [];
    for (let j = 0; j < runs; j++) {
      const p = Object.assign({}, qp, { seed: qp.seed + j });
      if (j === 0 && JSON.stringify(p) === JSON.stringify(Q_DEFAULT)) { const [Qp, meta] = loadProduction(); qr.push({ actions: greedyFromQ(Qp.slice(0, NS)), seconds: meta.trained_seconds || 0 }); }
      else qr.push(qTrain(p));
    }
    add('q_learning', 'Tabular Q-Learning', 'model-free', qr[0].actions, deterministicPolicy(qr[0].actions), qr[0].seconds, `${qEp} episodes, alpha=${qp.alpha}`, runs > 1 ? stability(qr.map(r => r.actions)) : null, 'SIMULATED', 'off-policy TD control, epsilon-greedy behaviour');
    const raw = {};
    for (const e of entries) {
      const t0 = now(); const ev = evaluatePolicy(e.policy_matrix, n, seed, g, sd, true); raw[e.key] = ev._raw_discounted; delete ev._raw_discounted;
      e.evaluation = ev; e.evaluation_seconds = (now() - t0) / 1000; e.agreement_with_dp = agreement(e.actions, ref);
      e.model_value_J = e.key === 'random' ? (() => { const V = policyEvaluation(model, uniformPolicy(), g, 1e-8).V; const d0 = startDistribution(sd).concat([0]); return d0.reduce((x, p, i) => x + p * V[i], 0); })() : modelValue(model, e.actions, g, sd);
      e.disagreements = (e.family !== 'baseline' || e.key === 'heuristic') ? disagreementAnalysis(e.actions, ref, vi.Q) : [];
      delete e.policy_matrix;
    }
    const base = raw.dp_value_iteration; for (const e of entries) { const d = raw[e.key].map((x, i) => x - base[i]); e.paired_diff_vs_dp = { mean: mean(d), ci95: ci95(d) }; }
    const keys = entries.map(e => e.key);
    return { policies: entries, agreement_matrix: { keys, values: entries.map(a => entries.map(b => agreement(a.actions, b.actions))) },
      settings: { n_episodes: n, seed, gamma: g, start_dist: sd, model_samples: samples, model_seed: 7, mc_episodes: mcEp, mc_epsilon: mcEps, q_params: qp, stability_runs: runs, common_random_numbers: true }, provenance: 'SIMULATED' };
  }

  // =============================================================== bandits
  const BANDIT_ALGOS = { random: 'Explore only (uniform random)', greedy: 'Greedy (exploit only)', explore_first: 'Fixed exploration + greedy', epsilon_greedy: 'ε-greedy', optimistic: 'Optimistic greedy', ucb: 'UCB' };
  class BanditEnv {
    constructor(means, rewardModel = 'bernoulli', sigma = 0.1, nonstat = 'none', driftSigma = 0.01, rounds = 1000, seed = 1) {
      this.means0 = means.slice(); this.means = means.slice(); this.k = means.length; this.rewardModel = rewardModel; this.sigma = sigma; this.nonstat = nonstat; this.driftSigma = driftSigma; this.rounds = rounds;
      this.rng = new Mulberry32(seed); this.driftRng = new Mulberry32(deriveSeed(seed, 99)); this.t = 0; this.perm = null;
      if (nonstat === 'patch') {
        const prng = new Mulberry32(deriveSeed(seed, 98)); const perm = Array.from({ length: this.k }, (_, i) => i);
        for (let i = this.k - 1; i > 0; i--) { const j = prng.choiceIndex(i + 1); [perm[i], perm[j]] = [perm[j], perm[i]]; }
        const best = argmaxFirst(this.means0); if (perm[best] === best && this.k > 1) { const o = (best + 1) % this.k; [perm[best], perm[o]] = [perm[o], perm[best]]; }
        this.perm = perm;
      }
    }
    pull(a) { const m = this.means[a]; return this.rewardModel === 'bernoulli' ? (this.rng.random() < m ? 1 : 0) : this.rng.normal(m, this.sigma); }
    advance() {
      this.t++;
      if (this.nonstat === 'drift') this.means = this.means.map(m => Math.min(0.98, Math.max(0.02, m + this.driftRng.normal(0, this.driftSigma))));
      else if (this.nonstat === 'patch' && this.t === Math.floor(this.rounds / 2)) this.means = this.means0.map((_, i) => this.means0[this.perm[i]]);
    }
  }
  function bArgmax(Q, rng = null) {
    const m = Math.max(...Q);
    if (!rng) { for (let a = 0; a < Q.length; a++) if (Q[a] >= m - 1e-12) return a; }
    const ties = []; for (let a = 0; a < Q.length; a++) if (Q[a] >= m - 1e-12) ties.push(a);
    return ties.length === 1 ? ties[0] : ties[rng.choiceIndex(ties.length)];
  }
  const ucbScores = (Q, N, t, c) => Q.map((q, i) => (N[i] === 0 ? Infinity : q + c * Math.sqrt(Math.log(t) / N[i])));
  function runBandit(alg, means, rounds = 1000, seed = 1, o = {}) {
    const eps = o.epsilon === undefined ? 0.1 : o.epsilon, c = o.c === undefined ? 1.414 : o.c, q0 = o.q0 || 0, alpha = o.alpha || null, exploreRounds = o.explore_rounds === undefined ? 50 : o.explore_rounds;
    const recordEvery = o.record_q_every === undefined ? 1 : o.record_q_every;
    const env = new BanditEnv(means, o.reward_model || 'bernoulli', o.sigma || 0.1, o.nonstationary || 'none', o.drift_sigma === undefined ? 0.01 : o.drift_sigma, rounds, seed);
    const rng = new Mulberry32(deriveSeed(seed, 7)); const k = env.k; const init = alg === 'optimistic' ? q0 : 0;
    const Q = new Array(k).fill(init), N = new Array(k).fill(0);
    const arms = [], rewards = [], reg = [], cr = [], opt = [], expl = [], qh = []; let ucbLast = null, cumR = 0, cumReg = 0, nEx = 0, nXp = 0;
    for (let t = 1; t <= rounds; t++) {
      let a, explored = false;
      if (alg === 'random') { a = rng.choiceIndex(k); explored = true; }
      else if (alg === 'greedy' || alg === 'optimistic') a = bArgmax(Q, rng);
      else if (alg === 'explore_first') { if (t <= exploreRounds) { a = (t - 1) % k; explored = true; } else a = bArgmax(Q, rng); }
      else if (alg === 'epsilon_greedy') { if (rng.random() < eps) { a = rng.choiceIndex(k); explored = true; } else a = bArgmax(Q, rng); }
      else { const sc = ucbScores(Q, N, t, c); a = bArgmax(sc, rng); explored = Q[a] < Math.max(...Q) - 1e-12; ucbLast = { t, Q: Q.slice(), N: N.slice(), bonus: N.map(n => (n === 0 ? null : c * Math.sqrt(Math.log(t) / n))), scores: sc.map(x => (x === Infinity ? null : x)), chosen: a }; }
      const best = Math.max(...env.means); const r = env.pull(a); N[a]++; const step = alpha ? alpha : 1 / N[a]; Q[a] = Q[a] + step * (r - Q[a]);
      cumR += r; cumReg += best - env.means[a]; arms.push(a); rewards.push(r); cr.push(cumR); reg.push(cumReg); opt.push(env.means[a] >= best - 1e-12 ? 1 : 0); expl.push(explored ? 1 : 0);
      if (explored) nEx++; else nXp++;
      if (recordEvery && (t % recordEvery === 0 || t === rounds)) qh.push(Q.map(x => +x.toFixed(5)));
      env.advance();
    }
    return { algorithm: alg, name: BANDIT_ALGOS[alg], arms, rewards, cumulative_reward: cr, cumulative_regret: reg, optimal: opt, explore: expl, Q, N, q_history: qh, n_explore: nEx, n_exploit: nXp, final_means: env.means, ucb_last: ucbLast,
      realised_regret: (o.nonstationary || 'none') === 'none' ? rounds * Math.max(...env.means0) - cumR : null, best_arm_final: argmaxFirst(env.means), total_reward: cumR, total_regret: cumReg };
  }
  function runArena(algs, means, rounds = 1000, runs = 50, seed = 1, o = {}) {
    const t0 = now(); const results = {};
    for (const alg of algs) {
      const reg = new Float64Array(rounds), rew = new Float64Array(rounds), opt = new Float64Array(rounds), expl = new Float64Array(rounds); const totals = []; let first = null;
      for (let j = 0; j < runs; j++) {
        const r = runBandit(alg, means, rounds, deriveSeed(seed, j), Object.assign({}, o, { record_q_every: j === 0 ? Math.max(1, Math.floor(rounds / 200)) : 0 }));
        for (let i = 0; i < rounds; i++) { reg[i] += r.cumulative_regret[i]; rew[i] += r.cumulative_reward[i]; opt[i] += r.optimal[i]; expl[i] += r.explore[i]; }
        totals.push(r.total_reward); if (j === 0) first = r;
      }
      const n = runs, stride = Math.max(1, Math.floor(rounds / 200)); const idx = []; for (let i = stride - 1; i < rounds; i += stride) idx.push(i); if (idx[idx.length - 1] !== rounds - 1) idx.push(rounds - 1);
      const win = (arr, i) => { const lo = Math.max(0, i - stride + 1); let s = 0; for (let j = lo; j <= i; j++) s += arr[j]; return s / (i - lo + 1); };
      const tail = Math.max(1, Math.floor(rounds / 10)); let ts = 0; for (let i = rounds - tail; i < rounds; i++) ts += opt[i];
      results[alg] = { name: BANDIT_ALGOS[alg], rounds_index: idx.map(i => i + 1), mean_cumulative_regret: idx.map(i => reg[i] / n), mean_cumulative_reward: idx.map(i => rew[i] / n),
        pct_optimal: idx.map(i => win(opt, i) / n), explore_rate: idx.map(i => win(expl, i) / n), final_regret: reg[rounds - 1] / n, final_reward: rew[rounds - 1] / n,
        final_reward_std: runs > 1 ? std(totals, 1) : 0, final_reward_ci95: runs > 1 ? 1.96 * std(totals, 1) / Math.sqrt(runs) : 0, final_pct_optimal: ts / tail / n,
        example_run: { Q: first.Q, N: first.N, q_history: first.q_history, n_explore: first.n_explore, n_exploit: first.n_exploit, ucb_last: first.ucb_last, final_means: first.final_means, realised_regret: first.realised_regret, total_reward: first.total_reward, total_regret: first.total_regret,
          arms: first.arms.slice(0, Math.min(rounds, 400)), rewards: first.rewards.slice(0, Math.min(rounds, 400)), q_history_stride: Math.max(1, Math.floor(rounds / 200)) } };
    }
    return { results, rounds, runs, seed, means: means.slice(), settings: o, seconds: (now() - t0) / 1000, provenance: { arm_means: 'DERIVED FEATURE', rewards: 'SIMULATED' } };
  }
  const ARCH_KW = [['Aggressive Entry', ['aggress', 'entry', 'rush', 'slayer', 'duelist', 'assault']], ['Passive Support', ['support', 'passive', 'healer', 'utility']], ['Objective Anchor', ['objective', 'anchor', 'hold', 'defend', 'obj']], ['Sniper/Isolation', ['sniper', 'scout', 'isolat', 'flank', 'recon']], ['Flex Rotator', ['flex', 'rotat', 'hybrid', 'all-round', 'allround']]];
  function closestArchetype(text) { if (!text) return 'Flex Rotator'; const t = String(text).toLowerCase(); for (const [a, kws] of ARCH_KW) if (kws.some(k => t.includes(k))) return a; return 'Flex Rotator'; }
  function mapFitScore(m, roster) {
    const counts = {}; roster.forEach(p => { counts[p.archetype] = (counts[p.archetype] || 0) + 1; });
    const mx = Math.max(...Object.values(counts)); const dominant = Object.keys(counts).filter(a => counts[a] === mx).sort()[0];
    const sniper = roster.filter(p => p.archetype === 'Sniper/Isolation').length / roster.length;
    let s = 0; if (closestArchetype(m.recommended_playstyle) === dominant) s += 2; if (m.sniper_friendly && sniper > 0.25) s += 1.5; if (!m.sniper_friendly && sniper <= 0.25) s += 0.5;
    s += Math.max(0, (21 - m.rank) / 40); return s;
  }
  function mapArms(maps, roster, k = null) {
    const pool = k ? maps.slice(0, k) : maps;
    const fits = pool.map(m => mapFitScore(m, roster)); const lo = Math.min(...fits), hi = Math.max(...fits);
    return pool.map((m, i) => ({ map_name: m.map_name, fit_score: +fits[i].toFixed(4), q_star: +(0.25 + 0.60 * (hi > lo ? (fits[i] - lo) / (hi - lo) : 0.5)).toFixed(4), recommended_playstyle: m.recommended_playstyle, sniper_friendly: m.sniper_friendly }));
  }
  const DEFAULT_ROSTER = [].concat(Array(4).fill('Aggressive Entry'), Array(3).fill('Sniper/Isolation'), Array(2).fill('Flex Rotator'), ['Passive Support']);

  // ----------------------------------------------- notes numericals (Unit 2)
  function notesExploration(sel, win, us, vs) { const cdf = []; let a = 0; sel.forEach(p => { a += p; cdf.push(a); }); let tot = 0; const rows = us.map((u, i) => { const arm = cdf.findIndex(c => u <= c + 1e-12); const r = vs[i] <= win[arm] ? 1 : 0; tot += r; return { iteration: i + 1, u_select: u, machine: `M${arm + 1}`, p_win: win[arm], u_reward: vs[i], reward: r }; }); const best = Math.max(...win) * us.length; return { cdf, rows, total_reward: tot, max_expected: best, regret: +(best - tot).toFixed(6) }; }
  function notesExploitation(sel, win, u0, vs) { const cdf = []; let a = 0; sel.forEach(p => { a += p; cdf.push(a); }); const arm = cdf.findIndex(c => u0 <= c + 1e-12); let tot = 0; const rows = vs.map((v, i) => { const r = v <= win[arm] ? 1 : 0; tot += r; return { iteration: i + 1, machine: `M${arm + 1}`, u_reward: v, reward: r }; }); const best = Math.max(...win) * vs.length; return { machine: `M${arm + 1}`, rows, total_reward: tot, max_expected: best, regret: +(best - tot).toFixed(6) }; }
  function notesFixedExploration(win, cnts, wins, vs, total) { const Q = win.map((_, i) => (cnts[i] ? wins[i] / cnts[i] : 0)); const best = bArgmax(Q); let tot = 0; const rows = vs.map((v, i) => { const r = v <= win[best] ? 1 : 0; tot += r; return { iteration: sum(cnts) + i + 1, machine: `M${best + 1}`, u_reward: v, reward: r }; }); const all = tot + sum(wins); const mx = Math.max(...win) * total; return { Q_after_exploration: Q, exploited_machine: `M${best + 1}`, rows, exploitation_reward: tot, total_reward: all, max_expected: mx, regret: +(mx - all).toFixed(6) }; }
  function notesEpsAssignment(win, eps, xs, choices, vs) { const k = win.length, Q = new Array(k).fill(0), N = new Array(k).fill(0); let tot = 0, ei = 0; const rows = xs.map((x, t) => { let a, mode; if (x <= eps) { a = choices[ei++]; mode = 'explore'; } else { a = bArgmax(Q); mode = 'exploit'; } const r = vs[t] <= win[a] ? 1 : 0; N[a]++; Q[a] += (r - Q[a]) / N[a]; tot += r; return { step: t + 1, x, mode, machine: `M${a + 1}`, u_reward: vs[t], reward: r, Q: Q.map(q => +q.toFixed(4)), N: N.slice() }; }); return { rows, total_reward: tot, final_Q: Q }; }
  function notesUcb(Q, N, t, c = Math.SQRT2) { const sc = ucbScores(Q, N, t, c); return { t, c, Q, N, bonus: N.map(n => c * Math.sqrt(Math.log(t) / n)), scores: sc, chosen_arm: bArgmax(sc) + 1 }; }
  function notesEpsDetect(acts, rews, k) { const Q = new Array(k).fill(0), N = new Array(k).fill(0); const rows = acts.map((a, i) => { const m = Math.max(...Q); const greedy = Q.map((q, j) => (Math.abs(q - m) < 1e-12 ? j + 1 : null)).filter(x => x !== null); const row = { t: i + 1, action: a, reward: rews[i], Q_before: Q.map(q => +q.toFixed(4)), greedy_set: greedy, verdict: greedy.includes(a) ? 'possibly (greedy action, but a random pick could also select it)' : 'DEFINITELY exploration', definitely_random: !greedy.includes(a) }; N[a - 1]++; Q[a - 1] += (rews[i] - Q[a - 1]) / N[a - 1]; return row; }); return { rows, definitely: rows.filter(r => r.definitely_random).map(r => r.t), possibly: rows.map(r => r.t) }; }
  function notesConstStep(alpha, trials) { const Q = {}; const rows = trials.map(([arm, r]) => { const old = Q[arm] || 0; Q[arm] = old + alpha * (r - old); return { arm, reward: r, old_Q: old, new_Q: +Q[arm].toFixed(6) }; }); return { rows, Q }; }
  const close = (a, b, tol = 1e-6) => Math.abs(a - b) <= tol;
  function banditNumericals() {
    const out = [];
    let ex = notesExploration([0.3, 0.4, 0.2, 0.1], [0.6, 0.4, 0.7, 0.9], [0.68, 0.55, 0.23, 0.95, 0.8], [0.6, 0.2, 0.3, 0.4, 0.9]);
    out.push({ id: 'u2-exploration', title: 'Exploration (random selection via CDF)', unit: 2, result: ex, notes_answer: { total_reward: 3, regret: 1.5 }, match: ex.total_reward === 3 && close(ex.regret, 1.5), code: 'bandits.py' });
    ex = notesExploitation([0.3, 0.4, 0.2, 0.1], [0.6, 0.4, 0.7, 0.9], 0.65, [0.6, 0.2, 0.3, 0.5, 0.9]);
    out.push({ id: 'u2-exploitation', title: 'Exploitation (one machine for all steps)', unit: 2, result: ex, notes_answer: { machine: 'M2', total_reward: 2, regret: 2.5 }, match: ex.machine === 'M2' && ex.total_reward === 2 && close(ex.regret, 2.5), code: 'bandits.py' });
    ex = notesFixedExploration([0.6, 0.4, 0.7, 0.9], [1, 0, 3, 0], [1, 0, 2, 0], [0.33, 0.25, 0.55, 0.45, 0.6, 0.8], 10);
    out.push({ id: 'u2-fixed-exploration', title: 'Fixed exploration + exploitation', unit: 2, result: ex, notes_answer: { exploited_machine: 'M1', total_reward: 8, regret: 1 }, match: ex.exploited_machine === 'M1' && ex.total_reward === 8 && close(ex.regret, 1), code: 'bandits.py' });
    ex = notesUcb([0.55, 0.63, 0.61, 0.40], [3, 4, 3, 2], 12);
    out.push({ id: 'u2-ucb-12', title: 'UCB1 after 12 iterations (4 arms)', unit: 2, result: ex, notes_answer: { chosen_arm: 4, scores: [1.837, 1.745, 1.897, 1.976] }, match: ex.chosen_arm === 4 && [1.837, 1.745, 1.897, 1.976].every((v, i) => close(ex.scores[i], v, 1e-3)), code: 'bandits.py' });
    ex = notesUcb([1.73, 1.83, 1.89, 1.55], [25, 20, 30, 25], 100);
    out.push({ id: 'u2-ucb-100', title: 'UCB after 100 iterations (4 arms)', unit: 2, result: ex, notes_answer: { chosen_arm: 2 }, match: ex.chosen_arm === 2, code: 'bandits.py' });
    ex = notesEpsDetect([1, 2, 2, 2, 3], [-1, 1, -2, 2, 0], 4);
    out.push({ id: 'u2-eps-detect', title: 'ε-greedy: when did exploration definitely occur? (Ex. 2.2)', unit: 2, result: ex, notes_answer: { definitely: [4, 5], possibly: [1, 2, 3, 4, 5] }, match: JSON.stringify(ex.definitely) === '[4,5]', code: 'bandits.py' });
    ex = notesConstStep(0.2, [['a1', 2], ['a2', 0.8], ['a1', 0], ['a2', 0.8]]);
    out.push({ id: 'u2-const-step', title: 'Incremental update with constant step η = 0.2', unit: 2, result: ex, notes_answer: { after_trial_1: 0.4, after_trial_2: 0.16 }, match: close(ex.rows[0].new_Q, 0.4) && close(ex.rows[1].new_Q, 0.16), code: 'bandits.py' });
    ex = notesEpsAssignment([0.4, 0.7], 0.2, [0.1, 0.3, 0.4, 0.2, 0.5, 0.1, 0.6, 0.7, 0.4, 0.9], [0, 1, 1], [0.3, 0.45, 0.75, 0.9, 0.2, 0.55, 0.6, 0.8, 0.2, 0.15]);
    out.push({ id: 'u2-eps-assignment', title: 'ε-greedy Assignment 1 (2 machines, ε = 0.2)', unit: 2, result: ex, notes_answer: null, match: null, note: 'The notes pose this as an assignment without a printed answer; ties in Q are broken toward M1.', code: 'bandits.py' });
    return out;
  }
  function allNumericals() {
    const out = [];
    const G = returnsFromRewards([-1, 2, 6, 3, 2], 0.5).concat([0]);
    out.push({ id: 'u3-ex3.8', unit: 3, title: 'Returns worked backwards (Ex. 3.8, γ = 0.5)', result: { G }, notes_answer: { G: [2, 6, 8, 4, 2, 0] }, match: [2, 6, 8, 4, 2, 0].every((v, i) => close(G[i], v)), code: 'rl_core.returns_from_rewards' });
    const g13 = discountedReturn(new Array(200).fill(13), 0.5);
    out.push({ id: 'u3-const13', unit: 3, title: 'Constant reward 13, γ = 0.5 → G = 13/(1−γ)', result: { G: g13 }, notes_answer: { G: 26 }, match: close(g13, 26), code: 'rl_core.discounted_return' });
    const chain = new FiniteMDP([[[0.3, 0.7]], [[0.2, 0.8]]], [[[0, 0]], [[0, 0]]], [false, false], 1, ['Rain', 'Dry'], ['-']);
    const p = chain.sequenceProbability([1, 1, 0, 0], [[1], [1]], [0.4, 0.6]);
    out.push({ id: 'u3-rain-dry', unit: 3, title: 'Markov chain: P(Dry, Dry, Rain, Rain)', result: { probability: p, formula: 'P(Dry)·P(Dry|Dry)·P(Rain|Dry)·P(Rain|Rain) = 0.6·0.8·0.2·0.3' }, notes_answer: { probability: 0.0288 }, match: close(p, 0.0288), code: 'mdp_model.FiniteMDP.sequence_probability' });
    const dice = diceGame(); const rand = [[0.5, 0.5], [0.5, 0.5]];
    const pe = policyEvaluation(dice, rand, 1, 1e-12, 100000, null, false, 8); const stay = policyEvaluation(dice, [0, 0], 1, 1e-12, 100000); const qr = dice.qFromV(pe.V, 1);
    out.push({ id: 'u4-dice-random', unit: 4, title: 'Dice game: iterative policy evaluation, random policy', result: { V_IN: pe.V[0], sweeps: pe.snapshots.map(s => +s[0].toFixed(4)), q_IN_STAY: qr[0][0], q_IN_QUIT: qr[0][1], closed_form_V_IN: dice.mrpClosedForm(rand, 1)[0] }, notes_answer: { V_IN: 10.5, sweeps: [7, 9.33, 10.11, 10.37, 10.46, 10.49, 10.50], q_IN_STAY: 11, q_IN_QUIT: 10 }, match: close(pe.V[0], 10.5) && close(qr[0][0], 11) && close(qr[0][1], 10), code: 'dynamic_programming.policy_evaluation, mdp_model.mrp_closed_form' });
    const vi = valueIteration(dice, 1, 1e-12, 100000);
    out.push({ id: 'u4-dice-stay', unit: 4, title: 'Dice game: always STAY / optimal policy', result: { V_IN_always_stay: stay.V[0], optimal_action: dice.actionNames[vi.actions[0]], V_star_IN: vi.V[0] }, notes_answer: { V_IN_always_stay: 12 }, match: close(stay.V[0], 12), code: 'dynamic_programming.value_iteration' });
    const gw = gridworld(); const V = policyEvaluation(gw, Array.from({ length: 16 }, () => [0.25, 0.25, 0.25, 0.25]), 1, 1e-10, 100000).V; const expd = [0, -14, -20, -22, -14, -18, -20, -20, -20, -20, -18, -14, -22, -20, -14, 0];
    out.push({ id: 'u4-gridworld', unit: 4, title: '4×4 gridworld, equiprobable random policy', result: { V: V.map(v => +v.toFixed(2)) }, notes_answer: { V: expd }, match: expd.every((v, i) => close(V[i], v, 1e-3)), code: 'dynamic_programming.policy_evaluation' });
    const rows = []; let ok = true;
    for (const [g, best] of [[0, 'left'], [0.9, 'right'], [0.5, 'both']]) { const m = ex322(g); const vl = policyEvaluation(m, [0, 0, 0], g, 1e-12, 100000).V[0]; const vr = policyEvaluation(m, [1, 1, 1], g, 1e-12, 100000).V[0]; const got = Math.abs(vl - vr) < 1e-6 ? 'both' : (vl > vr ? 'left' : 'right'); ok = ok && got === best; rows.push({ gamma: g, V_left: vl, V_right: vr, optimal: got }); }
    out.push({ id: 'u3-ex3.22', unit: 3, title: 'Ex. 3.22: which policy is optimal for γ = 0, 0.9, 0.5?', result: { rows }, notes_answer: { 'γ=0': 'left', 'γ=0.9': 'right (5.26 vs 9.47)', 'γ=0.5': 'both (1.33)' }, match: ok, code: 'dynamic_programming.policy_evaluation' });
    const AB = [[[0, 0, 1, 1], [2, 6, 1, 0]], [[0, 1, 0, 1, 1], [3, 2, 4, 2, 0]], [[1, 1, 0, 1], [3, 6, 1, 0]], [[0, 1, 0, 1, 1, 1], [0, 2, 4, 4, 2, 0]], [[1, 1], [8, 0]]];
    const fv = firstVisitPrediction(AB, 1, 2), evv = everyVisitPrediction(AB, 1, 2);
    out.push({ id: 'u5-mc-ab', unit: 5, title: 'First-visit vs every-visit MC on 5 episodes (A/B, γ = 1)', result: { first_visit: { A: fv.V[0], B: fv.V[1] }, every_visit: { A: evv.V[0], B: evv.V[1] }, visits_every: { A: evv.N[0], B: evv.N[1] } }, notes_answer: { first_visit: { A: 8.25, B: 7.8 }, every_visit: { A: 8.0, B: 4.0 } }, match: close(fv.V[0], 8.25) && close(fv.V[1], 7.8) && close(evv.V[0], 8) && close(evv.V[1], 4), code: 'monte_carlo.first_visit_prediction / every_visit_prediction' });
    return out.concat(banditNumericals());
  }
  function diceGame() { const P = [[[2 / 3, 1 / 3], [0, 1]], [[0, 1], [0, 1]]], R = [[[4, 4], [0, 10]], [[0, 0], [0, 0]]]; return new FiniteMDP(P, R, [false, true], 1, ['IN', 'END'], ['STAY', 'QUIT']); }
  function gridworld() { const P = [], R = []; const mv = [[-1, 0], [1, 0], [0, -1], [0, 1]]; for (let s = 0; s < 16; s++) { const r = Math.floor(s / 4), c = s % 4; P.push(mv.map(([dr, dc]) => { const row = new Array(16).fill(0); const nr = r + dr, nc = c + dc; row[(nr >= 0 && nr < 4 && nc >= 0 && nc < 4) ? nr * 4 + nc : s] = 1; return row; })); R.push(mv.map(() => new Array(16).fill(-1))); } return new FiniteMDP(P, R, Array.from({ length: 16 }, (_, i) => i === 0 || i === 15), 1, null, ['U', 'D', 'L', 'R']); }
  function ex322(g) { const z = () => [0, 0, 0]; const P = [[[0, 1, 0], [0, 0, 1]], [[1, 0, 0], [1, 0, 0]], [[1, 0, 0], [1, 0, 0]]]; const R = [[[0, 1, 0], z()], [z(), z()], [[2, 0, 0], [2, 0, 0]]]; return new FiniteMDP(P, R, [false, false, false], g, ['top', 'left', 'right'], ['left', 'right']); }

  // ===================================================== endpoint emulation
  function now() { return (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now(); }
  const _dpCache = new Map();
  function dpSolution(g, samples = 1000, seed = 7) { const k = `${g}|${samples}|${seed}`; if (!_dpCache.has(k)) { const m = getModel(samples, seed, g); _dpCache.set(k, [m, valueIteration(m, g, 1e-10, 5000)]); } return _dpCache.get(k); }
  function resolvePolicy(name, g = 0.9, samples = 1000, seed = 7, eps = 0) {
    let acts;
    if (name === 'dp_optimal') acts = dpSolution(g, samples, seed)[1].actions.slice();
    else if (name === 'q_learning') acts = greedyFromQ(loadProduction()[0]);
    else if (name === 'heuristic') acts = heuristicActions().concat([1]);
    else if (['keep', 'lower', 'raise'].includes(name)) acts = new Array(NS1).fill({ lower: 0, keep: 1, raise: 2 }[name]);
    else if (name === 'random') return uniformPolicy();
    else throw httpError(400, `unknown policy '${name}'`);
    return eps > 0 ? epsilonSoft(acts, eps) : deterministicPolicy(acts);
  }
  function httpError(status, detail) { const e = new Error(detail); e.status = status; e.detail = detail; return e; }
  const d0dot = (sd, V) => startDistribution(sd).concat([0]).reduce((x, p, i) => x + p * V[i], 0);
  const opsStats = it => ({ backups_per_sweep: NS * NA, operations_per_sweep: NS * NA * (NS + 1), total_operations: it * NS * NA * (NS + 1) });
  function parseState(st) { const s = String(st).includes('-') ? keyToState(String(st)) : parseInt(st, 10); if (!(s >= 0 && s < NS)) throw httpError(400, 'state index out of range 0..47'); return s; }
  function arms(spec = {}) {
    if (spec.source === 'custom') { const means = spec.means || []; return [means.map((m, i) => ({ map_name: (spec.labels || [])[i] || `Arm ${i + 1}`, q_star: m, fit_score: null })), 'custom (user-specified)']; }
    if (spec.source === 'tournament' && spec.roster_archetypes && spec.map_pool) return [mapArms(spec.map_pool, spec.roster_archetypes.map(a => ({ archetype: a })), spec.k || 10), spec.label || 'tournament match'];
    const maps = (root.CLUTCH_STATIC && root.CLUTCH_STATIC.sample_maps) || [];
    return [mapArms(maps, DEFAULT_ROSTER.map(a => ({ archetype: a })), spec.k || 10), 'sample_maps.csv + demo roster (4 entry, 3 sniper, 2 flex, 1 support)'];
  }
  const D = (v, d) => (v === undefined || v === null ? d : v);

  const routes = {
    'GET /health': () => ({ status: 'ok', datasets_loaded: 0, version: '4.0', engine: 'browser', rl_endpoints: true }),
    'GET /rl/spec': () => { const S = root.CLUTCH_STATIC || {}; const sp = Object.assign({}, S.spec || {}); sp.states = Array.from({ length: NS }, (_, s) => stateInfo(s)); sp.step_model = Array.from({ length: NS }, (_, s) => [0, 1, 2].map(a => stepModel(s, a))); return sp; },
    'POST /rl/trajectory': b => { const g = D(b.gamma, 0.9); const pol = D(b.policy, 'dp_optimal'); const pi = resolvePolicy(pol, g, 1000, 7, D(b.policy_epsilon, 0)); let Q = null; if (pol === 'q_learning') Q = loadProduction()[0]; else if (pol === 'dp_optimal') Q = dpSolution(g)[1].Q; return trajectory(pi, D(b.seed, 42), g, D(b.start_state, null), D(b.start_dist, 'session_start'), Q); },
    'GET /rl/mdp': q => { const m = getModel(D(q.samples, 1000), D(q.seed, 7), D(q.gamma, 0.9)); return { meta: m.meta, quality_vs_simulator: modelQuality(m), check: m.check(), n_states: NS, terminal: TERMINAL, actions: ACTION_NAMES, states: Array.from({ length: NS }, (_, s) => stateInfo(s)), provenance: 'MODEL ESTIMATE', P: m.P, R: m.R, r_sa: m.r_sa }; },
    'GET /rl/mdp/state': (q, st) => {
      const s = parseState(st); const g = D(q.gamma, 0.9), samples = D(q.samples, 1000), seed = D(q.seed, 7), pol = D(q.policy, 'dp_optimal'), pe_ = D(q.policy_epsilon, 0), topK = D(q.top_k, 6);
      const [model, vi] = dpSolution(g, samples, seed); const pi = resolvePolicy(pol, g, samples, seed, pe_);
      const Vpi = policyEvaluation(model, pi, g, 1e-10, 5000).V; const Qpi = model.qFromV(Vpi, g);
      return { state: stateInfo(s), gamma: g, policy: pol, policy_epsilon: pe_, V_pi: Vpi[s], V_star: vi.V[s], best_action: ACTION_NAMES[vi.actions[s]], best_action_index: vi.actions[s],
        per_action: [0, 1, 2].map(a => ({ action: a, name: ACTION_NAMES[a], pi: pi[s][a], expected_reward: model.r_sa[s][a], q_pi: Qpi[s][a], q_star: vi.Q[s][a], next_states: topTransitions(model, s, a, topK), terminal_probability: model.P[s][a][TERMINAL], simulator: stepModel(s, a) })),
        bellman_expectation: backupBreakdown(model, s, Vpi, g, pi, topK), bellman_optimality: backupBreakdown(model, s, vi.V, g, null, topK),
        V_pi_all: Vpi.slice(0, NS), V_star_all: vi.V.slice(0, NS), policy_actions: policyActions(pi).slice(0, NS), optimal_actions: vi.actions.slice(0, NS), model_meta: model.meta, provenance: 'MODEL ESTIMATE' };
    },
    'GET /rl/mdp/markov-chain': q => {
      const g = D(q.gamma, 0.9); const [model] = dpSolution(g, D(q.samples, 1000), D(q.seed, 7)); const pi = resolvePolicy(D(q.policy, 'dp_optimal'), g); const [Ppi, Rpi] = model.policyMRP(pi);
      let d = startDistribution(D(q.start_dist, 'session_start')).concat([0]); const dists = [d]; const H = Math.min(60, Math.max(1, D(q.horizon, 20)));
      for (let i = 0; i < H; i++) { const nd = new Array(NS1).fill(0); for (let s = 0; s < NS1; s++) if (d[s]) for (let j = 0; j < NS1; j++) nd[j] += d[s] * Ppi[s][j]; d = nd; dists.push(d); }
      const alive = dists.map(x => 1 - x[TERMINAL]);
      return { P_pi: Ppi, R_pi: Rpi, distributions: dists, prob_session_alive: alive, expected_matches_within_horizon: sum(alive.slice(0, -1)), V_closed_form: model.mrpClosedForm(pi, g), provenance: 'MODEL ESTIMATE' };
    },
    'POST /rl/dp/policy-evaluation': b => {
      const g = D(b.gamma, 0.9), samples = D(b.samples, 1000), seed = D(b.seed, 7); const model = getModel(samples, seed, g); const pi = resolvePolicy(D(b.policy, 'random'), g, samples, seed, D(b.policy_epsilon, 0));
      const r = policyEvaluation(model, pi, g, D(b.theta, 1e-6), D(b.max_iter, 1000), null, !!b.in_place, D(b.snapshots, 25)); const closed = model.mrpClosedForm(pi, g);
      return { V: r.V.slice(0, NS), iterations: r.iterations, deltas: r.deltas, converged: r.converged, snapshots: r.snapshots.map(s => s.slice(0, NS)), seconds: r.seconds, V_closed_form_mrp: closed.slice(0, NS),
        max_abs_diff_vs_closed_form: Math.max(...closed.map((v, i) => Math.abs(v - r.V[i]))), bellman_residual: bellmanResidual(model, r.V, g, pi), policy_actions: policyActions(pi).slice(0, NS), policy_matrix: pi.slice(0, NS),
        backup_example: backupBreakdown(model, D(b.explain_state, 42), r.V, g, pi), compute: opsStats(r.iterations), model_meta: model.meta, provenance: 'MODEL ESTIMATE' };
    },
    'POST /rl/dp/policy-iteration': b => {
      const g = D(b.gamma, 0.9); const model = getModel(D(b.samples, 1000), D(b.seed, 7), g); const r = policyIteration(model, g, D(b.theta, 1e-6), 100, D(b.max_iter, 1000));
      return { V: r.V.slice(0, NS), Q: r.Q.slice(0, NS), actions: r.actions.slice(0, NS), policy_stable: r.policy_stable, improvement_steps: r.improvement_steps, evaluation_sweeps_total: r.evaluation_sweeps_total,
        history: r.history.map(h => ({ iteration: h.iteration, evaluation_sweeps: h.evaluation_sweeps, n_changed: h.n_changed, changed_states: h.changed_states, policy: h.policy.slice(0, NS), V: h.V.slice(0, NS) })),
        seconds: r.seconds, initial_policy: 'KEEP everywhere', compute: opsStats(r.evaluation_sweeps_total), model_meta: model.meta, provenance: 'MODEL ESTIMATE' };
    },
    'POST /rl/dp/value-iteration': b => {
      const g = D(b.gamma, 0.9); const model = getModel(D(b.samples, 1000), D(b.seed, 7), g); const r = valueIteration(model, g, D(b.theta, 1e-6), D(b.max_iter, 1000), D(b.snapshots, 25), !!b.in_place);
      const qa = greedyFromQ(loadProduction()[0]).slice(0, NS);
      return { V: r.V.slice(0, NS), Q: r.Q.slice(0, NS), actions: r.actions.slice(0, NS), iterations: r.iterations, deltas: r.deltas, converged: r.converged, contraction_ratios: r.contraction_ratios, max_contraction_ratio: r.max_contraction_ratio,
        snapshots: r.snapshots.map(s => s.slice(0, NS)), seconds: r.seconds, bellman_residual: bellmanResidual(model, r.V, g), backup_example: backupBreakdown(model, D(b.explain_state, 42), r.V, g),
        q_learning_actions: qa, agreement_with_q_learning: agreement(r.actions, qa), disagreements_with_q_learning: disagreementAnalysis(qa, r.actions, r.Q), compute: opsStats(r.iterations), model_meta: model.meta, provenance: 'MODEL ESTIMATE' };
    },
    'POST /rl/dp/gamma-sweep': b => {
      const gs = b.gammas || [0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99];
      return { rows: gs.map(g => { const [, vi] = dpSolution(g, D(b.samples, 1000), D(b.seed, 7)); const acts = vi.actions.slice(0, NS); const e = evaluatePolicy(deterministicPolicy(acts), 300, 2026, 0.9, 'session_start'); const counts = [0, 0, 0]; acts.forEach(a => counts[a]++);
        return { gamma: g, actions: acts, counts, iterations: vi.iterations, V_session_start: d0dot('session_start', vi.V), sim_rage_quit_rate: e.rage_quit_rate.mean, sim_session_length: e.session_length.mean, sim_close_match_rate: e.close_match_rate.mean }; }), provenance: 'MODEL ESTIMATE (policies) / SIMULATED (rates)' };
    },
    'POST /rl/mc/predict': b => {
      const g = D(b.gamma, 0.9), samples = D(b.samples, 1000), ms = D(b.model_seed, 7); const model = getModel(samples, ms, g); const pi = resolvePolicy(D(b.policy, 'heuristic'), g, samples, ms, D(b.policy_epsilon, 0));
      const ref = policyEvaluation(model, pi, g, 1e-10, 5000).V; const r = mcPredictionRun(pi, D(b.episodes, 3000), g, D(b.seed, 11), D(b.start_dist, 'uniform'), ref);
      return { V_first_visit: r.V_first_visit.slice(0, NS), V_every_visit: r.V_every_visit.slice(0, NS), N_first_visit: r.N_first_visit.slice(0, NS), N_every_visit: r.N_every_visit.slice(0, NS), V_dp_reference: ref.slice(0, NS), curve: r.curve, sample_episode: r.sample_episode,
        mean_return: r.mean_return, std_return: r.std_return, mean_length: r.mean_length, n_episodes: r.n_episodes, seconds: r.seconds, policy_actions: policyActions(pi).slice(0, NS), reference: 'DP policy evaluation on the simulation-estimated model', provenance: 'SIMULATED' };
    },
    'POST /rl/mc/control': b => {
      const g = D(b.gamma, 0.9); const [model, vi] = dpSolution(g, D(b.samples, 1000), D(b.model_seed, 7));
      const r = mcControl(D(b.variant, 'on_policy'), D(b.episodes, 10000), g, D(b.epsilon, 0.1), D(b.seed, 7), 'uniform', vi.actions, model, 20, D(b.behaviour_epsilon, 0.3));
      let qm = 0; for (let s = 0; s < NS; s++) for (let a = 0; a < NA; a++) qm += Math.abs(r.Q[s][a] - vi.Q[s][a]);
      return { variant: r.variant, Q: r.Q.slice(0, NS), N: r.N.slice(0, NS), actions: r.actions, history: r.history, seconds: r.seconds, unvisited_pairs: r.unvisited_pairs, dp_actions: vi.actions.slice(0, NS), agreement_with_dp: agreement(r.actions, vi.actions),
        disagreements: disagreementAnalysis(r.actions, vi.actions, vi.Q), J_learned: modelValue(model, r.actions, g), J_dp_optimal: modelValue(model, vi.actions, g), q_mae_vs_dp: qm / (NS * NA), transitions_used: r.transitions_used || null, provenance: 'SIMULATED' };
    },
    'POST /rl/mc/off-policy': b => {
      const g = D(b.gamma, 0.9), samples = D(b.samples, 1000), ms = D(b.model_seed, 7), sd = D(b.start_dist, 'session_start'); const model = getModel(samples, ms, g);
      const target = resolvePolicy(D(b.target, 'dp_optimal'), g, samples, ms, D(b.target_epsilon, 0.1)), beh = resolvePolicy(D(b.behaviour, 'heuristic'), g, samples, ms, D(b.behaviour_epsilon, 0.3));
      const Vt = policyEvaluation(model, target, g, 1e-10, 5000).V, Vb = policyEvaluation(model, beh, g, 1e-10, 5000).V; const n = D(b.episodes, 5000);
      const r = offPolicyPrediction(target, beh, n, g, D(b.seed, 4), sd, d0dot(sd, Vt), Vt); if (r.error) throw httpError(400, r.error);
      const onp = evaluatePolicy(target, Math.min(3000, Math.max(500, n)), D(b.seed, 4) + 1000, g, sd);
      Object.assign(r, { reference_J_target_dp: d0dot(sd, Vt), reference_J_behaviour_dp: d0dot(sd, Vb), on_policy_mc_J_target: onp.discounted_return, target_actions: policyActions(target).slice(0, NS), behaviour_actions: policyActions(beh).slice(0, NS), policy_agreement: agreement(policyActions(target), policyActions(beh)), V_dp_target: Vt.slice(0, NS) });
      r.V_ordinary_is = r.V_ordinary_is.slice(0, NS); r.V_weighted_is = r.V_weighted_is.slice(0, NS); r.visits = r.visits.slice(0, NS); return r;
    },
    'GET /rl/q/table': q => { const g = D(q.gamma, 0.9); const [Q, meta] = loadProduction(); const acts = greedyFromQ(Q.slice(0, NS)); const [, vi] = dpSolution(g); return { Q: Q.slice(0, NS), actions: acts, meta, keys: Array.from({ length: NS }, (_, s) => stateKey(s)), dp_actions: vi.actions.slice(0, NS), agreement_with_dp: agreement(acts, vi.actions), disagreements: disagreementAnalysis(acts, vi.actions, vi.Q), algorithm: 'Tabular Q-learning (NOT a DQN: no neural network, replay buffer or target network)', provenance: 'SIMULATED' }; },
    'POST /rl/q/train': b => {
      const g = D(b.gamma, 0.9); const [model, vi] = dpSolution(g);
      const r = qTrain({ episodes: D(b.episodes, 30000), alpha: D(b.alpha, 0.05), gamma: g, eps_start: D(b.eps_start, 1), eps_decay: D(b.eps_decay, 0.9998), eps_min: D(b.eps_min, 0.05), seed: D(b.seed, 7), start_dist: 'uniform', reference_actions: vi.actions, model, n_checkpoints: 40, td_log_last: 30, watch_state: D(b.watch_state, null) });
      return { Q: r.Q.slice(0, NS), actions: r.actions, curve: r.curve, td_log: r.td_log, first_update: r.first_update, watch_log: r.watch_log.slice(0, 60), visits: r.visits.slice(0, NS), explore_steps: r.explore_steps, exploit_steps: r.exploit_steps, final_epsilon: r.final_epsilon, params: r.params, seconds: r.seconds,
        dp_actions: vi.actions.slice(0, NS), agreement_with_dp: agreement(r.actions, vi.actions), disagreements: disagreementAnalysis(r.actions, vi.actions, vi.Q), J_learned: modelValue(model, r.actions, g), J_dp_optimal: modelValue(model, vi.actions, g), saved_as_default: false, note: 'Production policy unchanged (browser engine never saves)', provenance: 'SIMULATED' };
    },
    'POST /rl/q/td-step': b => { const Q = loadProduction()[0].map(r => r.slice()); return tdUpdate(Q, b.state, b.action, b.reward, b.next_state, !!b.done || b.next_state === TERMINAL, D(b.alpha, 0.05), D(b.gamma, 0.9)); },
    'POST /rl/compare': b => comparePolicies(b),
    'POST /bandit/arms': b => { const [a, src] = arms(b); return { arms: a, roster_source: src, provenance: 'DERIVED FEATURE' }; },
    'POST /bandit/run': b => {
      const [a, src] = arms(b.arms || {}); const means = a.map(x => x.q_star); const algs = b.algorithms || ['greedy', 'epsilon_greedy', 'optimistic', 'ucb'];
      const r = runArena(algs, means, D(b.rounds, 1000), D(b.runs, 50), D(b.seed, 1), { epsilon: D(b.epsilon, 0.1), c: D(b.c, 1.414), q0: D(b.q0, 1.0), alpha: D(b.alpha, null), explore_rounds: D(b.explore_rounds, 50), reward_model: D(b.reward_model, 'bernoulli'), sigma: D(b.sigma, 0.1), nonstationary: D(b.nonstationary, 'none'), drift_sigma: D(b.drift_sigma, 0.01) });
      let best = null; for (const [k, v] of Object.entries(r.results)) if (!best || v.final_reward > best[1].final_reward) best = [k, v];
      Object.assign(r, { arms: a, roster_source: src, best_algorithm: best[0], bandit_recommended_map: a[argmaxFirst(best[1].example_run.Q)].map_name, rule_based_map: a[argmaxFirst(means)].map_name }); return r;
    },
    'GET /bandit/numericals': () => banditNumericals(),
    'GET /academic/numericals': () => { const items = allNumericals(); return { items, checked: items.filter(i => i.match !== null).length, matched: items.filter(i => i.match).length }; },
    'GET /academic/coverage': () => (root.CLUTCH_STATIC || {}).coverage || { rows: [] },
    'GET /academic/viva-snapshot': q => {
      const s = Math.max(0, Math.min(47, D(q.state, 42))); const g = D(q.gamma, 0.9); const [model, vi] = dpSolution(g); const [Q] = loadProduction(); const a = argmaxRow(Q[s]);
      const rng = new Mulberry32(deriveSeed(99, s)); const [s2, r, done] = envStep(s, a, rng); const td = tdUpdate(Q.map(x => x.slice()), s, a, r, s2, done, 0.05, g);
      const ep = generateEpisode(deterministicPolicy(vi.actions), 5, { startState: s, recordInfo: true }); const b = epsilonSoft(heuristicActions().concat([1]), 0.3), tgt = deterministicPolicy(vi.actions);
      const steps = ep.transitions.slice(0, 5).map(t => ({ state: stateShort(t.state), action: ACTION_NAMES[t.action], pi: tgt[t.state][t.action], b: b[t.state][t.action] })); let rho = 1; steps.forEach(x => { rho *= x.pi / x.b; });
      return { state: stateInfo(s), gamma: g, value_iteration_backup: backupBreakdown(model, s, vi.V, g, null, 4), V_star: vi.V[s], best_action: ACTION_NAMES[vi.actions[s]], td_update: td, mc_episode: episodeTable(ep, g), is_example: { steps, rho, target: 'DP-optimal (deterministic)', behaviour: 'heuristic, ε = 0.3' }, bandit_arms: arms({})[0].slice(0, 5), ucb_example: notesUcb([0.55, 0.63, 0.61, 0.40], [3, 4, 3, 2], 12), provenance: 'MODEL ESTIMATE + SIMULATED' };
    },
    'POST /optimize': b => { const [Q] = loadProduction(); const s = stateFromFeatures(b.fatigue, b.tilt, b.streak); const qv = Q[s]; const best = argmaxFirst(qv); const ms = [0, 1, 2].map(a => stepModel(s, a));
      return { state_bucket: stateKey(s), state_index: s, state: stateInfo(s), action: best, action_label: ACTION_LABELS[best], action_name: ACTION_NAMES[best], q_values: { 'Lower Difficulty': qv[0], 'Keep Steady': qv[1], 'Raise Difficulty': qv[2] }, q_vector: qv.slice(), predicted_rage_quit_risk: ms[best].rage_prob,
        per_action_model: ms.map((m, a) => ({ action: a, p_win: m.p_win, closeness: m.closeness, rage_quit_prob: m.rage_prob, boredom_quit_prob: m.boredom_prob, expected_immediate_reward: m.expected_reward_immediate })), policy: 'Tabular Q-learning (greedy w.r.t. production Q-table)', risk_provenance: 'SIMULATED (simulator model, not fitted to real players)' }; },
  };

  /** Emulated endpoint call. path like '/rl/mdp/state/2-2-0?gamma=0.9' */
  function call(method, path, body = {}) {
    const [p, qs] = path.split('?'); const q = {};
    if (qs) for (const kv of qs.split('&')) { const [k, v] = kv.split('='); const dv = decodeURIComponent(v || ''); q[k] = dv === '' ? dv : (isNaN(Number(dv)) ? dv : Number(dv)); }
    let key = `${method.toUpperCase()} ${p}`; let arg = null;
    const m = p.match(/^\/rl\/mdp\/state\/([^/]+)$/); if (m) { key = 'GET /rl/mdp/state'; arg = m[1]; }
    const fn = routes[key]; if (!fn) throw httpError(404, `offline engine does not implement ${key}`);
    return method.toUpperCase() === 'GET' ? fn(q, arg) : fn(body || {}, arg);
  }

  const API = {
    version: '4.0', Mulberry32, deriveSeed, NS, NA, TERMINAL, ACTION_NAMES, ACTION_LABELS, FATIGUE_LEVELS, TILT_LEVELS, STREAK_LEVELS, ENV_PARAMS, W1, W2, W3,
    encode, decode, stateKey, keyToState, stateLabel, stateShort, stateInfo, stateFromFeatures, bucketFatigue, bucketTilt, bucketStreak,
    rewardFunction, pWin, closeness, stepModel, envStep, exactTransition, startDistribution, sampleStartState,
    uniformPolicy, constantPolicy, deterministicPolicy, heuristicPolicy, heuristicActions, epsilonSoft, greedyFromQ, policyActions,
    returnsFromRewards, discountedReturn, generateEpisode, FiniteMDP, estimateModel, exactModel, getModel, modelQuality,
    policyEvaluation, policyImprovement, policyIteration, valueIteration, bellmanResidual, backupBreakdown,
    firstVisitPrediction, everyVisitPrediction, mcPredictionRun, mcControl, offPolicyPrediction, coverageCheck,
    qTrain, tdUpdate, loadProduction, evaluatePolicy, comparePolicies, trajectory, agreement,
    runBandit, runArena, mapArms, ucbScores, allNumericals, banditNumericals, resolvePolicy, dpSolution, call, routes,
  };
  root.ClutchRL = API;
  if (typeof module !== 'undefined' && module.exports) module.exports = API;
})(typeof window !== 'undefined' ? window : globalThis);
