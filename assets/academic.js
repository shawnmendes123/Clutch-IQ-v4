/* Clutch IQ — Academic / viva screens. */
(function (root) {
  'use strict';
  const App = root.App, RL = root.ClutchRL, C = root.Charts, esc = C.esc, $ = App.$, $$ = App.$$;
  const f = App.f, S = root.CLUTCH_STATIC || {};
  const EXP = (S.experiments || {}).experiments || [];
  const E = id => EXP.find(e => e.id === id);
  const SCREEN_IDS = { 'Live Session': 'ctl-session', 'Q-Learning': 'ctl-q', 'MDP Explorer': 'ctl-mdp', 'DP Lab': 'ctl-dp', 'Bandit Arena': 'lab-bandit', 'Monte Carlo Lab': 'lab-mc', 'Algorithm Comparison': 'lab-compare', 'Notes Numericals': 'acad-numericals', 'Formulation': 'acad-formulation', 'Limitations': 'acad-limitations', 'Control Room': 'ctl-session' };
  const screenFor = txt => { for (const [k, v] of Object.entries(SCREEN_IDS)) if (String(txt).includes(k)) return v; return null; };

  // =================================================================== VIVA QUESTIONS
  App.screens['acad-viva'] = {
    render(el) {
      const e6 = E('E6'), e5 = E('E5'), e7 = E('E7'), e3 = E('E3'), e1 = E('E1'), e8 = E('E8');
      const row = k => e6 && e6.rows.find(r => r.policy.startsWith(k));
      const dp = row('Dynamic Programming — Value'), keep = row('Always KEEP'), ql = row('Tabular Q'), heur = row('Manual');
      const QA = [
        ['What is the agent?', 'The Clutch IQ session controller: software that, before each match, chooses the lobby difficulty for one player.', 'ctl-session'],
        ['What is the environment?', 'One player\'s competitive play session. Here it is a seeded simulator (rl_core.SessionEnv) whose behavioural assumptions are documented on the Formulation screen — it is not fitted to real players.', 'acad-formulation'],
        ['What are the states?', 'S = (fatigue bucket 0–3) × (tilt bucket 0–3) × (streak bucket losing/neutral/winning) = 48 states, plus an absorbing END state. The same buckets are computed from uploaded match histories.', 'ctl-mdp'],
        ['What are the actions?', 'A = {LOWER, KEEP, RAISE} lobby difficulty — one difficulty band (≈15 pp of win probability) easier, unchanged, or harder.', 'ctl-mdp'],
        ['What is the reward?', 'R = 1.0·(match completed and session continues) + 1.2·C − 3.0·(rage quit), where C = 1 − 2|p<sub>win</sub> − ½| is predicted closeness. Per-step rewards are bounded: rage quit ∈ [−3, −1.8], completed match ∈ [1, 2.2].', 'acad-formulation'],
        ['What is a policy?', 'A mapping π(a|s) from each of the 48 states to action probabilities. Deterministic policies (DP, greedy Q) put probability 1 on one action; ε-soft and random policies are stochastic.', 'ctl-mdp'],
        ['What is discounted return?', 'G<sub>t</sub> = R<sub>t+1</sub> + γR<sub>t+2</sub> + … = R<sub>t+1</sub> + γG<sub>t+1</sub>. The Live session table computes it backwards for a real episode and the γ curve shows the same rewards under every γ.', 'ctl-session'],
        ['What is exploration vs exploitation?', 'Exploiting picks the action that currently looks best; exploring tries others to improve the estimates. Pure greedy map selection gets stuck: in E7 its regret is ' + (e7 ? f(e7.stationary.find(r => r.key === 'greedy').final_regret, 0) : '—') + ' vs ' + (e7 ? f(e7.stationary.find(r => r.key === 'epsilon_greedy').final_regret, 0) : '—') + ' for ε-greedy.', 'lab-bandit'],
        ['How does ε-greedy behave?', 'With probability ε it picks a uniformly random arm, otherwise the arg-max of Q. ε = 0 is greedy, ε = 1 is explore-only; the ε-sweep in E7 shows regret is U-shaped in ε (best around 0.05–0.1 here).', 'lab-bandit'],
        ['How does UCB behave?', 'It picks argmax Q<sub>t</sub>(a) + c√(ln t / N<sub>t</sub>(a)): untried arms first, then arms that are either good or uncertain. The bonus shrinks as an arm is pulled and grows slowly with t. The Bandit arena shows the bonus and score of every arm at the last step.', 'lab-bandit'],
        ['What is regret?', 'Expected (pseudo-)regret Σ<sub>t</sub>[max<sub>a</sub>q*(a) − q*(A<sub>t</sub>)]: the reward lost by not always pulling the best arm. The notes\' realised regret T·max q* − ΣR is also reported.', 'lab-bandit'],
        ['Why is this a Markov Decision Process?', 'Sequential decisions, stochastic transitions, additive rewards, and a state that is sufficient for the future: the simulator\'s next-state law depends only on (s, a) — the Markov property holds in the 48 buckets by construction. A real player\'s psychology would only be approximately Markov in these features (stated in Limitations).', 'ctl-mdp'],
        ['What are transition probabilities?', 'p(s′|s,a): the probability of the next state given state and action. They are estimated from simulator rollouts (1000 per (s,a) by default) and labelled “model estimate”; E1 shows the estimation error ' + (e1 ? f(e1.rows[2].max_abs_error, 3) : '—') + ' (max) at 1000 samples.', 'ctl-mdp'],
        ['What is V(s)?', 'V<sub>π</sub>(s) = E<sub>π</sub>[G<sub>t</sub> | S<sub>t</sub> = s], the expected discounted return from s following π. The MDP explorer shows V<sub>π</sub>(s) and V*(s) for any state.', 'ctl-mdp'],
        ['What is Q(s,a)?', 'Q<sub>π</sub>(s,a) = E<sub>π</sub>[G<sub>t</sub> | S<sub>t</sub> = s, A<sub>t</sub> = a]: the value of taking a first, then following π. Greedy w.r.t. Q* is optimal.', 'ctl-mdp'],
        ['What does the Bellman equation do?', 'It writes a value recursively: V<sub>π</sub>(s) = Σ<sub>a</sub>π(a|s)Σ<sub>s′</sub>p(s′|s,a)[r + γV<sub>π</sub>(s′)]. The explorer expands this sum term by term with the actual p, r and V; the optimality version replaces Σπ by max.', 'ctl-mdp'],
        ['How does policy evaluation work?', 'Start with V = 0 and repeatedly apply the Bellman expectation backup to every state until max change < θ. The result matches the closed-form MRP solution (I − γP<sub>π</sub>)⁻¹R<sub>π</sub> to ~1e-6.', 'ctl-dp'],
        ['How does policy iteration work?', 'Evaluate the current policy, make it greedy w.r.t. its values, repeat until no state changes. Here it stabilises in ' + (E('E2') ? E('E2').rows[1].pi_improvement_steps : '—') + ' improvement steps from “KEEP everywhere”.', 'ctl-dp'],
        ['How does value iteration work?', 'Apply the Bellman optimality backup V<sub>k+1</sub>(s) = max<sub>a</sub>Σp[r + γV<sub>k</sub>(s′)] until max Δ < θ, then act greedily. Each sweep shrinks the error by at least γ (measured contraction ratio ≤ γ).', 'ctl-dp'],
        ['What is Monte Carlo prediction?', 'Estimate V<sub>π</sub>(s) by averaging the returns observed after visits to s in complete episodes — no model needed. E4: the error vs DP falls from ' + (E('E4') ? f(E('E4').rows[3].mae_first_visit_mean, 2) : '—') + ' (100 episodes) to ' + (E('E4') ? f(E('E4').rows[E('E4').rows.length - 1].mae_first_visit_mean, 2) : '—') + ' (20,000).', 'lab-mc'],
        ['What is first-visit vs every-visit MC?', 'First-visit averages only the return after the first visit to s in each episode; every-visit averages all of them. Both converge; the notes\' A/B example gives V(A) = 8.25 vs 8.0 and V(B) = 7.8 vs 4.0 — reproduced exactly.', 'acad-numericals'],
        ['What is MC control?', 'Generalised policy iteration with sampled returns: estimate Q(s,a) from episodes, make the policy (ε-)greedy w.r.t. Q, repeat. Implemented with exploring starts, on-policy ε-soft and off-policy weighted IS.', 'lab-mc'],
        ['What is off-policy evaluation?', 'Estimating the value of a target policy π from data generated by a different behaviour policy b — e.g. judging a new matchmaker from sessions logged under the current one, before deployment.', 'lab-mc'],
        ['What is importance sampling?', 'Re-weighting each return by ρ = Π π(A<sub>k</sub>|S<sub>k</sub>)/b(A<sub>k</sub>|S<sub>k</sub>) so it counts as if generated by π. Ordinary IS is unbiased but high-variance; weighted IS is biased but stable. E8: for a small policy change WIS lands within ' + (e8 ? f(Math.abs(e8.rows[0].weighted_is - e8.rows[0].truth_dp), 2) : '—') + ' of the truth; for a large change the effective sample size collapses to ' + (e8 ? f(e8.rows[1].ess, 0) : '—') + ' and the estimate is flagged unreliable.', 'lab-mc'],
        ['What does Q-learning update?', 'One table entry per step: Q(S,A) ← Q(S,A) + α[R + γ·max<sub>a′</sub>Q(S′,a′) − Q(S,A)] (no bootstrap after a terminal state). The Q-learning screen shows the TD target, TD error and new value with real numbers.', 'ctl-q'],
        ['What is model-based vs model-free RL?', 'Model-based (DP) uses p(s′|s,a) and r to plan; model-free (MC, Q-learning) learns only from experienced transitions. DP plans in milliseconds but depends on the model; the learners need on the order of 10⁵ simulated steps.', 'lab-compare'],
        ['Why would the different algorithms produce different policies?', 'They use different information (model vs samples), have different targets (on-policy MC learns the best ε-soft policy) and different noise. Disagreements concentrate where two actions have nearly equal value or where a state is rarely visited: Q-learning agrees with DP on ' + (ql ? App.pct(ql.agreement_with_dp, 0) : '—') + ' of states yet reaches model value ' + (ql ? f(ql.model_value_J, 2) : '—') + ' vs ' + (dp ? f(dp.model_value_J, 2) : '—') + '.', 'lab-compare'],
        ['What parts of the system use real data?', 'Uploaded match histories (kills, deaths, results, dates) → derived fatigue/tilt/streak features, player profiles, chemistry clusters, burnout trends, coach reports, and MC prediction of V(s) on real sessions. Player/tournament CSVs drive drafting and map fit.', 'ops-data'],
        ['What parts use simulation?', 'How players react to lobby difficulty (the MDP dynamics), every RL training/evaluation episode, bandit rewards, and the synthetic queue in simulation mode. Transition probabilities are a model estimate from that simulator. Everything is labelled in the UI.', 'acad-limitations'],
        ['What are the project\'s limitations?', 'The dynamics are hand-specified, not fitted to players; the 48-bucket state is a coarse, possibly non-Markov summary of real psychology; rage quits and closeness are proxies in typical data; no real interventions were run. See Limitations for the full list.', 'acad-limitations'],
      ];
      el.innerHTML = `${App.header('Viva questions', 'Thirty questions a professor is likely to ask, each answered from this project and linked to the screen that demonstrates it live. Numbers quoted here come from the seeded experiment suite.', [['u1', 'U1'], ['u2', 'U2'], ['u3', 'U3'], ['u4', 'U4'], ['u5', 'U5']])}
      <div class="panel">${QA.map((q, i) => `<div class="qa"><h4><span>${String(i + 1).padStart(2, '0')}</span>${q[0]}</h4><p>${q[1]}</p><div class="row"><button class="btn sm" data-go3="${q[2]}">Show it live</button></div></div>`).join('')}</div>
      ${dp && keep ? `<div class="panel" style="margin-top:16px"><h2>One-line result to quote</h2><p class="dim">On 1000 identical simulated sessions, the DP-optimal controller raised the discounted return from ${f(keep.discounted_return.mean, 2)} (always KEEP, v3-style) to ${f(dp.discounted_return.mean, 2)} and cut rage-quit sessions from ${App.pct(keep.rage_quit_rate.mean, 1)} to ${App.pct(dp.rage_quit_rate.mean, 1)}; the model-free Q-learner reached ${f(ql.discounted_return.mean, 2)} and the hand-written heuristic ${f(heur.discounted_return.mean, 2)}. ${App.prov('SIMULATED')}</p></div>` : ''}`;
      $$('[data-go3]', el).forEach(b => b.addEventListener('click', () => App.go(b.dataset.go3)));
    },
  };

  // =================================================================== SYLLABUS
  App.screens['acad-syllabus'] = {
    async render(el) {
      const cov = await App.api('GET', '/academic/coverage');
      const counts = cov.status_counts; const rows = cov.rows;
      el.innerHTML = `${App.header('Syllabus mapping', 'Every concept from the uploaded Units 1–5 notes that the project uses, where it is implemented and where to see it. Status is honest: “demonstrated” means measured on live runs, not formally proved; “not covered” items are listed too.', [['u1', 'Unit 1'], ['u2', 'Unit 2'], ['u3', 'Unit 3'], ['u4', 'Unit 4'], ['u5', 'Unit 5']])}
        <div class="kpis" style="margin-bottom:16px">${Object.entries(counts).map(([k, v]) => App.kpi(v, k.toLowerCase())).join('')}</div>
        <div class="panel"><div class="row" style="margin-bottom:12px"><div class="seg" id="syU">${['All', 1, 2, 3, 4, 5, 'Other'].map((u, i) => `<button aria-pressed="${i === 0}" data-u="${u}">${u === 'All' || u === 'Other' ? u : 'Unit ' + u}</button>`).join('')}</div></div><div id="syT"></div></div>`;
      const draw = u => { const rs = rows.filter(r => u === 'All' ? true : u === 'Other' ? r.unit === 0 : r.unit === +u); $('#syT', el).innerHTML = App.table([{ h: 'Unit', f: r => r.unit ? `<span class="unit u${r.unit}">U${r.unit}</span>` : '<span class="unit ux">—</span>' }, { h: 'Concept', f: r => `<b>${esc(r.concept)}</b>` }, { h: 'Project use', f: r => esc(r.project_use) }, { h: 'Module / function', f: r => `<span class="mono" style="font-size:11.5px">${esc(r.module)}</span><div class="faint" style="font-size:11px">${esc(r.function)}</div>` }, { h: 'Demo screen', f: r => { const id = screenFor(r.screen); return id ? `<a href="#${id}" data-go4="${id}">${esc(r.screen)}</a>` : esc(r.screen); } }, { h: 'Status', f: r => `<span class="tag" style="color:${{ IMPLEMENTED: 'var(--green)', DEMONSTRATED: 'var(--cyan)', EXPLAINED: 'var(--ink-2)', 'NOT COVERED': 'var(--amber)' }[r.status]}">${r.status.toLowerCase()}</span>` }], rs); $$('[data-go4]', el).forEach(a => a.addEventListener('click', ev => { ev.preventDefault(); App.go(a.dataset.go4); })); };
      $$('#syU button', el).forEach(b => b.addEventListener('click', () => { $$('#syU button', el).forEach(x => x.setAttribute('aria-pressed', String(x === b))); draw(b.dataset.u); }));
      draw('All');
    },
  };

  // =================================================================== FORMULATION
  App.screens['acad-formulation'] = {
    async render(el) {
      const sp = await App.api('GET', '/rl/spec'); const P = sp.env_params;
      el.innerHTML = `${App.header('Mathematical formulation', 'The finite MDP ⟨S, A, P, R, γ⟩ behind every RL screen, with the simulator assumptions written out. All of it is generated from rl_core.spec(), the same object the algorithms import.', [['u1', 'Unit 1'], ['u3', 'Unit 3']])}
      <div class="grid g2">
        <div class="panel"><h2>The MDP</h2><div class="eq wrap" style="line-height:2">
          <span class="c">S</span> = {0,1,2,3}<sub>fatigue</sub> × {0,1,2,3}<sub>tilt</sub> × {L, N, W}<sub>streak</sub>  ∪ {END},  |S| = 48 + 1<br>
          <span class="m">A</span> = {LOWER, KEEP, RAISE}, δ(a) = −1, 0, +1 difficulty band<br>
          <span class="a">P</span>(s′|s,a): simulator law, estimated as p̂ = N(s,a,s′)/N(s,a) from ${'1000'} rollouts per (s,a)<br>
          <span class="g">R</span>(s,a,s′) = ${sp.reward.weights.w1}·cont + ${sp.reward.weights.w2}·C(s,a) − ${sp.reward.weights.w3}·rage,   C = 1 − 2|p<sub>win</sub>(s,a) − ½|<br>
          γ ∈ [0,1], default 0.9;  episode ends at rage quit, boredom quit or exhaustion</div>
          <h3>Win probability (drives closeness)</h3><div class="eq">p<sub>win</sub>(s,a) = clip(0.5 + ${P.streak_w}(k−1) − ${P.fatigue_w}·f − ${P.tilt_w}·t − ${P.action_w}·δ(a), ${P.p_win_min}, ${P.p_win_max})</div>
          <h3>Terminal events after a match</h3><div class="eq wrap">P(rage | loss) = ${P.rage_base} + ${P.rage_tilt}·t + ${P.rage_fatigue}·f + ${P.rage_losing_streak}·[losing] + ${P.rage_raise}·[RAISE] + ${P.rage_stomp}·[p<sub>win</sub> < ${P.stomp_threshold}]<br>P(boredom | win) = [p<sub>win</sub> ≥ ${P.boredom_threshold}]·(${P.boredom_base} + ${P.boredom_hot}·[winning])<br>P(fatigue ↑) = ${P.fatigue_up.join(' / ')} for LOWER / KEEP / RAISE; an EXHAUSTED player who tires further logs off (no penalty)</div>
          <h3>Tilt and streak</h3><div class="eq wrap">loss: tilt +1 w.p. ${P.tilt_up_loss} (+${P.tilt_up_stomp_bonus} if stomped);  win: tilt −1 w.p. ${P.tilt_down_win} (+${P.tilt_down_close_bonus} if C ≥ ${P.close_threshold});  one opposite result breaks a streak, a same result extends NEUTRAL w.p. ${P.streak_step}</div>
        </div>
        <div class="stack">
          <div class="panel"><h2>Why it is Markov</h2><p class="dim">env_step reads only the current bucketed state (f, t, k) and the action, and draws exactly five uniforms — no hidden history. So P(S<sub>t+1</sub> | S<sub>t</sub>, A<sub>t</sub>, S<sub>t−1</sub>, …) = P(S<sub>t+1</sub> | S<sub>t</sub>, A<sub>t</sub>) holds exactly for the simulator. For real players the buckets are a summary, so the property is an assumption (see Limitations).</p></div>
          <div class="panel"><h2>Environment classification (Unit 1)</h2>${App.table([{ h: 'Property', k: 'k' }, { h: 'Clutch IQ session', k: 'v' }], Object.entries(sp.environment_properties).map(([k, v]) => ({ k: k.replace(/_/g, ' '), v })))}</div>
          <div class="panel"><h2>Reward scaling review</h2><p class="dim">v3 used R = 1.0·(session minutes Δ, −20…+15) + 1.2·closeness − 3·rage, so the minutes term dominated and the rage penalty was smaller than one bad match. v4 keeps the same weights and structure but makes each term comparable: the session term is the indicator “match completed and session continues” (0/1). Per-step rewards: rage quit ${sp.reward.range_rage_quit.map(x => x.toFixed(1)).join(' … ')}, completed match ${sp.reward.range_completed.map(x => x.toFixed(1)).join(' … ')}. A rage quit also ends the episode, forfeiting all future continuation reward.</p></div>
          <div class="panel"><h2>Start-state distributions</h2>${App.table([{ h: 'Name', k: 'k' }, { h: 'Definition', k: 'v' }], Object.entries(sp.start_distributions).map(([k, v]) => ({ k, v })))}<p class="sub" style="margin-top:8px">Training and model estimation use all 48 states (coverage); evaluation uses realistic session starts.</p></div>
          <div class="panel"><h2>Naming correction</h2><p class="dim">The v3 optimiser was described as a “DQN”. It is tabular Q-learning: no neural approximator, no experience replay, no target network. A true DQN is listed as a possible extension only; for 48 discrete states a table is exact and the Units 1–5 notes do not cover deep RL.</p></div>
        </div>
      </div>`;
    },
  };

  // =================================================================== NOTES NUMERICALS
  App.screens['acad-numericals'] = {
    async render(el) {
      const d = await App.api('GET', '/academic/numericals');
      el.innerHTML = `${App.header('Lecture-note numericals', `Worked examples printed in the uploaded Units 2–5 notes, recomputed by the same functions that run Clutch IQ. ${d.matched} of ${d.checked} answers match exactly; one assignment has no printed answer and is shown for reference.`, [['u2', 'Unit 2'], ['u3', 'Unit 3'], ['u4', 'Unit 4'], ['u5', 'Unit 5']])}
        <div class="grid g2" id="nuG"></div>`;
      $('#nuG', el).innerHTML = d.items.map(it => `<div class="panel"><div class="row between"><h2><span class="unit u${it.unit}">U${it.unit}</span> ${esc(it.title)}</h2><span class="verdict ${it.match === true ? 'ok' : it.match === false ? 'no' : 'na'}">${it.match === true ? 'matches notes' : it.match === false ? 'mismatch' : 'no printed answer'}</span></div>${this.body(it)}<div class="row between" style="margin-top:8px"><span class="faint mono" style="font-size:11px">${esc(it.code || '')}</span>${it.notes_answer ? `<span class="faint" style="font-size:11.5px">notes: ${esc(JSON.stringify(it.notes_answer)).slice(0, 140)}</span>` : ''}</div>${it.note ? `<p class="sub" style="margin-top:6px">${esc(it.note)}</p>` : ''}</div>`).join('');
    },
    body(it) {
      const r = it.result;
      if (r.rows && Array.isArray(r.rows) && r.rows.length && typeof r.rows[0] === 'object') {
        const keys = Object.keys(r.rows[0]).filter(k => !['verdict'].includes(k));
        const tbl = App.table(keys.map(k => ({ h: esc(k), n: typeof r.rows[0][k] === 'number' ? 1 : 0, f: x => Array.isArray(x[k]) ? x[k].map(v => typeof v === 'number' ? f(v, 3) : v).join(', ') : typeof x[k] === 'number' ? (Number.isInteger(x[k]) ? x[k] : f(x[k], 4)) : esc(String(x[k])) })), r.rows, { scroll: true });
        const extra = Object.entries(r).filter(([k, v]) => k !== 'rows' && typeof v !== 'object').map(([k, v]) => `<span class="tag">${esc(k)} = ${typeof v === 'number' ? f(v, 4) : esc(v)}</span>`).join(' ');
        return tbl + (extra ? `<div class="row" style="margin-top:8px">${extra}</div>` : '');
      }
      const nf = x => Number.isInteger(x) ? String(x) : f(x, 4);
      const fmtv = v => Array.isArray(v) ? v.map(x => typeof x === 'number' ? nf(x) : x).join(', ') : (v && typeof v === 'object') ? Object.entries(v).map(([a, b]) => `${a}: ${typeof b === 'number' ? f(b, 4) : b}`).join(' · ') : (typeof v === 'number' ? nf(v) : esc(String(v)));
      return `<div class="kv" style="grid-template-columns:auto 1fr">${Object.entries(r).map(([k, v]) => `<span>${esc(k)}</span><span style="text-align:left">${fmtv(v)}</span>`).join('')}</div>`;
    },
  };

  // =================================================================== EXPERIMENTS
  App.screens['acad-experiments'] = {
    render(el) {
      const meta = S.experiments || {};
      if (!EXP.length) { el.innerHTML = App.header('Experiment results', 'Run <span class="mono">python rl_experiments.py</span> to generate results.', []); return; }
      const when = meta.generated_at_unix ? new Date(meta.generated_at_unix * 1000).toISOString().slice(0, 16).replace('T', ' ') + ' UTC' : '';
      const e1 = E('E1'), e2 = E('E2'), e3 = E('E3'), e4 = E('E4'), e5 = E('E5'), e6 = E('E6'), e7 = E('E7'), e8 = E('E8');
      el.innerHTML = `${App.header('Experiment results', `Produced by <span class="mono">python rl_experiments.py</span> (${when}, ${meta.total_seconds}s, seeds fixed). Every value below is read from experiments/results.json — nothing is typed by hand. ${esc(meta.provenance || '')}`, [['u1', 'Seeded'], ['ux', 'Reproducible']])}
      <div class="panel"><h2>E1 · ${esc(e1.title)}</h2>${App.table([{ h: 'Samples / (s,a)', n: 1, f: r => r.samples_per_pair }, { h: 'max |p̂ − p|', n: 1, f: r => f(r.max_abs_error, 4) }, { h: 'mean |p̂ − p|', n: 1, f: r => f(r.mean_abs_error, 4) }, { h: 'binomial s.e. bound', n: 1, f: r => f(r.worst_case_se, 4) }, { h: 'π̂* agrees with exact π*', n: 1, f: r => App.pct(r.policy_agreement_with_exact_optimum, 1) }, { h: 'true J of planned policy', n: 1, f: r => f(r.J_of_planned_policy_on_exact_model, 4) }, { h: 'exact J*', n: 1, f: r => f(r.J_exact_optimum, 4) }, { h: 'build time', n: 1, f: r => f(r.build_seconds, 2) + ' s' }], e1.rows)}<p class="sub" style="margin-top:8px">More samples shrink model error roughly as 1/√n, and the policy planned on the estimated model loses less value when executed on the true dynamics.</p></div>
      <div class="grid g2" style="margin-top:16px"><div class="panel"><h2>E2 · ${esc(e2.title)}</h2>${App.table([{ h: 'γ', n: 1, f: r => r.gamma }, { h: 'VI sweeps', n: 1, f: r => r.vi_iterations }, { h: 'max ratio', n: 1, f: r => f(r.vi_max_contraction_ratio, 4) }, { h: 'PI steps', n: 1, f: r => r.pi_improvement_steps }, { h: 'PI eval sweeps', n: 1, f: r => r.pi_eval_sweeps }, { h: 'PI = VI', n: 1, f: r => App.pct(r.pi_equals_vi, 0) }, { h: 'closed-form gap', n: 1, f: r => f(r.closed_form_gap, 8) }], e2.rows)}<p class="sub" style="margin-top:8px">Sweeps grow with γ because the contraction factor approaches 1; the measured ratio never exceeds γ.</p></div>
        <div class="panel"><h2>E3 · ${esc(e3.title)}</h2><div id="ex3"></div></div></div>
      <div class="grid g2" style="margin-top:16px"><div class="panel"><h2>E4 · ${esc(e4.title)}</h2><div id="ex4"></div><p class="sub">Mean over seeds ${e4.seeds.join(', ')}; shaded band = ±1 sd.</p></div>
        <div class="panel"><h2>E5 · ${esc(e5.title)}</h2>${App.table([{ h: 'Algorithm', f: r => esc(r.algorithm) }, { h: 'episodes', n: 1, f: r => r.episodes.toLocaleString() }, { h: 'J mean ± sd', n: 1, f: r => `${f(r.J_mean, 3)} ± ${f(r.J_std, 3)}` }, { h: '% of J*', n: 1, f: r => App.pct(r.J_mean / e5.J_star, 1) }, { h: 'agreement', n: 1, f: r => App.pct(r.agreement_mean, 1) }, { h: 'seeds', n: 1, f: r => r.seeds }], e5.rows)}<p class="sub" style="margin-top:8px">References: J* = ${f(e5.J_star, 3)}, heuristic ${f(e5.J_heuristic, 3)}, always KEEP ${f(e5.J_keep, 3)}.</p></div></div>
      <div class="panel" style="margin-top:16px"><h2>E6 · ${esc(e6.title)}</h2>${App.table([{ h: 'Policy', f: r => esc(r.policy) }, { h: 'Discounted return', n: 1, f: r => `${f(r.discounted_return.mean, 3)} ± ${f(r.discounted_return.ci95, 3)}` }, { h: 'Δ vs DP (paired)', n: 1, f: r => `${App.sgn(r.paired_diff_vs_dp.mean, 3)} ± ${f(r.paired_diff_vs_dp.ci95, 3)}` }, { h: 'Length', n: 1, f: r => f(r.session_length.mean, 2) }, { h: 'Rage quit', n: 1, f: r => App.pct(r.rage_quit_rate.mean, 1) }, { h: 'Close match', n: 1, f: r => App.pct(r.close_match_rate.mean, 1) }, { h: 'Boredom quit', n: 1, f: r => App.pct(r.boredom_quit_rate.mean, 1) }, { h: 'Agree w/ DP', n: 1, f: r => App.pct(r.agreement_with_dp, 0) }, { h: 'Stability', n: 1, f: r => r.stability === null ? '—' : App.pct(r.stability, 0) }], e6.rows, { rowClass: r => r.policy.startsWith('Dynamic Programming — Value') ? 'best' : '' })}<p class="sub" style="margin-top:8px">${e6.settings.n_episodes} episodes, seed ${e6.settings.seed}, start = ${e6.settings.start_dist}, γ = ${e6.settings.gamma}, MC ${e6.settings.mc_episodes} episodes, Q-learning ${e6.settings.q_params.episodes} episodes, ${e6.settings.stability_runs} stability runs.</p></div>
      <div class="grid g2" style="margin-top:16px"><div class="panel"><h2>E7 · ${esc(e7.title)}</h2><div id="ex7"></div>${App.table([{ h: 'Algorithm', f: r => esc(r.algorithm) }, { h: 'regret', n: 1, f: r => f(r.final_regret, 1) }, { h: 'reward ± CI', n: 1, f: r => `${f(r.final_reward, 1)} ± ${f(r.final_reward_ci95, 1)}` }, { h: '% optimal (last 10%)', n: 1, f: r => App.pct(r.pct_optimal_last_10pct, 1) }], e7.stationary)}<p class="sub" style="margin-top:8px">${e7.runs} runs. Balance patch at T/2 (2000 rounds): ${e7.nonstationary_patch.map(r => `${esc(r.update)} regret after patch ${f(r.regret_after_patch, 1)}`).join(' vs ')} — constant α tracks the change.</p></div>
        <div class="panel"><h2>E8 · ${esc(e8.title)}</h2>${App.table([{ h: 'Scenario', f: r => esc(r.scenario) }, { h: 'truth', n: 1, f: r => f(r.truth_dp, 3) }, { h: 'OIS', n: 1, f: r => f(r.ordinary_is, 3) }, { h: 'WIS', n: 1, f: r => f(r.weighted_is, 3) }, { h: 'PDIS', n: 1, f: r => f(r.per_decision_is, 3) }, { h: 'ESS', n: 1, f: r => f(r.ess, 0) }, { h: 'diagnostic', f: r => r.reliability === 'OK' ? '<span class="good">reliable</span>' : '<span class="warn">unreliable</span>' }], e8.rows)}<p class="sub" style="margin-top:8px">Importance sampling is trustworthy only when the new policy stays close to the logged one; the effective-sample-size diagnostic says when not to trust it.</p>
        <h3>ε sweep (E7)</h3><div id="ex7e"></div></div></div>`;
      C.bars($('#ex3', el), { categories: e3.rows.map(r => 'γ=' + r.gamma), series: [{ name: 'LOWER', values: e3.rows.map(r => r.n_lower), color: App.ACOL[0] }, { name: 'KEEP', values: e3.rows.map(r => r.n_keep), color: '#7d8da3' }, { name: 'RAISE', values: e3.rows.map(r => r.n_raise), color: App.ACOL[2] }], height: 210, fmt: v => v.toFixed(0) });
      $('#ex3', el).insertAdjacentHTML('beforeend', App.table([{ h: 'γ', n: 1, f: r => r.gamma }, { h: 'rage quit', n: 1, f: r => App.pct(r.sim_rage_quit_rate, 1) }, { h: 'length', n: 1, f: r => f(r.sim_session_length, 2) }, { h: 'close match', n: 1, f: r => App.pct(r.sim_close_match_rate, 1) }], e3.rows));
      C.line($('#ex4', el), { series: [{ name: 'first-visit', points: e4.rows.map(r => [r.episodes, r.mae_first_visit_mean]), band: e4.rows.map(r => [r.episodes, r.mae_first_visit_mean - r.mae_first_visit_std, r.mae_first_visit_mean + r.mae_first_visit_std]), markers: true }, { name: 'every-visit', points: e4.rows.map(r => [r.episodes, r.mae_every_visit_mean]), markers: true }], logX: true, xLabel: 'episodes (log)', yLabel: 'MAE vs DP', height: 240, zero: true });
      const ALGC = { greedy: '#0aa5c4', epsilon_greedy: '#c43fb0', optimistic: '#2f9e5e', ucb: '#7f6ff0', explore_first: '#c98500', random: '#3d78e0' };
      C.line($('#ex7', el), { series: e7.stationary.map(r => ({ name: r.algorithm.replace(' (uniform random)', '').replace(' (exploit only)', ''), color: ALGC[r.key], points: r.curve_rounds.map((t, i) => [t, r.curve_regret[i]]) })), xLabel: 'round', yLabel: 'mean cumulative regret', height: 240 });
      C.line($('#ex7e', el), { series: [{ name: 'regret', points: e7.epsilon_sweep.map(r => [r.epsilon, r.final_regret]), markers: true }], xLabel: 'ε', yLabel: 'regret after 1000 rounds', height: 200 });
    },
  };

  // =================================================================== ARCHITECTURE
  App.screens['acad-architecture'] = {
    render(el) {
      const box = (x, y, w, h, t, s, c) => `<g><rect x="${x}" y="${y}" width="${w}" height="${h}" rx="9" fill="#0e141e" stroke="${c}"/><text x="${x + 12}" y="${y + 22}" style="font-family:'Chakra Petch';font-weight:600;font-size:14px;fill:#e7eef6">${t}</text>${s.map((l, i) => `<text x="${x + 12}" y="${y + 42 + i * 16}" style="font-size:11.5px;fill:#a9b6c6">${l}</text>`).join('')}</g>`;
      const ar = (x1, y1, x2, y2) => `<path d="M${x1},${y1} L${x2},${y2}" stroke="#6c7b8e" stroke-width="1.4" marker-end="url(#ah2)"/>`;
      const svg = `<svg viewBox="0 0 1000 560" class="arch" role="img" aria-label="Architecture diagram"><defs><marker id="ah2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0L10,5L0,10z" fill="#6c7b8e"/></marker></defs>
        ${box(20, 20, 220, 110, 'Data in', ['Match-history CSV (real)', 'maps.csv + players.csv', 'logged RL sessions (optional)'], '#3fdf8f')}
        ${box(20, 160, 220, 110, 'Operations', ['schema_detector · feature_engine', 'gnn_chemistry · cluster_labeler', 'tournament_builder · burnout · coach'], '#3fdf8f')}
        ${box(300, 20, 260, 130, 'rl_core.py — the formal problem', ['48 states, 3 actions, reward', 'SessionEnv simulator (seeded)', 'policies · episodes · returns', 'Mulberry32 PRNG (shared with JS)'], '#22d8f5')}
        ${box(300, 180, 260, 90, 'mdp_model.py', ['simulation-estimated p̂(s′|s,a), r̂', 'Markov chain P_π, MRP closed form'], '#22d8f5')}
        ${box(620, 20, 170, 90, 'bandits.py', ['map arms · ε-greedy · UCB', 'optimistic · regret'], '#ff3dc8')}
        ${box(620, 130, 170, 70, 'dynamic_programming', ['PE · PI · VI · backups'], '#9d8cff')}
        ${box(620, 215, 170, 70, 'monte_carlo.py', ['FV/EV · control · IS'], '#ffb020')}
        ${box(810, 130, 170, 70, 'q_learning.py', ['tabular TD control'], '#ff3dc8')}
        ${box(810, 215, 170, 70, 'rl_evaluation', ['seeded arena · CIs'], '#a9b6c6')}
        ${box(300, 320, 480, 80, 'FastAPI — backend_app.py + rl_api.py', ['/api/upload … /api/tournament (v3, unchanged)  ·  /api/rl/*  ·  /api/bandit/*  ·  /api/academic/*'], '#22d8f5')}
        ${box(20, 320, 220, 80, 'rl_experiments.py', ['seeded suite → results.json', 'feeds EXPERIMENTS.md'], '#a9b6c6')}
        ${box(300, 440, 480, 100, 'Frontend (index.html + assets/)', ['app_core · ops · control · lab · academic · charts', 'rl_engine.js: offline mirror of every RL endpoint (parity-tested)', 'static_data.js: production Q-table, maps, syllabus, experiments'], '#ff3dc8')}
        ${box(820, 440, 160, 100, 'Tests (pytest)', ['algorithms · API', 'notes numericals', 'Python ↔ JS parity'], '#3fdf8f')}
        ${ar(130, 130, 130, 160)}${ar(240, 75, 300, 75)}${ar(430, 150, 430, 180)}${ar(560, 70, 620, 65)}${ar(560, 210, 620, 165)}${ar(560, 240, 620, 250)}${ar(560, 110, 810, 160)}${ar(790, 250, 810, 250)}${ar(130, 270, 300, 340)}${ar(540, 285, 540, 320)}${ar(240, 360, 300, 360)}${ar(540, 400, 540, 440)}
      </svg>`;
      el.innerHTML = `${App.header('Architecture', 'One shared problem definition, several algorithm families, one API, and a browser engine that mirrors it so the demo never depends on a server.', [['ux', 'System']])}
        <div class="panel">${svg}</div>
        <div class="grid g2" style="margin-top:16px"><div class="panel"><h2>Request flow</h2><p class="dim">The UI calls <span class="mono">App.api(method, path)</span>. If the Python backend answers <span class="mono">/api/health</span>, requests go to FastAPI; otherwise the same path is served by <span class="mono">ClutchRL.call()</span> in the browser with identical JSON shapes. Upload-based features (CSV, tournament, burnout, logged episodes) always need the backend.</p><h3>Design rules</h3><ul style="margin-left:18px;color:var(--ink-2);font-size:13px;display:grid;gap:4px"><li>Every algorithm imports S, A, R and the simulator from rl_core — no duplicated definitions.</li><li>All randomness is seeded; environment and agent use separate streams (common random numbers).</li><li>Every response carries a provenance label.</li><li>The production Q-table is changed only by an explicit request.</li></ul></div>
        <div class="panel"><h2>Endpoints added in v4</h2>${App.table([{ h: 'Endpoint', f: r => `<span class="mono" style="font-size:11.5px">${r[0]}</span>` }, { h: 'Purpose', f: r => r[1] }], [['GET /api/rl/spec', 'formal MDP definition'], ['GET /api/rl/mdp', 'estimated P, R, model quality'], ['GET /api/rl/mdp/state/{s}', 'explorer: V, Q, Bellman backups'], ['GET /api/rl/mdp/markov-chain', 'P_π, survival, closed form'], ['POST /api/rl/dp/policy-evaluation | policy-iteration | value-iteration | gamma-sweep', 'Unit 4'], ['POST /api/rl/mc/predict | control | off-policy', 'Unit 5'], ['GET /api/rl/mc/real/{dataset}', 'MC on uploaded real sessions'], ['POST /api/rl/episodes/upload · GET /api/rl/episodes/sample', 'logged sessions'], ['GET /api/rl/q/table · POST /api/rl/q/train | td-step', 'Q-learning'], ['POST /api/rl/trajectory', 'one seeded episode'], ['POST /api/rl/compare', 'comparison arena'], ['POST /api/bandit/run | arms · GET /api/bandit/numericals', 'Unit 2'], ['GET /api/academic/numericals | coverage | viva-snapshot', 'academic']])}</div></div>`;
    },
  };

  // =================================================================== LIMITATIONS
  App.screens['acad-limitations'] = {
    render(el) {
      const L = [
        ['Simulated dynamics', 'How a player reacts to an easier or harder lobby is specified by documented assumptions in rl_core.ENV_PARAMS, not fitted to real players. All RL policies are optimal for the simulator, not proven for people.', 'SIMULATED'],
        ['Coarse state', '48 buckets of fatigue, tilt and streak summarise a player. Real psychology has more dimensions (time of day, party, role), so the Markov property is exact only for the simulator.', 'DERIVED FEATURE'],
        ['Proxies in real data', 'Typical match histories log neither rage quits nor matchmaking actions. Rage quit is a proxy (loss + tilt > 0.5 on a session\'s last match); closeness defaults to 0.5 without scorelines. Fatigue uses the session length, which is only known after the session.', 'FALLBACK/PROXY'],
        ['Estimated model', 'DP plans on p̂ from 1000 rollouts per (s,a): max error ≈ 0.04 (E1). The planned policy loses ≈ 0.5% of value vs the exact optimum.', 'MODEL ESTIMATE'],
        ['Off-policy evaluation variance', 'Importance sampling is reliable only for small policy changes; for large changes the effective sample size collapses (E8). The UI flags this instead of hiding it.', 'SIMULATED'],
        ['Constant-step Q-learning', 'With a constant α the table keeps fluctuating and near-tied actions flip between seeds; a decaying step size or more episodes would tighten it.', 'SIMULATED'],
        ['Bandit arm means', 'Map satisfaction q*(map) is a monotone transform of the rule-based fit score, so the bandit cannot discover preferences the fit score does not encode. Real per-match feedback would replace it.', 'DERIVED FEATURE'],
        ['Tournament session state', 'players.csv has no session history, so tilt/fatigue in the tournament builder are small name-seeded fallback values, labelled as such.', 'FALLBACK/PROXY'],
        ['No live deployment', 'No intervention was run on real players; there is no online A/B test. Ethical deployment would need consent and wellbeing safeguards (the exhaustion terminal and burnout module are only a start).', 'SIMULATED'],
        ['Not covered from the syllabus', 'Thompson sampling (named in Unit 2 only), Green\'s equation (Unit 3) and deep RL (DQN) are not implemented. Cauchy-sequence and Banach fixed-point properties are demonstrated by measurement, not proved.', 'DERIVED FEATURE'],
        ['In-memory storage', 'Uploaded datasets and logs live in server memory (like v3); a restart or a new serverless instance forgets them.', 'DERIVED FEATURE'],
      ];
      el.innerHTML = `${App.header('Limitations & honesty', 'What this project does not show, and where each number comes from.', [['ux', 'Academic honesty']])}
        <div class="grid g2">${L.map(l => `<div class="panel"><div class="row between"><h2>${l[0]}</h2>${App.prov(l[2])}</div><p class="dim" style="font-size:13px">${l[1]}</p></div>`).join('')}</div>`;
    },
  };
})(window);
