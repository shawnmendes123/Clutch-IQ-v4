# Subject mapping — Units 1–5 → Clutch IQ implementation

Generated from `syllabus.py` by `tools/build_docs.py` (the same data drives the
**Academic › Syllabus mapping** screen and `GET /api/academic/coverage`).

Status vocabulary: **IMPLEMENTED** = real computation, visible in the app and
unit-tested · **DEMONSTRATED** = property measured on live runs, not formally
proved · **EXPLAINED** = covered in explanation panels only · **NOT COVERED** =
in the syllabus but not part of this project.

Totals: 47 implemented, 1 explained, 3 not covered, 2 demonstrated.

Every worked example with a printed answer in the uploaded notes (Units 2–5) is
recomputed by the project code and checked — see **Academic › Notes numericals**
and `tests/`.

## Unit 1 — Introduction to RL

| Concept | Project use | Module / function | Demo screen | Status |
|---|---|---|---|---|
| RL problem: agent, environment, state, action, reward | Session controller (agent) adjusting lobby difficulty for a simulated player session (environment) | `rl_core.py` — spec(), SessionEnv, env_step | Control Room › Live Session | **IMPLEMENTED** |
| Policy π(a|s) (deterministic & stochastic) | Every policy is a 49×3 matrix; ε-soft, uniform, greedy and learned policies | `rl_core.py` — uniform_policy, epsilon_soft, deterministic_policy | Live Session, MDP Explorer | **IMPLEMENTED** |
| Immediate reward vs delayed reward | r_t shown per match; G_t shows the delayed consequence (tilt → rage quit several matches later) | `rl_evaluation.py` — trajectory() | Live Session › Trajectory | **IMPLEMENTED** |
| Discounted return G_t and discount factor γ | γ slider recomputes every γ^t r_{t+1} and the optimal policy (γ-sweep) | `rl_core.py, dynamic_programming.py` — returns_from_rewards, value_iteration | Live Session, DP Lab › γ sweep | **IMPLEMENTED** |
| Episode and terminal state | Session ends on rage quit, boredom quit or exhaustion → absorbing END state | `rl_core.py` — env_step, TERMINAL | Live Session | **IMPLEMENTED** |
| Elements of RL: value function and model | V(s), Q(s,a) everywhere; the simulation-estimated model is the 'model' element | `mdp_model.py` — estimate_model | MDP Explorer | **IMPLEMENTED** |
| Model-based vs model-free RL | DP plans with p(s'|s,a); MC and Q-learning learn from episodes only — compared on identical seeds | `rl_evaluation.py` — compare_policies | RL Lab › Algorithm Comparison | **IMPLEMENTED** |
| Task-environment classification (observable, stochastic, discrete, episodic, single-agent) | Documented for the session environment | `rl_core.py` — spec()['environment_properties'] | Academic › Formulation | **EXPLAINED** |
| Immediate RL vs full RL | Map choice = immediate RL (bandit); session control = full RL (MDP) — kept on separate screens | `bandits.py, rl_core.py` — module docstrings | Bandit Arena, Control Room | **IMPLEMENTED** |

## Unit 2 — Multi-armed bandits

| Concept | Project use | Module / function | Demo screen | Status |
|---|---|---|---|---|
| k-armed bandit, q*(a) = E[R | A=a] | Maps are arms for a roster; q*(map) derived from the map-fit score | `bandits.py` — map_arms, BanditEnv | RL Lab › Bandit Arena | **IMPLEMENTED** |
| Bernoulli bandits | Default reward model: post-match thumbs-up ~ Bernoulli(q*) | `bandits.py` — BanditEnv.pull | Bandit Arena | **IMPLEMENTED** |
| Sample-average action-value estimation | Q_t(a) for every map | `bandits.py` — run_bandit | Bandit Arena › Q_t(a) | **IMPLEMENTED** |
| Incremental implementation Q_{n+1} = Q_n + (1/n)(R_n − Q_n) | Live update shown with real numbers | `bandits.py` — run_bandit | Bandit Arena › update explainer | **IMPLEMENTED** |
| Greedy / exploit-only | Baseline algorithm (gets stuck — shown by % optimal) | `bandits.py` — run_bandit('greedy') | Bandit Arena | **IMPLEMENTED** |
| Explore-only and fixed exploration + greedy | Baselines from the notes' worked examples | `bandits.py` — run_bandit('random'|'explore_first') | Bandit Arena | **IMPLEMENTED** |
| ε-greedy | Main exploration algorithm; ε slider; explore/exploit counts | `bandits.py` — run_bandit('epsilon_greedy') | Bandit Arena | **IMPLEMENTED** |
| Optimistic initial values | Q_0 slider | `bandits.py` — run_bandit('optimistic') | Bandit Arena | **IMPLEMENTED** |
| Upper Confidence Bound | Live Q + c·√(ln t / N) breakdown per arm | `bandits.py` — ucb_scores | Bandit Arena › UCB panel | **IMPLEMENTED** |
| Non-stationary problems, constant step size α | Meta drift / balance patch modes; α slider (exponential recency-weighted average) | `bandits.py` — BanditEnv(nonstationary=…) | Bandit Arena | **IMPLEMENTED** |
| Cumulative reward and regret | Expected regret Σ(max q* − q*(A_t)) and the notes' realised regret T·max q* − ΣR | `bandits.py` — run_bandit, run_arena | Bandit Arena | **IMPLEMENTED** |
| 10-armed testbed style averaging over runs | Same-seed multi-run comparison of algorithms | `bandits.py` — run_arena | Bandit Arena | **IMPLEMENTED** |
| Notes' worked numericals (exploration, exploitation, fixed exploration, UCB, ε-detection, η-update) | Recomputed by the engine and checked against the printed answers | `bandits.py` — all_notes_numericals | Academic › Notes Numericals | **IMPLEMENTED** |
| Thompson sampling | Named in the notes only as an alternative; not implemented | `—` — — | — | **NOT COVERED** |

## Unit 3 — Markov process & optimality

| Concept | Project use | Module / function | Demo screen | Status |
|---|---|---|---|---|
| Markov property | State (fatigue, tilt, streak) buckets are, by construction of the simulator, sufficient for the next-state law | `rl_core.py` — env_step (depends only on s, a) | MDP Explorer | **IMPLEMENTED** |
| Markov chain / transition matrix | Fixing a policy turns the MDP into a 49×49 chain P_π; sequence probability by the chain rule | `mdp_model.py` — policy_markov_chain, sequence_probability | MDP Explorer › Markov chain | **IMPLEMENTED** |
| Markov Reward Process (MRP) and its Bellman equation v = R + γPv | Closed-form solution (I − γP_π)⁻¹R_π compared with iterative policy evaluation | `mdp_model.py` — policy_mrp, mrp_closed_form | DP Lab › Policy Evaluation | **IMPLEMENTED** |
| Finite MDP (S, A, p(s'|s,a), r(s,a), r(s,a,s'), γ) | 48-state session MDP with a simulation-estimated transition model | `mdp_model.py` — FiniteMDP, estimate_model | MDP Explorer | **IMPLEMENTED** |
| State-value V_π(s) and action-value Q_π(s,a) | Shown for any state under any policy | `dynamic_programming.py` — policy_evaluation, FiniteMDP.q_from_v | MDP Explorer | **IMPLEMENTED** |
| Bellman expectation equation | Numerical backup with the program's actual p, r and successor values | `dynamic_programming.py` — backup_breakdown | MDP Explorer › Bellman backup | **IMPLEMENTED** |
| Bellman optimality equation | max over actions of the same backup; V* and Q* | `dynamic_programming.py` — value_iteration, backup_breakdown | MDP Explorer, DP Lab | **IMPLEMENTED** |
| Cauchy sequence (convergence of value sequences) | Measured: successive max-Δ shrink geometrically, so {V_k} is Cauchy in the max norm | `dynamic_programming.py` — value_iteration()['deltas'] | DP Lab › convergence chart | **DEMONSTRATED** |
| Green's equation | Not used in this project | `—` — — | — | **NOT COVERED** |

## Unit 4 — Dynamic programming

| Concept | Project use | Module / function | Demo screen | Status |
|---|---|---|---|---|
| Iterative policy evaluation | Evaluates any session policy on the estimated model; max-Δ per sweep | `dynamic_programming.py` — policy_evaluation | DP Lab | **IMPLEMENTED** |
| Policy improvement (and its monotonicity) | Greedy improvement; unit test checks V_π' ≥ V_π | `dynamic_programming.py` — policy_improvement | DP Lab | **IMPLEMENTED** |
| Policy iteration | Evaluation/improvement loop until policy-stable; history of changed states | `dynamic_programming.py` — policy_iteration | DP Lab | **IMPLEMENTED** |
| Value iteration | Optimal session-control policy π* over all 48 states | `dynamic_programming.py` — value_iteration | DP Lab, Algorithm Comparison | **IMPLEMENTED** |
| Banach fixed-point / γ-contraction | Measured contraction ratio Δ_{k+1}/Δ_k ≤ γ on every run (unit-tested) | `dynamic_programming.py` — value_iteration()['contraction_ratios'] | DP Lab | **DEMONSTRATED** |
| Efficiency of DP | Sweeps × |S|²|A| operations and wall-clock time reported | `dynamic_programming.py` — seconds / iterations fields | DP Lab | **IMPLEMENTED** |
| Notes' DP examples (dice game V(IN)=10.5, 4×4 gridworld) | The same DP code solves them in the unit tests and the Notes Numericals screen | `tests/test_dynamic_programming.py` — dice_game, gridworld | Academic › Notes Numericals | **IMPLEMENTED** |

## Unit 5 — Monte Carlo methods

| Concept | Project use | Module / function | Demo screen | Status |
|---|---|---|---|---|
| Monte Carlo prediction from complete episodes | Estimate V_π of a session policy from simulated sessions; MAE vs DP over episode count | `monte_carlo.py` — mc_prediction_run | RL Lab › Monte Carlo Lab | **IMPLEMENTED** |
| First-visit MC | Side-by-side with every-visit | `monte_carlo.py` — first_visit_prediction | Monte Carlo Lab | **IMPLEMENTED** |
| Every-visit MC | Side-by-side with first-visit | `monte_carlo.py` — every_visit_prediction | Monte Carlo Lab | **IMPLEMENTED** |
| MC estimation of action values Q(s,a) | Q(s,a) and visit counts per state-action pair | `monte_carlo.py` — mc_action_values, mc_control | Monte Carlo Lab | **IMPLEMENTED** |
| MC control with exploring starts (MC ES) | Simulator can start any (s, a) pair | `monte_carlo.py` — mc_control('es') | Monte Carlo Lab › Control | **IMPLEMENTED** |
| On-policy MC control with ε-soft policies | Learned session policy compared with DP and Q-learning | `monte_carlo.py` — mc_control('on_policy') | Monte Carlo Lab, Algorithm Comparison | **IMPLEMENTED** |
| Off-policy prediction: target vs behaviour policy, coverage | Estimate a NEW matchmaking policy's value from sessions logged under the OLD policy | `monte_carlo.py` — off_policy_prediction, coverage_check | Monte Carlo Lab › Off-policy | **IMPLEMENTED** |
| Importance-sampling ratio ρ_{t:T−1} | Per-step π/b ratios shown for real logged episodes | `monte_carlo.py` — off_policy_prediction | Monte Carlo Lab › Off-policy | **IMPLEMENTED** |
| Ordinary and weighted importance sampling | Both estimators with effective sample size diagnostics | `monte_carlo.py` — off_policy_prediction | Monte Carlo Lab › Off-policy | **IMPLEMENTED** |
| Off-policy MC control | Weighted-IS control (Sutton §5.7) with ε-soft behaviour | `monte_carlo.py` — mc_control('off_policy') | Monte Carlo Lab › Control | **IMPLEMENTED** |
| Importance sampling on truncated returns | Discounting-aware IS on flat partial returns + per-decision IS (variance reduction) | `monte_carlo.py` — off_policy_prediction (J_discounting_aware_*, J_per_decision_is) | Monte Carlo Lab › Off-policy | **IMPLEMENTED** |
| Notes' MC example (A/B episodes: FV 8.25/7.8, EV 8/4) | Recomputed by the same first/every-visit code | `tests/test_monte_carlo.py` — first_visit_prediction | Academic › Notes Numericals | **IMPLEMENTED** |

## Beyond Units 1–5 (syllabus overview topic)

| Concept | Project use | Module / function | Demo screen | Status |
|---|---|---|---|---|
| Tabular Q-learning (TD control) | Original Clutch IQ optimiser — kept as the model-free baseline, with TD target/error explainer and Training Lab | `q_learning.py` — train, td_update | Control Room › Q-Learning | **IMPLEMENTED** |
| Deep Q-Network (neural approximator, replay, target net) | Not implemented — unnecessary for 48 states and not in Units 1–5; described as a future extension | `—` — — | Academic › Limitations | **NOT COVERED** |
