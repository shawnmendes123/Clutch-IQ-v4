"""
syllabus.py — single source of truth for the subject-coverage matrix.

Served by /api/academic/coverage, rendered on the "Syllabus Mapping" screen
and used to generate SUBJECT_MAPPING.md, so the three can never disagree.

Status vocabulary (honest by construction):
  IMPLEMENTED   real computation in the codebase, visible in the app, unit-tested
  DEMONSTRATED  property measured on live computations (e.g. contraction ratio),
                but no formal proof is implemented
  EXPLAINED     covered only in the viva/explanation panels
  NOT COVERED   listed in the syllabus but not part of this project
"""

COVERAGE = [
    # ------------------------------------------------------------------ UNIT 1
    {"unit": 1, "concept": "RL problem: agent, environment, state, action, reward", "project_use": "Session controller (agent) adjusting lobby difficulty for a simulated player session (environment)", "module": "rl_core.py", "function": "spec(), SessionEnv, env_step", "screen": "Control Room › Live Session", "status": "IMPLEMENTED"},
    {"unit": 1, "concept": "Policy π(a|s) (deterministic & stochastic)", "project_use": "Every policy is a 49×3 matrix; ε-soft, uniform, greedy and learned policies", "module": "rl_core.py", "function": "uniform_policy, epsilon_soft, deterministic_policy", "screen": "Live Session, MDP Explorer", "status": "IMPLEMENTED"},
    {"unit": 1, "concept": "Immediate reward vs delayed reward", "project_use": "r_t shown per match; G_t shows the delayed consequence (tilt → rage quit several matches later)", "module": "rl_evaluation.py", "function": "trajectory()", "screen": "Live Session › Trajectory", "status": "IMPLEMENTED"},
    {"unit": 1, "concept": "Discounted return G_t and discount factor γ", "project_use": "γ slider recomputes every γ^t r_{t+1} and the optimal policy (γ-sweep)", "module": "rl_core.py, dynamic_programming.py", "function": "returns_from_rewards, value_iteration", "screen": "Live Session, DP Lab › γ sweep", "status": "IMPLEMENTED"},
    {"unit": 1, "concept": "Episode and terminal state", "project_use": "Session ends on rage quit, boredom quit or exhaustion → absorbing END state", "module": "rl_core.py", "function": "env_step, TERMINAL", "screen": "Live Session", "status": "IMPLEMENTED"},
    {"unit": 1, "concept": "Elements of RL: value function and model", "project_use": "V(s), Q(s,a) everywhere; the simulation-estimated model is the 'model' element", "module": "mdp_model.py", "function": "estimate_model", "screen": "MDP Explorer", "status": "IMPLEMENTED"},
    {"unit": 1, "concept": "Model-based vs model-free RL", "project_use": "DP plans with p(s'|s,a); MC and Q-learning learn from episodes only — compared on identical seeds", "module": "rl_evaluation.py", "function": "compare_policies", "screen": "RL Lab › Algorithm Comparison", "status": "IMPLEMENTED"},
    {"unit": 1, "concept": "Task-environment classification (observable, stochastic, discrete, episodic, single-agent)", "project_use": "Documented for the session environment", "module": "rl_core.py", "function": "spec()['environment_properties']", "screen": "Academic › Formulation", "status": "EXPLAINED"},
    {"unit": 1, "concept": "Immediate RL vs full RL", "project_use": "Map choice = immediate RL (bandit); session control = full RL (MDP) — kept on separate screens", "module": "bandits.py, rl_core.py", "function": "module docstrings", "screen": "Bandit Arena, Control Room", "status": "IMPLEMENTED"},
    # ------------------------------------------------------------------ UNIT 2
    {"unit": 2, "concept": "k-armed bandit, q*(a) = E[R | A=a]", "project_use": "Maps are arms for a roster; q*(map) derived from the map-fit score", "module": "bandits.py", "function": "map_arms, BanditEnv", "screen": "RL Lab › Bandit Arena", "status": "IMPLEMENTED"},
    {"unit": 2, "concept": "Bernoulli bandits", "project_use": "Default reward model: post-match thumbs-up ~ Bernoulli(q*)", "module": "bandits.py", "function": "BanditEnv.pull", "screen": "Bandit Arena", "status": "IMPLEMENTED"},
    {"unit": 2, "concept": "Sample-average action-value estimation", "project_use": "Q_t(a) for every map", "module": "bandits.py", "function": "run_bandit", "screen": "Bandit Arena › Q_t(a)", "status": "IMPLEMENTED"},
    {"unit": 2, "concept": "Incremental implementation Q_{n+1} = Q_n + (1/n)(R_n − Q_n)", "project_use": "Live update shown with real numbers", "module": "bandits.py", "function": "run_bandit", "screen": "Bandit Arena › update explainer", "status": "IMPLEMENTED"},
    {"unit": 2, "concept": "Greedy / exploit-only", "project_use": "Baseline algorithm (gets stuck — shown by % optimal)", "module": "bandits.py", "function": "run_bandit('greedy')", "screen": "Bandit Arena", "status": "IMPLEMENTED"},
    {"unit": 2, "concept": "Explore-only and fixed exploration + greedy", "project_use": "Baselines from the notes' worked examples", "module": "bandits.py", "function": "run_bandit('random'|'explore_first')", "screen": "Bandit Arena", "status": "IMPLEMENTED"},
    {"unit": 2, "concept": "ε-greedy", "project_use": "Main exploration algorithm; ε slider; explore/exploit counts", "module": "bandits.py", "function": "run_bandit('epsilon_greedy')", "screen": "Bandit Arena", "status": "IMPLEMENTED"},
    {"unit": 2, "concept": "Optimistic initial values", "project_use": "Q_0 slider", "module": "bandits.py", "function": "run_bandit('optimistic')", "screen": "Bandit Arena", "status": "IMPLEMENTED"},
    {"unit": 2, "concept": "Upper Confidence Bound", "project_use": "Live Q + c·√(ln t / N) breakdown per arm", "module": "bandits.py", "function": "ucb_scores", "screen": "Bandit Arena › UCB panel", "status": "IMPLEMENTED"},
    {"unit": 2, "concept": "Non-stationary problems, constant step size α", "project_use": "Meta drift / balance patch modes; α slider (exponential recency-weighted average)", "module": "bandits.py", "function": "BanditEnv(nonstationary=…)", "screen": "Bandit Arena", "status": "IMPLEMENTED"},
    {"unit": 2, "concept": "Cumulative reward and regret", "project_use": "Expected regret Σ(max q* − q*(A_t)) and the notes' realised regret T·max q* − ΣR", "module": "bandits.py", "function": "run_bandit, run_arena", "screen": "Bandit Arena", "status": "IMPLEMENTED"},
    {"unit": 2, "concept": "10-armed testbed style averaging over runs", "project_use": "Same-seed multi-run comparison of algorithms", "module": "bandits.py", "function": "run_arena", "screen": "Bandit Arena", "status": "IMPLEMENTED"},
    {"unit": 2, "concept": "Notes' worked numericals (exploration, exploitation, fixed exploration, UCB, ε-detection, η-update)", "project_use": "Recomputed by the engine and checked against the printed answers", "module": "bandits.py", "function": "all_notes_numericals", "screen": "Academic › Notes Numericals", "status": "IMPLEMENTED"},
    {"unit": 2, "concept": "Thompson sampling", "project_use": "Named in the notes only as an alternative; not implemented", "module": "—", "function": "—", "screen": "—", "status": "NOT COVERED"},
    # ------------------------------------------------------------------ UNIT 3
    {"unit": 3, "concept": "Markov property", "project_use": "State (fatigue, tilt, streak) buckets are, by construction of the simulator, sufficient for the next-state law", "module": "rl_core.py", "function": "env_step (depends only on s, a)", "screen": "MDP Explorer", "status": "IMPLEMENTED"},
    {"unit": 3, "concept": "Markov chain / transition matrix", "project_use": "Fixing a policy turns the MDP into a 49×49 chain P_π; sequence probability by the chain rule", "module": "mdp_model.py", "function": "policy_markov_chain, sequence_probability", "screen": "MDP Explorer › Markov chain", "status": "IMPLEMENTED"},
    {"unit": 3, "concept": "Markov Reward Process (MRP) and its Bellman equation v = R + γPv", "project_use": "Closed-form solution (I − γP_π)⁻¹R_π compared with iterative policy evaluation", "module": "mdp_model.py", "function": "policy_mrp, mrp_closed_form", "screen": "DP Lab › Policy Evaluation", "status": "IMPLEMENTED"},
    {"unit": 3, "concept": "Finite MDP (S, A, p(s'|s,a), r(s,a), r(s,a,s'), γ)", "project_use": "48-state session MDP with a simulation-estimated transition model", "module": "mdp_model.py", "function": "FiniteMDP, estimate_model", "screen": "MDP Explorer", "status": "IMPLEMENTED"},
    {"unit": 3, "concept": "State-value V_π(s) and action-value Q_π(s,a)", "project_use": "Shown for any state under any policy", "module": "dynamic_programming.py", "function": "policy_evaluation, FiniteMDP.q_from_v", "screen": "MDP Explorer", "status": "IMPLEMENTED"},
    {"unit": 3, "concept": "Bellman expectation equation", "project_use": "Numerical backup with the program's actual p, r and successor values", "module": "dynamic_programming.py", "function": "backup_breakdown", "screen": "MDP Explorer › Bellman backup", "status": "IMPLEMENTED"},
    {"unit": 3, "concept": "Bellman optimality equation", "project_use": "max over actions of the same backup; V* and Q*", "module": "dynamic_programming.py", "function": "value_iteration, backup_breakdown", "screen": "MDP Explorer, DP Lab", "status": "IMPLEMENTED"},
    {"unit": 3, "concept": "Cauchy sequence (convergence of value sequences)", "project_use": "Measured: successive max-Δ shrink geometrically, so {V_k} is Cauchy in the max norm", "module": "dynamic_programming.py", "function": "value_iteration()['deltas']", "screen": "DP Lab › convergence chart", "status": "DEMONSTRATED"},
    {"unit": 3, "concept": "Green's equation", "project_use": "Not used in this project", "module": "—", "function": "—", "screen": "—", "status": "NOT COVERED"},
    # ------------------------------------------------------------------ UNIT 4
    {"unit": 4, "concept": "Iterative policy evaluation", "project_use": "Evaluates any session policy on the estimated model; max-Δ per sweep", "module": "dynamic_programming.py", "function": "policy_evaluation", "screen": "DP Lab", "status": "IMPLEMENTED"},
    {"unit": 4, "concept": "Policy improvement (and its monotonicity)", "project_use": "Greedy improvement; unit test checks V_π' ≥ V_π", "module": "dynamic_programming.py", "function": "policy_improvement", "screen": "DP Lab", "status": "IMPLEMENTED"},
    {"unit": 4, "concept": "Policy iteration", "project_use": "Evaluation/improvement loop until policy-stable; history of changed states", "module": "dynamic_programming.py", "function": "policy_iteration", "screen": "DP Lab", "status": "IMPLEMENTED"},
    {"unit": 4, "concept": "Value iteration", "project_use": "Optimal session-control policy π* over all 48 states", "module": "dynamic_programming.py", "function": "value_iteration", "screen": "DP Lab, Algorithm Comparison", "status": "IMPLEMENTED"},
    {"unit": 4, "concept": "Banach fixed-point / γ-contraction", "project_use": "Measured contraction ratio Δ_{k+1}/Δ_k ≤ γ on every run (unit-tested)", "module": "dynamic_programming.py", "function": "value_iteration()['contraction_ratios']", "screen": "DP Lab", "status": "DEMONSTRATED"},
    {"unit": 4, "concept": "Efficiency of DP", "project_use": "Sweeps × |S|²|A| operations and wall-clock time reported", "module": "dynamic_programming.py", "function": "seconds / iterations fields", "screen": "DP Lab", "status": "IMPLEMENTED"},
    {"unit": 4, "concept": "Notes' DP examples (dice game V(IN)=10.5, 4×4 gridworld)", "project_use": "The same DP code solves them in the unit tests and the Notes Numericals screen", "module": "tests/test_dynamic_programming.py", "function": "dice_game, gridworld", "screen": "Academic › Notes Numericals", "status": "IMPLEMENTED"},
    # ------------------------------------------------------------------ UNIT 5
    {"unit": 5, "concept": "Monte Carlo prediction from complete episodes", "project_use": "Estimate V_π of a session policy from simulated sessions; MAE vs DP over episode count", "module": "monte_carlo.py", "function": "mc_prediction_run", "screen": "RL Lab › Monte Carlo Lab", "status": "IMPLEMENTED"},
    {"unit": 5, "concept": "First-visit MC", "project_use": "Side-by-side with every-visit", "module": "monte_carlo.py", "function": "first_visit_prediction", "screen": "Monte Carlo Lab", "status": "IMPLEMENTED"},
    {"unit": 5, "concept": "Every-visit MC", "project_use": "Side-by-side with first-visit", "module": "monte_carlo.py", "function": "every_visit_prediction", "screen": "Monte Carlo Lab", "status": "IMPLEMENTED"},
    {"unit": 5, "concept": "MC estimation of action values Q(s,a)", "project_use": "Q(s,a) and visit counts per state-action pair", "module": "monte_carlo.py", "function": "mc_action_values, mc_control", "screen": "Monte Carlo Lab", "status": "IMPLEMENTED"},
    {"unit": 5, "concept": "MC control with exploring starts (MC ES)", "project_use": "Simulator can start any (s, a) pair", "module": "monte_carlo.py", "function": "mc_control('es')", "screen": "Monte Carlo Lab › Control", "status": "IMPLEMENTED"},
    {"unit": 5, "concept": "On-policy MC control with ε-soft policies", "project_use": "Learned session policy compared with DP and Q-learning", "module": "monte_carlo.py", "function": "mc_control('on_policy')", "screen": "Monte Carlo Lab, Algorithm Comparison", "status": "IMPLEMENTED"},
    {"unit": 5, "concept": "Off-policy prediction: target vs behaviour policy, coverage", "project_use": "Estimate a NEW matchmaking policy's value from sessions logged under the OLD policy", "module": "monte_carlo.py", "function": "off_policy_prediction, coverage_check", "screen": "Monte Carlo Lab › Off-policy", "status": "IMPLEMENTED"},
    {"unit": 5, "concept": "Importance-sampling ratio ρ_{t:T−1}", "project_use": "Per-step π/b ratios shown for real logged episodes", "module": "monte_carlo.py", "function": "off_policy_prediction", "screen": "Monte Carlo Lab › Off-policy", "status": "IMPLEMENTED"},
    {"unit": 5, "concept": "Ordinary and weighted importance sampling", "project_use": "Both estimators with effective sample size diagnostics", "module": "monte_carlo.py", "function": "off_policy_prediction", "screen": "Monte Carlo Lab › Off-policy", "status": "IMPLEMENTED"},
    {"unit": 5, "concept": "Off-policy MC control", "project_use": "Weighted-IS control (Sutton §5.7) with ε-soft behaviour", "module": "monte_carlo.py", "function": "mc_control('off_policy')", "screen": "Monte Carlo Lab › Control", "status": "IMPLEMENTED"},
    {"unit": 5, "concept": "Importance sampling on truncated returns", "project_use": "Discounting-aware IS on flat partial returns + per-decision IS (variance reduction)", "module": "monte_carlo.py", "function": "off_policy_prediction (J_discounting_aware_*, J_per_decision_is)", "screen": "Monte Carlo Lab › Off-policy", "status": "IMPLEMENTED"},
    {"unit": 5, "concept": "Notes' MC example (A/B episodes: FV 8.25/7.8, EV 8/4)", "project_use": "Recomputed by the same first/every-visit code", "module": "tests/test_monte_carlo.py", "function": "first_visit_prediction", "screen": "Academic › Notes Numericals", "status": "IMPLEMENTED"},
    # --------------------------------------------------------- beyond Units 1–5
    {"unit": 0, "concept": "Tabular Q-learning (TD control)", "project_use": "Original Clutch IQ optimiser — kept as the model-free baseline, with TD target/error explainer and Training Lab", "module": "q_learning.py", "function": "train, td_update", "screen": "Control Room › Q-Learning", "status": "IMPLEMENTED"},
    {"unit": 0, "concept": "Deep Q-Network (neural approximator, replay, target net)", "project_use": "Not implemented — unnecessary for 48 states and not in Units 1–5; described as a future extension", "module": "—", "function": "—", "screen": "Academic › Limitations", "status": "NOT COVERED"},
]

UNIT_TITLES = {
    1: "Introduction to RL",
    2: "Multi-armed bandits",
    3: "Markov process & optimality",
    4: "Dynamic programming",
    5: "Monte Carlo methods",
    0: "Beyond Units 1–5 (syllabus overview topic)",
}


def coverage() -> dict:
    counts = {}
    for row in COVERAGE:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    return {"rows": COVERAGE, "unit_titles": UNIT_TITLES, "status_counts": counts}


def to_markdown() -> str:
    out = []
    for unit in (1, 2, 3, 4, 5, 0):
        rows = [r for r in COVERAGE if r["unit"] == unit]
        title = f"Unit {unit} — {UNIT_TITLES[unit]}" if unit else UNIT_TITLES[0]
        out.append(f"\n## {title}\n")
        out.append("| Concept | Project use | Module / function | Demo screen | Status |")
        out.append("|---|---|---|---|---|")
        for r in rows:
            out.append(f"| {r['concept']} | {r['project_use']} | `{r['module']}` — {r['function']} | {r['screen']} | **{r['status']}** |")
    return "\n".join(out)
