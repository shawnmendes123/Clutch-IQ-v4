"""
rl_experiments.py — seeded, reproducible experiment suite for Clutch IQ.

    python rl_experiments.py            # full suite (~1-2 min), writes experiments/results.json
    python rl_experiments.py --quick    # smaller budgets for a smoke run

Every number in EXPERIMENTS.md and on the Academic › Experiments screen is
produced by this file. Nothing is hand-typed or cherry-picked: all seeds and
budgets are listed in the output, and multi-seed experiments report mean and
standard deviation across ALL seeds that were run.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import time

import numpy as np

import rl_core as core
import mdp_model as mm
import dynamic_programming as dp
import monte_carlo as mc
import q_learning as ql
import rl_evaluation as ev
import bandits as bd
import notes_examples

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "experiments")
GAMMA = 0.9
D0 = np.array(core.start_distribution("session_start") + [0.0])


def J_of(model, actions, gamma=GAMMA):
    return ev.model_value(model, actions, gamma)


def e1_model_estimation(quick):
    exact = mm.exact_model().with_gamma(GAMMA)
    vi_exact = dp.value_iteration(exact, GAMMA, 1e-10)
    rows = []
    for n in ([100, 300, 1000] if quick else [100, 300, 1000, 3000]):
        m = mm.get_model(n, 7, GAMMA)
        q = mm.model_quality(m)
        vi = dp.value_iteration(m, GAMMA, 1e-10)
        rows.append({"samples_per_pair": n, "max_abs_error": q["max_abs_transition_error"],
                     "mean_abs_error": q["mean_abs_transition_error"], "worst_case_se": q["worst_case_std_error"],
                     "policy_agreement_with_exact_optimum": ev.agreement(vi["actions"], vi_exact["actions"]),
                     "J_of_planned_policy_on_exact_model": J_of(exact, vi["actions"]),
                     "J_exact_optimum": float(D0 @ vi_exact["V"]), "build_seconds": m.meta["build_seconds"]})
    return {"id": "E1", "title": "Simulation-estimated model: accuracy vs samples per (s,a)", "rows": rows}


def e2_dp(quick):
    rows = []
    for g in (0.5, 0.9, 0.99):
        m = mm.get_model(1000, 7, g)
        vi = dp.value_iteration(m, g, 1e-8, 20000)
        pi = dp.policy_iteration(m, g, 1e-8)
        pe = dp.policy_evaluation(m, np.array(core.uniform_policy()), g, 1e-8, 20000)
        rows.append({"gamma": g, "vi_iterations": vi["iterations"], "vi_max_contraction_ratio": vi["max_contraction_ratio"],
                     "pi_improvement_steps": pi["improvement_steps"], "pi_eval_sweeps": pi["evaluation_sweeps_total"],
                     "pi_equals_vi": ev.agreement(pi["actions"], vi["actions"]),
                     "random_policy_eval_sweeps": pe["iterations"],
                     "closed_form_gap": float(np.max(np.abs(m.mrp_closed_form(np.array(core.uniform_policy()), g) - pe["V"]))),
                     "bellman_residual": dp.bellman_residual(m, vi["V"], g),
                     "vi_seconds": vi["seconds"], "pi_seconds": pi["seconds"]})
    return {"id": "E2", "title": "Dynamic programming convergence (estimated model, 1000 samples/pair)", "rows": rows}


def e3_gamma_sweep(quick):
    rows = []
    for g in (0.0, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99):
        m = mm.get_model(1000, 7, g)
        vi = dp.value_iteration(m, g, 1e-10, 20000)
        acts = vi["actions"][:core.N_STATES]
        e = ev.evaluate_policy(core.deterministic_policy(list(acts)), 300 if quick else 1000, 2026, 0.9, "session_start")
        rows.append({"gamma": g, "n_lower": int((acts == 0).sum()), "n_keep": int((acts == 1).sum()), "n_raise": int((acts == 2).sum()),
                     "actions": [int(a) for a in acts],
                     "sim_session_length": e["session_length"]["mean"], "sim_rage_quit_rate": e["rage_quit_rate"]["mean"],
                     "sim_close_match_rate": e["close_match_rate"]["mean"], "sim_undiscounted_return": e["episodic_return"]["mean"]})
    return {"id": "E3", "title": "Effect of the discount factor on the optimal session policy", "rows": rows,
            "note": "Simulated rates are measured with the same 1000 seeded session-start episodes for every gamma."}


def e4_mc_prediction(quick):
    m = mm.get_model(1000, 7, GAMMA)
    pol = core.heuristic_policy()
    ref = dp.policy_evaluation(m, np.array(pol), GAMMA, 1e-10)["V"]
    seeds = [11, 12, 13] if quick else [11, 12, 13, 14, 15]
    n = 5000 if quick else 20000
    curves = []
    for sd in seeds:
        r = mc.mc_prediction_run(pol, n, GAMMA, sd, "uniform", ref)
        curves.append(r["curve"])
    eps = [c["episodes"] for c in curves[0]]
    agg = []
    for i, e in enumerate(eps):
        fv = [c[i]["mae_first_visit"] for c in curves]
        evv = [c[i]["mae_every_visit"] for c in curves]
        cov = [c[i]["coverage"] for c in curves]
        agg.append({"episodes": e, "mae_first_visit_mean": float(np.mean(fv)), "mae_first_visit_std": float(np.std(fv, ddof=1)),
                    "mae_every_visit_mean": float(np.mean(evv)), "mae_every_visit_std": float(np.std(evv, ddof=1)),
                    "coverage_mean": float(np.mean(cov))})
    return {"id": "E4", "title": "MC prediction error vs episodes (heuristic policy, reference = DP policy evaluation)",
            "seeds": seeds, "rows": agg}


def e5_control(quick):
    m = mm.get_model(1000, 7, GAMMA)
    vi = dp.value_iteration(m, GAMMA, 1e-10)
    j_star = float(D0 @ vi["V"])
    out = []
    n_mc = 5000 if quick else 20000
    for variant in ("es", "on_policy", "off_policy"):
        js, ag = [], []
        for sd in ([7, 8] if quick else [7, 8, 9]):
            r = mc.mc_control(variant, n_mc, GAMMA, 0.1, sd)
            js.append(J_of(m, r["actions"])); ag.append(ev.agreement(r["actions"], vi["actions"]))
        out.append({"algorithm": f"MC control ({variant})", "episodes": n_mc, "J_mean": float(np.mean(js)), "J_std": float(np.std(js, ddof=1)),
                    "agreement_mean": float(np.mean(ag)), "seeds": len(js)})
    js, ag, secs = [], [], []
    for sd in ([7, 8] if quick else [7, 8, 9, 10, 11]):
        r = ql.train(**dict(ql.DEFAULT_PARAMS, seed=sd, episodes=10000 if quick else ql.DEFAULT_PARAMS["episodes"]))
        js.append(J_of(m, r["actions"])); ag.append(ev.agreement(r["actions"], vi["actions"])); secs.append(r["seconds"])
    out.append({"algorithm": "Tabular Q-learning", "episodes": 10000 if quick else ql.DEFAULT_PARAMS["episodes"],
                "J_mean": float(np.mean(js)), "J_std": float(np.std(js, ddof=1)), "agreement_mean": float(np.mean(ag)),
                "seeds": len(js), "train_seconds_mean": float(np.mean(secs))})
    return {"id": "E5", "title": "Model-free control vs the DP optimum (J = expected discounted return from session start, on the estimated model)",
            "J_star": j_star, "J_keep": J_of(m, [1] * 48), "J_heuristic": J_of(m, core.policy_actions(core.heuristic_policy())),
            "rows": out}


def e6_arena(quick):
    res = ev.compare_policies(n_episodes=300 if quick else 1000, seed=2026, gamma=GAMMA, mc_episodes=5000 if quick else 20000,
                              stability_runs=2 if quick else 3)
    rows = []
    for p in res["policies"]:
        e = p["evaluation"]
        rows.append({"policy": p["name"], "family": p["family"], "discounted_return": e["discounted_return"],
                     "episodic_return": e["episodic_return"], "session_length": e["session_length"],
                     "rage_quit_rate": e["rage_quit_rate"], "close_match_rate": e["close_match_rate"],
                     "boredom_quit_rate": e["boredom_quit_rate"], "agreement_with_dp": p["agreement_with_dp"],
                     "paired_diff_vs_dp": p["paired_diff_vs_dp"], "model_value_J": p["model_value_J"],
                     "stability": p["stability"], "compute_seconds": p["compute_seconds"], "convergence": p["convergence"],
                     "actions": p["actions"]})
    return {"id": "E6", "title": "Algorithm comparison arena (identical seeded episodes, common random numbers)",
            "settings": res["settings"], "rows": rows}


def e7_bandits(quick):
    import rl_api
    arms, src = rl_api.build_arms(rl_api.ArmsSpec())
    means = [a["q_star"] for a in arms]
    runs = 50 if quick else 200
    algs = ["random", "greedy", "explore_first", "epsilon_greedy", "optimistic", "ucb"]
    st = bd.run_arena(algs, means, 1000, runs, 1, epsilon=0.1, c=1.414, q0=1.0, explore_rounds=50)
    rows = [{"algorithm": v["name"], "key": k, "final_regret": v["final_regret"], "final_reward": v["final_reward"],
             "final_reward_ci95": v["final_reward_ci95"], "pct_optimal_last_10pct": v["final_pct_optimal"],
             "curve_rounds": v["rounds_index"], "curve_regret": v["mean_cumulative_regret"], "curve_pct_optimal": v["pct_optimal"]}
            for k, v in st["results"].items()]
    ns = []
    for label, alpha in (("sample average (1/n)", None), ("constant α = 0.1", 0.1)):
        r = bd.run_arena(["epsilon_greedy"], means, 2000, runs, 3, epsilon=0.1, alpha=alpha, nonstationary="patch")
        v = r["results"]["epsilon_greedy"]
        idx = v["rounds_index"]
        half = next(i for i, x in enumerate(idx) if x >= 1000)
        ns.append({"update": label, "final_regret": v["final_regret"],
                   "regret_after_patch": v["mean_cumulative_regret"][-1] - v["mean_cumulative_regret"][half],
                   "pct_optimal_last_10pct": v["final_pct_optimal"]})
    eps_sweep = []
    for e in (0.0, 0.01, 0.05, 0.1, 0.2, 0.5):
        r = bd.run_arena(["epsilon_greedy"], means, 1000, runs, 1, epsilon=e)["results"]["epsilon_greedy"]
        eps_sweep.append({"epsilon": e, "final_regret": r["final_regret"], "pct_optimal_last_10pct": r["final_pct_optimal"]})
    return {"id": "E7", "title": "Bandit Arena: map recommendation (10 arms, Bernoulli satisfaction, 1000 rounds)",
            "arms": arms, "arm_source": src, "runs": runs, "stationary": rows, "nonstationary_patch": ns, "epsilon_sweep": eps_sweep}


def e8_off_policy(quick):
    m = mm.get_model(1000, 7, GAMMA)
    vi = dp.value_iteration(m, GAMMA, 1e-10)
    heur = core.policy_actions(core.heuristic_policy())
    presets = [
        ("Small change: heuristic ε=0.1 from logs of heuristic ε=0.3", core.epsilon_soft(heur, 0.1), core.epsilon_soft(heur, 0.3)),
        ("Large change: DP-optimal ε=0.1 from logs of heuristic ε=0.3", core.epsilon_soft(list(vi["actions"]), 0.1), core.epsilon_soft(heur, 0.3)),
        ("Uniform-random logs: heuristic ε=0.1", core.epsilon_soft(heur, 0.1), core.uniform_policy()),
    ]
    rows = []
    for name, t, b in presets:
        truth = float(D0 @ dp.policy_evaluation(m, np.array(t), GAMMA, 1e-10)["V"])
        r = mc.off_policy_prediction(t, b, 3000 if quick else 10000, GAMMA, 4, "session_start", truth)
        rows.append({"scenario": name, "truth_dp": truth, "behaviour_value": r["J_behaviour_on_policy"],
                     "ordinary_is": r["J_ordinary_is"], "weighted_is": r["J_weighted_is"], "per_decision_is": r["J_per_decision_is"],
                     "discounting_aware_wis": r["J_discounting_aware_wis"],
                     "ess": r["rho_stats"]["effective_sample_size"], "reliability": r["reliability"], "episodes": r["n_episodes"]})
    return {"id": "E8", "title": "Off-policy evaluation by importance sampling (session-start value J)", "rows": rows}


def run(quick=False):
    t0 = time.time()
    results = {"generated_at_unix": int(t0), "quick": quick, "python": platform.python_version(),
               "gamma": GAMMA, "provenance": "SIMULATED / MODEL ESTIMATE — no real-player data is used in these experiments"}
    exps = [e1_model_estimation, e2_dp, e3_gamma_sweep, e4_mc_prediction, e5_control, e6_arena, e7_bandits, e8_off_policy]
    results["experiments"] = []
    for fn in exps:
        t = time.time()
        r = fn(quick)
        r["seconds"] = round(time.time() - t, 2)
        results["experiments"].append(r)
        print(f"{r['id']} {r['title'][:60]:60s} {r['seconds']:6.1f}s")
    nums = notes_examples.all_numericals()
    results["notes_numericals"] = {"checked": sum(1 for n in nums if n["match"] is not None),
                                   "matched": sum(1 for n in nums if n["match"]),
                                   "ids": [n["id"] for n in nums]}
    results["total_seconds"] = round(time.time() - t0, 1)
    os.makedirs(OUT_DIR, exist_ok=True)
    import rl_api
    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump(rl_api.J(results, 5), f, indent=1)
    print(f"wrote experiments/results.json in {results['total_seconds']}s")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    run(ap.parse_args().quick)
