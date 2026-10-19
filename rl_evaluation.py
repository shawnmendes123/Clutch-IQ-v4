"""
rl_evaluation.py — common, seeded evaluation utilities for sequential RL policies.

All sequential algorithms (DP, MC control, Q-learning, baselines) are
evaluated in the SAME simulated environment with the SAME episode seeds
(common random numbers): episode i of every policy uses derive_seed(seed, i)
for the environment stream, so differences between policies are not luck of
the draw. Bandit algorithms are deliberately NOT evaluated here — a one-step
bandit and a sequential MDP controller optimise different objectives.
"""
from __future__ import annotations

import math
import time

import numpy as np

import rl_core as core
import mdp_model as mm
import dynamic_programming as dp


def _ci95(x: np.ndarray) -> float:
    return float(1.96 * x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else 0.0


def evaluate_policy(policy, n_episodes: int = 1000, seed: int = 2026, gamma: float = 0.9,
                    start_dist: str = "session_start", keep_raw: bool = False) -> dict:
    ret, dret, length, close, rage, bored, natural, mean_c = [], [], [], [], [], [], [], []
    for i in range(n_episodes):
        ep = core.generate_episode(policy, core.derive_seed(seed, i), start_dist=start_dist, record_info=True)
        R = [t.reward for t in ep.transitions]
        ret.append(sum(R))
        dret.append(core.discounted_return(R, gamma))
        length.append(len(R))
        cs = [t.info["closeness"] for t in ep.transitions]
        close.append(np.mean([c >= 0.8 for c in cs]))
        mean_c.append(np.mean(cs))
        last = ep.transitions[-1].info
        rage.append(1.0 if last["rage_quit"] else 0.0)
        natural.append(1.0 if last["natural_end"] else 0.0)
        bored.append(1.0 if (not last["rage_quit"] and not last["natural_end"]) else 0.0)

    def summ(x):
        a = np.asarray(x, dtype=float)
        return {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0, "ci95": _ci95(a)}

    out = {"n_episodes": n_episodes, "seed": seed, "gamma": gamma, "start_dist": start_dist,
           "episodic_return": summ(ret), "discounted_return": summ(dret), "session_length": summ(length),
           "close_match_rate": summ(close), "mean_closeness": summ(mean_c), "rage_quit_rate": summ(rage),
           "boredom_quit_rate": summ(bored), "natural_end_rate": summ(natural), "provenance": "SIMULATED"}
    if keep_raw:
        out["_raw_discounted"] = np.asarray(dret)
        out["_raw_return"] = np.asarray(ret)
    return out


def agreement(a, b) -> float:
    a = np.asarray(a)[:core.N_STATES]
    b = np.asarray(b)[:core.N_STATES]
    return float(np.mean(a == b))


def disagreement_analysis(actions, reference_actions, Q_ref: np.ndarray) -> list[dict]:
    """For each state where a policy disagrees with the reference, report the
    reference action gap q*(s, a_ref) - q*(s, a_policy): small gaps mean the
    two actions are nearly equally good (a near-tie), large gaps mean a real error."""
    out = []
    for s in range(core.N_STATES):
        a, r = int(actions[s]), int(reference_actions[s])
        if a != r:
            out.append({"state": s, "label": core.state_label(s), "short": core.state_short(s),
                        "action": core.ACTION_NAMES[a], "reference_action": core.ACTION_NAMES[r],
                        "q_gap": float(Q_ref[s, r] - Q_ref[s, a])})
    return sorted(out, key=lambda d: -d["q_gap"])


def model_value(model: mm.FiniteMDP, actions, gamma: float, start_dist: str = "session_start") -> float:
    acts = np.array(list(actions)[:core.N_STATES] + [1])
    V = dp.policy_evaluation(model, acts, gamma, 1e-8, 2000)["V"]
    d0 = np.array(core.start_distribution(start_dist) + [0.0])
    return float(d0 @ V)


def trajectory(policy, seed: int, gamma: float, start_state: int | None = None,
               start_dist: str = "session_start", Q: np.ndarray | None = None) -> dict:
    """Unit 1 interaction loop s0 -> a0 -> r1 -> s1 -> ... with discounted
    contributions gamma^t r_{t+1}, cumulative reward and return G_0."""
    ep = core.generate_episode(policy, seed, start_state=start_state, start_dist=start_dist, record_info=True)
    rows = []
    cum = 0.0
    disc_cum = 0.0
    R = [t.reward for t in ep.transitions]
    Gt = core.returns_from_rewards(R, gamma)
    for t, tr in enumerate(ep.transitions):
        cum += tr.reward
        contrib = (gamma ** t) * tr.reward
        disc_cum += contrib
        row = {"t": t, "state": tr.state, "state_label": core.state_label(tr.state), "state_short": core.state_short(tr.state),
               "action": tr.action, "action_name": core.ACTION_NAMES[tr.action], "pi": policy[tr.state],
               "reward": tr.reward, "next_state": tr.next_state, "next_label": core.state_label(tr.next_state),
               "done": tr.done, "discount": gamma ** t, "discounted_contribution": contrib,
               "cumulative_reward": cum, "discounted_cumulative": disc_cum, "G_t": Gt[t], "info": tr.info}
        if Q is not None:
            row["q"] = [float(x) for x in Q[tr.state]]
        rows.append(row)
    end = "rage quit" if ep.transitions[-1].info["rage_quit"] else ("natural end (exhausted)" if ep.transitions[-1].info["natural_end"] else "boredom quit")
    return {"steps": rows, "G0": Gt[0] if Gt else 0.0, "total_reward": cum, "length": len(rows),
            "terminated_by": end, "gamma": gamma, "seed": seed, "provenance": "SIMULATED"}


def compare_policies(n_episodes: int = 500, seed: int = 2026, gamma: float = 0.9, start_dist: str = "session_start",
                     model_samples: int = 1000, model_seed: int = 7, q_params: dict | None = None,
                     mc_episodes: int = 20000, mc_epsilon: float = 0.1, stability_runs: int = 3,
                     include: list[str] | None = None) -> dict:
    """The Algorithm Comparison Arena. Every learner gets the same budget
    definition it would get in the labs; every policy is evaluated on the same
    seeded episodes."""
    import monte_carlo as mc
    import q_learning as ql
    include = include or ["random", "keep", "heuristic", "dp_value_iteration", "dp_policy_iteration",
                          "mc_control", "q_learning"]
    model = mm.get_model(model_samples, model_seed, gamma)
    t_model = model.meta.get("build_seconds", 0.0)
    vi = dp.value_iteration(model, gamma, 1e-8)
    ref = vi["actions"][:core.N_STATES]
    entries = {}

    def add(key, name, family, actions, policy, seconds, convergence, stability, provenance, notes=""):
        entries[key] = {"key": key, "name": name, "family": family, "actions": [int(a) for a in actions][:core.N_STATES],
                        "policy_matrix": policy, "compute_seconds": seconds, "convergence": convergence,
                        "stability": stability, "provenance": provenance, "notes": notes}

    if "random" in include:
        add("random", "Random baseline", "baseline", core.policy_actions(core.uniform_policy()), core.uniform_policy(),
            0.0, "n/a", None, "SIMULATED", "uniform pi(a|s)=1/3 (stochastic)")
    if "keep" in include:
        add("keep", "Always KEEP (v3-style rank-only matchmaking)", "baseline", [1] * core.N_STATES, core.constant_policy(1),
            0.0, "n/a", 1.0, "SIMULATED", "ignores fatigue/tilt/streak")
    if "heuristic" in include:
        hp = core.heuristic_policy()
        add("heuristic", "Manual rule (designer heuristic)", "baseline", core.policy_actions(hp), hp, 0.0, "n/a", 1.0,
            "SIMULATED", "LOWER if tilted or losing; RAISE if calm, winning and fresh")
    if "dp_value_iteration" in include:
        stab = (_stability([dp.value_iteration(mm.get_model(model_samples, model_seed + j, gamma), gamma, 1e-8)["actions"][:core.N_STATES]
                            for j in range(stability_runs)]) if stability_runs > 1 else None)
        add("dp_value_iteration", "Dynamic Programming — Value Iteration", "model-based", ref,
            core.deterministic_policy(list(ref)), t_model + vi["seconds"],
            f"{vi['iterations']} sweeps to theta=1e-8", stab, "MODEL ESTIMATE",
            f"plans on the simulation-estimated model ({model_samples} samples/pair); time includes model estimation")
    if "dp_policy_iteration" in include:
        pi_res = dp.policy_iteration(model, gamma, 1e-8)
        add("dp_policy_iteration", "Dynamic Programming — Policy Iteration", "model-based", pi_res["actions"][:core.N_STATES],
            core.deterministic_policy(list(pi_res["actions"])), t_model + pi_res["seconds"],
            f"{pi_res['improvement_steps']} improvements / {pi_res['evaluation_sweeps_total']} eval sweeps", None,
            "MODEL ESTIMATE", "should match value iteration up to exact ties")
    if "mc_control" in include:
        runs = [mc.mc_control("on_policy", mc_episodes, gamma, mc_epsilon, seed=7 + j) for j in range(stability_runs)]
        r0 = runs[0]
        add("mc_control", f"Monte Carlo Control (on-policy, eps={mc_epsilon})", "model-free", r0["actions"],
            core.deterministic_policy(r0["actions"]), r0["seconds"], f"{mc_episodes} episodes",
            _stability([r["actions"] for r in runs]) if len(runs) > 1 else None, "SIMULATED", "greedy policy w.r.t. learned Q; learned from complete episodes")
    if "q_learning" in include:
        qp = dict(ql.DEFAULT_PARAMS, **(q_params or {}), gamma=gamma)
        runs = []
        for j in range(stability_runs):
            p = dict(qp, seed=qp["seed"] + j)
            if j == 0 and p == ql.DEFAULT_PARAMS:
                # identical to retraining with the documented defaults (deterministic) -> reuse production table
                Qp, meta = ql.load_production()
                runs.append({"actions": core.greedy_from_q(Qp[:core.N_STATES].tolist()),
                             "seconds": meta.get("trained_seconds", 0.0)})
            else:
                runs.append(ql.train(**p))
        r0 = runs[0]
        add("q_learning", "Tabular Q-Learning", "model-free", r0["actions"], core.deterministic_policy(r0["actions"]),
            r0["seconds"], f"{qp['episodes']} episodes, alpha={qp['alpha']}",
            _stability([r["actions"] for r in runs]) if len(runs) > 1 else None,
            "SIMULATED", "off-policy TD control, epsilon-greedy behaviour")

    # ---- evaluation on identical seeded episodes
    raw = {}
    for key, e in entries.items():
        t0 = time.perf_counter()
        ev = evaluate_policy(e["policy_matrix"], n_episodes, seed, gamma, start_dist, keep_raw=True)
        raw[key] = ev.pop("_raw_discounted")
        ev.pop("_raw_return", None)
        e["evaluation"] = ev
        e["evaluation_seconds"] = time.perf_counter() - t0
        e["agreement_with_dp"] = agreement(e["actions"], ref)
        e["model_value_J"] = model_value(model, e["actions"], gamma, start_dist) if key != "random" else \
            float(np.array(core.start_distribution(start_dist) + [0.0]) @ dp.policy_evaluation(model, np.array(core.uniform_policy()), gamma, 1e-8)["V"])
        e["disagreements"] = disagreement_analysis(e["actions"], ref, vi["Q"]) if e["family"] != "baseline" or key == "heuristic" else []
        del e["policy_matrix"]
    # ---- paired differences vs DP optimal (common random numbers)
    if "dp_value_iteration" in raw:
        base = raw["dp_value_iteration"]
        for key, e in entries.items():
            d = raw[key] - base
            e["paired_diff_vs_dp"] = {"mean": float(d.mean()), "ci95": _ci95(d)}
    keys = list(entries.keys())
    matrix = [[agreement(entries[a]["actions"], entries[b]["actions"]) for b in keys] for a in keys]
    return {"policies": [entries[k] for k in keys], "agreement_matrix": {"keys": keys, "values": matrix},
            "settings": {"n_episodes": n_episodes, "seed": seed, "gamma": gamma, "start_dist": start_dist,
                         "model_samples": model_samples, "model_seed": model_seed, "mc_episodes": mc_episodes,
                         "mc_epsilon": mc_epsilon, "q_params": dict(ql.DEFAULT_PARAMS, **(q_params or {}), gamma=gamma),
                         "stability_runs": stability_runs, "common_random_numbers": True},
            "provenance": "SIMULATED"}


def _stability(action_lists) -> float:
    """Mean pairwise agreement between policies learned with different seeds."""
    if len(action_lists) < 2:
        return 1.0
    vals = []
    for i in range(len(action_lists)):
        for j in range(i + 1, len(action_lists)):
            vals.append(agreement(action_lists[i], action_lists[j]))
    return float(np.mean(vals))
