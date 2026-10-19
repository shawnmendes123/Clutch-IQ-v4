"""
rl_api.py — FastAPI router exposing the RL engine (mounted by backend_app.py).

Every response carries a `provenance` label (REAL DATA / SIMULATED / MODEL
ESTIMATE / DERIVED FEATURE / FALLBACK/PROXY) so the UI can show where each
number came from. All endpoints are deterministic for a given seed.
"""
from __future__ import annotations

import io
import math
import time
from functools import lru_cache
from typing import Literal, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

import rl_core as core
import mdp_model as mm
import dynamic_programming as dp
import monte_carlo as mc
import q_learning as ql
import rl_evaluation as ev
import bandits as bd
import notes_examples
import syllabus

router = APIRouter()

PolicyName = Literal["dp_optimal", "q_learning", "heuristic", "keep", "lower", "raise", "random"]

# shared stores (in-memory, like the rest of the backend)
EPISODE_LOGS: dict[str, dict] = {}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def J(x, nd: int = 6):
    """Recursively convert numpy / floats into JSON-safe, rounded values."""
    if isinstance(x, dict):
        return {str(k): J(v, nd) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [J(v, nd) for v in x]
    if isinstance(x, np.ndarray):
        return J(x.tolist(), nd)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (float, np.floating)):
        f = float(x)
        if math.isnan(f):
            return None
        if math.isinf(f):
            return "inf" if f > 0 else "-inf"
        return round(f, nd)
    return x


@lru_cache(maxsize=32)
def dp_solution(gamma: float, samples: int, seed: int):
    model = mm.get_model(samples, seed, gamma)
    vi = dp.value_iteration(model, gamma, 1e-10, 5000)
    return model, vi


def resolve_policy(name: str, gamma: float = 0.9, samples: int = 1000, seed: int = 7, epsilon: float = 0.0):
    """Policy name -> 49x3 probability matrix."""
    if name == "dp_optimal":
        _, vi = dp_solution(gamma, samples, seed)
        acts = [int(a) for a in vi["actions"]]
    elif name == "q_learning":
        Q, _ = ql.load_production()
        acts = core.greedy_from_q(Q.tolist())
    elif name == "heuristic":
        acts = core.policy_actions(core.heuristic_policy())
    elif name in ("keep", "lower", "raise"):
        acts = [{"lower": 0, "keep": 1, "raise": 2}[name]] * core.N_STATES_WITH_TERMINAL
    elif name == "random":
        return core.uniform_policy()
    else:
        raise HTTPException(400, f"unknown policy '{name}'")
    if epsilon > 0:
        return core.epsilon_soft(acts, epsilon)
    return core.deterministic_policy(acts)


def parse_state(state: str) -> int:
    try:
        if "-" in state:
            s = core.key_to_state(state)
        else:
            s = int(state)
    except Exception:
        raise HTTPException(400, "state must be an index 0..47 or a 'f-t-k' key such as '2-3-0'")
    if not 0 <= s < core.N_STATES:
        raise HTTPException(400, "state index out of range 0..47")
    return s


def policy_actions_list(pi) -> list[int]:
    return [int(a) for a in core.policy_actions(pi)[:core.N_STATES]]


# --------------------------------------------------------------------------
# Unit 1: formal definition, trajectory
# --------------------------------------------------------------------------
@router.get("/rl/spec")
def rl_spec():
    sp = core.spec()
    sp["states"] = [core.state_info(s) for s in range(core.N_STATES)]
    sp["step_model"] = [[J(core.step_model(s, a), 4) for a in core.ACTIONS] for s in range(core.N_STATES)]
    return J(sp)


class TrajectoryRequest(BaseModel):
    policy: PolicyName = "dp_optimal"
    policy_epsilon: float = Field(0.0, ge=0, le=1)
    seed: int = Field(42, ge=0, le=2**31)
    gamma: float = Field(0.9, ge=0, le=1)
    start_state: Optional[int] = Field(None, ge=0, le=47)
    start_dist: Literal["uniform", "session_start"] = "session_start"


@router.post("/rl/trajectory")
def rl_trajectory(req: TrajectoryRequest):
    pi = resolve_policy(req.policy, req.gamma, epsilon=req.policy_epsilon)
    Q = None
    if req.policy == "q_learning":
        Q, _ = ql.load_production()
    elif req.policy == "dp_optimal":
        _, vi = dp_solution(req.gamma, 1000, 7)
        Q = vi["Q"]
    return J(ev.trajectory(pi, req.seed, req.gamma, req.start_state, req.start_dist, Q))


# --------------------------------------------------------------------------
# Unit 3: MDP explorer
# --------------------------------------------------------------------------
@router.get("/rl/mdp")
def rl_mdp(gamma: float = 0.9, samples: int = 1000, seed: int = 7, include_matrices: bool = True):
    if not (0 <= gamma <= 1) or not (50 <= samples <= 20000):
        raise HTTPException(400, "gamma in [0,1], samples in [50, 20000]")
    model = mm.get_model(samples, seed, gamma)
    out = {"meta": model.meta, "quality_vs_simulator": mm.model_quality(model), "check": model.check(),
           "n_states": core.N_STATES, "terminal": core.TERMINAL, "actions": core.ACTION_NAMES,
           "states": [core.state_info(s) for s in range(core.N_STATES)],
           "provenance": "MODEL ESTIMATE"}
    if include_matrices:
        out["P"] = np.round(model.P, 4)
        out["R"] = np.round(model.R, 4)
        out["r_sa"] = np.round(model.r_sa, 4)
    return J(out, 4)


@router.get("/rl/mdp/state/{state}")
def rl_mdp_state(state: str, gamma: float = 0.9, samples: int = 1000, seed: int = 7,
                 policy: PolicyName = "dp_optimal", policy_epsilon: float = 0.0, top_k: int = 6):
    s = parse_state(state)
    model, vi = dp_solution(gamma, samples, seed)
    pi = np.array(resolve_policy(policy, gamma, samples, seed, policy_epsilon))
    pe = dp.policy_evaluation(model, pi, gamma, 1e-10, 5000)
    Vpi = pe["V"]
    Qpi = model.q_from_v(Vpi, gamma)
    bd_pi = dp.backup_breakdown(model, s, Vpi, gamma, pi=pi, top_k=top_k)
    bd_opt = dp.backup_breakdown(model, s, vi["V"], gamma, top_k=top_k)
    per_action = []
    for a in core.ACTIONS:
        sm = core.step_model(s, a)
        per_action.append({
            "action": a, "name": core.ACTION_NAMES[a], "pi": float(pi[s, a]),
            "expected_reward": float(model.r_sa[s, a]), "q_pi": float(Qpi[s, a]), "q_star": float(vi["Q"][s, a]),
            "next_states": mm.top_transitions(model, s, a, top_k),
            "terminal_probability": float(model.P[s, a, core.TERMINAL]),
            "simulator": {k: round(v, 4) for k, v in sm.items()},
        })
    return J({
        "state": core.state_info(s), "gamma": gamma, "policy": policy, "policy_epsilon": policy_epsilon,
        "V_pi": float(Vpi[s]), "V_star": float(vi["V"][s]),
        "best_action": core.ACTION_NAMES[int(vi["actions"][s])], "best_action_index": int(vi["actions"][s]),
        "per_action": per_action, "bellman_expectation": bd_pi, "bellman_optimality": bd_opt,
        "V_pi_all": Vpi[:core.N_STATES], "V_star_all": vi["V"][:core.N_STATES],
        "policy_actions": policy_actions_list(pi), "optimal_actions": vi["actions"][:core.N_STATES],
        "model_meta": model.meta, "provenance": "MODEL ESTIMATE",
    })


@router.get("/rl/mdp/markov-chain")
def rl_markov_chain(policy: PolicyName = "dp_optimal", gamma: float = 0.9, samples: int = 1000, seed: int = 7,
                    start_dist: Literal["uniform", "session_start"] = "session_start", horizon: int = 20):
    """Unit 3: the Markov chain P_pi induced by a fixed policy, and the state
    distribution d_t = d_0 P_pi^t over the first `horizon` matches."""
    model, _ = dp_solution(gamma, samples, seed)
    pi = np.array(resolve_policy(policy, gamma, samples, seed))
    Ppi, Rpi = model.policy_mrp(pi)
    d = np.array(core.start_distribution(start_dist) + [0.0])
    dists = [d]
    for _ in range(min(60, max(1, horizon))):
        d = d @ Ppi
        dists.append(d)
    alive = [float(1 - x[core.TERMINAL]) for x in dists]
    expected_len = float(sum(alive[:-1]))
    v_closed = model.mrp_closed_form(pi, gamma)
    return J({"P_pi": np.round(Ppi, 4), "R_pi": Rpi, "distributions": np.round(np.array(dists), 5),
              "prob_session_alive": alive, "expected_matches_within_horizon": expected_len,
              "V_closed_form": v_closed, "provenance": "MODEL ESTIMATE"}, 5)


# --------------------------------------------------------------------------
# Unit 4: dynamic programming
# --------------------------------------------------------------------------
class DPRequest(BaseModel):
    gamma: float = Field(0.9, ge=0, le=1)
    theta: float = Field(1e-6, gt=0, le=1)
    max_iter: int = Field(1000, ge=1, le=20000)
    samples: int = Field(1000, ge=50, le=20000)
    seed: int = Field(7, ge=0)
    policy: PolicyName = "random"
    policy_epsilon: float = Field(0.0, ge=0, le=1)
    snapshots: int = Field(25, ge=0, le=200)
    in_place: bool = False
    explain_state: int = Field(42, ge=0, le=47)


def _compute_stats(model, iterations):
    nS, nA = core.N_STATES, core.N_ACTIONS
    return {"backups_per_sweep": nS * nA, "operations_per_sweep": nS * nA * (nS + 1),
            "total_operations": iterations * nS * nA * (nS + 1)}


@router.post("/rl/dp/policy-evaluation")
def rl_policy_evaluation(req: DPRequest):
    model = mm.get_model(req.samples, req.seed, req.gamma)
    pi = np.array(resolve_policy(req.policy, req.gamma, req.samples, req.seed, req.policy_epsilon))
    res = dp.policy_evaluation(model, pi, req.gamma, req.theta, req.max_iter, in_place=req.in_place,
                               record_snapshots=req.snapshots)
    closed = model.mrp_closed_form(pi, req.gamma)   # MRP: v = (I - gamma P_pi)^-1 R_pi
    bd_ = dp.backup_breakdown(model, req.explain_state, res["V"], req.gamma, pi=pi)
    return J({"V": res["V"][:core.N_STATES], "iterations": res["iterations"], "deltas": res["deltas"],
              "converged": res["converged"], "snapshots": [s[:core.N_STATES] for s in res["snapshots"]],
              "seconds": res["seconds"], "V_closed_form_mrp": closed[:core.N_STATES],
              "max_abs_diff_vs_closed_form": float(np.max(np.abs(closed - res["V"]))),
              "bellman_residual": dp.bellman_residual(model, res["V"], req.gamma, pi),
              "policy_actions": policy_actions_list(pi), "policy_matrix": pi[:core.N_STATES],
              "backup_example": bd_, "compute": _compute_stats(model, res["iterations"]),
              "model_meta": model.meta, "provenance": "MODEL ESTIMATE"})


@router.post("/rl/dp/policy-iteration")
def rl_policy_iteration(req: DPRequest):
    model = mm.get_model(req.samples, req.seed, req.gamma)
    res = dp.policy_iteration(model, req.gamma, req.theta, 100, req.max_iter)
    hist = [{"iteration": h["iteration"], "evaluation_sweeps": h["evaluation_sweeps"], "n_changed": h["n_changed"],
             "changed_states": h["changed_states"], "policy": h["policy"][:core.N_STATES], "V": h["V"][:core.N_STATES]}
            for h in res["history"]]
    return J({"V": res["V"][:core.N_STATES], "Q": res["Q"][:core.N_STATES], "actions": res["actions"][:core.N_STATES],
              "policy_stable": res["policy_stable"], "improvement_steps": res["improvement_steps"],
              "evaluation_sweeps_total": res["evaluation_sweeps_total"], "history": hist, "seconds": res["seconds"],
              "initial_policy": "KEEP everywhere", "compute": _compute_stats(model, res["evaluation_sweeps_total"]),
              "model_meta": model.meta, "provenance": "MODEL ESTIMATE"})


@router.post("/rl/dp/value-iteration")
def rl_value_iteration(req: DPRequest):
    model = mm.get_model(req.samples, req.seed, req.gamma)
    res = dp.value_iteration(model, req.gamma, req.theta, req.max_iter, record_snapshots=req.snapshots, in_place=req.in_place)
    q_actions = core.greedy_from_q(ql.load_production()[0].tolist())[:core.N_STATES]
    bd_ = dp.backup_breakdown(model, req.explain_state, res["V"], req.gamma)
    # previous-sweep values for a literal 'one backup' illustration
    prev = res["snapshots"][-2] if len(res["snapshots"]) >= 2 else np.zeros(model.nS)
    return J({"V": res["V"][:core.N_STATES], "Q": res["Q"][:core.N_STATES], "actions": res["actions"][:core.N_STATES],
              "iterations": res["iterations"], "deltas": res["deltas"], "converged": res["converged"],
              "contraction_ratios": res["contraction_ratios"], "max_contraction_ratio": res["max_contraction_ratio"],
              "snapshots": [s[:core.N_STATES] for s in res["snapshots"]], "seconds": res["seconds"],
              "bellman_residual": dp.bellman_residual(model, res["V"], req.gamma),
              "backup_example": bd_, "q_learning_actions": q_actions,
              "agreement_with_q_learning": ev.agreement(res["actions"], q_actions),
              "disagreements_with_q_learning": ev.disagreement_analysis(q_actions, res["actions"], res["Q"]),
              "compute": _compute_stats(model, res["iterations"]), "model_meta": model.meta,
              "provenance": "MODEL ESTIMATE"})


class GammaSweepRequest(BaseModel):
    gammas: list[float] = Field(default_factory=lambda: [0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99])
    samples: int = Field(1000, ge=50, le=20000)
    seed: int = 7


@router.post("/rl/dp/gamma-sweep")
def rl_gamma_sweep(req: GammaSweepRequest):
    """Unit 1: how the discount factor changes the optimal session policy."""
    rows = []
    for g in req.gammas:
        if not 0 <= g <= 1:
            raise HTTPException(400, "gammas must be in [0,1]")
        model, vi = dp_solution(float(g), req.samples, req.seed)
        acts = vi["actions"][:core.N_STATES]
        ev_ = ev.evaluate_policy(core.deterministic_policy(list(acts)), 300, 2026, 0.9, "session_start")
        rows.append({"gamma": g, "actions": acts, "counts": np.bincount(acts, minlength=3),
                     "iterations": vi["iterations"],
                     "V_session_start": float(np.array(core.start_distribution("session_start") + [0]) @ vi["V"]),
                     "sim_rage_quit_rate": ev_["rage_quit_rate"]["mean"], "sim_session_length": ev_["session_length"]["mean"],
                     "sim_close_match_rate": ev_["close_match_rate"]["mean"]})
    return J({"rows": rows, "provenance": "MODEL ESTIMATE (policies) / SIMULATED (rates)"})


# --------------------------------------------------------------------------
# Unit 5: Monte Carlo
# --------------------------------------------------------------------------
class MCPredictRequest(BaseModel):
    policy: PolicyName = "heuristic"
    policy_epsilon: float = Field(0.0, ge=0, le=1)
    episodes: int = Field(3000, ge=10, le=50000)
    gamma: float = Field(0.9, ge=0, le=1)
    seed: int = Field(11, ge=0)
    start_dist: Literal["uniform", "session_start"] = "uniform"
    samples: int = Field(1000, ge=50, le=20000)
    model_seed: int = 7
    episodes_id: Optional[str] = None


@router.post("/rl/mc/predict")
def rl_mc_predict(req: MCPredictRequest):
    model = mm.get_model(req.samples, req.model_seed, req.gamma)
    if req.episodes_id:
        log = EPISODE_LOGS.get(req.episodes_id)
        if not log:
            raise HTTPException(404, "unknown episodes_id")
        eps = log["episodes"]
        fv = mc.first_visit_prediction(eps, req.gamma, core.N_STATES_WITH_TERMINAL)
        evv = mc.every_visit_prediction(eps, req.gamma, core.N_STATES_WITH_TERMINAL)
        return J({"V_first_visit": fv["V"][:core.N_STATES], "V_every_visit": evv["V"][:core.N_STATES],
                  "N_first_visit": fv["N"][:core.N_STATES], "N_every_visit": evv["N"][:core.N_STATES],
                  "n_episodes": len(eps), "provenance": log["provenance"], "notes": log["notes"]})
    pi = resolve_policy(req.policy, req.gamma, req.samples, req.model_seed, req.policy_epsilon)
    ref = dp.policy_evaluation(model, np.array(pi), req.gamma, 1e-10, 5000)["V"]
    res = mc.mc_prediction_run(pi, req.episodes, req.gamma, req.seed, req.start_dist, ref)
    return J({"V_first_visit": res["V_first_visit"][:core.N_STATES], "V_every_visit": res["V_every_visit"][:core.N_STATES],
              "N_first_visit": res["N_first_visit"][:core.N_STATES], "N_every_visit": res["N_every_visit"][:core.N_STATES],
              "V_dp_reference": ref[:core.N_STATES], "curve": res["curve"], "sample_episode": res["sample_episode"],
              "mean_return": res["mean_return"], "std_return": res["std_return"], "mean_length": res["mean_length"],
              "n_episodes": res["n_episodes"], "seconds": res["seconds"], "policy_actions": policy_actions_list(pi),
              "reference": "DP policy evaluation on the simulation-estimated model", "provenance": "SIMULATED"})


class MCControlRequest(BaseModel):
    variant: Literal["es", "on_policy", "off_policy"] = "on_policy"
    episodes: int = Field(10000, ge=100, le=60000)
    gamma: float = Field(0.9, ge=0, le=1)
    epsilon: float = Field(0.1, ge=0.0, le=1)
    behaviour_epsilon: float = Field(0.3, gt=0.0, le=1)
    seed: int = Field(7, ge=0)
    samples: int = Field(1000, ge=50, le=20000)
    model_seed: int = 7


@router.post("/rl/mc/control")
def rl_mc_control(req: MCControlRequest):
    model, vi = dp_solution(req.gamma, req.samples, req.model_seed)
    res = mc.mc_control(req.variant, req.episodes, req.gamma, req.epsilon, req.seed, "uniform", vi["actions"], model,
                        behaviour_epsilon=req.behaviour_epsilon)
    acts = res["actions"]
    return J({"variant": req.variant, "Q": res["Q"][:core.N_STATES], "N": res["N"][:core.N_STATES], "actions": acts,
              "history": res["history"], "seconds": res["seconds"], "unvisited_pairs": res["unvisited_pairs"],
              "dp_actions": vi["actions"][:core.N_STATES], "agreement_with_dp": ev.agreement(acts, vi["actions"]),
              "disagreements": ev.disagreement_analysis(acts, vi["actions"], vi["Q"]),
              "J_learned": ev.model_value(model, acts, req.gamma), "J_dp_optimal": ev.model_value(model, vi["actions"], req.gamma),
              "q_mae_vs_dp": float(np.mean(np.abs(res["Q"][:core.N_STATES] - vi["Q"][:core.N_STATES]))),
              "transitions_used": res.get("transitions_used"), "provenance": "SIMULATED"})


class OffPolicyRequest(BaseModel):
    target: PolicyName = "dp_optimal"
    target_epsilon: float = Field(0.1, ge=0, le=1)
    behaviour: PolicyName = "heuristic"
    behaviour_epsilon: float = Field(0.3, ge=0, le=1)
    episodes: int = Field(5000, ge=10, le=50000)
    gamma: float = Field(0.9, ge=0, le=1)
    seed: int = Field(4, ge=0)
    start_dist: Literal["uniform", "session_start"] = "session_start"
    samples: int = Field(1000, ge=50, le=20000)
    model_seed: int = 7
    episodes_id: Optional[str] = None


@router.post("/rl/mc/off-policy")
def rl_mc_off_policy(req: OffPolicyRequest):
    model = mm.get_model(req.samples, req.model_seed, req.gamma)
    target = resolve_policy(req.target, req.gamma, req.samples, req.model_seed, req.target_epsilon)
    behaviour = resolve_policy(req.behaviour, req.gamma, req.samples, req.model_seed, req.behaviour_epsilon)
    logged = None
    if req.episodes_id:
        log = EPISODE_LOGS.get(req.episodes_id)
        if not log:
            raise HTTPException(404, "unknown episodes_id")
        if not log["has_behaviour_probs"]:
            raise HTTPException(400, "These logged episodes have no behaviour-policy probabilities. Importance sampling "
                                     "needs b(a|s) for every logged action — Clutch IQ will not fabricate them.")
        logged = log["core_episodes"]
        behaviour = log["behaviour_matrix"]
    d0 = np.array(core.start_distribution(req.start_dist) + [0.0])
    Vt = dp.policy_evaluation(model, np.array(target), req.gamma, 1e-10, 5000)["V"]
    Vb = dp.policy_evaluation(model, np.array(behaviour), req.gamma, 1e-10, 5000)["V"]
    n = len(logged) if logged is not None else req.episodes
    res = mc.off_policy_prediction(target, behaviour, n, req.gamma, req.seed, req.start_dist, float(d0 @ Vt), Vt,
                                   logged_episodes=logged)
    if "error" in res:
        raise HTTPException(400, res["error"])
    # ground truth by on-policy MC of the target (what deploying it would give)
    onp = ev.evaluate_policy(target, min(3000, max(500, n)), req.seed + 1000, req.gamma, req.start_dist)
    res.update({"reference_J_target_dp": float(d0 @ Vt), "reference_J_behaviour_dp": float(d0 @ Vb),
                "on_policy_mc_J_target": onp["discounted_return"],
                "target_actions": policy_actions_list(target), "behaviour_actions": policy_actions_list(behaviour),
                "policy_agreement": ev.agreement(policy_actions_list(target), policy_actions_list(behaviour)),
                "V_dp_target": Vt[:core.N_STATES]})
    for k in ("V_ordinary_is", "V_weighted_is", "visits"):
        res[k] = res[k][:core.N_STATES]
    return J(res)


@router.post("/rl/episodes/upload")
async def rl_episodes_upload(file: UploadFile = File(...)):
    """Upload logged sessions: columns episode_id, step, fatigue, tilt, streak,
    action (0/1/2 or LOWER/KEEP/RAISE), reward, optional behaviour_prob, done."""
    raw = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(400, f"Could not parse CSV: {e}")
    need = {"episode_id", "step", "fatigue", "tilt", "streak", "action", "reward"}
    missing = need - set(c.lower() for c in df.columns)
    if missing:
        raise HTTPException(400, f"missing columns: {sorted(missing)}")
    df.columns = [c.lower() for c in df.columns]
    amap = {"lower": 0, "keep": 1, "raise": 2, "0": 0, "1": 1, "2": 2}
    has_b = "behaviour_prob" in df.columns or "behavior_prob" in df.columns
    bcol = "behaviour_prob" if "behaviour_prob" in df.columns else ("behavior_prob" if has_b else None)
    episodes, core_eps = [], []
    bm = np.zeros((core.N_STATES_WITH_TERMINAL, core.N_ACTIONS)); bc = np.zeros(core.N_STATES_WITH_TERMINAL)
    for eid, g in df.sort_values(["episode_id", "step"]).groupby("episode_id", sort=False):
        S, A, R, trans = [], [], [], []
        for _, row in g.iterrows():
            s = core.state_from_features(float(row["fatigue"]), float(row["tilt"]), int(row["streak"]))
            a = amap.get(str(row["action"]).strip().lower())
            if a is None:
                raise HTTPException(400, f"unrecognised action '{row['action']}'")
            S.append(s); A.append(a); R.append(float(row["reward"]))
            bp = float(row[bcol]) if bcol and pd.notna(row[bcol]) else None
            if bp is not None:
                if not 0 < bp <= 1:
                    raise HTTPException(400, "behaviour_prob must be in (0, 1]")
                bm[s, a] = bp; bc[s] += 1
            trans.append(core.Transition(s, a, float(row["reward"]), core.TERMINAL, False, bp))
        for i in range(len(trans) - 1):
            trans[i].next_state = trans[i + 1].state
        trans[-1].done = True
        episodes.append({"states": S, "actions": A, "rewards": R})
        core_eps.append(core.Episode(trans, "LOGGED"))
    if has_b:
        # complete each visited state's behaviour row; unvisited rows default uniform
        for s in range(core.N_STATES_WITH_TERMINAL):
            row = bm[s]
            if row.sum() == 0:
                bm[s] = 1.0 / core.N_ACTIONS
            elif row.sum() < 1 - 1e-6:
                missing_a = row == 0
                if missing_a.any():
                    bm[s, missing_a] = (1 - row.sum()) / missing_a.sum()
    eid = f"log-{int(time.time() * 1000) % 10**8}"
    declared = str(df["source"].iloc[0]).upper() if "source" in df.columns and len(df) else ""
    prov = "SIMULATED (declared by the file's source column)" if "SIMUL" in declared else "LOGGED (uploaded file)"
    EPISODE_LOGS[eid] = {"episodes": episodes, "core_episodes": core_eps, "has_behaviour_probs": has_b,
                         "behaviour_matrix": bm.tolist(), "provenance": prov,
                         "notes": ["States are bucketed from the logged fatigue/tilt/streak columns."]}
    return {"episodes_id": eid, "n_episodes": len(episodes), "n_steps": int(len(df)), "has_behaviour_probs": has_b,
            "provenance": prov,
            "capabilities": ["MC prediction"] + (["off-policy importance sampling"] if has_b else []),
            "warning": None if has_b else "No behaviour_prob column: off-policy evaluation is disabled for this log."}


def sample_logs_csv(n_episodes: int = 200, seed: int = 31, epsilon: float = 0.3) -> str:
    """Generates the clearly-labelled SIMULATED sample log shipped with the project."""
    b = core.epsilon_soft(core.policy_actions(core.heuristic_policy()), epsilon)
    rows = ["episode_id,step,fatigue,tilt,streak,action,reward,behaviour_prob,done,source"]
    mid = [0.125, 0.375, 0.625, 0.875]
    streak_val = [-3, 0, 3]
    for i in range(n_episodes):
        ep = core.generate_episode(b, core.derive_seed(seed, i), start_dist="session_start")
        for t, tr in enumerate(ep.transitions):
            f, ti, k = core.decode(tr.state)
            rows.append(f"{i},{t},{mid[f]},{mid[ti]},{streak_val[k]},{core.ACTION_NAMES[tr.action]},{tr.reward:.4f},"
                        f"{tr.behaviour_prob:.4f},{int(tr.done)},SIMULATED")
    return "\n".join(rows) + "\n"


@router.get("/rl/episodes/sample")
def rl_episodes_sample():
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse(sample_logs_csv(), media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=sample_session_logs_SIMULATED.csv"})


# --------------------------------------------------------------------------
# Q-learning
# --------------------------------------------------------------------------
@router.get("/rl/q/table")
def rl_q_table(gamma: float = 0.9):
    Q, meta = ql.load_production()
    acts = core.greedy_from_q(Q[:core.N_STATES].tolist())
    _, vi = dp_solution(gamma, 1000, 7)
    return J({"Q": Q[:core.N_STATES], "actions": acts, "meta": meta, "keys": [core.state_key(s) for s in range(core.N_STATES)],
              "dp_actions": vi["actions"][:core.N_STATES], "agreement_with_dp": ev.agreement(acts, vi["actions"]),
              "disagreements": ev.disagreement_analysis(acts, vi["actions"], vi["Q"]),
              "algorithm": "Tabular Q-learning (NOT a DQN: no neural network, replay buffer or target network)",
              "provenance": "SIMULATED"})


class QTrainRequest(BaseModel):
    episodes: int = Field(30000, ge=100, le=100000)
    alpha: float = Field(0.05, gt=0, le=1)
    gamma: float = Field(0.9, ge=0, le=1)
    eps_start: float = Field(1.0, ge=0, le=1)
    eps_decay: float = Field(0.9998, gt=0.9, le=1)
    eps_min: float = Field(0.05, ge=0, le=1)
    seed: int = Field(7, ge=0)
    watch_state: Optional[int] = Field(None, ge=0, le=47)
    save_as_default: bool = False


@router.post("/rl/q/train")
def rl_q_train(req: QTrainRequest):
    model, vi = dp_solution(req.gamma, 1000, 7)
    res = ql.train(req.episodes, req.alpha, req.gamma, req.eps_start, req.eps_decay, req.eps_min, req.seed,
                   "uniform", vi["actions"], model, 40, 30, req.watch_state)
    saved = False
    if req.save_as_default:
        try:
            ql.save_production(res)
            ql._CACHE.clear()
            saved = True
        except OSError:
            saved = False
    acts = res["actions"]
    return J({"Q": res["Q"][:core.N_STATES], "actions": acts, "curve": res["curve"], "td_log": res["td_log"],
              "first_update": res["first_update"], "watch_log": res["watch_log"][:60], "visits": res["visits"][:core.N_STATES],
              "explore_steps": res["explore_steps"], "exploit_steps": res["exploit_steps"],
              "final_epsilon": res["final_epsilon"], "params": res["params"], "seconds": res["seconds"],
              "dp_actions": vi["actions"][:core.N_STATES], "agreement_with_dp": ev.agreement(acts, vi["actions"]),
              "disagreements": ev.disagreement_analysis(acts, vi["actions"], vi["Q"]),
              "J_learned": ev.model_value(model, acts, req.gamma), "J_dp_optimal": ev.model_value(model, vi["actions"], req.gamma),
              "saved_as_default": saved,
              "note": "Production policy unchanged" if not saved else "Saved as the new production q_table.json",
              "provenance": "SIMULATED"})


class TDStepRequest(BaseModel):
    state: int = Field(..., ge=0, le=47)
    action: int = Field(..., ge=0, le=2)
    reward: float
    next_state: int = Field(..., ge=0, le=48)
    done: bool = False
    alpha: float = Field(0.05, gt=0, le=1)
    gamma: float = Field(0.9, ge=0, le=1)


@router.post("/rl/q/td-step")
def rl_td_step(req: TDStepRequest):
    """Explain ONE Q-learning update on a copy of the production table (the
    production table itself is never modified by this endpoint)."""
    Q, _ = ql.load_production()
    Qc = Q.copy()
    return J(ql.td_update(Qc, req.state, req.action, req.reward, req.next_state, req.done or req.next_state == core.TERMINAL,
                          req.alpha, req.gamma))


# --------------------------------------------------------------------------
# comparison arena
# --------------------------------------------------------------------------
class CompareRequest(BaseModel):
    n_episodes: int = Field(500, ge=50, le=5000)
    seed: int = Field(2026, ge=0)
    gamma: float = Field(0.9, ge=0, le=1)
    start_dist: Literal["uniform", "session_start"] = "session_start"
    mc_episodes: int = Field(10000, ge=500, le=60000)
    mc_epsilon: float = Field(0.1, ge=0, le=1)
    q_episodes: int = Field(30000, ge=500, le=100000)
    stability_runs: int = Field(1, ge=1, le=5)
    samples: int = Field(1000, ge=50, le=20000)


@lru_cache(maxsize=8)
def _compare_cached(key):
    req = CompareRequest(**dict(key))
    return ev.compare_policies(req.n_episodes, req.seed, req.gamma, req.start_dist, req.samples, 7,
                               {"episodes": req.q_episodes}, req.mc_episodes, req.mc_epsilon, req.stability_runs)


@router.post("/rl/compare")
def rl_compare(req: CompareRequest):
    t0 = time.perf_counter()
    res = _compare_cached(tuple(sorted(req.model_dump().items())))
    out = J(res)
    out["wall_seconds"] = round(time.perf_counter() - t0, 3)
    return out


# --------------------------------------------------------------------------
# Unit 2: bandit arena
# --------------------------------------------------------------------------
DEFAULT_ROSTER = (["Aggressive Entry"] * 4 + ["Sniper/Isolation"] * 3 + ["Flex Rotator"] * 2 + ["Passive Support"])


def _sample_maps():
    import os
    from tournament_builder import load_maps_csv
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_maps.csv")
    return load_maps_csv(pd.read_csv(path))


class ArmsSpec(BaseModel):
    source: Literal["sample_maps", "tournament", "custom"] = "sample_maps"
    tournament_id: Optional[str] = None
    match_id: Optional[str] = None
    k: int = Field(10, ge=2, le=30)
    means: Optional[list[float]] = None
    labels: Optional[list[str]] = None


def build_arms(spec: ArmsSpec):
    if spec.source == "custom":
        if not spec.means or len(spec.means) < 2:
            raise HTTPException(400, "custom arms need >= 2 means")
        if any(not 0 <= m <= 1 for m in spec.means):
            raise HTTPException(400, "custom means must be in [0, 1]")
        labels = spec.labels or [f"Arm {i + 1}" for i in range(len(spec.means))]
        return [{"map_name": l, "q_star": m, "fit_score": None} for l, m in zip(labels, spec.means)], "custom (user-specified)"
    if spec.source == "tournament":
        from backend_app import TOURNAMENTS
        t = TOURNAMENTS.get(spec.tournament_id or "")
        if not t:
            raise HTTPException(404, "unknown tournament_id")
        match = next((m for m in t["matches"] if m["match_id"] == spec.match_id), t["matches"][0])
        roster = [{"archetype": a} for a in match["roster_archetypes"]]
        return bd.map_arms(t["map_pool"], roster, spec.k), f"tournament {spec.tournament_id} · {match['match_id']} ({match['team_a']} vs {match['team_b']})"
    roster = [{"archetype": a} for a in DEFAULT_ROSTER]
    return bd.map_arms(_sample_maps(), roster, spec.k), "sample_maps.csv + demo roster (4 entry, 3 sniper, 2 flex, 1 support)"


@router.post("/bandit/arms")
def bandit_arms(spec: ArmsSpec):
    arms, src = build_arms(spec)
    return J({"arms": arms, "roster_source": src, "provenance": "DERIVED FEATURE"})


class BanditRequest(BaseModel):
    algorithms: list[Literal["random", "greedy", "explore_first", "epsilon_greedy", "optimistic", "ucb"]] = \
        Field(default_factory=lambda: ["greedy", "epsilon_greedy", "optimistic", "ucb"])
    rounds: int = Field(1000, ge=10, le=10000)
    runs: int = Field(50, ge=1, le=500)
    seed: int = Field(1, ge=0)
    epsilon: float = Field(0.1, ge=0, le=1)
    c: float = Field(1.414, ge=0, le=10)
    q0: float = Field(1.0, ge=-5, le=10)
    alpha: Optional[float] = Field(None, gt=0, le=1)
    explore_rounds: int = Field(50, ge=0, le=5000)
    reward_model: Literal["bernoulli", "gaussian"] = "bernoulli"
    sigma: float = Field(0.1, gt=0, le=2)
    nonstationary: Literal["none", "drift", "patch"] = "none"
    drift_sigma: float = Field(0.01, ge=0, le=0.2)
    arms: ArmsSpec = Field(default_factory=ArmsSpec)


@router.post("/bandit/run")
def bandit_run(req: BanditRequest):
    if req.runs * req.rounds * len(req.algorithms) > 3_000_000:
        raise HTTPException(400, "budget too large: runs x rounds x algorithms must be <= 3,000,000")
    arms, src = build_arms(req.arms)
    means = [a["q_star"] for a in arms]
    res = bd.run_arena(req.algorithms, means, req.rounds, req.runs, req.seed, epsilon=req.epsilon, c=req.c, q0=req.q0,
                       alpha=req.alpha, explore_rounds=req.explore_rounds, reward_model=req.reward_model,
                       sigma=req.sigma, nonstationary=req.nonstationary, drift_sigma=req.drift_sigma)
    best = max(res["results"].items(), key=lambda kv: kv[1]["final_reward"])
    ex = best[1]["example_run"]
    res.update({"arms": arms, "roster_source": src, "best_algorithm": best[0],
                "bandit_recommended_map": arms[int(np.argmax(ex["Q"]))]["map_name"],
                "rule_based_map": arms[int(np.argmax(means))]["map_name"]})
    return J(res, 5)


@router.get("/bandit/numericals")
def bandit_numericals():
    return J(bd.all_notes_numericals())


# --------------------------------------------------------------------------
# academic
# --------------------------------------------------------------------------
@router.get("/academic/numericals")
def academic_numericals():
    items = notes_examples.all_numericals()
    return J({"items": items, "checked": sum(1 for i in items if i["match"] is not None),
              "matched": sum(1 for i in items if i["match"])})


@router.get("/academic/coverage")
def academic_coverage():
    return syllabus.coverage()


@router.get("/academic/viva-snapshot")
def academic_viva_snapshot(state: int = 42, gamma: float = 0.9):
    """Live numerical examples for every viva card, computed from the current
    production artefacts (DP on the estimated model, production Q-table, a
    seeded episode, a bandit step and an IS ratio)."""
    s = max(0, min(47, state))
    model, vi = dp_solution(gamma, 1000, 7)
    bd_opt = dp.backup_breakdown(model, s, vi["V"], gamma, top_k=4)
    Q, _ = ql.load_production()
    a = int(np.argmax(Q[s]))
    rng = core.Mulberry32(core.derive_seed(99, s))
    s2, r, done, info = core.env_step(s, a, rng)
    td = ql.td_update(Q.copy(), s, a, r, s2, done, 0.05, gamma)
    ep = core.generate_episode(core.deterministic_policy(list(vi["actions"])), 5, start_state=s, record_info=True)
    ep_rows = mc.episode_table(ep, gamma)
    heur = core.policy_actions(core.heuristic_policy())
    b = core.epsilon_soft(heur, 0.3); tgt = core.deterministic_policy(list(vi["actions"]))
    steps = [{"state": core.state_short(t.state), "action": core.ACTION_NAMES[t.action],
              "pi": tgt[t.state][t.action], "b": b[t.state][t.action]} for t in ep.transitions[:5]]
    rho = 1.0
    for st in steps:
        rho *= st["pi"] / st["b"]
    arms, _ = build_arms(ArmsSpec())
    ucb = bd.notes_ucb_example([0.55, 0.63, 0.61, 0.40], [3, 4, 3, 2], 12)
    return J({"state": core.state_info(s), "gamma": gamma, "value_iteration_backup": bd_opt,
              "V_star": float(vi["V"][s]), "best_action": core.ACTION_NAMES[int(vi["actions"][s])],
              "td_update": td, "mc_episode": ep_rows, "is_example": {"steps": steps, "rho": rho,
              "target": "DP-optimal (deterministic)", "behaviour": "heuristic, ε = 0.3"},
              "bandit_arms": arms[:5], "ucb_example": ucb,
              "provenance": "MODEL ESTIMATE + SIMULATED"})
