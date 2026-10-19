"""
dynamic_programming.py — UNIT 4: planning in a known finite MDP.

Implements exactly the four algorithms in the Unit 4 notes, for ANY
mdp_model.FiniteMDP (unit tests run them on the notes' dice game and the
4x4 gridworld as well as on the Clutch IQ session MDP):

  1. Iterative policy evaluation   V(s) <- sum_a pi(a|s) sum_s' p(s'|s,a)[r + gamma V(s')]
  2. Policy improvement            pi'(s) = argmax_a sum_s' p(s'|s,a)[r + gamma V(s')]
  3. Policy iteration              E -> I -> E -> I ... until policy-stable
  4. Value iteration               V(s) <- max_a sum_s' p(s'|s,a)[r + gamma V(s')]

Model-based: every backup uses p(s'|s,a) and r(s,a,s') explicitly.

Sweeps are synchronous ("two-array") by default so that the recorded max-delta
sequence demonstrates the gamma-contraction property behind the Banach
fixed-point argument (||V_{k+1} - V_k|| <= gamma ||V_k - V_{k-1}|| for value
iteration). In-place (Gauss–Seidel) sweeps, as in the Sutton pseudocode, are
available with in_place=True.
"""
from __future__ import annotations

import time

import numpy as np

from mdp_model import FiniteMDP


def _as_policy(pi, nS: int, nA: int) -> np.ndarray:
    pi = np.asarray(pi, dtype=float)
    if pi.ndim == 1:  # deterministic action list
        out = np.zeros((nS, nA))
        out[np.arange(nS), pi.astype(int)] = 1.0
        return out
    return pi


def bellman_expectation_backup(mdp: FiniteMDP, V: np.ndarray, pi: np.ndarray, gamma: float) -> np.ndarray:
    q = mdp.q_from_v(V, gamma)
    newV = (pi * q).sum(axis=1)
    newV[mdp.terminal] = 0.0
    return newV


def policy_evaluation(mdp: FiniteMDP, pi, gamma: float | None = None, theta: float = 1e-6,
                      max_iter: int = 1000, V0: np.ndarray | None = None, in_place: bool = False,
                      record_snapshots: int = 0) -> dict:
    """Iterative policy evaluation. Returns V_pi, per-sweep max delta, and
    optional V snapshots (first `record_snapshots` sweeps) for animation."""
    g = mdp.gamma if gamma is None else float(gamma)
    pi = _as_policy(pi, mdp.nS, mdp.nA)
    V = np.zeros(mdp.nS) if V0 is None else np.array(V0, dtype=float)
    deltas, snaps = [], []
    t0 = time.perf_counter()
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        if in_place:
            delta = 0.0
            for s in range(mdp.nS):
                if mdp.terminal[s]:
                    continue
                v = V[s]
                V[s] = float((pi[s] * (mdp.P[s] * (mdp.R[s] + g * V[None, :])).sum(axis=1)).sum())
                delta = max(delta, abs(v - V[s]))
        else:
            newV = bellman_expectation_backup(mdp, V, pi, g)
            delta = float(np.max(np.abs(newV - V)))
            V = newV
        deltas.append(delta)
        if it <= record_snapshots:
            snaps.append(V.copy())
        if delta < theta:
            converged = True
            break
    return {"V": V, "iterations": it, "deltas": deltas, "converged": converged,
            "snapshots": snaps, "seconds": time.perf_counter() - t0, "gamma": g, "theta": theta}


def greedy_policy(mdp: FiniteMDP, V: np.ndarray, gamma: float, old_actions: np.ndarray | None = None,
                  tol: float = 1e-9) -> tuple[np.ndarray, np.ndarray]:
    """Policy improvement step. Ties within `tol` keep the old action, which
    prevents policy iteration from cycling between equally good policies."""
    q = mdp.q_from_v(V, gamma)
    best = np.argmax(q, axis=1)
    if old_actions is not None:
        qmax = q[np.arange(mdp.nS), best]
        keep = q[np.arange(mdp.nS), old_actions] >= qmax - tol
        best = np.where(keep, old_actions, best)
    return best.astype(int), q


def policy_improvement(mdp: FiniteMDP, V: np.ndarray, pi_actions: np.ndarray, gamma: float | None = None) -> dict:
    g = mdp.gamma if gamma is None else float(gamma)
    new, q = greedy_policy(mdp, V, g, np.asarray(pi_actions, int))
    changed = [int(s) for s in np.where((new != pi_actions) & ~mdp.terminal)[0]]
    return {"actions": new, "Q": q, "changed_states": changed, "policy_stable": len(changed) == 0}


def policy_iteration(mdp: FiniteMDP, gamma: float | None = None, theta: float = 1e-6, max_iter: int = 100,
                     eval_max_iter: int = 1000, initial_actions=None) -> dict:
    g = mdp.gamma if gamma is None else float(gamma)
    actions = np.ones(mdp.nS, dtype=int) if initial_actions is None else np.asarray(initial_actions, int).copy()
    V = np.zeros(mdp.nS)
    history = []
    t0 = time.perf_counter()
    total_sweeps = 0
    stable = False
    for k in range(1, max_iter + 1):
        ev = policy_evaluation(mdp, actions, g, theta, eval_max_iter, V0=V)
        V = ev["V"]
        total_sweeps += ev["iterations"]
        imp = policy_improvement(mdp, V, actions, g)
        history.append({"iteration": k, "evaluation_sweeps": ev["iterations"], "final_delta": ev["deltas"][-1],
                        "changed_states": imp["changed_states"], "n_changed": len(imp["changed_states"]),
                        "policy": actions.tolist(), "V": V.copy()})
        actions = imp["actions"]
        if imp["policy_stable"]:
            stable = True
            break
    Q = mdp.q_from_v(V, g)
    return {"V": V, "Q": Q, "actions": actions, "history": history, "policy_stable": stable,
            "improvement_steps": len(history), "evaluation_sweeps_total": total_sweeps,
            "seconds": time.perf_counter() - t0, "gamma": g, "theta": theta}


def value_iteration(mdp: FiniteMDP, gamma: float | None = None, theta: float = 1e-6, max_iter: int = 2000,
                    record_snapshots: int = 0, in_place: bool = False) -> dict:
    g = mdp.gamma if gamma is None else float(gamma)
    V = np.zeros(mdp.nS)
    deltas, snaps = [], []
    t0 = time.perf_counter()
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        if in_place:
            delta = 0.0
            for s in range(mdp.nS):
                if mdp.terminal[s]:
                    continue
                v = V[s]
                V[s] = float((mdp.P[s] * (mdp.R[s] + g * V[None, :])).sum(axis=1).max())
                delta = max(delta, abs(v - V[s]))
        else:
            q = mdp.q_from_v(V, g)
            newV = q.max(axis=1)
            newV[mdp.terminal] = 0.0
            delta = float(np.max(np.abs(newV - V)))
            V = newV
        deltas.append(delta)
        if it <= record_snapshots:
            snaps.append(V.copy())
        if delta < theta:
            converged = True
            break
    actions, Q = greedy_policy(mdp, V, g)
    ratios = [deltas[i + 1] / deltas[i] for i in range(len(deltas) - 1) if deltas[i] > 0]
    return {"V": V, "Q": Q, "actions": actions, "iterations": it, "deltas": deltas, "converged": converged,
            "contraction_ratios": ratios, "max_contraction_ratio": max(ratios) if ratios else 0.0,
            "snapshots": snaps, "seconds": time.perf_counter() - t0, "gamma": g, "theta": theta}


def bellman_residual(mdp: FiniteMDP, V: np.ndarray, gamma: float, pi=None) -> float:
    """max_s |V(s) - (T V)(s)|. With pi: expectation operator; without: optimality operator."""
    if pi is None:
        TV = mdp.q_from_v(V, gamma).max(axis=1)
    else:
        TV = bellman_expectation_backup(mdp, V, _as_policy(pi, mdp.nS, mdp.nA), gamma)
    TV[mdp.terminal] = 0.0
    return float(np.max(np.abs(TV - V)))


def backup_breakdown(mdp: FiniteMDP, s: int, V: np.ndarray, gamma: float, pi=None, top_k: int = 6,
                     labels: list[str] | None = None) -> dict:
    """Numerical Bellman backup for ONE state using the program's actual
    numbers. For each action: the successor terms p(s'|s,a)*[r(s,a,s') + gamma V(s')],
    q(s,a), and then either the expectation over pi or the max (optimality)."""
    labels = labels or mdp.state_names
    per_action = []
    for a in range(mdp.nA):
        row = mdp.P[s, a]
        order = [int(x) for x in np.argsort(-row) if row[x] > 0]
        terms = []
        for s2 in order:
            terms.append({"next_state": s2, "label": labels[s2], "p": float(row[s2]), "r": float(mdp.R[s, a, s2]),
                          "V_next": float(V[s2]), "contribution": float(row[s2] * (mdp.R[s, a, s2] + gamma * V[s2]))})
        q = float(sum(t["contribution"] for t in terms))
        shown = terms[:top_k]
        rest = terms[top_k:]
        per_action.append({"action": a, "action_name": mdp.action_names[a], "q": q,
                           "expected_reward": float(mdp.r_sa[s, a]),
                           "expected_next_value": float((row * V).sum()),
                           "terms": shown,
                           "other_terms": {"count": len(rest), "p": float(sum(t["p"] for t in rest)),
                                           "contribution": float(sum(t["contribution"] for t in rest))}})
    qs = [pa["q"] for pa in per_action]
    out = {"state": s, "gamma": gamma, "per_action": per_action,
           "optimality": {"max_q": max(qs), "argmax": int(np.argmax(qs))}}
    if pi is not None:
        pi = _as_policy(pi, mdp.nS, mdp.nA)
        out["expectation"] = {"pi": pi[s].tolist(), "value": float(sum(pi[s][a] * qs[a] for a in range(mdp.nA)))}
    return out
