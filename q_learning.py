"""
q_learning.py — tabular Q-learning (model-free, off-policy TD control).

This is the algorithm the original project called a "DQN". It is NOT a deep
Q-network: there is no neural network, no experience replay and no target
network. It is a table Q[s, a] over the 48 bucketed states, updated by

    TD target  = R_{t+1} + gamma * max_a' Q(S_{t+1}, a')      (= R_{t+1} if S_{t+1} is terminal)
    TD error   = TD target - Q(S_t, A_t)
    Q(S_t,A_t) <- Q(S_t,A_t) + alpha * TD error

with epsilon-greedy behaviour (epsilon decays multiplicatively to a floor).

Fixes relative to v3: (1) the TD target no longer bootstraps from the state
reached after a terminal transition (v3 added gamma*max Q(s') even after a
rage quit); (2) rewards are bounded per step instead of being dominated by a
minutes-scale term; (3) training is seeded with the shared Mulberry32 PRNG so
the browser engine reproduces it exactly.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

import rl_core as core

Q_TABLE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "q_table.json")
Q_META_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "q_table_meta.json")

DEFAULT_PARAMS = {"episodes": 30000, "alpha": 0.05, "gamma": 0.9, "eps_start": 1.0, "eps_decay": 0.9998,
                  "eps_min": 0.05, "seed": 7, "start_dist": "uniform"}


def epsilon_greedy(Qs, epsilon: float, rng: core.Mulberry32) -> tuple[int, bool]:
    """Returns (action, explored?). One uniform decides explore/exploit; a
    second picks the random action (same draw order as rl_engine.js)."""
    if rng.random() < epsilon:
        return rng.choice_index(core.N_ACTIONS), True
    best = 0
    for a in range(1, core.N_ACTIONS):
        if Qs[a] > Qs[best] + 1e-12:
            best = a
    return best, False


def td_update(Q: np.ndarray, s: int, a: int, r: float, s2: int, done: bool, alpha: float, gamma: float) -> dict:
    """One Q-learning update, returning every intermediate number for display."""
    old = float(Q[s, a])
    max_next = 0.0 if done else float(np.max(Q[s2]))
    target = r + (0.0 if done else gamma * max_next)
    td_error = target - old
    Q[s, a] = old + alpha * td_error
    return {"state": int(s), "state_label": core.state_short(s), "action": int(a), "action_name": core.ACTION_NAMES[a],
            "reward": float(r), "next_state": int(s2), "next_label": core.state_short(s2), "done": bool(done),
            "old_q": old, "max_next_q": max_next, "td_target": target, "td_error": td_error,
            "alpha": alpha, "gamma": gamma, "new_q": float(Q[s, a])}


def train(episodes: int = 30000, alpha: float = 0.05, gamma: float = 0.9, eps_start: float = 1.0,
          eps_decay: float = 0.9998, eps_min: float = 0.05, seed: int = 7, start_dist: str = "uniform",
          reference_actions=None, model=None, n_checkpoints: int = 40, td_log_last: int = 30,
          watch_state: int | None = None) -> dict:
    t0 = time.perf_counter()
    nS, nA = core.N_STATES_WITH_TERMINAL, core.N_ACTIONS
    Q = np.zeros((nS, nA))
    eps = float(eps_start)
    every = max(1, episodes // n_checkpoints)
    curve, td_log, watch_log = [], [], []
    ep_ret, ep_disc, ep_len, td_abs = [], [], [], []
    explore_steps = exploit_steps = 0
    first_update = None
    visits = np.zeros((nS, nA), dtype=int)
    for ep in range(episodes):
        ep_seed = core.derive_seed(seed, ep)
        env_rng = core.Mulberry32(core.derive_seed(ep_seed, 0))
        act_rng = core.Mulberry32(core.derive_seed(ep_seed, 1))
        s = core.sample_start_state(env_rng, start_dist)
        G = Gd = 0.0
        disc = 1.0
        n = 0
        abs_sum = 0.0
        for _ in range(core.MAX_EPISODE_STEPS):
            a, explored = epsilon_greedy(Q[s], eps, act_rng)
            if explored:
                explore_steps += 1
            else:
                exploit_steps += 1
            s2, r, done, _ = core.env_step(s, a, env_rng)
            upd = td_update(Q, s, a, r, s2, done, alpha, gamma)
            visits[s, a] += 1
            if first_update is None:
                first_update = dict(upd, episode=ep, epsilon=eps)
            if watch_state is not None and s == watch_state and len(watch_log) < 200:
                watch_log.append(dict(upd, episode=ep))
            abs_sum += abs(upd["td_error"])
            G += r
            Gd += disc * r
            disc *= gamma
            n += 1
            s = s2
            if done:
                break
        ep_ret.append(G); ep_disc.append(Gd); ep_len.append(n); td_abs.append(abs_sum / max(1, n))
        if (ep + 1) % every == 0 or ep + 1 == episodes:
            greedy = core.greedy_from_q(Q[:core.N_STATES].tolist())
            row = {"episodes": ep + 1, "epsilon": eps,
                   "mean_return": float(np.mean(ep_ret[-every:])),
                   "mean_discounted_return": float(np.mean(ep_disc[-every:])),
                   "mean_length": float(np.mean(ep_len[-every:])),
                   "mean_abs_td_error": float(np.mean(td_abs[-every:]))}
            if reference_actions is not None:
                row["agreement_with_reference"] = float(np.mean(np.array(greedy) == np.asarray(reference_actions)[:core.N_STATES]))
            if model is not None:
                from monte_carlo import _policy_value_under_model
                row["greedy_policy_value"] = _policy_value_under_model(greedy, model, gamma, "session_start")[0]
            curve.append(row)
        eps = max(eps_min, eps * eps_decay)
    # a short replay of the final greedy behaviour's updates for the TD explainer
    replay_Q = Q.copy()
    env_rng = core.Mulberry32(core.derive_seed(core.derive_seed(seed, episodes + 1), 0))
    s = core.sample_start_state(env_rng, "session_start")
    for _ in range(td_log_last):
        a = core.greedy_from_q([replay_Q[s].tolist()])[0]
        s2, r, done, _ = core.env_step(s, a, env_rng)
        td_log.append(td_update(replay_Q, s, a, r, s2, done, alpha, gamma))
        s = s2
        if done:
            s = core.sample_start_state(env_rng, "session_start")
    return {"Q": Q, "actions": core.greedy_from_q(Q[:core.N_STATES].tolist()), "curve": curve,
            "td_log": td_log, "first_update": first_update, "watch_log": watch_log,
            "explore_steps": explore_steps, "exploit_steps": exploit_steps, "final_epsilon": eps,
            "params": {"episodes": episodes, "alpha": alpha, "gamma": gamma, "eps_start": eps_start,
                       "eps_decay": eps_decay, "eps_min": eps_min, "seed": seed, "start_dist": start_dist},
            "seconds": time.perf_counter() - t0, "provenance": "SIMULATED", "visits": visits}


# --------------------------------------------------------------------------
# production table I/O (backwards-compatible "f-t-k" key format)
# --------------------------------------------------------------------------
def q_to_dict(Q: np.ndarray) -> dict:
    return {core.state_key(s): [float(x) for x in Q[s]] for s in range(core.N_STATES)}


def dict_to_q(d: dict) -> np.ndarray:
    Q = np.zeros((core.N_STATES_WITH_TERMINAL, core.N_ACTIONS))
    for k, v in d.items():
        if k.startswith("_"):
            continue
        Q[core.key_to_state(k)] = v
    return Q


def save_production(result: dict, path: str = Q_TABLE_PATH, meta_path: str = Q_META_PATH):
    with open(path, "w") as f:
        json.dump(q_to_dict(result["Q"]), f)
    meta = {"algorithm": "Tabular Q-learning (off-policy TD control)", "params": result["params"],
            "trained_seconds": round(result["seconds"], 3), "provenance": "SIMULATED",
            "environment": "rl_core.SessionEnv (v4 formal MDP)", "final_epsilon": result["final_epsilon"]}
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


_CACHE: dict = {}


def load_production() -> tuple[np.ndarray, dict]:
    """Loads the production Q-table. If it's missing, trains it with the
    documented DEFAULT_PARAMS (deterministic, ~2 s)."""
    if "Q" in _CACHE:
        return _CACHE["Q"], _CACHE["meta"]
    if os.path.exists(Q_TABLE_PATH) and os.path.exists(Q_META_PATH):
        with open(Q_TABLE_PATH) as f:
            Q = dict_to_q(json.load(f))
        with open(Q_META_PATH) as f:
            meta = json.load(f)
    else:
        res = train(**DEFAULT_PARAMS)
        Q = res["Q"]
        meta = {"algorithm": "Tabular Q-learning", "params": res["params"], "provenance": "SIMULATED",
                "note": "trained at startup because q_table.json was missing"}
        try:
            save_production(res)
        except OSError:
            pass  # read-only filesystem (e.g. serverless) — keep in memory
    _CACHE["Q"], _CACHE["meta"] = Q, meta
    return Q, meta


if __name__ == "__main__":
    res = train(**DEFAULT_PARAMS)
    save_production(res)
    print(f"Trained tabular Q-learning in {res['seconds']:.2f}s; saved {Q_TABLE_PATH}")
