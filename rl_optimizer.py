"""
rl_optimizer.py — Tabular Q-Learning Session Optimizer (backwards-compatible facade).

v3 described this component as a "Deep-Q"/"DQN-lite" optimizer. That was
academically inaccurate: the policy is a lookup table Q[s, a] over 48
discrete states learned by *tabular Q-learning*. There is no neural network,
experience replay or target network, so it is not a DQN. v4 renames it and
moves the implementation into the shared RL modules:

    rl_core.py        state/action/reward definitions + simulator
    q_learning.py     tabular Q-learning (training, TD-update explainer)

This file keeps every public name other modules and the existing endpoints
use (reward_function, bucket_*, state_key, recommend_action, QTableAgent,
train, get_agent), so nothing that worked in v3 breaks.

A genuine DQN (neural Q-approximator + replay buffer + target network) is
documented in RL_METHODOLOGY.md as an optional extension; it is not needed
for a 48-state problem and is not part of the uploaded Units 1-5 syllabus.
"""
from __future__ import annotations

import numpy as np

import rl_core as core
import q_learning as ql
from rl_core import (W1, W2, W3, ACTIONS, ACTION_LABELS, bucket_fatigue, bucket_tilt,  # noqa: F401  (re-exported)
                     bucket_streak, reward_function)

Q_TABLE_PATH = ql.Q_TABLE_PATH


def state_key(fatigue: float, tilt: float, streak: int) -> str:
    return core.state_key(fatigue, tilt, streak)


class QTableAgent:
    """Thin object wrapper kept for v3 compatibility."""

    def __init__(self, Q: np.ndarray | None = None):
        self.Q = np.zeros((core.N_STATES_WITH_TERMINAL, core.N_ACTIONS)) if Q is None else Q

    @property
    def q(self) -> dict:
        return ql.q_to_dict(self.Q)

    def get_q(self, s_key: str):
        return self.Q[core.key_to_state(s_key)].tolist()

    def act(self, fatigue, tilt, streak, epsilon=0.0, rng=None):
        rng = rng or core.Mulberry32(0)
        s = core.state_from_features(fatigue, tilt, streak)
        return ql.epsilon_greedy(self.Q[s], epsilon, rng)[0]

    def update(self, s_key, action, reward, next_s_key, alpha=0.05, gamma=0.9, done=False):
        s2 = core.TERMINAL if next_s_key == "END" else core.key_to_state(next_s_key)
        return ql.td_update(self.Q, core.key_to_state(s_key), action, reward, s2, done, alpha, gamma)

    def save(self, path=Q_TABLE_PATH):
        ql.save_production({"Q": self.Q, "params": {}, "seconds": 0.0, "final_epsilon": None}, path)

    def load(self, path=Q_TABLE_PATH):
        Q, _ = ql.load_production()
        self.Q = Q
        return True


def train(episodes: int = ql.DEFAULT_PARAMS["episodes"], seed: int = ql.DEFAULT_PARAMS["seed"]) -> QTableAgent:
    params = dict(ql.DEFAULT_PARAMS, episodes=episodes, seed=seed)
    res = ql.train(**params)
    ql.save_production(res)
    ql._CACHE.clear()
    return QTableAgent(res["Q"])


def get_agent() -> QTableAgent:
    Q, _ = ql.load_production()
    return QTableAgent(Q)


def recommend_action(fatigue: float, tilt: float, streak: int) -> dict:
    """Greedy action of the production Q-table for a live player state.
    Response keys from v3 are preserved; v4 adds the decoded state and the
    simulator-model risk numbers for every action (labelled SIMULATED)."""
    Q, meta = ql.load_production()
    s = core.state_from_features(fatigue, tilt, streak)
    qvals = Q[s]
    best = int(np.argmax(qvals))
    models = [core.step_model(s, a) for a in core.ACTIONS]
    return {
        "state_bucket": core.state_key(s),
        "state_index": s,
        "state": core.state_info(s),
        "action": best,
        "action_label": ACTION_LABELS[best],
        "action_name": core.ACTION_NAMES[best],
        "q_values": {ACTION_LABELS[a]: round(float(qvals[a]), 3) for a in ACTIONS},
        "q_vector": [float(x) for x in qvals],
        "predicted_rage_quit_risk": round(models[best]["rage_prob"], 3),
        "per_action_model": [{"action": a, "p_win": round(m["p_win"], 3), "closeness": round(m["closeness"], 3),
                              "rage_quit_prob": round(m["rage_prob"], 4), "boredom_quit_prob": round(m["boredom_prob"], 4),
                              "expected_immediate_reward": round(m["expected_reward_immediate"], 4)}
                             for a, m in zip(ACTIONS, models)],
        "policy": "Tabular Q-learning (greedy w.r.t. production Q-table)",
        "risk_provenance": "SIMULATED (simulator model, not fitted to real players)",
        "training": meta.get("params", {}),
    }


# Legacy simulator name kept for any external script that imported it.
SimEnvironment = core.SessionEnv


if __name__ == "__main__":
    agent = train()
    print("Trained tabular Q-learning session optimizer.")
    print(recommend_action(fatigue=0.8, tilt=0.75, streak=-4))
