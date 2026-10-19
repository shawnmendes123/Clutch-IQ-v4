"""
mdp_model.py — UNIT 3: Markov chains, MRPs, finite MDPs.

FiniteMDP stores the four-argument dynamics in the (equivalent) two-array
form used throughout the Unit 3 notes:

    p(s'|s,a)   = P[s, a, s']                      (state-transition probabilities)
    r(s,a,s')   = R[s, a, s']                      (expected reward for the triple)
    r(s,a)      = sum_s' p(s'|s,a) r(s,a,s')       (expected reward for the pair)

Terminal states are absorbing with zero reward, so V(terminal) = 0 exactly as
in the Sutton & Barto pseudocode reproduced in the Unit 4 notes.

For Clutch IQ the true transition law of real players is unknown, so the
model the planners use is a SIMULATION-ESTIMATED TRANSITION MODEL: the
maximum-likelihood estimate built from repeated rollouts of the session
simulator (rl_core.env_step) from every (state, action) pair:

    p_hat(s'|s,a) = N(s,a,s') / N(s,a)
    r_hat(s,a,s') = (sum of rewards observed on s,a -> s') / N(s,a,s')

It is labelled MODEL ESTIMATE everywhere in the API and UI.
"""
from __future__ import annotations

import time
from functools import lru_cache

import numpy as np

import rl_core as core


class FiniteMDP:
    def __init__(self, P: np.ndarray, R: np.ndarray, terminal: np.ndarray, gamma: float = 0.9,
                 state_names: list[str] | None = None, action_names: list[str] | None = None,
                 meta: dict | None = None):
        P = np.asarray(P, dtype=float)
        R = np.asarray(R, dtype=float)
        assert P.ndim == 3 and P.shape == R.shape, "P and R must both be [S, A, S]"
        self.P, self.R = P, R
        self.nS, self.nA = P.shape[0], P.shape[1]
        self.terminal = np.asarray(terminal, dtype=bool)
        self.gamma = float(gamma)
        self.state_names = state_names or [str(s) for s in range(self.nS)]
        self.action_names = action_names or [str(a) for a in range(self.nA)]
        self.meta = meta or {}
        # absorbing terminal: self-loop, zero reward
        for s in np.where(self.terminal)[0]:
            self.P[s, :, :] = 0.0
            self.P[s, :, s] = 1.0
            self.R[s, :, :] = 0.0

    # expected reward r(s,a) = sum_s' p(s'|s,a) r(s,a,s')
    @property
    def r_sa(self) -> np.ndarray:
        return (self.P * self.R).sum(axis=2)

    def with_gamma(self, gamma: float) -> "FiniteMDP":
        m = FiniteMDP(self.P.copy(), self.R.copy(), self.terminal.copy(), gamma,
                      self.state_names, self.action_names, dict(self.meta))
        return m

    # ----------------------------------------------------- Markov chain / MRP
    def policy_markov_chain(self, pi: np.ndarray) -> np.ndarray:
        """P_pi[s, s'] = sum_a pi(a|s) p(s'|s,a): fixing a policy turns the
        MDP into a Markov chain (Unit 3)."""
        return np.einsum("sa,sap->sp", np.asarray(pi, float), self.P)

    def policy_mrp(self, pi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Markov Reward Process (P_pi, R_pi) induced by policy pi, where
        R_pi(s) = sum_a pi(a|s) r(s,a)."""
        pi = np.asarray(pi, float)
        return self.policy_markov_chain(pi), (pi * self.r_sa).sum(axis=1)

    def mrp_closed_form(self, pi: np.ndarray, gamma: float | None = None) -> np.ndarray:
        """Bellman equation for an MRP in matrix form, v = R + gamma P v,
        solved directly: v = (I - gamma P)^-1 R (non-terminal block only).
        Works for gamma = 1 too because every policy here terminates w.p. 1."""
        g = self.gamma if gamma is None else gamma
        Ppi, Rpi = self.policy_mrp(pi)
        nt = ~self.terminal
        A = np.eye(nt.sum()) - g * Ppi[np.ix_(nt, nt)]
        v = np.zeros(self.nS)
        v[nt] = np.linalg.solve(A, Rpi[nt])
        return v

    def sequence_probability(self, states: list[int], pi: np.ndarray, start: np.ndarray | None = None) -> float:
        """P(s_1,...,s_k) = P(s_1) * prod P_pi(s_i | s_{i-1})  (Markov chain rule from the notes)."""
        Ppi = self.policy_markov_chain(pi)
        p = 1.0 if start is None else float(start[states[0]])
        for a, b in zip(states[:-1], states[1:]):
            p *= Ppi[a, b]
        return p

    # ------------------------------------------------------------- helpers
    def q_from_v(self, V: np.ndarray, gamma: float | None = None) -> np.ndarray:
        """q(s,a) = sum_s' p(s'|s,a) [r(s,a,s') + gamma V(s')]."""
        g = self.gamma if gamma is None else gamma
        return (self.P * (self.R + g * V[None, None, :])).sum(axis=2)

    def check(self, tol: float = 1e-9) -> dict:
        sums = self.P.sum(axis=2)
        return {"rows_sum_to_one": bool(np.all(np.abs(sums - 1) < tol)),
                "max_row_error": float(np.max(np.abs(sums - 1))),
                "min_probability": float(self.P.min())}

    def to_dict(self, decimals: int = 5) -> dict:
        return {"n_states": self.nS, "n_actions": self.nA, "gamma": self.gamma,
                "terminal": self.terminal.tolist(),
                "P": np.round(self.P, decimals).tolist(),
                "R": np.round(self.R, decimals).tolist(),
                "r_sa": np.round(self.r_sa, decimals).tolist(),
                "state_names": self.state_names, "action_names": self.action_names,
                "meta": self.meta}


# ---------------------------------------------------------------------------
# Clutch IQ session MDP
# ---------------------------------------------------------------------------
def _terminal_mask() -> np.ndarray:
    m = np.zeros(core.N_STATES_WITH_TERMINAL, dtype=bool)
    m[core.TERMINAL] = True
    return m


def _names() -> list[str]:
    return [core.state_short(s) for s in range(core.N_STATES_WITH_TERMINAL)]


@lru_cache(maxsize=16)
def estimate_model(samples_per_pair: int = 1000, seed: int = 7) -> FiniteMDP:
    """SIMULATION-ESTIMATED TRANSITION MODEL.

    For every (s, a) we place the simulator in state s, apply action a and
    record (s', r) — samples_per_pair independent times. Cached: the same
    (samples, seed) always yields the same model (deterministic PRNG).
    """
    t0 = time.perf_counter()
    nS, nA = core.N_STATES_WITH_TERMINAL, core.N_ACTIONS
    counts = np.zeros((nS, nA, nS))
    rsum = np.zeros((nS, nA, nS))
    for s in range(core.N_STATES):
        for a in range(nA):
            rng = core.Mulberry32(core.derive_seed(seed, 1000 + s * nA + a))
            for _ in range(samples_per_pair):
                s2, r, _, _ = core.env_step(s, a, rng)
                counts[s, a, s2] += 1
                rsum[s, a, s2] += r
    P = counts / np.maximum(counts.sum(axis=2, keepdims=True), 1)
    R = np.divide(rsum, counts, out=np.zeros_like(rsum), where=counts > 0)
    meta = {"kind": "SIMULATION-ESTIMATED TRANSITION MODEL", "provenance": "MODEL ESTIMATE",
            "samples_per_pair": samples_per_pair, "seed": seed,
            "total_simulated_transitions": int(samples_per_pair * core.N_STATES * nA),
            "build_seconds": round(time.perf_counter() - t0, 3)}
    m = FiniteMDP(P, R, _terminal_mask(), 0.9, _names(), core.ACTION_NAMES, meta)
    m.counts = counts  # kept for confidence reporting
    return m


@lru_cache(maxsize=1)
def exact_model() -> FiniteMDP:
    """The simulator's exact law (validation reference only — see rl_core.exact_transition)."""
    nS, nA = core.N_STATES_WITH_TERMINAL, core.N_ACTIONS
    P = np.zeros((nS, nA, nS))
    R = np.zeros((nS, nA, nS))
    for s in range(core.N_STATES):
        for a in range(nA):
            probs, rewards = core.exact_transition(s, a)
            for s2, p in probs.items():
                P[s, a, s2] = p
                R[s, a, s2] = rewards[s2]
    return FiniteMDP(P, R, _terminal_mask(), 0.9, _names(), core.ACTION_NAMES,
                     {"kind": "SIMULATOR GROUND TRUTH (validation only)", "provenance": "SIMULATED"})


def get_model(samples: int = 1000, seed: int = 7, gamma: float = 0.9, source: str = "estimated") -> FiniteMDP:
    base = exact_model() if source == "exact" else estimate_model(int(samples), int(seed))
    m = base.with_gamma(gamma)
    if hasattr(base, "counts"):
        m.counts = base.counts
    return m


def model_quality(m: FiniteMDP) -> dict:
    """How good is the simulation-estimated model? Compares with the
    simulator's exact law and reports a binomial standard error bound."""
    ex = exact_model()
    nt = core.N_STATES
    abs_err = np.abs(m.P[:nt] - ex.P[:nt])
    n = m.meta.get("samples_per_pair", 0) or 1
    r_err = np.abs(m.r_sa[:nt] - ex.r_sa[:nt])
    return {
        "max_abs_transition_error": float(abs_err.max()),
        "mean_abs_transition_error": float(abs_err[ex.P[:nt] > 0].mean()),
        "max_expected_reward_error": float(r_err.max()),
        "worst_case_std_error": float(np.sqrt(0.25 / n)),
        "nonzero_transitions_per_pair_mean": float((m.P[:nt] > 0).sum(axis=2).mean()),
        "rows_sum_to_one": m.check()["rows_sum_to_one"],
    }


def top_transitions(m: FiniteMDP, s: int, a: int, k: int = 6) -> list[dict]:
    row = m.P[s, a]
    order = np.argsort(-row)
    out = []
    for s2 in order[:k]:
        if row[s2] <= 0:
            break
        item = {"next_state": int(s2), "label": core.state_label(int(s2)), "short": core.state_short(int(s2)),
                "p": float(row[s2]), "r": float(m.R[s, a, s2])}
        if hasattr(m, "counts"):
            item["count"] = int(m.counts[s, a, s2])
        out.append(item)
    rest = float(row[order[k:]].sum()) if len(order) > k else 0.0
    if rest > 1e-12:
        out.append({"next_state": None, "label": f"{int((row[order[k:]] > 0).sum())} other states", "short": "other", "p": rest, "r": None})
    return out
