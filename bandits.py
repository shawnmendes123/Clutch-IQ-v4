"""
bandits.py — UNIT 2: Multi-armed bandits for map recommendation.

WHY A BANDIT (and not the MDP): choosing the map for ONE scheduled match is
a one-step decision. The map shown does not change the next state the way
lobby difficulty changes a player's tilt/fatigue trajectory, so the problem
is "immediate RL" (Unit 1): estimate q*(a) = E[R | A = a] for each arm and
balance exploration against exploitation. The session controller (Units 3-5)
is a sequential MDP and is evaluated separately — the two are never compared
on the same leaderboard.

Arms      : candidate maps from maps.csv for the selected roster.
q*(a)     : DERIVED FEATURE — a monotone transform of the tournament builder's
            existing map-fit score (playstyle match + sniper affinity + rank).
Reward R_t: SIMULATED post-match satisfaction:
              'bernoulli' -> R ~ Bernoulli(q*(a))        (thumbs-up / no)
              'gaussian'  -> R ~ N(q*(a), sigma^2)        (match-quality score)
Non-stationarity (optional): 'drift' (Gaussian random walk of q*, like a
shifting meta) or 'patch' (arm values re-shuffled half-way, like a balance patch).

Algorithms (all from the Unit 2 notes): explore-only (random), exploit-only
(greedy), fixed exploration + greedy, epsilon-greedy, optimistic greedy, UCB.
Value updates: sample average  Q_{n+1} = Q_n + (1/n)(R_n - Q_n)
               constant step   Q_{n+1} = Q_n + alpha (R_n - Q_n)   (non-stationary)
Regret      : sum_t [max_a q*_t(a) - q*_t(A_t)]   (expected / pseudo-regret)
Realised regret (notes' worked examples): T * max_a q*(a) - sum_t R_t
"""
from __future__ import annotations

import math
import time

import numpy as np

import rl_core as core

ALGORITHMS = {
    "random": "Explore only (uniform random)",
    "greedy": "Greedy (exploit only)",
    "explore_first": "Fixed exploration + greedy",
    "epsilon_greedy": "ε-greedy",
    "optimistic": "Optimistic greedy",
    "ucb": "UCB",
}


# --------------------------------------------------------------------------
# environment
# --------------------------------------------------------------------------
class BanditEnv:
    def __init__(self, means, reward_model: str = "bernoulli", sigma: float = 0.1, nonstationary: str = "none",
                 drift_sigma: float = 0.01, rounds: int = 1000, seed: int = 1):
        self.means0 = [float(m) for m in means]
        self.means = list(self.means0)
        self.k = len(means)
        self.reward_model = reward_model
        self.sigma = sigma
        self.nonstationary = nonstationary
        self.drift_sigma = drift_sigma
        self.rounds = rounds
        self.rng = core.Mulberry32(seed)
        self.drift_rng = core.Mulberry32(core.derive_seed(seed, 99))
        self.t = 0
        self.patch_perm = None
        if nonstationary == "patch":
            prng = core.Mulberry32(core.derive_seed(seed, 98))
            perm = list(range(self.k))
            for i in range(self.k - 1, 0, -1):     # Fisher–Yates
                j = prng.choice_index(i + 1)
                perm[i], perm[j] = perm[j], perm[i]
            # guarantee the best arm changes
            best = int(np.argmax(self.means0))
            if perm[best] == best and self.k > 1:
                perm[best], perm[(best + 1) % self.k] = perm[(best + 1) % self.k], perm[best]
            self.patch_perm = perm

    def pull(self, a: int) -> float:
        """Draws exactly one uniform (Bernoulli) or two (Gaussian) per pull."""
        m = self.means[a]
        if self.reward_model == "bernoulli":
            return 1.0 if self.rng.random() < m else 0.0
        return self.rng.normal(m, self.sigma)

    def advance(self):
        self.t += 1
        if self.nonstationary == "drift":
            self.means = [min(0.98, max(0.02, m + self.drift_rng.normal(0, self.drift_sigma))) for m in self.means]
        elif self.nonstationary == "patch" and self.t == self.rounds // 2:
            self.means = [self.means0[self.patch_perm[i]] for i in range(self.k)]


# --------------------------------------------------------------------------
# agents
# --------------------------------------------------------------------------
def _argmax(Q, rng=None) -> int:
    """Greedy argmax. Without rng: ties -> lowest index (deterministic, used to
    reproduce the notes' tables). With rng: ties broken uniformly at random, as
    the notes prescribe for the greedy method ("select randomly from the
    actions that are optimal")."""
    m = max(Q)
    if rng is None:
        for a, q in enumerate(Q):
            if q >= m - 1e-12:
                return a
    ties = [a for a, q in enumerate(Q) if q >= m - 1e-12]
    if len(ties) == 1:
        return ties[0]
    return ties[rng.choice_index(len(ties))]


def ucb_scores(Q, N, t: int, c: float) -> list[float]:
    """A_t = argmax_a [ Q_t(a) + c * sqrt( ln t / N_t(a) ) ]; N_t(a) = 0 -> +inf (maximising)."""
    out = []
    for q, n in zip(Q, N):
        out.append(float("inf") if n == 0 else q + c * math.sqrt(math.log(t) / n))
    return out


def run_bandit(algorithm: str, means, rounds: int = 1000, seed: int = 1, epsilon: float = 0.1, c: float = 1.414,
               q0: float = 0.0, alpha: float | None = None, explore_rounds: int = 50, reward_model: str = "bernoulli",
               sigma: float = 0.1, nonstationary: str = "none", drift_sigma: float = 0.01, record_q_every: int = 1) -> dict:
    """One run. The environment's reward stream depends only on `seed`, so
    every algorithm run with the same seed faces the same reward sequence per arm."""
    if algorithm not in ALGORITHMS:
        raise ValueError(f"unknown algorithm {algorithm}")
    env = BanditEnv(means, reward_model, sigma, nonstationary, drift_sigma, rounds, seed)
    agent_rng = core.Mulberry32(core.derive_seed(seed, 7))
    k = env.k
    init = q0 if algorithm == "optimistic" else 0.0
    Q = [float(init)] * k
    N = [0] * k
    arms, rewards, regret_curve, cum_reward, optimal_flags, explore_flags = [], [], [], [], [], []
    q_hist, ucb_last = [], None
    cum_r = cum_reg = 0.0
    n_explore = n_exploit = 0
    for t in range(1, rounds + 1):
        explored = False
        if algorithm == "random":
            a = agent_rng.choice_index(k); explored = True
        elif algorithm == "greedy" or algorithm == "optimistic":
            a = _argmax(Q, agent_rng)
        elif algorithm == "explore_first":
            if t <= explore_rounds:
                a = (t - 1) % k; explored = True      # round-robin exploration phase
            else:
                a = _argmax(Q, agent_rng)
        elif algorithm == "epsilon_greedy":
            if agent_rng.random() < epsilon:
                a = agent_rng.choice_index(k); explored = True
            else:
                a = _argmax(Q, agent_rng)
        else:  # ucb
            scores = ucb_scores(Q, N, t, c)
            a = _argmax(scores, agent_rng)
            m = max(Q)
            explored = Q[a] < m - 1e-12          # bonus made it pick a non-greedy arm
            ucb_last = {"t": t, "Q": list(Q), "N": list(N), "bonus": [None if n == 0 else c * math.sqrt(math.log(t) / n) for n in N],
                        "scores": [None if math.isinf(x) else x for x in scores], "chosen": a}
        best_mean = max(env.means)
        r = env.pull(a)
        N[a] += 1
        step = alpha if alpha else 1.0 / N[a]
        Q[a] = Q[a] + step * (r - Q[a])
        cum_r += r
        cum_reg += best_mean - env.means[a]
        arms.append(a); rewards.append(r)
        cum_reward.append(cum_r); regret_curve.append(cum_reg)
        optimal_flags.append(1 if env.means[a] >= best_mean - 1e-12 else 0)
        explore_flags.append(1 if explored else 0)
        if explored:
            n_explore += 1
        else:
            n_exploit += 1
        if record_q_every and (t % record_q_every == 0 or t == rounds):
            q_hist.append([round(x, 5) for x in Q])
        env.advance()
    best_final = max(env.means)
    return {"algorithm": algorithm, "name": ALGORITHMS[algorithm], "arms": arms, "rewards": rewards,
            "cumulative_reward": cum_reward, "cumulative_regret": regret_curve, "optimal": optimal_flags,
            "explore": explore_flags, "Q": Q, "N": N, "q_history": q_hist, "n_explore": n_explore,
            "n_exploit": n_exploit, "final_means": env.means, "ucb_last": ucb_last,
            "realised_regret": rounds * max(env.means0) - cum_r if nonstationary == "none" else None,
            "best_arm_final": int(np.argmax(env.means)), "total_reward": cum_r, "total_regret": cum_reg}


def run_arena(algorithms, means, rounds: int = 1000, runs: int = 50, seed: int = 1, **kw) -> dict:
    """Run several algorithms on the SAME seeded environments (run j of every
    algorithm uses environment seed derive_seed(seed, j)) and average."""
    t0 = time.perf_counter()
    out = {}
    for alg in algorithms:
        reg = np.zeros(rounds); rew = np.zeros(rounds); opt = np.zeros(rounds); expl = np.zeros(rounds)
        totals = []
        first = None
        for j in range(runs):
            r = run_bandit(alg, means, rounds, core.derive_seed(seed, j),
                           record_q_every=(max(1, rounds // 200) if j == 0 else 0), **kw)
            reg += np.array(r["cumulative_regret"]); rew += np.array(r["cumulative_reward"])
            opt += np.array(r["optimal"]); expl += np.array(r["explore"])
            totals.append(r["total_reward"])
            if j == 0:
                first = r
        n = float(runs)
        stride = max(1, rounds // 200)
        idx = list(range(stride - 1, rounds, stride))
        if idx[-1] != rounds - 1:
            idx.append(rounds - 1)
        tot = np.array(totals)
        out[alg] = {
            "name": ALGORITHMS[alg], "rounds_index": [i + 1 for i in idx],
            "mean_cumulative_regret": [float(reg[i] / n) for i in idx],
            "mean_cumulative_reward": [float(rew[i] / n) for i in idx],
            "pct_optimal": [float(_window(opt, i, stride) / n) for i in idx],
            "explore_rate": [float(_window(expl, i, stride) / n) for i in idx],
            "final_regret": float(reg[-1] / n), "final_reward": float(rew[-1] / n),
            "final_reward_std": float(tot.std(ddof=1)) if runs > 1 else 0.0,
            "final_reward_ci95": float(1.96 * tot.std(ddof=1) / math.sqrt(runs)) if runs > 1 else 0.0,
            "final_pct_optimal": float(opt[-max(1, rounds // 10):].mean() / n),
            "example_run": dict({k: first[k] for k in ("Q", "N", "q_history", "n_explore", "n_exploit",
                                                        "ucb_last", "final_means", "realised_regret", "total_reward", "total_regret")},
                                arms=first["arms"][:min(rounds, 400)], rewards=first["rewards"][:min(rounds, 400)],
                                q_history_stride=max(1, rounds // 200)),
        }
    return {"results": out, "rounds": rounds, "runs": runs, "seed": seed, "means": list(means), "settings": kw,
            "seconds": time.perf_counter() - t0, "provenance": {"arm_means": "DERIVED FEATURE", "rewards": "SIMULATED"}}


def _window(arr, i, w):
    lo = max(0, i - w + 1)
    return arr[lo:i + 1].mean()


# --------------------------------------------------------------------------
# arms from the map pool (connects the bandit to the tournament builder)
# --------------------------------------------------------------------------
def map_arms(maps: list[dict], roster: list[dict], k: int | None = None) -> list[dict]:
    """Expected satisfaction q*(map) for a roster: the tournament builder's
    map-fit score of the first k maps, min-max rescaled to [0.25, 0.85] over
    those k arms (highest fit = highest mean)."""
    from tournament_builder import map_fit_score
    pool = maps[:k] if k else maps
    scored = [(map_fit_score(m, roster), m) for m in pool]
    fits = [f for f, _ in scored]
    lo, hi = min(fits), max(fits)
    arms = []
    for f, m in scored:
        q = 0.25 + 0.60 * ((f - lo) / (hi - lo) if hi > lo else 0.5)
        arms.append({"map_name": m["map_name"], "fit_score": round(f, 4), "q_star": round(q, 4),
                     "recommended_playstyle": m.get("recommended_playstyle"), "sniper_friendly": m.get("sniper_friendly")})
    return arms


# --------------------------------------------------------------------------
# Unit 2 worked numericals, recomputed by the same code paths
# --------------------------------------------------------------------------
def notes_exploration_example(select_probs, win_probs, select_randoms, reward_randoms):
    """'Example — Exploration' (Unit 2 numericals): arms chosen by inverse CDF
    of a fixed selection distribution; reward = 1 if random <= P_win."""
    cdf = np.cumsum(select_probs).tolist()
    rows, total = [], 0
    for i, (u, v) in enumerate(zip(select_randoms, reward_randoms)):
        arm = next(j for j, c in enumerate(cdf) if u <= c + 1e-12)
        r = 1 if v <= win_probs[arm] else 0
        total += r
        rows.append({"iteration": i + 1, "u_select": u, "machine": f"M{arm + 1}", "p_win": win_probs[arm], "u_reward": v, "reward": r})
    best = max(win_probs) * len(select_randoms)
    return {"cdf": cdf, "rows": rows, "total_reward": total, "max_expected": best, "regret": round(best - total, 6)}


def notes_exploitation_example(select_probs, win_probs, first_random, reward_randoms):
    cdf = np.cumsum(select_probs).tolist()
    arm = next(j for j, c in enumerate(cdf) if first_random <= c + 1e-12)
    rows, total = [], 0
    for i, v in enumerate(reward_randoms):
        r = 1 if v <= win_probs[arm] else 0
        total += r
        rows.append({"iteration": i + 1, "machine": f"M{arm + 1}", "u_reward": v, "reward": r})
    best = max(win_probs) * len(reward_randoms)
    return {"machine": f"M{arm + 1}", "rows": rows, "total_reward": total, "max_expected": best, "regret": round(best - total, 6)}


def notes_fixed_exploration_example(win_probs, explore_counts, explore_wins, reward_randoms, total_steps):
    Q = [explore_wins[i] / explore_counts[i] if explore_counts[i] else 0.0 for i in range(len(win_probs))]
    best = _argmax(Q)
    rows, total = [], 0
    for i, v in enumerate(reward_randoms):
        r = 1 if v <= win_probs[best] else 0
        total += r
        rows.append({"iteration": sum(explore_counts) + i + 1, "machine": f"M{best + 1}", "u_reward": v, "reward": r})
    total_all = total + sum(explore_wins)
    mx = max(win_probs) * total_steps
    return {"Q_after_exploration": Q, "exploited_machine": f"M{best + 1}", "rows": rows, "exploitation_reward": total,
            "total_reward": total_all, "max_expected": mx, "regret": round(mx - total_all, 6)}


def notes_epsilon_greedy_assignment(win_probs, epsilon, x_randoms, explore_choices, reward_randoms):
    """'Numerical on Epsilon greedy — Assignment 1'. Explore when x <= eps
    (machines taken from explore_choices in order); otherwise greedy on the
    sample-average Q (ties -> lowest index). Reward = 1 if random <= P_win."""
    k = len(win_probs)
    Q, N = [0.0] * k, [0] * k
    rows, total, ex_i = [], 0, 0
    for t, (x, v) in enumerate(zip(x_randoms, reward_randoms)):
        if x <= epsilon:
            a = explore_choices[ex_i]; ex_i += 1; mode = "explore"
        else:
            a = _argmax(Q); mode = "exploit"
        r = 1 if v <= win_probs[a] else 0
        N[a] += 1
        Q[a] = Q[a] + (r - Q[a]) / N[a]
        total += r
        rows.append({"step": t + 1, "x": x, "mode": mode, "machine": f"M{a + 1}", "u_reward": v, "reward": r,
                     "Q": [round(q, 4) for q in Q], "N": list(N)})
    return {"rows": rows, "total_reward": total, "final_Q": Q}


def notes_ucb_example(Q, N, t, c=math.sqrt(2)):
    scores = ucb_scores(Q, N, t, c)
    return {"t": t, "c": c, "Q": Q, "N": N, "bonus": [c * math.sqrt(math.log(t) / n) for n in N],
            "scores": scores, "chosen_arm": _argmax(scores) + 1}


def notes_epsilon_case_detection(actions, rewards, k):
    """Sutton Exercise 2.2 (shown in the Unit 2 notes): with sample-average
    estimates from Q1 = 0, on which steps did the epsilon case DEFINITELY
    occur (action not greedy) and on which could it POSSIBLY have occurred?"""
    Q, N = [0.0] * k, [0] * k
    rows = []
    for t, (a, r) in enumerate(zip(actions, rewards), start=1):
        idx = a - 1
        m = max(Q)
        greedy = [i + 1 for i, q in enumerate(Q) if abs(q - m) < 1e-12]
        verdict = "possibly (greedy action, but a random pick could also select it)" if a in greedy else "DEFINITELY exploration"
        rows.append({"t": t, "action": a, "reward": r, "Q_before": [round(q, 4) for q in Q], "greedy_set": greedy,
                     "verdict": verdict, "definitely_random": a not in greedy})
        N[idx] += 1
        Q[idx] += (r - Q[idx]) / N[idx]
    return {"rows": rows, "definitely": [x["t"] for x in rows if x["definitely_random"]],
            "possibly": [x["t"] for x in rows]}


def notes_constant_step_example(alpha, trials):
    """'Example — Incremental formula' (Unit 2): Q_{n+1} = Q_n + eta (r - Q_n)."""
    Q = {}
    rows = []
    for arm, r in trials:
        old = Q.get(arm, 0.0)
        Q[arm] = old + alpha * (r - old)
        rows.append({"arm": arm, "reward": r, "old_Q": old, "new_Q": round(Q[arm], 6)})
    nxt = max(Q, key=lambda a: (Q[a], -int(a[1:]) if a[1:].isdigit() else 0))
    return {"rows": rows, "Q": Q, "next_greedy": nxt}


def all_notes_numericals() -> list[dict]:
    """Every Unit 2 worked example whose answer is printed in the notes,
    recomputed by this module and checked against that answer."""
    out = []
    ex = notes_exploration_example([0.3, 0.4, 0.2, 0.1], [0.6, 0.4, 0.7, 0.9],
                                   [0.68, 0.55, 0.23, 0.95, 0.8], [0.6, 0.2, 0.3, 0.4, 0.9])
    out.append({"id": "u2-exploration", "title": "Exploration (random selection via CDF)", "unit": 2,
                "result": ex, "notes_answer": {"total_reward": 3, "regret": 1.5},
                "match": ex["total_reward"] == 3 and abs(ex["regret"] - 1.5) < 1e-9})
    ex = notes_exploitation_example([0.3, 0.4, 0.2, 0.1], [0.6, 0.4, 0.7, 0.9], 0.65, [0.6, 0.2, 0.3, 0.5, 0.9])
    out.append({"id": "u2-exploitation", "title": "Exploitation (one machine for all steps)", "unit": 2,
                "result": ex, "notes_answer": {"machine": "M2", "total_reward": 2, "regret": 2.5},
                "match": ex["machine"] == "M2" and ex["total_reward"] == 2 and abs(ex["regret"] - 2.5) < 1e-9})
    ex = notes_fixed_exploration_example([0.6, 0.4, 0.7, 0.9], [1, 0, 3, 0], [1, 0, 2, 0],
                                         [0.33, 0.25, 0.55, 0.45, 0.6, 0.8], 10)
    out.append({"id": "u2-fixed-exploration", "title": "Fixed exploration + exploitation", "unit": 2,
                "result": ex, "notes_answer": {"exploited_machine": "M1", "total_reward": 8, "regret": 1},
                "match": ex["exploited_machine"] == "M1" and ex["total_reward"] == 8 and abs(ex["regret"] - 1) < 1e-9})
    ex = notes_ucb_example([0.55, 0.63, 0.61, 0.40], [3, 4, 3, 2], 12)
    out.append({"id": "u2-ucb-12", "title": "UCB1 after 12 iterations (4 arms)", "unit": 2, "result": ex,
                "notes_answer": {"chosen_arm": 4, "scores": [1.837, 1.745, 1.897, 1.976]},
                "match": ex["chosen_arm"] == 4 and all(abs(a - b) < 1e-3 for a, b in zip(ex["scores"], [1.837, 1.745, 1.897, 1.976]))})
    ex = notes_ucb_example([1.73, 1.83, 1.89, 1.55], [25, 20, 30, 25], 100)
    out.append({"id": "u2-ucb-100", "title": "UCB after 100 iterations (4 arms)", "unit": 2, "result": ex,
                "notes_answer": {"chosen_arm": 2}, "match": ex["chosen_arm"] == 2})
    ex = notes_epsilon_case_detection([1, 2, 2, 2, 3], [-1, 1, -2, 2, 0], 4)
    out.append({"id": "u2-eps-detect", "title": "ε-greedy: when did exploration definitely occur? (Ex. 2.2)", "unit": 2,
                "result": ex, "notes_answer": {"definitely": [4, 5], "possibly": [1, 2, 3, 4, 5]},
                "match": ex["definitely"] == [4, 5]})
    ex = notes_constant_step_example(0.2, [("a1", 2), ("a2", 0.8), ("a1", 0), ("a2", 0.8)])
    out.append({"id": "u2-const-step", "title": "Incremental update with constant step η = 0.2", "unit": 2,
                "result": ex, "notes_answer": {"after_trial_1": 0.4, "after_trial_2": 0.16},
                "match": abs(ex["rows"][0]["new_Q"] - 0.4) < 1e-9 and abs(ex["rows"][1]["new_Q"] - 0.16) < 1e-9})
    ex = notes_epsilon_greedy_assignment([0.4, 0.7], 0.2, [0.1, 0.3, 0.4, 0.2, 0.5, 0.1, 0.6, 0.7, 0.4, 0.9], [0, 1, 1],
                                         [0.3, 0.45, 0.75, 0.9, 0.2, 0.55, 0.6, 0.8, 0.2, 0.15])
    out.append({"id": "u2-eps-assignment", "title": "ε-greedy Assignment 1 (2 machines, ε = 0.2)", "unit": 2,
                "result": ex, "notes_answer": None, "match": None,
                "note": "The notes pose this as an assignment without a printed answer; ties in Q are broken toward M1."})
    return out
