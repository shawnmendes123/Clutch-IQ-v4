"""
monte_carlo.py — UNIT 5: Monte Carlo prediction and control (model-free).

Monte Carlo methods learn from COMPLETE EPISODES only — no p(s'|s,a) is used
anywhere in this file. Each episode is a list of transitions
(S_t, A_t, R_{t+1}, S_{t+1}, done) and returns are computed backwards:

        G_t = R_{t+1} + gamma * G_{t+1},    G_T = 0

Implemented (Unit 5 notes / Sutton & Barto ch. 5):
  * First-visit MC prediction            V(s) = mean of G at first visits to s
  * Every-visit MC prediction            V(s) = mean of G at every visit to s
  * MC estimation of action values       Q(s,a) = mean of G after first visits to (s,a)
  * MC control with Exploring Starts (MC ES)
  * On-policy first-visit MC control with an epsilon-soft policy
  * Off-policy prediction via importance sampling (ordinary + weighted)
        rho_{t:T-1} = prod_k pi(A_k|S_k) / b(A_k|S_k)
    plus two variance-reduction estimators from the syllabus line
    "importance sampling on truncated returns": discounting-aware IS on flat
    partial returns (Sutton §5.8) and per-decision IS (§5.9)
  * Off-policy MC control (weighted importance sampling, Sutton §5.7)

Data honesty: episodes produced by rl_core.generate_episode are SIMULATED.
Episodes built from an uploaded dataset (episodes_from_engineered) are REAL
DATA but carry no logged actions, so only prediction is possible on them.
"""
from __future__ import annotations

import time

import numpy as np

import rl_core as core


# --------------------------------------------------------------------------
# generic helpers (work for ANY finite state space, incl. the notes' A/B MRP)
# --------------------------------------------------------------------------
def _sar(ep):
    """Normalise an episode into lists (states, actions, rewards)."""
    if isinstance(ep, core.Episode):
        return ([t.state for t in ep.transitions], [t.action for t in ep.transitions],
                [t.reward for t in ep.transitions])
    if isinstance(ep, dict):
        return ep["states"], ep.get("actions") or [None] * len(ep["states"]), ep["rewards"]
    states, rewards = ep
    return list(states), [None] * len(states), list(rewards)


def episode_returns(rewards, gamma):
    return core.returns_from_rewards(list(rewards), gamma)


def first_visit_prediction(episodes, gamma: float, n_states: int) -> dict:
    """First-visit MC prediction (Unit 5 pseudocode, incremental sums)."""
    N = np.zeros(n_states)
    Gsum = np.zeros(n_states)
    for ep in episodes:
        S, _, R = _sar(ep)
        G = episode_returns(R, gamma)
        seen = set()
        for t, s in enumerate(S):
            if s in seen:
                continue
            seen.add(s)
            N[s] += 1
            Gsum[s] += G[t]
    V = np.divide(Gsum, N, out=np.zeros(n_states), where=N > 0)
    return {"V": V, "N": N}


def every_visit_prediction(episodes, gamma: float, n_states: int) -> dict:
    N = np.zeros(n_states)
    Gsum = np.zeros(n_states)
    for ep in episodes:
        S, _, R = _sar(ep)
        G = episode_returns(R, gamma)
        for t, s in enumerate(S):
            N[s] += 1
            Gsum[s] += G[t]
    V = np.divide(Gsum, N, out=np.zeros(n_states), where=N > 0)
    return {"V": V, "N": N}


def mc_action_values(episodes, gamma: float, n_states: int, n_actions: int, first_visit: bool = True) -> dict:
    N = np.zeros((n_states, n_actions))
    Gsum = np.zeros((n_states, n_actions))
    for ep in episodes:
        S, A, R = _sar(ep)
        G = episode_returns(R, gamma)
        seen = set()
        for t, (s, a) in enumerate(zip(S, A)):
            if first_visit and (s, a) in seen:
                continue
            seen.add((s, a))
            N[s, a] += 1
            Gsum[s, a] += G[t]
    Q = np.divide(Gsum, N, out=np.zeros((n_states, n_actions)), where=N > 0)
    return {"Q": Q, "N": N}


def _mae(V, ref, mask):
    if mask.sum() == 0:
        return None
    return float(np.mean(np.abs(V[mask] - ref[mask])))


def episode_table(ep: core.Episode, gamma: float) -> list[dict]:
    """Per-step breakdown of one episode: s, a, r, G_t (for the UI)."""
    S, A, R = _sar(ep)
    G = episode_returns(R, gamma)
    rows = []
    for t in range(len(S)):
        tr = ep.transitions[t]
        rows.append({"t": t, "state": S[t], "state_label": core.state_short(S[t]), "action": A[t],
                     "action_name": core.ACTION_NAMES[A[t]] if A[t] is not None else None,
                     "reward": round(R[t], 4), "next_state": tr.next_state, "done": tr.done,
                     "G_t": round(G[t], 4), "discount": round(gamma ** t, 6), "info": tr.info})
    return rows


# --------------------------------------------------------------------------
# MC prediction on the Clutch IQ simulator
# --------------------------------------------------------------------------
DEFAULT_CHECKPOINTS = [10, 25, 50, 100, 250, 500, 1000, 2000, 5000, 10000, 20000]


def mc_prediction_run(policy, n_episodes: int, gamma: float, seed: int, start_dist: str = "uniform",
                      reference_V: np.ndarray | None = None, checkpoints=None) -> dict:
    """Generate n SIMULATED episodes under `policy` and estimate V_pi with both
    first-visit and every-visit MC; track MAE against a reference (DP)."""
    t0 = time.perf_counter()
    nS = core.N_STATES_WITH_TERMINAL
    cps = sorted(set([c for c in (checkpoints or DEFAULT_CHECKPOINTS) if c <= n_episodes] + [n_episodes]))
    Nf, Gf = np.zeros(nS), np.zeros(nS)
    Ne, Ge = np.zeros(nS), np.zeros(nS)
    curve = []
    lengths, ep_returns = [], []
    sample = None
    for i in range(n_episodes):
        ep = core.generate_episode(policy, core.derive_seed(seed, i), start_dist=start_dist, record_info=(i == 0))
        if i == 0:
            sample = ep
        S, _, R = _sar(ep)
        G = episode_returns(R, gamma)
        lengths.append(len(S))
        ep_returns.append(G[0])
        seen = set()
        for t, s in enumerate(S):
            Ne[s] += 1
            Ge[s] += G[t]
            if s not in seen:
                seen.add(s)
                Nf[s] += 1
                Gf[s] += G[t]
        if (i + 1) in cps:
            Vf = np.divide(Gf, Nf, out=np.zeros(nS), where=Nf > 0)
            Ve = np.divide(Ge, Ne, out=np.zeros(nS), where=Ne > 0)
            row = {"episodes": i + 1, "coverage": float((Nf[:core.N_STATES] > 0).mean())}
            if reference_V is not None:
                mask = Nf[:core.N_STATES] > 0
                row["mae_first_visit"] = _mae(Vf[:core.N_STATES], reference_V[:core.N_STATES], mask)
                row["mae_every_visit"] = _mae(Ve[:core.N_STATES], reference_V[:core.N_STATES], mask)
            curve.append(row)
    Vf = np.divide(Gf, Nf, out=np.zeros(nS), where=Nf > 0)
    Ve = np.divide(Ge, Ne, out=np.zeros(nS), where=Ne > 0)
    return {"V_first_visit": Vf, "V_every_visit": Ve, "N_first_visit": Nf, "N_every_visit": Ne,
            "curve": curve, "sample_episode": episode_table(sample, gamma) if sample else [],
            "mean_return": float(np.mean(ep_returns)), "std_return": float(np.std(ep_returns)),
            "mean_length": float(np.mean(lengths)), "n_episodes": n_episodes,
            "seconds": time.perf_counter() - t0, "provenance": "SIMULATED"}


# --------------------------------------------------------------------------
# MC control
# --------------------------------------------------------------------------
def _policy_value_under_model(actions, model, gamma, start_dist):
    """J(pi) = sum_s d0(s) V_pi(s) computed on the (estimated) model — used
    only to *monitor* learning progress, never by the MC learner itself."""
    from dynamic_programming import policy_evaluation
    ev = policy_evaluation(model, np.array(list(actions) + [1] * (model.nS - len(actions))), gamma, 1e-6, 500)
    d0 = np.array(core.start_distribution(start_dist) + [0.0])
    return float(d0 @ ev["V"]), ev["V"]


def mc_control(variant: str = "on_policy", n_episodes: int = 5000, gamma: float = 0.9, epsilon: float = 0.1,
               seed: int = 7, start_dist: str = "uniform", reference_actions=None, model=None,
               n_checkpoints: int = 20, behaviour_epsilon: float = 0.3) -> dict:
    """variant: 'es'         Monte Carlo Exploring Starts
                'on_policy'  first-visit on-policy MC control, epsilon-soft
                'off_policy' off-policy MC control, weighted importance sampling
    """
    t0 = time.perf_counter()
    nS, nA = core.N_STATES_WITH_TERMINAL, core.N_ACTIONS
    Q = np.zeros((nS, nA))
    Nsa = np.zeros((nS, nA))     # visit counts (first visits) / cumulative weights for off-policy
    Csa = np.zeros((nS, nA))     # off-policy cumulative IS weights C(s,a)
    greedy = np.ones(nS, dtype=int)   # initial arbitrary policy: KEEP everywhere
    every = max(1, n_episodes // n_checkpoints)
    history = []
    ep_returns = []
    rng_es = core.Mulberry32(core.derive_seed(seed, 777))
    updates_used = 0
    for i in range(n_episodes):
        ep_seed = core.derive_seed(seed, i)
        if variant == "es":
            s0 = rng_es.choice_index(core.N_STATES)
            a0 = rng_es.choice_index(nA)
            ep = core.generate_episode(core.deterministic_policy(greedy.tolist()), ep_seed, start_state=s0, first_action=a0)
        elif variant == "on_policy":
            ep = core.generate_episode(core.epsilon_soft(greedy.tolist(), epsilon), ep_seed, start_dist=start_dist)
        elif variant == "off_policy":
            bpol = core.epsilon_soft(greedy.tolist(), behaviour_epsilon)
            ep = core.generate_episode(bpol, ep_seed, start_dist=start_dist)
        else:
            raise ValueError("variant must be es | on_policy | off_policy")
        S, A, R = _sar(ep)
        ep_returns.append(core.discounted_return(R, gamma))
        if variant in ("es", "on_policy"):
            G = episode_returns(R, gamma)
            first = {}
            for t, (s, a) in enumerate(zip(S, A)):
                if (s, a) not in first:
                    first[(s, a)] = t
            for (s, a), t in first.items():
                Nsa[s, a] += 1
                Q[s, a] += (G[t] - Q[s, a]) / Nsa[s, a]       # incremental sample average
                best = int(np.argmax(Q[s]))
                greedy[s] = best
        else:
            G, W = 0.0, 1.0
            for t in range(len(S) - 1, -1, -1):
                s, a, r = S[t], A[t], R[t]
                G = gamma * G + r
                Csa[s, a] += W
                Nsa[s, a] += 1
                Q[s, a] += (W / Csa[s, a]) * (G - Q[s, a])
                greedy[s] = int(np.argmax(Q[s]))
                updates_used += 1
                if a != greedy[s]:
                    break
                W = W / bpol[s][a]
        if (i + 1) % every == 0 or i + 1 == n_episodes:
            row = {"episodes": i + 1, "mean_return_window": float(np.mean(ep_returns[-every:]))}
            if reference_actions is not None:
                row["agreement_with_reference"] = float(np.mean(greedy[:core.N_STATES] == np.asarray(reference_actions)[:core.N_STATES]))
            if model is not None:
                row["greedy_policy_value"] = _policy_value_under_model(greedy[:core.N_STATES], model, gamma, "session_start")[0]
            history.append(row)
    out = {"variant": variant, "Q": Q, "N": Nsa, "actions": greedy[:core.N_STATES].tolist(),
           "history": history, "n_episodes": n_episodes, "gamma": gamma, "epsilon": epsilon,
           "behaviour_epsilon": behaviour_epsilon if variant == "off_policy" else None,
           "seconds": time.perf_counter() - t0, "provenance": "SIMULATED",
           "unvisited_pairs": int((Nsa[:core.N_STATES] == 0).sum())}
    if variant == "off_policy":
        out["cumulative_weights"] = Csa
        out["transitions_used"] = updates_used
    return out


# --------------------------------------------------------------------------
# Off-policy prediction via importance sampling
# --------------------------------------------------------------------------
def coverage_check(target, behaviour) -> dict:
    """Assumption of coverage: pi(a|s) > 0  =>  b(a|s) > 0."""
    bad = []
    for s in range(core.N_STATES):
        for a in range(core.N_ACTIONS):
            if target[s][a] > 0 and behaviour[s][a] <= 0:
                bad.append({"state": s, "action": a})
    return {"satisfied": len(bad) == 0, "violations": bad[:20], "n_violations": len(bad)}


def off_policy_prediction(target, behaviour, n_episodes: int, gamma: float, seed: int,
                          start_dist: str = "session_start", reference_value: float | None = None,
                          reference_V: np.ndarray | None = None, checkpoints=None,
                          logged_episodes=None) -> dict:
    """Estimate V_pi using episodes generated by b.

    Ordinary IS :  V(s) = sum_{t in T(s)} rho_{t:T-1} G_t / |T(s)|
    Weighted IS :  V(s) = sum rho G / sum rho
    (first-visit sets T(s)). Also returns the start-state estimate of
    J(pi) = E[G_0] — 'would the NEW matchmaking policy be better?' — estimated
    purely from sessions logged under the OLD policy.
    """
    t0 = time.perf_counter()
    cov = coverage_check(target, behaviour)
    if not cov["satisfied"]:
        return {"error": "coverage assumption violated: target takes actions the behaviour policy never takes",
                "coverage": cov}
    nS = core.N_STATES_WITH_TERMINAL
    num = np.zeros(nS); den_w = np.zeros(nS); cnt = np.zeros(nS)
    rhos, G0s, wG0 = [], [], []
    pdis, da_num, da_den = [], [], []
    cps = sorted(set([c for c in (checkpoints or DEFAULT_CHECKPOINTS) if c <= n_episodes] + [n_episodes]))
    curve = []
    samples = []
    on_policy_like = 0
    for i in range(n_episodes):
        if logged_episodes is not None:
            ep = logged_episodes[i]
        else:
            ep = core.generate_episode(behaviour, core.derive_seed(seed, i), start_dist=start_dist)
        S, A, R = _sar(ep)
        G = episode_returns(R, gamma)
        T = len(S)
        # rho_{t:T-1} for every t, computed backwards
        rho_t = [0.0] * T
        acc = 1.0
        for t in range(T - 1, -1, -1):
            acc *= target[S[t]][A[t]] / behaviour[S[t]][A[t]]
            rho_t[t] = acc
        seen = set()
        for t, s in enumerate(S):
            if s in seen:
                continue
            seen.add(s)
            num[s] += rho_t[t] * G[t]
            den_w[s] += rho_t[t]
            cnt[s] += 1
        rhos.append(rho_t[0]); G0s.append(G[0]); wG0.append(rho_t[0] * G[0])
        # per-decision IS (Sutton 5.9): reward R_{k+1} weighted only by rho_{0:k}
        pref = []
        acc = 1.0
        for k in range(T):
            acc *= target[S[k]][A[k]] / behaviour[S[k]][A[k]]
            pref.append(acc)
        pdis.append(sum((gamma ** k) * pref[k] * R[k] for k in range(T)))
        # discounting-aware IS on truncated (flat partial) returns (Sutton 5.8)
        flat = np.cumsum(R)
        n_, d_ = 0.0, 0.0
        for h in range(1, T):
            w = (1 - gamma) * gamma ** (h - 1) * pref[h - 1]
            n_ += w * flat[h - 1]; d_ += w
        w = gamma ** (T - 1) * pref[T - 1]
        n_ += w * flat[T - 1]; d_ += w
        da_num.append(n_); da_den.append(d_)
        if rho_t[0] > 0:
            on_policy_like += 1
        if i < 6:
            samples.append({"episode": i, "length": T, "G0": round(G[0], 4), "rho": rho_t[0],
                            "steps": [{"t": t, "state": core.state_short(S[t]), "action": core.ACTION_NAMES[A[t]],
                                       "pi": target[S[t]][A[t]], "b": behaviour[S[t]][A[t]],
                                       "ratio": target[S[t]][A[t]] / behaviour[S[t]][A[t]], "reward": round(R[t], 4)}
                                      for t in range(min(T, 12))]})
        if (i + 1) in cps:
            n = i + 1
            ois = float(np.sum(wG0) / n)
            wsum = float(np.sum(rhos))
            wis = float(np.sum(wG0) / wsum) if wsum > 0 else None
            dd = float(np.sum(da_den))
            row = {"episodes": n, "ordinary_is": ois, "weighted_is": wis,
                   "per_decision_is": float(np.mean(pdis)),
                   "discounting_aware_ois": float(np.mean(da_num)),
                   "discounting_aware_wis": float(np.sum(da_num) / dd) if dd > 0 else None,
                   "behaviour_mean_return": float(np.mean(G0s))}
            curve.append(row)
    rhos_a = np.array(rhos)
    V_ois = np.divide(num, cnt, out=np.zeros(nS), where=cnt > 0)
    V_wis = np.divide(num, den_w, out=np.zeros(nS), where=den_w > 0)
    ess = float(rhos_a.sum() ** 2 / (rhos_a ** 2).sum()) if (rhos_a ** 2).sum() > 0 else 0.0
    out = {
        "J_ordinary_is": curve[-1]["ordinary_is"], "J_weighted_is": curve[-1]["weighted_is"],
        "J_per_decision_is": curve[-1]["per_decision_is"],
        "J_discounting_aware_ois": curve[-1]["discounting_aware_ois"],
        "J_discounting_aware_wis": curve[-1]["discounting_aware_wis"],
        "reliability": ("OK" if ess >= 0.05 * n_episodes and ess >= 100 else
                        "LOW — effective sample size is small; importance weights are heavy-tailed, treat the estimate with caution"),
        "J_behaviour_on_policy": float(np.mean(G0s)), "reference_J_target": reference_value,
        "curve": curve, "V_ordinary_is": V_ois, "V_weighted_is": V_wis, "visits": cnt,
        "rho_stats": {"mean": float(rhos_a.mean()), "max": float(rhos_a.max()), "fraction_nonzero": float((rhos_a > 0).mean()),
                      "effective_sample_size": ess, "histogram": _hist(rhos_a)},
        "coverage": cov, "sample_episodes": samples, "n_episodes": n_episodes,
        "seconds": time.perf_counter() - t0, "provenance": "SIMULATED" if logged_episodes is None else "LOGGED",
    }
    if reference_V is not None:
        mask = cnt[:core.N_STATES] > 0
        out["mae_ordinary_is"] = _mae(V_ois[:core.N_STATES], reference_V[:core.N_STATES], mask)
        out["mae_weighted_is"] = _mae(V_wis[:core.N_STATES], reference_V[:core.N_STATES], (den_w[:core.N_STATES] > 0))
    return out


def _hist(x: np.ndarray, bins: int = 12) -> dict:
    nz = x[x > 0]
    zero = int((x == 0).sum())
    if len(nz) == 0:
        return {"zero": zero, "edges": [], "counts": []}
    lo, hi = np.log10(nz.min()), np.log10(nz.max())
    if hi - lo < 1e-9:
        hi = lo + 1
    edges = np.logspace(lo, hi, bins + 1)
    counts, _ = np.histogram(nz, edges)
    return {"zero": zero, "edges": edges.tolist(), "counts": counts.tolist(), "log_scale": True}


# --------------------------------------------------------------------------
# REAL DATA: episodes from an engineered uploaded dataset (prediction only)
# --------------------------------------------------------------------------
def episodes_from_engineered(engineered) -> tuple[list, list[str]]:
    """Each (player, session) in the uploaded CSV becomes one episode of
    (state, reward) pairs. States are bucketed DERIVED FEATURES; rewards use
    the same formula with the dataset's proxies. No matchmaking actions are
    present in such data, so actions are None and only prediction is valid."""
    notes = []
    episodes = []
    if engineered is None or len(engineered) == 0:
        return [], ["No engineered data."]
    group_cols = [c for c in ("player_ref", "session_key") if c in engineered.columns]
    for _, g in engineered.groupby(group_cols, sort=False):
        g = g.sort_values("match_index_in_session") if "match_index_in_session" in g.columns else g
        states, rewards = [], []
        n = len(g)
        for j, (_, row) in enumerate(g.iterrows()):
            s = core.state_from_features(float(row.get("fatigue_index", 0) or 0), float(row.get("tilt_score", 0) or 0),
                                         int(row.get("win_streak", 0) or 0))
            rq = bool(row.get("rage_quit_proxy", 0))
            close = float(row.get("close_match_baseline", 0.5) or 0.5)
            cont = 0.0 if (j == n - 1 and rq) else 1.0
            states.append(s)
            rewards.append(core.reward_function(cont, close, rq))
        if states:
            episodes.append({"states": states, "rewards": rewards, "actions": None})
    notes.append("Episodes = (player, session) groups from the uploaded file; states are bucketed derived features (fatigue, tilt, streak).")
    notes.append("Rewards use the project's reward formula with dataset proxies: closeness defaults to 0.5 (FALLBACK) and rage quit is the feature-engine proxy (loss + tilt > 0.5 on the last match).")
    notes.append("The dataset contains no matchmaking actions or behaviour-policy probabilities, so MC control and importance sampling are NOT run on it (nothing is fabricated).")
    return episodes, notes
