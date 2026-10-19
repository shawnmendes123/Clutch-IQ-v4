# Viva guide — Clutch IQ v4

Short, defensible answers. Every answer can be demonstrated live in the app
(screen names in **bold**); switch on **Viva mode** in the top bar to show the
unit, formula, live worked example and source location on each screen.

## The 30-second pitch
Clutch IQ is a competitive-gaming matchmaker that reasons about more than rank:
it balances teams on skill and chemistry, and a reinforcement-learning
controller adjusts each player's lobby difficulty from their fatigue, tilt and
streak. We formalised that controller as a 48-state finite MDP and solved/learned
it with dynamic programming, Monte Carlo and Q-learning on the *same*
simulated environment, used a multi-armed bandit for the one-step map choice,
and evaluated everything on identical seeded episodes.

## Formulation (Unit 1, Unit 3)
- **Agent** — the session controller. **Environment** — one player's session (simulated). **State** — (fatigue, tilt, streak) buckets, 4×4×3 = 48 + END. **Actions** — LOWER / KEEP / RAISE. **Reward** — 1.0·(match completed) + 1.2·closeness − 3.0·(rage quit). **Policy** — π(a|s), a 49×3 matrix. → **Mathematical formulation**, **Live session**.
- **Why an MDP?** Decisions are sequential, outcomes stochastic, rewards additive, and the next state depends only on the current bucketed state and action (Markov property — exact for our simulator, an assumption for real players).
- **Discounted return** G_t = R_{t+1} + γG_{t+1}. **Live session** computes it backwards and plots G₀ against γ for the same episode. γ = 0 is myopic; γ → 1 far-sighted (E3: the far-sighted π* lowers difficulty more and has fewer rage quits).
- **Transition probabilities** p(s′|s,a) are *estimated* by 1000 simulator rollouts per (s,a) — labelled "model estimate". We can measure their error only because we built the simulator (E1).
- **V vs Q.** V_π(s): expected return from s following π. Q_π(s,a): first take a, then follow π. **MDP explorer** shows both and the full Bellman sum term by term.

## Bandits (Unit 2)
- **Why a bandit for maps?** The map does not change the player's next state, so there is nothing to plan — maximise E[R | map]. Session control is kept separate; the two are never ranked together.
- **Exploration vs exploitation** — greedy can lock onto a mediocre map; ε-greedy explores at random with probability ε; optimistic Q₀ makes every arm look good until tried; UCB adds c√(ln t/N): untried arms first, then good-or-uncertain arms.
- **Regret** = Σ(max q* − q*(A_t)). The app also shows the notes' realised regret T·max q* − ΣR.
- **Non-stationary** — constant α weights recent rewards exponentially (α(1−α)^{n−i}); after a "balance patch" it recovers much faster than sample averages (E7).
- The notes' numericals (exploration/exploitation/fixed-exploration examples, both UCB examples, ε-detection exercise, η = 0.2 update) are recomputed in **Notes numericals**.

## Dynamic programming (Unit 4)
- **Policy evaluation** — repeat the Bellman expectation backup until max Δ < θ; equals (I − γP_π)⁻¹R_π.
- **Policy improvement** — make π greedy w.r.t. Q_π; never decreases value (unit-tested).
- **Policy iteration** — evaluate ↔ improve until no state changes (≈3 improvements here).
- **Value iteration** — one optimality backup per sweep; Δ shrinks by ≤ γ each sweep (contraction / Cauchy sequence, measured).
- **Model-based** — needs p and r; exact for the model it is given; milliseconds.
- Validated on the notes' dice game (V(IN) = 10.5, always-STAY 12) and the 4×4 gridworld.

## Monte Carlo (Unit 5)
- **Prediction** — average complete-episode returns after visits to s; no model. First-visit uses the first occurrence per episode, every-visit all occurrences. Notes' A/B example: FV 8.25/7.8, EV 8.0/4.0 — reproduced.
- **Error falls with experience** — MAE vs DP over episode counts (E4, **Monte Carlo lab**).
- **Control** — exploring starts (our simulator can start anywhere), on-policy ε-soft, off-policy weighted IS.
- **Off-policy evaluation** — "would a new matchmaker be better?" from logs of the current one. ρ = Π π/b; ordinary IS unbiased but high-variance, weighted IS biased but stable; per-decision and discounting-aware IS reduce variance. Coverage is checked. The effective sample size tells you when not to trust it (E8).
- **Real data** — uploaded histories have no actions, so we run MC *prediction* on them and refuse control/IS rather than invent actions.

## Q-learning and the DQN question
- Update: Q ← Q + α[R + γ max Q(S′,·) − Q], no bootstrap after terminal. **Q-learning** screen shows target, error and new value with real numbers.
- "Why not a DQN?" — v3 called it one, incorrectly. It is a table; 48 states need no function approximation; deep RL is not in Units 1–5 and torch would bloat the Vercel build. DQN is listed as future work.
- Fixed in v4: terminal bootstrapping bug; minutes-scale reward; unseeded training.

## Model-based vs model-free, and why policies differ
DP uses the (estimated) model; MC and Q-learning use only samples. Disagreements concentrate where two actions have nearly equal value (small q*-gap) or where a state is rarely visited. On-policy MC control learns the best ε-soft policy, not the greedy optimum. Off-policy MC control learns from episode tails only. **Algorithm comparison** shows agreement, q*-gaps and paired return differences.

## Real vs simulated (be precise)
- **Real**: uploaded match histories → derived features, clusters, burnout trend, coach report, MC prediction of V(s) on real sessions; player/map CSVs → drafting and map fit.
- **Simulated**: player reactions to difficulty, every RL episode, bandit rewards, the demo queue.
- **Model estimate**: p̂(s′|s,a), r̂.
- **Fallback/proxy**: rage-quit flag and closeness in data without such logs; tilt/fatigue in the tournament builder.

## Limitations to volunteer
Hand-specified dynamics; coarse 48-bucket state; proxies for rage quits; estimated model; IS variance for large policy changes; constant-α Q-learning noise; bandit arm means derived from a rule; no live A/B test; in-memory storage.

<!-- NUMBERS:BEGIN -->
## Numbers to quote (generated from experiments/results.json)

- DP-optimal vs always-KEEP on 1000 identical simulated sessions: discounted return 8.53 → 11.10; rage-quit sessions 45.4% → 28.9%.
- Tabular Q-learning agrees with DP on 79% of states; paired gap -0.207 ± 0.090.
- Model-free J as % of J*: MC control (es) 98.5%, MC control (on_policy) 97.1%, MC control (off_policy) 82.6%, Tabular Q-learning 98.1%.
- Map bandit, 1000 rounds: regret greedy 214, ε-greedy 54, optimistic 27, UCB 106.
- Importance sampling, small change: truth 10.490, WIS 10.498 (ESS 3144); large change ESS 36 → flagged unreliable.
- Notes numericals reproduced: 15/15.
<!-- NUMBERS:END -->
