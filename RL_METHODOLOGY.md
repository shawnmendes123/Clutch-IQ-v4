# RL methodology — Clutch IQ v4

This document is the formal specification behind every RL screen and endpoint.
All definitions live in code in `rl_core.py` (`spec()` returns them as JSON at
`GET /api/rl/spec`), so the document, the backend and the browser engine share
one source of truth.

## 1. The decision problem (Unit 1)

| Element | Clutch IQ |
|---|---|
| Agent | Session controller that sets lobby difficulty before each match |
| Environment | One player's play session — **simulated** by `rl_core.SessionEnv` |
| Time step | One match |
| State `S_t` | (fatigue bucket, tilt bucket, streak bucket) |
| Action `A_t` | LOWER / KEEP / RAISE lobby difficulty |
| Reward `R_{t+1}` | match completed, closeness, rage-quit penalty (§3) |
| Episode | One session; ends in a terminal state (§2.3) |
| Policy `π(a|s)` | 49×3 probability matrix |

Environment classification (Unit 1 vocabulary): fully observable with respect to
the bucketed state, stochastic, discrete, episodic, single agent.

**Immediate vs full RL.** Map recommendation is *immediate RL* — a bandit (§6).
Session control is *full RL* — actions change future states, so returns, not
single rewards, are optimised.

## 2. Finite MDP ⟨S, A, P, R, γ⟩ (Unit 3)

### 2.1 State space
```
fatigue f ∈ {0 FRESH, 1 WARM, 2 TIRED, 3 EXHAUSTED}     f = min(3, ⌊4·fatigue⌋)
tilt    t ∈ {0 CALM, 1 UNEASY, 2 TILTED, 3 VOLATILE}    t = min(3, ⌊4·tilt⌋)
streak  k ∈ {0 LOSING (≤ −2), 1 NEUTRAL (−1..1), 2 WINNING (≥ 2)}
s = 12f + 3t + k ∈ {0,…,47};   s = 48 is the absorbing END state (V(END) = 0)
```
The key format `"f-t-k"` is unchanged from v3, so `q_table.json` stays compatible.

### 2.2 Actions
`A = {0 LOWER, 1 KEEP, 2 RAISE}` with difficulty offset δ = −1, 0, +1 band.

### 2.3 Simulator dynamics (assumptions — SIMULATED, not fitted to players)
Per match, with parameters from `rl_core.ENV_PARAMS`:

```
p_win(s,a) = clip(0.5 + 0.20(k−1) − 0.03f − 0.06t − 0.15δ(a), 0.05, 0.95)
C(s,a)     = 1 − 2|p_win − 0.5|                          predicted closeness
after a loss:  rage quit w.p. 0.005 + 0.025t + 0.01f + 0.02[k=LOSING] + 0.02[a=RAISE] + 0.04[p_win<0.3]
after a win:   boredom quit w.p. [p_win ≥ 0.7]·(0.06 + 0.04[k=WINNING])
tilt:   loss → +1 w.p. 0.30 (+0.25 if stomped);  win → −1 w.p. 0.35 (+0.20 if C ≥ 0.8)
streak: an opposite result breaks a streak; a same result moves NEUTRAL→LOSING/WINNING w.p. 0.5
fatigue: +1 level w.p. 0.33 / 0.33 / 0.36 (LOWER / KEEP / RAISE); an EXHAUSTED player who tires further logs off
```
Terminal states: rage quit (penalised), boredom quit (no continuation), natural
end (exhaustion, no penalty). Because fatigue rises with probability ≥ 0.33 every
match, **every policy terminates with probability 1**, so γ = 1 is well defined
and Monte Carlo episodes are finite.

Exactly five uniforms are drawn per step in a fixed order, so two policies run
with the same seed face the same random numbers (common random numbers).

**Markov property.** `env_step` reads only (f, t, k) and the action. Hence
P(S_{t+1} | S_t, A_t, history) = P(S_{t+1} | S_t, A_t) exactly for the simulator.
For real players the buckets are a summary; Markovness there is an assumption.

### 2.4 Simulation-estimated transition model (MODEL ESTIMATE)
The planners never read the formulas above. For every (s, a) the simulator is
placed in s, action a applied, and the outcome recorded `N` times (default
1000, seed 7; 144 000 simulated transitions):
```
p̂(s′|s,a) = N(s,a,s′) / N(s,a)          r̂(s,a,s′) = mean reward observed on s,a→s′
r̂(s,a)    = Σ_s′ p̂(s′|s,a) r̂(s,a,s′)
```
Because we wrote the simulator, its exact law (`rl_core.exact_transition`) is also
available — used **only** to measure the estimate's error (experiment E1) and in tests.

### 2.5 Markov chain and MRP
Fixing π gives the chain `P_π(s′|s) = Σ_a π(a|s) p(s′|s,a)` and the MRP reward
`R_π(s) = Σ_a π(a|s) r(s,a)`. Its Bellman equation `v = R_π + γ P_π v` is solved in
closed form `v = (I − γP_π)⁻¹ R_π` (`mdp_model.mrp_closed_form`) and compared
with iterative policy evaluation (agreement ≈ 1e-8).

### 2.6 Value functions and Bellman equations
```
V_π(s)   = Σ_a π(a|s) Σ_s′ p(s′|s,a) [ r(s,a,s′) + γ V_π(s′) ]
Q_π(s,a) = Σ_s′ p(s′|s,a) [ r(s,a,s′) + γ V_π(s′) ]
V*(s)    = max_a Σ_s′ p(s′|s,a) [ r(s,a,s′) + γ V*(s′) ]
Q*(s,a)  = Σ_s′ p(s′|s,a) [ r(s,a,s′) + γ max_a′ Q*(s′,a′) ]
```
`dynamic_programming.backup_breakdown` returns every term p·[r + γV] of these
sums for one state; the MDP explorer prints them.

## 3. Reward (reviewed)

```
R = w1·(match completed and session continues) + w2·C − w3·(rage quit),   w = (1.0, 1.2, 3.0)
```
v3 used `w1·(session-length Δ in minutes, −20…+15)`; that term dwarfed the others
and a rage quit cost less than an ordinary bad match. v4 keeps the weights and
the three-term structure but makes the session term an indicator, so each step's
reward is bounded:

| Outcome | R |
|---|---|
| completed match | 1.0 + 1.2·C ∈ [1.0, 2.2] |
| boredom quit | 1.2·C ∈ [0, 1.2] |
| rage quit | 1.2·C − 3.0 ∈ [−3.0, −1.8] |

A rage quit also ends the episode, forfeiting future continuation reward — the
delayed cost that makes this a sequential problem. Discounted return:
`G_t = R_{t+1} + γ G_{t+1}`, `G_T = 0`.

## 4. Dynamic programming (Unit 4) — `dynamic_programming.py`
- **Iterative policy evaluation**: synchronous sweeps `V_{k+1} = T_π V_k` until `max_s|V_{k+1}−V_k| < θ` (in-place sweeps optional).
- **Policy improvement**: `π′(s) = argmax_a Q_π(s,a)`; ties keep the old action (prevents cycling).
- **Policy iteration**: start KEEP everywhere; evaluate → improve until policy-stable.
- **Value iteration**: `V_{k+1} = T* V_k`; π* greedy w.r.t. V*.
- **Contraction**: each run records Δ_{k+1}/Δ_k; it never exceeds γ (unit-tested), which is the Banach fixed-point / Cauchy-sequence property demonstrated numerically.
- Validated on the notes' dice game (V(IN) = 10.5 random, 12 always-STAY) and the 4×4 gridworld (−14, −18, −20, −22).

## 5. Monte Carlo (Unit 5) — `monte_carlo.py`
- **First-visit / every-visit prediction** of V_π; reproduces the notes' A/B example (FV 8.25/7.8, EV 8.0/4.0).
- **MC action values** Q(s,a) from first visits.
- **MC control**: exploring starts (the simulator can start from any (s,a)); on-policy first-visit ε-soft control; off-policy control with weighted importance sampling (Sutton §5.7: C(s,a) += W, Q += W/C·(G−Q), stop at first non-greedy action, W /= b(A|S)).
- **Off-policy prediction**: ρ_{t:T−1} = Π π(A_k|S_k)/b(A_k|S_k). Estimators: ordinary IS, weighted IS, per-decision IS, and discounting-aware IS on flat partial (truncated) returns. Coverage (π>0 ⇒ b>0) is checked and violations are refused. The effective sample size (Σρ)²/Σρ² flags unreliable estimates.
- **Real data**: uploaded match histories become episodes of bucketed states with proxy rewards; they contain no actions, so only prediction runs on them.

## 6. Multi-armed bandits (Unit 2) — `bandits.py`
Arms are candidate maps for a scheduled match. `q*(map) = 0.25 + 0.6·(fit − min)/(max − min)` over the shown arms, where `fit` is the tournament builder's rule-based map-fit score (DERIVED FEATURE). Rewards are SIMULATED: Bernoulli(q*) thumbs-up (default) or Gaussian quality. Algorithms: explore-only, greedy, fixed exploration + greedy, ε-greedy, optimistic initial values, UCB (`Q + c√(ln t / N)`, untried arms first). Updates: sample average `Q += (R−Q)/n` or constant α (exponential recency weighting). Non-stationarity: Gaussian random-walk drift or a "balance patch" that permutes arm values at T/2. Metrics: expected regret Σ(max q* − q*(A_t)), the notes' realised regret T·max q* − ΣR, % optimal, explore/exploit counts. All algorithms in an arena run on identical seeded environments; ties in greedy choices are broken at random (notes' rule), while the notes' worked tables use lowest-index ties to reproduce printed answers.

## 7. Tabular Q-learning — `q_learning.py`
```
target = R + γ·max_a′ Q(S′,a′)   (target = R if S′ is terminal)
Q(S,A) ← Q(S,A) + α (target − Q(S,A))
```
ε-greedy behaviour, ε ← max(ε_min, ε·decay) per episode. Production defaults:
30 000 episodes, α = 0.05, γ = 0.9, ε 1.0 → 0.05 (decay 0.9998), seed 7,
exploring starts over all 48 states. Two v3 bugs are fixed: the target no longer
bootstraps after a terminal transition, and rewards are bounded per step.

**Naming.** This is not a DQN. A genuine DQN (neural Q-approximator, replay
buffer, target network, minibatch updates) is a possible extension but is not
implemented: 48 discrete states need no function approximation, a torch
dependency would weigh down the Vercel deployment, and deep RL is not part of
the uploaded Units 1–5.

## 8. Evaluation methodology — `rl_evaluation.py`, `rl_experiments.py`
- Every sequential policy is evaluated on the **same seeded episodes** (episode i uses `derive_seed(seed, i)` for the environment stream; agents sample actions from a separate stream).
- Metrics: discounted and undiscounted return, session length, rage-quit / boredom-quit / natural-end rates, close-match rate (C ≥ 0.8); mean, sd and 95% CI (1.96·sd/√n).
- **Paired differences** vs the DP policy on identical episodes give much tighter intervals than comparing independent means.
- Policy agreement (% of 48 states), q*-gap of each disagreement, model value J = Σ d₀(s)V^π(s), stability (agreement between runs with different seeds), compute time, convergence.
- Bandits are evaluated separately (regret, % optimal) — never on the sequential leaderboard.
- The browser engine (`assets/rl_engine.js`) reproduces the Python results with the same seeds; `tests/test_js_parity.py` checks episodes, models, DP, MC, IS, Q-learning and bandits.
