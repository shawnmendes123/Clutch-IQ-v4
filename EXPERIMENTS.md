# Experiments — reproducible results

All numbers below were produced by `python rl_experiments.py` (seeded; full
suite ≈ 82 s on a laptop CPU) and rendered by
`tools/build_docs.py` from `experiments/results.json`. Nothing is typed by hand
and no run was selected or discarded. **Provenance:** SIMULATED / MODEL ESTIMATE — no real-player data is used in these experiments.

Common settings: γ = 0.9 unless stated; the session MDP of
`rl_core.py`; the simulation-estimated model with 1000 samples per (s,a), seed 7;
evaluation starts from the realistic session-start distribution.

Reproduce: `python rl_experiments.py` (or `--quick` for a smoke run), then
`python tools/build_docs.py`.


## E1 — Simulation-estimated model: accuracy vs samples per (s,a)

| samples/(s,a) | max \|p̂−p\| | mean \|p̂−p\| | s.e. bound | π̂* = exact π* | J of planned π on true dynamics | exact J* |
|---|---|---|---|---|---|---|
| 100 | 0.1360 | 0.0220 | 0.0500 | 85.4% | 11.0584 | 11.1983 |
| 300 | 0.0901 | 0.0128 | 0.0289 | 91.7% | 11.1676 | 11.1983 |
| 1000 | 0.0431 | 0.0073 | 0.0158 | 95.8% | 11.1449 | 11.1983 |
| 3000 | 0.0283 | 0.0043 | 0.0091 | 97.9% | 11.1914 | 11.1983 |

**Observation.** Model error shrinks roughly as 1/√n (max error 0.136 → 0.028). The policy planned on the estimate agrees with the exact optimum on 85.4% → 97.9% of states and loses at most 1.2% of value. **Interpretation:** planning on a simulation-estimated model is sound; residual disagreements are in near-tied states.

## E2 — Dynamic programming convergence (estimated model, 1000 samples/pair)

| γ | VI sweeps (θ=1e-8) | max Δ ratio | PI improvement steps | PI evaluation sweeps | PI = VI | closed-form MRP gap |
|---|---|---|---|---|---|---|
| 0.5 | 23 | 0.4872 | 3 | 57 | 100% | 0.00e+00 |
| 0.9 | 52 | 0.8770 | 3 | 133 | 100% | 0.00e+00 |
| 0.99 | 64 | 0.9647 | 3 | 164 | 100% | 0.00e+00 |

**Observation.** The measured contraction ratio is always below γ (Banach fixed point), so sweeps grow as γ → 1. Policy iteration needs only a few improvement steps and returns the same policy as value iteration. Iterative policy evaluation matches the closed-form MRP solution (I − γP_π)⁻¹R_π.

## E3 — Effect of the discount factor on the optimal session policy

| γ | LOWER | KEEP | RAISE | sim. rage-quit rate | sim. session length | sim. close-match rate |
|---|---|---|---|---|---|---|
| 0.0 | 29 | 13 | 6 | 31.2% | 9.28 | 74.7% |
| 0.3 | 29 | 13 | 6 | 31.2% | 9.28 | 74.7% |
| 0.5 | 29 | 13 | 6 | 31.2% | 9.28 | 74.7% |
| 0.7 | 29 | 14 | 5 | 30.9% | 9.31 | 70.6% |
| 0.9 | 31 | 14 | 3 | 28.9% | 9.47 | 64.1% |
| 0.95 | 32 | 14 | 2 | 28.9% | 9.49 | 62.5% |
| 0.99 | 33 | 13 | 2 | 28.8% | 9.50 | 62.4% |

**Observation.** As γ grows the optimal controller lowers difficulty in more states (29 → 33) and raises it in fewer (6 → 2): a myopic controller chases close matches now (close-match rate 74.7%), a far-sighted one protects tilt and session length (rage quits 31.2% → 28.8%). Simulated rates are measured with the same 1000 seeded session-start episodes for every gamma.

## E4 — MC prediction error vs episodes (heuristic policy, reference = DP policy evaluation)

Seeds: 11, 12, 13, 14, 15. Mean ± sd across seeds.

| episodes | MAE first-visit | MAE every-visit | states covered |
|---|---|---|---|
| 10 | 2.475 ± 0.403 | 2.419 ± 0.363 | 54% |
| 25 | 2.346 ± 0.370 | 2.275 ± 0.370 | 85% |
| 50 | 1.595 ± 0.039 | 1.611 ± 0.080 | 94% |
| 100 | 1.310 ± 0.239 | 1.344 ± 0.269 | 99% |
| 250 | 0.783 ± 0.144 | 0.842 ± 0.152 | 100% |
| 500 | 0.536 ± 0.066 | 0.593 ± 0.099 | 100% |
| 1000 | 0.388 ± 0.071 | 0.422 ± 0.078 | 100% |
| 2000 | 0.285 ± 0.040 | 0.305 ± 0.038 | 100% |
| 5000 | 0.202 ± 0.047 | 0.215 ± 0.050 | 100% |
| 10000 | 0.150 ± 0.025 | 0.161 ± 0.024 | 100% |
| 20000 | 0.115 ± 0.012 | 0.118 ± 0.013 | 100% |

**Observation.** Both estimators converge toward the DP values as experience accumulates (MAE 1.31 at 100 episodes → 0.12 at 20000); the residual includes the DP reference's own model error. First- and every-visit behave similarly here because revisits within an episode are short-range.

## E5 — Model-free control vs the DP optimum (J = expected discounted return from session start, on the estimated model)

References: J* (DP) = 11.204, manual heuristic = 10.796, always KEEP = 8.550.

| algorithm | episodes | J (mean ± sd over seeds) | % of J* | agreement with DP | seeds |
|---|---|---|---|---|---|
| MC control (es) | 20,000 | 11.036 ± 0.103 | 98.5% | 85.4% | 3 |
| MC control (on_policy) | 20,000 | 10.879 ± 0.159 | 97.1% | 83.3% | 3 |
| MC control (off_policy) | 20,000 | 9.251 ± 0.476 | 82.6% | 59.0% | 3 |
| Tabular Q-learning | 30,000 | 10.994 ± 0.123 | 98.1% | 84.6% | 5 |

**Observation.** Model-free learners reach most of the DP optimum from experience alone; on-policy MC control converges to the best ε-soft policy; off-policy MC control learns only from episode tails after the last exploratory action and is clearly slowest at this budget.

## E6 — Algorithm comparison arena (identical seeded episodes, common random numbers)

1000 episodes, seed 2026, start = session_start, γ = 0.9, MC control 20,000 episodes (ε = 0.1), Q-learning 30,000 episodes, 3 stability runs. ± = 95% CI; Δ is paired on identical episodes.

| policy | discounted return | Δ vs DP (paired) | session length | rage quit | close match | agreement w/ DP | stability |
|---|---|---|---|---|---|---|---|
| Random baseline | 7.781 ± 0.272 | -3.317 ± 0.228 | 7.82 | 46.5% | 26.5% | 65% | — |
| Always KEEP (v3-style rank-only matchmaking) | 8.530 ± 0.276 | -2.568 ± 0.201 | 8.38 | 45.4% | 28.7% | 29% | 100% |
| Manual rule (designer heuristic) | 10.788 ± 0.277 | -0.310 ± 0.101 | 9.34 | 31.3% | 53.2% | 71% | 100% |
| Dynamic Programming — Value Iteration | 11.098 ± 0.279 | +0.000 ± 0.000 | 9.47 | 28.9% | 64.1% | 100% | 97% |
| Dynamic Programming — Policy Iteration | 11.098 ± 0.279 | +0.000 ± 0.000 | 9.47 | 28.9% | 64.1% | 100% | — |
| Monte Carlo Control (on-policy, eps=0.1) | 10.992 ± 0.283 | -0.106 ± 0.087 | 9.32 | 31.4% | 66.1% | 83% | 81% |
| Tabular Q-Learning | 10.891 ± 0.276 | -0.207 ± 0.090 | 9.44 | 29.2% | 52.9% | 79% | 74% |

**Observation.** Relative to rank-only matchmaking (always KEEP), the DP policy raises the discounted return from 8.53 to 11.10 and cuts rage-quit sessions from 45.4% to 28.9%. Learned policies sit between the heuristic and DP; paired intervals show which gaps are real. DP's stability < 100% reflects model-estimation noise across model seeds (near-tied states).

## E7 — Bandit Arena: map recommendation (10 arms, Bernoulli satisfaction, 1000 rounds)

Arms: sample_maps.csv + demo roster (4 entry, 3 sniper, 2 flex, 1 support); 200 runs per algorithm, same seeds for every algorithm; ε = 0.1, UCB c = 1.414 (UCB1), optimistic Q₀ = 1.0, 50 exploration rounds.

| map | fit score | q* |
|---|---|---|
| Nuketown | 2.5 | 0.85 |
| Crash | 0.475 | 0.3039 |
| Standoff | 0.45 | 0.2972 |
| Firing Range | 2.425 | 0.8298 |
| Raid | 0.4 | 0.2837 |
| Killhouse | 2.375 | 0.8163 |
| Hijacked | 0.35 | 0.2702 |
| Summit | 1.825 | 0.668 |
| Array | 1.8 | 0.6612 |
| Crossfire | 0.275 | 0.25 |

| algorithm | expected regret | total reward ± 95% CI | % optimal (last 10%) |
|---|---|---|---|
| Explore only (uniform random) | 327.7 | 521.9 ± 2.1 | 10.2% |
| Greedy (exploit only) | 213.9 | 636.8 ± 32.2 | 17.5% |
| Fixed exploration + greedy | 30.7 | 819.9 ± 2.3 | 46.0% |
| ε-greedy | 53.6 | 797.0 ± 2.8 | 52.7% |
| Optimistic greedy | 26.7 | 823.8 ± 5.3 | 44.5% |
| UCB | 106.3 | 743.7 ± 1.9 | 32.3% |

ε sweep (ε-greedy, 1000 rounds):

| ε | regret | % optimal (last 10%) |
|---|---|---|
| 0.0 | 230.4 | 16.5% |
| 0.01 | 96.8 | 39.9% |
| 0.05 | 44.1 | 56.1% |
| 0.1 | 53.6 | 52.7% |
| 0.2 | 82.1 | 43.1% |
| 0.5 | 173.4 | 35.7% |

Non-stationary (balance patch at T/2, 2000 rounds, ε-greedy):

| update | total regret | regret after the patch | % optimal (last 10%) |
|---|---|---|---|
| sample average (1/n) | 210.4 | 153.7 | 18.1% |
| constant α = 0.1 | 153.7 | 66.2 | 35.4% |

**Observation.** Pure exploitation gets stuck; any systematic exploration cuts regret sharply. Regret is U-shaped in ε. UCB1's c = √2 is conservative for rewards in [0, 1] (it keeps exploring); smaller c lets UCB overtake ε-greedy (try it in the Bandit arena). After a balance patch, a constant step size α tracks the new best map while sample averages react slowly — the notes' argument for α in non-stationary problems.

## E8 — Off-policy evaluation by importance sampling (session-start value J)

| scenario | truth (DP) | logged policy's value | ordinary IS | weighted IS | per-decision IS | discounting-aware WIS | ESS | diagnostic |
|---|---|---|---|---|---|---|---|---|
| Small change: heuristic ε=0.1 from logs of heuristic ε=0.3 | 10.490 | 9.841 | 10.586 | 10.498 | 10.461 | 10.452 | 3144 | reliable |
| Large change: DP-optimal ε=0.1 from logs of heuristic ε=0.3 | 10.849 | 9.841 | 6.136 | 9.711 | 9.920 | 9.658 | 36 | unreliable |
| Uniform-random logs: heuristic ε=0.1 | 10.490 | 7.793 | 3.747 | 6.870 | 10.073 | 7.554 | 27 | unreliable |

**Observation.** For a small policy change, importance sampling recovers the new policy's value from old logs (weighted IS within a few hundredths). For a large change, or uniform-random logs, the ratios become heavy-tailed, the effective sample size collapses and the estimates are biased low at this sample size; the diagnostic flags them instead of reporting false confidence. Per-decision IS is the most robust of the estimators here.

## Lecture-note numericals

15 of 15 worked examples with a printed answer reproduce exactly (u3-ex3.8, u3-const13, u3-rain-dry, u4-dice-random, u4-dice-stay, u4-gridworld, u3-ex3.22, u5-mc-ab, u2-exploration, u2-exploitation, u2-fixed-exploration, u2-ucb-12, u2-ucb-100, u2-eps-detect, u2-const-step, u2-eps-assignment).

