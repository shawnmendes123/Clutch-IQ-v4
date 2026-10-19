# Architecture — Clutch IQ v4

```
                 ┌──────────────── Data in ────────────────┐
                 │ match-history CSV (REAL)   maps/players │
                 │ logged RL sessions (optional)            │
                 └───────────────┬─────────────────────────┘
                                 │
   Operations (v3, preserved)    │            RL engine (v4)
 ┌───────────────────────────┐   │   ┌─────────────────────────────────────┐
 │ schema_detector           │   │   │ rl_core.py  ← single source of truth │
 │ feature_engine (fixed)    │───┼──▶│   S, A, reward, SessionEnv, PRNG     │
 │ gnn_chemistry             │   │   │   policies, episodes, returns        │
 │ cluster_labeler           │   │   └──────┬──────────────┬───────────────┘
 │ tournament_builder        │   │          │              │
 │ burnout_forecast          │   │   mdp_model.py     bandits.py (U2)
 │ coach_report              │   │   (U3 model)        map arms ← tournament fit
 └─────────────┬─────────────┘   │      │
               │                 │   dynamic_programming.py (U4)
               │                 │   monte_carlo.py (U5)
               │                 │   q_learning.py (+ rl_optimizer facade)
               │                 │   rl_evaluation.py · rl_experiments.py
               │                 │   notes_examples.py · syllabus.py
               ▼                 ▼
        ┌──────────────────────────────────────────────┐
        │ FastAPI: backend_app.py (+ rl_api.py router)  │
        │ /api/* v3 endpoints unchanged                 │
        │ /api/rl/* · /api/bandit/* · /api/academic/*   │
        └───────────────────┬──────────────────────────┘
                            │ JSON (same shapes)
        ┌───────────────────▼──────────────────────────┐
        │ index.html + assets/                          │
        │  app_core.js  engine routing, nav, viva mode  │
        │  ops.js · control.js · lab.js · academic.js   │
        │  charts.js    SVG charts                      │
        │  rl_engine.js offline mirror of every RL API  │
        │  static_data.js  Q-table, maps, syllabus,     │
        │                  experiment results           │
        └──────────────────────────────────────────────┘
```

## Files

| File | Role |
|---|---|
| `rl_core.py` | State/action/reward definitions, simulator, Mulberry32 PRNG, policies, episodes, returns, `spec()` |
| `mdp_model.py` | `FiniteMDP`, simulation-estimated model, exact law (validation), Markov chain, MRP closed form |
| `dynamic_programming.py` | Policy evaluation / improvement / iteration, value iteration, Bellman residual, backup breakdown |
| `monte_carlo.py` | First/every-visit prediction, MC action values, MC control (ES, on-policy, off-policy), importance sampling, real-data episodes |
| `q_learning.py` | Tabular Q-learning, TD update explainer, production table I/O |
| `rl_optimizer.py` | v3-compatible facade (`recommend_action`, `reward_function`, `QTableAgent`, …) |
| `bandits.py` | Bandit environment and algorithms, arena, map arms, notes' Unit 2 numericals |
| `rl_evaluation.py` | Seeded evaluation, comparison arena, agreement / disagreement analysis, trajectories |
| `rl_experiments.py` | Reproducible experiment suite → `experiments/results.json` |
| `notes_examples.py` | Lecture-note worked examples recomputed and checked |
| `syllabus.py` | Coverage matrix (served to the UI and rendered into SUBJECT_MAPPING.md) |
| `rl_api.py` | FastAPI router for all RL / bandit / academic endpoints |
| `backend_app.py` | v3 API (fixed) + router + real-data MC endpoint + local UI serving |
| `assets/rl_engine.js` | Browser port of the RL engine; `ClutchRL.call()` emulates the endpoints |
| `tools/build_static.py` | Regenerates `assets/static_data.js` |
| `tools/build_docs.py` | Regenerates SUBJECT_MAPPING.md and EXPERIMENTS.md from code/results |
| `tests/` | pytest: algorithms, notes numericals, API, Python↔JS parity |

## Engine routing
`App.api(method, path, body)` calls the FastAPI backend when `/api/health`
answers; otherwise `ClutchRL.call()` serves the same path in the browser.
Upload-based features (CSV ingestion, tournament, burnout, logged episodes,
real-data MC) always need the backend and say so.

## Determinism
All randomness flows from `Mulberry32(derive_seed(seed, k))`. The environment
and the agent use separate streams. Python and JavaScript implement the PRNG
bit-identically (checked against the reference JS implementation in the tests).

## Deployment (Vercel)
`api/index.py` exposes `backend_app.app`; static files (`index.html`, `assets/`)
are served by the platform. `vercel.json` raises the function timeout to 60 s for
the heavier experiments. NumPy/pandas/scikit-learn only — no torch.
