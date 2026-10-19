# Clutch IQ v4 — adaptive matchmaking + a reinforcement-learning laboratory

Clutch IQ (formerly Neural Queue) is a competitive-gaming matchmaking system that
reasons about more than rank: skill, chemistry, fatigue, tilt, streaks and
session behaviour. v4 turns its RL layer into a complete, academically rigorous
RL platform: the *same* session environment is studied through **multi-armed
bandits, a finite MDP, dynamic programming, Monte Carlo methods and tabular
Q-learning**, all seeded, unit-tested and explainable in the UI.

## Run it

```bash
pip install -r requirements.txt
uvicorn backend_app:app --port 8000      # then open http://localhost:8000/
```
- API docs: http://localhost:8000/docs
- No server? Open `clutchiq.html` (single file) or `index.html` directly — the
  in-browser engine reproduces every RL endpoint with the same seeds. Only
  CSV-upload features (dataset ingestion, tournament, burnout, logged episodes)
  need the backend.
- Vercel: `vercel --prod` (see README_VERCEL.md).

Tests and experiments:
```bash
pip install -r requirements-dev.txt
pytest -q                       # algorithms, notes numericals, API, Python↔JS parity
python rl_experiments.py        # seeded experiment suite → experiments/results.json
python tools/build_docs.py      # regenerates EXPERIMENTS.md, SUBJECT_MAPPING.md, viva numbers
python tools/build_static.py    # refreshes the browser engine's data (after retraining)
python tools/build_standalone.py  # rebuilds the single-file clutchiq.html
```

## What is in the app

| Area | Screens |
|---|---|
| Operations (v3, preserved and fixed) | Data & upload · Live queue & matchmaking · Tournament · Burnout & coach |
| RL control room | **Live session** (Unit 1 loop, discounted return) · **Q-learning** (TD explainer, training lab) · **MDP explorer** (Unit 3, Bellman backups) · **Dynamic programming** (Unit 4: PE, PI, VI, γ sweep) |
| RL lab | **Bandit arena** (Unit 2: map arms, ε-greedy, optimistic, UCB, regret) · **Monte Carlo lab** (Unit 5: FV/EV prediction, MC control, importance sampling) · **Algorithm comparison** (seeded arena) |
| Academic | Viva questions · Syllabus mapping · Mathematical formulation · Notes numericals · Experiment results · Architecture · Limitations |

Toggle **Viva mode** to show, on every screen, the subject unit, concept, why it
is used, the formula, inputs/outputs, assumptions, a worked example computed from
the current run, and the source location.

## What changed from v3 (audit summary)

**Academic corrections**
- The optimiser was described as a "DQN"/"Deep-Q". It is **tabular Q-learning** — renamed everywhere; DQN is documented as a non-implemented extension.
- Q-learning bootstrapped from the next state even after a terminal rage quit — fixed.
- Reward's session term was minutes (−20…+15) and dominated the closeness and rage-quit terms — now a bounded indicator with the same weights (RL_METHODOLOGY.md §3).
- Training was unseeded and the UI's live simulator differed from the training simulator — now one seeded simulator everywhere.

**Bugs fixed**
- Uploading any CSV with a date column crashed `/api/engineer` (`KeyError: 'player_ref'`).
- Sessions, streaks and tilt baselines are now computed per player; score/assists reach player profiles; per-player rage-quit rate is per session.

**Added**
- `rl_core.py` single problem definition (48-state MDP, simulator, PRNG shared with JS); `mdp_model.py` simulation-estimated transition model, Markov chain, MRP closed form; `dynamic_programming.py`; `monte_carlo.py` (incl. ordinary/weighted/per-decision/discounting-aware IS and off-policy MC control); `bandits.py`; `rl_evaluation.py`; `rl_experiments.py`; `notes_examples.py`; `syllabus.py`; `rl_api.py` (25 endpoints).
- Browser engine `assets/rl_engine.js` with endpoint emulation and parity tests.
- Provenance labels on every result: REAL DATA / SIMULATED / MODEL ESTIMATE / DERIVED FEATURE / FALLBACK/PROXY.
- 130 tests, including every worked example with a printed answer in the uploaded notes.

Everything that worked in v3 still works: simulation mode, CSV upload with
schema detection, feature engineering, chemistry clustering and labels,
balanced matchmaking, tournament drafting/scheduling/map recommendation,
burnout forecast, coach report, `/api/optimize`, `/api/reward`, synthetic players.
The v3 Q-table and page are kept in `legacy/`.

## Documentation
- `RL_METHODOLOGY.md` — formal MDP, reward, simulator assumptions, algorithms, evaluation
- `SUBJECT_MAPPING.md` — Units 1–5 → implementation, demo screen, honest status
- `EXPERIMENTS.md` — reproducible setup, results, observations (generated)
- `VIVA_GUIDE.md` — answers to likely questions, numbers to quote
- `ARCHITECTURE.md` — modules, request flow, determinism, deployment

## Honesty rules
The RL environment is a simulator with documented assumptions; it is not fitted
to real players. Uploaded match histories contain no matchmaking actions, so on
real data only Monte Carlo *prediction* runs — control and importance sampling
are refused rather than fabricated. Importance-sampling estimates carry an
effective-sample-size diagnostic. See **Limitations & honesty** in the app.
