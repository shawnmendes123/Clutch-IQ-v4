# Demo datasets (all SIMULATED)

Invented gamertags and statistics, generated with a fixed seed by
`tools/make_sample_datasets.py`. They contain no real players. Use them to try the app;
results on them say nothing about real players.

| File | Upload to | What it shows |
|---|---|---|
| `match_history_demo.csv` | Data & upload | 16 players x 10 days, W/L, kills/deaths/assists, dates. Features, chemistry clusters, matchmaking, burnout forecast, coach report, real-data Monte Carlo prediction |
| `scoreline_history_demo.csv` | Data & upload | 12 players, results as scorelines ("13-9"), so match closeness comes from the scoreline instead of the 0.5 fallback |
| `tournament_players_demo.csv` + `tournament_maps_demo.csv` | Tournament (players + maps) | 24 players with stats and 8 maps: team drafting, schedule, map recommendation, "Learn map with a bandit" |
| `rl_session_logs_demo_SIMULATED.csv` | Logged-episodes upload (Monte Carlo lab) | Logs with actions and behaviour probabilities, the only kind that allows off-policy importance sampling |

Uploads need the backend running (`uvicorn backend_app:app --port 8000`).
