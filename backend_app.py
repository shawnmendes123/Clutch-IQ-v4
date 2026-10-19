"""
backend_app.py — Clutch IQ v4 backend (formerly Neural Queue v3)

v4 adds the Reinforcement Learning engine under /api/rl/*, /api/bandit/* and
/api/academic/* (see rl_api.py) while keeping every v3 endpoint unchanged.

Original v3 description:
Same matchmaking engine as v1, but data-source agnostic: upload ANY
competitive-game CSV and the pipeline auto-detects columns, engineers
tilt/fatigue/streak/reward features, then runs the same GNN chemistry +
RL session optimizer on top of it.

Run with:
    pip install -r requirements.txt
    python rl_optimizer.py          # trains q_table.json once
    uvicorn app:app --reload --port 8000

Then open http://localhost:8000/docs for the interactive API, or point
neural_queue_demo.html's "Connect Backend" panel at this server.

Endpoints
---------
POST /upload                 -> upload a CSV, get back a dataset_id + schema report
POST /schema/confirm         -> (optional) override auto-detected column mapping
POST /engineer/{dataset_id}  -> run feature engineering, returns engineered player records
GET  /players/{dataset_id}   -> engineered players + chemistry clusters + auto-labels
POST /matchmake/{dataset_id} -> assemble a balanced, chemistry-aware slate
POST /optimize                -> RL session optimizer decision for a given state
POST /reward                  -> compute reward for a given outcome
GET  /forecast/{dataset_id}/{player_id} -> burnout forecast for one player
GET  /report/{dataset_id}/{player_id}   -> auto-generated coach report (markdown)
GET  /players/synthetic        -> fallback synthetic pool (no upload needed, demo mode)
"""
import io
import uuid

import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from schema_detector import detect_schema, build_schema_report
from feature_engine import engineer_features, to_player_records
from gnn_chemistry import cluster_chemistry, assemble_slate
from cluster_labeler import label_clusters
from burnout_forecast import forecast_player
from coach_report import generate_report
from rl_optimizer import recommend_action, reward_function
from tournament_builder import build_tournament
import models  # synthetic fallback generator (unchanged from v1)
import rl_api
import monte_carlo as mc
import rl_core as core

app = FastAPI(title="Clutch IQ v4 — RL matchmaking engine",
              description="Matchmaking (chemistry, tournament, burnout) + a full RL laboratory: bandits, finite MDP, "
                          "dynamic programming, Monte Carlo and tabular Q-learning. Every result carries a provenance label.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(rl_api.router, prefix="/api")

# in-memory dataset store keyed by dataset_id (fine for a demo/prototype;
# swap for Redis/Postgres for a real deployment)
DATASETS: dict[str, dict] = {}


@app.get("/api/health")
def health():
    return {"status": "ok", "datasets_loaded": len(DATASETS), "version": "4.0",
            "engine": "python", "rl_endpoints": True}


# ------------------------------------------------------------------ upload
@app.post("/api/upload")
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Please upload a .csv file.")
    raw = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(400, f"Could not parse CSV: {e}")

    if len(df) < 5:
        raise HTTPException(400, "Dataset needs at least 5 rows.")

    schema = detect_schema(df)
    report = build_schema_report(df)

    dataset_id = str(uuid.uuid4())[:10]
    DATASETS[dataset_id] = {
        "df": df, "schema": schema, "filename": file.filename,
        "engineered": None, "players": None, "clusters": None, "cluster_labels": None,
    }

    return {
        "dataset_id": dataset_id,
        "filename": file.filename,
        "row_count": len(df),
        "columns": list(df.columns),
        "schema_report": report,
    }


class SchemaOverride(BaseModel):
    dataset_id: str
    overrides: dict  # e.g. {"kills": "MyKillColumn", "date": "MatchDate"}


@app.post("/api/schema/confirm")
def confirm_schema(body: SchemaOverride):
    ds = _get_dataset(body.dataset_id)
    for role, col in body.overrides.items():
        if col and col not in ds["df"].columns:
            raise HTTPException(400, f"Column '{col}' not found in dataset.")
        ds["schema"]["detected_columns"][role] = col or None
    return {"status": "updated", "detected_columns": ds["schema"]["detected_columns"]}


# ------------------------------------------------------------- engineering
@app.post("/api/engineer/{dataset_id}")
def engineer(dataset_id: str, n_clusters: int = 5):
    ds = _get_dataset(dataset_id)
    engineered, notes = engineer_features(ds["df"], ds["schema"])
    players = to_player_records(engineered)

    if len(players) < 2:
        raise HTTPException(400,
            "Only 1 distinct player detected in this dataset — chemistry clustering needs "
            "multiple players. This dataset is still usable for the RL session optimizer "
            "(see /forecast and /report), just not for /matchmake.")

    labels, centers, _ = cluster_chemistry(players, n_clusters=min(n_clusters, len(players)))
    cluster_info = label_clusters(players, labels, centers)

    ds["engineered"] = engineered
    ds["players"] = players
    ds["clusters"] = labels
    ds["cluster_labels"] = cluster_info

    return {
        "dataset_id": dataset_id,
        "player_count": len(players),
        "engineering_notes": notes,
        "cluster_labels": cluster_info,
    }


# ------------------------------------------------------------------ reads
@app.get("/api/players/synthetic")
def get_synthetic_players(n: int = 40, seed: int = None):
    """Fallback demo mode — no upload needed."""
    pool = models.generate_player_pool(n=n, seed=seed)
    players = [p.to_dict() for p in pool]
    labels, centers, _ = cluster_chemistry(players, n_clusters=5)
    cluster_info = label_clusters(players, labels, centers)
    return {"players": players, "chemistry_clusters": labels, "cluster_labels": cluster_info}


@app.get("/api/players/{dataset_id}")
def get_players(dataset_id: str):
    ds = _get_dataset(dataset_id)
    if ds["players"] is None:
        raise HTTPException(400, "Call /engineer/{dataset_id} first.")
    return {
        "players": ds["players"],
        "chemistry_clusters": ds["clusters"],
        "cluster_labels": ds["cluster_labels"],
    }


# -------------------------------------------------------------- matchmake
class MatchmakeRequest(BaseModel):
    slate_size: int = 10


@app.post("/api/matchmake/{dataset_id}")
def matchmake(dataset_id: str, req: MatchmakeRequest):
    ds = _get_dataset(dataset_id)
    if ds["players"] is None:
        raise HTTPException(400, "Call /engineer/{dataset_id} first.")
    return assemble_slate(ds["players"], slate_size=req.slate_size)


# -------------------------------------------------------------- RL layer
class OptimizeRequest(BaseModel):
    fatigue: float
    tilt: float
    streak: int


@app.post("/api/optimize")
def optimize(req: OptimizeRequest):
    return recommend_action(req.fatigue, req.tilt, req.streak)


class RewardRequest(BaseModel):
    session_length_delta: float
    close_match_baseline: float
    rage_quit: bool


@app.post("/api/reward")
def reward(req: RewardRequest):
    r = reward_function(req.session_length_delta, req.close_match_baseline, req.rage_quit)
    return {"reward": round(r, 3)}


# --------------------------------------------------------- crazy features
@app.get("/api/forecast/{dataset_id}/{player_id}")
def forecast(dataset_id: str, player_id: str):
    ds = _get_dataset(dataset_id)
    if ds["engineered"] is None:
        raise HTTPException(400, "Call /engineer/{dataset_id} first.")
    eng = ds["engineered"]
    history = eng[eng["player_ref"].astype(str).str[:12] == player_id].reset_index(drop=True)
    if history.empty:
        raise HTTPException(404, "Player not found in this dataset.")
    session_history = [
        {"session_index": i, "tilt_score": float(row["tilt_score"]), "fatigue_index": float(row["fatigue_index"])}
        for i, row in history.iterrows()
    ]
    return forecast_player(session_history)


@app.get("/api/report/{dataset_id}/{player_id}")
def report(dataset_id: str, player_id: str):
    ds = _get_dataset(dataset_id)
    if ds["players"] is None:
        raise HTTPException(400, "Call /engineer/{dataset_id} first.")
    player = next((p for p in ds["players"] if p["id"] == player_id), None)
    if not player:
        raise HTTPException(404, "Player not found.")
    try:
        fc = forecast(dataset_id, player_id)
    except HTTPException:
        fc = None
    return {"markdown": generate_report(player, fc)}


# ------------------------------------------------------ v3 tournament builder
TOURNAMENTS: dict[str, dict] = {}


@app.post("/api/tournament/build")
async def tournament_build(maps_file: UploadFile = File(...), players_file: UploadFile = File(...)):
    """Upload a maps.csv + players.csv pair and get back N balanced 5-player
    teams, a scheduled matchslate (closest-skill pairing), a recommended map
    per match, and the reward function score for each projected match."""
    for f in (maps_file, players_file):
        if not f.filename.lower().endswith(".csv"):
            raise HTTPException(400, f"'{f.filename}' must be a .csv file.")

    maps_raw = await maps_file.read()
    players_raw = await players_file.read()
    try:
        maps_df = pd.read_csv(io.BytesIO(maps_raw))
        players_df = pd.read_csv(io.BytesIO(players_raw))
    except Exception as e:
        raise HTTPException(400, f"Could not parse CSV: {e}")

    try:
        result = build_tournament(maps_df, players_df)
    except ValueError as e:
        raise HTTPException(400, str(e))

    tournament_id = str(uuid.uuid4())[:10]
    TOURNAMENTS[tournament_id] = result
    return {"tournament_id": tournament_id, **result}


@app.get("/api/tournament/{tournament_id}")
def tournament_get(tournament_id: str):
    result = TOURNAMENTS.get(tournament_id)
    if not result:
        raise HTTPException(404, "Unknown tournament_id — build a tournament first via /tournament/build.")
    return {"tournament_id": tournament_id, **result}


# ------------------------------------------- v4: MC prediction on REAL data
@app.get("/api/rl/mc/real/{dataset_id}")
def mc_real(dataset_id: str, gamma: float = 0.9):
    """First-/every-visit MC prediction on the uploaded dataset's own sessions.
    REAL DATA (derived states) — prediction only, no actions are fabricated."""
    ds = _get_dataset(dataset_id)
    if ds["engineered"] is None:
        raise HTTPException(400, "Call /engineer/{dataset_id} first.")
    episodes, notes = mc.episodes_from_engineered(ds["engineered"])
    if not episodes:
        raise HTTPException(400, "No sessions could be built from this dataset.")
    fv = mc.first_visit_prediction(episodes, gamma, core.N_STATES_WITH_TERMINAL)
    ev = mc.every_visit_prediction(episodes, gamma, core.N_STATES_WITH_TERMINAL)
    lengths = [len(e["states"]) for e in episodes]
    returns = [mc.episode_returns(e["rewards"], gamma)[0] for e in episodes]
    return rl_api.J({
        "n_episodes": len(episodes), "mean_length": float(sum(lengths) / len(lengths)),
        "mean_return": float(sum(returns) / len(returns)),
        "V_first_visit": fv["V"][:core.N_STATES], "V_every_visit": ev["V"][:core.N_STATES],
        "N_first_visit": fv["N"][:core.N_STATES], "N_every_visit": ev["N"][:core.N_STATES],
        "states_observed": int((fv["N"][:core.N_STATES] > 0).sum()),
        "notes": notes, "provenance": "REAL DATA (derived features); rewards use FALLBACK/PROXY terms",
        "control_available": False,
        "why_no_control": "The dataset has no matchmaking actions or behaviour probabilities.",
    })


# ------------------------------------------------ local dev: serve the UI too
# (On Vercel the static files are served by the platform; this only matters
#  for `uvicorn backend_app:app`, so http://localhost:8000/ opens the app.)
import os as _os
from fastapi.responses import FileResponse as _FileResponse
from fastapi.staticfiles import StaticFiles as _StaticFiles

_ROOT = _os.path.dirname(_os.path.abspath(__file__))
if _os.path.isdir(_os.path.join(_ROOT, "assets")):
    app.mount("/assets", _StaticFiles(directory=_os.path.join(_ROOT, "assets")), name="assets")


@app.get("/", include_in_schema=False)
def _index():
    return _FileResponse(_os.path.join(_ROOT, "index.html"))


def _get_dataset(dataset_id: str) -> dict:
    ds = DATASETS.get(dataset_id)
    if not ds:
        raise HTTPException(404, "Unknown dataset_id — upload a dataset first via /upload.")
    return ds
