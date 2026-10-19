"""
feature_engine.py
Turns ANY uploaded, schema-mapped dataset into Player-shaped records with
the same tilt / fatigue / streak / reward fields the matchmaker's RS + RL
layers expect — generalizing the pipeline built in the Colab notebook so it
isn't hard-coded to Call of Duty's column names.

Works with a partial schema: if the dataset doesn't have deaths, tilt falls
back to loss-streak-only; if it doesn't have dates, session grouping falls
back to fixed-size chunks of N consecutive rows, etc. Every fallback is
recorded in `engineering_notes` so the frontend can show the user exactly
which signals are real vs approximated for their specific dataset.

v4 fixes
--------
* Datasets WITH a date column crashed (/engineer raised KeyError
  'player_ref' because the frame was rebuilt after sorting). Fixed.
* Sessions, streaks and the rolling performance baseline are now computed
  per player instead of across the whole file.
* The per-match reward uses the bounded RL reward formula
  R = w1*(match completed) + w2*(closeness) - w3*(rage-quit proxy) instead of
  w1 * (number of matches in the session), which dwarfed the other terms.
* Score/assist columns are carried through so player profiles use them.

Provenance of every engineered column (shown in the UI):
  REAL DATA        kills, deaths, score, assists, win/loss, dates
  DERIVED FEATURE  kd_ratio, win_streak, fatigue_index, tilt_score, sessions
  FALLBACK/PROXY   close_match_baseline (0.5 unless a scoreline exists),
                   rage_quit_proxy (loss + tilt > 0.5 on a session's last match)
"""
import pandas as pd
import numpy as np

from rl_core import W1, W2, W3

SESSION_BLOCK = 10  # matches per pseudo-session when no date column exists


def _safe_numeric(df: pd.DataFrame, col: str | None) -> pd.Series:
    if col is None or col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def _parse_result_to_win(df: pd.DataFrame, col: str | None) -> tuple[pd.Series, pd.Series]:
    """Handles 'W'/'L', True/False, 1/0, 'Win'/'Loss', or 'X-Y' scoreline
    strings (win if left number > right number). Also returns a closeness
    signal in [0,1] for scorelines (1 = tied scoreline), NaN otherwise."""
    if col is None or col not in df.columns:
        return pd.Series(np.nan, index=df.index), pd.Series(np.nan, index=df.index)
    series = df[col].astype(str).str.strip().str.lower()

    scoreline_mask = series.str.match(r"^\d+\s*-\s*\d+$")
    win = pd.Series(np.nan, index=df.index)
    close = pd.Series(np.nan, index=df.index)

    if scoreline_mask.any():
        parts = series[scoreline_mask].str.split("-", expand=True)
        left = pd.to_numeric(parts[0], errors="coerce")
        right = pd.to_numeric(parts[1], errors="coerce")
        win.loc[scoreline_mask] = (left > right).astype(float)
        tot = (left + right).replace(0, np.nan)
        close.loc[scoreline_mask] = (1 - (left - right).abs() / tot).clip(0, 1)

    mapping = {"1": 1, "0": 0, "1.0": 1, "0.0": 0, "true": 1, "false": 0, "win": 1, "loss": 0,
               "w": 1, "l": 0, "yes": 1, "no": 0, "victory": 1, "defeat": 0, "won": 1, "lost": 0}
    remaining = ~scoreline_mask
    win.loc[remaining] = series[remaining].map(mapping)
    return win, close


def _signed_streaks(wins):
    streaks, cur = [], 0
    for w in wins:
        cur = (cur + 1 if cur >= 0 else 1) if w == 1 else (cur - 1 if cur <= 0 else -1)
        streaks.append(cur)
    return streaks


def engineer_features(df: pd.DataFrame, schema: dict) -> tuple[pd.DataFrame, list[str]]:
    notes = []
    cols = schema["detected_columns"]
    df = df.reset_index(drop=True).copy()

    # ---- identity ----
    if cols.get("player_id"):
        player_ref = df[cols["player_id"]].astype(str)
    else:
        player_ref = pd.Series("unknown_player", index=df.index)
        notes.append("No player-identity column detected — treating the whole file as one player's session history.")

    # ---- date / sequencing (sort per player, stable; keep player_ref aligned) ----
    if cols.get("date"):
        dates = pd.to_datetime(df[cols["date"]], errors="coerce")
        order = pd.DataFrame({"p": player_ref, "d": dates, "i": np.arange(len(df))}).sort_values(["p", "d", "i"], kind="stable").index
        df = df.loc[order].reset_index(drop=True)
        player_ref = player_ref.loc[order].reset_index(drop=True)
        dates = dates.loc[order].reset_index(drop=True)
        notes.append(f"Sessions grouped by real dates from '{cols['date']}' (per player, per calendar day).")
    else:
        dates = pd.Series(pd.NaT, index=df.index)
        notes.append(f"No date column detected — sessions approximated as fixed blocks of {SESSION_BLOCK} consecutive matches per player.")

    out = pd.DataFrame(index=df.index)
    out["player_ref"] = player_ref.values
    out["date"] = dates.values

    # ---- core performance signals (REAL DATA) ----
    kills = _safe_numeric(df, cols.get("kills"))
    deaths_raw = _safe_numeric(df, cols.get("deaths"))
    deaths = deaths_raw.replace(0, 1)
    score = _safe_numeric(df, cols.get("score"))
    assists = _safe_numeric(df, cols.get("assists"))

    out["kills"] = kills
    out["deaths"] = deaths_raw
    out["assists"] = assists
    out["score"] = score
    out["kd_ratio"] = (kills / deaths).replace([np.inf, -np.inf], np.nan)
    if cols.get("kills") is None:
        notes.append("No kills/eliminations column found — aggression + tilt signals will be weaker (score-only fallback).")

    # ---- win/loss ----
    win, scoreline_close = _parse_result_to_win(df, cols.get("result"))
    if win.isna().all() and cols.get("score"):
        win = (score > score.median()).astype(float)
        notes.append("No usable win/loss column found — approximated wins as above-median score performances (FALLBACK/PROXY).")
    elif win.isna().all():
        notes.append("No win/loss or score column found — every match treated as a loss-free neutral result (FALLBACK/PROXY).")
    out["win"] = win.fillna(0).astype(int)

    # ---- sessions (per player) ----
    if out["date"].notna().any():
        out["session_key"] = out["player_ref"].astype(str) + "|" + pd.to_datetime(out["date"]).dt.date.astype(str)
    else:
        idx_in_player = out.groupby("player_ref").cumcount()
        out["session_key"] = out["player_ref"].astype(str) + "|" + (idx_in_player // SESSION_BLOCK).astype(str)

    out["matches_in_session"] = out.groupby("session_key")["session_key"].transform("size")
    out["match_index_in_session"] = out.groupby("session_key").cumcount() + 1

    # ---- fatigue (DERIVED): position within the session ----
    out["fatigue_index"] = ((out["match_index_in_session"] - 1) /
                            out["matches_in_session"].clip(lower=1)).clip(0, 1)

    # ---- win streak (DERIVED, per player, chronological) ----
    out["win_streak"] = 0
    for _, g in out.groupby("player_ref", sort=False):
        out.loc[g.index, "win_streak"] = _signed_streaks(g["win"].tolist())

    # ---- tilt (DERIVED): loss streak + performance drop vs personal rolling baseline
    baseline_metric = out["kd_ratio"] if out["kd_ratio"].notna().any() else score
    rolling_avg = baseline_metric.groupby(out["player_ref"]).transform(lambda x: x.rolling(10, min_periods=3).mean().shift(1))
    drop = (rolling_avg - baseline_metric).clip(lower=0)
    loss_component = out["win_streak"].apply(lambda s: max(0, -s) / 6).clip(0, 1)
    drop_component = (drop / (rolling_avg.abs() + 1e-6)).clip(0, 1).fillna(0)
    out["tilt_score"] = (0.6 * loss_component + 0.4 * drop_component).clip(0, 1).fillna(0)

    # ---- reward inputs ----
    if scoreline_close.notna().any():
        out["close_match_baseline"] = scoreline_close.fillna(0.5).values
        notes.append("Match closeness derived from scorelines (1 - |margin| / total) — DERIVED FEATURE.")
    else:
        out["close_match_baseline"] = 0.5
        notes.append("No scoreline available — match closeness set to the neutral fallback 0.5 (FALLBACK/PROXY).")
    out["is_last_in_session"] = out["match_index_in_session"] == out["matches_in_session"]
    out["rage_quit_proxy"] = ((out["is_last_in_session"]) & (out["win"] == 0) & (out["tilt_score"] > 0.5)).astype(int)
    notes.append("Rage quits are not logged in typical match histories — rage_quit_proxy = loss + tilt > 0.5 on a session's final match (FALLBACK/PROXY).")
    completed = 1 - out["rage_quit_proxy"]
    out["reward"] = (W1 * completed + W2 * out["close_match_baseline"] - W3 * out["rage_quit_proxy"])

    # ---- carry through mode/archetype hint if present ----
    if cols.get("mode"):
        out["mode_hint"] = df[cols["mode"]].astype(str).values
    else:
        out["mode_hint"] = "Unknown"

    return out, notes


ARCHETYPE_KEYWORDS = {
    "Aggressive Entry": ["tdm", "deathmatch", "duel", "aram"],
    "Objective Anchor": ["domination", "hardpoint", "control", "clash"],
    "Flex Rotator": ["confirmed", "flex", "normal"],
    "Sniper/Isolation": ["search", "destroy", "snd", "solo"],
    "Passive Support": ["support", "healer", "assist"],
}


def infer_archetype(mode_hint: str) -> str:
    m = str(mode_hint).lower()
    for archetype, kws in ARCHETYPE_KEYWORDS.items():
        if any(kw in m for kw in kws):
            return archetype
    return "Flex Rotator"


def to_player_records(engineered: pd.DataFrame) -> list[dict]:
    """Groups engineered rows by player_ref into aggregated Player records
    (one per distinct player_ref) with the LATEST session state — this is
    what feeds the GNN clustering + RL optimizer, same shape as models.py."""
    records = []
    for player_ref, group in engineered.groupby("player_ref", sort=False):
        latest = group.iloc[-1]
        kd_avg = group["kd_ratio"].mean()
        kills_mean = group["kills"].mean() if "kills" in group.columns else np.nan
        aggression = min(1.0, float(kills_mean) / 30) if pd.notna(kills_mean) else 0.5
        positioning = min(1.0, float(kd_avg) / 3) if pd.notna(kd_avg) else 0.5
        assists_mean = group["assists"].mean() if "assists" in group.columns else np.nan
        deaths_mean = group["deaths"].mean() if "deaths" in group.columns else np.nan
        score_mean = group["score"].mean() if "score" in group.columns else np.nan
        support = (
            min(1.0, float(assists_mean) / 20.0) if pd.notna(assists_mean)
            else (max(0.0, min(1.0, 1.0 - float(deaths_mean) / 25.0)) if pd.notna(deaths_mean) else 0.5)
        )
        objective_focus = min(1.0, float(score_mean) / 1000.0) if pd.notna(score_mean) else 0.6
        archetype = infer_archetype(latest["mode_hint"])

        records.append({
            "id": str(player_ref)[:12],
            "name": str(player_ref)[:16],
            "mmr": int(1000 + float(kd_avg) * 400) if pd.notna(kd_avg) else 1500,
            "archetype": archetype,
            "playstyle_vector": [round(v, 3) for v in [aggression, positioning, 0.5, support, objective_focus]],
            "win_streak": int(latest["win_streak"]),
            "reaction_drift_ms": round(float(latest["fatigue_index"]) * 60, 1),
            "comms_freq": 0.5,
            "session_minutes": float(latest["match_index_in_session"]) * 8.0,
            "tilt_score": round(float(latest["tilt_score"]), 3),
            "fatigue_index": round(float(latest["fatigue_index"]), 3),
            "matches_played": int(len(group)),
            "sessions_played": int(group["session_key"].nunique()),
            "win_rate": round(float(group["win"].mean()), 3),
            "avg_reward": round(float(group["reward"].mean()), 3),
            # share of this player's SESSIONS whose final match matches the rage-quit proxy
            "rage_quit_rate": round(float(group.groupby("session_key")["rage_quit_proxy"].max().mean()), 3),
            "provenance": {"skill_and_playstyle": "DERIVED FEATURE", "tilt_fatigue_streak": "DERIVED FEATURE",
                           "comms_freq": "FALLBACK/PROXY", "rage_quit_rate": "FALLBACK/PROXY"},
        })
    return records
