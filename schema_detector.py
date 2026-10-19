"""
schema_detector.py
"CRAZY FEATURE #1" — Universal Dataset Adapter.

Instead of hard-coding "this backend only works with COD data," this module
inspects ANY uploaded CSV (League of Legends, Dota 2, CS:GO/CS2, Valorant,
COD, a custom scrim tracker, whatever) and automatically guesses which
column plays which role in the matchmaking pipeline:

    kills / eliminations column   -> aggression signal
    deaths column                 -> survivability / tilt signal
    score / damage / assists      -> performance signal
    date / timestamp column       -> session + streak sequencing
    win/result/outcome column     -> win/loss for streak computation
    player-id / name column       -> identity for grouping (if present)
    game-mode / map column        -> archetype hinting

Detection is done with column-name heuristics (regex over common gaming
vocab across multiple titles) PLUS dtype/statistics fallbacks (e.g. a
0/1 or True/False column with ~50% mean is very likely a win flag even if
it's named something unhelpful like "col_14").

This is what lets a professor hand you a totally different CSV and have the
same backend "just work" instead of needing a rewrite per dataset.
"""
import re
import pandas as pd
import numpy as np

# vocabulary is deliberately cross-game: COD, LoL, Dota2, CS:GO/CS2, Valorant,
# Overwatch, generic esports trackers
ROLE_PATTERNS = {
    "kills": r"kill|elim|frag|takedown",
    "deaths": r"death|died|down",
    "assists": r"assist|support",
    "score": r"\bscore\b|points?$|rating",
    "damage": r"damage|dmg",
    "xp": r"\bxp\b|experience",
    "date": r"date|timestamp|time_?stamp|match_?date|created_?at",
    "result": r"result|outcome|win|victory|w_?l\b",
    "player_id": r"player|summoner|username|gamertag|steam_?id|user_?id|name",
    "match_id": r"match_?id|game_?id|round_?id",
    "mode": r"mode|game_?type|queue|map",
    "duration": r"duration|length|time_?min|time_?sec|minutes?$",
    "team": r"\bteam\b|side",
}

DETECT_ORDER = ["match_id", "player_id", "date", "result", "kills", "deaths",
                 "assists", "score", "damage", "xp", "mode", "duration", "team"]


def score_column_name(col_name: str, pattern: str) -> float:
    return 1.0 if re.search(pattern, col_name.lower()) else 0.0


def detect_schema(df: pd.DataFrame) -> dict:
    """Returns {role: column_name_or_None, confidence: {role: float}} for
    every role we know how to use. Falls back to dtype/statistics heuristics
    when column names are uninformative."""
    columns = list(df.columns)
    detected = {}
    confidence = {}

    for role in DETECT_ORDER:
        pattern = ROLE_PATTERNS[role]
        candidates = [(c, score_column_name(c, pattern)) for c in columns]
        candidates = [c for c in candidates if c[1] > 0]
        if candidates:
            # prefer the candidate whose dtype makes sense for the role
            best = _pick_best_candidate(df, candidates, role)
            detected[role] = best
            confidence[role] = 0.9 if best else 0.0
        else:
            detected[role] = None
            confidence[role] = 0.0

    # dtype-fallback pass for result/win column if name-based search failed
    if detected.get("result") is None:
        detected["result"], confidence["result"] = _fallback_binary_column(df)

    # dtype-fallback for date if nothing matched by name
    if detected.get("date") is None:
        detected["date"], confidence["date"] = _fallback_date_column(df)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    used = set(v for v in detected.values() if v)
    unused_numeric = [c for c in numeric_cols if c not in used]

    return {
        "detected_columns": detected,
        "confidence": confidence,
        "unused_numeric_columns": unused_numeric,
        "all_columns": columns,
        "row_count": len(df),
    }


def _pick_best_candidate(df, candidates, role):
    # reject columns that are clearly binary yes/no flags for identity-like roles —
    # a "DidPlayerVote" Yes/No column shouldn't win the player_id role
    if role == "player_id":
        filtered = [c for c in candidates if df[c[0]].nunique(dropna=True) > 2]
        if not filtered:
            return None  # every name-matched candidate is a binary flag, not a real identity column
        candidates = filtered

    if role == "mode" and len(candidates) > 1:
        # prefer an explicit "type"/"mode"/"queue" column over a "map" column
        specific = [c for c in candidates if re.search(r"mode|game_?type|queue", c[0].lower())]
        if specific:
            candidates = specific

    if len(candidates) == 1:
        return candidates[0][0]
    # numeric roles prefer numeric dtype columns
    numeric_roles = {"kills", "deaths", "assists", "score", "damage", "xp", "duration"}
    if role in numeric_roles:
        for c, _ in candidates:
            if pd.api.types.is_numeric_dtype(df[c]):
                return c
    return candidates[0][0]


def _fallback_binary_column(df: pd.DataFrame):
    """Looks for a 0/1, True/False, or Win/Loss style column by statistics
    rather than name, in case the dataset calls it something odd."""
    for col in df.columns:
        series = df[col]
        uniques = series.dropna().unique()
        if len(uniques) == 2:
            vals = set(str(u).strip().lower() for u in uniques)
            binary_like = vals.issubset({"0", "1", "true", "false", "win", "loss",
                                          "w", "l", "yes", "no"})
            if binary_like:
                return col, 0.6
    return None, 0.0


def _fallback_date_column(df: pd.DataFrame):
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            continue  # plain numeric columns are never dates here
        sample = df[col].dropna().astype(str).head(20)
        if len(sample) == 0:
            continue
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                parsed = pd.to_datetime(sample, errors="coerce")
            if parsed.notna().mean() > 0.8:
                return col, 0.5
        except Exception:
            continue
    return None, 0.0


def build_schema_report(df: pd.DataFrame) -> dict:
    """Human-readable version for the frontend confirmation screen."""
    schema = detect_schema(df)
    report_rows = []
    for role, col in schema["detected_columns"].items():
        report_rows.append({
            "role": role,
            "matched_column": col,
            "confidence": schema["confidence"].get(role, 0.0),
            "sample_values": (
                df[col].dropna().astype(str).head(3).tolist() if col else []
            ),
        })
    return {
        "rows": report_rows,
        "row_count": schema["row_count"],
        "unused_numeric_columns": schema["unused_numeric_columns"],
        "usable": schema["detected_columns"].get("kills") is not None
                  or schema["detected_columns"].get("score") is not None,
    }
