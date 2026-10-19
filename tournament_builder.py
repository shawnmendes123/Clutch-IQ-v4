"""
tournament_builder.py — Neural Queue v3
Two-CSV Tournament Builder

The tournament builder uses the supplied player-performance statistics as the
primary source of skill and tactical profile. A synthetic/hash-based rating is
only retained as a last-resort fallback for generic demo CSVs that contain no
usable performance signals.
"""
from __future__ import annotations

import hashlib
import random
import re
from itertools import combinations

import numpy as np
import pandas as pd

from models import PLAYSTYLE_ARCHETYPES
from gnn_chemistry import chemistry_score
from rl_optimizer import reward_function

TEAM_SIZE = 5


def _norm(col: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(col).lower())


def _find_col(df: pd.DataFrame, *candidates: str) -> str | None:
    normed = {_norm(c): c for c in df.columns}
    for cand in candidates:
        if _norm(cand) in normed:
            return normed[_norm(cand)]
    for cand in candidates:
        for n, orig in normed.items():
            if _norm(cand) in n:
                return orig
    return None


def _numeric_series(df: pd.DataFrame, col: str | None) -> pd.Series:
    if not col or col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _minmax(values: pd.Series, default: float = 0.5) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    if values.notna().sum() == 0:
        return pd.Series(default, index=values.index, dtype=float)
    lo, hi = values.min(), values.max()
    if pd.isna(lo) or pd.isna(hi) or hi <= lo:
        return pd.Series(default, index=values.index, dtype=float)
    return ((values - lo) / (hi - lo)).clip(0, 1).fillna(default)


def _closest_archetype(text: str) -> str:
    if not text:
        return "Flex Rotator"
    t = text.lower()
    keyword_map = {
        "Aggressive Entry": ["aggress", "entry", "rush", "slayer", "duelist", "assault"],
        "Passive Support": ["support", "passive", "healer", "utility"],
        "Objective Anchor": ["objective", "anchor", "hold", "defend", "obj"],
        "Sniper/Isolation": ["sniper", "scout", "isolat", "flank", "recon"],
        "Flex Rotator": ["flex", "rotat", "hybrid", "all-round", "allround"],
    }
    for arche, kws in keyword_map.items():
        if any(kw in t for kw in kws):
            return arche
    idx = int(hashlib.md5(t.encode()).hexdigest(), 16) % len(PLAYSTYLE_ARCHETYPES)
    return list(PLAYSTYLE_ARCHETYPES.keys())[idx]


def _pseudo_skill(name: str, archetype: str) -> int:
    h = int(hashlib.md5(f"{name}|{archetype}".encode()).hexdigest(), 16)
    return 900 + (h % 1701)


def _performance_profile(df: pd.DataFrame, cols: dict[str, str | None]) -> pd.DataFrame:
    """Derive comparable 0..1 player-performance signals from the supplied CSV.

    This keeps the builder data-driven: when players.csv contains standard
    esports statistics, skill and playstyle are derived from those statistics
    rather than from player-name hashes.
    """
    kills = _numeric_series(df, cols.get("kills"))
    deaths = _numeric_series(df, cols.get("deaths"))
    assists = _numeric_series(df, cols.get("assists"))
    score = _numeric_series(df, cols.get("score"))
    damage = _numeric_series(df, cols.get("damage"))
    accuracy = _numeric_series(df, cols.get("accuracy"))
    hs_pct = _numeric_series(df, cols.get("headshot_percentage"))
    kd = _numeric_series(df, cols.get("kd_ratio"))
    if kd.isna().all() and kills.notna().any():
        kd = kills / deaths.replace(0, np.nan)

    kills_n = _minmax(kills)
    deaths_n = _minmax(deaths)
    assists_n = _minmax(assists)
    score_n = _minmax(score)
    damage_n = _minmax(damage)
    accuracy_n = _minmax(accuracy)
    hs_n = _minmax(hs_pct)
    kd_n = _minmax(kd)

    survival_n = 1.0 - deaths_n
    aggression = (0.40 * kills_n + 0.25 * damage_n + 0.35 * kd_n).clip(0, 1)
    support = (0.65 * assists_n + 0.20 * accuracy_n + 0.15 * survival_n).clip(0, 1)
    sniper = (0.50 * hs_n + 0.30 * accuracy_n + 0.20 * kd_n).clip(0, 1)
    objective = (0.55 * score_n + 0.25 * assists_n + 0.20 * survival_n).clip(0, 1)
    overall = (
        0.25 * kd_n
        + 0.20 * score_n
        + 0.20 * damage_n
        + 0.15 * kills_n
        + 0.10 * assists_n
        + 0.05 * accuracy_n
        + 0.05 * hs_n
    ).clip(0, 1)

    profile = pd.DataFrame({
        "aggression": aggression,
        "support": support,
        "sniper": sniper,
        "objective": objective,
        "overall": overall,
        "positioning": (0.55 * accuracy_n + 0.45 * survival_n).clip(0, 1),
        "comms": assists_n,
    }, index=df.index)

    archetypes = []
    for idx in profile.index:
        a = float(profile.at[idx, "aggression"])
        s = float(profile.at[idx, "support"])
        sn = float(profile.at[idx, "sniper"])
        o = float(profile.at[idx, "objective"])
        # Require a meaningful margin before specializing; otherwise classify as flex.
        scores = {
            "Aggressive Entry": a,
            "Passive Support": s,
            "Sniper/Isolation": sn,
            "Objective Anchor": o,
        }
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        if ranked[0][1] - ranked[1][1] < 0.10:
            archetypes.append("Flex Rotator")
        else:
            archetypes.append(ranked[0][0])
    profile["archetype"] = archetypes
    return profile


# --------------------------------------------------------------- CSV load
def load_maps_csv(df: pd.DataFrame) -> list[dict]:
    rank_c = _find_col(df, "rank")
    name_c = _find_col(df, "map_name", "map", "name")
    size_c = _find_col(df, "size")
    layout_c = _find_col(df, "layout", "layout_style")
    modes_c = _find_col(df, "best_modes", "best_game_modes", "modes", "best mode")
    style_c = _find_col(df, "recommended_playstyle", "recommended_archetype", "playstyle", "recommended style")
    sniper_c = _find_col(df, "sniper_friendly", "sniper_friendliness", "sniper", "sniper-friendliness")
    desc_c = _find_col(df, "description", "desc")

    if name_c is None:
        raise ValueError("maps.csv needs a map name column (e.g. 'map_name').")

    maps = []
    for i, row in df.iterrows():
        sniper_raw = str(row[sniper_c]).strip().lower() if sniper_c else ""
        sniper_friendly = sniper_raw in ("yes", "y", "true", "1", "high", "medium") or (
            sniper_raw.replace(".", "", 1).isdigit() and float(sniper_raw) >= 0.6
        )
        maps.append({
            "rank": int(row[rank_c]) if rank_c and str(row[rank_c]).strip().isdigit() else i + 1,
            "map_name": str(row[name_c]).strip(),
            "size": str(row[size_c]).strip() if size_c else "Medium",
            "layout": str(row[layout_c]).strip() if layout_c else "Balanced",
            "best_modes": str(row[modes_c]).strip() if modes_c else "Team Deathmatch",
            "recommended_playstyle": str(row[style_c]).strip() if style_c else "Flex Rotator",
            "sniper_friendly": bool(sniper_friendly),
            "description": str(row[desc_c]).strip() if desc_c else "",
        })
    if not maps:
        raise ValueError("maps.csv has no usable rows.")
    return maps


def load_players_csv(df: pd.DataFrame) -> list[dict]:
    name_c = _find_col(df, "name", "player_name", "player", "gamertag")
    id_c = _find_col(df, "player_id", "playerid", "user_id", "steam_id")
    style_c = _find_col(df, "playstyle", "play_style", "role", "archetype")
    skill_c = _find_col(df, "skill", "mmr", "elo", "rating")
    team_c = _find_col(df, "team", "side")
    perf_cols = {
        "kills": _find_col(df, "kills", "eliminations", "frags"),
        "deaths": _find_col(df, "deaths"),
        "assists": _find_col(df, "assists"),
        "score": _find_col(df, "score", "points"),
        "damage": _find_col(df, "damage", "dmg"),
        "accuracy": _find_col(df, "accuracy"),
        "headshot_percentage": _find_col(df, "headshot_percentage", "headshot_pct", "hs_pct"),
        "kd_ratio": _find_col(df, "kd_ratio", "kda", "kill_death"),
    }

    if name_c is None and id_c is None:
        raise ValueError("players.csv needs a player name or player_id column.")

    profile = _performance_profile(df, perf_cols)
    has_perf = any(v is not None for v in perf_cols.values())
    seen = set()
    players = []
    for idx, row in df.iterrows():
        raw_name = str(row[name_c]).strip() if name_c else str(row[id_c]).strip()
        if not raw_name or raw_name.lower() == "nan":
            continue
        name = raw_name
        suffix = 1
        while name in seen:
            suffix += 1
            name = f"{raw_name}#{suffix}"
        seen.add(name)

        archetype = (
            _closest_archetype(str(row[style_c]).strip())
            if style_c and pd.notna(row[style_c])
            else str(profile.at[idx, "archetype"])
        )
        base_vec = PLAYSTYLE_ARCHETYPES[archetype]
        if style_c:
            rng = random.Random(name)
            vec = [max(0.0, min(1.0, v + rng.uniform(-0.05, 0.05))) for v in base_vec]
        else:
            vec = [
                round(float(profile.at[idx, "aggression"]), 3),
                round(float(profile.at[idx, "positioning"]), 3),
                round(float(profile.at[idx, "comms"]), 3),
                round(float(profile.at[idx, "support"]), 3),
                round(float(profile.at[idx, "objective"]), 3),
            ]

        if skill_c and pd.notna(row[skill_c]) and str(row[skill_c]).strip():
            try:
                mmr = int(round(float(row[skill_c])))
                rating_source = "provided_mmr"
            except ValueError:
                mmr = _pseudo_skill(name, archetype)
                rating_source = "fallback_hash"
        elif has_perf:
            mmr = int(round(1200 + 1000 * float(profile.at[idx, "overall"])))
            rating_source = "derived_from_performance"
        else:
            mmr = _pseudo_skill(name, archetype)
            rating_source = "fallback_hash"

        rng = random.Random(name)
        tilt = round(rng.uniform(0.05, 0.25), 3)
        fatigue = round(rng.uniform(0.05, 0.25), 3)
        players.append({
            "id": str(row[id_c]).strip() if id_c and pd.notna(row[id_c]) else hashlib.md5(name.encode()).hexdigest()[:8],
            "name": name,
            "source_team": str(row[team_c]).strip() if team_c and pd.notna(row[team_c]) else None,
            "playstyle_label": str(row[style_c]).strip() if style_c and pd.notna(row[style_c]) else None,
            "archetype": archetype,
            "playstyle_vector": vec,
            "mmr": mmr,
            "rating_source": rating_source,
            "performance_score": round(float(profile.at[idx, "overall"]), 3) if has_perf else None,
            "tilt_score": tilt,
            "fatigue_index": fatigue,
            "win_streak": 0,
        })

    if len(players) < TEAM_SIZE * 2:
        raise ValueError(
            f"players.csv needs at least {TEAM_SIZE * 2} players to form 2+ teams of "
            f"{TEAM_SIZE}, found {len(players)}."
        )
    return players


# ------------------------------------------------------------- team draft
def snake_draft_teams(players: list[dict]) -> tuple[list[list[dict]], list[dict]]:
    ordered = sorted(players, key=lambda p: p["mmr"], reverse=True)
    n_teams = len(ordered) // TEAM_SIZE
    draft_pool = ordered[: n_teams * TEAM_SIZE]
    benched = ordered[n_teams * TEAM_SIZE:]

    teams = [[] for _ in range(n_teams)]
    order = list(range(n_teams))
    i = 0
    for pick_round in range(len(draft_pool) // n_teams):
        pick_order = order if pick_round % 2 == 0 else list(reversed(order))
        for team_idx in pick_order:
            teams[team_idx].append(draft_pool[i])
            i += 1

    starting_totals = [sum(p["mmr"] for p in t) for t in teams]

    def combined_chem(tms):
        return sum(chemistry_score(t) for t in tms)

    improved = True
    passes = 0
    while improved and passes < 3:
        improved = False
        passes += 1
        for a, b in combinations(range(n_teams), 2):
            for ia in range(len(teams[a])):
                for ib in range(len(teams[b])):
                    trial = [t[:] for t in teams]
                    trial[a][ia], trial[b][ib] = trial[b][ib], trial[a][ia]
                    totals = [sum(p["mmr"] for p in t) for t in trial]
                    if any(
                        abs(totals[idx] - starting_totals[idx]) > 0.05 * starting_totals[idx]
                        for idx in (a, b)
                    ):
                        continue
                    if combined_chem(trial) > combined_chem(teams) + 1e-9:
                        teams = trial
                        improved = True
    return teams, benched


def _team_summary(team: list[dict], idx: int) -> dict:
    mmr_total = sum(p["mmr"] for p in team)
    dominant_archetype = pd.Series([p["archetype"] for p in team]).mode().iat[0]
    return {
        "team_id": f"T{idx + 1}",
        "players": [p["name"] for p in team],
        "avg_mmr": round(mmr_total / len(team), 1),
        "mmr_total": mmr_total,
        "chemistry_score": chemistry_score(team),
        "dominant_playstyle": dominant_archetype,
        "roster": team,
    }


def map_fit_score(m: dict, roster: list[dict]) -> float:
    """Rule-based map-fit score used for recommendations (and, in v4, as the
    DERIVED expected value of each bandit arm in bandits.map_arms)."""
    archetypes = pd.Series([p["archetype"] for p in roster])
    dominant = archetypes.mode().iat[0]
    sniper_share = sum(1 for p in roster if p["archetype"] == "Sniper/Isolation") / len(roster)
    score = 0.0
    map_style = _closest_archetype(m["recommended_playstyle"])
    if map_style == dominant:
        score += 2.0
    if m["sniper_friendly"] and sniper_share > 0.25:
        score += 1.5
    if not m["sniper_friendly"] and sniper_share <= 0.25:
        score += 0.5
    score += max(0.0, (21 - m["rank"]) / 40.0)
    return score


def recommend_map(team_a: dict, team_b: dict, maps: list[dict]) -> dict:
    roster = team_a["roster"] + team_b["roster"]
    archetypes = pd.Series([p["archetype"] for p in roster])
    dominant = archetypes.mode().iat[0]
    sniper_share = sum(1 for p in roster if p["archetype"] == "Sniper/Isolation") / len(roster)

    scored = [(map_fit_score(m, roster), m) for m in maps]
    scored.sort(key=lambda x: x[0], reverse=True)
    best = scored[0][1]
    return {
        "map_name": best["map_name"],
        "layout": best["layout"],
        "best_modes": best["best_modes"],
        "size": best["size"],
        "sniper_friendly": best["sniper_friendly"],
        "fit_score": round(scored[0][0], 3),
        "reason": f"Roster's dominant playstyle is '{dominant}' "
                  f"({sniper_share*100:.0f}% sniper/isolation players) — "
                  f"'{best['map_name']}' is recommended for {best['recommended_playstyle']} play.",
    }


def schedule_matches(team_summaries: list[dict], maps: list[dict]) -> list[dict]:
    pool = sorted(team_summaries, key=lambda t: t["avg_mmr"])
    matches = []
    used = set()
    match_num = 1

    for i, team in enumerate(pool):
        if team["team_id"] in used:
            continue
        candidates = [
            (j, other) for j, other in enumerate(pool)
            if j != i and other["team_id"] not in used
        ]
        if not candidates:
            break
        best_j, team_b = min(candidates, key=lambda pair: abs(team["avg_mmr"] - pair[1]["avg_mmr"]))
        used.add(team["team_id"])
        used.add(team_b["team_id"])

        mmr_gap = abs(team["avg_mmr"] - team_b["avg_mmr"])
        closeness = round(1 - min(1.0, mmr_gap / 600.0), 3)
        avg_chem = round((team["chemistry_score"] + team_b["chemistry_score"]) / 2, 3)
        avg_tilt = float(np.mean([p["tilt_score"] for p in team["roster"] + team_b["roster"]]))
        rage_quit_risk = bool(avg_tilt > 0.3 and closeness < 0.35)
        session_length_delta = round(0.4 + 0.6 * closeness - 0.3 * avg_tilt, 3)
        reward = round(reward_function(session_length_delta, closeness, rage_quit_risk), 3)
        map_rec = recommend_map(team, team_b, maps)

        matches.append({
            "roster_archetypes": [p["archetype"] for p in team["roster"] + team_b["roster"]],
            "match_id": f"M{match_num}",
            "team_a": team["team_id"],
            "team_b": team_b["team_id"],
            "team_a_players": team["players"],
            "team_b_players": team_b["players"],
            "mmr_gap": round(mmr_gap, 1),
            "predicted_closeness": closeness,
            "avg_chemistry": avg_chem,
            "recommended_map": map_rec,
            "rage_quit_risk": rage_quit_risk,
            "reward": reward,
        })
        match_num += 1

    return matches


def build_tournament(maps_df: pd.DataFrame, players_df: pd.DataFrame) -> dict:
    maps = load_maps_csv(maps_df)
    players = load_players_csv(players_df)

    raw_teams, benched = snake_draft_teams(players)
    team_summaries = [_team_summary(t, i) for i, t in enumerate(raw_teams)]
    matches = schedule_matches(team_summaries, maps)

    for t in team_summaries:
        t.pop("roster", None)

    return {
        "player_count": len(players),
        "benched_players": [p["name"] for p in benched],
        "team_count": len(team_summaries),
        "teams": team_summaries,
        "match_count": len(matches),
        "matches": matches,
        "map_pool_size": len(maps),
        "map_pool": maps,
        "provenance": {
            "skill": "REAL DATA when players.csv has MMR/skill; DERIVED FEATURE from performance stats; FALLBACK/PROXY hash otherwise",
            "tilt_fatigue": "FALLBACK/PROXY — players.csv carries no session state, so small name-seeded values are used",
            "rage_quit_risk": "MODEL ESTIMATE (rule: avg tilt > 0.3 and closeness < 0.35)",
            "map_recommendation": "rule-based map-fit score; compare with the learned Bandit Arena recommendation",
        },
        "rating_basis": "Derived from player performance statistics when available; explicit MMR/skill is preferred; deterministic hash is last-resort fallback.",
    }
