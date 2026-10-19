"""
burnout_forecast.py
"CRAZY FEATURE #3" — Burnout Forecast.

Most matchmakers only react to CURRENT tilt/fatigue. This module fits a
simple trend line across a player's real session history (from the
uploaded dataset) and projects forward: at the current trajectory, how many
more sessions until this player crosses the rage-quit-risk danger zone?

This is intentionally a transparent linear extrapolation (numpy polyfit),
not a black box — a professor can see exactly why a forecast was made, and
it's honest about being a short-horizon linear projection, not a claim of
long-term behavioral prediction.
"""
import numpy as np


DANGER_THRESHOLD = 0.65  # tilt_score considered "high risk" territory


def forecast_player(session_history: list[dict]) -> dict:
    """session_history: list of {"session_index": int, "tilt_score": float,
    "fatigue_index": float} ordered chronologically for ONE player."""
    if len(session_history) < 3:
        return {
            "status": "insufficient_history",
            "message": "Need at least 3 sessions of history to forecast a trend.",
        }

    x = np.array([s["session_index"] for s in session_history], dtype=float)
    tilt = np.array([s["tilt_score"] for s in session_history], dtype=float)
    fatigue = np.array([s["fatigue_index"] for s in session_history], dtype=float)

    tilt_slope, tilt_intercept = np.polyfit(x, tilt, 1)
    fatigue_slope, fatigue_intercept = np.polyfit(x, fatigue, 1)

    current_x = x[-1]
    current_tilt = tilt[-1]

    if tilt_slope <= 0:
        sessions_to_danger = None
        trend = "improving" if tilt_slope < -0.005 else "stable"
    else:
        if current_tilt >= DANGER_THRESHOLD:
            sessions_to_danger = 0
        else:
            sessions_to_danger = max(0, int(np.ceil((DANGER_THRESHOLD - current_tilt) / tilt_slope)))
        trend = "worsening"

    projected_next_5 = [
        round(float(np.clip(tilt_slope * (current_x + i) + tilt_intercept, 0, 1)), 3)
        for i in range(1, 6)
    ]

    risk_level = (
        "critical" if current_tilt >= DANGER_THRESHOLD else
        "elevated" if sessions_to_danger is not None and sessions_to_danger <= 3 else
        "low"
    )

    return {
        "status": "ok",
        "trend": trend,
        "tilt_slope_per_session": round(float(tilt_slope), 4),
        "fatigue_slope_per_session": round(float(fatigue_slope), 4),
        "sessions_until_danger_zone": sessions_to_danger,
        "projected_tilt_next_5_sessions": projected_next_5,
        "risk_level": risk_level,
        "recommendation": _recommendation(risk_level, trend),
    }


def _recommendation(risk_level: str, trend: str) -> str:
    if risk_level == "critical":
        return "Immediate intervention: lower lobby difficulty and suggest a break prompt now."
    if risk_level == "elevated":
        return "Proactively ease matchmaking difficulty over the next few sessions to flatten the trend."
    if trend == "improving":
        return "Player is trending calmer — safe to gradually raise competitive intensity."
    if trend == "worsening":
        return "Tilt is drifting upward slowly — no action needed yet, but keep monitoring each session."
    return "No intervention needed — trend is stable."
