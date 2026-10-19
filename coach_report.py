"""
coach_report.py
"CRAZY FEATURE #4" — Auto-Generated Coach Report.

Produces a natural-language player profile summary purely from the
engineered features — no external LLM call required, so this works fully
offline and deterministically (good for a live demo with no API-key risk).
Template-based generation, but the templates are chosen dynamically from
real thresholds on real computed stats, not static text.
"""


def generate_report(player: dict, forecast: dict | None = None) -> str:
    name = player.get("name", "Player")
    archetype = player.get("archetype", "Flex Rotator")
    tilt = player.get("tilt_score", 0)
    fatigue = player.get("fatigue_index", 0)
    win_rate = player.get("win_rate")
    matches = player.get("matches_played")
    rage_rate = player.get("rage_quit_rate")
    mmr = player.get("mmr")

    lines = [f"## Player Report — {name}", ""]
    lines.append(f"**Archetype:** {archetype}  ·  **MMR:** {mmr}")
    if matches:
        lines.append(f"**Matches analyzed:** {matches}"
                      + (f"  ·  **Win rate:** {win_rate:.0%}" if win_rate is not None else ""))
    lines.append("")

    # tilt narrative
    if tilt >= 0.7:
        lines.append(f"⚠️ **Tilt is high ({tilt:.2f}).** This player is showing strong signs of "
                      f"frustration — likely a losing streak combined with below-baseline performance. "
                      f"Matchmaking should actively avoid stacking additional high-tilt teammates with them "
                      f"right now.")
    elif tilt >= 0.4:
        lines.append(f"**Tilt is moderate ({tilt:.2f}).** Worth monitoring — not yet critical, but the "
                      f"session optimizer should lean toward easier lobbies if this keeps climbing.")
    else:
        lines.append(f"**Tilt is low ({tilt:.2f}).** Player is in a good headspace — safe to raise "
                      f"competitive intensity if they're winning.")

    # fatigue narrative
    if fatigue >= 0.75:
        lines.append(f"🔋 **Fatigue is high ({fatigue:.2f}).** Deep into a long session — reaction time and "
                      f"decision quality are likely degraded. A forgiving lobby protects both this player's "
                      f"experience and their teammates'.")
    elif fatigue >= 0.4:
        lines.append(f"**Fatigue is building ({fatigue:.2f}).** Still fine to compete, but approaching the "
                      f"point where difficulty should start easing.")
    else:
        lines.append(f"**Fatigue is low ({fatigue:.2f}).** Fresh session — full competitive intensity is fine.")

    if rage_rate is not None and rage_rate > 0:
        lines.append(f"\n📉 Historically, **{rage_rate:.0%}** of this player's sessions ended in a "
                      f"probable rage-quit pattern (loss + high tilt on their last match of the day).")

    if forecast and forecast.get("status") == "ok":
        lines.append(f"\n**Trend forecast:** tilt is currently **{forecast['trend']}** "
                      f"({forecast['tilt_slope_per_session']:+.3f}/session). "
                      + (f"Projected to cross the danger threshold in **{forecast['sessions_until_danger_zone']} "
                         f"more sessions** at this rate."
                         if forecast.get("sessions_until_danger_zone") is not None
                         else "No danger-zone crossing projected at the current trend."))
        lines.append(f"\n**Recommendation:** {forecast['recommendation']}")

    return "\n".join(lines)
