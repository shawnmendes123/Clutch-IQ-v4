"""Regenerates the data-driven parts of the documentation.

    python tools/build_docs.py

* SUBJECT_MAPPING.md        — fully generated from syllabus.py
* EXPERIMENTS.md            — tables generated from experiments/results.json
* VIVA_GUIDE.md             — the block between <!-- NUMBERS:BEGIN --> and
                              <!-- NUMBERS:END --> is regenerated
No result is typed by hand.
"""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import syllabus  # noqa: E402


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def subject_mapping():
    cov = syllabus.coverage()
    counts = ", ".join(f"{v} {k.lower()}" for k, v in cov["status_counts"].items())
    head = f"""# Subject mapping — Units 1–5 → Clutch IQ implementation

Generated from `syllabus.py` by `tools/build_docs.py` (the same data drives the
**Academic › Syllabus mapping** screen and `GET /api/academic/coverage`).

Status vocabulary: **IMPLEMENTED** = real computation, visible in the app and
unit-tested · **DEMONSTRATED** = property measured on live runs, not formally
proved · **EXPLAINED** = covered in explanation panels only · **NOT COVERED** =
in the syllabus but not part of this project.

Totals: {counts}.

Every worked example with a printed answer in the uploaded notes (Units 2–5) is
recomputed by the project code and checked — see **Academic › Notes numericals**
and `tests/`.
"""
    return head + syllabus.to_markdown() + "\n"


def experiments_md(r):
    E = {e["id"]: e for e in r["experiments"]}
    out = [f"""# Experiments — reproducible results

All numbers below were produced by `python rl_experiments.py` (seeded; full
suite ≈ {r['total_seconds']:.0f} s on a laptop CPU) and rendered by
`tools/build_docs.py` from `experiments/results.json`. Nothing is typed by hand
and no run was selected or discarded. **Provenance:** {r['provenance']}.

Common settings: γ = {r['gamma']} unless stated; the session MDP of
`rl_core.py`; the simulation-estimated model with 1000 samples per (s,a), seed 7;
evaluation starts from the realistic session-start distribution.

Reproduce: `python rl_experiments.py` (or `--quick` for a smoke run), then
`python tools/build_docs.py`.
"""]
    e = E["E1"]
    out.append(f"\n## E1 — {e['title']}\n\n| samples/(s,a) | max \\|p̂−p\\| | mean \\|p̂−p\\| | s.e. bound | π̂* = exact π* | J of planned π on true dynamics | exact J* |\n|---|---|---|---|---|---|---|")
    for x in e["rows"]:
        out.append(f"| {x['samples_per_pair']} | {x['max_abs_error']:.4f} | {x['mean_abs_error']:.4f} | {x['worst_case_se']:.4f} | {pct(x['policy_agreement_with_exact_optimum'])} | {x['J_of_planned_policy_on_exact_model']:.4f} | {x['J_exact_optimum']:.4f} |")
    first, last = e["rows"][0], e["rows"][-1]
    out.append(f"\n**Observation.** Model error shrinks roughly as 1/√n (max error {first['max_abs_error']:.3f} → {last['max_abs_error']:.3f}). The policy planned on the estimate agrees with the exact optimum on {pct(first['policy_agreement_with_exact_optimum'])} → {pct(last['policy_agreement_with_exact_optimum'])} of states and loses at most {pct(1 - min(x['J_of_planned_policy_on_exact_model'] for x in e['rows']) / first['J_exact_optimum'])} of value. **Interpretation:** planning on a simulation-estimated model is sound; residual disagreements are in near-tied states.")
    e = E["E2"]
    out.append(f"\n## E2 — {e['title']}\n\n| γ | VI sweeps (θ=1e-8) | max Δ ratio | PI improvement steps | PI evaluation sweeps | PI = VI | closed-form MRP gap |\n|---|---|---|---|---|---|---|")
    for x in e["rows"]:
        out.append(f"| {x['gamma']} | {x['vi_iterations']} | {x['vi_max_contraction_ratio']:.4f} | {x['pi_improvement_steps']} | {x['pi_eval_sweeps']} | {pct(x['pi_equals_vi'], 0)} | {x['closed_form_gap']:.2e} |")
    out.append("\n**Observation.** The measured contraction ratio is always below γ (Banach fixed point), so sweeps grow as γ → 1. Policy iteration needs only a few improvement steps and returns the same policy as value iteration. Iterative policy evaluation matches the closed-form MRP solution (I − γP_π)⁻¹R_π.")
    e = E["E3"]
    out.append(f"\n## E3 — {e['title']}\n\n| γ | LOWER | KEEP | RAISE | sim. rage-quit rate | sim. session length | sim. close-match rate |\n|---|---|---|---|---|---|---|")
    for x in e["rows"]:
        out.append(f"| {x['gamma']} | {x['n_lower']} | {x['n_keep']} | {x['n_raise']} | {pct(x['sim_rage_quit_rate'])} | {x['sim_session_length']:.2f} | {pct(x['sim_close_match_rate'])} |")
    a, b = e["rows"][0], e["rows"][-1]
    out.append(f"\n**Observation.** As γ grows the optimal controller lowers difficulty in more states ({a['n_lower']} → {b['n_lower']}) and raises it in fewer ({a['n_raise']} → {b['n_raise']}): a myopic controller chases close matches now (close-match rate {pct(a['sim_close_match_rate'])}), a far-sighted one protects tilt and session length (rage quits {pct(a['sim_rage_quit_rate'])} → {pct(b['sim_rage_quit_rate'])}). {e.get('note', '')}")
    e = E["E4"]
    out.append(f"\n## E4 — {e['title']}\n\nSeeds: {', '.join(map(str, e['seeds']))}. Mean ± sd across seeds.\n\n| episodes | MAE first-visit | MAE every-visit | states covered |\n|---|---|---|---|")
    for x in e["rows"]:
        out.append(f"| {x['episodes']} | {x['mae_first_visit_mean']:.3f} ± {x['mae_first_visit_std']:.3f} | {x['mae_every_visit_mean']:.3f} ± {x['mae_every_visit_std']:.3f} | {pct(x['coverage_mean'], 0)} |")
    out.append(f"\n**Observation.** Both estimators converge toward the DP values as experience accumulates (MAE {e['rows'][3]['mae_first_visit_mean']:.2f} at 100 episodes → {e['rows'][-1]['mae_first_visit_mean']:.2f} at {e['rows'][-1]['episodes']}); the residual includes the DP reference's own model error. First- and every-visit behave similarly here because revisits within an episode are short-range.")
    e = E["E5"]
    out.append(f"\n## E5 — {e['title']}\n\nReferences: J* (DP) = {e['J_star']:.3f}, manual heuristic = {e['J_heuristic']:.3f}, always KEEP = {e['J_keep']:.3f}.\n\n| algorithm | episodes | J (mean ± sd over seeds) | % of J* | agreement with DP | seeds |\n|---|---|---|---|---|---|")
    for x in e["rows"]:
        out.append(f"| {x['algorithm']} | {x['episodes']:,} | {x['J_mean']:.3f} ± {x['J_std']:.3f} | {pct(x['J_mean'] / e['J_star'])} | {pct(x['agreement_mean'])} | {x['seeds']} |")
    out.append("\n**Observation.** Model-free learners reach most of the DP optimum from experience alone; on-policy MC control converges to the best ε-soft policy; off-policy MC control learns only from episode tails after the last exploratory action and is clearly slowest at this budget.")
    e = E["E6"]
    s = e["settings"]
    out.append(f"\n## E6 — {e['title']}\n\n{s['n_episodes']} episodes, seed {s['seed']}, start = {s['start_dist']}, γ = {s['gamma']}, MC control {s['mc_episodes']:,} episodes (ε = {s['mc_epsilon']}), Q-learning {s['q_params']['episodes']:,} episodes, {s['stability_runs']} stability runs. ± = 95% CI; Δ is paired on identical episodes.\n\n| policy | discounted return | Δ vs DP (paired) | session length | rage quit | close match | agreement w/ DP | stability |\n|---|---|---|---|---|---|---|---|")
    for x in e["rows"]:
        st = "—" if x["stability"] is None else pct(x["stability"], 0)
        out.append(f"| {x['policy']} | {x['discounted_return']['mean']:.3f} ± {x['discounted_return']['ci95']:.3f} | {x['paired_diff_vs_dp']['mean']:+.3f} ± {x['paired_diff_vs_dp']['ci95']:.3f} | {x['session_length']['mean']:.2f} | {pct(x['rage_quit_rate']['mean'])} | {pct(x['close_match_rate']['mean'])} | {pct(x['agreement_with_dp'], 0)} | {st} |")
    rows = {x["policy"]: x for x in e["rows"]}
    dp = next(v for k, v in rows.items() if k.startswith("Dynamic Programming — Value"))
    keep = next(v for k, v in rows.items() if k.startswith("Always KEEP"))
    out.append(f"\n**Observation.** Relative to rank-only matchmaking (always KEEP), the DP policy raises the discounted return from {keep['discounted_return']['mean']:.2f} to {dp['discounted_return']['mean']:.2f} and cuts rage-quit sessions from {pct(keep['rage_quit_rate']['mean'])} to {pct(dp['rage_quit_rate']['mean'])}. Learned policies sit between the heuristic and DP; paired intervals show which gaps are real. DP's stability < 100% reflects model-estimation noise across model seeds (near-tied states).")
    e = E["E7"]
    out.append(f"\n## E7 — {e['title']}\n\nArms: {e['arm_source']}; {e['runs']} runs per algorithm, same seeds for every algorithm; ε = 0.1, UCB c = 1.414 (UCB1), optimistic Q₀ = 1.0, 50 exploration rounds.\n\n| map | fit score | q* |\n|---|---|---|")
    for a in e["arms"]:
        out.append(f"| {a['map_name']} | {a['fit_score']} | {a['q_star']} |")
    out.append("\n| algorithm | expected regret | total reward ± 95% CI | % optimal (last 10%) |\n|---|---|---|---|")
    for x in e["stationary"]:
        out.append(f"| {x['algorithm']} | {x['final_regret']:.1f} | {x['final_reward']:.1f} ± {x['final_reward_ci95']:.1f} | {pct(x['pct_optimal_last_10pct'])} |")
    out.append("\nε sweep (ε-greedy, 1000 rounds):\n\n| ε | regret | % optimal (last 10%) |\n|---|---|---|")
    for x in e["epsilon_sweep"]:
        out.append(f"| {x['epsilon']} | {x['final_regret']:.1f} | {pct(x['pct_optimal_last_10pct'])} |")
    out.append("\nNon-stationary (balance patch at T/2, 2000 rounds, ε-greedy):\n\n| update | total regret | regret after the patch | % optimal (last 10%) |\n|---|---|---|---|")
    for x in e["nonstationary_patch"]:
        out.append(f"| {x['update']} | {x['final_regret']:.1f} | {x['regret_after_patch']:.1f} | {pct(x['pct_optimal_last_10pct'])} |")
    out.append("\n**Observation.** Pure exploitation gets stuck; any systematic exploration cuts regret sharply. Regret is U-shaped in ε. UCB1's c = √2 is conservative for rewards in [0, 1] (it keeps exploring); smaller c lets UCB overtake ε-greedy (try it in the Bandit arena). After a balance patch, a constant step size α tracks the new best map while sample averages react slowly — the notes' argument for α in non-stationary problems.")
    e = E["E8"]
    out.append(f"\n## E8 — {e['title']}\n\n| scenario | truth (DP) | logged policy's value | ordinary IS | weighted IS | per-decision IS | discounting-aware WIS | ESS | diagnostic |\n|---|---|---|---|---|---|---|---|---|")
    for x in e["rows"]:
        out.append(f"| {x['scenario']} | {x['truth_dp']:.3f} | {x['behaviour_value']:.3f} | {x['ordinary_is']:.3f} | {x['weighted_is']:.3f} | {x['per_decision_is']:.3f} | {x['discounting_aware_wis']:.3f} | {x['ess']:.0f} | {'reliable' if x['reliability'] == 'OK' else 'unreliable'} |")
    out.append("\n**Observation.** For a small policy change, importance sampling recovers the new policy's value from old logs (weighted IS within a few hundredths). For a large change, or uniform-random logs, the ratios become heavy-tailed, the effective sample size collapses and the estimates are biased low at this sample size; the diagnostic flags them instead of reporting false confidence. Per-decision IS is the most robust of the estimators here.")
    nn = r["notes_numericals"]
    out.append(f"\n## Lecture-note numericals\n\n{nn['matched']} of {nn['checked']} worked examples with a printed answer reproduce exactly ({', '.join(nn['ids'])}).\n")
    return "\n".join(out) + "\n"


def viva_numbers(r):
    E = {e["id"]: e for e in r["experiments"]}
    rows = {x["policy"]: x for x in E["E6"]["rows"]}
    dp = next(v for k, v in rows.items() if k.startswith("Dynamic Programming — Value"))
    keep = next(v for k, v in rows.items() if k.startswith("Always KEEP"))
    ql = next(v for k, v in rows.items() if k.startswith("Tabular"))
    e5, e7, e8 = E["E5"], E["E7"], E["E8"]
    g = {x["key"]: x for x in e7["stationary"]}
    return "\n".join([
        "<!-- NUMBERS:BEGIN -->",
        "## Numbers to quote (generated from experiments/results.json)",
        "",
        f"- DP-optimal vs always-KEEP on {E['E6']['settings']['n_episodes']} identical simulated sessions: discounted return {keep['discounted_return']['mean']:.2f} → {dp['discounted_return']['mean']:.2f}; rage-quit sessions {pct(keep['rage_quit_rate']['mean'])} → {pct(dp['rage_quit_rate']['mean'])}.",
        f"- Tabular Q-learning agrees with DP on {pct(ql['agreement_with_dp'], 0)} of states; paired gap {ql['paired_diff_vs_dp']['mean']:+.3f} ± {ql['paired_diff_vs_dp']['ci95']:.3f}.",
        f"- Model-free J as % of J*: " + ", ".join(f"{x['algorithm']} {pct(x['J_mean'] / e5['J_star'])}" for x in e5["rows"]) + ".",
        f"- Map bandit, 1000 rounds: regret greedy {g['greedy']['final_regret']:.0f}, ε-greedy {g['epsilon_greedy']['final_regret']:.0f}, optimistic {g['optimistic']['final_regret']:.0f}, UCB {g['ucb']['final_regret']:.0f}.",
        f"- Importance sampling, small change: truth {e8['rows'][0]['truth_dp']:.3f}, WIS {e8['rows'][0]['weighted_is']:.3f} (ESS {e8['rows'][0]['ess']:.0f}); large change ESS {e8['rows'][1]['ess']:.0f} → flagged unreliable.",
        f"- Notes numericals reproduced: {r['notes_numericals']['matched']}/{r['notes_numericals']['checked']}.",
        "<!-- NUMBERS:END -->",
    ])


def main():
    with open(os.path.join(ROOT, "SUBJECT_MAPPING.md"), "w", encoding="utf-8") as f:
        f.write(subject_mapping())
    rp = os.path.join(ROOT, "experiments", "results.json")
    if os.path.exists(rp):
        with open(rp) as f:
            r = json.load(f)
        with open(os.path.join(ROOT, "EXPERIMENTS.md"), "w", encoding="utf-8") as f:
            f.write(experiments_md(r))
        vp = os.path.join(ROOT, "VIVA_GUIDE.md")
        if os.path.exists(vp):
            txt = open(vp, encoding="utf-8").read()
            txt = re.sub(r"<!-- NUMBERS:BEGIN -->.*<!-- NUMBERS:END -->", lambda m: viva_numbers(r), txt, flags=re.S)
            open(vp, "w", encoding="utf-8").write(txt)
    print("docs regenerated")


if __name__ == "__main__":
    main()
