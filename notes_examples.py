"""
notes_examples.py — the worked examples printed in the uploaded lecture notes,
recomputed by the SAME project code that drives Clutch IQ.

Each entry reports the engine's answer, the answer printed in the notes and
whether they match. This proves the generic implementations (MDP, DP, MC,
bandits) are correct before they are applied to the matchmaking problem.
Served by /api/academic/numericals and shown on Academic › Notes Numericals.
"""
from __future__ import annotations

import numpy as np

import rl_core as core
import mdp_model as mm
import dynamic_programming as dp
import monte_carlo as mc
import bandits


def dice_game() -> mm.FiniteMDP:
    """Units 3/4: IN/END, actions STAY/QUIT (Kuang, Fundamentals of RL)."""
    P = np.zeros((2, 2, 2)); R = np.zeros((2, 2, 2))
    STAY, QUIT, IN, END = 0, 1, 0, 1
    P[IN, QUIT, END] = 1.0; R[IN, QUIT, END] = 10
    P[IN, STAY, IN] = 2 / 3; R[IN, STAY, IN] = 4
    P[IN, STAY, END] = 1 / 3; R[IN, STAY, END] = 4
    return mm.FiniteMDP(P, R, np.array([False, True]), 1.0, ["IN", "END"], ["STAY", "QUIT"])


def gridworld_4x4() -> mm.FiniteMDP:
    """Unit 4 notes: 4x4 gridworld, reward -1 per move, terminal corners, gamma = 1."""
    n = 16
    P = np.zeros((n, 4, n)); R = np.full((n, 4, n), -1.0)
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for s in range(n):
        r, c = divmod(s, 4)
        for a, (dr, dc) in enumerate(moves):
            nr, nc = r + dr, c + dc
            P[s, a, nr * 4 + nc if 0 <= nr < 4 and 0 <= nc < 4 else s] = 1.0
    term = np.zeros(n, bool); term[0] = term[15] = True
    return mm.FiniteMDP(P, R, term, 1.0, [str(i) for i in range(n)], ["U", "D", "L", "R"])


def exercise_3_22(gamma: float) -> mm.FiniteMDP:
    """Unit 3 notes (Sutton Ex. 3.22): continuing MDP, top state chooses left/right.
    States: 0 = top, 1 = left, 2 = right. left: +1 then 0; right: 0 then +2."""
    P = np.zeros((3, 2, 3)); R = np.zeros((3, 2, 3))
    P[0, 0, 1] = 1; R[0, 0, 1] = 1        # left
    P[0, 1, 2] = 1; R[0, 1, 2] = 0        # right
    for a in (0, 1):
        P[1, a, 0] = 1; R[1, a, 0] = 0
        P[2, a, 0] = 1; R[2, a, 0] = 2
    return mm.FiniteMDP(P, R, np.zeros(3, bool), gamma, ["top", "left", "right"], ["left", "right"])


MC_AB_EPISODES = [
    ([0, 0, 1, 1], [2, 6, 1, 0]),
    ([0, 1, 0, 1, 1], [3, 2, 4, 2, 0]),
    ([1, 1, 0, 1], [3, 6, 1, 0]),
    ([0, 1, 0, 1, 1, 1], [0, 2, 4, 4, 2, 0]),
    ([1, 1], [8, 0]),
]


def _close(a, b, tol=1e-6):
    return abs(float(a) - float(b)) <= tol


def all_numericals() -> list[dict]:
    out = []
    # ---------------- Unit 1/3: returns -----------------------------------
    G = core.returns_from_rewards([-1, 2, 6, 3, 2], 0.5)
    out.append({"id": "u3-ex3.8", "unit": 3, "title": "Returns worked backwards (Ex. 3.8, γ = 0.5)",
                "result": {"G": G + [0.0]}, "notes_answer": {"G": [2, 6, 8, 4, 2, 0]},
                "match": all(_close(a, b) for a, b in zip(G + [0.0], [2, 6, 8, 4, 2, 0])),
                "code": "rl_core.returns_from_rewards"})
    g13 = core.discounted_return([13] * 200, 0.5)
    out.append({"id": "u3-const13", "unit": 3, "title": "Constant reward 13, γ = 0.5 → G = 13/(1−γ)",
                "result": {"G": g13}, "notes_answer": {"G": 26}, "match": _close(g13, 26, 1e-6),
                "code": "rl_core.discounted_return"})
    # ---------------- Unit 3: Markov chain ---------------------------------
    P = np.zeros((2, 1, 2)); P[0, 0] = [0.3, 0.7]; P[1, 0] = [0.2, 0.8]
    chain = mm.FiniteMDP(P, np.zeros_like(P), np.array([False, False]), 1.0, ["Rain", "Dry"], ["-"])
    p = chain.sequence_probability([1, 1, 0, 0], np.array([[1.0], [1.0]]), start=np.array([0.4, 0.6]))
    out.append({"id": "u3-rain-dry", "unit": 3, "title": "Markov chain: P(Dry, Dry, Rain, Rain)",
                "result": {"probability": p, "formula": "P(Dry)·P(Dry|Dry)·P(Rain|Dry)·P(Rain|Rain) = 0.6·0.8·0.2·0.3"},
                "notes_answer": {"probability": 0.0288}, "match": _close(p, 0.0288), "code": "mdp_model.FiniteMDP.sequence_probability"})
    # ---------------- Units 3/4: dice game ---------------------------------
    m = dice_game()
    rand = np.array([[0.5, 0.5], [0.5, 0.5]])
    pe = dp.policy_evaluation(m, rand, 1.0, 1e-12, 100000, record_snapshots=8)
    stay = dp.policy_evaluation(m, np.array([0, 0]), 1.0, 1e-12, 100000)
    q_rand = m.q_from_v(pe["V"], 1.0)
    out.append({"id": "u4-dice-random", "unit": 4, "title": "Dice game: iterative policy evaluation, random policy",
                "result": {"V_IN": pe["V"][0], "sweeps": [round(s[0], 4) for s in pe["snapshots"]],
                           "q_IN_STAY": q_rand[0, 0], "q_IN_QUIT": q_rand[0, 1],
                           "closed_form_V_IN": m.mrp_closed_form(rand, 1.0)[0]},
                "notes_answer": {"V_IN": 10.5, "sweeps": [7, 9.33, 10.11, 10.37, 10.46, 10.49, 10.50],
                                 "q_IN_STAY": 11, "q_IN_QUIT": 10},
                "match": _close(pe["V"][0], 10.5) and _close(q_rand[0, 0], 11) and _close(q_rand[0, 1], 10)
                and _close(pe["snapshots"][1][0], 9.3333, 1e-3),
                "code": "dynamic_programming.policy_evaluation, mdp_model.mrp_closed_form"})
    vi = dp.value_iteration(m, 1.0, 1e-12, 100000)
    out.append({"id": "u4-dice-stay", "unit": 4, "title": "Dice game: always STAY / optimal policy",
                "result": {"V_IN_always_stay": stay["V"][0], "optimal_action": m.action_names[vi["actions"][0]], "V_star_IN": vi["V"][0]},
                "notes_answer": {"V_IN_always_stay": 12}, "match": _close(stay["V"][0], 12),
                "code": "dynamic_programming.value_iteration"})
    # ---------------- Unit 3/4: gridworld ----------------------------------
    gw = gridworld_4x4()
    V = dp.policy_evaluation(gw, np.full((16, 4), 0.25), 1.0, 1e-10, 100000)["V"]
    exp = [0, -14, -20, -22, -14, -18, -20, -20, -20, -20, -18, -14, -22, -20, -14, 0]
    out.append({"id": "u4-gridworld", "unit": 4, "title": "4×4 gridworld, equiprobable random policy",
                "result": {"V": [round(v, 2) for v in V]}, "notes_answer": {"V": exp},
                "match": all(_close(a, b, 1e-3) for a, b in zip(V, exp)), "code": "dynamic_programming.policy_evaluation"})
    # ---------------- Unit 3: Exercise 3.22 --------------------------------
    rows, ok = [], True
    for g, best in ((0.0, "left"), (0.9, "right"), (0.5, "both")):
        mdp_ = exercise_3_22(g)
        vl = dp.policy_evaluation(mdp_, np.array([0, 0, 0]), g, 1e-12, 100000)["V"][0]
        vr = dp.policy_evaluation(mdp_, np.array([1, 1, 1]), g, 1e-12, 100000)["V"][0]
        got = "both" if abs(vl - vr) < 1e-6 else ("left" if vl > vr else "right")
        ok &= got == best
        rows.append({"gamma": g, "V_left": vl, "V_right": vr, "optimal": got})
    out.append({"id": "u3-ex3.22", "unit": 3, "title": "Ex. 3.22: which policy is optimal for γ = 0, 0.9, 0.5?",
                "result": {"rows": rows}, "notes_answer": {"γ=0": "left", "γ=0.9": "right (5.26 vs 9.47)", "γ=0.5": "both (1.33)"},
                "match": ok, "code": "dynamic_programming.policy_evaluation"})
    # ---------------- Unit 5: MC A/B --------------------------------------
    fv = mc.first_visit_prediction(MC_AB_EPISODES, 1.0, 2)
    ev = mc.every_visit_prediction(MC_AB_EPISODES, 1.0, 2)
    out.append({"id": "u5-mc-ab", "unit": 5, "title": "First-visit vs every-visit MC on 5 episodes (A/B, γ = 1)",
                "result": {"first_visit": {"A": fv["V"][0], "B": fv["V"][1]}, "every_visit": {"A": ev["V"][0], "B": ev["V"][1]},
                           "visits_every": {"A": int(ev["N"][0]), "B": int(ev["N"][1])}},
                "notes_answer": {"first_visit": {"A": 8.25, "B": 7.8}, "every_visit": {"A": 8.0, "B": 4.0}},
                "match": _close(fv["V"][0], 8.25) and _close(fv["V"][1], 7.8) and _close(ev["V"][0], 8.0) and _close(ev["V"][1], 4.0),
                "code": "monte_carlo.first_visit_prediction / every_visit_prediction"})
    # ---------------- Unit 2: bandits -------------------------------------
    for item in bandits.all_notes_numericals():
        item.setdefault("code", "bandits.py")
        out.append(item)
    return out
