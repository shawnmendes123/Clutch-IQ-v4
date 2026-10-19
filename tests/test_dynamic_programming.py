"""Unit 4: policy evaluation, improvement, policy iteration, value iteration.
Validated on the notes' own examples (dice game, 4x4 gridworld) and on Clutch IQ."""
import numpy as np
import pytest

import rl_core as core
import mdp_model as mm
import dynamic_programming as dp


def dice_game():
    """Unit 4 notes: states IN, END; QUIT -> END with $10; STAY -> $4 and IN w.p. 2/3, END w.p. 1/3."""
    P = np.zeros((2, 2, 2)); R = np.zeros((2, 2, 2))
    STAY, QUIT, IN, END = 0, 1, 0, 1
    P[IN, QUIT, END] = 1.0; R[IN, QUIT, END] = 10
    P[IN, STAY, IN] = 2 / 3; R[IN, STAY, IN] = 4
    P[IN, STAY, END] = 1 / 3; R[IN, STAY, END] = 4
    return mm.FiniteMDP(P, R, np.array([False, True]), gamma=1.0)


def test_dice_game_random_policy_value_is_10_5():
    m = dice_game()
    pi = np.array([[0.5, 0.5], [0.5, 0.5]])
    res = dp.policy_evaluation(m, pi, 1.0, 1e-10, 10000)
    assert res["V"][0] == pytest.approx(10.5, abs=1e-6)
    # first sweeps match the notes' table: V1 = 7.0, V2 = 9.33, V3 = 10.11
    res3 = dp.policy_evaluation(m, pi, 1.0, 0.0, 3)
    assert res3["deltas"][0] == pytest.approx(7.0)
    assert m.mrp_closed_form(pi, 1.0)[0] == pytest.approx(10.5)


def test_dice_game_optimal_policy_is_stay():
    m = dice_game()
    vi = dp.value_iteration(m, 1.0, 1e-10, 100000)
    assert vi["actions"][0] == 0           # STAY
    assert vi["V"][0] == pytest.approx(12.0, abs=1e-4)   # v = 4 + (2/3) v  ->  v = 12


def gridworld():
    """Sutton 4x4 gridworld from the Unit 4 notes: reward -1 per move, terminal corners."""
    n = 16
    P = np.zeros((n, 4, n)); R = np.full((n, 4, n), -1.0)
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for s in range(n):
        r, c = divmod(s, 4)
        for a, (dr, dc) in enumerate(moves):
            nr, nc = r + dr, c + dc
            s2 = nr * 4 + nc if 0 <= nr < 4 and 0 <= nc < 4 else s
            P[s, a, s2] = 1.0
    term = np.zeros(n, bool); term[0] = term[15] = True
    return mm.FiniteMDP(P, R, term, 1.0)


def test_gridworld_random_policy_values():
    m = gridworld()
    pi = np.full((16, 4), 0.25)
    V = dp.policy_evaluation(m, pi, 1.0, 1e-8, 20000)["V"].reshape(4, 4)
    expected = np.array([[0, -14, -20, -22], [-14, -18, -20, -20], [-20, -20, -18, -14], [-22, -20, -14, 0]])
    assert np.allclose(V, expected, atol=1e-3)


@pytest.fixture(scope="module")
def clutch():
    return mm.get_model(1000, 7, 0.9)


def test_policy_evaluation_converges(clutch):
    res = dp.policy_evaluation(clutch, np.array(core.uniform_policy()), 0.9, 1e-8)
    assert res["converged"] and res["deltas"][-1] < 1e-8
    assert dp.bellman_residual(clutch, res["V"], 0.9, np.array(core.uniform_policy())) < 1e-7


def test_value_iteration_converges_and_contracts(clutch):
    vi = dp.value_iteration(clutch, 0.9, 1e-8)
    assert vi["converged"]
    assert vi["max_contraction_ratio"] <= 0.9 + 1e-9          # gamma-contraction (Banach)
    assert dp.bellman_residual(clutch, vi["V"], 0.9) < 1e-7


def test_policy_iteration_returns_stable_policy_equal_to_value_iteration(clutch):
    pi = dp.policy_iteration(clutch, 0.9, 1e-10)
    vi = dp.value_iteration(clutch, 0.9, 1e-10)
    assert pi["policy_stable"]
    assert np.max(np.abs(pi["V"] - vi["V"])) < 1e-6
    # the policies agree except possibly on exact ties
    q = vi["Q"]
    for s in range(core.N_STATES):
        a, b = pi["actions"][s], vi["actions"][s]
        assert a == b or abs(q[s, a] - q[s, b]) < 1e-6


def test_policy_improvement_is_monotone(clutch):
    keep = np.ones(clutch.nS, int)
    v0 = dp.policy_evaluation(clutch, keep, 0.9, 1e-10)["V"]
    imp = dp.policy_improvement(clutch, v0, keep, 0.9)
    v1 = dp.policy_evaluation(clutch, imp["actions"], 0.9, 1e-10)["V"]
    assert np.all(v1 >= v0 - 1e-8)        # policy improvement theorem


def test_optimal_value_dominates_baselines(clutch):
    vi = dp.value_iteration(clutch, 0.9, 1e-10)
    for pol in (core.uniform_policy(), core.constant_policy(1), core.heuristic_policy()):
        v = dp.policy_evaluation(clutch, np.array(pol), 0.9, 1e-10)["V"]
        assert np.all(vi["V"] >= v - 1e-7)


def test_backup_breakdown_reproduces_q(clutch):
    vi = dp.value_iteration(clutch, 0.9, 1e-10)
    s = core.encode(2, 2, 0)
    bd = dp.backup_breakdown(clutch, s, vi["V"], 0.9, pi=core.deterministic_policy(list(vi["actions"])), top_k=50)
    for pa in bd["per_action"]:
        assert pa["q"] == pytest.approx(vi["Q"][s, pa["action"]], abs=1e-9)
    assert bd["optimality"]["max_q"] == pytest.approx(vi["V"][s], abs=1e-7)
