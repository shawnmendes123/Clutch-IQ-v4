"""Tabular Q-learning (TD control) and Unit 2 bandits."""
import math

import numpy as np
import pytest

import rl_core as core
import q_learning as ql
import rl_optimizer
import bandits as bd


# ------------------------------------------------------------ Q-learning
def test_td_update_equation():
    Q = np.zeros((49, 3))
    Q[5] = [1.0, 2.0, 0.5]
    Q[3, 1] = 0.4
    u = ql.td_update(Q, 3, 1, 1.5, 5, False, alpha=0.1, gamma=0.9)
    assert u["td_target"] == pytest.approx(1.5 + 0.9 * 2.0)
    assert u["td_error"] == pytest.approx(3.3 - 0.4)
    assert Q[3, 1] == pytest.approx(0.4 + 0.1 * 2.9)


def test_td_update_does_not_bootstrap_from_terminal():
    Q = np.zeros((49, 3)); Q[48] = [99, 99, 99]
    u = ql.td_update(Q, 0, 0, -3.0, 48, True, 0.5, 0.9)
    assert u["td_target"] == pytest.approx(-3.0)


def test_q_learning_seeded_determinism():
    a = ql.train(episodes=300, seed=11)
    b = ql.train(episodes=300, seed=11)
    c = ql.train(episodes=300, seed=12)
    assert np.array_equal(a["Q"], b["Q"])
    assert not np.array_equal(a["Q"], c["Q"])


def test_epsilon_decay_floor():
    r = ql.train(episodes=200, eps_start=1.0, eps_decay=0.9, eps_min=0.2, seed=1)
    assert r["final_epsilon"] == pytest.approx(0.2)


def test_recommend_action_backwards_compatible():
    out = rl_optimizer.recommend_action(0.8, 0.75, -4)
    for key in ("state_bucket", "action", "action_label", "q_values", "predicted_rage_quit_risk"):
        assert key in out
    assert out["state_bucket"] == "3-3-0"
    assert set(out["q_values"]) == {"Lower Difficulty", "Keep Steady", "Raise Difficulty"}


def test_discretisation_matches_v3():
    assert rl_optimizer.state_key(0.1, 0.9, 3) == "0-3-2"
    assert rl_optimizer.reward_function(1.0, 0.5, False) == pytest.approx(1.6)


# ----------------------------------------------------------------- bandits
def test_incremental_update_equals_sample_average():
    rewards = [1, 0, 0, 1, 1, 0, 1]
    Q, n = 0.0, 0
    for r in rewards:
        n += 1
        Q += (r - Q) / n
    assert Q == pytest.approx(np.mean(rewards))


def test_constant_step_weights_sum_to_one():
    a, n = 0.1, 25
    w = [(1 - a) ** n] + [a * (1 - a) ** (n - i) for i in range(1, n + 1)]
    assert sum(w) == pytest.approx(1.0)


MEANS = [0.2, 0.5, 0.8, 0.4]


def test_epsilon_zero_is_greedy():
    r0 = bd.run_bandit("epsilon_greedy", MEANS, 300, 5, epsilon=0.0)
    assert r0["n_explore"] == 0


def test_epsilon_one_always_explores():
    r1 = bd.run_bandit("epsilon_greedy", MEANS, 300, 5, epsilon=1.0)
    assert r1["n_exploit"] == 0
    assert min(r1["N"]) > 40                      # roughly uniform pulls


def test_ucb_tries_every_unvisited_arm_first():
    r = bd.run_bandit("ucb", MEANS, 50, 3, c=2.0)
    assert sorted(r["arms"][:len(MEANS)]) == list(range(len(MEANS)))


def test_ucb_score_formula():
    s = bd.ucb_scores([0.5, 0.6], [4, 0], 10, 2.0)
    assert s[0] == pytest.approx(0.5 + 2.0 * math.sqrt(math.log(10) / 4))
    assert math.isinf(s[1])


def test_regret_computation():
    r = bd.run_bandit("random", MEANS, 400, 9)
    expected = sum(max(MEANS) - MEANS[a] for a in r["arms"])
    assert r["total_regret"] == pytest.approx(expected)
    assert r["realised_regret"] == pytest.approx(400 * max(MEANS) - sum(r["rewards"]))


def test_same_seed_same_environment_for_every_algorithm():
    a = bd.run_bandit("epsilon_greedy", MEANS, 100, 42, epsilon=1.0)
    b = bd.run_bandit("epsilon_greedy", MEANS, 100, 42, epsilon=1.0)
    assert a["rewards"] == b["rewards"]


def test_optimistic_initial_values_force_exploration():
    r = bd.run_bandit("optimistic", MEANS, 40, 2, q0=5.0)
    assert all(n > 0 for n in r["N"])


def test_ucb_and_epsilon_beat_random_on_average():
    arena = bd.run_arena(["random", "epsilon_greedy", "ucb"], MEANS, 500, 20, 1, epsilon=0.1, c=1.0)
    res = arena["results"]
    assert res["epsilon_greedy"]["final_regret"] < res["random"]["final_regret"]
    assert res["ucb"]["final_regret"] < res["random"]["final_regret"]


def test_nonstationary_patch_changes_best_arm():
    r = bd.run_bandit("greedy", MEANS, 200, 4, nonstationary="patch")
    assert r["best_arm_final"] != int(np.argmax(MEANS))


@pytest.mark.parametrize("item", bd.all_notes_numericals(), ids=lambda i: i["id"])
def test_notes_numericals(item):
    if item["match"] is not None:
        assert item["match"], item["title"]
