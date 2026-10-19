"""Unit 5: Monte Carlo prediction, control and importance sampling."""
import numpy as np
import pytest

import rl_core as core
import mdp_model as mm
import dynamic_programming as dp
import monte_carlo as mc

A, B = 0, 1
# Unit 5 notes example (IIT Kharagpur midsem 2022): state, reward pairs, gamma = 1
NOTES_EPISODES = [
    ([A, A, B, B], [2, 6, 1, 0]),
    ([A, B, A, B, B], [3, 2, 4, 2, 0]),
    ([B, B, A, B], [3, 6, 1, 0]),
    ([A, B, A, B, B, B], [0, 2, 4, 4, 2, 0]),
    ([B, B], [8, 0]),
]


def test_first_visit_matches_notes():
    res = mc.first_visit_prediction(NOTES_EPISODES, 1.0, 2)
    assert res["V"][A] == pytest.approx(8.25)
    assert res["V"][B] == pytest.approx(7.8)


def test_every_visit_matches_notes():
    res = mc.every_visit_prediction(NOTES_EPISODES, 1.0, 2)
    assert res["V"][A] == pytest.approx(8.0)
    assert res["V"][B] == pytest.approx(4.0)
    assert res["N"][A] == 7 and res["N"][B] == 14


def test_first_and_every_visit_identical_without_revisits():
    eps = [([0, 1, 2], [1, 1, 1]), ([1, 2], [0, 5])]
    fv = mc.first_visit_prediction(eps, 0.9, 3)
    ev = mc.every_visit_prediction(eps, 0.9, 3)
    assert np.allclose(fv["V"], ev["V"])


def test_return_calculation_backwards():
    G = mc.episode_returns([1, 0, 2], 0.5)
    assert G == pytest.approx([1 + 0 + 0.25 * 2, 0 + 0.5 * 2, 2])


def test_mc_action_values_first_visit():
    eps = [{"states": [0, 0], "actions": [1, 1], "rewards": [1, 3]}]
    q = mc.mc_action_values(eps, 1.0, 1, 2)
    assert q["Q"][0, 1] == pytest.approx(4.0) and q["N"][0, 1] == 1


def test_importance_sampling_ratio_hand_computed():
    """pi deterministic KEEP; b uniform: rho over k matching steps = 3^k, 0 if any mismatch."""
    target = core.constant_policy(1)
    behaviour = core.uniform_policy()
    res = mc.off_policy_prediction(target, behaviour, 200, 0.9, 5)
    for smp in res["sample_episodes"]:
        acts = [st["action"] for st in smp["steps"]]
        if smp["length"] <= 12:
            expected = 3.0 ** smp["length"] if all(a == "KEEP" for a in acts) else 0.0
            assert smp["rho"] == pytest.approx(expected)


def test_weighted_is_equals_on_policy_when_target_equals_behaviour():
    pol = core.heuristic_policy()
    soft = core.epsilon_soft(core.policy_actions(pol), 0.2)
    res = mc.off_policy_prediction(soft, soft, 300, 0.9, 9)
    assert res["rho_stats"]["max"] == pytest.approx(1.0)
    assert res["J_weighted_is"] == pytest.approx(res["J_behaviour_on_policy"])
    assert res["J_ordinary_is"] == pytest.approx(res["J_behaviour_on_policy"])


def test_coverage_violation_detected():
    res = mc.off_policy_prediction(core.constant_policy(2), core.constant_policy(1), 10, 0.9, 1)
    assert "error" in res and not res["coverage"]["satisfied"]


@pytest.fixture(scope="module")
def model():
    return mm.get_model(1000, 7, 0.9)


def test_mc_prediction_error_decreases_with_episodes(model):
    pol = core.heuristic_policy()
    ref = dp.policy_evaluation(model, np.array(pol), 0.9, 1e-10)["V"]
    res = mc.mc_prediction_run(pol, 4000, 0.9, 3, "uniform", ref, checkpoints=[100, 4000])
    c = {r["episodes"]: r for r in res["curve"]}
    assert c[4000]["mae_first_visit"] < c[100]["mae_first_visit"]
    assert c[4000]["mae_first_visit"] < 0.35
    assert c[4000]["coverage"] == 1.0


def test_off_policy_estimate_close_to_dp_for_small_policy_change(model):
    heur = core.policy_actions(core.heuristic_policy())
    target = core.epsilon_soft(heur, 0.1)
    behaviour = core.epsilon_soft(heur, 0.3)
    d0 = np.array(core.start_distribution("session_start") + [0.0])
    truth = float(d0 @ dp.policy_evaluation(model, np.array(target), 0.9, 1e-10)["V"])
    res = mc.off_policy_prediction(target, behaviour, 10000, 0.9, 4, "session_start", truth)
    assert abs(res["J_weighted_is"] - truth) < 0.3
    assert abs(res["J_ordinary_is"] - truth) < 0.5
    assert res["reliability"] == "OK"


def test_off_policy_flags_unreliable_large_policy_change(model):
    vi = dp.value_iteration(model, 0.9, 1e-10)
    target = core.deterministic_policy(list(vi["actions"]))
    behaviour = core.epsilon_soft([1] * 49, 0.3)
    res = mc.off_policy_prediction(target, behaviour, 2000, 0.9, 4)
    assert res["reliability"].startswith("LOW")


def test_per_decision_is_equals_ordinary_when_on_policy():
    pol = core.epsilon_soft(core.policy_actions(core.heuristic_policy()), 0.2)
    res = mc.off_policy_prediction(pol, pol, 200, 0.9, 2)
    assert res["J_per_decision_is"] == pytest.approx(res["J_ordinary_is"])
    assert res["J_discounting_aware_wis"] == pytest.approx(res["J_behaviour_on_policy"], rel=0.2)


def test_mc_control_variants_learn_good_policies(model):
    vi = dp.value_iteration(model, 0.9, 1e-10)
    d0 = np.array(core.start_distribution("session_start") + [0.0])
    j_opt = float(d0 @ vi["V"])
    j_keep = float(d0 @ dp.policy_evaluation(model, np.ones(49, int), 0.9, 1e-10)["V"])
    for variant in ("es", "on_policy"):
        r = mc.mc_control(variant, 6000, 0.9, 0.1, 7, reference_actions=vi["actions"])
        acts = np.array(r["actions"] + [1])
        j = float(d0 @ dp.policy_evaluation(model, acts, 0.9, 1e-10)["V"])
        assert j > j_keep            # better than the rank-only baseline
        assert j > 0.93 * j_opt      # close to the DP optimum
