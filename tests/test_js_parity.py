"""The in-browser engine (assets/rl_engine.js) must reproduce the Python engine.

Runs the JS engine under Node (skipped if Node is unavailable) and compares:
identical integer/discrete outputs (episodes, policies, arm choices) and
floating-point outputs to 1e-9 (differences can only come from summation order).
"""
import json
import os
import shutil
import subprocess

import numpy as np
import pytest

import rl_core as core
import mdp_model as mm
import dynamic_programming as dp
import monte_carlo as mc
import q_learning as ql
import rl_evaluation as ev
import bandits as bd
import notes_examples

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node not installed")


@pytest.fixture(scope="module")
def js():
    out = subprocess.run([NODE, os.path.join(ROOT, "tests", "js_parity_runner.js")], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def close(a, b, tol=1e-9):
    return np.allclose(np.asarray(a, float), np.asarray(b, float), atol=tol, rtol=0)


def test_prng_and_episode(js):
    r = core.Mulberry32(12345)
    assert js["prng"] == [r.random(), r.random(), r.random()]
    ep = core.generate_episode(core.heuristic_policy(), 123, start_dist="session_start")
    assert [[t.state, t.action, t.next_state] for t in ep.transitions] == [[a, b, d] for a, b, c, d in js["episode"]]
    assert close([t.reward for t in ep.transitions], [c for _, _, c, _ in js["episode"]])


def test_model_and_dp(js):
    m = mm.get_model(300, 3, 0.9)
    assert close(m.P[17, 2], js["model_P_row"], 0) and close(m.R[17, 2], js["model_R_row"])
    vi = dp.value_iteration(m, 0.9, 1e-10)
    assert close(vi["V"], js["vi_V"]) and list(vi["actions"]) == js["vi_actions"] and vi["iterations"] == js["vi_iters"]
    assert list(dp.policy_iteration(m, 0.9, 1e-10)["actions"]) == js["pi_actions"]
    assert close(dp.policy_evaluation(m, np.array(core.uniform_policy()), 0.9, 1e-10)["V"], js["pe_V"])
    assert close(m.mrp_closed_form(np.array(core.uniform_policy())), js["mrp"], 1e-8)


def test_monte_carlo(js):
    r = mc.mc_prediction_run(core.heuristic_policy(), 400, 0.9, 11, "uniform", None, [400])
    assert close(r["V_first_visit"], js["mc_fv"]) and close(r["V_every_visit"], js["mc_ev"])
    c = mc.mc_control("on_policy", 800, 0.9, 0.1, 7)
    assert c["actions"] == js["mcc_actions"] and close(c["Q"], js["mcc_Q"])
    assert mc.mc_control("es", 500, 0.9, 0.1, 7)["actions"] == js["mces_actions"]
    assert close(mc.mc_control("off_policy", 500, 0.9, 0.1, 7, behaviour_epsilon=0.3)["Q"], js["mcoff_Q"])
    heur = core.policy_actions(core.heuristic_policy())
    o = mc.off_policy_prediction(core.epsilon_soft(heur, 0.1), core.epsilon_soft(heur, 0.3), 300, 0.9, 4)
    assert close([o["J_ordinary_is"], o["J_weighted_is"], o["J_per_decision_is"], o["J_discounting_aware_ois"], o["J_discounting_aware_wis"]], js["ope"], 1e-8)


def test_q_learning_and_evaluation(js):
    q = ql.train(episodes=1500, seed=11)
    assert close(q["Q"], js["q_Q"], 1e-8)
    e = ev.evaluate_policy(core.heuristic_policy(), 200, 2026, 0.9, "session_start")
    assert close([e["discounted_return"]["mean"], e["rage_quit_rate"]["mean"], e["session_length"]["mean"]], js["eval"])


def test_bandits_and_arms(js):
    assert bd.run_bandit("ucb", [0.2, 0.5, 0.8, 0.4], 300, 9, c=1.0)["arms"] == js["bandit_ucb_arms"]
    be = bd.run_bandit("epsilon_greedy", [0.2, 0.5, 0.8, 0.4], 300, 9, epsilon=0.1, reward_model="gaussian", sigma=0.2, nonstationary="drift")
    assert close(be["Q"], js["bandit_eps_Q"], 1e-7)
    assert close(bd.run_bandit("greedy", [0.2, 0.5, 0.8, 0.4], 200, 4, nonstationary="patch")["final_means"], js["bandit_patch_means"])
    import rl_api
    arms, _ = rl_api.build_arms(rl_api.ArmsSpec())
    assert close([a["q_star"] for a in arms], js["arms"])


def test_numericals_and_endpoint_emulation(js):
    py = [[n["id"], n["match"]] for n in notes_examples.all_numericals()]
    assert py == js["numericals"]
    import rl_evaluation
    pi = core.deterministic_policy(list(dp.value_iteration(mm.get_model(1000, 7, 0.9), 0.9, 1e-10)["actions"]))
    t = rl_evaluation.trajectory(pi, 3, 0.9)
    assert [[s["state"], s["action"]] for s in t["steps"]] == js["traj"]
