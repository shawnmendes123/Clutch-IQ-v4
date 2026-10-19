"""End-to-end API tests: every v3 endpoint still works, plus the new RL endpoints."""
import io
import os

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend_app import app

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
client = TestClient(app)


def _history_csv(with_date=True, scoreline=False):
    rng = np.random.default_rng(0)
    rows = []
    for p in ["alpha", "bravo", "charlie", "delta", "echo", "fox"]:
        for d in range(5):
            for m in range(8):
                row = {"player": p, "kills": int(rng.integers(5, 30)), "deaths": int(rng.integers(5, 25)),
                       "mode": "Team Deathmatch", "score": int(rng.integers(500, 3000))}
                if with_date:
                    row["date"] = f"2024-01-{d + 1:02d} {10 + m}:00"
                row["result"] = f"{rng.integers(0, 10)}-{rng.integers(0, 10)}" if scoreline else rng.choice(["W", "L"])
                rows.append(row)
    buf = io.StringIO()
    pd.DataFrame(rows).to_csv(buf, index=False)
    return buf.getvalue().encode()


# ---------------------------------------------------------------- v3 surface
def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


@pytest.mark.parametrize("with_date,scoreline", [(True, False), (False, False), (True, True)])
def test_upload_engineer_players_matchmake_forecast_report(with_date, scoreline):
    r = client.post("/api/upload", files={"file": ("hist.csv", _history_csv(with_date, scoreline), "text/csv")})
    assert r.status_code == 200
    ds = r.json()["dataset_id"]
    r = client.post(f"/api/engineer/{ds}")
    assert r.status_code == 200, r.text            # v3 crashed here when a date column existed
    assert r.json()["player_count"] == 6
    r = client.get(f"/api/players/{ds}")
    players = r.json()["players"]
    assert len(players) == 6
    r = client.post(f"/api/matchmake/{ds}", json={"slate_size": 6})
    assert r.status_code == 200 and "team_a" in r.json()
    pid = players[0]["id"]
    assert client.get(f"/api/forecast/{ds}/{pid}").status_code == 200
    rep = client.get(f"/api/report/{ds}/{pid}")
    assert rep.status_code == 200 and "Player Report" in rep.json()["markdown"]
    real = client.get(f"/api/rl/mc/real/{ds}")
    assert real.status_code == 200
    body = real.json()
    assert body["control_available"] is False and body["n_episodes"] > 0


def test_synthetic_players_optimize_reward():
    assert client.get("/api/players/synthetic?n=20&seed=3").status_code == 200
    r = client.post("/api/optimize", json={"fatigue": 0.8, "tilt": 0.7, "streak": -3})
    assert r.status_code == 200 and r.json()["state_bucket"] == "3-2-0"
    r = client.post("/api/reward", json={"session_length_delta": 1, "close_match_baseline": 0.5, "rage_quit": False})
    assert r.json()["reward"] == pytest.approx(1.6)


def test_tournament_build_and_bandit_on_it():
    with open(os.path.join(ROOT, "sample_maps.csv"), "rb") as m, open(os.path.join(ROOT, "sample_players.csv"), "rb") as p:
        r = client.post("/api/tournament/build", files={"maps_file": ("m.csv", m, "text/csv"), "players_file": ("p.csv", p, "text/csv")})
    assert r.status_code == 200
    t = r.json()
    assert t["match_count"] >= 1 and client.get(f"/api/tournament/{t['tournament_id']}").status_code == 200
    r = client.post("/api/bandit/run", json={"rounds": 200, "runs": 5, "algorithms": ["epsilon_greedy", "ucb"],
                                             "arms": {"source": "tournament", "tournament_id": t["tournament_id"], "k": 8}})
    assert r.status_code == 200
    assert len(r.json()["arms"]) == 8


# ---------------------------------------------------------------- RL surface
def test_rl_spec_and_mdp():
    sp = client.get("/api/rl/spec").json()
    assert sp["n_states"] == 48 and len(sp["actions"]) == 3
    m = client.get("/api/rl/mdp?samples=200").json()
    P = np.array(m["P"])
    assert P.shape == (49, 3, 49)
    assert np.allclose(P.sum(axis=2), 1, atol=1e-3)


def test_state_explorer_bellman_numbers_consistent():
    r = client.get("/api/rl/mdp/state/2-2-0?policy=heuristic").json()
    be = r["bellman_expectation"]
    assert be["expectation"]["value"] == pytest.approx(r["V_pi"], abs=1e-4)
    assert r["bellman_optimality"]["optimality"]["max_q"] == pytest.approx(r["V_star"], abs=1e-4)


def test_dp_endpoints():
    pe = client.post("/api/rl/dp/policy-evaluation", json={"policy": "random"}).json()
    assert pe["converged"] and pe["max_abs_diff_vs_closed_form"] < 1e-4
    pi = client.post("/api/rl/dp/policy-iteration", json={}).json()
    assert pi["policy_stable"]
    vi = client.post("/api/rl/dp/value-iteration", json={}).json()
    assert vi["converged"] and vi["max_contraction_ratio"] <= 0.9 + 1e-9
    assert len(vi["actions"]) == 48


def test_mc_endpoints():
    p = client.post("/api/rl/mc/predict", json={"episodes": 500}).json()
    assert p["curve"][-1]["episodes"] == 500
    c = client.post("/api/rl/mc/control", json={"episodes": 1000, "variant": "es"}).json()
    assert len(c["actions"]) == 48
    o = client.post("/api/rl/mc/off-policy", json={"episodes": 300}).json()
    assert "J_weighted_is" in o and o["coverage"]["satisfied"]


def test_off_policy_refuses_without_coverage():
    r = client.post("/api/rl/mc/off-policy", json={"target": "raise", "target_epsilon": 0, "behaviour": "keep",
                                                   "behaviour_epsilon": 0, "episodes": 50})
    assert r.status_code == 400


def test_logged_episode_upload_and_honesty_rules():
    csv = client.get("/api/rl/episodes/sample").text
    r = client.post("/api/rl/episodes/upload", files={"file": ("logs.csv", csv.encode(), "text/csv")})
    assert r.status_code == 200 and r.json()["has_behaviour_probs"]
    assert r.json()["provenance"].startswith("SIMULATED")
    eid = r.json()["episodes_id"]
    assert client.post("/api/rl/mc/predict", json={"episodes_id": eid}).status_code == 200
    assert client.post("/api/rl/mc/off-policy", json={"episodes_id": eid}).status_code == 200
    # strip behaviour probabilities -> off-policy must refuse instead of fabricating them
    df = pd.read_csv(io.StringIO(csv)).drop(columns=["behaviour_prob"])
    r = client.post("/api/rl/episodes/upload", files={"file": ("logs2.csv", df.to_csv(index=False).encode(), "text/csv")})
    eid2 = r.json()["episodes_id"]
    assert client.post("/api/rl/mc/off-policy", json={"episodes_id": eid2}).status_code == 400


def test_q_learning_endpoints_do_not_overwrite_production():
    before = client.get("/api/rl/q/table").json()["Q"]
    r = client.post("/api/rl/q/train", json={"episodes": 500, "seed": 99}).json()
    assert r["saved_as_default"] is False
    after = client.get("/api/rl/q/table").json()["Q"]
    assert before == after
    td = client.post("/api/rl/q/td-step", json={"state": 3, "action": 1, "reward": 1.5, "next_state": 48}).json()
    assert td["td_target"] == pytest.approx(1.5)


def test_validation_rejects_bad_parameters():
    assert client.post("/api/rl/dp/value-iteration", json={"gamma": 1.5}).status_code == 422
    assert client.post("/api/bandit/run", json={"epsilon": -0.1}).status_code == 422
    assert client.get("/api/rl/mdp/state/99").status_code == 400


def test_trajectory_and_compare_and_academic():
    t = client.post("/api/rl/trajectory", json={"seed": 3}).json()
    steps = t["steps"]
    assert steps[-1]["done"] and t["G0"] == pytest.approx(steps[0]["G_t"])
    assert sum(s["discounted_contribution"] for s in steps) == pytest.approx(t["G0"], abs=1e-4)
    cmp_ = client.post("/api/rl/compare", json={"n_episodes": 100, "mc_episodes": 1000, "q_episodes": 1000}).json()
    keys = [p["key"] for p in cmp_["policies"]]
    assert "dp_value_iteration" in keys and "q_learning" in keys
    nums = client.get("/api/academic/numericals").json()
    assert nums["matched"] == nums["checked"]
    cov = client.get("/api/academic/coverage").json()
    assert cov["status_counts"]["IMPLEMENTED"] > 30
