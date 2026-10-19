"""Unit 1 + Unit 3: shared definitions, simulator, finite MDP, Markov chain / MRP."""
import numpy as np
import pytest

import rl_core as core
import mdp_model as mm
import dynamic_programming as dp


# ----------------------------------------------------------- PRNG parity
def test_mulberry32_reference_sequence():
    # Reference values produced by the canonical JavaScript mulberry32(12345)
    r = core.Mulberry32(12345)
    ref = [0.9797282677609473, 0.3067522644996643, 0.484205421525985, 0.817934412509203, 0.5094283693470061]
    assert [r.random() for _ in range(5)] == ref


def test_derive_seed_reference():
    assert core.derive_seed(7, 0) == 2619039839
    assert core.derive_seed(7, 1) == 3687617024
    assert core.derive_seed(2026, 999) == 2313696697


# ------------------------------------------------------ state discretisation
@pytest.mark.parametrize("s", range(core.N_STATES))
def test_encode_decode_roundtrip(s):
    f, t, k = core.decode(s)
    assert core.encode(f, t, k) == s
    assert core.key_to_state(core.state_key(s)) == s


def test_state_space_size_and_buckets():
    assert core.N_STATES == 48 and core.TERMINAL == 48
    assert core.bucket_fatigue(0.0) == 0 and core.bucket_fatigue(0.249) == 0 and core.bucket_fatigue(0.25) == 1
    assert core.bucket_fatigue(1.0) == 3 and core.bucket_tilt(0.99) == 3
    assert core.bucket_streak(-2) == 0 and core.bucket_streak(-1) == 1 and core.bucket_streak(1) == 1 and core.bucket_streak(2) == 2
    # v3 key format preserved: "fatigue-tilt-streak"
    assert core.state_key(0.8, 0.7, -3) == "3-2-0"


def test_reward_bounds():
    for C in np.linspace(0, 1, 11):
        assert -3.0 - 1e-12 <= core.reward_function(0, C, True) <= -1.8 + 1e-12
        assert 1.0 - 1e-12 <= core.reward_function(1, C, False) <= 2.2 + 1e-12


# ------------------------------------------------------------- simulator law
def test_exact_transition_probabilities_sum_to_one():
    for s in range(core.N_STATES):
        for a in core.ACTIONS:
            probs, rewards = core.exact_transition(s, a)
            assert abs(sum(probs.values()) - 1.0) < 1e-12
            assert all(p >= 0 for p in probs.values())


def test_simulator_matches_its_exact_law():
    """Empirical frequencies of env_step converge to exact_transition."""
    rng = core.Mulberry32(99)
    s, a, n = core.encode(1, 2, 0), 1, 40000
    counts = {}
    for _ in range(n):
        s2, *_ = core.env_step(s, a, rng)
        counts[s2] = counts.get(s2, 0) + 1
    probs, _ = core.exact_transition(s, a)
    for s2, p in probs.items():
        se = np.sqrt(p * (1 - p) / n)
        assert abs(counts.get(s2, 0) / n - p) < 5 * se + 1e-9


def test_returns():
    assert core.discounted_return([1, 1, 1], 0.5) == pytest.approx(1.75)
    assert core.returns_from_rewards([1, 2, 3], 0.9) == pytest.approx([1 + 0.9 * 2 + 0.81 * 3, 2 + 0.9 * 3, 3])
    assert core.discounted_return([5, 7], 0.0) == 5


def test_generate_episode_is_deterministic_and_terminates():
    pol = core.uniform_policy()
    e1 = core.generate_episode(pol, 123)
    e2 = core.generate_episode(pol, 123)
    assert [(t.state, t.action, t.reward) for t in e1.transitions] == [(t.state, t.action, t.reward) for t in e2.transitions]
    assert e1.transitions[-1].done and e1.transitions[-1].next_state == core.TERMINAL


def test_common_random_numbers():
    """Two deterministic policies with the same seed see the same first win/loss draw."""
    e1 = core.generate_episode(core.constant_policy(1), 77, record_info=True)
    e2 = core.generate_episode(core.constant_policy(1), 77, record_info=True)
    assert e1.transitions[0].state == e2.transitions[0].state


# ------------------------------------------------------------ MDP / model
def test_estimated_model_rows_sum_to_one():
    m = mm.get_model(300, 3, 0.9)
    assert m.check()["rows_sum_to_one"]
    assert m.P.shape == (49, 3, 49)


def test_estimated_model_close_to_simulator_law():
    m = mm.get_model(1000, 7, 0.9)
    q = mm.model_quality(m)
    # 1000 samples per pair -> binomial s.e. <= 0.0158; allow a 5 s.e. envelope over ~1200 cells
    assert q["max_abs_transition_error"] < 5 * q["worst_case_std_error"]


def test_mrp_closed_form_equals_iterative_policy_evaluation():
    m = mm.get_model(500, 11, 0.9)
    pi = np.array(core.uniform_policy())
    v_closed = m.mrp_closed_form(pi)
    v_iter = dp.policy_evaluation(m, pi, 0.9, 1e-10, 5000)["V"]
    assert np.max(np.abs(v_closed - v_iter)) < 1e-6


def test_markov_chain_sequence_probability_rain_dry_example():
    """Unit 3 notes: P(Dry, Dry, Rain, Rain) = P(Dry) P(Dry|Dry) P(Rain|Dry) P(Rain|Rain) = 0.6*0.8*0.2*0.3."""
    P = np.zeros((2, 1, 2))
    P[0, 0] = [0.3, 0.7]   # Rain -> Rain 0.3, Rain -> Dry 0.7
    P[1, 0] = [0.2, 0.8]   # Dry  -> Rain 0.2, Dry  -> Dry 0.8
    chain = mm.FiniteMDP(P, np.zeros_like(P), np.array([False, False]))
    p = chain.sequence_probability([1, 1, 0, 0], np.array([[1.0], [1.0]]), start=np.array([0.4, 0.6]))
    assert p == pytest.approx(0.6 * 0.8 * 0.2 * 0.3)


def test_policy_markov_chain_is_stochastic():
    m = mm.get_model(300, 3, 0.9)
    Ppi = m.policy_markov_chain(np.array(core.heuristic_policy()))
    assert np.allclose(Ppi.sum(axis=1), 1.0)
