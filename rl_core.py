"""
rl_core.py — shared Reinforcement Learning definitions for Clutch IQ.

Every sequential RL algorithm in the project (Dynamic Programming, Monte
Carlo, tabular Q-learning, policy evaluation/comparison) imports its state
space, action space, reward function and environment from THIS module, so
all of them are provably solving the same problem.

UNIT 1 mapping
--------------
Agent        : the Clutch IQ session controller (chooses lobby difficulty).
Environment  : one player's competitive-gaming session (SessionEnv below).
State  S_t   : (fatigue bucket, tilt bucket, streak bucket)  -> 4 x 4 x 3 = 48
Action A_t   : 0 LOWER difficulty, 1 KEEP, 2 RAISE difficulty
Reward R_t+1 : R = w1*(match completed) + w2*(predicted closeness) - w3*(rage quit)
Episode      : one play session; ends at a rage-quit or at natural exhaustion
Return  G_t  : sum_k gamma^k R_{t+k+1}
Policy pi    : array pi[s, a] = probability of action a in state s

Environment classification (Unit 1 "task environment" vocabulary):
fully observable (w.r.t. the bucketed state), stochastic, discrete,
episodic, single-agent controller.

Data honesty
------------
SessionEnv is a SIMULATOR. Its dynamics are hand-specified, documented
behavioural assumptions (see ENV_PARAMS) — they were NOT fitted to real
players. Anything produced by stepping this environment is SIMULATED.

Reproducibility
---------------
All randomness goes through Mulberry32, a tiny 32-bit PRNG implemented
identically in assets/rl_engine.js, so the Python backend and the in-browser
fallback engine generate the *same* random streams for the same seed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict

# --------------------------------------------------------------------------
# Deterministic PRNG shared with the JavaScript engine
# --------------------------------------------------------------------------
_M32 = 0xFFFFFFFF


def _imul(a: int, b: int) -> int:
    """32-bit integer multiply (low 32 bits, unsigned) == JS Math.imul bits."""
    return ((a & _M32) * (b & _M32)) & _M32


def derive_seed(seed: int, k: int) -> int:
    """Derive an independent sub-stream seed (identical in rl_engine.js)."""
    return (_imul((seed & _M32) ^ 0x9E3779B9, 0x85EBCA6B) ^ _imul((k + 1) & _M32, 0xC2B2AE35)) & _M32


class Mulberry32:
    """Mulberry32 PRNG. random() returns a float in [0, 1)."""

    __slots__ = ("a",)

    def __init__(self, seed: int):
        self.a = int(seed) & _M32

    def random(self) -> float:
        self.a = (self.a + 0x6D2B79F5) & _M32
        a = self.a
        t = _imul(a ^ (a >> 15), 1 | a)
        t = ((t + _imul(t ^ (t >> 7), 61 | t)) & _M32) ^ t
        return ((t ^ (t >> 14)) & _M32) / 4294967296.0

    def randint(self, lo: int, hi: int) -> int:
        """Uniform integer in [lo, hi] inclusive."""
        return lo + int(math.floor(self.random() * (hi - lo + 1)))

    def choice_index(self, n: int) -> int:
        return int(math.floor(self.random() * n))

    def normal(self, mu: float = 0.0, sigma: float = 1.0) -> float:
        """Box–Muller transform (one sample per call, no caching)."""
        u1 = self.random()
        u2 = self.random()
        if u1 < 1e-12:
            u1 = 1e-12
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        return mu + sigma * z

    def sample_discrete(self, probs) -> int:
        """Inverse-CDF sampling (the 'CDF table' method from the Unit 2 notes)."""
        u = self.random()
        acc = 0.0
        for i, p in enumerate(probs):
            acc += p
            if u < acc:
                return i
        return len(probs) - 1


# --------------------------------------------------------------------------
# State space (finite, 48 non-terminal states + 1 absorbing terminal)
# --------------------------------------------------------------------------
FATIGUE_LEVELS = ["FRESH", "WARM", "TIRED", "EXHAUSTED"]   # bucket of fatigue in [0,1]
TILT_LEVELS = ["CALM", "UNEASY", "TILTED", "VOLATILE"]     # bucket of tilt in [0,1]
STREAK_LEVELS = ["LOSING", "NEUTRAL", "WINNING"]           # streak <= -2 / -1..1 / >= 2

N_F, N_T, N_K = 4, 4, 3
N_STATES = N_F * N_T * N_K          # 48 decision states
TERMINAL = N_STATES                  # index 48 = absorbing END state
N_STATES_WITH_TERMINAL = N_STATES + 1

ACTIONS = [0, 1, 2]
N_ACTIONS = 3
ACTION_NAMES = ["LOWER", "KEEP", "RAISE"]
ACTION_LABELS = {0: "Lower Difficulty", 1: "Keep Steady", 2: "Raise Difficulty"}
ACTION_DELTA = {0: -1, 1: 0, 2: +1}  # lobby difficulty offset relative to the player's MMR band


def bucket_fatigue(f: float) -> int:
    return max(0, min(3, int(float(f) * 4)))


def bucket_tilt(t: float) -> int:
    return max(0, min(3, int(float(t) * 4)))


def bucket_streak(streak: int) -> int:
    if streak <= -2:
        return 0
    if streak >= 2:
        return 2
    return 1


def encode(f: int, t: int, k: int) -> int:
    """(fatigue, tilt, streak) buckets -> state index 0..47."""
    return f * (N_T * N_K) + t * N_K + k


def decode(s: int) -> tuple[int, int, int]:
    if s == TERMINAL:
        return (-1, -1, -1)
    return s // (N_T * N_K), (s // N_K) % N_T, s % N_K


def state_from_features(fatigue: float, tilt: float, streak: int) -> int:
    return encode(bucket_fatigue(fatigue), bucket_tilt(tilt), bucket_streak(streak))


def state_key(s_or_f, tilt=None, streak=None) -> str:
    """Key format 'f-t-k' used by q_table.json (backwards compatible).
    Accepts either a state index or the raw continuous (fatigue, tilt, streak)."""
    if tilt is None:
        f, t, k = decode(int(s_or_f))
    else:
        f, t, k = bucket_fatigue(s_or_f), bucket_tilt(tilt), bucket_streak(streak)
    return f"{f}-{t}-{k}"


def key_to_state(key: str) -> int:
    f, t, k = (int(x) for x in key.split("-"))
    return encode(f, t, k)


def state_label(s: int) -> str:
    if s == TERMINAL:
        return "END"
    f, t, k = decode(s)
    return f"{FATIGUE_LEVELS[f]} · {TILT_LEVELS[t]} · {STREAK_LEVELS[k]}"


def state_short(s: int) -> str:
    if s == TERMINAL:
        return "END"
    f, t, k = decode(s)
    return f"F{f}T{t}{'LNW'[k]}"


def state_info(s: int) -> dict:
    if s == TERMINAL:
        return {"index": s, "key": "END", "label": "END (session over)", "terminal": True}
    f, t, k = decode(s)
    return {
        "index": s, "key": f"{f}-{t}-{k}", "short": state_short(s), "label": state_label(s),
        "fatigue_bucket": f, "tilt_bucket": t, "streak_bucket": k,
        "fatigue": FATIGUE_LEVELS[f], "tilt": TILT_LEVELS[t], "streak": STREAK_LEVELS[k],
        "fatigue_range": [f / 4, (f + 1) / 4], "tilt_range": [t / 4, (t + 1) / 4],
        "streak_range": ["<= -2", "-1..+1", ">= +2"][k], "terminal": False,
    }


# --------------------------------------------------------------------------
# Reward  (Unit 1: immediate reward r_t)
# --------------------------------------------------------------------------
# Weights preserved from the original Clutch IQ brief: rage quits cost w3 = 3,
# i.e. 2.5x the best possible closeness bonus (w2 * 1.0 = 1.2).
W1, W2, W3 = 1.0, 1.2, 3.0


def reward_function(session_length_delta: float, close_match_baseline: float, rage_quit: bool) -> float:
    """R = w1*(session-length term) + w2*(close-match term) - w3*(rage quit).

    In the RL simulator the session-length term is the indicator 'this match
    was completed and the session continued' (1 or 0) and the closeness term
    is the predicted match closeness C in [0,1], so every per-step reward is
    bounded:  rage quit -> [-3.0, -1.8],  completed match -> [1.0, 2.2].
    The function itself is kept general so the tournament builder and
    /api/reward can pass their own (documented) session-length proxies.
    """
    return W1 * float(session_length_delta) + W2 * float(close_match_baseline) - W3 * (1.0 if rage_quit else 0.0)


# --------------------------------------------------------------------------
# Simulated session environment (the MDP "ground truth" generator)
# --------------------------------------------------------------------------
ENV_PARAMS = {
    # ---- match outcome -------------------------------------------------------
    # p_win = clip(0.5 + streak_w*(k-1) - fatigue_w*f - tilt_w*t - action_w*delta, 0.05, 0.95)
    "p_win_base": 0.5,
    "streak_w": 0.20,     # hot hand: a WINNING-bucket player over-performs their MMR band (+20 pp)
    "fatigue_w": 0.03,    # each fatigue level costs 3 pp of win probability
    "tilt_w": 0.06,       # each tilt level costs 6 pp
    "action_w": 0.15,     # LOWER / RAISE moves the lobby one difficulty band (about 15 pp)
    "p_win_min": 0.05, "p_win_max": 0.95,
    # ---- rage quit (only possible after a LOSS) --------------------------------
    "rage_base": 0.005, "rage_tilt": 0.025, "rage_fatigue": 0.01,
    "rage_losing_streak": 0.02, "rage_raise": 0.02, "rage_stomp": 0.04,
    "stomp_threshold": 0.30,   # losing a match you had < 30% to win = being stomped
    # ---- boredom quit (only possible after an easy WIN) ------------------------
    "boredom_threshold": 0.70, "boredom_base": 0.06, "boredom_hot": 0.04,
    # ---- tilt dynamics -----------------------------------------------------------
    "tilt_up_loss": 0.30, "tilt_up_stomp_bonus": 0.25,
    "tilt_down_win": 0.35, "tilt_down_close_bonus": 0.20, "close_threshold": 0.80,
    # ---- streak dynamics (bucket level) ----------------------------------------
    "streak_step": 0.50,
    # ---- fatigue: probability of moving up one level per match (LOWER, KEEP, RAISE)
    "fatigue_up": [0.33, 0.33, 0.36],
}


def p_win(f: int, t: int, k: int, a: int) -> float:
    P = ENV_PARAMS
    p = P["p_win_base"] + P["streak_w"] * (k - 1) - P["fatigue_w"] * f - P["tilt_w"] * t - P["action_w"] * ACTION_DELTA[a]
    return min(P["p_win_max"], max(P["p_win_min"], p))


def closeness(pw: float) -> float:
    """Predicted match closeness C = 1 - 2|p_win - 0.5|  (1 = perfect 50/50 match)."""
    return 1.0 - 2.0 * abs(pw - 0.5)


def rage_prob_after_loss(f: int, t: int, k: int, a: int, pw: float) -> float:
    P = ENV_PARAMS
    q = (P["rage_base"] + P["rage_tilt"] * t + P["rage_fatigue"] * f
         + (P["rage_losing_streak"] if k == 0 else 0.0)
         + (P["rage_raise"] if a == 2 else 0.0)
         + (P["rage_stomp"] if pw < P["stomp_threshold"] else 0.0))
    return min(0.95, max(0.0, q))


def boredom_prob_after_win(f: int, t: int, k: int, a: int, pw: float) -> float:
    """An easy stomp win (p_win >= threshold) can end the session out of boredom."""
    P = ENV_PARAMS
    if pw < P["boredom_threshold"]:
        return 0.0
    return P["boredom_base"] + (P["boredom_hot"] if k == 2 else 0.0)


def step_model(s: int, a: int) -> dict:
    """Closed-form per-step quantities for (s, a) — used for display and for
    computing the simulator's exact transition law (validation only)."""
    f, t, k = decode(s)
    pw = p_win(f, t, k, a)
    C = closeness(pw)
    q = rage_prob_after_loss(f, t, k, a, pw)
    b = boredom_prob_after_win(f, t, k, a, pw)
    return {"p_win": pw, "closeness": C, "rage_given_loss": q, "boredom_given_win": b,
            "rage_prob": (1 - pw) * q, "boredom_prob": pw * b,
            "fatigue_up": ENV_PARAMS["fatigue_up"][a],
            "expected_reward_immediate": (1 - pw) * q * reward_function(0, C, True)
                                         + pw * b * reward_function(0, C, False)
                                         + (1 - (1 - pw) * q - pw * b) * reward_function(1, C, False)}


@dataclass
class StepInfo:
    win: bool
    rage_quit: bool
    natural_end: bool
    p_win: float
    closeness: float
    rage_prob: float
    boredom_quit: bool = False


class SessionEnv:
    """Finite-state stochastic simulator of one play session.

    Exactly five uniforms are drawn per step, in a fixed order, regardless of
    the action. Two policies evaluated with the same episode seed therefore
    face the *same* random numbers (common random numbers) — a fair, low-
    variance comparison.
    """

    def __init__(self, seed: int = 7):
        self.rng = Mulberry32(seed)
        self.s = encode(0, 0, 1)

    def reset(self, state: int | None = None, start_dist: str = "uniform") -> int:
        if state is not None:
            self.s = int(state)
        else:
            self.s = sample_start_state(self.rng, start_dist)
        return self.s

    def step(self, a: int):
        s2, r, done, info = env_step(self.s, a, self.rng)
        self.s = s2
        return s2, r, done, info


def env_step(s: int, a: int, rng: Mulberry32):
    """One environment transition. Returns (next_state, reward, done, StepInfo)."""
    P = ENV_PARAMS
    f, t, k = decode(s)
    pw = p_win(f, t, k, a)
    C = closeness(pw)
    u_win = rng.random(); u_rage = rng.random(); u_tilt = rng.random(); u_streak = rng.random(); u_fat = rng.random()

    win = u_win < pw
    rp = (1 - pw) * rage_prob_after_loss(f, t, k, a, pw)
    if win:
        if u_rage < boredom_prob_after_win(f, t, k, a, pw):
            # session ends after a one-sided win: no rage penalty, but no continuation either
            return TERMINAL, reward_function(0.0, C, False), True, StepInfo(True, False, False, pw, C, rp, True)
    elif u_rage < rage_prob_after_loss(f, t, k, a, pw):
        return TERMINAL, reward_function(0.0, C, True), True, StepInfo(False, True, False, pw, C, rp)

    # tilt transition
    if win:
        p_down = P["tilt_down_win"] + (P["tilt_down_close_bonus"] if C >= P["close_threshold"] else 0.0)
        t2 = t - 1 if (t > 0 and u_tilt < p_down) else t
    else:
        p_up = P["tilt_up_loss"] + (P["tilt_up_stomp_bonus"] if pw < P["stomp_threshold"] else 0.0)
        t2 = min(3, t + 1) if u_tilt < p_up else t
    # streak transition (bucket level: a single opposite result breaks a streak)
    if win:
        k2 = 1 if k == 0 else (2 if (k == 1 and u_streak < P["streak_step"]) else k)
    else:
        k2 = 1 if k == 2 else (0 if (k == 1 and u_streak < P["streak_step"]) else k)
    # fatigue transition; exhausted players who tire further log off (natural end)
    r = reward_function(1.0, C, False)
    if u_fat < P["fatigue_up"][a]:
        if f == 3:
            return TERMINAL, r, True, StepInfo(win, False, True, pw, C, rp)
        f2 = f + 1
    else:
        f2 = f
    return encode(f2, t2, k2), r, False, StepInfo(win, False, False, pw, C, rp)


def exact_transition(s: int, a: int) -> tuple[dict, dict]:
    """The simulator's exact one-step law p(s'|s,a) and r(s,a,s').

    This is available only because WE wrote the simulator. The RL algorithms
    use the *simulation-estimated* model (mdp_model.estimate_model); this
    function exists to validate that estimate and is reported as such.
    """
    P = ENV_PARAMS
    f, t, k = decode(s)
    pw = p_win(f, t, k, a)
    C = closeness(pw)
    q = rage_prob_after_loss(f, t, k, a, pw)
    probs: dict[int, float] = {}
    rew_mass: dict[int, float] = {}

    def add(s2, p, r):
        if p <= 0:
            return
        probs[s2] = probs.get(s2, 0.0) + p
        rew_mass[s2] = rew_mass.get(s2, 0.0) + p * r

    r_rage = reward_function(0.0, C, True)
    r_bored = reward_function(0.0, C, False)
    r_ok = reward_function(1.0, C, False)
    pf = P["fatigue_up"][a]
    for win, p_out in ((True, pw), (False, 1 - pw)):
        if p_out <= 0:
            continue
        if win:
            p_quit = boredom_prob_after_win(f, t, k, a, pw)
            add(TERMINAL, p_out * p_quit, r_bored)
        else:
            p_quit = q
            add(TERMINAL, p_out * p_quit, r_rage)
        p_cont = p_out * (1 - p_quit)
        # tilt outcomes
        if win:
            p_down = P["tilt_down_win"] + (P["tilt_down_close_bonus"] if C >= P["close_threshold"] else 0.0)
            tilt_out = [(t - 1, p_down), (t, 1 - p_down)] if t > 0 else [(t, 1.0)]
            streak_out = [(1, 1.0)] if k == 0 else ([(2, P["streak_step"]), (1, 1 - P["streak_step"])] if k == 1 else [(2, 1.0)])
        else:
            p_up = P["tilt_up_loss"] + (P["tilt_up_stomp_bonus"] if pw < P["stomp_threshold"] else 0.0)
            tilt_out = [(min(3, t + 1), p_up), (t, 1 - p_up)] if t < 3 else [(t, 1.0)]
            streak_out = [(1, 1.0)] if k == 2 else ([(0, P["streak_step"]), (1, 1 - P["streak_step"])] if k == 1 else [(0, 1.0)])
        for t2, pt in tilt_out:
            for k2, pk in streak_out:
                base = p_cont * pt * pk
                if f == 3:
                    add(TERMINAL, base * pf, r_ok)
                else:
                    add(encode(f + 1, t2, k2), base * pf, r_ok)
                add(encode(f, t2, k2), base * (1 - pf), r_ok)
    rewards = {s2: rew_mass[s2] / probs[s2] for s2 in probs}
    return probs, rewards


# --------------------------------------------------------------------------
# Start-state distributions
# --------------------------------------------------------------------------
def start_distribution(name: str = "uniform") -> list[float]:
    """'uniform'       : every one of the 48 states equally likely (exploring
                         starts; used for training/estimation coverage).
       'session_start' : realistic session openings — mostly fresh players,
                         mostly calm, neutral streak (used for evaluation)."""
    if name == "uniform":
        return [1.0 / N_STATES] * N_STATES
    if name == "session_start":
        pf = [0.70, 0.30, 0.0, 0.0]
        pt = [0.50, 0.30, 0.15, 0.05]
        pk = [0.20, 0.60, 0.20]
        d = [0.0] * N_STATES
        for f in range(N_F):
            for t in range(N_T):
                for k in range(N_K):
                    d[encode(f, t, k)] = pf[f] * pt[t] * pk[k]
        return d
    raise ValueError(f"unknown start distribution '{name}'")


def sample_start_state(rng: Mulberry32, name: str = "uniform") -> int:
    if name == "uniform":
        return rng.choice_index(N_STATES)
    return rng.sample_discrete(start_distribution(name))


# --------------------------------------------------------------------------
# Policies (Unit 1: pi(a|s))
# --------------------------------------------------------------------------
def uniform_policy() -> list[list[float]]:
    return [[1.0 / N_ACTIONS] * N_ACTIONS for _ in range(N_STATES_WITH_TERMINAL)]


def constant_policy(action: int) -> list[list[float]]:
    return [[1.0 if a == action else 0.0 for a in ACTIONS] for _ in range(N_STATES_WITH_TERMINAL)]


def deterministic_policy(actions: list[int]) -> list[list[float]]:
    """actions: list of length 48 (or 49) with the action per state."""
    pol = []
    for s in range(N_STATES_WITH_TERMINAL):
        a = actions[s] if s < len(actions) else 1
        pol.append([1.0 if b == a else 0.0 for b in ACTIONS])
    return pol


def heuristic_policy() -> list[list[float]]:
    """Hand-written rule a human designer might ship (the 'manual baseline'):
    lower difficulty when tilted or losing, raise when calm and winning."""
    acts = []
    for s in range(N_STATES):
        f, t, k = decode(s)
        if t >= 2 or k == 0:
            acts.append(0)
        elif t == 0 and k == 2 and f <= 1:
            acts.append(2)
        else:
            acts.append(1)
    return deterministic_policy(acts)


def epsilon_soft(greedy_actions: list[int], epsilon: float) -> list[list[float]]:
    """Unit 5 epsilon-soft policy: 1-eps+eps/|A| on the greedy action, eps/|A| elsewhere."""
    pol = []
    for s in range(N_STATES_WITH_TERMINAL):
        g = greedy_actions[s] if s < len(greedy_actions) else 1
        pol.append([(1 - epsilon + epsilon / N_ACTIONS) if a == g else epsilon / N_ACTIONS for a in ACTIONS])
    return pol


def greedy_from_q(Q, tie_break: str = "first") -> list[int]:
    """argmax_a Q(s,a) per state (ties -> lowest action index, deterministic)."""
    acts = []
    for s in range(len(Q)):
        row = Q[s]
        best = 0
        for a in range(1, len(row)):
            if row[a] > row[best] + 1e-12:
                best = a
        acts.append(best)
    return acts


def sample_action(pi_row, rng: Mulberry32) -> int:
    return rng.sample_discrete(pi_row)


def policy_actions(pi) -> list[int]:
    """Most probable action per state (for display of stochastic policies)."""
    return [max(range(N_ACTIONS), key=lambda a: (pi[s][a], -a)) for s in range(len(pi))]


# --------------------------------------------------------------------------
# Episodes and returns (Unit 1 + Unit 5)
# --------------------------------------------------------------------------
@dataclass
class Transition:
    state: int
    action: int
    reward: float
    next_state: int
    done: bool
    behaviour_prob: float | None = None   # b(a|s) when generated by a known stochastic policy
    info: dict = field(default_factory=dict)


@dataclass
class Episode:
    transitions: list[Transition]
    source: str = "SIMULATED"     # SIMULATED | REAL DATA
    meta: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.transitions)

    def to_dict(self):
        return {"source": self.source, "meta": self.meta,
                "transitions": [asdict(t) for t in self.transitions]}


def discounted_return(rewards: list[float], gamma: float) -> float:
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
    return G


def returns_from_rewards(rewards: list[float], gamma: float) -> list[float]:
    """G_t for every t, computed backwards: G_t = R_{t+1} + gamma * G_{t+1}."""
    G = 0.0
    out = [0.0] * len(rewards)
    for i in range(len(rewards) - 1, -1, -1):
        G = rewards[i] + gamma * G
        out[i] = G
    return out


MAX_EPISODE_STEPS = 200  # safety cap; every policy terminates w.p. 1 long before this


def generate_episode(policy, seed: int, start_state: int | None = None, start_dist: str = "uniform",
                     first_action: int | None = None, record_info: bool = False) -> Episode:
    """Roll out one complete episode.

    Two independent random streams are used: one for the environment (start
    state + transitions) and one for the agent's action sampling, so changing
    the policy never changes the environment's random numbers.
    """
    env_rng = Mulberry32(derive_seed(seed, 0))
    act_rng = Mulberry32(derive_seed(seed, 1))
    s = start_state if start_state is not None else sample_start_state(env_rng, start_dist)
    trans: list[Transition] = []
    for step in range(MAX_EPISODE_STEPS):
        if step == 0 and first_action is not None:
            a = first_action
            bprob = None
        else:
            row = policy[s]
            a = sample_action(row, act_rng)
            bprob = row[a]
        s2, r, done, info = env_step(s, a, env_rng)
        extra = {}
        if record_info:
            extra = {"win": info.win, "rage_quit": info.rage_quit, "natural_end": info.natural_end,
                     "p_win": round(info.p_win, 4), "closeness": round(info.closeness, 4)}
        trans.append(Transition(s, a, r, s2, done, bprob, extra))
        s = s2
        if done:
            break
    return Episode(trans, "SIMULATED", {"seed": seed})


def spec() -> dict:
    """Machine-readable formal definition of the problem (served by /api/rl/spec)."""
    return {
        "agent": "Clutch IQ session controller (lobby-difficulty policy)",
        "environment": "Simulated competitive-gaming session (rl_core.SessionEnv) — SIMULATED dynamics",
        "environment_properties": {
            "observability": "fully observable w.r.t. the bucketed state (a real player's psychology is only partially observable — see limitations)",
            "dynamics": "stochastic", "state_space": "discrete, finite (48 + terminal)",
            "action_space": "discrete (3)", "task": "episodic", "agents": "single agent",
        },
        "state_dimensions": {
            "fatigue": {"levels": FATIGUE_LEVELS, "bucket": "min(3, floor(4*fatigue))"},
            "tilt": {"levels": TILT_LEVELS, "bucket": "min(3, floor(4*tilt))"},
            "streak": {"levels": STREAK_LEVELS, "bucket": "<=-2 / -1..+1 / >=+2"},
        },
        "n_states": N_STATES, "terminal_index": TERMINAL,
        "actions": [{"index": a, "name": ACTION_NAMES[a], "label": ACTION_LABELS[a], "difficulty_delta": ACTION_DELTA[a]} for a in ACTIONS],
        "reward": {
            "formula": "R = w1*(match completed) + w2*C - w3*(rage quit)",
            "weights": {"w1": W1, "w2": W2, "w3": W3},
            "C": "predicted closeness = 1 - 2|p_win - 0.5|",
            "range_rage_quit": [W2 * 0 - W3, W2 * 1 - W3], "range_completed": [W1, W1 + W2],
        },
        "env_params": ENV_PARAMS,
        "start_distributions": {"uniform": "all 48 states equally likely", "session_start": "70% FRESH / 30% WARM; tilt 50/30/15/5%; streak 20/60/20%"},
        "terminal_conditions": ["rage quit after a loss (penalised by w3)", "boredom quit after a one-sided win (no penalty, no continuation)", "natural end: an EXHAUSTED player tires further and logs off"],
        "provenance": "SIMULATED",
    }
