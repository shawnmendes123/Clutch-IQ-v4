"""
models.py
Core data structures for the Adaptive Pro-Gamer Matchmaker.

A Player is represented by:
  - Static skill (Elo/MMR baseline)
  - Playstyle vector (learned embedding input): aggression, positioning,
    communication, support tendency, objective focus
  - Live psychological/session state: streak, tilt, fatigue, reaction drift,
    comms frequency, session length so far
"""
from dataclasses import dataclass, field
import random
import uuid


PLAYSTYLE_ARCHETYPES = {
    "Aggressive Entry":  [0.90, 0.30, 0.40, 0.15, 0.55],
    "Passive Support":   [0.15, 0.55, 0.85, 0.90, 0.45],
    "Objective Anchor":  [0.35, 0.70, 0.55, 0.40, 0.95],
    "Flex Rotator":      [0.55, 0.60, 0.60, 0.50, 0.60],
    "Sniper/Isolation":  [0.70, 0.85, 0.20, 0.10, 0.35],
}
# vector dims: [aggression, positioning, comms, support, objective_focus]


@dataclass
class Player:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    mmr: int = 1500
    archetype: str = "Flex Rotator"
    playstyle_vector: list = field(default_factory=list)

    # Live psychological / session state (the "Player State" segmentation)
    win_streak: int = 0          # positive = winning streak, negative = losing streak
    reaction_drift_ms: float = 0.0   # increase = fatigue-driven slowdown
    comms_freq: float = 0.5          # 0-1, messages/pings per minute normalized
    session_minutes: float = 0.0     # time played this session
    tilt_score: float = 0.0          # 0 (calm) - 1 (about to rage quit)
    fatigue_index: float = 0.0       # 0 (fresh) - 1 (exhausted)

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "mmr": self.mmr,
            "archetype": self.archetype,
            "playstyle_vector": self.playstyle_vector,
            "win_streak": self.win_streak,
            "reaction_drift_ms": round(self.reaction_drift_ms, 1),
            "comms_freq": round(self.comms_freq, 2),
            "session_minutes": round(self.session_minutes, 1),
            "tilt_score": round(self.tilt_score, 3),
            "fatigue_index": round(self.fatigue_index, 3),
        }


NAME_POOL = [
    "Vexen", "Ryza", "Kael", "Nova", "Zeph", "Ashra", "Tinker", "Ori",
    "Quill", "Draven", "Mira", "Sable", "Fenix", "Lyric", "Bram", "Ixel",
    "Corvo", "Wynn", "Talon", "Juno", "Rook", "Nyx", "Ember", "Skye",
    "Halo", "Grim", "Petal", "Vantage", "Onyx", "Riven"
]


def random_player(rng: random.Random) -> Player:
    archetype = rng.choice(list(PLAYSTYLE_ARCHETYPES.keys()))
    base = PLAYSTYLE_ARCHETYPES[archetype]
    vec = [max(0, min(1, v + rng.uniform(-0.12, 0.12))) for v in base]

    win_streak = rng.choice([rng.randint(-6, -1), rng.randint(1, 6), 0])
    session_minutes = rng.uniform(0, 240)
    fatigue = min(1.0, max(0.0, (session_minutes / 240) + rng.uniform(-0.1, 0.1)))
    tilt = min(1.0, max(0.0, (max(0, -win_streak) / 6) * 0.8 + rng.uniform(0, 0.2)))
    reaction_drift = fatigue * rng.uniform(20, 90)  # ms slowdown
    comms = max(0.0, min(1.0, rng.uniform(0.1, 0.9) - tilt * 0.3))

    return Player(
        name=rng.choice(NAME_POOL) + str(rng.randint(10, 99)),
        mmr=rng.randint(900, 2600),
        archetype=archetype,
        playstyle_vector=vec,
        win_streak=win_streak,
        reaction_drift_ms=reaction_drift,
        comms_freq=comms,
        session_minutes=session_minutes,
        tilt_score=tilt,
        fatigue_index=fatigue,
    )


def generate_player_pool(n: int = 40, seed: int = None) -> list[Player]:
    rng = random.Random(seed)
    return [random_player(rng) for _ in range(n)]
