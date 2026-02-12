"""Pokemon-inspired micro battle environment (tabular MDP wrapper).

V1 goal: small discrete state/action space suitable for DP (policy iteration).
Later: extend to Markov game + partial observability + belief modeling.
"""

from dataclasses import dataclass
from typing import Any, Dict, Tuple

@dataclass(frozen=True)
class EnvConfig:
    hp_bins: int = 3
    gamma: float = 0.98
    opponent_policy: str = "random"

class PokemonMicroEnv:
    def __init__(self, config: EnvConfig = EnvConfig()):
        self.config = config

    def reset(self, seed: int | None = None) -> Tuple[Any, Dict]:
        """Return (observation, info)."""
        raise NotImplementedError

    def step(self, action: int):
        """Return (observation, reward, terminated, truncated, info)."""
        raise NotImplementedError

    def render(self, mode: str = "human") -> None:
        raise NotImplementedError
