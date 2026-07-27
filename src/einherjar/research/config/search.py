"""
Search Engine Configuration
"""

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True, frozen=True)
class SearchConfig:
    """Discovery Engine configuration."""

    max_conditions: int = 3

    beam_width: int = 200

    max_depth: int = 3

    max_candidates_per_family: int = 100

    exploration_ratio: float = 0.25

    exploitation_ratio: float = 0.75

    novelty_weight: float = 0.30

    diversity_weight: float = 0.25

    family_balance_weight: float = 0.20

    random_seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_conditions": self.max_conditions,
            "beam_width": self.beam_width,
            "max_depth": self.max_depth,
            "max_candidates_per_family": self.max_candidates_per_family,
            "exploration_ratio": self.exploration_ratio,
            "exploitation_ratio": self.exploitation_ratio,
            "novelty_weight": self.novelty_weight,
            "diversity_weight": self.diversity_weight,
            "family_balance_weight": self.family_balance_weight,
            "random_seed": self.random_seed,
        }
