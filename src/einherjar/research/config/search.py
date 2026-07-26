"""
Search Engine Configuration
"""

from dataclasses import dataclass


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