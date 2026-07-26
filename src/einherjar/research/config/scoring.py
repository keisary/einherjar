"""
Global scoring configuration.
"""

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class ScoringConfig:

    novelty: float = 0.20

    diversity: float = 0.20

    robustness: float = 0.20

    persistence: float = 0.20

    profitability: float = 0.20