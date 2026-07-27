"""
Global scoring configuration.
"""

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True, frozen=True)
class ScoringConfig:

    novelty: float = 0.20

    diversity: float = 0.20

    robustness: float = 0.20

    persistence: float = 0.20

    profitability: float = 0.20

    def to_dict(self) -> dict[str, Any]:
        return {
            "novelty": self.novelty,
            "diversity": self.diversity,
            "robustness": self.robustness,
            "persistence": self.persistence,
            "profitability": self.profitability,
        }
