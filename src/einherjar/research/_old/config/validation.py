"""
Validation Engine Configuration
"""

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True, frozen=True)
class ValidationConfig:
    """Scientific validation parameters."""

    # --- Champs historiques (conservés pour compatibilité) ---
    min_trades: int = 60
    min_psr: float = 0.95
    min_dsr: float = 0.80
    min_persistence: float = 0.80
    min_profit_factor: float = 1.20
    min_expectancy: float = 0.0
    max_drawdown: float = 0.25
    walk_forward_splits: int = 5

    # --- Champs consommés par ValidationSettings ---
    split_name: str = "validation"
    batch_size: int = 50_000
    sample_size: int | None = None
    windows: int = 4

    min_conditions: int = 1
    max_conditions: int = 12

    min_support: int = 50
    min_coverage: float = 0.005

    min_score: float = 0.60
    min_significance: float = 0.55
    min_robustness: float = 0.55
    min_temporal_stability: float = 0.50

    require_positive_lift: bool = False
    positive_target_threshold: float = 0.0

    allow_duplicate_candidates: bool = False
    enable_binary_metrics: bool = True

    scoring_weight_significance: float = 0.30
    scoring_weight_robustness: float = 0.25
    scoring_weight_persistence: float = 0.20
    scoring_weight_temporal: float = 0.15
    scoring_weight_structural: float = 0.10

    random_seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_trades": self.min_trades,
            "min_psr": self.min_psr,
            "min_dsr": self.min_dsr,
            "min_persistence": self.min_persistence,
            "min_profit_factor": self.min_profit_factor,
            "min_expectancy": self.min_expectancy,
            "max_drawdown": self.max_drawdown,
            "walk_forward_splits": self.walk_forward_splits,
            "split_name": self.split_name,
            "batch_size": self.batch_size,
            "sample_size": self.sample_size,
            "windows": self.windows,
            "min_conditions": self.min_conditions,
            "max_conditions": self.max_conditions,
            "min_support": self.min_support,
            "min_coverage": self.min_coverage,
            "min_score": self.min_score,
            "min_significance": self.min_significance,
            "min_robustness": self.min_robustness,
            "min_temporal_stability": self.min_temporal_stability,
            "require_positive_lift": self.require_positive_lift,
            "positive_target_threshold": self.positive_target_threshold,
            "allow_duplicate_candidates": self.allow_duplicate_candidates,
            "enable_binary_metrics": self.enable_binary_metrics,
            "scoring_weight_significance": self.scoring_weight_significance,
            "scoring_weight_robustness": self.scoring_weight_robustness,
            "scoring_weight_persistence": self.scoring_weight_persistence,
            "scoring_weight_temporal": self.scoring_weight_temporal,
            "scoring_weight_structural": self.scoring_weight_structural,
            "random_seed": self.random_seed,
        }
