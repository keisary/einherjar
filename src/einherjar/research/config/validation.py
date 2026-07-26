"""
Validation Engine Configuration
"""

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class ValidationConfig:
    """Scientific validation parameters."""

    min_trades: int = 60

    min_psr: float = 0.95

    min_dsr: float = 0.80

    min_persistence: float = 0.80

    min_profit_factor: float = 1.20

    min_expectancy: float = 0.0

    max_drawdown: float = 0.25

    walk_forward_splits: int = 5