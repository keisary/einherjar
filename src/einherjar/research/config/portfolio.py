"""
Portfolio Engine Configuration
"""

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True, frozen=True)
class PortfolioConfig:
    """Portfolio selection parameters."""

    # --- Champs historiques ---
    max_correlation: float = 0.70
    max_per_family: int = 3
    max_per_asset: int = 5
    max_per_timeframe: int = 5
    target_portfolio_size: int = 25

    # --- Champs consommés par PortfolioSelectorSettings ---
    max_selected: int = 12
    min_trade_count: int = 3
    min_win_rate: float = 0.40
    min_profit_factor: float = 1.0
    min_expectancy: float = 0.0
    min_total_pnl: float = 0.0
    require_healthy: bool = False
    exclude_duplicates: bool = True
    keep_best_per_family: int = 3
    max_same_profile: int = 2
    max_same_family: int = 4
    weight_total_pnl: float = 0.28
    weight_expectancy: float = 0.18
    weight_win_rate: float = 0.18
    weight_profit_factor: float = 0.14
    weight_health: float = 0.10
    weight_drawdown: float = 0.08
    weight_mae_mfe: float = 0.04
    weight_coverage: float = 0.04

    # --- Champs consommés par CapitalSettings ---
    total_capital: float = 1.0
    reserve_ratio: float = 0.0
    min_position_ratio: float = 0.0
    max_position_ratio: float = 0.35
    min_position_value: float = 0.0
    max_position_value: float = float("inf")
    min_positions: int = 1
    max_positions: int = 12
    allow_residual: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_correlation": self.max_correlation,
            "max_per_family": self.max_per_family,
            "max_per_asset": self.max_per_asset,
            "max_per_timeframe": self.max_per_timeframe,
            "target_portfolio_size": self.target_portfolio_size,
            "max_selected": self.max_selected,
            "min_trade_count": self.min_trade_count,
            "min_win_rate": self.min_win_rate,
            "min_profit_factor": self.min_profit_factor,
            "min_expectancy": self.min_expectancy,
            "min_total_pnl": self.min_total_pnl,
            "require_healthy": self.require_healthy,
            "exclude_duplicates": self.exclude_duplicates,
            "keep_best_per_family": self.keep_best_per_family,
            "max_same_profile": self.max_same_profile,
            "max_same_family": self.max_same_family,
            "weight_total_pnl": self.weight_total_pnl,
            "weight_expectancy": self.weight_expectancy,
            "weight_win_rate": self.weight_win_rate,
            "weight_profit_factor": self.weight_profit_factor,
            "weight_health": self.weight_health,
            "weight_drawdown": self.weight_drawdown,
            "weight_mae_mfe": self.weight_mae_mfe,
            "weight_coverage": self.weight_coverage,
            "total_capital": self.total_capital,
            "reserve_ratio": self.reserve_ratio,
            "min_position_ratio": self.min_position_ratio,
            "max_position_ratio": self.max_position_ratio,
            "min_position_value": self.min_position_value,
            "max_position_value": self.max_position_value,
            "min_positions": self.min_positions,
            "max_positions": self.max_positions,
            "allow_residual": self.allow_residual,
        }
