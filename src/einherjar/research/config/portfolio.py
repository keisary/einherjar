"""
Portfolio Engine Configuration
"""

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class PortfolioConfig:
    """Portfolio selection parameters."""

    max_correlation: float = 0.70

    max_per_family: int = 3

    max_per_asset: int = 5

    max_per_timeframe: int = 5

    target_portfolio_size: int = 25
