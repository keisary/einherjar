"""
Execution Engine Configuration
"""

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class ExecutionConfig:
    """Replay / execution configuration."""

    fees: float = 0.0006

    slippage: float = 0.0002

    spread: float = 0.0001

    allow_long: bool = True

    allow_short: bool = True

    max_open_positions: int = 1