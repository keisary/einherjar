"""
Execution Engine Configuration
"""

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True, frozen=True)
class ExecutionConfig:
    """Replay / execution configuration."""

    fees: float = 0.0006

    slippage: float = 0.0002

    spread: float = 0.0001

    allow_long: bool = True

    allow_short: bool = True

    max_open_positions: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "fees": self.fees,
            "slippage": self.slippage,
            "spread": self.spread,
            "allow_long": self.allow_long,
            "allow_short": self.allow_short,
            "max_open_positions": self.max_open_positions,
        }
