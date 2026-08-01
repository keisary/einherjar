"""
==========================================================
Trade Model
==========================================================

Un Trade représente une position individuelle exécutée
pendant une simulation ou un backtest.

Il s'agit d'un simple objet de données.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True, slots=True)
class Trade:
    """
    Représentation d'un trade.
    """

    entry_time: datetime

    exit_time: datetime

    entry_price: float

    exit_price: float

    quantity: float

    pnl: float

    # ==================================================
    # VALIDATION
    # ==================================================

    def __post_init__(self) -> None:

        if self.exit_time < self.entry_time:
            raise ValueError(
                "exit_time must be after entry_time."
            )

        if self.quantity <= 0:
            raise ValueError(
                "quantity must be > 0."
            )

    # ==================================================
    # PROPERTIES
    # ==================================================

    @property
    def duration(self):
        return self.exit_time - self.entry_time

    @property
    def is_profitable(self) -> bool:
        return self.pnl > 0

    # ==================================================
    # SERIALIZATION
    # ==================================================

    def to_dict(self) -> dict[str, Any]:

        return {
            "entry_time": self.entry_time.isoformat(),
            "exit_time": self.exit_time.isoformat(),
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "quantity": self.quantity,
            "pnl": self.pnl,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
    ) -> "Trade":

        return cls(
            entry_time=datetime.fromisoformat(
                data["entry_time"]
            ),
            exit_time=datetime.fromisoformat(
                data["exit_time"]
            ),
            entry_price=data["entry_price"],
            exit_price=data["exit_price"],
            quantity=data["quantity"],
            pnl=data["pnl"],
        )

    # ==================================================
    # PYTHON PROTOCOL
    # ==================================================

    def __repr__(self) -> str:
        return (
            "Trade("
            f"entry={self.entry_price}, "
            f"exit={self.exit_price}, "
            f"pnl={self.pnl:.2f}"
            ")"
        )