"""
==========================================================
Trade Journal Model
==========================================================

Un Journal est une collection immuable de Trade.

Il ne réalise aucun calcul complexe ; il centralise
simplement les trades produits par une simulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import Iterable

from .trade import Trade


@dataclass(frozen=True, slots=True)
class Journal:
    """
    Collection de Trade.
    """

    trades: tuple[Trade, ...]

    # ==================================================
    # INITIALIZATION
    # ==================================================

    def __init__(
        self,
        trades: Iterable[Trade],
    ) -> None:

        trades = tuple(trades)

        for trade in trades:
            if not isinstance(
                trade,
                Trade,
            ):
                raise TypeError(
                    "All elements must be Trade."
                )

        object.__setattr__(
            self,
            "trades",
            trades,
        )

    # ==================================================
    # PROPERTIES
    # ==================================================

    @property
    def trade_count(self) -> int:
        return len(self.trades)

    @property
    def winning_trades(self) -> int:
        return sum(
            trade.is_profitable
            for trade in self.trades
        )

    @property
    def losing_trades(self) -> int:
        return self.trade_count - self.winning_trades

    @property
    def total_pnl(self) -> float:
        return sum(
            trade.pnl
            for trade in self.trades
        )

    # ==================================================
    # SERIALIZATION
    # ==================================================

    def to_dict(self) -> dict[str, Any]:

        return {
            "trades": [
                trade.to_dict()
                for trade in self.trades
            ]
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
    ) -> "Journal":

        return cls(
            trades=[
                Trade.from_dict(trade)
                for trade in data["trades"]
            ]
        )

    # ==================================================
    # PYTHON PROTOCOL
    # ==================================================

    def __len__(self) -> int:
        return len(self.trades)

    def __iter__(self):
        return iter(self.trades)

    def __getitem__(
        self,
        index: int,
    ) -> Trade:
        return self.trades[index]

    def __hash__(self) -> int:
        return hash(self.trades)

    def __repr__(self) -> str:
        return (
            "Journal("
            f"trades={self.trade_count}, "
            f"pnl={self.total_pnl:.2f}"
            ")"
        )