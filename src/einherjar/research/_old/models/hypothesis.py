"""
==========================================================
Hypothesis Model
==========================================================

Une Hypothesis représente un ensemble de Conditions.

Elle est une description IMMUTABLE d'une hypothèse de
trading.

Une Hypothesis ne possède aucune logique d'évaluation,
de validation ou de simulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import Iterable

from .condition import Condition


@dataclass(frozen=True, slots=True)
class Hypothesis:
    """
    Agrégation immuable de Conditions.
    """

    conditions: tuple[Condition, ...]

    # ==================================================
    # INITIALIZATION
    # ==================================================

    def __init__(
        self,
        conditions: Iterable[Condition],
    ) -> None:

        conditions = tuple(conditions)

        if not conditions:
            raise ValueError(
                "Hypothesis must contain at least one Condition."
            )

        for condition in conditions:
            if not isinstance(condition, Condition):
                raise TypeError(
                    "All elements must be Condition instances."
                )

        object.__setattr__(
            self,
            "conditions",
            conditions,
        )

    # ==================================================
    # PROPERTIES
    # ==================================================

    @property
    def condition_count(self) -> int:
        return len(self.conditions)

    @property
    def is_empty(self) -> bool:
        return len(self.conditions) == 0

    # ==================================================
    # SERIALIZATION
    # ==================================================

    def to_dict(self) -> dict[str, Any]:

        return {
            "conditions": [
                condition.to_dict()
                for condition in self.conditions
            ]
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        registry,
    ) -> "Hypothesis":

        return cls(
            conditions=[
                Condition.from_dict(
                    condition,
                    registry,
                )
                for condition in data["conditions"]
            ]
        )

    # ==================================================
    # PYTHON PROTOCOL
    # ==================================================

    def __len__(self) -> int:
        return len(self.conditions)

    def __iter__(self):
        return iter(self.conditions)

    def __getitem__(
        self,
        index: int,
    ) -> Condition:
        return self.conditions[index]

    def __contains__(
        self,
        condition: Condition,
    ) -> bool:
        return condition in self.conditions

    def __hash__(self) -> int:
        return hash(self.conditions)

    def __repr__(self) -> str:
        return (
            "Hypothesis("
            f"conditions={len(self.conditions)}"
            ")"
        )