"""
==========================================================
Condition Model
==========================================================

Une Condition représente une règle élémentaire utilisée
pour construire une Hypothesis.

Exemples :

    close > ema_20
    rsi_14 >= 70
    volume < sma_volume_20

Une Condition est une description IMMUTABLE.
Elle ne contient aucune logique d'évaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .enums import ComparisonOperator
from .feature import Feature


@dataclass(frozen=True, slots=True)
class Condition:
    """
    Représente une condition élémentaire.

    Une Condition compare une Feature à une autre Feature
    ou à une valeur constante.
    """

    # ==================================================
    # OPERANDS
    # ==================================================

    left: Feature

    operator: ComparisonOperator

    right: Feature | int | float | bool

    # ==================================================
    # VALIDATION
    # ==================================================

    def __post_init__(self) -> None:

        if not isinstance(self.left, Feature):
            raise TypeError(
                "left must be a Feature."
            )

        if not isinstance(
            self.operator,
            ComparisonOperator,
        ):
            raise TypeError(
                "operator must be a ComparisonOperator."
            )

        if not isinstance(
            self.right,
            (
                Feature,
                int,
                float,
                bool,
            ),
        ):
            raise TypeError(
                "right must be a Feature, int, float or bool."
            )

    # ==================================================
    # PROPERTIES
    # ==================================================

    @property
    def compares_feature(self) -> bool:
        """
        True si la condition compare deux Features.
        """
        return isinstance(self.right, Feature)

    @property
    def compares_constant(self) -> bool:
        """
        True si la condition compare une Feature à une
        constante.
        """
        return not isinstance(self.right, Feature)

    # ==================================================
    # SERIALIZATION
    # ==================================================

    def to_dict(self) -> dict[str, Any]:

        return {
            "left": self.left.column_index,
            "operator": self.operator.value,
            "right": (
                {
                    "type": "feature",
                    "value": self.right.column_index,
                }
                if isinstance(self.right, Feature)
                else {
                    "type": "constant",
                    "value": self.right,
                }
            ),
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        registry,
    ) -> "Condition":

        right = data["right"]

        if right["type"] == "feature":
            right_operand = registry[right["value"]]
        else:
            right_operand = right["value"]

        return cls(
            left=registry[data["left"]],
            operator=ComparisonOperator(
                data["operator"]
            ),
            right=right_operand,
        )

    # ==================================================
    # OBJECT PROTOCOL
    # ==================================================

    def __hash__(self) -> int:

        right = (
            self.right.column_index
            if isinstance(self.right, Feature)
            else self.right
        )

        return hash(
            (
                self.left.column_index,
                self.operator,
                right,
            )
        )

    def __repr__(self) -> str:

        right = (
            self.right.name
            if isinstance(self.right, Feature)
            else repr(self.right)
        )

        return (
            "Condition("
            f"{self.left.name} "
            f"{self.operator.value} "
            f"{right}"
            ")"
        )
