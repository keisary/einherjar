"""
==========================================================
Candidate Model
==========================================================

Un Candidate représente une Hypothesis générée par le moteur.

Il constitue l'unité de travail échangée entre les différents
modules du système jusqu'à sa validation.

Le Candidate ne contient aucun résultat de backtest ni aucune
métrique de performance. Ces informations appartiennent au
ValidatedCandidate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .hypothesis import Hypothesis


@dataclass(frozen=True, slots=True)
class Candidate:
    """
    Candidate généré par le moteur de découverte.
    """

    hypothesis: Hypothesis

    # ==================================================
    # VALIDATION
    # ==================================================

    def __post_init__(self) -> None:

        if not isinstance(
            self.hypothesis,
            Hypothesis,
        ):
            raise TypeError(
                "hypothesis must be a Hypothesis."
            )

    # ==================================================
    # PROPERTIES
    # ==================================================

    @property
    def condition_count(self) -> int:
        return self.hypothesis.condition_count

    # ==================================================
    # SERIALIZATION
    # ==================================================

    def to_dict(self) -> dict[str, Any]:

        return {
            "hypothesis": self.hypothesis.to_dict(),
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        registry,
    ) -> "Candidate":

        return cls(
            hypothesis=Hypothesis.from_dict(
                data["hypothesis"],
                registry,
            ),
        )

    # ==================================================
    # PYTHON PROTOCOL
    # ==================================================

    def __len__(self) -> int:
        return len(self.hypothesis)

    def __iter__(self):
        return iter(self.hypothesis)

    def __hash__(self) -> int:
        return hash(self.hypothesis)

    def __repr__(self) -> str:
        return (
            "Candidate("
            f"conditions={self.condition_count}"
            ")"
        )