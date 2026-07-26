"""
==========================================================
Report Model
==========================================================

Un Report représente le résultat global d'une exécution du
moteur.

Il centralise les objets produits durant le pipeline sans
effectuer de calcul métier.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .journal import Journal
from .validated_candidate import ValidatedCandidate


@dataclass(frozen=True, slots=True)
class Report:
    """
    Résultat d'une exécution complète.
    """

    candidate: ValidatedCandidate

    journal: Journal

    metadata: dict[str, Any] | None = None

    # ==================================================
    # VALIDATION
    # ==================================================

    def __post_init__(self) -> None:

        if not isinstance(
            self.candidate,
            ValidatedCandidate,
        ):
            raise TypeError(
                "candidate must be a ValidatedCandidate."
            )

        if not isinstance(
            self.journal,
            Journal,
        ):
            raise TypeError(
                "journal must be a Journal."
            )

        object.__setattr__(
            self,
            "metadata",
            {} if self.metadata is None else dict(self.metadata),
        )

    # ==================================================
    # SERIALIZATION
    # ==================================================

    def to_dict(self) -> dict[str, Any]:

        return {
            "candidate": self.candidate.to_dict(),
            "journal": self.journal.to_dict(),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        registry,
    ) -> "Report":

        return cls(
            candidate=ValidatedCandidate.from_dict(
                data["candidate"],
                registry,
            ),
            journal=Journal.from_dict(
                data["journal"],
            ),
            metadata=data.get(
                "metadata",
                {},
            ),
        )

    # ==================================================
    # PYTHON PROTOCOL
    # ==================================================

    def __hash__(self) -> int:
        return hash(
            (
                self.candidate,
                self.journal,
            )
        )

    def __repr__(self) -> str:
        return (
            "Report("
            f"trades={len(self.journal)}, "
            f"conditions={self.candidate.condition_count}"
            ")"
        )