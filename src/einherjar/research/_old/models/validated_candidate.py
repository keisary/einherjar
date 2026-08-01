"""
==========================================================
Validated Candidate Model
==========================================================

Un ValidatedCandidate représente un Candidate ayant passé
avec succès le processus de validation.

Il regroupe le Candidate d'origine ainsi que les résultats
produits par le Validator.

Aucune logique métier n'est implémentée dans ce modèle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .candidate import Candidate


@dataclass(frozen=True, slots=True)
class ValidatedCandidate:
    """
    Candidate validé par le moteur.
    """

    candidate: Candidate

    metrics: dict[str, Any]

    # ==================================================
    # VALIDATION
    # ==================================================

    def __post_init__(self) -> None:

        if not isinstance(
            self.candidate,
            Candidate,
        ):
            raise TypeError(
                "candidate must be a Candidate."
            )

        object.__setattr__(
            self,
            "metrics",
            dict(self.metrics),
        )

    # ==================================================
    # PROPERTIES
    # ==================================================

    @property
    def hypothesis(self):
        return self.candidate.hypothesis

    @property
    def condition_count(self) -> int:
        return self.candidate.condition_count

    # ==================================================
    # SERIALIZATION
    # ==================================================

    def to_dict(self) -> dict[str, Any]:

        return {
            "candidate": self.candidate.to_dict(),
            "metrics": dict(self.metrics),
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        registry,
    ) -> "ValidatedCandidate":

        return cls(
            candidate=Candidate.from_dict(
                data["candidate"],
                registry,
            ),
            metrics=data.get("metrics", {}),
        )

    # ==================================================
    # PYTHON PROTOCOL
    # ==================================================

    def __hash__(self) -> int:
        return hash(
            (
                self.candidate,
                tuple(sorted(self.metrics.items())),
            )
        )

    def __repr__(self) -> str:
        return (
            "ValidatedCandidate("
            f"conditions={self.condition_count}, "
            f"metrics={len(self.metrics)}"
            ")"
        )