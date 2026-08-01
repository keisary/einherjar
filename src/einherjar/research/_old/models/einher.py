"""
==========================================================
Einher Model
==========================================================

Un Einher représente une stratégie découverte par le
moteur.

Il regroupe l'ensemble des objets produits durant le
pipeline de découverte.

Cet objet est purement descriptif.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .fingerprint import fingerprint_model
from .journal import Journal
from .profile import Profile
from .report import Report
from .validated_candidate import ValidatedCandidate


@dataclass(frozen=True, slots=True)
class Einher:
    """
    Représentation complète d'un Einher.
    """

    profile: Profile

    candidate: ValidatedCandidate

    journal: Journal

    report: Report

    execution_result: Any | None = None

    # ==================================================
    # VALIDATION
    # ==================================================

    def __post_init__(self) -> None:

        if not isinstance(
            self.profile,
            Profile,
        ):
            raise TypeError(
                "profile must be a Profile."
            )

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

        if not isinstance(
            self.report,
            Report,
        ):
            raise TypeError(
                "report must be a Report."
            )

    # ==================================================
    # PROPERTIES
    # ==================================================

    @property
    def fingerprint(self) -> str:
        return fingerprint_model(self)

    # ==================================================
    # SERIALIZATION
    # ==================================================

    def to_dict(self) -> dict[str, Any]:

        return {
            "profile": self.profile.to_dict(),
            "candidate": self.candidate.to_dict(),
            "journal": self.journal.to_dict(),
            "report": self.report.to_dict(),
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        registry,
    ) -> "Einher":

        return cls(
            profile=Profile.from_dict(
                data["profile"],
            ),
            candidate=ValidatedCandidate.from_dict(
                data["candidate"],
                registry,
            ),
            journal=Journal.from_dict(
                data["journal"],
            ),
            report=Report.from_dict(
                data["report"],
                registry,
            ),
        )

    @classmethod
    def from_execution_result(cls, result: Any) -> "Einher":
        from execution.execution_report import ExecutionResult
        from execution.profiler import ExecutionProfile

        if not isinstance(result, ExecutionResult):
            raise TypeError("result must be an ExecutionResult.")

        profile = (
            result.profile.to_profile_model()
            if isinstance(result.profile, ExecutionProfile)
            else Profile(name="unknown")
        )

        candidate = result.validated_candidate
        if candidate is None:
            candidate = result.candidate
        if not isinstance(candidate, ValidatedCandidate):
            raise TypeError(
                "Cannot build Einher: validated_candidate is missing or invalid."
            )

        report = Report(
            candidate=candidate,
            journal=result.journal,
            metadata=dict(result.metadata),
        )

        return cls(
            profile=profile,
            candidate=candidate,
            journal=result.journal,
            report=report,
            execution_result=result,
        )

    # ==================================================
    # PYTHON PROTOCOL
    # ==================================================

    def __hash__(self) -> int:
        return hash(self.fingerprint)

    def __repr__(self) -> str:
        return (
            "Einher("
            f"name='{self.profile.name}', "
            f"conditions={self.candidate.condition_count}, "
            f"trades={len(self.journal)}"
            ")"
        )
