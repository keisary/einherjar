"""
==========================================================
Validation Rejection
==========================================================

Structures et outils de suivi des rejets pendant la phase
Validation.

Ce module centralise :
- la normalisation des raisons de rejet,
- la représentation immuable d'un rejet,
- un registre simple de comptage et d'historique.

Il ne valide rien à lui seul.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from typing import Mapping
from typing import Sequence

__all__ = [
    "RejectionReason",
    "ValidationRejection",
    "RejectionRegistry",
    "normalize_reasons",
]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _to_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}

    if hasattr(value, "to_dict") and callable(value.to_dict):
        value = value.to_dict()

    if isinstance(value, Mapping):
        return dict(value)

    return {}


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default

    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "no", "n", "off"}:
            return False

    return bool(value)


def _coerce_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default

    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_reason_text(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, RejectionReason):
        return value.value

    text = str(value).strip().lower()
    return text


def normalize_reasons(reasons: Sequence[Any] | Any) -> tuple[str, ...]:
    if reasons is None:
        return ()

    if isinstance(reasons, (str, RejectionReason)):
        reasons = (reasons,)

    output: list[str] = []
    seen: set[str] = set()

    for reason in reasons:
        normalized = _normalize_reason_text(reason)
        if not normalized:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        output.append(normalized)

    return tuple(output)


class RejectionReason(str, Enum):
    """
    Raisons de rejet standardisées pour la Validation.
    """

    DUPLICATE_CANDIDATE = "duplicate_candidate"
    TOO_FEW_CONDITIONS = "too_few_conditions"
    TOO_MANY_CONDITIONS = "too_many_conditions"
    INSUFFICIENT_SUPPORT = "insufficient_support"
    COVERAGE_TOO_LOW = "coverage_too_low"
    SIGNIFICANCE_TOO_LOW = "significance_too_low"
    ROBUSTNESS_TOO_LOW = "robustness_too_low"
    PERSISTENCE_TOO_LOW = "persistence_too_low"
    TEMPORAL_STABILITY_TOO_LOW = "temporal_stability_too_low"
    OVERALL_SCORE_TOO_LOW = "overall_score_too_low"
    NON_POSITIVE_LIFT = "non_positive_lift"
    DUPLICATE_CONDITIONS = "duplicate_conditions"
    EMPTY_HYPOTHESIS = "empty_hypothesis"
    INVALID_STRUCTURE = "invalid_structure"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ValidationRejection:
    """
    Rejet immuable d'un candidat pendant la Validation.
    """

    candidate_fingerprint: str
    hypothesis_fingerprint: str

    reasons: tuple[str, ...] = ()
    split_name: str = "validation"
    duplicate: bool = False
    score: float = 0.0

    metrics: dict[str, Any] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)

    created_at: datetime = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_fingerprint", str(self.candidate_fingerprint))
        object.__setattr__(self, "hypothesis_fingerprint", str(self.hypothesis_fingerprint))
        object.__setattr__(self, "reasons", normalize_reasons(self.reasons))
        object.__setattr__(self, "split_name", str(self.split_name).strip() or "validation")
        object.__setattr__(self, "duplicate", _coerce_bool(self.duplicate, False))
        object.__setattr__(self, "score", float(self.score))
        object.__setattr__(self, "metrics", dict(self.metrics))
        object.__setattr__(self, "details", dict(self.details))

    @property
    def primary_reason(self) -> str:
        return self.reasons[0] if self.reasons else RejectionReason.UNKNOWN.value

    @property
    def is_duplicate(self) -> bool:
        return self.duplicate or RejectionReason.DUPLICATE_CANDIDATE.value in self.reasons

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_fingerprint": self.candidate_fingerprint,
            "hypothesis_fingerprint": self.hypothesis_fingerprint,
            "reasons": list(self.reasons),
            "split_name": self.split_name,
            "duplicate": self.duplicate,
            "score": self.score,
            "metrics": dict(self.metrics),
            "details": dict(self.details),
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ValidationRejection":
        return cls(
            candidate_fingerprint=data["candidate_fingerprint"],
            hypothesis_fingerprint=data["hypothesis_fingerprint"],
            reasons=tuple(data.get("reasons", ())),
            split_name=data.get("split_name", "validation"),
            duplicate=_coerce_bool(data.get("duplicate"), False),
            score=_coerce_float(data.get("score"), 0.0),
            metrics=_to_mapping(data.get("metrics", {})),
            details=_to_mapping(data.get("details", {})),
            created_at=(
                datetime.fromisoformat(data["created_at"])
                if isinstance(data.get("created_at"), str) and data.get("created_at")
                else _utc_now()
            ),
        )

    def __repr__(self) -> str:
        return (
            "ValidationRejection("
            f"split='{self.split_name}', "
            f"duplicate={self.duplicate}, "
            f"reasons={len(self.reasons)}"
            ")"
        )


@dataclass(slots=True)
class RejectionRegistry:
    """
    Registre simple des rejets de Validation.
    """

    name: str = "validation"

    records: list[ValidationRejection] = field(default_factory=list)
    reason_counts: Counter[str] = field(default_factory=Counter)
    split_counts: Counter[str] = field(default_factory=Counter)

    first_seen_at: datetime | None = None
    last_seen_at: datetime | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", str(self.name).strip() or "validation")
        object.__setattr__(self, "records", list(self.records))
        object.__setattr__(self, "reason_counts", Counter(self.reason_counts))
        object.__setattr__(self, "split_counts", Counter(self.split_counts))

    def add(self, rejection: ValidationRejection | Mapping[str, Any]) -> ValidationRejection:
        if not isinstance(rejection, ValidationRejection):
            rejection = ValidationRejection.from_dict(rejection)

        self.records.append(rejection)

        if self.first_seen_at is None:
            self.first_seen_at = rejection.created_at
        self.last_seen_at = rejection.created_at

        self.split_counts[rejection.split_name] += 1
        for reason in rejection.reasons:
            self.reason_counts[reason] += 1

        return rejection

    def reject(
        self,
        candidate_fingerprint: str,
        hypothesis_fingerprint: str,
        *,
        reasons: Sequence[Any] | Any,
        split_name: str = "validation",
        duplicate: bool = False,
        score: float = 0.0,
        metrics: Mapping[str, Any] | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> ValidationRejection:
        rejection = ValidationRejection(
            candidate_fingerprint=candidate_fingerprint,
            hypothesis_fingerprint=hypothesis_fingerprint,
            reasons=normalize_reasons(reasons),
            split_name=split_name,
            duplicate=duplicate,
            score=score,
            metrics=_to_mapping(metrics),
            details=_to_mapping(details),
        )
        return self.add(rejection)

    def explain(self, reason: Any) -> str:
        key = _normalize_reason_text(reason)

        if key == RejectionReason.DUPLICATE_CANDIDATE.value:
            return "Le candidat est déjà connu et a été rejeté comme doublon."
        if key == RejectionReason.TOO_FEW_CONDITIONS.value:
            return "L'hypothèse contient trop peu de conditions pour être validée."
        if key == RejectionReason.TOO_MANY_CONDITIONS.value:
            return "L'hypothèse dépasse la complexité autorisée."
        if key == RejectionReason.INSUFFICIENT_SUPPORT.value:
            return "Le support observé est insuffisant."
        if key == RejectionReason.COVERAGE_TOO_LOW.value:
            return "La couverture du signal est trop faible."
        if key == RejectionReason.SIGNIFICANCE_TOO_LOW.value:
            return "La significativité statistique est trop faible."
        if key == RejectionReason.ROBUSTNESS_TOO_LOW.value:
            return "La robustesse globale est insuffisante."
        if key == RejectionReason.PERSISTENCE_TOO_LOW.value:
            return "Le signal ne persiste pas suffisamment dans le temps."
        if key == RejectionReason.TEMPORAL_STABILITY_TOO_LOW.value:
            return "La stabilité temporelle est trop faible."
        if key == RejectionReason.OVERALL_SCORE_TOO_LOW.value:
            return "Le score global de validation est trop faible."
        if key == RejectionReason.NON_POSITIVE_LIFT.value:
            return "Le signal n'apporte pas de lift positif."
        if key == RejectionReason.DUPLICATE_CONDITIONS.value:
            return "L'hypothèse contient des conditions dupliquées."
        if key == RejectionReason.EMPTY_HYPOTHESIS.value:
            return "L'hypothèse est vide."
        if key == RejectionReason.INVALID_STRUCTURE.value:
            return "La structure du candidat est invalide."
        return "Raison de rejet non spécifiée."

    def most_common(self, n: int = 10) -> tuple[tuple[str, int], ...]:
        return tuple(self.reason_counts.most_common(max(1, _coerce_int(n, 10))))

    def clear(self) -> None:
        self.records.clear()
        self.reason_counts.clear()
        self.split_counts.clear()
        self.first_seen_at = None
        self.last_seen_at = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "records": [record.to_dict() for record in self.records],
            "reason_counts": dict(self.reason_counts),
            "split_counts": dict(self.split_counts),
            "first_seen_at": self.first_seen_at.isoformat() if self.first_seen_at else None,
            "last_seen_at": self.last_seen_at.isoformat() if self.last_seen_at else None,
            "total": len(self.records),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RejectionRegistry":
        registry = cls(
            name=data.get("name", "validation"),
            reason_counts=Counter(_to_mapping(data.get("reason_counts", {}))),
            split_counts=Counter(_to_mapping(data.get("split_counts", {}))),
        )

        registry.records = [ValidationRejection.from_dict(item) for item in data.get("records", [])]
        if data.get("first_seen_at"):
            registry.first_seen_at = datetime.fromisoformat(data["first_seen_at"])
        if data.get("last_seen_at"):
            registry.last_seen_at = datetime.fromisoformat(data["last_seen_at"])
        return registry

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self):
        return iter(self.records)

    def __repr__(self) -> str:
        return (
            "RejectionRegistry("
            f"name='{self.name}', "
            f"records={len(self.records)}, "
            f"reasons={len(self.reason_counts)}"
            ")"
        )