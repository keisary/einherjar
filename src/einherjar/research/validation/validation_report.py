"""
==========================================================
Validation Report
==========================================================

Rapport cumulatif de la phase Validation.

Ce module agrège :
- les candidats évalués,
- les validations acceptées,
- les rejets,
- les scores,
- les répartitions par split,
- les meilleures validations.

Il ne valide rien à lui seul.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from typing import Iterable
from typing import Mapping

from .rejection import RejectionRegistry
from .rejection import ValidationRejection

__all__ = [
    "ValidationReport",
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


def _assessment_get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _assessment_to_dict(obj: Any) -> dict[str, Any]:
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return obj.to_dict()
    if isinstance(obj, Mapping):
        return dict(obj)
    return {"value": repr(obj)}


def _assessment_score(obj: Any) -> float:
    return _coerce_float(_assessment_get(obj, "score", 0.0), 0.0)


def _assessment_passed(obj: Any) -> bool:
    return _coerce_bool(_assessment_get(obj, "passed", False), False)


def _assessment_candidate_fingerprint(obj: Any) -> str:
    return str(_assessment_get(obj, "candidate_fingerprint", "") or "")


def _assessment_hypothesis_fingerprint(obj: Any) -> str:
    return str(_assessment_get(obj, "hypothesis_fingerprint", "") or "")


def _assessment_split_name(obj: Any) -> str:
    return str(_assessment_get(obj, "split_name", "validation") or "validation").strip() or "validation"


def _assessment_rejection_reasons(obj: Any) -> tuple[str, ...]:
    reasons = _assessment_get(obj, "rejection_reasons", ())
    if reasons is None:
        return ()
    if isinstance(reasons, (str, bytes)):
        return (str(reasons),)
    return tuple(str(item) for item in reasons)


@dataclass(frozen=True, slots=True)
class ValidationSummary:
    """
    Résumé compact de la validation.
    """

    evaluated: int
    passed: int
    rejected: int
    duplicates: int

    average_score: float
    best_score: float
    best_candidate_fingerprint: str | None = None
    best_hypothesis_fingerprint: str | None = None

    pass_rate: float = 0.0
    rejection_rate: float = 0.0
    duplicate_rate: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "evaluated", max(0, _coerce_int(self.evaluated, 0)))
        object.__setattr__(self, "passed", max(0, _coerce_int(self.passed, 0)))
        object.__setattr__(self, "rejected", max(0, _coerce_int(self.rejected, 0)))
        object.__setattr__(self, "duplicates", max(0, _coerce_int(self.duplicates, 0)))
        object.__setattr__(self, "average_score", float(self.average_score))
        object.__setattr__(self, "best_score", float(self.best_score))
        object.__setattr__(self, "pass_rate", min(1.0, max(0.0, float(self.pass_rate))))
        object.__setattr__(self, "rejection_rate", min(1.0, max(0.0, float(self.rejection_rate))))
        object.__setattr__(self, "duplicate_rate", min(1.0, max(0.0, float(self.duplicate_rate))))

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluated": self.evaluated,
            "passed": self.passed,
            "rejected": self.rejected,
            "duplicates": self.duplicates,
            "average_score": self.average_score,
            "best_score": self.best_score,
            "best_candidate_fingerprint": self.best_candidate_fingerprint,
            "best_hypothesis_fingerprint": self.best_hypothesis_fingerprint,
            "pass_rate": self.pass_rate,
            "rejection_rate": self.rejection_rate,
            "duplicate_rate": self.duplicate_rate,
        }


@dataclass(slots=True)
class ValidationReport:
    """
    Rapport cumulatif de la phase Validation.

    Le rapport agrège les assessments produits par
    ValidationEvaluator sans intervenir dans les règles
    scientifiques de décision.
    """

    name: str = "validation"
    metadata: dict[str, Any] = field(default_factory=dict)

    started_at: datetime | None = None
    finished_at: datetime | None = None

    assessments: list[Any] = field(default_factory=list)
    rejections: RejectionRegistry = field(default_factory=RejectionRegistry)

    total_evaluated: int = 0
    total_passed: int = 0
    total_rejected: int = 0
    total_duplicates: int = 0

    split_counts: Counter[str] = field(default_factory=Counter)
    pass_counts: Counter[str] = field(default_factory=Counter)
    reject_counts: Counter[str] = field(default_factory=Counter)

    score_sum: float = 0.0
    best_score: float = float("-inf")
    best_assessment_index: int | None = None
    best_candidate_fingerprint: str | None = None
    best_hypothesis_fingerprint: str | None = None

    last_reason: str | None = None
    exhausted: bool = False
    stopped_reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", str(self.name).strip() or "validation")
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "assessments", list(self.assessments))
        if not isinstance(self.rejections, RejectionRegistry):
            object.__setattr__(self, "rejections", RejectionRegistry.from_dict(_to_mapping(self.rejections)))

        object.__setattr__(self, "split_counts", Counter(self.split_counts))
        object.__setattr__(self, "pass_counts", Counter(self.pass_counts))
        object.__setattr__(self, "reject_counts", Counter(self.reject_counts))

    # ==================================================
    # LIFECYCLE
    # ==================================================

    def start(self) -> None:
        if self.started_at is None:
            self.started_at = _utc_now()
        self.finished_at = None
        self.exhausted = False
        self.stopped_reason = None

    def finish(self, reason: str | None = None) -> None:
        if self.started_at is None:
            self.start()

        self.finished_at = _utc_now()
        self.exhausted = True
        self.stopped_reason = reason or self.stopped_reason or "finished"

    def reset(self) -> None:
        self.started_at = None
        self.finished_at = None

        self.assessments.clear()
        self.rejections.clear()

        self.total_evaluated = 0
        self.total_passed = 0
        self.total_rejected = 0
        self.total_duplicates = 0

        self.split_counts.clear()
        self.pass_counts.clear()
        self.reject_counts.clear()

        self.score_sum = 0.0
        self.best_score = float("-inf")
        self.best_assessment_index = None
        self.best_candidate_fingerprint = None
        self.best_hypothesis_fingerprint = None

        self.last_reason = None
        self.exhausted = False
        self.stopped_reason = None

    # ==================================================
    # PROPERTIES
    # ==================================================

    @property
    def assessment_count(self) -> int:
        return len(self.assessments)

    @property
    def duration_seconds(self) -> float:
        if self.started_at is None:
            return 0.0

        end = self.finished_at or _utc_now()
        return max(0.0, (end - self.started_at).total_seconds())

    @property
    def is_running(self) -> bool:
        return self.started_at is not None and self.finished_at is None

    @property
    def pass_rate(self) -> float:
        if self.total_evaluated == 0:
            return 0.0
        return self.total_passed / self.total_evaluated

    @property
    def rejection_rate(self) -> float:
        if self.total_evaluated == 0:
            return 0.0
        return self.total_rejected / self.total_evaluated

    @property
    def duplicate_rate(self) -> float:
        if self.total_evaluated == 0:
            return 0.0
        return self.total_duplicates / self.total_evaluated

    @property
    def average_score(self) -> float:
        if self.total_evaluated == 0:
            return 0.0
        return self.score_sum / self.total_evaluated

    @property
    def summary(self) -> ValidationSummary:
        return ValidationSummary(
            evaluated=self.total_evaluated,
            passed=self.total_passed,
            rejected=self.total_rejected,
            duplicates=self.total_duplicates,
            average_score=self.average_score,
            best_score=self.best_score if self.best_score != float("-inf") else 0.0,
            best_candidate_fingerprint=self.best_candidate_fingerprint,
            best_hypothesis_fingerprint=self.best_hypothesis_fingerprint,
            pass_rate=self.pass_rate,
            rejection_rate=self.rejection_rate,
            duplicate_rate=self.duplicate_rate,
        )

    # ==================================================
    # RECORDING
    # ==================================================

    def record_assessment(self, assessment: Any) -> Any:
        self.assessments.append(assessment)

        split_name = _assessment_split_name(assessment)
        score = _assessment_score(assessment)
        passed = _assessment_passed(assessment)
        reasons = _assessment_rejection_reasons(assessment)

        self.total_evaluated += 1
        self.split_counts[split_name] += 1
        self.score_sum += score

        if passed:
            self.total_passed += 1
            self.pass_counts[split_name] += 1
        else:
            self.total_rejected += 1
            self.reject_counts[split_name] += 1

        if _coerce_bool(_assessment_get(assessment, "duplicate", False), False):
            self.total_duplicates += 1

        if reasons:
            self.last_reason = reasons[0]
            for reason in reasons:
                self.rejections.reason_counts[reason] += 1

        candidate_fp = _assessment_candidate_fingerprint(assessment)
        hyp_fp = _assessment_hypothesis_fingerprint(assessment)

        if score > self.best_score:
            self.best_score = score
            self.best_assessment_index = self.total_evaluated - 1
            self.best_candidate_fingerprint = candidate_fp or self.best_candidate_fingerprint
            self.best_hypothesis_fingerprint = hyp_fp or self.best_hypothesis_fingerprint

        return assessment

    def record_rejection(self, rejection: ValidationRejection) -> ValidationRejection:
        self.rejections.add(rejection)
        self.total_rejected += 1
        self.total_evaluated += 1
        self.reject_counts[rejection.split_name] += 1
        self.score_sum += float(rejection.score)
        self.last_reason = rejection.primary_reason
        if rejection.duplicate:
            self.total_duplicates += 1
        return rejection

    def absorb(self, items: Iterable[Any]) -> None:
        for item in items:
            if hasattr(item, "rejection_reasons"):
                self.record_assessment(item)
            elif isinstance(item, ValidationRejection):
                self.record_rejection(item)

    # ==================================================
    # SERIALIZATION
    # ==================================================

    def to_dict(self, *, summary_only: bool = False) -> dict[str, Any]:
        base = {
            "name": self.name,
            "metadata": dict(self.metadata),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "total_evaluated": self.total_evaluated,
            "total_passed": self.total_passed,
            "total_rejected": self.total_rejected,
            "total_duplicates": self.total_duplicates,
            "split_counts": dict(self.split_counts),
            "pass_counts": dict(self.pass_counts),
            "reject_counts": dict(self.reject_counts),
            "score_sum": self.score_sum,
            "average_score": self.average_score,
            "best_score": None if self.best_score == float("-inf") else self.best_score,
            "best_assessment_index": self.best_assessment_index,
            "best_candidate_fingerprint": self.best_candidate_fingerprint,
            "best_hypothesis_fingerprint": self.best_hypothesis_fingerprint,
            "last_reason": self.last_reason,
            "exhausted": self.exhausted,
            "stopped_reason": self.stopped_reason,
            "duration_seconds": self.duration_seconds,
            "pass_rate": self.pass_rate,
            "rejection_rate": self.rejection_rate,
            "duplicate_rate": self.duplicate_rate,
            "summary": self.summary.to_dict(),
            "rejections": self.rejections.to_dict(),
        }

        if summary_only:
            return base

        base["assessments"] = [
            _assessment_to_dict(assessment) for assessment in self.assessments
        ]
        return base

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ValidationReport":
        report = cls(
            name=data.get("name", "validation"),
            metadata=_to_mapping(data.get("metadata", {})),
            started_at=(
                datetime.fromisoformat(data["started_at"])
                if data.get("started_at")
                else None
            ),
            finished_at=(
                datetime.fromisoformat(data["finished_at"])
                if data.get("finished_at")
                else None
            ),
            total_evaluated=_coerce_int(data.get("total_evaluated"), 0),
            total_passed=_coerce_int(data.get("total_passed"), 0),
            total_rejected=_coerce_int(data.get("total_rejected"), 0),
            total_duplicates=_coerce_int(data.get("total_duplicates"), 0),
            split_counts=Counter(_to_mapping(data.get("split_counts", {}))),
            pass_counts=Counter(_to_mapping(data.get("pass_counts", {}))),
            reject_counts=Counter(_to_mapping(data.get("reject_counts", {}))),
            score_sum=_coerce_float(data.get("score_sum"), 0.0),
            best_score=_coerce_float(data.get("best_score"), float("-inf")),
            best_assessment_index=data.get("best_assessment_index"),
            best_candidate_fingerprint=data.get("best_candidate_fingerprint"),
            best_hypothesis_fingerprint=data.get("best_hypothesis_fingerprint"),
            last_reason=data.get("last_reason"),
            exhausted=_coerce_bool(data.get("exhausted"), False),
            stopped_reason=data.get("stopped_reason"),
            rejections=RejectionRegistry.from_dict(_to_mapping(data.get("rejections", {}))),
        )

        report.assessments = data.get("assessments", [])
        return report

    # ==================================================
    # PYTHON PROTOCOL
    # ==================================================

    def __len__(self) -> int:
        return len(self.assessments)

    def __iter__(self):
        return iter(self.assessments)

    def __bool__(self) -> bool:
        return bool(self.assessments)

    def __repr__(self) -> str:
        return (
            "ValidationReport("
            f"name='{self.name}', "
            f"evaluated={self.total_evaluated}, "
            f"passed={self.total_passed}, "
            f"best_score={None if self.best_score == float('-inf') else round(self.best_score, 4)}"
            ")"
        )