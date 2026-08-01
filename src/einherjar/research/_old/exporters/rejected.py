# exporters/rejected.py
"""
==========================================================
Rejected Export
==========================================================

Export des éléments rejetés pendant le pipeline.

Ce module ne décide rien :
- il normalise les rejets,
- il les regroupe,
- il les prépare pour les exports.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

from portfolio.allocator import PortfolioAllocation
from portfolio.allocator import PortfolioAllocationEntry
from portfolio.portfolio_report import PortfolioReport
from portfolio.portfolio_report import PortfolioReportEntry
from portfolio.selector import PortfolioSelection
from portfolio.selector import PortfolioSelectionEntry

__all__ = [
    "RejectedEntry",
    "RejectedSummary",
    "RejectedCorpus",
    "RejectedBuilder",
]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _coerce_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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


def _to_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "to_dict") and callable(value.to_dict):
        value = value.to_dict()
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _family_key(entry: Any) -> str:
    family = getattr(entry, "family", None)
    if family:
        return str(family).strip().lower()
    result = getattr(entry, "result", None)
    if result is not None:
        metadata = _to_mapping(getattr(result, "metadata", None))
        for key in ("family", "target_family", "portfolio_family"):
            if key in metadata and metadata[key] is not None:
                value = str(metadata[key]).strip().lower()
                if value:
                    return value
    return "unknown"


def _profile_name(entry: Any) -> str:
    profile = getattr(entry, "profile_name", None)
    if profile:
        return str(profile).strip().lower()

    result = getattr(entry, "result", None)
    if result is not None:
        profile_obj = getattr(result, "profile", None)
        if profile_obj is not None and getattr(profile_obj, "name", None):
            value = str(profile_obj.name).strip().lower()
            if value:
                return value
    return "unknown"


def _subject_fingerprint(entry: Any) -> str:
    if getattr(entry, "subject_fingerprint", None):
        return str(entry.subject_fingerprint)

    result = getattr(entry, "result", None)
    if result is not None:
        value = getattr(result, "subject_fingerprint", None)
        if value:
            return str(value)
        fp = getattr(result, "execution_fingerprint", None)
        if fp is not None:
            digest = getattr(fp, "digest", None)
            if digest:
                return str(digest)
    return ""


def _collect_reasons(entry: Any) -> tuple[str, ...]:
    reasons = getattr(entry, "reasons", ())
    if reasons is None:
        return ()
    if isinstance(reasons, (str, bytes)):
        return (str(reasons),)
    return tuple(str(reason) for reason in reasons)


@dataclass(frozen=True, slots=True)
class RejectedEntry:
    """
    Entrée rejetée.
    """

    subject_fingerprint: str
    family: str = "unknown"
    profile_name: str = "unknown"

    source_stage: str = "portfolio"
    reasons: tuple[str, ...] = ()
    severity: str = "warning"

    score: float = 0.0
    weight: float = 0.0
    capital: float = 0.0

    details: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        object.__setattr__(self, "subject_fingerprint", str(self.subject_fingerprint))
        object.__setattr__(self, "family", str(self.family).strip().lower() or "unknown")
        object.__setattr__(self, "profile_name", str(self.profile_name).strip().lower() or "unknown")
        object.__setattr__(self, "source_stage", str(self.source_stage).strip().lower() or "portfolio")
        object.__setattr__(self, "reasons", tuple(str(reason) for reason in self.reasons))
        object.__setattr__(self, "severity", str(self.severity).strip().lower() or "warning")
        object.__setattr__(self, "score", float(self.score))
        object.__setattr__(self, "weight", max(0.0, float(self.weight)))
        object.__setattr__(self, "capital", max(0.0, float(self.capital)))
        object.__setattr__(self, "details", dict(self.details))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject_fingerprint": self.subject_fingerprint,
            "family": self.family,
            "profile_name": self.profile_name,
            "source_stage": self.source_stage,
            "reasons": list(self.reasons),
            "severity": self.severity,
            "score": self.score,
            "weight": self.weight,
            "capital": self.capital,
            "details": dict(self.details),
            "metadata": dict(self.metadata),
            "created_at": self.created_at.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class RejectedSummary:
    """
    Résumé des rejets.
    """

    entry_count: int
    reason_counts: dict[str, int] = field(default_factory=dict)
    stage_counts: dict[str, int] = field(default_factory=dict)
    family_counts: dict[str, int] = field(default_factory=dict)
    profile_counts: dict[str, int] = field(default_factory=dict)

    severe_count: int = 0
    warning_count: int = 0
    info_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "entry_count", max(0, _coerce_int(self.entry_count, 0)))
        object.__setattr__(self, "reason_counts", dict(self.reason_counts))
        object.__setattr__(self, "stage_counts", dict(self.stage_counts))
        object.__setattr__(self, "family_counts", dict(self.family_counts))
        object.__setattr__(self, "profile_counts", dict(self.profile_counts))
        object.__setattr__(self, "severe_count", max(0, _coerce_int(self.severe_count, 0)))
        object.__setattr__(self, "warning_count", max(0, _coerce_int(self.warning_count, 0)))
        object.__setattr__(self, "info_count", max(0, _coerce_int(self.info_count, 0)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_count": self.entry_count,
            "reason_counts": dict(self.reason_counts),
            "stage_counts": dict(self.stage_counts),
            "family_counts": dict(self.family_counts),
            "profile_counts": dict(self.profile_counts),
            "severe_count": self.severe_count,
            "warning_count": self.warning_count,
            "info_count": self.info_count,
        }


@dataclass(slots=True)
class RejectedCorpus:
    """
    Conteneur des rejets.
    """

    name: str = "rejected"
    entries: list[RejectedEntry] = field(default_factory=list)
    summary: RejectedSummary | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", str(self.name).strip() or "rejected")
        object.__setattr__(self, "entries", list(self.entries))
        object.__setattr__(self, "metadata", dict(self.metadata))
        if self.summary is None:
            object.__setattr__(self, "summary", RejectedSummaryBuilder.build(self.entries))

    def to_dict(self, *, summary_only: bool = False) -> dict[str, Any]:
        return {
            "name": self.name,
            "created_at": self.created_at.isoformat(),
            "entries": [] if summary_only else [entry.to_dict() for entry in self.entries],
            "summary": self.summary.to_dict() if self.summary is not None else None,
            "metadata": dict(self.metadata),
        }

    def to_records(self) -> list[dict[str, Any]]:
        return [entry.to_dict() for entry in self.entries]

    def __iter__(self):
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)


class RejectedSummaryBuilder:
    """
    Construit un résumé des rejets.
    """

    @staticmethod
    def build(entries: Sequence[RejectedEntry]) -> RejectedSummary:
        entries = tuple(entries)
        if not entries:
            return RejectedSummary(entry_count=0)

        reason_counts = Counter()
        stage_counts = Counter()
        family_counts = Counter()
        profile_counts = Counter()
        severe_count = warning_count = info_count = 0

        for entry in entries:
            reason_counts.update(entry.reasons)
            stage_counts[entry.source_stage] += 1
            family_counts[entry.family] += 1
            profile_counts[entry.profile_name] += 1
            if entry.severity == "error":
                severe_count += 1
            elif entry.severity == "warning":
                warning_count += 1
            else:
                info_count += 1

        return RejectedSummary(
            entry_count=len(entries),
            reason_counts=dict(reason_counts),
            stage_counts=dict(stage_counts),
            family_counts=dict(family_counts),
            profile_counts=dict(profile_counts),
            severe_count=severe_count,
            warning_count=warning_count,
            info_count=info_count,
        )


class RejectedBuilder:
    """
    Construit les rejets à partir des sorties des modules.
    """

    @staticmethod
    def from_selection(selection: PortfolioSelection, *, metadata: Mapping[str, Any] | None = None) -> RejectedCorpus:
        entries = [
            RejectedEntry(
                subject_fingerprint=item.subject_fingerprint,
                family=item.family,
                profile_name=item.profile_name,
                source_stage="portfolio_selection",
                reasons=item.reasons,
                severity="warning",
                score=_coerce_float(item.score, 0.0),
                weight=0.0,
                capital=0.0,
                details=item.to_dict(),
                metadata=dict(metadata or {}),
            )
            for item in selection.rejected
        ]
        return RejectedCorpus(name="rejected", entries=entries, metadata=dict(metadata or {}))

    @staticmethod
    def from_report(report: PortfolioReport, *, metadata: Mapping[str, Any] | None = None) -> RejectedCorpus:
        entries: list[RejectedEntry] = []

        for item in report.rejected:
            entries.append(
                RejectedEntry(
                    subject_fingerprint=str(item.get("subject_fingerprint", "")),
                    family=str(item.get("family", "unknown")),
                    profile_name=str(item.get("profile_name", "unknown")),
                    source_stage=str(item.get("stage", "portfolio")),
                    reasons=tuple(item.get("reasons", item.get("reason", ()) if isinstance(item.get("reason"), (list, tuple)) else (item.get("reason"),))),
                    severity=str(item.get("severity", "warning")),
                    score=_coerce_float(item.get("score"), 0.0),
                    weight=_coerce_float(item.get("weight"), 0.0),
                    capital=_coerce_float(item.get("capital"), 0.0),
                    details=_to_mapping(item.get("details", item)),
                    metadata=dict(metadata or {}),
                )
            )

        return RejectedCorpus(name="rejected", entries=entries, metadata=dict(metadata or {}))

    @staticmethod
    def from_allocation(allocation: PortfolioAllocation, *, metadata: Mapping[str, Any] | None = None) -> RejectedCorpus:
        entries: list[RejectedEntry] = []
        for item in allocation.entries:
            if item.accepted and item.capital > 0:
                continue
            entries.append(
                RejectedEntry(
                    subject_fingerprint=item.subject_fingerprint,
                    family=item.family,
                    profile_name=item.profile_name,
                    source_stage="portfolio_allocation",
                    reasons=("not_allocated",),
                    severity="warning",
                    score=_coerce_float(item.score, 0.0),
                    weight=_coerce_float(item.target_weight, 0.0),
                    capital=_coerce_float(item.capital, 0.0),
                    details=item.to_dict(),
                    metadata=dict(metadata or {}),
                )
            )
        return RejectedCorpus(name="rejected", entries=entries, metadata=dict(metadata or {}))