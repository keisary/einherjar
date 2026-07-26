# memory/feature_history.py
"""
==========================================================
Feature History
==========================================================

Historique d'usage des features par le moteur.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "FeatureUsage",
    "FeatureHistorySummary",
    "FeatureHistory",
    "FeatureHistoryBuilder",
]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str) and value:
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        return datetime.fromisoformat(text)
    return None


def _to_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "to_dict") and callable(value.to_dict):
        value = value.to_dict()
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _normalize_text(value: Any, default: str = "") -> str:
    text = str(value or "").strip()
    return text if text else default


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True, slots=True)
class FeatureUsage:
    feature_key: str
    family: str = "unknown"
    phase: str = "unknown"
    usage_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    mean_score: float = 0.0
    best_score: float = 0.0
    first_seen_at: datetime = field(default_factory=_utc_now)
    last_seen_at: datetime = field(default_factory=_utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "feature_key", _normalize_text(self.feature_key))
        object.__setattr__(self, "family", _normalize_text(self.family, "unknown"))
        object.__setattr__(self, "phase", _normalize_text(self.phase, "unknown"))
        object.__setattr__(self, "usage_count", max(0, _coerce_int(self.usage_count)))
        object.__setattr__(self, "success_count", max(0, _coerce_int(self.success_count)))
        object.__setattr__(self, "failure_count", max(0, _coerce_int(self.failure_count)))
        object.__setattr__(self, "mean_score", _coerce_float(self.mean_score))
        object.__setattr__(self, "best_score", _coerce_float(self.best_score))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def success_rate(self) -> float:
        if self.usage_count <= 0:
            return 0.0
        return self.success_count / self.usage_count

    def touch(self, *, success: bool = False, score: float = 0.0, metadata: Mapping[str, Any] | None = None) -> "FeatureUsage":
        count = self.usage_count + 1
        success_count = self.success_count + (1 if success else 0)
        failure_count = self.failure_count + (0 if success else 1)
        mean_score = ((self.mean_score * self.usage_count) + _coerce_float(score)) / max(1, count)
        best_score = max(self.best_score, _coerce_float(score))
        return FeatureUsage(
            feature_key=self.feature_key,
            family=self.family,
            phase=self.phase,
            usage_count=count,
            success_count=success_count,
            failure_count=failure_count,
            mean_score=mean_score,
            best_score=best_score,
            first_seen_at=self.first_seen_at,
            last_seen_at=_utc_now(),
            metadata={**self.metadata, **_to_mapping(metadata or {})},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_key": self.feature_key,
            "family": self.family,
            "phase": self.phase,
            "usage_count": self.usage_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "mean_score": self.mean_score,
            "best_score": self.best_score,
            "success_rate": self.success_rate,
            "first_seen_at": self.first_seen_at.isoformat(),
            "last_seen_at": self.last_seen_at.isoformat(),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FeatureUsage":
        return cls(
            feature_key=data.get("feature_key", ""),
            family=data.get("family", "unknown"),
            phase=data.get("phase", "unknown"),
            usage_count=data.get("usage_count", 0),
            success_count=data.get("success_count", 0),
            failure_count=data.get("failure_count", 0),
            mean_score=data.get("mean_score", 0.0),
            best_score=data.get("best_score", 0.0),
            first_seen_at=_parse_datetime(data.get("first_seen_at")) or _utc_now(),
            last_seen_at=_parse_datetime(data.get("last_seen_at")) or _utc_now(),
            metadata=_to_mapping(data.get("metadata", {})),
        )


@dataclass(frozen=True, slots=True)
class FeatureHistorySummary:
    entry_count: int
    feature_count: int
    total_usage: int
    total_success: int
    total_failure: int
    average_success_rate: float
    phase_counts: dict[str, int] = field(default_factory=dict)
    family_counts: dict[str, int] = field(default_factory=dict)
    top_features: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "entry_count", max(0, _coerce_int(self.entry_count)))
        object.__setattr__(self, "feature_count", max(0, _coerce_int(self.feature_count)))
        object.__setattr__(self, "total_usage", max(0, _coerce_int(self.total_usage)))
        object.__setattr__(self, "total_success", max(0, _coerce_int(self.total_success)))
        object.__setattr__(self, "total_failure", max(0, _coerce_int(self.total_failure)))
        object.__setattr__(self, "average_success_rate", _coerce_float(self.average_success_rate))
        object.__setattr__(self, "phase_counts", dict(self.phase_counts))
        object.__setattr__(self, "family_counts", dict(self.family_counts))
        object.__setattr__(self, "top_features", tuple(self.top_features))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_count": self.entry_count,
            "feature_count": self.feature_count,
            "total_usage": self.total_usage,
            "total_success": self.total_success,
            "total_failure": self.total_failure,
            "average_success_rate": self.average_success_rate,
            "phase_counts": dict(self.phase_counts),
            "family_counts": dict(self.family_counts),
            "top_features": list(self.top_features),
            "metadata": dict(self.metadata),
        }


class FeatureHistory:
    def __init__(self, entries: Iterable[FeatureUsage] | None = None, *, metadata: Mapping[str, Any] | None = None) -> None:
        self.entries: list[FeatureUsage] = list(entries or [])
        self.metadata: dict[str, Any] = dict(metadata or {})

    def add(self, usage: FeatureUsage) -> FeatureUsage:
        self.entries.append(usage)
        return usage

    def register(
        self,
        feature_key: str,
        *,
        family: str = "unknown",
        phase: str = "unknown",
        success: bool = False,
        score: float = 0.0,
        metadata: Mapping[str, Any] | None = None,
    ) -> FeatureUsage:
        usage = FeatureHistoryBuilder.build(
            feature_key=feature_key,
            family=family,
            phase=phase,
            success=success,
            score=score,
            metadata=metadata,
        )
        return self.add(usage)

    def touch(self, feature_key: str, *, success: bool = False, score: float = 0.0, metadata: Mapping[str, Any] | None = None) -> FeatureUsage | None:
        feature_key = _normalize_text(feature_key)
        for index, usage in enumerate(self.entries):
            if usage.feature_key == feature_key:
                updated = usage.touch(success=success, score=score, metadata=metadata)
                self.entries[index] = updated
                return updated
        return None

    def by_feature(self, feature_key: str) -> tuple[FeatureUsage, ...]:
        feature_key = _normalize_text(feature_key)
        return tuple(entry for entry in self.entries if entry.feature_key == feature_key)

    def by_family(self, family: str) -> tuple[FeatureUsage, ...]:
        family = _normalize_text(family, "unknown")
        return tuple(entry for entry in self.entries if entry.family == family)

    def by_phase(self, phase: str) -> tuple[FeatureUsage, ...]:
        phase = _normalize_text(phase, "unknown")
        return tuple(entry for entry in self.entries if entry.phase == phase)

    @property
    def summary(self) -> FeatureHistorySummary:
        if not self.entries:
            return FeatureHistorySummary(0, 0, 0, 0, 0, 0.0, metadata=dict(self.metadata))
        phase_counts = Counter(entry.phase for entry in self.entries)
        family_counts = Counter(entry.family for entry in self.entries)
        top_features = tuple(feature for feature, _ in Counter(entry.feature_key for entry in self.entries).most_common(10))
        return FeatureHistorySummary(
            entry_count=len(self.entries),
            feature_count=len({entry.feature_key for entry in self.entries}),
            total_usage=sum(entry.usage_count for entry in self.entries),
            total_success=sum(entry.success_count for entry in self.entries),
            total_failure=sum(entry.failure_count for entry in self.entries),
            average_success_rate=sum(entry.success_rate for entry in self.entries) / len(self.entries),
            phase_counts=dict(phase_counts),
            family_counts=dict(family_counts),
            top_features=top_features,
            metadata=dict(self.metadata),
        )

    def to_dict(self, *, summary_only: bool = False) -> dict[str, Any]:
        return {
            "entries": [] if summary_only else [entry.to_dict() for entry in self.entries],
            "summary": self.summary.to_dict(),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FeatureHistory":
        return cls(
            entries=[FeatureUsage.from_dict(item) for item in data.get("entries", [])],
            metadata=_to_mapping(data.get("metadata", {})),
        )

    def __iter__(self):
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)


class FeatureHistoryBuilder:
    @staticmethod
    def build(
        *,
        feature_key: str,
        family: str = "unknown",
        phase: str = "unknown",
        success: bool = False,
        score: float = 0.0,
        metadata: Mapping[str, Any] | None = None,
        usage_count: int = 1,
        success_count: int | None = None,
        failure_count: int | None = None,
        first_seen_at: datetime | None = None,
        last_seen_at: datetime | None = None,
    ) -> FeatureUsage:
        if success_count is None:
            success_count = 1 if success else 0
        if failure_count is None:
            failure_count = 0 if success else 1
        return FeatureUsage(
            feature_key=feature_key,
            family=family,
            phase=phase,
            usage_count=usage_count,
            success_count=success_count,
            failure_count=failure_count,
            mean_score=score,
            best_score=score,
            first_seen_at=first_seen_at or _utc_now(),
            last_seen_at=last_seen_at or _utc_now(),
            metadata=_to_mapping(metadata or {}),
        )