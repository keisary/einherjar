# memory/family_history.py
"""
==========================================================
Family History
==========================================================

Historique des familles explorées et de leur rendement.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "FamilyUsage",
    "FamilyHistorySummary",
    "FamilyHistory",
    "FamilyHistoryBuilder",
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
class FamilyUsage:
    family_key: str
    usage_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    mean_score: float = 0.0
    best_score: float = 0.0
    saturation: float = 0.0
    first_seen_at: datetime = field(default_factory=_utc_now)
    last_seen_at: datetime = field(default_factory=_utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "family_key", _normalize_text(self.family_key, "unknown"))
        object.__setattr__(self, "usage_count", max(0, _coerce_int(self.usage_count)))
        object.__setattr__(self, "success_count", max(0, _coerce_int(self.success_count)))
        object.__setattr__(self, "failure_count", max(0, _coerce_int(self.failure_count)))
        object.__setattr__(self, "mean_score", _coerce_float(self.mean_score))
        object.__setattr__(self, "best_score", _coerce_float(self.best_score))
        object.__setattr__(self, "saturation", _coerce_float(self.saturation))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def success_rate(self) -> float:
        if self.usage_count <= 0:
            return 0.0
        return self.success_count / self.usage_count

    def touch(self, *, success: bool = False, score: float = 0.0, metadata: Mapping[str, Any] | None = None) -> "FamilyUsage":
        count = self.usage_count + 1
        success_count = self.success_count + (1 if success else 0)
        failure_count = self.failure_count + (0 if success else 1)
        mean_score = ((self.mean_score * self.usage_count) + _coerce_float(score)) / max(1, count)
        best_score = max(self.best_score, _coerce_float(score))
        saturation = min(1.0, count / max(1, count + 4))
        return FamilyUsage(
            family_key=self.family_key,
            usage_count=count,
            success_count=success_count,
            failure_count=failure_count,
            mean_score=mean_score,
            best_score=best_score,
            saturation=saturation,
            first_seen_at=self.first_seen_at,
            last_seen_at=_utc_now(),
            metadata={**self.metadata, **_to_mapping(metadata or {})},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "family_key": self.family_key,
            "usage_count": self.usage_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "mean_score": self.mean_score,
            "best_score": self.best_score,
            "saturation": self.saturation,
            "success_rate": self.success_rate,
            "first_seen_at": self.first_seen_at.isoformat(),
            "last_seen_at": self.last_seen_at.isoformat(),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FamilyUsage":
        return cls(
            family_key=data.get("family_key", "unknown"),
            usage_count=data.get("usage_count", 0),
            success_count=data.get("success_count", 0),
            failure_count=data.get("failure_count", 0),
            mean_score=data.get("mean_score", 0.0),
            best_score=data.get("best_score", 0.0),
            saturation=data.get("saturation", 0.0),
            first_seen_at=_parse_datetime(data.get("first_seen_at")) or _utc_now(),
            last_seen_at=_parse_datetime(data.get("last_seen_at")) or _utc_now(),
            metadata=_to_mapping(data.get("metadata", {})),
        )


@dataclass(frozen=True, slots=True)
class FamilyHistorySummary:
    entry_count: int
    family_count: int
    total_usage: int
    total_success: int
    total_failure: int
    average_success_rate: float
    dominant_families: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "entry_count", max(0, _coerce_int(self.entry_count)))
        object.__setattr__(self, "family_count", max(0, _coerce_int(self.family_count)))
        object.__setattr__(self, "total_usage", max(0, _coerce_int(self.total_usage)))
        object.__setattr__(self, "total_success", max(0, _coerce_int(self.total_success)))
        object.__setattr__(self, "total_failure", max(0, _coerce_int(self.total_failure)))
        object.__setattr__(self, "average_success_rate", _coerce_float(self.average_success_rate))
        object.__setattr__(self, "dominant_families", tuple(self.dominant_families))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_count": self.entry_count,
            "family_count": self.family_count,
            "total_usage": self.total_usage,
            "total_success": self.total_success,
            "total_failure": self.total_failure,
            "average_success_rate": self.average_success_rate,
            "dominant_families": list(self.dominant_families),
            "metadata": dict(self.metadata),
        }


class FamilyHistory:
    def __init__(self, entries: Iterable[FamilyUsage] | None = None, *, metadata: Mapping[str, Any] | None = None) -> None:
        self.entries: list[FamilyUsage] = list(entries or [])
        self.metadata: dict[str, Any] = dict(metadata or {})

    def add(self, usage: FamilyUsage) -> FamilyUsage:
        self.entries.append(usage)
        return usage

    def register(
        self,
        family_key: str,
        *,
        success: bool = False,
        score: float = 0.0,
        metadata: Mapping[str, Any] | None = None,
    ) -> FamilyUsage:
        usage = FamilyHistoryBuilder.build(
            family_key=family_key,
            success=success,
            score=score,
            metadata=metadata,
        )
        return self.add(usage)

    def touch(self, family_key: str, *, success: bool = False, score: float = 0.0, metadata: Mapping[str, Any] | None = None) -> FamilyUsage | None:
        family_key = _normalize_text(family_key, "unknown")
        for index, usage in enumerate(self.entries):
            if usage.family_key == family_key:
                updated = usage.touch(success=success, score=score, metadata=metadata)
                self.entries[index] = updated
                return updated
        return None

    def by_family(self, family_key: str) -> tuple[FamilyUsage, ...]:
        family_key = _normalize_text(family_key, "unknown")
        return tuple(entry for entry in self.entries if entry.family_key == family_key)

    @property
    def summary(self) -> FamilyHistorySummary:
        if not self.entries:
            return FamilyHistorySummary(0, 0, 0, 0, 0, 0.0, metadata=dict(self.metadata))
        dominant_families = tuple(family for family, _ in Counter(entry.family_key for entry in self.entries).most_common(10))
        return FamilyHistorySummary(
            entry_count=len(self.entries),
            family_count=len({entry.family_key for entry in self.entries}),
            total_usage=sum(entry.usage_count for entry in self.entries),
            total_success=sum(entry.success_count for entry in self.entries),
            total_failure=sum(entry.failure_count for entry in self.entries),
            average_success_rate=sum(entry.success_rate for entry in self.entries) / len(self.entries),
            dominant_families=dominant_families,
            metadata=dict(self.metadata),
        )

    def to_dict(self, *, summary_only: bool = False) -> dict[str, Any]:
        return {
            "entries": [] if summary_only else [entry.to_dict() for entry in self.entries],
            "summary": self.summary.to_dict(),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FamilyHistory":
        return cls(
            entries=[FamilyUsage.from_dict(item) for item in data.get("entries", [])],
            metadata=_to_mapping(data.get("metadata", {})),
        )

    def __iter__(self):
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)


class FamilyHistoryBuilder:
    @staticmethod
    def build(
        *,
        family_key: str,
        success: bool = False,
        score: float = 0.0,
        metadata: Mapping[str, Any] | None = None,
        usage_count: int = 1,
        success_count: int | None = None,
        failure_count: int | None = None,
        first_seen_at: datetime | None = None,
        last_seen_at: datetime | None = None,
    ) -> FamilyUsage:
        if success_count is None:
            success_count = 1 if success else 0
        if failure_count is None:
            failure_count = 0 if success else 1
        return FamilyUsage(
            family_key=family_key,
            usage_count=usage_count,
            success_count=success_count,
            failure_count=failure_count,
            mean_score=score,
            best_score=score,
            saturation=min(1.0, usage_count / max(1, usage_count + 4)),
            first_seen_at=first_seen_at or _utc_now(),
            last_seen_at=last_seen_at or _utc_now(),
            metadata=_to_mapping(metadata or {}),
        )