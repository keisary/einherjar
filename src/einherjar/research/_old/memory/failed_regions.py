# memory/failed_regions.py
"""
==========================================================
Failed Regions
==========================================================

Mémoire des régions explorées sans résultat utile.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "FailedRegion",
    "FailedRegionsSummary",
    "FailedRegions",
    "FailedRegionsBuilder",
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
class FailedRegion:
    region_key: str
    phase: str = "unknown"
    family: str = "unknown"
    feature: str = "unknown"
    attempts: int = 0
    failure_count: int = 0
    score: float = 0.0
    reason: str = ""
    first_seen_at: datetime = field(default_factory=_utc_now)
    last_seen_at: datetime = field(default_factory=_utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "region_key", _normalize_text(self.region_key))
        object.__setattr__(self, "phase", _normalize_text(self.phase, "unknown"))
        object.__setattr__(self, "family", _normalize_text(self.family, "unknown"))
        object.__setattr__(self, "feature", _normalize_text(self.feature, "unknown"))
        object.__setattr__(self, "attempts", max(0, _coerce_int(self.attempts)))
        object.__setattr__(self, "failure_count", max(0, _coerce_int(self.failure_count)))
        object.__setattr__(self, "score", _coerce_float(self.score))
        object.__setattr__(self, "reason", _normalize_text(self.reason))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def touch(self, *, failure: bool = True, score: float | None = None, reason: str | None = None, metadata: Mapping[str, Any] | None = None) -> "FailedRegion":
        return FailedRegion(
            region_key=self.region_key,
            phase=self.phase,
            family=self.family,
            feature=self.feature,
            attempts=self.attempts + 1,
            failure_count=self.failure_count + (1 if failure else 0),
            score=self.score if score is None else _coerce_float(score),
            reason=self.reason if reason is None else _normalize_text(reason),
            first_seen_at=self.first_seen_at,
            last_seen_at=_utc_now(),
            metadata={**self.metadata, **_to_mapping(metadata or {})},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "region_key": self.region_key,
            "phase": self.phase,
            "family": self.family,
            "feature": self.feature,
            "attempts": self.attempts,
            "failure_count": self.failure_count,
            "score": self.score,
            "reason": self.reason,
            "first_seen_at": self.first_seen_at.isoformat(),
            "last_seen_at": self.last_seen_at.isoformat(),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FailedRegion":
        return cls(
            region_key=data.get("region_key", ""),
            phase=data.get("phase", "unknown"),
            family=data.get("family", "unknown"),
            feature=data.get("feature", "unknown"),
            attempts=data.get("attempts", 0),
            failure_count=data.get("failure_count", 0),
            score=data.get("score", 0.0),
            reason=data.get("reason", ""),
            first_seen_at=_parse_datetime(data.get("first_seen_at")) or _utc_now(),
            last_seen_at=_parse_datetime(data.get("last_seen_at")) or _utc_now(),
            metadata=_to_mapping(data.get("metadata", {})),
        )


@dataclass(frozen=True, slots=True)
class FailedRegionsSummary:
    entry_count: int
    region_count: int
    total_attempts: int
    total_failure_count: int
    average_score: float
    phase_counts: dict[str, int] = field(default_factory=dict)
    family_counts: dict[str, int] = field(default_factory=dict)
    feature_counts: dict[str, int] = field(default_factory=dict)
    reason_counts: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "entry_count", max(0, _coerce_int(self.entry_count)))
        object.__setattr__(self, "region_count", max(0, _coerce_int(self.region_count)))
        object.__setattr__(self, "total_attempts", max(0, _coerce_int(self.total_attempts)))
        object.__setattr__(self, "total_failure_count", max(0, _coerce_int(self.total_failure_count)))
        object.__setattr__(self, "average_score", _coerce_float(self.average_score))
        object.__setattr__(self, "phase_counts", dict(self.phase_counts))
        object.__setattr__(self, "family_counts", dict(self.family_counts))
        object.__setattr__(self, "feature_counts", dict(self.feature_counts))
        object.__setattr__(self, "reason_counts", dict(self.reason_counts))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_count": self.entry_count,
            "region_count": self.region_count,
            "total_attempts": self.total_attempts,
            "total_failure_count": self.total_failure_count,
            "average_score": self.average_score,
            "phase_counts": dict(self.phase_counts),
            "family_counts": dict(self.family_counts),
            "feature_counts": dict(self.feature_counts),
            "reason_counts": dict(self.reason_counts),
            "metadata": dict(self.metadata),
        }


class FailedRegions:
    def __init__(self, entries: Iterable[FailedRegion] | None = None, *, metadata: Mapping[str, Any] | None = None) -> None:
        self.entries: list[FailedRegion] = list(entries or [])
        self.metadata: dict[str, Any] = dict(metadata or {})

    def add(self, region: FailedRegion) -> FailedRegion:
        self.entries.append(region)
        return region

    def register(
        self,
        region_key: str,
        *,
        phase: str = "unknown",
        family: str = "unknown",
        feature: str = "unknown",
        attempts: int = 1,
        failure_count: int = 1,
        score: float = 0.0,
        reason: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> FailedRegion:
        region = FailedRegionsBuilder.build(
            region_key=region_key,
            phase=phase,
            family=family,
            feature=feature,
            attempts=attempts,
            failure_count=failure_count,
            score=score,
            reason=reason,
            metadata=metadata,
        )
        return self.add(region)

    def touch(self, region_key: str, *, failure: bool = True, score: float | None = None, reason: str | None = None, metadata: Mapping[str, Any] | None = None) -> FailedRegion | None:
        region_key = _normalize_text(region_key)
        for index, region in enumerate(self.entries):
            if region.region_key == region_key:
                updated = region.touch(failure=failure, score=score, reason=reason, metadata=metadata)
                self.entries[index] = updated
                return updated
        return None

    def by_phase(self, phase: str) -> tuple[FailedRegion, ...]:
        phase = _normalize_text(phase, "unknown")
        return tuple(entry for entry in self.entries if entry.phase == phase)

    def by_family(self, family: str) -> tuple[FailedRegion, ...]:
        family = _normalize_text(family, "unknown")
        return tuple(entry for entry in self.entries if entry.family == family)

    def by_feature(self, feature: str) -> tuple[FailedRegion, ...]:
        feature = _normalize_text(feature, "unknown")
        return tuple(entry for entry in self.entries if entry.feature == feature)

    @property
    def summary(self) -> FailedRegionsSummary:
        if not self.entries:
            return FailedRegionsSummary(0, 0, 0, 0, 0.0, metadata=dict(self.metadata))
        phase_counts = Counter(entry.phase for entry in self.entries)
        family_counts = Counter(entry.family for entry in self.entries)
        feature_counts = Counter(entry.feature for entry in self.entries)
        reason_counts = Counter(entry.reason for entry in self.entries if entry.reason)
        return FailedRegionsSummary(
            entry_count=len(self.entries),
            region_count=len({entry.region_key for entry in self.entries}),
            total_attempts=sum(entry.attempts for entry in self.entries),
            total_failure_count=sum(entry.failure_count for entry in self.entries),
            average_score=sum(entry.score for entry in self.entries) / len(self.entries),
            phase_counts=dict(phase_counts),
            family_counts=dict(family_counts),
            feature_counts=dict(feature_counts),
            reason_counts=dict(reason_counts),
            metadata=dict(self.metadata),
        )

    def to_dict(self, *, summary_only: bool = False) -> dict[str, Any]:
        return {
            "entries": [] if summary_only else [entry.to_dict() for entry in self.entries],
            "summary": self.summary.to_dict(),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FailedRegions":
        return cls(
            entries=[FailedRegion.from_dict(item) for item in data.get("entries", [])],
            metadata=_to_mapping(data.get("metadata", {})),
        )

    def __iter__(self):
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)


class FailedRegionsBuilder:
    @staticmethod
    def build(
        *,
        region_key: str,
        phase: str = "unknown",
        family: str = "unknown",
        feature: str = "unknown",
        attempts: int = 1,
        failure_count: int = 1,
        score: float = 0.0,
        reason: str = "",
        metadata: Mapping[str, Any] | None = None,
        first_seen_at: datetime | None = None,
        last_seen_at: datetime | None = None,
    ) -> FailedRegion:
        return FailedRegion(
            region_key=region_key,
            phase=phase,
            family=family,
            feature=feature,
            attempts=attempts,
            failure_count=failure_count,
            score=score,
            reason=reason,
            first_seen_at=first_seen_at or _utc_now(),
            last_seen_at=last_seen_at or _utc_now(),
            metadata=_to_mapping(metadata or {}),
        )