# memory/successful_regions.py
"""
==========================================================
Successful Regions
==========================================================

Mémoire des régions qui ont produit des résultats utiles.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "SuccessfulRegion",
    "SuccessfulRegionsSummary",
    "SuccessfulRegions",
    "SuccessfulRegionsBuilder",
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
class SuccessfulRegion:
    region_key: str
    phase: str = "unknown"
    family: str = "unknown"
    feature: str = "unknown"
    hits: int = 0
    success_count: int = 0
    score: float = 0.0
    yield_rate: float = 0.0
    first_seen_at: datetime = field(default_factory=_utc_now)
    last_seen_at: datetime = field(default_factory=_utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "region_key", _normalize_text(self.region_key))
        object.__setattr__(self, "phase", _normalize_text(self.phase, "unknown"))
        object.__setattr__(self, "family", _normalize_text(self.family, "unknown"))
        object.__setattr__(self, "feature", _normalize_text(self.feature, "unknown"))
        object.__setattr__(self, "hits", max(0, _coerce_int(self.hits)))
        object.__setattr__(self, "success_count", max(0, _coerce_int(self.success_count)))
        object.__setattr__(self, "score", _coerce_float(self.score))
        object.__setattr__(self, "yield_rate", _coerce_float(self.yield_rate))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def touch(self, *, success: bool = True, score: float | None = None, metadata: Mapping[str, Any] | None = None) -> "SuccessfulRegion":
        return SuccessfulRegion(
            region_key=self.region_key,
            phase=self.phase,
            family=self.family,
            feature=self.feature,
            hits=self.hits + 1,
            success_count=self.success_count + (1 if success else 0),
            score=self.score if score is None else _coerce_float(score),
            yield_rate=(self.success_count + (1 if success else 0)) / max(1, self.hits + 1),
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
            "hits": self.hits,
            "success_count": self.success_count,
            "score": self.score,
            "yield_rate": self.yield_rate,
            "first_seen_at": self.first_seen_at.isoformat(),
            "last_seen_at": self.last_seen_at.isoformat(),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SuccessfulRegion":
        return cls(
            region_key=data.get("region_key", ""),
            phase=data.get("phase", "unknown"),
            family=data.get("family", "unknown"),
            feature=data.get("feature", "unknown"),
            hits=data.get("hits", 0),
            success_count=data.get("success_count", 0),
            score=data.get("score", 0.0),
            yield_rate=data.get("yield_rate", 0.0),
            first_seen_at=_parse_datetime(data.get("first_seen_at")) or _utc_now(),
            last_seen_at=_parse_datetime(data.get("last_seen_at")) or _utc_now(),
            metadata=_to_mapping(data.get("metadata", {})),
        )


@dataclass(frozen=True, slots=True)
class SuccessfulRegionsSummary:
    entry_count: int
    region_count: int
    total_hits: int
    total_success_count: int
    average_score: float
    average_yield_rate: float
    phase_counts: dict[str, int] = field(default_factory=dict)
    family_counts: dict[str, int] = field(default_factory=dict)
    feature_counts: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "entry_count", max(0, _coerce_int(self.entry_count)))
        object.__setattr__(self, "region_count", max(0, _coerce_int(self.region_count)))
        object.__setattr__(self, "total_hits", max(0, _coerce_int(self.total_hits)))
        object.__setattr__(self, "total_success_count", max(0, _coerce_int(self.total_success_count)))
        object.__setattr__(self, "average_score", _coerce_float(self.average_score))
        object.__setattr__(self, "average_yield_rate", _coerce_float(self.average_yield_rate))
        object.__setattr__(self, "phase_counts", dict(self.phase_counts))
        object.__setattr__(self, "family_counts", dict(self.family_counts))
        object.__setattr__(self, "feature_counts", dict(self.feature_counts))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_count": self.entry_count,
            "region_count": self.region_count,
            "total_hits": self.total_hits,
            "total_success_count": self.total_success_count,
            "average_score": self.average_score,
            "average_yield_rate": self.average_yield_rate,
            "phase_counts": dict(self.phase_counts),
            "family_counts": dict(self.family_counts),
            "feature_counts": dict(self.feature_counts),
            "metadata": dict(self.metadata),
        }


class SuccessfulRegions:
    def __init__(self, entries: Iterable[SuccessfulRegion] | None = None, *, metadata: Mapping[str, Any] | None = None) -> None:
        self.entries: list[SuccessfulRegion] = list(entries or [])
        self.metadata: dict[str, Any] = dict(metadata or {})

    def add(self, region: SuccessfulRegion) -> SuccessfulRegion:
        self.entries.append(region)
        return region

    def register(
        self,
        region_key: str,
        *,
        phase: str = "unknown",
        family: str = "unknown",
        feature: str = "unknown",
        hits: int = 1,
        success_count: int = 1,
        score: float = 0.0,
        yield_rate: float = 1.0,
        metadata: Mapping[str, Any] | None = None,
    ) -> SuccessfulRegion:
        region = SuccessfulRegionsBuilder.build(
            region_key=region_key,
            phase=phase,
            family=family,
            feature=feature,
            hits=hits,
            success_count=success_count,
            score=score,
            yield_rate=yield_rate,
            metadata=metadata,
        )
        return self.add(region)

    def touch(self, region_key: str, *, success: bool = True, score: float | None = None, metadata: Mapping[str, Any] | None = None) -> SuccessfulRegion | None:
        region_key = _normalize_text(region_key)
        for index, region in enumerate(self.entries):
            if region.region_key == region_key:
                updated = region.touch(success=success, score=score, metadata=metadata)
                self.entries[index] = updated
                return updated
        return None

    def by_phase(self, phase: str) -> tuple[SuccessfulRegion, ...]:
        phase = _normalize_text(phase, "unknown")
        return tuple(entry for entry in self.entries if entry.phase == phase)

    def by_family(self, family: str) -> tuple[SuccessfulRegion, ...]:
        family = _normalize_text(family, "unknown")
        return tuple(entry for entry in self.entries if entry.family == family)

    def by_feature(self, feature: str) -> tuple[SuccessfulRegion, ...]:
        feature = _normalize_text(feature, "unknown")
        return tuple(entry for entry in self.entries if entry.feature == feature)

    @property
    def summary(self) -> SuccessfulRegionsSummary:
        if not self.entries:
            return SuccessfulRegionsSummary(0, 0, 0, 0, 0.0, 0.0, metadata=dict(self.metadata))
        phase_counts = Counter(entry.phase for entry in self.entries)
        family_counts = Counter(entry.family for entry in self.entries)
        feature_counts = Counter(entry.feature for entry in self.entries)
        return SuccessfulRegionsSummary(
            entry_count=len(self.entries),
            region_count=len({entry.region_key for entry in self.entries}),
            total_hits=sum(entry.hits for entry in self.entries),
            total_success_count=sum(entry.success_count for entry in self.entries),
            average_score=sum(entry.score for entry in self.entries) / len(self.entries),
            average_yield_rate=sum(entry.yield_rate for entry in self.entries) / len(self.entries),
            phase_counts=dict(phase_counts),
            family_counts=dict(family_counts),
            feature_counts=dict(feature_counts),
            metadata=dict(self.metadata),
        )

    def to_dict(self, *, summary_only: bool = False) -> dict[str, Any]:
        return {
            "entries": [] if summary_only else [entry.to_dict() for entry in self.entries],
            "summary": self.summary.to_dict(),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SuccessfulRegions":
        return cls(
            entries=[SuccessfulRegion.from_dict(item) for item in data.get("entries", [])],
            metadata=_to_mapping(data.get("metadata", {})),
        )

    def __iter__(self):
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)


class SuccessfulRegionsBuilder:
    @staticmethod
    def build(
        *,
        region_key: str,
        phase: str = "unknown",
        family: str = "unknown",
        feature: str = "unknown",
        hits: int = 1,
        success_count: int = 1,
        score: float = 0.0,
        yield_rate: float = 1.0,
        metadata: Mapping[str, Any] | None = None,
        first_seen_at: datetime | None = None,
        last_seen_at: datetime | None = None,
    ) -> SuccessfulRegion:
        return SuccessfulRegion(
            region_key=region_key,
            phase=phase,
            family=family,
            feature=feature,
            hits=hits,
            success_count=success_count,
            score=score,
            yield_rate=yield_rate,
            first_seen_at=first_seen_at or _utc_now(),
            last_seen_at=last_seen_at or _utc_now(),
            metadata=_to_mapping(metadata or {}),
        )