# memory/search_history.py
"""
==========================================================
Search History
==========================================================

Mémoire chronologique des recherches du moteur.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "SearchEntry",
    "SearchSummary",
    "SearchHistory",
    "SearchHistoryBuilder",
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


def _normalize_tuple(values: Any) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes)):
        values = (values,)
    return tuple(_normalize_text(item).lower() for item in values if _normalize_text(item))


def _stable_digest(payload: Mapping[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str, separators=(",", ":"))
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class SearchEntry:
    search_key: str
    query: str
    phase: str = "unknown"
    objective: str = ""
    seed: str = ""
    features: tuple[str, ...] = ()
    families: tuple[str, ...] = ()
    regions: tuple[str, ...] = ()
    parameters: dict[str, Any] = field(default_factory=dict)
    result_count: int = 0
    accepted_count: int = 0
    rejected_count: int = 0
    useful: bool = False
    success: bool = False
    score: float = 0.0
    reason: str = ""
    notes: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=_utc_now)
    completed_at: datetime | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "search_key", _normalize_text(self.search_key))
        object.__setattr__(self, "query", _normalize_text(self.query))
        object.__setattr__(self, "phase", _normalize_text(self.phase, "unknown"))
        object.__setattr__(self, "objective", _normalize_text(self.objective))
        object.__setattr__(self, "seed", _normalize_text(self.seed))
        object.__setattr__(self, "features", tuple(dict.fromkeys(_normalize_tuple(self.features))))
        object.__setattr__(self, "families", tuple(dict.fromkeys(_normalize_tuple(self.families))))
        object.__setattr__(self, "regions", tuple(dict.fromkeys(_normalize_tuple(self.regions))))
        object.__setattr__(self, "parameters", dict(self.parameters))
        object.__setattr__(self, "result_count", max(0, _coerce_int(self.result_count)))
        object.__setattr__(self, "accepted_count", max(0, _coerce_int(self.accepted_count)))
        object.__setattr__(self, "rejected_count", max(0, _coerce_int(self.rejected_count)))
        object.__setattr__(self, "useful", _coerce_bool(self.useful))
        object.__setattr__(self, "success", _coerce_bool(self.success))
        object.__setattr__(self, "score", _coerce_float(self.score))
        object.__setattr__(self, "reason", _normalize_text(self.reason))
        object.__setattr__(self, "notes", tuple(dict.fromkeys(_normalize_tuple(self.notes))))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def is_positive(self) -> bool:
        return self.success or self.useful or self.score > 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "search_key": self.search_key,
            "query": self.query,
            "phase": self.phase,
            "objective": self.objective,
            "seed": self.seed,
            "features": list(self.features),
            "families": list(self.families),
            "regions": list(self.regions),
            "parameters": dict(self.parameters),
            "result_count": self.result_count,
            "accepted_count": self.accepted_count,
            "rejected_count": self.rejected_count,
            "useful": self.useful,
            "success": self.success,
            "score": self.score,
            "reason": self.reason,
            "notes": list(self.notes),
            "metadata": dict(self.metadata),
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SearchEntry":
        return cls(
            search_key=data.get("search_key", ""),
            query=data.get("query", ""),
            phase=data.get("phase", "unknown"),
            objective=data.get("objective", ""),
            seed=data.get("seed", ""),
            features=tuple(data.get("features", ())),
            families=tuple(data.get("families", ())),
            regions=tuple(data.get("regions", ())),
            parameters=_to_mapping(data.get("parameters", {})),
            result_count=data.get("result_count", 0),
            accepted_count=data.get("accepted_count", 0),
            rejected_count=data.get("rejected_count", 0),
            useful=data.get("useful", False),
            success=data.get("success", False),
            score=data.get("score", 0.0),
            reason=data.get("reason", ""),
            notes=tuple(data.get("notes", ())),
            metadata=_to_mapping(data.get("metadata", {})),
            created_at=_parse_datetime(data.get("created_at")) or _utc_now(),
            completed_at=_parse_datetime(data.get("completed_at")),
        )


@dataclass(frozen=True, slots=True)
class SearchSummary:
    entry_count: int
    positive_count: int
    success_count: int
    useful_count: int
    average_score: float
    max_score: float
    phase_counts: dict[str, int] = field(default_factory=dict)
    family_counts: dict[str, int] = field(default_factory=dict)
    feature_counts: dict[str, int] = field(default_factory=dict)
    region_counts: dict[str, int] = field(default_factory=dict)
    top_queries: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "entry_count", max(0, _coerce_int(self.entry_count)))
        object.__setattr__(self, "positive_count", max(0, _coerce_int(self.positive_count)))
        object.__setattr__(self, "success_count", max(0, _coerce_int(self.success_count)))
        object.__setattr__(self, "useful_count", max(0, _coerce_int(self.useful_count)))
        object.__setattr__(self, "average_score", _coerce_float(self.average_score))
        object.__setattr__(self, "max_score", _coerce_float(self.max_score))
        object.__setattr__(self, "phase_counts", dict(self.phase_counts))
        object.__setattr__(self, "family_counts", dict(self.family_counts))
        object.__setattr__(self, "feature_counts", dict(self.feature_counts))
        object.__setattr__(self, "region_counts", dict(self.region_counts))
        object.__setattr__(self, "top_queries", tuple(self.top_queries))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_count": self.entry_count,
            "positive_count": self.positive_count,
            "success_count": self.success_count,
            "useful_count": self.useful_count,
            "average_score": self.average_score,
            "max_score": self.max_score,
            "phase_counts": dict(self.phase_counts),
            "family_counts": dict(self.family_counts),
            "feature_counts": dict(self.feature_counts),
            "region_counts": dict(self.region_counts),
            "top_queries": list(self.top_queries),
            "metadata": dict(self.metadata),
        }


class SearchHistory:
    def __init__(self, entries: Iterable[SearchEntry] | None = None, *, metadata: Mapping[str, Any] | None = None) -> None:
        self.entries: list[SearchEntry] = list(entries or [])
        self.metadata: dict[str, Any] = dict(metadata or {})

    def add(self, entry: SearchEntry) -> SearchEntry:
        self.entries.append(entry)
        return entry

    def record(self, **kwargs: Any) -> SearchEntry:
        entry = SearchHistoryBuilder.build(**kwargs)
        return self.add(entry)

    def latest(self) -> SearchEntry | None:
        return self.entries[-1] if self.entries else None

    def by_phase(self, phase: str) -> tuple[SearchEntry, ...]:
        phase = _normalize_text(phase, "unknown")
        return tuple(entry for entry in self.entries if entry.phase == phase)

    def by_query(self, query: str) -> tuple[SearchEntry, ...]:
        query = _normalize_text(query)
        return tuple(entry for entry in self.entries if query and query.lower() in entry.query.lower())

    def by_feature(self, feature: str) -> tuple[SearchEntry, ...]:
        feature = _normalize_text(feature).lower()
        return tuple(entry for entry in self.entries if feature in entry.features)

    def by_family(self, family: str) -> tuple[SearchEntry, ...]:
        family = _normalize_text(family).lower()
        return tuple(entry for entry in self.entries if family in entry.families)

    def recent(self, limit: int = 10) -> tuple[SearchEntry, ...]:
        limit = max(1, _coerce_int(limit, 10))
        return tuple(self.entries[-limit:])

    @property
    def summary(self) -> SearchSummary:
        if not self.entries:
            return SearchSummary(
                entry_count=0,
                positive_count=0,
                success_count=0,
                useful_count=0,
                average_score=0.0,
                max_score=0.0,
                metadata=dict(self.metadata),
            )

        phase_counts = Counter(entry.phase for entry in self.entries)
        family_counts = Counter(fam for entry in self.entries for fam in entry.families)
        feature_counts = Counter(feat for entry in self.entries for feat in entry.features)
        region_counts = Counter(region for entry in self.entries for region in entry.regions)
        top_queries = tuple(
            query for query, _ in Counter(entry.query for entry in self.entries if entry.query).most_common(10)
        )
        scores = [entry.score for entry in self.entries]
        return SearchSummary(
            entry_count=len(self.entries),
            positive_count=sum(1 for entry in self.entries if entry.is_positive),
            success_count=sum(1 for entry in self.entries if entry.success),
            useful_count=sum(1 for entry in self.entries if entry.useful),
            average_score=sum(scores) / len(scores),
            max_score=max(scores),
            phase_counts=dict(phase_counts),
            family_counts=dict(family_counts),
            feature_counts=dict(feature_counts),
            region_counts=dict(region_counts),
            top_queries=top_queries,
            metadata=dict(self.metadata),
        )

    def to_dict(self, *, summary_only: bool = False) -> dict[str, Any]:
        return {
            "entries": [] if summary_only else [entry.to_dict() for entry in self.entries],
            "summary": self.summary.to_dict(),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SearchHistory":
        return cls(
            entries=[SearchEntry.from_dict(item) for item in data.get("entries", [])],
            metadata=_to_mapping(data.get("metadata", {})),
        )

    def __iter__(self):
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    def __repr__(self) -> str:
        return f"SearchHistory(entries={len(self.entries)})"


class SearchHistoryBuilder:
    @staticmethod
    def build(
        *,
        query: str,
        phase: str = "unknown",
        objective: str = "",
        seed: str = "",
        features: Sequence[str] | None = None,
        families: Sequence[str] | None = None,
        regions: Sequence[str] | None = None,
        parameters: Mapping[str, Any] | None = None,
        result_count: int = 0,
        accepted_count: int = 0,
        rejected_count: int = 0,
        useful: bool = False,
        success: bool = False,
        score: float = 0.0,
        reason: str = "",
        notes: Sequence[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
        search_key: str | None = None,
        created_at: datetime | None = None,
        completed_at: datetime | None = None,
    ) -> SearchEntry:
        payload = {
            "query": _normalize_text(query),
            "phase": _normalize_text(phase, "unknown"),
            "objective": _normalize_text(objective),
            "seed": _normalize_text(seed),
            "features": list(_normalize_tuple(features)),
            "families": list(_normalize_tuple(families)),
            "regions": list(_normalize_tuple(regions)),
            "parameters": _to_mapping(parameters or {}),
            "result_count": _coerce_int(result_count),
            "accepted_count": _coerce_int(accepted_count),
            "rejected_count": _coerce_int(rejected_count),
            "useful": _coerce_bool(useful),
            "success": _coerce_bool(success),
            "score": _coerce_float(score),
            "reason": _normalize_text(reason),
            "notes": list(_normalize_tuple(notes)),
            "metadata": _to_mapping(metadata or {}),
        }
        key = search_key or _stable_digest(payload)
        return SearchEntry(
            search_key=key,
            query=payload["query"],
            phase=payload["phase"],
            objective=payload["objective"],
            seed=payload["seed"],
            features=tuple(payload["features"]),
            families=tuple(payload["families"]),
            regions=tuple(payload["regions"]),
            parameters=payload["parameters"],
            result_count=payload["result_count"],
            accepted_count=payload["accepted_count"],
            rejected_count=payload["rejected_count"],
            useful=payload["useful"],
            success=payload["success"],
            score=payload["score"],
            reason=payload["reason"],
            notes=tuple(payload["notes"]),
            metadata=payload["metadata"],
            created_at=created_at or _utc_now(),
            completed_at=completed_at,
        )