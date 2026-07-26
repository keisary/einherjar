# memory/corpus_history.py
"""
==========================================================
Corpus History
==========================================================

Historique des corpus successifs d'Einhers.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "CorpusVersion",
    "CorpusHistorySummary",
    "CorpusHistory",
    "CorpusHistoryBuilder",
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
class CorpusVersion:
    corpus_key: str
    version: int = 1
    entry_count: int = 0
    selected_count: int = 0
    total_capital: float = 0.0
    total_weight: float = 0.0
    total_pnl: float = 0.0
    best_subject_fingerprint: str = ""
    fingerprints: tuple[str, ...] = ()
    summary: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        object.__setattr__(self, "corpus_key", _normalize_text(self.corpus_key, "corpus"))
        object.__setattr__(self, "version", max(1, _coerce_int(self.version, 1)))
        object.__setattr__(self, "entry_count", max(0, _coerce_int(self.entry_count)))
        object.__setattr__(self, "selected_count", max(0, _coerce_int(self.selected_count)))
        object.__setattr__(self, "total_capital", _coerce_float(self.total_capital))
        object.__setattr__(self, "total_weight", _coerce_float(self.total_weight))
        object.__setattr__(self, "total_pnl", _coerce_float(self.total_pnl))
        object.__setattr__(self, "best_subject_fingerprint", _normalize_text(self.best_subject_fingerprint))
        object.__setattr__(self, "fingerprints", tuple(dict.fromkeys(str(fp) for fp in self.fingerprints if str(fp).strip())))
        object.__setattr__(self, "summary", dict(self.summary))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "corpus_key": self.corpus_key,
            "version": self.version,
            "entry_count": self.entry_count,
            "selected_count": self.selected_count,
            "total_capital": self.total_capital,
            "total_weight": self.total_weight,
            "total_pnl": self.total_pnl,
            "best_subject_fingerprint": self.best_subject_fingerprint,
            "fingerprints": list(self.fingerprints),
            "summary": dict(self.summary),
            "metadata": dict(self.metadata),
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CorpusVersion":
        return cls(
            corpus_key=data.get("corpus_key", "corpus"),
            version=data.get("version", 1),
            entry_count=data.get("entry_count", 0),
            selected_count=data.get("selected_count", 0),
            total_capital=data.get("total_capital", 0.0),
            total_weight=data.get("total_weight", 0.0),
            total_pnl=data.get("total_pnl", 0.0),
            best_subject_fingerprint=data.get("best_subject_fingerprint", ""),
            fingerprints=tuple(data.get("fingerprints", ())),
            summary=_to_mapping(data.get("summary", {})),
            metadata=_to_mapping(data.get("metadata", {})),
            created_at=_parse_datetime(data.get("created_at")) or _utc_now(),
        )


@dataclass(frozen=True, slots=True)
class CorpusHistorySummary:
    version_count: int
    latest_version: int
    total_entries: int
    total_selected: int
    total_capital: float
    total_weight: float
    total_pnl: float
    average_entries: float
    version_counts_by_corpus: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "version_count", max(0, _coerce_int(self.version_count)))
        object.__setattr__(self, "latest_version", max(0, _coerce_int(self.latest_version)))
        object.__setattr__(self, "total_entries", max(0, _coerce_int(self.total_entries)))
        object.__setattr__(self, "total_selected", max(0, _coerce_int(self.total_selected)))
        object.__setattr__(self, "total_capital", _coerce_float(self.total_capital))
        object.__setattr__(self, "total_weight", _coerce_float(self.total_weight))
        object.__setattr__(self, "total_pnl", _coerce_float(self.total_pnl))
        object.__setattr__(self, "average_entries", _coerce_float(self.average_entries))
        object.__setattr__(self, "version_counts_by_corpus", dict(self.version_counts_by_corpus))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "version_count": self.version_count,
            "latest_version": self.latest_version,
            "total_entries": self.total_entries,
            "total_selected": self.total_selected,
            "total_capital": self.total_capital,
            "total_weight": self.total_weight,
            "total_pnl": self.total_pnl,
            "average_entries": self.average_entries,
            "version_counts_by_corpus": dict(self.version_counts_by_corpus),
            "metadata": dict(self.metadata),
        }


class CorpusHistory:
    def __init__(self, versions: Iterable[CorpusVersion] | None = None, *, metadata: Mapping[str, Any] | None = None) -> None:
        self.versions: list[CorpusVersion] = list(versions or [])
        self.metadata: dict[str, Any] = dict(metadata or {})

    def add(self, version: CorpusVersion) -> CorpusVersion:
        self.versions.append(version)
        return version

    def register(
        self,
        corpus_key: str,
        *,
        version: int = 1,
        entry_count: int = 0,
        selected_count: int = 0,
        total_capital: float = 0.0,
        total_weight: float = 0.0,
        total_pnl: float = 0.0,
        fingerprints: Sequence[str] | None = None,
        summary: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> CorpusVersion:
        version_obj = CorpusHistoryBuilder.build(
            corpus_key=corpus_key,
            version=version,
            entry_count=entry_count,
            selected_count=selected_count,
            total_capital=total_capital,
            total_weight=total_weight,
            total_pnl=total_pnl,
            fingerprints=fingerprints,
            summary=summary,
            metadata=metadata,
        )
        return self.add(version_obj)

    def latest(self, corpus_key: str | None = None) -> CorpusVersion | None:
        if not self.versions:
            return None
        if corpus_key is None:
            return self.versions[-1]
        corpus_key = _normalize_text(corpus_key, "corpus")
        for version in reversed(self.versions):
            if version.corpus_key == corpus_key:
                return version
        return None

    def versions_for(self, corpus_key: str) -> tuple[CorpusVersion, ...]:
        corpus_key = _normalize_text(corpus_key, "corpus")
        return tuple(version for version in self.versions if version.corpus_key == corpus_key)

    def best(self) -> CorpusVersion | None:
        if not self.versions:
            return None
        return max(self.versions, key=lambda item: (item.total_pnl, item.selected_count, item.total_capital))

    @property
    def summary(self) -> CorpusHistorySummary:
        if not self.versions:
            return CorpusHistorySummary(0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, metadata=dict(self.metadata))
        version_counts = Counter(version.corpus_key for version in self.versions)
        return CorpusHistorySummary(
            version_count=len(self.versions),
            latest_version=max(version.version for version in self.versions),
            total_entries=sum(version.entry_count for version in self.versions),
            total_selected=sum(version.selected_count for version in self.versions),
            total_capital=sum(version.total_capital for version in self.versions),
            total_weight=sum(version.total_weight for version in self.versions),
            total_pnl=sum(version.total_pnl for version in self.versions),
            average_entries=sum(version.entry_count for version in self.versions) / len(self.versions),
            version_counts_by_corpus=dict(version_counts),
            metadata=dict(self.metadata),
        )

    def to_dict(self, *, summary_only: bool = False) -> dict[str, Any]:
        return {
            "versions": [] if summary_only else [version.to_dict() for version in self.versions],
            "summary": self.summary.to_dict(),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CorpusHistory":
        return cls(
            versions=[CorpusVersion.from_dict(item) for item in data.get("versions", [])],
            metadata=_to_mapping(data.get("metadata", {})),
        )

    def __iter__(self):
        return iter(self.versions)

    def __len__(self) -> int:
        return len(self.versions)


class CorpusHistoryBuilder:
    @staticmethod
    def build(
        *,
        corpus_key: str,
        version: int = 1,
        entry_count: int = 0,
        selected_count: int = 0,
        total_capital: float = 0.0,
        total_weight: float = 0.0,
        total_pnl: float = 0.0,
        best_subject_fingerprint: str = "",
        fingerprints: Sequence[str] | None = None,
        summary: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        created_at: datetime | None = None,
    ) -> CorpusVersion:
        if summary is None:
            summary = {}
        if best_subject_fingerprint:
            summary = {**_to_mapping(summary), "best_subject_fingerprint": best_subject_fingerprint}
        return CorpusVersion(
            corpus_key=corpus_key,
            version=version,
            entry_count=entry_count,
            selected_count=selected_count,
            total_capital=total_capital,
            total_weight=total_weight,
            total_pnl=total_pnl,
            best_subject_fingerprint=best_subject_fingerprint,
            fingerprints=tuple(fingerprints or ()),
            summary=_to_mapping(summary),
            metadata=_to_mapping(metadata or {}),
            created_at=created_at or _utc_now(),
        )