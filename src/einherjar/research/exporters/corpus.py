# exporters/corpus.py
"""
==========================================================
Corpus Export
==========================================================

Représentation canonique du corpus final d'Einhers.

Ce module ne sélectionne rien :
- il reçoit un portefeuille final,
- il le normalise,
- il le prépare pour les exports.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from execution.execution_report import ExecutionResult
from portfolio.allocator import PortfolioAllocation
from portfolio.allocator import PortfolioAllocationEntry
from portfolio.portfolio_report import PortfolioReport
from portfolio.portfolio_report import PortfolioReportEntry
from portfolio.selector import PortfolioSelection
from portfolio.selector import PortfolioSelectionEntry

__all__ = [
    "CorpusEntry",
    "CorpusSummary",
    "Corpus",
    "CorpusBuilder",
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


def _bounded_unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _safe_mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=float)))


def _safe_max(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(np.max(np.asarray(values, dtype=float)))


def _safe_min(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(np.min(np.asarray(values, dtype=float)))


def _result_key(result: ExecutionResult) -> str:
    value = getattr(result, "subject_fingerprint", None)
    if value:
        return str(value)
    fp = getattr(result, "execution_fingerprint", None)
    if fp is not None:
        digest = getattr(fp, "digest", None)
        if digest:
            return str(digest)
    return ""


def _family_key(result: ExecutionResult) -> str:
    metadata = _to_mapping(result.metadata)
    for key in ("family", "target_family", "portfolio_family"):
        if key in metadata and metadata[key] is not None:
            value = str(metadata[key]).strip().lower()
            if value:
                return value

    candidate = getattr(result, "candidate", None)
    hypothesis = getattr(result, "hypothesis", None)
    for source in (candidate, hypothesis):
        if source is None:
            continue
        src_meta = _to_mapping(getattr(source, "metadata", None))
        for key in ("family", "target_family", "portfolio_family"):
            if key in src_meta and src_meta[key] is not None:
                value = str(src_meta[key]).strip().lower()
                if value:
                    return value

    try:
        conditions = getattr(hypothesis, "conditions", None)
        if conditions:
            fam = conditions[0].left.economic_family.value
            if fam:
                return str(fam).strip().lower()
    except Exception:
        pass

    return "unknown"


def _profile_name(result: ExecutionResult) -> str:
    profile = getattr(result, "profile", None)
    if profile is not None and getattr(profile, "name", None):
        value = str(profile.name).strip().lower()
        if value:
            return value

    metadata = _to_mapping(result.metadata)
    for key in ("profile_name", "strategy_name", "einher_name"):
        if key in metadata and metadata[key] is not None:
            value = str(metadata[key]).strip().lower()
            if value:
                return value

    return "unknown"


def _entry_from_any(entry: Any) -> tuple[ExecutionResult, float, float, float, str, str, int, dict[str, Any]]:
    """
    Normalise une entrée de portfolio en un tuple canonique.

    Retour :
    - result
    - weight
    - capital
    - score
    - family
    - profile_name
    - rank
    - metadata
    """
    if isinstance(entry, PortfolioReportEntry):
        return (
            entry.result,
            _coerce_float(entry.weight, 0.0),
            _coerce_float(entry.capital, 0.0),
            _coerce_float(entry.score, 0.0),
            str(entry.family),
            str(entry.profile_name),
            _coerce_int(entry.rank, 0),
            _to_mapping(entry.metadata),
        )

    if isinstance(entry, PortfolioAllocationEntry):
        return (
            entry.result,
            _coerce_float(entry.target_weight, 0.0),
            _coerce_float(entry.capital, 0.0),
            _coerce_float(entry.score, 0.0),
            str(entry.family),
            str(entry.profile_name),
            _coerce_int(entry.rank, 0),
            _to_mapping(entry.metadata),
        )

    if isinstance(entry, PortfolioSelectionEntry):
        result = entry.result
        return (
            result,
            _coerce_float(getattr(entry, "target_weight", getattr(entry, "score", 0.0)), 0.0),
            _coerce_float(getattr(entry, "capital", 0.0), 0.0),
            _coerce_float(entry.score, 0.0),
            str(entry.family),
            str(entry.profile_name),
            _coerce_int(entry.rank_hint, 0),
            _to_mapping(entry.metadata),
        )

    if isinstance(entry, ExecutionResult):
        return (
            entry,
            0.0,
            0.0,
            _coerce_float(entry.replay.metrics.total_pnl, 0.0),
            _family_key(entry),
            _profile_name(entry),
            0,
            {},
        )

    raise TypeError(f"Unsupported corpus entry type: {type(entry)!r}")


@dataclass(frozen=True, slots=True)
class CorpusEntry:
    """
    Entrée canonique du corpus final.
    """

    subject_fingerprint: str
    execution_fingerprint: str

    family: str = "unknown"
    profile_name: str = "unknown"
    source_kind: str = "portfolio"

    score: float = 0.0
    weight: float = 0.0
    capital: float = 0.0

    trade_count: int = 0
    total_pnl: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    max_drawdown: float = 0.0
    exposure_ratio: float = 0.0
    signal_coverage: float = 0.0

    healthy: bool = True
    issue_count: int = 0

    mae_mfe: dict[str, Any] = field(default_factory=dict)
    profile: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    risk: dict[str, Any] = field(default_factory=dict)
    diversification: dict[str, Any] = field(default_factory=dict)
    selection: dict[str, Any] = field(default_factory=dict)
    allocation: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    rank: int = 0
    created_at: datetime = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        object.__setattr__(self, "subject_fingerprint", str(self.subject_fingerprint))
        object.__setattr__(self, "execution_fingerprint", str(self.execution_fingerprint))
        object.__setattr__(self, "family", str(self.family).strip().lower() or "unknown")
        object.__setattr__(self, "profile_name", str(self.profile_name).strip().lower() or "unknown")
        object.__setattr__(self, "source_kind", str(self.source_kind).strip().lower() or "portfolio")
        object.__setattr__(self, "score", _bounded_unit(self.score))
        object.__setattr__(self, "weight", max(0.0, float(self.weight)))
        object.__setattr__(self, "capital", max(0.0, float(self.capital)))
        object.__setattr__(self, "trade_count", max(0, _coerce_int(self.trade_count, 0)))
        object.__setattr__(self, "total_pnl", float(self.total_pnl))
        object.__setattr__(self, "win_rate", _bounded_unit(self.win_rate))
        object.__setattr__(self, "profit_factor", float(self.profit_factor))
        object.__setattr__(self, "expectancy", float(self.expectancy))
        object.__setattr__(self, "max_drawdown", max(0.0, float(self.max_drawdown)))
        object.__setattr__(self, "exposure_ratio", _bounded_unit(self.exposure_ratio))
        object.__setattr__(self, "signal_coverage", _bounded_unit(self.signal_coverage))
        object.__setattr__(self, "healthy", _coerce_bool(self.healthy, True))
        object.__setattr__(self, "issue_count", max(0, _coerce_int(self.issue_count, 0)))
        object.__setattr__(self, "mae_mfe", dict(self.mae_mfe))
        object.__setattr__(self, "profile", dict(self.profile))
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))
        object.__setattr__(self, "risk", dict(self.risk))
        object.__setattr__(self, "diversification", dict(self.diversification))
        object.__setattr__(self, "selection", dict(self.selection))
        object.__setattr__(self, "allocation", dict(self.allocation))
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "rank", max(0, _coerce_int(self.rank, 0)))

    @property
    def is_final(self) -> bool:
        return self.capital > 0 and self.weight > 0

    @property
    def short_fingerprint(self) -> str:
        return self.subject_fingerprint[:12]

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject_fingerprint": self.subject_fingerprint,
            "execution_fingerprint": self.execution_fingerprint,
            "family": self.family,
            "profile_name": self.profile_name,
            "source_kind": self.source_kind,
            "score": self.score,
            "weight": self.weight,
            "capital": self.capital,
            "trade_count": self.trade_count,
            "total_pnl": self.total_pnl,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "expectancy": self.expectancy,
            "max_drawdown": self.max_drawdown,
            "exposure_ratio": self.exposure_ratio,
            "signal_coverage": self.signal_coverage,
            "healthy": self.healthy,
            "issue_count": self.issue_count,
            "mae_mfe": dict(self.mae_mfe),
            "profile": dict(self.profile),
            "diagnostics": dict(self.diagnostics),
            "risk": dict(self.risk),
            "diversification": dict(self.diversification),
            "selection": dict(self.selection),
            "allocation": dict(self.allocation),
            "metadata": dict(self.metadata),
            "rank": self.rank,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_result(
        cls,
        result: ExecutionResult,
        *,
        weight: float = 0.0,
        capital: float = 0.0,
        score: float | None = None,
        source_kind: str = "portfolio",
        rank: int = 0,
        metadata: Mapping[str, Any] | None = None,
    ) -> "CorpusEntry":
        execution_fp = getattr(result, "execution_fingerprint", None)
        execution_fingerprint = getattr(execution_fp, "digest", None) if execution_fp is not None else None
        if not execution_fingerprint:
            execution_fingerprint = _result_key(result)

        profile = getattr(result, "profile", None)
        diagnostics = getattr(result, "diagnostics", None)
        mae_mfe = getattr(result, "mae_mfe", None)

        trade_count = int(getattr(result.replay.metrics, "trade_count", 0))
        total_pnl = float(getattr(result.replay.metrics, "total_pnl", 0.0))
        win_rate = float(getattr(result.replay.metrics, "win_rate", 0.0))
        profit_factor = float(getattr(result.replay.metrics, "profit_factor", 0.0))
        expectancy = float(getattr(result.replay.metrics, "expectancy", 0.0))
        max_drawdown = float(getattr(result.replay.metrics, "max_drawdown", 0.0))
        exposure_ratio = float(getattr(result.replay.metrics, "exposure_ratio", 0.0))
        signal_coverage = float(getattr(result.replay.metrics, "signal_coverage", 0.0))

        healthy = bool(getattr(diagnostics, "healthy", True)) if diagnostics is not None else True
        issue_count = int(getattr(diagnostics, "issue_count", 0)) if diagnostics is not None else 0

        return cls(
            subject_fingerprint=_result_key(result),
            execution_fingerprint=str(execution_fingerprint),
            family=_family_key(result),
            profile_name=_profile_name(result),
            source_kind=source_kind,
            score=_coerce_float(score, float(getattr(diagnostics, "score", 0.0)) if diagnostics is not None else total_pnl),
            weight=weight,
            capital=capital,
            trade_count=trade_count,
            total_pnl=total_pnl,
            win_rate=win_rate,
            profit_factor=profit_factor,
            expectancy=expectancy,
            max_drawdown=max_drawdown,
            exposure_ratio=exposure_ratio,
            signal_coverage=signal_coverage,
            healthy=healthy,
            issue_count=issue_count,
            mae_mfe=mae_mfe.to_dict() if mae_mfe is not None else {},
            profile=profile.to_dict() if profile is not None else {},
            diagnostics=diagnostics.to_dict() if diagnostics is not None else {},
            risk=_to_mapping(metadata.get("risk") if metadata else None),
            diversification=_to_mapping(metadata.get("diversification") if metadata else None),
            selection=_to_mapping(metadata.get("selection") if metadata else None),
            allocation=_to_mapping(metadata.get("allocation") if metadata else None),
            metadata=dict(metadata or {}),
            rank=rank,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CorpusEntry":
        created_at = data.get("created_at")
        if isinstance(created_at, str) and created_at:
            created_at = datetime.fromisoformat(created_at)
        else:
            created_at = _utc_now()

        return cls(
            subject_fingerprint=data.get("subject_fingerprint", ""),
            execution_fingerprint=data.get("execution_fingerprint", ""),
            family=data.get("family", "unknown"),
            profile_name=data.get("profile_name", "unknown"),
            source_kind=data.get("source_kind", "portfolio"),
            score=_coerce_float(data.get("score"), 0.0),
            weight=_coerce_float(data.get("weight"), 0.0),
            capital=_coerce_float(data.get("capital"), 0.0),
            trade_count=_coerce_int(data.get("trade_count"), 0),
            total_pnl=_coerce_float(data.get("total_pnl"), 0.0),
            win_rate=_coerce_float(data.get("win_rate"), 0.0),
            profit_factor=_coerce_float(data.get("profit_factor"), 0.0),
            expectancy=_coerce_float(data.get("expectancy"), 0.0),
            max_drawdown=_coerce_float(data.get("max_drawdown"), 0.0),
            exposure_ratio=_coerce_float(data.get("exposure_ratio"), 0.0),
            signal_coverage=_coerce_float(data.get("signal_coverage"), 0.0),
            healthy=_coerce_bool(data.get("healthy"), True),
            issue_count=_coerce_int(data.get("issue_count"), 0),
            mae_mfe=_to_mapping(data.get("mae_mfe", {})),
            profile=_to_mapping(data.get("profile", {})),
            diagnostics=_to_mapping(data.get("diagnostics", {})),
            risk=_to_mapping(data.get("risk", {})),
            diversification=_to_mapping(data.get("diversification", {})),
            selection=_to_mapping(data.get("selection", {})),
            allocation=_to_mapping(data.get("allocation", {})),
            metadata=_to_mapping(data.get("metadata", {})),
            rank=_coerce_int(data.get("rank"), 0),
            created_at=created_at,
        )

    def to_record(self) -> dict[str, Any]:
        record = self.to_dict()
        record["is_final"] = self.is_final
        record["short_fingerprint"] = self.short_fingerprint
        return record


@dataclass(frozen=True, slots=True)
class CorpusSummary:
    """
    Résumé global du corpus.
    """

    entry_count: int
    selected_count: int
    total_capital: float
    total_weight: float
    total_pnl: float

    average_score: float
    best_score: float
    best_subject_fingerprint: str | None = None

    average_win_rate: float = 0.0
    average_profit_factor: float = 0.0
    average_expectancy: float = 0.0
    average_drawdown: float = 0.0

    healthy_count: int = 0
    unhealthy_count: int = 0

    unique_family_count: int = 0
    unique_profile_count: int = 0

    family_counts: dict[str, int] = field(default_factory=dict)
    profile_counts: dict[str, int] = field(default_factory=dict)

    min_weight: float = 0.0
    max_weight: float = 0.0
    mean_weight: float = 0.0

    min_capital: float = 0.0
    max_capital: float = 0.0
    mean_capital: float = 0.0

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "entry_count", max(0, _coerce_int(self.entry_count, 0)))
        object.__setattr__(self, "selected_count", max(0, _coerce_int(self.selected_count, 0)))
        object.__setattr__(self, "total_capital", max(0.0, float(self.total_capital)))
        object.__setattr__(self, "total_weight", max(0.0, float(self.total_weight)))
        object.__setattr__(self, "total_pnl", float(self.total_pnl))
        object.__setattr__(self, "average_score", _bounded_unit(self.average_score))
        object.__setattr__(self, "best_score", _bounded_unit(self.best_score))
        object.__setattr__(self, "average_win_rate", _bounded_unit(self.average_win_rate))
        object.__setattr__(self, "average_profit_factor", float(self.average_profit_factor))
        object.__setattr__(self, "average_expectancy", float(self.average_expectancy))
        object.__setattr__(self, "average_drawdown", max(0.0, float(self.average_drawdown)))
        object.__setattr__(self, "healthy_count", max(0, _coerce_int(self.healthy_count, 0)))
        object.__setattr__(self, "unhealthy_count", max(0, _coerce_int(self.unhealthy_count, 0)))
        object.__setattr__(self, "unique_family_count", max(0, _coerce_int(self.unique_family_count, 0)))
        object.__setattr__(self, "unique_profile_count", max(0, _coerce_int(self.unique_profile_count, 0)))
        object.__setattr__(self, "family_counts", dict(self.family_counts))
        object.__setattr__(self, "profile_counts", dict(self.profile_counts))
        object.__setattr__(self, "min_weight", max(0.0, float(self.min_weight)))
        object.__setattr__(self, "max_weight", max(0.0, float(self.max_weight)))
        object.__setattr__(self, "mean_weight", max(0.0, float(self.mean_weight)))
        object.__setattr__(self, "min_capital", max(0.0, float(self.min_capital)))
        object.__setattr__(self, "max_capital", max(0.0, float(self.max_capital)))
        object.__setattr__(self, "mean_capital", max(0.0, float(self.mean_capital)))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_count": self.entry_count,
            "selected_count": self.selected_count,
            "total_capital": self.total_capital,
            "total_weight": self.total_weight,
            "total_pnl": self.total_pnl,
            "average_score": self.average_score,
            "best_score": self.best_score,
            "best_subject_fingerprint": self.best_subject_fingerprint,
            "average_win_rate": self.average_win_rate,
            "average_profit_factor": self.average_profit_factor,
            "average_expectancy": self.average_expectancy,
            "average_drawdown": self.average_drawdown,
            "healthy_count": self.healthy_count,
            "unhealthy_count": self.unhealthy_count,
            "unique_family_count": self.unique_family_count,
            "unique_profile_count": self.unique_profile_count,
            "family_counts": dict(self.family_counts),
            "profile_counts": dict(self.profile_counts),
            "min_weight": self.min_weight,
            "max_weight": self.max_weight,
            "mean_weight": self.mean_weight,
            "min_capital": self.min_capital,
            "max_capital": self.max_capital,
            "mean_capital": self.mean_capital,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class Corpus:
    """
    Corpus final exportable.
    """

    name: str = "corpus"
    entries: list[CorpusEntry] = field(default_factory=list)
    summary: CorpusSummary | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", str(self.name).strip() or "corpus")
        object.__setattr__(self, "entries", list(self.entries))
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "rejected", list(self.rejected))

        if self.summary is None:
            object.__setattr__(self, "summary", CorpusSummaryBuilder.build(self.entries, metadata=self.metadata))

    @property
    def selected_count(self) -> int:
        return sum(1 for entry in self.entries if entry.is_final)

    @property
    def final_entries(self) -> tuple[CorpusEntry, ...]:
        return tuple(entry for entry in self.entries if entry.is_final)

    def to_dict(self, *, summary_only: bool = False) -> dict[str, Any]:
        payload = {
            "name": self.name,
            "created_at": self.created_at.isoformat(),
            "entries": [] if summary_only else [entry.to_dict() for entry in self.entries],
            "rejected": [] if summary_only else [dict(item) for item in self.rejected],
            "summary": self.summary.to_dict() if self.summary is not None else None,
            "metadata": dict(self.metadata),
        }
        return payload

    def to_records(self) -> list[dict[str, Any]]:
        return [entry.to_record() for entry in self.entries]

    def to_rows(self) -> list[dict[str, Any]]:
        return self.to_records()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Corpus":
        summary_data = data.get("summary")
        summary = CorpusSummary.from_dict(summary_data) if isinstance(summary_data, Mapping) else None
        return cls(
            name=data.get("name", "corpus"),
            entries=[CorpusEntry.from_dict(item) for item in data.get("entries", [])],
            summary=summary,
            metadata=_to_mapping(data.get("metadata", {})),
            rejected=[dict(item) for item in data.get("rejected", [])],
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else _utc_now(),
        )

    def __iter__(self):
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    def __repr__(self) -> str:
        return (
            "Corpus("
            f"name='{self.name}', "
            f"entries={len(self.entries)}, "
            f"selected={self.selected_count}"
            ")"
        )


class CorpusSummaryBuilder:
    """
    Construit un résumé de corpus.
    """

    @staticmethod
    def build(entries: Sequence[CorpusEntry], *, metadata: Mapping[str, Any] | None = None) -> CorpusSummary:
        entries = tuple(entries)
        if not entries:
            return CorpusSummary(
                entry_count=0,
                selected_count=0,
                total_capital=0.0,
                total_weight=0.0,
                total_pnl=0.0,
                average_score=0.0,
                best_score=0.0,
                best_subject_fingerprint=None,
                average_win_rate=0.0,
                average_profit_factor=0.0,
                average_expectancy=0.0,
                average_drawdown=0.0,
                healthy_count=0,
                unhealthy_count=0,
                unique_family_count=0,
                unique_profile_count=0,
                family_counts={},
                profile_counts={},
                min_weight=0.0,
                max_weight=0.0,
                mean_weight=0.0,
                min_capital=0.0,
                max_capital=0.0,
                mean_capital=0.0,
                metadata=dict(metadata or {}),
            )

        family_counts = Counter(entry.family for entry in entries)
        profile_counts = Counter(entry.profile_name for entry in entries)

        weights = [entry.weight for entry in entries]
        capitals = [entry.capital for entry in entries]
        scores = [entry.score for entry in entries]
        win_rates = [entry.win_rate for entry in entries]
        pfs = [entry.profit_factor for entry in entries]
        expectancies = [entry.expectancy for entry in entries]
        drawdowns = [entry.max_drawdown for entry in entries]

        selected_entries = [entry for entry in entries if entry.is_final]

        best_entry = max(entries, key=lambda entry: (entry.score, entry.capital, entry.weight, entry.total_pnl))

        return CorpusSummary(
            entry_count=len(entries),
            selected_count=len(selected_entries),
            total_capital=float(sum(capitals)),
            total_weight=float(sum(weights)),
            total_pnl=float(sum(entry.total_pnl for entry in entries)),
            average_score=_safe_mean(scores),
            best_score=best_entry.score,
            best_subject_fingerprint=best_entry.subject_fingerprint,
            average_win_rate=_safe_mean(win_rates),
            average_profit_factor=_safe_mean(pfs),
            average_expectancy=_safe_mean(expectancies),
            average_drawdown=_safe_mean(drawdowns),
            healthy_count=sum(1 for entry in entries if entry.healthy),
            unhealthy_count=sum(1 for entry in entries if not entry.healthy),
            unique_family_count=len(family_counts),
            unique_profile_count=len(profile_counts),
            family_counts=dict(family_counts),
            profile_counts=dict(profile_counts),
            min_weight=_safe_min(weights),
            max_weight=_safe_max(weights),
            mean_weight=_safe_mean(weights),
            min_capital=_safe_min(capitals),
            max_capital=_safe_max(capitals),
            mean_capital=_safe_mean(capitals),
            metadata=dict(metadata or {}),
        )


class CorpusBuilder:
    """
    Construit le corpus final à partir d'un portefeuille.
    """

    @staticmethod
    def from_portfolio_report(
        report: PortfolioReport,
        *,
        include_rejected: bool = True,
        source_kind: str = "portfolio_report",
        metadata: Mapping[str, Any] | None = None,
    ) -> Corpus:
        entries: list[CorpusEntry] = []
        for idx, entry in enumerate(report.entries):
            entries.append(
                CorpusEntry.from_result(
                    entry.result,
                    weight=_coerce_float(entry.weight, 0.0),
                    capital=_coerce_float(entry.capital, 0.0),
                    score=_coerce_float(entry.score, 0.0),
                    source_kind=source_kind,
                    rank=_coerce_int(entry.rank, idx),
                    metadata={
                        **_to_mapping(report.metadata),
                        **_to_mapping(entry.metadata),
                        "selection": entry.to_dict(),
                    },
                )
            )

        summary = CorpusSummaryBuilder.build(entries, metadata={**_to_mapping(report.metadata), **_to_mapping(metadata)})
        rejected = list(report.rejected) if include_rejected else []

        return Corpus(
            name=report.name if getattr(report, "name", None) else "corpus",
            entries=entries,
            summary=summary,
            metadata={**_to_mapping(report.metadata), **_to_mapping(metadata)},
            rejected=rejected,
            created_at=getattr(report, "created_at", _utc_now()),
        )

    @staticmethod
    def from_portfolio_allocation(
        allocation: PortfolioAllocation,
        *,
        include_rejected: bool = True,
        metadata: Mapping[str, Any] | None = None,
    ) -> Corpus:
        entries: list[CorpusEntry] = []
        rejected: list[dict[str, Any]] = []

        for idx, entry in enumerate(allocation.entries):
            if not entry.accepted or entry.capital <= 0:
                rejected.append(
                    {
                        "subject_fingerprint": entry.subject_fingerprint,
                        "family": entry.family,
                        "profile_name": entry.profile_name,
                        "reason": "allocation_not_accepted",
                        "score": entry.score,
                        "weight": entry.target_weight,
                        "capital": entry.capital,
                    }
                )
                continue

            entries.append(
                CorpusEntry.from_result(
                    entry.result,
                    weight=_coerce_float(entry.target_weight, 0.0),
                    capital=_coerce_float(entry.capital, 0.0),
                    score=_coerce_float(entry.score, 0.0),
                    source_kind="portfolio_allocation",
                    rank=idx,
                    metadata={
                        **_to_mapping(entry.metadata),
                        "allocation": entry.to_dict(),
                    },
                )
            )

        summary = CorpusSummaryBuilder.build(entries, metadata={**_to_mapping(metadata)})
        if not include_rejected:
            rejected = []

        return Corpus(
            name="corpus",
            entries=entries,
            summary=summary,
            metadata={**_to_mapping(metadata)},
            rejected=rejected,
        )

    @staticmethod
    def from_portfolio_selection(
        selection: PortfolioSelection,
        *,
        include_rejected: bool = True,
        metadata: Mapping[str, Any] | None = None,
    ) -> Corpus:
        entries: list[CorpusEntry] = []
        rejected: list[dict[str, Any]] = []

        for idx, entry in enumerate(selection.selected):
            entries.append(
                CorpusEntry.from_result(
                    entry.result,
                    weight=_coerce_float(getattr(entry, "score", 0.0), 0.0),
                    capital=_coerce_float(getattr(entry, "capital", 0.0), 0.0),
                    score=_coerce_float(entry.score, 0.0),
                    source_kind="portfolio_selection",
                    rank=idx,
                    metadata={
                        **_to_mapping(entry.metadata),
                        "selection": entry.to_dict(),
                    },
                )
            )

        if include_rejected:
            for item in selection.rejected:
                rejected.append(
                    {
                        "subject_fingerprint": item.subject_fingerprint,
                        "family": item.family,
                        "profile_name": item.profile_name,
                        "reason": list(item.reasons),
                        "score": item.score,
                        "accepted": item.accepted,
                    }
                )

        summary = CorpusSummaryBuilder.build(entries, metadata={**_to_mapping(metadata)})
        return Corpus(
            name="corpus",
            entries=entries,
            summary=summary,
            metadata={**_to_mapping(metadata)},
            rejected=rejected,
        )

    @staticmethod
    def from_results(
        results: Iterable[ExecutionResult],
        *,
        include_rejected: bool = False,
        metadata: Mapping[str, Any] | None = None,
    ) -> Corpus:
        entries = [
            CorpusEntry.from_result(
                result,
                weight=0.0,
                capital=0.0,
                score=float(result.replay.metrics.total_pnl),
                source_kind="execution_result",
                rank=index,
                metadata={"result": result.to_dict(summary_only=True)},
            )
            for index, result in enumerate(results)
        ]
        summary = CorpusSummaryBuilder.build(entries, metadata={**_to_mapping(metadata)})
        return Corpus(
            name="corpus",
            entries=entries,
            summary=summary,
            metadata={**_to_mapping(metadata)},
            rejected=[] if not include_rejected else [],
        )