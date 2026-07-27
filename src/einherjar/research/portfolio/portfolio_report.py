# portfolio/portfolio_report.py
"""
==========================================================
Portfolio Report
==========================================================

Agrège le résultat final du module portfolio.

Le rapport fige le corpus des Einhers retenus avec :
- leurs poids,
- leur capital,
- le risque global,
- la diversification,
- la corrélation,
- et les exclusions éventuelles.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from typing import Iterable
from typing import Mapping

from execution.execution_report import ExecutionResult

from .allocator import PortfolioAllocation
from .allocator import PortfolioAllocationEntry
from .capital import CapitalPlan
from .correlation import PortfolioCorrelationMatrix
from .diversification import DiversificationAssessment
from .risk import PortfolioRiskAssessment
from .selector import PortfolioSelection
from .selector import PortfolioSelectionEntry

__all__ = [
    "PortfolioReportEntry",
    "PortfolioReport",
    "PortfolioReporter",
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


@dataclass(frozen=True, slots=True)
class PortfolioReportEntry:
    """
    Entrée du rapport final.
    """

    result: ExecutionResult
    weight: float
    capital: float
    score: float
    family: str = "unknown"
    profile_name: str = "unknown"
    rank: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "weight", max(0.0, float(self.weight)))
        object.__setattr__(self, "capital", max(0.0, float(self.capital)))
        object.__setattr__(self, "score", max(0.0, min(1.0, float(self.score))))
        object.__setattr__(self, "family", str(self.family).strip().lower() or "unknown")
        object.__setattr__(self, "profile_name", str(self.profile_name).strip().lower() or "unknown")
        object.__setattr__(self, "rank", max(0, _coerce_int(self.rank, 0)))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def subject_fingerprint(self) -> str:
        return _result_key(self.result)

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject_fingerprint": self.subject_fingerprint,
            "weight": self.weight,
            "capital": self.capital,
            "score": self.score,
            "family": self.family,
            "profile_name": self.profile_name,
            "rank": self.rank,
            "result": self.result.to_dict(summary_only=True),
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class PortfolioReport:
    """
    Rapport final du portefeuille.
    """

    name: str = "portfolio"
    created_at: datetime = field(default_factory=_utc_now)

    entries: list[PortfolioReportEntry] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)

    allocation: PortfolioAllocation | None = None
    selection: PortfolioSelection | None = None
    risk: PortfolioRiskAssessment | None = None
    diversification: DiversificationAssessment | None = None
    correlation: PortfolioCorrelationMatrix | None = None
    capital_plan: CapitalPlan | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    selected_count: int = 0
    rejected_count: int = 0
    family_counts: Counter[str] = field(default_factory=Counter)
    profile_counts: Counter[str] = field(default_factory=Counter)

    best_score: float = float("-inf")
    best_subject_fingerprint: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", str(self.name).strip() or "portfolio")
        object.__setattr__(self, "entries", list(self.entries))
        object.__setattr__(self, "rejected", list(self.rejected))
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "family_counts", Counter(self.family_counts))
        object.__setattr__(self, "profile_counts", Counter(self.profile_counts))

    @property
    def total_capital(self) -> float:
        return float(sum(entry.capital for entry in self.entries))

    @property
    def total_weight(self) -> float:
        return float(sum(entry.weight for entry in self.entries))

    @property
    def average_score(self) -> float:
        if not self.entries:
            return 0.0
        return float(sum(entry.score for entry in self.entries) / len(self.entries))

    @property
    def best_entry(self) -> PortfolioReportEntry | None:
        if not self.entries:
            return None
        return max(self.entries, key=lambda item: (item.score, item.capital, item.weight))

    @property
    def corpus(self) -> tuple[ExecutionResult, ...]:
        return tuple(entry.result for entry in self.entries)

    def add_entry(self, entry: PortfolioReportEntry) -> None:
        self.entries.append(entry)
        self.selected_count += 1
        self.family_counts[entry.family] += 1
        self.profile_counts[entry.profile_name] += 1
        if entry.score > self.best_score:
            self.best_score = entry.score
            self.best_subject_fingerprint = entry.subject_fingerprint

    def add_rejected(self, entry: Mapping[str, Any]) -> None:
        self.rejected.append(dict(entry))
        self.rejected_count += 1

    def to_dict(self, *, summary_only: bool = False) -> dict[str, Any]:
        payload = {
            "name": self.name,
            "created_at": self.created_at.isoformat(),
            "entries": [] if summary_only else [entry.to_dict() for entry in self.entries],
            "rejected": [] if summary_only else [dict(item) for item in self.rejected],
            "selected_count": self.selected_count,
            "rejected_count": self.rejected_count,
            "total_capital": self.total_capital,
            "total_weight": self.total_weight,
            "average_score": self.average_score,
            "best_score": None if self.best_score == float("-inf") else self.best_score,
            "best_subject_fingerprint": self.best_subject_fingerprint,
            "family_counts": dict(self.family_counts),
            "profile_counts": dict(self.profile_counts),
            "allocation": None if self.allocation is None else self.allocation.to_dict(),
            "selection": None if self.selection is None else self.selection.to_dict(),
            "risk": None if self.risk is None else self.risk.to_dict(),
            "diversification": None if self.diversification is None else self.diversification.to_dict(),
            "correlation": None if self.correlation is None else self.correlation.to_dict(),
            "capital_plan": None if self.capital_plan is None else self.capital_plan.to_dict(),
            "metadata": dict(self.metadata),
        }
        return payload

    def __iter__(self):
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    def __repr__(self) -> str:
        return (
            "PortfolioReport("
            f"name='{self.name}', "
            f"selected={self.selected_count}, "
            f"capital={self.total_capital:.4f}"
            ")"
        )


class PortfolioReporter:
    """
    Construit un rapport final de portefeuille.
    """

    def __init__(self, *, config: Any | None = None) -> None:
        self._config = config

    def build(
        self,
        *,
        allocation: PortfolioAllocation | None = None,
        selection: PortfolioSelection | None = None,
        risk: PortfolioRiskAssessment | None = None,
        diversification: DiversificationAssessment | None = None,
        correlation: PortfolioCorrelationMatrix | None = None,
        capital_plan: CapitalPlan | None = None,
        rejected: Iterable[Mapping[str, Any]] | None = None,
        name: str = "portfolio",
        metadata: Mapping[str, Any] | None = None,
    ) -> PortfolioReport:
        report = PortfolioReport(
            name=name,
            allocation=allocation,
            selection=selection,
            risk=risk,
            diversification=diversification,
            correlation=correlation,
            capital_plan=capital_plan,
            metadata=dict(metadata or {}),
        )

        if selection is not None:
            for idx, item in enumerate(selection.selected):
                report.add_entry(
                    PortfolioReportEntry(
                        result=item.result,
                        weight=float(getattr(item, "target_weight", getattr(item, "score", 0.0))),
                        capital=float(getattr(item, "capital", 0.0)),
                        score=float(getattr(item, "score", 0.0)),
                        family=getattr(item, "family", _family_key(item.result)),
                        profile_name=getattr(item, "profile_name", _profile_name(item.result)),
                        rank=idx,
                        metadata=_to_mapping(getattr(item, "metadata", {})),
                    )
                )

        if allocation is not None and not report.entries:
            for idx, item in enumerate(allocation.entries):
                if not item.accepted:
                    continue
                report.add_entry(
                    PortfolioReportEntry(
                        result=item.result,
                        weight=float(item.target_weight),
                        capital=float(item.capital),
                        score=float(item.score),
                        family=item.family,
                        profile_name=item.profile_name,
                        rank=idx,
                        metadata=_to_mapping(item.metadata),
                    )
                )

        if rejected is not None:
            for item in rejected:
                report.add_rejected(item)

        if allocation is not None:
            report.allocation = allocation
        if selection is not None:
            report.selection = selection
        if risk is not None:
            report.risk = risk
        if diversification is not None:
            report.diversification = diversification
        if correlation is not None:
            report.correlation = correlation
        if capital_plan is not None:
            report.capital_plan = capital_plan

        return report

    def summarize(self, **kwargs: Any) -> PortfolioReport:
        return self.build(**kwargs)

    def __repr__(self) -> str:
        return "PortfolioReporter()"
