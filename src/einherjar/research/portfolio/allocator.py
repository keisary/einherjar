# portfolio/allocator.py
"""
==========================================================
Portfolio Allocator
==========================================================

Transforme une sélection d'Einhers en allocation finale.

Le module combine :
- les résultats d'exécution,
- le capital disponible,
- les contraintes de risque,
- la diversification,
- et renvoie une allocation prête à être figée.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from execution.execution_report import ExecutionResult
from execution.mae_mfe import MAEMFESummary
from execution.profiler import ExecutionProfile

from .capital import CapitalManager
from .capital import CapitalPlan
from .capital import CapitalSettings
from .correlation import PortfolioCorrelationMatrix
from .diversification import DiversificationAssessment
from .risk import PortfolioRiskAssessment

try:  # optional config module
    from config.portfolio import PortfolioConfig  # type: ignore
except Exception:  # pragma: no cover
    PortfolioConfig = Any  # type: ignore[misc,assignment]

__all__ = [
    "PortfolioAllocatorSettings",
    "PortfolioAllocationEntry",
    "PortfolioAllocation",
    "PortfolioAllocator",
]


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


def _safe_std(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(np.std(np.asarray(values, dtype=float)))


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


def _base_score(result: ExecutionResult) -> float:
    metrics = result.replay.metrics
    profile = getattr(result, "profile", None)
    mae_mfe = getattr(result, "mae_mfe", None)
    diagnostics = getattr(result, "diagnostics", None)

    value = (
        0.25 * _bounded_unit(0.5 * (1.0 + np.tanh(metrics.total_pnl)))
        + 0.15 * _bounded_unit(0.5 * (1.0 + np.tanh(metrics.expectancy)))
        + 0.15 * _bounded_unit(metrics.win_rate)
        + 0.12 * (_bounded_unit(1.0 - np.exp(-max(0.0, metrics.profit_factor - 1.0))) if np.isfinite(metrics.profit_factor) else 0.0)
        + 0.10 * _bounded_unit(metrics.signal_coverage)
        + 0.08 * (1.0 / (1.0 + max(0.0, metrics.max_drawdown)))
        + 0.08 * (1.0 if diagnostics is None or bool(getattr(diagnostics, "healthy", True)) else 0.75)
    )

    if profile is not None:
        value += 0.04 * (1.0 / (1.0 + max(0.0, float(getattr(profile, "max_drawdown", 0.0)))))

    if mae_mfe is not None:
        ratio = float(getattr(mae_mfe, "avg_mfe_to_mae_ratio", 0.0))
        value += 0.03 * _bounded_unit(1.0 - np.exp(-max(0.0, ratio)))

    return float(max(0.0, min(1.0, value)))


def _normalize_results(subjects: Iterable[Any]) -> tuple[ExecutionResult, ...]:
    results: list[ExecutionResult] = []
    for item in subjects:
        if isinstance(item, ExecutionResult):
            results.append(item)
            continue
        if hasattr(item, "result") and isinstance(item.result, ExecutionResult):
            results.append(item.result)
            continue
        if hasattr(item, "execution_result") and isinstance(item.execution_result, ExecutionResult):
            results.append(item.execution_result)
            continue
        raise TypeError("PortfolioAllocator expects ExecutionResult objects or entries exposing a result.")
    return tuple(results)


@dataclass(frozen=True, slots=True)
class PortfolioAllocatorSettings:
    """
    Paramètres de l'allocator.
    """

    max_selected: int = 12
    min_weight: float = 0.0
    max_weight: float = 0.35
    normalize_weights: bool = True
    use_scores: bool = True
    use_risk_adjustment: bool = True
    use_diversification_adjustment: bool = True
    total_capital: float = 1.0

    score_floor: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_selected", max(1, _coerce_int(self.max_selected, 12)))
        object.__setattr__(self, "min_weight", _bounded_unit(_coerce_float(self.min_weight, 0.0)))
        object.__setattr__(self, "max_weight", _bounded_unit(_coerce_float(self.max_weight, 0.35)))
        object.__setattr__(self, "normalize_weights", _coerce_bool(self.normalize_weights, True))
        object.__setattr__(self, "use_scores", _coerce_bool(self.use_scores, True))
        object.__setattr__(self, "use_risk_adjustment", _coerce_bool(self.use_risk_adjustment, True))
        object.__setattr__(self, "use_diversification_adjustment", _coerce_bool(self.use_diversification_adjustment, True))
        object.__setattr__(self, "total_capital", max(0.0, _coerce_float(self.total_capital, 1.0)))
        object.__setattr__(self, "score_floor", _bounded_unit(_coerce_float(self.score_floor, 0.0)))

    @classmethod
    def from_config(cls, config: Any | None) -> "PortfolioAllocatorSettings":
        if config is None:
            return cls()

        root = _to_mapping(config)
        port = _to_mapping(root.get("portfolio", root.get("portfolio_config", {})))
        alloc = _to_mapping(port.get("allocator", port.get("allocation", {})))

        return cls(
            max_selected=_coerce_int(alloc.get("max_selected", root.get("max_selected", 12)), 12),
            min_weight=_coerce_float(alloc.get("min_weight", root.get("min_weight", 0.0)), 0.0),
            max_weight=_coerce_float(alloc.get("max_weight", root.get("max_weight", 0.35)), 0.35),
            normalize_weights=_coerce_bool(alloc.get("normalize_weights", root.get("normalize_weights", True)), True),
            use_scores=_coerce_bool(alloc.get("use_scores", root.get("use_scores", True)), True),
            use_risk_adjustment=_coerce_bool(alloc.get("use_risk_adjustment", root.get("use_risk_adjustment", True)), True),
            use_diversification_adjustment=_coerce_bool(alloc.get("use_diversification_adjustment", root.get("use_diversification_adjustment", True)), True),
            total_capital=_coerce_float(alloc.get("total_capital", root.get("total_capital", 1.0)), 1.0),
            score_floor=_coerce_float(alloc.get("score_floor", root.get("score_floor", 0.0)), 0.0),
        )


@dataclass(frozen=True, slots=True)
class PortfolioAllocationEntry:
    """
    Allocation finale d'un Einher.
    """

    result: ExecutionResult
    raw_weight: float
    target_weight: float
    capital: float
    score: float
    family: str = "unknown"
    profile_name: str = "unknown"
    accepted: bool = True
    rank: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "raw_weight", max(0.0, float(self.raw_weight)))
        object.__setattr__(self, "target_weight", max(0.0, float(self.target_weight)))
        object.__setattr__(self, "capital", max(0.0, float(self.capital)))
        object.__setattr__(self, "score", _bounded_unit(self.score))
        object.__setattr__(self, "family", str(self.family).strip().lower() or "unknown")
        object.__setattr__(self, "profile_name", str(self.profile_name).strip().lower() or "unknown")
        object.__setattr__(self, "accepted", _coerce_bool(self.accepted, True))
        object.__setattr__(self, "rank", max(0, _coerce_int(self.rank, 0)))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def subject_fingerprint(self) -> str:
        return _result_key(self.result)

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject_fingerprint": self.subject_fingerprint,
            "raw_weight": self.raw_weight,
            "target_weight": self.target_weight,
            "capital": self.capital,
            "score": self.score,
            "family": self.family,
            "profile_name": self.profile_name,
            "accepted": self.accepted,
            "rank": self.rank,
            "result": self.result.to_dict(summary_only=True),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class PortfolioAllocation:
    """
    Allocation finale du portefeuille.
    """

    entries: tuple[PortfolioAllocationEntry, ...]
    capital_plan: CapitalPlan
    risk: PortfolioRiskAssessment | None = None
    diversification: DiversificationAssessment | None = None
    correlation: PortfolioCorrelationMatrix | None = None

    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "entries", tuple(self.entries))
        object.__setattr__(self, "score", _bounded_unit(self.score))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def selected_results(self) -> tuple[ExecutionResult, ...]:
        return tuple(entry.result for entry in self.entries if entry.accepted and entry.capital > 0)

    @property
    def total_weight(self) -> float:
        return float(sum(entry.target_weight for entry in self.entries if entry.accepted))

    @property
    def total_capital(self) -> float:
        return float(sum(entry.capital for entry in self.entries if entry.accepted))

    @property
    def selected_count(self) -> int:
        return sum(1 for entry in self.entries if entry.accepted and entry.capital > 0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entries": [entry.to_dict() for entry in self.entries],
            "capital_plan": self.capital_plan.to_dict(),
            "risk": None if self.risk is None else self.risk.to_dict(),
            "diversification": None if self.diversification is None else self.diversification.to_dict(),
            "correlation": None if self.correlation is None else self.correlation.to_dict(),
            "score": self.score,
            "total_weight": self.total_weight,
            "total_capital": self.total_capital,
            "metadata": dict(self.metadata),
        }


class PortfolioAllocator:
    """
    Alloue le capital entre les Einhers retenus.
    """

    def __init__(
        self,
        settings: PortfolioAllocatorSettings | None = None,
        *,
        config: PortfolioConfig | Any | None = None,
        capital_manager: CapitalManager | None = None,
    ) -> None:
        self._settings = settings or PortfolioAllocatorSettings.from_config(config)
        self._capital_manager = capital_manager or CapitalManager(
            settings=CapitalSettings.from_config(config)
        )

    @property
    def settings(self) -> PortfolioAllocatorSettings:
        return self._settings

    @property
    def capital_manager(self) -> CapitalManager:
        return self._capital_manager

    def allocate(
        self,
        subjects: Iterable[Any],
        *,
        weights: Sequence[float] | Mapping[str, float] | None = None,
        total_capital: float | None = None,
        risk: PortfolioRiskAssessment | None = None,
        diversification: DiversificationAssessment | None = None,
        correlation: PortfolioCorrelationMatrix | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> PortfolioAllocation:
        results = _normalize_results(subjects)
        settings = self._settings

        if not results:
            capital_plan = self._capital_manager.plan(
                (),
                weights=(),
                total_capital=total_capital if total_capital is not None else settings.total_capital,
                metadata=metadata,
            )
            return PortfolioAllocation(
                entries=(),
                capital_plan=capital_plan,
                risk=risk,
                diversification=diversification,
                correlation=correlation,
                score=0.0,
                metadata=dict(metadata or {}),
            )

        raw_weights = self._resolve_raw_weights(results, weights)
        adjusted_weights = self._adjust_weights(
            results,
            raw_weights,
            risk=risk,
            diversification=diversification,
            correlation=correlation,
        )

        capital_plan = self._capital_manager.plan(
            results,
            weights=adjusted_weights,
            total_capital=total_capital if total_capital is not None else settings.total_capital,
            metadata=metadata,
        )

        entries: list[PortfolioAllocationEntry] = []
        for rank, (result, raw_weight, adjusted_weight, capital_entry) in enumerate(
            zip(results, raw_weights, adjusted_weights, capital_plan.entries)
        ):
            accepted = capital_entry.accepted and adjusted_weight >= settings.score_floor and rank < settings.max_selected
            score = self._entry_score(result, raw_weight, adjusted_weight, risk=risk, diversification=diversification)
            entries.append(
                PortfolioAllocationEntry(
                    result=result,
                    raw_weight=float(raw_weight),
                    target_weight=float(adjusted_weight),
                    capital=float(capital_entry.capital if accepted else 0.0),
                    score=score,
                    family=_family_key(result),
                    profile_name=_profile_name(result),
                    accepted=accepted,
                    rank=rank,
                    metadata={
                        "subject_fingerprint": _result_key(result),
                        "capital_plan": capital_entry.to_dict(),
                    },
                )
            )

        score = self._allocation_score(entries, risk=risk, diversification=diversification, correlation=correlation, capital_plan=capital_plan)

        return PortfolioAllocation(
            entries=tuple(entries),
            capital_plan=capital_plan,
            risk=risk,
            diversification=diversification,
            correlation=correlation,
            score=score,
            metadata=dict(metadata or {}),
        )

    def plan(
        self,
        subjects: Iterable[Any],
        *,
        weights: Sequence[float] | Mapping[str, float] | None = None,
        total_capital: float | None = None,
        risk: PortfolioRiskAssessment | None = None,
        diversification: DiversificationAssessment | None = None,
        correlation: PortfolioCorrelationMatrix | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> PortfolioAllocation:
        return self.allocate(
            subjects,
            weights=weights,
            total_capital=total_capital,
            risk=risk,
            diversification=diversification,
            correlation=correlation,
            metadata=metadata,
        )

    def _resolve_raw_weights(
        self,
        results: Sequence[ExecutionResult],
        weights: Sequence[float] | Mapping[str, float] | None,
    ) -> np.ndarray:
        if weights is None:
            raw = np.asarray([_base_score(result) for result in results], dtype=float)
        elif isinstance(weights, Mapping):
            raw_values = []
            for result in results:
                key = _result_key(result)
                raw_values.append(float(weights.get(key, weights.get(_profile_name(result), 0.0))))
            raw = np.asarray(raw_values, dtype=float)
        else:
            raw = np.asarray(list(weights), dtype=float)

        if raw.size != len(results):
            raise ValueError("weights must match subjects length.")

        raw = np.maximum(raw, 0.0)
        if raw.sum() <= 1e-12:
            raw = np.full(len(results), 1.0 / len(results), dtype=float)
        else:
            raw = raw / raw.sum()
        return raw

    def _adjust_weights(
        self,
        results: Sequence[ExecutionResult],
        raw_weights: np.ndarray,
        *,
        risk: PortfolioRiskAssessment | None = None,
        diversification: DiversificationAssessment | None = None,
        correlation: PortfolioCorrelationMatrix | None = None,
    ) -> np.ndarray:
        settings = self._settings
        adjusted = raw_weights.copy()

        if settings.use_scores:
            scores = np.asarray([_base_score(result) for result in results], dtype=float)
            scores = np.maximum(scores, 0.0)
            if scores.sum() > 1e-12:
                scores = scores / scores.sum()
                adjusted = 0.65 * adjusted + 0.35 * scores

        if risk is not None and settings.use_risk_adjustment:
            risk_factor = 0.5 + 0.5 * risk.score
            if not risk.acceptable:
                risk_factor *= 0.85
            adjusted = adjusted * risk_factor

        if diversification is not None and settings.use_diversification_adjustment:
            div_factor = 0.5 + 0.5 * diversification.score
            if not diversification.diversified:
                div_factor *= 0.90
            adjusted = adjusted * div_factor

        if correlation is not None and correlation.pairs:
            penalty = 1.0 - min(0.50, float(np.mean([abs(pair.correlation) for pair in correlation.pairs])))
            adjusted = adjusted * max(0.50, penalty)

        adjusted = np.maximum(adjusted, settings.min_weight)
        total = float(adjusted.sum())
        if total <= 1e-12:
            adjusted = np.full(len(results), 1.0 / len(results), dtype=float)
        else:
            adjusted = adjusted / total

        adjusted = np.minimum(adjusted, settings.max_weight)
        total = float(adjusted.sum())
        if total > 1e-12:
            adjusted = adjusted / total
        return adjusted

    def _entry_score(
        self,
        result: ExecutionResult,
        raw_weight: float,
        adjusted_weight: float,
        *,
        risk: PortfolioRiskAssessment | None,
        diversification: DiversificationAssessment | None,
    ) -> float:
        metrics = result.replay.metrics
        profile = getattr(result, "profile", None)
        mae_mfe = getattr(result, "mae_mfe", None)
        diagnostics = getattr(result, "diagnostics", None)

        score = _base_score(result)
        score = 0.55 * score + 0.20 * _bounded_unit(adjusted_weight) + 0.10 * _bounded_unit(raw_weight)

        if profile is not None:
            score += 0.05 * (1.0 / (1.0 + max(0.0, float(getattr(profile, "max_drawdown", 0.0)))))
            score += 0.03 * _bounded_unit(float(getattr(profile, "recovery_factor", 0.0)) if np.isfinite(float(getattr(profile, "recovery_factor", 0.0))) else 0.0)

        if mae_mfe is not None:
            ratio = float(getattr(mae_mfe, "avg_mfe_to_mae_ratio", 0.0))
            score += 0.04 * _bounded_unit(1.0 - np.exp(-max(0.0, ratio)))

        if diagnostics is not None and not bool(getattr(diagnostics, "healthy", True)):
            score *= 0.90

        if risk is not None:
            score *= 0.85 + 0.15 * risk.score
        if diversification is not None:
            score *= 0.85 + 0.15 * diversification.score

        return float(max(0.0, min(1.0, score)))

    def _allocation_score(
        self,
        entries: Sequence[PortfolioAllocationEntry],
        *,
        risk: PortfolioRiskAssessment | None,
        diversification: DiversificationAssessment | None,
        correlation: PortfolioCorrelationMatrix | None,
        capital_plan: CapitalPlan,
    ) -> float:
        if not entries:
            return 0.0

        accepted = [entry for entry in entries if entry.accepted and entry.capital > 0]
        if not accepted:
            return 0.0

        weights = np.asarray([entry.target_weight for entry in accepted], dtype=float)
        scores = np.asarray([entry.score for entry in accepted], dtype=float)
        capitals = np.asarray([entry.capital for entry in accepted], dtype=float)

        concentration = float(np.sum(weights ** 2)) if weights.size else 1.0
        mean_score = _safe_mean(scores.tolist())
        capital_utilization = capital_plan.utilization_ratio

        base = 0.45 * mean_score + 0.15 * capital_utilization + 0.15 * (1.0 - concentration)

        if risk is not None:
            base += 0.15 * risk.score
        if diversification is not None:
            base += 0.10 * diversification.score
        if correlation is not None and correlation.pairs:
            avg_abs = float(np.mean([abs(pair.correlation) for pair in correlation.pairs]))
            base += 0.10 * (1.0 - avg_abs)

        return float(max(0.0, min(1.0, base)))

    def __repr__(self) -> str:
        return (
            "PortfolioAllocator("
            f"max_selected={self._settings.max_selected}, "
            f"max_weight={self._settings.max_weight}"
            ")"
        )