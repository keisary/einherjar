# portfolio/optimizer.py
"""
==========================================================
Portfolio Optimizer
==========================================================

Cherche la meilleure combinaison de portefeuille à partir
des Einhers déjà sélectionnés.

L'optimiseur n'invente pas de nouvelles stratégies :
- il teste des règles de pondération,
- il compare les scores de portefeuille,
- il garde la meilleure allocation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from execution.execution_report import ExecutionResult

from .allocator import PortfolioAllocation
from .allocator import PortfolioAllocator
from .capital import CapitalSettings
from .correlation import PortfolioCorrelationAnalyzer
from .correlation import PortfolioCorrelationMatrix
from .diversification import DiversificationAssessment
from .diversification import DiversificationEngine
from .risk import PortfolioRiskAssessment
from .risk import PortfolioRiskModel

try:  # optional config module
    from config.portfolio import PortfolioConfig  # type: ignore
except Exception:  # pragma: no cover
    PortfolioConfig = Any  # type: ignore[misc,assignment]

__all__ = [
    "PortfolioOptimizationSettings",
    "PortfolioOptimizationTrial",
    "PortfolioOptimizationResult",
    "PortfolioOptimizer",
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


def _extract_results(subjects: Iterable[Any]) -> tuple[ExecutionResult, ...]:
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
        raise TypeError("PortfolioOptimizer expects ExecutionResult objects or entries exposing a result.")
    return tuple(results)


def _score_result(result: ExecutionResult) -> float:
    metrics = result.replay.metrics
    profile = getattr(result, "profile", None)
    mae_mfe = getattr(result, "mae_mfe", None)
    diagnostics = getattr(result, "diagnostics", None)

    score = (
        0.25 * _bounded_unit(0.5 * (1.0 + np.tanh(metrics.total_pnl)))
        + 0.15 * _bounded_unit(0.5 * (1.0 + np.tanh(metrics.expectancy)))
        + 0.15 * _bounded_unit(metrics.win_rate)
        + 0.12 * (_bounded_unit(1.0 - np.exp(-max(0.0, metrics.profit_factor - 1.0))) if np.isfinite(metrics.profit_factor) else 0.0)
        + 0.10 * _bounded_unit(metrics.signal_coverage)
        + 0.08 * (1.0 / (1.0 + max(0.0, metrics.max_drawdown)))
        + 0.08 * (1.0 if diagnostics is None or bool(getattr(diagnostics, "healthy", True)) else 0.75)
    )
    if profile is not None:
        score += 0.04 * (1.0 / (1.0 + max(0.0, float(getattr(profile, "max_drawdown", 0.0)))))
        score += 0.03 * _bounded_unit(float(getattr(profile, "recovery_factor", 0.0)) if np.isfinite(float(getattr(profile, "recovery_factor", 0.0))) else 0.0)
    if mae_mfe is not None:
        ratio = float(getattr(mae_mfe, "avg_mfe_to_mae_ratio", 0.0))
        score += 0.04 * _bounded_unit(1.0 - np.exp(-max(0.0, ratio)))
    return float(max(0.0, min(1.0, score)))


@dataclass(frozen=True, slots=True)
class PortfolioOptimizationSettings:
    """
    Paramètres de l'optimiseur de portefeuille.
    """

    max_trials: int = 12
    top_k: int = 1

    use_equal: bool = True
    use_performance: bool = True
    use_risk_adjusted: bool = True
    use_diversified: bool = True
    use_conservative: bool = True

    min_improvement: float = 0.0
    total_capital: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_trials", max(1, _coerce_int(self.max_trials, 12)))
        object.__setattr__(self, "top_k", max(1, _coerce_int(self.top_k, 1)))
        object.__setattr__(self, "use_equal", _coerce_bool(self.use_equal, True))
        object.__setattr__(self, "use_performance", _coerce_bool(self.use_performance, True))
        object.__setattr__(self, "use_risk_adjusted", _coerce_bool(self.use_risk_adjusted, True))
        object.__setattr__(self, "use_diversified", _coerce_bool(self.use_diversified, True))
        object.__setattr__(self, "use_conservative", _coerce_bool(self.use_conservative, True))
        object.__setattr__(self, "min_improvement", max(0.0, _coerce_float(self.min_improvement, 0.0)))
        object.__setattr__(self, "total_capital", max(0.0, _coerce_float(self.total_capital, 1.0)))

    @classmethod
    def from_config(cls, config: Any | None) -> "PortfolioOptimizationSettings":
        if config is None:
            return cls()

        root = _to_mapping(config)
        port = _to_mapping(root.get("portfolio", root.get("portfolio_config", {})))
        opt = _to_mapping(port.get("optimizer", port.get("optimization", {})))

        return cls(
            max_trials=_coerce_int(opt.get("max_trials", root.get("max_trials", 12)), 12),
            top_k=_coerce_int(opt.get("top_k", root.get("top_k", 1)), 1),
            use_equal=_coerce_bool(opt.get("use_equal", True), True),
            use_performance=_coerce_bool(opt.get("use_performance", True), True),
            use_risk_adjusted=_coerce_bool(opt.get("use_risk_adjusted", True), True),
            use_diversified=_coerce_bool(opt.get("use_diversified", True), True),
            use_conservative=_coerce_bool(opt.get("use_conservative", True), True),
            min_improvement=_coerce_float(opt.get("min_improvement", root.get("min_improvement", 0.0)), 0.0),
            total_capital=_coerce_float(opt.get("total_capital", root.get("total_capital", 1.0)), 1.0),
        )


@dataclass(frozen=True, slots=True)
class PortfolioOptimizationTrial:
    """
    Essai d'optimisation.
    """

    mode: str
    allocation: PortfolioAllocation
    score: float
    risk: PortfolioRiskAssessment | None = None
    diversification: DiversificationAssessment | None = None
    correlation: PortfolioCorrelationMatrix | None = None
    rank: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", str(self.mode).strip().lower() or "unknown")
        object.__setattr__(self, "score", _bounded_unit(self.score))
        object.__setattr__(self, "rank", max(0, _coerce_int(self.rank, 0)))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "score": self.score,
            "allocation": self.allocation.to_dict(),
            "risk": None if self.risk is None else self.risk.to_dict(),
            "diversification": None if self.diversification is None else self.diversification.to_dict(),
            "correlation": None if self.correlation is None else self.correlation.to_dict(),
            "rank": self.rank,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class PortfolioOptimizationResult:
    """
    Résultat global d'une optimisation.
    """

    best_trial: PortfolioOptimizationTrial | None
    trials: tuple[PortfolioOptimizationTrial, ...]
    best_allocation: PortfolioAllocation | None = None
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "trials", tuple(self.trials))
        object.__setattr__(self, "score", _bounded_unit(self.score))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "best_trial": None if self.best_trial is None else self.best_trial.to_dict(),
            "trials": [trial.to_dict() for trial in self.trials],
            "best_allocation": None if self.best_allocation is None else self.best_allocation.to_dict(),
            "score": self.score,
            "metadata": dict(self.metadata),
        }


class PortfolioOptimizer:
    """
    Explore plusieurs modes d'allocation et garde le meilleur.
    """

    def __init__(
        self,
        settings: PortfolioOptimizationSettings | None = None,
        *,
        config: PortfolioConfig | Any | None = None,
        allocator: PortfolioAllocator | None = None,
        risk_model: PortfolioRiskModel | None = None,
        diversification_engine: DiversificationEngine | None = None,
        correlation_analyzer: PortfolioCorrelationAnalyzer | None = None,
    ) -> None:
        self._settings = settings or PortfolioOptimizationSettings.from_config(config)
        self._allocator = allocator or PortfolioAllocator(config=config)
        self._risk_model = risk_model or PortfolioRiskModel(config=config)
        self._diversification_engine = diversification_engine or DiversificationEngine(config=config)
        self._correlation_analyzer = correlation_analyzer or PortfolioCorrelationAnalyzer()

    @property
    def settings(self) -> PortfolioOptimizationSettings:
        return self._settings

    @property
    def allocator(self) -> PortfolioAllocator:
        return self._allocator

    def optimize(
        self,
        subjects: Iterable[Any],
        *,
        weights: Sequence[float] | Mapping[str, float] | None = None,
        total_capital: float | None = None,
        correlation: PortfolioCorrelationMatrix | None = None,
        diversification: DiversificationAssessment | None = None,
        risk: PortfolioRiskAssessment | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> PortfolioOptimizationResult:
        results = _extract_results(subjects)
        if not results:
            return PortfolioOptimizationResult(
                best_trial=None,
                trials=(),
                best_allocation=None,
                score=0.0,
                metadata=dict(metadata or {}),
            )

        correlation = correlation or self._correlation_analyzer.correlate(results)
        diversification = diversification or self._diversification_engine.assess(results, correlation=correlation)
        risk = risk or self._risk_model.assess(results, correlation=correlation, diversification=diversification)

        modes = self._candidate_modes()
        trials: list[PortfolioOptimizationTrial] = []

        for rank, mode in enumerate(modes):
            if len(trials) >= self._settings.max_trials:
                break

            mode_weights = self._mode_weights(results, mode=mode, base_weights=weights)
            allocation = self._allocator.allocate(
                results,
                weights=mode_weights,
                total_capital=total_capital if total_capital is not None else self._settings.total_capital,
                risk=risk,
                diversification=diversification,
                correlation=correlation,
                metadata=metadata,
            )
            score = self._trial_score(allocation, risk=risk, diversification=diversification, correlation=correlation, mode=mode)
            trial = PortfolioOptimizationTrial(
                mode=mode,
                allocation=allocation,
                score=score,
                risk=risk,
                diversification=diversification,
                correlation=correlation,
                rank=rank,
                metadata={
                    "selected_count": allocation.selected_count,
                    "total_capital": allocation.total_capital,
                },
            )
            trials.append(trial)

        ordered = tuple(sorted(trials, key=lambda item: (-item.score, -item.allocation.score, item.rank, item.mode)))
        best_trial = ordered[0] if ordered else None
        best_allocation = best_trial.allocation if best_trial is not None else None
        best_score = best_trial.score if best_trial is not None else 0.0

        if best_trial is not None and self._settings.min_improvement > 0 and best_trial.score < self._settings.min_improvement:
            # on garde quand même le meilleur, mais le score final reflète la faiblesse du gain
            best_score = best_trial.score

        return PortfolioOptimizationResult(
            best_trial=best_trial,
            trials=ordered[: self._settings.top_k],
            best_allocation=best_allocation,
            score=best_score,
            metadata=dict(metadata or {}),
        )

    def tune(self, subjects: Iterable[Any], **kwargs: Any) -> PortfolioOptimizationResult:
        return self.optimize(subjects, **kwargs)

    def search(self, subjects: Iterable[Any], **kwargs: Any) -> PortfolioOptimizationResult:
        return self.optimize(subjects, **kwargs)

    def _candidate_modes(self) -> tuple[str, ...]:
        modes: list[str] = []
        if self._settings.use_equal:
            modes.append("equal")
        if self._settings.use_performance:
            modes.append("performance")
        if self._settings.use_risk_adjusted:
            modes.append("risk_adjusted")
        if self._settings.use_diversified:
            modes.append("diversified")
        if self._settings.use_conservative:
            modes.append("conservative")
        return tuple(modes or ["equal"])

    def _mode_weights(
        self,
        results: Sequence[ExecutionResult],
        *,
        mode: str,
        base_weights: Sequence[float] | Mapping[str, float] | None,
    ) -> np.ndarray:
        if base_weights is None:
            base = np.asarray([_score_result(result) for result in results], dtype=float)
        elif isinstance(base_weights, Mapping):
            arr = []
            for result in results:
                key = _result_key(result)
                arr.append(float(base_weights.get(key, base_weights.get(_profile_name(result), 0.0))))
            base = np.asarray(arr, dtype=float)
        else:
            base = np.asarray(list(base_weights), dtype=float)

        if base.size != len(results):
            raise ValueError("weights must match results length.")

        base = np.maximum(base, 0.0)
        if base.sum() <= 1e-12:
            base = np.full(len(results), 1.0 / len(results), dtype=float)
        else:
            base = base / base.sum()

        metrics = np.asarray([_score_result(result) for result in results], dtype=float)

        if mode == "equal":
            weights = np.full(len(results), 1.0 / len(results), dtype=float)
        elif mode == "performance":
            weights = metrics
        elif mode == "risk_adjusted":
            weights = base * (0.5 + 0.5 * metrics)
        elif mode == "diversified":
            family_counts = {}
            profile_counts = {}
            for result in results:
                family_counts[_family_key(result)] = family_counts.get(_family_key(result), 0) + 1
                profile_counts[_profile_name(result)] = profile_counts.get(_profile_name(result), 0) + 1
            weights = np.asarray(
                [
                    _score_result(result)
                    / max(1.0, family_counts.get(_family_key(result), 1))
                    / max(1.0, profile_counts.get(_profile_name(result), 1))
                    for result in results
                ],
                dtype=float,
            )
        elif mode == "conservative":
            weights = np.asarray(
                [
                    _score_result(result)
                    * (1.0 / (1.0 + max(0.0, result.replay.metrics.max_drawdown)))
                    * (1.0 if (result.diagnostics is None or bool(getattr(result.diagnostics, "healthy", True))) else 0.8)
                    for result in results
                ],
                dtype=float,
            )
        else:
            weights = base

        weights = np.maximum(weights, 0.0)
        if weights.sum() <= 1e-12:
            weights = np.full(len(results), 1.0 / len(results), dtype=float)
        else:
            weights = weights / weights.sum()
        return weights

    def _trial_score(
        self,
        allocation: PortfolioAllocation,
        *,
        risk: PortfolioRiskAssessment,
        diversification: DiversificationAssessment,
        correlation: PortfolioCorrelationMatrix,
        mode: str,
    ) -> float:
        score = allocation.score
        score = 0.35 * score + 0.25 * risk.score + 0.20 * diversification.score
        if correlation.pairs:
            avg_abs = float(np.mean([abs(pair.correlation) for pair in correlation.pairs]))
            score += 0.10 * (1.0 - avg_abs)
        score += 0.10 * allocation.capital_plan.utilization_ratio

        if mode == "conservative":
            score += 0.03
        elif mode == "diversified":
            score += 0.04 * diversification.score
        elif mode == "risk_adjusted":
            score += 0.04 * risk.score

        return float(max(0.0, min(1.0, score)))

    def __repr__(self) -> str:
        return (
            "PortfolioOptimizer("
            f"max_trials={self._settings.max_trials}, "
            f"top_k={self._settings.top_k}"
            ")"
        )