# portfolio/risk.py
"""
==========================================================
Portfolio Risk
==========================================================

Évalue le risque d'un portefeuille d'Einhers déjà sélectionnés.

Le module ne choisit pas les stratégies :
- il analyse la concentration,
- l'exposition,
- la corrélation,
- la robustesse de la répartition,
- et produit un verdict de risque.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from execution.execution_report import ExecutionResult
from execution.profiler import ExecutionProfile

from .correlation import PortfolioCorrelationMatrix
from .diversification import DiversificationAssessment

try:  # optional config module
    from config.portfolio import PortfolioConfig  # type: ignore
except Exception:  # pragma: no cover
    PortfolioConfig = Any  # type: ignore[misc,assignment]

__all__ = [
    "PortfolioRiskSettings",
    "PortfolioRiskAssessment",
    "PortfolioRiskModel",
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


def _safe_max(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(np.max(np.asarray(values, dtype=float)))


def _safe_min(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(np.min(np.asarray(values, dtype=float)))


def _portfolio_weights(
    results: Sequence[ExecutionResult],
    weights: Sequence[float] | Mapping[str, float] | None,
) -> np.ndarray:
    if not results:
        return np.asarray([], dtype=float)

    if weights is None:
        return np.full(len(results), 1.0 / len(results), dtype=float)

    if isinstance(weights, Mapping):
        arr = []
        for result in results:
            key = str(getattr(result, "subject_fingerprint", "") or getattr(result.execution_fingerprint, "digest", "") or "")
            profile_name = _profile_name(result)
            arr.append(float(weights.get(key, weights.get(profile_name, 0.0))))
        arr = np.asarray(arr, dtype=float)
    else:
        arr = np.asarray(list(weights), dtype=float)

    if arr.size != len(results):
        raise ValueError("weights must match results length.")

    arr = np.maximum(arr, 0.0)
    total = float(arr.sum())
    if total <= 1e-12:
        return np.full(len(results), 1.0 / len(results), dtype=float)
    return arr / total


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


@dataclass(frozen=True, slots=True)
class PortfolioRiskSettings:
    """
    Paramètres de risque portefeuille.
    """

    min_score: float = 0.55
    max_gross_exposure: float = 1.0
    max_single_weight: float = 0.35
    max_hhi: float = 0.35
    min_effective_bets: float = 2.0

    max_average_abs_correlation: float = 0.60
    max_drawdown: float = 0.40
    max_total_drawdown: float = 0.60

    min_expected_pnl: float = 0.0
    min_expected_win_rate: float = 0.0
    min_expected_profit_factor: float = 0.0

    require_diversified: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "min_score", _bounded_unit(_coerce_float(self.min_score, 0.55)))
        object.__setattr__(self, "max_gross_exposure", max(0.0, _coerce_float(self.max_gross_exposure, 1.0)))
        object.__setattr__(self, "max_single_weight", _bounded_unit(_coerce_float(self.max_single_weight, 0.35)))
        object.__setattr__(self, "max_hhi", _bounded_unit(_coerce_float(self.max_hhi, 0.35)))
        object.__setattr__(self, "min_effective_bets", max(1.0, _coerce_float(self.min_effective_bets, 2.0)))
        object.__setattr__(self, "max_average_abs_correlation", _bounded_unit(_coerce_float(self.max_average_abs_correlation, 0.60)))
        object.__setattr__(self, "max_drawdown", _bounded_unit(_coerce_float(self.max_drawdown, 0.40)))
        object.__setattr__(self, "max_total_drawdown", _bounded_unit(_coerce_float(self.max_total_drawdown, 0.60)))
        object.__setattr__(self, "min_expected_pnl", _coerce_float(self.min_expected_pnl, 0.0))
        object.__setattr__(self, "min_expected_win_rate", _bounded_unit(_coerce_float(self.min_expected_win_rate, 0.0)))
        object.__setattr__(self, "min_expected_profit_factor", max(0.0, _coerce_float(self.min_expected_profit_factor, 0.0)))
        object.__setattr__(self, "require_diversified", _coerce_bool(self.require_diversified, False))

    @classmethod
    def from_config(cls, config: Any | None) -> "PortfolioRiskSettings":
        if config is None:
            return cls()

        root = _to_mapping(config)
        port = _to_mapping(root.get("portfolio", root.get("portfolio_config", {})))
        risk = _to_mapping(port.get("risk", port.get("risk_management", {})))

        return cls(
            min_score=_coerce_float(risk.get("min_score", root.get("min_score", 0.55)), 0.55),
            max_gross_exposure=_coerce_float(risk.get("max_gross_exposure", root.get("max_gross_exposure", 1.0)), 1.0),
            max_single_weight=_coerce_float(risk.get("max_single_weight", root.get("max_single_weight", 0.35)), 0.35),
            max_hhi=_coerce_float(risk.get("max_hhi", root.get("max_hhi", 0.35)), 0.35),
            min_effective_bets=_coerce_float(risk.get("min_effective_bets", root.get("min_effective_bets", 2.0)), 2.0),
            max_average_abs_correlation=_coerce_float(risk.get("max_average_abs_correlation", root.get("max_average_abs_correlation", 0.60)), 0.60),
            max_drawdown=_coerce_float(risk.get("max_drawdown", root.get("max_drawdown", 0.40)), 0.40),
            max_total_drawdown=_coerce_float(risk.get("max_total_drawdown", root.get("max_total_drawdown", 0.60)), 0.60),
            min_expected_pnl=_coerce_float(risk.get("min_expected_pnl", root.get("min_expected_pnl", 0.0)), 0.0),
            min_expected_win_rate=_coerce_float(risk.get("min_expected_win_rate", root.get("min_expected_win_rate", 0.0)), 0.0),
            min_expected_profit_factor=_coerce_float(risk.get("min_expected_profit_factor", root.get("min_expected_profit_factor", 0.0)), 0.0),
            require_diversified=_coerce_bool(risk.get("require_diversified", root.get("require_diversified", False)), False),
        )


@dataclass(frozen=True, slots=True)
class PortfolioRiskAssessment:
    """
    Résultat de risque portefeuille.
    """

    acceptable: bool
    score: float

    selected_count: int
    total_weight: float
    max_weight: float
    hhi: float
    effective_bets: float

    expected_total_pnl: float
    expected_win_rate: float
    expected_profit_factor: float
    weighted_drawdown: float

    average_abs_correlation: float
    max_abs_correlation: float
    family_concentration: float
    profile_concentration: float

    concentration_penalty: float
    correlation_penalty: float
    drawdown_penalty: float

    reasons: tuple[str, ...] = ()
    recommendations: tuple[str, ...] = ()
    correlation: PortfolioCorrelationMatrix | None = None
    diversification: DiversificationAssessment | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "acceptable", _coerce_bool(self.acceptable, False))
        object.__setattr__(self, "score", _bounded_unit(self.score))
        object.__setattr__(self, "selected_count", max(0, _coerce_int(self.selected_count, 0)))
        object.__setattr__(self, "total_weight", max(0.0, float(self.total_weight)))
        object.__setattr__(self, "max_weight", max(0.0, float(self.max_weight)))
        object.__setattr__(self, "hhi", _bounded_unit(self.hhi))
        object.__setattr__(self, "effective_bets", max(0.0, float(self.effective_bets)))
        object.__setattr__(self, "expected_total_pnl", float(self.expected_total_pnl))
        object.__setattr__(self, "expected_win_rate", _bounded_unit(self.expected_win_rate))
        object.__setattr__(self, "expected_profit_factor", float(self.expected_profit_factor))
        object.__setattr__(self, "weighted_drawdown", _bounded_unit(self.weighted_drawdown))
        object.__setattr__(self, "average_abs_correlation", _bounded_unit(self.average_abs_correlation))
        object.__setattr__(self, "max_abs_correlation", _bounded_unit(self.max_abs_correlation))
        object.__setattr__(self, "family_concentration", _bounded_unit(self.family_concentration))
        object.__setattr__(self, "profile_concentration", _bounded_unit(self.profile_concentration))
        object.__setattr__(self, "concentration_penalty", _bounded_unit(self.concentration_penalty))
        object.__setattr__(self, "correlation_penalty", _bounded_unit(self.correlation_penalty))
        object.__setattr__(self, "drawdown_penalty", _bounded_unit(self.drawdown_penalty))
        object.__setattr__(self, "reasons", tuple(str(x) for x in self.reasons))
        object.__setattr__(self, "recommendations", tuple(str(x) for x in self.recommendations))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def passed(self) -> bool:
        return self.acceptable

    def to_dict(self) -> dict[str, Any]:
        return {
            "acceptable": self.acceptable,
            "score": self.score,
            "selected_count": self.selected_count,
            "total_weight": self.total_weight,
            "max_weight": self.max_weight,
            "hhi": self.hhi,
            "effective_bets": self.effective_bets,
            "expected_total_pnl": self.expected_total_pnl,
            "expected_win_rate": self.expected_win_rate,
            "expected_profit_factor": self.expected_profit_factor,
            "weighted_drawdown": self.weighted_drawdown,
            "average_abs_correlation": self.average_abs_correlation,
            "max_abs_correlation": self.max_abs_correlation,
            "family_concentration": self.family_concentration,
            "profile_concentration": self.profile_concentration,
            "concentration_penalty": self.concentration_penalty,
            "correlation_penalty": self.correlation_penalty,
            "drawdown_penalty": self.drawdown_penalty,
            "reasons": list(self.reasons),
            "recommendations": list(self.recommendations),
            "correlation": None if self.correlation is None else self.correlation.to_dict(),
            "diversification": None if self.diversification is None else self.diversification.to_dict(),
            "metadata": dict(self.metadata),
        }


class PortfolioRiskModel:
    """
    Évalue le risque d'une allocation de portefeuille.
    """

    def __init__(
        self,
        settings: PortfolioRiskSettings | None = None,
        *,
        config: PortfolioConfig | Any | None = None,
    ) -> None:
        self._settings = settings or PortfolioRiskSettings.from_config(config)

    @property
    def settings(self) -> PortfolioRiskSettings:
        return self._settings

    def assess(
        self,
        results: Iterable[ExecutionResult],
        *,
        weights: Sequence[float] | Mapping[str, float] | None = None,
        correlation: PortfolioCorrelationMatrix | None = None,
        diversification: DiversificationAssessment | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> PortfolioRiskAssessment:
        results = tuple(results)
        if not results:
            return PortfolioRiskAssessment(
                acceptable=False,
                score=0.0,
                selected_count=0,
                total_weight=0.0,
                max_weight=0.0,
                hhi=1.0,
                effective_bets=0.0,
                expected_total_pnl=0.0,
                expected_win_rate=0.0,
                expected_profit_factor=0.0,
                weighted_drawdown=1.0,
                average_abs_correlation=1.0,
                max_abs_correlation=1.0,
                family_concentration=1.0,
                profile_concentration=1.0,
                concentration_penalty=1.0,
                correlation_penalty=1.0,
                drawdown_penalty=1.0,
                reasons=("empty_portfolio",),
                recommendations=("add_candidates",),
                correlation=correlation,
                diversification=diversification,
                metadata=dict(metadata or {}),
            )

        weights_arr = _portfolio_weights(results, weights)
        if correlation is None:
            from .correlation import PortfolioCorrelationAnalyzer

            correlation = PortfolioCorrelationAnalyzer().correlate(results)

        families = np.asarray([_family_key(result) for result in results], dtype=object)
        profiles = np.asarray([_profile_name(result) for result in results], dtype=object)

        family_weights: dict[str, float] = {}
        profile_weights: dict[str, float] = {}

        expected_total_pnl = 0.0
        expected_win_rate = 0.0
        expected_profit_factor = 0.0
        weighted_drawdown = 0.0
        fee_pressure = 0.0

        for weight, result, family, profile in zip(weights_arr, results, families, profiles):
            metrics = result.replay.metrics
            expected_total_pnl += float(weight) * float(metrics.total_pnl)
            expected_win_rate += float(weight) * float(metrics.win_rate)
            pf = float(metrics.profit_factor if np.isfinite(metrics.profit_factor) else 0.0)
            expected_profit_factor += float(weight) * max(0.0, pf)
            profile_obj = getattr(result, "profile", None)
            drawdown = float(getattr(profile_obj, "max_drawdown", metrics.max_drawdown) if profile_obj is not None else metrics.max_drawdown)
            weighted_drawdown += float(weight) * max(0.0, drawdown)

            family_weights[str(family)] = family_weights.get(str(family), 0.0) + float(weight)
            profile_weights[str(profile)] = profile_weights.get(str(profile), 0.0) + float(weight)

            diagnostics = getattr(result, "diagnostics", None)
            if diagnostics is not None:
                summary = _to_mapping(getattr(diagnostics, "summary", {}))
                fee_pressure += float(weight) * float(summary.get("fee_pressure", 0.0))

        total_weight = float(weights_arr.sum())
        max_weight = float(np.max(weights_arr))
        hhi = float(np.sum(weights_arr ** 2))
        effective_bets = float(1.0 / hhi) if hhi > 1e-12 else float(len(results))

        pair_abs = [abs(pair.correlation) for pair in correlation.pairs]
        average_abs_correlation = float(np.mean(pair_abs)) if pair_abs else 0.0
        max_abs_correlation = float(np.max(pair_abs)) if pair_abs else 0.0

        family_share = max(family_weights.values()) if family_weights else 0.0
        profile_share = max(profile_weights.values()) if profile_weights else 0.0

        family_concentration = family_share
        profile_concentration = profile_share

        concentration_penalty = max(
            hhi,
            max_weight,
            family_concentration,
            profile_concentration,
        )
        correlation_penalty = max(average_abs_correlation, max_abs_correlation)
        drawdown_penalty = weighted_drawdown

        score = (
            0.25 * (1.0 - concentration_penalty)
            + 0.25 * (1.0 - correlation_penalty)
            + 0.20 * (1.0 - drawdown_penalty)
            + 0.15 * _bounded_unit(0.5 * expected_win_rate + 0.5 * min(1.0, expected_profit_factor / 2.0))
            + 0.15 * _bounded_unit(0.5 * (expected_total_pnl > 0) + 0.5 * (1.0 - min(1.0, fee_pressure)))
        )

        reasons: list[str] = []
        recommendations: list[str] = []

        if total_weight > self._settings.max_gross_exposure + 1e-12:
            reasons.append("gross_exposure_too_high")
            recommendations.append("reduce_total_exposure")
        if max_weight > self._settings.max_single_weight:
            reasons.append("single_position_too_large")
            recommendations.append("cap_single_weight")
        if hhi > self._settings.max_hhi:
            reasons.append("concentration_too_high")
            recommendations.append("rebalance_weights")
        if effective_bets < self._settings.min_effective_bets:
            reasons.append("not_enough_effective_bets")
            recommendations.append("increase_diversification")
        if average_abs_correlation > self._settings.max_average_abs_correlation:
            reasons.append("correlation_too_high")
            recommendations.append("reduce_strategy_overlap")
        if weighted_drawdown > self._settings.max_drawdown:
            reasons.append("drawdown_too_high")
            recommendations.append("reduce_risk_budget")
        if diversification is not None and self._settings.require_diversified and not diversification.diversified:
            reasons.append("not_diversified_enough")
            recommendations.append("improve_diversification")

        if expected_total_pnl < self._settings.min_expected_pnl:
            reasons.append("expected_pnl_too_low")
        if expected_win_rate < self._settings.min_expected_win_rate:
            reasons.append("expected_win_rate_too_low")
        if np.isfinite(expected_profit_factor) and expected_profit_factor < self._settings.min_expected_profit_factor:
            reasons.append("expected_profit_factor_too_low")

        acceptable = (
            score >= self._settings.min_score
            and total_weight <= self._settings.max_gross_exposure + 1e-12
            and max_weight <= self._settings.max_single_weight
            and hhi <= self._settings.max_hhi
            and effective_bets >= self._settings.min_effective_bets
            and average_abs_correlation <= self._settings.max_average_abs_correlation
            and weighted_drawdown <= self._settings.max_drawdown
            and expected_total_pnl >= self._settings.min_expected_pnl
            and expected_win_rate >= self._settings.min_expected_win_rate
            and (not np.isfinite(expected_profit_factor) or expected_profit_factor >= self._settings.min_expected_profit_factor)
            and (not self._settings.require_diversified or diversification is None or diversification.diversified)
        )

        return PortfolioRiskAssessment(
            acceptable=acceptable,
            score=_bounded_unit(score),
            selected_count=len(results),
            total_weight=total_weight,
            max_weight=max_weight,
            hhi=hhi,
            effective_bets=effective_bets,
            expected_total_pnl=expected_total_pnl,
            expected_win_rate=expected_win_rate,
            expected_profit_factor=expected_profit_factor,
            weighted_drawdown=weighted_drawdown,
            average_abs_correlation=average_abs_correlation,
            max_abs_correlation=max_abs_correlation,
            family_concentration=family_concentration,
            profile_concentration=profile_concentration,
            concentration_penalty=concentration_penalty,
            correlation_penalty=correlation_penalty,
            drawdown_penalty=drawdown_penalty,
            reasons=tuple(dict.fromkeys(reasons)),
            recommendations=tuple(dict.fromkeys(recommendations)),
            correlation=correlation,
            diversification=diversification,
            metadata=dict(metadata or {}),
        )

    def score(
        self,
        results: Iterable[ExecutionResult],
        *,
        weights: Sequence[float] | Mapping[str, float] | None = None,
        correlation: PortfolioCorrelationMatrix | None = None,
        diversification: DiversificationAssessment | None = None,
    ) -> float:
        return self.assess(results, weights=weights, correlation=correlation, diversification=diversification).score

    def __repr__(self) -> str:
        return (
            "PortfolioRiskModel("
            f"min_score={self._settings.min_score}, "
            f"max_single_weight={self._settings.max_single_weight}"
            ")"
        )