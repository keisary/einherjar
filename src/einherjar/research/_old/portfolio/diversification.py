# portfolio/diversification.py
"""
==========================================================
Portfolio Diversification
==========================================================

Mesure et pilote la diversification du portefeuille.

Le module utilise la corrélation, les familles et les
profils pour éviter un portefeuille trop homogène.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from math import log
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from execution.execution_report import ExecutionResult

from .correlation import PortfolioCorrelationAnalyzer
from .correlation import PortfolioCorrelationMatrix
from .correlation import PortfolioCorrelationSettings

try:  # optional config module
    from config.portfolio import PortfolioConfig  # type: ignore
except Exception:  # pragma: no cover
    PortfolioConfig = Any  # type: ignore[misc,assignment]

__all__ = [
    "DiversificationSettings",
    "DiversificationAssessment",
    "DiversificationEngine",
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


def _safe_entropy(weights: Sequence[float]) -> float:
    arr = np.asarray([w for w in weights if w > 0], dtype=float)
    if arr.size == 0:
        return 0.0
    arr = arr / arr.sum()
    entropy = -float(np.sum(arr * np.log(arr)))
    if arr.size <= 1:
        return 0.0
    return entropy / log(arr.size)


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


def _weights_from_mapping(
    results: Sequence[ExecutionResult],
    weights: Sequence[float] | Mapping[str, float] | None,
) -> np.ndarray:
    if weights is None:
        return np.full(len(results), 1.0 / max(1, len(results)), dtype=float)

    if isinstance(weights, Mapping):
        arr = []
        for result in results:
            key = getattr(result, "subject_fingerprint", None) or getattr(result.execution_fingerprint, "digest", None)
            key = str(key or "")
            arr.append(float(weights.get(key, weights.get(_profile_name(result), 0.0))))
        arr = np.asarray(arr, dtype=float)
    else:
        arr = np.asarray(list(weights), dtype=float)

    if arr.size != len(results):
        raise ValueError("weights must match results length.")

    arr = np.maximum(arr, 0.0)
    if arr.sum() <= 1e-12:
        return np.full(len(results), 1.0 / max(1, len(results)), dtype=float)
    return arr / arr.sum()


@dataclass(frozen=True, slots=True)
class DiversificationSettings:
    """
    Paramètres de diversification.
    """

    min_score: float = 0.55
    min_family_entropy: float = 0.50
    max_abs_correlation: float = 0.75
    max_same_family_share: float = 0.60
    min_unique_profiles: int = 2

    correlation_settings: PortfolioCorrelationSettings = field(default_factory=PortfolioCorrelationSettings)

    def __post_init__(self) -> None:
        object.__setattr__(self, "min_score", _bounded_unit(_coerce_float(self.min_score, 0.55)))
        object.__setattr__(self, "min_family_entropy", _bounded_unit(_coerce_float(self.min_family_entropy, 0.50)))
        object.__setattr__(self, "max_abs_correlation", _bounded_unit(_coerce_float(self.max_abs_correlation, 0.75)))
        object.__setattr__(self, "max_same_family_share", _bounded_unit(_coerce_float(self.max_same_family_share, 0.60)))
        object.__setattr__(self, "min_unique_profiles", max(1, _coerce_int(self.min_unique_profiles, 2)))

    @classmethod
    def from_config(cls, config: Any | None) -> "DiversificationSettings":
        if config is None:
            return cls()

        root = _to_mapping(config)
        port = _to_mapping(root.get("portfolio", root.get("portfolio_config", {})))
        div = _to_mapping(port.get("diversification", port.get("diversify", {})))

        corr_settings = PortfolioCorrelationSettings.from_config(config)

        return cls(
            min_score=_coerce_float(div.get("min_score", root.get("min_score", 0.55)), 0.55),
            min_family_entropy=_coerce_float(div.get("min_family_entropy", root.get("min_family_entropy", 0.50)), 0.50),
            max_abs_correlation=_coerce_float(div.get("max_abs_correlation", root.get("max_abs_correlation", 0.75)), 0.75),
            max_same_family_share=_coerce_float(div.get("max_same_family_share", root.get("max_same_family_share", 0.60)), 0.60),
            min_unique_profiles=_coerce_int(div.get("min_unique_profiles", root.get("min_unique_profiles", 2)), 2),
            correlation_settings=corr_settings,
        )


@dataclass(frozen=True, slots=True)
class DiversificationAssessment:
    """
    Résultat de diversification d'un portefeuille.
    """

    diversified: bool
    score: float

    selected_count: int
    unique_family_count: int
    unique_profile_count: int

    family_entropy: float
    profile_entropy: float
    same_family_share: float

    average_abs_correlation: float
    max_abs_correlation: float
    redundancy_score: float
    coverage_score: float

    reasons: tuple[str, ...] = ()
    recommendations: tuple[str, ...] = ()
    correlation: PortfolioCorrelationMatrix | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "diversified", _coerce_bool(self.diversified, False))
        object.__setattr__(self, "score", _bounded_unit(self.score))
        object.__setattr__(self, "selected_count", max(0, _coerce_int(self.selected_count, 0)))
        object.__setattr__(self, "unique_family_count", max(0, _coerce_int(self.unique_family_count, 0)))
        object.__setattr__(self, "unique_profile_count", max(0, _coerce_int(self.unique_profile_count, 0)))
        object.__setattr__(self, "family_entropy", _bounded_unit(self.family_entropy))
        object.__setattr__(self, "profile_entropy", _bounded_unit(self.profile_entropy))
        object.__setattr__(self, "same_family_share", _bounded_unit(self.same_family_share))
        object.__setattr__(self, "average_abs_correlation", _bounded_unit(self.average_abs_correlation))
        object.__setattr__(self, "max_abs_correlation", _bounded_unit(self.max_abs_correlation))
        object.__setattr__(self, "redundancy_score", _bounded_unit(self.redundancy_score))
        object.__setattr__(self, "coverage_score", _bounded_unit(self.coverage_score))
        object.__setattr__(self, "reasons", tuple(str(x) for x in self.reasons))
        object.__setattr__(self, "recommendations", tuple(str(x) for x in self.recommendations))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def passed(self) -> bool:
        return self.diversified

    def to_dict(self) -> dict[str, Any]:
        return {
            "diversified": self.diversified,
            "score": self.score,
            "selected_count": self.selected_count,
            "unique_family_count": self.unique_family_count,
            "unique_profile_count": self.unique_profile_count,
            "family_entropy": self.family_entropy,
            "profile_entropy": self.profile_entropy,
            "same_family_share": self.same_family_share,
            "average_abs_correlation": self.average_abs_correlation,
            "max_abs_correlation": self.max_abs_correlation,
            "redundancy_score": self.redundancy_score,
            "coverage_score": self.coverage_score,
            "reasons": list(self.reasons),
            "recommendations": list(self.recommendations),
            "correlation": None if self.correlation is None else self.correlation.to_dict(),
            "metadata": dict(self.metadata),
        }


class DiversificationEngine:
    """
    Évalue la diversification d'un portefeuille d'Einhers.
    """

    def __init__(
        self,
        settings: DiversificationSettings | None = None,
        *,
        config: PortfolioConfig | Any | None = None,
        correlation_analyzer: PortfolioCorrelationAnalyzer | None = None,
    ) -> None:
        self._settings = settings or DiversificationSettings.from_config(config)
        self._correlation_analyzer = correlation_analyzer or PortfolioCorrelationAnalyzer(
            settings=self._settings.correlation_settings
        )

    @property
    def settings(self) -> DiversificationSettings:
        return self._settings

    @property
    def correlation_analyzer(self) -> PortfolioCorrelationAnalyzer:
        return self._correlation_analyzer

    def assess(
        self,
        results: Iterable[ExecutionResult],
        *,
        weights: Sequence[float] | Mapping[str, float] | None = None,
        correlation: PortfolioCorrelationMatrix | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> DiversificationAssessment:
        results = tuple(results)
        if not results:
            return DiversificationAssessment(
                diversified=False,
                score=0.0,
                selected_count=0,
                unique_family_count=0,
                unique_profile_count=0,
                family_entropy=0.0,
                profile_entropy=0.0,
                same_family_share=0.0,
                average_abs_correlation=0.0,
                max_abs_correlation=0.0,
                redundancy_score=1.0,
                coverage_score=0.0,
                reasons=("empty_portfolio",),
                recommendations=("add_candidates",),
                correlation=correlation,
                metadata=dict(metadata or {}),
            )

        if correlation is None:
            correlation = self._correlation_analyzer.correlate(results)

        weights_arr = _weights_from_mapping(results, weights)
        families = np.asarray([_family_key(result) for result in results], dtype=object)
        profiles = np.asarray([_profile_name(result) for result in results], dtype=object)

        family_weights: dict[str, float] = defaultdict(float)
        profile_weights: dict[str, float] = defaultdict(float)

        for weight, family, profile in zip(weights_arr, families, profiles):
            family_weights[str(family)] += float(weight)
            profile_weights[str(profile)] += float(weight)

        family_entropy = _safe_entropy(list(family_weights.values()))
        profile_entropy = _safe_entropy(list(profile_weights.values()))
        same_family_share = max(family_weights.values()) if family_weights else 0.0

        unique_family_count = len(family_weights)
        unique_profile_count = len(profile_weights)

        pair_values = [abs(pair.correlation) for pair in correlation.pairs]
        average_abs_corr = float(np.average(pair_values, weights=None)) if pair_values else 0.0
        max_abs_corr = float(max(pair_values)) if pair_values else 0.0
        redundancy_score = average_abs_corr

        coverage_score = _bounded_unit(
            0.5 * family_entropy
            + 0.3 * profile_entropy
            + 0.2 * (1.0 - same_family_share)
        )

        score = (
            0.35 * (1.0 - average_abs_corr)
            + 0.25 * family_entropy
            + 0.20 * profile_entropy
            + 0.10 * (1.0 - same_family_share)
            + 0.10 * coverage_score
        )

        reasons: list[str] = []
        recommendations: list[str] = []

        if family_entropy < self._settings.min_family_entropy:
            reasons.append("family_entropy_too_low")
            recommendations.append("increase_family_diversity")
        if max_abs_corr > self._settings.max_abs_correlation:
            reasons.append("correlation_too_high")
            recommendations.append("remove_highly_correlated_strategies")
        if same_family_share > self._settings.max_same_family_share:
            reasons.append("family_concentration_too_high")
            recommendations.append("rebalance_family_weights")
        if unique_profile_count < self._settings.min_unique_profiles:
            reasons.append("too_few_profiles")
            recommendations.append("add_profile_variety")
        if average_abs_corr > 0.5:
            recommendations.append("reduce_redundancy")

        diversified = (
            score >= self._settings.min_score
            and family_entropy >= self._settings.min_family_entropy
            and max_abs_corr <= self._settings.max_abs_correlation
            and same_family_share <= self._settings.max_same_family_share
            and unique_profile_count >= self._settings.min_unique_profiles
        )

        return DiversificationAssessment(
            diversified=diversified,
            score=_bounded_unit(score),
            selected_count=len(results),
            unique_family_count=unique_family_count,
            unique_profile_count=unique_profile_count,
            family_entropy=family_entropy,
            profile_entropy=profile_entropy,
            same_family_share=same_family_share,
            average_abs_correlation=average_abs_corr,
            max_abs_correlation=max_abs_corr,
            redundancy_score=redundancy_score,
            coverage_score=coverage_score,
            reasons=tuple(dict.fromkeys(reasons)),
            recommendations=tuple(dict.fromkeys(recommendations)),
            correlation=correlation,
            metadata=dict(metadata or {}),
        )

    def score(
        self,
        results: Iterable[ExecutionResult],
        *,
        weights: Sequence[float] | Mapping[str, float] | None = None,
        correlation: PortfolioCorrelationMatrix | None = None,
    ) -> float:
        return self.assess(results, weights=weights, correlation=correlation).score

    def __repr__(self) -> str:
        return (
            "DiversificationEngine("
            f"min_score={self._settings.min_score}, "
            f"max_abs_correlation={self._settings.max_abs_correlation}"
            ")"
        )