"""
==========================================================
Validation Significance
==========================================================

Mesure la significativité d'un signal issu de la phase
Validation.

Ce module ne valide rien à lui seul. Il calcule un score de
significativité à partir de statistiques résumées ou de
séries numériques, puis produit un diagnostic exploitable
par evaluator.py.

L'idée est simple :
- plus le lift est net,
- plus la taille d'effet est forte,
- plus le support est suffisant,
- plus la confiance monte.

Aucune dépendance circulaire avec evaluator.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import Any
from typing import Iterable
from typing import Mapping
from typing import Sequence

import numpy as np

from config.scoring import ScoringConfig
from config.search import SearchConfig

__all__ = [
    "SignificanceSettings",
    "SignificanceAssessment",
    "SignificanceScorer",
]


# ==========================================================
# HELPERS
# ==========================================================

def _resolve_path(obj: Any, path: Sequence[str]) -> Any | None:
    current = obj
    for part in path:
        if current is None:
            return None

        if isinstance(current, Mapping):
            if part not in current:
                return None
            current = current[part]
            continue

        if not hasattr(current, part):
            return None

        current = getattr(current, part)

    return current


def _first_non_none(obj: Any, *paths: Sequence[str], default: Any = None) -> Any:
    for path in paths:
        value = _resolve_path(obj, path)
        if value is not None:
            return value
    return default


def _to_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}

    if hasattr(value, "to_dict") and callable(value.to_dict):
        value = value.to_dict()

    if isinstance(value, Mapping):
        return dict(value)

    return {}


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


def _safe_mean(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.mean(values))


def _safe_std(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.std(values))


def _safe_var(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.var(values))


def _cohen_d(signal: np.ndarray, baseline: np.ndarray) -> float:
    if signal.size == 0 or baseline.size == 0:
        return 0.0

    s_var = _safe_var(signal)
    b_var = _safe_var(baseline)
    dof = signal.size + baseline.size - 2
    if dof <= 0:
        return 0.0

    pooled = (((signal.size - 1) * s_var) + ((baseline.size - 1) * b_var)) / dof
    if pooled <= 0:
        return 0.0

    return float((_safe_mean(signal) - _safe_mean(baseline)) / sqrt(pooled))


def _t_statistic(signal: np.ndarray, baseline: np.ndarray) -> float:
    if signal.size < 2 or baseline.size < 2:
        return 0.0

    signal_std = _safe_std(signal)
    baseline_std = _safe_std(baseline)

    se = sqrt((signal_std ** 2) / signal.size + (baseline_std ** 2) / baseline.size)
    if se <= 0:
        return 0.0

    return float((_safe_mean(signal) - _safe_mean(baseline)) / se)


def _normalize_array(values: Any) -> np.ndarray:
    if values is None:
        return np.asarray([], dtype=float)

    if isinstance(values, np.ndarray):
        if values.ndim == 0:
            return np.asarray([float(values)], dtype=float)
        return values.astype(float, copy=False).reshape(-1)

    if isinstance(values, (list, tuple, set)):
        return np.asarray(list(values), dtype=float).reshape(-1)

    try:
        return np.asarray([float(values)], dtype=float)
    except (TypeError, ValueError):
        return np.asarray([], dtype=float)


def _build_search_config(source: Any | None) -> SearchConfig:
    if isinstance(source, SearchConfig):
        return source

    if source is None:
        return SearchConfig()

    return SearchConfig(
        max_conditions=_coerce_int(_first_non_none(source, ("max_conditions",), ("search", "max_conditions"), default=3), 3),
        beam_width=_coerce_int(_first_non_none(source, ("beam_width",), ("search", "beam_width"), default=200), 200),
        max_depth=_coerce_int(_first_non_none(source, ("max_depth",), ("search", "max_depth"), default=3), 3),
        max_candidates_per_family=_coerce_int(_first_non_none(source, ("max_candidates_per_family",), ("search", "max_candidates_per_family"), default=100), 100),
        exploration_ratio=_coerce_float(_first_non_none(source, ("exploration_ratio",), ("search", "exploration_ratio"), default=0.25), 0.25),
        exploitation_ratio=_coerce_float(_first_non_none(source, ("exploitation_ratio",), ("search", "exploitation_ratio"), default=0.75), 0.75),
        novelty_weight=_coerce_float(_first_non_none(source, ("novelty_weight",), ("search", "novelty_weight"), default=0.30), 0.30),
        diversity_weight=_coerce_float(_first_non_none(source, ("diversity_weight",), ("search", "diversity_weight"), default=0.25), 0.25),
        family_balance_weight=_coerce_float(_first_non_none(source, ("family_balance_weight",), ("search", "family_balance_weight"), default=0.20), 0.20),
        random_seed=_coerce_int(_first_non_none(source, ("random_seed",), ("search", "random_seed"), default=42), 42),
    )


def _build_scoring_config(source: Any | None) -> ScoringConfig:
    if isinstance(source, ScoringConfig):
        return source

    if source is None:
        return ScoringConfig()

    return ScoringConfig(
        novelty=_coerce_float(_first_non_none(source, ("novelty",), ("scoring", "novelty"), default=0.20), 0.20),
        diversity=_coerce_float(_first_non_none(source, ("diversity",), ("scoring", "diversity"), default=0.20), 0.20),
        robustness=_coerce_float(_first_non_none(source, ("robustness",), ("scoring", "robustness"), default=0.20), 0.20),
        persistence=_coerce_float(_first_non_none(source, ("persistence",), ("scoring", "persistence"), default=0.20), 0.20),
        profitability=_coerce_float(_first_non_none(source, ("profitability",), ("scoring", "profitability"), default=0.20), 0.20),
    )


# ==========================================================
# SETTINGS
# ==========================================================

@dataclass(frozen=True, slots=True)
class SignificanceSettings:
    """
    Paramètres du calcul de significativité.
    """

    min_support: int = 50
    min_coverage: float = 0.005

    min_t_stat: float = 1.96
    min_effect_size: float = 0.20
    min_lift: float = 0.0

    require_positive_lift: bool = False
    positive_target_threshold: float = 0.0

    weight_t_stat: float = 0.40
    weight_effect_size: float = 0.30
    weight_lift: float = 0.20
    weight_support: float = 0.10

    min_score: float = 0.55
    random_seed: int = 42

    def __post_init__(self) -> None:
        object.__setattr__(self, "min_support", max(1, _coerce_int(self.min_support, 50)))
        object.__setattr__(self, "min_coverage", min(1.0, max(0.0, _coerce_float(self.min_coverage, 0.005))))
        object.__setattr__(self, "min_t_stat", max(0.0, _coerce_float(self.min_t_stat, 1.96)))
        object.__setattr__(self, "min_effect_size", max(0.0, _coerce_float(self.min_effect_size, 0.20)))
        object.__setattr__(self, "min_lift", _coerce_float(self.min_lift, 0.0))
        object.__setattr__(self, "require_positive_lift", _coerce_bool(self.require_positive_lift, False))
        object.__setattr__(self, "positive_target_threshold", _coerce_float(self.positive_target_threshold, 0.0))

        weights = {
            "t": max(0.0, _coerce_float(self.weight_t_stat, 0.40)),
            "effect": max(0.0, _coerce_float(self.weight_effect_size, 0.30)),
            "lift": max(0.0, _coerce_float(self.weight_lift, 0.20)),
            "support": max(0.0, _coerce_float(self.weight_support, 0.10)),
        }
        total = sum(weights.values())
        if total <= 0:
            weights = {"t": 0.40, "effect": 0.30, "lift": 0.20, "support": 0.10}

        object.__setattr__(self, "weight_t_stat", weights["t"])
        object.__setattr__(self, "weight_effect_size", weights["effect"])
        object.__setattr__(self, "weight_lift", weights["lift"])
        object.__setattr__(self, "weight_support", weights["support"])
        object.__setattr__(self, "min_score", min(1.0, max(0.0, _coerce_float(self.min_score, 0.55))))
        object.__setattr__(self, "random_seed", _coerce_int(self.random_seed, 42))

    @classmethod
    def from_config(cls, config: Any | None) -> "SignificanceSettings":
        if config is None:
            return cls()

        validation = _first_non_none(
            config,
            ("validation",),
            ("validation_config",),
            default=config,
        )
        search = _build_search_config(_first_non_none(config, ("search",), ("search_config",), default=None))
        scoring = _build_scoring_config(_first_non_none(config, ("scoring",), ("scoring_config",), default=None))

        return cls(
            min_support=_coerce_int(_first_non_none(validation, ("min_support",), default=50), 50),
            min_coverage=_coerce_float(_first_non_none(validation, ("min_coverage",), default=0.005), 0.005),
            min_t_stat=_coerce_float(_first_non_none(validation, ("min_t_stat",), default=1.96), 1.96),
            min_effect_size=_coerce_float(_first_non_none(validation, ("min_effect_size",), default=0.20), 0.20),
            min_lift=_coerce_float(_first_non_none(validation, ("min_lift",), default=0.0), 0.0),
            require_positive_lift=_coerce_bool(_first_non_none(validation, ("require_positive_lift",), default=False), False),
            positive_target_threshold=_coerce_float(_first_non_none(validation, ("positive_target_threshold",), default=0.0), 0.0),
            weight_t_stat=_coerce_float(_first_non_none(validation, ("weight_t_stat",), default=scoring.novelty), scoring.novelty),
            weight_effect_size=_coerce_float(_first_non_none(validation, ("weight_effect_size",), default=scoring.robustness), scoring.robustness),
            weight_lift=_coerce_float(_first_non_none(validation, ("weight_lift",), default=scoring.persistence), scoring.persistence),
            weight_support=_coerce_float(_first_non_none(validation, ("weight_support",), default=0.10), 0.10),
            min_score=_coerce_float(_first_non_none(validation, ("min_score",), default=0.55), 0.55),
            random_seed=_coerce_int(_first_non_none(validation, ("random_seed",), default=search.random_seed), search.random_seed),
        )


# ==========================================================
# ASSESSMENT
# ==========================================================

@dataclass(frozen=True, slots=True)
class SignificanceAssessment:
    """
    Résultat complet du calcul de significativité.
    """

    score: float

    significant: bool
    support: int
    coverage: float

    baseline_mean: float
    baseline_std: float

    signal_mean: float
    signal_std: float

    lift: float
    lift_ratio: float

    t_stat: float
    effect_size: float
    p_value_proxy: float

    support_score: float
    effect_score: float
    lift_score: float
    t_score: float

    confidence: float = 0.0
    positive_rate: float = 0.0
    signal_positive_rate: float = 0.0

    sample_count: int = 0
    reasons: tuple[str, ...] = ()

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "score", min(1.0, max(0.0, float(self.score))))
        object.__setattr__(self, "significant", _coerce_bool(self.significant, False))
        object.__setattr__(self, "support", max(0, _coerce_int(self.support, 0)))
        object.__setattr__(self, "coverage", min(1.0, max(0.0, float(self.coverage))))
        object.__setattr__(self, "baseline_mean", float(self.baseline_mean))
        object.__setattr__(self, "baseline_std", max(0.0, float(self.baseline_std)))
        object.__setattr__(self, "signal_mean", float(self.signal_mean))
        object.__setattr__(self, "signal_std", max(0.0, float(self.signal_std)))
        object.__setattr__(self, "lift", float(self.lift))
        object.__setattr__(self, "lift_ratio", float(self.lift_ratio))
        object.__setattr__(self, "t_stat", float(self.t_stat))
        object.__setattr__(self, "effect_size", float(self.effect_size))
        object.__setattr__(self, "p_value_proxy", min(1.0, max(0.0, float(self.p_value_proxy))))
        object.__setattr__(self, "support_score", min(1.0, max(0.0, float(self.support_score))))
        object.__setattr__(self, "effect_score", min(1.0, max(0.0, float(self.effect_score))))
        object.__setattr__(self, "lift_score", min(1.0, max(0.0, float(self.lift_score))))
        object.__setattr__(self, "t_score", min(1.0, max(0.0, float(self.t_score))))
        object.__setattr__(self, "confidence", min(1.0, max(0.0, float(self.confidence))))
        object.__setattr__(self, "positive_rate", min(1.0, max(0.0, float(self.positive_rate))))
        object.__setattr__(self, "signal_positive_rate", min(1.0, max(0.0, float(self.signal_positive_rate))))
        object.__setattr__(self, "sample_count", max(0, _coerce_int(self.sample_count, 0)))
        object.__setattr__(self, "reasons", tuple(str(reason) for reason in self.reasons))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def passed(self) -> bool:
        return self.significant

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "significant": self.significant,
            "support": self.support,
            "coverage": self.coverage,
            "baseline_mean": self.baseline_mean,
            "baseline_std": self.baseline_std,
            "signal_mean": self.signal_mean,
            "signal_std": self.signal_std,
            "lift": self.lift,
            "lift_ratio": self.lift_ratio,
            "t_stat": self.t_stat,
            "effect_size": self.effect_size,
            "p_value_proxy": self.p_value_proxy,
            "support_score": self.support_score,
            "effect_score": self.effect_score,
            "lift_score": self.lift_score,
            "t_score": self.t_score,
            "confidence": self.confidence,
            "positive_rate": self.positive_rate,
            "signal_positive_rate": self.signal_positive_rate,
            "sample_count": self.sample_count,
            "reasons": list(self.reasons),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SignificanceAssessment":
        return cls(
            score=_coerce_float(data.get("score"), 0.0),
            significant=_coerce_bool(data.get("significant"), False),
            support=_coerce_int(data.get("support"), 0),
            coverage=_coerce_float(data.get("coverage"), 0.0),
            baseline_mean=_coerce_float(data.get("baseline_mean"), 0.0),
            baseline_std=_coerce_float(data.get("baseline_std"), 0.0),
            signal_mean=_coerce_float(data.get("signal_mean"), 0.0),
            signal_std=_coerce_float(data.get("signal_std"), 0.0),
            lift=_coerce_float(data.get("lift"), 0.0),
            lift_ratio=_coerce_float(data.get("lift_ratio"), 0.0),
            t_stat=_coerce_float(data.get("t_stat"), 0.0),
            effect_size=_coerce_float(data.get("effect_size"), 0.0),
            p_value_proxy=_coerce_float(data.get("p_value_proxy"), 1.0),
            support_score=_coerce_float(data.get("support_score"), 0.0),
            effect_score=_coerce_float(data.get("effect_score"), 0.0),
            lift_score=_coerce_float(data.get("lift_score"), 0.0),
            t_score=_coerce_float(data.get("t_score"), 0.0),
            confidence=_coerce_float(data.get("confidence"), 0.0),
            positive_rate=_coerce_float(data.get("positive_rate"), 0.0),
            signal_positive_rate=_coerce_float(data.get("signal_positive_rate"), 0.0),
            sample_count=_coerce_int(data.get("sample_count"), 0),
            reasons=tuple(data.get("reasons", ())),
            metadata=_to_mapping(data.get("metadata", {})),
        )

    def __repr__(self) -> str:
        return (
            "SignificanceAssessment("
            f"score={self.score:.4f}, "
            f"significant={self.significant}, "
            f"support={self.support}, "
            f"lift={self.lift:.6f}"
            ")"
        )


# ==========================================================
# SCORER
# ==========================================================

class SignificanceScorer:
    """
    Calcule la significativité d'un signal.

    Le score combine :
    - le t-statistic,
    - la taille d'effet,
    - le lift,
    - le support,
    - la couverture.
    """

    def __init__(
        self,
        settings: SignificanceSettings | None = None,
        *,
        config: Any | None = None,
        search_config: SearchConfig | None = None,
        scoring_config: ScoringConfig | None = None,
        rng: np.random.Generator | int | None = None,
    ) -> None:
        if settings is None:
            settings = SignificanceSettings.from_config(config)

        self._settings = settings
        self._search_config = search_config or _build_search_config(_first_non_none(config, ("search",), ("search_config",), default=None))
        self._scoring_config = scoring_config or _build_scoring_config(_first_non_none(config, ("scoring",), ("scoring_config",), default=None))

        if isinstance(rng, np.random.Generator):
            self._rng = rng
        elif rng is not None:
            self._rng = np.random.default_rng(rng)
        else:
            self._rng = np.random.default_rng(self._settings.random_seed)

    @classmethod
    def from_config(
        cls,
        config: Any,
        *,
        search_config: SearchConfig | None = None,
        scoring_config: ScoringConfig | None = None,
        rng: np.random.Generator | int | None = None,
    ) -> "SignificanceScorer":
        return cls(
            settings=SignificanceSettings.from_config(config),
            config=config,
            search_config=search_config,
            scoring_config=scoring_config,
            rng=rng,
        )

    @property
    def settings(self) -> SignificanceSettings:
        return self._settings

    @property
    def search_config(self) -> SearchConfig:
        return self._search_config

    @property
    def scoring_config(self) -> ScoringConfig:
        return self._scoring_config

    # ==================================================
    # PUBLIC API
    # ==================================================

    def assess_from_series(
        self,
        signal: Any,
        baseline: Any,
        *,
        support: int | None = None,
        coverage: float | None = None,
        positive_rate: float | None = None,
        signal_positive_rate: float | None = None,
        target_threshold: float | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> SignificanceAssessment:
        signal_arr = _normalize_array(signal)
        baseline_arr = _normalize_array(baseline)

        if signal_arr.size == 0 or baseline_arr.size == 0:
            return self._empty_assessment(
                support=support or 0,
                coverage=coverage or 0.0,
                positive_rate=positive_rate or 0.0,
                signal_positive_rate=signal_positive_rate or 0.0,
                metadata=metadata,
                reasons=("insufficient_data",),
            )

        baseline_mean = _safe_mean(baseline_arr)
        baseline_std = _safe_std(baseline_arr)
        signal_mean = _safe_mean(signal_arr)
        signal_std = _safe_std(signal_arr)

        support = int(signal_arr.size if support is None else max(0, _coerce_int(support, signal_arr.size)))
        coverage = float(signal_arr.size / baseline_arr.size if coverage is None else coverage)

        lift = signal_mean - baseline_mean
        lift_ratio = signal_mean / baseline_mean if abs(baseline_mean) > 1e-12 else 0.0
        t_stat = _t_statistic(signal_arr, baseline_arr)
        effect_size = _cohen_d(signal_arr, baseline_arr)

        threshold = self._settings.positive_target_threshold if target_threshold is None else float(target_threshold)
        if positive_rate is None:
            positive_rate = float(np.mean(baseline_arr > threshold))
        if signal_positive_rate is None:
            signal_positive_rate = float(np.mean(signal_arr > threshold))

        return self._build_assessment(
            support=support,
            coverage=coverage,
            baseline_mean=baseline_mean,
            baseline_std=baseline_std,
            signal_mean=signal_mean,
            signal_std=signal_std,
            lift=lift,
            lift_ratio=lift_ratio,
            t_stat=t_stat,
            effect_size=effect_size,
            positive_rate=positive_rate,
            signal_positive_rate=signal_positive_rate,
            sample_count=int(baseline_arr.size),
            metadata=metadata,
        )

    def assess_from_metrics(
        self,
        metrics: Any,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> SignificanceAssessment:
        support = _coerce_int(_first_non_none(metrics, ("support",), ("sample_count",), default=0), 0)
        coverage = _coerce_float(_first_non_none(metrics, ("coverage",), default=0.0), 0.0)
        baseline_mean = _coerce_float(_first_non_none(metrics, ("baseline_mean",), default=0.0), 0.0)
        baseline_std = _coerce_float(_first_non_none(metrics, ("baseline_std",), default=0.0), 0.0)
        signal_mean = _coerce_float(_first_non_none(metrics, ("signal_mean",), default=0.0), 0.0)
        signal_std = _coerce_float(_first_non_none(metrics, ("signal_std",), default=0.0), 0.0)
        lift = _coerce_float(_first_non_none(metrics, ("lift",), default=0.0), 0.0)
        lift_ratio = _coerce_float(_first_non_none(metrics, ("lift_ratio",), default=0.0), 0.0)
        t_stat = _coerce_float(_first_non_none(metrics, ("t_stat",), default=0.0), 0.0)
        effect_size = _coerce_float(_first_non_none(metrics, ("effect_size",), default=0.0), 0.0)
        positive_rate = _coerce_float(_first_non_none(metrics, ("positive_rate",), default=0.0), 0.0)
        signal_positive_rate = _coerce_float(_first_non_none(metrics, ("signal_positive_rate",), default=0.0), 0.0)
        sample_count = _coerce_int(_first_non_none(metrics, ("sample_count",), default=support), support)

        return self._build_assessment(
            support=support,
            coverage=coverage,
            baseline_mean=baseline_mean,
            baseline_std=baseline_std,
            signal_mean=signal_mean,
            signal_std=signal_std,
            lift=lift,
            lift_ratio=lift_ratio,
            t_stat=t_stat,
            effect_size=effect_size,
            positive_rate=positive_rate,
            signal_positive_rate=signal_positive_rate,
            sample_count=sample_count,
            metadata=metadata,
        )

    def score(self, *args: Any, **kwargs: Any) -> float:
        if args and isinstance(args[0], (Mapping, object)) and not isinstance(args[0], (np.ndarray, list, tuple, set, float, int)):
            return self.assess_from_metrics(args[0], **kwargs).score

        return self.assess_from_series(*args, **kwargs).score

    def assess(
        self,
        signal: Any = None,
        baseline: Any = None,
        *,
        metrics: Any | None = None,
        support: int | None = None,
        coverage: float | None = None,
        positive_rate: float | None = None,
        signal_positive_rate: float | None = None,
        target_threshold: float | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> SignificanceAssessment:
        if metrics is not None:
            return self.assess_from_metrics(metrics, metadata=metadata)

        return self.assess_from_series(
            signal,
            baseline,
            support=support,
            coverage=coverage,
            positive_rate=positive_rate,
            signal_positive_rate=signal_positive_rate,
            target_threshold=target_threshold,
            metadata=metadata,
        )

    def is_significant(self, assessment: SignificanceAssessment | Mapping[str, Any] | Any) -> bool:
        if not isinstance(assessment, SignificanceAssessment):
            assessment = SignificanceAssessment.from_dict(_to_mapping(assessment))
        return assessment.significant

    # ==================================================
    # INTERNAL BUILDERS
    # ==================================================

    def _empty_assessment(
        self,
        *,
        support: int,
        coverage: float,
        positive_rate: float,
        signal_positive_rate: float,
        metadata: Mapping[str, Any] | None,
        reasons: Sequence[str] = (),
    ) -> SignificanceAssessment:
        return SignificanceAssessment(
            score=0.0,
            significant=False,
            support=support,
            coverage=coverage,
            baseline_mean=0.0,
            baseline_std=0.0,
            signal_mean=0.0,
            signal_std=0.0,
            lift=0.0,
            lift_ratio=0.0,
            t_stat=0.0,
            effect_size=0.0,
            p_value_proxy=1.0,
            support_score=0.0,
            effect_score=0.0,
            lift_score=0.0,
            t_score=0.0,
            confidence=0.0,
            positive_rate=positive_rate,
            signal_positive_rate=signal_positive_rate,
            sample_count=0,
            reasons=tuple(reasons),
            metadata=dict(metadata or {}),
        )

    def _build_assessment(
        self,
        *,
        support: int,
        coverage: float,
        baseline_mean: float,
        baseline_std: float,
        signal_mean: float,
        signal_std: float,
        lift: float,
        lift_ratio: float,
        t_stat: float,
        effect_size: float,
        positive_rate: float,
        signal_positive_rate: float,
        sample_count: int,
        metadata: Mapping[str, Any] | None,
    ) -> SignificanceAssessment:
        settings = self._settings

        support_score = min(1.0, support / max(1.0, float(settings.min_support)))
        coverage_score = min(1.0, coverage / max(1e-12, settings.min_coverage if settings.min_coverage > 0 else 1.0))

        t_score = 1.0 - float(np.exp(-abs(t_stat) / max(1e-12, settings.min_t_stat if settings.min_t_stat > 0 else 1.0)))
        effect_score = 1.0 - float(np.exp(-abs(effect_size) / max(1e-12, settings.min_effect_size if settings.min_effect_size > 0 else 1.0)))

        lift_abs = abs(lift)
        lift_threshold = abs(settings.min_lift) if settings.min_lift != 0 else max(1e-9, abs(baseline_std) * 0.01)
        lift_score = 1.0 - float(np.exp(-lift_abs / lift_threshold)) if lift_threshold > 0 else 0.0

        p_value_proxy = 1.0 / (1.0 + abs(t_stat))

        score = (
            settings.weight_t_stat * t_score
            + settings.weight_effect_size * effect_score
            + settings.weight_lift * lift_score
            + settings.weight_support * min(1.0, 0.5 * support_score + 0.5 * coverage_score)
        )

        confidence = (
            0.35 * t_score
            + 0.35 * effect_score
            + 0.15 * support_score
            + 0.15 * coverage_score
        )

        reasons: list[str] = []

        if support < settings.min_support:
            reasons.append("insufficient_support")
        if coverage < settings.min_coverage:
            reasons.append("coverage_too_low")
        if abs(t_stat) < settings.min_t_stat:
            reasons.append("t_stat_too_low")
        if abs(effect_size) < settings.min_effect_size:
            reasons.append("effect_size_too_low")
        if settings.require_positive_lift and lift <= 0:
            reasons.append("non_positive_lift")

        significant = score >= settings.min_score and not reasons

        if significant and settings.require_positive_lift and lift <= 0:
            significant = False
            if "non_positive_lift" not in reasons:
                reasons.append("non_positive_lift")

        score = max(0.0, min(1.0, score))
        confidence = max(0.0, min(1.0, confidence))

        return SignificanceAssessment(
            score=score,
            significant=significant,
            support=support,
            coverage=coverage,
            baseline_mean=baseline_mean,
            baseline_std=baseline_std,
            signal_mean=signal_mean,
            signal_std=signal_std,
            lift=lift,
            lift_ratio=lift_ratio,
            t_stat=t_stat,
            effect_size=effect_size,
            p_value_proxy=p_value_proxy,
            support_score=support_score,
            effect_score=effect_score,
            lift_score=lift_score,
            t_score=t_score,
            confidence=confidence,
            positive_rate=positive_rate,
            signal_positive_rate=signal_positive_rate,
            sample_count=sample_count,
            reasons=tuple(reasons),
            metadata=dict(metadata or {}),
        )

    # ==================================================
    # PYTHON PROTOCOL
    # ==================================================

    def __repr__(self) -> str:
        return (
            "SignificanceScorer("
            f"min_support={self._settings.min_support}, "
            f"min_score={self._settings.min_score}"
            ")"
        )