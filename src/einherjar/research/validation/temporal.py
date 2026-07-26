"""
==========================================================
Validation Temporal
==========================================================

Analyse la stabilité temporelle d'un signal issu de la phase
Validation.

Ce module ne valide rien à lui seul. Il mesure la manière
dont un signal se comporte au fil des fenêtres temporelles :
- tendance globale,
- cohérence des variations,
- amplitude des ruptures,
- dérive entre le début et la fin,
- stabilité locale entre fenêtres successives.

Il complète robustness.py et persistence.py :
- robustness.py regarde surtout la cohérence globale,
- persistence.py regarde la survie du signal,
- temporal.py regarde la forme temporelle du signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Mapping
from typing import Sequence

import numpy as np

from config.scoring import ScoringConfig
from config.search import SearchConfig

__all__ = [
    "TemporalSettings",
    "TemporalAssessment",
    "TemporalAnalyzer",
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


def _normalize_array(values: Any) -> np.ndarray:
    if values is None:
        return np.asarray([], dtype=float)

    if isinstance(values, np.ndarray):
        if values.ndim == 0:
            return np.asarray([float(values)], dtype=float)
        return values.astype(float, copy=False).reshape(-1)

    if isinstance(values, (list, tuple, set)):
        out: list[float] = []
        for item in values:
            if isinstance(item, np.ndarray):
                if item.size == 0:
                    continue
                out.append(float(np.mean(item.astype(float, copy=False).reshape(-1))))
            elif isinstance(item, (list, tuple, set)):
                arr = np.asarray(list(item), dtype=float).reshape(-1)
                if arr.size > 0:
                    out.append(float(np.mean(arr)))
            else:
                try:
                    out.append(float(item))
                except (TypeError, ValueError):
                    continue
        return np.asarray(out, dtype=float)

    try:
        return np.asarray([float(values)], dtype=float)
    except (TypeError, ValueError):
        return np.asarray([], dtype=float)


def _safe_mean(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.mean(values))


def _safe_std(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.std(values))


def _safe_div(numerator: float, denominator: float) -> float:
    if abs(denominator) <= 1e-12:
        return 0.0
    return float(numerator / denominator)


def _window_split(values: np.ndarray, windows: int) -> tuple[np.ndarray, ...]:
    if values.size == 0:
        return ()

    windows = max(1, min(int(windows), values.size))
    parts = np.array_split(values, windows)
    return tuple(part for part in parts if part.size > 0)


def _normalize_window_entries(window_metrics: Any) -> tuple[dict[str, Any], ...]:
    if window_metrics is None:
        return ()

    if isinstance(window_metrics, Mapping):
        window_metrics = [window_metrics]

    if not isinstance(window_metrics, (list, tuple)):
        return ()

    normalized: list[dict[str, Any]] = []
    for item in window_metrics:
        if isinstance(item, Mapping):
            normalized.append(dict(item))
        elif hasattr(item, "to_dict") and callable(item.to_dict):
            normalized.append(dict(item.to_dict()))
    return tuple(normalized)


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
class TemporalSettings:
    """
    Paramètres du calcul temporel.
    """

    min_windows: int = 4
    min_support: int = 50
    min_coverage: float = 0.005

    min_trend_consistency: float = 0.55
    min_shift_stability: float = 0.50
    min_score: float = 0.55

    weight_trend: float = 0.30
    weight_consistency: float = 0.30
    weight_shift_stability: float = 0.20
    weight_decay: float = 0.20

    require_positive_temporal: bool = False
    positive_target_threshold: float = 0.0

    random_seed: int = 42

    def __post_init__(self) -> None:
        object.__setattr__(self, "min_windows", max(1, _coerce_int(self.min_windows, 4)))
        object.__setattr__(self, "min_support", max(1, _coerce_int(self.min_support, 50)))
        object.__setattr__(self, "min_coverage", min(1.0, max(0.0, _coerce_float(self.min_coverage, 0.005))))
        object.__setattr__(self, "min_trend_consistency", min(1.0, max(0.0, _coerce_float(self.min_trend_consistency, 0.55))))
        object.__setattr__(self, "min_shift_stability", min(1.0, max(0.0, _coerce_float(self.min_shift_stability, 0.50))))
        object.__setattr__(self, "min_score", min(1.0, max(0.0, _coerce_float(self.min_score, 0.55))))
        object.__setattr__(self, "require_positive_temporal", _coerce_bool(self.require_positive_temporal, False))
        object.__setattr__(self, "positive_target_threshold", _coerce_float(self.positive_target_threshold, 0.0))

        weights = {
            "trend": max(0.0, _coerce_float(self.weight_trend, 0.30)),
            "consistency": max(0.0, _coerce_float(self.weight_consistency, 0.30)),
            "shift": max(0.0, _coerce_float(self.weight_shift_stability, 0.20)),
            "decay": max(0.0, _coerce_float(self.weight_decay, 0.20)),
        }
        total = sum(weights.values())
        if total <= 0:
            weights = {"trend": 0.30, "consistency": 0.30, "shift": 0.20, "decay": 0.20}

        object.__setattr__(self, "weight_trend", weights["trend"])
        object.__setattr__(self, "weight_consistency", weights["consistency"])
        object.__setattr__(self, "weight_shift_stability", weights["shift"])
        object.__setattr__(self, "weight_decay", weights["decay"])
        object.__setattr__(self, "random_seed", _coerce_int(self.random_seed, 42))

    @classmethod
    def from_config(cls, config: Any | None) -> "TemporalSettings":
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
            min_windows=_coerce_int(_first_non_none(validation, ("min_windows",), ("windows",), default=4), 4),
            min_support=_coerce_int(_first_non_none(validation, ("min_support",), default=50), 50),
            min_coverage=_coerce_float(_first_non_none(validation, ("min_coverage",), default=0.005), 0.005),
            min_trend_consistency=_coerce_float(_first_non_none(validation, ("min_trend_consistency",), default=0.55), 0.55),
            min_shift_stability=_coerce_float(_first_non_none(validation, ("min_shift_stability",), default=0.50), 0.50),
            min_score=_coerce_float(_first_non_none(validation, ("min_score",), default=0.55), 0.55),
            weight_trend=_coerce_float(_first_non_none(validation, ("weight_trend",), default=scoring.diversity), scoring.diversity),
            weight_consistency=_coerce_float(_first_non_none(validation, ("weight_consistency",), default=scoring.novelty), scoring.novelty),
            weight_shift_stability=_coerce_float(_first_non_none(validation, ("weight_shift_stability",), default=scoring.robustness), scoring.robustness),
            weight_decay=_coerce_float(_first_non_none(validation, ("weight_decay",), default=scoring.persistence), scoring.persistence),
            require_positive_temporal=_coerce_bool(_first_non_none(validation, ("require_positive_temporal",), default=False), False),
            positive_target_threshold=_coerce_float(_first_non_none(validation, ("positive_target_threshold",), default=0.0), 0.0),
            random_seed=_coerce_int(_first_non_none(validation, ("random_seed",), default=search.random_seed), search.random_seed),
        )


# ==========================================================
# ASSESSMENT
# ==========================================================

@dataclass(frozen=True, slots=True)
class TemporalAssessment:
    """
    Résultat complet de l'analyse temporelle.
    """

    score: float

    stable: bool
    support: int
    coverage: float

    window_count: int
    active_windows: int
    active_windows_ratio: float

    trend_slope: float
    trend_consistency: float
    shift_stability: float
    decay_ratio: float
    decay_penalty: float

    window_mean: float
    window_std: float
    window_consistency: float

    autocorrelation: float
    volatility: float
    longest_run_ratio: float

    support_score: float
    trend_score: float
    consistency_score: float
    shift_score: float
    decay_score: float

    positive_rate: float = 0.0
    signal_positive_rate: float = 0.0

    sample_count: int = 0
    reasons: tuple[str, ...] = ()

    window_metrics: tuple[dict[str, Any], ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "score", min(1.0, max(0.0, float(self.score))))
        object.__setattr__(self, "stable", _coerce_bool(self.stable, False))
        object.__setattr__(self, "support", max(0, _coerce_int(self.support, 0)))
        object.__setattr__(self, "coverage", min(1.0, max(0.0, float(self.coverage))))
        object.__setattr__(self, "window_count", max(0, _coerce_int(self.window_count, 0)))
        object.__setattr__(self, "active_windows", max(0, _coerce_int(self.active_windows, 0)))
        object.__setattr__(self, "active_windows_ratio", min(1.0, max(0.0, float(self.active_windows_ratio))))
        object.__setattr__(self, "trend_slope", float(self.trend_slope))
        object.__setattr__(self, "trend_consistency", min(1.0, max(0.0, float(self.trend_consistency))))
        object.__setattr__(self, "shift_stability", min(1.0, max(0.0, float(self.shift_stability))))
        object.__setattr__(self, "decay_ratio", min(1.0, max(0.0, float(self.decay_ratio))))
        object.__setattr__(self, "decay_penalty", min(1.0, max(0.0, float(self.decay_penalty))))
        object.__setattr__(self, "window_mean", float(self.window_mean))
        object.__setattr__(self, "window_std", max(0.0, float(self.window_std)))
        object.__setattr__(self, "window_consistency", min(1.0, max(0.0, float(self.window_consistency))))
        object.__setattr__(self, "autocorrelation", min(1.0, max(-1.0, float(self.autocorrelation))))
        object.__setattr__(self, "volatility", max(0.0, float(self.volatility)))
        object.__setattr__(self, "longest_run_ratio", min(1.0, max(0.0, float(self.longest_run_ratio))))
        object.__setattr__(self, "support_score", min(1.0, max(0.0, float(self.support_score))))
        object.__setattr__(self, "trend_score", min(1.0, max(0.0, float(self.trend_score))))
        object.__setattr__(self, "consistency_score", min(1.0, max(0.0, float(self.consistency_score))))
        object.__setattr__(self, "shift_score", min(1.0, max(0.0, float(self.shift_score))))
        object.__setattr__(self, "decay_score", min(1.0, max(0.0, float(self.decay_score))))
        object.__setattr__(self, "positive_rate", min(1.0, max(0.0, float(self.positive_rate))))
        object.__setattr__(self, "signal_positive_rate", min(1.0, max(0.0, float(self.signal_positive_rate))))
        object.__setattr__(self, "sample_count", max(0, _coerce_int(self.sample_count, 0)))
        object.__setattr__(self, "reasons", tuple(str(reason) for reason in self.reasons))
        object.__setattr__(self, "window_metrics", tuple(dict(item) for item in self.window_metrics))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def passed(self) -> bool:
        return self.stable

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "stable": self.stable,
            "support": self.support,
            "coverage": self.coverage,
            "window_count": self.window_count,
            "active_windows": self.active_windows,
            "active_windows_ratio": self.active_windows_ratio,
            "trend_slope": self.trend_slope,
            "trend_consistency": self.trend_consistency,
            "shift_stability": self.shift_stability,
            "decay_ratio": self.decay_ratio,
            "decay_penalty": self.decay_penalty,
            "window_mean": self.window_mean,
            "window_std": self.window_std,
            "window_consistency": self.window_consistency,
            "autocorrelation": self.autocorrelation,
            "volatility": self.volatility,
            "longest_run_ratio": self.longest_run_ratio,
            "support_score": self.support_score,
            "trend_score": self.trend_score,
            "consistency_score": self.consistency_score,
            "shift_score": self.shift_score,
            "decay_score": self.decay_score,
            "positive_rate": self.positive_rate,
            "signal_positive_rate": self.signal_positive_rate,
            "sample_count": self.sample_count,
            "reasons": list(self.reasons),
            "window_metrics": [dict(item) for item in self.window_metrics],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TemporalAssessment":
        return cls(
            score=_coerce_float(data.get("score"), 0.0),
            stable=_coerce_bool(data.get("stable"), False),
            support=_coerce_int(data.get("support"), 0),
            coverage=_coerce_float(data.get("coverage"), 0.0),
            window_count=_coerce_int(data.get("window_count"), 0),
            active_windows=_coerce_int(data.get("active_windows"), 0),
            active_windows_ratio=_coerce_float(data.get("active_windows_ratio"), 0.0),
            trend_slope=_coerce_float(data.get("trend_slope"), 0.0),
            trend_consistency=_coerce_float(data.get("trend_consistency"), 0.0),
            shift_stability=_coerce_float(data.get("shift_stability"), 0.0),
            decay_ratio=_coerce_float(data.get("decay_ratio"), 0.0),
            decay_penalty=_coerce_float(data.get("decay_penalty"), 1.0),
            window_mean=_coerce_float(data.get("window_mean"), 0.0),
            window_std=_coerce_float(data.get("window_std"), 0.0),
            window_consistency=_coerce_float(data.get("window_consistency"), 0.0),
            autocorrelation=_coerce_float(data.get("autocorrelation"), 0.0),
            volatility=_coerce_float(data.get("volatility"), 0.0),
            longest_run_ratio=_coerce_float(data.get("longest_run_ratio"), 0.0),
            support_score=_coerce_float(data.get("support_score"), 0.0),
            trend_score=_coerce_float(data.get("trend_score"), 0.0),
            consistency_score=_coerce_float(data.get("consistency_score"), 0.0),
            shift_score=_coerce_float(data.get("shift_score"), 0.0),
            decay_score=_coerce_float(data.get("decay_score"), 0.0),
            positive_rate=_coerce_float(data.get("positive_rate"), 0.0),
            signal_positive_rate=_coerce_float(data.get("signal_positive_rate"), 0.0),
            sample_count=_coerce_int(data.get("sample_count"), 0),
            reasons=tuple(data.get("reasons", ())),
            window_metrics=_normalize_window_entries(data.get("window_metrics", ())),
            metadata=_to_mapping(data.get("metadata", {})),
        )

    def __repr__(self) -> str:
        return (
            "TemporalAssessment("
            f"score={self.score:.4f}, "
            f"stable={self.stable}, "
            f"support={self.support}, "
            f"windows={self.window_count}"
            ")"
        )


# ==========================================================
# ANALYZER
# ==========================================================

class TemporalAnalyzer:
    """
    Analyse la forme temporelle d'un signal.

    Le score combine :
    - la cohérence de la tendance,
    - la stabilité des fenêtres,
    - la résistance aux ruptures,
    - la pénalité de dérive entre première et seconde moitié.
    """

    def __init__(
        self,
        settings: TemporalSettings | None = None,
        *,
        config: Any | None = None,
        search_config: SearchConfig | None = None,
        scoring_config: ScoringConfig | None = None,
        rng: np.random.Generator | int | None = None,
    ) -> None:
        if settings is None:
            settings = TemporalSettings.from_config(config)

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
    ) -> "TemporalAnalyzer":
        return cls(
            settings=TemporalSettings.from_config(config),
            config=config,
            search_config=search_config,
            scoring_config=scoring_config,
            rng=rng,
        )

    @property
    def settings(self) -> TemporalSettings:
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

    def assess_from_metrics(
        self,
        metrics: Any,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> TemporalAssessment:
        data = _to_mapping(metrics)

        support = _coerce_int(_first_non_none(data, ("support",), ("sample_count",), default=0), 0)
        coverage = _coerce_float(_first_non_none(data, ("coverage",), default=0.0), 0.0)
        sample_count = _coerce_int(_first_non_none(data, ("sample_count",), default=support), support)

        window_metrics = _normalize_window_entries(
            _first_non_none(data, ("window_metrics",), ("windows",), default=())
        )

        return self._build_from_windows(
            window_metrics=window_metrics,
            support=support,
            coverage=coverage,
            sample_count=sample_count,
            metadata=metadata,
            base_metadata=data,
        )

    def assess_from_series(
        self,
        signal: Any,
        baseline: Any | None = None,
        *,
        windows: int | None = None,
        support: int | None = None,
        coverage: float | None = None,
        positive_rate: float | None = None,
        signal_positive_rate: float | None = None,
        target_threshold: float | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> TemporalAssessment:
        signal_arr = _normalize_array(signal)
        if signal_arr.size == 0:
            return self._empty_assessment(
                support=support or 0,
                coverage=coverage or 0.0,
                positive_rate=positive_rate or 0.0,
                signal_positive_rate=signal_positive_rate or 0.0,
                metadata=metadata,
                reasons=("insufficient_data",),
            )

        windows = max(1, _coerce_int(windows if windows is not None else self._settings.min_windows, self._settings.min_windows))
        chunks = _window_split(signal_arr, windows)
        base_arr = _normalize_array(baseline)
        threshold = self._settings.positive_target_threshold if target_threshold is None else float(target_threshold)

        window_metrics: list[dict[str, Any]] = []
        offset = 0
        for index, chunk in enumerate(chunks):
            if chunk.size == 0:
                continue

            mean_value = _safe_mean(chunk)
            std_value = _safe_std(chunk)
            chunk_positive_rate = float(np.mean(chunk > threshold)) if chunk.size > 0 else 0.0

            if base_arr.size > 0 and base_arr.size >= offset + chunk.size:
                base_chunk = base_arr[offset : offset + chunk.size]
                baseline_mean = _safe_mean(base_chunk)
            elif base_arr.size > 0:
                base_chunk = base_arr
                baseline_mean = _safe_mean(base_chunk)
            else:
                baseline_mean = 0.0

            window_metrics.append(
                {
                    "index": index,
                    "count": int(chunk.size),
                    "support": int(chunk.size),
                    "mean": mean_value,
                    "std": std_value,
                    "lift": mean_value - baseline_mean,
                    "positive_rate": chunk_positive_rate,
                }
            )
            offset += chunk.size

        return self._build_from_windows(
            window_metrics=tuple(window_metrics),
            support=support if support is not None else int(signal_arr.size),
            coverage=coverage if coverage is not None else 1.0 if signal_arr.size > 0 else 0.0,
            sample_count=int(signal_arr.size),
            metadata=metadata,
            base_metadata={
                "positive_rate": positive_rate,
                "signal_positive_rate": signal_positive_rate,
            },
        )

    def assess(
        self,
        signal: Any = None,
        baseline: Any = None,
        *,
        metrics: Any | None = None,
        windows: int | None = None,
        support: int | None = None,
        coverage: float | None = None,
        positive_rate: float | None = None,
        signal_positive_rate: float | None = None,
        target_threshold: float | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> TemporalAssessment:
        if metrics is not None:
            return self.assess_from_metrics(metrics, metadata=metadata)

        return self.assess_from_series(
            signal,
            baseline,
            windows=windows,
            support=support,
            coverage=coverage,
            positive_rate=positive_rate,
            signal_positive_rate=signal_positive_rate,
            target_threshold=target_threshold,
            metadata=metadata,
        )

    def score(self, *args: Any, **kwargs: Any) -> float:
        return self.assess(*args, **kwargs).score

    def is_stable(self, assessment: TemporalAssessment | Mapping[str, Any] | Any) -> bool:
        if not isinstance(assessment, TemporalAssessment):
            assessment = TemporalAssessment.from_dict(_to_mapping(assessment))
        return assessment.stable

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
    ) -> TemporalAssessment:
        return TemporalAssessment(
            score=0.0,
            stable=False,
            support=support,
            coverage=coverage,
            window_count=0,
            active_windows=0,
            active_windows_ratio=0.0,
            trend_slope=0.0,
            trend_consistency=0.0,
            shift_stability=0.0,
            decay_ratio=0.0,
            decay_penalty=0.0,
            window_mean=0.0,
            window_std=0.0,
            window_consistency=0.0,
            autocorrelation=0.0,
            volatility=0.0,
            longest_run_ratio=0.0,
            support_score=0.0,
            trend_score=0.0,
            consistency_score=0.0,
            shift_score=0.0,
            decay_score=0.0,
            positive_rate=positive_rate,
            signal_positive_rate=signal_positive_rate,
            sample_count=0,
            reasons=tuple(reasons),
            window_metrics=(),
            metadata=dict(metadata or {}),
        )

    def _build_from_windows(
        self,
        *,
        window_metrics: tuple[dict[str, Any], ...],
        support: int,
        coverage: float,
        sample_count: int,
        metadata: Mapping[str, Any] | None,
        base_metadata: Mapping[str, Any] | None = None,
    ) -> TemporalAssessment:
        settings = self._settings
        base_metadata = dict(base_metadata or {})
        meta = dict(metadata or {})
        meta.update(base_metadata)

        if not window_metrics:
            return self._empty_assessment(
                support=support,
                coverage=coverage,
                positive_rate=_coerce_float(base_metadata.get("positive_rate"), 0.0),
                signal_positive_rate=_coerce_float(base_metadata.get("signal_positive_rate"), 0.0),
                metadata=meta,
                reasons=("insufficient_windows",),
            )

        window_count = len(window_metrics)
        if window_count < settings.min_windows:
            return self._empty_assessment(
                support=support,
                coverage=coverage,
                positive_rate=_coerce_float(base_metadata.get("positive_rate"), 0.0),
                signal_positive_rate=_coerce_float(base_metadata.get("signal_positive_rate"), 0.0),
                metadata=meta,
                reasons=("insufficient_windows",),
            )

        supports = np.asarray([_coerce_int(item.get("support", item.get("count", 0)), 0) for item in window_metrics], dtype=float)
        means = np.asarray([_coerce_float(item.get("mean", 0.0), 0.0) for item in window_metrics], dtype=float)
        lifts = np.asarray([_coerce_float(item.get("lift", 0.0), 0.0) for item in window_metrics], dtype=float)
        positives = np.asarray([_coerce_float(item.get("positive_rate", 0.0), 0.0) for item in window_metrics], dtype=float)

        active_windows_mask = supports > 0
        active_windows = int(np.sum(active_windows_mask))
        active_windows_ratio = active_windows / max(1, window_count)

        window_mean = _safe_mean(means)
        window_std = _safe_std(means)
        volatility = _safe_div(window_std, abs(window_mean) if abs(window_mean) > 1e-12 else (window_std if window_std > 0 else 1.0))

        if window_count > 1:
            x = np.arange(window_count, dtype=float)
            try:
                trend_slope = float(np.polyfit(x, means, 1)[0])
            except Exception:
                trend_slope = 0.0
        else:
            trend_slope = 0.0

        trend_scale = max(1e-12, abs(window_mean) if abs(window_mean) > 1e-12 else (np.max(np.abs(means)) if means.size > 0 else 1.0))
        trend_consistency = max(0.0, min(1.0, 1.0 - min(1.0, abs(trend_slope) / trend_scale)))

        diffs = np.diff(means) if means.size > 1 else np.asarray([], dtype=float)
        if diffs.size > 0:
            sign_ref = 1 if trend_slope >= 0 else -1
            same_direction = np.mean((np.sign(diffs) == sign_ref) | (np.abs(diffs) <= 1e-12))
            sign_changes = np.sum(np.sign(diffs[1:]) != np.sign(diffs[:-1])) if diffs.size > 1 else 0
            shift_stability = max(0.0, min(1.0, 0.5 * float(same_direction) + 0.5 * (1.0 - sign_changes / max(1, diffs.size - 1))))
        else:
            shift_stability = 1.0

        if window_count > 1:
            first_half = means[: max(1, window_count // 2)]
            second_half = means[max(1, window_count // 2) :]
            if second_half.size == 0:
                second_half = means[-1:]
        else:
            first_half = means
            second_half = means

        first_half_mean = _safe_mean(first_half)
        second_half_mean = _safe_mean(second_half)

        denom = max(1e-12, abs(first_half_mean) + abs(second_half_mean))
        decay_ratio = abs(first_half_mean - second_half_mean) / denom
        decay_penalty = 1.0 - min(1.0, decay_ratio)

        if means.size > 1 and _safe_std(means) > 1e-12:
            autocorrelation = float(np.corrcoef(means[:-1], means[1:])[0, 1]) if means.size > 2 else 0.0
            if np.isnan(autocorrelation):
                autocorrelation = 0.0
        else:
            autocorrelation = 0.0

        longest_run_ratio = self._longest_run_ratio(means)

        support_score = min(1.0, support / max(1.0, settings.min_support))
        trend_score = max(0.0, min(1.0, trend_consistency))
        consistency_score = max(0.0, min(1.0, shift_stability))
        shift_score = max(0.0, min(1.0, shift_stability))
        decay_score = max(0.0, min(1.0, decay_penalty))

        positive_rate = _coerce_float(base_metadata.get("positive_rate"), float(np.mean(positives)) if positives.size > 0 else 0.0)
        signal_positive_rate = _coerce_float(base_metadata.get("signal_positive_rate"), float(np.mean(positives)) if positives.size > 0 else 0.0)

        reasons: list[str] = []
        if support < settings.min_support:
            reasons.append("insufficient_support")
        if coverage < settings.min_coverage:
            reasons.append("coverage_too_low")
        if window_count < settings.min_windows:
            reasons.append("insufficient_windows")
        if active_windows_ratio < 0.5:
            reasons.append("too_few_active_windows")
        if trend_consistency < settings.min_trend_consistency:
            reasons.append("trend_inconsistency")
        if shift_stability < settings.min_shift_stability:
            reasons.append("shift_instability")
        if settings.require_positive_temporal and trend_slope <= 0:
            reasons.append("non_positive_temporal")

        score = (
            settings.weight_trend * trend_score
            + settings.weight_consistency * consistency_score
            + settings.weight_shift_stability * shift_score
            + settings.weight_decay * decay_score
        )

        score *= min(1.0, 0.60 + 0.40 * support_score)
        score *= min(1.0, 0.60 + 0.40 * min(1.0, coverage / max(1e-12, settings.min_coverage if settings.min_coverage > 0 else 1.0)))

        if settings.require_positive_temporal and trend_slope <= 0:
            score *= 0.0

        stable = score >= settings.min_score and not reasons

        return TemporalAssessment(
            score=max(0.0, min(1.0, score)),
            stable=stable,
            support=support,
            coverage=coverage,
            window_count=window_count,
            active_windows=active_windows,
            active_windows_ratio=active_windows_ratio,
            trend_slope=trend_slope,
            trend_consistency=trend_consistency,
            shift_stability=shift_stability,
            decay_ratio=decay_ratio,
            decay_penalty=decay_penalty,
            window_mean=window_mean,
            window_std=window_std,
            window_consistency=trend_consistency,
            autocorrelation=autocorrelation,
            volatility=volatility,
            longest_run_ratio=longest_run_ratio,
            support_score=support_score,
            trend_score=trend_score,
            consistency_score=consistency_score,
            shift_score=shift_score,
            decay_score=decay_score,
            positive_rate=positive_rate,
            signal_positive_rate=signal_positive_rate,
            sample_count=sample_count,
            reasons=tuple(reasons),
            window_metrics=window_metrics,
            metadata=meta,
        )

    def _longest_run_ratio(self, values: np.ndarray) -> float:
        if values.size == 0:
            return 0.0

        signs = np.sign(values)
        if np.all(signs == 0):
            return 1.0

        best = current = 1
        prev = signs[0]

        for sign in signs[1:]:
            if sign == prev or sign == 0:
                current += 1
            else:
                best = max(best, current)
                current = 1
                prev = sign

        best = max(best, current)
        return best / max(1, values.size)

    # ==================================================
    # PYTHON PROTOCOL
    # ==================================================

    def __repr__(self) -> str:
        return (
            "TemporalAnalyzer("
            f"min_windows={self._settings.min_windows}, "
            f"min_score={self._settings.min_score}"
            ")"
        )