"""
==========================================================
Validation Persistence
==========================================================

Mesure la persistance d'un signal issu de la phase
Validation.

Ce module ne valide rien à lui seul. Il calcule un score de
persistance à partir de métriques résumées ou de séries
fenêtrées, puis produit un diagnostic exploitable par
evaluator.py.

L'idée est simple :
- plus le signal survit dans le temps,
- plus il reste cohérent entre fenêtres,
- plus il conserve une direction stable,
- plus la persistance est forte.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from math import sqrt
from typing import Any
from typing import Iterable
from typing import Mapping
from typing import Sequence

import numpy as np

from config.scoring import ScoringConfig
from config.search import SearchConfig

__all__ = [
    "PersistenceSettings",
    "PersistenceAssessment",
    "PersistenceScorer",
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


def _run_lengths(mask: Sequence[bool]) -> tuple[int, ...]:
    lengths: list[int] = []
    current = 0

    for value in mask:
        if value:
            current += 1
        elif current > 0:
            lengths.append(current)
            current = 0

    if current > 0:
        lengths.append(current)

    return tuple(lengths)


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


def _safe_div(numerator: float, denominator: float) -> float:
    if abs(denominator) <= 1e-12:
        return 0.0
    return float(numerator / denominator)


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


def _binary_direction(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


# ==========================================================
# SETTINGS
# ==========================================================

@dataclass(frozen=True, slots=True)
class PersistenceSettings:
    """
    Paramètres du calcul de persistance.
    """

    min_support: int = 50
    min_coverage: float = 0.005
    min_windows: int = 3

    min_active_windows_ratio: float = 0.50
    min_consistency: float = 0.55
    min_score: float = 0.55

    weight_active_windows: float = 0.30
    weight_consistency: float = 0.30
    weight_stability: float = 0.20
    weight_decay: float = 0.20

    require_positive_persistence: bool = False
    positive_target_threshold: float = 0.0

    random_seed: int = 42

    def __post_init__(self) -> None:
        object.__setattr__(self, "min_support", max(1, _coerce_int(self.min_support, 50)))
        object.__setattr__(self, "min_coverage", min(1.0, max(0.0, _coerce_float(self.min_coverage, 0.005))))
        object.__setattr__(self, "min_windows", max(1, _coerce_int(self.min_windows, 3)))
        object.__setattr__(self, "min_active_windows_ratio", min(1.0, max(0.0, _coerce_float(self.min_active_windows_ratio, 0.50))))
        object.__setattr__(self, "min_consistency", min(1.0, max(0.0, _coerce_float(self.min_consistency, 0.55))))
        object.__setattr__(self, "min_score", min(1.0, max(0.0, _coerce_float(self.min_score, 0.55))))
        object.__setattr__(self, "require_positive_persistence", _coerce_bool(self.require_positive_persistence, False))
        object.__setattr__(self, "positive_target_threshold", _coerce_float(self.positive_target_threshold, 0.0))

        weights = {
            "active": max(0.0, _coerce_float(self.weight_active_windows, 0.30)),
            "consistency": max(0.0, _coerce_float(self.weight_consistency, 0.30)),
            "stability": max(0.0, _coerce_float(self.weight_stability, 0.20)),
            "decay": max(0.0, _coerce_float(self.weight_decay, 0.20)),
        }
        total = sum(weights.values())
        if total <= 0:
            weights = {"active": 0.30, "consistency": 0.30, "stability": 0.20, "decay": 0.20}

        object.__setattr__(self, "weight_active_windows", weights["active"])
        object.__setattr__(self, "weight_consistency", weights["consistency"])
        object.__setattr__(self, "weight_stability", weights["stability"])
        object.__setattr__(self, "weight_decay", weights["decay"])
        object.__setattr__(self, "random_seed", _coerce_int(self.random_seed, 42))

    @classmethod
    def from_config(cls, config: Any | None) -> "PersistenceSettings":
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
            min_windows=_coerce_int(_first_non_none(validation, ("min_windows",), ("windows",), default=3), 3),
            min_active_windows_ratio=_coerce_float(_first_non_none(validation, ("min_active_windows_ratio",), default=0.50), 0.50),
            min_consistency=_coerce_float(_first_non_none(validation, ("min_consistency",), default=0.55), 0.55),
            min_score=_coerce_float(_first_non_none(validation, ("min_score",), default=0.55), 0.55),
            weight_active_windows=_coerce_float(_first_non_none(validation, ("weight_active_windows",), default=scoring.persistence), scoring.persistence),
            weight_consistency=_coerce_float(_first_non_none(validation, ("weight_consistency",), default=scoring.robustness), scoring.robustness),
            weight_stability=_coerce_float(_first_non_none(validation, ("weight_stability",), default=scoring.diversity), scoring.diversity),
            weight_decay=_coerce_float(_first_non_none(validation, ("weight_decay",), default=0.20), 0.20),
            require_positive_persistence=_coerce_bool(_first_non_none(validation, ("require_positive_persistence",), default=False), False),
            positive_target_threshold=_coerce_float(_first_non_none(validation, ("positive_target_threshold",), default=0.0), 0.0),
            random_seed=_coerce_int(_first_non_none(validation, ("random_seed",), default=search.random_seed), search.random_seed),
        )


# ==========================================================
# ASSESSMENT
# ==========================================================

@dataclass(frozen=True, slots=True)
class PersistenceAssessment:
    """
    Résultat complet du calcul de persistance.
    """

    score: float

    persistent: bool
    support: int
    coverage: float

    window_count: int
    active_windows: int
    active_windows_ratio: float

    window_mean: float
    window_std: float
    window_consistency: float

    first_half_mean: float
    second_half_mean: float
    decay_ratio: float
    decay_penalty: float

    longest_run_ratio: float
    stability_score: float
    support_score: float
    active_score: float
    consistency_score: float

    positive_rate: float = 0.0
    signal_positive_rate: float = 0.0

    sample_count: int = 0
    reasons: tuple[str, ...] = ()

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "score", min(1.0, max(0.0, float(self.score))))
        object.__setattr__(self, "persistent", _coerce_bool(self.persistent, False))
        object.__setattr__(self, "support", max(0, _coerce_int(self.support, 0)))
        object.__setattr__(self, "coverage", min(1.0, max(0.0, float(self.coverage))))
        object.__setattr__(self, "window_count", max(0, _coerce_int(self.window_count, 0)))
        object.__setattr__(self, "active_windows", max(0, _coerce_int(self.active_windows, 0)))
        object.__setattr__(self, "active_windows_ratio", min(1.0, max(0.0, float(self.active_windows_ratio))))
        object.__setattr__(self, "window_mean", float(self.window_mean))
        object.__setattr__(self, "window_std", max(0.0, float(self.window_std)))
        object.__setattr__(self, "window_consistency", min(1.0, max(0.0, float(self.window_consistency))))
        object.__setattr__(self, "first_half_mean", float(self.first_half_mean))
        object.__setattr__(self, "second_half_mean", float(self.second_half_mean))
        object.__setattr__(self, "decay_ratio", min(1.0, max(0.0, float(self.decay_ratio))))
        object.__setattr__(self, "decay_penalty", min(1.0, max(0.0, float(self.decay_penalty))))
        object.__setattr__(self, "longest_run_ratio", min(1.0, max(0.0, float(self.longest_run_ratio))))
        object.__setattr__(self, "stability_score", min(1.0, max(0.0, float(self.stability_score))))
        object.__setattr__(self, "support_score", min(1.0, max(0.0, float(self.support_score))))
        object.__setattr__(self, "active_score", min(1.0, max(0.0, float(self.active_score))))
        object.__setattr__(self, "consistency_score", min(1.0, max(0.0, float(self.consistency_score))))
        object.__setattr__(self, "positive_rate", min(1.0, max(0.0, float(self.positive_rate))))
        object.__setattr__(self, "signal_positive_rate", min(1.0, max(0.0, float(self.signal_positive_rate))))
        object.__setattr__(self, "sample_count", max(0, _coerce_int(self.sample_count, 0)))
        object.__setattr__(self, "reasons", tuple(str(reason) for reason in self.reasons))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def passed(self) -> bool:
        return self.persistent

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "persistent": self.persistent,
            "support": self.support,
            "coverage": self.coverage,
            "window_count": self.window_count,
            "active_windows": self.active_windows,
            "active_windows_ratio": self.active_windows_ratio,
            "window_mean": self.window_mean,
            "window_std": self.window_std,
            "window_consistency": self.window_consistency,
            "first_half_mean": self.first_half_mean,
            "second_half_mean": self.second_half_mean,
            "decay_ratio": self.decay_ratio,
            "decay_penalty": self.decay_penalty,
            "longest_run_ratio": self.longest_run_ratio,
            "stability_score": self.stability_score,
            "support_score": self.support_score,
            "active_score": self.active_score,
            "consistency_score": self.consistency_score,
            "positive_rate": self.positive_rate,
            "signal_positive_rate": self.signal_positive_rate,
            "sample_count": self.sample_count,
            "reasons": list(self.reasons),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PersistenceAssessment":
        return cls(
            score=_coerce_float(data.get("score"), 0.0),
            persistent=_coerce_bool(data.get("persistent"), False),
            support=_coerce_int(data.get("support"), 0),
            coverage=_coerce_float(data.get("coverage"), 0.0),
            window_count=_coerce_int(data.get("window_count"), 0),
            active_windows=_coerce_int(data.get("active_windows"), 0),
            active_windows_ratio=_coerce_float(data.get("active_windows_ratio"), 0.0),
            window_mean=_coerce_float(data.get("window_mean"), 0.0),
            window_std=_coerce_float(data.get("window_std"), 0.0),
            window_consistency=_coerce_float(data.get("window_consistency"), 0.0),
            first_half_mean=_coerce_float(data.get("first_half_mean"), 0.0),
            second_half_mean=_coerce_float(data.get("second_half_mean"), 0.0),
            decay_ratio=_coerce_float(data.get("decay_ratio"), 0.0),
            decay_penalty=_coerce_float(data.get("decay_penalty"), 1.0),
            longest_run_ratio=_coerce_float(data.get("longest_run_ratio"), 0.0),
            stability_score=_coerce_float(data.get("stability_score"), 0.0),
            support_score=_coerce_float(data.get("support_score"), 0.0),
            active_score=_coerce_float(data.get("active_score"), 0.0),
            consistency_score=_coerce_float(data.get("consistency_score"), 0.0),
            positive_rate=_coerce_float(data.get("positive_rate"), 0.0),
            signal_positive_rate=_coerce_float(data.get("signal_positive_rate"), 0.0),
            sample_count=_coerce_int(data.get("sample_count"), 0),
            reasons=tuple(data.get("reasons", ())),
            metadata=_to_mapping(data.get("metadata", {})),
        )

    def __repr__(self) -> str:
        return (
            "PersistenceAssessment("
            f"score={self.score:.4f}, "
            f"persistent={self.persistent}, "
            f"support={self.support}, "
            f"windows={self.window_count}"
            ")"
        )


# ==========================================================
# SCORER
# ==========================================================

class PersistenceScorer:
    """
    Calcule la persistance d'un signal.

    Le score combine :
    - la proportion de fenêtres actives,
    - la cohérence inter-fenêtres,
    - la stabilité de la moyenne,
    - la pénalité de décroissance,
    - la couverture et le support.
    """

    def __init__(
        self,
        settings: PersistenceSettings | None = None,
        *,
        config: Any | None = None,
        search_config: SearchConfig | None = None,
        scoring_config: ScoringConfig | None = None,
        rng: np.random.Generator | int | None = None,
    ) -> None:
        if settings is None:
            settings = PersistenceSettings.from_config(config)

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
    ) -> "PersistenceScorer":
        return cls(
            settings=PersistenceSettings.from_config(config),
            config=config,
            search_config=search_config,
            scoring_config=scoring_config,
            rng=rng,
        )

    @property
    def settings(self) -> PersistenceSettings:
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
    ) -> PersistenceAssessment:
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
    ) -> PersistenceAssessment:
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
        chunks = np.array_split(signal_arr, min(windows, signal_arr.size))
        base_arr = _normalize_array(baseline)

        window_metrics: list[dict[str, Any]] = []
        threshold = self._settings.positive_target_threshold if target_threshold is None else float(target_threshold)

        for index, chunk in enumerate(chunks):
            if chunk.size == 0:
                continue

            mean_value = _safe_mean(chunk)
            std_value = _safe_std(chunk)
            chunk_positive_rate = float(np.mean(chunk > threshold)) if chunk.size > 0 else 0.0

            if base_arr.size > 0:
                base_chunk = base_arr if base_arr.size == chunk.size else base_arr[: min(base_arr.size, chunk.size)]
                base_mean = _safe_mean(base_chunk)
            else:
                base_mean = 0.0

            window_metrics.append(
                {
                    "index": index,
                    "count": int(chunk.size),
                    "support": int(chunk.size),
                    "mean": mean_value,
                    "std": std_value,
                    "lift": mean_value - base_mean,
                    "positive_rate": chunk_positive_rate,
                }
            )

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
    ) -> PersistenceAssessment:
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

    def is_persistent(self, assessment: PersistenceAssessment | Mapping[str, Any] | Any) -> bool:
        if not isinstance(assessment, PersistenceAssessment):
            assessment = PersistenceAssessment.from_dict(_to_mapping(assessment))
        return assessment.persistent

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
    ) -> PersistenceAssessment:
        return PersistenceAssessment(
            score=0.0,
            persistent=False,
            support=support,
            coverage=coverage,
            window_count=0,
            active_windows=0,
            active_windows_ratio=0.0,
            window_mean=0.0,
            window_std=0.0,
            window_consistency=0.0,
            first_half_mean=0.0,
            second_half_mean=0.0,
            decay_ratio=0.0,
            decay_penalty=0.0,
            longest_run_ratio=0.0,
            stability_score=0.0,
            support_score=0.0,
            active_score=0.0,
            consistency_score=0.0,
            positive_rate=positive_rate,
            signal_positive_rate=signal_positive_rate,
            sample_count=0,
            reasons=tuple(reasons),
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
    ) -> PersistenceAssessment:
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
        lifts = np.asarray([_coerce_float(item.get("lift", item.get("mean", 0.0)), 0.0) for item in window_metrics], dtype=float)
        means = np.asarray([_coerce_float(item.get("mean", 0.0), 0.0) for item in window_metrics], dtype=float)
        positives = np.asarray([_coerce_float(item.get("positive_rate", 0.0), 0.0) for item in window_metrics], dtype=float)

        active_windows_mask = supports > 0
        active_windows = int(np.sum(active_windows_mask))
        active_windows_ratio = active_windows / max(1, window_count)

        window_mean = _safe_mean(means)
        window_std = _safe_std(means)

        # Cohérence : plus les lifts sont proches les uns des autres, mieux c’est.
        lift_std = float(np.std(lifts)) if lifts.size > 0 else 0.0
        lift_abs_mean = float(np.mean(np.abs(lifts))) if lifts.size > 0 else 0.0
        if lift_abs_mean <= 1e-12:
            window_consistency = 0.0 if np.any(np.abs(lifts) > 0) else 1.0
        else:
            window_consistency = 1.0 / (1.0 + lift_std / max(1e-12, lift_abs_mean))

        # Décroissance temporelle : compare la première moitié à la seconde.
        mid = max(1, window_count // 2)
        first_half = lifts[:mid]
        second_half = lifts[mid:] if mid < window_count else lifts[mid - 1 : mid]

        first_half_mean = _safe_mean(first_half)
        second_half_mean = _safe_mean(second_half)

        denom = max(1e-12, abs(first_half_mean) + abs(second_half_mean))
        decay_ratio = abs(first_half_mean - second_half_mean) / denom
        decay_penalty = 1.0 - min(1.0, decay_ratio)

        # Longest run de fenêtres au même signe.
        signed = [_binary_direction(value) for value in lifts]
        if signed:
            sign_ref = 1 if _safe_mean(lifts) >= 0 else -1
            same_sign_runs = _run_lengths([value == sign_ref and value != 0 for value in signed])
            longest_run = max(same_sign_runs, default=0)
        else:
            longest_run = 0

        longest_run_ratio = longest_run / max(1, window_count)

        support_score = min(1.0, support / max(1.0, settings.min_support))
        active_score = min(1.0, active_windows_ratio / max(1e-12, settings.min_active_windows_ratio if settings.min_active_windows_ratio > 0 else 1.0))
        consistency_score = max(0.0, min(1.0, window_consistency))

        stability_score = max(
            0.0,
            min(
                1.0,
                0.40 * active_score
                + 0.35 * consistency_score
                + 0.15 * longest_run_ratio
                + 0.10 * decay_penalty,
            ),
        )

        positive_rate = _coerce_float(base_metadata.get("positive_rate"), float(np.mean(positives)) if positives.size > 0 else 0.0)
        signal_positive_rate = _coerce_float(base_metadata.get("signal_positive_rate"), float(np.mean(positives)) if positives.size > 0 else 0.0)

        reasons: list[str] = []
        if support < settings.min_support:
            reasons.append("insufficient_support")
        if coverage < settings.min_coverage:
            reasons.append("coverage_too_low")
        if window_count < settings.min_windows:
            reasons.append("insufficient_windows")
        if active_windows_ratio < settings.min_active_windows_ratio:
            reasons.append("too_few_active_windows")
        if window_consistency < settings.min_consistency:
            reasons.append("window_inconsistency")
        if settings.require_positive_persistence and _safe_mean(lifts) <= 0:
            reasons.append("non_positive_persistence")

        score = (
            settings.weight_active_windows * active_score
            + settings.weight_consistency * consistency_score
            + settings.weight_stability * stability_score
            + settings.weight_decay * decay_penalty
        )

        # Ajustements légers liés au support et à la couverture.
        score *= min(1.0, 0.50 + 0.50 * support_score)
        score *= min(1.0, 0.50 + 0.50 * min(1.0, coverage / max(1e-12, settings.min_coverage if settings.min_coverage > 0 else 1.0)))

        if settings.require_positive_persistence and _safe_mean(lifts) <= 0:
            score *= 0.0

        persistent = score >= settings.min_score and not reasons

        return PersistenceAssessment(
            score=max(0.0, min(1.0, score)),
            persistent=persistent,
            support=support,
            coverage=coverage,
            window_count=window_count,
            active_windows=active_windows,
            active_windows_ratio=active_windows_ratio,
            window_mean=window_mean,
            window_std=window_std,
            window_consistency=window_consistency,
            first_half_mean=first_half_mean,
            second_half_mean=second_half_mean,
            decay_ratio=decay_ratio,
            decay_penalty=decay_penalty,
            longest_run_ratio=longest_run_ratio,
            stability_score=stability_score,
            support_score=support_score,
            active_score=active_score,
            consistency_score=consistency_score,
            positive_rate=positive_rate,
            signal_positive_rate=signal_positive_rate,
            sample_count=sample_count,
            reasons=tuple(reasons),
            metadata=meta,
        )

    # ==================================================
    # PYTHON PROTOCOL
    # ==================================================

    def __repr__(self) -> str:
        return (
            "PersistenceScorer("
            f"min_windows={self._settings.min_windows}, "
            f"min_score={self._settings.min_score}"
            ")"
        )