"""
==========================================================
Validation Evaluator
==========================================================

Évalue un Candidate issu du module Discovery et décide s'il
peut devenir un ValidatedCandidate.

Le ValidatedCandidate reste un objet métier pur ; ce fichier
porte toute la logique scientifique de validation :
- cohérence structurelle,
- significativité,
- robustesse,
- persistance,
- stabilité temporelle,
- score global,
- motifs de rejet.

L'évaluation s'appuie sur :
- Candidate / Hypothesis
- DatasetLoader / DatasetSplit
- EngineContext
- config.search / config.scoring
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime, timezone
from math import sqrt
from typing import Any
from typing import Iterable
from typing import Mapping
from typing import Sequence

import numpy as np

from config.scoring import ScoringConfig
from config.search import SearchConfig
from core.context import EngineContext
from core.exceptions import ValidationError
from dataset.loader import DatasetLoader
from dataset.loader import DatasetSplit
from models.candidate import Candidate
from models.condition import Condition
from models.enums import ConditionOperator
from models.feature import Feature
from models.fingerprint import fingerprint_model
from models.hypothesis import Hypothesis
from models.validated_candidate import ValidatedCandidate

__all__ = [
    "ValidationSettings",
    "ValidationMetrics",
    "ValidationAssessment",
    "ValidationEvaluator",
]


# ==========================================================
# HELPERS
# ==========================================================

def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


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


def _first_non_none(
    obj: Any,
    *paths: Sequence[str],
    default: Any = None,
) -> Any:
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


def _coerce_family_text(value: Any) -> str:
    if value is None:
        return "unknown"
    return str(value).strip().lower() or "unknown"


def _coerce_operator_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, ConditionOperator):
        return str(value.value).strip().lower()
    return str(value).strip().lower()


def _condition_operator_value(condition: Condition) -> str:
    return _coerce_operator_text(condition.operator)


def _condition_uses_feature_right(condition: Condition) -> bool:
    return isinstance(condition.right, Feature)


def _condition_right_signature(condition: Condition) -> str:
    if isinstance(condition.right, Feature):
        return f"feature:{condition.right.column_index}"
    return f"constant:{repr(condition.right)}"


def _condition_signature(condition: Condition) -> tuple[Any, ...]:
    right = (
        condition.right.column_index
        if isinstance(condition.right, Feature)
        else repr(condition.right)
    )
    return (
        condition.left.column_index,
        _condition_operator_value(condition),
        "feature" if isinstance(condition.right, Feature) else "constant",
        right,
    )


def _dedupe_preserve_order(values: Iterable[Any]) -> tuple[Any, ...]:
    seen: set[str] = set()
    result: list[Any] = []
    for value in values:
        key = repr(value)
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return tuple(result)


def _build_search_config(source: Any | None) -> SearchConfig:
    if isinstance(source, SearchConfig):
        return source

    if source is None:
        return SearchConfig()

    return SearchConfig(
        max_conditions=_coerce_int(
            _first_non_none(source, ("max_conditions",), ("search", "max_conditions"), default=3),
            3,
        ),
        beam_width=_coerce_int(
            _first_non_none(source, ("beam_width",), ("search", "beam_width"), default=200),
            200,
        ),
        max_depth=_coerce_int(
            _first_non_none(source, ("max_depth",), ("search", "max_depth"), default=3),
            3,
        ),
        max_candidates_per_family=_coerce_int(
            _first_non_none(
                source,
                ("max_candidates_per_family",),
                ("search", "max_candidates_per_family"),
                default=100,
            ),
            100,
        ),
        exploration_ratio=_coerce_float(
            _first_non_none(source, ("exploration_ratio",), ("search", "exploration_ratio"), default=0.25),
            0.25,
        ),
        exploitation_ratio=_coerce_float(
            _first_non_none(source, ("exploitation_ratio",), ("search", "exploitation_ratio"), default=0.75),
            0.75,
        ),
        novelty_weight=_coerce_float(
            _first_non_none(source, ("novelty_weight",), ("search", "novelty_weight"), default=0.30),
            0.30,
        ),
        diversity_weight=_coerce_float(
            _first_non_none(source, ("diversity_weight",), ("search", "diversity_weight"), default=0.25),
            0.25,
        ),
        family_balance_weight=_coerce_float(
            _first_non_none(source, ("family_balance_weight",), ("search", "family_balance_weight"), default=0.20),
            0.20,
        ),
        random_seed=_coerce_int(
            _first_non_none(source, ("random_seed",), ("search", "random_seed"), default=42),
            42,
        ),
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


def _is_binary_feature(feature: Feature) -> bool:
    if feature.value_type.value == "boolean":
        return True

    if feature.feature_type.value == "pattern":
        return True

    if feature.name.lower().endswith("_signal"):
        return True

    metadata = feature.metadata or {}
    if _coerce_bool(metadata.get("binary"), False):
        return True

    profile = str(metadata.get("profile", "")).strip().lower()
    return profile == "binary"


def _feature_profile(feature: Feature) -> str:
    name = feature.name.lower()
    metadata = feature.metadata or {}

    explicit = str(
        metadata.get(
            "profile",
            metadata.get("kind", metadata.get("mode", "")),
        )
    ).strip().lower()

    if explicit in {
        "binary",
        "normalized",
        "ratio",
        "distance",
        "oscillator",
        "raw_scale",
        "statistical",
        "generic_numeric",
    }:
        return explicit

    if _is_binary_feature(feature):
        return "binary"

    if any(token in name for token in ("norm", "normalized", "standardized", "zscore", "z_score")):
        return "normalized"

    if any(token in name for token in ("percentile", "percent", "ratio")):
        return "ratio"

    if any(token in name for token in ("distance", "spread", "diff", "delta", "gap", "divergence")):
        return "distance"

    if any(token in name for token in ("rsi", "stoch", "williams", "cci", "macd", "roc", "momentum", "mfi", "adx")):
        return "oscillator"

    if any(token in name for token in ("ema", "sma", "vwap", "close", "open", "high", "low", "volume", "obv", "atr", "sar", "support", "resistance")):
        return "raw_scale"

    if any(token in name for token in ("entropy", "hurst", "autocorr", "skew", "kurtosis", "variance", "cvar", "drawdown", "illiquidity", "lambda", "efficiency", "fractal", "dfa")):
        return "statistical"

    if feature.feature_type.value == "quantitative" or name.startswith("quant_"):
        return "statistical"

    if feature.value_type.value in {"float", "integer", "ordinal"}:
        return "generic_numeric"

    return "unsupported"


def _target_array(y: np.ndarray) -> np.ndarray:
    if y.ndim == 1:
        return y.astype(float, copy=False)

    if y.ndim == 2 and y.shape[1] > 0:
        return y[:, 0].astype(float, copy=False)

    raise ValueError("Unsupported target array shape.")


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


def _window_bounds(n: int, windows: int) -> tuple[tuple[int, int], ...]:
    if n <= 0:
        return ()

    windows = max(1, windows)
    windows = min(windows, n)

    boundaries = np.linspace(0, n, windows + 1, dtype=int)
    output: list[tuple[int, int]] = []

    for i in range(windows):
        start = int(boundaries[i])
        end = int(boundaries[i + 1])
        if end > start:
            output.append((start, end))

    return tuple(output)


def _normalize_indices(n: int, sample_size: int | None, seed: int | None) -> np.ndarray:
    if sample_size is None or sample_size <= 0 or sample_size >= n:
        return np.arange(n, dtype=int)

    if sample_size == n:
        return np.arange(n, dtype=int)

    # Échantillonnage stratifié simple et ordonné dans le temps.
    if sample_size < n:
        idx = np.linspace(0, n - 1, sample_size, dtype=int)
        return np.unique(idx)

    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=sample_size, replace=False)
    return np.sort(idx.astype(int))


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


def _binary_classification_metrics(
    y_true: np.ndarray,
    y_mask: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    if y_true.size == 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "accuracy": 0.0,
            "directional_accuracy": 0.0,
        }

    y_binary = y_true > threshold
    support = y_mask.sum()
    if support <= 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "accuracy": 0.0,
            "directional_accuracy": 0.0,
        }

    tp = int(np.sum(y_binary & y_mask))
    fp = int(np.sum((~y_binary) & y_mask))
    fn = int(np.sum(y_binary & (~y_mask)))
    tn = int(np.sum((~y_binary) & (~y_mask)))

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = (2 * precision * recall / max(1e-12, precision + recall)) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / max(1, y_true.size)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "directional_accuracy": float(precision),
    }


# ==========================================================
# SETTINGS
# ==========================================================

@dataclass(frozen=True, slots=True)
class ValidationSettings:
    """
    Paramètres de validation d'un Candidate.

    Les seuils restent volontairement explicites :
    - minimiser les faux positifs,
    - empêcher les stratégies trop faibles d'entrer dans
      ValidatedCandidate,
    - préserver la stabilité temporelle.
    """

    split_name: str = "validation"
    batch_size: int = 50_000
    sample_size: int | None = None
    windows: int = 4

    min_conditions: int = 1
    max_conditions: int = 12

    min_support: int = 50
    min_coverage: float = 0.005

    min_score: float = 0.60
    min_significance: float = 0.55
    min_robustness: float = 0.55
    min_persistence: float = 0.55
    min_temporal_stability: float = 0.50

    require_positive_lift: bool = False
    positive_target_threshold: float = 0.0

    allow_duplicate_candidates: bool = False
    enable_binary_metrics: bool = True

    scoring_weight_significance: float = 0.30
    scoring_weight_robustness: float = 0.25
    scoring_weight_persistence: float = 0.20
    scoring_weight_temporal: float = 0.15
    scoring_weight_structural: float = 0.10

    random_seed: int = 42

    def __post_init__(self) -> None:
        object.__setattr__(self, "split_name", str(self.split_name).strip() or "validation")
        object.__setattr__(self, "batch_size", max(1, _coerce_int(self.batch_size, 50_000)))
        object.__setattr__(self, "sample_size", None if self.sample_size is None else max(1, _coerce_int(self.sample_size, 1)))
        object.__setattr__(self, "windows", max(1, _coerce_int(self.windows, 4)))

        object.__setattr__(self, "min_conditions", max(1, _coerce_int(self.min_conditions, 1)))
        object.__setattr__(self, "max_conditions", max(1, _coerce_int(self.max_conditions, 12)))

        object.__setattr__(self, "min_support", max(1, _coerce_int(self.min_support, 50)))
        object.__setattr__(self, "min_coverage", min(1.0, max(0.0, _coerce_float(self.min_coverage, 0.005))))

        object.__setattr__(self, "min_score", min(1.0, max(0.0, _coerce_float(self.min_score, 0.60))))
        object.__setattr__(self, "min_significance", min(1.0, max(0.0, _coerce_float(self.min_significance, 0.55))))
        object.__setattr__(self, "min_robustness", min(1.0, max(0.0, _coerce_float(self.min_robustness, 0.55))))
        object.__setattr__(self, "min_persistence", min(1.0, max(0.0, _coerce_float(self.min_persistence, 0.55))))
        object.__setattr__(self, "min_temporal_stability", min(1.0, max(0.0, _coerce_float(self.min_temporal_stability, 0.50))))

        object.__setattr__(self, "require_positive_lift", _coerce_bool(self.require_positive_lift, False))
        object.__setattr__(self, "positive_target_threshold", _coerce_float(self.positive_target_threshold, 0.0))
        object.__setattr__(self, "allow_duplicate_candidates", _coerce_bool(self.allow_duplicate_candidates, False))
        object.__setattr__(self, "enable_binary_metrics", _coerce_bool(self.enable_binary_metrics, True))

        weights = {
            "significance": max(0.0, _coerce_float(self.scoring_weight_significance, 0.30)),
            "robustness": max(0.0, _coerce_float(self.scoring_weight_robustness, 0.25)),
            "persistence": max(0.0, _coerce_float(self.scoring_weight_persistence, 0.20)),
            "temporal": max(0.0, _coerce_float(self.scoring_weight_temporal, 0.15)),
            "structural": max(0.0, _coerce_float(self.scoring_weight_structural, 0.10)),
        }
        object.__setattr__(self, "scoring_weight_significance", weights["significance"])
        object.__setattr__(self, "scoring_weight_robustness", weights["robustness"])
        object.__setattr__(self, "scoring_weight_persistence", weights["persistence"])
        object.__setattr__(self, "scoring_weight_temporal", weights["temporal"])
        object.__setattr__(self, "scoring_weight_structural", weights["structural"])

        total = sum(weights.values())
        if total <= 0:
            object.__setattr__(self, "scoring_weight_significance", 0.30)
            object.__setattr__(self, "scoring_weight_robustness", 0.25)
            object.__setattr__(self, "scoring_weight_persistence", 0.20)
            object.__setattr__(self, "scoring_weight_temporal", 0.15)
            object.__setattr__(self, "scoring_weight_structural", 0.10)

        object.__setattr__(self, "random_seed", _coerce_int(self.random_seed, 42))

    @classmethod
    def from_config(cls, config: Any | None) -> "ValidationSettings":
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
            split_name=_first_non_none(
                validation,
                ("split_name",),
                ("dataset_split",),
                ("split",),
                default="validation",
            ),
            batch_size=_coerce_int(
                _first_non_none(validation, ("batch_size",), ("chunk_size",), default=50_000),
                50_000,
            ),
            sample_size=_first_non_none(
                validation,
                ("sample_size",),
                ("max_rows",),
                default=None,
            ),
            windows=_coerce_int(
                _first_non_none(validation, ("windows",), ("temporal_windows",), default=4),
                4,
            ),
            min_conditions=_coerce_int(
                _first_non_none(validation, ("min_conditions",), default=1),
                1,
            ),
            max_conditions=_coerce_int(
                _first_non_none(validation, ("max_conditions",), default=search.max_conditions),
                search.max_conditions,
            ),
            min_support=_coerce_int(
                _first_non_none(validation, ("min_support",), default=50),
                50,
            ),
            min_coverage=_coerce_float(
                _first_non_none(validation, ("min_coverage",), default=0.005),
                0.005,
            ),
            min_score=_coerce_float(
                _first_non_none(validation, ("min_score",), default=0.60),
                0.60,
            ),
            min_significance=_coerce_float(
                _first_non_none(validation, ("min_significance",), ("significance_threshold",), default=0.55),
                0.55,
            ),
            min_robustness=_coerce_float(
                _first_non_none(validation, ("min_robustness",), default=0.55),
                0.55,
            ),
            min_persistence=_coerce_float(
                _first_non_none(validation, ("min_persistence",), default=0.55),
                0.55,
            ),
            min_temporal_stability=_coerce_float(
                _first_non_none(validation, ("min_temporal_stability",), default=0.50),
                0.50,
            ),
            require_positive_lift=_coerce_bool(
                _first_non_none(validation, ("require_positive_lift",), default=False),
                False,
            ),
            positive_target_threshold=_coerce_float(
                _first_non_none(validation, ("positive_target_threshold",), ("target_threshold",), default=0.0),
                0.0,
            ),
            allow_duplicate_candidates=_coerce_bool(
                _first_non_none(validation, ("allow_duplicate_candidates",), default=False),
                False,
            ),
            enable_binary_metrics=_coerce_bool(
                _first_non_none(validation, ("enable_binary_metrics",), default=True),
                True,
            ),
            scoring_weight_significance=_coerce_float(
                _first_non_none(validation, ("scoring_weight_significance",), default=scoring.novelty),
                scoring.novelty,
            ),
            scoring_weight_robustness=_coerce_float(
                _first_non_none(validation, ("scoring_weight_robustness",), default=scoring.robustness),
                scoring.robustness,
            ),
            scoring_weight_persistence=_coerce_float(
                _first_non_none(validation, ("scoring_weight_persistence",), default=scoring.persistence),
                scoring.persistence,
            ),
            scoring_weight_temporal=_coerce_float(
                _first_non_none(validation, ("scoring_weight_temporal",), default=scoring.diversity),
                scoring.diversity,
            ),
            scoring_weight_structural=_coerce_float(
                _first_non_none(validation, ("scoring_weight_structural",), default=0.10),
                0.10,
            ),
            random_seed=_coerce_int(
                _first_non_none(validation, ("random_seed",), default=search.random_seed),
                search.random_seed,
            ),
        )


# ==========================================================
# METRICS
# ==========================================================

@dataclass(frozen=True, slots=True)
class ValidationMetrics:
    """
    Mesures calculées pendant la validation.
    """

    sample_count: int
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

    significance_score: float
    robustness_score: float
    persistence_score: float
    temporal_stability: float

    structural_score: float
    binary_precision: float = 0.0
    binary_recall: float = 0.0
    binary_f1: float = 0.0
    directional_accuracy: float = 0.0

    positive_rate: float = 0.0
    signal_positive_rate: float = 0.0

    condition_count: int = 0
    feature_count: int = 0
    family_count: int = 0
    operator_count: int = 0

    windows: int = 0
    window_metrics: tuple[dict[str, Any], ...] = ()

    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "sample_count", max(0, _coerce_int(self.sample_count, 0)))
        object.__setattr__(self, "support", max(0, _coerce_int(self.support, 0)))
        object.__setattr__(self, "coverage", min(1.0, max(0.0, _coerce_float(self.coverage, 0.0))))
        object.__setattr__(self, "baseline_mean", float(self.baseline_mean))
        object.__setattr__(self, "baseline_std", max(0.0, float(self.baseline_std)))
        object.__setattr__(self, "signal_mean", float(self.signal_mean))
        object.__setattr__(self, "signal_std", max(0.0, float(self.signal_std)))
        object.__setattr__(self, "lift", float(self.lift))
        object.__setattr__(self, "lift_ratio", float(self.lift_ratio))
        object.__setattr__(self, "t_stat", float(self.t_stat))
        object.__setattr__(self, "effect_size", float(self.effect_size))
        object.__setattr__(self, "significance_score", min(1.0, max(0.0, float(self.significance_score))))
        object.__setattr__(self, "robustness_score", min(1.0, max(0.0, float(self.robustness_score))))
        object.__setattr__(self, "persistence_score", min(1.0, max(0.0, float(self.persistence_score))))
        object.__setattr__(self, "temporal_stability", min(1.0, max(0.0, float(self.temporal_stability))))
        object.__setattr__(self, "structural_score", min(1.0, max(0.0, float(self.structural_score))))
        object.__setattr__(self, "binary_precision", min(1.0, max(0.0, float(self.binary_precision))))
        object.__setattr__(self, "binary_recall", min(1.0, max(0.0, float(self.binary_recall))))
        object.__setattr__(self, "binary_f1", min(1.0, max(0.0, float(self.binary_f1))))
        object.__setattr__(self, "directional_accuracy", min(1.0, max(0.0, float(self.directional_accuracy))))
        object.__setattr__(self, "positive_rate", min(1.0, max(0.0, float(self.positive_rate))))
        object.__setattr__(self, "signal_positive_rate", min(1.0, max(0.0, float(self.signal_positive_rate))))
        object.__setattr__(self, "condition_count", max(0, _coerce_int(self.condition_count, 0)))
        object.__setattr__(self, "feature_count", max(0, _coerce_int(self.feature_count, 0)))
        object.__setattr__(self, "family_count", max(0, _coerce_int(self.family_count, 0)))
        object.__setattr__(self, "operator_count", max(0, _coerce_int(self.operator_count, 0)))
        object.__setattr__(self, "windows", max(0, _coerce_int(self.windows, 0)))
        object.__setattr__(self, "window_metrics", tuple(dict(item) for item in self.window_metrics))
        object.__setattr__(self, "score", min(1.0, max(0.0, float(self.score))))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_count": self.sample_count,
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
            "significance_score": self.significance_score,
            "robustness_score": self.robustness_score,
            "persistence_score": self.persistence_score,
            "temporal_stability": self.temporal_stability,
            "structural_score": self.structural_score,
            "binary_precision": self.binary_precision,
            "binary_recall": self.binary_recall,
            "binary_f1": self.binary_f1,
            "directional_accuracy": self.directional_accuracy,
            "positive_rate": self.positive_rate,
            "signal_positive_rate": self.signal_positive_rate,
            "condition_count": self.condition_count,
            "feature_count": self.feature_count,
            "family_count": self.family_count,
            "operator_count": self.operator_count,
            "windows": self.windows,
            "window_metrics": [dict(item) for item in self.window_metrics],
            "score": self.score,
            "metadata": dict(self.metadata),
        }


# ==========================================================
# ASSESSMENT
# ==========================================================

@dataclass(frozen=True, slots=True)
class ValidationAssessment:
    """
    Diagnostic complet d'une validation.
    """

    candidate_fingerprint: str
    hypothesis_fingerprint: str

    passed: bool
    score: float

    metrics: ValidationMetrics

    rejection_reasons: tuple[str, ...] = ()
    split_name: str = "validation"

    duplicate: bool = False
    validated_candidate: ValidatedCandidate | None = None

    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        object.__setattr__(self, "passed", _coerce_bool(self.passed, False))
        object.__setattr__(self, "score", min(1.0, max(0.0, float(self.score))))
        object.__setattr__(self, "rejection_reasons", tuple(str(reason) for reason in self.rejection_reasons))
        object.__setattr__(self, "split_name", str(self.split_name).strip() or "validation")
        object.__setattr__(self, "duplicate", _coerce_bool(self.duplicate, False))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_fingerprint": self.candidate_fingerprint,
            "hypothesis_fingerprint": self.hypothesis_fingerprint,
            "passed": self.passed,
            "score": self.score,
            "metrics": self.metrics.to_dict(),
            "rejection_reasons": list(self.rejection_reasons),
            "split_name": self.split_name,
            "duplicate": self.duplicate,
            "validated_candidate": (
                None if self.validated_candidate is None else self.validated_candidate.to_dict()
            ),
            "metadata": dict(self.metadata),
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ValidationAssessment":
        return cls(
            candidate_fingerprint=data["candidate_fingerprint"],
            hypothesis_fingerprint=data["hypothesis_fingerprint"],
            passed=_coerce_bool(data.get("passed"), False),
            score=_coerce_float(data.get("score"), 0.0),
            metrics=ValidationMetrics(**_to_mapping(data.get("metrics", {}))),
            rejection_reasons=tuple(data.get("rejection_reasons", ())),
            split_name=data.get("split_name", "validation"),
            duplicate=_coerce_bool(data.get("duplicate"), False),
            validated_candidate=None,
            metadata=_to_mapping(data.get("metadata", {})),
            created_at=(
                datetime.fromisoformat(data["created_at"])
                if isinstance(data.get("created_at"), str) and data.get("created_at")
                else _utc_now()
            ),
        )

    def __repr__(self) -> str:
        return (
            "ValidationAssessment("
            f"passed={self.passed}, "
            f"score={self.score:.4f}, "
            f"reasons={len(self.rejection_reasons)}, "
            f"split='{self.split_name}'"
            ")"
        )


# ==========================================================
# EVALUATOR
# ==========================================================

class ValidationEvaluator:
    """
    Valide un Candidate issu de Discovery.

    Le rôle du Validateur est de transformer un candidat
    structurellement plausible en valide candidat statistique
    si, et seulement si, les mesures dépassent les seuils
    configurés.
    """

    def __init__(
        self,
        *,
        settings: ValidationSettings | None = None,
        search_config: SearchConfig | None = None,
        scoring_config: ScoringConfig | None = None,
        config: Any | None = None,
        context: EngineContext | None = None,
        dataset: DatasetLoader | DatasetSplit | None = None,
        rng: np.random.Generator | int | None = None,
    ) -> None:
        if context is not None:
            config = context.config
            dataset = context.dataset_loader

        self._config = config
        self._settings = settings or ValidationSettings.from_config(config)
        self._search_config = search_config or _build_search_config(_first_non_none(config, ("search",), ("search_config",), default=None))
        self._scoring_config = scoring_config or _build_scoring_config(_first_non_none(config, ("scoring",), ("scoring_config",), default=None))

        if isinstance(rng, np.random.Generator):
            self._rng = rng
        elif rng is not None:
            self._rng = np.random.default_rng(rng)
        else:
            self._rng = np.random.default_rng(self._settings.random_seed)

        self._dataset_source = dataset
        self._history: list[ValidationAssessment] = []
        self._seen_fingerprints: Counter[str] = Counter()
        self._accepted_fingerprints: set[str] = set()
        self._rejection_counts: Counter[str] = Counter()
        self._split_counts: Counter[str] = Counter()

    # ==================================================
    # CONSTRUCTION
    # ==================================================

    @classmethod
    def from_config(
        cls,
        config: Any,
        *,
        context: EngineContext | None = None,
        dataset: DatasetLoader | DatasetSplit | None = None,
        settings: ValidationSettings | None = None,
        search_config: SearchConfig | None = None,
        scoring_config: ScoringConfig | None = None,
        rng: np.random.Generator | int | None = None,
    ) -> "ValidationEvaluator":
        return cls(
            settings=settings,
            search_config=search_config,
            scoring_config=scoring_config,
            config=config,
            context=context,
            dataset=dataset,
            rng=rng,
        )

    @classmethod
    def from_context(
        cls,
        context: EngineContext,
        *,
        settings: ValidationSettings | None = None,
        search_config: SearchConfig | None = None,
        scoring_config: ScoringConfig | None = None,
        rng: np.random.Generator | int | None = None,
    ) -> "ValidationEvaluator":
        return cls(
            settings=settings,
            search_config=search_config,
            scoring_config=scoring_config,
            context=context,
            rng=rng,
        )

    # ==================================================
    # PROPERTIES
    # ==================================================

    @property
    def settings(self) -> ValidationSettings:
        return self._settings

    @property
    def search_config(self) -> SearchConfig:
        return self._search_config

    @property
    def scoring_config(self) -> ScoringConfig:
        return self._scoring_config

    @property
    def history(self) -> tuple[ValidationAssessment, ...]:
        return tuple(self._history)

    @property
    def rejection_counts(self) -> dict[str, int]:
        return dict(self._rejection_counts)

    @property
    def split_counts(self) -> dict[str, int]:
        return dict(self._split_counts)

    @property
    def accepted_count(self) -> int:
        return len(self._accepted_fingerprints)

    @property
    def evaluated_count(self) -> int:
        return len(self._history)

    @property
    def duplicate_count(self) -> int:
        return sum(1 for item in self._history if item.duplicate)

    # ==================================================
    # PUBLIC API
    # ==================================================

    def assess(
        self,
        candidate: Candidate | Hypothesis,
        *,
        context: EngineContext | None = None,
        dataset: DatasetLoader | DatasetSplit | None = None,
        split_name: str | None = None,
        batch_size: int | None = None,
        sample_size: int | None = None,
    ) -> ValidationAssessment:
        candidate_obj, hypothesis = self._normalize_candidate(candidate)
        candidate_fingerprint = fingerprint_model(candidate_obj)
        hypothesis_fingerprint = fingerprint_model(hypothesis)

        split = self._resolve_split(
            context=context,
            dataset=dataset,
            split_name=split_name,
        )

        settings = self._settings
        batch_size = max(1, _coerce_int(batch_size if batch_size is not None else settings.batch_size, settings.batch_size))
        sample_size = _coerce_int(sample_size if sample_size is not None else settings.sample_size, settings.sample_size or 0) if (sample_size is not None or settings.sample_size is not None) else None

        if len(hypothesis.conditions) < settings.min_conditions:
            return self._reject_assessment(
                candidate_obj,
                hypothesis,
                candidate_fingerprint=candidate_fingerprint,
                hypothesis_fingerprint=hypothesis_fingerprint,
                split_name=split_name or settings.split_name,
                reasons=("too_few_conditions",),
                metrics=self._empty_metrics(hypothesis, split_name or settings.split_name),
            )

        if len(hypothesis.conditions) > settings.max_conditions:
            return self._reject_assessment(
                candidate_obj,
                hypothesis,
                candidate_fingerprint=candidate_fingerprint,
                hypothesis_fingerprint=hypothesis_fingerprint,
                split_name=split_name or settings.split_name,
                reasons=("too_many_conditions",),
                metrics=self._empty_metrics(hypothesis, split_name or settings.split_name),
            )

        if not settings.allow_duplicate_candidates and candidate_fingerprint in self._seen_fingerprints:
            return self._reject_assessment(
                candidate_obj,
                hypothesis,
                candidate_fingerprint=candidate_fingerprint,
                hypothesis_fingerprint=hypothesis_fingerprint,
                split_name=split_name or settings.split_name,
                reasons=("duplicate_candidate",),
                duplicate=True,
                metrics=self._empty_metrics(hypothesis, split_name or settings.split_name),
            )

        metrics = self._evaluate_on_split(
            hypothesis,
            split,
            batch_size=batch_size,
            sample_size=sample_size,
            split_name=split_name or settings.split_name,
        )

        reasons = self._rejection_reasons(metrics, hypothesis, candidate_fingerprint)

        passed = len(reasons) == 0

        validated_candidate: ValidatedCandidate | None = None
        if passed:
            validated_candidate = ValidatedCandidate(
                candidate=candidate_obj,
                metrics={
                    "validation": metrics.to_dict(),
                    "assessment": {
                        "passed": True,
                        "score": metrics.score,
                        "split_name": split_name or settings.split_name,
                    },
                },
            )

        assessment = ValidationAssessment(
            candidate_fingerprint=candidate_fingerprint,
            hypothesis_fingerprint=hypothesis_fingerprint,
            passed=passed,
            score=metrics.score,
            metrics=metrics,
            rejection_reasons=tuple(reasons),
            split_name=split_name or settings.split_name,
            duplicate=False,
            validated_candidate=validated_candidate,
            metadata={
                "candidate": candidate_obj.to_dict(),
                "hypothesis": hypothesis.to_dict(),
            },
        )

        self._commit(assessment)
        return assessment

    def validate(
        self,
        candidate: Candidate | Hypothesis,
        *,
        context: EngineContext | None = None,
        dataset: DatasetLoader | DatasetSplit | None = None,
        split_name: str | None = None,
        batch_size: int | None = None,
        sample_size: int | None = None,
    ) -> ValidatedCandidate:
        assessment = self.assess(
            candidate,
            context=context,
            dataset=dataset,
            split_name=split_name,
            batch_size=batch_size,
            sample_size=sample_size,
        )

        if not assessment.passed or assessment.validated_candidate is None:
            raise ValidationError(
                f"Validation failed: {', '.join(assessment.rejection_reasons) or 'unknown reason'}"
            )

        return assessment.validated_candidate

    def evaluate(
        self,
        candidate: Candidate | Hypothesis,
        *,
        context: EngineContext | None = None,
        dataset: DatasetLoader | DatasetSplit | None = None,
        split_name: str | None = None,
        batch_size: int | None = None,
        sample_size: int | None = None,
    ) -> ValidationAssessment:
        return self.assess(
            candidate,
            context=context,
            dataset=dataset,
            split_name=split_name,
            batch_size=batch_size,
            sample_size=sample_size,
        )

    def evaluate_hypothesis(
        self,
        hypothesis: Hypothesis,
        *,
        context: EngineContext | None = None,
        dataset: DatasetLoader | DatasetSplit | None = None,
        split_name: str | None = None,
        batch_size: int | None = None,
        sample_size: int | None = None,
    ) -> ValidationAssessment:
        return self.assess(
            hypothesis,
            context=context,
            dataset=dataset,
            split_name=split_name,
            batch_size=batch_size,
            sample_size=sample_size,
        )

    def evaluate_candidate(
        self,
        candidate: Candidate,
        *,
        context: EngineContext | None = None,
        dataset: DatasetLoader | DatasetSplit | None = None,
        split_name: str | None = None,
        batch_size: int | None = None,
        sample_size: int | None = None,
    ) -> ValidationAssessment:
        return self.assess(
            candidate,
            context=context,
            dataset=dataset,
            split_name=split_name,
            batch_size=batch_size,
            sample_size=sample_size,
        )

    def is_duplicate(self, candidate: Candidate | Hypothesis) -> bool:
        candidate_obj, _ = self._normalize_candidate(candidate)
        return fingerprint_model(candidate_obj) in self._seen_fingerprints

    def clear(self) -> None:
        self._history.clear()
        self._seen_fingerprints.clear()
        self._accepted_fingerprints.clear()
        self._rejection_counts.clear()
        self._split_counts.clear()

    # ==================================================
    # INTERNAL NORMALIZATION
    # ==================================================

    def _normalize_candidate(
        self,
        candidate: Candidate | Hypothesis,
    ) -> tuple[Candidate, Hypothesis]:
        if isinstance(candidate, Candidate):
            return candidate, candidate.hypothesis

        if isinstance(candidate, Hypothesis):
            return Candidate(hypothesis=candidate), candidate

        raise TypeError("candidate must be a Candidate or a Hypothesis.")

    def _resolve_dataset_source(
        self,
        *,
        context: EngineContext | None = None,
        dataset: DatasetLoader | DatasetSplit | None = None,
    ) -> DatasetLoader | DatasetSplit:
        if dataset is not None:
            return dataset

        if context is not None:
            return context.dataset_loader

        if self._dataset_source is not None:
            return self._dataset_source

        raise ValidationError("A dataset or context is required for validation.")

    def _resolve_split(
        self,
        *,
        context: EngineContext | None = None,
        dataset: DatasetLoader | DatasetSplit | None = None,
        split_name: str | None = None,
    ) -> DatasetSplit:
        source = self._resolve_dataset_source(context=context, dataset=dataset)

        if isinstance(source, DatasetSplit):
            return source

        name = split_name or self._settings.split_name

        if hasattr(source, "get") and callable(source.get):
            try:
                split = source.get(name)
                if isinstance(split, DatasetSplit):
                    return split
            except Exception:
                pass

        if hasattr(source, name) and isinstance(getattr(source, name), DatasetSplit):
            return getattr(source, name)

        if name == "validation" and hasattr(source, "validation") and callable(source.validation):
            split = source.validation()
            if isinstance(split, DatasetSplit):
                return split

        if name == "train" and hasattr(source, "train") and callable(source.train):
            split = source.train()
            if isinstance(split, DatasetSplit):
                return split

        if name == "test" and hasattr(source, "test") and callable(source.test):
            split = source.test()
            if isinstance(split, DatasetSplit):
                return split

        raise ValidationError(f"Unable to resolve dataset split '{name}'.")

    # ==================================================
    # INTERNAL EVALUATION
    # ==================================================

    def _evaluate_on_split(
        self,
        hypothesis: Hypothesis,
        split: DatasetSplit,
        *,
        batch_size: int,
        sample_size: int | None,
        split_name: str,
    ) -> ValidationMetrics:
        x = split.X
        y = split.Y

        if x.ndim != 2:
            raise ValidationError("Dataset X must be 2-dimensional.")
        if y.ndim not in (1, 2):
            raise ValidationError("Dataset Y must be 1D or 2D.")

        n_rows = int(x.shape[0])
        if n_rows <= 0:
            raise ValidationError("Dataset split is empty.")

        if y.shape[0] != n_rows:
            raise ValidationError("X and Y must have the same number of rows.")

        indices = _normalize_indices(n_rows, sample_size, self._settings.random_seed)
        n_selected = int(indices.size)
        if n_selected <= 0:
            raise ValidationError("No sample selected for validation.")

        windows = _window_bounds(n_selected, self._settings.windows)
        window_stats = [
            {
                "start": start,
                "end": end,
                "count": 0,
                "support": 0,
                "baseline_mean": 0.0,
                "signal_mean": 0.0,
                "lift": 0.0,
                "coverage": 0.0,
            }
            for start, end in windows
        ]

        total_sum = 0.0
        total_sq_sum = 0.0
        total_support = 0
        signal_sum = 0.0
        signal_sq_sum = 0.0

        signal_targets: list[np.ndarray] = []
        baseline_targets: list[np.ndarray] = []
        # IMPORTANT : on accumule le mask COMPLET par batch
        # (full-size) pour pouvoir reconstruire un mask global
        # aligné avec baseline_array lors du calcul des métriques
        # binaires. Avant ce fix, le mask était mal aligné et
        # provoquait un crash en mode MIDAS.
        signal_masks: list[np.ndarray] = []

        positive_total = 0
        positive_signal = 0

        feature_names: list[str] = []
        families: set[str] = set()
        operators: set[str] = set()
        features: set[int] = set()

        target_threshold = self._settings.positive_target_threshold

        # Previous rows are required for crossover style operators.
        previous_rows: dict[int, float] = {}

        # Iterate by batches over selected indices.
        for batch_start in range(0, n_selected, batch_size):
            batch_end = min(batch_start + batch_size, n_selected)
            batch_indices = indices[batch_start:batch_end]
            x_batch = x[batch_indices]
            y_batch = _target_array(y[batch_indices])

            if x_batch.ndim != 2:
                raise ValidationError("Batch X must be 2-dimensional.")
            if y_batch.ndim != 1:
                raise ValidationError("Target array must be 1-dimensional after normalization.")

            batch_mask = self._evaluate_hypothesis_batch(
                hypothesis,
                x_batch,
                previous_rows=previous_rows,
            )

            if batch_mask.shape[0] != y_batch.shape[0]:
                raise ValidationError("Signal mask and target batch must have the same length.")

            batch_size_actual = int(y_batch.shape[0])
            support = int(batch_mask.sum())

            total_sum += float(np.sum(y_batch))
            total_sq_sum += float(np.sum(y_batch ** 2))
            total_support += support

            # On stocke TOUJOURS le mask complet du batch (True =
            # sample qui matche l'hypothèse, False = sample qui
            # ne matche pas). Il sert aux métriques binaires.
            signal_masks.append(np.asarray(batch_mask, dtype=bool))

            if support > 0:
                signal_values = y_batch[batch_mask]
                signal_sum += float(np.sum(signal_values))
                signal_sq_sum += float(np.sum(signal_values ** 2))
                signal_targets.append(signal_values)
            baseline_targets.append(y_batch)

            if self._settings.enable_binary_metrics:
                positive_total += int(np.sum(y_batch > target_threshold))
                positive_signal += int(np.sum(y_batch[batch_mask] > target_threshold))

            # Temporal windows.
            for window_index, (start, end) in enumerate(windows):
                window_offset_start = max(start, batch_start)
                window_offset_end = min(end, batch_end)
                if window_offset_end <= window_offset_start:
                    continue

                local_start = window_offset_start - batch_start
                local_end = window_offset_end - batch_start
                local_mask = batch_mask[local_start:local_end]
                local_y = y_batch[local_start:local_end]
                local_count = int(local_y.shape[0])
                local_support = int(local_mask.sum())

                if local_count <= 0:
                    continue

                window_stats[window_index]["count"] += local_count
                window_stats[window_index]["support"] += local_support
                window_stats[window_index]["baseline_mean"] += float(np.sum(local_y))
                window_stats[window_index]["signal_mean"] += float(np.sum(local_y[local_mask])) if local_support > 0 else 0.0

            # collect feature metadata for metrics
            for condition in hypothesis.conditions:
                families.add(condition.left.economic_family.value)
                features.add(condition.left.column_index)
                operators.add(_condition_operator_value(condition))
                feature_names.append(condition.left.name)

        baseline_mean = total_sum / n_selected
        baseline_var = max(0.0, total_sq_sum / n_selected - baseline_mean ** 2)
        baseline_std = sqrt(baseline_var)

        if total_support > 0:
            signal_mean = signal_sum / total_support
            signal_var = max(0.0, signal_sq_sum / total_support - signal_mean ** 2)
            signal_std = sqrt(signal_var)
        else:
            signal_mean = 0.0
            signal_std = 0.0

        coverage = total_support / n_selected
        lift = signal_mean - baseline_mean
        lift_ratio = signal_mean / baseline_mean if abs(baseline_mean) > 1e-12 else 0.0

        signal_array = np.concatenate(signal_targets) if signal_targets else np.array([], dtype=float)
        baseline_array = np.concatenate(baseline_targets) if baseline_targets else np.array([], dtype=float)
        # mask complet aligné avec baseline_array : True si
        # l'hypothèse matche le sample, False sinon. C'est ce
        # mask qu'on passe aux métriques binaires.
        signal_mask_full = np.concatenate(signal_masks) if signal_masks else np.zeros(baseline_array.size, dtype=bool)

        t_stat = _t_statistic(signal_array, baseline_array)
        effect_size = _cohen_d(signal_array, baseline_array)

        significance_score = self._significance_score(
            t_stat=t_stat,
            effect_size=effect_size,
            lift=lift,
            support=total_support,
            total=n_selected,
        )

        temporal_stability, window_metrics = self._temporal_stability(
            windows=windows,
            window_stats=window_stats,
        )

        persistence_score = self._persistence_score(
            signal_targets=signal_targets,
            window_metrics=window_metrics,
            support=total_support,
            total=n_selected,
        )

        robustness_score = self._robustness_score(
            significance_score=significance_score,
            temporal_stability=temporal_stability,
            persistence_score=persistence_score,
            coverage=coverage,
            support=total_support,
            total=n_selected,
        )

        structural_score = self._structural_score(hypothesis)

        binary_precision = 0.0
        binary_recall = 0.0
        binary_f1 = 0.0
        directional_accuracy = 0.0

        positive_rate = positive_total / n_selected if n_selected > 0 else 0.0
        signal_positive_rate = positive_signal / total_support if total_support > 0 else 0.0

        if self._settings.enable_binary_metrics and total_support > 0:
            # signal_mask_full est maintenant aligné avec
            # baseline_array (True = sample qui matche l'hypothèse).
            # Avant ce fix, le mask était mal aligné (taille
            # différente) et _binary_classification_metrics levait
            # un ValueError sur les opérations booléennes.
            binary_metrics = _binary_classification_metrics(
                baseline_array,
                signal_mask_full,
                target_threshold,
            )
            binary_precision = binary_metrics["precision"]
            binary_recall = binary_metrics["recall"]
            binary_f1 = binary_metrics["f1"]
            directional_accuracy = binary_metrics["directional_accuracy"]

        score = self._overall_score(
            significance_score=significance_score,
            robustness_score=robustness_score,
            persistence_score=persistence_score,
            temporal_stability=temporal_stability,
            structural_score=structural_score,
            coverage=coverage,
            lift=lift,
            support=total_support,
            baseline_mean=baseline_mean,
        )

        feature_count = len(features)
        family_count = len(families)
        operator_count = len(operators)

        return ValidationMetrics(
            sample_count=n_selected,
            support=total_support,
            coverage=coverage,
            baseline_mean=baseline_mean,
            baseline_std=baseline_std,
            signal_mean=signal_mean,
            signal_std=signal_std,
            lift=lift,
            lift_ratio=lift_ratio,
            t_stat=t_stat,
            effect_size=effect_size,
            significance_score=significance_score,
            robustness_score=robustness_score,
            persistence_score=persistence_score,
            temporal_stability=temporal_stability,
            structural_score=structural_score,
            binary_precision=binary_precision,
            binary_recall=binary_recall,
            binary_f1=binary_f1,
            directional_accuracy=directional_accuracy,
            positive_rate=positive_rate,
            signal_positive_rate=signal_positive_rate,
            condition_count=len(hypothesis.conditions),
            feature_count=feature_count,
            family_count=family_count,
            operator_count=operator_count,
            windows=len(windows),
            window_metrics=tuple(window_metrics),
            score=score,
            metadata={
                "split_name": split_name,
                "feature_names": _dedupe_preserve_order(feature_names),
            },
        )

    def _evaluate_hypothesis_batch(
        self,
        hypothesis: Hypothesis,
        x_batch: np.ndarray,
        *,
        previous_rows: dict[int, float],
    ) -> np.ndarray:
        mask = np.ones(x_batch.shape[0], dtype=bool)

        for condition in hypothesis.conditions:
            condition_mask, updated_previous = self._evaluate_condition_batch(
                condition,
                x_batch,
                previous_rows=previous_rows,
            )
            previous_rows.update(updated_previous)
            mask &= condition_mask

            if not mask.any():
                break

        return mask

    def _evaluate_condition_batch(
        self,
        condition: Condition,
        x_batch: np.ndarray,
        *,
        previous_rows: dict[int, float],
    ) -> tuple[np.ndarray, dict[int, float]]:
        left_idx = condition.left.column_index
        left = x_batch[:, left_idx]
        operator = _condition_operator_value(condition)
        updated_previous: dict[int, float] = {}

        if isinstance(condition.right, Feature):
            right_idx = condition.right.column_index
            right = x_batch[:, right_idx]
        else:
            right = condition.right

        if operator in {"cross_over", "crossunder", "cross_under"}:
            # Normalisation du nom si l'enum varie.
            cross_under = operator in {"crossunder", "cross_under"}
            return self._evaluate_crossover_condition(
                left_idx=left_idx,
                left=left,
                right=right,
                previous_rows=previous_rows,
                cross_under=cross_under,
            )

        if operator in {"between"}:
            return self._evaluate_between_condition(left, right), updated_previous

        if operator in {"is_true"}:
            return self._evaluate_truthy_condition(left, positive=True), updated_previous

        if operator in {"is_false"}:
            return self._evaluate_truthy_condition(left, positive=False), updated_previous

        if operator in {"gt", "greater_than", ">"}:
            return (left > right), updated_previous

        if operator in {"ge", "gte", "greater_equal", ">="}:
            return (left >= right), updated_previous

        if operator in {"lt", "less_than", "<"}:
            return (left < right), updated_previous

        if operator in {"le", "lte", "less_equal", "<="}:
            return (left <= right), updated_previous

        if operator in {"eq", "=="}:
            return np.isclose(left, right, equal_nan=False), updated_previous

        if operator in {"ne", "neq", "!="}:
            return ~np.isclose(left, right, equal_nan=False), updated_previous

        if operator in {"cross_over", "crossover"}:
            return self._evaluate_crossover_condition(
                left_idx=left_idx,
                left=left,
                right=right,
                previous_rows=previous_rows,
                cross_under=False,
            )

        if operator in {"cross_under", "crossunder"}:
            return self._evaluate_crossover_condition(
                left_idx=left_idx,
                left=left,
                right=right,
                previous_rows=previous_rows,
                cross_under=True,
            )

        # fallback safe
        return np.zeros(x_batch.shape[0], dtype=bool), updated_previous

    def _evaluate_crossover_condition(
        self,
        *,
        left_idx: int,
        left: np.ndarray,
        right: Any,
        previous_rows: dict[int, float],
        cross_under: bool,
    ) -> tuple[np.ndarray, dict[int, float]]:
        if isinstance(right, np.ndarray):
            right_values = right
        else:
            right_values = np.full(left.shape[0], right, dtype=float if np.issubdtype(left.dtype, np.number) else object)

        if left.shape[0] == 0:
            return np.zeros(0, dtype=bool), {}

        prev_left = np.empty_like(left)
        prev_right = np.empty_like(right_values)

        # Première observation : on récupère l'état précédent si disponible.
        if left_idx in previous_rows:
            prev_left[0] = previous_rows[left_idx]
        else:
            prev_left[0] = left[0]

        if isinstance(right_values, np.ndarray) and right_values.shape[0] > 0:
            prev_right[0] = right_values[0]
        else:
            prev_right[0] = right_values

        if left.shape[0] > 1:
            prev_left[1:] = left[:-1]
            if isinstance(right_values, np.ndarray):
                prev_right[1:] = right_values[:-1]
            else:
                prev_right[1:] = right_values

        if cross_under:
            mask = (prev_left >= prev_right) & (left < right_values)
        else:
            mask = (prev_left <= prev_right) & (left > right_values)

        previous_rows[left_idx] = float(left[-1])
        return mask, {left_idx: float(left[-1])}

    def _evaluate_between_condition(
        self,
        left: np.ndarray,
        right: Any,
    ) -> np.ndarray:
        if isinstance(right, (tuple, list, np.ndarray)) and len(right) == 2:
            low, high = right
        else:
            return np.zeros(left.shape[0], dtype=bool)

        return (left >= low) & (left <= high)

    def _evaluate_truthy_condition(
        self,
        left: np.ndarray,
        *,
        positive: bool,
    ) -> np.ndarray:
        if positive:
            return left.astype(bool, copy=False)
        return ~left.astype(bool, copy=False)

    def _signal_mask_from_array(
        self,
        signal_targets: list[np.ndarray],
        baseline: np.ndarray,
        support: int,
    ) -> np.ndarray:
        """
        Conservé pour rétro-compatibilité externe.

        Le calcul correct des métriques binaires se fait
        désormais via `signal_mask_full` accumulé directement
        dans `_evaluate_on_split`. Cette méthode n'est plus
        utilisée par l'Engine et n'est PAS appelée.

        Sa signature est préservée pour ne pas casser
        d'éventuels imports externes, mais elle lève une
        DeprecationWarning si elle est appelée.
        """
        import warnings
        warnings.warn(
            "_signal_mask_from_array is deprecated and no "
            "longer used by the Engine. Binary metrics are now "
            "computed via the proper signal_mask_full accumulated "
            "in _evaluate_on_split.",
            DeprecationWarning,
            stacklevel=2,
        )
        if baseline.size == 0:
            return np.zeros(0, dtype=bool)
        if not signal_targets:
            return np.zeros(baseline.size, dtype=bool)
        # Comportement historique (incorrect) conservé pour
        # rétro-compat : retourne un mask basé sur le dernier
        # batch. NE PAS UTILISER.
        last = signal_targets[-1]
        return last > self._settings.positive_target_threshold

    # ==================================================
    # METRIC COMPONENTS
    # ==================================================

    def _significance_score(
        self,
        *,
        t_stat: float,
        effect_size: float,
        lift: float,
        support: int,
        total: int,
    ) -> float:
        if support <= 0 or total <= 0:
            return 0.0

        t_component = 1.0 - np.exp(-abs(t_stat) / 3.0)
        effect_component = 1.0 - np.exp(-abs(effect_size) / 1.0)
        support_component = min(1.0, support / max(1.0, total * 0.10))

        direction_component = 1.0
        if self._settings.require_positive_lift:
            direction_component = 1.0 if lift > 0 else 0.0

        return float(
            max(
                0.0,
                min(1.0, 0.45 * t_component + 0.35 * effect_component + 0.20 * support_component * direction_component),
            )
        )

    def _temporal_stability(
        self,
        *,
        windows: tuple[tuple[int, int], ...],
        window_stats: list[dict[str, Any]],
    ) -> tuple[float, tuple[dict[str, Any], ...]]:
        if not windows:
            return 0.0, ()

        lifts: list[float] = []
        coverage_values: list[float] = []
        details: list[dict[str, Any]] = []

        for info in window_stats:
            count = max(0, _coerce_int(info["count"], 0))
            support = max(0, _coerce_int(info["support"], 0))
            if count <= 0:
                baseline_mean = 0.0
                signal_mean = 0.0
                lift = 0.0
                coverage = 0.0
            else:
                baseline_mean = _coerce_float(info["baseline_mean"], 0.0) / count
                signal_mean = _coerce_float(info["signal_mean"], 0.0) / support if support > 0 else 0.0
                lift = signal_mean - baseline_mean
                coverage = support / count

            lifts.append(lift)
            coverage_values.append(coverage)
            details.append(
                {
                    "start": info["start"],
                    "end": info["end"],
                    "count": count,
                    "support": support,
                    "baseline_mean": baseline_mean,
                    "signal_mean": signal_mean,
                    "lift": lift,
                    "coverage": coverage,
                }
            )

        if len(lifts) <= 1:
            stability = 1.0 if any(c > 0 for c in coverage_values) else 0.0
            return stability, tuple(details)

        lift_std = float(np.std(np.asarray(lifts, dtype=float)))
        lift_mean_abs = float(np.mean(np.abs(np.asarray(lifts, dtype=float))))
        coverage_mean = float(np.mean(np.asarray(coverage_values, dtype=float)))

        consistency = 1.0 / (1.0 + lift_std / max(1e-9, lift_mean_abs if lift_mean_abs > 0 else 1.0))
        stability = max(0.0, min(1.0, 0.55 * consistency + 0.45 * coverage_mean))
        return stability, tuple(details)

    def _persistence_score(
        self,
        *,
        signal_targets: list[np.ndarray],
        window_metrics: tuple[dict[str, Any], ...],
        support: int,
        total: int,
    ) -> float:
        if support <= 0 or total <= 0:
            return 0.0

        window_supports = [int(item.get("support", 0)) for item in window_metrics]
        active_windows = sum(1 for value in window_supports if value > 0)
        window_ratio = active_windows / max(1, len(window_supports))

        if signal_targets:
            support_ratio = min(1.0, support / total)
        else:
            support_ratio = 0.0

        # Variance intra-fenêtre : plus elle est faible, plus la persistance est bonne.
        lift_values = np.asarray([float(item.get("lift", 0.0)) for item in window_metrics], dtype=float)
        if lift_values.size > 1:
            lift_consistency = 1.0 / (1.0 + float(np.std(lift_values)))
        else:
            lift_consistency = 1.0 if support > 0 else 0.0

        return max(0.0, min(1.0, 0.45 * window_ratio + 0.30 * support_ratio + 0.25 * lift_consistency))

    def _robustness_score(
        self,
        *,
        significance_score: float,
        temporal_stability: float,
        persistence_score: float,
        coverage: float,
        support: int,
        total: int,
    ) -> float:
        support_component = min(1.0, support / max(1.0, total * 0.05))
        coverage_component = min(1.0, coverage / max(1e-9, self._settings.min_coverage if self._settings.min_coverage > 0 else 1.0))
        return max(
            0.0,
            min(
                1.0,
                0.40 * significance_score
                + 0.30 * temporal_stability
                + 0.20 * persistence_score
                + 0.10 * min(1.0, 0.5 * support_component + 0.5 * coverage_component),
            ),
        )

    def _structural_score(self, hypothesis: Hypothesis) -> float:
        if not hypothesis.conditions:
            return 0.0

        signatures = {_condition_signature(condition) for condition in hypothesis.conditions}
        features = {condition.left.column_index for condition in hypothesis.conditions}
        families = {condition.left.economic_family.value for condition in hypothesis.conditions}
        operators = {_condition_operator_value(condition) for condition in hypothesis.conditions}

        uniqueness = (
            len(signatures) / max(1.0, len(hypothesis.conditions)),
            len(features) / max(1.0, len(hypothesis.conditions)),
            len(families) / max(1.0, len(hypothesis.conditions)),
            len(operators) / max(1.0, len(hypothesis.conditions)),
        )
        repetition_penalty = 1.0 / (1.0 + max(0, len(hypothesis.conditions) - len(signatures)))
        size_penalty = 1.0 / (1.0 + 0.10 * max(0, len(hypothesis.conditions) - 1))

        score = (
            0.35 * min(1.0, uniqueness[0])
            + 0.25 * min(1.0, uniqueness[1])
            + 0.20 * min(1.0, uniqueness[2])
            + 0.20 * min(1.0, uniqueness[3])
        )
        score *= repetition_penalty
        score *= size_penalty
        return max(0.0, min(1.0, score))

    def _overall_score(
        self,
        *,
        significance_score: float,
        robustness_score: float,
        persistence_score: float,
        temporal_stability: float,
        structural_score: float,
        coverage: float,
        lift: float,
        support: int,
        baseline_mean: float,
    ) -> float:
        weights = self._settings

        score = (
            weights.scoring_weight_significance * significance_score
            + weights.scoring_weight_robustness * robustness_score
            + weights.scoring_weight_persistence * persistence_score
            + weights.scoring_weight_temporal * temporal_stability
            + weights.scoring_weight_structural * structural_score
        )

        # Ajustement léger en fonction de la couverture et de la direction.
        support_component = min(1.0, support / max(1.0, 0.10 * max(1, support)))
        coverage_component = min(1.0, coverage / max(1e-9, self._settings.min_coverage if self._settings.min_coverage > 0 else 1.0))
        direction_component = 1.0
        if self._settings.require_positive_lift:
            direction_component = 1.0 if lift > 0 else 0.0
        else:
            direction_component = 1.0 if abs(lift) > 0 else 0.85

        baseline_component = 1.0
        if abs(baseline_mean) > 1e-12:
            lift_ratio = abs(lift / baseline_mean)
            baseline_component = max(0.70, min(1.10, 0.90 + min(0.20, lift_ratio)))

        score *= (0.85 + 0.15 * coverage_component)
        score *= (0.85 + 0.15 * direction_component)
        score *= (0.90 + 0.10 * baseline_component)

        return max(0.0, min(1.0, score))

    # ==================================================
    # REJECTIONS / COMMIT
    # ==================================================

    def _rejection_reasons(
        self,
        metrics: ValidationMetrics,
        hypothesis: Hypothesis,
        candidate_fingerprint: str,
    ) -> list[str]:
        reasons: list[str] = []

        if not self._settings.allow_duplicate_candidates and candidate_fingerprint in self._seen_fingerprints:
            reasons.append("duplicate_candidate")

        if metrics.condition_count < self._settings.min_conditions:
            reasons.append("too_few_conditions")

        if metrics.condition_count > self._settings.max_conditions:
            reasons.append("too_many_conditions")

        if metrics.support < self._settings.min_support:
            reasons.append("insufficient_support")

        if metrics.coverage < self._settings.min_coverage:
            reasons.append("coverage_too_low")

        if metrics.significance_score < self._settings.min_significance:
            reasons.append("significance_too_low")

        if metrics.robustness_score < self._settings.min_robustness:
            reasons.append("robustness_too_low")

        if metrics.persistence_score < self._settings.min_persistence:
            reasons.append("persistence_too_low")

        if metrics.temporal_stability < self._settings.min_temporal_stability:
            reasons.append("temporal_stability_too_low")

        if metrics.score < self._settings.min_score:
            reasons.append("overall_score_too_low")

        if self._settings.require_positive_lift and metrics.lift <= 0:
            reasons.append("non_positive_lift")

        if len(hypothesis.conditions) != len(_dedupe_preserve_order(hypothesis.conditions)):
            reasons.append("duplicate_conditions")

        return reasons

    def _reject_assessment(
        self,
        candidate: Candidate,
        hypothesis: Hypothesis,
        *,
        candidate_fingerprint: str,
        hypothesis_fingerprint: str,
        split_name: str,
        reasons: Sequence[str],
        duplicate: bool = False,
        metrics: ValidationMetrics,
    ) -> ValidationAssessment:
        assessment = ValidationAssessment(
            candidate_fingerprint=candidate_fingerprint,
            hypothesis_fingerprint=hypothesis_fingerprint,
            passed=False,
            score=metrics.score,
            metrics=metrics,
            rejection_reasons=tuple(reasons),
            split_name=split_name,
            duplicate=duplicate,
            validated_candidate=None,
            metadata={
                "candidate": candidate.to_dict(),
                "hypothesis": hypothesis.to_dict(),
            },
        )
        self._commit(assessment)
        return assessment

    def _commit(self, assessment: ValidationAssessment) -> None:
        self._history.append(assessment)
        self._seen_fingerprints[assessment.candidate_fingerprint] += 1
        self._split_counts[assessment.split_name] += 1

        for reason in assessment.rejection_reasons:
            self._rejection_counts[reason] += 1

        if assessment.passed and assessment.validated_candidate is not None:
            self._accepted_fingerprints.add(assessment.candidate_fingerprint)

    def _empty_metrics(self, hypothesis: Hypothesis, split_name: str) -> ValidationMetrics:
        families = {condition.left.economic_family.value for condition in hypothesis.conditions}
        features = {condition.left.column_index for condition in hypothesis.conditions}
        operators = {_condition_operator_value(condition) for condition in hypothesis.conditions}
        return ValidationMetrics(
            sample_count=0,
            support=0,
            coverage=0.0,
            baseline_mean=0.0,
            baseline_std=0.0,
            signal_mean=0.0,
            signal_std=0.0,
            lift=0.0,
            lift_ratio=0.0,
            t_stat=0.0,
            effect_size=0.0,
            significance_score=0.0,
            robustness_score=0.0,
            persistence_score=0.0,
            temporal_stability=0.0,
            structural_score=self._structural_score(hypothesis),
            binary_precision=0.0,
            binary_recall=0.0,
            binary_f1=0.0,
            directional_accuracy=0.0,
            positive_rate=0.0,
            signal_positive_rate=0.0,
            condition_count=len(hypothesis.conditions),
            feature_count=len(features),
            family_count=len(families),
            operator_count=len(operators),
            windows=0,
            window_metrics=(),
            score=0.0,
            metadata={
                "split_name": split_name,
                "reason": "empty_metrics",
            },
        )

    # ==================================================
    # PYTHON PROTOCOL
    # ==================================================

    def __contains__(self, item: Candidate | Hypothesis | str) -> bool:
        if isinstance(item, str):
            return item in self._seen_fingerprints

        if isinstance(item, Candidate):
            return fingerprint_model(item) in self._seen_fingerprints

        if isinstance(item, Hypothesis):
            return fingerprint_model(item) in self._seen_fingerprints

        return False

    def __len__(self) -> int:
        return len(self._history)

    def __iter__(self):
        return iter(self._history)

    def __repr__(self) -> str:
        return (
            "ValidationEvaluator("
            f"evaluated={len(self._history)}, "
            f"accepted={len(self._accepted_fingerprints)}, "
            f"rejected={len(self._rejection_counts)}"
            ")"
        )