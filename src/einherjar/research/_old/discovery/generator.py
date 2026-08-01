"""
==========================================================
Discovery Generator
==========================================================

Fabrique les hypothèses candidates du Discovery Engine.

Le générateur ne valide rien et ne simule rien.
Il construit uniquement des hypothèses, puis leurs
transformations élémentaires :

- seed
- expand
- mutate
- replace
- prune

Le moteur de recherche central utilise cette classe pour
faire évoluer une population d'hypothèses.
"""

from __future__ import annotations

from collections import Counter
from collections import defaultdict
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
import math
import re
from typing import Any
from typing import Iterable
from typing import Mapping
from typing import Sequence

import numpy as np

from models.condition import Condition
from models.enums import ConditionOperator as ComparisonOperator
from models.enums import EconomicFamily
from models.enums import FeatureType
from models.enums import FeatureValueType
from models.feature import Feature
from models.feature_registry import FeatureRegistry
from models.fingerprint import fingerprint_model
from models.hypothesis import Hypothesis


__all__ = [
    "GeneratorSettings",
    "GenerationResult",
    "DiscoveryGenerator",
]


RAW_FEATURE_NAMES = {"open", "high", "low", "close", "volume"}

DEFAULT_ACTION_WEIGHTS = {
    "expand": 0.45,
    "mutate": 0.30,
    "replace": 0.20,
    "prune": 0.05,
}

DEFAULT_MUTATION_WEIGHTS = {
    "threshold": 0.50,
    "operator": 0.25,
    "feature": 0.25,
}

DEFAULT_THRESHOLD_PERCENTILES = (0.25, 0.50, 0.75)


def _resolve_path(obj: Any, path: Sequence[str]) -> Any | None:
    current = obj
    for part in path:
        if current is None:
            return None
        if isinstance(current, Mapping):
            if part not in current:
                return None
            current = current[part]
        else:
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


def _coerce_float(value: Any, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_int(value: Any, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_bool(value: Any, default: bool) -> bool:
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


def _normalize_tuple(values: Any) -> tuple[Any, ...]:
    if values is None:
        return ()
    if isinstance(values, tuple):
        return values
    if isinstance(values, list):
        return tuple(values)
    if isinstance(values, set):
        return tuple(values)
    return (values,)


def _normalize_weight_mapping(mapping: Any) -> dict[str, float]:
    data = _to_mapping(mapping)
    normalized: dict[str, float] = {}

    for key, value in data.items():
        if isinstance(key, EconomicFamily):
            normalized[key.value.lower()] = max(0.0, _coerce_float(value, 1.0))
            continue

        normalized[str(key).strip().lower()] = max(0.0, _coerce_float(value, 1.0))

    return normalized


def _normalize_feature_weight_mapping(mapping: Any) -> dict[str, float]:
    data = _to_mapping(mapping)
    normalized: dict[str, float] = {}

    for key, value in data.items():
        if isinstance(key, Feature):
            normalized[str(key.column_index)] = max(0.0, _coerce_float(value, 1.0))
            continue

        normalized[str(key).strip().lower()] = max(0.0, _coerce_float(value, 1.0))

    return normalized


def _serialize_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, Feature):
        return value.column_index

    if isinstance(value, Enum):
        return value.value

    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _serialize_value(value.to_dict())

    if isinstance(value, Mapping):
        return {
            str(key): _serialize_value(item)
            for key, item in value.items()
        }

    if isinstance(value, tuple):
        return [_serialize_value(item) for item in value]

    if isinstance(value, list):
        return [_serialize_value(item) for item in value]

    if isinstance(value, set):
        items = [_serialize_value(item) for item in value]
        return sorted(items, key=lambda item: repr(item))

    return value


def _fingerprint_key(value: Any) -> str:
    serialized = _serialize_value(value)
    return repr(serialized)


def _dedupe_preserve_order(values: Iterable[Any]) -> tuple[Any, ...]:
    seen: set[str] = set()
    output: list[Any] = []

    for value in values:
        key = _fingerprint_key(value)
        if key in seen:
            continue
        seen.add(key)
        output.append(value)

    return tuple(output)


def _coerce_family(value: EconomicFamily | str | None) -> EconomicFamily | None:
    if value is None:
        return None

    if isinstance(value, EconomicFamily):
        return value

    text = str(value).strip()

    try:
        return EconomicFamily(text)
    except ValueError:
        normalized = text.upper()
        try:
            return EconomicFamily[normalized]
        except KeyError as exc:
            raise ValueError(f"Unknown family: {value}") from exc


def _coerce_operator(value: ComparisonOperator | str | None) -> ComparisonOperator | None:
    if value is None:
        return None

    if isinstance(value, ComparisonOperator):
        return value

    text = str(value).strip()

    try:
        return ComparisonOperator(text)
    except ValueError:
        normalized = text.upper()
        try:
            return ComparisonOperator[normalized]
        except KeyError as exc:
            raise ValueError(f"Unknown operator: {value}") from exc


def _coerce_operand_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, (bool, int, float)):
        return value

    return value


def _is_numeric_value_type(value_type: FeatureValueType) -> bool:
    return value_type in {
        FeatureValueType.FLOAT,
        FeatureValueType.INTEGER,
        FeatureValueType.ORDINAL,
    }


def _is_binary_feature(feature: Feature) -> bool:
    name = feature.name.lower()

    if feature.feature_type == FeatureType.PATTERN:
        return True

    if feature.value_type == FeatureValueType.BOOLEAN:
        return True

    if name.endswith("_signal"):
        return True

    metadata = feature.metadata or {}
    if _coerce_bool(metadata.get("binary"), False):
        return True

    if str(metadata.get("profile", "")).strip().lower() == "binary":
        return True

    return False


def _feature_profile(feature: Feature) -> str:
    name = feature.name.lower()
    metadata = feature.metadata or {}

    explicit = str(metadata.get("profile", metadata.get("kind", metadata.get("mode", "")))).strip().lower()
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

    if "score" in name and not name.startswith("raw_"):
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

    if feature.feature_type == FeatureType.QUANTITATIVE or name.startswith("quant_"):
        return "statistical"

    if _is_numeric_value_type(feature.value_type):
        return "generic_numeric"

    return "unsupported"


def _default_feature_right_probability(profile: str) -> float:
    if profile == "binary":
        return 0.0
    if profile == "raw_scale":
        return 0.85
    if profile == "oscillator":
        return 0.15
    if profile in {"normalized", "ratio", "distance"}:
        return 0.05
    if profile == "statistical":
        return 0.20
    return 0.25


def _canonical_thresholds_for_feature_name(name: str) -> tuple[Any, ...]:
    lower = name.lower()

    if "rsi" in lower:
        return (20, 30, 50, 70, 80)

    if "stoch" in lower:
        return (20, 50, 80)

    if "williams" in lower:
        return (-80, -50, -20)

    if "cci" in lower:
        return (-200, -100, 0, 100, 200)

    if "adx" in lower:
        return (15, 20, 25, 30, 40)

    if "mfi" in lower:
        return (20, 50, 80)

    if "roc" in lower:
        return (-10, -5, 0, 5, 10)

    if "momentum" in lower:
        return (-2, -1, 0, 1, 2)

    if "macd" in lower:
        return (-2, -1, 0, 1, 2)

    if "ratio" in lower:
        return (0.8, 1.0, 1.2, 1.5, 2.0, 3.0)

    if "percentile" in lower:
        return (20, 50, 80)

    if "percent" in lower:
        return (20, 50, 80)

    if "norm" in lower:
        return (0.25, 0.50, 0.75)

    if "score" in lower:
        return (0.25, 0.50, 0.75)

    if "distance" in lower:
        return (-1.0, -0.5, 0.0, 0.5, 1.0)

    if "spread" in lower:
        return (-1.0, 0.0, 1.0)

    if "diff" in lower or "delta" in lower or "gap" in lower:
        return (-1.0, 0.0, 1.0)

    if "entropy" in lower or "hurst" in lower or "autocorr" in lower or "skew" in lower or "kurtosis" in lower:
        return (-1.0, 0.0, 1.0)

    return ()


@dataclass(frozen=True, slots=True, eq=False)
class GeneratorSettings:
    """
    Paramètres de génération du Discovery Engine.
    """

    min_conditions: int = 1
    max_conditions: int = 0

    seed_population_size: int = 0
    max_seed_attempts: int = 256
    max_generation_attempts: int = 512

    allow_raw_features: bool = False
    prefer_non_other_families: bool = True
    family_balance: bool = True

    feature_right_probability: float = 0.25
    allow_feature_reuse: bool = False

    random_seed: int | None = None

    family_weights: dict[str, float] = field(default_factory=dict)
    feature_weights: dict[str, float] = field(default_factory=dict)

    action_weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_ACTION_WEIGHTS))
    mutation_weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_MUTATION_WEIGHTS))

    threshold_percentiles: tuple[float, ...] = DEFAULT_THRESHOLD_PERCENTILES

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "min_conditions",
            max(1, _coerce_int(self.min_conditions, 1)),
        )

        object.__setattr__(
            self,
            "max_conditions",
            max(0, _coerce_int(self.max_conditions, 0)),
        )

        object.__setattr__(
            self,
            "seed_population_size",
            max(0, _coerce_int(self.seed_population_size, 0)),
        )

        object.__setattr__(
            self,
            "max_seed_attempts",
            max(1, _coerce_int(self.max_seed_attempts, 256)),
        )

        object.__setattr__(
            self,
            "max_generation_attempts",
            max(1, _coerce_int(self.max_generation_attempts, 512)),
        )

        object.__setattr__(
            self,
            "allow_raw_features",
            _coerce_bool(self.allow_raw_features, False),
        )

        object.__setattr__(
            self,
            "prefer_non_other_families",
            _coerce_bool(self.prefer_non_other_families, True),
        )

        object.__setattr__(
            self,
            "family_balance",
            _coerce_bool(self.family_balance, True),
        )

        object.__setattr__(
            self,
            "feature_right_probability",
            min(1.0, max(0.0, _coerce_float(self.feature_right_probability, 0.25))),
        )

        object.__setattr__(
            self,
            "allow_feature_reuse",
            _coerce_bool(self.allow_feature_reuse, False),
        )

        object.__setattr__(
            self,
            "random_seed",
            None if self.random_seed is None else _coerce_int(self.random_seed, 0),
        )

        object.__setattr__(
            self,
            "family_weights",
            _normalize_weight_mapping(self.family_weights),
        )

        object.__setattr__(
            self,
            "feature_weights",
            _normalize_feature_weight_mapping(self.feature_weights),
        )

        action_weights = _normalize_weight_mapping(self.action_weights)
        if not action_weights:
            action_weights = dict(DEFAULT_ACTION_WEIGHTS)
        object.__setattr__(self, "action_weights", action_weights)

        mutation_weights = _normalize_weight_mapping(self.mutation_weights)
        if not mutation_weights:
            mutation_weights = dict(DEFAULT_MUTATION_WEIGHTS)
        object.__setattr__(self, "mutation_weights", mutation_weights)

        percentiles = tuple(
            float(value)
            for value in _normalize_tuple(self.threshold_percentiles)
        )
        if not percentiles:
            percentiles = DEFAULT_THRESHOLD_PERCENTILES
        object.__setattr__(self, "threshold_percentiles", percentiles)

    @classmethod
    def from_config(cls, config: Any | None) -> "GeneratorSettings":
        if config is None:
            return cls()

        return cls(
            min_conditions=_coerce_int(
                _first_non_none(
                    config,
                    ("discovery", "min_conditions"),
                    ("search", "min_conditions"),
                    ("generator", "min_conditions"),
                    ("min_conditions",),
                    default=1,
                ),
                1,
            ),
            max_conditions=_coerce_int(
                _first_non_none(
                    config,
                    ("discovery", "max_conditions"),
                    ("search", "max_conditions"),
                    ("search", "max_depth"),
                    ("generator", "max_conditions"),
                    ("max_conditions",),
                    default=0,
                ),
                0,
            ),
            seed_population_size=_coerce_int(
                _first_non_none(
                    config,
                    ("discovery", "seed_population_size"),
                    ("search", "seed_population_size"),
                    ("discovery", "initial_population_size"),
                    ("search", "initial_population_size"),
                    ("seed_population_size",),
                    default=0,
                ),
                0,
            ),
            max_seed_attempts=_coerce_int(
                _first_non_none(
                    config,
                    ("discovery", "max_seed_attempts"),
                    ("search", "max_seed_attempts"),
                    ("generator", "max_seed_attempts"),
                    ("max_seed_attempts",),
                    default=256,
                ),
                256,
            ),
            max_generation_attempts=_coerce_int(
                _first_non_none(
                    config,
                    ("discovery", "max_generation_attempts"),
                    ("search", "max_generation_attempts"),
                    ("generator", "max_generation_attempts"),
                    ("max_generation_attempts",),
                    default=512,
                ),
                512,
            ),
            allow_raw_features=_coerce_bool(
                _first_non_none(
                    config,
                    ("discovery", "allow_raw_features"),
                    ("search", "allow_raw_features"),
                    ("dataset", "allow_raw_features"),
                    ("allow_raw_features",),
                    default=False,
                ),
                False,
            ),
            prefer_non_other_families=_coerce_bool(
                _first_non_none(
                    config,
                    ("discovery", "prefer_non_other_families"),
                    ("families", "prefer_non_other_families"),
                    ("prefer_non_other_families",),
                    default=True,
                ),
                True,
            ),
            family_balance=_coerce_bool(
                _first_non_none(
                    config,
                    ("discovery", "family_balance"),
                    ("search", "family_balance"),
                    ("family_balance",),
                    default=True,
                ),
                True,
            ),
            feature_right_probability=_coerce_float(
                _first_non_none(
                    config,
                    ("discovery", "feature_right_probability"),
                    ("search", "feature_right_probability"),
                    ("generator", "feature_right_probability"),
                    ("feature_right_probability",),
                    default=0.25,
                ),
                0.25,
            ),
            allow_feature_reuse=_coerce_bool(
                _first_non_none(
                    config,
                    ("discovery", "allow_feature_reuse"),
                    ("search", "allow_feature_reuse"),
                    ("generator", "allow_feature_reuse"),
                    ("allow_feature_reuse",),
                    default=False,
                ),
                False,
            ),
            random_seed=_first_non_none(
                config,
                ("discovery", "random_seed"),
                ("search", "random_seed"),
                ("generator", "random_seed"),
                ("random_seed",),
                ("seed",),
                default=None,
            ),
            family_weights=_to_mapping(
                _first_non_none(
                    config,
                    ("discovery", "family_weights"),
                    ("search", "family_weights"),
                    ("families", "weights"),
                    ("family_weights",),
                    default={},
                )
            ),
            feature_weights=_to_mapping(
                _first_non_none(
                    config,
                    ("discovery", "feature_weights"),
                    ("search", "feature_weights"),
                    ("generator", "feature_weights"),
                    ("feature_weights",),
                    default={},
                )
            ),
            action_weights=_to_mapping(
                _first_non_none(
                    config,
                    ("discovery", "action_weights"),
                    ("search", "action_weights"),
                    ("generator", "action_weights"),
                    ("action_weights",),
                    default=dict(DEFAULT_ACTION_WEIGHTS),
                )
            ),
            mutation_weights=_to_mapping(
                _first_non_none(
                    config,
                    ("discovery", "mutation_weights"),
                    ("search", "mutation_weights"),
                    ("generator", "mutation_weights"),
                    ("mutation_weights",),
                    default=dict(DEFAULT_MUTATION_WEIGHTS),
                )
            ),
            threshold_percentiles=_normalize_tuple(
                _first_non_none(
                    config,
                    ("discovery", "threshold_percentiles"),
                    ("search", "threshold_percentiles"),
                    ("generator", "threshold_percentiles"),
                    ("threshold_percentiles",),
                    default=DEFAULT_THRESHOLD_PERCENTILES,
                )
            ),
        )


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """
    Résultat d'une génération ou d'une transformation.
    """

    action: str
    hypothesis: Hypothesis

    parent_fingerprint: str | None = None

    family: str | None = None

    feature_index: int | None = None
    feature_name: str | None = None

    operator: str | None = None
    operand_kind: str | None = None
    operand_value: Any = None

    condition_index: int | None = None

    threshold_source: str | None = None

    attempts: int = 1

    details: dict[str, Any] = field(default_factory=dict, compare=False, hash=False)

    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.hypothesis, Hypothesis):
            raise TypeError("hypothesis must be a Hypothesis.")

        object.__setattr__(self, "attempts", max(1, _coerce_int(self.attempts, 1)))
        object.__setattr__(self, "details", dict(self.details))
        object.__setattr__(self, "fingerprint", fingerprint_model(self.hypothesis))

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "fingerprint": self.fingerprint,
            "parent_fingerprint": self.parent_fingerprint,
            "family": self.family,
            "feature_index": self.feature_index,
            "feature_name": self.feature_name,
            "operator": self.operator,
            "operand_kind": self.operand_kind,
            "operand_value": _serialize_value(self.operand_value),
            "condition_index": self.condition_index,
            "threshold_source": self.threshold_source,
            "attempts": self.attempts,
            "hypothesis": self.hypothesis.to_dict(),
            "details": _serialize_value(self.details),
        }

    def __repr__(self) -> str:
        short = self.fingerprint[:12]
        return f"GenerationResult(action='{self.action}', fingerprint='{short}', attempts={self.attempts})"


class DiscoveryGenerator:
    """
    Fabrique d'hypothèses du Discovery Engine.

    Le générateur maintient une mémoire de :
    - fingerprints déjà vus
    - historique des transformations
    - relations parent/enfants
    - usage des features, familles, opérateurs et seuils
    """

    def __init__(
        self,
        config: Any | None,
        registry: FeatureRegistry,
        feature_statistics: Mapping[Any, Mapping[str, Any]] | None = None,
        rng: np.random.Generator | int | None = None,
    ) -> None:
        if not isinstance(registry, FeatureRegistry):
            raise TypeError("registry must be a FeatureRegistry.")

        self._config = config
        self._settings = GeneratorSettings.from_config(config)

        self._registry = registry
        self._threshold_policy = _first_non_none(
            config,
            ("discovery", "threshold_policy"),
            ("search", "threshold_policy"),
            ("generator", "threshold_policy"),
            ("threshold_policy",),
            default=None,
        )

        self._rng = self._build_rng(rng, self._settings.random_seed)
        self._feature_statistics: dict[Any, Mapping[str, Any]] = dict(feature_statistics or {})

        self._all_features: tuple[Feature, ...] = tuple(
            feature for feature in registry.features if feature.enabled
        )
        self._supported_features: tuple[Feature, ...] = tuple(
            feature for feature in self._all_features if _feature_profile(feature) != "unsupported"
        )
        self._seedable_features: tuple[Feature, ...] = tuple(
            feature
            for feature in self._supported_features
            if self._is_seedable_feature(feature)
        )

        self._family_features: dict[EconomicFamily, tuple[Feature, ...]] = self._build_family_features()
        self._seed_families: tuple[EconomicFamily, ...] = self._build_seed_families()

        self._seen_fingerprints: set[str] = set()
        self._history: list[GenerationResult] = []
        self._records: dict[str, GenerationResult] = {}

        self._parent_by_fingerprint: dict[str, str | None] = {}
        self._children_by_fingerprint: defaultdict[str, list[str]] = defaultdict(list)

        self._feature_usage: Counter[int] = Counter()
        self._family_usage: Counter[str] = Counter()
        self._operator_usage: Counter[str] = Counter()
        self._threshold_usage: Counter[str] = Counter()

    # ==================================================
    # RNG
    # ==================================================

    @staticmethod
    def _build_rng(
        rng: np.random.Generator | int | None,
        seed: int | None,
    ) -> np.random.Generator:
        if isinstance(rng, np.random.Generator):
            return rng

        if rng is not None:
            return np.random.default_rng(rng)

        if seed is not None:
            return np.random.default_rng(seed)

        return np.random.default_rng()

    # ==================================================
    # PUBLIC PROPERTIES
    # ==================================================

    @property
    def settings(self) -> GeneratorSettings:
        return self._settings

    @property
    def registry(self) -> FeatureRegistry:
        return self._registry

    @property
    def history(self) -> tuple[GenerationResult, ...]:
        return tuple(self._history)

    @property
    def records(self) -> dict[str, GenerationResult]:
        return dict(self._records)

    @property
    def families(self) -> tuple[EconomicFamily, ...]:
        return tuple(self._family_features.keys())

    @property
    def seed_families(self) -> tuple[EconomicFamily, ...]:
        return self._seed_families

    @property
    def seedable_features(self) -> tuple[Feature, ...]:
        return self._seedable_features

    @property
    def supported_features(self) -> tuple[Feature, ...]:
        return self._supported_features

    @property
    def feature_usage(self) -> dict[int, int]:
        return dict(self._feature_usage)

    @property
    def family_usage(self) -> dict[str, int]:
        return dict(self._family_usage)

    @property
    def operator_usage(self) -> dict[str, int]:
        return dict(self._operator_usage)

    @property
    def threshold_usage(self) -> dict[str, int]:
        return dict(self._threshold_usage)

    # ==================================================
    # PUBLIC MEMORY API
    # ==================================================

    def register(self, hypothesis: Hypothesis | GenerationResult) -> str:
        """
        Marque une hypothèse comme déjà connue.

        N'ajoute pas l'objet à l'historique.
        """
        hyp = self._extract_hypothesis(hypothesis)
        fingerprint = fingerprint_model(hyp)
        self._seen_fingerprints.add(fingerprint)
        self._parent_by_fingerprint.setdefault(fingerprint, None)
        return fingerprint

    def learn(self, hypothesis: Hypothesis | GenerationResult) -> str:
        """
        Marque une hypothèse comme connue et met à jour les
        statistiques d'usage du générateur.
        """
        hyp = self._extract_hypothesis(hypothesis)
        fingerprint = self.register(hyp)
        self._update_usage(hyp)
        return fingerprint

    def is_known(self, hypothesis: Hypothesis | GenerationResult | str) -> bool:
        if isinstance(hypothesis, str):
            return hypothesis in self._seen_fingerprints
        hyp = self._extract_hypothesis(hypothesis)
        return fingerprint_model(hyp) in self._seen_fingerprints

    def parent_of(self, fingerprint: str) -> str | None:
        return self._parent_by_fingerprint.get(fingerprint)

    def children_of(self, fingerprint: str) -> tuple[str, ...]:
        return tuple(self._children_by_fingerprint.get(fingerprint, ()))

    def attach_feature_statistics(
        self,
        feature_statistics: Mapping[Any, Mapping[str, Any]] | None,
    ) -> None:
        """
        Remplace les statistiques de features utilisées pour
        la sélection des seuils.
        """
        self._feature_statistics = dict(feature_statistics or {})

    # ==================================================
    # PUBLIC STRUCTURAL HELPERS
    # ==================================================

    def feature_profile(self, feature: Feature) -> str:
        return _feature_profile(feature)

    def family_candidates(
        self,
        family: EconomicFamily | str,
        *,
        seedable: bool = False,
    ) -> tuple[Feature, ...]:
        family_enum = _coerce_family(family)
        if family_enum is None:
            raise ValueError("family cannot be None.")

        pool = self._seedable_features if seedable else self._family_features.get(family_enum, ())
        return tuple(pool)

    def threshold_candidates(self, feature: Feature) -> tuple[Any, ...]:
        candidates, _source = self._threshold_candidates(feature)
        return candidates

    def build_hypothesis(self, conditions: Iterable[Condition]) -> Hypothesis:
        return Hypothesis(conditions)

    def build_condition(
        self,
        feature: Feature | None = None,
        *,
        family: EconomicFamily | str | None = None,
        prefer_feature_right: bool | None = None,
        operator: ComparisonOperator | str | None = None,
        right: Feature | int | float | bool | None = None,
        avoid_features: Iterable[int | Feature] | None = None,
    ) -> Condition:
        condition, _details = self._build_condition_spec(
            feature=feature,
            family=family,
            prefer_feature_right=prefer_feature_right,
            operator=operator,
            right=right,
            avoid_features=avoid_features,
        )
        return condition

    def available_actions(self, hypothesis: Hypothesis, **kwargs: Any) -> tuple[str, ...]:
        actions: list[str] = ["mutate", "replace"]

        if self._settings.max_conditions <= 0 or len(hypothesis) < self._settings.max_conditions:
            actions.insert(0, "expand")

        if len(hypothesis) > self._settings.min_conditions:
            actions.append("prune")

        return tuple(actions)

    def choose_action(self, hypothesis: Hypothesis) -> str:
        actions = self.available_actions(hypothesis)
        weights = [self._settings.action_weights.get(action, 1.0) for action in actions]
        return str(self._weighted_choice(actions, weights))

    # ==================================================
    # SEED
    # ==================================================

    def seed(
        self,
        family: EconomicFamily | str | None = None,
    ) -> GenerationResult:
        family_enum = _coerce_family(family)

        def builder(attempt: int) -> tuple[Hypothesis, dict[str, Any]]:
            condition, details = self._build_condition_spec(
                family=family_enum,
                avoid_features=None,
            )
            hypothesis = Hypothesis((condition,))

            return hypothesis, {
                "parent_fingerprint": None,
                "family": details["family"],
                "feature_index": details["feature_index"],
                "feature_name": details["feature_name"],
                "operator": details["operator"],
                "operand_kind": details["operand_kind"],
                "operand_value": details["operand_value"],
                "condition_index": 0,
                "threshold_source": details["threshold_source"],
                "details": {
                    "mode": "seed",
                    "attempt": attempt,
                    "feature_profile": details["feature_profile"],
                    "selection_score": details["selection_score"],
                },
            }

        if family_enum is not None:
            pool = self.family_candidates(family_enum, seedable=True)
            if not pool:
                raise ValueError(f"No seedable features available for family {family_enum.value}.")

        result = self._generate_unique("seed", None, builder, max_attempts=self._settings.max_seed_attempts)
        return result

    def seed_population(
        self,
        size: int | None = None,
        families: Iterable[EconomicFamily | str] | None = None,
    ) -> tuple[GenerationResult, ...]:
        family_list = (
            tuple(_coerce_family(item) for item in families)
            if families is not None
            else self.seed_families
        )

        family_list = tuple(item for item in family_list if item is not None)

        if not family_list:
            family_list = self._seed_families or tuple(self.families)

        if size is None:
            size = len(family_list)

        size = max(0, _coerce_int(size, 0))
        if size == 0:
            return ()

        results: list[GenerationResult] = []
        index = 0

        while len(results) < size:
            family = family_list[index % len(family_list)]
            results.append(self.seed(family))
            index += 1

        return tuple(results)

    def bootstrap(
        self,
        size: int | None = None,
    ) -> tuple[GenerationResult, ...]:
        """
        Alias pratique pour démarrer la population initiale.
        """
        return self.seed_population(size=size)

    # ==================================================
    # TRANSFORMATIONS
    # ==================================================

    def expand(
        self,
        hypothesis: Hypothesis,
        *,
        family: EconomicFamily | str | None = None,
    ) -> GenerationResult:
        parent_fp = fingerprint_model(hypothesis)
        family_hint = _coerce_family(family)

        if self._settings.max_conditions > 0 and len(hypothesis) >= self._settings.max_conditions:
            raise ValueError("Maximum number of conditions reached for expansion.")

        current_features = {condition.left.column_index for condition in hypothesis.conditions}
        current_families = {condition.left.economic_family for condition in hypothesis.conditions}

        def builder(attempt: int) -> tuple[Hypothesis, dict[str, Any]]:
            target_family = self._pick_expansion_family(
                current_families=current_families,
                family_hint=family_hint,
            )

            condition, details = self._build_condition_spec(
                family=target_family,
                avoid_features=current_features,
            )

            if self._condition_signature(condition) in {
                self._condition_signature(item) for item in hypothesis.conditions
            }:
                raise ValueError("Expansion produced a duplicate condition.")

            conditions = tuple(list(hypothesis.conditions) + [condition])
            candidate = Hypothesis(conditions)

            return candidate, {
                "parent_fingerprint": parent_fp,
                "family": details["family"],
                "feature_index": details["feature_index"],
                "feature_name": details["feature_name"],
                "operator": details["operator"],
                "operand_kind": details["operand_kind"],
                "operand_value": details["operand_value"],
                "condition_index": len(conditions) - 1,
                "threshold_source": details["threshold_source"],
                "details": {
                    "mode": "expand",
                    "attempt": attempt,
                    "selection_score": details["selection_score"],
                    "target_family": details["family"],
                    "before_size": len(hypothesis.conditions),
                    "after_size": len(conditions),
                },
            }

        return self._generate_unique("expand", parent_fp, builder, max_attempts=self._settings.max_generation_attempts)

    def mutate(
        self,
        hypothesis: Hypothesis,
        *,
        family: EconomicFamily | str | None = None,
        condition_index: int | None = None,
    ) -> GenerationResult:
        parent_fp = fingerprint_model(hypothesis)
        family_hint = _coerce_family(family)

        if not hypothesis.conditions:
            raise ValueError("Cannot mutate an empty hypothesis.")

        def builder(attempt: int) -> tuple[Hypothesis, dict[str, Any]]:
            idx = (
                _coerce_int(condition_index, -1)
                if condition_index is not None
                else self._pick_condition_index(
                    hypothesis,
                    family_hint=family_hint,
                    strategy="weighted",
                )
            )

            old_condition = hypothesis.conditions[idx]
            new_condition, details = self._mutate_condition(
                old_condition,
                target_family=family_hint,
                current_hypothesis=hypothesis,
            )

            conditions = list(hypothesis.conditions)
            conditions[idx] = new_condition
            candidate = Hypothesis(tuple(conditions))

            return candidate, {
                "parent_fingerprint": parent_fp,
                "family": details["family"],
                "feature_index": details["feature_index"],
                "feature_name": details["feature_name"],
                "operator": details["operator"],
                "operand_kind": details["operand_kind"],
                "operand_value": details["operand_value"],
                "condition_index": idx,
                "threshold_source": details["threshold_source"],
                "details": {
                    "mode": "mutate",
                    "attempt": attempt,
                    "mutation_subaction": details["mutation_subaction"],
                    "old_condition_fingerprint": fingerprint_model(old_condition),
                    "new_condition_fingerprint": fingerprint_model(new_condition),
                    "before_size": len(hypothesis.conditions),
                    "after_size": len(conditions),
                },
            }

        return self._generate_unique("mutate", parent_fp, builder, max_attempts=self._settings.max_generation_attempts)

    def replace(
        self,
        hypothesis: Hypothesis,
        *,
        family: EconomicFamily | str | None = None,
        condition_index: int | None = None,
    ) -> GenerationResult:
        parent_fp = fingerprint_model(hypothesis)
        family_hint = _coerce_family(family)

        if not hypothesis.conditions:
            raise ValueError("Cannot replace inside an empty hypothesis.")

        def builder(attempt: int) -> tuple[Hypothesis, dict[str, Any]]:
            idx = (
                _coerce_int(condition_index, -1)
                if condition_index is not None
                else self._pick_condition_index(
                    hypothesis,
                    family_hint=family_hint,
                    strategy="weakest",
                )
            )

            old_condition = hypothesis.conditions[idx]
            replacement_family = family_hint or old_condition.left.economic_family

            new_condition, details = self._build_condition_spec(
                family=replacement_family,
                avoid_features={cond.left.column_index for cond in hypothesis.conditions if cond is not old_condition},
            )

            conditions = list(hypothesis.conditions)
            conditions[idx] = new_condition
            candidate = Hypothesis(tuple(conditions))

            return candidate, {
                "parent_fingerprint": parent_fp,
                "family": details["family"],
                "feature_index": details["feature_index"],
                "feature_name": details["feature_name"],
                "operator": details["operator"],
                "operand_kind": details["operand_kind"],
                "operand_value": details["operand_value"],
                "condition_index": idx,
                "threshold_source": details["threshold_source"],
                "details": {
                    "mode": "replace",
                    "attempt": attempt,
                    "old_condition_fingerprint": fingerprint_model(old_condition),
                    "new_condition_fingerprint": fingerprint_model(new_condition),
                    "before_size": len(hypothesis.conditions),
                    "after_size": len(conditions),
                },
            }

        return self._generate_unique("replace", parent_fp, builder, max_attempts=self._settings.max_generation_attempts)

    def prune(
        self,
        hypothesis: Hypothesis,
        *,
        condition_index: int | None = None,
    ) -> GenerationResult:
        parent_fp = fingerprint_model(hypothesis)

        if len(hypothesis.conditions) <= self._settings.min_conditions:
            raise ValueError("Hypothesis cannot be pruned below the minimum number of conditions.")

        def builder(attempt: int) -> tuple[Hypothesis, dict[str, Any]]:
            idx = (
                _coerce_int(condition_index, -1)
                if condition_index is not None
                else self._pick_prune_index(hypothesis)
            )

            removed = hypothesis.conditions[idx]
            conditions = list(hypothesis.conditions)
            del conditions[idx]

            if not conditions:
                raise ValueError("Pruning would produce an empty hypothesis.")

            candidate = Hypothesis(tuple(conditions))
            importance = self._condition_importance(removed, hypothesis)

            return candidate, {
                "parent_fingerprint": parent_fp,
                "family": removed.left.economic_family.value,
                "feature_index": removed.left.column_index,
                "feature_name": removed.left.name,
                "operator": removed.operator.value,
                "operand_kind": "feature" if isinstance(removed.right, Feature) else "constant",
                "operand_value": _serialize_value(removed.right),
                "condition_index": idx,
                "threshold_source": "prune",
                "details": {
                    "mode": "prune",
                    "attempt": attempt,
                    "removed_condition_fingerprint": fingerprint_model(removed),
                    "importance": importance,
                    "before_size": len(hypothesis.conditions),
                    "after_size": len(conditions),
                },
            }

        return self._generate_unique("prune", parent_fp, builder, max_attempts=self._settings.max_generation_attempts)

    def evolve(
        self,
        hypothesis: Hypothesis,
        *,
        action: str | None = None,
        family: EconomicFamily | str | None = None,
        condition_index: int | None = None,
    ) -> GenerationResult:
        chosen = "auto" if action is None else str(action).strip().lower()
        family_hint = _coerce_family(family)

        if chosen == "auto":
            chosen = self.choose_action(hypothesis)

        if chosen == "seed":
            raise ValueError("Seed is not an evolution action; call seed() instead.")

        if chosen == "expand":
            return self.expand(hypothesis, family=family_hint)

        if chosen == "mutate":
            return self.mutate(hypothesis, family=family_hint, condition_index=condition_index)

        if chosen == "replace":
            return self.replace(hypothesis, family=family_hint, condition_index=condition_index)

        if chosen == "prune":
            return self.prune(hypothesis, condition_index=condition_index)

        raise ValueError(f"Unknown evolution action: {action}")

    # ==================================================
    # PYTHON PROTOCOL
    # ==================================================

    def __contains__(self, item: Hypothesis | GenerationResult | str) -> bool:
        if isinstance(item, str):
            return item in self._seen_fingerprints
        if isinstance(item, GenerationResult):
            return item.fingerprint in self._seen_fingerprints
        if isinstance(item, Hypothesis):
            return fingerprint_model(item) in self._seen_fingerprints
        return False

    def __len__(self) -> int:
        return len(self._history)

    def __iter__(self):
        return iter(self._history)

    def __repr__(self) -> str:
        return (
            "DiscoveryGenerator("
            f"known={len(self._seen_fingerprints)}, "
            f"history={len(self._history)}, "
            f"families={len(self._family_features)}"
            ")"
        )

    # ==================================================
    # INTERNAL FAMILY HELPERS
    # ==================================================

    def _build_family_features(self) -> dict[EconomicFamily, tuple[Feature, ...]]:
        groups: dict[EconomicFamily, list[Feature]] = defaultdict(list)
        for feature in self._supported_features:
            groups[feature.economic_family].append(feature)

        ordered: dict[EconomicFamily, tuple[Feature, ...]] = {}
        for family in sorted(groups.keys(), key=lambda item: item.value):
            ordered[family] = tuple(sorted(groups[family], key=lambda item: item.column_index))

        return ordered

    def _build_seed_families(self) -> tuple[EconomicFamily, ...]:
        families = tuple(
            family for family, features in self._family_features.items()
            if any(self._is_seedable_feature(feature) for feature in features)
        )

        if not families:
            families = tuple(self._family_features.keys())

        if self._settings.prefer_non_other_families:
            non_other = tuple(family for family in families if family != EconomicFamily.OTHER)
            if non_other:
                families = non_other

        return families

    def _is_seedable_feature(self, feature: Feature) -> bool:
        if not feature.enabled:
            return False

        metadata = feature.metadata or {}
        if _coerce_bool(metadata.get("seedable"), True) is False:
            return False

        if _coerce_bool(metadata.get("include_in_discovery"), True) is False:
            return False

        if not self._settings.allow_raw_features and feature.name.lower() in RAW_FEATURE_NAMES:
            return False

        profile = _feature_profile(feature)
        if profile == "unsupported":
            return False

        return True

    def _build_raw_feature_set(self) -> set[str]:
        return {name.lower() for name in RAW_FEATURE_NAMES}

    # ==================================================
    # INTERNAL SELECTION
    # ==================================================

    def _pick_family(
        self,
        *,
        current_families: set[EconomicFamily] | None = None,
        family_hint: EconomicFamily | None = None,
    ) -> EconomicFamily:
        if family_hint is not None:
            pool = self._family_features.get(family_hint, ())
            if pool:
                return family_hint

        families = tuple(self._family_features.keys())
        if not families:
            raise RuntimeError("No families available in the registry.")

        current_families = current_families or set()
        scores: list[float] = []

        for family in families:
            pool = self._family_features.get(family, ())
            score = self._family_score(family, pool)

            if family in current_families:
                score *= 0.35

            scores.append(score)

        choice = self._weighted_choice(list(families), scores)
        return choice

    def _pick_expansion_family(
        self,
        *,
        current_families: set[EconomicFamily],
        family_hint: EconomicFamily | None = None,
    ) -> EconomicFamily:
        if family_hint is not None and self._family_features.get(family_hint):
            return family_hint

        families = tuple(self._family_features.keys())
        if not families:
            raise RuntimeError("No families available in the registry.")

        preferred = [family for family in families if family not in current_families]
        if preferred:
            scores = [self._family_score(family, self._family_features.get(family, ())) for family in preferred]
            return self._weighted_choice(preferred, scores)

        scores = [self._family_score(family, self._family_features.get(family, ())) for family in families]
        return self._weighted_choice(list(families), scores)

    def _pick_feature(
        self,
        *,
        family: EconomicFamily | None = None,
        avoid_features: set[int] | None = None,
        seedable: bool = False,
    ) -> Feature:
        pool = self.seedable_features if seedable else self.supported_features

        if family is not None:
            family_pool = [feature for feature in pool if feature.economic_family == family]
            if family_pool:
                pool = tuple(family_pool)

        if not pool:
            raise RuntimeError("No feature candidates available.")

        avoid_features = avoid_features or set()
        scores = [self._feature_score(feature, avoid_features=avoid_features) for feature in pool]

        if max(scores, default=0.0) <= 0:
            scores = [1.0 for _ in pool]

        return self._weighted_choice(list(pool), scores)

    def _pick_right_feature(
        self,
        left_feature: Feature,
        *,
        family_hint: EconomicFamily | None = None,
        avoid_features: set[int] | None = None,
    ) -> Feature:
        avoid_features = set(avoid_features or set())
        avoid_features.add(left_feature.column_index)

        same_family_pool = [
            feature
            for feature in self.supported_features
            if feature.column_index not in avoid_features
            and feature.economic_family == left_feature.economic_family
        ]

        if family_hint is not None:
            hinted_pool = [
                feature
                for feature in self.supported_features
                if feature.column_index not in avoid_features
                and feature.economic_family == family_hint
            ]
            if hinted_pool:
                same_family_pool = hinted_pool

        if same_family_pool:
            scores = [self._feature_score(feature, avoid_features=avoid_features) for feature in same_family_pool]
            if max(scores, default=0.0) <= 0:
                scores = [1.0 for _ in same_family_pool]
            return self._weighted_choice(list(same_family_pool), scores)

        fallback_pool = [
            feature
            for feature in self.supported_features
            if feature.column_index not in avoid_features
        ]
        if not fallback_pool:
            raise RuntimeError("No right-feature candidate available.")

        scores = [self._feature_score(feature, avoid_features=avoid_features) for feature in fallback_pool]
        if max(scores, default=0.0) <= 0:
            scores = [1.0 for _ in fallback_pool]
        return self._weighted_choice(list(fallback_pool), scores)

    def _pick_condition_index(
        self,
        hypothesis: Hypothesis,
        *,
        family_hint: EconomicFamily | None = None,
        strategy: str = "weighted",
    ) -> int:
        if not hypothesis.conditions:
            raise ValueError("Hypothesis has no conditions.")

        candidates = list(range(len(hypothesis.conditions)))

        if family_hint is not None:
            family_candidates = [
                index
                for index, condition in enumerate(hypothesis.conditions)
                if condition.left.economic_family == family_hint
            ]
            if family_candidates:
                candidates = family_candidates

        if strategy == "random":
            return int(self._rng.choice(candidates))

        if strategy == "weakest":
            scores = [self._condition_importance(hypothesis.conditions[index], hypothesis) for index in candidates]
            weakest_index = int(np.argmin(np.asarray(scores, dtype=float)))
            return candidates[weakest_index]

        if strategy == "weighted":
            weights = []
            for index in candidates:
                importance = self._condition_importance(hypothesis.conditions[index], hypothesis)
                weights.append(1.0 / max(0.0001, importance))
            return self._weighted_choice(candidates, weights)

        raise ValueError(f"Unknown strategy: {strategy}")

    def _pick_prune_index(self, hypothesis: Hypothesis) -> int:
        return self._pick_condition_index(hypothesis, strategy="weakest")

    # ==================================================
    # INTERNAL SCORING
    # ==================================================

    def _family_score(self, family: EconomicFamily, pool: tuple[Feature, ...]) -> float:
        if not pool:
            return 0.0

        key = family.value.lower()
        base = self._settings.family_weights.get(key, 1.0)
        if family == EconomicFamily.OTHER and self._settings.prefer_non_other_families and len(self._family_features) > 1:
            base *= 0.35

        usage = 1.0 + float(self._family_usage.get(key, 0))
        size_penalty = math.sqrt(max(1, len(pool)))

        return base / usage / size_penalty

    def _feature_weight(self, feature: Feature) -> float:
        metadata = feature.metadata or {}
        weight = 1.0

        if "weight" in metadata:
            weight *= max(0.0, _coerce_float(metadata.get("weight"), 1.0))

        if "priority" in metadata:
            weight *= max(0.0, _coerce_float(metadata.get("priority"), 1.0))

        if "discovery_weight" in metadata:
            weight *= max(0.0, _coerce_float(metadata.get("discovery_weight"), 1.0))

        if "exploration_weight" in metadata:
            weight *= max(0.0, _coerce_float(metadata.get("exploration_weight"), 1.0))

        if "novelty_bonus" in metadata:
            weight *= max(0.0, _coerce_float(metadata.get("novelty_bonus"), 1.0))

        if "family_weight" in metadata:
            weight *= max(0.0, _coerce_float(metadata.get("family_weight"), 1.0))

        config_key_candidates = [
            str(feature.column_index),
            feature.name.lower(),
            feature.name,
        ]
        for key in config_key_candidates:
            if key.lower() in self._settings.feature_weights:
                weight *= self._settings.feature_weights[key.lower()]

        if not self._settings.allow_raw_features and feature.name.lower() in RAW_FEATURE_NAMES:
            weight *= 0.0

        return weight

    def _feature_score(
        self,
        feature: Feature,
        *,
        avoid_features: set[int] | None = None,
    ) -> float:
        if not feature.enabled:
            return 0.0

        if _feature_profile(feature) == "unsupported":
            return 0.0

        weight = self._feature_weight(feature)
        if weight <= 0:
            return 0.0

        family_key = feature.economic_family.value.lower()
        family_weight = self._settings.family_weights.get(family_key, 1.0)

        if feature.economic_family == EconomicFamily.OTHER and self._settings.prefer_non_other_families and len(self._family_features) > 1:
            family_weight *= 0.35

        if avoid_features and feature.column_index in avoid_features:
            weight *= 0.35

        usage = 1.0 + float(self._feature_usage.get(feature.column_index, 0))
        family_usage = 1.0 + float(self._family_usage.get(family_key, 0))
        size_penalty = math.sqrt(max(1, len(self._family_features.get(feature.economic_family, ()))))

        return (weight * family_weight) / usage / family_usage / size_penalty

    def _condition_importance(
        self,
        condition: Condition,
        hypothesis: Hypothesis,
    ) -> float:
        feature = condition.left
        family_key = feature.economic_family.value.lower()
        operator_key = condition.operator.value
        operand_key = self._operand_key(condition.right)

        feature_novelty = 1.0 / (1.0 + float(self._feature_usage.get(feature.column_index, 0)))
        family_novelty = 1.0 / (1.0 + float(self._family_usage.get(family_key, 0)))
        operator_novelty = 1.0 / (1.0 + float(self._operator_usage.get(operator_key, 0)))
        threshold_novelty = 1.0 / (1.0 + float(self._threshold_usage.get(self._threshold_usage_key(feature, condition.operator, operand_key), 0)))

        family_count = sum(1 for item in hypothesis.conditions if item.left.economic_family == feature.economic_family)
        local_family_novelty = 1.0 / max(1.0, float(family_count))

        if isinstance(condition.right, Feature):
            right_bonus = 0.20
        else:
            right_bonus = 0.0

        if feature.name.lower() in RAW_FEATURE_NAMES:
            raw_penalty = 0.20
        else:
            raw_penalty = 0.0

        operator_family_bonus = 0.0
        if condition.operator in {
            ComparisonOperator.CROSS_OVER,
            ComparisonOperator.CROSS_UNDER,
        }:
            operator_family_bonus += 0.15

        if condition.operator in {
            ComparisonOperator.EQ,
            ComparisonOperator.NE,
        } and _is_binary_feature(feature):
            operator_family_bonus -= 0.10

        importance = (
            0.30 * feature_novelty
            + 0.20 * family_novelty
            + 0.20 * operator_novelty
            + 0.15 * threshold_novelty
            + 0.10 * local_family_novelty
            + right_bonus
            + operator_family_bonus
            - raw_penalty
        )

        return max(0.0001, importance)

    # ==================================================
    # INTERNAL CONDITION BUILDING
    # ==================================================

    def _build_condition_spec(
        self,
        *,
        feature: Feature | None = None,
        family: EconomicFamily | str | None = None,
        prefer_feature_right: bool | None = None,
        operator: ComparisonOperator | str | None = None,
        right: Feature | int | float | bool | None = None,
        avoid_features: Iterable[int | Feature] | None = None,
    ) -> tuple[Condition, dict[str, Any]]:
        family_enum = _coerce_family(family)
        operator_enum = _coerce_operator(operator)

        avoid_set = self._normalize_feature_avoidance(avoid_features)

        selected_feature = feature
        selection_score = 0.0

        if selected_feature is None:
            selected_feature = self._pick_feature(
                family=family_enum,
                avoid_features=avoid_set,
                seedable=False,
            )
            selection_score = self._feature_score(selected_feature, avoid_features=avoid_set)
        else:
            if not isinstance(selected_feature, Feature):
                raise TypeError("feature must be a Feature when provided.")
            selection_score = self._feature_score(selected_feature, avoid_features=avoid_set)

        profile = _feature_profile(selected_feature)
        if profile == "unsupported":
            raise ValueError(f"Unsupported feature profile for {selected_feature.name}")

        explicit_right_kind: str | None = None
        right_operand: Feature | int | float | bool | None = None
        threshold_source = "heuristic"

        if right is not None:
            right_operand = self._coerce_right_operand(right)
            explicit_right_kind = "feature" if isinstance(right_operand, Feature) else "constant"
            threshold_source = "explicit"
        else:
            if prefer_feature_right is None:
                prefer_feature_right = self._should_use_feature_right(selected_feature)

            if profile == "binary":
                prefer_feature_right = False

            if prefer_feature_right:
                explicit_right_kind = "feature"
                right_operand = self._pick_right_feature(
                    selected_feature,
                    family_hint=family_enum,
                    avoid_features=avoid_set | {selected_feature.column_index},
                )
                threshold_source = "feature"
            else:
                threshold_candidates, threshold_source = self._threshold_candidates(selected_feature)
                if threshold_candidates:
                    right_operand = self._pick_threshold_value(
                        selected_feature,
                        threshold_candidates,
                    )
                    explicit_right_kind = "constant"
                else:
                    if profile in {"raw_scale", "unsupported"}:
                        right_operand = self._pick_right_feature(
                            selected_feature,
                            family_hint=family_enum,
                            avoid_features=avoid_set | {selected_feature.column_index},
                        )
                        explicit_right_kind = "feature"
                        threshold_source = "fallback_feature"
                    else:
                        fallback_candidates = self._fallback_constants_for_profile(profile)
                        right_operand = self._pick_threshold_value(
                            selected_feature,
                            fallback_candidates,
                        )
                        explicit_right_kind = "constant"
                        threshold_source = "fallback"

        if right_operand is None:
            raise RuntimeError("Unable to build a right operand.")

        if operator_enum is None:
            operator_enum = self._pick_operator(
                selected_feature,
                right_operand,
                right_kind=explicit_right_kind,
            )

        self._validate_operator_compatibility(operator_enum, explicit_right_kind)

        if isinstance(right_operand, Feature) and right_operand.column_index == selected_feature.column_index:
            raise ValueError("Self-comparisons are not allowed.")

        condition = Condition(
            left=selected_feature,
            operator=operator_enum,
            right=right_operand,
        )

        details = {
            "family": selected_feature.economic_family.value,
            "feature_index": selected_feature.column_index,
            "feature_name": selected_feature.name,
            "feature_profile": profile,
            "operator": operator_enum.value,
            "operand_kind": explicit_right_kind,
            "operand_value": _serialize_value(right_operand),
            "threshold_source": threshold_source,
            "selection_score": selection_score,
        }

        return condition, details

    def _mutate_condition(
        self,
        condition: Condition,
        *,
        target_family: EconomicFamily | None,
        current_hypothesis: Hypothesis,
    ) -> tuple[Condition, dict[str, Any]]:
        profile = _feature_profile(condition.left)
        current_right_kind = "feature" if isinstance(condition.right, Feature) else "constant"
        mutation_kind = self._pick_mutation_kind(condition, current_right_kind)

        if mutation_kind == "threshold":
            if current_right_kind == "constant":
                new_right = self._mutate_constant_operand(condition.left, condition.right)
                if new_right is None:
                    mutation_kind = "operator"
                else:
                    new_operator = self._pick_operator(
                        condition.left,
                        new_right,
                        right_kind="constant",
                        avoid_operator=condition.operator,
                    )
                    mutated = Condition(condition.left, new_operator, new_right)
                    return mutated, {
                        "family": condition.left.economic_family.value,
                        "feature_index": condition.left.column_index,
                        "feature_name": condition.left.name,
                        "operator": new_operator.value,
                        "operand_kind": "constant",
                        "operand_value": _serialize_value(new_right),
                        "threshold_source": "mutation",
                        "mutation_subaction": "threshold",
                    }

            if current_right_kind == "feature":
                threshold_candidates, threshold_source = self._threshold_candidates(condition.left)
                if threshold_candidates:
                    new_right = self._pick_threshold_value(condition.left, threshold_candidates, exclude=condition.right)
                    new_operator = self._pick_operator(
                        condition.left,
                        new_right,
                        right_kind="constant",
                        avoid_operator=condition.operator,
                    )
                    mutated = Condition(condition.left, new_operator, new_right)
                    return mutated, {
                        "family": condition.left.economic_family.value,
                        "feature_index": condition.left.column_index,
                        "feature_name": condition.left.name,
                        "operator": new_operator.value,
                        "operand_kind": "constant",
                        "operand_value": _serialize_value(new_right),
                        "threshold_source": threshold_source,
                        "mutation_subaction": "threshold",
                    }

                mutation_kind = "feature"

        if mutation_kind == "operator":
            if current_right_kind == "feature":
                new_operator = self._pick_operator(
                    condition.left,
                    condition.right,
                    right_kind="feature",
                    avoid_operator=condition.operator,
                )
                mutated = Condition(condition.left, new_operator, condition.right)
                return mutated, {
                    "family": condition.left.economic_family.value,
                    "feature_index": condition.left.column_index,
                    "feature_name": condition.left.name,
                    "operator": new_operator.value,
                    "operand_kind": "feature",
                    "operand_value": _serialize_value(condition.right),
                    "threshold_source": "operator_mutation",
                    "mutation_subaction": "operator",
                }

            new_operator = self._pick_operator(
                condition.left,
                condition.right,
                right_kind="constant",
                avoid_operator=condition.operator,
            )
            mutated = Condition(condition.left, new_operator, condition.right)
            return mutated, {
                "family": condition.left.economic_family.value,
                "feature_index": condition.left.column_index,
                "feature_name": condition.left.name,
                "operator": new_operator.value,
                "operand_kind": "constant",
                "operand_value": _serialize_value(condition.right),
                "threshold_source": "operator_mutation",
                "mutation_subaction": "operator",
            }

        if mutation_kind == "feature":
            family_to_use = target_family or condition.left.economic_family
            new_left = self._pick_feature(
                family=family_to_use,
                avoid_features={condition.left.column_index},
                seedable=False,
            )

            if isinstance(condition.right, Feature):
                candidate, details = self._build_condition_spec(
                    feature=new_left,
                    family=family_to_use,
                    prefer_feature_right=True,
                    avoid_features={condition.left.column_index, condition.right.column_index},
                )
                return candidate, {
                    **details,
                    "mutation_subaction": "feature",
                }

            threshold_candidates, threshold_source = self._threshold_candidates(new_left)
            if threshold_candidates:
                new_right = self._pick_threshold_value(new_left, threshold_candidates)
                new_operator = self._pick_operator(
                    new_left,
                    new_right,
                    right_kind="constant",
                    avoid_operator=condition.operator,
                )
            else:
                new_right = self._pick_right_feature(
                    new_left,
                    family_hint=family_to_use,
                    avoid_features={condition.left.column_index},
                )
                new_operator = self._pick_operator(
                    new_left,
                    new_right,
                    right_kind="feature",
                    avoid_operator=condition.operator,
                )
                threshold_source = "fallback_feature"

            mutated = Condition(new_left, new_operator, new_right)
            return mutated, {
                "family": new_left.economic_family.value,
                "feature_index": new_left.column_index,
                "feature_name": new_left.name,
                "operator": new_operator.value,
                "operand_kind": "feature" if isinstance(new_right, Feature) else "constant",
                "operand_value": _serialize_value(new_right),
                "threshold_source": threshold_source,
                "mutation_subaction": "feature",
            }

        raise RuntimeError("Unable to mutate condition.")

    # ==================================================
    # INTERNAL SELECTION OF OPERATOR / OPERAND
    # ==================================================

    def _should_use_feature_right(self, feature: Feature) -> bool:
        profile = _feature_profile(feature)

        if profile == "binary":
            return False

        base = self._settings.feature_right_probability

        if profile == "raw_scale":
            return max(base, 0.85)

        if profile == "oscillator":
            return min(max(base, 0.15), 0.35)

        if profile in {"normalized", "ratio", "distance"}:
            return min(base, 0.10)

        if profile == "statistical":
            return min(max(base, 0.10), 0.30)

        return base

    def _threshold_candidates(
        self,
        feature: Feature,
    ) -> tuple[tuple[Any, ...], str]:
        metadata = feature.metadata or {}

        if callable(self._threshold_policy):
            try:
                result = self._threshold_policy(
                    feature=feature,
                    statistics=self._stats_for_feature(feature),
                    generator=self,
                    config=self._config,
                    profile=_feature_profile(feature),
                )
            except TypeError:
                result = None

            if result is not None:
                if isinstance(result, Mapping):
                    if "candidates" in result:
                        result = result["candidates"]
                    elif "values" in result:
                        result = result["values"]
                    elif "thresholds" in result:
                        result = result["thresholds"]
                    elif "value" in result:
                        result = result["value"]

                values = _normalize_tuple(result)
                values = tuple(_coerce_operand_value(item) for item in values)
                values = _dedupe_preserve_order(values)
                if values:
                    return values, "policy"

        for key in ("threshold_candidates", "thresholds", "candidates"):
            if key in metadata:
                values = metadata[key]
                if isinstance(values, Mapping):
                    values = list(values.values())
                values = tuple(_coerce_operand_value(item) for item in _normalize_tuple(values))
                values = _dedupe_preserve_order(values)
                if values:
                    return values, "metadata"

        for key in ("quantiles", "percentiles"):
            if key in metadata:
                values = self._extract_values_from_quantiles(metadata[key])
                values = tuple(_coerce_operand_value(item) for item in values)
                values = _dedupe_preserve_order(values)
                if values:
                    return values, "metadata"

        stats = self._stats_for_feature(feature)
        if stats:
            values = self._threshold_candidates_from_stats(feature, stats)
            values = tuple(_coerce_operand_value(item) for item in values)
            values = _dedupe_preserve_order(values)
            if values:
                return values, "statistics"

        profile = _feature_profile(feature)
        if profile == "binary":
            return (True, False), "heuristic"

        if profile == "normalized":
            return (0.25, 0.50, 0.75), "heuristic"

        if profile == "ratio":
            return (0.80, 1.00, 1.20, 1.50, 2.00, 3.00), "heuristic"

        if profile == "distance":
            return (-1.0, -0.50, 0.0, 0.50, 1.0), "heuristic"

        if profile == "oscillator":
            levels = _canonical_thresholds_for_feature_name(feature.name)
            if levels:
                return levels, "heuristic"
            return (-1.0, 0.0, 1.0), "heuristic"

        if profile == "raw_scale":
            return (), "heuristic"

        if profile == "statistical":
            return (-1.0, 0.0, 1.0), "heuristic"

        if profile == "generic_numeric":
            return (-1.0, 0.0, 1.0), "heuristic"

        return (), "heuristic"

    def _threshold_candidates_from_stats(
        self,
        feature: Feature,
        stats: Mapping[str, Any],
    ) -> tuple[Any, ...]:
        values: list[Any] = []

        quantiles = None
        for key in ("quantiles", "percentiles"):
            if key in stats:
                quantiles = stats[key]
                break

        if quantiles is not None:
            values.extend(self._extract_values_from_quantiles(quantiles))

        mean = stats.get("mean")
        std = stats.get("std")
        minimum = stats.get("min")
        maximum = stats.get("max")

        if mean is not None and std is not None:
            mean = float(mean)
            std = float(std)
            values.extend([mean - std, mean, mean + std])

        elif minimum is not None and maximum is not None:
            minimum = float(minimum)
            maximum = float(maximum)
            spread = maximum - minimum
            values.extend(
                [
                    minimum + 0.25 * spread,
                    minimum + 0.50 * spread,
                    minimum + 0.75 * spread,
                ]
            )

        elif minimum is not None:
            minimum = float(minimum)
            values.append(minimum)

        elif maximum is not None:
            maximum = float(maximum)
            values.append(maximum)

        if feature.name.lower() in RAW_FEATURE_NAMES and values:
            return tuple(values)

        return tuple(values)

    def _extract_values_from_quantiles(self, quantiles: Any) -> tuple[Any, ...]:
        if isinstance(quantiles, Mapping):
            items = list(quantiles.items())

            def sort_key(item: tuple[Any, Any]) -> float:
                key = item[0]
                try:
                    return float(key)
                except (TypeError, ValueError):
                    text = str(key).lower().replace("q", "")
                    try:
                        return float(text)
                    except (TypeError, ValueError):
                        return 0.0

            items.sort(key=sort_key)
            return tuple(_coerce_operand_value(value) for _key, value in items)

        values = _normalize_tuple(quantiles)
        return tuple(_coerce_operand_value(value) for value in values)

    def _stats_for_feature(self, feature: Feature) -> Mapping[str, Any]:
        candidates = [
            feature.column_index,
            str(feature.column_index),
            feature.name,
            feature.name.lower(),
        ]

        for key in candidates:
            if key in self._feature_statistics:
                return _to_mapping(self._feature_statistics[key])

        metadata = feature.metadata or {}
        for key in ("statistics", "stats", "distribution"):
            value = metadata.get(key)
            if value is not None:
                return _to_mapping(value)

        return {}

    def _pick_threshold_value(
        self,
        feature: Feature,
        candidates: Sequence[Any],
        *,
        exclude: Any | None = None,
    ) -> Any:
        if not candidates:
            raise RuntimeError("No threshold candidate available.")

        values = [item for item in candidates if _fingerprint_key(item) != _fingerprint_key(exclude)]
        if not values:
            values = list(candidates)

        if not values:
            raise RuntimeError("No usable threshold candidate available.")

        profile = _feature_profile(feature)
        if profile == "binary":
            weights = []
            for value in values:
                key = self._threshold_usage_key(feature, ComparisonOperator.EQ, value)
                usage = 1.0 + float(self._threshold_usage.get(key, 0))
                weights.append(1.0 / usage)
            choice = self._weighted_choice(values, weights)
            return bool(choice)

        weights = []
        for value in values:
            key = self._threshold_usage_key(feature, ComparisonOperator.EQ, value)
            usage = 1.0 + float(self._threshold_usage.get(key, 0))
            weights.append(1.0 / usage)

        choice = self._weighted_choice(values, weights)
        return _coerce_operand_value(choice)

    def _fallback_constants_for_profile(self, profile: str) -> tuple[Any, ...]:
        if profile == "binary":
            return (True, False)
        if profile == "normalized":
            return (0.25, 0.50, 0.75)
        if profile == "ratio":
            return (0.80, 1.00, 1.20, 1.50, 2.00)
        if profile == "distance":
            return (-1.0, -0.50, 0.0, 0.50, 1.0)
        if profile == "oscillator":
            return (-1.0, 0.0, 1.0)
        if profile == "statistical":
            return (-1.0, 0.0, 1.0)
        if profile == "generic_numeric":
            return (-1.0, 0.0, 1.0)
        return (0.0, 1.0, -1.0)

    def _pick_operator(
        self,
        feature: Feature,
        right_operand: Feature | int | float | bool,
        *,
        right_kind: str | None,
        avoid_operator: ComparisonOperator | None = None,
    ) -> ComparisonOperator:
        pool = self._operator_pool(feature, right_operand, right_kind=right_kind)
        if avoid_operator is not None:
            pool = [operator for operator in pool if operator != avoid_operator]

        if not pool:
            raise RuntimeError("No compatible operator available.")

        weights = self._operator_weights(feature, right_operand, pool, right_kind=right_kind)
        return self._weighted_choice(pool, weights)

    def _operator_pool(
        self,
        feature: Feature,
        right_operand: Feature | int | float | bool,
        *,
        right_kind: str | None,
    ) -> list[ComparisonOperator]:
        profile = _feature_profile(feature)
        metadata = feature.metadata or {}
        allowed = self._metadata_operator_set(metadata)

        if right_kind == "feature":
            pool = [
                ComparisonOperator.LT,
                ComparisonOperator.LE,
                ComparisonOperator.GT,
                ComparisonOperator.GE,
                ComparisonOperator.EQ,
                ComparisonOperator.NE,
                ComparisonOperator.CROSS_OVER,
                ComparisonOperator.CROSS_UNDER,
            ]
        else:
            if profile == "binary":
                pool = [
                    ComparisonOperator.IS_TRUE,
                    ComparisonOperator.IS_FALSE,
                    ComparisonOperator.EQ,
                    ComparisonOperator.NE,
                ]
            else:
                pool = [
                    ComparisonOperator.LT,
                    ComparisonOperator.LE,
                    ComparisonOperator.GT,
                    ComparisonOperator.GE,
                    ComparisonOperator.EQ,
                    ComparisonOperator.NE,
                ]

        if allowed:
            pool = [operator for operator in pool if operator in allowed]

        return pool

    def _metadata_operator_set(self, metadata: Mapping[str, Any]) -> set[ComparisonOperator]:
        for key in ("allowed_operators", "operators", "comparators", "operator_candidates"):
            if key not in metadata:
                continue

            values = metadata[key]
            if isinstance(values, Mapping):
                values = values.keys()

            operators: set[ComparisonOperator] = set()
            for item in _normalize_tuple(values):
                try:
                    operators.add(_coerce_operator(item))  # type: ignore[arg-type]
                except ValueError:
                    continue

            operators.discard(None)  # type: ignore[arg-type]
            return {operator for operator in operators if operator is not None}

        return set()

    def _operator_weights(
        self,
        feature: Feature,
        right_operand: Feature | int | float | bool,
        pool: Sequence[ComparisonOperator],
        *,
        right_kind: str | None,
    ) -> list[float]:
        metadata = feature.metadata or {}
        operator_weights = _to_mapping(metadata.get("operator_weights"))
        profile = _feature_profile(feature)

        threshold_value = right_operand if not isinstance(right_operand, Feature) else None

        weights: list[float] = []
        for operator in pool:
            weight = 1.0

            if str(operator.value) in operator_weights:
                weight *= max(0.0, _coerce_float(operator_weights[str(operator.value)], 1.0))
            elif operator.name.lower() in operator_weights:
                weight *= max(0.0, _coerce_float(operator_weights[operator.name.lower()], 1.0))

            if self._settings.family_balance:
                weight *= self._operator_family_bias(feature.economic_family, operator)

            usage = 1.0 + float(self._operator_usage.get(operator.value, 0))
            weight /= usage

            if right_kind == "feature":
                if operator in {ComparisonOperator.CROSS_OVER, ComparisonOperator.CROSS_UNDER}:
                    weight *= 1.50
                elif operator in {ComparisonOperator.GT, ComparisonOperator.GE, ComparisonOperator.LT, ComparisonOperator.LE}:
                    weight *= 1.20
                elif operator in {ComparisonOperator.EQ, ComparisonOperator.NE}:
                    weight *= 0.85

            else:
                if profile == "binary":
                    if right_operand is True and operator in {ComparisonOperator.IS_TRUE, ComparisonOperator.EQ}:
                        weight *= 1.75
                    elif right_operand is False and operator in {ComparisonOperator.IS_FALSE, ComparisonOperator.NE}:
                        weight *= 1.75
                elif isinstance(right_operand, (int, float)):
                    center = self._threshold_center(feature, threshold_value)
                    if center is not None:
                        try:
                            threshold = float(right_operand)
                            if threshold < center and operator in {ComparisonOperator.LT, ComparisonOperator.LE}:
                                weight *= 1.75
                            elif threshold > center and operator in {ComparisonOperator.GT, ComparisonOperator.GE}:
                                weight *= 1.75
                            elif threshold == center and operator in {ComparisonOperator.EQ, ComparisonOperator.NE}:
                                weight *= 1.25
                        except (TypeError, ValueError):
                            pass

            weights.append(max(0.0001, weight))

        return weights

    def _operator_family_bias(
        self,
        family: EconomicFamily,
        operator: ComparisonOperator,
    ) -> float:
        if family in {EconomicFamily.MOMENTUM, EconomicFamily.TREND}:
            if operator in {ComparisonOperator.GT, ComparisonOperator.GE, ComparisonOperator.CROSS_OVER}:
                return 1.35
            if operator in {ComparisonOperator.LT, ComparisonOperator.LE, ComparisonOperator.CROSS_UNDER}:
                return 0.95

        if family in {EconomicFamily.VOLATILITY, EconomicFamily.RISK}:
            if operator in {ComparisonOperator.GT, ComparisonOperator.GE}:
                return 1.20
            if operator in {ComparisonOperator.LT, ComparisonOperator.LE}:
                return 0.95

        if family in {EconomicFamily.VOLUME_FLOW, EconomicFamily.MICROSTRUCTURE, EconomicFamily.CROSS_ASSET, EconomicFamily.MARKET_STRUCTURE}:
            if operator in {ComparisonOperator.CROSS_OVER, ComparisonOperator.CROSS_UNDER}:
                return 1.40

        if family in {EconomicFamily.STATISTICAL, EconomicFamily.SENTIMENT}:
            if operator in {ComparisonOperator.EQ, ComparisonOperator.NE}:
                return 1.10

        return 1.0

    def _threshold_center(self, feature: Feature, threshold_value: Any) -> float | None:
        candidates, _source = self._threshold_candidates(feature)
        numeric_candidates = [float(item) for item in candidates if isinstance(item, (int, float))]
        if not numeric_candidates:
            return None
        return float(sum(numeric_candidates) / len(numeric_candidates))

    def _mutate_constant_operand(
        self,
        feature: Feature,
        current_value: Any,
    ) -> Any | None:
        candidates, _source = self._threshold_candidates(feature)
        if not candidates:
            candidates = self._fallback_constants_for_profile(_feature_profile(feature))

        values = [item for item in candidates if _fingerprint_key(item) != _fingerprint_key(current_value)]
        if not values:
            return None

        weights = []
        for value in values:
            key = self._threshold_usage_key(feature, ComparisonOperator.EQ, value)
            usage = 1.0 + float(self._threshold_usage.get(key, 0))
            weights.append(1.0 / usage)

        return _coerce_operand_value(self._weighted_choice(values, weights))

    def _pick_mutation_kind(
        self,
        condition: Condition,
        right_kind: str,
    ) -> str:
        profile = _feature_profile(condition.left)
        weights = dict(self._settings.mutation_weights)

        available: list[str] = []
        if right_kind == "constant":
            available.extend(["threshold", "operator", "feature"])
        else:
            available.extend(["operator", "feature", "threshold"])

        if profile == "binary":
            available = [kind for kind in available if kind in {"threshold", "operator", "feature"}]
        elif profile == "raw_scale" and right_kind == "constant":
            available = [kind for kind in available if kind in {"operator", "feature", "threshold"}]

        available = tuple(dict.fromkeys(available))

        filtered_weights = []
        for kind in available:
            filtered_weights.append(max(0.0, _coerce_float(weights.get(kind, 1.0), 1.0)))

        if max(filtered_weights, default=0.0) <= 0:
            filtered_weights = [1.0 for _ in available]

        return str(self._weighted_choice(list(available), filtered_weights))

    # ==================================================
    # INTERNAL VALIDATION / NORMALIZATION
    # ==================================================

    def _validate_operator_compatibility(
        self,
        operator: ComparisonOperator,
        right_kind: str | None,
    ) -> None:
        if right_kind == "feature":
            allowed = {
                ComparisonOperator.LT,
                ComparisonOperator.LE,
                ComparisonOperator.GT,
                ComparisonOperator.GE,
                ComparisonOperator.EQ,
                ComparisonOperator.NE,
                ComparisonOperator.CROSS_OVER,
                ComparisonOperator.CROSS_UNDER,
            }
            if operator not in allowed:
                raise ValueError(f"Operator {operator.value} is incompatible with a feature operand.")
            return

        allowed = {
            ComparisonOperator.LT,
            ComparisonOperator.LE,
            ComparisonOperator.GT,
            ComparisonOperator.GE,
            ComparisonOperator.EQ,
            ComparisonOperator.NE,
            ComparisonOperator.IS_TRUE,
            ComparisonOperator.IS_FALSE,
        }
        if operator not in allowed:
            raise ValueError(f"Operator {operator.value} is incompatible with a constant operand.")

    def _normalize_feature_avoidance(
        self,
        avoid_features: Iterable[int | Feature] | None,
    ) -> set[int]:
        if avoid_features is None:
            return set()

        output: set[int] = set()
        for item in avoid_features:
            if isinstance(item, Feature):
                output.add(item.column_index)
            else:
                output.add(_coerce_int(item, -1))
        output.discard(-1)
        return output

    def _coerce_right_operand(self, right: Feature | int | float | bool) -> Feature | int | float | bool:
        if isinstance(right, Feature):
            return right
        return _coerce_operand_value(right)

    def _operand_key(self, operand: Feature | int | float | bool) -> str:
        return _fingerprint_key(_serialize_value(operand))

    def _threshold_usage_key(
        self,
        feature: Feature,
        operator: ComparisonOperator,
        operand_key: Any,
    ) -> str:
        return f"{feature.column_index}:{operator.value}:{_fingerprint_key(operand_key)}"

    def _condition_signature(self, condition: Condition) -> str:
        return fingerprint_model(condition)

    def _has_duplicate_conditions(self, conditions: Sequence[Condition]) -> bool:
        signatures = [self._condition_signature(condition) for condition in conditions]
        return len(signatures) != len(set(signatures))

    # ==================================================
    # INTERNAL RESULT / COMMIT
    # ==================================================

    def _generate_unique(
        self,
        action: str,
        parent_fingerprint: str | None,
        builder: Any,
        *,
        max_attempts: int,
    ) -> GenerationResult:
        last_error: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                hypothesis, payload = builder(attempt)
            except Exception as exc:
                last_error = exc
                continue

            if not isinstance(hypothesis, Hypothesis):
                last_error = TypeError("builder must return a Hypothesis.")
                continue

            if self._has_duplicate_conditions(hypothesis.conditions):
                last_error = ValueError("Generated hypothesis contains duplicate conditions.")
                continue

            fingerprint = fingerprint_model(hypothesis)
            if fingerprint in self._seen_fingerprints:
                last_error = ValueError("Duplicate hypothesis fingerprint.")
                continue

            result = GenerationResult(
                action=action,
                hypothesis=hypothesis,
                attempts=attempt,
                **payload,
            )
            self._commit_result(result)
            return result

        if last_error is not None:
            raise RuntimeError(f"Unable to generate a unique hypothesis after {max_attempts} attempts.") from last_error

        raise RuntimeError(f"Unable to generate a unique hypothesis after {max_attempts} attempts.")

    def _commit_result(self, result: GenerationResult) -> None:
        fingerprint = result.fingerprint
        self._seen_fingerprints.add(fingerprint)
        self._history.append(result)
        self._records[fingerprint] = result
        self._parent_by_fingerprint[fingerprint] = result.parent_fingerprint

        if result.parent_fingerprint is not None:
            self._children_by_fingerprint[result.parent_fingerprint].append(fingerprint)

        self._update_usage(result.hypothesis)

    def _update_usage(self, hypothesis: Hypothesis) -> None:
        for condition in hypothesis.conditions:
            feature = condition.left
            family_key = feature.economic_family.value.lower()
            operator_key = condition.operator.value
            operand_key = self._operand_key(condition.right)
            threshold_key = self._threshold_usage_key(feature, condition.operator, operand_key)

            self._feature_usage[feature.column_index] += 1
            self._family_usage[family_key] += 1
            self._operator_usage[operator_key] += 1
            self._threshold_usage[threshold_key] += 1

    def _extract_hypothesis(
        self,
        hypothesis: Hypothesis | GenerationResult,
    ) -> Hypothesis:
        if isinstance(hypothesis, GenerationResult):
            return hypothesis.hypothesis
        if isinstance(hypothesis, Hypothesis):
            return hypothesis
        raise TypeError("Expected a Hypothesis or a GenerationResult.")

    # ==================================================
    # GENERIC UTILITIES
    # ==================================================

    def _weighted_choice(
        self,
        items: Sequence[Any],
        weights: Sequence[float],
    ) -> Any:
        if not items:
            raise ValueError("items cannot be empty.")

        if len(items) != len(weights):
            raise ValueError("items and weights must have the same length.")

        normalized = np.asarray(weights, dtype=float)
        normalized = np.where(np.isfinite(normalized), normalized, 0.0)

        total = float(normalized.sum())
        if total <= 0.0:
            index = int(self._rng.integers(0, len(items)))
            return items[index]

        probabilities = normalized / total
        index = int(self._rng.choice(len(items), p=probabilities))
        return items[index]