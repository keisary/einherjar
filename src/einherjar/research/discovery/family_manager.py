"""
==========================================================
Discovery Family Manager
==========================================================

Gère l'équilibrage des familles de features pendant la
phase de Discovery.

Ce composant ne valide rien et ne simule rien.
Il fournit au moteur des choix structurés pour éviter :
- la domination d'une seule famille,
- la répétition excessive des mêmes features,
- la recherche non diversifiée.
"""

from __future__ import annotations

from collections import Counter
from collections import defaultdict
from dataclasses import dataclass
from dataclasses import fields
import math
from typing import Any
from typing import Iterable
from typing import Mapping
from typing import Sequence

import numpy as np

from config.scoring import ScoringConfig
from config.search import SearchConfig
from models.condition import Condition
from models.enums import EconomicFamily
from models.feature import Feature
from models.feature_registry import FeatureRegistry
from models.hypothesis import Hypothesis


__all__ = [
    "FamilyManager",
]


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


def _read_value(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)

    return getattr(obj, name, default)


def _build_dataclass(
    cls: type[Any],
    source: Any | None,
) -> Any:
    if source is None:
        return cls()

    if isinstance(source, cls):
        return source

    kwargs: dict[str, Any] = {}
    for field in fields(cls):
        value = _read_value(source, field.name, None)
        if value is not None:
            kwargs[field.name] = value

    return cls(**kwargs)


def _to_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}

    if hasattr(value, "to_dict") and callable(value.to_dict):
        value = value.to_dict()

    if isinstance(value, Mapping):
        return dict(value)

    return {}


def _coerce_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default

    try:
        return int(value)
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


def _normalize_avoid_features(
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


@dataclass(slots=True)
class FamilyManager:
    """
    Gestionnaire d'équilibrage des familles.

    Il maintient un état d'usage des features et des familles
    pour guider la génération d'hypothèses sans laisser une
    seule famille monopoliser la recherche.
    """

    def __init__(
        self,
        registry: FeatureRegistry,
        search_config: SearchConfig | None = None,
        scoring_config: ScoringConfig | None = None,
        *,
        family_weights: Mapping[str, float] | None = None,
        feature_weights: Mapping[str, float] | None = None,
        feature_statistics: Mapping[Any, Mapping[str, Any]] | None = None,
        rng: np.random.Generator | int | None = None,
    ) -> None:
        if not isinstance(registry, FeatureRegistry):
            raise TypeError("registry must be a FeatureRegistry.")

        self._registry = registry
        self._search_config = search_config or SearchConfig()
        self._scoring_config = scoring_config or ScoringConfig()

        self._family_weights = {
            str(key).strip().lower(): max(0.0, _coerce_float(value, 1.0))
            for key, value in (family_weights or {}).items()
        }

        self._feature_weights = {
            str(key).strip().lower(): max(0.0, _coerce_float(value, 1.0))
            for key, value in (feature_weights or {}).items()
        }

        self._feature_statistics: dict[Any, Mapping[str, Any]] = {
            key: _to_mapping(value)
            for key, value in (feature_statistics or {}).items()
        }

        if isinstance(rng, np.random.Generator):
            self._rng = rng
        elif rng is not None:
            self._rng = np.random.default_rng(rng)
        else:
            self._rng = np.random.default_rng(self._search_config.random_seed)

        self._family_features: dict[EconomicFamily, tuple[Feature, ...]] = self._build_family_features()
        self._families: tuple[EconomicFamily, ...] = tuple(self._family_features.keys())

        self._feature_usage: Counter[int] = Counter()
        self._family_usage: Counter[str] = Counter()

    # ==================================================
    # CONSTRUCTION FROM CONFIG
    # ==================================================

    @classmethod
    def from_config(
        cls,
        config: Any,
        registry: FeatureRegistry,
        *,
        feature_statistics: Mapping[Any, Mapping[str, Any]] | None = None,
        rng: np.random.Generator | int | None = None,
    ) -> "FamilyManager":
        search_source = _first_non_none(
            config,
            ("search",),
            ("search_config",),
            ("discovery", "search"),
            default=None,
        )
        scoring_source = _first_non_none(
            config,
            ("scoring",),
            ("scoring_config",),
            ("discovery", "scoring"),
            default=None,
        )

        search_config = _build_dataclass(SearchConfig, search_source)
        scoring_config = _build_dataclass(ScoringConfig, scoring_source)

        family_weights = _to_mapping(
            _first_non_none(
                config,
                ("family_weights",),
                ("families", "weights"),
                ("search", "family_weights"),
                ("discovery", "family_weights"),
                default={},
            )
        )

        feature_weights = _to_mapping(
            _first_non_none(
                config,
                ("feature_weights",),
                ("search", "feature_weights"),
                ("discovery", "feature_weights"),
                default={},
            )
        )

        return cls(
            registry=registry,
            search_config=search_config,
            scoring_config=scoring_config,
            family_weights=family_weights,
            feature_weights=feature_weights,
            feature_statistics=feature_statistics,
            rng=rng,
        )

    # ==================================================
    # PROPERTIES
    # ==================================================

    @property
    def registry(self) -> FeatureRegistry:
        return self._registry

    @property
    def search_config(self) -> SearchConfig:
        return self._search_config

    @property
    def scoring_config(self) -> ScoringConfig:
        return self._scoring_config

    @property
    def families(self) -> tuple[EconomicFamily, ...]:
        return self._families

    @property
    def family_count(self) -> int:
        return len(self._families)

    @property
    def feature_usage(self) -> dict[int, int]:
        return dict(self._feature_usage)

    @property
    def family_usage(self) -> dict[str, int]:
        return dict(self._family_usage)

    @property
    def total_usage(self) -> int:
        return sum(self._feature_usage.values())

    @property
    def features_by_family(self) -> dict[EconomicFamily, tuple[Feature, ...]]:
        return dict(self._family_features)

    # ==================================================
    # FAMILY / FEATURE POOLS
    # ==================================================

    def features(
        self,
        family: EconomicFamily | str | None = None,
        *,
        seedable: bool = False,
        avoid_features: Iterable[int | Feature] | None = None,
        limit: int | None = None,
    ) -> tuple[Feature, ...]:
        pool = self._all_features() if family is None else self._family_features.get(self._coerce_family(family), ())

        if seedable:
            pool = tuple(feature for feature in pool if self._is_seedable_feature(feature))

        avoid = _normalize_avoid_features(avoid_features)
        if avoid:
            pool = tuple(feature for feature in pool if feature.column_index not in avoid)

        ranked = self._rank_features(pool)

        max_candidates = self._candidate_limit()
        effective_limit = limit if limit is not None else max_candidates
        if effective_limit > 0:
            ranked = ranked[:effective_limit]

        return ranked

    def seedable_features(
        self,
        family: EconomicFamily | str | None = None,
        *,
        avoid_features: Iterable[int | Feature] | None = None,
        limit: int | None = None,
    ) -> tuple[Feature, ...]:
        return self.features(
            family=family,
            seedable=True,
            avoid_features=avoid_features,
            limit=limit,
        )

    def available_families(
        self,
        *,
        seedable: bool = False,
        avoid_families: Iterable[EconomicFamily | str] | None = None,
    ) -> tuple[EconomicFamily, ...]:
        avoid = {
            self._coerce_family(item)
            for item in avoid_families or ()
        }
        avoid.discard(None)

        families = []
        for family in self._families:
            if family in avoid:
                continue

            if seedable and not any(self._is_seedable_feature(feature) for feature in self._family_features.get(family, ())):
                continue

            if not self._family_features.get(family, ()):
                continue

            families.append(family)

        return tuple(families)

    # ==================================================
    # SELECTION
    # ==================================================

    def choose_family(
        self,
        *,
        seedable: bool = False,
        avoid_families: Iterable[EconomicFamily | str] | None = None,
        current_families: Iterable[EconomicFamily | str] | None = None,
    ) -> EconomicFamily:
        families = self.available_families(
            seedable=seedable,
            avoid_families=avoid_families,
        )

        if not families:
            raise RuntimeError("No family available for selection.")

        current = {
            self._coerce_family(item)
            for item in current_families or ()
        }
        current.discard(None)

        weights = [
            self._family_score(family, current_families=current)
            for family in families
        ]

        return self._weighted_choice(list(families), weights)

    def choose_feature(
        self,
        family: EconomicFamily | str | None = None,
        *,
        seedable: bool = False,
        avoid_features: Iterable[int | Feature] | None = None,
    ) -> Feature:
        if family is None:
            family = self.choose_family(seedable=seedable)

        family_enum = self._coerce_family(family)
        if family_enum is None:
            raise ValueError("family cannot be None.")

        pool = self.features(
            family_enum,
            seedable=seedable,
            avoid_features=avoid_features,
        )

        if not pool:
            raise RuntimeError(f"No feature available for family {family_enum.value}.")

        weights = [self._feature_score(feature) for feature in pool]
        return self._weighted_choice(list(pool), weights)

    def choose_family_and_feature(
        self,
        *,
        seedable: bool = False,
        avoid_families: Iterable[EconomicFamily | str] | None = None,
        avoid_features: Iterable[int | Feature] | None = None,
        current_families: Iterable[EconomicFamily | str] | None = None,
    ) -> tuple[EconomicFamily, Feature]:
        family = self.choose_family(
            seedable=seedable,
            avoid_families=avoid_families,
            current_families=current_families,
        )
        feature = self.choose_feature(
            family,
            seedable=seedable,
            avoid_features=avoid_features,
        )
        return family, feature

    def pick_seed(
        self,
        *,
        avoid_families: Iterable[EconomicFamily | str] | None = None,
        avoid_features: Iterable[int | Feature] | None = None,
    ) -> tuple[EconomicFamily, Feature]:
        return self.choose_family_and_feature(
            seedable=True,
            avoid_families=avoid_families,
            avoid_features=avoid_features,
        )

    # ==================================================
    # RANKING
    # ==================================================

    def rank_families(
        self,
        *,
        seedable: bool = False,
    ) -> tuple[tuple[EconomicFamily, float], ...]:
        families = self.available_families(seedable=seedable)
        ranked = sorted(
            ((family, self._family_score(family)) for family in families),
            key=lambda item: item[1],
            reverse=True,
        )
        return tuple(ranked)

    def rank_features(
        self,
        family: EconomicFamily | str | None = None,
        *,
        seedable: bool = False,
    ) -> tuple[tuple[Feature, float], ...]:
        pool = self.features(family=family, seedable=seedable, limit=0)
        ranked = sorted(
            ((feature, self._feature_score(feature)) for feature in pool),
            key=lambda item: item[1],
            reverse=True,
        )
        return tuple(ranked)

    def summary(self) -> dict[str, Any]:
        return {
            "family_count": self.family_count,
            "feature_count": len(self._all_features()),
            "total_usage": self.total_usage,
            "families": [
                {
                    "family": family.value,
                    "feature_count": len(self._family_features.get(family, ())),
                    "usage": self._family_usage.get(family.value, 0),
                    "score": self._family_score(family),
                }
                for family in self._families
            ],
        }

    # ==================================================
    # MEMORY / LEARNING
    # ==================================================

    def record_feature(
        self,
        feature: Feature,
        *,
        count: int = 1,
    ) -> None:
        if not isinstance(feature, Feature):
            raise TypeError("feature must be a Feature.")

        count = max(1, _coerce_int(count, 1))
        self._feature_usage[feature.column_index] += count
        self._family_usage[feature.economic_family.value] += count

    def record_condition(
        self,
        condition: Condition,
        *,
        count: int = 1,
    ) -> None:
        if not isinstance(condition, Condition):
            raise TypeError("condition must be a Condition.")

        self.record_feature(condition.left, count=count)

        if isinstance(condition.right, Feature):
            self.record_feature(condition.right, count=count)

    def record_hypothesis(
        self,
        hypothesis: Hypothesis,
        *,
        count: int = 1,
    ) -> None:
        if not isinstance(hypothesis, Hypothesis):
            raise TypeError("hypothesis must be a Hypothesis.")

        for condition in hypothesis.conditions:
            self.record_condition(condition, count=count)

    def reset_usage(self) -> None:
        self._feature_usage.clear()
        self._family_usage.clear()

    # ==================================================
    # INTERNAL BUILDERS
    # ==================================================

    def _build_family_features(self) -> dict[EconomicFamily, tuple[Feature, ...]]:
        grouped: defaultdict[EconomicFamily, list[Feature]] = defaultdict(list)

        for feature in self._registry.features:
            if not feature.enabled:
                continue

            grouped[feature.economic_family].append(feature)

        ordered: dict[EconomicFamily, tuple[Feature, ...]] = {}
        for family in EconomicFamily:
            features = grouped.get(family, [])
            if features:
                ordered[family] = tuple(
                    sorted(
                        features,
                        key=lambda item: item.column_index,
                    )
                )

        return ordered

    def _all_features(self) -> tuple[Feature, ...]:
        features: list[Feature] = []
        for family in self._families:
            features.extend(self._family_features.get(family, ()))
        return tuple(features)

    def _candidate_limit(self) -> int:
        limit = _coerce_int(self._search_config.max_candidates_per_family, 0)
        if limit > 0:
            return limit

        beam_width = _coerce_int(self._search_config.beam_width, 0)
        if beam_width > 0 and self.family_count > 0:
            return max(1, beam_width // self.family_count)

        return 0

    # ==================================================
    # SCORING
    # ==================================================

    def _family_score(
        self,
        family: EconomicFamily,
        *,
        current_families: set[EconomicFamily] | None = None,
    ) -> float:
        pool = self._family_features.get(family, ())
        if not pool:
            return 0.0

        current_families = current_families or set()
        usage = float(self._family_usage.get(family.value, 0))
        family_size = float(len(pool))

        novelty = 1.0 / (1.0 + usage)
        breadth = 1.0 / math.sqrt(1.0 + family_size)
        balance = 1.0 / (1.0 + (1.0 if family in current_families else 0.0) + usage / max(1.0, family_size))

        quality = sum(self._feature_quality(feature) for feature in pool) / max(1.0, family_size)

        custom_family_weight = self._family_weights.get(family.value.lower(), 1.0)

        base_score = (
            self._search_config.exploration_ratio * novelty
            + self._search_config.exploitation_ratio * breadth
            + self._search_config.family_balance_weight * balance
        )

        discovery_bias = (
            self._scoring_config.novelty * novelty
            + self._scoring_config.diversity * breadth
        )

        return max(
            0.0,
            custom_family_weight * quality * (base_score + discovery_bias),
        )

    def _feature_score(self, feature: Feature) -> float:
        if not feature.enabled:
            return 0.0

        family_pool = self._family_features.get(feature.economic_family, ())
        if not family_pool:
            return 0.0

        usage = float(self._feature_usage.get(feature.column_index, 0))
        family_usage = float(self._family_usage.get(feature.economic_family.value, 0))
        family_size = float(len(family_pool))

        novelty = 1.0 / (1.0 + usage)
        diversity = 1.0 / math.sqrt(1.0 + family_usage)
        balance = 1.0 / (1.0 + family_usage / max(1.0, family_size))

        quality = self._feature_quality(feature)

        custom_feature_weight = self._feature_weights.get(
            str(feature.column_index),
            self._feature_weights.get(feature.name.lower(), 1.0),
        )

        base_score = (
            self._search_config.novelty_weight * novelty
            + self._search_config.diversity_weight * diversity
            + self._search_config.family_balance_weight * balance
        )

        discovery_bias = (
            self._scoring_config.novelty * novelty
            + self._scoring_config.diversity * diversity
        )

        return max(
            0.0,
            custom_feature_weight * quality * (base_score + discovery_bias),
        )

    def _feature_quality(self, feature: Feature) -> float:
        if not feature.enabled:
            return 0.0

        metadata = feature.metadata or {}

        quality = 1.0
        quality *= max(0.0, _coerce_float(feature.exploration_weight, 1.0))
        quality *= max(0.0, _coerce_float(feature.novelty_bonus, 1.0))
        quality /= max(1e-9, _coerce_float(feature.complexity_cost, 1.0))

        if "exploration_weight" in metadata:
            quality *= max(0.0, _coerce_float(metadata.get("exploration_weight"), 1.0))

        if "novelty_bonus" in metadata:
            quality *= max(0.0, _coerce_float(metadata.get("novelty_bonus"), 1.0))

        if "complexity_cost" in metadata:
            quality /= max(1e-9, _coerce_float(metadata.get("complexity_cost"), 1.0))

        if "family_weight" in metadata:
            quality *= max(0.0, _coerce_float(metadata.get("family_weight"), 1.0))

        stats = self._feature_stats(feature)
        if stats:
            support = _first_non_none(
                stats,
                ("support",),
                ("coverage",),
                ("valid_ratio",),
                ("sample_ratio",),
                default=None,
            )
            if support is not None:
                support_value = max(0.0, _coerce_float(support, 1.0))
                quality *= min(2.0, max(0.1, support_value))

            signal = _first_non_none(
                stats,
                ("signal_score",),
                ("importance",),
                ("score",),
                default=None,
            )
            if signal is not None:
                signal_value = max(0.0, _coerce_float(signal, 0.0))
                quality *= 1.0 + min(1.0, signal_value)

        return max(0.0, quality)

    def _feature_stats(self, feature: Feature) -> Mapping[str, Any]:
        candidates = (
            feature.column_index,
            str(feature.column_index),
            feature.name,
            feature.name.lower(),
        )

        for key in candidates:
            if key in self._feature_statistics:
                return self._feature_statistics[key]

        metadata = feature.metadata or {}
        for key in ("statistics", "stats", "distribution", "feature_statistics"):
            value = metadata.get(key)
            if value is not None:
                return _to_mapping(value)

        return {}

    # ==================================================
    # INTERNAL FILTERS
    # ==================================================

    def _is_seedable_feature(self, feature: Feature) -> bool:
        if not feature.enabled:
            return False

        metadata = feature.metadata or {}

        if "seedable" in metadata:
            return _coerce_bool(metadata.get("seedable"), True)

        if "include_in_discovery" in metadata:
            return _coerce_bool(metadata.get("include_in_discovery"), True)

        if "discovery_enabled" in metadata:
            return _coerce_bool(metadata.get("discovery_enabled"), True)

        return True

    def _coerce_family(self, family: EconomicFamily | str | None) -> EconomicFamily | None:
        if family is None:
            return None

        if isinstance(family, EconomicFamily):
            return family

        text = str(family).strip()

        try:
            return EconomicFamily(text)
        except ValueError:
            try:
                return EconomicFamily[text.upper()]
            except KeyError as exc:
                raise ValueError(f"Unknown family: {family}") from exc

    def _rank_features(self, pool: Sequence[Feature]) -> tuple[Feature, ...]:
        ranked = sorted(
            pool,
            key=lambda feature: (
                -self._feature_score(feature),
                self._feature_usage.get(feature.column_index, 0),
                feature.column_index,
            ),
        )
        return tuple(ranked)

    # ==================================================
    # PYTHON PROTOCOL
    # ==================================================

    def __contains__(self, item: EconomicFamily | str) -> bool:
        try:
            family = self._coerce_family(item)
        except ValueError:
            return False

        if family is None:
            return False

        return family in self._family_features

    def __len__(self) -> int:
        return self.family_count

    def __iter__(self):
        return iter(self._families)

    def __repr__(self) -> str:
        return (
            "FamilyManager("
            f"families={self.family_count}, "
            f"features={len(self._all_features())}, "
            f"usage={self.total_usage}"
            ")"
        )