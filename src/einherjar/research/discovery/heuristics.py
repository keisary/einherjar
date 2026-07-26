"""
==========================================================
Discovery Heuristics
==========================================================

Politiques de décision du Discovery Engine.

Ce module ne génère aucune condition et ne construit aucune
hypothèse. Il décide uniquement :
- quelle action appliquer,
- quelle famille privilégier,
- quelle condition cibler,
- comment pondérer la recherche,
- quand arrêter ou simplifier.

Il s'appuie sur :
- SearchConfig
- ScoringConfig
- FamilyManager
- Hypothesis
- FeatureRegistry
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import math
from typing import Any
from typing import Iterable
from typing import Mapping
from typing import Sequence

import numpy as np

from config.scoring import ScoringConfig
from config.search import SearchConfig
from models.condition import Condition
from models.enums import ConditionOperator as ComparisonOperator
from models.enums import EconomicFamily
from models.feature import Feature
from models.hypothesis import Hypothesis

from .family_manager import FamilyManager


__all__ = [
    "HeuristicDecision",
    "DiscoveryHeuristics",
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


def _coerce_family(value: EconomicFamily | str | None) -> EconomicFamily | None:
    if value is None:
        return None

    if isinstance(value, EconomicFamily):
        return value

    text = str(value).strip()

    try:
        return EconomicFamily(text)
    except ValueError:
        try:
            return EconomicFamily[text.upper()]
        except KeyError as exc:
            raise ValueError(f"Unknown family: {value}") from exc


def _operator_name(operator: ComparisonOperator | str) -> str:
    if isinstance(operator, ComparisonOperator):
        return operator.value
    return str(operator).strip().lower()


def _coerce_operator(value: ComparisonOperator | str | None) -> ComparisonOperator | None:
    if value is None:
        return None

    if isinstance(value, ComparisonOperator):
        return value

    text = str(value).strip()

    try:
        return ComparisonOperator(text)
    except ValueError:
        try:
            return ComparisonOperator[text.upper()]
        except KeyError as exc:
            raise ValueError(f"Unknown operator: {value}") from exc


def _normalize_feature_avoidance(
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


def _feature_family_count(hypothesis: Hypothesis) -> dict[EconomicFamily, int]:
    counts: dict[EconomicFamily, int] = {}
    for condition in hypothesis.conditions:
        counts[condition.left.economic_family] = counts.get(condition.left.economic_family, 0) + 1
    return counts


def _unique_families(hypothesis: Hypothesis) -> set[EconomicFamily]:
    return {condition.left.economic_family for condition in hypothesis.conditions}


def _unique_feature_indices(hypothesis: Hypothesis) -> set[int]:
    return {condition.left.column_index for condition in hypothesis.conditions}


def _is_numeric_feature(feature: Feature) -> bool:
    return feature.value_type.value in {"float", "integer", "ordinal"}


def _is_binary_feature(feature: Feature) -> bool:
    name = feature.name.lower()

    if feature.value_type.value == "boolean":
        return True

    if feature.feature_type.value == "pattern":
        return True

    if name.endswith("_signal"):
        return True

    metadata = feature.metadata or {}
    if _coerce_bool(metadata.get("binary"), False):
        return True

    profile = str(metadata.get("profile", "")).strip().lower()
    if profile == "binary":
        return True

    return False


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

    if _is_numeric_feature(feature):
        return "generic_numeric"

    return "unsupported"


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


def _dedupe_preserve_order(values: Iterable[Any]) -> tuple[Any, ...]:
    seen: set[str] = set()
    output: list[Any] = []

    for value in values:
        key = repr(value)
        if key in seen:
            continue
        seen.add(key)
        output.append(value)

    return tuple(output)


def _build_search_config(source: Any | None) -> SearchConfig:
    if isinstance(source, SearchConfig):
        return source

    if source is None:
        return SearchConfig()

    return SearchConfig(
        max_conditions=_coerce_int(_first_non_none(source, ("max_conditions",), ("search", "max_conditions"), default=3), 3),
        beam_width=_coerce_int(_first_non_none(source, ("beam_width",), ("search", "beam_width"), default=200), 200),
        max_depth=_coerce_int(_first_non_none(source, ("max_depth",), ("search", "max_depth"), default=3), 3),
        max_candidates_per_family=_coerce_int(
            _first_non_none(source, ("max_candidates_per_family",), ("search", "max_candidates_per_family"), default=100),
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
# DECISION OBJECT
# ==========================================================

@dataclass(frozen=True, slots=True)
class HeuristicDecision:
    """
    Décision produite par les heuristiques de Discovery.
    """

    action: str

    family: EconomicFamily | None = None

    condition_index: int | None = None

    seedable: bool = False

    score: float = 0.0

    depth: int = 0

    reason: str = ""

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "action", str(self.action).strip().lower())
        object.__setattr__(self, "score", float(self.score))
        object.__setattr__(self, "depth", max(0, _coerce_int(self.depth, 0)))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "family": None if self.family is None else self.family.value,
            "condition_index": self.condition_index,
            "seedable": self.seedable,
            "score": self.score,
            "depth": self.depth,
            "reason": self.reason,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "HeuristicDecision":
        family_value = data.get("family")
        family = None if family_value is None else EconomicFamily(family_value)

        return cls(
            action=data["action"],
            family=family,
            condition_index=data.get("condition_index"),
            seedable=_coerce_bool(data.get("seedable"), False),
            score=_coerce_float(data.get("score"), 0.0),
            depth=_coerce_int(data.get("depth"), 0),
            reason=data.get("reason", ""),
            metadata=_to_mapping(data.get("metadata", {})),
        )

    def __repr__(self) -> str:
        family = None if self.family is None else self.family.value
        return (
            "HeuristicDecision("
            f"action='{self.action}', "
            f"family='{family}', "
            f"condition_index={self.condition_index}, "
            f"score={self.score:.4f}, "
            f"depth={self.depth}"
            ")"
        )


# ==========================================================
# HEURISTICS
# ==========================================================

class DiscoveryHeuristics:
    """
    Politiques de décision du Discovery Engine.

    Cette classe ne modifie pas le registre et ne construit
    pas d'hypothèses. Elle décide seulement quoi faire ensuite.
    """

    def __init__(
        self,
        family_manager: FamilyManager,
        search_config: SearchConfig | None = None,
        scoring_config: ScoringConfig | None = None,
        *,
        config: Any | None = None,
        rng: np.random.Generator | int | None = None,
    ) -> None:
        if not isinstance(family_manager, FamilyManager):
            raise TypeError("family_manager must be a FamilyManager.")

        if config is not None:
            if search_config is None:
                search_config = _build_search_config(_first_non_none(config, ("search",), ("search_config",), ("discovery", "search"), default=config))
            if scoring_config is None:
                scoring_config = _build_scoring_config(_first_non_none(config, ("scoring",), ("scoring_config",), ("discovery", "scoring"), default=config))

        self._config = config
        self._family_manager = family_manager
        self._search_config = search_config or SearchConfig()
        self._scoring_config = scoring_config or ScoringConfig()

        if isinstance(rng, np.random.Generator):
            self._rng = rng
        elif rng is not None:
            self._rng = np.random.default_rng(rng)
        else:
            self._rng = np.random.default_rng(self._search_config.random_seed)

        self._max_conditions = max(1, _coerce_int(self._search_config.max_conditions, 1))
        self._max_depth = max(1, _coerce_int(self._search_config.max_depth, 1))
        self._beam_width = max(1, _coerce_int(self._search_config.beam_width, 1))

        self._action_weights = self._extract_action_weights(config)
        self._mutation_weights = self._extract_mutation_weights(config)

    # ==================================================
    # FROM CONFIG
    # ==================================================

    @classmethod
    def from_config(
        cls,
        config: Any,
        family_manager: FamilyManager,
        *,
        rng: np.random.Generator | int | None = None,
    ) -> "DiscoveryHeuristics":
        return cls(
            family_manager=family_manager,
            config=config,
            rng=rng,
        )

    # ==================================================
    # PROPERTIES
    # ==================================================

    @property
    def family_manager(self) -> FamilyManager:
        return self._family_manager

    @property
    def search_config(self) -> SearchConfig:
        return self._search_config

    @property
    def scoring_config(self) -> ScoringConfig:
        return self._scoring_config

    @property
    def max_conditions(self) -> int:
        return self._max_conditions

    @property
    def max_depth(self) -> int:
        return self._max_depth

    @property
    def beam_width(self) -> int:
        return self._beam_width

    @property
    def action_weights_config(self) -> dict[str, float]:
        return dict(self._action_weights)

    @property
    def mutation_weights_config(self) -> dict[str, float]:
        return dict(self._mutation_weights)

    # ==================================================
    # ACTION WEIGHTS
    # ==================================================

    def action_weights(
        self,
        hypothesis: Hypothesis | None,
        *,
        depth: int = 0,
        available_actions: Iterable[str] | None = None,
    ) -> dict[str, float]:
        if hypothesis is None:
            return {"seed": 1.0}

        available = {
            str(action).strip().lower()
            for action in (available_actions or self.available_actions(hypothesis, depth=depth))
        }

        if not available:
            return {}

        score = self.score_hypothesis(hypothesis)
        size = len(hypothesis.conditions)
        density = size / max(1.0, float(self._max_conditions))
        depth_ratio = min(1.0, max(0.0, depth / float(self._max_depth)))
        balance = self._family_balance(hypothesis)
        novelty = self._novelty(hypothesis)
        simplicity = 1.0 / (1.0 + max(0, size - 1))

        expand = (
            self._search_config.exploration_ratio
            * (0.35 + 0.65 * novelty)
            * max(0.10, 1.0 - density)
            * max(0.15, 1.0 - depth_ratio)
        )

        mutate = (
            self._search_config.exploitation_ratio
            * (0.25 + 0.75 * score)
            * (0.65 + 0.35 * balance)
        )

        replace = (
            self._search_config.exploitation_ratio
            * (0.20 + 0.80 * (1.0 - score))
            * (0.70 + 0.30 * (1.0 - novelty))
        )

        prune = (
            self._search_config.diversity_weight
            * (0.20 + 0.80 * (1.0 - balance))
            * (0.35 + 0.65 * density)
            * (0.35 + 0.65 * (1.0 - simplicity))
        )

        weights = {
            "expand": expand,
            "mutate": mutate,
            "replace": replace,
            "prune": prune,
        }

        if size >= self._max_conditions:
            weights["expand"] = 0.0

        if size <= 1:
            weights["prune"] = 0.0

        if depth >= self._max_depth:
            weights["expand"] = 0.0

        for action in list(weights.keys()):
            if action not in available:
                weights[action] = 0.0

        for action, custom_weight in self._action_weights.items():
            if action in weights:
                weights[action] *= max(0.0, custom_weight)

        return weights

    def choose_action(
        self,
        hypothesis: Hypothesis | None,
        *,
        depth: int = 0,
        available_actions: Iterable[str] | None = None,
    ) -> str:
        if hypothesis is None:
            return "seed"

        weights = self.action_weights(
            hypothesis,
            depth=depth,
            available_actions=available_actions,
        )

        actions = [action for action, weight in weights.items() if weight > 0]
        if not actions:
            raise RuntimeError("No available discovery action.")

        scores = [weights[action] for action in actions]
        return str(self._weighted_choice(actions, scores))

    def should_expand(
        self,
        hypothesis: Hypothesis,
        *,
        depth: int = 0,
    ) -> bool:
        if depth >= self._max_depth:
            return False
        if len(hypothesis.conditions) >= self._max_conditions:
            return False

        weights = self.action_weights(hypothesis, depth=depth, available_actions=("expand",))
        return weights.get("expand", 0.0) > 0.0

    def should_prune(
        self,
        hypothesis: Hypothesis,
        *,
        depth: int = 0,
    ) -> bool:
        if len(hypothesis.conditions) <= 1:
            return False

        weights = self.action_weights(hypothesis, depth=depth, available_actions=("prune",))
        return weights.get("prune", 0.0) > 0.0

    def should_continue(
        self,
        hypothesis: Hypothesis,
        *,
        depth: int = 0,
    ) -> bool:
        if depth >= self._max_depth and len(hypothesis.conditions) >= self._max_conditions:
            return False
        return True

    # ==================================================
    # FAMILY / FEATURE SELECTION
    # ==================================================

    def choose_family(
        self,
        hypothesis: Hypothesis | None = None,
        *,
        action: str = "expand",
        seedable: bool = False,
        avoid_families: Iterable[EconomicFamily | str] | None = None,
    ) -> EconomicFamily:
        avoid = {
            self._coerce_family(item)
            for item in (avoid_families or ())
        }
        avoid.discard(None)

        current_families = set()
        if hypothesis is not None:
            current_families = _unique_families(hypothesis)

        prefer_new = action in {"seed", "expand"}

        if hypothesis is None:
            return self._family_manager.choose_family(
                seedable=seedable,
                avoid_families=avoid,
                current_families=current_families,
            )

        if prefer_new:
            candidate_avoid = set(avoid)
            candidate_avoid.update(current_families)
            available = self._family_manager.available_families(
                seedable=seedable,
                avoid_families=candidate_avoid,
            )
            if available:
                return self._family_manager.choose_family(
                    seedable=seedable,
                    avoid_families=candidate_avoid,
                    current_families=current_families,
                )

        return self._family_manager.choose_family(
            seedable=seedable,
            avoid_families=avoid,
            current_families=current_families,
        )

    def choose_feature(
        self,
        family: EconomicFamily | str | None = None,
        *,
        seedable: bool = False,
        avoid_features: Iterable[int | Feature] | None = None,
    ) -> Feature:
        return self._family_manager.choose_feature(
            family=family,
            seedable=seedable,
            avoid_features=avoid_features,
        )

    def choose_condition_index(
        self,
        hypothesis: Hypothesis,
        *,
        strategy: str | None = None,
    ) -> int:
        if not hypothesis.conditions:
            raise ValueError("Hypothesis has no conditions.")

        if len(hypothesis.conditions) == 1:
            return 0

        strategy = (strategy or self._auto_condition_strategy(hypothesis)).strip().lower()

        if strategy == "random":
            return int(self._rng.integers(0, len(hypothesis.conditions)))

        scores = [
            self._condition_score(condition, hypothesis)
            for condition in hypothesis.conditions
        ]

        if strategy == "strongest":
            return int(np.argmax(np.asarray(scores, dtype=float)))

        if strategy == "weakest":
            return int(np.argmin(np.asarray(scores, dtype=float)))

        if strategy == "weighted":
            inverse = [1.0 / max(0.0001, score) for score in scores]
            return int(self._weighted_choice(list(range(len(hypothesis.conditions))), inverse))

        raise ValueError(f"Unknown strategy: {strategy}")

    # ==================================================
    # SCORING
    # ==================================================

    def score_hypothesis(
        self,
        hypothesis: Hypothesis,
        *,
        depth: int = 0,
    ) -> float:
        if not hypothesis.conditions:
            return 0.0

        novelty = self._novelty(hypothesis)
        diversity = self._diversity(hypothesis)
        robustness = self._robustness(hypothesis)
        persistence = self._persistence(hypothesis)
        profitability = self._profitability_proxy(hypothesis)

        weights = self._scoring_config

        numerator = (
            weights.novelty * novelty
            + weights.diversity * diversity
            + weights.robustness * robustness
            + weights.persistence * persistence
            + weights.profitability * profitability
        )

        denominator = max(
            1e-9,
            weights.novelty
            + weights.diversity
            + weights.robustness
            + weights.persistence
            + weights.profitability,
        )

        score = numerator / denominator

        depth_ratio = min(1.0, max(0.0, depth / float(self._max_depth)))
        score *= 1.0 - (0.15 * depth_ratio)

        return max(0.0, min(1.0, score))

    # ==================================================
    # DECISION PLAN
    # ==================================================

    def seed_decision(
        self,
        *,
        avoid_families: Iterable[EconomicFamily | str] | None = None,
    ) -> HeuristicDecision:
        family = self.choose_family(
            None,
            action="seed",
            seedable=True,
            avoid_families=avoid_families,
        )
        return HeuristicDecision(
            action="seed",
            family=family,
            seedable=True,
            score=1.0,
            reason="initial_seed",
            metadata={
                "strategy": "seed",
                "seedable": True,
            },
        )

    def plan(
        self,
        hypothesis: Hypothesis | None,
        *,
        depth: int = 0,
        available_actions: Iterable[str] | None = None,
        family_hint: EconomicFamily | str | None = None,
    ) -> HeuristicDecision:
        if hypothesis is None:
            return self.seed_decision()

        action = self.choose_action(
            hypothesis,
            depth=depth,
            available_actions=available_actions,
        )

        score = self.score_hypothesis(hypothesis, depth=depth)

        if action == "expand":
            family = self.choose_family(
                hypothesis,
                action="expand",
                seedable=False,
            )
            return HeuristicDecision(
                action=action,
                family=family,
                seedable=False,
                score=score,
                depth=depth,
                reason="expand_with_new_family",
                metadata={
                    "strategy": "expand",
                    "family_selection": "new_family_preferred",
                },
            )

        if action in {"mutate", "replace", "prune"}:
            condition_index = self.choose_condition_index(
                hypothesis,
                strategy="weakest" if action == "prune" else "weighted",
            )
            selected_family = hypothesis.conditions[condition_index].left.economic_family

            if family_hint is not None:
                hinted = self._coerce_family(family_hint)
                if hinted is not None:
                    selected_family = hinted

            return HeuristicDecision(
                action=action,
                family=selected_family,
                condition_index=condition_index,
                seedable=False,
                score=score,
                depth=depth,
                reason=f"{action}_condition_{condition_index}",
                metadata={
                    "strategy": action,
                    "condition_index": condition_index,
                    "family": selected_family.value,
                },
            )

        raise ValueError(f"Unsupported action: {action}")

    # ==================================================
    # INTERNAL SCORING HELPERS
    # ==================================================

    def _novelty(self, hypothesis: Hypothesis) -> float:
        if not hypothesis.conditions:
            return 0.0

        values = []
        for condition in hypothesis.conditions:
            feature_usage = float(self._family_manager.feature_usage.get(condition.left.column_index, 0))
            family_usage = float(self._family_manager.family_usage.get(condition.left.economic_family.value, 0))
            values.append(1.0 / (1.0 + feature_usage + 0.5 * family_usage))

        return float(sum(values) / len(values))

    def _diversity(self, hypothesis: Hypothesis) -> float:
        if not hypothesis.conditions:
            return 0.0

        families = _unique_families(hypothesis)
        return len(families) / max(1.0, float(len(hypothesis.conditions)))

    def _family_balance(self, hypothesis: Hypothesis) -> float:
        if not hypothesis.conditions:
            return 0.0

        counts = _feature_family_count(hypothesis)
        maximum = max(counts.values(), default=1)
        total = len(hypothesis.conditions)
        imbalance = maximum / max(1.0, float(total))
        return max(0.0, 1.0 - imbalance)

    def _robustness(self, hypothesis: Hypothesis) -> float:
        if not hypothesis.conditions:
            return 0.0

        diversity = self._diversity(hypothesis)
        balance = self._family_balance(hypothesis)
        size = len(hypothesis.conditions)

        simplicity = 1.0 / (1.0 + max(0, size - 1))
        return max(0.0, min(1.0, 0.5 * diversity + 0.3 * balance + 0.2 * simplicity))

    def _persistence(self, hypothesis: Hypothesis) -> float:
        if not hypothesis.conditions:
            return 0.0

        size = len(hypothesis.conditions)
        repeated_families = size - len(_unique_families(hypothesis))
        repeated_features = size - len(_unique_feature_indices(hypothesis))
        penalty = repeated_families + repeated_features
        return max(0.0, 1.0 / (1.0 + penalty))

    def _profitability_proxy(self, hypothesis: Hypothesis) -> float:
        if not hypothesis.conditions:
            return 0.0

        size = len(hypothesis.conditions)
        balance = self._family_balance(hypothesis)
        novelty = self._novelty(hypothesis)
        simplicity = 1.0 / (1.0 + 0.25 * max(0, size - 1))

        return max(0.0, min(1.0, 0.4 * balance + 0.35 * novelty + 0.25 * simplicity))

    def _condition_score(
        self,
        condition: Condition,
        hypothesis: Hypothesis,
    ) -> float:
        feature = condition.left
        family_key = feature.economic_family.value
        feature_usage = float(self._family_manager.feature_usage.get(feature.column_index, 0))
        family_usage = float(self._family_manager.family_usage.get(family_key, 0))

        feature_novelty = 1.0 / (1.0 + feature_usage)
        family_novelty = 1.0 / (1.0 + family_usage)

        local_family_count = sum(
            1
            for item in hypothesis.conditions
            if item.left.economic_family == feature.economic_family
        )
        local_balance = 1.0 / max(1.0, float(local_family_count))

        if isinstance(condition.right, Feature):
            right_bonus = 0.08
        else:
            right_bonus = 0.0

        operator_bias = 1.0
        if condition.operator in {
            ComparisonOperator.GT,
            ComparisonOperator.GE,
            ComparisonOperator.LT,
            ComparisonOperator.LE,
            ComparisonOperator.CROSS_OVER,
            ComparisonOperator.CROSS_UNDER,
        }:
            operator_bias *= 1.1
        elif condition.operator in {
            ComparisonOperator.EQ,
            ComparisonOperator.NE,
            ComparisonOperator.IS_TRUE,
            ComparisonOperator.IS_FALSE,
        }:
            operator_bias *= 0.9

        profile = _feature_profile(feature)
        profile_bonus = 1.0
        if profile in {"oscillator", "normalized", "ratio", "distance"}:
            profile_bonus *= 1.1
        elif profile in {"raw_scale", "statistical"}:
            profile_bonus *= 1.0
        elif profile == "binary":
            profile_bonus *= 0.95

        score = (
            0.40 * feature_novelty
            + 0.25 * family_novelty
            + 0.20 * local_balance
            + 0.10 * operator_bias
            + 0.05 * right_bonus
        ) * profile_bonus

        return max(0.0001, score)

    def _auto_condition_strategy(self, hypothesis: Hypothesis) -> str:
        size = len(hypothesis.conditions)
        if size <= 1:
            return "random"

        if self._family_balance(hypothesis) < 0.40:
            return "weakest"

        if self._novelty(hypothesis) < 0.35:
            return "weighted"

        if size >= max(2, self._max_conditions - 1):
            return "weakest"

        return "weighted"

    # ==================================================
    # INTERNAL CONFIG EXTRACTION
    # ==================================================

    def _extract_action_weights(self, config: Any | None) -> dict[str, float]:
        default = {
            "expand": 1.0,
            "mutate": 1.0,
            "replace": 1.0,
            "prune": 1.0,
        }

        if config is None:
            return default

        mapping = _to_mapping(
            _first_non_none(
                config,
                ("action_weights",),
                ("discovery", "action_weights"),
                ("search", "action_weights"),
                default={},
            )
        )

        if not mapping:
            return default

        result = dict(default)
        for key, value in mapping.items():
            key_name = str(key).strip().lower()
            if key_name in result:
                result[key_name] = max(0.0, _coerce_float(value, 1.0))

        return result

    def _extract_mutation_weights(self, config: Any | None) -> dict[str, float]:
        default = {
            "threshold": 1.0,
            "operator": 1.0,
            "feature": 1.0,
        }

        if config is None:
            return default

        mapping = _to_mapping(
            _first_non_none(
                config,
                ("mutation_weights",),
                ("discovery", "mutation_weights"),
                ("search", "mutation_weights"),
                default={},
            )
        )

        if not mapping:
            return default

        result = dict(default)
        for key, value in mapping.items():
            key_name = str(key).strip().lower()
            if key_name in result:
                result[key_name] = max(0.0, _coerce_float(value, 1.0))

        return result

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

    # ==================================================
    # PYTHON PROTOCOL
    # ==================================================

    def __repr__(self) -> str:
        return (
            "DiscoveryHeuristics("
            f"max_conditions={self._max_conditions}, "
            f"max_depth={self._max_depth}, "
            f"families={self._family_manager.family_count}"
            ")"
        )