"""
==========================================================
Diversity Engine
==========================================================

Mesure la diversité d'une hypothèse pendant la phase
Discovery.

La diversité ne désigne pas la rareté brute d'une
hypothèse, mais sa capacité à couvrir plusieurs familles,
à éviter les répétitions internes et à enrichir la
population de recherche avec une structure différente.

Ce module :
- n'évalue pas la rentabilité,
- ne valide rien statistiquement,
- ne simule rien,
- ne modifie pas les modèles métier.

Il sert à guider Explorer, Generator et les heuristiques
de recherche.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime, timezone
from math import log
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
from models.fingerprint import fingerprint_model
from models.hypothesis import Hypothesis

from .family_manager import FamilyManager


__all__ = [
    "DiversityAssessment",
    "DiversityEngine",
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


def _family_key(family: EconomicFamily | str | None) -> str:
    if family is None:
        return "unknown"

    if isinstance(family, EconomicFamily):
        return family.value

    return str(family).strip().lower() or "unknown"


def _normalize_items(values: Any) -> tuple[Any, ...]:
    if values is None:
        return ()

    if isinstance(values, tuple):
        return values

    if isinstance(values, list):
        return tuple(values)

    if isinstance(values, set):
        return tuple(values)

    return (values,)


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

    name = feature.name.lower()
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

    if feature.value_type.value in {"float", "integer", "ordinal"}:
        return "generic_numeric"

    return "unsupported"


def _count_entropy(counts: Iterable[int]) -> float:
    values = [max(0, int(v)) for v in counts if int(v) > 0]
    total = sum(values)

    if total <= 0:
        return 0.0

    if len(values) <= 1:
        return 0.0

    entropy = 0.0
    for value in values:
        p = value / total
        entropy -= p * log(p)

    max_entropy = log(len(values))
    if max_entropy <= 0:
        return 0.0

    return max(0.0, min(1.0, entropy / max_entropy))


def _jaccard(left: set[Any], right: set[Any]) -> float:
    if not left and not right:
        return 1.0

    union = left | right
    if not union:
        return 1.0

    return len(left & right) / len(union)


def _condition_signature(condition: Condition) -> tuple[Any, ...]:
    return (
        condition.left.column_index,
        condition.operator.value,
        "feature" if isinstance(condition.right, Feature) else "constant",
        condition.right.column_index if isinstance(condition.right, Feature) else repr(condition.right),
    )


def _unique_by_repr(values: Iterable[Any]) -> tuple[Any, ...]:
    seen: set[str] = set()
    result: list[Any] = []

    for value in values:
        key = repr(value)
        if key in seen:
            continue
        seen.add(key)
        result.append(value)

    return tuple(result)


# ==========================================================
# ASSESSMENT
# ==========================================================

@dataclass(frozen=True, slots=True)
class DiversityAssessment:
    """
    Diagnostic complet de la diversité d'une hypothèse.
    """

    fingerprint: str
    score: float

    family_diversity: float
    feature_diversity: float
    operator_diversity: float
    structural_diversity: float
    population_rarity: float
    balance: float
    coverage: float
    repetition_penalty: float
    complexity_penalty: float

    seen_count: int = 0
    duplicate: bool = False

    depth: int = 0
    condition_count: int = 0

    families: tuple[str, ...] = ()
    features: tuple[int, ...] = ()
    operators: tuple[str, ...] = ()

    parent_fingerprint: str | None = None
    related_fingerprints: tuple[str, ...] = ()

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "score", float(self.score))
        object.__setattr__(self, "family_diversity", float(self.family_diversity))
        object.__setattr__(self, "feature_diversity", float(self.feature_diversity))
        object.__setattr__(self, "operator_diversity", float(self.operator_diversity))
        object.__setattr__(self, "structural_diversity", float(self.structural_diversity))
        object.__setattr__(self, "population_rarity", float(self.population_rarity))
        object.__setattr__(self, "balance", float(self.balance))
        object.__setattr__(self, "coverage", float(self.coverage))
        object.__setattr__(self, "repetition_penalty", float(self.repetition_penalty))
        object.__setattr__(self, "complexity_penalty", float(self.complexity_penalty))
        object.__setattr__(self, "seen_count", max(0, _coerce_int(self.seen_count, 0)))
        object.__setattr__(self, "duplicate", _coerce_bool(self.duplicate, False))
        object.__setattr__(self, "depth", max(0, _coerce_int(self.depth, 0)))
        object.__setattr__(self, "condition_count", max(0, _coerce_int(self.condition_count, 0)))
        object.__setattr__(self, "families", tuple(self.families))
        object.__setattr__(self, "features", tuple(self.features))
        object.__setattr__(self, "operators", tuple(self.operators))
        object.__setattr__(self, "related_fingerprints", tuple(self.related_fingerprints))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "fingerprint": self.fingerprint,
            "score": self.score,
            "family_diversity": self.family_diversity,
            "feature_diversity": self.feature_diversity,
            "operator_diversity": self.operator_diversity,
            "structural_diversity": self.structural_diversity,
            "population_rarity": self.population_rarity,
            "balance": self.balance,
            "coverage": self.coverage,
            "repetition_penalty": self.repetition_penalty,
            "complexity_penalty": self.complexity_penalty,
            "seen_count": self.seen_count,
            "duplicate": self.duplicate,
            "depth": self.depth,
            "condition_count": self.condition_count,
            "families": list(self.families),
            "features": list(self.features),
            "operators": list(self.operators),
            "parent_fingerprint": self.parent_fingerprint,
            "related_fingerprints": list(self.related_fingerprints),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DiversityAssessment":
        return cls(
            fingerprint=data["fingerprint"],
            score=_coerce_float(data.get("score"), 0.0),
            family_diversity=_coerce_float(data.get("family_diversity"), 0.0),
            feature_diversity=_coerce_float(data.get("feature_diversity"), 0.0),
            operator_diversity=_coerce_float(data.get("operator_diversity"), 0.0),
            structural_diversity=_coerce_float(data.get("structural_diversity"), 0.0),
            population_rarity=_coerce_float(data.get("population_rarity"), 0.0),
            balance=_coerce_float(data.get("balance"), 0.0),
            coverage=_coerce_float(data.get("coverage"), 0.0),
            repetition_penalty=_coerce_float(data.get("repetition_penalty"), 1.0),
            complexity_penalty=_coerce_float(data.get("complexity_penalty"), 1.0),
            seen_count=_coerce_int(data.get("seen_count"), 0),
            duplicate=_coerce_bool(data.get("duplicate"), False),
            depth=_coerce_int(data.get("depth"), 0),
            condition_count=_coerce_int(data.get("condition_count"), 0),
            families=tuple(data.get("families", ())),
            features=tuple(int(item) for item in data.get("features", ())),
            operators=tuple(data.get("operators", ())),
            parent_fingerprint=data.get("parent_fingerprint"),
            related_fingerprints=tuple(data.get("related_fingerprints", ())),
            metadata=_to_mapping(data.get("metadata", {})),
        )

    def __repr__(self) -> str:
        return (
            "DiversityAssessment("
            f"score={self.score:.4f}, "
            f"seen={self.seen_count}, "
            f"duplicate={self.duplicate}, "
            f"conditions={self.condition_count}"
            ")"
        )


# ==========================================================
# DIVERSITY ENGINE
# ==========================================================

class DiversityEngine:
    """
    Mesure et mémorise la diversité des hypothèses.

    Contrairement à NoveltyEngine, ce composant s'intéresse
    d'abord à la répartition structurelle :
    - combien de familles sont couvertes,
    - si les conditions sont équilibrées,
    - si les features sont variées,
    - si les opérateurs sont diversifiés,
    - si la composition d'ensemble évite les répétitions.
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
                search_config = _build_search_config(
                    _first_non_none(
                        config,
                        ("search",),
                        ("search_config",),
                        ("discovery", "search"),
                        default=config,
                    )
                )
            if scoring_config is None:
                scoring_config = _build_scoring_config(
                    _first_non_none(
                        config,
                        ("scoring",),
                        ("scoring_config",),
                        ("discovery", "scoring"),
                        default=config,
                    )
                )

        self._family_manager = family_manager
        self._search_config = search_config or SearchConfig()
        self._scoring_config = scoring_config or ScoringConfig()
        self._config = config

        if isinstance(rng, np.random.Generator):
            self._rng = rng
        elif rng is not None:
            self._rng = np.random.default_rng(rng)
        else:
            self._rng = np.random.default_rng(self._search_config.random_seed)

        self._fingerprint_counts: Counter[str] = Counter()
        self._condition_counts: Counter[str] = Counter()
        self._feature_counts: Counter[int] = Counter()
        self._family_counts: Counter[str] = Counter()
        self._operator_counts: Counter[str] = Counter()
        self._depth_counts: Counter[int] = Counter()
        self._parent_counts: Counter[str] = Counter()
        self._related: dict[str, set[str]] = {}

    # ==================================================
    # CONSTRUCTION FROM CONFIG
    # ==================================================

    @classmethod
    def from_config(
        cls,
        config: Any,
        family_manager: FamilyManager,
        *,
        rng: np.random.Generator | int | None = None,
    ) -> "DiversityEngine":
        return cls(
            family_manager=family_manager,
            config=config,
            rng=rng,
        )

    @classmethod
    def from_report(
        cls,
        report: Any,
        family_manager: FamilyManager,
        *,
        config: Any | None = None,
        search_config: SearchConfig | None = None,
        scoring_config: ScoringConfig | None = None,
        rng: np.random.Generator | int | None = None,
    ) -> "DiversityEngine":
        engine = cls(
            family_manager=family_manager,
            search_config=search_config,
            scoring_config=scoring_config,
            config=config,
            rng=rng,
        )
        engine.observe_report(report)
        return engine

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
    def fingerprint_counts(self) -> dict[str, int]:
        return dict(self._fingerprint_counts)

    @property
    def feature_counts(self) -> dict[int, int]:
        return dict(self._feature_counts)

    @property
    def family_counts(self) -> dict[str, int]:
        return dict(self._family_counts)

    @property
    def operator_counts(self) -> dict[str, int]:
        return dict(self._operator_counts)

    @property
    def depth_counts(self) -> dict[int, int]:
        return dict(self._depth_counts)

    @property
    def total_seen(self) -> int:
        return sum(self._fingerprint_counts.values())

    @property
    def coverage(self) -> float:
        families = len(self._family_counts)
        features = len(self._feature_counts)
        operators = len(self._operator_counts)
        if families + features + operators == 0:
            return 0.0
        return min(1.0, (families + features + operators) / 100.0)

    # ==================================================
    # OBSERVATION
    # ==================================================

    def observe(
        self,
        hypothesis: Hypothesis,
        *,
        parent_fingerprint: str | None = None,
        depth: int = 0,
        metadata: Mapping[str, Any] | None = None,
    ) -> DiversityAssessment:
        assessment = self.assess(
            hypothesis,
            parent_fingerprint=parent_fingerprint,
            depth=depth,
            metadata=metadata,
        )
        self.record_assessment(
            assessment,
            hypothesis=hypothesis,
            depth=depth,
            parent_fingerprint=parent_fingerprint,
        )
        return assessment

    def observe_report(self, report: Any) -> None:
        if report is None:
            return

        events = getattr(report, "events", None)
        if events is None and isinstance(report, Mapping):
            events = report.get("events")

        if not events:
            return

        for event in events:
            self.observe_event(event)

    def observe_event(self, event: Any) -> None:
        if event is None:
            return

        mapping = _to_mapping(event)
        fingerprint = str(
            mapping.get("fingerprint")
            or mapping.get("hypothesis_fingerprint")
            or mapping.get("result_fingerprint")
            or ""
        ).strip()

        result = mapping.get("result")
        if not fingerprint and isinstance(result, Mapping):
            hyp = result.get("hypothesis")
            if hasattr(hyp, "to_dict"):
                fingerprint = fingerprint_model(hyp)
            else:
                fingerprint = str(result.get("fingerprint") or "")

        if not fingerprint:
            return

        parent_fingerprint = mapping.get("parent_fingerprint")
        family = mapping.get("family", "unknown")
        depth = _coerce_int(mapping.get("depth"), 0)

        self._fingerprint_counts[fingerprint] += 1
        self._depth_counts[depth] += 1
        self._parent_counts[str(parent_fingerprint or "")] += 1

        if family is not None:
            self._family_counts[_family_key(family)] += 1

        if isinstance(result, Mapping):
            hyp = result.get("hypothesis")
            if isinstance(hyp, Hypothesis):
                self.record_hypothesis(hyp, count=1)

    def register(
        self,
        hypothesis: Hypothesis,
        *,
        parent_fingerprint: str | None = None,
        depth: int = 0,
        metadata: Mapping[str, Any] | None = None,
    ) -> DiversityAssessment:
        return self.observe(
            hypothesis,
            parent_fingerprint=parent_fingerprint,
            depth=depth,
            metadata=metadata,
        )

    def record_assessment(
        self,
        assessment: DiversityAssessment,
        *,
        hypothesis: Hypothesis | None = None,
        depth: int | None = None,
        parent_fingerprint: str | None = None,
    ) -> None:
        if not isinstance(assessment, DiversityAssessment):
            raise TypeError("assessment must be a DiversityAssessment.")

        self._fingerprint_counts[assessment.fingerprint] += 1
        self._depth_counts[assessment.depth if depth is None else max(0, _coerce_int(depth, 0))] += 1

        if assessment.parent_fingerprint or parent_fingerprint:
            self._parent_counts[str(assessment.parent_fingerprint or parent_fingerprint or "")] += 1

        for feature_index in assessment.features:
            self._feature_counts[int(feature_index)] += 1

        for family in assessment.families:
            self._family_counts[str(family)] += 1

        for operator in assessment.operators:
            self._operator_counts[str(operator)] += 1

        if assessment.parent_fingerprint:
            self._related.setdefault(assessment.fingerprint, set()).add(assessment.parent_fingerprint)
            self._related.setdefault(assessment.parent_fingerprint, set()).add(assessment.fingerprint)

        if hypothesis is not None:
            self._register_condition_signatures(hypothesis)

    def record_hypothesis(
        self,
        hypothesis: Hypothesis,
        *,
        count: int = 1,
    ) -> None:
        if not isinstance(hypothesis, Hypothesis):
            raise TypeError("hypothesis must be a Hypothesis.")

        count = max(1, _coerce_int(count, 1))

        for condition in hypothesis.conditions:
            self._feature_counts[condition.left.column_index] += count
            self._family_counts[condition.left.economic_family.value] += count
            self._operator_counts[condition.operator.value] += count
            self._condition_counts[repr(_condition_signature(condition))] += count

    def record_condition(
        self,
        condition: Condition,
        *,
        count: int = 1,
    ) -> None:
        if not isinstance(condition, Condition):
            raise TypeError("condition must be a Condition.")

        count = max(1, _coerce_int(count, 1))
        self._feature_counts[condition.left.column_index] += count
        self._family_counts[condition.left.economic_family.value] += count
        self._operator_counts[condition.operator.value] += count
        self._condition_counts[repr(_condition_signature(condition))] += count

    def reset(self) -> None:
        self._fingerprint_counts.clear()
        self._condition_counts.clear()
        self._feature_counts.clear()
        self._family_counts.clear()
        self._operator_counts.clear()
        self._depth_counts.clear()
        self._parent_counts.clear()
        self._related.clear()

    # ==================================================
    # ASSESSMENT
    # ==================================================

    def assess(
        self,
        hypothesis: Hypothesis,
        *,
        parent_fingerprint: str | None = None,
        depth: int = 0,
        metadata: Mapping[str, Any] | None = None,
    ) -> DiversityAssessment:
        if not isinstance(hypothesis, Hypothesis):
            raise TypeError("hypothesis must be a Hypothesis.")

        fingerprint = fingerprint_model(hypothesis)
        seen_count = self._fingerprint_counts.get(fingerprint, 0)
        duplicate = seen_count > 0

        families = tuple(
            condition.left.economic_family.value
            for condition in hypothesis.conditions
        )
        features = tuple(
            condition.left.column_index
            for condition in hypothesis.conditions
        )
        operators = tuple(
            condition.operator.value
            for condition in hypothesis.conditions
        )

        condition_count = len(hypothesis.conditions)

        family_diversity = self._normalized_entropy(families)
        feature_diversity = self._normalized_entropy(features)
        operator_diversity = self._normalized_entropy(operators)

        balance = self._family_balance(hypothesis)
        coverage = self._local_coverage(hypothesis)

        structural_diversity = self._structural_diversity(
            family_diversity=family_diversity,
            feature_diversity=feature_diversity,
            operator_diversity=operator_diversity,
            balance=balance,
            coverage=coverage,
        )

        population_rarity = self._population_rarity(hypothesis)
        repetition_penalty = self._repetition_penalty(hypothesis)
        complexity_penalty = self._complexity_penalty(hypothesis)

        score = self._combine_scores(
            family_diversity=family_diversity,
            feature_diversity=feature_diversity,
            operator_diversity=operator_diversity,
            structural_diversity=structural_diversity,
            population_rarity=population_rarity,
            balance=balance,
            coverage=coverage,
            repetition_penalty=repetition_penalty,
            complexity_penalty=complexity_penalty,
        )

        related_fingerprints = tuple(sorted(self._related.get(fingerprint, set())))

        return DiversityAssessment(
            fingerprint=fingerprint,
            score=score,
            family_diversity=family_diversity,
            feature_diversity=feature_diversity,
            operator_diversity=operator_diversity,
            structural_diversity=structural_diversity,
            population_rarity=population_rarity,
            balance=balance,
            coverage=coverage,
            repetition_penalty=repetition_penalty,
            complexity_penalty=complexity_penalty,
            seen_count=seen_count,
            duplicate=duplicate,
            depth=max(0, _coerce_int(depth, 0)),
            condition_count=condition_count,
            families=families,
            features=features,
            operators=operators,
            parent_fingerprint=parent_fingerprint,
            related_fingerprints=related_fingerprints,
            metadata=dict(metadata or {}),
        )

    def score(
        self,
        hypothesis: Hypothesis,
        *,
        parent_fingerprint: str | None = None,
        depth: int = 0,
        metadata: Mapping[str, Any] | None = None,
    ) -> float:
        return self.assess(
            hypothesis,
            parent_fingerprint=parent_fingerprint,
            depth=depth,
            metadata=metadata,
        ).score

    def is_diverse(
        self,
        hypothesis: Hypothesis,
        *,
        threshold: float = 0.5,
        parent_fingerprint: str | None = None,
        depth: int = 0,
    ) -> bool:
        assessment = self.assess(
            hypothesis,
            parent_fingerprint=parent_fingerprint,
            depth=depth,
        )
        return assessment.score >= threshold and not assessment.duplicate

    def diversity_of(
        self,
        hypothesis: Hypothesis,
        *,
        parent_fingerprint: str | None = None,
        depth: int = 0,
    ) -> DiversityAssessment:
        return self.assess(
            hypothesis,
            parent_fingerprint=parent_fingerprint,
            depth=depth,
        )

    # ==================================================
    # COMPARISON
    # ==================================================

    def similarity(
        self,
        left: Hypothesis,
        right: Hypothesis,
    ) -> float:
        return 1.0 - self.distance(left, right)

    def distance(
        self,
        left: Hypothesis,
        right: Hypothesis,
    ) -> float:
        if not isinstance(left, Hypothesis) or not isinstance(right, Hypothesis):
            raise TypeError("left and right must be Hypothesis instances.")

        left_features = {condition.left.column_index for condition in left.conditions}
        right_features = {condition.left.column_index for condition in right.conditions}
        left_families = {condition.left.economic_family.value for condition in left.conditions}
        right_families = {condition.left.economic_family.value for condition in right.conditions}
        left_operators = {condition.operator.value for condition in left.conditions}
        right_operators = {condition.operator.value for condition in right.conditions}

        feature_sim = _jaccard(left_features, right_features)
        family_sim = _jaccard(left_families, right_families)
        operator_sim = _jaccard(left_operators, right_operators)
        size_sim = 1.0 / (1.0 + abs(len(left.conditions) - len(right.conditions)))

        similarity = (
            0.35 * feature_sim
            + 0.25 * family_sim
            + 0.20 * operator_sim
            + 0.20 * size_sim
        )

        return max(0.0, min(1.0, 1.0 - similarity))

    def compare(
        self,
        left: Hypothesis,
        right: Hypothesis,
    ) -> dict[str, float]:
        return {
            "similarity": self.similarity(left, right),
            "distance": self.distance(left, right),
        }

    # ==================================================
    # RANKING
    # ==================================================

    def rank(
        self,
        hypotheses: Iterable[Hypothesis],
        *,
        parent_fingerprint: str | None = None,
        depth: int = 0,
    ) -> tuple[tuple[Hypothesis, DiversityAssessment], ...]:
        scored = [
            (
                hypothesis,
                self.assess(
                    hypothesis,
                    parent_fingerprint=parent_fingerprint,
                    depth=depth,
                ),
            )
            for hypothesis in hypotheses
        ]

        scored.sort(key=lambda item: item[1].score, reverse=True)
        return tuple(scored)

    # ==================================================
    # INTERNAL SCORING
    # ==================================================

    def _normalized_entropy(self, values: Iterable[Any]) -> float:
        values = tuple(values)
        if not values:
            return 0.0

        counts = Counter(values)
        if len(counts) <= 1:
            return 0.0

        return _count_entropy(counts.values())

    def _family_balance(self, hypothesis: Hypothesis) -> float:
        if not hypothesis.conditions:
            return 0.0

        counts = Counter(
            condition.left.economic_family.value
            for condition in hypothesis.conditions
        )
        total = sum(counts.values())
        if total <= 0:
            return 0.0

        max_share = max(counts.values()) / total
        return max(0.0, min(1.0, 1.0 - max_share))

    def _local_coverage(self, hypothesis: Hypothesis) -> float:
        if not hypothesis.conditions:
            return 0.0

        total = max(1, len(hypothesis.conditions))
        unique_families = len({condition.left.economic_family.value for condition in hypothesis.conditions})
        unique_features = len({condition.left.column_index for condition in hypothesis.conditions})
        unique_operators = len({condition.operator.value for condition in hypothesis.conditions})

        family_cov = unique_families / total
        feature_cov = unique_features / total
        operator_cov = unique_operators / total

        return max(0.0, min(1.0, (family_cov + feature_cov + operator_cov) / 3.0))

    def _structural_diversity(
        self,
        *,
        family_diversity: float,
        feature_diversity: float,
        operator_diversity: float,
        balance: float,
        coverage: float,
    ) -> float:
        return max(
            0.0,
            min(
                1.0,
                0.28 * family_diversity
                + 0.28 * feature_diversity
                + 0.18 * operator_diversity
                + 0.16 * balance
                + 0.10 * coverage,
            ),
        )

    def _population_rarity(self, hypothesis: Hypothesis) -> float:
        if not hypothesis.conditions:
            return 0.0

        family_values: list[float] = []
        feature_values: list[float] = []
        operator_values: list[float] = []

        for condition in hypothesis.conditions:
            family = condition.left.economic_family.value
            feature = condition.left.column_index
            operator = condition.operator.value

            family_seen = float(self._family_counts.get(family, 0))
            feature_seen = float(self._feature_counts.get(feature, 0))
            operator_seen = float(self._operator_counts.get(operator, 0))

            if hasattr(self._family_manager, "family_usage"):
                family_seen += float(self._family_manager.family_usage.get(family, 0))

            if hasattr(self._family_manager, "feature_usage"):
                feature_seen += float(self._family_manager.feature_usage.get(feature, 0))

            family_values.append(1.0 / (1.0 + family_seen))
            feature_values.append(1.0 / (1.0 + feature_seen))
            operator_values.append(1.0 / (1.0 + operator_seen))

        rarity = (
            0.34 * (sum(family_values) / len(family_values))
            + 0.33 * (sum(feature_values) / len(feature_values))
            + 0.33 * (sum(operator_values) / len(operator_values))
        )
        return max(0.0, min(1.0, rarity))

    def _repetition_penalty(self, hypothesis: Hypothesis) -> float:
        if not hypothesis.conditions:
            return 1.0

        signatures = [_condition_signature(condition) for condition in hypothesis.conditions]
        unique_signatures = len(set(signatures))
        unique_features = len({condition.left.column_index for condition in hypothesis.conditions})
        unique_families = len({condition.left.economic_family.value for condition in hypothesis.conditions})
        unique_operators = len({condition.operator.value for condition in hypothesis.conditions})

        repeated = (
            max(0, len(hypothesis.conditions) - unique_signatures)
            + max(0, len(hypothesis.conditions) - unique_features)
            + max(0, len(hypothesis.conditions) - unique_families)
            + max(0, len(hypothesis.conditions) - unique_operators)
        )

        penalty = 1.0 / (1.0 + 0.18 * repeated)
        return max(0.25, min(1.0, penalty))

    def _complexity_penalty(self, hypothesis: Hypothesis) -> float:
        if not hypothesis.conditions:
            return 1.0

        size = len(hypothesis.conditions)
        repeated_families = size - len({condition.left.economic_family.value for condition in hypothesis.conditions})
        repeated_features = size - len({condition.left.column_index for condition in hypothesis.conditions})

        penalty = 1.0 / (
            1.0
            + 0.06 * max(0, size - 1)
            + 0.10 * max(0, repeated_families)
            + 0.08 * max(0, repeated_features)
        )
        return max(0.30, min(1.0, penalty))

    def _combine_scores(
        self,
        *,
        family_diversity: float,
        feature_diversity: float,
        operator_diversity: float,
        structural_diversity: float,
        population_rarity: float,
        balance: float,
        coverage: float,
        repetition_penalty: float,
        complexity_penalty: float,
    ) -> float:
        weights = self._scoring_config

        base = (
            (weights.diversity + self._search_config.diversity_weight) * structural_diversity
            + 0.18 * family_diversity
            + 0.16 * feature_diversity
            + 0.12 * operator_diversity
            + 0.16 * population_rarity
            + 0.10 * balance
            + 0.08 * coverage
        )

        if weights.novelty > 0:
            base += 0.04 * family_diversity
        if weights.persistence > 0:
            base += 0.02 * repetition_penalty

        denominator = 1.0 + max(0.0, self._search_config.diversity_weight) + max(0.0, weights.diversity)
        score = base / denominator
        score *= repetition_penalty
        score *= complexity_penalty

        return max(0.0, min(1.0, score))

    # ==================================================
    # INTERNAL REGISTRATION
    # ==================================================

    def _register_condition_signatures(self, hypothesis: Hypothesis) -> None:
        for condition in hypothesis.conditions:
            self._condition_counts[repr(_condition_signature(condition))] += 1

    # ==================================================
    # PYTHON PROTOCOL
    # ==================================================

    def __contains__(self, item: Hypothesis | str) -> bool:
        if isinstance(item, str):
            return item in self._fingerprint_counts
        if isinstance(item, Hypothesis):
            return fingerprint_model(item) in self._fingerprint_counts
        return False

    def __len__(self) -> int:
        return self.total_seen

    def __iter__(self):
        return iter(self._fingerprint_counts.items())

    def __repr__(self) -> str:
        return (
            "DiversityEngine("
            f"seen={self.total_seen}, "
            f"features={len(self._feature_counts)}, "
            f"families={len(self._family_counts)}"
            ")"
        )