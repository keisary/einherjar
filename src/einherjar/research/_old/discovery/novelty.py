"""
==========================================================
Novelty Engine
==========================================================

Mesure la nouveauté d'une hypothèse pendant la phase
Discovery.

La nouveauté ne désigne pas la diversité globale du pool,
mais le fait qu'une hypothèse apporte quelque chose de peu
vu, peu fréquent, ou structurellement différent par rapport
à l'historique observé.

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
from models.fingerprint import fingerprint_model
from models.hypothesis import Hypothesis

from .family_manager import FamilyManager


__all__ = [
    "NoveltyAssessment",
    "NoveltyEngine",
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


def _operator_key(condition: Condition) -> str:
    return condition.operator.value


def _operand_kind(condition: Condition) -> str:
    return "feature" if isinstance(condition.right, Feature) else "constant"


def _operand_signature(condition: Condition) -> str:
    if isinstance(condition.right, Feature):
        return f"feature:{condition.right.column_index}"
    return f"constant:{repr(condition.right)}"


def _condition_signature(condition: Condition) -> tuple[Any, ...]:
    return (
        condition.left.column_index,
        condition.operator.value,
        _operand_kind(condition),
        _operand_signature(condition),
    )


def _unique_values(values: Iterable[Any]) -> tuple[Any, ...]:
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

    if feature.value_type.value == "boolean":
        return "binary"

    if feature.feature_type.value == "pattern":
        return "binary"

    if name.endswith("_signal"):
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


# ==========================================================
# ASSESSMENT
# ==========================================================

@dataclass(frozen=True, slots=True)
class NoveltyAssessment:
    """
    Diagnostic complet de la nouveauté d'une hypothèse.
    """

    fingerprint: str

    score: float

    exact_novelty: float
    structural_novelty: float
    family_novelty: float
    feature_novelty: float
    operator_novelty: float
    lineage_novelty: float
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
        object.__setattr__(self, "exact_novelty", float(self.exact_novelty))
        object.__setattr__(self, "structural_novelty", float(self.structural_novelty))
        object.__setattr__(self, "family_novelty", float(self.family_novelty))
        object.__setattr__(self, "feature_novelty", float(self.feature_novelty))
        object.__setattr__(self, "operator_novelty", float(self.operator_novelty))
        object.__setattr__(self, "lineage_novelty", float(self.lineage_novelty))
        object.__setattr__(self, "complexity_penalty", float(self.complexity_penalty))
        object.__setattr__(self, "seen_count", max(0, _coerce_int(self.seen_count, 0)))
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
            "exact_novelty": self.exact_novelty,
            "structural_novelty": self.structural_novelty,
            "family_novelty": self.family_novelty,
            "feature_novelty": self.feature_novelty,
            "operator_novelty": self.operator_novelty,
            "lineage_novelty": self.lineage_novelty,
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
    def from_dict(cls, data: Mapping[str, Any]) -> "NoveltyAssessment":
        return cls(
            fingerprint=data["fingerprint"],
            score=_coerce_float(data.get("score"), 0.0),
            exact_novelty=_coerce_float(data.get("exact_novelty"), 0.0),
            structural_novelty=_coerce_float(data.get("structural_novelty"), 0.0),
            family_novelty=_coerce_float(data.get("family_novelty"), 0.0),
            feature_novelty=_coerce_float(data.get("feature_novelty"), 0.0),
            operator_novelty=_coerce_float(data.get("operator_novelty"), 0.0),
            lineage_novelty=_coerce_float(data.get("lineage_novelty"), 0.0),
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
            "NoveltyAssessment("
            f"score={self.score:.4f}, "
            f"seen={self.seen_count}, "
            f"duplicate={self.duplicate}, "
            f"conditions={self.condition_count}"
            ")"
        )


# ==========================================================
# NOVELTY ENGINE
# ==========================================================

class NoveltyEngine:
    """
    Mesure et mémorise la nouveauté des hypothèses.

    La nouveauté est calculée à partir de :
    - l'unicité exacte de la structure,
    - la rareté des familles utilisées,
    - la rareté des features utilisées,
    - la rareté des opérateurs,
    - la filiation entre versions,
    - une pénalité légère sur la complexité.
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
    ) -> "NoveltyEngine":
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
    ) -> "NoveltyEngine":
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
    ) -> NoveltyAssessment:
        assessment = self.assess(
            hypothesis,
            parent_fingerprint=parent_fingerprint,
            depth=depth,
            metadata=metadata,
        )
        self.record_assessment(assessment, hypothesis=hypothesis, depth=depth, parent_fingerprint=parent_fingerprint)
        return assessment

    def record_assessment(
        self,
        assessment: NoveltyAssessment,
        *,
        hypothesis: Hypothesis | None = None,
        depth: int | None = None,
        parent_fingerprint: str | None = None,
    ) -> None:
        if not isinstance(assessment, NoveltyAssessment):
            raise TypeError("assessment must be a NoveltyAssessment.")

        self._fingerprint_counts[assessment.fingerprint] += 1
        self._depth_counts[assessment.depth if depth is None else max(0, _coerce_int(depth, 0))] += 1

        self._parent_counts[assessment.parent_fingerprint or parent_fingerprint or ""] += 1

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

        if not fingerprint:
            hyp = mapping.get("hypothesis")
            if hyp is not None:
                fingerprint = fingerprint_model(hyp) if hasattr(hyp, "to_dict") else repr(hyp)

        if not fingerprint:
            return

        parent_fingerprint = mapping.get("parent_fingerprint")
        family = mapping.get("family", "unknown")
        depth = _coerce_int(mapping.get("depth"), 0)
        condition_count = _coerce_int(mapping.get("condition_count"), 0)

        self._fingerprint_counts[fingerprint] += 1
        self._depth_counts[depth] += 1
        self._parent_counts[str(parent_fingerprint or "")] += 1

        if family is not None:
            self._family_counts[_family_key(family)] += 1

        action = str(mapping.get("action", "")).strip().lower()
        if action:
            self._operator_counts[action] += 1

        result = mapping.get("result")
        if isinstance(result, Mapping):
            hypothesis = result.get("hypothesis")
            if hasattr(hypothesis, "conditions"):
                for condition in hypothesis.conditions:
                    self._feature_counts[condition.left.column_index] += 1
                    self._family_counts[condition.left.economic_family.value] += 1
                    self._operator_counts[condition.operator.value] += 1

        if condition_count > 0:
            self._condition_counts[fingerprint] += condition_count

        related = mapping.get("related_fingerprints")
        if related:
            related_set = self._related.setdefault(fingerprint, set())
            for item in _normalize_items(related):
                related_set.add(str(item))
                self._related.setdefault(str(item), set()).add(fingerprint)

    def register(
        self,
        hypothesis: Hypothesis,
        *,
        parent_fingerprint: str | None = None,
        depth: int = 0,
        metadata: Mapping[str, Any] | None = None,
    ) -> NoveltyAssessment:
        return self.observe(
            hypothesis,
            parent_fingerprint=parent_fingerprint,
            depth=depth,
            metadata=metadata,
        )

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
    ) -> NoveltyAssessment:
        if not isinstance(hypothesis, Hypothesis):
            raise TypeError("hypothesis must be a Hypothesis.")

        fingerprint = fingerprint_model(hypothesis)
        seen_count = self._fingerprint_counts.get(fingerprint, 0)
        duplicate = seen_count > 0

        features = tuple(condition.left.column_index for condition in hypothesis.conditions)
        families = tuple(condition.left.economic_family.value for condition in hypothesis.conditions)
        operators = tuple(condition.operator.value for condition in hypothesis.conditions)

        exact_novelty = 1.0 / (1.0 + float(seen_count))

        feature_novelty = self._feature_novelty(hypothesis)
        family_novelty = self._family_novelty(hypothesis)
        operator_novelty = self._operator_novelty(hypothesis)
        structural_novelty = self._structural_novelty(hypothesis)
        lineage_novelty = self._lineage_novelty(parent_fingerprint)

        complexity_penalty = self._complexity_penalty(hypothesis)

        score = self._combine_scores(
            exact_novelty=exact_novelty,
            structural_novelty=structural_novelty,
            family_novelty=family_novelty,
            feature_novelty=feature_novelty,
            operator_novelty=operator_novelty,
            lineage_novelty=lineage_novelty,
            complexity_penalty=complexity_penalty,
        )

        related_fingerprints = tuple(sorted(self._related.get(fingerprint, set())))

        return NoveltyAssessment(
            fingerprint=fingerprint,
            score=score,
            exact_novelty=exact_novelty,
            structural_novelty=structural_novelty,
            family_novelty=family_novelty,
            feature_novelty=feature_novelty,
            operator_novelty=operator_novelty,
            lineage_novelty=lineage_novelty,
            complexity_penalty=complexity_penalty,
            seen_count=seen_count,
            duplicate=duplicate,
            depth=max(0, _coerce_int(depth, 0)),
            condition_count=len(hypothesis.conditions),
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

    def is_novel(
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

    def novelty_of(
        self,
        hypothesis: Hypothesis,
        *,
        parent_fingerprint: str | None = None,
        depth: int = 0,
    ) -> NoveltyAssessment:
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

        feature_sim = self._jaccard(left_features, right_features)
        family_sim = self._jaccard(left_families, right_families)
        operator_sim = self._jaccard(left_operators, right_operators)
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
    ) -> tuple[tuple[Hypothesis, NoveltyAssessment], ...]:
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

    def _feature_novelty(self, hypothesis: Hypothesis) -> float:
        if not hypothesis.conditions:
            return 0.0

        values: list[float] = []
        for condition in hypothesis.conditions:
            feature = condition.left
            seen = self._feature_counts.get(feature.column_index, 0)
            if hasattr(self._family_manager, "feature_usage"):
                seen += _coerce_int(self._family_manager.feature_usage.get(feature.column_index, 0), 0)
            values.append(1.0 / (1.0 + float(seen)))

        return sum(values) / len(values)

    def _family_novelty(self, hypothesis: Hypothesis) -> float:
        if not hypothesis.conditions:
            return 0.0

        values: list[float] = []
        for condition in hypothesis.conditions:
            family = condition.left.economic_family.value
            seen = self._family_counts.get(family, 0)
            if hasattr(self._family_manager, "family_usage"):
                seen += _coerce_int(self._family_manager.family_usage.get(family, 0), 0)
            values.append(1.0 / (1.0 + float(seen)))

        return sum(values) / len(values)

    def _operator_novelty(self, hypothesis: Hypothesis) -> float:
        if not hypothesis.conditions:
            return 0.0

        values: list[float] = []
        for condition in hypothesis.conditions:
            seen = self._operator_counts.get(condition.operator.value, 0)
            values.append(1.0 / (1.0 + float(seen)))

        return sum(values) / len(values)

    def _structural_novelty(self, hypothesis: Hypothesis) -> float:
        if not hypothesis.conditions:
            return 0.0

        signatures = [_condition_signature(condition) for condition in hypothesis.conditions]
        unique_signatures = len(set(signatures))
        unique_features = len({condition.left.column_index for condition in hypothesis.conditions})
        unique_families = len({condition.left.economic_family.value for condition in hypothesis.conditions})
        unique_operators = len({condition.operator.value for condition in hypothesis.conditions})

        repetition_penalty = (
            max(0, len(hypothesis.conditions) - unique_signatures)
            + max(0, len(hypothesis.conditions) - unique_features)
            + max(0, len(hypothesis.conditions) - unique_families)
            + max(0, len(hypothesis.conditions) - unique_operators)
        )

        coverage = (
            0.35 * (unique_features / max(1.0, float(len(hypothesis.conditions))))
            + 0.30 * (unique_families / max(1.0, float(len(hypothesis.conditions))))
            + 0.20 * (unique_operators / max(1.0, float(len(hypothesis.conditions))))
            + 0.15 * (unique_signatures / max(1.0, float(len(hypothesis.conditions))))
        )

        return max(
            0.0,
            min(
                1.0,
                0.50 * coverage + 0.50 * (1.0 / (1.0 + float(repetition_penalty))),
            ),
        )

    def _lineage_novelty(self, parent_fingerprint: str | None) -> float:
        if not parent_fingerprint:
            return 0.75

        seen = self._fingerprint_counts.get(parent_fingerprint, 0)
        if seen <= 0:
            return 0.90

        return 1.0 / (1.0 + float(seen))

    def _complexity_penalty(self, hypothesis: Hypothesis) -> float:
        if not hypothesis.conditions:
            return 1.0

        size = len(hypothesis.conditions)
        repeated_families = size - len({condition.left.economic_family.value for condition in hypothesis.conditions})
        repeated_features = size - len({condition.left.column_index for condition in hypothesis.conditions})
        repeated_operators = size - len({condition.operator.value for condition in hypothesis.conditions})

        penalty = 1.0 / (
            1.0
            + 0.08 * max(0, size - 1)
            + 0.12 * max(0, repeated_families)
            + 0.10 * max(0, repeated_features)
            + 0.06 * max(0, repeated_operators)
        )
        return max(0.25, min(1.0, penalty))

    def _combine_scores(
        self,
        *,
        exact_novelty: float,
        structural_novelty: float,
        family_novelty: float,
        feature_novelty: float,
        operator_novelty: float,
        lineage_novelty: float,
        complexity_penalty: float,
    ) -> float:
        weights = {
            "exact": max(0.0, self._search_config.novelty_weight) + max(0.0, self._scoring_config.novelty),
            "structural": max(0.0, self._search_config.diversity_weight) + max(0.0, self._scoring_config.diversity),
            "family": max(0.0, self._search_config.family_balance_weight),
            "feature": max(0.0, self._scoring_config.novelty),
            "operator": max(0.0, self._scoring_config.diversity * 0.5),
            "lineage": max(0.0, self._scoring_config.persistence),
        }

        components = {
            "exact": exact_novelty,
            "structural": structural_novelty,
            "family": family_novelty,
            "feature": feature_novelty,
            "operator": operator_novelty,
            "lineage": lineage_novelty,
        }

        numerator = sum(weights[name] * components[name] for name in components)
        denominator = sum(weights.values())

        if denominator <= 0.0:
            return max(0.0, min(1.0, exact_novelty * complexity_penalty))

        base = numerator / denominator
        return max(0.0, min(1.0, base * complexity_penalty))

    # ==================================================
    # INTERNAL REGISTRATION
    # ==================================================

    def _register_condition_signatures(self, hypothesis: Hypothesis) -> None:
        for condition in hypothesis.conditions:
            signature = repr(_condition_signature(condition))
            self._condition_counts[signature] += 1

    # ==================================================
    # INTERNAL COMPARISON
    # ==================================================

    def _jaccard(
        self,
        left: set[Any],
        right: set[Any],
    ) -> float:
        if not left and not right:
            return 1.0

        union = left | right
        if not union:
            return 1.0

        intersection = left & right
        return len(intersection) / len(union)

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
            "NoveltyEngine("
            f"seen={self.total_seen}, "
            f"features={len(self._feature_counts)}, "
            f"families={len(self._family_counts)}"
            ")"
        )


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