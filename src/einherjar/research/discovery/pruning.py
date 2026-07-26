"""
==========================================================
Pruning Engine
==========================================================

Orchestre la simplification d'une hypothèse pendant la phase
Discovery.

Le pruning ne valide rien statistiquement. Il ne fait que :
- choisir quelles conditions retirer,
- générer des variantes simplifiées,
- mesurer la nouveauté et la diversité des variantes,
- signaler si une branche mérite de continuer.

Il s'appuie sur :
- DiscoveryGenerator
- DiscoveryHeuristics
- NoveltyEngine
- DiversityEngine
- SearchBudget
- SearchReport
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from dataclasses import field
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

from .diversity import DiversityAssessment
from .diversity import DiversityEngine
from .family_manager import FamilyManager
from .generator import DiscoveryGenerator
from .generator import GenerationResult
from .heuristics import DiscoveryHeuristics
from .novelty import NoveltyAssessment
from .novelty import NoveltyEngine
from .search_budget import BudgetSnapshot
from .search_budget import SearchBudget
from .search_report import SearchReport


__all__ = [
    "PruningSettings",
    "PruningCandidate",
    "PruningBatch",
    "PruningEngine",
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


def _is_binary_feature(feature: Feature) -> bool:
    return _feature_profile(feature) == "binary"


# ==========================================================
# SETTINGS
# ==========================================================

@dataclass(frozen=True, slots=True)
class PruningSettings:
    """
    Paramètres de la phase de pruning.

    Le pruning doit simplifier sans casser la structure
    scientifique de l'hypothèse.
    """

    max_children_per_parent: int = 4
    max_attempts_per_parent: int = 24

    min_conditions: int = 1

    novelty_floor: float = 0.15
    diversity_floor: float = 0.15
    combined_floor: float = 0.20

    prefer_best_prunes: bool = True
    keep_duplicates: bool = False
    record_rejections: bool = True

    action_bias: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "max_children_per_parent",
            max(1, _coerce_int(self.max_children_per_parent, 4)),
        )
        object.__setattr__(
            self,
            "max_attempts_per_parent",
            max(1, _coerce_int(self.max_attempts_per_parent, 24)),
        )
        object.__setattr__(
            self,
            "min_conditions",
            max(1, _coerce_int(self.min_conditions, 1)),
        )
        object.__setattr__(
            self,
            "novelty_floor",
            min(1.0, max(0.0, _coerce_float(self.novelty_floor, 0.15))),
        )
        object.__setattr__(
            self,
            "diversity_floor",
            min(1.0, max(0.0, _coerce_float(self.diversity_floor, 0.15))),
        )
        object.__setattr__(
            self,
            "combined_floor",
            min(1.0, max(0.0, _coerce_float(self.combined_floor, 0.20))),
        )
        object.__setattr__(self, "prefer_best_prunes", _coerce_bool(self.prefer_best_prunes, True))
        object.__setattr__(self, "keep_duplicates", _coerce_bool(self.keep_duplicates, False))
        object.__setattr__(self, "record_rejections", _coerce_bool(self.record_rejections, True))
        object.__setattr__(
            self,
            "action_bias",
            {str(k).strip().lower(): max(0.0, _coerce_float(v, 1.0)) for k, v in dict(self.action_bias).items()},
        )

    @classmethod
    def from_config(cls, config: Any | None) -> "PruningSettings":
        if config is None:
            return cls()

        return cls(
            max_children_per_parent=_coerce_int(
                _first_non_none(
                    config,
                    ("discovery", "max_children_per_parent"),
                    ("pruning", "max_children_per_parent"),
                    ("search", "max_children_per_parent"),
                    default=4,
                ),
                4,
            ),
            max_attempts_per_parent=_coerce_int(
                _first_non_none(
                    config,
                    ("discovery", "max_attempts_per_parent"),
                    ("pruning", "max_attempts_per_parent"),
                    ("search", "max_attempts_per_parent"),
                    default=24,
                ),
                24,
            ),
            min_conditions=_coerce_int(
                _first_non_none(
                    config,
                    ("discovery", "min_conditions"),
                    ("pruning", "min_conditions"),
                    ("search", "min_conditions"),
                    default=1,
                ),
                1,
            ),
            novelty_floor=_coerce_float(
                _first_non_none(
                    config,
                    ("discovery", "novelty_floor"),
                    ("pruning", "novelty_floor"),
                    ("search", "novelty_floor"),
                    default=0.15,
                ),
                0.15,
            ),
            diversity_floor=_coerce_float(
                _first_non_none(
                    config,
                    ("discovery", "diversity_floor"),
                    ("pruning", "diversity_floor"),
                    ("search", "diversity_floor"),
                    default=0.15,
                ),
                0.15,
            ),
            combined_floor=_coerce_float(
                _first_non_none(
                    config,
                    ("discovery", "combined_floor"),
                    ("pruning", "combined_floor"),
                    ("search", "combined_floor"),
                    default=0.20,
                ),
                0.20,
            ),
            prefer_best_prunes=_coerce_bool(
                _first_non_none(
                    config,
                    ("discovery", "prefer_best_prunes"),
                    ("pruning", "prefer_best_prunes"),
                    ("search", "prefer_best_prunes"),
                    default=True,
                ),
                True,
            ),
            keep_duplicates=_coerce_bool(
                _first_non_none(
                    config,
                    ("discovery", "keep_duplicates"),
                    ("pruning", "keep_duplicates"),
                    ("search", "keep_duplicates"),
                    default=False,
                ),
                False,
            ),
            record_rejections=_coerce_bool(
                _first_non_none(
                    config,
                    ("discovery", "record_rejections"),
                    ("pruning", "record_rejections"),
                    ("search", "record_rejections"),
                    default=True,
                ),
                True,
            ),
            action_bias=_to_mapping(
                _first_non_none(
                    config,
                    ("discovery", "action_bias"),
                    ("pruning", "action_bias"),
                    ("search", "action_bias"),
                    default={},
                )
            ),
        )


# ==========================================================
# RESULTS
# ==========================================================

@dataclass(frozen=True, slots=True)
class PruningCandidate:
    """
    Candidat obtenu après suppression d'une condition.
    """

    parent_fingerprint: str
    child_fingerprint: str

    action: str
    generation: GenerationResult

    novelty: NoveltyAssessment | None = None
    diversity: DiversityAssessment | None = None

    combined_score: float = 0.0
    accepted: bool = True
    reason: str = ""

    removed_condition_index: int | None = None
    removed_condition_fingerprint: str | None = None

    family: str | None = None
    depth: int = 0
    condition_count: int = 0

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "action", str(self.action).strip().lower())
        object.__setattr__(self, "combined_score", float(self.combined_score))
        object.__setattr__(self, "accepted", _coerce_bool(self.accepted, True))
        object.__setattr__(self, "reason", str(self.reason))
        object.__setattr__(self, "depth", max(0, _coerce_int(self.depth, 0)))
        object.__setattr__(self, "condition_count", max(0, _coerce_int(self.condition_count, 0)))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def hypothesis(self) -> Hypothesis:
        return self.generation.hypothesis

    def to_dict(self) -> dict[str, Any]:
        return {
            "parent_fingerprint": self.parent_fingerprint,
            "child_fingerprint": self.child_fingerprint,
            "action": self.action,
            "generation": self.generation.to_dict(),
            "novelty": None if self.novelty is None else self.novelty.to_dict(),
            "diversity": None if self.diversity is None else self.diversity.to_dict(),
            "combined_score": self.combined_score,
            "accepted": self.accepted,
            "reason": self.reason,
            "removed_condition_index": self.removed_condition_index,
            "removed_condition_fingerprint": self.removed_condition_fingerprint,
            "family": self.family,
            "depth": self.depth,
            "condition_count": self.condition_count,
            "metadata": dict(self.metadata),
        }

    def __repr__(self) -> str:
        return (
            "PruningCandidate("
            f"action='{self.action}', "
            f"score={self.combined_score:.4f}, "
            f"accepted={self.accepted}, "
            f"removed={self.removed_condition_index}"
            ")"
        )


@dataclass(frozen=True, slots=True)
class PruningBatch:
    """
    Lot de candidats produits pour un même parent.
    """

    parent_fingerprint: str
    depth: int
    candidates: tuple[PruningCandidate, ...]

    exhausted: bool = False
    reason: str = ""

    budget_snapshot: BudgetSnapshot | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "depth", max(0, _coerce_int(self.depth, 0)))
        object.__setattr__(self, "candidates", tuple(self.candidates))
        object.__setattr__(self, "exhausted", _coerce_bool(self.exhausted, False))
        object.__setattr__(self, "reason", str(self.reason))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def accepted(self) -> tuple[PruningCandidate, ...]:
        return tuple(candidate for candidate in self.candidates if candidate.accepted)

    @property
    def rejected(self) -> tuple[PruningCandidate, ...]:
        return tuple(candidate for candidate in self.candidates if not candidate.accepted)

    @property
    def best(self) -> PruningCandidate | None:
        if not self.candidates:
            return None
        return max(self.candidates, key=lambda candidate: candidate.combined_score)

    @property
    def best_hypothesis(self) -> Hypothesis | None:
        best = self.best
        return None if best is None else best.hypothesis

    def to_dict(self) -> dict[str, Any]:
        return {
            "parent_fingerprint": self.parent_fingerprint,
            "depth": self.depth,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "exhausted": self.exhausted,
            "reason": self.reason,
            "budget_snapshot": None if self.budget_snapshot is None else self.budget_snapshot,
            "metadata": dict(self.metadata),
        }

    def __len__(self) -> int:
        return len(self.candidates)

    def __iter__(self):
        return iter(self.candidates)

    def __repr__(self) -> str:
        return (
            "PruningBatch("
            f"depth={self.depth}, "
            f"candidates={len(self.candidates)}, "
            f"exhausted={self.exhausted}"
            ")"
        )


# ==========================================================
# ENGINE
# ==========================================================

class PruningEngine:
    """
    Moteur de pruning des hypothèses.

    Il choisit les conditions les moins utiles, génère des
    variantes simplifiées, puis évalue leur capacité à
    rester novatrices et diverses.
    """

    def __init__(
        self,
        generator: DiscoveryGenerator,
        heuristics: DiscoveryHeuristics,
        *,
        novelty_engine: NoveltyEngine | None = None,
        diversity_engine: DiversityEngine | None = None,
        budget: SearchBudget | None = None,
        report: SearchReport | None = None,
        search_config: SearchConfig | None = None,
        scoring_config: ScoringConfig | None = None,
        config: Any | None = None,
        settings: PruningSettings | None = None,
        rng: np.random.Generator | int | None = None,
    ) -> None:
        if not isinstance(generator, DiscoveryGenerator):
            raise TypeError("generator must be a DiscoveryGenerator.")
        if not isinstance(heuristics, DiscoveryHeuristics):
            raise TypeError("heuristics must be a DiscoveryHeuristics.")

        self._generator = generator
        self._heuristics = heuristics
        self._family_manager = heuristics.family_manager

        if config is not None:
            if search_config is None:
                search_config = _build_search_config(
                    _first_non_none(config, ("search",), ("search_config",), ("discovery", "search"), default=config)
                )
            if scoring_config is None:
                scoring_config = _build_scoring_config(
                    _first_non_none(config, ("scoring",), ("scoring_config",), ("discovery", "scoring"), default=config)
                )

        self._search_config = search_config or heuristics.search_config
        self._scoring_config = scoring_config or heuristics.scoring_config

        self._settings = settings or PruningSettings.from_config(config)

        self._novelty = novelty_engine or NoveltyEngine(
            family_manager=self._family_manager,
            search_config=self._search_config,
            scoring_config=self._scoring_config,
            config=config,
            rng=rng,
        )
        self._diversity = diversity_engine or DiversityEngine(
            family_manager=self._family_manager,
            search_config=self._search_config,
            scoring_config=self._scoring_config,
            config=config,
            rng=rng,
        )

        self._budget = budget
        self._report = report

        if isinstance(rng, np.random.Generator):
            self._rng = rng
        elif rng is not None:
            self._rng = np.random.default_rng(rng)
        else:
            self._rng = np.random.default_rng(self._search_config.random_seed)

        self._action_counts: Counter[str] = Counter()
        self._removed_counts: Counter[str] = Counter()
        self._accepted_counts: Counter[str] = Counter()
        self._rejected_counts: Counter[str] = Counter()

    # ==========================================================
    # PROPERTIES
    # ==========================================================

    @property
    def generator(self) -> DiscoveryGenerator:
        return self._generator

    @property
    def heuristics(self) -> DiscoveryHeuristics:
        return self._heuristics

    @property
    def novelty_engine(self) -> NoveltyEngine:
        return self._novelty

    @property
    def diversity_engine(self) -> DiversityEngine:
        return self._diversity

    @property
    def budget(self) -> SearchBudget | None:
        return self._budget

    @property
    def report(self) -> SearchReport | None:
        return self._report

    @property
    def settings(self) -> PruningSettings:
        return self._settings

    @property
    def action_counts(self) -> dict[str, int]:
        return dict(self._action_counts)

    @property
    def removed_counts(self) -> dict[str, int]:
        return dict(self._removed_counts)

    @property
    def accepted_counts(self) -> dict[str, int]:
        return dict(self._accepted_counts)

    @property
    def rejected_counts(self) -> dict[str, int]:
        return dict(self._rejected_counts)

    # ==========================================================
    # PUBLIC API
    # ==========================================================

    def prune(
        self,
        hypothesis: Hypothesis,
        *,
        depth: int = 0,
        limit: int | None = None,
        available_actions: Iterable[str] | None = None,
    ) -> PruningBatch:
        if not isinstance(hypothesis, Hypothesis):
            raise TypeError("hypothesis must be a Hypothesis.")

        parent_fingerprint = fingerprint_model(hypothesis)

        if len(hypothesis.conditions) <= self._settings.min_conditions:
            batch = PruningBatch(
                parent_fingerprint=parent_fingerprint,
                depth=depth,
                candidates=(),
                exhausted=False,
                reason="minimum_condition_count_reached",
                budget_snapshot=(self._budget.snapshot if self._budget is not None else None),
                metadata={
                    "attempts": 0,
                    "available_actions": list(available_actions or ("prune",)),
                },
            )
            if self._report is not None:
                self._report.record_budget(batch.budget_snapshot) if batch.budget_snapshot is not None else None
            return batch

        chosen_limit = max(1, _coerce_int(limit, self._settings.max_children_per_parent))
        actions = tuple(
            str(action).strip().lower()
            for action in (available_actions if available_actions is not None else ("prune",))
        )
        if "prune" not in actions:
            actions = actions + ("prune",)

        ranked_indices = self._rank_condition_indices(hypothesis)
        candidates: list[PruningCandidate] = []
        attempts = 0
        exhausted = False
        reason: str | None = None

        for idx in ranked_indices:
            if len(candidates) >= chosen_limit:
                break
            if attempts >= self._settings.max_attempts_per_parent:
                reason = "attempt_limit_reached"
                break

            attempts += 1

            if self._budget is not None and not self._budget.can_generate(
                family=hypothesis.conditions[idx].left.economic_family,
                depth=depth + 1,
                condition_count=len(hypothesis.conditions),
                amount=1,
            ):
                exhausted = True
                reason = "budget_exhausted"
                self._budget.exhaust(reason)
                break

            result = self._apply_prune(
                hypothesis,
                condition_index=idx,
                depth=depth,
            )

            if result is None:
                self._rejected_counts["prune"] += 1
                continue

            candidate = self._evaluate_candidate(
                result,
                depth=depth,
                parent_fingerprint=parent_fingerprint,
                removed_index=idx,
            )

            if not candidate.accepted and not self._settings.keep_duplicates and candidate.reason == "duplicate":
                self._rejected_counts["prune"] += 1
                if self._report is not None and self._settings.record_rejections:
                    self._report.record_generation(
                        result,
                        depth=depth + 1,
                        accepted=False,
                        reason=candidate.reason,
                        duplicate=True,
                        budget_exhausted=False,
                        metadata={
                            "pruning": candidate.to_dict(),
                        },
                    )
                continue

            candidates.append(candidate)

            self._action_counts["prune"] += 1
            if candidate.removed_condition_fingerprint is not None:
                self._removed_counts[candidate.removed_condition_fingerprint] += 1

            if candidate.accepted:
                self._accepted_counts["prune"] += 1
            else:
                self._rejected_counts["prune"] += 1

            self._commit(candidate, depth=depth, budget_consumed=True)

        batch = PruningBatch(
            parent_fingerprint=parent_fingerprint,
            depth=depth,
            candidates=tuple(
                sorted(
                    candidates,
                    key=lambda item: (
                        not item.accepted,
                        -item.combined_score,
                        item.child_fingerprint,
                    ),
                )
            ),
            exhausted=exhausted,
            reason=reason or "",
            budget_snapshot=(self._budget.snapshot if self._budget is not None else None),
            metadata={
                "attempts": attempts,
                "available_actions": list(actions),
                "ranked_indices": list(ranked_indices),
            },
        )

        if self._report is not None:
            if batch.budget_snapshot is not None:
                self._report.record_budget(batch.budget_snapshot)

            if batch.candidates:
                for candidate in batch.candidates:
                    self._report.record_generation(
                        candidate.generation,
                        depth=depth + 1,
                        accepted=candidate.accepted,
                        reason=candidate.reason,
                        duplicate=candidate.reason == "duplicate",
                        budget_exhausted=candidate.reason == "budget_exhausted",
                        metadata={
                            "action": candidate.action,
                            "removed_condition_index": candidate.removed_condition_index,
                            "removed_condition_fingerprint": candidate.removed_condition_fingerprint,
                            "novelty": None if candidate.novelty is None else candidate.novelty.to_dict(),
                            "diversity": None if candidate.diversity is None else candidate.diversity.to_dict(),
                            "combined_score": candidate.combined_score,
                        },
                    )

        return batch

    def prune_population(
        self,
        population: Iterable[Hypothesis],
        *,
        depth: int = 0,
        limit_per_parent: int | None = None,
    ) -> tuple[PruningBatch, ...]:
        batches: list[PruningBatch] = []
        for hypothesis in population:
            batches.append(
                self.prune(
                    hypothesis,
                    depth=depth,
                    limit=limit_per_parent,
                )
            )
        return tuple(batches)

    def simplify(
        self,
        hypothesis: Hypothesis,
        *,
        depth: int = 0,
        condition_index: int | None = None,
    ) -> PruningCandidate | None:
        if not isinstance(hypothesis, Hypothesis):
            raise TypeError("hypothesis must be a Hypothesis.")

        parent_fingerprint = fingerprint_model(hypothesis)

        if condition_index is None:
            condition_index = self._heuristics.choose_condition_index(
                hypothesis,
                strategy="weakest",
            )

        result = self._apply_prune(
            hypothesis,
            condition_index=condition_index,
            depth=depth,
        )
        if result is None:
            return None

        candidate = self._evaluate_candidate(
            result,
            depth=depth,
            parent_fingerprint=parent_fingerprint,
            removed_index=condition_index,
        )
        self._commit(candidate, depth=depth, budget_consumed=False)
        return candidate

    def drain(
        self,
        hypothesis: Hypothesis,
        *,
        depth: int = 0,
        limit: int | None = None,
    ) -> tuple[PruningCandidate, ...]:
        batch = self.prune(
            hypothesis,
            depth=depth,
            limit=limit,
        )
        return batch.candidates

    # ==========================================================
    # INTERNAL ACTIONS
    # ==========================================================

    def _apply_prune(
        self,
        hypothesis: Hypothesis,
        *,
        condition_index: int,
        depth: int = 0,
    ) -> GenerationResult | None:
        try:
            return self._generator.prune(
                hypothesis,
                condition_index=condition_index,
            )
        except Exception:
            return None

    def _evaluate_candidate(
        self,
        result: GenerationResult,
        *,
        depth: int,
        parent_fingerprint: str,
        removed_index: int | None,
    ) -> PruningCandidate:
        hypothesis = result.hypothesis

        novelty = self._novelty.assess(
            hypothesis,
            parent_fingerprint=parent_fingerprint,
            depth=depth + 1,
        )
        diversity = self._diversity.assess(
            hypothesis,
            parent_fingerprint=parent_fingerprint,
            depth=depth + 1,
        )

        heuristic_score = self._heuristics.score_hypothesis(
            hypothesis,
            depth=depth + 1,
        )

        combined_score = self._combine_scores(
            novelty=novelty.score,
            diversity=diversity.score,
            heuristic=heuristic_score,
            generation=result,
            depth=depth,
        )

        accepted = (
            not novelty.duplicate
            and not diversity.duplicate
            and novelty.score >= self._settings.novelty_floor
            and diversity.score >= self._settings.diversity_floor
            and combined_score >= self._settings.combined_floor
        )

        reason = "accepted"
        if novelty.duplicate or diversity.duplicate:
            reason = "duplicate"
            accepted = False
        elif novelty.score < self._settings.novelty_floor:
            reason = "novelty_too_low"
            accepted = False
        elif diversity.score < self._settings.diversity_floor:
            reason = "diversity_too_low"
            accepted = False
        elif combined_score < self._settings.combined_floor:
            reason = "score_too_low"
            accepted = False

        removed_condition = None
        if removed_index is not None and 0 <= removed_index < len(result.hypothesis.conditions) + 1:
            try:
                original_condition = result.details.get("removed_condition") if hasattr(result, "details") else None
            except Exception:
                original_condition = None
            if isinstance(original_condition, Condition):
                removed_condition = original_condition

        removed_fingerprint = None
        if removed_condition is not None:
            removed_fingerprint = fingerprint_model(removed_condition)

        family = None
        if hypothesis.conditions:
            family = hypothesis.conditions[0].left.economic_family.value

        return PruningCandidate(
            parent_fingerprint=parent_fingerprint,
            child_fingerprint=result.fingerprint,
            action="prune",
            generation=result,
            novelty=novelty,
            diversity=diversity,
            combined_score=combined_score,
            accepted=accepted,
            reason=reason,
            removed_condition_index=removed_index,
            removed_condition_fingerprint=removed_fingerprint,
            family=family,
            depth=depth + 1,
            condition_count=len(hypothesis.conditions),
            metadata={
                "heuristic_score": heuristic_score,
                "novelty_score": novelty.score,
                "diversity_score": diversity.score,
            },
        )

    def _combine_scores(
        self,
        *,
        novelty: float,
        diversity: float,
        heuristic: float,
        generation: GenerationResult,
        depth: int,
    ) -> float:
        score = (
            0.40 * max(0.0, min(1.0, novelty))
            + 0.35 * max(0.0, min(1.0, diversity))
            + 0.25 * max(0.0, min(1.0, heuristic))
        )

        simplicity_bonus = 0.03 * (1.0 / max(1.0, float(len(generation.hypothesis.conditions))))
        score += simplicity_bonus

        if generation.parent_fingerprint is not None:
            score *= 0.99

        depth_penalty = 1.0 - min(0.20, 0.03 * max(0, depth))
        score *= depth_penalty

        return max(0.0, min(1.0, score * max(0.0, self._settings.action_bias.get("prune", 1.0))))

    def _commit(
        self,
        candidate: PruningCandidate,
        *,
        depth: int,
        budget_consumed: bool,
    ) -> None:
        hypothesis = candidate.hypothesis

        self._family_manager.record_hypothesis(hypothesis)

        self._novelty.record_assessment(
            candidate.novelty or self._novelty.assess(hypothesis, parent_fingerprint=candidate.parent_fingerprint, depth=depth + 1),
            hypothesis=hypothesis,
            depth=depth + 1,
            parent_fingerprint=candidate.parent_fingerprint,
        )
        self._diversity.record_assessment(
            candidate.diversity or self._diversity.assess(hypothesis, parent_fingerprint=candidate.parent_fingerprint, depth=depth + 1),
            hypothesis=hypothesis,
            depth=depth + 1,
            parent_fingerprint=candidate.parent_fingerprint,
        )

        if self._budget is not None and budget_consumed:
            family = candidate.family or (hypothesis.conditions[0].left.economic_family if hypothesis.conditions else None)
            if family is not None:
                self._budget.consume(
                    family=family,
                    depth=depth + 1,
                    active=False,
                    amount=1,
                )
            else:
                self._budget.consume(
                    family=None,
                    depth=depth + 1,
                    active=False,
                    amount=1,
                )

        if self._report is not None:
            if candidate.accepted:
                self._report.mark_accepted(candidate.child_fingerprint, candidate.combined_score)
            else:
                self._report.mark_rejected(candidate.reason)

    # ==========================================================
    # RANKING
    # ==========================================================

    def _rank_condition_indices(
        self,
        hypothesis: Hypothesis,
    ) -> tuple[int, ...]:
        scores = [
            (index, self._condition_importance(condition, hypothesis))
            for index, condition in enumerate(hypothesis.conditions)
        ]

        scores.sort(key=lambda item: (item[1], item[0]))
        return tuple(index for index, _score in scores)

    def _condition_importance(
        self,
        condition: Condition,
        hypothesis: Hypothesis,
    ) -> float:
        feature = condition.left
        family_key = feature.economic_family.value
        feature_novelty = 1.0 / (1.0 + float(self._family_manager.feature_usage.get(feature.column_index, 0)))
        family_novelty = 1.0 / (1.0 + float(self._family_manager.family_usage.get(family_key, 0)))
        operator_novelty = 1.0 / (1.0 + float(self._heuristics.operator_weights_config.get(condition.operator.value, 1.0)))

        local_family_count = sum(
            1
            for item in hypothesis.conditions
            if item.left.economic_family == feature.economic_family
        )
        local_balance = 1.0 / max(1.0, float(local_family_count))

        if isinstance(condition.right, Feature):
            right_bonus = 0.06
        else:
            right_bonus = 0.0

        profile = _feature_profile(feature)
        profile_bonus = 1.0
        if profile in {"oscillator", "normalized", "ratio", "distance"}:
            profile_bonus *= 1.1
        elif profile == "binary":
            profile_bonus *= 0.95

        score = (
            0.38 * feature_novelty
            + 0.24 * family_novelty
            + 0.18 * operator_novelty
            + 0.12 * local_balance
            + 0.08 * right_bonus
        ) * profile_bonus

        return max(0.0001, score)

    # ==========================================================
    # PYTHON PROTOCOL
    # ==========================================================

    def __len__(self) -> int:
        return len(self._action_counts)

    def __iter__(self):
        return iter(self._action_counts.items())

    def __repr__(self) -> str:
        return (
            "PruningEngine("
            f"actions={dict(self._action_counts)}, "
            f"accepted={dict(self._accepted_counts)}, "
            f"rejected={dict(self._rejected_counts)}"
            ")"
        )