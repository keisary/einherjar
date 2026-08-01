"""
==========================================================
Expansion Engine
==========================================================

Orchestre les transformations d'une hypothèse pendant la
phase Discovery.

Ce module ne valide rien statistiquement. Il ne fait que :
- choisir une action de recherche,
- appliquer une transformation,
- mesurer la nouveauté et la diversité obtenues,
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
from .heuristics import HeuristicDecision
from .novelty import NoveltyAssessment
from .novelty import NoveltyEngine
from .search_budget import BudgetSnapshot
from .search_budget import SearchBudget
from .search_report import SearchReport


__all__ = [
    "ExpansionSettings",
    "ExpansionCandidate",
    "ExpansionBatch",
    "ExpansionEngine",
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


# ==========================================================
# SETTINGS
# ==========================================================

@dataclass(frozen=True, slots=True)
class ExpansionSettings:
    """
    Paramètres de la phase d'expansion.

    Les valeurs par défaut restent volontairement modestes :
    l'objectif est de produire une petite frontière de
    candidats bien évalués plutôt qu'une explosion de
    variantes.
    """

    max_children_per_parent: int = 6
    max_attempts_per_parent: int = 24

    novelty_floor: float = 0.20
    diversity_floor: float = 0.20
    combined_floor: float = 0.25

    prefer_new_families: bool = True
    prefer_best_actions: bool = True
    keep_duplicates: bool = False
    record_rejections: bool = True

    action_bias: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "max_children_per_parent",
            max(1, _coerce_int(self.max_children_per_parent, 6)),
        )
        object.__setattr__(
            self,
            "max_attempts_per_parent",
            max(1, _coerce_int(self.max_attempts_per_parent, 24)),
        )
        object.__setattr__(
            self,
            "novelty_floor",
            min(1.0, max(0.0, _coerce_float(self.novelty_floor, 0.20))),
        )
        object.__setattr__(
            self,
            "diversity_floor",
            min(1.0, max(0.0, _coerce_float(self.diversity_floor, 0.20))),
        )
        object.__setattr__(
            self,
            "combined_floor",
            min(1.0, max(0.0, _coerce_float(self.combined_floor, 0.25))),
        )
        object.__setattr__(self, "prefer_new_families", _coerce_bool(self.prefer_new_families, True))
        object.__setattr__(self, "prefer_best_actions", _coerce_bool(self.prefer_best_actions, True))
        object.__setattr__(self, "keep_duplicates", _coerce_bool(self.keep_duplicates, False))
        object.__setattr__(self, "record_rejections", _coerce_bool(self.record_rejections, True))
        object.__setattr__(self, "action_bias", {str(k).strip().lower(): max(0.0, _coerce_float(v, 1.0)) for k, v in dict(self.action_bias).items()})

    @classmethod
    def from_config(cls, config: Any | None) -> "ExpansionSettings":
        if config is None:
            return cls()

        return cls(
            max_children_per_parent=_coerce_int(
                _first_non_none(
                    config,
                    ("discovery", "max_children_per_parent"),
                    ("expansion", "max_children_per_parent"),
                    ("search", "max_children_per_parent"),
                    default=6,
                ),
                6,
            ),
            max_attempts_per_parent=_coerce_int(
                _first_non_none(
                    config,
                    ("discovery", "max_attempts_per_parent"),
                    ("expansion", "max_attempts_per_parent"),
                    ("search", "max_attempts_per_parent"),
                    default=24,
                ),
                24,
            ),
            novelty_floor=_coerce_float(
                _first_non_none(
                    config,
                    ("discovery", "novelty_floor"),
                    ("expansion", "novelty_floor"),
                    ("search", "novelty_floor"),
                    default=0.20,
                ),
                0.20,
            ),
            diversity_floor=_coerce_float(
                _first_non_none(
                    config,
                    ("discovery", "diversity_floor"),
                    ("expansion", "diversity_floor"),
                    ("search", "diversity_floor"),
                    default=0.20,
                ),
                0.20,
            ),
            combined_floor=_coerce_float(
                _first_non_none(
                    config,
                    ("discovery", "combined_floor"),
                    ("expansion", "combined_floor"),
                    ("search", "combined_floor"),
                    default=0.25,
                ),
                0.25,
            ),
            prefer_new_families=_coerce_bool(
                _first_non_none(
                    config,
                    ("discovery", "prefer_new_families"),
                    ("expansion", "prefer_new_families"),
                    ("search", "prefer_new_families"),
                    default=True,
                ),
                True,
            ),
            prefer_best_actions=_coerce_bool(
                _first_non_none(
                    config,
                    ("discovery", "prefer_best_actions"),
                    ("expansion", "prefer_best_actions"),
                    ("search", "prefer_best_actions"),
                    default=True,
                ),
                True,
            ),
            keep_duplicates=_coerce_bool(
                _first_non_none(
                    config,
                    ("discovery", "keep_duplicates"),
                    ("expansion", "keep_duplicates"),
                    ("search", "keep_duplicates"),
                    default=False,
                ),
                False,
            ),
            record_rejections=_coerce_bool(
                _first_non_none(
                    config,
                    ("discovery", "record_rejections"),
                    ("expansion", "record_rejections"),
                    ("search", "record_rejections"),
                    default=True,
                ),
                True,
            ),
            action_bias=_normalize_tuple(
                _first_non_none(
                    config,
                    ("discovery", "action_bias"),
                    ("expansion", "action_bias"),
                    ("search", "action_bias"),
                    default={},
                )
            ),
        )


# ==========================================================
# RESULTS
# ==========================================================

@dataclass(frozen=True, slots=True)
class ExpansionCandidate:
    """
    Candidat généré pendant une expansion.

    Il regroupe la transformation produite, ainsi que les
    scores de nouveauté et de diversité associés.
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
            "family": self.family,
            "depth": self.depth,
            "condition_count": self.condition_count,
            "metadata": dict(self.metadata),
        }

    def __repr__(self) -> str:
        return (
            "ExpansionCandidate("
            f"action='{self.action}', "
            f"score={self.combined_score:.4f}, "
            f"accepted={self.accepted}, "
            f"conditions={self.condition_count}"
            ")"
        )


@dataclass(frozen=True, slots=True)
class ExpansionBatch:
    """
    Lot de candidats produits pour une hypothèse par le
    moteur d'expansion.
    """

    parent_fingerprint: str
    depth: int
    candidates: tuple[ExpansionCandidate, ...]

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
    def accepted(self) -> tuple[ExpansionCandidate, ...]:
        return tuple(candidate for candidate in self.candidates if candidate.accepted)

    @property
    def rejected(self) -> tuple[ExpansionCandidate, ...]:
        return tuple(candidate for candidate in self.candidates if not candidate.accepted)

    @property
    def best(self) -> ExpansionCandidate | None:
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
            "ExpansionBatch("
            f"depth={self.depth}, "
            f"candidates={len(self.candidates)}, "
            f"exhausted={self.exhausted}"
            ")"
        )


# ==========================================================
# ENGINE
# ==========================================================

class ExpansionEngine:
    """
    Moteur d'expansion des hypothèses.

    Il utilise :
    - le Generator pour fabriquer les transformations,
    - les Heuristics pour choisir la prochaine action,
    - Novelty et Diversity pour scorer les enfants,
    - le SearchBudget pour éviter les dérives,
    - le SearchReport pour journaliser le résultat.
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
        settings: ExpansionSettings | None = None,
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
                search_config = _build_search_config(_first_non_none(config, ("search",), ("search_config",), ("discovery", "search"), default=config))
            if scoring_config is None:
                scoring_config = _build_scoring_config(_first_non_none(config, ("scoring",), ("scoring_config",), ("discovery", "scoring"), default=config))

        self._search_config = search_config or heuristics.search_config
        self._scoring_config = scoring_config or heuristics.scoring_config

        self._settings = settings or ExpansionSettings.from_config(config)

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
        self._family_counts: Counter[str] = Counter()
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
    def settings(self) -> ExpansionSettings:
        return self._settings

    @property
    def action_counts(self) -> dict[str, int]:
        return dict(self._action_counts)

    @property
    def family_counts(self) -> dict[str, int]:
        return dict(self._family_counts)

    @property
    def accepted_counts(self) -> dict[str, int]:
        return dict(self._accepted_counts)

    @property
    def rejected_counts(self) -> dict[str, int]:
        return dict(self._rejected_counts)

    # ==========================================================
    # PUBLIC API
    # ==========================================================

    def seed(
        self,
        *,
        family: EconomicFamily | str | None = None,
        depth: int = 0,
    ) -> ExpansionCandidate:
        generated = self._generator.seed(family=family)
        return self._decorate_generated(
            generated,
            parent_fingerprint="",
            action="seed",
            depth=depth,
        )

    def expand(
        self,
        hypothesis: Hypothesis,
        *,
        depth: int = 0,
        family_hint: EconomicFamily | str | None = None,
        limit: int | None = None,
        available_actions: Iterable[str] | None = None,
    ) -> ExpansionBatch:
        if not isinstance(hypothesis, Hypothesis):
            raise TypeError("hypothesis must be a Hypothesis.")

        parent_fingerprint = fingerprint_model(hypothesis)
        chosen_limit = max(1, _coerce_int(limit, self._settings.max_children_per_parent))
        actions = tuple(
            str(action).strip().lower()
            for action in (
                available_actions
                if available_actions is not None
                else self._heuristics.action_weights(hypothesis, depth=depth).keys()
            )
        )

        if not actions:
            actions = self._heuristics.available_actions(hypothesis, depth=depth)

        action_weights = self._heuristics.action_weights(
            hypothesis,
            depth=depth,
            available_actions=actions,
        )

        ordered_actions = self._order_actions(action_weights, preferred_first=self._settings.prefer_best_actions)

        candidates: list[ExpansionCandidate] = []
        attempts = 0
        exhausted = False
        reason: str | None = None

        while len(candidates) < chosen_limit and attempts < self._settings.max_attempts_per_parent:
            attempts += 1

            if self._budget is not None and not self._budget.can_generate(
                family=self._resolve_target_family(
                    hypothesis,
                    action=None,
                    family_hint=family_hint,
                ),
                depth=depth + 1,
                condition_count=len(hypothesis.conditions),
                amount=1,
            ):
                exhausted = True
                reason = "budget_exhausted"
                self._budget.exhaust(reason)
                break

            action = self._pick_action(
                hypothesis,
                ordered_actions=ordered_actions,
                depth=depth,
                available_actions=actions,
            )

            if action is None:
                reason = "no_action_available"
                break

            result = self._apply_action(
                hypothesis,
                action=action,
                family_hint=family_hint,
                depth=depth,
            )

            if result is None:
                self._rejected_counts[action] += 1
                continue

            candidate = self._evaluate_candidate(
                result,
                action=action,
                depth=depth,
                parent_fingerprint=parent_fingerprint,
            )

            if not candidate.accepted and not self._settings.keep_duplicates and candidate.reason == "duplicate":
                self._rejected_counts[action] += 1
                if self._report is not None and self._settings.record_rejections:
                    self._report.record_generation(
                        result,
                        depth=depth + 1,
                        accepted=False,
                        reason=candidate.reason,
                        duplicate=True,
                        budget_exhausted=False,
                        metadata={
                            "expansion": candidate.to_dict(),
                        },
                    )
                continue

            candidates.append(candidate)

            self._action_counts[action] += 1
            if candidate.family is not None:
                self._family_counts[candidate.family] += 1

            if candidate.accepted:
                self._accepted_counts[action] += 1
            else:
                self._rejected_counts[action] += 1

            self._commit(
                candidate,
                depth=depth,
                budget_consumed=True,
            )

        batch = ExpansionBatch(
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
                "actions": list(actions),
                "ordered_actions": list(ordered_actions),
            },
        )

        if self._report is not None:
            self._report.record_budget(batch.budget_snapshot) if batch.budget_snapshot is not None else None
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
                            "novelty": None if candidate.novelty is None else candidate.novelty.to_dict(),
                            "diversity": None if candidate.diversity is None else candidate.diversity.to_dict(),
                            "combined_score": candidate.combined_score,
                        },
                    )
        return batch

    def expand_population(
        self,
        population: Iterable[Hypothesis],
        *,
        depth: int = 0,
        limit_per_parent: int | None = None,
    ) -> tuple[ExpansionBatch, ...]:
        batches: list[ExpansionBatch] = []
        for hypothesis in population:
            batches.append(
                self.expand(
                    hypothesis,
                    depth=depth,
                    limit=limit_per_parent,
                )
            )
        return tuple(batches)

    def transform(
        self,
        hypothesis: Hypothesis,
        *,
        action: str | None = None,
        family_hint: EconomicFamily | str | None = None,
        condition_index: int | None = None,
        depth: int = 0,
    ) -> ExpansionCandidate | None:
        if not isinstance(hypothesis, Hypothesis):
            raise TypeError("hypothesis must be a Hypothesis.")

        parent_fingerprint = fingerprint_model(hypothesis)

        if action is None:
            action = self._heuristics.choose_action(
                hypothesis,
                depth=depth,
                available_actions=self._heuristics.available_actions(hypothesis, depth=depth),
            )

        action = str(action).strip().lower()

        result = self._apply_action(
            hypothesis,
            action=action,
            family_hint=family_hint,
            condition_index=condition_index,
            depth=depth,
        )
        if result is None:
            return None

        candidate = self._evaluate_candidate(
            result,
            action=action,
            depth=depth,
            parent_fingerprint=parent_fingerprint,
        )
        self._commit(candidate, depth=depth, budget_consumed=False)
        return candidate

    def propose(
        self,
        hypothesis: Hypothesis,
        *,
        depth: int = 0,
        family_hint: EconomicFamily | str | None = None,
        available_actions: Iterable[str] | None = None,
    ) -> ExpansionCandidate | None:
        batch = self.expand(
            hypothesis,
            depth=depth,
            family_hint=family_hint,
            limit=1,
            available_actions=available_actions,
        )
        return batch.best

    def drain(
        self,
        hypothesis: Hypothesis,
        *,
        depth: int = 0,
        family_hint: EconomicFamily | str | None = None,
        limit: int | None = None,
    ) -> tuple[ExpansionCandidate, ...]:
        batch = self.expand(
            hypothesis,
            depth=depth,
            family_hint=family_hint,
            limit=limit,
        )
        return batch.candidates

    # ==========================================================
    # INTERNAL ACTIONS
    # ==========================================================

    def _apply_action(
        self,
        hypothesis: Hypothesis,
        *,
        action: str,
        family_hint: EconomicFamily | str | None = None,
        condition_index: int | None = None,
        depth: int = 0,
    ) -> GenerationResult | None:
        family = self._coerce_family(family_hint)

        try:
            if action == "seed":
                return self._generator.seed(family=family)

            if action == "expand":
                return self._generator.expand(hypothesis, family=family)

            if action == "mutate":
                idx = condition_index
                if idx is None:
                    idx = self._heuristics.choose_condition_index(hypothesis, strategy="weighted")
                return self._generator.mutate(hypothesis, family=family, condition_index=idx)

            if action == "replace":
                idx = condition_index
                if idx is None:
                    idx = self._heuristics.choose_condition_index(hypothesis, strategy="weakest")
                return self._generator.replace(hypothesis, family=family, condition_index=idx)

            if action == "prune":
                idx = condition_index
                if idx is None:
                    idx = self._heuristics.choose_condition_index(hypothesis, strategy="weakest")
                return self._generator.prune(hypothesis, condition_index=idx)

            if action == "evolve":
                return self._generator.evolve(
                    hypothesis,
                    action=self._heuristics.choose_action(
                        hypothesis,
                        depth=depth,
                        available_actions=self._heuristics.available_actions(hypothesis, depth=depth),
                    ),
                    family=family,
                    condition_index=condition_index,
                )

        except Exception:
            return None

        return None

    def _pick_action(
        self,
        hypothesis: Hypothesis,
        *,
        ordered_actions: Sequence[str],
        depth: int,
        available_actions: Iterable[str] | None = None,
    ) -> str | None:
        if not ordered_actions:
            return None

        if not self._settings.prefer_best_actions:
            return str(self._rng.choice(list(ordered_actions)))

        weights = self._heuristics.action_weights(
            hypothesis,
            depth=depth,
            available_actions=available_actions,
        )

        actions = [action for action in ordered_actions if weights.get(action, 0.0) > 0.0]
        if not actions:
            return None

        scores = [weights[action] for action in actions]
        action = self._weighted_choice(actions, scores)

        return str(action)

    def _order_actions(
        self,
        weights: Mapping[str, float],
        *,
        preferred_first: bool = True,
    ) -> tuple[str, ...]:
        items = [
            (str(action).strip().lower(), max(0.0, _coerce_float(weight, 0.0)))
            for action, weight in weights.items()
        ]
        items = [item for item in items if item[1] > 0.0]

        if not items:
            return ()

        if preferred_first:
            items.sort(key=lambda item: (-item[1], item[0]))
        else:
            items.sort(key=lambda item: item[0])

        return tuple(action for action, _weight in items)

    def _resolve_target_family(
        self,
        hypothesis: Hypothesis,
        *,
        action: str | None,
        family_hint: EconomicFamily | str | None,
    ) -> EconomicFamily | None:
        family = self._coerce_family(family_hint)
        if family is not None:
            return family

        if hypothesis.conditions:
            return hypothesis.conditions[-1].left.economic_family

        if action in {"seed", "expand"}:
            try:
                return self._heuristics.choose_family(
                    hypothesis if hypothesis.conditions else None,
                    action=action or "expand",
                    seedable=(action == "seed"),
                )
            except Exception:
                return None

        return None

    def _evaluate_candidate(
        self,
        result: GenerationResult,
        *,
        action: str,
        depth: int,
        parent_fingerprint: str,
    ) -> ExpansionCandidate:
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
            action=action,
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

        family = None
        if hypothesis.conditions:
            family = hypothesis.conditions[0].left.economic_family.value

        return ExpansionCandidate(
            parent_fingerprint=parent_fingerprint,
            child_fingerprint=result.fingerprint,
            action=action,
            generation=result,
            novelty=novelty,
            diversity=diversity,
            combined_score=combined_score,
            accepted=accepted,
            reason=reason,
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
        action: str,
        depth: int,
    ) -> float:
        action_bias = self._settings.action_bias.get(action, 1.0)

        score = (
            0.40 * max(0.0, min(1.0, novelty))
            + 0.35 * max(0.0, min(1.0, diversity))
            + 0.25 * max(0.0, min(1.0, heuristic))
        )

        if generation.parent_fingerprint is not None:
            score *= 0.98

        depth_penalty = 1.0 - min(0.25, 0.04 * max(0, depth))
        score *= depth_penalty

        return max(0.0, min(1.0, score * max(0.0, action_bias)))

    def _decorate_generated(
        self,
        generated: GenerationResult,
        *,
        parent_fingerprint: str,
        action: str,
        depth: int,
    ) -> ExpansionCandidate:
        novelty = self._novelty.assess(
            generated.hypothesis,
            parent_fingerprint=parent_fingerprint or generated.parent_fingerprint,
            depth=depth + 1,
        )
        diversity = self._diversity.assess(
            generated.hypothesis,
            parent_fingerprint=parent_fingerprint or generated.parent_fingerprint,
            depth=depth + 1,
        )

        combined_score = self._combine_scores(
            novelty=novelty.score,
            diversity=diversity.score,
            heuristic=self._heuristics.score_hypothesis(generated.hypothesis, depth=depth + 1),
            generation=generated,
            action=action,
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

        family = None
        if generated.hypothesis.conditions:
            family = generated.hypothesis.conditions[0].left.economic_family.value

        return ExpansionCandidate(
            parent_fingerprint=parent_fingerprint,
            child_fingerprint=generated.fingerprint,
            action=action,
            generation=generated,
            novelty=novelty,
            diversity=diversity,
            combined_score=combined_score,
            accepted=accepted,
            reason=reason,
            family=family,
            depth=depth + 1,
            condition_count=len(generated.hypothesis.conditions),
            metadata={
                "seeded": True,
            },
        )

    def _commit(
        self,
        candidate: ExpansionCandidate,
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
                    active=True,
                    amount=1,
                )
            else:
                self._budget.consume(
                    family=None,
                    depth=depth + 1,
                    active=True,
                    amount=1,
                )

        if self._report is not None:
            if candidate.accepted:
                self._report.mark_accepted(candidate.child_fingerprint, candidate.combined_score)
            else:
                self._report.mark_rejected(candidate.reason)

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

    def _weighted_choice(self, items: Sequence[Any], weights: Sequence[float]) -> Any:
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

    # ==========================================================
    # PYTHON PROTOCOL
    # ==========================================================

    def __len__(self) -> int:
        return len(self._action_counts)

    def __iter__(self):
        return iter(self._action_counts.items())

    def __repr__(self) -> str:
        return (
            "ExpansionEngine("
            f"actions={dict(self._action_counts)}, "
            f"accepted={dict(self._accepted_counts)}, "
            f"rejected={dict(self._rejected_counts)}"
            ")"
        )