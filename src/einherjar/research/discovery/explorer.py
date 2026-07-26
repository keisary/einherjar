"""
==========================================================
Discovery Explorer
==========================================================

Orchestrateur principal de la phase Discovery.

Explorer pilote :
- le seed initial,
- la sélection des actions,
- la génération des hypothèses,
- l'évaluation de leur nouveauté et de leur diversité,
- le maintien de la frontière active,
- la consommation du budget de recherche,
- la journalisation du processus.

Il ne valide rien statistiquement.
Il ne simule rien.
Il construit et fait évoluer une population vivante d'hypothèses.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Iterable
from typing import Mapping

import numpy as np

from config.scoring import ScoringConfig
from config.search import SearchConfig
from core.context import EngineContext
from models.enums import EconomicFamily
from models.feature import Feature
from models.feature_registry import FeatureRegistry
from models.fingerprint import fingerprint_model
from models.hypothesis import Hypothesis

from .budget import BudgetSnapshot
from .diversity import DiversityAssessment
from .diversity import DiversityEngine
from .family_manager import FamilyManager
from .generator import DiscoveryGenerator
from .generator import GenerationResult
from .heuristics import DiscoveryHeuristics
from .novelty import NoveltyAssessment
from .novelty import NoveltyEngine
from .search_budget import SearchBudget
from .search_report import SearchReport


__all__ = [
    "DiscoveryNode",
    "DiscoveryResult",
    "Explorer",
]


def _coerce_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


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


def _family_value(family: EconomicFamily | str | None) -> str | None:
    if family is None:
        return None
    if isinstance(family, EconomicFamily):
        return family.value
    return str(family).strip().lower() or None


def _node_family(hypothesis: Hypothesis) -> str | None:
    if not hypothesis.conditions:
        return None
    return hypothesis.conditions[0].left.economic_family.value


def _unique_by_fingerprint(nodes: Iterable["DiscoveryNode"]) -> tuple["DiscoveryNode", ...]:
    best_by_fp: dict[str, DiscoveryNode] = {}

    for node in nodes:
        existing = best_by_fp.get(node.fingerprint)
        if existing is None or node.score > existing.score:
            best_by_fp[node.fingerprint] = node

    return tuple(best_by_fp.values())


@dataclass(frozen=True, slots=True)
class DiscoveryNode:
    """
    Nœud actif ou découvert pendant la phase Discovery.
    """

    hypothesis: Hypothesis
    fingerprint: str

    depth: int = 0
    score: float = 0.0

    novelty: NoveltyAssessment | None = None
    diversity: DiversityAssessment | None = None

    action: str = "seed"
    family: str | None = None
    parent_fingerprint: str | None = None

    accepted: bool = True
    reason: str = ""

    generation: GenerationResult | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "depth", max(0, _coerce_int(self.depth, 0)))
        object.__setattr__(self, "score", float(self.score))
        object.__setattr__(self, "action", str(self.action).strip().lower())
        object.__setattr__(self, "family", _family_value(self.family))
        object.__setattr__(self, "accepted", bool(self.accepted))
        object.__setattr__(self, "reason", str(self.reason))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def condition_count(self) -> int:
        return len(self.hypothesis.conditions)

    @property
    def has_parent(self) -> bool:
        return self.parent_fingerprint is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "fingerprint": self.fingerprint,
            "depth": self.depth,
            "score": self.score,
            "novelty": None if self.novelty is None else self.novelty.to_dict(),
            "diversity": None if self.diversity is None else self.diversity.to_dict(),
            "action": self.action,
            "family": self.family,
            "parent_fingerprint": self.parent_fingerprint,
            "accepted": self.accepted,
            "reason": self.reason,
            "generation": None if self.generation is None else self.generation.to_dict(),
            "hypothesis": self.hypothesis.to_dict(),
            "metadata": dict(self.metadata),
        }

    def __hash__(self) -> int:
        return hash(self.fingerprint)

    def __repr__(self) -> str:
        return (
            "DiscoveryNode("
            f"action='{self.action}', "
            f"depth={self.depth}, "
            f"score={self.score:.4f}, "
            f"accepted={self.accepted}"
            ")"
        )


@dataclass(frozen=True, slots=True)
class DiscoveryResult:
    """
    Résultat final d'une exécution du Discovery Explorer.
    """

    frontier: tuple[DiscoveryNode, ...]
    history: tuple[DiscoveryNode, ...]
    report: SearchReport

    iterations: int = 0
    stopped_reason: str | None = None
    budget_snapshot: BudgetSnapshot | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "frontier", tuple(self.frontier))
        object.__setattr__(self, "history", tuple(self.history))

    @property
    def best(self) -> DiscoveryNode | None:
        if not self.history:
            return None
        return max(self.history, key=lambda node: node.score)

    @property
    def best_fingerprint(self) -> str | None:
        best = self.best
        return None if best is None else best.fingerprint

    def to_dict(self) -> dict[str, Any]:
        return {
            "iterations": self.iterations,
            "stopped_reason": self.stopped_reason,
            "budget_snapshot": None if self.budget_snapshot is None else self.budget_snapshot,
            "frontier": [node.to_dict() for node in self.frontier],
            "history": [node.to_dict() for node in self.history],
            "report": self.report.to_dict(),
            "best_fingerprint": self.best_fingerprint,
        }

    def __repr__(self) -> str:
        return (
            "DiscoveryResult("
            f"iterations={self.iterations}, "
            f"frontier={len(self.frontier)}, "
            f"history={len(self.history)}, "
            f"best='{self.best_fingerprint}'"
            ")"
        )


class Explorer:
    """
    Cerveau de la phase Discovery.

    Explorer fait évoluer une population d'hypothèses en
    combinant :
    - Generator,
    - Heuristics,
    - Novelty,
    - Diversity,
    - SearchBudget,
    - SearchReport.
    """

    def __init__(
        self,
        registry: FeatureRegistry | None = None,
        config: Any | None = None,
        *,
        context: EngineContext | None = None,
        family_manager: FamilyManager | None = None,
        generator: DiscoveryGenerator | None = None,
        heuristics: DiscoveryHeuristics | None = None,
        novelty_engine: NoveltyEngine | None = None,
        diversity_engine: DiversityEngine | None = None,
        budget: SearchBudget | None = None,
        report: SearchReport | None = None,
        search_config: SearchConfig | None = None,
        scoring_config: ScoringConfig | None = None,
        rng: np.random.Generator | int | None = None,
        report_name: str = "discovery",
    ) -> None:
        if context is not None:
            config = context.config
            registry = context.feature_registry

        if registry is None:
            raise ValueError("registry is required.")

        if not isinstance(registry, FeatureRegistry):
            raise TypeError("registry must be a FeatureRegistry.")

        self._config = config
        self._registry = registry

        self._rng = (
            rng
            if isinstance(rng, np.random.Generator)
            else np.random.default_rng(rng)
            if rng is not None
            else np.random.default_rng(
                _coerce_int(getattr(search_config, "random_seed", None), 42)
                if search_config is not None
                else 42
            )
        )

        self._family_manager = family_manager or FamilyManager.from_config(
            config,
            registry,
            rng=self._rng,
        )

        self._generator = generator or DiscoveryGenerator(
            config,
            registry,
            rng=self._rng,
        )

        if heuristics is None:
            if search_config is not None or scoring_config is not None:
                heuristics = DiscoveryHeuristics(
                    self._family_manager,
                    search_config or SearchConfig(),
                    scoring_config or ScoringConfig(),
                    config=config,
                    rng=self._rng,
                )
            else:
                heuristics = DiscoveryHeuristics.from_config(
                    config,
                    self._family_manager,
                    rng=self._rng,
                )
        self._heuristics = heuristics

        self._search_config = search_config or self._heuristics.search_config
        self._scoring_config = scoring_config or self._heuristics.scoring_config

        self._novelty = novelty_engine or NoveltyEngine(
            family_manager=self._family_manager,
            search_config=self._search_config,
            scoring_config=self._scoring_config,
            config=config,
            rng=self._rng,
        )

        self._diversity = diversity_engine or DiversityEngine(
            family_manager=self._family_manager,
            search_config=self._search_config,
            scoring_config=self._scoring_config,
            config=config,
            rng=self._rng,
        )

        self._budget = budget or SearchBudget.from_config(
            config,
            search_config=self._search_config,
        )

        self._report = report or SearchReport(
            name=report_name,
            metadata={
                "module": "discovery",
            },
        )

        self._frontier: tuple[DiscoveryNode, ...] = ()
        self._history_index: dict[str, DiscoveryNode] = {}
        self._history_order: list[DiscoveryNode] = []

        self._iterations = 0
        self._stopped_reason: str | None = None

    # ==================================================
    # CONSTRUCTION
    # ==================================================

    @classmethod
    def from_context(
        cls,
        context: EngineContext,
        *,
        family_manager: FamilyManager | None = None,
        generator: DiscoveryGenerator | None = None,
        heuristics: DiscoveryHeuristics | None = None,
        novelty_engine: NoveltyEngine | None = None,
        diversity_engine: DiversityEngine | None = None,
        budget: SearchBudget | None = None,
        report: SearchReport | None = None,
        search_config: SearchConfig | None = None,
        scoring_config: ScoringConfig | None = None,
        rng: np.random.Generator | int | None = None,
        report_name: str = "discovery",
    ) -> "Explorer":
        return cls(
            context=context,
            family_manager=family_manager,
            generator=generator,
            heuristics=heuristics,
            novelty_engine=novelty_engine,
            diversity_engine=diversity_engine,
            budget=budget,
            report=report,
            search_config=search_config,
            scoring_config=scoring_config,
            rng=rng,
            report_name=report_name,
        )

    # ==================================================
    # PROPERTIES
    # ==================================================

    @property
    def config(self) -> Any | None:
        return self._config

    @property
    def registry(self) -> FeatureRegistry:
        return self._registry

    @property
    def family_manager(self) -> FamilyManager:
        return self._family_manager

    @property
    def generator(self) -> DiscoveryGenerator:
        return self._generator

    @property
    def heuristics(self) -> DiscoveryHeuristics:
        return self._heuristics

    @property
    def novelty(self) -> NoveltyEngine:
        return self._novelty

    @property
    def diversity(self) -> DiversityEngine:
        return self._diversity

    @property
    def budget(self) -> SearchBudget:
        return self._budget

    @property
    def report(self) -> SearchReport:
        return self._report

    @property
    def frontier(self) -> tuple[DiscoveryNode, ...]:
        return self._frontier

    @property
    def history(self) -> tuple[DiscoveryNode, ...]:
        return tuple(self._history_order)

    @property
    def iterations(self) -> int:
        return self._iterations

    @property
    def stopped_reason(self) -> str | None:
        return self._stopped_reason

    @property
    def best(self) -> DiscoveryNode | None:
        if not self._history_order:
            return None
        return max(self._history_order, key=lambda node: node.score)

    # ==================================================
    # LIFECYCLE
    # ==================================================

    def reset(self) -> None:
        self._frontier = ()
        self._history_index.clear()
        self._history_order.clear()
        self._iterations = 0
        self._stopped_reason = None

        self._generator.reset() if hasattr(self._generator, "reset") else None
        self._novelty.reset()
        self._diversity.reset()
        self._budget.reset()
        self._report.reset()

    def start(self) -> None:
        self._budget.start()
        self._report.start()
        self._stopped_reason = None

    def stop(self, reason: str | None = None) -> None:
        self._stopped_reason = reason or self._stopped_reason or "stopped"
        self._budget.stop(self._stopped_reason)
        self._report.finish(self._stopped_reason)

    # ==================================================
    # SEEDING
    # ==================================================

    def seed(
        self,
        *,
        size: int | None = None,
        families: Iterable[EconomicFamily | str] | None = None,
    ) -> tuple[DiscoveryNode, ...]:
        if size is None:
            size = getattr(self._generator.settings, "seed_population_size", 0) or self._search_config.beam_width or 1

        size = max(1, _coerce_int(size, 1))
        family_list = tuple(_coerce_family(item) for item in families) if families is not None else ()

        generated = self._generator.seed_population(
            size=size,
            families=family_list if family_list else None,
        )

        nodes: list[DiscoveryNode] = []

        for result in generated:
            family = _node_family(result.hypothesis)

            novelty = self._novelty.observe(
                result.hypothesis,
                parent_fingerprint=result.parent_fingerprint,
                depth=0,
            )
            diversity = self._diversity.observe(
                result.hypothesis,
                parent_fingerprint=result.parent_fingerprint,
                depth=0,
            )

            score = self._combine_scores(
                novelty=novelty.score,
                diversity=diversity.score,
                heuristic=self._heuristics.score_hypothesis(result.hypothesis, depth=0),
            )

            accepted = not novelty.duplicate and not diversity.duplicate
            reason = "accepted" if accepted else "duplicate"

            if self._budget.can_generate(
                family=family,
                depth=0,
                condition_count=len(result.hypothesis.conditions),
                amount=1,
            ):
                self._budget.consume(
                    family=family,
                    depth=0,
                    active=accepted,
                    amount=1,
                )
            else:
                self._budget.exhaust("budget_exhausted")
                self._stopped_reason = "budget_exhausted"
                break

            if accepted:
                self._family_manager.record_hypothesis(result.hypothesis)

            self._report.record_generation(
                result,
                depth=0,
                accepted=accepted,
                reason=reason,
                duplicate=not accepted,
                budget_exhausted=False,
                metadata={
                    "novelty": novelty.to_dict(),
                    "diversity": diversity.to_dict(),
                    "combined_score": score,
                },
            )

            node = DiscoveryNode(
                hypothesis=result.hypothesis,
                fingerprint=result.fingerprint,
                depth=0,
                score=score,
                novelty=novelty,
                diversity=diversity,
                action="seed",
                family=family,
                parent_fingerprint=None,
                accepted=accepted,
                reason=reason,
                generation=result,
                metadata={
                    "seed": True,
                },
            )

            if accepted:
                self._register_node(node)
                nodes.append(node)

        self._frontier = self._rank_nodes(nodes)
        self._report.record_budget(self._budget.snapshot)
        return self._frontier

    # ==================================================
    # RUN / STEP
    # ==================================================

    def run(
        self,
        *,
        seed_size: int | None = None,
        families: Iterable[EconomicFamily | str] | None = None,
        max_iterations: int | None = None,
        initial_population: Iterable[Hypothesis | DiscoveryNode] | None = None,
    ) -> DiscoveryResult:
        self.start()

        if initial_population is not None:
            self._frontier = self._bootstrap_from_population(initial_population)
        elif not self._frontier:
            self.seed(size=seed_size, families=families)

        if max_iterations is None:
            max_iterations = max(1, _coerce_int(self._search_config.max_depth, 1))

        max_iterations = max(1, _coerce_int(max_iterations, 1))

        while self._frontier and self._iterations < max_iterations and not self._budget.exhausted:
            next_frontier = self.step()

            self._iterations += 1

            if not next_frontier:
                self._stopped_reason = self._stopped_reason or "frontier_exhausted"
                break

            if self._frontier_signature(next_frontier) == self._frontier_signature(self._frontier):
                self._stopped_reason = self._stopped_reason or "stagnation"
                break

        if self._stopped_reason is None:
            if self._budget.exhausted:
                self._stopped_reason = self._budget.reason or "budget_exhausted"
            else:
                self._stopped_reason = "completed"

        self.stop(self._stopped_reason)

        return DiscoveryResult(
            frontier=self._frontier,
            history=self.history,
            report=self._report,
            iterations=self._iterations,
            stopped_reason=self._stopped_reason,
            budget_snapshot=self._budget.snapshot,
        )

    def explore(
        self,
        *,
        seed_size: int | None = None,
        families: Iterable[EconomicFamily | str] | None = None,
        max_iterations: int | None = None,
        initial_population: Iterable[Hypothesis | DiscoveryNode] | None = None,
    ) -> DiscoveryResult:
        return self.run(
            seed_size=seed_size,
            families=families,
            max_iterations=max_iterations,
            initial_population=initial_population,
        )

    def step(
        self,
        frontier: Iterable[DiscoveryNode] | None = None,
    ) -> tuple[DiscoveryNode, ...]:
        current = tuple(frontier) if frontier is not None else self._frontier
        if not current:
            return ()

        current = self._rank_nodes(current)[: max(1, self._search_config.beam_width)]
        next_pool: list[DiscoveryNode] = list(current)

        for node in current:
            if self._budget.exhausted:
                self._stopped_reason = self._stopped_reason or "budget_exhausted"
                break

            child = self._transform(node)
            if child is not None and child.accepted:
                next_pool.append(child)

        next_frontier = self._rank_nodes(_unique_by_fingerprint(next_pool))[: max(1, self._search_config.beam_width)]

        kept = {node.fingerprint for node in next_frontier}
        for node in next_pool:
            if node.accepted and node.fingerprint not in kept:
                self._budget.release(
                    family=node.family,
                    amount=1,
                )

        self._frontier = next_frontier
        self._report.record_budget(self._budget.snapshot)

        return self._frontier

    # ==================================================
    # INTERNAL TRANSFORM
    # ==================================================

    def _transform(self, node: DiscoveryNode) -> DiscoveryNode | None:
        hypothesis = node.hypothesis
        depth = node.depth + 1

        available_actions = self._heuristics.available_actions(
            hypothesis,
            depth=node.depth,
        )

        decision = self._heuristics.plan(
            hypothesis,
            depth=node.depth,
            available_actions=available_actions,
        )

        action = decision.action
        family = decision.family

        if self._budget.exhausted:
            self._stopped_reason = self._stopped_reason or "budget_exhausted"
            return None

        if not self._budget.can_generate(
            family=family,
            depth=depth,
            condition_count=len(hypothesis.conditions),
            amount=1,
        ):
            self._budget.exhaust("budget_exhausted")
            self._stopped_reason = "budget_exhausted"
            self._report.mark_budget_pruned()
            return None

        generation = self._generate(
            hypothesis,
            action=action,
            family=family,
            condition_index=decision.condition_index,
            depth=node.depth,
        )

        if generation is None:
            self._budget.consume(
                family=family,
                depth=depth,
                active=False,
                amount=1,
            )
            self._report.mark_rejected("generation_failed")
            self._report.record_budget(self._budget.snapshot)
            return None

        novelty = self._novelty.observe(
            generation.hypothesis,
            parent_fingerprint=node.fingerprint,
            depth=depth,
        )
        diversity = self._diversity.observe(
            generation.hypothesis,
            parent_fingerprint=node.fingerprint,
            depth=depth,
        )

        heuristic_score = self._heuristics.score_hypothesis(
            generation.hypothesis,
            depth=depth,
        )

        score = self._combine_scores(
            novelty=novelty.score,
            diversity=diversity.score,
            heuristic=heuristic_score,
        )

        accepted = not novelty.duplicate and not diversity.duplicate
        reason = "accepted" if accepted else "duplicate"

        if accepted:
            self._budget.consume(
                family=family or _node_family(generation.hypothesis),
                depth=depth,
                active=True,
                amount=1,
            )
            self._family_manager.record_hypothesis(generation.hypothesis)
        else:
            self._budget.consume(
                family=family or _node_family(generation.hypothesis),
                depth=depth,
                active=False,
                amount=1,
            )

        self._report.record_generation(
            generation,
            depth=depth,
            accepted=accepted,
            reason=reason,
            duplicate=not accepted,
            budget_exhausted=False,
            metadata={
                "novelty": novelty.to_dict(),
                "diversity": diversity.to_dict(),
                "combined_score": score,
                "heuristic_score": heuristic_score,
                "decision": decision.to_dict(),
            },
        )

        self._report.record_budget(self._budget.snapshot)

        if not accepted:
            return None

        child = DiscoveryNode(
            hypothesis=generation.hypothesis,
            fingerprint=generation.fingerprint,
            depth=depth,
            score=score,
            novelty=novelty,
            diversity=diversity,
            action=action,
            family=family.value if isinstance(family, EconomicFamily) else _node_family(generation.hypothesis),
            parent_fingerprint=node.fingerprint,
            accepted=True,
            reason=reason,
            generation=generation,
            metadata={
                "heuristic_score": heuristic_score,
                "decision": decision.to_dict(),
            },
        )

        self._register_node(child)
        return child

    def _generate(
        self,
        hypothesis: Hypothesis,
        *,
        action: str,
        family: EconomicFamily | None,
        condition_index: int | None,
        depth: int,
    ) -> GenerationResult | None:
        try:
            if action == "seed":
                return self._generator.seed(family=family)

            if action == "expand":
                return self._generator.expand(hypothesis, family=family)

            if action == "mutate":
                return self._generator.mutate(
                    hypothesis,
                    family=family,
                    condition_index=condition_index,
                )

            if action == "replace":
                return self._generator.replace(
                    hypothesis,
                    family=family,
                    condition_index=condition_index,
                )

            if action == "prune":
                return self._generator.prune(
                    hypothesis,
                    condition_index=condition_index,
                )

            if action == "evolve":
                return self._generator.evolve(
                    hypothesis,
                    action=self._heuristics.choose_action(
                        hypothesis,
                        depth=depth,
                        available_actions=self._heuristics.available_actions(
                            hypothesis,
                            depth=depth,
                        ),
                    ),
                    family=family,
                    condition_index=condition_index,
                )
        except Exception:
            return None

        return None

    # ==================================================
    # INTERNAL POPULATION
    # ==================================================

    def _bootstrap_from_population(
        self,
        population: Iterable[Hypothesis | DiscoveryNode],
    ) -> tuple[DiscoveryNode, ...]:
        nodes: list[DiscoveryNode] = []

        for item in population:
            if isinstance(item, DiscoveryNode):
                node = item
            elif isinstance(item, Hypothesis):
                fp = fingerprint_model(item)
                novelty = self._novelty.observe(item, depth=0)
                diversity = self._diversity.observe(item, depth=0)
                score = self._combine_scores(
                    novelty=novelty.score,
                    diversity=diversity.score,
                    heuristic=self._heuristics.score_hypothesis(item, depth=0),
                )
                node = DiscoveryNode(
                    hypothesis=item,
                    fingerprint=fp,
                    depth=0,
                    score=score,
                    novelty=novelty,
                    diversity=diversity,
                    action="seed",
                    family=_node_family(item),
                    accepted=True,
                    reason="bootstrap",
                )
            else:
                raise TypeError("population must contain Hypothesis or DiscoveryNode instances.")

            if node.accepted:
                self._register_node(node)
                nodes.append(node)

        nodes = list(_unique_by_fingerprint(nodes))
        nodes = list(self._rank_nodes(nodes)[: max(1, self._search_config.beam_width)])

        for node in nodes:
            self._budget.consume(
                family=node.family,
                depth=node.depth,
                active=True,
                amount=1,
            )

        self._report.record_budget(self._budget.snapshot)
        return tuple(nodes)

    def _register_node(self, node: DiscoveryNode) -> None:
        existing = self._history_index.get(node.fingerprint)
        if existing is None or node.score > existing.score:
            self._history_index[node.fingerprint] = node

        self._history_order = list(self._history_index.values())

    # ==================================================
    # INTERNAL SCORING
    # ==================================================

    def _combine_scores(
        self,
        *,
        novelty: float,
        diversity: float,
        heuristic: float,
    ) -> float:
        score = (
            0.40 * max(0.0, min(1.0, novelty))
            + 0.35 * max(0.0, min(1.0, diversity))
            + 0.25 * max(0.0, min(1.0, heuristic))
        )
        return max(0.0, min(1.0, score))

    def _rank_nodes(
        self,
        nodes: Iterable[DiscoveryNode],
    ) -> tuple[DiscoveryNode, ...]:
        ranked = sorted(
            nodes,
            key=lambda node: (
                -node.score,
                node.depth,
                node.fingerprint,
            ),
        )
        return tuple(ranked)

    def _frontier_signature(
        self,
        frontier: Iterable[DiscoveryNode],
    ) -> tuple[str, ...]:
        return tuple(node.fingerprint for node in frontier)

    # ==================================================
    # PYTHON PROTOCOL
    # ==================================================

    def __contains__(self, item: Hypothesis | str | DiscoveryNode) -> bool:
        if isinstance(item, DiscoveryNode):
            return item.fingerprint in self._history_index
        if isinstance(item, Hypothesis):
            return fingerprint_model(item) in self._history_index
        if isinstance(item, str):
            return item in self._history_index
        return False

    def __len__(self) -> int:
        return len(self._history_index)

    def __iter__(self):
        return iter(self._frontier)

    def __repr__(self) -> str:
        return (
            "Explorer("
            f"frontier={len(self._frontier)}, "
            f"history={len(self._history_index)}, "
            f"iterations={self._iterations}"
            ")"
        )