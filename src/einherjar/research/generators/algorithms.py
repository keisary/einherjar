"""generators/algorithms.py — Les 5-6 générateurs candidats (UN fichier par convention).

Tous les générateurs implémentent `BaseGenerator.generate(protocol) -> list[Hypothesis]`.
Le choix du générateur final V1 se fait APRÈS la comparaison empirique
(étape 2 du pipeline), pas avant.

Candidats implémentés :
  - RandomSearchGenerator  : random search sous contraintes (typage, profondeur).
  - BeamSearchGenerator    : beam search à profondeur fixe, K=64 par défaut.
  - TypedGPGenerator       : Strongly-Typed GP (sans BNF, types explicites).
  - GrammaticalEvolutionGenerator : GE — nécessite une grammaire BNF (placeholder).
  - MemeticGenerator       : EA + local search (placeholder : GE + hill climbing).
  - NSGA2Generator         : NSGA-II multi-objectif (placeholder : métrique composite).

Conforme à ALGORITHME_RESEARCH.md § 10.2 étape 2.
"""

from __future__ import annotations

import logging
import random
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from einherjar.research.config.loader import EinherjarConfig
from einherjar.research.generators.protocol import GenerationProtocol
from einherjar.research.utils.types import (
    Amplitude,
    AmplitudeUnit,
    CompareOp,
    Condition,
    ConditionNode,
    Direction,
    FeatureType,
    Hypothesis,
    LogicalOp,
    Universe,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Sortie commune
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GeneratorResult:
    """Sortie d'un générateur."""

    generator_name: str
    hypotheses: tuple[Hypothesis, ...]
    n_generated: int
    n_evaluated: int
    n_passed_admission: int
    generation_time_s: float
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generator_name": self.generator_name,
            "n_generated": self.n_generated,
            "n_evaluated": self.n_evaluated,
            "n_passed_admission": self.n_passed_admission,
            "generation_time_s": round(self.generation_time_s, 3),
            "meta": self.meta,
        }


class BaseGenerator(ABC):
    """Interface commune à tous les générateurs.

    Attributes:
        protocol: Protocole de génération (seed, budget, splits, etc.).
    """

    def __init__(self, protocol: GenerationProtocol) -> None:
        self.protocol = protocol
        self._rng = random.Random(protocol.seed)
        self.name: str = type(self).__name__
        logger.info("Générateur instancié : %s (seed=%d)", self.name, protocol.seed)

    @abstractmethod
    def generate(self) -> GeneratorResult:
        """Génère les hypothèses sous le protocole."""
        raise NotImplementedError

    def _make_amplitude(self, direction: Direction) -> Amplitude:
        return Amplitude(
            valeur=self.protocol.amplitude_value,
            unité=AmplitudeUnit.PRICE_ABSOLU,
            direction_implicite=direction,
        )

    def _make_universe(self) -> Universe:
        return Universe(assets=self.protocol.assets, timeframes=self.protocol.timeframes)


# --------------------------------------------------------------------------- #
# Generateur 1 : Random Search
# --------------------------------------------------------------------------- #


class RandomSearchGenerator(BaseGenerator):
    """Random search sous contraintes (typage, profondeur, ratios)."""

    def __init__(
        self,
        protocol: GenerationProtocol,
        config: EinherjarConfig,
    ) -> None:
        super().__init__(protocol)
        self.config = config

    def generate(self) -> GeneratorResult:
        import time
        t0 = time.time()
        continuous = [
            f for f in self.config.usable_feature_names
            if self._feature_type(f) in (FeatureType.ATOMIC, FeatureType.QUANTITATIVE, FeatureType.FACTOR)
        ]
        hyps: list[Hypothesis] = []
        i = 0
        while i < self.protocol.n_eval_budget:
            direction = self._rng.choice([Direction.LONG, Direction.SHORT])
            n_cond = self._rng.randint(1, self.protocol.max_conditions)
            if n_cond == 1 or self._rng.random() > self.protocol.p_compound:
                cond = self._sample_atomic(continuous)
            else:
                left = self._sample_atomic(continuous)
                right = self._sample_atomic(continuous)
                cond = ConditionNode(op=LogicalOp.AND, left=left, right=right)
            h = Hypothesis(
                id=f"{self.name}_{i:06d}",
                condition_tree=cond,
                amplitude=self._make_amplitude(direction),
                direction=direction,
                universe=self._make_universe(),
                cooldown_k=self.protocol.cooldown_k,
            )
            hyps.append(h)
            i += 1
        return GeneratorResult(
            generator_name=self.name,
            hypotheses=tuple(hyps),
            n_generated=len(hyps),
            n_evaluated=0,
            n_passed_admission=0,
            generation_time_s=time.time() - t0,
            meta={"budget_used": len(hyps)},
        )

    def _sample_atomic(self, pool: Sequence[str]) -> Condition:
        feat = self._rng.choice(pool)
        op = self._rng.choice([CompareOp.LT, CompareOp.GT])
        value = round(self._rng.uniform(-2.0, 2.0), 4)
        return Condition(feature_ref=feat, operator=op, value=value, transformation=None)

    def _feature_type(self, name: str) -> FeatureType | None:
        info = self.config.features_taxonomy.get("features", {}).get(name, {})
        type_str = info.get("feature_type")
        try:
            return FeatureType(type_str) if type_str else None
        except ValueError:
            return None


# --------------------------------------------------------------------------- #
# Generateur 2 : Beam Search
# --------------------------------------------------------------------------- #


class BeamSearchGenerator(BaseGenerator):
    """Beam search à profondeur fixe (1-2 conditions), K=64 par défaut.

    Maintient les K meilleurs candidats à chaque niveau, expand level by level.
    Évalue chaque candidat (via callback), garde le top K.
    """

    def __init__(
        self,
        protocol: GenerationProtocol,
        config: EinherjarConfig,
        beam_width: int = 64,
        depth: int = 2,
    ) -> None:
        super().__init__(protocol)
        self.config = config
        self.beam_width = beam_width
        self.depth = depth

    def generate(self) -> GeneratorResult:
        import time
        t0 = time.time()
        # Beam initial : K features × 2 directions × 2 operators × n_thresholds.
        # Pour V1, on génère juste le beam initial, sans expansion.
        continuous = [
            f for f in self.config.usable_feature_names
            if self._feature_type(f) in (FeatureType.ATOMIC, FeatureType.QUANTITATIVE, FeatureType.FACTOR)
        ]
        hyps: list[Hypothesis] = []
        i = 0
        for feat in continuous[: self.beam_width]:
            for direction in (Direction.LONG, Direction.SHORT):
                for op in (CompareOp.LT, CompareOp.GT):
                    for v in (0.0,):
                        h = Hypothesis(
                            id=f"{self.name}_{i:06d}",
                            condition_tree=Condition(
                                feature_ref=feat, operator=op, value=v, transformation=None,
                            ),
                            amplitude=self._make_amplitude(direction),
                            direction=direction,
                            universe=self._make_universe(),
                            cooldown_k=self.protocol.cooldown_k,
                        )
                        hyps.append(h)
                        i += 1
                        if i >= self.protocol.n_eval_budget:
                            break
                    if i >= self.protocol.n_eval_budget:
                        break
                if i >= self.protocol.n_eval_budget:
                    break
            if i >= self.protocol.n_eval_budget:
                break
        return GeneratorResult(
            generator_name=self.name,
            hypotheses=tuple(hyps),
            n_generated=len(hyps),
            n_evaluated=0,
            n_passed_admission=0,
            generation_time_s=time.time() - t0,
            meta={"beam_width": self.beam_width, "depth": self.depth},
        )

    def _feature_type(self, name: str) -> FeatureType | None:
        info = self.config.features_taxonomy.get("features", {}).get(name, {})
        type_str = info.get("feature_type")
        try:
            return FeatureType(type_str) if type_str else None
        except ValueError:
            return None


# --------------------------------------------------------------------------- #
# Generateur 3 : Strongly-Typed GP (sans BNF)
# --------------------------------------------------------------------------- #


class TypedGPGenerator(BaseGenerator):
    """Strongly-Typed Genetic Programming (sans grammaire BNF).

    Chaque noeud de l'arbre est typé (feature, op, value). Mutation et
    crossover respectent les types. Pour V1, on implémente juste l'initialisation
    (grow + full), pas l'évolution complète (à brancher sur DEAP ou ECJ).
    """

    def __init__(
        self,
        protocol: GenerationProtocol,
        config: EinherjarConfig,
        population_size: int = 200,
    ) -> None:
        super().__init__(protocol)
        self.config = config
        self.population_size = population_size

    def generate(self) -> GeneratorResult:
        import time
        t0 = time.time()
        continuous = [
            f for f in self.config.usable_feature_names
            if self._feature_type(f) in (FeatureType.ATOMIC, FeatureType.QUANTITATIVE, FeatureType.FACTOR)
        ]
        hyps: list[Hypothesis] = []
        for i in range(min(self.population_size, self.protocol.n_eval_budget)):
            direction = self._rng.choice([Direction.LONG, Direction.SHORT])
            tree = self._grow_tree(continuous, max_depth=self.protocol.max_conditions)
            h = Hypothesis(
                id=f"{self.name}_{i:06d}",
                condition_tree=tree,
                amplitude=self._make_amplitude(direction),
                direction=direction,
                universe=self._make_universe(),
                cooldown_k=self.protocol.cooldown_k,
            )
            hyps.append(h)
        return GeneratorResult(
            generator_name=self.name,
            hypotheses=tuple(hyps),
            n_generated=len(hyps),
            n_evaluated=0,
            n_passed_admission=0,
            generation_time_s=time.time() - t0,
            meta={"population_size": self.population_size, "method": "grow"},
        )

    def _grow_tree(self, pool: Sequence[str], max_depth: int) -> Condition | ConditionNode:
        if max_depth <= 1 or self._rng.random() < 0.5:
            return self._atom(pool)
        left = self._grow_tree(pool, max_depth - 1)
        right = self._grow_tree(pool, max_depth - 1)
        return ConditionNode(op=LogicalOp.AND, left=left, right=right)

    def _atom(self, pool: Sequence[str]) -> Condition:
        feat = self._rng.choice(pool)
        op = self._rng.choice([CompareOp.LT, CompareOp.GT])
        value = round(self._rng.uniform(-2.0, 2.0), 4)
        return Condition(feature_ref=feat, operator=op, value=value, transformation=None)

    def _feature_type(self, name: str) -> FeatureType | None:
        info = self.config.features_taxonomy.get("features", {}).get(name, {})
        type_str = info.get("feature_type")
        try:
            return FeatureType(type_str) if type_str else None
        except ValueError:
            return None


# --------------------------------------------------------------------------- #
# Generateur 4 : Grammatical Evolution (placeholder — BNF pas encore écrite)
# --------------------------------------------------------------------------- #


class GrammaticalEvolutionGenerator(BaseGenerator):
    """Grammatical Evolution (GE) — placeholder tant que la BNF n'est pas écrite.

    Pour V1, ce générateur est un placeholder qui refuse de tourner
    explicitement (NotImplementedError) tant que la grammaire BNF n'est
    pas fournie. Voir § 11.5 d'ALGORITHME_RESEARCH.md (BNF à écrire).
    """

    def __init__(self, protocol: GenerationProtocol, bnf_grammar: str | None = None) -> None:
        super().__init__(protocol)
        self.bnf_grammar = bnf_grammar

    def generate(self) -> GeneratorResult:
        if not self.bnf_grammar:
            logger.warning(
                "%s : grammaire BNF absente — retourne un résultat vide. "
                "Voir ALGORITHME_RESEARCH.md § 11.5 (BNF à écrire).",
                self.name,
            )
            return GeneratorResult(
                generator_name=self.name,
                hypotheses=(),
                n_generated=0,
                n_evaluated=0,
                n_passed_admission=0,
                generation_time_s=0.0,
                meta={"bnf_grammar_provided": False, "note": "placeholder"},
            )
        raise NotImplementedError(
            "GE complète non implémentée — BNF fournie mais mapping chromosome→arbre à coder."
        )


# --------------------------------------------------------------------------- #
# Generateur 5 : Memetic (EA + local search) — placeholder
# --------------------------------------------------------------------------- #


class MemeticGenerator(BaseGenerator):
    """Memetic : EA + local search — placeholder V1.

    Pour V1, on délègue à TypedGPGenerator (EA) + hill climbing minimal.
    Le hill climbing : pour chaque hypothèse, on mute un paramètre et on garde
    si le critère (passé en callback) s'améliore. Le callback n'est pas branché
    ici — c'est l'affaire du comparator de le faire.
    """

    def __init__(
        self,
        protocol: GenerationProtocol,
        config: EinherjarConfig,
        population_size: int = 200,
    ) -> None:
        super().__init__(protocol)
        self.config = config
        self.population_size = population_size
        self._ea = TypedGPGenerator(protocol, config, population_size=population_size)

    def generate(self) -> GeneratorResult:
        # V1 : on délègue à l'EA. Le local search sera branché dans le comparator.
        result = self._ea.generate()
        return GeneratorResult(
            generator_name=self.name,
            hypotheses=result.hypotheses,
            n_generated=result.n_generated,
            n_evaluated=result.n_evaluated,
            n_passed_admission=result.n_passed_admission,
            generation_time_s=result.generation_time_s,
            meta={"delegated_to": "TypedGPGenerator", "local_search": "deferred_to_comparator"},
        )


# --------------------------------------------------------------------------- #
# Generateur 6 : NSGA-II — placeholder
# --------------------------------------------------------------------------- #


class NSGA2Generator(BaseGenerator):
    """NSGA-II multi-objectif — placeholder V1.

    NSGA-II nécessite une métrique composite stable pour fonctionner
    (retour vs drawdown vs Sharpe). Tant qu'elle n'est pas recalibrée
    empiriquement (S-3.4, § 7.11 d'ALGORITHME_RESEARCH.md), on délègue
    à TypedGPGenerator (single-objective) en attendant.
    """

    def __init__(
        self,
        protocol: GenerationProtocol,
        config: EinherjarConfig,
        population_size: int = 200,
    ) -> None:
        super().__init__(protocol)
        self.config = config
        self.population_size = population_size
        self._ea = TypedGPGenerator(protocol, config, population_size=population_size)

    def generate(self) -> GeneratorResult:
        result = self._ea.generate()
        return GeneratorResult(
            generator_name=self.name,
            hypotheses=result.hypotheses,
            n_generated=result.n_generated,
            n_evaluated=result.n_evaluated,
            n_passed_admission=result.n_passed_admission,
            generation_time_s=result.generation_time_s,
            meta={"delegated_to": "TypedGPGenerator", "note": "composite_metric_not_calibrated"},
        )


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #


def make_all_generators(
    protocol: GenerationProtocol,
    config: EinherjarConfig,
) -> list[BaseGenerator]:
    """Construit les 5 candidats principaux (sans GE — BNF pas encore écrite).

    Returns:
        Liste [RandomSearch, BeamSearch, TypedGP, Memetic, NSGA2].
    """
    return [
        RandomSearchGenerator(protocol=protocol, config=config),
        BeamSearchGenerator(protocol=protocol, config=config),
        TypedGPGenerator(protocol=protocol, config=config),
        MemeticGenerator(protocol=protocol, config=config),
        NSGA2Generator(protocol=protocol, config=config),
    ]
