"""generators/typedgp.py — Strongly-Typed Genetic Programming (STGP) — Phase 1.

Extraction propre depuis l'ancien generators/algorithms.py. Seul générateur
du système. Contient :
  - TypedGPGenerator (Koza 1992 + Montana 1995)
  - BaseGenerator (classe parente commune)
  - GeneratorResult (sortie normalisée)
  - Helpers (seuils calibrés, tasting, fitness CROISSANCE)

Conforme à ALGORITHME_RESEARCH.md § 10.2.
"""

from __future__ import annotations

import logging
import math
import random
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from einherjar.research.config.loader import EinherjarConfig
from einherjar.research.utils.stats import periods_per_year_for_timeframe
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
# Helpers
# --------------------------------------------------------------------------- #

def _value_type_of(config, name: str) -> str | None:
    """value_type ('float'|'boolean'|None) d'une feature via la taxonomie."""
    info = config.features_taxonomy.get("features", {}).get(name, {})
    vt = info.get("value_type")
    return vt if vt in ("float", "boolean") else None


_NODE_CATEGORIES = ("atomic", "compound")


# --------------------------------------------------------------------------- #
# Sortie commune
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GeneratorResult:
    """Sortie normalisée d'un générateur."""

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


# --------------------------------------------------------------------------- #
# Classe parente — BaseGenerator
# --------------------------------------------------------------------------- #


class BaseGenerator(ABC):
    """Interface commune à tous les générateurs (ici uniquement TypedGP).

    Attributes:
        protocol: Protocole de génération (seed, budget, splits, etc.).
        engine: Moteur d'évaluation (requis pour TypedGP).
    """

    def __init__(
        self,
        config: EinherjarConfig | None = None,
        engine: Any | None = None,
    ) -> None:
        self.config = config
        self.engine = engine
        self._rng = random.Random(config.seed if config else 42)
        self.name: str = type(self).__name__
        self._taste_cache: tuple | None = None
        self._threshold_quantiles: dict[str, list[float]] | None = None
        self._train_ohlcv = None
        self._train_features = None
        self._val_ohlcv = None
        self._val_features = None
        self._search_timeframe = "1h"
        self._search_train_ohlcv = None
        self._search_train_features = None
        self._search_val_ohlcv = None
        self._search_val_features = None

    @abstractmethod
    def generate(self) -> GeneratorResult:
        """Génère les hypothèses sous la config."""
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Fitness CROISSANCE (alignée sur l'admission)
    # ------------------------------------------------------------------ #

    def _growth_fitness(
        self, m: Any,
        periods_per_year: float | None = None,
        min_trades: int | None = None,
        soft: bool = False,
    ) -> float:
        """log(1 + CAGR) annuel : objectif que l'admission va tester.

        Porte dure (soft=False) : n_signals < min_trades -> -inf.
        Porte douce (soft=True, beam interne) : pénalité multiplicative.
        """
        n = getattr(m, "n_signals", 0) or 0
        held = getattr(m, "avg_holding_period", 0.0) or 0.0
        ret_mean = getattr(m, "ret_mean_pct_net", float("nan"))
        if n <= 0 or held <= 0:
            return float("-inf")
        if ret_mean != ret_mean or ret_mean <= -1.0:
            return float("-inf")
        if periods_per_year is None:
            periods_per_year = periods_per_year_for_timeframe(self._search_timeframe or "1h")
        if min_trades is None:
            cfg_t = getattr(self, "config", None)
            min_trades = int((cfg_t.thresholds.get("n_trades", {}).get("min_total", 30) or 30)
                             if cfg_t is not None else 30)
        if n < min_trades:
            if soft:
                return (periods_per_year / held) * math.log1p(max(ret_mean, 1e-9)) * (n / min_trades)
            return float("-inf")
        return (periods_per_year / held) * math.log1p(ret_mean)

    # ------------------------------------------------------------------ #
    # Tasting
    # ------------------------------------------------------------------ #

    def _taste_frames(self, val_ohlcv: Any, val_features: Any, n_samples: int = 0) -> tuple[Any, Any]:
        """Sous-échantillonne la validation pour l'évolution (seedé)."""
        if n_samples <= 0:
            return val_ohlcv, val_features
        _cache = getattr(self, "_taste_cache", None)
        if _cache is not None and _cache[0] is val_ohlcv and _cache[1] == n_samples:
            return _cache[2], _cache[3]
        n_total = val_ohlcv.n_bougies
        if n_total <= n_samples:
            self._taste_cache = (val_ohlcv, n_samples, val_ohlcv, val_features)
            return val_ohlcv, val_features
        n_blocks = max(2, min(6, n_total // max(200, n_samples // 6)))
        block_size = max(200, n_samples // n_blocks)
        rng = random.Random(int(self.config.seed) ^ 0x7A57)
        usable = n_total - block_size
        starts = sorted(rng.sample(range(usable), n_blocks))
        import polars as pl
        from einherjar.research.data.features import FeaturesFrame
        from einherjar.research.data.ohlcv import OhlcvFrame
        ohlcv_slices = [val_ohlcv.df.slice(s, block_size) for s in starts]
        feat_slices = [val_features.df.slice(s, block_size) for s in starts]
        tasted_ohlcv = OhlcvFrame(
            asset=val_ohlcv.asset, timeframe=val_ohlcv.timeframe,
            df=pl.concat(ohlcv_slices), data_version=val_ohlcv.data_version,
        )
        tasted_features = FeaturesFrame(
            asset=val_features.asset, timeframe=val_features.timeframe,
            df=pl.concat(feat_slices), feature_names=val_features.feature_names,
            data_version=val_features.data_version,
        )
        self._taste_cache = (val_ohlcv, n_samples, tasted_ohlcv, tasted_features)
        return tasted_ohlcv, tasted_features

    def _make_amplitude(self, direction: Direction) -> Amplitude:
        return Amplitude(
            valeur=self.config.amplitude_value,
            unité=AmplitudeUnit.MULTIPLE_ATR,
            direction_implicite=direction,
        )

    def _make_universe(self) -> Universe:
        return Universe(assets=(self.config.asset,), timeframes=(self.config.timeframe,))

    def bind_data(self, train_ohlcv, train_features, val_ohlcv, val_features) -> None:
        """Lie les données avant la génération."""
        self._train_ohlcv = train_ohlcv
        self._train_features = train_features
        self._val_ohlcv = val_ohlcv
        self._val_features = val_features
        self._threshold_quantiles = None
        cut = max(1, int(train_ohlcv.n_bougies * 0.8))

        def _slice(frame: Any, start: int, end: int) -> Any:
            return type(frame)(
                asset=frame.asset, timeframe=frame.timeframe,
                df=frame.df.slice(start, end - start),
                **(dict(feature_names=frame.feature_names) if hasattr(frame, "feature_names") else {}),
                data_version=frame.data_version,
            )

        self._search_timeframe = getattr(train_ohlcv, "timeframe", "1h")
        self._search_train_ohlcv = _slice(train_ohlcv, 0, cut)
        self._search_train_features = _slice(train_features, 0, cut)
        self._search_val_ohlcv = _slice(train_ohlcv, cut, train_ohlcv.n_bougies)
        self._search_val_features = _slice(train_features, cut, train_features.n_bougies)

    # ------------------------------------------------------------------ #
    # Seuils calibrés (P1 #1)
    # ------------------------------------------------------------------ #

    _FALLBACK_THRESHOLD_POOL: tuple[float, ...] = (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)

    def _ensure_threshold_quantiles(self) -> dict[str, list[float]]:
        if self._threshold_quantiles is not None:
            return self._threshold_quantiles
        from einherjar.research.data.threshold_calibration import (
            compute_feature_quantiles, merge_quantile_pools,
        )
        train_features = getattr(self, "_search_train_features", None)
        if train_features is None:
            train_features = getattr(self, "_train_features", None)
        if train_features is None:
            self._threshold_quantiles = {
                name: list(self._FALLBACK_THRESHOLD_POOL)
                for name in self.config.usable_feature_names
            }
        else:
            raw = compute_feature_quantiles(train_features)
            self._threshold_quantiles = merge_quantile_pools(
                raw, fallback_pool=self._FALLBACK_THRESHOLD_POOL,
            )
        return self._threshold_quantiles

    def _sample_threshold_for(self, feature_name: str) -> float:
        pools = self._ensure_threshold_quantiles()
        pool = pools.get(feature_name) or list(self._FALLBACK_THRESHOLD_POOL)
        return float(self._rng.choice(pool))


# --------------------------------------------------------------------------- #
# Le cœur : TypedGPGenerator
# --------------------------------------------------------------------------- #


class TypedGPGenerator(BaseGenerator):
    """Strongly-Typed Genetic Programming (STGP, Koza 1992 + Montana 1995).

    Phase 1 : extraction propre, compteurs ajoutés, logique génétique conservée
    telle quelle (améliorations réservées à Phase 2).

    Cycle évolutif :
      1. Initialisation : 50% grow, 50% full
      2. Évaluation : fitness = CAGR log (alignée admission)
      3. Boucle : tournoi → crossover sous-arbre → mutation → élitisme
      4. Retour : population dédupliquée
    """

    def __init__(
        self,
        config: Any,
        engine: Any = None,
        population_size: int = 50,
        n_generations: int = 10,
        crossover_prob: float = 0.8,
        mutation_prob: float = 0.2,
        tournament_size: int = 3,
        elitism: int = 2,
    ) -> None:
        """Initialise TypedGP.

        Args:
            config: TypedGPConfig (contenant seed, max_depth, horizon_index, ...).
            engine: Moteur d'évaluation (REQUIS).
            population_size: Taille de la population.
            n_generations: Nombre de générations.
            crossover_prob: Probabilité de crossover par paire.
            mutation_prob: Probabilité de mutation par enfant.
            tournament_size: Taille du tournoi.
            elitism: Nombre de meilleurs individus préservés.
        """
        super().__init__(config, engine=engine)
        if engine is None:
            raise ValueError(
                "TypedGPGenerator requiert un moteur d'évaluation (engine=...)"
            )
        self.population_size = population_size
        self.n_generations = n_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.tournament_size = tournament_size
        self.elitism = elitism

        # Pools de features
        self._continuous_features: list[str] = [
            f for f in config.usable_feature_names
            if _value_type_of(config, f) == "float"
        ]
        self._pattern_features: list[str] = [
            f for f in config.usable_feature_names
            if _value_type_of(config, f) == "boolean"
        ]
        if not self._continuous_features and not self._pattern_features:
            raise ValueError("Aucune feature exploitable pour TypedGP")

        logger.info(
            "TypedGPGenerator : pop=%d, gen=%d, %d float + %d bool, "
            "crossover=%.2f, mutation=%.2f, tournament=%d, elitism=%d, "
            "horizon_index=%s",
            population_size, n_generations,
            len(self._continuous_features), len(self._pattern_features),
            crossover_prob, mutation_prob, tournament_size, elitism,
            getattr(config, "horizon_index", "N/A"),
        )

    def generate(self) -> GeneratorResult:
        """Lance l'évolution TypedGP et retourne la population finale."""
        import time
        t0 = time.time()
        max_depth = self.config.max_depth
        population: list[Hypothesis] = []

        # 1. Population initiale : 50% grow, 50% full
        for i in range(self.population_size):
            method = "grow" if i % 2 == 0 else "full"
            tree = self._init_tree(method=method, max_depth=max_depth)
            direction = self._rng.choice([Direction.LONG, Direction.SHORT])
            h = Hypothesis(
                id=f"{self.name}_{i:06d}",
                condition_tree=tree,
                amplitude=self._make_amplitude(direction),
                direction=direction,
                universe=self._make_universe(),
                cooldown_k=getattr(self.config, "cooldown_k", 5),
            )
            population.append(h)
        # Stats population initiale
        n_long = sum(1 for h in population if h.direction == Direction.LONG)
        n_short = sum(1 for h in population if h.direction == Direction.SHORT)
        depths = [self._tree_depth(h.condition_tree) for h in population]
        logger.info(
            "Population initiale : %d individus (Long=%d, Short=%d), "
            "profondeur moyenne=%.2f, max=%d",
            len(population), n_long, n_short,
            sum(depths) / max(1, len(depths)), max(depths),
        )

        # 2. Évaluation initiale
        fitness = self._evaluate_population(population)
        logger.info(
            "Fitness initiale : validés=%d, -inf=%d",
            sum(1 for f in fitness if f != float("-inf")),
            sum(1 for f in fitness if f == float("-inf")),
        )

        # 3. Boucle évolutionnaire
        for gen in range(self.n_generations):
            parents = self._tournament_selection(population, fitness, n=self.population_size)
            offspring: list[Hypothesis] = []
            for i in range(0, len(parents) - 1, 2):
                p1, p2 = parents[i], parents[i + 1]
                if self._rng.random() < self.crossover_prob:
                    c1, c2 = self._subtree_crossover(p1, p2)
                else:
                    c1, c2 = p1, p2
                c1 = self._subtree_mutation(c1)
                c2 = self._subtree_mutation(c2)
                offspring.append(c1)
                offspring.append(c2)
            if len(offspring) < self.population_size:
                offspring.append(self._subtree_mutation(parents[-1]))
            offspring = offspring[: self.population_size]

            train_ohlcv = getattr(self, "_search_train_ohlcv", None)
            train_features = getattr(self, "_search_train_features", None)
            val_ohlcv = getattr(self, "_search_val_ohlcv", None)
            val_features = getattr(self, "_search_val_features", None)
            offspring_fitness = self._evaluate_population(
                offspring, train_ohlcv, train_features, val_ohlcv, val_features,
            )

            union_pop = population + offspring
            union_fit = fitness + offspring_fitness
            order = sorted(range(len(union_pop)), key=lambda i: union_fit[i], reverse=True)
            order = order[: self.population_size]
            population = [union_pop[i] for i in order]
            fitness = [union_fit[i] for i in order]

            n_valid = sum(1 for f in fitness if f != float("-inf"))
            logger.info(
                "Gen %d/%d : best=%.4f, mean=%.4f, valides=%d/%d",
                gen + 1, self.n_generations,
                fitness[0] if fitness else -1,
                sum(f for f in fitness if f != float("-inf")) / max(1, n_valid),
                n_valid, len(fitness),
            )

        # 4. Déduplication
        seen: set[tuple] = set()
        unique: list[Hypothesis] = []
        for h in population:
            sig = (h.condition_tree, h.direction, h.cooldown_k)
            if sig not in seen:
                seen.add(sig)
                unique.append(h)

        # Stats finales
        n_long_final = sum(1 for h in unique if h.direction == Direction.LONG)
        n_short_final = sum(1 for h in unique if h.direction == Direction.SHORT)
        depths_final = [self._tree_depth(h.condition_tree) for h in unique]
        # Features les plus utilisées
        feature_usage: dict[str, int] = {}
        for h in unique:
            for f in self._collect_features(h.condition_tree):
                feature_usage[f] = feature_usage.get(f, 0) + 1
        top_features = sorted(feature_usage.items(), key=lambda x: -x[1])[:10]
        logger.info(
            "Génération terminée : %d uniques (Long=%d, Short=%d), "
            "profondeur moy=%.2f, max=%d, top features: %s",
            len(unique), n_long_final, n_short_final,
            sum(depths_final) / max(1, len(depths_final)),
            max(depths_final) if depths_final else 0,
            top_features,
        )
        logger.info(
            "Horizon config : index=%s, timeframe=%s",
            getattr(self.config, "horizon_index", "N/A"),
            getattr(self, "_search_timeframe", "?"),
        )

        return GeneratorResult(
            generator_name=self.name,
            hypotheses=tuple(unique),
            n_generated=len(population),
            n_evaluated=len(unique),
            n_passed_admission=0,
            generation_time_s=time.time() - t0,
            meta={
                "method": "TypedGP-Koza+Montana",
                "n_generations": self.n_generations,
                "population_size": self.population_size,
                "init_methods": ("grow", "full"),
                "crossover": "subtree_type_preserving",
                "mutation": "subtree_regrow",
                "selection": f"tournament_k={self.tournament_size}",
                "horizon_index": str(getattr(self.config, "horizon_index", "N/A")),
                "timeframe": getattr(self, "_search_timeframe", "?"),
            },
        )

    # ------------------------------------------------------------------ #
    # Initialisation (Koza : grow + full)
    # ------------------------------------------------------------------ #

    def _init_tree(self, method: str, max_depth: int) -> Condition | ConditionNode:
        if method == "grow":
            return self._grow(max_depth=max_depth, depth=0)
        return self._full(max_depth=max_depth, depth=0)

    def _grow(self, max_depth: int, depth: int) -> Condition | ConditionNode:
        if depth >= max_depth or (depth > 0 and self._rng.random() < 0.5):
            return self._atom()
        op = self._rng.choice(list(LogicalOp))
        if op == LogicalOp.NOT:
            child = self._grow(max_depth=max_depth, depth=depth + 1)
            return ConditionNode(op=op, left=child, right=None)
        left = self._grow(max_depth=max_depth, depth=depth + 1)
        right = self._grow(max_depth=max_depth, depth=depth + 1)
        return ConditionNode(op=op, left=left, right=right)

    def _full(self, max_depth: int, depth: int) -> Condition | ConditionNode:
        if depth >= max_depth:
            return self._atom()
        op = self._rng.choice(list(LogicalOp))
        if op == LogicalOp.NOT:
            child = self._full(max_depth=max_depth, depth=depth + 1)
            return ConditionNode(op=op, left=child, right=None)
        left = self._full(max_depth=max_depth, depth=depth + 1)
        right = self._full(max_depth=max_depth, depth=depth + 1)
        return ConditionNode(op=op, left=left, right=right)

    def _atom(self) -> Condition:
        """Crée une feuille : feature + op + valeur."""
        if self._pattern_features and self._rng.random() < 0.35:
            feat = self._rng.choice(self._pattern_features)
            op = self._rng.choice([CompareOp.EQ, CompareOp.NE])
            value = float(self._rng.randint(0, 1))
            return Condition(feature_ref=feat, operator=op, value=value, transformation=None)
        feat = self._rng.choice(self._continuous_features)
        op = self._rng.choice([CompareOp.LT, CompareOp.GT, CompareOp.LE, CompareOp.GE])
        value = round(self._sample_threshold_for(feat), 4)
        return Condition(feature_ref=feat, operator=op, value=value, transformation=None)

    def _tree_depth(self, node: Condition | ConditionNode) -> int:
        """Calcule la profondeur d'un arbre de conditions."""
        if isinstance(node, Condition):
            return 1
        left_depth = self._tree_depth(node.left)
        right_depth = self._tree_depth(node.right) if node.right is not None else 0
        return 1 + max(left_depth, right_depth)

    def _collect_features(self, node: Condition | ConditionNode) -> list[str]:
        """Collecte tous les noms de features utilisés dans un arbre."""
        if isinstance(node, Condition):
            return [node.feature_ref]
        result = self._collect_features(node.left)
        if node.right is not None:
            result += self._collect_features(node.right)
        return result

    # ------------------------------------------------------------------ #
    # Évaluation de la fitness (CROISSANCE)
    # ------------------------------------------------------------------ #

    def _evaluate_population(
        self,
        population: list[Hypothesis],
        train_ohlcv: Any | None = None,
        train_features: Any | None = None,
        val_ohlcv: Any | None = None,
        val_features: Any | None = None,
    ) -> list[float]:
        """Évalue la fitness CROISSANCE de chaque individu via engine.

        Retourne une liste de floats (-inf si invalide).
        """
        train_ohlcv = train_ohlcv or getattr(self, "_search_train_ohlcv", None)
        train_features = train_features or getattr(self, "_search_train_features", None)
        val_ohlcv = val_ohlcv or getattr(self, "_search_val_ohlcv", None)
        val_features = val_features or getattr(self, "_search_val_features", None)
        if train_ohlcv is None or val_ohlcv is None:
            return [float("-inf")] * len(population)
        n_samples = getattr(self.config, "taste_samples", 0) or 0
        if n_samples > 0:
            val_ohlcv, val_features = self._taste_frames(val_ohlcv, val_features, n_samples)
        min_trades = int(
            (self.config.thresholds.get("n_trades", {}).get("min_total", 30) or 30)
            if hasattr(self.config, "thresholds") and self.config.thresholds
            else 30
        )
        timeframe = getattr(val_ohlcv, "timeframe", "1h")
        periods_per_year = periods_per_year_for_timeframe(timeframe)
        fitness: list[float] = []
        for h in population:
            try:
                calibrated = self.engine.train_calibrate(h, train_ohlcv, train_features)
                m = self.engine.test_on(
                    h, val_ohlcv, val_features, calibrated, "val",
                    with_bootstrap=False,
                )
                fitness.append(self._growth_fitness(m, periods_per_year, min_trades))
            except Exception as exc:
                logger.warning("  Fitness -inf pour %s : %s", h.id, exc)
                fitness.append(float("-inf"))
        return fitness

    # ------------------------------------------------------------------ #
    # Sélection par tournoi
    # ------------------------------------------------------------------ #

    def _tournament_selection(
        self, population: list[Hypothesis], fitness: list[float], n: int,
    ) -> list[Hypothesis]:
        if not population:
            return []
        selected: list[Hypothesis] = []
        for _ in range(n):
            indices = self._rng.sample(
                range(len(population)), min(self.tournament_size, len(population)),
            )
            best_i = max(
                indices,
                key=lambda i: (fitness[i] if fitness[i] == fitness[i] else float("-inf")),
            )
            selected.append(population[best_i])
        return selected

    # ------------------------------------------------------------------ #
    # Crossover sous-arbre (type-preserving)
    # ------------------------------------------------------------------ #

    def _subtree_crossover(
        self, p1: Hypothesis, p2: Hypothesis,
    ) -> tuple[Hypothesis, Hypothesis]:
        nodes1 = self._collect_nodes_by_category(p1.condition_tree)
        nodes2 = self._collect_nodes_by_category(p2.condition_tree)
        common = [c for c in _NODE_CATEGORIES if nodes1[c] and nodes2[c]]
        if not common:
            return p1, p2
        cat = self._rng.choice(common)
        path1 = self._rng.choice(nodes1[cat])
        path2 = self._rng.choice(nodes2[cat])
        new_tree1 = self._swap_subtree(p1.condition_tree, path1, p2.condition_tree, path2)
        new_tree2 = self._swap_subtree(p2.condition_tree, path2, p1.condition_tree, path1)
        return (
            self._clone_with_tree(p1, new_tree1),
            self._clone_with_tree(p2, new_tree2),
        )

    def _collect_nodes_by_category(
        self, tree: Condition | ConditionNode,
    ) -> dict[str, list[list[bool]]]:
        result: dict[str, list[list[bool]]] = {"atomic": [], "compound": []}

        def _walk(node: Condition | ConditionNode, path: list[bool]) -> None:
            if isinstance(node, Condition):
                result["atomic"].append(path)
                return
            result["compound"].append(path)
            _walk(node.left, path + [False])
            if node.right is not None:
                _walk(node.right, path + [True])

        _walk(tree, [])
        return result

    def _swap_subtree(
        self, tree: Condition | ConditionNode, path: list[bool],
        donor: Condition | ConditionNode, donor_path: list[bool],
    ) -> Condition | ConditionNode:
        donor_subtree = self._get_subtree(donor, donor_path)
        if not path:
            return donor_subtree
        return self._replace_subtree(tree, path, donor_subtree)

    def _get_subtree(
        self, node: Condition | ConditionNode, path: list[bool],
    ) -> Condition | ConditionNode:
        if not path:
            return node
        if isinstance(node, Condition):
            return node
        if path[0]:
            assert node.right is not None
            return self._get_subtree(node.right, path[1:])
        return self._get_subtree(node.left, path[1:])

    def _replace_subtree(
        self, node: Condition | ConditionNode, path: list[bool],
        new_subtree: Condition | ConditionNode,
    ) -> Condition | ConditionNode:
        import copy
        if not path:
            return copy.deepcopy(new_subtree)
        if isinstance(node, Condition):
            return node
        if path[0]:
            new_right = self._replace_subtree(node.right, path[1:], new_subtree) if node.right else None
            return ConditionNode(op=node.op, left=copy.deepcopy(node.left), right=new_right)
        new_left = self._replace_subtree(node.left, path[1:], new_subtree)
        return ConditionNode(op=node.op, left=new_left, right=copy.deepcopy(node.right))

    # ------------------------------------------------------------------ #
    # Mutation sous-arbre
    # ------------------------------------------------------------------ #

    def _subtree_mutation(self, h: Hypothesis) -> Hypothesis:
        if self._rng.random() > self.mutation_prob:
            return h
        nodes = self._collect_nodes_by_category(h.condition_tree)
        cat = self._rng.choice(_NODE_CATEGORIES)
        if not nodes[cat]:
            return h
        path = self._rng.choice(nodes[cat])
        max_depth = self.config.max_depth
        if cat == "atomic":
            new_sub = self._atom()
        else:
            depth_remaining = max(1, max_depth - len(path))
            new_sub = self._grow(max_depth=depth_remaining, depth=0)
        new_tree = self._swap_subtree(h.condition_tree, path, new_sub, [])
        return self._clone_with_tree(h, new_tree)

    def _clone_with_tree(
        self, h: Hypothesis, new_tree: Condition | ConditionNode,
    ) -> Hypothesis:
        import copy
        return Hypothesis(
            id=f"{h.id}_x{self._rng.randint(0, 9999):04d}",
            condition_tree=copy.deepcopy(new_tree),
            amplitude=h.amplitude,
            direction=h.direction,
            universe=h.universe,
            cooldown_k=h.cooldown_k,
        )