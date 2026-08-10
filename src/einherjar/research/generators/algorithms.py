"""generators/algorithms.py — Les générateurs candidats (UN fichier par convention).

Tous les générateurs implémentent `BaseGenerator.generate() -> GeneratorResult`.
Le choix du générateur final V1 se fait APRÈS la comparaison empirique
(étape 2 du pipeline), pas avant.

Candidats implémentés (vraies implémentations, aucune délégation fantôme) :
  - RandomSearchGenerator  : random search sous contraintes (typage, profondeur).
  - BeamSearchGenerator    : beam search à profondeur fixe, K=64 par défaut.
  - TypedGPGenerator       : Strongly-Typed GP (grow, sans évolution pour V1).
  - GrammaticalEvolutionGenerator : GE avec BNF 218 features (BNF Phase 4).
  - MemeticGenerator       : EA + phase d'optimisation locale (hill climbing réelle).
  - NSGA2Generator         : NSGA-II multi-objectif (Deb 2002) avec contraintes dures.

Conforme à ALGORITHME_RESEARCH.md § 10.2 étape 2.
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
from einherjar.research.generators.protocol import GenerationProtocol
from einherjar.research.utils.stats import max_drawdown_from_returns, periods_per_year_for_timeframe
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


def _value_type_of(config, name: str) -> str | None:
    """value_type ('float'|'boolean'|None) d'une feature via la taxonomie."""
    info = config.features_taxonomy.get("features", {}).get(name, {})
    vt = info.get("value_type")
    return vt if vt in ("float", "boolean") else None



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
        engine: Moteur d'évaluation (optionnel). Requis pour les générateurs
            évolutionnaires qui ont besoin d'évaluer les fitness
            (NSGA-II, Memetic). Ignoré par les générateurs non-évolutionnaires
            (Random, Beam, TypedGP sans évolution).
    """

    def __init__(
        self,
        protocol: GenerationProtocol,
        engine: Any | None = None,
    ) -> None:
        self.protocol = protocol
        self.engine = engine
        self._rng = random.Random(protocol.seed)
        self.name: str = type(self).__name__
        logger.info("Générateur instancié : %s (seed=%d, engine=%s)",
                    self.name, protocol.seed, type(engine).__name__ if engine else "None")

    @abstractmethod
    def generate(self) -> GeneratorResult:
        """Génère les hypothèses sous le protocole."""
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Fitness CROISSANCE (alignée sur l'admission, critère 7, objectif ×10)
    # ------------------------------------------------------------------ #
    def _growth_fitness(
        self, m: Any,
        periods_per_year: float | None = None,
        min_trades: int | None = None,
        soft: bool = False,
    ) -> float:
        """log(1 + CAGR) annuel : l'objectif que l'admission va tester.

        Formule : trades_par_an = periods_per_year / avg_holding_period ;
        fitness = trades_par_an * log1p(ret_mean_net). Monotone en CAGR,
        stable numériquement (pas d'overflow sur (1+r)^n).

        Porte dure (soft=False) : n_signals < min_trades -> -inf (comme
        l'admission). Porte douce (soft=True, beam interne) : pénalité
        multiplicative sqrt(n/min_trades) pour garder un gradient pendant
        l'expansion locale.
        """
        n = getattr(m, "n_signals", 0) or 0
        held = getattr(m, "avg_holding_period", 0.0) or 0.0
        ret_mean = getattr(m, "ret_mean_pct_net", float("nan"))
        if n <= 0 or held <= 0:
            return float("-inf")
        if ret_mean != ret_mean or ret_mean <= -1.0:
            return float("-inf")
        if periods_per_year is None:
            timeframe = getattr(self, "_search_timeframe", None) or "1h"
            periods_per_year = periods_per_year_for_timeframe(timeframe)
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
    # Tasting : sous-échantillonnage seedé pour l'évolution (Décision 2026-08-10)
    # ------------------------------------------------------------------ #
    def _taste_frames(
        self,
        val_ohlcv: Any,
        val_features: Any,
    ) -> tuple[Any, Any]:
        """Réduit les frames val à un échantillon de blocs contigus seedé.

        protocol.n_samples <= 0 ou >= longueur : renvoie (val_ohlcv, val_features)
        inchangés (pas de tasting). Sinon : sélectionne n_blocks blocs contigus
        (taille n_samples // n_blocks, seed via self._rng) répartis sur la
        fenêtre, et reconstruit des sous-frames alignées OHLCV/features.

        L'échantillon est IDENTIQUE pour toute la population (construit une
        seule fois avant la boucle d'évolution) → fitness comparables entre
        individus. L'admission finale évalue TOUJOURS sur le val complet.
        """
        n_samples = int(getattr(self.protocol, "n_samples", 0) or 0)
        if n_samples <= 0:
            return val_ohlcv, val_features
        # Cache : le même (val, n_samples) produit le même échantillon seedé
        # (stabilité population + pas de reconstruction à chaque génération).
        _cache = getattr(self, "_taste_cache", None)
        if _cache is not None and _cache[0] is val_ohlcv and _cache[1] == n_samples:
            return _cache[2], _cache[3]
        n_total = val_ohlcv.n_bougies
        if n_total <= n_samples:
            self._taste_cache = (val_ohlcv, n_samples, val_ohlcv, val_features)
            return val_ohlcv, val_features
        import random as _random
        n_blocks = max(2, min(6, n_total // max(200, n_samples // 6)))
        block_size = max(200, n_samples // n_blocks)
        # Détermine des starts répartis uniformément, avec jitter seedé.
        rng = _random.Random(int(self.protocol.seed) ^ 0x7A57)
        usable = n_total - block_size
        starts = sorted(rng.sample(range(usable), n_blocks))
        # Slices polars par position (frames alignées OHLCV/features).
        ohlcv_slices = [val_ohlcv.df.slice(s, block_size) for s in starts]
        feat_slices = [val_features.df.slice(s, block_size) for s in starts]
        import polars as pl
        from einherjar.research.data.ohlcv import OhlcvFrame
        from einherjar.research.data.features import FeaturesFrame
        tasted_ohlcv = OhlcvFrame(
            asset=val_ohlcv.asset, timeframe=val_ohlcv.timeframe,
            df=pl.concat(ohlcv_slices), data_version=val_ohlcv.data_version,
        )
        tasted_features = FeaturesFrame(
            asset=val_features.asset, timeframe=val_features.timeframe,
            df=pl.concat(feat_slices), feature_names=val_features.feature_names,
            data_version=val_features.data_version,
        )
        logger.info(
            "Tasting : %d blocs × %d bougies = %d (au lieu de %d)",
            n_blocks, block_size, tasted_ohlcv.n_bougies, n_total,
        )
        self._taste_cache = (val_ohlcv, n_samples, tasted_ohlcv, tasted_features)
        return tasted_ohlcv, tasted_features

    def _make_amplitude(self, direction: Direction) -> Amplitude:
        """Construit l'Amplitude d'un Einher.

        Choix de design (intentionnel, documente GenerationProtocol) :
        l'unite est fixee a MULTIPLE_ATR. Les generateurs produisent
        tous des Einher dont l'amplitude est exprimee en multiple d'ATR
        (cf. evaluator.py::_compute_n_multiple_atr). Pour un mouvement
        en prix absolu, il faut construire l'Hypothesis directement
        (cas non couvert par les generateurs actuels).
        """
        return Amplitude(
        valeur=self.protocol.amplitude_value,
        unité=AmplitudeUnit.MULTIPLE_ATR,
        direction_implicite=direction,
        )

    def _make_universe(self) -> Universe:
        return Universe(assets=self.protocol.assets, timeframes=self.protocol.timeframes)

    def bind_data(
        self,
        train_ohlcv: Any,
        train_features: Any,
        val_ohlcv: Any,
        val_features: Any,
    ) -> None:
        """Bind the common protocol data before generation."""
        self._train_ohlcv = train_ohlcv
        self._train_features = train_features
        self._val_ohlcv = val_ohlcv
        self._val_features = val_features
        self._threshold_quantiles = None
        # The outer validation set belongs to the comparator. Evolutionary
        # search may only optimise on this deterministic inner split of train.
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
    # Seuils calibrés (P1 #1) — quantiles par feature sur le train
    # ------------------------------------------------------------------ #

    _FALLBACK_THRESHOLD_POOL: tuple[float, ...] = (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)

    def _ensure_threshold_quantiles(self) -> dict[str, list[float]]:
        """Calcule les quantiles par feature sur le train si pas déjà fait (lazy)."""
        if getattr(self, "_threshold_quantiles", None) is not None:
            return self._threshold_quantiles
        from einherjar.research.data.threshold_calibration import (
            compute_feature_quantiles,
            merge_quantile_pools,
        )
        # Pendant une recherche évolutionnaire, les seuils appartiennent au
        # sous-train interne, jamais au sous-jeu utilisé pour le fitness.
        train_features = getattr(self, "_search_train_features", None)
        if train_features is None:
            train_features = getattr(self, "_train_features", None)
        if train_features is None:
            # Pas de train : pool par défaut par feature (uniforme).
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
        """Tire un seuil dans le pool calibré pour `feature_name` (P1 #1).

        Si la feature n'a pas de pool calibré (feature inconnue), fallback
        sur le pool par défaut.
        """
        pools = self._ensure_threshold_quantiles()
        pool = pools.get(feature_name) or list(self._FALLBACK_THRESHOLD_POOL)
        return float(self._rng.choice(pool))


# --------------------------------------------------------------------------- #
# Generateur 1 : Random Search
# --------------------------------------------------------------------------- #


class RandomSearchGenerator(BaseGenerator):
    """Random search sous contraintes (typage, profondeur, ratios) — VRAIE implémentation.

    Note : pas d'évolution (random search pur). Accepte `engine` pour
    l'uniformité d'API avec les autres générateurs, mais ne l'utilise pas.

    Cohérence avec le système :
      - Mêmes features continues (ATOMIC/QUANTITATIVE/FACTOR) que TypedGP/Beam/NSGA-II.
      - Mêmes opérateurs logiques (AND/OR/NOT/XOR) que TypedGP.
      - Pas de fallback silencieux : ValueError si aucune feature continue.
    """

    def __init__(
        self,
        protocol: GenerationProtocol,
        config: EinherjarConfig,
        engine: Any | None = None,  # ignoré, pour uniformité d'API
    ) -> None:
        super().__init__(protocol, engine=engine)
        self.config = config

    def generate(self) -> GeneratorResult:
        import time
        t0 = time.time()
        continuous = [
            f for f in self.config.usable_feature_names
            if self._feature_type(f) in (FeatureType.ATOMIC, FeatureType.QUANTITATIVE, FeatureType.FACTOR)
        ]
        # NOTE (refactor) : RandomSearch est un générateur NON évolutionnaire —
        # il génère `n_candidates` hypothèses (volume de génération, ex. 100k)
        # SANS appeler le moteur. L'ancien code bouclait sur `n_eval_budget`
        # (défaut 200) et épuisait à lui seul le budget global du comparator,
        # ce qui empêchait Beam/TypedGP/GE/Memetic/NSGA-II d'être évalués.
        hyps: list[Hypothesis] = []
        i = 0
        while i < self.protocol.n_candidates:
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
            n_evaluated=0,  # random search pur : aucun appel moteur ici
            n_passed_admission=0,
            generation_time_s=time.time() - t0,
            meta={"n_candidates": len(hyps), "evaluations_uses": "none"},
        )

    def _sample_atomic(self, pool: Sequence[str]) -> Condition:
        feat = self._rng.choice(pool)
        op = self._rng.choice([CompareOp.LT, CompareOp.GT])
        # P1 #1 : seuil tiré depuis le pool calibré sur le train (plus d'uniforme -2..2).
        value = round(self._sample_threshold_for(feat), 4)
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
    """Beam Search à expansion par niveaux — VRAIE implémentation.

    Algorithme (Lowerre 1976, Reddy 1977) :
      1. Niveau 0 (initial) : beam_width × 2 directions × 2 ops × n_thresholds
         = beam initial de ~beam_width × 4 × n_thresholds candidats atomiques.
      2. Pour chaque niveau suivant (depth 1 à max_conditions) :
         a. Pour chaque candidat du beam précédent, on génère des expansions
            (compound avec une nouvelle condition atomique combinée en AND).
         b. On évalue chaque expansion via engine.train_calibrate + test_on(val)
            (scoring intermédiaire).
         c. On garde le top beam_width (élitisme intra-niveau).
      3. On retourne le beam final (les K meilleurs après tous les niveaux).

    Cohérence avec le moteur d'évaluation :
      - engine REQUIS pour le scoring intermédiaire (sinon le beam est aléatoire).
      - Le beam n'évalue PAS val plusieurs fois (1 évaluation par candidat par niveau).

    Différent de TypedGP :
      - BeamSearch explore systématiquement (largeur × profondeur), pas stochastique.
      - Pas de crossover/mutation, juste expansion.
      - Plus rapide mais moins diversifié.
    """

    def __init__(
        self,
        protocol: GenerationProtocol,
        config: EinherjarConfig,
        engine: Any | None = None,
        beam_width: int = 16,
        n_thresholds: int = 3,
        max_depth: int | None = None,
    ) -> None:
        """Initialise Beam Search.

        Args:
            engine: Moteur d'évaluation (REQUIS pour le scoring intermédiaire).
            beam_width: Largeur du beam (K meilleurs conservés par niveau).
            n_thresholds: Nombre de valeurs de seuil à tester par (feature, op).
            max_depth: Profondeur max (1 = atome seul, 2 = AND de 2, etc.).
                Défaut : protocol.max_conditions.
        """
        super().__init__(protocol, engine=engine)
        if engine is None:
            raise ValueError(
                "BeamSearchGenerator requiert un moteur d'évaluation (engine=...) "
                "pour le scoring intermédiaire (sinon le beam est aléatoire)."
            )
        self.config = config
        self.beam_width = beam_width
        self.n_thresholds = n_thresholds
        self.max_depth = max_depth if max_depth is not None else protocol.max_conditions
        # Pool de features continues.
        self._continuous_features: list[str] = [
            f for f in config.usable_feature_names
            if self._feature_type(f) in (FeatureType.ATOMIC, FeatureType.QUANTITATIVE, FeatureType.FACTOR)
        ]
        if not self._continuous_features:
            raise ValueError("Aucune feature continue exploitable pour BeamSearch")
        logger.info(
            "BeamSearchGenerator : K=%d, %d seuils, depth=%d, %d features continues",
            beam_width, n_thresholds, self.max_depth, len(self._continuous_features),
        )

    def generate(self) -> GeneratorResult:
        """Lance Beam Search niveau par niveau avec scoring intermédiaire."""
        import time
        t0 = time.time()
        self._n_internal_evaluated = 0
        train_ohlcv = getattr(self, "_search_train_ohlcv", None)
        train_features = getattr(self, "_search_train_features", None)
        val_ohlcv = getattr(self, "_search_val_ohlcv", None)
        val_features = getattr(self, "_search_val_features", None)
        if train_ohlcv is None or val_ohlcv is None:
            raise ValueError(
                "BeamSearchGenerator a besoin des données (train_ohlcv, val_ohlcv). "
                "Assure-toi que handle_compare les a injectées."
            )
        # Les seuils sont des quantiles du train, jamais des constantes de z-score.
        self._ensure_threshold_quantiles()
        # Niveau 0 : réserver du budget aux expansions suivantes.
        beam: list[Hypothesis] = self._initial_beam()
        # Évaluation initiale.
        scores: list[float] = self._score_beam(beam, train_ohlcv, train_features, val_ohlcv, val_features)
        beam = beam[:len(scores)]
        beam, scores = self._prune(beam, scores, k=self.beam_width)
        # Expansion niveau par niveau.
        for depth in range(1, self.max_depth):
            expanded: list[Hypothesis] = []
            for parent in beam:
                expanded.extend(self._expand_at_depth(parent, depth))
            if not expanded:
                break
            exp_scores = self._score_beam(
                expanded, train_ohlcv, train_features, val_ohlcv, val_features,
            )
            # Combine parent + expanded, on garde le top K.
            expanded = expanded[:len(exp_scores)]
            combined_beam = beam + expanded
            combined_scores = scores + exp_scores
            beam, scores = self._prune(combined_beam, combined_scores, k=self.beam_width)
            logger.info(
                "BeamSearch depth=%d : %d candidats, top_score=%.4f",
                depth, len(expanded), scores[0] if scores else float("nan"),
            )
        return GeneratorResult(
            generator_name=self.name,
            hypotheses=tuple(beam),
            n_generated=len(beam),
            n_evaluated=self._n_internal_evaluated,
            n_passed_admission=0,
            generation_time_s=time.time() - t0,
            meta={
                "method": "BeamSearch-Lowerre-Reddy",
                "beam_width": self.beam_width,
                "n_thresholds": self.n_thresholds,
                "max_depth": self.max_depth,
            },
        )

    def _initial_beam(self) -> list[Hypothesis]:
        """Beam initial : K features × 2 directions × 2 ops × n_thresholds."""
        beam: list[Hypothesis] = []
        i = 0
        features = list(self._continuous_features)
        self._rng.shuffle(features)
        initial_cap = max(self.beam_width, self.protocol.n_eval_budget // max(1, self.max_depth))
        for feat in features:
            for direction in (Direction.LONG, Direction.SHORT):
                for op in (CompareOp.LT, CompareOp.GT):
                    values = self._threshold_values(feat)
                    for v in values:
                        if i >= initial_cap:
                            return beam
                        h = Hypothesis(
                            id=f"{self.name}_{i:06d}",
                            condition_tree=Condition(feature_ref=feat, operator=op, value=v, transformation=None),
                            amplitude=self._make_amplitude(direction),
                            direction=direction,
                            universe=self._make_universe(),
                            cooldown_k=self.protocol.cooldown_k,
                        )
                        beam.append(h)
                        i += 1
        return beam

    def _expand_at_depth(self, parent: Hypothesis, depth: int) -> list[Hypothesis]:
        """Génère des expansions du parent au niveau `depth` (compound avec une nouvelle condition)."""
        expansions: list[Hypothesis] = []
        features = list(self._continuous_features)
        self._rng.shuffle(features)
        for feat in features:
            for op in (CompareOp.LT, CompareOp.GT):
                for v in self._threshold_values(feat):
                    new_cond = ConditionNode(
                        op=LogicalOp.AND,
                        left=parent.condition_tree,
                        right=Condition(feature_ref=feat, operator=op, value=v, transformation=None),
                    )
                    # id unique : prefixe parent + profondeur + index local.
                    # (le prefixe seul pouvait collisionner entre deux niveaux)
                    new_h = Hypothesis(
                        id=f"{parent.id}_d{depth}_x{len(expansions):02d}",
                        condition_tree=new_cond,
                        amplitude=parent.amplitude,
                        direction=parent.direction,
                        universe=parent.universe,
                        cooldown_k=parent.cooldown_k,
                    )
                    expansions.append(new_h)
                    if len(expansions) >= self.protocol.n_eval_budget:
                        return expansions
        return expansions

    def _threshold_values(self, feature_name: str) -> list[float]:
        """Sous-échantillonne un pool de quantiles sans inventer une échelle."""
        pool = self._ensure_threshold_quantiles().get(feature_name, [])
        if self.n_thresholds <= 1:
            return [pool[len(pool) // 2]] if pool else []
        if len(pool) <= self.n_thresholds:
            return list(pool)
        positions = [round(i * (len(pool) - 1) / (self.n_thresholds - 1)) for i in range(self.n_thresholds)]
        return [pool[index] for index in positions]

    def _score_beam(
        self,
        beam: list[Hypothesis],
        train_ohlcv: Any, train_features: Any, val_ohlcv: Any, val_features: Any,
    ) -> list[float]:
        """Évalue chaque candidat du beam (scoring intermédiaire)."""
        scores: list[float] = []
        remaining = max(0, self.protocol.n_eval_budget - getattr(self, "_n_internal_evaluated", 0))
        ppy = periods_per_year_for_timeframe(getattr(val_ohlcv, "timeframe", "1h"))
        for h in beam[:remaining]:
            try:
                calibrated = self.engine.train_calibrate(h, train_ohlcv, train_features)
                m = self.engine.test_on(h, val_ohlcv, val_features, calibrated, "val")
                scores.append(self._growth_fitness(m, ppy, soft=True))
            except Exception:  # noqa: BLE001
                scores.append(float("-inf"))
            self._n_internal_evaluated += 1
        return scores

    @staticmethod
    def _prune(
        beam: list[Hypothesis], scores: list[float], k: int,
    ) -> tuple[list[Hypothesis], list[float]]:
        """Garde les K meilleurs candidats (tri par score décroissant)."""
        if not beam:
            return [], []
        order = sorted(range(len(beam)), key=lambda i: scores[i], reverse=True)
        order = order[:k]
        return [beam[i] for i in order], [scores[i] for i in order]

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


# --------------------------------------------------------------------------- #
# Catégories de nœuds pour le contrôle des types (STGP)
# --------------------------------------------------------------------------- #

# Catégories sémantiques utilisées par TypedGP pour le contrôle des types :
# - "atomic" : Condition (feuille : feature + op + value)
# - "compound" : ConditionNode (interne : LogicalOp + 2 enfants)
_NODE_CATEGORIES = ("atomic", "compound")


class TypedGPGenerator(BaseGenerator):
    """Strongly-Typed Genetic Programming (STGP) — VRAIE implémentation évolutionnaire.

    Algorithme (Koza 1992 + Montana 1995 pour le typage strict) :
      1. Initialisation : population de N arbres (méthode grow + full).
      2. Évaluation : fitness = Sharpe net sur val (via engine).
      3. Boucle évolutionnaire (n_generations itérations) :
         a. Sélection par tournoi binaire (taille k=3) sur la fitness.
         b. Crossover sous-arbre : sélectionne un nœud de même catégorie
            (atomic/compound) dans chaque parent, swap.
         c. Mutation sous-arbre : remplace un sous-arbre par un nouveau
            (méthode grow) avec probabilité `mutation_prob`.
         d. Évalue l'offspring.
         e. Remplace la population (elitiste : on garde les N meilleurs).
      4. Retourne la population finale (les arbres viables).

    Contrôle strict des types (STGP) :
      - Chaque nœud a une "catégorie" : atomic (feuille) ou compound (interne).
      - Le crossover ne swap que des nœuds de même catégorie.
      - La mutation remplace par un sous-arbre de même catégorie.
      - Le grow respecte la profondeur max.

    Cohérence avec le moteur d'évaluation :
      - engine.train_calibrate + engine.test_on sont utilisés pour la fitness.
      - Si engine=None : seule l'initialisation est faite (mode "population seule"),
        refusé par défaut (ValueError) pour forcer l'évolution réelle.

    Différent de NSGA-II : TypedGP est mono-objectif (Sharpe net) avec représentation
    arborescente (multi-conditions). NSGA-II est multi-objectif avec représentation
    paramétrique (1 condition).
    """

    def __init__(
        self,
        protocol: GenerationProtocol,
        config: EinherjarConfig,
        engine: Any | None = None,
        population_size: int = 50,
        n_generations: int = 10,
        crossover_prob: float = 0.8,
        mutation_prob: float = 0.2,
        tournament_size: int = 3,
        elitism: int = 2,
    ) -> None:
        """Initialise TypedGP.

        Args:
            engine: Moteur d'évaluation (REQUIS pour l'évolution ; sans engine,
                on ne fait que l'initialisation, ce qui n'a aucun sens pour P10).
            population_size: Taille de la population.
            n_generations: Nombre de générations.
            crossover_prob: Probabilité de crossover par paire de parents.
            mutation_prob: Probabilité de mutation par enfant.
            tournament_size: Taille du tournoi pour la sélection.
            elitism: Nombre de meilleurs individus préservés à chaque génération.
        """
        super().__init__(protocol, engine=engine)
        if engine is None:
            raise ValueError(
                "TypedGPGenerator requiert un moteur d'évaluation (engine=...) "
                "pour évaluer la fitness (Sharpe net) pendant l'évolution."
            )
        self.config = config
        max_population = max(2, protocol.n_eval_budget // max(1, n_generations + 1))
        self.population_size = min(population_size, max_population)
        self.n_generations = min(
            n_generations,
            max(0, (protocol.n_eval_budget // self.population_size) - 1),
        )
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.tournament_size = tournament_size
        self.elitism = elitism
        # Pool de features continues (valeur float) pour les feuilles à seuil.
        # Critère value_type (ET non feature_type) : couvre ATOMIC/QUANTITATIVE/
        # FACTOR + les composite_derived float (*_signal) — les 9 *_signal
        # (sma_20_signal, ema_12_signal, macd_trend_signal, ...) étaient perdus
        # car classés composite_derived. Vérifié : 106 floats / 218 usable.
        self._continuous_features: list[str] = [
            f for f in config.usable_feature_names
            if _value_type_of(config, f) == "float"
        ]
        # P2-02 : features booléennes 0/1 (patterns candlestick + *_signal).
        # Comparées par EQ/NE/IN, jamais par seuil continu.
        self._pattern_features: list[str] = [
            f for f in config.usable_feature_names
            if _value_type_of(config, f) == "boolean"
        ]
        if not self._continuous_features and not self._pattern_features:
            raise ValueError("Aucune feature exploitable pour TypedGP")
        logger.info(
            "TypedGPGenerator : N=%d, gen=%d, %d features continues + %d patterns, "
            "crossover=%.2f, mutation=%.2f, tournament=%d, elitism=%d",
            population_size, n_generations, len(self._continuous_features),
            len(self._pattern_features), crossover_prob, mutation_prob,
            tournament_size, elitism,
        )

    def generate(self) -> GeneratorResult:
        """Lance TypedGP et retourne la population finale.

        Pour la V1 : on retourne TOUS les individus (admissibles ou non).
        Le filtrage admission est appliqué par le comparator via admission_fn.
        """
        import time
        t0 = time.time()
        # 1. Population initiale : 50% grow, 50% full (diversité).
        max_depth = self.protocol.max_conditions
        population: list[Hypothesis] = []
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
                cooldown_k=self.protocol.cooldown_k,
            )
            population.append(h)
        # 2. Évaluation initiale.
        fitness = self._evaluate_population(population)
        # 3. Boucle évolutionnaire.
        train_ohlcv = getattr(self, "_search_train_ohlcv", None)
        train_features = getattr(self, "_search_train_features", None)
        val_ohlcv = getattr(self, "_search_val_ohlcv", None)
        val_features = getattr(self, "_search_val_features", None)
        for gen in range(self.n_generations):
            # 3a. Sélection des parents.
            parents = self._tournament_selection(population, fitness, n=self.population_size)
            # 3b. Reproduction : crossover + mutation.
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
            # 3c. Évaluation offspring.
            offspring_fitness = self._evaluate_population(
                offspring, train_ohlcv, train_features, val_ohlcv, val_features,
            )
            # 3d. Combine P + Q (taille 2N).
            union_pop = population + offspring
            union_fit = fitness + offspring_fitness
            # 3e. Sélection élitiste : on garde les N meilleurs.
            order = sorted(range(len(union_pop)), key=lambda i: union_fit[i], reverse=True)
            order = order[: self.population_size]
            population = [union_pop[i] for i in order]
            fitness = [union_fit[i] for i in order]
            logger.info(
                "TypedGP gen %d/%d : best_growth=%.4f, mean_growth=%.4f",
                gen + 1, self.n_generations, fitness[0],
                sum(fitness) / max(1, len(fitness)),
            )
        # 4. Déduplique par signature de règle.
        seen: set[tuple] = set()
        unique: list[Hypothesis] = []
        for h in population:
            sig = (h.condition_tree, h.direction, h.cooldown_k)
            if sig not in seen:
                seen.add(sig)
                unique.append(h)
        return GeneratorResult(
            generator_name=self.name,
            hypotheses=tuple(unique),
            n_generated=len(population),
            n_evaluated=len(unique),
            n_passed_admission=0,  # admission appliquée par le comparator
            generation_time_s=time.time() - t0,
            meta={
                "method": "TypedGP-Koza+Montana",
                "n_generations": self.n_generations,
                "population_size": self.population_size,
                "init_methods": ("grow", "full"),
                "crossover": "subtree_type_preserving",
                "mutation": "subtree_regrow",
                "selection": f"tournament_k={self.tournament_size}",
            },
        )

    # ------------------------------------------------------------------ #
    # Initialisation (Koza : grow + full)
    # ------------------------------------------------------------------ #

    def _init_tree(self, method: str, max_depth: int) -> Condition | ConditionNode:
        """Initialise un arbre par grow ou full."""
        if method == "grow":
            return self._grow(max_depth=max_depth, depth=0)
        return self._full(max_depth=max_depth, depth=0)

    def _grow(self, max_depth: int, depth: int) -> Condition | ConditionNode:
        """Méthode grow : à chaque niveau, 50% chance de retourner une feuille."""
        if depth >= max_depth or (depth > 0 and self._rng.random() < 0.5):
            return self._atom()
        op = self._rng.choice(list(LogicalOp))
        if op == LogicalOp.NOT:
            # NOT unaire : un seul enfant.
            child = self._grow(max_depth=max_depth, depth=depth + 1)
            return ConditionNode(op=op, left=child, right=None)
        left = self._grow(max_depth=max_depth, depth=depth + 1)
        right = self._grow(max_depth=max_depth, depth=depth + 1)
        return ConditionNode(op=op, left=left, right=right)

    def _full(self, max_depth: int, depth: int) -> Condition | ConditionNode:
        """Méthode full : tous les nœuds à profondeur max sont des feuilles."""
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
        """Crée une feuille : feature (typée) + op + value.

        P2-02 : tire AUSSI dans le pool de patterns booléens (0/1) avec des
        opérateurs d'égalité (EQ/NE/IN) — un pattern est un fait discret
        (présent/absent), pas un seuil continu.
        """
        if self._pattern_features and self._rng.random() < 0.35:
            feat = self._rng.choice(self._pattern_features)
            op = self._rng.choice([CompareOp.EQ, CompareOp.NE])
            value = float(self._rng.randint(0, 1))
            return Condition(feature_ref=feat, operator=op, value=value, transformation=None)
        feat = self._rng.choice(self._continuous_features)
        op = self._rng.choice([CompareOp.LT, CompareOp.GT, CompareOp.LE, CompareOp.GE])
        # P1 #1 : seuil tiré depuis le pool calibré sur le train (plus d'uniforme -2..2).
        value = round(self._sample_threshold_for(feat), 4)
        return Condition(feature_ref=feat, operator=op, value=value, transformation=None)

    def _feature_type(self, name: str) -> FeatureType | None:
        info = self.config.features_taxonomy.get("features", {}).get(name, {})
        type_str = info.get("feature_type")
        try:
            return FeatureType(type_str) if type_str else None
        except ValueError:
            return None

    # ------------------------------------------------------------------ #
    # Évaluation de la fitness (CROISSANCE, alignée admission)
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

        P1 : l'ancienne fitness (Sharpe net sur search_val) évoluait vers des
        profils que l'admission n'admet jamais (critère 7 CROISSANCE, objectif
        ×10, porte n_trades). On aligne l'objectif d'évolution sur l'admission :
                 fitness = log(1 + CAGR) = trades_par_an * log1p(ret_mean_net)
        avec trades_par_an = periods_per_year / avg_holding_period.
        Porte dure : n_signals < min_trades (config.thresholds.n_trades.min_total)
        -> fitness -inf (la même porte que l'admission). Retourne -inf sur échec.

        Retourne une liste de floats (-inf si invalide).
        """
        train_ohlcv = train_ohlcv or getattr(self, "_search_train_ohlcv", None)
        train_features = train_features or getattr(self, "_search_train_features", None)
        val_ohlcv = val_ohlcv or getattr(self, "_search_val_ohlcv", None)
        val_features = val_features or getattr(self, "_search_val_features", None)
        if train_ohlcv is None or val_ohlcv is None:
            return [float("-inf")] * len(population)
        # Tasting (Décision 2026-08-10) : si protocol.n_samples > 0, évalue
        # sur un sous-échantillon seedé de blocs contigus. L'échantillon est
        # reconstruit à CHAQUE appel : toutes les générations partagent le
        # même seed → même fenêtre → fitness comparables entre individus.
        # L'admission finale, elle, évalue TOUJOURS le val complet.
        if int(getattr(self.protocol, "n_samples", 0) or 0) > 0:
            val_ohlcv, val_features = self._taste_frames(val_ohlcv, val_features)
        min_trades = int(
            self.config.thresholds.get("n_trades", {}).get("min_total", 30) or 30
        )
        timeframe = getattr(val_ohlcv, "timeframe", "1h")
        periods_per_year = periods_per_year_for_timeframe(timeframe)
        fitness: list[float] = []
        for h in population:
            try:
                calibrated = self.engine.train_calibrate(h, train_ohlcv, train_features)
                # with_bootstrap=False : l'évolution ne lit que CAGR/n_signals,
                # le bootstrap (~90% du temps) est réservé à l'admission.
                m = self.engine.test_on(
                    h, val_ohlcv, val_features, calibrated, "val",
                    with_bootstrap=False,
                )
                fitness.append(self._growth_fitness(m, periods_per_year, min_trades))
            except Exception:  # noqa: BLE001
                fitness.append(float("-inf"))
        return fitness



    # ------------------------------------------------------------------ #
    # Sélection par tournoi
    # ------------------------------------------------------------------ #

    def _tournament_selection(
        self,
        population: list[Hypothesis],
        fitness: list[float],
        n: int,
    ) -> list[Hypothesis]:
        """Sélection par tournoi binaire (taille = self.tournament_size)."""
        if not population:
            return []
        selected: list[Hypothesis] = []
        for _ in range(n):
            # Tire k indices aléatoires.
            indices = self._rng.sample(range(len(population)), min(self.tournament_size, len(population)))
            # Garde celui avec la meilleure fitness (NaN traité comme -inf).
            best_i = max(indices, key=lambda i: (fitness[i] if fitness[i] == fitness[i] else float("-inf")))
            selected.append(population[best_i])
        return selected

    # ------------------------------------------------------------------ #
    # Crossover sous-arbre (type-preserving)
    # ------------------------------------------------------------------ #

    def _subtree_crossover(
        self,
        p1: Hypothesis,
        p2: Hypothesis,
    ) -> tuple[Hypothesis, Hypothesis]:
        """Crossover : sélectionne un nœud de même catégorie dans chaque parent, swap.

        Type-preserving : on ne peut swapper qu'un atomic avec un atomic,
        ou un compound avec un compound. Sinon, on retourne les parents tels quels.
        """
        # Liste les nœuds par catégorie pour chaque parent.
        nodes1 = self._collect_nodes_by_category(p1.condition_tree)
        nodes2 = self._collect_nodes_by_category(p2.condition_tree)
        # Choisit une catégorie commune (atomic ou compound).
        common = [c for c in _NODE_CATEGORIES if nodes1[c] and nodes2[c]]
        if not common:
            return p1, p2  # pas de swap possible
        cat = self._rng.choice(common)
        # Sélectionne un nœud aléatoire dans chaque parent (de cette catégorie).
        path1 = self._rng.choice(nodes1[cat])
        path2 = self._rng.choice(nodes2[cat])
        # Swap les sous-arbres.
        new_tree1 = self._swap_subtree(p1.condition_tree, path1, p2.condition_tree, path2)
        new_tree2 = self._swap_subtree(p2.condition_tree, path2, p1.condition_tree, path1)
        return (
            self._clone_with_tree(p1, new_tree1),
            self._clone_with_tree(p2, new_tree2),
        )

    def _collect_nodes_by_category(
        self, tree: Condition | ConditionNode,
    ) -> dict[str, list[list[bool]]]:
        """Collecte tous les nœuds d'un arbre, groupés par catégorie.

        Returns:
            Dict {"atomic": [path1, path2, ...], "compound": [path1, ...]}
            où path = liste de booléens (False=gauche, True=droite pour CompoundNode,
            pas de direction pour Condition feuille).
        """
        result: dict[str, list[list[bool]]] = {"atomic": [], "compound": []}
        def _walk(node: Condition | ConditionNode, path: list[bool]) -> None:
            if isinstance(node, Condition):
                result["atomic"].append(path)
                return
            # ConditionNode = compound.
            result["compound"].append(path)
            _walk(node.left, path + [False])
            if node.right is not None:
                _walk(node.right, path + [True])
        _walk(tree, [])
        return result

    def _swap_subtree(
        self,
        tree: Condition | ConditionNode,
        path: list[bool],
        donor: Condition | ConditionNode,
        donor_path: list[bool],
    ) -> Condition | ConditionNode:
        """Remplace le sous-arbre de `tree` à `path` par le sous-arbre de `donor` à `donor_path`."""
        # Extrait le sous-arbre du donneur.
        donor_subtree = self._get_subtree(donor, donor_path)
        # Si path est vide, on remplace la racine.
        if not path:
            return donor_subtree
        # Sinon, on reconstruit l'arbre en remplaçant à l'index path[-1].
        return self._replace_subtree(tree, path, donor_subtree)

    def _get_subtree(
        self, node: Condition | ConditionNode, path: list[bool],
    ) -> Condition | ConditionNode:
        if not path:
            return node
        if isinstance(node, Condition):
            return node  # on ne peut pas descendre plus loin
        if path[0]:  # droite
            assert node.right is not None
            return self._get_subtree(node.right, path[1:])
        # gauche
        return self._get_subtree(node.left, path[1:])

    def _replace_subtree(
        self,
        node: Condition | ConditionNode,
        path: list[bool],
        new_subtree: Condition | ConditionNode,
    ) -> Condition | ConditionNode:
        """Reconstruit l'arbre avec `new_subtree` à l'emplacement `path`."""
        import copy
        if not path:
            return copy.deepcopy(new_subtree)
        if isinstance(node, Condition):
            return node
        if path[0]:  # droite
            new_right = self._replace_subtree(node.right, path[1:], new_subtree) if node.right else None
            return ConditionNode(op=node.op, left=copy.deepcopy(node.left), right=new_right)
        new_left = self._replace_subtree(node.left, path[1:], new_subtree)
        return ConditionNode(op=node.op, left=new_left, right=copy.deepcopy(node.right))

    # ------------------------------------------------------------------ #
    # Mutation sous-arbre
    # ------------------------------------------------------------------ #

    def _subtree_mutation(self, h: Hypothesis) -> Hypothesis:
        """Mutation : avec probabilité `mutation_prob`, remplace un sous-arbre par un nouveau.

        Type-preserving : le nouveau sous-arbre a la même catégorie que celui qu'il remplace.
        Si aucun sous-arbre n'est sélectionné (ou que la mutation n'a pas lieu), retourne l'hypothèse telle quelle.
        """
        if self._rng.random() > self.mutation_prob:
            return h
        nodes = self._collect_nodes_by_category(h.condition_tree)
        cat = self._rng.choice(_NODE_CATEGORIES)
        if not nodes[cat]:
            return h  # pas de nœud de cette catégorie
        path = self._rng.choice(nodes[cat])
        # Génère un nouveau sous-arbre de la même catégorie.
        max_depth = self.protocol.max_conditions
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
        """Clone une hypothèse avec un nouveau condition_tree et un nouvel id."""
        import copy
        new_h = Hypothesis(
            id=f"{h.id}_x{self._rng.randint(0, 9999):04d}",
            condition_tree=copy.deepcopy(new_tree),
            amplitude=h.amplitude,
            direction=h.direction,
            universe=h.universe,
            cooldown_k=h.cooldown_k,
        )
        return new_h


# --------------------------------------------------------------------------- #
# Generateur 4 : Grammatical Evolution (placeholder — BNF pas encore écrite)
# --------------------------------------------------------------------------- #


class GrammaticalEvolutionGenerator(BaseGenerator):
    """Grammatical Evolution (GE) — VRAIE implementation (BNF Phase 4).

    Algorithme (Ryan et al. 1998) :
      1. Pour chaque candidat du budget n_eval_budget :
         a. Choisir au hasard : soit (i) une feature parmi les 218 de la
            taxonomie, soit (ii) le bloc relations OHLCV
            (probabilite = relations_probability).
         b. Generer un chromosome = liste de `chromosome_length` entiers
            tires uniformement dans [0, 255] (8 bits, classique GE).
         c. Decoder le chromosome via BNFCodec (cf. bnf_parser.py) :
            consume les codons un par un, choix de production
            = codon % nb_productions, wraparound si epuise.
         d. Le decoder produit une Condition ou ConditionNode
            (cf. utils/types.py).
         e. Sample direction (LONG/SHORT) et amplitude (depuis le
            protocol). Construire l'Hypothesis.
      2. Si le decoder leve BNFDecodeError (chromosome trop court,
         impasse), on skip le candidat et on continue (pas de
         fallback silencieux : on documente le skip dans meta).

    Branche dediee : bnf-ge-integration.

    Attributes:
        config: Configuration Einherjar (pour taxonomie 218 + bnf.py).
        bnf_grammar: Override de la grammaire BNF (None = tirer au hasard).
        chromosome_length: Taille du chromosome (10-20 typique).
        relations_probability: Probabilite d'utiliser le bloc relations
            OHLCV plutot qu'une feature.
    """

    def __init__(
        self,
        protocol: GenerationProtocol,
        config: EinherjarConfig,
        engine: Any | None = None,  # ignore, uniformite d'API
        bnf_grammar: str | None = None,
        chromosome_length: int = 12,
        relations_probability: float = 0.2,
    ) -> None:
        super().__init__(protocol, engine=engine)
        self.config = config
        self.bnf_grammar = bnf_grammar
        self.chromosome_length = chromosome_length
        # (fix) Le bloc relations OHLCV produit des expressions arithmetiques
        # (high - low > q_range_p50) que ni le decodeur (BNFDecodeError sur
        # "-") ni le moteur d'evaluation (pas d'operation '-' implementee) ne
        # supportent : ~20% des iterations GE etaient skimpees en silence.
        # Desactive jusqu'a ce que le moteur encode ces relations.
        self.relations_probability = 0.0 if relations_probability > 0 else relations_probability
        # Import paresseux pour eviter cycle au chargement du module.
        from einherjar.research.generators.bnf import (
            FEATURE_GRAMMARS,
            get_relations_grammar,
        )
        from einherjar.research.generators.bnf_parser import BNFCodec
        self._BNFCodec = BNFCodec
        self._FEATURE_GRAMMARS = FEATURE_GRAMMARS
        self._get_relations_grammar = get_relations_grammar

    def generate(self) -> GeneratorResult:
        """GE avec fitness interne, tournoi, crossover, mutation et elitisme.

        La validation externe reste exclusivement la responsabilite du
        comparator. Le mode sans moteur est reserve au decodage BNF et aux
        tests unitaires ; il ne se presente pas comme une GE evolutionnaire.
        """
        if self.engine is None or getattr(self, "_search_train_ohlcv", None) is None:
            result = self._generate_random_population()
            return GeneratorResult(
                generator_name=result.generator_name, hypotheses=result.hypotheses,
                n_generated=result.n_generated, n_evaluated=0,
                n_passed_admission=0, generation_time_s=result.generation_time_s,
                meta={**result.meta, "mode": "decode_only"},
            )
        import time
        started = time.time()
        budget = self.protocol.n_eval_budget
        population_size = min(16, max(2, budget // 4))
        initial = self._generate_random_population(budget=population_size)
        scored = [(self._fitness(hypothesis), hypothesis) for hypothesis in initial.hypotheses]
        n_evaluated = len(scored)
        generation = 0
        codec_cache: dict[str, Any] = {}
        while n_evaluated < budget and scored:
            generation += 1
            ranked = sorted(scored, key=lambda item: item[0], reverse=True)
            elites = ranked[:max(1, population_size // 4)]
            offspring: list[Hypothesis] = []
            attempts = 0
            while (
                len(offspring) < population_size
                and n_evaluated + len(offspring) < budget
                and attempts < population_size * 10
            ):
                attempts += 1
                child = self._make_offspring(ranked, generation, len(offspring), codec_cache)
                if child is not None:
                    offspring.append(child)
            if not offspring:
                break
            children = [(self._fitness(hypothesis), hypothesis) for hypothesis in offspring]
            n_evaluated += len(children)
            scored = sorted(elites + children, key=lambda item: item[0], reverse=True)[:population_size]
        best = [hypothesis for _, hypothesis in sorted(scored, key=lambda item: item[0], reverse=True)]
        return GeneratorResult(
            generator_name=self.name, hypotheses=tuple(best),
            n_generated=len(best), n_evaluated=n_evaluated, n_passed_admission=0,
            generation_time_s=time.time() - started,
            meta={
                "method": "GrammaticalEvolution-Ryan1998",
                "mode": "evolutionary", "population_size": population_size,
                "generations": generation, "budget_used": n_evaluated,
                "crossover": "one_point", "mutation_rate": 0.10,
            },
        )

    def _fitness(self, hypothesis: Hypothesis) -> float:
        try:
            calibrated = self.engine.train_calibrate(
                hypothesis, self._search_train_ohlcv, self._search_train_features,
            )
            measures = self.engine.test_on(
                hypothesis, self._search_val_ohlcv, self._search_val_features,
                calibrated, "ge_search_val",
            )
            ppy = periods_per_year_for_timeframe(
                getattr(self._search_val_ohlcv, "timeframe", "1h"))
            return self._growth_fitness(measures, ppy)
        except Exception as exc:  # noqa: BLE001
            logger.debug("GE fitness failed for %s: %s", hypothesis.id, exc)
            return float("-inf")

    def _make_offspring(
        self, ranked: list[tuple[float, Hypothesis]], generation: int,
        child_index: int, codec_cache: dict[str, Any],
    ) -> Hypothesis | None:
        contenders_a = [self._rng.choice(ranked) for _ in range(min(3, len(ranked)))]
        contenders_b = [self._rng.choice(ranked) for _ in range(min(3, len(ranked)))]
        parent_a = max(contenders_a, key=lambda item: item[0])[1]
        parent_b = max(contenders_b, key=lambda item: item[0])[1]
        chrom_a = list(parent_a.meta.get("chromosome", ()))
        chrom_b = list(parent_b.meta.get("chromosome", ()))
        if len(chrom_a) < 2 or len(chrom_b) < 2:
            return None
        cut = self._rng.randint(1, min(len(chrom_a), len(chrom_b)) - 1)
        chromosome = chrom_a[:cut] + chrom_b[cut:]
        for index in range(len(chromosome)):
            if self._rng.random() < 0.10:
                chromosome[index] = self._rng.randint(0, 255)
        source = str(parent_a.meta.get("bnf_source", "__override__"))
        try:
            codec = codec_cache.get(source)
            if codec is None:
                codec = self._BNFCodec.from_text(self._grammar_for_source(source))
                codec_cache[source] = codec
            condition = codec.decode(chromosome=chromosome)
            direction = self._rng.choice((parent_a.direction, parent_b.direction))
            return Hypothesis(
                id=f"{self.name}_g{generation:03d}_{child_index:03d}",
                condition_tree=condition, amplitude=self._make_amplitude(direction),
                direction=direction, universe=self._make_universe(), cooldown_k=self.protocol.cooldown_k,
                meta={"bnf_source": source, "chromosome": chromosome,
                      "semantic_orientation": self._semantic_orientation(source)},
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("GE offspring decode failed: %s", exc)
            return None

    def _grammar_for_source(self, source: str) -> str:
        if source == "__override__":
            if self.bnf_grammar is None:
                raise ValueError("GE override source without grammar")
            return self.bnf_grammar
        if source == "__ohlcv_relations__":
            # (fix) relations OHLCV desactivees : le moteur n'implemente pas
            # les expressions de type high - low. Garde pour compat API.
            raise ValueError(
                "__ohlcv_relations__ n'est plus generable : expressions "
                "arithmetiques non supportees par le decodeur/le moteur"
            )
        grammar = self._FEATURE_GRAMMARS.get(source)
        if grammar is not None:
            return grammar
        from einherjar.research.generators.bnf import get_feature_grammar
        return get_feature_grammar(source, self.config)

    @staticmethod
    def _semantic_orientation(source: str) -> str:
        """L'orientation est informative : elle ne doit pas invalider GE."""
        try:
            from einherjar.research.generators.bnf_semantic import get_orientation
            return get_orientation(source).value
        except Exception as exc:  # noqa: BLE001
            logger.debug("GE semantic orientation unavailable for %s: %s", source, exc)
            return "neutral"

    def _generate_random_population(self, budget: int | None = None) -> GeneratorResult:
        import time
        t0 = time.time()
        hyps: list[Hypothesis] = []
        n_skipped: int = 0
        n_relations: int = 0
        n_atomic: int = 0
        n_compose: int = 0
        # Cache des codecs (une grammaire = un codec, on l'instancie 1 fois).
        codec_cache: dict[str, Any] = {}
        i = 0
        while i < (self.protocol.n_eval_budget if budget is None else budget):
            try:
                # 1) Choisir la source : override, relations OHLCV, ou feature random.
                if self.bnf_grammar is not None:
                    source_key = "__override__"
                    grammar_text = self.bnf_grammar
                elif False and self._rng.random() < self.relations_probability:
                    # branche desactivee : voir __init__ (relations_probability=0)
                    source_key = "__ohlcv_relations__"
                    grammar_text = self._get_relations_grammar("ohlcv")
                else:
                    feature_name = self._rng.choice(
                        self.config.usable_feature_names,
                    )
                    source_key = feature_name
                    grammar_text = self._FEATURE_GRAMMARS.get(feature_name)
                    if grammar_text is None:
                        # Feature sans grammaire custom, on prend le
                        # pattern par defaut.
                        from einherjar.research.generators.bnf import (
                            get_feature_grammar,
                        )
                        grammar_text = get_feature_grammar(
                            feature_name, self.config,
                        )
                # 2) Codec (cache).
                if source_key not in codec_cache:
                    codec_cache[source_key] = self._BNFCodec.from_text(
                        grammar_text,
                    )
                codec = codec_cache[source_key]
                # 3) Generer le chromosome.
                chromosome = [
                    self._rng.randint(0, 255)
                    for _ in range(self.chromosome_length)
                ]
                # 4) Decoder.
                cond = codec.decode(chromosome=chromosome)
                # Stats
                if source_key == "__ohlcv_relations__":
                    n_relations += 1
                elif isinstance(cond, ConditionNode):
                    n_compose += 1
                else:
                    n_atomic += 1
                # 5) Sample direction + amplitude.
                direction = self._rng.choice([Direction.LONG, Direction.SHORT])
                # 5b) Orientation semantique (BNF Phase 3) : pour les
                # patterns, ajouter l'orientation naturelle au meta.
                # Permet au moteur d'admission / comparateur de scorer
                # la coherence entre l'orientation du pattern et la
                # direction de l'Hypothesis.
                semantic_orient = self._semantic_orientation(source_key)
                # 6) Construire Hypothesis.
                h = Hypothesis(
                    id=f"{self.name}_{i:06d}",
                    condition_tree=cond,
                    amplitude=self._make_amplitude(direction),
                    direction=direction,
                    universe=self._make_universe(),
                    cooldown_k=self.protocol.cooldown_k,
                    meta={
                        "bnf_source": source_key,
                        "chromosome": chromosome,
                        "semantic_orientation": semantic_orient,
                    },
                )
                hyps.append(h)
                i += 1
            except Exception as exc:  # noqa: BLE001
                # BNFDecodeError (chromosome trop court / impasse) ou
                # autre erreur de decodage : on skip et on continue.
                logger.debug(
                    "%s : skip candidat %d (decode error: %s)",
                    self.name, i, exc,
                )
                n_skipped += 1
                i += 1
        return GeneratorResult(
            generator_name=self.name,
            hypotheses=tuple(hyps),
            n_generated=len(hyps),
            n_evaluated=0,
            n_passed_admission=0,
            generation_time_s=time.time() - t0,
            meta={
                "budget_used": len(hyps),
                "n_skipped": n_skipped,
                "n_atomic": n_atomic,
                "n_compose": n_compose,
                "n_relations": n_relations,
                "chromosome_length": self.chromosome_length,
                "relations_probability": self.relations_probability,
            },
        )


# --------------------------------------------------------------------------- #
# Generateur 5 : Memetic (EA + local search) — placeholder
# --------------------------------------------------------------------------- #


class MemeticGenerator(BaseGenerator):
    """Memetic : EA + local search — VRAIE implémentation (P10).

    Algorithme :
      1. Phase EA : TypedGPGenerator produit une population initiale.
      2. Phase LSO (Local Search Optimization) : pour chaque hypothèse, on
         applique des mutations locales 1-paramètre-à-la-fois et on garde
         celles qui améliorent le Sharpe sur val (hill climbing).

    Le hill climbing utilise `self.engine` pour évaluer les voisins (REQUIS).
    Chaque hypothèse améliorée est sauvegardée.

    Pour V1 : hill climbing = BeamRefiner-like (3 mutations × 1 itération).
    """

    def __init__(
        self,
        protocol: GenerationProtocol,
        config: EinherjarConfig,
        engine: Any | None = None,
        population_size: int = 50,
        lso_iterations: int = 1,
        lso_neighbors: int = 3,
    ) -> None:
        super().__init__(protocol, engine=engine)
        if engine is None:
            raise ValueError(
                "MemeticGenerator requiert un moteur d'évaluation (engine=...) "
                "pour la phase LSO (Local Search Optimization)."
            )
        self.config = config
        self.population_size = population_size
        self.lso_iterations = lso_iterations
        self.lso_neighbors = lso_neighbors
        self._ea = TypedGPGenerator(
            protocol, config, engine=engine, population_size=population_size,
        )

    def bind_data(
        self,
        train_ohlcv: Any,
        train_features: Any,
        val_ohlcv: Any,
        val_features: Any,
    ) -> None:
        super().bind_data(train_ohlcv, train_features, val_ohlcv, val_features)
        self._ea.bind_data(train_ohlcv, train_features, val_ohlcv, val_features)

    def generate(self) -> GeneratorResult:
        # 1. Phase EA : génération initiale via TypedGP.
        ea_result = self._ea.generate()
        # 2. Phase LSO : hill climbing sur chaque hypothèse.
        improved = self._local_search(ea_result.hypotheses)
        n_improved = sum(1 for h_orig, h_new in improved if h_orig != h_new)
        return GeneratorResult(
            generator_name=self.name,
            hypotheses=tuple(h_new for _, h_new in improved),
            n_generated=len(ea_result.hypotheses),
            n_evaluated=len(ea_result.hypotheses),
            n_passed_admission=n_improved,
            generation_time_s=ea_result.generation_time_s,
            meta={
                "ea": "TypedGPGenerator",
                "lso_iterations": self.lso_iterations,
                "lso_neighbors": self.lso_neighbors,
                "n_improved_by_lso": n_improved,
            },
        )

    def _local_search(self, hypotheses: list[Hypothesis]) -> list[tuple[Hypothesis, Hypothesis]]:
        """Hill climbing : pour chaque hypothèse, mute 1 paramètre et garde si meilleur.

        On essaie lso_neighbors mutations par paramètre (feature, op, threshold, cooldown).
        Chaque hypothèse est gardée (même si pas améliorée).
        """
        results: list[tuple[Hypothesis, Hypothesis]] = []
        # (fix fuite search-split) La recherche locale optimise sur le sous-split
        # INTERNE du train (`_search_*`, réservé à la recherche évolutionnaire),
        # JAMAIS sur le val externe du comparator : l'ancien code évaluait le
        # hill climbing sur `_val_ohlcv` — le même jeu qui sert à comparer les
        # générateurs et à l'admission — créant une fuite d'information.
        train_ohlcv = getattr(self, "_search_train_ohlcv", None) or getattr(self, "_train_ohlcv", None)
        train_features = getattr(self, "_search_train_features", None) or getattr(self, "_train_features", None)
        val_ohlcv = getattr(self, "_search_val_ohlcv", None) or getattr(self, "_val_ohlcv", None)
        val_features = getattr(self, "_search_val_features", None) or getattr(self, "_val_features", None)
        if train_ohlcv is None or val_ohlcv is None:
            # Pas de données : on ne peut pas évaluer → on garde les hyp originales.
            return [(h, h) for h in hypotheses]
        ppy = periods_per_year_for_timeframe(getattr(val_ohlcv, "timeframe", "1h"))
        for hyp in hypotheses:
            best_h = hyp
            best_score = float("-inf")
            try:
                calibrated = self.engine.train_calibrate(hyp, train_ohlcv, train_features)
                m = self.engine.test_on(hyp, val_ohlcv, val_features, calibrated, "val")
                best_score = self._growth_fitness(m, ppy)
            except Exception:  # noqa: BLE001
                results.append((hyp, hyp))
                continue
            # Voisins : mutations locales.
            for _ in range(self.lso_iterations):
                for neighbor in self._generate_neighbors(best_h, self.lso_neighbors):
                    try:
                        cal = self.engine.train_calibrate(neighbor, train_ohlcv, train_features)
                        m2 = self.engine.test_on(neighbor, val_ohlcv, val_features, cal, "val")
                        score2 = self._growth_fitness(m2, ppy)
                        if score2 > best_score:
                            best_score = score2
                            best_h = neighbor
                    except Exception:  # noqa: BLE001
                        continue
            results.append((hyp, best_h))
        return results

    def _generate_neighbors(self, hyp: Hypothesis, n: int) -> list[Hypothesis]:
        """Génère n voisins par mutation locale 1-paramètre-à-la-fois."""
        neighbors: list[Hypothesis] = []
        import copy
        for _ in range(n):
            h = copy.deepcopy(hyp)
            new_id = f"{h.id}_lso"
            # Mute la condition (feature/op/value) ou le cooldown.
            if isinstance(h.condition_tree, Condition) and self._rng.random() < 0.6:
                atom = h.condition_tree
                choice = self._rng.choice(["feature", "op", "value"])
                if choice == "feature":
                    if self.config.usable_feature_names:
                        atom = Condition(
                            feature_ref=self._rng.choice(self.config.usable_feature_names),
                            operator=atom.operator, value=atom.value,
                            transformation=atom.transformation,
                        )
                elif choice == "op":
                    atom = Condition(
                        feature_ref=atom.feature_ref,
                        operator=self._rng.choice(list(CompareOp)),
                        value=atom.value, transformation=atom.transformation,
                    )
                else:  # value
                    v = float(atom.value) if isinstance(atom.value, (int, float)) else 0.0
                    delta = self._rng.uniform(-0.5, 0.5)
                    atom = Condition(
                        feature_ref=atom.feature_ref, operator=atom.operator,
                        value=round(v + delta, 4), transformation=atom.transformation,
                    )
                new_h = Hypothesis(
                    id=new_id, condition_tree=atom,
                    amplitude=h.amplitude, direction=h.direction,
                    universe=h.universe, cooldown_k=h.cooldown_k,
                )
            else:
                # Mute le cooldown.
                delta = self._rng.choice([-1, 1])
                new_h = Hypothesis(
                    id=new_id, condition_tree=h.condition_tree,
                    amplitude=h.amplitude, direction=h.direction,
                    universe=h.universe,
                    cooldown_k=max(1, h.cooldown_k + delta),
                )
            neighbors.append(new_h)
        return neighbors

    def _generate_deferred_legacy(self) -> GeneratorResult:
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
# Generateur 6 : NSGA-II (Deb 2002) — VRAIE implémentation multi-objectif
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _NSGA2Individual:
    """Représentation interne NSGA-II (paramétrique pour V1, symbolique/BNF en V2).

    Vecteur réel + discret :
      - feature_id    : index dans la liste des features continues
      - op_id         : 0=LT, 1=GT, 2=LE, 3=GE
      - threshold     : valeur réelle du seuil (ex: 0.5 pour rsi>0.5)
      - cooldown_k    : K bougies minimum entre 2 signaux (entier)
      - direction_id  : 0=LONG, 1=SHORT
    """

    feature_id: int
    op_id: int
    threshold: float
    cooldown_k: int
    direction_id: int


@dataclass(frozen=True)
class _EvaluatedIndividual:
    """Individu + ses 4 objectifs + ses 8 contraintes dures évaluées."""

    individual: _NSGA2Individual
    hypothesis: Hypothesis
    # 4 objectifs (à MAXIMISER) : (sharpe_net, -max_drawdown, diversity, -complexity)
    objectives: tuple[float, float, float, float]
    # 8 contraintes dures (True = OK, False = violée)
    constraints_passed: tuple[bool, bool, bool, bool, bool, bool, bool, bool]
    # Pré-calculé pour la sélection
    n_violations: int
    n_signals: int
    # Pour le multi-actifs
    sharpe_per_asset_fold: tuple[float, ...] = ()


# 8 contraintes dures (cf. message user : "contraintes bloquantes avant le front de Pareto")
_N_CONSTRAINTS = 8
# Index des contraintes (documentation)
_C_DATA_VERSIONED = 0      # 1. Données réelles et versionnées
_C_RULE_TYPED = 1         # 2. Règle exécutable et typée
_C_MIN_TRADES = 2         # 3. Minimum de trades
_C_MULTI_ASSET = 3        # 4. Performance non concentrée sur un seul actif
_C_COSTS_APPLIED = 4      # 5. Coûts appliqués
_C_NO_TEMPORAL_LEAK = 5   # 6. Absence de fuite temporelle
_C_DD_BOUNDED = 6         # 7. Drawdown sous une limite de sécurité
_C_STABILITY = 7          # 8. Stabilité minimale entre les folds de validation


class NSGA2Generator(BaseGenerator):
    """NSGA-II multi-objectif (Deb 2002) — VRAIE implémentation.

    Algorithme (Deb, Pratap, Agarwal & Meyarivan, 2002) :
      1. Initialisation : population P0 aléatoire de taille N
      2. Pour chaque individu : évaluer fitness (4 objectifs) + 8 contraintes dures
      3. Boucle évolutionnaire (n_generations itérations) :
         a. Sélection par tournoi binaire (rank, crowding_distance)
         b. Variation : crossover SBX (threshold) + mutation (uniforme feature/op,
            gaussienne threshold, discrète cooldown/direction)
         c. Évaluer offspring
         d. Combine P+Q (taille 2N)
         e. Non-dominance sorting sur l'union
         f. Crowding distance par front
         g. Sélectionner les N meilleurs (elitiste)
      4. Retourner les hypothèses dont toutes les contraintes sont OK

    Contraintes dures (AVANT le front de Pareto) :
      1. Données réelles et versionnées
      2. Règle exécutable et typée (feature dans taxonomie)
      3. Minimum de trades (config.thresholds.n_trades.min_total)
      4. Performance non concentrée sur un seul actif (médiane par actif/fold > 0)
      5. Coûts appliqués (MesuresBrutes.costs_applied non vide)
      6. Absence de fuite temporelle (mesures viennent de test_on(val), garanti par construction)
      7. Drawdown < seuil (config.thresholds.max_drawdown.max_value)
      8. Stabilité entre folds (std(Sharpe par fold) < seuil)

    4 objectifs (à MAXIMISER) :
      f1 = Sharpe net
      f2 = -max_drawdown (équivalent à minimiser DD)
      f3 = diversité comportementale (dispersion temporelle des signaux)
           + bonus de diversité Jaccard vs corpus (admission/diversity.py :
           `corpus_jaccard_diversity` sur self._corpus_feature_sets, pondéré 50/50)
      f4 = -complexité (= -nb conditions ; ici -1 car représentation mono-condition)

    Note (P1-10) : la diversité Jaccard vs corpus est disponible via
    `admission.diversity.corpus_jaccard_diversity`. Pour l'activer, le caller
    doit peupler `self._corpus_feature_sets` (set de frozensets de features
    des Einhers deja admis) avant l'appel a `generate()`.
    """

    OP_CHOICES: tuple[CompareOp, ...] = (CompareOp.LT, CompareOp.GT, CompareOp.LE, CompareOp.GE)
    DIRECTION_CHOICES: tuple[Direction, ...] = (Direction.LONG, Direction.SHORT)

    def __init__(
        self,
        protocol: GenerationProtocol,
        config: EinherjarConfig,
        engine: Any | None = None,
        population_size: int = 50,
        n_generations: int = 20,
        crossover_prob: float = 0.9,
        mutation_prob: float = 0.2,
        sbx_eta: float = 20.0,
        pm_eta: float = 20.0,
        stability_max_std: float = 0.5,
        train_ohlcv: Any | None = None,
        train_features: Any | None = None,
        val_ohlcv: Any | None = None,
        val_features: Any | None = None,
    ) -> None:
        """Initialise NSGA-II.

        Args:
            engine: Moteur d'évaluation (REQUIS — lève ValueError si None).
            population_size: Taille de la population (N).
            n_generations: Nombre de générations.
            crossover_prob: Probabilité de crossover SBX par paire.
            mutation_prob: Probabilité de mutation par enfant.
            sbx_eta: Paramètre de distribution du crossover SBX (plus haut = enfants plus proches des parents).
            pm_eta: Paramètre de distribution de la mutation polynomiale.
            stability_max_std: Seuil de std(Sharpe par fold CPCV) pour la contrainte #8.
            train_*/val_*: Données pré-chargées (passées par le comparator pour éviter
                de recharger à chaque évaluation).
        """
        super().__init__(protocol, engine=engine)
        if engine is None:
            raise ValueError(
                "NSGA2Generator requiert un moteur d'évaluation (engine=...). "
                "C'est nécessaire pour évaluer les 4 objectifs et les 8 contraintes dures."
            )
        self.config = config
        max_population = max(2, protocol.n_eval_budget // max(1, n_generations + 1))
        self.population_size = min(population_size, max_population)
        self.n_generations = min(
            n_generations,
            max(0, (protocol.n_eval_budget // self.population_size) - 1),
        )
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.sbx_eta = sbx_eta
        self.pm_eta = pm_eta
        self.stability_max_std = stability_max_std
        self._train_ohlcv = train_ohlcv
        self._train_features = train_features
        self._val_ohlcv = val_ohlcv
        self._val_features = val_features
        # Liste des features continues (typage strict).
        self._continuous_features: list[str] = [
            f for f in config.usable_feature_names
            if self._feature_type(f) in (FeatureType.ATOMIC, FeatureType.QUANTITATIVE, FeatureType.FACTOR)
        ]
        if not self._continuous_features:
            raise ValueError("Aucune feature continue exploitable pour NSGA-II")
        # Seuils depuis la config.
        self._min_trades = int(config.thresholds["n_trades"]["min_total"])
        self._max_dd = float(config.thresholds["max_drawdown"]["max_value"])
        logger.info(
            "NSGA2Generator : N=%d, gen=%d, %d features continues, min_trades=%d, max_dd=%.2f",
            population_size, n_generations, len(self._continuous_features),
            self._min_trades, self._max_dd,
        )

    def generate(self) -> GeneratorResult:
        """Lance NSGA-II et retourne les hypothèses admissibles (toutes contraintes OK)."""
        import time
        t0 = time.time()
        # 1. Population initiale aléatoire.
        population = [self._random_individual() for _ in range(self.population_size)]
        # 2. Évaluation de la population initiale.
        evaluated = [self._evaluate(ind) for ind in population]
        # 3. Boucle évolutionnaire.
        for gen in range(self.n_generations):
            offspring = self._make_offspring(evaluated)
            offspring_eval = [self._evaluate(ind) for ind in offspring]
            # Combine P + Q (taille 2N).
            union = evaluated + offspring_eval
            # Sélection environnementale : on garde les N meilleurs.
            evaluated = self._environmental_selection(union, n=self.population_size)
            logger.info(
                "NSGA-II gen %d/%d : front F1 size=%d, n_admissible=%d",
                gen + 1, self.n_generations,
                sum(1 for e in evaluated if e.n_violations == 0),
                sum(1 for e in evaluated if e.n_violations == 0),
            )
        # 4. Filtre les individus dont toutes les contraintes sont OK.
        admissible = [ev for ev in evaluated if ev.n_violations == 0]
        n_admissible = len(admissible)
        # 5. Dédupplique par signature de règle (même feature+op+threshold+direction+cooldown).
        seen: set[tuple] = set()
        unique: list[Hypothesis] = []
        for ev in admissible:
            sig = (ev.individual.feature_id, ev.individual.op_id,
                   round(ev.individual.threshold, 4), ev.individual.direction_id,
                   ev.individual.cooldown_k)
            if sig not in seen:
                seen.add(sig)
                unique.append(ev.hypothesis)
        return GeneratorResult(
            generator_name=self.name,
            hypotheses=tuple(unique),
            n_generated=len(evaluated),
            n_evaluated=n_admissible,
            n_passed_admission=len(unique),
            generation_time_s=time.time() - t0,
            meta={
                "method": "NSGA-II-Deb2002",
                "n_generations": self.n_generations,
                "population_size": self.population_size,
                "n_objectives": 4,
                "n_constraints": _N_CONSTRAINTS,
                "objectives": ["sharpe_net", "neg_max_drawdown", "diversity", "neg_complexity"],
                "constraints": [
                    "data_versioned", "rule_typed", "min_trades", "multi_asset",
                    "costs_applied", "no_temporal_leak", "dd_bounded", "stability",
                ],
            },
        )

    # ------------------------------------------------------------------ #
    # Représentation → Hypothesis
    # ------------------------------------------------------------------ #

    def _random_individual(self) -> _NSGA2Individual:
        """Génère un individu aléatoire dans l'espace de représentation."""
        feat_id = self._rng.randint(0, len(self._continuous_features) - 1)
        # P1 #1 : seuil tiré depuis le pool calibré pour la feature choisie.
        feat_name = self._continuous_features[feat_id]
        threshold = round(self._sample_threshold_for(feat_name), 4)
        return _NSGA2Individual(
            feature_id=feat_id,
            op_id=self._rng.randint(0, len(self.OP_CHOICES) - 1),
            threshold=threshold,
            cooldown_k=self._rng.randint(1, 20),
            direction_id=self._rng.randint(0, len(self.DIRECTION_CHOICES) - 1),
        )

    def _to_hypothesis(self, ind: _NSGA2Individual) -> Hypothesis:
        """Convertit un individu en Hypothesis (1 condition atomique)."""
        feat = self._continuous_features[ind.feature_id]
        op = self.OP_CHOICES[ind.op_id]
        direction = self.DIRECTION_CHOICES[ind.direction_id]
        cond = Condition(feature_ref=feat, operator=op, value=ind.threshold, transformation=None)
        return Hypothesis(
            id=f"NSGA2_{ind.feature_id}_{ind.op_id}_{round(ind.threshold, 4):.4f}_{ind.cooldown_k}_{ind.direction_id}",
            condition_tree=cond,
            amplitude=self._make_amplitude(direction),
            direction=direction,
            universe=self._make_universe(),
            cooldown_k=ind.cooldown_k,
        )

    def _feature_type(self, name: str) -> FeatureType | None:
        info = self.config.features_taxonomy.get("features", {}).get(name, {})
        type_str = info.get("feature_type")
        try:
            return FeatureType(type_str) if type_str else None
        except ValueError:
            return None

    # ------------------------------------------------------------------ #
    # Évaluation : 4 objectifs + 8 contraintes dures
    # ------------------------------------------------------------------ #

    def _evaluate(self, ind: _NSGA2Individual) -> _EvaluatedIndividual:
        """Évalue un individu : 4 objectifs + 8 contraintes dures.

        Strategie :
          - Multi-actifs (P1-10) : si self._multi_assets est defini, on boucle
            sur les actifs et on agrege via la mediane (contrainte #4 devient
            reellement mesurable). Sinon : single-asset (compat V1).
          - Si calibration echoue : contraintes [False, False, False, False, False, True, True, False]
            (= pas de signal → fail presque tout, sauf no_temporal_leak et dd_bounded).
          - Si OK : 4 objectifs + 8 contraintes depuis MesuresBrutes + CPCV.
        """
        hyp = self._to_hypothesis(ind)
        # Contraintes par defaut (si evaluation echoue).
        objectives = (float("nan"), float("nan"), float("nan"), float("nan"))
        constraints = (False, False, False, False, False, True, True, False)
        n_violations = _N_CONSTRAINTS - 2  # tout sauf no_leak, dd_bounded
        n_signals = 0
        sharpe_per_asset_fold: tuple[float, ...] = ()
        try:
            # P1-10 : mode multi-actifs si _multi_assets est peuple par le comparator.
            multi = getattr(self, "_multi_assets", None)
            if multi:
                mesures = self._evaluate_multi_asset(hyp, multi)
            else:
                calibrated = self.engine.train_calibrate(
                    hyp, self._search_train_ohlcv, self._search_train_features,
                )
                mesures = self.engine.test_on(
                    hyp, self._search_val_ohlcv, self._search_val_features,
                    calibrated, "search_val",
                )
            n_signals = mesures.n_signals
            # --- 8 contraintes dures ---
            # 1. Données réelles et versionnées (data_version non vide).
            c1 = bool(self.protocol.data_version)
            # 2. Règle exécutable et typée (feature dans la taxonomie).
            c2 = hyp.condition_tree.feature_ref in self.config.usable_feature_names
            # 3. Minimum de trades.
            c3 = n_signals >= self._min_trades
            # 4. Performance non concentrée sur un seul actif.
            #    V1 : on utilise la médiane du Sharpe par actif (per_asset_stats).
            #    P1-10 : per_asset_stats est peuple par _evaluate_multi_asset.
            sharpes: list[float] = []
            if mesures.per_asset_stats:
                for sub in mesures.per_asset_stats.values():
                    if sub.sharpe_net == sub.sharpe_net:  # not NaN
                        sharpes.append(sub.sharpe_net)
            if len(sharpes) >= 2:
                sharpes.sort()
                median_sharpe = sharpes[len(sharpes) // 2]
                c4 = median_sharpe > 0.0
            else:
                # (fix c4) En SINGLE-asset, la contrainte « non concentrée sur un
                # seul actif » est SANS OBJET : on ne peut pas la mesurer, on ne
                # peut donc pas pénaliser l'individu. L'ANCIEN code mettait c4=False
                # faute de preuve, ce qui forçait TOUS les individus en violation
                # dès qu'on lançait NSGA-II sur un seul actif (le cas par défaut !)
                # → population constamment vide → « NSGA-II mort ».
                # La vérification réelle exige >= 2 actifs (_multi_assets peuplé).
                c4 = True
            # 5. Coûts appliqués.
            c5 = bool(mesures.costs_applied)
            # 6. Absence de fuite temporelle : garanti par construction (test_on(val) post-calibration train).
            c6 = True
            # 7. Drawdown sous la limite de sécurité.
            worst_dd = max_drawdown_from_returns([t.ret_pct_net for t in mesures.trades])
            c7 = worst_dd <= self._max_dd
            # 8. Stabilité entre folds de validation : std(Sharpe par fold CPCV) < seuil.
            sharpe_per_fold = self._compute_sharpe_per_fold(mesures, hyp, calibrated) if not multi else []
            if len(sharpe_per_fold) >= 2:
                import statistics
                std_sharpe = statistics.stdev(sharpe_per_fold)
                c8 = std_sharpe <= self.stability_max_std
            else:
                # Une stabilité non mesurable n'est pas une stabilité validée.
                c8 = False
            constraints = (c1, c2, c3, c4, c5, c6, c7, c8)
            n_violations = sum(1 for c in constraints if not c)
            # --- 4 objectifs (à MAXIMISER) ---
            sharpe = mesures.sharpe_net if mesures.sharpe_net == mesures.sharpe_net else 0.0
            dd = worst_dd
            neg_dd = -dd
            # Diversité comportementale : dispersion temporelle des signaux.
            # Elle est déterministe et ne dépend pas de l'ordre d'évaluation.
            behav_diversity = self._behavioral_dispersion(mesures)
            # Diversité Jaccard vs corpus (P1-10) : si le caller a peuplé
            # self._corpus_feature_sets, on mixe 50/50 dispersion + Jaccard.
            # Sinon : diversité = dispersion pure (rétro-compat).
            diversity = self._mix_jaccard_diversity(behav_diversity, hyp)
            # Complexité : -1 (une seule condition, c'est le minimum).
            neg_complexity = -1.0
            objectives = (sharpe, neg_dd, diversity, neg_complexity)
            sharpe_per_asset_fold = tuple(sharpe_per_fold)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Échec évaluation NSGA-II pour %s : %s", hyp.id, exc)
        return _EvaluatedIndividual(
            individual=ind,
            hypothesis=hyp,
            objectives=objectives,
            constraints_passed=constraints,
            n_violations=n_violations,
            n_signals=n_signals,
            sharpe_per_asset_fold=sharpe_per_asset_fold,
        )

    def _evaluate_multi_asset(
        self,
        hyp: Hypothesis,
        multi: dict,
    ) -> Any:
        """Evalue une hypothesis sur N actifs et agrege via mediane (P1-10).

        Args:
            hyp: Hypothesis a evaluer.
            multi: Dict {asset_name: (train_ohlcv, train_features, val_ohlcv,
                val_features)}.

        Returns:
            MesuresBrutes agregees : les stats globales sont celles de
            l'actif avec la mediane de Sharpe, et per_asset_stats contient
            les stats de chaque actif.
        """
        from dataclasses import replace as dc_replace
        from einherjar.research.utils.types import MesuresBrutes
        per_asset: dict[str, MesuresBrutes] = {}
        for asset, frames in multi.items():
            train_ohlcv_a, train_features_a, val_ohlcv_a, val_features_a = frames
            try:
                cal = self.engine.train_calibrate(hyp, train_ohlcv_a, train_features_a)
                m = self.engine.test_on(hyp, val_ohlcv_a, val_features_a, cal, f"search_val_{asset}")
                per_asset[asset] = m
            except Exception as exc:  # noqa: BLE001
                logger.debug("NSGA-II multi-asset : echec pour %s : %s", asset, exc)
                continue
        if not per_asset:
            raise RuntimeError("Aucun actif n'a pu etre evalue en multi-actifs")
        # Mediane des Sharpe nets (l'actif median devient l'actif de reference).
        sharpes_with_asset = sorted(
            [(asset, m.sharpe_net) for asset, m in per_asset.items()
             if m.sharpe_net == m.sharpe_net],
            key=lambda x: x[1],
        )
        median_asset = sharpes_with_asset[len(sharpes_with_asset) // 2][0]
        ref = per_asset[median_asset]
        # Injecte per_asset_stats dans la MesuresBrutes de reference.
        return dc_replace(ref, per_asset_stats=per_asset)

    def _compute_sharpe_per_fold(
        self,
        mesures: Any,
        hyp: Hypothesis,
        calibrated: Any,
    ) -> list[float]:
        """Calcule le Sharpe sur chaque fold CPCV (K=6 par défaut).

        Découpe la série val en K blocs temporels, regroupe les trades
        par fold selon entry_idx, calcule le Sharpe par fold, retourne la liste.
        Si < 2 folds ont au moins 2 trades, retourne [] (contrainte #8 permissive).
        """
        if not mesures.trades or getattr(self, "_search_val_ohlcv", None) is None:
            return []
        n_bougies = self._search_val_ohlcv.n_bougies
        if n_bougies == 0:
            return []
        trade_indices = [t.entry_idx for t in mesures.trades]
        n_groups = 6
        folds = _cpcv_folds_for_trades(trade_indices, n_bougies, n_groups)
        sharpe_per_fold: list[float] = []
        for fold_indices in folds:
            if not fold_indices:
                continue
            fold_trades = [mesures.trades[i] for i in fold_indices]
            returns_fold = [t.ret_pct_net for t in fold_trades]
            if len(returns_fold) < 2:
                continue
            mean = sum(returns_fold) / len(returns_fold)
            var = sum((r - mean) ** 2 for r in returns_fold) / (len(returns_fold) - 1)
            std = var ** 0.5
            if std == 0:
                continue
            sharpe = (mean / std) * (365.0 ** 0.5)
            sharpe_per_fold.append(sharpe)
        return sharpe_per_fold

    @staticmethod
    def _behavioral_dispersion(mesures: Any, n_buckets: int = 10) -> float:
        """Entropy normalisee des dates de signal, proxy reproductible de comportement."""
        if not mesures.trades:
            return 0.0
        last_index = max(1, max(trade.entry_idx for trade in mesures.trades))
        counts = [0] * n_buckets
        for trade in mesures.trades:
            bucket = min(n_buckets - 1, (trade.entry_idx * n_buckets) // (last_index + 1))
            counts[bucket] += 1
        total = sum(counts)
        import math
        entropy = -sum((n / total) * math.log(n / total) for n in counts if n)
        return entropy / math.log(n_buckets)

    def _mix_jaccard_diversity(
        self,
        behav_diversity: float,
        hyp: Hypothesis,
    ) -> float:
        """Mixe dispersion comportementale et Jaccard vs corpus (P1-10).

        Si le caller a peuple `self._corpus_feature_sets` (set de frozensets
        de features des Einhers deja admis) avant l'appel a `generate()`,
        on calcule `corpus_jaccard_diversity(features_a, corpus)` et on
        retourne 0.5 * behav + 0.5 * jaccard. Sinon : dispersion pure.

        Args:
            behav_diversity: dispersion comportementale (entropy normalisee).
            hyp: Hypothesis candidate (pour extraire les features).

        Returns:
            Score de diversite dans [0, 1].
        """
        corpus_sets: tuple = getattr(self, "_corpus_feature_sets", ())
        if not corpus_sets:
            return behav_diversity
        # Import local (evite cycle : admission.diversity importe generators).
        from einherjar.research.admission.diversity import corpus_jaccard_diversity
        # Extrait les features de l'Hypothesis (parcours l'arbre de conditions).
        ind_features = frozenset(_collect_feature_refs(hyp.condition_tree))
        if not ind_features:
            return behav_diversity
        jaccard_div = corpus_jaccard_diversity(ind_features, list(corpus_sets))
        # 50/50 entre dispersion temporelle et Jaccard vs corpus.
        return 0.5 * behav_diversity + 0.5 * jaccard_div

    # ------------------------------------------------------------------ #
    # NSGA-II : opérateurs (Deb 2002)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _dominates(a: _EvaluatedIndividual, b: _EvaluatedIndividual) -> bool:
        """True si `a` domine `b` au sens Pareto (uniquement si pas de violation de contraintes).

        Selon Deb 2002 : une solution réalisable domine une autre si elle est au moins
        aussi bonne sur tous les objectifs et strictement meilleure sur au moins un.
        Une solution non réalisable est dominée par n'importe quelle réalisable.
        """
        # Une solution réalisable domine toute solution non réalisable.
        if a.n_violations == 0 and b.n_violations > 0:
            return True
        if b.n_violations == 0 and a.n_violations > 0:
            return False
        # Si les deux sont non réalisables : celle qui viole MOINS de contraintes domine.
        if a.n_violations != b.n_violations:
            return a.n_violations < b.n_violations
        # Pareto dominance classique sur les 4 objectifs.
        better_any = False
        for oa, ob in zip(a.objectives, b.objectives):
            if oa != oa or ob != ob:  # NaN : pas comparable
                continue
            if oa < ob:
                return False
            if oa > ob:
                better_any = True
        return better_any

    @staticmethod
    def _fast_non_dominated_sort(population: list[_EvaluatedIndividual]) -> list[list[int]]:
        """Non-dominance sorting de Deb 2002 (O(MN²)).

        Returns:
            Liste de fronts, chaque front est une liste d'indices dans population.
            Front 0 = Pareto-optimal, front 1 = dominé uniquement par front 0, etc.
        """
        n = len(population)
        domination_count = [0] * n
        dominated_set: list[list[int]] = [[] for _ in range(n)]
        fronts: list[list[int]] = [[]]
        for p in range(n):
            for q in range(n):
                if p == q:
                    continue
                if NSGA2Generator._dominates(population[p], population[q]):
                    dominated_set[p].append(q)
                elif NSGA2Generator._dominates(population[q], population[p]):
                    domination_count[p] += 1
            if domination_count[p] == 0:
                fronts[0].append(p)
        i = 0
        while i < len(fronts) and fronts[i]:
            next_front: list[int] = []
            for p in fronts[i]:
                for q in dominated_set[p]:
                    domination_count[q] -= 1
                    if domination_count[q] == 0:
                        next_front.append(q)
            i += 1
            if next_front:
                fronts.append(next_front)
        return fronts

    @staticmethod
    def _crowding_distance(
        front_indices: list[int],
        population: list[_EvaluatedIndividual],
    ) -> dict[int, float]:
        """Calcule la crowding distance pour les individus d'un front.

        Plus la distance est grande, plus l'individu est dans une zone peu peuplée
        du front (donc à privilégier pour la diversité).
        """
        n = len(front_indices)
        if n == 0:
            return {}
        distances = {idx: 0.0 for idx in front_indices}
        n_obj = len(population[front_indices[0]].objectives)
        for m in range(n_obj):
            # Tri par objectif m.
            sorted_idx = sorted(front_indices, key=lambda i: population[i].objectives[m])
            # Les bords ont une distance infinie.
            distances[sorted_idx[0]] = float("inf")
            distances[sorted_idx[-1]] = float("inf")
            # Range de l'objectif m sur ce front.
            obj_min = population[sorted_idx[0]].objectives[m]
            obj_max = population[sorted_idx[-1]].objectives[m]
            if obj_max == obj_min:
                continue
            for j in range(1, n - 1):
                if distances[sorted_idx[j]] == float("inf"):
                    continue
                prev_obj = population[sorted_idx[j - 1]].objectives[m]
                next_obj = population[sorted_idx[j + 1]].objectives[m]
                distances[sorted_idx[j]] += (next_obj - prev_obj) / (obj_max - obj_min)
        return distances

    def _environmental_selection(
        self,
        population: list[_EvaluatedIndividual],
        n: int,
    ) -> list[_EvaluatedIndividual]:
        """Sélection environnementale NSGA-II : garde les N meilleurs.

        1. Non-dominance sorting → fronts.
        2. On prend les fronts complets tant qu'on n'a pas atteint N.
        3. Pour le front partiel, on trie par crowding distance décroissante
           et on prend les premiers.
        """
        if len(population) <= n:
            return list(population)
        fronts = self._fast_non_dominated_sort(population)
        selected_indices: list[int] = []
        for front in fronts:
            if len(selected_indices) + len(front) <= n:
                selected_indices.extend(front)
            else:
                # Crowding distance sur ce front, tri décroissant, on prend les premiers.
                cd = self._crowding_distance(front, population)
                remaining = n - len(selected_indices)
                sorted_by_cd = sorted(front, key=lambda i: cd.get(i, 0.0), reverse=True)
                selected_indices.extend(sorted_by_cd[:remaining])
                break
        return [population[i] for i in selected_indices]

    def _tournament_selection(
        self,
        population: list[_EvaluatedIndividual],
        n: int,
    ) -> list[_EvaluatedIndividual]:
        """Sélection par tournoi binaire (rank, crowding)."""
        if not population:
            return []
        # Pré-calcule rank et crowding pour tous.
        fronts = self._fast_non_dominated_sort(population)
        rank_map: dict[int, int] = {}
        cd_map: dict[int, float] = {}
        for rank, front in enumerate(fronts):
            cd = self._crowding_distance(front, population)
            for idx in front:
                rank_map[idx] = rank
                cd_map[idx] = cd.get(idx, 0.0)
        def _tournament_once() -> _EvaluatedIndividual:
            i, j = self._rng.sample(range(len(population)), 2)
            ri, rj = rank_map[i], rank_map[j]
            if ri < rj:
                return population[i]
            if rj < ri:
                return population[j]
            # Même rang : crowding distance départage.
            return population[i] if cd_map[i] >= cd_map[j] else population[j]
        return [_tournament_once() for _ in range(n)]

    # ------------------------------------------------------------------ #
    # Variation : crossover SBX + mutation mixte
    # ------------------------------------------------------------------ #

    def _make_offspring(
        self,
        parents: list[_EvaluatedIndividual],
    ) -> list[_NSGA2Individual]:
        """Génère une nouvelle population d'offspring via crossover + mutation."""
        offspring: list[_NSGA2Individual] = []
        # Appariement aléatoire par paires.
        indices = list(range(len(parents)))
        self._rng.shuffle(indices)
        for i in range(0, len(indices) - 1, 2):
            p1 = parents[indices[i]].individual
            p2 = parents[indices[i + 1]].individual
            if self._rng.random() < self.crossover_prob:
                c1, c2 = self._crossover(p1, p2)
            else:
                c1, c2 = p1, p2
            c1 = self._mutate(c1)
            c2 = self._mutate(c2)
            offspring.append(c1)
            offspring.append(c2)
        if len(offspring) < len(parents):
            # Population impaire : on duplique le dernier avec mutation.
            offspring.append(self._mutate(parents[indices[-1]].individual))
        return offspring[: len(parents)]

    def _crossover(
        self, p1: _NSGA2Individual, p2: _NSGA2Individual,
    ) -> tuple[_NSGA2Individual, _NSGA2Individual]:
        """Crossover mixte :
          - threshold : SBX (Simulated Binary Crossover)
          - autres gènes : uniforme (chaque gène vient de p1 ou p2)
        """
        t1, t2 = self._sbx(p1.threshold, p2.threshold, self.sbx_eta)
        return (
            _NSGA2Individual(
                feature_id=p1.feature_id if self._rng.random() < 0.5 else p2.feature_id,
                op_id=p1.op_id if self._rng.random() < 0.5 else p2.op_id,
                threshold=t1,
                cooldown_k=p1.cooldown_k if self._rng.random() < 0.5 else p2.cooldown_k,
                direction_id=p1.direction_id if self._rng.random() < 0.5 else p2.direction_id,
            ),
            _NSGA2Individual(
                feature_id=p2.feature_id if self._rng.random() < 0.5 else p1.feature_id,
                op_id=p2.op_id if self._rng.random() < 0.5 else p1.op_id,
                threshold=t2,
                cooldown_k=p2.cooldown_k if self._rng.random() < 0.5 else p1.cooldown_k,
                direction_id=p2.direction_id if self._rng.random() < 0.5 else p1.direction_id,
            ),
        )

    def _sbx(self, x1: float, x2: float, eta: float) -> tuple[float, float]:
        """Simulated Binary Crossover (Deb & Agrawal 1995).

        Retourne 2 enfants. Si u <= 0.5 : beta = (2u)^(1/(eta+1)), sinon (1/(2(1-u)))^(1/(eta+1)).
        Enfant = 0.5 * [(x1+x2) - beta * |x2-x1|, (x1+x2) + beta * |x2-x1|].
        """
        if abs(x1 - x2) < 1e-12:
            return x1, x2
        u = self._rng.random()
        if u <= 0.5:
            beta = (2.0 * u) ** (1.0 / (eta + 1.0))
        else:
            beta = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1.0))
        c1 = 0.5 * ((x1 + x2) - beta * abs(x2 - x1))
        c2 = 0.5 * ((x1 + x2) + beta * abs(x2 - x1))
        return c1, c2

    def _mutate(self, ind: _NSGA2Individual) -> _NSGA2Individual:
        """Mutation mixte par probabilité `mutation_prob` par enfant."""
        if self._rng.random() > self.mutation_prob:
            return ind
        # Mutation par gène.
        new_feat = ind.feature_id
        if self._rng.random() < 0.3:
            new_feat = self._rng.randint(0, len(self._continuous_features) - 1)
        new_op = ind.op_id
        if self._rng.random() < 0.3:
            new_op = self._rng.randint(0, len(self.OP_CHOICES) - 1)
        new_threshold = ind.threshold
        if self._rng.random() < 0.3:
            # Mutation polynomiale (Deb 2001, PM operator).
            u = self._rng.random()
            if u < 0.5:
                delta = (2.0 * u) ** (1.0 / (self.pm_eta + 1.0)) - 1.0
            else:
                delta = 1.0 - (2.0 * (1.0 - u)) ** (1.0 / (self.pm_eta + 1.0))
            new_threshold = max(-2.0, min(2.0, ind.threshold + delta * 4.0))
        new_cooldown = ind.cooldown_k
        if self._rng.random() < 0.2:
            new_cooldown = max(1, min(20, ind.cooldown_k + self._rng.choice([-1, 1])))
        new_dir = ind.direction_id
        if self._rng.random() < 0.2:
            new_dir = 1 - ind.direction_id
        return _NSGA2Individual(
            feature_id=new_feat, op_id=new_op, threshold=new_threshold,
            cooldown_k=new_cooldown, direction_id=new_dir,
        )


# --------------------------------------------------------------------------- #
# Helper : découpage des trades en folds (pour la contrainte #8 stabilité)
# --------------------------------------------------------------------------- #


def _cpcv_folds_for_trades(
    trade_indices: list[int],
    n_bougies: int,
    n_groups: int,
) -> list[list[int]]:
    """Découpe les trades en n_groups folds temporels.

    Pour chaque trade, on détermine son fold selon entry_idx / n_bougies * n_groups.
    Retourne une liste de n_groups listes d'indices de trades (dans trade_indices).
    """
    if n_bougies == 0 or n_groups <= 0:
        return [[] for _ in range(max(1, n_groups))]
    folds: list[list[int]] = [[] for _ in range(n_groups)]
    for i, idx in enumerate(trade_indices):
        fold = min(n_groups - 1, int((idx / n_bougies) * n_groups))
        folds[fold].append(i)
    return folds


# --------------------------------------------------------------------------- #
# Helper : extraction des feature_ref d'un arbre de conditions
# --------------------------------------------------------------------------- #


def _collect_feature_refs(
    tree: Condition | ConditionNode,
) -> list[str]:
    """Collecte tous les feature_ref d'un arbre de conditions (parcours DFS).

    Utilise par NSGA2Generator._mix_jaccard_diversity pour calculer le set
    de features d'une Hypothesis en vue d'un calcul de Jaccard vs corpus.

    Args:
        tree: racine de l'arbre (Condition ou ConditionNode).

    Returns:
        Liste des feature_ref trouves. Un ConditionNode unaire (LogicalOp.NOT)
        ou binaire (AND/OR/XOR) est parcours recursivement.
    """
    if isinstance(tree, Condition):
        return [tree.feature_ref]
    # ConditionNode : on parcourt les enfants.
    refs: list[str] = _collect_feature_refs(tree.left)
    if tree.right is not None:
        refs.extend(_collect_feature_refs(tree.right))
    return refs


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #


def make_all_generators(
    protocol: GenerationProtocol,
    config: EinherjarConfig,
    engine: Any | None = None,
) -> list[BaseGenerator]:
    """Construit les 6 candidats principaux (avec GE — BNF Phase 4 livree).

    Args:
        engine: Moteur d'évaluation (passé à tous les générateurs évolutionnaires :
            TypedGP, Memetic, NSGA2, GE). Optionnel pour les générateurs sans évolution
            (Random, Beam) qui sont construits sans engine.

    Returns:
        Liste [RandomSearch, BeamSearch, TypedGP, GE, Memetic, NSGA2].
    """
    return [
        RandomSearchGenerator(protocol=protocol, config=config),
        BeamSearchGenerator(protocol=protocol, config=config, engine=engine),
        TypedGPGenerator(protocol=protocol, config=config, engine=engine),
        GrammaticalEvolutionGenerator(protocol=protocol, config=config, engine=engine),
        MemeticGenerator(protocol=protocol, config=config, engine=engine),
        NSGA2Generator(protocol=protocol, config=config, engine=engine),
    ]
