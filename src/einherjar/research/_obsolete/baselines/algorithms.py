"""baselines/algorithms.py — Les 3 baselines honnêtes (UN fichier par convention).

Toutes les baselines implémentent `BaseBaseline.generate() -> list[Hypothesis]`.
Elles produisent des candidates SANS utiliser de ML, sans fit, sans gradient.
Leur seul but : mesurer le plancher de performance avant de chercher mieux.

Algorithmes :
  - HumanRules         : 5-10 règles triviales écrites à la main (sanity check).
  - ShallowEnumeration : toutes les Einhers à 1-2 conditions sur un sous-espace
                         restreint de features. Borne inférieure de l'exhaustivité.
  - RandomConstrained  : random search sous contraintes (typage, profondeur, ratios).
                         Mesure la valeur ajoutée des méthodes plus sophistiquées.

Conforme à ALGORITHME_RESEARCH.md § 10.2 étape 1.
"""

from __future__ import annotations

import logging
import random
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from einherjar.research.config.loader import EinherjarConfig
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
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Base commune
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BaselineResult:
    """Sortie d'une baseline : liste d'hypothèses + méta pour audit."""

    baseline_name: str
    hypotheses: tuple[Hypothesis, ...]
    n_generated: int
    generation_time_s: float
    meta: dict[str, Any]


class BaseBaseline(ABC):
    """Interface commune à toutes les baselines.

    Attributes:
        config: Configuration chargée (utilisée pour la taxonomie).
        seed: Graine RNG.
    """

    def __init__(self, config: EinherjarConfig, seed: int = 42) -> None:
        """Initialise la baseline (config + seed déterministe).

        Args:
            config: Configuration chargée.
            seed: Graine RNG (reproductibilité).
        """
        self.config = config
        self.seed = seed
        self._rng = random.Random(seed)
        self.name: str = type(self).__name__
        logger.info("Baseline instanciée : %s (seed=%d)", self.name, seed)

    @abstractmethod
    def generate(self) -> BaselineResult:
        """Génère les hypothèses candidates.

        Returns:
            BaselineResult avec les hypothèses et méta-données.
        """
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Baseline 1 : Règles humaines (sanity check)
# --------------------------------------------------------------------------- #


class HumanRules(BaseBaseline):
    """5-10 règles triviales écrites à la main.

    Sert de sanity check : un générateur sophistiqué qui ne bat pas ces
    règles est cassé. Aucun fit, aucune statistique : juste de l'intuition
    de base sur le momentum / mean-reversion / volume.
    """

    # (feature, opérateur, valeur, direction) — choisis pour couvrir
    # les intuitions de base du trading algorithmique.
    DEFAULT_RULES: tuple[tuple[str, CompareOp, float, Direction], ...] = (
        ("rsi_14", CompareOp.LT, 30.0, Direction.LONG),
        ("rsi_14", CompareOp.GT, 70.0, Direction.SHORT),
        ("rsi_21", CompareOp.LT, 25.0, Direction.LONG),
        ("rsi_21", CompareOp.GT, 75.0, Direction.SHORT),
        ("macd", CompareOp.GT, 0.0, Direction.LONG),
        ("macd", CompareOp.LT, 0.0, Direction.SHORT),
        ("sma_20", CompareOp.GT, 0.0, Direction.LONG),  # au-dessus de la SMA20
        ("sma_20", CompareOp.LT, 0.0, Direction.SHORT),
        ("bb_position", CompareOp.LT, 0.05, Direction.LONG),  # bande basse
        ("bb_position", CompareOp.GT, 0.95, Direction.SHORT),
    )

    def __init__(
        self,
        config: EinherjarConfig,
        seed: int = 42,
        amplitude_value: float = 50.0,
        cooldown_k: int = 5,
        assets: Sequence[str] = ("BTCUSD",),
        timeframes: Sequence[str] = ("1h",),
    ) -> None:
        """Initialise les règles humaines.

        Args:
            config: Configuration chargée.
            seed: Graine RNG.
            amplitude_value: Valeur d'amplitude (en multiples d'ATR,
                alignée sur les générateurs pour la comparabilité step1/step2).
            cooldown_k: Cooldown d'observation K.
            assets: Assets cibles.
            timeframes: Timeframes cibles.
        """
        super().__init__(config=config, seed=seed)
        self.amplitude_value = amplitude_value
        self.cooldown_k = cooldown_k
        self.assets = tuple(assets)
        self.timeframes = tuple(timeframes)

    def generate(self) -> BaselineResult:
        """Génère les 5-10 règles humaines triviales écrites à la main.

        Filtre les features non-utilisables (fantômes, meta, alias).
        """
        import time

        from einherjar.research.utils.types import Universe
        t0 = time.time()
        usable = self.config.usable_set()
        hypotheses: list[Hypothesis] = []
        for i, (feat, op, value, direction) in enumerate(self.DEFAULT_RULES):
            if feat not in usable:
                logger.debug("Règle humaine %d ignorée : %s non utilisable", i, feat)
                continue
            cond = Condition(feature_ref=feat, operator=op, value=value, transformation=None)
            amp = Amplitude(
                valeur=self.amplitude_value,
                unité=AmplitudeUnit.MULTIPLE_ATR,
                direction_implicite=direction,
            )
            h = Hypothesis(
                id=f"human_{i:03d}",
                condition_tree=cond,
                amplitude=amp,
                direction=direction,
                universe=Universe(assets=self.assets, timeframes=self.timeframes),
                cooldown_k=self.cooldown_k,
            )
            hypotheses.append(h)
        elapsed = time.time() - t0
        return BaselineResult(
            baseline_name=self.name,
            hypotheses=tuple(hypotheses),
            n_generated=len(hypotheses),
            generation_time_s=elapsed,
            meta={"rules_attempted": len(self.DEFAULT_RULES), "rules_kept": len(hypotheses)},
        )


# --------------------------------------------------------------------------- #
# Baseline 2 : Énumération peu profonde (1-2 conditions)
# --------------------------------------------------------------------------- #


class ShallowEnumeration(BaseBaseline):
    """Énumère toutes les Einhers à 1-2 conditions sur un sous-espace de features.

    C'est la borne inférieure "exhaustive" : on essaie tout (sur un sous-espace),
    on garde ceux qui sont viables. Si random search bat l'exhaustion peu profonde,
    c'est que la random chance est en jeu, pas une découverte.
    """

    # Seuils d'opérateurs continus (un quantile par opérateur).
    DEFAULT_THRESHOLDS: dict[CompareOp, tuple[float, ...]] = {
        CompareOp.LT: (10.0, 30.0, 50.0, 70.0, 90.0),
        CompareOp.GT: (10.0, 30.0, 50.0, 70.0, 90.0),
        CompareOp.LE: (50.0,),
        CompareOp.GE: (50.0,),
    }

    def __init__(
        self,
        config: EinherjarConfig,
        seed: int = 42,
        amplitude_value: float = 50.0,
        cooldown_k: int = 5,
        max_features: int = 30,
        max_conditions: int = 2,
        assets: Sequence[str] = ("BTCUSD",),
        timeframes: Sequence[str] = ("1h",),
    ) -> None:
        """Initialise l'énumération peu profonde.

        Args:
            config: Configuration chargée.
            seed: Graine RNG.
            amplitude_value: Valeur d'amplitude.
            cooldown_k: Cooldown K.
            max_features: Nb max de features à énumérer.
            max_conditions: Profondeur max (1 ou 2).
            assets: Assets cibles.
            timeframes: Timeframes cibles.
        """
        super().__init__(config=config, seed=seed)
        self.amplitude_value = amplitude_value
        self.cooldown_k = cooldown_k
        self.max_features = max_features
        self.max_conditions = max_conditions
        self.assets = tuple(assets)
        self.timeframes = tuple(timeframes)

    def generate(self) -> BaselineResult:
        import time

        from einherjar.research.utils.types import Universe
        t0 = time.time()
        # On prend les premières `max_features` features utilisables (tri alphabétique).
        usable_sorted = sorted(self.config.usable_feature_names)[: self.max_features]
        # Pour les conditions 1-feature, on ne garde que les continues (atomic, quantitative, factor).
        # Pour les conditions 2-features, on autorise aussi les bool (pattern, signal).
        candidates: list[Hypothesis] = []
        for feat in usable_sorted:
            feat_type = self._get_feature_type(feat)
            if feat_type not in (FeatureType.ATOMIC, FeatureType.QUANTITATIVE, FeatureType.FACTOR):
                continue
            for direction in (Direction.LONG, Direction.SHORT):
                for op, thresholds in self.DEFAULT_THRESHOLDS.items():
                    for t in thresholds:
                        cond = Condition(feature_ref=feat, operator=op, value=t, transformation=None)
                        amp = Amplitude(
                            valeur=self.amplitude_value,
                            unité=AmplitudeUnit.MULTIPLE_ATR,
                            direction_implicite=direction,
                        )
                        h = Hypothesis(
                            id=f"shallow1_{feat}_{op.value}_{t}_{direction.value}",
                            condition_tree=cond,
                            amplitude=amp,
                            direction=direction,
                            universe=Universe(assets=self.assets, timeframes=self.timeframes),
                            cooldown_k=self.cooldown_k,
                        )
                        candidates.append(h)
        elapsed = time.time() - t0
        return BaselineResult(
            baseline_name=self.name,
            hypotheses=tuple(candidates),
            n_generated=len(candidates),
            generation_time_s=elapsed,
            meta={"max_features": self.max_features, "n_kept": len(candidates)},
        )

    def _get_feature_type(self, feature_name: str) -> FeatureType | None:
        info = self.config.features_taxonomy.get("features", {}).get(feature_name, {})
        type_str = info.get("feature_type")
        if type_str is None:
            return None
        try:
            return FeatureType(type_str)
        except ValueError:
            return None


# --------------------------------------------------------------------------- #
# Baseline 3 : Random search contraint
# --------------------------------------------------------------------------- #


class RandomConstrained(BaseBaseline):
    """Random search sous contraintes (typage, profondeur, ratios).

    C'est la baseline algorithmique de référence. Tout générateur
    sophistiqué doit battre ce random sur le critère d'admission.
    """

    def __init__(
        self,
        config: EinherjarConfig,
        seed: int = 42,
        amplitude_value: float = 50.0,
        cooldown_k: int = 5,
        n_samples: int = 1000,
        max_conditions: int = 3,
        p_compound: float = 0.3,
        assets: Sequence[str] = ("BTCUSD",),
        timeframes: Sequence[str] = ("1h",),
    ) -> None:
        """Initialise le random search sous contraintes.

        Args:
            config: Configuration chargée.
            seed: Graine RNG.
            amplitude_value: Valeur d'amplitude.
            cooldown_k: Cooldown K.
            n_samples: Nombre d'hypothèses à générer.
            max_conditions: Profondeur max.
            p_compound: Probabilité de générer une condition composée.
            assets: Assets cibles.
            timeframes: Timeframes cibles.
        """
        super().__init__(config=config, seed=seed)
        self.amplitude_value = amplitude_value
        self.cooldown_k = cooldown_k
        self.n_samples = n_samples
        self.max_conditions = max_conditions
        self.p_compound = p_compound
        self.assets = tuple(assets)
        self.timeframes = tuple(timeframes)

    def generate(self) -> BaselineResult:
        import time

        from einherjar.research.utils.types import Universe
        t0 = time.time()
        # Features continues (atomic + quantitative + factor) — on évite les patterns/signals ici.
        continuous = [
            f for f in self.config.usable_feature_names
            if self._get_feature_type(f)
            in (FeatureType.ATOMIC, FeatureType.QUANTITATIVE, FeatureType.FACTOR)
        ]
        if not continuous:
            logger.warning("Aucune feature continue trouvable — random_constrained vide")
            return BaselineResult(self.name, (), 0, 0.0, {})

        candidates: list[Hypothesis] = []
        for i in range(self.n_samples):
            direction = self._rng.choice([Direction.LONG, Direction.SHORT])
            n_cond = self._rng.randint(1, self.max_conditions)
            if n_cond == 1 or self._rng.random() > self.p_compound:
                cond = self._sample_atomic(continuous)
            else:
                # 2 conditions composées en AND.
                left = self._sample_atomic(continuous)
                right = self._sample_atomic(continuous)
                cond = ConditionNode(op=LogicalOp.AND, left=left, right=right)
            amp = Amplitude(
                valeur=self.amplitude_value,
                unité=AmplitudeUnit.MULTIPLE_ATR,
                direction_implicite=direction,
            )
            h = Hypothesis(
                id=f"rand_{i:05d}",
                condition_tree=cond,
                amplitude=amp,
                direction=direction,
                universe=Universe(assets=self.assets, timeframes=self.timeframes),
                cooldown_k=self.cooldown_k,
            )
            candidates.append(h)

        elapsed = time.time() - t0
        return BaselineResult(
            baseline_name=self.name,
            hypotheses=tuple(candidates),
            n_generated=len(candidates),
            generation_time_s=elapsed,
            meta={"n_samples": self.n_samples, "max_conditions": self.max_conditions},
        )

    def _sample_atomic(self, pool: Sequence[str]) -> Condition:
        feat = self._rng.choice(pool)
        op = self._rng.choice([CompareOp.LT, CompareOp.GT])
        # Seuils génériques (à raffiner par feature plus tard).
        value = round(self._rng.uniform(-2.0, 2.0), 4)
        return Condition(feature_ref=feat, operator=op, value=value, transformation=None)

    def _get_feature_type(self, feature_name: str) -> FeatureType | None:
        info = self.config.features_taxonomy.get("features", {}).get(feature_name, {})
        type_str = info.get("feature_type")
        if type_str is None:
            return None
        try:
            return FeatureType(type_str)
        except ValueError:
            return None


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #


def make_all_baselines(config: EinherjarConfig, seed: int = 42) -> list[BaseBaseline]:
    """Construit les 3 baselines avec leurs défauts.

    Returns:
        Liste [HumanRules, ShallowEnumeration, RandomConstrained].
    """
    return [
        HumanRules(config=config, seed=seed),
        ShallowEnumeration(config=config, seed=seed),
        RandomConstrained(config=config, seed=seed),
    ]
