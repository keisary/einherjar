"""refinement/beam.py — Raffinement beam local autour des Einhers viables.

⚠ P1 #2 : BeamRefiner est DEPRECIE pour le sprint en cours.
Raison : il score les voisins sur val (le caller passe val_ohlcv/val_features),
ce qui constitue une fuite de selection (la regle est optimisee sur val,
donc son score val est surestime).

Directive user : "regle BNF = identite stable ; toute variante = nouveau
candidat reevalue integralement". Donc :
  - Toute "variante" generee par BeamRefiner doit etre consideree comme
    un NOUVEAU candidat, pas comme une amelioration de l'original.
  - Le scoring sur val ne doit PAS optimiser la regle : chaque variante
    doit passer par le pipeline COMPLET (calibration sur train + test sur val
    + admission) pour etre evaluee.

Pour V1, BeamRefiner reste implemente pour retrocompatibilite, mais il
emet un DeprecationWarning au premier appel. Le caller (handle_refine)
DOIT migrer vers la strategie "generer N nouvelles hypotheses via les
generateurs (Random/TypedGP/NSGA-II) et les faire passer par le pipeline
complet". C'est la migration recommandee.

Regle dure conservee : SL et TP sont figes depuis le train (S-3.2 d'ONTOLOGY.md).
Le raffinement ne peut modifier que :
  - les conditions (features, operateurs, seuils),
  - le cooldown K,
  - l'amplitude (en multiple_ATR par exemple).

Mutations appliquees :
  - Swap feature : remplace une feature par une autre du meme type.
  - Swap operator : < → >, <= → >=, == → !=.
  - Tweak threshold : ±10%, ±25%, ±50%.
  - Tweak cooldown : K±1.

Stratégie : beam search de largeur K, profondeur max 2 niveaux, 100
itérations max OU plus aucune amélioration.

Conforme à ONTOLOGY.md S-3.5 et ALGORITHME_RESEARCH.md § 10.2 étape 4.
"""

from __future__ import annotations

import copy
import logging
import random
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from einherjar.research.config.loader import EinherjarConfig
from einherjar.research.data.features import FeaturesFrame
from einherjar.research.data.ohlcv import OhlcvFrame
from einherjar.research.engine.evaluator import CalibratedParams, EvaluationEngine
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
    MesuresBrutes,
    Universe,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Types
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RefinementResult:
    """Résultat du raffinement d'un Einher."""

    original_hypothesis_id: str
    refined_hypotheses: tuple[Hypothesis, ...]
    best_hypothesis: Hypothesis | None
    best_sharpe_val: float
    n_evaluated: int
    n_iterations: int
    improved: bool
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_hypothesis_id": self.original_hypothesis_id,
            "n_refined": len(self.refined_hypotheses),
            "best_sharpe_val": self.best_sharpe_val,
            "n_evaluated": self.n_evaluated,
            "n_iterations": self.n_iterations,
            "improved": self.improved,
            "meta": self.meta,
        }


# --------------------------------------------------------------------------- #
# Base refiner
# --------------------------------------------------------------------------- #


class BaseRefiner(ABC):
    """Interface commune des refineurs.

    ⚠ P1 #2 : DEPRECIE pour V2. Voir docstring du module.
    Le caller DOIT migrer vers la strategie "generer N nouvelles hypotheses
    via les generateurs (Random/TypedGP/NSGA-II) et les faire passer par le
    pipeline complet (calibration + test + admission)".
    """

    def __init__(self, config: EinherjarConfig, engine: EvaluationEngine, seed: int = 42) -> None:
        self.config = config
        self.engine = engine
        self.seed = seed
        self._rng = random.Random(seed)
        self.name: str = type(self).__name__
        # P1 #2 : un seul warning par process (pas un warning par appel).
        import warnings
        if not getattr(self.__class__, "_p1_2_warned", False):
            warnings.warn(
                f"{self.name} est déprécié (P1 #2). Il score les voisins sur val, "
                "ce qui constitue une fuite de sélection. Migrez vers la "
                "génération de nouveaux candidats (Random/TypedGP/NSGA-II) + "
                "pipeline complet (calibration + test + admission).",
                DeprecationWarning,
                stacklevel=2,
            )
            self.__class__._p1_2_warned = True
        logger.info("Refiner instancié : %s (seed=%d, ⚠ P1 #2 deprécié)", self.name, seed)

    @abstractmethod
    def refine(
        self,
        hypothesis: Hypothesis,
        calibrated: CalibratedParams,
        train_ohlcv: OhlcvFrame,
        train_features: FeaturesFrame,
        val_ohlcv: OhlcvFrame,
        val_features: FeaturesFrame,
    ) -> RefinementResult:
        """Raffine un Einher (sans recalibrer SL/TP).

        Returns:
            RefinementResult avec les hypothèses raffinées.
        """
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Beam search refiner
# --------------------------------------------------------------------------- #


class BeamRefiner(BaseRefiner):
    """Beam search local 1-paramètre-à-la-fois.

    Pour chaque itération :
      1. Génère K=beam_width voisins de l'Einher courant (mutations).
      2. Évalue chaque voisin sur val (calibré sur train, sans recalibrer).
      3. Garde le top 1 (le meilleur) pour l'itération suivante.
      4. Arrêt si plus d'amélioration OU max_iterations atteint.
    """

    def __init__(
        self,
        config: EinherjarConfig,
        engine: EvaluationEngine,
        seed: int = 42,
        beam_width: int = 8,
        max_iterations: int = 100,
        improvement_threshold: float = 1e-4,
    ) -> None:
        super().__init__(config=config, engine=engine, seed=seed)
        self.beam_width = beam_width
        self.max_iterations = max_iterations
        self.improvement_threshold = improvement_threshold

    def refine(
        self,
        hypothesis: Hypothesis,
        calibrated: CalibratedParams,
        train_ohlcv: OhlcvFrame,
        train_features: FeaturesFrame,
        val_ohlcv: OhlcvFrame,
        val_features: FeaturesFrame,
    ) -> RefinementResult:
        # Évalue l'Einher original (baseline).
        original_mesures = self.engine.test_on(
            hypothesis, val_ohlcv, val_features, calibrated, "val",
        )
        best_h = hypothesis
        best_sharpe = original_mesures.sharpe_net if original_mesures.sharpe_net == original_mesures.sharpe_net else 0.0
        all_refined: list[Hypothesis] = []
        n_evaluated = 0
        improved_overall = False

        for it in range(self.max_iterations):
            # Génère K voisins.
            neighbors = self._generate_neighbors(best_h)
            if not neighbors:
                logger.info("BeamRefiner it=%d : plus de voisins possibles, arrêt.", it)
                break
            # Évalue les voisins et garde le meilleur.
            best_neighbor: Hypothesis | None = None
            best_neighbor_sharpe = best_sharpe
            for n_h in neighbors[: self.beam_width]:
                try:
                    # Any condition or amplitude mutation changes the train
                    # calibration contract; reusing the parent's parameters
                    # would compare different strategies with stale SL/TP/N.
                    neighbor_calibrated = self.engine.train_calibrate(
                        n_h, train_ohlcv, train_features,
                    )
                    m = self.engine.test_on(
                        n_h, val_ohlcv, val_features, neighbor_calibrated, "val",
                    )
                    n_evaluated += 1
                    if m.sharpe_net != m.sharpe_net:
                        continue
                    if m.sharpe_net > best_neighbor_sharpe + self.improvement_threshold:
                        best_neighbor = n_h
                        best_neighbor_sharpe = m.sharpe_net
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Échec évaluation voisin : %s", exc)
            if best_neighbor is None:
                logger.info("BeamRefiner it=%d : pas d'amélioration, arrêt.", it)
                break
            # Amélioration.
            delta = best_neighbor_sharpe - best_sharpe
            logger.info(
                "BeamRefiner it=%d : Sharpe %.4f → %.4f (delta=%.4f)",
                it, best_sharpe, best_neighbor_sharpe, delta,
            )
            best_h = best_neighbor
            best_sharpe = best_neighbor_sharpe
            all_refined.append(best_neighbor)
            improved_overall = True

        return RefinementResult(
            original_hypothesis_id=hypothesis.id,
            refined_hypotheses=tuple(all_refined),
            best_hypothesis=best_h if improved_overall else None,
            best_sharpe_val=best_sharpe,
            n_evaluated=n_evaluated,
            n_iterations=min(it + 1, self.max_iterations) if all_refined else 0,
            improved=improved_overall,
            meta={"beam_width": self.beam_width, "max_iterations": self.max_iterations},
        )

    def _generate_neighbors(self, h: Hypothesis) -> list[Hypothesis]:
        """Génère des voisins par mutation locale.

        Mutate :
          - 1-feature (atomic) : swap feature, swap operator, tweak threshold.
          - ConditionNode (composé) : descend dans l'arbre, mute une feuille.
        """
        neighbors: list[Hypothesis] = []
        # Mute la condition principale.
        new_cond = self._mutate_condition(h.condition_tree)
        if new_cond is None:
            return neighbors
        # Construit un nouvel Einher avec la même amplitude/direction/universe/calibrated.
        new_h = Hypothesis(
            id=f"refined_{h.id}_{len(neighbors):04d}",
            condition_tree=new_cond,
            amplitude=h.amplitude,
            direction=h.direction,
            universe=h.universe,
            cooldown_k=h.cooldown_k,
            meta={**h.meta, "parent_id": h.id, "mutation": "feature_or_op_or_threshold"},
        )
        neighbors.append(new_h)
        # Mute le cooldown.
        for delta in (-1, 1, -2, 2):
            new_k = max(1, h.cooldown_k + delta)
            if new_k == h.cooldown_k:
                continue
            n_h = Hypothesis(
                id=f"refined_{h.id}_k{new_k}",
                condition_tree=h.condition_tree,
                amplitude=h.amplitude,
                direction=h.direction,
                universe=h.universe,
                cooldown_k=new_k,
                meta={**h.meta, "parent_id": h.id, "mutation": f"cooldown_{delta}"},
            )
            neighbors.append(n_h)
        # Mute l'amplitude (uniquement en multiple_ATR, sinon on laisse).
        if h.amplitude.unité == AmplitudeUnit.MULTIPLE_ATR:
            for delta in (-0.5, 0.5, -1.0, 1.0):
                new_val = max(0.1, h.amplitude.valeur + delta)
                new_amp = Amplitude(
                    valeur=new_val,
                    unité=h.amplitude.unité,
                    direction_implicite=h.amplitude.direction_implicite,
                )
                n_h = Hypothesis(
                    id=f"refined_{h.id}_a{new_val:.1f}",
                    condition_tree=h.condition_tree,
                    amplitude=new_amp,
                    direction=h.direction,
                    universe=h.universe,
                    cooldown_k=h.cooldown_k,
                    meta={**h.meta, "parent_id": h.id, "mutation": f"amp_{delta}"},
                )
                neighbors.append(n_h)
        return neighbors

    def _mutate_condition(
        self,
        cond: Condition | ConditionNode,
    ) -> Condition | ConditionNode | None:
        """Mute la condition : swap feature, swap operator, ou tweak threshold.

        Pour ConditionNode, on descend dans l'arbre avec probabilité 0.5.
        """
        if isinstance(cond, Condition):
            return self._mutate_atomic(cond)
        # Noeud composé : on mute une feuille au hasard.
        if self._rng.random() < 0.5 and isinstance(cond.left, Condition):
            new_left = self._mutate_atomic(cond.left)
            if new_left is not None:
                return ConditionNode(op=cond.op, left=new_left, right=cond.right)
        if cond.right is not None and isinstance(cond.right, Condition):
            new_right = self._mutate_atomic(cond.right)
            if new_right is not None:
                return ConditionNode(op=cond.op, left=cond.left, right=new_right)
        return None

    def _mutate_atomic(self, c: Condition) -> Condition | None:
        """Mute une condition atomique."""
        # Tire un type de mutation.
        mtype = self._rng.choice(["feature", "operator", "threshold"])
        if mtype == "feature":
            new_feat = self._swap_feature(c.feature_ref)
            if new_feat is None:
                return None
            return Condition(
                feature_ref=new_feat,
                operator=c.operator,
                value=c.value,
                transformation=c.transformation,
            )
        if mtype == "operator":
            new_op = self._swap_operator(c.operator)
            if new_op is None:
                return None
            return Condition(
                feature_ref=c.feature_ref,
                operator=new_op,
                value=c.value,
                transformation=c.transformation,
            )
        if mtype == "threshold":
            new_val = self._tweak_threshold(c.value)
            return Condition(
                feature_ref=c.feature_ref,
                operator=c.operator,
                value=new_val,
                transformation=c.transformation,
            )
        return None

    def _swap_feature(self, current: str) -> str | None:
        """Remplace la feature par une autre du même type (utilisable)."""
        info = self.config.features_taxonomy.get("features", {}).get(current, {})
        current_type_str = info.get("feature_type")
        try:
            current_type = FeatureType(current_type_str) if current_type_str else None
        except ValueError:
            return None
        if current_type is None:
            return None
        # Trouve une autre feature utilisable du même type.
        candidates = [
            f for f in self.config.usable_feature_names
            if f != current
            and (self.config.features_taxonomy.get("features", {}).get(f, {}).get("feature_type") == current_type_str)
        ]
        if not candidates:
            return None
        return self._rng.choice(candidates)

    @staticmethod
    def _swap_operator(op: CompareOp) -> CompareOp | None:
        """Inverse l'opérateur (LT ↔ GT, LE ↔ GE, EQ ↔ NE)."""
        swap = {
            CompareOp.LT: CompareOp.GT,
            CompareOp.GT: CompareOp.LT,
            CompareOp.LE: CompareOp.GE,
            CompareOp.GE: CompareOp.LE,
            CompareOp.EQ: CompareOp.NE,
            CompareOp.NE: CompareOp.EQ,
        }
        return swap.get(op)

    def _tweak_threshold(self, value: Any, multipliers: Sequence[float] | None = None) -> Any:
        """Tweak ±10%, ±25% ou ±50% (équivalent à ±0.1, ±0.25, ±0.5 en fraction).

        Methode d'instance (utilise self._rng pour reproductibilite via seed).
        """
        if not isinstance(value, (int, float)):
            return value
        mults = list(multipliers) if multipliers is not None else [0.5, 0.75, 0.9, 1.1, 1.25, 1.5]
        m = self._rng.choice(mults)
        return round(float(value) * m, 4)


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #


def make_default_refiner(
    config: EinherjarConfig,
    engine: EvaluationEngine,
    seed: int = 42,
) -> BeamRefiner:
    """Construit un BeamRefiner avec les défauts."""
    return BeamRefiner(config=config, engine=engine, seed=seed)
