"""holdout/evaluator.py — Évaluation finale unique sur le holdout (SACRÉ).

Règles dures (ONTOLOGY.md S-3.8) :
  - Le holdout n'est consulté qu'UNE SEULE FOIS par Einher final retenu.
  - Le résultat est publié tel quel, sans recalibrage.
  - N, SL, TP sont figés depuis le train (S-3.2).
  - Toute différence importante entre val et holdout déclenche un flag
    mais ne modifie pas l'admission (le holdout n'a pas le droit de
    "réparer" une admission contestable).

L'évaluateur :
  1. Fait UNE passe sur le holdout via le moteur d'évaluation.
  2. Publie les métriques + IC bootstrap + descripteurs comportementaux.
  3. Archive le résultat avec data_version / seed / splits figés.
  4. Garantit l'unicité d'accès au holdout (lève une erreur au 2e appel).

Conforme à ONTOLOGY.md S-3.8 et ALGORITHME_RESEARCH.md § 10.2 étape 6.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from einherjar.research.config.loader import EinherjarConfig
from einherjar.research.data.features import FeaturesFrame
from einherjar.research.data.ohlcv import OhlcvFrame
from einherjar.research.engine.evaluator import CalibratedParams, EvaluationEngine
from einherjar.research.utils.types import MesuresBrutes

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Types
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class HoldoutResult:
    """Résultat de l'évaluation finale sur le holdout."""

    hypothesis_id: str
    metrics_holdout: MesuresBrutes
    metrics_val_snapshot: dict[str, Any]       # pour comparaison val → holdout
    degradation_flag: str                       # 'OK' | 'WARNING' | 'CRITICAL'
    degradation_ratio: float                    # abs((sharpe_val - sharpe_holdout) / sharpe_val)
    timestamp: str                              # ISO 8601 UTC
    data_version: str
    seed: int
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "metrics_holdout": self.metrics_holdout.to_dict(),
            "metrics_val_snapshot": self.metrics_val_snapshot,
            "degradation_flag": self.degradation_flag,
            "degradation_ratio": self.degradation_ratio,
            "timestamp": self.timestamp,
            "data_version": self.data_version,
            "seed": self.seed,
            "meta": self.meta,
        }


# --------------------------------------------------------------------------- #
# Evaluator
# --------------------------------------------------------------------------- #


class HoldoutEvaluator:
    """Wrapper d'évaluation finale sur le holdout (UNE seule passe).

    Attributes:
        engine: Moteur d'évaluation (calibré depuis train).
        config: Configuration chargée.
        data_version: Identifiant de version de données.
        seed: Graine RNG maître.
        degradation_warning_ratio: Seuil de Sharpe loss pour WARNING (défaut 0.30).
        degradation_critical_ratio: Seuil de Sharpe loss pour CRITICAL (défaut 0.60).
    """

    def __init__(
        self,
        engine: EvaluationEngine,
        config: EinherjarConfig,
        data_version: str,
        seed: int = 42,
        degradation_warning_ratio: float = 0.30,
        degradation_critical_ratio: float = 0.60,
    ) -> None:
        self.engine = engine
        self.config = config
        self.data_version = data_version
        self.seed = seed
        self.degradation_warning_ratio = degradation_warning_ratio
        self.degradation_critical_ratio = degradation_critical_ratio
        # Drapeau d'accès au holdout (True après le 1er evaluate).
        self._holdout_used: bool = False
        logger.info(
            "HoldoutEvaluator instancié (data_version=%s, seed=%d, "
            "warn=%.2f, crit=%.2f)",
            data_version, seed, degradation_warning_ratio, degradation_critical_ratio,
        )

    def evaluate(
        self,
        hypothesis: Any,                      # Hypothesis
        calibrated: CalibratedParams,
        holdout_ohlcv: OhlcvFrame,
        holdout_features: FeaturesFrame,
        val_sharpe: float,
        val_metrics_snapshot: dict[str, Any] | None = None,
    ) -> HoldoutResult:
        """Évalue UN Einher sur le holdout (UNE SEULE FOIS).

        Args:
            hypothesis: L'Einher final retenu.
            calibrated: CalibratedParams figée depuis train.
            holdout_ohlcv: Frame OHLCV du holdout.
            holdout_features: FeaturesFrame du holdout.
            val_sharpe: Sharpe sur val (pour calculer la dégradation).
            val_metrics_snapshot: Snapshot des métriques val (pour traçabilité).

        Returns:
            HoldoutResult avec les métriques holdout + flag de dégradation.

        Raises:
            RuntimeError: si evaluate() est appelé une 2e fois (le holdout est sacré).
        """
        if self._holdout_used:
            raise RuntimeError(
                "HoldoutEvaluator.evaluate() ne peut être appelé qu'UNE SEULE FOIS. "
                "Le holdout est sacré (ONTOLOGY.md S-3.8, ALGORITHME_RESEARCH.md § 10.2 étape 6). "
                "Si tu veux comparer plusieurs Einhers finaux, IL FAUT un nouveau holdout "
                "(et donc un nouveau data_version)."
            )
        self._holdout_used = True
        timestamp = datetime.now(timezone.utc).isoformat()
        logger.warning(
            "⚠ HOLDOUT ÉVALUATION : %s, hypothesis=%s, timestamp=%s",
            holdout_ohlcv.asset, hypothesis.id, timestamp,
        )
        # Évaluation unique sur le holdout (via le moteur, qui tracke déjà 1 accès max).
        metrics_holdout = self.engine.test_on(
            hypothesis, holdout_ohlcv, holdout_features, calibrated, "holdout",
        )
        # Calcul de la dégradation val → holdout.
        holdout_sharpe = metrics_holdout.sharpe_net
        if holdout_sharpe != holdout_sharpe:  # NaN
            degradation_ratio = float("inf")
        elif val_sharpe > 0:
            degradation_ratio = abs((val_sharpe - holdout_sharpe) / val_sharpe)
        else:
            degradation_ratio = 0.0
        # Flag de dégradation.
        if degradation_ratio >= self.degradation_critical_ratio:
            flag = "CRITICAL"
        elif degradation_ratio >= self.degradation_warning_ratio:
            flag = "WARNING"
        else:
            flag = "OK"
        logger.info(
            "Holdout OK : %s — Sharpe val=%.4f, holdout=%.4f, degradation=%.2f%%, flag=%s",
            hypothesis.id, val_sharpe, holdout_sharpe, degradation_ratio * 100, flag,
        )
        return HoldoutResult(
            hypothesis_id=hypothesis.id,
            metrics_holdout=metrics_holdout,
            metrics_val_snapshot=val_metrics_snapshot or {},
            degradation_flag=flag,
            degradation_ratio=degradation_ratio,
            timestamp=timestamp,
            data_version=self.data_version,
            seed=self.seed,
            meta={"degradation_warning_ratio": self.degradation_warning_ratio,
                  "degradation_critical_ratio": self.degradation_critical_ratio},
        )


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #


def make_default_holdout_evaluator(
    engine: EvaluationEngine,
    config: EinherjarConfig,
    data_version: str,
    seed: int = 42,
) -> HoldoutEvaluator:
    """Construit un HoldoutEvaluator avec les défauts."""
    return HoldoutEvaluator(
        engine=engine, config=config, data_version=data_version, seed=seed,
    )
