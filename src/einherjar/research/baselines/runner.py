"""baselines/runner.py — Lance les 3 baselines et produit la distribution de Sharpe/PnL.

Le runner :
  1. Instancie les 3 baselines (HumanRules, ShallowEnumeration, RandomConstrained).
  2. Pour chaque baseline, génère les hypothèses.
  3. Passe chaque hypothèse dans le moteur d'évaluation (calibrate + test sur val).
  4. Agrège les MesuresBrutes en une distribution par baseline.
  5. Retourne un BaselineReport pour audit / comparaison avec les générateurs.

Conforme à ALGORITHME_RESEARCH.md § 10.2 étape 1.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from einherjar.research.baselines.algorithms import (
    BaseBaseline,
    HumanRules,
    RandomConstrained,
    ShallowEnumeration,
)
from einherjar.research.data.features import FeaturesFrame
from einherjar.research.data.ohlcv import OhlcvFrame
from einherjar.research.engine.evaluator import EvaluationEngine
from einherjar.research.utils.types import Hypothesis, MesuresBrutes

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Sortie du runner
# --------------------------------------------------------------------------- #


@dataclass
class BaselineEvalRow:
    """Une ligne = une hypothèse évaluée par le moteur."""

    hypothesis_id: str
    baseline_name: str
    mesures_val: MesuresBrutes | None       # None si la calibration a échoué
    error: str | None = None                # message d'erreur si échec

    @property
    def succeeded(self) -> bool:
        return self.mesures_val is not None and self.error is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "baseline_name": self.baseline_name,
            "succeeded": self.succeeded,
            "error": self.error,
            "mesures": self.mesures_val.to_dict() if self.mesures_val else None,
        }


@dataclass
class BaselineReport:
    """Rapport global des baselines : toutes les évaluations + statistiques agrégées."""

    rows: list[BaselineEvalRow] = field(default_factory=list)
    n_total: int = 0
    n_succeeded: int = 0
    n_failed: int = 0
    sharpe_distribution: dict[str, list[float]] = field(default_factory=dict)
    n_passed_admission: int = 0
    elapsed_s: float = 0.0

    def summary(self) -> dict[str, Any]:
        """Résumé compact pour logs / rapports."""
        return {
            "n_total": self.n_total,
            "n_succeeded": self.n_succeeded,
            "n_failed": self.n_failed,
            "n_passed_admission": self.n_passed_admission,
            "sharpe_median": {k: (sorted(v)[len(v) // 2] if v else None) for k, v in self.sharpe_distribution.items()},
            "sharpe_max": {k: (max(v) if v else None) for k, v in self.sharpe_distribution.items()},
            "elapsed_s": round(self.elapsed_s, 2),
        }


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #


class BaselineRunner:
    """Orchestre l'évaluation des baselines.

    Attributes:
        engine: Moteur d'évaluation (calibrate + test_on).
        baselines: Liste des baselines à exécuter.
    """

    def __init__(self, engine: EvaluationEngine, baselines: list[BaseBaseline] | None = None, eval_budget: int | None = None) -> None:
        self.engine = engine
        self.baselines: list[BaseBaseline] = baselines or [
            HumanRules(config=engine.config, seed=engine.seed),
            ShallowEnumeration(config=engine.config, seed=engine.seed),
            RandomConstrained(config=engine.config, seed=engine.seed),
        ]
        # Budget moteur P1-08 : si fourni, chaque baseline reçoit une part égale.
        self.eval_budget = eval_budget
        logger.info(
            "BaselineRunner instancié : %d baselines, engine=%s, eval_budget=%s",
            len(self.baselines), type(engine).__name__,
            eval_budget if eval_budget is not None else "illimité",
        )

    def run(
        self,
        train_ohlcv: OhlcvFrame,
        train_features: FeaturesFrame,
        val_ohlcv: OhlcvFrame,
        val_features: FeaturesFrame,
        *,
        admission_fn: Any | None = None,
    ) -> BaselineReport:
        """Lance toutes les baselines, évalue chaque hypothèse sur val.

        Args:
            train_ohlcv/train_features: split train (pour calibration).
            val_ohlcv/val_features: split val (pour évaluation).
            admission_fn: Callable optionnel (hypothesis, calibrated, mesures_val) -> bool.
                Si fourni, on l'appelle pour filtrer les hypothèses qui passent.
                Si None, tout succès est marqué comme admis.

        Returns:
            BaselineReport complet.
        """
        t0 = time.time()
        report = BaselineReport()
        # Part égale du budget par baseline (au moins 1) — philosophie P1-08.
        per_baseline_cap: int | None = None
        if self.eval_budget is not None:
            per_baseline_cap = max(1, self.eval_budget // max(len(self.baselines), 1))
        for baseline in self.baselines:
            logger.info("=" * 60)
            logger.info("Baseline : %s", baseline.name)
            logger.info("=" * 60)
            result = baseline.generate()
            hypotheses = result.hypotheses
            if per_baseline_cap is not None and len(hypotheses) > per_baseline_cap:
                logger.warning(
                    "Budget atteint : %d/%d hypothèses évaluées pour %s (cap=%d)",
                    per_baseline_cap, len(hypotheses), baseline.name, per_baseline_cap,
                )
                hypotheses = hypotheses[:per_baseline_cap]
            logger.info(
                "Génération OK : %d/%d hypothèses en %.2fs",
                len(hypotheses), result.n_generated, result.generation_time_s,
            )
            report.sharpe_distribution[baseline.name] = []
            for hyp in hypotheses:
                row = self._eval_one(
                    hyp, baseline.name,
                    train_ohlcv, train_features,
                    val_ohlcv, val_features,
                    admission_fn=admission_fn,
                    report=report,
                )
                report.rows.append(row)
        report.n_total = len(report.rows)
        report.n_succeeded = sum(1 for r in report.rows if r.succeeded)
        report.n_failed = report.n_total - report.n_succeeded
        report.elapsed_s = time.time() - t0
        logger.info(
            "Baselines terminées : %d/%d succès, %.2fs",
            report.n_succeeded, report.n_total, report.elapsed_s,
        )
        return report

    def _eval_one(
        self,
        hypothesis: Hypothesis,
        baseline_name: str,
        train_ohlcv: OhlcvFrame,
        train_features: FeaturesFrame,
        val_ohlcv: OhlcvFrame,
        val_features: FeaturesFrame,
        *,
        admission_fn: Any | None,
        report: BaselineReport,
    ) -> BaselineEvalRow:
        """Évalue une hypothèse sur train+val. Robuste aux échecs."""
        try:
            calibrated = self.engine.train_calibrate(hypothesis, train_ohlcv, train_features)
            mesures = self.engine.test_on(
                hypothesis, val_ohlcv, val_features, calibrated, "val",
            )
            # Enregistre le Sharpe pour la distribution.
            if not _is_nan(mesures.sharpe_net):
                report.sharpe_distribution[baseline_name].append(mesures.sharpe_net)
            # Vérifie l'admission si fournie.
            if admission_fn is not None:
                if admission_fn(hypothesis, calibrated, mesures):
                    report.n_passed_admission += 1
            else:
                report.n_passed_admission += 1
            return BaselineEvalRow(
                hypothesis_id=hypothesis.id,
                baseline_name=baseline_name,
                mesures_val=mesures,
            )
        except Exception as exc:  # noqa: BLE001 — on veut tous les messages
            logger.warning(
                "Échec évaluation %s (hyp=%s) : %s",
                baseline_name, hypothesis.id, exc,
            )
            return BaselineEvalRow(
                hypothesis_id=hypothesis.id,
                baseline_name=baseline_name,
                mesures_val=None,
                error=str(exc),
            )


def _is_nan(x: float) -> bool:
    """True si x est NaN (None n'est pas NaN)."""
    try:
        return x != x  # noqa: PLR0124 — intentionnel
    except TypeError:
        return False


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #


def make_default_runner(engine: EvaluationEngine) -> BaselineRunner:
    """Construit un BaselineRunner avec les 3 baselines par défaut."""
    return BaselineRunner(engine=engine)
