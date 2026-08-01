"""generators/comparator.py — Compare les générateurs et classe les résultats.

Le comparator applique le protocole reproductible à chaque générateur,
collecte les GeneratorResult, et produit un GeneratorRanking qui classe
les candidats sur le critère principal (taux d'admission × qualité
médiane × diversité comportementale).

Conforme à ALGORITHME_RESEARCH.md § 10.2 étape 2.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from einherjar.research.config.loader import EinherjarConfig
from einherjar.research.engine.evaluator import EvaluationEngine
from einherjar.research.generators.algorithms import (
    BaseGenerator,
    GeneratorResult,
)
from einherjar.research.generators.protocol import GenerationProtocol

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Sortie
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GeneratorRanking:
    """Classement d'un générateur par le comparator."""

    generator_name: str
    rank: int                                  # 1 = meilleur
    score: float                               # score composite pour le classement
    n_generated: int
    n_evaluated: int
    n_passed_admission: int
    admission_rate: float                      # n_passed_admission / n_evaluated
    median_sharpe: float                       # Sharpe médian des Einhers admis
    median_sharpe_all: float                   # Sharpe médian de TOUS les évalués
    elapsed_s: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "generator_name": self.generator_name,
            "rank": self.rank,
            "score": self.score,
            "n_generated": self.n_generated,
            "n_evaluated": self.n_evaluated,
            "n_passed_admission": self.n_passed_admission,
            "admission_rate": self.admission_rate,
            "median_sharpe": self.median_sharpe,
            "median_sharpe_all": self.median_sharpe_all,
            "elapsed_s": round(self.elapsed_s, 3),
        }


@dataclass
class ComparisonReport:
    """Rapport global de comparaison."""

    protocol: GenerationProtocol
    rankings: list[GeneratorRanking] = field(default_factory=list)
    raw_results: dict[str, GeneratorResult] = field(default_factory=dict)
    sharpe_distributions: dict[str, list[float]] = field(default_factory=dict)
    elapsed_s: float = 0.0
    winner_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol.to_dict(),
            "rankings": [r.to_dict() for r in self.rankings],
            "sharpe_distributions": {k: sorted(v) for k, v in self.sharpe_distributions.items()},
            "elapsed_s": round(self.elapsed_s, 3),
            "winner_name": self.winner_name,
        }


# --------------------------------------------------------------------------- #
# Comparator
# --------------------------------------------------------------------------- #


class GeneratorComparator:
    """Compare les générateurs sous un protocole reproductible.

    Le comparator peut :
      1. Appliquer chaque générateur (génération d'hypothèses).
      2. Évaluer chaque hypothèse via le moteur + admission.
      3. Classer les générateurs.
      4. Retourner le rapport + le nom du gagnant.

    L'admission peut être :
      - `None` : on note juste les métriques brutes (rapide).
      - Un callable `(hypothesis, calibrated, mesures_val) -> bool` : on filtre.
    """

    def __init__(
        self,
        generators: list[BaseGenerator],
        protocol: GenerationProtocol,
        engine: EvaluationEngine,
        config: EinherjarConfig,
    ) -> None:
        self.generators = generators
        self.protocol = protocol
        self.engine = engine
        self.config = config
        logger.info(
            "GeneratorComparator instancié : %d générateurs, protocol=%s",
            len(generators), protocol.to_dict(),
        )

    def run(
        self,
        train_ohlcv: Any,
        train_features: Any,
        val_ohlcv: Any,
        val_features: Any,
        *,
        admission_fn: Callable | None = None,
    ) -> ComparisonReport:
        """Compare les générateurs sous le protocole.

        Args:
            train_ohlcv/train_features: split train.
            val_ohlcv/val_features: split val.
            admission_fn: Filtre optionnel (cf. docstring).

        Returns:
            ComparisonReport avec rankings + winner.
        """
        t0 = time.time()
        report = ComparisonReport(protocol=self.protocol)
        for gen in self.generators:
            logger.info("=" * 60)
            logger.info("Générateur : %s", gen.name)
            logger.info("=" * 60)
            t_gen = time.time()
            result = gen.generate()
            n_gen = result.n_generated
            n_eval = 0
            n_adm = 0
            sharpes: list[float] = []
            sharpes_all: list[float] = []
            for hyp in result.hypotheses:
                try:
                    calibrated = self.engine.train_calibrate(hyp, train_ohlcv, train_features)
                    mesures = self.engine.test_on(
                        hyp, val_ohlcv, val_features, calibrated, "val",
                    )
                    n_eval += 1
                    if mesures.sharpe_net == mesures.sharpe_net:  # not NaN
                        sharpes_all.append(mesures.sharpe_net)
                    if admission_fn is None or admission_fn(hyp, calibrated, mesures):
                        n_adm += 1
                        if mesures.sharpe_net == mesures.sharpe_net:
                            sharpes.append(mesures.sharpe_net)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Échec eval %s sur %s : %s", gen.name, hyp.id, exc)
            elapsed = time.time() - t_gen
            # Mise à jour des résultats avec les vrais compteurs.
            raw = GeneratorResult(
                generator_name=result.generator_name,
                hypotheses=result.hypotheses,
                n_generated=n_gen,
                n_evaluated=n_eval,
                n_passed_admission=n_adm,
                generation_time_s=elapsed,
                meta=result.meta,
            )
            report.raw_results[gen.name] = raw
            report.sharpe_distributions[gen.name] = sharpes_all
            # Score composite simple (à raffiner en V2) : admission_rate × median_sharpe_admis.
            adm_rate = n_adm / max(n_eval, 1)
            med_sharpe = _median(sharpes)
            med_sharpe_all = _median(sharpes_all)
            score = adm_rate * (med_sharpe if med_sharpe > 0 else 0.0)
            ranking = GeneratorRanking(
                generator_name=gen.name,
                rank=0,  # sera fixé après le tri
                score=score,
                n_generated=n_gen,
                n_evaluated=n_eval,
                n_passed_admission=n_adm,
                admission_rate=adm_rate,
                median_sharpe=med_sharpe,
                median_sharpe_all=med_sharpe_all,
                elapsed_s=elapsed,
            )
            report.rankings.append(ranking)
        # Tri par score décroissant, fix des rangs.
        report.rankings.sort(key=lambda r: r.score, reverse=True)
        for i, r in enumerate(report.rankings):
            # Rangs immutables — on rebuild le dataclass.
            report.rankings[i] = GeneratorRanking(
                generator_name=r.generator_name,
                rank=i + 1,
                score=r.score,
                n_generated=r.n_generated,
                n_evaluated=r.n_evaluated,
                n_passed_admission=r.n_passed_admission,
                admission_rate=r.admission_rate,
                median_sharpe=r.median_sharpe,
                median_sharpe_all=r.median_sharpe_all,
                elapsed_s=r.elapsed_s,
            )
        report.winner_name = report.rankings[0].generator_name if report.rankings else None
        report.elapsed_s = time.time() - t0
        logger.info(
            "Comparaison terminée : winner=%s, %d générateurs, %.2fs",
            report.winner_name, len(self.generators), report.elapsed_s,
        )
        return report


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _median(values: list[float]) -> float:
    """Médiane (tri + index central)."""
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    return s[n // 2] if n % 2 == 1 else (s[n // 2 - 1] + s[n // 2]) / 2
