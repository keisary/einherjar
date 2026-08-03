"""pilotage.py — Rapport de pilotage d'une run de decouverte.

⚠ Chantier pilotage (apres BNF Phase 1-4, comparateur multi-obj).
Ce module agrege les GeneratorResult et les GeneratorRanking en un
rapport structuré qui explique COMMENT et POURQUOI la run s'est
passee ainsi (volume, performance, diversite, admissions, rejets).

Le comparateur (comparator.py) classifie les generateurs (qui gagne ?).
Le pilotage les explique (combien, comment, ou, pourquoi).

Sortie :
  - PilotageReport : synthese globale + EngineStats par moteur
  - RejectionBreakdown : distribution des rejets par raison
  - to_dict() pour serialisation JSON (archivage, logging, dashboard)

Metriques exposees par moteur :
  - Volume     : generated, evaluated, passed_admission
  - Performance: elapsed_s
  - Diversite  : n_features_distinct, n_patterns_distinct, ratio
                 atomiques/composees, ratio quantiles/discrets
  - Ranking    : rank, score composite multi-obj, sub-scores
  - Meta       : meta du GeneratorResult (bnf_source, chromosome, etc.)

Conforme a ALGORITHME_RESEARCH.md § 10.4 (pilotage).
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from einherjar.research.generators.algorithms import GeneratorResult
from einherjar.research.generators.comparator import GeneratorRanking
from einherjar.research.utils.types import (
    Condition,
    ConditionNode,
    RejectionReason,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Dataclasses de sortie
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RejectionBreakdown:
    """Distribution des rejets par raison (S-3.6).

    Attributes:
        counts: dict {RejectionReason.value: count}.
        total: nombre total de rejets.
    """

    counts: dict[str, int] = field(default_factory=dict)
    total: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "counts": dict(self.counts),
            "total": self.total,
        }


@dataclass(frozen=True)
class DiversityStats:
    """Statistiques de diversite pour un generateur.

    Attributes:
        n_hypotheses: nombre total d'hypotheses generees.
        n_features_distinct: cardinalite des features distinctes utilisees.
        n_patterns_distinct: cardinalite des patterns distincts utilises.
        n_atomic_conditions: nombre de conditions atomiques (vs composees).
        n_compound_conditions: nombre de conditions composees.
        ratio_compound: n_compound / n_hypotheses (0..1).
        n_quantile_thresholds: nb de seuils de type "quantile" utilises.
        n_discrete_thresholds: nb de seuils de type "v_<feat>_N" utilises.
        n_featureref_thresholds: nb de comparaisons inter-features
            (e.g., close > open).
    """

    n_hypotheses: int = 0
    n_features_distinct: int = 0
    n_patterns_distinct: int = 0
    n_atomic_conditions: int = 0
    n_compound_conditions: int = 0
    ratio_compound: float = 0.0
    n_quantile_thresholds: int = 0
    n_discrete_thresholds: int = 0
    n_featureref_thresholds: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_hypotheses": self.n_hypotheses,
            "n_features_distinct": self.n_features_distinct,
            "n_patterns_distinct": self.n_patterns_distinct,
            "n_atomic_conditions": self.n_atomic_conditions,
            "n_compound_conditions": self.n_compound_conditions,
            "ratio_compound": round(self.ratio_compound, 4),
            "n_quantile_thresholds": self.n_quantile_thresholds,
            "n_discrete_thresholds": self.n_discrete_thresholds,
            "n_featureref_thresholds": self.n_featureref_thresholds,
        }


@dataclass(frozen=True)
class EngineStats:
    """Statistiques completes pour un generateur.

    Attributes:
        generator_name: nom du generateur.
        n_generated: nb d'hypotheses generees.
        n_evaluated: nb d'hypotheses effectivement evaluees.
        n_passed_admission: nb d'hypotheses admises.
        admission_rate: n_passed_admission / n_evaluated.
        elapsed_s: temps CPU en secondes.
        rank: rang dans le comparateur (1 = meilleur), None si non classe.
        score: score composite multi-obj, None si non classe.
        subscores: dict sharpe/admission/diversity/coherence bruts.
        diversity: stats de diversite.
    """

    generator_name: str
    n_generated: int
    n_evaluated: int
    n_passed_admission: int
    admission_rate: float
    elapsed_s: float
    rank: int | None = None
    score: float | None = None
    subscores: dict[str, float] = field(default_factory=dict)
    diversity: DiversityStats = field(default_factory=DiversityStats)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generator_name": self.generator_name,
            "n_generated": self.n_generated,
            "n_evaluated": self.n_evaluated,
            "n_passed_admission": self.n_passed_admission,
            "admission_rate": round(self.admission_rate, 4),
            "elapsed_s": round(self.elapsed_s, 3),
            "rank": self.rank,
            "score": round(self.score, 4) if self.score is not None else None,
            "subscores": {k: round(v, 4) for k, v in self.subscores.items()},
            "diversity": self.diversity.to_dict(),
        }


@dataclass(frozen=True)
class PilotageReport:
    """Rapport de pilotage global d'une run de decouverte.

    Attributes:
        engine_stats: stats par moteur (cle = nom du moteur).
        total_generated: somme des n_generated de tous les moteurs.
        total_evaluated: somme des n_evaluated.
        total_admitted: somme des n_passed_admission.
        mean_admission_rate: moyenne des admission_rate par moteur.
        total_elapsed_s: somme des elapsed_s.
        winner_name: nom du generateur gagnant (rank=1), None si pas de classement.
        rejection_breakdown: distribution des rejets, None si pas fourni.
        timestamp: ISO 8601 UTC du moment de la generation.
    """

    engine_stats: dict[str, EngineStats] = field(default_factory=dict)
    total_generated: int = 0
    total_evaluated: int = 0
    total_admitted: int = 0
    mean_admission_rate: float = 0.0
    total_elapsed_s: float = 0.0
    winner_name: str | None = None
    rejection_breakdown: RejectionBreakdown | None = None
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine_stats": {k: v.to_dict() for k, v in self.engine_stats.items()},
            "total_generated": self.total_generated,
            "total_evaluated": self.total_evaluated,
            "total_admitted": self.total_admitted,
            "mean_admission_rate": round(self.mean_admission_rate, 4),
            "total_elapsed_s": round(self.total_elapsed_s, 3),
            "winner_name": self.winner_name,
            "rejection_breakdown": (
                self.rejection_breakdown.to_dict()
                if self.rejection_breakdown is not None else None
            ),
            "timestamp": self.timestamp,
        }


# --------------------------------------------------------------------------- #
# Helpers de calcul
# --------------------------------------------------------------------------- #


def _walk_conditions(node: Any) -> Any:
    """Yield recursivement les conditions (atomiques) d'un arbre.

    Yields: Condition (jamais ConditionNode).
    """
    if isinstance(node, Condition):
        yield node
    elif isinstance(node, ConditionNode):
        yield from _walk_conditions(node.left)
        if node.right is not None:
            yield from _walk_conditions(node.right)


def compute_diversity(hypotheses: list[Any]) -> DiversityStats:
    """Calcule les stats de diversite pour une liste d'hypotheses.

    Args:
        hypotheses: liste d'Hypothesis (ou tout objet avec .condition_tree).

    Returns:
        DiversityStats agrege.
    """
    features: set[str] = set()
    patterns: set[str] = set()
    n_atomic = 0
    n_compound = 0
    n_quantile = 0
    n_discrete = 0
    n_featureref = 0
    for hyp in hypotheses:
        tree = hyp.condition_tree
        if isinstance(tree, ConditionNode):
            n_compound += 1
        for cond in _walk_conditions(tree):
            features.add(cond.feature_ref)
            if cond.feature_ref.startswith("pattern_"):
                patterns.add(cond.feature_ref)
            trans = cond.transformation
            if trans is None:
                pass
            elif trans.startswith("quantile("):
                n_quantile += 1
            elif trans.startswith("featureref:"):
                n_featureref += 1
            # v_<feat>_N : transformation is None (value porte l'info)
        n_atomic = len(hypotheses) - n_compound
    n_hyp = len(hypotheses)
    ratio_compound = n_compound / n_hyp if n_hyp > 0 else 0.0
    # n_discrete est approx : on regarde le pattern de value
    n_discrete = sum(
        1 for hyp in hypotheses
        for cond in _walk_conditions(hyp.condition_tree)
        if isinstance(cond.value, int) and not isinstance(cond.value, bool)
    )
    return DiversityStats(
        n_hypotheses=n_hyp,
        n_features_distinct=len(features),
        n_patterns_distinct=len(patterns),
        n_atomic_conditions=n_atomic,
        n_compound_conditions=n_compound,
        ratio_compound=ratio_compound,
        n_quantile_thresholds=n_quantile,
        n_discrete_thresholds=n_discrete,
        n_featureref_thresholds=n_featureref,
    )


def build_rejection_breakdown(
    rejection_log: list[RejectionReason] | None,
) -> RejectionBreakdown | None:
    """Construit une breakdown de rejets a partir d'un log."""
    if rejection_log is None:
        return None
    counts: Counter[str] = Counter()
    for r in rejection_log:
        if isinstance(r, RejectionReason):
            counts[r.value] += 1
        elif isinstance(r, str):
            counts[r] += 1
    return RejectionBreakdown(
        counts=dict(counts),
        total=len(rejection_log),
    )


# --------------------------------------------------------------------------- #
# Fonction principale
# --------------------------------------------------------------------------- #


def build_pilotage_report(
    results: list[GeneratorResult],
    rankings: list[GeneratorRanking] | None = None,
    rejection_log: list[RejectionReason] | None = None,
) -> PilotageReport:
    """Construit un rapport de pilotage a partir des resultats de run.

    Args:
        results: liste des GeneratorResult (un par generateur).
        rankings: liste des GeneratorRanking (sortie du comparateur),
            optionnel. Si fourni, les rangs/scores sont ajoutes.
        rejection_log: liste des raisons de rejet (RejectionReason ou str),
            optionnel. Si fourni, la breakdown est incluse.

    Returns:
        PilotageReport complet.
    """
    from datetime import datetime, timezone

    # Indexer les rankings par nom de generateur pour lookup rapide.
    ranking_by_name: dict[str, GeneratorRanking] = {}
    if rankings:
        for r in rankings:
            ranking_by_name[r.generator_name] = r

    engine_stats: dict[str, EngineStats] = {}
    total_generated = 0
    total_evaluated = 0
    total_admitted = 0
    total_elapsed = 0.0
    admission_rates: list[float] = []
    for res in results:
        n_eval = max(res.n_evaluated, 1)
        adm_rate = res.n_passed_admission / n_eval
        ranking = ranking_by_name.get(res.generator_name)
        diversity = compute_diversity(list(res.hypotheses))
        # Subsores : depuis ranking si dispo, sinon reconstruction minimale.
        if ranking is not None:
            sub = dict(ranking.subscores) if ranking.subscores else {}
            engine_stats[res.generator_name] = EngineStats(
                generator_name=res.generator_name,
                n_generated=res.n_generated,
                n_evaluated=res.n_evaluated,
                n_passed_admission=res.n_passed_admission,
                admission_rate=adm_rate,
                elapsed_s=ranking.elapsed_s,
                rank=ranking.rank,
                score=ranking.score,
                subscores=sub,
                diversity=diversity,
            )
            total_elapsed += ranking.elapsed_s
        else:
            engine_stats[res.generator_name] = EngineStats(
                generator_name=res.generator_name,
                n_generated=res.n_generated,
                n_evaluated=res.n_evaluated,
                n_passed_admission=res.n_passed_admission,
                admission_rate=adm_rate,
                elapsed_s=res.generation_time_s,
                rank=None,
                score=None,
                subscores={},
                diversity=diversity,
            )
            total_elapsed += res.generation_time_s
        total_generated += res.n_generated
        total_evaluated += res.n_evaluated
        total_admitted += res.n_passed_admission
        admission_rates.append(adm_rate)
    mean_adm = (
        sum(admission_rates) / len(admission_rates) if admission_rates else 0.0
    )
    # Winner = rank 1 dans les rankings.
    winner_name: str | None = None
    if rankings:
        sorted_rankings = sorted(rankings, key=lambda r: r.rank or 1_000_000)
        if sorted_rankings and sorted_rankings[0].rank == 1:
            winner_name = sorted_rankings[0].generator_name
    rejection = build_rejection_breakdown(rejection_log)
    timestamp = datetime.now(tz=timezone.utc).isoformat()
    logger.info(
        "Pilotage report : %d generateurs, %d candidats, %d admis, winner=%s",
        len(results), total_generated, total_admitted, winner_name,
    )
    return PilotageReport(
        engine_stats=engine_stats,
        total_generated=total_generated,
        total_evaluated=total_evaluated,
        total_admitted=total_admitted,
        mean_admission_rate=mean_adm,
        total_elapsed_s=total_elapsed,
        winner_name=winner_name,
        rejection_breakdown=rejection,
        timestamp=timestamp,
    )
