"""Tests pour le module de pilotage."""

import unittest
from datetime import datetime

from einherjar.research.generators.algorithms import GeneratorResult
from einherjar.research.generators.comparator import GeneratorRanking
from einherjar.research.pilotage import (
    DiversityStats,
    EngineStats,
    PilotageReport,
    RejectionBreakdown,
    build_pilotage_report,
    compute_diversity,
)
from einherjar.research.utils.types import (
    Amplitude,
    AmplitudeUnit,
    CompareOp,
    Condition,
    ConditionNode,
    Direction,
    Hypothesis,
    LogicalOp,
    RejectionReason,
    Universe,
)


def _make_hyp(
    id: str,
    feature: str = "rsi_14",
    operator: CompareOp = CompareOp.GT,
    value: float = 0.0,
    transformation: str | None = "quantile(50)",
    direction: Direction = Direction.LONG,
    cond_tree: Condition | ConditionNode | None = None,
    meta: dict | None = None,
) -> Hypothesis:
    """Helper pour creer une Hypothesis de test."""
    if cond_tree is None:
        cond_tree = Condition(
            feature_ref=feature, operator=operator,
            value=value, transformation=transformation,
        )
    return Hypothesis(
        id=id,
        condition_tree=cond_tree,
        amplitude=Amplitude(
            valeur=0.02, unité=AmplitudeUnit.PRICE_ABSOLU,
            direction_implicite=direction,
        ),
        direction=direction,
        universe=Universe(assets=("BTCUSD",), timeframes=("1h",)),
        cooldown_k=5,
        meta=meta or {},
    )


def _make_result(name: str, hyps: list[Hypothesis], elapsed: float = 1.0) -> GeneratorResult:
    """Helper pour creer un GeneratorResult."""
    n_adm = max(1, len(hyps) // 10)
    return GeneratorResult(
        generator_name=name,
        hypotheses=tuple(hyps),
        n_generated=len(hyps),
        n_evaluated=len(hyps),
        n_passed_admission=n_adm,
        generation_time_s=elapsed,
        meta={},
    )


def _make_ranking(
    name: str, rank: int, score: float,
    adm_rate: float = 0.1, elapsed: float = 1.0,
    sharpe: float = 1.0, n_features: int = 10,
    coherence: float = 0.5,
) -> GeneratorRanking:
    """Helper pour creer un GeneratorRanking."""
    return GeneratorRanking(
        generator_name=name,
        rank=rank, score=score,
        n_generated=100, n_evaluated=100, n_passed_admission=10,
        admission_rate=adm_rate,
        median_sharpe=sharpe, median_sharpe_all=0.5,
        n_distinct_features=n_features,
        semantic_coherence=coherence,
        elapsed_s=elapsed,
        subscores={
            "sharpe": sharpe, "admission_rate": adm_rate,
            "diversity": float(n_features), "coherence": coherence,
            "norm_sharpe": 0.5, "norm_admission_rate": 0.5,
            "norm_diversity": 0.5, "norm_coherence": 0.5,
            "composite": score,
        },
    )


class TestComputeDiversity(unittest.TestCase):
    """Tests de compute_diversity()."""

    def test_empty(self) -> None:
        """Liste vide -> toutes les stats a 0."""
        d = compute_diversity([])
        self.assertEqual(d.n_hypotheses, 0)
        self.assertEqual(d.n_features_distinct, 0)
        self.assertEqual(d.n_patterns_distinct, 0)
        self.assertEqual(d.n_atomic_conditions, 0)
        self.assertEqual(d.n_compound_conditions, 0)
        self.assertEqual(d.ratio_compound, 0.0)

    def test_atomic_only(self) -> None:
        """3 hypotheses atomiques, 2 features distinctes."""
        hyps = [
            _make_hyp("h1", feature="rsi_14"),
            _make_hyp("h2", feature="rsi_21"),
            _make_hyp("h3", feature="rsi_14"),
        ]
        d = compute_diversity(hyps)
        self.assertEqual(d.n_hypotheses, 3)
        self.assertEqual(d.n_features_distinct, 2)
        self.assertEqual(d.n_atomic_conditions, 3)
        self.assertEqual(d.n_compound_conditions, 0)
        self.assertEqual(d.ratio_compound, 0.0)

    def test_compound(self) -> None:
        """1 hypothese composee + 1 atomique."""
        compound = ConditionNode(
            op=LogicalOp.AND,
            left=Condition(feature_ref="open", operator=CompareOp.GT, value=0.0, transformation="quantile(50)"),
            right=Condition(feature_ref="close", operator=CompareOp.LT, value=0.0, transformation="quantile(50)"),
        )
        hyps = [
            _make_hyp("h1", cond_tree=compound),
            _make_hyp("h2", feature="rsi_14"),
        ]
        d = compute_diversity(hyps)
        self.assertEqual(d.n_hypotheses, 2)
        self.assertEqual(d.n_features_distinct, 3)
        self.assertEqual(d.n_compound_conditions, 1)
        self.assertEqual(d.n_atomic_conditions, 1)
        self.assertEqual(d.ratio_compound, 0.5)

    def test_patterns_counted(self) -> None:
        """Les patterns sont comptes separement des autres features."""
        hyps = [
            _make_hyp("h1", feature="pattern_hammer"),
            _make_hyp("h2", feature="pattern_shooting_star"),
            _make_hyp("h3", feature="rsi_14"),
        ]
        d = compute_diversity(hyps)
        self.assertEqual(d.n_features_distinct, 3)
        self.assertEqual(d.n_patterns_distinct, 2)

    def test_quantile_vs_discrete(self) -> None:
        """Compte les seuils quantile vs discrets vs featureref."""
        hyps = [
            _make_hyp("h1", feature="rsi_14", transformation="quantile(50)"),
            _make_hyp("h2", feature="rsi_14", transformation="quantile(75)"),
            _make_hyp("h3", feature="pattern_doji", transformation=None, value=1),
            _make_hyp("h4", feature="close", transformation="featureref:open"),
        ]
        d = compute_diversity(hyps)
        self.assertEqual(d.n_quantile_thresholds, 2)
        self.assertEqual(d.n_discrete_thresholds, 1)
        self.assertEqual(d.n_featureref_thresholds, 1)


class TestRejectionBreakdown(unittest.TestCase):
    """Tests de build_rejection_breakdown()."""

    def test_none(self) -> None:
        """None -> None."""
        from einherjar.research.pilotage import build_rejection_breakdown
        self.assertIsNone(build_rejection_breakdown(None))

    def test_empty(self) -> None:
        """Liste vide -> breakdown vide."""
        from einherjar.research.pilotage import build_rejection_breakdown
        b = build_rejection_breakdown([])
        self.assertEqual(b.total, 0)
        self.assertEqual(b.counts, {})

    def test_distribution(self) -> None:
        """Distribution correcte par raison."""
        from einherjar.research.pilotage import build_rejection_breakdown
        log = [
            RejectionReason.DSR_FAIL,
            RejectionReason.DSR_FAIL,
            RejectionReason.PBO_FAIL,
            RejectionReason.DD_FAIL,
        ]
        b = build_rejection_breakdown(log)
        self.assertEqual(b.total, 4)
        self.assertEqual(b.counts["DSR_FAIL"], 2)
        self.assertEqual(b.counts["PBO_FAIL"], 1)
        self.assertEqual(b.counts["DD_FAIL"], 1)


class TestBuildPilotageReport(unittest.TestCase):
    """Tests de build_pilotage_report()."""

    def test_empty_results(self) -> None:
        """Aucun resultat -> rapport avec tous les totaux a 0."""
        report = build_pilotage_report([])
        self.assertEqual(report.total_generated, 0)
        self.assertEqual(report.total_evaluated, 0)
        self.assertEqual(report.total_admitted, 0)
        self.assertEqual(report.total_elapsed_s, 0.0)
        self.assertEqual(report.mean_admission_rate, 0.0)
        self.assertIsNone(report.winner_name)
        self.assertIsNone(report.rejection_breakdown)
        self.assertEqual(report.engine_stats, {})

    def test_single_result_no_ranking(self) -> None:
        """1 resultat sans ranking -> EngineStats construits mais rank=None."""
        hyps = [_make_hyp("h1"), _make_hyp("h2", feature="rsi_21")]
        results = [_make_result("Random", hyps, elapsed=2.5)]
        report = build_pilotage_report(results)
        self.assertEqual(len(report.engine_stats), 1)
        es = report.engine_stats["Random"]
        self.assertEqual(es.generator_name, "Random")
        self.assertEqual(es.n_generated, 2)
        self.assertEqual(es.elapsed_s, 2.5)
        self.assertIsNone(es.rank)
        self.assertIsNone(es.score)
        self.assertEqual(es.diversity.n_hypotheses, 2)
        self.assertEqual(es.diversity.n_features_distinct, 2)
        self.assertIsNone(report.winner_name)

    def test_with_rankings(self) -> None:
        """Avec rankings, les rangs et scores sont ajoutes."""
        hyps1 = [_make_hyp("h1", feature="rsi_14")]
        hyps2 = [_make_hyp("h2", feature="pattern_hammer", direction=Direction.LONG)]
        results = [
            _make_result("Gen1", hyps1, elapsed=1.0),
            _make_result("Gen2", hyps2, elapsed=2.0),
        ]
        rankings = [
            _make_ranking("Gen1", rank=2, score=0.5),
            _make_ranking("Gen2", rank=1, score=0.8, sharpe=1.2, coherence=1.0),
        ]
        report = build_pilotage_report(results, rankings=rankings)
        self.assertEqual(report.winner_name, "Gen2")
        self.assertEqual(report.engine_stats["Gen2"].rank, 1)
        self.assertEqual(report.engine_stats["Gen2"].score, 0.8)
        self.assertEqual(report.engine_stats["Gen2"].subscores["sharpe"], 1.2)
        self.assertEqual(report.engine_stats["Gen2"].diversity.n_patterns_distinct, 1)

    def test_totals_consistent(self) -> None:
        """Les totaux sont coherents avec la somme par moteur."""
        hyps = [_make_hyp(f"h{i}") for i in range(10)]
        results = [
            _make_result("Gen1", hyps[:5]),
            _make_result("Gen2", hyps[5:]),
        ]
        report = build_pilotage_report(results)
        self.assertEqual(report.total_generated, 10)
        # n_adm est 1 par resultat (max(1, 10//10) = 1), donc total = 2
        self.assertEqual(report.total_admitted, 2)
        # Mean admission rate = moyenne des 2 admission rates
        self.assertGreater(report.mean_admission_rate, 0.0)

    def test_with_rejection_log(self) -> None:
        """Avec rejection_log, la breakdown est incluse."""
        results = [_make_result("Gen1", [_make_hyp("h1")])]
        log = [RejectionReason.DSR_FAIL, RejectionReason.DSR_FAIL]
        report = build_pilotage_report(results, rejection_log=log)
        self.assertIsNotNone(report.rejection_breakdown)
        self.assertEqual(report.rejection_breakdown.total, 2)
        self.assertEqual(report.rejection_breakdown.counts["DSR_FAIL"], 2)


class TestPilotageReportSerialization(unittest.TestCase):
    """Tests de serialisation to_dict()."""

    def test_to_dict(self) -> None:
        """to_dict() est serialisable JSON (via json.dumps)."""
        import json
        results = [_make_result("Gen1", [_make_hyp("h1")])]
        rankings = [_make_ranking("Gen1", rank=1, score=0.7)]
        log = [RejectionReason.DSR_FAIL]
        report = build_pilotage_report(results, rankings=rankings, rejection_log=log)
        d = report.to_dict()
        # Doit etre serialisable sans erreur.
        json.dumps(d)
        # Verifier les cles principales
        self.assertIn("engine_stats", d)
        self.assertIn("total_generated", d)
        self.assertIn("winner_name", d)
        self.assertIn("rejection_breakdown", d)
        self.assertIn("timestamp", d)

    def test_to_dict_engine_stats(self) -> None:
        """to_dict() sur EngineStats contient les metriques detaillees."""
        es = EngineStats(
            generator_name="Gen1",
            n_generated=100, n_evaluated=90, n_passed_admission=10,
            admission_rate=0.111, elapsed_s=1.5,
            rank=1, score=0.7,
            subscores={"sharpe": 1.2},
            diversity=DiversityStats(n_hypotheses=100, n_features_distinct=15),
        )
        d = es.to_dict()
        self.assertEqual(d["generator_name"], "Gen1")
        self.assertEqual(d["rank"], 1)
        # La diversite est dans une sous-clef.
        self.assertEqual(d["diversity"]["n_features_distinct"], 15)


if __name__ == "__main__":
    unittest.main()
