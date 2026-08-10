"""Tests des upgrades « générateurs d'arbres » (décision 2026-08-09).

Couvre :
  1. Étiquetage asset/timeframe propagé depuis les données du run (bug #4).
  2. Pool ouvert aux features PATTERN + opérateurs EQ/NE dans les arbres.
  3. Fitness alignée CROISSANCE (log(1+CAGR)) avec porte n_trades.
  4. 'select' consomme le rapport de 'compare' (pas de re-comparaison).
"""

import argparse
import math
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from einherjar.research.config.loader import load_config
from einherjar.research.generators.algorithms import (
    BaseGenerator,
    BeamSearchGenerator,
    NSGA2Generator,
    RandomSearchGenerator,
    TypedGPGenerator,
)
from einherjar.research.generators.protocol import make_protocol
from einherjar.research.utils.types import (
    Amplitude,
    AmplitudeUnit,
    CompareOp,
    Direction,
    MesuresBrutes,
    Universe,
)

CONFIG_PATH = Path("src/einherjar/research/config")


def _make_mesures(
    n_signals: int = 50,
    ret_mean: float = 0.001,
    held: float = 5.0,
) -> MesuresBrutes:
    return MesuresBrutes(
        n_signals=n_signals, n_tp_hit=int(n_signals * 0.5), n_sl_hit=int(n_signals * 0.3),
        n_timeout=int(n_signals * 0.2),
        mfe_mean_pct=0.02, mae_mean_pct=0.01,
        mfe_p50=0.015, mfe_p75=0.025, mfe_p90=0.04,
        mae_p50=0.008, mae_p75=0.015, mae_p90=0.025,
        ret_mean_pct_net=ret_mean, ret_std_pct=0.01, sharpe_net=1.5,
        tp_hit_rate=0.5, sl_hit_rate=0.3, timeout_rate=0.2,
        avg_holding_period=held, avg_time_to_amplitude=3.0,
        bootstrap_sharpe_ci_low=0.5, bootstrap_sharpe_ci_high=2.5,
        bootstrap_ret_ci_low=0.1, bootstrap_ret_ci_high=2.0,
    )


def _make_args(**over) -> argparse.Namespace:
    base = dict(
        data_asset="BTCUSD", data_timeframe="1h", seed=42,
        n_eval=1200, max_conditions=4, generators=None, generator=None,
    )
    base.update(over)
    return argparse.Namespace(**base)


class TestEtiquetageTF(unittest.TestCase):
    """Bug #4 : les artefacts portaient universe.timeframes=['1h'] quel que
    soit le TF du run. Le protocole doit porter l'étiquette du run réel."""

    @classmethod
    def setUpClass(cls):
        cls.config = load_config(CONFIG_PATH)

    def test_make_protocol_propage_asset_et_timeframe(self):
        proto = make_protocol(
            self.config, data_version="v1", seed=7,
            assets=("BTCUSD",), timeframes=("15m",), max_conditions=4,
        )
        self.assertEqual(proto.assets, ("BTCUSD",))
        self.assertEqual(proto.timeframes, ("15m",))
        dd = proto.to_dict()
        self.assertEqual(dd["timeframes"], ["15m"])
        self.assertEqual(dd["assets"], ["BTCUSD"])

    def test_make_protocol_max_conditions_par_defaut_4(self):
        proto = make_protocol(self.config, data_version="v1", seed=7)
        self.assertEqual(proto.max_conditions, 4)

    def test_typedgp_universe_porte_le_tf_du_protocol(self):
        proto = make_protocol(
            self.config, data_version="v1", seed=7,
            assets=("BTCUSD",), timeframes=("4h",), max_conditions=5,
        )
        g = TypedGPGenerator(proto, self.config, engine=MagicMock())
        universe = g._make_universe()
        self.assertEqual(universe.timeframes, ("4h",))
        self.assertEqual(universe.assets, ("BTCUSD",))


class TestPoolPatterns(unittest.TestCase):
    """P2-02 : les arbres tiraient uniquement les features continues
    (114/246). Les 107 patterns candlestick doivent être échantillonnables
    avec EQ/NE (u n pattern est un fait discret, pas un seuil)."""

    @classmethod
    def setUpClass(cls):
        cls.config = load_config(CONFIG_PATH)

    def setUp(self):
        self.g = TypedGPGenerator(
            make_protocol(self.config, data_version="v1", seed=42, n_eval_budget=2000),
            self.config, engine=MagicMock(),
        )

    def test_pool_patterns_non_vide(self):
        self.assertGreater(len(self.g._pattern_features), 0)

    def test_atom_peut_tirer_un_pattern_eq_ne(self):
        found_pattern = False
        for _ in range(500):
            c = self.g._atom()
            if c.feature_ref in self.g._pattern_features:
                found_pattern = True
                self.assertIn(c.operator, (CompareOp.EQ, CompareOp.NE))
                self.assertIn(c.value, (0.0, 1.0))
                break
        self.assertTrue(found_pattern, "Aucun tirage pattern sur 500 essais")


class TestFitnessCroissance(unittest.TestCase):
    """P1 : la fitness d'évolution doit être alignée sur l'admission
    (critère 7 CROISSANCE) : log(1+CAGR), porte n_trades dure."""

    @classmethod
    def setUpClass(cls):
        cls.config = load_config(CONFIG_PATH)

    def test_cagr_positif_avec_trades_suffisants(self):
        proto = make_protocol(self.config, data_version="v1", seed=1)
        g = TypedGPGenerator(proto, self.config, engine=MagicMock())
        ppy = 8760.0  # 1h
        m = _make_mesures(n_signals=50, ret_mean=0.0005, held=5.0)
        fit = g._growth_fitness(m, ppy)
        expected = (8760.0 / 5.0) * math.log1p(0.0005)
        self.assertAlmostEqual(fit, expected, places=6)
        self.assertGreater(fit, 0.0)

    def test_porte_ntrades_inf_pour_sous_frequence(self):
        proto = make_protocol(self.config, data_version="v1", seed=1)
        g = TypedGPGenerator(proto, self.config, engine=MagicMock())
        m = _make_mesures(n_signals=5, ret_mean=0.001, held=5.0)  # < 30
        self.assertEqual(g._growth_fitness(m, 8760.0), float("-inf"))

    def test_inf_sans_trades(self):
        proto = make_protocol(self.config, data_version="v1", seed=1)
        g = TypedGPGenerator(proto, self.config, engine=MagicMock())
        m = _make_mesures(n_signals=0, ret_mean=0.0, held=5.0)
        self.assertEqual(g._growth_fitness(m, 8760.0), float("-inf"))

    def test_inf_si_perte_totale(self):
        proto = make_protocol(self.config, data_version="v1", seed=1)
        g = TypedGPGenerator(proto, self.config, engine=MagicMock())
        m = _make_mesures(n_signals=50, ret_mean=-1.0, held=5.0)
        self.assertEqual(g._growth_fitness(m, 8760.0), float("-inf"))

    def test_fitness_helpers_disponibles_dans_beam_et_base(self):
        # La méthode vit dans BaseGenerator : toutes les classes en héritent.
        self.assertTrue(hasattr(BaseGenerator, "_growth_fitness"))
        self.assertTrue(hasattr(BeamSearchGenerator, "_growth_fitness"))
        self.assertTrue(hasattr(NSGA2Generator, "_growth_fitness"))

    def test_soft_penalty_garde_un_gradient(self):
        proto = make_protocol(self.config, data_version="v1", seed=1)
        g = TypedGPGenerator(proto, self.config, engine=MagicMock())
        m = _make_mesures(n_signals=10, ret_mean=0.001, held=5.0)
        soft = g._growth_fitness(m, 8760.0, soft=True)
        self.assertGreater(soft, float("-inf"))
        self.assertGreater(soft, 0.0)


class TestSelectConsommeRapport(unittest.TestCase):
    """P1 : handle_select doit consommer le rapport de handle_compare au
    lieu de relancer la comparaison (~540 évals économisées par TF)."""

    @classmethod
    def setUpClass(cls):
        cls.config = load_config(CONFIG_PATH)

    def test_load_compare_report_frais(self, tmp_path=None):
        import json as _json
        from einherjar.research.discovery import _load_compare_report

        # Écrit un rapport factice correspondant au run demandé.
        payload = {
            "meta": {"asset": "BTCUSD", "timeframe": "1h", "seed": 42,
                     "n_eval": 1200, "max_conditions": 4},
            "report": {"winner_name": "TypedGPGenerator"},
        }
        args = _make_args()
        path = _load_compare_report.__globals__["Path"]
        # On pointe le fichier du run : compare_report_BTCUSD_1h.json
        from einherjar.research.discovery import _compare_report_path
        p = _compare_report_path(args)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(_json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        try:
            report = _load_compare_report(args)
            self.assertIsNotNone(report)
            self.assertEqual(report["winner_name"], "TypedGPGenerator")
        finally:
            p.unlink(missing_ok=True)

    def test_load_compare_report_stale(self):
        import json as _json
        from einherjar.research.discovery import _compare_report_path, _load_compare_report

        payload = {
            "meta": {"asset": "BTCUSD", "timeframe": "1h", "seed": 42,
                     "n_eval": 300, "max_conditions": 4},  # n_eval != 1200
            "report": {"winner_name": "TypedGPGenerator"},
        }
        args = _make_args()
        p = _compare_report_path(args)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(_json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        try:
            self.assertIsNone(_load_compare_report(args))
        finally:
            p.unlink(missing_ok=True)

    def test_comparison_report_from_dict_roundtrip(self):
        from einherjar.research.generators.comparator import (
            ComparisonReport,
            GeneratorRanking,
        )
        from einherjar.research.generators.protocol import GenerationProtocol

        proto = make_protocol(self.config, data_version="v1", seed=1)
        ranking = GeneratorRanking(
            generator_name="TypedGPGenerator",
            rank=1, score=0.9, admission_rate=0.1, median_sharpe=1.2,
            median_sharpe_all=0.4, n_generated=500, n_evaluated=50,
            n_passed_admission=5, n_distinct_features=10,
            semantic_coherence=0.5, elapsed_s=1.2,
        )
        rep = ComparisonReport(
            protocol=proto,
            rankings=(ranking,),
            sharpe_distributions={},
            elapsed_s=12.3, winner_name="TypedGPGenerator",
            total_evaluations=1200, budget=1200,
        )
        d = rep.to_dict()
        rep2 = ComparisonReport.from_dict(d)
        self.assertEqual(rep2.winner_name, "TypedGPGenerator")
        self.assertEqual(rep2.rankings[0].generator_name, "TypedGPGenerator")
        self.assertEqual(rep2.protocol.timeframes, proto.timeframes)
        self.assertEqual(rep2.protocol.max_conditions, proto.max_conditions)


class TestCompetitionArbresSeuls(unittest.TestCase):
    """Décision 2026-08-09 : random et GE sortent de la compétition par
    défaut ; on peut les ré-inclure explicitement."""

    @classmethod
    def setUpClass(cls):
        cls.config = load_config(CONFIG_PATH)

    def test_filtre_par_defaut_exclut_random_et_ge(self):
        from einherjar.research.discovery import _filter_competition_generators
        proto = make_protocol(self.config, data_version="v1", seed=1)
        all_gens = [RandomSearchGenerator(proto, self.config),
                    BeamSearchGenerator(proto, self.config, engine=MagicMock())]
        args = _make_args()
        kept = _filter_competition_generators(all_gens, args)
        names = [type(g).__name__ for g in kept]
        self.assertNotIn("RandomSearchGenerator", names)
        self.assertEqual(names, ["BeamSearchGenerator"])

    def test_filtre_inclut_avec_generators_explicite(self):
        from einherjar.research.discovery import _filter_competition_generators
        proto = make_protocol(self.config, data_version="v1", seed=1)
        all_gens = [RandomSearchGenerator(proto, self.config),
                    BeamSearchGenerator(proto, self.config, engine=MagicMock())]
        args = _make_args(generators="RandomSearchGenerator")
        kept = _filter_competition_generators(all_gens, args)
        self.assertEqual([type(g).__name__ for g in kept],
                         ["RandomSearchGenerator"])


if __name__ == "__main__":
    unittest.main()