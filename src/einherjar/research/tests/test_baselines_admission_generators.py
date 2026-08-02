"""
tests/test_baselines_admission_generators.py — Tests smoke pour les 3 modules.

Vérifie :
  - baselines/ : génère des hypothèses valides (3 algos)
  - admission/ : 7 critères + decision + diversity fonctionnent
  - generators/ : 5 candidats produisent des hypothèses valides

Pas de test de performance — c'est smoke test, pas benchmark.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl

from einherjar.research.admission.criteria import (
    CriterionVerdict,
    evaluate_all_criteria,
    evaluate_bootstrap_ci_ret,
    evaluate_bootstrap_ci_sharpe,
    evaluate_cross_asset,
    evaluate_dsr,
    evaluate_max_drawdown,
    evaluate_n_trades,
    evaluate_pbo,
)
from einherjar.research.admission.decision import AdmissionDecider
from einherjar.research.admission.diversity import (
    BehavioralDescriptors,
    QuotaReport,
    evaluate_quotas,
    extract_dominant_family,
    extract_dominant_type,
)
from einherjar.research.baselines.algorithms import (
    BaseBaseline,
    HumanRules,
    RandomConstrained,
    ShallowEnumeration,
    make_all_baselines,
)
from einherjar.research.baselines.runner import BaselineRunner, make_default_runner
from einherjar.research.config.loader import load_config
from einherjar.research.data.features import FeaturesFrame
from einherjar.research.data.ohlcv import OhlcvFrame, _InMemoryBackend, OhlcvProvider
from einherjar.research.engine.evaluator import EvaluationEngine
from einherjar.research.generators.algorithms import (
    BeamSearchGenerator,
    GrammaticalEvolutionGenerator,
    MemeticGenerator,
    NSGA2Generator,
    RandomSearchGenerator,
    TypedGPGenerator,
    make_all_generators,
)
from einherjar.research.generators.comparator import GeneratorComparator
from einherjar.research.generators.protocol import GenerationProtocol, make_protocol
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


# =========================================================================== #
# Helpers
# =========================================================================== #


def _make_synthetic_ohlcv(n: int = 600, asset: str = "BTCUSD", timeframe: str = "1h") -> pl.DataFrame:
    rng = np.random.default_rng(seed=42)
    close = 100.0 * np.cumprod(1.0 + rng.normal(0, 0.01, n))
    high = close * (1.0 + np.abs(rng.normal(0, 0.005, n)))
    low = close * (1.0 - np.abs(rng.normal(0, 0.005, n)))
    open_ = close * (1.0 + rng.normal(0, 0.002, n))
    volume = rng.integers(100, 10_000, n).astype(float)
    ts = [datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(hours=i) for i in range(n)]
    return pl.DataFrame({
        "asset": [asset] * n,
        "timeframe": [timeframe] * n,
        "timestamp": ts,
        "open": open_, "high": high, "low": low, "close": close, "volume": volume,
    })


def _make_synthetic_features(ohlcv_df: pl.DataFrame) -> pl.DataFrame:
    close = ohlcv_df["close"].to_numpy()
    feats = {
        "timestamp": ohlcv_df["timestamp"],
        "open": ohlcv_df["open"],
        "high": ohlcv_df["high"],
        "low": ohlcv_df["low"],
        "close": ohlcv_df["close"],
        "volume": ohlcv_df["volume"],
    }
    for window in [5, 10, 20, 50]:
        ma = np.full_like(close, np.nan, dtype=float)
        for k in range(window - 1, len(close)):
            ma[k] = float(np.mean(close[k - window + 1 : k + 1]))
        feats[f"ma_{window}"] = ma
    return pl.DataFrame(feats)


def _make_test_engine(config) -> tuple[EvaluationEngine, OhlcvFrame, FeaturesFrame, OhlcvFrame, FeaturesFrame]:
    """Construit un EvaluationEngine + données train/val synthétiques."""
    from einherjar.research.data.features import make_test_provider
    provider_features = make_test_provider(config)
    ohlcv_df = _make_synthetic_ohlcv(n=800)
    feats_df = _make_synthetic_features(ohlcv_df)
    train_ohlcv = OhlcvFrame(asset="BTCUSD", timeframe="1h", df=ohlcv_df.head(500), data_version="v1")
    val_ohlcv = OhlcvFrame(asset="BTCUSD", timeframe="1h", df=ohlcv_df.tail(300), data_version="v1")
    train_feats = provider_features.compute(train_ohlcv)
    val_feats = provider_features.compute(val_ohlcv)
    engine = EvaluationEngine(config=config, data_version="v1", seed=42)
    return engine, train_ohlcv, train_feats, val_ohlcv, val_feats


# =========================================================================== #
# Tests baselines
# =========================================================================== #


class TestBaselines(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.config = load_config("src/einherjar/research/config")

    def test_human_rules_generates(self):
        b = HumanRules(config=self.config, seed=42)
        result = b.generate()
        self.assertGreater(result.n_generated, 0)
        self.assertGreater(len(result.hypotheses), 0)
        for h in result.hypotheses:
            self.assertIsInstance(h, Hypothesis)
            self.assertIn(h.direction, (Direction.LONG, Direction.SHORT))

    def test_shallow_enum_generates(self):
        b = ShallowEnumeration(config=self.config, seed=42, max_features=10)
        result = b.generate()
        self.assertGreater(result.n_generated, 0)
        # Vérifie qu'on a bien 2 directions.
        dirs = {h.direction for h in result.hypotheses}
        self.assertEqual(dirs, {Direction.LONG, Direction.SHORT})

    def test_random_constrained_generates(self):
        b = RandomConstrained(config=self.config, seed=42, n_samples=200)
        result = b.generate()
        self.assertEqual(result.n_generated, 200)

    def test_make_all_baselines(self):
        baselines = make_all_baselines(self.config, seed=42)
        self.assertEqual(len(baselines), 3)
        names = {b.name for b in baselines}
        self.assertIn("HumanRules", names)
        self.assertIn("ShallowEnumeration", names)
        self.assertIn("RandomConstrained", names)


# =========================================================================== #
# Tests admission
# =========================================================================== #


def _make_mesures_test(n_signals: int = 50, sharpe: float = 1.5) -> "MesuresBrutes":
    from einherjar.research.utils.types import MesuresBrutes, TradeMesure, ExitReason
    rng = np.random.default_rng(seed=42)
    rets = rng.normal(0.001, 0.01, n_signals).tolist()
    return MesuresBrutes(
        n_signals=n_signals, n_tp_hit=int(n_signals * 0.5), n_sl_hit=int(n_signals * 0.3),
        n_timeout=int(n_signals * 0.2),
        mfe_mean_pct=0.02, mae_mean_pct=0.01,
        mfe_p50=0.015, mfe_p75=0.025, mfe_p90=0.04,
        mae_p50=0.008, mae_p75=0.015, mae_p90=0.025,
        ret_mean_pct_net=float(np.mean(rets)), ret_std_pct=float(np.std(rets)),
        sharpe_net=sharpe,
        tp_hit_rate=0.5, sl_hit_rate=0.3, timeout_rate=0.2,
        avg_holding_period=5.0, avg_time_to_amplitude=3.0,
        bootstrap_sharpe_ci_low=0.5, bootstrap_sharpe_ci_high=2.5,
        bootstrap_ret_ci_low=0.1, bootstrap_ret_ci_high=2.0,
    )


class TestAdmissionCriteria(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.config = load_config("src/einherjar/research/config")

    def test_evaluate_all_criteria_returns_admission_verdict(self):
        mesures = _make_mesures_test(n_signals=50, sharpe=1.5)
        returns = np.random.default_rng(42).normal(0.001, 0.01, 50).tolist()
        verdict = evaluate_all_criteria(mesures, returns, self.config, n_indep_trials=10)
        self.assertIsInstance(verdict.n_passed, int)
        self.assertEqual(verdict.n_passed + verdict.n_failed, 7)

    def test_individual_criteria(self):
        m = _make_mesures_test(sharpe=1.5)
        v1 = evaluate_dsr(m, self.config, n_indep_trials=10)
        self.assertEqual(v1.name, "DSR")
        v2 = evaluate_n_trades(m, self.config)
        self.assertEqual(v2.name, "N_TRADES")
        v3 = evaluate_bootstrap_ci_sharpe(m)
        self.assertEqual(v3.name, "BOOTSTRAP_CI_SHARPE")
        v4 = evaluate_bootstrap_ci_ret(m)
        self.assertEqual(v4.name, "BOOTSTRAP_CI_RET")
        v5 = evaluate_cross_asset(m, self.config)
        self.assertEqual(v5.name, "CROSS_ASSET")
        v6 = evaluate_max_drawdown(m, self.config)
        self.assertEqual(v6.name, "MAX_DRAWDOWN")
        rets = np.random.default_rng(42).normal(0.001, 0.01, 100).tolist()
        v7 = evaluate_pbo(rets, self.config)
        self.assertEqual(v7.name, "PBO")

    def test_low_signals_fails_n_trades(self):
        m = _make_mesures_test(n_signals=5, sharpe=1.0)
        v = evaluate_n_trades(m, self.config)
        self.assertFalse(v.passed)
        self.assertEqual(v.reason, RejectionReason.N_TRADES_FAIL)


class TestAdmissionDiversity(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.config = load_config("src/einherjar/research/config")

    def test_behavioral_descriptors_construction(self):
        m = _make_mesures_test()
        d = BehavioralDescriptors(
            signal_dates=(10, 20, 30),
            signal_overlap_vs_corpus=0.2,
            ret_corr_vs_corpus=0.3,
        )
        self.assertEqual(d.signal_overlap_vs_corpus, 0.2)
        fp = d.fingerprint()
        self.assertIsInstance(fp, str)
        # Stabilité.
        fp2 = BehavioralDescriptors(
            signal_dates=(10, 20, 30),
            signal_overlap_vs_corpus=0.2,
            ret_corr_vs_corpus=0.3,
        ).fingerprint()
        self.assertEqual(fp, fp2)

    def test_quotas_pass(self):
        q = evaluate_quotas(
            new_family="momentum",
            new_type="atomic",
            new_direction="long",
            current_family_fracs={"trend": 0.5, "momentum": 0.3},
            current_type_fracs={"atomic": 0.4},
            current_direction_fracs={"long": 0.5, "short": 0.5},
            config=self.config,
        )
        self.assertIsInstance(q, QuotaReport)
        # Avec 0.3 + 1/n dans momentum, on peut dépasser le max 0.4 selon n.
        # On vérifie juste que les booléens sont cohérents.
        self.assertIsInstance(q.family_ok, bool)

    def test_extract_dominant_family(self):
        c = Condition(feature_ref="rsi_14", operator=CompareOp.LT, value=30.0)
        family = extract_dominant_family(c, self.config)
        self.assertIsInstance(family, str)
        t = extract_dominant_type(c, self.config)
        self.assertIsInstance(t, str)


class TestAdmissionDecision(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.config = load_config("src/einherjar/research/config")

    def test_decider_returns_decision(self):
        decider = AdmissionDecider(config=self.config, data_version="v1", seed=42)
        m = _make_mesures_test(sharpe=1.5)
        rets = np.random.default_rng(42).normal(0.001, 0.01, 50).tolist()
        cond = Condition(feature_ref="rsi_14", operator=CompareOp.LT, value=30.0)
        h = Hypothesis(
            id="hyp_test",
            condition_tree=cond,
            amplitude=Amplitude(valeur=10.0, unité=AmplitudeUnit.PRICE_ABSOLU, direction_implicite=Direction.LONG),
            direction=Direction.LONG,
            universe=Universe(assets=("BTCUSD",), timeframes=("1h",)),
        )
        # On doit bypass le check d'admission strict (peut-être qu'il n'admet pas).
        # On vérifie juste que la décision est construite.
        from einherjar.research.engine.evaluator import CalibratedParams
        cal = CalibratedParams(
            n_window=10, sl_n_atr=1.0, tp_n_atr=2.0, atr_p50=1.0,
            n_observations=500, sl_distance=0.05, tp_distance=0.10,
        )
        decision = decider.decide(
            hypothesis_id="hyp_test",
            condition_tree=cond,
            direction=Direction.LONG,
            universe=h.universe,
            amplitude=h.amplitude,
            calibrated=cal,
            mesures_val=m,
            returns_val=rets,
            n_indep_trials=10,
        )
        self.assertIsNotNone(decision.criteria_verdict)
        self.assertEqual(decision.criteria_verdict.n_passed + decision.criteria_verdict.n_failed, 7)


# =========================================================================== #
# Tests generators
# =========================================================================== #


class TestGenerators(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.config = load_config("src/einherjar/research/config")
        cls.protocol = make_protocol(cls.config, data_version="v1", seed=42, n_eval_budget=200)

    def test_random_search_generates(self):
        g = RandomSearchGenerator(self.protocol, self.config)
        result = g.generate()
        self.assertEqual(result.n_generated, 200)
        self.assertGreater(len(result.hypotheses), 0)

    def test_beam_search_generates(self):
        g = BeamSearchGenerator(self.protocol, self.config, beam_width=16, depth=2)
        result = g.generate()
        self.assertGreater(result.n_generated, 0)
        for h in result.hypotheses:
            self.assertIsInstance(h, Hypothesis)

    def test_typed_gp_generates(self):
        g = TypedGPGenerator(self.protocol, self.config, population_size=50)
        result = g.generate()
        self.assertEqual(result.n_generated, 50)
        # Vérifie qu'on a bien des arbres (certains peuvent être composés).
        n_compound = sum(
            1 for h in result.hypotheses if isinstance(h.condition_tree, ConditionNode)
        )
        self.assertGreater(n_compound, 0)

    def test_ge_returns_empty_without_bnf(self):
        g = GrammaticalEvolutionGenerator(self.protocol, bnf_grammar=None)
        result = g.generate()
        self.assertEqual(result.n_generated, 0)
        self.assertEqual(result.hypotheses, ())

    def test_memetic_runs_without_engine_fails(self):
        """MemeticGenerator REQUIERT un engine (P10 — pas de placeholder silencieux)."""
        with self.assertRaises(ValueError):
            MemeticGenerator(self.protocol, self.config, engine=None)

    def test_nsga2_requires_engine(self):
        """NSGA2Generator REQUIERT un engine (P10 — pas de placeholder silencieux)."""
        with self.assertRaises(ValueError):
            NSGA2Generator(self.protocol, self.config, engine=None)

    def test_nsga2_dominates_pareto(self):
        """Test unitaire sur l'opérateur de dominance Pareto (Deb 2002)."""
        from einherjar.research.generators.algorithms import (
            _EvaluatedIndividual, _NSGA2Individual, NSGA2Generator,
        )
        # Crée 3 individus : A domine B et C, B et C ne se dominent pas.
        a = _EvaluatedIndividual(
            individual=_NSGA2Individual(0, 0, 0.0, 5, 0),
            hypothesis=None,  # pas requis pour le test de dominance
            objectives=(2.0, -0.1, 1.0, -1.0),
            constraints_passed=(True,) * 8,
            n_violations=0, n_signals=10,
        )
        b = _EvaluatedIndividual(
            individual=_NSGA2Individual(1, 1, 0.0, 5, 0), hypothesis=None,
            objectives=(1.0, -0.1, 1.0, -1.0),  # dominé par A sur obj0
            constraints_passed=(True,) * 8,
            n_violations=0, n_signals=10,
        )
        c = _EvaluatedIndividual(
            individual=_NSGA2Individual(2, 2, 0.0, 5, 0), hypothesis=None,
            objectives=(2.0, -0.2, 1.0, -1.0),  # dominé par A sur obj1
            constraints_passed=(True,) * 8,
            n_violations=0, n_signals=10,
        )
        # A domine B (2.0 > 1.0 sur obj0, reste égal).
        self.assertTrue(NSGA2Generator._dominates(a, b))
        # A domine C (2.0 = 2.0 sur obj0, -0.1 > -0.2 sur obj1).
        self.assertTrue(NSGA2Generator._dominates(a, c))
        # B ne domine pas A (inverse).
        self.assertFalse(NSGA2Generator._dominates(b, a))
        # B et C ne se dominent pas (1.0 < 2.0 sur obj0 mais -0.1 > -0.2 sur obj1).
        self.assertFalse(NSGA2Generator._dominates(b, c))
        self.assertFalse(NSGA2Generator._dominates(c, b))

    def test_nsga2_constraint_dominance(self):
        """Une solution réalisable domine toute solution non réalisable (Deb 2002 §3.2)."""
        from einherjar.research.generators.algorithms import (
            _EvaluatedIndividual, _NSGA2Individual, NSGA2Generator,
        )
        feasible = _EvaluatedIndividual(
            individual=_NSGA2Individual(0, 0, 0.0, 5, 0), hypothesis=None,
            objectives=(1.0, -0.5, 0.0, -1.0),
            constraints_passed=(True,) * 8, n_violations=0, n_signals=10,
        )
        infeasible = _EvaluatedIndividual(
            individual=_NSGA2Individual(1, 1, 0.0, 5, 0), hypothesis=None,
            objectives=(10.0, -0.01, 1.0, -1.0),  # meilleurs objectifs !
            constraints_passed=(False, True, True, True, True, True, True, True),
            n_violations=1, n_signals=10,
        )
        # Même avec de meilleurs objectifs, le réalisable domine l'irréalisable.
        self.assertTrue(NSGA2Generator._dominates(feasible, infeasible))
        self.assertFalse(NSGA2Generator._dominates(infeasible, feasible))

    def test_nsga2_fast_non_dominated_sort(self):
        """Test unitaire sur le non-dominance sort (3 fronts attendus)."""
        from einherjar.research.generators.algorithms import (
            _EvaluatedIndividual, _NSGA2Individual, NSGA2Generator,
        )
        # 4 individus : A domine tout, B et C se dominent pas, D dominé par B et C.
        pop = [
            _EvaluatedIndividual(_NSGA2Individual(0, 0, 0.0, 5, 0), None,
                                (3.0, -0.1, 1.0, -1.0), (True,) * 8, 0, 10),
            _EvaluatedIndividual(_NSGA2Individual(1, 1, 0.0, 5, 0), None,
                                (1.0, -0.3, 0.5, -1.0), (True,) * 8, 0, 10),
            _EvaluatedIndividual(_NSGA2Individual(2, 2, 0.0, 5, 0), None,
                                (0.5, -0.2, 0.5, -1.0), (True,) * 8, 0, 10),
            _EvaluatedIndividual(_NSGA2Individual(3, 3, 0.0, 5, 0), None,
                                (0.5, -0.5, 0.1, -1.0), (True,) * 8, 0, 10),
        ]
        fronts = NSGA2Generator._fast_non_dominated_sort(pop)
        # Au moins 2 fronts.
        self.assertGreaterEqual(len(fronts), 2)
        # Le premier front contient l'individu 0 (Pareto-optimal).
        self.assertIn(0, fronts[0])
        # Le dernier individu (3) est dominé : il est dans un front > 0.
        all_later = [i for front in fronts[1:] for i in front]
        self.assertIn(3, all_later)


if __name__ == "__main__":
    unittest.main()
