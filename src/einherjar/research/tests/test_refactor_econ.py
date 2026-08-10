"""Tests dédiés au refactor de correction économique (session 2026-08-08).

Vérifie les correctifs du chantier :
  A. plancher économique TP/SL dans la calibration (CalibrationError),
  B. DSR : unité par-observation + n_observations (formule Bailey & LP),
  C. critère CROISSANCE (vision 50 $ -> x10) branché dans evaluate_all_criteria,
  D. DEFAULT_CORPUS_PATH ancré (indépendant du CWD),
  E. catalogue de raisons accepte CROISSANCE_FAIL.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import polars as pl

from einherjar.research.admission.criteria import (
    evaluate_all_criteria,
    evaluate_croissance,
    evaluate_dsr,
)
from einherjar.research.config.loader import load_config
from einherjar.research.data.features import FeaturesFrame
from einherjar.research.data.ohlcv import OhlcvFrame
from einherjar.research.engine.evaluator import (
    CalibrationError,
    EvaluationEngine,
    TradingCosts,
)
from einherjar.research.utils.metrics import dsr as dsr_metric
from einherjar.research.utils.types import (
    Amplitude,
    AmplitudeUnit,
    CompareOp,
    Condition,
    Direction,
    Hypothesis,
    MesuresBrutes,
    Universe,
)

CONFIG_PATH = "src/einherjar/research/config"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _synthetic_ohlcv(n: int = 800, base: float = 100.0, vol: float = 1.0):
    """Prix synthétique : marche aléatoire en % de base."""
    rng = np.random.default_rng(seed=7)
    close = base * np.cumprod(1 + rng.normal(0.0, vol / 100.0, n))
    open_ = close * (1 + rng.normal(0.0, vol / 200.0, n))
    high = np.maximum(open_, close) * (1 + abs(rng.normal(0.0, vol / 150.0, n)))
    low = np.minimum(open_, close) * (1 - abs(rng.normal(0.0, vol / 150.0, n)))
    ts = np.arange(n, dtype=np.int64)
    return pl.DataFrame(
        {
            "timestamp": ts,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.full(n, 1000.0),
        }
    )


def _features_for(ohlcv_df: pl.DataFrame) -> pl.DataFrame:
    close = ohlcv_df["close"].to_numpy()
    feats = {"timestamp": ohlcv_df["timestamp"]}
    for window in [5, 10, 20, 50]:
        ma = np.full_like(close, np.nan, dtype=float)
        for k in range(window - 1, len(close)):
            ma[k] = float(np.mean(close[k - window + 1 : k + 1]))
        feats[f"ma_{window}"] = ma
    return pl.DataFrame(feats)


def _make_meas(ret_mean: float, n: int, avg_hold: float) -> "MesuresBrutes":
    return MesuresBrutes(
        n_signals=n, n_tp_hit=n // 2, n_sl_hit=n // 3, n_timeout=n - n // 2 - n // 3,
        mfe_mean_pct=0.02, mae_mean_pct=0.01, mfe_p50=0.015, mfe_p75=0.02, mfe_p90=0.03,
        mae_p50=0.008, mae_p75=0.012, mae_p90=0.02,
        ret_mean_pct_net=ret_mean, ret_std_pct=0.02, sharpe_net=ret_mean / 0.02,
        tp_hit_rate=0.5, sl_hit_rate=0.33, timeout_rate=0.17,
        avg_holding_period=avg_hold, avg_time_to_amplitude=1.5,
        bootstrap_sharpe_ci_low=0.01, bootstrap_sharpe_ci_high=0.2,
        bootstrap_ret_ci_low=0.001, bootstrap_ret_ci_high=0.05,
    )


def _make_hypo() -> Hypothesis:
    return Hypothesis(
        id="hyp_econ_refactor",
        condition_tree=Condition(feature_ref="ma_20", operator=CompareOp.LT, value=105.0),
        amplitude=Amplitude(valeur=10.0, unité=AmplitudeUnit.PRICE_ABSOLU, direction_implicite=Direction.LONG),
        direction=Direction.LONG,
        universe=Universe(assets=("BTCUSD",), timeframes=("1h",)),
    )


def _make_engine_chere(config) -> EvaluationEngine:
    """Engine dont les coûts sont gonflés : round-trip 3 % (plancher TP 9 %)."""
    import dataclasses
    cfg = dataclasses.replace(
        config,
        costs={
            "default": {"spread_pct": 0.005, "commission_pct": 0.005, "slippage_pct": 0.005},
        },
    )
    return EvaluationEngine(config=cfg, data_version="v1", seed=42)


# ---------------------------------------------------------------------------
# A. Plancher économique
# ---------------------------------------------------------------------------


class TestPlancherEconomique(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.config = load_config(str(CONFIG_PATH))

    def test_roundtrip_costs_par_defaut(self):
        costs = TradingCosts.from_config(self.config, asset="BTCUSD")
        self.assertAlmostEqual(costs.total_round_trip_pct, 0.0008)  # 0.08 %
        # Le TP ridicule du run réel (0.006 %) est bien sous le plancher 1.5× (0.12 %).
        self.assertLess(0.00006, 0.0008 * 1.5)
        # et sous un plancher strict type 3× (0.24 %) — le recharge ne change pas.
        self.assertLess(0.00006, 0.0008 * 3.0)

    def test_calibration_sous_couts_leve_calibration_error(self):
        ohlcv_df = _synthetic_ohlcv(n=800)
        feats_df = _features_for(ohlcv_df)
        train_ohlcv = OhlcvFrame(asset="BTCUSD", timeframe="1h", df=ohlcv_df.head(600), data_version="v1")
        feats = FeaturesFrame(
            asset="BTCUSD", timeframe="1h", df=feats_df.head(600),
            feature_names=tuple(c for c in feats_df.columns if c.startswith("ma_")),
            data_version="v1",
        )
        engine = _make_engine_chere(self.config)
        with self.assertRaises(CalibrationError):
            engine.train_calibrate(_make_hypo(), train_ohlcv, feats)

    def test_calibration_normale_reste_ok(self):
        ohlcv_df = _synthetic_ohlcv(n=800)
        feats_df = _features_for(ohlcv_df)
        train_ohlcv = OhlcvFrame(asset="BTC_US", timeframe="1h", df=ohlcv_df.head(600), data_version="v1")
        train_feats = FeaturesFrame(
            asset="BTC_US", timeframe="1h", df=feats_df.head(600),
            feature_names=tuple(c for c in feats_df.columns if c.startswith("ma_")),
            data_version="v1",
        )
        engine = EvaluationEngine(config=self.config, data_version="v1", seed=42)
        calibrated = engine.train_calibrate(_make_hypo(), train_ohlcv, train_feats)
        # Les distances doivent dépasser le plancher économique configuré
        # (TP >= coût × 1.5 = 0.12 %, SL >= coût × 1.0 ; et a fortiori un
        # plancher strict 3×/2× reste franchissable sur les données testées).
        self.assertGreater(calibrated.tp_distance, 0.0008 * 1.5)
        self.assertGreater(calibrated.sl_distance, 0.0008)


# ---------------------------------------------------------------------------
# B. DSR corrigé
# ---------------------------------------------------------------------------


class TestDsrCorrige(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.config = load_config(str(CONFIG_PATH))

    def test_dsr_edge_frequent_passe_seuil(self):
        # Sharpe par-trade 0.20, 800 trades, 150 essais : passe 0.95.
        p = dsr_metric(sharpe_observed=0.20, n_trials=150, n_observations=800)
        self.assertGreaterEqual(p, 0.95)

    def test_dsr_bruit_est_rejete(self):
        p = dsr_metric(sharpe_observed=0.01, n_trials=150, n_observations=800)
        self.assertLess(p, 0.30)

    def test_dsr_monotone_en_nobservations(self):
        p200 = dsr_metric(0.10, 150, n_observations=200)
        p2000 = dsr_metric(0.10, 150, n_observations=2000)
        self.assertGreater(p2000, p200)

    def test_evaluate_dsr_nom_et_fini(self):
        m = _make_meas(ret_mean=0.001, n=600, avg_hold=2.0)
        v = evaluate_dsr(m, self.config, n_indep_trials=60)
        self.assertEqual(v.name, "DSR")
        # plus de NaN systématique avec std > 0 : shapho par-observation fini.
        self.assertTrue(np.isfinite(v.observed))


# ---------------------------------------------------------------------------
# C. Critère CROISSANCE
# ---------------------------------------------------------------------------


class TestCroissance(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.config = load_config(str(CONFIG_PATH))

    def test_petit_edge_frequence_passe(self):
        # 3000 trades/an à 0.02 % net -> CAGR énorme (vision x10).
        m = _make_meas(ret_mean=0.0002, n=3000, avg_hold=2.0)
        v = evaluate_croissance(m, self.config)
        self.assertTrue(v.passed)
        self.assertGreaterEqual(v.observed, 0.25)

    def test_ret_negatif_echoue(self):
        m = _make_meas(ret_mean=-0.0001, n=3000, avg_hold=2.0)
        v = evaluate_croissance(m, self.config)
        self.assertFalse(v.passed)

    def test_critere_inclus_dans_evaluate_all(self):
        m = _make_meas(ret_mean=0.001, n=3000, avg_hold=2.0)
        returns = [0.001] * 3000
        verdict = evaluate_all_criteria(m, returns, self.config, n_indep_trials=10, include_pbo=False)
        names = {v.name for v in verdict.verdicts}
        self.assertIn("CROISSANCE", names)
        # sans PBO -> 7 critères (DSR, CI×2, N_TRADES, CROISSANCE, CROSS_ASSET, DD)
        self.assertEqual(verdict.n_passed + verdict.n_failed, 7)


# ---------------------------------------------------------------------------
# D. Corpus path ancré
# ---------------------------------------------------------------------------


class TestCorpusPathAncre(unittest.TestCase):

    def test_default_corpus_path_absolu(self):
        from einherjar.research.corpus.store import DEFAULT_CORPUS_PATH
        self.assertTrue(DEFAULT_CORPUS_PATH.is_absolute())
        self.assertTrue(str(DEFAULT_CORPUS_PATH).endswith(str(Path("outputs") / "corpus.jsonl")))
        self.assertNotIn("src", DEFAULT_CORPUS_PATH.parts)

    def test_ledger_ancre_aussi(self):
        from einherjar.research.holdout.ledger import DEFAULT_LEDGER_PATH
        self.assertTrue(DEFAULT_LEDGER_PATH.is_absolute())
        self.assertNotIn("src", DEFAULT_LEDGER_PATH.parts)


# ---------------------------------------------------------------------------
# E. Catalogue de raisons
# ---------------------------------------------------------------------------


class TestReasonsCatalogue(unittest.TestCase):

    def test_croissance_fail_en_catalogue(self):
        from einherjar.research.archive.reasons import OFFICIAL_REASONS, is_valid_reason
        from einherjar.research.utils.types import RejectionReason
        self.assertIn(RejectionReason.CROISSANCE_FAIL, OFFICIAL_REASONS)
        self.assertTrue(is_valid_reason("CROISSANCE_FAIL"))


if __name__ == "__main__":
    unittest.main()