"""
tests/test_refinement_selection_holdout.py — Tests smoke pour les 3 derniers modules.

Vérifie :
  - refinement/beam.py : BeamRefiner mute sans recalibrer SL/TP
  - selection/selector.py : GeneratorSelector extrait le winner + save/load
  - holdout/evaluator.py : HoldoutEvaluator fait 1 seule passe, refuse au 2e
"""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl

from einherjar.research.config.loader import load_config
from einherjar.research.data.features import FeaturesFrame
from einherjar.research.data.ohlcv import OhlcvFrame
from einherjar.research.engine.evaluator import CalibratedParams, EvaluationEngine
from einherjar.research.generators.algorithms import (
    RandomSearchGenerator,
    make_all_generators,
)
from einherjar.research.generators.comparator import ComparisonReport, GeneratorRanking, GeneratorComparator
from einherjar.research.generators.protocol import make_protocol
from einherjar.research.holdout.evaluator import HoldoutEvaluator, HoldoutResult
from einherjar.research.refinement.beam import BeamRefiner, RefinementResult
from einherjar.research.selection.selector import GeneratorSelector, SelectedGenerator
from einherjar.research.generators.protocol import GenerationProtocol
from einherjar.research.utils.types import (
    Amplitude,
    AmplitudeUnit,
    CompareOp,
    Condition,
    Direction,
    Hypothesis,
    Universe,
)


def _make_synthetic_ohlcv(n: int = 800) -> pl.DataFrame:
    rng = np.random.default_rng(seed=42)
    close = 100.0 * np.cumprod(1.0 + rng.normal(0, 0.01, n))
    high = close * (1.0 + np.abs(rng.normal(0, 0.005, n)))
    low = close * (1.0 - np.abs(rng.normal(0, 0.005, n)))
    open_ = close * (1.0 + rng.normal(0, 0.002, n))
    volume = rng.integers(100, 10_000, n).astype(float)
    ts = [datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(hours=i) for i in range(n)]
    return pl.DataFrame({
        "asset": ["BTCUSD"] * n,
        "timeframe": ["1h"] * n,
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


def _make_engine_and_data(config, n: int = 800):
    from einherjar.research.data.features import make_test_provider
    provider_features = make_test_provider(config)
    ohlcv_df = _make_synthetic_ohlcv(n=n)
    feats_df = _make_synthetic_features(ohlcv_df)
    train_ohlcv = OhlcvFrame(asset="BTCUSD", timeframe="1h", df=ohlcv_df.head(500), data_version="v1")
    val_ohlcv = OhlcvFrame(asset="BTCUSD", timeframe="1h", df=ohlcv_df.slice(500, 200), data_version="v1")
    holdout_ohlcv = OhlcvFrame(asset="BTCUSD", timeframe="1h", df=ohlcv_df.tail(100), data_version="v1")
    train_feats = provider_features.compute(train_ohlcv)
    val_feats = provider_features.compute(val_ohlcv)
    holdout_feats = provider_features.compute(holdout_ohlcv)
    engine = EvaluationEngine(config=config, data_version="v1", seed=42)
    return engine, train_ohlcv, train_feats, val_ohlcv, val_feats, holdout_ohlcv, holdout_feats


# =========================================================================== #
# Tests refinement
# =========================================================================== #


class TestRefinement(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.config = load_config("src/einherjar/research/config")
        cls.engine, cls.train_ohlcv, cls.train_feats, cls.val_ohlcv, cls.val_feats, cls.holdout_ohlcv, cls.holdout_feats = _make_engine_and_data(cls.config)

    def _make_hypothesis(self) -> Hypothesis:
        # Condition permissive (close > 0) pour garantir le déclenchement.
        return Hypothesis(
            id="hyp_test",
            condition_tree=Condition(feature_ref="close", operator=CompareOp.GT, value=0.0),
            amplitude=Amplitude(valeur=10.0, unité=AmplitudeUnit.PRICE_ABSOLU, direction_implicite=Direction.LONG),
            direction=Direction.LONG,
            universe=Universe(assets=("BTCUSD",), timeframes=("1h",)),
        )

    def test_beam_refiner_runs(self):
        h = self._make_hypothesis()
        calibrated = self.engine.train_calibrate(h, self.train_ohlcv, self.train_feats)
        refiner = BeamRefiner(self.config, self.engine, seed=42, beam_width=4, max_iterations=3)
        result = refiner.refine(h, calibrated, self.train_ohlcv, self.train_feats, self.val_ohlcv, self.val_feats)
        self.assertIsInstance(result, RefinementResult)
        self.assertEqual(result.original_hypothesis_id, h.id)
        self.assertGreaterEqual(result.n_evaluated, 0)

    def test_refiner_does_not_recalibrate_sl_tp(self):
        """Le refiner doit utiliser la CalibratedParams fournie, sans la muter."""
        h = self._make_hypothesis()
        calibrated = self.engine.train_calibrate(h, self.train_ohlcv, self.train_feats)
        sl_before = calibrated.sl_n_atr
        tp_before = calibrated.tp_n_atr
        n_before = calibrated.n_window
        refiner = BeamRefiner(self.config, self.engine, seed=42, beam_width=4, max_iterations=3)
        refiner.refine(h, calibrated, self.train_ohlcv, self.train_feats, self.val_ohlcv, self.val_feats)
        self.assertEqual(calibrated.sl_n_atr, sl_before)
        self.assertEqual(calibrated.tp_n_atr, tp_before)
        self.assertEqual(calibrated.n_window, n_before)


# =========================================================================== #
# Tests selection
# =========================================================================== #


def _make_fake_report(winner_name: str = "RandomSearchGenerator") -> ComparisonReport:
    rankings = [
        GeneratorRanking(
            generator_name="RandomSearchGenerator", rank=1, score=1.5,
            n_generated=200, n_evaluated=180, n_passed_admission=10,
            admission_rate=0.055, median_sharpe=1.2, median_sharpe_all=0.5,
            n_distinct_features=15, semantic_coherence=0.0,
            elapsed_s=1.0,
        ),
        GeneratorRanking(
            generator_name="BeamSearchGenerator", rank=2, score=1.2,
            n_generated=200, n_evaluated=190, n_passed_admission=8,
            admission_rate=0.042, median_sharpe=1.0, median_sharpe_all=0.4,
            n_distinct_features=12, semantic_coherence=0.0,
            elapsed_s=2.0,
        ),
    ]
    rankings = sorted(rankings, key=lambda r: r.score, reverse=True)
    for i, r in enumerate(rankings):
        rankings[i] = GeneratorRanking(
            generator_name=r.generator_name, rank=i + 1, score=r.score,
            n_generated=r.n_generated, n_evaluated=r.n_evaluated,
            n_passed_admission=r.n_passed_admission, admission_rate=r.admission_rate,
            median_sharpe=r.median_sharpe, median_sharpe_all=r.median_sharpe_all,
            n_distinct_features=r.n_distinct_features,
            semantic_coherence=r.semantic_coherence,
            elapsed_s=r.elapsed_s,
        )
    return ComparisonReport(
        protocol=GenerationProtocol(seed=42, data_version="v1"),
        rankings=rankings,
        winner_name=rankings[0].generator_name if rankings else None,
    )


class TestSelection(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.config = load_config("src/einherjar/research/config")
        cls.protocol = make_protocol(cls.config, data_version="v1", seed=42)

    def test_selector_picks_winner(self):
        report = _make_fake_report()
        selector = GeneratorSelector(protocol=self.protocol)
        selected = selector.select(report)
        self.assertIsInstance(selected, SelectedGenerator)
        self.assertEqual(selected.generator_name, "RandomSearchGenerator")
        self.assertEqual(selected.rank, 1)
        self.assertEqual(selected.score, 1.5)

    def test_selector_save_load(self):
        report = _make_fake_report()
        selector = GeneratorSelector(protocol=self.protocol)
        selected = selector.select(report)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "selection.json"
            selector.save(selected, path)
            loaded = GeneratorSelector.load(path)
            self.assertEqual(loaded.generator_name, selected.generator_name)
            self.assertEqual(loaded.score, selected.score)
            self.assertEqual(loaded.protocol.seed, selected.protocol.seed)

    def test_selector_instantiate_winner(self):
        report = _make_fake_report()
        selector = GeneratorSelector(protocol=self.protocol)
        selected = selector.select(report)
        gen = GeneratorSelector.instantiate(selected, self.config)
        # Le RandomSearchGenerator a un attribut `protocol` ; vérifions juste qu'il existe.
        self.assertTrue(hasattr(gen, "protocol"))


# =========================================================================== #
# Tests holdout
# =========================================================================== #


class TestHoldout(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.config = load_config("src/einherjar/research/config")
        cls.engine, cls.train_ohlcv, cls.train_feats, cls.val_ohlcv, cls.val_feats, cls.holdout_ohlcv, cls.holdout_feats = _make_engine_and_data(cls.config)

    def _make_hypothesis_and_calibrated(self) -> tuple[Hypothesis, CalibratedParams]:
        h = Hypothesis(
            id="hyp_holdout",
            condition_tree=Condition(feature_ref="close", operator=CompareOp.GT, value=0.0),
            amplitude=Amplitude(valeur=10.0, unité=AmplitudeUnit.PRICE_ABSOLU, direction_implicite=Direction.LONG),
            direction=Direction.LONG,
            universe=Universe(assets=("BTCUSD",), timeframes=("1h",)),
        )
        calibrated = self.engine.train_calibrate(h, self.train_ohlcv, self.train_feats)
        return h, calibrated

    def test_holdout_evaluator_runs_once(self):
        h, calibrated = self._make_hypothesis_and_calibrated()
        val_mesures = self.engine.test_on(h, self.val_ohlcv, self.val_feats, calibrated, "val")
        from einherjar.research.engine.evaluator import EvaluationEngine
        from einherjar.research.holdout.ledger import HoldoutLedger
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmpdir:
            fresh_engine = EvaluationEngine(config=self.config, data_version="v1", seed=42)
            holdout_eval = HoldoutEvaluator(
                engine=fresh_engine, config=self.config, data_version="v1", seed=42,
                ledger=HoldoutLedger(path=Path(tmpdir) / "ledger.jsonl"),
            )
            result = holdout_eval.evaluate(
                h, calibrated, self.holdout_ohlcv, self.holdout_feats,
                val_sharpe=val_mesures.sharpe_net,
            )
            self.assertIsInstance(result, HoldoutResult)
            self.assertEqual(result.hypothesis_id, h.id)
            self.assertIn(result.degradation_flag, ("OK", "WARNING", "CRITICAL"))

    def test_holdout_evaluator_refuses_second_call(self):
        h, calibrated = self._make_hypothesis_and_calibrated()
        from einherjar.research.engine.evaluator import EvaluationEngine
        from einherjar.research.holdout.ledger import HoldoutLedger
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            fresh_engine = EvaluationEngine(config=self.config, data_version="v1", seed=42)
            holdout_eval = HoldoutEvaluator(
                engine=fresh_engine, config=self.config, data_version="v1", seed=42,
                ledger=HoldoutLedger(path=__import__("pathlib").Path(tmpdir) / "ledger.jsonl"),
            )
            holdout_eval.evaluate(h, calibrated, self.holdout_ohlcv, self.holdout_feats, val_sharpe=1.0)
            with self.assertRaises(RuntimeError):
                holdout_eval.evaluate(h, calibrated, self.holdout_ohlcv, self.holdout_feats, val_sharpe=1.0)


# =========================================================================== #
# Tests discovery CLI
# =========================================================================== #


class TestDiscoveryCLI(unittest.TestCase):

    def test_dry_run_engine(self):
        import subprocess
        import sys
        result = subprocess.run(
            [
                sys.executable, "-m", "einherjar.research.discovery",
                "engine", "--config", "src/einherjar/research/config", "--dry-run",
            ],
            capture_output=True, text=True, cwd="D:\\midas_v2\\Einherjar",
            env={**__import__("os").environ, "PYTHONPATH": "src"},
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("DRY-RUN", result.stdout + result.stderr)

    def test_dry_run_pipeline(self):
        import os
        import subprocess
        import sys
        result = subprocess.run(
            [
                sys.executable, "-m", "einherjar.research.discovery",
                "run", "--config", "src/einherjar/research/config", "--dry-run",
            ],
            capture_output=True, text=True, cwd="D:\\midas_v2\\Einherjar",
            env={**os.environ, "PYTHONPATH": "src"},
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)


if __name__ == "__main__":
    unittest.main()
