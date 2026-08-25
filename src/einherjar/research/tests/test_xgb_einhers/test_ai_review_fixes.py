"""test_ai_review_fixes.py - Tests pour les fix de la revue AI 2026-08-20.

Couvre :
- P1-1 : one-sided p-value (t<=0 => p=1.0)
- P0-5 : admission check holdout_metrics
- P2-1 : merge_paths_or pour DNF
- P0-2 : cost floor crypto-only (implicite dans runner)
"""
import unittest
from datetime import datetime

import numpy as np
import polars as pl

from einherjar.research.xgb_einhers.admission import AdmissionConfig, check_admission
from einherjar.research.xgb_einhers.backtester import compute_metrics
from einherjar.research.xgb_einhers.condition_tree import merge_paths_or
from einherjar.research.xgb_einhers.path_extractor import XGBPath
from einherjar.research.xgb_einhers.types import (
    Condition,
    ConditionNode,
    Einher,
    EinherMetrics,
    TradeResult,
)


def make_trade(net: float, n: int = 30, noise: float = 0.005) -> list[TradeResult]:
    """Construit N trades avec mean ~net et un peu de bruit (std > 0)."""
    np.random.seed(42)
    rets = [net + np.random.normal(0, noise) for _ in range(n)]
    return [
        TradeResult(
            entry_idx=i, exit_idx=i+1, entry_price=100.0,
            exit_price=100.0 * (1 + r), exit_reason="tp",
            gross_return=r + 0.001, net_return=r,
            n_bars_held=2, entry_timestamp_ms=0, exit_timestamp_ms=0,
        )
        for i, r in enumerate(rets)
    ]


def make_einher(name: str, val_m: EinherMetrics, holdout_m: EinherMetrics = None) -> Einher:
    cond = Condition(feature_ref="x", operator="<", value=0.0)
    return Einher(
        id=f"einher_{name}",
        condition_tree=cond,
        direction="BUY",
        amplitude_bars=48,
        tp_pct=0.02,
        sl_pct=0.01,
        universe={"asset": "BTCUSD", "asset_class": "crypto", "timeframe": "1h", "horizon": "2d", "horizon_bars": 48},
        metrics=val_m,
        scope="asset",
        holdout_metrics=holdout_m,
    )


class TestP1_1_OneSidedPValue(unittest.TestCase):
    """P1-1 : p-value one-sided upper tail (t<=0 => p=1.0)."""

    def test_losing_strategy_pvalue_is_one(self):
        """Strat perdante (mean<0) doit avoir p=1.0 (ne peut pas rejeter H0:mu<=0)."""
        trades = make_trade(-0.01)  # 30 trades perdants
        m = compute_metrics(trades, buy_hold_return=0.0)
        self.assertLess(m.t_statistic, 0)
        self.assertEqual(m.p_value, 1.0,
                         f"Expected p=1.0 for t<0, got {m.p_value}")

    def test_winning_strategy_pvalue_is_low(self):
        """Strat gagnante (mean>0) doit avoir p petit."""
        trades = make_trade(+0.02)  # 30 trades gagnants
        m = compute_metrics(trades, buy_hold_return=0.0)
        self.assertGreater(m.t_statistic, 0)
        self.assertLess(m.p_value, 0.05,
                        f"Expected p<0.05 for t>0, got {m.p_value}")

    def test_zero_tstat_pvalue_is_one(self):
        """t=0 doit avoir p=1.0 (ne peut pas rejeter)."""
        # Construire des trades avec mean exact = 0 (difficile mais possible)
        # On utilise returns symetriques
        rets = [-0.02, -0.01, 0, 0.01, 0.02, -0.02, -0.01, 0, 0.01, 0.02] * 3
        trades = [
            TradeResult(
                entry_idx=i, exit_idx=i+1, entry_price=100.0,
                exit_price=100.0 * (1 + r), exit_reason="tp",
                gross_return=r, net_return=r,
                n_bars_held=2, entry_timestamp_ms=0, exit_timestamp_ms=0,
            )
            for i, r in enumerate(rets)
        ]
        m = compute_metrics(trades, buy_hold_return=0.0)
        # mean = 0 exactement
        self.assertAlmostEqual(m.avg_net_return, 0.0, places=6)
        self.assertEqual(m.p_value, 1.0)

    def test_two_sided_was_buggy(self):
        """Test que confirme le bug P1-1 etait bien present.

        Avant le fix : un strat tres perdante avait p_value petit
        (parce que deux-sided utilise abs(t)).
        Apres le fix : p_value=1.0 pour t<=0.
        """
        # 50 trades tres perdants (tres consistants)
        trades = make_trade(-0.02, n=50)
        m = compute_metrics(trades, buy_hold_return=0.0)
        # Verification : le deux-sided aurait donne ~0
        from math import erf, sqrt
        t = abs(m.t_statistic)
        two_sided_p = 2.0 * (1.0 - 0.5 * (1.0 + erf(t / sqrt(2.0))))
        # Two-sided : tres petit
        self.assertLess(two_sided_p, 0.001)
        # Mais one-sided upper : 1.0
        self.assertEqual(m.p_value, 1.0)


class TestP0_5_HoldoutAdmission(unittest.TestCase):
    """P0-5 : admission check holdout_metrics performance."""

    def test_admit_if_holdout_good(self):
        """Holdout OK (Sharpe>0, wr>=0.36) -> admitted."""
        val_m = EinherMetrics(
            n_trades=50, n_tp=40, n_sl=10, n_timeout=0, win_rate=0.80,
            avg_net_return=0.02, total_return=1.0, sharpe_ratio=5.0,
            max_drawdown=-0.05, profit_factor=3.0, avg_holding_bars=4,
            buy_hold_return=0.1, alpha=0.9,
        )
        holdout_m = EinherMetrics(
            n_trades=20, n_tp=15, n_sl=5, n_timeout=0, win_rate=0.75,
            avg_net_return=0.015, total_return=0.3, sharpe_ratio=2.5,
            max_drawdown=-0.08, profit_factor=2.0, avg_holding_bars=4,
            buy_hold_return=0.05, alpha=0.25,
        )
        einher = make_einher("ok", val_m, holdout_m)
        passed, reason = check_admission(einher, AdmissionConfig(min_holdout_trades=5))
        self.assertTrue(passed, f"Should be admitted but: {reason}")

    def test_reject_if_holdout_sharpe_negative(self):
        """Sharpe holdout < 0 (alors que val > 0) -> rejete (overfit)."""
        val_m = EinherMetrics(
            n_trades=50, n_tp=40, n_sl=10, n_timeout=0, win_rate=0.80,
            avg_net_return=0.02, total_return=1.0, sharpe_ratio=5.0,
            max_drawdown=-0.05, profit_factor=3.0, avg_holding_bars=4,
            buy_hold_return=0.1, alpha=0.9,
        )
        # Holdout collapse : Sharpe=-1.5
        holdout_m = EinherMetrics(
            n_trades=20, n_tp=5, n_sl=15, n_timeout=0, win_rate=0.25,
            avg_net_return=-0.01, total_return=-0.2, sharpe_ratio=-1.5,
            max_drawdown=-0.25, profit_factor=0.4, avg_holding_bars=4,
            buy_hold_return=0.0, alpha=-0.2,
        )
        einher = make_einher("overfit", val_m, holdout_m)
        passed, reason = check_admission(einher, AdmissionConfig(min_holdout_trades=5))
        self.assertFalse(passed)
        self.assertIn("Holdout", reason)

    def test_reject_if_holdout_no_trades(self):
        """Holdout 0 trades -> rejete (min_holdout_trades)."""
        val_m = EinherMetrics(
            n_trades=50, n_tp=40, n_sl=10, n_timeout=0, win_rate=0.80,
            avg_net_return=0.02, total_return=1.0, sharpe_ratio=5.0,
            max_drawdown=-0.05, profit_factor=3.0, avg_holding_bars=4,
            buy_hold_return=0.1, alpha=0.9,
        )
        holdout_m = EinherMetrics(
            n_trades=0, n_tp=0, n_sl=0, n_timeout=0, win_rate=0.0,
            avg_net_return=0.0, total_return=0.0, sharpe_ratio=0.0,
            max_drawdown=0.0, profit_factor=0.0, avg_holding_bars=0.0,
            buy_hold_return=0.0, alpha=0.0,
        )
        einher = make_einher("no_holdout", val_m, holdout_m)
        passed, reason = check_admission(einher, AdmissionConfig(min_holdout_trades=5))
        self.assertFalse(passed)
        self.assertIn("Holdout", reason)


class TestP2_1_MergePathsOr(unittest.TestCase):
    """P2-1 : merge_paths_or combine plusieurs paths en DNF (OR)."""

    def test_single_path_returns_ast(self):
        path = XGBPath(conditions=(("x", "<", 0.5),), score=0.5, tree_idx=0, path_idx=0)
        ast = merge_paths_or([path])
        self.assertIsInstance(ast, Condition)

    def test_two_paths_or_node(self):
        p1 = XGBPath(conditions=(("x", "<", 0.5),), score=0.5, tree_idx=0, path_idx=0)
        p2 = XGBPath(conditions=(("y", ">", 0.7),), score=0.5, tree_idx=0, path_idx=1)
        ast = merge_paths_or([p1, p2])
        self.assertIsInstance(ast, ConditionNode)
        self.assertEqual(ast.op, "OR")

    def test_three_paths_nested_or(self):
        p1 = XGBPath(conditions=(("x", "<", 0.5),), score=0.5, tree_idx=0, path_idx=0)
        p2 = XGBPath(conditions=(("y", ">", 0.7),), score=0.5, tree_idx=0, path_idx=1)
        p3 = XGBPath(conditions=(("z", "<=", 0.3),), score=0.5, tree_idx=0, path_idx=2)
        ast = merge_paths_or([p1, p2, p3])
        # OR(p1, OR(p2, p3))
        self.assertEqual(ast.op, "OR")
        self.assertIsInstance(ast.left, Condition)
        self.assertEqual(ast.right.op, "OR")
        self.assertIsInstance(ast.right.left, Condition)
        self.assertIsInstance(ast.right.right, Condition)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            merge_paths_or([])

    def test_multi_condition_paths_in_or(self):
        p1 = XGBPath(conditions=(("x", "<", 0.5), ("y", ">", 0.7)), score=0.5, tree_idx=0, path_idx=0)
        p2 = XGBPath(conditions=(("z", "<=", 0.3), ("w", ">=", 0.1)), score=0.5, tree_idx=0, path_idx=1)
        ast = merge_paths_or([p1, p2])
        # OR(AND(x, y), AND(z, w))
        self.assertEqual(ast.op, "OR")
        self.assertEqual(ast.left.op, "AND")
        self.assertEqual(ast.right.op, "AND")


class TestP0_1_ATRBasedTPSL(unittest.TestCase):
    """P0-1 : TP/SL dynamiques ATR-based au lieu de hardcoded 2.5%/1.5%."""

    def test_low_vol_series_has_low_tp(self):
        """Serie avec faible volatilite (forex-like) doit avoir TP < 1%."""
        from einherjar.research.xgb_einhers.backtester import backtest_einher
        n = 500
        base_price = 1.1000
        # 0.05% range par bar (forex-like)
        np.random.seed(42)
        noise = np.random.normal(0, 0.0003, n).cumsum()
        close = base_price + noise
        high = close + 0.0003
        low = close - 0.0003
        # Construire timestamps avec arange (1h par pas depuis epoch)
        ts = pl.datetime_range(
            datetime(2023, 1, 1),
            datetime(2023, 1, 1) + __import__("datetime").timedelta(hours=n - 1),
            interval="1h", eager=True,
        ).alias("timestamp")
        df = pl.DataFrame({
            "timestamp": ts,
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.ones(n) * 1000.0,
        })
        cond = Condition(feature_ref="x", operator="<", value=999.0)  # always True
        e = Einher(
            id="e1", condition_tree=cond, direction="BUY",
            amplitude_bars=6, tp_pct=0.0, sl_pct=0.0,
            universe={"asset": "EURUSD", "asset_class": "forex",
                      "timeframe": "1h", "horizon": "6h", "horizon_bars": 6},
            metrics=EinherMetrics(
                n_trades=0, n_tp=0, n_sl=0, n_timeout=0,
                win_rate=0.0, avg_net_return=0.0, total_return=0.0,
                sharpe_ratio=0.0, max_drawdown=0.0, profit_factor=0.0,
                avg_holding_bars=0.0, buy_hold_return=0.0, alpha=0.0,
            ),
            scope="asset",
        )
        X = np.random.randn(n, 1).astype(np.float32)
        result = backtest_einher(
            einher=e, ohlcv_df=df, X=X, feature_names=["x"],
            costs_pct=0.0002,
        )
        # TP doit etre < 1% (forex-like)
        self.assertLess(result.effective_tp_pct, 0.01,
                        f"Forex TP should be < 1%, got {result.effective_tp_pct:.4f}")
        self.assertLess(result.effective_sl_pct, 0.01,
                        f"Forex SL should be < 1%, got {result.effective_sl_pct:.4f}")


if __name__ == "__main__":
    unittest.main()
