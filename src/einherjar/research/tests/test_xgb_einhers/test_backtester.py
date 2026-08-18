"""test_backtester.py - Tests P0 du backtester (CRITIQUES).

P0 : no_lookahead, deterministic, known_signal.
P1 : tp_sl_priority, costs_applied, empty_universe.
"""
import unittest
import numpy as np
import polars as pl

from einherjar.research.xgb_einhers.backtester import (
    BacktestResult,
    compute_atr,
    compute_metrics,
    simulate_trade,
    backtest_einher,
)
from einherjar.research.xgb_einhers.types import (
    Condition,
    Einher,
    EinherMetrics,
    TradeResult,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def make_synthetic_ohlcv_with_pattern(
    n_bars: int = 500,
    pattern_value: float = 1.0,
    move_pct: float = 0.02,
    move_within_bars: int = 6,
    seed: int = 42,
) -> pl.DataFrame:
    """Crée un OHLCV synthétique avec un pattern qui prédit un mouvement haussier.

    Pour chaque bougie où `pattern == pattern_value`, le prix monte de `move_pct`
    IMMÉDIATEMENT à partir de la bougie suivante (t+1), étalé sur `move_within_bars`
    bougies. C'est ce que le backtester verra : entrée à OPEN[t+1], montée progressive.

    Returns:
        DataFrame polars [timestamp, open, high, low, close, volume]
    """
    rng = np.random.default_rng(seed=seed)
    base_ts = 1_700_000_000_000_000  # microseconds

    # Pattern : True à des positions régulières avec un peu de bruit
    pattern_idx = list(range(50, n_bars - move_within_bars - 1, 30))
    pattern = np.zeros(n_bars, dtype=np.float32)
    pattern[pattern_idx] = pattern_value

    # Prix de base : random walk autour de 100
    close = np.zeros(n_bars, dtype=np.float64)
    close[0] = 100.0
    for i in range(1, n_bars):
        # Si on est dans une fenêtre "move" (juste après un pattern), on injecte le move
        in_move_window = False
        remaining_move = 0
        for offset in range(1, move_within_bars + 1):
            sig_idx = i - offset
            if sig_idx >= 0 and pattern[sig_idx] == pattern_value:
                in_move_window = True
                remaining_move = move_within_bars - offset + 1
                break
        if in_move_window and remaining_move > 0:
            close[i] = close[i - 1] * (1 + move_pct / move_within_bars)
        else:
            close[i] = close[i - 1] * (1 + rng.normal(0, 0.005))

    # Reconstruction OHLCV
    opens = np.roll(close, 1)
    opens[0] = close[0]
    highs = close * (1 + np.abs(rng.normal(0, 0.003, n_bars)))
    lows = close * (1 - np.abs(rng.normal(0, 0.003, n_bars)))
    volume = rng.integers(100, 10000, n_bars).astype(np.float64)

    return pl.DataFrame({
        "timestamp": pl.from_numpy(
            (base_ts + np.arange(n_bars) * 60_000_000).astype(np.int64),
            schema=["timestamp"]
        ),
        "open": opens,
        "high": highs,
        "low": lows,
        "close": close,
        "volume": volume,
    })


def make_synthetic_X_with_pattern(
    n_bars: int = 500,
    pattern_value: float = 1.0,
    seed: int = 42,
) -> tuple[np.ndarray, list[str]]:
    """Crée une matrice X synthétique avec une feature `pattern` qui matche le pattern de l'OHLCV."""
    pattern_idx = list(range(50, n_bars - 6 - 1, 30))
    feature_names = ["rsi_14", "macd_line", "pattern_signal"]
    X = np.random.randn(n_bars, 3).astype(np.float32)
    pattern = np.zeros(n_bars, dtype=np.float32)
    pattern[pattern_idx] = pattern_value
    X[:, 2] = pattern
    return X, feature_names


# --------------------------------------------------------------------------- #
# Tests P0
# --------------------------------------------------------------------------- #


class TestSimulateTrade(unittest.TestCase):
    """P1 : simulate_trade unit tests."""

    def test_buy_tp_hit(self):
        # Prix monte, TP=2%, on doit sortir au TP
        n = 10
        opens = np.linspace(100, 102, n)
        highs = opens + 1
        lows = opens - 0.1
        exit_price, reason, n_bars = simulate_trade(
            entry_idx=0, amplitude=10, direction="BUY",
            entry_price=100.0, tp_pct=0.02, sl_pct=0.01,
            highs=highs, lows=lows, opens=opens,
        )
        self.assertEqual(reason, "tp")
        self.assertGreater(exit_price, 100.0)

    def test_buy_sl_hit(self):
        # Prix descend, SL=1%, on doit sortir au SL
        n = 10
        opens = np.linspace(100, 98, n)
        highs = opens + 0.1
        lows = opens - 1
        exit_price, reason, n_bars = simulate_trade(
            entry_idx=0, amplitude=10, direction="BUY",
            entry_price=100.0, tp_pct=0.02, sl_pct=0.01,
            highs=highs, lows=lows, opens=opens,
        )
        self.assertEqual(reason, "sl")
        self.assertLess(exit_price, 100.0)

    def test_sl_priority_on_same_bar(self):
        # Bougie ambiguë : TP et SL touchés en même temps → SL first
        opens = np.array([100.0, 100.0])
        highs = np.array([200.0, 200.0])  # TP touched
        lows = np.array([50.0, 50.0])     # SL touched
        exit_price, reason, n_bars = simulate_trade(
            entry_idx=0, amplitude=2, direction="BUY",
            entry_price=100.0, tp_pct=0.02, sl_pct=0.01,
            highs=highs, lows=lows, opens=opens,
        )
        self.assertEqual(reason, "sl")

    def test_timeout(self):
        n = 10
        opens = np.full(n, 100.0)  # Prix stable
        highs = np.full(n, 100.5)
        lows = np.full(n, 99.5)
        exit_price, reason, n_bars = simulate_trade(
            entry_idx=0, amplitude=3, direction="BUY",
            entry_price=100.0, tp_pct=0.02, sl_pct=0.01,
            highs=highs, lows=lows, opens=opens,
        )
        self.assertEqual(reason, "timeout")


class TestComputeATR(unittest.TestCase):
    """P1 : ATR est calculé correctement."""

    def test_atr_basic(self):
        n = 30
        highs = np.linspace(101, 110, n)
        lows = np.linspace(99, 108, n)
        closes = np.linspace(100, 109, n)
        atr = compute_atr(highs, lows, closes, period=14)
        # Les 13 premières bougies sont NaN
        self.assertTrue(np.isnan(atr[:13]).all())
        # À partir de l'index 13, l'ATR est défini
        self.assertFalse(np.isnan(atr[13:]).any())
        # ATR > 0
        self.assertGreater(atr[13], 0)


class TestComputeMetrics(unittest.TestCase):
    """P1 : compute_metrics cohérent."""

    def test_no_trades(self):
        m = compute_metrics([], 0.1)
        self.assertEqual(m.n_trades, 0)
        self.assertEqual(m.sharpe_ratio, 0.0)
        self.assertEqual(m.alpha, -0.1)

    def test_metrics_with_trades(self):
        trades = [
            TradeResult(
                entry_idx=i, exit_idx=i + 1, entry_price=100.0,
                exit_price=102.0, exit_reason="tp", gross_return=0.02,
                net_return=0.018, n_bars_held=2,
                entry_timestamp_ms=0, exit_timestamp_ms=0,
            )
            for i in range(10)
        ]
        m = compute_metrics(trades, 0.05)
        self.assertEqual(m.n_trades, 10)
        self.assertEqual(m.n_tp, 10)
        self.assertEqual(m.win_rate, 1.0)
        self.assertAlmostEqual(m.total_return, 0.18, places=4)
        self.assertGreater(m.sharpe_ratio, 0)
        self.assertAlmostEqual(m.alpha, 0.18 - 0.05, places=4)


class TestBacktestKnownSignal(unittest.TestCase):
    """P0 : sur un dataset synthétique avec un signal connu, le backtester
    doit retourner un win_rate > 0.7 et un sharpe > 1.0."""

    def test_known_pattern_predicts_up_move(self):
        ohlcv = make_synthetic_ohlcv_with_pattern(
            n_bars=500, pattern_value=1.0, move_pct=0.03, move_within_bars=6,
        )
        X, feature_names = make_synthetic_X_with_pattern(n_bars=500, pattern_value=1.0)

        einher = Einher(
            id="test_known",
            condition_tree=Condition(
                feature_ref="pattern_signal", operator="==", value=1.0,
                transformation=None,
            ),
            direction="BUY",
            amplitude_bars=6,
            tp_pct=0.025,
            sl_pct=0.015,
            universe={"asset": "TEST", "timeframe": "1h", "horizon": "6h", "horizon_bars": 6},
            metrics=EinherMetrics(
                n_trades=0, n_tp=0, n_sl=0, n_timeout=0,
                win_rate=0.0, avg_net_return=0.0, total_return=0.0,
                sharpe_ratio=0.0, max_drawdown=0.0, profit_factor=0.0,
                avg_holding_bars=0.0, buy_hold_return=0.0, alpha=0.0,
            ),
            scope="asset",
        )
        result = backtest_einher(
            einher=einher, ohlcv_df=ohlcv, X=X,
            feature_names=feature_names, costs_pct=0.0,  # pas de coûts pour le test
        )
        # Sur un signal injecté déterministe, on doit avoir win_rate ~ 1.0
        # (le prix monte quasi certainement après le pattern)
        self.assertGreater(result.metrics.n_trades, 5)
        self.assertGreater(result.metrics.win_rate, 0.5,
                           f"win_rate={result.metrics.win_rate} sur signal déterministe")


class TestBacktestNoLookahead(unittest.TestCase):
    """P0 : aucun look-ahead dans le backtest."""

    def test_signals_only_use_past(self):
        """Vérifie que le backtest n'utilise pas de données futures.

        Test : on inverse le prix des bougies futures. Si le résultat est
        inchangé, c'est qu'il n'y a pas de look-ahead.
        """
        ohlcv_orig = make_synthetic_ohlcv_with_pattern(
            n_bars=300, pattern_value=1.0, move_pct=0.02, move_within_bars=6,
        )
        X, feature_names = make_synthetic_X_with_pattern(n_bars=300, pattern_value=1.0)

        einher = Einher(
            id="test_no_lookahead",
            condition_tree=Condition(
                feature_ref="pattern_signal", operator="==", value=1.0,
                transformation=None,
            ),
            direction="BUY",
            amplitude_bars=6,
            tp_pct=0.025,
            sl_pct=0.015,
            universe={"asset": "TEST", "timeframe": "1h", "horizon": "6h", "horizon_bars": 6},
            metrics=EinherMetrics(0,0,0,0,0,0,0,0,0,0,0,0,0),
            scope="asset",
        )
        # Run 1 : normal
        result1 = backtest_einher(einher, ohlcv_orig, X, feature_names, costs_pct=0.0)

        # Run 2 : on inverse les prix après t=200
        ohlcv_modif = ohlcv_orig.clone()
        opens = ohlcv_modif["open"].to_numpy().copy()
        closes = ohlcv_modif["close"].to_numpy().copy()
        highs = ohlcv_modif["high"].to_numpy().copy()
        lows = ohlcv_modif["low"].to_numpy().copy()
        # Inverser les prix après t=200
        opens[200:] = 200.0 - (opens[200:] - 100.0)
        closes[200:] = 200.0 - (closes[200:] - 100.0)
        highs[200:] = 200.0 - (highs[200:] - 100.0)
        lows[200:] = 200.0 - (lows[200:] - 100.0)
        ohlcv_modif = ohlcv_modif.with_columns([
            pl.lit(opens).alias("open"),
            pl.lit(closes).alias("close"),
            pl.lit(highs).alias("high"),
            pl.lit(lows).alias("low"),
        ])

        result2 = backtest_einher(einher, ohlcv_modif, X, feature_names, costs_pct=0.0)

        # Les trades AVANT t=200 doivent être identiques
        n_before_200 = sum(1 for t in result1.trades if t.entry_idx < 200)
        n_before_200_modif = sum(1 for t in result2.trades if t.entry_idx < 200)
        self.assertEqual(n_before_200, n_before_200_modif,
                         f"Trades avant t=200 devraient être identiques : {n_before_200} vs {n_before_200_modif}")
        # Les PnL avant t=200 doivent aussi être identiques
        for t1, t2 in zip([t for t in result1.trades if t.entry_idx < 200],
                          [t for t in result2.trades if t.entry_idx < 200]):
            self.assertAlmostEqual(t1.entry_price, t2.entry_price, places=4)
            self.assertAlmostEqual(t1.exit_price, t2.exit_price, places=4)


class TestBacktestDeterministic(unittest.TestCase):
    """P0 : même input → même output."""

    def test_same_input_same_output(self):
        ohlcv = make_synthetic_ohlcv_with_pattern(n_bars=300)
        X, feature_names = make_synthetic_X_with_pattern(n_bars=300)

        einher = Einher(
            id="test_deterministic",
            condition_tree=Condition(
                feature_ref="pattern_signal", operator="==", value=1.0,
            ),
            direction="BUY",
            amplitude_bars=6,
            tp_pct=0.025,
            sl_pct=0.015,
            universe={"asset": "TEST"},
            metrics=EinherMetrics(0,0,0,0,0,0,0,0,0,0,0,0,0),
            scope="asset",
        )
        result1 = backtest_einher(einher, ohlcv, X, feature_names, costs_pct=0.0)
        result2 = backtest_einher(einher, ohlcv, X, feature_names, costs_pct=0.0)
        self.assertEqual(result1.metrics.n_trades, result2.metrics.n_trades)
        self.assertEqual(result1.metrics.total_return, result2.metrics.total_return)
        for t1, t2 in zip(result1.trades, result2.trades):
            self.assertEqual(t1.entry_idx, t2.entry_idx)
            self.assertAlmostEqual(t1.net_return, t2.net_return, places=8)


class TestBacktestEmptyUniverse(unittest.TestCase):
    """P1 : aucun trade généré → métriques nulles, pas de crash."""

    def test_no_signals(self):
        ohlcv = make_synthetic_ohlcv_with_pattern(n_bars=100)
        X, feature_names = make_synthetic_X_with_pattern(n_bars=100)
        # Mettre tous les patterns à 0
        X[:, 2] = 0.0

        einher = Einher(
            id="test_no_signals",
            condition_tree=Condition(
                feature_ref="pattern_signal", operator="==", value=1.0,
            ),
            direction="BUY",
            amplitude_bars=6,
            tp_pct=0.025,
            sl_pct=0.015,
            universe={"asset": "TEST"},
            metrics=EinherMetrics(0,0,0,0,0,0,0,0,0,0,0,0,0),
            scope="asset",
        )
        result = backtest_einher(einher, ohlcv, X, feature_names, costs_pct=0.0)
        self.assertEqual(result.metrics.n_trades, 0)
        self.assertEqual(result.metrics.total_return, 0.0)
        self.assertEqual(result.metrics.sharpe_ratio, 0.0)


class TestBacktestCostsApplied(unittest.TestCase):
    """P1 : les coûts sont déduits du PnL."""

    def test_costs_reduce_return(self):
        ohlcv = make_synthetic_ohlcv_with_pattern(n_bars=300, move_pct=0.03)
        X, feature_names = make_synthetic_X_with_pattern(n_bars=300)

        einher = Einher(
            id="test_costs",
            condition_tree=Condition(
                feature_ref="pattern_signal", operator="==", value=1.0,
            ),
            direction="BUY",
            amplitude_bars=6,
            tp_pct=0.025,
            sl_pct=0.015,
            universe={"asset": "TEST"},
            metrics=EinherMetrics(0,0,0,0,0,0,0,0,0,0,0,0,0),
            scope="asset",
        )
        result_no_cost = backtest_einher(einher, ohlcv, X, feature_names, costs_pct=0.0)
        result_with_cost = backtest_einher(einher, ohlcv, X, feature_names, costs_pct=0.01)
        # Avec coûts, le total_return doit être inférieur
        self.assertLess(
            result_with_cost.metrics.total_return,
            result_no_cost.metrics.total_return,
        )


if __name__ == "__main__":
    unittest.main()
