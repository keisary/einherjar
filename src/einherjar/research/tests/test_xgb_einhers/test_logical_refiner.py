"""Tests P3-4 : OR-de-régimes et veto-NOT (logical_refiner)."""
# pyright: reportAttributeAccessIssue=false
from __future__ import annotations

import numpy as np
import polars as pl

from einherjar.research.xgb_einhers.logical_refiner import (
    build_or_einher,
    evaluate_or_pairs,
    find_veto_condition,
    wracc,
)
from einherjar.research.xgb_einhers.path_extractor import XGBPath
from einherjar.research.xgb_einhers.types import Condition, Einher, EinherMetrics


def _mk_path(i: int, feat: str, score: float) -> XGBPath:
    return XGBPath(
        conditions=((feat, ">", 0.5),),  # seuil binaire standard
        score=score,
        tree_idx=i,
        path_idx=i,
    )


class TestWracc:
    def test_zero_for_full_coverage(self) -> None:
        y = np.array([1.0, -1.0, 1.0, -1.0])
        mask = np.ones(4, dtype=bool)
        assert abs(wracc(mask, y)) < 1e-12

    def test_positive_for_good_subgroup(self) -> None:
        # sous-groupe concentrant les positifs
        y = np.array([1.0, -1.0, 1.0, -1.0])
        mask = np.array([True, False, True, False])
        assert wracc(mask, y) > 0.0

    def test_empty_mask(self) -> None:
        y = np.array([1.0, -1.0])
        assert wracc(np.zeros(2, dtype=bool), y) == 0.0


class TestOrRegimes:
    def _two_regimes_dataset(self):
        """Deux régimes disjoints (features différents), même signal +."""
        rng = np.random.default_rng(7)
        n = 30000
        y = rng.normal(0, 0.008, n).astype(np.float32)
        X = np.zeros((n, 4), dtype=np.float32)
        regA = rng.random(n) > 0.995
        regB = rng.random(n) > 0.995
        X[regA, 0] = 1.0
        y[regA] += 0.004
        X[regB & ~regA, 1] = 1.0
        y[regB & ~regA] += 0.004
        names = ["feat_A", "feat_B", "noise1", "noise2"]
        pa = _mk_path(0, "feat_A", 0.01)
        pb = _mk_path(1, "feat_B", 0.008)
        return X, y.astype(np.float32), names, pa, pb

    def test_detects_complementary_pair(self) -> None:
        X, y, names, pa, pb = self._two_regimes_dataset()
        cands = evaluate_or_pairs([pa, pb], X, y, names, min_branch_t_stat=2.0)
        assert len(cands) == 1
        c = cands[0]
        assert {c.path_a.conditions[0][0], c.path_b.conditions[0][0]} == {"feat_A", "feat_B"}
        assert c.wracc_union > 0.0

    def test_rejects_opposite_signs(self) -> None:
        X, y, names, pa, _pb = self._two_regimes_dataset()
        pc = _mk_path(2, "noise1", -0.02)  # sens opposé
        cands = evaluate_or_pairs([pa, pc], X, y, names)
        assert not any(pc in (c.path_a, c.path_b) for c in cands)

    def test_rejects_same_head_feature(self) -> None:
        X, y, names, pa, _pb = self._two_regimes_dataset()
        pd_ = _mk_path(3, "feat_A", 0.009)  # même feature dominant que pa
        cands = evaluate_or_pairs([pa, pd_], X, y, names)
        assert not any({c.path_a, c.path_b} == {pa, pd_} for c in cands)

    def test_rejects_weak_branch(self) -> None:
        X, y, names, pa, _pb = self._two_regimes_dataset()
        weak = XGBPath(
            conditions=(("noise2", "<", 0.001),), score=0.005, tree_idx=9, path_idx=9
        )
        cands = evaluate_or_pairs([pa, weak], X, y, names, min_branch_t_stat=2.0)
        assert not any(weak in (c.path_a, c.path_b) for c in cands)

    def test_build_or_einher_structure(self) -> None:
        X, y, names, pa, pb = self._two_regimes_dataset()
        cands = evaluate_or_pairs([pa, pb], X, y, names)
        template_metrics = EinherMetrics(
            n_trades=10, n_tp=4, n_sl=3, n_timeout=3, win_rate=0.4,
            avg_net_return=0.001, total_return=0.01, sharpe_ratio=1.0,
            max_drawdown=-0.02, profit_factor=1.2, avg_holding_bars=3,
            buy_hold_return=0.0, alpha=0.01,
        )
        template = Einher(
            id="tpl", condition_tree=Condition(feature_ref="x", operator=">", value=0.0),
            direction="BUY", amplitude_bars=6, tp_pct=0.02, sl_pct=0.01,
            universe={"asset": "BTCUSD", "timeframe": "1h", "horizon": "6h"},
            metrics=template_metrics, scope="asset",
        )
        oe = build_or_einher(cands[0], template)
        assert oe.condition_tree.op == "OR"
        assert oe.direction == "BUY"
        assert oe.amplitude_bars == 6
        assert oe.source["model"] == "or_regimes"


class TestVetoNot:
    def _flat_ohlcv(self, n: int, close: np.ndarray) -> pl.DataFrame:
        return pl.DataFrame({
            "timestamp": pl.int_range(0, n, eager=True).cast(pl.Datetime("us")),
            "open": close, "high": close * 1.001, "low": close * 0.999,
            "close": close, "volume": [1000.0] * n,
        })

    def _mk_einher(self) -> Einher:
        ast = Condition(feature_ref="signal", operator=">", value=0.5)
        m = EinherMetrics(
            n_trades=200, n_tp=60, n_sl=80, n_timeout=60, win_rate=0.3,
            avg_net_return=-0.0002, total_return=-0.02, sharpe_ratio=0.05,
            max_drawdown=-0.06, profit_factor=0.85, avg_holding_bars=3,
            buy_hold_return=0.0, alpha=0.0, t_statistic=0.5, p_value=0.3,
        )
        return Einher(
            id=f"b_{np.random.default_rng().integers(1e6)}",
            condition_tree=ast, direction="BUY", amplitude_bars=6,
            tp_pct=0.02, sl_pct=0.01,
            universe={"asset": "BTCUSD", "timeframe": "1h"},
            metrics=m, scope="asset",
        )

    def test_returns_none_on_tiny_val(self) -> None:
        from einherjar.research.xgb_einhers.backtester import backtest_einher

        base = self._mk_einher()
        ohlcv = self._flat_ohlcv(50, np.full(50, 100.0))
        res = find_veto_condition(
            base, ohlcv[:50], np.zeros((50, 2), dtype=np.float32),
            ["a", "b"], 0.001, backtest_einher,
        )
        assert res is None

    def test_veto_respects_max_removed(self) -> None:
        """Un veto qui retirerait >20% des trades est refusé."""
        from einherjar.research.xgb_einhers.backtester import backtest_einher

        rng = np.random.default_rng(7)
        n_val = 8000
        close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.001, n_val)))
        ohlcv = self._flat_ohlcv(n_val, close)
        Xv = np.zeros((n_val, 2), dtype=np.float32)
        sig = rng.random(n_val) > 0.97
        Xv[sig, 0] = 1.0
        guard = rng.random(n_val).astype(np.float32)
        Xv[:, 1] = guard
        base = self._mk_einher()
        res = find_veto_condition(base, ohlcv, Xv, ["signal", "guard"], 0.001, backtest_einher)
        if res is not None:
            _, info = res
            assert info["removed_frac"] <= 0.20
            assert info["sharpe_after"] > info["sharpe_before"]

    def test_veto_ast_structure_when_found(self) -> None:
        from einherjar.research.xgb_einhers.backtester import backtest_einher
        from einherjar.research.xgb_einhers.types import ConditionNode

        rng = np.random.default_rng(7)
        n_val = 8000
        close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.001, n_val)))
        ohlcv = self._flat_ohlcv(n_val, close)
        Xv = np.zeros((n_val, 2), dtype=np.float32)
        sig = rng.random(n_val) > 0.97
        Xv[sig, 0] = 1.0
        Xv[:, 1] = rng.random(n_val).astype(np.float32)
        base = self._mk_einher()
        res = find_veto_condition(base, ohlcv, Xv, ["signal", "guard"], 0.001, backtest_einher)
        if res is not None:
            cand, info = res
            ast = cand.condition_tree
            assert isinstance(ast, ConditionNode) and ast.op == "AND"
            right = ast.right
            assert isinstance(right, ConditionNode) and right.op == "NOT"
            assert cand.source.get("veto_base_id") == base.id
