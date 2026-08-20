"""Tests des modules recherche (fitness, descripteurs, MAP-Elites)."""
from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from einherjar.research.search_engine.descriptors import describe, dominant_family, market_regime
from einherjar.research.search_engine.fitness import cheap_fitness
from einherjar.research.search_engine.generator import generate_random_bool_expr
from einherjar.research.search_engine.map_elites import CellEntry, MapElitesArchive, run_map_elites
from einherjar.research.search_engine.space import SpaceConfig
from einherjar.research.search_engine.threshold_pool import ThresholdPool

TAXO = {
    "mom": {"economic_family": "momentum"},
    "vol": {"economic_family": "volatility"},
    "volu": {"economic_family": "volume_flow"},
    "rsi": {"economic_family": "momentum"},
    "skew": {"economic_family": "statistical"},
    "kurt": {"economic_family": "statistical"},
}


def _ohlcv(n: int = 2000, seed: int = 0, vol: float = 0.002) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(loc=0.0002, scale=vol, size=n)
    close = 100.0 * np.cumprod(1 + rets)
    open_ = np.concatenate([[100.0], close[:-1]]) * (1 + rng.normal(0, 1e-4, n))
    high = np.maximum(open_, close) * 1.001
    low = np.minimum(open_, close) * 0.999
    ts = [
        dt.datetime(2020, 1, 1) + dt.timedelta(hours=k) for k in range(n)
    ]
    return pl.DataFrame({
        "timestamp": ts,
        "open": open_, "high": high, "low": low, "close": close,
        "volume": rng.uniform(100, 1000, n),
    })


@pytest.fixture()
def env() -> dict:
    n = 1500
    cfg = SpaceConfig(data_version="test", feature_names=("mom", "vol", "volu", "rsi", "skew", "kurt"))
    pool = ThresholdPool.build(
        np.random.default_rng(1).normal(size=(500, 6)).astype(np.float32),
        list(cfg.feature_names), cfg, np.random.default_rng(1),
    )
    X = np.random.default_rng(2).normal(size=(n, 6)).astype(np.float32)
    return {
        "cfg": cfg, "pool": pool,
        "ohlcv_df": _ohlcv(n, seed=3),
        "X": X, "feature_names": list(cfg.feature_names),
        "universe": {"asset": "TEST", "asset_class": "crypto", "timeframe": "1h", "horizon": "1d", "horizon_bars": 24},
    }


class TestFitness:
    def test_deterministic_and_finite(self, env: dict) -> None:
        cfg, pool = env["cfg"], env["pool"]
        expr = generate_random_bool_expr(np.random.default_rng(7), cfg, pool)
        s1, _, _ = cheap_fitness(expr, "BUY", 24, env["universe"], env["ohlcv_df"], env["X"],
                                 env["feature_names"], np.random.default_rng(8), costs_pct=0.0014)
        s2, _, _ = cheap_fitness(expr, "BUY", 24, env["universe"], env["ohlcv_df"], env["X"],
                                 env["feature_names"], np.random.default_rng(8), costs_pct=0.0014)
        assert s1 == s2
        assert np.isfinite(s1)

    def test_different_seed_different_sample(self, env: dict) -> None:
        cfg, pool = env["cfg"], env["pool"]
        expr = generate_random_bool_expr(np.random.default_rng(9), cfg, pool)
        a, _, sub_a = cheap_fitness(expr, "SELL", 24, env["universe"], env["ohlcv_df"], env["X"],
                                    env["feature_names"], np.random.default_rng(10), costs_pct=0.0014)
        b, _, sub_b = cheap_fitness(expr, "SELL", 24, env["universe"], env["ohlcv_df"], env["X"],
                                    env["feature_names"], np.random.default_rng(11), costs_pct=0.0014)
        assert len(sub_a) == len(sub_b) == int(1500 * 0.5)
        assert a != b or sub_a["timestamp"][0] != sub_b["timestamp"][0]


class TestDescriptors:
    def test_dominant_family(self) -> None:
        from einherjar.research.search_engine.expression import BoolOp, Cmp, Feature

        e = BoolOp(
            op="AND",
            left=Cmp(expr=Feature("mom"), operator=">", value=0.0),
            right=Cmp(expr=Feature("rsi"), operator="<", value=0.5),
        )
        assert dominant_family(e, TAXO) == "momentum"

    def test_dominant_family_vol(self) -> None:
        from einherjar.research.search_engine.expression import BoolOp, Cmp, Feature

        e = BoolOp(
            op="OR",
            left=Cmp(expr=Feature("vol"), operator=">", value=0.0),
            right=Cmp(expr=Feature("volu"), operator="<", value=0.5),
        )
        assert dominant_family(e, TAXO) == "volatility"  # vol avant volume_flow (ordre trié)

    def test_regime_high_vs_low(self) -> None:
        assert market_regime(_ohlcv(500, vol=0.002)) == "low_vol"
        assert market_regime(_ohlcv(500, vol=0.05)) == "high_vol"

    def test_describe_tuple(self, env: dict) -> None:
        cfg, pool = env["cfg"], env["pool"]
        expr = generate_random_bool_expr(np.random.default_rng(12), cfg, pool)
        d = describe(expr, "BUY", env["ohlcv_df"].head(300), TAXO)
        assert d[0] == "BUY"
        assert d[1] in {"momentum", "volatility", "volume_flow", "statistical"}
        assert d[2] in {"low_vol", "high_vol"}


class TestMapElites:
    def test_archive_insert_semantics(self) -> None:
        a = MapElitesArchive()
        cell = ("BUY", "momentum", "low_vol")
        e1 = CellEntry(expr=None, einher=None, fitness=0.5, direction="BUY")
        e2 = CellEntry(expr=None, einher=None, fitness=1.2, direction="BUY")
        e3 = CellEntry(expr=None, einher=None, fitness=0.9, direction="BUY")
        assert a.insert(cell, e1) is True
        assert a.insert(cell, e2) is True
        assert a.insert(cell, e3) is False  # 0.9 < 1.2 → pas de remplacement
        assert a.cells[cell].fitness == 1.2

    def test_run_map_elites_smoke(self, env: dict) -> None:
        cfg, pool = env["cfg"], env["pool"]
        archive = run_map_elites(
            np.random.default_rng(42), cfg, pool, TAXO,
            {"ohlcv_df": env["ohlcv_df"], "X": env["X"], "feature_names": env["feature_names"]},
            costs_pct=0.0014, amplitude_bars=24, universe=env["universe"],
            n_pop=12, n_generations=3,
        )
        cells = archive.occupied_cells()
        assert len(cells) >= 1
        assert len(cells) <= archive.max_cells
        best = archive.best()
        assert best is not None and np.isfinite(best.fitness)

    def test_deterministic_run(self, env: dict) -> None:
        cfg, pool = env["cfg"], env["pool"]
        kw = dict(
            cfg=cfg, pool=pool, taxonomy=TAXO,
            data={"ohlcv_df": env["ohlcv_df"], "X": env["X"], "feature_names": env["feature_names"]},
            costs_pct=0.0014, amplitude_bars=24, universe=env["universe"],
            n_pop=10, n_generations=2,
        )
        a1 = run_map_elites(np.random.default_rng(5), **kw)
        a2 = run_map_elites(np.random.default_rng(5), **kw)
        assert a1.best() is not None and a1.best().fitness == a2.best().fitness
        assert sorted(a1.cells) == sorted(a2.cells)