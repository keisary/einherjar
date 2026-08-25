"""Tests du générateur STGP (generator.py + threshold_pool.py)."""
from __future__ import annotations

import numpy as np
import pytest

from einherjar.research.search_engine.expression import (
    BoolOp,
    Cmp,
    depth,
    eval_bool,
    render,
    size,
)
from einherjar.research.search_engine.generator import (
    crossover,
    generate_population,
    generate_random_bool_expr,
    generate_random_num_expr,
    mutate,
)
from einherjar.research.search_engine.space import SpaceConfig
from einherjar.research.search_engine.threshold_pool import ThresholdPool


@pytest.fixture()
def cfg() -> SpaceConfig:
    return SpaceConfig(
        data_version="test",
        feature_names=("mom", "vol", "rsi", "volu", "skew", "kurt"),
    )


@pytest.fixture()
def pool(cfg: SpaceConfig) -> ThresholdPool:
    rng = np.random.default_rng(1)
    X = rng.normal(size=(500, len(cfg.feature_names))).astype(np.float32)
    return ThresholdPool.build(X, list(cfg.feature_names), cfg, rng)


def _names(cfg: SpaceConfig) -> list[str]:
    return list(cfg.feature_names)


class TestGeneration:
    def test_deterministic(self, cfg: SpaceConfig, pool: ThresholdPool) -> None:
        a = generate_random_bool_expr(np.random.default_rng(7), cfg, pool)
        b = generate_random_bool_expr(np.random.default_rng(7), cfg, pool)
        assert render(a) == render(b)

    def test_depth_bounds(self, cfg: SpaceConfig, pool: ThresholdPool) -> None:
        rng = np.random.default_rng(3)
        for _ in range(100):
            e = generate_random_bool_expr(rng, cfg, pool)
            assert 1 <= depth(e) <= cfg.max_depth

    def test_xor_coverage(self, cfg: SpaceConfig, pool: ThresholdPool) -> None:
        """Le langage inclut XOR (décision Jovanny) : il doit apparaître."""
        rng = np.random.default_rng(42)
        exprs = [generate_random_bool_expr(rng, cfg, pool) for _ in range(400)]

        def has_xor(e: object) -> bool:
            if isinstance(e, BoolOp) and e.op == "XOR":
                return True
            if isinstance(e, BoolOp):
                return has_xor(e.left) or (e.right is not None and has_xor(e.right))
            return False

        assert any(has_xor(e) for e in exprs)

    def test_population_ramped(self, cfg: SpaceConfig, pool: ThresholdPool) -> None:
        pop = generate_population(np.random.default_rng(5), cfg, pool, 20)
        assert len(pop) == 20
        depths = {depth(e) for e in pop}
        assert len(depths) >= 3  # plusieurs profondeurs (ramped half-and-half)

    def test_threshold_from_train_pool(self, cfg: SpaceConfig, pool: ThresholdPool) -> None:
        """Les seuils des atomes Feature sont dans les bornes train de la feature."""
        rng = np.random.default_rng(9)
        for _ in range(50):
            e = generate_random_bool_expr(rng, cfg, pool)

            def check(e: object) -> None:
                if isinstance(e, Cmp):
                    from einherjar.research.search_engine.expression import Feature

                    if isinstance(e.expr, Feature):
                        lo = pool.per_feature[e.expr.feature_ref][0]
                        hi = pool.per_feature[e.expr.feature_ref][-1]
                        assert lo <= e.value <= hi
                elif isinstance(e, BoolOp):
                    check(e.left)
                    if e.right is not None:
                        check(e.right)

            check(e)


class TestGeneticOperators:
    def test_crossover_typed_and_bounded(self, cfg: SpaceConfig, pool: ThresholdPool) -> None:
        rng = np.random.default_rng(11)
        pop = generate_population(rng, cfg, pool, 40)
        for _ in range(100):
            a, b = pop[int(rng.integers(40))], pop[int(rng.integers(40))]
            c = crossover(a, b, rng, cfg, pool)
            assert depth(c) <= cfg.max_depth
            assert size(c) <= cfg.max_size
            assert isinstance(c, Cmp | BoolOp)

    def test_mutation_keeps_bounds(self, cfg: SpaceConfig, pool: ThresholdPool) -> None:
        rng = np.random.default_rng(13)
        pop = generate_population(rng, cfg, pool, 40)
        for _ in range(100):
            a = pop[int(rng.integers(40))]
            b = mutate(a, rng, cfg, pool)
            assert depth(b) <= cfg.max_depth
            assert size(b) <= cfg.max_size
            assert isinstance(b, Cmp | BoolOp)

    def test_mutation_changes_sometimes(self, cfg: SpaceConfig, pool: ThresholdPool) -> None:
        rng = np.random.default_rng(17)
        changed = 0
        for _ in range(200):
            e = generate_random_bool_expr(rng, cfg, pool)
            m = mutate(e, rng, cfg, pool)
            if render(m) != render(e):
                changed += 1
        assert changed > 20  # la mutation doit réellement muter

    def test_num_expr_evals(self, cfg: SpaceConfig, pool: ThresholdPool) -> None:
        rng = np.random.default_rng(19)
        X = np.random.default_rng(0).normal(size=(100, 6)).astype(np.float32)
        names = _names(cfg)
        for _ in range(20):
            e = generate_random_num_expr(rng, cfg, max_depth=3)
            v = eval_bool(Cmp(expr=e, operator=">", value=0.0), X, {n: i for i, n in enumerate(names)})
            assert v.shape == (100,) and v.dtype == bool
