"""Tests des baselines anti-hasard (Option B du plan)."""
from __future__ import annotations

import numpy as np
import pytest

from einherjar.research.baselines.random_gen import (
    MAX_TRIGGER_RATE,
    MIN_AND_TRIGGER_RATE,
    MIN_TRIGGER_RATE,
    generate_random_einhers,
)
from einherjar.research.baselines.runner import condition_str, n_leaves, parse_horizon
from einherjar.research.baselines.vector_eval import eval_cond_ast
from einherjar.research.xgb_einhers.types import Condition, ConditionNode


class TestParseHorizon:
    """parse_horizon : conversions de texte en nombre de bars."""

    def test_6h(self) -> None:
        assert parse_horizon("6h") == 6

    def test_2d(self) -> None:
        assert parse_horizon("2d") == 48

    def test_invalid(self) -> None:
        with pytest.raises(ValueError):
            parse_horizon("nope")


class TestVectorEval:
    """Évaluateur vectorisé : doit égaler les comparaisons numpy directes."""

    @staticmethod
    def _x() -> np.ndarray:
        return np.random.default_rng(0).normal(size=(50, 3))

    def test_atomic_matches_numpy(self) -> None:
        X = self._x()
        names = ["a", "b", "c"]
        for op, expected in (
            ("<", np.less),
            ("<=", np.less_equal),
            (">", np.greater),
            (">=", np.greater_equal),
        ):
            ast = Condition(feature_ref="b", operator=op, value=0.0)
            got = eval_cond_ast(ast, X, names)
            assert np.array_equal(got, expected(X[:, 1], 0.0))

    def test_and_tree(self) -> None:
        X = self._x()
        names = ["a", "b", "c"]
        ast = ConditionNode(
            op="AND",
            left=Condition(feature_ref="a", operator=">", value=0.0),
            right=Condition(feature_ref="b", operator=">", value=0.0),
        )
        got = eval_cond_ast(ast, X, names)
        assert np.array_equal(got, (X[:, 0] > 0) & (X[:, 1] > 0))

    def test_nan_is_false(self) -> None:
        X = np.array([[np.nan], [1.0]])
        ast = Condition(feature_ref="a", operator=">", value=0.0)
        assert np.array_equal(eval_cond_ast(ast, X, ["a"]), [False, True])


class TestRandomGen:
    """Génération aléatoire : déterministe, bornes respectées, pas de lookahead."""

    @staticmethod
    def _train() -> tuple[np.ndarray, list[str]]:
        X = np.random.default_rng(1).normal(size=(500, 5))
        return X, [f"f{i}" for i in range(5)]

    def test_deterministic(self) -> None:
        X, names = self._train()
        a1 = generate_random_einhers(
            np.random.default_rng(7), 5, "BTCUSD", "crypto", "1h", "2d", 48, names, X,
        )
        a2 = generate_random_einhers(
            np.random.default_rng(7), 5, "BTCUSD", "crypto", "1h", "2d", 48, names, X,
        )
        assert [e.id for e in a1] == [e.id for e in a2]
        assert [condition_str(e.condition_tree) for e in a1] == [
            condition_str(e.condition_tree) for e in a2
        ]

    def test_directions_in_pool(self) -> None:
        X, names = self._train()
        einhers = generate_random_einhers(
            np.random.default_rng(3), 20, "BTCUSD", "crypto", "1h", "2d", 48, names, X,
        )
        assert {e.direction for e in einhers} <= {"BUY", "SELL"}

    def test_thresholds_within_train_bounds(self) -> None:
        X, names = self._train()
        einhers = generate_random_einhers(
            np.random.default_rng(5), 10, "BTCUSD", "crypto", "1h", "2d", 48, names, X,
        )
        for e in einhers:
            for c in _collect_leaves(e.condition_tree):
                col = X[:, names.index(c.feature_ref)]
                assert col.min() <= c.value <= col.max()

    def test_trigger_rate_bounds(self) -> None:
        X, names = self._train()
        einhers = generate_random_einhers(
            np.random.default_rng(11), 30, "BTCUSD", "crypto", "1h", "2d", 48, names, X,
        )
        for e in einhers:
            rate = float(eval_cond_ast(e.condition_tree, X, names).mean())
            # Chaque feuille est bornée [MIN_TRIGGER_RATE, MAX_TRIGGER_RATE] ;
            # le AND combiné est >= MIN_AND_TRIGGER_RATE et <= MAX_TRIGGER_RATE
            assert MIN_AND_TRIGGER_RATE <= rate <= MAX_TRIGGER_RATE

    def test_n_leaves(self) -> None:
        ast = ConditionNode(
            op="AND",
            left=Condition(feature_ref="a", operator=">", value=0.0),
            right=ConditionNode(
                op="AND",
                left=Condition(feature_ref="b", operator="<", value=1.0),
                right=Condition(feature_ref="c", operator=">=", value=2.0),
            ),
        )
        assert n_leaves(ast) == 3


def _collect_leaves(ast: Condition | ConditionNode) -> list[Condition]:
    """Feuilles (Conditions) d'un AST."""
    if isinstance(ast, Condition):
        return [ast]
    leaves = _collect_leaves(ast.left)
    if ast.right is not None:
        leaves += _collect_leaves(ast.right)
    return leaves