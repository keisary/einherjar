"""Tests du langage STGP (expression.py + evaluator.py)."""
from __future__ import annotations

import numpy as np

from einherjar.research.search_engine.evaluator import (
    from_condition_tree,
    has_expr_atoms,
    to_condition_tree,
)
from einherjar.research.search_engine.expression import (
    BinNum,
    BoolOp,
    Cmp,
    Const,
    Feature,
    collect_features,
    depth,
    eval_bool,
    eval_num,
    render,
    size,
)
from einherjar.research.xgb_einhers.condition_tree import evaluate_ast_on_array
from einherjar.research.xgb_einhers.types import Condition, ConditionNode


def _ctx() -> tuple[np.ndarray, dict[str, int]]:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4)).astype(np.float32)
    names = ["mom", "vol", "rsi", "volu"]
    return X, {n: i for i, n in enumerate(names)}


class TestNumExpr:
    def test_bin_ops_match_numpy(self) -> None:
        X, m = _ctx()
        for op in ("+", "-", "*", "min", "max"):
            e = BinNum(op=op, left=Feature("mom"), right=Feature("vol"))
            got = eval_num(e, X, m)
            a, b = X[:, m["mom"]], X[:, m["vol"]]
            if op == "+":
                exp = a + b
            elif op == "-":
                exp = a - b
            elif op == "*":
                exp = a * b
            elif op == "min":
                exp = np.minimum(a, b)
            else:
                exp = np.maximum(a, b)
            assert np.allclose(got, exp)

    def test_protected_div(self) -> None:
        X, m = _ctx()
        X[:, m["vol"]] = 0.0  # forcer des divisions par zéro
        e = BinNum(op="/", left=Feature("mom"), right=Feature("vol"))
        got = eval_num(e, X, m)
        assert np.all(np.isfinite(got))

    def test_const(self) -> None:
        X, m = _ctx()
        assert np.all(eval_num(Const(2.5), X, m) == 2.5)


class TestBoolExpr:
    def test_xor_truth_table(self) -> None:
        X, m = _ctx()
        X[:4, m["mom"]] = np.array([1.0, 1.0, -1.0, -1.0])
        X[:4, m["vol"]] = np.array([1.0, -1.0, 1.0, -1.0])
        e = BoolOp(
            op="XOR",
            left=Cmp(expr=Feature("mom"), operator=">", value=0.0),
            right=Cmp(expr=Feature("vol"), operator=">", value=0.0),
        )
        got = eval_bool(e, X[:4], m)
        assert got.tolist() == [False, True, True, False]

    def test_and_or_not(self) -> None:
        X, m = _ctx()
        X[:2, m["mom"]] = np.array([1.0, -1.0])
        X[:2, m["vol"]] = np.array([1.0, -1.0])
        a = Cmp(expr=Feature("mom"), operator=">", value=0.0)
        b = Cmp(expr=Feature("vol"), operator=">", value=0.0)
        assert eval_bool(BoolOp(op="AND", left=a, right=b), X[:2], m).tolist() == [True, False]
        assert eval_bool(BoolOp(op="OR", left=a, right=b), X[:2], m).tolist() == [True, False]
        assert eval_bool(BoolOp(op="NOT", left=a), X[:2], m).tolist() == [False, True]

    def test_nan_is_false(self) -> None:
        X, m = _ctx()
        X[:2, m["mom"]] = np.array([np.nan, 1.0])
        e = Cmp(expr=Feature("mom"), operator=">", value=0.0)
        assert eval_bool(e, X[:2], m).tolist() == [False, True]


class TestTreeMetrics:
    def test_depth_size(self) -> None:
        e = BoolOp(
            op="AND",
            left=Cmp(expr=BinNum(op="+", left=Feature("a"), right=Feature("b")), operator=">", value=0.0),
            right=Cmp(expr=Feature("c"), operator="<", value=1.0),
        )
        assert depth(e) == 3
        assert size(e) == 7

    def test_collect_features(self) -> None:
        e = BinNum(op="+", left=Feature("a"), right=Feature("b"))
        assert collect_features(e) == ["a", "b"]

    def test_render(self) -> None:
        e = Cmp(expr=BinNum(op="min", left=Feature("mom"), right=Feature("vol")), operator=">", value=0.5)
        assert "mom" in render(e) and "min" in render(e)


class TestEvaluatorBridge:
    def test_has_expr_atoms(self) -> None:
        plain = Condition(feature_ref="mom", operator=">", value=0.0)
        with_expr = Condition(feature_ref="", operator=">", value=0.0, expr=Feature("mom"))
        assert not has_expr_atoms(plain)
        assert has_expr_atoms(with_expr)
        assert has_expr_atoms(ConditionNode(op="AND", left=plain, right=with_expr))

    def test_to_from_condition_tree_roundtrip(self) -> None:
        e = BoolOp(
            op="XOR",
            left=Cmp(expr=Feature("mom"), operator=">", value=0.3),
            right=Cmp(expr=BinNum(op="*", left=Feature("vol"), right=Const(2.0)), operator="<", value=1.5),
        )
        ast = to_condition_tree(e)
        back = from_condition_tree(ast)
        X, m = _ctx()
        assert np.array_equal(eval_bool(e, X, m), eval_bool(back, X, m))
        assert has_expr_atoms(ast)

    def test_evaluate_ast_on_array_dispatch(self) -> None:
        """Le backtester (evaluate_ast_on_array) évalue les atomes expr via le pont."""
        e = BoolOp(
            op="XOR",
            left=Cmp(expr=Feature("mom"), operator=">", value=0.0),
            right=Cmp(expr=Feature("vol"), operator=">", value=0.0),
        )
        ast = to_condition_tree(e)
        X, m = _ctx()
        names = ["mom", "vol", "rsi", "volu"]
        got = evaluate_ast_on_array(ast, X, names)
        assert np.array_equal(got, eval_bool(e, X, m))

    def test_classic_condition_unchanged(self) -> None:
        """Sans expr : le chemin xgb classique fonctionne toujours."""
        ast = ConditionNode(
            op="AND",
            left=Condition(feature_ref="mom", operator=">", value=0.0),
            right=Condition(feature_ref="vol", operator="<", value=0.5),
        )
        X, m = _ctx()
        names = ["mom", "vol", "rsi", "volu"]
        got = evaluate_ast_on_array(ast, X, names)
        exp = (X[:, 0] > 0.0) & (X[:, 1] < 0.5)
        assert np.array_equal(got, exp)

    def test_condition_to_dict_with_expr(self) -> None:
        c = Condition(feature_ref="", operator=">", value=0.5, expr=Feature("mom"))
        d = c.to_dict()
        assert d["expr"]["kind"] == "feature"
        assert d["expr"]["feature_ref"] == "mom"
