"""evaluator.py — Pont expression STGP ↔ Condition/ConditionNode xgb_einhers.

Le backtester (backtest_einher) évalue la condition d'un einher via
`evaluate_ast_on_array(condition_tree, X, feature_names)` ; ce module :
- détecte les AST contenant des atomes `expr` (FIX SE-01 dans condition_tree.py),
- les évalue vectorisé (numpy, O(N)),
- convertit une BoolExpr STGP en Condition/ConditionNode sérialisable
  (convention : atome expr → Condition(feature_ref="", expr=NumExpr)).

Le chemin xgb classique (Condition sans expr) est inchangé.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from einherjar.research.search_engine.expression import (
    BoolOp,
    Cmp,
    eval_bool,
    eval_num,
)
from einherjar.research.xgb_einhers.types import Condition, ConditionNode


def has_expr_atoms(ast: Condition | ConditionNode) -> bool:
    """True si l'AST contient au moins un atome avec expression arithmétique."""
    if isinstance(ast, Condition):
        return ast.expr is not None
    if has_expr_atoms(ast.left):
        return True
    if ast.right is not None:
        return has_expr_atoms(ast.right)
    return False


def eval_condition_ast(
    ast: Condition | ConditionNode,
    X: np.ndarray,
    feature_names: list[str],
) -> np.ndarray:
    """Évalue l'AST (avec ou sans atomes expr) sur X → (N,) bool vectorisé."""
    name_to_idx = {n: i for i, n in enumerate(feature_names)}
    return _eval_ast(ast, X, name_to_idx)


def _eval_ast(ast: Condition | ConditionNode, X: np.ndarray, name_to_idx: dict[str, int]) -> np.ndarray:
    if isinstance(ast, Condition):
        if ast.expr is not None:
            v = eval_num(ast.expr, X, name_to_idx)
            mask = _cmp(ast.operator, v, ast.value)
            return np.where(np.isnan(v), False, mask)
        # chemin classique xgb (feature seule)
        col = X[:, name_to_idx[ast.feature_ref]]
        mask = _cmp(ast.operator, col, ast.value)
        return np.where(np.isnan(col), False, mask)
    left = _eval_ast(ast.left, X, name_to_idx)
    if ast.op == "NOT":
        return ~left
    if ast.right is None:
        raise ValueError(f"Opérateur {ast.op} requiert un nœud right")
    right = _eval_ast(ast.right, X, name_to_idx)
    if ast.op == "AND":
        return left & right
    if ast.op == "OR":
        return left | right
    if ast.op == "XOR":
        return left ^ right
    raise ValueError(f"Opérateur logique non supporté : {ast.op}")


def _cmp(op: str, a: np.ndarray, b: object) -> np.ndarray:
    if op == "<":
        return a < b
    if op == "<=":
        return a <= b
    if op == ">":
        return a > b
    if op == ">=":
        return a >= b
    if op == "==":
        return np.abs(a - b) < 1e-9
    if op == "!=":
        return np.abs(a - b) >= 1e-9
    raise ValueError(f"Opérateur de comparaison non supporté : {op}")


def to_condition_tree(expr: object) -> Condition | ConditionNode:
    """Convertit une BoolExpr STGP en Condition/ConditionNode (sérialisable)."""
    if isinstance(expr, Cmp):
        return Condition(feature_ref="", operator=expr.operator, value=expr.value, expr=expr.expr)
    if isinstance(expr, BoolOp):
        if expr.op == "NOT":
            return ConditionNode(op="NOT", left=to_condition_tree(expr.left))
        return ConditionNode(
            op=expr.op,
            left=to_condition_tree(expr.left),
            right=to_condition_tree(expr.right),
        )
    raise TypeError(f"BoolExpr attendue, reçu : {type(expr).__name__}")


def _from_num_tree(node: object) -> object:
    """NumExpr STGP (dans Condition.expr) → NumExpr STGP neuf (roundtrip)."""
    from einherjar.research.search_engine.expression import BinNum, Const, Feature

    if isinstance(node, Feature):
        return Feature(node.feature_ref)
    if isinstance(node, Const):
        return Const(node.value)
    if isinstance(node, BinNum):
        return BinNum(
            op=node.op,
            left=_from_num_tree(node.left),
            right=_from_num_tree(node.right),
        )
    raise TypeError(f"NumExpr inconnue : {type(node).__name__}")


def from_condition_tree(ast: Condition | ConditionNode) -> object:
    """Inverse de to_condition_tree : Condition/ConditionNode → BoolExpr STGP."""
    from einherjar.research.search_engine.expression import BoolOp, Cmp, Feature

    if isinstance(ast, Condition):
        if ast.expr is not None:
            return Cmp(expr=_from_num_tree(ast.expr), operator=ast.operator, value=ast.value)
        return Cmp(expr=Feature(ast.feature_ref), operator=ast.operator, value=ast.value)
    if isinstance(ast, ConditionNode):
        if ast.op == "NOT":
            return BoolOp(op="NOT", left=from_condition_tree(ast.left))
        return BoolOp(
            op=ast.op,
            left=from_condition_tree(ast.left),
            right=from_condition_tree(ast.right),
        )
    raise TypeError(f"Condition attendue, reçu : {type(ast).__name__}")


def collect_tree_features(ast: object) -> set[str]:
    """Features référencées dans un Condition/ConditionNode (incl. atomes expr)."""
    from einherjar.research.search_engine.expression import collect_features

    if isinstance(ast, Condition):
        if ast.expr is not None:
            return set(collect_features(ast.expr))
        return {ast.feature_ref} if ast.feature_ref else set()
    if isinstance(ast, ConditionNode):
        out = collect_tree_features(ast.left)
        if ast.right is not None:
            out |= collect_tree_features(ast.right)
        return out
    return set()

    if isinstance(ast, Condition):
        if ast.expr is not None:
            return Cmp(expr=ast.expr, operator=ast.operator, value=ast.value)
        from einherjar.research.search_engine.expression import Feature

        return Cmp(expr=Feature(ast.feature_ref), operator=ast.operator, value=ast.value)
    if ast.op == "NOT":
        return BoolOp(op="NOT", left=from_condition_tree(ast.left))
    return BoolOp(op=ast.op, left=from_condition_tree(ast.left), right=from_condition_tree(ast.right))