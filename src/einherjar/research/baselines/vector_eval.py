"""vector_eval.py — Évaluation vectorisée des AST de conditions.

Le `evaluate_ast_on_array` existant (condition_tree.py:131) construit un dict
{feature: valeur} par ligne : O(N*F) en Python pur, inutilisable pour des
centaines de backtests de baseline. Cette version évalue les `Condition` et
`ConditionNode` directement sur les colonnes numpy.

Convention NaN -> False (conservateur, cohérente avec condition_tree.py:103).
"""
from __future__ import annotations

import numpy as np

from einherjar.research.xgb_einhers.types import Condition, ConditionNode

_OPS = {
    "<": np.less,
    "<=": np.less_equal,
    ">": np.greater,
    ">=": np.greater_equal,
    "==": np.equal,
    "!=": np.not_equal,
}


def eval_cond_ast(
    ast: Condition | ConditionNode,
    X: np.ndarray,
    feature_names: list[str],
) -> np.ndarray:
    """Évalue l'AST sur toute la matrice X.

    Args:
        ast : Condition ou ConditionNode (AND/OR/NOT/XOR).
        X : (N, F) features.
        feature_names : noms des colonnes de X.

    Returns:
        mask : (N,) bool, True aux indices où la condition est vraie.
    """
    name_to_idx = {n: i for i, n in enumerate(feature_names)}
    return _eval(ast, X, name_to_idx)


def _eval(
    ast: Condition | ConditionNode,
    X: np.ndarray,
    name_to_idx: dict[str, int],
) -> np.ndarray:
    """Évaluation récursive vectorisée."""
    if isinstance(ast, Condition):
        col = X[:, name_to_idx[ast.feature_ref]]
        mask = _OPS[ast.operator](col, ast.value)
        # NaN -> False : np.where garde le dtype bool
        return np.where(np.isnan(col), False, mask)

    left = _eval(ast.left, X, name_to_idx)
    if ast.op == "NOT":
        return ~left
    if ast.right is None:
        raise ValueError(f"Opérateur {ast.op} requiert un nœud right")
    right = _eval(ast.right, X, name_to_idx)
    if ast.op == "AND":
        return left & right
    if ast.op == "OR":
        return left | right
    if ast.op == "XOR":
        return left ^ right
    raise ValueError(f"Opérateur logique non supporté : {ast.op}")