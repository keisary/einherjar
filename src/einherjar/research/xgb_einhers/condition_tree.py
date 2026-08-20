"""condition_tree.py - Conversion XGBPath -> AST de conditions Einher.

Réponse Q12 (a ou c) : on commence AND-only (XGBoost est naturellement AND).
On garde une porte d'extension pour OR/NOT/XOR en V2 si besoin.

Représentation AST :
- Condition atomique : Condition(feature_ref, operator, value)
- Nœud composé : ConditionNode(op='AND', left, right=None pour NOT)
"""
from __future__ import annotations

import logging
from typing import Any

from einherjar.research.xgb_einhers.path_extractor import XGBPath
from einherjar.research.xgb_einhers.types import Condition, ConditionNode

logger = logging.getLogger(__name__)


# Mapping XGBoost op -> op Einher (string normalisée)
OP_MAP = {
    "<": "<",
    "<=": "<=",
    ">": ">",
    ">=": ">=",
    "==": "==",
    "!=": "!=",
}


def path_to_ast(path: XGBPath) -> Condition | ConditionNode:
    """Convertit un XGBPath en AST de conditions (AND-only par défaut).

    Args:
        path : XGBPath avec une liste de (feature, op, threshold)

    Returns:
        Condition si 1 seule condition, ConditionNode(AND) sinon.
    """
    if len(path.conditions) == 0:
        raise ValueError("Chemin vide : impossible de construire un AST")

    # Convertir chaque condition en Condition atomique
    conditions = [
        Condition(
            feature_ref=feat,
            operator=OP_MAP.get(op, op),
            value=value,
            transformation=None,
        )
        for feat, op, value in path.conditions
    ]

    if len(conditions) == 1:
        return conditions[0]

    # AND récursif
    # AND(left, AND(rest...)) pour préserver l'ordre
    result = conditions[0]
    for c in conditions[1:]:
        result = ConditionNode(op="AND", left=result, right=c)
    return result


def ast_to_dict(ast: Condition | ConditionNode) -> dict[str, Any]:
    """Sérialise un AST en dict JSON-compatible."""
    return ast.to_dict()


def evaluate_condition_on_value(
    ast: Condition | ConditionNode,
    features_at_t: dict[str, float | int | None],
) -> bool:
    """Évalue l'AST sur les valeurs des features au temps t.

    Args:
        ast : Condition ou ConditionNode
        features_at_t : dict {feature_name: value} au temps t

    Returns:
        True si la condition est satisfaite, False sinon.
        NaN/None -> False (conservateur, conforme à ONTOLOGY S-1).
    """
    if isinstance(ast, Condition):
        return _eval_atomic(ast, features_at_t)
    # ConditionNode
    left_result = evaluate_condition_on_value(ast.left, features_at_t)
    if ast.op == "NOT":
        return not left_result
    if ast.right is None:
        raise ValueError(f"Opérateur {ast.op} requiert un nœud right")
    right_result = evaluate_condition_on_value(ast.right, features_at_t)
    if ast.op == "AND":
        return left_result and right_result
    if ast.op == "OR":
        return left_result or right_result
    if ast.op == "XOR":
        return left_result != right_result
    raise ValueError(f"Opérateur logique non supporté : {ast.op}")


def _eval_atomic(c: Condition, features_at_t: dict[str, float | int | None]) -> bool:
    """Évalue une condition atomique sur une valeur scalaire."""
    v = features_at_t.get(c.feature_ref)
    if v is None:
        return False
    # NaN check
    try:
        if v != v:  # NaN check (NaN != NaN)
            return False
    except Exception:
        return False

    op = c.operator
    if op == "<":
        return v < c.value
    if op == "<=":
        return v <= c.value
    if op == ">":
        return v > c.value
    if op == ">=":
        return v >= c.value
    if op == "==":
        return abs(v - c.value) < 1e-9
    if op == "!=":
        return abs(v - c.value) >= 1e-9
    raise ValueError(f"Opérateur de comparaison non supporté : {op}")


def evaluate_ast_on_array(
    ast: Condition | ConditionNode,
    X: "np.ndarray",          # (N, F) float32
    feature_names: list[str],
) -> "np.ndarray":
    """Évalue l'AST sur toute une matrice X.

    Returns:
        mask : (N,) bool, True aux indices où la condition est vraie.
    """
    # FIX SE-01 (2026-08-20) : si l'AST contient des atomes expr (STGP),
    # basculer sur l'evaluateur vectorise de search_engine (les expressions
    # arithmetiques ne peuvent pas passer par le parcours scalaire par ligne).
    from einherjar.research.search_engine import evaluator  # import lazy (cycle)
    if evaluator.has_expr_atoms(ast):
        return evaluator.eval_condition_ast(ast, X, feature_names)
    import numpy as np
    name_to_idx = {n: i for i, n in enumerate(feature_names)}
    n = X.shape[0]
    mask = np.zeros(n, dtype=bool)
    for i in range(n):
        features_at_t = {name: X[i, name_to_idx[name]] for name in feature_names}
        if evaluate_condition_on_value(ast, features_at_t):
            mask[i] = True
    return mask
