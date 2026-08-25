# pyright: reportAttributeAccessIssue=false, reportReturnType=false, reportArgumentType=false, reportAssignmentType=false
"""expression.py — Langage STGP fortement typé des Einhers.

Deux types d'expressions (STGP = Strongly-Typed GP, plan A1) :
- NumExpr  : expression numérique sur les features
             (Feature | Const | BinNum {+, -, *, /, min, max}).
- BoolExpr : expression booléenne — l'atome de déclenchement d'un einher
             (Cmp : NumExpr OP seuil) combiné par BoolOp
             {AND, OR, XOR, NOT}.

Décision de langage (Jovanny, 2026-08-20) : XOR inclus dans les opérateurs
booléens. Bornes : max_depth 6 (plan ligne 318-321), seuils tirés d'un pool
de quantiles empiriques calculé sur la fenêtre TRAIN (threshold_pool.py).

Évaluation vectorisée numpy (O(N)) : indispensable pour la fitness cheap
sur des centaines de milliers de lignes × milliers de candidats.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# --------------------------------------------------------------------------- #
# Types numériques
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Const:
    """Constante numérique."""

    value: float

    def to_dict(self) -> dict[str, float]:
        """to_dict."""
        return {"kind": "const", "value": self.value}


@dataclass(frozen=True)
class Feature:
    """Référence à une feature du pool (borné par SpaceConfig)."""

    feature_ref: str

    def to_dict(self) -> dict[str, str]:
        """to_dict."""
        return {"kind": "feature", "feature_ref": self.feature_ref}


@dataclass(frozen=True)
class BinNum:
    """Opération numérique binaire : + - * / min max."""

    op: str
    left: object  # NumExpr
    right: object  # NumExpr

    def to_dict(self) -> dict[str, object]:
        """to_dict."""
        return {"kind": "binnum", "op": self.op, "left": self.left.to_dict(), "right": self.right.to_dict()}


# --------------------------------------------------------------------------- #
# Types booléens
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Cmp:
    """Atome : comparaison (NumExpr OP seuil). OP ∈ {<, <=, >, >=}."""

    expr: object  # NumExpr
    operator: str
    value: float

    def to_dict(self) -> dict[str, object]:
        """to_dict."""
        return {"kind": "cmp", "expr": self.expr.to_dict(), "operator": self.operator, "value": self.value}


@dataclass(frozen=True)
class BoolOp:
    """Opération booléenne : AND | OR | XOR (binaire), NOT (unaire)."""

    op: str
    left: object  # BoolExpr
    right: object | None = None  # None pour NOT

    def to_dict(self) -> dict[str, object]:
        """to_dict."""
        d: dict[str, object] = {"kind": "boolop", "op": self.op, "left": self.left.to_dict()}
        if self.right is not None:
            d["right"] = self.right.to_dict()
        return d


# --------------------------------------------------------------------------- #
# Évaluation vectorisée
# --------------------------------------------------------------------------- #

_NUM_OPS = {
    "+": np.add,
    "-": np.subtract,
    "*": np.multiply,
    "min": np.minimum,
    "max": np.maximum,
}

_CMP_OPS = {
    "<": np.less,
    "<=": np.less_equal,
    ">": np.greater,
    ">=": np.greater_equal,
    "==": np.equal,
    "!=": np.not_equal,
}


def eval_num(
    expr: object,
    X: np.ndarray,
    name_to_idx: dict[str, int],
) -> np.ndarray:
    """Évalue une NumExpr sur X → (N,) float64. NaN propagé (borné par les Cmp)."""
    if isinstance(expr, Const):
        return np.full(X.shape[0], expr.value, dtype=np.float64)
    if isinstance(expr, Feature):
        return X[:, name_to_idx[expr.feature_ref]].astype(np.float64)
    if isinstance(expr, BinNum):
        a = eval_num(expr.left, X, name_to_idx)
        b = eval_num(expr.right, X, name_to_idx)
        if expr.op == "/":
            # division protégée : a / b avec b≈0 -> 0 (évite inf/NaN dans les arbres)
            with np.errstate(divide="ignore", invalid="ignore"):
                c = a / b
            return np.where(np.isfinite(c), c, 0.0)
        return _NUM_OPS[expr.op](a, b)
    raise TypeError(f"NumExpr inconnue : {type(expr).__name__}")


def eval_bool(
    expr: object,
    X: np.ndarray,
    name_to_idx: dict[str, int],
) -> np.ndarray:
    """Évalue une BoolExpr sur X → (N,) bool. NaN -> False (convention ONTOLOGY S-1)."""
    if isinstance(expr, Cmp):
        v = eval_num(expr.expr, X, name_to_idx)
        mask = _CMP_OPS[expr.operator](v, expr.value)
        return np.where(np.isnan(v), False, mask)
    if isinstance(expr, BoolOp):
        if expr.op == "NOT":
            return ~eval_bool(expr.left, X, name_to_idx)
        left = eval_bool(expr.left, X, name_to_idx)
        right = eval_bool(expr.right, X, name_to_idx)
        if expr.op == "AND":
            return left & right
        if expr.op == "OR":
            return left | right
        if expr.op == "XOR":
            return left ^ right
        raise ValueError(f"Opérateur booléen non supporté : {expr.op}")
    raise TypeError(f"BoolExpr inconnue : {type(expr).__name__}")


# --------------------------------------------------------------------------- #
# Métriques d'arbre + rendu lisible
# --------------------------------------------------------------------------- #


def depth(expr: object) -> int:
    """Profondeur max de l'arbre (feuille = 0)."""
    if isinstance(expr, Const | Feature):
        return 0
    if isinstance(expr, Cmp):
        return 1 + depth(expr.expr)
    if isinstance(expr, BinNum):
        return 1 + max(depth(expr.left), depth(expr.right))
    if isinstance(expr, BoolOp):
        if expr.op == "NOT":
            return 1 + depth(expr.left)
        return 1 + max(depth(expr.left), depth(expr.right))
    raise TypeError(f"Type inconnu : {type(expr).__name__}")


def size(expr: object) -> int:
    """Nombre de nœuds de l'arbre (anti-bloat)."""
    if isinstance(expr, Const | Feature):
        return 1
    if isinstance(expr, Cmp):
        return 1 + size(expr.expr)
    if isinstance(expr, BinNum):
        return 1 + size(expr.left) + size(expr.right)
    if isinstance(expr, BoolOp):
        if expr.op == "NOT":
            return 1 + size(expr.left)
        return 1 + size(expr.left) + size(expr.right)
    raise TypeError(f"Type inconnu : {type(expr).__name__}")


def collect_features(expr: object, acc: list[str] | None = None) -> list[str]:
    """Toutes les features référencées (ordre de parcours)."""
    if acc is None:
        acc = []
    if isinstance(expr, Feature):
        acc.append(expr.feature_ref)
    elif isinstance(expr, BinNum | Cmp):
        collect_features(expr.left if hasattr(expr, "left") else expr.expr, acc)
        if isinstance(expr, BinNum):
            collect_features(expr.right, acc)
    elif isinstance(expr, BoolOp):
        collect_features(expr.left, acc)
        if expr.right is not None:
            collect_features(expr.right, acc)
    return acc


def render(expr: object) -> str:
    """Rendu lisible de l'expression (pour corpus/descripteurs/débogage)."""
    if isinstance(expr, Const):
        return f"{expr.value:g}"
    if isinstance(expr, Feature):
        return expr.feature_ref
    if isinstance(expr, BinNum):
        return f"({render(expr.left)} {expr.op} {render(expr.right)})"
    if isinstance(expr, Cmp):
        return f"({render(expr.expr)} {expr.operator} {expr.value:g})"
    if isinstance(expr, BoolOp):
        if expr.op == "NOT":
            return f"NOT({render(expr.left)})"
        return f"({render(expr.left)} {expr.op} {render(expr.right)})"
    raise TypeError(f"Type inconnu : {type(expr).__name__}")
