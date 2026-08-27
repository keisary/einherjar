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

import numpy as np

from .path_extractor import XGBPath
from .types import Condition, ConditionNode

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
    """Convertit un XGBPath en AST de conditions.

    Problème 3 (2026-08-21) : gère les variantes logiques.
    - logical_op = 'OR'  : OR(path_to_ast(p1), path_to_ast(p2)) via sub_paths
    - logical_op = 'XOR' : XOR(left, right) sur les conditions
    - logical_op = 'NOT' : NOT(left) sur la condition atomique négativée
    - logical_op = 'AND' (défaut) : AND des conditions (comportement original)

    Args:
        path : XGBPath avec une liste de (feature, op, threshold)

    Returns:
        Condition si 1 seule condition, ConditionNode sinon.
    """
    if len(path.conditions) == 0:
        # Variantes OR : les conditions sont dans sub_paths
        if path.logical_op == "OR" and path.sub_paths:
            asts = [path_to_ast(sp) for sp in path.sub_paths]
            result = asts[0]
            for a in asts[1:]:
                result = ConditionNode(op="OR", left=result, right=a)
            return result
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

    # XOR : (left XOR right) sur les 2 premières conditions
    if path.logical_op == 'XOR' and len(conditions) >= 2:
        return ConditionNode(op='XOR', left=conditions[0], right=conditions[1])

    # NOT : NOT(c1) puis AND avec le reste des conditions (conservées)
    # Résultat : NOT(c1) AND c2 AND c3 ... (on capture le complémentaire
    # du trigger principal tout en gardant le contexte du chemin).
    if path.logical_op == 'NOT':
        not_node = ConditionNode(op='NOT', left=conditions[0])
        result = not_node
        for c in conditions[1:]:
            result = ConditionNode(op='AND', left=result, right=c)
        return result

    if len(conditions) == 1:
        return conditions[0]

    # AND récursif
    # AND(left, AND(rest...)) pour préserver l'ordre
    result = conditions[0]
    for c in conditions[1:]:
        result = ConditionNode(op='AND', left=result, right=c)
    return result

def merge_paths_or(
    paths: list[XGBPath],
) -> Condition | ConditionNode:
    """Combine plusieurs XGBPaths en DNF (Disjunctive Normal Form).

    Chaque path est converti en AST AND (via path_to_ast), puis tous
    les ASTs sont combines en OR : (P1 AND P1b) OR (P2 AND P2b) OR ...

    P2-1 (AI Review 2026-08-20) : permet d'exprimer des regles
    disjonctives que XGBoost ne peut pas capturer naturellement.

    Construction right-associative pour faciliter l'evaluation :
        OR(p1, OR(p2, OR(p3, ...)))

    Args:
        paths : liste de XGBPaths

    Returns:
        Condition si 1 path a 1 condition, ConditionNode sinon.
    """
    if not paths:
        raise ValueError("Liste de paths vide : impossible de merger")
    asts = [path_to_ast(p) for p in paths]
    if len(asts) == 1:
        return asts[0]
    # OR recursif right-associative : OR(p1, OR(p2, p3, ...))
    # On part de la fin : OR(p_{n-1}, p_n), puis OR(p_{n-2}, ...), etc.
    result = asts[-1]
    for a in reversed(asts[:-1]):
        result = ConditionNode(op="OR", left=a, right=result)
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
    X: np.ndarray,          # (N, F) float32
    feature_names: list[str],
) -> np.ndarray:
    """Évalue l'AST sur toute une matrice X (vectorisé numpy).

    FIX PERF (2026-08-21) : l'ancienne version bouclait ligne par ligne et
    reconstruisait un dict de 218 features à CHAQUE bougie, juste pour évaluer
    une condition utilisant 1-6 colonnes. C'était le goulot principal du backtest.
    Maintenant on évalue récursivement des masks numpy (N,) sur les seules colonnes
    utilisées par l'AST : O(N) numpy pur, sans boucle Python ni dict par ligne.

    Semantic conservée : NaN -> False (comme l'ancien _eval_atomic), NOT/AND/OR/XOR
    appliqués vectorisés.

    Returns:
        mask : (N,) bool, True aux indices où la condition est vraie.
    """
    # FIX SE-01 (2026-08-20) : si l'AST contient des atomes expr (STGP),
    # basculer sur l'evaluateur vectorise de search_engine (les expressions
    # arithmetiques ne peuvent pas passer par le parcours scalaire par ligne).
    from einherjar.research.search_engine import evaluator  # import lazy (cycle)
    if evaluator.has_expr_atoms(ast):
        return evaluator.eval_condition_ast(ast, X, feature_names)
    name_to_idx = {n: i for i, n in enumerate(feature_names)}
    return _eval_ast_numpy(ast, X, name_to_idx)


def _eval_ast_numpy(
    node: Condition | ConditionNode,
    X: np.ndarray,
    name_to_idx: dict,
) -> np.ndarray:
    """Évalue récursivement un sous-arbre et retourne un mask numpy (N,)."""
    import numpy as np
    n = X.shape[0]
    if isinstance(node, Condition):
        idx = name_to_idx.get(node.feature_ref)
        if idx is None:
            return np.zeros(n, dtype=bool)
        col = X[:, idx]
        op = node.operator
        v = node.value
        if op == "<":
            return col < v
        if op == "<=":
            return col <= v
        if op == ">":
            return col > v
        if op == ">=":
            return col >= v
        if op == "==":
            return np.abs(col - v) < 1e-9
        if op == "!=":
            valid = ~np.isnan(col)
            return valid & (np.abs(col - v) >= 1e-9)
        raise ValueError(f"Opérateur de comparaison non supporté : {op}")
    # ConditionNode : évaluer left (et right si binaire)
    left_mask = _eval_ast_numpy(node.left, X, name_to_idx)
    if node.op == "NOT":
        return ~left_mask
    if node.right is None:
        raise ValueError(f"Opérateur {node.op} requiert un nœud right")
    right_mask = _eval_ast_numpy(node.right, X, name_to_idx)
    if node.op == "AND":
        return left_mask & right_mask
    if node.op == "OR":
        return left_mask | right_mask
    if node.op == "XOR":
        return left_mask ^ right_mask
    raise ValueError(f"Opérateur logique non supporté : {node.op}")

def simplify_ast(ast: Condition | ConditionNode) -> Condition | ConditionNode:
    """Simplifie un AST de conditions en fusionnant les bornes redondantes.

    FIX QUALITE (2026-08-21) : les chemins XGBoost produisent parfois des
    conditions redondantes sur le meme feature (ex. `RSI_14 < 70 AND RSI_14 < 50`
    -> logiquement equivalent a `RSI_14 < 50`). On les fusionne pour eviter des
    Einhers doublons / inutilement complexes.

    Regles (sur un AND plat, cas courant apres path_to_ast) :
      x < a AND x < b  -> x < min(a,b)          (le plus contraignant)
      x <= a AND x <= b -> x <= min(a,b)
      x >  a AND x >  b  -> x > max(a,b)        (le plus contraignant)
      x >= a AND x >= b -> x >= max(a,b)
      conditions identiques (doublon) -> une seule
      Bornes croisees (x< a AND x>b avec a<=b) -> inutilement serre mais on conserve.

    La simplification SEULEMENT sur les noeuds AND compose d'atomes, a un niveau
    a la fois (pas de transformation algebrique profonde).

    Returns:
        AST simplifie (moins de noeuds, bornes fusionnees).
    """
    # Gathering : collecter les atomes d'un AND plat
    def collect(node, atoms):
        if isinstance(node, Condition):
            atoms.append(node)
        elif isinstance(node, ConditionNode) and node.op == "AND":
            collect(node.left, atoms)
            if node.right is not None:
                collect(node.right, atoms)
        else:
            atoms.append(node)  # noeud non-AND : on le laisse tel quel
        return atoms

    # Si pas un AND pur, retourner tel quel
    if not (isinstance(ast, ConditionNode) and ast.op == "AND"):
        return ast

    atoms = []
    collect(ast, atoms)
    # Regrouper par feature et operande (<,<=,>,>=,==,!=)
    from collections import defaultdict
    groups = defaultdict(list)
    for a in atoms:
        if isinstance(a, Condition):
            groups[(a.feature_ref, a.operator)].append(a.value)
        else:
            groups[(id(a), "NODE")].append(a)

    simplified = []
    for key, vals in groups.items():
        if key[1] == "NODE":
            simplified.extend(vals)
            continue
        feat, op = key
        if op in ("<", "<="):
            # conserver le MIN (le plus contraignant)
            chosen = min(vals)
        elif op in (">", ">="):
            chosen = max(vals)
        else:
            chosen = vals[0]  # == et != : on garde la 1ere (dedup simple)
        simplified.append(Condition(feature_ref=feat, operator=op, value=chosen,
                                    transformation=None))

    # FIX TAUTOLOGIE (2026-08-27) : si un feature a des bornes < et >=,
    # verifier que l'intervalle n'est pas trop serre (< 5% du range observe).
    # Ex: x < 0.7008 AND x >= 0.7000 → intervalle de 0.0008 = tautologie.
    # On regroupe par feature et on detecte les paires bornées.
    from collections import defaultdict
    feat_bounds = defaultdict(dict)
    for c in simplified:
        if isinstance(c, Condition):
            if c.operator in ("<", "<="):
                feat_bounds[c.feature_ref]["upper"] = c.value
            elif c.operator in (">", ">="):
                feat_bounds[c.feature_ref]["lower"] = c.value

    # Supprimer les paires tautologiques (intervalle < 1% de la borne supérieure)
    final_simplified = []
    for c in simplified:
        if isinstance(c, Condition):
            bounds = feat_bounds.get(c.feature_ref, {})
            if "upper" in bounds and "lower" in bounds:
                interval = bounds["upper"] - bounds["lower"]
                ref = max(abs(bounds["upper"]), abs(bounds["lower"]), 1e-10)
                if interval < 0.01 * ref:
                    # Intervalle trop serre : garder seulement la borne la plus contraignante
                    if c.operator in ("<", "<=") and c.value == bounds["upper"]:
                        continue  # on saute la borne haute, on garde la basse
                    elif c.operator in (">", ">=") and c.value == bounds["lower"]:
                        continue  # on saute la borne basse, on garde la haute
            final_simplified.append(c)
        else:
            final_simplified.append(c)
    simplified = final_simplified

    # Reconstruire l'AND chaine
    if len(simplified) == 1:
        return simplified[0]
    result = simplified[0]
    for c in simplified[1:]:
        result = ConditionNode(op="AND", left=result, right=c)
    return result

