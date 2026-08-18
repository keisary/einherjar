"""path_extractor.py - Extraction des chemins d'arbres GBDT (xgboost + sklearn).

Supporte deux formats de sortie :
- xgboost : `booster.get_dump()` → texte avec format `0:[f5<70] yes=1,no=2,missing=1\n1:leaf=0.012`
- sklearn : `estimator.estimators_` → array 2D de `_tree.Tree` avec attributs `feature`, `threshold`, `children_left`, `children_right`, `value`

API unifiée : `extract_paths(model, backend, feature_names, ...)`.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class XGBPath:
    """Un chemin dans un arbre GBDT."""
    conditions: tuple[tuple[str, str, float], ...]
    score: float
    tree_idx: int
    path_idx: int


# --------------------------------------------------------------------------- #
# Parser xgboost (texte)
# --------------------------------------------------------------------------- #


def parse_xgb_dump(dump_str: str) -> list[XGBPath]:
    """Parse le dump texte d'un arbre XGBoost."""
    node_re = re.compile(
        r"^(\d+):\[(.+?)\s*([<>=!]+)\s*([\-\d\.eE+]+)\]\s*yes=(\d+),no=(\d+),missing=(\d+)"
    )
    leaf_re = re.compile(r"^(\d+):leaf=([\-\d\.eE+]+)")
    nodes = {}
    for line in dump_str.strip().split("\n"):
        line = line.strip()
        m = node_re.match(line)
        if m:
            nodes[int(m.group(1))] = {
                "type": "internal",
                "feature": m.group(2),
                "op": m.group(3),
                "threshold": float(m.group(4)),
                "yes": int(m.group(5)),
                "no": int(m.group(6)),
                "missing": int(m.group(7)),
            }
            continue
        m = leaf_re.match(line)
        if m:
            nodes[int(m.group(1))] = {"type": "leaf", "value": float(m.group(2))}
    if not nodes:
        return []
    # Trouver la racine
    target_ids = set()
    for n_data in nodes.values():
        if n_data["type"] == "internal":
            target_ids.add(n_data["yes"])
            target_ids.add(n_data["no"])
    root_candidates = [nid for nid in nodes.keys() if nid not in target_ids]
    if not root_candidates:
        return []
    paths = []
    _walk_xgb(root_candidates[0], [], nodes, paths)
    return paths


def _walk_xgb(node_id, conditions, nodes, out, tree_idx=0):
    node = nodes.get(node_id)
    if node is None:
        return
    if node["type"] == "leaf":
        out.append(XGBPath(
            conditions=tuple(conditions),
            score=node["value"],
            tree_idx=tree_idx,
            path_idx=len(out),
        ))
        return
    feat = node["feature"]
    op = node["op"]
    threshold = node["threshold"]
    if op == "<":
        yes_cond = (feat, "<", threshold)
        no_cond = (feat, ">=", threshold)
    elif op == "<=":
        yes_cond = (feat, "<=", threshold)
        no_cond = (feat, ">", threshold)
    elif op == ">":
        yes_cond = (feat, ">", threshold)
        no_cond = (feat, "<=", threshold)
    elif op == ">=":
        yes_cond = (feat, ">=", threshold)
        no_cond = (feat, "<", threshold)
    elif op == "==":
        yes_cond = (feat, "==", threshold)
        no_cond = (feat, "!=", threshold)
    elif op == "!=":
        yes_cond = (feat, "!=", threshold)
        no_cond = (feat, "==", threshold)
    else:
        return
    _walk_xgb(node["yes"], conditions + [yes_cond], nodes, out, tree_idx)
    _walk_xgb(node["no"], conditions + [no_cond], nodes, out, tree_idx)


# --------------------------------------------------------------------------- #
# Parser sklearn (Tree interne)
# --------------------------------------------------------------------------- #


def parse_sklearn_tree(
    tree: Any,
    feature_names: list[str],
    tree_idx: int,
) -> list[XGBPath]:
    """Parse un arbre sklearn et retourne tous les chemins.

    Pour GradientBoostingRegressor, `model.estimators_` est shape (n_estimators, 1)
    (1 sortie par arbre). Chaque estimateur a un attribut `tree_` (sklearn.tree._tree.Tree).
    """
    children_left = tree.children_left
    children_right = tree.children_right
    feature = tree.feature
    threshold = tree.threshold
    value = tree.value  # shape (n_nodes, 1, 1) pour régression

    paths = []
    _walk_sklearn(
        node_id=0,
        conditions=[],
        children_left=children_left,
        children_right=children_right,
        feature=feature,
        threshold=threshold,
        value=value,
        feature_names=feature_names,
        out=paths,
        tree_idx=tree_idx,
    )
    return paths


def _walk_sklearn(
    node_id, conditions, children_left, children_right,
    feature, threshold, value, feature_names, out, tree_idx,
):
    if node_id == -1:
        return
    if children_left[node_id] == -1:  # feuille
        # value[node_id] est shape (1, 1) ou (1, 1, 1) selon version
        v = float(np.asarray(value[node_id]).flatten()[0])
        out.append(XGBPath(
            conditions=tuple(conditions),
            score=v,
            tree_idx=tree_idx,
            path_idx=len(out),
        ))
        return
    feat_idx = int(feature[node_id])
    if feat_idx < 0 or feat_idx >= len(feature_names):
        return
    feat_name = feature_names[feat_idx]
    thresh = float(threshold[node_id])
    # Branche "left" : feature <= threshold (sklearn convention)
    _walk_sklearn(
        children_left[node_id],
        conditions + [(feat_name, "<=", thresh)],
        children_left, children_right, feature, threshold, value,
        feature_names, out, tree_idx,
    )
    # Branche "right" : feature > threshold
    _walk_sklearn(
        children_right[node_id],
        conditions + [(feat_name, ">", thresh)],
        children_left, children_right, feature, threshold, value,
        feature_names, out, tree_idx,
    )


# --------------------------------------------------------------------------- #
# Extracteur unifié
# --------------------------------------------------------------------------- #


def extract_paths(
    model: Any,
    backend: str,
    feature_names: list[str],
    min_score: float = 0.005,
    max_score: float = 0.10,
    min_path_length: int = 1,
    max_path_length: int = 4,
    max_paths: int = 100,
) -> list[XGBPath]:
    """Extrait et filtre les chemins d'un modèle GBDT (xgboost ou sklearn).

    Returns:
        Liste de XGBPath triée par |score| décroissant.
    """
    if backend == "xgboost":
        all_paths = _extract_xgb(model, feature_names)
    else:
        all_paths = _extract_sklearn(model, feature_names)
    # Filtrer
    filtered = [
        p for p in all_paths
        if min_path_length <= len(p.conditions) <= max_path_length
        and min_score <= abs(p.score) <= max_score
    ]
    filtered.sort(key=lambda p: abs(p.score), reverse=True)
    result = filtered[:max_paths]
    logger.info(
        "extract_paths (backend=%s) : %d bruts → %d filtrés → top %d retenus",
        backend, len(all_paths), len(filtered), len(result),
    )
    return result


def _extract_xgb(model: Any, feature_names: list[str]) -> list[XGBPath]:
    """Extrait les chemins d'un XGBoost model (texte)."""
    booster = model.get_booster()
    dumps = booster.get_dump()
    all_paths = []
    for tree_idx, dump_str in enumerate(dumps):
        dump_named = _name_features_in_dump(dump_str, feature_names)
        paths = parse_xgb_dump(dump_named)
        for p in paths:
            all_paths.append(p)
    return all_paths


def _extract_sklearn(model: Any, feature_names: list[str]) -> list[XGBPath]:
    """Extrait les chemins d'un sklearn GradientBoostingRegressor."""
    all_paths = []
    # model.estimators_ est shape (n_estimators, n_outputs) pour GBR
    # Pour régression simple, n_outputs=1
    estimators = model.estimators_
    # Aplatir en 1D
    if hasattr(estimators, "flatten"):
        estimators = estimators.flatten()
    for tree_idx, est in enumerate(estimators):
        # est est un DecisionTreeRegressor, on accède à .tree_
        if hasattr(est, "tree_"):
            paths = parse_sklearn_tree(est.tree_, feature_names, tree_idx)
            for p in paths:
                all_paths.append(p)
    return all_paths


def _name_features_in_dump(dump_str: str, feature_names: list[str]) -> str:
    """Remplace les indices 'f0', 'f1', ... par les noms réels (xgboost)."""
    out = dump_str
    for i, name in enumerate(feature_names):
        for op in ("<", "<=", ">", ">=", "==", "!="):
            out = out.replace(f"[f{i}{op}", f"[{name}{op}")
    return out
