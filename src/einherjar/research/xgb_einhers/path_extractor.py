r"""path_extractor.py - Extraction des chemins d'arbres GBDT (xgboost + sklearn).

Supporte deux formats de sortie :
- xgboost : `booster.get_dump()` → texte avec format `0:[f5<70] yes=1,no=2,missing=1\n1:leaf=0.012`
- sklearn : `estimator.estimators_` -> array 2D de `_tree.Tree` avec attributs
  `feature`, `threshold`, `children_left`, `children_right`, `value`

API unifiée : `extract_paths(model, backend, feature_names, ...)`.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class XGBPath:
    """Un chemin dans un arbre GBDT.

    log (2026-08-21, problème 3) : on étend pour supporter la génération
    de variantes logiques (OR/NOT/XOR) en plus des chemins AND purs.
    - logical_op : 'AND' (défaut) | 'OR' | 'NOT' | 'XOR'
    - sub_paths   : pour OR/XOR, liste des sous-chemins combinés (chacun
                    un XGBPath AND). Pour NOT, conditions déjà négativées.
    """

    conditions: tuple[tuple[str, str, float], ...]
    score: float
    tree_idx: int
    path_idx: int
    logical_op: str = "AND"
    sub_paths: tuple[XGBPath, ...] = ()


# --------------------------------------------------------------------------- #
# Parser xgboost (texte)
# --------------------------------------------------------------------------- #


def parse_xgb_dump(dump_str: str) -> list[XGBPath]:
    """Parse le dump texte d'un arbre XGBoost."""
    node_re = re.compile(r"^(\d+):\[(.+?)\s*([<>=!]+)\s*([\-\d\.eE+]+)\]\s*yes=(\d+),no=(\d+),missing=(\d+)")
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
        out.append(
            XGBPath(
                conditions=tuple(conditions),
                score=node["value"],
                tree_idx=tree_idx,
                path_idx=len(out),
            )
        )
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
    node_id,
    conditions,
    children_left,
    children_right,
    feature,
    threshold,
    value,
    feature_names,
    out,
    tree_idx,
):
    if node_id == -1:
        return
    if children_left[node_id] == -1:  # feuille
        # value[node_id] est shape (1, 1) ou (1, 1, 1) selon version
        v = float(np.asarray(value[node_id]).flatten()[0])
        out.append(
            XGBPath(
                conditions=tuple(conditions),
                score=v,
                tree_idx=tree_idx,
                path_idx=len(out),
            )
        )
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
        children_left,
        children_right,
        feature,
        threshold,
        value,
        feature_names,
        out,
        tree_idx,
    )
    # Branche "right" : feature > threshold
    _walk_sklearn(
        children_right[node_id],
        conditions + [(feat_name, ">", thresh)],
        children_left,
        children_right,
        feature,
        threshold,
        value,
        feature_names,
        out,
        tree_idx,
    )


# --------------------------------------------------------------------------- #
# Extracteur unifié
# --------------------------------------------------------------------------- #


def _negate_condition(cond: tuple[str, str, float]) -> tuple[str, str, float]:
    """Négation d'une condition atomique (feature op value)."""
    f, op, v = cond
    neg = {"<": ">=", "<=": ">", ">": "<=", ">=": "<", "==": "!=", "!=": "=="}
    return (f, neg.get(op, "!="), v)


_VARIANT_UID_COUNTER = [0]


def _next_variant_uid(kind_base: int) -> int:
    """P2-1bis : uid STRICTEMENT unique pour une variante logique.

    L'ancien encodage 20000+path_idx entrait en collision des que deux chemins
    partageaient le meme path_idx dans des arbres differents.
    """
    _VARIANT_UID_COUNTER[0] = (_VARIANT_UID_COUNTER[0] + 1) % 9000
    return kind_base + _VARIANT_UID_COUNTER[0]


def build_logical_variants(
    paths: list[XGBPath],
    top_n: int = 5,
) -> list[XGBPath]:
    """Génère des variantes OR/NOT/XOR depuis les chemins AND (problème 3).

    Stratégie (validée Jovanny) :
    - DNF/OR : combine les top chemins AND (ceux qui capturent des régimes
      complémentaires) en OR — (P1 AND ...) OR (P2 AND ...).
    - NOT : pour chaque chemin, génère une variante qui NÉGIE une condition
      atomique (ex. `feat < th` → `feat >= th`) pour capter le complémentaire.
    - XOR : combine 2 conditions d'un même feature en disjonction exclusive.

    Returns:
        Liste des variantes logiques (logical_op != 'AND'), en plus des AND.
    """
    variants: list[XGBPath] = []
    if not paths:
        return variants
    # --- OR : combiner les top_n chemins AND en OR (DNF) ---
    if len(paths) >= 2:
        top = sorted(paths, key=lambda p: abs(p.score), reverse=True)[:top_n]
        # OR sur les 2 meilleurs chemins de direction opposée si possible
        best_pair = None
        for i in range(len(top)):
            for j in range(i + 1, len(top)):
                if top[i].conditions and top[j].conditions:
                    best_pair = (top[i], top[j])
                    break
            if best_pair:
                break
        if best_pair:
            p1, p2 = best_pair
            # P2-1bis : path_idx UNIQUE (l'ancien 10001 fixe entrait en collision
            # entre arbres -> ids d'Einhers instables).
            _or_uid = _next_variant_uid(10000)
            variants.append(
                XGBPath(
                    conditions=(),
                    score=max(p1.score, p2.score),
                    tree_idx=p1.tree_idx,
                    path_idx=_or_uid,
                    logical_op="OR",
                    sub_paths=(p1, p2),
                )
            )

    # --- NOT : négation d'une condition atomique d'un chemin ---
    # Le condition d'origine (non niée) est passé tel quel ; c'est
    # path_to_ast qui applique le NOT() autour. Si on pré-négatait ici
    # ET que path_to_ast en remettait un → double négation = condition
    # d'origine (variante tautologique inutile).
    for p in paths[:top_n]:
        if not p.conditions:
            continue
        # P2-1bis : uid unique par (tree_idx, path_idx)
        _not_uid = _next_variant_uid(20000)
        # On marque le chemin comme NOT ; path_to_ast fera NOT(c1) AND reste
        variants.append(
            XGBPath(
                conditions=p.conditions,  # pas négativé ici
                score=p.score,
                tree_idx=p.tree_idx,
                path_idx=_not_uid,
                logical_op="NOT",
            )
        )

    # NOTE P3-4 (2026-08-26) : le XOR est SUPPRIME de la generation.
    # Recherche documentee : aucune litterature de finance quantitative
    # n'utilise XOR entre conditions techniques. Entre deux conditions du
    # meme feature c'est une redondance (equivalent a un intervalle), entre
    # features differents aucune interpretation economique n'existe.
    # Les disjonctions legitimes sont gerees par logical_refiner.evaluate_or_pairs
    # (OR-de-regimes, fondes Disjunctive Emerging Patterns).
    return variants


def extract_paths(
    model: Any,
    backend: str,
    feature_names: list[str],
    min_score: float = 0.0005,
    max_score: float = 0.10,
    min_path_length: int = 1,
    max_path_length: int = 6,
    max_paths: int = 100,
    enable_logical_variants: bool = False,
    family_map: dict[str, str] | None = None,
    macro_family_cap: float = 0.40,
) -> list[XGBPath]:
    """Extrait et filtre les chemins d'un modèle GBDT (xgboost ou sklearn).

    FIX (2026-08-21, problème 2) :
    - max_path_length relève à 6 par défaut (le grid peut choisir depth=6 ;
      auparavant 4 excluait tous les chemins d'arbres profonds → 1 filtré / 4294).
    - si min_score <= 0 (auto) : on calcule un seuil RELATIF = 33e percentile des
      |scores| des chemins. Un seuil quasi-nul (1e-9) gardait les feuilles extrêmes
      et rares → 0 trades au backtest. Le percentile ne garde que les feuilles
      vraiment significatives, adapté à la vol (crypto vs forex).

    Args:
            model: TODO: documenter.
            backend: TODO: documenter.
            feature_names: TODO: documenter.
            min_score: TODO: documenter.
            max_score: TODO: documenter.
            min_path_length: TODO: documenter.
            max_path_length: TODO: documenter.
            max_paths: TODO: documenter.
            family_map: mapping feature_name -> economic_family (taxonomie).
            macro_family_cap: part maximale du budget pour une macro-famille.
            feature_names: TODO: documenter.
            max_path_length: TODO: documenter.
            max_paths: TODO: documenter.
            max_score: TODO: documenter.
            min_path_length: TODO: documenter.
            min_score: TODO: documenter.
            model: TODO: documenter.
                family_map: TODO: documenter.
                macro_family_cap: TODO: documenter.

    Args:
        enable_logical_variants : si True (problème 3), ajoute des variantes
            OR/NOT/XOR générées depuis les chemins AND purs.

    Returns:
        Liste de XGBPath triée par |score| décroissant.
    """
    if backend == "xgboost":
        all_paths = _extract_xgb(model, feature_names)
    else:
        all_paths = _extract_sklearn(model, feature_names)

    # Auto min_score : percentile des |scores| des chemins (volée-adaptée)
    effective_min = min_score
    if effective_min <= 0 and all_paths:
        abs_scores = sorted(abs(p.score) for p in all_paths)
        # 33e percentile : on garde ~2/3 des feuilles les plus marquées
        effective_min = float(np.percentile(abs_scores, 33))
        effective_min = max(effective_min, 1e-9)
        logger.info("extract_paths auto-min_score : p33 des |scores| = %.6g", effective_min)

    # Filtrer
    filtered = [
        p
        for p in all_paths
        if min_path_length <= len(p.conditions) <= max_path_length and effective_min <= abs(p.score) <= max_score
    ]
    filtered.sort(key=lambda p: abs(p.score), reverse=True)
    # FIX DIVERSITE (2026-08-21) : CAP par feature dominante (1re condition)
    cap_per_feature = max(1, max_paths // 15)  # FIX DIVERSITE (2026-08-27) : 2 au lieu de 3-4
    result: list[XGBPath] = []
    feat_count: dict[str, int] = {}

    # P3-2 (2026-08-25) : CAP PAR MACRO-FAMILLE avec redistribution.
    # Un plafond (pas un plancher) : une famille ne peut pas monopoliser plus de
    # macro_family_cap du budget ; les budgets non consommes sont redistribues
    # aux familles suivantes. Aucun remplissage artificiel si une famille n'a
    # pas de chemins qualifies.
    def _macro_family(feature: str) -> str:
        if not family_map:
            return "default"
        fam = family_map.get(feature, "unknown")
        if fam in ("price_action", "market_structure"):
            return "binary"
        if fam == "market_regime":
            return "regime"
        return "continuous"

    macro_counts: dict[str, int] = {}
    macro_cap = max(1, int(max_paths * macro_family_cap))
    skipped_macro: list[XGBPath] = []

    # Passe 1 : top-N avec caps (feature ET macro-famille)
    for p in filtered:
        if len(result) >= max_paths:
            break
        head = p.conditions[0][0] if p.conditions else p.tree_idx
        head_key = str(head)
        if feat_count.get(head_key, 0) >= cap_per_feature:
            continue
        macro = _macro_family(head) # type: ignore
        if family_map and macro_counts.get(macro, 0) >= macro_cap:
            skipped_macro.append(p)
            continue
        feat_count[head_key] = feat_count.get(head_key, 0) + 1
        macro_counts[macro] = macro_counts.get(macro, 0) + 1
        result.append(p)

    # Passe 2 (redistribution) : reprendre les chemins skippés par le cap macro
    # si le budget global n'est pas consomme. Le cap par feature reste actif.
    if skipped_macro and len(result) < max_paths and family_map:
        for p in skipped_macro:
            if len(result) >= max_paths:
                break
            head = p.conditions[0][0] if p.conditions else p.tree_idx
            head_key = str(head)
            if feat_count.get(head_key, 0) >= cap_per_feature:
                continue
            feat_count[head_key] = feat_count.get(head_key, 0) + 1
            result.append(p)
        logger.info(
            "extract_paths redistribution : %d chemins repris apres cap macro-familles",
            len([p for p in result if p not in filtered[:len(result)]]),
        )
    if family_map:
        logger.info(
            "extract_paths repartition macro-familles : %s (cap=%d/%d)",
            dict(macro_counts), macro_cap, max_paths,
        )

    # Problème 3 : ajouter les variantes logiques depuis les chemins retenus
    if enable_logical_variants and result:
        variants = build_logical_variants(result, top_n=min(5, len(result)))
        result = result + variants
        # Re-trier : les AND purs d'abord, puis les variantes
        result = sorted(result, key=lambda p: (p.logical_op != "AND", -abs(p.score)))
    logger.info(
        "extract_paths (backend=%s) : %d bruts → %d filtrés → top %d retenus (+%d variantes logiques)",
        backend,
        len(all_paths),
        len(filtered),
        len(result) - (len(variants) if enable_logical_variants and result else 0), # type: ignore
        (len(variants) if enable_logical_variants and result else 0), # type: ignore
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
    """Remplace les indices 'f0', 'f1', ... par les noms réels (xgboost).

    FIX PERF (2026-08-21) : l'ancienne version faisait jusqu'à F×6 = ~1308
    scans de la chaîne (`.replace()`) par arbre. Maintenant un seul `re.sub`
    avec une fonction de lookup : O(dump) en une passe.
    """
    import re

    # Seul le motif "[f<N><op>" (avec un opérateur de comparaison) doit être
    # renommé : on évite de toucher aux feuilles "leaf=".
    name_by_idx = {i: n for i, n in enumerate(feature_names)}

    def _repl(m: re.Match) -> str:
        idx = int(m.group(1))
        name = name_by_idx.get(idx, m.group(0)[1:])
        return f"[{name}{m.group(2)}"

    # motif : '[f' + digits + (l'un des opérateurs) + -> remplace l'indice
    return re.sub(r"\[f(\d+)(<=|>=|==|!=|<|>)", _repl, dump_str)
