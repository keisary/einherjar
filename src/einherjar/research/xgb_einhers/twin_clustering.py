"""twin_clustering.py - Detection et generalisation des Einhers quasi-jumeaux.

2026-08-29 (decision Jovanny) : dans le corpus, plusieurs Einhers partagent
les memes features avec des seuils proches (ex: 4 einhers avec les memes
2-3 conditions a 0.001 pres). Ce sont des "quasi-jumeaux" : meme signal,
bruit de seuil en plus.

Contrairement a une dedup destructive, ce module :
1. Detecte les groupes de quasi-jumeaux (features communes + seuils proches)
2. Construit une version GENERALISEE (bornes elargies couvrant le groupe)
3. La backteste (le meme circuit val+holdout que les sources)
4. L'AJOUTE au corpus si elle passe l'admission - SANS supprimer les originaux

Usage :
    from .twin_clustering import find_twin_groups, build_generalized_einher

Critere de jumeaux :
- Jaccard des features >= 0.8 (meme famille de features)
- Pour chaque feature partagee : ecart relatif des seuils < 10%
- Meme direction (BUY/SELL)
- Meme amplitude_bars (horizon)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Seuils de similarite (accord avec le design 2026-08-28)
FEATURE_JACCARD_MIN = 0.8
THRESHOLD_REL_TOL = 0.10  # 10% d'ecart relatif max par feature


def _features_of(einher: Any) -> dict[str, tuple[str, float]]:
    """Extrait {feature: (operateur, seuil)} de la condition_tree d'un Einher.

    Parcourt recursivement l'AST (ConditionNode -> dict) pour recuperer les
    atoms. Fonctionne avec l'AST simplifie (AND d'atomes).
    """
    feat_map: dict[str, tuple[str, float]] = {}
    ct = einher.condition_tree
    if hasattr(ct, "to_dict"):
        ct = ct.to_dict()
    if not isinstance(ct, dict):
        return feat_map

    def _walk(node: Any) -> None:
        if not isinstance(node, dict):
            return
        if "feature_ref" in node and "operator" in node and "value" in node:
            f = node["feature_ref"]
            op = node.get("operator", "<")
            val = node.get("value")
            if isinstance(val, (int, float)):
                # Garder la borne la plus stricte si la feature apparait 2x
                if f not in feat_map:
                    feat_map[f] = (op, float(val))
        for v in node.values():
            if isinstance(v, dict):
                _walk(v)
            elif isinstance(v, list):
                for i in v:
                    _walk(i)

    _walk(ct)
    return feat_map


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _thresholds_close(fa: dict, fb: dict) -> bool:
    """Vrai si tous les seuils des features partagees sont a <10% pres."""
    shared = set(fa) & set(fb)
    if not shared:
        return True  # pas de feature partagee : rien a comparer
    for f in shared:
        op_a, val_a = fa[f]
        op_b, val_b = fb[f]
        if op_a != op_b:
            return False  # operateur different = signal different
        denom = max(abs(val_a), abs(val_b), 1e-9)
        if abs(val_a - val_b) / denom > THRESHOLD_REL_TOL:
            return False
    return True


@dataclass
class TwinGroup:
    """Groupe d'Einhers quasi-jumeaux."""

    members: list[Any]  # e_inhers originaux (objets Einher)
    features: dict[str, tuple[str, float]]  # union des features/bornes

    @property
    def size(self) -> int:
        return len(self.members)

    def feature_set(self) -> set[str]:
        return set(self.features.keys())


def find_twin_groups(einhers: list[Any], max_groups: int = 50) -> list[TwinGroup]:
    """Detecte les groupes de quasi-jumeaux parmi une liste d'Einhers.

    Args:
        einhers : liste d'Einhers (objets avec condition_tree, direction,
            amplitude_bars).
        max_groups : nombre max de groupes retournes (garde-fou perf).

    Returns:
        Liste de TwinGroup (membres + union des features/bornes).
    """
    feats = [_features_of(e) for e in einhers]
    n = len(einhers)
    used = [False] * n
    groups: list[TwinGroup] = []

    for i in range(n):
        if used[i]:
            continue
        members = [einhers[i]]
        used[i] = True
        base_f = feats[i]
        for j in range(i + 1, n):
            if used[j]:
                continue
            f_j = feats[j]
            # Meme direction + meme horizon
            if einhers[i].direction != einhers[j].direction:
                continue
            if einhers[i].amplitude_bars != einhers[j].amplitude_bars:
                continue
            sim = _jaccard(set(base_f), set(f_j))
            if sim >= FEATURE_JACCARD_MIN and _thresholds_close(base_f, f_j):
                members.append(einhers[j])
                used[j] = True
                # Etendre les bornes : min des valeurs basses, max des hautes
                for f, (op, val) in f_j.items():
                    if f not in base_f:
                        base_f[f] = (op, val)
        if len(members) >= 2:
            groups.append(TwinGroup(members=members, features=base_f))
            if len(groups) >= max_groups:
                break

    return groups


def build_generalized_einher(
    group: TwinGroup,
    model_tag: str = "twin_generalized",
) -> Any:
    """Construit un Einher GENERALISE depuis un groupe de quasi-jumeaux.

    La condition generalisee est l'intersection des conditions des membres :
    une feature commune garde la borne la plus STRICTE (couvre tous les
    membres). Les features non partagees par tous sont DROPPEES (elles
    peuvent diverger entre membres).

    L'Einher generalise est clone du premier membre (univers, direction,
    amplitude, tp/sl) avec la nouvelle condition et un nouvel ID.
    """
    from .types import Einher

    if not group.members:
        return None
    base = group.members[0]

    # Compter les occurrences de chaque feature parmi les membres
    feat_counts: dict[str, int] = {}
    for m in group.members:
        fm = _features_of(m)
        for f in fm:
            feat_counts[f] = feat_counts.get(f, 0) + 1

    n_members = len(group.members)
    # Features partagees par TOUS les membres
    common = [f for f, c in feat_counts.items() if c == n_members]
    if not common:
        logger.debug("Groupe sans feature commune : skip generalisation")
        return None

    # Bornes les plus strictes (min pour '<', max pour '>')
    cond_atoms = []
    for f in sorted(common):
        ops_vals = [group.features.get(f)]
        # Recuperer les (op, val) de chaque membre
        per_member = [_features_of(m).get(f) for m in group.members]
        per_member = [pv for pv in per_member if pv is not None]
        if not per_member:
            continue
        ops = {pv[0] for pv in per_member}
        if len(ops) > 1:
            continue  # operateurs differents : pas generalisable simplement
        op = per_member[0][0]
        values = [pv[1] for pv in per_member]
        if op in ("<", "<="):
            val = min(values)  # plus stricte
        elif op in (">", ">="):
            val = max(values)  # plus stricte
        else:
            continue
        cond_atoms.append({"feature_ref": f, "operator": op, "value": val})

    if not cond_atoms:
        return None

    # AST : Condition [AND ConditionNode]
    # FIX TYPES (2026-08-30) : condition_tree est un Condition | ConditionNode
    # (pas un dict). Einher est frozen -> dataclasses.replace, pas d'assignation.
    from .types import Condition, ConditionNode

    atoms = []
    for a in cond_atoms:
        atoms.append(Condition(
            feature_ref=a["feature_ref"],
            operator=a["operator"],
            value=a["value"],
        ))
    if len(atoms) == 1:
        ast_new = atoms[0]
    else:
        # AND droite-associatif comme le reste du code (condition_tree.py)
        node = atoms[0]
        for a in atoms[1:]:
            node = ConditionNode(op="AND", left=node, right=a)
        ast_new = node

    # Cloner le premier membre avec la nouvelle condition (frozen-safe)
    import dataclasses
    import uuid

    gen_id = (
        f"twin_{base.universe.get('asset', 'x')}_{base.universe.get('timeframe', 'x')}_"
        f"{base.universe.get('horizon', 'x')}_{base.direction.lower()}_"
        f"{uuid.uuid4().hex[:8]}"
    )
    gen = dataclasses.replace(
        base,
        id=gen_id,
        condition_tree=ast_new,
        source={
            "model": model_tag,
            "n_members": n_members,
            "member_ids": [m.id for m in group.members[:10]],
            "features": sorted(common),
        },
    )
    logger.info(
        "Generalisation twin : %d membres -> %s (%d features communes)",
        n_members, gen.id, len(common),
    )
    return gen