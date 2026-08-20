"""generator.py — Génération STGP des arbres de conditions.

Koza (1992) ramped half-and-half : moitié « full » (toutes les branches à
profondeur max), moitié « grow » (profondeur variable). Crossover et mutation
respectent le typage fort (numérique ↔ numérique, booléen ↔ booléen) et la
borne max_depth (plan ligne 318-321). Anti-bloat : taille plafonnée à
cfg.max_size (plan ligne 49). Les arbres sont immuables (dataclasses frozen);
les opérateurs reconstruisent le chemin racine → nœud.
"""
from __future__ import annotations

import numpy as np

from einherjar.research.search_engine.expression import (
    BinNum,
    BoolOp,
    Cmp,
    Const,
    Feature,
    depth,
    size,
)
from einherjar.research.search_engine.space import SpaceConfig

_P_BINARY_BOOL = 0.7


# --------------------------------------------------------------------------- #
# Génération aléatoire
# --------------------------------------------------------------------------- #


def _random_const(cfg: SpaceConfig, rng: np.random.Generator) -> Const:
    return Const(float(rng.choice(cfg.const_values)))


def _random_feature(cfg: SpaceConfig, rng: np.random.Generator) -> Feature:
    return Feature(str(rng.choice(cfg.feature_names)))


def generate_random_num_expr(
    rng: np.random.Generator,
    cfg: SpaceConfig,
    max_depth: int | None = None,
    method: str = "grow",
) -> object:
    """Génère une NumExpr aléatoire (profondeur <= max_depth)."""
    max_depth = cfg.max_depth if max_depth is None else max_depth

    def build(d: int) -> object:
        if d <= 0 or (rng.random() < 0.3 and method != "full"):
            return _random_const(cfg, rng) if rng.random() < 0.3 else _random_feature(cfg, rng)
        op = str(rng.choice(cfg.numeric_ops))
        return BinNum(op=op, left=build(d - 1), right=build(d - 1))

    return build(max_depth)


def _threshold_for(expr: object, pool: object, rng: np.random.Generator) -> float:
    """Tire un seuil du pool : per-feature si atome Feature, sinon global."""
    if isinstance(expr, Feature):
        return float(rng.choice(pool.per_feature[expr.feature_ref]))
    return float(rng.choice(pool.global_values))


def generate_random_bool_expr(
    rng: np.random.Generator,
    cfg: SpaceConfig,
    pool: object,
    max_depth: int | None = None,
    method: str = "grow",
) -> object:
    """Génère une BoolExpr aléatoire (profondeur <= max_depth, >= 1 atome Cmp)."""
    max_depth = cfg.max_depth if max_depth is None else max_depth

    def build(d: int) -> object:
        if d < 2 or rng.random() < 0.3:
            num = generate_random_num_expr(rng, cfg, max_depth=max(0, d - 1), method=method)
            op = str(rng.choice(cfg.cmp_ops))
            return Cmp(expr=num, operator=op, value=_threshold_for(num, pool, rng))
        if rng.random() < _P_BINARY_BOOL:
            op = str(rng.choice(("AND", "OR", "XOR")))
            return BoolOp(op=op, left=build(d - 1), right=build(d - 1))
        return BoolOp(op="NOT", left=build(d - 1))

    for _ in range(6):
        e = build(max_depth)
        if size(e) <= cfg.max_size:
            return e
    num = _random_feature(cfg, rng)
    return Cmp(expr=num, operator=str(rng.choice(cfg.cmp_ops)), value=_threshold_for(num, pool, rng))


def generate_population(
    rng: np.random.Generator,
    cfg: SpaceConfig,
    pool: object,
    n: int,
) -> list[object]:
    """Population initiale : ramped half-and-half (full + grow, min..max_depth)."""
    pop: list[object] = []
    span = cfg.max_depth - cfg.min_depth + 1
    for i in range(n):
        d = cfg.min_depth + i % span
        method = "full" if (i // span) % 2 == 0 else "grow"
        pop.append(generate_random_bool_expr(rng, cfg, pool, max_depth=d, method=method))
    return pop


# --------------------------------------------------------------------------- #
# Chemin racinaux et reconstruction immutable
# --------------------------------------------------------------------------- #


def _collect_paths(expr: object, kind: str, path: tuple[str, ...] = ()):
    """Itère (path, node) pour tous les nœuds du `kind` demandé.

    kind='bool' : nœuds BoolExpr (Cmp, BoolOp). kind='num' : nœuds NumExpr
    (Const, Feature, BinNum) — y compris ceux sous les Cmp.
    """
    if kind == "bool":
        if isinstance(expr, (Cmp, BoolOp)):
            yield path, expr
        if isinstance(expr, BoolOp):
            yield from _collect_paths(expr.left, kind, path + ("left",))
            if expr.right is not None:
                yield from _collect_paths(expr.right, kind, path + ("right",))
    else:
        if isinstance(expr, (Const, Feature, BinNum)):
            yield path, expr
        if isinstance(expr, BinNum):
            yield from _collect_paths(expr.left, kind, path + ("left",))
            yield from _collect_paths(expr.right, kind, path + ("right",))
        elif isinstance(expr, Cmp):
            yield from _collect_paths(expr.expr, kind, path + ("expr",))


def _with_slot(node: object, slot: str, child: object) -> object:
    """Reconstruit `node` avec son enfant `slot` remplacé par `child`."""
    if isinstance(node, BoolOp):
        if slot == "left":
            return BoolOp(op=node.op, left=child, right=node.right)
        return BoolOp(op=node.op, left=node.left, right=child)
    if isinstance(node, BinNum):
        if slot == "left":
            return BinNum(op=node.op, left=child, right=node.right)
        return BinNum(op=node.op, left=node.left, right=child)
    if isinstance(node, Cmp) and slot == "expr":
        return Cmp(expr=child, operator=node.operator, value=node.value)
    raise ValueError(f"Slot {slot} invalide pour {type(node).__name__}")


def _rebuild(node: object, path: tuple[str, ...], new: object) -> object:
    """Remplace le nœud au `path` par `new` (reconstruction immutable chemin entier)."""
    if not path:
        return new
    slot = path[0]
    child = getattr(node, slot)
    return _with_slot(node, slot, _rebuild(child, path[1:], new))


def _in_bounds(root: object, cfg: SpaceConfig) -> bool:
    return depth(root) <= cfg.max_depth and size(root) <= cfg.max_size


# --------------------------------------------------------------------------- #
# Opérateurs génétiques
# --------------------------------------------------------------------------- #


def crossover(
    a: object,
    b: object,
    rng: np.random.Generator,
    cfg: SpaceConfig,
    pool: object,
) -> object:
    """Crossover typé : sous-arbre booléen de a ↔ sous-arbre booléen de b.

    Si la profondeur/taille résultante dépasse les bornes → a inchangé.
    """
    paths_a = list(_collect_paths(a, "bool"))
    paths_b = list(_collect_paths(b, "bool"))
    path_a, _ = paths_a[int(rng.integers(len(paths_a)))]
    _, node_b = paths_b[int(rng.integers(len(paths_b)))]
    child = _rebuild(a, path_a, node_b)
    if not _in_bounds(child, cfg):
        return a
    return child


def mutate(
    expr: object,
    rng: np.random.Generator,
    cfg: SpaceConfig,
    pool: object,
    p_point: float = 0.4,
    p_grow: float = 0.4,
) -> object:
    """Mutation : point (feuille/op), grow (sous-arbre frais) ou shrink (feuille)."""
    kind = "bool" if rng.random() < 0.7 else "num"
    paths = list(_collect_paths(expr, kind))
    if not paths:
        kind = "bool"
        paths = list(_collect_paths(expr, kind))
    path, node = paths[int(rng.integers(len(paths)))]
    r = rng.random()
    if r < p_point:
        new = _point_mutate(node, rng, cfg, pool, kind)
    elif r < p_point + p_grow:
        if kind == "bool":
            new = generate_random_bool_expr(rng, cfg, pool, max_depth=3, method="grow")
        else:
            new = generate_random_num_expr(rng, cfg, max_depth=2, method="grow")
    else:
        if kind == "num":
            new = _random_const(cfg, rng) if rng.random() < 0.5 else _random_feature(cfg, rng)
        else:
            num = _random_feature(cfg, rng)
            new = Cmp(expr=num, operator=str(rng.choice(cfg.cmp_ops)), value=_threshold_for(num, pool, rng))
    if new is None:
        return expr
    child = _rebuild(expr, path, new)
    if not _in_bounds(child, cfg):
        return expr
    return child


def _point_mutate(
    node: object,
    rng: np.random.Generator,
    cfg: SpaceConfig,
    pool: object,
    kind: str,
) -> object | None:
    """Mutation ponctuelle : feuille/op changés, structure conservée."""
    if kind == "bool":
        if isinstance(node, Cmp):
            num = node.expr if rng.random() < 0.5 else _random_feature(cfg, rng)
            return Cmp(expr=num, operator=str(rng.choice(cfg.cmp_ops)), value=_threshold_for(num, pool, rng))
        if isinstance(node, BoolOp):
            ops = [o for o in cfg.bool_ops if o != node.op]
            op = str(rng.choice(ops))
            if op == "NOT":
                return BoolOp(op="NOT", left=node.left)
            right = node.right if node.right is not None else _random_bool_leaf(rng, cfg, pool)
            return BoolOp(op=op, left=node.left, right=right)
        return None
    if isinstance(node, Const):
        return _random_const(cfg, rng)
    if isinstance(node, Feature):
        return _random_feature(cfg, rng)
    if isinstance(node, BinNum):
        return BinNum(op=str(rng.choice(cfg.numeric_ops)), left=node.left, right=node.right)
    return None


def _random_bool_leaf(rng: np.random.Generator, cfg: SpaceConfig, pool: object) -> object:
    num = _random_feature(cfg, rng)
    return Cmp(expr=num, operator=str(rng.choice(cfg.cmp_ops)), value=_threshold_for(num, pool, rng))