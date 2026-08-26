"""logical_refiner.py - Raffinement post-génération : OR-de-régimes et veto-NOT.

P3-4 (Phase 3, 2026-08-26) — fondé sur la recherche documentée :

OR (disjonction) — légitime UNIQUEMENT pour des mécanismes complémentaires :
  La théorie des Disjunctive Emerging Patterns (Fan & Ramamohanarao ;
  Loekito & Bailey, ACM 2006) valide la disjonction quand les cas positifs
  relèvent de MÉCANISMES DIFFÉRENTS (leur exemple : deux voies biologiques
  causant la même maladie). En trading, c'est le cas du même signal actif
  dans deux régimes de marché disjoints (littérature regime-switching :
  KAMA+MSR arXiv:2208.11574, systèmes HMM multi-régimes).
  Critères retenus :
    - les 2 chemins ont le MÊME signe de score (même direction économique)
    - features dominants DIFFÉRENTS (2 mécanismes, pas 1)
    - chaque branche seule est significative (t-stat >= 2 sur train)
    - l'union couvre PLUS que chaque branche (complémentarité réelle)
    - qualité mesurée en WRAcc > 0 (Subgroup Discovery, Lavrac et al.)

NOT (négation) — veto uniquement, jamais déclencheur primaire :
  La littérature pattern mining à items négatifs (arXiv:2004.08015) emploie
  la négation comme compacteur descriptif. En trading, l'usage économiquement
  sensé est le filtre de veto ("momentum long MAIS PAS régime de crise").
  Un veto réduit l'univers des trades ; il ne crée pas de signal.
  Critères retenus :
    - appliqué APRÈS admission sur un Einher déjà validé
    - la condition de veto doit retirer <= 20% des trades val
    - le sharpe val après veto doit être >= sharpe val avant veto
    - la variante vetoée est archivée avec sa comparaison

XOR — SUPPRIMÉ : aucune justification économique trouvée dans la littérature
de finance quantitative ; entre conditions d'un même feature c'est une
redondance (intervalle), entre features différents aucune interprétation.
"""
from __future__ import annotations

import itertools
import logging
import uuid
from dataclasses import dataclass

import numpy as np

from .types import Condition, ConditionNode, Einher, EinherMetrics

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Utilitaires WRAcc (Subgroup Discovery)
# --------------------------------------------------------------------------- #


def wracc(
    mask: np.ndarray,
    y: np.ndarray,
    positive_mask: np.ndarray | None = None,
) -> float:
    """Weighted Relative Accuracy d'un sous-groupe.

    WRAcc = coverage * (precision_cond - precision_base)
          = p(S et T) - p(S) * p(T)

    Args:
        mask : (N,) bool — sous-groupe décrit par la règle.
        y : (N,) target signé ; un trade "positif" = y > 0.
        positive_mask : optionnel, mask des exemples positifs globaux.
            Si None, dérivé de y > 0.

    Returns:
        float dans [-0.25, 0.25] ; 0 = sous-groupe sans intérêt.
    """
    n = len(y)
    if n == 0 or not mask.any():
        return 0.0
    pos = (y > 0) if positive_mask is None else positive_mask
    p_s = float(mask.mean())
    p_t = float(pos.mean())
    p_st = float((mask & pos).mean())
    return p_st - p_s * p_t


def branch_t_stat(y: np.ndarray, mask: np.ndarray) -> float:
    """t-stat one-sided du rendement conditionnel d'une branche."""
    k = int(mask.sum())
    if k < 5:
        return 0.0
    vals = y[mask].astype(np.float64)
    std = float(np.std(vals, ddof=1))
    if std <= 0.0:
        return 0.0
    return float(np.mean(vals) / (std / np.sqrt(k)))


# --------------------------------------------------------------------------- #
# OR-de-régimes
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class OrCandidate:
    """Une paire complémentaire candidate à la disjonction."""

    path_a: object            # XGBPath branche A
    path_b: object            # XGBPath branche B
    t_stat_a: float
    t_stat_b: float
    wracc_union: float


def evaluate_or_pairs(
    paths: list,
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    min_branch_t_stat: float = 2.0,
    max_pairs: int = 5,
) -> list[OrCandidate]:
    """Cherche les paires de chemins complémentaires (mécanismes distincts).

    Critères (fondés DEP + regime-switching, voir docstring module) :
      - même signe de score (même direction économique)
      - features dominants différents (deux mécanismes, pas un)
      - chaque branche : |t-stat| >= min_branch_t_stat sur train
      - union : WRAcc > max(WRAcc(branche A), WRAcc(branche B)) > 0
        (la disjonction doit apporter quelque chose que chaque branche
        seule n'a pas — critère de minimalité des DEP)

    Returns:
        Candidats triés par WRAcc union décroissant.
    """
    name_to_idx = {n: i for i, n in enumerate(feature_names)}
    y = y_train.astype(np.float64)

    def eval_path(p):
        """Retourne (mask, t_stat, wracc) d'un chemin sur le train."""
        mask = np.ones(len(y), dtype=bool)
        for feat, op, val in p.conditions:
            idx = name_to_idx.get(feat)
            if idx is None:
                return None, 0.0, 0.0
            col = X_train[:, idx]
            if op == "<":
                mask &= col < val
            elif op == "<=":
                mask &= col <= val
            elif op == ">":
                mask &= col > val
            elif op == ">=":
                mask &= col >= val
            else:
                return None, 0.0, 0.0
        if not mask.any():
            return None, 0.0, 0.0
        return mask, branch_t_stat(y, mask), wracc(mask, y)

    evaluated = []
    for p in paths:
        if not p.conditions:
            continue
        mask, tstat, w = eval_path(p)
        if mask is None or abs(tstat) < min_branch_t_stat:
            continue
        evaluated.append((p, mask, tstat, w))

    candidates: list[OrCandidate] = []
    for (pa, ma, ta, wa), (pb, mb, tb, wb) in itertools.combinations(evaluated, 2):
        # même direction économique
        if np.sign(pa.score) != np.sign(pb.score):
            continue
        # mécanismes distincts : feature dominant différent
        head_a = pa.conditions[0][0]
        head_b = pb.conditions[0][0]
        if head_a == head_b:
            continue
        # complémentarité : l'union doit couvrir plus que chaque branche seule
        union = ma | mb
        if union.sum() <= max(ma.sum(), mb.sum()):
            continue
        wu = wracc(union, y)
        best_branch_w = max(wa, wb)
        if wu <= best_branch_w or wu <= 0.0:
            continue
        candidates.append(OrCandidate(pa, pb, ta, tb, wu))

    candidates.sort(key=lambda c: -c.wracc_union)
    logger.info(
        "or_refiner : %d paires evaluees, %d candidats complementaires",
        len(evaluated) * (len(evaluated) - 1) // 2, len(candidates),
    )
    return candidates[:max_pairs]


def build_or_einher(candidate: OrCandidate, base_einher: Einher) -> Einher:
    """Construit un Einher OR depuis une paire complémentaire.

    L'univers/direction/amplitude sont hérités du contexte du triplet
    (repris du premier Einher généré du run).
    """
    def cond_of(path):
        feat, op, val = path.conditions[0]
        return Condition(feature_ref=feat, operator=op, value=val, transformation=None)

    ast = ConditionNode(op="OR", left=cond_of(candidate.path_a), right=cond_of(candidate.path_b))
    eid = (
        f"or_{base_einher.universe.get('asset', 'X')}_{base_einher.universe.get('timeframe', '')}_"
        f"{base_einher.universe.get('horizon', '')}_{uuid.uuid4().hex[:6]}"
    )
    empty = EinherMetrics(
        n_trades=0, n_tp=0, n_sl=0, n_timeout=0,
        win_rate=0.0, avg_net_return=0.0, total_return=0.0,
        sharpe_ratio=0.0, max_drawdown=0.0, profit_factor=0.0,
        avg_holding_bars=0.0, buy_hold_return=0.0, alpha=0.0,
    )
    return Einher(
        id=eid,
        condition_tree=ast,
        direction=base_einher.direction,
        amplitude_bars=base_einher.amplitude_bars,
        tp_pct=base_einher.tp_pct,
        sl_pct=base_einher.sl_pct,
        universe=dict(base_einher.universe),
        metrics=empty,
        scope=base_einher.scope,
        source={
            "model": "or_regimes",
            "branch_a": {
                "feature": candidate.path_a.conditions[0][0],
                "path_score": float(candidate.path_a.score),
                "t_stat_train": candidate.t_stat_a,
            },
            "branch_b": {
                "feature": candidate.path_b.conditions[0][0],
                "path_score": float(candidate.path_b.score),
                "t_stat_train": candidate.t_stat_b,
            },
            "wracc_union_train": candidate.wracc_union,
            "n_conditions": 2,
        },
        data_version="",
    )


# --------------------------------------------------------------------------- #
# Veto-NOT post-admission
# --------------------------------------------------------------------------- #

VETO_MAX_TRADES_REMOVED = 0.20   # le veto retire au plus 20% des trades val


def find_veto_condition(
    einher: Einher,
    ohlcv_val,
    X_val: np.ndarray,
    feature_names: list[str],
    costs_pct: float,
    backtest_fn,
) -> tuple[Einher, dict] | None:
    """Cherche une condition de veto améliorant le sharpe val d'un Einher admis.

    Un veto est NOT(condition) évalué AVANT l'entrée : seuls les trades dont
    la condition de veto est fausse sont conservés. On teste toutes les
    conditions atomiques simples (feature, op, seuil par quantiles).

    Args:
        einher : Einher ADMIS (avec métriques val remplies).
        ohlcv_val : fenêtre OHLCV val alignée.
        X_val : features val alignées.
        feature_names : noms des colonnes.
        costs_pct : coût round-trip.
        backtest_fn : fonction backtest_einher(einher, ohlcv_df, X, names, costs_pct).

    Returns:
        (Einher avec veto, info dict) si amélioration trouvée, sinon None.
        Le retour contient la condition retenue et les sharpe avant/après.
    """
    base_sharpe = einher.metrics.sharpe_ratio
    base_trades = einher.metrics.n_trades
    if base_trades < 10 or X_val.shape[0] == 0:
        return None

    name_to_idx = {n: i for i, n in enumerate(feature_names)}
    best = None
    tested = 0

    for fname in feature_names:
        idx = name_to_idx.get(fname)
        if idx is None:
            continue
        col = X_val[:, idx].astype(np.float64)
        if np.all(col == col[0]):
            continue  # constante
        qs = np.unique(np.quantile(col, [0.25, 0.5, 0.75]))
        for q in qs:
            for op in ("<", ">"):
                tested += 1
                # mask de veto : True où la condition de blocage est active
                if op == "<":
                    block_mask = col < q
                else:
                    block_mask = col > q
                frac_blocked = float(block_mask.mean())
                if frac_blocked <= 0.02 or frac_blocked >= VETO_MAX_TRADES_REMOVED:
                    continue
                # construire l'AST avec veto : condition AND NOT(veto)
                inner_veto = Condition(feature_ref=fname, operator=op, value=float(q),
                                       transformation=None)
                not_node = ConditionNode(op="NOT", left=inner_veto)
                new_ast = ConditionNode(op="AND", left=einher.condition_tree, right=not_node)
                from dataclasses import replace as dc_replace
                cand = dc_replace(
                    einher,
                    id=f"{einher.id}_veto_{uuid.uuid4().hex[:4]}",
                    condition_tree=new_ast,
                    source={
                        **einher.source,
                        "veto_condition": {"feature": fname, "operator": op, "quantile": float(q)},
                        "veto_base_id": einher.id,
                        "model": einher.source.get("model", "?") + "+veto",
                    },
                )
                res = backtest_fn(
                    einher=cand,
                    ohlcv_df=ohlcv_val,
                    X=X_val,
                    feature_names=feature_names,
                    costs_pct=costs_pct,
                )
                cand = dc_replace(cand, metrics=res.metrics)
                new_trades = res.metrics.n_trades
                removed_frac = 1.0 - (new_trades / base_trades) if base_trades else 1.0
                if (
                    res.metrics.sharpe_ratio > base_sharpe
                    and removed_frac <= VETO_MAX_TRADES_REMOVED
                    and new_trades >= 10
                ):
                    if best is None or res.metrics.sharpe_ratio > best[1]["sharpe_after"]:
                        info = {
                            "veto_feature": fname,
                            "veto_operator": op,
                            "veto_quantile": float(q),
                            "sharpe_before": base_sharpe,
                            "sharpe_after": res.metrics.sharpe_ratio,
                            "trades_before": base_trades,
                            "trades_after": new_trades,
                            "removed_frac": round(removed_frac, 3),
                            "n_conditions_tested": tested,
                        }
                        best = (cand, info)
    if best:
        logger.info(
            "veto_refiner : veto retenu (%s %s q=%.2f) sharpe %.2f -> %.2f",
            best[1]["veto_feature"], best[1]["veto_operator"], best[1]["veto_quantile"],
            best[1]["sharpe_before"], best[1]["sharpe_after"],
        )
    else:
        logger.info("veto_refiner : aucun veto ameliorant (%d conditions testees)", tested)
    return best
