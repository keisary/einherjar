"""random_gen.py — Génération d'Einhers aléatoires (baseline anti-hasard).

Référence A6 du plan : échantillonner l'espace de combinaisons aléatoirement
pour établir la référence que le STGP devra battre.

Règles anti-biais :
- Conditions AND-only, 1 à 3 feuilles (limite plan ligne 318-321).
- Opérateurs `<` / `>` uniquement (comparaisons continues ; `==`/`!=` sont
  inutiles sur des features float).
- Seuils tirés des QUANTILES EMPIRIQUES de la fenêtre TRAIN uniquement
  (aucun lookahead : le train est la seule fenêtre d'exploration).
- Taux de déclenchement contraint [min, max] sur le train : trop rare ->
  condition inutile ; trop fréquent -> équivalent au hasard.
- Direction uniforme BUY/SELL ; amplitude = horizon en bars ; TP/SL = défauts
  du backtester (2.5% / 1.5%, backtester.py:301-308).
"""
from __future__ import annotations

import numpy as np

from einherjar.research.baselines.vector_eval import eval_cond_ast
from einherjar.research.xgb_einhers.types import Condition, ConditionNode, Einher, EinherMetrics

MIN_TRIGGER_RATE = 0.02
MAX_TRIGGER_RATE = 0.70
MIN_AND_TRIGGER_RATE = 0.005
MAX_ATTEMPTS = 12
Q_LO, Q_HI = 0.05, 0.95


def _zero_metrics() -> EinherMetrics:
    """Métriques placeholder (le backtest les remplacera)."""
    return EinherMetrics(
        n_trades=0,
        n_tp=0,
        n_sl=0,
        n_timeout=0,
        win_rate=0.0,
        avg_net_return=0.0,
        total_return=0.0,
        sharpe_ratio=0.0,
        max_drawdown=0.0,
        profit_factor=0.0,
        avg_holding_bars=0.0,
        buy_hold_return=0.0,
        alpha=0.0,
    )


def usable_feature_indices(train_X: np.ndarray) -> list[int]:
    """Indices des features non constantes et sans NaN sur le train."""
    out = []
    for i in range(train_X.shape[1]):
        col = train_X[:, i]
        if np.isnan(col).any():
            continue
        if np.nanstd(col) <= 1e-12:
            continue
        out.append(i)
    return out


def _atomic_condition(
    rng: np.random.Generator,
    feature_indices: list[int],
    train_X: np.ndarray,
    feature_names: list[str],
    min_rate: float,
    max_rate: float,
    max_attempts: int,
) -> Condition:
    """Génère une Condition avec un seuil accepté (taux de déclenchement borné)."""
    for _ in range(max_attempts):
        idx = int(rng.choice(feature_indices))
        side_lt = rng.random() < 0.5
        q = float(rng.uniform(Q_LO, Q_HI))
        threshold = float(np.quantile(train_X[:, idx], q))
        op = "<" if side_lt else ">"
        condition = Condition(feature_ref=feature_names[idx], operator=op, value=threshold)
        rate = float(eval_cond_ast(condition, train_X, feature_names).mean())
        if min_rate <= rate <= max_rate:
            return condition
    # Aucun seuil acceptable : on garde la dernière tentative (toujours aléatoire)
    return condition


def _random_condition(
    rng: np.random.Generator,
    feature_indices: list[int],
    train_X: np.ndarray,
    feature_names: list[str],
    min_rate: float,
    max_rate: float,
    max_attempts: int,
) -> Condition | ConditionNode:
    """Génère une condition AND de 1 à 3 feuilles, taux global >= MIN_AND_TRIGGER_RATE."""
    n_conds = int(rng.integers(1, 4))
    for _ in range(max_attempts):
        conds = [
            _atomic_condition(rng, feature_indices, train_X, feature_names, min_rate, max_rate, max_attempts)
            for _ in range(n_conds)
        ]
        if len(conds) == 1:
            ast: Condition | ConditionNode = conds[0]
        else:
            ast = conds[0]
            for c in conds[1:]:
                ast = ConditionNode(op="AND", left=ast, right=c)
        rate = float(eval_cond_ast(ast, train_X, feature_names).mean())
        if rate >= MIN_AND_TRIGGER_RATE:
            return ast
    return ast


def generate_random_einhers(
    rng: np.random.Generator,
    n: int,
    asset: str,
    asset_class: str,
    timeframe: str,
    horizon: str,
    horizon_bars: int,
    feature_names: list[str],
    train_X: np.ndarray,
    tag: str = "baseline_random",
) -> list[Einher]:
    """Génère n Einhers aléatoires (conditions AND, seuils du train uniquement).

    Args:
        rng : générateur numpy (seed fixe pour reproductibilité).
        n : nombre d'Einhers.
        asset, asset_class, timeframe, horizon : univers de l'Einher.
        horizon_bars : amplitude (nb de bougies).
        feature_names : noms des colonnes de X.
        train_X : fenêtre TRAIN (pour les quantiles ET le taux de déclenchement).
        tag : provenance (source["kind"]).

    Returns:
        Liste de n Einhers avec metrics placeholder (à backtester).
    """
    feature_indices = usable_feature_indices(train_X)
    if not feature_indices:
        raise ValueError("Aucune feature utilisable sur la fenêtre train")

    einhers = []
    universe = {
        "asset": asset,
        "asset_class": asset_class,
        "timeframe": timeframe,
        "horizon": horizon,
        "horizon_bars": horizon_bars,
    }
    for i in range(n):
        ast = _random_condition(
            rng, feature_indices, train_X, feature_names,
            MIN_TRIGGER_RATE, MAX_TRIGGER_RATE, MAX_ATTEMPTS,
        )
        direction = "BUY" if rng.random() < 0.5 else "SELL"
        einhers.append(
            Einher(
                id=f"bl_{asset}_{timeframe}_{horizon}_{i:04d}",
                condition_tree=ast,
                direction=direction,
                amplitude_bars=horizon_bars,
                tp_pct=0.0,
                sl_pct=0.0,
                universe=universe,
                metrics=_zero_metrics(),
                scope="asset",
                source={"kind": tag, "seed": int(rng.integers(0, 2**31))},
            )
        )
    return einhers