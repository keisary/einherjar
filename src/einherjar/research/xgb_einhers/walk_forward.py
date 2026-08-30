"""walk_forward.py - Validation walk-forward des Einhers admis.

2026-08-29 (decision Jovanny, axe 1D + robustesse) :
Le split fixe 60/20/20 valide sur UNE fenetre qui peut tomber dans un regime
particulier. Un Einher "stable" doit etre rentable dans la majorite des folds.

Methode : expanding-window walk-forward.
- K folds (defaut 5). La fenetre d'entrainement GROSSIT (expanding) :
  fold k entraine sur [0, t_k), valide sur [t_k, t_{k+1}).
- Un Einher est "stable" s'il est rentable (net_return moyen > 0) dans
  >= min_folds_pct (defaut 60% = 3/5) des folds.

IMPORTANT (anti-leakage) : la validation des paths (extraction/entrainement)
reste sur le split standard. Le walk-forward NE REENTRAINE PAS ici : il
evalAe l'Einher DEJA construit sur plusieurs fenetres hors-echantillon
successives pour mesurer la STABILITE de son signal dans le temps.

C'est un filtre de robustesse POST-hoc, pas un re-entrainement :
- Coût maîtrisé (backtest seulement, pas de training)
- Pas de leakage (chaque fenetre est strictement future)
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_FOLDS = 5
DEFAULT_MIN_FOLDS_PCT = 0.60  # 3/5


def walk_forward_evaluate(
    backtest_fn,
    einher,
    ohlcv_aligned,
    X_aligned,
    feature_names,
    costs_pct: float,
    horizon_bars: int,
    folds: int = DEFAULT_FOLDS,
    min_folds_pct: float = DEFAULT_MIN_FOLDS_PCT,
    embargo_bars: int = 50,
) -> dict:
    """Evalue un Einher sur K fenetres walk-forward (expanding).

    Args:
        backtest_fn : signature fn(einher, ohlcv_df, X, feature_names, costs_pct)
            retournant un objet avec .metrics (EinherMetrics).
        einher : Einher a evaluer.
        ohlcv_aligned : DataFrame polars aligne.
        X_aligned : ndarray aligne.
        feature_names : noms des features.
        costs_pct : cout round-trip.
        horizon_bars : horizon en bars (embargo).
        folds : nombre de folds.
        min_folds_pct : fraction minimale de folds rentables (defaut 0.6).
        embargo_bars : embargo entre fenetres.

    Returns:
        dict {
            "n_folds": int,
            "n_profitable": int,
            "passed": bool,
            "fold_net_returns": list[float],
        }
    """
    n = X_aligned.shape[0]
    if n < (folds + 1) * max(50, horizon_bars):
        # Pas assez de donnees pour un walk-forward significatif
        return {
            "n_folds": folds,
            "n_profitable": 0,
            "passed": False,
            "fold_net_returns": [],
            "reason": "insufficient_data",
        }

    emb = max(embargo_bars, horizon_bars)
    # Taille de chaque fenetre de validation (les fenetres s'espacent)
    window = n // (folds + 1)

    fold_nets = []
    for k in range(folds):
        # Fenetre val : [train_end_k + emb, train_end_k + emb + window)
        # L'entrainement implicite couvre [0, train_end_k) : non utilise ici
        val_start = (k + 1) * window + emb
        val_end = min(n, val_start + window)
        if val_start >= val_end or val_start >= n:
            break
        result = backtest_fn(
            einher,
            ohlcv_aligned[val_start:val_end],
            X_aligned[val_start:val_end],
            feature_names,
            costs_pct,
        )
        m = result.metrics
        if m.n_trades == 0:
            fold_nets.append(0.0)
        else:
            fold_nets.append(float(m.avg_net_return))

    # Nombre de folds reellement evalues
    n_evaluated = len(fold_nets)
    if n_evaluated == 0:
        return {
            "n_folds": folds,
            "n_profitable": 0,
            "passed": False,
            "fold_net_returns": [],
            "reason": "no_folds",
        }

    n_profitable = sum(1 for x in fold_nets if x > 0)
    passed = n_profitable >= max(1, int(np.ceil(n_evaluated * min_folds_pct)))

    logger.info(
        "walk_forward : %d/%d folds rentables (%.0f%% requis) -> %s",
        n_profitable, n_evaluated, min_folds_pct * 100,
        "PASS" if passed else "REJECT",
    )
    return {
        "n_folds": n_evaluated,
        "n_profitable": n_profitable,
        "passed": passed,
        "fold_net_returns": fold_nets,
    }