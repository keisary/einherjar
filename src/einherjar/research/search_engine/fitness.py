"""fitness.py — Fitness CHEAP de la recherche (plan lignes 449-450).

Principe : Sharpe annuel NET de coûts, calculé sur un ÉCHANTILLON aléatoire
de la fenêtre de validation, SANS bootstrap (le bootstrap est réservé à
l'admission C6). L'échantillon est un bloc contigu tiré aléatoirement
(cohérent avec l'entrée à OPEN[t+1] et l'historique des positions), déterministe
pour un seed donné.

Remarque : un seul bloc biaise le régime ; c'est assumé pour la fitness CHEAP —
la validation lourde (C1-C6) se fait ensuite sur la fenêtre de validation
COMPLÈTE avant toute admission au corpus.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from einherjar.research.search_engine.builder import build_einher
from einherjar.research.xgb_einhers.backtester import backtest_einher


def cheap_fitness(
    expr: object,
    direction: str,
    amplitude_bars: int,
    universe: dict,
    ohlcv_df: pl.DataFrame,
    X: np.ndarray,
    feature_names: list[str],
    rng: np.random.Generator,
    *,
    costs_pct: float,
    sample_frac: float = 0.5,
    data_version: str = "",
) -> tuple[float, object]:
    """Backtest sur un bloc contigu aléatoire; retourne (sharpe_net, einher, sub).

    sub = le sous-échantillon évalué (pour les descripteurs).
    """
    einher = build_einher(
        expr, direction, amplitude_bars, universe,
        costs_pct=costs_pct, data_version=data_version,
    )
    n = len(ohlcv_df)
    block = max(2, int(n * sample_frac))
    start = int(rng.integers(0, n - block + 1))
    sub = ohlcv_df[start : start + block]
    res = backtest_einher(
        einher, sub, X[start : start + block], feature_names, costs_pct=costs_pct,
    )
    sharpe = float(res.metrics.sharpe_ratio)
    # Anti trigger-rate : un arbre trop spécifique ne trade quasi jamais hors
    # de l'échantillon (0 trade en val complète). Pénalité lourde au lieu d'une
    # fitness flat (convention baselines : taux de déclenchement borné).
    if res.metrics.n_trades < 30:
        sharpe = -10.0
    return sharpe, einher, sub
