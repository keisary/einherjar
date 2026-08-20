"""descriptors.py — Descripteurs MAP-Elites (plan lignes 401-408).

Axes choisis (validés Jovanny, Option A) :
- direction        : BUY / SELL (l'axe sépare naturellement les cellules)
- famille dominante: famille économique (taxonomie feature_taxonomy_corrected.json,
                      champ economic_family) la plus fréquente dans la condition
- régime volatilité: volatilité annualisée du marché sur la fenêtre évaluée
                     (low_vol < 40% < high_vol)

Les descripteurs NE recopient PAS la fitness (le régime est calculé sur le
marché, pas sur la stratégie — aucune fuite de la qualité).
"""
from __future__ import annotations

import numpy as np
import polars as pl

from einherjar.research.search_engine.expression import BoolOp, Cmp, collect_features, Feature

VOL_REGIME_THRESHOLD = 0.40  # volatilité annualisée (decimal)

FAMILY_OTHER = "other"


def dominant_family(expr: object, taxonomy: dict) -> str:
    """Famille économique dominante des features de la condition."""
    features = collect_features(expr)
    if not features:
        return FAMILY_OTHER
    counts: dict[str, int] = {}
    for f in features:
        fam = taxonomy.get(f, {}).get("economic_family", FAMILY_OTHER)
        counts[fam] = counts.get(fam, 0) + 1
    # Tie-break déterministe : ordre alphabétique des familles
    best = max(sorted(counts), key=counts.get)
    return best


def market_regime(ohlcv_sub: pl.DataFrame) -> str:
    """Régime de volatilité annualisée de la fenêtre (retours sur close)."""
    closes = ohlcv_sub["close"].to_numpy().astype(np.float64)
    if len(closes) < 30:
        return "low_vol"
    rets = np.diff(closes) / closes[:-1]
    rets = rets[np.isfinite(rets)]
    if len(rets) < 30:
        return "low_vol"
    ann_vol = float(np.std(rets, ddof=1) * np.sqrt(365 * 24))  # données horaires
    return "high_vol" if ann_vol > VOL_REGIME_THRESHOLD else "low_vol"


def describe(expr: object, direction: str, ohlcv_sub: pl.DataFrame, taxonomy: dict) -> tuple[str, str, str]:
    """Descripteur (direction, famille dominante, régime volatilité)."""
    return (direction, dominant_family(expr, taxonomy), market_regime(ohlcv_sub))