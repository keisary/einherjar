"""einher_builder.py - Construction d'un Einher depuis un XGBPath.

Réponse Q8 : direction BUY si score > 0, SELL si score < 0, skip si ≈ 0.
Réponse Q13 : amplitude FIXE par horizon XGBoost (1 modèle = 1 horizon).
Réponse Q14 : SL/TP en multiple d'ATR (stratégie A du plan), pas basé sur Y_ret prédit.

Règle anti-tautologie : on n'utilise PAS Y_ret prédit comme TP/SL pour éviter
que le backtest ne soit "trop parfait" par construction (overfit garanti).
"""
from __future__ import annotations

import logging
import uuid
from typing import Any

import numpy as np

from einherjar.research.xgb_einhers.condition_tree import path_to_ast
from einherjar.research.xgb_einhers.path_extractor import XGBPath
from einherjar.research.xgb_einhers.types import (
    Condition,
    ConditionNode,
    Einher,
    EinherMetrics,
)

logger = logging.getLogger(__name__)


# Sprint 3.5 : seuil minimum pour la direction (BUY/SELL)
# Avant : 0.003 (0.3%) - trop restrictif
# Apres : 0.0005 (0.05%) - aligne sur min_score
MIN_ABS_SCORE_FOR_DIRECTION = 0.0005


def build_einher_from_path(
    path: XGBPath,
    asset: str,
    asset_class: str,
    timeframe: str,
    horizon_str: str,
    horizon_bars: int,
    tp_atr_mult: float = 2.5,
    sl_atr_mult: float = 1.5,
    data_version: str = "",
    min_abs_score: float = MIN_ABS_SCORE_FOR_DIRECTION,
) -> Einher | None:
    """Construit un Einher depuis un XGBPath.

    Si |score| < min_abs_score, retourne None (signal trop faible).

    Args:
        path : XGBPath
        asset : 'BTCUSD'
        asset_class : 'crypto'
        timeframe : '1h'
        horizon_str : '6h' (nom de l'horizon)
        horizon_bars : 6 (horizon en bars)
        tp_atr_mult : multiple d'ATR pour TP (défaut 2.5)
        sl_atr_mult : multiple d'ATR pour SL (défaut 1.5)
        data_version : hash de la version de données
        min_abs_score : score absolu minimum pour BUY/SELL (Sprint 2.3)

    Returns:
        Einher ou None si signal trop faible
    """
    # Direction depuis le signe du score
    if path.score > min_abs_score:
        direction = "BUY"
    elif path.score < -min_abs_score:
        direction = "SELL"
    else:
        return None  # Signal trop faible, on skip

    # AST de la condition
    ast = path_to_ast(path)

    # ID unique
    einher_id = f"xgb_{asset}_{timeframe}_{horizon_str}_{path.tree_idx:04d}_{path.path_idx:04d}_{uuid.uuid4().hex[:6]}"

    # Métriques vides pour l'instant (seront remplies par le backtester)
    empty_metrics = EinherMetrics(
        n_trades=0, n_tp=0, n_sl=0, n_timeout=0,
        win_rate=0.0, avg_net_return=0.0, total_return=0.0,
        sharpe_ratio=0.0, max_drawdown=0.0, profit_factor=0.0,
        avg_holding_bars=0.0, buy_hold_return=0.0, alpha=0.0,
    )

    einher = Einher(
        id=einher_id,
        condition_tree=ast,
        direction=direction,
        amplitude_bars=horizon_bars,
        tp_pct=0.0,     # Sera calculé en ATR-based lors du backtest
        sl_pct=0.0,     # Sera calculé en ATR-based lors du backtest
        universe={
            "asset": asset,
            "asset_class": asset_class,
            "timeframe": timeframe,
            "horizon": horizon_str,
            "horizon_bars": horizon_bars,
        },
        metrics=empty_metrics,
        scope="asset",
        source={
            "model": "XGBRegressor",
            "tree_idx": path.tree_idx,
            "path_idx": path.path_idx,
            "path_score": float(path.score),
            "n_conditions": len(path.conditions),
            "feature_names": [c[0] for c in path.conditions],
        },
        data_version=data_version,
    )
    logger.debug(
        "build_einher : %s, dir=%s, score=%.4f, %d conditions",
        einher_id, direction, path.score, len(path.conditions),
    )
    return einher


def set_einher_metrics(einher: Einher, metrics: EinherMetrics) -> Einher:
    """Retourne un nouvel Einher avec les métriques mises à jour (immutabilité)."""
    import dataclasses
    return dataclasses.replace(einher, metrics=metrics)


def set_einher_tp_sl(einher: Einher, tp_pct: float, sl_pct: float) -> Einher:
    """Retourne un nouvel Einher avec les SL/TP mis à jour (immutabilité).

    Corrige le bug où les Einhers étaient sauvegardés avec tp_pct=0 et sl_pct=0,
    alors que le backtester utilisait des valeurs par défaut (2.5%/1.5%).
    """
    import dataclasses
    return dataclasses.replace(einher, tp_pct=float(tp_pct), sl_pct=float(sl_pct))


def set_einher_holdout_metrics(einher: Einher, holdout_metrics) -> Einher:
    """Sprint 2.4.1 : attache les métriques du holdout à l'Einher.

    Permet à l'admission de filtrer les Einhers non significatifs
    (trop peu de trades sur le holdout = variance explose).
    """
    import dataclasses
    return dataclasses.replace(einher, holdout_metrics=holdout_metrics)
