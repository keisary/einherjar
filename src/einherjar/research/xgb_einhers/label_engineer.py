"""label_engineer.py - Construction du target supervisé Y_ret.

Réponse Q8 : on utilise une régression sur Y_ret directement (signed return).
Réponse Q11 : on exclut les bougies invalides (Y_dir == -100).
Réponse Q9 : coûts variables par actif depuis fees_ctrader.json.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .paths import FEES_CONFIG_PATH
from .types import LoadedData

logger = logging.getLogger(__name__)


def build_target(
    loaded: LoadedData,
    horizon_idx: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construit le target supervisé pour un horizon donné.

    Args:
        loaded : LoadedData
        horizon_idx : index de l'horizon (0 à n_horizons-1)

    Returns:
        (target, valid_mask, y_hor) où :
        - target : (N,) float32, Y_ret[:, horizon_idx] (signed return, dans [-0.15, 0.15])
        - valid_mask : (N,) bool, True si Y_dir[:, horizon_idx] != -100
        - y_hor : (N,) float32, Y_hor[:, horizon_idx] (horizon en bars)
    """
    valid_mask = loaded.Y_dir[:, horizon_idx] != -100
    target = loaded.Y_ret[:, horizon_idx].copy()
    y_hor = loaded.Y_hor[:, horizon_idx].copy()
    return target, valid_mask, y_hor


def build_direction_labels(
    loaded: LoadedData,
    horizon_idx: int,
    min_ret_threshold: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Construit un label de direction (BUY / SELL / HOLD) basé sur Y_dir et Y_ret.

    Args:
        loaded : LoadedData
        horizon_idx : index de l'horizon
        min_ret_threshold : Y_ret doit être > threshold (en valeur absolue) pour BUY/SELL

    Returns:
        (direction_labels, valid_mask) où :
        - direction_labels : (N,) int8 dans {-1=SELL, 0=HOLD, 1=BUY, -100=invalide}
        - valid_mask : (N,) bool

    Logique :
    - Si Y_dir == 2 (BUY) ET Y_ret > min_ret_threshold → 1 (BUY)
    - Si Y_dir == 0 (SELL) ET Y_ret < -min_ret_threshold → -1 (SELL)
    - Si Y_dir == 1 (HOLD) → 0 (HOLD)
    - Si Y_dir == -100 → -100 (invalide)
    """
    y_dir = loaded.Y_dir[:, horizon_idx]
    y_ret = loaded.Y_ret[:, horizon_idx]
    labels = np.full(loaded.n_samples, -100, dtype=np.int8)
    valid_mask = y_dir != -100

    # BUY
    buy_mask = (y_dir == 2) & (y_ret > min_ret_threshold)
    labels[buy_mask] = 1
    # SELL
    sell_mask = (y_dir == 0) & (y_ret < -min_ret_threshold)
    labels[sell_mask] = -1
    # HOLD
    hold_mask = (y_dir == 1)
    labels[hold_mask] = 0
    return labels, valid_mask


def load_costs(
    asset: str,
    fees_config_path: Path | None = None,
) -> float:
    """Charge le coût round-trip (decimal) pour un actif depuis fees_ctrader.json.

    FIX COUT (2026-08-21) : la conversion commission $/lot -> % n'etait pas
        implementee (TODO). Le fallback silencieux traitait une commission_per_lot
        de 3.5$ comme 0.0001% (1bp), SOUS-ESTIMANT les coûts reels de tout actif.
        On remplace le fallback NUL par une estimation CONSERVATRICE documentee
        (2bp/leg = 0.0002) lorsque commission_per_lot>0 sans commission_pct, avec
        un warning explicite — plutôt que 1bp sous-estimé ou une erreur qui casse
        tout le run multi-asset.
        - commission_pct explicite (si present) -> utilise tel quel.
        - commission_per_lot present (>0) SANS commission_pct -> estimation
          conservatrice 0.0002/leg + warning (conversion exacte impossible sans
          taille de lot moyenne fiable).
        - ni l'un ni l'autre -> fallback 0.0001 (1bp/leg).

    Returns:
        round_trip_cost : float (ex: 0.0008 pour 0.08%)
    """
    import json
    if fees_config_path is None:
        fees_config_path = FEES_CONFIG_PATH
    with open(fees_config_path) as f:
        fees = json.load(f)

    per_symbol = fees.get("per_symbol", {})
    if asset in per_symbol:
        sym = per_symbol[asset]
        spread = sym.get("spread_pct", 0.0001)
        if "commission_pct" in sym:
            commission_pct = float(sym["commission_pct"])
        else:
            comm_lot = sym.get("commission_per_lot", 0.0) or 0.0
            if comm_lot > 0:
                # FIX COUT : estimation conservatrice (2bp/leg) + warning explicite.
                logger.warning(
                    "%s: commission_per_lot=%.2f$ sans commission_pct -> estimation "
                    "conservatrice 0.0002/leg (conversion exacte impossible sans taille de lot).",
                    asset, comm_lot,
                )
                commission_pct = 0.0002
            else:
                commission_pct = 0.0001
    else:
        spread = fees.get("default", {}).get("spread_pct", 0.0001)
        if "commission_pct" in fees.get("default", {}):
            commission_pct = float(fees["default"]["commission_pct"])
        else:
            comm_lot = fees.get("default", {}).get("commission_per_lot", 0.0) or 0.0
            if comm_lot > 0:
                logger.warning(
                    "%s: default.commission_per_lot=%.2f$ sans commission_pct -> "
                    "estimation conservatrice 0.0002/leg.", asset, comm_lot,
                )
                commission_pct = 0.0002
            else:
                commission_pct = 0.0001

    slippage = 0.0001  # défaut
    # Round-trip = (spread + slippage) * 2 + commission * 2
    return (spread + slippage) * 2 + commission_pct * 2
