"""engine/simulator.py — Simulation intrabar TP/SL (S-2 d'ONTOLOGY.md).

Implémente la simulation déterministe d'un trade :
  - Entrée à l'OPEN de la bougie t+1 (jamais au close de t)
  - Test TP/SL sur high/low de chaque bougie de la fenêtre [t+1, t+N]
  - Convention : SL touché avant TP sur la même bougie (conservateur)
  - Retourne (exit_price, exit_reason, mfe, mae)
"""

from __future__ import annotations

from collections.abc import Sequence

from einherjar.research.utils.types import Direction, ExitReason


def simulate_long(
    entry: float,
    sl: float,
    tp: float,
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
) -> tuple[float, ExitReason, float, float, int]:
    """Simule un trade LONG.

    Args:
        entry: prix d'entrée (OPEN de t+1)
        sl: prix du stop-loss (entry - sl_distance)
        tp: prix du take-profit (entry + tp_distance)
        highs: high de chaque bougie de la fenêtre [t+1, t+N]
        lows: low de chaque bougie de la fenêtre
        closes: close de chaque bougie

    Returns:
        (exit_price, exit_reason, mfe, mae, n_bougies_held)
    """
    mfe = 0.0
    mae = 0.0
    for i, (h, l, c) in enumerate(zip(highs, lows, closes)):
        cur_mfe = h - entry
        cur_mae = entry - l
        if cur_mfe > mfe:
            mfe = cur_mfe
        if cur_mae > mae:
            mae = cur_mae
        # Convention : SL avant TP sur la même bougie
        if l <= sl:
            return sl, ExitReason.SL, mfe, entry - l, i + 1
        if h >= tp:
            return tp, ExitReason.TP, max(h, entry) - entry, entry - min(l, entry), i + 1
    return closes[-1], ExitReason.TIMEOUT, mfe, mae, len(closes)


def simulate_short(
    entry: float,
    sl: float,
    tp: float,
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
) -> tuple[float, ExitReason, float, float, int]:
    """Simule un trade SHORT.

    Pour un short, sl > entry (le prix monte contre nous) et tp < entry.
    """
    mfe = 0.0
    mae = 0.0
    for i, (h, l, c) in enumerate(zip(highs, lows, closes)):
        cur_mfe = entry - l
        cur_mae = h - entry
        if cur_mfe > mfe:
            mfe = cur_mfe
        if cur_mae > mae:
            mae = cur_mae
        # Convention : SL avant TP sur la même bougie
        if h >= sl:
            return sl, ExitReason.SL, mfe, h - entry, i + 1
        if l <= tp:
            return tp, ExitReason.TP, max(entry - l, 0.0), max(h - entry, 0.0), i + 1
    return closes[-1], ExitReason.TIMEOUT, mfe, mae, len(closes)


def simulate(
    direction: Direction,
    entry: float,
    sl_price: float,
    tp_price: float,
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
) -> tuple[float, ExitReason, float, float, int]:
    """Dispatch selon la direction."""
    if direction == Direction.LONG:
        return simulate_long(entry, sl_price, tp_price, highs, lows, closes)
    return simulate_short(entry, sl_price, tp_price, highs, lows, closes)
