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
    opens: Sequence[float] | None = None,
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
    opens = opens if opens is not None else closes
    for i, (o, h, l, c) in enumerate(zip(opens, highs, lows, closes)):
        cur_mfe = h - entry
        cur_mae = entry - l
        if cur_mfe > mfe:
            mfe = cur_mfe
        if cur_mae > mae:
            mae = cur_mae
        # Convention : SL avant TP sur la même bougie
        if o <= sl:
            return o, ExitReason.SL, mfe, entry - min(l, entry), i + 1
        if o >= tp:
            return o, ExitReason.TP, max(h, entry) - entry, entry - min(l, entry), i + 1
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
    opens: Sequence[float] | None = None,
) -> tuple[float, ExitReason, float, float, int]:
    """Simule un trade SHORT.

    Pour un short, sl > entry (le prix monte contre nous) et tp < entry.
    """
    mfe = 0.0
    mae = 0.0
    opens = opens if opens is not None else closes
    for i, (o, h, l, c) in enumerate(zip(opens, highs, lows, closes)):
        cur_mfe = entry - l
        cur_mae = h - entry
        if cur_mfe > mfe:
            mfe = cur_mfe
        if cur_mae > mae:
            mae = cur_mae
        # Convention : SL avant TP sur la même bougie
        if o >= sl:
            return o, ExitReason.SL, mfe, max(h - entry, 0.0), i + 1
        if o <= tp:
            return o, ExitReason.TP, max(entry - l, 0.0), max(h - entry, 0.0), i + 1
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
    opens: Sequence[float] | None = None,
) -> tuple[float, ExitReason, float, float, int]:
    """Dispatch selon la direction."""
    if direction == Direction.LONG:
        return simulate_long(entry, sl_price, tp_price, highs, lows, closes, opens)
    return simulate_short(entry, sl_price, tp_price, highs, lows, closes, opens)


def simulate_hold(
    direction: Direction,
    entry: float,
    closes: Sequence[float],
    highs: Sequence[float] | None = None,
    lows: Sequence[float] | None = None,
) -> tuple[float, ExitReason, float, float, int]:
    """Pure hold to end of window — NO SL/TP check.

    Enter at ``entry`` (OPEN of t+1), exit at the LAST close of the window.
    No stop-loss or take-profit is evaluated.  MFE/MAE are still computed
    from high/low for diagnostics.

    Args:
        direction: Trade direction.
        entry: Entry price (OPEN of t+1).
        closes: Close prices over the holding window [t+1 .. t+N].
        highs: High prices (optional, for MFE).
        lows: Low prices (optional, for MAE).

    Returns:
        (exit_price, TIMEOUT, mfe, mae, n_held)
    """
    n = len(closes)
    if n == 0:
        return entry, ExitReason.TIMEOUT, 0.0, 0.0, 0
    exit_price = float(closes[-1])
    if highs is not None and lows is not None and len(highs) == n and len(lows) == n:
        if direction == Direction.LONG:
            mfe = max(highs) - entry
            mae = entry - min(lows)
        else:
            mfe = entry - min(lows)
            mae = max(highs) - entry
    else:
        # Fallback: approximate from closes only
        if direction == Direction.LONG:
            mfe = max(closes) - entry
            mae = entry - min(closes)
        else:
            mfe = entry - min(closes)
            mae = max(closes) - entry
    return exit_price, ExitReason.TIMEOUT, max(mfe, 0.0), max(mae, 0.0), n
