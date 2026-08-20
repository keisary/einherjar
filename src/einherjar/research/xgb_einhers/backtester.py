"""backtester.py - NOUVEAU moteur de backtest (remplace le moteur buggé).

Réponse Q14 : le moteur d'évaluation actuel est buggé, on en construit un nouveau.

Principe :
- Entrée à OPEN[t+1] (bougie suivante après le signal)
- Pendant [t+1, t+amplitude] :
  * TP touché avant SL → win
  * SL touché avant TP → loss
  * Sinon → exit à OPEN[t+amplitude] (timeout)
- Convention SL-first sur bougie ambiguë (conservateur)
- Coûts déduits du PnL brut

Anti-lookahead : on n'utilise que des bougies <= t+amplitude.
Anti-bug : on évite toute indexation négative, on borne les indices,
on vérifie les shapes.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from einherjar.research.xgb_einhers.condition_tree import (
    evaluate_ast_on_array,
)
from einherjar.research.xgb_einhers.types import (
    Condition,
    ConditionNode,
    Einher,
    EinherMetrics,
    TradeResult,
)

logger = logging.getLogger(__name__)


# P0 : tests critiques
# 1. test_backtester_no_lookahead
# 2. test_backtester_deterministic
# 3. test_backtester_known_signal


@dataclass
class BacktestResult:
    trades: list[TradeResult]
    metrics: EinherMetrics
    equity_curve: np.ndarray          # (n_trades + 1,) cumsum des net_returns, commence à 0
    effective_tp_pct: float = 0.0     # SL/TP effectivement utilisés (2.1.1 fix)
    effective_sl_pct: float = 0.0


def compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Calcule l'ATR Wilder (period bougies) sur des prix bruts.

    Returns:
        atr : (N,) float64, NaN pour les premières 'period' bougies.
    """
    n = len(high)
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )
    # Wilder smoothing (RMA)
    atr = np.full(n, np.nan, dtype=np.float64)
    if n < period:
        return atr
    # Premier ATR = moyenne simple des period premières TR
    atr[period - 1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def evaluate_signals(
    einher: Einher,
    X: np.ndarray,
    feature_names: list[str],
) -> np.ndarray:
    """Évalue l'AST de l'Einher sur toute la matrice X.

    Returns:
        mask : (N,) bool, True aux indices où la condition est vraie.
    """
    return evaluate_ast_on_array(einher.condition_tree, X, feature_names)


def simulate_trade(
    entry_idx: int,
    amplitude: int,
    direction: str,
    entry_price: float,
    tp_pct: float,
    sl_pct: float,
    highs: np.ndarray,
    lows: np.ndarray,
    opens: np.ndarray,
) -> tuple[float, str, int]:
    """Simule un trade intrabar sur la fenêtre [entry_idx, entry_idx+amplitude-1].

    Convention : si TP et SL touchés sur la même bougie, on prend SL d'abord
    (conservateur).

    Args:
        entry_idx : index d'entrée (= t+1 où t est le signal)
        amplitude : nb de bougies max
        direction : 'BUY' | 'SELL'
        entry_price : prix d'entrée (OPEN[entry_idx])
        tp_pct : take-profit en decimal (ex: 0.025 = 2.5%)
        sl_pct : stop-loss en decimal (ex: 0.015 = 1.5%)
        highs, lows, opens : arrays numpy de l'OHLCV

    Returns:
        (exit_price, exit_reason, n_bars_held) où :
        - exit_reason : 'tp' | 'sl' | 'timeout'
        - n_bars_held : nb de bougies dans la position (>= 1)
    """
    if direction == "BUY":
        tp_price = entry_price * (1 + tp_pct)
        sl_price = entry_price * (1 - sl_pct)
    else:  # SELL
        tp_price = entry_price * (1 - tp_pct)
        sl_price = entry_price * (1 + sl_pct)

    for offset in range(amplitude):
        idx = entry_idx + offset
        h = highs[idx]
        l = lows[idx]
        if direction == "BUY":
            tp_hit = h >= tp_price
            sl_hit = l <= sl_price
        else:
            tp_hit = l <= tp_price
            sl_hit = h >= sl_price
        # Si les deux touchés, convention SL-first
        if sl_hit and tp_hit:
            return sl_price, "sl", offset + 1
        if sl_hit:
            return sl_price, "sl", offset + 1
        if tp_hit:
            return tp_price, "tp", offset + 1
    # Timeout : exit à la dernière bougie au close... mais on n'a pas close ici
    # Convention : exit à OPEN de la bougie suivante (= entry_idx + amplitude)
    next_open_idx = entry_idx + amplitude
    if next_open_idx < len(opens):
        return float(opens[next_open_idx]), "timeout", amplitude
    # Cas dégénéré : on sort au close de la dernière bougie
    return float(opens[entry_idx + amplitude - 1]), "timeout", amplitude


def compute_metrics(
    trades: list[TradeResult],
    buy_hold_return: float,
    years_in_period: float = 1.0,
) -> EinherMetrics:
    """Calcule les métriques d'un Einher depuis la liste de trades.

    Args:
        trades : liste de TradeResult
        buy_hold_return : rendement buy & hold sur la même période (en decimal)
        years_in_period : durée du backtest en années (pour annualisation)

    Returns:
        EinherMetrics
    """
    n = len(trades)
    if n == 0:
        return EinherMetrics(
            n_trades=0, n_tp=0, n_sl=0, n_timeout=0,
            win_rate=0.0, avg_net_return=0.0, total_return=0.0,
            sharpe_ratio=0.0, max_drawdown=0.0, profit_factor=0.0,
            avg_holding_bars=0.0, buy_hold_return=buy_hold_return,
            alpha=0.0 - buy_hold_return,   # alpha négatif si pas de trade
        )

    rets = np.array([t.net_return for t in trades], dtype=np.float64)
    reasons = np.array([t.exit_reason for t in trades])
    n_tp = int((reasons == "tp").sum())
    n_sl = int((reasons == "sl").sum())
    n_timeout = int((reasons == "timeout").sum())

    win_rate = n_tp / n
    avg_net = float(np.mean(rets))
    total = float(np.sum(rets))

    # Sprint 3.0 FIX #1 : Sharpe annualisé CORRECT
    # AVANT (bug) : sharpe = avg_net / std * sqrt(n_trades)
    #   → sqrt(n) gonfle artificiellement le score avec le nombre de trades
    #   → c'est une t-stat, pas un Sharpe annualisé
    # APRES (fix) : sharpe = avg_net / std * sqrt(trades_per_year)
    #   où trades_per_year = n_trades / years_in_period
    std = float(np.std(rets, ddof=1)) if n > 1 else 0.0
    # FIX BASELINE-01 (2026-08-20) : garde anti-degenerescence (std numerique ~0).
    degenerate = std <= 1e-12 * max(1e-12, abs(avg_net))
    if std > 0 and not degenerate and years_in_period > 0:
        trades_per_year = n / years_in_period
        sharpe = float(avg_net / std * np.sqrt(trades_per_year))
    else:
        sharpe = 0.0

    # Sprint 3.3 FIX BUG-02 : vraie t-stat pour correction multi-tests (BH)
    # t = mean(rets) / (std(rets) / sqrt(n))
    # p-value bilaterale H0: mean(rets) = 0
    # On utilise Student t (approx normale si n > 30)
    if n > 1 and std > 0 and not degenerate:
        t_stat = float(avg_net / (std / np.sqrt(n)))
        from math import erf, sqrt
        if n > 30:
            p_val = 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(t_stat) / sqrt(2.0))))
        else:
            p_val = 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(t_stat) / sqrt(2.0))))
        p_val = max(p_val, 1e-10)
    else:
        t_stat = 0.0
        p_val = 1.0

    # Max drawdown sur equity_curve
    eq = np.cumsum(rets)
    peak = np.maximum.accumulate(eq)
    dd = eq - peak
    max_dd = float(np.min(dd)) if len(dd) > 0 else 0.0

    # Profit factor
    gains = rets[rets > 0].sum()
    losses = abs(rets[rets < 0].sum())
    pf = float(gains / losses) if losses > 0 else (float("inf") if gains > 0 else 0.0)

    avg_hold = float(np.mean([t.n_bars_held for t in trades]))

    return EinherMetrics(
        n_trades=n,
        n_tp=n_tp,
        n_sl=n_sl,
        n_timeout=n_timeout,
        win_rate=win_rate,
        avg_net_return=avg_net,
        total_return=total,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        profit_factor=pf,
        avg_holding_bars=avg_hold,
        buy_hold_return=buy_hold_return,
        alpha=total - buy_hold_return,
        t_statistic=t_stat,
        p_value=p_val,
        trade_returns=tuple(rets.tolist()),
    )


def backtest_einher(
    einher: Einher,
    ohlcv_df: pl.DataFrame,
    X: np.ndarray,
    feature_names: list[str],
    costs_pct: float = 0.0010,
    atr_period: int = 14,
) -> BacktestResult:
    """Backtest complet d'un Einher sur OHLCV + features.

    Args:
        einher : Einher avec condition_tree, direction, amplitude_bars, tp_pct, sl_pct
        ohlcv_df : DataFrame polars [timestamp, open, high, low, close, volume]
        X : (N, F) features alignées sur ohlcv_df
        feature_names : noms des colonnes de X
        costs_pct : coût round-trip (decimal, default 0.0010 = 0.10% Sprint 3.0)
                    Ancien default 0.0008 etait sous-estime pour crypto.
                    Realiste crypto : taker 0.05% x 2 = 0.10% + slippage.
        atr_period : période ATR pour calculer SL/TP dynamiques si tp_pct=0

    Returns:
        BacktestResult avec trades, metrics, equity_curve
    """
    n = len(ohlcv_df)
    if len(X) != n:
        raise ValueError(f"X et ohlcv_df ont des longueurs différentes : {len(X)} vs {n}")

    opens = ohlcv_df["open"].to_numpy().astype(np.float64)
    highs = ohlcv_df["high"].to_numpy().astype(np.float64)
    lows = ohlcv_df["low"].to_numpy().astype(np.float64)
    closes = ohlcv_df["close"].to_numpy().astype(np.float64)
    timestamps = ohlcv_df["timestamp"].to_numpy().astype(np.int64)

    # 1. Évaluer les conditions → mask
    signal_mask = evaluate_signals(einher, X, feature_names)
    signal_indices = np.where(signal_mask)[0]
    # Exclure les signaux dont la fenêtre d'amplitude déborde
    valid_mask = signal_indices + einher.amplitude_bars < n
    signal_indices = signal_indices[valid_mask]

    # 2. SL/TP : utiliser ceux de l'Einher directement, ou défauts si 0
    #    (les Einhers issus d'XGBoost ont tp_pct=0, on utilise les défauts ATR)
    if einher.tp_pct > 0:
        tp_pct = einher.tp_pct
    else:
        tp_pct = 0.025  # 2.5% défaut
    if einher.sl_pct > 0:
        sl_pct = einher.sl_pct
    else:
        sl_pct = 0.015  # 1.5% défaut

    # 2.5 Exposer les SL/TP effectifs (pour sauvegarde correcte)
    #    (cf. bug 2.1.1 du Sprint 2.1 : avant, les Einhers étaient sauvés avec 0)
    effective_tp_pct = tp_pct
    effective_sl_pct = sl_pct

    # 3. Simuler chaque trade
    # Sprint 3.3 FIX BUG-06 : tracker in_position pour eviter le stacking
    # Un seul trade ouvert a la fois. Si un signal survient pendant qu'on est
    # deja en position, on l'ignore (pas d'effet de levier implicite).
    trades = []
    in_position_until_idx = -1  # index jusqu'auquel une position est ouverte
    for sig_idx in signal_indices:
        entry_idx = sig_idx + 1  # entrée à OPEN[t+1]
        if entry_idx >= n:
            break
        # FIX BUG-06 : ignorer si on est deja en position
        if entry_idx <= in_position_until_idx:
            continue
        entry_price = float(opens[entry_idx])

        # SL/TP = pourcentage fixe du prix d'entrée (déjà calibré)
        exit_price, exit_reason, n_bars = simulate_trade(
            entry_idx=entry_idx,
            amplitude=einher.amplitude_bars,
            direction=einher.direction,
            entry_price=entry_price,
            tp_pct=tp_pct,
            sl_pct=sl_pct,
            highs=highs,
            lows=lows,
            opens=opens,
        )

        # PnL brut et net
        if einher.direction == "BUY":
            gross = (exit_price - entry_price) / entry_price
        else:
            gross = (entry_price - exit_price) / entry_price
        net = gross - costs_pct

        exit_idx = entry_idx + n_bars - 1
        in_position_until_idx = exit_idx
        trades.append(TradeResult(
            entry_idx=entry_idx,
            exit_idx=exit_idx,
            entry_price=entry_price,
            exit_price=exit_price,
            exit_reason=exit_reason,
            gross_return=gross,
            net_return=net,
            n_bars_held=n_bars,
            entry_timestamp_ms=int(timestamps[entry_idx] // 1_000_000),  # us -> ms
            exit_timestamp_ms=int(timestamps[exit_idx] // 1_000_000),
        ))

    # 5. Buy & hold
    if n > 0:
        buy_hold = (closes[-1] - closes[0]) / closes[0] if closes[0] > 0 else 0.0
    else:
        buy_hold = 0.0

    # 5b. Sprint 3.0 FIX #1 : calculer years_in_period pour annualisation Sharpe
    if n > 1 and len(timestamps) > 1:
        # timestamps en us (polars datetime[us, UTC])
        # durée en heures, puis années
        duration_hours = (timestamps[-1] - timestamps[0]) / 3_600_000_000  # us → h
        years_in_period = duration_hours / 8_760  # 8760h = 1 an
    else:
        years_in_period = 1.0

    # 6. Métriques
    metrics = compute_metrics(trades, buy_hold, years_in_period=years_in_period)

    # 7. Equity curve
    if trades:
        rets = np.array([t.net_return for t in trades])
        equity_curve = np.concatenate([[0.0], np.cumsum(rets)])
    else:
        equity_curve = np.array([0.0])

    return BacktestResult(
        trades=trades, metrics=metrics, equity_curve=equity_curve,
        effective_tp_pct=effective_tp_pct, effective_sl_pct=effective_sl_pct,
    )
