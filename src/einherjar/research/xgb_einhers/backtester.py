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

import numpy as np
import polars as pl

from .condition_tree import (
    evaluate_ast_on_array,
)
from .types import (
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
    """BacktestResult."""
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
    # FIX PERF (2026-08-21) : True Range vectorisé (au lieu de la boucle Python).
    tr = np.zeros(n, dtype=np.float64)
    if n > 0:
        tr[0] = high[0] - low[0]  # équivaut à l'ancienne boucle (tr[0] initial)
    tr[1:] = np.maximum(
        high[1:] - low[1:],
        np.maximum(
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1]),
        ),
    )
    # Wilder smoothing (RMA) : séquentiel, garde la formule exacte
    # atr[i] = (atr[i-1]*(period-1)+tr[i])/period.
    atr = np.full(n, np.nan, dtype=np.float64)
    if n < period:
        return atr
    atr[period - 1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr
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
            highs: TODO: documenter.
            lows: TODO: documenter.
            opens: TODO: documenter.

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
        low_ = lows[idx]
        if direction == "BUY":
            tp_hit = h >= tp_price
            sl_hit = low_ <= sl_price
        else:
            tp_hit = low_ <= tp_price
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

    win_rate = float((rets > 0).mean()) if n > 0 else 0.0   # FIX: inclut timeouts gagnants
    tp_hit_rate = n_tp / n if n > 0 else 0.0
    avg_net = float(np.mean(rets)) if n > 0 else 0.0
    # FIX METRICS (2026-08-21) : total_return COMPOSE, pas une somme.
    # sum(+10%,+10%,+10%) = 30% mais prod(1.1^3)-1 = 33.1%.
    total = float(np.prod(1.0 + rets) - 1.0) if n > 0 else 0.0

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
    # FIX P1-1 (AI Review 2026-08-20) : p-value one-sided upper tail.
    # H0: mean(rets) <= 0 (pas H0: mean = 0).
    # En bilaterale, une strat perdante (t=-3.5) avait p=0.0005 et volait
    # le quota BH aux strategies gagnantes. En one-sided upper :
    #   t <= 0  -> p = 1.0 (ne peut pas rejeter H0)
    #   t >  0  -> p = 1 - Phi(t)
    if n > 1 and std > 0 and not degenerate:
        t_stat = float(avg_net / (std / np.sqrt(n)))
        from math import erf, sqrt
        # Tolerance pour eviter les artefacts de precision flottante
        # (mean=0 mais t_stat=1e-17 a cause de float64)
        if t_stat <= 1e-9:
            # t <= epsilon : ne peut pas rejeter H0: mu <= 0
            p_val = 1.0
        else:
            # One-sided upper tail : P(X > t_stat | H0)
            p_val = 1.0 - 0.5 * (1.0 + erf(t_stat / sqrt(2.0)))
            p_val = max(p_val, 1e-10)
    else:
        t_stat = 0.0
        p_val = 1.0

    # Max drawdown sur EQUITY COMPOSEE (FIX METRICS 2026-08-21).
    # Avant : eq = cumsum(rets) (approche additive, fausse pour gros rendements).
    # Apres : eq = cumprod(1+rets), sauf si certains rets <= -1 (protection).
    eq_comp = np.cumprod(np.maximum(1.0 + rets, 1e-9))
    peak = np.maximum.accumulate(eq_comp)
    dd = eq_comp - peak
    # drawdown en fraction : (eq - peak)/peak
    dd_frac = dd / np.maximum(peak, 1e-12)
    max_dd = float(np.min(dd_frac)) if len(dd_frac) > 0 else 0.0

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
        alpha=total - buy_hold_return,   # excess return vs buy&hold (composé)
        t_statistic=t_stat,
        p_value=p_val,
        trade_returns=tuple(rets.tolist()),
        tp_hit_rate=tp_hit_rate,
    )




def backtest_einher_multi(
    einher: Einher,
    per_asset: list[tuple[pl.DataFrame, np.ndarray]],
    feature_names: list[str],
    costs_pct: float = 0.0010,
    val_frac: float = 0.6,
    holdout_embargo: int = 50,
    phase: str = "val",
) -> BacktestResult:
    """Backtest multi-actif : evalue l'Einher sur CHACUN des actifs du scope.

    (per_asset = liste de (ohlcv_df, X) alignes), puis AGREGGE les trades.

    FIX BUG-1 (2026-08-21) : avant, en scope=market/general, le modele etait
    entraine multi-actif mais le backtest ne tournait que sur l'actif primaire.
    Les metriques d'admission ne refletent alors qu'UN actif de l'univers.
    Ici on backteste sur tous les actifs et on fusionne les trades (meme split
    temporel val 60-80% applique par actif), puis on recalcule les metriques
    sur l'union. Retourne aussi la metrique de l'actif primaire pour reference.

    Returns:
        BacktestResult agr e g e (trades = union des trades val sur tous actifs).
    """
    all_trades: list[TradeResult] = []
    primary_result = None
    n_aligned_list = [len(o) for o, _ in per_asset]
    sum(n_aligned_list)
    # Bornes temporelles communes (proportionnelles a chaque serie)
    def _phase_slice(n):
        """_phase_slice.

        Args:
            n: TODO document.
        """
        te = int(n * val_frac)
        ve_1 = min(n, te + int(n * 0.2))
        hs = ve_1 + max(holdout_embargo, einher.amplitude_bars)
        if phase == "val":
            vs = te + max(holdout_embargo, einher.amplitude_bars)
            ve = min(n, vs + int(n * 0.2))
            return vs, ve
        else:  # holdout
            return hs, n
    for ohlcv_df, X in per_asset:
        n_this = len(ohlcv_df)
        if n_this == 0:
            continue
        vs, ve = _phase_slice(n_this)
        if vs < ve:
            r = backtest_einher(einher, ohlcv_df[vs:ve], X[vs:ve], feature_names, costs_pct)
        else:
            r = backtest_einher(einher, ohlcv_df[:0], X[:0], feature_names, costs_pct)
        if primary_result is None:
            primary_result = r
        all_trades.extend(r.trades)
    if not all_trades:
        # pas de trades : on retourne des metriques vides (val)
        return primary_result if primary_result else BacktestResult(
            trades=[],
            metrics=(  # pyright: ignore[reportArgumentType]
                backtest_einher(einher, per_asset[0][0][:0], per_asset[0][1][:0], feature_names, costs_pct).metrics
                if per_asset
                else None
            ),
            equity_curve=np.array([0.0]))
    # Recalculer les metriques sur l'union (agreg)
    # Recalculer les metriques sur l'union (agreg)
    buy_hold = 0.0
    import numpy as _np
    # FIX : estimer les annees a partir des timestamps reels (ms) des trades
    _ts_min = min(t.entry_timestamp_ms for t in all_trades)
    _ts_max = max(t.exit_timestamp_ms for t in all_trades)
    dur_h = (_ts_max - _ts_min) / 3_600_000.0 if _ts_max > _ts_min else 1.0
    years = max(dur_h / 8_760.0, 1.0 / 365.0)  # minimum ~1 jour
    metrics = compute_metrics(all_trades, buy_hold, years_in_period=years)
    _tp = primary_result.effective_tp_pct if primary_result else 0.0
    _sl = primary_result.effective_sl_pct if primary_result else 0.0
    return BacktestResult(
        trades=all_trades, metrics=metrics,
        equity_curve=_np.concatenate([[0.0], _np.cumsum([t.net_return for t in all_trades])]),
        effective_tp_pct=_tp, effective_sl_pct=_sl,
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
    # FIX P0-1 (AI Review 2026-08-20) : défauts dynamiques ATR-based au lieu
    # de hardcoder 2.5%/1.5% (qui tuent forex/indices où 6h = 0.05-0.20%).
    if einher.tp_pct > 0 and einher.sl_pct > 0:
        tp_pct = einher.tp_pct
        sl_pct = einher.sl_pct
    else:
        # Calcul median ATR sur la serie
        atr_arr = compute_atr(highs, lows, closes, period=atr_period)
        valid_atr = atr_arr[~np.isnan(atr_arr)]
        if len(valid_atr) > 0:
            median_atr = float(np.median(valid_atr))
            median_price = float(np.median(closes))
            median_atr_pct = median_atr / max(median_price, 1e-12)
        else:
            # Fallback : 1% du prix median
            median_atr_pct = 0.01
        # Scale TP/SL avec l'amplitude (sqrt scaling)
        horizon_factor = float(np.sqrt(max(1.0, einher.amplitude_bars / 6.0)))
        # Planchers adaptes par classe de vol
        tp_pct = max(0.0010, 2.5 * median_atr_pct * horizon_factor)
        sl_pct = max(0.0005, 1.5 * median_atr_pct * horizon_factor)
        logger.debug(
            "ATR-based SL/TP : median_atr=%.6f, median_price=%.4f, "
            "median_atr_pct=%.5f, horizon_factor=%.2f -> tp=%.5f sl=%.5f",
            median_atr, median_price, median_atr_pct, horizon_factor, tp_pct, sl_pct,
        )

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
