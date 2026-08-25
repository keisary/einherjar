# pyright: reportReturnType=false, reportOptionalCall=false, reportOptionalMemberAccess=false, reportArgumentType=false
"""Module d'indicateurs techniques optimisés avec Numba pour MIDAS V3.

Contient toutes les fonctions de calcul d'indicateurs techniques avec JIT compilation.
"""

import numpy as np
import pandas as pd
import polars as pl

try:
    from numba import jit, njit, prange

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def jit(*args, **kwargs):
        """Jit."""
        def decorator(func):
            """Decorator.

            Args:
            func: TODO document.
            """
            return func

        return decorator if args and callable(args[0]) else decorator

    njit = jit
    prange = range

import gc
import hashlib
import os

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count

# Imports Dask (optionnels)
try:
    import dask
    import dask.dataframe as dd
    from dask.distributed import Client, LocalCluster

    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False
    dask = None
    dd = None
    LocalCluster = None
    Client = None

# Type optimal pour les calculs (float32 pour économiser mémoire)
OPTIMAL_FLOAT = np.float32

# ============================================================================
# CONSTANTES DE CONFIGURATION
# ============================================================================

# Configuration Dask
DASK_CONFIG = {
    "n_workers": min(cpu_count() - 1, 4),  # Limite à 4 workers
    "threads_per_worker": 2,
    "memory_limit": "4GB",
    "scheduler": "threads",
}

# Configuration des chunks
CHUNK_SIZE_THRESHOLD = 1_000_000  # 1M lignes
MIN_CHUNK_SIZE = 100_000  # 100K lignes minimum

# ============================================================================
# FONCTIONS NUMBA OPTIMISÉES POUR INDICATEURS TECHNIQUES
# ============================================================================


@jit(nopython=True, cache=True, parallel=True)
def _numba_ema_vectorized(prices, periods_array):
    """Calcul vectorisé multi-périodes de l'EMA ultra-rapide."""
    n = len(prices)
    num_periods = len(periods_array)
    results = np.full((num_periods, n), np.nan, dtype=OPTIMAL_FLOAT)

    if n < 2:
        return results

    # Calcul pour chaque période
    for p_idx in prange(num_periods):
        period = periods_array[p_idx]

        if period >= n:
            continue

        # Premier EMA = SMA
        results[p_idx, period - 1] = np.mean(prices[:period])

        # Coefficient de lissage
        alpha = 2.0 / (period + 1.0)

        # Calculs EMA suivants
        for i in range(period, n):
            results[p_idx, i] = alpha * prices[i] + (1.0 - alpha) * results[p_idx, i - 1]

    return results


@jit(nopython=True, cache=True, parallel=True)
def _numba_sma_vectorized(prices, periods_array):
    """Calcul vectorisé multi-périodes du SMA ultra-rapide."""
    n = len(prices)
    num_periods = len(periods_array)
    results = np.full((num_periods, n), np.nan, dtype=OPTIMAL_FLOAT)

    if n < 1:
        return results

    # Calcul pour chaque période
    for p_idx in prange(num_periods):
        period = periods_array[p_idx]

        if period >= n:
            continue

        # Calcul SMA pour chaque position
        for i in range(period - 1, n):
            results[p_idx, i] = np.mean(prices[i - period + 1 : i + 1])

    return results


@jit(nopython=True, cache=True, parallel=True)
def _numba_rsi_vectorized(prices, periods_array):
    """Calcul vectorisé multi-périodes du RSI ultra-rapide."""
    n = len(prices)
    num_periods = len(periods_array)
    results = np.full((num_periods, n), np.nan, dtype=OPTIMAL_FLOAT)

    if n < 2:
        return results

    # Calcul des variations
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    # Calcul pour chaque période
    for p_idx in prange(num_periods):
        period = periods_array[p_idx]

        if period >= n:
            continue

        # Premier RSI (SMA)
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])

        if avg_loss == 0:
            results[p_idx, period] = 100.0
        else:
            rs = avg_gain / avg_loss
            results[p_idx, period] = 100.0 - (100.0 / (1.0 + rs))

        # Calculs suivants avec EMA
        for i in range(period + 1, n):
            avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period

            if avg_loss == 0:
                results[p_idx, i] = 100.0
            else:
                rs = avg_gain / avg_loss
                results[p_idx, i] = 100.0 - (100.0 / (1.0 + rs))

    return results


@jit(nopython=True, cache=True)
def _numba_macd_complete(prices, fast=12, slow=26, signal=9):
    """MACD complet ultra-rapide avec ligne de signal et histogramme."""
    n = len(prices)

    # EMA rapide et lente
    alpha_fast = 2.0 / (fast + 1.0)
    alpha_slow = 2.0 / (slow + 1.0)
    alpha_signal = 2.0 / (signal + 1.0)

    ema_fast = np.full(n, np.nan, dtype=OPTIMAL_FLOAT)
    ema_slow = np.full(n, np.nan, dtype=OPTIMAL_FLOAT)
    macd_line = np.full(n, np.nan, dtype=OPTIMAL_FLOAT)
    signal_line = np.full(n, np.nan, dtype=OPTIMAL_FLOAT)
    histogram = np.full(n, np.nan, dtype=OPTIMAL_FLOAT)

    # Initialisation EMA
    ema_fast[fast - 1] = np.mean(prices[:fast])
    ema_slow[slow - 1] = np.mean(prices[:slow])

    # Calcul EMA
    for i in range(fast, n):
        ema_fast[i] = alpha_fast * prices[i] + (1 - alpha_fast) * ema_fast[i - 1]

    for i in range(slow, n):
        ema_slow[i] = alpha_slow * prices[i] + (1 - alpha_slow) * ema_slow[i - 1]

    # MACD line
    for i in range(slow - 1, n):
        if not np.isnan(ema_fast[i]) and not np.isnan(ema_slow[i]):
            macd_line[i] = ema_fast[i] - ema_slow[i]

    # Signal line
    signal_start = slow + signal - 2
    if signal_start < n:
        signal_line[signal_start] = macd_line[signal_start]

        for i in range(signal_start + 1, n):
            if not np.isnan(macd_line[i]):
                signal_line[i] = (
                    alpha_signal * macd_line[i] + (1 - alpha_signal) * signal_line[i - 1]
                )

    # Histogramme
    for i in range(n):
        if not np.isnan(macd_line[i]) and not np.isnan(signal_line[i]):
            histogram[i] = macd_line[i] - signal_line[i]

    return macd_line, signal_line, histogram


@jit(nopython=True, cache=True)
def _numba_bollinger_bands(prices, period=20, std_dev=2.0):
    """Bollinger Bands ultra-rapides."""
    n = len(prices)
    middle = np.full(n, np.nan, dtype=OPTIMAL_FLOAT)
    upper = np.full(n, np.nan, dtype=OPTIMAL_FLOAT)
    lower = np.full(n, np.nan, dtype=OPTIMAL_FLOAT)

    for i in range(period - 1, n):
        window = prices[i - period + 1 : i + 1]
        sma = np.mean(window)
        std = np.std(window)

        middle[i] = sma
        upper[i] = sma + (std_dev * std)
        lower[i] = sma - (std_dev * std)

    return upper, middle, lower


@jit(nopython=True, cache=True)
def _numba_atr(high, low, close, period=14):
    """Average True Range ultra-rapide."""
    n = len(high)
    tr = np.full(n, np.nan, dtype=OPTIMAL_FLOAT)
    atr = np.full(n, np.nan, dtype=OPTIMAL_FLOAT)

    # True Range
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = abs(high[i] - close[i - 1])
        tr3 = abs(low[i] - close[i - 1])
        tr[i] = max(tr1, tr2, tr3)

    tr[0] = high[0] - low[0]  # Premier TR

    # ATR (moyenne mobile du TR)
    atr[period - 1] = np.mean(tr[:period])

    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    return atr


@jit(nopython=True, cache=True)
def _numba_stochastic(high, low, close, k_period=14, d_period=3):
    """Stochastic Oscillator ultra-rapide."""
    n = len(high)
    k_percent = np.full(n, np.nan, dtype=OPTIMAL_FLOAT)
    d_percent = np.full(n, np.nan, dtype=OPTIMAL_FLOAT)

    # %K
    for i in range(k_period - 1, n):
        highest = np.max(high[i - k_period + 1 : i + 1])
        lowest = np.min(low[i - k_period + 1 : i + 1])

        if highest != lowest:
            k_percent[i] = 100.0 * (close[i] - lowest) / (highest - lowest)
        else:
            k_percent[i] = 50.0

    # %D (SMA de %K)
    for i in range(k_period + d_period - 2, n):
        d_percent[i] = np.mean(k_percent[i - d_period + 1 : i + 1])

    return k_percent, d_percent


@jit(nopython=True, cache=True)
def _numba_williams_r(high, low, close, period=14):
    """Williams %R ultra-rapide."""
    n = len(high)
    result = np.full(n, np.nan, dtype=OPTIMAL_FLOAT)

    for i in range(period - 1, n):
        highest = np.max(high[i - period + 1 : i + 1])
        lowest = np.min(low[i - period + 1 : i + 1])

        if highest != lowest:
            result[i] = -100.0 * (highest - close[i]) / (highest - lowest)
        else:
            result[i] = -50.0

    return result


@jit(nopython=True, cache=True)
def _numba_cci(high, low, close, period=20):
    """Commodity Channel Index ultra-rapide."""
    n = len(high)
    result = np.full(n, np.nan, dtype=OPTIMAL_FLOAT)

    # Typical Price
    tp = (high + low + close) / 3.0

    for i in range(period - 1, n):
        window = tp[i - period + 1 : i + 1]
        sma_tp = np.mean(window)
        mean_deviation = np.mean(np.abs(window - sma_tp))

        if mean_deviation != 0:
            result[i] = (tp[i] - sma_tp) / (0.015 * mean_deviation)
        else:
            result[i] = 0.0

    return result


@jit(nopython=True, cache=True)
def _numba_momentum(prices, period=10):
    """Momentum ultra-rapide."""
    n = len(prices)
    result = np.full(n, np.nan, dtype=OPTIMAL_FLOAT)

    for i in range(period, n):
        result[i] = prices[i] - prices[i - period]

    return result


@jit(nopython=True, cache=True)
def _numba_roc(prices, period=12):
    """Rate of Change ultra-rapide."""
    n = len(prices)
    result = np.full(n, np.nan, dtype=OPTIMAL_FLOAT)

    for i in range(period, n):
        if prices[i - period] != 0:
            result[i] = 100.0 * (prices[i] - prices[i - period]) / prices[i - period]
        else:
            result[i] = 0.0

    return result


@jit(nopython=True, cache=True)
def _numba_vwap(prices, volumes):
    """VWAP ultra-rapide."""
    n = len(prices)
    result = np.full(n, np.nan, dtype=OPTIMAL_FLOAT)

    cum_pv = 0.0
    cum_vol = 0.0

    for i in range(n):
        if not np.isnan(volumes[i]) and volumes[i] > 0:
            cum_pv += prices[i] * volumes[i]
            cum_vol += volumes[i]

            if cum_vol > 0:
                result[i] = cum_pv / cum_vol

    return result


@jit(nopython=True, cache=True)
def _numba_twap(prices, period=20):
    """TWAP ultra-rapide."""
    n = len(prices)
    result = np.full(n, np.nan, dtype=OPTIMAL_FLOAT)

    for i in range(period - 1, n):
        result[i] = np.mean(prices[i - period + 1 : i + 1])

    return result


@jit(nopython=True, cache=True)
def _numba_ichimoku_complete(high, low, close, tenkan=9, kijun=26, senkou_b=52, displacement=26):
    """Ichimoku complet ultra-rapide."""
    n = len(high)

    tenkan_sen = np.full(n, np.nan, dtype=OPTIMAL_FLOAT)
    kijun_sen = np.full(n, np.nan, dtype=OPTIMAL_FLOAT)
    senkou_span_a = np.full(n, np.nan, dtype=OPTIMAL_FLOAT)
    senkou_span_b = np.full(n, np.nan, dtype=OPTIMAL_FLOAT)
    chikou_span = np.full(n, np.nan, dtype=OPTIMAL_FLOAT)

    # Tenkan-sen (ligne de conversion)
    for i in range(tenkan - 1, n):
        highest = np.max(high[i - tenkan + 1 : i + 1])
        lowest = np.min(low[i - tenkan + 1 : i + 1])
        tenkan_sen[i] = (highest + lowest) / 2.0

    # Kijun-sen (ligne de base)
    for i in range(kijun - 1, n):
        highest = np.max(high[i - kijun + 1 : i + 1])
        lowest = np.min(low[i - kijun + 1 : i + 1])
        kijun_sen[i] = (highest + lowest) / 2.0

    # Senkou Span A (première ligne du nuage)
    for i in range(kijun - 1, n):
        if not np.isnan(tenkan_sen[i]) and not np.isnan(kijun_sen[i]):
            if i + displacement < n:
                senkou_span_a[i + displacement] = (tenkan_sen[i] + kijun_sen[i]) / 2.0

    # Senkou Span B (deuxième ligne du nuage)
    for i in range(senkou_b - 1, n):
        highest = np.max(high[i - senkou_b + 1 : i + 1])
        lowest = np.min(low[i - senkou_b + 1 : i + 1])
        if i + displacement < n:
            senkou_span_b[i + displacement] = (highest + lowest) / 2.0

    # Chikou Span (ligne de retard)
    for i in range(displacement, n):
        chikou_span[i - displacement] = close[i]

    return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span


@jit(nopython=True, cache=True)
def _numba_keltner_channels(high, low, close, period=20, multiplier=2.0):
    """Keltner Channels ultra-rapides."""
    n = len(high)

    # EMA du prix typique
    typical_price = (high + low + close) / 3.0
    alpha = 2.0 / (period + 1.0)

    ema = np.full(n, np.nan, dtype=OPTIMAL_FLOAT)
    ema[period - 1] = np.mean(typical_price[:period])

    for i in range(period, n):
        ema[i] = alpha * typical_price[i] + (1 - alpha) * ema[i - 1]

    # ATR pour les bandes
    atr = _numba_atr(high, low, close, period)

    upper = np.full(n, np.nan, dtype=OPTIMAL_FLOAT)
    lower = np.full(n, np.nan, dtype=OPTIMAL_FLOAT)

    for i in range(n):
        if not np.isnan(ema[i]) and not np.isnan(atr[i]):
            upper[i] = ema[i] + multiplier * atr[i]
            lower[i] = ema[i] - multiplier * atr[i]

    return upper, ema, lower


@jit(nopython=True, cache=True)
def _numba_donchian_channels(high, low, period=20):
    """Donchian Channels ultra-rapides."""
    n = len(high)
    upper = np.full(n, np.nan, dtype=OPTIMAL_FLOAT)
    lower = np.full(n, np.nan, dtype=OPTIMAL_FLOAT)
    middle = np.full(n, np.nan, dtype=OPTIMAL_FLOAT)

    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1 : i + 1])
        lower[i] = np.min(low[i - period + 1 : i + 1])
        middle[i] = (upper[i] + lower[i]) / 2.0

    return upper, middle, lower


@jit(nopython=True, cache=True)
def _numba_parabolic_sar(high, low, af_start=0.02, af_increment=0.02, af_max=0.2):
    """Parabolic SAR ultra-rapide."""
    n = len(high)
    sar = np.full(n, np.nan, dtype=OPTIMAL_FLOAT)

    if n < 2:
        return sar

    # Initialisation
    trend = 1  # 1 pour haussier, -1 pour baissier
    af = af_start
    ep = high[0]  # Extreme Point
    sar[0] = low[0]

    for i in range(1, n):
        # Calcul SAR
        sar[i] = sar[i - 1] + af * (ep - sar[i - 1])

        # Vérification du changement de tendance
        if trend == 1:  # Tendance haussière
            if low[i] <= sar[i]:
                # Changement vers baissier
                trend = -1
                sar[i] = ep
                ep = low[i]
                af = af_start
            else:
                # Continuer haussier
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + af_increment, af_max)

                # Ajustement SAR
                sar[i] = min(sar[i], low[i - 1])
                if i > 1:
                    sar[i] = min(sar[i], low[i - 2])

        else:  # Tendance baissière
            if high[i] >= sar[i]:
                # Changement vers haussier
                trend = 1
                sar[i] = ep
                ep = high[i]
                af = af_start
            else:
                # Continuer baissier
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + af_increment, af_max)

                # Ajustement SAR
                sar[i] = max(sar[i], high[i - 1])
                if i > 1:
                    sar[i] = max(sar[i], high[i - 2])

    return sar


@jit(nopython=True, cache=True)
def _numba_cmo(prices, period=14):
    """Chande Momentum Oscillator ultra-rapide."""
    n = len(prices)
    result = np.full(n, np.nan, dtype=OPTIMAL_FLOAT)

    if n < period + 1:
        return result

    # Calcul des variations
    deltas = np.diff(prices)

    for i in range(period, n):
        window = deltas[i - period : i]
        sum_up = np.sum(np.where(window > 0, window, 0))
        sum_down = np.sum(np.where(window < 0, -window, 0))

        if sum_up + sum_down != 0:
            result[i] = 100.0 * (sum_up - sum_down) / (sum_up + sum_down)
        else:
            result[i] = 0.0

    return result


@jit(nopython=True, cache=True, parallel=True)
def _numba_volume_indicators(volumes, periods_array):
    """Indicateurs de volume vectorisés."""
    n = len(volumes)
    num_periods = len(periods_array)
    results = np.full((num_periods, n), np.nan, dtype=OPTIMAL_FLOAT)

    for p_idx in prange(num_periods):
        period = periods_array[p_idx]
        for i in range(period - 1, n):
            results[p_idx, i] = np.mean(volumes[i - period + 1 : i + 1])

    return results


@jit(nopython=True, cache=True)
def _numba_adx_complete(high, low, close, period=14):
    """ADX complet avec DI+ et DI- ultra-rapide."""
    n = len(high)

    # Initialiser les arrays
    tr = np.full(n, np.nan, dtype=OPTIMAL_FLOAT)
    dm_plus = np.full(n, np.nan, dtype=OPTIMAL_FLOAT)
    dm_minus = np.full(n, np.nan, dtype=OPTIMAL_FLOAT)

    di_plus = np.full(n, np.nan, dtype=OPTIMAL_FLOAT)
    di_minus = np.full(n, np.nan, dtype=OPTIMAL_FLOAT)
    adx = np.full(n, np.nan, dtype=OPTIMAL_FLOAT)

    # Calcul TR, DM+ et DM-
    for i in range(1, n):
        # True Range
        tr1 = high[i] - low[i]
        tr2 = abs(high[i] - close[i - 1])
        tr3 = abs(low[i] - close[i - 1])
        tr[i] = max(tr1, tr2, tr3)

        # Directional Movement
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]

        if up_move > down_move and up_move > 0:
            dm_plus[i] = up_move
        else:
            dm_plus[i] = 0.0

        if down_move > up_move and down_move > 0:
            dm_minus[i] = down_move
        else:
            dm_minus[i] = 0.0

    # Premier TR
    tr[0] = high[0] - low[0]
    dm_plus[0] = 0.0
    dm_minus[0] = 0.0

    # Calcul des moyennes mobiles
    if n > period:
        # ATR
        atr_sum = np.sum(tr[1 : period + 1])
        atr = atr_sum

        # DM+ et DM- moyennes
        dm_plus_sum = np.sum(dm_plus[1 : period + 1])
        dm_minus_sum = np.sum(dm_minus[1 : period + 1])

        dm_plus_avg = dm_plus_sum
        dm_minus_avg = dm_minus_sum

        # Calcul DI+ et DI-
        if atr != 0:
            di_plus[period] = 100.0 * dm_plus_avg / atr
            di_minus[period] = 100.0 * dm_minus_avg / atr

        # Calculs suivants avec lissage
        for i in range(period + 1, n):
            # ATR lissé
            atr = atr - (atr / period) + tr[i]

            # DM lissés
            dm_plus_avg = dm_plus_avg - (dm_plus_avg / period) + dm_plus[i]
            dm_minus_avg = dm_minus_avg - (dm_minus_avg / period) + dm_minus[i]

            # DI+ et DI-
            if atr != 0:
                di_plus[i] = 100.0 * dm_plus_avg / atr
                di_minus[i] = 100.0 * dm_minus_avg / atr

        # Calcul ADX
        dx_values = np.full(n, np.nan, dtype=OPTIMAL_FLOAT)
        for i in range(period, n):
            if not np.isnan(di_plus[i]) and not np.isnan(di_minus[i]):
                di_sum = di_plus[i] + di_minus[i]
                if di_sum != 0:
                    dx_values[i] = 100.0 * abs(di_plus[i] - di_minus[i]) / di_sum

        # ADX comme moyenne mobile de DX
        if period * 2 < n:
            adx_start = period * 2 - 1
            adx[adx_start] = np.mean(dx_values[period : adx_start + 1])

            for i in range(adx_start + 1, n):
                if not np.isnan(dx_values[i]):
                    adx[i] = (adx[i - 1] * (period - 1) + dx_values[i]) / period

    return adx, di_plus, di_minus


@jit(nopython=True, cache=True)
def _numba_obv(prices, volumes):
    """On Balance Volume ultra-rapide."""
    n = len(prices)
    obv = np.full(n, 0.0, dtype=OPTIMAL_FLOAT)

    if n > 0:
        obv[0] = volumes[0]

        for i in range(1, n):
            if prices[i] > prices[i - 1]:
                obv[i] = obv[i - 1] + volumes[i]
            elif prices[i] < prices[i - 1]:
                obv[i] = obv[i - 1] - volumes[i]
            else:
                obv[i] = obv[i - 1]

    return obv


@jit(nopython=True, cache=True)
def _numba_ema_single(data, period):
    """Calcule une seule EMA pour une série de données, gère les NaNs."""
    n = len(data)
    ema = np.full(n, np.nan, dtype=OPTIMAL_FLOAT)
    alpha = 2.0 / (period + 1.0)

    # Trouver le premier point de départ valide
    start_idx = -1
    for i in range(n):
        if not np.isnan(data[i]):
            start_idx = i
            break

    if start_idx == -1 or start_idx + period > n:
        return ema  # Pas assez de données

    # Initialiser avec SMA
    ema[start_idx + period - 1] = np.mean(data[start_idx : start_idx + period])

    # Calculer l'EMA
    for i in range(start_idx + period, n):
        if not np.isnan(data[i]):
            if np.isnan(ema[i - 1]):
                # Si la valeur précédente est NaN, on réinitialise
                ema[i] = np.mean(data[i - period + 1 : i + 1])
            else:
                ema[i] = alpha * data[i] + (1.0 - alpha) * ema[i - 1]
    return ema


@jit(nopython=True, cache=True)
def _numba_trix(prices, period=14):
    """Calcule TRIX ultra-rapide avec Numba."""
    ema1 = _numba_ema_single(prices, period)
    ema2 = _numba_ema_single(ema1, period)
    ema3 = _numba_ema_single(ema2, period)

    n = len(ema3)
    trix = np.full(n, np.nan, dtype=OPTIMAL_FLOAT)
    for i in range(1, n):
        if not np.isnan(ema3[i - 1]) and ema3[i - 1] != 0:
            trix[i] = 100.0 * (ema3[i] - ema3[i - 1]) / ema3[i - 1]
    return trix


@jit(nopython=True, cache=True)
def _numba_ultimate_oscillator(high, low, close, period1=7, period2=14, period3=28):
    """Calcule l'Ultimate Oscillator ultra-rapide avec Numba."""
    n = len(close)
    result = np.full(n, np.nan, dtype=OPTIMAL_FLOAT)

    bp = np.zeros(n, dtype=OPTIMAL_FLOAT)  # Buying Pressure
    tr = np.zeros(n, dtype=OPTIMAL_FLOAT)  # True Range

    for i in range(1, n):
        prev_close = close[i - 1]
        bp[i] = close[i] - min(low[i], prev_close)
        tr[i] = max(high[i], prev_close) - min(low[i], prev_close)

    max_period = max(period1, period2, period3)
    for i in range(max_period - 1, n):
        sum_bp1 = np.sum(bp[i - period1 + 1 : i + 1])
        sum_tr1 = np.sum(tr[i - period1 + 1 : i + 1])

        sum_bp2 = np.sum(bp[i - period2 + 1 : i + 1])
        sum_tr2 = np.sum(tr[i - period2 + 1 : i + 1])

        sum_bp3 = np.sum(bp[i - period3 + 1 : i + 1])
        sum_tr3 = np.sum(tr[i - period3 + 1 : i + 1])

        avg1 = sum_bp1 / sum_tr1 if sum_tr1 != 0 else 0
        avg2 = sum_bp2 / sum_tr2 if sum_tr2 != 0 else 0
        avg3 = sum_bp3 / sum_tr3 if sum_tr3 != 0 else 0

        result[i] = 100.0 * (4 * avg1 + 2 * avg2 + avg3) / 7.0

    return result


@jit(nopython=True, cache=True)
def _numba_mfi(high, low, close, volume, period=14):
    """Calcule le Money Flow Index ultra-rapide avec Numba."""
    n = len(close)
    result = np.full(n, np.nan, dtype=OPTIMAL_FLOAT)

    tp = (high + low + close) / 3.0

    pos_mf = np.zeros(n, dtype=OPTIMAL_FLOAT)
    neg_mf = np.zeros(n, dtype=OPTIMAL_FLOAT)

    for i in range(1, n):
        if tp[i] > tp[i - 1]:
            pos_mf[i] = tp[i] * volume[i]
        elif tp[i] < tp[i - 1]:
            neg_mf[i] = tp[i] * volume[i]

    for i in range(period, n):
        sum_pos_mf = np.sum(pos_mf[i - period + 1 : i + 1])
        sum_neg_mf = np.sum(neg_mf[i - period + 1 : i + 1])

        if sum_neg_mf != 0:
            mf_ratio = sum_pos_mf / sum_neg_mf
            result[i] = 100.0 - (100.0 / (1.0 + mf_ratio))
        else:
            result[i] = 100.0

    return result


@jit(nopython=True, cache=True)
def _numba_chaikin_oscillator(high, low, close, volume, fast=3, slow=10):
    """Calcule le Chaikin Oscillator ultra-rapide avec Numba."""
    n = len(close)
    adl = np.zeros(n, dtype=OPTIMAL_FLOAT)

    # Calcul de l'Accumulation/Distribution Line (ADL)
    for i in range(n):
        if high[i] != low[i]:
            mfm = ((close[i] - low[i]) - (high[i] - close[i])) / (high[i] - low[i])
            mfv = mfm * volume[i]
            adl[i] = (adl[i - 1] if i > 0 else 0) + mfv

    # EMA de l'ADL
    ema_fast = _numba_ema_single(adl, fast)
    ema_slow = _numba_ema_single(adl, slow)

    return ema_fast - ema_slow


@jit(nopython=True, cache=True)
def _numba_aroon(high, low, period=14):
    """Calcule Aroon Up et Aroon Down ultra-rapide avec Numba."""
    n = len(high)
    aroon_up = np.full(n, np.nan, dtype=OPTIMAL_FLOAT)
    aroon_down = np.full(n, np.nan, dtype=OPTIMAL_FLOAT)

    for i in range(period, n):
        window_high = high[i - period : i + 1]
        window_low = low[i - period : i + 1]

        # Jours depuis le plus haut / plus bas (fenêtre de period+1 éléments)
        days_since_high = period - np.argmax(window_high)
        days_since_low = period - np.argmin(window_low)

        aroon_up[i] = 100.0 * (period - days_since_high) / period
        aroon_down[i] = 100.0 * (period - days_since_low) / period

    return aroon_up, aroon_down


@jit(nopython=True, cache=True)
def _numba_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calcule le SuperTrend ultra-rapide avec Numba."""
    n = len(high)
    supertrend = np.full(n, np.nan, dtype=OPTIMAL_FLOAT)
    trend = np.full(n, 1, dtype=np.int8)  # 1 means Up, -1 means Down

    # ATR calculation components
    tr = np.full(n, np.nan, dtype=OPTIMAL_FLOAT)
    atr = np.full(n, np.nan, dtype=OPTIMAL_FLOAT)

    # Basic Upper/Lower Bands
    basic_upper = np.full(n, np.nan, dtype=OPTIMAL_FLOAT)
    basic_lower = np.full(n, np.nan, dtype=OPTIMAL_FLOAT)
    final_upper = np.full(n, np.nan, dtype=OPTIMAL_FLOAT)
    final_lower = np.full(n, np.nan, dtype=OPTIMAL_FLOAT)

    # 1. Calculate ATR first
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = abs(high[i] - close[i - 1])
        tr3 = abs(low[i] - close[i - 1])
        tr[i] = max(tr1, tr2, tr3)

    atr[period - 1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    # 2. Calculate SuperTrend
    for i in range(period, n):
        hl2 = (high[i] + low[i]) / 2.0

        # Basic bands
        basic_upper[i] = hl2 + (multiplier * atr[i])
        basic_lower[i] = hl2 - (multiplier * atr[i])

        # Initialize final bands
        if i == period:
            final_upper[i] = basic_upper[i]
            final_lower[i] = basic_lower[i]
        else:
            # Final Upper Band
            if (basic_upper[i] < final_upper[i - 1]) or (close[i - 1] > final_upper[i - 1]):
                final_upper[i] = basic_upper[i]
            else:
                final_upper[i] = final_upper[i - 1]

            # Final Lower Band
            if (basic_lower[i] > final_lower[i - 1]) or (close[i - 1] < final_lower[i - 1]):
                final_lower[i] = basic_lower[i]
            else:
                final_lower[i] = final_lower[i - 1]

        # Trend Direction
        if i == period:
            trend[i] = 1  # Default init
        else:
            prev_trend = trend[i - 1]
            if prev_trend == 1:
                if close[i] <= final_lower[i]:
                    trend[i] = -1
                else:
                    trend[i] = 1
            else:  # prev_trend == -1
                if close[i] >= final_upper[i]:
                    trend[i] = 1
                else:
                    trend[i] = -1

        # SuperTrend Value
        if trend[i] == 1:
            supertrend[i] = final_lower[i]
        else:
            supertrend[i] = final_upper[i]

    return supertrend, trend


@jit(nopython=True, cache=True)
def _numba_choppiness_index(high, low, close, period=14):
    """Calcule le Choppiness Index ultra-rapide avec Numba."""
    n = len(high)
    chop = np.full(n, np.nan, dtype=OPTIMAL_FLOAT)

    # ATR components (True Range only)
    tr = np.full(n, np.nan, dtype=OPTIMAL_FLOAT)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = abs(high[i] - close[i - 1])
        tr3 = abs(low[i] - close[i - 1])
        tr[i] = max(tr1, tr2, tr3)

    for i in range(period, n):
        sum_tr = np.sum(tr[i - period + 1 : i + 1])
        max_high = np.max(high[i - period + 1 : i + 1])
        min_low = np.min(low[i - period + 1 : i + 1])

        range_diff = max_high - min_low
        if range_diff != 0 and sum_tr != 0:
            # CHOP = 100 * LOG10( SUM(TR,14) / (MAX(Hi,14) - MIN(Lo,14)) ) / LOG10(14)
            chop[i] = 100.0 * np.log10(sum_tr / range_diff) / np.log10(period)
        else:
            chop[i] = 0.0  # Fallback

    return chop


@jit(nopython=True, cache=True)
def _numba_vortex(high, low, close, period=14):
    """Calcule le Vortex Indicator (VI+ et VI-) ultra-rapide avec Numba."""
    n = len(high)
    vip = np.full(n, np.nan, dtype=OPTIMAL_FLOAT)  # VI+
    vim = np.full(n, np.nan, dtype=OPTIMAL_FLOAT)  # VI-

    # Pre-calculate movements
    vm_plus = np.zeros(n, dtype=OPTIMAL_FLOAT)
    vm_minus = np.zeros(n, dtype=OPTIMAL_FLOAT)
    tr = np.zeros(n, dtype=OPTIMAL_FLOAT)

    for i in range(1, n):
        # VM+ = Abs(Current High - Previous Low)
        vm_plus[i] = abs(high[i] - low[i - 1])
        # VM- = Abs(Current Low - Previous High)
        vm_minus[i] = abs(low[i] - high[i - 1])

        # TR
        tr1 = high[i] - low[i]
        tr2 = abs(high[i] - close[i - 1])
        tr3 = abs(low[i] - close[i - 1])
        tr[i] = max(tr1, tr2, tr3)

    for i in range(period, n):
        sum_vm_plus = np.sum(vm_plus[i - period + 1 : i + 1])
        sum_vm_minus = np.sum(vm_minus[i - period + 1 : i + 1])
        sum_tr = np.sum(tr[i - period + 1 : i + 1])

        if sum_tr != 0:
            vip[i] = sum_vm_plus / sum_tr
            vim[i] = sum_vm_minus / sum_tr

    return vip, vim


# ============================================================================
# CACHE INTELLIGENT
# ============================================================================


class IntelligentCache:
    """Cache intelligent basé sur hash des données."""

    def __init__(self, max_size=1000):
        """__init__.

        Args:
            max_size: TODO document.
        """
        self.cache = {}
        self.max_size = max_size
        self.access_count = {}

    def _get_key(self, data, indicator_name, **params):
        """Génère une clé unique pour le cache."""
        if hasattr(data, "tobytes"):
            data_hash = hashlib.md5(data.tobytes()).hexdigest()[:16]
        else:
            data_hash = hashlib.md5(str(data).encode()).hexdigest()[:16]
        params_str = "_".join([f"{k}_{v}" for k, v in sorted(params.items())])
        return f"{indicator_name}_{data_hash}_{params_str}"

    def get(self, data, indicator_name, **params):
        """Récupère du cache ou None si absent."""
        key = self._get_key(data, indicator_name, **params)
        if key in self.cache:
            self.access_count[key] = self.access_count.get(key, 0) + 1
            return self.cache[key]
        return None

    def set(self, data, indicator_name, result, **params):
        """Stocke dans le cache avec gestion de la taille."""
        if len(self.cache) >= self.max_size:
            # Supprimer l'élément le moins utilisé
            least_used = min(self.access_count.items(), key=lambda x: x[1])[0]
            del self.cache[least_used]
            del self.access_count[least_used]

        key = self._get_key(data, indicator_name, **params)
        self.cache[key] = result
        self.access_count[key] = 1


# ============================================================================
# CONFIGURATION DES MODES D'INDICATEURS
# ============================================================================

INDICATOR_MODES = {
    "fast": [
        # 10 indicateurs essentiels
        "rsi_14",
        "macd_line",
        "ema_20",
        "sma_50",
        "bb_upper",
        "bb_lower",
        "volume_sma_20",
        "atr_14",
        "stoch_k",
        "vwap",
    ],
    "balanced": [
        # 24 indicateurs (fast + 14 supplémentaires)
        "rsi_14",
        "rsi_21",
        "macd_line",
        "macd_signal",
        "macd_histogram",
        "ema_20",
        "ema_50",
        "ema_200",
        "sma_20",
        "sma_50",
        "sma_200",
        "bb_upper",
        "bb_middle",
        "bb_lower",
        "bb_width",
        "bb_percent",
        "volume_sma_20",
        "volume_ratio",
        "atr_14",
        "atr_21",
        "stoch_k",
        "stoch_d",
        "williams_r",
        "cci_20",
        "adx_14",
        "obv",
        "momentum_10",
        "roc_10",
        "vwap",
    ],
    "full": [
        # Tous les 51  indicateurs - LISTE COMPLÈTE CORRIGÉE
        "rsi_14",
        "rsi_21",
        "rsi_30",
        "macd_line",
        "macd_signal",
        "macd_histogram",
        "ema_9",
        "ema_12",
        "ema_21",
        "ema_26",
        "ema_50",
        "ema_100",
        "ema_200",
        "sma_20",
        "sma_50",
        "sma_100",
        "sma_200",
        "bb_upper",
        "bb_middle",
        "bb_lower",
        "bb_width",
        "bb_percent",
        "volume_sma_20",
        "volume_ratio",
        "atr_14",
        "atr_21",
        "stoch_k",
        "stoch_d",
        "williams_r",
        "cci_20",
        "adx_14",
        "di_plus",
        "di_minus",
        "obv",
        "obv_ema",
        "vwap",
        "momentum_10",
        "momentum_20",
        "roc_10",
        "roc_20",
        # 7 INDICATEURS MANQUANTS AJOUTÉS
        "trix_14",
        "ultimate_oscillator",
        "money_flow_index",
        "chaikin_oscillator",
        "aroon_up",
        "aroon_down",
        "parabolic_sar",
        # NOUVEAUX INDICATEURS
        "supertrend",
        "supertrend_signal",
        "choppiness_index",
        "vortex_pos",
        "vortex_neg",
    ],
}
# ============================================================================
# CLASSE PRINCIPALE OPTIMISÉE
# ============================================================================


class TechnicalIndicatorsEnricher:
    """Enrichisseur d'indicateurs techniques optimisé.

    FONCTIONNALITÉS:
    - 3 modes: fast(10), balanced(24), full(47)
    - Chunking intelligent par asset
    - Workers adaptatifs selon n_jobs
    - Cache intelligent
    - Gestion mémoire automatique
    """

    def __init__(
        self,
        mode="full",
        chunk_size=None,
        max_memory_gb=8.0,
        n_jobs=None,
        use_dask=True,
        _is_worker: bool = False,  # <-- MODIFICATION: Ajout du paramètre
    ):
        """Initialisation de l'enrichisseur optimisé.

        Args:
            mode: 'fast'(10), 'balanced'(24), 'full'(47)
            chunk_size: Taille des chunks (auto si None)
            max_memory_gb: Limite mémoire
            n_jobs: Nombre de workers (auto si None)
            use_dask: Utiliser Dask si disponible
            _is_worker: (Interne) Active le mode d'initialisation léger et silencieux pour les workers Dask.
        """
        # Logique de base requise par toutes les instances (principale et workers)
        if mode not in INDICATOR_MODES:
            raise ValueError(
                f"Mode '{mode}' invalide. Modes disponibles: {list(INDICATOR_MODES.keys())}"
            )
        self.mode = mode
        self.indicators_to_compute = INDICATOR_MODES[mode]
        self.base_columns = [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "asset",
            "timeframe",
        ]
        self.final_column_order = self.base_columns + self.indicators_to_compute

        # Si l'instance est un worker, on effectue une initialisation minimale et on s'arrête.
        # Cela évite les configurations lourdes et les messages `print` inutiles.
        if _is_worker:
            self.cache = None  # Pas de cache pour les workers
            self.n_jobs = 1  # Le parallélisme est déjà géré par Dask
            self.stats = {}  # Initialiser pour éviter les erreurs d'attribut
            # On quitte la fonction ici pour ne pas exécuter le code ci-dessous.
            return

        # Le code ci-dessous ne s'exécute QUE pour l'instance principale, pas pour les workers.
        # Configuration performance (optimisée pour gros fichiers)
        self.max_memory_gb = max_memory_gb
        self.n_jobs = min(cpu_count(), 2)  # Réduit de 4 à 2 pour éviter surcharge
        self.chunk_size = chunk_size or self._calculate_optimal_chunk_size()
        self.use_dask = use_dask and DASK_AVAILABLE

        # Cache intelligent
        self.cache = IntelligentCache(max_size=2000)

        # Configuration Dask adaptative
        if self.use_dask:
            self._setup_dask_config()

        # Statistiques
        self.stats = {
            "mode": mode,
            "indicators_count": len(self.indicators_to_compute),
            "chunk_size": self.chunk_size,
            "n_jobs": self.n_jobs,
            "use_dask": self.use_dask,
        }

        print("🚀 TechnicalIndicatorsEnricher initialisé")
        print(f"   Mode: {mode} ({len(self.indicators_to_compute)} indicateurs)")
        print(
            f"   Workers: {self.n_jobs} | Chunk: {self.chunk_size:,} | Dask: {'✅' if self.use_dask else '❌'}"
        )

    def _calculate_optimal_chunk_size(self):
        """Calcule la taille optimale des chunks basée sur la mémoire disponible."""
        if PSUTIL_AVAILABLE and psutil is not None:
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
        else:
            available_memory_gb = 4.0

        # Estimation: 1M lignes ≈ 100MB avec tous les indicateurs
        # Ajuster selon le mode
        mode_factor = {"fast": 0.3, "balanced": 0.6, "full": 1.0}[self.mode]
        memory_per_million = 0.1 * mode_factor  # GB par million de lignes

        # Utiliser 60% de la mémoire disponible, divisé par le nombre de workers
        usable_memory = (available_memory_gb * 0.8) / self.n_jobs
        optimal_chunk = int((usable_memory / memory_per_million) * 1_000_000)

        # Limites raisonnables
        return max(10_000, min(optimal_chunk, 200_000))

    def _setup_dask_config(self):
        """Configure Dask selon le nombre de workers demandés."""
        self.dask_config = {
            "n_workers": self.n_jobs,
            "threads_per_worker": 1,
            "memory_limit": f"{(self.max_memory_gb) / self.n_jobs:.1f}GB",
            **DASK_CONFIG,
        }
        print(
            f"🌊 Configuration Dask: {self.n_jobs} workers, {self.dask_config['memory_limit']} par worker"
        )

    def _create_asset_chunks(self, df: pd.DataFrame) -> list[pd.DataFrame]:
        """Crée des chunks intelligents par asset avec division automatique.

        LOGIQUE:
        1. Grouper par asset
        2. Si asset > CHUNK_SIZE_THRESHOLD (50k) → diviser en chunks
        3. Sinon → garder l'asset entier
        """
        chunks = []

        if "asset" in df.columns:
            asset_groups = df.groupby("asset")
            print(f"📦 Traitement de {len(asset_groups)} assets")

            for asset_name, asset_data in asset_groups:
                asset_size = len(asset_data)

                if asset_size > CHUNK_SIZE_THRESHOLD:
                    # Asset trop gros → diviser en chunks
                    num_chunks = max(2, asset_size // MIN_CHUNK_SIZE)
                    chunk_size = asset_size // num_chunks

                    print(
                        f"   📊 Asset {asset_name}: {asset_size:,} lignes → {num_chunks} chunks de ~{chunk_size:,}"
                    )

                    for i in range(0, asset_size, chunk_size):
                        chunk = asset_data.iloc[i : i + chunk_size].copy()
                        if len(chunk) > 0:
                            chunks.append(chunk)
                else:
                    # Asset de taille raisonnable → garder entier
                    print(f"   ✅ Asset {asset_name}: {asset_size:,} lignes (chunk unique)")
                    chunks.append(asset_data.copy())
        else:
            # Pas d'asset → chunking simple par taille
            print(f"📦 Pas d'asset détecté → chunking par taille ({self.chunk_size:,})")
            for i in range(0, len(df), self.chunk_size):
                chunk = df.iloc[i : i + self.chunk_size].copy()
                if len(chunk) > 0:
                    chunks.append(chunk)

        print(f"📊 Total: {len(chunks)} chunks créés")
        return chunks

    def _compute_indicators_for_chunk(self, chunk_df: pd.DataFrame) -> pd.DataFrame:
        """Calcule les indicateurs pour un chunk selon le mode sélectionné."""
        if chunk_df.empty:
            return chunk_df

        try:
            # Copier le chunk
            result_df = chunk_df.copy()

            # S'assurer que les colonnes de base sont présentes
            for col in self.base_columns:
                if col not in result_df.columns:
                    if col in ["asset", "timeframe"]:
                        result_df[col] = "unknown"
                    else:
                        result_df[col] = np.nan

            # Calculer seulement les indicateurs du mode sélectionné
            result_df = self._compute_selected_indicators(result_df)

            # S'assurer que toutes les colonnes attendues sont présentes
            for col in self.indicators_to_compute:
                if col not in result_df.columns:
                    result_df[col] = np.nan

            # Réorganiser dans l'ordre final
            return result_df[self.final_column_order]

        except Exception as e:
            print(f"⚠️ Erreur chunk (taille: {len(chunk_df)}): {e}")
            import traceback

            print(f"   Détails: {traceback.format_exc()}")

            # Créer un DataFrame de secours avec toutes les colonnes nécessaires
            result_df = chunk_df.copy()

            # S'assurer que les colonnes de base sont présentes
            for col in self.base_columns:
                if col not in result_df.columns:
                    if col in ["asset", "timeframe"]:
                        result_df[col] = "unknown"
                    else:
                        result_df[col] = np.nan

            # Ajouter les colonnes d'indicateurs avec des NaN
            for col in self.indicators_to_compute:
                result_df[col] = np.nan

            # Retourner dans l'ordre final
            try:
                return result_df[self.final_column_order]
            except KeyError:
                # Si même ça échoue, retourner le chunk original
                return chunk_df

    def _compute_selected_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcule seulement les indicateurs sélectionnés selon le mode."""
        # Vérifications de sécurité
        if df.empty:
            return df

        # Vérifier que les colonnes OHLCV sont présentes
        required_cols = ["open", "high", "low", "close", "volume"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"⚠️ Colonnes manquantes: {missing_cols}")
            for col in missing_cols:
                df[col] = np.nan

        # Vérifier que les données ne sont pas toutes NaN
        if df["close"].isna().all():  # pyright: ignore[reportGeneralTypeIssues]
            print("⚠️ Toutes les valeurs 'close' sont NaN")
            for indicator in self.indicators_to_compute:
                df[indicator] = np.nan
            return df

        # Vérifier le cache
        try:
            f"{self.mode}_{len(df)}"
            # CORRECTION: On vérifie que le cache existe avant de l'utiliser
            if self.cache is not None:
                cached_result = self.cache.get(df["close"].values, "indicators", mode=self.mode)
                if cached_result is not None:
                    return cached_result
        except Exception:
            pass  # Ignorer les erreurs de cache

        # Calculer les indicateurs selon le mode
        for indicator in self.indicators_to_compute:
            try:
                if indicator.startswith("rsi_"):
                    period = int(indicator.split("_")[1])
                    df[indicator] = self._calculate_rsi(df["close"], period)

                elif indicator.startswith("macd_"):
                    if "macd_line" not in df.columns:  # Calculer MACD une seule fois
                        macd_data = self._calculate_macd(df["close"])
                        df["macd_line"] = macd_data["macd"]
                        df["macd_signal"] = macd_data["signal"]
                        df["macd_histogram"] = macd_data["histogram"]

                elif indicator.startswith("ema_"):
                    period = int(indicator.split("_")[1])
                    df[indicator] = self._calculate_ema(df["close"], period)

                elif indicator.startswith("sma_"):
                    period = int(indicator.split("_")[1])
                    df[indicator] = self._calculate_sma(df["close"], period)

                elif indicator.startswith("bb_"):
                    if "bb_upper" not in df.columns:  # Calculer BB une seule fois
                        bb_data = self._calculate_bollinger_bands(df["close"])
                        df["bb_upper"] = bb_data["upper"]
                        df["bb_middle"] = bb_data["middle"]
                        df["bb_lower"] = bb_data["lower"]
                        df["bb_width"] = bb_data["width"]
                        df["bb_percent"] = bb_data["percent"]

                elif indicator.startswith("volume_"):
                    if indicator == "volume_sma_20":
                        df[indicator] = self._calculate_sma(df["volume"], 20)
                    elif indicator == "volume_ratio":
                        if "volume_sma_20" not in df.columns:
                            df["volume_sma_20"] = self._calculate_sma(df["volume"], 20)
                        df[indicator] = df["volume"] / df["volume_sma_20"]

                elif indicator.startswith("atr_"):
                    period = int(indicator.split("_")[1])
                    df[indicator] = self._calculate_atr(df, period)

                elif indicator.startswith("stoch_"):
                    if "stoch_k" not in df.columns:  # Calculer Stochastic une seule fois
                        stoch_data = self._calculate_stochastic(df)
                        df["stoch_k"] = stoch_data["k"]
                        df["stoch_d"] = stoch_data["d"]

                elif indicator == "williams_r":
                    df[indicator] = self._calculate_williams_r(df)

                elif indicator == "cci_20":
                    df[indicator] = self._calculate_cci(df)

                elif indicator.startswith("momentum_"):
                    period = int(indicator.split("_")[1])
                    df[indicator] = self._calculate_momentum(df["close"], period)

                elif indicator.startswith("roc_"):
                    period = int(indicator.split("_")[1])
                    df[indicator] = self._calculate_roc(df["close"], period)

                elif indicator == "vwap":
                    df[indicator] = self._calculate_vwap(df["close"], df["volume"])

                elif indicator.startswith("adx_") or indicator in [
                    "di_plus",
                    "di_minus",
                ]:
                    if "adx_14" not in df.columns:  # Calculer ADX une seule fois
                        adx_data = self._calculate_adx_complete(df)
                        df["adx_14"] = adx_data["adx"]
                        df["di_plus"] = adx_data["di_plus"]
                        df["di_minus"] = adx_data["di_minus"]

                elif indicator.startswith("obv"):
                    if indicator == "obv":
                        df[indicator] = self._calculate_obv(df["close"], df["volume"])
                    elif indicator == "obv_ema":
                        if "obv" not in df.columns:
                            df["obv"] = self._calculate_obv(df["close"], df["volume"])
                        df[indicator] = self._calculate_ema(df["obv"], 20)
                elif indicator == "trix_14":
                    df[indicator] = self._calculate_trix(df["close"])

                elif indicator == "ultimate_oscillator":
                    df[indicator] = self._calculate_ultimate_oscillator(df)

                elif indicator == "money_flow_index":
                    df[indicator] = self._calculate_money_flow_index(df)

                elif indicator == "chaikin_oscillator":
                    df[indicator] = self._calculate_chaikin_oscillator(df)

                elif indicator.startswith("aroon_"):
                    if "aroon_up" not in df.columns:  # Calculer Aroon une seule fois
                        aroon_data = self._calculate_aroon(df)
                        df["aroon_up"] = aroon_data["up"]
                        df["aroon_down"] = aroon_data["down"]

                elif indicator == "parabolic_sar":
                    df[indicator] = self._calculate_parabolic_sar(df)

                elif indicator == "supertrend":
                    st_data = self._calculate_supertrend(df)
                    df["supertrend"] = st_data["value"]
                    df["supertrend_signal"] = st_data["signal"]

                elif indicator == "choppiness_index":
                    df[indicator] = self._calculate_choppiness_index(df)

                elif indicator == "vortex_pos":
                    if "vortex_pos" not in df.columns:  # Calculer une seule fois
                        vortex_data = self._calculate_vortex_indicator(df)
                        df["vortex_pos"] = vortex_data["vip"]
                        df["vortex_neg"] = vortex_data["vim"]

            except Exception as e:
                print(f"⚠️ Erreur calcul {indicator}: {e}")
                df[indicator] = np.nan

        # Mettre en cache SEULEMENT si le cache est activé (pas pour les workers)
        if self.cache is not None:
            self.cache.set(df["close"].values, "indicators", df, mode=self.mode)

        return df

    # ========================================================================
    # MÉTHODES DE CALCUL DES INDICATEURS (utilisant les fonctions Numba)
    # ========================================================================

    def _calculate_rsi(self, prices, period):
        """Calcule RSI en utilisant Numba."""
        try:
            if len(prices) < period:
                return pd.Series([np.nan] * len(prices), index=prices.index)

            prices_array = prices.values.astype(OPTIMAL_FLOAT)
            periods_array = np.array([period], dtype=np.int32)
            results = _numba_rsi_vectorized(prices_array, periods_array)

            if len(results) == 0 or len(results[0]) == 0:
                return pd.Series([np.nan] * len(prices), index=prices.index)

            return pd.Series(results[0], index=prices.index)
        except Exception as e:
            print(f"⚠️ Erreur calcul RSI: {e}")
            return pd.Series([np.nan] * len(prices), index=prices.index)

    def _calculate_macd(self, prices):
        """Calcule MACD complet en utilisant Numba."""
        prices_array = prices.values.astype(OPTIMAL_FLOAT)
        macd_line, signal_line, histogram = _numba_macd_complete(prices_array)
        return {
            "macd": pd.Series(macd_line, index=prices.index),
            "signal": pd.Series(signal_line, index=prices.index),
            "histogram": pd.Series(histogram, index=prices.index),
        }

    def _calculate_ema(self, prices, period):
        """Calcule EMA en utilisant Numba."""
        try:
            if len(prices) < period:
                return pd.Series([np.nan] * len(prices), index=prices.index)

            prices_array = prices.values.astype(OPTIMAL_FLOAT)
            periods_array = np.array([period], dtype=np.int32)
            results = _numba_ema_vectorized(prices_array, periods_array)

            if len(results) == 0 or len(results[0]) == 0:
                return pd.Series([np.nan] * len(prices), index=prices.index)

            return pd.Series(results[0], index=prices.index)
        except Exception as e:
            print(f"⚠️ Erreur calcul EMA: {e}")
            return pd.Series([np.nan] * len(prices), index=prices.index)

    def _calculate_sma(self, prices, period):
        """Calcule SMA en utilisant Numba."""
        try:
            if len(prices) < period:
                return pd.Series([np.nan] * len(prices), index=prices.index)

            prices_array = prices.values.astype(OPTIMAL_FLOAT)
            periods_array = np.array([period], dtype=np.int32)
            results = _numba_sma_vectorized(prices_array, periods_array)

            if len(results) == 0 or len(results[0]) == 0:
                return pd.Series([np.nan] * len(prices), index=prices.index)

            return pd.Series(results[0], index=prices.index)
        except Exception as e:
            print(f"⚠️ Erreur calcul SMA: {e}")
            return pd.Series([np.nan] * len(prices), index=prices.index)

    def _calculate_bollinger_bands(self, prices, period=20, std_dev=2.0):
        """Calcule Bollinger Bands en utilisant Numba."""
        prices_array = prices.values.astype(OPTIMAL_FLOAT)
        upper, middle, lower = _numba_bollinger_bands(prices_array, period, std_dev)

        upper_series = pd.Series(upper, index=prices.index)
        middle_series = pd.Series(middle, index=prices.index)
        lower_series = pd.Series(lower, index=prices.index)

        return {
            "upper": upper_series,
            "middle": middle_series,
            "lower": lower_series,
            "width": (upper_series - lower_series) / middle_series,
            "percent": (prices - lower_series) / (upper_series - lower_series),
        }

    def _calculate_atr(self, df, period=14):
        """Calcule ATR en utilisant Numba."""
        high_array = df["high"].values.astype(OPTIMAL_FLOAT)
        low_array = df["low"].values.astype(OPTIMAL_FLOAT)
        close_array = df["close"].values.astype(OPTIMAL_FLOAT)
        atr_values = _numba_atr(high_array, low_array, close_array, period)
        return pd.Series(atr_values, index=df.index)

    def _calculate_stochastic(self, df, k_period=14, d_period=3):
        """Calcule Stochastic en utilisant Numba."""
        high_array = df["high"].values.astype(OPTIMAL_FLOAT)
        low_array = df["low"].values.astype(OPTIMAL_FLOAT)
        close_array = df["close"].values.astype(OPTIMAL_FLOAT)
        k_values, d_values = _numba_stochastic(
            high_array, low_array, close_array, k_period, d_period
        )
        return {
            "k": pd.Series(k_values, index=df.index),
            "d": pd.Series(d_values, index=df.index),
        }

    def _calculate_williams_r(self, df, period=14):
        """Calcule Williams %R en utilisant Numba."""
        high_array = df["high"].values.astype(OPTIMAL_FLOAT)
        low_array = df["low"].values.astype(OPTIMAL_FLOAT)
        close_array = df["close"].values.astype(OPTIMAL_FLOAT)
        williams_values = _numba_williams_r(high_array, low_array, close_array, period)
        return pd.Series(williams_values, index=df.index)

    def _calculate_cci(self, df, period=20):
        """Calcule CCI en utilisant Numba."""
        high_array = df["high"].values.astype(OPTIMAL_FLOAT)
        low_array = df["low"].values.astype(OPTIMAL_FLOAT)
        close_array = df["close"].values.astype(OPTIMAL_FLOAT)
        cci_values = _numba_cci(high_array, low_array, close_array, period)
        return pd.Series(cci_values, index=df.index)

    def _calculate_momentum(self, prices, period=10):
        """Calcule Momentum en utilisant Numba."""
        prices_array = prices.values.astype(OPTIMAL_FLOAT)
        momentum_values = _numba_momentum(prices_array, period)
        return pd.Series(momentum_values, index=prices.index)

    def _calculate_roc(self, prices, period=12):
        """Calcule ROC en utilisant Numba."""
        prices_array = prices.values.astype(OPTIMAL_FLOAT)
        roc_values = _numba_roc(prices_array, period)
        return pd.Series(roc_values, index=prices.index)

    def _calculate_vwap(self, prices, volumes):
        """Calcule VWAP en utilisant Numba."""
        prices_array = prices.values.astype(OPTIMAL_FLOAT)
        volumes_array = volumes.values.astype(OPTIMAL_FLOAT)
        vwap_values = _numba_vwap(prices_array, volumes_array)
        return pd.Series(vwap_values, index=prices.index)

    def _calculate_adx_complete(self, df, period=14):
        """Calcule ADX complet en utilisant Numba."""
        high_array = df["high"].values.astype(OPTIMAL_FLOAT)
        low_array = df["low"].values.astype(OPTIMAL_FLOAT)
        close_array = df["close"].values.astype(OPTIMAL_FLOAT)
        adx_values, di_plus_values, di_minus_values = _numba_adx_complete(
            high_array, low_array, close_array, period
        )
        return {
            "adx": pd.Series(adx_values, index=df.index),
            "di_plus": pd.Series(di_plus_values, index=df.index),
            "di_minus": pd.Series(di_minus_values, index=df.index),
        }

    def _calculate_obv(self, prices, volumes):
        """Calcule OBV en utilisant Numba."""
        prices_array = prices.values.astype(OPTIMAL_FLOAT)
        volumes_array = volumes.values.astype(OPTIMAL_FLOAT)
        obv_values = _numba_obv(prices_array, volumes_array)
        return pd.Series(obv_values, index=prices.index)

    def _calculate_trix(self, prices, period=14):
        """Calcule TRIX en utilisant une fonction Numba dédiée."""
        prices_array = prices.values.astype(OPTIMAL_FLOAT)
        trix_values = _numba_trix(prices_array, period)
        return pd.Series(trix_values, index=prices.index)

    def _calculate_ultimate_oscillator(self, df):
        """Calcule l'Ultimate Oscillator en utilisant une fonction Numba dédiée."""
        high_array = df["high"].values.astype(OPTIMAL_FLOAT)
        low_array = df["low"].values.astype(OPTIMAL_FLOAT)
        close_array = df["close"].values.astype(OPTIMAL_FLOAT)
        uo_values = _numba_ultimate_oscillator(high_array, low_array, close_array)
        return pd.Series(uo_values, index=df.index)

    def _calculate_money_flow_index(self, df, period=14):
        """Calcule le Money Flow Index (MFI) en utilisant une fonction Numba dédiée."""
        high_array = df["high"].values.astype(OPTIMAL_FLOAT)
        low_array = df["low"].values.astype(OPTIMAL_FLOAT)
        close_array = df["close"].values.astype(OPTIMAL_FLOAT)
        volume_array = df["volume"].values.astype(OPTIMAL_FLOAT)
        mfi_values = _numba_mfi(high_array, low_array, close_array, volume_array, period)
        return pd.Series(mfi_values, index=df.index)

    def _calculate_chaikin_oscillator(self, df):
        """Calcule le Chaikin Oscillator en utilisant une fonction Numba dédiée."""
        high_array = df["high"].values.astype(OPTIMAL_FLOAT)
        low_array = df["low"].values.astype(OPTIMAL_FLOAT)
        close_array = df["close"].values.astype(OPTIMAL_FLOAT)
        volume_array = df["volume"].values.astype(OPTIMAL_FLOAT)
        adosc_values = _numba_chaikin_oscillator(high_array, low_array, close_array, volume_array)
        return pd.Series(adosc_values, index=df.index)

    def _calculate_aroon(self, df, period=14):
        """Calcule Aroon Up et Aroon Down en utilisant une fonction Numba dédiée."""
        high_array = df["high"].values.astype(OPTIMAL_FLOAT)
        low_array = df["low"].values.astype(OPTIMAL_FLOAT)
        aroon_up, aroon_down = _numba_aroon(high_array, low_array, period)
        return {
            "up": pd.Series(aroon_up, index=df.index),
            "down": pd.Series(aroon_down, index=df.index),
        }

    def _calculate_parabolic_sar(self, df):
        """Calcule le Parabolic SAR en utilisant la fonction Numba existante."""
        high_array = df["high"].values.astype(OPTIMAL_FLOAT)
        low_array = df["low"].values.astype(OPTIMAL_FLOAT)
        sar_values = _numba_parabolic_sar(high_array, low_array)
        return pd.Series(sar_values, index=df.index)

    # ========================================================================
    # NOUVEAUX INDICATEURS
    # ========================================================================

    def _calculate_supertrend(self, df, period=10, multiplier=3.0):
        """Calcule le SuperTrend."""
        high_array = df["high"].values.astype(OPTIMAL_FLOAT)
        low_array = df["low"].values.astype(OPTIMAL_FLOAT)
        close_array = df["close"].values.astype(OPTIMAL_FLOAT)

        st_val, st_sig = _numba_supertrend(high_array, low_array, close_array, period, multiplier)

        return {
            "value": pd.Series(st_val, index=df.index),
            "signal": pd.Series(st_sig, index=df.index),
        }

    def _calculate_choppiness_index(self, df, period=14):
        """Calcule le Choppiness Index."""
        high_array = df["high"].values.astype(OPTIMAL_FLOAT)
        low_array = df["low"].values.astype(OPTIMAL_FLOAT)
        close_array = df["close"].values.astype(OPTIMAL_FLOAT)

        chop_val = _numba_choppiness_index(high_array, low_array, close_array, period)
        return pd.Series(chop_val, index=df.index)

    def _calculate_vortex_indicator(self, df, period=14):
        """Calcule le Vortex Indicator."""
        high_array = df["high"].values.astype(OPTIMAL_FLOAT)
        low_array = df["low"].values.astype(OPTIMAL_FLOAT)
        close_array = df["close"].values.astype(OPTIMAL_FLOAT)

        vip, vim = _numba_vortex(high_array, low_array, close_array, period)
        return {"vip": pd.Series(vip, index=df.index), "vim": pd.Series(vim, index=df.index)}

    # ========================================================================
    # MÉTHODES PRINCIPALES D'ENRICHISSEMENT
    # ========================================================================

    def enrich_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enrichit un DataFrame avec les indicateurs techniques.

        PROCESSUS:
        1. Créer des chunks intelligents par asset
        2. Traiter en parallèle avec ThreadPoolExecutor
        3. Concaténer les résultats
        """
        start_time = time.time()
        print(f"🚀 Enrichissement DataFrame: {len(df):,} lignes, mode '{self.mode}'")

        # Créer des chunks intelligents
        chunks = self._create_asset_chunks(df)

        # Traitement parallèle
        if len(chunks) == 1:
            # Un seul chunk → traitement direct
            print("📊 Traitement direct (chunk unique)")
            result_df = self._compute_indicators_for_chunk(chunks[0])
        else:
            # Plusieurs chunks → traitement parallèle avec suivi de progression
            print(f"⚡ Traitement parallèle avec {self.n_jobs} workers")
            print(f"📦 {len(chunks)} chunks à traiter")

            processed_chunks = []

            with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                # Soumettre tous les chunks
                futures = [
                    executor.submit(self._compute_indicators_for_chunk, chunk) for chunk in chunks
                ]

                # Récupérer les résultats avec suivi de progression
                for i, future in enumerate(futures):
                    try:
                        result_chunk = future.result(timeout=300)  # 5 min timeout par chunk
                        processed_chunks.append(result_chunk)

                        # Afficher progression tous les 10 chunks
                        if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
                            progress_pct = ((i + 1) / len(chunks)) * 100
                            print(
                                f"   📊 Progression: {i + 1}/{len(chunks)} chunks traités ({progress_pct:.1f}%)"
                            )

                    except Exception as e:
                        print(f"   ⚠️ Erreur chunk {i + 1}: {e}")
                        # Ajouter un chunk vide en cas d'erreur pour maintenir l'ordre
                        if i < len(chunks):
                            processed_chunks.append(chunks[i])

            # Concaténer les résultats
            result_df = pd.concat(processed_chunks, ignore_index=True)

        # Tri final par timestamp si disponible
        if "timestamp" in result_df.columns:
            result_df = result_df.sort_values("timestamp").reset_index(drop=True)

        elapsed = time.time() - start_time
        print(f"✅ Enrichissement terminé: {len(result_df):,} lignes en {elapsed:.1f}s")
        print(f"📊 Indicateurs calculés: {len(self.indicators_to_compute)}")

        return result_df

    def enrich_dataset(self, input_path: str, output_path: str) -> bool:
        """Enrichit un dataset depuis un fichier CSV.

        PROCESSUS:
        1. Charger le dataset par chunks si nécessaire
        2. Enrichir avec la méthode appropriée
        3. Sauvegarder le résultat
        """
        try:
            print(f"📂 Chargement dataset: {input_path}")

            # Vérifier la taille du fichier
            file_size_mb = os.path.getsize(input_path) / (1024 * 1024)
            print(f"📊 Taille fichier: {file_size_mb:.1f} MB")

            if file_size_mb > 500:  # > 500MB → utiliser Dask si disponible
                return self._enrich_large_dataset_with_dask(input_path, output_path)
            else:
                # Fichier de taille raisonnable → charger en mémoire
                df = pd.read_csv(input_path)
                enriched_df = self.enrich_dataframe(df)
                enriched_df.to_csv(output_path, index=False)
                print(f"✅ Dataset enrichi sauvegardé: {output_path}")
                return True

        except Exception as e:
            print(f"❌ Erreur enrichissement dataset: {e}")
            return False

    # Dans technical_indicators.py

    def _enrich_large_dataset_with_dask(self, input_path: str, output_path: str) -> bool:
        """VERSION CORRIGÉE - Orchestre Dask manuellement pour un contrôle total.

        sur le chunking, le logging et surtout une fermeture propre du cluster.
        """
        if not self.use_dask:
            return self._enrich_large_dataset_chunked(input_path, output_path)

        start_time = time.time()
        print("🌊 Enrichissement Dask (mode Orchestration Manuelle)")

        # --- NOUVELLE LOGIQUE DE GESTION EXPLICITE DU CLUSTER ---
        cluster = None
        client = None
        try:
            # 1. Créer le cluster et le client séparément
            # Cela nous donne un contrôle total pour les fermer proprement plus tard.
            cluster = LocalCluster(
                processes=True,
                n_workers=self.n_jobs,
                threads_per_worker=1,
                memory_limit=f"{(self.max_memory_gb * 0.9) / self.n_jobs:.1f}GB",  # 90% pour marge sécurité
            )
            client = Client(cluster)

            print(f"🚀 Cluster Dask démarré: {len(client.scheduler_info()['workers'])} workers")

            # --- ÉTAPE 1: Pré-calcul des chunks (logique identique) ---
            print("📊 Pré-calcul des chunks par asset...")
            df_full = pd.read_csv(input_path)
            chunks_to_process = self._create_asset_chunks(df_full)
            del df_full  # Libérer la mémoire du DF complet
            gc.collect()

            total_chunks = len(chunks_to_process)
            if total_chunks == 0:
                print("⚠️ Aucun chunk à traiter.")
                return False

            # --- ÉTAPE 2: Soumettre les chunks comme tâches indépendantes (logique identique) ---
            print(f"⚡ Soumission de {total_chunks} chunks au cluster Dask...")
            futures = client.map(
                _compute_indicators_for_dask_worker, chunks_to_process, mode=self.mode
            )

            # --- ÉTAPE 3: Suivre la progression et récupérer les résultats (logique identique) ---
            from dask.distributed import as_completed

            future_map = {future: i for i, future in enumerate(futures)}
            results_in_order = [None] * total_chunks
            completed_count = 0

            for future in as_completed(iter(future_map)):
                try:
                    result_chunk = future.result()
                    original_index = future_map[future]
                    results_in_order[original_index] = result_chunk

                    completed_count += 1

                    if completed_count % 10 == 0 or completed_count == total_chunks:
                        progress_pct = (completed_count / total_chunks) * 100
                        print(
                            f"   📊 Progression: {completed_count}/{total_chunks} chunks traités ({progress_pct:.1f}%)"
                        )
                        gc.collect()
                        client.run(gc.collect)
                except Exception as e:
                    print(f"   ❌ Erreur lors de la récupération d'un résultat: {e}")
                gc.collect()
            final_chunks = [
                chunk for chunk in results_in_order if chunk is not None and not chunk.empty
            ]

            if not final_chunks:
                raise RuntimeError("Aucun chunk n'a pu être traité avec succès.")

            # --- ÉTAPE 4: Combiner et Sauvegarder (logique identique) ---
            print("🔗 Combinaison des résultats dans l'ordre...")
            final_df = pd.concat(final_chunks, ignore_index=True)

            print(f"💾 Sauvegarde en format CSV vers: {output_path}")
            write_csv_polars_optimized(final_df, output_path)

            elapsed = time.time() - start_time
            print(f"✅ Dataset Dask enrichi et sauvegardé en {elapsed:.1f}s: {output_path}")
            return True

        except Exception as e:
            import traceback

            print(f"❌ Erreur Dask: {e}")
            print(traceback.format_exc())
            return False
        finally:
            # --- ÉTAPE 5 (CRUCIALE): Assurer la fermeture propre dans tous les cas ---
            print("\n🧹 Nettoyage et fermeture du cluster Dask...")
            if client:
                client.close()
            if cluster:
                cluster.close()
            print("✅ Cluster Dask arrêté proprement.")

    def _enrich_large_dataset_chunked(self, input_path: str, output_path: str) -> bool:
        """Enrichit un gros dataset par chunks manuels."""
        try:
            print("📦 Enrichissement par chunks manuels")

            chunk_size = 100000  # 100k lignes par chunk
            processed_chunks = []

            # Compter d'abord le nombre total de chunks
            total_chunks = 0
            for _ in pd.read_csv(input_path, chunksize=chunk_size):
                total_chunks += 1

            print(f"📊 {total_chunks} chunks à traiter")

            # Traiter les chunks avec suivi de progression
            for i, chunk in enumerate(pd.read_csv(input_path, chunksize=chunk_size)):
                enriched_chunk = self.enrich_dataframe(chunk)
                processed_chunks.append(enriched_chunk)

                # Afficher progression tous les 10 chunks
                if (i + 1) % 10 == 0 or (i + 1) == total_chunks:
                    progress_pct = ((i + 1) / total_chunks) * 100
                    print(
                        f"   📊 Progression: {i + 1}/{total_chunks} chunks traités ({progress_pct:.1f}%) - {len(chunk):,} lignes"
                    )

            # Concaténer et sauvegarder
            final_df = pd.concat(processed_chunks, ignore_index=True)
            final_df.to_csv(output_path, index=False)

            print(f"✅ Dataset chunked enrichi: {output_path}")
            return True

        except Exception as e:
            print(f"❌ Erreur chunking manuel: {e}")
            return False

    def _create_dask_meta(self) -> pd.DataFrame:
        """Crée les métadonnées pour Dask avec les colonnes du mode sélectionné."""
        meta_dict = {}

        # Colonnes de base
        for col in self.base_columns:
            if col == "timestamp":
                meta_dict[col] = pd.Series([], dtype="datetime64[ns]")
            elif col in ["asset", "timeframe"]:
                meta_dict[col] = pd.Series([], dtype="object")
            else:
                meta_dict[col] = pd.Series([], dtype="float32")

        # Indicateurs du mode sélectionné
        for col in self.indicators_to_compute:
            meta_dict[col] = pd.Series([], dtype="float32")

        meta_df = pd.DataFrame(meta_dict)
        meta_df = meta_df[self.final_column_order]

        print(f"📊 Métadonnées Dask: {len(meta_df.columns)} colonnes (mode {self.mode})")
        return meta_df

    def get_stats(self) -> dict:
        """Retourne les statistiques de l'enrichisseur."""
        # Garde pour workers (self.cache = None)
        cache_size = len(self.cache.cache) if self.cache is not None else 0
        return {
            **self.stats,
            "cache_size": cache_size,
            "memory_usage_gb": psutil.Process().memory_info().rss / (1024**3) if PSUTIL_AVAILABLE and psutil is not None else 0.0,
        }

    def clear_cache(self):
        """Vide le cache."""
        self.cache.cache.clear()
        self.cache.access_count.clear()
        print("🧹 Cache vidé")


# ============================================================================
# FONCTION UTILITAIRE POUR DASK
# ============================================================================


def _compute_indicators_for_partition_optimized(
    partition_df: pd.DataFrame, mode="full"
) -> pd.DataFrame:
    """Fonction statique optimisée pour les partitions Dask."""
    if partition_df.empty:
        return partition_df

    try:
        # Créer une instance temporaire légère
        temp_enricher = TechnicalIndicatorsEnricher(
            mode=mode,
            n_jobs=1,  # Pas de parallélisme dans les workers Dask
            use_dask=False,
            _is_worker=True,  # Initialisation légère, silencieuse, sans cache
        )

        # Enrichir la partition
        return temp_enricher.enrich_dataframe(partition_df)

    except Exception as e:
        # Imprimer l'erreur du worker pour le débogage
        print(f"⚠️ Erreur critique dans un worker Dask: {e}")
        import traceback

        traceback.print_exc()
        # Retourner un DataFrame vide en cas d'échec pour ne pas faire planter le tout.
        return pd.DataFrame()


def _compute_indicators_for_dask_worker(chunk_df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Fonction worker SILENCIEUSE et autonome pour Dask.

    Reçoit un chunk de données (déjà groupé par asset) et le traite.
    """
    if chunk_df.empty:
        return pd.DataFrame()
    try:
        # On passe `_is_worker=True` pour activer l'initialisation légère et silencieuse.
        # Cela empêche les messages `print` d'inonder la console.
        temp_enricher = TechnicalIndicatorsEnricher(
            mode=mode, n_jobs=1, use_dask=False, _is_worker=True
        )

        # Utiliser directement la méthode de calcul sur le chunk.
        result_df = temp_enricher._compute_indicators_for_chunk(chunk_df)
        return result_df
    except Exception as e:
        asset_name = (
            chunk_df["asset"].iloc[0]
            if "asset" in chunk_df.columns and not chunk_df.empty
            else "Inconnu"
        )
        print(f"⚠️ Erreur worker Dask sur asset {asset_name} (chunk de taille {len(chunk_df)}): {e}")
        # Retourner un DF vide en cas d'erreur pour ne pas planter la tâche principale.
        return pd.DataFrame()
    finally:
        # Cette section s'exécutera toujours, même après le 'return' ou une erreur.
        # On supprime les références aux objets lourds.
        del chunk_df
        try:
            del result_df  # result_df peut ne pas exister si exception avant son affectation
        except NameError:
            pass
        # On force le ramasse-miettes à faire un nettoyage complet.
        gc.collect()


def write_csv_polars_optimized(df: pd.DataFrame, file_path: str) -> bool:
    """Écriture optimisée avec Polars puis conversion depuis Pandas.

    3-5x plus rapide que df.to_csv() direct.
    """
    try:
        # Créer le dossier de destination si nécessaire
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Conversion vers Polars
        df_polars = pl.from_pandas(df)

        # Écriture avec Polars (ultra-rapide)
        df_polars.write_csv(
            file_path,
            separator=",",
            line_terminator="\n",
        )

        return True

    except Exception as e:
        print(f"⚠️ Erreur écriture Polars, fallback vers Pandas: {e}")
        # Fallback vers Pandas si Polars échoue
        df.to_csv(file_path, index=False)
        return True


if __name__ == "__main__":
    # Exemple d'utilisation
    enricher = TechnicalIndicatorsEnricher(mode="full")
