"""
utils/stats.py — Statistiques : percentiles, block bootstrap, ATR(14).

Implémente les briques de base utilisées par `engine/bootstrap.py` et
`engine/evaluator.py`.
"""

from __future__ import annotations

import math
import statistics
from typing import Sequence


# --------------------------------------------------------------------------- #
# Percentiles (sans dépendance externe, robuste aux petits N)
# --------------------------------------------------------------------------- #


def percentile(values: Sequence[float], p: float) -> float:
    """Percentile p (0..100) par interpolation linéaire (méthode NumPy par défaut)."""
    if not values:
        return float("nan")
    if not (0.0 <= p <= 100.0):
        raise ValueError(f"p doit être dans [0, 100], got {p}")
    s = sorted(values)
    n = len(s)
    if n == 1:
        return float(s[0])
    rank = (p / 100.0) * (n - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return float(s[lo])
    frac = rank - lo
    return float(s[lo] * (1.0 - frac) + s[hi] * frac)


# --------------------------------------------------------------------------- #
# Block bootstrap (pour IC sur Sharpe/ret en présence d'autocorrélation)
# --------------------------------------------------------------------------- #


def block_bootstrap_ci(
    values: Sequence[float],
    statistic,
    n_resamples: int = 2000,
    block_length: int = 1,
    ci_level: float = 0.95,
    rng_seed: int = 42,
) -> tuple[float, float, float]:
    """Block bootstrap : IC et valeur observée d'une statistic.

    Args:
        values: série temporelle (ex: rendements nets sur val)
        statistic: fonction (Sequence[float]) -> float (ex: sharpe_ratio)
        n_resamples: nombre de tirages bootstrap
        block_length: longueur des blocs (>=1). Pour autocorrélation, >1.
        ci_level: niveau de confiance (0..1)
        rng_seed: graine pour reproductibilité

    Returns:
        (ci_low, ci_high, observed)
    """
    if not values:
        return float("nan"), float("nan"), float("nan")
    if block_length < 1:
        raise ValueError(f"block_length doit être >= 1, got {block_length}")
    if n_resamples < 1:
        raise ValueError(f"n_resamples doit être >= 1, got {n_resamples}")
    if not (0.0 < ci_level < 1.0):
        raise ValueError(f"ci_level doit être dans ]0,1[, got {ci_level}")

    import random
    rng = random.Random(rng_seed)
    n = len(values)
    observed = float(statistic(values))

    # Construction des blocs
    if block_length >= n:
        # Si la série est plus courte que le bloc, on retombe sur i.i.d.
        block_length = n

    n_blocks = math.ceil(n / block_length)
    blocks = [tuple(values[i * block_length : (i + 1) * block_length]) for i in range(n_blocks)]

    boot_stats: list[float] = []
    for _ in range(n_resamples):
        sample: list[float] = []
        while len(sample) < n:
            b = rng.choice(blocks)
            sample.extend(b)
        sample = sample[:n]  # tronque à exactement n
        boot_stats.append(float(statistic(sample)))

    boot_stats.sort()
    alpha = 1.0 - ci_level
    lo_idx = max(0, int(math.floor(alpha / 2.0 * n_resamples)))
    hi_idx = min(n_resamples - 1, int(math.ceil((1.0 - alpha / 2.0) * n_resamples)) - 1)
    # correction : on prend l'index quantile
    lo_idx = int(math.floor((alpha / 2.0) * n_resamples))
    hi_idx = int(math.ceil((1.0 - alpha / 2.0) * n_resamples)) - 1
    lo_idx = max(0, min(lo_idx, n_resamples - 1))
    hi_idx = max(0, min(hi_idx, n_resamples - 1))
    return boot_stats[lo_idx], boot_stats[hi_idx], observed


# --------------------------------------------------------------------------- #
# ATR(14) — Average True Range
# --------------------------------------------------------------------------- #


def true_range(high: float, low: float, prev_close: float) -> float:
    """True Range d'une bougie : max(high-low, |high-prev_close|, |low-prev_close|)."""
    return max(high - low, abs(high - prev_close), abs(low - prev_close))


def atr_wilder(highs: Sequence[float], lows: Sequence[float], closes: Sequence[float], period: int = 14) -> list[float]:
    """ATR(period) par méthode de Wilder (lissage exponentiel).

    Retourne une liste de longueur len(highs), avec NaN pour les indices
    où l'historique est insuffisant (< period).
    """
    n = len(highs)
    if not (len(lows) == n == len(closes)):
        raise ValueError("highs, lows, closes doivent avoir la même longueur")
    if period < 1:
        raise ValueError(f"period doit être >= 1, got {period}")
    if n < period + 1:
        # Pas assez de données
        return [float("nan")] * n

    trs = [float("nan")] * n
    for i in range(1, n):
        trs[i] = true_range(highs[i], lows[i], closes[i - 1])

    # Premier ATR = moyenne simple des `period` premiers TR
    atr: list[float] = [float("nan")] * n
    first_atr = sum(trs[1 : period + 1]) / period
    atr[period] = first_atr
    # Lissage de Wilder pour les suivants
    for i in range(period + 1, n):
        atr[i] = (atr[i - 1] * (period - 1) + trs[i]) / period

    return atr


# --------------------------------------------------------------------------- #
# Statistiques simples
# --------------------------------------------------------------------------- #


def mean(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    return statistics.fmean(values)


def std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return float("nan")
    return statistics.stdev(values)


def sharpe_ratio(returns: Sequence[float], periods_per_year: float = 365.0) -> float:
    """Sharpe brut annualisé, sans risk-free rate (à 0)."""
    if len(returns) < 2:
        return float("nan")
    m = mean(returns)
    s = std(returns)
    if s == 0 or math.isnan(s):
        return float("nan")
    return (m / s) * math.sqrt(periods_per_year)


# --------------------------------------------------------------------------- #
# Annualisation cohérente avec le timeframe
# --------------------------------------------------------------------------- #


# Mapping timeframe -> bougies par an (1 bougie = 1 trade dans notre cas).
# Convention : 24/7 pour crypto (8760 h/an), jours ouvrés ~252 pour actions.
# Pour notre moteur (qui trade 24/7 sur crypto), on prend crypto = 24/7.
_PERIODS_PER_YEAR: dict[str, float] = {
    "1m": 365.0 * 24 * 60,         # 525 600
    "5m": 365.0 * 24 * 12,         # 105 120
    "15m": 365.0 * 24 * 4,         # 35 040
    "30m": 365.0 * 24 * 2,         # 17 520
    "1h": 365.0 * 24,              # 8 760
    "2h": 365.0 * 12,              # 4 380
    "4h": 365.0 * 6,               # 2 190
    "6h": 365.0 * 4,               # 1 460
    "8h": 365.0 * 3,               # 1 095
    "12h": 365.0 * 2,              # 730
    "1d": 365.0,                    # 365
    "3d": 365.0 / 3,                # ~122
    "1w": 52.0,                     # 52
}
_DEFAULT_PERIODS_PER_YEAR: float = 365.0


def periods_per_year_for_timeframe(timeframe: str) -> float:
    """Retourne le nb de périodes par an pour un timeframe donné.

    Utilisé par le moteur d'évaluation pour annualiser le Sharpe de manière
    cohérente avec la fréquence des trades (plus de sqrt(365) codé en dur).

    Args:
        timeframe: Code timeframe ('1m', '5m', '15m', '30m', '1h', '2h', '4h',
                   '6h', '8h', '12h', '1d', '3d', '1w').

    Returns:
        Nombre de périodes par an (float). Défaut : 365 si timeframe inconnu.
    """
    return _PERIODS_PER_YEAR.get(timeframe, _DEFAULT_PERIODS_PER_YEAR)


# --------------------------------------------------------------------------- #
# Max drawdown (sur courbe d'equity cumulée)
# --------------------------------------------------------------------------- #


def max_drawdown_from_returns(returns: Sequence[float]) -> float:
    """Max drawdown (en fraction positive, ex: 0.25 = -25%) sur courbe d'equity.

    Reconstruit une equity_curve à partir des rendements périodiques (1 trade = 1 période) :
      equity[0] = 1.0
      equity[t+1] = equity[t] * (1 + returns[t])
    Puis calcule la chute max depuis un pic :
      dd[t] = (peak[t] - equity[t]) / peak[t]
      max_dd = max(dd)

    Convention : on renvoie une valeur POSITIVE (0.25 signifie -25%).
    Pas de trade → 0.0 (pas de drawdown).
    Rendements tous nuls → 0.0.

    Args:
        returns: Rendements nets par trade (ex: MesuresBrutes.trades.ret_pct_net).

    Returns:
        Max drawdown en fraction positive (0.0 = pas de perte, 1.0 = ruine totale).
    """
    if not returns:
        return 0.0
    equity = 1.0
    peak = 1.0
    worst = 0.0
    for r in returns:
        # Rendement NaN → on l'ignore (ne dégrade pas l'equity ni le peak).
        if math.isnan(r) or math.isinf(r):
            continue
        equity *= (1.0 + r)
        if equity > peak:
            peak = equity
        if peak > 0:
            dd = (peak - equity) / peak
            if dd > worst:
                worst = dd
    return float(worst)
