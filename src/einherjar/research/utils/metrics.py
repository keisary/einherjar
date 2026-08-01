"""
utils/metrics.py — Métriques standardisées (Sharpe, Sortino, MAR, expectancy, etc.).

Implémente les briques de calcul pour S-3.5 (MétriquesPortefeuille).
Ne fait PAS de bootstrap ici (voir utils/stats.py + engine/bootstrap.py).
"""

from __future__ import annotations

import math
from typing import Sequence

from einherjar.research.utils.stats import mean, std


# --------------------------------------------------------------------------- #
# Sharpe / Sortino
# --------------------------------------------------------------------------- #


def sharpe(returns: Sequence[float], periods_per_year: float = 365.0, rf: float = 0.0) -> float:
    """Sharpe annualisé. `rf` = risk-free rate annualisé (défaut 0)."""
    if len(returns) < 2:
        return float("nan")
    m = mean(returns) - rf / periods_per_year
    s = std(returns)
    if s == 0 or math.isnan(s):
        return float("nan")
    return (m / s) * math.sqrt(periods_per_year)


def sortino(returns: Sequence[float], periods_per_year: float = 365.0, rf: float = 0.0) -> float:
    """Sortino annualisé (downside deviation seulement)."""
    if len(returns) < 2:
        return float("nan")
    target = rf / periods_per_year
    downside = [min(0.0, r - target) for r in returns]
    dd = math.sqrt(sum(d * d for d in downside) / len(downside))
    if dd == 0 or math.isnan(dd):
        return float("nan")
    m = mean(returns) - target
    return (m / dd) * math.sqrt(periods_per_year)


# --------------------------------------------------------------------------- #
# Expectancy / Profit Factor
# --------------------------------------------------------------------------- #


def expectancy(returns: Sequence[float]) -> float:
    """Espérance de gain par trade. expectancy = mean(returns) = (winrate × gain_moyen) + ((1-winrate) × perte_moyenne)."""
    if not returns:
        return float("nan")
    return mean(returns)


def profit_factor(returns: Sequence[float]) -> float:
    """Profit factor = sum(gains) / |sum(pertes)|. NaN si aucune perte."""
    gains = sum(r for r in returns if r > 0)
    losses = sum(r for r in returns if r < 0)
    if losses == 0:
        return float("inf") if gains > 0 else float("nan")
    return gains / abs(losses)


# --------------------------------------------------------------------------- #
# Drawdown
# --------------------------------------------------------------------------- #


def max_drawdown(equity_curve: Sequence[float]) -> float:
    """Max drawdown en valeur absolue (0..1), à partir d'une equity curve.

    Convention : equity_curve[0] = 1.0 (capital initial normalisé).
    Retourne une valeur positive (0.25 = -25%).
    """
    if not equity_curve:
        return float("nan")
    peak = equity_curve[0]
    max_dd = 0.0
    for v in equity_curve:
        if v > peak:
            peak = v
        dd = (peak - v) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    return max_dd


def ulcer_index(equity_curve: Sequence[float]) -> float:
    """Ulcer Index : sqrt(mean(drawdown²))."""
    if not equity_curve:
        return float("nan")
    peak = equity_curve[0]
    dds: list[float] = []
    for v in equity_curve:
        if v > peak:
            peak = v
        if peak > 0:
            dds.append(((peak - v) / peak) ** 2)
    if not dds:
        return 0.0
    return math.sqrt(sum(dds) / len(dds))


# --------------------------------------------------------------------------- #
# CAGR / MAR
# --------------------------------------------------------------------------- #


def cagr(total_return: float, years: float) -> float:
    """CAGR = (1 + total_return)^(1/years) - 1. `total_return` est décimal (0.5 = +50%)."""
    if years <= 0:
        return float("nan")
    if total_return <= -1.0:
        return -1.0  # capital perdu
    return (1.0 + total_return) ** (1.0 / years) - 1.0


def mar_ratio(cagr_value: float, max_dd: float) -> float:
    """MAR = CAGR / |max_drawdown|. NaN si drawdown nul."""
    if max_dd is None or max_dd == 0 or math.isnan(max_dd):
        return float("nan")
    return cagr_value / max_dd


# --------------------------------------------------------------------------- #
# DSR / PBO (placeholders, implémentation complète dans admission/)
# --------------------------------------------------------------------------- #


def dsr(
    sharpe_observed: float,
    n_trials: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Deflated Sharpe Ratio (Bailey & López de Prado).

    Réduit le Sharpe en fonction du nombre d'essais et de la non-normalité.
    Retourne une probabilité que le Sharpe observé soit > 0 sous H0.

    Note : implémentation simplifiée. La version complète (avec backtest
    overfitting) sera dans `admission/criteria.py` (Step 5).
    """
    if n_trials < 1:
        return float("nan")
    if sharpe_observed is None or math.isnan(sharpe_observed):
        return float("nan")
    # Approximation grossière : on retourne le p-value d'un test z
    # ajusté par sqrt(n_trials) (correction de Bonferroni simplifiée).
    e_max_sharpe = math.sqrt(2.0 * math.log(max(n_trials, 2)))
    se = math.sqrt(1.0 + skewness * sharpe_observed + (kurtosis - 1.0) / 4.0 * sharpe_observed ** 2)
    if se == 0:
        return float("nan")
    z = (sharpe_observed - e_max_sharpe) / se
    # p-value bilatéral : on veut P(Sharpe > 0) — conservateur, on retranche
    from math import erf, sqrt
    p = 0.5 * (1.0 + erf(z / sqrt(2.0)))
    return p
