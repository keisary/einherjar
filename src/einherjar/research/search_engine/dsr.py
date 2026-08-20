"""dsr.py — Deflated Sharpe Ratio (Bailey & López de Prado 2014).

Le SR observé est gonflé par la sélection : parmi N stratégies testées, la
meilleure a un SR attendu > 0 même sans edge réel. Le DSR mesure
Prob(SR_vrai > SR_benchmark) en corrigeant la non-normalité (skew/kurtosis)
des retours par trade — plan ligne 594, critère C2 (seuil 0.95 par défaut).

Formule (per-trade, non annualisé) :
    DSR = Phi( (SR_hat - SR_0) * sqrt(n-1) / sqrt(1 - g3*SR_hat + (g4-1)/4 * SR_hat^2) )
où g3 = skewness, g4 = kurtosis (excès = moment4/sd^4).
"""
from __future__ import annotations

import math

import numpy as np


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def dsr_probability(
    trade_returns,
    *,
    sr_benchmark: float = 0.0,
    min_trades: int = 30,
) -> float:
    """Probabilité que le vrai Sharpe (per-trade) dépasse sr_benchmark.

    Retourne 0.0 si pas assez de trades ou variance dégénérée (formule non
    définie → le candidat n'est pas significatif, refus conservateur).
    """
    rets = np.asarray(trade_returns, dtype=np.float64)
    n = len(rets)
    if n < min_trades:
        return 0.0
    sd = float(np.std(rets, ddof=1))
    if sd <= 0.0 or not np.isfinite(sd):
        return 0.0
    mu = float(np.mean(rets))
    sr_hat = mu / sd
    # Moments centrés (estimateurs biaisés acceptés : mêmes conventions que
    # Bailey & López de Prado sur les échantillons de trades)
    m2 = float(np.mean((rets - mu) ** 2))
    m3 = float(np.mean((rets - mu) ** 3))
    m4 = float(np.mean((rets - mu) ** 4))
    g3 = m3 / (m2 ** 1.5) if m2 > 0 else 0.0
    g4 = m4 / (m2 ** 2) if m2 > 0 else 0.0
    denom = 1.0 - g3 * sr_hat + (g4 - 1.0) / 4.0 * sr_hat**2
    if denom <= 0.0:
        return 0.0
    z = (sr_hat - sr_benchmark) * math.sqrt(n - 1) / math.sqrt(denom)
    return float(_normal_cdf(z))