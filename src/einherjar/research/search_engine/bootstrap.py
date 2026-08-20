"""bootstrap.py — Intervalle de confiance bootstrap par blocs (validation C6).

Les rendements de trades ne sont PAS indépendants (positions successives
corrélées au marché) → CI naïf par rééchantillonnage iid sous-estime la
variance. Le block bootstrap (blocs circulaires) préserve la dépendance locale.

Critère d'admission (plan C6) : borne basse du CI à 95 % > 0.
"""
from __future__ import annotations

import numpy as np


def block_bootstrap_ci(
    trade_returns,
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
    block_size: int | None = None,
) -> tuple[float, float, float]:
    """CI bootstrap par blocs sur la moyenne des retours par trade.

    Returns:
        (lo, hi, mean) — percentiles (alpha/2, 1-alpha/2) de la distribution
        bootstrap, et moyenne observée. NaN si pas assez de trades.
    """
    rets = np.asarray(trade_returns, dtype=np.float64)
    n = len(rets)
    if n < 30:
        return (float("nan"), float("nan"), float("nan"))
    bs = block_size or max(5, int(np.sqrt(n)))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / bs))
    means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        blocks = [rets[s : s + bs] for s in starts]
        sample = np.concatenate(blocks)[:n]
        means[i] = float(np.mean(sample))
    lo, hi = np.percentile(means, [100.0 * alpha / 2, 100.0 * (1.0 - alpha / 2)])
    return (float(lo), float(hi), float(np.mean(rets)))