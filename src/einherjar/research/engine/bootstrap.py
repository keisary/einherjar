"""engine/bootstrap.py — Block bootstrap CI pour Sharpe et ret total (S-2 / S-3.4).

Wrap de `utils/stats.py::block_bootstrap_ci` avec les défauts du projet
(config/evaluation.yaml) et la sérialisation des résultats.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from einherjar.research.config.loader import EinherjarConfig
from einherjar.research.utils.stats import block_bootstrap_ci, sharpe_ratio

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BootstrapResult:
    ci_low: float
    ci_high: float
    observed: float
    n_resamples: int
    block_length: int
    ci_level: float
    statistic_name: str

    def to_dict(self) -> dict:
        return {
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "observed": self.observed,
            "n_resamples": self.n_resamples,
            "block_length": self.block_length,
            "ci_level": self.ci_level,
            "statistic_name": self.statistic_name,
        }


def _auto_block_length(n: int, factor: float) -> int:
    """Calcule une longueur de bloc par défaut, clampée à la taille de la série."""
    if n < 2:
        return 1
    bl = int(factor * max(1, n // 20))
    return max(1, min(bl, n))


def bootstrap_sharpe(
    returns: Sequence[float],
    config: EinherjarConfig,
    *,
    periods_per_year: float = 365.0,
) -> BootstrapResult:
    """Block bootstrap CI sur le Sharpe annualisé."""
    bs = config.evaluation["bootstrap"]
    block_length = _auto_block_length(len(returns), bs.get("block_length_factor", 1.5))
    ci_low, ci_high, observed = block_bootstrap_ci(
        values=returns,
        statistic=lambda v: sharpe_ratio(v, periods_per_year=periods_per_year),
        n_resamples=int(bs.get("n_resamples", 2000)),
        block_length=block_length,
        ci_level=float(bs.get("ci_level", 0.95)),
        rng_seed=int(bs.get("seed", 42)),
    )
    return BootstrapResult(
        ci_low=ci_low, ci_high=ci_high, observed=observed,
        n_resamples=int(bs["n_resamples"]),
        block_length=block_length, ci_level=float(bs["ci_level"]),
        statistic_name="sharpe",
    )


def bootstrap_ret_total(
    returns: Sequence[float],
    config: EinherjarConfig,
) -> BootstrapResult:
    """Block bootstrap CI sur le retour total cumulé (somme des rendements)."""
    bs = config.evaluation["bootstrap"]
    block_length = _auto_block_length(len(returns), bs.get("block_length_factor", 1.5))
    ci_low, ci_high, observed = block_bootstrap_ci(
        values=returns,
        statistic=sum,
        n_resamples=int(bs.get("n_resamples", 2000)),
        block_length=block_length,
        ci_level=float(bs.get("ci_level", 0.95)),
        rng_seed=int(bs.get("seed", 42)) + 1,
    )
    return BootstrapResult(
        ci_low=ci_low, ci_high=ci_high, observed=observed,
        n_resamples=int(bs["n_resamples"]),
        block_length=block_length, ci_level=float(bs["ci_level"]),
        statistic_name="ret_total",
    )


def bootstrap_mdd(
    returns: Sequence[float],
    config: EinherjarConfig,
) -> BootstrapResult:
    """Block bootstrap CI sur le max drawdown (sur equity curve reconstruite)."""
    from einherjar.research.utils.metrics import max_drawdown
    bs = config.evaluation["bootstrap"]
    block_length = _auto_block_length(len(returns), bs.get("block_length_factor", 1.5))

    def equity_mdd(r: Sequence[float]) -> float:
        eq = [1.0]
        for x in r:
            eq.append(eq[-1] * (1.0 + x))
        return max_drawdown(eq)

    ci_low, ci_high, observed = block_bootstrap_ci(
        values=returns,
        statistic=equity_mdd,
        n_resamples=int(bs.get("n_resamples", 2000)),
        block_length=block_length,
        ci_level=float(bs.get("ci_level", 0.95)),
        rng_seed=int(bs.get("seed", 42)) + 2,
    )
    return BootstrapResult(
        ci_low=ci_low, ci_high=ci_high, observed=observed,
        n_resamples=int(bs["n_resamples"]),
        block_length=block_length, ci_level=float(bs["ci_level"]),
        statistic_name="max_drawdown",
    )
