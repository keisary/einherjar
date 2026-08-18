"""multiple_testing.py - Corrections pour le multiple testing bias.

Sprint 3.1 - P1 (correction multi-tests).

Quand on genere ~30 candidats par run x 4 horizons x 5 seeds = 600 hypotheses,
le taux de faux positifs explose si on n'applique pas de correction.

Reference : Benjamini & Hochberg (1995) "Controlling the False Discovery Rate"
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)


def bootstrap_pvalue(
    returns: np.ndarray,
    n_bootstrap: int = 1000,
    statistic: str = "mean",
    random_state: int = 42,
) -> float:
    """P-value bootstrap pour H0: statistic(returns) <= 0.

    Args:
        returns : array des rendements par trade (net)
        n_bootstrap : nb de replications
        statistic : "mean" (H0: mean <= 0) ou "sharpe" (H0: sharpe <= 0)
        random_state : seed

    Returns:
        p-value (float dans [0, 1])
    """
    n = len(returns)
    if n < 2:
        return 1.0  # pas assez de samples
    rng = np.random.default_rng(random_state)
    obs_stat = float(np.mean(returns)) if statistic == "mean" else _compute_sharpe(returns)
    count = 0
    for _ in range(n_bootstrap):
        # Resample avec replacement, centrer sur 0 (H0)
        sample = rng.choice(returns, size=n, replace=True)
        # Centrer sur 0 : on enleve la moyenne observee pour tester H0: mean = 0
        sample_centered = sample - np.mean(returns)
        boot_stat = float(np.mean(sample_centered)) if statistic == "mean" else _compute_sharpe(sample_centered)
        if abs(boot_stat) >= abs(obs_stat):
            count += 1
    return count / n_bootstrap


def _compute_sharpe(returns: np.ndarray) -> float:
    """Sharpe simple (mean/std) sans annualisation."""
    if len(returns) < 2:
        return 0.0
    std = float(np.std(returns, ddof=1))
    if std == 0:
        return 0.0
    return float(np.mean(returns) / std)


def benjamini_hochberg(
    pvalues: Sequence[float],
    fdr: float = 0.05,
) -> list[bool]:
    """Benjamini-Hochberg : identifie les hypotheses rejetees (significatives).

    Controle le False Discovery Rate (FDR) au niveau `fdr`.
    Renvoie une liste de booleens de meme longueur que pvalues,
    True = hypothese rejetee (significative).

    Args:
        pvalues : sequence de p-values (une par Einher)
        fdr : taux de faux positifs acceptable (default 0.05)

    Returns:
        list[bool] : True si l'hypothese est rejetee (significative)
    """
    n = len(pvalues)
    if n == 0:
        return []
    pvals = np.asarray(pvalues, dtype=float)
    # Trier les p-values
    sorted_idx = np.argsort(pvals)
    sorted_pvals = pvals[sorted_idx]
    # Critere BH : p_(i) <= (i+1)/n * fdr
    # Le plus grand i qui satisfait : rejeter toutes les hypotheses <= i
    thresholds = np.arange(1, n + 1) / n * fdr
    # Trouver le max i tel que sorted_pvals[i] <= thresholds[i]
    significant_sorted = sorted_pvals <= thresholds
    if not significant_sorted.any():
        return [False] * n
    # Rejeter toutes les hypotheses jusqu'au dernier significant
    max_significant_idx = np.where(significant_sorted)[0].max()
    rejected_sorted = np.zeros(n, dtype=bool)
    rejected_sorted[:max_significant_idx + 1] = True
    # Remapper a l'ordre original
    rejected = np.zeros(n, dtype=bool)
    rejected[sorted_idx] = rejected_sorted
    n_rejected = int(rejected.sum())
    logger.info(
        "Benjamini-Hochberg : %d/%d hypotheses rejetees au FDR=%.3f",
        n_rejected, n, fdr,
    )
    return rejected.tolist()


def apply_bh_to_einhers(
    einhers: list,
    fdr: float = 0.05,
    n_bootstrap: int = 1000,
    random_state: int = 42,
) -> tuple[list, list[float], list[bool]]:
    """Applique BH sur une liste d'Einhers.

    Pour chaque Einher, calcule la p-value bootstrap (H0: mean_return <= 0)
    sur les rendements de trades val. Puis applique BH.

    Args:
        einhers : liste d'Einhers avec .metrics
        fdr : seuil FDR
        n_bootstrap : nb de replications bootstrap
        random_state : seed

    Returns:
        (einhers_filtered, pvalues, rejected)
        - einhers_filtered : Einhers qui passent BH
        - pvalues : p-value de chaque Einher (dans l'ordre original)
        - rejected : True/False pour chaque Einher
    """
    if not einhers:
        return [], [], []
    pvalues = []
    for e in einhers:
        # On utilise les rendements val (stockes dans source si dispo, sinon metrics)
        rets = getattr(e, "_trade_returns", None)
        if rets is None:
            # Fallback : approximation a partir de sharpe et n_trades
            # Si sharpe > 0 et n > 30, on considere p-value faible
            sharpe = e.metrics.sharpe_ratio
            n = e.metrics.n_trades
            if n < 2 or sharpe <= 0:
                pvalue = 1.0
            else:
                # Approximation : p-value = 2 * (1 - Phi(|sharpe|))
                # ou plus simple : p-value = exp(-sharpe) borne
                from math import erf, sqrt
                pvalue = 2 * (1 - 0.5 * (1 + erf(abs(sharpe) / sqrt(2))))
            pvalues.append(pvalue)
        else:
            pvalues.append(bootstrap_pvalue(np.asarray(rets), n_bootstrap=n_bootstrap, random_state=random_state))
    rejected = benjamini_hochberg(pvalues, fdr=fdr)
    einhers_filtered = [e for e, r in zip(einhers, rejected) if r]
    return einhers_filtered, pvalues, rejected
