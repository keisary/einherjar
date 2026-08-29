"""feature_reduction.py - Reduction de features par IC pour petits echantillons.

2026-08-29 (decision Jovanny, axe 1D de la recherche) :
Le 1D a ~2000 lignes/actif pour 213 features (ratio obs/features ~9:1,
dangereux pour XGBoost). La recherche (Gu, Kelly & Xiu 2020 + pratique
quant) recommande de reduire a 30-50 features AVANT l'entrainement.

Methode (sans leakage) :
1. IC univarie : Spearman(feature, target_future_return) sur TRAIN uniquement
2. |IC| > seuil (0.02 par defaut) ET t-stat de l'IC > 2.0
3. Dedup par correlation (|r| > 0.7) en gardant la feature a IC le plus haut
4. Si trop de features restent : garder le top-N (40 par defaut)

Le target utilise le rendement futur (Y_ret de l'horizon) sur la fenetre
train. JAMAIS sur val/holdout -> pas de leakage.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Seuils par defaut (accord design 2026-08-28)
MIN_ABS_IC = 0.02
MIN_IC_TSTAT = 2.0
MAX_FEATURES = 40
CORR_DEDUP = 0.70


def _spearman_ic(x: np.ndarray, y: np.ndarray) -> float:
    """IC = correlation de Spearman entre feature et cible.

    Retourne l'IC (float) ou 0.0 si calcul impossible.
    """
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 30:
        return 0.0
    xv = x[mask]
    yv = y[mask]
    if np.all(xv == xv[0]) or np.all(yv == yv[0]):
        return 0.0
    # Spearman = Pearson sur les rangs
    from scipy.stats import rankdata

    rx = rankdata(xv)
    ry = rankdata(yv)
    xm = rx - rx.mean()
    ym = ry - ry.mean()
    denom = np.sqrt((xm**2).sum() * (ym**2).sum())
    if denom == 0:
        return 0.0
    return float((xm * ym).sum() / denom)


def select_features_by_ic(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    min_abs_ic: float = MIN_ABS_IC,
    min_ic_tstat: float = MIN_IC_TSTAT,
    max_features: int = MAX_FEATURES,
    corr_dedup: float = CORR_DEDUP,
) -> tuple[list[str], np.ndarray]:
    """Selectionne les features par IC univarie + dedup correlation.

    Args:
        X_train : (N, F) features d'entrainement.
        y_train : (N,) cible (rendement futur sur l'horizon).
        feature_names : noms des features (F,).
        min_abs_ic : |IC| minimum pour garder une feature.
        min_ic_tstat : t-stat de l'IC minimum.
        max_features : nombre max de features a garder.
        corr_dedup : seuil de correlation pour dedupliquer.

    Returns:
        (kept_names, keep_idx) : noms conserves et indices dans X.
    """
    n, f = X_train.shape
    if f == 0 or n < 50:
        return list(feature_names), np.arange(f)

    # 1. IC univarie sur train uniquement
    ics = np.zeros(f)
    tstats = np.zeros(f)
    for j in range(f):
        ic = _spearman_ic(X_train[:, j], y_train)
        ics[j] = ic
        # t-stat approxime : IC * sqrt(n_eff / (1 - IC^2))
        if abs(ic) > 1e-9:
            n_eff = max(30, int(n * (1 - 0.5)))  # autocorrelation approx
            tstats[j] = ic * np.sqrt(n_eff / max(1e-9, 1 - ic * ic))

    # 2. Filtre |IC| > seuil et t-stat > seuil
    keep = np.where(
        (np.abs(ics) >= min_abs_ic) & (np.abs(tstats) >= min_ic_tstat)
    )[0]
    logger.info(
        "IC selection : %d/%d features passent (|IC|>=%.3f, t>=%.1f)",
        len(keep), f, min_abs_ic, min_ic_tstat,
    )
    if len(keep) == 0:
        # Fallback : top 15 par |IC| absolu
        order = np.argsort(-np.abs(ics))[:15]
        keep = order
        logger.warning("  Aucune feature au seuil, fallback top-15 |IC|")

    # 3. Dedup par correlation (garde la feature a |IC| max)
    if len(keep) > 1:
        keep_sorted = sorted(keep, key=lambda j: -abs(ics[j]))
        deduped = []
        kept_vectors: list[np.ndarray] = []
        for j in keep_sorted:
            xj = X_train[:, j]
            redundant = False
            for xk in kept_vectors:
                mask = ~(np.isnan(xj) | np.isnan(xk) | np.isnan(y_train))
                if mask.sum() < 30:
                    continue
                a = xj[mask]
                b = xk[mask]
                if np.all(a == a[0]) or np.all(b == b[0]):
                    continue
                r = np.corrcoef(a, b)[0, 1]
                if abs(r) > corr_dedup:
                    redundant = True
                    break
            if not redundant:
                deduped.append(j)
                kept_vectors.append(xj)
        keep = np.array(deduped, dtype=int)
        logger.info("  Apres dedup correlation (|r|>%.2f) : %d features", corr_dedup, len(keep))

    # 4. Cap top-N
    if len(keep) > max_features:
        keep = np.array(sorted(keep, key=lambda j: -abs(ics[j]))[:max_features])
        logger.info("  Cap top-%d |IC| : %d features gardees", max_features, len(keep))

    keep = np.sort(keep)
    kept_names = [feature_names[j] for j in keep]
    logger.info(
        "Feature reduction : %d -> %d features (%s...)",
        f, len(keep), ", ".join(kept_names[:5]),
    )
    return kept_names, keep