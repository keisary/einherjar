"""data/threshold_calibration.py — Calibration des seuils par feature sur le train.

Implémente P1 #1 : les seuils ne sont plus tirés uniformément entre -2 et 2,
mais calculés comme quantiles de la distribution observée de chaque feature
sur le train. C'est le fondement des règles valides (anti-tautologies BNF).

Avantages :
  - Les seuils sont dans la distribution réelle des features (pas de "rsi > 0" qui
    catch 100% des bougies).
  - Les générateurs produisent des règles SEMANTIQUEMENT valides dès l'init.
  - Cohérence avec la BNF (à venir) : les seuils seront contraintes par la grammaire.

Usage typique :
    quantiles = compute_feature_quantiles(train_features, [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
    # quantiles['rsi_14'] = [10.2, 19.5, 28.7, 49.8, 71.3, 80.5, 89.8]
    # quantiles['momentum_10'] = [-0.05, -0.03, -0.01, 0.001, 0.02, 0.04, 0.06]
    # → les générateurs tirent leurs seuils depuis ces listes au lieu d'uniformes.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

from einherjar.research.data.features import FeaturesFrame
from einherjar.research.utils.stats import percentile

logger = logging.getLogger(__name__)


# Quantiles par défaut (étendues usuelles en analyse de données).
# On exclut 0.0 et 1.0 pour éviter les min/max exacts (peu informatifs).
DEFAULT_QUANTILES: tuple[float, ...] = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)


def compute_feature_quantiles(
    features: FeaturesFrame,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
) -> dict[str, list[float]]:
    """Calcule les quantiles de chaque feature sur la série.

    Args:
        features: FeaturesFrame (typiquement le train).
        quantiles: Liste de quantiles à calculer (entre 0 et 1).

    Returns:
        Dict {feature_name: [q1_value, q2_value, ...]} aligné sur quantiles.

    Raises:
        ValueError: si quantiles contient des valeurs hors ]0, 1[.
    """
    if not all(0.0 < q < 1.0 for q in quantiles):
        raise ValueError(f"quantiles doivent être dans ]0, 1[, got {quantiles}")
    result: dict[str, list[float]] = {}
    for name in features.feature_names:
        col = features.column(name).to_numpy()
        # Filtre les NaN/inf.
        clean = col[~np.isnan(col) & ~np.isinf(col)]
        if len(clean) < 2:
            # Pas assez de données : on met une liste vide.
            result[name] = []
            continue
        result[name] = [percentile(clean.tolist(), q * 100.0) for q in quantiles]
    logger.info(
        "Seuils calibrés sur %d features x %d quantiles (train=%d bougies)",
        sum(1 for v in result.values() if v), len(quantiles), features.n_bougies,
    )
    return result


def merge_quantile_pools(
    quantiles: dict[str, list[float]],
    fallback_pool: Sequence[float] = (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0),
) -> dict[str, list[float]]:
    """Fusionne les quantiles calibrés avec un pool de fallback par défaut.

    Pour chaque feature : si elle a ≥ 2 quantiles calibrés, on les utilise
    directement. Sinon, on prend le pool de fallback (utile pour les features
    avec peu de données ou des distributions dégénérées).

    Args:
        quantiles: Sortie de `compute_feature_quantiles`.
        fallback_pool: Pool de seuils par défaut si pas de quantiles calibrés.

    Returns:
        Dict {feature_name: [seuil1, seuil2, ...]} prêt à être échantillonné.
    """
    return {
        name: (qs if len(qs) >= 2 else list(fallback_pool))
        for name, qs in quantiles.items()
    }


def sample_threshold(
    pool: Sequence[float],
    rng,
) -> float:
    """Tire un seuil aléatoire dans un pool (uniforme parmi les valeurs)."""
    return float(rng.choice(list(pool)))
