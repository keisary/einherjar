"""feature_filter.py - Filtrage des features inutiles.

Sprint 2.3.1.

Cible les patterns trop rares (pct_True < 0.5%) qui ne peuvent pas
aider XGBoost (un split sur "== 1" n'isole RIEN si 99.5% des valeurs
sont 0).

Note : on ne drop PAS les features par importance (trop aggressif),
on drop uniquement celles qui sont STATISTIQUEMENT mortes.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# Seuil en dessous duquel une feature binaire est consideree comme trop rare
SPARSITY_THRESHOLD = 0.005  # 0.5%


def is_binary_feature(col: np.ndarray, tolerance: float = 1e-6) -> bool:
    """Detecte si une feature est binaire (0/1 ou 0.0/1.0)."""
    unique = np.unique(col)
    if len(unique) > 3:
        return False
    vals = set(round(float(v), 6) for v in unique)
    return vals.issubset({0.0, 1.0, -1.0}) or vals.issubset({0.0, 1.0})


def compute_sparsity(col: np.ndarray) -> float:
    """Pour une feature binaire : pct de True (1.0)."""
    return float(np.mean(col > 0.5))


def filter_sparse_patterns(
    X: np.ndarray,
    feature_names: list[str],
    threshold: float = SPARSITY_THRESHOLD,
    min_pct: float = 0.005,
    max_pct: float = 0.995,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Drop les features binaires trop sparse ou trop saturées.

    - pct_True < min_pct (0.5%) : trop rare, XGBoost ne peut rien en faire
    - pct_True > max_pct (99.5%) : trop sature (quasi-constante)

    Returns:
        (X_filtered, kept_names, dropped_names)
    """
    n_features = X.shape[1]
    keep_mask = np.ones(n_features, dtype=bool)
    dropped_info = []
    for i in range(n_features):
        col = X[:, i]
        if not is_binary_feature(col):
            continue
        pct = compute_sparsity(col)
        if pct < min_pct or pct > max_pct:
            keep_mask[i] = False
            dropped_info.append((feature_names[i], pct))
    kept_names = [n for n, k in zip(feature_names, keep_mask) if k]
    dropped_names = [n for n, k in zip(feature_names, keep_mask) if not k]
    logger.info(
        "filter_sparse_patterns : %d/%d features dropped (sparsity < %.1f%% ou > %.1f%%)",
        len(dropped_names), n_features, min_pct * 100, (1 - max_pct) * 100,
    )
    if dropped_info:
        for name, pct in dropped_info[:5]:
            logger.debug("  dropped %s (pct=%.4f)", name, pct)
    return X[:, keep_mask], kept_names, dropped_names
