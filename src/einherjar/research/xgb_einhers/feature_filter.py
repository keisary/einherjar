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
    min_pct: float = 0.003,
    max_pct: float = 0.997,
    min_occurrences: int = 100,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Drop les features binaires trop rares ou trop saturées.

    FIX (2026-08-21, problèmes Q4/filter) : un seuil purement en pourcentage
    (`pct_True < 0.5%`) jetait des patterns RARES mais potentiellement très
    rentables (ex. 0.2% de 5M bougies = 10000 occurrences). On ajoute donc un
    critère de NOMBRE ABSOLU d'occurrences : une feature n'est drop que si elle a
    moins de `min_occurrences` valeurs "True" (trop peu pour un split fiable),
    quel que soit le %. La saturation (> max_pct = quasi-constante) reste drop.

    FIX P2-2 (2026-08-24) : min_occurrences 300 -> 100.
    Preuve empirique (event-study BTC/1h/6h) : des patterns significatifs
    (|t|>2, ~20bp/trade conditionnel) avaient 200-250 occurrences et etaient
    tues par l'ancien seuil (ex. pattern_inverted_hammer=249 occ., |t|=2.32).
    100 occurrences reste au-dessus du minimum pour une moyenne fiable.
    Les patterns <100 occ. restent accessibles via le futur generateur
    event-study dedie (Phase 3).

    Returns:
        (X_filtered, kept_names, dropped_names)
    """
    n_features = X.shape[1]
    n_rows = X.shape[0]
    keep_mask = np.ones(n_features, dtype=bool)
    dropped_info = []
    for i in range(n_features):
        col = X[:, i]
        if not is_binary_feature(col):
            continue
        pct = compute_sparsity(col)
        # P2-2 : compter aussi les occurrences -1.0 (pattern actif sens inverse)
        n_occ = int(round(pct * n_rows)) + int((col < -0.5).sum())
        # FIX (2026-08-21) : on ne droppe PLUS par frequence relative basse
        # (`pct < min_pct` jetait des patterns rares mais massifs en absolu).
        # On ne drop que si : quasi-constante (saturation) OU trop peu
        # d'occurrences absolues pour un split stable.
        if pct > max_pct or n_occ < max(1, min_occurrences):
            keep_mask[i] = False
            dropped_info.append((feature_names[i], pct, n_occ))
    kept_names = [n for n, k in zip(feature_names, keep_mask) if k]
    dropped_names = [n for n, k in zip(feature_names, keep_mask) if not k]
    logger.info(
        "filter_sparse_patterns : %d/%d features dropped (sparsity<%.1f%% ou >%.1f%% ou n_occ<%d)",
        len(dropped_names), n_features, min_pct * 100, (1 - max_pct) * 100, min_occurrences,
    )
    if dropped_info:
        for name, pct, n_occ in dropped_info[:5]:
            logger.debug("  dropped %s (pct=%.4f, n_occ=%d)", name, pct, n_occ)
    return X[:, keep_mask], kept_names, dropped_names
