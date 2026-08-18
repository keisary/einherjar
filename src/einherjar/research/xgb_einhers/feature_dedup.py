"""feature_dedup.py - Anti-duplication de features par matrice de correlation.

Sprint 2.2.1.

Strategie :
- Calculer la matrice de correlation |r| sur X (Pearson).
- Pour chaque paire (i, j) avec |r| > threshold, on garde la feature
  avec la plus haute importance (gain XGBoost) et on drop l'autre.
- Iterer jusqu'a stabilisation (peut creer des cascades).

Pourquoi : 213 features -> beaucoup de redondance (ex: RSI_14 et RSI_28
sont correlees a 0.95+). Si on garde tout, XGBoost peut se focaliser
sur des features quasi-identiques et overfitter.

Note : on ne drop PAS les features sur la base de la correlation seule,
on preserve le signal en gardant la plus importante.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def compute_corr_matrix(X: np.ndarray) -> np.ndarray:
    """Matrice de correlation |r| (Pearson) sur X.

    Returns:
        corr : (F, F) float64 avec 0 sur la diagonale.
    """
    # np.corrcoef avec rowvar=False : features en colonnes
    # Il gere les std=0 en retournant nan, qu'on remplace par 0
    corr = np.corrcoef(X, rowvar=False)
    # Remplacer les NaN (features constantes) par 0
    corr = np.nan_to_num(corr, nan=0.0)
    # Forcer la diagonale a 0 (auto-correlation)
    np.fill_diagonal(corr, 0.0)
    # Valeur absolue
    return np.abs(corr)


def find_duplicate_pairs(
    corr: np.ndarray,
    feature_names: list[str],
    threshold: float = 0.85,
) -> list[tuple[str, str, float]]:
    """Trouve les paires de features correlees au-dessus du seuil.

    Returns:
        Liste de (feat_a, feat_b, |r|) pour |r| > threshold, i < j.
    """
    F = corr.shape[0]
    pairs = []
    for i in range(F):
        for j in range(i + 1, F):
            r = corr[i, j]
            if r > threshold:
                pairs.append((feature_names[i], feature_names[j], float(r)))
    pairs.sort(key=lambda p: -p[2])
    return pairs


def select_features_to_drop(
    X: np.ndarray,
    feature_names: list[str],
    importances: dict[str, float],
    corr_threshold: float = 0.85,
) -> list[str]:
    """Selectionne les features a drop pour eviter la duplication.

    Strategie gloutonne : tant qu'il existe une paire |r| > threshold,
    drop la moins importante des deux.

    Returns:
        Liste des noms de features a dropper (a enlever de X).
    """
    F = X.shape[1]
    if F != len(feature_names):
        raise ValueError(f"X et feature_names ont des tailles differentes : {F} vs {len(feature_names)}")
    # Importance par defaut = 0 si non fournie
    imp_vec = np.array([importances.get(name, 0.0) for name in feature_names], dtype=np.float64)
    keep_mask = np.ones(F, dtype=bool)
    current_names = list(feature_names)
    current_X = X
    current_imp = imp_vec.copy()

    # Iterer jusqu'a stabilisation
    while True:
        if current_X.shape[1] < 2:
            break
        corr = compute_corr_matrix(current_X)
        # Trouver la paire la plus correlee
        F_curr = corr.shape[0]
        max_r = 0.0
        max_i, max_j = -1, -1
        for i in range(F_curr):
            for j in range(i + 1, F_curr):
                if corr[i, j] > max_r:
                    max_r = corr[i, j]
                    max_i, max_j = i, j
        if max_r <= corr_threshold:
            break
        # Drop la moins importante
        if current_imp[max_i] < current_imp[max_j]:
            drop_idx = max_i
        else:
            drop_idx = max_j
        # Retirer
        keep_mask_global = np.array([
            keep_mask_global for keep_mask_global, name in zip(keep_mask, feature_names)
            if name in current_names
        ])
        # Reconstruire keep_mask dans l'espace original
        new_keep = np.zeros(F, dtype=bool)
        new_names = []
        new_imp = []
        for k, name in enumerate(current_names):
            if k != drop_idx:
                global_idx = feature_names.index(name)
                new_keep[global_idx] = True
                new_names.append(name)
                new_imp.append(current_imp[k])
        keep_mask = new_keep
        current_names = new_names
        current_X = X[:, keep_mask]
        current_imp = np.array(new_imp, dtype=np.float64)
        logger.debug(
            "Dedup : drop %s (|r|=%.3f avec %s), reste %d features",
            current_names[drop_idx] if drop_idx < len(current_names) else "?",
            max_r,
            "?",
            len(current_names),
        )

    dropped = [name for name, keep in zip(feature_names, keep_mask) if not keep]
    return dropped


def apply_dedup(
    X: np.ndarray,
    feature_names: list[str],
    importances: dict[str, float],
    corr_threshold: float = 0.85,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Pipeline complet : retourne X dedup + noms retenus + noms droppes.

    Returns:
        (X_dedup, kept_names, dropped_names)
    """
    dropped = select_features_to_drop(X, feature_names, importances, corr_threshold)
    keep_idx = [i for i, name in enumerate(feature_names) if name not in set(dropped)]
    return X[:, keep_idx], [feature_names[i] for i in keep_idx], dropped
