"""smart_sampling.py - Sampling intelligent pour les timeframes haute frequence.

2026-08-29 (decision Jovanny, axe 5M de la recherche) :
Le 5M a 37k-1.17M lignes par actif, domine par le bruit de microstructure.
La recherche (Lopez de Prado 2018, "Advances in Financial ML") recommande
de sampler par ACTIVITE (dollar/volume bars) au lieu du temps.

Ce module fournit :
1. dollar_bars() : replique les barres 5m en barres "tous les $N echanges"
   -> reduction 5-20x, returns quasi-IID, signal/bruit ameliore.
2. filter_volume() : garde seulement les barres volume > N x moyenne
   -> filtre le bruit sans resampler (complementaire).

IMPORTANT : ces fonctions retournent des INDICES de barres a garder.
Le pipeline charge ensuite X[keep] et y[keep] - pas de nouvelle
construction de features (les features restent celles des barres 5m).
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Defauts (accord design 2026-08-28)
DEFAULT_DOLLAR_THRESHOLD_MULT = 20.0  # ~20 barres 5m par dollar bar
DEFAULT_VOLUME_MULT = 1.5  # garde les barres volume > 1.5x moyenne


def dollar_bars_indices(
    timestamps: np.ndarray,
    prices: np.ndarray,
    volumes: np.ndarray,
    dollar_threshold: float | None = None,
    threshold_mult: float = DEFAULT_DOLLAR_THRESHOLD_MULT,
) -> np.ndarray:
    """Calcule les indices des dollar bars (fin de chaque barre).

    Une dollar bar se termine quand le volume echangé en $ (price*volume)
    cumule depuis la derniere barre depasse `dollar_threshold`.

    Args:
        timestamps : array des timestamps (pour l'ordre).
        prices : prix (close) de chaque barre.
        volumes : volume de chaque barre.
        dollar_threshold : seuil absolu (sinon auto : median(price*vol) * mult).
        threshold_mult : multiplicateur du seuil auto.

    Returns:
        np.ndarray d'indices [i1, i2, ...] : les barres a GARDER
        (les barres de fin de chaque dollar bar, l'echantillon reduit).
    """
    n = len(prices)
    if n < 100:
        return np.arange(n)

    dollar_vol = np.asarray(prices, dtype=np.float64) * np.asarray(volumes, dtype=np.float64)
    if dollar_threshold is None or dollar_threshold <= 0:
        dollar_threshold = float(np.median(dollar_vol[dollar_vol > 0]) * threshold_mult)
        dollar_threshold = max(dollar_threshold, 1.0)
    logger.info(
        "dollar_bars : seuil=$%.0f (mult=%.1f), %d barres 5m",
        dollar_threshold, threshold_mult, n,
    )

    keep = []
    cum = 0.0
    for i in range(n):
        cum += dollar_vol[i]
        if cum >= dollar_threshold:
            keep.append(i)
            cum = 0.0
    if not keep:
        # Seuil trop haut : au moins garder le dernier index
        return np.array([n - 1])
    arr = np.array(keep, dtype=np.int64)
    logger.info("dollar_bars : %d -> %d barres (reduction %.1fx)", n, len(arr), n / max(1, len(arr)))
    return arr


def filter_volume_indices(
    volumes: np.ndarray,
    volume_mult: float = DEFAULT_VOLUME_MULT,
    window: int = 20,
) -> np.ndarray:
    """Garde les indices des barres volume > mult x moyenne glissante.

    Args:
        volumes : volume de chaque barre.
        volume_mult : multiple de la moyenne (1.5 defaut).
        window : fenetre de la moyenne glissante.

    Returns:
        np.ndarray d'indices des barres a garder.
    """
    n = len(volumes)
    if n < 50:
        return np.arange(n)

    vols = np.asarray(volumes, dtype=np.float64)
    # Moyenne glissante (window) avec NaN au debut
    kernel = np.ones(window) / window
    ma = np.convolve(vols, kernel, mode="same")
    # Corriger le bord gauche (convolution 'same' sous-estime au debut)
    for i in range(min(window, n)):
        ma[i] = vols[: i + 1].mean()
    ma[ma <= 0] = np.nan

    mask = vols > volume_mult * ma
    keep = np.where(mask)[0]
    # Rejetter les barres de buildup (volume nul au debut)
    keep = keep[keep >= window]
    logger.info(
        "filter_volume : %d -> %d barres (mult=%.1f, reduction %.1fx)",
        n, len(keep), volume_mult, n / max(1, len(keep)),
    )
    return keep


def combine_indices(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Intersection (triee) de deux ensembles d'indices."""
    if len(a) == 0 or len(b) == 0:
        return np.array([], dtype=np.int64)
    sa = set(a.tolist())
    sb = set(b.tolist())
    inter = sorted(sa & sb)
    return np.array(inter, dtype=np.int64)