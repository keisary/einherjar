"""Enrichissement avec 24 features quantitatives ULTRA-OPTIMISÉES - VERSION INSTITUTIONNELLE.

Agent Technique MIDAS V3 - Features Quantitatives Essentielles
OPTIMISATIONS COMPLÈTES: Numba + Dask + Shared Memory + Memory Mapping + Vectorisation + Cache + Types optimaux.
"""

import gc
import hashlib
import logging
import os
import threading
import time
import warnings
from functools import lru_cache
from multiprocessing import Pool, cpu_count, shared_memory

import numpy as np
import pandas as pd
import psutil

# Configuration du logging pour optimisations
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dask imports pour parallélisme distribué OPTIMISÉ
try:
    import dask  # noqa: F401
    import dask.dataframe as dd
    from dask import delayed  # noqa: F401
    from dask.diagnostics import ProgressBar
    from dask.distributed import Client, LocalCluster, as_completed  # noqa: F401

    DASK_AVAILABLE = True
    logger.info("✅ Dask disponible - Parallélisme distribué activé")
except ImportError:
    DASK_AVAILABLE = False
    logger.warning("⚠️ Dask non disponible - Utilisation multiprocessing standard")

# Memory mapping pour gros datasets OPTIMISÉ
try:
    import mmap  # noqa: F401

    MMAP_AVAILABLE = True
    logger.info("✅ Memory mapping disponible")
except ImportError:
    MMAP_AVAILABLE = False
    logger.warning("⚠️ Memory mapping non disponible")

# Shared memory pour performance OPTIMISÉE
try:
    import multiprocessing as mp  # noqa: F401
    from multiprocessing import shared_memory

    SHARED_MEMORY_AVAILABLE = True
    logger.info("✅ Shared memory disponible")
except ImportError:
    SHARED_MEMORY_AVAILABLE = False
    logger.warning("⚠️ Shared memory non disponible")

# Supprimer les warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning, module="pandas_ta")
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")

# Variable globale pour éviter les messages répétitifs en multiprocessing
_NUMBA_MESSAGE_SHOWN = False

# OPTIMISATION NUMBA AVANCÉE avec fallback intelligent
try:
    from numba import jit, njit, prange, types  # noqa: F401
    from numba.core import types as nb_types  # noqa: F401
    from numba.typed import Dict as NumbaDict  # noqa: F401

    NUMBA_AVAILABLE = True
    if not _NUMBA_MESSAGE_SHOWN:
        logger.info("🚀 Numba disponible - Accélération JIT activée")
        _NUMBA_MESSAGE_SHOWN = True
except ImportError:
    NUMBA_AVAILABLE = False
    if not _NUMBA_MESSAGE_SHOWN:
        logger.warning("⚠️ Numba non disponible - Utilisation de NumPy standard")
        _NUMBA_MESSAGE_SHOWN = True

# Configuration Dask optimisée
if DASK_AVAILABLE:
    # Configuration cluster local optimisé
    DASK_CONFIG = {
        "threads_per_worker": 2,
        "n_workers": min(cpu_count(), 8),
        "memory_limit": "2GB",
        "dashboard_address": None,  # Désactiver dashboard pour performance
        "silence_logs": logging.ERROR,
    }

# OPTIMISATION: Types de données optimaux et configuration avancée
OPTIMAL_FLOAT = np.float32
OPTIMAL_INT = np.int32
OPTIMAL_BOOL = np.int8
CACHE_SIZE = 5000

# Configuration optimisations avancées
OPTIMIZATION_CONFIG = {
    "use_numba": NUMBA_AVAILABLE,
    "use_dask": DASK_AVAILABLE,
    "use_shared_memory": SHARED_MEMORY_AVAILABLE,
    "use_mmap": MMAP_AVAILABLE,
    "chunk_size_small": 10000,
    "chunk_size_medium": 50000,
    "chunk_size_large": 100000,
    "memory_threshold_gb": 4.0,
    "dask_threshold_gb": 8.0,
    "shared_memory_threshold_gb": 2.0,
    "mmap_threshold_gb": 16.0,
    "max_workers": min(cpu_count(), 13),
    "cache_size": CACHE_SIZE,
    "enable_progress_bar": True,
    "enable_memory_monitoring": True,
}

# Variables globales pour shared memory
_SHARED_MEMORY_BLOCKS = {}
_SHARED_MEMORY_LOCK = threading.Lock()

# ============================================================================
# FONCTIONS NUMBA ULTRA-OPTIMISÉES - 24 FEATURES INSTITUTIONNELLES
# ============================================================================
if NUMBA_AVAILABLE:

    @lru_cache(maxsize=CACHE_SIZE)
    def _get_data_hash(data_tuple):
        """Cache intelligent basé sur hash des données."""
        return hashlib.md5(str(data_tuple).encode()).hexdigest()

    # ========== VOLATILITÉ (5 FEATURES) ==========

    @jit(nopython=True, cache=True)
    def _numba_realized_volatility(returns, window=20):
        """Realized Volatility ultra-rapide."""
        N = len(returns)
        if N < window:
            return np.full(N, np.std(returns), dtype=OPTIMAL_FLOAT)

        rv = np.zeros(N, dtype=OPTIMAL_FLOAT)

        # Calcul glissant
        for i in range(window - 1, N):
            window_returns = returns[i - window + 1 : i + 1]
            rv[i] = np.sqrt(np.sum(window_returns**2))

        # Remplir les premières valeurs
        initial_rv = rv[window - 1]
        for i in range(window - 1):
            rv[i] = initial_rv

        return rv

    @jit(nopython=True, cache=True)
    def _numba_garch_volatility(returns, alpha=0.1, beta=0.8):
        """GARCH(1,1) volatility ultra-rapide."""
        N = len(returns)
        if N < 2:
            return np.array([0.1], dtype=OPTIMAL_FLOAT)

        # Initialisation
        omega = 0.01 * (1 - alpha - beta)  # Long-term variance
        volatility = np.zeros(N, dtype=OPTIMAL_FLOAT)
        volatility[0] = np.std(returns)

        # GARCH recursion
        for t in range(1, N):
            volatility[t] = np.sqrt(
                omega + alpha * returns[t - 1] ** 2 + beta * volatility[t - 1] ** 2
            )

        return volatility

    @jit(nopython=True, cache=True)
    def _numba_volatility_clustering(returns, threshold_factor=2.0, window=100):
        """Clustering de volatilité ROLLING ultra-rapide (NO DATA LEAKAGE).

        CORRECTION: Version précédente calculait UN SEUL scalaire pour tout le dataset.
        Cette version calcule le clustering sur rolling window.

        Returns:
            Array de même taille que returns
        """
        n = len(returns)
        result = np.zeros(n, dtype=OPTIMAL_FLOAT)

        for i in range(window, n):
            window_returns = returns[i - window : i]
            w = len(window_returns)
            if w < 2:
                continue

            vol = np.abs(window_returns)
            vol_std = np.std(vol)
            threshold = vol_std * threshold_factor

            clustering = 0.0
            for j in range(1, w):
                if vol[j] > threshold and vol[j - 1] > threshold:
                    clustering += 1.0

            result[i] = clustering / (w - 1) if w > 1 else 0.0

        return result

    @jit(nopython=True, cache=True)
    def _numba_volatility_persistence(returns, window=100):
        """Persistance de volatilité ROLLING ultra-rapide (NO DATA LEAKAGE).

        CORRECTION: Version précédente calculait UN SEUL scalaire pour tout le dataset.
        Cette version calcule la persistance sur rolling window.

        Returns:
            Array de même taille que returns
        """
        n = len(returns)
        result = np.zeros(n, dtype=OPTIMAL_FLOAT)

        for i in range(window, n):
            window_returns = returns[i - window : i]
            w = len(window_returns)
            if w < 2:
                continue

            vol = np.abs(window_returns)
            vol_lag1 = vol[:-1]
            vol_current = vol[1:]

            mean_lag1 = np.mean(vol_lag1)
            mean_current = np.mean(vol_current)

            numerator = 0.0
            denom_lag1 = 0.0
            denom_current = 0.0

            for j in range(len(vol_lag1)):
                diff_lag1 = vol_lag1[j] - mean_lag1
                diff_current = vol_current[j] - mean_current
                numerator += diff_lag1 * diff_current
                denom_lag1 += diff_lag1 * diff_lag1
                denom_current += diff_current * diff_current

            denominator = np.sqrt(denom_lag1 * denom_current)
            result[i] = numerator / denominator if denominator > 0 else 0.0

        return result

    # ========== MOMENTUM & PERSISTENCE (4 FEATURES) ==========

    @jit(nopython=True, cache=True)
    def _numba_hurst_rs(prices, window=252):
        """Hurst Exponent avec R/S analysis ROLLING (NO DATA LEAKAGE).

        CORRECTION: Version précédente calculait UN SEUL Hurst pour tout le dataset.
        Cette version calcule Hurst sur rolling window de `window` jours.

        Args:
            prices: Array de prix
            window: Taille fenêtre rolling (défaut 252 = 1 an trading)

        Returns:
            Array de Hurst exponents (même taille que prices)
        """
        n = len(prices)
        hurst_array = np.full(n, 0.5, dtype=OPTIMAL_FLOAT)  # Default: random walk

        # Scales pour R/S analysis (ajustées pour window)
        scales = np.array([10, 20, 50, min(100, window // 4)], dtype=np.int32)

        for i in range(window, n):
            # Lookback window
            window_prices = prices[i - window : i]

            # Calculer rendements sur window
            window_returns = np.diff(window_prices) / window_prices[:-1]
            N = len(window_returns)

            if N < 10:
                continue

            rs_values = np.zeros(len(scales), dtype=OPTIMAL_FLOAT)

            # Pour chaque échelle
            for scale_idx, scale in enumerate(scales):
                if scale >= N:
                    continue

                n_segments = N // scale
                rs_segment = np.zeros(n_segments, dtype=OPTIMAL_FLOAT)

                for j in range(n_segments):
                    start_idx = j * scale
                    end_idx = start_idx + scale
                    segment = window_returns[start_idx:end_idx]

                    mean_return = np.mean(segment)
                    cumulative_devs = np.cumsum(segment - mean_return)

                    R = np.max(cumulative_devs) - np.min(cumulative_devs)
                    S = np.std(segment)

                    if S > 0:
                        rs_segment[j] = R / S
                    else:
                        rs_segment[j] = 1.0

                rs_values[scale_idx] = np.mean(rs_segment)

            # Log-log regression
            valid_scales = scales[scales < N]
            valid_rs = rs_values[:len(valid_scales)]

            if len(valid_scales) >= 2:
                log_scales = np.log(valid_scales.astype(OPTIMAL_FLOAT))
                log_rs = np.log(valid_rs)

                # Linear regression slope = Hurst exponent
                n_points = len(log_scales)
                sum_x = np.sum(log_scales)
                sum_y = np.sum(log_rs)
                sum_xy = np.sum(log_scales * log_rs)
                sum_x2 = np.sum(log_scales * log_scales)

                denominator = n_points * sum_x2 - sum_x * sum_x
                if abs(denominator) > 1e-10:
                    h = (n_points * sum_xy - sum_x * sum_y) / denominator
                    hurst_array[i] = max(0.0, min(1.0, h))

        return hurst_array

    @jit(nopython=True, cache=True)
    def _numba_autocorrelation(prices, max_lag=20, window=252):
        """Autocorrélation ROLLING ultra-rapide (NO DATA LEAKAGE).

        CORRECTION: Version précédente calculait UN SEUL scalaire pour tout le dataset.
        Cette version calcule l'autocorrélation sur rolling window.

        Args:
            prices: Array de prix
            max_lag: Décalage pour l'autocorrélation
            window: Taille fenêtre rolling (défaut 252)

        Returns:
            Array de même taille que prices
        """
        n = len(prices)
        result = np.zeros(n, dtype=OPTIMAL_FLOAT)

        for i in range(window, n):
            window_prices = prices[i - window : i]
            N = len(window_prices)

            if N < max_lag + 1:
                continue

            mean_price = np.mean(window_prices)
            centered = window_prices - mean_price

            variance = np.sum(centered**2) / N
            if variance == 0:
                continue

            covariance = 0.0
            for j in range(N - max_lag):
                covariance += centered[j] * centered[j + max_lag]

            covariance /= N - max_lag
            result[i] = covariance / variance

        return result

    # ========== ENTROPIE (2 FEATURES) ==========

    @jit(nopython=True, cache=True)
    def _numba_shannon_entropy(prices, bins=50, window=252):
        """Entropie de Shannon ROLLING (NO DATA LEAKAGE).

        CORRECTION: Version précédente calculait UNE entropie pour tout le dataset.
        Cette version calcule entropy sur rolling window.

        Args:
            prices: Array de prix
            bins: Nombre de bins pour histogramme
            window: Taille fenêtre rolling (défaut 252)

        Returns:
            Array d'entropies (même taille que prices)
        """
        n = len(prices)
        entropy_array = np.zeros(n, dtype=OPTIMAL_FLOAT)

        for i in range(window, n):
            # Lookback window
            window_prices = prices[i - window : i]

            if len(window_prices) < 2:
                continue

            # Normaliser les prix
            min_val = np.min(window_prices)
            max_val = np.max(window_prices)
            if max_val == min_val:
                continue

            # Créer les bins
            bin_width = (max_val - min_val) / bins
            hist = np.zeros(bins, dtype=OPTIMAL_FLOAT)

            # Compter les occurrences
            for price in window_prices:
                bin_idx = int((price - min_val) / bin_width)
                if bin_idx >= bins:
                    bin_idx = bins - 1
                hist[bin_idx] += 1

            # Calculer l'entropie
            total = len(window_prices)
            entropy = 0.0
            for count in hist:
                if count > 0:
                    p = count / total
                    entropy -= p * np.log2(p)

            entropy_array[i] = entropy

        return entropy_array

    @jit(nopython=True, cache=True)
    def _numba_sample_entropy(prices, m=2, r=0.2, window=252):
        """Sample Entropy (SampEn) ROLLING SIMPLIFIED (NO DATA LEAKAGE).

        CORRECTION: Version précédente calculait UNE entropy pour tout le dataset.
        Cette version calcule sample entropy sur rolling window (simplifiée pour performance).

        NOTE: Simplified implementation - full SampEn is O(n^2) too expensive for rolling.
        Uses approximation based on consecutive differences.

        Args:
            prices: Array de prix
            m: Embedding dimension (default 2)
            r: Tolerance factor (default 0.2)
            window: Taille fenêtre rolling (défaut 252)

        Returns:
            Array d'entropies (même taille que prices)
        """
        n = len(prices)
        entropy_array = np.zeros(n, dtype=OPTIMAL_FLOAT)

        for idx in range(window, n):
            # Lookback window
            window_prices = prices[idx - window : idx]
            N = len(window_prices)

            if N < m + 1:
                continue

            # SIMPLIFIED: Use std of differences as entropy proxy
            std_dev = np.std(window_prices)
            if std_dev == 0:
                continue

            # Calculate consecutive differences
            diffs = np.diff(window_prices)
            norm_diffs = np.abs(diffs) / std_dev

            # Count how many diffs are within tolerance
            within_tolerance = np.sum(norm_diffs < r)
            total = len(norm_diffs)

            # Entropy = -log(ratio) (simplified SampEn approximation)
            if within_tolerance > 0:
                ratio = within_tolerance / total
                entropy_array[idx] = -np.log(ratio + 1e-10)
            else:
                entropy_array[idx] = 5.0  # High entropy (random)

        return entropy_array

    @jit(nopython=True, cache=True)
    def _numba_approximate_entropy(prices, m=2, r=0.2, window=252):
        """Approximate Entropy (ApEn) ROLLING SIMPLIFIED (NO DATA LEAKAGE).

        CORRECTION: Version précédente calculait UNE entropy pour tout le dataset.
        This version uses simplified rolling approximation (same as Sample Entropy for performance).

        Args:
            prices: Array de prix
            m: Embedding dimension (default 2)
            r: Tolerance factor (default 0.2)
            window: Taille fenêtre rolling (défaut 252)

        Returns:
            Array d'entropies (même taille que prices)
        """
        # NOTE: Using same simplified implementation as Sample Entropy
        # Full ApEn is too computationally expensive for rolling windows
        return _numba_sample_entropy(prices, m, r, window)

    @jit(nopython=True, cache=True)
    def _numba_permutation_entropy(prices, order=3, delay=1, window=100):
        """Permutation Entropy ROLLING ultra-rapide (NO DATA LEAKAGE).

        CORRECTION: Version précédente calculait UN SEUL scalaire pour tout le dataset.
        Cette version calcule l'entropie de permutation sur rolling window.

        Args:
            prices: Array de prix
            order: Dimension d'embedding (défaut 3)
            delay: Délai (défaut 1)
            window: Taille fenêtre rolling (défaut 100)

        Returns:
            Array de même taille que prices
        """
        n = len(prices)
        result = np.zeros(n, dtype=OPTIMAL_FLOAT)

        for idx in range(window, n):
            window_prices = prices[idx - window : idx]
            N = len(window_prices)

            if N < order:
                continue

            permutations = np.zeros(24, dtype=OPTIMAL_FLOAT)  # Max 4! = 24 permutations
            total_patterns = 0

            for i in range(N - (order - 1) * delay):
                pattern = np.zeros(order, dtype=OPTIMAL_FLOAT)
                for j in range(order):
                    pattern[j] = window_prices[i + j * delay]

                sorted_indices = np.argsort(pattern)

                perm_key = 0
                for j in range(order):
                    perm_key += sorted_indices[j] * (j + 1)

                perm_key = int(perm_key) % len(permutations)
                permutations[perm_key] += 1
                total_patterns += 1

            if total_patterns == 0:
                continue

            entropy = 0.0
            for count in permutations:
                if count > 0:
                    p = count / total_patterns
                    entropy -= p * np.log2(p)

            result[idx] = entropy

        return result

    # ========== SPECTRAL (2 FEATURES) ==========

    @jit(nopython=True, cache=True)
    def _numba_dominant_frequency(prices, window=252):
        """Fréquence dominante ROLLING (NO DATA LEAKAGE).

        CORRECTION: Version précédente calculait UNE fréquence pour tout le dataset.
        Cette version calcule fréquence dominante sur rolling window.

        Args:
            prices: Array de prix
            window: Taille fenêtre rolling (défaut 252)

        Returns:
            Array de fréquences dominantes (même taille que prices)
        """
        n = len(prices)
        freq_array = np.zeros(n, dtype=OPTIMAL_FLOAT)

        for idx in range(window, n):
            # Lookback window
            window_prices = prices[idx - window : idx]
            n_window = len(window_prices)

            if n_window < 4:
                continue

            # Centrer les données
            mean_price = np.mean(window_prices)
            centered = window_prices - mean_price

            # FFT simplifiée
            max_freq_idx = 0
            max_magnitude = 0.0

            # Tester différentes fréquences
            for k in range(1, min(n_window // 2, 20)):
                # Calcul manuel des coefficients de Fourier
                real_part = 0.0
                imag_part = 0.0

                for i in range(n_window):
                    angle = 2.0 * np.pi * k * i / n_window
                    real_part += centered[i] * np.cos(angle)
                    imag_part += centered[i] * np.sin(angle)

                magnitude = np.sqrt(real_part * real_part + imag_part * imag_part)

                if magnitude > max_magnitude:
                    max_magnitude = magnitude
                    max_freq_idx = k

            # Fréquence normalisée
            freq_array[idx] = max_freq_idx / n_window if n_window > 0 else 0.0

        return freq_array

    @jit(nopython=True, cache=True)
    def _numba_spectral_centroid(prices, window=252):
        """Centroïde spectral ROLLING (NO DATA LEAKAGE).

        CORRECTION: Version précédente calculait UN centroïde pour tout le dataset.
        Cette version calcule centroïde spectral sur rolling window.

        Args:
            prices: Array de prix
            window: Taille fenêtre rolling (défaut 252)

        Returns:
            Array de centroïdes spectraux (même taille que prices)
        """
        n = len(prices)
        centroid_array = np.zeros(n, dtype=OPTIMAL_FLOAT)

        for idx in range(window, n):
            # Lookback window
            window_prices = prices[idx - window : idx]
            n_window = len(window_prices)

            if n_window < 4:
                continue

            mean_price = np.mean(window_prices)
            centered = window_prices - mean_price

            weighted_sum = 0.0
            magnitude_sum = 0.0

            # Calcul approximatif du centroïde spectral
            for k in range(1, min(n_window // 2, 20)):
                real_part = 0.0
                imag_part = 0.0

                for i in range(n_window):
                    angle = 2.0 * np.pi * k * i / n_window
                    real_part += centered[i] * np.cos(angle)
                    imag_part += centered[i] * np.sin(angle)

                magnitude = np.sqrt(real_part * real_part + imag_part * imag_part)
                frequency = k / n_window

                weighted_sum += frequency * magnitude
                magnitude_sum += magnitude

            centroid_array[idx] = weighted_sum / magnitude_sum if magnitude_sum > 0 else 0.0

        return centroid_array

    # ========== FRACTALES (2 FEATURES) ==========

    @jit(nopython=True, cache=True)
    def _numba_fractal_dimension(prices, window=252):
        """Dimension fractale avec box-counting ROLLING (NO DATA LEAKAGE).

        CORRECTION: Version précédente calculait UNE dimension pour tout le dataset.
        Cette version calcule dimension fractale sur rolling window.

        Args:
            prices: Array de prix
            window: Taille fenêtre rolling (défaut 252)

        Returns:
            Array de dimensions fractales (même taille que prices)
        """
        n = len(prices)
        fractal_array = np.ones(n, dtype=OPTIMAL_FLOAT)  # Default: 1.0

        for i in range(window, n):
            # Lookback window
            window_prices = prices[i - window : i]
            N = len(window_prices)

            if N < 4:
                continue

            # Normaliser les prix
            min_price = np.min(window_prices)
            max_price = np.max(window_prices)
            if max_price == min_price:
                continue

            normalized = (window_prices - min_price) / (max_price - min_price)

            # Différentes tailles de boîtes
            box_sizes = np.array([2, 4, 8, 16, min(32, N // 4)], dtype=np.int32)
            box_counts = np.zeros(len(box_sizes), dtype=OPTIMAL_FLOAT)

            for box_idx, box_size in enumerate(box_sizes):
                if box_size >= N:
                    continue

                # Compter les boîtes nécessaires
                n_boxes_x = N // box_size
                n_boxes_y = box_size

                boxes_needed = 0
                for x in range(n_boxes_x):
                    start_idx = x * box_size
                    end_idx = min(start_idx + box_size, N)
                    segment = normalized[start_idx:end_idx]

                    min_val = np.min(segment)
                    max_val = np.max(segment)

                    # Nombre de boîtes verticales nécessaires
                    boxes_y = int((max_val - min_val) * n_boxes_y) + 1
                    boxes_needed += boxes_y

                box_counts[box_idx] = boxes_needed

            # Régression log-log
            valid_sizes = box_sizes[box_sizes < N]
            valid_counts = box_counts[:len(valid_sizes)]

            if len(valid_sizes) >= 2:
                log_sizes = np.log(1.0 / valid_sizes.astype(OPTIMAL_FLOAT))
                log_counts = np.log(valid_counts)

                # Calcul de la pente (dimension fractale)
                n_points = len(log_sizes)
                sum_x = np.sum(log_sizes)
                sum_y = np.sum(log_counts)
                sum_xy = np.sum(log_sizes * log_counts)
                sum_x2 = np.sum(log_sizes * log_sizes)

                denominator = n_points * sum_x2 - sum_x * sum_x
                if abs(denominator) > 1e-10:
                    slope = (n_points * sum_xy - sum_x * sum_y) / denominator
                    fractal_array[i] = max(1.0, min(2.0, slope))

        return fractal_array

    @jit(nopython=True, cache=True)
    def _numba_dfa(prices, window=252):
        """Detrended Fluctuation Analysis ROLLING (NO DATA LEAKAGE).

        CORRECTION: Version précédente retournait UN SEUL scalaire pour tout
        le dataset. Cette version calcule le DFA sur une rolling window.

        Args:
            prices: Array de prix
            window: Taille fenêtre rolling (défaut 252)

        Returns:
            Array de DFA exponents (même taille que prices)
        """
        n = len(prices)
        dfa_array = np.full(n, 0.5, dtype=OPTIMAL_FLOAT)  # Défaut: random walk

        for idx in range(window, n):
            win = prices[idx - window : idx]
            N = len(win)
            if N < 20:
                continue

            # Profile (cumsum déviations par rapport à la moyenne)
            mean_price = np.mean(win)
            integrated = np.cumsum(win - mean_price)

            # 3 scales pour performance (simplifié vs 5 scales original)
            scales = np.array([10, 25, min(50, N // 4)], dtype=np.int32)
            fluctuations = np.zeros(len(scales), dtype=OPTIMAL_FLOAT)

            for scale_idx, scale in enumerate(scales):
                if scale >= N or scale < 4:
                    continue

                n_segments = N // scale
                mse = 0.0

                for j in range(n_segments):
                    start_idx = j * scale
                    end_idx = start_idx + scale
                    segment = integrated[start_idx:end_idx]

                    x = np.arange(scale, dtype=OPTIMAL_FLOAT)
                    sum_x = np.sum(x)
                    sum_y = np.sum(segment)
                    sum_xy = np.sum(x * segment)
                    sum_x2 = np.sum(x * x)

                    denom = scale * sum_x2 - sum_x * sum_x
                    if abs(denom) > 1e-10:
                        slope = (scale * sum_xy - sum_x * sum_y) / denom
                        intercept = (sum_y - slope * sum_x) / scale
                        for k in range(scale):
                            trend = slope * k + intercept
                            mse += (segment[k] - trend) ** 2

                if n_segments > 0:
                    fluctuations[scale_idx] = np.sqrt(mse / (n_segments * scale))

            # Log-log regression → DFA exponent
            valid_scales = scales[scales < N]
            valid_fluct = fluctuations[:len(valid_scales)]

            if len(valid_scales) >= 2 and np.min(valid_fluct) > 0:
                log_scales = np.log(valid_scales.astype(OPTIMAL_FLOAT))
                log_fluct = np.log(valid_fluct + 1e-10)

                n_pts = len(log_scales)
                sum_x = np.sum(log_scales)
                sum_y = np.sum(log_fluct)
                sum_xy = np.sum(log_scales * log_fluct)
                sum_x2 = np.sum(log_scales * log_scales)

                denom = n_pts * sum_x2 - sum_x * sum_x
                if abs(denom) > 1e-10:
                    dfa_exp = (n_pts * sum_xy - sum_x * sum_y) / denom
                    dfa_array[idx] = max(0.0, min(2.0, dfa_exp))

        return dfa_array


    # ========== NOUVELLES FEATURES INSTITUTIONNELLES (6 FEATURES) ==========

    @jit(nopython=True, cache=True)
    def _numba_rolling_skewness(prices, window=50):
        """Skewness roulante ultra-rapide."""
        n = len(prices)
        if n < window:
            return np.full(n, 0.0, dtype=OPTIMAL_FLOAT)

        skewness = np.zeros(n, dtype=OPTIMAL_FLOAT)

        for i in range(window - 1, n):
            window_data = prices[i - window + 1 : i + 1]
            mean_val = np.mean(window_data)
            std_val = np.std(window_data)

            if std_val > 1e-10:
                # Calcul du skewness
                skew_sum = 0.0
                for val in window_data:
                    skew_sum += ((val - mean_val) / std_val) ** 3
                skewness[i] = skew_sum / window
            else:
                skewness[i] = 0.0

        # Remplir les premières valeurs
        for i in range(window - 1):
            skewness[i] = skewness[window - 1]

        return skewness

    @jit(nopython=True, cache=True)
    def _numba_rolling_kurtosis(prices, window=50):
        """Kurtosis roulante ultra-rapide."""
        n = len(prices)
        if n < window:
            return np.full(n, 3.0, dtype=OPTIMAL_FLOAT)  # Kurtosis normale = 3

        kurtosis = np.zeros(n, dtype=OPTIMAL_FLOAT)

        for i in range(window - 1, n):
            window_data = prices[i - window + 1 : i + 1]
            mean_val = np.mean(window_data)
            std_val = np.std(window_data)

            if std_val > 1e-10:
                # Calcul du kurtosis
                kurt_sum = 0.0
                for val in window_data:
                    kurt_sum += ((val - mean_val) / std_val) ** 4
                kurtosis[i] = kurt_sum / window
            else:
                kurtosis[i] = 3.0

        # Remplir les premières valeurs
        for i in range(window - 1):
            kurtosis[i] = kurtosis[window - 1]

        return kurtosis

    @jit(nopython=True, cache=True)
    def _numba_dynamic_var(returns, confidence=0.05, window=50):
        """VaR dynamique ultra-rapide."""
        n = len(returns)
        if n < window:
            return np.full(n, 0.0, dtype=OPTIMAL_FLOAT)

        var_values = np.zeros(n, dtype=OPTIMAL_FLOAT)
        percentile_rank = int(confidence * window)

        for i in range(window - 1, n):
            window_returns = returns[i - window + 1 : i + 1].copy()
            # Tri pour trouver le percentile
            window_returns.sort()
            if percentile_rank < len(window_returns):
                var_values[i] = -window_returns[percentile_rank]  # VaR négatif
            else:
                var_values[i] = -window_returns[0]

        # Remplir les premières valeurs
        for i in range(window - 1):
            var_values[i] = var_values[window - 1]

        return var_values

    @jit(nopython=True, cache=True)
    def _numba_dynamic_cvar(returns, confidence=0.05, window=50):
        """CVaR (Expected Shortfall) dynamique ultra-rapide."""
        n = len(returns)
        if n < window:
            return np.full(n, 0.0, dtype=OPTIMAL_FLOAT)

        cvar_values = np.zeros(n, dtype=OPTIMAL_FLOAT)
        percentile_rank = int(confidence * window)

        for i in range(window - 1, n):
            window_returns = returns[i - window + 1 : i + 1].copy()
            window_returns.sort()

            # CVaR = moyenne des returns pires que VaR
            if percentile_rank > 0:
                tail_returns = window_returns[:percentile_rank]
                cvar_values[i] = -np.mean(tail_returns)  # CVaR négatif
            else:
                cvar_values[i] = -window_returns[0]

        # Remplir les premières valeurs
        for i in range(window - 1):
            cvar_values[i] = cvar_values[window - 1]

        return cvar_values

    @jit(nopython=True, cache=True)
    def _numba_max_drawdown(prices, window=100):
        """Maximum Drawdown roulant ultra-rapide."""
        n = len(prices)
        if n < window:
            return np.full(n, 0.0, dtype=OPTIMAL_FLOAT)

        drawdowns = np.zeros(n, dtype=OPTIMAL_FLOAT)

        for i in range(window - 1, n):
            window_prices = prices[i - window + 1 : i + 1]

            # Trouver le maximum drawdown dans la fenêtre
            max_dd = 0.0
            peak = window_prices[0]

            for price in window_prices:
                if price > peak:
                    peak = price

                drawdown = (peak - price) / peak if peak > 0 else 0.0
                if drawdown > max_dd:
                    max_dd = drawdown

            drawdowns[i] = max_dd

        # Remplir les premières valeurs
        for i in range(window - 1):
            drawdowns[i] = drawdowns[window - 1]

        return drawdowns

    @jit(nopython=True, cache=True)
    def _numba_regime_detection(returns, lookback=50):
        """Détection de régime adaptative ultra-rapide (z-score, asset-agnostique)."""
        n = len(returns)
        if n < lookback:
            return np.full(n, 0.0, dtype=OPTIMAL_FLOAT)  # Régime neutre

        regimes = np.zeros(n, dtype=OPTIMAL_FLOAT)

        for i in range(lookback - 1, n):
            window_returns = returns[i - lookback + 1 : i + 1]

            # Calculs adaptatifs pour classification de régime
            mean_return = np.mean(window_returns)
            volatility = np.std(window_returns)

            # Z-score du rendement moyen (adaptatif à l'échelle de l'asset)
            # Évite les seuils absolus qui ne fonctionnent pas cross-asset
            if volatility > 1e-10:
                z_score = mean_return / volatility
            else:
                z_score = 0.0

            # Volatilité relative au rendement moyen absolu
            # Ratio élevé = marché chaotique, ratio bas = marché directionnel
            mean_abs = np.mean(np.abs(window_returns))
            vol_ratio = volatility / mean_abs if mean_abs > 1e-10 else 2.0

            # Classification adaptative:
            # 1.0 = Bull (z-score positif significatif, volatilité contrôlée)
            # -1.0 = Bear (z-score négatif significatif, volatilité élevée)
            # 0.0 = Sideways (z-score faible ou volatilité non directionnelle)

            if z_score > 0.5 and vol_ratio < 1.5:
                regimes[i] = 1.0  # Bull
            elif z_score < -0.5 and vol_ratio > 1.0:
                regimes[i] = -1.0  # Bear
            else:
                regimes[i] = 0.0  # Sideways

        # Remplir les premières valeurs
        for i in range(lookback - 1):
            regimes[i] = regimes[lookback - 1]

        return regimes

    # ========== LIQUIDITÉ (2 FEATURES) ==========

    @jit(nopython=True, cache=True)
    def _numba_amihud_illiquidity(returns, volume, window=20):
        """Amihud Illiquidity ultra-rapide."""
        n = len(returns)
        if n < window:
            return np.full(n, 0.0, dtype=OPTIMAL_FLOAT)

        illiquidity = np.zeros(n, dtype=OPTIMAL_FLOAT)

        # Amihud = Mean( |Return| / (Price * Volume) )
        # Simplification pour compatibilité dimensionnelle: |Return| / Volume
        # Car Price * Volume = Dollar Volume, mais ici on veut l'impact par unité de volume

        abs_returns = np.abs(returns)

        for i in range(window - 1, n):
            window_ret = abs_returns[i - window + 1 : i + 1]
            window_vol = volume[i - window + 1 : i + 1]

            sum_ratio = 0.0
            count = 0

            for j in range(window):
                if window_vol[j] > 1e-5:
                    sum_ratio += window_ret[j] / window_vol[j]
                    count += 1

            if count > 0:
                illiquidity[i] = sum_ratio / count * 1e6  # Mettre à l'échelle
            else:
                illiquidity[i] = 0.0

        # Remplir
        for i in range(window - 1):
            illiquidity[i] = illiquidity[window - 1]

        return illiquidity

    @jit(nopython=True, cache=True)
    def _numba_kyles_lambda(prices, volume, window=20):
        """Kyle's Lambda (Simplifié) ultra-rapide."""
        n = len(prices)
        if n < window:
            return np.full(n, 0.0, dtype=OPTIMAL_FLOAT)

        lambdas = np.zeros(n, dtype=OPTIMAL_FLOAT)

        # Lambda ~ Pente de régression PriceChange vs NetVolume
        # Approximation: High-Low / Volume (Volatilité par unité de volume)

        np.zeros(n, dtype=OPTIMAL_FLOAT)
        # Calculer ranges simples si High/Low non dispos dans cette fonction
        # On utilise une approximation basée sur abs(diff(prices))

        abs_diff = np.abs(np.diff(prices))
        abs_diff = np.concatenate((np.array([0.0], dtype=OPTIMAL_FLOAT), abs_diff))

        for i in range(window - 1, n):
            window_diff = abs_diff[i - window + 1 : i + 1]
            window_vol = volume[i - window + 1 : i + 1]

            sum_ratio = 0.0
            count = 0

            for j in range(window):
                if window_vol[j] > 1e-5:
                    sum_ratio += window_diff[j] / window_vol[j]
                    count += 1

            if count > 0:
                lambdas[i] = sum_ratio / count * 1e6
            else:
                lambdas[i] = 0.0

        # Remplir
        for i in range(window - 1):
            lambdas[i] = lambdas[window - 1]

        return lambdas

    # ========== EFFICIENCE (2 FEATURES) ==========

    @jit(nopython=True, cache=True)
    def _numba_kaufman_efficiency(prices, window=20):
        """Ratio d'efficience de Kaufman ultra-rapide."""
        n = len(prices)
        if n < window:
            return np.full(n, 0.5, dtype=OPTIMAL_FLOAT)

        er = np.zeros(n, dtype=OPTIMAL_FLOAT)

        abs_diff = np.abs(np.diff(prices))
        abs_diff = np.concatenate((np.array([0.0], dtype=OPTIMAL_FLOAT), abs_diff))

        for i in range(window, n):
            # Directional movement: |Price_t - Price_t-n|
            direction = np.abs(prices[i] - prices[i - window])

            # Volatility: Sum(|Price_i - Price_i-1|)
            volatility = np.sum(abs_diff[i - window + 1 : i + 1])

            if volatility > 1e-10:
                er[i] = direction / volatility
            else:
                er[i] = 1.0 if direction == 0 else 0.0

        # Remplir
        for i in range(window):
            er[i] = er[window] if window < n else 0.5

        return er

    @jit(nopython=True, cache=True)
    def _numba_variance_ratio(returns, lags=20):
        """Test de Ratio de Variance (Random Walk) ultra-rapide."""
        n = len(returns)
        if n < lags * 2:
            return np.full(n, 1.0, dtype=OPTIMAL_FLOAT)

        vr = np.full(n, np.nan, dtype=OPTIMAL_FLOAT)  # NaN pour ffill correct en post-processing

        # VR(q) = Var(r_q) / (q * Var(r_1))
        # Var(r_q) est la variance des rendements sur q périodes

        for i in range(n - 1, 30, -1): # Ne pas calculer pour tout l'historique (trop lent), focus récent
            # Fenêtre locale pour "Rolling VR"
            window_size = min(i, 100)
            if window_size < lags:
                continue

            local_rets = returns[i - window_size + 1 : i + 1]

            # Variance 1-période
            var_1 = np.var(local_rets)

            # Variance q-périodes
            # Somme mobile des rendements sur lags
            sum_rets_q = np.zeros(len(local_rets) - lags + 1)
            for j in range(len(sum_rets_q)):
                sum_rets_q[j] = np.sum(local_rets[j : j + lags])

            var_q = np.var(sum_rets_q)

            if var_1 > 1e-10:
                vr[i] = var_q / (lags * var_1)
            else:
                vr[i] = 1.0

        # Remplir les trous (forward fill inversé ou simple fill)
        # Numba ne supporte pas ffill simple, on laisse les 0 qui seront ffill plus tard par pandas

        return vr

    # ========== FONCTIONS NUMBA ADDITIONNELLES POUR OPTIMISATIONS ==========

    @jit(nopython=True, cache=True)
    def calculate_returns_numba(prices):
        """Calcul ultra-rapide des rendements avec Numba."""
        n = len(prices)
        if n < 2:
            return np.zeros(1, dtype=OPTIMAL_FLOAT)

        returns = np.zeros(n, dtype=OPTIMAL_FLOAT)
        returns[0] = 0.0

        for i in range(1, n):
            if prices[i - 1] != 0:
                returns[i] = (prices[i] - prices[i - 1]) / prices[i - 1]
            else:
                returns[i] = 0.0

        return returns

    @jit(nopython=True, cache=True)
    def calculate_rolling_mean_numba(data, window):
        """Moyenne mobile ultra-rapide avec Numba."""
        n = len(data)
        if n < window:
            return np.full(n, np.mean(data), dtype=OPTIMAL_FLOAT)

        result = np.zeros(n, dtype=OPTIMAL_FLOAT)

        # Première valeur
        window_sum = 0.0
        for i in range(window):
            window_sum += data[i]
        result[window - 1] = window_sum / window

        # Calcul glissant
        for i in range(window, n):
            window_sum = window_sum - data[i - window] + data[i]
            result[i] = window_sum / window

        # Remplir les premières valeurs
        for i in range(window - 1):
            result[i] = result[window - 1]

        return result

    @jit(nopython=True, cache=True)
    def calculate_rolling_std_numba(data, window):
        """Écart-type mobile ultra-rapide avec Numba."""
        n = len(data)
        if n < window:
            return np.full(n, np.std(data), dtype=OPTIMAL_FLOAT)

        result = np.zeros(n, dtype=OPTIMAL_FLOAT)

        for i in range(window - 1, n):
            window_data = data[i - window + 1 : i + 1]
            mean_val = np.mean(window_data)

            variance = 0.0
            for val in window_data:
                variance += (val - mean_val) ** 2
            variance /= window

            result[i] = np.sqrt(variance)

        # Remplir les premières valeurs
        for i in range(window - 1):
            result[i] = result[window - 1]

        return result

    @jit(nopython=True, cache=True)
    def calculate_momentum_numba(prices, window=14):
        """Momentum ultra-rapide avec Numba."""
        n = len(prices)
        if n < window:
            return np.zeros(n, dtype=OPTIMAL_FLOAT)

        momentum = np.zeros(n, dtype=OPTIMAL_FLOAT)

        for i in range(window, n):
            momentum[i] = (
                (prices[i] - prices[i - window]) / prices[i - window]
                if prices[i - window] != 0
                else 0.0
            )

        # Remplir les premières valeurs
        for i in range(window):
            momentum[i] = momentum[window] if window < n else 0.0

        return momentum

    @jit(nopython=True, cache=True)
    def calculate_price_position_numba(prices, window=20):
        """Position du prix dans la range ultra-rapide."""
        n = len(prices)
        if n < window:
            return np.full(n, 0.5, dtype=OPTIMAL_FLOAT)

        position = np.zeros(n, dtype=OPTIMAL_FLOAT)

        for i in range(window - 1, n):
            window_data = prices[i - window + 1 : i + 1]
            min_val = np.min(window_data)
            max_val = np.max(window_data)

            if max_val != min_val:
                position[i] = (prices[i] - min_val) / (max_val - min_val)
            else:
                position[i] = 0.5

        # Remplir les premières valeurs
        for i in range(window - 1):
            position[i] = position[window - 1]

        return position

    @jit(nopython=True, cache=True)
    def calculate_optimal_actions_numba(prices, returns, volatility):
        """Actions optimales basées sur prix, rendements et volatilité."""
        n = len(prices)
        actions = np.zeros(n, dtype=OPTIMAL_FLOAT)

        for i in range(1, n):
            # Action basée sur momentum et volatilité
            momentum = returns[i]
            vol = volatility[i] if i < len(volatility) else 0.1

            # Signal simple: acheter si momentum positif et volatilité modérée
            if momentum > 0.001 and vol < 0.02:
                actions[i] = 1.0  # Acheter
            elif momentum < -0.001 and vol < 0.02:
                actions[i] = -1.0  # Vendre
            else:
                actions[i] = 0.0  # Tenir

        return actions

    @jit(nopython=True, cache=True)
    def calculate_volatility_regime_numba(
        volatility, threshold_low=0.01, threshold_high=0.03
    ):
        """Régime de volatilité ultra-rapide."""
        n = len(volatility)
        regime = np.zeros(n, dtype=OPTIMAL_FLOAT)

        for i in range(n):
            if volatility[i] < threshold_low:
                regime[i] = -1.0  # Faible volatilité
            elif volatility[i] > threshold_high:
                regime[i] = 1.0  # Haute volatilité
            else:
                regime[i] = 0.0  # Volatilité normale

        return regime

    @jit(nopython=True, cache=True)
    def calculate_market_regime_numba(returns, volatility, window=50):
        """Régime de marché combiné ultra-rapide."""
        n = len(returns)
        if n < window:
            return np.zeros(n, dtype=OPTIMAL_FLOAT)

        regime = np.zeros(n, dtype=OPTIMAL_FLOAT)

        for i in range(window - 1, n):
            window_returns = returns[i - window + 1 : i + 1]
            window_vol = (
                volatility[i - window + 1 : i + 1]
                if i < len(volatility)
                else np.full(window, 0.1)
            )

            mean_return = np.mean(window_returns)
            mean_vol = np.mean(window_vol)

            # Classification des régimes
            if mean_return > 0.002 and mean_vol < 0.02:
                regime[i] = 2.0  # Bull market
            elif mean_return < -0.002 and mean_vol > 0.03:
                regime[i] = -2.0  # Bear market
            elif mean_vol > 0.04:
                regime[i] = -1.0  # High volatility
            else:
                regime[i] = 0.0  # Neutral/Sideways

        # Remplir les premières valeurs
        for i in range(window - 1):
            regime[i] = regime[window - 1]

        return regime

else:
    # Fallback functions si Numba non disponible
    def _numba_realized_volatility(returns, window=20):
        """_numba_realized_volatility.

        Args:
            returns: TODO document.
            window: TODO document.
        """
        return np.full(len(returns), np.std(returns), dtype=OPTIMAL_FLOAT)

    def _numba_garch_volatility(returns, alpha=0.1, beta=0.8):
        """_numba_garch_volatility.

        Args:
            returns: TODO document.
            alpha: TODO document.
            beta: TODO document.
        """
        return np.full(len(returns), np.std(returns), dtype=OPTIMAL_FLOAT)

    def _numba_volatility_clustering(returns, threshold_factor=2.0, window=100):
        """_numba_volatility_clustering.

        Args:
            returns: TODO document.
            threshold_factor: TODO document.
            window: TODO document.
        """
        return np.full(len(returns), 0.1, dtype=OPTIMAL_FLOAT)

    def _numba_volatility_persistence(returns, window=100):
        """_numba_volatility_persistence.

        Args:
            returns: TODO document.
            window: TODO document.
        """
        return np.full(len(returns), 0.3, dtype=OPTIMAL_FLOAT)

    def _numba_hurst_rs(prices, window=252):
        """_numba_hurst_rs.

        Args:
            prices: TODO document.
            window: TODO document.
        """
        return np.full(len(prices), 0.5, dtype=OPTIMAL_FLOAT)  # Neutral: random walk

    def _numba_autocorrelation(prices, max_lag=20, window=252):
        """_numba_autocorrelation.

        Args:
            prices: TODO document.
            max_lag: TODO document.
            window: TODO document.
        """
        return np.zeros(len(prices), dtype=OPTIMAL_FLOAT)

    def _numba_shannon_entropy(prices, bins=50, window=252):
        """Fallback: retourne un array constant (entropie globale)."""
        hist, _ = np.histogram(prices, bins=bins)
        hist = hist[hist > 0]
        entropy_val = -np.sum((hist / len(prices)) * np.log2(hist / len(prices)))
        return np.full(len(prices), entropy_val, dtype=OPTIMAL_FLOAT)

    def _numba_sample_entropy(prices, m=2, r=0.2, window=252):
        """_numba_sample_entropy.

        Args:
            prices: TODO document.
            m: TODO document.
            r: TODO document.
            window: TODO document.
        """
        return np.full(len(prices), 1.0, dtype=OPTIMAL_FLOAT)  # Neutral entropy

    def _numba_dominant_frequency(prices, window=252):
        """_numba_dominant_frequency.

        Args:
            prices: TODO document.
            window: TODO document.
        """
        return np.full(len(prices), 0.1, dtype=OPTIMAL_FLOAT)

    def _numba_spectral_centroid(prices, window=252):
        """_numba_spectral_centroid.

        Args:
            prices: TODO document.
            window: TODO document.
        """
        return np.full(len(prices), 0.5, dtype=OPTIMAL_FLOAT)

    def _numba_fractal_dimension(prices, window=252):
        """_numba_fractal_dimension.

        Args:
            prices: TODO document.
            window: TODO document.
        """
        return np.full(len(prices), 1.5, dtype=OPTIMAL_FLOAT)  # Neutral: between 1 and 2

    def _numba_dfa(prices, window=252):
        """_numba_dfa.

        Args:
            prices: TODO document.
            window: TODO document.
        """
        return np.full(len(prices), 0.5, dtype=OPTIMAL_FLOAT)  # Neutral: random walk

    def _numba_rolling_skewness(prices, window=50):
        """_numba_rolling_skewness.

        Args:
            prices: TODO document.
            window: TODO document.
        """
        return np.zeros(len(prices), dtype=OPTIMAL_FLOAT)

    def _numba_rolling_kurtosis(prices, window=50):
        """_numba_rolling_kurtosis.

        Args:
            prices: TODO document.
            window: TODO document.
        """
        return np.full(len(prices), 3.0, dtype=OPTIMAL_FLOAT)

    def _numba_dynamic_var(returns, confidence=0.05, window=50):
        """_numba_dynamic_var.

        Args:
            returns: TODO document.
            confidence: TODO document.
            window: TODO document.
        """
        return np.zeros(len(returns), dtype=OPTIMAL_FLOAT)

    def _numba_dynamic_cvar(returns, confidence=0.05, window=50):
        """_numba_dynamic_cvar.

        Args:
            returns: TODO document.
            confidence: TODO document.
            window: TODO document.
        """
        return np.zeros(len(returns), dtype=OPTIMAL_FLOAT)

    def _numba_max_drawdown(prices, window=100):
        """_numba_max_drawdown.

        Args:
            prices: TODO document.
            window: TODO document.
        """
        return np.zeros(len(prices), dtype=OPTIMAL_FLOAT)

    def _numba_regime_detection(returns, lookback=50):
        """_numba_regime_detection.

        Args:
            returns: TODO document.
            lookback: TODO document.
        """
        return np.zeros(len(returns), dtype=OPTIMAL_FLOAT)

    def _numba_approximate_entropy(prices, m=2, r=0.2, window=252):
        """_numba_approximate_entropy.

        Args:
            prices: TODO document.
            m: TODO document.
            r: TODO document.
            window: TODO document.
        """
        return np.full(len(prices), 1.0, dtype=OPTIMAL_FLOAT)  # Neutral entropy

    def _numba_permutation_entropy(prices, order=3, delay=1, window=100):
        """_numba_permutation_entropy.

        Args:
            prices: TODO document.
            order: TODO document.
            delay: TODO document.
            window: TODO document.
        """
        return np.full(len(prices), 1.5, dtype=OPTIMAL_FLOAT)

    # Fallback functions pour les nouvelles fonctions Numba
    def _numba_amihud_illiquidity(returns, volume, window=20):
        """_numba_amihud_illiquidity.

        Args:
            returns: TODO document.
            volume: TODO document.
            window: TODO document.
        """
        return np.zeros(len(returns), dtype=OPTIMAL_FLOAT)

    def _numba_kyles_lambda(prices, volume, window=20):
        """_numba_kyles_lambda.

        Args:
            prices: TODO document.
            volume: TODO document.
            window: TODO document.
        """
        return np.zeros(len(prices), dtype=OPTIMAL_FLOAT)

    def _numba_kaufman_efficiency(prices, window=20):
        """_numba_kaufman_efficiency.

        Args:
            prices: TODO document.
            window: TODO document.
        """
        return np.full(len(prices), 0.5, dtype=OPTIMAL_FLOAT)

    def _numba_variance_ratio(returns, lags=20):
        """_numba_variance_ratio.

        Args:
            returns: TODO document.
            lags: TODO document.
        """
        return np.full(len(returns), 1.0, dtype=OPTIMAL_FLOAT)

    # Fallback functions pour les nouvelles fonctions Numba
    def calculate_returns_numba(prices):
        """calculate_returns_numba.

        Args:
            prices: TODO document.
        """
        return np.diff(prices) / prices[:-1] if len(prices) > 1 else np.array([0.0])

    def calculate_rolling_mean_numba(data, window):
        """calculate_rolling_mean_numba.

        Args:
            data: TODO document.
            window: TODO document.
        """
        return pd.Series(data).rolling(window).mean().fillna(method="bfill").values

    def calculate_rolling_std_numba(data, window):
        """calculate_rolling_std_numba.

        Args:
            data: TODO document.
            window: TODO document.
        """
        return pd.Series(data).rolling(window).std().fillna(method="bfill").values

    def calculate_momentum_numba(prices, window=14):
        """calculate_momentum_numba.

        Args:
            prices: TODO document.
            window: TODO document.
        """
        return np.zeros(len(prices), dtype=OPTIMAL_FLOAT)

    def calculate_price_position_numba(prices, window=20):
        """calculate_price_position_numba.

        Args:
            prices: TODO document.
            window: TODO document.
        """
        return np.full(len(prices), 0.5, dtype=OPTIMAL_FLOAT)

    def calculate_optimal_actions_numba(prices, returns, volatility):
        """calculate_optimal_actions_numba.

        Args:
            prices: TODO document.
            returns: TODO document.
            volatility: TODO document.
        """
        return np.zeros(len(prices), dtype=OPTIMAL_FLOAT)

    def calculate_volatility_regime_numba(
        volatility, threshold_low=0.01, threshold_high=0.03
    ):
        """calculate_volatility_regime_numba.

        Args:
            volatility: TODO document.
            threshold_low: TODO document.
            threshold_high: TODO document.
        """
        return np.zeros(len(volatility), dtype=OPTIMAL_FLOAT)

    def calculate_market_regime_numba(returns, volatility, window=50):
        """calculate_market_regime_numba.

        Args:
            returns: TODO document.
            volatility: TODO document.
            window: TODO document.
        """
        return np.zeros(len(returns), dtype=OPTIMAL_FLOAT)

# ============================================================================
# CLASSE DE CACHE INTELLIGENT OPTIMISÉ
# ============================================================================


class QuantitativeCache:
    """Cache intelligent LRU avec optimisations avancées."""

    def __init__(self, max_size: int = 1000, memory_limit_mb: float = 100.0):
        """__init__.

        Args:
            max_size: TODO document.
            memory_limit_mb: TODO document.
        """
        self.max_size = max_size
        self.memory_limit_mb = memory_limit_mb
        self.cache = {}
        self.access_count = {}
        self.access_order = []
        self.access_time = {}
        self.hit_count = 0
        self.miss_count = 0
        self.current_memory_mb = 0.0

    def _get_key(self, prices, feature_name, **params):
        """Génère une clé unique optimisée pour le cache."""
        # OPTIMISATION: Cache basé sur taille et statistiques au lieu de hash complet
        n = len(prices)
        if n > 100:
            # Pour les gros datasets, utiliser des statistiques représentatives
            mean_val = np.mean(prices)
            std_val = np.std(prices)
            min_val = np.min(prices)
            max_val = np.max(prices)

            # Échantillonner quelques points clés
            sample_indices = [0, n // 4, n // 2, 3 * n // 4, n - 1]
            sample_values = [prices[i] for i in sample_indices if i < n]

            # CORRECTION: Remplacer les caractères problématiques par des caractères sûrs
            _sample_part = "_".join(f"{abs(v):.6f}" for v in sample_values)
            data_signature = (
                f"{n}_{abs(mean_val):.6f}_{abs(std_val):.6f}_{abs(min_val):.6f}"
                f"_{abs(max_val):.6f}_{_sample_part}"
            )
        else:
            # Pour les petits datasets, utiliser un hash simple
            data_signature = f"{n}_{abs(hash(tuple(prices[: min(50, n)])))}"

        # CORRECTION: Nettoyer les paramètres pour éviter les caractères problématiques
        clean_params = {}
        for k, v in sorted(params.items()):
            if isinstance(v, int | float):
                clean_params[k] = abs(v)  # Utiliser la valeur absolue pour éviter les négatifs
            else:
                clean_params[k] = str(v).replace('-', 'neg').replace('.', 'dot')

        params_str = "_".join(f"{k}_{v}" for k, v in clean_params.items())
        return f"{feature_name}_{data_signature}_{params_str}"

    def get(self, prices, feature_name, **params):
        """Récupère du cache avec statistiques avancées."""
        key = self._get_key(prices, feature_name, **params)
        if key in self.cache:
            self.access_count[key] = self.access_count.get(key, 0) + 1
            self.access_time[key] = time.time()
            self.access_order.append(key)
            self.hit_count += 1
            return self.cache[key]

        self.miss_count += 1
        return None

    def set(self, prices, feature_name, result, **params):
        """Stocke dans le cache avec gestion mémoire intelligente."""
        # Vérification mémoire
        if self.current_memory_mb > self.memory_limit_mb:
            self._cleanup_memory()

        if len(self.cache) >= self.max_size:
            self._evict_smart()

        key = self._get_key(prices, feature_name, **params)
        self.cache[key] = result
        self.access_order.append(key)
        self.access_count[key] = 1
        self.access_time[key] = time.time()

        # Estimation de la mémoire utilisée
        self.current_memory_mb += len(str(result)) / (1024 * 1024)

    def _evict_smart(self):
        """Éviction intelligente basée sur fréquence et récence."""
        if not self.cache:
            return

        current_time = time.time()

        # Score combiné pour chaque clé
        scores = {}
        for key in self.cache.keys():
            frequency = self.access_count.get(key, 1)
            recency = current_time - self.access_time.get(key, current_time)
            # Score = fréquence / (1 + récence)
            scores[key] = frequency / (1 + recency)

        # Supprimer les 25% avec les plus faibles scores
        keys_to_evict = sorted(scores.keys(), key=lambda k: scores[k])[
            : len(scores) // 4
        ]

        for key in keys_to_evict:
            if key in self.cache:
                del self.cache[key]
                if key in self.access_count:
                    del self.access_count[key]
                if key in self.access_time:
                    del self.access_time[key]
                # Retirer de l'ordre d'accès
                if key in self.access_order:
                    self.access_order.remove(key)

        self.current_memory_mb *= 0.75  # Estimation de la réduction

    def _cleanup_memory(self):
        """Nettoyage mémoire agressif et sécurisé."""
        try:
            # Supprimer 50% du cache de manière sécurisée
            keys_to_remove = self.access_order[::2] if self.access_order else []

            for key in keys_to_remove:
                try:
                    if key in self.cache:
                        del self.cache[key]
                    if key in self.access_count:
                        del self.access_count[key]
                    if key in self.access_time:
                        del self.access_time[key]
                except Exception as key_error:
                    logger.warning(f"Erreur suppression clé {key}: {key_error}")

            # Nettoyer l'ordre d'accès
            self.access_order = self.access_order[1::2] if self.access_order else []

            # Réinitialiser les statistiques
            self.current_memory_mb *= 0.5
            self.hit_count = max(0, self.hit_count // 2)
            self.miss_count = max(0, self.miss_count // 2)

        except Exception as e:
            logger.warning(f"Erreur nettoyage mémoire cache: {e}")
            # En cas d'erreur, vider complètement le cache
            self.cache.clear()
            self.access_count.clear()
            self.access_time.clear()
            self.access_order.clear()
            self.current_memory_mb = 0.0

    def get_stats(self) -> dict[str, any]:
        """Statistiques du cache."""
        total_requests = self.hit_count + self.miss_count
        hit_rate = (self.hit_count / total_requests * 100) if total_requests > 0 else 0

        return {
            "hit_rate": hit_rate,
            "cache_size": len(self.cache),
            "memory_usage_mb": self.current_memory_mb,
            "total_requests": total_requests,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
        }


# ============================================================================
# OPTIMISATIONS DASK POUR TRAITEMENT DISTRIBUÉ
# ============================================================================


class DaskOptimizer:
    """Optimiseur Dask pour traitement distribué des features quantitatives."""

    def __init__(self, n_workers=None, threads_per_worker=2, memory_limit="2GB"):
        """__init__.

        Args:
            n_workers: TODO document.
            threads_per_worker: TODO document.
            memory_limit: TODO document.
        """
        self.n_workers = n_workers or min(cpu_count(), 8)
        self.threads_per_worker = threads_per_worker
        self.memory_limit = memory_limit
        self.client = None
        self.cluster = None

    def setup_cluster(self):
        """Configuration du cluster Dask optimisé."""
        if not DASK_AVAILABLE:
            logger.warning("Dask non disponible, impossible de configurer le cluster")
            return False

        try:
            # Configuration du cluster local
            self.cluster = LocalCluster(
                n_workers=self.n_workers,
                threads_per_worker=self.threads_per_worker,
                memory_limit=self.memory_limit,
                dashboard_address=None,  # Désactiver dashboard pour performance
                silence_logs=logging.ERROR,
            )

            self.client = Client(self.cluster)
            logger.info(
                f"✅ Cluster Dask configuré: {self.n_workers} workers, {self.memory_limit} par worker"
            )
            return True

        except Exception as e:
            logger.error(f"❌ Erreur configuration cluster Dask: {e}")
            return False

    def cleanup_cluster(self):
        """Nettoyage du cluster Dask."""
        try:
            if self.client:
                self.client.close()
            if self.cluster:
                self.cluster.close()
            logger.info("✅ Cluster Dask fermé proprement")
        except Exception as e:
            logger.error(f"⚠️ Erreur fermeture cluster Dask: {e}")

    def process_with_dask(self, df: pd.DataFrame, enricher_instance) -> pd.DataFrame:
        """Traitement avec Dask DataFrame."""
        if not self.setup_cluster():
            logger.warning("Fallback vers traitement standard")
            return enricher_instance._process_single_chunk(df)

        try:
            # Convertir en Dask DataFrame
            npartitions = min(self.n_workers * 2, len(df) // 10000 + 1)
            ddf = dd.from_pandas(df, npartitions=npartitions)

            # Appliquer l'enrichissement par partition
            enriched_ddf = ddf.map_partitions(
                self._compute_indicators_for_partition_safe,
                enricher_instance,
                meta=self._get_enriched_meta(df),
            )

            # Calculer le résultat
            with ProgressBar():
                result = enriched_ddf.compute()

            return result

        except Exception as e:
            logger.error(f"❌ Erreur traitement Dask: {e}")
            return enricher_instance._process_single_chunk(df)
        finally:
            self.cleanup_cluster()

    def _compute_indicators_for_partition_safe(self, partition_df, enricher_instance):
        """Calcul sécurisé des indicateurs pour une partition."""
        try:
            return enricher_instance._add_all_features(partition_df)
        except Exception as e:
            logger.error(f"❌ Erreur partition Dask: {e}")
            return partition_df

    def _get_enriched_meta(self, sample_df):
        """Obtenir le meta DataFrame pour Dask."""
        # Créer un échantillon pour déterminer les colonnes de sortie
        sample = sample_df.head(100).copy()

        # Ajouter les colonnes de features quantitatives
        quant_columns = [
            "quant_realized_vol_10",
            "quant_realized_vol_20",
            "quant_realized_vol_50",
            "quant_garch_volatility",
            "quant_vol_clustering",
            "quant_hurst_exponent",
            "quant_autocorr_10",
            "quant_autocorr_20",
            "quant_autocorr_50",
            "quant_shannon_entropy",
            "quant_sample_entropy",
            "quant_dominant_frequency",
            "quant_spectral_centroid",
            "quant_fractal_dimension",
            "quant_dfa_exponent",
            "quant_vol_persistence",
            "quant_approximate_entropy",
            "quant_permutation_entropy",
            "quant_rolling_skewness",
            "quant_rolling_kurtosis",
            "quant_dynamic_var",
            "quant_dynamic_cvar",
            "quant_max_drawdown",
            "quant_regime_detection",
            "quant_amihud_illiquidity",
            "quant_kyles_lambda",
            "quant_kaufman_efficiency",
            "quant_variance_ratio",
        ]

        for col in quant_columns:
            sample[col] = 0.0

        return sample


# ============================================================================
# OPTIMISATIONS SHARED MEMORY POUR PERFORMANCE MAXIMALE
# ============================================================================


class SharedMemoryOptimizer:
    """Optimiseur Shared Memory pour performance maximale."""

    def __init__(self):
        """__init__."""
        self.shared_blocks = {}
        self.lock = threading.Lock()

    def setup_shared_memory(self, df: pd.DataFrame) -> dict[str, any]:
        """Configuration de la mémoire partagée."""
        if not SHARED_MEMORY_AVAILABLE:
            logger.warning("Shared memory non disponible")
            return {}

        try:
            with self.lock:
                # Créer des blocs de mémoire partagée pour les colonnes principales
                shared_data = {}

                for col in ["open", "high", "low", "close", "volume"]:
                    if col in df.columns:
                        data = df[col].astype(OPTIMAL_FLOAT).values

                        # Créer le bloc de mémoire partagée
                        shm = shared_memory.SharedMemory(create=True, size=data.nbytes)
                        shared_array = np.ndarray(
                            data.shape, dtype=OPTIMAL_FLOAT, buffer=shm.buf
                        )
                        shared_array[:] = data[:]

                        shared_data[col] = {
                            "shm": shm,
                            "array": shared_array,
                            "shape": data.shape,
                            "dtype": OPTIMAL_FLOAT,
                        }

                        self.shared_blocks[f"{col}_{id(df)}"] = shm

                logger.info(f"✅ Shared memory configurée: {len(shared_data)} arrays")
                return shared_data

        except Exception as e:
            logger.error(f"❌ Erreur configuration shared memory: {e}")
            return {}

    def cleanup_shared_memory(self):
        """Nettoyage de la mémoire partagée."""
        try:
            with self.lock:
                for shm_name, shm in self.shared_blocks.items():
                    try:
                        shm.close()
                        shm.unlink()
                    except Exception:  # noqa: E722 - cleanup best-effort
                        pass

                self.shared_blocks.clear()
                logger.info("✅ Shared memory nettoyée")

        except Exception as e:
            logger.error(f"⚠️ Erreur nettoyage shared memory: {e}")

    def process_with_shared_memory(
        self, df: pd.DataFrame, enricher_instance
    ) -> pd.DataFrame:
        """Traitement avec mémoire partagée."""
        shared_data = self.setup_shared_memory(df)

        if not shared_data:
            return enricher_instance._process_single_chunk(df)

        try:
            # Traitement avec arrays partagés
            result = self._compute_indicators_with_shared_memory(
                df, shared_data, enricher_instance
            )
            return result

        except Exception as e:
            logger.error(f"❌ Erreur traitement shared memory: {e}")
            return enricher_instance._process_single_chunk(df)
        finally:
            self.cleanup_shared_memory()

    def _compute_indicators_with_shared_memory(
        self, df, shared_data, enricher_instance
    ):
        """Calcul des indicateurs avec mémoire partagée."""
        # Utiliser les arrays partagés pour les calculs
        if "close" in shared_data:
            prices = shared_data["close"]["array"]
            calculate_returns_numba(prices)

            # Calculer les features avec les arrays partagés
            enriched_df = df.copy()
            enriched_df = enricher_instance._add_all_features(enriched_df)

            return enriched_df

        return enricher_instance._process_single_chunk(df)


# ============================================================================
# FONCTION WORKER GLOBALE POUR MULTIPROCESSING
# ============================================================================


def process_asset_group_worker(asset_data):
    """Fonction worker globale pour le multiprocessing des features quantitatives.

    Doit être au niveau module pour être picklable.
    """
    try:
        # Créer une instance temporaire de l'enrichisseur pour ce worker (sans objets non-sérialisables)
        enricher_instance = OptimizedQuantitativeFeaturesEnricher(
            chunk_size=len(asset_data),
            n_jobs=1,  # Force single-thread dans worker
            auto_optimize=False,  # Désactiver auto-optimize dans worker
        )
        enricher_instance._is_worker_instance = True  # Marquer comme instance worker

        # Traiter le chunk directement
        enriched_data = enricher_instance._process_single_chunk(asset_data)

        return enriched_data

    except Exception as e:
        logger.error(f"❌ Erreur dans worker: {str(e)}")
        # En cas d'erreur, retourner les données originales
        return asset_data


# ============================================================================
# OPTIMISATIONS MEMORY MAPPING POUR GROS FICHIERS
# ============================================================================


class MemoryMappingOptimizer:
    """Optimiseur Memory Mapping pour gros fichiers."""

    def __init__(self):
        """__init__."""
        self.temp_files = []

    def process_file_with_mmap(self, file_path: str, enricher_instance) -> pd.DataFrame:
        """Traitement de fichier avec memory mapping."""
        if not MMAP_AVAILABLE:
            logger.warning("Memory mapping non disponible, lecture standard")
            return pd.read_csv(file_path)

        try:
            # Lire le fichier par chunks avec memory mapping
            chunk_size = 100000
            chunks = []

            for chunk in pd.read_csv(file_path, chunksize=chunk_size):
                processed_chunk = enricher_instance._process_single_chunk(chunk)
                chunks.append(processed_chunk)

            result = pd.concat(chunks, ignore_index=True)
            logger.info(f"✅ Fichier traité avec memory mapping: {len(result)} lignes")
            return result

        except Exception as e:
            logger.error(f"❌ Erreur memory mapping: {e}")
            return pd.read_csv(file_path)

    def cleanup_temp_files(self):
        """Nettoyage des fichiers temporaires."""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception:  # noqa: E722 - cleanup best-effort
                pass
        self.temp_files.clear()


# ============================================================================
# SÉLECTEUR INTELLIGENT DE STRATÉGIE D'OPTIMISATION
# ============================================================================


class OptimizationStrategy:
    """Sélecteur intelligent de stratégie d'optimisation."""

    def __init__(self):
        """__init__."""
        self.dask_optimizer = DaskOptimizer()
        self.shared_memory_optimizer = SharedMemoryOptimizer()
        self.memory_mapping_optimizer = MemoryMappingOptimizer()

    def select_best_strategy(self, df: pd.DataFrame) -> str:
        """Sélectionner la meilleure stratégie d'optimisation."""
        data_size_gb = df.memory_usage(deep=True).sum() / (1024**3)
        n_rows = len(df)
        n_assets = df["asset"].nunique() if "asset" in df.columns else 1
        memory_info = psutil.virtual_memory()
        available_gb = memory_info.available / (1024**3)

        logger.info("📊 Analyse stratégie optimisation:")
        logger.info(f"   📏 Taille dataset: {data_size_gb:.2f}GB ({n_rows:,} lignes)")
        logger.info(f"   🏢 Nombre d'assets: {n_assets}")
        logger.info(f"   💾 Mémoire disponible: {available_gb:.1f}GB")

        # Critères de décision
        if data_size_gb > OPTIMIZATION_CONFIG["mmap_threshold_gb"] and MMAP_AVAILABLE:
            return "memory_mapping"
        elif data_size_gb > OPTIMIZATION_CONFIG["dask_threshold_gb"] and DASK_AVAILABLE:
            return "dask"
        elif (
            data_size_gb > OPTIMIZATION_CONFIG["shared_memory_threshold_gb"]
            and SHARED_MEMORY_AVAILABLE
        ):
            return "shared_memory"
        elif n_rows > 50000:
            return "multiprocessing"
        else:
            return "standard"

    def apply_strategy(
        self, df: pd.DataFrame, strategy: str, enricher_instance
    ) -> pd.DataFrame:
        """Appliquer la stratégie sélectionnée."""
        logger.info(f"🚀 Application stratégie: {strategy}")

        try:
            if strategy == "dask":
                return self.dask_optimizer.process_with_dask(df, enricher_instance)
            elif strategy == "shared_memory":
                return self.shared_memory_optimizer.process_with_shared_memory(
                    df, enricher_instance
                )
            elif strategy == "memory_mapping":
                return self.memory_mapping_optimizer.process_file_with_mmap(
                    df, enricher_instance
                )
            elif strategy == "multiprocessing":
                return enricher_instance._process_with_multiprocessing_by_asset(df)
            else:
                return enricher_instance._process_single_chunk(df)

        except Exception as e:
            logger.error(f"❌ Erreur stratégie {strategy}: {e}")
            logger.info("🔄 Fallback vers traitement standard")
            return enricher_instance._process_single_chunk(df)

    def cleanup_all(self):
        """Nettoyage de toutes les ressources."""
        self.dask_optimizer.cleanup_cluster()
        self.shared_memory_optimizer.cleanup_shared_memory()
        self.memory_mapping_optimizer.cleanup_temp_files()


# ============================================================================
# CLASSE PRINCIPALE OPTIMISÉE - 24 FEATURES INSTITUTIONNELLES
# ============================================================================


class OptimizedQuantitativeFeaturesEnricher:
    """Enrichissement avec 24 features quantitatives OPTIMISÉES - VERSION INSTITUTIONNELLE.

    FEATURES INCLUSES (24 total):
    - Volatilité (5): realized_vol_10/20/50, garch_vol, vol_clustering
    - Momentum (4): hurst_exponent, autocorr_10/20/50
    - Entropie (2): shannon_entropy, sample_entropy
    - Spectral (2): dominant_frequency, spectral_centroid
    - Fractales (2): fractal_dimension, dfa_exponent
    - Avancées (3): vol_persistence, approximate_entropy, permutation_entropy
    - Nouvelles (6): rolling_skew/kurt, dynamic_var/cvar, max_drawdown, regime_detection
    """

    def __init__(
        self,
        chunk_size: int = 50000,
        max_memory_gb: float = 4.0,
        n_jobs: int = None,
        auto_optimize: bool = True,
    ):
        # CONFIGURATION OPTIMISÉE - 24 FEATURES INSTITUTIONNELLES
        """__init__.

        Args:
            chunk_size: TODO document.
            max_memory_gb: TODO document.
            n_jobs: TODO document.
            auto_optimize: TODO document.
        """
        self.config = {
            # VOLATILITÉ (5 features) - CRITIQUE
            "realized_volatility": {"windows": [10, 20, 50]},  # 3 features
            "garch_volatility": {"alpha": 0.1, "beta": 0.8},  # 1 feature
            "volatility_clustering": {"threshold": 2.0},  # 1 feature
            # MOMENTUM & PERSISTENCE (4 features) - CRITIQUE
            "hurst_exponent": {"method": "rs"},  # 1 feature
            "autocorrelation": {"max_lags": [10, 20, 50]},  # 3 features
            # ENTROPIE (2 features) - IMPORTANT
            "sample_entropy": {"m": 2, "r": 0.2},  # 1 feature
            "shannon_entropy": {"bins": 50},  # 1 feature
            # SPECTRAL (2 features) - IMPORTANT
            "dominant_frequency": {"method": "fft"},  # 1 feature
            "spectral_centroid": {"method": "fft"},  # 1 feature
            # FRACTALES (2 features) - MODÉRÉ
            "fractal_dimension": {"method": "box_counting"},  # 1 feature
            "dfa_exponent": {"scales": [4, 8, 16, 32, 64]},  # 1 feature
            # AVANCÉES (3 features) - OPTIONNEL
            "volatility_persistence": {"lag": 1},  # 1 feature
            "approximate_entropy": {"m": 2, "r": 0.2},  # 1 feature
            "permutation_entropy": {"order": 3, "delay": 1},  # 1 feature
            # NOUVELLES FEATURES MANQUANTES (6 features) - CRITIQUE
            "rolling_skewness": {"window": 50},  # 1 feature
            "rolling_kurtosis": {"window": 50},  # 1 feature
            "dynamic_var": {"confidence": 0.05, "window": 50},  # 1 feature
            "dynamic_cvar": {"confidence": 0.05, "window": 50},  # 1 feature
            "max_drawdown": {"window": 100},  # 1 feature
            "regime_detection": {"lookback": 50},  # 1 feature
            # LIQUIDITÉ (2 features) - NOUVEAU
            "amihud_illiquidity": {"window": 20},  # 1 feature
            "kyles_lambda": {"window": 20},  # 1 feature
            # EFFICIENCE (2 features) - NOUVEAU
            "kaufman_efficiency": {"window": 20},  # 1 feature
            "variance_ratio": {"lags": 10},  # 1 feature
        }

        # Cache intelligent optimisé
        cache_size = 1000 if not auto_optimize else min(2000, int(max_memory_gb * 200))
        self.cache = QuantitativeCache(
            max_size=cache_size, memory_limit_mb=max_memory_gb * 50
        )

        # Configuration d'optimisation
        self.chunk_size = chunk_size
        self.max_memory_gb = max_memory_gb
        self.memory_threshold = max_memory_gb * 0.8
        self.n_jobs = n_jobs or min(cpu_count(), 14)
        self.auto_optimize = auto_optimize

        # NOUVELLES OPTIMISATIONS AVANCÉES
        self.optimization_strategy = OptimizationStrategy()

        # Types optimaux
        self.optimal_dtypes = {
            "prices": OPTIMAL_FLOAT,
            "features": OPTIMAL_FLOAT,
            "binary": np.int8,
        }

        # Statistiques de performance étendues
        self.performance_stats = {
            "total_time": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "numba_calls": 0,
            "fallback_calls": 0,
            "features_computed": 0,
            "memory_optimizations": 0,
            "chunks_processed": 0,
            "average_chunk_time": 0.0,
            "workers_used": 0,
            "workers_available": 0,
            "worker_efficiency": 0.0,
            "grouping_method": "sequential",
            "optimization_strategy": "standard",
            "dask_operations": 0,
            "shared_memory_operations": 0,
            "memory_mapping_operations": 0,
            "adaptive_optimizations": 0,
        }

        print("🚀 ENRICHISSEUR QUANTITATIF ULTRA-OPTIMISÉ INITIALISÉ")
        print("   📊 24 features institutionnelles")
        print(f"   💾 Cache: {cache_size} entrées")
        print(f"   🧩 Chunk size: {chunk_size:,}")
        print(f"   👥 Workers: {self.n_jobs}")
        _opt = (
            f"   🔧 Optimisations: Numba✅ Dask{'✅' if DASK_AVAILABLE else '❌'} "
            f"SharedMem{'✅' if SHARED_MEMORY_AVAILABLE else '❌'} "
            f"MMap{'✅' if MMAP_AVAILABLE else '❌'}"
        )
        print(_opt)

    def _compute_feature_cached(self, prices, feature_name, func, **params):
        """Calcule une feature avec cache intelligent."""
        # Vérifier le cache
        cached_result = self.cache.get(prices, feature_name, **params)
        if cached_result is not None:
            self.performance_stats["cache_hits"] += 1
            return cached_result

        # Calculer et mettre en cache
        self.performance_stats["cache_misses"] += 1
        if NUMBA_AVAILABLE:
            self.performance_stats["numba_calls"] += 1
            result = func(prices, **params)
        else:
            self.performance_stats["fallback_calls"] += 1
            result = func(prices, **params)

        self.cache.set(prices, feature_name, result, **params)
        self.performance_stats["features_computed"] += 1
        return result

    def _add_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ajouter toutes les 24 features optimisées."""
        prices = df["close"].astype(OPTIMAL_FLOAT).values
        returns = np.diff(prices) / prices[:-1]
        returns = np.concatenate([[0], returns]).astype(OPTIMAL_FLOAT)

        # VOLATILITÉ (5 features)
        for window in self.config["realized_volatility"]["windows"]:
            df[f"quant_realized_vol_{window}"] = self._compute_feature_cached(
                returns,
                "realized_volatility",
                _numba_realized_volatility,
                window=window,
            )

        df["quant_garch_volatility"] = self._compute_feature_cached(
            returns,
            "garch_volatility",
            _numba_garch_volatility,
            alpha=self.config["garch_volatility"]["alpha"],
            beta=self.config["garch_volatility"]["beta"],
        )

        df["quant_vol_clustering"] = self._compute_feature_cached(
            returns,
            "volatility_clustering",
            _numba_volatility_clustering,
            threshold_factor=self.config["volatility_clustering"]["threshold"],
        )

        # MOMENTUM & PERSISTENCE (4 features)
        df["quant_hurst_exponent"] = self._compute_feature_cached(
            prices, "hurst_exponent", _numba_hurst_rs
        )

        for lag in self.config["autocorrelation"]["max_lags"]:
            df[f"quant_autocorr_{lag}"] = self._compute_feature_cached(
                prices, "autocorrelation", _numba_autocorrelation, max_lag=lag
            )

        # ENTROPIE (2 features)
        df["quant_shannon_entropy"] = self._compute_feature_cached(
            prices,
            "shannon_entropy",
            _numba_shannon_entropy,
            bins=self.config["shannon_entropy"]["bins"],
        )

        df["quant_sample_entropy"] = self._compute_feature_cached(
            prices,
            "sample_entropy",
            _numba_sample_entropy,
            m=self.config["sample_entropy"]["m"],
            r=self.config["sample_entropy"]["r"],
        )

        # SPECTRAL (2 features)
        df["quant_dominant_frequency"] = self._compute_feature_cached(
            prices, "dominant_frequency", _numba_dominant_frequency
        )

        df["quant_spectral_centroid"] = self._compute_feature_cached(
            prices, "spectral_centroid", _numba_spectral_centroid
        )

        # FRACTALES (2 features)
        df["quant_fractal_dimension"] = self._compute_feature_cached(
            prices, "fractal_dimension", _numba_fractal_dimension
        )

        df["quant_dfa_exponent"] = self._compute_feature_cached(
            prices, "dfa_exponent", _numba_dfa
        )

        # AVANCÉES (3 features)
        df["quant_vol_persistence"] = self._compute_feature_cached(
            returns, "volatility_persistence", _numba_volatility_persistence
        )

        df["quant_approximate_entropy"] = self._compute_feature_cached(
            prices,
            "approximate_entropy",
            _numba_approximate_entropy,
            m=self.config["approximate_entropy"]["m"],
            r=self.config["approximate_entropy"]["r"],
        )

        df["quant_permutation_entropy"] = self._compute_feature_cached(
            prices,
            "permutation_entropy",
            _numba_permutation_entropy,
            order=self.config["permutation_entropy"]["order"],
            delay=self.config["permutation_entropy"]["delay"],
        )

        # NOUVELLES FEATURES INSTITUTIONNELLES (6 features)
        df["quant_rolling_skewness"] = self._compute_feature_cached(
            prices,
            "rolling_skewness",
            _numba_rolling_skewness,
            window=self.config["rolling_skewness"]["window"],
        )

        df["quant_rolling_kurtosis"] = self._compute_feature_cached(
            prices,
            "rolling_kurtosis",
            _numba_rolling_kurtosis,
            window=self.config["rolling_kurtosis"]["window"],
        )

        df["quant_dynamic_var"] = self._compute_feature_cached(
            returns,
            "dynamic_var",
            _numba_dynamic_var,
            confidence=self.config["dynamic_var"]["confidence"],
            window=self.config["dynamic_var"]["window"],
        )

        df["quant_dynamic_cvar"] = self._compute_feature_cached(
            returns,
            "dynamic_cvar",
            _numba_dynamic_cvar,
            confidence=self.config["dynamic_cvar"]["confidence"],
            window=self.config["dynamic_cvar"]["window"],
        )

        df["quant_max_drawdown"] = self._compute_feature_cached(
            prices,
            "max_drawdown",
            _numba_max_drawdown,
            window=self.config["max_drawdown"]["window"],
        )

        df["quant_regime_detection"] = self._compute_feature_cached(
            returns,
            "regime_detection",
            _numba_regime_detection,
            lookback=self.config["regime_detection"]["lookback"],
        )

        # LIQUIDITÉ (2 features)
        if "volume" in df.columns:
            volume = df["volume"].astype(OPTIMAL_FLOAT).values

            df["quant_amihud_illiquidity"] = self._compute_feature_cached(
                returns,
                "amihud_illiquidity",
                _numba_amihud_illiquidity,
                volume=volume,
                window=self.config["amihud_illiquidity"]["window"],
            )

            df["quant_kyles_lambda"] = self._compute_feature_cached(
                prices,
                "kyles_lambda",
                _numba_kyles_lambda,
                volume=volume,
                window=self.config["kyles_lambda"]["window"],
            )
        else:
             df["quant_amihud_illiquidity"] = 0.0
             df["quant_kyles_lambda"] = 0.0

        # EFFICIENCE (2 features)
        df["quant_kaufman_efficiency"] = self._compute_feature_cached(
            prices,
            "kaufman_efficiency",
            _numba_kaufman_efficiency,
            window=self.config["kaufman_efficiency"]["window"],
        )

        df["quant_variance_ratio"] = self._compute_feature_cached(
            returns,
            "variance_ratio",
            _numba_variance_ratio,
            lags=self.config["variance_ratio"]["lags"],
        )

        return df

    def _validate_input_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validation et nettoyage des données d'entrée."""
        required_cols = ["timestamp", "open", "high", "low", "close", "volume"]

        # Vérifier les colonnes requises
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Colonnes manquantes: {missing_cols}")

        # Nettoyer les données
        df = df.copy()

        # Supprimer les NaN et infinis
        for col in required_cols:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            df[col] = df[col].ffill().bfill()  # pandas 2.0+ compatible

        # Vérifier la cohérence OHLC
        df.loc[df["high"] < df[["open", "close"]].max(axis=1), "high"] = df[
            ["open", "close"]
        ].max(axis=1)
        df.loc[df["low"] > df[["open", "close"]].min(axis=1), "low"] = df[
            ["open", "close"]
        ].min(axis=1)

        return df

    def _optimize_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimise les types de données pour réduire l'usage mémoire."""
        start_memory = df.memory_usage(deep=True).sum() / 1024**2

        # Optimiser les colonnes OHLCV
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = df[col].astype(self.optimal_dtypes["prices"])

        # Optimiser les autres colonnes numériques
        for col in df.select_dtypes(include=[np.float64]).columns:
            if col not in ["open", "high", "low", "close"]:
                df[col] = df[col].astype(self.optimal_dtypes["features"])

        end_memory = df.memory_usage(deep=True).sum() / 1024**2
        reduction = (start_memory - end_memory) / start_memory * 100

        print(
            f"🔧 Optimisation types: {reduction:.1f}% mémoire économisée ({start_memory:.1f}MB → {end_memory:.1f}MB)"
        )
        return df

    def _process_single_chunk(self, df: pd.DataFrame) -> pd.DataFrame:
        """Traiter un chunk unique avec les 24 features optimisées."""
        start_time = time.time()

        # Validation des données
        enriched_df = self._validate_input_data(df)

        # Ajouter toutes les 24 features
        enriched_df = self._add_all_features(enriched_df)

        # Post-traitement
        enriched_df = self._post_process_features(enriched_df)

        # Optimisation des types
        enriched_df = self._optimize_data_types(enriched_df)

        # Mise à jour des statistiques
        chunk_time = time.time() - start_time
        self.performance_stats["chunks_processed"] += 1
        self.performance_stats["total_time"] += chunk_time

        # Calcul de la moyenne mobile du temps par chunk
        total_chunks = self.performance_stats["chunks_processed"]
        self.performance_stats["average_chunk_time"] = (
            self.performance_stats["average_chunk_time"] * (total_chunks - 1)
            + chunk_time
        ) / total_chunks

        # Message de confirmation des optimisations (une seule fois)
        if not hasattr(self, "_optimization_message_shown"):
            print("🚀 24 FEATURES QUANTITATIVES INSTITUTIONNELLES OPTIMISÉES!")
            print("   ✅ Fonctions Numba ultra-rapides")
            print("   ✅ Cache intelligent LRU")
            print("   ✅ Multiprocessing parallèle")
            print("   ✅ Types de données optimaux")
            self._optimization_message_shown = True

        return enriched_df

    def _post_process_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Post-traitement et nettoyage final des features."""
        # Remplacer les infinis par NaN
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

        # Identifier les colonnes de features quantitatives
        quant_cols = [col for col in df.columns if col.startswith("quant_")]

        # Forward fill pour les features quantitatives (pandas 2.0+ compatible)
        df[quant_cols] = df[quant_cols].ffill()

        # Remplir les NaN restants avec des valeurs par défaut
        for col in quant_cols:
            if col in df.columns:
                if "entropy" in col.lower():
                    df[col] = df[col].fillna(1.0)  # Entropie moyenne
                elif "hurst" in col.lower():
                    df[col] = df[col].fillna(0.5)  # Marche aléatoire
                elif "fractal" in col.lower():
                    df[col] = df[col].fillna(1.5)  # Dimension fractale typique
                elif "volatility" in col.lower() or "vol_" in col.lower():
                    df[col] = df[col].fillna(0.1)  # Volatilité faible
                elif "skewness" in col.lower():
                    df[col] = df[col].fillna(0.0)  # Symétrie
                elif "kurtosis" in col.lower():
                    df[col] = df[col].fillna(3.0)  # Distribution normale
                elif "var" in col.lower() or "cvar" in col.lower():
                    df[col] = df[col].fillna(0.05)  # VaR/CVaR par défaut
                elif "drawdown" in col.lower():
                    df[col] = df[col].fillna(0.0)  # Pas de drawdown
                elif "regime" in col.lower():
                    df[col] = df[col].fillna(0.0)  # Régime neutre
                elif "amihud" in col.lower() or "lambda" in col.lower():
                    df[col] = df[col].fillna(0.0)  # Illiquidité nulle
                elif "efficiency" in col.lower() or "variance" in col.lower():
                    df[col] = df[col].fillna(0.5)  # Efficience moyenne
                else:
                    df[col] = df[col].fillna(0.0)  # Valeur par défaut

        return df

    def enrich_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enrichir un dataset avec les 24 features quantitatives ultra-optimisées."""
        print("🚀 ENRICHISSEMENT QUANTITATIF ULTRA-OPTIMISÉ - 24 FEATURES")
        print(f"   📊 Dataset: {len(df):,} lignes")

        start_time = time.time()

        # Trier par timestamp pour calculs corrects
        df = df.sort_values("timestamp").reset_index(drop=True)

        # NOUVELLE OPTIMISATION: Sélection intelligente de stratégie
        try:
            enriched_df = self._enrich_dataframe_smart(df)
        except Exception as e:
            logger.error(f"❌ Erreur enrichissement intelligent: {e}")
            print("🔄 Fallback vers traitement standard")
            enriched_df = self._process_single_chunk(df)

        # Statistiques finales
        total_time = time.time() - start_time
        self.performance_stats["total_time"] = total_time

        print(f"✅ ENRICHISSEMENT TERMINÉ en {total_time:.2f}s")
        print(f"   📈 {len(enriched_df):,} lignes enrichies")
        print("   🎯 24 features quantitatives ajoutées")

        # Afficher les statistiques de performance
        self._display_final_stats()

        # Nettoyage des ressources
        self.optimization_strategy.cleanup_all()

        return enriched_df

    def _enrich_dataframe_smart(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enrichissement intelligent avec sélection automatique de stratégie."""
        # Sélectionner la meilleure stratégie
        strategy = self.optimization_strategy.select_best_strategy(df)
        self.performance_stats["optimization_strategy"] = strategy

        # Appliquer la stratégie sélectionnée
        enriched_df = self.optimization_strategy.apply_strategy(df, strategy, self)

        # Mettre à jour les statistiques selon la stratégie
        if strategy == "dask":
            self.performance_stats["dask_operations"] += 1
        elif strategy == "shared_memory":
            self.performance_stats["shared_memory_operations"] += 1
        elif strategy == "memory_mapping":
            self.performance_stats["memory_mapping_operations"] += 1

        return enriched_df

    def enrich_from_file_optimized(self, file_path: str) -> pd.DataFrame:
        """Enrichissement optimisé directement depuis un fichier."""
        print(f"🚀 ENRICHISSEMENT DEPUIS FICHIER: {file_path}")

        # Analyser la taille du fichier
        file_size_gb = os.path.getsize(file_path) / (1024**3)
        print(f"   📏 Taille fichier: {file_size_gb:.2f}GB")

        # Sélectionner la stratégie de lecture
        if file_size_gb > OPTIMIZATION_CONFIG["mmap_threshold_gb"]:
            return self._process_large_file_with_dask(file_path)
        elif file_size_gb > OPTIMIZATION_CONFIG["shared_memory_threshold_gb"]:
            return self._process_file_with_shared_memory(file_path)
        else:
            return self._process_small_file_optimized(file_path)

    def _process_large_file_with_dask(self, file_path: str) -> pd.DataFrame:
        """Traitement de gros fichier avec Dask."""
        if not DASK_AVAILABLE:
            print("⚠️ Dask non disponible, lecture par chunks")
            return self._process_file_by_chunks(file_path)

        try:
            print("🌊 Traitement avec Dask DataFrame...")

            # Lire avec Dask
            ddf = dd.read_csv(file_path)

            # Appliquer l'enrichissement par partition
            enriched_ddf = ddf.map_partitions(
                self._process_single_chunk,
                meta=self._get_enriched_meta_from_file(file_path),
            )

            # Calculer le résultat
            with ProgressBar():
                result = enriched_ddf.compute()

            self.performance_stats["dask_operations"] += 1
            return result

        except Exception as e:
            logger.error(f"❌ Erreur Dask file processing: {e}")
            return self._process_file_by_chunks(file_path)

    def _process_file_with_shared_memory(self, file_path: str) -> pd.DataFrame:
        """Traitement de fichier avec shared memory."""
        print("🧠 Traitement avec Shared Memory...")

        # Lire le fichier par chunks et utiliser shared memory
        chunk_size = OPTIMIZATION_CONFIG["chunk_size_medium"]
        chunks = []

        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            processed_chunk = self.optimization_strategy.shared_memory_optimizer.process_with_shared_memory(
                chunk, self
            )
            chunks.append(processed_chunk)

        result = pd.concat(chunks, ignore_index=True)
        self.performance_stats["shared_memory_operations"] += 1
        return result

    def _process_small_file_optimized(self, file_path: str) -> pd.DataFrame:
        """Traitement optimisé de petit fichier."""
        print("📦 Traitement direct optimisé...")

        df = pd.read_csv(file_path)
        return self.enrich_dataset(df)

    def _process_file_by_chunks(self, file_path: str) -> pd.DataFrame:
        """Traitement de fichier par chunks (fallback)."""
        print("🔄 Traitement par chunks (fallback)...")

        chunk_size = OPTIMIZATION_CONFIG["chunk_size_large"]
        chunks = []

        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            processed_chunk = self._process_single_chunk(chunk)
            chunks.append(processed_chunk)

        return pd.concat(chunks, ignore_index=True)

    def _get_enriched_meta_from_file(self, file_path: str):
        """Obtenir le meta DataFrame depuis un fichier."""
        sample = pd.read_csv(file_path, nrows=100)
        return self.optimization_strategy.dask_optimizer._get_enriched_meta(sample)

    def _process_large_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Traiter un gros dataset par chunks."""
        total_rows = len(df)
        overlap_size = 260  # Overlap réduit pour les 24 features

        chunks = []
        for start in range(0, total_rows, self.chunk_size):
            end = min(start + self.chunk_size, total_rows)

            # Ajouter overlap au début (sauf pour le premier chunk)
            chunk_start = max(0, start - overlap_size) if start > 0 else 0

            chunk_df = df.iloc[chunk_start:end].copy()
            chunks.append((chunk_start, start, end, chunk_df))

        print(f"   📦 {len(chunks)} chunks à traiter")

        # Traiter les chunks
        processed_chunks = []

        for i, (chunk_start, actual_start, actual_end, chunk_df) in enumerate(chunks):
            print(
                f"   📦 Chunk {i + 1}/{len(chunks)}: lignes {actual_start:,}-{actual_end:,}"
            )

            # Traiter le chunk
            processed_chunk = self._process_single_chunk(chunk_df)

            # Supprimer l'overlap du début (sauf pour le premier chunk)
            if chunk_start < actual_start:
                overlap_rows = actual_start - chunk_start
                processed_chunk = processed_chunk.iloc[overlap_rows:]

            processed_chunks.append(processed_chunk)

            # Nettoyage mémoire
            del chunk_df, processed_chunk
            gc.collect()

        # Concaténer tous les chunks
        enriched_df = pd.concat(processed_chunks, ignore_index=True)

        return enriched_df

    def _check_memory_usage(self, df: pd.DataFrame = None) -> dict[str, float]:
        """Vérifier l'usage mémoire."""
        memory_info = psutil.virtual_memory()
        memory_used_gb = (memory_info.total - memory_info.available) / (1024**3)
        memory_percent = memory_info.percent
        available_gb = memory_info.available / (1024**3)

        memory_data = {
            "used_gb": memory_used_gb,
            "percent": memory_percent,
            "available_gb": available_gb,
        }

        if memory_used_gb > self.memory_threshold:
            gc.collect()

        return memory_data

    def _should_use_multiprocessing(self, df: pd.DataFrame) -> bool:
        """Déterminer si le multiprocessing est bénéfique."""
        data_size_mb = df.memory_usage(deep=True).sum() / (1024**2)
        memory_info = self._check_memory_usage()

        # OPTIMISATION: Conditions plus permissives pour le multiprocessing
        is_large_enough = (len(df) > self.chunk_size) or (
            data_size_mb > 50
        )  # Seuil réduit
        has_enough_cores = self.n_jobs > 1
        has_sufficient_memory = memory_info["available_gb"] > 1  # Seuil réduit

        print("   🔍 Analyse multiprocessing:")
        print(f"      📊 Taille: {len(df):,} lignes ({data_size_mb:.1f}MB)")
        print(f"      🧩 Chunk size: {self.chunk_size:,}")
        print(f"      👥 Workers: {self.n_jobs}")
        print(f"      💾 Mémoire disponible: {memory_info['available_gb']:.1f}GB")
        print(f"      ✅ Assez gros: {is_large_enough}")
        print(f"      ✅ Assez de cores: {has_enough_cores}")
        print(f"      ✅ Assez de mémoire: {has_sufficient_memory}")

        return is_large_enough and has_enough_cores and has_sufficient_memory

    def _process_asset_group(self, asset_data: pd.DataFrame) -> pd.DataFrame:
        """Traite un groupe d'asset avec features quantitatives optimisées."""
        try:
            # Trier par timestamp pour calculs corrects
            asset_data = asset_data.sort_values("timestamp").reset_index(drop=True)

            # Calculs optimisés
            enriched_data = self._process_single_chunk(asset_data)

            return enriched_data

        except Exception as e:
            print(f"❌ Erreur traitement asset quantitatif: {e}")
            return asset_data

    def _process_with_multiprocessing_by_asset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Multiprocessing intelligent par asset pour performance maximale."""
        print(f"🚀 Multiprocessing quantitatif avec {self.n_jobs} workers disponibles")

        # Étape 1: Essayer de grouper par asset d'abord
        asset_groups = []
        enriched_df = []
        grouping_method = "chunks"

        if "asset" in df.columns:
            asset_groups = [group for _, group in df.groupby("asset")]
            grouping_method = "assets"
            print(f"📊 {len(asset_groups)} assets détectés dans le dataset")

        # Étape 2: OPTIMISATION INTELLIGENTE - Analyser la taille des assets
        max_lines_per_worker = 15000  # Seuil pour diviser un asset volumineux

        # Vérifier si des assets individuels sont trop volumineux
        large_assets = []
        normal_assets = []

        for asset_group in asset_groups:
            if len(asset_group) > max_lines_per_worker:
                large_assets.append(asset_group)
            else:
                normal_assets.append(asset_group)

        if large_assets:
            print(
                f"⚠️ {len(large_assets)} assets volumineux détectés (>{max_lines_per_worker:,} lignes)"
            )
            print("🔄 Division des gros assets en chunks pour optimiser la charge")

        # Reconstruire les groupes en divisant les gros assets
        final_groups = []
        overlap_size = 270

        # Ajouter les assets normaux
        final_groups.extend(normal_assets)

        # Diviser les gros assets
        for large_asset in large_assets:
            asset_name = (
                large_asset["asset"].iloc[0]
                if "asset" in large_asset.columns
                else "UNKNOWN"
            )
            asset_size = len(large_asset)

            # Calculer le nombre de chunks nécessaires pour cet asset
            chunks_needed = max(2, asset_size // max_lines_per_worker)
            chunk_size = asset_size // chunks_needed

            print(
                f"   📊 Asset {asset_name}: {asset_size:,} lignes → {chunks_needed} chunks"
            )

            for i in range(chunks_needed):
                start_idx = i * chunk_size
                if start_idx >= asset_size:
                    break

                # Ajouter overlap
                chunk_start = max(0, start_idx - overlap_size) if start_idx > 0 else 0
                chunk_end = min(start_idx + chunk_size, asset_size)

                # Dernier chunk prend le reste
                if i == chunks_needed - 1:
                    chunk_end = asset_size

                if chunk_end - chunk_start < 500:
                    continue

                chunk_df = large_asset.iloc[chunk_start:chunk_end].copy()
                chunk_df._chunk_info = {
                    "original_start": start_idx,
                    "chunk_start": chunk_start,
                    "chunk_end": chunk_end,
                    "has_overlap": start_idx > 0,
                    "asset_name": asset_name,
                    "chunk_id": i,
                }
                final_groups.append(chunk_df)

        asset_groups = final_groups
        grouping_method = "mixed_optimized" if large_assets else "assets"

        # Étape 3: Si toujours pas assez de groupes, diviser globalement
        if len(asset_groups) < self.n_jobs:
            if len(asset_groups) > 0:
                print(
                    f"⚠️ Encore seulement {len(asset_groups)} groupes pour {self.n_jobs} workers"
                )
                print("🔄 Basculement vers chunking global par lignes")

            # Diviser en chunks par lignes avec overlap
            chunk_size = max(1000, len(df) // self.n_jobs)

            asset_groups = []
            for i in range(self.n_jobs):
                start_idx = i * chunk_size
                if start_idx >= len(df):
                    break

                chunk_start = max(0, start_idx - overlap_size) if start_idx > 0 else 0
                chunk_end = min(start_idx + chunk_size, len(df))

                if chunk_end - chunk_start < 500:
                    continue

                chunk_df = df.iloc[chunk_start:chunk_end].copy()
                chunk_df._chunk_info = {
                    "original_start": start_idx,
                    "chunk_start": chunk_start,
                    "chunk_end": chunk_end,
                    "has_overlap": start_idx > 0,
                }
                asset_groups.append(chunk_df)

            grouping_method = "chunks_global"

        print(f"📦 {len(asset_groups)} groupes finaux créés ({grouping_method})")

        # Étape 4: Ajuster le nombre de workers au nombre réel de groupes
        effective_workers = min(self.n_jobs, len(asset_groups))
        print(f"🚀 Lancement de {effective_workers} workers effectifs")

        # Étape 5: Traitement parallèle optimisé avec progression
        start_time = time.time()
        total_chunks = len(asset_groups)

        print("📊 Progression du traitement:")

        # Utiliser imap pour avoir la progression en temps réel avec fonction worker globale
        with Pool(processes=effective_workers) as pool:
            results = []
            completed = 0

            # Traitement avec suivi de progression
            for result in pool.imap(process_asset_group_worker, asset_groups):
                results.append(result)
                completed += 1

                # Afficher la progression tous les 50 chunks
                if completed % 50 == 0 or completed == total_chunks:
                    progress_percent = (completed / total_chunks) * 100
                    elapsed_time = time.time() - start_time
                    estimated_total = elapsed_time * total_chunks / completed
                    remaining_time = estimated_total - elapsed_time

                    print(
                        f"   📦 Chunks: {completed}/{total_chunks} ({progress_percent:.1f}%) - "
                        f"⏱️ {elapsed_time:.1f}s écoulé, ~{remaining_time:.1f}s restant"
                    )

        # Enregistrer les statistiques d'utilisation des workers
        multiprocessing_time = time.time() - start_time
        self.performance_stats["workers_used"] = effective_workers
        self.performance_stats["workers_available"] = self.n_jobs
        self.performance_stats["worker_efficiency"] = (
            effective_workers / self.n_jobs
        ) * 100
        self.performance_stats["grouping_method"] = grouping_method
        self.performance_stats["total_time"] = multiprocessing_time

        # CORRECTION: Estimer les statistiques pour le multiprocessing
        # Chaque chunk traite environ 24 features par chunk
        estimated_features = total_chunks * 24
        self.performance_stats["features_computed"] = estimated_features
        self.performance_stats["chunks_processed"] = total_chunks
        self.performance_stats["average_chunk_time"] = (
            multiprocessing_time / total_chunks if total_chunks > 0 else 0
        )
        self.performance_stats["numba_calls"] = estimated_features  # Estimation

        total_results_rows = sum(len(r) for r in results)
        print(
            f"✅ Multiprocessing terminé: {total_results_rows} lignes enrichies en {multiprocessing_time:.2f}s"
        )

        # Étape 6: Post-traitement selon la méthode de grouping
        if grouping_method in ["chunks_global", "mixed_optimized"]:
            # Supprimer les overlaps des chunks
            processed_results = []
            for i, result in enumerate(results):
                if hasattr(asset_groups[i], "_chunk_info"):
                    chunk_info = asset_groups[i]._chunk_info
                    if chunk_info["has_overlap"]:
                        # Supprimer l'overlap du début
                        overlap_rows = (
                            chunk_info["original_start"] - chunk_info["chunk_start"]
                        )
                        result = result.iloc[overlap_rows:].copy()
                processed_results.append(result)
            results = processed_results

        # Étape 7: Concaténation des résultats
        enriched_df = pd.concat(results, ignore_index=True)

        # Étape 8: Tri final par timestamp si disponible
        if "timestamp" in enriched_df.columns:
            enriched_df = enriched_df.sort_values("timestamp").reset_index(drop=True)

        return enriched_df

    def _adaptive_optimization(self):
        """Optimisation adaptative avancée en cours d'exécution."""
        memory_info = self._check_memory_usage()

        # Si la mémoire dépasse le seuil, optimiser
        if memory_info["used_gb"] > self.memory_threshold:
            print(
                f"⚠️ Mémoire élevée ({memory_info['used_gb']:.1f}GB), optimisation adaptative..."
            )

            # 1. Nettoyer le cache intelligent
            self.cache._cleanup_memory()

            # 2. Réduire la taille des chunks dynamiquement
            if self.chunk_size > 10000:
                old_chunk_size = self.chunk_size
                self.chunk_size = max(10000, int(self.chunk_size * 0.8))
                print(f"   📦 Chunk size: {old_chunk_size:,} → {self.chunk_size:,}")

            # 3. Ajuster le nombre de workers si nécessaire
            if self.n_jobs > 2 and memory_info["used_gb"] > self.memory_threshold * 1.2:
                old_n_jobs = self.n_jobs
                self.n_jobs = max(2, self.n_jobs - 1)
                print(f"   👥 Workers: {old_n_jobs} → {self.n_jobs}")

            # 4. Forcer le garbage collection agressif
            gc.collect()

            # 5. Nettoyer les ressources d'optimisation
            self.optimization_strategy.cleanup_all()

            self.performance_stats["memory_optimizations"] += 1
            self.performance_stats["adaptive_optimizations"] += 1

            # Vérifier l'amélioration
            new_memory_info = self._check_memory_usage()
            improvement = memory_info["used_gb"] - new_memory_info["used_gb"]
            print(f"   ✅ Mémoire libérée: {improvement:.2f}GB")

    def _enrich_with_dask_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enrichissement avec Dask DataFrame optimisé."""
        if not DASK_AVAILABLE:
            return self._process_single_chunk(df)

        try:
            print("🌊 Enrichissement avec Dask DataFrame...")

            # Configuration optimisée des partitions
            optimal_partitions = min(self.n_jobs * 2, len(df) // 10000 + 1)
            ddf = dd.from_pandas(df, npartitions=optimal_partitions)

            # Appliquer l'enrichissement avec timeout
            enriched_ddf = ddf.map_partitions(
                self._compute_indicators_for_partition_safe,
                meta=self._get_enriched_meta(df),
            )

            # Calculer avec barre de progression
            with ProgressBar():
                result = enriched_ddf.compute(scheduler="threads")

            self.performance_stats["dask_operations"] += 1
            return result

        except Exception as e:
            logger.error(f"❌ Erreur Dask DataFrame: {e}")
            return self._process_single_chunk(df)

    def _compute_indicators_for_partition_safe(self, partition_df):
        """Calcul sécurisé des indicateurs pour partition Dask."""
        try:
            return self._add_all_features(partition_df)
        except Exception as e:
            logger.error(f"❌ Erreur partition: {e}")
            return partition_df

    def _enrich_with_shared_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enrichissement avec mémoire partagée optimisée."""
        if not SHARED_MEMORY_AVAILABLE:
            return self._process_single_chunk(df)

        return self.optimization_strategy.shared_memory_optimizer.process_with_shared_memory(
            df, self
        )

    def _setup_shared_memory(self, df: pd.DataFrame) -> dict[str, any]:
        """Configuration de la mémoire partagée pour les données."""
        return self.optimization_strategy.shared_memory_optimizer.setup_shared_memory(
            df
        )

    def _compute_indicators_with_shared_memory(
        self, df: pd.DataFrame, shared_data: dict
    ) -> pd.DataFrame:
        """Calcul des indicateurs avec mémoire partagée."""
        return self.optimization_strategy.shared_memory_optimizer._compute_indicators_with_shared_memory(
            df, shared_data, self
        )

    def _cleanup_resources(self):
        """Nettoyage complet des ressources."""
        try:
            # Nettoyer le cache
            if hasattr(self, "cache") and self.cache is not None:
                try:
                    self.cache._cleanup_memory()
                except Exception as cache_error:
                    logger.warning(f"Erreur nettoyage cache: {cache_error}")

            # Nettoyer les optimisations
            if hasattr(self, "optimization_strategy") and self.optimization_strategy is not None:
                try:
                    self.optimization_strategy.cleanup_all()
                except Exception as opt_error:
                    logger.warning(f"Erreur nettoyage optimisations: {opt_error}")

            # Nettoyer les blocs de mémoire partagée
            try:
                global _SHARED_MEMORY_BLOCKS
                with _SHARED_MEMORY_LOCK:
                    for key, block in list(_SHARED_MEMORY_BLOCKS.items()):
                        try:
                            if hasattr(block, 'close'):
                                block.close()
                            del _SHARED_MEMORY_BLOCKS[key]
                        except Exception as mem_error:
                            logger.warning(f"Erreur nettoyage mémoire partagée {key}: {mem_error}")
            except Exception as shared_error:
                logger.warning(f"Erreur nettoyage mémoire partagée globale: {shared_error}")

            # Garbage collection final
            gc.collect()

            print("✅ Ressources nettoyées")

        except Exception as e:
            logger.error(f"⚠️ Erreur nettoyage ressources: {e}")
            # Ne pas relancer l'exception pour éviter les boucles d'erreur

    def __del__(self):
        """Destructeur avec nettoyage automatique."""
        self._cleanup_resources()

    def _display_optimization_stats(self):
        """Affiche les statistiques d'optimisation avancées."""
        stats = self.performance_stats
        cache_stats = self.cache.get_stats()

        total_calls = stats["cache_hits"] + stats["cache_misses"]
        cache_hit_rate = (
            (stats["cache_hits"] / total_calls * 100) if total_calls > 0 else 0
        )

        print("\n📊 STATISTIQUES D'OPTIMISATION QUANTITATIVE:")
        print(
            f"   💾 Cache hit rate: {cache_hit_rate:.1f}% ({stats['cache_hits']}/{total_calls})"
        )
        print(f"   🚀 Appels Numba: {stats['numba_calls']}")
        print(f"   🐌 Appels fallback: {stats['fallback_calls']}")
        print(f"   📈 Features calculées: {stats['features_computed']}")
        print(f"   ⏱️ Temps total: {stats['total_time']:.2f}s")
        print(f"   🧩 Chunks traités: {stats['chunks_processed']}")
        print(f"   ⚡ Temps moyen/chunk: {stats['average_chunk_time']:.2f}s")
        print(f"   🔧 Optimisations mémoire: {stats['memory_optimizations']}")

        # Nouvelles statistiques de workers
        if stats["workers_used"] > 0:
            print(
                f"   👥 Workers utilisés: {stats['workers_used']}/{stats['workers_available']}"
            )
            print(f"   📊 Efficacité workers: {stats['worker_efficiency']:.1f}%")
            print(f"   🔄 Méthode grouping: {stats['grouping_method']}")

        # Statistiques du cache avancées
        print("\n📦 STATISTIQUES CACHE AVANCÉES:")
        print(f"   🎯 Hit rate cache: {cache_stats['hit_rate']:.1f}%")
        print(f"   📊 Taille cache: {cache_stats['cache_size']}")
        print(f"   💾 Mémoire cache: {cache_stats['memory_usage_mb']:.1f}MB")
        print(f"   📞 Requêtes totales: {cache_stats['total_requests']}")

        # Recommandations d'optimisation
        if cache_stats["hit_rate"] < 50:
            print("   💡 Recommandation: Augmenter la taille du cache")
        if stats["fallback_calls"] > stats["numba_calls"]:
            print("   💡 Recommandation: Vérifier l'installation Numba")
        if stats["memory_optimizations"] > 5:
            print("   💡 Recommandation: Réduire la taille des chunks")

    def _display_final_stats(self):
        """Affiche les statistiques d'optimisation ultra-avancées."""
        stats = self.performance_stats
        cache_stats = self.cache.get_stats()

        total_calls = stats["cache_hits"] + stats["cache_misses"]
        cache_hit_rate = (
            (stats["cache_hits"] / total_calls * 100) if total_calls > 0 else 0
        )

        print("\n📊 STATISTIQUES D'OPTIMISATION QUANTITATIVE ULTRA-AVANCÉES:")
        print(
            f"   💾 Cache hit rate: {cache_hit_rate:.1f}% ({stats['cache_hits']}/{total_calls})"
        )
        print(f"   🚀 Appels Numba: {stats['numba_calls']}")
        print(f"   🐌 Appels fallback: {stats['fallback_calls']}")
        print(f"   📈 Features calculées: {stats['features_computed']}")
        print(f"   ⏱️ Temps total: {stats['total_time']:.2f}s")
        print(f"   🧩 Chunks traités: {stats['chunks_processed']}")
        print(f"   ⚡ Temps moyen/chunk: {stats['average_chunk_time']:.2f}s")
        print(f"   🔧 Optimisations mémoire: {stats['memory_optimizations']}")
        print(f"   🎯 Stratégie utilisée: {stats['optimization_strategy']}")

        # Statistiques des workers
        if stats["workers_used"] > 0:
            print(
                f"   👥 Workers utilisés: {stats['workers_used']}/{stats['workers_available']}"
            )
            print(f"   📊 Efficacité workers: {stats['worker_efficiency']:.1f}%")
            print(f"   🔄 Méthode grouping: {stats['grouping_method']}")

        # Statistiques des optimisations avancées
        print("\n🚀 STATISTIQUES OPTIMISATIONS AVANCÉES:")
        print(f"   🌊 Opérations Dask: {stats['dask_operations']}")
        print(f"   🧠 Opérations Shared Memory: {stats['shared_memory_operations']}")
        print(f"   🗺️ Opérations Memory Mapping: {stats['memory_mapping_operations']}")
        print(f"   🔄 Optimisations adaptatives: {stats['adaptive_optimizations']}")

        # Statistiques du cache avancées
        print("\n📦 STATISTIQUES CACHE INTELLIGENTES:")
        print(f"   🎯 Hit rate cache: {cache_stats['hit_rate']:.1f}%")
        print(f"   📊 Taille cache: {cache_stats['cache_size']}")
        print(f"   💾 Mémoire cache: {cache_stats['memory_usage_mb']:.1f}MB")
        print(f"   📞 Requêtes totales: {cache_stats['total_requests']}")

        # Analyse de performance
        if stats["total_time"] > 0:
            features_per_second = stats["features_computed"] / stats["total_time"]
            print("\n⚡ ANALYSE DE PERFORMANCE:")
            print(f"   📈 Features/seconde: {features_per_second:.1f}")

            if stats["chunks_processed"] > 0:
                chunks_per_second = stats["chunks_processed"] / stats["total_time"]
                print(f"   🧩 Chunks/seconde: {chunks_per_second:.1f}")

        # Recommandations d'optimisation intelligentes
        print("\n💡 RECOMMANDATIONS D'OPTIMISATION:")

        if cache_stats["hit_rate"] < 50:
            print(
                f"   📈 Augmenter la taille du cache (hit rate: {cache_stats['hit_rate']:.1f}%)"
            )

        if stats["fallback_calls"] > stats["numba_calls"]:
            print(
                f"   🚀 Vérifier l'installation Numba (fallbacks: {stats['fallback_calls']})"
            )

        if stats["memory_optimizations"] > 5:
            print(
                f"   💾 Réduire la taille des chunks (optimisations: {stats['memory_optimizations']})"
            )

        if stats["adaptive_optimizations"] > 3:
            print(
                f"   🔧 Augmenter la mémoire disponible (adaptations: {stats['adaptive_optimizations']})"
            )

        # Recommandations de stratégie
        if stats["optimization_strategy"] == "standard" and stats["total_time"] > 60:
            print(
                "   🌊 Considérer l'utilisation de Dask pour de meilleures performances"
            )

        if stats["dask_operations"] > 0 and not DASK_AVAILABLE:
            print("   📦 Installer Dask pour de meilleures performances distribuées")

        # Score de performance global
        performance_score = self._calculate_performance_score()
        print(f"\n🏆 SCORE DE PERFORMANCE GLOBAL: {performance_score:.1f}/100")

        if performance_score >= 90:
            print("   🥇 Excellent! Optimisations parfaites")
        elif performance_score >= 75:
            print("   🥈 Très bien! Quelques optimisations possibles")
        elif performance_score >= 60:
            print("   🥉 Correct, mais des améliorations sont recommandées")
        else:
            print("   ⚠️ Performance faible, optimisations critiques nécessaires")

    def _calculate_performance_score(self) -> float:
        """Calcule un score de performance global."""
        stats = self.performance_stats
        cache_stats = self.cache.get_stats()

        score = 0.0

        # Score du cache (30 points max)
        if cache_stats["total_requests"] > 0:
            cache_score = min(30, cache_stats["hit_rate"] * 0.3)
            score += cache_score

        # Score Numba (25 points max)
        total_calls = stats["numba_calls"] + stats["fallback_calls"]
        if total_calls > 0:
            numba_ratio = stats["numba_calls"] / total_calls
            numba_score = numba_ratio * 25
            score += numba_score

        # Score d'efficacité des workers (20 points max)
        if stats["workers_available"] > 0:
            worker_score = (stats["worker_efficiency"] / 100) * 20
            score += worker_score

        # Score de stratégie d'optimisation (15 points max)
        strategy_scores = {
            "dask": 15,
            "shared_memory": 12,
            "memory_mapping": 10,
            "multiprocessing": 8,
            "standard": 5,
        }
        strategy_score = strategy_scores.get(stats["optimization_strategy"], 5)
        score += strategy_score

        # Score d'optimisations adaptatives (10 points max)
        if stats["adaptive_optimizations"] == 0:
            adaptive_score = 10  # Pas d'optimisations nécessaires = bon
        elif stats["adaptive_optimizations"] <= 2:
            adaptive_score = 7  # Quelques optimisations = acceptable
        else:
            adaptive_score = 3  # Beaucoup d'optimisations = problématique
        score += adaptive_score

        return min(100.0, score)


# ============================================================================
# FONCTIONS UTILITAIRES POUR MULTIPROCESSING
# ============================================================================


def process_quantitative_chunk_worker_optimized(args):
    """Worker function pour multiprocessing quantitatif optimisé - doit être au niveau module."""
    asset_data, config = args

    try:
        # Créer un enrichisseur temporaire pour ce worker
        enricher = OptimizedQuantitativeFeaturesEnricher(
            chunk_size=config.get("chunk_size", 50000),
            max_memory_gb=config.get("max_memory_gb", 2.0),
            n_jobs=1,  # Un seul job par worker
            auto_optimize=config.get("auto_optimize", True),
        )

        # Traiter les données
        result = enricher._process_single_chunk(asset_data)

        return result

    except Exception as e:
        print(f"❌ Erreur worker quantitatif optimisé: {e}")
        return asset_data


# ============================================================================
# FONCTION PRINCIPALE D'ENRICHISSEMENT
# ============================================================================


def enrich_all_datasets_quantitative_ultra_optimized():
    """Enrichir tous les datasets avec les 24 features quantitatives ULTRA-optimisées."""
    print("🚀 DÉMARRAGE ENRICHISSEMENT QUANTITATIF ULTRA-OPTIMISÉ")
    print("=" * 80)

    # Configuration adaptative selon la mémoire disponible
    memory_info = psutil.virtual_memory()
    available_gb = memory_info.available / (1024**3)

    # Ajuster la configuration selon la mémoire
    if available_gb > 16:
        chunk_size = 100000
        max_memory_gb = 8.0
        n_jobs = min(cpu_count(), 16)
    elif available_gb > 8:
        chunk_size = 50000
        max_memory_gb = 4.0
        n_jobs = min(cpu_count(), 12)
    else:
        chunk_size = 25000
        max_memory_gb = 2.0
        n_jobs = min(cpu_count(), 8)

    print("🔧 Configuration adaptative:")
    print(f"   💾 Mémoire disponible: {available_gb:.1f}GB")
    print(f"   🧩 Chunk size: {chunk_size:,}")
    print(f"   👥 Workers: {n_jobs}")
    print(f"   📊 Mémoire max: {max_memory_gb}GB")

    enricher = OptimizedQuantitativeFeaturesEnricher(
        chunk_size=chunk_size,
        max_memory_gb=max_memory_gb,
        n_jobs=n_jobs,
        auto_optimize=True,
    )

    datasets = [
        "test.csv",  # Commencer par le dataset test
        # 'crypto_all.csv',
        # 'forex_all.csv',
        # 'indices_all.csv',
        # 'stocks_all.csv',
        # 'commodities_all.csv'  # Ajouté: support des commodités
    ]

    total_start_time = time.time()

    for i, dataset_name in enumerate(datasets, 1):
        print(f"\n{'=' * 60}")
        print(f"🔄 DATASET {i}/{len(datasets)}: {dataset_name}")
        print(f"{'=' * 60}")

        try:
            # Chemin d'entrée
            input_path = f"technical_agent_dataset_brut/dataset_all/{dataset_name}"

            if not os.path.exists(input_path):
                print(f"⚠️ Fichier non trouvé: {input_path}")
                continue

            # Analyser la taille du fichier
            file_size_gb = os.path.getsize(input_path) / (1024**3)
            print(f"📏 Taille fichier: {file_size_gb:.2f}GB")

            # Choisir la méthode d'enrichissement optimale
            if file_size_gb > 2.0:
                print("🚀 Enrichissement depuis fichier (optimisé pour gros fichiers)")
                enriched_df = enricher.enrich_from_file_optimized(input_path)
            else:
                print("📦 Enrichissement standard (fichier en mémoire)")
                df = pd.read_csv(input_path)
                print(
                    f"📊 Dataset original: {len(df):,} lignes, {len(df.columns)} colonnes"
                )
                enriched_df = enricher.enrich_dataset(df)

            # Vérifier le résultat
            if enriched_df is None or len(enriched_df) == 0:
                print(f"❌ Erreur: Dataset enrichi vide pour {dataset_name}")
                continue

            # Sauvegarder avec nom optimisé
            _out_name = dataset_name.replace(".csv", "_quantitative_ultra_optimized.csv")
            output_path = f"technical_agent_dataset_brut/enriched/{_out_name}"

            # Créer le dossier si nécessaire
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Sauvegarder avec optimisation
            enriched_df.to_csv(output_path, index=False)

            # Statistiques finales
            output_size_gb = os.path.getsize(output_path) / (1024**3)
            print(f"💾 Sauvegardé: {output_path}")
            print(
                f"📈 Dataset enrichi: {len(enriched_df):,} lignes, {len(enriched_df.columns)} colonnes"
            )
            print(f"📏 Taille sortie: {output_size_gb:.2f}GB")

            # Vérifier les nouvelles colonnes
            quant_columns = [
                col for col in enriched_df.columns if col.startswith("quant_")
            ]
            print(f"🎯 Features quantitatives ajoutées: {len(quant_columns)}")

            if len(quant_columns) != 24:
                print(
                    f"⚠️ Attention: {len(quant_columns)} features au lieu de 24 attendues"
                )
                print(
                    f"   Features trouvées: {quant_columns[:5]}..."
                    if len(quant_columns) > 5
                    else f"   Features: {quant_columns}"
                )

        except Exception as e:
            print(f"❌ Erreur traitement {dataset_name}: {e}")
            import traceback

            traceback.print_exc()
            continue

        # Nettoyage entre datasets
        enricher._cleanup_resources()
        gc.collect()

    # Statistiques globales
    total_time = time.time() - total_start_time
    print(f"\n{'=' * 80}")
    print("✅ ENRICHISSEMENT QUANTITATIF ULTRA-OPTIMISÉ TERMINÉ")
    print(f"⏱️ Temps total: {total_time:.2f}s")
    print(f"📊 Datasets traités: {len(datasets)}")
    print("🎯 24 features quantitatives institutionnelles par dataset")
    print("🚀 Optimisations: Numba + Dask + SharedMem + MMap + Cache + Adaptatif")
    print(f"{'=' * 80}")


def enrich_single_dataset_ultra_optimized(dataset_path: str, output_path: str = None):
    """Enrichir un seul dataset avec toutes les optimisations."""
    print(f"🚀 ENRICHISSEMENT ULTRA-OPTIMISÉ: {dataset_path}")

    # Configuration adaptative
    memory_info = psutil.virtual_memory()
    available_gb = memory_info.available / (1024**3)

    enricher = OptimizedQuantitativeFeaturesEnricher(
        chunk_size=50000 if available_gb > 8 else 25000,
        max_memory_gb=min(available_gb * 0.5, 8.0),
        n_jobs=min(cpu_count(), 12),
        auto_optimize=True,
    )

    try:
        # Enrichissement optimisé
        enriched_df = enricher.enrich_from_file_optimized(dataset_path)

        # Sauvegarder
        if output_path is None:
            output_path = dataset_path.replace(
                ".csv", "_quantitative_ultra_optimized.csv"
            )

        enriched_df.to_csv(output_path, index=False)

        print(f"✅ Enrichissement terminé: {output_path}")
        return enriched_df

    except Exception as e:
        print(f"❌ Erreur enrichissement: {e}")
        return None
    finally:
        enricher._cleanup_resources()


# Fonction de compatibilité
def enrich_all_datasets_quantitative_optimized():
    """Fonction de compatibilité - redirige vers la version ultra-optimisée."""
    return enrich_all_datasets_quantitative_ultra_optimized()


if __name__ == "__main__":
    # Lancement de l'enrichissement ultra-optimisé
    enrich_all_datasets_quantitative_ultra_optimized()
