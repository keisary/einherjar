"""MIDASBridge — Interface unifiee vers les fonctions numba des modules MIDAS.

Ce module isole EINHERJAR de la complexite interne des enrichisseurs MIDAS
(batch, Dask, multiprocessing, shared memory). Il expose uniquement des
fonctions pures numpy -> numpy utilisables par le FeatureEngine.

Reference : Phase 2 du CDC EINHERJAR.
"""

from __future__ import annotations

import logging

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

# ============================================================================
# Indicateurs techniques (fonctions numba pures)
# ============================================================================
# ============================================================================
# Patterns (classe detectrice)
# ============================================================================
from einherjar.signals.numba_pattern_detectors import (  # noqa: E402
    PATTERN_THRESHOLDS,
    NumbaPatternDetectors,
    PatternMetadataManager,
)
from einherjar.signals.technical_indicators import (  # noqa: E402
    _numba_adx_complete,
    _numba_aroon,
    _numba_atr,
    _numba_bollinger_bands,
    _numba_cci,
    _numba_chaikin_oscillator,
    _numba_choppiness_index,
    _numba_cmo,
    _numba_donchian_channels,
    _numba_ema_single,
    _numba_ema_vectorized,
    _numba_ichimoku_complete,
    _numba_keltner_channels,
    _numba_macd_complete,
    _numba_mfi,
    _numba_momentum,
    _numba_obv,
    _numba_parabolic_sar,
    _numba_roc,
    _numba_rsi_vectorized,
    _numba_sma_vectorized,
    _numba_stochastic,
    _numba_supertrend,
    _numba_trix,
    _numba_ultimate_oscillator,
    _numba_vortex,
    _numba_vwap,
    _numba_williams_r,
)

# ============================================================================
# Features quantitatives (fonctions numba pures, import conditionnel)
# ============================================================================
try:
    from einherjar.signals.quantitative_features import (
        _numba_autocorrelation,
        _numba_dfa,
        _numba_dominant_frequency,
        _numba_dynamic_cvar,
        _numba_dynamic_var,
        _numba_fractal_dimension,
        _numba_garch_volatility,
        _numba_hurst_rs,
        _numba_max_drawdown,
        _numba_permutation_entropy,
        _numba_realized_volatility,
        _numba_regime_detection,
        _numba_rolling_kurtosis,
        _numba_rolling_skewness,
        _numba_sample_entropy,
        _numba_shannon_entropy,
        _numba_spectral_centroid,
        _numba_volatility_clustering,
        _numba_volatility_persistence,
    )

    QUANT_AVAILABLE = True
except ImportError as exc:
    logger.warning("Features quantitatives non disponibles: %s", exc)
    QUANT_AVAILABLE = False

    def _numba_realized_volatility(returns, window=20):
        return np.zeros(len(returns), dtype=np.float32)

    def _numba_garch_volatility(returns, alpha=0.1, beta=0.8):
        return np.zeros(len(returns), dtype=np.float32)

    def _numba_volatility_clustering(returns, threshold_factor=2.0, window=100):
        return np.zeros(len(returns), dtype=np.float32)

    def _numba_hurst_rs(prices, window=252):
        return np.full(len(prices), 0.5, dtype=np.float32)

    def _numba_autocorrelation(prices, max_lag=20, window=252):
        return np.zeros(len(prices), dtype=np.float32)

    def _numba_shannon_entropy(prices, bins=50, window=252):
        return np.zeros(len(prices), dtype=np.float32)

    def _numba_sample_entropy(prices, m=2, r=0.2, window=252):
        return np.zeros(len(prices), dtype=np.float32)

    def _numba_fractal_dimension(prices, window=252):
        return np.ones(len(prices), dtype=np.float32)

    def _numba_dfa(prices, window=252):
        return np.full(len(prices), 0.5, dtype=np.float32)

    def _numba_rolling_skewness(prices, window=50):
        return np.zeros(len(prices), dtype=np.float32)

    def _numba_rolling_kurtosis(prices, window=50):
        return np.full(len(prices), 3.0, dtype=np.float32)

    def _numba_dynamic_var(returns, confidence=0.05, window=50):
        return np.zeros(len(returns), dtype=np.float32)

    def _numba_dynamic_cvar(returns, confidence=0.05, window=50):
        return np.zeros(len(returns), dtype=np.float32)

    def _numba_max_drawdown(prices, window=100):
        return np.zeros(len(prices), dtype=np.float32)

    def _numba_regime_detection(returns, lookback=50):
        return np.zeros(len(returns), dtype=np.float32)

    def _numba_dominant_frequency(prices, window=252):
        return np.zeros(len(prices), dtype=np.float32)

    def _numba_spectral_centroid(prices, window=252):
        return np.zeros(len(prices), dtype=np.float32)

    def _numba_permutation_entropy(prices, order=3, delay=1, window=100):
        return np.zeros(len(prices), dtype=np.float32)

    def _numba_volatility_persistence(returns, window=100):
        return np.zeros(len(returns), dtype=np.float32)


class PatternBridge:
    """Wrapper simplifie autour de NumbaPatternDetectors.

    Attributs:
        detector: Instance NumbaPatternDetectors.
    """

    def __init__(self) -> None:
        """Initialise le detecteur."""
        metadata = PatternMetadataManager(PATTERN_THRESHOLDS)
        self.detector = NumbaPatternDetectors(metadata)

    def detect(
        self,
        df: pl.DataFrame,
        patterns: list[str] | None = None,
    ) -> dict[str, np.ndarray]:
        """Detecte les patterns sur un DataFrame polars.

        Args:
            df: DataFrame OHLCV polars.
            patterns: Liste de patterns a detecter. None = tous.

        Returns:
            Dict {pattern_name: array_detection (0 ou 1)}.
        """
        if patterns is None:
            patterns = list(PATTERN_THRESHOLDS.keys())
        return self.detector.detect(df, patterns)

    @staticmethod
    def structure_levels(
        df: pl.DataFrame,
        direction: str,
        lookback: int = 20,
    ) -> dict[str, float]:
        """Calcule les niveaux de structure disponibles pour les exits.

        Les detecteurs booleens ne retournent pas encore leurs points pivots.
        Cette base explicite fournit des niveaux reproductibles a partir des
        bougies du pattern, sans degrader les sorties en multiples ATR.
        """
        window = df.tail(max(2, lookback))
        high = window["high"].to_numpy()
        low = window["low"].to_numpy()
        close = window["close"].to_numpy()
        upper = float(np.nanmax(high))
        lower = float(np.nanmin(low))
        height = max(upper - lower, 0.0)
        entry = float(close[-1])
        if direction == "short":
            target_382 = entry - height * 0.382
            target_618 = entry - height * 0.618
            invalidation = upper
        else:
            target_382 = entry + height * 0.382
            target_618 = entry + height * 0.618
            invalidation = lower
        return {
            "range_high": upper,
            "range_low": lower,
            "pattern_height": height,
            "invalidation": invalidation,
            "fib_382": target_382,
            "fib_618": target_618,
        }


class IndicatorBridge:
    """Bridge vers les fonctions numba d'indicateurs techniques.

    Chaque methode prend les arrays numpy OHLCV et retourne un ou plusieurs
    arrays numpy. Aucune dependance polars ici.
    """

    @staticmethod
    def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """ATR(period)."""
        return _numba_atr(high, low, close, period)

    @staticmethod
    def rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
        """RSI(period) via vectorise."""
        periods = np.array([period], dtype=np.int32)
        result_2d = _numba_rsi_vectorized(close, periods)
        return result_2d[0] if result_2d.ndim == 2 else result_2d

    @staticmethod
    def rsi_multi(close: np.ndarray, periods: list[int]) -> dict[str, np.ndarray]:
        """RSI multi-periodes en une passe."""
        periods_arr = np.array(periods, dtype=np.int32)
        result_2d = _numba_rsi_vectorized(close, periods_arr)
        return {f"rsi_{p}": result_2d[i] for i, p in enumerate(periods)}

    @staticmethod
    def sma(close: np.ndarray, period: int = 20) -> np.ndarray:
        """SMA(period) via vectorise."""
        periods = np.array([period], dtype=np.int32)
        result_2d = _numba_sma_vectorized(close, periods)
        return result_2d[0] if result_2d.ndim == 2 else result_2d

    @staticmethod
    def sma_multi(close: np.ndarray, periods: list[int]) -> dict[str, np.ndarray]:
        """SMA multi-periodes en une passe."""
        periods_arr = np.array(periods, dtype=np.int32)
        result_2d = _numba_sma_vectorized(close, periods_arr)
        return {f"sma_{p}": result_2d[i] for i, p in enumerate(periods)}

    @staticmethod
    def ema(close: np.ndarray, period: int = 20) -> np.ndarray:
        """EMA(period)."""
        return _numba_ema_single(close, period)

    @staticmethod
    def ema_multi(close: np.ndarray, periods: list[int]) -> dict[str, np.ndarray]:
        """EMA multi-periodes en une passe via vectorise."""
        periods_arr = np.array(periods, dtype=np.int32)
        result_2d = _numba_ema_vectorized(close, periods_arr)
        return {f"ema_{p}": result_2d[i] for i, p in enumerate(periods)}

    @staticmethod
    def macd(close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> dict[str, np.ndarray]:
        """MACD complet."""
        macd_line, signal_line, histogram = _numba_macd_complete(close, fast, slow, signal)
        return {
            "macd_line": macd_line,
            "macd_signal": signal_line,
            "macd_histogram": histogram,
        }

    @staticmethod
    def bollinger(close: np.ndarray, period: int = 20, std_dev: float = 2.0) -> dict[str, np.ndarray]:
        """Bollinger Bands + derivees."""
        upper, middle, lower = _numba_bollinger_bands(close, period, std_dev)
        bb_width = np.where(middle != 0, (upper - lower) / middle, 0.0).astype(np.float32)
        bb_percent = np.where(upper != lower, (close - lower) / (upper - lower), 0.5).astype(np.float32)
        return {
            "bb_upper": upper,
            "bb_middle": middle,
            "bb_lower": lower,
            "bb_width": bb_width,
            "bb_percent": bb_percent,
        }

    @staticmethod
    def stochastic(high: np.ndarray, low: np.ndarray, close: np.ndarray, k_period: int = 14, d_period: int = 3) -> dict[str, np.ndarray]:
        """Stochastic Oscillator."""
        k, d = _numba_stochastic(high, low, close, k_period, d_period)
        return {"stoch_k": k, "stoch_d": d}

    @staticmethod
    def adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> dict[str, np.ndarray]:
        """ADX complet avec DI+ et DI-."""
        adx_arr, di_plus, di_minus = _numba_adx_complete(high, low, close, period)
        return {"adx_14": adx_arr, "di_plus": di_plus, "di_minus": di_minus}

    @staticmethod
    def obv(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """On Balance Volume."""
        return _numba_obv(close, volume)

    @staticmethod
    def mfi(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray, period: int = 14) -> np.ndarray:
        """Money Flow Index."""
        return _numba_mfi(high, low, close, volume, period)

    @staticmethod
    def williams_r(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """Williams %R."""
        return _numba_williams_r(high, low, close, period)

    @staticmethod
    def cci(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 20) -> np.ndarray:
        """Commodity Channel Index."""
        return _numba_cci(high, low, close, period)

    @staticmethod
    def trix(close: np.ndarray, period: int = 14) -> np.ndarray:
        """TRIX."""
        return _numba_trix(close, period)

    @staticmethod
    def ultimate_oscillator(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """Ultimate Oscillator."""
        return _numba_ultimate_oscillator(high, low, close, 7, 14, 28)

    @staticmethod
    def chaikin_oscillator(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray, fast: int = 3, slow: int = 10) -> np.ndarray:
        """Chaikin Oscillator."""
        return _numba_chaikin_oscillator(high, low, close, volume, fast, slow)

    @staticmethod
    def aroon(high: np.ndarray, low: np.ndarray, period: int = 14) -> dict[str, np.ndarray]:
        """Aroon Up/Down."""
        up, down = _numba_aroon(high, low, period)
        return {"aroon_up": up, "aroon_down": down}

    @staticmethod
    def parabolic_sar(high: np.ndarray, low: np.ndarray) -> np.ndarray:
        """Parabolic SAR."""
        return _numba_parabolic_sar(high, low, 0.02, 0.02, 0.2)

    @staticmethod
    def supertrend(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 10, multiplier: float = 3.0) -> dict[str, np.ndarray]:
        """SuperTrend."""
        st, trend = _numba_supertrend(high, low, close, period, multiplier)
        return {"supertrend": st, "supertrend_signal": trend.astype(np.float32)}

    @staticmethod
    def choppiness_index(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """Choppiness Index."""
        return _numba_choppiness_index(high, low, close, period)

    @staticmethod
    def vortex(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> dict[str, np.ndarray]:
        """Vortex Indicator."""
        vip, vim = _numba_vortex(high, low, close, period)
        return {"vortex_pos": vip, "vortex_neg": vim}

    @staticmethod
    def vwap(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """VWAP."""
        return _numba_vwap(close, volume)

    @staticmethod
    def cmo(close: np.ndarray, period: int = 14) -> np.ndarray:
        """Chande Momentum Oscillator."""
        return _numba_cmo(close, period)

    @staticmethod
    def momentum(close: np.ndarray, period: int = 10) -> np.ndarray:
        """Momentum."""
        return _numba_momentum(close, period)

    @staticmethod
    def roc(close: np.ndarray, period: int = 12) -> np.ndarray:
        """Rate of Change."""
        return _numba_roc(close, period)

    @staticmethod
    def ichimoku(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> dict[str, np.ndarray]:
        """Ichimoku complet."""
        tenkan, kijun, senkou_a, senkou_b, chikou = _numba_ichimoku_complete(high, low, close, 9, 26, 52, 26)
        return {
            "ichimoku_tenkan": tenkan,
            "ichimoku_kijun": kijun,
            "ichimoku_senkou_a": senkou_a,
            "ichimoku_senkou_b": senkou_b,
            "ichimoku_chikou": chikou,
        }

    @staticmethod
    def keltner(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 20, multiplier: float = 2.0) -> dict[str, np.ndarray]:
        """Keltner Channels."""
        upper, middle, lower = _numba_keltner_channels(high, low, close, period, multiplier)
        return {"keltner_upper": upper, "keltner_middle": middle, "keltner_lower": lower}

    @staticmethod
    def donchian(high: np.ndarray, low: np.ndarray, period: int = 20) -> dict[str, np.ndarray]:
        """Donchian Channels."""
        upper, middle, lower = _numba_donchian_channels(high, low, period)
        return {"donchian_upper": upper, "donchian_middle": middle, "donchian_lower": lower}

    @staticmethod
    def volume_sma(volume: np.ndarray, period: int = 20) -> np.ndarray:
        """SMA du volume."""
        periods = np.array([period], dtype=np.int32)
        result_2d = _numba_sma_vectorized(volume, periods)
        return result_2d[0] if result_2d.ndim == 2 else result_2d


class QuantBridge:
    """Bridge vers les fonctions numba de features quantitatives.

    Chaque methode prend un array numpy et retourne un array.
    """

    @staticmethod
    def realized_vol(returns: np.ndarray, window: int = 20) -> np.ndarray:
        """Realized Volatility."""
        return _numba_realized_volatility(returns, window)

    @staticmethod
    def garch_vol(returns: np.ndarray, alpha: float = 0.1, beta: float = 0.8) -> np.ndarray:
        """GARCH(1,1) volatility."""
        return _numba_garch_volatility(returns, alpha, beta)

    @staticmethod
    def vol_clustering(returns: np.ndarray, window: int = 100) -> np.ndarray:
        """Volatility Clustering."""
        return _numba_volatility_clustering(returns, 2.0, window)

    @staticmethod
    def vol_persistence(returns: np.ndarray, window: int = 100) -> np.ndarray:
        """Volatility Persistence."""
        return _numba_volatility_persistence(returns, window)

    @staticmethod
    def hurst(prices: np.ndarray, window: int = 100) -> np.ndarray:
        """Hurst Exponent (R/S analysis)."""
        return _numba_hurst_rs(prices, window)

    @staticmethod
    def autocorr(prices: np.ndarray, lag: int = 10, window: int = 100) -> np.ndarray:
        """Autocorrelation par lag."""
        return _numba_autocorrelation(prices, lag, window)

    @staticmethod
    def shannon_entropy(prices: np.ndarray, window: int = 50) -> np.ndarray:
        """Shannon Entropy."""
        return _numba_shannon_entropy(prices, bins=50, window=window)

    @staticmethod
    def sample_entropy(prices: np.ndarray, window: int = 50) -> np.ndarray:
        """Sample Entropy (approximation)."""
        return _numba_sample_entropy(prices, m=2, r=0.2, window=window)

    @staticmethod
    def approx_entropy(prices: np.ndarray, window: int = 50) -> np.ndarray:
        """Approximate Entropy."""
        return _numba_sample_entropy(prices, m=2, r=0.2, window=window)

    @staticmethod
    def permutation_entropy(prices: np.ndarray, window: int = 100) -> np.ndarray:
        """Permutation Entropy."""
        return _numba_permutation_entropy(prices, order=3, delay=1, window=window)

    @staticmethod
    def fractal_dimension(prices: np.ndarray, window: int = 64) -> np.ndarray:
        """Fractal Dimension (box-counting)."""
        return _numba_fractal_dimension(prices, window)

    @staticmethod
    def dfa(prices: np.ndarray, window: int = 64) -> np.ndarray:
        """Detrended Fluctuation Analysis."""
        return _numba_dfa(prices, window)

    @staticmethod
    def dominant_freq(prices: np.ndarray, window: int = 100) -> np.ndarray:
        """Dominant Frequency."""
        return _numba_dominant_frequency(prices, window)

    @staticmethod
    def spectral_centroid(prices: np.ndarray, window: int = 100) -> np.ndarray:
        """Spectral Centroid."""
        return _numba_spectral_centroid(prices, window)

    @staticmethod
    def rolling_skewness(prices: np.ndarray, window: int = 50) -> np.ndarray:
        """Skewness roulante."""
        return _numba_rolling_skewness(prices, window)

    @staticmethod
    def rolling_kurtosis(prices: np.ndarray, window: int = 50) -> np.ndarray:
        """Kurtosis roulante."""
        return _numba_rolling_kurtosis(prices, window)

    @staticmethod
    def dynamic_var(returns: np.ndarray, confidence: float = 0.05, window: int = 50) -> np.ndarray:
        """Value at Risk dynamique."""
        return _numba_dynamic_var(returns, confidence, window)

    @staticmethod
    def dynamic_cvar(returns: np.ndarray, confidence: float = 0.05, window: int = 50) -> np.ndarray:
        """Conditional VaR (Expected Shortfall)."""
        return _numba_dynamic_cvar(returns, confidence, window)

    @staticmethod
    def max_drawdown(prices: np.ndarray, window: int = 100) -> np.ndarray:
        """Maximum Drawdown roulant."""
        return _numba_max_drawdown(prices, window)

    @staticmethod
    def regime_detection(returns: np.ndarray, lookback: int = 50) -> np.ndarray:
        """Detection de regime (z-score adaptatif)."""
        return _numba_regime_detection(returns, lookback)

    @staticmethod
    def amihud_illiquidity(close: np.ndarray, volume: np.ndarray, window: int = 20) -> np.ndarray:
        """Amihud Illiquidity (numpy pur)."""
        n = len(close)
        result = np.zeros(n, dtype=np.float32)
        returns = np.abs(np.diff(close) / close[:-1])
        returns = np.concatenate(([0.0], returns))
        for i in range(window - 1, n):
            window_ret = returns[i - window + 1:i + 1]
            window_vol = volume[i - window + 1:i + 1]
            illiq = np.mean(window_ret / (window_vol + 1e-10))
            result[i] = illiq
        return result

    @staticmethod
    def kyles_lambda(close: np.ndarray, volume: np.ndarray, window: int = 20) -> np.ndarray:
        """Kyle's Lambda (numpy pur)."""
        n = len(close)
        result = np.zeros(n, dtype=np.float32)
        for i in range(window - 1, n):
            p = close[i - window + 1:i + 1]
            v = volume[i - window + 1:i + 1]
            dp = np.diff(p)
            dv = v[1:]
            cov = np.cov(dp, dv)[0, 1] if len(dp) > 1 else 0.0
            var_v = np.var(dv) if len(dv) > 1 else 1e-10
            result[i] = cov / var_v if var_v > 0 else 0.0
        return result

    @staticmethod
    def kaufman_efficiency(close: np.ndarray, window: int = 20) -> np.ndarray:
        """Kaufman Efficiency Ratio (numpy pur)."""
        n = len(close)
        result = np.zeros(n, dtype=np.float32)
        for i in range(window - 1, n):
            w = close[i - window + 1:i + 1]
            change = abs(w[-1] - w[0])
            volatility = np.sum(np.abs(np.diff(w)))
            result[i] = change / volatility if volatility > 0 else 0.0
        return result

    @staticmethod
    def variance_ratio(close: np.ndarray, lag: int = 10) -> np.ndarray:
        """Variance Ratio Test (numpy pur)."""
        n = len(close)
        result = np.zeros(n, dtype=np.float32)
        returns = np.diff(close) / close[:-1]
        returns = np.concatenate(([0.0], returns))
        for i in range(lag * 2, n):
            r1 = returns[i - lag + 1:i + 1]
            r_k = returns[i - lag * 2 + 1:i + 1]
            var_1 = np.var(r1) if len(r1) > 1 else 1e-10
            var_k = np.var(r_k) / lag if len(r_k) > lag else 1e-10
            result[i] = var_1 / var_k if var_k > 0 else 1.0
        return result
