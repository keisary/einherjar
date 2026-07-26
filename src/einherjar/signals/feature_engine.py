"""FeatureEngine — Calcul des features sur OHLCV via les modules MIDAS portes.

Architecture d'inference :
- Chaque feature a une fenetre de lookback
- A chaque nouvelle bougie, on recalcule UNIQUEMENT sur la fenetre necessaire
- Polars pour le DataFrame, numpy pour les fonctions numba

Le bridge `midas_bridge.py` isole EINHERJAR de la complexite interne des
enrichisseurs MIDAS (Dask, multiprocessing, etc.).

Reference : Section 2.2 du CDC EINHERJAR.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from einherjar.signals.midas_bridge import IndicatorBridge, PatternBridge, QuantBridge

# ============================================================================
# CONSTANTES : TOUTES LES FEATURES (107 patterns + 52 indicateurs + 24 quant)
# ============================================================================

# Les 107 patterns de PATTERN_THRESHOLDS
ALL_PATTERN_NAMES = [
    "hammer", "inverted_hammer", "dragonfly_doji", "morning_star", "piercing_line",
    "three_white_soldiers", "engulfing_bull", "engulfing_bear", "harami_bull",
    "pin_bar_bull", "pin_bar_bear", "marubozu_bull", "abandoned_baby_bull",
    "three_inside_up", "three_outside_up", "concealing_baby_swallow",
    "unique_three_river_bottom", "belt_hold_bull", "kicking_bull", "matching_low",
    "ladder_bottom", "breakaway_bull", "hanging_man", "shooting_star",
    "gravestone_doji", "evening_star", "dark_cloud_cover", "three_black_crows",
    "harami_bear", "marubozu_bear", "abandoned_baby_bear", "three_inside_down",
    "three_outside_down", "advance_block", "deliberation", "belt_hold_bear",
    "kicking_bear", "matching_high", "ladder_top", "breakaway_bear", "doji",
    "long_legged_doji", "spinning_top", "four_price_doji", "rickshaw_man",
    "high_wave_candle", "tri_star", "double_top", "double_bottom", "triple_top",
    "triple_bottom", "head_shoulders", "inv_head_shoulders", "rounding_top",
    "rounding_bottom", "diamond_top", "diamond_bottom", "v_top", "v_bottom",
    "island_top", "island_bottom", "spike_reversal", "ascending_triangle",
    "descending_triangle", "symmetrical_triangle", "rectangle", "bull_flag",
    "bear_flag", "bull_pennant", "bear_pennant", "rising_wedge", "falling_wedge",
    "broadening_wedge", "cup_handle", "channel_up", "channel_down",
    "measured_move", "gartley_bull", "gartley_bear", "butterfly_bull",
    "butterfly_bear", "bat_bull", "bat_bear", "crab_bull", "crab_bear",
    "shark_bull", "shark_bear", "support", "resistance", "uptrend",
    "downtrend", "sideways_trend", "gap_up", "gap_down", "gap_fill",
    "breakaway_gap", "runaway_gap", "exhaustion_gap", "island_reversal",
    "gap_and_go", "three_drives", "wolfe_wave", "elliott_wave_1",
    "elliott_wave_3", "elliott_wave_5", "fibonacci_retracement",
    "fibonacci_extension",
]

# Lookback par feature (en nombre de bougies)
LOOKBACK_WINDOWS = {
    # === PATTERNS : fenetre contextuelle adaptee au type ===
    "hammer": 10, "inverted_hammer": 10, "dragonfly_doji": 10,
    "morning_star": 15, "piercing_line": 15, "three_white_soldiers": 15,
    "engulfing_bull": 10, "engulfing_bear": 10, "harami_bull": 10,
    "pin_bar_bull": 10, "pin_bar_bear": 10, "marubozu_bull": 10,
    "abandoned_baby_bull": 15, "three_inside_up": 15, "three_outside_up": 15,
    "concealing_baby_swallow": 20, "unique_three_river_bottom": 15,
    "belt_hold_bull": 10, "kicking_bull": 15, "matching_low": 10,
    "ladder_bottom": 25, "breakaway_bull": 25, "hanging_man": 10,
    "shooting_star": 10, "gravestone_doji": 10, "evening_star": 15,
    "dark_cloud_cover": 15, "three_black_crows": 15, "harami_bear": 10,
    "marubozu_bear": 10, "abandoned_baby_bear": 15, "three_inside_down": 15,
    "three_outside_down": 15, "advance_block": 15, "deliberation": 15,
    "belt_hold_bear": 10, "kicking_bear": 15, "matching_high": 10,
    "ladder_top": 25, "breakaway_bear": 25, "doji": 10,
    "long_legged_doji": 10, "spinning_top": 10, "four_price_doji": 10,
    "rickshaw_man": 10, "high_wave_candle": 10, "tri_star": 15,
    # Chartistes retournement
    "double_top": 100, "double_bottom": 100, "triple_top": 80,
    "triple_bottom": 80, "head_shoulders": 80, "inv_head_shoulders": 80,
    "rounding_top": 100, "rounding_bottom": 100, "diamond_top": 60,
    "diamond_bottom": 60, "v_top": 20, "v_bottom": 20,
    "island_top": 30, "island_bottom": 30, "spike_reversal": 20,
    # Chartistes continuation
    "ascending_triangle": 60, "descending_triangle": 60,
    "symmetrical_triangle": 60, "rectangle": 40, "bull_flag": 50,
    "bear_flag": 50, "bull_pennant": 40, "bear_pennant": 40,
    "rising_wedge": 50, "falling_wedge": 50, "broadening_wedge": 50,
    "cup_handle": 80, "channel_up": 40, "channel_down": 40,
    "measured_move": 40,
    # Harmoniques
    "gartley_bull": 100, "gartley_bear": 100, "butterfly_bull": 100,
    "butterfly_bear": 100, "bat_bull": 100, "bat_bear": 100,
    "crab_bull": 100, "crab_bear": 100, "shark_bull": 100, "shark_bear": 100,
    # Support/resistance/trend
    "support": 50, "resistance": 50, "uptrend": 30, "downtrend": 30,
    "sideways_trend": 30,
    # Gaps
    "gap_up": 10, "gap_down": 10, "gap_fill": 20, "breakaway_gap": 20,
    "runaway_gap": 20, "exhaustion_gap": 20, "island_reversal": 30,
    "gap_and_go": 15, "three_drives": 60, "wolfe_wave": 60,
    # Elliott / Fibonacci
    "elliott_wave_1": 80, "elliott_wave_3": 80, "elliott_wave_5": 80,
    "fibonacci_retracement": 50, "fibonacci_extension": 50,
    # === INDICATEURS (52) ===
    "rsi_14": 14, "rsi_21": 21, "rsi_30": 30,
    "macd_line": 26, "macd_signal": 26, "macd_histogram": 26,
    "ema_9": 9, "ema_12": 12, "ema_21": 21, "ema_26": 26,
    "ema_50": 50, "ema_100": 100, "ema_200": 200,
    "sma_20": 20, "sma_50": 50, "sma_100": 100, "sma_200": 200,
    "bb_upper": 20, "bb_middle": 20, "bb_lower": 20, "bb_width": 20, "bb_percent": 20,
    "volume_sma_20": 20, "volume_ratio": 20,
    "atr_14": 14, "atr_21": 21,
    "stoch_k": 14, "stoch_d": 14,
    "williams_r": 14, "cci_20": 20,
    "adx_14": 14, "di_plus": 14, "di_minus": 14,
    "obv": 1, "obv_ema": 20,
    "vwap": 1,
    "momentum_10": 10, "momentum_20": 20,
    "roc_10": 10, "roc_20": 20,
    "trix_14": 14,
    "ultimate_oscillator": 28,
    "money_flow_index": 14,
    "chaikin_oscillator": 10,
    "aroon_up": 14, "aroon_down": 14,
    "parabolic_sar": 2,
    "supertrend": 10, "supertrend_signal": 10,
    "choppiness_index": 14,
    "vortex_pos": 14, "vortex_neg": 14,
    # === FEATURES QUANT (24) ===
    "realized_volatility": 20,
    "garch_volatility": 20,
    "volatility_clustering": 100,
    "hurst_exponent": 100,
    "autocorrelation": 100,
    "sample_entropy": 50,
    "shannon_entropy": 50,
    "dominant_frequency": 100,
    "spectral_centroid": 100,
    "fractal_dimension": 64,
    "dfa_exponent": 64,
    "volatility_persistence": 100,
    "approximate_entropy": 50,
    "permutation_entropy": 100,
    "rolling_skewness": 50,
    "rolling_kurtosis": 50,
    "dynamic_var": 50,
    "dynamic_cvar": 50,
    "max_drawdown": 100,
    "regime_detection": 50,
    "amihud_illiquidity": 20,
    "kyles_lambda": 20,
    "kaufman_efficiency": 20,
    "variance_ratio": 20,
}

MAX_LOOKBACK = max(LOOKBACK_WINDOWS.values())


class FeatureEngine:
    """Moteur de calcul des features sur OHLCV.

    Attributs:
        max_lookback: Nombre maximal de bougies historiques chargees pour le recalcul.
        pattern_bridge: Instance PatternBridge pour la detection de patterns.
        indicator_bridge: Instance IndicatorBridge pour les indicateurs techniques.
        quant_bridge: Instance QuantBridge pour les features quantitatives.
    """

    def __init__(self, max_lookback: int = MAX_LOOKBACK) -> None:
        """Initialise le moteur de features.

        Args:
            max_lookback: Fenetre historique max a charger. Defaut = MAX_LOOKBACK.
        """
        self.max_lookback = max_lookback
        self.pattern_bridge = PatternBridge()
        self.indicator_bridge = IndicatorBridge()
        self.quant_bridge = QuantBridge()

    def _prepare_numpy(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        """Convertit un DataFrame polars en dict numpy pour les fonctions numba."""
        return {
            "open": df["open"].to_numpy().astype(np.float32),
            "high": df["high"].to_numpy().astype(np.float32),
            "low": df["low"].to_numpy().astype(np.float32),
            "close": df["close"].to_numpy().astype(np.float32),
            "volume": df["volume"].to_numpy().astype(np.float32),
        }

    def _compute_indicators(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Calcule les 52 indicateurs techniques via le bridge."""
        o = data["open"]
        h = data["high"]
        l = data["low"]
        c = data["close"]
        v = data["volume"]

        features: dict[str, np.ndarray] = {}

        # --- RSI multi-periodes ---
        rsi_multi = self.indicator_bridge.rsi_multi(c, [14, 21, 30])
        features.update(rsi_multi)

        # --- MACD ---
        features.update(self.indicator_bridge.macd(c, 12, 26, 9))

        # --- EMA multi-periodes ---
        ema_multi = self.indicator_bridge.ema_multi(c, [9, 12, 21, 26, 50, 100, 200])
        features.update(ema_multi)

        # --- SMA multi-periodes ---
        sma_multi = self.indicator_bridge.sma_multi(c, [20, 50, 100, 200])
        features.update(sma_multi)

        # --- Bollinger Bands + derivees ---
        features.update(self.indicator_bridge.bollinger(c, 20, 2.0))

        # --- Volume SMA + ratio ---
        vol_sma = self.indicator_bridge.volume_sma(v, 20)
        features["volume_sma_20"] = vol_sma
        features["volume_ratio"] = np.where(vol_sma > 0, v / vol_sma, 1.0).astype(np.float32)

        # --- ATR ---
        features["atr_14"] = self.indicator_bridge.atr(h, l, c, 14)
        features["atr_21"] = self.indicator_bridge.atr(h, l, c, 21)

        # --- Stochastic ---
        features.update(self.indicator_bridge.stochastic(h, l, c, 14, 3))

        # --- Williams %R ---
        features["williams_r"] = self.indicator_bridge.williams_r(h, l, c, 14)

        # --- CCI ---
        features["cci_20"] = self.indicator_bridge.cci(h, l, c, 20)

        # --- ADX ---
        features.update(self.indicator_bridge.adx(h, l, c, 14))

        # --- OBV + OBV EMA ---
        obv = self.indicator_bridge.obv(c, v)
        features["obv"] = obv
        features["obv_ema"] = self.indicator_bridge.ema(obv, 20)

        # --- VWAP ---
        features["vwap"] = self.indicator_bridge.vwap(c, v)

        # --- Momentum ---
        features["momentum_10"] = self.indicator_bridge.momentum(c, 10)
        features["momentum_20"] = self.indicator_bridge.momentum(c, 20)

        # --- ROC ---
        features["roc_10"] = self.indicator_bridge.roc(c, 10)
        features["roc_20"] = self.indicator_bridge.roc(c, 20)

        # --- TRIX ---
        features["trix_14"] = self.indicator_bridge.trix(c, 14)

        # --- Ultimate Oscillator ---
        features["ultimate_oscillator"] = self.indicator_bridge.ultimate_oscillator(h, l, c)

        # --- Money Flow Index ---
        features["money_flow_index"] = self.indicator_bridge.mfi(h, l, c, v, 14)

        # --- Chaikin Oscillator ---
        features["chaikin_oscillator"] = self.indicator_bridge.chaikin_oscillator(h, l, c, v, 3, 10)

        # --- Aroon ---
        features.update(self.indicator_bridge.aroon(h, l, 14))

        # --- Parabolic SAR ---
        features["parabolic_sar"] = self.indicator_bridge.parabolic_sar(h, l)

        # --- SuperTrend ---
        features.update(self.indicator_bridge.supertrend(h, l, c, 10, 3.0))

        # --- Choppiness Index ---
        features["choppiness_index"] = self.indicator_bridge.choppiness_index(h, l, c, 14)

        # --- Vortex ---
        features.update(self.indicator_bridge.vortex(h, l, c, 14))

        return features

    def _compute_quant(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Calcule les 24 features quantitatives via le bridge."""
        c = data["close"]
        v = data["volume"]

        # Rendements
        returns = np.diff(c) / c[:-1]
        returns = np.concatenate(([0.0], returns)).astype(np.float32)

        features: dict[str, np.ndarray] = {}

        # --- Volatilite ---
        features["realized_volatility"] = self.quant_bridge.realized_vol(returns, 20)
        features["garch_volatility"] = self.quant_bridge.garch_vol(returns, 0.1, 0.8)
        features["volatility_clustering"] = self.quant_bridge.vol_clustering(returns, 100)
        features["volatility_persistence"] = self.quant_bridge.vol_persistence(returns, 100)

        # --- Momentum & Persistence ---
        features["hurst_exponent"] = self.quant_bridge.hurst(c, 100)
        features["autocorrelation"] = self.quant_bridge.autocorr(c, 10, 100)

        # --- Entropie ---
        features["shannon_entropy"] = self.quant_bridge.shannon_entropy(c, 50)
        features["sample_entropy"] = self.quant_bridge.sample_entropy(c, 50)
        features["approximate_entropy"] = self.quant_bridge.approx_entropy(c, 50)
        features["permutation_entropy"] = self.quant_bridge.permutation_entropy(c, 100)

        # --- Spectral ---
        features["dominant_frequency"] = self.quant_bridge.dominant_freq(c, 100)
        features["spectral_centroid"] = self.quant_bridge.spectral_centroid(c, 100)

        # --- Fractales ---
        features["fractal_dimension"] = self.quant_bridge.fractal_dimension(c, 64)
        features["dfa_exponent"] = self.quant_bridge.dfa(c, 64)

        # --- Statistiques roulantes ---
        features["rolling_skewness"] = self.quant_bridge.rolling_skewness(c, 50)
        features["rolling_kurtosis"] = self.quant_bridge.rolling_kurtosis(c, 50)

        # --- Risk ---
        features["dynamic_var"] = self.quant_bridge.dynamic_var(returns, 0.05, 50)
        features["dynamic_cvar"] = self.quant_bridge.dynamic_cvar(returns, 0.05, 50)
        features["max_drawdown"] = self.quant_bridge.max_drawdown(c, 100)
        features["regime_detection"] = self.quant_bridge.regime_detection(returns, 50)

        # --- Liquidite (numpy pur) ---
        features["amihud_illiquidity"] = self.quant_bridge.amihud_illiquidity(c, v, 20)
        features["kyles_lambda"] = self.quant_bridge.kyles_lambda(c, v, 20)

        # --- Efficience (numpy pur) ---
        features["kaufman_efficiency"] = self.quant_bridge.kaufman_efficiency(c, 20)
        features["variance_ratio"] = self.quant_bridge.variance_ratio(c, 10)

        return features

    def _compute_patterns(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        """Detecte les 107 patterns via le bridge."""
        return self.pattern_bridge.detect(df, ALL_PATTERN_NAMES)

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calcule toutes les features (183 total) sur le DataFrame OHLCV.

        Args:
            df: DataFrame OHLCV avec [timestamp, open, high, low, close, volume].

        Returns:
            DataFrame enrichi avec les 183 colonnes features.
        """
        if len(df) == 0:
            return df

        data = self._prepare_numpy(df)

        # Indicateurs techniques (52)
        indicator_features = self._compute_indicators(data)

        # Features quantitatives (24)
        quant_features = self._compute_quant(data)

        # Patterns (107)
        pattern_features = self._compute_patterns(df)

        # Assembler
        all_features: dict[str, pl.Series] = {}

        for name, arr in {**indicator_features, **quant_features}.items():
            all_features[name] = pl.Series(name, arr)

        for name, arr in pattern_features.items():
            all_features[name] = pl.Series(name, arr.astype(np.float32))

        enriched = df
        for name, series in all_features.items():
            enriched = enriched.with_columns(series)

        return enriched

    def compute_incremental(
        self,
        df_history: pl.DataFrame,
        new_candle: dict[str, Any],
    ) -> pl.DataFrame:
        """Recalcul cible sur la derniere bougie uniquement.

        Appelee a chaque cloture de bougie en live. Charge la fenetre
        necessaire, recalcule les features, retourne la nouvelle ligne enrichie.

        Args:
            df_history: Historique OHLCV.
            new_candle: Dict {timestamp, open, high, low, close, volume}.

        Returns:
            DataFrame avec la nouvelle bougie + features recalculees.
        """
        new_row = pl.DataFrame([new_candle])
        df = pl.concat([df_history, new_row], how="vertical")

        if len(df) > self.max_lookback:
            df = df.tail(self.max_lookback)

        return self.compute(df)

    def get_required_lookback(self, feature_name: str) -> int:
        """Retourne la fenetre necessaire pour une feature."""
        return LOOKBACK_WINDOWS.get(feature_name, self.max_lookback)
