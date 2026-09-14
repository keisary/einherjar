"""Configuration des facteurs alpha (copie de MIDAS `strategy_generator/config.py`).

Extrait verbatim : classes d'exceptions et `AlphaFactorConfig` (familles de features,
definition et resolution des facteurs alpha). Aucune dependance a MIDAS.
"""

from __future__ import annotations

from typing import Any


class MidasCorrelationError(Exception):
    """Exception de base pour le pipeline de découverte."""

    pass


class AlphaFactorError(MidasCorrelationError):
    """Erreur lors du calcul des facteurs alpha."""

    pass


class StrategyGenerationError(MidasCorrelationError):
    """Erreur lors de la génération de stratégies."""

    pass


class ValidationError(MidasCorrelationError):
    """Erreur lors de la validation des stratégies."""

    pass


class AlphaFactorConfig:
    """Configuration pour le calcul des facteurs alpha."""

    # Mapping des features brutes vers leurs familles
    FEATURE_FAMILIES = {
        # Famille MOMENTUM - Indicateurs de momentum et oscillateurs
        "MOMENTUM": [
            "rsi_14",
            "rsi_21",
            "rsi_30",
            "stoch_k",
            "stoch_d",
            "williams_r",
            "roc_10",
            "roc_20",
            "momentum_10",
            "momentum_20",
            "cci_20",
            "ultimate_oscillator",
            "money_flow_index",
            "chaikin_oscillator",
            "trix_14",
        ],
        # Famille TREND - Moyennes mobiles, MACD et indicateurs de tendance
        "TREND": [
            "sma_20",
            "sma_50",
            "sma_100",
            "sma_200",
            "ema_9",
            "ema_12",
            "ema_21",
            "ema_26",
            "ema_50",
            "ema_100",
            "ema_200",
            "macd_line",
            "macd_signal",
            "macd_histogram",
            "adx_14",
            "di_plus",
            "di_minus",
            "aroon_up",
            "aroon_down",
            "parabolic_sar",
        ],
        # Famille VOLUME - Indicateurs basés sur le volume
        "VOLUME": ["volume_sma_20", "volume_ratio", "obv", "obv_ema", "vwap"],
        # Famille VOLATILITY - Mesures de volatilité et bandes
        "VOLATILITY": [
            "atr_14",
            "atr_21",
            "bb_upper",
            "bb_middle",
            "bb_lower",
            "bb_width",
            "bb_percent",
        ],
        # Famille QUANTITATIVE - Indicateurs quantitatifs avancés
        "QUANTITATIVE": [
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
        ],
        # Famille CANDLESTICK_PATTERNS - Patterns de chandeliers
        "CANDLESTICK_PATTERNS": [
            "pattern_hammer",
            "pattern_inverted_hammer",
            "pattern_dragonfly_doji",
            "pattern_morning_star",
            "pattern_piercing_line",
            "pattern_three_white_soldiers",
            "pattern_engulfing_bull",
            "pattern_engulfing_bear",
            "pattern_harami_bull",
            "pattern_pin_bar_bull",
            "pattern_pin_bar_bear",
            "pattern_marubozu_bull",
            "pattern_abandoned_baby_bull",
            "pattern_three_inside_up",
            "pattern_three_outside_up",
            "pattern_concealing_baby_swallow",
            "pattern_unique_three_river_bottom",
            "pattern_belt_hold_bull",
            "pattern_kicking_bull",
            "pattern_matching_low",
            "pattern_ladder_bottom",
            "pattern_breakaway_bull",
            "pattern_hanging_man",
            "pattern_shooting_star",
            "pattern_gravestone_doji",
            "pattern_evening_star",
            "pattern_dark_cloud_cover",
            "pattern_three_black_crows",
            "pattern_harami_bear",
            "pattern_marubozu_bear",
            "pattern_abandoned_baby_bear",
            "pattern_three_inside_down",
            "pattern_three_outside_down",
            "pattern_advance_block",
            "pattern_deliberation",
            "pattern_belt_hold_bear",
            "pattern_kicking_bear",
            "pattern_matching_high",
            "pattern_ladder_top",
            "pattern_breakaway_bear",
            "pattern_doji",
            "pattern_long_legged_doji",
            "pattern_spinning_top",
            "pattern_four_price_doji",
            "pattern_rickshaw_man",
            "pattern_high_wave_candle",
            "pattern_tri_star",
        ],
        # Famille CHART_PATTERNS - Patterns de graphiques
        "CHART_PATTERNS": [
            "pattern_double_top",
            "pattern_double_bottom",
            "pattern_triple_top",
            "pattern_triple_bottom",
            "pattern_head_shoulders",
            "pattern_inv_head_shoulders",
            "pattern_rounding_top",
            "pattern_rounding_bottom",
            "pattern_diamond_top",
            "pattern_diamond_bottom",
            "pattern_v_top",
            "pattern_v_bottom",
            "pattern_island_top",
            "pattern_island_bottom",
            "pattern_spike_reversal",
            "pattern_ascending_triangle",
            "pattern_descending_triangle",
            "pattern_symmetrical_triangle",
            "pattern_rectangle",
            "pattern_bull_flag",
            "pattern_bear_flag",
            "pattern_bull_pennant",
            "pattern_bear_pennant",
            "pattern_rising_wedge",
            "pattern_falling_wedge",
            "pattern_broadening_wedge",
            "pattern_cup_handle",
            "pattern_channel_up",
            "pattern_channel_down",
            "pattern_measured_move",
        ],
        # Famille HARMONIC_PATTERNS - Patterns harmoniques et géométriques
        "HARMONIC_PATTERNS": [
            "pattern_gartley_bull",
            "pattern_gartley_bear",
            "pattern_butterfly_bull",
            "pattern_butterfly_bear",
            "pattern_bat_bull",
            "pattern_bat_bear",
            "pattern_crab_bull",
            "pattern_crab_bear",
            "pattern_shark_bull",
            "pattern_shark_bear",
            "pattern_three_drives",
            "pattern_wolfe_wave",
            "pattern_elliott_wave_1",
            "pattern_elliott_wave_3",
            "pattern_elliott_wave_5",
            "pattern_fibonacci_retracement",
            "pattern_fibonacci_extension",
        ],
        # Famille SUPPORT_RESISTANCE - Niveaux de support/résistance et gaps
        "SUPPORT_RESISTANCE": [
            "pattern_support",
            "pattern_resistance",
            "pattern_uptrend",
            "pattern_downtrend",
            "pattern_sideways_trend",
            "pattern_gap_up",
            "pattern_gap_down",
            "pattern_gap_fill",
            "pattern_breakaway_gap",
            "pattern_runaway_gap",
            "pattern_exhaustion_gap",
            "pattern_island_reversal",
            "pattern_gap_and_go",
        ],
        # ========== NOUVELLES FAMILLES POUR FACTEURS SOPHISTIQUÉS ==========
        # Famille CROSS_ASSET - Facteurs inter-actifs (nécessitent DataFrame unifié)
        "CROSS_ASSET": [
            # Ces features seront calculées dynamiquement à partir des facteurs préfixés
            "cross_asset_momentum_ratio",
            "cross_asset_correlation",
            "cross_asset_relative_strength",
        ],
        # Famille SECOND_ORDER - Facteurs dérivés (accélération, vitesse de changement)
        "SECOND_ORDER": [
            # Ces features sont des dérivées des facteurs de base
            "momentum_acceleration",
            "volatility_expansion_rate",
            "trend_acceleration",
        ],
        # Famille CONFLUENCE - Facteurs de confluence et divergence
        "CONFLUENCE": [
            # Ces features agrègent plusieurs signaux
            "divergence_signals",
            "confluence_signals",
            "climax_signals",
        ],
    }

    # Définition des facteurs alpha à calculer
    ALPHA_FACTORS = {
        "Factor_Momentum_Score": {
            "family": "MOMENTUM",
            "description": "Score agrégé de momentum basé sur RSI, Stochastic, ROC, CCI et oscillateurs",
            "features": FEATURE_FAMILIES["MOMENTUM"],
        },
        "Factor_Trend_Score": {
            "family": "TREND",
            "description": "Score de tendance basé sur moyennes mobiles, MACD, ADX et indicateurs directionnels",
            "features": FEATURE_FAMILIES["TREND"],
        },
        "Factor_Volume_Score": {
            "family": "VOLUME",
            "description": "Score de volume basé sur OBV, VWAP et ratios de volume",
            "features": FEATURE_FAMILIES["VOLUME"],
        },
        "Factor_Volatility_Score": {
            "family": "VOLATILITY",
            "description": "Score de volatilité basé sur ATR, Bollinger Bands et mesures de dispersion",
            "features": FEATURE_FAMILIES["VOLATILITY"],
        },
        "Factor_Quantitative_Score": {
            "family": "QUANTITATIVE",
            "description": "Score quantitatif avancé basé sur entropie, fractales et mesures statistiques",
            "features": FEATURE_FAMILIES["QUANTITATIVE"],
        },
        "Factor_Candlestick_Bullish_Score": {
            "family": "CANDLESTICK_PATTERNS",
            "description": "Score des patterns de chandeliers haussiers",
            "features": [
                f
                for f in FEATURE_FAMILIES["CANDLESTICK_PATTERNS"]
                if any(
                    bullish in f
                    for bullish in [
                        "bull",
                        "morning",
                        "white",
                        "piercing",
                        "hammer",
                        "dragonfly",
                    ]
                )
            ],
        },
        "Factor_Candlestick_Bearish_Score": {
            "family": "CANDLESTICK_PATTERNS",
            "description": "Score des patterns de chandeliers baissiers",
            "features": [
                f
                for f in FEATURE_FAMILIES["CANDLESTICK_PATTERNS"]
                if any(
                    bearish in f
                    for bearish in [
                        "bear",
                        "evening",
                        "black",
                        "dark",
                        "hanging",
                        "shooting",
                        "gravestone",
                    ]
                )
            ],
        },
        "Factor_Chart_Patterns_Score": {
            "family": "CHART_PATTERNS",
            "description": "Score des patterns de graphiques (triangles, flags, wedges)",
            "features": FEATURE_FAMILIES["CHART_PATTERNS"],
        },
        "Factor_Harmonic_Patterns_Score": {
            "family": "HARMONIC_PATTERNS",
            "description": "Score des patterns harmoniques (Gartley, Butterfly, Bat, etc.)",
            "features": FEATURE_FAMILIES["HARMONIC_PATTERNS"],
        },
        "Factor_Support_Resistance_Score": {
            "family": "SUPPORT_RESISTANCE",
            "description": "Score de support/résistance et analyse des gaps",
            "features": FEATURE_FAMILIES["SUPPORT_RESISTANCE"],
        },
        # ========== NOUVEAUX FACTEURS ALPHA SOPHISTIQUÉS ==========
        # Catégorie 1 : Facteurs de Régime et de Comportement du Marché
        "Factor_Regime_Hurst_Score": {
            "family": "QUANTITATIVE",
            "description": 'Mesure la "mémoire" du marché - tendance vs retour à la moyenne',
            "features": ["quant_hurst_exponent"],
            "calculation_type": "regime_analysis",
        },
        "Factor_Risk_TailEvent_Score": {
            "family": "QUANTITATIVE",
            "description": "Risque d'événements extrêmes basé sur skewness et kurtosis",
            "features": ["quant_rolling_skewness", "quant_rolling_kurtosis"],
            "calculation_type": "risk_analysis",
        },
        "Factor_Persistence_Score": {
            "family": "QUANTITATIVE",
            "description": "Mesure la persistance/momentum quantitatif via autocorrélations",
            "features": [
                "quant_Strat_RiskOn_autocorr_10",
                "quant_autocorr_20",
                "quant_autocorr_50",
            ],
            "calculation_type": "persistence_analysis",
        },
        # Catégorie 2 : Facteurs de Force Relative (Cross-Asset)
        "Factor_RelativeStrength_Momentum": {
            "family": "CROSS_ASSET",
            "description": "Momentum relatif entre actifs (nécessite DataFrame unifié)",
            "features": ["Factor_Momentum_Score"],  # Sera calculé dynamiquement
            "calculation_type": "cross_asset_relative",
        },
        "Factor_Correlation_Dynamic": {
            "family": "CROSS_ASSET",
            "description": "Corrélation glissante entre actifs majeurs",
            "features": ["close"],  # Sera calculé dynamiquement sur plusieurs actifs
            "calculation_type": "cross_asset_correlation",
        },
        # Catégorie 3 : Facteurs de "Second Ordre" (Dérivés)
        "Factor_Momentum_Acceleration_Score": {
            "family": "SECOND_ORDER",
            "description": "Accélération/décélération du momentum (indicateur avancé)",
            "features": ["Factor_Momentum_Score"],  # Dépend du facteur de base
            "calculation_type": "second_order_derivative",
        },
        "Factor_Volatility_Expansion_Rate": {
            "family": "SECOND_ORDER",
            "description": "Vitesse d'expansion de la volatilité",
            "features": ["Factor_Volatility_Score"],  # Dépend du facteur de base
            "calculation_type": "second_order_derivative",
        },
        # Catégorie 4 : Facteurs de Confluence et Divergence
        "Factor_Divergence_Score_Bullish": {
            "family": "CONFLUENCE",
            "description": "Score agrégé de divergences haussières multi-oscillateurs",
            "features": ["close", "Factor_Momentum_Score"],
            "calculation_type": "divergence_analysis",
        },
        "Factor_Divergence_Score_Bearish": {
            "family": "CONFLUENCE",
            "description": "Score agrégé de divergences baissières multi-oscillateurs",
            "features": ["close", "Factor_Momentum_Score"],
            "calculation_type": "divergence_analysis",
        },
        "Factor_Trend_Confirmation_Score": {
            "family": "CONFLUENCE",
            "description": "Confluence entre tendance, volume et persistance",
            "features": [
                "Factor_Trend_Score",
                "Factor_Volume_Score",
                "Factor_Persistence_Score",
            ],
            "calculation_type": "confluence_analysis",
        },
        "Factor_Reversal_Climax_Score": {
            "family": "CONFLUENCE",
            "description": "Détection de climax de marché (épuisement de tendance)",
            "features": [
                "Factor_Volatility_Score",
                "Factor_Volume_Score",
                "close",
                "bb_upper",
                "bb_lower",
            ],
            "calculation_type": "climax_analysis",
        },
    }
