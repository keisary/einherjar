import inspect
import logging
from typing import Any

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

# === IMPORTS NUMBA (avec fallback robuste) ===
try:
    from numba import njit, prange

    NUMBA_AVAILABLE = True
    logger.info("🚀 Numba disponible - Optimisations activées")
except ImportError:
    NUMBA_AVAILABLE = False
    logger.warning("⚠️ Numba non disponible - Mode fallback Python")

    # Fallback decorators
    def njit(*args, **kwargs):
        """Njit."""
        def decorator(func):
            """Decorator.

            Args:
            func: TODO document.
            """
            return func

        return decorator

    def prange(x):
        """Prange.

        Args:
            x: TODO document.
        """
        return range(x)

    # CORRECTION P1: safe_divide doit exister en mode fallback (sinón NameError)
    def safe_divide(numerator, denominator, default=0.0):
        """safe_divide.

        Args:
            numerator: TODO document.
            denominator: TODO document.
            default: TODO document.
        """
        if denominator == 0.0 or abs(denominator) < 1e-15:
            return default
        try:
            result = numerator / denominator
            if result != result or abs(result) > 1e10:  # NaN or overflow check
                return default
            return result
        except Exception:
            return default


PATTERN_THRESHOLDS = {
    # === CHANDELIERS JAPONAIS HAUSSIERS (20 patterns) ===
    "hammer": {
        "catégorie": "Chandelier Japonais",
        "type": "Haussier",
        "fenêtre": 1,
        "max_body_ratio": 0.33,
        "min_lower_shadow": 0.60,
        "max_upper_shadow": 0.10,
        "min_range_ratio": 0.003,
        "min_trend_lookback": 5,
    },
    "inverted_hammer": {
        "catégorie": "Chandelier Japonais",
        "type": "Haussier",
        "fenêtre": 1,
        "max_body_ratio": 0.33,
        "min_upper_shadow": 0.60,
        "max_lower_shadow": 0.10,
        "min_range_ratio": 0.003,
        "min_trend_lookback": 5,
    },
    "dragonfly_doji": {
        "catégorie": "Chandelier Japonais",
        "type": "Haussier",
        "fenêtre": 1,
        "max_body_ratio": 0.05,
        "min_lower_shadow": 0.70,
        "max_upper_shadow": 0.05,
        "min_range_ratio": 0.008,
        "proximity_threshold": 0.02,
        "body_weight": 0.25,
        "lower_weight": 0.40,
        "upper_weight": 0.20,
        "proximity_weight": 0.15,
    },
    "morning_star": {
        "catégorie": "Chandelier Japonais",
        "type": "Haussier",
        "fenêtre": 3,
        "min_body_ratio_1": 0.50,
        "max_body_ratio_2": 0.25,
        "min_body_ratio_3": 0.50,
        "body_weight_1": 0.30,
        "star_weight": 0.20,
        "body_weight_3": 0.35,
        "gap_weight": 0.15,
    },
    "piercing_line": {
        "catégorie": "Chandelier Japonais",
        "type": "Haussier",
        "fenêtre": 2,
        "min_body_ratio": 0.50,
        "min_penetration": 0.65,
        "body_weight_1": 0.30,
        "body_weight_2": 0.30,
        "penetration_weight": 0.40,
    },
    "three_white_soldiers": {
        "catégorie": "Chandelier Japonais",
        "type": "Haussier",
        "fenêtre": 3,
        "min_body_ratio": 0.55,
        "max_shadow_ratio": 0.20,
        "min_consecutive_advance": 0.25,  # Multiplicateur ATR: 0.25 × ATR local (remplace 0.02 absolu)
        "body_weight": 0.40,
        "shadow_weight": 0.20,
        "advance_weight": 0.40,
    },
    "engulfing_bull": {
        "catégorie": "Chandelier Japonais",
        "type": "Haussier",
        "fenêtre": 2,
        "min_body_ratio": 0.50,
        "min_engulf_ratio": 1.05,
        "body_weight_1": 0.30,
        "body_weight_2": 0.40,
        "engulf_weight": 0.30,
    },
    "engulfing_bear": {
        "catégorie": "Chandelier Japonais",
        "type": "Baissier",
        "fenêtre": 2,
        "min_body_ratio": 0.50,
        "min_engulf_ratio": 1.05,
        "body_weight_1": 0.30,
        "body_weight_2": 0.40,
        "engulf_weight": 0.30,
    },
    "harami_bull": {
        "catégorie": "Chandelier Japonais",
        "type": "Haussier",
        "fenêtre": 2,
        "min_body_ratio_1": 0.50,
        "max_body_ratio_2": 0.25,
        "body_weight_1": 0.40,
        "body_weight_2": 0.30,
        "containment_weight": 0.30,
    },
    "pin_bar_bull": {
        "catégorie": "Chandelier Japonais",
        "type": "Haussier",
        "fenêtre": 1,
        "max_body_ratio": 0.20,
        "min_lower_shadow": 0.75,
        "max_upper_shadow": 0.10,
        "body_weight": 0.25,
        "lower_weight": 0.45,
        "upper_weight": 0.10,
        "position_weight": 0.10,
        "context_weight": 0.10,
        "min_range_ratio": 0.005,
    },
    "pin_bar_bear": {
        "catégorie": "Chandelier Japonais",
        "type": "Baissier",
        "fenêtre": 1,
        "max_body_ratio": 0.20,
        "min_upper_shadow": 0.75,
        "max_lower_shadow": 0.10,
        "body_weight": 0.25,
        "upper_weight": 0.50,
        "lower_weight": 0.10,
        "position_weight": 0.15,
        "min_range_ratio": 0.005,
    },
    "marubozu_bull": {
        "catégorie": "Chandelier Japonais",
        "type": "Haussier",
        "fenêtre": 1,
        "min_body_ratio": 0.90,
        "max_shadow_ratio": 0.05,
        "min_range_ratio": 0.01,
        "body_weight": 0.70,
        "shadow_weight": 0.30,
    },
    "abandoned_baby_bull": {
        "catégorie": "Chandelier Japonais",
        "type": "Haussier",
        "fenêtre": 3,
        "min_body_ratio_1": 0.45,
        "max_body_ratio_2": 0.20,
        "min_body_ratio_3": 0.45,
        "body_weight_1": 0.25,
        "star_weight": 0.25,
        "body_weight_3": 0.25,
        "gap_weight": 0.25,
    },
    "three_inside_up": {
        "catégorie": "Chandelier Japonais",
        "type": "Haussier",
        "fenêtre": 3,
        "min_body_ratio_1": 0.45,
        "max_body_ratio_2": 0.40,
        "min_body_ratio_3": 0.50,
        "body_weight_1": 0.30,
        "body_weight_2": 0.30,
        "body_weight_3": 0.40,
    },
    "three_outside_up": {
        "catégorie": "Chandelier Japonais",
        "type": "Haussier",
        "fenêtre": 3,
        "min_body_ratio": 0.45,
        "body_weight_1": 0.30,
        "body_weight_2": 0.30,
        "body_weight_3": 0.40,
    },
    "concealing_baby_swallow": {
        "catégorie": "Chandelier Japonais",
        "type": "Haussier",
        "fenêtre": 4,
        "min_body_ratio": 0.45,
        "max_upper_shadow": 0.10,
        "body_weight": 0.60,
        "shadow_weight": 0.40,
    },
    "unique_three_river_bottom": {
        "catégorie": "Chandelier Japonais",
        "type": "Haussier",
        "fenêtre": 3,
        "min_body_ratio": 0.50,
        "max_body_ratio_2": 0.20,
        "body_weight": 0.50,
        "pattern_weight": 0.50,
    },
    "belt_hold_bull": {
        "catégorie": "Chandelier Japonais",
        "type": "Haussier",
        "fenêtre": 1,
        "min_body_ratio": 0.80,
        "max_lower_shadow": 0.05,
        "min_range_ratio": 0.01,
        "body_weight": 0.70,
        "shadow_weight": 0.30,
    },
    "kicking_bull": {
        "catégorie": "Chandelier Japonais",
        "type": "Haussier",
        "fenêtre": 2,
        "min_gap_ratio": 0.01,
        "min_body_ratio": 0.80,
        "min_volume_surge": 1.5,
        "body_weight": 0.35,
        "gap_weight": 0.40,
        "volume_weight": 0.25,
    },
    "matching_low": {
        "catégorie": "Chandelier Japonais",
        "type": "Haussier",
        "fenêtre": 2,
        "min_body_ratio": 0.60,
        "max_close_diff": 0.005,
        "min_range_ratio": 0.005,
        "body_weight": 0.50,
        "matching_weight": 0.50,
    },
    "ladder_bottom": {
        "catégorie": "Chandelier Japonais",
        "type": "Haussier",
        "fenêtre": 5,
        "min_body_ratio": 0.60,
        "min_pattern_length": 5,
        "max_upper_shadow": 0.15,
        "body_weight": 0.40,
        "pattern_weight": 0.35,
        "shadow_weight": 0.25,
    },
    "breakaway_bull": {
        "catégorie": "Chandelier Japonais",
        "type": "Haussier",
        "fenêtre": 5,
        "min_gap_ratio": 0.01,
        "min_continuation": 5,
        "min_volume_surge": 1.5,
        "gap_weight": 0.35,
        "continuation_weight": 0.35,
        "volume_weight": 0.30,
    },  # === CHANDELIERS JAPONAIS BAISSIERS (20 patterns) ===
    "hanging_man": {
        "catégorie": "Chandelier Japonais",
        "type": "Baissier",
        "fenêtre": 1,
        "max_body_ratio": 0.30,
        "min_lower_shadow": 0.60,
        "max_upper_shadow": 0.10,
        "min_range_ratio": 0.005,
        "context_weight": 0.10,
        "body_weight": 0.35,
        "lower_weight": 0.50,
        "upper_weight": 0.15,
    },
    "shooting_star": {
        "catégorie": "Chandelier Japonais",
        "type": "Baissier",
        "fenêtre": 1,
        "max_body_ratio": 0.30,
        "min_upper_shadow": 0.60,
        "max_lower_shadow": 0.10,
        "min_range_ratio": 0.005,
        "body_weight": 0.35,
        "upper_weight": 0.50,
        "lower_weight": 0.15,
    },
    "gravestone_doji": {
        "catégorie": "Chandelier Japonais",
        "type": "Baissier",
        "fenêtre": 1,
        "max_body_ratio": 0.05,
        "min_upper_shadow": 0.70,
        "max_lower_shadow": 0.05,
        "min_range_ratio": 0.008,
        "proximity_threshold": 0.02,
        "body_weight": 0.25,
        "upper_weight": 0.40,
        "lower_weight": 0.20,
        "proximity_weight": 0.15,
    },
    "evening_star": {
        "catégorie": "Chandelier Japonais",
        "type": "Baissier",
        "fenêtre": 3,
        "min_body_ratio_1": 0.50,
        "max_body_ratio_2": 0.25,
        "min_body_ratio_3": 0.50,
        "body_weight_1": 0.30,
        "star_weight": 0.20,
        "body_weight_3": 0.35,
        "gap_weight": 0.15,  # AUDIT FIX C5: réduit 0.30→0.15 (symétrique morning_star) — gaps quasi-absents en forex M5/M15/H1
    },
    "dark_cloud_cover": {
        "catégorie": "Chandelier Japonais",
        "type": "Baissier",
        "fenêtre": 2,
        "min_body_ratio": 0.50,
        "min_penetration": 0.65,
        "body_weight_1": 0.30,
        "body_weight_2": 0.30,
        "penetration_weight": 0.40,
    },
    "three_black_crows": {
        "catégorie": "Chandelier Japonais",
        "type": "Baissier",
        "fenêtre": 3,
        "min_body_ratio": 0.55,
        "max_shadow_ratio": 0.20,
        "min_consecutive_decline": 0.002,  # Normalisé: fallback absolu (ATR utilisé dans la fonction)
        "min_decline_atr_ratio": 0.25,  # 25% d'un ATR local → cross-asset
        "body_weight": 0.40,
        "shadow_weight": 0.20,
        "decline_weight": 0.40,
    },
    "harami_bear": {
        "catégorie": "Chandelier Japonais",
        "type": "Baissier",
        "fenêtre": 2,
        "min_body_ratio_1": 0.50,
        "max_body_ratio_2": 0.25,
        "body_weight_1": 0.40,
        "body_weight_2": 0.30,
        "containment_weight": 0.30,
    },
    "marubozu_bear": {
        "catégorie": "Chandelier Japonais",
        "type": "Baissier",
        "fenêtre": 1,
        "min_body_ratio": 0.90,
        "max_shadow_ratio": 0.10,
        "min_range_ratio": 0.01,
        "body_weight": 0.70,
        "shadow_weight": 0.30,
    },
    "abandoned_baby_bear": {
        "catégorie": "Chandelier Japonais",
        "type": "Baissier",
        "fenêtre": 3,
        "min_body_ratio_1": 0.45,
        "max_body_ratio_2": 0.20,
        "min_body_ratio_3": 0.45,
        "body_weight_1": 0.25,
        "star_weight": 0.25,
        "body_weight_3": 0.25,
        "gap_weight": 0.25,
    },
    "three_inside_down": {
        "catégorie": "Chandelier Japonais",
        "type": "Baissier",
        "fenêtre": 3,
        "min_body_ratio_1": 0.45,
        "max_body_ratio_2": 0.40,
        "min_body_ratio_3": 0.50,
        "body_weight_1": 0.30,
        "body_weight_2": 0.30,
        "body_weight_3": 0.40,
    },
    "three_outside_down": {
        "catégorie": "Chandelier Japonais",
        "type": "Baissier",
        "fenêtre": 3,
        "min_body_ratio": 0.45,
        "body_weight_1": 0.30,
        "body_weight_2": 0.30,
        "body_weight_3": 0.40,
    },
    "advance_block": {
        "catégorie": "Chandelier Japonais",
        "type": "Baissier",
        "fenêtre": 3,
        "min_body_ratio": 0.45,
        "max_upper_shadow": 0.20,
        "min_weakening": 0.05,
        "body_weight": 0.40,
        "shadow_weight": 0.30,
        "weakening_weight": 0.30,
    },
    "deliberation": {
        "catégorie": "Chandelier Japonais",
        "type": "Baissier",
        "fenêtre": 3,
        "min_body_ratio": 0.60,
        "max_body_ratio_3": 0.40,
        "max_upper_shadow": 0.15,
        "body_weight": 0.40,
        "weakening_weight": 0.30,
        "shadow_weight": 0.30,
    },
    "belt_hold_bear": {
        "catégorie": "Chandelier Japonais",
        "type": "Baissier",
        "fenêtre": 1,
        "min_body_ratio": 0.80,
        "max_upper_shadow": 0.05,
        "min_range_ratio": 0.01,
        "body_weight": 0.70,
        "shadow_weight": 0.30,
    },
    "kicking_bear": {
        "catégorie": "Chandelier Japonais",
        "type": "Baissier",
        "fenêtre": 2,
        "min_gap_ratio": 0.01,
        "min_body_ratio": 0.80,
        "min_volume_surge": 1.5,
        "body_weight": 0.35,
        "gap_weight": 0.40,
        "volume_weight": 0.25,
    },
    "matching_high": {
        "catégorie": "Chandelier Japonais",
        "type": "Baissier",
        "fenêtre": 2,
        "min_body_ratio": 0.60,
        "max_close_diff": 0.005,
        "min_range_ratio": 0.005,
        "body_weight": 0.50,
        "matching_weight": 0.50,
    },
    "ladder_top": {
        "catégorie": "Chandelier Japonais",
        "type": "Baissier",
        "fenêtre": 5,
        "min_body_ratio": 0.60,
        "min_pattern_length": 5,
        "max_lower_shadow": 0.15,
        "body_weight": 0.40,
        "pattern_weight": 0.35,
        "shadow_weight": 0.25,
    },
    "breakaway_bear": {
        "catégorie": "Chandelier Japonais",
        "type": "Baissier",
        "fenêtre": 5,
        "min_gap_ratio": 0.01,
        "min_continuation": 5,
        "min_volume_surge": 1.5,
        "gap_weight": 0.35,
        "continuation_weight": 0.35,
        "volume_weight": 0.30,
    },  # === CHANDELIERS JAPONAIS D'INDÉCISION (7 patterns) ===
    "doji": {
        "catégorie": "Chandelier Japonais",
        "type": "Indécision",
        "fenêtre": 1,
        "max_body_ratio": 0.10,  # Assoupli: 0.05 → 0.10 (cross-asset)
        "min_total_shadow": 0.60,  # Assoupli: 0.70 → 0.60 (plus de dojis)
        "min_range_ratio": 0.003,  # Assoupli: 0.008 → 0.003 (Forex + crypto)
        "body_weight": 0.60,
        "shadow_weight": 0.40,
    },
    "long_legged_doji": {
        "catégorie": "Chandelier Japonais",
        "type": "Indécision",
        "fenêtre": 1,
        "max_body_ratio": 0.05,
        "min_total_shadow": 0.80,
        "min_range_ratio": 0.01,
        "body_weight": 0.50,
        "shadow_weight": 0.50,
    },
    "spinning_top": {
        "catégorie": "Chandelier Japonais",
        "type": "Indécision",
        "fenêtre": 1,
        "max_body_ratio": 0.40,
        "min_total_shadow": 0.50,
        "min_range_ratio": 0.005,
        "body_weight": 0.40,
        "shadow_weight": 0.60,
    },
    "four_price_doji": {
        "catégorie": "Chandelier Japonais",
        "type": "Indécision",
        "fenêtre": 1,
        "max_body_ratio": 0.01,
        "min_range_ratio": 0.001,
        "body_weight": 0.60,
        "range_weight": 0.40,
    },
    "rickshaw_man": {
        "catégorie": "Chandelier Japonais",
        "type": "Indécision",
        "fenêtre": 1,
        "max_body_ratio": 0.10,
        "min_total_shadow": 0.75,
        "min_range_ratio": 0.01,
        "body_weight": 0.30,
        "shadow_weight": 0.70,
    },
    "high_wave_candle": {
        "catégorie": "Chandelier Japonais",
        "type": "Indécision",
        "fenêtre": 1,
        "max_body_ratio": 0.25,
        "min_total_shadow": 0.60,
        "min_range_ratio": 0.015,
        "body_weight": 0.40,
        "shadow_weight": 0.60,
    },
    "tri_star": {
        "catégorie": "Chandelier Japonais",
        "type": "Indécision",
        "fenêtre": 3,
        "max_body_ratio": 0.05,
        "min_gap_ratio": 0.001,
        "body_weight": 0.40,
        "gap_weight": 0.60,
    },
    # === PATTERNS CHARTISTES DE RETOURNEMENT (15 patterns) ===
    "double_top": {
        "catégorie": "Chartiste",
        "type": "Retournement",
        "fenêtre": {"min": 25},
        "min_peak_distance": 8,
        "max_height_diff_ratio": 0.05,
        "min_valley_depth_ratio": 0.04,
        "peak_weight": 0.40,
        "valley_weight": 0.30,
        "distance_weight": 0.30,
    },
    "double_bottom": {
        "catégorie": "Chartiste",
        "type": "Retournement",
        "fenêtre": {"min": 25},
        "min_valley_distance": 8,
        "max_depth_diff_ratio": 0.05,
        "min_peak_height_ratio": 0.04,
        "valley_weight": 0.40,
        "peak_weight": 0.30,
        "distance_weight": 0.30,
    },
    "triple_top": {
        "catégorie": "Chartiste",
        "type": "Retournement",
        "fenêtre": {"min": 16},
        "min_peak_distance": 8,
        "max_height_diff_ratio": 0.03,
        "min_valley_depth_ratio": 0.02,
        "peak_weight": 0.50,
        "valley_weight": 0.25,
        "distance_weight": 0.25,
    },
    "triple_bottom": {
        "catégorie": "Chartiste",
        "type": "Retournement",
        "fenêtre": {"min": 16},
        "min_valley_distance": 8,
        "max_depth_diff_ratio": 0.03,
        "min_peak_height_ratio": 0.02,
        "valley_weight": 0.50,
        "peak_weight": 0.25,
        "distance_weight": 0.25,
    },
    "head_shoulders": {
        "catégorie": "Chartiste",
        "type": "Retournement",
        "fenêtre": {"min": 30},
        "min_shoulder_distance": 10,
        "min_head_height_ratio": 0.01,
        "max_shoulder_diff_ratio": 0.35,
        "head_weight": 0.40,
        "shoulder_weight": 0.30,
        "neckline_weight": 0.30,
    },
    "inv_head_shoulders": {
        "catégorie": "Chartiste",
        "type": "Retournement",
        "fenêtre": {"min": 30},
        "min_shoulder_distance": 10,
        "min_head_depth_ratio": 0.01,
        "max_shoulder_diff_ratio": 0.35,
        "head_weight": 0.40,
        "shoulder_weight": 0.30,
        "neckline_weight": 0.30,
    },
    "rounding_top": {
        "catégorie": "Chartiste",
        "type": "Retournement",
        "fenêtre": {"min": 40},
        "min_curve_length": 30,  # NOUVEAU: Ajouté pour forcer la longueur
        "max_volatility_ratio": 0.03,
        "min_volume_decline": 0.30,
        "curve_weight": 0.40,
        "volume_weight": 0.30,
        "smoothness_weight": 0.30,
    },
    "rounding_bottom": {
        "catégorie": "Chartiste",
        "type": "Retournement",
        "fenêtre": {"min": 40},
        "min_curve_length": 30,
        "max_volatility_ratio": 0.03,
        "min_volume_increase": 0.30,
        "curve_weight": 0.40,
        "volume_weight": 0.30,
        "smoothness_weight": 0.30,
    },
    "diamond_top": {
        "catégorie": "Chartiste",
        "type": "Retournement",
        "fenêtre": {"min": 15},
        "min_pattern_length": 15,
        "min_volatility_expansion": 0.03,
        "min_volatility_contraction": 0.02,
        "expansion_weight": 0.35,
        "contraction_weight": 0.35,
        "symmetry_weight": 0.30,
    },
    "diamond_bottom": {
        "catégorie": "Chartiste",
        "type": "Retournement",
        "fenêtre": {"min": 15},
        "min_pattern_length": 15,
        "min_volatility_expansion": 0.03,
        "min_volatility_contraction": 0.02,
        "expansion_weight": 0.35,
        "contraction_weight": 0.35,
        "symmetry_weight": 0.30,
    },
    "v_top": {
        "catégorie": "Chartiste",
        "type": "Retournement",
        "fenêtre": {"min": 2, "max": 5},
        "min_spike_ratio": 0.02,
        "max_duration": 5,
        "min_volume_spike": 1.2,
        "spike_weight": 0.40,
        "duration_weight": 0.30,
        "volume_weight": 0.30,
    },
    "v_bottom": {
        "catégorie": "Chartiste",
        "type": "Retournement",
        "fenêtre": {"min": 2, "max": 5},
        "min_spike_ratio": 0.02,
        "max_duration": 5,
        "min_volume_spike": 1.2,
        "spike_weight": 0.40,
        "duration_weight": 0.30,
        "volume_weight": 0.30,
    },
    "island_top": {
        "catégorie": "Chartiste",
        "type": "Retournement",
        "fenêtre": {"min": 3, "max": 7},
        "min_gap_ratio": 0.001,
        "max_island_duration": 5,
        "min_volume_confirmation": 1.3,
        "gap_weight": 0.40,
        "isolation_weight": 0.30,
        "volume_weight": 0.30,
    },
    "island_bottom": {
        "catégorie": "Chartiste",
        "type": "Retournement",
        "fenêtre": {"min": 3, "max": 7},
        "min_gap_ratio": 0.005,
        "max_island_length": 5,
        "min_volume_surge": 2.0,
        "gap_weight": 0.40,
        "isolation_weight": 0.30,
        "volume_weight": 0.30,
    },
    "spike_reversal": {
        "catégorie": "Chartiste",
        "type": "Retournement",
        "fenêtre": {"min": 2},
        "min_spike_ratio": 0.015,
        "max_retracement": 0.8,
        "min_volume_spike": 1.3,
        "spike_weight": 0.40,
        "retracement_weight": 0.30,
        "volume_weight": 0.30,
    },
    # === PATTERNS CHARTISTES DE CONTINUATION (15 patterns) ===
    "ascending_triangle": {
        "catégorie": "Chartiste",
        "type": "Continuation",
        "fenêtre": {"min": 25},
        "min_pattern_length": 20,
        "max_resistance_slope": 0.01,
        "min_support_slope": 0.015,
        "min_convergence": 0.02,  # Convergence minimale des bornes
        "resistance_weight": 0.40,
        "support_weight": 0.35,
        "volume_weight": 0.25,
    },
    "descending_triangle": {
        "catégorie": "Chartiste",
        "type": "Continuation",
        "fenêtre": {"min": 25},
        "min_pattern_length": 20,
        "max_support_slope": -0.01,
        "min_resistance_slope": -0.015,
        "min_convergence": 0.02,  # Convergence minimale des bornes
        "support_weight": 0.40,
        "resistance_weight": 0.35,
        "volume_weight": 0.25,
    },
    "symmetrical_triangle": {
        "catégorie": "Chartiste",
        "type": "Continuation",
        "fenêtre": {"min": 25},
        "min_pattern_length": 20,
        "max_angle_diff": 0.1,
        "min_convergence": 0.03,
        "upper_weight": 0.35,
        "lower_weight": 0.35,
        "convergence_weight": 0.30,
    },
    "rectangle": {
        "catégorie": "Chartiste",
        "type": "Continuation",
        "fenêtre": {"min": 8},
        "min_pattern_length": 6,
        "max_slope_tolerance": 0.03,
        "min_touches": 3,
        "support_weight": 0.35,
        "resistance_weight": 0.35,
        "duration_weight": 0.30,
    },
    "bull_flag": {
        "catégorie": "Chartiste",
        "type": "Continuation",
        "fenêtre": {"min": 15},
        "min_pole_length": 5,
        "min_flag_length": 5,  # NOUVEAU — était hardcodé à 2 dans la fonction
        "max_flag_slope": -0.05,  # DURCI — était -0.10 (plage trop large)
        "min_volume_decline": 0.25,  # DURCI — était 0.20
        "pole_weight": 0.40,  # RÉÉQUILIBRÉ depuis 0.35
        "flag_weight": 0.35,
        "volume_weight": 0.25,  # RÉDUIT depuis 0.30 (volume forex moins fiable)
    },
    "bear_flag": {
        "catégorie": "Chartiste",
        "type": "Continuation",
        "fenêtre": {"min": 15},
        "min_pole_length": 5,
        "min_flag_length": 5,  # NOUVEAU — était hardcodé à 3 dans la fonction
        "max_flag_slope": 0.05,  # DURCI — était 0.10
        "min_volume_decline": 0.25,  # DURCI — était 0.20
        "pole_weight": 0.40,
        "flag_weight": 0.35,
        "volume_weight": 0.25,
    },
    "bull_pennant": {
        "catégorie": "Chartiste",
        "type": "Continuation",
        "fenêtre": {"min": 10, "max": 23},
        "min_pole_length": 5,
        "max_pennant_length": 20,
        "min_volume_decline": 0.25,
        "pole_weight": 0.35,
        "pennant_weight": 0.35,
        "volume_weight": 0.30,
    },
    "bear_pennant": {
        "catégorie": "Chartiste",
        "type": "Continuation",
        "fenêtre": {"min": 10, "max": 23},
        "min_pole_length": 5,
        "max_pennant_length": 20,
        "min_volume_decline": 0.25,
        "pole_weight": 0.35,
        "pennant_weight": 0.35,
        "volume_weight": 0.30,
    },
    "rising_wedge": {
        "catégorie": "Chartiste",
        "type": "Continuation",
        "fenêtre": {"min": 12},
        "min_pattern_length": 8,
        "max_angle_diff": 0.15,
        "min_convergence": 0.01,  # Convergence minimale des lignes du wedge
        "min_volume_decline": 0.15,
        "angle_weight": 0.40,
        "volume_weight": 0.30,
        "convergence_weight": 0.30,
    },
    "falling_wedge": {  # Zéro détection → seuils assouplis
        "catégorie": "Chartiste",
        "type": "Continuation",
        "fenêtre": {"min": 20},
        "min_pattern_length": 15,
        "max_angle_diff": 0.5,
        "min_convergence": 0.01,  # Convergence minimale des lignes du wedge
        "min_volume_decline": 0.10,
        "angle_weight": 0.40,
        "volume_weight": 0.30,
        "convergence_weight": 0.30,
    },
    "broadening_wedge": {
        "catégorie": "Chartiste",
        "type": "Continuation",
        "fenêtre": {"min": 12},
        "min_length": 12,
        "min_expansion_ratio": 0.03,
        "min_touches": 4,
        "expansion_weight": 0.40,
        "touch_weight": 0.30,
        "symmetry_weight": 0.30,
    },
    "cup_handle": {
        "catégorie": "Chartiste",
        "type": "Continuation",
        "fenêtre": {"min": 21, "max": 30},
        "min_cup_length": 20,
        "max_handle_length": 10,
        "min_depth_ratio": 0.15,
        "cup_weight": 0.40,
        "handle_weight": 0.30,
        "volume_weight": 0.30,
    },
    "channel_up": {
        "catégorie": "Chartiste",
        "type": "Continuation",
        "fenêtre": {"min": 15},
        "min_length": 15,
        "min_slope": 0.005,
        "max_width_variation": 0.50,
        "slope_weight": 0.35,
        "parallel_weight": 0.35,
        "duration_weight": 0.30,
    },
    "channel_down": {
        "catégorie": "Chartiste",
        "type": "Continuation",
        "fenêtre": {"min": 15},
        "min_length": 15,
        "min_slope": -0.005,
        "max_width_variation": 0.50,
        "slope_weight": 0.35,
        "parallel_weight": 0.35,
        "duration_weight": 0.30,
    },
    "measured_move": {
        "catégorie": "Chartiste",
        "type": "Continuation",
        "fenêtre": {"min": 10},
        "min_leg1_length": 8,
        "max_correction_ratio": 0.50,
        "min_target_ratio": 1.0,
        "leg1_weight": 0.35,
        "correction_weight": 0.30,
        "target_weight": 0.35,
    },  # === PATTERNS HARMONIQUES (10 patterns) ===
    "gartley_bull": {
        "catégorie": "Harmonique",
        "type": "Haussier",
        "fenêtre": {"min": 20, "max": 100},
        "xa_ratio": 0.618,
        "ab_ratio": 0.618,
        "bc_ratio": 0.886,
        "cd_ratio": 1.272,
        "tolerance": 0.15,
        "xa_weight": 0.25,
        "ab_weight": 0.25,
        "bc_weight": 0.25,
        "cd_weight": 0.25,
    },
    "gartley_bear": {
        "catégorie": "Harmonique",
        "type": "Baissier",
        "fenêtre": {"min": 20, "max": 100},
        "xa_ratio": 0.618,
        "ab_ratio": 0.618,
        "bc_ratio": 0.886,
        "cd_ratio": 1.272,
        "tolerance": 0.15,
        "xa_weight": 0.25,
        "ab_weight": 0.25,
        "bc_weight": 0.25,
        "cd_weight": 0.25,
    },
    "butterfly_bull": {
        "catégorie": "Harmonique",
        "type": "Haussier",
        "fenêtre": {"min": 20, "max": 100},
        "xa_ratio": 0.786,
        "ab_ratio": 0.618,
        "bc_ratio": 0.886,
        "cd_ratio": 1.618,
        "tolerance": 0.15,
        "xa_weight": 0.25,
        "ab_weight": 0.25,
        "bc_weight": 0.25,
        "cd_weight": 0.25,
    },
    "butterfly_bear": {
        "catégorie": "Harmonique",
        "type": "Baissier",
        "fenêtre": {"min": 20, "max": 100},
        "xa_ratio": 0.786,
        "ab_ratio": 0.618,
        "bc_ratio": 0.886,
        "cd_ratio": 1.618,
        "tolerance": 0.15,
        "xa_weight": 0.25,
        "ab_weight": 0.25,
        "bc_weight": 0.25,
        "cd_weight": 0.25,
    },
    # (Appliquer la même tolérance de 0.25 aux autres patterns harmoniques : Bat, Crab, Shark)
    "bat_bull": {
        "catégorie": "Harmonique",
        "type": "Haussier",
        "fenêtre": {"min": 20, "max": 100},
        "xa_ratio": 0.382,
        "ab_ratio": 0.618,
        "bc_ratio": 0.886,
        "cd_ratio": 2.618,
        "tolerance": 0.15,
        "xa_weight": 0.25,
        "ab_weight": 0.25,
        "bc_weight": 0.25,
        "cd_weight": 0.25,
    },
    "bat_bear": {
        "catégorie": "Harmonique",
        "type": "Baissier",
        "fenêtre": {"min": 20, "max": 100},
        "xa_ratio": 0.382,
        "ab_ratio": 0.618,
        "bc_ratio": 0.886,
        "cd_ratio": 2.618,
        "tolerance": 0.15,
        "xa_weight": 0.25,
        "ab_weight": 0.25,
        "bc_weight": 0.25,
        "cd_weight": 0.25,
    },
    "crab_bull": {
        "catégorie": "Harmonique",
        "type": "Haussier",
        "fenêtre": {"min": 20, "max": 100},
        "xa_ratio": 0.382,
        "ab_ratio": 0.618,
        "bc_ratio": 0.886,
        "cd_ratio": 3.618,
        "tolerance": 0.15,
        "xa_weight": 0.25,
        "ab_weight": 0.25,
        "bc_weight": 0.25,
        "cd_weight": 0.25,
    },
    "crab_bear": {
        "catégorie": "Harmonique",
        "type": "Baissier",
        "fenêtre": {"min": 20, "max": 100},
        "xa_ratio": 0.382,
        "ab_ratio": 0.618,
        "bc_ratio": 0.886,
        "cd_ratio": 3.618,
        "tolerance": 0.15,
        "xa_weight": 0.25,
        "ab_weight": 0.25,
        "bc_weight": 0.25,
        "cd_weight": 0.25,
    },
    "shark_bull": {
        "catégorie": "Harmonique",
        "type": "Haussier",
        "fenêtre": {"min": 20, "max": 100},
        "tolerance": 0.15,
        "min_pattern_size": 0.008,
        "pattern_weight": 1.0,
    },
    "shark_bear": {
        "catégorie": "Harmonique",
        "type": "Baissier",
        "fenêtre": {"min": 20, "max": 100},
        "tolerance": 0.15,
        "min_pattern_size": 0.008,
        "pattern_weight": 1.0,
    },
    # === PATTERNS safe_divide(SUPPORT, RÉSISTANCE) ET TENDANCES (5 patterns) ===
    "support": {
        "catégorie": "Tendance",
        "type": "Support",
        "fenêtre": {"min": 200},
        "window_size": 200,
        "min_touches": 3,
        "max_slope_tolerance": 0.02,
        "min_strength": 0.02,
        "touch_weight": 0.40,
        "slope_weight": 0.30,
        "strength_weight": 0.30,
    },
    "resistance": {
        "catégorie": "Tendance",
        "type": "Résistance",
        "fenêtre": {"min": 200},
        "window_size": 200,
        "min_touches": 3,
        "max_slope_tolerance": 0.02,
        "min_strength": 0.02,
        "touch_weight": 0.40,
        "slope_weight": 0.30,
        "strength_weight": 0.30,
    },
    "uptrend": {
        "catégorie": "Tendance",
        "type": "Tendance",
        "fenêtre": {"min": 20},
        "min_length": 20,
        "min_slope": 0.0005,
        "max_pullback_ratio": 0.40,
        "slope_weight": 0.40,
        "consistency_weight": 0.35,
        "volume_weight": 0.25,
    },
    "downtrend": {
        "catégorie": "Tendance",
        "type": "Tendance",
        "fenêtre": {"min": 20},
        "min_length": 20,
        "min_slope": -0.0005,
        "max_pullback_ratio": 0.40,
        "slope_weight": 0.40,
        "consistency_weight": 0.35,
        "volume_weight": 0.25,
    },
    "sideways_trend": {
        "catégorie": "Tendance",
        "type": "Tendance",
        "fenêtre": {"min": 40},
        "min_length": 30,
        "max_slope_tolerance": 0.0005,
        "max_range_volatility": 0.05,  # La volatilité interne du range doit être faible (2%)
        "min_containment_ratio": 0.85,  # NOUVEAU: Au moins 85% des bougies doivent être contenues dans le canal
        "slope_weight": 0.40,
        "containment_weight": 0.40,  # REMPLACE 'touch_weight'
        "duration_weight": 0.20,  # REMPLACE 'touch_weight'
    },
    # === PATTERNS DE GAP (8 patterns) ===
    "gap_up": {
        "catégorie": "Gap",
        "type": "Gap",
        "fenêtre": 2,
        "min_gap_ratio": 0.02,
        "min_volume_increase": 1.5,
        "gap_weight": 0.50,
        "volume_weight": 0.30,
        "follow_through_weight": 0.20,
    },
    "gap_down": {
        "catégorie": "Gap",
        "type": "Gap",
        "fenêtre": 2,
        "min_gap_ratio": 0.02,
        "min_volume_increase": 1.5,
        "gap_weight": 0.50,
        "volume_weight": 0.30,
        "follow_through_weight": 0.20,
    },
    "gap_fill": {
        "catégorie": "Gap",
        "type": "Gap",
        "fenêtre": {"min": 2, "max": 25},
        "min_gap_ratio": 0.02,  # AJOUTÉ: La fonction de remplissage utilisera ce seuil pour trouver des gaps pertinents
        "min_fill_ratio": 0.90,
        "max_fill_time": 25,
        "min_volume_confirmation": 1.2,
        "fill_weight": 0.40,
        "time_weight": 0.30,
        "volume_weight": 0.30,
    },
    "breakaway_gap": {
        "catégorie": "Gap",
        "type": "Gap",
        "fenêtre": {"min": 6},
        "min_gap_ratio": 0.02,
        "min_volume_surge": 2.0,
        "min_consolidation": 5,
        "min_continuation": 5,
        "gap_weight": 0.35,
        "volume_weight": 0.35,
        "consolidation_weight": 0.30,
    },
    "runaway_gap": {
        "catégorie": "Gap",
        "type": "Gap",
        "fenêtre": 2,
        "min_gap_ratio": 0.015,
        "min_trend_duration": 5,
        "min_volume_increase": 1.5,
        "gap_weight": 0.35,
        "trend_weight": 0.35,
        "volume_weight": 0.30,
    },
    "exhaustion_gap": {
        "catégorie": "Gap",
        "type": "Gap",
        "fenêtre": {"min": 2, "max": 4},
        "min_gap_ratio": 0.001,  # Abaissé: 0.005 → 0.001 (ATR-normalisé dans la fonction)
        "min_gap_atr_ratio": 0.1,  # 10% d'un ATR local → cross-asset
        "max_continuation": 3,
        "min_volume_climax": 1.5,  # Abaissé: 2.5 → 1.5 (Forex volume normalisé)
        "min_trend_atr_ratio": 1.5,  # Mouvement = 1.5× ATR sur 15 bougies (remplace trend_size 5%)
        "gap_weight": 0.35,
        "reversal_weight": 0.30,
        "volume_weight": 0.35,
    },
    "island_reversal": {
        "catégorie": "Gap",
        "type": "Gap",
        "fenêtre": {"min": 3, "max": 7},
        "min_gap_ratio": 0.025,
        "min_island_duration": 1,
        "max_island_duration": 5,
        "gap_weight": 0.40,
        "isolation_weight": 0.30,
        "reversal_weight": 0.30,
    },
    "gap_and_go": {
        "catégorie": "Gap",
        "type": "Gap",
        "fenêtre": 2,
        "min_gap_ratio": 0.01,
        "min_volume_increase": 1.8,
        "min_continuation_ratio": 0.02,
        "gap_weight": 0.35,
        "continuation_weight": 0.35,
        "volume_weight": 0.30,
    },  # === PATTERNS SPÉCIAUX ET AVANCÉS (7 patterns) ===
    "three_drives": {
        "catégorie": "Avancé",
        "type": "Vague",
        "fenêtre": {"min": 15},
        "min_drive_length": 3,
        "max_retracement_ratio": 0.85,  # Assoupli: 0.8 → 0.85
        "min_extension_ratio": 0.8,  # Assoupli: 1.0 → 0.8 (bug du list== réglé)
        "drive_weight": 0.40,
        "retracement_weight": 0.30,
        "extension_weight": 0.30,
    },
    "wolfe_wave": {
        "catégorie": "Avancé",
        "type": "Vague",
        "fenêtre": {"min": 5},
        "min_wave_count": 5,
        "symmetry_tolerance": 0.20,
        "projection_accuracy": 0.15,
        "wave_weight": 0.40,
        "symmetry_weight": 0.30,
        "projection_weight": 0.30,
    },
    "elliott_wave_1": {
        "catégorie": "Avancé",
        "type": "Vague",
        "fenêtre": {"min": 20},
        "min_wave_ratio": 0.50,
        "max_wave_ratio": 1.80,
        "min_volume_confirmation": 1.3,
        "wave_weight": 0.40,
        "ratio_weight": 0.30,
        "volume_weight": 0.30,
    },
    "elliott_wave_3": {
        "catégorie": "Avancé",
        "type": "Vague",
        "fenêtre": {"min": 20},
        "min_extension_ratio": 1.20,  # Abaissé: 1.618 → 1.20 (scoring graduel)
        "max_extension_ratio": 3.00,  # Élargi: 2.618 → 3.00
        "min_volume_surge": 1.2,  # Abaissé: 1.5 → 1.2; devient scoring, pas filtre dur
        "extension_weight": 0.45,
        "volume_weight": 0.25,
        "momentum_weight": 0.30,
    },
    "elliott_wave_5": {
        "catégorie": "Avancé",
        "type": "Vague",
        "fenêtre": {"min": 20},
        "min_divergence_ratio": 0.05,
        "max_extension_ratio": 1.618,
        "min_volume_decline": 0.8,
        "divergence_weight": 0.35,
        "extension_weight": 0.35,
        "volume_weight": 0.30,
    },
    "fibonacci_retracement": {
        "catégorie": "Avancé",
        "type": "Fibonacci",
        "fenêtre": {"min": 20},
        "min_swing_length": 5,  # AJOUTÉ: paramètre clé
        "retracement_tolerance": 0.03,  # AJOUTÉ: tolérance spécifique
        "min_bounce_strength": 0.01,  # AJOUTÉ: force du rebond
        "retracement_levels": [0.236, 0.382, 0.500, 0.618, 0.786],
        "tolerance": 0.1,
        "swing_weight": 0.35,  # MODIFIÉ
        "retracement_weight": 0.35,  # MODIFIÉ
        "bounce_weight": 0.30,  # MODIFIÉ
    },
    "fibonacci_extension": {
        "catégorie": "Avancé",
        "type": "Fibonacci",
        "fenêtre": {"min": 30},
        "min_swing_length": 4,  # AJOUTÉ: paramètre clé
        "extension_tolerance": 0.10,  # AJOUTÉ: tolérance spécifique
        "min_extension_strength": 0.02,  # AJOUTÉ: force de l'extension
        "extension_levels": [1.272, 1.414, 1.618, 2.000, 2.618],
        "tolerance": 0.1,
        "swing_weight": 0.35,  # AJOUTÉ
        "extension_weight": 0.35,  # AJOUTÉ
        "confirmation_weight": 0.30,  # AJOUTÉ
        "level_weight": 0.50,
        "volume_weight": 0.25,
        "momentum_weight": 0.25,
    },
}

# === SCALING TIMEFRAME ===
# Facteurs multiplicatifs par rapport à M5 (référence = 1.0).
# Basés sur la volatilité empirique observée : un ATR D1 ≈ 12× un ATR M5.
# Ces facteurs s'appliquent aux seuils absolus pour les rendre comparables
# entre timeframes (ex: min_slope trivial sur D1 si non scalé).
TF_MINUTES: dict[str, float] = {
    "5m": 5.0,   "M5": 5.0,
    "15m": 15.0, "M15": 15.0,
    "1h": 60.0,  "H1": 60.0,  "1H": 60.0,
    "4h": 240.0, "H4": 240.0, "4H": 240.0,
    "1d": 1440.0, "D1": 1440.0, "1D": 1440.0,
}

TF_SCALE_FACTORS: dict[str, float] = {
    "5m": 1.0,  "M5": 1.0,
    "15m": 1.5, "M15": 1.5,
    "1h": 3.0,  "H1": 3.0,  "1H": 3.0,
    "4h": 6.0,  "H4": 6.0,  "4H": 6.0,
    "1d": 12.0, "D1": 12.0, "1D": 12.0,
}

# Paramètres qui doivent être multipliés par le facteur de scaling TF.
# Ces paramètres représentent des seuils de mouvement minimal qui sont
# trivialement atteints sur les grandes timeframes avec leur valeur M5.
TF_SENSITIVE_PARAMS: set = {
    "min_slope",            # uptrend, downtrend : pente normalisée par barre
    "max_slope_tolerance",  # sideways, rectangle, support, resistance
    "min_strength",         # support, resistance : force minimale du niveau
    "min_range_ratio",      # doji, spinning_top, four_price_doji
    "min_bounce_strength",  # fibonacci_retracement
    "min_pattern_size",     # shark_bull, shark_bear
}

# === FONCTIONS UTILITAIRES COMMUNES ===
if NUMBA_AVAILABLE:

    @njit
    def calculate_body_ratio(open_val, high_val, low_val, close_val):
        """Calcule le ratio du corps par rapport au range total.

        Args:
            open_val: Prix d'ouverture
            high_val: Prix haut
            low_val: Prix bas
            close_val: Prix de clôture

        Returns:
            float: Ratio du corps (0.0 à 1.0)
        """
        range_total = high_val - low_val
        if range_total == 0.0:
            return 0.0
        body_size = abs(close_val - open_val)
        return safe_divide(body_size, range_total)

    @njit
    def calculate_upper_shadow_ratio(open_val, high_val, low_val, close_val):
        """Calcule le ratio de l'ombre haute par rapport au range total.

        Args:
            open_val: Prix d'ouverture
            high_val: Prix haut
            low_val: Prix bas
            close_val: Prix de clôture

        Returns:
            float: Ratio de l'ombre haute (0.0 à 1.0)
        """
        range_total = high_val - low_val
        if range_total == 0.0:
            return 0.0
        upper_shadow = high_val - max(open_val, close_val)
        return safe_divide(upper_shadow, range_total)

    @njit
    def calculate_lower_shadow_ratio(open_val, high_val, low_val, close_val):
        """Calcule le ratio de l'ombre basse par rapport au range total.

        Args:
            open_val: Prix d'ouverture
            high_val: Prix haut
            low_val: Prix bas
            close_val: Prix de clôture

        Returns:
            float: Ratio de l'ombre basse (0.0 à 1.0)
        """
        range_total = high_val - low_val
        if range_total == 0.0:
            return 0.0
        lower_shadow = min(open_val, close_val) - low_val
        return safe_divide(lower_shadow, range_total)

    @njit
    def is_bullish_candle(open_val, close_val):
        """Vérifie si la bougie est haussière (close > open).

        Args:
            open_val: Prix d'ouverture
            close_val: Prix de clôture

        Returns:
            bool: True si haussière, False sinon
        """
        return close_val > open_val

    @njit
    def is_bearish_candle(open_val, close_val):
        """Vérifie si la bougie est baissière (close < open).

        Args:
            open_val: Prix d'ouverture
            close_val: Prix de clôture

        Returns:
            bool: True si baissière, False sinon
        """
        return close_val < open_val

    @njit
    def calculate_range_ratio(high_val, low_val, reference_price):
        """Calcule le ratio du range par rapport à un prix de référence.

        Args:
            high_val: Prix haut
            low_val: Prix bas
            reference_price: Prix de référence

        Returns:
            float: Ratio du range par rapport au prix de référence
        """
        if reference_price == 0.0:
            return 0.0
        return safe_divide(high_val - low_val, reference_price)

    @njit
    def safe_divide(numerator, denominator, default=0.0):
        """Division sécurisée compatible Numba nopython.

        Définie en premier pour être disponible pour toutes les fonctions utilitaires.
        """
        if denominator == 0.0 or abs(denominator) < 1e-15:
            return default

        if (
            np.isnan(denominator)
            or np.isinf(denominator)
            or np.isnan(numerator)
            or np.isinf(numerator)
        ):
            return default

        result = numerator / denominator

        if np.isnan(result) or np.isinf(result) or abs(result) > 1e10:
            return default

        return result

    @njit
    def calculate_weighted_score(components, weights):
        """Calcule un score pondéré à partir des composants et de leurs poids.

        Args:
            components: Array des scores des composants
            weights: Array des poids correspondants

        Returns:
            float: Score pondéré final entre 0.0 et 1.0
        """
        total_score = 0.0
        total_weight = 0.0

        min_length = min(len(components), len(weights))

        for i in range(min_length):
            total_score += components[i] * weights[i]
            total_weight += weights[i]

        if total_weight == 0.0:
            return 0.0

        weighted_score = safe_divide(total_score, total_weight)
        return min(1.0, max(0.0, weighted_score))

    @njit
    def safe_mean(arr):
        """Calcul sécurisé de la moyenne compatible avec Numba.

        Args:
            arr: Array numpy ou liste

        Returns:
            float: Moyenne ou 0.0 si invalide
        """
        if len(arr) == 0:
            return 0.0

        total = 0.0
        count = 0

        for val in arr:
            if not (np.isnan(val) or np.isinf(val)):
                total += val
                count += 1

        if count == 0:
            return 0.0

        return safe_divide(total, count)

    @njit
    def calculate_std(arr):
        """Calcul sécurisé de l'écart-type compatible avec Numba.

        Args:
            arr: Array numpy ou liste

        Returns:
            float: Écart-type ou 0.0 si invalide
        """
        if len(arr) <= 1:
            return 0.0

        mean_val = safe_mean(arr)

        variance = 0.0
        count = 0

        for val in arr:
            if not (np.isnan(val) or np.isinf(val)):
                diff = val - mean_val
                variance += diff * diff
                count += 1

        if count <= 1:
            return 0.0

        variance = safe_divide(variance, (count - 1))
        return np.sqrt(variance) if variance >= 0 else 0.0

    @njit
    def validate_price_data(open_val, high_val, low_val, close_val):
        """Valide la cohérence des données OHLC.

        Args:
            open_val: Prix d'ouverture
            high_val: Prix haut
            low_val: Prix bas
            close_val: Prix de clôture

        Returns:
            bool: True si les données sont valides, False sinon
        """
        # Vérification des valeurs NaN ou infinies
        if (
            np.isnan(open_val)
            or np.isnan(high_val)
            or np.isnan(low_val)
            or np.isnan(close_val)
        ):
            return False

        if (
            np.isinf(open_val)
            or np.isinf(high_val)
            or np.isinf(low_val)
            or np.isinf(close_val)
        ):
            return False

        # Vérification de la cohérence OHLC
        # Le prix haut doit être >= aux prix d'ouverture et de clôture
        if high_val < max(open_val, close_val):
            return False

        # Le prix bas doit être <= aux prix d'ouverture et de clôture
        if low_val > min(open_val, close_val):
            return False

        # Vérification que high >= low
        if high_val < low_val:
            return False

        # Vérification que tous les prix sont positifs
        if open_val <= 0.0 or high_val <= 0.0 or low_val <= 0.0 or close_val <= 0.0:
            return False

        return True

    @njit
    def calculate_ema_numba(prices: np.ndarray, period: int) -> np.ndarray:
        """Calcule la Moyenne Mobile Exponentielle (EMA) de manière optimisée."""
        ema_values = np.full(len(prices), np.nan, dtype=np.float64)
        if len(prices) < period:
            return ema_values

        # L'initialisation se fait avec une Moyenne Mobile Simple
        sma = np.mean(prices[0:period])
        ema_values[int(period - 1)] = sma

        alpha = 2.0 / (period + 1.0)

        # Calcul itératif pour le reste des valeurs
        for i in range(int(period), len(prices)):
            ema_values[i] = (prices[i] * alpha) + (ema_values[i - 1] * (1 - alpha))

        return ema_values

    @njit
    def find_pivots(
        prices: np.ndarray,
        deviation_threshold: float = 0.05,
        min_pivot_distance: int = 5,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Version corrigée compatible Numba - Détection robuste de pivots.

        CORRECTIONS APPORTÉES:
        - Retourne des arrays NumPy typés au lieu de listes Python
        - Pré-allocation de mémoire pour éviter les erreurs de type
        - Logique simplifiée pour Numba

        Args:
            prices: Array des prix (close, high, low, etc.)
            deviation_threshold: Seuil de déviation minimum (5% par défaut)
            min_pivot_distance: Distance minimum entre pivots (5 par défaut)

        Returns:
            Tuple (indices, prices, types) où types: 1=high, -1=low
        """
        n = len(prices)
        if n < min_pivot_distance * 2:
            # Retourner des arrays vides typés (CORRECTION CRITIQUE)
            return (
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.int32),
            )

        # Pré-allouer des arrays de taille maximale possible
        max_pivots = n // min_pivot_distance + 2
        pivots_idx = np.zeros(max_pivots, dtype=np.int32)
        pivots_price = np.zeros(max_pivots, dtype=np.float64)
        pivots_type = np.zeros(max_pivots, dtype=np.int32)
        pivot_count = 0

        # État initial
        current_pivot_idx = 0
        current_pivot_price = prices[0]
        current_trend = 0  # 0=indéterminé, 1=haussier, -1=baissier

        # Parcourir les prix
        for i in range(min_pivot_distance, n):
            price = prices[i]

            # Calculer la déviation
            if current_pivot_price > 0:
                deviation = safe_divide(
                    (price - current_pivot_price), current_pivot_price
                )
            else:
                deviation = 0.0

            # Détecter un nouveau pivot significatif
            if abs(deviation) >= deviation_threshold:
                if deviation > 0:  # Prix monte
                    if current_trend <= 0:
                        # Confirmer le creux précédent
                        if pivot_count < max_pivots:
                            pivots_idx[pivot_count] = current_pivot_idx
                            pivots_price[pivot_count] = current_pivot_price
                            pivots_type[pivot_count] = -1  # Low
                            pivot_count += 1
                        current_trend = 1

                    current_pivot_idx = i
                    current_pivot_price = price

                else:  # Prix baisse
                    if current_trend >= 0:
                        # Confirmer le pic précédent
                        if pivot_count < max_pivots:
                            pivots_idx[pivot_count] = current_pivot_idx
                            pivots_price[pivot_count] = current_pivot_price
                            pivots_type[pivot_count] = 1  # High
                            pivot_count += 1
                        current_trend = -1

                    current_pivot_idx = i
                    current_pivot_price = price

            # Mettre à jour le pivot actuel si on continue dans la même direction
            elif current_trend == 1 and price > current_pivot_price:
                current_pivot_idx = i
                current_pivot_price = price
            elif current_trend == -1 and price < current_pivot_price:
                current_pivot_idx = i
                current_pivot_price = price

        # Ajouter le dernier pivot si nécessaire
        if pivot_count < max_pivots and pivot_count > 0:
            if current_pivot_idx != pivots_idx[pivot_count - 1]:
                pivots_idx[pivot_count] = current_pivot_idx
                pivots_price[pivot_count] = current_pivot_price
                pivots_type[pivot_count] = current_trend
                pivot_count += 1

        # Retourner seulement la partie utilisée (CORRECTION CRITIQUE)
        return (
            pivots_idx[:pivot_count].copy(),
            pivots_price[:pivot_count].copy(),
            pivots_type[:pivot_count].copy(),
        )

    @njit
    def get_pivot_sequence(
        pivot_indices: np.ndarray,
        pivot_prices: np.ndarray,
        pivot_types: np.ndarray,
        lookback: int = 10,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Retourne les N derniers pivots pour analyse de patterns.

        Args:
            pivot_indices: Indices des pivots
            pivot_prices: Prix des pivots
            pivot_types: Types des pivots (1=high, -1=low)
            lookback: Nombre de pivots à retourner

        Returns:
            Tuple (indices, prices, types) des derniers pivots
        """
        n = len(pivot_indices)
        if n == 0:
            # CORRECTION P2: arrays typés pour éviter TypingError Numba
            return (
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.int32),
            )

        # Prendre les derniers pivots
        start_idx = max(0, n - lookback)

        return (
            pivot_indices[start_idx:],
            pivot_prices[start_idx:],
            pivot_types[start_idx:],
        )

    @njit
    def validate_pivot_alternation(pivot_types: np.ndarray) -> bool:
        """Vérifie que les pivots alternent correctement (high-low-high-low...).

        Args:
            pivot_types: Array des types de pivots

        Returns:
            True si les pivots alternent correctement
        """
        if len(pivot_types) < 2:
            return True

        for i in range(1, len(pivot_types)):
            if pivot_types[i] == pivot_types[i - 1]:
                return False

        return True

    @njit
    def check_full_engulfment_bull(
        prev_open,
        prev_high,
        prev_low,
        prev_close,
        curr_open,
        curr_high,
        curr_low,
        curr_close,
    ):
        """Vérifie l'englobement du corps pour un pattern haussier:.

        - Le corps du chandelier actuel (haussier) doit englober complètement le corps du précédent (baissier).
        """
        # Un englobement haussier parfait sur le CORPS, mais on ignore l'ombre
        curr_body_top = curr_close
        curr_body_bottom = curr_open
        prev_body_top = prev_open
        prev_body_bottom = prev_close

        body_engulfment = (
            curr_body_bottom <= prev_body_bottom and curr_body_top >= prev_body_top
        )

        return body_engulfment

    @njit
    def check_full_engulfment_bear(
        prev_open,
        prev_high,
        prev_low,
        prev_close,
        curr_open,
        curr_high,
        curr_low,
        curr_close,
    ):
        """Vérifie l'englobement du corps pour un pattern baissier:.

        - Le corps du chandelier actuel (baissier) doit englober complètement le corps du précédent (haussier).
        """
        # Un englobement baissier parfait sur le CORPS, on ignore l'ombre
        curr_body_top = curr_open
        curr_body_bottom = curr_close
        prev_body_top = prev_close
        prev_body_bottom = prev_open

        body_engulfment = (
            curr_body_top >= prev_body_top and curr_body_bottom <= prev_body_bottom
        )

        return body_engulfment

    # === P0-4 HELPERS: Remplacement de sorted(key=lambda) — Numba nopython compatible ===

    @njit
    def _find_two_highest(pk_idx: np.ndarray, pk_price: np.ndarray, n: int):
        """Retourne (idx1, p1, idx2, p2) des 2 pics les plus hauts en O(n).

        idx1/p1 = plus haut, idx2/p2 = deuxième plus haut.
        Compatible Numba nopython — remplace sorted(key=lambda x: x[1], reverse=True).
        """
        best1_i, best2_i = np.int64(-1), np.int64(-1)
        best1_p, best2_p = -1e18, -1e18
        for k in range(n):
            p = pk_price[k]
            if p > best1_p:
                best2_p = best1_p
                best2_i = best1_i
                best1_p = p
                best1_i = pk_idx[k]
            elif p > best2_p:
                best2_p = p
                best2_i = pk_idx[k]
        return best1_i, best1_p, best2_i, best2_p

    @njit
    def _find_two_lowest(pk_idx: np.ndarray, pk_price: np.ndarray, n: int):
        """Retourne (idx1, p1, idx2, p2) des 2 creux les plus bas en O(n).

        idx1/p1 = plus bas, idx2/p2 = deuxième plus bas.
        Compatible Numba nopython — remplace sorted(key=lambda x: x[1]).
        """
        best1_i, best2_i = np.int64(-1), np.int64(-1)
        best1_p, best2_p = 1e18, 1e18
        for k in range(n):
            p = pk_price[k]
            if p < best1_p:
                best2_p = best1_p
                best2_i = best1_i
                best1_p = p
                best1_i = pk_idx[k]
            elif p < best2_p:
                best2_p = p
                best2_i = pk_idx[k]
        return best1_i, best1_p, best2_i, best2_p

    @njit
    def _find_first_last_by_time(pk_idx: np.ndarray, pk_price: np.ndarray, n: int):
        """Retourne (first_i, first_p, last_i, last_p) triés par index temporel.

        Compatible Numba nopython — remplace list.sort(key=lambda x: x[0]).
        """
        first_i, last_i = np.int64(999999), np.int64(-1)
        first_p, last_p = 0.0, 0.0
        for k in range(n):
            if pk_idx[k] < first_i:
                first_i = pk_idx[k]
                first_p = pk_price[k]
            if pk_idx[k] > last_i:
                last_i = pk_idx[k]
                last_p = pk_price[k]
        return first_i, first_p, last_i, last_p

    @njit
    def _find_pivots_array(
        prices: np.ndarray, is_high: bool, window: int, n_total: int
    ):
        """Collecte les pivots hauts (is_high=True) ou bas (is_high=False) dans prices[].

        Retourne (pk_idx, pk_price, n_found) sur des arrays pré-alloués.
        Compatible Numba nopython — remplace peaks.append() + sorted().
        """
        max_pts = window + 4
        pk_idx = np.empty(max_pts, dtype=np.int64)
        pk_price = np.empty(max_pts, dtype=np.float64)
        n_found = 0
        for j in range(1, len(prices) - 1):
            if is_high:
                if prices[j] > prices[j - 1] and prices[j] > prices[j + 1]:
                    pk_idx[n_found] = j
                    pk_price[n_found] = prices[j]
                    n_found += 1
            else:
                if prices[j] < prices[j - 1] and prices[j] < prices[j + 1]:
                    pk_idx[n_found] = j
                    pk_price[n_found] = prices[j]
                    n_found += 1
            if n_found >= max_pts:
                break
        return pk_idx, pk_price, n_found

    @njit
    def find_pivots_simple(
        prices: np.ndarray, deviation_threshold: float, min_pivot_distance: int
    ):
        """Détection simplifiée de pivots pour les patterns chartistes.

        Retourne les indices, prix et types des pivots (1=high, -1=low).
        """
        n = len(prices)
        if n < min_pivot_distance * 2:
            return (
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.int32),
            )

        # Pré-allouer des arrays - correction pour compatibilité Numba
        max_pivots = int(n // min_pivot_distance + 2)
        pivots_idx = np.empty(max_pivots, dtype=np.int32)
        pivots_price = np.empty(max_pivots, dtype=np.float64)
        pivots_type = np.empty(max_pivots, dtype=np.int32)
        pivot_count = 0

        # État initial
        current_pivot_idx = 0
        current_pivot_price = prices[0]
        current_trend = 0

        for i in range(min_pivot_distance, n):
            price = prices[i]

            if current_pivot_price > 0:
                deviation = safe_divide(
                    (price - current_pivot_price), current_pivot_price
                )
            else:
                deviation = 0.0

            if abs(deviation) >= deviation_threshold:
                if deviation > 0:  # Prix monte
                    if current_trend <= 0:
                        # Confirmer le creux précédent
                        if pivot_count < max_pivots:
                            pivots_idx[pivot_count] = current_pivot_idx
                            pivots_price[pivot_count] = current_pivot_price
                            pivots_type[pivot_count] = -1  # Low
                            pivot_count += 1
                        current_trend = 1
                    current_pivot_idx = i
                    current_pivot_price = price
                else:  # Prix baisse
                    if current_trend >= 0:
                        # Confirmer le pic précédent
                        if pivot_count < max_pivots:
                            pivots_idx[pivot_count] = current_pivot_idx
                            pivots_price[pivot_count] = current_pivot_price
                            pivots_type[pivot_count] = 1  # High
                            pivot_count += 1
                        current_trend = -1
                    current_pivot_idx = i
                    current_pivot_price = price
            elif current_trend == 1 and price > current_pivot_price:
                current_pivot_idx = i
                current_pivot_price = price
            elif current_trend == -1 and price < current_pivot_price:
                current_pivot_idx = i
                current_pivot_price = price

        # Ajouter le dernier pivot si nécessaire (Essentiel pour valider un pattern sur le jour courant i !!)
        if pivot_count < max_pivots and pivot_count > 0:
            if current_pivot_idx != pivots_idx[pivot_count - 1]:
                pivots_idx[pivot_count] = current_pivot_idx
                pivots_price[pivot_count] = current_pivot_price
                pivots_type[pivot_count] = current_trend if current_trend != 0 else 1
                pivot_count += 1

        # Retourner seulement la partie utilisée
        return (
            pivots_idx[:pivot_count].copy(),
            pivots_price[:pivot_count].copy(),
            pivots_type[:pivot_count].copy(),
        )

    @njit
    def calculate_linear_regression_slope(x_values: np.ndarray, y_values: np.ndarray):
        """Calcule la pente d'une régression linéaire de manière sécurisée."""
        n = len(x_values)
        if n < 2 or len(y_values) != n:
            return 0.0

        # Calcul des moyennes
        x_mean = 0.0
        y_mean = 0.0
        for i in range(n):
            x_mean += x_values[i]
            y_mean += y_values[i]
        x_mean = safe_divide(x_mean, n)
        y_mean = safe_divide(y_mean, n)

        # Calcul des sommes pour la pente
        numerator = 0.0
        denominator = 0.0
        for i in range(n):
            x_diff = x_values[i] - x_mean
            y_diff = y_values[i] - y_mean
            numerator += x_diff * y_diff
            denominator += x_diff * x_diff

        return safe_divide(numerator, denominator)

    @njit
    def calculate_fibonacci_ratio(
        price_a: float, price_b: float, price_c: float, price_d: float = 0.0
    ):
        """Calcule le ratio de Fibonacci entre des prix (sécurisé)."""
        if price_d == 0.0:  # Ratio AB/XA ou BC/AB
            if abs(price_a - price_c) < 1e-10:
                return 0.0
            return abs(price_b - price_c) / abs(price_a - price_c)
        else:  # Ratio CD/XA
            if abs(price_a - price_c) < 1e-10:
                return 0.0
            return abs(price_d - price_b) / abs(price_a - price_c)

    @njit
    def calculate_trend_strength(prices: np.ndarray, start_idx: int, end_idx: int):
        """Calcule la force d'une tendance de manière sécurisée."""
        if end_idx <= start_idx or start_idx < 0 or end_idx >= len(prices):
            return 0.0

        length = end_idx - start_idx + 1
        if length < 2:
            return 0.0

        # Calcul de la pente
        start_price = prices[start_idx]
        end_price = prices[end_idx]

        if start_price <= 0:
            return 0.0

        trend_change = safe_divide((end_price - start_price), start_price)
        trend_strength = abs(trend_change) / length  # Normalise par la durée

        return trend_strength

    @njit
    def calculate_trend_consistency(
        prices: np.ndarray, start_idx: int, end_idx: int, expected_direction: int
    ):
        """Calcule la consistance d'une tendance (% de mouvements dans la bonne direction).

        expected_direction: 1 pour haussier, -1 pour baissier, 0 pour sideways.
        """
        if end_idx <= start_idx or start_idx < 0 or end_idx >= len(prices):
            return 0.0

        consistent_moves = 0
        total_moves = 0

        for i in range(start_idx + 1, end_idx + 1):
            if prices[i - 1] > 0:  # Eviter division par zéro
                move = prices[i] - prices[i - 1]
                total_moves += 1

                if expected_direction == 1 and move > 0:  # Tendance haussière
                    consistent_moves += 1
                elif expected_direction == -1 and move < 0:  # Tendance baissière
                    consistent_moves += 1
                elif (
                    expected_direction == 0 and abs(move) < 0.01 * prices[i - 1]
                ):  # Tendance sideways
                    consistent_moves += 1

        if total_moves == 0:
            return 0.0

        return safe_divide(consistent_moves, total_moves)

    @njit
    def calculate_fibonacci_levels(start_price: float, end_price: float):
        """Calcule les niveaux de retracement et extension de Fibonacci de manière sécurisée."""
        if start_price <= 0 or end_price <= 0:
            return np.zeros(8, dtype=np.float64)

        price_range = end_price - start_price

        # Niveaux de retracement et extension Fibonacci
        ratios = np.array([0.236, 0.382, 0.5, 0.618, 0.786, 1.272, 1.618, 2.618])
        levels = np.zeros(8, dtype=np.float64)

        for i in range(8):
            levels[i] = start_price + (price_range * ratios[i])

        return levels

    @njit
    def calculate_slope_safe(x_values: np.ndarray, y_values: np.ndarray) -> float:
        """Calcule la pente d'une régression linéaire de manière sécurisée.

        Utilise la méthode des moindres carrés.

        Args:
            x_values: Valeurs X (indices temporels)
            y_values: Valeurs Y (prix)

        Returns:
            Pente de la droite de régression (0.0 si erreur)
        """
        n = len(x_values)

        # Vérifications de sécurité
        if n < 2 or len(y_values) != n:
            return 0.0

        # Vérifier qu'il y a de la variation dans les données
        all_same = True
        first_val = y_values[0]
        for i in range(1, n):
            if abs(y_values[i] - first_val) > 1e-10:
                all_same = False
                break

        if all_same:
            return 0.0  # Ligne horizontale

        # Calcul des moyennes
        sum_x = 0.0
        sum_y = 0.0
        for i in range(n):
            sum_x += x_values[i]
            sum_y += y_values[i]
        mean_x = sum_x / n
        mean_y = sum_y / n

        # Calcul de la pente par la méthode des moindres carrés
        numerator = 0.0
        denominator = 0.0

        for i in range(n):
            x_diff = x_values[i] - mean_x
            y_diff = y_values[i] - mean_y
            numerator += x_diff * y_diff
            denominator += x_diff * x_diff

        # Éviter la division par zéro
        if abs(denominator) < 1e-10:
            return 0.0

        slope = numerator / denominator

        # Limiter les valeurs extrêmes
        if abs(slope) > 1000.0:
            if slope > 0:
                return 1000.0
            else:
                return -1000.0

        return slope

    @njit
    def fit_parabola(x, y):
        """Calcule le coefficient de courbure 'a' d'une parabole (y = ax^2 + bx + c).

        CORRECTION P2: l'ancienne formule était mathématiquement incorrecte (système 2x2
        pour 3 inconnues). Cette version calcule 'a' via la méthode des moindres carrés
        complète 3x3 (Vandermonde) en utilisant seulement les moments nécessaires.
        """
        n = len(x)
        if n < 3:
            return 0.0, 0.0, 0.0

        # Moments pour résoudre le système 3x3 [a, b, c]
        S0 = float(n)
        S1 = 0.0
        S2 = 0.0
        S3 = 0.0
        S4 = 0.0
        Sy0 = 0.0
        Sy1 = 0.0
        Sy2 = 0.0
        for k in range(n):
            xi = x[k]
            yi = y[k]
            xi2 = xi * xi
            xi3 = xi2 * xi
            xi4 = xi3 * xi
            S1 += xi
            S2 += xi2
            S3 += xi3
            S4 += xi4
            Sy0 += yi
            Sy1 += xi * yi
            Sy2 += xi2 * yi

        # Système: [[S4,S3,S2],[S3,S2,S1],[S2,S1,S0]] * [a,b,c] = [Sy2,Sy1,Sy0]
        # Déterminant du système 3x3 (règle de Cramer)
        det = (
            S4 * (S2 * S0 - S1 * S1)
            - S3 * (S3 * S0 - S1 * S2)
            + S2 * (S3 * S1 - S2 * S2)
        )
        if abs(det) < 1e-12:
            return 0.0, 0.0, 0.0

        # Coefficient 'a' : numérateur via substitution de la première colonne
        det_a = (
            Sy2 * (S2 * S0 - S1 * S1)
            - S3 * (Sy1 * S0 - S1 * Sy0)
            + S2 * (Sy1 * S1 - S2 * Sy0)
        )
        a = safe_divide(det_a, det)
        return a, 0.0, 0.0  # b et c non utilisés par les callers

# === FONCTIONS DE DÉTECTION DES PATTERNS HAUSSIERS ===

if NUMBA_AVAILABLE:

    @njit(parallel=True)
    def detect_hammer_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        max_body_ratio: float,
        min_lower_shadow: float,
        max_upper_shadow: float,
        min_range_ratio: float,
        min_trend_lookback: int,
    ) -> np.ndarray:
        """Détection du pattern "hammer" en parallèle.

        Vérification robuste d'une tendance baissière préalable (majorité de jours baissiers).
        """
        n = len(open_prices)
        signals = np.zeros(n, dtype=np.float64)

        for i in prange(min_trend_lookback, n):
            # Tendance préalable baissière ?
            # Vérifier qu'une majorité des derniers jours étaient baissiers (Close < Close précédent)
            down_days = 0
            for j in range(max(2, i - min_trend_lookback + 1), i + 1):
                if close_prices[j - 1] < close_prices[j - 2]:
                    down_days += 1

            # Au moins la moitié des jours du lookback doivent être baissiers, ou le prix a fortement chuté
            is_valid_downtrend = (down_days >= (min_trend_lookback / 2.0)) or (
                close_prices[i - 1] < close_prices[i - min_trend_lookback] * 0.98
            )

            if not is_valid_downtrend:
                continue

            open_val = open_prices[i]
            high_val = high_prices[i]
            low_val = low_prices[i]
            close_val = close_prices[i]

            # Validation des données
            if (
                high_val < low_val
                or open_val < low_val
                or open_val > high_val
                or close_val < low_val
                or close_val > high_val
                or high_val <= 0
                or low_val <= 0
                or np.isnan(open_val)
                or np.isnan(high_val)
                or np.isnan(low_val)
                or np.isnan(close_val)
            ):
                continue

            range_total = high_val - low_val
            if range_total <= 0:
                continue

            body_size = abs(close_val - open_val)
            body_ratio = body_size / range_total
            lower_shadow = min(open_val, close_val) - low_val
            lower_shadow_ratio = lower_shadow / range_total
            upper_shadow = high_val - max(open_val, close_val)
            upper_shadow_ratio = upper_shadow / range_total
            range_ratio = range_total / close_val if close_val > 0 else 0

            if (
                body_ratio <= max_body_ratio
                and lower_shadow_ratio >= min_lower_shadow
                and upper_shadow_ratio <= max_upper_shadow
                and range_ratio >= min_range_ratio
            ):
                # P2-1 FIX: Score composé au lieu de 1.0 binaire
                lower_shadow_score = min(
                    1.0, safe_divide(lower_shadow_ratio, min_lower_shadow * 1.5)
                )
                upper_shadow_score = 1.0 - min(
                    1.0,
                    safe_divide(upper_shadow, body_size if body_size > 0 else 1e-10),
                )
                # AUDIT FIX Bloc1: body petit = meilleur Hammer (Nison)
                # body_ratio=0.05, max=0.33 → score≈0.85 ; body_ratio=0.30 → score≈0.09
                body_component = min(
                    1.0,
                    max(
                        0.0, 1.0 - safe_divide(body_size, range_total) / max_body_ratio
                    ),
                )

                components = np.array(
                    [lower_shadow_score, upper_shadow_score, body_component]
                )
                weights = np.array([0.5, 0.3, 0.2])
                signals[i] = calculate_weighted_score(components, weights)

        return signals

    @njit(parallel=True)
    def detect_inverted_hammer_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        max_body_ratio: float,
        min_upper_shadow: float,
        max_lower_shadow: float,
        min_range_ratio: float,
        min_trend_lookback: int,
    ) -> np.ndarray:
        """Détecte le pattern Inverted Hammer en parallèle.

        Vérification robuste d'une tendance baissière préalable (majorité de jours baissiers).
        """
        n = len(open_prices)
        signals = np.zeros(n, dtype=np.float64)

        for i in prange(min_trend_lookback, n):
            # Tendance préalable baissière ?
            down_days = 0
            for j in range(max(2, i - min_trend_lookback + 1), i + 1):
                if close_prices[j - 1] < close_prices[j - 2]:
                    down_days += 1

            is_valid_downtrend = (down_days >= (min_trend_lookback / 2.0)) or (
                close_prices[i - 1] < close_prices[i - min_trend_lookback] * 0.98
            )

            if not is_valid_downtrend:
                continue

            open_val = open_prices[i]
            high_val = high_prices[i]
            low_val = low_prices[i]
            close_val = close_prices[i]

            # Validation des données OHLC (intégrée)
            if (
                high_val < low_val
                or open_val < low_val
                or open_val > high_val
                or close_val < low_val
                or close_val > high_val
                or high_val <= 0
                or low_val <= 0
                or np.isnan(open_val)
                or np.isnan(high_val)
                or np.isnan(low_val)
                or np.isnan(close_val)
            ):
                continue

            # Calcul des ratios (intégré)
            range_total = high_val - low_val
            if range_total <= 0:
                continue

            # Body ratio
            body_size = abs(close_val - open_val)
            body_ratio = body_size / range_total

            # Lower shadow ratio
            lower_shadow = min(open_val, close_val) - low_val
            lower_shadow_ratio = lower_shadow / range_total

            # Upper shadow ratio
            upper_shadow = high_val - max(open_val, close_val)
            upper_shadow_ratio = upper_shadow / range_total

            # Range ratio (par rapport au prix de clôture)
            range_ratio = range_total / close_val if close_val > 0 else 0

            # Vérification des conditions
            if (
                body_ratio <= max_body_ratio
                and upper_shadow_ratio >= min_upper_shadow
                and lower_shadow_ratio <= max_lower_shadow
                and range_ratio >= min_range_ratio
            ):
                # P2-1 FIX: Score composé au lieu de 1.0 binaire
                upper_shadow_score = min(
                    1.0, safe_divide(upper_shadow_ratio, min_upper_shadow * 1.5)
                )
                lower_shadow_score = 1.0 - min(
                    1.0,
                    safe_divide(lower_shadow, body_size if body_size > 0 else 1e-10),
                )
                # AUDIT FIX Bloc1: body petit = meilleur Inverted Hammer (Nison)
                # Même correction que hammer — copier-coller inversé du bug
                body_component = min(
                    1.0,
                    max(
                        0.0, 1.0 - safe_divide(body_size, range_total) / max_body_ratio
                    ),
                )

                components = np.array(
                    [upper_shadow_score, lower_shadow_score, body_component]
                )
                weights = np.array([0.5, 0.3, 0.2])
                base_score = calculate_weighted_score(components, weights)

                # CORRECTION P2: suppression du lookahead (close_prices[i+1] non disponible à t=i)
                # Le score de base est assigné directement sans confirmation future
                signals[i] = base_score

        return signals

    @njit
    def detect_dragonfly_doji_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        max_body_ratio: float,
        min_lower_shadow: float,
        max_upper_shadow: float,
        min_range_ratio: float,
        proximity_threshold: float,
        body_weight: float,
        lower_weight: float,
        upper_weight: float,
        proximity_weight: float,
    ) -> np.ndarray:
        """Détecte le pattern Dragonfly Doji.

        Doji avec longue ombre basse et corps minimal.
        """
        signals = np.zeros(len(open_prices), dtype=np.float64)

        for i in range(1, len(open_prices)):
            open_val = open_prices[i]
            high_val = high_prices[i]
            low_val = low_prices[i]
            close_val = close_prices[i]

            if not validate_price_data(open_val, high_val, low_val, close_val):
                continue

            # Calculs des métriques
            body_ratio = calculate_body_ratio(open_val, high_val, low_val, close_val)
            lower_shadow_ratio = calculate_lower_shadow_ratio(
                open_val, high_val, low_val, close_val
            )
            upper_shadow_ratio = calculate_upper_shadow_ratio(
                open_val, high_val, low_val, close_val
            )
            range_ratio = calculate_range_ratio(high_val, low_val, close_val)

            # Validation des conditions
            if (
                body_ratio > max_body_ratio
                or lower_shadow_ratio < min_lower_shadow
                or upper_shadow_ratio > max_upper_shadow
                or range_ratio < min_range_ratio
            ):
                continue

            # Vérification de la proximité safe_divide(open, close)
            price_range = high_val - low_val
            proximity_score = 1.0
            if price_range > 0:
                open_close_diff = abs(safe_divide((open_val - close_val), price_range))
                if open_close_diff <= proximity_threshold:
                    proximity_score = 1.0 - safe_divide(
                        open_close_diff, proximity_threshold
                    )
                else:
                    continue

            # Calcul du score pondéré
            body_score = safe_divide((max_body_ratio - body_ratio), max_body_ratio)
            lower_score = min(1.0, safe_divide(lower_shadow_ratio, min_lower_shadow))
            upper_score = (
                safe_divide((max_upper_shadow - upper_shadow_ratio), max_upper_shadow)
                if max_upper_shadow > 0
                else 1.0
            )

            components = np.array(
                [body_score, lower_score, upper_score, proximity_score]
            )
            weights = np.array(
                [body_weight, lower_weight, upper_weight, proximity_weight]
            )

            signals[i] = calculate_weighted_score(components, weights)

        return signals

    @njit
    def detect_pin_bar_bull_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        max_body_ratio: float,
        min_lower_shadow: float,
        max_upper_shadow: float,
        min_range_ratio: float,
        body_weight: float,
        lower_weight: float,
        upper_weight: float,
        position_weight: float,
        context_weight: float,
    ) -> np.ndarray:
        """Détecte le pattern Pin Bar haussier.

        Petit corps avec longue ombre basse, corps dans la partie haute du range.

        CORRECTION S4 :
        Ajout du contexte baissier préalable — un Pin Bar haussier n'est pertinent
        que si le marché arrive d'une baisse (rejet d'un creux, mèche vers le bas).
        Sans ce contexte, n'importe quelle bougie à longue mèche basse est acceptée,
        même en pleine tendance haussière où le signal n'a aucune valeur de retournement.

        Logique : position du close dans le range [low, high] des 10 dernières barres.
        - Close proche du bas du range → marché baissier → context_score élevé ✓
        - Close proche du haut du range → marché haussier → context_score faible ✗
        [REF-PIN-BAR-BEAR] Symétrique inverse de detect_pin_bar_bear_numba.
        """
        signals = np.zeros(len(open_prices), dtype=np.float64)
        min_trend_lookback = 10

        for i in range(min_trend_lookback, len(open_prices)):
            open_val = open_prices[i]
            high_val = high_prices[i]
            low_val = low_prices[i]
            close_val = close_prices[i]

            if not validate_price_data(open_val, high_val, low_val, close_val):
                continue

            # Filtre range minimum — évite les barres plates parasites
            range_ratio = calculate_range_ratio(high_val, low_val, close_val)
            if range_ratio < min_range_ratio:
                continue

            # Métriques de la bougie
            body_ratio = calculate_body_ratio(open_val, high_val, low_val, close_val)
            lower_shadow_ratio = calculate_lower_shadow_ratio(
                open_val, high_val, low_val, close_val
            )
            upper_shadow_ratio = calculate_upper_shadow_ratio(
                open_val, high_val, low_val, close_val
            )

            # Filtres durs morphologiques
            if (
                body_ratio > max_body_ratio
                or lower_shadow_ratio < min_lower_shadow
                or upper_shadow_ratio > max_upper_shadow
            ):
                continue

            # Position du corps dans la bougie
            # Pour un pin bar haussier, le corps doit être dans la partie HAUTE
            total_range = high_val - low_val
            if total_range == 0:
                continue

            body_top = max(open_val, close_val)
            body_position = safe_divide(body_top - low_val, total_range)
            # body_position proche de 1.0 = corps en haut = morphologie correcte

            # ── Contexte baissier préalable (S4) ───────────────────────────────
            # On mesure la position du close dans le range des N barres précédentes.
            # Un pin bar haussier valide = rejet depuis un creux → close bas dans le range.
            lb_start = i - min_trend_lookback
            lb_high = np.max(high_prices[lb_start:i])
            lb_low = np.min(low_prices[lb_start:i])
            lb_range = lb_high - lb_low

            if lb_range > 0:
                # 0.0 = close au bas du range lookback (contexte baissier ✓)
                # 1.0 = close au haut du range lookback (contexte haussier ✗)
                relative_pos = safe_divide(close_val - lb_low, lb_range)
                context_score = 1.0 - min(1.0, max(0.0, relative_pos))
            else:
                # Range nul = marché flat → score neutre
                context_score = 0.3

            # ── Scoring ────────────────────────────────────────────────────────
            body_score = safe_divide(max_body_ratio - body_ratio, max_body_ratio)
            lower_score = min(1.0, safe_divide(lower_shadow_ratio, min_lower_shadow))
            upper_score = (
                1.0 - safe_divide(upper_shadow_ratio, max_upper_shadow)
                if max_upper_shadow > 0
                else 1.0
            )
            position_score = body_position

            components = np.array(
                [body_score, lower_score, upper_score, position_score, context_score]
            )
            weights = np.array(
                [
                    body_weight,
                    lower_weight,
                    upper_weight,
                    position_weight,
                    context_weight,
                ]
            )

            signals[i] = calculate_weighted_score(components, weights)

        return signals

    @njit
    def detect_marubozu_bull_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        min_body_ratio: float,
        max_shadow_ratio: float,
        min_range_ratio: float,
        body_weight: float,
        shadow_weight: float,
    ) -> np.ndarray:
        """Détecte le pattern Marubozu haussier.

        Corps plein sans ombres, close > open.
        """
        signals = np.zeros(len(open_prices), dtype=np.float64)

        for i in range(1, len(open_prices)):
            open_val = open_prices[i]
            high_val = high_prices[i]
            low_val = low_prices[i]
            close_val = close_prices[i]

            if not validate_price_data(open_val, high_val, low_val, close_val):
                continue

            # Doit être haussier
            if not is_bullish_candle(open_val, close_val):
                continue

            # Calculs des métriques
            body_ratio = calculate_body_ratio(open_val, high_val, low_val, close_val)
            upper_shadow_ratio = calculate_upper_shadow_ratio(
                open_val, high_val, low_val, close_val
            )
            lower_shadow_ratio = calculate_lower_shadow_ratio(
                open_val, high_val, low_val, close_val
            )
            range_ratio = calculate_range_ratio(high_val, low_val, close_val)

            # Validation des conditions
            if (
                body_ratio < min_body_ratio
                or upper_shadow_ratio > max_shadow_ratio
                or lower_shadow_ratio > max_shadow_ratio
                or range_ratio < min_range_ratio
            ):
                continue

            # Calcul du score pondéré
            body_score = min(1.0, safe_divide(body_ratio, min_body_ratio))
            shadow_score = (
                1.0
                - safe_divide(
                    max(upper_shadow_ratio, lower_shadow_ratio), max_shadow_ratio
                )
                if max_shadow_ratio > 0
                else 1.0
            )

            components = np.array([body_score, shadow_score])
            weights = np.array([body_weight, shadow_weight])

            signals[i] = calculate_weighted_score(components, weights)

        return signals

    @njit
    def detect_belt_hold_bull_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        min_body_ratio: float,
        max_lower_shadow: float,
        min_range_ratio: float,
        body_weight: float,
        shadow_weight: float,
    ) -> np.ndarray:
        """Détecte le pattern Belt Hold haussier.

        Ouverture au plus bas, corps long haussier, ombre basse minimale.
        """
        signals = np.zeros(len(open_prices), dtype=np.float64)

        for i in range(1, len(open_prices)):
            open_val = open_prices[i]
            high_val = high_prices[i]
            low_val = low_prices[i]
            close_val = close_prices[i]

            if not validate_price_data(open_val, high_val, low_val, close_val):
                continue

            # Doit être haussier
            if not is_bullish_candle(open_val, close_val):
                continue

            # Calculs des métriques
            body_ratio = calculate_body_ratio(open_val, high_val, low_val, close_val)
            lower_shadow_ratio = calculate_lower_shadow_ratio(
                open_val, high_val, low_val, close_val
            )
            range_ratio = calculate_range_ratio(high_val, low_val, close_val)

            # Validation des conditions
            if (
                body_ratio < min_body_ratio
                or lower_shadow_ratio > max_lower_shadow
                or range_ratio < min_range_ratio
            ):
                continue

            # Vérification que l'ouverture est proche du plus bas
            total_range = high_val - low_val
            if total_range == 0:
                continue

            open_to_low = abs(safe_divide((open_val - low_val), total_range))
            if open_to_low > max_lower_shadow:
                continue

            # Calcul du score pondéré
            body_score = min(1.0, safe_divide(body_ratio, min_body_ratio))
            shadow_score = (
                1.0 - safe_divide(lower_shadow_ratio, max_lower_shadow)
                if max_lower_shadow > 0
                else 1.0
            )

            components = np.array([body_score, shadow_score])
            weights = np.array([body_weight, shadow_weight])

            signals[i] = calculate_weighted_score(components, weights)

        return signals

    @njit
    def detect_morning_star_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        min_body_ratio_1: float,
        max_body_ratio_2: float,
        min_body_ratio_3: float,
        body_weight_1: float,
        star_weight: float,
        body_weight_3: float,
        gap_weight: float,
    ) -> np.ndarray:
        """Détecte le pattern Morning Star (3 bougies).

        Bougie 1: baissière avec gros corps
        Bougie 2: petit corps (star) avec gap down
        Bougie 3: haussière avec gros corps, gap up.
        """
        signals = np.zeros(len(open_prices), dtype=np.float64)

        for i in range(2, len(open_prices)):
            # Données des 3 bougies
            open1, high1, low1, close1 = (
                open_prices[i - 2],
                high_prices[i - 2],
                low_prices[i - 2],
                close_prices[i - 2],
            )
            open2, high2, low2, close2 = (
                open_prices[i - 1],
                high_prices[i - 1],
                low_prices[i - 1],
                close_prices[i - 1],
            )
            open3, high3, low3, close3 = (
                open_prices[i],
                high_prices[i],
                low_prices[i],
                close_prices[i],
            )

            # Validation des données
            if (
                not validate_price_data(open1, high1, low1, close1)
                or not validate_price_data(open2, high2, low2, close2)
                or not validate_price_data(open3, high3, low3, close3)
            ):
                continue

            # Bougie 1: doit être baissière avec gros corps
            if not is_bearish_candle(open1, close1):
                continue

            body_ratio_1 = calculate_body_ratio(open1, high1, low1, close1)
            if body_ratio_1 < min_body_ratio_1:
                continue

            # Bougie 2: petit corps (star)
            body_ratio_2 = calculate_body_ratio(open2, high2, low2, close2)
            if body_ratio_2 > max_body_ratio_2:
                continue

            # Bougie 3: doit être haussière avec gros corps
            if not is_bullish_candle(open3, close3):
                continue

            body_ratio_3 = calculate_body_ratio(open3, high3, low3, close3)
            if body_ratio_3 < min_body_ratio_3:
                continue

            # Vérification Nison : bougie 3 doit clôturer AU-DESSUS du mi-corps de bougie 1
            mid_body_1 = (open1 + close1) / 2.0
            if close3 <= mid_body_1:
                continue

            # Vérification des gaps
            gap_down = close1 > high2  # Gap entre bougie 1 et 2
            gap_up = low3 > close2  # Gap entre bougie 2 et 3

            # Calcul du score pondéré
            body_score_1 = min(1.0, safe_divide(body_ratio_1, min_body_ratio_1))
            star_score = (
                safe_divide((max_body_ratio_2 - body_ratio_2), max_body_ratio_2)
                if max_body_ratio_2 > 0
                else 1.0
            )
            body_score_3 = min(1.0, safe_divide(body_ratio_3, min_body_ratio_3))
            # P0-5 FIX: Supprimé le plancher 0.5 — sans gap, score = 0.0 (pas 0.5)
            gap_score = 0.5 * (1.0 if gap_down else 0.0) + 0.5 * (
                1.0 if gap_up else 0.0
            )

            components = np.array([body_score_1, star_score, body_score_3, gap_score])
            weights = np.array([body_weight_1, star_weight, body_weight_3, gap_weight])

            signals[i] = calculate_weighted_score(components, weights)

        return signals

    @njit
    def detect_piercing_line_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        min_body_ratio: float,
        min_penetration: float,
        body_weight_1: float,
        body_weight_2: float,
        penetration_weight: float,
    ) -> np.ndarray:
        """Détecte le pattern Piercing Line (2 bougies).

        Bougie 1: baissière avec gros corps
        Bougie 2: haussière qui pénètre dans le corps de la bougie 1.
        """
        signals = np.zeros(len(open_prices), dtype=np.float64)

        for i in range(1, len(open_prices)):
            # Données des 2 bougies
            open1, high1, low1, close1 = (
                open_prices[i - 1],
                high_prices[i - 1],
                low_prices[i - 1],
                close_prices[i - 1],
            )
            open2, high2, low2, close2 = (
                open_prices[i],
                high_prices[i],
                low_prices[i],
                close_prices[i],
            )

            # Validation des données
            if not validate_price_data(
                open1, high1, low1, close1
            ) or not validate_price_data(open2, high2, low2, close2):
                continue

            # Bougie 1: doit être baissière avec gros corps
            if not is_bearish_candle(open1, close1):
                continue

            body_ratio_1 = calculate_body_ratio(open1, high1, low1, close1)
            if body_ratio_1 < min_body_ratio:
                continue

            # Bougie 2: doit être haussière avec gros corps
            if not is_bullish_candle(open2, close2):
                continue

            # AUDIT FIX Bloc1: Gap-down obligatoire (Nison) — bougie 2 s'ouvre sous la clôture de la bougie 1
            # Sans ce filtre, la Piercing Line se confond avec un Engulfing Bull ordinaire
            if open2 >= close1:
                continue

            body_ratio_2 = calculate_body_ratio(open2, high2, low2, close2)
            if body_ratio_2 < min_body_ratio:
                continue

            # Vérification de la pénétration
            body_1_size = abs(open1 - close1)
            penetration = (
                safe_divide((close2 - close1), body_1_size) if body_1_size > 0 else 0
            )

            if penetration < min_penetration:
                continue

            # Calcul du score pondéré
            body_score_1 = min(1.0, safe_divide(body_ratio_1, min_body_ratio))
            body_score_2 = min(1.0, safe_divide(body_ratio_2, min_body_ratio))
            penetration_score = min(1.0, safe_divide(penetration, min_penetration))

            components = np.array([body_score_1, body_score_2, penetration_score])
            weights = np.array([body_weight_1, body_weight_2, penetration_weight])

            signals[i] = calculate_weighted_score(components, weights)

        return signals

    @njit
    def detect_harami_bull_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        min_body_ratio_1: float,
        max_body_ratio_2: float,
        body_weight_1: float,
        body_weight_2: float,
        containment_weight: float,
    ) -> np.ndarray:
        """Détecte le pattern Bullish Harami (2 bougies).

        Bougie 1: baissière avec gros corps
        Bougie 2: petit corps contenu dans le corps de la bougie 1.
        """
        signals = np.zeros(len(open_prices), dtype=np.float64)

        for i in range(1, len(open_prices)):
            # Données des 2 bougies
            open1, high1, low1, close1 = (
                open_prices[i - 1],
                high_prices[i - 1],
                low_prices[i - 1],
                close_prices[i - 1],
            )
            open2, high2, low2, close2 = (
                open_prices[i],
                high_prices[i],
                low_prices[i],
                close_prices[i],
            )

            # Validation des données
            if not validate_price_data(
                open1, high1, low1, close1
            ) or not validate_price_data(open2, high2, low2, close2):
                continue

            # Bougie 1: doit être baissière avec gros corps
            if not is_bearish_candle(open1, close1):
                continue

            body_ratio_1 = calculate_body_ratio(open1, high1, low1, close1)
            if body_ratio_1 < min_body_ratio_1:
                continue

            # Bougie 2: petit corps
            body_ratio_2 = calculate_body_ratio(open2, high2, low2, close2)
            if body_ratio_2 > max_body_ratio_2:
                continue

            # Vérification du containment (bougie 2 contenue dans bougie 1)
            contained = (
                open2 >= close1
                and open2 <= open1
                and close2 >= close1
                and close2 <= open1
            )

            if not contained:
                continue

            # Calcul de la qualité du containment
            body_1_size = abs(open1 - close1)
            body_2_size = abs(open2 - close2)
            containment_ratio = 1.0 - (
                safe_divide(body_2_size, body_1_size) if body_1_size > 0 else 0
            )

            # Calcul du score pondéré
            body_score_1 = min(1.0, safe_divide(body_ratio_1, min_body_ratio_1))
            body_score_2 = (
                safe_divide((max_body_ratio_2 - body_ratio_2), max_body_ratio_2)
                if max_body_ratio_2 > 0
                else 1.0
            )
            containment_score = containment_ratio

            components = np.array([body_score_1, body_score_2, containment_score])
            weights = np.array([body_weight_1, body_weight_2, containment_weight])

            signals[i] = calculate_weighted_score(components, weights)

        return signals

    @njit
    def detect_abandoned_baby_bull_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        min_body_ratio_1: float,
        max_body_ratio_2: float,
        min_body_ratio_3: float,
        body_weight_1: float,
        star_weight: float,
        body_weight_3: float,
        gap_weight: float,
    ) -> np.ndarray:
        """Détecte le pattern Abandoned Baby Bull (3 bougies).

        Bougie 1 : baissière avec gros corps.
        Bougie 2 : doji/star isolée par deux gaps stricts (sans chevauchement).
        Bougie 3 : haussière avec gros corps.

        CORRECTIONS S5 :
        S5a — Vérification Nison manquante : bougie 3 doit clôturer AU-DESSUS
              du mi-corps de bougie 1. Sans ce check, un signal haussier faible
              qui ne récupère pas la moitié de la bougie baissière initiale est
              accepté à tort. [REF-MORNING-STAR] Condition identique.

        S5b — gap_score gradué : les gaps étaient validés en filtre dur puis
              scorés 1.0 fixe (poids mort à 0.25). Le score reflète maintenant
              la taille relative des deux gaps — de gros gaps sont un signal
              plus fort qu'un gap minimal (Nison, Japanese Candlestick Charting).
        """
        signals = np.zeros(len(open_prices), dtype=np.float64)

        for i in range(2, len(open_prices)):
            # ── Données des 3 bougies ──────────────────────────────────────────
            open1, high1, low1, close1 = (
                open_prices[i - 2],
                high_prices[i - 2],
                low_prices[i - 2],
                close_prices[i - 2],
            )
            open2, high2, low2, close2 = (
                open_prices[i - 1],
                high_prices[i - 1],
                low_prices[i - 1],
                close_prices[i - 1],
            )
            open3, high3, low3, close3 = (
                open_prices[i],
                high_prices[i],
                low_prices[i],
                close_prices[i],
            )

            # ── Validation OHLC ────────────────────────────────────────────────
            if (
                not validate_price_data(open1, high1, low1, close1)
                or not validate_price_data(open2, high2, low2, close2)
                or not validate_price_data(open3, high3, low3, close3)
            ):
                continue

            # ── Bougie 1 : baissière avec gros corps ───────────────────────────
            if not is_bearish_candle(open1, close1):
                continue

            body_ratio_1 = calculate_body_ratio(open1, high1, low1, close1)
            if body_ratio_1 < min_body_ratio_1:
                continue

            # ── Bougie 2 : doji/star (petit corps, haussier ou baissier) ───────
            body_ratio_2 = calculate_body_ratio(open2, high2, low2, close2)
            if body_ratio_2 > max_body_ratio_2:
                continue

            # ── Bougie 3 : haussière avec gros corps ───────────────────────────
            if not is_bullish_candle(open3, close3):
                continue

            body_ratio_3 = calculate_body_ratio(open3, high3, low3, close3)
            if body_ratio_3 < min_body_ratio_3:
                continue

            # ── Gaps stricts — aucun chevauchement autorisé ────────────────────
            # Abandoned Baby est plus strict que Morning Star :
            # le haut de la star (high2) doit être sous le bas de la bougie 1 (low1)
            # et le bas de la bougie 3 (low3) doit être au-dessus du haut de la star (high2)
            gap_down = (
                high2 < low1
            )  # gap baissier strict : star isolée sous la bougie 1
            gap_up = (
                low3 > high2
            )  # gap haussier strict : bougie 3 isolée au-dessus de la star

            if not (gap_down and gap_up):
                continue

            # ── S5a : Vérification Nison — clôture au-dessus du mi-corps ───────
            # La bougie 3 doit récupérer plus de la moitié du corps de bougie 1.
            # Sans ce filtre, un signal haussier faible (qui ne récupère rien)
            # est accepté comme pattern de retournement valide — ce qui est faux.
            mid_body_1 = (open1 + close1) / 2.0
            if close3 <= mid_body_1:
                continue

            # ── S5b : gap_score gradué ─────────────────────────────────────────
            # Taille relative des deux gaps normalisée.
            # Un gap de 0.5% = score 1.0 ; un gap minimal (juste > 0) = score proche de 0.
            # low1 > 0 et high2 > 0 sont garantis par validate_price_data.
            gap1_size = safe_divide(low1 - high2, low1)  # gap baissier bougie 1 → star
            gap2_size = safe_divide(low3 - high2, high2)  # gap haussier star → bougie 3
            avg_gap = (gap1_size + gap2_size) / 2.0
            # Normalisé sur 0.5% de gap moyen = score 1.0
            gap_score = min(1.0, safe_divide(avg_gap, 0.005))

            # ── Scoring ────────────────────────────────────────────────────────
            body_score_1 = min(1.0, safe_divide(body_ratio_1, min_body_ratio_1))
            star_score = (
                safe_divide(max_body_ratio_2 - body_ratio_2, max_body_ratio_2)
                if max_body_ratio_2 > 0
                else 1.0
            )
            body_score_3 = min(1.0, safe_divide(body_ratio_3, min_body_ratio_3))

            components = np.array([body_score_1, star_score, body_score_3, gap_score])
            weights = np.array([body_weight_1, star_weight, body_weight_3, gap_weight])

            signals[i] = calculate_weighted_score(components, weights)

        return signals

    @njit
    def detect_three_inside_up_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        min_body_ratio_1: float,
        max_body_ratio_2: float,
        min_body_ratio_3: float,
        body_weight_1: float,
        body_weight_2: float,
        body_weight_3: float,
    ) -> np.ndarray:
        """Détecte le pattern Three Inside Up (3 bougies).

        Harami haussier suivi d'une confirmation haussière.
        """
        signals = np.zeros(len(open_prices), dtype=np.float64)

        for i in range(2, len(open_prices)):
            # Données des 3 bougies
            open1, high1, low1, close1 = (
                open_prices[i - 2],
                high_prices[i - 2],
                low_prices[i - 2],
                close_prices[i - 2],
            )
            open2, high2, low2, close2 = (
                open_prices[i - 1],
                high_prices[i - 1],
                low_prices[i - 1],
                close_prices[i - 1],
            )
            open3, high3, low3, close3 = (
                open_prices[i],
                high_prices[i],
                low_prices[i],
                close_prices[i],
            )

            # Validation des données
            if (
                not validate_price_data(open1, high1, low1, close1)
                or not validate_price_data(open2, high2, low2, close2)
                or not validate_price_data(open3, high3, low3, close3)
            ):
                continue

            # Bougie 1: baissière avec gros corps
            if not is_bearish_candle(open1, close1):
                continue

            body_ratio_1 = calculate_body_ratio(open1, high1, low1, close1)
            if body_ratio_1 < min_body_ratio_1:
                continue

            # Bougie 2: petit corps haussier contenu dans bougie 1
            if not is_bullish_candle(open2, close2):
                continue

            body_ratio_2 = calculate_body_ratio(open2, high2, low2, close2)
            if body_ratio_2 > max_body_ratio_2:
                continue

            # Vérification du containment
            contained = open2 >= close1 and close2 <= open1
            if not contained:
                continue

            # Bougie 3: haussière qui clôture au-dessus de la bougie 1
            if not is_bullish_candle(open3, close3):
                continue

            body_ratio_3 = calculate_body_ratio(open3, high3, low3, close3)
            if body_ratio_3 < min_body_ratio_3:
                continue

            # Confirmation: clôture au-dessus de l'ouverture de la bougie 1
            if close3 <= open1:
                continue

            # Calcul du score pondéré
            body_score_1 = min(1.0, safe_divide(body_ratio_1, min_body_ratio_1))
            body_score_2 = (
                safe_divide((max_body_ratio_2 - body_ratio_2), max_body_ratio_2)
                if max_body_ratio_2 > 0
                else 1.0
            )
            body_score_3 = min(1.0, safe_divide(body_ratio_3, min_body_ratio_3))

            components = np.array([body_score_1, body_score_2, body_score_3])
            weights = np.array([body_weight_1, body_weight_2, body_weight_3])

            signals[i] = calculate_weighted_score(components, weights)

        return signals

    @njit
    def detect_three_outside_up_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        min_body_ratio: float,
        body_weight_1: float,
        body_weight_2: float,
        body_weight_3: float,
    ) -> np.ndarray:
        """Détecte le pattern Three Outside Up (3 bougies).

        Engulfing haussier suivi d'une confirmation haussière.
        """
        signals = np.zeros(len(open_prices), dtype=np.float64)

        for i in range(2, len(open_prices)):
            # Données des 3 bougies
            open1, high1, low1, close1 = (
                open_prices[i - 2],
                high_prices[i - 2],
                low_prices[i - 2],
                close_prices[i - 2],
            )
            open2, high2, low2, close2 = (
                open_prices[i - 1],
                high_prices[i - 1],
                low_prices[i - 1],
                close_prices[i - 1],
            )
            open3, high3, low3, close3 = (
                open_prices[i],
                high_prices[i],
                low_prices[i],
                close_prices[i],
            )

            # Validation des données
            if (
                not validate_price_data(open1, high1, low1, close1)
                or not validate_price_data(open2, high2, low2, close2)
                or not validate_price_data(open3, high3, low3, close3)
            ):
                continue

            # Bougies 1 et 2: pattern engulfing haussier
            if not is_bearish_candle(open1, close1) or not is_bullish_candle(
                open2, close2
            ):
                continue

            body_ratio_1 = calculate_body_ratio(open1, high1, low1, close1)
            body_ratio_2 = calculate_body_ratio(open2, high2, low2, close2)

            if body_ratio_1 < min_body_ratio or body_ratio_2 < min_body_ratio:
                continue

            # Vérification de l'englobement
            engulfs = open2 < close1 and close2 > open1
            if not engulfs:
                continue

            # Bougie 3: confirmation haussière
            if not is_bullish_candle(open3, close3):
                continue

            body_ratio_3 = calculate_body_ratio(open3, high3, low3, close3)

            # AUDIT FIX Bloc1: Hard filter sur la bougie de confirmation (Nison)
            # Sans ce filtre, une doji en position 3 valide le signal — ce qui est faux
            if body_ratio_3 < min_body_ratio:
                continue

            # Confirmation: clôture au-dessus de la bougie 2
            if close3 <= close2:
                continue

            # Calcul du score pondéré
            body_score_1 = min(1.0, safe_divide(body_ratio_1, min_body_ratio))
            body_score_2 = min(1.0, safe_divide(body_ratio_2, min_body_ratio))
            body_score_3 = min(
                1.0,
                safe_divide(body_ratio_3, min_body_ratio)
                if min_body_ratio > 0
                else 1.0,
            )

            components = np.array([body_score_1, body_score_2, body_score_3])
            weights = np.array([body_weight_1, body_weight_2, body_weight_3])

            signals[i] = calculate_weighted_score(components, weights)

        return signals

    @njit
    def detect_concealing_baby_swallow_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        min_body_ratio: float,
        max_upper_shadow: float,
        body_weight: float,
        shadow_weight: float,
    ) -> np.ndarray:
        """Détecte le pattern Concealing Baby Swallow (4 bougies).

        Pattern rare de retournement haussier.
        """
        signals = np.zeros(len(open_prices), dtype=np.float64)

        for i in range(3, len(open_prices)):
            # Données des 4 bougies
            open1, high1, low1, close1 = (
                open_prices[i - 3],
                high_prices[i - 3],
                low_prices[i - 3],
                close_prices[i - 3],
            )
            open2, high2, low2, close2 = (
                open_prices[i - 2],
                high_prices[i - 2],
                low_prices[i - 2],
                close_prices[i - 2],
            )
            open3, high3, low3, close3 = (
                open_prices[i - 1],
                high_prices[i - 1],
                low_prices[i - 1],
                close_prices[i - 1],
            )
            open4, high4, low4, close4 = (
                open_prices[i],
                high_prices[i],
                low_prices[i],
                close_prices[i],
            )

            # Validation des données
            if (
                not validate_price_data(open1, high1, low1, close1)
                or not validate_price_data(open2, high2, low2, close2)
                or not validate_price_data(open3, high3, low3, close3)
                or not validate_price_data(open4, high4, low4, close4)
            ):
                continue

            # Toutes les bougies doivent être baissières sauf la dernière
            if (
                not is_bearish_candle(open1, close1)
                or not is_bearish_candle(open2, close2)
                or not is_bearish_candle(open3, close3)
            ):
                continue

            # La 4ème bougie doit être haussière et englober la 3ème
            if not is_bullish_candle(open4, close4):
                continue

            # AUDIT FIX Bloc1: Remplacement de la liste Python body_ratios = [...]
            # par des variables scalaires directes (Numba incompatible avec list Python)
            _body1 = calculate_body_ratio(open1, high1, low1, close1)
            _body2 = calculate_body_ratio(open2, high2, low2, close2)
            _body3 = calculate_body_ratio(open3, high3, low3, close3)
            _body4 = calculate_body_ratio(open4, high4, low4, close4)

            # Vérification que tous les corps sont >= min_body_ratio
            _all_bodies_ok = (
                _body1 >= min_body_ratio
                and _body2 >= min_body_ratio
                and _body3 >= min_body_ratio
                and _body4 >= min_body_ratio
            )

            if _all_bodies_ok:
                # Vérification des ombres hautes
                _shadow1 = calculate_upper_shadow_ratio(open1, high1, low1, close1)
                _shadow2 = calculate_upper_shadow_ratio(open2, high2, low2, close2)
                _shadow3 = calculate_upper_shadow_ratio(open3, high3, low3, close3)
                _shadow4 = calculate_upper_shadow_ratio(open4, high4, low4, close4)

                max_shadow = max(_shadow1, max(_shadow2, max(_shadow3, _shadow4)))
                if max_shadow <= max_upper_shadow:
                    # Vérification de l'englobement de la bougie 4 sur la bougie 3
                    if open4 < close3 and close4 > open3:
                        # AUDIT FIX: Utiliser les variables _bodyN calculées plus haut
                        total_body = _body1 + _body2 + _body3 + _body4
                        avg_body_ratio = safe_divide(total_body, 4.0)
                        body_score = min(
                            1.0, safe_divide(avg_body_ratio, min_body_ratio)
                        )
                        # P0-2 FIX: Remplacer sum(upper_shadows) par accumulation manuelle
                        avg_upper_shadow = safe_divide(
                            _shadow1 + _shadow2 + _shadow3 + _shadow4, 4.0
                        )
                        shadow_score = (
                            1.0 - safe_divide(avg_upper_shadow, max_upper_shadow)
                            if max_upper_shadow > 0
                            else 1.0
                        )

                        components = np.array([body_score, shadow_score])
                        weights = np.array([body_weight, shadow_weight])

                        signals[i] = calculate_weighted_score(components, weights)

        return signals

    @njit
    def detect_unique_three_river_bottom_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        min_body_ratio: float,
        max_body_ratio_2: float,
        body_weight: float,
        pattern_weight: float,
    ) -> np.ndarray:
        """Détecte le pattern Unique Three River Bottom (3 bougies).

        Pattern rare de retournement haussier.
        """
        signals = np.zeros(len(open_prices), dtype=np.float64)

        for i in range(2, len(open_prices)):
            # Données des 3 bougies
            open1, high1, low1, close1 = (
                open_prices[i - 2],
                high_prices[i - 2],
                low_prices[i - 2],
                close_prices[i - 2],
            )
            open2, high2, low2, close2 = (
                open_prices[i - 1],
                high_prices[i - 1],
                low_prices[i - 1],
                close_prices[i - 1],
            )
            open3, high3, low3, close3 = (
                open_prices[i],
                high_prices[i],
                low_prices[i],
                close_prices[i],
            )

            # Validation des données
            if (
                not validate_price_data(open1, high1, low1, close1)
                or not validate_price_data(open2, high2, low2, close2)
                or not validate_price_data(open3, high3, low3, close3)
            ):
                continue

            # Bougie 1: baissière avec gros corps
            if not is_bearish_candle(open1, close1):
                continue

            body_ratio_1 = calculate_body_ratio(open1, high1, low1, close1)
            if body_ratio_1 < min_body_ratio:
                continue

            # Bougie 2: petit corps (doji-like)
            body_ratio_2 = calculate_body_ratio(open2, high2, low2, close2)
            if body_ratio_2 > max_body_ratio_2:
                continue

            # P1-3 FIX: Bougie 3 doit être HAUSSIÈRE (confirmation du retournement)
            # La définition du pattern Unique Three River Bottom exige une bougie haussière en 3e position
            if not is_bullish_candle(open3, close3):
                continue

            body_ratio_3 = calculate_body_ratio(open3, high3, low3, close3)
            # Bougie 3 doit avoir un petit corps (< 30%) — Nison exacte
            if body_ratio_3 >= 0.30:
                continue

            # Conditions Nison exactes : low3 < low2 (river bottom) + close3 < open2
            if low3 >= low2:
                continue
            if close3 >= open2:
                continue

            # Conditions spéciales du pattern
            # La 3e bougie doit clôturer au-dessus de la clôture de la 2e (renforcement haussier)
            if close3 <= close2:
                continue

            # Calcul du score pondéré
            body_score = min(1.0, safe_divide(body_ratio_1, min_body_ratio))
            pattern_score = (
                safe_divide((max_body_ratio_2 - body_ratio_2), max_body_ratio_2)
                if max_body_ratio_2 > 0
                else 1.0
            )

            components = np.array([body_score, pattern_score])
            weights = np.array([body_weight, pattern_weight])

            signals[i] = calculate_weighted_score(components, weights)

        return signals

    @njit
    def detect_matching_low_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        min_body_ratio: float,
        max_close_diff: float,
        min_range_ratio: float,
        body_weight: float,
        matching_weight: float,
    ) -> np.ndarray:
        """Détecte le pattern Matching Low (2 bougies).

        Deux bougies baissières avec clôtures similaires.
        """
        signals = np.zeros(len(open_prices), dtype=np.float64)

        for i in range(1, len(open_prices)):
            # Données des 2 bougies
            open1, high1, low1, close1 = (
                open_prices[i - 1],
                high_prices[i - 1],
                low_prices[i - 1],
                close_prices[i - 1],
            )
            open2, high2, low2, close2 = (
                open_prices[i],
                high_prices[i],
                low_prices[i],
                close_prices[i],
            )

            # Validation des données
            if not validate_price_data(
                open1, high1, low1, close1
            ) or not validate_price_data(open2, high2, low2, close2):
                continue

            # Les deux bougies doivent être baissières
            if not is_bearish_candle(open1, close1) or not is_bearish_candle(
                open2, close2
            ):
                continue

            # Vérification des corps
            body_ratio_1 = calculate_body_ratio(open1, high1, low1, close1)
            body_ratio_2 = calculate_body_ratio(open2, high2, low2, close2)

            if body_ratio_1 < min_body_ratio or body_ratio_2 < min_body_ratio:
                continue

            # Vérification des ranges
            range_ratio_1 = calculate_range_ratio(high1, low1, close1)
            range_ratio_2 = calculate_range_ratio(high2, low2, close2)

            if range_ratio_1 < min_range_ratio or range_ratio_2 < min_range_ratio:
                continue

            # Vérification de la similarité des clôtures
            close_diff = abs(
                safe_divide(
                    close1 - close2,
                    max(close1, close2) if max(close1, close2) > 0 else 1.0,
                )
            )
            if close_diff > max_close_diff:
                continue

            # Calcul du score pondéré
            avg_body_ratio = safe_divide((body_ratio_1 + body_ratio_2), 2)
            body_score = min(1.0, safe_divide(avg_body_ratio, min_body_ratio))
            matching_score = (
                1.0 - safe_divide(close_diff, max_close_diff)
                if max_close_diff > 0
                else 1.0
            )

            components = np.array([body_score, matching_score])
            weights = np.array([body_weight, matching_weight])

            signals[i] = calculate_weighted_score(components, weights)

        return signals

    @njit
    def detect_ladder_bottom_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        min_body_ratio: float,
        min_pattern_length: float,
        max_upper_shadow: float,
        body_weight: float,
        pattern_weight: float,
        shadow_weight: float,
    ) -> np.ndarray:
        """Détecte le pattern Ladder Bottom (5 bougies).

        Séquence de 5 bougies formant un escalier descendant puis retournement.
        """
        signals = np.zeros(len(open_prices), dtype=np.float64)

        for i in range(4, len(open_prices)):
            # AUDIT FIX: Remplacer listes Python+tuples par accès directs aux arrays
            # Validation des données des 5 bougies (accès direct par index)
            valid = True
            for j in range(5):
                idx = i - 4 + j
                if not validate_price_data(
                    open_prices[idx],
                    high_prices[idx],
                    low_prices[idx],
                    close_prices[idx],
                ):
                    valid = False
                    break
            if not valid:
                continue

            # Les 4 premières bougies doivent être baissières
            for j in range(4):
                idx = i - 4 + j
                if not is_bearish_candle(open_prices[idx], close_prices[idx]):
                    valid = False
                    break
            if not valid:
                continue

            # La 5ème bougie doit être haussière
            if not is_bullish_candle(open_prices[i], close_prices[i]):
                continue

            # AUDIT FIX: Accumulation scalaire (pas de liste body_ratios)
            total_body = 0.0
            total_upper_shadow = 0.0
            for j in range(5):
                idx = i - 4 + j
                body_ratio = calculate_body_ratio(
                    open_prices[idx],
                    high_prices[idx],
                    low_prices[idx],
                    close_prices[idx],
                )
                if body_ratio < min_body_ratio:
                    valid = False
                    break
                upper_shadow = calculate_upper_shadow_ratio(
                    open_prices[idx],
                    high_prices[idx],
                    low_prices[idx],
                    close_prices[idx],
                )
                if upper_shadow > max_upper_shadow:
                    valid = False
                    break
                total_body += body_ratio
                total_upper_shadow += upper_shadow

            if not valid:
                continue

            # AUDIT FIX: Progression descendante (accès direct, pas de np.array([...for...]))
            descending = True
            for j in range(3):
                if close_prices[i - 4 + j] <= close_prices[i - 4 + j + 1]:
                    descending = False
                    break

            reversal = close_prices[i] > close_prices[i - 1]

            if not (descending and reversal):
                continue

            # AUDIT FIX: sum() builtin supprimé — accumulation scalaire utilisée
            avg_body_ratio = safe_divide(total_body, 5.0)
            body_score = min(1.0, safe_divide(avg_body_ratio, min_body_ratio))

            pattern_score = 1.0  # Pattern validé

            avg_upper_shadow = safe_divide(total_upper_shadow, 5.0)
            shadow_score = (
                1.0 - safe_divide(avg_upper_shadow, max_upper_shadow)
                if max_upper_shadow > 0
                else 1.0
            )

            components = np.array([body_score, pattern_score, shadow_score])
            weights = np.array([body_weight, pattern_weight, shadow_weight])

            signals[i] = calculate_weighted_score(components, weights)

        return signals

    @njit
    def detect_breakaway_bull_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        min_gap_ratio: float,
        min_continuation: int,
        min_volume_surge: float,
        gap_weight: float,
        continuation_weight: float,
        volume_weight: float,
    ) -> np.ndarray:
        """VERSION SANS LOOKAHEAD."""
        signals = np.zeros(len(open_prices), dtype=np.float64)
        for i in range(15, len(open_prices)):
            gap_up = open_prices[i] > high_prices[i - 1]
            if not gap_up:
                continue

            gap_ratio = safe_divide(
                (open_prices[i] - high_prices[i - 1]), high_prices[i - 1]
            )
            if gap_ratio < min_gap_ratio:
                continue

            prices_before = close_prices[i - 10 : i]
            trend = safe_divide(
                (prices_before[-1] - prices_before[0]), prices_before[0]
            )
            if trend < 0:  # Doit s'inscrire dans une dynamique haussière ou plate
                continue

            avg_vol = np.mean(volume[i - 10 : i])
            vol_score = 0.0
            if avg_vol > 0:
                vol_ratio = safe_divide(volume[i], avg_vol)
                if vol_ratio > min_volume_surge:
                    vol_score = min(1.0, safe_divide(vol_ratio, min_volume_surge * 2))

            if vol_score == 0:
                continue

            gap_score = min(1.0, safe_divide(gap_ratio, min_gap_ratio * 2))

            # AUDIT FIX C10: min_continuation était un paramètre mort (1.0 hardcodé).
            # continuation_score mesure la force de la tendance haussière:
            # trend (retour 10 barres) normalisé par min_continuation (seuil %) — plafonné à 1.0
            continuation_score = min(1.0, safe_divide(trend, min_continuation * 0.01))

            components = np.array([gap_score, continuation_score, vol_score])
            weights = np.array([gap_weight, continuation_weight, volume_weight])
            signals[i] = calculate_weighted_score(components, weights)
        return signals

    # === FONCTIONS DE DÉTECTION DES PATTERNS BAISSIERS ===

    @njit
    def detect_hanging_man_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        max_body_ratio: float,
        min_lower_shadow: float,
        max_upper_shadow: float,
        min_range_ratio: float,
        body_weight: float,
        lower_weight: float,
        upper_weight: float,
        context_weight: float,
    ) -> np.ndarray:
        """Détecte le pattern Hanging Man.

        Pattern baissier avec petit corps et longue ombre basse, apparaît en contexte haussier.
        """
        signals = np.zeros(len(open_prices), dtype=np.float64)

        for i in range(1, len(open_prices)):
            open_val = open_prices[i]
            high_val = high_prices[i]
            low_val = low_prices[i]
            close_val = close_prices[i]

            # Validation des données
            if not validate_price_data(open_val, high_val, low_val, close_val):
                continue

            # Calculs des métriques
            body_ratio = calculate_body_ratio(open_val, high_val, low_val, close_val)
            lower_shadow_ratio = calculate_lower_shadow_ratio(
                open_val, high_val, low_val, close_val
            )
            upper_shadow_ratio = calculate_upper_shadow_ratio(
                open_val, high_val, low_val, close_val
            )
            range_ratio = calculate_range_ratio(high_val, low_val, close_val)

            # Validation des conditions
            if (
                body_ratio > max_body_ratio
                or lower_shadow_ratio < min_lower_shadow
                or upper_shadow_ratio > max_upper_shadow
                or range_ratio < min_range_ratio
            ):
                continue

            # AUDIT FIX C2: context_score basé sur la position dans le range lookback
            # (close proche du high_lookback = contexte haussier confirmé)
            min_trend_lookback = 10
            context_score = 0.5  # valeur par défaut neutre
            if i >= min_trend_lookback:
                lookback_start = i - min_trend_lookback
                lookback_high = np.max(high_prices[lookback_start:i])
                lookback_low = np.min(low_prices[lookback_start:i])
                lookback_range = lookback_high - lookback_low
                if lookback_range > 0:
                    # Score = position du close dans le range [low, high] du lookback
                    context_score = safe_divide(
                        close_val - lookback_low, lookback_range
                    )
                    context_score = min(1.0, max(0.0, context_score))
                else:
                    context_score = 0.3

            # Calcul du score pondéré
            body_score = safe_divide((max_body_ratio - body_ratio), max_body_ratio)
            lower_score = min(1.0, safe_divide(lower_shadow_ratio, min_lower_shadow))
            upper_score = (
                safe_divide((max_upper_shadow - upper_shadow_ratio), max_upper_shadow)
                if max_upper_shadow > 0
                else 1.0
            )

            components = np.array([body_score, lower_score, upper_score, context_score])
            weights = np.array(
                [body_weight, lower_weight, upper_weight, context_weight]
            )

            signals[i] = calculate_weighted_score(components, weights)

        return signals

    @njit
    def detect_shooting_star_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        max_body_ratio: float,
        min_upper_shadow: float,
        max_lower_shadow: float,
        min_range_ratio: float,
        body_weight: float,
        upper_weight: float,
        lower_weight: float,
    ) -> np.ndarray:
        """Détecte le pattern Shooting Star.

        Pattern baissier avec petit corps et longue ombre haute.
        """
        signals = np.zeros(len(open_prices), dtype=np.float64)

        for i in range(1, len(open_prices)):
            open_val = open_prices[i]
            high_val = high_prices[i]
            low_val = low_prices[i]
            close_val = close_prices[i]

            # Validation des données
            if not validate_price_data(open_val, high_val, low_val, close_val):
                continue

            # Calculs des métriques
            body_ratio = calculate_body_ratio(open_val, high_val, low_val, close_val)
            upper_shadow_ratio = calculate_upper_shadow_ratio(
                open_val, high_val, low_val, close_val
            )
            lower_shadow_ratio = calculate_lower_shadow_ratio(
                open_val, high_val, low_val, close_val
            )
            range_ratio = calculate_range_ratio(high_val, low_val, close_val)

            # Validation des conditions
            if (
                body_ratio > max_body_ratio
                or upper_shadow_ratio < min_upper_shadow
                or lower_shadow_ratio > max_lower_shadow
                or range_ratio < min_range_ratio
            ):
                continue

            # AUDIT FIX C8: Contexte haussier préalable — tir en l'air depuis un sommet
            # [REF-HANGING-MAN] Même logique lookback-range que detect_hanging_man_numba
            min_trend_lookback = 10
            context_score = 0.5  # valeur neutre si pas assez de données
            if i >= min_trend_lookback:
                lb_start = i - min_trend_lookback
                lb_high = np.max(high_prices[lb_start:i])
                lb_low = np.min(low_prices[lb_start:i])
                lb_range = lb_high - lb_low
                if lb_range > 0:
                    # Shooting star = corps/clôture proche du haut = contexte haussier confirmé
                    context_score = safe_divide(close_val - lb_low, lb_range)
                    context_score = min(1.0, max(0.0, context_score))
                else:
                    context_score = 0.3

            # Calcul du score pondéré
            body_score = safe_divide((max_body_ratio - body_ratio), max_body_ratio)
            upper_score = min(1.0, safe_divide(upper_shadow_ratio, min_upper_shadow))
            lower_score = (
                safe_divide((max_lower_shadow - lower_shadow_ratio), max_lower_shadow)
                if max_lower_shadow > 0
                else 1.0
            )

            # context_score intégré dans body_weight (proportionnel au poids corps)
            body_score = body_score * context_score

            components = np.array([body_score, upper_score, lower_score])
            weights = np.array([body_weight, upper_weight, lower_weight])

            signals[i] = calculate_weighted_score(components, weights)

        return signals

    @njit
    def detect_gravestone_doji_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        max_body_ratio: float,
        min_upper_shadow: float,
        max_lower_shadow: float,
        min_range_ratio: float,
        proximity_threshold: float,
        body_weight: float,
        upper_weight: float,
        lower_weight: float,
        proximity_weight: float,
    ) -> np.ndarray:
        """Détecte le pattern Gravestone Doji.

        Doji avec longue ombre haute et corps minimal.
        """
        signals = np.zeros(len(open_prices), dtype=np.float64)

        for i in range(1, len(open_prices)):
            open_val = open_prices[i]
            high_val = high_prices[i]
            low_val = low_prices[i]
            close_val = close_prices[i]

            if not validate_price_data(open_val, high_val, low_val, close_val):
                continue

            # Calculs des métriques
            body_ratio = calculate_body_ratio(open_val, high_val, low_val, close_val)
            upper_shadow_ratio = calculate_upper_shadow_ratio(
                open_val, high_val, low_val, close_val
            )
            lower_shadow_ratio = calculate_lower_shadow_ratio(
                open_val, high_val, low_val, close_val
            )
            range_ratio = calculate_range_ratio(high_val, low_val, close_val)

            # Validation des conditions
            if (
                body_ratio > max_body_ratio
                or upper_shadow_ratio < min_upper_shadow
                or lower_shadow_ratio > max_lower_shadow
                or range_ratio < min_range_ratio
            ):
                continue

            # Vérification de la proximité safe_divide(open, close)
            price_range = high_val - low_val
            proximity_score = 1.0
            if price_range > 0:
                open_close_diff = abs(safe_divide((open_val - close_val), price_range))
                if open_close_diff <= proximity_threshold:
                    proximity_score = 1.0 - safe_divide(
                        open_close_diff, proximity_threshold
                    )
                else:
                    continue

            # Calcul du score pondéré
            body_score = safe_divide((max_body_ratio - body_ratio), max_body_ratio)
            upper_score = min(1.0, safe_divide(upper_shadow_ratio, min_upper_shadow))
            lower_score = (
                safe_divide((max_lower_shadow - lower_shadow_ratio), max_lower_shadow)
                if max_lower_shadow > 0
                else 1.0
            )

            components = np.array(
                [body_score, upper_score, lower_score, proximity_score]
            )
            weights = np.array(
                [body_weight, upper_weight, lower_weight, proximity_weight]
            )

            signals[i] = calculate_weighted_score(components, weights)

        return signals

    @njit
    def detect_pin_bar_bear_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        max_body_ratio: float,
        min_upper_shadow: float,
        max_lower_shadow: float,
        min_range_ratio: float,  # NOUVEAU PARAMÈTRE
        body_weight: float,
        upper_weight: float,
        lower_weight: float,
        position_weight: float,
    ) -> np.ndarray:
        """Détecte le pattern Pin Bar baissier.

        Petit corps avec longue ombre haute, corps dans la partie basse.
        """
        signals = np.zeros(len(open_prices), dtype=np.float64)

        for i in range(1, len(open_prices)):
            open_val = open_prices[i]
            high_val = high_prices[i]
            low_val = low_prices[i]
            close_val = close_prices[i]

            if not validate_price_data(open_val, high_val, low_val, close_val):
                continue

            # Calcul et validation du range minimum
            range_ratio = calculate_range_ratio(high_val, low_val, close_val)
            if range_ratio < min_range_ratio:
                continue

            # Calculs des métriques
            body_ratio = calculate_body_ratio(open_val, high_val, low_val, close_val)
            upper_shadow_ratio = calculate_upper_shadow_ratio(
                open_val, high_val, low_val, close_val
            )
            lower_shadow_ratio = calculate_lower_shadow_ratio(
                open_val, high_val, low_val, close_val
            )

            # Validation des conditions
            if (
                body_ratio > max_body_ratio
                or upper_shadow_ratio < min_upper_shadow
                or lower_shadow_ratio > max_lower_shadow
            ):
                continue

            # AUDIT FIX C9: Contexte haussier préalable — pin bar baissier N'EST PAS valide
            # si le marché est déjà baissier. Seul un rejet depuis un sommet est pertinent.
            # [REF-SHOOTING-STAR] Même logique lookback-range. range(1,n) → min 10 barres requises
            min_trend_lookback = 10
            context_score = 0.5  # valeur neutre si pas assez de données
            if i >= min_trend_lookback:
                lb_start = i - min_trend_lookback
                lb_high = np.max(high_prices[lb_start:i])
                lb_low = np.min(low_prices[lb_start:i])
                lb_range = lb_high - lb_low
                if lb_range > 0:
                    # Score = position haute du close = contexte haussier avant rejet
                    context_score = safe_divide(close_val - lb_low, lb_range)
                    context_score = min(1.0, max(0.0, context_score))
                else:
                    context_score = 0.3

            # Vérification de la position du corps (doit être dans la partie basse)
            total_range = high_val - low_val
            if total_range == 0:
                continue

            body_bottom = min(open_val, close_val)
            body_position = safe_divide((body_bottom - low_val), total_range)
            position_score = (
                1.0 - body_position
            )  # Plus le corps est bas, meilleur le score

            # Calcul du score pondéré
            body_score = safe_divide((max_body_ratio - body_ratio), max_body_ratio)
            upper_score = min(1.0, safe_divide(upper_shadow_ratio, min_upper_shadow))
            lower_score = (
                1.0 - safe_divide(lower_shadow_ratio, max_lower_shadow)
                if max_lower_shadow > 0
                else 1.0
            )

            # context_score intégré dans body_score (pondère la qualité de la bougie par contexte)
            body_score = body_score * context_score

            components = np.array(
                [body_score, upper_score, lower_score, position_score]
            )
            weights = np.array(
                [body_weight, upper_weight, lower_weight, position_weight]
            )

            signals[i] = calculate_weighted_score(components, weights)

        return signals

    @njit
    def detect_marubozu_bear_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        min_body_ratio: float,
        max_shadow_ratio: float,
        min_range_ratio: float,
        body_weight: float,
        shadow_weight: float,
    ) -> np.ndarray:
        """Détecte le pattern Marubozu baissier.

        Corps plein sans ombres, close < open.
        """
        signals = np.zeros(len(open_prices), dtype=np.float64)

        for i in range(1, len(open_prices)):
            open_val = open_prices[i]
            high_val = high_prices[i]
            low_val = low_prices[i]
            close_val = close_prices[i]

            if not validate_price_data(open_val, high_val, low_val, close_val):
                continue

            # Doit être baissier
            if not is_bearish_candle(open_val, close_val):
                continue

            # Calculs des métriques
            body_ratio = calculate_body_ratio(open_val, high_val, low_val, close_val)
            upper_shadow_ratio = calculate_upper_shadow_ratio(
                open_val, high_val, low_val, close_val
            )
            lower_shadow_ratio = calculate_lower_shadow_ratio(
                open_val, high_val, low_val, close_val
            )
            range_ratio = calculate_range_ratio(high_val, low_val, close_val)

            # Validation des conditions
            if (
                body_ratio < min_body_ratio
                or upper_shadow_ratio > max_shadow_ratio
                or lower_shadow_ratio > max_shadow_ratio
                or range_ratio < min_range_ratio
            ):
                continue

            # Calcul du score pondéré
            body_score = min(1.0, safe_divide(body_ratio, min_body_ratio))
            shadow_score = (
                1.0
                - safe_divide(
                    max(upper_shadow_ratio, lower_shadow_ratio), max_shadow_ratio
                )
                if max_shadow_ratio > 0
                else 1.0
            )

            components = np.array([body_score, shadow_score])
            weights = np.array([body_weight, shadow_weight])

            signals[i] = calculate_weighted_score(components, weights)

        return signals

    @njit
    def detect_belt_hold_bear_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        min_body_ratio: float,
        max_upper_shadow: float,
        min_range_ratio: float,
        body_weight: float,
        shadow_weight: float,
    ) -> np.ndarray:
        """Détecte le pattern Belt Hold baissier.

        Ouverture au plus haut, corps long baissier, ombre haute minimale.
        """
        signals = np.zeros(len(open_prices), dtype=np.float64)

        for i in range(1, len(open_prices)):
            open_val = open_prices[i]
            high_val = high_prices[i]
            low_val = low_prices[i]
            close_val = close_prices[i]

            if not validate_price_data(open_val, high_val, low_val, close_val):
                continue

            # Doit être baissier
            if not is_bearish_candle(open_val, close_val):
                continue

            # Calculs des métriques
            body_ratio = calculate_body_ratio(open_val, high_val, low_val, close_val)
            upper_shadow_ratio = calculate_upper_shadow_ratio(
                open_val, high_val, low_val, close_val
            )
            range_ratio = calculate_range_ratio(high_val, low_val, close_val)

            # Validation des conditions
            if (
                body_ratio < min_body_ratio
                or upper_shadow_ratio > max_upper_shadow
                or range_ratio < min_range_ratio
            ):
                continue

            # Vérification que l'ouverture est proche du plus haut (belt hold = ouverture au sommet)
            total_range = high_val - low_val
            if total_range == 0:
                continue

            # AUDIT FIX C4: Supprimé le double check redondant open_to_high > max_upper_shadow
            # (déjà filtré par upper_shadow_ratio > max_upper_shadow ci-dessus)

            # Calcul du score pondéré
            body_score = min(1.0, safe_divide(body_ratio, min_body_ratio))
            shadow_score = (
                1.0 - safe_divide(upper_shadow_ratio, max_upper_shadow)
                if max_upper_shadow > 0
                else 1.0
            )

            components = np.array([body_score, shadow_score])
            weights = np.array([body_weight, shadow_weight])

            signals[i] = calculate_weighted_score(components, weights)

        return signals

    @njit
    def detect_evening_star_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        min_body_ratio_1: float,
        max_body_ratio_2: float,
        min_body_ratio_3: float,
        body_weight_1: float,
        star_weight: float,
        body_weight_3: float,
        gap_weight: float,
    ) -> np.ndarray:
        """Détecte le pattern Evening Star (3 bougies).

        Bougie 1: haussière avec gros corps
        Bougie 2: petit corps (star) avec gap up
        Bougie 3: baissière avec gros corps, gap down.
        """
        signals = np.zeros(len(open_prices), dtype=np.float64)

        for i in range(2, len(open_prices)):
            # Données des 3 bougies
            open1, high1, low1, close1 = (
                open_prices[i - 2],
                high_prices[i - 2],
                low_prices[i - 2],
                close_prices[i - 2],
            )
            open2, high2, low2, close2 = (
                open_prices[i - 1],
                high_prices[i - 1],
                low_prices[i - 1],
                close_prices[i - 1],
            )
            open3, high3, low3, close3 = (
                open_prices[i],
                high_prices[i],
                low_prices[i],
                close_prices[i],
            )

            # Validation des données
            if (
                not validate_price_data(open1, high1, low1, close1)
                or not validate_price_data(open2, high2, low2, close2)
                or not validate_price_data(open3, high3, low3, close3)
            ):
                continue

            # Bougie 1: doit être haussière avec gros corps
            if not is_bullish_candle(open1, close1):
                continue

            body_ratio_1 = calculate_body_ratio(open1, high1, low1, close1)
            if body_ratio_1 < min_body_ratio_1:
                continue

            # Bougie 2: petit corps (star)
            body_ratio_2 = calculate_body_ratio(open2, high2, low2, close2)
            if body_ratio_2 > max_body_ratio_2:
                continue

            # Bougie 3: doit être baissière avec gros corps
            if not is_bearish_candle(open3, close3):
                continue

            body_ratio_3 = calculate_body_ratio(open3, high3, low3, close3)
            if body_ratio_3 < min_body_ratio_3:
                continue

            # Vérification Nison : bougie 3 doit clôturer EN-DESSOUS du mi-corps de bougie 1
            mid_body_1 = (open1 + close1) / 2.0
            if close3 >= mid_body_1:
                continue

            # Vérification des gaps
            gap_up = low2 > close1  # Gap entre bougie 1 et 2
            gap_down = close2 > high3  # Gap entre bougie 2 et 3

            # Calcul du score pondéré
            body_score_1 = min(1.0, safe_divide(body_ratio_1, min_body_ratio_1))
            star_score = (
                safe_divide((max_body_ratio_2 - body_ratio_2), max_body_ratio_2)
                if max_body_ratio_2 > 0
                else 1.0
            )
            body_score_3 = min(1.0, safe_divide(body_ratio_3, min_body_ratio_3))
            # P0-5 FIX: Supprimé le plancher 0.5 — sans gap, score = 0.0 (pas 0.5)
            gap_score = 0.5 * (1.0 if gap_up else 0.0) + 0.5 * (
                1.0 if gap_down else 0.0
            )

            components = np.array([body_score_1, star_score, body_score_3, gap_score])
            weights = np.array([body_weight_1, star_weight, body_weight_3, gap_weight])

            signals[i] = calculate_weighted_score(components, weights)

        return signals

    @njit
    def detect_dark_cloud_cover_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        min_body_ratio: float,
        min_penetration: float,
        body_weight_1: float,
        body_weight_2: float,
        penetration_weight: float,
    ) -> np.ndarray:
        """Détecte le pattern Dark Cloud Cover (2 bougies).

        Bougie 1: haussière avec gros corps
        Bougie 2: baissière qui pénètre dans le corps de la bougie 1.
        """
        signals = np.zeros(len(open_prices), dtype=np.float64)

        for i in range(1, len(open_prices)):
            # Données des 2 bougies
            open1, high1, low1, close1 = (
                open_prices[i - 1],
                high_prices[i - 1],
                low_prices[i - 1],
                close_prices[i - 1],
            )
            open2, high2, low2, close2 = (
                open_prices[i],
                high_prices[i],
                low_prices[i],
                close_prices[i],
            )

            # Validation des données
            if not validate_price_data(
                open1, high1, low1, close1
            ) or not validate_price_data(open2, high2, low2, close2):
                continue

            # Bougie 1: doit être haussière avec gros corps
            if not is_bullish_candle(open1, close1):
                continue

            body_ratio_1 = calculate_body_ratio(open1, high1, low1, close1)
            if body_ratio_1 < min_body_ratio:
                continue

            # Bougie 2: doit être baissière avec gros corps
            if not is_bearish_candle(open2, close2):
                continue

            # AUDIT FIX C3: Gap-up obligatoire — open2 doit être au-dessus du close1
            # [REF-PIERCING] Condition symétrique à detect_piercing_line_numba
            if open2 <= close1:
                continue

            body_ratio_2 = calculate_body_ratio(open2, high2, low2, close2)
            if body_ratio_2 < min_body_ratio:
                continue

            # Vérification de la pénétration
            body_1_size = abs(close1 - open1)
            penetration = (
                safe_divide((close1 - close2), body_1_size) if body_1_size > 0 else 0
            )

            if penetration < min_penetration:
                continue

            # Calcul du score pondéré
            body_score_1 = min(1.0, safe_divide(body_ratio_1, min_body_ratio))
            body_score_2 = min(1.0, safe_divide(body_ratio_2, min_body_ratio))
            penetration_score = min(1.0, safe_divide(penetration, min_penetration))

            components = np.array([body_score_1, body_score_2, penetration_score])
            weights = np.array([body_weight_1, body_weight_2, penetration_weight])

            signals[i] = calculate_weighted_score(components, weights)

        return signals

    @njit
    def detect_harami_bear_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        min_body_ratio_1: float,
        max_body_ratio_2: float,
        body_weight_1: float,
        body_weight_2: float,
        containment_weight: float,
    ) -> np.ndarray:
        """Détecte le pattern Harami baissier (2 bougies).

        Bougie 1: haussière avec gros corps
        Bougie 2: baissière avec petit corps contenu dans la bougie 1.
        """
        signals = np.zeros(len(open_prices), dtype=np.float64)

        for i in range(1, len(open_prices)):
            # Données des 2 bougies
            open1, high1, low1, close1 = (
                open_prices[i - 1],
                high_prices[i - 1],
                low_prices[i - 1],
                close_prices[i - 1],
            )
            open2, high2, low2, close2 = (
                open_prices[i],
                high_prices[i],
                low_prices[i],
                close_prices[i],
            )

            # Validation des données
            if not validate_price_data(
                open1, high1, low1, close1
            ) or not validate_price_data(open2, high2, low2, close2):
                continue

            # Bougie 1: doit être haussière avec gros corps
            if not is_bullish_candle(open1, close1):
                continue

            body_ratio_1 = calculate_body_ratio(open1, high1, low1, close1)
            if body_ratio_1 < min_body_ratio_1:
                continue

            # Bougie 2: doit être baissière avec petit corps
            if not is_bearish_candle(open2, close2):
                continue

            body_ratio_2 = calculate_body_ratio(open2, high2, low2, close2)
            if body_ratio_2 > max_body_ratio_2:
                continue

            # P0 FIX: Vérification du containment CORPS-à-CORPS (bougie 2 dans le corps de bougie 1)
            # Dans un harami bear, la bougie 1 est haussière (open1 < close1)
            # Le corps de bougie 2 doit être dans [open1, close1] — on teste open2/close2, PAS high2/low2
            # L'ancienne version (high2 >= close1 or low2 <= open1) utilisait les ombres → quasi-impossible en forex
            contained = (
                open2 >= open1
                and open2 <= close1
                and close2 >= open1
                and close2 <= close1
            )
            if not contained:
                continue

            # Calcul du score de containment
            body_1_size = abs(close1 - open1)
            body_2_size = abs(open2 - close2)
            containment_score = 1.0 - (
                safe_divide(body_2_size, body_1_size) if body_1_size > 0 else 0
            )

            # Calcul du score pondéré
            body_score_1 = min(1.0, safe_divide(body_ratio_1, min_body_ratio_1))
            body_score_2 = (
                safe_divide((max_body_ratio_2 - body_ratio_2), max_body_ratio_2)
                if max_body_ratio_2 > 0
                else 1.0
            )

            components = np.array([body_score_1, body_score_2, containment_score])
            weights = np.array([body_weight_1, body_weight_2, containment_weight])

            signals[i] = calculate_weighted_score(components, weights)

        return signals

    @njit
    def detect_abandoned_baby_bear_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        min_body_ratio_1: float,
        max_body_ratio_2: float,
        min_body_ratio_3: float,
        body_weight_1: float,
        star_weight: float,
        body_weight_3: float,
        gap_weight: float,
    ) -> np.ndarray:
        """Détecte le pattern Abandoned Baby Bear (3 bougies).

        Similaire au Evening Star mais avec des gaps plus stricts.
        """
        signals = np.zeros(len(open_prices), dtype=np.float64)

        for i in range(2, len(open_prices)):
            # Données des 3 bougies
            open1, high1, low1, close1 = (
                open_prices[i - 2],
                high_prices[i - 2],
                low_prices[i - 2],
                close_prices[i - 2],
            )
            open2, high2, low2, close2 = (
                open_prices[i - 1],
                high_prices[i - 1],
                low_prices[i - 1],
                close_prices[i - 1],
            )
            open3, high3, low3, close3 = (
                open_prices[i],
                high_prices[i],
                low_prices[i],
                close_prices[i],
            )

            # Validation des données
            if (
                not validate_price_data(open1, high1, low1, close1)
                or not validate_price_data(open2, high2, low2, close2)
                or not validate_price_data(open3, high3, low3, close3)
            ):
                continue

            # Bougie 1: haussière avec gros corps
            if not is_bullish_candle(open1, close1):
                continue

            body_ratio_1 = calculate_body_ratio(open1, high1, low1, close1)
            if body_ratio_1 < min_body_ratio_1:
                continue

            # Bougie 2: petit corps (star) - peut être haussière ou baissière
            body_ratio_2 = calculate_body_ratio(open2, high2, low2, close2)
            if body_ratio_2 > max_body_ratio_2:
                continue

            # Bougie 3: baissière avec gros corps
            if not is_bearish_candle(open3, close3):
                continue

            body_ratio_3 = calculate_body_ratio(open3, high3, low3, close3)
            if body_ratio_3 < min_body_ratio_3:
                continue

            # Vérification des gaps stricts (pas de chevauchement)
            gap_up = low2 > high1  # Gap strict entre bougie 1 et 2
            gap_down = high3 < low2  # Gap strict entre bougie 2 et 3

            if not (gap_up and gap_down):
                continue

            # AUDIT FIX C6: Vérification Nison — bougie 3 doit clôturer EN-DESSOUS du mi-corps de bougie 1
            # [REF-EVENING-STAR] Condition identique à detect_evening_star_numba (ligne ~4222)
            mid_body_1 = (open1 + close1) / 2.0
            if close3 >= mid_body_1:
                continue

            # Calcul du score pondéré
            body_score_1 = min(1.0, safe_divide(body_ratio_1, min_body_ratio_1))
            star_score = (
                safe_divide((max_body_ratio_2 - body_ratio_2), max_body_ratio_2)
                if max_body_ratio_2 > 0
                else 1.0
            )
            body_score_3 = min(1.0, safe_divide(body_ratio_3, min_body_ratio_3))
            gap_score = 1.0  # Gaps stricts déjà validés par filtre dur ci-dessus

            components = np.array([body_score_1, star_score, body_score_3, gap_score])
            weights = np.array([body_weight_1, star_weight, body_weight_3, gap_weight])

            signals[i] = calculate_weighted_score(components, weights)

        return signals

    @njit
    def detect_three_inside_down_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        min_body_ratio_1: float,
        max_body_ratio_2: float,
        min_body_ratio_3: float,
        body_weight_1: float,
        body_weight_2: float,
        body_weight_3: float,
    ) -> np.ndarray:
        """Détecte le pattern Three Inside Down (3 bougies).

        Bougie 1: haussière avec gros corps
        Bougie 2: baissière avec petit corps contenu dans la bougie 1 (harami)
        Bougie 3: baissière qui confirme le retournement.
        """
        signals = np.zeros(len(open_prices), dtype=np.float64)

        for i in range(2, len(open_prices)):
            # Données des 3 bougies
            open1, high1, low1, close1 = (
                open_prices[i - 2],
                high_prices[i - 2],
                low_prices[i - 2],
                close_prices[i - 2],
            )
            open2, high2, low2, close2 = (
                open_prices[i - 1],
                high_prices[i - 1],
                low_prices[i - 1],
                close_prices[i - 1],
            )
            open3, high3, low3, close3 = (
                open_prices[i],
                high_prices[i],
                low_prices[i],
                close_prices[i],
            )

            # Validation des données
            if (
                not validate_price_data(open1, high1, low1, close1)
                or not validate_price_data(open2, high2, low2, close2)
                or not validate_price_data(open3, high3, low3, close3)
            ):
                continue

            # Bougie 1: doit être haussière avec gros corps
            if not is_bullish_candle(open1, close1):
                continue

            body_ratio_1 = calculate_body_ratio(open1, high1, low1, close1)
            if body_ratio_1 < min_body_ratio_1:
                continue

            # Bougie 2: doit être baissière avec petit corps contenu dans bougie 1
            if not is_bearish_candle(open2, close2):
                continue

            body_ratio_2 = calculate_body_ratio(open2, high2, low2, close2)
            if body_ratio_2 > max_body_ratio_2:
                continue

            # Vérification du containment (bougie 2 contenue dans bougie 1)
            if not (open2 < close1 and close2 > open1):
                continue

            # Bougie 3: doit être baissière et confirmer le retournement
            if not is_bearish_candle(open3, close3):
                continue

            body_ratio_3 = calculate_body_ratio(open3, high3, low3, close3)
            if body_ratio_3 < min_body_ratio_3:
                continue

            # Confirmation: bougie 3 doit clôturer en dessous de la bougie 2
            if close3 >= close2:
                continue

            # Calcul du score pondéré
            body_score_1 = min(1.0, safe_divide(body_ratio_1, min_body_ratio_1))
            body_score_2 = (
                safe_divide((max_body_ratio_2 - body_ratio_2), max_body_ratio_2)
                if max_body_ratio_2 > 0
                else 1.0
            )
            body_score_3 = min(1.0, safe_divide(body_ratio_3, min_body_ratio_3))

            components = np.array([body_score_1, body_score_2, body_score_3])
            weights = np.array([body_weight_1, body_weight_2, body_weight_3])

            signals[i] = calculate_weighted_score(components, weights)

        return signals

    @njit
    def detect_three_outside_down_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        min_body_ratio: float,
        body_weight_1: float,
        body_weight_2: float,
        body_weight_3: float,
    ) -> np.ndarray:
        """Détecte le pattern Three Outside Down (3 bougies).

        Bougie 1: haussière
        Bougie 2: baissière qui englobe la bougie 1 (engulfing)
        Bougie 3: baissière qui confirme le retournement.
        """
        signals = np.zeros(len(open_prices), dtype=np.float64)

        for i in range(2, len(open_prices)):
            # Données des 3 bougies
            open1, high1, low1, close1 = (
                open_prices[i - 2],
                high_prices[i - 2],
                low_prices[i - 2],
                close_prices[i - 2],
            )
            open2, high2, low2, close2 = (
                open_prices[i - 1],
                high_prices[i - 1],
                low_prices[i - 1],
                close_prices[i - 1],
            )
            open3, high3, low3, close3 = (
                open_prices[i],
                high_prices[i],
                low_prices[i],
                close_prices[i],
            )

            # Validation des données
            if (
                not validate_price_data(open1, high1, low1, close1)
                or not validate_price_data(open2, high2, low2, close2)
                or not validate_price_data(open3, high3, low3, close3)
            ):
                continue

            # Bougie 1: doit être haussière
            if not is_bullish_candle(open1, close1):
                continue

            body_ratio_1 = calculate_body_ratio(open1, high1, low1, close1)
            if body_ratio_1 < min_body_ratio:
                continue

            # Bougie 2: doit être baissière et englober la bougie 1
            if not is_bearish_candle(open2, close2):
                continue

            body_ratio_2 = calculate_body_ratio(open2, high2, low2, close2)
            if body_ratio_2 < min_body_ratio:
                continue

            # Vérification de l'englobement (bougie 2 englobe bougie 1)
            if not (open2 > close1 and close2 < open1):
                continue

            # Bougie 3: doit être baissière et confirmer le retournement
            if not is_bearish_candle(open3, close3):
                continue

            body_ratio_3 = calculate_body_ratio(open3, high3, low3, close3)
            if body_ratio_3 < min_body_ratio:
                continue

            # Confirmation: bougie 3 doit clôturer en dessous de la bougie 2
            if close3 >= close2:
                continue

            # Calcul du score pondéré
            body_score_1 = min(1.0, safe_divide(body_ratio_1, min_body_ratio))
            body_score_2 = min(1.0, safe_divide(body_ratio_2, min_body_ratio))
            body_score_3 = min(1.0, safe_divide(body_ratio_3, min_body_ratio))

            components = np.array([body_score_1, body_score_2, body_score_3])
            weights = np.array([body_weight_1, body_weight_2, body_weight_3])

            signals[i] = calculate_weighted_score(components, weights)

        return signals

    @njit
    def detect_advance_block_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        min_body_ratio: float,
        max_upper_shadow: float,
        min_weakening: float,
        body_weight: float,
        shadow_weight: float,
        weakening_weight: float,
    ) -> np.ndarray:
        """Détecte le pattern Advance Block (3 bougies).

        Trois bougies haussières consécutives avec un affaiblissement progressif.
        """
        signals = np.zeros(len(open_prices), dtype=np.float64)

        for i in range(2, len(open_prices)):
            # Données des 3 bougies
            open1, high1, low1, close1 = (
                open_prices[i - 2],
                high_prices[i - 2],
                low_prices[i - 2],
                close_prices[i - 2],
            )
            open2, high2, low2, close2 = (
                open_prices[i - 1],
                high_prices[i - 1],
                low_prices[i - 1],
                close_prices[i - 1],
            )
            open3, high3, low3, close3 = (
                open_prices[i],
                high_prices[i],
                low_prices[i],
                close_prices[i],
            )

            # Validation des données
            if (
                not validate_price_data(open1, high1, low1, close1)
                or not validate_price_data(open2, high2, low2, close2)
                or not validate_price_data(open3, high3, low3, close3)
            ):
                continue

            # Les 3 bougies doivent être haussières
            if not (
                is_bullish_candle(open1, close1)
                and is_bullish_candle(open2, close2)
                and is_bullish_candle(open3, close3)
            ):
                continue

            # Vérification des ratios de corps
            body_ratio_1 = calculate_body_ratio(open1, high1, low1, close1)
            body_ratio_2 = calculate_body_ratio(open2, high2, low2, close2)
            body_ratio_3 = calculate_body_ratio(open3, high3, low3, close3)

            if (
                body_ratio_1 < min_body_ratio
                or body_ratio_2 < min_body_ratio
                or body_ratio_3 < min_body_ratio
            ):
                continue

            # P1-2 FIX: Vérification ombres supérieures croissantes (signal d'épuisement ESSENTIEL)
            # L'Advance Block est caractérisé par des ombres SUPÉRIEURES CROISSANTES
            upper_shadow_1 = high1 - max(open1, close1)
            upper_shadow_2 = high2 - max(open2, close2)
            upper_shadow_3 = high3 - max(open3, close3)

            # Les ombres supérieures DOIVENT être croissantes (croissance = signal d'épuisement)
            if not (upper_shadow_1 <= upper_shadow_2 <= upper_shadow_3):
                continue

            # Filtre: la 3e ombre ne doit pas dépasser max_upper_shadow en ratio
            range3 = high3 - low3
            shadow_ratio_3 = safe_divide(upper_shadow_3, range3) if range3 > 0 else 0.0
            if shadow_ratio_3 > max_upper_shadow:
                continue

            # Vérification de l'affaiblissement progressif
            body_size_1 = abs(close1 - open1)
            body_size_2 = abs(close2 - open2)
            body_size_3 = abs(close3 - open3)

            # Les corps doivent diminuer progressivement
            if not (body_size_1 > body_size_2 > body_size_3):
                continue

            # Calcul du score d'affaiblissement
            weakening_ratio = (
                safe_divide((body_size_1 - body_size_3), body_size_1)
                if body_size_1 > 0
                else 0
            )
            if weakening_ratio < min_weakening:
                continue

            # Calcul du score pondéré
            body_score = safe_divide((body_ratio_1 + body_ratio_2 + body_ratio_3), 3)
            # AUDIT FIX C7b: shadow_ratio_3 (ratio 0-1) divisé par max_upper_shadow (ratio 0-1)
            # L'ancienne version utilisait upper_shadow_3 (prix absolu) / max_upper_shadow (ratio) → incohérence dimensionnelle
            shadow_score = (
                1.0 - safe_divide(shadow_ratio_3, max_upper_shadow)
                if max_upper_shadow > 0
                else 1.0
            )
            weakening_score = min(1.0, safe_divide(weakening_ratio, min_weakening))

            components = np.array([body_score, shadow_score, weakening_score])
            weights = np.array([body_weight, shadow_weight, weakening_weight])

            signals[i] = calculate_weighted_score(components, weights)

        return signals

    @njit
    def detect_deliberation_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        min_body_ratio: float,
        max_body_ratio_3: float,
        max_upper_shadow: float,
        body_weight: float,
        weakening_weight: float,
        shadow_weight: float,
    ) -> np.ndarray:
        """Détecte le pattern Deliberation (3 bougies).

        Deux bougies haussières suivies d'une petite bougie d'hésitation.
        """
        signals = np.zeros(len(open_prices), dtype=np.float64)

        for i in range(2, len(open_prices)):
            # Données des 3 bougies
            open1, high1, low1, close1 = (
                open_prices[i - 2],
                high_prices[i - 2],
                low_prices[i - 2],
                close_prices[i - 2],
            )
            open2, high2, low2, close2 = (
                open_prices[i - 1],
                high_prices[i - 1],
                low_prices[i - 1],
                close_prices[i - 1],
            )
            open3, high3, low3, close3 = (
                open_prices[i],
                high_prices[i],
                low_prices[i],
                close_prices[i],
            )

            # Validation des données
            if (
                not validate_price_data(open1, high1, low1, close1)
                or not validate_price_data(open2, high2, low2, close2)
                or not validate_price_data(open3, high3, low3, close3)
            ):
                continue

            # Les 2 premières bougies doivent être haussières
            if not (
                is_bullish_candle(open1, close1) and is_bullish_candle(open2, close2)
            ):
                continue

            # Vérification des ratios de corps pour les 2 premières bougies
            body_ratio_1 = calculate_body_ratio(open1, high1, low1, close1)
            body_ratio_2 = calculate_body_ratio(open2, high2, low2, close2)

            if body_ratio_1 < min_body_ratio or body_ratio_2 < min_body_ratio:
                continue

            # La 3ème bougie doit avoir un petit corps (hésitation)
            body_ratio_3 = calculate_body_ratio(open3, high3, low3, close3)
            if body_ratio_3 > max_body_ratio_3:
                continue

            # Vérification de l'ombre haute de la 3ème bougie
            upper_shadow_3 = calculate_upper_shadow_ratio(open3, high3, low3, close3)
            if upper_shadow_3 > max_upper_shadow:
                continue

            # La 3ème bougie doit ouvrir dans le corps de la 2ème
            if not (open3 > open2 and open3 < close2):
                continue

            # Calcul du score pondéré
            # AUDIT FIX C-B3-2: body_score non normalisé → discrimination réduite.
            # Exemple: body_ratio=0.60 (min) → score=0.60 au lieu de 1.0 attendu.
            # Normalisation par min_body_ratio pour concentrer la discrimination sur [min, 1.0].
            body_score = min(
                1.0, safe_divide((body_ratio_1 + body_ratio_2) / 2.0, min_body_ratio)
            )
            weakening_score = (
                safe_divide((max_body_ratio_3 - body_ratio_3), max_body_ratio_3)
                if max_body_ratio_3 > 0
                else 1.0
            )
            shadow_score = (
                1.0 - safe_divide(upper_shadow_3, max_upper_shadow)
                if max_upper_shadow > 0
                else 1.0
            )

            components = np.array([body_score, weakening_score, shadow_score])
            weights = np.array([body_weight, weakening_weight, shadow_weight])

            signals[i] = calculate_weighted_score(components, weights)

        return signals

    @njit
    def detect_matching_high_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        min_body_ratio: float,
        max_close_diff: float,
        min_range_ratio: float,
        body_weight: float,
        matching_weight: float,
    ) -> np.ndarray:
        """Détecte le pattern Matching High (2 bougies).

        Deux bougies baissières avec des clôtures similaires (résistance).
        """
        signals = np.zeros(len(open_prices), dtype=np.float64)

        for i in range(1, len(open_prices)):
            # Données des 2 bougies
            open1, high1, low1, close1 = (
                open_prices[i - 1],
                high_prices[i - 1],
                low_prices[i - 1],
                close_prices[i - 1],
            )
            open2, high2, low2, close2 = (
                open_prices[i],
                high_prices[i],
                low_prices[i],
                close_prices[i],
            )

            # Validation des données
            if not validate_price_data(
                open1, high1, low1, close1
            ) or not validate_price_data(open2, high2, low2, close2):
                continue

            # Les deux bougies doivent être baissières
            if not (
                is_bearish_candle(open1, close1) and is_bearish_candle(open2, close2)
            ):
                continue

            # Vérification des ratios de corps
            body_ratio_1 = calculate_body_ratio(open1, high1, low1, close1)
            body_ratio_2 = calculate_body_ratio(open2, high2, low2, close2)

            if body_ratio_1 < min_body_ratio or body_ratio_2 < min_body_ratio:
                continue

            # Vérification du range minimum
            range_ratio_1 = calculate_range_ratio(high1, low1, close1)
            range_ratio_2 = calculate_range_ratio(high2, low2, close2)

            if range_ratio_1 < min_range_ratio or range_ratio_2 < min_range_ratio:
                continue

            # P1-1 FIX: Vérification de la similarité des HIGHS (pas des closes)
            # Le Matching High est défini par 2 hauts identiques, pas 2 clôtures identiques
            high_diff = abs(
                safe_divide(
                    high1 - high2,
                    max(high1, high2) if max(high1, high2) > 0 else 1.0,
                )
            )
            if (
                high_diff > max_close_diff
            ):  # param conservé mais sémantique = max_high_diff
                continue

            # Calcul du score pondéré
            body_score = safe_divide((body_ratio_1 + body_ratio_2), 2)
            matching_score = (
                1.0 - safe_divide(high_diff, max_close_diff)  # P1-1: utiliser high_diff
                if max_close_diff > 0
                else 1.0
            )

            components = np.array([body_score, matching_score])
            weights = np.array([body_weight, matching_weight])

            signals[i] = calculate_weighted_score(components, weights)

        return signals

    @njit
    def detect_ladder_top_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        min_body_ratio: float,
        min_pattern_length: int,
        max_lower_shadow: float,
        body_weight: float,
        pattern_weight: float,
        shadow_weight: float,
    ) -> np.ndarray:
        """Détecte le pattern Ladder Top (5 bougies).

        Formation de sommet en escalier avec affaiblissement progressif.
        """
        signals = np.zeros(len(open_prices), dtype=np.float64)

        for i in range(4, len(open_prices)):
            # Données des 5 bougies
            opens = np.array(
                [
                    open_prices[i - 4],
                    open_prices[i - 3],
                    open_prices[i - 2],
                    open_prices[i - 1],
                    open_prices[i],
                ]
            )
            highs = np.array(
                [
                    high_prices[i - 4],
                    high_prices[i - 3],
                    high_prices[i - 2],
                    high_prices[i - 1],
                    high_prices[i],
                ]
            )
            lows = np.array(
                [
                    low_prices[i - 4],
                    low_prices[i - 3],
                    low_prices[i - 2],
                    low_prices[i - 1],
                    low_prices[i],
                ]
            )
            closes = np.array(
                [
                    close_prices[i - 4],
                    close_prices[i - 3],
                    close_prices[i - 2],
                    close_prices[i - 1],
                    close_prices[i],
                ]
            )

            # Validation des données
            valid_data = True
            for j in range(5):
                if not validate_price_data(opens[j], highs[j], lows[j], closes[j]):
                    valid_data = False
                    break

            if not valid_data:
                continue

            # Les 4 premières bougies doivent être haussières
            bullish_count = 0
            for j in range(4):
                if is_bullish_candle(opens[j], closes[j]):
                    bullish_count += 1

            if bullish_count < 4:
                continue

            # La dernière bougie doit être baissière
            if not is_bearish_candle(opens[4], closes[4]):
                continue

            # Vérification des ratios de corps
            body_ratios = np.zeros(5)
            for j in range(5):
                body_ratios[j] = calculate_body_ratio(
                    opens[j], highs[j], lows[j], closes[j]
                )
                if body_ratios[j] < min_body_ratio:
                    valid_data = False
                    break

            # P0-2 FIX: Remplacement du for/else Numba-incompatible par flag booléen
            if valid_data:
                # Vérification de la formation en escalier (hauts croissants)
                ladder_formation = True
                for j in range(1, 4):
                    if highs[j] <= highs[j - 1]:
                        ladder_formation = False
                        break

                if ladder_formation:
                    # Vérification des ombres basses (doivent être petites)
                    shadow_score = 1.0
                    for j in range(5):
                        lower_shadow = calculate_lower_shadow_ratio(
                            opens[j], highs[j], lows[j], closes[j]
                        )
                        if lower_shadow > max_lower_shadow:
                            # AUDIT FIX C-B3-3: pénalité basée sur l'excès par rapport
                            # au seuil, pas sur la valeur absolue. lower_shadow=0.20 avec
                            # max=0.15 → excess=0.33 → pénalité=0.67 (juste).
                            # Avant: pénalité = 1.0 - lower_shadow (≈0.80 pour ombre à 0.20).
                            excess = safe_divide(
                                lower_shadow - max_lower_shadow, max_lower_shadow
                            )
                            shadow_score *= max(0.0, 1.0 - min(1.0, excess))

                    # Calcul du score pondéré
                    body_score = safe_mean(body_ratios)
                    pattern_score = 1.0  # Formation en escalier validée

                    components = np.array([body_score, pattern_score, shadow_score])
                    weights = np.array([body_weight, pattern_weight, shadow_weight])

                    signals[i] = calculate_weighted_score(components, weights)

        return signals

    # === FONCTIONS DE DÉTECTION DES PATTERNS D'INDÉCISION ===

    @njit
    def detect_doji_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        max_body_ratio: float,
        min_total_shadow: float,
        min_range_ratio: float,
        body_weight: float,
        shadow_weight: float,
    ) -> np.ndarray:
        """Détecte le pattern Doji.

        Corps minimal avec ombres équilibrées, indique l'indécision du marché.
        """
        signals = np.zeros(len(open_prices), dtype=np.float64)

        for i in range(1, len(open_prices)):
            open_val = open_prices[i]
            high_val = high_prices[i]
            low_val = low_prices[i]
            close_val = close_prices[i]

            # Validation des données
            if not validate_price_data(open_val, high_val, low_val, close_val):
                continue

            # Calculs des métriques
            body_ratio = calculate_body_ratio(open_val, high_val, low_val, close_val)
            upper_shadow_ratio = calculate_upper_shadow_ratio(
                open_val, high_val, low_val, close_val
            )
            lower_shadow_ratio = calculate_lower_shadow_ratio(
                open_val, high_val, low_val, close_val
            )
            total_shadow_ratio = upper_shadow_ratio + lower_shadow_ratio
            range_ratio = calculate_range_ratio(high_val, low_val, close_val)

            # Validation des conditions
            if (
                body_ratio > max_body_ratio
                or total_shadow_ratio < min_total_shadow
                or range_ratio < min_range_ratio
            ):
                continue

            # Calcul du score pondéré
            body_score = safe_divide((max_body_ratio - body_ratio), max_body_ratio)
            shadow_score = min(1.0, safe_divide(total_shadow_ratio, min_total_shadow))

            components = np.array([body_score, shadow_score])
            weights = np.array([body_weight, shadow_weight])

            signals[i] = calculate_weighted_score(components, weights)

        return signals

    @njit
    def detect_long_legged_doji_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        max_body_ratio: float,
        min_total_shadow: float,
        min_range_ratio: float,
        body_weight: float,
        shadow_weight: float,
    ) -> np.ndarray:
        """Détecte le pattern Long Legged Doji.

        Doji avec de très longues ombres, indique une forte indécision.
        """
        signals = np.zeros(len(open_prices), dtype=np.float64)

        for i in range(1, len(open_prices)):
            open_val = open_prices[i]
            high_val = high_prices[i]
            low_val = low_prices[i]
            close_val = close_prices[i]

            # Validation des données
            if not validate_price_data(open_val, high_val, low_val, close_val):
                continue

            # Calculs des métriques
            body_ratio = calculate_body_ratio(open_val, high_val, low_val, close_val)
            upper_shadow_ratio = calculate_upper_shadow_ratio(
                open_val, high_val, low_val, close_val
            )
            lower_shadow_ratio = calculate_lower_shadow_ratio(
                open_val, high_val, low_val, close_val
            )
            total_shadow_ratio = upper_shadow_ratio + lower_shadow_ratio
            range_ratio = calculate_range_ratio(high_val, low_val, close_val)

            # Validation des conditions
            if (
                body_ratio > max_body_ratio
                or total_shadow_ratio < min_total_shadow
                or range_ratio < min_range_ratio
            ):
                continue

            # Vérification que les deux ombres sont significatives (équilibrées)
            shadow_balance = (
                safe_divide(
                    min(upper_shadow_ratio, lower_shadow_ratio),
                    max(upper_shadow_ratio, lower_shadow_ratio),
                )
                if max(upper_shadow_ratio, lower_shadow_ratio) > 0
                else 0
            )
            if shadow_balance < 0.3:  # Les ombres doivent être relativement équilibrées
                continue

            # Calcul du score pondéré
            body_score = safe_divide((max_body_ratio - body_ratio), max_body_ratio)
            shadow_score = min(1.0, safe_divide(total_shadow_ratio, min_total_shadow))
            balance_score = shadow_balance

            components = np.array([body_score, shadow_score * balance_score])
            weights = np.array([body_weight, shadow_weight])

            signals[i] = calculate_weighted_score(components, weights)

        return signals

    @njit
    def detect_spinning_top_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        max_body_ratio: float,
        min_total_shadow: float,
        min_range_ratio: float,
        body_weight: float,
        shadow_weight: float,
    ) -> np.ndarray:
        """Détecte le pattern Spinning Top.

        Petit corps avec ombres moyennes, indique l'indécision mais moins extrême qu'un doji.
        """
        signals = np.zeros(len(open_prices), dtype=np.float64)

        for i in range(1, len(open_prices)):
            open_val = open_prices[i]
            high_val = high_prices[i]
            low_val = low_prices[i]
            close_val = close_prices[i]

            # Validation des données
            if not validate_price_data(open_val, high_val, low_val, close_val):
                continue

            # Calculs des métriques
            body_ratio = calculate_body_ratio(open_val, high_val, low_val, close_val)
            upper_shadow_ratio = calculate_upper_shadow_ratio(
                open_val, high_val, low_val, close_val
            )
            lower_shadow_ratio = calculate_lower_shadow_ratio(
                open_val, high_val, low_val, close_val
            )
            total_shadow_ratio = upper_shadow_ratio + lower_shadow_ratio
            range_ratio = calculate_range_ratio(high_val, low_val, close_val)

            # Validation des conditions
            if (
                body_ratio > max_body_ratio
                or total_shadow_ratio < min_total_shadow
                or range_ratio < min_range_ratio
            ):
                continue

            # Le spinning top a un corps plus grand qu'un doji mais reste petit
            # et les ombres sont présentes des deux côtés
            if upper_shadow_ratio < 0.1 or lower_shadow_ratio < 0.1:
                continue

            # FIX : symétrie des ombres — un doji asymétrique (ex. 80% haut / 11% bas)
            # ne représente pas l'indécision équilibrée du spinning top.
            # On exige un ratio compris entre 0.5 et 2.0 (facteur 2 max d'un côté).
            shadow_symmetry = safe_divide(upper_shadow_ratio, lower_shadow_ratio)
            if shadow_symmetry < 0.5 or shadow_symmetry > 2.0:
                continue

            # Calcul du score pondéré
            body_score = safe_divide((max_body_ratio - body_ratio), max_body_ratio)
            shadow_score = min(1.0, safe_divide(total_shadow_ratio, min_total_shadow))

            components = np.array([body_score, shadow_score])
            weights = np.array([body_weight, shadow_weight])

            signals[i] = calculate_weighted_score(components, weights)

        return signals

    @njit
    def detect_rickshaw_man_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        max_body_ratio: float,
        min_total_shadow: float,
        min_range_ratio: float,
        body_weight: float,
        shadow_weight: float,
    ) -> np.ndarray:
        """Détecte le pattern Rickshaw Man.

        Corps très petit avec de très longues ombres, similaire au Long Legged Doji mais avec un corps légèrement plus grand.
        """
        signals = np.zeros(len(open_prices), dtype=np.float64)

        for i in range(1, len(open_prices)):
            open_val = open_prices[i]
            high_val = high_prices[i]
            low_val = low_prices[i]
            close_val = close_prices[i]

            # Validation des données
            if not validate_price_data(open_val, high_val, low_val, close_val):
                continue

            # Calculs des métriques
            body_ratio = calculate_body_ratio(open_val, high_val, low_val, close_val)
            upper_shadow_ratio = calculate_upper_shadow_ratio(
                open_val, high_val, low_val, close_val
            )
            lower_shadow_ratio = calculate_lower_shadow_ratio(
                open_val, high_val, low_val, close_val
            )
            total_shadow_ratio = upper_shadow_ratio + lower_shadow_ratio
            range_ratio = calculate_range_ratio(high_val, low_val, close_val)

            # Validation des conditions
            if (
                body_ratio > max_body_ratio
                or total_shadow_ratio < min_total_shadow
                or range_ratio < min_range_ratio
            ):
                continue

            # Vérification que les deux ombres sont très longues
            if upper_shadow_ratio < 0.25 or lower_shadow_ratio < 0.25:
                continue

            # Le corps doit être positionné au centre (équilibre des ombres)
            shadow_balance = (
                safe_divide(
                    min(upper_shadow_ratio, lower_shadow_ratio),
                    max(upper_shadow_ratio, lower_shadow_ratio),
                )
                if max(upper_shadow_ratio, lower_shadow_ratio) > 0
                else 0
            )

            if shadow_balance < 0.4:  # Ombres relativement équilibrées
                continue

            # Calcul du score pondéré
            body_score = safe_divide((max_body_ratio - body_ratio), max_body_ratio)
            shadow_score = min(1.0, safe_divide(total_shadow_ratio, min_total_shadow))
            balance_bonus = shadow_balance  # Équilibre des ombres

            # AUDIT FIX C-B3-4: balance_bonus était multiplié dans shadow_score (composant 2)
            # → double-poids implicite : shadow_score × balance_bonus × shadow_weight.
            # Correction : balance_bonus devient un composant indépendant avec son propre poids.
            components = np.array([body_score, shadow_score, balance_bonus])
            weights = np.array([0.30, 0.40, 0.30])  # balance = 30% du score

            signals[i] = calculate_weighted_score(components, weights)

        return signals

    @njit
    def detect_high_wave_candle_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        max_body_ratio: float,
        min_total_shadow: float,
        min_range_ratio: float,
        body_weight: float,
        shadow_weight: float,
    ) -> np.ndarray:
        """Détecte le pattern High Wave Candle.

        Bougie avec corps moyen et longues ombres, indique une haute volatilité et indécision.
        """
        signals = np.zeros(len(open_prices), dtype=np.float64)

        for i in range(1, len(open_prices)):
            open_val = open_prices[i]
            high_val = high_prices[i]
            low_val = low_prices[i]
            close_val = close_prices[i]

            # Validation des données
            if not validate_price_data(open_val, high_val, low_val, close_val):
                continue

            # Calculs des métriques
            body_ratio = calculate_body_ratio(open_val, high_val, low_val, close_val)
            upper_shadow_ratio = calculate_upper_shadow_ratio(
                open_val, high_val, low_val, close_val
            )
            lower_shadow_ratio = calculate_lower_shadow_ratio(
                open_val, high_val, low_val, close_val
            )
            total_shadow_ratio = upper_shadow_ratio + lower_shadow_ratio
            range_ratio = calculate_range_ratio(high_val, low_val, close_val)

            # Validation des conditions
            if body_ratio > max_body_ratio or total_shadow_ratio < min_total_shadow:
                continue

            # Vérification de la haute volatilité (range important)
            # AUDIT FIX C-B3-5: L'ancien filtre `range_ratio < min_range_ratio` (retiré)
            # était absorbé silencieusement par ce seuil plus strict * 1.5.
            # Filtrer directement avec le seuil effectif.
            if range_ratio < min_range_ratio * 1.5:
                continue

            # Les ombres doivent être présentes des deux côtés
            if upper_shadow_ratio < 0.15 or lower_shadow_ratio < 0.15:
                continue

            # Calcul du score pondéré avec bonus pour la volatilité
            body_score = safe_divide((max_body_ratio - body_ratio), max_body_ratio)
            shadow_score = min(1.0, safe_divide(total_shadow_ratio, min_total_shadow))
            volatility_bonus = min(
                1.0, safe_divide(range_ratio, (min_range_ratio * 2.0))
            )  # Bonus pour la haute volatilité

            components = np.array([body_score, shadow_score * volatility_bonus])
            weights = np.array([body_weight, shadow_weight])

            signals[i] = calculate_weighted_score(components, weights)

        return signals

    @njit
    def detect_tri_star_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        max_body_ratio: float,
        min_gap_ratio: float,
        body_weight: float,
        gap_weight: float,
    ) -> np.ndarray:
        """Détecte le pattern Tri Star.

        Trois dojis consécutifs avec gaps, pattern très rare d'indécision extrême.
        """
        signals = np.zeros(len(open_prices), dtype=np.float64)

        for i in range(2, len(open_prices)):
            # Données des 3 bougies
            open1, high1, low1, close1 = (
                open_prices[i - 2],
                high_prices[i - 2],
                low_prices[i - 2],
                close_prices[i - 2],
            )
            open2, high2, low2, close2 = (
                open_prices[i - 1],
                high_prices[i - 1],
                low_prices[i - 1],
                close_prices[i - 1],
            )
            open3, high3, low3, close3 = (
                open_prices[i],
                high_prices[i],
                low_prices[i],
                close_prices[i],
            )

            # Validation des données
            if (
                not validate_price_data(open1, high1, low1, close1)
                or not validate_price_data(open2, high2, low2, close2)
                or not validate_price_data(open3, high3, low3, close3)
            ):
                continue

            # Toutes les bougies doivent être des dojis
            body_ratio_1 = calculate_body_ratio(open1, high1, low1, close1)
            body_ratio_2 = calculate_body_ratio(open2, high2, low2, close2)
            body_ratio_3 = calculate_body_ratio(open3, high3, low3, close3)

            if (
                body_ratio_1 > max_body_ratio
                or body_ratio_2 > max_body_ratio
                or body_ratio_3 > max_body_ratio
            ):
                continue

            # Vérification des gaps entre les dojis
            # Gap entre doji 1 et 2
            gap1_ratio = 0.0
            if close1 > 0:
                if open2 > close1:  # Gap up
                    gap1_ratio = safe_divide((open2 - close1), close1)
                elif open2 < close1:  # Gap down
                    gap1_ratio = safe_divide((close1 - open2), close1)

            # Gap entre doji 2 et 3
            gap2_ratio = 0.0
            if close2 > 0:
                if open3 > close2:  # Gap up
                    gap2_ratio = safe_divide((open3 - close2), close2)
                elif open3 < close2:  # Gap down
                    gap2_ratio = safe_divide((close2 - open3), close2)

            # Au moins un gap significatif doit être présent
            if gap1_ratio < min_gap_ratio and gap2_ratio < min_gap_ratio:
                continue

            # Calcul du score pondéré
            avg_body_ratio = safe_divide(
                (body_ratio_1 + body_ratio_2 + body_ratio_3), 3
            )
            body_score = (
                safe_divide((max_body_ratio - avg_body_ratio), max_body_ratio)
                if max_body_ratio > 0
                else 1.0
            )

            avg_gap_ratio = safe_divide((gap1_ratio + gap2_ratio), 2)
            gap_score = min(1.0, safe_divide(avg_gap_ratio, min_gap_ratio))

            components = np.array([body_score, gap_score])
            weights = np.array([body_weight, gap_weight])

            signals[i] = calculate_weighted_score(components, weights)

        return signals

    # === PATTERNS CHARTISTES DE RETOURNEMENT ===

    @njit
    def detect_double_top_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        min_peak_distance: int,
        max_height_diff_ratio: float,
        min_valley_depth_ratio: float,
        peak_weight: float,
        valley_weight: float,
        distance_weight: float,
    ) -> np.ndarray:
        """Détecte le pattern Double Top.

        Recherche de 2 pics similaires séparés par une vallée significative.
        """
        signals = np.zeros(len(open_prices), dtype=np.float64)
        min_window = min_peak_distance * 2

        for i in range(min_window, len(open_prices)):
            # P0-4 FIX: Remplacement de sorted(key=lambda) par _find_two_highest
            # Collecter les pics dans des arrays pré-alloués (Numba-compatible)
            _n_peaks = 0
            _pk_idx = np.empty(min_window, dtype=np.int64)
            _pk_price = np.empty(min_window, dtype=np.float64)
            for j in range(i - min_window, i - 2):
                if (
                    j > 0
                    and j < len(high_prices) - 1
                    and high_prices[j] > high_prices[j - 1]
                    and high_prices[j] > high_prices[j + 1]
                ):
                    _pk_idx[_n_peaks] = j
                    _pk_price[_n_peaks] = high_prices[j]
                    _n_peaks += 1
                    if _n_peaks >= min_window:
                        break

            if _n_peaks < 2:
                continue

            # Trouver les 2 pics les plus hauts sans sorted(lambda)
            peak1_idx, peak1_price, peak2_idx, peak2_price = _find_two_highest(
                _pk_idx[:_n_peaks], _pk_price[:_n_peaks], _n_peaks
            )

            # Validation de la distance entre les pics
            if abs(peak1_idx - peak2_idx) < min_peak_distance:
                continue

            # Validation de la similarité des hauteurs
            height_diff_ratio = abs(
                safe_divide(peak1_price - peak2_price, max(peak1_price, peak2_price))
            )
            if height_diff_ratio > max_height_diff_ratio:
                continue

            # Recherche de la vallée entre les pics
            start_idx = min(peak1_idx, peak2_idx)
            end_idx = max(peak1_idx, peak2_idx)
            valley_price = np.min(low_prices[start_idx : end_idx + 1])

            # Validation de la profondeur de la vallée
            avg_peak_price = safe_divide((peak1_price + peak2_price), 2)
            valley_depth_ratio = safe_divide(
                (avg_peak_price - valley_price), avg_peak_price
            )
            if valley_depth_ratio < min_valley_depth_ratio:
                continue

            # Calcul du score
            peak_score = 1.0 - safe_divide(height_diff_ratio, max_height_diff_ratio)
            valley_score = min(
                1.0, safe_divide(valley_depth_ratio, min_valley_depth_ratio)
            )
            distance_score = min(
                1.0,
                abs(peak1_idx - peak2_idx)
                / (min_peak_distance * 2.0 if (min_peak_distance * 2.0) != 0 else 0.0),
            )

            components = np.array([peak_score, valley_score, distance_score])
            weights = np.array([peak_weight, valley_weight, distance_weight])

            signals[i] = calculate_weighted_score(components, weights)

        return signals

    @njit
    def detect_double_bottom_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        min_valley_distance: int,
        max_depth_diff_ratio: float,
        min_peak_height_ratio: float,
        valley_weight: float,
        peak_weight: float,
        distance_weight: float,
    ) -> np.ndarray:
        """Détecte le pattern Double Bottom.

        Recherche de 2 creux similaires séparés par un pic significatif.
        """
        signals = np.zeros(len(open_prices), dtype=np.float64)
        min_window = min_valley_distance * 2

        for i in range(min_window, len(open_prices)):
            # P0-4 FIX: Remplacement de sorted(key=lambda) par _find_two_lowest
            _n_valleys = 0
            _vl_idx = np.empty(min_window, dtype=np.int64)
            _vl_price = np.empty(min_window, dtype=np.float64)
            for j in range(i - min_window, i - 2):
                if (
                    j > 0
                    and j < len(low_prices) - 1
                    and low_prices[j] < low_prices[j - 1]
                    and low_prices[j] < low_prices[j + 1]
                ):
                    _vl_idx[_n_valleys] = j
                    _vl_price[_n_valleys] = low_prices[j]
                    _n_valleys += 1
                    if _n_valleys >= min_window:
                        break

            if _n_valleys < 2:
                continue

            # Trouver les 2 creux les plus bas sans sorted(lambda)
            valley1_idx, valley1_price, valley2_idx, valley2_price = _find_two_lowest(
                _vl_idx[:_n_valleys], _vl_price[:_n_valleys], _n_valleys
            )

            # Validation de la distance entre les creux
            if abs(valley1_idx - valley2_idx) < min_valley_distance:
                continue

            # Validation de la similarité des profondeurs
            # AUDIT FIX C-B3-1: valley1_price < valley2_price par construction (_find_two_lowest
            # retourne le plus bas en premier) → depth_diff_ratio était TOUJOURS négatif.
            # Le filtre > max_depth_diff_ratio ne rejetait donc RIEN, et valley_score = 1.0
            # en permanence. Correction : abs() pour obtenir une distance absolue.
            depth_diff_ratio = abs(
                safe_divide(
                    (valley1_price - valley2_price), min(valley1_price, valley2_price)
                )
            )
            if depth_diff_ratio > max_depth_diff_ratio:
                continue

            # Recherche du pic entre les creux
            start_idx = min(valley1_idx, valley2_idx)
            end_idx = max(valley1_idx, valley2_idx)
            peak_price = np.max(high_prices[start_idx : end_idx + 1])

            # Validation de la hauteur du pic
            avg_valley_price = safe_divide((valley1_price + valley2_price), 2)
            peak_height_ratio = safe_divide(
                (peak_price - avg_valley_price), avg_valley_price
            )
            if peak_height_ratio < min_peak_height_ratio:
                continue

            # Calcul du score
            valley_score = 1.0 - safe_divide(depth_diff_ratio, max_depth_diff_ratio)
            peak_score = min(1.0, safe_divide(peak_height_ratio, min_peak_height_ratio))
            distance_score = min(
                1.0,
                abs(valley1_idx - valley2_idx)
                / (
                    min_valley_distance * 2.0
                    if (min_valley_distance * 2.0) != 0
                    else 0.0
                ),
            )

            components = np.array([valley_score, peak_score, distance_score])
            weights = np.array([valley_weight, peak_weight, distance_weight])

            signals[i] = calculate_weighted_score(components, weights)

        return signals

    @njit
    def detect_triple_top_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        min_peak_distance: int,
        max_height_diff_ratio: float,
        min_valley_depth_ratio: float,
        peak_weight: float,
        valley_weight: float,
        distance_weight: float,
    ) -> np.ndarray:
        """Détecte le pattern Triple Top.

        Recherche de 3 pics similaires séparés par 2 vallées significatives.
        """
        signals = np.zeros(len(open_prices), dtype=np.float64)
        min_window = min_peak_distance * 3

        for i in range(min_window, len(open_prices)):
            # Recherche des pics dans la fenêtre précédente — NumPy arrays pour Numba
            peak_indices = np.empty(min_window, dtype=np.int64)
            peak_prices = np.empty(min_window, dtype=np.float64)
            n_peaks = 0
            for j in range(i - min_window, i - 2):
                if (
                    j > 0
                    and j < len(high_prices) - 1
                    and high_prices[j] > high_prices[j - 1]
                    and high_prices[j] > high_prices[j + 1]
                ):
                    peak_indices[n_peaks] = j
                    peak_prices[n_peaks] = high_prices[j]
                    n_peaks += 1

            if n_peaks < 3:
                continue

            # Trier les pics par hauteur (décroissant) via argsort et prendre les 3 plus hauts
            sorted_by_price = np.argsort(-peak_prices[:n_peaks])
            top3_idx = np.array(
                [
                    peak_indices[sorted_by_price[0]],
                    peak_indices[sorted_by_price[1]],
                    peak_indices[sorted_by_price[2]],
                ],
                dtype=np.int64,
            )
            top3_price = np.array(
                [
                    peak_prices[sorted_by_price[0]],
                    peak_prices[sorted_by_price[1]],
                    peak_prices[sorted_by_price[2]],
                ],
                dtype=np.float64,
            )

            # Réorganiser par ordre chronologique via argsort
            chrono_order = np.argsort(top3_idx)
            chrono_idx = np.array(
                [
                    top3_idx[chrono_order[0]],
                    top3_idx[chrono_order[1]],
                    top3_idx[chrono_order[2]],
                ],
                dtype=np.int64,
            )
            chrono_price = np.array(
                [
                    top3_price[chrono_order[0]],
                    top3_price[chrono_order[1]],
                    top3_price[chrono_order[2]],
                ],
                dtype=np.float64,
            )

            # Validation des distances entre les pics
            if (
                chrono_idx[1] - chrono_idx[0] < min_peak_distance
                or chrono_idx[2] - chrono_idx[1] < min_peak_distance
            ):
                continue

            # Validation de la similarité des hauteurs
            max_price = chrono_price[0]
            for kk in range(1, 3):
                if chrono_price[kk] > max_price:
                    max_price = chrono_price[kk]
            min_price = chrono_price[0]
            for kk in range(1, 3):
                if chrono_price[kk] < min_price:
                    min_price = chrono_price[kk]
            if max_price <= 1e-10:  # Protection contre division par zéro
                continue
            height_diff_ratio = safe_divide((max_price - min_price), max_price)
            if height_diff_ratio > max_height_diff_ratio:
                continue

            # Recherche des vallées entre les pics
            valley1_price = np.min(low_prices[chrono_idx[0] : chrono_idx[1] + 1])
            valley2_price = np.min(low_prices[chrono_idx[1] : chrono_idx[2] + 1])

            # Validation de la profondeur des vallées
            avg_peak_price = safe_divide(
                (chrono_price[0] + chrono_price[1] + chrono_price[2]), 3.0
            )
            if avg_peak_price <= 1e-10:  # Protection contre division par zéro
                continue
            valley1_depth_ratio = safe_divide(
                (avg_peak_price - valley1_price), avg_peak_price
            )
            valley2_depth_ratio = safe_divide(
                (avg_peak_price - valley2_price), avg_peak_price
            )

            if (
                valley1_depth_ratio < min_valley_depth_ratio
                or valley2_depth_ratio < min_valley_depth_ratio
            ):
                continue

            # Calcul du score
            peak_score = 1.0 - safe_divide(height_diff_ratio, max_height_diff_ratio)
            valley_score = min(
                1.0,
                (valley1_depth_ratio + valley2_depth_ratio)
                / (
                    2.0 * min_valley_depth_ratio
                    if (2.0 * min_valley_depth_ratio) != 0
                    else 0.0
                ),
            )
            distance_score = min(
                1.0,
                (chrono_idx[2] - chrono_idx[0])
                / (min_peak_distance * 3.0 if (min_peak_distance * 3.0) != 0 else 0.0),
            )

            components = np.array([peak_score, valley_score, distance_score])
            weights = np.array([peak_weight, valley_weight, distance_weight])

            signals[i] = calculate_weighted_score(components, weights)

        return signals

    @njit
    def detect_triple_bottom_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        min_valley_distance: int,
        max_depth_diff_ratio: float,
        min_peak_height_ratio: float,
        valley_weight: float,
        peak_weight: float,
        distance_weight: float,
    ) -> np.ndarray:
        """Détecte le pattern Triple Bottom.

        Recherche de 3 creux similaires séparés par 2 pics significatifs.
        """
        signals = np.zeros(len(open_prices), dtype=np.float64)
        min_window = min_valley_distance * 3

        for i in range(min_window, len(open_prices)):
            # Recherche des creux dans la fenêtre précédente — NumPy arrays pour Numba
            valley_indices = np.empty(min_window, dtype=np.int64)
            valley_prices = np.empty(min_window, dtype=np.float64)
            n_valleys = 0
            for j in range(i - min_window, i - 2):
                if (
                    j > 0
                    and j < len(low_prices) - 1
                    and low_prices[j] < low_prices[j - 1]
                    and low_prices[j] < low_prices[j + 1]
                ):
                    valley_indices[n_valleys] = j
                    valley_prices[n_valleys] = low_prices[j]
                    n_valleys += 1

            if n_valleys < 3:
                continue

            # Trier les creux par profondeur (croissant) via argsort et prendre les 3 plus bas
            sorted_by_price = np.argsort(valley_prices[:n_valleys])
            top3_idx = np.array(
                [
                    valley_indices[sorted_by_price[0]],
                    valley_indices[sorted_by_price[1]],
                    valley_indices[sorted_by_price[2]],
                ],
                dtype=np.int64,
            )
            top3_price = np.array(
                [
                    valley_prices[sorted_by_price[0]],
                    valley_prices[sorted_by_price[1]],
                    valley_prices[sorted_by_price[2]],
                ],
                dtype=np.float64,
            )

            # Réorganiser par ordre chronologique via argsort
            chrono_order = np.argsort(top3_idx)
            chrono_idx = np.array(
                [
                    top3_idx[chrono_order[0]],
                    top3_idx[chrono_order[1]],
                    top3_idx[chrono_order[2]],
                ],
                dtype=np.int64,
            )
            chrono_price = np.array(
                [
                    top3_price[chrono_order[0]],
                    top3_price[chrono_order[1]],
                    top3_price[chrono_order[2]],
                ],
                dtype=np.float64,
            )

            # Validation des distances entre les creux
            if (
                chrono_idx[1] - chrono_idx[0] < min_valley_distance
                or chrono_idx[2] - chrono_idx[1] < min_valley_distance
            ):
                continue

            # Validation de la similarité des profondeurs
            max_price = chrono_price[0]
            for kk in range(1, 3):
                if chrono_price[kk] > max_price:
                    max_price = chrono_price[kk]
            min_price = chrono_price[0]
            for kk in range(1, 3):
                if chrono_price[kk] < min_price:
                    min_price = chrono_price[kk]
            if min_price <= 1e-10:  # Protection contre division par zéro
                continue
            depth_diff_ratio = safe_divide((max_price - min_price), min_price)
            if depth_diff_ratio > max_depth_diff_ratio:
                continue

            # Recherche des pics entre les creux
            peak1_price = np.max(high_prices[chrono_idx[0] : chrono_idx[1] + 1])
            peak2_price = np.max(high_prices[chrono_idx[1] : chrono_idx[2] + 1])

            # Validation de la hauteur des pics
            avg_valley_price = safe_divide(
                (chrono_price[0] + chrono_price[1] + chrono_price[2]), 3.0
            )
            if avg_valley_price <= 1e-10:  # Protection contre division par zéro
                continue
            peak1_height_ratio = safe_divide(
                (peak1_price - avg_valley_price), avg_valley_price
            )
            peak2_height_ratio = safe_divide(
                (peak2_price - avg_valley_price), avg_valley_price
            )

            if (
                peak1_height_ratio < min_peak_height_ratio
                or peak2_height_ratio < min_peak_height_ratio
            ):
                continue

            # Calcul du score
            valley_score = 1.0 - safe_divide(depth_diff_ratio, max_depth_diff_ratio)
            peak_score = min(
                1.0,
                (peak1_height_ratio + peak2_height_ratio)
                / (
                    2.0 * min_peak_height_ratio
                    if (2.0 * min_peak_height_ratio) != 0
                    else 0.0
                ),
            )
            distance_score = min(
                1.0,
                (chrono_idx[2] - chrono_idx[0])
                / (
                    min_valley_distance * 3.0
                    if (min_valley_distance * 3.0) != 0
                    else 0.0
                ),
            )

            components = np.array([valley_score, peak_score, distance_score])
            weights = np.array([valley_weight, peak_weight, distance_weight])

            signals[i] = calculate_weighted_score(components, weights)

        return signals

    @njit
    def detect_v_top_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        min_spike_ratio: float,
        max_duration: int,
        min_volume_spike: float,
        spike_weight: float,
        duration_weight: float,
        volume_weight: float,
    ) -> np.ndarray:
        """Détection du pattern V Top.

        Retournement rapide et brutal vers le bas après un pic.

        Problème de la version précédente :
        min_spike_ratio = 0.005 (absolu) était trivial sur D1 — un mouvement
        de 0.5% sur 5 jours est banal → 32% de détections sur D1.

        Fix — double condition de seuil :
        1. spike_ratio >= min_spike_ratio  (plancher absolu, paramétré)
        2. spike_in_atrs >= 2.0            (plancher relatif, adaptatif)

        La condition 2 est la clé : elle exige que le spike représente au moins
        2× l'ATR local, ce qui est automatiquement plus strict sur D1 (ATR élevé)
        et plus permissif sur M5 (ATR faible). Un vrai V-top est un mouvement
        exceptionnel par rapport à la volatilité ambiante.

        Le volume_score ne fait plus de `continue` sur score=0 — valeur neutre
        0.5 à la place pour ne pas éliminer les actifs forex à volume faible.
        """
        n = len(open_prices)
        signals = np.zeros(n, dtype=np.float64)
        atr_period = 14

        for i in range(max_duration + atr_period + 1, n):
            # ── ATR local sur 14 barres avant i ───────────────────────────
            _atr_sum = 0.0
            _atr_cnt = 0
            for _k in range(i - atr_period, i):
                if _k < 1:
                    continue
                _tr = high_prices[_k] - low_prices[_k]
                _pc = close_prices[_k - 1]
                if _pc > 0:
                    _tr = max(
                        _tr, abs(high_prices[_k] - _pc), abs(low_prices[_k] - _pc)
                    )
                if _tr > 0:
                    _atr_sum += _tr
                    _atr_cnt += 1

            local_atr = _atr_sum / _atr_cnt if _atr_cnt > 0 else 0.0
            if local_atr <= 0:
                continue

            # ── Recherche du pic dans la fenêtre ──────────────────────────
            peak_idx = -1
            peak_price = 0.0

            for j in range(i - max_duration, i):
                if (
                    j > 0
                    and j < n - 1
                    and high_prices[j] > high_prices[j - 1]
                    and high_prices[j] > high_prices[j + 1]
                    and high_prices[j] > peak_price
                ):
                    peak_idx = j
                    peak_price = high_prices[j]

            if peak_idx == -1:
                continue

            duration = i - peak_idx
            if duration > max_duration:
                continue

            # ── Spike montant (avant le pic) ───────────────────────────────
            pre_start = max(0, peak_idx - max_duration)
            pre_peak_low = (
                np.min(low_prices[pre_start:peak_idx])
                if peak_idx > pre_start
                else peak_price
            )
            if pre_peak_low <= 0:
                continue

            spike_size = peak_price - pre_peak_low
            spike_ratio = safe_divide(spike_size, pre_peak_low)

            # Condition 1 : plancher absolu (paramétré)
            if spike_ratio < min_spike_ratio:
                continue

            # Condition 2 : plancher relatif — spike doit valoir >= 2× ATR
            # Sans cette condition, un spike de 2% sur D1 passe alors que
            # l'ATR D1 est déjà à 1-1.5% → rien d'exceptionnel.
            spike_in_atrs = safe_divide(spike_size, local_atr)
            if spike_in_atrs < 3.0:
                continue

            # ── Chute après le pic ─────────────────────────────────────────
            current_price = close_prices[i]
            if peak_price <= 0:
                continue

            fall_size = peak_price - current_price
            fall_ratio = safe_divide(fall_size, peak_price)

            if fall_ratio < min_spike_ratio:
                continue

            # La chute doit aussi être >= 1.5× ATR
            if safe_divide(fall_size, local_atr) < 2.0:
                continue

            # ── Volume ─────────────────────────────────────────────────────
            volume_score = 0.5  # valeur neutre si volume non dispo
            if len(volume) > peak_idx and volume[peak_idx] > 0:
                vol_start = max(0, peak_idx - 10)
                avg_vol = safe_mean(volume[vol_start:peak_idx])
                if avg_vol > 0:
                    vol_ratio = safe_divide(volume[peak_idx], avg_vol)
                    if vol_ratio >= min_volume_spike:
                        volume_score = min(
                            1.0, safe_divide(vol_ratio, min_volume_spike * 1.5)
                        )
                    else:
                        volume_score = safe_divide(vol_ratio, min_volume_spike)
                        if volume_score > 1.0:
                            volume_score = 1.0

            # ── Scores ─────────────────────────────────────────────────────
            # spike_score : qualité relative (en ATRs, plafonné à 1)
            spike_score = min(1.0, safe_divide(spike_in_atrs, 4.0))
            duration_score = 1.0 - safe_divide(duration - 1, max_duration)

            components = np.array([spike_score, duration_score, volume_score])
            weights = np.array([spike_weight, duration_weight, volume_weight])

            signals[i] = calculate_weighted_score(components, weights)

        return signals

    @njit
    def detect_v_bottom_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        min_spike_ratio: float,
        max_duration: int,
        min_volume_spike: float,
        spike_weight: float,
        duration_weight: float,
        volume_weight: float,
    ) -> np.ndarray:
        """Détection du pattern V Bottom.

        Retournement rapide et brutal vers le haut après un creux.
        Même logique ATR que detect_v_top_numba (symétrique).
        """
        n = len(open_prices)
        signals = np.zeros(n, dtype=np.float64)
        atr_period = 14

        for i in range(max_duration + atr_period + 1, n):
            # ── ATR local ────────────────────────────────────────────────
            _atr_sum = 0.0
            _atr_cnt = 0
            for _k in range(i - atr_period, i):
                if _k < 1:
                    continue
                _tr = high_prices[_k] - low_prices[_k]
                _pc = close_prices[_k - 1]
                if _pc > 0:
                    _tr = max(
                        _tr, abs(high_prices[_k] - _pc), abs(low_prices[_k] - _pc)
                    )
                if _tr > 0:
                    _atr_sum += _tr
                    _atr_cnt += 1

            local_atr = _atr_sum / _atr_cnt if _atr_cnt > 0 else 0.0
            if local_atr <= 0:
                continue

            # ── Recherche du creux ────────────────────────────────────────
            valley_idx = -1
            valley_price = 1e18

            for j in range(i - max_duration, i):
                if (
                    j > 0
                    and j < n - 1
                    and low_prices[j] < low_prices[j - 1]
                    and low_prices[j] < low_prices[j + 1]
                    and low_prices[j] < valley_price
                ):
                    valley_idx = j
                    valley_price = low_prices[j]

            if valley_idx == -1:
                continue

            duration = i - valley_idx
            if duration > max_duration:
                continue

            # ── Spike descendant (avant le creux) ─────────────────────────
            pre_start = max(0, valley_idx - max_duration)
            pre_valley_high = (
                np.max(high_prices[pre_start:valley_idx])
                if valley_idx > pre_start
                else valley_price
            )
            if valley_price <= 0:
                continue

            spike_size = pre_valley_high - valley_price
            spike_ratio = safe_divide(spike_size, valley_price)

            if spike_ratio < min_spike_ratio:
                continue

            spike_in_atrs = safe_divide(spike_size, local_atr)
            if spike_in_atrs < 3.0:
                continue

            # ── Remontée après le creux ───────────────────────────────────
            current_price = close_prices[i]
            rise_size = current_price - valley_price
            rise_ratio = safe_divide(rise_size, valley_price)

            if rise_ratio < min_spike_ratio:
                continue

            if safe_divide(rise_size, local_atr) < 2.0:
                continue

            # ── Volume ───────────────────────────────────────────────────
            volume_score = 0.5
            if len(volume) > valley_idx and volume[valley_idx] > 0:
                vol_start = max(0, valley_idx - 10)
                avg_vol = safe_mean(volume[vol_start:valley_idx])
                if avg_vol > 0:
                    vol_ratio = safe_divide(volume[valley_idx], avg_vol)
                    if vol_ratio >= min_volume_spike:
                        volume_score = min(
                            1.0, safe_divide(vol_ratio, min_volume_spike * 1.5)
                        )
                    else:
                        volume_score = safe_divide(vol_ratio, min_volume_spike)
                        if volume_score > 1.0:
                            volume_score = 1.0

            # ── Scores ───────────────────────────────────────────────────
            spike_score = min(1.0, safe_divide(spike_in_atrs, 4.0))
            duration_score = 1.0 - safe_divide(duration - 1, max_duration)

            components = np.array([spike_score, duration_score, volume_score])
            weights = np.array([spike_weight, duration_weight, volume_weight])

            signals[i] = calculate_weighted_score(components, weights)

        return signals

    @njit
    def detect_rounding_top_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        min_curve_length: int,
        max_volatility_ratio: float,
        min_volume_decline: float,
        curve_weight: float,
        volume_weight: float,
        smoothness_weight: float,
    ) -> np.ndarray:
        """VERSION CORRIGÉE 2.0 - Détection de Rounding Top avec Contexte et Structure."""
        signals = np.zeros(len(open_prices), dtype=np.float64)

        # On a besoin de données avant le pattern pour vérifier le contexte
        for i in range(min_curve_length + 10, len(open_prices)):
            # Définition de la fenêtre du pattern
            start_idx = i - min_curve_length
            end_idx = i

            window_highs = high_prices[start_idx : end_idx + 1]

            # NOUVEAU: Vérification du contexte - doit être précédé par une HAUSSE
            context_start_price = close_prices[start_idx - 10]
            peak_price = np.max(window_highs)
            if (
                context_start_price >= peak_price * 0.95
            ):  # La hausse doit être d'au moins 5%
                continue

            # NOUVEAU: Vérification de la structure - les bords doivent être alignés
            start_rim_price = close_prices[start_idx]
            end_rim_price = close_prices[end_idx]
            avg_rim_price = (start_rim_price + end_rim_price) / 2.0
            if avg_rim_price <= 0:
                continue

            rim_diff_ratio = abs(start_rim_price - end_rim_price) / avg_rim_price
            if rim_diff_ratio > 0.15:  # Tolérance de 15% sur la hauteur des bords
                continue

            # P2-5 FIX: Utilisation de la parabole pour la courbure sur les PRIX HAUTS (x centré)
            x_raw = np.arange(len(window_highs), dtype=np.float64)
            x_mean = safe_mean(x_raw)
            x = x_raw - x_mean
            a, _, _ = fit_parabola(x, window_highs)
            # NOUVEAU: La parabole doit être ouverte vers le BAS (a < 0)
            if a >= -0.0001:  # La courbe doit être visiblement descendante
                continue

            # Vérifier la volatilité interne (smoothness)
            volatility_ratio = safe_divide(
                np.max(window_highs) - np.min(window_highs), np.mean(window_highs)
            )
            if volatility_ratio > max_volatility_ratio:
                continue

            # NOUVEAU: Vérifier le DÉCLIN du volume
            early_volume = np.mean(volume[start_idx : start_idx + 5])
            late_volume = np.mean(volume[end_idx - 4 : end_idx + 1])
            if early_volume <= 0:
                continue

            volume_decline = (early_volume - late_volume) / early_volume
            if volume_decline < min_volume_decline:
                continue

            # Calcul des scores
            curve_score = min(1.0, abs(a) * 1000)
            smoothness_score = 1.0 - safe_divide(volatility_ratio, max_volatility_ratio)
            volume_score = min(1.0, safe_divide(volume_decline, min_volume_decline * 2))

            components = np.array([curve_score, smoothness_score, volume_score])
            weights = np.array([curve_weight, smoothness_weight, volume_weight])
            signals[i] = calculate_weighted_score(components, weights)

        return signals

    @njit
    def detect_rounding_bottom_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        min_curve_length: int,
        max_volatility_ratio: float,
        min_volume_increase: float,
        curve_weight: float,
        volume_weight: float,
        smoothness_weight: float,
    ) -> np.ndarray:
        """VERSION CORRIGÉE 2.0 - Détection de Rounding Bottom avec Contexte et Structure."""
        signals = np.zeros(len(open_prices), dtype=np.float64)

        # NOUVEAU: On a besoin de données avant le pattern pour vérifier le contexte
        for i in range(min_curve_length + 10, len(open_prices)):
            # Définition de la fenêtre du pattern
            start_idx = i - min_curve_length
            end_idx = i

            window_lows = low_prices[start_idx : end_idx + 1]

            # NOUVEAU: Vérification du contexte - doit être précédé par une baisse
            context_start_price = close_prices[start_idx - 10]
            bottom_price = np.min(window_lows)
            if (
                context_start_price <= bottom_price * 1.05
            ):  # La baisse doit être d'au moins 5%
                continue

            # NOUVEAU: Vérification de la structure - les bords doivent être alignés
            start_rim_price = close_prices[start_idx]
            end_rim_price = close_prices[end_idx]
            avg_rim_price = (start_rim_price + end_rim_price) / 2.0

            if avg_rim_price <= 0:
                continue

            rim_diff_ratio = abs(start_rim_price - end_rim_price) / avg_rim_price
            if rim_diff_ratio > 0.15:  # Tolérance de 15% sur la hauteur des bords
                continue

            # P2-5 FIX: Utilisation de la parabole pour la courbure (x centré)
            x_raw = np.arange(len(window_lows), dtype=np.float64)
            x_mean = safe_mean(x_raw)
            x = x_raw - x_mean
            a, _, _ = fit_parabola(x, window_lows)
            if a <= 0.0001:  # La courbe doit être visiblement ascendante
                continue

            # Vérifier la volatilité interne (smoothness)
            volatility_ratio = safe_divide(
                np.max(window_lows) - np.min(window_lows), np.mean(window_lows)
            )
            if volatility_ratio > max_volatility_ratio:
                continue

            # Vérifier le volume
            early_volume = np.mean(volume[start_idx : start_idx + 5])
            late_volume = np.mean(volume[end_idx - 4 : end_idx + 1])
            if early_volume <= 0 or late_volume / early_volume < min_volume_increase:
                continue

            # Calcul des scores
            curve_score = min(1.0, a * 1000)
            smoothness_score = 1.0 - safe_divide(volatility_ratio, max_volatility_ratio)
            volume_score = min(
                1.0, safe_divide(late_volume / early_volume, min_volume_increase * 2)
            )

            components = np.array([curve_score, smoothness_score, volume_score])
            weights = np.array([curve_weight, smoothness_weight, volume_weight])
            signals[i] = calculate_weighted_score(components, weights)

        return signals

    @njit
    def detect_diamond_top_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        min_pattern_length: int,
        min_volatility_expansion: float,
        min_volatility_contraction: float,
        expansion_weight: float,
        contraction_weight: float,
        symmetry_weight: float,
    ) -> np.ndarray:
        """Détecte le pattern Diamond Top.

        Expansion de volatilité suivie d'une contraction formant un diamant.
        FIX: suppression diamond_high/diamond_low (dead variables — seuls les
        indices _idx sont utilisés pour le score de symétrie).
        """
        signals = np.zeros(len(open_prices), dtype=np.float64)

        for i in range(min_pattern_length, len(open_prices)):
            window_high = high_prices[i - min_pattern_length : i]
            window_low = low_prices[i - min_pattern_length : i]

            mid_point = min_pattern_length // 2
            first_half_high = window_high[:mid_point]
            first_half_low = window_low[:mid_point]
            second_half_high = window_high[mid_point:]
            second_half_low = window_low[mid_point:]

            first_half_range = np.max(first_half_high) - np.min(first_half_low)
            second_half_range = np.max(second_half_high) - np.min(second_half_low)

            combined_size = len(window_high) + len(window_low)
            combined_array = np.empty(combined_size)
            combined_array[: len(window_high)] = window_high
            combined_array[len(window_high) :] = window_low
            avg_price = safe_mean(combined_array)
            if avg_price == 0:
                continue

            first_half_volatility = safe_divide(first_half_range, avg_price)
            second_half_volatility = safe_divide(second_half_range, avg_price)

            if first_half_volatility < min_volatility_expansion:
                continue
            if second_half_volatility > first_half_volatility * (
                1 - min_volatility_contraction
            ):
                continue

            # Vérification de la tendance précédente (doit faire suite à une tendance haussière)
            prior_start = max(0, i - int(min_pattern_length * 1.5))
            if prior_start < i - min_pattern_length:
                prior_avg = safe_mean(
                    close_prices[prior_start : i - min_pattern_length]
                )
                if prior_avg >= avg_price:
                    continue

            # Un Diamond Top est un pattern de retournement baissier
            # Il doit casser à la baisse (clôture sous le prix moyen du pattern)
            if close_prices[i] >= avg_price:
                continue

            diamond_high_idx = np.argmax(window_high)
            diamond_low_idx = np.argmin(window_low)

            high_position = safe_divide(diamond_high_idx, min_pattern_length)
            low_position = safe_divide(diamond_low_idx, min_pattern_length)

            if (
                high_position < 0.2
                or high_position > 0.8
                or low_position < 0.2
                or low_position > 0.8
            ):
                continue

            # AUDIT FIX C-B3-6: symmetry_score pouvait être négatif si les deux
            # extrêmes sont proches des bords (ex: high_pos=0.8, low_pos=0.8 → -0.6).
            # Clamp à 0.0 pour éviter de récompenser les patterns asymétriques.
            symmetry_score = max(
                0.0,
                1.0 - abs(high_position - 0.5) - abs(low_position - 0.5),
            )

            expansion_ratio = safe_divide(
                first_half_volatility, min_volatility_expansion
            )
            expansion_score = min(1.0, expansion_ratio)

            contraction_ratio = safe_divide(
                (first_half_volatility - second_half_volatility), first_half_volatility
            )
            contraction_score = min(
                1.0, safe_divide(contraction_ratio, min_volatility_contraction)
            )

            components = np.array([expansion_score, contraction_score, symmetry_score])
            weights = np.array([expansion_weight, contraction_weight, symmetry_weight])

            signals[i] = calculate_weighted_score(components, weights)

        return signals

    @njit
    def detect_diamond_bottom_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        min_pattern_length: int,
        min_volatility_expansion: float,
        min_volatility_contraction: float,
        expansion_weight: float,
        contraction_weight: float,
        symmetry_weight: float,
    ) -> np.ndarray:
        """Détecte le pattern Diamond Bottom.

        FIX: suppression diamond_high/diamond_low (dead variables).
        """
        signals = np.zeros(len(open_prices), dtype=np.float64)

        for i in range(min_pattern_length, len(open_prices)):
            window_high = high_prices[i - min_pattern_length : i]
            window_low = low_prices[i - min_pattern_length : i]

            mid_point = min_pattern_length // 2
            first_half_high = window_high[:mid_point]
            first_half_low = window_low[:mid_point]
            second_half_high = window_high[mid_point:]
            second_half_low = window_low[mid_point:]

            first_half_range = np.max(first_half_high) - np.min(first_half_low)
            second_half_range = np.max(second_half_high) - np.min(second_half_low)

            combined_size = len(window_high) + len(window_low)
            combined_array = np.empty(combined_size)
            combined_array[: len(window_high)] = window_high
            combined_array[len(window_high) :] = window_low
            avg_price = safe_mean(combined_array)
            if avg_price == 0:
                continue

            first_half_volatility = safe_divide(first_half_range, avg_price)
            second_half_volatility = safe_divide(second_half_range, avg_price)

            if first_half_volatility < min_volatility_expansion:
                continue
            if second_half_volatility > first_half_volatility * (
                1 - min_volatility_contraction
            ):
                continue

            # Vérification de la tendance précédente (doit faire suite à une tendance baissière)
            prior_start = max(0, i - int(min_pattern_length * 1.5))
            if prior_start < i - min_pattern_length:
                prior_avg = safe_mean(
                    close_prices[prior_start : i - min_pattern_length]
                )
                if prior_avg <= avg_price:
                    continue

            # Un Diamond Bottom est un pattern de retournement haussier
            # Il doit casser à la hausse (clôture au-dessus du prix moyen du pattern)
            if close_prices[i] <= avg_price:
                continue

            # FIX: diamond_low/diamond_high supprimés (dead variables)
            diamond_low_idx = np.argmin(window_low)
            diamond_high_idx = np.argmax(window_high)

            low_position = safe_divide(diamond_low_idx, min_pattern_length)
            high_position = safe_divide(diamond_high_idx, min_pattern_length)

            if (
                low_position < 0.2
                or low_position > 0.8
                or high_position < 0.2
                or high_position > 0.8
            ):
                continue

            # FIX: symmetry_score pouvait être négatif. Clamp à 0.0.
            symmetry_score = max(
                0.0,
                1.0 - abs(low_position - 0.5) - abs(high_position - 0.5),
            )

            expansion_ratio = safe_divide(
                first_half_volatility, min_volatility_expansion
            )
            expansion_score = min(1.0, expansion_ratio)

            contraction_ratio = safe_divide(
                (first_half_volatility - second_half_volatility), first_half_volatility
            )
            contraction_score = min(
                1.0, safe_divide(contraction_ratio, min_volatility_contraction)
            )

            components = np.array([expansion_score, contraction_score, symmetry_score])
            weights = np.array([expansion_weight, contraction_weight, symmetry_weight])

            signals[i] = calculate_weighted_score(components, weights)

        return signals

    @njit
    def detect_island_top_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        min_gap_ratio: float,
        max_island_duration: int,
        min_volume_confirmation: float,
        gap_weight: float,
        isolation_weight: float,
        volume_weight: float,
    ) -> np.ndarray:
        """Détection du pattern Island Top.

        Définition : une zone de prix isolée par un gap haussier à l'entrée
        et un gap baissier à la sortie, dont le bas de l'île reste AU-DESSUS
        des hauts des barres adjacentes (isolation complète).

        Signal de retournement baissier — le prix monte par gap, stagne sur
        l'île, puis redescend par gap, laissant l'île entièrement isolée.

        Corrections vs version précédente :
        1. island_low utilisé pour valider l'isolation COMPLÈTE de l'île :
           le bas de l'île doit dépasser le haut des barres environnantes.
           Sans ce check, des structures où l'île chevauchait les barres
           adjacentes passaient malgré les gaps — ce n'est pas une île.
        2. Validation que le gap up (entrée) précède le gap down (sortie)
           et que la durée de l'île est dans les bornes.
        3. Calcul des tailles de gap normalisées pour le gap_score gradué
           (les deux gaps contribuent au score, pas seulement leur présence).
        4. isolation_score basé sur la profondeur réelle de l'isolation
           (distance entre island_low et surrounding_high), pas seulement
           sur la durée de l'île.
        5. volume_score pénalisant si insuffisant, au lieu de silencieusement
           ignorer via `continue` qui éliminait les assets à faible volume.

        Args:
            open_prices, high_prices, low_prices, close_prices : arrays OHLC
            volume                  : array volume
            min_gap_ratio           : taille minimum des gaps en % (ex: 0.01)
            max_island_duration     : durée max de l'île en barres (ex: 5)
            min_volume_confirmation : ratio volume île/avant (ex: 1.3)
            gap_weight              : poids du score gap dans le score final
            isolation_weight        : poids du score isolation
            volume_weight           : poids du score volume
            close_prices: TODO: documenter.
            high_prices: TODO: documenter.
            low_prices: TODO: documenter.
            open_prices: TODO: documenter.

        Returns:
            np.ndarray float64 : scores entre 0.0 et 1.0
        """
        n = len(open_prices)
        signals = np.zeros(n, dtype=np.float64)

        # Fenêtre minimale : 1 barre avant + île + 1 barre après
        min_lookback = max_island_duration + 2

        for i in range(min_lookback, n):
            # ── 1. Recherche du gap haussier d'ouverture ────────────────────
            # Gap up : low[j] > high[j-1] — la barre j ouvre en gap au-dessus
            # de la barre précédente. C'est le début de l'île.
            gap_up_found = False
            gap_up_idx = -1
            gap_up_size = 0.0

            search_start = i - max_island_duration - 1
            if search_start < 1:
                search_start = 1

            for j in range(search_start, i - 1):
                if high_prices[j - 1] <= 0:
                    continue
                gap_size = low_prices[j] - high_prices[j - 1]
                gap_ratio = safe_divide(gap_size, high_prices[j - 1])
                if gap_ratio >= min_gap_ratio:
                    gap_up_found = True
                    gap_up_idx = j
                    gap_up_size = gap_size
                    break

            if not gap_up_found:
                continue

            # ── 2. Recherche du gap baissier de fermeture ───────────────────

            gap_down_found = False
            gap_down_idx = -1
            gap_down_size = 0.0

            for j in range(gap_up_idx + 1, i):
                if j + 1 >= n:
                    break
                if high_prices[j] <= 0:
                    continue
                gap_size = high_prices[j] - low_prices[j + 1]
                gap_ratio = safe_divide(gap_size, high_prices[j])
                if gap_ratio >= min_gap_ratio:
                    gap_down_found = True
                    gap_down_idx = j
                    gap_down_size = gap_size
                    break

            if not gap_down_found:
                continue

            # Le signal est émis le lendemain du gap down (barre i = gap_down_idx + 1)
            if gap_down_idx + 1 != i:
                continue

            # ── 3. Validation de la durée de l'île ──────────────────────────
            # Durée = nombre de barres entre gap_up et gap_down (inclusif)
            island_duration = gap_down_idx - gap_up_idx + 1
            if island_duration <= 0 or island_duration > max_island_duration:
                continue

            # ── 4. Calcul des extrêmes de l'île ─────────────────────────────
            island_start = gap_up_idx
            island_end = gap_down_idx + 1  # +1 pour inclure dans le slice

            if island_end > n:
                continue

            island_high = np.max(high_prices[island_start:island_end])
            island_low = np.min(low_prices[island_start:island_end])

            if island_high <= 0 or island_low <= 0:
                continue

            # ── 5. Validation de l'isolation COMPLÈTE ───────────────────────
            # Contexte autour de l'île (5 barres de chaque côté)
            pre_start = max(0, gap_up_idx - 5)
            post_end = min(n, gap_down_idx + 6)

            if gap_up_idx <= pre_start:
                continue

            pre_high = np.max(high_prices[pre_start:gap_up_idx])
            post_high = np.max(high_prices[gap_down_idx + 1 : post_end])

            surrounding_high = max(pre_high, post_high)
            if surrounding_high <= 0:
                continue

            # Condition 1 (déjà présente) : le haut de l'île dépasse les hauts environnants
            if island_high <= surrounding_high:
                continue

            # Condition 2 (FIX) : le BAS de l'île est AU-DESSUS des hauts environnants
            # C'est la définition d'une vraie isolation par gap.
            # Sans cette condition, l'île peut chevaucher les prix adjacents.
            if island_low <= surrounding_high:
                continue

            # ── 6. Score volume ──────────────────────────────────────────────
            # Pas de continue si volume insuffisant — score pénalisé à la place
            # pour ne pas éliminer les actifs forex à volume non significatif
            volume_score = 0.5  # valeur neutre par défaut

            if len(volume) > gap_down_idx and gap_up_idx >= 5:
                island_vol = safe_mean(volume[island_start:island_end])
                pre_vol = safe_mean(volume[pre_start:gap_up_idx])

                if pre_vol > 0 and island_vol > 0:
                    vol_ratio = safe_divide(island_vol, pre_vol)
                    if vol_ratio >= min_volume_confirmation:
                        volume_score = min(
                            1.0, safe_divide(vol_ratio, min_volume_confirmation * 1.5)
                        )
                    else:
                        # Volume insuffisant : score dégradé proportionnellement
                        volume_score = safe_divide(vol_ratio, min_volume_confirmation)
                        if volume_score > 1.0:
                            volume_score = 1.0

            # ── 7. Score gap (taille des deux gaps combinée) ─────────────────
            gap_up_ratio = safe_divide(
                gap_up_size, max(high_prices[gap_up_idx - 1], 1e-10)
            )
            gap_down_ratio = safe_divide(
                gap_down_size, max(high_prices[gap_down_idx], 1e-10)
            )

            avg_gap_ratio = (gap_up_ratio + gap_down_ratio) / 2.0
            gap_score = min(1.0, safe_divide(avg_gap_ratio, min_gap_ratio * 2.0))

            # ── 8. Score isolation (profondeur de l'isolation) ───────────────
            # Plus island_low est au-dessus de surrounding_high, plus l'île est isolée
            isolation_depth = safe_divide(
                (island_low - surrounding_high), surrounding_high
            )
            isolation_score = min(
                1.0, safe_divide(isolation_depth, min_gap_ratio * 2.0)
            )
            if isolation_score < 0.0:
                isolation_score = 0.0

            # ── 9. Score final pondéré ───────────────────────────────────────
            components = np.array([gap_score, isolation_score, volume_score])
            weights = np.array([gap_weight, isolation_weight, volume_weight])

            signals[i] = calculate_weighted_score(components, weights)

        return signals

    # === PATTERNS CHARTISTES DE CONTINUATION - TRIANGLES ===

    @njit
    def detect_ascending_triangle_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        min_pattern_length: int,
        min_convergence: float,
        max_resistance_slope: float,
        min_support_slope: float,
        resistance_weight: float,
        support_weight: float,
        volume_weight: float,
    ) -> np.ndarray:
        """Détecte le pattern Ascending Triangle.

        Résistance horizontale avec support montant qui converge.
        """
        signals = np.zeros(len(open_prices), dtype=np.float64)

        for i in range(min_pattern_length + 5, len(open_prices)):
            # Recherche des points de résistance (pics)
            start_idx = i - min_pattern_length
            res_idx = np.empty(min_pattern_length, dtype=np.int64)
            res_price = np.empty(min_pattern_length, dtype=np.float64)
            res_count = 0

            sup_idx = np.empty(min_pattern_length, dtype=np.int64)
            sup_price = np.empty(min_pattern_length, dtype=np.float64)
            sup_count = 0

            for j in range(start_idx + 1, i - 1):
                if (
                    high_prices[j] > high_prices[j - 1]
                    and high_prices[j] > high_prices[j + 1]
                ):
                    if res_count < min_pattern_length:
                        res_idx[res_count] = j
                        res_price[res_count] = high_prices[j]
                        res_count += 1
                if (
                    low_prices[j] < low_prices[j - 1]
                    and low_prices[j] < low_prices[j + 1]
                ):
                    if sup_count < min_pattern_length:
                        sup_idx[sup_count] = j
                        sup_price[sup_count] = low_prices[j]
                        sup_count += 1

            if res_count < 2 or sup_count < 2:
                continue

            # AUDIT FIX C-B4-6: Remplacer la sélection par prix (argsort décroissant)
            # par une sélection temporelle via _find_first_last_by_time.
            # L'ancienne formule : sort_res = np.argsort(res_price)[::-1] prenait le
            # prix max et le 3ème prix max, sans garantie d'ordre chronologique
            # → pente de résistance potentiellement inversée temporellement.
            # Correction : utiliser les 1er/dernier pivots chronologiques
            # (identique à detect_descending_triangle pour la résistance).
            x1, y1, x2, y2 = _find_first_last_by_time(
                res_idx[:res_count], res_price[:res_count], res_count
            )

            if x2 != x1 and y1 != 0.0:
                resistance_slope = safe_divide(
                    safe_divide((y2 - y1), float(x2 - x1)), y1
                )
                if abs(resistance_slope) > max_resistance_slope:
                    continue
            else:
                continue

            # Calcul de la pente de support (doit être positive - montante)
            sx1 = sup_idx[0]
            sy1 = sup_price[0]
            sx2 = sup_idx[sup_count - 1]
            sy2 = sup_price[sup_count - 1]

            if sx2 != sx1:
                support_slope = (
                    safe_divide((sy2 - sy1), float(abs(sx2 - sx1))) / sy1
                    if sy1 != 0
                    else 0.0
                )
                if support_slope < min_support_slope:
                    continue
            else:
                continue

            # Vérification de la convergence
            resistance_level = safe_mean(res_price[:res_count])
            latest_support = sup_price[sup_count - 1]

            convergence_ratio = safe_divide(
                (resistance_level - latest_support), resistance_level
            )
            if convergence_ratio < min_convergence:  # Trop convergé
                continue

            # Validation du volume (décroissant pendant la formation)
            volume_score = 1.0
            if len(volume) > i:
                early_volume = safe_mean(
                    volume[start_idx : start_idx + min_pattern_length // 3]
                )
                late_volume = safe_mean(volume[i - min_pattern_length // 3 : i])

                if early_volume > 0:
                    volume_decline = safe_divide(
                        (early_volume - late_volume), early_volume
                    )
                    volume_score = min(1.0, max(0.0, volume_decline * 2))

            # Calcul des scores
            resistance_score = 1.0 - abs(
                safe_divide((resistance_slope), max_resistance_slope)
            )
            support_score = min(1.0, safe_divide(support_slope, min_support_slope))

            components = np.array([resistance_score, support_score, volume_score])
            weights = np.array([resistance_weight, support_weight, volume_weight])

            signals[i] = calculate_weighted_score(components, weights)

        return signals

    @njit
    def detect_descending_triangle_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        min_pattern_length: int,
        min_convergence: float,
        max_support_slope: float,
        min_resistance_slope: float,
        support_weight: float,
        resistance_weight: float,
        volume_weight: float,
    ) -> np.ndarray:
        """Détecte le pattern Descending Triangle.

        Support horizontal avec résistance descendante qui converge.
        """
        signals = np.zeros(len(open_prices), dtype=np.float64)

        for i in range(min_pattern_length + 5, len(open_prices)):
            # Recherche des points de support et résistance
            start_idx = i - min_pattern_length
            res_idx = np.empty(min_pattern_length, dtype=np.int64)
            res_price = np.empty(min_pattern_length, dtype=np.float64)
            res_count = 0

            sup_idx = np.empty(min_pattern_length, dtype=np.int64)
            sup_price = np.empty(min_pattern_length, dtype=np.float64)
            sup_count = 0

            for j in range(start_idx + 1, i - 1):
                if (
                    high_prices[j] > high_prices[j - 1]
                    and high_prices[j] > high_prices[j + 1]
                ):
                    if res_count < min_pattern_length:
                        res_idx[res_count] = j
                        res_price[res_count] = high_prices[j]
                        res_count += 1
                if (
                    low_prices[j] < low_prices[j - 1]
                    and low_prices[j] < low_prices[j + 1]
                ):
                    if sup_count < min_pattern_length:
                        sup_idx[sup_count] = j
                        sup_price[sup_count] = low_prices[j]
                        sup_count += 1

            if res_count < 2 or sup_count < 2:
                continue

            # Calcul de la pente de support (doit être proche de 0 - horizontale)
            sort_sup = np.argsort(sup_price[:sup_count])  # croissant
            bot_k = min(3, sup_count)
            x1 = sup_idx[sort_sup[0]]
            y1 = sup_price[sort_sup[0]]
            x2 = sup_idx[sort_sup[bot_k - 1]]
            y2 = sup_price[sort_sup[bot_k - 1]]

            if x2 != x1:
                support_slope = (
                    safe_divide((y2 - y1), float(abs(x2 - x1))) / y1 if y1 != 0 else 0.0
                )
                if abs(support_slope) > abs(max_support_slope):
                    continue
            else:
                continue

            # Calcul de la pente de résistance (doit être négative - descendante)
            rx1 = res_idx[0]
            ry1 = res_price[0]
            rx2 = res_idx[res_count - 1]
            ry2 = res_price[res_count - 1]

            if rx2 != rx1:
                resistance_slope = (
                    safe_divide((ry2 - ry1), float(abs(rx2 - rx1))) / ry1
                    if ry1 != 0
                    else 0.0
                )
                if resistance_slope > min_resistance_slope:
                    continue
            else:
                continue

            # Vérification de la convergence (vectorisée)
            bot_sup_prices = np.empty(bot_k, dtype=np.float64)
            for k in range(bot_k):
                bot_sup_prices[k] = sup_price[sort_sup[k]]

            support_level = safe_mean(bot_sup_prices)
            latest_resistance = res_price[res_count - 1]

            convergence_ratio = safe_divide(
                (latest_resistance - support_level), latest_resistance
            )
            if convergence_ratio < min_convergence:  # Trop convergé
                continue

            # Validation du volume (décroissant pendant la formation)
            volume_score = 1.0
            if len(volume) > i:
                early_volume = safe_mean(
                    volume[start_idx : start_idx + min_pattern_length // 3]
                )
                late_volume = safe_mean(volume[i - min_pattern_length // 3 : i])

                if early_volume > 0:
                    volume_decline = safe_divide(
                        (early_volume - late_volume), early_volume
                    )
                    volume_score = min(
                        1.0, max(0.0, volume_decline * 2)
                    )  # Note: third parameter removed

            # Calcul des scores
            support_score = 1.0 - abs(
                safe_divide(support_slope, abs(max_support_slope))
            )
            # AUDIT FIX C-B4-3: Reformuler resistance_score pour qu'il soit réellement
            # discriminant. La formule 1/(slope + threshold) retourne toujours ~1.0
            # pour toute pente légèrement négative (le poids était un poids mort).
            # Nouvelle formule cohérente avec support_score : plus la pente de
            # résistance est fortement négative, moins bon le score.
            resistance_score = 1.0 - min(
                1.0, abs(safe_divide(resistance_slope, abs(min_resistance_slope) * 2.0))
            )

            components = np.array([support_score, resistance_score, volume_score])
            weights = np.array([support_weight, resistance_weight, volume_weight])

            signals[i] = calculate_weighted_score(components, weights)

        return signals

    # === PATTERNS CHARTISTES DE CONTINUATION - CONSOLIDATION ===

    @njit
    def detect_rectangle_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        min_pattern_length: int,
        max_slope_tolerance: float,
        min_touches: int,
        support_weight: float,
        resistance_weight: float,
        duration_weight: float,
    ) -> np.ndarray:
        """Détecte le pattern Rectangle.

        Niveaux de support et résistance horizontaux avec plusieurs touches.
        """
        signals = np.zeros(len(open_prices), dtype=np.float64)

        for i in range(min_pattern_length + 5, len(open_prices)):
            # AUDIT FIX C-B4-1: Remplacer listes Python de tuples par arrays pré-alloués
            # Les listes de tuples (j, price) et les list-comprehensions sont interdites
            # en Numba nopython — zéro détection rectangle en prod avant cette correction.
            start_idx = i - min_pattern_length
            max_pts = min_pattern_length + 4

            res_idx_arr = np.empty(max_pts, dtype=np.int64)
            res_price_arr = np.empty(max_pts, dtype=np.float64)
            sup_idx_arr = np.empty(max_pts, dtype=np.int64)
            sup_price_arr = np.empty(max_pts, dtype=np.float64)
            res_count = 0
            sup_count = 0

            for j in range(start_idx + 1, i - 1):
                # Détection des pics (résistance)
                if (
                    high_prices[j] > high_prices[j - 1]
                    and high_prices[j] > high_prices[j + 1]
                ):
                    if res_count < max_pts:
                        res_idx_arr[res_count] = j
                        res_price_arr[res_count] = high_prices[j]
                        res_count += 1

                # Détection des creux (support)
                if (
                    low_prices[j] < low_prices[j - 1]
                    and low_prices[j] < low_prices[j + 1]
                ):
                    if sup_count < max_pts:
                        sup_idx_arr[sup_count] = j
                        sup_price_arr[sup_count] = low_prices[j]
                        sup_count += 1

            # Besoin d'au moins min_touches // 2 points de chaque
            if res_count < min_touches // 2 or sup_count < min_touches // 2:
                continue

            # Calcul du niveau de résistance moyen via slices numpy (Numba-OK)
            resistance_level = safe_mean(res_price_arr[:res_count])
            resistance_std = (
                calculate_std(res_price_arr[:res_count]) if res_count > 1 else 0.0
            )

            # Calcul du niveau de support moyen
            support_level = safe_mean(sup_price_arr[:sup_count])
            support_std = (
                calculate_std(sup_price_arr[:sup_count]) if sup_count > 1 else 0.0
            )

            # Vérification de l'horizontalité (faible écart-type)
            resistance_tolerance = resistance_level * max_slope_tolerance
            support_tolerance = support_level * max_slope_tolerance

            if resistance_std > resistance_tolerance or support_std > support_tolerance:
                continue

            # Vérification de la séparation entre support et résistance
            range_ratio = safe_divide(
                (resistance_level - support_level), resistance_level
            )
            if range_ratio < 0.02:  # Trop étroit
                continue

            # Comptage des touches valides (boucle sur arrays, pas sur listes Python)
            resistance_touches = 0
            support_touches = 0

            for k in range(res_count):
                if abs(res_price_arr[k] - resistance_level) <= resistance_tolerance:
                    resistance_touches += 1

            for k in range(sup_count):
                if abs(sup_price_arr[k] - support_level) <= support_tolerance:
                    support_touches += 1

            total_touches = resistance_touches + support_touches
            if total_touches < min_touches:
                continue

            # Validation du volume (relativement stable) — np.std remplacé par calculate_std
            volume_score = 1.0
            if len(volume) > i:
                pattern_volume = volume[start_idx:i]
                vol_mean = safe_mean(pattern_volume)
                volume_cv = (
                    safe_divide(calculate_std(pattern_volume), vol_mean)
                    if vol_mean > 0
                    else 0.0
                )
                # P1-5 FIX: volume_score intégré dans le scoring
                volume_score = max(0.0, 1.0 - volume_cv)

            # Calcul des scores
            support_score = 1.0 - (
                safe_divide(support_std, support_tolerance)
                if support_tolerance > 0
                else 1.0
            )
            resistance_score = 1.0 - (
                safe_divide(resistance_std, resistance_tolerance)
                if resistance_tolerance > 0
                else 1.0
            )
            duration_score = min(
                1.0,
                (i - start_idx) / min_pattern_length
                if min_pattern_length != 0
                else 0.0,
            )

            # Bonus pour le nombre de touches
            touch_bonus = min(1.0, safe_divide(total_touches, min_touches))
            support_score *= touch_bonus
            resistance_score *= touch_bonus

            # P1-5 FIX: volume_score inclus dans les components
            components = np.array(
                [support_score, resistance_score, duration_score, volume_score]
            )
            weights = np.array(
                [
                    support_weight * 0.85,
                    resistance_weight * 0.85,
                    duration_weight * 0.85,
                    0.15,
                ]
            )

            signals[i] = calculate_weighted_score(components, weights)

        return signals

    @njit
    def detect_bear_flag_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        min_pole_length: int,
        min_flag_length: int,
        max_flag_slope: float,
        min_volume_decline: float,
        pole_weight: float,
        flag_weight: float,
        volume_weight: float,
    ) -> np.ndarray:
        """Détection Bear Flag — mât baissier + consolidation plate/légèrement haussière.

        CORRECTIONS v3 (symétrique du bull_flag) :
        C1 — min_flag_length : drapeau minimum 5 barres (était 3, hardcodé).
        C2 — min_pole_atrs   : durci à 3.0 (était 2.0).
        C3 — bearish_ratio   : durci à 0.60 (était 0.55).
        C4 — flag_slope      : borne haute +0.05 (était +0.10), borne basse -0.01 (était -0.02).
        C5 — volume_score    : plancher supprimé (était 0.2).
        C6 — Architecture    : meilleur mât identifié d'abord, drapeau évalué une seule fois.
        """
        n = len(open_prices)
        signals = np.zeros(n, dtype=np.float64)
        min_pole_atrs = 3.0
        atr_period = 14
        min_lookback = min_pole_length * 2 + min_flag_length + atr_period + 2

        for i in range(min_lookback, n):
            # ── Étape 1 : trouver le MEILLEUR mât baissier ─────────────────────
            best_pole_atrs = 0.0
            best_bear_ratio = 0.0
            pole_start_best = -1
            pole_end_best = -1

            flag_end_bound = i - min_flag_length

            for pole_end_c in range(
                flag_end_bound - 1,
                max(flag_end_bound - min_pole_length * 2 - 1, min_pole_length - 1),
                -1,
            ):
                for pole_start_c in range(
                    max(0, pole_end_c - min_pole_length * 2),
                    pole_end_c - min_pole_length + 1,
                ):
                    pole_length = pole_end_c - pole_start_c + 1
                    if pole_length < min_pole_length:
                        continue

                    pole_start_price = close_prices[pole_start_c]
                    pole_end_price = close_prices[pole_end_c]

                    if pole_start_price <= 0 or pole_end_price <= 0:
                        continue

                    # Le mât doit être baissier
                    if pole_end_price >= pole_start_price:
                        continue

                    pole_loss = pole_start_price - pole_end_price

                    # ATR moyen sur les barres du mât
                    _atr_sum = 0.0
                    _atr_cnt = 0
                    for _k in range(pole_start_c + 1, pole_end_c + 1):
                        if _k >= n or _k < 1:
                            continue
                        _tr = high_prices[_k] - low_prices[_k]
                        _pc = close_prices[_k - 1]
                        if _pc > 0:
                            _tr = max(
                                _tr,
                                abs(high_prices[_k] - _pc),
                                abs(low_prices[_k] - _pc),
                            )
                        if _tr > 0:
                            _atr_sum += _tr
                            _atr_cnt += 1

                    if _atr_cnt == 0:
                        continue
                    avg_atr = _atr_sum / _atr_cnt

                    # C2 : mât doit valoir >= 3 ATRs
                    pole_in_atrs = safe_divide(pole_loss, avg_atr)
                    if pole_in_atrs < min_pole_atrs:
                        continue

                    # C3 : consistance directionnelle >= 60% barres baissières
                    bearish_bars = 0
                    total_bars = 0
                    for j in range(pole_start_c, pole_end_c + 1):
                        if j < n:
                            total_bars += 1
                            if close_prices[j] < open_prices[j]:
                                bearish_bars += 1

                    if total_bars == 0:
                        continue
                    bear_ratio = safe_divide(bearish_bars, total_bars)
                    if bear_ratio < 0.60:
                        continue

                    # Garder le mât avec le plus d'ATRs (le plus fort)
                    if pole_in_atrs > best_pole_atrs:
                        best_pole_atrs = pole_in_atrs
                        best_bear_ratio = bear_ratio
                        pole_start_best = pole_start_c
                        pole_end_best = pole_end_c

            if pole_end_best == -1:
                continue

            # ── Étape 2 : évaluer le drapeau UNE seule fois ────────────────────
            flag_start = pole_end_best + 1
            flag_end = i
            flag_length = flag_end - flag_start + 1

            # C1 : longueur minimale réelle du drapeau
            if flag_length < min_flag_length:
                continue

            flag_start_price = close_prices[flag_start]
            flag_end_price = close_prices[flag_end]

            if flag_start_price <= 0 or flag_end_price <= 0:
                continue

            # Retracement max 50% du mât
            pole_loss_best = close_prices[pole_start_best] - close_prices[pole_end_best]
            flag_highest = np.max(high_prices[flag_start : flag_end + 1])
            if (flag_highest - close_prices[pole_end_best]) > (pole_loss_best * 0.50):
                continue

            # C4 : pente du drapeau — plat ou légèrement haussier uniquement
            flag_slope = safe_divide(
                (flag_end_price - flag_start_price),
                flag_start_price * flag_length,
            )
            if flag_slope < -0.01:  # rejet si trop baissier (continuation)
                continue
            if flag_slope > max_flag_slope:  # rejet si trop haussier (> +0.05)
                continue

            # ── Étape 3 : volume ───────────────────────────────────────────────
            # C5 : pas de plancher artificiel
            volume_score = 0.3  # valeur neutre par défaut
            if len(volume) > i:
                pole_vol_sum = 0.0
                pole_vol_cnt = 0
                flag_vol_sum = 0.0
                flag_vol_cnt = 0

                for j in range(pole_start_best, pole_end_best + 1):
                    if j < len(volume) and volume[j] > 0:
                        pole_vol_sum += volume[j]
                        pole_vol_cnt += 1

                for j in range(flag_start, flag_end + 1):
                    if j < len(volume) and volume[j] > 0:
                        flag_vol_sum += volume[j]
                        flag_vol_cnt += 1

                if pole_vol_cnt > 0 and flag_vol_cnt > 0:
                    pole_avg = safe_divide(pole_vol_sum, pole_vol_cnt)
                    flag_avg = safe_divide(flag_vol_sum, flag_vol_cnt)
                    if pole_avg > 0:
                        vol_decline = safe_divide((pole_avg - flag_avg), pole_avg)
                        if vol_decline >= min_volume_decline:
                            volume_score = min(
                                1.0,
                                safe_divide(vol_decline, min_volume_decline * 1.5),
                            )
                        elif vol_decline > 0:
                            volume_score = safe_divide(vol_decline, min_volume_decline)
                        else:
                            # Volume stable ou croissant dans le flag → signal négatif
                            volume_score = 0.0

            # ── Étape 4 : score final ──────────────────────────────────────────
            pole_score = min(1.0, safe_divide(best_pole_atrs, min_pole_atrs * 2.0))
            pole_score *= min(1.0, best_bear_ratio * 1.5)

            _abs_slope = flag_slope if flag_slope >= 0 else -flag_slope
            _ref_slope = abs(max_flag_slope) if max_flag_slope != 0 else 0.05
            flag_score = 1.0 - min(1.0, safe_divide(_abs_slope, _ref_slope))

            components = np.array([pole_score, flag_score, volume_score])
            weights = np.array([pole_weight, flag_weight, volume_weight])

            signals[i] = calculate_weighted_score(components, weights)

        return signals

    @njit
    def detect_bull_pennant_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        min_pole_length: int,
        max_pennant_length: int,
        min_volume_decline: float,
        pole_weight: float,
        pennant_weight: float,
        volume_weight: float,
    ) -> np.ndarray:
        """Détecte le pattern Bull Pennant.

        Mât haussier suivi d'une consolidation triangulaire (fanion).
        """
        signals = np.zeros(len(open_prices), dtype=np.float64)

        for i in range(min_pole_length * 2 + 5, len(open_prices)):
            # Recherche du mât haussier (même logique que bull_flag)
            pole_found = False
            pole_start = -1
            pole_end = -1

            for pole_end_candidate in range(i - max_pennant_length - 2, i - 3):
                for pole_start_candidate in range(
                    max(0, pole_end_candidate - min_pole_length * 2),
                    pole_end_candidate - min_pole_length + 1,
                ):
                    pole_length = pole_end_candidate - pole_start_candidate + 1
                    if pole_length < min_pole_length:
                        continue

                    # Vérification que c'est un mouvement haussier fort
                    pole_start_price = close_prices[pole_start_candidate]
                    pole_end_price = close_prices[pole_end_candidate]

                    if pole_end_price <= pole_start_price:
                        continue

                    pole_gain = safe_divide(
                        (pole_end_price - pole_start_price), pole_start_price
                    )
                    if pole_gain < 0.05:  # Gain minimum de 5%
                        continue

                    # Vérification de la consistance haussière
                    bullish_bars = 0
                    for j in range(pole_start_candidate, pole_end_candidate + 1):
                        if is_bullish_candle(open_prices[j], close_prices[j]):
                            bullish_bars += 1

                    bullish_ratio = safe_divide(bullish_bars, pole_length)
                    if bullish_ratio < 0.6:
                        continue

                    pole_found = True
                    pole_start = pole_start_candidate
                    pole_end = pole_end_candidate
                    break

                if pole_found:
                    break

            if not pole_found:
                continue

            # Analyse du fanion (consolidation triangulaire)
            pennant_start = pole_end + 1
            pennant_end = i
            pennant_length = pennant_end - pennant_start + 1

            if pennant_length < 3 or pennant_length > max_pennant_length:
                continue

            # AUDIT FIX C-B4-2: Remplacer listes Python + sort(key=lambda) par arrays
            # pré-alloués + _find_first_last_by_time (identique au fix bear_pennant).
            # pennant_highs/pennant_lows avec .sort(key=lambda x: x[0]) → interdit Numba.
            _ph_max = pennant_length + 4
            _ph_idx = np.empty(_ph_max, dtype=np.int64)
            _ph_price = np.empty(_ph_max, dtype=np.float64)
            _pl_idx = np.empty(_ph_max, dtype=np.int64)
            _pl_price = np.empty(_ph_max, dtype=np.float64)
            _n_ph = 0
            _n_pl = 0

            for j in range(pennant_start + 1, pennant_end):
                if (
                    high_prices[j] > high_prices[j - 1]
                    and high_prices[j] > high_prices[j + 1]
                ):
                    if _n_ph < _ph_max:
                        _ph_idx[_n_ph] = j
                        _ph_price[_n_ph] = high_prices[j]
                        _n_ph += 1

                if (
                    low_prices[j] < low_prices[j - 1]
                    and low_prices[j] < low_prices[j + 1]
                ):
                    if _n_pl < _ph_max:
                        _pl_idx[_n_pl] = j
                        _pl_price[_n_pl] = low_prices[j]
                        _n_pl += 1

            # Besoin d'au moins 2 points de chaque pour former un triangle
            if _n_ph < 2 or _n_pl < 2:
                continue

            # _find_first_last_by_time retourne premier/dernier chronologiquement
            x1_h, y1_h, x2_h, y2_h = _find_first_last_by_time(
                _ph_idx[:_n_ph], _ph_price[:_n_ph], _n_ph
            )
            if x2_h == x1_h or y1_h == 0.0:
                continue
            upper_slope = safe_divide(
                safe_divide(y2_h - y1_h, float(x2_h - x1_h)), y1_h
            )

            x1_l, y1_l, x2_l, y2_l = _find_first_last_by_time(
                _pl_idx[:_n_pl], _pl_price[:_n_pl], _n_pl
            )
            if x2_l == x1_l or y1_l == 0.0:
                continue
            lower_slope = safe_divide(
                safe_divide(y2_l - y1_l, float(x2_l - x1_l)), y1_l
            )

            # Vérification de la convergence (pentes opposées)
            if upper_slope * lower_slope > 0:  # Même signe = pas de convergence
                continue

            # Validation du volume (décroissant dans le fanion)
            volume_score = 1.0
            if len(volume) > i:
                pole_volume = safe_mean(volume[pole_start : pole_end + 1])
                pennant_volume = safe_mean(volume[pennant_start : pennant_end + 1])

                if pole_volume > 0:
                    volume_decline = safe_divide(
                        (pole_volume - pennant_volume), pole_volume
                    )
                    volume_score = min(
                        1.0,
                        safe_divide(volume_decline, min_volume_decline)
                        if volume_decline >= min_volume_decline
                        else 0,
                    )
                    if volume_score == 0:
                        continue

            # Calcul des scores
            pole_gain = safe_divide(
                (close_prices[pole_end] - close_prices[pole_start]),
                close_prices[pole_start],
            )
            pole_score = min(1.0, pole_gain * 10)

            # Score du fanion basé sur la convergence
            convergence_quality = abs(upper_slope) + abs(lower_slope)
            pennant_score = min(1.0, convergence_quality * 20)

            components = np.array([pole_score, pennant_score, volume_score])
            weights = np.array([pole_weight, pennant_weight, volume_weight])

            signals[i] = calculate_weighted_score(components, weights)

        return signals

    @njit
    def detect_bear_pennant_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        min_pole_length: int,
        max_pennant_length: int,
        min_volume_decline: float,
        pole_weight: float,
        pennant_weight: float,
        volume_weight: float,
    ) -> np.ndarray:
        """Détecte le pattern Bear Pennant.

        Mât baissier suivi d'une consolidation triangulaire (fanion).
        """
        signals = np.zeros(len(open_prices), dtype=np.float64)

        for i in range(min_pole_length * 2 + 5, len(open_prices)):
            # Recherche du mât baissier (même logique que bear_flag)
            pole_found = False
            pole_start = -1
            pole_end = -1

            for pole_end_candidate in range(i - max_pennant_length - 2, i - 3):
                for pole_start_candidate in range(
                    max(0, pole_end_candidate - min_pole_length * 2),
                    pole_end_candidate - min_pole_length + 1,
                ):
                    pole_length = pole_end_candidate - pole_start_candidate + 1
                    if pole_length < min_pole_length:
                        continue

                    # Vérification que c'est un mouvement baissier fort
                    pole_start_price = close_prices[pole_start_candidate]
                    pole_end_price = close_prices[pole_end_candidate]

                    if pole_end_price >= pole_start_price:
                        continue

                    pole_loss = safe_divide(
                        (pole_start_price - pole_end_price), pole_start_price
                    )
                    if pole_loss < 0.05:  # Perte minimum de 5%
                        continue

                    # Vérification de la consistance baissière
                    bearish_bars = 0
                    for j in range(pole_start_candidate, pole_end_candidate + 1):
                        if is_bearish_candle(open_prices[j], close_prices[j]):
                            bearish_bars += 1

                    bearish_ratio = safe_divide(bearish_bars, pole_length)
                    if bearish_ratio < 0.6:
                        continue

                    pole_found = True
                    pole_start = pole_start_candidate
                    pole_end = pole_end_candidate
                    break

                if pole_found:
                    break

            if not pole_found:
                continue

            # Analyse du fanion (consolidation triangulaire)
            pennant_start = pole_end + 1
            pennant_end = i
            pennant_length = pennant_end - pennant_start + 1

            if pennant_length < 3 or pennant_length > max_pennant_length:
                continue

            # P0-4 FIX: Remplacement de list.sort(key=lambda) par _find_first_last_by_time
            # Collecter les pivots hauts/bas dans des arrays pré-alloués
            _ph_idx = np.empty(pennant_length + 4, dtype=np.int64)
            _ph_price = np.empty(pennant_length + 4, dtype=np.float64)
            _pl_idx = np.empty(pennant_length + 4, dtype=np.int64)
            _pl_price = np.empty(pennant_length + 4, dtype=np.float64)
            _n_ph = 0
            _n_pl = 0

            for j in range(pennant_start + 1, pennant_end):
                if (
                    high_prices[j] > high_prices[j - 1]
                    and high_prices[j] > high_prices[j + 1]
                ):
                    _ph_idx[_n_ph] = j
                    _ph_price[_n_ph] = high_prices[j]
                    _n_ph += 1

                if (
                    low_prices[j] < low_prices[j - 1]
                    and low_prices[j] < low_prices[j + 1]
                ):
                    _pl_idx[_n_pl] = j
                    _pl_price[_n_pl] = low_prices[j]
                    _n_pl += 1

            # Besoin d'au moins 2 points de chaque pour former un triangle
            if _n_ph < 2 or _n_pl < 2:
                continue

            # Calcul des pentes (doivent converger) — sans sort(lambda)
            x1, y1, x2, y2 = _find_first_last_by_time(
                _ph_idx[:_n_ph], _ph_price[:_n_ph], _n_ph
            )
            if x2 != x1:
                upper_slope = safe_divide(
                    safe_divide((y2 - y1), (x2 - x1)), y1 if y1 != 0 else 0.0
                )
            else:
                continue

            # P0-4 FIX: Pente des bas — utiliser _find_first_last_by_time sur les arrays pré-alloués
            x1, y1, x2, y2 = _find_first_last_by_time(
                _pl_idx[:_n_pl], _pl_price[:_n_pl], _n_pl
            )
            if x2 != x1:
                lower_slope = safe_divide(
                    safe_divide((y2 - y1), (x2 - x1)), y1 if y1 != 0 else 0.0
                )
            else:
                continue

            # Vérification de la convergence (pentes opposées)
            if upper_slope * lower_slope > 0:  # Même signe = pas de convergence
                continue

            # Validation du volume (décroissant dans le fanion)
            volume_score = 1.0
            if len(volume) > i:
                pole_volume = safe_mean(volume[pole_start : pole_end + 1])
                pennant_volume = safe_mean(volume[pennant_start : pennant_end + 1])

                if pole_volume > 0:
                    volume_decline = safe_divide(
                        (pole_volume - pennant_volume), pole_volume
                    )
                    volume_score = min(
                        1.0,
                        safe_divide(volume_decline, min_volume_decline)
                        if volume_decline >= min_volume_decline
                        else 0,
                    )
                    if volume_score == 0:
                        continue

            # Calcul des scores
            pole_loss = safe_divide(
                (close_prices[pole_start] - close_prices[pole_end]),
                close_prices[pole_start],
            )
            pole_score = min(1.0, pole_loss * 10)

            # Score du fanion basé sur la convergence
            convergence_quality = abs(upper_slope) + abs(lower_slope)
            pennant_score = min(1.0, convergence_quality * 20)

            components = np.array([pole_score, pennant_score, volume_score])
            weights = np.array([pole_weight, pennant_weight, volume_weight])

            signals[i] = calculate_weighted_score(components, weights)

        return signals

    # === PATTERNS CHARTISTES DE CONTINUATION - WEDGES ET PATTERNS AVANCÉS ===

    @njit
    def detect_rising_wedge_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        min_pattern_length: int,
        min_convergence: float,
        max_angle_diff: float,
        min_volume_decline: float,
        angle_weight: float,
        volume_weight: float,
        convergence_weight: float,
    ) -> np.ndarray:
        """VERSION CORRIGÉE - Détection du pattern Rising Wedge.

        Modifié pour utiliser la détection robuste de pivots adaptative comme Falling Wedge.
        """
        signals = np.zeros(len(open_prices), dtype=np.float64)
        for i in range(min_pattern_length, len(open_prices)):
            start_idx = i - min_pattern_length
            window_highs = high_prices[start_idx : i + 1]
            window_lows = low_prices[start_idx : i + 1]

            # ATR adaptatif au lieu de hardcodé
            _rw_atr_sum = 0.0
            _rw_atr_cnt = 0
            for _k in range(max(1, start_idx), i + 1):
                _rw_tr = max(
                    high_prices[_k] - low_prices[_k],
                    abs(high_prices[_k] - close_prices[_k - 1]),
                    abs(low_prices[_k] - close_prices[_k - 1]),
                )
                _rw_atr_sum += _rw_tr
                _rw_atr_cnt += 1
            _rw_atr = _rw_atr_sum / _rw_atr_cnt if _rw_atr_cnt > 0 else 0.015
            _rw_ref = close_prices[i] if close_prices[i] > 0 else 1.0
            _raw_thr = _rw_atr / _rw_ref
            _pivot_thr = max(0.008, min(0.025, _raw_thr))

            # Utiliser la détection de pivots pour trouver les points des lignes de tendance
            high_pivots_idx, high_pivots_price, _ = find_pivots_simple(
                window_highs, _pivot_thr, 3
            )
            low_pivots_idx, low_pivots_price, _ = find_pivots_simple(
                window_lows, _pivot_thr, 3
            )

            if len(high_pivots_idx) < 2 or len(low_pivots_idx) < 2:
                continue

            resistance_slope = calculate_linear_regression_slope(
                high_pivots_idx, high_pivots_price
            )
            support_slope = calculate_linear_regression_slope(
                low_pivots_idx, low_pivots_price
            )

            # Les deux pentes doivent être ascendantes
            if resistance_slope <= 0 or support_slope <= 0:
                continue

            # La ligne de support doit monter plus vite que la résistance
            if support_slope <= resistance_slope:
                continue

            # Le ratio des pentes ne doit pas être trop extrême
            slope_ratio = safe_divide(support_slope, resistance_slope)
            if slope_ratio > (max_angle_diff * 10):  # Tolérance augmentée
                continue

            # Le volume devrait décliner
            volume_score = 1.0 - (
                safe_divide(
                    np.mean(volume[i - 5 : i]),
                    np.mean(volume[start_idx : start_idx + 5]),
                )
                if np.mean(volume[start_idx : start_idx + 5]) > 0
                else 0
            )

            # Convergence robuste sur moyenne 3 barres
            _start_range = 0.0
            for _k in range(min(3, min_pattern_length)):
                _start_range += high_prices[start_idx + _k] - low_prices[start_idx + _k]
            _start_range /= min(3.0, float(min_pattern_length))

            _end_range = 0.0
            for _k in range(min(3, min_pattern_length)):
                _end_range += high_prices[i - 2 + _k] - low_prices[i - 2 + _k]
            _end_range /= min(3.0, float(min_pattern_length))

            if _start_range <= 0:
                continue
            convergence_score = 1.0 - safe_divide(_end_range, _start_range)
            slope_score = 1.0 - safe_divide(
                (support_slope - resistance_slope), support_slope
            )

            components = np.array(
                [convergence_score, slope_score, max(0.0, volume_score)]
            )
            weights = np.array([convergence_weight, angle_weight, volume_weight])
            signals[i] = calculate_weighted_score(components, weights)

        return signals

    @njit
    def detect_broadening_wedge_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        min_length: int,
        min_expansion_ratio: float,
        min_touches: int,
        expansion_weight: float,
        touch_weight: float,
        symmetry_weight: float,
    ) -> np.ndarray:
        """Détecte le pattern Broadening Wedge.

        Lignes de support et résistance qui s'écartent (expansion de volatilité).
        """
        signals = np.zeros(len(open_prices), dtype=np.float64)

        for i in range(min_length + 5, len(open_prices)):
            start_idx = i - min_length
            window_highs = high_prices[start_idx : i + 1]
            window_lows = low_prices[start_idx : i + 1]

            # Trouver les pivots dans la fenêtre
            high_pivots_idx, high_pivots_price, _ = find_pivots_simple(
                window_highs, 0.015, 3
            )
            low_pivots_idx, low_pivots_price, _ = find_pivots_simple(
                window_lows, 0.015, 3
            )

            n_resistance = len(high_pivots_idx)
            n_support = len(low_pivots_idx)

            # Besoin d'au moins min_touches // 2 points de chaque
            if n_resistance < min_touches // 2 or n_support < min_touches // 2:
                continue

            # Calcul de l'expansion du range
            early_range = safe_mean(
                high_prices[start_idx : start_idx + min_length // 3]
                - low_prices[start_idx : start_idx + min_length // 3]
            )
            late_range = safe_mean(
                high_prices[i - min_length // 3 : i]
                - low_prices[i - min_length // 3 : i]
            )

            if early_range <= 0.0:
                continue

            expansion_ratio = safe_divide((late_range - early_range), early_range)
            if expansion_ratio < min_expansion_ratio:
                continue

            if n_resistance >= 2 and n_support >= 2:
                # Régression linéaire sur tous les pivots pour des lignes robustes
                upper_slope = calculate_linear_regression_slope(
                    high_pivots_idx, high_pivots_price
                )
                lower_slope = calculate_linear_regression_slope(
                    low_pivots_idx, low_pivots_price
                )

                # Vérification de la divergence (pentes opposées)
                if upper_slope <= 0.0 or lower_slope >= 0.0:
                    continue
            else:
                continue

            # Comptage des touches
            total_touches = n_resistance + n_support
            if total_touches < min_touches:
                continue

            # Calcul des scores
            expansion_score = min(
                1.0, safe_divide(expansion_ratio, min_expansion_ratio)
            )
            touch_score = min(1.0, safe_divide(total_touches, min_touches))

            # Score de symétrie basé sur l'équilibre des pentes
            max_slope_val = max(abs(upper_slope), abs(lower_slope))
            symmetry_score = safe_divide(
                (1.0 - abs(abs(upper_slope) - abs(lower_slope))),
                max_slope_val if max_slope_val > 0.0 else 1.0,
            )
            symmetry_score = max(0.0, min(1.0, symmetry_score))

            components = np.array([expansion_score, touch_score, symmetry_score])
            weights = np.array([expansion_weight, touch_weight, symmetry_weight])

            signals[i] = calculate_weighted_score(components, weights)

        return signals

    @njit
    def detect_cup_handle_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        min_cup_length: int,
        max_handle_length: int,
        min_depth_ratio: float,
        cup_weight: float,
        handle_weight: float,
        volume_weight: float,
    ) -> np.ndarray:
        """Détecte le pattern Cup and Handle.

        Formation en forme de coupe suivie d'une petite consolidation (anse).
        """
        signals = np.zeros(len(open_prices), dtype=np.float64)

        for i in range(min_cup_length + max_handle_length + 5, len(open_prices)):
            # Recherche de la coupe
            cup_found = False
            cup_start = -1
            cup_bottom = -1
            cup_end = -1

            # Recherche dans une fenêtre appropriée
            for cup_end_candidate in range(i - max_handle_length, i - 2):
                for cup_start_candidate in range(
                    max(0, cup_end_candidate - min_cup_length * 2),
                    cup_end_candidate - min_cup_length,
                ):
                    cup_length = cup_end_candidate - cup_start_candidate + 1
                    if cup_length < min_cup_length:
                        continue

                    # Recherche du fond de la coupe
                    cup_segment = low_prices[
                        cup_start_candidate : cup_end_candidate + 1
                    ]
                    cup_bottom_candidate = cup_start_candidate + np.argmin(cup_segment)

                    # Vérification de la forme en U
                    cup_start_price = high_prices[cup_start_candidate]
                    cup_end_price = high_prices[cup_end_candidate]
                    cup_bottom_price = low_prices[cup_bottom_candidate]

                    # Les bords de la coupe doivent être à des niveaux similaires
                    rim_diff_ratio = abs(
                        safe_divide(
                            cup_start_price - cup_end_price,
                            max(cup_start_price, cup_end_price),
                        )
                    )
                    # AUDIT FIX C-B4-4b: Assouplir la tolérance des bords coupe 5%→10%.
                    # 5% élimine ~70% des coupes légitimes forex (bords jamais exactement
                    # au même niveau). 10% aligne sur la réalité des D1/H4.
                    if rim_diff_ratio > 0.10:  # Tolérance 10% au lieu de 5%
                        continue

                    # Profondeur de la coupe
                    avg_rim_price = safe_divide((cup_start_price + cup_end_price), 2)
                    depth_ratio = safe_divide(
                        (avg_rim_price - cup_bottom_price), avg_rim_price
                    )

                    if depth_ratio < min_depth_ratio:
                        continue

                    # Vérification de la forme arrondie (pas de V sharp)
                    # Le fond doit être relativement plat
                    bottom_segment_length = max(3, cup_length // 4)
                    bottom_start = max(
                        cup_bottom_candidate - bottom_segment_length // 2,
                        cup_start_candidate,
                    )
                    bottom_end = min(
                        cup_bottom_candidate + bottom_segment_length // 2,
                        cup_end_candidate,
                    )

                    bottom_segment = low_prices[bottom_start : bottom_end + 1]
                    # AUDIT FIX C-B4-4a: Remplacer np.std() par calculate_std().
                    # np.std() n'est pas supporté en Numba nopython sur des slices
                    # — zéro détection cup_handle avant cette correction.
                    bottom_mean = safe_mean(bottom_segment)
                    bottom_volatility = (
                        safe_divide(calculate_std(bottom_segment), bottom_mean)
                        if bottom_mean > 0
                        else 0.0
                    )

                    if bottom_volatility > 0.05:  # Trop volatil pour être une coupe
                        continue

                    cup_found = True
                    cup_start = cup_start_candidate
                    cup_bottom = cup_bottom_candidate
                    cup_end = cup_end_candidate
                    break

                if cup_found:
                    break

            if not cup_found:
                continue

            # Analyse de l'anse (handle)
            handle_start = cup_end + 1
            handle_end = i
            handle_length = handle_end - handle_start + 1

            if handle_length < 3 or handle_length > max_handle_length:
                continue

            # L'anse doit être une petite consolidation
            handle_high = np.max(high_prices[handle_start : handle_end + 1])
            handle_low = np.min(low_prices[handle_start : handle_end + 1])
            cup_rim_price = safe_divide(
                (high_prices[cup_start] + high_prices[cup_end]), 2
            )

            # L'anse ne doit pas dépasser le bord de la coupe
            if handle_high > cup_rim_price * 1.02:  # Tolérance de 2%
                continue

            # L'anse doit avoir une profondeur raisonnable
            handle_depth_ratio = safe_divide(
                (cup_rim_price - handle_low), cup_rim_price
            )
            if handle_depth_ratio > 0.15:  # Pas plus de 15% de retracement
                continue

            # Validation du volume (décroissant dans l'anse)
            volume_score = 1.0
            if len(volume) > i:
                cup_volume = safe_mean(volume[cup_start : cup_end + 1])
                handle_volume = safe_mean(volume[handle_start : handle_end + 1])

                if cup_volume > 0:
                    volume_decline = safe_divide(
                        (cup_volume - handle_volume), cup_volume
                    )
                    volume_score = min(
                        1.0, max(0.0, volume_decline * 2)
                    )  # Note: third parameter  removed

            # Calcul des scores
            cup_depth = safe_divide(
                (cup_rim_price - low_prices[cup_bottom]), cup_rim_price
            )
            cup_score = min(1.0, safe_divide(cup_depth, min_depth_ratio))

            handle_quality = 1.0 - safe_divide(
                handle_depth_ratio, 1.0
            )  # Moins profond = mieux
            handle_score = min(1.0, handle_quality)

            components = np.array([cup_score, handle_score, volume_score])
            weights = np.array([cup_weight, handle_weight, volume_weight])

            signals[i] = calculate_weighted_score(components, weights)

        return signals

    @njit
    def detect_measured_move_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        min_leg1_length: int,
        max_correction_ratio: float,
        min_target_ratio: float,
        leg1_weight: float,
        correction_weight: float,
        target_weight: float,
    ) -> np.ndarray:
        """Détecte le pattern Measured Move.

        Mouvement initial, correction, puis continuation avec objectif mesuré.
        """
        signals = np.zeros(len(open_prices), dtype=np.float64)

        for i in range(min_leg1_length * 3 + 5, len(open_prices)):
            # Recherche du premier leg (mouvement initial)
            leg1_found = False
            leg1_start = -1
            leg1_end = -1
            leg1_direction = 0  # 1 pour haussier, -1 pour baissier

            # Recherche dans une fenêtre appropriée
            for leg1_end_candidate in range(
                i - min_leg1_length * 2, i - min_leg1_length
            ):
                for leg1_start_candidate in range(
                    max(0, leg1_end_candidate - min_leg1_length * 2),
                    leg1_end_candidate - min_leg1_length + 1,
                ):
                    leg1_length = leg1_end_candidate - leg1_start_candidate + 1
                    if leg1_length < min_leg1_length:
                        continue

                    # Vérification du mouvement directionnel
                    leg1_start_price = close_prices[leg1_start_candidate]
                    leg1_end_price = close_prices[leg1_end_candidate]

                    if leg1_start_price == 0:
                        continue

                    leg1_move = safe_divide(
                        (leg1_end_price - leg1_start_price), leg1_start_price
                    )

                    # Mouvement minimum requis (0.5% — adapté forex)
                    if abs(leg1_move) < 0.005:
                        continue

                    # Déterminer la direction
                    if leg1_move > 0:
                        leg1_direction = 1  # Haussier
                    else:
                        leg1_direction = -1  # Baissier

                    # Vérification de la consistance directionnelle
                    consistent_bars = 0
                    for j in range(leg1_start_candidate, leg1_end_candidate + 1):
                        if leg1_direction == 1 and is_bullish_candle(
                            open_prices[j], close_prices[j]
                        ):
                            consistent_bars += 1
                        elif leg1_direction == -1 and is_bearish_candle(
                            open_prices[j], close_prices[j]
                        ):
                            consistent_bars += 1

                    consistency_ratio = safe_divide(consistent_bars, leg1_length)
                    if (
                        consistency_ratio < 0.6
                    ):  # Au moins 60% de barres dans la bonne direction
                        continue

                    leg1_found = True
                    leg1_start = leg1_start_candidate
                    leg1_end = leg1_end_candidate
                    break

                if leg1_found:
                    break

            if not leg1_found:
                continue

            # Recherche de la correction
            correction_start = leg1_end + 1
            correction_found = False
            correction_end = -1

            for correction_end_candidate in range(correction_start + 2, i - 2):
                correction_length = correction_end_candidate - correction_start + 1

                # La correction ne doit pas être trop longue
                if correction_length > min_leg1_length:
                    continue

                correction_start_price = close_prices[correction_start]
                correction_end_price = close_prices[correction_end_candidate]

                if correction_start_price == 0:
                    continue

                correction_move = safe_divide(
                    (correction_end_price - correction_start_price),
                    correction_start_price,
                )

                # La correction doit être dans la direction opposée au leg1
                if (
                    leg1_direction == 1 and correction_move >= 0
                ):  # Correction haussière après leg1 haussier
                    continue
                if (
                    leg1_direction == -1 and correction_move <= 0
                ):  # Correction baissière après leg1 baissier
                    continue

                # P2-4 FIX: La correction ne doit pas être trop profonde (tout en %)
                leg1_move_abs = abs(close_prices[leg1_end] - close_prices[leg1_start])
                leg1_pct = abs(safe_divide(leg1_move_abs, close_prices[leg1_start]))
                correction_pct = abs(correction_move)

                correction_ratio = (
                    safe_divide(correction_pct, leg1_pct) if leg1_pct > 0 else 0.0
                )

                if correction_ratio > max_correction_ratio:
                    continue

                correction_found = True
                correction_end = correction_end_candidate
                break

            if not correction_found:
                continue

            # Analyse du leg2 (continuation)
            leg2_start = correction_end + 1
            leg2_end = i
            leg2_length = leg2_end - leg2_start + 1

            if leg2_length < 3:  # Leg2 trop court
                continue

            # Le leg2 doit être dans la même direction que leg1
            leg2_start_price = close_prices[leg2_start]
            leg2_end_price = close_prices[leg2_end]

            if leg2_start_price == 0:
                continue

            leg2_move = safe_divide(
                (leg2_end_price - leg2_start_price), leg2_start_price
            )

            # Vérification de la direction
            if leg1_direction == 1 and leg2_move <= 0:  # Leg2 doit être haussier
                continue
            if leg1_direction == -1 and leg2_move >= 0:  # Leg2 doit être baissier
                continue

            # Calcul du ratio de target (leg2 par rapport à leg1)
            leg1_size = abs(close_prices[leg1_end] - close_prices[leg1_start])
            leg2_size = abs(leg2_move)

            target_ratio = safe_divide(leg2_size, leg1_size) if leg1_size > 0 else 0

            if target_ratio < min_target_ratio:
                continue

            # Calcul des scores
            # AUDIT FIX C-B4-5: Normaliser leg1_quality sur le % de mouvement
            # (leg1_move est déjà calculé en %, ligne ~8342).
            # Ancienne formule : leg1_size * 20 utilise la valeur absolue de prix
            # → EUR/USD (leg1_size≈0.005) = score 0.10 ; BTC (leg1_size≈500) = 1.0.
            # Discrimination purement dépendante du prix absolu, invalide cross-asset.
            # Nouvelle : normalisé sur 2% de mouvement → 0.5%=0.25, 1%=0.50, 2%+=1.0
            leg1_quality = min(1.0, abs(leg1_move) / 0.02)  # leg1_move déjà en %

            # P2-4 FIX: Utiliser le correction_ratio en pourcentage
            correction_quality = 1.0 - min(
                1.0, safe_divide(correction_ratio, max_correction_ratio)
            )

            target_quality = min(1.0, safe_divide(target_ratio, min_target_ratio))

            components = np.array([leg1_quality, correction_quality, target_quality])
            weights = np.array([leg1_weight, correction_weight, target_weight])

            signals[i] = calculate_weighted_score(components, weights)

        return signals

    # === PATTERNS DE GAP (8 patterns) ===

    @njit
    def detect_gap_up_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        min_gap_ratio: float,
        min_volume_increase: float,
        gap_weight: float,
        volume_weight: float,
        follow_through_weight: float,
    ) -> np.ndarray:
        """Détecte les gaps haussiers.

        Un gap up se produit quand le prix d'ouverture est significativement au-dessus de la clôture précédente.
        """
        signals = np.zeros(len(open_prices), dtype=np.float64)

        for i in range(1, len(open_prices)):
            # AUDIT FIX C-B8-1: dead guard `if i < 1` supprimé (toujours False dans range(1,...))
            prev_close = close_prices[i - 1]
            current_open = open_prices[i]
            current_close = close_prices[i]
            current_volume = volume[i]
            prev_volume = volume[i - 1]

            # Validation des données
            if prev_close <= 0 or current_open <= 0:
                continue

            # Calcul du gap ratio
            gap_ratio = safe_divide((current_open - prev_close), prev_close)

            # Vérification que c'est un gap haussier
            if gap_ratio < min_gap_ratio:
                continue

            # Vérification du suivi (follow-through)
            follow_through = 1.0 if current_close > current_open else 0.5

            # Calcul des scores
            gap_score = min(1.0, safe_divide(gap_ratio, (min_gap_ratio * 2)))
            # AUDIT FIX C-B8-1: scoring graduel au lieu de coupure binaire
            vol_ratio = (
                safe_divide(current_volume, prev_volume) if prev_volume > 0 else 1.0
            )
            volume_score = min(1.0, safe_divide(vol_ratio, min_volume_increase))
            follow_through_score = follow_through

            # Score pondéré
            components = np.array([gap_score, volume_score, follow_through_score])
            weights = np.array([gap_weight, volume_weight, follow_through_weight])

            signals[i] = calculate_weighted_score(components, weights)

        return signals

    @njit
    def detect_gap_down_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        min_gap_ratio: float,
        min_volume_increase: float,
        gap_weight: float,
        volume_weight: float,
        follow_through_weight: float,
    ) -> np.ndarray:
        """Détecte les gaps baissiers.

        Un gap down se produit quand le prix d'ouverture est significativement en dessous de la clôture précédente.
        """
        signals = np.zeros(len(open_prices), dtype=np.float64)

        for i in range(1, len(open_prices)):
            # CORRECTION P2: guard `if i < 1` supprimé (toujours False ici)
            prev_close = close_prices[i - 1]
            current_open = open_prices[i]
            current_close = close_prices[i]
            current_volume = volume[i]
            prev_volume = volume[i - 1]

            # Validation des données
            if prev_close <= 0 or current_open <= 0:
                continue

            # Calcul du gap ratio (négatif pour gap down)
            gap_ratio = safe_divide((prev_close - current_open), prev_close)

            # Vérification que c'est un gap baissier
            if gap_ratio < min_gap_ratio:
                continue

            # Vérification du suivi (follow-through)
            follow_through = 1.0 if current_close < current_open else 0.5

            # Calcul des scores
            gap_score = min(1.0, safe_divide(gap_ratio, (min_gap_ratio * 2)))
            # AUDIT FIX C-B8-1b: scoring graduel au lieu de coupure binaire
            vol_ratio = (
                safe_divide(current_volume, prev_volume) if prev_volume > 0 else 1.0
            )
            volume_score = min(1.0, safe_divide(vol_ratio, min_volume_increase))
            follow_through_score = follow_through

            # Score pondéré
            components = np.array([gap_score, volume_score, follow_through_score])
            weights = np.array([gap_weight, volume_weight, follow_through_weight])

            signals[i] = calculate_weighted_score(components, weights)

        return signals

    @njit
    def detect_gap_fill_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        min_gap_ratio: float,
        max_fill_time: int,
        fill_weight: float,
        time_weight: float,
    ) -> np.ndarray:
        """VERSION SANS LOOKAHEAD: vérifie si un gap passé est rempli aujourd'hui."""
        signals = np.zeros(len(open_prices), dtype=np.float64)
        for i in range(2, len(open_prices)):
            # Chercher dans le passé s'il y a un gap non rempli
            for j in range(max(1, i - max_fill_time), i):
                gap_up = open_prices[j] > high_prices[j - 1]
                gap_down = open_prices[j] < low_prices[j - 1]

                if gap_up:
                    gap_size = safe_divide(
                        open_prices[j] - high_prices[j - 1], high_prices[j - 1]
                    )
                    if gap_size < min_gap_ratio:
                        continue
                    # Le gap s'étend de high_prices[j-1] à open_prices[j]
                    gap_bottom = high_prices[j - 1]
                    gap_top = open_prices[j]

                    # Vérifier s'il a déjà été rempli entre j et i-1
                    already_filled = False
                    for k in range(j + 1, i):
                        if low_prices[k] <= gap_bottom:
                            already_filled = True
                            break
                    if already_filled:
                        continue

                    # Est-il rempli aujourd'hui ?
                    if low_prices[i] <= gap_bottom:
                        # AUDIT FIX C-B8-2: scoring graduel fill + time
                        _fill_depth = gap_bottom - low_prices[i]
                        _gap_range = (
                            gap_top - gap_bottom if gap_top > gap_bottom else 1e-10
                        )
                        _fill_score = min(1.0, safe_divide(_fill_depth, _gap_range))
                        _time_elapsed = float(i - j)
                        _time_score = max(
                            0.0, 1.0 - safe_divide(_time_elapsed, float(max_fill_time))
                        )
                        signals[i] = (
                            fill_weight * _fill_score + time_weight * _time_score
                        )
                        break  # Un seul gap rempli suffit pour signaler

                elif gap_down:
                    gap_size = safe_divide(
                        low_prices[j - 1] - open_prices[j], low_prices[j - 1]
                    )
                    if gap_size < min_gap_ratio:
                        continue
                    gap_top = low_prices[j - 1]
                    gap_bottom = open_prices[j]

                    already_filled = False
                    for k in range(j + 1, i):
                        if high_prices[k] >= gap_top:
                            already_filled = True
                            break
                    if already_filled:
                        continue

                    if high_prices[i] >= gap_top:
                        # AUDIT FIX C-B8-2: scoring graduel fill + time
                        _fill_depth2 = high_prices[i] - gap_top
                        _gap_range2 = (
                            gap_top - gap_bottom if gap_top > gap_bottom else 1e-10
                        )
                        _fill_score2 = min(1.0, safe_divide(_fill_depth2, _gap_range2))
                        _time_elapsed2 = float(i - j)
                        _time_score2 = max(
                            0.0, 1.0 - safe_divide(_time_elapsed2, float(max_fill_time))
                        )
                        signals[i] = (
                            fill_weight * _fill_score2 + time_weight * _time_score2
                        )
                        break

        return signals

    @njit
    def detect_runaway_gap_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        min_gap_ratio: float,
        min_volume_increase: float,
        min_trend_duration: int,
        gap_weight: float,
        volume_weight: float,
        trend_weight: float,
    ) -> np.ndarray:
        """Détecte les gaps de continuation (runaway gaps).

        Ces gaps se produisent au milieu d'une tendance forte existante.
        """
        signals = np.zeros(len(open_prices), dtype=np.float64)

        for i in range(min_trend_duration + 1, len(open_prices)):
            prev_close = close_prices[i - 1]
            current_open = open_prices[i]
            current_volume = volume[i]
            prev_volume = volume[i - 1] if i > 0 else current_volume

            # Validation des données
            if prev_close <= 0 or current_open <= 0:
                continue

            # Calcul du gap ratio
            gap_ratio = abs(safe_divide((current_open - prev_close), prev_close))

            if gap_ratio < min_gap_ratio:
                continue

            # Analyse de la tendance précédente
            lookback = int(min_trend_duration)  # Conversion explicite en int
            start_price = close_prices[i - lookback]
            end_price = close_prices[i - 1]

            if start_price <= 0:
                continue

            # Calcul de la force de la tendance
            trend_strength = abs(
                safe_divide((end_price - start_price), start_price)
            )  # Note: third parameter start_price removed
            trend_direction = 1.0 if end_price > start_price else -1.0
            gap_direction = 1.0 if current_open > prev_close else -1.0

            # Le gap doit être dans la même direction que la tendance
            if trend_direction * gap_direction <= 0:
                continue

            # Calcul de l'augmentation de volume
            volume_increase = (
                safe_divide(current_volume, prev_volume) if prev_volume > 0 else 1.0
            )

            # Calcul des scores
            gap_score = min(1.0, safe_divide(gap_ratio, (min_gap_ratio * 2)))
            volume_score = min(
                1.0,
                safe_divide(volume_increase, min_volume_increase)
                if volume_increase >= min_volume_increase
                else 0.0,
            )
            # CORRECTION P1: min() avec 3 args non supporté par Numba @njit
            # 5% de force de tendance = score 1.0, normalisé linéairement
            trend_score = min(1.0, safe_divide(trend_strength, 0.05))

            # Score pondéré
            components = np.array([gap_score, volume_score, trend_score])
            weights = np.array([gap_weight, volume_weight, trend_weight])

            signals[i] = calculate_weighted_score(components, weights)

        return signals

    @njit
    def detect_island_reversal_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        min_gap_ratio: float,
        min_island_duration: int,
        max_island_duration: int,
        gap_weight: float,
        isolation_weight: float,
        reversal_weight: float,
    ) -> np.ndarray:
        """Détecte les island reversals : deux gaps opposés isolant une zone de prix.

        PROBLÈMES DE L'ANCIENNE VERSION :

        Problème 1 — min_gap_ratio = 0.025 (2.5%) trop strict pour forex 24h/5j.
          Sur forex continu, les gaps de 2.5% n'existent pratiquement qu'à
          l'ouverture du dimanche. La quasi-totalité des island reversals forex
          réels ont des gaps de 0.3–1.0%.
          → Abaisser à 0.005 dans PATTERN_THRESHOLDS.

        Problème 2 — Condition d'isolation binaire.
          Ancienne : si un seul low/high touche le niveau pré-gap → rejet total.
          Sur forex continu, le prix revient souvent légèrement sur le niveau
          du gap sans pour autant invalider le pattern. On veut scorer la qualité
          de l'isolation, pas l'éliminer dès le moindre chevauchement.
          → Remplacement par isolation_score gradué :
            - Si isolation parfaite (aucun chevauchement) → 1.0
            - Si faible chevauchement → score entre 0.3 et 0.8
            - Si chevauchement important → score proche de 0, mais pas rejet dur

        L'isolation_score est calculé comme le ratio de barres qui respectent
        l'isolation. Une île dont 80% des barres sont bien isolées score 0.8,
        ce qui avec isolation_weight=0.30 reste un signal utile.
        """
        n = len(open_prices)
        signals = np.zeros(n, dtype=np.float64)

        for i in range(max_island_duration + 2, n):
            prev_close = close_prices[i - 1]
            current_open = open_prices[i]

            if prev_close <= 0 or current_open <= 0:
                continue

            # ── Gap de sortie ────────────────────────────────────────────
            exit_gap_ratio = abs(safe_divide(current_open - prev_close, prev_close))
            if exit_gap_ratio < min_gap_ratio:
                continue

            exit_gap_direction = 1.0 if current_open > prev_close else -1.0

            # ── Recherche du gap d'entrée ─────────────────────────────────
            for island_start in range(
                max(1, i - max_island_duration - 1), i - min_island_duration
            ):
                if island_start < 1:
                    continue

                entry_prev_close = close_prices[island_start - 1]
                entry_open = open_prices[island_start]

                if entry_prev_close <= 0 or entry_open <= 0:
                    continue

                entry_gap_ratio = abs(
                    safe_divide(entry_open - entry_prev_close, entry_prev_close)
                )
                if entry_gap_ratio < min_gap_ratio:
                    continue

                entry_gap_direction = 1.0 if entry_open > entry_prev_close else -1.0

                # Les deux gaps doivent être opposés
                if entry_gap_direction * exit_gap_direction >= 0:
                    continue

                island_duration = i - island_start
                if (
                    island_duration < min_island_duration
                    or island_duration > max_island_duration
                ):
                    continue

                # ── Isolation gradée (FIX : remplace binaire) ─────────────
                # On mesure la proportion de barres de l'île qui respectent
                # la séparation par rapport au niveau pré-gap d'entrée.
                isolated_bars = 0
                total_bars = 0

                for j in range(island_start, i):
                    total_bars += 1
                    if entry_gap_direction > 0:
                        # Gap up d'entrée : île haute → low doit rester > entry_prev_close
                        if low_prices[j] > entry_prev_close:
                            isolated_bars += 1
                    else:
                        # Gap down d'entrée : île basse → high doit rester < entry_prev_close
                        if high_prices[j] < entry_prev_close:
                            isolated_bars += 1

                # Ratio d'isolation : 1.0 = parfaitement isolée, 0.0 = aucune isolation
                isolation_ratio = (
                    safe_divide(isolated_bars, total_bars) if total_bars > 0 else 0.0
                )

                # Seuil minimum souple : au moins 50% des barres doivent être isolées
                # (remplace le rejet binaire total de l'ancienne version)
                if isolation_ratio < 0.5:
                    continue

                isolation_score = isolation_ratio  # directement proportionnel

                # ── Gap score ─────────────────────────────────────────────
                avg_gap_ratio = safe_divide(entry_gap_ratio + exit_gap_ratio, 2.0)
                gap_score = min(1.0, safe_divide(avg_gap_ratio, min_gap_ratio * 2.0))

                # ── Reversal score (durée courte = retournement plus net) ──
                duration_range = max_island_duration - min_island_duration
                if duration_range > 0:
                    reversal_score = max(
                        0.5,
                        1.0
                        - safe_divide(
                            island_duration - min_island_duration, duration_range
                        ),
                    )
                else:
                    reversal_score = 0.5

                # ── Score final ───────────────────────────────────────────
                components = np.array([gap_score, isolation_score, reversal_score])
                weights = np.array([gap_weight, isolation_weight, reversal_weight])

                signals[i] = max(
                    signals[i], calculate_weighted_score(components, weights)
                )

        return signals

    @njit
    def detect_gap_and_go_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        min_gap_ratio: float,
        min_volume_increase: float,
        min_continuation_ratio: float,
        gap_weight: float,
        volume_weight: float,
        continuation_weight: float,
    ) -> np.ndarray:
        """Détecte les patterns gap and go.

        Un gap suivi d'une continuation forte dans la même direction.
        """
        signals = np.zeros(len(open_prices), dtype=np.float64)

        # P0-1 FIX: Suppression du lookahead bias — continuation mesurée sur les 3 barres PASSÉES
        # La boucle commence à 4 (besoin de 4 bars: i-3, i-2, i-1, i pour le gap + continuation)
        for i in range(4, len(open_prices)):
            prev_close = close_prices[i - 1]
            current_open = open_prices[i]
            current_volume = volume[i]
            prev_volume = volume[i - 1]

            # Validation des données
            if prev_close <= 0 or current_open <= 0:
                continue

            # Calcul du gap ratio (gap entre la clôture précédente et l'ouverture courante)
            gap_ratio = abs(safe_divide((current_open - prev_close), prev_close))

            if gap_ratio < min_gap_ratio:
                continue

            gap_direction = 1.0 if current_open > prev_close else -1.0

            # Calcul de l'augmentation de volume
            volume_increase = (
                safe_divide(current_volume, prev_volume) if prev_volume > 0 else 1.0
            )

            # P0-1 FIX: Vérification de la continuation sur les 3 barres PASSÉES (i-2, i-1, i)
            # On vérifie que le prix a continué DANS LA DIRECTION DU GAP depuis l'ouverture du gap
            # Le "gap and go" est validé maintenant qu'on est à la barre i, avec 3 barres de confirmation
            continuation_score = 0.0
            total_continuation = 0.0
            gap_open_price = open_prices[
                i - 3
            ]  # Prix d'ouverture au moment du gap (i-3)

            for j in range(i - 2, i + 1):  # barres i-2, i-1, i (passé uniquement)
                price_move = (close_prices[j] - gap_open_price) * gap_direction
                continuation_ratio = (
                    safe_divide(price_move, gap_open_price)
                    if gap_open_price > 0
                    else 0.0
                )

                if continuation_ratio >= min_continuation_ratio:
                    continuation_score += 0.33
                    total_continuation += continuation_ratio

            # Le pattern nécessite une continuation significative sur au moins 2 barres sur 3
            if continuation_score < 0.66:
                continue

            # Calcul des scores
            gap_score = min(1.0, safe_divide(gap_ratio, (min_gap_ratio * 2)))
            volume_score = min(
                1.0,
                safe_divide(volume_increase, min_volume_increase)
                if volume_increase >= min_volume_increase
                else 0.0,
            )
            continuation_score = min(
                1.0, safe_divide(total_continuation, (min_continuation_ratio * 3))
            )

            # Score pondéré
            components = np.array([gap_score, volume_score, continuation_score])
            weights = np.array([gap_weight, volume_weight, continuation_weight])

            signals[i] = calculate_weighted_score(components, weights)

        return signals

    # === SUPPORT, RÉSISTANCE ET TENDANCES ===

    @njit
    def detect_support_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        window_size: int,
        min_touches: int,
        max_slope_tolerance: float,
        min_strength: float,
        touch_weight: float,
        slope_weight: float,
        strength_weight: float,
    ) -> np.ndarray:
        """Détecte un niveau de support en analysant une fenêtre de données.

        Attribue un score à la fin de la fenêtre si un support valide est trouvé.
        """
        signals = np.zeros(len(open_prices), dtype=np.float64)

        for i in range(window_size, len(open_prices)):
            start_idx = i - window_size
            window_lows = low_prices[start_idx:i]

            # Le niveau de support potentiel est le prix le plus bas de la fenêtre
            support_level = np.min(window_lows)

            if support_level <= 0:
                continue

            # Compter le nombre de "touches" proches de ce niveau
            touches = 0
            bounce_strength = 0.0
            last_touch_idx = -1

            for j in range(window_size):
                # Une "touche" est un prix bas proche du niveau de support
                if (
                    safe_divide((abs(window_lows[j] - support_level)), support_level)
                    < max_slope_tolerance
                ):
                    if last_touch_idx == -1 or (j - last_touch_idx) >= 3:
                        touches += 1
                        # Mesurer la force du rebond après la touche
                        bounce = safe_divide(
                            (high_prices[start_idx + j] - low_prices[start_idx + j]),
                            low_prices[start_idx + j],
                        )
                        bounce_strength += bounce
                    last_touch_idx = j

            if touches >= min_touches:
                avg_bounce = safe_divide(bounce_strength, touches if touches > 0 else 0)

                # Vérifier si la force moyenne des rebonds est suffisante
                if avg_bounce < min_strength:
                    continue

                # Calcul des scores
                touch_score = min(1.0, safe_divide(touches, (min_touches * 1.5)))

                # AUDIT FIX C-B5-2: slope_score=1.0 était hardcodé → le slope_weight
                # (0.30) ne discriminait jamais. Calcul réel : la déviation standard des
                # niveaux de touche mesure l'horizontalité réelle du support.
                # std=0 (touches parfaitement alignées) → slope_score=1.0
                # std≈max_slope_tolerance*support_level → slope_score→0.0
                if touches > 1:
                    _touch_vals = np.empty(touches, dtype=np.float64)
                    _tkc = 0
                    for _j in range(window_size):
                        if (
                            safe_divide(
                                abs(window_lows[_j] - support_level), support_level
                            )
                            < max_slope_tolerance
                        ):
                            if _tkc < touches:
                                _touch_vals[_tkc] = window_lows[_j]
                                _tkc += 1
                    touch_std = calculate_std(_touch_vals[:_tkc]) if _tkc > 1 else 0.0
                    max_touch_dev = support_level * max_slope_tolerance
                    slope_score = (
                        max(0.0, 1.0 - safe_divide(touch_std, max_touch_dev))
                        if max_touch_dev > 0
                        else 1.0
                    )
                else:
                    slope_score = 0.5  # 1 seule touche : pas d'info sur l'horizontalité

                strength_score = safe_divide((min(1.0, avg_bounce)), (min_strength * 2))

                components = np.array([touch_score, slope_score, strength_score])
                weights = np.array([touch_weight, slope_weight, strength_weight])

                # Assigner le score à la fin de la fenêtre de détection
                signals[i] = calculate_weighted_score(components, weights)

        return signals

    @njit
    def detect_resistance_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        window_size: int,
        min_touches: int,
        max_slope_tolerance: float,
        min_strength: float,
        touch_weight: float,
        slope_weight: float,
        strength_weight: float,
    ) -> np.ndarray:
        """Détecte un niveau de résistance en analysant une fenêtre de données.

        Attribue un score à la fin de la fenêtre si une résistance valide est trouvée.
        """
        signals = np.zeros(len(open_prices), dtype=np.float64)

        for i in range(window_size, len(open_prices)):
            start_idx = i - window_size
            window_highs = high_prices[start_idx:i]

            # Le niveau de résistance potentiel est le prix le plus haut de la fenêtre
            resistance_level = np.max(window_highs)

            if resistance_level <= 0:
                continue

            # Compter le nombre de "touches" proches de ce niveau
            touches = 0
            rejection_strength = 0.0
            last_touch_idx = -1

            for j in range(window_size):
                # Une "touche" est un prix haut proche du niveau de résistance
                if (
                    safe_divide(
                        (abs(window_highs[j] - resistance_level)), resistance_level
                    )
                    < max_slope_tolerance
                ):
                    if last_touch_idx == -1 or (j - last_touch_idx) >= 3:
                        touches += 1
                        # Mesurer la force du rejet après la touche
                        rejection = safe_divide(
                            (high_prices[start_idx + j] - low_prices[start_idx + j]),
                            high_prices[start_idx + j],
                        )
                        rejection_strength += rejection
                    last_touch_idx = j

            if touches >= min_touches:
                avg_rejection = safe_divide(
                    rejection_strength, touches if touches > 0 else 0
                )

                # Vérifier si la force moyenne des rejets est suffisante
                if avg_rejection < min_strength:
                    continue

                # Calcul des scores
                touch_score = min(1.0, safe_divide(touches, (min_touches * 1.5)))

                # AUDIT FIX C-B5-2 (symétrique support): calcul réel d'horizontalité
                # via l'écart-type des niveaux de touche (resistance).
                if touches > 1:
                    _touch_vals_r = np.empty(touches, dtype=np.float64)
                    _tkc_r = 0
                    for _j in range(window_size):
                        if (
                            safe_divide(
                                abs(window_highs[_j] - resistance_level),
                                resistance_level,
                            )
                            < max_slope_tolerance
                        ):
                            if _tkc_r < touches:
                                _touch_vals_r[_tkc_r] = window_highs[_j]
                                _tkc_r += 1
                    touch_std_r = (
                        calculate_std(_touch_vals_r[:_tkc_r]) if _tkc_r > 1 else 0.0
                    )
                    max_touch_dev_r = resistance_level * max_slope_tolerance
                    slope_score = (
                        max(0.0, 1.0 - safe_divide(touch_std_r, max_touch_dev_r))
                        if max_touch_dev_r > 0
                        else 1.0
                    )
                else:
                    slope_score = 0.5  # 1 seule touche : pas d'info sur l'horizontalité

                strength_score = safe_divide(
                    (min(1.0, avg_rejection)), (min_strength * 2)
                )

                components = np.array([touch_score, slope_score, strength_score])
                weights = np.array([touch_weight, slope_weight, strength_weight])

                signals[i] = calculate_weighted_score(components, weights)

        return signals

    # === PATTERNS SPÉCIAUX ET AVANCÉS (7 patterns) ===

    @njit
    def detect_wolfe_wave_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        min_wave_count: int,
        symmetry_tolerance: float,
        projection_accuracy: float,
        wave_weight: float,
        symmetry_weight: float,
        projection_weight: float,
    ) -> np.ndarray:
        """Détecte le pattern Wolfe Wave.

        Version simplifiée pour compatibilité Numba.
        """
        signals = np.zeros(len(open_prices), dtype=np.float64)

        # P0-6 FIX: Réécriture complète avec pivots O(n) — structure 5 points correcte
        # Complexité: O(n × k) avec k = nb_pivots_par_fenêtre (≤10) au lieu de O(n × 875)
        n = len(open_prices)

        for i in range(30, n):
            best_score = 0.0
            window_start = max(0, i - 50)

            # Extraire les pivots hauts et bas dans la fenêtre
            window_h = high_prices[window_start : i + 1]
            window_l = low_prices[window_start : i + 1]
            wlen = len(window_h)

            # Collecter pivots hauts (type=1) et bas (type=-1) en alternance
            piv_idx = np.empty(wlen, dtype=np.int64)
            piv_price = np.empty(wlen, dtype=np.float64)
            piv_type = np.empty(wlen, dtype=np.int64)
            n_piv = 0

            for j in range(1, wlen - 1):
                is_peak = (
                    window_h[j] > window_h[j - 1] and window_h[j] > window_h[j + 1]
                )
                is_trough = (
                    window_l[j] < window_l[j - 1] and window_l[j] < window_l[j + 1]
                )
                if is_peak and not is_trough:
                    piv_idx[n_piv] = j + window_start
                    piv_price[n_piv] = window_h[j]
                    piv_type[n_piv] = 1
                    n_piv += 1
                elif is_trough and not is_peak:
                    piv_idx[n_piv] = j + window_start
                    piv_price[n_piv] = window_l[j]
                    piv_type[n_piv] = -1
                    n_piv += 1

            # Il faut au moins 5 pivots pour un Wolfe Wave
            if n_piv < 5:
                continue

            # Glisser une fenêtre de 5 pivots consécutifs
            for k in range(4, n_piv):
                P1i = piv_idx[k - 4]
                P1p = piv_price[k - 4]
                P1t = piv_type[k - 4]
                P2i = piv_idx[k - 3]
                P2p = piv_price[k - 3]
                P2t = piv_type[k - 3]
                P3i = piv_idx[k - 2]
                P3p = piv_price[k - 2]
                P3t = piv_type[k - 2]
                P4i = piv_idx[k - 1]
                P4p = piv_price[k - 1]
                P4t = piv_type[k - 1]
                P5i = piv_idx[k]
                P5p = piv_price[k]
                P5t = piv_type[k]

                # Structure alternante nécessaire
                if not (
                    P1t == -1 and P2t == 1 and P3t == -1 and P4t == 1 and P5t == -1
                ):
                    continue

                # Règles Wolfe Wave haussier
                # P3 > P1 (creux montants), P4 < P2 (sommets descendants), P5 < P3
                if P3p <= P1p or P4p >= P2p or P5p >= P3p:
                    continue

                # Projection P5 sur la ligne P1-P3
                if P3i == P1i:
                    continue
                slope_13 = safe_divide((P3p - P1p), float(P3i - P1i))
                proj_P5_on_13 = P1p + slope_13 * (P5i - P1i)
                if proj_P5_on_13 == 0.0:
                    continue

                p5_dev = abs(safe_divide((P5p - proj_P5_on_13), abs(proj_P5_on_13)))
                if p5_dev > symmetry_tolerance:
                    continue

                # Symétrie temporelle des legs
                leg1 = P2i - P1i
                leg2 = P3i - P2i
                leg3 = P4i - P3i
                leg4 = P5i - P4i
                if leg1 == 0 or leg3 == 0:
                    continue

                sym_score = max(
                    0.0, 1.0 - abs(safe_divide(float(leg2 - leg1), float(leg1)))
                )
                sym_score2 = max(
                    0.0, 1.0 - abs(safe_divide(float(leg4 - leg3), float(leg3)))
                )
                symmetry_score = (sym_score + sym_score2) * 0.5
                if symmetry_score < (1.0 - symmetry_tolerance):
                    continue

                wave_score = 1.0
                projection_score = max(
                    0.0, 1.0 - safe_divide(p5_dev, symmetry_tolerance)
                )

                components = np.array([wave_score, symmetry_score, projection_score])
                weights = np.array([wave_weight, symmetry_weight, projection_weight])
                score = calculate_weighted_score(components, weights)

                target_bar = min(P5i + 1, n - 1)
                if score > signals[target_bar]:
                    signals[target_bar] = score

                if score > best_score:
                    best_score = score

        return signals

    @njit
    def detect_four_price_doji_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        max_body_ratio: float = 0.05,  # Augmenté de 0.01 à 0.05
        min_range_ratio: float = 0.003,  # Réduit de 0.01 à 0.003
        body_weight: float = 0.6,
        range_weight: float = 0.4,
    ) -> np.ndarray:
        """Version corrigée de four_price_doji avec seuils réalistes.

        CORRECTIONS:
        - Ratio relatif au lieu de valeur absolue
        - Seuils plus permissifs
        """
        signals = np.zeros(len(open_prices), dtype=np.float64)

        for i in range(len(open_prices)):
            open_val = open_prices[i]
            high_val = high_prices[i]
            low_val = low_prices[i]
            close_val = close_prices[i]

            if not validate_price_data(open_val, high_val, low_val, close_val):
                continue

            # Calcul du range total
            total_range = high_val - low_val
            if total_range <= 0:
                continue

            # Corps très petit (ratio relatif - CORRECTION CRITIQUE)
            body_size = abs(close_val - open_val)
            body_ratio = safe_divide(body_size, total_range)

            # Range minimum pour éviter le bruit
            range_ratio = safe_divide(total_range, close_val)

            # Conditions assouplies
            if body_ratio <= max_body_ratio and range_ratio >= min_range_ratio:
                # AUDIT FIX C-B5-1: np.unique() interdit en Numba nopython → crash silencieux.
                # Remplacement par comptage manuel de valeurs distinctes sur les 4 prix.
                vals = np.array([open_val, high_val, low_val, close_val])
                unique_count = 0
                for _vi in range(4):
                    _is_dup = False
                    for _vj in range(_vi):
                        if abs(vals[_vi] - vals[_vj]) < 1e-10:
                            _is_dup = True
                            break
                    if not _is_dup:
                        unique_count += 1

                if unique_count >= 3:  # Au moins 3 prix différents (assoupli de 4 à 3)
                    # P1-6 FIX: Score inversé — plus le corps est PETIT, meilleur est le score
                    # Un Doji parfait a un corps nul, donc body_score doit être maximal quand body_ratio ≈ 0
                    body_score = 1.0 - safe_divide(body_ratio, max_body_ratio)
                    range_score = safe_divide((min(1.0, range_ratio)), min_range_ratio)

                    components = np.array([body_score, range_score])
                    weights = np.array([body_weight, range_weight])

                    signals[i] = calculate_weighted_score(components, weights)

        return signals

    @njit
    def detect_shark_bull_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        tolerance: float = 0.15,
        min_pattern_size: float = 0.008,
        pattern_weight: float = 1.0,
    ) -> np.ndarray:
        """Shark Bull — Ratios Scott Carney (Harmonic Trading Vol.1).

        Points : O, X, A, B, C
        AB/XA : 1.13  1.618
        BC/AB : 1.618  2.24
        OC/OX : 0.886  1.13  (completion zone).
        """
        signals = np.zeros(len(open_prices), dtype=np.float64)
        window = 30  # Fenêtre de recherche max
        tol = tolerance  # ex: 0.10 = 10% de tolérance sur chaque ratio

        for i in range(window, len(open_prices)):
            best = 0.0
            # Chercher O = point de départ dans la fenêtre
            for o_idx in range(i - window, i - 4):
                O = low_prices[o_idx]  # Shark bullish part d'un creux O  # noqa: E741

                # Chercher X = premier sommet après O
                for x_idx in range(o_idx + 1, i - 3):
                    X = high_prices[x_idx]
                    XA_len = X - O
                    if XA_len <= 0 or XA_len < O * min_pattern_size:
                        continue

                    # Chercher A = premier creux après X
                    for a_idx in range(x_idx + 1, i - 2):
                        A = low_prices[a_idx]
                        if A >= X:
                            continue  # A doit être sous X
                        XA_retracement = (X - A) / XA_len
                        # AB/XA standard : Shark part généralement d'un AB entre 0.382 et 0.618
                        if not (0.382 - tol <= XA_retracement <= 0.618 + tol):
                            continue

                        # Chercher B = sommet après A
                        for b_idx in range(a_idx + 1, i - 1):
                            B = high_prices[b_idx]
                            AB = B - A
                            XA = XA_len
                            if AB <= 0:
                                continue
                            # AB/XA : 1.13  1.618
                            ab_xa_ratio = AB / XA
                            if not (
                                1.13 * (1 - tol) <= ab_xa_ratio <= 1.618 * (1 + tol)
                            ):
                                continue

                            # C = point de completion = close actuel ou creux récent
                            C = low_prices[i]
                            BC = B - C
                            if BC <= 0 or AB <= 0:
                                continue
                            # BC/AB : 1.618  2.24
                            bc_ab_ratio = BC / AB
                            if not (
                                1.618 * (1 - tol) <= bc_ab_ratio <= 2.24 * (1 + tol)
                            ):
                                continue

                            # OC/OX : 0.886  1.13 (completion zone du Shark)
                            OX = X - O
                            # P0-4 FIX: OC doit être mesuré en valeur absolue
                            OC = abs(C - O)
                            if OX <= 0:
                                continue
                            oc_ox_ratio = safe_divide(OC, OX)
                            if not (
                                0.886 * (1 - tol) <= oc_ox_ratio <= 1.13 * (1 + tol)
                            ):
                                continue

                            # Score basé sur la précision de chaque ratio
                            score_ab = (
                                1.0 - abs(ab_xa_ratio - 1.272) / 0.346
                            )  # centré sur 1.272
                            score_bc = (
                                1.0 - abs(bc_ab_ratio - 1.929) / 0.311
                            )  # centré sur 1.929
                            score_oc = (
                                1.0 - abs(oc_ox_ratio - 1.0) / 0.124
                            )  # centré sur 1.0
                            pattern_score = (score_ab + score_bc + score_oc) / 3.0
                            if pattern_score > best:
                                best = pattern_score
            signals[i] = best
        return signals

    @njit
    def detect_engulfing_bull_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        min_body_ratio: float,
        min_engulf_ratio: float,
        body_weight_1: float,
        body_weight_2: float,
        engulf_weight: float,
    ) -> np.ndarray:
        """VERSION CORRIGÉE - Détection du pattern Engulfing Bull.

        CORRECTIONS APPORTÉES:
        1. Validation complète des données OHLC avec validate_price_data()
        2. Logique d'englobement COMPLÈTE (corps + ombres) avec check_full_engulfment_bull()
        3. Vérifications strictes de direction des chandelles
        4. Système de scoring pondéré amélioré
        5. Gestion des erreurs silencieuses
        6. Utilisation des paramètres de configuration au lieu de valeurs codées en dur

        PATTERN:
        - Chandelier précédent: BAISSIER (close < open) avec corps significatif
        - Chandelier actuel: HAUSSIER (close > open) avec corps significatif
        - ENGLOBEMENT COMPLET: Le chandelier haussier englobe totalement le baissier (corps + ombres)
        - Ratio d'englobement: Le nouveau corps doit être min_engulf_ratio fois plus grand

        Args:
            open_prices, high_prices, low_prices, close_prices: Arrays des prix OHLC
            volume: Array des volumes
            min_body_ratio: Ratio minimum du corps par rapport au range (0.40 recommandé)
            min_engulf_ratio: Ratio minimum d'englobement (1.1 recommandé)
            body_weight_1, body_weight_2, engulf_weight: Poids pour le scoring pondéré
            body_weight_1: TODO: documenter.
            body_weight_2: TODO: documenter.
            close_prices: TODO: documenter.
            engulf_weight: TODO: documenter.
            high_prices: TODO: documenter.
            low_prices: TODO: documenter.
            open_prices: TODO: documenter.

        Returns:
            Array des scores de détection (0.0 à 1.0)
        """
        signals = np.zeros(len(open_prices), dtype=np.float64)

        for i in range(1, len(open_prices)):
            # === EXTRACTION DES DONNÉES ===

            # Chandelier précédent (doit être baissier)
            prev_open = open_prices[i - 1]
            prev_close = close_prices[i - 1]
            prev_high = high_prices[i - 1]
            prev_low = low_prices[i - 1]

            # Chandelier actuel (doit être haussier)
            curr_open = open_prices[i]
            curr_close = close_prices[i]
            curr_high = high_prices[i]
            curr_low = low_prices[i]

            # === VALIDATION DES DONNÉES ===

            # Validation OHLC stricte pour éviter les erreurs silencieuses
            if not validate_price_data(prev_open, prev_high, prev_low, prev_close):
                continue

            if not validate_price_data(curr_open, curr_high, curr_low, curr_close):
                continue

            # === VÉRIFICATION DES DIRECTIONS ===

            # Le chandelier précédent DOIT être baissier
            if not is_bearish_candle(prev_open, prev_close):
                continue

            # Le chandelier actuel DOIT être haussier
            if not is_bullish_candle(curr_open, curr_close):
                continue

            # === CALCUL DES MÉTRIQUES ===

            # Corps des chandeliers
            prev_body = prev_open - prev_close  # Corps baissier (positif)
            curr_body = curr_close - curr_open  # Corps haussier (positif)

            # Ranges totaux
            prev_range = prev_high - prev_low
            curr_range = curr_high - curr_low

            # Vérification des ranges valides
            if prev_range <= 0 or curr_range <= 0:
                continue

            # Ratios des corps par rapport aux ranges
            prev_body_ratio = safe_divide(prev_body, prev_range)
            curr_body_ratio = safe_divide(curr_body, curr_range)

            # === CONDITIONS DU PATTERN ===

            # Les deux corps doivent être significatifs
            if prev_body_ratio < min_body_ratio or curr_body_ratio < min_body_ratio:
                continue

            # VÉRIFICATION DE L'ENGLOBEMENT COMPLET (CORRECTION PRINCIPALE)
            if not check_full_engulfment_bull(
                prev_open,
                prev_high,
                prev_low,
                prev_close,
                curr_open,
                curr_high,
                curr_low,
                curr_close,
            ):
                continue

            # Ratio d'englobement (le nouveau corps doit être suffisamment plus grand)
            engulf_ratio = safe_divide(curr_body, prev_body)
            if engulf_ratio < min_engulf_ratio:
                continue

            # === CALCUL DU SCORE PONDÉRÉ ===

            # Score du corps précédent (plus il est grand, mieux c'est)
            body_score_1 = min(1.0, safe_divide(prev_body_ratio, min_body_ratio))

            # Score du corps actuel (plus il est grand, mieux c'est)
            body_score_2 = min(1.0, safe_divide(curr_body_ratio, min_body_ratio))

            # Score d'englobement (plus l'englobement est fort, mieux c'est)
            engulf_score = min(1.0, safe_divide(engulf_ratio, min_engulf_ratio * 2))

            # Combinaison pondérée
            components = np.array([body_score_1, body_score_2, engulf_score])
            weights = np.array([body_weight_1, body_weight_2, engulf_weight])

            final_score = calculate_weighted_score(components, weights)
            signals[i] = final_score

        return signals

    @njit
    def detect_engulfing_bear_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        min_body_ratio: float,
        min_engulf_ratio: float,
        body_weight_1: float,
        body_weight_2: float,
        engulf_weight: float,
    ) -> np.ndarray:
        """VERSION CORRIGÉE - Détection du pattern Engulfing Bear.

        CORRECTIONS APPORTÉES:
        1. Validation complète des données OHLC avec validate_price_data()
        2. Logique d'englobement COMPLÈTE (corps + ombres) avec check_full_engulfment_bear()
        3. Vérifications strictes de direction des chandelles
        4. Système de scoring pondéré amélioré
        5. Gestion des erreurs silencieuses
        6. Utilisation des paramètres de configuration au lieu de valeurs codées en dur

        PATTERN:
        - Chandelier précédent: HAUSSIER (close > open) avec corps significatif
        - Chandelier actuel: BAISSIER (close < open) avec corps significatif
        - ENGLOBEMENT COMPLET: Le chandelier baissier englobe totalement le haussier (corps + ombres)
        - Ratio d'englobement: Le nouveau corps doit être min_engulf_ratio fois plus grand

        Args:
            open_prices, high_prices, low_prices, close_prices: Arrays des prix OHLC
            volume: Array des volumes
            min_body_ratio: Ratio minimum du corps par rapport au range (0.40 recommandé)
            min_engulf_ratio: Ratio minimum d'englobement (1.1 recommandé)
            body_weight_1, body_weight_2, engulf_weight: Poids pour le scoring pondéré
            body_weight_1: TODO: documenter.
            body_weight_2: TODO: documenter.
            close_prices: TODO: documenter.
            engulf_weight: TODO: documenter.
            high_prices: TODO: documenter.
            low_prices: TODO: documenter.
            open_prices: TODO: documenter.

        Returns:
            Array des scores de détection (0.0 à 1.0)
        """
        signals = np.zeros(len(open_prices), dtype=np.float64)

        for i in range(1, len(open_prices)):
            # === EXTRACTION DES DONNÉES ===

            # Chandelier précédent (doit être haussier)
            prev_open = open_prices[i - 1]
            prev_close = close_prices[i - 1]
            prev_high = high_prices[i - 1]
            prev_low = low_prices[i - 1]

            # Chandelier actuel (doit être baissier)
            curr_open = open_prices[i]
            curr_close = close_prices[i]
            curr_high = high_prices[i]
            curr_low = low_prices[i]

            # === VALIDATION DES DONNÉES ===

            # Validation OHLC stricte pour éviter les erreurs silencieuses
            if not validate_price_data(prev_open, prev_high, prev_low, prev_close):
                continue

            if not validate_price_data(curr_open, curr_high, curr_low, curr_close):
                continue

            # === VÉRIFICATION DES DIRECTIONS ===

            # Le chandelier précédent DOIT être haussier
            if not is_bullish_candle(prev_open, prev_close):
                continue

            # Le chandelier actuel DOIT être baissier
            if not is_bearish_candle(curr_open, curr_close):
                continue

            # === CALCUL DES MÉTRIQUES ===

            # Corps des chandeliers
            prev_body = prev_close - prev_open  # Corps haussier (positif)
            curr_body = curr_open - curr_close  # Corps baissier (positif)

            # Ranges totaux
            prev_range = prev_high - prev_low
            curr_range = curr_high - curr_low

            # Vérification des ranges valides
            if prev_range <= 0 or curr_range <= 0:
                continue

            # Ratios des corps par rapport aux ranges
            prev_body_ratio = safe_divide(prev_body, prev_range)
            curr_body_ratio = safe_divide(curr_body, curr_range)

            # === CONDITIONS DU PATTERN ===

            # Les deux corps doivent être significatifs
            if prev_body_ratio < min_body_ratio or curr_body_ratio < min_body_ratio:
                continue

            # VÉRIFICATION DE L'ENGLOBEMENT COMPLET (CORRECTION PRINCIPALE)
            if not check_full_engulfment_bear(
                prev_open,
                prev_high,
                prev_low,
                prev_close,
                curr_open,
                curr_high,
                curr_low,
                curr_close,
            ):
                continue

            # Ratio d'englobement (le nouveau corps doit être suffisamment plus grand)
            engulf_ratio = safe_divide(curr_body, prev_body)
            if engulf_ratio < min_engulf_ratio:
                continue

            # === CALCUL DU SCORE PONDÉRÉ ===

            # Score du corps précédent (plus il est grand, mieux c'est)
            body_score_1 = min(1.0, safe_divide(prev_body_ratio, min_body_ratio))

            # Score du corps actuel (plus il est grand, mieux c'est)
            body_score_2 = min(1.0, safe_divide(curr_body_ratio, min_body_ratio))

            # Score d'englobement (plus l'englobement est fort, mieux c'est)
            engulf_score = min(1.0, safe_divide(engulf_ratio, min_engulf_ratio * 2))

            # Combinaison pondérée
            components = np.array([body_score_1, body_score_2, engulf_score])
            weights = np.array([body_weight_1, body_weight_2, engulf_weight])

            final_score = calculate_weighted_score(components, weights)
            signals[i] = final_score

        return signals

    @njit
    def detect_head_shoulders_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        min_shoulder_distance: int,
        min_head_height_ratio: float,
        max_shoulder_diff_ratio: float,
        head_weight: float,
        shoulder_weight: float,
        neckline_weight: float,
    ) -> np.ndarray:
        """Détection Head and Shoulders — 3 pics (épaule G, tête, épaule D).

        FIX 1 — seuil pivot ATR adaptatif [0.008, 0.025] (remplace 0.003 hardcodé).
        FIX 2 — min_head_height_ratio abaissé à 0.01 dans PATTERN_THRESHOLDS.
        Reste inchangé : logique 3 pivots, symétrie, neckline scoring.
        """
        n = len(open_prices)
        signals = np.zeros(n, dtype=np.float64)
        atr_period = 14
        min_window_size = min_shoulder_distance * 3

        for i in range(min_window_size + atr_period, n):
            # ── ATR adaptatif ─────────────────────────────────────────────
            _atr_start = i - atr_period + 1
            _atr_sum = 0.0
            _atr_cnt = 0

            for _k in range(_atr_start, i + 1):
                _tr = high_prices[_k] - low_prices[_k]
                _pc = close_prices[_k - 1]
                if _pc > 0.0:
                    _hi_pc = abs(high_prices[_k] - _pc)
                    _lo_pc = abs(low_prices[_k] - _pc)
                    if _hi_pc > _tr:
                        _tr = _hi_pc
                    if _lo_pc > _tr:
                        _tr = _lo_pc
                if _tr > 0.0:
                    _atr_sum += _tr
                    _atr_cnt += 1

            _ref = close_prices[i]
            if _ref <= 0.0:
                continue

            # Seuil pivot adaptatif [0.008, 0.025]
            if _atr_cnt > 0:
                _raw_thr = (_atr_sum / _atr_cnt) / _ref
                _pivot_thr = _raw_thr
                if _pivot_thr < 0.008:
                    _pivot_thr = 0.008
                if _pivot_thr > 0.025:
                    _pivot_thr = 0.025
            else:
                _pivot_thr = 0.012

            # ── Recherche des pivots hauts ────────────────────────────────
            start_idx = i - min_window_size

            pivots_idx, pivots_price, pivots_type = find_pivots_simple(
                high_prices[start_idx : i + 1], _pivot_thr, 3
            )

            high_pivots = []
            for j in range(len(pivots_type)):
                if pivots_type[j] == 1:
                    high_pivots.append((pivots_idx[j] + start_idx, pivots_price[j]))

            if len(high_pivots) < 3:
                continue

            left_idx, left_price = high_pivots[-3]
            head_idx, head_price = high_pivots[-2]
            right_idx, right_price = high_pivots[-1]

            # ── Validation données ────────────────────────────────────────
            if not (start_idx <= left_idx < head_idx < right_idx <= i):
                continue
            if head_price <= 0 or left_price <= 0 or right_price <= 0:
                continue

            # ── Conditions pattern ────────────────────────────────────────

            # 1. Tête plus haute que les deux épaules
            if head_price <= left_price or head_price <= right_price:
                continue

            # 2. Hauteur significative de la tête
            head_height_ratio = safe_divide(
                head_price - max(left_price, right_price), head_price
            )
            if head_height_ratio < min_head_height_ratio:
                continue

            # 3. Symétrie des épaules (scoring graduel)
            shoulder_ratio = safe_divide(right_price, left_price)
            shoulder_symmetry_score = 1.0 - min(
                1.0, abs(1.0 - shoulder_ratio) / max_shoulder_diff_ratio
            )
            if shoulder_symmetry_score < 0.3:
                continue

            # 4. Distance minimum entre les pics
            left_to_head_distance = head_idx - left_idx
            head_to_right_distance = right_idx - head_idx

            if (
                left_to_head_distance < min_shoulder_distance
                or head_to_right_distance < min_shoulder_distance
            ):
                continue

            # ── Scoring ───────────────────────────────────────────────────
            head_score = min(
                1.0, safe_divide(head_height_ratio, min_head_height_ratio * 2)
            )

            symmetry_score = shoulder_symmetry_score

            distance_ratio = safe_divide(head_to_right_distance, left_to_head_distance)
            neckline_score = 1.0 - min(1.0, abs(1.0 - distance_ratio) / 1.0)

            components = np.array([head_score, symmetry_score, neckline_score])
            weights = np.array([head_weight, shoulder_weight, neckline_weight])

            signals[i] = calculate_weighted_score(components, weights)

        return signals

    @njit
    def detect_inv_head_shoulders_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        min_shoulder_distance: int,
        min_head_depth_ratio: float,
        max_shoulder_diff_ratio: float,
        head_weight: float,
        shoulder_weight: float,
        neckline_weight: float,
    ) -> np.ndarray:
        """Détection Inverse Head and Shoulders — 3 creux (épaule G, tête, épaule D).

        FIX 1 — seuil pivot ATR adaptatif [0.008, 0.025] (remplace 0.003 hardcodé).
        FIX 2 — min_head_depth_ratio abaissé à 0.01 dans PATTERN_THRESHOLDS.
        Reste inchangé : logique 3 pivots, symétrie, neckline scoring.
        """
        n = len(open_prices)
        signals = np.zeros(n, dtype=np.float64)
        atr_period = 14
        min_window_size = min_shoulder_distance * 3

        for i in range(min_window_size + atr_period, n):
            # ── ATR adaptatif ─────────────────────────────────────────────
            _atr_start = i - atr_period + 1
            _atr_sum = 0.0
            _atr_cnt = 0

            for _k in range(_atr_start, i + 1):
                _tr = high_prices[_k] - low_prices[_k]
                _pc = close_prices[_k - 1]
                if _pc > 0.0:
                    _hi_pc = abs(high_prices[_k] - _pc)
                    _lo_pc = abs(low_prices[_k] - _pc)
                    if _hi_pc > _tr:
                        _tr = _hi_pc
                    if _lo_pc > _tr:
                        _tr = _lo_pc
                if _tr > 0.0:
                    _atr_sum += _tr
                    _atr_cnt += 1

            _ref = close_prices[i]
            if _ref <= 0.0:
                continue

            # Seuil pivot adaptatif [0.008, 0.025]
            if _atr_cnt > 0:
                _raw_thr = (_atr_sum / _atr_cnt) / _ref
                _pivot_thr = _raw_thr
                if _pivot_thr < 0.008:
                    _pivot_thr = 0.008
                if _pivot_thr > 0.025:
                    _pivot_thr = 0.025
            else:
                _pivot_thr = 0.012

            # ── Recherche des pivots bas ──────────────────────────────────
            start_idx = i - min_window_size

            pivots_idx, pivots_price, pivots_type = find_pivots_simple(
                low_prices[start_idx : i + 1], _pivot_thr, 3
            )

            low_pivots = []
            for j in range(len(pivots_type)):
                if pivots_type[j] == -1:
                    low_pivots.append((pivots_idx[j] + start_idx, pivots_price[j]))

            if len(low_pivots) < 3:
                continue

            left_idx, left_price = low_pivots[-3]
            head_idx, head_price = low_pivots[-2]
            right_idx, right_price = low_pivots[-1]

            # ── Validation données ────────────────────────────────────────
            if not (start_idx <= left_idx < head_idx < right_idx <= i):
                continue
            if head_price <= 0 or left_price <= 0 or right_price <= 0:
                continue

            # ── Conditions pattern ────────────────────────────────────────

            # 1. Tête plus basse que les deux épaules
            if head_price >= left_price or head_price >= right_price:
                continue

            # 2. Profondeur significative de la tête
            head_depth_ratio = safe_divide(
                min(left_price, right_price) - head_price,
                min(left_price, right_price),
            )
            if head_depth_ratio < min_head_depth_ratio:
                continue

            # 3. Symétrie des épaules (scoring graduel)
            shoulder_ratio = safe_divide(right_price, left_price)
            shoulder_symmetry_score = 1.0 - min(
                1.0, abs(1.0 - shoulder_ratio) / max_shoulder_diff_ratio
            )
            if shoulder_symmetry_score < 0.3:
                continue

            # 4. Distance minimum entre les creux
            left_to_head_distance = head_idx - left_idx
            head_to_right_distance = right_idx - head_idx

            if (
                left_to_head_distance < min_shoulder_distance
                or head_to_right_distance < min_shoulder_distance
            ):
                continue

            # ── Scoring ───────────────────────────────────────────────────
            head_score = min(
                1.0, safe_divide(head_depth_ratio, min_head_depth_ratio * 2)
            )

            symmetry_score = shoulder_symmetry_score

            distance_ratio = safe_divide(head_to_right_distance, left_to_head_distance)
            neckline_score = 1.0 - min(1.0, abs(1.0 - distance_ratio) / 1.0)

            components = np.array([head_score, symmetry_score, neckline_score])
            weights = np.array([head_weight, shoulder_weight, neckline_weight])

            signals[i] = calculate_weighted_score(components, weights)

        return signals

    @njit
    def detect_bull_flag_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        min_pole_length: int,
        min_flag_length: int,
        max_flag_slope: float,
        min_volume_decline: float,
        pole_weight: float,
        flag_weight: float,
        volume_weight: float,
    ) -> np.ndarray:
        """Détection Bull Flag — mât haussier + consolidation plate/légèrement baissière.

        CORRECTIONS v3 :
        C1 — min_flag_length : drapeau minimum 5 barres (était 2, non structuré).
        C2 — min_pole_atrs   : durci à 3.0 (était 2.0) — mât doit être un vrai mouvement.
        C3 — bullish_ratio   : durci à 0.60 (était 0.55) — consistance directionnelle plus stricte.
        C4 — flag_slope      : borne basse -0.05 (était -0.10), borne haute +0.01 (était +0.02).
        C5 — volume_score    : plancher supprimé (était 0.2 garantit même sans déclin de volume).
                               0.0 si volume croît dans le drapeau (signal négatif).
                               0.3 neutre si données volume absentes.
        C6 — Architecture    : on identifie d'abord le MEILLEUR mât (par pole_in_atrs),
                               puis on évalue le drapeau UNE SEULE FOIS.
                               L'ancien best_score = max() sur toutes les combinaisons
                               garantissait un hit sur ~20% des barres.
        """
        n = len(open_prices)
        signals = np.zeros(n, dtype=np.float64)
        min_pole_atrs = 3.0  # C2 : durci depuis 2.0
        atr_period = 14
        min_lookback = min_pole_length * 2 + min_flag_length + atr_period + 2

        for i in range(min_lookback, n):
            # ── Étape 1 : trouver le MEILLEUR mât haussier ─────────────────────
            # On cherche dans la fenêtre qui précède le drapeau.
            # Le drapeau commence au plus tôt à (i - min_flag_length) → le mât
            # doit finir avant cette borne.
            best_pole_atrs = 0.0
            best_bull_ratio = 0.0
            pole_start_best = -1
            pole_end_best = -1

            flag_end_bound = i - min_flag_length  # le mât doit finir avant ici

            for pole_end_c in range(
                flag_end_bound - 1,
                max(flag_end_bound - min_pole_length * 2 - 1, min_pole_length - 1),
                -1,
            ):
                for pole_start_c in range(
                    max(0, pole_end_c - min_pole_length * 2),
                    pole_end_c - min_pole_length + 1,
                ):
                    pole_length = pole_end_c - pole_start_c + 1
                    if pole_length < min_pole_length:
                        continue

                    pole_start_price = close_prices[pole_start_c]
                    pole_end_price = close_prices[pole_end_c]

                    if pole_start_price <= 0 or pole_end_price <= 0:
                        continue

                    # Le mât doit être haussier
                    if pole_end_price <= pole_start_price:
                        continue

                    pole_gain = pole_end_price - pole_start_price

                    # ATR moyen sur les barres du mât
                    _atr_sum = 0.0
                    _atr_cnt = 0
                    for _k in range(pole_start_c + 1, pole_end_c + 1):
                        if _k >= n or _k < 1:
                            continue
                        _tr = high_prices[_k] - low_prices[_k]
                        _pc = close_prices[_k - 1]
                        if _pc > 0:
                            _tr = max(
                                _tr,
                                abs(high_prices[_k] - _pc),
                                abs(low_prices[_k] - _pc),
                            )
                        if _tr > 0:
                            _atr_sum += _tr
                            _atr_cnt += 1

                    if _atr_cnt == 0:
                        continue
                    avg_atr = _atr_sum / _atr_cnt

                    # C2 : mât doit valoir >= 3 ATRs
                    pole_in_atrs = safe_divide(pole_gain, avg_atr)
                    if pole_in_atrs < min_pole_atrs:
                        continue

                    # C3 : consistance directionnelle >= 60% barres haussières
                    bullish_bars = 0
                    total_bars = 0
                    for j in range(pole_start_c, pole_end_c + 1):
                        if j < n:
                            total_bars += 1
                            if close_prices[j] >= open_prices[j]:
                                bullish_bars += 1

                    if total_bars == 0:
                        continue
                    bull_ratio = safe_divide(bullish_bars, total_bars)
                    if bull_ratio < 0.60:
                        continue

                    # Garder le mât avec le plus d'ATRs (le plus fort)
                    if pole_in_atrs > best_pole_atrs:
                        best_pole_atrs = pole_in_atrs
                        best_bull_ratio = bull_ratio
                        pole_start_best = pole_start_c
                        pole_end_best = pole_end_c

            if pole_end_best == -1:
                continue

            # ── Étape 2 : évaluer le drapeau UNE seule fois ────────────────────
            flag_start = pole_end_best + 1
            flag_end = i
            flag_length = flag_end - flag_start + 1

            # C1 : longueur minimale réelle du drapeau
            if flag_length < min_flag_length:
                continue

            flag_start_price = close_prices[flag_start]
            flag_end_price = close_prices[flag_end]

            if flag_start_price <= 0 or flag_end_price <= 0:
                continue

            # Retracement max 50% du mât
            pole_gain_best = close_prices[pole_end_best] - close_prices[pole_start_best]
            flag_lowest = np.min(low_prices[flag_start : flag_end + 1])
            if (close_prices[pole_end_best] - flag_lowest) > (pole_gain_best * 0.50):
                continue

            # C4 : pente du drapeau — plat ou légèrement baissier uniquement
            flag_slope = safe_divide(
                (flag_end_price - flag_start_price),
                flag_start_price * flag_length,
            )
            if flag_slope > 0.01:  # rejet si trop haussier
                continue
            if flag_slope < max_flag_slope:  # rejet si trop baissier (< -0.05)
                continue

            # ── Étape 3 : volume ───────────────────────────────────────────────
            # C5 : pas de plancher artificiel — volume croissant dans le flag = 0.0
            volume_score = 0.3  # valeur neutre par défaut (données volume absentes)
            if len(volume) > i:
                pole_vol_sum = 0.0
                pole_vol_cnt = 0
                flag_vol_sum = 0.0
                flag_vol_cnt = 0

                for j in range(pole_start_best, pole_end_best + 1):
                    if j < len(volume) and volume[j] > 0:
                        pole_vol_sum += volume[j]
                        pole_vol_cnt += 1

                for j in range(flag_start, flag_end + 1):
                    if j < len(volume) and volume[j] > 0:
                        flag_vol_sum += volume[j]
                        flag_vol_cnt += 1

                if pole_vol_cnt > 0 and flag_vol_cnt > 0:
                    pole_avg = safe_divide(pole_vol_sum, pole_vol_cnt)
                    flag_avg = safe_divide(flag_vol_sum, flag_vol_cnt)
                    if pole_avg > 0:
                        vol_decline = safe_divide((pole_avg - flag_avg), pole_avg)
                        if vol_decline >= min_volume_decline:
                            # Bon déclin de volume → score plein proportionnel
                            volume_score = min(
                                1.0,
                                safe_divide(vol_decline, min_volume_decline * 1.5),
                            )
                        elif vol_decline > 0:
                            # Déclin partiel → score proportionnel sans plancher
                            volume_score = safe_divide(vol_decline, min_volume_decline)
                        else:
                            # Volume stable ou croissant dans le flag → signal négatif
                            volume_score = 0.0

            # ── Étape 4 : score final ──────────────────────────────────────────
            pole_score = min(1.0, safe_divide(best_pole_atrs, min_pole_atrs * 2.0))
            pole_score *= min(1.0, best_bull_ratio * 1.5)

            _abs_slope = flag_slope if flag_slope >= 0 else -flag_slope
            _ref_slope = abs(max_flag_slope) if max_flag_slope != 0 else 0.05
            flag_score = 1.0 - min(1.0, safe_divide(_abs_slope, _ref_slope))

            components = np.array([pole_score, flag_score, volume_score])
            weights = np.array([pole_weight, flag_weight, volume_weight])

            signals[i] = calculate_weighted_score(components, weights)

        return signals

    @njit
    def detect_channel_up_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        min_length: int,
        min_slope: float,
        max_width_variation: float,
        slope_weight: float,
        parallel_weight: float,
        duration_weight: float,
    ) -> np.ndarray:
        """VERSION CORRIGÉE - Détection du pattern Channel Up.

        CORRECTIONS APPORTÉES:
        1. Seuils de pente plus réalistes (0.005 au lieu de 0.02)
        2. Variation de largeur assouplie (0.50 au lieu de 0.20)
        3. Validation avec safe_divide partout
        4. Système de scoring pondéré
        5. Gestion des erreurs silencieuses

        PATTERN:
        - Canal haussier avec 2 lignes parallèles ascendantes
        - Ligne de support (creux) et résistance (pics) quasi-parallèles
        - Pente positive minimum
        - Largeur du canal relativement stable

        Args:
            Prix OHLC et volume
            min_length: Longueur minimum du canal (15 recommandé)
            min_slope: Pente minimum (0.005 recommandé)
            max_width_variation: Variation maximum de largeur (0.50 recommandé)
            Poids pour le scoring pondéré
            close_prices: TODO: documenter.
            duration_weight: TODO: documenter.
            high_prices: TODO: documenter.
            low_prices: TODO: documenter.
            open_prices: TODO: documenter.
            parallel_weight: TODO: documenter.
            slope_weight: TODO: documenter.
            volume: TODO: documenter.

        Returns:
            Array des scores de détection (0.0 à 1.0)
        """
        signals = np.zeros(len(open_prices), dtype=np.float64)

        for i in range(min_length, len(open_prices)):
            start_idx = i - min_length

            # Trouver les pivots de support (creux) et résistance (pics)
            high_pivots_idx, high_pivots_price, high_pivots_type = find_pivots_simple(
                high_prices[start_idx : i + 1], 0.003, 3
            )
            low_pivots_idx, low_pivots_price, low_pivots_type = find_pivots_simple(
                low_prices[start_idx : i + 1], 0.003, 3
            )

            # Extraire les pivots hauts et bas (Numba doesn't support list of tuples)
            highs_idx = np.empty(len(high_pivots_type), dtype=np.int64)
            highs_price = np.empty(len(high_pivots_type), dtype=np.float64)
            lows_idx = np.empty(len(low_pivots_type), dtype=np.int64)
            lows_price = np.empty(len(low_pivots_type), dtype=np.float64)

            h_count = 0
            for j in range(len(high_pivots_type)):
                if high_pivots_type[j] == 1:
                    highs_idx[h_count] = high_pivots_idx[j] + start_idx
                    highs_price[h_count] = high_pivots_price[j]
                    h_count += 1

            l_count = 0
            for j in range(len(low_pivots_type)):
                if low_pivots_type[j] == -1:
                    lows_idx[l_count] = low_pivots_idx[j] + start_idx
                    lows_price[l_count] = low_pivots_price[j]
                    l_count += 1

            # Besoin d'au moins 2 points hauts et 2 points bas
            if h_count < 2 or l_count < 2:
                continue

            # Prendre les 2 derniers de chaque type
            high1_idx, high1_price = highs_idx[h_count - 2], highs_price[h_count - 2]
            high2_idx, high2_price = highs_idx[h_count - 1], highs_price[h_count - 1]
            low1_idx, low1_price = lows_idx[l_count - 2], lows_price[l_count - 2]
            low2_idx, low2_price = lows_idx[l_count - 1], lows_price[l_count - 1]

            # === VALIDATION DES DONNÉES ===
            if not (
                start_idx <= min(high1_idx, low1_idx) and max(high2_idx, low2_idx) <= i
            ):
                continue

            if (
                high1_price <= 0
                or high2_price <= 0
                or low1_price <= 0
                or low2_price <= 0
            ):
                continue

            # === CALCUL DES PENTES ===

            # Pente de la ligne de résistance (pics)
            resistance_slope = safe_divide(
                (high2_price - high1_price), (high2_idx - high1_idx)
            )

            # Pente de la ligne de support (creux)
            support_slope = safe_divide(
                (low2_price - low1_price), (low2_idx - low1_idx)
            )

            # === CONDITIONS DU PATTERN (ASSOUPLIES) ===

            # 1. Pentes positives minimum (condition assouplie)
            if resistance_slope < min_slope or support_slope < min_slope:
                continue

            # 2. Les lignes doivent être approximativement parallèles (condition assouplie)
            slope_diff = abs(resistance_slope - support_slope)
            avg_slope = safe_divide((resistance_slope + support_slope), 2.0)
            # AUDIT FIX C-B5-3: initialiser slope_diff_ratio=0.0 pour éviter unbound variable
            slope_diff_ratio = 0.0
            if avg_slope > 0:
                slope_diff_ratio = safe_divide(slope_diff, avg_slope)
                if slope_diff_ratio > max_width_variation:
                    continue

            # 3. Largeur du canal relativement stable
            width1 = high1_price - low1_price
            width2 = high2_price - low2_price
            if width1 > 0 and width2 > 0:
                width_ratio = safe_divide(max(width1, width2), min(width1, width2))
                if width_ratio > (1.0 + max_width_variation):
                    continue

            # === CALCUL DU SCORE PONDÉRÉ ===

            # Score de pente (plus la pente est forte, mieux c'est, mais pas trop)
            slope_score = min(1.0, safe_divide(avg_slope, min_slope * 2))

            # Score de parallélisme (plus les pentes sont proches, mieux c'est)
            if avg_slope > 0:
                parallel_score = 1.0 - min(
                    1.0, safe_divide(slope_diff_ratio, max_width_variation)
                )
            else:
                parallel_score = 0.0

            # Score de durée (plus le canal est long, mieux c'est)
            channel_length = max(high2_idx, low2_idx) - min(high1_idx, low1_idx)
            duration_score = min(1.0, safe_divide(channel_length, min_length * 2))

            # Combinaison pondérée
            components = np.array([slope_score, parallel_score, duration_score])
            weights = np.array([slope_weight, parallel_weight, duration_weight])

            final_score = calculate_weighted_score(components, weights)
            signals[i] = final_score

        return signals

    @njit
    def detect_channel_down_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        min_length: int,
        min_slope: float,  # Sera négatif pour canal descendant
        max_width_variation: float,
        slope_weight: float,
        parallel_weight: float,
        duration_weight: float,
    ) -> np.ndarray:
        """VERSION CORRIGÉE - Détection du pattern Channel Down.

        CORRECTIONS APPORTÉES:
        1. Seuils de pente plus réalistes (-0.005 au lieu de -0.02)
        2. Variation de largeur assouplie (0.50 au lieu de 0.20)
        3. Validation avec safe_divide partout
        4. Système de scoring pondéré
        5. Gestion des erreurs silencieuses

        PATTERN:
        - Canal baissier avec 2 lignes parallèles descendantes
        - Ligne de support (creux) et résistance (pics) quasi-parallèles
        - Pente négative minimum
        - Largeur du canal relativement stable

        Args:
            Prix OHLC et volume
            min_length: Longueur minimum du canal (15 recommandé)
            min_slope: Pente minimum négative (-0.005 recommandé)
            max_width_variation: Variation maximum de largeur (0.50 recommandé)
            Poids pour le scoring pondéré
            close_prices: TODO: documenter.
            duration_weight: TODO: documenter.
            high_prices: TODO: documenter.
            low_prices: TODO: documenter.
            open_prices: TODO: documenter.
            parallel_weight: TODO: documenter.
            slope_weight: TODO: documenter.
            volume: TODO: documenter.

        Returns:
            Array des scores de détection (0.0 à 1.0)
        """
        signals = np.zeros(len(open_prices), dtype=np.float64)

        for i in range(min_length, len(open_prices)):
            start_idx = i - min_length

            # Trouver les pivots de support (creux) et résistance (pics)
            high_pivots_idx, high_pivots_price, high_pivots_type = find_pivots_simple(
                high_prices[start_idx : i + 1], 0.003, 3
            )
            low_pivots_idx, low_pivots_price, low_pivots_type = find_pivots_simple(
                low_prices[start_idx : i + 1], 0.003, 3
            )

            # Extraire les pivots hauts et bas (Numba doesn't support list of tuples)
            highs_idx = np.empty(len(high_pivots_type), dtype=np.int64)
            highs_price = np.empty(len(high_pivots_type), dtype=np.float64)
            lows_idx = np.empty(len(low_pivots_type), dtype=np.int64)
            lows_price = np.empty(len(low_pivots_type), dtype=np.float64)

            h_count = 0
            for j in range(len(high_pivots_type)):
                if high_pivots_type[j] == 1:
                    highs_idx[h_count] = high_pivots_idx[j] + start_idx
                    highs_price[h_count] = high_pivots_price[j]
                    h_count += 1

            l_count = 0
            for j in range(len(low_pivots_type)):
                if low_pivots_type[j] == -1:
                    lows_idx[l_count] = low_pivots_idx[j] + start_idx
                    lows_price[l_count] = low_pivots_price[j]
                    l_count += 1

            # Besoin d'au moins 2 points hauts et 2 points bas
            if h_count < 2 or l_count < 2:
                continue

            # Prendre les 2 derniers de chaque type
            high1_idx, high1_price = highs_idx[h_count - 2], highs_price[h_count - 2]
            high2_idx, high2_price = highs_idx[h_count - 1], highs_price[h_count - 1]
            low1_idx, low1_price = lows_idx[l_count - 2], lows_price[l_count - 2]
            low2_idx, low2_price = lows_idx[l_count - 1], lows_price[l_count - 1]

            # === VALIDATION DES DONNÉES ===
            if not (
                start_idx <= min(high1_idx, low1_idx) and max(high2_idx, low2_idx) <= i
            ):
                continue

            if (
                high1_price <= 0
                or high2_price <= 0
                or low1_price <= 0
                or low2_price <= 0
            ):
                continue

            # === CALCUL DES PENTES ===

            # Pente de la ligne de résistance (pics)
            resistance_slope = safe_divide(
                (high2_price - high1_price), (high2_idx - high1_idx)
            )

            # Pente de la ligne de support (creux)
            support_slope = safe_divide(
                (low2_price - low1_price), (low2_idx - low1_idx)
            )

            # === CONDITIONS DU PATTERN (ASSOUPLIES) ===

            # 1. Pentes négatives minimum (condition assouplie)
            if (
                resistance_slope > min_slope or support_slope > min_slope
            ):  # min_slope est négatif
                continue

            # 2. Les lignes doivent être approximativement parallèles (condition assouplie)
            slope_diff = abs(resistance_slope - support_slope)
            avg_slope = safe_divide((abs(resistance_slope) + abs(support_slope)), 2.0)
            # AUDIT FIX C-B5-4: initialiser slope_diff_ratio=0.0 pour éviter unbound variable
            slope_diff_ratio = 0.0
            if avg_slope > 0:
                slope_diff_ratio = safe_divide(slope_diff, avg_slope)
                if slope_diff_ratio > max_width_variation:
                    continue

            # 3. Largeur du canal relativement stable
            width1 = high1_price - low1_price
            width2 = high2_price - low2_price
            if width1 > 0 and width2 > 0:
                width_ratio = safe_divide(max(width1, width2), min(width1, width2))
                if width_ratio > (1.0 + max_width_variation):
                    continue

            # === CALCUL DU SCORE PONDÉRÉ ===

            # Score de pente (plus la pente est forte en négatif, mieux c'est)
            slope_score = min(1.0, safe_divide(avg_slope, abs(min_slope) * 2))

            # Score de parallélisme (plus les pentes sont proches, mieux c'est)
            if avg_slope > 0:
                parallel_score = 1.0 - min(
                    1.0, safe_divide(slope_diff_ratio, max_width_variation)
                )
            else:
                parallel_score = 0.0

            # Score de durée (plus le canal est long, mieux c'est)
            channel_length = max(high2_idx, low2_idx) - min(high1_idx, low1_idx)
            duration_score = min(1.0, safe_divide(channel_length, min_length * 2))

            # Combinaison pondérée
            components = np.array([slope_score, parallel_score, duration_score])
            weights = np.array([slope_weight, parallel_weight, duration_weight])

            final_score = calculate_weighted_score(components, weights)
            signals[i] = final_score

        return signals

    @njit
    def detect_gartley_bull_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        xa_ratio: float,
        ab_ratio: float,
        bc_ratio: float,
        cd_ratio: float,
        tolerance: float,
        xa_weight: float,
        ab_weight: float,
        bc_weight: float,
        cd_weight: float,
    ) -> np.ndarray:
        """Détection du pattern Gartley Bull.

        FIX: cd_xa_ratio désormais utilisé dans xa_score (cible 0.786 selon Carney).
        Le score xa_score n'est plus binaire (1.0 si xa>0) mais mesure la proximité
        du ratio CD/XA avec 0.786, qui est la contrainte distinctive du Gartley.
        """
        signals = np.zeros(len(open_prices), dtype=np.float64)
        min_pattern_length = 20

        for i in range(min_pattern_length * 2, len(open_prices)):
            start_idx = i - min_pattern_length * 2

            pivots_idx, pivots_price, pivots_type = find_pivots_simple(
                close_prices[start_idx : i + 1], 0.003, 3
            )

            if len(pivots_idx) < 5:
                continue

            for start_pattern in range(
                max(0, len(pivots_idx) - 8), len(pivots_idx) - 4
            ):
                if start_pattern + 4 >= len(pivots_idx):
                    continue

                x_idx = pivots_idx[start_pattern] + start_idx
                a_idx = pivots_idx[start_pattern + 1] + start_idx
                b_idx = pivots_idx[start_pattern + 2] + start_idx
                c_idx = pivots_idx[start_pattern + 3] + start_idx
                d_idx = pivots_idx[start_pattern + 4] + start_idx

                x_price = pivots_price[start_pattern]
                a_price = pivots_price[start_pattern + 1]
                b_price = pivots_price[start_pattern + 2]
                c_price = pivots_price[start_pattern + 3]
                d_price = pivots_price[start_pattern + 4]

                x_type = pivots_type[start_pattern]
                a_type = pivots_type[start_pattern + 1]
                b_type = pivots_type[start_pattern + 2]
                c_type = pivots_type[start_pattern + 3]
                d_type = pivots_type[start_pattern + 4]

                if not (x_idx < a_idx < b_idx < c_idx < d_idx <= i):
                    continue
                if (
                    x_price <= 0
                    or a_price <= 0
                    or b_price <= 0
                    or c_price <= 0
                    or d_price <= 0
                ):
                    continue
                if d_idx != i:
                    continue

                # X(high) -> A(low) -> B(high) -> C(low) -> D(high)
                if not (
                    x_type == 1
                    and a_type == -1
                    and b_type == 1
                    and c_type == -1
                    and d_type == 1
                ):
                    continue

                xa_move = abs(x_price - a_price)
                if xa_move <= 0:
                    continue

                ab_move = abs(b_price - a_price)
                ab_calculated_ratio = safe_divide(ab_move, xa_move)

                bc_move = abs(c_price - b_price)
                bc_calculated_ratio = safe_divide(bc_move, ab_move)

                cd_move = abs(d_price - c_price)
                cd_calculated_ratio = safe_divide(cd_move, bc_move)

                # FIX: cd_xa_ratio utilisé dans xa_score
                # Gartley: D complète à ~78.6% du mouvement XA (Carney, Harmonic Trading)
                cd_xa_ratio = safe_divide(cd_move, xa_move)
                cd_xa_target = 0.786
                cd_xa_error = abs(cd_xa_ratio - cd_xa_target)
                xa_score = max(0.0, 1.0 - safe_divide(cd_xa_error, tolerance))

                ab_error = abs(ab_calculated_ratio - ab_ratio)
                ab_score = max(0.0, 1.0 - safe_divide(ab_error, tolerance))

                bc_error = abs(bc_calculated_ratio - bc_ratio)
                bc_score = max(0.0, 1.0 - safe_divide(bc_error, tolerance))

                cd_error = abs(cd_calculated_ratio - cd_ratio)
                cd_score = max(0.0, 1.0 - safe_divide(cd_error, tolerance))

                min_acceptable_score = 0.3
                if (
                    xa_score < min_acceptable_score
                    or ab_score < min_acceptable_score
                    or bc_score < min_acceptable_score
                    or cd_score < min_acceptable_score
                ):
                    continue

                components = np.array([xa_score, ab_score, bc_score, cd_score])
                weights = np.array([xa_weight, ab_weight, bc_weight, cd_weight])
                pattern_score = calculate_weighted_score(components, weights)
                signals[i] = max(signals[i], pattern_score)

        return signals

    @njit
    def detect_gartley_bear_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        xa_ratio: float,
        ab_ratio: float,
        bc_ratio: float,
        cd_ratio: float,
        tolerance: float,
        xa_weight: float,
        ab_weight: float,
        bc_weight: float,
        cd_weight: float,
    ) -> np.ndarray:
        """Détection du pattern Gartley Bear.

        FIX: cd_xa_ratio utilisé dans xa_score (cible 0.786).
        """
        signals = np.zeros(len(open_prices), dtype=np.float64)
        min_pattern_length = 20

        for i in range(min_pattern_length * 2, len(open_prices)):
            start_idx = i - min_pattern_length * 2

            pivots_idx, pivots_price, pivots_type = find_pivots_simple(
                close_prices[start_idx : i + 1], 0.003, 3
            )

            if len(pivots_idx) < 5:
                continue

            for start_pattern in range(
                max(0, len(pivots_idx) - 8), len(pivots_idx) - 4
            ):
                if start_pattern + 4 >= len(pivots_idx):
                    continue

                x_idx = pivots_idx[start_pattern] + start_idx
                a_idx = pivots_idx[start_pattern + 1] + start_idx
                b_idx = pivots_idx[start_pattern + 2] + start_idx
                c_idx = pivots_idx[start_pattern + 3] + start_idx
                d_idx = pivots_idx[start_pattern + 4] + start_idx

                x_price = pivots_price[start_pattern]
                a_price = pivots_price[start_pattern + 1]
                b_price = pivots_price[start_pattern + 2]
                c_price = pivots_price[start_pattern + 3]
                d_price = pivots_price[start_pattern + 4]

                x_type = pivots_type[start_pattern]
                a_type = pivots_type[start_pattern + 1]
                b_type = pivots_type[start_pattern + 2]
                c_type = pivots_type[start_pattern + 3]
                d_type = pivots_type[start_pattern + 4]

                if not (x_idx < a_idx < b_idx < c_idx < d_idx <= i):
                    continue
                if (
                    x_price <= 0
                    or a_price <= 0
                    or b_price <= 0
                    or c_price <= 0
                    or d_price <= 0
                ):
                    continue
                if d_idx != i:
                    continue

                # X(low) -> A(high) -> B(low) -> C(high) -> D(low)
                if not (
                    x_type == -1
                    and a_type == 1
                    and b_type == -1
                    and c_type == 1
                    and d_type == -1
                ):
                    continue

                xa_move = abs(a_price - x_price)
                if xa_move <= 0:
                    continue

                ab_move = abs(a_price - b_price)
                ab_calculated_ratio = safe_divide(ab_move, xa_move)

                bc_move = abs(c_price - b_price)
                bc_calculated_ratio = safe_divide(bc_move, ab_move)

                cd_move = abs(b_price - d_price)
                cd_calculated_ratio = safe_divide(cd_move, bc_move)

                # FIX: cd_xa_ratio utilisé dans xa_score (cible 0.786)
                cd_xa_ratio = safe_divide(cd_move, xa_move)
                cd_xa_error = abs(cd_xa_ratio - 0.786)
                xa_score = max(0.0, 1.0 - safe_divide(cd_xa_error, tolerance))

                ab_error = abs(ab_calculated_ratio - ab_ratio)
                ab_score = max(0.0, 1.0 - safe_divide(ab_error, tolerance))

                bc_error = abs(bc_calculated_ratio - bc_ratio)
                bc_score = max(0.0, 1.0 - safe_divide(bc_error, tolerance))

                cd_error = abs(cd_calculated_ratio - cd_ratio)
                cd_score = max(0.0, 1.0 - safe_divide(cd_error, tolerance))

                min_acceptable_score = 0.3
                if (
                    xa_score < min_acceptable_score
                    or ab_score < min_acceptable_score
                    or bc_score < min_acceptable_score
                    or cd_score < min_acceptable_score
                ):
                    continue

                components = np.array([xa_score, ab_score, bc_score, cd_score])
                weights = np.array([xa_weight, ab_weight, bc_weight, cd_weight])
                pattern_score = calculate_weighted_score(components, weights)
                signals[i] = max(signals[i], pattern_score)

        return signals

    @njit
    def detect_butterfly_bull_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        xa_ratio: float,
        ab_ratio: float,
        bc_ratio: float,
        cd_ratio: float,
        tolerance: float,
        xa_weight: float,
        ab_weight: float,
        bc_weight: float,
        cd_weight: float,
    ) -> np.ndarray:
        """Détection du pattern Butterfly Bull.

        FIX: cd_xa_ratio utilisé dans xa_score.
        Butterfly: D s'étend AU-DELÀ de X → cd_xa_ratio cible = 1.272 (Carney).
        C'est ce qui distingue le Butterfly du Gartley.
        """
        signals = np.zeros(len(open_prices), dtype=np.float64)
        min_pattern_length = 20

        for i in range(min_pattern_length * 2, len(open_prices)):
            start_idx = i - min_pattern_length * 2

            pivots_idx, pivots_price, pivots_type = find_pivots_simple(
                close_prices[start_idx : i + 1], 0.003, 3
            )

            if len(pivots_idx) < 5:
                continue

            for start_pattern in range(
                max(0, len(pivots_idx) - 8), len(pivots_idx) - 4
            ):
                if start_pattern + 4 >= len(pivots_idx):
                    continue

                x_idx = pivots_idx[start_pattern] + start_idx
                a_idx = pivots_idx[start_pattern + 1] + start_idx
                b_idx = pivots_idx[start_pattern + 2] + start_idx
                c_idx = pivots_idx[start_pattern + 3] + start_idx
                d_idx = pivots_idx[start_pattern + 4] + start_idx

                x_price = pivots_price[start_pattern]
                a_price = pivots_price[start_pattern + 1]
                b_price = pivots_price[start_pattern + 2]
                c_price = pivots_price[start_pattern + 3]
                d_price = pivots_price[start_pattern + 4]

                x_type = pivots_type[start_pattern]
                a_type = pivots_type[start_pattern + 1]
                b_type = pivots_type[start_pattern + 2]
                c_type = pivots_type[start_pattern + 3]
                d_type = pivots_type[start_pattern + 4]

                if not (x_idx < a_idx < b_idx < c_idx < d_idx <= i):
                    continue
                if (
                    x_price <= 0
                    or a_price <= 0
                    or b_price <= 0
                    or c_price <= 0
                    or d_price <= 0
                ):
                    continue
                if d_idx != i:
                    continue

                # X(high) -> A(low) -> B(high) -> C(low) -> D(high)
                if not (
                    x_type == 1
                    and a_type == -1
                    and b_type == 1
                    and c_type == -1
                    and d_type == 1
                ):
                    continue

                xa_move = abs(x_price - a_price)
                if xa_move <= 0:
                    continue

                ab_move = abs(b_price - a_price)
                ab_calculated_ratio = safe_divide(ab_move, xa_move)

                bc_move = abs(c_price - b_price)
                bc_calculated_ratio = safe_divide(bc_move, ab_move)

                cd_move = abs(d_price - c_price)
                cd_calculated_ratio = safe_divide(cd_move, bc_move)

                # FIX: cd_xa_ratio utilisé dans xa_score
                # Butterfly: D dépasse X, cd_xa_ratio cible = 1.272
                cd_xa_ratio = safe_divide(cd_move, xa_move)
                cd_xa_error = abs(cd_xa_ratio - 1.272)
                xa_score = max(0.0, 1.0 - safe_divide(cd_xa_error, tolerance))

                ab_error = abs(
                    ab_calculated_ratio - xa_ratio
                )  # xa_ratio = 0.786 pour butterfly
                ab_score = max(0.0, 1.0 - safe_divide(ab_error, tolerance))

                bc_error = abs(bc_calculated_ratio - bc_ratio)
                bc_score = max(0.0, 1.0 - safe_divide(bc_error, tolerance))

                cd_error = abs(cd_calculated_ratio - cd_ratio)
                cd_score = max(0.0, 1.0 - safe_divide(cd_error, tolerance))

                min_acceptable_score = 0.3
                if (
                    xa_score < min_acceptable_score
                    or ab_score < min_acceptable_score
                    or bc_score < min_acceptable_score
                    or cd_score < min_acceptable_score
                ):
                    continue

                components = np.array([xa_score, ab_score, bc_score, cd_score])
                weights = np.array([xa_weight, ab_weight, bc_weight, cd_weight])
                pattern_score = calculate_weighted_score(components, weights)
                signals[i] = max(signals[i], pattern_score)

        return signals

    @njit
    def detect_butterfly_bear_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        xa_ratio: float,
        ab_ratio: float,
        bc_ratio: float,
        cd_ratio: float,
        tolerance: float,
        xa_weight: float,
        ab_weight: float,
        bc_weight: float,
        cd_weight: float,
    ) -> np.ndarray:
        """Détection du pattern Butterfly Bear.

        FIX: cd_xa_ratio utilisé dans xa_score (cible 1.272).
        """
        signals = np.zeros(len(open_prices), dtype=np.float64)
        min_pattern_length = 20

        for i in range(min_pattern_length * 2, len(open_prices)):
            start_idx = i - min_pattern_length * 2

            pivots_idx, pivots_price, pivots_type = find_pivots_simple(
                close_prices[start_idx : i + 1], 0.003, 3
            )

            if len(pivots_idx) < 5:
                continue

            for start_pattern in range(
                max(0, len(pivots_idx) - 8), len(pivots_idx) - 4
            ):
                if start_pattern + 4 >= len(pivots_idx):
                    continue

                x_idx = pivots_idx[start_pattern] + start_idx
                a_idx = pivots_idx[start_pattern + 1] + start_idx
                b_idx = pivots_idx[start_pattern + 2] + start_idx
                c_idx = pivots_idx[start_pattern + 3] + start_idx
                d_idx = pivots_idx[start_pattern + 4] + start_idx

                x_price = pivots_price[start_pattern]
                a_price = pivots_price[start_pattern + 1]
                b_price = pivots_price[start_pattern + 2]
                c_price = pivots_price[start_pattern + 3]
                d_price = pivots_price[start_pattern + 4]

                x_type = pivots_type[start_pattern]
                a_type = pivots_type[start_pattern + 1]
                b_type = pivots_type[start_pattern + 2]
                c_type = pivots_type[start_pattern + 3]
                d_type = pivots_type[start_pattern + 4]

                if not (x_idx < a_idx < b_idx < c_idx < d_idx <= i):
                    continue
                if (
                    x_price <= 0
                    or a_price <= 0
                    or b_price <= 0
                    or c_price <= 0
                    or d_price <= 0
                ):
                    continue
                if d_idx != i:
                    continue

                # X(low) -> A(high) -> B(low) -> C(high) -> D(low)
                if not (
                    x_type == -1
                    and a_type == 1
                    and b_type == -1
                    and c_type == 1
                    and d_type == -1
                ):
                    continue

                xa_move = abs(a_price - x_price)
                if xa_move <= 0:
                    continue

                ab_move = abs(a_price - b_price)
                ab_calculated_ratio = safe_divide(ab_move, xa_move)

                bc_move = abs(c_price - b_price)
                bc_calculated_ratio = safe_divide(bc_move, ab_move)

                cd_move = abs(b_price - d_price)
                cd_calculated_ratio = safe_divide(cd_move, bc_move)

                # FIX: cd_xa_ratio utilisé dans xa_score (cible 1.272)
                cd_xa_ratio = safe_divide(cd_move, xa_move)
                cd_xa_error = abs(cd_xa_ratio - 1.272)
                xa_score = max(0.0, 1.0 - safe_divide(cd_xa_error, tolerance))

                ab_error = abs(
                    ab_calculated_ratio - xa_ratio
                )  # xa_ratio = 0.786 pour butterfly
                ab_score = max(0.0, 1.0 - safe_divide(ab_error, tolerance))

                bc_error = abs(bc_calculated_ratio - bc_ratio)
                bc_score = max(0.0, 1.0 - safe_divide(bc_error, tolerance))

                cd_error = abs(cd_calculated_ratio - cd_ratio)
                cd_score = max(0.0, 1.0 - safe_divide(cd_error, tolerance))

                min_acceptable_score = 0.3
                if (
                    xa_score < min_acceptable_score
                    or ab_score < min_acceptable_score
                    or bc_score < min_acceptable_score
                    or cd_score < min_acceptable_score
                ):
                    continue

                components = np.array([xa_score, ab_score, bc_score, cd_score])
                weights = np.array([xa_weight, ab_weight, bc_weight, cd_weight])
                pattern_score = calculate_weighted_score(components, weights)
                signals[i] = max(signals[i], pattern_score)

        return signals

    @njit
    def detect_bat_bull_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        xa_ratio: float,
        ab_ratio: float,
        bc_ratio: float,
        cd_ratio: float,
        tolerance: float,
        xa_weight: float,
        ab_weight: float,
        bc_weight: float,
        cd_weight: float,
    ) -> np.ndarray:
        """VERSION CORRIGÉE - Détection du pattern Bat Bull.

        CORRECTIONS APPORTÉES:
        1. Gestion des erreurs silencieuses avec validation stricte
        2. Tolérance augmentée à 0.15 pour marchés volatils
        3. Système de scoring pondéré au lieu de conditions binaires
        4. Logging des étapes de validation intégré
        5. Recherche de pivots robuste avec find_pivots_simple()

        PATTERN BAT HAUSSIER:
        - Point X: Pic initial
        - Point A: Creux (retracement de X)
        - Point B: Pic (rebond de A, ~61.8% du mouvement XA)
        - Point C: Creux (retracement de B, ~88.6% du mouvement AB)
        - Point D: Pic de completion (extension ~261.8% du mouvement BC)

        Ratios cibles: XA=base, AB=0.382*XA, BC=0.886*AB, CD=2.618*BC

        Args:
            Prix OHLC et volume
            xa_ratio, ab_ratio, bc_ratio, cd_ratio: Ratios de Fibonacci cibles
            tolerance: Tolérance pour les ratios (0.15 recommandé)
            Poids pour le scoring pondéré
            ab_ratio: TODO: documenter.
            ab_weight: TODO: documenter.
            bc_ratio: TODO: documenter.
            bc_weight: TODO: documenter.
            cd_ratio: TODO: documenter.
            cd_weight: TODO: documenter.
            close_prices: TODO: documenter.
            high_prices: TODO: documenter.
            low_prices: TODO: documenter.
            open_prices: TODO: documenter.
            volume: TODO: documenter.
            xa_ratio: TODO: documenter.
            xa_weight: TODO: documenter.

        Returns:
            Array des scores de détection (0.0 à 1.0)
        """
        signals = np.zeros(len(open_prices), dtype=np.float64)
        min_pattern_length = 20
        # AUDIT FIX C-B6-1: supprimé patterns_found (variable morte)

        for i in range(min_pattern_length * 2, len(open_prices)):
            start_idx = i - min_pattern_length * 2

            # Trouver tous les pivots dans la fenêtre
            pivots_idx, pivots_price, pivots_type = find_pivots_simple(
                close_prices[start_idx : i + 1], 0.003, 3
            )

            if len(pivots_idx) < 5:  # Besoin d'au moins 5 points XABCD
                continue

            # Chercher les patterns XABCD potentiels dans les derniers pivots
            for start_pattern in range(
                max(0, len(pivots_idx) - 8), len(pivots_idx) - 4
            ):
                if start_pattern + 4 >= len(pivots_idx):
                    continue

                # Extraire 5 pivots consécutifs (X-A-B-C-D)
                x_idx = pivots_idx[start_pattern] + start_idx
                a_idx = pivots_idx[start_pattern + 1] + start_idx
                b_idx = pivots_idx[start_pattern + 2] + start_idx
                c_idx = pivots_idx[start_pattern + 3] + start_idx
                d_idx = pivots_idx[start_pattern + 4] + start_idx

                x_price = pivots_price[start_pattern]
                a_price = pivots_price[start_pattern + 1]
                b_price = pivots_price[start_pattern + 2]
                c_price = pivots_price[start_pattern + 3]
                d_price = pivots_price[start_pattern + 4]

                x_type = pivots_type[start_pattern]
                a_type = pivots_type[start_pattern + 1]
                b_type = pivots_type[start_pattern + 2]
                c_type = pivots_type[start_pattern + 3]
                d_type = pivots_type[start_pattern + 4]

                # === VALIDATION DES DONNÉES STRICTE ===
                if not (x_idx < a_idx < b_idx < c_idx < d_idx <= i):
                    continue

                if (
                    x_price <= 0
                    or a_price <= 0
                    or b_price <= 0
                    or c_price <= 0
                    or d_price <= 0
                ):
                    continue

                # Correction: Pattern must complete exactly today
                if d_idx != i:
                    continue

                # === VÉRIFICATION DE LA SÉQUENCE BAT HAUSSIER ===
                # X(high) -> A(low) -> B(high) -> C(low) -> D(high)
                if not (
                    x_type == 1
                    and a_type == -1
                    and b_type == 1
                    and c_type == -1
                    and d_type == 1
                ):
                    continue

                # === CALCUL DES RATIOS DE FIBONACCI ===

                # Mouvement de base XA
                xa_move = abs(x_price - a_price)
                if xa_move <= 0:
                    continue

                # Ratio AB/XA (doit être ~0.382)
                ab_move = abs(b_price - a_price)
                ab_calculated_ratio = safe_divide(ab_move, xa_move)

                # Ratio BC/AB (doit être ~0.886)
                bc_move = abs(c_price - b_price)
                bc_calculated_ratio = safe_divide(bc_move, ab_move)

                # Ratio CD/BC (doit être ~2.618)
                cd_move = abs(d_price - c_price)
                cd_calculated_ratio = safe_divide(cd_move, bc_move)

                # === ÉVALUATION DES RATIOS (SCORING GRADUEL) ===

                # Score AB (proximité avec 0.382)
                ab_error = abs(
                    ab_calculated_ratio - xa_ratio
                )  # xa_ratio = 0.382 pour bat
                ab_score = max(0.0, 1.0 - safe_divide(ab_error, tolerance))

                # Score BC (proximité avec 0.886)
                bc_error = abs(bc_calculated_ratio - bc_ratio)
                bc_score = max(0.0, 1.0 - safe_divide(bc_error, tolerance))

                # Score CD (proximité avec 2.618)
                cd_error = abs(cd_calculated_ratio - cd_ratio)
                cd_score = max(0.0, 1.0 - safe_divide(cd_error, tolerance))

                # AUDIT FIX C-B6-1: xa_score graduel basé sur cd_xa_ratio (cible Bat=0.886)
                # Un Bat se distingue du Gartley/Butterfly par sa zone D à 88.6% du mouvement XA
                cd_xa_ratio = safe_divide(cd_move, xa_move)
                cd_xa_error = abs(cd_xa_ratio - 0.886)
                xa_score = max(0.0, 1.0 - safe_divide(cd_xa_error, tolerance))

                # Seuil minimum (légèrement relevé)
                min_acceptable_score = 0.30
                if (
                    ab_score < min_acceptable_score
                    or bc_score < min_acceptable_score
                    or cd_score < min_acceptable_score
                ):
                    continue

                # === CALCUL DU SCORE PONDÉRÉ FINAL ===

                components = np.array([xa_score, ab_score, bc_score, cd_score])
                weights = np.array([xa_weight, ab_weight, bc_weight, cd_weight])

                pattern_score = calculate_weighted_score(components, weights)
                signals[i] = max(signals[i], pattern_score)

        return signals

    @njit
    def detect_bat_bear_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        xa_ratio: float,
        ab_ratio: float,
        bc_ratio: float,
        cd_ratio: float,
        tolerance: float,
        xa_weight: float,
        ab_weight: float,
        bc_weight: float,
        cd_weight: float,
    ) -> np.ndarray:
        """VERSION CORRIGÉE - Détection du pattern Bat Bear.

        CORRECTIONS APPORTÉES:
        1. Conditions réduites - scoring pondéré au lieu de conditions binaires strictes
        2. Gestion des erreurs mathématiques avec safe_divide partout
        3. Gestion des erreurs silencieuses avec validation stricte
        4. Tolérance augmentée à 0.15 pour marchés volatils
        5. Recherche de pivots robuste

        PATTERN BAT BAISSIER:
        - Point X: Creux initial
        - Point A: Pic (rebond de X)
        - Point B: Creux (retracement de A, ~38.2% du mouvement XA)
        - Point C: Pic (rebond de B, ~88.6% du mouvement AB)
        - Point D: Creux de completion (extension ~261.8% du mouvement BC)

        Ratios cibles: XA=base, AB=0.382*XA, BC=0.886*AB, CD=2.618*BC

        Args:
            Prix OHLC et volume
            xa_ratio, ab_ratio, bc_ratio, cd_ratio: Ratios de Fibonacci cibles
            tolerance: Tolérance pour les ratios (0.15 recommandé)
            Poids pour le scoring pondéré
            ab_ratio: TODO: documenter.
            ab_weight: TODO: documenter.
            bc_ratio: TODO: documenter.
            bc_weight: TODO: documenter.
            cd_ratio: TODO: documenter.
            cd_weight: TODO: documenter.
            close_prices: TODO: documenter.
            high_prices: TODO: documenter.
            low_prices: TODO: documenter.
            open_prices: TODO: documenter.
            volume: TODO: documenter.
            xa_ratio: TODO: documenter.
            xa_weight: TODO: documenter.

        Returns:
            Array des scores de détection (0.0 à 1.0)
        """
        signals = np.zeros(len(open_prices), dtype=np.float64)
        min_pattern_length = 20
        # AUDIT FIX C-B6-1: supprimé patterns_found (variable morte)

        for i in range(min_pattern_length * 2, len(open_prices)):
            start_idx = i - min_pattern_length * 2

            # Trouver tous les pivots dans la fenêtre
            pivots_idx, pivots_price, pivots_type = find_pivots_simple(
                close_prices[start_idx : i + 1], 0.003, 3
            )

            if len(pivots_idx) < 5:
                continue

            # Chercher les patterns XABCD potentiels
            for start_pattern in range(
                max(0, len(pivots_idx) - 8), len(pivots_idx) - 4
            ):
                if start_pattern + 4 >= len(pivots_idx):
                    continue

                # Extraire 5 pivots consécutifs (X-A-B-C-D)
                x_idx = pivots_idx[start_pattern] + start_idx
                a_idx = pivots_idx[start_pattern + 1] + start_idx
                b_idx = pivots_idx[start_pattern + 2] + start_idx
                c_idx = pivots_idx[start_pattern + 3] + start_idx
                d_idx = pivots_idx[start_pattern + 4] + start_idx

                x_price = pivots_price[start_pattern]
                a_price = pivots_price[start_pattern + 1]
                b_price = pivots_price[start_pattern + 2]
                c_price = pivots_price[start_pattern + 3]
                d_price = pivots_price[start_pattern + 4]

                x_type = pivots_type[start_pattern]
                a_type = pivots_type[start_pattern + 1]
                b_type = pivots_type[start_pattern + 2]
                c_type = pivots_type[start_pattern + 3]
                d_type = pivots_type[start_pattern + 4]

                # === VALIDATION DES DONNÉES STRICTE ===
                if not (x_idx < a_idx < b_idx < c_idx < d_idx <= i):
                    continue

                if (
                    x_price <= 0
                    or a_price <= 0
                    or b_price <= 0
                    or c_price <= 0
                    or d_price <= 0
                ):
                    continue

                # Correction: Pattern must complete exactly today
                if d_idx != i:
                    continue

                # === VÉRIFICATION DE LA SÉQUENCE BAT BAISSIER ===
                # X(low) -> A(high) -> B(low) -> C(high) -> D(low)
                if not (
                    x_type == -1
                    and a_type == 1
                    and b_type == -1
                    and c_type == 1
                    and d_type == -1
                ):
                    continue

                # === CALCUL DES RATIOS DE FIBONACCI SÉCURISÉ ===

                # Mouvement de base XA
                xa_move = abs(a_price - x_price)
                if xa_move <= 0:
                    continue

                # Ratio AB/XA (doit être ~0.382)
                ab_move = abs(a_price - b_price)
                ab_calculated_ratio = safe_divide(ab_move, xa_move)

                # Ratio BC/AB (doit être ~0.886)
                bc_move = abs(c_price - b_price)
                bc_calculated_ratio = safe_divide(bc_move, ab_move)

                # Ratio CD/BC (doit être ~2.618)
                cd_move = abs(b_price - d_price)
                cd_calculated_ratio = safe_divide(cd_move, bc_move)

                # === ÉVALUATION DES RATIOS (SCORING GRADUEL) ===

                # Score AB (proximité avec 0.382)
                ab_error = abs(ab_calculated_ratio - xa_ratio)
                ab_score = max(0.0, 1.0 - safe_divide(ab_error, tolerance))

                # Score BC (proximité avec 0.886)
                bc_error = abs(bc_calculated_ratio - bc_ratio)
                bc_score = max(0.0, 1.0 - safe_divide(bc_error, tolerance))

                # Score CD (proximité avec 2.618)
                cd_error = abs(cd_calculated_ratio - cd_ratio)
                cd_score = max(0.0, 1.0 - safe_divide(cd_error, tolerance))

                # AUDIT FIX C-B6-1: xa_score graduel basé sur cd_xa_ratio (cible Bat=0.886)
                # Un Bat se distingue du Gartley/Butterfly par sa zone D à 88.6% du mouvement XA
                cd_xa_ratio = safe_divide(cd_move, xa_move)
                cd_xa_error = abs(cd_xa_ratio - 0.886)
                xa_score = max(0.0, 1.0 - safe_divide(cd_xa_error, tolerance))

                # Seuil minimum (légèrement relevé)
                min_acceptable_score = 0.30
                if (
                    ab_score < min_acceptable_score
                    or bc_score < min_acceptable_score
                    or cd_score < min_acceptable_score
                ):
                    continue

                # === CALCUL DU SCORE PONDÉRÉ FINAL ===

                components = np.array([xa_score, ab_score, bc_score, cd_score])
                weights = np.array([xa_weight, ab_weight, bc_weight, cd_weight])

                pattern_score = calculate_weighted_score(components, weights)
                signals[i] = max(signals[i], pattern_score)

        return signals

    @njit
    def detect_crab_bull_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        xa_ratio: float,
        ab_ratio: float,
        bc_ratio: float,
        cd_ratio: float,
        tolerance: float,
        xa_weight: float,
        ab_weight: float,
        bc_weight: float,
        cd_weight: float,
    ) -> np.ndarray:
        """VERSION CORRIGÉE - Détection du pattern Crab Bull.

        CORRECTIONS APPORTÉES:
        1. Conditions réduites - scoring pondéré au lieu de conditions binaires strictes
        2. Gestion des erreurs silencieuses avec validation stricte
        3. Tolérance augmentée à 0.15 pour marchés volatils
        4. Seuil minimum très permissif pour augmenter les détections

        PATTERN CRAB HAUSSIER:
        - Point X: Pic initial
        - Point A: Creux (retracement de X)
        - Point B: Pic (rebond de A, ~61.8% du mouvement XA)
        - Point C: Creux (retracement de B, ~88.6% du mouvement AB)
        - Point D: Pic de completion (extension ~361.8% du mouvement BC)

        Ratios cibles: XA=base, AB=0.382*XA, BC=0.886*AB, CD=3.618*BC

        Args:
            Prix OHLC et volume
            xa_ratio, ab_ratio, bc_ratio, cd_ratio: Ratios de Fibonacci cibles
            tolerance: Tolérance pour les ratios (0.15 recommandé)
            Poids pour le scoring pondéré
            ab_ratio: TODO: documenter.
            ab_weight: TODO: documenter.
            bc_ratio: TODO: documenter.
            bc_weight: TODO: documenter.
            cd_ratio: TODO: documenter.
            cd_weight: TODO: documenter.
            close_prices: TODO: documenter.
            high_prices: TODO: documenter.
            low_prices: TODO: documenter.
            open_prices: TODO: documenter.
            volume: TODO: documenter.
            xa_ratio: TODO: documenter.
            xa_weight: TODO: documenter.

        Returns:
            Array des scores de détection (0.0 à 1.0)
        """
        signals = np.zeros(len(open_prices), dtype=np.float64)
        min_pattern_length = 20
        # AUDIT FIX C-B6-2: supprimé patterns_found (variable morte)

        for i in range(min_pattern_length * 2, len(open_prices)):
            start_idx = i - min_pattern_length * 2

            # Trouver tous les pivots dans la fenêtre
            pivots_idx, pivots_price, pivots_type = find_pivots_simple(
                close_prices[start_idx : i + 1], 0.003, 3
            )

            if len(pivots_idx) < 5:
                continue

            # Chercher les patterns XABCD potentiels
            for start_pattern in range(
                max(0, len(pivots_idx) - 8), len(pivots_idx) - 4
            ):
                if start_pattern + 4 >= len(pivots_idx):
                    continue

                # Extraire 5 pivots consécutifs (X-A-B-C-D)
                x_idx = pivots_idx[start_pattern] + start_idx
                a_idx = pivots_idx[start_pattern + 1] + start_idx
                b_idx = pivots_idx[start_pattern + 2] + start_idx
                c_idx = pivots_idx[start_pattern + 3] + start_idx
                d_idx = pivots_idx[start_pattern + 4] + start_idx

                x_price = pivots_price[start_pattern]
                a_price = pivots_price[start_pattern + 1]
                b_price = pivots_price[start_pattern + 2]
                c_price = pivots_price[start_pattern + 3]
                d_price = pivots_price[start_pattern + 4]

                x_type = pivots_type[start_pattern]
                a_type = pivots_type[start_pattern + 1]
                b_type = pivots_type[start_pattern + 2]
                c_type = pivots_type[start_pattern + 3]
                d_type = pivots_type[start_pattern + 4]

                # === VALIDATION DES DONNÉES STRICTE ===
                if not (x_idx < a_idx < b_idx < c_idx < d_idx <= i):
                    continue

                if (
                    x_price <= 0
                    or a_price <= 0
                    or b_price <= 0
                    or c_price <= 0
                    or d_price <= 0
                ):
                    continue

                # Correction: Pattern must complete exactly today
                if d_idx != i:
                    continue

                # === VÉRIFICATION DE LA SÉQUENCE CRAB HAUSSIER ===
                # X(high) -> A(low) -> B(high) -> C(low) -> D(high)
                if not (
                    x_type == 1
                    and a_type == -1
                    and b_type == 1
                    and c_type == -1
                    and d_type == 1
                ):
                    continue

                # === CALCUL DES RATIOS DE FIBONACCI ===

                # Mouvement de base XA
                xa_move = abs(x_price - a_price)
                if xa_move <= 0:
                    continue

                # Ratio AB/XA (doit être ~0.382)
                ab_move = abs(b_price - a_price)
                ab_calculated_ratio = safe_divide(ab_move, xa_move)

                # Ratio BC/AB (doit être ~0.886)
                bc_move = abs(c_price - b_price)
                bc_calculated_ratio = safe_divide(bc_move, ab_move)

                # Ratio CD/BC (doit être ~3.618)
                cd_move = abs(d_price - c_price)
                cd_calculated_ratio = safe_divide(cd_move, bc_move)

                # === ÉVALUATION DES RATIOS (SCORING GRADUEL) ===

                # Score AB (proximité avec 0.382)
                ab_error = abs(ab_calculated_ratio - ab_ratio)
                ab_score = max(0.0, 1.0 - safe_divide(ab_error, tolerance))

                # Score BC (proximité avec 0.886)
                bc_error = abs(bc_calculated_ratio - bc_ratio)
                bc_score = max(0.0, 1.0 - safe_divide(bc_error, tolerance))

                # Score CD (proximité avec 3.618)
                cd_tolerance = tolerance * (cd_ratio if cd_ratio > 1.0 else 1.0)
                cd_error = abs(cd_calculated_ratio - cd_ratio)
                cd_score = max(0.0, 1.0 - safe_divide(cd_error, cd_tolerance))

                # AUDIT FIX C-B6-2: xa_score graduel basé sur cd_xa_ratio (cible Crab=1.618)
                # Un Crab se distingue par sa zone D à 161.8% du mouvement XA (dépassement X)
                cd_xa_tolerance = tolerance * 1.618
                cd_xa_ratio = safe_divide(cd_move, xa_move)
                cd_xa_error = abs(cd_xa_ratio - 1.618)
                xa_score = max(0.0, 1.0 - safe_divide(cd_xa_error, cd_xa_tolerance))

                # Seuil minimum relevé (cohérent avec Gartley/Butterfly)
                min_acceptable_score = 0.30
                if (
                    ab_score < min_acceptable_score
                    or bc_score < min_acceptable_score
                    or cd_score < min_acceptable_score
                ):
                    continue

                # === CALCUL DU SCORE PONDÉRÉ FINAL ===

                components = np.array([xa_score, ab_score, bc_score, cd_score])
                weights = np.array([xa_weight, ab_weight, bc_weight, cd_weight])

                pattern_score = calculate_weighted_score(components, weights)
                signals[i] = max(signals[i], pattern_score)

        return signals

    @njit
    def detect_crab_bear_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        xa_ratio: float,
        ab_ratio: float,
        bc_ratio: float,
        cd_ratio: float,
        tolerance: float,
        xa_weight: float,
        ab_weight: float,
        bc_weight: float,
        cd_weight: float,
    ) -> np.ndarray:
        """VERSION CORRIGÉE - Détection du pattern Crab Bear.

        CORRECTIONS APPORTÉES:
        1. Conditions réduites - scoring pondéré au lieu de conditions binaires strictes
        2. Gestion des erreurs mathématiques avec safe_divide partout
        3. Gestion des erreurs silencieuses avec validation stricte
        4. Tolérance augmentée et seuil minimum très permissif

        PATTERN CRAB BAISSIER:
        - Point X: Creux initial
        - Point A: Pic (rebond de X)
        - Point B: Creux (retracement de A, ~38.2% du mouvement XA)
        - Point C: Pic (rebond de B, ~88.6% du mouvement AB)
        - Point D: Creux de completion (extension ~361.8% du mouvement BC)

        Ratios cibles: XA=base, AB=0.382*XA, BC=0.886*AB, CD=3.618*BC

        Args:
            Prix OHLC et volume
            xa_ratio, ab_ratio, bc_ratio, cd_ratio: Ratios de Fibonacci cibles
            tolerance: Tolérance pour les ratios (0.15 recommandé)
            Poids pour le scoring pondéré
            ab_ratio: TODO: documenter.
            ab_weight: TODO: documenter.
            bc_ratio: TODO: documenter.
            bc_weight: TODO: documenter.
            cd_ratio: TODO: documenter.
            cd_weight: TODO: documenter.
            close_prices: TODO: documenter.
            high_prices: TODO: documenter.
            low_prices: TODO: documenter.
            open_prices: TODO: documenter.
            volume: TODO: documenter.
            xa_ratio: TODO: documenter.
            xa_weight: TODO: documenter.

        Returns:
            Array des scores de détection (0.0 à 1.0)
        """
        signals = np.zeros(len(open_prices), dtype=np.float64)
        min_pattern_length = 20
        # AUDIT FIX C-B6-3: supprimé patterns_found (variable morte)

        for i in range(min_pattern_length * 2, len(open_prices)):
            start_idx = i - min_pattern_length * 2

            # Trouver tous les pivots dans la fenêtre
            pivots_idx, pivots_price, pivots_type = find_pivots_simple(
                close_prices[start_idx : i + 1], 0.003, 3
            )

            if len(pivots_idx) < 5:
                continue

            # Chercher les patterns XABCD potentiels
            for start_pattern in range(
                max(0, len(pivots_idx) - 8), len(pivots_idx) - 4
            ):
                if start_pattern + 4 >= len(pivots_idx):
                    continue

                # Extraire 5 pivots consécutifs (X-A-B-C-D)
                x_idx = pivots_idx[start_pattern] + start_idx
                a_idx = pivots_idx[start_pattern + 1] + start_idx
                b_idx = pivots_idx[start_pattern + 2] + start_idx
                c_idx = pivots_idx[start_pattern + 3] + start_idx
                d_idx = pivots_idx[start_pattern + 4] + start_idx

                x_price = pivots_price[start_pattern]
                a_price = pivots_price[start_pattern + 1]
                b_price = pivots_price[start_pattern + 2]
                c_price = pivots_price[start_pattern + 3]
                d_price = pivots_price[start_pattern + 4]

                x_type = pivots_type[start_pattern]
                a_type = pivots_type[start_pattern + 1]
                b_type = pivots_type[start_pattern + 2]
                c_type = pivots_type[start_pattern + 3]
                d_type = pivots_type[start_pattern + 4]

                # === VALIDATION DES DONNÉES STRICTE ===
                if not (x_idx < a_idx < b_idx < c_idx < d_idx <= i):
                    continue

                if (
                    x_price <= 0
                    or a_price <= 0
                    or b_price <= 0
                    or c_price <= 0
                    or d_price <= 0
                ):
                    continue

                # Correction: Pattern must complete exactly today
                if d_idx != i:
                    continue

                # === VÉRIFICATION DE LA SÉQUENCE CRAB BAISSIER ===
                # X(low) -> A(high) -> B(low) -> C(high) -> D(low)
                if not (
                    x_type == -1
                    and a_type == 1
                    and b_type == -1
                    and c_type == 1
                    and d_type == -1
                ):
                    continue

                # === CALCUL DES RATIOS DE FIBONACCI SÉCURISÉ ===

                # Mouvement de base XA
                xa_move = abs(a_price - x_price)
                if xa_move <= 0:
                    continue

                # Ratio AB/XA (doit être ~0.382)
                ab_move = abs(a_price - b_price)
                ab_calculated_ratio = safe_divide(ab_move, xa_move)

                # Ratio BC/AB (doit être ~0.886)
                bc_move = abs(c_price - b_price)
                bc_calculated_ratio = safe_divide(bc_move, ab_move)

                # Ratio CD/BC (doit être ~3.618)
                cd_move = abs(b_price - d_price)
                cd_calculated_ratio = safe_divide(cd_move, bc_move)

                # === ÉVALUATION DES RATIOS (SCORING GRADUEL) ===

                # Score AB (proximité avec 0.382)
                ab_error = abs(ab_calculated_ratio - ab_ratio)
                ab_score = max(0.0, 1.0 - safe_divide(ab_error, tolerance))

                # Score BC (proximité avec 0.886)
                bc_error = abs(bc_calculated_ratio - bc_ratio)
                bc_score = max(0.0, 1.0 - safe_divide(bc_error, tolerance))

                # Score CD (proximité avec 3.618)
                cd_tolerance = tolerance * (cd_ratio if cd_ratio > 1.0 else 1.0)
                cd_error = abs(cd_calculated_ratio - cd_ratio)
                cd_score = max(0.0, 1.0 - safe_divide(cd_error, cd_tolerance))

                # AUDIT FIX C-B6-3: xa_score graduel basé sur cd_xa_ratio (cible Crab=1.618)
                cd_xa_tolerance = tolerance * 1.618
                cd_xa_ratio = safe_divide(cd_move, xa_move)
                cd_xa_error = abs(cd_xa_ratio - 1.618)
                xa_score = max(0.0, 1.0 - safe_divide(cd_xa_error, cd_xa_tolerance))

                # Seuil minimum relevé (0.10 était beaucoup trop permissif)
                min_acceptable_score = 0.30
                if (
                    ab_score < min_acceptable_score
                    or bc_score < min_acceptable_score
                    or cd_score < min_acceptable_score
                ):
                    continue

                # === CALCUL DU SCORE PONDÉRÉ FINAL ===

                components = np.array([xa_score, ab_score, bc_score, cd_score])
                weights = np.array([xa_weight, ab_weight, bc_weight, cd_weight])

                pattern_score = calculate_weighted_score(components, weights)
                signals[i] = max(signals[i], pattern_score)

        return signals

    @njit
    def detect_shark_bear_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        tolerance: float = 0.15,
        min_pattern_size: float = 0.008,
        pattern_weight: float = 1.0,
    ) -> np.ndarray:
        """VERSION CORRIGÉE - Shark Bear fortement contraint.

        Évite les >80% de faux positifs en exigeant un vrai ^-shape profond.
        """
        signals = np.zeros(len(open_prices), dtype=np.float64)
        window_size = 15

        for i in range(window_size, len(open_prices)):
            start_idx = i - window_size

            # Trouver le maximum absolu de la fenêtre pour le sommet du ^
            max_price = high_prices[start_idx]
            max_idx = start_idx
            for j in range(start_idx + 1, i):
                if high_prices[j] > max_price:
                    max_price = high_prices[j]
                    max_idx = j

            # Le max doit être encadré pour former un vrai ^
            if max_idx < start_idx + 3 or max_idx > i - 2:
                continue

            min_price_before = np.min(low_prices[start_idx:max_idx])
            climb_size = max_price - min_price_before

            if climb_size <= 0:
                continue

            current_price = close_prices[i]

            if current_price < max_price:
                drop_recovery_size = max_price - current_price
                recovery_ratio = safe_divide(drop_recovery_size, climb_size)

                # Shark Bear strict : descente "en V inversé" de 88.6% à 113% de la hausse précédente
                if (
                    0.886 < recovery_ratio < 1.13
                    and climb_size > min_price_before * min_pattern_size * 2
                ):
                    # AUDIT FIX C-B6-4: volume hard filter →  score graduel
                    # En forex, le tick volume n'est pas garanti 1.2× → transformé en bonus
                    if max_idx > start_idx:
                        vol_climb = np.mean(volume[start_idx:max_idx])
                    else:
                        vol_climb = 0.0
                    if i >= max_idx:
                        vol_drop = np.mean(volume[max_idx : i + 1])
                    else:
                        vol_drop = 0.0
                    if vol_climb > 0:
                        vol_ratio = safe_divide(vol_drop, vol_climb)
                        volume_score = min(1.0, safe_divide(vol_ratio, 1.2))
                    else:
                        volume_score = 0.5  # neutre si pas de volume

                    pattern_score = 1.0 - abs(1.0 - recovery_ratio)
                    final_score = pattern_score * 0.7 + volume_score * 0.3
                    signals[i] = max(0.0, min(1.0, final_score))

        return signals

    @njit
    def detect_uptrend_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        min_length: int,
        min_slope: float,
        max_pullback_ratio: float,
        slope_weight: float,
        consistency_weight: float,
        volume_weight: float,
    ) -> np.ndarray:
        """VERSION CORRIGÉE - Détection de Tendance Haussière avec Régression Linéaire."""
        signals = np.zeros(len(open_prices), dtype=np.float64)

        for i in range(min_length, len(open_prices)):
            start_idx = i - min_length
            window_closes = close_prices[start_idx : i + 1]

            # Utiliser la régression linéaire pour une pente robuste
            x_values = np.arange(len(window_closes), dtype=np.float64)
            slope = calculate_slope_safe(x_values, window_closes)

            # Normaliser la pente par le prix moyen pour la rendre comparable
            mean_price = np.mean(window_closes)
            if mean_price <= 0:
                continue
            normalized_slope = safe_divide(slope, mean_price)

            if normalized_slope < min_slope:
                continue

            # Vérifier la consistance (la plupart des mouvements sont haussiers)
            consistency = calculate_trend_consistency(
                window_closes, 0, len(window_closes) - 1, 1
            )
            if consistency < 0.55:  # Au moins 55% de bougies haussières/plates
                continue

            # Vérifier les retracements (pullbacks)
            max_pullback = 0.0
            running_high = window_closes[0]
            for j in range(1, len(window_closes)):
                if window_closes[j] > running_high:
                    running_high = window_closes[j]
                else:
                    pullback = safe_divide(
                        (running_high - window_closes[j]), running_high
                    )
                    max_pullback = max(max_pullback, pullback)

            if max_pullback > max_pullback_ratio:
                continue

            # Calcul des scores
            slope_score = min(
                1.0, safe_divide(normalized_slope, min_slope * 5)
            )  # Bonus pour les pentes plus fortes
            consistency_score = (consistency - 0.5) * 2  # Mettre à l'échelle de 0-1
            pullback_score = 1.0 - safe_divide(max_pullback, max_pullback_ratio)

            components = np.array([slope_score, consistency_score, pullback_score])
            weights = np.array(
                [slope_weight, consistency_weight, volume_weight]
            )  # AUDIT FIX C-B6-5: volume_weight remplace 0.2 hardcodé (pullback_weight)
            signals[i] = calculate_weighted_score(components, weights)

        return signals

    @njit
    def detect_downtrend_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        min_length: int,
        min_slope: float,
        max_pullback_ratio: float,
        slope_weight: float,
        consistency_weight: float,
        volume_weight: float,
    ) -> np.ndarray:
        """VERSION CORRIGÉE - Détection de Tendance Baissière avec Régression Linéaire."""
        signals = np.zeros(len(open_prices), dtype=np.float64)

        for i in range(min_length, len(open_prices)):
            start_idx = i - min_length
            window_closes = close_prices[start_idx : i + 1]

            # Utiliser la régression linéaire pour une pente robuste
            x_values = np.arange(len(window_closes), dtype=np.float64)
            slope = calculate_slope_safe(x_values, window_closes)

            # Normaliser la pente
            mean_price = np.mean(window_closes)
            if mean_price <= 0:
                continue
            normalized_slope = safe_divide(slope, mean_price)

            if normalized_slope > min_slope:  # min_slope est négatif
                continue

            # Vérifier la consistance
            consistency = calculate_trend_consistency(
                window_closes, 0, len(window_closes) - 1, -1
            )
            if consistency < 0.55:
                continue

            # Vérifier les rebonds (pullbacks)
            max_pullback = 0.0
            running_low = window_closes[0]
            for j in range(1, len(window_closes)):
                if window_closes[j] < running_low:
                    running_low = window_closes[j]
                else:
                    pullback = safe_divide(
                        (window_closes[j] - running_low), running_low
                    )
                    max_pullback = max(max_pullback, pullback)

            if max_pullback > max_pullback_ratio:
                continue

            # Calcul des scores
            slope_score = min(
                1.0, safe_divide(abs(normalized_slope), abs(min_slope) * 5)
            )
            consistency_score = (consistency - 0.5) * 2
            pullback_score = 1.0 - safe_divide(max_pullback, max_pullback_ratio)

            components = np.array([slope_score, consistency_score, pullback_score])
            weights = np.array(
                [slope_weight, consistency_weight, volume_weight]
            )  # AUDIT FIX C-B6-5
            signals[i] = calculate_weighted_score(components, weights)

        return signals

    @njit
    def detect_sideways_trend_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        min_length: int,
        max_slope_tolerance: float,
        max_range_volatility: float,
        min_containment_ratio: float,
        slope_weight: float,
        containment_weight: float,
        duration_weight: float,
    ) -> np.ndarray:
        """VERSION CORRIGÉE 2.0 - Détection du DÉBUT d'un Sideways Trend (Détection d'Événement).

        Utilise une variable d'état pour marquer uniquement la transition vers un état de range.
        """
        signals = np.zeros(len(open_prices), dtype=np.float64)
        # NOUVEAU: Variable d'état pour suivre si nous sommes déjà dans un range.
        in_sideways_trend = False

        for i in range(min_length, len(open_prices)):
            start_idx = i - min_length
            window_closes = close_prices[start_idx : i + 1]
            window_highs = high_prices[start_idx : i + 1]
            window_lows = low_prices[start_idx : i + 1]

            # Étape 1: Calculer si la fenêtre ACTUELLE est en range
            is_currently_sideways = False

            # Calcul de la pente
            x_values = np.arange(len(window_closes), dtype=np.float64)
            slope = calculate_slope_safe(x_values, window_closes)

            mean_price = np.mean(window_closes)
            # AUDIT FIX C-B8-3: init défensive — évite UnboundLocalError si mean_price <= 0
            normalized_slope = 0.0
            if mean_price > 0:
                normalized_slope = safe_divide(slope, mean_price)

                # Condition de pente faible
                if abs(normalized_slope) <= max_slope_tolerance:
                    # Condition de faible volatilité et de confinement
                    std_dev = calculate_std(window_closes)
                    channel_half_width = max(
                        std_dev * 1.5, mean_price * max_range_volatility
                    )
                    upper_band = mean_price + channel_half_width
                    lower_band = mean_price - channel_half_width

                    contained_candles = 0
                    for j in range(len(window_closes)):
                        if (
                            window_highs[j] <= upper_band
                            and window_lows[j] >= lower_band
                        ):
                            contained_candles += 1

                    containment_ratio = safe_divide(
                        contained_candles, len(window_closes)
                    )

                    if containment_ratio >= min_containment_ratio:
                        is_currently_sideways = True

            # LOGIQUE DE TRANSITION CLÉ
            # On ne marque un signal que si on ENTRE dans un état de range.
            if is_currently_sideways and not in_sideways_trend:
                # Calcul des scores uniquement au moment de la détection
                slope_score = 1.0 - safe_divide(
                    abs(normalized_slope), max_slope_tolerance
                )
                containment_score = safe_divide(
                    containment_ratio - min_containment_ratio,
                    1.0 - min_containment_ratio,
                )
                # AUDIT FIX C-B8-3: scoring graduel de la durée
                duration_score = min(
                    1.0, safe_divide(float(i - start_idx), float(min_length * 2))
                )

                components = np.array([slope_score, containment_score, duration_score])
                weights = np.array([slope_weight, containment_weight, duration_weight])

                signals[i] = calculate_weighted_score(components, weights)

            # MISE À JOUR DE L'ÉTAT pour la prochaine itération
            in_sideways_trend = is_currently_sideways

        return signals

    @njit
    def detect_exhaustion_gap_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        min_gap_ratio: float,
        max_continuation: int,
        min_volume_climax: float,
        gap_weight: float,
        reversal_weight: float,
        volume_weight: float,
    ) -> np.ndarray:
        """VERSION SANS LOOKAHEAD."""
        signals = np.zeros(len(open_prices), dtype=np.float64)
        for i in range(15, len(open_prices)):
            gap_up = open_prices[i] > high_prices[i - 1]
            gap_down = open_prices[i] < low_prices[i - 1]
            if not gap_up and not gap_down:
                continue

            gap_ratio = (
                safe_divide((open_prices[i] - high_prices[i - 1]), high_prices[i - 1])
                if gap_up
                else safe_divide(
                    (low_prices[i - 1] - open_prices[i]), low_prices[i - 1]
                )
            )
            if gap_ratio < min_gap_ratio:
                continue

            # FIX: volume en scoring graduel (pas filtre dur) → universel cross-asset
            volume_climax_score = 0.3  # score neutre par défaut (pas de données vol)
            avg_vol = np.mean(volume[i - 15 : i])
            if avg_vol > 0:
                vol_ratio = safe_divide(volume[i - 1], avg_vol)
                # seuil abaissé 1.5× (Forex volume normalisé, rarement ×2.5)
                volume_climax_score = min(
                    1.0, safe_divide(vol_ratio, min_volume_climax)
                )
            # volume_climax_score == 0 → ne bloque plus (supprimé)

            # FIX: normalisation ATR du mouvement de tendance (cross-asset)
            prices_before = close_prices[i - 15 : i]
            atr_local = 0.0
            for _j in range(i - 14, i):
                if _j > 0:
                    tr_j = max(
                        high_prices[_j] - low_prices[_j],
                        abs(high_prices[_j] - close_prices[_j - 1]),
                        abs(low_prices[_j] - close_prices[_j - 1]),
                    )
                    atr_local += tr_j
            atr_local = (
                atr_local / 14.0 if atr_local > 0 else (high_prices[i] - low_prices[i])
            )
            if atr_local <= 0:
                atr_local = high_prices[i] - low_prices[i]
            if atr_local <= 0:
                continue
            # Mouvement normalisé: doit représenter ≥ 0.5 ATR × 10 bougies
            trend_size_normalized = safe_divide(
                abs(prices_before[-1] - prices_before[0]), atr_local * 10
            )
            if trend_size_normalized < 0.5:
                continue

            gap_score = min(1.0, safe_divide(gap_ratio, min_gap_ratio * 2))

            # AUDIT FIX C-B8-4: reversal_score réel au lieu de 1.0 hardcodé
            if gap_up:
                reversal_score = 1.0 if close_prices[i] < open_prices[i] else 0.3
            else:
                reversal_score = 1.0 if close_prices[i] > open_prices[i] else 0.3
            components = np.array([gap_score, volume_climax_score, reversal_score])
            weights = np.array([gap_weight, volume_weight, reversal_weight])
            signals[i] = calculate_weighted_score(components, weights)
        return signals

    @njit
    def detect_three_drives_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        min_drive_length: int,
        max_retracement_ratio: float,
        min_extension_ratio: float,
        drive_weight: float,
        retracement_weight: float,
        extension_weight: float,
    ) -> np.ndarray:
        """Détection du pattern Three Drives.

        FIX: direction utilisée pour valider que chaque drive successif
        progresse bien dans la bonne direction (pivots ascendants pour bull,
        descendants pour bear). Sans ce check, une alternance correcte
        mais sans progression nette passait comme Three Drives valide.
        """
        signals = np.zeros(len(open_prices), dtype=np.float64)
        min_pattern_length = min_drive_length * 3
        # AUDIT FIX: supprimé patterns_found (variable morte)

        for i in range(min_pattern_length, len(open_prices)):
            start_idx = i - min_pattern_length

            # Calcul ATR adaptatif
            td_atr_sum = 0.0
            td_atr_count = 0
            for _k in range(start_idx + 1, i + 1):
                if _k < len(high_prices) and _k - 1 >= 0:
                    td_tr = high_prices[_k] - low_prices[_k]
                    td_prev_close = close_prices[_k - 1]
                    if td_prev_close > 0:
                        td_close_range = abs(high_prices[_k] - td_prev_close)
                        td_low_range = abs(low_prices[_k] - td_prev_close)
                        td_tr = max(td_tr, td_close_range, td_low_range)
                    if td_tr > 0:
                        td_atr_sum += td_tr
                        td_atr_count += 1

            _td_close_ref = close_prices[i] if close_prices[i] > 0 else 1.0
            _td_threshold = (
                max(0.001, safe_divide(td_atr_sum, td_atr_count) / _td_close_ref)
                if td_atr_count > 0
                else 0.003
            )

            pivots_idx, pivots_price, pivots_type = find_pivots_simple(
                close_prices[start_idx : i + 1], _td_threshold, min_drive_length
            )

            if len(pivots_idx) < 6:
                continue

            for start_pattern in range(len(pivots_idx) - 5):
                if start_pattern + 5 >= len(pivots_idx):
                    continue

                # AUDIT FIX C-B8-5: arrays NumPy au lieu de listes Python (.append() crash Numba nopython)
                pivots_indices = np.zeros(6, dtype=np.int64)
                pivots_prices = np.zeros(6, dtype=np.float64)
                pivots_types = np.zeros(6, dtype=np.int64)
                n_collected = 0

                for j in range(6):
                    if start_pattern + j < len(pivots_idx):
                        pivots_indices[j] = pivots_idx[start_pattern + j] + start_idx
                        pivots_prices[j] = pivots_price[start_pattern + j]
                        pivots_types[j] = pivots_type[start_pattern + j]
                        n_collected += 1

                if n_collected < 6:
                    continue

                all_valid = True
                for price in pivots_prices:
                    if price <= 0:
                        all_valid = False
                        break
                if not all_valid:
                    continue

                bull_seq = True
                bear_seq = True
                exp_bull = [-1, 1, -1, 1, -1, 1]
                exp_bear = [1, -1, 1, -1, 1, -1]
                for _k in range(6):
                    if pivots_types[_k] != exp_bull[_k]:
                        bull_seq = False
                    if pivots_types[_k] != exp_bear[_k]:
                        bear_seq = False

                if bull_seq:
                    direction = 1
                elif bear_seq:
                    direction = -1
                else:
                    continue

                # FIX: direction utilisée pour valider la progression des pivots
                # Bull : chaque drive high est plus haut que le précédent
                #        p1 < p3 < p5 (sommets ascendants)
                #        p0 < p2 < p4 (creux ascendants)
                # Bear : chaque drive low est plus bas que le précédent
                #        p1 > p3 > p5 (creux descendants)
                #        p0 > p2 > p4 (sommets descendants)
                if direction == 1:
                    if not (
                        pivots_prices[1] > pivots_prices[0]  # premier drive monte
                        and pivots_prices[3] > pivots_prices[1]  # drive 2 > drive 1
                        and pivots_prices[5] > pivots_prices[3]  # drive 3 > drive 2
                        and pivots_prices[2] > pivots_prices[0]  # creux 1 > départ
                        and pivots_prices[4] > pivots_prices[2]  # creux 2 > creux 1
                    ):
                        continue
                else:  # direction == -1
                    if not (
                        pivots_prices[1] < pivots_prices[0]
                        and pivots_prices[3] < pivots_prices[1]
                        and pivots_prices[5] < pivots_prices[3]
                        and pivots_prices[2] < pivots_prices[0]
                        and pivots_prices[4] < pivots_prices[2]
                    ):
                        continue

                drive1 = abs(pivots_prices[1] - pivots_prices[0])
                retr1 = abs(pivots_prices[2] - pivots_prices[1])
                drive2 = abs(pivots_prices[3] - pivots_prices[2])
                retr2 = abs(pivots_prices[4] - pivots_prices[3])
                drive3 = abs(pivots_prices[5] - pivots_prices[4])

                if drive1 <= 0 or drive2 <= 0 or drive3 <= 0:
                    continue

                retr1_ratio = safe_divide(retr1, drive1)
                retr2_ratio = safe_divide(retr2, drive2)

                if (
                    retr1_ratio > max_retracement_ratio
                    or retr2_ratio > max_retracement_ratio
                ):
                    continue

                retr1_error = min(abs(retr1_ratio - 0.618), abs(retr1_ratio - 0.786))
                retr2_error = min(abs(retr2_ratio - 0.618), abs(retr2_ratio - 0.786))

                retr1_score = max(0.0, 1.0 - safe_divide(retr1_error, 0.2))
                retr2_score = max(0.0, 1.0 - safe_divide(retr2_error, 0.2))
                retracement_score = (retr1_score + retr2_score) / 2.0

                # Drive 2 and 3 should be 1.272 or 1.618 extensions of the *retracements*
                drive2_ext_ratio = safe_divide(drive2, retr1)
                drive3_ext_ratio = safe_divide(drive3, retr2)

                ext2_score = 1.0
                ext3_score = 1.0

                if drive2_ext_ratio > 0:
                    if drive2_ext_ratio < min_extension_ratio:
                        ext2_score = 0.0
                    else:
                        ext2_error = min(
                            abs(drive2_ext_ratio - 1.272), abs(drive2_ext_ratio - 1.618)
                        )
                        ext2_score = max(0.0, 1.0 - safe_divide(ext2_error, 0.3))

                if drive3_ext_ratio > 0:
                    if drive3_ext_ratio < min_extension_ratio:
                        ext3_score = 0.0
                    else:
                        ext3_error = min(
                            abs(drive3_ext_ratio - 1.272), abs(drive3_ext_ratio - 1.618)
                        )
                        ext3_score = max(0.0, 1.0 - safe_divide(ext3_error, 0.3))

                extension_score = (ext2_score + ext3_score) / 2.0

                drive_score = 1.0
                if drive1 > 0 and drive2 > 0 and drive3 > 0:
                    drive_consistency = (
                        1.0
                        - (
                            safe_divide(abs(drive1 - drive2), drive1)
                            + safe_divide(abs(drive2 - drive3), drive2)
                        )
                        / 2.0
                    )
                    drive_score = max(0.0, drive_consistency)

                if (
                    retracement_score < 0.15
                    or extension_score < 0.10
                    or drive_score < 0.15
                ):
                    continue

                if i - pivots_indices[-1] > min_drive_length:
                    continue

                components = np.array([drive_score, retracement_score, extension_score])
                weights = np.array([drive_weight, retracement_weight, extension_weight])
                pattern_score = calculate_weighted_score(components, weights)
                signals[i] = max(signals[i], pattern_score)

        return signals

    @njit
    def detect_elliott_wave_1_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        min_wave_ratio: float,
        max_wave_ratio: float,
        min_volume_confirmation: float,
        wave_weight: float,
        ratio_weight: float,
        volume_weight: float,
    ) -> np.ndarray:
        """Détection Elliott Wave 1 — premier mouvement impulsif d'un cycle.

        Définition (Prechter & Frost) :
        - Wave 1 = premier mouvement impulsif après une correction (Wave 0)
        - Alternance directionnelle obligatoire entre Wave 0 et Wave 1
        - Volume croissant pendant Wave 1 vs Wave 0
        - Ratio Wave1/Wave0 typiquement entre 0.5 et 1.8 en forex

        CORRECTIONS :
        P1 — Timing exact `p2g != i` remplacé par tolérance ±min_wave_bars.
             L'ancienne condition exigeait que le dernier pivot tombe exactement
             sur la barre courante — probabilité quasi-nulle → zéro détection.
        P7 — Variable morte `sig_tol` supprimée.

        Seuils pivot ATR [0.015, 0.030] conservés (calibrés pour éviter
        les micro-pivots sur M5 tout en restant adaptatifs sur H1/H4/D1).
        """
        n = len(open_prices)
        signals = np.zeros(n, dtype=np.float64)

        atr_period    = 14
        min_wave_bars = 5
        lookback      = 40
        # P7 : sig_tol supprimée — tolérance directement via min_wave_bars

        # Ratios Fibonacci cibles Wave1/Wave0 (Prechter)
        fib_targets = np.array([0.618, 1.0, 1.618], dtype=np.float64)

        for i in range(lookback + atr_period, n):

            # ── 1. ATR sur 14 barres ──────────────────────────────────────────
            _atr_start = i - atr_period + 1
            _atr_sum   = 0.0
            _atr_cnt   = 0

            for _k in range(_atr_start, i + 1):
                _tr = high_prices[_k] - low_prices[_k]
                _pc = close_prices[_k - 1]
                if _pc > 0.0:
                    _hi_pc = abs(high_prices[_k] - _pc)
                    _lo_pc = abs(low_prices[_k]  - _pc)
                    if _hi_pc > _tr:
                        _tr = _hi_pc
                    if _lo_pc > _tr:
                        _tr = _lo_pc
                if _tr > 0.0:
                    _atr_sum += _tr
                    _atr_cnt += 1

            _ref = close_prices[i]
            if _ref <= 0.0:
                continue

            # ── 2. Seuil pivot adaptatif [0.015, 0.030] ──────────────────────
            if _atr_cnt > 0:
                _raw_thr   = (_atr_sum / _atr_cnt) / _ref
                _pivot_thr = _raw_thr
                if _pivot_thr < 0.015:
                    _pivot_thr = 0.015
                if _pivot_thr > 0.030:
                    _pivot_thr = 0.030
            else:
                _pivot_thr = 0.015

            # ── 3. Recherche des pivots ───────────────────────────────────────
            _start = i - lookback
            pivots_idx, pivots_price, pivots_type = find_pivots_simple(
                close_prices[_start : i + 1], _pivot_thr, min_wave_bars
            )

            n_piv = len(pivots_idx)
            if n_piv < 3:
                continue

            # ── 4. Évaluation des triplettes de pivots consécutifs ────────────
            best_score = 0.0

            for w in range(n_piv - 2):
                # Indices locaux → globaux
                p0g = pivots_idx[w]     + _start
                p1g = pivots_idx[w + 1] + _start
                p2g = pivots_idx[w + 2] + _start

                # P1 : tolérance temporelle ±min_wave_bars
                # L'ancien code : `if p2g != i: continue` causait 0 détection.
                # Un pattern récent dans la fenêtre est valide.
                if i - p2g > min_wave_bars:
                    continue

                p0p = pivots_price[w]
                p1p = pivots_price[w + 1]
                p2p = pivots_price[w + 2]

                # Validation données
                if p0p <= 0.0 or p1p <= 0.0 or p2p <= 0.0:
                    continue
                if p0g >= p1g or p1g >= p2g:
                    continue

                # Longueur minimale par vague
                if (p1g - p0g) < min_wave_bars or (p2g - p1g) < min_wave_bars:
                    continue

                # Alternance directionnelle obligatoire
                wave0_dir = 1 if p1p > p0p else -1
                wave1_dir = 1 if p2p > p1p else -1
                if wave0_dir == wave1_dir:
                    continue

                # Amplitudes
                wave0_size = abs(p1p - p0p)
                wave1_size = abs(p2p - p1p)

                if wave0_size <= 0.0:
                    continue

                # Amplitude minimale = 1 ATR par vague
                _atr_val = _atr_sum / _atr_cnt if _atr_cnt > 0 else _ref * 0.01
                if wave0_size < _atr_val or wave1_size < _atr_val:
                    continue

                # Ratio Wave1/Wave0
                wave_ratio = safe_divide(wave1_size, wave0_size)
                if wave_ratio < min_wave_ratio or wave_ratio > max_wave_ratio:
                    continue

                # Score ratio Fibonacci — proximité aux cibles 0.618, 1.0, 1.618
                _min_fib_err = 999.0
                for _f in range(len(fib_targets)):
                    _err = abs(wave_ratio - fib_targets[_f])
                    if _err < _min_fib_err:
                        _min_fib_err = _err
                ratio_score = max(0.0, 1.0 - safe_divide(_min_fib_err, 0.20))

                # Score amplitude
                wave_score = min(1.0, safe_divide(wave1_size, wave0_size * max_wave_ratio))

                # Score volume — Wave 1 doit avoir plus de volume que Wave 0
                volume_score = 0.5  # neutre si données insuffisantes

                if p0g >= 0 and p2g < len(volume):
                    _w0v = 0.0
                    _w0c = 0
                    for _j in range(p0g, p1g + 1):
                        if _j < len(volume) and volume[_j] > 0.0:
                            _w0v += volume[_j]
                            _w0c += 1

                    _w1v = 0.0
                    _w1c = 0
                    for _j in range(p1g, p2g + 1):
                        if _j < len(volume) and volume[_j] > 0.0:
                            _w1v += volume[_j]
                            _w1c += 1

                    if _w0c > 0 and _w1c > 0:
                        _avg_w0 = safe_divide(_w0v, _w0c)
                        _avg_w1 = safe_divide(_w1v, _w1c)
                        if _avg_w0 > 0.0:
                            _vol_ratio = safe_divide(_avg_w1, _avg_w0)
                            if _vol_ratio >= min_volume_confirmation:
                                volume_score = min(
                                    1.0,
                                    safe_divide(_vol_ratio, min_volume_confirmation * 1.5),
                                )
                            else:
                                volume_score = min(
                                    1.0,
                                    safe_divide(_vol_ratio, min_volume_confirmation),
                                )

                # Score final
                _components = np.array(
                    [wave_score, ratio_score, volume_score], dtype=np.float64
                )
                _weights = np.array(
                    [wave_weight, ratio_weight, volume_weight], dtype=np.float64
                )
                _score = calculate_weighted_score(_components, _weights)
                if _score > best_score:
                    best_score = _score

            if best_score > 0.0:
                signals[i] = best_score

        return signals

    @njit
    def detect_elliott_wave_3_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        min_extension_ratio: float,
        max_extension_ratio: float,
        min_volume_surge: float,
        extension_weight: float,
        volume_weight: float,
        momentum_weight: float,
    ) -> np.ndarray:
        """Détection Elliott Wave 3 — vague impulsive la plus forte du cycle.

        Caractéristiques (Prechter & Frost) :
        - Wave 3 = la plus longue et la plus rapide des vagues impulsives
        - Précédée de Wave 1 (impulsion) et Wave 2 (correction ≤ 100% de Wave 1)
        - Wave 3 doit dépasser l'extrême de Wave 1 dans la même direction
        - Volume et momentum supérieurs à Wave 1

        CORRECTIONS :
        P1 — Timing exact `indices[3] != i` remplacé par tolérance ±min_wave_length.
        P2 — Variable morte `waves_found` supprimée.
        P3 — List comprehensions remplacées par np.zeros + boucle for (Numba-safe).
        P6 — Seuils pivot uniformisés à [0.015, 0.030] comme Wave 1
             (anciens : [0.008, 0.025] incohérents avec Wave 1 et Wave 5).
        """
        n = len(open_prices)
        signals = np.zeros(n, dtype=np.float64)
        # P2 : waves_found supprimée (variable morte — jamais retournée)

        atr_period      = 14
        min_wave_length = 8
        lookback        = min_wave_length * 4  # 32 barres

        for i in range(lookback + atr_period, n):

            # ── 1. ATR adaptatif sur 14 barres ────────────────────────────────
            _atr_start = i - atr_period + 1
            _atr_sum   = 0.0
            _atr_cnt   = 0

            for _k in range(_atr_start, i + 1):
                _tr = high_prices[_k] - low_prices[_k]
                _pc = close_prices[_k - 1]
                if _pc > 0.0:
                    _hi_pc = abs(high_prices[_k] - _pc)
                    _lo_pc = abs(low_prices[_k]  - _pc)
                    if _hi_pc > _tr:
                        _tr = _hi_pc
                    if _lo_pc > _tr:
                        _tr = _lo_pc
                if _tr > 0.0:
                    _atr_sum += _tr
                    _atr_cnt += 1

            _ref = close_prices[i]
            if _ref <= 0.0:
                continue

            # ── 2. Seuil pivot adaptatif [0.015, 0.030] ──────────────────────
            # P6 : uniformisé avec Wave 1 (était [0.008, 0.025])
            if _atr_cnt > 0:
                _raw_thr   = (_atr_sum / _atr_cnt) / _ref
                _pivot_thr = _raw_thr
                if _pivot_thr < 0.015:
                    _pivot_thr = 0.015
                if _pivot_thr > 0.030:
                    _pivot_thr = 0.030
            else:
                _pivot_thr = 0.015  # P6 : fallback uniformisé (était 0.012)

            # ── 3. Recherche des pivots ───────────────────────────────────────
            start_idx = i - lookback
            pivots_idx, pivots_price, pivots_type = find_pivots_simple(
                close_prices[start_idx : i + 1], _pivot_thr, 5
            )

            if len(pivots_idx) < 4:
                continue

            # ── 4. Évaluation des quadruplettes de pivots ─────────────────────
            for wave_start in range(len(pivots_idx) - 3):
                if wave_start + 3 >= len(pivots_idx):
                    continue

                # P3 : np.zeros + boucle for (remplace list comprehension)
                indices = np.zeros(4, dtype=np.int64)
                prices  = np.zeros(4, dtype=np.float64)
                types   = np.zeros(4, dtype=np.int64)
                for _j in range(4):
                    indices[_j] = pivots_idx[wave_start + _j] + start_idx
                    prices[_j]  = pivots_price[wave_start + _j]
                    types[_j]   = pivots_type[wave_start + _j]

                # Validation données
                all_valid = True
                for _j in range(4):
                    if prices[_j] <= 0:
                        all_valid = False
                        break

                indices_valid = True
                for _j in range(3):
                    if indices[_j] >= indices[_j + 1]:
                        indices_valid = False
                        break

                if not all_valid or not indices_valid:
                    continue

                # Alternance obligatoire : 1,-1,1,-1 ou -1,1,-1,1
                alternation_valid = True
                for _k in range(1, 4):
                    if types[_k] == types[_k - 1]:
                        alternation_valid = False
                        break
                if not alternation_valid:
                    continue

                # Direction : Wave 3 doit dépasser l'extrême de Wave 1
                if types[0] == -1 and types[3] == 1:
                    if prices[3] <= prices[1]:   # Haussier : W3 high > W1 high
                        continue
                elif types[0] == 1 and types[3] == -1:
                    if prices[3] >= prices[1]:   # Baissier : W3 low < W1 low
                        continue
                else:
                    continue

                # Amplitudes
                wave1 = abs(prices[1] - prices[0])
                wave2 = abs(prices[2] - prices[1])
                wave3 = abs(prices[3] - prices[2])

                if wave1 <= 0 or wave3 <= 0:
                    continue

                # Règle absolue Prechter : Wave 2 ne peut retracer > 100% de Wave 1
                if safe_divide(wave2, wave1) >= 1.0:
                    continue

                # Ratio Wave3/Wave1
                extension_ratio = safe_divide(wave3, wave1)
                if extension_ratio < min_extension_ratio or extension_ratio > max_extension_ratio:
                    continue

                # P1 : tolérance temporelle ±min_wave_length
                # L'ancien code : `if indices[3] != i: continue` → zéro détection.
                if i - indices[3] > min_wave_length:
                    continue

                # ── Scoring ───────────────────────────────────────────────────

                # Score extension — Wave 3 plus forte que Wave 1
                extension_score = min(
                    1.0, safe_divide(extension_ratio, min_extension_ratio)
                )

                # Score momentum — Wave 3 plus rapide que Wave 1 (vitesse = amplitude/durée)
                momentum_score = 1.0
                wave3_length = indices[3] - indices[2]
                wave1_length = indices[1] - indices[0]
                if wave3_length > 0 and wave1_length > 0:
                    wave3_speed = safe_divide(wave3, float(wave3_length))
                    wave1_speed = safe_divide(wave1, float(wave1_length))
                    if wave1_speed > 0:
                        speed_ratio    = safe_divide(wave3_speed, wave1_speed)
                        momentum_score = min(1.0, safe_divide(speed_ratio, 1.5))

                # Score volume — Wave 3 doit avoir plus de volume que Wave 1
                volume_score = 0.5  # neutre si données insuffisantes
                if len(volume) > indices[3]:
                    wave3_vol = 0.0
                    wave3_cnt = 0
                    for _j in range(indices[2], indices[3] + 1):
                        if _j < len(volume) and volume[_j] > 0:
                            wave3_vol += volume[_j]
                            wave3_cnt += 1

                    wave1_vol = 0.0
                    wave1_cnt = 0
                    for _j in range(indices[0], indices[1] + 1):
                        if _j < len(volume) and volume[_j] > 0:
                            wave1_vol += volume[_j]
                            wave1_cnt += 1

                    if wave3_cnt > 0 and wave1_cnt > 0:
                        wave3_avg = safe_divide(wave3_vol, float(wave3_cnt))
                        wave1_avg = safe_divide(wave1_vol, float(wave1_cnt))
                        vol_ratio = safe_divide(wave3_avg, wave1_avg)
                        if vol_ratio >= min_volume_surge:
                            volume_score = min(
                                1.0,
                                safe_divide(vol_ratio, min_volume_surge * 1.5),
                            )
                        else:
                            volume_score = min(1.0, safe_divide(vol_ratio, min_volume_surge))

                components = np.array(
                    [extension_score, volume_score, momentum_score], dtype=np.float64
                )
                weights = np.array(
                    [extension_weight, volume_weight, momentum_weight], dtype=np.float64
                )
                pattern_score = calculate_weighted_score(components, weights)
                signals[i] = max(signals[i], pattern_score)

        return signals

    @njit
    def detect_elliott_wave_5_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        min_divergence_ratio: float,
        max_extension_ratio: float,
        min_volume_decline: float,
        divergence_weight: float,
        extension_weight: float,
        volume_weight: float,
    ) -> np.ndarray:
        """Détection Elliott Wave 5 — dernière vague impulsive du cycle.

        Caractéristiques (Prechter & Frost) :
        - Wave 5 = vague finale plus courte/lente que Wave 3 (divergence)
        - Volume décroissant vs Wave 3 (signe d'épuisement)
        - Wave 5 doit dépasser l'extrême de Wave 3 dans la même direction

        CORRECTIONS :
        P1 — Timing exact `indices[5] != i` remplacé par tolérance ±min_wave_length.
        P3 — List comprehensions remplacées par np.zeros + boucle for (Numba-safe).
        P6 — Seuils pivot uniformisés à [0.015, 0.030] comme Wave 1
             (anciens : [0.008, 0.025]).
        """
        n = len(open_prices)
        signals = np.zeros(n, dtype=np.float64)

        atr_period      = 14
        min_wave_length = 8
        lookback        = min_wave_length * 6  # 48 barres

        for i in range(lookback + atr_period, n):

            # ── 1. ATR adaptatif sur 14 barres ────────────────────────────────
            _atr_start = i - atr_period + 1
            _atr_sum   = 0.0
            _atr_cnt   = 0

            for _k in range(_atr_start, i + 1):
                _tr = high_prices[_k] - low_prices[_k]
                _pc = close_prices[_k - 1]
                if _pc > 0.0:
                    _hi_pc = abs(high_prices[_k] - _pc)
                    _lo_pc = abs(low_prices[_k]  - _pc)
                    if _hi_pc > _tr:
                        _tr = _hi_pc
                    if _lo_pc > _tr:
                        _tr = _lo_pc
                if _tr > 0.0:
                    _atr_sum += _tr
                    _atr_cnt += 1

            _ref = close_prices[i]
            if _ref <= 0.0:
                continue

            # ── 2. Seuil pivot adaptatif [0.015, 0.030] ──────────────────────
            # P6 : uniformisé avec Wave 1 et Wave 3 (était [0.008, 0.025])
            if _atr_cnt > 0:
                _raw_thr   = (_atr_sum / _atr_cnt) / _ref
                _pivot_thr = _raw_thr
                if _pivot_thr < 0.015:
                    _pivot_thr = 0.015
                if _pivot_thr > 0.030:
                    _pivot_thr = 0.030
            else:
                _pivot_thr = 0.015  # P6 : fallback uniformisé (était 0.012)

            # ── 3. Recherche des pivots ───────────────────────────────────────
            start_idx = i - lookback
            pivots_idx, pivots_price, pivots_type = find_pivots_simple(
                close_prices[start_idx : i + 1], _pivot_thr, 5
            )

            if len(pivots_idx) < 6:
                continue

            # ── 4. Évaluation des sextuplettes de pivots ──────────────────────
            for wave_start in range(len(pivots_idx) - 5):
                if wave_start + 5 >= len(pivots_idx):
                    continue

                # P3 : np.zeros + boucle for (remplace list comprehension)
                indices = np.zeros(6, dtype=np.int64)
                prices  = np.zeros(6, dtype=np.float64)
                types   = np.zeros(6, dtype=np.int64)
                for _j in range(6):
                    indices[_j] = pivots_idx[wave_start + _j] + start_idx
                    prices[_j]  = pivots_price[wave_start + _j]
                    types[_j]   = pivots_type[wave_start + _j]

                # Validation données
                all_valid = True
                for _j in range(6):
                    if prices[_j] <= 0:
                        all_valid = False
                        break

                indices_valid = True
                for _j in range(5):
                    if indices[_j] >= indices[_j + 1]:
                        indices_valid = False
                        break

                if not all_valid or not indices_valid:
                    continue

                # Alternance obligatoire sur les 6 pivots
                alternation_valid = True
                for _k in range(1, 6):
                    if types[_k] == types[_k - 1]:
                        alternation_valid = False
                        break
                if not alternation_valid:
                    continue

                # Cohérence directionnelle : Wave 5 doit dépasser Wave 3
                if types[0] == -1 and types[5] == 1:
                    if prices[5] <= prices[3]:   # Haussier : W5 high > W3 high
                        continue
                elif types[0] == 1 and types[5] == -1:
                    if prices[5] >= prices[3]:   # Baissier : W5 low < W3 low
                        continue
                else:
                    continue

                # Amplitudes des 3 vagues impulsives
                wave1 = abs(prices[1] - prices[0])
                wave3 = abs(prices[3] - prices[2])
                wave5 = abs(prices[5] - prices[4])

                if wave1 <= 0 or wave3 <= 0 or wave5 <= 0:
                    continue

                # Wave 5 ne doit pas dépasser max_extension_ratio vs Wave 1 ET Wave 3
                extension_ratio_vs_1 = safe_divide(wave5, wave1)
                extension_ratio_vs_3 = safe_divide(wave5, wave3)

                if (
                    extension_ratio_vs_1 > max_extension_ratio
                    or extension_ratio_vs_3 > max_extension_ratio
                ):
                    continue

                # Divergence : Wave 5 doit être plus courte que Wave 3
                divergence_ratio = safe_divide((wave3 - wave5), wave3)
                if divergence_ratio < min_divergence_ratio:
                    continue

                # P1 : tolérance temporelle ±min_wave_length
                # L'ancien code : `if indices[5] != i: continue` → zéro détection.
                if i - indices[5] > min_wave_length:
                    continue

                # ── Scoring ───────────────────────────────────────────────────

                # Score divergence — plus Wave 5 est courte vs Wave 3, meilleur le signal
                divergence_score = min(
                    1.0, safe_divide(divergence_ratio, min_divergence_ratio * 2)
                )

                # Score extension — Wave 5 bien en dessous du max autorisé
                extension_score = max(
                    0.0,
                    1.0 - max(
                        safe_divide(extension_ratio_vs_1, max_extension_ratio),
                        safe_divide(extension_ratio_vs_3, max_extension_ratio),
                    ),
                )

                # Score volume — Wave 5 doit avoir MOINS de volume que Wave 3 (épuisement)
                volume_score = 1.0  # 1.0 si pas de données (pas pénalisé)
                if len(volume) > indices[5]:
                    wave5_vol = 0.0
                    wave5_cnt = 0
                    for _j in range(indices[4], indices[5] + 1):
                        if _j < len(volume) and volume[_j] > 0:
                            wave5_vol += volume[_j]
                            wave5_cnt += 1

                    wave3_vol = 0.0
                    wave3_cnt = 0
                    for _j in range(indices[2], indices[3] + 1):
                        if _j < len(volume) and volume[_j] > 0:
                            wave3_vol += volume[_j]
                            wave3_cnt += 1

                    if wave5_cnt > 0 and wave3_cnt > 0:
                        wave5_avg = safe_divide(wave5_vol, float(wave5_cnt))
                        wave3_avg = safe_divide(wave3_vol, float(wave3_cnt))
                        vol_ratio = safe_divide(wave5_avg, wave3_avg)

                        # vol_ratio < min_volume_decline → volume baisse bien → score 1.0
                        # vol_ratio > min_volume_decline → volume ne baisse pas assez → pénalisé
                        if vol_ratio <= min_volume_decline:
                            volume_score = 1.0
                        else:
                            volume_score = safe_divide(min_volume_decline, vol_ratio)

                components = np.array(
                    [divergence_score, extension_score, volume_score], dtype=np.float64
                )
                weights = np.array(
                    [divergence_weight, extension_weight, volume_weight], dtype=np.float64
                )
                pattern_score = calculate_weighted_score(components, weights)
                signals[i] = max(signals[i], pattern_score)

        return signals

    @njit
    def detect_fibonacci_retracement_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        min_swing_length: int,
        retracement_tolerance: float,
        min_bounce_strength: float,
        swing_weight: float,
        retracement_weight: float,
        bounce_weight: float,
    ) -> np.ndarray:
        """Détection Fibonacci Retracement.

        Recherche d'un swing AB + retracement BC proche d'un niveau Fibonacci
        + rebond CD confirmant le niveau.

        CORRECTIONS :
        P4 — Lookahead bias : la boucle de détection du rebond était bornée à
             `retr_end_idx + min_swing_length`, ce qui pouvait dépasser `i`
             si le point de retracement était proche de la barre courante.
             Correction : `min(..., i)` borne le regard au passé uniquement.
        P5 — Liste Python `fib_levels` recréée à chaque itération interne.
             Remplacée par np.array défini une seule fois hors boucle.
        """
        signals = np.zeros(len(open_prices), dtype=np.float64)

        # P5 : np.array hors boucle (évite réallocation à chaque itération)
        fib_levels = np.array([0.236, 0.382, 0.5, 0.618, 0.786], dtype=np.float64)

        for i in range(min_swing_length * 2, len(open_prices)):
            start_idx = i - min_swing_length * 2
            min_dist  = max(1, min_swing_length // 2)

            pivots_idx, pivots_price, pivots_type = find_pivots_simple(
                close_prices[start_idx : i + 1], 0.02, min_dist
            )

            if len(pivots_idx) < 3:
                continue

            for swing_start in range(len(pivots_idx) - 2):
                if swing_start + 2 >= len(pivots_idx):
                    continue

                swing_start_idx = np.int64(pivots_idx[swing_start]     + start_idx)
                swing_end_idx   = np.int64(pivots_idx[swing_start + 1] + start_idx)
                retr_end_idx    = np.int64(pivots_idx[swing_start + 2] + start_idx)

                if (
                    swing_start_idx < 0 or swing_start_idx >= len(close_prices)
                    or swing_end_idx   < 0 or swing_end_idx   >= len(close_prices)
                    or retr_end_idx    < 0 or retr_end_idx    >= len(close_prices)
                ):
                    continue

                swing_start_price = pivots_price[swing_start]
                swing_end_price   = pivots_price[swing_start + 1]
                retr_end_price    = pivots_price[swing_start + 2]

                if (
                    swing_start_price <= 0
                    or swing_end_price <= 0
                    or retr_end_price  <= 0
                    or swing_start_idx >= swing_end_idx
                    or swing_end_idx   >= retr_end_idx
                ):
                    continue

                swing_size       = abs(swing_end_price - swing_start_price)
                retracement_size = abs(retr_end_price  - swing_end_price)

                if swing_size <= 0:
                    continue

                retracement_ratio = safe_divide(retracement_size, swing_size)

                # Proximité au niveau Fibonacci le plus proche
                closest_fib_distance = 999.0
                for _f in range(len(fib_levels)):
                    _dist = abs(retracement_ratio - fib_levels[_f])
                    if _dist < closest_fib_distance:
                        closest_fib_distance = _dist

                if closest_fib_distance > retracement_tolerance:
                    continue

                # Détection du rebond après le point de retracement
                bounce_detected = False
                bounce_strength = 0.0

                # P4 : borner à i pour éviter le lookahead
                # L'ancien code : min(retr_end_idx + min_swing_length, len(close_prices))
                # pouvait regarder au-delà de i si retr_end_idx était proche de i.
                max_idx = np.int64(min(retr_end_idx + min_swing_length, i))
                start_j = np.int64(retr_end_idx + 1)

                for j in range(start_j, max_idx):
                    idx_j    = np.int64(j)
                    idx_retr = np.int64(retr_end_idx)

                    if idx_j >= len(close_prices) or idx_retr >= len(close_prices):
                        break
                    if close_prices[idx_j] <= 0 or close_prices[idx_retr] <= 0:
                        continue

                    move = safe_divide(
                        (close_prices[idx_j] - close_prices[idx_retr]),
                        close_prices[idx_retr],
                    )

                    if abs(move) >= min_bounce_strength:
                        bounce_detected = True
                        bounce_strength = abs(move)
                        break

                # Score swing — normalisé en % du prix (cross-asset)
                if swing_start_price > 0:
                    swing_pct   = safe_divide(swing_size, swing_start_price)
                    swing_score = min(1.0, safe_divide(swing_pct, 0.02))  # 2% = score 1.0
                else:
                    swing_score = 0.0

                retr_tolerance_safe  = max(retracement_tolerance, 0.001)
                retracement_score = max(
                    0.0,
                    1.0 - safe_divide(closest_fib_distance, retr_tolerance_safe),
                )

                bounce_score = 0.5  # neutre si aucun rebond détecté
                if bounce_detected:
                    min_bounce_safe = max(min_bounce_strength * 2, 0.001)
                    bounce_score = min(
                        1.0, safe_divide(bounce_strength, min_bounce_safe)
                    )

                components = np.array(
                    [swing_score, retracement_score, bounce_score], dtype=np.float64
                )
                weights = np.array(
                    [swing_weight, retracement_weight, bounce_weight], dtype=np.float64
                )
                pattern_score = calculate_weighted_score(components, weights)
                signals[i] = max(signals[i], pattern_score)

        return signals


    @njit
    def detect_fibonacci_extension_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        min_swing_length: int,
        extension_tolerance: float,
        min_extension_strength: float,
        swing_weight: float,
        extension_weight: float,
        confirmation_weight: float,
    ) -> np.ndarray:
        """Détection Fibonacci Extension.

        Structure AB (swing) + BC (retracement Fibonacci) + CD (extension Fibonacci).

        CORRECTIONS :
        P5 — Liste Python `ext_levels` recréée à chaque itération interne.
             Remplacée par np.array défini une seule fois hors boucle.

        Corrections précédentes conservées :
        - FIX 1 : suppression de closest_ext_level (dead variable).
        - FIX 2 : validation BC/AB — ratio doit être un retracement Fibonacci
                  (0.236 à 0.886). Filtre les structures aberrantes.
        """
        signals = np.zeros(len(open_prices), dtype=np.float64)

        # P5 : np.array hors boucle (évite réallocation à chaque itération)
        ext_levels = np.array([1.272, 1.414, 1.618, 2.0, 2.618], dtype=np.float64)

        for i in range(min_swing_length * 3, len(open_prices)):
            start_idx = i - min_swing_length * 3
            min_dist  = max(1, min_swing_length // 2)

            pivots_idx, pivots_price, pivots_type = find_pivots_simple(
                close_prices[start_idx : i + 1], 0.02, min_dist
            )

            if len(pivots_idx) < 4:
                continue

            for pattern_start in range(len(pivots_idx) - 3):
                if pattern_start + 3 >= len(pivots_idx):
                    continue

                point_a_price = pivots_price[pattern_start]
                point_b_price = pivots_price[pattern_start + 1]
                point_c_price = pivots_price[pattern_start + 2]
                point_d_price = pivots_price[pattern_start + 3]

                if (
                    point_a_price <= 0
                    or point_b_price <= 0
                    or point_c_price <= 0
                    or point_d_price <= 0
                ):
                    continue

                swing_ab = abs(point_b_price - point_a_price)
                swing_bc = abs(point_c_price - point_b_price)
                swing_cd = abs(point_d_price - point_c_price)

                if swing_ab <= 0 or swing_cd <= 0:
                    continue

                # Validation BC/AB — doit être un retracement Fibonacci [0.236, 0.886]
                # Filtre les structures où la correction B→C est aberrante
                bc_ab_ratio = safe_divide(swing_bc, swing_ab)
                if bc_ab_ratio < 0.236 or bc_ab_ratio > 0.886:
                    continue

                # Ratio d'extension CD/AB
                extension_ratio = safe_divide(swing_cd, swing_ab)

                # Proximité au niveau d'extension Fibonacci le plus proche
                closest_ext_distance = 999.0
                for _f in range(len(ext_levels)):
                    _dist = abs(extension_ratio - ext_levels[_f])
                    if _dist < closest_ext_distance:
                        closest_ext_distance = _dist

                if closest_ext_distance > extension_tolerance:
                    continue

                # Force de l'extension — amplitude CD en % du prix C
                if point_c_price > 0:
                    extension_strength = safe_divide(
                        abs(point_d_price - point_c_price), point_c_price
                    )
                    if extension_strength < min_extension_strength:
                        continue
                else:
                    continue

                # ── Scoring ───────────────────────────────────────────────────

                # Score swing — normalisé en % du prix (cross-asset)
                swing_pct   = safe_divide(swing_ab, point_a_price) if point_a_price > 0 else 0.0
                swing_score = min(1.0, safe_divide(swing_pct, 0.02))  # 2% = score 1.0

                ext_tolerance_safe = max(extension_tolerance, 0.001)
                extension_score = max(
                    0.0,
                    1.0 - safe_divide(closest_ext_distance, ext_tolerance_safe),
                )

                min_ext_safe = max(min_extension_strength * 2, 0.001)
                confirmation_score = min(
                    1.0, safe_divide(extension_strength, min_ext_safe)
                )

                components = np.array(
                    [swing_score, extension_score, confirmation_score], dtype=np.float64
                )
                weights = np.array(
                    [swing_weight, extension_weight, confirmation_weight], dtype=np.float64
                )
                pattern_score = calculate_weighted_score(components, weights)
                signals[i] = max(signals[i], pattern_score)

        return signals

    @njit
    def detect_symmetrical_triangle_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        min_pattern_length: int,
        max_angle_diff: float,
        min_convergence: float,
        upper_weight: float,
        lower_weight: float,
        convergence_weight: float,
    ) -> np.ndarray:
        """VERSION CORRIGÉE 2.0 - Détection de Symmetrical Triangle avec Régression Linéaire."""
        signals = np.zeros(len(open_prices), dtype=np.float64)

        for i in range(min_pattern_length, len(open_prices)):
            start_idx = i - min_pattern_length
            window_highs = high_prices[start_idx : i + 1]
            window_lows = low_prices[start_idx : i + 1]

            # AUDIT FIX C-B8-7a: ATR-based pivot threshold au lieu de 0.015 hardcodé
            atr_sum = 0.0
            atr_count = 0
            for k in range(start_idx, i + 1):
                atr_sum += high_prices[k] - low_prices[k]
                atr_count += 1
            local_atr = atr_sum / max(atr_count, 1)
            mid_price = (high_prices[i] + low_prices[i]) / 2.0
            pivot_threshold = local_atr / mid_price if mid_price > 0 else 0.015
            pivot_threshold = max(0.008, min(0.025, pivot_threshold))

            # Trouver les pivots dans la fenêtre
            high_pivots_idx, high_pivots_price, _ = find_pivots_simple(
                window_highs, pivot_threshold, 3
            )
            low_pivots_idx, low_pivots_price, _ = find_pivots_simple(
                window_lows, pivot_threshold, 3
            )

            if len(high_pivots_idx) < 2 or len(low_pivots_idx) < 2:
                continue

            # NOUVEAU: Régression linéaire sur tous les pivots pour des lignes robustes
            resistance_slope = calculate_linear_regression_slope(
                high_pivots_idx, high_pivots_price
            )
            support_slope = calculate_linear_regression_slope(
                low_pivots_idx, low_pivots_price
            )

            # Conditions: résistance descendante, support ascendant
            if resistance_slope >= 0 or support_slope <= 0:
                continue

            # Condition de symétrie (pentes opposées mais de magnitude similaire)
            slope_divergence = abs(abs(resistance_slope) - abs(support_slope))
            if slope_divergence > max_angle_diff:
                continue

            # AUDIT FIX C-B8-7c: convergence sur moyenne 3 barres au lieu de barre unique
            n_avg = min(3, len(window_highs))
            start_range = 0.0
            end_range = 0.0
            for k in range(n_avg):
                start_range += window_highs[k] - window_lows[k]
                end_range += window_highs[-(k + 1)] - window_lows[-(k + 1)]
            start_range /= max(n_avg, 1)
            end_range /= max(n_avg, 1)

            if start_range <= 0:
                continue

            convergence_ratio = 1.0 - safe_divide(end_range, start_range)
            if convergence_ratio < min_convergence:
                continue

            # Scoring
            convergence_score = min(
                1.0, safe_divide(convergence_ratio, min_convergence * 2)
            )
            symmetry_score = 1.0 - safe_divide(slope_divergence, max_angle_diff)

            # AUDIT FIX C-B8-7b: utiliser les 3 poids (upper, lower, convergence) au lieu de 0.5 hardcodé
            # upper_weight → convergence_score, lower_weight → symmetry_score, convergence_weight reste
            components = np.array(
                [convergence_score, symmetry_score, convergence_ratio]
            )
            weights = np.array([upper_weight, lower_weight, convergence_weight])

            signals[i] = calculate_weighted_score(components, weights)

        return signals

    @njit
    def detect_island_bottom_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        min_gap_ratio: float,
        max_island_length: int,
        min_volume_surge: float,
        gap_weight: float,
        isolation_weight: float,
        volume_weight: float,
    ) -> np.ndarray:
        """VERSION CORRIGÉE - Détection du pattern Island Bottom.

        CORRECTIONS APPORTÉES:
        1. Gestion des erreurs silencieuses avec validation stricte
        2. Système de scoring pondéré au lieu de conditions binaires
        3. Validation des gaps et isolation
        4. Paramètres plus réalistes

        PATTERN:
        - Gap baissier qui isole une zone de prix
        - Zone isolée ("île") de quelques barres
        - Gap haussier qui complète l'isolation
        - Volume élevé sur les gaps
        - Signal de retournement haussier

        Args:
            Prix OHLC et volume
            min_gap_ratio: Ratio minimum des gaps (0.01 recommandé)
            max_island_length: Longueur maximum de l'île (5 recommandé)
            min_volume_surge: Surge de volume minimum (2.0 recommandé)
            Poids pour le scoring pondéré
            close_prices: TODO: documenter.
            gap_weight: TODO: documenter.
            high_prices: TODO: documenter.
            isolation_weight: TODO: documenter.
            low_prices: TODO: documenter.
            open_prices: TODO: documenter.
            volume: TODO: documenter.
            volume_weight: TODO: documenter.

        Returns:
            Array des scores de détection (0.0 à 1.0)
        """
        signals = np.zeros(len(open_prices), dtype=np.float64)
        # AUDIT FIX C-B7-1: supprimé islands_found (variable morte)

        for i in range(3, len(open_prices) - 1):
            # === RECHERCHE DU GAP BAISSIER (DÉBUT DE L'ÎLE) ===
            gap_down_detected = False
            gap_down_idx = -1
            gap_down_size = 0.0

            for j in range(max(1, i - max_island_length - 2), i):
                if (
                    j < len(high_prices)
                    and j + 1 < len(low_prices)
                    and close_prices[j] > 0
                    and open_prices[j + 1] < high_prices[j]
                ):
                    gap_size = high_prices[j] - open_prices[j + 1]
                    gap_ratio = safe_divide(gap_size, close_prices[j])

                    if gap_ratio >= min_gap_ratio:
                        gap_down_detected = True
                        gap_down_idx = j
                        gap_down_size = gap_size
                        break

            if not gap_down_detected:
                continue

            # === RECHERCHE DU GAP HAUSSIER (FIN DE L'ÎLE) ===
            gap_up_detected = False
            gap_up_idx = -1
            gap_up_size = 0.0

            # Chercher le gap haussier après le gap baissier
            for j in range(i, min(i + max_island_length + 2, len(open_prices) - 1)):
                if (
                    j < len(low_prices)
                    and j + 1 < len(open_prices)
                    and close_prices[j] > 0
                    and open_prices[j + 1] > high_prices[j]
                ):
                    gap_size = open_prices[j + 1] - high_prices[j]
                    gap_ratio = safe_divide(gap_size, close_prices[j])

                    if gap_ratio >= min_gap_ratio:
                        gap_up_detected = True
                        gap_up_idx = j
                        gap_up_size = gap_size
                        break

            if not gap_up_detected:
                continue

            # === VALIDATION DE L'ÎLE ===
            island_length = gap_up_idx - gap_down_idx - 1
            if island_length <= 0 or island_length > max_island_length:
                continue

            # === VALIDATION DE L'ISOLATION ===
            # Vérifier que l'île est bien isolée (prix dans les gaps)
            island_isolated = True
            island_high = 0.0
            island_low = 999999.0

            for j in range(gap_down_idx + 1, gap_up_idx + 1):
                if j < len(high_prices) and j < len(low_prices):
                    island_high = max(island_high, high_prices[j])
                    island_low = min(island_low, low_prices[j])

            # L'île doit être en dessous du niveau avant le gap down
            # et au-dessus du niveau après le gap up
            if (
                gap_down_idx >= 0
                and gap_down_idx < len(low_prices)
                and gap_up_idx >= 0
                and gap_up_idx < len(high_prices)
            ):
                if (
                    island_high > low_prices[gap_down_idx]
                    or island_low < high_prices[gap_up_idx]
                ):
                    island_isolated = False

            if not island_isolated:
                continue

            # === ANALYSE DU VOLUME ===
            volume_score = 1.0
            if len(volume) > gap_up_idx:
                # Volume sur les gaps
                gap_down_volume = (
                    volume[gap_down_idx + 1] if gap_down_idx + 1 < len(volume) else 0
                )
                gap_up_volume = (
                    volume[gap_up_idx + 1] if gap_up_idx + 1 < len(volume) else 0
                )

                # Volume moyen avant le pattern
                avg_volume = 0.0
                count = 0
                for j in range(max(0, gap_down_idx - 5), gap_down_idx):
                    if j < len(volume) and volume[j] > 0:
                        avg_volume += volume[j]
                        count += 1

                if count > 0 and avg_volume > 0:
                    avg_volume = safe_divide(avg_volume, count)

                    # Score basé sur le volume des gaps
                    gap_down_ratio = safe_divide(gap_down_volume, avg_volume)
                    gap_up_ratio = safe_divide(gap_up_volume, avg_volume)
                    avg_gap_volume_ratio = (gap_down_ratio + gap_up_ratio) / 2.0

                    volume_score = min(
                        1.0, safe_divide(avg_gap_volume_ratio, min_volume_surge)
                    )

            # === CALCUL DU SCORE PONDÉRÉ ===

            # Score des gaps (plus ils sont grands, mieux c'est)
            gap_score = (
                safe_divide(gap_down_size, close_prices[gap_down_idx] * min_gap_ratio)
                + safe_divide(gap_up_size, close_prices[gap_up_idx] * min_gap_ratio)
            ) / 2.0
            gap_score = min(1.0, gap_score)

            # Score d'isolation (plus l'île est petite, mieux c'est)
            isolation_score = max(
                0.0, 1.0 - safe_divide(island_length, max_island_length)
            )

            # Combinaison pondérée
            components = np.array([gap_score, isolation_score, volume_score])
            weights = np.array([gap_weight, isolation_weight, volume_weight])

            pattern_score = calculate_weighted_score(components, weights)
            signals[i] = pattern_score

        return signals

    @njit
    def detect_three_black_crows_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        min_body_ratio: float,
        max_shadow_ratio: float,
        min_consecutive_decline: float,
        body_weight: float,
        shadow_weight: float,
        decline_weight: float,
    ) -> np.ndarray:
        """Détection Three Black Crows.

        FIX: prev_body_mid utilisé comme borne basse de curr_open_in_body.
        Nison définit que chaque ouverture doit se situer DANS le corps
        de la bougie précédente. L'ancienne borne `close * 0.95` permettait
        une ouverture 5% sous la clôture baissière, c'est-à-dire hors du corps.
        La correction utilise prev_body_mid (mi-corps) comme borne basse,
        ce qui garantit une ouverture dans la moitié haute du corps précédent
        — condition classique des Three Black Crows de qualité.
        """
        signals = np.zeros(len(open_prices), dtype=np.float64)
        # AUDIT FIX C-B7-2: supprimé crows_found (variable morte)

        for i in range(2, len(open_prices)):
            candles_valid = True
            candle_scores = np.zeros(3, dtype=np.float64)
            shadow_scores = np.zeros(
                3, dtype=np.float64
            )  # AUDIT FIX C-B7-2: séparé du body

            for j in range(3):
                idx = i - 2 + j

                if (
                    idx < 0
                    or open_prices[idx] <= 0
                    or high_prices[idx] <= 0
                    or low_prices[idx] <= 0
                    or close_prices[idx] <= 0
                ):
                    candles_valid = False
                    break

                body_size = abs(close_prices[idx] - open_prices[idx])
                total_range = high_prices[idx] - low_prices[idx]
                upper_shadow = high_prices[idx] - max(
                    open_prices[idx], close_prices[idx]
                )
                lower_shadow = (
                    min(open_prices[idx], close_prices[idx]) - low_prices[idx]
                )

                if total_range <= 0:
                    candles_valid = False
                    break

                if close_prices[idx] >= open_prices[idx]:
                    candles_valid = False
                    break

                body_ratio = safe_divide(body_size, total_range)
                body_score = min(1.0, safe_divide(body_ratio, min_body_ratio))

                shadow_ratio = safe_divide((upper_shadow + lower_shadow), total_range)
                shadow_score = max(
                    0.0, 1.0 - safe_divide(shadow_ratio, max_shadow_ratio)
                )

                candle_scores[j] = (body_score + shadow_score) / 2.0

            if not candles_valid:
                continue

            progression_valid = True
            decline_scores = np.zeros(2, dtype=np.float64)

            for j in range(2):
                idx1 = i - 2 + j
                idx2 = i - 1 + j

                # FIX: prev_body_mid utilisé comme borne basse
                # Pour Three Black Crows, l'ouverture de idx2 doit être
                # dans la MOITIÉ HAUTE du corps de idx1.
                # Corps baissier = [close[idx1] (bas), open[idx1] (haut)].
                # prev_body_mid = milieu du corps → borne basse effective.
                prev_body_mid = (open_prices[idx1] + close_prices[idx1]) / 2.0
                curr_open_in_body = (
                    open_prices[idx2]
                    <= open_prices[idx1]  # ne dépasse pas le haut du corps
                    and open_prices[idx2]
                    >= prev_body_mid  # dans la moitié HAUTE du corps
                )

                if not curr_open_in_body:
                    progression_valid = False
                    break

                _atr_start = max(1, idx1 - 13)
                _atr_sum = 0.0
                _atr_cnt = 0
                for _aj in range(_atr_start, idx1 + 1):
                    _tr = max(
                        high_prices[_aj] - low_prices[_aj],
                        abs(high_prices[_aj] - close_prices[_aj - 1]),
                        abs(low_prices[_aj] - close_prices[_aj - 1]),
                    )
                    _atr_sum += _tr
                    _atr_cnt += 1

                _atr_local = (
                    _atr_sum / _atr_cnt
                    if _atr_cnt > 0
                    else (high_prices[idx1] - low_prices[idx1])
                )
                if _atr_local <= 0:
                    _atr_local = abs(close_prices[idx1] - open_prices[idx1]) + 1e-10

                min_decline_atr = min_consecutive_decline * _atr_local
                decline = close_prices[idx1] - close_prices[idx2]

                if decline < min_decline_atr:
                    progression_valid = False
                    break

                decline_scores[j] = min(1.0, safe_divide(decline, min_decline_atr * 2))

            if not progression_valid:
                continue

            avg_body_score = (
                candle_scores[0] + candle_scores[1] + candle_scores[2]
            ) / 3.0
            # AUDIT FIX C-B7-2: avg_shadow_score calculé indépendamment
            avg_shadow_score = (
                shadow_scores[0] + shadow_scores[1] + shadow_scores[2]
            ) / 3.0
            avg_decline_score = (decline_scores[0] + decline_scores[1]) / 2.0

            components = np.array([avg_body_score, avg_shadow_score, avg_decline_score])
            weights = np.array([body_weight, shadow_weight, decline_weight])
            pattern_score = calculate_weighted_score(components, weights)
            signals[i] = pattern_score

        return signals

    @njit
    def detect_falling_wedge_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        min_pattern_length: int,
        min_convergence: float,
        max_angle_diff: float,
        min_volume_decline: float,
        angle_weight: float,
        volume_weight: float,
        convergence_weight: float,
    ) -> np.ndarray:
        """VERSION CORRIGÉE - Détection du pattern Falling Wedge."""
        signals = np.zeros(len(open_prices), dtype=np.float64)
        for i in range(min_pattern_length, len(open_prices)):
            start_idx = i - min_pattern_length
            window_highs = high_prices[start_idx : i + 1]
            window_lows = low_prices[start_idx : i + 1]

            # AUDIT FIX C-B7-6: ATR adaptatif au lieu du 0.015 hardcodé
            _fw_atr_sum = 0.0
            _fw_atr_cnt = 0
            for _k in range(max(1, start_idx), i + 1):
                _fw_tr = max(
                    high_prices[_k] - low_prices[_k],
                    abs(high_prices[_k] - close_prices[_k - 1]),
                    abs(low_prices[_k] - close_prices[_k - 1]),
                )
                _fw_atr_sum += _fw_tr
                _fw_atr_cnt += 1
            _fw_atr = _fw_atr_sum / _fw_atr_cnt if _fw_atr_cnt > 0 else 0.015
            _fw_ref = close_prices[i] if close_prices[i] > 0 else 1.0
            _raw_thr = _fw_atr / _fw_ref
            _pivot_thr = max(0.008, min(0.025, _raw_thr))

            # Utiliser la détection de pivots pour trouver les points des lignes de tendance
            high_pivots_idx, high_pivots_price, _ = find_pivots_simple(
                window_highs, _pivot_thr, 3
            )
            low_pivots_idx, low_pivots_price, _ = find_pivots_simple(
                window_lows, _pivot_thr, 3
            )

            if len(high_pivots_idx) < 2 or len(low_pivots_idx) < 2:
                continue

            resistance_slope = calculate_linear_regression_slope(
                high_pivots_idx, high_pivots_price
            )
            support_slope = calculate_linear_regression_slope(
                low_pivots_idx, low_pivots_price
            )

            # Les deux pentes doivent être descendantes
            if resistance_slope >= 0 or support_slope >= 0:
                continue

            # La ligne de support doit descendre plus vite (pente plus négative)
            if abs(support_slope) <= abs(resistance_slope):
                continue

            # Le ratio des pentes ne doit pas être trop extrême
            slope_ratio = safe_divide(abs(support_slope), abs(resistance_slope))
            if slope_ratio > (max_angle_diff * 10):  # Tolérance augmentée
                continue

            # Le volume devrait décliner
            volume_score = 1.0 - (
                safe_divide(
                    np.mean(volume[i - 5 : i]),
                    np.mean(volume[start_idx : start_idx + 5]),
                )
                if np.mean(volume[start_idx : start_idx + 5]) > 0
                else 0
            )

            # AUDIT FIX C-B7-6: convergence robuste sur moyenne 3 barres
            _start_range = 0.0
            for _k in range(min(3, min_pattern_length)):
                _start_range += high_prices[start_idx + _k] - low_prices[start_idx + _k]
            _start_range /= min(3.0, float(min_pattern_length))

            _end_range = 0.0
            for _k in range(min(3, min_pattern_length)):
                _end_range += high_prices[i - 2 + _k] - low_prices[i - 2 + _k]
            _end_range /= min(3.0, float(min_pattern_length))

            if _start_range <= 0:
                continue
            convergence_score = 1.0 - safe_divide(_end_range, _start_range)
            slope_score = 1.0 - safe_divide(
                abs(abs(support_slope) - abs(resistance_slope)), abs(support_slope)
            )

            components = np.array(
                [convergence_score, slope_score, max(0, volume_score)]
            )
            weights = np.array([convergence_weight, angle_weight, volume_weight])
            signals[i] = calculate_weighted_score(components, weights)

        return signals

    @njit
    def detect_kicking_bear_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        min_gap_ratio: float,
        min_body_ratio: float,
        min_volume_surge: float,
        gap_weight: float,
        body_weight: float,
        volume_weight: float,
    ) -> np.ndarray:
        """VERSION CORRIGÉE - Détection du pattern Kicking Bear.

        CORRECTIONS APPORTÉES:
        1. Conditions quasi-parfaites remplacées par tolérances réalistes
        2. Gestion des erreurs silencieuses avec validation stricte
        3. Système de scoring pondéré au lieu de conditions binaires
        4. Paramètres plus réalistes

        PATTERN:
        - Marubozu haussier suivi d'un gap baissier
        - Marubozu baissier après le gap
        - Volume élevé
        - Signal de continuation baissière forte

        Args:
            Prix OHLC et volume
            min_gap_ratio: Ratio minimum du gap (0.005 recommandé)
            min_body_ratio: Ratio minimum du corps (0.8 recommandé)
            min_volume_surge: Surge de volume minimum (1.5 recommandé)
            Poids pour le scoring pondéré
            body_weight: TODO: documenter.
            close_prices: TODO: documenter.
            gap_weight: TODO: documenter.
            high_prices: TODO: documenter.
            low_prices: TODO: documenter.
            open_prices: TODO: documenter.
            volume: TODO: documenter.
            volume_weight: TODO: documenter.

        Returns:
            Array des scores de détection (0.0 à 1.0)
        """
        signals = np.zeros(len(open_prices), dtype=np.float64)
        # AUDIT FIX C-B7-3: supprimé kicks_found (variable morte) dans kicking_bear

        for i in range(1, len(open_prices)):
            # === VALIDATION DES DONNÉES ===
            if (
                open_prices[i - 1] <= 0
                or high_prices[i - 1] <= 0
                or low_prices[i - 1] <= 0
                or close_prices[i - 1] <= 0
                or open_prices[i] <= 0
                or high_prices[i] <= 0
                or low_prices[i] <= 0
                or close_prices[i] <= 0
            ):
                continue

            # === VÉRIFICATION DU GAP BAISSIER ===
            gap_down = open_prices[i] < low_prices[i - 1]
            if not gap_down:
                continue

            gap_size = low_prices[i - 1] - open_prices[i]
            gap_ratio = safe_divide(gap_size, close_prices[i - 1])

            if gap_ratio < min_gap_ratio:
                continue

            # === VÉRIFICATION DU MARUBOZU HAUSSIER (CHANDELLE PRÉCÉDENTE) ===
            prev_body_size = abs(close_prices[i - 1] - open_prices[i - 1])
            prev_total_range = high_prices[i - 1] - low_prices[i - 1]
            prev_upper_shadow = high_prices[i - 1] - max(
                open_prices[i - 1], close_prices[i - 1]
            )
            prev_lower_shadow = (
                min(open_prices[i - 1], close_prices[i - 1]) - low_prices[i - 1]
            )

            if prev_total_range <= 0:
                continue

            # Doit être haussier
            if close_prices[i - 1] <= open_prices[i - 1]:
                continue

            # Évaluation du corps (ASSOUPLIE)
            prev_body_ratio = safe_divide(prev_body_size, prev_total_range)
            prev_body_score = min(1.0, safe_divide(prev_body_ratio, min_body_ratio))

            # Évaluation des ombres (doivent être petites)
            prev_shadow_ratio = safe_divide(
                (prev_upper_shadow + prev_lower_shadow), prev_total_range
            )
            prev_shadow_score = max(
                0.0, 1.0 - safe_divide(prev_shadow_ratio, 1.0 - min_body_ratio)
            )  # AUDIT FIX C-B7-3: max_shadow = 1 - min_body_ratio (cohérent Marubozu)

            prev_marubozu_score = (prev_body_score + prev_shadow_score) / 2.0

            # === VÉRIFICATION DU MARUBOZU BAISSIER (CHANDELLE ACTUELLE) ===
            curr_body_size = abs(close_prices[i] - open_prices[i])
            curr_total_range = high_prices[i] - low_prices[i]
            curr_upper_shadow = high_prices[i] - max(open_prices[i], close_prices[i])
            curr_lower_shadow = min(open_prices[i], close_prices[i]) - low_prices[i]

            if curr_total_range <= 0:
                continue

            # Doit être baissier
            if close_prices[i] >= open_prices[i]:
                continue

            # Évaluation du corps (ASSOUPLIE)
            curr_body_ratio = safe_divide(curr_body_size, curr_total_range)
            curr_body_score = min(1.0, safe_divide(curr_body_ratio, min_body_ratio))

            # Évaluation des ombres
            curr_shadow_ratio = safe_divide(
                (curr_upper_shadow + curr_lower_shadow), curr_total_range
            )
            curr_shadow_score = max(
                0.0, 1.0 - safe_divide(curr_shadow_ratio, 1.0 - min_body_ratio)
            )  # AUDIT FIX C-B7-3: max_shadow = 1 - min_body_ratio (cohérent Marubozu)

            curr_marubozu_score = (curr_body_score + curr_shadow_score) / 2.0

            # === ANALYSE DU VOLUME ===
            volume_score = 1.0
            if len(volume) > i and i >= 2:
                # Volume des deux chandelles du pattern
                pattern_volume = (volume[i - 1] + volume[i]) / 2.0

                # Volume moyen avant le pattern
                avg_volume = 0.0
                count = 0
                for j in range(max(0, i - 5), i - 1):
                    if j < len(volume) and volume[j] > 0:
                        avg_volume += volume[j]
                        count += 1

                if count > 0 and avg_volume > 0:
                    avg_volume = safe_divide(avg_volume, count)
                    volume_ratio = safe_divide(pattern_volume, avg_volume)
                    volume_score = min(1.0, safe_divide(volume_ratio, min_volume_surge))

            # === CALCUL DU SCORE PONDÉRÉ ===

            # Score du gap (plus il est grand, mieux c'est)
            gap_score = min(1.0, safe_divide(gap_ratio, min_gap_ratio * 2))

            # Score des corps (moyenne des deux marubozu)
            body_score = (prev_marubozu_score + curr_marubozu_score) / 2.0

            # Combinaison pondérée
            components = np.array([gap_score, body_score, volume_score])
            weights = np.array([gap_weight, body_weight, volume_weight])

            pattern_score = calculate_weighted_score(components, weights)
            signals[i] = pattern_score

        return signals

    @njit
    def detect_kicking_bull_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        min_gap_ratio: float,
        min_body_ratio: float,
        min_volume_surge: float,
        gap_weight: float,
        body_weight: float,
        volume_weight: float,
    ) -> np.ndarray:
        """VERSION CORRIGÉE - Détection du pattern Kicking Bull.

        CORRECTIONS APPORTÉES:
        1. Conditions quasi-parfaites remplacées par tolérances réalistes
        2. Gestion des erreurs silencieuses avec validation stricte
        3. Système de scoring pondéré au lieu de conditions binaires
        4. Paramètres plus réalistes

        PATTERN:
        - Marubozu baissier suivi d'un gap haussier
        - Marubozu haussier après le gap
        - Volume élevé
        - Signal de continuation haussière forte

        Args:
            Prix OHLC et volume
            min_gap_ratio: Ratio minimum du gap (0.005 recommandé)
            min_body_ratio: Ratio minimum du corps (0.8 recommandé)
            min_volume_surge: Surge de volume minimum (1.5 recommandé)
            Poids pour le scoring pondéré
            body_weight: TODO: documenter.
            close_prices: TODO: documenter.
            gap_weight: TODO: documenter.
            high_prices: TODO: documenter.
            low_prices: TODO: documenter.
            open_prices: TODO: documenter.
            volume: TODO: documenter.
            volume_weight: TODO: documenter.

        Returns:
            Array des scores de détection (0.0 à 1.0)
        """
        signals = np.zeros(len(open_prices), dtype=np.float64)
        # AUDIT FIX C-B7-3: supprimé kicks_found (variable morte) dans kicking_bull

        # Seuil max pour les ombres d'un Marubozu (30% du range total)
        max_shadow_ratio = 0.3

        for i in range(1, len(open_prices)):
            # === VALIDATION DES DONNÉES ===
            if (
                open_prices[i - 1] <= 0
                or high_prices[i - 1] <= 0
                or low_prices[i - 1] <= 0
                or close_prices[i - 1] <= 0
                or open_prices[i] <= 0
                or high_prices[i] <= 0
                or low_prices[i] <= 0
                or close_prices[i] <= 0
            ):
                continue

            # === VÉRIFICATION DU GAP HAUSSIER ===
            gap_up = open_prices[i] > high_prices[i - 1]
            if not gap_up:
                continue

            gap_size = open_prices[i] - high_prices[i - 1]
            gap_ratio = safe_divide(gap_size, close_prices[i - 1])

            if gap_ratio < min_gap_ratio:
                continue

            # === VÉRIFICATION DU MARUBOZU BAISSIER (CHANDELLE PRÉCÉDENTE) ===
            prev_body_size = abs(close_prices[i - 1] - open_prices[i - 1])
            prev_total_range = high_prices[i - 1] - low_prices[i - 1]
            prev_upper_shadow = high_prices[i - 1] - max(
                open_prices[i - 1], close_prices[i - 1]
            )
            prev_lower_shadow = (
                min(open_prices[i - 1], close_prices[i - 1]) - low_prices[i - 1]
            )

            if prev_total_range <= 0:
                continue

            # Doit être baissier
            if close_prices[i - 1] >= open_prices[i - 1]:
                continue

            # Évaluation du corps (ASSOUPLIE)
            prev_body_ratio = safe_divide(prev_body_size, prev_total_range)
            prev_body_score = min(1.0, safe_divide(prev_body_ratio, min_body_ratio))

            # Évaluation des ombres (doivent être petites)
            prev_shadow_ratio = safe_divide(
                (prev_upper_shadow + prev_lower_shadow), prev_total_range
            )
            prev_shadow_score = max(
                0.0, 1.0 - safe_divide(prev_shadow_ratio, max_shadow_ratio)
            )  # AUDIT FIX C-B7-3: 3.0 hardcodé → safe_divide(ratio, max_shadow_ratio)

            prev_marubozu_score = (prev_body_score + prev_shadow_score) / 2.0

            # === VÉRIFICATION DU MARUBOZU HAUSSIER (CHANDELLE ACTUELLE) ===
            curr_body_size = abs(close_prices[i] - open_prices[i])
            curr_total_range = high_prices[i] - low_prices[i]
            curr_upper_shadow = high_prices[i] - max(open_prices[i], close_prices[i])
            curr_lower_shadow = min(open_prices[i], close_prices[i]) - low_prices[i]

            if curr_total_range <= 0:
                continue

            # Doit être haussier
            if close_prices[i] <= open_prices[i]:
                continue

            # Évaluation du corps (ASSOUPLIE)
            curr_body_ratio = safe_divide(curr_body_size, curr_total_range)
            curr_body_score = min(1.0, safe_divide(curr_body_ratio, min_body_ratio))

            # Évaluation des ombres
            curr_shadow_ratio = safe_divide(
                (curr_upper_shadow + curr_lower_shadow), curr_total_range
            )
            curr_shadow_score = max(
                0.0, 1.0 - safe_divide(curr_shadow_ratio, max_shadow_ratio)
            )  # AUDIT FIX C-B7-3: 3.0 hardcodé → safe_divide(ratio, max_shadow_ratio)

            curr_marubozu_score = (curr_body_score + curr_shadow_score) / 2.0

            # === ANALYSE DU VOLUME ===
            volume_score = 1.0
            if len(volume) > i and i >= 2:
                # Volume des deux chandelles du pattern
                pattern_volume = (volume[i - 1] + volume[i]) / 2.0

                # Volume moyen avant le pattern
                avg_volume = 0.0
                count = 0
                for j in range(max(0, i - 5), i - 1):
                    if j < len(volume) and volume[j] > 0:
                        avg_volume += volume[j]
                        count += 1

                if count > 0 and avg_volume > 0:
                    avg_volume = safe_divide(avg_volume, count)
                    volume_ratio = safe_divide(pattern_volume, avg_volume)
                    volume_score = min(1.0, safe_divide(volume_ratio, min_volume_surge))

            # === CALCUL DU SCORE PONDÉRÉ ===

            # Score du gap (plus il est grand, mieux c'est)
            gap_score = min(1.0, safe_divide(gap_ratio, min_gap_ratio * 2))

            # Score des corps (moyenne des deux marubozu)
            body_score = (prev_marubozu_score + curr_marubozu_score) / 2.0

            # Combinaison pondérée
            components = np.array([gap_score, body_score, volume_score])
            weights = np.array([gap_weight, body_weight, volume_weight])

            pattern_score = calculate_weighted_score(components, weights)
            signals[i] = pattern_score

        return signals

    @njit
    def detect_breakaway_gap_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        min_gap_ratio: float,
        min_consolidation: int,
        min_volume_surge: float,
        min_continuation: int,
        gap_weight: float,
        volume_weight: float,
        consolidation_weight: float,
    ) -> np.ndarray:
        """VERSION SANS LOOKAHEAD."""
        signals = np.zeros(len(open_prices), dtype=np.float64)
        for i in range(min_consolidation + 2, len(open_prices)):
            gap_up = open_prices[i] > high_prices[i - 1]
            gap_down = open_prices[i] < low_prices[i - 1]
            if not gap_up and not gap_down:
                continue

            gap_ratio = (
                safe_divide((open_prices[i] - high_prices[i - 1]), high_prices[i - 1])
                if gap_up
                else safe_divide(
                    (low_prices[i - 1] - open_prices[i]), low_prices[i - 1]
                )
            )
            if gap_ratio < min_gap_ratio:
                continue

            # Check consolidation before gap
            if min_consolidation <= 0:
                continue
            consolidation_closes = close_prices[i - min_consolidation : i]
            if len(consolidation_closes) == 0:
                continue
            cons_max = np.max(consolidation_closes)
            cons_min = np.min(consolidation_closes)
            if cons_min == 0:
                continue
            # cons_range était utilisé avec ancien seuil absolu (supprimé AUDIT FIX C-B7-4)
            # cons_range = safe_divide((cons_max - cons_min), cons_min)  # remplacé par ATR

            # AUDIT FIX C-B7-4: cons_range > 0.05 absolu → filtre ATR adaptatif cross-asset
            # Une vraie consolidation doit avoir un range absolu < 1.5 × ATR local
            _atr_cons = 0.0
            _atr_cnt_c = 0
            for _k in range(max(1, i - min_consolidation), i):
                _tr_c = max(
                    high_prices[_k] - low_prices[_k],
                    abs(high_prices[_k] - close_prices[_k - 1]),
                    abs(low_prices[_k] - close_prices[_k - 1]),
                )
                _atr_cons += _tr_c
                _atr_cnt_c += 1
            avg_atr_cons = (
                _atr_cons / _atr_cnt_c
                if _atr_cnt_c > 0
                else (high_prices[i] - low_prices[i])
            )
            if avg_atr_cons <= 0:
                avg_atr_cons = abs(close_prices[i] - close_prices[i - 1]) + 1e-10
            abs_cons_range = cons_max - cons_min
            if abs_cons_range > 1.5 * avg_atr_cons:
                continue

            avg_vol = np.mean(volume[i - min_consolidation : i])
            vol_score = 0.0
            if avg_vol > 0:
                vol_ratio = safe_divide(volume[i], avg_vol)
                if vol_ratio > min_volume_surge:
                    vol_score = min(1.0, safe_divide(vol_ratio, min_volume_surge * 2))

            if vol_score == 0:
                continue

            gap_score = min(1.0, safe_divide(gap_ratio, min_gap_ratio * 2))
            # cons_score: range relatif normalisé sur ATR (plus la conso est étroite, meilleur le score)
            cons_score = max(0.0, 1.0 - safe_divide(abs_cons_range, 1.5 * avg_atr_cons))

            components = np.array([gap_score, vol_score, cons_score])
            weights = np.array([gap_weight, volume_weight, consolidation_weight])
            signals[i] = calculate_weighted_score(components, weights)
        return signals

    @njit
    def detect_breakaway_bear_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        min_gap_ratio: float,
        min_continuation: int,
        min_volume_surge: float,
        gap_weight: float,
        continuation_weight: float,
        volume_weight: float,
    ) -> np.ndarray:
        """VERSION SANS LOOKAHEAD."""
        signals = np.zeros(len(open_prices), dtype=np.float64)
        for i in range(15, len(open_prices)):
            gap_down = open_prices[i] < low_prices[i - 1]
            if not gap_down:
                continue

            gap_ratio = safe_divide(
                (low_prices[i - 1] - open_prices[i]), low_prices[i - 1]
            )
            if gap_ratio < min_gap_ratio:
                continue

            prices_before = close_prices[i - 10 : i]
            trend = safe_divide(
                (prices_before[-1] - prices_before[0]), prices_before[0]
            )
            if trend > 0:  # Doit s'inscrire dans une dynamique baissière ou plate
                continue

            avg_vol = np.mean(volume[i - 10 : i])
            vol_score = 0.0
            if avg_vol > 0:
                vol_ratio = safe_divide(volume[i], avg_vol)
                if vol_ratio > min_volume_surge:
                    vol_score = min(1.0, safe_divide(vol_ratio, min_volume_surge * 2))

            if vol_score == 0:
                continue

            gap_score = min(1.0, safe_divide(gap_ratio, min_gap_ratio * 2))

            # AUDIT FIX C11: min_continuation était un paramètre mort (1.0 hardcodé).
            # [REF-BREAKAWAY-BULL] Symétrique à detect_breakaway_bull_numba.
            # trend est négatif pour la tendance baissière → on prend abs(trend)
            continuation_score = min(
                1.0, safe_divide(abs(trend), min_continuation * 0.01)
            )

            components = np.array([gap_score, continuation_score, vol_score])
            weights = np.array([gap_weight, continuation_weight, volume_weight])
            signals[i] = calculate_weighted_score(components, weights)
        return signals

    @njit
    def detect_three_white_soldiers_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        min_body_ratio: float,
        max_shadow_ratio: float,
        min_consecutive_advance: float,
        body_weight: float,
        shadow_weight: float,
        advance_weight: float,
    ) -> np.ndarray:
        """VERSION CORRIGÉE - Détection du pattern Three White Soldiers.

        CORRECTIONS APPORTÉES:
        1. Conditions quasi-parfaites remplacées par tolérances réalistes
        2. Gestion des erreurs silencieuses avec validation stricte
        3. Système de scoring pondéré au lieu de conditions binaires
        4. Paramètres plus réalistes pour les candlesticks

        PATTERN:
        - 3 chandelles haussières consécutives
        - Corps longs et ombres courtes
        - Chaque ouverture dans le corps précédent
        - Clôture près du haut de chaque chandelle
        - Signal de retournement haussier

        Args:
            Prix OHLC et volume
            min_body_ratio: Ratio minimum du corps (0.6 recommandé)
            max_shadow_ratio: Ratio maximum des ombres (0.3 recommandé)
            min_consecutive_advance: Avancée minimum consécutive (0.02 recommandé)
            Poids pour le scoring pondéré
            advance_weight: TODO: documenter.
            body_weight: TODO: documenter.
            close_prices: TODO: documenter.
            high_prices: TODO: documenter.
            low_prices: TODO: documenter.
            open_prices: TODO: documenter.
            shadow_weight: TODO: documenter.
            volume: TODO: documenter.

        Returns:
            Array des scores de détection (0.0 à 1.0)
        """
        signals = np.zeros(len(open_prices), dtype=np.float64)
        soldiers_found = 0

        for i in range(2, len(open_prices)):
            # Vérifier 3 chandelles consécutives
            candles_valid = True
            candle_scores = np.zeros(3, dtype=np.float64)

            for j in range(3):
                idx = i - 2 + j

                # === VALIDATION DES DONNÉES ===
                if (
                    idx < 0
                    or open_prices[idx] <= 0
                    or high_prices[idx] <= 0
                    or low_prices[idx] <= 0
                    or close_prices[idx] <= 0
                ):
                    candles_valid = False
                    break

                # === CALCUL DES COMPOSANTS DE LA CHANDELLE ===
                body_size = abs(close_prices[idx] - open_prices[idx])
                total_range = high_prices[idx] - low_prices[idx]
                upper_shadow = high_prices[idx] - max(
                    open_prices[idx], close_prices[idx]
                )
                lower_shadow = (
                    min(open_prices[idx], close_prices[idx]) - low_prices[idx]
                )

                if total_range <= 0:
                    candles_valid = False
                    break

                # === VÉRIFICATION CHANDELLE HAUSSIÈRE ===
                if close_prices[idx] <= open_prices[idx]:  # Doit être haussière
                    candles_valid = False
                    break

                # === ÉVALUATION DU CORPS ===
                body_ratio = safe_divide(body_size, total_range)
                if body_ratio < min_body_ratio:
                    candles_valid = False
                    break
                body_score = min(1.0, safe_divide(body_ratio, min_body_ratio))

                # === ÉVALUATION DES OMBRES — filtre DUR + score graduel ===
                shadow_ratio = safe_divide((upper_shadow + lower_shadow), total_range)
                # FIX : filtre dur — une chandelle avec trop d'ombres est éliminée
                # (shadow_score graduel seul ne suffisait pas : 4% de taux résiduel).
                if shadow_ratio > max_shadow_ratio:
                    candles_valid = False
                    break
                shadow_score = max(
                    0.0, 1.0 - safe_divide(shadow_ratio, max_shadow_ratio)
                )

                # Score combiné pour cette chandelle
                candle_scores[j] = (body_score + shadow_score) / 2.0

            if not candles_valid:
                continue

            # === VÉRIFICATION DE LA PROGRESSION HAUSSIÈRE ===
            progression_valid = True
            advance_scores = np.zeros(2, dtype=np.float64)

            for j in range(2):
                idx1 = i - 2 + j
                idx2 = i - 1 + j

                # P1-4 FIX: Activer le filtre curr_open_in_body (clef de confirmation)
                # Chaque ouverture doit être dans le corps de la bougie précédente
                curr_open_in_body = (
                    open_prices[idx2] >= open_prices[idx1]
                    and open_prices[idx2] <= close_prices[idx1] * 1.05
                )  # Tolérance 5%

                if not curr_open_in_body:
                    progression_valid = False
                    break

                # FIX: normalisation ATR pour l'avance (cross-asset universel)
                # Calcul ATR local sur 14 bougies avant idx1
                _atr_start = max(1, idx1 - 13)
                _atr_sum = 0.0
                _atr_cnt = 0
                for _aj in range(_atr_start, idx1 + 1):
                    _tr = max(
                        high_prices[_aj] - low_prices[_aj],
                        abs(high_prices[_aj] - close_prices[_aj - 1]),
                        abs(low_prices[_aj] - close_prices[_aj - 1]),
                    )
                    _atr_sum += _tr
                    _atr_cnt += 1
                _atr_local = (
                    _atr_sum / _atr_cnt
                    if _atr_cnt > 0
                    else (high_prices[idx1] - low_prices[idx1])
                )
                if _atr_local <= 0:
                    _atr_local = abs(close_prices[idx1] - open_prices[idx1]) + 1e-10

                # min_advance_atr = 0.25 × ATR (remplace seuil absolu 0.002)
                min_advance_atr = min_consecutive_advance * _atr_local  # param = 0.25
                advance = close_prices[idx2] - close_prices[idx1]
                if advance < min_advance_atr:
                    progression_valid = False
                    break

                advance_scores[j] = min(1.0, safe_divide(advance, min_advance_atr * 2))

            if not progression_valid:
                continue

            # === CALCUL DU SCORE PONDÉRÉ ===

            # FIX : avg_body_score et avg_shadow_score séparés.
            # candle_scores[j] = (body_score + shadow_score) / 2 → on re-dérive
            # les composantes individuelles via le score combiné :
            # body_score  ≈ candle_scores (filtre dur sur min_body_ratio garanti)
            # shadow_score ≈ 1 - shadow_ratio/max (filtre dur garanti aussi maintenant)
            # On utilise candle_scores directement pour body ET on calcule shadow
            # à partir de shadow_scores stockés dans un tableau dédié.
            avg_body_score = (
                candle_scores[0] + candle_scores[1] + candle_scores[2]
            ) / 3.0

            # Score des ombres : moyenne réelle des shadow_scores par chandelle.
            # shadow_score_j = max(0, 1 - shadow_ratio_j / max_shadow_ratio)
            # On le récalcule proprement ici (évite la dead variable précédente).
            _s0_range = high_prices[i - 2] - low_prices[i - 2]
            _s1_range = high_prices[i - 1] - low_prices[i - 1]
            _s2_range = high_prices[i] - low_prices[i]
            _s0_shadow = (
                high_prices[i - 2]
                - max(open_prices[i - 2], close_prices[i - 2])
                + min(open_prices[i - 2], close_prices[i - 2])
                - low_prices[i - 2]
            )
            _s1_shadow = (
                high_prices[i - 1]
                - max(open_prices[i - 1], close_prices[i - 1])
                + min(open_prices[i - 1], close_prices[i - 1])
                - low_prices[i - 1]
            )
            _s2_shadow = (
                high_prices[i]
                - max(open_prices[i], close_prices[i])
                + min(open_prices[i], close_prices[i])
                - low_prices[i]
            )
            _ss0 = max(
                0.0,
                1.0
                - safe_divide(
                    safe_divide(_s0_shadow, _s0_range if _s0_range > 0 else 1.0),
                    max_shadow_ratio,
                ),
            )
            _ss1 = max(
                0.0,
                1.0
                - safe_divide(
                    safe_divide(_s1_shadow, _s1_range if _s1_range > 0 else 1.0),
                    max_shadow_ratio,
                ),
            )
            _ss2 = max(
                0.0,
                1.0
                - safe_divide(
                    safe_divide(_s2_shadow, _s2_range if _s2_range > 0 else 1.0),
                    max_shadow_ratio,
                ),
            )
            avg_shadow_score = (_ss0 + _ss1 + _ss2) / 3.0

            # Score de l'avancée (moyenne des 2 transitions)
            avg_advance_score = (advance_scores[0] + advance_scores[1]) / 2.0

            # Combinaison pondérée
            components = np.array([avg_body_score, avg_shadow_score, avg_advance_score])
            weights = np.array([body_weight, shadow_weight, advance_weight])

            pattern_score = calculate_weighted_score(components, weights)
            signals[i] = pattern_score
            soldiers_found += 1

        return signals

    @njit
    def detect_spike_reversal_numba(
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
        volume: np.ndarray,
        min_spike_ratio: float,
        max_retracement: float,
        min_volume_spike: float,
        spike_weight: float,
        retracement_weight: float,
        volume_weight: float,
    ) -> np.ndarray:
        """VERSION CORRIGÉE - Détection du pattern Spike Reversal."""
        signals = np.zeros(len(open_prices), dtype=np.float64)
        min_reversal_ratio = 1.0 - max_retracement

        for i in range(20, len(open_prices)):
            # AUDIT FIX C-B7-5: ATR local pour seuil adaptatif cross-asset
            _atr_sum_s = 0.0
            _atr_cnt_s = 0
            for _k in range(max(1, i - 15), i - 1):
                _tr_s = max(
                    high_prices[_k] - low_prices[_k],
                    abs(high_prices[_k] - close_prices[_k - 1]),
                    abs(low_prices[_k] - close_prices[_k - 1]),
                )
                _atr_sum_s += _tr_s
                _atr_cnt_s += 1
            local_atr_s = _atr_sum_s / _atr_cnt_s if _atr_cnt_s > 0 else 0.0
            if local_atr_s <= 0:
                continue

            # Spike haussier
            spike_up_val = high_prices[i - 1]
            spike_up_base = close_prices[i - 2]
            spike_size_up = spike_up_val - spike_up_base
            spike_up_atrs = safe_divide(spike_size_up, local_atr_s)

            if spike_up_atrs >= 2.0:  # spike_up doit être >= 2 ATRs
                reversal_down = safe_divide(
                    spike_up_val - close_prices[i], spike_up_val
                )
                if reversal_down > min_reversal_ratio:
                    # spike_score normalisé sur 4 ATRs (score parfait à 4×ATR)
                    spike_score = min(1.0, safe_divide(spike_up_atrs, 4.0))
                    reversal_score = min(
                        1.0, safe_divide(reversal_down, min_reversal_ratio * 2)
                    )
                    volume_score = min(
                        1.0,
                        safe_divide(
                            volume[i - 1],
                            np.mean(volume[i - 6 : i - 1]) * min_volume_spike
                            if np.mean(volume[i - 6 : i - 1]) > 0
                            else 1.0,
                        ),
                    )

                    components = np.array([spike_score, reversal_score, volume_score])
                    weights = np.array(
                        [spike_weight, retracement_weight, volume_weight]
                    )
                    signals[i] = max(
                        signals[i], calculate_weighted_score(components, weights)
                    )

            # Spike baissier
            spike_down_val = low_prices[i - 1]
            spike_down_base = close_prices[i - 2]
            spike_size_down = spike_down_base - spike_down_val
            spike_down_atrs = safe_divide(spike_size_down, local_atr_s)

            if spike_down_atrs >= 2.0:  # spike_down doit être >= 2 ATRs
                reversal_up = safe_divide(
                    close_prices[i] - spike_down_val, spike_down_val
                )
                if reversal_up > min_reversal_ratio:
                    spike_score = min(1.0, safe_divide(spike_down_atrs, 4.0))
                    reversal_score = min(
                        1.0, safe_divide(reversal_up, min_reversal_ratio * 2)
                    )
                    volume_score = min(
                        1.0,
                        safe_divide(
                            volume[i - 1],
                            np.mean(volume[i - 6 : i - 1]) * min_volume_spike
                            if np.mean(volume[i - 6 : i - 1]) > 0
                            else 1.0,
                        ),
                    )

                    components = np.array([spike_score, reversal_score, volume_score])
                    weights = np.array(
                        [spike_weight, retracement_weight, volume_weight]
                    )
                    signals[i] = max(
                        signals[i], calculate_weighted_score(components, weights)
                    )

        return signals


class PatternMetadataManager:
    """Gère l'accès aux métadonnées des patterns (catégorie, type, fenêtre, etc.)."""

    def __init__(self, thresholds: dict[str, dict[str, Any]]):
        """__init__.

        Args:
            thresholds: TODO document.
        """
        self._metadata = {}
        for name, params in thresholds.items():
            self._metadata[name] = {
                "name": name,
                "category": params.get("catégorie", "Inconnue"),
                "type": params.get("type", "Inconnu"),
                "window": params.get("fenêtre", 1),
            }

    def get_pattern_metadata(self, pattern_name: str) -> dict[str, Any] | None:
        """Récupère les métadonnées pour un pattern spécifique."""
        return self._metadata.get(pattern_name)

    def get_all_patterns(self) -> list[str]:
        """Retourne la liste de tous les patterns disponibles."""
        return list(self._metadata.keys())

    def get_max_window(self, patterns: list[str]) -> int:
        """Calcule la fenêtre d'analyse maximale requise pour une liste de patterns."""
        max_window = 0
        for name in patterns:
            meta = self.get_pattern_metadata(name)
            if meta:
                window = meta["window"]
                if isinstance(window, dict):
                    # Pour les fenêtres de type {"min": x, "max": y} ou {"min": x}
                    current_max = window.get("max", window.get("min", 0))
                else:
                    current_max = window
                if current_max > max_window:
                    max_window = current_max
        return max_window if max_window > 0 else 1


NUMBA_FUNCTIONS = {
    "hammer": detect_hammer_numba,
    "inverted_hammer": detect_inverted_hammer_numba,
    "dragonfly_doji": detect_dragonfly_doji_numba,
    "pin_bar_bull": detect_pin_bar_bull_numba,
    "marubozu_bull": detect_marubozu_bull_numba,
    "belt_hold_bull": detect_belt_hold_bull_numba,
    "morning_star": detect_morning_star_numba,
    "piercing_line": detect_piercing_line_numba,
    "harami_bull": detect_harami_bull_numba,
    "abandoned_baby_bull": detect_abandoned_baby_bull_numba,
    "three_inside_up": detect_three_inside_up_numba,
    "three_outside_up": detect_three_outside_up_numba,
    "concealing_baby_swallow": detect_concealing_baby_swallow_numba,
    "unique_three_river_bottom": detect_unique_three_river_bottom_numba,
    "matching_low": detect_matching_low_numba,
    "ladder_bottom": detect_ladder_bottom_numba,
    "breakaway_bull": detect_breakaway_bull_numba,
    "hanging_man": detect_hanging_man_numba,
    "shooting_star": detect_shooting_star_numba,
    "gravestone_doji": detect_gravestone_doji_numba,
    "pin_bar_bear": detect_pin_bar_bear_numba,
    "marubozu_bear": detect_marubozu_bear_numba,
    "belt_hold_bear": detect_belt_hold_bear_numba,
    "evening_star": detect_evening_star_numba,
    "dark_cloud_cover": detect_dark_cloud_cover_numba,
    "harami_bear": detect_harami_bear_numba,
    "abandoned_baby_bear": detect_abandoned_baby_bear_numba,
    "three_inside_down": detect_three_inside_down_numba,
    "three_outside_down": detect_three_outside_down_numba,
    "advance_block": detect_advance_block_numba,
    "deliberation": detect_deliberation_numba,
    "matching_high": detect_matching_high_numba,
    "ladder_top": detect_ladder_top_numba,
    "doji": detect_doji_numba,
    "long_legged_doji": detect_long_legged_doji_numba,
    "spinning_top": detect_spinning_top_numba,
    "rickshaw_man": detect_rickshaw_man_numba,
    "high_wave_candle": detect_high_wave_candle_numba,
    "tri_star": detect_tri_star_numba,
    "double_top": detect_double_top_numba,
    "double_bottom": detect_double_bottom_numba,
    "triple_top": detect_triple_top_numba,
    "triple_bottom": detect_triple_bottom_numba,
    "v_top": detect_v_top_numba,
    "v_bottom": detect_v_bottom_numba,
    "rounding_top": detect_rounding_top_numba,
    "rounding_bottom": detect_rounding_bottom_numba,
    "diamond_top": detect_diamond_top_numba,
    "diamond_bottom": detect_diamond_bottom_numba,
    "island_top": detect_island_top_numba,
    "ascending_triangle": detect_ascending_triangle_numba,
    "descending_triangle": detect_descending_triangle_numba,
    "rectangle": detect_rectangle_numba,
    "bear_flag": detect_bear_flag_numba,
    "bull_pennant": detect_bull_pennant_numba,
    "bear_pennant": detect_bear_pennant_numba,
    "rising_wedge": detect_rising_wedge_numba,
    "broadening_wedge": detect_broadening_wedge_numba,
    "cup_handle": detect_cup_handle_numba,
    "measured_move": detect_measured_move_numba,
    "gap_up": detect_gap_up_numba,
    "gap_down": detect_gap_down_numba,
    "gap_fill": detect_gap_fill_numba,
    "runaway_gap": detect_runaway_gap_numba,
    "island_reversal": detect_island_reversal_numba,
    "gap_and_go": detect_gap_and_go_numba,
    "support": detect_support_numba,
    "resistance": detect_resistance_numba,
    "wolfe_wave": detect_wolfe_wave_numba,
    "four_price_doji": detect_four_price_doji_numba,
    "shark_bull": detect_shark_bull_numba,
    "engulfing_bull": detect_engulfing_bull_numba,
    "engulfing_bear": detect_engulfing_bear_numba,
    "head_shoulders": detect_head_shoulders_numba,
    "inv_head_shoulders": detect_inv_head_shoulders_numba,
    "bull_flag": detect_bull_flag_numba,
    "channel_up": detect_channel_up_numba,
    "channel_down": detect_channel_down_numba,
    "gartley_bull": detect_gartley_bull_numba,
    "gartley_bear": detect_gartley_bear_numba,
    "butterfly_bull": detect_butterfly_bull_numba,
    "butterfly_bear": detect_butterfly_bear_numba,
    "bat_bull": detect_bat_bull_numba,
    "bat_bear": detect_bat_bear_numba,
    "crab_bull": detect_crab_bull_numba,
    "crab_bear": detect_crab_bear_numba,
    "shark_bear": detect_shark_bear_numba,
    "uptrend": detect_uptrend_numba,
    "downtrend": detect_downtrend_numba,
    "sideways_trend": detect_sideways_trend_numba,
    "exhaustion_gap": detect_exhaustion_gap_numba,
    "three_drives": detect_three_drives_numba,
    "elliott_wave_1": detect_elliott_wave_1_numba,
    "elliott_wave_3": detect_elliott_wave_3_numba,
    "elliott_wave_5": detect_elliott_wave_5_numba,
    "fibonacci_retracement": detect_fibonacci_retracement_numba,
    "fibonacci_extension": detect_fibonacci_extension_numba,
    "symmetrical_triangle": detect_symmetrical_triangle_numba,
    "island_bottom": detect_island_bottom_numba,
    "three_black_crows": detect_three_black_crows_numba,
    "falling_wedge": detect_falling_wedge_numba,
    "kicking_bear": detect_kicking_bear_numba,
    "kicking_bull": detect_kicking_bull_numba,
    "breakaway_gap": detect_breakaway_gap_numba,
    "breakaway_bear": detect_breakaway_bear_numba,
    "three_white_soldiers": detect_three_white_soldiers_numba,
    "spike_reversal": detect_spike_reversal_numba,
}


class NumbaPatternDetectors:
    """Orchestrateur de détection de patterns qui génère dynamiquement.

    les fonctions de détection pour une extensibilité maximale.
    """

    def __init__(self, metadata_manager: PatternMetadataManager):
        """__init__.

        Args:
            metadata_manager: TODO document.
        """
        self.metadata_manager = metadata_manager
        self._detection_functions = {}
        self._register_detectors()

        print(
            f"✅ UnifiedPatternDetector initialisé avec {len(self._detection_functions)} détecteurs générés dynamiquement."
        )

    def _create_detector_for_pattern(
    self, pattern_name: str, numba_func: callable
    ) -> callable:
        """Factory qui crée un wrapper de détection pour un pattern donné.

        Le wrapper accepte désormais un tf_scale optionnel pour adapter
        les seuils sensibles à la timeframe courante.
        """
        params = PATTERN_THRESHOLDS[pattern_name]
        sig = inspect.signature(numba_func)
        expected_params = list(sig.parameters.keys())

        data_mapping = {
            "open_prices": "open",
            "high_prices": "high",
            "low_prices":  "low",
            "close_prices": "close",
            "volume":      "volume",
        }

        def detector_wrapper(
            data: dict[str, np.ndarray],
            tf_scale: float = 1.0,
        ) -> np.ndarray:
            """Closure de détection. tf_scale est injecté par detect() selon.

            la timeframe des données. Les paramètres listés dans
            TF_SENSITIVE_PARAMS sont multipliés par tf_scale avant injection.
            """
            args = []

            for param_name in expected_params:
                if param_name in data_mapping:
                    data_key = data_mapping[param_name]
                    if data_key in data:
                        args.append(data[data_key])
                    else:
                        raise ValueError(f"Données manquantes: {data_key}")

                elif param_name in params:
                    value = params[param_name]
                    # Scaling : multiplier par tf_scale si paramètre sensible
                    if tf_scale != 1.0 and param_name in TF_SENSITIVE_PARAMS:
                        if isinstance(value, float):
                            # Pour min_slope (négatif sur downtrend), on scale la magnitude
                            if value < 0:
                                value = value * tf_scale  # -0.0005 × 3 = -0.0015
                            else:
                                value = value * tf_scale  # 0.0005 × 3 = 0.0015
                    args.append(value)

                else:
                    logger.warning(
                        f"⚠️ Paramètre manquant '{param_name}' pour '{pattern_name}'. "
                        f"Valeur par défaut 0.0 injectée."
                    )
                    args.append(0.0)

            return numba_func(*args)

        return detector_wrapper

    def _register_detectors(self):
        """Parcourt PATTERN_THRESHOLDS, trouve la fonction Numba correspondante.

        et génère dynamiquement une fonction de détection pour chaque pattern.
        """
        logger.info("🛠️ Génération dynamique des fonctions de détection...")

        # On itère sur tous les patterns connus
        for pattern_name in PATTERN_THRESHOLDS.keys():
            # On construit le nom de la fonction Numba attendue
            numba_func_name = f"detect_{pattern_name}_numba"

            # On cherche cette fonction dans le scope global du module
            numba_func = NUMBA_FUNCTIONS.get(pattern_name)

            if callable(numba_func):
                # Si la fonction existe, on crée son 'wrapper' de détection
                detector = self._create_detector_for_pattern(pattern_name, numba_func)
                # Et on l'enregistre dans notre dictionnaire de dispatch
                self._detection_functions[pattern_name] = detector
                logger.info(f"   -> Détecteur '{pattern_name}' généré et enregistré.")
            else:
                logger.warning(
                    f"   ⚠️ Avertissement : Fonction Numba '{numba_func_name}' non trouvée pour le pattern '{pattern_name}'."
                )

    def detect(
    self, data: pl.DataFrame, patterns_to_detect: list[str]
    ) -> dict[str, np.ndarray]:
        """Méthode unifiée de détection avec scaling dynamique par timeframe."""
        print(f"🚀 Lancement de la détection pour {len(patterns_to_detect)} patterns.")

        valid_patterns = [
            p for p in patterns_to_detect if p in self._detection_functions
        ]
        invalid_patterns = [
            p for p in patterns_to_detect if p not in self._detection_functions
        ]

        if invalid_patterns:
            print(f"⚠️ Patterns ignorés (non implémentés) : {invalid_patterns}")

        if not valid_patterns:
            print("❌ Aucun pattern valide à détecter.")
            return {}

        required_cols = {"open", "high", "low", "close", "volume"}
        if not required_cols.issubset(data.columns):
            raise ValueError(
                f"Le DataFrame doit contenir les colonnes : {required_cols}"
            )

        # ── Conversion OHLCV en NumPy ─────────────────────────────────────────
        numpy_data = {
            "open":   data["open"].to_numpy(),
            "high":   data["high"].to_numpy(),
            "low":    data["low"].to_numpy(),
            "close":  data["close"].to_numpy(),
            "volume": data["volume"].to_numpy(),
        }

        # ── Détermination du facteur de scaling timeframe ─────────────────────
        # On lit la timeframe depuis la colonne du DataFrame si elle existe.
        # Par défaut : M5 (scale = 1.0) pour ne pas casser les appels sans colonne TF.
        tf_label = "5m"
        if "timeframe" in data.columns:
            try:
                tf_label = str(data["timeframe"][0])
            except Exception:
                tf_label = "5m"

        tf_scale = TF_SCALE_FACTORS.get(tf_label, 1.0)
        if tf_scale != 1.0:
            print(f"   📐 Scaling timeframe '{tf_label}' : ×{tf_scale} sur paramètres sensibles")

        # ── Seuil de score minimal ────────────────────────────────────────────
        MIN_SCORE_THRESHOLD = 0.65

        # ── Boucle de détection ───────────────────────────────────────────────
        results = {}

        for pattern_name in valid_patterns:
            print(f"   -> Détection de '{pattern_name}'...")
            detector_func = self._detection_functions[pattern_name]
            try:
                # Injection du tf_scale dans le wrapper
                result = detector_func(numpy_data, tf_scale=tf_scale)

                if isinstance(result, np.ndarray):
                    if result.dtype in [np.float64, np.float32]:
                        result = np.where(result >= MIN_SCORE_THRESHOLD, result, 0.0)
                    results[pattern_name] = result
                else:
                    results[pattern_name] = np.zeros(len(data), dtype=np.bool_)
                    logger.warning(
                        f"  ⚠️ {pattern_name} n'a pas renvoyé un tableau NumPy "
                        f"(type: {type(result)})."
                    )

            except Exception as e:
                print(f"❌ Erreur lors de la détection de '{pattern_name}': {e}")
                results[pattern_name] = np.zeros(len(data), dtype=np.bool_)

        logger.info("✅ Détection terminée.")
        return results
