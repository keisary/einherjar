"""
Generation du corpus brut EINHERJAR v2 — vrais noms MIDAS, strategies diversifiees.

Principe : chaque feature est un signal isole. Un Einher peut avoir n'importe quelle
feature comme trigger (pas seulement les patterns). Max 3 conditions (trigger + 2 filtres).

Noms MIDAS exacts :
- Patterns : pattern_hammer, pattern_engulfing_bull, pattern_double_top, etc.
- Indicateurs : rsi_14, macd_line, ema_12, sma_50, bb_percent, adx_14, etc.
- Quant : quant_hurst_exponent, quant_shannon_entropy, quant_regime_detection, etc.
"""

import json
import itertools
from pathlib import Path

# ============================================================================
# FEATURES PAR FAMILLE — noms MIDAS exacts
# ============================================================================

PATTERNS = [
    "pattern_hammer", "pattern_inverted_hammer", "pattern_dragonfly_doji",
    "pattern_morning_star", "pattern_piercing_line", "pattern_three_white_soldiers",
    "pattern_engulfing_bull", "pattern_engulfing_bear", "pattern_harami_bull",
    "pattern_harami_bear", "pattern_pin_bar_bull", "pattern_pin_bar_bear",
    "pattern_marubozu_bull", "pattern_marubozu_bear", "pattern_doji",
    "pattern_long_legged_doji", "pattern_spinning_top", "pattern_hanging_man",
    "pattern_shooting_star", "pattern_gravestone_doji", "pattern_evening_star",
    "pattern_dark_cloud_cover", "pattern_three_black_crows",
    "pattern_abandoned_baby_bull", "pattern_abandoned_baby_bear",
    "pattern_three_inside_up", "pattern_three_inside_down",
    "pattern_three_outside_up", "pattern_three_outside_down",
    "pattern_belt_hold_bull", "pattern_belt_hold_bear",
    "pattern_kicking_bull", "pattern_kicking_bear",
    "pattern_breakaway_bull", "pattern_breakaway_bear",
    "pattern_double_top", "pattern_double_bottom",
    "pattern_triple_top", "pattern_triple_bottom",
    "pattern_head_shoulders", "pattern_inv_head_shoulders",
    "pattern_rounding_top", "pattern_rounding_bottom",
    "pattern_ascending_triangle", "pattern_descending_triangle",
    "pattern_symmetrical_triangle", "pattern_rectangle",
    "pattern_bull_flag", "pattern_bear_flag",
    "pattern_bull_pennant", "pattern_bear_pennant",
    "pattern_rising_wedge", "pattern_falling_wedge",
    "pattern_cup_handle", "pattern_channel_up", "pattern_channel_down",
    "pattern_broadening_wedge", "pattern_measured_move",
    "pattern_gartley_bull", "pattern_gartley_bear",
    "pattern_butterfly_bull", "pattern_butterfly_bear",
    "pattern_bat_bull", "pattern_bat_bear",
    "pattern_crab_bull", "pattern_crab_bear",
    "pattern_shark_bull", "pattern_shark_bear",
    "pattern_support", "pattern_resistance",
    "pattern_uptrend", "pattern_downtrend", "pattern_sideways_trend",
    "pattern_gap_up", "pattern_gap_down", "pattern_gap_fill",
    "pattern_breakaway_gap", "pattern_runaway_gap", "pattern_exhaustion_gap",
    "pattern_island_reversal", "pattern_gap_and_go",
    "pattern_three_drives", "pattern_wolfe_wave",
    "pattern_elliott_wave_1", "pattern_elliott_wave_3", "pattern_elliott_wave_5",
    "pattern_fibonacci_retracement", "pattern_fibonacci_extension",
    "pattern_diamond_top", "pattern_diamond_bottom",
    "pattern_v_top", "pattern_v_bottom",
    "pattern_island_top", "pattern_island_bottom",
    "pattern_spike_reversal", "pattern_four_price_doji",
    "pattern_rickshaw_man", "pattern_high_wave_candle", "pattern_tri_star",
    "pattern_concealing_baby_swallow", "pattern_unique_three_river_bottom",
    "pattern_matching_low", "pattern_matching_high",
    "pattern_ladder_bottom", "pattern_ladder_top",
    "pattern_advance_block", "pattern_deliberation",
]

INDICATORS = [
    "rsi_14", "rsi_21", "rsi_30",
    "macd_line", "macd_signal", "macd_histogram",
    "ema_9", "ema_12", "ema_21", "ema_26", "ema_50", "ema_100", "ema_200",
    "sma_20", "sma_50", "sma_100", "sma_200",
    "bb_upper", "bb_middle", "bb_lower", "bb_width", "bb_percent",
    "volume_sma_20", "volume_ratio",
    "atr_14", "atr_21",
    "stoch_k", "stoch_d",
    "williams_r",
    "cci_20",
    "adx_14", "di_plus", "di_minus",
    "obv", "obv_ema",
    "vwap",
    "momentum_10", "momentum_20",
    "roc_10", "roc_20",
    "trix_14",
    "ultimate_oscillator",
    "money_flow_index",
    "chaikin_oscillator",
    "aroon_up", "aroon_down",
    "parabolic_sar",
    "supertrend", "supertrend_signal",
    "choppiness_index",
    "vortex_pos", "vortex_neg",
]

QUANTS = [
    "quant_hurst_exponent",
    "quant_shannon_entropy", "quant_sample_entropy", "quant_permutation_entropy", "quant_approximate_entropy",
    "quant_fractal_dimension", "quant_dfa_exponent",
    "quant_regime_detection",
    "quant_vol_persistence",
    "quant_autocorr_10", "quant_autocorr_20", "quant_autocorr_50",
    "quant_rolling_skewness", "quant_rolling_kurtosis",
    "quant_dynamic_var", "quant_dynamic_cvar", "quant_max_drawdown",
    "quant_amihud_illiquidity", "quant_kyles_lambda", "quant_kaufman_efficiency",
    "quant_variance_ratio",
    "quant_realized_vol_10", "quant_realized_vol_20", "quant_realized_vol_50",
    "quant_garch_volatility", "quant_vol_clustering",
    "quant_dominant_frequency", "quant_spectral_centroid",
]

TIMEFRAMES = ["5m", "15m", "1h", "4h", "1d"]
DIRECTIONS = ["long", "short"]

# ============================================================================
# EXPRESSIONS DE TRIGGER PAR TYPE DE FEATURE
# ============================================================================

def _pattern_expr(pattern: str, direction: str) -> str:
    """Expression pour un pattern detecte."""
    return f"{pattern} == 1"


def _indicator_trigger_exprs(ind: str, direction: str) -> list[tuple[str, str]]:
    """
    Retourne les expressions de trigger valides pour un indicateur.
    Chaque tuple : (nom_logique, expression_polars)
    """
    d = direction
    exprs = []

    if ind == "rsi_14":
        exprs.append(("rsi14_survente", "rsi_14 < 30"))
        exprs.append(("rsi14_surachat", "rsi_14 > 70"))
        exprs.append(("rsi14_neutre_haut", "rsi_14 > 50 AND rsi_14 < 70"))
        exprs.append(("rsi14_neutre_bas", "rsi_14 > 30 AND rsi_14 < 50"))
    elif ind == "rsi_21":
        exprs.append(("rsi21_survente", "rsi_21 < 30"))
        exprs.append(("rsi21_surachat", "rsi_21 > 70"))
    elif ind == "rsi_30":
        exprs.append(("rsi30_survente", "rsi_30 < 30"))
        exprs.append(("rsi30_surachat", "rsi_30 > 70"))

    elif ind == "macd_line":
        if d == "long":
            exprs.append(("macd_above_signal", "macd_line > macd_signal AND macd_histogram > 0"))
            exprs.append(("macd_hist_rising", "macd_histogram > 0"))
        else:
            exprs.append(("macd_below_signal", "macd_line < macd_signal AND macd_histogram < 0"))
            exprs.append(("macd_hist_falling", "macd_histogram < 0"))
    elif ind == "macd_histogram":
        if d == "long":
            exprs.append(("macd_hist_positive", "macd_histogram > 0"))
        else:
            exprs.append(("macd_hist_negative", "macd_histogram < 0"))

    elif ind == "ema_12":
        if d == "long":
            exprs.append(("ema12_above_26", "ema_12 > ema_26 AND close > ema_12"))
            exprs.append(("price_above_ema12", "close > ema_12"))
        else:
            exprs.append(("ema12_below_26", "ema_12 < ema_26 AND close < ema_12"))
            exprs.append(("price_below_ema12", "close < ema_12"))
    elif ind == "ema_50":
        if d == "long":
            exprs.append(("ema50_above_200", "ema_50 > ema_200 AND close > ema_50"))
            exprs.append(("price_above_ema50", "close > ema_50"))
        else:
            exprs.append(("ema50_below_200", "ema_50 < ema_200 AND close < ema_50"))
            exprs.append(("price_below_ema50", "close < ema_50"))
    elif ind in ("ema_9", "ema_21", "ema_26", "ema_100", "ema_200"):
        if d == "long":
            exprs.append((f"price_above_{ind}", f"close > {ind}"))
        else:
            exprs.append((f"price_below_{ind}", f"close < {ind}"))

    elif ind == "sma_20":
        if d == "long":
            exprs.append(("sma20_above_50", "sma_20 > sma_50 AND close > sma_20"))
        else:
            exprs.append(("sma20_below_50", "sma_20 < sma_50 AND close < sma_20"))
    elif ind in ("sma_50", "sma_100", "sma_200"):
        if d == "long":
            exprs.append((f"price_above_{ind}", f"close > {ind}"))
        else:
            exprs.append((f"price_below_{ind}", f"close < {ind}"))

    elif ind == "bb_percent":
        if d == "long":
            exprs.append(("bb_pct_low", "bb_percent < 0.05 AND close < bb_middle"))
            exprs.append(("bb_pct_oversold", "bb_percent < 0.1"))
        else:
            exprs.append(("bb_pct_high", "bb_percent > 0.95 AND close > bb_middle"))
            exprs.append(("bb_pct_overbought", "bb_percent > 0.9"))
    elif ind == "bb_width":
        exprs.append(("bb_squeeze", "bb_width < 0.02"))
        exprs.append(("bb_expansion", "bb_width > 0.05"))

    elif ind == "adx_14":
        exprs.append(("adx_strong", "adx_14 > 25"))
        exprs.append(("adx_weak", "adx_14 < 20"))
    elif ind == "di_plus":
        if d == "long":
            exprs.append(("di_plus_above_minus", "di_plus > di_minus AND adx_14 > 20"))
    elif ind == "di_minus":
        if d == "short":
            exprs.append(("di_minus_above_plus", "di_minus > di_plus AND adx_14 > 20"))

    elif ind == "stoch_k":
        if d == "long":
            exprs.append(("stoch_survente", "stoch_k < 20"))
        else:
            exprs.append(("stoch_surachat", "stoch_k > 80"))
    elif ind == "stoch_d":
        if d == "long":
            exprs.append(("stochd_survente", "stoch_d < 20"))
        else:
            exprs.append(("stochd_surachat", "stoch_d > 80"))

    elif ind == "williams_r":
        if d == "long":
            exprs.append(("williams_r_survente", "williams_r < -80"))
        else:
            exprs.append(("williams_r_surachat", "williams_r > -20"))

    elif ind == "cci_20":
        if d == "long":
            exprs.append(("cci_survente", "cci_20 < -100"))
        else:
            exprs.append(("cci_surachat", "cci_20 > 100"))

    elif ind == "volume_ratio":
        if d == "long":
            exprs.append(("volume_spike", "volume_ratio > 2.0"))
        else:
            exprs.append(("volume_spike_bear", "volume_ratio > 2.0"))

    elif ind == "atr_14":
        exprs.append(("atr_expansion", "atr_14 > atr_21"))
        exprs.append(("atr_low", "atr_14 < atr_21"))

    elif ind == "momentum_10":
        if d == "long":
            exprs.append(("momentum_positive", "momentum_10 > 0"))
        else:
            exprs.append(("momentum_negative", "momentum_10 < 0"))
    elif ind == "roc_10":
        if d == "long":
            exprs.append(("roc_positive", "roc_10 > 0"))
        else:
            exprs.append(("roc_negative", "roc_10 < 0"))

    elif ind == "supertrend_signal":
        if d == "long":
            exprs.append(("supertrend_bull", "supertrend_signal == 1"))
        else:
            exprs.append(("supertrend_bear", "supertrend_signal == -1"))

    elif ind == "vwap":
        if d == "long":
            exprs.append(("price_above_vwap", "close > vwap"))
        else:
            exprs.append(("price_below_vwap", "close < vwap"))

    elif ind == "obv":
        if d == "long":
            exprs.append(("obv_rising", "obv > obv_ema"))
        else:
            exprs.append(("obv_falling", "obv < obv_ema"))

    elif ind == "aroon_up":
        if d == "long":
            exprs.append(("aroon_up_high", "aroon_up > 70 AND aroon_up > aroon_down"))
    elif ind == "aroon_down":
        if d == "short":
            exprs.append(("aroon_down_high", "aroon_down > 70 AND aroon_down > aroon_up"))

    elif ind == "parabolic_sar":
        if d == "long":
            exprs.append(("sar_bull", "close > parabolic_sar"))
        else:
            exprs.append(("sar_bear", "close < parabolic_sar"))

    elif ind == "money_flow_index":
        if d == "long":
            exprs.append(("mfi_survente", "money_flow_index < 20"))
        else:
            exprs.append(("mfi_surachat", "money_flow_index > 80"))

    elif ind == "ultimate_oscillator":
        if d == "long":
            exprs.append(("uo_survente", "ultimate_oscillator < 30"))
        else:
            exprs.append(("uo_surachat", "ultimate_oscillator > 70"))

    elif ind == "trix_14":
        if d == "long":
            exprs.append(("trix_positive", "trix_14 > 0"))
        else:
            exprs.append(("trix_negative", "trix_14 < 0"))

    elif ind == "chaikin_oscillator":
        if d == "long":
            exprs.append(("chaikin_positive", "chaikin_oscillator > 0"))
        else:
            exprs.append(("chaikin_negative", "chaikin_oscillator < 0"))

    elif ind == "choppiness_index":
        exprs.append(("chop_trending", "choppiness_index < 38.2"))
        exprs.append(("chop_choppy", "choppiness_index > 61.8"))

    elif ind == "vortex_pos":
        if d == "long":
            exprs.append(("vi_plus_above", "vortex_pos > vortex_neg"))
    elif ind == "vortex_neg":
        if d == "short":
            exprs.append(("vi_minus_above", "vortex_neg > vortex_pos"))

    return exprs


def _quant_trigger_exprs(q: str, direction: str) -> list[tuple[str, str]]:
    """Expressions de trigger pour features quantitatives."""
    d = direction
    exprs = []

    if q == "quant_hurst_exponent":
        if d == "long":
            exprs.append(("hurst_trending", "quant_hurst_exponent > 0.6"))
        else:
            exprs.append(("hurst_trending_short", "quant_hurst_exponent > 0.6"))
        exprs.append(("hurst_meanrev", "quant_hurst_exponent < 0.4"))

    elif q == "quant_shannon_entropy":
        exprs.append(("entropy_compression", "quant_shannon_entropy < 3.5"))
        exprs.append(("entropy_expansion", "quant_shannon_entropy > 5.0"))

    elif q == "quant_regime_detection":
        if d == "long":
            exprs.append(("regime_bull", "quant_regime_detection > 0.5"))
        else:
            exprs.append(("regime_bear", "quant_regime_detection < -0.5"))

    elif q == "quant_vol_clustering":
        exprs.append(("vol_cluster_high", "quant_vol_clustering > 0.5"))

    elif q == "quant_fractal_dimension":
        exprs.append(("fractal_high", "quant_fractal_dimension > 1.5"))
        exprs.append(("fractal_low", "quant_fractal_dimension < 1.3"))

    elif q == "quant_rolling_skewness":
        if d == "long":
            exprs.append(("skew_neg", "quant_rolling_skewness < -0.5"))
        else:
            exprs.append(("skew_pos", "quant_rolling_skewness > 0.5"))

    elif q == "quant_rolling_kurtosis":
        exprs.append(("kurt_high", "quant_rolling_kurtosis > 3.0"))

    elif q == "quant_dynamic_var":
        exprs.append(("var_extreme", "quant_dynamic_var < -0.02"))

    elif q == "quant_kaufman_efficiency":
        if d == "long":
            exprs.append(("kaufman_trending", "quant_kaufman_efficiency > 0.6"))

    elif q == "quant_variance_ratio":
        if d == "long":
            exprs.append(("vr_trend", "quant_variance_ratio > 1.0"))
        else:
            exprs.append(("vr_meanrev", "quant_variance_ratio < 1.0"))

    elif q == "quant_vol_persistence":
        exprs.append(("vol_persist_high", "quant_vol_persistence > 0.5"))

    elif q == "quant_dfa_exponent":
        if d == "long":
            exprs.append(("dfa_trending", "quant_dfa_exponent > 0.6"))
        exprs.append(("dfa_meanrev", "quant_dfa_exponent < 0.4"))

    elif q == "quant_sample_entropy":
        exprs.append(("sampen_low", "quant_sample_entropy < 0.2"))

    elif q == "quant_permutation_entropy":
        exprs.append(("permen_low", "quant_permutation_entropy < 0.3"))

    elif q == "quant_approximate_entropy":
        exprs.append(("appen_low", "quant_approximate_entropy < 0.3"))

    elif q == "quant_realized_vol_10":
        exprs.append(("rvol_spike", "quant_realized_vol_10 > quant_realized_vol_50 * 1.5"))

    elif q == "quant_amihud_illiquidity":
        exprs.append(("amihud_high", "quant_amihud_illiquidity > 0.01"))

    elif q == "quant_autocorr_10":
        exprs.append(("autocorr_pos", "quant_autocorr_10 > 0.3"))
        exprs.append(("autocorr_neg", "quant_autocorr_10 < -0.3"))

    return exprs


# ============================================================================
# FILTRES COMMUNS (utilisables par n'importe quel trigger)
# ============================================================================

def _common_filters(direction: str) -> list[tuple[str, str]]:
    """Filtres generiques applicables a tous les Einhers."""
    d = direction
    filters = []

    # Tendance
    if d == "long":
        filters.append(("adx_trend", "adx_14 > 20"))
        filters.append(("ema_bull", "ema_12 > ema_26"))
        filters.append(("sma_bull", "sma_20 > sma_50"))
        filters.append(("price_above_ema50", "close > ema_50"))
        filters.append(("di_plus_dominant", "di_plus > di_minus"))
        filters.append(("supertrend_bull", "supertrend_signal == 1"))
    else:
        filters.append(("adx_trend", "adx_14 > 20"))
        filters.append(("ema_bear", "ema_12 < ema_26"))
        filters.append(("sma_bear", "sma_20 < sma_50"))
        filters.append(("price_below_ema50", "close < ema_50"))
        filters.append(("di_minus_dominant", "di_minus > di_plus"))
        filters.append(("supertrend_bear", "supertrend_signal == -1"))

    # Volume
    filters.append(("volume_confirm", "volume_ratio > 1.2"))
    filters.append(("volume_spike", "volume_ratio > 1.5"))

    # Volatilite
    filters.append(("atr_normal", "atr_14 > 0"))
    filters.append(("not_choppy", "choppiness_index < 61.8"))

    # RSI
    if d == "long":
        filters.append(("rsi_not_overbought", "rsi_14 < 70"))
    else:
        filters.append(("rsi_not_oversold", "rsi_14 > 30"))

    # Quant
    filters.append(("hurst_ok", "quant_hurst_exponent > 0.4 AND quant_hurst_exponent < 0.9"))
    filters.append(("entropy_ok", "quant_shannon_entropy > 2.0 AND quant_shannon_entropy < 6.0"))
    filters.append(("kurt_ok", "quant_rolling_kurtosis < 5.0"))

    # BB
    if d == "long":
        filters.append(("bb_not_squeezed", "bb_width > 0.01"))
    else:
        filters.append(("bb_not_squeezed", "bb_width > 0.01"))

    # MACD
    if d == "long":
        filters.append(("macd_bull", "macd_histogram > 0"))
    else:
        filters.append(("macd_bear", "macd_histogram < 0"))

    return filters


# ============================================================================
# CONSTRUCTEUR D'EINHER
# ============================================================================

def _make_einher(name: str, domain: str, direction: str, tf: str,
                 trigger: str, filters: list, tp_mult: float = 2.5, sl_mult: float = 1.5) -> dict:
    """Construit un dict Einher standardise."""
    return {
        "name": name,
        "domain": domain,
        "direction": direction,
        "timeframes": [tf],
        "trigger": trigger,
        "filters": [{"expr": f} for f in filters],
        "assets": "all",
        "tp_rule": {"type": "atr_multiple", "value": tp_mult},
        "sl_rule": {"type": "atr_multiple", "value": sl_mult},
        "max_holding": "1d" if tf in ("5m", "15m") else "3d",
        "cooldown": "4h",
    }


def _einher_id(domain: str, base_name: str, tf: str, idx: int) -> str:
    """Genere un nom unique d'Einher."""
    return f"E_{domain.upper().replace(' ', '_')}_{base_name}_{tf}_{idx}"


# ============================================================================
# GENERATION DES DOMAINES
# ============================================================================

def generate_corpus() -> list[dict]:
    """Genere le corpus brut complet avec vrais noms MIDAS."""
    einhers = []
    idx = 1

    # ------------------------------------------------------------------
    # DOMAINE 1 : Pattern pur (pattern seul)
    # ------------------------------------------------------------------
    for tf in TIMEFRAMES:
        for pat in PATTERNS[:50]:  # 50 patterns les plus connus
            for d in DIRECTIONS:
                # Certains patterns sont directionnels
                if d == "long" and any(x in pat for x in ["bear", "hanging", "shooting", "evening", "dark", "black", "top", "descending"]):
                    continue
                if d == "short" and any(x in pat for x in ["bull", "hammer", "morning", "white", "bottom", "ascending", "dragonfly", "piercing"]):
                    continue
                trigger = _pattern_expr(pat, d)
                einhers.append(_make_einher(
                    _einher_id("PATTERN", pat.replace("pattern_", "").upper(), tf, idx),
                    "Pattern pur", d, tf, trigger, [], 2.5, 1.5
                ))
                idx += 1

    # ------------------------------------------------------------------
    # DOMAINE 2 : Indicateur pur (indicateur seul en trigger)
    # ------------------------------------------------------------------
    for tf in TIMEFRAMES:
        for ind in INDICATORS:
            for d in DIRECTIONS:
                exprs = _indicator_trigger_exprs(ind, d)
                for logic_name, expr in exprs[:2]:  # max 2 expressions par indicateur
                    einhers.append(_make_einher(
                        _einher_id("INDICATOR", logic_name.upper(), tf, idx),
                        "Indicateur classique", d, tf, expr, [], 2.5, 1.5
                    ))
                    idx += 1

    # ------------------------------------------------------------------
    # DOMAINE 3 : Quantitatif pur (quant seul en trigger)
    # ------------------------------------------------------------------
    for tf in TIMEFRAMES:
        for q in QUANTS:
            for d in DIRECTIONS:
                exprs = _quant_trigger_exprs(q, d)
                for logic_name, expr in exprs[:2]:
                    einhers.append(_make_einher(
                        _einher_id("QUANT", logic_name.upper(), tf, idx),
                        "Quantitatif", d, tf, expr, [], 2.5, 1.5
                    ))
                    idx += 1

    # ------------------------------------------------------------------
    # DOMAINE 4 : Pattern + confluence (pattern + 1-2 filtres)
    # ------------------------------------------------------------------
    common_f = {d: _common_filters(d) for d in DIRECTIONS}
    for tf in TIMEFRAMES:
        for pat in PATTERNS[:30]:
            for d in DIRECTIONS:
                if d == "long" and any(x in pat for x in ["bear", "hanging", "shooting", "evening", "top"]):
                    continue
                if d == "short" and any(x in pat for x in ["bull", "hammer", "morning", "bottom"]):
                    continue
                trigger = _pattern_expr(pat, d)
                # Ajouter 1 filtre
                for flogic, fexpr in common_f[d][:8]:
                    einhers.append(_make_einher(
                        _einher_id("PAT_CONF", f"{pat.replace('pattern_', '').upper()}_{flogic.upper()}", tf, idx),
                        "Pattern + confluence", d, tf, trigger, [fexpr], 2.5, 1.5
                    ))
                    idx += 1
                # Ajouter 2 filtres
                for (fl1, fe1), (fl2, fe2) in itertools.combinations(common_f[d][:5], 2):
                    einhers.append(_make_einher(
                        _einher_id("PAT_CONF2", f"{pat.replace('pattern_', '').upper()}_{fl1.upper()}_{fl2.upper()}", tf, idx),
                        "Pattern + confluence", d, tf, trigger, [fe1, fe2], 2.5, 1.5
                    ))
                    idx += 1

    # ------------------------------------------------------------------
    # DOMAINE 5 : Indicateur + confluence (indicateur + quant/pattern)
    # ------------------------------------------------------------------
    for tf in TIMEFRAMES:
        for ind in INDICATORS[:25]:
            for d in DIRECTIONS:
                exprs = _indicator_trigger_exprs(ind, d)
                if not exprs:
                    continue
                trigger = exprs[0][1]
                # Filtre quant
                for q in ["quant_hurst_exponent", "quant_shannon_entropy", "quant_regime_detection", "quant_vol_persistence"]:
                    q_exprs = _quant_trigger_exprs(q, d)
                    if q_exprs:
                        einhers.append(_make_einher(
                            _einher_id("IND_CONF", f"{ind.upper()}_{q.upper()}", tf, idx),
                            "Indicateur + confluence", d, tf, trigger, [q_exprs[0][1]], 2.5, 1.5
                        ))
                        idx += 1
                # Filtre indicateur
                for flogic, fexpr in common_f[d][:4]:
                    einhers.append(_make_einher(
                        _einher_id("IND_CONF2", f"{ind.upper()}_{flogic.upper()}", tf, idx),
                        "Indicateur + confluence", d, tf, trigger, [fexpr], 2.5, 1.5
                    ))
                    idx += 1

    # ------------------------------------------------------------------
    # DOMAINE 6 : Quant + confluence (quant + indicateur)
    # ------------------------------------------------------------------
    for tf in TIMEFRAMES:
        for q in QUANTS[:15]:
            for d in DIRECTIONS:
                exprs = _quant_trigger_exprs(q, d)
                if not exprs:
                    continue
                trigger = exprs[0][1]
                # Filtre indicateur
                for flogic, fexpr in common_f[d][:5]:
                    einhers.append(_make_einher(
                        _einher_id("QUANT_CONF", f"{q.replace('quant_', '').upper()}_{flogic.upper()}", tf, idx),
                        "Quant + confluence", d, tf, trigger, [fexpr], 2.5, 1.5
                    ))
                    idx += 1
                # 2 filtres
                for (fl1, fe1), (fl2, fe2) in itertools.combinations(common_f[d][:3], 2):
                    einhers.append(_make_einher(
                        _einher_id("QUANT_CONF2", f"{q.replace('quant_', '').upper()}_{fl1.upper()}_{fl2.upper()}", tf, idx),
                        "Quant + confluence", d, tf, trigger, [fe1, fe2], 2.5, 1.5
                    ))
                    idx += 1

    # ------------------------------------------------------------------
    # DOMAINE 7 : Multi-features mixtes (indicateur + quant, ou 2 indicateurs)
    # ------------------------------------------------------------------
    for tf in TIMEFRAMES:
        for d in DIRECTIONS:
            # RSI + Hurst
            einhers.append(_make_einher(
                _einher_id("MULTI", "RSI_HURST", tf, idx),
                "Multi-features", d, tf,
                "rsi_14 < 30" if d == "long" else "rsi_14 > 70",
                ["quant_hurst_exponent > 0.6" if d == "long" else "quant_hurst_exponent > 0.6", "adx_14 > 20"],
                2.5, 1.5
            ))
            idx += 1

            # MACD + Volume
            einhers.append(_make_einher(
                _einher_id("MULTI", "MACD_VOL", tf, idx),
                "Multi-features", d, tf,
                "macd_histogram > 0" if d == "long" else "macd_histogram < 0",
                ["volume_ratio > 1.5", "adx_14 > 20"],
                2.5, 1.5
            ))
            idx += 1

            # EMA + BB
            einhers.append(_make_einher(
                _einher_id("MULTI", "EMA_BB", tf, idx),
                "Multi-features", d, tf,
                "ema_12 > ema_26" if d == "long" else "ema_12 < ema_26",
                ["bb_percent < 0.1" if d == "long" else "bb_percent > 0.9", "close > vwap" if d == "long" else "close < vwap"],
                2.5, 1.5
            ))
            idx += 1

            # Stoch + RSI
            einhers.append(_make_einher(
                _einher_id("MULTI", "STOCH_RSI", tf, idx),
                "Multi-features", d, tf,
                "stoch_k < 20" if d == "long" else "stoch_k > 80",
                ["rsi_14 < 35" if d == "long" else "rsi_14 > 65", "adx_14 > 15"],
                2.5, 1.5
            ))
            idx += 1

            # Quant regime + EMA
            einhers.append(_make_einher(
                _einher_id("MULTI", "REGIME_EMA", tf, idx),
                "Multi-features", d, tf,
                "quant_regime_detection > 0.5" if d == "long" else "quant_regime_detection < -0.5",
                ["ema_50 > ema_200" if d == "long" else "ema_50 < ema_200", "volume_ratio > 1.2"],
                2.5, 1.5
            ))
            idx += 1

            # ATR expansion + momentum
            einhers.append(_make_einher(
                _einher_id("MULTI", "ATR_MOM", tf, idx),
                "Multi-features", d, tf,
                "momentum_10 > 0" if d == "long" else "momentum_10 < 0",
                ["atr_14 > atr_21", "quant_vol_persistence > 0.3"],
                2.5, 1.5
            ))
            idx += 1

            # VWAP + Supertrend
            einhers.append(_make_einher(
                _einher_id("MULTI", "VWAP_ST", tf, idx),
                "Multi-features", d, tf,
                "close > vwap" if d == "long" else "close < vwap",
                ["supertrend_signal == 1" if d == "long" else "supertrend_signal == -1", "adx_14 > 20"],
                2.5, 1.5
            ))
            idx += 1

            # Williams R + volume
            einhers.append(_make_einher(
                _einher_id("MULTI", "WILLIAMS_VOL", tf, idx),
                "Multi-features", d, tf,
                "williams_r < -80" if d == "long" else "williams_r > -20",
                ["volume_ratio > 1.3", "bb_width > 0.01"],
                2.5, 1.5
            ))
            idx += 1

            # OBV + MACD
            einhers.append(_make_einher(
                _einher_id("MULTI", "OBV_MACD", tf, idx),
                "Multi-features", d, tf,
                "obv > obv_ema" if d == "long" else "obv < obv_ema",
                ["macd_histogram > 0" if d == "long" else "macd_histogram < 0", "rsi_14 > 30 AND rsi_14 < 70"],
                2.5, 1.5
            ))
            idx += 1

    return einhers


def main():
    einhers = generate_corpus()

    # Stats par domaine
    from collections import Counter
    domains = Counter(e["domain"] for e in einhers)
    directions = Counter(e["direction"] for e in einhers)

    print(f"Corpus brut genere : {len(einhers)} Einhers")
    print(f"\nPar domaine:")
    for d, c in domains.most_common():
        print(f"  {d}: {c}")
    print(f"\nPar direction:")
    for d, c in directions.most_common():
        print(f"  {d}: {c}")

    # Verifier que tous les triggers utilisent les vrais noms
    all_features = set(PATTERNS + INDICATORS + QUANTS + ["close", "volume", "open", "high", "low"])
    bad_exprs = []
    for e in einhers:
        trig = e["trigger"]
        if "col_" in trig:
            bad_exprs.append((e["name"], trig))

    if bad_exprs:
        print(f"\nATTENTION : {len(bad_exprs)} triggers contiennent encore 'col_'")
        for name, trig in bad_exprs[:5]:
            print(f"  {name}: {trig}")
    else:
        print("\nOK : Aucun trigger ne contient 'col_'")

    # Sauvegarde
    output = {
        "_comment": "Corpus BRUT v2 — vrais noms MIDAS, strategies diversifiees (pattern, indicateur, quant, mixes). NON calibre.",
        "meta": {
            "total_einhers": len(einhers),
            "domains": list(domains.keys()),
            "timeframes": TIMEFRAMES,
            "max_conditions_per_einher": 3,
            "feature_families": {
                "patterns": len(PATTERNS),
                "indicators": len(INDICATORS),
                "quants": len(QUANTS),
            }
        },
        "einhers": einhers,
    }

    out_path = Path("config/corpus_brut_v2.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\nSauvegarde : {out_path}")


if __name__ == "__main__":
    main()
