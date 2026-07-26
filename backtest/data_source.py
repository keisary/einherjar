"""
Chargement des donnees MIDAS pour le backtest EINHERJAR.

Reconstruction des prix absolus depuis X.npy normalise.
Les 5 premieres colonnes de X.npy sont OHLCV (normalise en log-returns
a partir de la ligne 1). Le mapping des features suit les noms MIDAS.
"""

import json
import numpy as np
import polars as pl
from pathlib import Path
from typing import Optional

MIDAS_ROOT = Path("D:/midas_v2/midasV3/src/data/compiled")
EINHERJAR_ROOT = Path(__file__).resolve().parent.parent

# Mapping noms corpus courts -> noms MIDAS complets
# Utilise defensivement si un nom sans prefixe est rencontre.
PATTERN_MAP = {
    "hammer": "pattern_hammer",
    "inverted_hammer": "pattern_inverted_hammer",
    "dragonfly_doji": "pattern_dragonfly_doji",
    "morning_star": "pattern_morning_star",
    "piercing_line": "pattern_piercing_line",
    "three_white_soldiers": "pattern_three_white_soldiers",
    "engulfing_bull": "pattern_engulfing_bull",
    "engulfing_bear": "pattern_engulfing_bear",
    "harami_bull": "pattern_harami_bull",
    "harami_bear": "pattern_harami_bear",
    "pin_bar_bull": "pattern_pin_bar_bull",
    "pin_bar_bear": "pattern_pin_bar_bear",
    "marubozu_bull": "pattern_marubozu_bull",
    "marubozu_bear": "pattern_marubozu_bear",
    "doji": "pattern_doji",
    "long_legged_doji": "pattern_long_legged_doji",
    "spinning_top": "pattern_spinning_top",
    "hanging_man": "pattern_hanging_man",
    "shooting_star": "pattern_shooting_star",
    "gravestone_doji": "pattern_gravestone_doji",
    "evening_star": "pattern_evening_star",
    "dark_cloud_cover": "pattern_dark_cloud_cover",
    "three_black_crows": "pattern_three_black_crows",
    "double_top": "pattern_double_top",
    "double_bottom": "pattern_double_bottom",
    "triple_top": "pattern_triple_top",
    "triple_bottom": "pattern_triple_bottom",
    "head_shoulders": "pattern_head_shoulders",
    "inv_head_shoulders": "pattern_inv_head_shoulders",
    "rounding_top": "pattern_rounding_top",
    "rounding_bottom": "pattern_rounding_bottom",
    "ascending_triangle": "pattern_ascending_triangle",
    "descending_triangle": "pattern_descending_triangle",
    "symmetrical_triangle": "pattern_symmetrical_triangle",
    "rectangle": "pattern_rectangle",
    "bull_flag": "pattern_bull_flag",
    "bear_flag": "pattern_bear_flag",
    "bull_pennant": "pattern_bull_pennant",
    "bear_pennant": "pattern_bear_pennant",
    "rising_wedge": "pattern_rising_wedge",
    "falling_wedge": "pattern_falling_wedge",
    "cup_handle": "pattern_cup_handle",
    "channel_up": "pattern_channel_up",
    "channel_down": "pattern_channel_down",
    "support": "pattern_support",
    "resistance": "pattern_resistance",
    "uptrend": "pattern_uptrend",
    "downtrend": "pattern_downtrend",
    "sideways": "pattern_sideways_trend",
    "gap_up": "pattern_gap_up",
    "gap_down": "pattern_gap_down",
    "gap_fill": "pattern_gap_fill",
    "breakaway_gap": "pattern_breakaway_gap",
    "runaway_gap": "pattern_runaway_gap",
    "exhaustion_gap": "pattern_exhaustion_gap",
    "island_reversal": "pattern_island_reversal",
    "gap_and_go": "pattern_gap_and_go",
    "three_drives": "pattern_three_drives",
    "wolfe_wave": "pattern_wolfe_wave",
    "elliott_wave_1": "pattern_elliott_wave_1",
    "elliott_wave_3": "pattern_elliott_wave_3",
    "elliott_wave_5": "pattern_elliott_wave_5",
    "fibonacci_retracement": "pattern_fibonacci_retracement",
    "fibonacci_extension": "pattern_fibonacci_extension",
    "diamond_top": "pattern_diamond_top",
    "diamond_bottom": "pattern_diamond_bottom",
    "v_top": "pattern_v_top",
    "v_bottom": "pattern_v_bottom",
    "island_top": "pattern_island_top",
    "island_bottom": "pattern_island_bottom",
    "spike_reversal": "pattern_spike_reversal",
    "broadening_wedge": "pattern_broadening_wedge",
    "measured_move": "pattern_measured_move",
    "gartley_bull": "pattern_gartley_bull",
    "gartley_bear": "pattern_gartley_bear",
    "butterfly_bull": "pattern_butterfly_bull",
    "butterfly_bear": "pattern_butterfly_bear",
    "bat_bull": "pattern_bat_bull",
    "bat_bear": "pattern_bat_bear",
    "crab_bull": "pattern_crab_bull",
    "crab_bear": "pattern_crab_bear",
    "shark_bull": "pattern_shark_bull",
    "shark_bear": "pattern_shark_bear",
    "three_inside_up": "pattern_three_inside_up",
    "three_inside_down": "pattern_three_inside_down",
    "three_outside_up": "pattern_three_outside_up",
    "three_outside_down": "pattern_three_outside_down",
    "advance_block": "pattern_advance_block",
    "deliberation": "pattern_deliberation",
    "belt_hold_bull": "pattern_belt_hold_bull",
    "belt_hold_bear": "pattern_belt_hold_bear",
    "kicking_bull": "pattern_kicking_bull",
    "kicking_bear": "pattern_kicking_bear",
    "matching_low": "pattern_matching_low",
    "matching_high": "pattern_matching_high",
    "ladder_bottom": "pattern_ladder_bottom",
    "ladder_top": "pattern_ladder_top",
    "breakaway_bull": "pattern_breakaway_bull",
    "breakaway_bear": "pattern_breakaway_bear",
    "abandoned_baby_bull": "pattern_abandoned_baby_bull",
    "abandoned_baby_bear": "pattern_abandoned_baby_bear",
    "concealing_baby_swallow": "pattern_concealing_baby_swallow",
    "unique_three_river_bottom": "pattern_unique_three_river_bottom",
    "rickshaw_man": "pattern_rickshaw_man",
    "high_wave_candle": "pattern_high_wave_candle",
    "tri_star": "pattern_tri_star",
    "four_price_doji": "pattern_four_price_doji",
}

QUANT_MAP = {
    "quant_hurst": "quant_hurst_exponent",
    "quant_entropy": "quant_shannon_entropy",
    "quant_fractal": "quant_fractal_dimension",
    "quant_regime": "quant_regime_detection",
    "quant_vol_persist": "quant_vol_persistence",
    "quant_autocorr_10": "quant_autocorr_10",
    "quant_autocorr_20": "quant_autocorr_20",
    "quant_autocorr_50": "quant_autocorr_50",
    "quant_skew": "quant_rolling_skewness",
    "quant_kurt": "quant_rolling_kurtosis",
    "quant_var": "quant_dynamic_var",
    "quant_cvar": "quant_dynamic_cvar",
    "quant_maxdd": "quant_max_drawdown",
    "quant_amihud": "quant_amihud_illiquidity",
    "quant_kyles": "quant_kyles_lambda",
    "quant_kaufman": "quant_kaufman_efficiency",
    "quant_variance_ratio": "quant_variance_ratio",
    "quant_dfa": "quant_dfa_exponent",
    "quant_sample_entropy": "quant_sample_entropy",
    "quant_perm_entropy": "quant_permutation_entropy",
    "quant_approx_entropy": "quant_approximate_entropy",
    "quant_dominant_freq": "quant_dominant_frequency",
    "quant_spectral_centroid": "quant_spectral_centroid",
    "quant_realized_vol_10": "quant_realized_vol_10",
    "quant_realized_vol_20": "quant_realized_vol_20",
    "quant_realized_vol_50": "quant_realized_vol_50",
    "quant_garch": "quant_garch_volatility",
    "quant_vol_clustering": "quant_vol_clustering",
}


def map_feature_name(corpus_name: str) -> str:
    """Convertit un nom de feature du corpus en nom MIDAS.

    Le corpus v2 utilise deja les vrais noms MIDAS. Cette fonction reste
    defensive pour supporter d'eventuels residus `col_` ou noms courts.
    """
    if corpus_name.startswith("col_"):
        key = corpus_name[4:]
        if key in PATTERN_MAP:
            return PATTERN_MAP[key]
        if key in QUANT_MAP:
            return QUANT_MAP[key]
        return f"pattern_{key}"
    if corpus_name in QUANT_MAP:
        return QUANT_MAP[corpus_name]
    return corpus_name


def load_ohlcv(asset: str, tf: str, asset_class: str) -> Optional[pl.DataFrame]:
    """
    Charge X.npy et ts.npy pour un actif/timeframe.
    Reconstruit les prix absolus OHLCV depuis la normalisation log-return.
    Retourne un DataFrame polars avec toutes les features nommees.
    """
    base = MIDAS_ROOT / asset_class / tf
    x_path = base / f"{asset}_X.npy"
    ts_path = base / f"{asset}_ts.npy"
    meta_path = base / "metadata.json"

    if not x_path.exists() or not ts_path.exists():
        return None

    X = np.load(x_path)
    ts = np.load(ts_path)

    if X.shape[0] == 0:
        return None

    # Reconstruction prix absolus
    o0, h0, l0, c0 = float(X[0, 0]), float(X[0, 1]), float(X[0, 2]), float(X[0, 3])
    v0 = float(X[0, 4])

    # Cumsum des log-returns (rows 1+)
    log_rets = np.vstack([np.zeros((1, 4), dtype=np.float64), X[1:, :4].astype(np.float64)])
    cum_log = np.cumsum(log_rets, axis=0)

    opens = np.exp(cum_log[:, 0] + np.log(max(o0, 1e-12)))
    highs = np.exp(cum_log[:, 1] + np.log(max(h0, 1e-12)))
    lows = np.exp(cum_log[:, 2] + np.log(max(l0, 1e-12)))
    closes = np.exp(cum_log[:, 3] + np.log(max(c0, 1e-12)))
    volumes = np.expm1(X[:, 4].astype(np.float64))

    df = pl.DataFrame({
        "timestamp": ts.astype(np.int64),
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    })

    # Charger noms features depuis metadata
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        feature_names = meta.get("feature_names", [])
        if len(feature_names) == X.shape[1]:
            for i, name in enumerate(feature_names[5:], start=5):
                df = df.with_columns(pl.Series(name, X[:, i]))

    del X, ts
    return df


def build_feature_map(df: pl.DataFrame) -> dict[str, pl.Series]:
    """Indexe les colonnes du DataFrame par nom pour acces rapide."""
    return {name: df[name] for name in df.columns}
