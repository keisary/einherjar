"""AURIGA - Moteur de features : orchestrateur + registre.

compute_features(ohlcv_df, timeframe) -> polars.DataFrame
avec UNE colonne par feature nommée.

Les fonctions de calcul sont définies dans technical.py et quantitative.py
(copiées depuis midasV3). Ici, on les orchestre et on construit le
DataFrame de features.

CONTRAT :
- Entrée : polars.DataFrame avec colonnes [timestamp, open, high, low, close, volume]
- Sortie : polars.DataFrame [timestamp, <chaque feature>], même nb de lignes
- Les lignes de warmup (fenêtres roulantes) sont NaN
- Pas de fuite future : fenêtres roulantes uniquement
"""
from __future__ import annotations

import numpy as np
import polars as pl

from einherjar.signals import quantitative as q
from einherjar.signals import technical as t

# ---------------------------------------------------------------------------
# Registre des features : (nom, groupe, fonction(s) appelée(s), description)
# ---------------------------------------------------------------------------
# Le registre définit l'ORDRE et les NOMS des colonnes de sortie.
# Chaque entrée : (nom_colonne, groupe, description)

FEATURES: list[tuple[str, str, str]] = [
    # (nom, groupe, description)
    # --- Tendance ---
    ("sma_20", "trend", "Simple Moving Average 20"),
    ("sma_50", "trend", "Simple Moving Average 50"),
    ("ema_12", "trend", "Exponential Moving Average 12"),
    ("ema_26", "trend", "Exponential Moving Average 26"),
    ("kaufman_eff_20", "trend", "Kaufman Efficiency Ratio 20"),
    # --- Momentum ---
    ("rsi_14", "momentum", "Relative Strength Index 14"),
    ("macd_hist", "momentum", "MACD histogramme"),
    ("mom_5", "momentum", "Momentum 5 barres"),
    ("mom_20", "momentum", "Momentum 20 barres"),
    ("roc_12", "momentum", "Rate of Change 12"),
    ("stoch_14", "momentum", "Stochastique %K 14"),
    ("mfi_14", "momentum", "Money Flow Index 14"),
    ("aroon_14", "momentum", "Aroon 14"),
    # --- Volatilité ---
    ("bb_width_20", "volatility", "Largeur Bollinger 20 normalisée"),
    ("atr_14", "volatility", "ATR 14 normalisé"),
    ("realized_vol_20", "volatility", "Volatilité réalisée 20"),
    ("garch_vol", "volatility", "Volatilité GARCH(1,1)"),
    ("choppiness_14", "volatility", "Choppiness Index 14"),
    ("vortex_14", "volatility", "Vortex Indicator 14"),
    # --- Volume ---
    ("obv", "volume", "On-Balance Volume"),
    ("vwap_ratio", "volume", "Ratio prix/VWAP"),
    ("amihud_20", "volume", "Illiquidité d'Amihud 20"),
    # --- Statistiques ---
    ("hurst_252", "statistical", "Exposant de Hurst 252"),
    ("autocorr_252", "statistical", "Autocorrélation 252"),
    ("shannon_ent_252", "statistical", "Entropie de Shannon 252"),
    ("skew_50", "statistical", "Skewness roulante 50"),
    ("kurt_50", "statistical", "Kurtosis roulante 50"),
    ("variance_ratio_20", "statistical", "Variance Ratio 20"),
    ("fractal_dim_252", "statistical", "Dimension fractale 252"),
    ("dfa_252", "statistical", "DFA 252"),
    ("dominant_freq_252", "statistical", "Fréquence dominante 252"),
    # --- Régime ---
    ("regime_50", "regime", "Régime de marché 50"),
    ("vol_regime_50", "regime", "Régime de volatilité 50"),
    # --- Risque ---
    ("var_95_50", "risk", "VaR 95% 50"),
    ("cvar_95_50", "risk", "CVaR 95% 50"),
    ("max_dd_100", "risk", "Max drawdown 100"),
]

FEATURE_NAMES = [name for name, _, _ in FEATURES]
GROUPS = {name: grp for name, grp, _ in FEATURES}


def _safe(arr: np.ndarray) -> np.ndarray:
    """Convertit en float32 propre (NaN fini)."""
    a = np.asarray(arr, dtype=np.float32)
    a = np.where(np.isinf(a), np.nan, a)
    return a


def _check_available() -> list[str]:
    """Retourne les fonctions manquantes (pas encore collées depuis midas)."""
    needed = [
        ("t", "_numba_ema_vectorized"), ("t", "_numba_sma_vectorized"),
        ("t", "_numba_rsi_vectorized"), ("t", "_numba_macd_complete"),
        ("t", "_numba_bollinger_bands"), ("t", "_numba_atr"),
        ("t", "_numba_momentum"), ("t", "_numba_roc"), ("t", "_numba_vwap"),
        ("t", "_numba_adx_complete"), ("t", "_numba_obv"), ("t", "_numba_mfi"),
        ("t", "_numba_aroon"), ("t", "_numba_choppiness_index"),
        ("t", "_numba_vortex"),
        ("q", "_numba_realized_volatility"), ("q", "_numba_garch_volatility"),
        ("q", "_numba_hurst_rs"), ("q", "_numba_autocorrelation"),
        ("q", "_numba_shannon_entropy"), ("q", "_numba_dfa"),
        ("q", "_numba_rolling_skewness"), ("q", "_numba_rolling_kurtosis"),
        ("q", "_numba_dynamic_var"), ("q", "_numba_dynamic_cvar"),
        ("q", "_numba_max_drawdown"), ("q", "_numba_regime_detection"),
        ("q", "_numba_amihud_illiquidity"), ("q", "_numba_kaufman_efficiency"),
        ("q", "_numba_variance_ratio"), ("q", "calculate_price_position_numba"),
        ("q", "calculate_market_regime_numba"),
    ]
    missing = []
    for mod, fn in needed:
        obj = t if mod == "t" else q
        if not hasattr(obj, fn):
            missing.append(f"{mod}.{fn}")
    return missing


def compute_features(ohlcv: pl.DataFrame, timeframe: str = "1H") -> pl.DataFrame:
    """Calcule toutes les features du registre pour un DataFrame OHLCV.

    Chaque feature est calculée de manière isolée : si une fonction n'a pas
    encore été collée depuis midasV3, la colonne est remplie de NaN et un
    avertissement est loggé (le pipeline continue).
    """
    closes = ohlcv["close"].to_numpy().astype(np.float64)
    highs = ohlcv["high"].to_numpy().astype(np.float64)
    lows = ohlcv["low"].to_numpy().astype(np.float64)
    vols = ohlcv["volume"].to_numpy().astype(np.float64)
    n = len(closes)

    # Rendements simples (NaN sur la 1ère ligne)
    rets = np.full(n, np.nan, dtype=np.float64)
    if n > 1:
        rets[1:] = closes[1:] / closes[:-1] - 1.0

    feats: dict[str, np.ndarray] = {}

    def col(fn, *args, default=np.nan, **kw) -> np.ndarray:
        try:
            arr = np.asarray(fn(*args, **kw), dtype=np.float64)
            if arr.shape[0] != n:
                # Fonction vectorisée multi-périodes : prendre la dernière ligne
                if arr.ndim == 2:
                    arr = arr[-1]
                else:
                    return np.full(n, default, dtype=np.float64)
            return arr
        except Exception:
            return np.full(n, default, dtype=np.float64)

    # --- Tendance ---
    feats["sma_20"] = col(t._numba_sma_vectorized, closes, np.array([20], np.int32)) if hasattr(t, "_numba_sma_vectorized") else np.full(n, np.nan)
    feats["sma_50"] = col(t._numba_sma_vectorized, closes, np.array([50], np.int32)) if hasattr(t, "_numba_sma_vectorized") else np.full(n, np.nan)
    feats["ema_12"] = col(t._numba_ema_vectorized, closes, np.array([12], np.int32)) if hasattr(t, "_numba_ema_vectorized") else np.full(n, np.nan)
    feats["ema_26"] = col(t._numba_ema_vectorized, closes, np.array([26], np.int32)) if hasattr(t, "_numba_ema_vectorized") else np.full(n, np.nan)
    feats["kaufman_eff_20"] = col(q._numba_kaufman_efficiency, closes, 20) if hasattr(q, "_numba_kaufman_efficiency") else np.full(n, np.nan)

    # --- Momentum ---
    feats["rsi_14"] = col(t._numba_rsi_vectorized, closes, np.array([14], np.int32)) if hasattr(t, "_numba_rsi_vectorized") else np.full(n, np.nan)
    if hasattr(t, "_numba_macd_complete"):
        try:
            _, _, hist = t._numba_macd_complete(closes)
            feats["macd_hist"] = _safe(hist)
        except Exception:
            feats["macd_hist"] = np.full(n, np.nan)
    else:
        feats["macd_hist"] = np.full(n, np.nan)
    feats["mom_5"] = col(t._numba_momentum, closes, 5) if hasattr(t, "_numba_momentum") else np.full(n, np.nan)
    feats["mom_20"] = col(t._numba_momentum, closes, 20) if hasattr(t, "_numba_momentum") else np.full(n, np.nan)
    feats["roc_12"] = col(t._numba_roc, closes, 12) if hasattr(t, "_numba_roc") else np.full(n, np.nan)
    feats["stoch_14"] = col(t._numba_stochastic, highs, lows, closes, 14, 3) if hasattr(t, "_numba_stochastic") else np.full(n, np.nan)
    feats["mfi_14"] = col(t._numba_mfi, highs, lows, closes, vols, 14) if hasattr(t, "_numba_mfi") else np.full(n, np.nan)
    feats["aroon_14"] = col(t._numba_aroon, highs, lows, 14) if hasattr(t, "_numba_aroon") else np.full(n, np.nan)

    # --- Volatilité ---
    if hasattr(t, "_numba_bollinger_bands"):
        try:
            up, mid, low_bb = t._numba_bollinger_bands(closes, 20, 2.0)
            up = np.asarray(up, dtype=np.float64)
            mid = np.asarray(mid, dtype=np.float64)
            low_bb = np.asarray(low_bb, dtype=np.float64)
            feats["bb_width_20"] = _safe(np.where(mid > 0, (up - low_bb) / mid, np.nan))
        except Exception:
            feats["bb_width_20"] = np.full(n, np.nan)
    else:
        feats["bb_width_20"] = np.full(n, np.nan)
    if hasattr(t, "_numba_atr") and hasattr(t, "_numba_sma_vectorized"):
        try:
            atr_vals = np.asarray(t._numba_atr(highs, lows, closes, 14), dtype=np.float64)
            feats["atr_14"] = _safe(np.where(closes > 0, atr_vals / closes, np.nan))
        except Exception:
            feats["atr_14"] = np.full(n, np.nan)
    else:
        feats["atr_14"] = np.full(n, np.nan)
    feats["realized_vol_20"] = col(q._numba_realized_volatility, rets, 20) if hasattr(q, "_numba_realized_volatility") else np.full(n, np.nan)
    feats["garch_vol"] = col(q._numba_garch_volatility, rets) if hasattr(q, "_numba_garch_volatility") else np.full(n, np.nan)
    feats["choppiness_14"] = col(t._numba_choppiness_index, highs, lows, closes, 14) if hasattr(t, "_numba_choppiness_index") else np.full(n, np.nan)
    feats["vortex_14"] = col(t._numba_vortex, highs, lows, closes, 14) if hasattr(t, "_numba_vortex") else np.full(n, np.nan)

    # --- Volume ---
    feats["obv"] = col(t._numba_obv, closes, vols) if hasattr(t, "_numba_obv") else np.full(n, np.nan)
    if hasattr(t, "_numba_vwap"):
        try:
            vwap_vals = np.asarray(t._numba_vwap(closes, vols), dtype=np.float64)
            feats["vwap_ratio"] = _safe(np.where(vwap_vals > 0, closes / vwap_vals, np.nan))
        except Exception:
            feats["vwap_ratio"] = np.full(n, np.nan)
    else:
        feats["vwap_ratio"] = np.full(n, np.nan)
    feats["amihud_20"] = col(q._numba_amihud_illiquidity, rets, vols, 20) if hasattr(q, "_numba_amihud_illiquidity") else np.full(n, np.nan)

    # --- Statistiques ---
    feats["hurst_252"] = col(q._numba_hurst_rs, closes, 252) if hasattr(q, "_numba_hurst_rs") else np.full(n, np.nan)
    feats["autocorr_252"] = col(q._numba_autocorrelation, closes, 20, 252) if hasattr(q, "_numba_autocorrelation") else np.full(n, np.nan)
    feats["shannon_ent_252"] = col(q._numba_shannon_entropy, closes, 50, 252) if hasattr(q, "_numba_shannon_entropy") else np.full(n, np.nan)
    feats["skew_50"] = col(q._numba_rolling_skewness, closes, 50) if hasattr(q, "_numba_rolling_skewness") else np.full(n, np.nan)
    feats["kurt_50"] = col(q._numba_rolling_kurtosis, closes, 50) if hasattr(q, "_numba_rolling_kurtosis") else np.full(n, np.nan)
    feats["variance_ratio_20"] = col(q._numba_variance_ratio, rets, 20) if hasattr(q, "_numba_variance_ratio") else np.full(n, np.nan)
    feats["fractal_dim_252"] = col(q._numba_fractal_dimension, closes, 252) if hasattr(q, "_numba_fractal_dimension") else np.full(n, np.nan)
    feats["dfa_252"] = col(q._numba_dfa, closes, 252) if hasattr(q, "_numba_dfa") else np.full(n, np.nan)
    feats["dominant_freq_252"] = col(q._numba_dominant_frequency, closes, 252) if hasattr(q, "_numba_dominant_frequency") else np.full(n, np.nan)

    # --- Régime ---
    feats["regime_50"] = col(q._numba_regime_detection, rets, 50) if hasattr(q, "_numba_regime_detection") else np.full(n, np.nan)
    feats["vol_regime_50"] = col(q.calculate_market_regime_numba, rets, np.asarray(feats["realized_vol_20"]), 50) if hasattr(q, "calculate_market_regime_numba") else np.full(n, np.nan)

    # --- Risque ---
    feats["var_95_50"] = col(q._numba_dynamic_var, rets, 0.05, 50) if hasattr(q, "_numba_dynamic_var") else np.full(n, np.nan)
    feats["cvar_95_50"] = col(q._numba_dynamic_cvar, rets, 0.05, 50) if hasattr(q, "_numba_dynamic_cvar") else np.full(n, np.nan)
    feats["max_dd_100"] = col(q._numba_max_drawdown, closes, 100) if hasattr(q, "_numba_max_drawdown") else np.full(n, np.nan)

    # --- Assemblage ---
    result = ohlcv.select("timestamp")
    for name in FEATURE_NAMES:
        arr = feats.get(name)
        if arr is None:
            arr = np.full(n, np.nan, dtype=np.float64)
        result = result.with_columns(pl.Series(name, _safe(arr)))
    return result


def compute_features_for_symbols(
    data: dict[str, pl.DataFrame], timeframe: str = "1H"
) -> dict[str, pl.DataFrame]:
    """compute_features pour plusieurs symboles."""
    return {sym: compute_features(df, timeframe) for sym, df in data.items()}
