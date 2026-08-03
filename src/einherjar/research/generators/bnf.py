"""generators/bnf.py — Grammaire BNF pour la recherche grammaticale (GE).

⚠ Chantier BNF en DERNIER (par decision user). Ce module genere
dynamiquement les terminaux de la grammaire BNF a partir de :
  - features_taxonomy.json : 218 features utilisables, classes par type/famille
  - threshold_calibration.py : quantiles par feature sur le train (P1 #1)

Le pipeline GE :
  [genome d'entiers] -> [BNF + parser] -> [Condition | ConditionNode]
                                       -> [Hypothesis] -> [Evaluation]

Conventions :
  - Chaque feature a sa propre grammaire concue individuellement (procedure
    "Identifier -> Comprendre -> Concevoir -> Verifier -> Valider -> Ecrire").
  - Les regles specifiques a chaque feature sont codees dans
    `FEATURE_GRAMMARS` (override du pattern par defaut).
  - Le pattern par defaut (`_default_atomic_grammar`) couvre les features
    atomiques standard : "<feat> <op> <quantile>".

Statut :
  - 5/218 features OHLCV (Lot 0) : open, high, low, close, volume
    (famille price_action + volume_flow, log-returns / log1p).
  - 49/218 indicateurs techniques (Lot 1) : RSI, MACD, SMA, EMA, ATR,
    Stochastic, Williams %R, CCI, ROC, Momentum, Ultimate Oscillator,
    ADX, DI+/DI-, Aroon, Parabolic SAR, Supertrend, TRIX, Vortex,
    Bollinger Bands (5), OBV (2), volume_sma_20, volume_ratio,
    Chaikin Oscillator, Money Flow Index.
  - 31/218 features QUANTITATIVES (Lot 2) : entropies (4), autocorrelations
    (6), volatilite (5), risque (5), regime de marche (4), momentum quant (2),
    microstructure (2), spectral (2), variance ratio (1).
    NOTE : 4 features quantitative `_signal` (quant_hurst_exponent_signal,
    quant_realized_vol_10_signal, quant_shannon_entropy_signal,
    quant_vol_persistence_signal) sont dans la taxonomie mais marquees
    excluded=True (fantomes), donc 31/35 traitees.
  - 26/218 features (Lot 3) : 4 atomic restants (choppiness_index,
    kurtosis_risk, skewness_risk, vwap) + 9 composite_derived signaux
    (adx/aroon/bb_width/macd/obv/supertrend/volume_ratio/volume_sma/vwap
    _signal) + 13 factors (scores agreges en [0, 1]).
  - 1 bloc de relations OHLCV : `OHLCV_RELATIONS_GRAMMAR` (traitement
    conjoint open/high/low/close/volume, e.g. "close > open").
  - Reste : 107 features `pattern` (gros morceau, BNF specialisee).
"""

from __future__ import annotations

import logging
from typing import Any

from einherjar.research.config.loader import EinherjarConfig
from einherjar.research.utils.types import CompareOp

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Operateurs logiques
# --------------------------------------------------------------------------- #

# Operateurs de comparaison valides pour les features atomiques standard.
# (==, !=, IN) sont exclus car peu informatifs pour des series continues
# (log-returns, log1p, indicateurs techniques bornes ou non).
DEFAULT_ATOMIC_OPERATORS: tuple[CompareOp, ...] = (
    CompareOp.GT, CompareOp.LT, CompareOp.GE, CompareOp.LE,
)


# --------------------------------------------------------------------------- #
# Quantiles utilises comme seuils
# --------------------------------------------------------------------------- #

# Quantiles par defaut (cf. data/threshold_calibration.py).
DEFAULT_QUANTILES: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 0.90)


# --------------------------------------------------------------------------- #
# Pattern par defaut (features atomiques standards)
# --------------------------------------------------------------------------- #


def _default_atomic_grammar(feature_name: str) -> str:
    """Genere la grammaire BNF par defaut pour une feature atomique.

    Pattern : "<feat> <op> <quantile>" avec op dans {>, <, >=, <=}
    et quantile dans {p10, p25, p50, p75, p90}.

    Valide pour les features ou le seuil a un sens continu (momentum, RSI,
    indicateurs techniques, etc.). Les features specifiques (binary,
    categorical, pattern, factor) necessitent un override dans FEATURE_GRAMMARS.
    """
    ops = " | ".join(f'"{op.value}"' for op in DEFAULT_ATOMIC_OPERATORS)
    thresholds = " | ".join(
        f"q_{feature_name}_p{int(q * 100)}" for q in DEFAULT_QUANTILES
    )
    return (
        f'<{feature_name}_cond> ::= "{feature_name}" <{feature_name}_op> <{feature_name}_threshold>\n'
        f'<{feature_name}_op> ::= {ops}\n'
        f'<{feature_name}_threshold> ::= {thresholds}'
    )


# --------------------------------------------------------------------------- #
# Grammaires specifiques : 5 features OHLCV (traitees conjointement)
# --------------------------------------------------------------------------- #
#
# Convention MIDAS V3 (cf. data/npy_real_loader.py) :
#   - open  = log(open[t])  - log(close[t-1])   (log-return d'ouverture / gap overnight)
#   - high  = log(high[t])  - log(open[t])      (amplitude haute intra-bougie)
#   - low   = log(low[t])   - log(open[t])      (amplitude basse intra-bougie)
#   - close = log(close[t]) - log(open[t])      (rendement de la bougie)
#   - volume = log1p(volume[t])                 (volume transforme log)
#
# Toutes les series OHLCV sont continues, ~0 centrees, donc operateurs
# de comparaison (>, <, >=, <=) uniquement, et seuils = quantiles train
# (P1 #1 : pas d'uniformes -2..2).


# Feature 1/218 : "open" (log-return d'ouverture / gap overnight).
#   - Identifier  : open, type=atomic, value_type=float, family=price_action
#   - Comprendre  : log(open[t]) - log(close[t-1]). Capte le gap overnight
#     (positif = gap up, negatif = gap down). En crypto 24/7 c'est souvent
#     proche de zero (peu de gap structurels) ; reste utile sur actions/forex.
#   - Concevoir   : pattern atomique standard. Operateurs : >, <, >=, <=.
#     Seuils : quantiles p10/p25/p50/p75/p90 du log-return d'ouverture sur le train.
#   - Verifier    : pas de tautologie (log-returns centres sur 0, pas de open > 0).
#   - Valider     : OK.
OPEN_GRAMMAR: str = (
    '<open_cond>     ::= "open" <open_op> <open_threshold>\n'
    '<open_op>        ::= ">" | "<" | ">=" | "<="\n'
    '<open_threshold> ::= q_open_p10 | q_open_p25 | q_open_p50 | q_open_p75 | q_open_p90'
)


# Feature 2/218 : "high" (amplitude haute intra-bougie).
#   - Identifier  : high, type=atomic, value_type=float, family=price_action
#   - Comprendre  : log(high[t]) - log(open[t]). Toujours >= 0 (le plus haut
#     est forcement au-dessus de l'ouverture). Mesure la volatilite intra-bougie
#     vers le haut (mecche haute).
#   - Concevoir   : pattern atomique standard. Operateurs : >, <, >=, <=
#     (mais vu que toujours >= 0, "< 0" sera toujours False — laisse
#     quand meme dans la grammaire, c'est au moteur d'eliminer via admission).
#     Seuils : quantiles p10/p25/p50/p75/p90 sur le train.
#   - Verifier    : pas de tautologie (quantiles strictement > 0 en pratique).
#   - Valider     : OK.
HIGH_GRAMMAR: str = (
    '<high_cond>     ::= "high" <high_op> <high_threshold>\n'
    '<high_op>        ::= ">" | "<" | ">=" | "<="\n'
    '<high_threshold> ::= q_high_p10 | q_high_p25 | q_high_p50 | q_high_p75 | q_high_p90'
)


# Feature 3/218 : "low" (amplitude basse intra-bougie).
#   - Identifier  : low, type=atomic, value_type=float, family=price_action
#   - Comprendre  : log(low[t]) - log(open[t]). Toujours <= 0 (le plus bas
#     est forcement sous l'ouverture). Mesure la volatilite intra-bougie
#     vers le bas (mecche basse).
#   - Concevoir   : pattern atomique standard. Operateurs : >, <, >=, <=.
#     Seuils : quantiles p10/p25/p50/p75/p90 sur le train.
#   - Verifier    : pas de tautologie.
#   - Valider     : OK.
LOW_GRAMMAR: str = (
    '<low_cond>     ::= "low" <low_op> <low_threshold>\n'
    '<low_op>        ::= ">" | "<" | ">=" | "<="\n'
    '<low_threshold> ::= q_low_p10 | q_low_p25 | q_low_p50 | q_low_p75 | q_low_p90'
)


# Feature 4/218 : "close" (rendement de la bougie).
#   - Identifier  : close, type=atomic, value_type=float, family=price_action
#   - Comprendre  : log(close[t]) - log(open[t]). C'est LE rendement de la
#     bougie (la variation de l'ouverture a la fermeture). Signe = direction,
#     valeur absolue = amplitude. Feature la plus chargee semantiquement.
#   - Concevoir   : pattern atomique standard. Operateurs : >, <, >=, <=.
#     Seuils : quantiles p10/p25/p50/p75/p90 sur le train.
#   - Verifier    : pas de tautologie.
#   - Valider     : OK.
CLOSE_GRAMMAR: str = (
    '<close_cond>     ::= "close" <close_op> <close_threshold>\n'
    '<close_op>        ::= ">" | "<" | ">=" | "<="\n'
    '<close_threshold> ::= q_close_p10 | q_close_p25 | q_close_p50 | q_close_p75 | q_close_p90'
)


# Feature 5/218 : "volume" (log1p du volume de la bougie).
#   - Identifier  : volume, type=atomic, value_type=float, family=volume_flow
#   - Comprendre  : log1p(volume[t]). Toujours >= 0 (log1p(x) >= 0 pour x >= 1).
#     Transformation log pour reduire la skewness du volume brut (tres
#     asymmetrique : peu de barres avec tres gros volume, beaucoup avec peu).
#   - Concevoir   : pattern atomique standard. Operateurs : >, <, >=, <=.
#     Seuils : quantiles p10/p25/p50/p75/p90 sur le train.
#   - Verifier    : pas de tautologie (log1p > 0 presque toujours sauf volume nul).
#   - Valider     : OK.
#   - NOTE        : volume est dans un espace different des log-returns OHLCV
#     (log1p vs log-return) ; les seuils sont calibres separement par feature.
VOLUME_GRAMMAR: str = (
    '<volume_cond>     ::= "volume" <volume_op> <volume_threshold>\n'
    '<volume_op>        ::= ">" | "<" | ">=" | "<="\n'
    '<volume_threshold> ::= q_volume_p10 | q_volume_p25 | q_volume_p50 | q_volume_p75 | q_volume_p90'
)


# --------------------------------------------------------------------------- #
# Lot 1 : indicateurs techniques (atomic) — famille momentum / trend /
#         volatility / volume_flow
# --------------------------------------------------------------------------- #
#
# Procedure stricte (Identifier -> Comprendre -> Concevoir -> Verifier ->
# Valider -> Ecrire) appliquee pour chaque feature ci-dessous. Le pattern
# par defaut `_default_atomic_grammar` fonctionne pour la majorite d'entre
# elles (continuous, single-featured, sens monotonie clair). Les 3 RSI et
# les 4 SMA/EMA ont des periodes differentes mais une semantique strictement
# identique, donc on les groupe par un helper explicite.
#
# Bornes par famille :
#   - momentum (oscillateurs) : bornes connues [a, b] (RSI [0,100], Stoch [0,100],
#     Williams_R [-100, 0], CCI non borne, ROC % non borne).
#   - trend (moyennes) : pas de borne theorique, mais coherentes avec le prix.
#   - volatility (ATR, BB) : ATR >= 0, BB_percent [0,1], BB_width >= 0.
#   - volume_flow : volume-derived, >= 0.


def _oscillator_grammar(feature_name: str, period: int) -> str:
    """Helper pour oscillateurs bornes type RSI / Stochastic.

    Args:
        feature_name: nom exact dans la taxonomie (ex: "rsi_14").
        period: periode de calcul (info documentaire uniquement).
    """
    ops = " | ".join(f'"{op.value}"' for op in DEFAULT_ATOMIC_OPERATORS)
    thresholds = " | ".join(
        f"q_{feature_name}_p{int(q * 100)}" for q in DEFAULT_QUANTILES
    )
    return (
        f'# {feature_name} : oscillateur sur {period} periodes '
        f'(borne par construction, quantiles dans [0, 100] en pratique)\n'
        f'<{feature_name}_cond> ::= "{feature_name}" <{feature_name}_op> <{feature_name}_threshold>\n'
        f'<{feature_name}_op> ::= {ops}\n'
        f'<{feature_name}_threshold> ::= {thresholds}'
    )


def _unit_bounded_grammar(feature_name: str, lower: float, upper: float) -> str:
    """Helper pour features dans [lower, upper] (Hurst, Kaufman, regime, etc.).

    Args:
        feature_name: nom exact dans la taxonomie (ex: "quant_hurst_exponent").
        lower: borne inferieure theorique.
        upper: borne superieure theorique.
    """
    ops = " | ".join(f'"{op.value}"' for op in DEFAULT_ATOMIC_OPERATORS)
    thresholds = " | ".join(
        f"q_{feature_name}_p{int(q * 100)}" for q in DEFAULT_QUANTILES
    )
    return (
        f'# {feature_name} : feature bornee dans [{lower}, {upper}]\n'
        f'<{feature_name}_cond> ::= "{feature_name}" <{feature_name}_op> <{feature_name}_threshold>\n'
        f'<{feature_name}_op> ::= {ops}\n'
        f'<{feature_name}_threshold> ::= {thresholds}'
    )


def _correlation_grammar(feature_name: str, lag: int | None = None) -> str:
    """Helper pour autocorrelations / correlations (bornee [-1, +1]).

    Args:
        feature_name: nom exact dans la taxonomie.
        lag: lag de l'autocorrelation (info documentaire, ex: 10/20/50).
    """
    lag_info = f"lag {lag}" if lag is not None else "correlation"
    ops = " | ".join(f'"{op.value}"' for op in DEFAULT_ATOMIC_OPERATORS)
    thresholds = " | ".join(
        f"q_{feature_name}_p{int(q * 100)}" for q in DEFAULT_QUANTILES
    )
    return (
        f'# {feature_name} : {lag_info}, borne dans [-1, +1]\n'
        f'<{feature_name}_cond> ::= "{feature_name}" <{feature_name}_op> <{feature_name}_threshold>\n'
        f'<{feature_name}_op> ::= {ops}\n'
        f'<{feature_name}_threshold> ::= {thresholds}'
    )


def _signal_grammar(feature_name: str, values: tuple[int, ...]) -> str:
    """Helper pour signaux discrets (binarises ou trinaires).

    Args:
        feature_name: nom exact dans la taxonomie (ex: "obv_signal").
        values: valeurs discretes prises par le signal (ex: (0, 1) ou (-1, 0, 1)).

    Note: les terminaux sont nommes `v_<feat>_<N>` (vs `q_<feat>_pX` pour
    les quantiles continus) pour distinguer valeur discrete vs quantile.
    """
    ops = " | ".join(f'"{op.value}"' for op in DEFAULT_ATOMIC_OPERATORS)
    thresholds = " | ".join(f"v_{feature_name}_{v}" for v in values)
    return (
        f'# {feature_name} : signal discret dans {values}\n'
        f'<{feature_name}_cond> ::= "{feature_name}" <{feature_name}_op> <{feature_name}_threshold>\n'
        f'<{feature_name}_op> ::= {ops}\n'
        f'<{feature_name}_threshold> ::= {thresholds}'
    )


# --- Sous-famille : Relative Strength Index (RSI) --- #
# Feature 6/218 : "rsi_14" (Welles Wilder 1978, periode 14).
#   - Identifier  : rsi_14, type=atomic, value_type=float, family=momentum
#   - Comprendre  : RSI = 100 - 100/(1+RS) avec RS = avg_gains(14)/avg_losses(14).
#     Borne [0, 100]. > 70 = surachete, < 30 = survendu, ~50 = neutre.
#   - Concevoir   : pattern oscillateur borne. Operateurs : >, <, >=, <=.
#     Seuils : quantiles p10/p25/p50/p75/p90 (entre ~20 et ~80 en pratique).
#   - Verifier    : pas de tautologie aux bornes (0 et 100 jamais atteints
#     sauf cas extremes, et q_pX ne sera pas 0 ou 100).
#   - Valider     : OK.
# Feature 7/218 : "rsi_21" (periode 21, plus long terme).
#   - Meme semantique, N=21 -> moins de bruit, plus de retard.
# Feature 8/218 : "rsi_30" (periode 30, encore plus long terme).
#   - Meme semantique, N=30 -> signal encore plus lisse.
RSI_14_GRAMMAR: str = _oscillator_grammar("rsi_14", 14)
RSI_21_GRAMMAR: str = _oscillator_grammar("rsi_21", 21)
RSI_30_GRAMMAR: str = _oscillator_grammar("rsi_30", 30)


# --- Sous-famille : Moving Average Convergence Divergence (MACD) --- #
# Feature 9/218 : "macd_line" (Gerald Appel, difference EMA12 - EMA26).
#   - Identifier  : macd_line, type=atomic, value_type=float, family=trend
#   - Comprendre  : EMA(close, 12) - EMA(close, 26). Oscillateur non borne
#     centre sur 0, capte le momentum court terme vs long terme. > 0 =
#     momentum haussier (court terme au-dessus du long terme), < 0 = baissier.
#   - Concevoir   : pattern atomique (log-return-like, ~0 centre).
#     Operateurs : >, <, >=, <=. Seuils : quantiles train.
#   - Verifier    : pas de tautologie (centree sur 0, pas de macd_line > 0).
#   - Valider     : OK.
# Feature 10/218 : "macd_signal" (EMA9 de macd_line, "ligne de signal").
#   - Meme semantique, plus lisse. Croisement macd_line vs macd_signal =
#     signal de trading classique (mais gere via relation, pas ici).
# Feature 11/218 : "macd_histogram" (macd_line - macd_signal, "histogramme").
#   - Identique a la difference des deux, captent la force du mouvement.
MACD_LINE_GRAMMAR: str = _default_atomic_grammar("macd_line")
MACD_SIGNAL_GRAMMAR: str = _default_atomic_grammar("macd_signal")
MACD_HISTOGRAM_GRAMMAR: str = _default_atomic_grammar("macd_histogram")


# --- Sous-famille : Simple Moving Average (SMA) --- #
# Feature 12/218 : "sma_20" (moyenne simple 20 periodes).
#   - Identifier  : sma_20, type=atomic, value_type=float, family=trend
#   - Comprendre  : moyenne arithmetique des 20 derniers close. Indicateur
#     de tendance retardé (~10 periodes de retard). Pas de borne theorique,
#     suit le prix.
#   - Concevoir   : pattern atomique. Operateurs : >, <, >=, <=.
#     Seuils : quantiles train (entre prix min et max du train).
#   - Verifier    : pas de tautologie (smas ne touchent jamais 0 sur crypto).
#   - Valider     : OK.
# Feature 13/218 : "sma_50" (moyen terme, plus lisse).
# Feature 14/218 : "sma_100" (long terme).
# Feature 15/218 : "sma_200" (tres long terme, tendance de fond).
SMA_20_GRAMMAR: str = _default_atomic_grammar("sma_20")
SMA_50_GRAMMAR: str = _default_atomic_grammar("sma_50")
SMA_100_GRAMMAR: str = _default_atomic_grammar("sma_100")
SMA_200_GRAMMAR: str = _default_atomic_grammar("sma_200")


# --- Sous-famille : Exponential Moving Average (EMA) --- #
# Feature 16/218 : "ema_9"  | Feature 17/218 : "ema_12"
# Feature 18/218 : "ema_21" | Feature 19/218 : "ema_26"
# Feature 20/218 : "ema_50" | Feature 21/218 : "ema_100"
# Feature 22/218 : "ema_200"
#   - Identifier  : ema_X, type=atomic, value_type=float, family=trend
#   - Comprendre  : moyenne exponentielle, plus de poids sur les valeurs
#     recentes que la SMA. Meme semantique que SMA (indicateur de tendance
#     retardé), mais plus reactive. Les periodes 12/26 sont standard pour MACD.
#   - Concevoir   : pattern atomique. Operateurs : >, <, >=, <=.
#     Seuils : quantiles train.
#   - Verifier    : pas de tautologie.
#   - Valider     : OK.
EMA_9_GRAMMAR: str = _default_atomic_grammar("ema_9")
EMA_12_GRAMMAR: str = _default_atomic_grammar("ema_12")
EMA_21_GRAMMAR: str = _default_atomic_grammar("ema_21")
EMA_26_GRAMMAR: str = _default_atomic_grammar("ema_26")
EMA_50_GRAMMAR: str = _default_atomic_grammar("ema_50")
EMA_100_GRAMMAR: str = _default_atomic_grammar("ema_100")
EMA_200_GRAMMAR: str = _default_atomic_grammar("ema_200")


# --- Sous-famille : Average True Range (ATR) --- #
# Feature 23/218 : "atr_14" (Wilder, volatilite sur 14 periodes).
#   - Identifier  : atr_14, type=atomic, value_type=float, family=volatility
#   - Comprendre  : moyenne mobile exponentielle des True Range (max des 3
#     ecarts high-low, high-close_prev, low-close_prev). Toujours >= 0.
#     Unites : log-return (puisque OHLCV en log-returns). Mesure la
#     volatilite recente.
#   - Concevoir   : pattern atomique. Operateurs : >, <, >=, <=.
#     Seuils : quantiles train (strictement > 0 en pratique).
#   - Verifier    : pas de tautologie (ATR toujours > 0 sur serie reelle).
#   - Valider     : OK.
# Feature 24/218 : "atr_21" (periode 21, plus long terme).
ATR_14_GRAMMAR: str = _default_atomic_grammar("atr_14")
ATR_21_GRAMMAR: str = _default_atomic_grammar("atr_21")


# --- Sous-famille : Stochastic Oscillator (George Lane, 1950s) --- #
# Feature 25/218 : "stoch_k" (%K = position du close dans le range high/low).
#   - Identifier  : stoch_k, type=atomic, value_type=float, family=momentum
#   - Comprendre  : 100 * (close - low_N) / (high_N - low_N) sur N periodes.
#     Borne [0, 100]. > 80 = surachete, < 20 = survendu.
#   - Concevoir   : pattern oscillateur borne (meme structure que RSI).
#   - Verifier    : pas de tautologie.
#   - Valider     : OK.
# Feature 26/218 : "stoch_d" (%D = SMA3 de %K, plus lisse).
STOCH_K_GRAMMAR: str = _oscillator_grammar("stoch_k", 14)
STOCH_D_GRAMMAR: str = _oscillator_grammar("stoch_d", 14)


# --- Sous-famille : Williams %R (Larry Williams) --- #
# Feature 27/218 : "williams_r".
#   - Identifier  : williams_r, type=atomic, value_type=float, family=momentum
#   - Comprendre  : -100 * (high_N - close) / (high_N - low_N). Borne [-100, 0].
#     > -20 = surachete, < -80 = survendu. Inverse du Stochastic %K.
#   - Concevoir   : pattern oscillateur borne. Seuils : quantiles train.
#   - Verifier    : pas de tautologie (borne sup 0 jamais atteinte, borne inf
#     -100 rarement ; quantiles dans [-100, 0]).
#   - Valider     : OK.
WILLIAMS_R_GRAMMAR: str = _oscillator_grammar("williams_r", 14)


# --- Sous-famille : Commodity Channel Index (CCI, Donald Lambert 1980) --- #
# Feature 28/218 : "cci_20".
#   - Identifier  : cci_20, type=atomic, value_type=float, family=momentum
#   - Comprendre  : (typical_price - SMA_N) / (0.015 * mean_dev_N). PAS borne
#     theoriquement (la convention veut que > +100 = surachete, < -100 =
#     survendu, mais en pratique ca peut depasser +-400 sur series tres
#     volatiles).
#   - Concevoir   : pattern atomique (non borne). Operateurs : >, <, >=, <=.
#     Seuils : quantiles train (calibration sur distribution reelle).
#   - Verifier    : pas de tautologie.
#   - Valider     : OK.
CCI_20_GRAMMAR: str = _default_atomic_grammar("cci_20")


# --- Sous-famille : Rate of Change (ROC) --- #
# Feature 29/218 : "roc_10" (variation % sur 10 periodes).
#   - Identifier  : roc_10, type=atomic, value_type=float, family=momentum
#   - Comprendre  : 100 * (close[t] / close[t-10] - 1). PAS borne (peut
#     etre tres negatif en krash, tres positif en pump). Signe = direction.
#   - Concevoir   : pattern atomique. Seuils : quantiles train.
#   - Verifier    : pas de tautologie.
#   - Valider     : OK.
# Feature 30/218 : "roc_20" (periode 20).
ROC_10_GRAMMAR: str = _default_atomic_grammar("roc_10")
ROC_20_GRAMMAR: str = _default_atomic_grammar("roc_20")


# --- Sous-famille : Momentum --- #
# Feature 31/218 : "momentum_10" (close[t] - close[t-10]).
#   - Identifier  : momentum_10, type=atomic, value_type=float, family=momentum
#   - Comprendre  : difference de close sur 10 periodes (PAS en %). PAS
#     borne, signe = direction, amplitude = force du mouvement.
#   - Concevoir   : pattern atomique. Seuils : quantiles train.
#   - Verifier    : pas de tautologie.
#   - Valider     : OK.
# Feature 32/218 : "momentum_20".
MOMENTUM_10_GRAMMAR: str = _default_atomic_grammar("momentum_10")
MOMENTUM_20_GRAMMAR: str = _default_atomic_grammar("momentum_20")


# --- Sous-famille : Ultimate Oscillator (Larry Williams 1985) --- #
# Feature 33/218 : "ultimate_oscillator".
#   - Identifier  : ultimate_oscillator, type=atomic, value_type=float,
#     family=momentum
#   - Comprendre  : combinaison ponderee de 3 periodes (7, 14, 28) de
#     "true range" et "buying pressure". Borne [0, 100]. > 70 = surachete,
#     < 30 = survendu. Plus robuste que Stochastic seul.
#   - Concevoir   : pattern oscillateur borne (meme structure que RSI).
#   - Verifier    : pas de tautologie.
#   - Valider     : OK.
ULTIMATE_OSCILLATOR_GRAMMAR: str = _oscillator_grammar("ultimate_oscillator", 28)


# --- Sous-famille : Average Directional Index (ADX, Welles Wilder) --- #
# Feature 34/218 : "adx_14" (force de la tendance, pas sa direction).
#   - Identifier  : adx_14, type=atomic, value_type=float, family=trend
#   - Comprendre  : moyenne du DX sur 14 periodes. Borne [0, 100].
#     > 25 = tendance forte, < 20 = range / pas de tendance.
#     IMPORTANT : ADX mesure la FORCE, pas la direction (les DI+ / DI-
#     s'en chargent).
#   - Concevoir   : pattern oscillateur borne.
#   - Verifier    : pas de tautologie.
#   - Valider     : OK.
ADX_14_GRAMMAR: str = _oscillator_grammar("adx_14", 14)


# --- Sous-famille : Directional Indicators (DI+, DI-) --- #
# Feature 35/218 : "di_plus" (force directionnelle haussiere).
#   - Identifier  : di_plus, type=atomic, value_type=float, family=trend
#   - Comprendre  : 100 * (+DM) / ATR. Borne [0, 100]. > 25 = direction
#     haussiere forte.
#   - Concevoir   : pattern oscillateur borne.
#   - Verifier    : pas de tautologie.
#   - Valider     : OK.
# Feature 36/218 : "di_minus" (force directionnelle baissiere).
#   - Symetrique de DI+. Borne [0, 100]. > 25 = direction baissiere forte.
DI_PLUS_GRAMMAR: str = _oscillator_grammar("di_plus", 14)
DI_MINUS_GRAMMAR: str = _oscillator_grammar("di_minus", 14)


# --- Sous-famille : Aroon (Tushar Chande 1995) --- #
# Feature 37/218 : "aroon_up" (periodes depuis le plus haut sur N).
#   - Identifier  : aroon_up, type=atomic, value_type=float, family=trend
#   - Comprendre  : 100 * (N - periods_since_high) / N. Borne [0, 100].
#     Proche de 100 = plus haut recent (tendance haussiere).
#   - Concevoir   : pattern oscillateur borne.
#   - Verifier    : pas de tautologie.
#   - Valider     : OK.
# Feature 38/218 : "aroon_down" (symetrique baissier).
AROON_UP_GRAMMAR: str = _oscillator_grammar("aroon_up", 25)
AROON_DOWN_GRAMMAR: str = _oscillator_grammar("aroon_down", 25)


# --- Sous-famille : Parabolic SAR (Welles Wilder) --- #
# Feature 39/218 : "parabolic_sar".
#   - Identifier  : parabolic_sar, type=atomic, value_type=float, family=trend
#   - Comprendre  : indicateur de tendance跟随 le prix avec un facteur
#     d'acceleration. PAS borne theoriquement, mais reste proche du prix
#     (avec un offset dependant de l'AF). Signe par rapport au prix = tendance.
#   - Concevoir   : pattern atomique. Seuils : quantiles train.
#   - Verifier    : pas de tautologie.
#   - Valider     : OK.
PARABOLIC_SAR_GRAMMAR: str = _default_atomic_grammar("parabolic_sar")


# --- Sous-famille : Supertrend (multi-timeframe trend indicator) --- #
# Feature 40/218 : "supertrend".
#   - Identifier  : supertrend, type=atomic, value_type=float, family=trend
#   - Comprendre  : combinaison ATR + multiplicateur pour跟踪 la tendance.
#     PAS borne theoriquement, fluctue autour du prix. Signe = direction.
#   - Concevoir   : pattern atomique. Seuils : quantiles train.
#   - Verifier    : pas de tautologie.
#   - Valider     : OK.
SUPERTREND_GRAMMAR: str = _default_atomic_grammar("supertrend")


# --- Sous-famille : TRIX (Jack Hutson 1980s) --- #
# Feature 41/218 : "trix_14".
#   - Identifier  : trix_14, type=atomic, value_type=float, family=trend
#   - Comprendre  : pourcentage de variation d'un triple EMA lissage sur
#     14 periodes. PAS borne, centre sur 0, capte le momentum filtre
#     (elimine le bruit court terme).
#   - Concevoir   : pattern atomique. Seuils : quantiles train.
#   - Verifier    : pas de tautologie.
#   - Valider     : OK.
TRIX_14_GRAMMAR: str = _default_atomic_grammar("trix_14")


# --- Sous-famille : Vortex (Etienne Botes & Douglas Siepman 2010) --- #
# Feature 42/218 : "vortex_pos" (Vortex Indicator positif).
#   - Identifier  : vortex_pos, type=atomic, value_type=float, family=trend
#   - Comprendre  : ratio de la somme des mouvements haussiers sur la somme
#     du True Range sur N periodes. PAS borne theoriquement (typiquement
#     entre 0.5 et 2.0). > 1 = tendance haussiere, < 1 = baissiere.
#   - Concevoir   : pattern atomique (ratio, non borne). Seuils : quantiles train.
#   - Verifier    : pas de tautologie.
#   - Valider     : OK.
# Feature 43/218 : "vortex_neg" (symetrique baissier).
VORTEX_POS_GRAMMAR: str = _default_atomic_grammar("vortex_pos")
VORTEX_NEG_GRAMMAR: str = _default_atomic_grammar("vortex_neg")


# --- Sous-famille : Bollinger Bands (John Bollinger 1980s) --- #
# Feature 44/218 : "bb_upper" (moyenne + k*ecart-type).
#   - Identifier  : bb_upper, type=atomic, value_type=float, family=volatility
#   - Comprendre  : SMA(N) + k * std(N) avec k=2 standard. PAS borne
#     theoriquement, suit le prix + volatilite. > prix = zone surachetee.
#   - Concevoir   : pattern atomique. Seuils : quantiles train.
#   - Verifier    : pas de tautologie.
#   - Valider     : OK.
# Feature 45/218 : "bb_middle" (SMA simple, equivalent a sma_X).
# Feature 46/218 : "bb_lower" (SMA - k*ecart-type).
BB_UPPER_GRAMMAR: str = _default_atomic_grammar("bb_upper")
BB_MIDDLE_GRAMMAR: str = _default_atomic_grammar("bb_middle")
BB_LOWER_GRAMMAR: str = _default_atomic_grammar("bb_lower")


# --- Sous-famille : Bollinger Band derived features --- #
# Feature 47/218 : "bb_percent" (%B = position du close dans les bandes).
#   - Identifier  : bb_percent, type=atomic, value_type=float, family=volatility
#   - Comprendre  : (close - bb_lower) / (bb_upper - bb_lower). Borne
#     theorique [0, 1] (peut legerement depasser). > 1 = au-dessus bande
#     haute, < 0 = en-dessous bande basse, 0.5 = sur la moyenne.
#   - Concevoir   : pattern borne (operateurs : >, <, >=, <=).
#   - Verifier    : pas de tautologie (bornes rarement atteintes).
#   - Valider     : OK.
# Feature 48/218 : "bb_width" (largeur relative des bandes = volatilite).
#   - Identifier  : bb_width, type=atomic, value_type=float, family=volatility
#   - Comprendre  : (bb_upper - bb_lower) / bb_middle. PAS borne theoriquement,
#     toujours >= 0. Mesure la volatilite relative (Bollinger squeeze = bb_width
#     faible, suivi d'un breakout).
#   - Concevoir   : pattern atomique (non borne, >= 0).
#   - Verifier    : pas de tautologie.
#   - Valider     : OK.
BB_PERCENT_GRAMMAR: str = _oscillator_grammar("bb_percent", 20)
BB_WIDTH_GRAMMAR: str = _default_atomic_grammar("bb_width")


# --- Sous-famille : On-Balance Volume (OBV, Joe Granville 1963) --- #
# Feature 49/218 : "obv" (volume cumule pondere par direction).
#   - Identifier  : obv, type=atomic, value_type=float, family=volume_flow
#   - Comprendre  : somme cumulee du volume, +volume si close > close_prev,
#     -volume si close < close_prev, 0 si egalite. PAS borne, peut etre
#     tres grand (cumulatif). Signe et tendance = confirmation de tendance.
#   - Concevoir   : pattern atomique. Seuils : quantiles train.
#   - Verifier    : pas de tautologie.
#   - Valider     : OK.
OBV_GRAMMAR: str = _default_atomic_grammar("obv")
# Feature 50/218 : "obv_ema" (EMA de OBV, plus lisse).
OBV_EMA_GRAMMAR: str = _default_atomic_grammar("obv_ema")


# --- Sous-famille : Volume --- #
# Feature 51/218 : "volume_sma_20" (moyenne 20 periodes du volume).
#   - Identifier  : volume_sma_20, type=atomic, value_type=float, family=volume_flow
#   - Comprendre  : SMA 20 periodes du volume (probablement en log1p, vu la
#     normalisation MIDAS V3). Toujours > 0.
#   - Concevoir   : pattern atomique. Seuils : quantiles train.
#   - Verifier    : pas de tautologie.
#   - Valider     : OK.
VOLUME_SMA_20_GRAMMAR: str = _default_atomic_grammar("volume_sma_20")
# Feature 52/218 : "volume_ratio" (ratio volume actuel / SMA volume).
#   - Identifier  : volume_ratio, type=atomic, value_type=float, family=volume_flow
#   - Comprendre  : volume / volume_sma. PAS borne, typique [0.5, 2.0].
#     > 1 = volume superieur a la moyenne, < 1 = inferieur. Confirmation
#     de breakout classique.
#   - Concevoir   : pattern atomique (ratio, non borne). Seuils : quantiles train.
#   - Verifier    : pas de tautologie.
#   - Valider     : OK.
VOLUME_RATIO_GRAMMAR: str = _default_atomic_grammar("volume_ratio")


# --- Sous-famille : Chaikin Oscillator (Marc Chaikin 1980s) --- #
# Feature 53/218 : "chaikin_oscillator".
#   - Identifier  : chaikin_oscillator, type=atomic, value_type=float,
#     family=volume_flow
#   - Comprendre  : EMA(ADL, 3) - EMA(ADL, 10) ou ADL = Accumulation/
#     Distribution Line. PAS borne, centre sur 0, capte la pression
#     achat/vente basee sur prix + volume.
#   - Concevoir   : pattern atomique (oscillateur non borne). Seuils : quantiles train.
#   - Verifier    : pas de tautologie.
#   - Valider     : OK.
CHAIKIN_OSCILLATOR_GRAMMAR: str = _default_atomic_grammar("chaikin_oscillator")


# --- Sous-famille : Money Flow Index (MFI, Gene Quong & Avrum Soudack) --- #
# Feature 54/218 : "money_flow_index".
#   - Identifier  : money_flow_index, type=atomic, value_type=float,
#     family=volume_flow
#   - Comprendre  : 100 - 100/(1 + money_ratio) avec money_ratio = positive
#     money flow / negative money flow. Borne [0, 100]. > 80 = surachete,
#     < 20 = survendu. Surnomme "RSI pondere par le volume".
#   - Concevoir   : pattern oscillateur borne.
#   - Verifier    : pas de tautologie.
#   - Valider     : OK.
MONEY_FLOW_INDEX_GRAMMAR: str = _oscillator_grammar("money_flow_index", 14)


# Mapping etendu pour le Lot 1 (indicateurs techniques atomic).
FEATURE_GRAMMARS_LOT1: dict[str, str] = {
    # Sous-famille RSI
    "rsi_14": RSI_14_GRAMMAR,
    "rsi_21": RSI_21_GRAMMAR,
    "rsi_30": RSI_30_GRAMMAR,
    # Sous-famille MACD
    "macd_line":      MACD_LINE_GRAMMAR,
    "macd_signal":    MACD_SIGNAL_GRAMMAR,
    "macd_histogram": MACD_HISTOGRAM_GRAMMAR,
    # Sous-famille SMA
    "sma_20":  SMA_20_GRAMMAR,
    "sma_50":  SMA_50_GRAMMAR,
    "sma_100": SMA_100_GRAMMAR,
    "sma_200": SMA_200_GRAMMAR,
    # Sous-famille EMA
    "ema_9":   EMA_9_GRAMMAR,
    "ema_12":  EMA_12_GRAMMAR,
    "ema_21":  EMA_21_GRAMMAR,
    "ema_26":  EMA_26_GRAMMAR,
    "ema_50":  EMA_50_GRAMMAR,
    "ema_100": EMA_100_GRAMMAR,
    "ema_200": EMA_200_GRAMMAR,
    # Sous-famille ATR
    "atr_14": ATR_14_GRAMMAR,
    "atr_21": ATR_21_GRAMMAR,
    # Sous-famille Stochastic
    "stoch_k": STOCH_K_GRAMMAR,
    "stoch_d": STOCH_D_GRAMMAR,
    # Sous-famille Williams %R
    "williams_r": WILLIAMS_R_GRAMMAR,
    # Sous-famille CCI
    "cci_20": CCI_20_GRAMMAR,
    # Sous-famille ROC
    "roc_10": ROC_10_GRAMMAR,
    "roc_20": ROC_20_GRAMMAR,
    # Sous-famille Momentum
    "momentum_10": MOMENTUM_10_GRAMMAR,
    "momentum_20": MOMENTUM_20_GRAMMAR,
    # Ultimate Oscillator
    "ultimate_oscillator": ULTIMATE_OSCILLATOR_GRAMMAR,
    # ADX
    "adx_14": ADX_14_GRAMMAR,
    # DI+ / DI-
    "di_plus":  DI_PLUS_GRAMMAR,
    "di_minus": DI_MINUS_GRAMMAR,
    # Aroon
    "aroon_up":   AROON_UP_GRAMMAR,
    "aroon_down": AROON_DOWN_GRAMMAR,
    # Parabolic SAR
    "parabolic_sar": PARABOLIC_SAR_GRAMMAR,
    # Supertrend
    "supertrend": SUPERTREND_GRAMMAR,
    # TRIX
    "trix_14": TRIX_14_GRAMMAR,
    # Vortex
    "vortex_pos": VORTEX_POS_GRAMMAR,
    "vortex_neg": VORTEX_NEG_GRAMMAR,
    # Bollinger Bands
    "bb_upper":  BB_UPPER_GRAMMAR,
    "bb_middle": BB_MIDDLE_GRAMMAR,
    "bb_lower":  BB_LOWER_GRAMMAR,
    "bb_percent": BB_PERCENT_GRAMMAR,
    "bb_width":   BB_WIDTH_GRAMMAR,
    # OBV
    "obv":     OBV_GRAMMAR,
    "obv_ema": OBV_EMA_GRAMMAR,
    # Volume-derived
    "volume_sma_20": VOLUME_SMA_20_GRAMMAR,
    "volume_ratio":  VOLUME_RATIO_GRAMMAR,
    # Chaikin
    "chaikin_oscillator": CHAIKIN_OSCILLATOR_GRAMMAR,
    # Money Flow Index
    "money_flow_index": MONEY_FLOW_INDEX_GRAMMAR,
}


# --------------------------------------------------------------------------- #
# Lot 2 : features QUANTITATIVES (31) — statistical, volatility, risk,
#         market_regime, momentum, microstructure, spectral
# --------------------------------------------------------------------------- #
#
# Procedure stricte (Identifier -> Comprendre -> Concevoir -> Verifier ->
# Valider -> Ecrire) appliquee pour chaque feature ci-dessous.
#
# NOTE inventaire : la taxonomie reference 35 features quantitative, mais
# 4 d'entre elles (`quant_hurst_exponent_signal`, `quant_realized_vol_10_signal`,
# `quant_shannon_entropy_signal`, `quant_vol_persistence_signal`) sont
# marquees `excluded=True` (fantomes) et donc absentes des 218 utilisables.
# Ce lot traite donc 31 features sur 35.
#
# 3 helpers specialises pour eviter la duplication :
#   - _default_atomic_grammar : series continues, pas de borne stricte
#   - _unit_bounded_grammar   : series dans [lower, upper] (Hurst, Shannon, etc.)
#   - _correlation_grammar    : series dans [-1, +1] (autocorrelations)


# --- Sous-famille 1 : Entropies (4) --- #
# Mesure l'incertitude / regularite / structure d'une serie.
# Feature 55/218 : "quant_approximate_entropy" (ApEn, Pincus 1991).
#   - Identifier  : quantitative, float, family=statistical
#   - Comprendre  : regularite via patterns de longueur m, tolerance r.
#     PAS de borne stricte, en pratique ~[0, 2.5]. Plus ApEn est grand,
#     plus la serie est irreguliere/alatoire.
#   - Concevoir   : _default_atomic_grammar (non bornee strictement).
#   - Verifier    : pas de tautologie.
#   - Valider     : OK.
# Feature 56/218 : "quant_sample_entropy" (SampEn, Richman & Moorman 2000).
#   - Amelioration de ApEn (pas d'auto-match, biais reduit). Meme semantique.
# Feature 57/218 : "quant_permutation_entropy" (Bandt & Pompe 2002).
#   - Patterns ordinaux (rangs). Borne theorique [0, log(m!)] (m=ordre).
#     Plus grand = plus irregulier. Robuste au bruit.
#   - Concevoir   : _unit_bounded_grammar(lower=0, upper=log(m!)) - on approx
#     upper=2.0 (m=3 donne log(6)~1.79, m=5 donne log(120)~4.79).
# Feature 58/218 : "quant_shannon_entropy" (Shannon 1948).
#   - H = -sum(p_i * log(p_i)) apres binning. Borne [0, log(K)] (K=bins).
#     Grand = aleatoire (range/choppy), petit = structure (tendance).
#   - Concevoir   : _unit_bounded_grammar(lower=0, upper=log(K)) - approx upper=5.0.
# Note : quant_shannon_entropy_signal est EXCLUE de la taxonomie (fantome).
QUANT_APPROXIMATE_ENTROPY_GRAMMAR: str = _default_atomic_grammar("quant_approximate_entropy")
QUANT_SAMPLE_ENTROPY_GRAMMAR: str = _default_atomic_grammar("quant_sample_entropy")
QUANT_PERMUTATION_ENTROPY_GRAMMAR: str = _unit_bounded_grammar(
    "quant_permutation_entropy", lower=0.0, upper=2.0,
)
QUANT_SHANNON_ENTROPY_GRAMMAR: str = _unit_bounded_grammar(
    "quant_shannon_entropy", lower=0.0, upper=5.0,
)


# --- Sous-famille 2 : Autocorrelations (6) --- #
# Feature 59/218 : "quant_autocorr_10" (Pearson, lag 10).
#   - Identifier  : quantitative, float, family=statistical
#   - Comprendre  : correlation entre r[t] et r[t-10]. Borne stricte [-1, +1].
#     > 0 = serie persistante (momentum), < 0 = mean-reverting, ~0 = aleatoire.
#   - Concevoir   : _correlation_grammar(lag=10).
#   - Verifier    : pas de tautologie (quantiles rarement +-1).
#   - Valider     : OK.
# Feature 60/218 : "quant_autocorr_10_weighted" (variante ponderee exponentiellement).
# Feature 61/218 : "quant_autocorr_20" et 62/218 "quant_autocorr_20_weighted".
# Feature 63/218 : "quant_autocorr_50" et 64/218 "quant_autocorr_50_weighted".
QUANT_AUTOCORR_10_GRAMMAR: str = _correlation_grammar("quant_autocorr_10", lag=10)
QUANT_AUTOCORR_10_WEIGHTED_GRAMMAR: str = _correlation_grammar(
    "quant_autocorr_10_weighted", lag=10,
)
QUANT_AUTOCORR_20_GRAMMAR: str = _correlation_grammar("quant_autocorr_20", lag=20)
QUANT_AUTOCORR_20_WEIGHTED_GRAMMAR: str = _correlation_grammar(
    "quant_autocorr_20_weighted", lag=20,
)
QUANT_AUTOCORR_50_GRAMMAR: str = _correlation_grammar("quant_autocorr_50", lag=50)
QUANT_AUTOCORR_50_WEIGHTED_GRAMMAR: str = _correlation_grammar(
    "quant_autocorr_50_weighted", lag=50,
)


# --- Sous-famille 3 : Volatilite (5) --- #
# Toutes >= 0, pas de borne superieure stricte (peut exploser en cas de crash).
# Feature 65/218 : "quant_garch_volatility" (GARCH(p,q) conditionnelle).
#   - Identifier  : quantitative, float, family=volatility
#   - Comprendre  : variance conditionnelle estimee par modele GARCH. Toujours
#     > 0. Plus grande = marche plus volatile.
#   - Concevoir   : _default_atomic_grammar (>= 0, pas de borne sup stricte).
#   - Verifier    : pas de tautologie.
#   - Valider     : OK.
# Feature 66/218 : "quant_realized_vol_10" (somme carree des rendements sur 10).
#   - Realized vol = sqrt(sum(r_i^2)) sur N periodes. > 0, ~[0, 0.5] typique 1h.
# Feature 67/218 : "quant_realized_vol_20" (fenetre 20).
# Feature 68/218 : "quant_realized_vol_50" (fenetre 50).
# Feature 69/218 : "quant_vol_clustering" (autocorrelation des |r|^2).
#   - Mesure si la volatilite est "clusterisee" (periodes calmes/perturbees).
#     Borne [-1, +1] en theorie (correlation), typique [0, 0.5] sur marche.
# Note : quant_realized_vol_10_signal est EXCLUE de la taxonomie (fantome).
QUANT_GARCH_VOLATILITY_GRAMMAR: str = _default_atomic_grammar("quant_garch_volatility")
QUANT_REALIZED_VOL_10_GRAMMAR: str = _default_atomic_grammar("quant_realized_vol_10")
QUANT_REALIZED_VOL_20_GRAMMAR: str = _default_atomic_grammar("quant_realized_vol_20")
QUANT_REALIZED_VOL_50_GRAMMAR: str = _default_atomic_grammar("quant_realized_vol_50")
QUANT_VOL_CLUSTERING_GRAMMAR: str = _correlation_grammar("quant_vol_clustering")


# --- Sous-famille 4 : Risque (5) --- #
# Feature 70/218 : "quant_dynamic_var" (Value-at-Risk dynamique).
#   - Identifier  : quantitative, float, family=risk
#   - Comprendre  : quantile de perte sur fenetre glissante. Borne typique
#     [-1, 0] (perte exprimee en log-return negatif ou positif selon signe).
#   - Concevoir   : _unit_bounded_grammar(lower=-1.0, upper=1.0) (en log-return).
#   - Verifier    : pas de tautologie.
#   - Valider     : OK.
# Feature 71/218 : "quant_dynamic_cvar" (Conditional VaR / Expected Shortfall).
#   - CVaR >= VaR (perte conditionnelle au-dela du seuil VaR). Meme plage.
# Feature 72/218 : "quant_max_drawdown" (drawdown maximum sur fenetre).
#   - Borne [0, 1] (drawdown = (peak - trough) / peak). Toujours >= 0.
#   - Concevoir   : _unit_bounded_grammar(lower=0.0, upper=1.0).
# Feature 73/218 : "quant_rolling_kurtosis" (exces de kurtosis glissant).
#   - PAS de borne stricte (typiquement [0, 30] pour series financieres,
#     0 = gaussien, > 0 = fat tails). Pas borne.
# Feature 74/218 : "quant_rolling_skewness" (asymetrie glissante).
#   - PAS de borne stricte (typiquement [-3, +3]). Pas borne.
QUANT_DYNAMIC_VAR_GRAMMAR: str = _unit_bounded_grammar("quant_dynamic_var", -1.0, 1.0)
QUANT_DYNAMIC_CVAR_GRAMMAR: str = _unit_bounded_grammar("quant_dynamic_cvar", -1.0, 1.0)
QUANT_MAX_DRAWDOWN_GRAMMAR: str = _unit_bounded_grammar("quant_max_drawdown", 0.0, 1.0)
QUANT_ROLLING_KURTOSIS_GRAMMAR: str = _default_atomic_grammar("quant_rolling_kurtosis")
QUANT_ROLLING_SKEWNESS_GRAMMAR: str = _default_atomic_grammar("quant_rolling_skewness")


# --- Sous-famille 5 : Regime de marche (4) --- #
# Feature 75/218 : "quant_hurst_exponent" (Hurst 1951).
#   - Identifier  : quantitative, float, family=market_regime
#   - Comprendre  : H<0.5 mean-reverting, H=0.5 random walk, H>0.5 trending.
#     Borne stricte [0, 1] pour signaux 1D.
#   - Concevoir   : _unit_bounded_grammar(lower=0.0, upper=1.0).
#   - Verifier    : pas de tautologie.
#   - Valider     : OK.
# Note : quant_hurst_exponent_signal est EXCLUE de la taxonomie (fantome).
# Feature 76/218 : "quant_dfa_exponent" (Detrended Fluctuation Analysis).
#   - Similaire a Hurst, plus robuste aux tendances locales. Borne [0, 1].
# Feature 77/218 : "quant_fractal_dimension" (Higuchi 1988 ou box-counting).
#   - D = 2 - H pour signaux 1D. Borne [1, 2].
#   - Concevoir   : _unit_bounded_grammar(lower=1.0, upper=2.0).
# Feature 78/218 : "quant_regime_detection" (indicateur de regime).
#   - Probablement un entier code (0/1/2) ou continu selon impl. Traite
#     comme continu par defaut.
#   - Concevoir   : _default_atomic_grammar (on verra au moment du parser si
#     un override categoriel est necessaire).
QUANT_HURST_EXPONENT_GRAMMAR: str = _unit_bounded_grammar(
    "quant_hurst_exponent", 0.0, 1.0,
)
QUANT_DFA_EXPONENT_GRAMMAR: str = _unit_bounded_grammar("quant_dfa_exponent", 0.0, 1.0)
QUANT_FRACTAL_DIMENSION_GRAMMAR: str = _unit_bounded_grammar(
    "quant_fractal_dimension", 1.0, 2.0,
)
QUANT_REGIME_DETECTION_GRAMMAR: str = _default_atomic_grammar("quant_regime_detection")


# --- Sous-famille 6 : Momentum quantitatif (2) --- #
# Feature 79/218 : "quant_kaufman_efficiency" (Kaufman 1995).
#   - Identifier  : quantitative, float, family=momentum
#   - Comprendre  : |close - close_prev_N| / sum(|r_i|). Borne [0, 1].
#     Proche de 1 = directionnel, proche de 0 = choppy.
#   - Concevoir   : _unit_bounded_grammar(lower=0.0, upper=1.0).
#   - Verifier    : pas de tautologie.
#   - Valider     : OK.
# Feature 80/218 : "quant_vol_persistence" (autocorrelation des |r|^2 a lag).
#   - Correlation de la vol avec elle-meme. Borne [-1, +1].
#   - Concevoir   : _correlation_grammar.
# Note : quant_vol_persistence_signal est EXCLUE de la taxonomie (fantome).
QUANT_KAUFMAN_EFFICIENCY_GRAMMAR: str = _unit_bounded_grammar(
    "quant_kaufman_efficiency", 0.0, 1.0,
)
QUANT_VOL_PERSISTENCE_GRAMMAR: str = _correlation_grammar("quant_vol_persistence")


# --- Sous-famille 7 : Microstructure (2) --- #
# Feature 81/218 : "quant_amihud_illiquidity" (Amihud 2002).
#   - Identifier  : quantitative, float, family=microstructure
#   - Comprendre  : |return| / volume. Prix de l'illiquidite. Toujours >= 0.
#   - Concevoir   : _default_atomic_grammar (>= 0, pas de borne sup stricte).
#   - Verifier    : pas de tautologie.
#   - Valider     : OK.
# Feature 82/218 : "quant_kyles_lambda" (Kyle 1985).
#   - Coefficient de regression price_change sur signed_volume. >= 0.
#   - Plus eleve = plus le prix reagit aux ordres informes.
QUANT_AMIHUD_ILLIQUIDITY_GRAMMAR: str = _default_atomic_grammar("quant_amihud_illiquidity")
QUANT_KYLES_LAMBDA_GRAMMAR: str = _default_atomic_grammar("quant_kyles_lambda")


# --- Sous-famille 8 : Spectral (2) --- #
# Feature 83/218 : "quant_dominant_frequency".
#   - Identifier  : quantitative, float, family=statistical
#   - Comprendre  : frequence dominante du spectre (inverse de la periode
#     du cycle dominant). Borne [0, Nyquist_frequency].
#   - Concevoir   : _unit_bounded_grammar (Nyquist depend du timeframe, on
#     approx upper=1.0 et le parser l'adaptera).
#   - Verifier    : pas de tautologie.
#   - Valider     : OK.
# Feature 84/218 : "quant_spectral_centroid".
#   - Centroide du spectre = moyenne ponderee des frequences par leur
#     energie. Borne [0, Nyquist]. Plus eleve = plus d'energie HF.
QUANT_DOMINANT_FREQUENCY_GRAMMAR: str = _unit_bounded_grammar(
    "quant_dominant_frequency", 0.0, 1.0,
)
QUANT_SPECTRAL_CENTROID_GRAMMAR: str = _unit_bounded_grammar(
    "quant_spectral_centroid", 0.0, 1.0,
)


# --- Sous-famille 9 : Variance ratio (1) --- #
# Feature 85/218 : "quant_variance_ratio" (Lo & MacKinlay 1988).
#   - Identifier  : quantitative, float, family=statistical
#   - Comprendre  : Var(r_q) / (q * Var(r_1)). < 1 = mean-reverting,
#     > 1 = trending, = 1 = random walk. PAS de borne stricte, typique [0, 3].
#   - Concevoir   : _default_atomic_grammar.
#   - Verifier    : pas de tautologie.
#   - Valider     : OK.
QUANT_VARIANCE_RATIO_GRAMMAR: str = _default_atomic_grammar("quant_variance_ratio")


# Mapping etendu pour le Lot 2 (features quantitative, 31 features utilisables).
FEATURE_GRAMMARS_LOT2: dict[str, str] = {
    # Entropies (4)
    "quant_approximate_entropy":      QUANT_APPROXIMATE_ENTROPY_GRAMMAR,
    "quant_sample_entropy":           QUANT_SAMPLE_ENTROPY_GRAMMAR,
    "quant_permutation_entropy":      QUANT_PERMUTATION_ENTROPY_GRAMMAR,
    "quant_shannon_entropy":          QUANT_SHANNON_ENTROPY_GRAMMAR,
    # Autocorrelations (6)
    "quant_autocorr_10":              QUANT_AUTOCORR_10_GRAMMAR,
    "quant_autocorr_10_weighted":     QUANT_AUTOCORR_10_WEIGHTED_GRAMMAR,
    "quant_autocorr_20":              QUANT_AUTOCORR_20_GRAMMAR,
    "quant_autocorr_20_weighted":     QUANT_AUTOCORR_20_WEIGHTED_GRAMMAR,
    "quant_autocorr_50":              QUANT_AUTOCORR_50_GRAMMAR,
    "quant_autocorr_50_weighted":     QUANT_AUTOCORR_50_WEIGHTED_GRAMMAR,
    # Volatilite (5)
    "quant_garch_volatility":         QUANT_GARCH_VOLATILITY_GRAMMAR,
    "quant_realized_vol_10":          QUANT_REALIZED_VOL_10_GRAMMAR,
    "quant_realized_vol_20":          QUANT_REALIZED_VOL_20_GRAMMAR,
    "quant_realized_vol_50":          QUANT_REALIZED_VOL_50_GRAMMAR,
    "quant_vol_clustering":           QUANT_VOL_CLUSTERING_GRAMMAR,
    # Risque (5)
    "quant_dynamic_var":              QUANT_DYNAMIC_VAR_GRAMMAR,
    "quant_dynamic_cvar":             QUANT_DYNAMIC_CVAR_GRAMMAR,
    "quant_max_drawdown":             QUANT_MAX_DRAWDOWN_GRAMMAR,
    "quant_rolling_kurtosis":         QUANT_ROLLING_KURTOSIS_GRAMMAR,
    "quant_rolling_skewness":         QUANT_ROLLING_SKEWNESS_GRAMMAR,
    # Regime (4)
    "quant_hurst_exponent":           QUANT_HURST_EXPONENT_GRAMMAR,
    "quant_dfa_exponent":             QUANT_DFA_EXPONENT_GRAMMAR,
    "quant_fractal_dimension":        QUANT_FRACTAL_DIMENSION_GRAMMAR,
    "quant_regime_detection":         QUANT_REGIME_DETECTION_GRAMMAR,
    # Momentum quant (2)
    "quant_kaufman_efficiency":       QUANT_KAUFMAN_EFFICIENCY_GRAMMAR,
    "quant_vol_persistence":          QUANT_VOL_PERSISTENCE_GRAMMAR,
    # Microstructure (2)
    "quant_amihud_illiquidity":       QUANT_AMIHUD_ILLIQUIDITY_GRAMMAR,
    "quant_kyles_lambda":             QUANT_KYLES_LAMBDA_GRAMMAR,
    # Spectral (2)
    "quant_dominant_frequency":       QUANT_DOMINANT_FREQUENCY_GRAMMAR,
    "quant_spectral_centroid":        QUANT_SPECTRAL_CENTROID_GRAMMAR,
    # Variance ratio (1)
    "quant_variance_ratio":           QUANT_VARIANCE_RATIO_GRAMMAR,
}


# --------------------------------------------------------------------------- #
# Lot 3 : 4 atomic restants + 9 composite_derived (signaux) + 13 factors
#         (26 features au total)
# --------------------------------------------------------------------------- #
#
# Procedure stricte (Identifier -> Comprendre -> Concevoir -> Verifier ->
# Valider -> Ecrire) appliquee pour chaque feature ci-dessous.
#
# Verification inventaire : toutes les features sont filtrees sur
# `usable_feature_names` (pas seulement la taxonomie JSON brute), suite a
# l'erreur du Lot 2 ou 4 features quantitative `_signal` fantomes avaient
# ete incluses a tort.
#
# Helper additionnel :
#   - _signal_grammar(feat, values) : signaux discrets (valeurs fixes, pas
#     quantiles continus) ; terminaux nommes `v_<feat>_<N>`.


# --- Sous-famille 1 : 4 atomic restants --- #
# Feature 86/218 : "choppiness_index" (E.W. Dreiss).
#   - Identifier  : atomic, float, family=market_regime
#   - Comprendre  : 100 * log10(sum(ATR_N) / range_N) / log10(N). Borne
#     [0, 100]. > 61.8 = tres choppy (range), < 38.2 = trending.
#   - Concevoir   : _oscillator_grammar (borne [0, 100]).
#   - Verifier    : pas de tautologie.
#   - Valider     : OK.
# Feature 87/218 : "kurtosis_risk" (kurtosis rolling).
#   - Identifier  : atomic, float, family=risk
#   - Comprendre  : exces de kurtosis sur fenetre glissante. PAS de borne
#     stricte, typique [0, 30]. > 0 = fat tails vs gaussien.
#   - Concevoir   : _default_atomic_grammar.
#   - Verifier    : pas de tautologie.
#   - Valider     : OK.
# Feature 88/218 : "skewness_risk" (skewness rolling).
#   - Identifier  : atomic, float, family=risk
#   - Comprendre  : asymetrie sur fenetre. PAS de borne stricte, typique
#     [-3, +3]. < 0 = skewed a gauche (pertes), > 0 = skewed a droite.
#   - Concevoir   : _default_atomic_grammar.
#   - Verifier    : pas de tautologie.
#   - Valider     : OK.
# Feature 89/218 : "vwap" (Volume-Weighted Average Price).
#   - Identifier  : atomic, float, family=price_action
#   - Comprendre  : sum(close * volume) / sum(volume) (rolling). Suit le
#     prix, pas de borne stricte. Indicateur intraday de prix moyen pondere
#     par le volume (zones de gros volume = support/resistance).
#   - Concevoir   : _default_atomic_grammar.
#   - Verifier    : pas de tautologie.
#   - Valider     : OK.
CHOPPINESS_INDEX_GRAMMAR: str = _oscillator_grammar("choppiness_index", 14)
KURTOSIS_RISK_GRAMMAR: str = _default_atomic_grammar("kurtosis_risk")
SKEWNESS_RISK_GRAMMAR: str = _default_atomic_grammar("skewness_risk")
VWAP_GRAMMAR: str = _default_atomic_grammar("vwap")


# --- Sous-famille 2 : 9 composite_derived (signaux discrets) --- #
# Pattern : les `_signal` sont des features binarisees (0/1) ou trinaires
# (-1/0/1) obtenues par seuillage de la feature continue parente. Operateurs
# >, <, >=, <= fonctionnent aussi, mais on garde les memes pour coherence.
# Seuils = valeurs discretes (PAS quantiles).
# Feature 90/218 : "adx_strength_signal" (seuillage adx_14).
#   - Identifier  : composite_derived, family=trend
#   - Comprendre  : typiquement 1 si adx_14 > 25 (tendance forte), 0 sinon.
#     Valeurs discretes {0, 1}.
#   - Concevoir   : _signal_grammar((0, 1)).
#   - Verifier    : pas de tautologie.
#   - Valider     : OK.
# Feature 91/218 : "aroon_trend_signal" (seuillage aroon_up vs aroon_down).
#   - Composite_derived, family=trend. Typiquement {-1, 0, 1}.
#   - Concevoir   : _signal_grammar((-1, 0, 1)).
# Feature 92/218 : "bb_width_signal" (seuillage volatilite BB squeeze).
#   - Composite_derived, family=volatility. Typiquement {0, 1}.
#   - Concevoir   : _signal_grammar((0, 1)).
# Feature 93/218 : "macd_trend_signal" (croisement macd_line / macd_signal).
#   - Composite_derived, family=trend. Typiquement {-1, 0, 1}.
#   - Concevoir   : _signal_grammar((-1, 0, 1)).
# Feature 94/218 : "obv_signal" (seuillage tendance OBV).
#   - Composite_derived, family=volume_flow. Typiquement {0, 1}.
#   - Concevoir   : _signal_grammar((0, 1)).
# Feature 95/218 : "supertrend_signal" (sens du supertrend).
#   - Composite_derived, family=trend. Typiquement {-1, 0, 1}.
#   - Concevoir   : _signal_grammar((-1, 0, 1)).
# Feature 96/218 : "volume_ratio_signal" (seuillage volume_ratio > 1).
#   - Composite_derived, family=volume_flow. Typiquement {0, 1}.
#   - Concevoir   : _signal_grammar((0, 1)).
# Feature 97/218 : "volume_sma_signal" (seuillage volume_sma_20).
#   - Composite_derived, family=volume_flow. Typiquement {0, 1}.
#   - Concevoir   : _signal_grammar((0, 1)).
# Feature 98/218 : "vwap_signal" (close > vwap ou close < vwap).
#   - Composite_derived, family=price_action. Typiquement {-1, 0, 1}.
#   - Concevoir   : _signal_grammar((-1, 0, 1)).
ADX_STRENGTH_SIGNAL_GRAMMAR: str = _signal_grammar("adx_strength_signal", (0, 1))
AROON_TREND_SIGNAL_GRAMMAR: str = _signal_grammar("aroon_trend_signal", (-1, 0, 1))
BB_WIDTH_SIGNAL_GRAMMAR: str = _signal_grammar("bb_width_signal", (0, 1))
MACD_TREND_SIGNAL_GRAMMAR: str = _signal_grammar("macd_trend_signal", (-1, 0, 1))
OBV_SIGNAL_GRAMMAR: str = _signal_grammar("obv_signal", (0, 1))
SUPERTREND_SIGNAL_GRAMMAR: str = _signal_grammar("supertrend_signal", (-1, 0, 1))
VOLUME_RATIO_SIGNAL_GRAMMAR: str = _signal_grammar("volume_ratio_signal", (0, 1))
VOLUME_SMA_SIGNAL_GRAMMAR: str = _signal_grammar("volume_sma_signal", (0, 1))
VWAP_SIGNAL_GRAMMAR: str = _signal_grammar("vwap_signal", (-1, 0, 1))


# --- Sous-famille 3 : 13 factors (scores agreges) --- #
# Scores combinant plusieurs features en un seul signal. Bornes typiques
# [0, 1] (scores normalises) ou [0, 100] (scores en %). On part sur [0, 1]
# par defaut, le parser verifiera la distribution reelle.
# Feature 99/218 : "Factor_Candlestick_Bullish_Score" (famille price_action).
#   - Identifier  : factor, family=price_action
#   - Comprendre  : score agrege des patterns chandeliers haussiers. Borne
#     typique [0, 1]. Plus eleve = plus de patterns haussiers.
#   - Concevoir   : _unit_bounded_grammar(0.0, 1.0).
#   - Verifier    : pas de tautologie.
#   - Valider     : OK.
# Feature 100/218 : "Factor_Candlestick_Bearish_Score" (price_action). Idem mais baissier.
# Feature 101/218 : "Factor_Chart_Patterns_Score" (market_structure). Patterns chartistes reconnus.
# Feature 102/218 : "Factor_Harmonic_Patterns_Score" (market_structure). Patterns harmoniques.
# Feature 103/218 : "Factor_Momentum_Score" (momentum). Score momentum agrege.
# Feature 104/218 : "Factor_Persistence_Score" (statistical). Score autocorrelation/persistence.
# Feature 105/218 : "Factor_Quantitative_Score" (statistical). Score agrege des quantitative.
# Feature 106/218 : "Factor_Regime_Hurst_Score" (market_regime). Score regime base sur Hurst.
# Feature 107/218 : "Factor_Risk_TailEvent_Score" (risk). Score risque de queues epaisses.
# Feature 108/218 : "Factor_Support_Resistance_Score" (market_structure). Proximite S/R.
# Feature 109/218 : "Factor_Trend_Score" (trend). Score tendance agrege.
# Feature 110/218 : "Factor_Volatility_Score" (volatility). Score volatilite.
# Feature 111/218 : "Factor_Volume_Score" (volume_flow). Score volume.
FACTOR_CANDLESTICK_BULLISH_SCORE_GRAMMAR: str = _unit_bounded_grammar(
    "Factor_Candlestick_Bullish_Score", 0.0, 1.0,
)
FACTOR_CANDLESTICK_BEARISH_SCORE_GRAMMAR: str = _unit_bounded_grammar(
    "Factor_Candlestick_Bearish_Score", 0.0, 1.0,
)
FACTOR_CHART_PATTERNS_SCORE_GRAMMAR: str = _unit_bounded_grammar(
    "Factor_Chart_Patterns_Score", 0.0, 1.0,
)
FACTOR_HARMONIC_PATTERNS_SCORE_GRAMMAR: str = _unit_bounded_grammar(
    "Factor_Harmonic_Patterns_Score", 0.0, 1.0,
)
FACTOR_MOMENTUM_SCORE_GRAMMAR: str = _unit_bounded_grammar(
    "Factor_Momentum_Score", 0.0, 1.0,
)
FACTOR_PERSISTENCE_SCORE_GRAMMAR: str = _unit_bounded_grammar(
    "Factor_Persistence_Score", 0.0, 1.0,
)
FACTOR_QUANTITATIVE_SCORE_GRAMMAR: str = _unit_bounded_grammar(
    "Factor_Quantitative_Score", 0.0, 1.0,
)
FACTOR_REGIME_HURST_SCORE_GRAMMAR: str = _unit_bounded_grammar(
    "Factor_Regime_Hurst_Score", 0.0, 1.0,
)
FACTOR_RISK_TAILEVENT_SCORE_GRAMMAR: str = _unit_bounded_grammar(
    "Factor_Risk_TailEvent_Score", 0.0, 1.0,
)
FACTOR_SUPPORT_RESISTANCE_SCORE_GRAMMAR: str = _unit_bounded_grammar(
    "Factor_Support_Resistance_Score", 0.0, 1.0,
)
FACTOR_TREND_SCORE_GRAMMAR: str = _unit_bounded_grammar(
    "Factor_Trend_Score", 0.0, 1.0,
)
FACTOR_VOLATILITY_SCORE_GRAMMAR: str = _unit_bounded_grammar(
    "Factor_Volatility_Score", 0.0, 1.0,
)
FACTOR_VOLUME_SCORE_GRAMMAR: str = _unit_bounded_grammar(
    "Factor_Volume_Score", 0.0, 1.0,
)


# Mapping etendu pour le Lot 3 (4 atomic + 9 signals + 13 factors = 26).
FEATURE_GRAMMARS_LOT3: dict[str, str] = {
    # 4 atomic restants
    "choppiness_index":  CHOPPINESS_INDEX_GRAMMAR,
    "kurtosis_risk":     KURTOSIS_RISK_GRAMMAR,
    "skewness_risk":     SKEWNESS_RISK_GRAMMAR,
    "vwap":              VWAP_GRAMMAR,
    # 9 composite_derived (signaux discrets)
    "adx_strength_signal":  ADX_STRENGTH_SIGNAL_GRAMMAR,
    "aroon_trend_signal":   AROON_TREND_SIGNAL_GRAMMAR,
    "bb_width_signal":      BB_WIDTH_SIGNAL_GRAMMAR,
    "macd_trend_signal":    MACD_TREND_SIGNAL_GRAMMAR,
    "obv_signal":           OBV_SIGNAL_GRAMMAR,
    "supertrend_signal":    SUPERTREND_SIGNAL_GRAMMAR,
    "volume_ratio_signal":  VOLUME_RATIO_SIGNAL_GRAMMAR,
    "volume_sma_signal":    VOLUME_SMA_SIGNAL_GRAMMAR,
    "vwap_signal":          VWAP_SIGNAL_GRAMMAR,
    # 13 factors
    "Factor_Candlestick_Bullish_Score":  FACTOR_CANDLESTICK_BULLISH_SCORE_GRAMMAR,
    "Factor_Candlestick_Bearish_Score":  FACTOR_CANDLESTICK_BEARISH_SCORE_GRAMMAR,
    "Factor_Chart_Patterns_Score":       FACTOR_CHART_PATTERNS_SCORE_GRAMMAR,
    "Factor_Harmonic_Patterns_Score":    FACTOR_HARMONIC_PATTERNS_SCORE_GRAMMAR,
    "Factor_Momentum_Score":             FACTOR_MOMENTUM_SCORE_GRAMMAR,
    "Factor_Persistence_Score":          FACTOR_PERSISTENCE_SCORE_GRAMMAR,
    "Factor_Quantitative_Score":         FACTOR_QUANTITATIVE_SCORE_GRAMMAR,
    "Factor_Regime_Hurst_Score":         FACTOR_REGIME_HURST_SCORE_GRAMMAR,
    "Factor_Risk_TailEvent_Score":       FACTOR_RISK_TAILEVENT_SCORE_GRAMMAR,
    "Factor_Support_Resistance_Score":   FACTOR_SUPPORT_RESISTANCE_SCORE_GRAMMAR,
    "Factor_Trend_Score":                FACTOR_TREND_SCORE_GRAMMAR,
    "Factor_Volatility_Score":           FACTOR_VOLATILITY_SCORE_GRAMMAR,
    "Factor_Volume_Score":               FACTOR_VOLUME_SCORE_GRAMMAR,
}


# Bloc de relations OHLCV (traitement conjoint des 5 features).
# Permet de generer des conditions qui exploitent la semantique partagee
# des 5 features OHLCV (par opposition a des comparaisons feat-vs-quantile).
#
# Les terminaux `q_range_pX` et `q_volume_pX` seront resolus par le parser
# depuis des quantiles calibres sur le train :
#   - q_range_pX  = quantiles de (high - low) sur le train (range intra-bougie)
#   - q_volume_pX = quantiles de volume sur le train (cf. VOLUME_GRAMMAR)
#
# Note d'implementation : la fonction `compute_ohlcv_range_quantiles` qui
# produit q_range_pX sera ajoutee dans data/threshold_calibration.py quand
# le parser GE en aura besoin (BNF Phase 2 / anti-tautologies). En attendant,
# ces terminaux sont documentes mais non resolus — le moteur n'utilise pas
# encore OHLCV_RELATIONS_GRAMMAR, donc pas de simulation cachee.
OHLCV_RELATIONS_GRAMMAR: str = (
    '<ohlcv_relation>      ::= <bullish_candle> | <bearish_candle> '
    '| <wide_range_candle> | <bullish_breakout_on_volume>\n'
    '<bullish_candle>        ::= "close" ">" "open"\n'
    '<bearish_candle>        ::= "close" "<" "open"\n'
    '<wide_range_candle>     ::= "high" "-" "low" ">" <range_threshold>\n'
    '<bullish_breakout_on_volume> ::= "close" ">" "open" "AND" "volume" ">" <volume_threshold>\n'
    '<range_threshold>       ::= q_range_p10 | q_range_p25 | q_range_p50 | q_range_p75 | q_range_p90\n'
    '<volume_threshold>      ::= q_volume_p10 | q_volume_p25 | q_volume_p50 | q_volume_p75 | q_volume_p90'
)


# Mapping feature_name -> grammaire specifique (override le pattern par defaut).
# Pour les features non listees ici, _default_atomic_grammar(feature_name) est utilise.
# Initialise avec les 5 features OHLCV (Lot 0), puis etendu avec le Lot 1.
FEATURE_GRAMMARS: dict[str, str] = {
    "open":   OPEN_GRAMMAR,
    "high":   HIGH_GRAMMAR,
    "low":    LOW_GRAMMAR,
    "close":  CLOSE_GRAMMAR,
    "volume": VOLUME_GRAMMAR,
}
FEATURE_GRAMMARS.update(FEATURE_GRAMMARS_LOT1)
FEATURE_GRAMMARS.update(FEATURE_GRAMMARS_LOT2)
FEATURE_GRAMMARS.update(FEATURE_GRAMMARS_LOT3)


# Mapping des grammaires de relations (vs. grammaires par feature).
# Sert au parser GE pour piocher des conditions relationnelles plutot
# que purement atomiques.
RELATIONS_GRAMMARS: dict[str, str] = {
    "ohlcv": OHLCV_RELATIONS_GRAMMAR,
}


# --------------------------------------------------------------------------- #
# API publique
# --------------------------------------------------------------------------- #


def get_feature_grammar(feature_name: str, config: EinherjarConfig) -> str:
    """Retourne la grammaire BNF pour une feature donnee.

    Cherche d'abord dans FEATURE_GRAMMARS (override), puis fallback sur
    le pattern par defaut _default_atomic_grammar.

    Leve ValueError si la feature n'est pas dans la taxonomie.
    """
    if feature_name not in config.usable_feature_names:
        raise ValueError(
            f"Feature {feature_name!r} absente de la taxonomie "
            f"({len(config.usable_feature_names)} features utilisables)."
        )
    if feature_name in FEATURE_GRAMMARS:
        return FEATURE_GRAMMARS[feature_name]
    return _default_atomic_grammar(feature_name)


def get_relations_grammar(relations_key: str) -> str:
    """Retourne la grammaire BNF pour un bloc de relations donne.

    Args:
        relations_key: Cle du bloc dans RELATIONS_GRAMMARS (ex: "ohlcv").

    Raises:
        ValueError: si la cle est inconnue.
    """
    if relations_key not in RELATIONS_GRAMMARS:
        raise ValueError(
            f"Bloc de relations {relations_key!r} inconnu. "
            f"Disponibles : {sorted(RELATIONS_GRAMMARS.keys())}."
        )
    return RELATIONS_GRAMMARS[relations_key]


def get_feature_grammar_metadata(
    feature_name: str, config: EinherjarConfig,
) -> dict[str, Any]:
    """Retourne les metadonnees de la feature (type, famille, etc.)."""
    info = config.features_taxonomy.get("features", {}).get(feature_name)
    if info is None:
        raise ValueError(f"Feature {feature_name!r} absente de la taxonomie")
    return info
