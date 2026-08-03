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
  - 1 bloc de relations OHLCV : `OHLCV_RELATIONS_GRAMMAR` (traitement
    conjoint open/high/low/close/volume, e.g. "close > open").
  - Les features specialisees (pattern, factor, signal, quantitative
    statistiques) necessiteront des regles specifiques par la suite.
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
