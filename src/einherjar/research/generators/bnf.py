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
  - 5/218 features traitees : open, high, low, close, volume
    (famille OHLCV = price_action + volume_flow, log-returns / log1p).
  - 1 bloc de relations OHLCV : `OHLCV_RELATIONS_GRAMMAR` (traitement
    conjoint open/high/low/close/volume, e.g. "close > open").
  - Les features specialisees (pattern, factor, signal) necessiteront
    des regles specifiques par la suite.
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
FEATURE_GRAMMARS: dict[str, str] = {
    "open":   OPEN_GRAMMAR,
    "high":   HIGH_GRAMMAR,
    "low":    LOW_GRAMMAR,
    "close":  CLOSE_GRAMMAR,
    "volume": VOLUME_GRAMMAR,
    # Prochaines features a ajouter au fur et a mesure du traitement...
}


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
