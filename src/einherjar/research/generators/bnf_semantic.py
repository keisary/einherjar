"""generators/bnf_semantic.py — Orientation semantique des patterns.

⚠ Chantier BNF Phase 3 (apres Phase 1 terminaux, Phase 2 anti-tautologies
differee, Phase 4 parser+GE livrees). Ce module attribue a chaque pattern
de la taxonomie une **orientation semantique naturelle** (BULLISH, BEARISH,
ou NEUTRAL) qui reflete la semantique classique du pattern, independamment
du contexte d'utilisation.

Utilite :
  - Le moteur d'admission peut scorer la coherence entre l'orientation
    semantique du pattern et la direction de l'Hypothesis (e.g., penaliser
    un pattern haussier utilise pour un SHORT).
  - Le generateur NSGA-II peut ajouter une dimension semantique au front
    de Pareto (e.g., preferer les hypotheses dont la direction Hypothesis
    matche l'orientation du pattern).
  - Le comparateur peut filtrer ou regrouper les generateurs par
    "qualite semantique" moyenne.

Heuristique de classification (cf. _classify_pattern_orientation) :
  - Suffixes bullish : _bull, _bottom, _up, _long, _white_soldiers, _uptrend
  - Suffixes bearish : _bear, _top, _down, _short, _black_crows, _downtrend
  - Suffixes neutres  : _doji, _reversal, _fill, _sideways, _match,
                        _three_drives, _crab, _bat, _shark, _butterfly,
                        _gartley, _elliott_*, _harmonic_*, _channel, _flag,
                        _pennant, _wedge, _triangle, _diamond_*, _rounding_*,
                        _island_reversal, _support, _resistance, _wolfe_*,
                        _fibonacci_*, _cup_handle, _spike_reversal,
                        _measured_move, _v_top, _v_bottom, _advance_block,
                        _unique_three_river_bottom, _three_*_down (mixte),
                        _three_*_up (mixte), _abandoned_baby_* (mixte)

Statut : 107 patterns classifies via heuristique, validation manuelle
prevue au moment de l'integration avec le moteur d'evaluation.
"""

from __future__ import annotations

import logging
import re
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Enum orientation semantique
# --------------------------------------------------------------------------- #


class SemanticOrientation(str, Enum):
    """Orientation semantique naturelle d'un pattern de la taxonomie.

    - BULLISH  : le pattern suggere un mouvement haussier
                 (e.g., hammer, morning_star, ascending_triangle).
    - BEARISH  : le pattern suggere un mouvement baissier
                 (e.g., shooting_star, evening_star, descending_triangle).
    - NEUTRAL  : le pattern est ambigu ou de continuation,
                 sans direction inherente (e.g., doji, sideways, gap_fill).
    """

    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


# --------------------------------------------------------------------------- #
# Heuristique de classification
# --------------------------------------------------------------------------- #


# Cas explicites (ordres de priorite) : on teste ces noms EXACTS en premier
# pour les cas ambigus (dojis, triangles, wedges, etc.) qui sinon seraient
# classifies NEUTRAL par les regles par defaut.
_EXACT_RULES: tuple[tuple[str, SemanticOrientation], ...] = (
    # Dojis (sauf le doji pur qui est NEUTRAL)
    ("pattern_dragonfly_doji", SemanticOrientation.BULLISH),
    ("pattern_gravestone_doji", SemanticOrientation.BEARISH),
    # Triangles (continuation directionnelle)
    ("pattern_ascending_triangle", SemanticOrientation.BULLISH),
    ("pattern_descending_triangle", SemanticOrientation.BEARISH),
    ("pattern_symmetrical_triangle", SemanticOrientation.NEUTRAL),
    # Wedges (biseaux)
    ("pattern_rising_wedge", SemanticOrientation.BEARISH),
    ("pattern_falling_wedge", SemanticOrientation.BULLISH),
    ("pattern_broadening_wedge", SemanticOrientation.NEUTRAL),
    # Divers
    ("pattern_advance_block", SemanticOrientation.BEARISH),
    # Gaps (neutres car direction depend du contexte)
    ("pattern_breakaway_gap", SemanticOrientation.NEUTRAL),
    ("pattern_exhaustion_gap", SemanticOrientation.BEARISH),
    ("pattern_runaway_gap", SemanticOrientation.NEUTRAL),
    # Dojis purs (neutres)
    ("pattern_doji", SemanticOrientation.NEUTRAL),
    ("pattern_four_price_doji", SemanticOrientation.NEUTRAL),
    ("pattern_long_legged_doji", SemanticOrientation.NEUTRAL),
    # Stars (neutres pour les doji, contexte-dependant pour les autres)
    ("pattern_morning_star", SemanticOrientation.BULLISH),
    ("pattern_evening_star", SemanticOrientation.BEARISH),
    ("pattern_tri_star", SemanticOrientation.NEUTRAL),
    ("pattern_rickshaw_man", SemanticOrientation.NEUTRAL),
    # Multi-candle / indcision
    ("pattern_spinning_top", SemanticOrientation.NEUTRAL),
    ("pattern_high_wave_candle", SemanticOrientation.NEUTRAL),
    ("pattern_piercing_line", SemanticOrientation.BULLISH),
    ("pattern_dark_cloud_cover", SemanticOrientation.BEARISH),
    # Confirmed reversal
    ("pattern_three_inside_up", SemanticOrientation.BULLISH),
    ("pattern_three_inside_down", SemanticOrientation.BEARISH),
    ("pattern_three_outside_up", SemanticOrientation.BULLISH),
    ("pattern_three_outside_down", SemanticOrientation.BEARISH),
    # Divers
    ("pattern_unique_three_river_bottom", SemanticOrientation.BULLISH),
    ("pattern_deliberation", SemanticOrientation.NEUTRAL),
    ("pattern_concealing_baby_swallow", SemanticOrientation.BULLISH),
    ("pattern_kicking_bull", SemanticOrientation.BULLISH),
    ("pattern_kicking_bear", SemanticOrientation.BEARISH),
    ("pattern_ladder_bottom", SemanticOrientation.BULLISH),
    ("pattern_ladder_top", SemanticOrientation.BEARISH),
    ("pattern_matching_high", SemanticOrientation.NEUTRAL),
    ("pattern_matching_low", SemanticOrientation.NEUTRAL),
    # Regimes
    ("pattern_uptrend", SemanticOrientation.BULLISH),
    ("pattern_downtrend", SemanticOrientation.BEARISH),
    ("pattern_sideways_trend", SemanticOrientation.NEUTRAL),
    # S/R (neutres)
    ("pattern_support", SemanticOrientation.NEUTRAL),
    ("pattern_resistance", SemanticOrientation.NEUTRAL),
    # Islands (neutres par design — gap rare, direction contexte)
    ("pattern_island_reversal", SemanticOrientation.NEUTRAL),
    ("pattern_island_top", SemanticOrientation.NEUTRAL),
    ("pattern_island_bottom", SemanticOrientation.NEUTRAL),
    # Three drives (neutre)
    ("pattern_three_drives", SemanticOrientation.NEUTRAL),
    # Elliott (neutres — direction = contexte)
    ("pattern_elliott_wave_1", SemanticOrientation.NEUTRAL),
    ("pattern_elliott_wave_3", SemanticOrientation.NEUTRAL),
    ("pattern_elliott_wave_5", SemanticOrientation.NEUTRAL),
    # Harmoniques (neutres — direction explicite via bull/bear)
    ("pattern_gartley_bull", SemanticOrientation.BULLISH),
    ("pattern_gartley_bear", SemanticOrientation.BEARISH),
    ("pattern_butterfly_bull", SemanticOrientation.BULLISH),
    ("pattern_butterfly_bear", SemanticOrientation.BEARISH),
    ("pattern_bat_bull", SemanticOrientation.BULLISH),
    ("pattern_bat_bear", SemanticOrientation.BEARISH),
    ("pattern_crab_bull", SemanticOrientation.BULLISH),
    ("pattern_crab_bear", SemanticOrientation.BEARISH),
    ("pattern_shark_bull", SemanticOrientation.BULLISH),
    ("pattern_shark_bear", SemanticOrientation.BEARISH),
    # Fibonacci (neutres)
    ("pattern_fibonacci_retracement", SemanticOrientation.NEUTRAL),
    ("pattern_fibonacci_extension", SemanticOrientation.NEUTRAL),
    # Wolfe Wave (neutre)
    ("pattern_wolfe_wave", SemanticOrientation.NEUTRAL),
    # Cup & Handle (haussier continuation)
    ("pattern_cup_handle", SemanticOrientation.BULLISH),
    # Spike reversal (neutre)
    ("pattern_spike_reversal", SemanticOrientation.NEUTRAL),
    # Measured move, rectangle (neutres)
    ("pattern_measured_move", SemanticOrientation.NEUTRAL),
    ("pattern_rectangle", SemanticOrientation.NEUTRAL),
    # Pin bars
    ("pattern_pin_bar_bull", SemanticOrientation.BULLISH),
    ("pattern_pin_bar_bear", SemanticOrientation.BEARISH),
)


# Suffixes qui indiquent une orientation BULLISH (fallback apres exact rules)
_BULLISH_SUFFIXES: tuple[str, ...] = (
    "_bull",
    "_bottom",
    "_up",
    "_long",
    "_white_soldiers",
    "_uptrend",
    "_inv_head_shoulders",
    "_bullish_breakout_on_volume",
    "_rounding_bottom",
    "_v_bottom",
    "_engulfing_bull",
    "_harami_bull",
    "_abandoned_baby_bull",
    "_breakaway_bull",
    "_belt_hold_bull",
    "_morning_star",
    "_dragonfly_doji",
    "_hammer",
    "_inverted_hammer",
    "_three_white_soldiers",
)

# Suffixes qui indiquent une orientation BEARISH
_BEARISH_SUFFIXES: tuple[str, ...] = (
    "_bear",
    "_top",
    "_down",
    "_short",
    "_black_crows",
    "_downtrend",
    "_head_shoulders",
    "_rising_wedge",
    "_three_black_crows",
    "_rounding_top",
    "_v_top",
    "_engulfing_bear",
    "_harami_bear",
    "_dark_cloud_cover",
    "_abandoned_baby_bear",
    "_breakaway_bear",
    "_belt_hold_bear",
    "_evening_star",
    "_gravestone_doji",
    "_hanging_man",
    "_shooting_star",
    "_pin_bar_bear",
    "_advance_block",
    "_three_inside_down",
    "_three_outside_down",
)


def _classify_pattern_orientation(pattern_name: str) -> SemanticOrientation:
    """Classifie un pattern en BULLISH / BEARISH / NEUTRAL via heuristique.

    Algorithme par ordre de priorite :
      1. Cas exacts (dojis speciaux, triangles, wedges, etc.).
      2. Suffixes BULLISH.
      3. Suffixes BEARISH.
      4. NEUTRAL par defaut.

    Args:
        pattern_name: nom du pattern (ex: "pattern_hammer", "pattern_doji").

    Returns:
        SemanticOrientation correspondant.
    """
    if not pattern_name.startswith("pattern_"):
        return SemanticOrientation.NEUTRAL
    # 1) Cas exacts
    for name, orient in _EXACT_RULES:
        if name == pattern_name:
            return orient
    # 2) Suffixes bullish
    for suffix in _BULLISH_SUFFIXES:
        if pattern_name.endswith(suffix):
            return SemanticOrientation.BULLISH
    # 3) Suffixes bearish
    for suffix in _BEARISH_SUFFIXES:
        if pattern_name.endswith(suffix):
            return SemanticOrientation.BEARISH
    # 4) Defaut
    return SemanticOrientation.NEUTRAL


# --------------------------------------------------------------------------- #
# Mapping global (cache a l'import)
# --------------------------------------------------------------------------- #


def _build_orientation_table() -> dict[str, SemanticOrientation]:
    """Construit le mapping pattern_name -> SemanticOrientation au chargement.

    Importe la taxonomie depuis config (lazy pour eviter cycles).
    """
    from einherjar.research.config.loader import load_config
    from einherjar.research.generators.bnf import FEATURE_GRAMMARS

    cfg = load_config("src/einherjar/research/config")
    table: dict[str, SemanticOrientation] = {}
    for feat_name in cfg.usable_feature_names:
        if feat_name.startswith("pattern_"):
            table[feat_name] = _classify_pattern_orientation(feat_name)
    # Aussi les features du bloc relations OHLCV
    for rel_key, rel_grammar in [
        ("ohlcv", "ohlcv"),
    ]:
        # Les relations OHLCV sont classees par le caller (cas special,
        # pas un pattern mais un bloc). On les marque NEUTRAL ici.
        table[f"__relations_{rel_key}__"] = SemanticOrientation.NEUTRAL
    logger.info(
        "Orientation semantique : %d patterns classes "
        "(BULLISH=%d, BEARISH=%d, NEUTRAL=%d)",
        len(table),
        sum(1 for v in table.values() if v == SemanticOrientation.BULLISH),
        sum(1 for v in table.values() if v == SemanticOrientation.BEARISH),
        sum(1 for v in table.values() if v == SemanticOrientation.NEUTRAL),
    )
    return table


# Construit une seule fois au premier import (cache module-level).
PATTERN_ORIENTATION: dict[str, SemanticOrientation] = _build_orientation_table()


# --------------------------------------------------------------------------- #
# API publique
# --------------------------------------------------------------------------- #


def get_orientation(feature_or_source: str) -> SemanticOrientation:
    """Retourne l'orientation semantique d'une feature ou d'une source BNF.

    Args:
        feature_or_source: nom de feature (ex: "pattern_hammer") ou
            cle speciale de bloc (ex: "__ohlcv_relations__").

    Returns:
        SemanticOrientation (NEUTRAL par defaut si non trouve).
    """
    # Cas special : bloc relations OHLCV
    if feature_or_source == "__ohlcv_relations__":
        return SemanticOrientation.NEUTRAL
    return PATTERN_ORIENTATION.get(
        feature_or_source, SemanticOrientation.NEUTRAL,
    )


def orientation_summary() -> dict[str, int]:
    """Retourne un resume du nombre de patterns par orientation.

    Utile pour le pilotage (rapport par moteur, statistiques globales).
    """
    counts: dict[str, int] = {
        SemanticOrientation.BULLISH.value: 0,
        SemanticOrientation.BEARISH.value: 0,
        SemanticOrientation.NEUTRAL.value: 0,
    }
    for orient in PATTERN_ORIENTATION.values():
        if orient.value in counts:
            counts[orient.value] += 1
    return counts
