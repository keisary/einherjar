"""Tests pour l'orientation semantique des patterns (BNF Phase 3).

Couvre :
  - Enum SemanticOrientation (3 valeurs)
  - PATTERN_ORIENTATION : couverture des 107 patterns
  - Heuristique de classification : cas exacts, suffixes, defaut
  - Cas speciaux (dojis, triangles, wedges, harmoniques, regimes)
  - get_orientation() : cle de bloc relations OHLCV
  - orientation_summary() : comptage par orientation
"""

import unittest

from einherjar.research.generators.bnf_semantic import (
    PATTERN_ORIENTATION,
    SemanticOrientation,
    get_orientation,
    orientation_summary,
    _classify_pattern_orientation,
)


class TestSemanticOrientation(unittest.TestCase):
    """Tests basiques de l'enum et des helpers."""

    def test_enum_values(self) -> None:
        """L'enum a 3 valeurs : BULLISH, BEARISH, NEUTRAL."""
        self.assertEqual(SemanticOrientation.BULLISH.value, "bullish")
        self.assertEqual(SemanticOrientation.BEARISH.value, "bearish")
        self.assertEqual(SemanticOrientation.NEUTRAL.value, "neutral")

    def test_get_orientation_non_pattern(self) -> None:
        """Une feature non-pattern (OHLCV, quantitative) est NEUTRAL."""
        self.assertEqual(get_orientation("open"), SemanticOrientation.NEUTRAL)
        self.assertEqual(get_orientation("rsi_14"), SemanticOrientation.NEUTRAL)
        self.assertEqual(get_orientation("Factor_Momentum_Score"), SemanticOrientation.NEUTRAL)

    def test_get_orientation_unknown(self) -> None:
        """Une cle inconnue retourne NEUTRAL (defaut conservateur)."""
        self.assertEqual(get_orientation("unknown_thing"), SemanticOrientation.NEUTRAL)

    def test_get_orientation_ohlcv_relations(self) -> None:
        """Le bloc relations OHLCV est explicitement NEUTRAL."""
        self.assertEqual(
            get_orientation("__ohlcv_relations__"), SemanticOrientation.NEUTRAL,
        )

    def test_orientation_summary_keys(self) -> None:
        """Le summary a 3 cles (BULLISH, BEARISH, NEUTRAL) avec compteurs >= 0."""
        summary = orientation_summary()
        self.assertIn("bullish", summary)
        self.assertIn("bearish", summary)
        self.assertIn("neutral", summary)
        self.assertGreaterEqual(summary["bullish"], 0)
        self.assertGreaterEqual(summary["bearish"], 0)
        self.assertGreaterEqual(summary["neutral"], 0)
        # Au moins un pattern dans chaque categorie (coherence avec la taxonomie)
        self.assertGreater(summary["bullish"] + summary["bearish"] + summary["neutral"], 100)


class TestPatternOrientationCoverage(unittest.TestCase):
    """Tests de couverture : tous les 107 patterns ont une orientation."""

    def test_all_patterns_covered(self) -> None:
        """Les 107 patterns de la taxonomie sont tous dans PATTERN_ORIENTATION."""
        from einherjar.research.config.loader import load_config
        cfg = load_config("src/einherjar/research/config")
        patterns = [
            f for f in cfg.usable_feature_names if f.startswith("pattern_")
        ]
        self.assertEqual(len(patterns), 107)
        for p in patterns:
            self.assertIn(p, PATTERN_ORIENTATION, f"Pattern {p!r} non classifie")
            orient = PATTERN_ORIENTATION[p]
            self.assertIsInstance(orient, SemanticOrientation)

    def test_orientation_summary_total(self) -> None:
        """Le total du summary = 107 + 1 (relations OHLCV)."""
        summary = orientation_summary()
        total = sum(summary.values())
        self.assertEqual(total, 108)


class TestHeuristicBullish(unittest.TestCase):
    """Patterns clairement haussiers."""

    def test_hammer(self) -> None:
        self.assertEqual(
            _classify_pattern_orientation("pattern_hammer"),
            SemanticOrientation.BULLISH,
        )

    def test_dragonfly_doji(self) -> None:
        """Dragonfly doji (open = close = high, longue meche basse) = BULLISH."""
        self.assertEqual(
            _classify_pattern_orientation("pattern_dragonfly_doji"),
            SemanticOrientation.BULLISH,
        )

    def test_morning_star(self) -> None:
        self.assertEqual(
            _classify_pattern_orientation("pattern_morning_star"),
            SemanticOrientation.BULLISH,
        )

    def test_ascending_triangle(self) -> None:
        self.assertEqual(
            _classify_pattern_orientation("pattern_ascending_triangle"),
            SemanticOrientation.BULLISH,
        )

    def test_falling_wedge(self) -> None:
        self.assertEqual(
            _classify_pattern_orientation("pattern_falling_wedge"),
            SemanticOrientation.BULLISH,
        )

    def test_inv_head_shoulders(self) -> None:
        self.assertEqual(
            _classify_pattern_orientation("pattern_inv_head_shoulders"),
            SemanticOrientation.BULLISH,
        )

    def test_cup_handle(self) -> None:
        """Cup & Handle = continuation haussiere."""
        self.assertEqual(
            _classify_pattern_orientation("pattern_cup_handle"),
            SemanticOrientation.BULLISH,
        )

    def test_three_white_soldiers(self) -> None:
        self.assertEqual(
            _classify_pattern_orientation("pattern_three_white_soldiers"),
            SemanticOrientation.BULLISH,
        )

    def test_uptrend(self) -> None:
        self.assertEqual(
            _classify_pattern_orientation("pattern_uptrend"),
            SemanticOrientation.BULLISH,
        )

    def test_gartley_bull(self) -> None:
        """Gartley bull = harmonique haussiere."""
        self.assertEqual(
            _classify_pattern_orientation("pattern_gartley_bull"),
            SemanticOrientation.BULLISH,
        )

    def test_unique_three_river_bottom(self) -> None:
        self.assertEqual(
            _classify_pattern_orientation("pattern_unique_three_river_bottom"),
            SemanticOrientation.BULLISH,
        )


class TestHeuristicBearish(unittest.TestCase):
    """Patterns clairement baissiers."""

    def test_hanging_man(self) -> None:
        self.assertEqual(
            _classify_pattern_orientation("pattern_hanging_man"),
            SemanticOrientation.BEARISH,
        )

    def test_shooting_star(self) -> None:
        self.assertEqual(
            _classify_pattern_orientation("pattern_shooting_star"),
            SemanticOrientation.BEARISH,
        )

    def test_evening_star(self) -> None:
        self.assertEqual(
            _classify_pattern_orientation("pattern_evening_star"),
            SemanticOrientation.BEARISH,
        )

    def test_descending_triangle(self) -> None:
        self.assertEqual(
            _classify_pattern_orientation("pattern_descending_triangle"),
            SemanticOrientation.BEARISH,
        )

    def test_rising_wedge(self) -> None:
        self.assertEqual(
            _classify_pattern_orientation("pattern_rising_wedge"),
            SemanticOrientation.BEARISH,
        )

    def test_head_shoulders(self) -> None:
        self.assertEqual(
            _classify_pattern_orientation("pattern_head_shoulders"),
            SemanticOrientation.BEARISH,
        )

    def test_three_black_crows(self) -> None:
        self.assertEqual(
            _classify_pattern_orientation("pattern_three_black_crows"),
            SemanticOrientation.BEARISH,
        )

    def test_downtrend(self) -> None:
        self.assertEqual(
            _classify_pattern_orientation("pattern_downtrend"),
            SemanticOrientation.BEARISH,
        )

    def test_gartley_bear(self) -> None:
        self.assertEqual(
            _classify_pattern_orientation("pattern_gartley_bear"),
            SemanticOrientation.BEARISH,
        )

    def test_advance_block(self) -> None:
        """Advance block = essoufflement haussier = signal BEARISH."""
        self.assertEqual(
            _classify_pattern_orientation("pattern_advance_block"),
            SemanticOrientation.BEARISH,
        )


class TestHeuristicNeutral(unittest.TestCase):
    """Patterns NEUTRAL (continuation, indecision, ou direction contexte)."""

    def test_doji(self) -> None:
        self.assertEqual(
            _classify_pattern_orientation("pattern_doji"),
            SemanticOrientation.NEUTRAL,
        )

    def test_spinning_top(self) -> None:
        self.assertEqual(
            _classify_pattern_orientation("pattern_spinning_top"),
            SemanticOrientation.NEUTRAL,
        )

    def test_sideways_trend(self) -> None:
        self.assertEqual(
            _classify_pattern_orientation("pattern_sideways_trend"),
            SemanticOrientation.NEUTRAL,
        )

    def test_support(self) -> None:
        self.assertEqual(
            _classify_pattern_orientation("pattern_support"),
            SemanticOrientation.NEUTRAL,
        )

    def test_resistance(self) -> None:
        self.assertEqual(
            _classify_pattern_orientation("pattern_resistance"),
            SemanticOrientation.NEUTRAL,
        )

    def test_symmetrical_triangle(self) -> None:
        self.assertEqual(
            _classify_pattern_orientation("pattern_symmetrical_triangle"),
            SemanticOrientation.NEUTRAL,
        )

    def test_broadening_wedge(self) -> None:
        self.assertEqual(
            _classify_pattern_orientation("pattern_broadening_wedge"),
            SemanticOrientation.NEUTRAL,
        )

    def test_elliott_wave_3(self) -> None:
        """Elliott : direction = contexte, NEUTRAL par defaut."""
        self.assertEqual(
            _classify_pattern_orientation("pattern_elliott_wave_3"),
            SemanticOrientation.NEUTRAL,
        )

    def test_fibonacci_retracement(self) -> None:
        self.assertEqual(
            _classify_pattern_orientation("pattern_fibonacci_retracement"),
            SemanticOrientation.NEUTRAL,
        )

    def test_wolfe_wave(self) -> None:
        self.assertEqual(
            _classify_pattern_orientation("pattern_wolfe_wave"),
            SemanticOrientation.NEUTRAL,
        )

    def test_island_reversal(self) -> None:
        """Island reversal : direction = contexte, force NEUTRAL."""
        self.assertEqual(
            _classify_pattern_orientation("pattern_island_reversal"),
            SemanticOrientation.NEUTRAL,
        )


if __name__ == "__main__":
    unittest.main()
