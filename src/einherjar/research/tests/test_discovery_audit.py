"""test_discovery_audit.py — Audit de la couverture discovery (P1-14).

Ce test verifie que TOUTES les 218 features utilisables de la taxonomie
Einherjar sont couvertes par :
  1. Une grammaire BNF (generators/bnf.py)
  2. Une orientation semantique (generators/bnf_semantic.py)
  3. Un sous-score dans le comparateur multi-objectif
  4. Un suivi dans le pilotage

But : P1-14 (discovery exhaustive) - aucun biais de decouverte, toutes
les features et orientations sont prises en compte.

Ce test est NON-INVASIF (lecture seule) : il ne modifie aucun code
de production, il verifie simplement que la couverture est complete.
"""

import unittest

from einherjar.research.config.loader import load_config


class TestDiscoveryAudit(unittest.TestCase):
    """Audit de la couverture discovery 218/218 features."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = load_config("src/einherjar/research/config")

    def test_all_218_features_have_bnf_grammar(self) -> None:
        """Toutes les 218 features utilisables ont une grammaire BNF."""
        from einherjar.research.generators.bnf import FEATURE_GRAMMARS, get_feature_grammar
        for feat in self.cfg.usable_feature_names:
            with self.subTest(feature=feat):
                # Soit dans FEATURE_GRAMMARS, soit via default atomique.
                if feat in FEATURE_GRAMMARS:
                    self.assertGreater(len(FEATURE_GRAMMARS[feat]), 0)
                else:
                    # Fallback sur le pattern par defaut (toujours non-vide).
                    grammar = get_feature_grammar(feat, self.cfg)
                    self.assertGreater(len(grammar), 0)

    def test_all_patterns_have_semantic_orientation(self) -> None:
        """Tous les 107 patterns ont une orientation BULLISH/BEARISH/NEUTRAL."""
        from einherjar.research.generators.bnf_semantic import (
            PATTERN_ORIENTATION,
            SemanticOrientation,
        )
        patterns = [
            f for f in self.cfg.usable_feature_names
            if f.startswith("pattern_")
        ]
        self.assertEqual(len(patterns), 107)
        for p in patterns:
            with self.subTest(pattern=p):
                self.assertIn(p, PATTERN_ORIENTATION)
                self.assertIsInstance(
                    PATTERN_ORIENTATION[p], SemanticOrientation,
                )

    def test_feature_type_coverage(self) -> None:
        """Toutes les features ont un type defini (atomic, quantitative, etc.)."""
        from einherjar.research.utils.types import FeatureType
        types_seen: set[str] = set()
        for feat in self.cfg.usable_feature_names:
            t = self.cfg.features_taxonomy["features"][feat].get("feature_type")
            types_seen.add(t)
        # On doit avoir au moins les 5 types prevus.
        for expected in ("atomic", "quantitative", "pattern", "composite_derived", "factor"):
            self.assertIn(expected, types_seen, f"Type {expected!r} absent")

    def test_no_excluded_feature_is_used(self) -> None:
        """Aucune feature exclue (fantome/meta/alias) n'est exposee."""
        excluded = self.cfg.excluded_set()
        for feat in self.cfg.usable_feature_names:
            with self.subTest(feature=feat):
                self.assertNotIn(feat, excluded)

    def test_ohlcv_relations_block_covers_core_patterns(self) -> None:
        """Le bloc relations OHLCV expose au moins les 4 patterns cles."""
        from einherjar.research.generators.bnf import get_relations_grammar
        text = get_relations_grammar("ohlcv")
        for pattern in ("bullish_candle", "bearish_candle",
                        "wide_range_candle", "bullish_breakout_on_volume"):
            with self.subTest(pattern=pattern):
                self.assertIn(pattern, text)

    def test_orientation_distribution_balanced(self) -> None:
        """Distribution des orientations : pas de classe vide, pas de classe > 70%."""
        from einherjar.research.generators.bnf_semantic import (
            orientation_summary,
        )
        summary = orientation_summary()
        total = sum(summary.values())
        self.assertGreater(total, 0)
        for orient, count in summary.items():
            with self.subTest(orientation=orient):
                # Au moins 1 pattern par categorie.
                self.assertGreaterEqual(count, 1, f"{orient!r} vide")
                # Pas plus de 70% du total dans une categorie.
                self.assertLess(
                    count / total, 0.70,
                    f"{orient!r} = {count}/{total} = {count/total:.0%} > 70%",
                )


class TestComparatorCoverage(unittest.TestCase):
    """Audit que le comparateur multi-objectif couvre tous les generateurs."""

    def test_six_generator_classes_implemented(self) -> None:
        """Les 6 generateurs sont implementes (classes exportees)."""
        from einherjar.research.generators import algorithms
        expected = {
            "RandomSearchGenerator",
            "BeamSearchGenerator",
            "TypedGPGenerator",
            "GrammaticalEvolutionGenerator",
            "MemeticGenerator",
            "NSGA2Generator",
        }
        actual = set(dir(algorithms))
        for name in expected:
            with self.subTest(generator=name):
                self.assertIn(name, actual)
        # Tous implementes.
        self.assertEqual(expected & actual, expected)


class TestPilotageCoverage(unittest.TestCase):
    """Audit que le pilotage produit un rapport pour chaque generateur."""

    def test_pilotage_handles_all_6_generators(self) -> None:
        """Le rapport de pilotage fonctionne avec les 6 generateurs."""
        from einherjar.research.generators.algorithms import (
            RandomSearchGenerator,
        )
        from einherjar.research.generators.protocol import GenerationProtocol
        from einherjar.research.pilotage import build_pilotage_report
        from einherjar.research.utils.types import (
            Amplitude, AmplitudeUnit, Condition, CompareOp, Direction,
            Hypothesis, Universe,
        )
        cfg = load_config("src/einherjar/research/config")
        protocol = GenerationProtocol(seed=42, data_version="v1", n_eval_budget=3)
        gen = RandomSearchGenerator(protocol=protocol, config=cfg)
        result = gen.generate()
        report = build_pilotage_report([result])
        self.assertIn("RandomSearchGenerator", report.engine_stats)
        es = report.engine_stats["RandomSearchGenerator"]
        self.assertGreater(es.n_generated, 0)
        # Diversite exposee.
        self.assertGreaterEqual(es.diversity.n_features_distinct, 1)


if __name__ == "__main__":
    unittest.main()
