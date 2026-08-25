"""test_feature_dedup.py - Test Sprint 2.2.1.

Anti-duplication de features par matrice de correlation.

Verifie que :
- Apres dedup, plus aucune paire n'a |r| > threshold
- Les features avec importance elevee sont preservees
- Le nombre de features retenues < nombre initial
"""
from __future__ import annotations

import unittest

import numpy as np

from einherjar.research.xgb_einhers.data_loader import load_xy
from einherjar.research.xgb_einhers.feature_dedup import (
    apply_dedup,
    compute_corr_matrix,
    find_duplicate_pairs,
)


class TestCorrelation(unittest.TestCase):
    """Tests unitaires sur la matrice de correlation."""

    def test_perfect_correlation_detected(self):
        """Deux features identiques => |r| = 1.0, doit etre detecte."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(100)
        X = np.column_stack([x, x, rng.standard_normal(100)])  # col 0 == col 1
        corr = compute_corr_matrix(X)
        # |corr(a, b)| doit etre 1.0
        self.assertAlmostEqual(corr[0, 1], 1.0, places=5)

    def test_diagonal_is_zero(self):
        """La diagonale (auto-correlation) doit etre mise a 0."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((50, 4))
        corr = compute_corr_matrix(X)
        np.testing.assert_array_equal(np.diag(corr), np.zeros(4))

    def test_orthogonal_features_have_low_corr(self):
        """Features independantes => |r| proche de 0."""
        rng = np.random.default_rng(1)
        X = rng.standard_normal((1000, 3))
        corr = compute_corr_matrix(X)
        for i in range(3):
            for j in range(3):
                if i != j:
                    self.assertLess(abs(corr[i, j]), 0.1)

    def test_find_duplicate_pairs(self):
        """Trouve les paires correlees au-dessus du seuil."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(100)
        X = np.column_stack([x, x * 0.99 + 0.01, rng.standard_normal(100), rng.standard_normal(100)])
        names = ["dup_a", "dup_b", "rand_a", "rand_b"]
        corr = compute_corr_matrix(X)
        pairs = find_duplicate_pairs(corr, names, threshold=0.85)
        # Doit trouver (dup_a, dup_b) au moins
        self.assertGreaterEqual(len(pairs), 1)
        feat_names_in_pairs = {p[0] for p in pairs} | {p[1] for p in pairs}
        self.assertIn("dup_a", feat_names_in_pairs)
        self.assertIn("dup_b", feat_names_in_pairs)


class TestDedupPipeline(unittest.TestCase):
    """Tests integration sur BTCUSD."""

    @classmethod
    def setUpClass(cls):
        cls.loaded = load_xy("BTCUSD", "1h", "crypto")
        cls.X = cls.loaded.X[:5000]  # subset pour vitesse
        cls.feature_names = list(cls.loaded.feature_names)

    def test_dedup_drops_correlated_features(self):
        """Apres dedup, on doit avoir moins de features."""
        # Importance uniforme (toutes egales) -> on garde la 1re de chaque paire
        importances = {name: 1.0 for name in self.feature_names}
        X_dedup, kept, dropped = apply_dedup(
            self.X, self.feature_names, importances, corr_threshold=0.85,
        )
        self.assertLess(X_dedup.shape[1], self.X.shape[1],
                        f"Dedup n'a rien droppe : {X_dedup.shape[1]} == {self.X.shape[1]}")
        self.assertEqual(X_dedup.shape[1], len(kept))
        self.assertEqual(len(kept) + len(dropped), len(self.feature_names))

    def test_dedup_respects_importance(self):
        """Si une feature a une importance elevee, elle est preservee."""
        # Importance = 100 pour rsi_14, 1 pour le reste
        # Si rsi_14 a une copie (correlation > 0.85 avec rsi_28 ou similaire),
        # elle doit etre preservee.
        importances = {name: 1.0 for name in self.feature_names}
        if "rsi_14" in importances:
            importances["rsi_14"] = 100.0
        X_dedup, kept, dropped = apply_dedup(
            self.X, self.feature_names, importances, corr_threshold=0.85,
        )
        if "rsi_14" in self.feature_names:
            self.assertIn("rsi_14", kept,
                          "rsi_14 avec importance 100 doit etre preserve")

    def test_no_pair_above_threshold_after_dedup(self):
        """Apres dedup, aucune paire ne doit avoir |r| > 0.85."""
        importances = {name: 1.0 for name in self.feature_names}
        X_dedup, kept, dropped = apply_dedup(
            self.X, self.feature_names, importances, corr_threshold=0.85,
        )
        if X_dedup.shape[1] < 2:
            self.skipTest("Trop peu de features apres dedup")
        corr = compute_corr_matrix(X_dedup)
        max_corr = float(np.max(corr))
        self.assertLessEqual(max_corr, 0.85 + 1e-6,
                             f"Apres dedup, max |r| = {max_corr:.3f} > 0.85")

    def test_dedup_idempotent(self):
        """Appliquer dedup 2x doit donner le meme resultat."""
        importances = {name: 1.0 for name in self.feature_names}
        X1, kept1, _ = apply_dedup(self.X, self.feature_names, importances, corr_threshold=0.85)
        # Re-appliquer sur le resultat
        importances2 = {name: 1.0 for name in kept1}
        X2, kept2, _ = apply_dedup(X1, kept1, importances2, corr_threshold=0.85)
        self.assertEqual(kept1, kept2, "Dedup n'est pas idempotent")


if __name__ == "__main__":
    unittest.main()
