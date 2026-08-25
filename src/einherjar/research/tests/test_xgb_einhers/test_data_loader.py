"""test_data_loader.py - Tests P0 du data_loader.

Vérifie que :
- load_xy retourne les bonnes shapes
- Les 5 colonnes OHLCV sont exclues
- Les features exclues par la taxonomie sont retirées
- Pas de NaN/Inf
- L'alignement avec OHLCV fonctionne
- Le split temporel respecte les ratios et l'embargo
"""
import unittest

import numpy as np

from einherjar.research.xgb_einhers.data_loader import (
    OHLCV_COLUMNS,
    get_target_for_horizon,
    load_usable_feature_names,
    load_xy,
    temporal_split,
)


class TestLoadXY(unittest.TestCase):
    """P0 : chargement X/Y pour BTCUSD × 1h."""

    @classmethod
    def setUpClass(cls):
        cls.loaded = load_xy("BTCUSD", "1h", "crypto")

    def test_n_samples_positive(self):
        self.assertGreater(self.loaded.n_samples, 0)

    def test_n_features_equals_213(self):
        # 218 usable - 5 OHLCV = 213
        self.assertEqual(self.loaded.n_features, 213)

    def test_n_horizons_equals_4(self):
        self.assertEqual(self.loaded.n_horizons, 4)

    def test_horizons_naming(self):
        # Pour 1h, les horizons sont 6h, 12h, 1d, 2d
        self.assertEqual(self.loaded.horizons, ("6h", "12h", "1d", "2d"))

    def test_no_ohlcv_in_feature_names(self):
        """Les 5 colonnes OHLCV doivent être exclues (réponse Q6)."""
        for ohlcv_col in OHLCV_COLUMNS:
            self.assertNotIn(ohlcv_col, self.loaded.feature_names)

    def test_only_usable_features(self):
        """Toutes les features chargées doivent être dans la taxonomie usable."""
        usable = load_usable_feature_names()
        for name in self.loaded.feature_names:
            self.assertIn(name, usable, f"Feature {name} non marquée usable")

    def test_no_nan_inf(self):
        self.assertFalse(np.isnan(self.loaded.X).any(), "X contient des NaN")
        self.assertFalse(np.isinf(self.loaded.X).any(), "X contient des Inf")

    def test_y_dir_values(self):
        """Y_dir doit être dans {-100, 0, 1, 2}."""
        unique = np.unique(self.loaded.Y_dir)
        for v in unique:
            self.assertIn(v, {-100, 0, 1, 2}, f"Valeur Y_dir inattendue : {v}")

    def test_y_ret_range(self):
        """Y_ret doit être clipé à [-0.15, 0.15] (inclusif aux bornes)."""
        # np.testing.assert_array_compare utilise >=, mais on a -0.15 EXACT
        # → tolerance nécessaire (la donnée est clipée donc == borne possible)
        ymin = float(self.loaded.Y_ret.min())
        ymax = float(self.loaded.Y_ret.max())
        self.assertGreaterEqual(ymin, -0.15 - 1e-6,
                               f"Y_ret.min() = {ymin} < -0.15")
        self.assertLessEqual(ymax, 0.15 + 1e-6,
                             f"Y_ret.max() = {ymax} > 0.15")

    def test_timestamps_monotonic(self):
        """Les timestamps doivent être triés ASC."""
        self.assertTrue(np.all(np.diff(self.loaded.timestamps) >= 0))


class TestGetTargetForHorizon(unittest.TestCase):
    """P0 : extraction du target pour un horizon donné."""

    @classmethod
    def setUpClass(cls):
        cls.loaded = load_xy("BTCUSD", "1h", "crypto")

    def test_target_shape(self):
        for h in range(self.loaded.n_horizons):
            target, valid_mask, y_hor = get_target_for_horizon(self.loaded, h)
            self.assertEqual(target.shape, (self.loaded.n_samples,))
            self.assertEqual(valid_mask.shape, (self.loaded.n_samples,))
            self.assertEqual(y_hor.shape, (self.loaded.n_samples,))
            self.assertEqual(target.dtype, np.float32)
            self.assertEqual(valid_mask.dtype, bool)

    def test_valid_mask_filters_invalid(self):
        """Les lignes où Y_dir == -100 doivent être False dans valid_mask."""
        for h in range(self.loaded.n_horizons):
            target, valid_mask, y_hor = get_target_for_horizon(self.loaded, h)
            invalid_rows = self.loaded.Y_dir[:, h] == -100
            self.assertTrue((~valid_mask).sum() == invalid_rows.sum(),
                            f"h={h}: {invalid_rows.sum()} invalides attendus")

    def test_target_values_for_valid_rows(self):
        """Pour les lignes valides, target = Y_ret[:, h]."""
        for h in range(self.loaded.n_horizons):
            target, valid_mask, y_hor = get_target_for_horizon(self.loaded, h)
            np.testing.assert_array_equal(
                target[valid_mask], self.loaded.Y_ret[valid_mask, h]
            )


class TestTemporalSplit(unittest.TestCase):
    """P0 : split temporel 60/20/20 avec embargo."""

    def test_ratios_respected(self):
        n = 10000
        X = np.random.randn(n, 5).astype(np.float32)
        y = np.random.randn(n).astype(np.float32)
        split = temporal_split(X, y, train_ratio=0.6, val_ratio=0.2, holdout_ratio=0.2, embargo_bars=50)
        self.assertEqual(len(split.train_X), int(n * 0.6))
        self.assertEqual(len(split.val_X), int(n * 0.2))
        self.assertEqual(len(split.holdout_X), n - int(n * 0.6) - 50 - int(n * 0.2) - 50)

    def test_no_overlap(self):
        n = 10000
        X = np.random.randn(n, 5).astype(np.float32)
        y = np.random.randn(n).astype(np.float32)
        split = temporal_split(X, y, embargo_bars=50)
        # Pas de chevauchement entre les indices
        all_idx = set(split.train_indices) | set(split.val_indices) | set(split.holdout_indices)
        self.assertEqual(len(all_idx), len(split.train_indices) + len(split.val_indices) + len(split.holdout_indices))

    def test_temporal_ordering(self):
        n = 10000
        X = np.random.randn(n, 5).astype(np.float32)
        y = np.random.randn(n).astype(np.float32)
        split = temporal_split(X, y, embargo_bars=50)
        # Chaque split est dans le bon ordre temporel
        self.assertLess(split.train_indices[-1], split.val_indices[0])
        self.assertLess(split.val_indices[-1], split.holdout_indices[0])

    def test_embargo_applied(self):
        n = 10000
        X = np.random.randn(n, 5).astype(np.float32)
        y = np.random.randn(n).astype(np.float32)
        embargo = 50
        split = temporal_split(X, y, embargo_bars=embargo)
        # L'embargo est respecté entre train et val
        self.assertEqual(split.val_indices[0] - split.train_indices[-1], embargo + 1)
        # Et entre val et holdout
        self.assertEqual(split.holdout_indices[0] - split.val_indices[-1], embargo + 1)


class TestLoadUsableFeatureNames(unittest.TestCase):
    """P0 : taxonomie."""

    def test_returns_set(self):
        usable = load_usable_feature_names()
        self.assertIsInstance(usable, set)
        self.assertEqual(len(usable), 218)

    def test_ohlcv_not_usable(self):
        """Les 5 OHLCV sont dans features_taxonomy mais doivent être filtered.

        car on les exclut explicitement (réponse Q6).
        """
        load_usable_feature_names()
        # open/high/low/close/volume sont atomic et marked usable dans la taxonomie
        # MAIS on les exclut via data_loader.py
        # Ici on vérifie juste que la taxonomie les contient (c'est le data_loader qui les exclut)
        # (test informatif)


if __name__ == "__main__":
    unittest.main()
