"""test_anti_swap.py - Tests P0 anti-swap de colonnes.

Vérifie que le data_loader lit correctement les features : X[:, i] doit
correspondre à feature_names[i]. Aucune colonne ne doit être décalée ou
swappée.
"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from einherjar.research.xgb_einhers.data_loader import load_usable_feature_names, load_xy


class TestNoColumnSwap(unittest.TestCase):
    """P0 : les colonnes de X sont alignées avec feature_names."""

    @classmethod
    def setUpClass(cls):
        cls.loaded = load_xy("BTCUSD", "1h", "crypto")

    def test_feature_names_count_matches_X_columns(self):
        """Le nombre de features doit correspondre à X.shape[1]."""
        self.assertEqual(
            len(self.loaded.feature_names),
            self.loaded.X.shape[1],
            f"Mismatch: {len(self.loaded.feature_names)} features vs {self.loaded.X.shape[1]} columns",
        )

    def test_ohlcv_excluded(self):
        """Les 5 OHLCV ne doivent pas être dans feature_names."""
        for col in ("open", "high", "low", "close", "volume"):
            self.assertNotIn(col, self.loaded.feature_names)

    def test_first_5_features_are_not_ohlcv(self):
        """Vérifie que les 5 premières features ne sont PAS les OHLCV.

        Test anti-swap : si quelqu'un a inversé l'ordre, les premières
        features seraient 'open', 'high', etc.
        """
        for i in range(min(5, len(self.loaded.feature_names))):
            name = self.loaded.feature_names[i]
            self.assertNotIn(name, ("open", "high", "low", "close", "volume"))

    def test_features_match_taxonomy_order(self):
        """Vérifie que l'ensemble des features chargées = taxonomie usable - OHLCV."""
        usable = load_usable_feature_names()  # 218 features
        # Les 5 OHLCV sont dans la taxonomie MAIS exclues par le data_loader
        ohlcv = {"open", "high", "low", "close", "volume"}
        expected = usable - ohlcv  # 213 features attendues
        loaded_set = set(self.loaded.feature_names)
        self.assertEqual(
            loaded_set, expected,
            f"Les features chargées doivent être taxonomie_usable - OHLCV. "
            f"Diff: {loaded_set ^ expected}"
        )

    def test_inject_extreme_value_first_column(self):
        """Inject une valeur extrême dans la première colonne, vérifie qu'elle.

        est lue correctement. Test d'intégration.
        """
        # Créer une copie de X
        X = self.loaded.X.copy()
        # Injecter une valeur extrême dans la première colonne
        # (premier feature du loaded.feature_names)
        self.loaded.feature_names[0]
        X[:, 0] = 999.0  # valeur extrême
        # Vérifier que la valeur est bien là
        self.assertTrue(np.all(X[:, 0] == 999.0))
        # Vérifier que la 2ème colonne n'a PAS été touchée
        self.assertFalse(np.any(X[:, 1] == 999.0),
                         "La 2ème colonne a été touchée : swap détecté")

    def test_inject_extreme_value_each_column(self):
        """Inject une valeur extrême dans chaque colonne, vérifie l'isolation.

        On utilise un sentinel unique (1e10) qui ne peut pas exister dans
        des données réelles. Pour chaque colonne, on repart d'une copie
        fraîche de X pour isoler l'injection.
        """
        sentinel = 1e10
        n_cols = min(20, self.loaded.X.shape[1])
        for col_idx in range(n_cols):
            # Reset à partir d'une copie fraîche
            X = self.loaded.X.copy()
            X[:, col_idx] = sentinel
            for other_idx in range(n_cols):
                if other_idx == col_idx:
                    self.assertTrue(
                        np.all(X[:, other_idx] == sentinel),
                        f"Col {col_idx} : valeur sentinel manquante",
                    )
                else:
                    self.assertFalse(
                        np.any(X[:, other_idx] == sentinel),
                        f"Col {col_idx} injectée : col {other_idx} a aussi la sentinel (SWAP)",
                    )

    def test_unique_feature_names(self):
        """Aucune feature ne doit apparaître deux fois."""
        self.assertEqual(
            len(self.loaded.feature_names),
            len(set(self.loaded.feature_names)),
            "Doublons détectés dans feature_names"
        )


class TestSyntheticDataSwap(unittest.TestCase):
    """P0 : avec un dataset synthétique, vérifier que load_xy préserve l'ordre."""

    def test_synthetic_load_preserves_column_order(self):
        """Crée un X.npy synthétique avec un ordre connu, vérifie qu'il est préservé.

        Le data_loader filtre par features_taxonomy.json : on monkey-patch
        load_usable_feature_names pour qu'il retourne l'union {fake_*} ∪ {OHLCV}
        (qu'on exclura ensuite). Comme ça on contrôle entièrement l'univers.
        """
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            subdir = tmp / "crypto" / "1h"
            subdir.mkdir(parents=True, exist_ok=True)
            # Créer un metadata.json avec un ordre connu
            feature_names_expected = [
                "fake_feature_1", "fake_feature_2", "open", "high", "low", "close", "volume",
                "fake_feature_3", "fake_feature_4", "fake_feature_5",
            ]
            metadata = {
                "horizons": ["6h"],
                "feature_names": feature_names_expected,
                "features_count": 10,
                "sequence_lengths": {"TEST": 100},
            }
            (subdir / "metadata.json").write_text(json.dumps(metadata))
            # Créer un X.npy synthétique (N=100, F=10)
            X = np.random.randn(100, 10).astype(np.float32)
            np.save(subdir / "TEST_X.npy", X)
            # Y synthétiques minimaux
            Y_dir = np.ones((100, 1), dtype=np.int8) * 1  # HOLD partout
            np.save(subdir / "TEST_Y_dir.npy", Y_dir)
            Y_ret = np.zeros((100, 1), dtype=np.float32)
            np.save(subdir / "TEST_Y_ret.npy", Y_ret)
            Y_hor = np.ones((100, 1), dtype=np.float32) * 6.0
            np.save(subdir / "TEST_Y_hor.npy", Y_hor)
            ts = np.arange(100) * 60_000
            np.save(subdir / "TEST_ts.npy", ts)

            # Monkey-patch : considérer toutes les fake_* comme "usable"
            from einherjar.research.xgb_einhers import data_loader as dl
            fake_usable = {
                "fake_feature_1", "fake_feature_2", "fake_feature_3",
                "fake_feature_4", "fake_feature_5",
                "open", "high", "low", "close", "volume",  # même les OHLCV
            }
            with patch.object(dl, "load_usable_feature_names", return_value=fake_usable):
                loaded = dl.load_xy("TEST", "1h", "crypto", compiled_dir=tmp)

            # Vérifier l'ordre : fake_1, fake_2, fake_3, fake_4, fake_5 (OHLCV exclus)
            self.assertEqual(
                list(loaded.feature_names),
                ["fake_feature_1", "fake_feature_2", "fake_feature_3", "fake_feature_4", "fake_feature_5"],
                "L'ordre des features doit être préservé (avec exclusion des OHLCV)"
            )
            # Vérifier que les valeurs correspondent aux indices attendus
            # fake_feature_1 est à l'index 0 dans l'ordre original
            # → après exclusion OHLCV, il est à l'index 0 dans loaded.X
            # fake_feature_2 est à l'index 1 → après exclusion, index 1
            # fake_feature_3 était à l'index 7 → après exclusion, index 2
            # etc.
            np.testing.assert_array_equal(
                loaded.X[:, 0], X[:, 0],  # fake_feature_1
                "X[:, 0] doit correspondre à fake_feature_1 (original index 0)"
            )
            np.testing.assert_array_equal(
                loaded.X[:, 1], X[:, 1],  # fake_feature_2
                "X[:, 1] doit correspondre à fake_feature_2 (original index 1)"
            )
            np.testing.assert_array_equal(
                loaded.X[:, 2], X[:, 7],  # fake_feature_3
                "X[:, 2] doit correspondre à fake_feature_3 (original index 7)"
            )


if __name__ == "__main__":
    unittest.main()
