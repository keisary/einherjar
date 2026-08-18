"""test_model.py - Tests du model.py (dual backend xgboost/sklearn).

P2 : tests standards, pas P0.
"""
import unittest
import numpy as np

from einherjar.research.xgb_einhers.model import (
    GBDTConfig,
    has_xgboost,
    train_gbdt,
    predict_gbdt,
    feature_importance,
)
from einherjar.research.xgb_einhers.path_extractor import (
    extract_paths,
    parse_xgb_dump,
    parse_sklearn_tree,
)


class TestBackendDetection(unittest.TestCase):
    def test_has_xgboost_returns_bool(self):
        self.assertIsInstance(has_xgboost(), bool)


class TestSklearnFallback(unittest.TestCase):
    """P2 : sklearn GBR fonctionne en fallback."""

    def setUp(self):
        rng = np.random.default_rng(42)
        n, f = 1000, 20
        self.X_train = rng.standard_normal((n, f)).astype(np.float32)
        # y = X[:, 0] + X[:, 1] * 0.5 + noise
        self.y_train = (self.X_train[:, 0] + self.X_train[:, 1] * 0.5
                        + rng.standard_normal(n) * 0.1).astype(np.float32)
        self.X_val = rng.standard_normal((200, f)).astype(np.float32)
        self.y_val = (self.X_val[:, 0] + self.X_val[:, 1] * 0.5
                      + rng.standard_normal(200) * 0.1).astype(np.float32)

    def test_train_sklearn(self):
        config = GBDTConfig(backend="sklearn", n_estimators=10, max_depth=3)
        model, backend = train_gbdt(
            self.X_train, self.y_train, self.X_val, self.y_val, config,
        )
        self.assertEqual(backend, "sklearn")
        # Prédictions cohérentes
        preds = predict_gbdt(model, self.X_val, backend)
        self.assertEqual(preds.shape, (200,))
        self.assertTrue(np.isfinite(preds).all())

    def test_extract_paths_sklearn(self):
        config = GBDTConfig(backend="sklearn", n_estimators=10, max_depth=3)
        model, backend = train_gbdt(
            self.X_train, self.y_train, self.X_val, self.y_val, config,
        )
        feature_names = [f"f{i}" for i in range(self.X_train.shape[1])]
        paths = extract_paths(
            model, backend, feature_names,
            min_score=0.001, max_score=1.0, max_paths=20,
        )
        # On doit avoir des chemins
        self.assertGreater(len(paths), 0)
        # Chaque chemin a des conditions
        for p in paths:
            self.assertGreater(len(p.conditions), 0)
            self.assertLessEqual(len(p.conditions), 4)

    def test_feature_importance(self):
        config = GBDTConfig(backend="sklearn", n_estimators=10, max_depth=3)
        model, backend = train_gbdt(
            self.X_train, self.y_train, self.X_val, self.y_val, config,
        )
        feature_names = [f"f{i}" for i in range(self.X_train.shape[1])]
        imp = feature_importance(model, backend, feature_names)
        # Toutes les features ont une importance
        self.assertEqual(len(imp), 20)
        # f0 et f1 doivent être les plus importantes (ce sont elles qui
        # déterminent y)
        top2 = list(imp.keys())[:2]
        self.assertIn("f0", top2)
        self.assertIn("f1", top2)


class TestAutoBackend(unittest.TestCase):
    """P2 : auto-détection du backend."""

    def setUp(self):
        rng = np.random.default_rng(42)
        self.X = rng.standard_normal((200, 5)).astype(np.float32)
        self.y = (self.X[:, 0] + rng.standard_normal(200) * 0.1).astype(np.float32)

    def test_auto_selects_available(self):
        config = GBDTConfig(backend="auto", n_estimators=5, max_depth=2)
        model, backend = train_gbdt(self.X, self.y, self.X, self.y, config)
        # Doit sélectionner xgboost si dispo, sinon sklearn
        expected = "xgboost" if has_xgboost() else "sklearn"
        self.assertEqual(backend, expected)


if __name__ == "__main__":
    unittest.main()
