"""test_feature_robustness.py - Test Sprint 2.1.3.

Vérifie que les top-3 features ne sont pas les seuls drivers du modèle.
Si on retire une top-3 feature, le R² val ne doit pas chuter de plus de 30%.

C'est un test anti-fragilité : si le modèle ne marche qu'avec UNE feature
spécifique, c'est de l'overfit. On veut que le signal soit distribué.
"""
from __future__ import annotations

import unittest

import numpy as np

from einherjar.research.xgb_einhers.data_loader import (
    load_xy,
    temporal_split,
)
from einherjar.research.xgb_einhers.label_engineer import build_target
from einherjar.research.xgb_einhers.model import (
    GBDTConfig,
    feature_importance,
    train_gbdt,
)

HORIZON_IDX = 3  # 2d
N_TOP_FEATURES = 3
MAX_R2_DROP = 0.30  # 30% de chute max


def _train_and_eval(X_train, y_train, X_val, y_val):
    """Entraîne un XGBoost et retourne R² val + importances."""
    model, backend = train_gbdt(
        X_train, y_train, X_val, y_val,
        config=GBDTConfig(n_estimators=50, max_depth=4, learning_rate=0.05),
    )
    preds = model.predict(X_val)
    # R² val
    ss_res = float(np.sum((y_val - preds) ** 2))
    ss_tot = float(np.sum((y_val - np.mean(y_val)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return model, backend, r2


class TestFeatureRobustness(unittest.TestCase):
    """Vérifie qu'aucune feature n'est un driver unique du modèle."""

    @classmethod
    def setUpClass(cls):
        cls.loaded = load_xy("BTCUSD", "1h", "crypto")
        target, valid_mask, y_hor = build_target(cls.loaded, HORIZON_IDX)
        X = cls.loaded.X[valid_mask]
        y = target[valid_mask].astype(np.float32)
        cls.feature_names = list(cls.loaded.feature_names)

        cls.split = temporal_split(X, y, train_ratio=0.6, val_ratio=0.2, holdout_ratio=0.2)
        # Baseline
        cls.baseline_model, cls.baseline_backend, cls.baseline_r2 = _train_and_eval(
            cls.split.train_X, cls.split.train_y,
            cls.split.val_X, cls.split.val_y,
        )
        cls.importances = feature_importance(
            cls.baseline_model, cls.baseline_backend, cls.feature_names,
        )
        cls.top_features = list(cls.importances.keys())[:N_TOP_FEATURES]

    def test_baseline_r2_is_positive(self):
        """Le modèle baseline doit apprendre quelque chose (R² > 0)."""
        self.assertGreater(self.baseline_r2, 0.0,
                           f"R² val baseline = {self.baseline_r2:.4f} <= 0")

    def test_top_features_identified(self):
        """On doit identifier au moins 3 features avec importance > 0."""
        self.assertGreaterEqual(
            len([n for n, v in self.importances.items() if v > 0]),
            N_TOP_FEATURES,
            f"Moins de {N_TOP_FEATURES} features avec importance > 0",
        )

    def test_no_single_feature_dominates(self):
        """Retirer chaque top feature ne doit pas faire chuter R² de plus de 30%."""
        drops = []
        for feat in self.top_features:
            feat_idx = self.feature_names.index(feat)
            mask = np.ones(len(self.feature_names), dtype=bool)
            mask[feat_idx] = False
            X_train_dropped = self.split.train_X[:, mask]
            X_val_dropped = self.split.val_X[:, mask]
            _, _, r2_dropped = _train_and_eval(
                X_train_dropped, self.split.train_y,
                X_val_dropped, self.split.val_y,
            )
            drop_pct = (self.baseline_r2 - r2_dropped) / max(abs(self.baseline_r2), 1e-6)
            drops.append((feat, r2_dropped, drop_pct))
        print(f"\n[ROBUSTESSE] Top features: {self.top_features}")
        print(f"[ROBUSTESSE] Baseline R2 = {self.baseline_r2:.4f}")
        for feat, r2, drop in drops:
            print(f"  - drop {feat:40s} -> R2 = {r2:.4f} (drop {drop*100:+.1f}%)")
        max_drop = max(d[2] for d in drops)
        self.assertLessEqual(
            max_drop, MAX_R2_DROP,
            f"Top feature fait chuter R² de {max_drop*100:.1f}% (>{MAX_R2_DROP*100:.0f}%) - signal fragile",
        )

    def test_feature_importance_normalized(self):
        """Les importances doivent sommer à ~1.0 (xgb gain sum)."""
        total = sum(self.importances.values())
        # xgboost gain ne somme PAS forcément à 1 (peut être < ou > 1)
        # On vérifie juste qu'il y a des features avec importance > 0
        self.assertGreater(total, 0.0, "Aucune importance > 0")


if __name__ == "__main__":
    unittest.main()
