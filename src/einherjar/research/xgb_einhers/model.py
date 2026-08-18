"""model.py - Entraînement GBDT supervisé avec double backend (xgboost / sklearn).

Réponse Q10 : 1 modèle par (asset, TF, horizon).
Réponse Q18 : GPU first → device='cuda' sur xgboost, fallback CPU.

Double backend :
- xgboost (préféré) : `XGBRegressor` + `booster.get_dump()` → texte
- sklearn (fallback) : `GradientBoostingRegressor` + arbres sklearn

L'API publique expose `train_gbdt(X_train, y_train, X_val, y_val, config)`
qui retourne un objet unifié. Le path_extractor inspecte le type pour
savoir quel parser utiliser.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Backend detection
# --------------------------------------------------------------------------- #


_HAS_XGBOOST = False
try:
    import xgboost as _xgb
    _HAS_XGBOOST = True
except ImportError:
    _xgb = None
    logger.info("xgboost non disponible, fallback sklearn activé")


def has_xgboost() -> bool:
    return _HAS_XGBOOST


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GBDTConfig:
    """Hyperparamètres GBDT (xgboost ou sklearn)."""
    backend: str = "auto"              # 'auto' | 'xgboost' | 'sklearn'
    n_estimators: int = 100
    max_depth: int = 4
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8      # ignoré par sklearn
    min_child_weight: int = 10          # ignoré par sklearn
    reg_alpha: float = 0.1              # ignoré par sklearn
    reg_lambda: float = 1.0             # ignoré par sklearn
    random_state: int = 42
    tree_method: str = "hist"          # xgboost only
    device: str = "cpu"                # xgboost only
    early_stopping_rounds: int = 10    # xgboost only

    @classmethod
    def regularized(cls) -> "GBDTConfig":
        """Config régularisée pour anti-overfit (Sprint 2.3.3).

        - max_depth=3 (au lieu de 4) : arbres moins profonds
        - min_child_weight=50 (au lieu de 10) : feuilles plus grosses
        - reg_alpha=1.0 (au lieu de 0.1) : L1 plus fort
        - reg_lambda=5.0 (au lieu de 1.0) : L2 plus fort
        - colsample_bytree=0.6 (au lieu de 0.8) : sous-ensemble de features par arbre
        - subsample=0.7 (au lieu de 0.8) : bagging plus agressif
        """
        return cls(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.7,
            colsample_bytree=0.6,
            min_child_weight=50,
            reg_alpha=1.0,
            reg_lambda=5.0,
            random_state=42,
            early_stopping_rounds=20,
        )


# --------------------------------------------------------------------------- #
# Training unifié
# --------------------------------------------------------------------------- #


def train_gbdt(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: GBDTConfig = GBDTConfig(),
) -> tuple[Any, str]:
    """Entraîne un GBDT supervisé (xgboost ou sklearn).

    Returns:
        (model, backend_name) où backend_name ∈ {'xgboost', 'sklearn'}
    """
    backend = _resolve_backend(config.backend)
    if backend == "xgboost":
        return _train_xgb(X_train, y_train, X_val, y_val, config), "xgboost"
    else:
        return _train_sklearn(X_train, y_train, X_val, y_val, config), "sklearn"


def _resolve_backend(requested: str) -> str:
    if requested == "auto":
        return "xgboost" if _HAS_XGBOOST else "sklearn"
    if requested == "xgboost" and not _HAS_XGBOOST:
        logger.warning("xgboost demandé mais non disponible, fallback sklearn")
        return "sklearn"
    return requested


def _train_xgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: GBDTConfig,
):
    """Entraîne un XGBRegressor."""
    params = {
        "n_estimators": config.n_estimators,
        "max_depth": config.max_depth,
        "learning_rate": config.learning_rate,
        "subsample": config.subsample,
        "colsample_bytree": config.colsample_bytree,
        "min_child_weight": config.min_child_weight,
        "reg_alpha": config.reg_alpha,
        "reg_lambda": config.reg_lambda,
        "random_state": config.random_state,
        "tree_method": config.tree_method,
        "eval_metric": "rmse",
        "objective": "reg:squarederror",
    }
    if config.device == "cuda":
        params["device"] = "cuda"
    else:
        params["n_jobs"] = -1
    model = _xgb.XGBRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    logger.info("XGBoost trained : best_iteration=%s", getattr(model, "best_iteration", "N/A"))
    return model


def _train_sklearn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: GBDTConfig,
):
    """Entraîne un sklearn GradientBoostingRegressor (fallback)."""
    from sklearn.ensemble import GradientBoostingRegressor
    model = GradientBoostingRegressor(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        learning_rate=config.learning_rate,
        subsample=config.subsample,
        random_state=config.random_state,
    )
    # sklearn GBR n'a pas d'early_stopping_rounds ; on fit directement
    model.fit(X_train, y_train)
    # Validation score (juste pour le log)
    val_score = model.score(X_val, y_val)
    logger.info("sklearn GBR trained : val R² = %.4f", val_score)
    return model


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #


def predict_gbdt(model: Any, X: np.ndarray, backend: str) -> np.ndarray:
    """Prédit Y_ret sur X (compatible xgboost et sklearn)."""
    return model.predict(X)


# --------------------------------------------------------------------------- #
# Feature importance
# --------------------------------------------------------------------------- #


def feature_importance(
    model: Any,
    backend: str,
    feature_names: list[str],
) -> dict[str, float]:
    """Retourne l'importance des features (gain)."""
    if backend == "xgboost":
        booster = model.get_booster()
        score = booster.get_score(importance_type="gain")
        out = {}
        for i, name in enumerate(feature_names):
            out[name] = float(score.get(f"f{i}", 0.0))
    else:
        # sklearn : attribute feature_importances_
        imp = model.feature_importances_
        out = {name: float(imp[i]) for i, name in enumerate(feature_names)}
    return dict(sorted(out.items(), key=lambda x: x[1], reverse=True))
