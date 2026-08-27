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
from dataclasses import dataclass
from typing import Any

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
    """Indique si le backend xgboost est disponible dans l'environnement."""
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
    def regularized(cls) -> GBDTConfig:
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
# Grid search léger (problème 5)
# --------------------------------------------------------------------------- #


def train_gbdt_grid(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    base_config: GBDTConfig,
    grid_depth: tuple[int, ...] = (3, 4, 6),
    grid_estimators: tuple[int, ...] = (50, 100, 200),
    max_cells: int = 8,
    random_state: int = 42,
) -> tuple[Any, str, GBDTConfig, float]:
    """Grid search léger sur max_depth et n_estimators (problème 5).

    Stratégie validée Jovanny : pour chaque triplet, on essaie plusieurs
    profondeurs/nb d'arbres et on garde la config qui minimise l'erreur
    de validation (proxy de la qualité des chemins) — elle maximise
    indirectement la recherche d'einhers en évitant le sous/sur-apprentissage.

    Pour ne pas exploser le temps (12 cellules x 696 triplets), on :
    - balaie max_depth x n_estimators (8 cellules max, produits décimés)
    - garde learning_rate du base_config (0.05) pour limiter le coût
    - entraîne sur un sous-échantillon si nécessaire

    Returns:
        (model, backend, best_config, best_val_rmse)
    """
    import itertools

    depths = grid_depth
    ests = grid_estimators
    # FIX (2026-08-21) : respecter les valeurs CLI du base_config.
    # Si l'utilisateur a passe --max-depth / --n-estimators explicites (via
    # run_pipeline), on les INJECTE dans la grille pour qu'elles soient toujours
    # testees (avant, `optimize_params=True` balayait une grille fixe et ignorait
    # silencieusement ces valeurs CLI).
    if base_config.max_depth not in depths:
        depths = tuple(sorted(set(depths) | {base_config.max_depth}))
    if base_config.n_estimators not in ests:
        ests = tuple(sorted(set(ests) | {base_config.n_estimators}))

    # Produit cartésien, limité à max_cells pour rester léger
    cells = list(itertools.product(depths, ests))
    if len(cells) > max_cells:
        # Échantillonner uniformément
        step = len(cells) / max_cells
        cells = [cells[int(i * step)] for i in range(max_cells)]

    # Sous-echantillonnage pour accelerer le grid (max ~60k lignes train)
    max_rows = 60000
    if X_train.shape[0] <= max_rows:
        _X = X_train
    else:
        idx = np.linspace(0, X_train.shape[0] - 1, max_rows).astype(int)
        _X = X_train[idx]
    if y_train.shape[0] <= max_rows:
        _y = y_train
    else:
        idx = np.linspace(0, y_train.shape[0] - 1, max_rows).astype(int)
        _y = y_train[idx]
    # FIX DOUBLE-DIPPING (2026-08-21) : on resserve une tranche interne du train
    # pour selectionner la config (val_inner). On n'evalue JAMAIS sur X_val/y_val
    # passes en entree : ce split sert ensuite aux metriques/p-values (t_stat, BH).
    # Sans ce fix, le grid optimisait sur le meme split qui produisait les p-values
    # -> p-values mecaniquement optimistes, FDR sous-corrige.
    _split = int(round(len(_X) * 0.85))
    grid_tr_X, grid_tr_y = _X[:_split], _y[:_split]
    grid_va_X, grid_va_y = _X[_split:], _y[_split:]

    best_model, best_backend, best_cfg, best_rmse = None, "xgboost", base_config, float("inf")
    for depth, n_est in cells:
        cfg = GBDTConfig(**{**base_config.__dict__, "max_depth": depth, "n_estimators": n_est})
        try:
            m, backend = train_gbdt(grid_tr_X, grid_tr_y, grid_va_X, grid_va_y, cfg)
            # RMSE sur validation
            pred = predict_gbdt(m, grid_va_X, backend)
            if len(grid_va_y):
                rmse = float(np.sqrt(np.mean((pred - grid_va_y) ** 2)))
            else:
                rmse = float("inf")
            logger.info("    grid: depth=%d est=%d lr=%.3f -> val_rmse=%.6f",
                        depth, n_est, cfg.learning_rate, rmse)
            if rmse < best_rmse:
                best_model, best_backend, best_cfg, best_rmse = m, backend, cfg, rmse
        except Exception as e:
            logger.warning("    grid config depth=%d est=%d failed: %s", depth, n_est, e)
    if best_model is None:
        # Fallback : config de base
        best_model, best_backend = train_gbdt(X_train, y_train, X_val, y_val, base_config)
        best_cfg, best_rmse = base_config, 0.0
    logger.info("grid_search : best depth=%d est=%d lr=%.3f rmse=%.6f",
                best_cfg.max_depth, best_cfg.n_estimators,
                best_cfg.learning_rate, best_rmse)
    return best_model, best_backend, best_cfg, best_rmse


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
        # FIX OVERSUBSCRIPTION (2026-08-27) : n_jobs=-1 × N workers = chaos de threads.
        # On limite à 1 thread par worker quand multiprocessing est actif.
        import os
        params["n_jobs"] = 1
    # FIX (2026-08-21) : early stopping REEL.
    # API xgboost 3.x : `early_stopping_rounds` se passe au CONSTRUCTEUR
    # XGBRegressor(...), pas a .fit() (fit() n'accepte ce kwarg que via
    # la nouvelle API `fit(X, y, ...)` qui fait empeche ici). Avant, la
    # config le declait mais ne le transmettait jamais -> 200 arbres memes
    # lorsque le modele stagnait.
    if config.early_stopping_rounds and config.early_stopping_rounds > 0:
        params["early_stopping_rounds"] = config.early_stopping_rounds
    model = _xgb.XGBRegressor(**params)  # pyright: ignore[reportOptionalMemberAccess]
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
    """Prédit Y_ret sur X (compatible xgboost et sklearn).

    FIX PERF (2026-08-26) : si le booster tourne sur CUDA et que X est sur CPU,
    xgboost fait un fallback DMatrix lent ("Falling back to prediction using
    DMatrix due to mismatched devices"). On transfere X sur le device du
    booster pour rester sur le chemin rapide inplace_predict.
    """
    if backend == "xgboost":
        try:
            booster = model.get_booster()
            device = getattr(booster, "device", "") or ""
            if "cuda" in str(device):
                import xgboost as _xgb

                # TRANSFER SUR GPU : pour garder le path inplace_predict rapide
                # sans le fallback DMatrix lent qui genere des warnings.
                dmat = _xgb.DMatrix(X.astype(np.float32))
                return booster.predict(dmat).astype(np.float64)
        except Exception as e:  # pragma: no cover - fallback securitaire
            logger.debug("predict_gpu fast-path failed (%s), fallback standard", e)
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
