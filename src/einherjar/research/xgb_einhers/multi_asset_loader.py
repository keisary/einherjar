"""multi_asset_loader.py - Chargement multi-actifs.

Sprint 2.3.2 + Sprint 3.4 (FIX BUG-03).

Concatene X, Y_dir, Y_ret, Y_hor, ts de N actifs du meme asset_class
et timeframe. Utile pour augmenter la taille d'entrainement
(28 actifs crypto x 70k = ~2M samples au lieu de 70k) et capturer
des invariants cross-actifs.

API :
- list_available_assets(asset_class, timeframe) -> list[str]
- load_multi_asset(assets, asset_class, timeframe) -> MultiAssetData
- load_multi_asset_split(...) -> MultiAssetSplit (Sprint 3.4, FIX BUG-03)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .data_loader import (
    COMPILED_DIR,
    load_xy,
    temporal_split,
)
from .types import LoadedData

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MultiAssetData:
    """Donnees concatenees de N actifs.

    Les arrays sont concatenes sur l'axe 0, avec un tag asset_idx
    pour identifier l'actif d'origine de chaque ligne.
    """
    X: np.ndarray                       # (N_total, F) float32
    Y_dir: np.ndarray                   # (N_total, H) int8
    Y_ret: np.ndarray                   # (N_total, H) float32
    Y_hor: np.ndarray                   # (N_total, H) float32
    asset_idx: np.ndarray               # (N_total,) int : index de l'actif (0..N-1)
    feature_names: tuple[str, ...]
    horizons: tuple[str, ...]
    assets: tuple[str, ...]
    timestamps: np.ndarray              # (N_total,) int64


@dataclass(frozen=True)
class MultiAssetSplit:
    """Sprint 3.4 FIX BUG-03 : split temporel par actif PUIS concat.

    Chaque actif est splitte temporellement en (train, val, holdout)
    AVANT la concatenation. L'invariant causal est garanti :
    max(timestamp(train_global)) < min(timestamp(val_global)) < min(timestamp(holdout_global))
    """
    train_X: np.ndarray
    train_y: np.ndarray                  # target pour l'horizon demande
    val_X: np.ndarray
    val_y: np.ndarray
    holdout_X: np.ndarray
    holdout_y: np.ndarray
    feature_names: tuple[str, ...]
    horizons: tuple[str, ...]
    assets: tuple[str, ...]
    n_train: int
    n_val: int
    n_holdout: int
    horizon_idx: int


# Mapping des classes qui pointent vers un dossier OHLCV different.
# Sprint 3.6 : les 3 sous-classes stocks (growth/tech/value) partagent
# le meme dossier OHLCV "stocks/".
STOCKS_OHLCV_ALIAS = {
    "stocks_growth": "stocks",
    "stocks_tech": "stocks",
    "stocks_value": "stocks",
}


def resolve_ohlcv_class(asset_class: str) -> str:
    """Retourne le nom de classe reellement utilise pour OHLCV.

    Les 3 sous-classes stocks pointent toutes vers le meme dossier
    OHLCV `stocks/`.
    """
    return STOCKS_OHLCV_ALIAS.get(asset_class, asset_class)


def list_available_assets(
    asset_class: str = "crypto",
    timeframe: str = "1h",
    compiled_dir: Path = COMPILED_DIR,
    require_ohlcv: bool = False,
    ohlcv_dir: Optional[Path] = None,
) -> list[str]:
    """Liste les actifs disponibles pour un (asset_class, timeframe).

    Args:
        require_ohlcv : si True, ne retourne que les actifs qui ont
                        aussi un dossier OHLCV brut.
    """
    base = Path(compiled_dir) / asset_class / timeframe
    if not base.exists():
        return []
    assets = set()
    for f in base.glob("*_X.npy"):
        asset = f.name.replace("_X.npy", "")
        if (base / f"{asset}_Y_dir.npy").exists() and (base / f"{asset}_Y_ret.npy").exists():
            assets.add(asset)
    assets = sorted(assets)
    # FIX BUG-11 (Sprint 3.6) : require_ohlcv=True DOIT filter, sinon
    # les classes stocks (qui ont beaucoup d'actifs sans OHLCV)
    # plantent sur le 1er actif alphabetique.
    if require_ohlcv:
        # Si ohlcv_dir n'est pas passe, on prend le defaut connu
        if ohlcv_dir is None:
            from .data_loader import OHLCV_DIR
            ohlcv_dir = OHLCV_DIR
        # Sprint 3.6 FIX BUG-12 : les sous-classes stocks (growth/tech/value)
        # partagent le meme dossier OHLCV "stocks/". On doit resoudre la
        # vraie classe OHLCV avant de tester l'existence du dossier.
        ohlcv_class = resolve_ohlcv_class(asset_class)
        filtered = []
        for a in assets:
            if (Path(ohlcv_dir) / ohlcv_class / a / timeframe).is_dir():
                filtered.append(a)
        return filtered
    return assets


def load_multi_asset(
    assets: list[str],
    asset_class: str = "crypto",
    timeframe: str = "1h",
    compiled_dir: Path = COMPILED_DIR,
) -> MultiAssetData:
    """Charge et concatene N actifs.

    ⚠️ ATTENTION : cette fonction CONCATENE puis laisse l'appelant faire
    le split. Cela peut creer un LOOK-AHEAD en multi-actif si le split
    est fait par index sur la globale. Preferez `load_multi_asset_split()`
    (Sprint 3.4 FIX BUG-03).

    Hypothese : tous les actifs ont les memes horizons et feature_names
    (ce qui est vrai pour MIDAS V3 compile).

    Returns:
        MultiAssetData
    """
    if not assets:
        raise ValueError("Au moins 1 actif requis")
    loaded_list: list[LoadedData] = []
    for asset in assets:
        try:
            d = load_xy(asset, timeframe, asset_class, compiled_dir=compiled_dir)
            loaded_list.append(d)
        except FileNotFoundError as e:
            logger.warning("Asset %s absent : %s", asset, e)
            continue
    if not loaded_list:
        raise FileNotFoundError(f"Aucun actif chargeable parmi {assets}")
    # Verifier la coherence des feature_names
    ref_names = loaded_list[0].feature_names
    ref_horizons = loaded_list[0].horizons
    for d in loaded_list[1:]:
        if d.feature_names != ref_names:
            raise ValueError(
                f"Feature names incoherents entre {loaded_list[0].asset} et {d.asset}",
            )
        if d.horizons != ref_horizons:
            raise ValueError(
                f"Horizons incoherents entre {loaded_list[0].asset} et {d.asset}",
            )
    # Concatenation
    X = np.concatenate([d.X for d in loaded_list], axis=0)
    Y_dir = np.concatenate([d.Y_dir for d in loaded_list], axis=0)
    Y_ret = np.concatenate([d.Y_ret for d in loaded_list], axis=0)
    Y_hor = np.concatenate([d.Y_hor for d in loaded_list], axis=0)
    ts = np.concatenate([d.timestamps for d in loaded_list], axis=0)
    asset_idx = np.concatenate([
        np.full(d.X.shape[0], i, dtype=np.int32)
        for i, d in enumerate(loaded_list)
    ])
    logger.info(
        "load_multi_asset : %d actifs, %d samples, %d features",
        len(loaded_list), X.shape[0], X.shape[1],
    )
    return MultiAssetData(
        X=X,
        Y_dir=Y_dir,
        Y_ret=Y_ret,
        Y_hor=Y_hor,
        asset_idx=asset_idx,
        feature_names=ref_names,
        horizons=ref_horizons,
        assets=tuple(d.asset for d in loaded_list),
        timestamps=ts,
    )


def load_multi_asset_split(
    assets: list[str],
    horizon_idx: int,
    asset_class: str = "crypto",
    timeframe: str = "1h",
    compiled_dir: Path = COMPILED_DIR,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    holdout_ratio: float = 0.2,
    embargo_bars: int = 50,
) -> MultiAssetSplit:
    """Sprint 3.4 FIX BUG-03 : split par actif PUIS concat des splits separes.

    Pour chaque actif :
    1. Charger X, Y, Y_dir, Y_hor
    2. Construire valid_mask (Y_dir[:, horizon_idx] != -100)
    3. Filtrer X, target
    4. Split temporel (train/val/holdout) AVEC embargo
    5. Stocker les indices

    PUIS :
    6. Concat les X_train de tous les actifs
    7. Concat les X_val de tous les actifs
    8. Concat les X_holdout de tous les actifs

    Invariant : max(timestamp(train_global)) <= min(timestamp(val_global))
    <= min(timestamp(holdout_global)) - pour chaque actif individuellement.

    Returns:
        MultiAssetSplit avec X et y pour chaque split.
    """
    if not assets:
        raise ValueError("Au moins 1 actif requis")

    # 1. Charger chaque actif
    loaded_list: list[LoadedData] = []
    for asset in assets:
        try:
            d = load_xy(asset, timeframe, asset_class, compiled_dir=compiled_dir)
            loaded_list.append(d)
        except FileNotFoundError as e:
            logger.warning("Asset %s absent : %s", asset, e)
            continue
    if not loaded_list:
        raise FileNotFoundError(f"Aucun actif chargeable parmi {assets}")
    # Verifier coherence
    ref_names = loaded_list[0].feature_names
    ref_horizons = loaded_list[0].horizons
    for d in loaded_list[1:]:
        if d.feature_names != ref_names:
            raise ValueError(f"Feature names incoherents : {loaded_list[0].asset} vs {d.asset}")
        if d.horizons != ref_horizons:
            raise ValueError(f"Horizons incoherents : {loaded_list[0].asset} vs {d.asset}")

    # 2. Pour chaque actif : split temporel INDIVIDUEL
    # On split en X_global (=X sans filtrer valid), y, valid_mask.
    # Puis on applique valid_mask AVANT temporal_split pour eviter leakage.
    horizon_bars_est = 48  # pour 2d, peut etre ajuste
    train_Xs, train_ys, val_Xs, val_ys, holdout_Xs, holdout_ys = [], [], [], [], [], []

    for d in loaded_list:
        # valid_mask sur l'horizon demande
        valid_mask = d.Y_dir[:, horizon_idx] != -100
        X_valid = d.X[valid_mask]
        y_valid = d.Y_ret[valid_mask, horizon_idx].astype(np.float32)
        if X_valid.shape[0] < 100:
            logger.warning(f"Asset {d.asset} trop petit ({X_valid.shape[0]}), skip")
            continue
        # Split temporel AVEC embargo
        split = temporal_split(
            X_valid, y_valid,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            holdout_ratio=holdout_ratio,
            embargo_bars=embargo_bars,
            horizon_bars=horizon_bars_est,
        )
        train_Xs.append(split.train_X)
        train_ys.append(split.train_y)
        val_Xs.append(split.val_X)
        val_ys.append(split.val_y)
        holdout_Xs.append(split.holdout_X)
        holdout_ys.append(split.holdout_y)

    if not train_Xs:
        raise RuntimeError("Aucun actif n'a pu etre splitte")

    # 3. Concat des splits separes
    train_X = np.concatenate(train_Xs, axis=0)
    train_y = np.concatenate(train_ys, axis=0)
    val_X = np.concatenate(val_Xs, axis=0)
    val_y = np.concatenate(val_ys, axis=0)
    holdout_X = np.concatenate(holdout_Xs, axis=0)
    holdout_y = np.concatenate(holdout_ys, axis=0)

    n_train = train_X.shape[0]
    n_val = val_X.shape[0]
    n_holdout = holdout_X.shape[0]
    logger.info(
        "load_multi_asset_split : %d actifs, train=%d, val=%d, holdout=%d (FIX BUG-03)",
        len(train_Xs), n_train, n_val, n_holdout,
    )
    return MultiAssetSplit(
        train_X=train_X, train_y=train_y,
        val_X=val_X, val_y=val_y,
        holdout_X=holdout_X, holdout_y=holdout_y,
        feature_names=ref_names,
        horizons=ref_horizons,
        assets=tuple(d.asset for d in loaded_list),
        n_train=n_train, n_val=n_val, n_holdout=n_holdout,
        horizon_idx=horizon_idx,
    )
