"""multi_asset_loader.py - Chargement multi-actifs.

Sprint 2.3.2.

Concatene X, Y_dir, Y_ret, Y_hor, ts de N actifs du meme asset_class
et timeframe. Utile pour augmenter la taille d'entrainement
(28 actifs crypto x 70k = ~2M samples au lieu de 70k) et capturer
des invariants cross-actifs.

API :
- list_available_assets(asset_class, timeframe) -> list[str]
- load_multi_asset(assets, asset_class, timeframe) -> MultiAssetData
- align_multi_asset_ohlcv(loaded_list, ohlcv_dir) -> ...
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from einherjar.research.xgb_einhers.data_loader import (
    COMPILED_DIR,
    load_xy,
)
from einherjar.research.xgb_einhers.types import LoadedData

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
    if require_ohlcv and ohlcv_dir is not None:
        filtered = []
        for a in assets:
            if (Path(ohlcv_dir) / asset_class / a / timeframe).is_dir():
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
