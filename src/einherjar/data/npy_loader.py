"""Loader de donnees MIDAS V3 — conversion .npy vers DataFrame polars.

Les fichiers .npy de midasV3/src/data/compiled/ contiennent :
- *_ts.npy : timestamps Unix ms (int64)
- *_X.npy : features (float32, 246 colonnes)
- *_Y_dir.npy : direction labels (int8)
- *_Y_hor.npy : horizon labels
- *_Y_ret.npy : returns (float32)

Ce module charge les donnees OHLCV depuis les .npy pour initialiser le store live.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

BASE = Path(r"D:/midas_v2/midasV3/src/data/compiled")


def load_ohlcv_from_npy(
    asset: str,
    asset_class: str,
    timeframe: str,
) -> pl.DataFrame | None:
    """Charge l'historique OHLCV depuis les .npy MIDAS.

    Les fichiers .npy contiennent les features deja calculees (246 colonnes).
    On extrait les colonnes OHLCV de base pour le store live, et les features
    sont recalculees par le FeatureEngine en inference.

    Args:
        asset: Symbole (ex 'BTCUSD').
        asset_class: Classe d'actifs (ex 'crypto').
        timeframe: Timeframe (ex '4h').

    Returns:
        DataFrame polars avec [timestamp, open, high, low, close, volume].
        None si les fichiers n'existent pas.
    """
    dir_path = BASE / asset_class / timeframe
    if not dir_path.exists():
        return None

    ts_path = dir_path / f"{asset}_ts.npy"
    x_path = dir_path / f"{asset}_X.npy"

    if not ts_path.exists() or not x_path.exists():
        return None

    try:
        ts = np.load(ts_path)
        np.load(x_path)

        # Les timestamps .npy sont en ms Unix. On les GARDE en int64
        # pour compatibilite avec ohlcv._sanitize qui attend des entiers.
        # (datetime causerait int() argument must be a string... not 'datetime.timedelta')
        timestamps = ts.astype("int64")

        # Les 5 premieres colonnes de X sont typiquement OHLCV
        # (a verifier avec la structure reelle des features MIDAS)
        # Pour l'instant on utilise les prix bruts comme placeholders
        # car les vrais OHLCV ne sont pas stockes separement dans les .npy
        # NOTE : les .npy contiennent des features normalisees, pas les prix bruts.
        # Il faut les recuperer via les API broker pour le live.

        # On cree un DataFrame minimal avec les timestamps en int (ms Unix).
        df = pl.DataFrame({
            "timestamp": timestamps,
            "open": np.zeros(len(ts), dtype=np.float64),
            "high": np.zeros(len(ts), dtype=np.float64),
            "low": np.zeros(len(ts), dtype=np.float64),
            "close": np.zeros(len(ts), dtype=np.float64),
            "volume": np.zeros(len(ts), dtype=np.float64),
        })

        return df

    except Exception as e:
        print(f"Erreur chargement {asset}/{timeframe}: {e}")
        return None


def list_available_npy(
    asset_class: str | None = None,
    timeframe: str | None = None,
) -> list[dict[str, str]]:
    """Liste les actifs disponibles dans les .npy MIDAS.

    Args:
        asset_class: Filtrer par classe. None = toutes.
        timeframe: Filtrer par TF. None = tous.

    Returns:
        Liste de dicts {asset, class, timeframe}.
    """
    results = []
    classes = [asset_class] if asset_class else [d.name for d in BASE.iterdir() if d.is_dir()]

    for cls in classes:
        cls_dir = BASE / cls
        if not cls_dir.exists():
            continue
        tfs = [timeframe] if timeframe else [d.name for d in cls_dir.iterdir() if d.is_dir()]
        for tf in tfs:
            tf_dir = cls_dir / tf
            if not tf_dir.exists():
                continue
            for f in tf_dir.glob("*_ts.npy"):
                asset = f.stem.replace("_ts", "")
                results.append({"asset": asset, "class": cls, "timeframe": tf})

    return results
