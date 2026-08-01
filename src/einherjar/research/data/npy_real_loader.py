"""data/npy_real_loader.py — Loader des données réelles (format .npy MIDAS V3).

Charge les fichiers .npy produits par le pipeline MIDAS V3 :
  - *_ts.npy : timestamps Unix ms (int64)
  - *_X.npy   : features normalisées (float32, 246 colonnes)

Les features contiennent l'OHLCV brut (colonnes 0-4) avant normalisation.
On reconstitue un OhlcvFrame valide à partir de ces colonnes.

ATTENTION : pour P0 #3 (CLI sur données réelles), on accepte que les 4
premières colonnes de X sont open/high/low/close. Le volume est pris
depuis la 5e colonne. Si la structure change, ce loader lèvera une erreur
explicite (pas de fallback silencieux).

Format de sortie : OhlcvFrame avec colonnes OHLCV + un FeaturesFrame
construit à partir des features normalisées.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from einherjar.research.data.features import FeaturesFrame
from einherjar.research.data.ohlcv import OhlcvFrame

logger = logging.getLogger(__name__)


# Colonnes OHLCV dans le fichier X.npy (convention MIDAS V3)
_OHLCV_COLUMN_INDICES: dict[str, int] = {
    "open": 0,
    "high": 1,
    "low": 2,
    "close": 3,
    "volume": 4,
}


class NpyRealLoaderError(Exception):
    """Erreur de chargement des données réelles."""


def load_ohlcv_from_npy(
    asset: str,
    asset_class: str,
    timeframe: str,
    data_root: Path | None = None,
) -> OhlcvFrame:
    """Charge la série OHLCV réelle depuis les .npy MIDAS V3.

    Args:
        asset: Symbole (ex: 'BTCUSD').
        asset_class: Classe d'actifs (ex: 'crypto', 'forex', 'indices', etc.).
        timeframe: Timeframe (ex: '1h', '4h', '1d').
        data_root: Racine des données (défaut: midasV3/src/data/compiled).

    Returns:
        OhlcvFrame avec colonnes [timestamp, open, high, low, close, volume].

    Raises:
        NpyRealLoaderError: si les fichiers n'existent pas, sont vides,
            ou ont une structure incompatible (pas de fallback silencieux).
    """
    root = data_root or Path(r"D:/midas_v2/midasV3/src/data/compiled")
    dir_path = root / asset_class / timeframe
    ts_path = dir_path / f"{asset}_ts.npy"
    x_path = dir_path / f"{asset}_X.npy"

    if not dir_path.exists():
        raise NpyRealLoaderError(
            f"Répertoire de données absent : {dir_path}. "
            f"Vérifie --data-root (défaut: {root})."
        )
    if not ts_path.exists() or not x_path.exists():
        raise NpyRealLoaderError(
            f"Fichiers .npy absents pour {asset} × {timeframe} : "
            f"attendu {ts_path.name} et {x_path.name} dans {dir_path}."
        )

    try:
        ts = np.load(ts_path)
        x = np.load(x_path)
    except Exception as exc:
        raise NpyRealLoaderError(
            f"Erreur de lecture des .npy pour {asset} × {timeframe} : {exc}"
        ) from exc

    if ts.size == 0 or x.size == 0:
        raise NpyRealLoaderError(
            f"Données vides pour {asset} × {timeframe} : ts.shape={ts.shape}, x.shape={x.shape}"
        )
    if x.ndim != 2 or x.shape[0] != ts.shape[0]:
        raise NpyRealLoaderError(
            f"Incohérence de shape pour {asset} × {timeframe} : "
            f"ts.shape={ts.shape}, x.shape={x.shape}"
        )
    if x.shape[1] < 5:
        raise NpyRealLoaderError(
            f"Pas assez de colonnes pour extraire OHLCV : x.shape={x.shape} (besoin >= 5)"
        )

    # Reconstruction OHLCV depuis les 5 premières colonnes.
    timestamps = [_ts_to_datetime(t) for t in ts]
    df = pl.DataFrame({
        "asset": [asset] * len(ts),
        "timeframe": [timeframe] * len(ts),
        "timestamp": timestamps,
        "open": x[:, _OHLCV_COLUMN_INDICES["open"]].astype("float64"),
        "high": x[:, _OHLCV_COLUMN_INDICES["high"]].astype("float64"),
        "low": x[:, _OHLCV_COLUMN_INDICES["low"]].astype("float64"),
        "close": x[:, _OHLCV_COLUMN_INDICES["close"]].astype("float64"),
        "volume": x[:, _OHLCV_COLUMN_INDICES["volume"]].astype("float64"),
    })
    # Validation minimale : OHLC cohérent (low <= high, etc.).
    invalid = df.filter(
        (pl.col("low") > pl.col("high"))
        | (pl.col("open") < pl.col("low"))
        | (pl.col("open") > pl.col("high"))
        | (pl.col("close") < pl.col("low"))
        | (pl.col("close") > pl.col("high"))
    )
    if invalid.height > 0:
        logger.warning(
            "%d bougies avec OHLC incoherent (low>high, etc.) pour %s %s — ignorees",
            invalid.height, asset, timeframe,
        )
        df = df.filter(
            (pl.col("low") <= pl.col("high"))
            & (pl.col("open") >= pl.col("low"))
            & (pl.col("open") <= pl.col("high"))
            & (pl.col("close") >= pl.col("low"))
            & (pl.col("close") <= pl.col("high"))
        )
    if df.is_empty():
        raise NpyRealLoaderError(
            f"Toutes les bougies OHLCV sont invalides pour {asset} × {timeframe}"
        )

    logger.info(
        "OHLCV réel chargé : %s × %s, %d bougies [%s → %s]",
        asset, timeframe, df.height, df["timestamp"][0], df["timestamp"][-1],
    )
    return OhlcvFrame(
        asset=asset,
        timeframe=timeframe,
        df=df,
        data_version=f"npy:{asset_class}/{timeframe}/{asset}",
    )


def load_features_from_npy(
    asset: str,
    asset_class: str,
    timeframe: str,
    config: Any,
    data_root: Path | None = None,
) -> FeaturesFrame:
    """Charge les features réelles depuis les .npy MIDAS V3.

    On garde les 218 features utilisables (filtre via la taxonomie chargée).
    Les features fantômes/meta-factors/alias sont exclues.
    """
    root = data_root or Path(r"D:/midas_v2/midasV3/src/data/compiled")
    dir_path = root / asset_class / timeframe
    ts_path = dir_path / f"{asset}_ts.npy"
    x_path = dir_path / f"{asset}_X.npy"

    if not ts_path.exists() or not x_path.exists():
        raise NpyRealLoaderError(
            f"Fichiers .npy absents pour {asset} × {timeframe}"
        )

    ts = np.load(ts_path)
    x = np.load(x_path)
    if x.ndim != 2 or x.shape[0] != ts.shape[0]:
        raise NpyRealLoaderError(
            f"Incohérence de shape pour {asset} × {timeframe} : "
            f"ts={ts.shape}, x={x.shape}"
        )

    # Charge la taxonomie pour mapper les colonnes aux noms.
    taxonomy = config.features_taxonomy.get("features", {})
    if not taxonomy:
        raise NpyRealLoaderError("Taxonomie features vide — charge la config d'abord")

    # On suppose que les colonnes de X sont dans le même ordre que la taxonomie.
    # Si ce n'est pas le cas, on ne peut pas mapper (fail explicite).
    feature_names_in_order = list(taxonomy.keys())
    n_taxo = len(feature_names_in_order)
    n_x = x.shape[1]
    if n_x != n_taxo:
        raise NpyRealLoaderError(
            f"Nombre de features incoherent : .npy={n_x}, taxonomie={n_taxo}. "
            f"Vérifier que features_taxonomy_corrected.json correspond bien aux .npy."
        )

    # Construit le dict {col_name: array}
    feature_dict: dict[str, pl.Series] = {}
    for i, name in enumerate(feature_names_in_order):
        feature_dict[name] = pl.Series(name, x[:, i].astype("float64"))

    # Ajoute les colonnes OHLCV+timestamp.
    timestamps = [_ts_to_datetime(t) for t in ts]
    feature_dict["timestamp"] = pl.Series("timestamp", timestamps)
    feature_dict["open"] = pl.Series("open", x[:, 0].astype("float64"))
    feature_dict["high"] = pl.Series("high", x[:, 1].astype("float64"))
    feature_dict["low"] = pl.Series("low", x[:, 2].astype("float64"))
    feature_dict["close"] = pl.Series("close", x[:, 3].astype("float64"))
    feature_dict["volume"] = pl.Series("volume", x[:, 4].astype("float64"))

    df = pl.DataFrame(feature_dict)
    usable = config.usable_set()
    usable_names = tuple(c for c in df.columns if c in usable)
    return FeaturesFrame(
        asset=asset,
        timeframe=timeframe,
        df=df,
        feature_names=usable_names,
        data_version=f"npy:{asset_class}/{timeframe}/{asset}",
    )


def _ts_to_datetime(ts_ms: int) -> datetime:
    """Convertit un timestamp Unix ms en datetime UTC."""
    return datetime.fromtimestamp(int(ts_ms) / 1000, tz=timezone.utc)
