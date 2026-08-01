"""data/npy_real_loader.py — Loader des données réelles (format .npy MIDAS V3).

Charge les fichiers .npy produits par le pipeline MIDAS V3 :
  - *_ts.npy        : timestamps Unix ms (int64)
  - *_X.npy         : features (float32, 246 colonnes)
  - metadata.json   : noms de features dans l'ordre EXACT des colonnes du .npy
  - *_Y_dir.npy     : labels direction (int8)
  - *_Y_hor.npy     : labels horizon (float32)
  - *_Y_ret.npy     : labels return (float32)

ATTENTION — ordre des colonnes :
  - Le fichier `metadata.json` (côte-à-côte avec le .npy) est la SOURCE
    DE VÉRITÉ pour l'ordre des colonnes.
  - La taxonomie `config/features_taxonomy.json` a un ORDRE DIFFÉRENT
    (et est utilisée uniquement pour les métadonnées de chaque feature :
    type, famille économique, exclusion). Toute correspondance entre nom
    de feature et colonne du .npy DOIT passer par metadata.json.
  - Bug historique : les versions précédentes du code utilisaient la
    taxonomie comme ordre de colonnes, ce qui décalait tout d'un cran
    après `macd_signal` (position 9). Ce loader corrige ce bug.

Normalisation (cf. _normalize dans compile_dataset.py) :
  - Colonnes 0-3 (open/high/low/close) : log-returns (log(price[t]) - log(price[t-1]))
  - Colonne 4 (volume)                 : log1p(volume)
  - Colonnes 5+                        : features BRUTES (non normalisées)

Les log-returns ne sont pas des prix exploitables directement par le
moteur pour SL/TP. Le moteur travaille donc dans l'espace des rendements
(volatility-based SL/TP via ATR sur les log-returns).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from einherjar.research.data.features import FeaturesFrame
from einherjar.research.data.ohlcv import OhlcvFrame

logger = logging.getLogger(__name__)

# Colonnes OHLCV dans le fichier X.npy (convention MIDAS V3, fixée par metadata.json).
_OHLCV_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")


class NpyRealLoaderError(Exception):
    """Erreur de chargement des données réelles."""


def load_metadata(data_root: Path, asset_class: str, timeframe: str) -> dict[str, Any]:
    """Charge metadata.json pour une (asset_class, timeframe).

    Returns:
        Dict avec 'feature_names' (liste des 246 noms dans l'ordre des colonnes
        du .npy), 'horizons' (liste des horizons), 'sequence_lengths', etc.

    Raises:
        NpyRealLoaderError: si le fichier n'existe pas ou est invalide.
    """
    meta_path = data_root / asset_class / timeframe / "metadata.json"
    if not meta_path.exists():
        raise NpyRealLoaderError(
            f"metadata.json absent : {meta_path}. "
            f"Vérifie --data-root (défaut: {data_root})."
        )
    try:
        with meta_path.open(encoding="utf-8") as fp:
            meta = json.load(fp)
    except json.JSONDecodeError as exc:
        raise NpyRealLoaderError(f"metadata.json invalide ({meta_path}): {exc}") from exc

    if "feature_names" not in meta:
        raise NpyRealLoaderError(
            f"metadata.json ne contient pas 'feature_names' ({meta_path})"
        )
    return meta


def load_ohlcv_from_npy(
    asset: str,
    asset_class: str,
    timeframe: str,
    data_root: Path | None = None,
) -> tuple[OhlcvFrame, np.ndarray]:
    """Charge la série OHLCV réelle depuis les .npy MIDAS V3.

    ATTENTION : les 5 premières colonnes sont des LOG-RETURNS, pas des
    prix bruts. Le moteur travaille donc dans l'espace des rendements.

    Returns:
        Tuple (OhlcvFrame, validity_mask) :
            - OhlcvFrame avec colonnes [timestamp, open, high, low, close, volume]
              (les valeurs OHLCV sont des log-returns pour les 4 premières,
              log1p pour le volume).
            - validity_mask (np.ndarray de bool) : True pour les bougies
              conservées, False pour celles droppées. Le caller peut
              appliquer ce mask aux features pour garantir l'alignement
              OHLCV/Features.

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

    # Vérifie la cohérence avec metadata.json (5 premières colonnes = OHLCV).
    meta = load_metadata(root, asset_class, timeframe)
    meta_names = meta["feature_names"]
    if len(meta_names) != x.shape[1]:
        raise NpyRealLoaderError(
            f"Coherence OHLCV impossible : metadata.json annonce "
            f"{len(meta_names)} features, .npy a {x.shape[1]} colonnes"
        )
    for i, name in enumerate(_OHLCV_COLUMNS):
        if meta_names[i] != name:
            raise NpyRealLoaderError(
                f"Coherence OHLCV cassee : metadata.json colonne {i} = {meta_names[i]!r}, "
                f"attendu {name!r}. Le dataset est-il compile avec la bonne version ?"
            )

    # Construction OHLCV.
    timestamps = [_ts_to_datetime(t) for t in ts]
    df = pl.DataFrame({
        "asset": [asset] * len(ts),
        "timeframe": [timeframe] * len(ts),
        "timestamp": timestamps,
        "open": x[:, 0].astype("float64"),
        "high": x[:, 1].astype("float64"),
        "low": x[:, 2].astype("float64"),
        "close": x[:, 3].astype("float64"),
        "volume": x[:, 4].astype("float64"),
    })
    mask = compute_validity_mask(df)
    df_sanitized = df.filter(pl.Series("m", mask))
    if df_sanitized.is_empty():
        raise NpyRealLoaderError(
            f"Toutes les bougies OHLCV sont invalides pour {asset} × {timeframe}"
        )

    logger.info(
        "OHLCV reel charge : %s × %s, %d/%d bougies valides [%s → %s] (log-returns)",
        asset, timeframe, df_sanitized.height, df.height,
        df_sanitized["timestamp"][0], df_sanitized["timestamp"][-1],
    )
    return (
        OhlcvFrame(
            asset=asset,
            timeframe=timeframe,
            df=df_sanitized,
            data_version=f"npy:{asset_class}/{timeframe}/{asset}",
        ),
        mask,
    )


def load_features_from_npy(
    asset: str,
    asset_class: str,
    timeframe: str,
    config: Any,
    data_root: Path | None = None,
    *,
    validity_mask: np.ndarray | None = None,
) -> FeaturesFrame:
    """Charge les features réelles depuis les .npy MIDAS V3.

    SOURCE DE VÉRITÉ :
      - **metadata.json** donne l'ordre EXACT des colonnes du .npy (pour
        l'indexation correcte des arrays).
      - **features_taxonomy.json** (config) donne la liste des 218 features
        à GARDER. Les 28 features exclues (19 fantômes + 8 meta-factors
        + 1 alias) sont filtrées ici.

    Args:
        validity_mask: Masque optionnel (np.ndarray de bool) indiquant
            quelles bougies sont valides (alignement avec l'OHLCV sanitisé).
            Si fourni, on filtre les features avec ce masque.
            Si None, on garde toutes les bougies brutes.

    Returns:
        FeaturesFrame avec feature_names = intersection (taxonomie × metadata).
        Les colonnes absentes du .npy lèvent une erreur explicite.
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

    meta = load_metadata(root, asset_class, timeframe)
    meta_names = meta["feature_names"]
    if len(meta_names) != x.shape[1]:
        raise NpyRealLoaderError(
            f"Coherence impossible : metadata.json={len(meta_names)} features, "
            f".npy={x.shape[1]} colonnes"
        )

    # Applique le validity_mask si fourni (alignement avec OHLCV sanitisé).
    if validity_mask is not None:
        if len(validity_mask) != len(ts):
            raise NpyRealLoaderError(
                f"validity_mask length mismatch : mask={len(validity_mask)}, ts={len(ts)}"
            )
        ts = ts[validity_mask]
        x = x[validity_mask]

    # Construit un index : nom de feature -> index de colonne dans le .npy
    name_to_idx: dict[str, int] = {n: i for i, n in enumerate(meta_names)}

    # Liste des features à garder (intersection taxonomie × metadata).
    kept_names: list[str] = []
    for taxo_name in config.usable_feature_names:
        if taxo_name in _OHLCV_COLUMNS:
            kept_names.append(taxo_name)
        elif taxo_name in name_to_idx:
            kept_names.append(taxo_name)
        else:
            logger.warning(
                "Feature %s dans la taxonomie mais absente du .npy (%s %s) — ignorée",
                taxo_name, asset, timeframe,
            )
    if not kept_names:
        raise NpyRealLoaderError(
            f"Aucune feature utilisable dans le .npy pour {asset} × {timeframe}"
        )

    # Construit le DataFrame.
    timestamps = [_ts_to_datetime(t) for t in ts]
    feature_dict: dict[str, pl.Series] = {"timestamp": pl.Series("timestamp", timestamps)}
    for name in kept_names:
        if name in _OHLCV_COLUMNS:
            idx = _OHLCV_COLUMNS.index(name)
        else:
            idx = name_to_idx[name]
        feature_dict[name] = pl.Series(name, x[:, idx].astype("float64"))
    df = pl.DataFrame(feature_dict)

    logger.info(
        "Features reelles chargees : %s × %s, %d/%d features (taxonomie × metadata), %d bougies",
        asset, timeframe, len(kept_names), len(meta_names), df.height,
    )
    return FeaturesFrame(
        asset=asset,
        timeframe=timeframe,
        df=df,
        feature_names=tuple(kept_names),
        data_version=f"npy:{asset_class}/{timeframe}/{asset}",
    )


def compute_validity_mask(ohlcv_df: pl.DataFrame) -> np.ndarray:
    """Calcule un masque bool des bougies valides (alignement OHLCV/Features).

    Une bougie est invalide si :
      - open, high, low, close contient NaN/inf
      - low > high, ou open/close hors [low, high]

    Returns:
        np.ndarray de bool, True = valide.
    """
    if ohlcv_df.is_empty():
        return np.array([], dtype=bool)
    critical = ("open", "high", "low", "close")
    arr = ohlcv_df.select(critical).to_numpy()
    valid = np.ones(len(arr), dtype=bool)
    # NaN / inf
    for c in critical:
        col = ohlcv_df[c].to_numpy()
        valid &= ~np.isnan(col) & ~np.isinf(col)
    # low <= high
    valid &= arr[:, 2] <= arr[:, 1]  # low <= high
    # open in [low, high]
    valid &= (arr[:, 0] >= arr[:, 2]) & (arr[:, 0] <= arr[:, 1])
    # close in [low, high]
    valid &= (arr[:, 3] >= arr[:, 2]) & (arr[:, 3] <= arr[:, 1])
    return valid


def _sanitize_ohlcv(df: pl.DataFrame) -> pl.DataFrame:
    """Nettoie les bougies avec des NaN ou des OHLC invalides."""
    critical = ("open", "high", "low", "close")
    null_or_nan = [pl.col(c).is_null() | pl.col(c).is_nan() for c in critical]
    df = df.filter(~pl.any_horizontal(null_or_nan))
    # low <= high, low <= open, low <= close, high >= open, high >= close
    df = df.filter(
        (pl.col("low") <= pl.col("high"))
        & (pl.col("open") >= pl.col("low"))
        & (pl.col("open") <= pl.col("high"))
        & (pl.col("close") >= pl.col("low"))
        & (pl.col("close") <= pl.col("high"))
    )
    return df


def _ts_to_datetime(ts_ms: int) -> datetime:
    """Convertit un timestamp Unix ms en datetime UTC."""
    return datetime.fromtimestamp(int(ts_ms) / 1000, tz=timezone.utc)
