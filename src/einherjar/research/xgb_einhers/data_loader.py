"""data_loader.py - Chargement et nettoyage des données MIDAS V3.

Charge X, Y_dir, Y_ret, Y_hor + metadata.json pour un (asset, TF).
Exclut :
- Les 5 colonnes OHLCV de X (open, high, low, close, volume) → réponse Q6
- Les features marquées excluded=True dans features_taxonomy.json
Garde :
- 213 features utilisables (218 - 5 OHLCV)

API publique :
- load_xy(asset, tf, asset_class, ...) -> LoadedData
- load_ohlcv(asset, tf, asset_class, raw_root) -> pl.DataFrame
- align_xy_with_ohlcv(loaded, ohlcv_df) -> (X_aligned, ohlcv_aligned, ts_aligned)
- temporal_split(X, y, ratios, embargo) -> TrainValHoldoutSplit
- get_target_for_horizon(loaded, horizon_idx) -> (target, valid_mask, Y_hor_col)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import polars as pl

from .paths import COMPILED_DIR, OHLCV_DIR, TAXONOMY_PATH
from .types import LoadedData, TrainValHoldoutSplit

logger = logging.getLogger(__name__)

# Features OHLCV toujours exclues (réponse Q6 : on a les prix via CSV bruts)
OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")


# --------------------------------------------------------------------------- #
# Taxonomie
# --------------------------------------------------------------------------- #


def load_usable_feature_names() -> set[str]:
    """Charge les noms de features marquées usable (excluded != True) depuis features_taxonomy.json.

    Returns:
        Set des noms de features utilisables.
    """
    with open(TAXONOMY_PATH) as f:
        tax = json.load(f)
    return {k for k, v in tax["features"].items() if not v.get("excluded", False)}


def load_feature_meta(
    asset: str,
    timeframe: str,
    asset_class: str = "crypto",
    compiled_dir: Path = COMPILED_DIR,
) -> tuple[list[str], list[str]]:
    """Lit (feature_names, horizons) depuis metadata.json SANS charger les arrays.

    FIX MEM-01 (2026-08-24) : remplace load_multi_asset() quand on n'a besoin
    QUE des metadonnees. Les feature_names/horizons sont identiques entre actifs
    d'un meme (asset_class, timeframe) - verifie par la coherence imposee dans
    load_multi_asset_split.
    """
    meta_path = Path(compiled_dir) / asset_class / timeframe / "metadata.json"
    with open(meta_path) as f:
        meta = json.load(f)
    return list(meta["feature_names"]), list(meta["horizons"])


# --------------------------------------------------------------------------- #
# Chargement X / Y
# --------------------------------------------------------------------------- #


def load_xy(
    asset: str,
    timeframe: str,
    asset_class: str = "crypto",
    compiled_dir: Path = COMPILED_DIR,
) -> LoadedData:
    """Charge X, Y_dir, Y_ret, Y_hor + metadata pour un (asset, TF).

    Args:
        asset : ex 'BTCUSD'
        timeframe : ex '1h'
        asset_class : ex 'crypto'
        compiled_dir : racine des .npy MIDAS V3

    Returns:
        LoadedData avec :
        - X : (N, 213) features utilisables (OHLCV exclues)
        - Y_dir : (N, H) int8
        - Y_ret : (N, H) float32
        - Y_hor : (N, H) float32
        - feature_names : 213 noms
        - horizons : H noms (ex ['6h', '12h', '1d', '2d'])
    """
    base = Path(compiled_dir) / asset_class / timeframe
    if not base.exists():
        raise FileNotFoundError(f"Répertoire absent : {base}")

    # Charger arrays
    ts = np.load(base / f"{asset}_ts.npy")
    X_raw = np.load(base / f"{asset}_X.npy")
    Y_dir = np.load(base / f"{asset}_Y_dir.npy")
    Y_ret = np.load(base / f"{asset}_Y_ret.npy")
    Y_hor = np.load(base / f"{asset}_Y_hor.npy")

    # Charger metadata
    with open(base / "metadata.json") as f:
        meta = json.load(f)
    all_feature_names = tuple(meta["feature_names"])
    horizons = tuple(meta["horizons"])

    # Exclure les 5 colonnes OHLCV (réponse Q6)
    ohlcv_idx = [i for i, n in enumerate(all_feature_names) if n in OHLCV_COLUMNS]
    keep_idx = [i for i in range(len(all_feature_names)) if i not in ohlcv_idx]
    X = X_raw[:, keep_idx]
    feature_names = tuple(n for i, n in enumerate(all_feature_names) if i in keep_idx)

    # Filtrer la taxonomie : on garde uniquement les features usable
    # (déjà fait via la taxonomie dans metadata.json, mais on double-check)
    usable = load_usable_feature_names()
    final_idx = [i for i, n in enumerate(feature_names) if n in usable]
    X = X[:, final_idx]
    feature_names = tuple(n for i, n in enumerate(feature_names) if i in final_idx)

    # Sanity : pas de NaN/Inf (déjà vérifié mais on assert)
    assert not np.isnan(X).any(), "X contient des NaN"
    assert not np.isinf(X).any(), "X contient des Inf"

    logger.info(
        "load_xy(%s, %s) : N=%d, F=%d, H=%d, horizons=%s",
        asset, timeframe, X.shape[0], X.shape[1], len(horizons), horizons,
    )
    return LoadedData(
        asset=asset,
        asset_class=asset_class,
        timeframe=timeframe,
        timestamps=ts,
        X=X,
        Y_dir=Y_dir,
        Y_ret=Y_ret,
        Y_hor=Y_hor,
        feature_names=feature_names,
        horizons=horizons,
    )


# --------------------------------------------------------------------------- #
# Chargement OHLCV (CSV bruts)
# --------------------------------------------------------------------------- #


def load_ohlcv(
    asset: str,
    timeframe: str,
    asset_class: str = "crypto",
    ohlcv_dir: Path = OHLCV_DIR,
) -> pl.DataFrame:
    """Charge les OHLCV depuis les CSV annuels bruts.

    Returns:
        DataFrame polars avec colonnes [timestamp (datetime[us, UTC]),
        open, high, low, close, volume]. Trié ASC par timestamp.
    """
    # FIX BUG-12 (Sprint 3.6) : les 3 sous-classes stocks (growth/tech/value)
    # partagent le meme dossier OHLCV "stocks/".
    from .multi_asset_loader import resolve_ohlcv_class
    ohlcv_class = resolve_ohlcv_class(asset_class)
    root = Path(ohlcv_dir) / ohlcv_class / asset / timeframe
    if not root.is_dir():
        raise FileNotFoundError(f"Dossier OHLCV absent : {root}")

    pattern = str(root / f"{asset}_*_{timeframe}.csv")
    import glob
    csv_files = sorted(glob.glob(pattern))
    if not csv_files:
        raise FileNotFoundError(f"Aucun CSV trouvé : {pattern}")

    frame_list = []
    for f in csv_files:
        df = pl.read_csv(
            f,
            try_parse_dates=True,
            schema_overrides={"volume": pl.Float64},
        )
        # Garder les colonnes OHLCV + timestamp
        cols = ["timestamp", "open", "high", "low", "close", "volume"]
        df = df.select([c for c in cols if c in df.columns])
        frame_list.append(df)
    concat = pl.concat(frame_list)
    concat = concat.sort("timestamp")
    # Forcer timestamp en datetime[us, UTC]
    if concat.schema["timestamp"] != pl.Datetime("us", "UTC"):
        if concat.schema["timestamp"] in (pl.Datetime,):
            concat = concat.with_columns(
                pl.col("timestamp").dt.replace_time_zone("UTC").dt.cast_time_unit("us")
            )
        elif concat.schema["timestamp"] in (pl.Int64, pl.Int32):
            concat = concat.with_columns(
                pl.from_epoch("timestamp", time_unit="ms")
                .dt.replace_time_zone("UTC")
                .dt.cast_time_unit("us")
            )
    return concat


def align_xy_with_ohlcv(
    loaded: LoadedData,
    ohlcv_df: pl.DataFrame,
) -> tuple[np.ndarray, pl.DataFrame, np.ndarray]:
    """Aligne X/Y sur les timestamps OHLCV par inner join.

    Returns:
        (X_aligned, ohlcv_aligned, ts_aligned) tous triés ASC et de même longueur.
    """
    # Convertir les timestamps ms epoch en datetime[us, UTC] pour le join
    ts_dt = pl.from_numpy(loaded.timestamps.astype(np.int64), schema=["ts_ms"])
    ts_dt = ts_dt.with_columns(
        pl.from_epoch(pl.col("ts_ms"), time_unit="ms")
        .dt.replace_time_zone("UTC")
        .dt.cast_time_unit("us")
        .alias("timestamp")
    )

    # DF avec timestamp + indices originaux
    n = loaded.n_samples
    xy_df = ts_dt.with_row_index(name="orig_idx").select(
        pl.col("timestamp"), pl.col("orig_idx")
    )
    # P1-ALIGN (2026-08-24) : l'ancienne version faisait un filter() PUIS un
    # second join() sur les memes donnees - deux copies inutiles du DataFrame
    # OHLCV. Un unique gather par ohlcv_idx suffit : moins de RAM, ~2x plus vite.
    # NB : si ohlcv_df contenait des timestamps dupliques, le join inner les
    # dupliquerait aussi cote X ; on deduplique donc explicitement en amont.
    if ohlcv_df["timestamp"].is_duplicated().any():
        logger.warning(
            "align_xy_with_ohlcv : %d timestamps dupliques dans OHLCV -> keep=first",
            int(ohlcv_df["timestamp"].is_duplicated().sum()),
        )
        ohlcv_df = ohlcv_df.unique(subset=["timestamp"], keep="first").sort("timestamp")

    joined = (
        ohlcv_df.with_row_index(name="ohlcv_idx")
        .join(xy_df, on="timestamp", how="inner")
        .sort("timestamp")
    )

    # Réindexer X selon l'ordre du join (gather numpy, pas de copie intermediaire)
    orig_idx = joined["orig_idx"].to_numpy()
    X_aligned = loaded.X[orig_idx]
    ts_aligned = joined["timestamp"]

    # OHLCV aligne = gather des lignes du join (meme ordre que X_aligned)
    ohlcv_aligned = ohlcv_df[joined["ohlcv_idx"].to_numpy()]

    logger.info(
        "align_xy_with_ohlcv : %d/%d lignes conservées (%.1f%%)",
        X_aligned.shape[0], n, 100 * X_aligned.shape[0] / n,
    )
    return X_aligned, ohlcv_aligned, ts_aligned  # pyright: ignore[reportReturnType]


# --------------------------------------------------------------------------- #
# Target pour un horizon donné
# --------------------------------------------------------------------------- #


def get_target_for_horizon(
    loaded: LoadedData,
    horizon_idx: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construit le target supervisé pour un horizon donné.

    Returns:
        (target, valid_mask, y_hor) où :
        - target : (N,) float32 = Y_ret[:, h] (signed return)
        - valid_mask : (N,) bool = Y_dir[:, h] != -100
        - y_hor : (N,) float32 = Y_hor[:, h] (horizon en bars)
    """
    valid_mask = loaded.Y_dir[:, horizon_idx] != -100
    target = loaded.Y_ret[:, horizon_idx].copy()
    y_hor = loaded.Y_hor[:, horizon_idx].copy()
    return target, valid_mask, y_hor


# --------------------------------------------------------------------------- #
# Split temporel
# --------------------------------------------------------------------------- #


def temporal_split(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    holdout_ratio: float = 0.2,
    embargo_bars: int = 50,
    horizon_bars: int = 0,
) -> TrainValHoldoutSplit:
    """Split temporel 60/20/20 avec embargo entre train/val et val/holdout.

    Args:
        X : (N, F)
        y : (N,) ou (N, ...)
        train_ratio : proportion pour le train (defaut 0.6).
        val_ratio : proportion pour la val (defaut 0.2).
        holdout_ratio : proportion pour le holdout (defaut 0.2).
        embargo_bars : nb de bougies exclues aux frontières (minimum).
        horizon_bars : horizon du label en bars (Sprint 3.0 FIX #2).
                       L'embargo effectif est max(embargo_bars, horizon_bars)
                       pour eviter le leakage du target entre splits.

    Returns:
        TrainValHoldoutSplit
    """
    n = X.shape[0]
    assert abs(train_ratio + val_ratio + holdout_ratio - 1.0) < 1e-6

    # Sprint 3.0 FIX #2 : embargo proportionnel a l'horizon
    # Si horizon_bars > embargo_bars, l'utiliser pour eviter le leakage
    # du target (le label d'une bougie t utilise des bougies jusqu'a t+horizon).
    effective_embargo = max(embargo_bars, horizon_bars)

    train_end = int(n * train_ratio)
    val_start = train_end + effective_embargo
    val_end = val_start + int(n * val_ratio)
    holdout_start = val_end + effective_embargo

    assert val_start < val_end, f"Train/val overlap : {val_start} < {val_end}"
    assert holdout_start < n, f"Holdout déborde : {holdout_start} >= {n}"

    train_idx = np.arange(0, train_end)
    val_idx = np.arange(val_start, val_end)
    holdout_idx = np.arange(holdout_start, n)

    return TrainValHoldoutSplit(
        train_X=X[train_idx],
        train_y=y[train_idx],
        val_X=X[val_idx],
        val_y=y[val_idx],
        holdout_X=X[holdout_idx],
        holdout_y=y[holdout_idx],
        train_indices=train_idx,
        val_indices=val_idx,
        holdout_indices=holdout_idx,
        embargo_bars=effective_embargo,
    )
