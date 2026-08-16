"""data/ohlcv.py — Chargement des bougies OHLCV réelles depuis les CSV bruts.

Rôle SIMPLE (périmètre research) :
  - fournir au moteur une série OHLCV en VRAIS prix (open/high/low/close/volume)
    pour l'évaluation (ATR, SL/TP, simulation intrabar, MFE/MAE) ;
  - rien d'autre. Pas de DuckDB, pas de DataStore, pas de brokers,
    pas de calcul de features (ce rôle appartient aux features .npy).

Source : technical_agent_dataset_brut/{asset_class}/{asset}/{timeframe}/*.csv
Format CSV : timestamp,asset,timeframe,open,high,low,close,volume
Les CSV sont étendus par année ; on les concatène, on trie et on dédoublonne.

NOTE IMPORTANTE (Q4) : les .npy compilés ne contiennent que des log-returns
normalisés, JAMAIS des prix. C'est pourquoi l'OHLCV d'exécution doit venir
des CSV bruts, et jamais des .npy. Le module data/npy_loader.py (stub zéros)
n'est plus utilisé ici.
"""

from __future__ import annotations

import glob
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Erreurs
# --------------------------------------------------------------------------- #

OHLCV_REQUIRED_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")


class OhlcvError(Exception):
    """Erreur générique OHLCV."""


class OhlcvSchemaError(OhlcvError):
    """Schéma invalide (colonnes manquantes ou inattendues)."""


class OhlcvEmptyError(OhlcvError):
    """Aucune bougie disponible pour (asset, timeframe)."""


# --------------------------------------------------------------------------- #
# Value object
# --------------------------------------------------------------------------- #


class OhlcvFrame:
    """Frame OHLCV alignée sur une série temporelle (asset × timeframe).

    Attributs:
        asset: Symbole (ex: 'BTCUSD').
        timeframe: Granularité (ex: '1h').
        df: DataFrame polars [timestamp, open, high, low, close, volume].
        data_version: Identifiant de la version de données utilisée.
    """

    asset: str
    timeframe: str
    df: pl.DataFrame
    data_version: str

    def __init__(
        self,
        asset: str,
        timeframe: str,
        df: pl.DataFrame,
        data_version: str,
    ) -> None:
        self.asset = asset
        self.timeframe = timeframe
        self.df = df
        self.data_version = data_version

    @property
    def n_bougies(self) -> int:
        """Nombre de bougies dans la frame."""
        return self.df.height

    @property
    def start_ts(self) -> datetime | None:
        """Timestamp de la première bougie (None si frame vide)."""
        if self.df.is_empty():
            return None
        return _to_datetime(self.df["timestamp"][0])

    @property
    def end_ts(self) -> datetime | None:
        """Timestamp de la dernière bougie (None si frame vide)."""
        if self.df.is_empty():
            return None
        return _to_datetime(self.df["timestamp"][-1])

    def to_arrays(self) -> dict[str, Any]:
        """Expose les colonnes OHLCV en arrays numpy (pour les calculs vectorisés).

        Returns:
            Dict {open, high, low, close, volume} → np.ndarray[float64],
            plus timestamp (dtype source).
        """
        return {
            "open": self.df["open"].to_numpy().astype("float64"),
            "high": self.df["high"].to_numpy().astype("float64"),
            "low": self.df["low"].to_numpy().astype("float64"),
            "close": self.df["close"].to_numpy().astype("float64"),
            "volume": self.df["volume"].to_numpy().astype("float64"),
            "timestamp": self.df["timestamp"].to_numpy(),
        }


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #


class _OhlcvBackend:
    """Interface minimale de backend OHLCV."""

    def fetch(
        self,
        asset: str,
        timeframe: str,
        data_version: str,
        *,
        asset_class: str = "indices",
    ) -> pl.DataFrame:
        """Retourne [timestamp, open, high, low, close, volume] trié ASC."""
        raise NotImplementedError


# NOTE sur le nom "raw": dans cette couche, "brut" = CSV de prix historiques
# (technical_agent_dataset_brut), pas d'abstraction de source. On oublie le
# naming "DataStore/DuckDB" de la version précédente.


class _CsvRawBackend(_OhlcvBackend):
    """Backend par défaut : lit les CSV de prix bruts (vrais prix).

    Chemin attendu :
        <raw_root>/<asset_class>/<asset>/<timeframe>/<asset>_<year>_<timeframe>.csv

    Les CSV contiennent : timestamp,asset,timeframe,open,high,low,close,volume.
    Deux pièges gérés :
      - le volume est parfois Int64, parfois Float64 selon l'année → on force
        le schéma (Float64) à la lecture pour pouvoir concaténer.
      - les CSV annuels sont concaténés, triés par timestamp et dédoublonnés.
    """

    def __init__(self, raw_root: str | Path | None = None) -> None:
        # NOTE: chemin réel des données téléchargées (scripts downloaders
        # de midasV3). Ajustable via make_default_provider(raw_root=...).
        self.raw_root = Path(raw_root) if raw_root else Path(
            r"D:/midas_v2/technical_agent_dataset_brut"
        )

    def fetch(
        self,
        asset: str,
        timeframe: str,
        data_version: str,
        *,
        asset_class: str = "indices",
    ) -> pl.DataFrame:
        # Mapping brut↔npy : les classes .npy 'stocks_tech'/'stocks_growth'/
        # 'stocks_value' partagent un SEUL dossier CSV brut 'stocks' (les
        # downloaders MIDAS V3 n'ont pas séparé les actions par style). Sans
        # ce mapping, la recherche sur les actions US échoue (OhlcvEmptyError).
        _RAW_CLASS_MAP = {
            "stocks_tech": "stocks",
            "stocks_growth": "stocks",
            "stocks_value": "stocks",
            "stock_tech": "stocks",
            "stock_growth": "stocks",
            "stock_value": "stocks",
        }
        raw_class = _RAW_CLASS_MAP.get(asset_class, asset_class)
        root = self.raw_root / raw_class / asset / timeframe
        if not root.is_dir():
            raise OhlcvEmptyError(
                f"Données brutes absentes : {root}. "
                f"Réassure-toi de (re)télécharger la classe '{asset_class}' "
                f"(scripts downloaders) — pas de prix CSV = pas d'exécution."
            )
        # Pattern fichier annuel : <asset>_<year>_<tf>.csv
        pattern = str(root / f"{asset}_*_{timeframe}.csv")
        csv_files = sorted(glob.glob(pattern))
        if not csv_files:
            raise OhlcvEmptyError(
                f"Aucun CSV {asset} × {timeframe} dans {root} "
                "(pattern: <asset>_<year>_<tf>.csv)"
            )
        # Forcer le schéma du volume (Int64 ↔ Float64 selon l'année).
        frame_list: list[pl.DataFrame] = []
        for f in csv_files:
            df = pl.read_csv(
                f,
                try_parse_dates=True,
                schema_overrides={"volume": pl.Float64},
            )
            df = df.select(
                [c for c in OHLCV_REQUIRED_COLUMNS if c in df.columns]
            )
            if len(df.columns) != len(OHLCV_REQUIRED_COLUMNS):
                missing = [c for c in OHLCV_REQUIRED_COLUMNS if c not in df.columns]
                raise OhlcvSchemaError(f"{Path(f).name}: colonnes manquantes {missing}")
            frame_list.append(df)
        if not frame_list:
            raise OhlcvEmptyError(f"Aucune bougie lue depuis {root}")
        concat = pl.concat(frame_list)
        # Le CSV contient timestamp ISO avec timezone ; on normalise en
        # datetime[us, UTC] (même format que les FeaturesFrame).
        concat = concat.sort("timestamp")
        return concat


# --------------------------------------------------------------------------- #
# Backend mémoire (tests uniquement)
# --------------------------------------------------------------------------- #


class _InMemoryBackend(_OhlcvBackend):
    """Backend de test : dictionnaire {(asset, tf): DataFrame}."""

    def __init__(self, frames: dict[tuple[str, str], pl.DataFrame] | None = None) -> None:
        self._frames: dict[tuple[str, str], pl.DataFrame] = dict(frames or {})

    def register(self, asset: str, timeframe: str, df: pl.DataFrame) -> None:
        self._frames[(asset, timeframe)] = df

    def fetch(
        self,
        asset: str,
        timeframe: str,
        data_version: str,
        *,
        asset_class: str = "indices",
    ) -> pl.DataFrame:
        key = (asset, timeframe)
        if key not in self._frames:
            raise OhlcvEmptyError(f"Aucune bougie en mémoire pour ({asset}, {timeframe})")
        return self._frames[key]


# --------------------------------------------------------------------------- #
# Loader public (singleton avec cache)
# --------------------------------------------------------------------------- #


class OhlcvProvider:
    """Loader OHLCV simple : un backend (CSV bruts) + cache mémoire.

    Attributes:
        backend: Backend de données (CSV bruts par défaut).
    """

    def __init__(self, backend: _OhlcvBackend | None = None) -> None:
        self._backend: _OhlcvBackend = backend or _CsvRawBackend()
        self._cache: dict[tuple[str, str, str], OhlcvFrame] = {}
        logger.info("OhlcvProvider instancié (backend=%s)", type(self._backend).__name__)

    # ------------------------------------------------------------------ #
    # API publique
    # ------------------------------------------------------------------ #

    def load(
        self,
        asset: str,
        timeframe: str,
        data_version: str,
        *,
        use_cache: bool = True,
        asset_class: str = "indices",
    ) -> OhlcvFrame:
        """Charge (ou récupère du cache) la série OHLCV réelle.

        Args:
            asset: Symbole (ex: 'NASDAQ100').
            timeframe: Granularité (15m/5m/1h/4h/1d).
            data_version: Identifiant de version (clé de cache).
            use_cache: Si True (défaut), renvoie le cache si disponible.
            asset_class: Classe d'actifs — dossier de la donnée brute
                ('indices', 'crypto', 'forex', 'commodities',
                 'stocks_growth', 'stocks_tech', 'stocks_value').

        Returns:
            OhlcvFrame validée et triée ASC par timestamp.

        Raises:
            OhlcvEmptyError: si le backend n'a rien pour (asset, timeframe).
            OhlcvSchemaError: si la frame n'a pas les colonnes OHLCV minimales.
        """
        cache_key = (asset, timeframe, data_version)
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        raw = self._backend.fetch(
            asset=asset, timeframe=timeframe, data_version=data_version,
            asset_class=asset_class,
        )
        frame = self._sanitize(raw, asset=asset, timeframe=timeframe, data_version=data_version)
        self._cache[cache_key] = frame
        logger.info(
            "OHLCV chargé : %s × %s version=%s, %d bougies [%s → %s]",
            asset, timeframe, data_version, frame.n_bougies,
            frame.start_ts, frame.end_ts,
        )
        return frame

    def invalidate(self, asset: str | None = None, timeframe: str | None = None) -> int:
        """Invalide le cache. Retourne le nombre d'entrées supprimées."""
        to_drop = [
            k for k in self._cache
            if (asset is None or k[0] == asset) and (timeframe is None or k[1] == timeframe)
        ]
        for k in to_drop:
            del self._cache[k]
        return len(to_drop)

    def list_assets(self) -> list[tuple[str, str]]:
        """Liste les couples (asset, timeframe) déjà chargés en cache."""
        return sorted({(k[0], k[1]) for k in self._cache})

    # ------------------------------------------------------------------ #
    # Helpers privés
    # ------------------------------------------------------------------ #

    @staticmethod
    def _sanitize(
        df: pl.DataFrame,
        asset: str,
        timeframe: str,
        data_version: str,
    ) -> OhlcvFrame:
        """Valide le schema, normalise le timestamp, dédoublonne, trie, drop NaN critiques."""
        _validate_schema(df)
        # Normalisation du timestamp vers datetime[us, UTC].
        ts_dtype = df.schema["timestamp"]
        if ts_dtype == pl.Datetime("us", "UTC"):
            pass  # déjà normalisé
        elif ts_dtype == pl.Datetime:
            df = df.with_columns(
                pl.col("timestamp").dt.replace_time_zone("UTC").dt.cast_time_unit("us")
            )
        elif ts_dtype in (pl.Int64, pl.Int32):
            # int = ms Unix (convention npy_loader)
            df = df.with_columns(
                pl.from_epoch("timestamp", time_unit="ms")
                .dt.replace_time_zone("UTC")
                .dt.cast_time_unit("us")
                .alias("timestamp")
            )
        else:
            try:
                df = df.with_columns(
                    pl.col("timestamp").str.to_datetime(time_unit="us").alias("timestamp")
                )
            except Exception as exc:
                raise OhlcvSchemaError(
                    f"Timestamp OHLCV non convertible en datetime[us, UTC] : "
                    f"dtype={ts_dtype}, erreur={exc}"
                ) from exc
        df = _dedupe_and_sort(df)
        df = _drop_critical_nans(df)
        return OhlcvFrame(
            asset=asset,
            timeframe=timeframe,
            df=df,
            data_version=data_version,
        )


# --------------------------------------------------------------------------- #
# Helpers module-level
# --------------------------------------------------------------------------- #


def _validate_schema(df: pl.DataFrame) -> None:
    """Vérifie la présence des colonnes OHLCV minimales."""
    missing = [c for c in OHLCV_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise OhlcvSchemaError(
            f"Schéma OHLCV invalide : colonnes manquantes {missing}. "
            f"Colonnes présentes : {df.columns}"
        )


def _dedupe_and_sort(df: pl.DataFrame) -> pl.DataFrame:
    """Trie ASC par timestamp, dédoublonne sur (timestamp) en gardant la première ligne."""
    return df.sort("timestamp").unique(subset=["timestamp"], keep="first")


def _drop_critical_nans(df: pl.DataFrame) -> pl.DataFrame:
    """Drop les bougies dont OHLC contient un NaN (volume peut être NaN en crypto peu liquide)."""
    critical = ("open", "high", "low", "close")
    null_or_nan = [pl.col(c).is_null() | pl.col(c).is_nan() for c in critical]
    return df.filter(~pl.any_horizontal(null_or_nan))


def _to_datetime(value: Any) -> datetime | None:
    """Convertit une valeur polars/timestamp en datetime Python, ou None."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return value.to_pydatetime() if hasattr(value, "to_pydatetime") else None
    except (AttributeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #


def make_default_provider(raw_root: str | Path | None = None) -> OhlcvProvider:
    """Construit un OhlcvProvider avec le backend CSV bruts (vrais prix).

    Args:
        raw_root: Racine des CSV bruts. Défaut :
            D:/midas_v2/technical_agent_dataset_brut
    """
    return OhlcvProvider(backend=_CsvRawBackend(raw_root=raw_root))


def make_test_provider(
    frames: dict[tuple[str, str], pl.DataFrame] | None = None,
) -> OhlcvProvider:
    """Construit un OhlcvProvider avec un backend en mémoire (pour les tests)."""
    return OhlcvProvider(backend=_InMemoryBackend(frames=frames))