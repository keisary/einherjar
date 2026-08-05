"""Interface OHLCV pour le moteur de découverte.

Wrap synchrone de `einherjar.data.store.DataStore` (DuckDB) en surface
pure polars. Fournit un cache mémoire par (asset, timeframe, data_version)
pour éviter de recharger la même série à chaque évaluation.

Responsabilités :
  - Lecture synchrone des bougies OHLCV depuis le store persistant.
  - Validation du schéma minimal (timestamp, open, high, low, close, volume).
  - Tri ascendant par timestamp, déduplication, NaN handling.
  - Cache en lecture, invalidation explicite par data_version.

Hors périmètre :
  - Calcul des features (voir data/features.py).
  - Logique d'évaluation (voir engine/evaluator.py).
  - Fetch broker distant (voir data/ohlcv_manager.py pour le live).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

logger = logging.getLogger(__name__)

# Colonnes OHLCV minimales exigées par le moteur de découverte.
OHLCV_REQUIRED_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
)


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #


class OhlcvError(Exception):
    """Erreur générique du loader OHLCV."""


class OhlcvSchemaError(OhlcvError):
    """Schéma OHLCV invalide (colonnes manquantes ou types incorrects)."""


class OhlcvEmptyError(OhlcvError):
    """Aucune bougie OHLCV disponible pour (asset, timeframe)."""


# --------------------------------------------------------------------------- #
# Frame value object
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
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

    @property
    def n_bougies(self) -> int:
        """Nombre de bougies dans la frame."""
        return self.df.height

    @property
    def start_ts(self) -> datetime | None:
        """Timestamp de la première bougie (None si frame vide)."""
        if self.df.is_empty():
            return None
        ts = self.df["timestamp"][0]
        return _to_datetime(ts)

    @property
    def end_ts(self) -> datetime | None:
        """Timestamp de la dernière bougie (None si frame vide)."""
        if self.df.is_empty():
            return None
        ts = self.df["timestamp"][-1]
        return _to_datetime(ts)

    def to_arrays(self) -> dict[str, Any]:
        """Expose les colonnes OHLCV en arrays numpy (pour les calculs vectorisés).

        Returns:
            Dict {open, high, low, close, volume} → np.ndarray[float64].
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
# Loader — privé
# --------------------------------------------------------------------------- #


class _OhlcvBackend:
    """Interface abstraite de backend de données OHLCV.

    Permet de substituer le backend par défaut (DataStore DuckDB) par un
    backend de test (NpyLoader, DataFrame en mémoire) sans modifier
    l'API publique.
    """

    def fetch(
        self,
        asset: str,
        timeframe: str,
        data_version: str,
    ) -> pl.DataFrame:
        """Charge les bougies OHLCV pour (asset, timeframe).

        Args:
            asset: Symbole.
            timeframe: Granularité.
            data_version: Identifiant de version (peut être ignoré par le backend).

        Returns:
            DataFrame polars [timestamp, open, high, low, close, volume] trié ASC.

        Raises:
            OhlcvError: en cas d'échec du backend.
        """
        raise NotImplementedError


class _DataStoreBackend(_OhlcvBackend):
    """Backend par défaut : DuckDB via DataStore."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        try:
            from einherjar.data.store import DataStore
        except ImportError as exc:
            raise OhlcvError(
                "DataStore indisponible — vérifie que einherjar.data.store est importable"
            ) from exc
        self._store = DataStore(db_path=db_path)
        logger.debug("DataStore ouvert : %s", self._store.db_path)

    def fetch(
        self,
        asset: str,
        timeframe: str,
        data_version: str,
        *,
        asset_class: str = "crypto",
    ) -> pl.DataFrame:
        """Fetch OHLCV depuis le DataStore. Fallback npy_loader si vide.

        P0-02 (contrat OHLC) : si le DataStore n'a pas l'asset, on tente
        un fallback vers npy_loader.load_ohlcv_from_npy qui lit les
        .npy MIDAS V3 directement. Le store est la voie principale (live),
        les .npy sont la voie bootstrap.
        """
        df = self._store.query_ohlcv(asset=asset, timeframe=timeframe, since=None, limit=10_000_000)
        if df.is_empty():
            # Fallback : lecture directe des .npy MIDAS V3.
            try:
                from einherjar.data.npy_loader import load_ohlcv_from_npy
                fallback = load_ohlcv_from_npy(
                    asset=asset, asset_class=asset_class, timeframe=timeframe,
                )
                if fallback is not None and not fallback.is_empty():
                    logger.info(
                        "OHLCV fallback via npy_loader : %s x %s, %d bougies",
                        asset, timeframe, fallback.height,
                    )
                    return fallback
            except Exception as exc:  # noqa: BLE001
                logger.debug("npy_loader fallback echoue (%s)", exc)
            raise OhlcvEmptyError(
                f"Aucune bougie OHLCV pour ({asset}, {timeframe}) — "
                f"DataStore vide et fallback npy_loader indisponible. "
                f"Verifie --data-root ({asset_class}/{timeframe}/{asset}_*.npy)."
            )
        return df


class _InMemoryBackend(_OhlcvBackend):
    """Backend de test : dictionnaire {(asset, tf): DataFrame} injecté manuellement."""

    def __init__(self, frames: dict[tuple[str, str], pl.DataFrame] | None = None) -> None:
        self._frames: dict[tuple[str, str], pl.DataFrame] = dict(frames or {})

    def register(self, asset: str, timeframe: str, df: pl.DataFrame) -> None:
        self._frames[(asset, timeframe)] = df

    def fetch(self, asset: str, timeframe: str, data_version: str) -> pl.DataFrame:
        key = (asset, timeframe)
        if key not in self._frames:
            raise OhlcvEmptyError(f"Aucune bougie en mémoire pour ({asset}, {timeframe})")
        return self._frames[key]


# --------------------------------------------------------------------------- #
# Loader public (singleton avec cache)
# --------------------------------------------------------------------------- #


class OhlcvProvider:
    """Loader OHLCV synchrone pour le moteur de découverte.

    Encapsule un backend de données (DataStore par défaut) et un cache
    mémoire indexé par (asset, timeframe, data_version).

    Attributes:
        backend: Backend de données (DataStore, InMemory, custom).
    """

    def __init__(self, backend: _OhlcvBackend | None = None) -> None:
        """Initialise le provider OHLCV.

        Args:
            backend: Backend de données (DataStore DuckDB par défaut).
        """
        self._backend: _OhlcvBackend = backend or _DataStoreBackend()
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
        asset_class: str = "crypto",
    ) -> OhlcvFrame:
        """Charge (ou récupère du cache) la série OHLCV.

        Args:
            asset: Symbole.
            timeframe: Granularité.
            data_version: Identifiant de version (utilisé comme clé de cache).
            use_cache: Si True (défaut), renvoie le cache si disponible.
            asset_class: Classe d'actifs (pour le fallback npy_loader).

        Returns:
            OhlcvFrame validée et triée ASC par timestamp.

        Raises:
            OhlcvEmptyError: si le backend n'a rien pour (asset, timeframe).
            OhlcvSchemaError: si la frame n'a pas les colonnes OHLCV minimales.
        """
        cache_key = (asset, timeframe, data_version)
        if use_cache and cache_key in self._cache:
            logger.debug("OHLCV cache hit : %s", cache_key)
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
        """Invalide le cache. Retourne le nombre d'entrées supprimées.

        Args:
            asset: Si fourni, ne supprime que les frames de cet asset.
            timeframe: Si fourni, ne supprime que les frames de ce timeframe.

        Returns:
            Nombre d'entrées de cache supprimées.
        """
        to_drop = [
            k for k in self._cache
            if (asset is None or k[0] == asset) and (timeframe is None or k[1] == timeframe)
        ]
        for k in to_drop:
            del self._cache[k]
        if to_drop:
            logger.debug("Cache OHLCV invalidé : %d entrées", len(to_drop))
        return len(to_drop)

    def list_assets(self) -> list[tuple[str, str]]:
        """Liste les couples (asset, timeframe) déjà chargés en cache.

        Returns:
            Liste triée de tuples (asset, timeframe).
        """
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
        """Valide le schema, normalise le timestamp, dedoublonne, trie, drop les NaN critiques.

        P0-02 + fix npy_loader : on normalise le timestamp en datetime[us, UTC]
        (le format des FeaturesFrame) pour que les join OHLCV x features
        fonctionnent sans cast manuel. Le backend peut renvoyer int64 (ms Unix)
        ou datetime ; on normalise vers datetime[us, UTC].
        """
        _validate_schema(df)
        # Normalisation du timestamp vers datetime[us, UTC].
        ts_dtype = df.schema["timestamp"]
        if ts_dtype == pl.Datetime("us", "UTC"):
            pass  # deja normalise
        elif ts_dtype == pl.Datetime:
            # autre timezone ou precision : on cast en us + UTC
            df = df.with_columns(
                pl.col("timestamp").dt.replace_time_zone("UTC").dt.cast_time_unit("us")
            )
        elif ts_dtype == pl.Int64 or ts_dtype == pl.Int32:
            # int = ms Unix (convention npy_loader)
            # from_epoch cree un Datetime sans timezone ; on l'attache UTC,
            # puis on convertit en us.
            df = df.with_columns(
                pl.from_epoch("timestamp", time_unit="ms")
                .dt.replace_time_zone("UTC")
                .dt.cast_time_unit("us")
                .alias("timestamp")
            )
        else:
            # Autres types (string, etc.) : on tente un cast datetime direct.
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
    return (
        df
        .sort("timestamp")
        .unique(subset=["timestamp"], keep="first")
    )


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


def make_default_provider(db_path: str | Path | None = None) -> OhlcvProvider:
    """Construit un OhlcvProvider avec le backend DataStore par défaut.

    Args:
        db_path: Chemin optionnel vers la base DuckDB. Si None, utilise
            le chemin par défaut du DataStore.

    Returns:
        OhlcvProvider prêt à l'emploi.
    """
    backend = _DataStoreBackend(db_path=db_path)
    return OhlcvProvider(backend=backend)


def make_test_provider(
    frames: dict[tuple[str, str], pl.DataFrame] | None = None,
) -> OhlcvProvider:
    """Construit un OhlcvProvider avec un backend en mémoire (pour les tests).

    Args:
        frames: Dict {(asset, timeframe): DataFrame polars OHLCV}.

    Returns:
        OhlcvProvider avec backend InMemory.
    """
    return OhlcvProvider(backend=_InMemoryBackend(frames=frames))
