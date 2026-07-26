"""LiveDataStore — Stockage incremental des donnees de marche par asset/TF.

Architecture de stockage live :
- Un fichier Parquet par (asset, timeframe) dans data/live/
- Chaque fichier contient OHLCV + features calculees
- Append incremental a chaque nouvelle bougie
- Fenêtre glissante : on garde les N dernieres bougies en memoire

Reference : Section 1.6 et 4.4 du CDC EINHERJAR.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

LIVE_DIR = Path(__file__).resolve().parents[2] / "data" / "live"


class LiveDataStore:
    """Store incremental pour les donnees de marche en temps reel.

    Attributs:
        base_dir: Repertoire racine des fichiers live.
        window_size: Nombre de bougies conservees en memoire par asset/TF.
        _cache: Cache memoire { (asset, tf) -> DataFrame }.
    """

    def __init__(self, base_dir: str | Path | None = None, window_size: int = 500) -> None:
        """Initialise le store.

        Args:
            base_dir: Repertoire des fichiers Parquet. Defaut data/live/.
            window_size: Taille de la fenetre glissante en memoire.
        """
        self.base_dir = Path(base_dir) if base_dir else LIVE_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.window_size = window_size
        self._cache: dict[tuple[str, str], pl.DataFrame] = {}

    def _get_path(self, asset: str, timeframe: str) -> Path:
        """Retourne le chemin du fichier Parquet pour un asset/TF.

        Args:
            asset: Symbole.
            timeframe: Timeframe.

        Returns:
            Chemin absolu du fichier Parquet.
        """
        # Normaliser le nom de fichier (remplacer / par _)
        safe_asset = asset.replace("/", "_")
        return self.base_dir / f"{safe_asset}_{timeframe}.parquet"

    def load(self, asset: str, timeframe: str) -> pl.DataFrame:
        """Charge l'historique d'un asset/TF depuis le Parquet.

        Args:
            asset: Symbole.
            timeframe: Timeframe.

        Returns:
            DataFrame polars. Vide si le fichier n'existe pas.
        """
        path = self._get_path(asset, timeframe)
        if path.exists():
            return pl.read_parquet(path)
        return pl.DataFrame(
            schema={
                "timestamp": pl.Datetime,
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Float64,
            }
        )

    def append(
        self,
        asset: str,
        timeframe: str,
        candle: dict[str, Any],
        features: dict[str, Any] | None = None,
    ) -> pl.DataFrame:
        """Ajoute une bougie et ecrit sur disque.

        Cette methode est appelee a chaque cloture de bougie.

        Args:
            asset: Symbole.
            timeframe: Timeframe.
            candle: Dict {timestamp, open, high, low, close, volume}.
            features: Dict optionnel de features calculees pour cette bougie.

        Returns:
            DataFrame mis a jour (fenetre glissante).
        """
        key = (asset, timeframe)

        # Charger depuis le cache ou le disque
        df = self._cache.get(key)
        if df is None:
            df = self.load(asset, timeframe)

        # Construire la nouvelle ligne
        row = dict(candle)
        if features:
            row.update(features)

        new_df = pl.DataFrame([row])

        # Concatener et tronquer a la fenetre
        df = pl.concat([df, new_df], how="vertical_relaxed")
        if len(df) > self.window_size:
            df = df.tail(self.window_size)

        # Mettre a jour le cache
        self._cache[key] = df

        # Ecrire sur disque (append mode pas supporte par polars, donc overwrite)
        path = self._get_path(asset, timeframe)
        df.write_parquet(path)

        return df

    def get_window(
        self,
        asset: str,
        timeframe: str,
        n: int | None = None,
    ) -> pl.DataFrame:
        """Retourne les N dernieres bougies d'un asset/TF.

        Args:
            asset: Symbole.
            timeframe: Timeframe.
            n: Nombre de bougies. None = tout.

        Returns:
            DataFrame des N dernieres bougies.
        """
        key = (asset, timeframe)
        df = self._cache.get(key)
        if df is None:
            df = self.load(asset, timeframe)
            self._cache[key] = df

        if n is not None and len(df) > n:
            return df.tail(n)
        return df

    def list_assets(self) -> list[tuple[str, str]]:
        """Liste tous les (asset, timeframe) disponibles.

        Returns:
            Liste des tuples (asset, timeframe).
        """
        assets = []
        for f in self.base_dir.glob("*.parquet"):
            stem = f.stem
            parts = stem.rsplit("_", 1)
            if len(parts) == 2:
                assets.append((parts[0], parts[1]))
        return assets

    def clear_cache(self) -> None:
        """Vide le cache memoire."""
        self._cache.clear()

    def get_last_timestamp(self, asset: str, timeframe: str) -> datetime | None:
        """Retourne le timestamp de la derniere bougie.

        Args:
            asset: Symbole.
            timeframe: Timeframe.

        Returns:
            Timestamp ou None si pas de donnees.
        """
        df = self.get_window(asset, timeframe, n=1)
        if len(df) > 0 and "timestamp" in df.columns:
            ts = df["timestamp"][0]
            if isinstance(ts, datetime):
                return ts
        return None
