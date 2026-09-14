"""LiveDataStore — Stockage incremental des donnees de marche par asset/TF.

Architecture de stockage live :
- Un fichier Parquet par (asset, timeframe) dans data/live/
- Chaque fichier contient l'OHLCV (les features sont recalculees, jamais stockees ici)
- Append incremental a chaque nouvelle bougie
- Fenetre glissante : on garde les N dernieres bougies en memoire

Ecriture disque : polars ne sait pas appender un Parquet, chaque sauvegarde
reecrit le fichier entier. Avec ~70 couples et 1 500 bougies (1,3 Mo au total),
reecrire a chaque bougie est du churn inutile : les ecritures sont donc
**throttlees** (`write_throttle_s`). Compromis assume : en cas de crash brutal,
jusqu'a `write_throttle_s` de bougies ne sont pas sur disque (la memoire fait foi).

Reference : Section 1.6 et 4.4 du CDC EINHERJAR.
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

LIVE_DIR = Path(__file__).resolve().parents[3] / "data" / "live"

# Colonnes brutes du store : les features ne sont JAMAIS stockees ici (elles sont
# recalculees a la lecture), sinon le concat casse le schema.
OHLCV_COLUMNS: tuple[str, ...] = ("timestamp", "open", "high", "low", "close", "volume")


class LiveDataStore:
    """Store incremental pour les donnees de marche en temps reel.

    Attributs:
        base_dir: Repertoire racine des fichiers live.
        window_size: Nombre de bougies conservees en memoire par asset/TF.
        _cache: Cache memoire { (asset, tf) -> DataFrame }.
    """

    def __init__(
        self,
        base_dir: str | Path | None = None,
        window_size: int = 500,
        write_throttle_s: float = 15.0,
    ) -> None:
        """Initialise le store.

        Args:
            base_dir: Repertoire des fichiers Parquet. Defaut data/live/.
            window_size: Taille de la fenetre glissante en memoire.
            write_throttle_s: Intervalle minimal entre deux ecritures disque du
                meme couple (0 = ecriture a chaque bougie). Limite le churn Parquet.
        """
        self.base_dir = Path(base_dir) if base_dir else LIVE_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.window_size = window_size
        self.write_throttle_s = write_throttle_s
        self._cache: dict[tuple[str, str], pl.DataFrame] = {}
        self._last_write: dict[tuple[str, str], float] = {}
        self._dirty: set[tuple[str, str]] = set()

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

    def bulk_append(
        self,
        asset: str,
        timeframe: str,
        rows: list[dict[str, Any]] | pl.DataFrame,
    ) -> pl.DataFrame:
        """Ajoute plusieurs bougies en une seule ecriture disque (amorcage historique).

        `append` reecrit le parquet a chaque bougie : pour amorcer 1500 bougies cela
        ferait 1500 ecritures. L'amorcage passe donc par cette methode.

        Args:
            asset: Symbole.
            timeframe: Timeframe.
            rows: Liste de dicts {timestamp, open, high, low, close, volume},
                ou un DataFrame polars equivalent.

        Returns:
            DataFrame mis a jour (fenetre glissante).
        """
        if isinstance(rows, pl.DataFrame):
            rows = rows.to_dicts()
        if not rows:
            return self.get_window(asset, timeframe)
        key = (asset, timeframe)
        df = self._cache.get(key)
        if df is None:
            df = self.load(asset, timeframe)
        new_df = pl.DataFrame(rows)
        if df.height:
            # Aligner les dtypes (les dicts Python peuvent donner des types differents)
            new_df = new_df.with_columns(
                [pl.col(c).cast(df[c].dtype) for c in new_df.columns if c in df.columns
                 and new_df[c].dtype != df[c].dtype]
            )
            df = pl.concat([df, new_df], how="vertical_relaxed")
        else:
            df = new_df
        df = df.unique(subset=["timestamp"], keep="last", maintain_order=True)
        if len(df) > self.window_size:
            df = df.tail(self.window_size)
        self._cache[key] = df
        self._dirty.discard(key)
        self._write(key)
        return df

    def _maybe_write(self, key: tuple[str, str]) -> None:
        """Ecrit le couple si le throttle est ecoule (sinon l'ecriture est differee).

        La PREMIERE ecriture d'un couple est immediate (le fichier existe des le
        demarrage) ; les suivantes respectent `write_throttle_s`.
        """
        dernier = self._last_write.get(key)
        if (
            self.write_throttle_s <= 0
            or dernier is None
            or (time.monotonic() - dernier) >= self.write_throttle_s
        ):
            self._write(key)

    def _write(self, key: tuple[str, str]) -> None:
        """Ecrit la fenetre courante d'un couple sur disque (reecriture complete)."""
        df = self._cache.get(key)
        if df is None:
            return
        df.write_parquet(self._get_path(key[0], key[1]))
        self._last_write[key] = time.monotonic()
        self._dirty.discard(key)

    def flush(self, asset: str | None = None, timeframe: str | None = None) -> int:
        """Ecrit immediatement les fenetres en attente.

        A appeler a l'arret de la boucle (ou avant une lecture disque externe)
        pour ne rien perdre des bougies accumulees depuis la derniere sauvegarde.

        Args:
            asset: Limiter le flush a un actif (None = tous).
            timeframe: Limiter le flush a un timeframe (None = tous).

        Returns:
            Nombre de fichiers ecrits.
        """
        cibles = [
            key for key in self._dirty
            if (asset is None or key[0] == asset) and (timeframe is None or key[1] == timeframe)
        ]
        for key in cibles:
            self._write(key)
        return len(cibles)

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

        # Construire la nouvelle ligne. Le filtre est base sur le SCHEMA du store
        # (pas sur `df.height` : un store vide a deja ses colonnes) — une ligne
        # enrichie de features ne doit jamais casser le schema OHLCV, sinon
        # pl.concat leve "schema lengths differ".
        colonnes = set(df.columns) or set(OHLCV_COLUMNS)
        row = {k: v for k, v in candle.items() if k in colonnes}
        if features:
            row.update({k: v for k, v in features.items() if k in colonnes})
        if not row:
            raise ValueError(f"Aucune colonne commune avec le store {asset}/{timeframe}")

        new_df = pl.DataFrame([row])
        if df.height and any(new_df[c].dtype != df[c].dtype for c in new_df.columns):
            new_df = new_df.with_columns(
                [pl.col(c).cast(df[c].dtype) for c in new_df.columns if new_df[c].dtype != df[c].dtype]
            )

        # Concatener et tronquer a la fenetre
        df = pl.concat([df, new_df], how="vertical_relaxed")
        if len(df) > self.window_size:
            df = df.tail(self.window_size)

        # Mettre a jour le cache
        self._cache[key] = df

        # Ecriture disque throttle (polars reecrit tout le fichier) : la memoire
        # fait foi entre deux sauvegardes, `flush()` force l'ecriture.
        self._dirty.add(key)
        self._maybe_write(key)

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
