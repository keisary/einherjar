"""Gestionnaire de donnees OHLCV — fetch, stockage incremental, fraicheur.

Interface entre les brokers et le store local. Priorite :
1. Donnees deja en store DuckDB.
2. Complement via API broker (CCXT) si trous.

Reference : Section 1.6 du CDC EINHERJAR.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import polars as pl

from einherjar.brokers.adapter import BrokerAdapter
from einherjar.data.store import DataStore


class OHLCVManager:
    """Orchestre le flux de donnees de marche.

    Attributs:
        store: DataStore DuckDB local.
        broker: Adaptateur broker pour les fetch externes.
        freshness_threshold: Seuil en secondes avant qu'une serie soit consideree
            comme perimee.
    """

    def __init__(
        self,
        store: DataStore,
        broker: BrokerAdapter,
        freshness_threshold: int = 300,
    ) -> None:
        """Initialise le manager.

        Args:
            store: Store local DuckDB.
            broker: Broker pour fetch externe.
            freshness_threshold: Duree en secondes. Defaut 5 minutes.
        """
        self.store = store
        self.broker = broker
        self.freshness_threshold = freshness_threshold

    async def get(
        self,
        asset: str,
        timeframe: str,
        since: datetime | None = None,
        limit: int = 500,
    ) -> pl.DataFrame:
        """Recupere l'historique OHLCV, fetch si necessaire.

        Args:
            asset: Symbole.
            timeframe: Timeframe.
            since: Date de depart minimale.
            limit: Nombre de bougies.

        Returns:
            DataFrame polars [timestamp, open, high, low, close, volume].
        """
        local = self.store.query_ohlcv(
            asset=asset,
            timeframe=timeframe,
            since=since.isoformat() if since else None,
            limit=limit,
        )

        if len(local) < limit:
            missing = limit - len(local)
            remote = await self.broker.get_ohlcv(
                asset=asset,
                timeframe=timeframe,
                limit=missing,
            )
            if len(remote) > 0:
                remote = remote.with_columns(
                    pl.lit(asset).alias("asset"),
                    pl.lit(timeframe).alias("timeframe"),
                )
                self.store.append_ohlcv(remote)
                local = self.store.query_ohlcv(
                    asset=asset,
                    timeframe=timeframe,
                    since=since.isoformat() if since else None,
                    limit=limit,
                )

        return local

    async def append_candle(
        self,
        asset: str,
        timeframe: str,
        candle: dict[str, Any],
    ) -> None:
        """Ajoute une bougie clôturee au store.

        Args:
            asset: Symbole.
            timeframe: Timeframe.
            candle: Dict {timestamp, open, high, low, close, volume}.
        """
        df = pl.DataFrame(
            {
                "asset": [asset],
                "timeframe": [timeframe],
                "timestamp": [candle["timestamp"]],
                "open": [candle["open"]],
                "high": [candle["high"]],
                "low": [candle["low"]],
                "close": [candle["close"]],
                "volume": [candle["volume"]],
            }
        )
        self.store.append_ohlcv(df)

    def is_fresh(self, asset: str, timeframe: str) -> bool:
        """Verifie si les dernieres donnees sont fraiches.

        Args:
            asset: Symbole.
            timeframe: Timeframe.

        Returns:
            True si derniere bougie < threshold secondes.
        """
        latest = self.store.query_ohlcv(asset, timeframe, limit=1)
        if len(latest) == 0:
            return False
        last_ts = latest["timestamp"][0]
        if isinstance(last_ts, str):
            last_ts = datetime.fromisoformat(last_ts)
        return (datetime.now(UTC) - last_ts).total_seconds() < self.freshness_threshold
