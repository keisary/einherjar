"""Protocole BrokerAdapter — interface unique pour tous les brokers.

Chaque implementation fournit : historique OHLCV, flux live,
passage d'ordres, etat du compte, frais.

Reference : Section 1.3 et 4.1 du CDC EINHERJAR.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import polars as pl

from einherjar.core.models import AccountState, Fill, Order, Position


@runtime_checkable
class BrokerAdapter(Protocol):
    """Interface que doit implementer tout adaptateur broker.

    Exemples concrets : BinanceAdapter, AlpacaAdapter, OandaAdapter,
    PaperBroker.
    """

    name: str

    async def get_ohlcv(
        self,
        asset: str,
        timeframe: str,
        since: int | None = None,
        limit: int = 500,
    ) -> pl.DataFrame:
        """Recupere l'historique OHLCV d'un actif sur un timeframe.

        Args:
            asset: Symbole normalise (ex "BTC/USDT").
            timeframe: Timeframe ("5m", "15m", "1h", "4h", "1d").
            since: Timestamp Unix ms de depart. None = depuis le debut disponible.
            limit: Nombre de bougies max par appel.

        Returns:
            DataFrame polars avec colonnes [timestamp, open, high, low, close, volume].
        """
        ...

    async def subscribe_live(
        self,
        assets: list[str],
        callback: callable,
    ) -> None:
        """Souscrit aux flux de prix temps reel.

        Args:
            assets: Liste des symboles.
            callback: Fonction appelee a chaque tick (symbole, prix, timestamp).
        """
        ...

    async def place_order(self, order: Order) -> Fill:
        """Passe un ordre sur le broker.

        Args:
            order: Intention d'ordre dimensionnee par le Risk Manager.

        Returns:
            Fill initial (peut etre partiel).
        """
        ...

    async def cancel_order(self, order_id: str) -> bool:
        """Annule un ordre ouvert.

        Args:
            order_id: Identifiant de l'ordre.

        Returns:
            True si annule avec succes.
        """
        ...

    async def get_positions(self) -> list[Position]:
        """Retourne les positions ouvertes sur le broker.

        Returns:
            Liste des positions actives.
        """
        ...

    async def get_account(self) -> AccountState:
        """Retourne l'etat du compte.

        Returns:
            Cash, equity, marge utilisee/disponible.
        """
        ...

    def get_fees(self, asset: str) -> dict:
        """Retourne la structure de frais pour un actif.

        Args:
            asset: Symbole.

        Returns:
            Dictionnaire {maker, taker, slippage, etc.} charge depuis fees_{broker}.json.
        """
        ...
