# pyright: reportMissingImports=false, reportAttributeAccessIssue=false
"""CTraderAdapter — Interface unique cTrader Open API (cloud-native).

Utilise la librairie officielle `ctrader-open-api` (Spotware) qui s'appuie
sur Twisted + Protobuf. Le reactor Twisted tourne dans un thread dédié ;
toutes les methodes publiques sont async et utilisent `asyncio.to_thread()`
pour communiquer avec le thread Twisted.

Resilience integree : circuit breaker + rate limiter (composition de
resilience.py) + reconnexion automatique.

Reference : docs/PLAN_REFONTE_CTRADER.md
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from concurrent.futures import Future as ConcurrentFuture
from datetime import UTC, datetime
from typing import Any

import polars as pl

from einherjar.brokers.broker_utils import (  # noqa: F401
    denormalize_symbol,
    load_fees,
    normalize_symbol,
    now_utc_ms,
    ohlcv_to_polars,
    timeframe_to_ctrader_period,
)
from einherjar.brokers.resilience import CircuitBreaker, RateLimiter
from einherjar.core.enums import AssetClass, Direction
from einherjar.core.models import AccountState, Fill, Order, Position

logger = logging.getLogger("einherjar.ctrader")

# ---------------------------------------------------------------------------
# Detection librairie cTrader
# ---------------------------------------------------------------------------
try:
    from ctrader_open_api import Client, EndPoints, Protobuf, TcpProtocol  # noqa: F401
    from ctrader_open_api.messages.OpenApiCommonMessages_pb2 import (  # noqa: F401
        ProtoOAErrorRes,  # noqa: F401
    )
    from ctrader_open_api.messages.OpenApiMessages_pb2 import (  # noqa: F401
        ProtoOAAccountAuthReq,
        ProtoOAAccountAuthRes,  # noqa: F401
        ProtoOAApplicationAuthReq,
        ProtoOAApplicationAuthRes,  # noqa: F401
        ProtoOAClosePositionReq,
        ProtoOAExecutionEvent,
        ProtoOAGetAccountListReq,
        ProtoOAGetAccountListRes,
        ProtoOAGetPositionListReq,
        ProtoOAGetPositionListRes,
        ProtoOANewOrderReq,
        ProtoOASymbolsListReq,
        ProtoOASymbolsListRes,
        ProtoOATrendbarReq,
        ProtoOATrendbarRes,
    )

    CTRADER_AVAILABLE = True
except ImportError as _imp_err:
    CTRADER_AVAILABLE = False
    CTRADER_IMPORT_ERROR = str(_imp_err)
    logger.warning("ctrader-open-api non installe. CTraderAdapter fonctionnera en mode stub.")


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class CTraderError(RuntimeError):
    """Erreur specifique cTrader."""

    pass


# ---------------------------------------------------------------------------
# Thread Twisted interne
# ---------------------------------------------------------------------------

class _CTraderTwistedThread:
    """Client Twisted cTrader qui tourne dans un thread dédié.

    Toutes les methodes publiques de cette classe sont synchrones (bloquantes)
    car elles attendent une reponse du reactor Twisted via ConcurrentFuture.
    """

    def __init__(
        self,
        host: str,
        port: int,
        client_id: str,
        client_secret: str,
        access_token: str,
        account_id: int,
        broker_name: str,
    ) -> None:
        self.host = host
        self.port = port
        self.client_id = client_id
        self.client_secret = client_secret
        self.access_token = access_token
        self.account_id = account_id
        self.broker_name = broker_name

        self._client: Any | None = None
        self._reactor: Any | None = None
        self._thread: threading.Thread | None = None
        self._connected_event = threading.Event()
        self._pending: dict[str, tuple[type, ConcurrentFuture]] = {}
        self._symbol_cache: dict[str, int] = {}
        self._symbol_meta: dict[int, dict[str, Any]] = {}
        self._shutdown = False

    # -- Cycle de vie -------------------------------------------------------

    def start(self, timeout: float = 15.0) -> None:
        """Demarre le reactor Twisted dans un thread daemon."""
        if not CTRADER_AVAILABLE:
            raise CTraderError(f"ctrader-open-api manquant : {CTRADER_IMPORT_ERROR}")
        self._thread = threading.Thread(target=self._run, daemon=True, name="CTraderTwisted")
        self._thread.start()
        if not self._connected_event.wait(timeout=timeout):
            raise CTraderError(f"Timeout connexion cTrader ({timeout}s)")
        logger.info("CTraderTwistedThread connecte (account_id=%s)", self.account_id)

    def stop(self) -> None:
        """Arrete proprement le reactor."""
        self._shutdown = True
        self._connected_event.clear()
        if self._reactor is not None:
            self._reactor.callFromThread(self._reactor.stop)
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def _run(self) -> None:
        from twisted.internet import reactor

        self._reactor = reactor
        self._client = Client(self.host, self.port, TcpProtocol)
        self._client.setConnectedCallback(self._on_connected)
        self._client.setDisconnectedCallback(self._on_disconnected)
        self._client.setMessageReceivedCallback(self._on_message)
        self._client.startService()
        reactor.run(installSignalHandlers=0)

    # -- Callbacks Twisted --------------------------------------------------

    def _on_connected(self, client: Any) -> None:
        logger.debug("CTrader socket connected")
        req = ProtoOAApplicationAuthReq()
        req.clientId = self.client_id
        req.clientSecret = self.client_secret
        d = client.send(req)
        d.addCallback(self._on_app_auth_ok)
        d.addErrback(self._on_error)

    def _on_disconnected(self, client: Any, reason: Any) -> None:
        logger.warning("CTrader deconnecte : %s", reason)
        self._connected_event.clear()

    def _on_app_auth_ok(self, _msg: Any) -> None:
        req = ProtoOAAccountAuthReq()
        req.ctidTraderAccountId = self.account_id
        req.accessToken = self.access_token
        d = self._client.send(req)  # pyright: ignore[reportOptionalMemberAccess]
        d.addCallback(self._on_account_auth_ok)
        d.addErrback(self._on_error)

    def _on_account_auth_ok(self, _msg: Any) -> None:
        logger.info("CTrader authentification account OK")
        self._connected_event.set()
        # Pre-charger la liste des symboles pour resoudre les IDs
        self._preload_symbols()

    def _on_error(self, failure: Any) -> None:
        logger.error("CTrader erreur Twisted : %s", failure)

    def _on_message(self, _client: Any, msg_wrapper: Any) -> None:
        msg = Protobuf.extract(msg_wrapper)
        req_id = getattr(msg, "clientMsgId", None)
        if req_id and req_id in self._pending:
            _expected_type, future = self._pending.pop(req_id)
            future.set_result(msg)
            return
        # Messages spontanes (market data, execution events) — ignore pour l'instant
        logger.debug("CTrader message spontane type=%s", type(msg).__name__)

    # -- Internes -----------------------------------------------------------

    def _send_request(self, request: Any, response_type: type) -> ConcurrentFuture:
        """Envoie une requete protobuf et retourne un Future bloquant."""
        if not self._connected_event.is_set():
            raise CTraderError("Non connecte")
        future: ConcurrentFuture = ConcurrentFuture()
        req_id = str(uuid.uuid4())
        request.clientMsgId = req_id
        self._pending[req_id] = (response_type, future)
        self._reactor.callFromThread(self._client.send, request)  # pyright: ignore[reportOptionalMemberAccess]
        return future

    def _preload_symbols(self) -> None:
        """Charge les metadonnees des symboles disponibles."""
        try:
            req = ProtoOASymbolsListReq()
            req.ctidTraderAccountId = self.account_id
            deferred = self._client.send(req)  # pyright: ignore[reportOptionalMemberAccess]
            deferred.addCallback(self._cache_symbols)
            deferred.addErrback(self._on_error)
        except Exception as exc:
            logger.warning("Preload symbols echoue : %s", exc)

    def _cache_symbols(self, message: Any) -> None:
        """Construit le cache symbole cTrader vers ID reel."""
        try:
            response = Protobuf.extract(message)
        except Exception:
            response = message
        if not isinstance(response, ProtoOASymbolsListRes):
            logger.warning("Reponse symboles inattendue: %s", type(response).__name__)
            return
        for symbol in getattr(response, "symbol", []):
            symbol_id = int(getattr(symbol, "symbolId", 0))
            symbol_name = str(getattr(symbol, "symbolName", ""))
            if symbol_id <= 0 or not symbol_name:
                continue
            self._symbol_cache[symbol_name.upper()] = symbol_id
            self._symbol_meta[symbol_id] = {"name": symbol_name}
        logger.info("cTrader: %d symboles reels charges", len(self._symbol_meta))

    def _resolve_symbol_sync(self, asset: str) -> int:
        """Retourne le symbolId cTrader pour un asset MIDAS (avec cache)."""
        if asset in self._symbol_cache:
            return self._symbol_cache[asset]
        symbol_name = normalize_symbol(asset, self.broker_name)
        # Tentative de resolution via ProtoOASymbolByIdReq si on connait l'ID
        # Sinon on essaye de chercher par nom — cela necessite ProtoOAGetSymbolsReq
        # Pour l'instant on simule une resolution par defaut basee sur un hash
        # (a remplacer par une vraie requete API quand les stubs sont confirmes)
        symbol_id = self._symbol_cache.get(symbol_name.upper())
        if symbol_id is None:
            raise CTraderError(f"Symbole indisponible chez le broker: {asset} ({symbol_name})")
        self._symbol_cache[asset] = symbol_id
        logger.debug("Symbol resolu %s -> %s (id=%d)", asset, symbol_name, symbol_id)
        return symbol_id

    # -- API synchrones (appelees via asyncio.to_thread) --------------------

    def get_ohlcv_sync(self, asset: str, timeframe: str, limit: int = 500) -> pl.DataFrame:
        """Recupere les trendbars (candles) cTrader."""
        symbol_id = self._resolve_symbol_sync(asset)
        period = timeframe_to_ctrader_period(timeframe)
        now_ms = now_utc_ms()
        # cTrader TrendbarReq : from/to en milliseconds
        # On demande un historique suffisant pour obtenir `limit` bougies
        bar_duration_ms = period * 60_000
        from_ms = now_ms - (limit * bar_duration_ms * 2)

        req = ProtoOATrendbarReq()
        req.ctidTraderAccountId = self.account_id
        req.symbolId = symbol_id
        req.period = period
        req.fromTimestamp = from_ms
        req.toTimestamp = now_ms

        future = self._send_request(req, ProtoOATrendbarRes)
        try:
            res = future.result(timeout=10.0)
        except Exception as exc:
            raise CTraderError(f"get_ohlcv timeout/error : {exc}") from exc

        bars = getattr(res, "trendbar", [])
        if not bars:
            return ohlcv_to_polars([])

        rows = []
        # Les trendbars cTrader encodent les prix en ticks ;
        # la conversion exacte necessite le digit du symbole.
        # On suppose ici que les champs deltaHigh/deltaLow/deltaOpen/volume sont dispo.
        for bar in bars:
            ts = getattr(bar, "utcTimestampInMinutes", 0) * 60_000
            # Prix en points — approximation : on prend close comme reference
            close = getattr(bar, "close", 0)
            # Si les deltas existent :
            high = close + getattr(bar, "deltaHigh", 0)
            low = close - getattr(bar, "deltaLow", 0)
            open_p = close - getattr(bar, "deltaOpen", 0)
            vol = getattr(bar, "volume", 0)
            rows.append([ts, open_p, high, low, close, vol])

        return ohlcv_to_polars(rows[-limit:])

    def place_order_sync(self, order: Order) -> Fill:
        """Passe un ordre de marche sur cTrader."""
        symbol_id = self._resolve_symbol_sync(order.asset)
        req = ProtoOANewOrderReq()
        req.ctidTraderAccountId = self.account_id
        req.symbolId = symbol_id
        req.orderType = 1 if order.order_type.value == "MARKET" else 2  # MARKET=1, LIMIT=2
        req.tradeSide = 1 if order.direction == Direction.LONG else 2
        req.volume = int(order.quantity * 100)  # cTrader utilise parfois des centi-lots
        if order.sl_price is not None:
            req.stopLoss = order.sl_price
        if order.tp_price is not None:
            req.takeProfit = order.tp_price

        future = self._send_request(req, ProtoOAExecutionEvent)
        try:
            res = future.result(timeout=10.0)
        except Exception as exc:
            raise CTraderError(f"place_order timeout/error : {exc}") from exc

        # ProtoOAExecutionEvent contient le fill
        position = getattr(res, "position", None)
        fill_qty = getattr(position, "volume", 0) / 100.0 if position else order.quantity
        fill_price = getattr(position, "tradeData.price", order.entry_price or 0)

        return Fill(
            fill_id=f"FIL_{uuid.uuid4().hex}",
            order_id=order.order_id,
            asset=order.asset,
            filled_qty=fill_qty,
            filled_price=float(fill_price),
            fee=0.0,
            timestamp=datetime.now(UTC),
        )

    def get_positions_sync(self) -> list[Position]:
        """Retourne les positions ouvertes."""
        req = ProtoOAGetPositionListReq()
        req.ctidTraderAccountId = self.account_id
        future = self._send_request(req, ProtoOAGetPositionListRes)
        try:
            res = future.result(timeout=10.0)
        except Exception as exc:
            raise CTraderError(f"get_positions timeout/error : {exc}") from exc

        positions = []
        for p in getattr(res, "position", []):
            symbol_id = int(getattr(getattr(p, "tradeData", None), "symbolId", 0))
            broker_symbol = self._symbol_meta.get(symbol_id, {}).get("name", str(symbol_id))
            asset = denormalize_symbol(broker_symbol, self.broker_name)
            positions.append(
                Position(
                    position_id=str(getattr(p, "positionId", 0)),
                    asset=asset,
                    direction=Direction.LONG if getattr(p, "tradeSide", 1) == 1 else Direction.SHORT,
                    quantity=getattr(p, "volume", 0) / 100.0,
                    avg_entry_price=float(getattr(getattr(p, "tradeData", None), "price", 0)),
                    opened_at=datetime.now(UTC),
                    asset_class=AssetClass.CRYPTO,
                )
            )
        return positions

    def get_account_sync(self) -> AccountState:
        """Retourne l'etat du compte."""
        req = ProtoOAGetAccountListReq()
        req.ctidTraderAccountId = self.account_id
        future = self._send_request(req, ProtoOAGetAccountListRes)
        try:
            res = future.result(timeout=10.0)
        except Exception as exc:
            raise CTraderError(f"get_account timeout/error : {exc}") from exc

        # La reponse contient une liste de comptes ; on prend le premier correspondant
        account = None
        for acc in getattr(res, "tradingAccount", []):
            if getattr(acc, "ctidTraderAccountId", 0) == self.account_id:
                account = acc
                break
        if account is None:
            raise CTraderError("Compte introuvable dans la reponse")

        return AccountState(
            cash=float(getattr(account, "balance", 0)),
            equity=float(getattr(account, "equity", 0)),
            margin_used=float(getattr(account, "marginUsed", 0)),
            margin_available=float(getattr(account, "margin", 0)),
            leverage=int(getattr(account, "leverage", 1)),
        )

    def close_position_sync(self, position_id: int) -> bool:
        """Ferme une position par son ID."""
        req = ProtoOAClosePositionReq()
        req.ctidTraderAccountId = self.account_id
        req.positionId = position_id
        future = self._send_request(req, ProtoOAExecutionEvent)
        try:
            future.result(timeout=10.0)
            return True
        except Exception as exc:
            logger.error("close_position echoue : %s", exc)
            return False


# ---------------------------------------------------------------------------
# Adapter public async
# ---------------------------------------------------------------------------

class CTraderAdapter:
    """Adapter cloud-native cTrader Open API.

    Encapsule le thread Twisted, ajoute circuit breaker + rate limiter,
    et expose une interface async compatible BrokerAdapter.
    """

    name = "ctrader"

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        access_token: str,
        account_id: int,
        host: str = "demo.ctraderapi.com",
        port: int = 5035,
        broker_name: str = "ic_markets",
    ) -> None:
        """Initialise l'adapter cTrader.

        Args:
            client_id: Client ID de l'application cTrader (54 chars).
            client_secret: Client Secret de l'application (50 chars).
            access_token: Access token OAuth2 (43 chars).
            account_id: ctidTraderAccountId (entier).
            host: Host gRPC (demo.ctraderapi.com ou live).
            port: Port gRPC (defaut 5035).
            broker_name: "ic_markets" ou "pepperstone" (pour le mapping symboles).
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.access_token = access_token
        self.account_id = account_id
        self.host = host
        self.port = port
        self.broker_name = broker_name

        self._twisted: _CTraderTwistedThread | None = None
        self.circuit = CircuitBreaker()
        self.rate_limiter = RateLimiter()
        self._connected = False
        self._last_ping = 0.0

        self._fees = load_fees("ctrader")

    # -- Connexion ----------------------------------------------------------

    async def connect(self) -> bool:
        """Etablit la connexion gRPC + auth."""
        if not CTRADER_AVAILABLE:
            logger.error("ctrader-open-api n'est pas installe. Executez : pip install ctrader-open-api")
            return False
        try:
            self._twisted = _CTraderTwistedThread(
                self.host, self.port,
                self.client_id, self.client_secret,
                self.access_token, self.account_id,
                self.broker_name,
            )
            await asyncio.to_thread(self._twisted.start)
            self._connected = True
            self._last_ping = time.time()
            return True
        except Exception as exc:
            logger.error("Connexion cTrader echoue : %s", exc)
            self._connected = False
            return False

    async def disconnect(self) -> None:
        """Ferme la connexion."""
        if self._twisted is not None:
            self._twisted.stop()
        self._connected = False

    async def ensure_connected(self) -> bool:
        """Verifie / restaure la connexion si necessaire."""
        if self._connected and (time.time() - self._last_ping) < 30:
            return True
        logger.info("Reconnexion cTrader necessaire...")
        return await self.connect()

    # -- Resilience wrapper -------------------------------------------------

    async def _safe_call(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        """Execute un appel avec circuit breaker + rate limiter + reconnexion."""
        if not self.circuit.can_execute():
            raise RuntimeError("Circuit breaker ouvert pour cTrader")
        await self.rate_limiter.acquire()
        if not self._connected or (time.time() - self._last_ping) > 30:
            ok = await self.ensure_connected()
            if not ok:
                raise RuntimeError("Impossible de reconnecter cTrader")
        try:
            method = getattr(self._twisted, f"{method_name}_sync")
            result = await asyncio.to_thread(method, *args, **kwargs)
            self.circuit.record_success()
            self._last_ping = time.time()
            return result
        except Exception:
            self.circuit.record_failure()
            raise

    # -- BrokerAdapter interface --------------------------------------------

    async def get_ohlcv(
        self,
        asset: str,
        timeframe: str,
        since: int | None = None,
        limit: int = 500,
    ) -> pl.DataFrame:
        """Recupere l'historique OHLCV via Trendbars cTrader."""
        return await self._safe_call("get_ohlcv", asset, timeframe, limit)

    async def subscribe_live(self, assets: list[str], callback: callable) -> None:  # pyright: ignore[reportGeneralTypeIssues]
        """Souscription live — non implemente pour cTrader Open API (pas de streaming temps reel dans cette version)."""
        logger.warning("subscribe_live non implemente pour cTrader")

    async def place_order(self, order: Order) -> Fill:
        """Passe un ordre marche avec SL/TP inline."""
        return await self._safe_call("place_order", order)

    async def cancel_order(self, order_id: str) -> bool:
        """Annulation d'ordre — non supporte directement ; on ferme la position si necessaire."""
        logger.warning("cancel_order non implemente pour cTrader (utiliser close_position)")
        return False

    async def get_positions(self) -> list[Position]:
        """Positions ouvertes."""
        return await self._safe_call("get_positions")

    async def get_account(self) -> AccountState:
        """Etat du compte (balance, equity, margin, leverage)."""
        return await self._safe_call("get_account")

    def get_fees(self, asset: str) -> dict[str, Any]:
        """Frais pour un actif."""
        symbol = normalize_symbol(asset, self.broker_name)
        per_symbol = self._fees.get("per_symbol", {})
        return per_symbol.get(symbol, self._fees.get("default", {}))

    def get_status(self) -> dict[str, Any]:
        """Statut de l'adapter (connexion, circuit breaker, rate limiter)."""
        return {
            "broker": self.name,
            "connected": self._connected,
            "host": f"{self.host}:{self.port}",
            "account_id": self.account_id,
            "circuit_state": self.circuit.state,
            "circuit_failures": self.circuit.failures,
            "rate_second_calls": len(self.rate_limiter.second_calls),
            "rate_minute_calls": len(self.rate_limiter.minute_calls),
        }

    # -- Methodes supplementaires -------------------------------------------

    async def close_position(self, position_id: int) -> bool:
        """Ferme une position par son ID cTrader."""
        return await self._safe_call("close_position", position_id)
