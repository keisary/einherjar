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
        ProtoHeartbeatEvent,  # noqa: F401
    )
    from ctrader_open_api.messages.OpenApiMessages_pb2 import (  # noqa: F401
        ProtoOAAccountAuthReq,
        ProtoOAAccountAuthRes,  # noqa: F401
        ProtoOAApplicationAuthReq,
        ProtoOAApplicationAuthRes,  # noqa: F401
        ProtoOAClosePositionReq,
        ProtoOAErrorRes,  # noqa: F401
        ProtoOAExecutionEvent,
        ProtoOAGetAccountListByAccessTokenReq,
        ProtoOAGetAccountListByAccessTokenRes,
        ProtoOAGetTrendbarsReq,
        ProtoOAGetTrendbarsRes,
        ProtoOANewOrderReq,
        ProtoOAReconcileReq,
        ProtoOAReconcileRes,
        ProtoOASymbolByIdReq,
        ProtoOASymbolByIdRes,
        ProtoOASymbolsListReq,
        ProtoOASymbolsListRes,
        ProtoOATraderReq,
        ProtoOATraderRes,
    )

    CTRADER_AVAILABLE = True
    CTRADER_IMPORT_ERROR = ""
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
# Conversion des trendbars (fonction pure, testable sans API)
# ---------------------------------------------------------------------------
def trendbars_to_ohlcv(bars: list[Any], digits: int, limit: int | None = None) -> pl.DataFrame:
    """Convertit des `ProtoOATrendbar` en OHLCV exploitable.

    Format cTrader Open API : pour une trendbar, seuls `low`, `deltaOpen`,
    `deltaClose`, `deltaHigh` (et `volume`) sont transmis. Ces valeurs sont des
    ENTIERS en POINTS, exprimes par rapport au close de la bougie PRECEDENTE, et
    doivent etre divises par `10 ** digits` pour obtenir des prix :

        open  = close_precedent + deltaOpen  / 10**digits
        close = close_precedent + deltaClose / 10**digits
        high  = low                        + deltaHigh  / 10**digits
        low   = low                        / 10**digits

    La PREMIERE bougie servie ne peut donc pas etre reconstruite (pas de close
    precedent) : elle est ecartee. L'appelant demande `limit + 1` bougies.

    Args:
        bars: Trendbars brutes de l'API (objets protobuf ou equivalents).
        digits: Nombre de decimales du symbole (`ProtoOASymbol.digits`).
        limit: Ne garder que les `limit` dernieres bougies.

    Returns:
        DataFrame polars [timestamp, open, high, low, close, volume].

    Raises:
        CTraderError: si `digits` est invalide (prix non interpretables).
    """
    if digits is None or int(digits) < 0:
        raise CTraderError(
            "digits du symbole inconnu : les prix des trendbars ne sont pas "
            "interpretables (appeler ProtoOASymbolByIdReq avant get_ohlcv)"
        )
    echelle = float(10 ** int(digits))
    rows: list[list[float]] = []
    close_precedent: float | None = None
    for bar in bars:
        low = float(getattr(bar, "low", 0.0)) / echelle
        delta_open = float(getattr(bar, "deltaOpen", 0.0)) / echelle
        delta_close = float(getattr(bar, "deltaClose", 0.0)) / echelle
        delta_high = float(getattr(bar, "deltaHigh", 0.0)) / echelle
        ts = int(getattr(bar, "utcTimestampInMinutes", 0)) * 60_000
        if close_precedent is None:
            # Pas de reference : bougie de chauffe, ecartee (open = low + deltaOpen
            # serait une approximation, on ne la publie pas).
            close_precedent = low + delta_close
            continue
        open_p = close_precedent + delta_open
        close = close_precedent + delta_close
        high = low + delta_high
        rows.append([ts, open_p, high, low, close, float(getattr(bar, "volume", 0.0))])
        close_precedent = close
    return ohlcv_to_polars(rows[-limit:] if limit else rows)


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
        # Resolution par le cache alimente par ProtoOASymbolsListRes (symbolName ->
        # symbolId) : on refuse de deviner un symbolId, un ordre sur le mauvais
        # symbole serait pire qu'une erreur explicite.
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

        req = ProtoOAGetTrendbarsReq()
        req.ctidTraderAccountId = self.account_id
        req.symbolId = symbol_id
        req.period = period
        req.fromTimestamp = from_ms
        req.toTimestamp = now_ms
        # Une bougie de plus : la premiere est ecartee (pas de close de reference).
        req.count = limit + 1

        future = self._send_request(req, ProtoOAGetTrendbarsRes)
        try:
            res = future.result(timeout=10.0)
        except Exception as exc:
            raise CTraderError(f"get_ohlcv timeout/error : {exc}") from exc

        bars = getattr(res, "trendbar", [])
        if not bars:
            return ohlcv_to_polars([])

        # `trendbars_to_ohlcv` ecarte la premiere bougie (elle n'a pas de close de
        # reference) : la fenetre demandee ci-dessus est volontairement large.
        digits = self._digits_for_symbol(symbol_id)
        return trendbars_to_ohlcv(bars, digits, limit=limit)

    def _digits_for_symbol(self, symbol_id: int) -> int:
        """Retourne le nombre de decimales du symbole (requete SymbolById si besoin).

        Les prix des trendbars sont des entiers en points : sans `digits` ils ne
        sont pas interpretables, donc on refuse de continuer plutot que de
        retourner des prix faux.
        """
        digits = self._symbol_meta.get(symbol_id, {}).get("digits")
        if digits is None:
            digits = self._load_symbol_details_sync(symbol_id)
        if digits is None:
            raise CTraderError(
                f"digits indisponible pour symbolId={symbol_id} : impossible de "
                "convertir les trendbars en prix"
            )
        return int(digits)

    def _load_symbol_details_sync(self, symbol_id: int) -> int | None:
        """Charge les details d'un symbole (digits, volumes, lot) via SymbolByIdReq.

        Returns:
            Le nombre de decimales du symbole, ou None si l'API ne le fournit pas.
        """
        req = ProtoOASymbolByIdReq()
        req.ctidTraderAccountId = self.account_id
        req.symbolId.append(symbol_id)
        try:
            future = self._send_request(req, ProtoOASymbolByIdRes)
            res = future.result(timeout=10.0)
        except Exception as exc:  # noqa: BLE001 - remonte tel quel a l'appelant
            raise CTraderError(f"ProtoOASymbolByIdReq echoue pour {symbol_id}: {exc}") from exc

        symbol = next(iter(getattr(res, "symbol", []) or []), None)
        if symbol is None:
            return None
        meta = self._symbol_meta.setdefault(symbol_id, {})
        meta.setdefault("name", str(getattr(symbol, "symbolName", symbol_id)))
        for champ in ("digits", "pipPosition", "minVolume", "stepVolume", "maxVolume", "lotSize"):
            valeur = getattr(symbol, champ, None)
            if valeur is not None:
                meta[champ] = valeur
        logger.info("cTrader: details symbole %s -> %s", symbol_id, meta)
        return meta.get("digits")

    def place_order_sync(self, order: Order) -> Fill:
        """Passe un ordre de marche sur cTrader (TP/SL transmis au broker)."""
        symbol_id = self._resolve_symbol_sync(order.asset)
        meta = self._symbol_meta.get(symbol_id, {})
        # cTrader compte le volume en 0,01 unite de l'actif de base.
        volume = int(round(order.quantity * 100))
        min_volume = meta.get("minVolume")
        step_volume = meta.get("stepVolume")
        if step_volume:
            volume = max(int(step_volume), (volume // int(step_volume)) * int(step_volume))
        if min_volume and volume < int(min_volume):
            raise CTraderError(
                f"volume {volume} (centiemes) sous le minimum broker {min_volume} pour "
                f"{order.asset} : ordre refuse cote client"
            )
        req = ProtoOANewOrderReq()
        req.ctidTraderAccountId = self.account_id
        req.symbolId = symbol_id
        req.orderType = 1 if order.order_type.value == "MARKET" else 2  # MARKET=1, LIMIT=2
        req.tradeSide = 1 if order.direction == Direction.LONG else 2
        req.volume = volume
        # Le label transporte le nom de l'einher : il revient avec la position
        # (tradeData.label) et permet la sortie sur duree de tenue.
        req.label = str(getattr(order, "einher_name", "") or "")[:100]
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
        # Prix d'execution : ProtoOAExecutionEvent ne porte pas de champ `price` ;
        # le prix reel du fill est celui de la position ouverte (ProtoOAPosition.price),
        # a defaut le prix theorique de l'ordre.
        trade_data = getattr(position, "tradeData", None)
        fill_price = (
            getattr(position, "price", None)
            or getattr(res, "price", None)
            or getattr(trade_data, "price", None)
            or order.entry_price
            or 0.0
        )

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
        """Retourne les positions ouvertes (ProtoOAReconcileRes).

        Champs reels : `tradeData` porte symbolId/volume/tradeSide/openTimestamp/
        label, et le prix d'entree est `ProtoOAPosition.price` (il n'existe pas de
        `tradeData.price`).
        """
        req = ProtoOAReconcileReq()
        req.ctidTraderAccountId = self.account_id
        future = self._send_request(req, ProtoOAReconcileRes)
        try:
            res = future.result(timeout=10.0)
        except Exception as exc:
            raise CTraderError(f"get_positions timeout/error : {exc}") from exc

        positions = []
        for p in getattr(res, "position", []):
            trade = getattr(p, "tradeData", None)
            symbol_id = int(getattr(trade, "symbolId", 0))
            broker_symbol = self._symbol_meta.get(symbol_id, {}).get("name", str(symbol_id))
            asset = denormalize_symbol(broker_symbol, self.broker_name)
            ouvert_ms = int(getattr(trade, "openTimestamp", 0) or 0)
            positions.append(
                Position(
                    position_id=str(getattr(p, "positionId", 0)),
                    asset=asset,
                    direction=(
                        Direction.LONG if int(getattr(trade, "tradeSide", 1)) == 1 else Direction.SHORT
                    ),
                    quantity=float(getattr(trade, "volume", 0)) / 100.0,
                    avg_entry_price=float(getattr(p, "price", 0)),
                    tp_price=float(getattr(p, "takeProfit", 0)) or None,
                    sl_price=float(getattr(p, "stopLoss", 0)) or None,
                    # Le label porte le nom de l'einher emetteur (pose a l'ouverture) :
                    # la gestion des sorties peut ainsi appliquer la duree de tenue.
                    einher_name=str(getattr(trade, "label", "") or ""),
                    opened_at=(
                        datetime.fromtimestamp(ouvert_ms / 1000, tz=UTC)
                        if ouvert_ms
                        else datetime.now(UTC)
                    ),
                    asset_class=ASSET_CLASS_MAP.get(asset, AssetClass.INDICES),
                )
            )
        return positions

    def get_account_sync(self) -> AccountState:
        """Retourne l'etat du compte cTrader.

        L'API n'expose pas d'equity : `ProtoOATrader` fournit le solde (entier mis
        a l'echelle par `moneyDigits`) et `leverageInCents` ; la marge utilisee est
        la somme des `usedMargin` des positions ouvertes. Le P&L latent exigerait
        un abonnement aux cotations (ProtoOASpotEvent) : sans lui, l'equity vaut le
        solde — on n'invente pas de P&L.
        """
        req = ProtoOATraderReq()
        req.ctidTraderAccountId = self.account_id
        future = self._send_request(req, ProtoOATraderRes)
        try:
            res = future.result(timeout=10.0)
        except Exception as exc:
            raise CTraderError(f"get_account timeout/error : {exc}") from exc

        trader = getattr(res, "trader", None)
        if trader is None:
            raise CTraderError("Reponse ProtoOATraderRes sans trader")

        echelle = float(10 ** int(getattr(trader, "moneyDigits", 0) or 0))
        solde = float(getattr(trader, "balance", 0)) / echelle
        leverage = int(getattr(trader, "leverageInCents", 100) or 100) / 100.0

        marge = 0.0
        try:
            req_pos = ProtoOAReconcileReq()
            req_pos.ctidTraderAccountId = self.account_id
            res_pos = self._send_request(req_pos, ProtoOAReconcileRes).result(timeout=10.0)
            for p in getattr(res_pos, "position", []):
                marge += float(getattr(p, "usedMargin", 0)) / echelle
        except Exception as exc:  # noqa: BLE001 - la marge n'est pas bloquante
            logger.warning("cTrader: marge indisponible (%s)", exc)

        return AccountState(
            cash=solde,
            equity=solde,
            margin_used=marge,
            margin_available=max(0.0, solde - marge),
            leverage=int(leverage) or 1,
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
