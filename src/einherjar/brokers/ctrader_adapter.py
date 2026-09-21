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
    ASSET_CLASS_MAP,
    denormalize_symbol,
    load_fees,
    normalize_symbol,
    now_utc_ms,
    ohlcv_to_polars,
    timeframe_to_ctrader_period,
)
from einherjar.brokers.resilience import CircuitBreaker, RateLimiter
from einherjar.core.enums import AssetClass, Direction, OrderType
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
        ProtoOAOrderErrorEvent,
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
    # Le message generique masquait la cause reelle (dependance manquante du paquet) :
    # on journalise l'erreur d'import exacte.
    logger.warning(
        "ctrader-open-api indisponible (%s) : CTraderAdapter fonctionnera en mode stub. "
        "Cause : %s",
        type(_imp_err).__name__,
        CTRADER_IMPORT_ERROR,
    )


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class CTraderError(RuntimeError):
    """Erreur specifique cTrader."""

    pass


# ---------------------------------------------------------------------------
# Conversion des trendbars (fonction pure, testable sans API)
# ---------------------------------------------------------------------------
# Conversion des trendbars (fonction pure, testable sans API)
# ---------------------------------------------------------------------------

# Types d'ordre cTrader (`ProtoOAOrderType` : MARKET=1, LIMIT=2, STOP=3).
# `OrderType.MARKET.value` vaut "market" (minuscules) : comparer a "MARKET" faisait
# partir TOUT ordre marche en LIMIT sans prix limite, donc rejete par le broker.
CTRADER_ORDER_TYPES: dict[OrderType, int] = {
    OrderType.MARKET: 1,
    OrderType.LIMIT: 2,
    OrderType.STOP_MARKET: 3,
}

# 1 point = 1/100000 d'unite de prix (doc officielle cTrader : `low`,
# `deltaOpen`/`deltaClose`/`deltaHigh` sont des entiers en POINTS, pas en
# 10**digits du symbole). Mesure confirmee sur le compte demo reel (2026-09-19) :
# EURUSD low=114858 -> 1.14858 ; APPLE low=33508000 -> 335.08 ;
# US 500 low=764500000 -> 7645.00 ; BTCUSD low=8086545000 -> 80865.45.
CTRADER_POINT_SCALE = 100_000


def trendbars_to_ohlcv(bars: list[Any], limit: int | None = None) -> pl.DataFrame:
    """Convertit des `ProtoOATrendbar` en OHLCV exploitable.

    Encodage cTrader Open API (`ProtoOATrendbar`, doc du proto officiel) :

      - `low` est le SEUL prix ABSOLU de la bougie, en points (1 point = 1/100000) ;
      - `deltaOpen`, `deltaClose` et `deltaHigh` sont des ECARTS signes par rapport
        au `low` de LA MEME bougie ::

            open  = low + deltaOpen
            close = low + deltaClose
            high  = low + deltaHigh

    Les deltas ne se CHAINENT PAS d'une bougie a l'autre. Les enchainer fait
    deriver les prix proportionnellement au nombre de bougies — mesure sur le
    compte reel : EURUSD 1h finissait a 1.2445 au lieu de 1.1488 sur 200 bougies,
    BTCUSD a 116910 au lieu de 80865. Chaque bougie se reconstruit donc seule, et
    aucune n'a besoin d'etre ecartee.

    Args:
        bars: Trendbars brutes de l'API (objets protobuf ou equivalents).
        limit: Ne garder que les `limit` dernieres bougies.

    Returns:
        DataFrame polars [timestamp, open, high, low, close, volume].
    """
    rows: list[list[float]] = []
    for bar in bars:
        low = float(getattr(bar, "low", 0)) / CTRADER_POINT_SCALE
        rows.append(
            [
                int(getattr(bar, "utcTimestampInMinutes", 0)) * 60_000,
                low + float(getattr(bar, "deltaOpen", 0)) / CTRADER_POINT_SCALE,
                low + float(getattr(bar, "deltaHigh", 0)) / CTRADER_POINT_SCALE,
                low,
                low + float(getattr(bar, "deltaClose", 0)) / CTRADER_POINT_SCALE,
                float(getattr(bar, "volume", 0.0)),
            ]
        )
    return ohlcv_to_polars(rows[-limit:] if limit else rows)


def _extraire_message(message: Any) -> Any:
    """Extrait le message protobuf type de l'enveloppe `ProtoMessage`.

    Les callbacks de `Client.send` recoivent l'enveloppe de la librairie, pas le
    message : sans cet extrait, tout `isinstance` echoue et aucun champ n'est
    lisible. Les messages deja extraits sont rendus tels quels.
    """
    try:
        return Protobuf.extract(message)
    except Exception:  # noqa: BLE001 - deja extrait ou message non protobuf
        return message


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
        self._auth_error: str | None = None
        # Vrai seulement si CE thread a demarre le reactor (singleton par process).
        self._owns_reactor = False
        self._symbol_cache: dict[str, int] = {}
        self._symbol_meta: dict[int, dict[str, Any]] = {}
        self._shutdown = False

    # -- Cycle de vie -------------------------------------------------------

    def start(self, timeout: float = 15.0) -> None:
        """Demarre le reactor Twisted et attend l'authentification effective.

        Le thread qui demarre ne prouve rien : ce qui compte est que l'auth
        application PUIS l'auth compte aient ete ACCEPTEES. Un demarrage de socket
        reussi sur un compte refuse doit lever, pas rendre la main en silence.

        Raises:
            CTraderError: paquet absent, autorisation refusee, ou delai depasse.
        """
        if not CTRADER_AVAILABLE:
            raise CTraderError(f"ctrader-open-api manquant : {CTRADER_IMPORT_ERROR}")
        self._auth_error = None
        self._thread = threading.Thread(target=self._run, daemon=True, name="CTraderTwisted")
        self._thread.start()
        echeance = time.monotonic() + timeout
        while time.monotonic() < echeance:
            if self._connected_event.is_set():
                logger.info("CTraderTwistedThread connecte (account_id=%s)", self.account_id)
                return
            if self._auth_error is not None:
                self.stop()
                raise CTraderError(self._auth_error)
            time.sleep(0.05)
        self.stop()
        raise CTraderError(f"Timeout connexion cTrader ({timeout}s)")

    def est_connecte(self) -> bool:
        """Vrai si l'auth application ET compte sont en place sur ce socket."""
        return self._connected_event.is_set()

    def reconnect(self, timeout: float = 15.0) -> None:
        """Relance la connexion sur le reactor DEJA en cours.

        Creer un second `_CTraderTwistedThread` est impossible (reactor Twisted
        singleton par process) : une reconnexion doit passer par le reactor
        existant, sinon elle echoue — et l'ancienne version, en croyant
        reconnecter, laissait la boucle live sans aucune donnee.

        Raises:
            CTraderError: aucun reactor reutilisable, ou auth refusee/timeout.
        """
        if not self._owns_reactor or self._reactor is None or self._client is None:
            raise CTraderError(
                "Reconnexion cTrader impossible : aucun reactor Twisted reutilisable "
                "dans ce process (partager l'adaptateur au lieu d'en creer un autre)"
            )
        self._auth_error = None
        self._connected_event.clear()
        logger.info("cTrader: relance de la connexion sur le reactor en cours")

        def _relancer() -> None:
            # `stopService` ferme le transport ; on ne relance qu'une fois termine,
            # sinon le nouveau socket demarre sur une connexion encore fermee.
            self._client.stopService().addBoth(  # pyright: ignore[reportOptionalMemberAccess]
                lambda _resultat: self._client.startService()  # pyright: ignore[reportOptionalMemberAccess]
            )

        self._reactor.callFromThread(_relancer)
        if not self._connected_event.wait(timeout=timeout):
            raise CTraderError(self._auth_error or f"Timeout reconnexion cTrader ({timeout}s)")

    def stop(self) -> None:
        """Arrete la connexion et le reactor — seulement s'il est a nous.

        Le reactor Twisted est partage par tout le process : l'arreter depuis un
        adaptateur qui ne l'a pas demarre coupe les connexions des autres.
        """
        self._shutdown = True
        self._connected_event.clear()
        if self._owns_reactor and self._reactor is not None:
            self._reactor.callFromThread(self._reactor.stop)
            self._owns_reactor = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def _run(self) -> None:
        from twisted.internet import error as twisted_error
        from twisted.internet import reactor

        self._reactor = reactor
        self._client = Client(self.host, self.port, TcpProtocol)
        self._client.setConnectedCallback(self._on_connected)
        self._client.setDisconnectedCallback(self._on_disconnected)
        self._client.setMessageReceivedCallback(self._on_message)
        self._client.startService()
        self._owns_reactor = True
        try:
            reactor.run(installSignalHandlers=0)
        except twisted_error.ReactorAlreadyRunning:
            # Le reactor Twisted est un singleton par process : un second
            # adaptateur ne peut pas ouvrir de connexion. Ne PAS marquer le reactor
            # comme le notre, sinon le `stop()` de l'appelant couperait la
            # connexion du premier adaptateur (panne constatee : tout le live).
            self._owns_reactor = False
            self._auth_error = (
                "reactor Twisted deja en cours dans ce process : un seul CTraderAdapter "
                "peut etre connecte (partager l'instance, ne pas en creer une seconde)"
            )
            logger.error("cTrader %s", self._auth_error)

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

    def _on_app_auth_ok(self, message: Any) -> None:
        reponse = _extraire_message(message)
        if not isinstance(reponse, ProtoOAApplicationAuthRes):
            self._auth_refusee("authentification application", reponse)
            return
        req = ProtoOAAccountAuthReq()
        req.ctidTraderAccountId = self.account_id
        req.accessToken = self.access_token
        d = self._client.send(req)  # pyright: ignore[reportOptionalMemberAccess]
        d.addCallback(self._on_account_auth_ok)
        d.addErrback(self._on_error)

    def _on_account_auth_ok(self, message: Any) -> None:
        reponse = _extraire_message(message)
        if not isinstance(reponse, ProtoOAAccountAuthRes):
            # Cause la plus frequente : `account_id` est le `traderLogin` affiche par
            # le broker et non le `ctidTraderAccountId` (voir ProtoOAGetAccountListByAccessTokenReq).
            self._auth_refusee(f"authentification du compte {self.account_id}", reponse)
            return
        logger.info("CTrader authentification account OK")
        self._connected_event.set()
        # Pre-charger la liste des symboles pour resoudre les IDs
        self._preload_symbols()

    def _auth_refusee(self, etape: str, reponse: Any) -> None:
        """Enregistre un refus d'autorisation pour que `start()` le remonte."""
        if isinstance(reponse, ProtoOAErrorRes):
            detail = f"{reponse.errorCode} - {reponse.description}"
        else:
            detail = f"reponse inattendue ({type(reponse).__name__})"
        self._auth_error = f"{etape} refusee : {detail}"
        self._connected_event.clear()
        logger.error("cTrader %s", self._auth_error)

    def _on_error(self, failure: Any) -> None:
        logger.error("CTrader erreur Twisted : %s", failure)

    def _on_message(self, _client: Any, msg_wrapper: Any) -> None:
        # La correlation requete/reponse est faite par la librairie (Deferred de
        # `Client.send`) : ici ne passent que les messages SPONTANES du broker
        # (execution events, market data) — non exploites dans cette version.
        msg = _extraire_message(msg_wrapper)
        logger.debug("CTrader message spontane type=%s", type(msg).__name__)

    # -- Internes -----------------------------------------------------------

    def _send_request(
        self, request: Any, response_type: type, timeout_s: float = 15.0
    ) -> ConcurrentFuture:
        """Envoie une requete et rend un Future resolvant la REPONSE TYPEE.

        La correlation requete/reponse est faite par la librairie, via le Deferred
        rendu par `Client.send` : elle pose elle-meme l'enveloppe `ProtoMessage`.
        Les messages `ProtoOA*Req` n'ont PAS de champ `clientMsgId` (ce champ
        n'existe que sur l'enveloppe) — l'ecrire levait
        `AttributeError: ... has no "clientMsgId" field` sur toutes les requetes.

        Une reponse `ProtoOAErrorRes` resout le Future en `CTraderError` : un refus
        de l'API ne doit jamais remonter comme une reponse valide.

        Args:
            request: Message `ProtoOA*Req` a envoyer.
            response_type: Classe du message de reponse attendu.
            timeout_s: Delai de reponse cote librairie. Le defaut de `Client.send`
                est de 5 s, trop court des qu'une rafale de requetes part (l'amorcage
                de l'historique en envoie des dizaines) : un depassement se traduit
                par un `TimeoutError` du Deferred, donc par une mesure manquante.

        Returns:
            `concurrent.futures.Future` a attendre par `future.result(timeout)`.
        """
        if not self._connected_event.is_set():
            raise CTraderError("Non connecte")
        future: ConcurrentFuture = ConcurrentFuture()

        def _resoudre(reponse: Any) -> None:
            # Un ordre refuse (marche ferme, volume invalide, marge...) ne revient pas
            # en ProtoOAErrorRes mais en ProtoOAOrderErrorEvent : sans ce cas, la
            # raison exacte serait perdue ("reponse inattendue").
            if isinstance(reponse, ProtoOAOrderErrorEvent):
                if not future.done():
                    future.set_exception(
                        CTraderError(
                            f"ordre refuse par cTrader : {reponse.errorCode} - "
                            f"{reponse.description}"
                        )
                    )
                return
            if isinstance(reponse, ProtoOAErrorRes):
                if not future.done():
                    future.set_exception(
                        CTraderError(
                            f"{response_type.__name__} refuse par l'API : "
                            f"{reponse.errorCode} - {reponse.description}"
                        )
                    )
                return
            if not isinstance(reponse, response_type):
                if not future.done():
                    future.set_exception(
                        CTraderError(
                            f"reponse inattendue ({type(reponse).__name__}) au lieu de "
                            f"{response_type.__name__}"
                        )
                    )
                return
            if not future.done():
                future.set_result(reponse)

        def _annuler(failure: Any) -> None:
            if not future.done():
                future.set_exception(
                    CTraderError(f"envoi {response_type.__name__} echoue : {failure}")
                )

        def _envoyer() -> None:
            # `Client.send` doit partir du thread du reactor (les transports Twisted
            # ne sont pas thread-safe) : on y delegue l'envoi et l'enregistrement.
            try:
                # `clientMsgId` explicite : le defaut de la librairie est
                # `str(id(responseDeferred))`, un identifiant RECYCLE par Python des que
                # le Deferred precedent est libere. Une reponse tardive peut alors etre
                # remise a la NOUVELLE requete (constate : un ProtoOAExecutionEvent
                # rendu en reponse a un Reconcile).
                self._client.send(  # pyright: ignore[reportOptionalMemberAccess]
                    request,
                    clientMsgId=f"EINHERJAR-{uuid.uuid4().hex}",
                    responseTimeoutInSeconds=timeout_s,
                ).addCallbacks(
                    lambda message: _resoudre(_extraire_message(message)), _annuler
                )
            except Exception as exc:  # noqa: BLE001 - remonte a l'appelant bloque
                if not future.done():
                    future.set_exception(
                        CTraderError(f"envoi {response_type.__name__} impossible : {exc}")
                    )

        self._reactor.callFromThread(_envoyer)  # pyright: ignore[reportOptionalMemberAccess]
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
        response = _extraire_message(message)
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

    def _resolve_symbol_sync(self, asset: str, attente_s: float = 10.0) -> int:
        """Retourne le symbolId cTrader pour un asset MIDAS (avec cache).

        Le cache est rempli de facon ASYNCHRONE a la connexion (reponse a
        `ProtoOASymbolsListReq`) : une requete envoyee juste apres `connect()` ne
        doit pas echouer parce que la reponse n'est pas encore arrivee.

        Args:
            asset: Symbole MIDAS (ex. "BTCUSD").
            attente_s: Delai maximal d'attente du chargement des symboles.

        Raises:
            CTraderError: si le symbole n'est pas servi par le broker.
        """
        if asset in self._symbol_cache:
            return self._symbol_cache[asset]
        echeance = time.monotonic() + attente_s
        while not self._symbol_cache and time.monotonic() < echeance:
            time.sleep(0.1)
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
        # Une bougie de plus : la derniere bougie peut ne pas etre encore close
        # cote broker, on demande donc large puis on garde les `limit` dernieres.
        req.count = limit + 1

        future = self._send_request(req, ProtoOAGetTrendbarsRes)
        try:
            res = future.result(timeout=20.0)
        except CTraderError:
            raise
        except Exception as exc:
            raise CTraderError(f"get_ohlcv timeout/error : {exc}") from exc

        bars = getattr(res, "trendbar", [])
        if not bars:
            return ohlcv_to_polars([])

        # Chaque bougie se reconstruit seule (`low + delta`) : aucune n'est ecartee,
        # `count` est demande large (limit + 1) pour absorber la derniere bougie
        # encore ouverte que le broker peut omettre.
        return trendbars_to_ohlcv(bars, limit=limit)

    def _load_symbol_details_sync(self, symbol_id: int) -> int | None:
        """Charge les details d'un symbole (digits, min/step volume, lot).

        Necessaire avant de passer un ordre : le volume cTrader se donne en 0,01
        unite et doit rester dans les bornes du broker.

        Returns:
            Le nombre de decimales du symbole, ou None si l'API ne le fournit pas
            (les autres champs restent dans `self._symbol_meta`).
        """
        req = ProtoOASymbolByIdReq()
        req.ctidTraderAccountId = self.account_id
        req.symbolId.append(symbol_id)
        try:
            future = self._send_request(req, ProtoOASymbolByIdRes)
            res = future.result(timeout=20.0)
        except CTraderError:
            raise
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
        """Passe un ordre sur cTrader (TP/SL transmis au broker).

        `ProtoOANewOrderReq.stopLoss` et `takeProfit` sont des PRIX ABSOLUS en
        `double` (pas des points en 1/100000, contrairement a `relativeStopLoss` /
        `relativeTakeProfit` qui sont des int64) : le prix de l'ordre est transmis
        tel quel.

        Raises:
            CTraderError: type d'ordre non supporte, volume hors bornes broker,
                ou refus de l'API (remonte par `_send_request`).
        """
        if order.order_type not in CTRADER_ORDER_TYPES:
            raise CTraderError(
                f"type d'ordre non supporte par cTrader: {order.order_type} "
                f"(supportes: {', '.join(t.value for t in CTRADER_ORDER_TYPES)})"
            )
        symbol_id = self._resolve_symbol_sync(order.asset)
        # Sans les metadonnees du symbole, `minVolume`/`stepVolume` sont absents et
        # l'ordre partirait avec un volume que le broker refusera : on les charge
        # avant de calculer (SymbolByIdReq, leve si l'API ne repond pas).
        meta = self._symbol_meta.get(symbol_id) or {}
        if "stepVolume" not in meta:
            self._load_symbol_details_sync(symbol_id)
            meta = self._symbol_meta.get(symbol_id, {})
        # cTrader compte le volume en 0,01 unite de l'actif de base.
        volume = int(round(order.quantity * 100))
        min_volume = meta.get("minVolume")
        step_volume = meta.get("stepVolume")
        if step_volume:
            # Arrondi vers le BAS au pas du broker, jamais vers le haut : arrondir a la
            # hausse (max(pas, ...)) transformait une taille minuscule en taille minimale
            # et multipliait le risque reel sans que personne ne le voie.
            volume = (volume // int(step_volume)) * int(step_volume)
        if volume <= 0 or (min_volume and volume < int(min_volume)):
            risque = (
                float(order.quantity)
                * abs(float(order.entry_price or 0) - float(order.sl_price or 0))
            )
            raise CTraderError(
                f"volume {volume} (centiemes) sous le minimum broker {min_volume} pour "
                f"{order.asset} : taille demandee {order.quantity} (risque ~{risque:.2f}), "
                "ordre refuse cote client au lieu d'etre agrandi"
            )
        req = ProtoOANewOrderReq()
        req.ctidTraderAccountId = self.account_id
        req.symbolId = symbol_id
        digits = int(meta.get("digits") or 0)
        if order.order_type == OrderType.MARKET:
            if not order.sl_price or float(order.sl_price) <= 0:
                raise CTraderError(
                    f"ordre marche {order.asset} sans stop-loss valide "
                    f"({order.sl_price!r}) : ordre refuse cote client"
                )
            if not order.tp_price or float(order.tp_price) <= 0:
                raise CTraderError(
                    f"ordre marche {order.asset} sans take-profit valide "
                    f"({order.tp_price!r}) : ordre refuse cote client"
                )
        req.orderType = CTRADER_ORDER_TYPES[order.order_type]
        req.tradeSide = 1 if order.direction == Direction.LONG else 2
        req.volume = volume
        # Les ordres a cours limite/stop EXIGENT leur prix : sans lui l'API refuse.
        if order.order_type == OrderType.LIMIT:
            if order.entry_price is None:
                raise CTraderError(f"ordre LIMIT sans prix d'entree pour {order.asset}")
            req.limitPrice = round(float(order.entry_price), digits)
        elif order.order_type == OrderType.STOP_MARKET:
            if order.entry_price is None:
                raise CTraderError(f"ordre STOP sans prix de declenchement pour {order.asset}")
            req.stopPrice = round(float(order.entry_price), digits)
        # Le label transporte le nom de l'einher : il revient avec la position
        # (tradeData.label) et permet la sortie sur duree de tenue.
        req.label = str(getattr(order, "einher_name", "") or "")[:100]
        # Les prix doivent etre ARRONDIS aux decimales du symbole : le broker refuse
        # tout prix trop precis ("Order price = 82233.19089321411 has more digits than
        # symbol allows. Allowed 3 digits"). Une confluence calculee sur des prix
        # ponderes produit naturellement des flottants a 14 decimales.
        if order.sl_price is not None and float(order.sl_price) > 0:
            req.stopLoss = round(float(order.sl_price), digits)
        if order.tp_price is not None and float(order.tp_price) > 0:
            req.takeProfit = round(float(order.tp_price), digits)

        if req.stopLoss and req.takeProfit:
            if req.tradeSide == 1 and not (req.stopLoss < req.takeProfit):
                raise CTraderError(
                    f"stop-loss {req.stopLoss} >= take-profit {req.takeProfit} sur un "
                    f"acha t {order.asset} : protections incoherentes, ordre non envoye"
                )
            if req.tradeSide == 2 and not (req.stopLoss > req.takeProfit):
                raise CTraderError(
                    f"stop-loss {req.stopLoss} <= take-profit {req.takeProfit} sur une "
                    f"vente {order.asset} : protections incoherentes, ordre non envoye"
                )

        future = self._send_request(req, ProtoOAExecutionEvent)
        try:
            res = future.result(timeout=20.0)
        except CTraderError:
            raise
        except Exception as exc:
            raise CTraderError(f"place_order timeout/error : {exc}") from exc

        # ProtoOAExecutionEvent ne porte pas de champ `price` : le prix du fill est
        # celui de la position ouverte (ProtoOAPosition.price, un double) et le volume
        # est sur `position.tradeData.volume` (en 0,01 unite) — `ProtoOAPosition`
        # n'a PAS de champ `volume` (lire `position.volume` rendait 0).
        position = getattr(res, "position", None)
        execution_type = int(getattr(res, "executionType", 0) or 0)
        # 2 = ORDER_ACCEPTED, 3 = ORDER_FILLED, 11 = ORDER_PARTIAL_FILL : sur un marche
        # l'ordre est d'abord ACCEPTE, donc l'evenement ne porte pas encore le prix ni
        # le volume d'execution. On relit alors la position que le broker vient d'ouvrir.
        if execution_type not in (3, 11) and position is not None:
            position_id = int(getattr(position, "positionId", 0))
            if position_id:
                relue = self._position_brute_sync(position_id)
                if relue is not None:
                    position = relue
        trade_data = getattr(position, "tradeData", None) if position is not None else None
        volume_centiemes = getattr(trade_data, "volume", None) if trade_data is not None else None
        if not volume_centiemes:
            volume_centiemes = None
        fill_qty = (
            float(volume_centiemes) / 100.0
            if volume_centiemes is not None
            else float(order.quantity)
        )
        fill_price = (
            getattr(position, "price", None)
            or getattr(res, "price", None)
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

    def _position_brute_sync(self, position_id: int) -> Any | None:
        """Relit une position ouverte par son id (message brut `ProtoOAPosition`).

        Indispensable juste apres un ordre marche : le premier `ProtoOAExecutionEvent`
        est un ORDER_ACCEPTED (executionType=2) ou le prix et le volume du fill ne sont
        pas encore remplis — les lire la donne 0.
        """
        req = ProtoOAReconcileReq()
        req.ctidTraderAccountId = self.account_id
        try:
            res = self._send_request(req, ProtoOAReconcileRes).result(timeout=20.0)
        except Exception as exc:  # noqa: BLE001 - l'appelant retombe sur l'evenement
            logger.warning("cTrader: relecture de la position %s impossible (%s)", position_id, exc)
            return None
        for position in getattr(res, "position", []):
            if int(getattr(position, "positionId", 0)) == int(position_id):
                return position
        return None

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
            res = future.result(timeout=20.0)
        except CTraderError:
            raise
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
                    # Defaut volontairement neutre : un actif hors ASSET_CLASS_MAP ne doit pas
            # empecher la lecture des positions (le broker reste la source de verite).
            asset_class=ASSET_CLASS_MAP.get(asset, AssetClass.INDEX),
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
            res = future.result(timeout=20.0)
        except CTraderError:
            raise
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
            res_pos = self._send_request(req_pos, ProtoOAReconcileRes).result(timeout=20.0)
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

    def get_comptes_autorises_sync(self) -> list[dict[str, Any]]:
        """Comptes autorises par le token -> {account_id, is_live, login}.

        `account_id` est le `ctidTraderAccountId` attendu dans credentials.json ;
        `login` est le numero de compte affiche chez le broker. Les deux sont
        exposes : les confondre fait refuser toutes les requetes suivantes.
        """
        req = ProtoOAGetAccountListByAccessTokenReq()
        req.accessToken = self.access_token
        try:
            res = self._send_request(req, ProtoOAGetAccountListByAccessTokenRes).result(
                timeout=15.0
            )
        except CTraderError:
            raise
        except Exception as exc:
            raise CTraderError(f"get_comptes_autorises timeout/error : {exc}") from exc

        return [
            {
                "account_id": int(getattr(compte, "ctidTraderAccountId", 0)),
                "is_live": bool(getattr(compte, "isLive", False)),
                "login": int(getattr(compte, "traderLogin", 0) or 0),
            }
            for compte in getattr(res, "ctidTraderAccount", [])
        ]

    def get_symboles_disponibles_sync(self, attente_s: float = 10.0) -> dict[str, int]:
        """Symboles reellement servis par le broker -> symbolId.

        Le cache est rempli de facon ASYNCHRONE a la connexion (reponse a
        `ProtoOASymbolsListReq`) : on attend son arrivee, bornee par `attente_s`.

        Returns:
            Table nom de symbole (majuscules) -> symbolId ; vide si indisponible.
        """
        echeance = time.monotonic() + attente_s
        while not self._symbol_cache and time.monotonic() < echeance:
            time.sleep(0.1)
        return dict(self._symbol_cache)

    def close_position_sync(self, position_id: int, volume: int | None = None) -> bool:
        """Ferme une position par son ID.

        Args:
            position_id: Identifiant de la position (`ProtoOAPosition.positionId`).
            volume: Volume a fermer en 0,01 unite. Si absent, il est relu depuis la
                position ouverte — `ProtoOAClosePositionReq.volume` est un champ
                OBLIGATOIRE : sans lui la requete ne peut meme pas etre serialisee
                (`EncodeError: missing required fields: volume`).

        Raises:
            CTraderError: position introuvable et volume non fourni.
        """
        if volume is None:
            position = self._position_brute_sync(position_id)
            if position is None:
                raise CTraderError(
                    f"close_position: position {position_id} introuvable "
                    "(volume non fourni et absente du Reconcile)"
                )
            volume = int(getattr(getattr(position, "tradeData", None), "volume", 0))
            if volume <= 0:
                raise CTraderError(
                    f"close_position: volume illisible pour la position {position_id}"
                )
        req = ProtoOAClosePositionReq()
        req.ctidTraderAccountId = self.account_id
        req.positionId = position_id
        req.volume = volume
        future = self._send_request(req, ProtoOAExecutionEvent)
        try:
            future.result(timeout=20.0)
            return True
        except CTraderError as exc:
            logger.error("close_position echoue : %s", exc)
            return False
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
        self.last_error: str | None = None

        self._fees = load_fees("ctrader")

    # -- Connexion ----------------------------------------------------------

    async def connect(self) -> bool:
        """Etablit la connexion gRPC + auth."""
        if not CTRADER_AVAILABLE:
            self.last_error = (
                "ctrader-open-api non installe : pip install ctrader-open-api"
            )
            logger.error("%s", self.last_error)
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
            self.last_error = None
            self._last_ping = time.time()
            return True
        except Exception as exc:
            self.last_error = str(exc)
            logger.error("Connexion cTrader echoue : %s", exc)
            self._connected = False
            return False

    async def disconnect(self) -> None:
        """Ferme la connexion."""
        if self._twisted is not None:
            self._twisted.stop()
        self._connected = False

    async def ensure_connected(self) -> bool:
        """Verifie / restaure la connexion si necessaire.

        La fraicheur du dernier appel n'est PAS un critere : une connexion cTrader
        reste ouverte sans trafic (echauffement JIT de 95 s, marche ferme...).
        Seul l'etat reel du socket compte ; si la connexion est tombee, on la relance
        sur le reactor existant au lieu d'en creer un second (impossible : le
        reactor Twisted est un singleton par process).

        Returns:
            True si la connexion est (re)etablie.
        """
        if self._twisted is not None and self._connected and self._twisted.est_connecte():
            self._last_ping = time.time()
            return True
        if self._twisted is None:
            return await self.connect()
        logger.info("Reconnexion cTrader necessaire...")
        try:
            await asyncio.to_thread(self._twisted.reconnect)
        except Exception as exc:
            self.last_error = str(exc)
            self._connected = False
            logger.error("Reconnexion cTrader echouee : %s", exc)
            return False
        self._connected = True
        self.last_error = None
        self._last_ping = time.time()
        return True

    # -- Resilience wrapper -------------------------------------------------

    async def _safe_call(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        """Execute un appel avec circuit breaker + rate limiter + reconnexion."""
        if not self.circuit.can_execute():
            raise RuntimeError("Circuit breaker ouvert pour cTrader")
        await self.rate_limiter.acquire()
        if not self._connected or self._twisted is None or not self._twisted.est_connecte():
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

    async def get_comptes_autorises(self) -> list[dict[str, Any]]:
        """Comptes cTrader autorises par ce token d'acces.

        Indispensable pour renseigner `account_id` : le `ctidTraderAccountId` de
        l'Open API ne correspond PAS au numero de compte affiche par le broker.

        Returns:
            Liste de {account_id, is_live, login}.
        """
        return await self._safe_call("get_comptes_autorises")

    async def get_symboles_disponibles(self, attente_s: float = 10.0) -> dict[str, int]:
        """Symboles reellement disponibles chez le broker -> symbolId.

        Sert a verifier que l'univers du corpus est negociable avant de trader.

        Args:
            attente_s: Delai maximal d'attente du chargement des symboles.

        Returns:
            Table nom de symbole (majuscules) -> symbolId.
        """
        return await self._safe_call("get_symboles_disponibles", attente_s)

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
            "last_error": self.last_error,
        }

    # -- Methodes supplementaires -------------------------------------------

    async def close_position(self, position_id: int, volume: int | None = None) -> bool:
        """Ferme une position par son ID cTrader.

        Args:
            position_id: Identifiant de la position.
            volume: Volume en 0,01 unite ; relu depuis le broker si absent.
        """
        return await self._safe_call("close_position", position_id, volume)
