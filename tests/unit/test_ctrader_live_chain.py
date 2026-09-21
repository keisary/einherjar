"""Chaine cTrader : periodes, symbols et transport requete/reponse.

Verrouille les 3 bugs constates le 2026-09-19 contre le compte demo reel :

1. `timeframe_to_ctrader_period` renvoyait des MINUTES alors que
   `ProtoOATrendbarPeriod` est une enumeration (`1h` = 9, pas 60) -> l'API levait
   `ValueError: invalid enumerator` sur 15m/1h/4h/1d.
2. `_send_request` ecrivait `request.clientMsgId`, champ inexistant sur les
   messages `ProtoOA*Req` -> `AttributeError` sur TOUTES les requetes (compte,
   positions, bougies, ordres) ; la librairie rend un Deferred a la place.
3. Les callbacks d'auth ne verifiaient pas le type de la reponse : un
   `ProtoOAErrorRes` faisait quand meme passer la connexion pour bonne.

Aucun de ces tests ne touche le reseau : le transport est remplace par un client
factice, le code teste reste le vrai.
"""

from __future__ import annotations

import pytest

pytest.importorskip("ctrader_open_api")

from ctrader_open_api.messages.OpenApiCommonMessages_pb2 import ProtoMessage  # noqa: E402
from ctrader_open_api.messages.OpenApiMessages_pb2 import (  # noqa: E402
    ProtoOAAccountAuthRes,
    ProtoOAErrorRes,
    ProtoOAExecutionEvent,
    ProtoOAReconcileReq,
    ProtoOAReconcileRes,
    ProtoOASymbolsListRes,
    ProtoOATraderReq,
)
from ctrader_open_api.messages.OpenApiModelMessages_pb2 import (  # noqa: E402
    ProtoOATrendbarPeriod,
)

from einherjar.brokers import broker_utils  # noqa: E402
from einherjar.core.enums import Direction, OrderType  # noqa: E402
from einherjar.core.models import Order  # noqa: E402

from einherjar.brokers.ctrader_adapter import (  # noqa: E402
    CTRADER_AVAILABLE,
    CTraderError,
    _CTraderTwistedThread,
)

pytestmark = pytest.mark.skipif(
    not CTRADER_AVAILABLE, reason="ctrader-open-api indisponible"
)

# ---------------------------------------------------------------------------
# 1. Periodes : enum, pas minutes
# ---------------------------------------------------------------------------

VALEURS_ATTENDUES = {"5m": 5, "15m": 7, "1h": 9, "4h": 10, "1d": 12}
NOMS_ENUM = {"5m": "M5", "15m": "M15", "1h": "H1", "4h": "H4", "1d": "D1"}


@pytest.mark.parametrize(("tf", "attendu"), sorted(VALEURS_ATTENDUES.items()))
def test_periode_est_la_valeur_d_enumeration(tf: str, attendu: int):
    """Les valeurs sont celles de `ProtoOATrendbarPeriod`, PAS des minutes."""
    assert broker_utils.timeframe_to_ctrader_period(tf) == attendu


@pytest.mark.parametrize(("tf", "nom"), sorted(NOMS_ENUM.items()))
def test_periode_correspond_a_l_enum_installee(tf: str, nom: str):
    """Garde-fou : le mapping doit suivre l'enum du paquet REELLEMENT installe.

    Passer des minutes ne tombe juste que pour 5m (M5=5) ; cette comparaison par
    NOM detecte toute derive de version du paquet.
    """
    valeurs = {v.number: v.name for v in ProtoOATrendbarPeriod.DESCRIPTOR.values}
    assert valeurs[broker_utils.timeframe_to_ctrader_period(tf)] == nom


@pytest.mark.parametrize("tf", sorted(VALEURS_ATTENDUES))
def test_periode_round_trip(tf: str):
    periode = broker_utils.timeframe_to_ctrader_period(tf)
    assert broker_utils.ctrader_period_to_timeframe(periode) == tf


def test_timeframe_non_supporte_refuse():
    with pytest.raises(ValueError, match="non supporte"):
        broker_utils.timeframe_to_ctrader_period("3m")
    assert broker_utils.ctrader_period_to_timeframe(6) == "period_6"


# ---------------------------------------------------------------------------
# 2. Symboles : le nommage depend du broker
# ---------------------------------------------------------------------------

SPOTWARE = {
    "AAPL": "APPLE",
    "MSFT": "MICROSOFT",
    "NVDA": "NVIDIA",
    "AMZN": "AMAZON",
    "GOOGL": "ALPHABET",
    "TSLA": "TESLA MOTORS",
    "JPM": "JPMORGAN",
    "XOM": "EXXON",
    "SP500": "US 500",
    "NASDAQ100": "US TECH 100",
    "DOWJONES": "US 30",
    "DAX40": "GERMANY 40",
    "WTIUSD": "XTIUSD",
    "BRENT": "XBRUSD",
}


@pytest.mark.parametrize(("midas", "ctrader"), sorted(SPOTWARE.items()))
def test_mapping_spotware(midas: str, ctrader: str):
    """Ce broker sert les actions/indices sous leur NOM COMPLET."""
    assert broker_utils.normalize_symbol(midas, "spotware") == ctrader
    assert broker_utils.denormalize_symbol(ctrader, "spotware") == midas


def test_mapping_spotware_insensible_a_la_casse():
    assert broker_utils.normalize_symbol("AAPL", "Spotware") == "APPLE"
    assert broker_utils.normalize_symbol("AAPL", " spotware ") == "APPLE"


def test_mapping_ic_markets_inchange():
    """Le mapping historique ne doit pas bouger (autres comptes)."""
    assert broker_utils.normalize_symbol("AAPL", "ic_markets") == "AAPL"
    assert broker_utils.normalize_symbol("SP500", "ic_markets") == "US500"
    assert broker_utils.normalize_symbol("WTIUSD", "ic_markets") == "USOUSD"
    assert broker_utils.normalize_symbol("BTCUSD", "ic_markets") == "BTCUSD"


def test_broker_inconnu_retombe_sur_le_defaut():
    assert broker_utils.normalize_symbol("SP500", "") == "US500"
    assert broker_utils.normalize_symbol("SP500", "broker_inexistant") == "US500"
    # Forex et crypto non concernes par les overrides
    assert broker_utils.normalize_symbol("EURUSD", "spotware") == "EURUSD"


# ---------------------------------------------------------------------------
# 3. Transport : Deferred de la librairie, champ clientMsgId interdit
# ---------------------------------------------------------------------------


class _Enveloppe:
    """Deferred minimal : appelle le callback de succes immediatement."""

    def __init__(self, reponse) -> None:
        self._reponse = reponse

    def addCallbacks(self, ok, _err):  # noqa: N802 - API Twisted
        ok(self._reponse)
        return self


class _FauxClient:
    """Reproduit la signature reelle de `Client.send` (timeout de reponse inclus).

    Le defaut de la librairie est de 5 s : l'adaptateur doit passer explicitement un
    delai plus large, sinon une rafale de requetes (amorcage de l'historique) part en
    `TimeoutError` sans que rien ne le signale.
    """

    def __init__(self, reponse) -> None:
        self.reponse = reponse
        self.envoyes: list[object] = []
        self.timeouts: list[float] = []
        self.client_msg_ids: list[str] = []

    def send(self, request, clientMsgId=None, responseTimeoutInSeconds=5, **params):
        self.envoyes.append(request)
        self.timeouts.append(responseTimeoutInSeconds)
        self.client_msg_ids.append(clientMsgId)
        return _Enveloppe(self.reponse)


class _FauxReactor:
    """Execute l'envoi comme si nous etions dans le thread du reactor."""

    def callFromThread(self, fonction, *args, **kwargs):  # noqa: N802
        fonction(*args, **kwargs)


def _remplir_champs_requis(message) -> None:
    """Renseigne les champs proto2 obligatoires (`ctidTraderAccountId`, `errorCode`).

    Sans eux, `SerializeToString` leve `EncodeError` : c'est la meme contrainte que
    celle que l'API impose aux requetes reelles.
    """
    champs = message.DESCRIPTOR.fields_by_name
    # `ProtoOAErrorRes.errorCode` est un CHAMP CHAINE dans ce paquet (les enums y sont
    # generees en string) : l'affecter en entier leve TypeError.
    for nom, valeur in (("ctidTraderAccountId", 4242), ("errorCode", "ACCOUNT_NOT_AUTHORIZED")):
        if nom in champs and not message.HasField(nom):
            setattr(message, nom, valeur)


def _enveloppe(message) -> ProtoMessage:
    """Enveloppe reelle `ProtoMessage` (celle que `Client.send` pose)."""
    _remplir_champs_requis(message)
    env = ProtoMessage()
    env.payloadType = type(message)().payloadType
    env.payload = message.SerializeToString()
    return env


def _thread(reponse) -> _CTraderTwistedThread:
    thread = _CTraderTwistedThread("host", 1, "id", "secret", "token", 4242, "spotware")
    thread._connected_event.set()
    thread._client = _FauxClient(reponse)  # type: ignore[assignment]
    thread._reactor = _FauxReactor()  # type: ignore[assignment]
    return thread


def test_send_request_ne_touche_pas_aux_champs_du_message():
    """Regression BUG 2 : `ProtoOA*Req` n'a pas de `clientMsgId`.

    Avant le correctif, cet appel levait `AttributeError: Protocol message
    ProtoOATraderReq has no "clientMsgId" field` et cassait toute la chaine.
    """
    thread = _thread(_enveloppe(ProtoOAReconcileRes()))
    future = thread._send_request(ProtoOATraderReq(), ProtoOAReconcileRes)
    assert isinstance(future.result(timeout=1.0), ProtoOAReconcileRes)
    assert len(thread._client.envoyes) == 1  # type: ignore[union-attr]


def test_send_request_demande_un_delai_de_reponse_large():
    """Regression : le defaut de 5 s de `Client.send` faisait echouer les rafales."""
    thread = _thread(_enveloppe(ProtoOAReconcileRes()))
    thread._send_request(ProtoOATraderReq(), ProtoOAReconcileRes).result(timeout=2.0)
    assert thread._client.timeouts[0] >= 10.0  # type: ignore[union-attr]


def test_send_request_extrait_le_message_type():
    """La reponse revient emballee : elle doit etre extraite du `ProtoMessage`."""
    reponse = ProtoOAReconcileRes()
    reponse.ctidTraderAccountId = 4242
    thread = _thread(_enveloppe(reponse))
    resultat = thread._send_request(ProtoOATraderReq(), ProtoOAReconcileRes).result(
        timeout=1.0
    )
    assert resultat.ctidTraderAccountId == 4242


def test_send_request_refus_api_leve_ctrader_error():
    """Un `ProtoOAErrorRes` ne doit JAMAIS passer pour une reponse valide."""
    erreur = ProtoOAErrorRes()
    erreur.description = "Trading account is not authorized"
    thread = _thread(_enveloppe(erreur))
    with pytest.raises(CTraderError, match="authorized"):
        thread._send_request(ProtoOATraderReq(), ProtoOAReconcileRes).result(timeout=1.0)


def test_send_request_type_inattendu_leve():
    thread = _thread(_enveloppe(ProtoOASymbolsListRes()))
    with pytest.raises(CTraderError, match="reponse inattendue"):
        thread._send_request(ProtoOATraderReq(), ProtoOAReconcileRes).result(timeout=1.0)


def test_send_request_refuse_si_non_connecte():
    thread = _CTraderTwistedThread("host", 1, "id", "secret", "token", 4242, "spotware")
    with pytest.raises(CTraderError, match="Non connecte"):
        thread._send_request(ProtoOATraderReq(), ProtoOAReconcileRes)


# ---------------------------------------------------------------------------
# 4. Authentification : un refus ne doit pas produire un "connecte"
# ---------------------------------------------------------------------------


def test_account_auth_refusee_ne_rend_pas_la_main():
    """Regression BUG 3 : `connect()` disait OK alors que le compte etait refuse."""
    erreur = ProtoOAErrorRes()
    erreur.description = "Trading account is not authorized"
    thread = _thread(_enveloppe(erreur))
    thread._connected_event.clear()

    thread._on_account_auth_ok(_enveloppe(erreur))

    assert not thread._connected_event.is_set()
    assert thread._auth_error is not None
    assert "authorized" in thread._auth_error


def test_account_auth_acceptee_leve_l_evenement():
    thread = _thread(_enveloppe(ProtoOAAccountAuthRes()))
    thread._connected_event.clear()

    thread._on_account_auth_ok(_enveloppe(ProtoOAAccountAuthRes()))

    assert thread._connected_event.is_set()
    assert thread._auth_error is None


def test_start_remonte_le_refus_au_lieu_de_connecter():
    """`start()` doit lever quand l'auth est refusee (jamais de faux 'connecte')."""
    erreur = ProtoOAErrorRes()
    erreur.description = "Trading account is not authorized"

    class _ThreadRefuse(_CTraderTwistedThread):
        def _run(self) -> None:  # pas de socket : on simule un refus immediat
            self._auth_refusee("authentification du compte 5958", erreur)

    thread = _ThreadRefuse("host", 1, "id", "secret", "token", 5958, "spotware")
    with pytest.raises(CTraderError, match="authorized"):
        thread.start(timeout=2.0)


# ---------------------------------------------------------------------------
# 5. Cache des symboles (alimente a la connexion)
# ---------------------------------------------------------------------------


def test_cache_symbols_remplit_le_cache():
    reponse = ProtoOASymbolsListRes()
    for nom, identifiant in (("APPLE", 21508), ("US 500", 21499), ("EURUSD", 1)):
        symbole = reponse.symbol.add()
        symbole.symbolId = identifiant
        symbole.symbolName = nom

    thread = _thread(_enveloppe(reponse))
    thread._cache_symbols(_enveloppe(reponse))

    assert thread._symbol_cache["APPLE"] == 21508
    assert thread._symbol_cache["US 500"] == 21499
    assert thread._symbol_cache["EURUSD"] == 1
    assert thread._resolve_symbol_sync("AAPL") == 21508
    assert thread._resolve_symbol_sync("SP500") == 21499


def test_cache_symbols_ignore_une_erreur_api():
    erreur = ProtoOAErrorRes()
    erreur.description = "Trading account is not authorized"
    thread = _thread(_enveloppe(erreur))
    thread._cache_symbols(_enveloppe(erreur))
    assert thread._symbol_cache == {}


# ---------------------------------------------------------------------------
# 6. Reactor Twisted : singleton par process (un seul adaptateur a la fois)
# ---------------------------------------------------------------------------


class _FauxReactorTrace:
    """Reactor factice qui enregistre ce qu'on lui demande d'executer."""

    def __init__(self) -> None:
        self.executes: list[str] = []

    def stop(self) -> None:
        """Equivalent de `reactor.stop()` (ce que `stop()` de l'adaptateur appelle)."""
        self.executes.append("stop")

    def callFromThread(self, fonction, *args, **kwargs):  # noqa: N802
        if fonction == self.stop:
            self.stop()
            return
        self.executes.append(getattr(fonction, "__name__", "fonction"))
        fonction(*args, **kwargs)


def test_stop_ne_coupe_pas_un_reactor_qui_n_est_pas_le_sien():
    """Regression : le 2e adaptateur coupait le reactor du 1er (tout le live).

    Contexte mesure : `main.py` creait un second adaptateur pour l'API ; son
    `start()` echouait (`ReactorAlreadyRunning`) puis son `stop()` appelait
    `reactor.stop()` sur le reactor PARTAGE, tuant une connexion valide.
    """
    thread = _CTraderTwistedThread("host", 1, "id", "secret", "token", 1, "spotware")
    trace = _FauxReactorTrace()
    thread._reactor = trace  # type: ignore[assignment]
    thread._owns_reactor = False

    thread.stop()

    assert trace.executes == [], "un reactor non possede ne doit jamais etre arrete"


def test_stop_arrete_le_reactor_qu_on_a_demarre():
    thread = _CTraderTwistedThread("host", 1, "id", "secret", "token", 1, "spotware")
    trace = _FauxReactorTrace()
    thread._reactor = trace  # type: ignore[assignment]
    thread._owns_reactor = True

    thread.stop()

    assert trace.executes == ["stop"]  # reactor.stop() du reactor Twisted
    assert thread._owns_reactor is False


def test_reconnect_refuse_sans_reactor_reutilisable():
    thread = _CTraderTwistedThread("host", 1, "id", "secret", "token", 1, "spotware")
    with pytest.raises(CTraderError, match="reactor"):
        thread.reconnect()


def test_est_connecte_suit_l_evenement_d_auth():
    thread = _CTraderTwistedThread("host", 1, "id", "secret", "token", 1, "spotware")
    assert thread.est_connecte() is False
    thread._connected_event.set()
    assert thread.est_connecte() is True


class _TwistedFactice:
    """Transport factice : compte les reconnexions demandees."""

    def __init__(self, connecte: bool) -> None:
        self.connecte = connecte
        self.reconnexions = 0

    def est_connecte(self) -> bool:
        return self.connecte

    def reconnect(self) -> None:
        self.reconnexions += 1
        self.connecte = True


def test_ensure_connected_ne_recree_pas_une_connexion_saine():
    """Un appel apres une longue inactivite ne doit pas relancer la connexion.

    Cas reel : 95 s d'echauffement JIT sans aucun appel broker ; l'ancienne regle
    « dernier appel > 30 s = reconnexion » recreait une connexion... impossible
    (reactor unique), donc la boucle live restait sans donnees.
    """
    import asyncio

    from einherjar.brokers.ctrader_adapter import CTraderAdapter

    adapter = CTraderAdapter("id", "secret", "token", 1, broker_name="spotware")
    transport = _TwistedFactice(connecte=True)
    adapter._twisted = transport  # type: ignore[assignment]
    adapter._connected = True
    adapter._last_ping = 0.0  # dernier appel tres ancien

    assert asyncio.run(adapter.ensure_connected()) is True
    assert transport.reconnexions == 0


def test_ensure_connected_relance_quand_le_socket_est_tombe():
    import asyncio

    from einherjar.brokers.ctrader_adapter import CTraderAdapter

    adapter = CTraderAdapter("id", "secret", "token", 1, broker_name="spotware")
    transport = _TwistedFactice(connecte=False)
    adapter._twisted = transport  # type: ignore[assignment]
    adapter._connected = True

    assert asyncio.run(adapter.ensure_connected()) is True
    assert transport.reconnexions == 1


# ---------------------------------------------------------------------------
# 7. Ordres : volume de fermeture, remplissage du Fill, refus explicite
# ---------------------------------------------------------------------------


class _FauxClientParType:
    """Rend une reponse differente selon le message envoye."""

    def __init__(self, reponses: dict[type, object]) -> None:
        self.reponses = reponses
        self.envoyes: list[object] = []
        self.client_msg_ids: list[str] = []

    def send(self, request, clientMsgId=None, responseTimeoutInSeconds=5, **params):
        self.envoyes.append(request)
        self.client_msg_ids.append(clientMsgId)
        reponse = self.reponses.get(type(request))
        if reponse is None:
            raise AssertionError(f"requete inattendue: {type(request).__name__}")
        return _Enveloppe(reponse)


def _reconcile_avec_position(position_id: int = 291000228, volume_centiemes: int = 1):
    res = ProtoOAReconcileRes()
    position = res.position.add()
    position.positionId = position_id
    position.price = 85192.52
    position.tradeData.symbolId = 22395
    position.tradeData.volume = volume_centiemes
    position.tradeData.tradeSide = 1
    position.tradeData.label = "VALIDATION"
    _remplir_champs_requis(res)
    return res


def _thread_par_type(reponses: dict[type, object]) -> _CTraderTwistedThread:
    thread = _CTraderTwistedThread("host", 1, "id", "secret", "token", 4242, "spotware")
    thread._connected_event.set()
    thread._client = _FauxClientParType(reponses)  # type: ignore[assignment]
    thread._reactor = _FauxReactor()  # type: ignore[assignment]
    return thread


def test_send_request_utilise_un_client_msg_id_unique():
    """Regression : le defaut `str(id(deferred))` est recycle par Python.

    Une reponse tardive pouvait alors etre livree a une NOUVELLE requete (observe :
    un ProtoOAExecutionEvent recu en reponse a un Reconcile).
    """
    thread = _thread(_enveloppe(ProtoOAReconcileRes()))
    for _ in range(2):
        thread._send_request(ProtoOATraderReq(), ProtoOAReconcileRes).result(timeout=2.0)
    ids = thread._client.client_msg_ids  # type: ignore[union-attr]
    assert all(ids) and len(set(ids)) == 2


def test_ordre_refuse_par_order_error_event_leve_avec_la_raison():
    """Un ordre refuse revient en ProtoOAOrderErrorEvent, pas en ProtoOAErrorRes."""
    from ctrader_open_api.messages.OpenApiMessages_pb2 import ProtoOAOrderErrorEvent

    erreur = ProtoOAOrderErrorEvent()
    erreur.description = "Market is closed"
    _remplir_champs_requis(erreur)
    thread = _thread(_enveloppe(erreur))

    with pytest.raises(CTraderError, match="Market is closed"):
        thread._send_request(ProtoOATraderReq(), ProtoOAExecutionEvent).result(timeout=2.0)


def test_close_position_exige_un_volume():
    """`ProtoOAClosePositionReq.volume` est OBLIGATOIRE : sans lui la requete ne part pas.

    Regression : la requete n'etait meme pas serialisable
    (`EncodeError: missing required fields: volume`).
    """
    from ctrader_open_api.messages.OpenApiMessages_pb2 import ProtoOAClosePositionReq

    thread = _thread_par_type(
        {
            ProtoOAReconcileReq: _reconcile_avec_position(volume_centiemes=7),
            ProtoOAClosePositionReq: ProtoOAExecutionEvent(),
        }
    )
    assert thread.close_position_sync(291000228) is True
    requete = next(r for r in thread._client.envoyes if isinstance(r, ProtoOAClosePositionReq))  # type: ignore[union-attr]
    assert requete.volume == 7
    assert requete.positionId == 291000228


def test_close_position_sans_position_leve():
    vide = ProtoOAReconcileRes()
    _remplir_champs_requis(vide)
    thread = _thread_par_type({ProtoOAReconcileReq: vide})
    with pytest.raises(CTraderError, match="introuvable"):
        thread.close_position_sync(999)


def test_place_order_relit_la_position_quand_l_evenement_est_un_order_accepted():
    """Le 1er evenement d'un ordre marche est ORDER_ACCEPTED (prix/volume a 0).

    Regression : le Fill sortait avec qty=0.0 et prix=0.0 ; il faut relire la
    position ouverte par le broker.
    """
    from ctrader_open_api.messages.OpenApiMessages_pb2 import ProtoOANewOrderReq

    accepte = ProtoOAExecutionEvent()
    accepte.executionType = 2  # ORDER_ACCEPTED
    position = accepte.position
    position.positionId = 291000228
    _remplir_champs_requis(accepte)

    thread = _thread_par_type(
        {
            ProtoOANewOrderReq: accepte,
            ProtoOAReconcileReq: _reconcile_avec_position(volume_centiemes=1),
        }
    )
    # symboles + bornes de volume deja charges (evite une attente de 10 s sur le cache)
    thread._symbol_cache["BTCUSD"] = 22395
    thread._symbol_meta[22395] = {"name": "BTCUSD", "minVolume": 1, "stepVolume": 1, "digits": 3}
    ordre = Order(
        order_id="O1", asset="BTCUSD", order_type=OrderType.MARKET,
        direction=Direction.LONG, quantity=0.01, sl_price=77206.79, tp_price=85333.83,
        einher_name="TEST",
    )
    fill = thread.place_order_sync(ordre)

    assert fill.filled_qty == pytest.approx(0.01)
    assert fill.filled_price == pytest.approx(85192.52)
