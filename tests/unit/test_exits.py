"""Tests des regles de sortie (duree de tenue) et de la fermeture de position.

Le broker gere les TP/SL ; EINHERJAR doit fermer les positions dont la duree de
tenue maximale (issue de `amplitude_bars` du corpus) est depassee, sinon elles
restent ouvertes indefiniment.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from einherjar.brokers.local_replay import LocalReplayBroker
from einherjar.core.enums import Direction, OrderType
from einherjar.core.models import Einher, Order
from einherjar.risk.exits import doit_fermer, parse_duree


# ---------------------------------------------------------------------------
# parse_duree
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("texte", "attendu"),
    [
        ("6h", timedelta(hours=6)),
        ("45m", timedelta(minutes=45)),
        ("3d", timedelta(days=3)),
        ("1h", timedelta(hours=1)),
        (" 4h ", timedelta(hours=4)),
    ],
)
def test_parse_duree_formats_corpus(texte: str, attendu: timedelta):
    assert parse_duree(texte) == attendu


def test_parse_duree_en_bougies_utilise_le_timeframe():
    """`N bars` (timeframe inconnu a la construction) se convertit avec le TF."""
    assert parse_duree("12 bars", ["15m"]) == timedelta(hours=3)
    assert parse_duree("10 bars", ["1h"]) == timedelta(hours=10)
    assert parse_duree("12 bars", None) is None, "sans timeframe, mieux vaut renoncer"
    assert parse_duree("12 bars", ["tf_inconnu"]) is None


@pytest.mark.parametrize("texte", [None, "", "bientot", "0h", "-3h", "6 hours", "h6"])
def test_parse_duree_refuse_le_non_interpretable(texte):
    assert parse_duree(texte) is None


# ---------------------------------------------------------------------------
# doit_fermer
# ---------------------------------------------------------------------------
def _einher(max_holding: str | None) -> Einher:
    return Einher(
        name="e1",
        domain="technical",
        direction="long",
        timeframes=["1h"],
        trigger="close > 0",
        assets="BTCUSD",
        max_holding=max_holding,
    )


def _position(ouverte_il_y_a: timedelta, tz_aware: bool = True):
    class _P:
        position_id = "POS_1"
        asset = "BTCUSD"
        einher_name = "e1"

    p = _P()
    base = datetime.now(UTC) - ouverte_il_y_a
    p.opened_at = base if tz_aware else base.replace(tzinfo=None)
    return p


def test_ferme_quand_la_duree_est_depassee():
    assert doit_fermer(_position(timedelta(hours=7)), _einher("6h")) == "duree_max_depassee"


def test_garde_la_position_avant_la_duree():
    assert doit_fermer(_position(timedelta(hours=1)), _einher("6h")) is None


def test_horodatage_naif_traite_comme_utc():
    assert doit_fermer(_position(timedelta(hours=7), tz_aware=False), _einher("6h")) == "duree_max_depassee"


def test_sans_duree_ni_ouverture_aucune_sortie():
    assert doit_fermer(_position(timedelta(days=10)), _einher(None)) is None
    assert doit_fermer(None, _einher("6h")) is None
    assert doit_fermer(_position(timedelta(hours=7)), None) is None


# ---------------------------------------------------------------------------
# Fermeture cote broker de rejeu
# ---------------------------------------------------------------------------
def _ordre() -> Order:
    return Order(
        order_id="ORD_TEST_1",
        asset="BTCUSD",
        direction=Direction.LONG,
        order_type=OrderType.MARKET,
        quantity=0.1,
        entry_price=100.0,
        tp_price=110.0,
        sl_price=95.0,
        einher_name="e1",
    )


def test_fermeture_position_replay_realise_le_pnl():
    broker = LocalReplayBroker(cash=100_000.0)
    asyncio.run(broker.place_order(_ordre()))

    positions = asyncio.run(broker.get_positions())
    assert len(positions) == 1
    identifiant = positions[0].position_id
    pnl_latent = positions[0].unrealized_pnl
    cash_avant = broker._cash

    assert asyncio.run(broker.close_position(identifiant)) is True
    assert asyncio.run(broker.get_positions()) == []
    assert broker._cash == pytest.approx(cash_avant + pnl_latent), "le P&L est realise dans le cash"


def test_fermeture_position_inconnue_renvoie_faux():
    broker = LocalReplayBroker(cash=100_000.0)
    assert asyncio.run(broker.close_position("POS_INEXISTANT")) is False
