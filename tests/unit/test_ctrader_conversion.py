"""Conversion des trendbars cTrader -> OHLCV.

`ProtoOATrendbar` ne transmet PAS de prix open/close/high : la bougie porte
`low` (absolu, en points) et trois ecarts PAR RAPPORT A CE LOW :

    open  = low + deltaOpen
    close = low + deltaClose
    high  = low + deltaHigh

avec 1 point = 1/100000 d'unite de prix (`CTRADER_POINT_SCALE`).

Ces tests verrouillent la formule ET l'anti-derive : la version precedente
chainait les deltas d'une bougie a l'autre (`close = close_precedent + deltaClose`),
ce qui faisait s'eloigner les prix proportionnellement au nombre de bougies
(mesure sur le compte reel : EURUSD 1h a 1.2445 au lieu de 1.1488 sur 200 bougies,
BTCUSD a 116910 au lieu de 80865). L'assertion qui attrape cette classe de bug est
`close - low == deltaClose / 100000` pour CHAQUE bougie.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from einherjar.brokers.ctrader_adapter import CTRADER_POINT_SCALE, trendbars_to_ohlcv

ECHELLE = CTRADER_POINT_SCALE


def _bar(low: int, d_open: int, d_close: int, d_high: int, minute: int, volume: int = 10):
    """Trendbar cTrader minimale (entiers en points, 1 point = 1/100000)."""
    return SimpleNamespace(
        low=low,
        deltaOpen=d_open,
        deltaClose=d_close,
        deltaHigh=d_high,
        utcTimestampInMinutes=minute,
        volume=volume,
    )


def test_bougie_reconstruite_depuis_son_low():
    """Cas reel EURUSD 1h (compte demo, 2026-09-19)."""
    bar = _bar(low=114_858, d_open=6, d_close=21, d_high=32, minute=29_829_360)
    df = trendbars_to_ohlcv([bar])

    assert df.height == 1, "aucune bougie ne doit etre ecartee (pas de chainage)"
    row = df.row(0, named=True)
    assert row["low"] == pytest.approx(1.14858)
    assert row["open"] == pytest.approx(1.14864)
    assert row["close"] == pytest.approx(1.14879)
    assert row["high"] == pytest.approx(1.14890)
    assert row["timestamp"] == 29_829_360 * 60_000


def test_echelle_un_point_sur_cent_mille():
    """L'echelle est fixe (1/100000), PAS `10 ** digits` du symbole.

    Mesures du compte reel : APPLE low=33_508_000 -> 335.08 (digits=2),
    US 500 low=764_500_000 -> 7645.00, BTCUSD low=8_086_545_000 -> 80865.45.
    """
    df = trendbars_to_ohlcv(
        [
            _bar(low=33_508_000, d_open=15_000, d_close=1_000, d_high=127_000, minute=1),
            _bar(low=764_500_000, d_open=50_000, d_close=1_240_000, d_high=1_700_000, minute=2),
            _bar(low=8_086_545_000, d_open=25_587_000, d_close=19_509_000, d_high=32_884_000, minute=3),
        ]
    )
    assert df["low"].to_list() == pytest.approx([335.08, 7645.00, 80865.45])
    assert df["close"].to_list() == pytest.approx([335.09, 7657.40, 81060.54])


def _serie_sans_derive(n: int = 200, low_depart: int = 80_000_000) -> list:
    """Serie ou chaque bougie a son propre low : aucune accumulation possible."""
    bars = []
    low = low_depart
    for i in range(n):
        low += 100_000  # +1.00 par bougie
        bars.append(_bar(low=low, d_open=30_000, d_close=50_000, d_high=90_000, minute=60 * i))
    return bars


def test_les_prix_ne_derivent_pas_avec_la_taille_de_la_fenetre():
    """Regression : le chainage faisait deriver les prix avec le nombre de bougies."""
    for n in (3, 10, 50, 200):
        df = trendbars_to_ohlcv(_serie_sans_derive(n))
        assert df.height == n
        # close = low + deltaClose / 100000, quelle que soit la position de la bougie
        for row in df.iter_rows(named=True):
            assert row["close"] - row["low"] == pytest.approx(50_000 / ECHELLE)
            assert row["open"] - row["low"] == pytest.approx(30_000 / ECHELLE)
            assert row["high"] - row["low"] == pytest.approx(90_000 / ECHELLE)


def test_invariant_ohlc_respecte():
    """low <= min(open, close) et high >= max(open, close) sur la serie complete."""
    df = trendbars_to_ohlcv(_serie_sans_derive(200))
    for row in df.iter_rows(named=True):
        assert row["low"] <= min(row["open"], row["close"]) + 1e-12
        assert row["high"] >= max(row["open"], row["close"]) - 1e-12
        assert row["low"] > 0.0


def test_limit_garde_les_dernieres_bougies():
    bars = _serie_sans_derive(5)
    assert trendbars_to_ohlcv(bars, limit=2).height == 2
    assert trendbars_to_ohlcv(bars).height == 5
    # ce sont bien les 2 DERNIERES
    assert trendbars_to_ohlcv(bars, limit=2)["timestamp"].to_list() == [
        bars[-2].utcTimestampInMinutes * 60_000,
        bars[-1].utcTimestampInMinutes * 60_000,
    ]


def test_liste_vide():
    df = trendbars_to_ohlcv([])
    assert df.height == 0
    assert df.columns == ["timestamp", "open", "high", "low", "close", "volume"]
