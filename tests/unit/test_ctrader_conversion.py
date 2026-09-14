"""Tests de la conversion cTrader (trendbars -> OHLCV) et des regles d'ordre.

`ProtoOATrendbar` ne transmet pas de close : les prix sont des entiers en points
exprimes par rapport au close precedent. Ces tests verrouillent la formule
(l'ancien code lisait `bar.close`, inexistant -> tous les prix valaient 0) et
l'invariant OHLC, sans dependre de la librairie ctrader-open-api.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from einherjar.brokers.ctrader_adapter import CTraderError, trendbars_to_ohlcv


def _bar(low: int, d_open: int, d_close: int, d_high: int, minute: int, volume: int = 10):
    """Trendbar cTrader minimale (entiers en points)."""
    return SimpleNamespace(
        low=low,
        deltaOpen=d_open,
        deltaClose=d_close,
        deltaHigh=d_high,
        utcTimestampInMinutes=minute,
        volume=volume,
    )


def test_conversion_deltas_et_digits():
    """digits=2 : les deltas sont ajoutes au close precedent, divises par 100."""
    bars = [
        _bar(low=10_000, d_open=0, d_close=0, d_high=5, minute=60),      # reference (ecartee)
        _bar(low=10_050, d_open=10, d_close=50, d_high=60, minute=120),
    ]
    df = trendbars_to_ohlcv(bars, digits=2)

    assert df.height == 1, "la premiere bougie (sans close de reference) doit etre ecartee"
    row = df.row(0, named=True)
    assert row["open"] == pytest.approx(100.10)
    assert row["close"] == pytest.approx(100.50)
    assert row["low"] == pytest.approx(100.50)
    assert row["high"] == pytest.approx(101.10)
    assert row["timestamp"] == 120 * 60_000


def test_conversion_chaine_de_deltas():
    """Chaque bougie part du close de la PRECEDENTE (pas du sien)."""
    bars = [
        _bar(low=10_000, d_open=0, d_close=0, d_high=0, minute=60),
        _bar(low=10_100, d_open=0, d_close=100, d_high=100, minute=120),   # open 100.00 -> close 101.00
        _bar(low=10_150, d_open=20, d_close=30, d_high=40, minute=180),   # depuis 101.00
    ]
    df = trendbars_to_ohlcv(bars, digits=2)
    assert df.height == 2
    assert df["open"].to_list() == pytest.approx([100.00, 101.20])
    assert df["close"].to_list() == pytest.approx([101.00, 101.30])


def _serie_coherente(n: int = 20, digits: int = 2) -> list:
    """Trendbars construites depuis de vrais prix (invariant OHLC garanti)."""
    echelle = 10 ** digits
    bars = [_bar(low=10_000, d_open=0, d_close=0, d_high=0, minute=0)]
    close_prec = 100.00
    for i in range(1, n + 1):
        d_open_pts = (i % 7) - 3
        d_close_pts = (i % 5) - 2
        open_i = close_prec + d_open_pts / echelle
        close_i = close_prec + d_close_pts / echelle
        low_pts = int(round((min(open_i, close_i) - 0.05) * echelle))
        high_pts = int(round((max(open_i, close_i) + 0.10) * echelle))
        bars.append(
            _bar(
                low=low_pts,
                d_open=d_open_pts,
                d_close=d_close_pts,
                d_high=high_pts - low_pts,
                minute=60 * i,
            )
        )
        close_prec = close_i
    return bars


def test_invariant_ohlc_respecte():
    """high >= max(open, close) et low <= min(open, close) sur une serie coherente."""
    df = trendbars_to_ohlcv(_serie_coherente(30), digits=2)
    assert df.height == 30
    for row in df.iter_rows(named=True):
        assert row["high"] >= max(row["open"], row["close"]) - 1e-9
        assert row["low"] <= min(row["open"], row["close"]) + 1e-9
        assert row["low"] > 0.0


def test_digits_absent_refuse_de_produire_des_prix():
    """Sans digits, mieux vaut une erreur explicite que des prix faux."""
    with pytest.raises(CTraderError, match="digits"):
        trendbars_to_ohlcv([_bar(10_000, 0, 0, 0, 0)], digits=None)


def test_limit_garde_les_dernieres_bougies():
    bars = [_bar(low=10_000, d_open=0, d_close=0, d_high=0, minute=0)] + [
        _bar(low=10_000 + i, d_open=1, d_close=1, d_high=2, minute=60 * i) for i in range(1, 6)
    ]
    assert trendbars_to_ohlcv(bars, digits=2, limit=2).height == 2
    assert trendbars_to_ohlcv(bars, digits=2).height == 5


def test_liste_vide():
    df = trendbars_to_ohlcv([], digits=5)
    assert df.height == 0
