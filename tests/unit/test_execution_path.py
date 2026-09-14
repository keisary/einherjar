"""Tests du chemin d'execution live : signal -> RiskManager -> ordre.

Couvre la chaine reellement utilisee par `InferenceLoop._run_cycle` (hors broker) :
`EinherEngine` construit un `Signal` (entree/TP/SL reels), `RiskManager.evaluate`
le dimensionne et le convertit en `Order` avec ses sorties. Aucune doublure : les
composants sont ceux de production.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from einherjar.core.enums import TimeFrame  # noqa: E402
from einherjar.core.models import AccountState, Einher, Order, Rejection, Signal  # noqa: E402
from einherjar.risk.manager import RiskManager  # noqa: E402
from einherjar.signals.einher_engine import EinherEngine  # noqa: E402


def _ohlcv(n: int = 300) -> pl.DataFrame:
    """OHLCV synthetique en tendance haussiere (close > 0 toujours vrai)."""
    rng = np.random.default_rng(7)
    close = 100.0 + np.cumsum(rng.normal(0.05, 0.5, n))
    high = close + 0.6
    low = close - 0.6
    return pl.DataFrame({
        "timestamp": pl.datetime_range(
            datetime(2025, 1, 1), datetime(2025, 12, 31), interval="1h", eager=True
        ).head(n),
        "open": close,
        "high": high,
        "low": low,
        "close": close,
        "volume": np.full(n, 1000.0),
        "atr_14": np.full(n, 1.5),
    })


def _account(equity: float = 100_000.0) -> AccountState:
    """Compte de test."""
    return AccountState(
        cash=equity, equity=equity, margin_used=0.0,
        margin_available=equity, leverage=100,
    )


def test_signal_puis_ordre_transmet_les_sorties():
    """Un einher declenche produit un Signal, et le risque un Order avec TP/SL."""
    df = _ohlcv()
    einher = Einher(
        name="test_einher", domain="test", direction="long", timeframes=["1h"],
        trigger="(close > 0)", filters=[], assets=["BTCUSD"],
        tp_rule={"type": "atr_multiple", "value": 2.5},
        sl_rule={"type": "atr_multiple", "value": 1.5},
    )
    engine = EinherEngine(einhers=[einher])
    signals, _ = engine.evaluate(df, "BTCUSD", TimeFrame.H1)
    assert len(signals) == 1, "le trigger trivial doit produire un signal"

    signal = signals[0]
    assert signal.entry_price > 0
    assert signal.tp_price > signal.entry_price, "TP au-dessus de l'entree pour un LONG"
    assert signal.sl_price < signal.entry_price, "SL sous l'entree pour un LONG"
    assert signal.tp_price - signal.entry_price == pytest.approx(2.5 * 1.5, rel=1e-6)
    assert signal.entry_price - signal.sl_price == pytest.approx(1.5 * 1.5, rel=1e-6)

    result = RiskManager().evaluate(signal, _account(), [])
    assert isinstance(result, Order), f"attendu un ordre, obtenu {result!r}"
    assert result.quantity > 0
    assert result.tp_price == signal.tp_price
    assert result.sl_price == signal.sl_price
    assert result.einher_name == "test_einher"
    assert result.direction == signal.direction
    assert result.status == "pending"


def test_short_a_un_tp_sous_l_entree():
    """Un einher SELL place le TP sous l'entree et le SL au-dessus."""
    df = _ohlcv()
    einher = Einher(
        name="test_short", domain="test", direction="short", timeframes=["1h"],
        trigger="(close > 0)", filters=[], assets=["BTCUSD"],
    )
    engine = EinherEngine(einhers=[einher])
    signals, _ = engine.evaluate(df, "BTCUSD", TimeFrame.H1)
    signal = signals[0]
    assert signal.tp_price < signal.entry_price
    assert signal.sl_price > signal.entry_price


def test_compte_vide_est_rejete():
    """Sans capital, le dimensionnement echoue : Rejection, jamais un ordre a 0."""
    signal = Signal(
        asset="BTCUSD", direction=__import__("einherjar.core.enums", fromlist=["Direction"]).Direction.LONG,
        timeframe=TimeFrame.H1, einher_name="test", entry_price=100.0,
        tp_price=105.0, sl_price=99.0, confidence=0.8,
    )
    result = RiskManager().evaluate(signal, _account(0.0), [])
    assert isinstance(result, Rejection)
    assert result.reason


def test_sizing_respecte_le_risque_par_trade():
    """volume ~ (equity x risk_per_trade x confiance) / distance_SL, arrondi au lot."""
    rm = RiskManager()
    signal = Signal(
        asset="BTCUSD", direction=__import__("einherjar.core.enums", fromlist=["Direction"]).Direction.LONG,
        timeframe=TimeFrame.H1, einher_name="test", entry_price=100.0,
        tp_price=110.0, sl_price=95.0, confidence=1.0,
    )
    order = rm.evaluate(signal, _account(100_000.0), [])
    assert isinstance(order, Order)
    attendu = (100_000.0 * rm.risk_per_trade * 1.0) / 5.0
    assert order.quantity == pytest.approx(round(attendu, 4), rel=1e-6)


def test_signal_sans_prix_est_rejete_pour_prix_manquant():
    """Un signal sans prix d'entree est rejete : pas de controle de marge sur 1.0."""
    signal = Signal(
        asset="BTCUSD", direction=__import__("einherjar.core.enums", fromlist=["Direction"]).Direction.LONG,
        timeframe=TimeFrame.H1, einher_name="test", entry_price=0.0,
        tp_price=110.0, sl_price=95.0, confidence=1.0,
    )
    result = RiskManager().evaluate(signal, _account(100_000.0), [])
    assert isinstance(result, Rejection)
    assert result.reason == "price_missing"
