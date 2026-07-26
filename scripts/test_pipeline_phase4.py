"""
Test integre du pipeline EINHERJAR — Phase 4.

Simule un cycle complet : fetch -> features -> einher -> risk -> paper broker -> journal.
Objectif : valider que tous les composants se connectent correctement.

Usage :
  cd D:/midas_v2/einherjar
  python -m scripts.test_pipeline_phase4
"""

import asyncio
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(ROOT))

import numpy as np
import polars as pl

from einherjar.brokers.paper_broker import PaperBroker
from einherjar.core.config import SystemConfig, RiskLimits, ValidationConfig
from einherjar.core.enums import Direction, TimeFrame
from einherjar.core.models import Einher, Signal
from einherjar.data.store import DataStore
from einherjar.risk.manager import RiskManager
from einherjar.signals.einher_engine import EinherEngine


def create_dummy_ohlcv(n: int = 100) -> pl.DataFrame:
    """Genere des donnees OHLCV de test avec features."""
    np.random.seed(42)
    base_time = datetime(2026, 7, 20, 10, 0, 0, tzinfo=timezone.utc)
    timestamps = [base_time + timedelta(minutes=i) for i in range(n)]

    base = 100.0
    closes = [base]
    for _ in range(1, n):
        closes.append(closes[-1] * (1 + np.random.normal(0, 0.01)))

    closes = np.array(closes)
    opens = closes * (1 + np.random.normal(0, 0.002, n))
    highs = np.maximum(opens, closes) * (1 + np.abs(np.random.normal(0, 0.005, n)))
    lows = np.minimum(opens, closes) * (1 - np.abs(np.random.normal(0, 0.005, n)))
    volumes = np.random.uniform(1000, 10000, n)
    atrs = np.abs(np.random.normal(1.5, 0.5, n))

    # Generer quelques patterns hammer (1) aleatoirement
    hammers = np.zeros(n, dtype=int)
    for i in range(10, n):
        if np.random.random() < 0.05:
            hammers[i] = 1

    df = pl.DataFrame({
        "timestamp": timestamps,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
        "atr_14": atrs,
        "pattern_hammer": hammers,
    })
    return df
    """Genere des donnees OHLCV de test avec features."""
    np.random.seed(42)
    base_time = datetime(2026, 7, 20, 10, 0, 0, tzinfo=timezone.utc)
    timestamps = [base_time + timedelta(minutes=i) for i in range(n)]
    """Genere des donnees OHLCV de test avec features."""
    np.random.seed(42)
    base = 100.0
    timestamps = [datetime(2026, 7, 20, 10, 0, 0, tzinfo=timezone.utc)]
    for i in range(1, n):
        timestamps.append(datetime(2026, 7, 20, 10, i, 0, tzinfo=timezone.utc))

    closes = [base]
    for _ in range(1, n):
        closes.append(closes[-1] * (1 + np.random.normal(0, 0.01)))

    closes = np.array(closes)
    opens = closes * (1 + np.random.normal(0, 0.002, n))
    highs = np.maximum(opens, closes) * (1 + np.abs(np.random.normal(0, 0.005, n)))
    lows = np.minimum(opens, closes) * (1 - np.abs(np.random.normal(0, 0.005, n)))
    volumes = np.random.uniform(1000, 10000, n)
    atrs = np.abs(np.random.normal(1.5, 0.5, n))

    # Generer quelques patterns hammer (1) aleatoirement
    hammers = np.zeros(n, dtype=int)
    for i in range(10, n):
        if np.random.random() < 0.05:
            hammers[i] = 1

    df = pl.DataFrame({
        "timestamp": timestamps,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
        "atr_14": atrs,
        "pattern_hammer": hammers,
    })
    return df


def create_test_corpus() -> list[Einher]:
    """Cree un corpus de demo pour le test."""
    return [
        Einher(
            name="E_HAMMER_demo",
            domain="Pattern pur",
            direction="long",
            timeframes=["5m"],
            trigger="pattern_hammer == 1",
            filters=[],
            assets="all",
            tp_rule={"type": "atr_multiple", "value": 2.5},
            sl_rule={"type": "atr_multiple", "value": 1.5},
            max_holding="1h",
            cooldown="4h",
        ),
        Einher(
            name="E_HAMMER_RSI_filter_demo",
            domain="Pattern + confluence",
            direction="long",
            timeframes=["5m"],
            trigger="pattern_hammer == 1",
            filters=[{"expr": "close > 100"}],
            assets="all",
            tp_rule={"type": "atr_multiple", "value": 2.5},
            sl_rule={"type": "atr_multiple", "value": 1.5},
            max_holding="1h",
            cooldown="4h",
        ),
    ]


async def main():
    print("=" * 60)
    print("  EINHERJAR — Test Pipeline Phase 4")
    print("=" * 60)

    # 1. Initialisation des composants
    db_path = Path(__file__).resolve().parent.parent / "data" / "test_phase4.db"
    store = DataStore(db_path)
    broker = PaperBroker("paper_test", "binance", initial_balance=10000.0)
    risk_mgr = RiskManager(SystemConfig(
        risk_limits=RiskLimits(),
        validation_config=ValidationConfig(),
        risk_per_trade=0.01,
    ))
    engine = EinherEngine(create_test_corpus())

    # 2. Generer des donnees de test
    df = create_dummy_ohlcv(200)
    print(f"\n[1] Donnees generees : {len(df)} bougies")

    # 3. Simuler quelques cycles
    asset = "BTCUSD"
    tf = TimeFrame.M5

    triggered_total = 0
    forming_total = 0
    orders_total = 0
    rejections_total = 0
    fills_total = 0

    # On prend les bougies par groupes de 5 pour simuler des clotures
    for i in range(5, len(df), 5):
        window = df[:i]
        last_price = float(window["close"][-1])

        # Mettre a jour le prix du broker
        fills_from_sl_tp = await broker.update_price(asset, last_price)
        fills_total += len(fills_from_sl_tp)

        # Evaluer les Einhers
        signals, forming = engine.evaluate(window, asset, tf)
        triggered_total += len(signals)
        forming_total += len(forming)

        # Journaliser les signaux
        for sig in signals:
            store.append_signal(sig)

        # Journaliser les forming
        for f in forming:
            print(f"  FORMING : {f.name} (asset={asset}, tf={tf.value})")

        # Passer au Risk Manager
        account = await broker.get_account()
        positions = await broker.get_positions()

        for sig in signals:
            result = risk_mgr.evaluate(sig, account, positions)
            from einherjar.core.models import Order, Rejection

            if isinstance(result, Order):
                orders_total += 1
                store.append_order(result)
                fill = await broker.place_order(result)
                fills_total += 1
                store.append_fill(fill)
                print(f"  EXEC    : {sig.einher_name} | qty={result.quantity:.4f} | price={fill.filled_price:.2f}")
            elif isinstance(result, Rejection):
                rejections_total += 1
                store.append_rejection(result)
                print(f"  REJECT  : {sig.einher_name} | {result.reason}")

        # Snapshot equity
        account = await broker.get_account()
        store.snapshot_equity(account)

    # 4. Resume
    print(f"\n[2] Resume simulation")
    print(f"  Signaux triggered : {triggered_total}")
    print(f"  Einhers forming   : {forming_total}")
    print(f"  Ordres acceptes   : {orders_total}")
    print(f"  Rejections        : {rejections_total}")
    print(f"  Fills totaux      : {fills_total}")
    print(f"  Positions finales : {len(broker.positions)}")
    print(f"  Balance finale    : {broker.balance:.2f}")
    print(f"  P&L realise       : {broker._realized_pnl:.2f}")

    # 5. Verifier la base de donnees
    print(f"\n[3] Verification base de donnees ({db_path})")
    signals_db = store.conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
    orders_db = store.conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
    fills_db = store.conn.execute("SELECT COUNT(*) FROM fills").fetchone()[0]
    rejections_db = store.conn.execute("SELECT COUNT(*) FROM rejections").fetchone()[0]
    equity_db = store.conn.execute("SELECT COUNT(*) FROM equity_curve").fetchone()[0]

    print(f"  signals    : {signals_db}")
    print(f"  orders     : {orders_db}")
    print(f"  fills      : {fills_db}")
    print(f"  rejections : {rejections_db}")
    print(f"  equity     : {equity_db}")

    store.close()
    print("\n[4] Test Phase 4 OK — Pipeline integre fonctionnel.")


if __name__ == "__main__":
    asyncio.run(main())
