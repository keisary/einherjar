"""Test rapide de tous les brokers configures.

Usage:
    python scripts/test_brokers.py

Verifie la connectivite OHLCV pour chaque broker defini dans
config/credentials.json.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any
import json
import sys
from pathlib import Path

# Ajouter src au PYTHONPATH
SRC_PATH = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_PATH))

from einherjar.brokers.binance_adapter import BinanceAdapter
from einherjar.brokers.binance_futures_adapter import BinanceFuturesAdapter
from einherjar.brokers.oanda_adapter import OandaAdapter
from einherjar.brokers.alpaca_adapter import AlpacaAdapter

CRED_PATH = Path(__file__).resolve().parent.parent / "config" / "credentials.json"


async def test_broker(name: str, adapter, test_asset: str, test_tf: str = "1h") -> None:
    """Teste la recuperation OHLCV d'un broker."""
    print(f"\n=== {name} ===")
    try:
        df = await adapter.get_ohlcv(test_asset, test_tf, limit=5)
        if len(df) > 0:
            last_close = float(df["close"][-1])
            print(f"  OK   | {len(df)} bougies recuperees")
            print(f"       | Dernier close: {last_close}")
        else:
            print(f"  WARN | Aucune donnee retournee")
    except Exception as exc:
        print(f"  FAIL | {exc}")


async def main() -> None:
    """Point d'entree principal."""
    if not CRED_PATH.exists():
        print(f"[ERROR] credentials.json non trouve: {CRED_PATH}")
        print("        Copiez config/credentials.json.template vers config/credentials.json")
        print("        et remplissez vos cles API.")
        sys.exit(1)

    with open(CRED_PATH, encoding="utf-8") as f:
        creds = json.load(f)

    adapters: list[tuple[str, Any, str]] = []

    if "binance_spot" in creds:
        bc = creds["binance_spot"]
        adapters.append((
            "Binance Spot",
            BinanceAdapter(api_key=bc.get("api_key"), secret=bc.get("secret"), testnet=bc.get("testnet", True)),
            "BTCUSD",
        ))

    if "binance_futures" in creds:
        bf = creds["binance_futures"]
        adapters.append((
            "Binance Futures",
            BinanceFuturesAdapter(api_key=bf.get("api_key"), secret=bf.get("secret"), testnet=bf.get("testnet", True), leverage=bf.get("leverage", 1)),
            "BTCUSD",
        ))

    if "oanda" in creds:
        oa = creds["oanda"]
        adapters.append((
            "OANDA",
            OandaAdapter(account_id=oa.get("account_id"), access_token=oa.get("access_token"), practice=oa.get("practice", True)),
            "EURUSD",
        ))

    if "alpaca" in creds:
        al = creds["alpaca"]
        adapters.append((
            "Alpaca",
            AlpacaAdapter(api_key=al.get("api_key"), api_secret=al.get("api_secret"), paper=al.get("paper", True)),
            "AAPL",
        ))

    if not adapters:
        print("[WARN] Aucun broker configure dans credentials.json")
        sys.exit(0)

    print(f"[TEST] {len(adapters)} broker(s) a tester...")
    for name, adapter, asset in adapters:
        await test_broker(name, adapter, asset)

    print("\n[TEST] Termine.")


if __name__ == "__main__":
    asyncio.run(main())
