#!/usr/bin/env python3
"""test_ctrader.py — Script de validation de la connexion cTrader Open API.

Usage:
    python scripts/test_ctrader.py

Pre-requis:
    pip install ctrader-open-api
    Avoir un fichier config/credentials.json valide (voir docs/GUIDE_CTRADER.md)

Le script tente :
1. Connexion gRPC + auth application + auth account
2. Recuperation du solde/equity/marge
3. Recuperation des positions ouvertes
4. Recuperation de 5 bougies M5 sur EURUSD
5. Affichage du statut de l'adapter

Reference : docs/PLAN_REFONTE_CTRADER.md
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))

CREDENTIALS_PATH = PROJECT_ROOT / "config" / "credentials.json"


def load_credentials() -> dict:
    if not CREDENTIALS_PATH.exists():
        print(f"[ERREUR] {CREDENTIALS_PATH} introuvable.")
        print("         Creez-le d'abord (voir docs/GUIDE_CTRADER.md)")
        sys.exit(1)
    with open(CREDENTIALS_PATH, encoding="utf-8") as f:
        return json.load(f)


async def main() -> int:
    print("=" * 60)
    print("  EINHERJAR  —  Test cTrader Open API")
    print("=" * 60)

    creds = load_credentials()

    try:
        from einherjar.brokers import CTraderAdapter
    except ImportError as exc:
        print(f"[ERREUR] Importer CTraderAdapter : {exc}")
        sys.exit(1)

    adapter = CTraderAdapter(
        client_id=creds.get("client_id", ""),
        client_secret=creds.get("client_secret", ""),
        access_token=creds.get("access_token", ""),
        account_id=int(creds.get("account_id", 0)),
        host=creds.get("host", "demo.ctraderapi.com"),
        port=int(creds.get("port", 5035)),
        broker_name=creds.get("broker_name", "ic_markets"),
    )

    print("\n[1/4] Connexion cTrader...")
    connected = await adapter.connect()
    if not connected:
        print("[FAIL] Connexion echouee. Verifiez vos credentials et la connexion reseau.")
        return 1
    print("[OK]  Connecte")

    print("\n[2/4] Recuperation du compte...")
    try:
        account = await adapter.get_account()
        print(f"  Cash      : {account.cash:,.2f} USD")
        print(f"  Equity    : {account.equity:,.2f} USD")
        print(f"  Margin    : {account.margin_used:,.2f} USD")
        print(f"  Margin Free: {account.margin_available:,.2f} USD")
        print(f"  Leverage  : {account.leverage}:1")
    except Exception as exc:
        print(f"[WARN] get_account echoue : {exc}")

    print("\n[3/4] Recuperation des positions...")
    try:
        positions = await adapter.get_positions()
        if positions:
            print(f"  {len(positions)} position(s) ouverte(s)")
            for p in positions:
                print(f"    - {p.asset} {p.direction.value} {p.quantity} @ {p.avg_entry_price}")
        else:
            print("  Aucune position ouverte")
    except Exception as exc:
        print(f"[WARN] get_positions echoue : {exc}")

    print("\n[4/4] Recuperation OHLCV EURUSD M5 (5 dernieres bougies)...")
    try:
        import polars as pl
        df = await adapter.get_ohlcv("EURUSD", "5m", limit=5)
        if len(df) > 0:
            print(f"  {len(df)} bougies recuperees")
            print(df.tail(3).to_pandas().to_string(index=False))
        else:
            print("  Aucune bougie retournee")
    except Exception as exc:
        print(f"[WARN] get_ohlcv echoue : {exc}")

    print("\n[STATUT ADAPTER]")
    print(json.dumps(adapter.get_status(), indent=2))

    print("\n[5/5] Deconnexion...")
    await adapter.disconnect()
    print("[OK]  Deconnecte")

    print("\n" + "=" * 60)
    print("  TEST TERMINE AVEC SUCCES")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
