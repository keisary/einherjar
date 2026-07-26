"""main.py -- Point d'entree unique du systeme EINHERJAR.

Fonctionnement:
1. Verification des composants (DuckDB, corpus, config)
2. Connexion cTrader (si credentials disponibles)
3. Affichage du statut en console
4. Lancement du serveur FastAPI (port 8000)
5. Lancement de l'inference loop (si broker connecte)
6. Affichage du lien vers le dashboard

Usage:
    python main.py

Le serveur API est accessible sur http://localhost:8000
Le dashboard est accessible sur http://localhost:3166 (via Vite dev server)
ou http://localhost:8000 (via FastAPI static files en production).
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger("einherjar.main")

# Chemins
PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config" / "settings.json"
CREDENTIALS_PATH = PROJECT_ROOT / "config" / "credentials.json"
CORPUS_PATH = PROJECT_ROOT / "config" / "corpus_v2.json"
DB_PATH = PROJECT_ROOT / "data" / "einherjar.db"
SRC_PATH = PROJECT_ROOT / "src"

# Ajouter src au PYTHONPATH
sys.path.insert(0, str(SRC_PATH))


class StatusChecker:
    """Verificateur de statut des composants du systeme."""

    def __init__(self) -> None:
        self.results: dict[str, dict[str, str]] = {}
        self.ctrader_adapter: Any | None = None
        self.demo_mode = False

    def check_all(self) -> bool:
        """Verifie tous les composants et retourne True si OK."""
        self.results = {}
        all_ok = True

        # 1. Config
        if CONFIG_PATH.exists():
            self.results["CONFIG"] = {"status": "OK", "path": str(CONFIG_PATH)}
        else:
            self.results["CONFIG"] = {"status": "MISSING", "path": str(CONFIG_PATH)}
            all_ok = False

        # 2. Corpus
        if CORPUS_PATH.exists():
            with open(CORPUS_PATH, encoding="utf-8") as f:
                data = json.load(f)
            count = len(data.get("einhers", data)) if isinstance(data, dict) else len(data)
            self.results["CORPUS"] = {"status": f"OK ({count} einhers)", "path": str(CORPUS_PATH)}
        else:
            self.results["CORPUS"] = {"status": "MISSING", "path": str(CORPUS_PATH)}
            all_ok = False

        # 3. DuckDB / DataStore
        try:
            from einherjar.data.store import DataStore
            store = DataStore(db_path=DB_PATH)
            store.conn.execute("SELECT 1")
            store.close()
            self.results["DATABASE"] = {"status": "OK", "path": str(DB_PATH)}
        except Exception as exc:
            self.results["DATABASE"] = {"status": f"ERROR: {exc}", "path": str(DB_PATH)}
            all_ok = False

        # 4. Feature Engine
        try:
            from einherjar.signals.feature_engine import FeatureEngine
            fe = FeatureEngine()
            self.results["FEATURE_ENGINE"] = {"status": "OK", "version": getattr(fe, "version", "1.0")}
        except Exception as exc:
            self.results["FEATURE_ENGINE"] = {"status": f"ERROR: {exc}"}
            all_ok = False

        # 5. Einher Engine
        try:
            from einherjar.signals.einher_engine import EinherEngine
            ee = EinherEngine()
            ee.load_corpus(str(CORPUS_PATH))
            self.results["EINHER_ENGINE"] = {"status": "OK", "loaded": len(ee.einhers)}
        except Exception as exc:
            self.results["EINHER_ENGINE"] = {"status": f"ERROR: {exc}"}
            all_ok = False

        # 6. Risk Manager
        try:
            from einherjar.risk.manager import RiskManager
            RiskManager()
            self.results["RISK_MANAGER"] = {"status": "OK"}
        except Exception as exc:
            self.results["RISK_MANAGER"] = {"status": f"ERROR: {exc}"}
            all_ok = False

        # 7. Scheduler
        try:
            from einherjar.scheduler.loop import InferenceLoop
            _ = InferenceLoop
            self.results["SCHEDULER"] = {"status": "OK"}
        except Exception as exc:
            self.results["SCHEDULER"] = {"status": f"ERROR: {exc}"}
            all_ok = False

        # 8. API Server
        try:
            from einherjar.api.server import app
            _ = app
            self.results["API_SERVER"] = {"status": "OK"}
        except Exception as exc:
            self.results["API_SERVER"] = {"status": f"ERROR: {exc}"}
            all_ok = False

        # 9. cTrader Connexion (optionnel mais affiche)
        self._check_ctrader()

        return all_ok

    def _check_ctrader(self) -> None:
        """Tente de connecter cTrader si credentials disponibles."""
        if not CREDENTIALS_PATH.exists():
            self.results["CTRADER"] = {"status": "NO CREDENTIALS (demo mode)", "path": str(CREDENTIALS_PATH)}
            self.demo_mode = True
            return

        try:
            with open(CREDENTIALS_PATH, encoding="utf-8") as f:
                creds = json.load(f)
            if not isinstance(creds, dict):
                raise ValueError("credentials.json must contain a JSON object")
        except (json.JSONDecodeError, ValueError) as exc:
            self.results["CTRADER"] = {"status": f"INVALID CREDENTIALS: {exc}", "path": str(CREDENTIALS_PATH)}
            self.demo_mode = True
            return

        try:
            from einherjar.brokers import CTraderAdapter
            self.ctrader_adapter = CTraderAdapter(
                client_id=creds.get("client_id", ""),
                client_secret=creds.get("client_secret", ""),
                access_token=creds.get("access_token", ""),
                account_id=int(creds.get("account_id", 0)),
                host=creds.get("host", "demo.ctraderapi.com"),
                port=int(creds.get("port", 5035)),
                broker_name=creds.get("broker_name", "ic_markets"),
            )
            connected = asyncio.run(self.ctrader_adapter.connect())
            if connected:
                acc = asyncio.run(self.ctrader_adapter.get_account())
                self.results["CTRADER"] = {
                    "status": f"OK | Equity=${acc.equity:,.2f} | Leverage={acc.leverage}x",
                    "host": self.ctrader_adapter.host,
                }
            else:
                self.results["CTRADER"] = {"status": "FAIL (connexion refused)", "host": self.ctrader_adapter.host}
                self.demo_mode = True
        except Exception as exc:
            self.results["CTRADER"] = {"status": f"ERROR: {exc}"}
            self.demo_mode = True

    def print_banner(self) -> None:
        """Affiche la banniere de statut."""
        print("\n" + "=" * 60)
        print("  EINHERJAR  --  Systeme de Trading Algorithmique")
        print("  Valhalla Protocol v2.0  (cTrader Cloud)")
        print("=" * 60)
        for name, info in self.results.items():
            status = info["status"]
            if status.startswith("OK"):
                icon = "  OK  "
            elif status.startswith("FAIL") or status.startswith("ERROR") or status.startswith("MISSING"):
                icon = " FAIL "
            else:
                icon = " WARN "
            print(f"  [{icon}] {name:20s} | {status}")
        print("=" * 60)
        if self.demo_mode:
            print("\n  [MODE DEMO] Aucun compte cTrader connecte.")
            print("  Creez config/credentials.json pour activer le trading live.")


# ---------------------------------------------------------------------------
# Mock broker pour mode demo (permet de tester l'inference loop sans compte)
# ---------------------------------------------------------------------------

class _MockBrokerAdapter:
    """Broker factice pour tests/demo. Retourne des bougies synthetiques."""

    name = "mock"

    def __init__(self) -> None:
        self._prices: dict[str, float] = {}
        self._seed_prices()

    def _seed_prices(self) -> None:
        from einherjar.brokers.broker_utils import ASSET_CLASS_MAP
        defaults = {
            "EURUSD": 1.0850, "GBPUSD": 1.2650, "USDJPY": 149.50,
            "AUDUSD": 0.6520, "USDCAD": 1.3520, "USDCHF": 0.8820,
            "EURGBP": 0.8570, "NZDUSD": 0.6120,
            "BTCUSD": 67500.0, "ETHUSD": 3520.0, "ADAUSD": 0.48,
            "BCHUSD": 230.0, "LTCUSD": 72.0,
            "AAPL": 185.0, "MSFT": 420.0, "NVDA": 880.0,
            "AMZN": 180.0, "GOOGL": 175.0, "TSLA": 175.0,
            "JPM": 195.0, "XOM": 105.0,
            "SP500": 5200.0, "NASDAQ100": 18500.0,
            "DOWJONES": 39000.0, "DAX40": 18200.0,
            "XAUUSD": 2320.0, "WTIUSD": 78.5,
            "BRENT": 82.0, "COPPER": 4.35,
        }
        for asset in ASSET_CLASS_MAP.keys():
            self._prices[asset] = defaults.get(asset, 100.0)

    async def get_ohlcv(self, asset: str, timeframe: str, since: int | None = None, limit: int = 500) -> pl.DataFrame:
        import random
        base = self._prices.get(asset, 100.0)
        # Genere 2 bougies synthetiques
        rows = []
        now = datetime.now(timezone.utc)
        for i in range(2):
            ts = int((now.timestamp() - (1 - i) * 300) * 1000)
            noise = (random.random() - 0.5) * base * 0.002
            close = base + noise
            open_p = close - (random.random() - 0.5) * base * 0.001
            high = max(open_p, close) + random.random() * base * 0.001
            low = min(open_p, close) - random.random() * base * 0.001
            rows.append([ts, open_p, high, low, close, 1000.0])
        return pl.DataFrame({
            "timestamp": [r[0] for r in rows],
            "open": [r[1] for r in rows],
            "high": [r[2] for r in rows],
            "low": [r[3] for r in rows],
            "close": [r[4] for r in rows],
            "volume": [r[5] for r in rows],
        })

    async def subscribe_live(self, assets: list[str], callback: callable) -> None:
        pass

    async def place_order(self, order: Any) -> Any:
        from einherjar.core.models import Fill
        return Fill(
            fill_id="MOCK_FILL",
            order_id=order.order_id,
            asset=order.asset,
            filled_qty=order.quantity,
            filled_price=order.entry_price or 100.0,
            fee=0.0,
        )

    async def cancel_order(self, order_id: str) -> bool:
        return True

    async def get_positions(self) -> list[Any]:
        return []

    async def get_account(self) -> Any:
        from einherjar.core.models import AccountState
        return AccountState(
            cash=100000.0,
            equity=100000.0,
            margin_used=0.0,
            margin_available=100000.0,
            leverage=100,
        )

    def get_fees(self, asset: str) -> dict[str, Any]:
        return {"spread_pct": 0.0001, "commission_per_lot": 0.0, "swap_long": 0.0, "swap_short": 0.0}


# ---------------------------------------------------------------------------
# Lancement async des services
# ---------------------------------------------------------------------------

async def start_api_server() -> None:
    """Lance le serveur FastAPI via uvicorn de maniere asynchrone."""
    import uvicorn
    from einherjar.api.server import app

    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=False,
    )
    server = uvicorn.Server(config)
    await server.serve()


async def start_inference_loop(checker: StatusChecker, use_mock: bool = False) -> None:
    """Lance la boucle d'inference live."""
    from einherjar.scheduler.loop import InferenceLoop
    from einherjar.signals.feature_engine import FeatureEngine
    from einherjar.signals.einher_engine import EinherEngine
    from einherjar.risk.manager import RiskManager
    from einherjar.data.live_store import LiveDataStore
    from einherjar.data.store import DataStore
    from einherjar.core.config import load_settings
    from einherjar.brokers.broker_utils import ASSET_CLASS_MAP

    system_config = load_settings(CONFIG_PATH)
    feature_engine = FeatureEngine()
    einher_engine = EinherEngine()
    einher_engine.load_corpus(str(CORPUS_PATH))
    risk_manager = RiskManager(system_config)
    live_store = LiveDataStore()
    data_store = DataStore(db_path=DB_PATH)

    assets = list(ASSET_CLASS_MAP.keys())
    timeframes = list(system_config.timeframes)
    assets_timeframes = [(a, tf) for a in assets for tf in timeframes]

    if use_mock or checker.demo_mode or checker.ctrader_adapter is None:
        broker = _MockBrokerAdapter()
        logger.info("InferenceLoop en mode DEMO (broker factice)")
    else:
        broker = checker.ctrader_adapter
        logger.info("InferenceLoop en mode LIVE (cTrader)")

    loop = InferenceLoop(
        broker=broker,
        assets_timeframes=assets_timeframes,
        feature_engine=feature_engine,
        einher_engine=einher_engine,
        risk_manager=risk_manager,
        live_store=live_store,
        data_store=data_store,
        config=system_config,
    )
    await loop.run()


def main() -> int:
    """Point d'entree principal.

    Returns:
        Code de sortie (0 = OK).
    """
    print("\n[INIT] Verification des composants...")
    checker = StatusChecker()
    ok = checker.check_all()
    checker.print_banner()

    if not ok:
        print("\n[ERROR] Certains composants sont manquants ou defectueux.")
        print("        Corrigez les erreurs avant de continuer.\n")
        return 1

    print("\n[SUCCES] Tous les composants sont operationnels.")
    print("\n" + "-" * 60)
    print("  LANCEMENT DES SERVICES")
    print("-" * 60)
    print("  API REST    : http://localhost:8000")
    print("  Health      : http://localhost:8000/api/health")
    print("  Account     : http://localhost:8000/api/account")
    print("  Dashboard   : http://localhost:3166  (Vite dev)")
    print("-" * 60 + "\n")

    async def _run_services() -> None:
        api_task = asyncio.create_task(start_api_server(), name="api")
        loop_task = asyncio.create_task(start_inference_loop(checker), name="inference")
        try:
            await asyncio.gather(api_task, loop_task)
        except asyncio.CancelledError:
            logger.info("Arret des services demande")
            api_task.cancel()
            loop_task.cancel()
            try:
                await api_task
            except asyncio.CancelledError:
                pass
            try:
                await loop_task
            except asyncio.CancelledError:
                pass

    try:
        asyncio.run(_run_services())
    except KeyboardInterrupt:
        logger.info("Interruption clavier detectee, arret du systeme")
    return 0


if __name__ == "__main__":
    sys.exit(main())
