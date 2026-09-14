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
CORPUS_PATH = PROJECT_ROOT / "outputs" / "corpus.jsonl"
DB_PATH = PROJECT_ROOT / "data" / "einherjar.db"
SRC_PATH = PROJECT_ROOT / "src"

# Ajouter src au PYTHONPATH
sys.path.insert(0, str(SRC_PATH))


def _count_corpus(path: Path) -> int:
    """Compte les einhers d'un corpus (JSONL une ligne par einher, ou JSON historique)."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    stripped = text.lstrip()
    if stripped.startswith("[") or stripped.startswith("{"):
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            pass
        else:
            return len(data.get("einhers", data)) if isinstance(data, dict) else len(data)
    count = 0
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            json.loads(line)
        except json.JSONDecodeError:
            continue
        count += 1
    return count


class StatusChecker:
    """Verificateur de statut des composants du systeme."""

    def __init__(self) -> None:
        self.results: dict[str, dict[str, str]] = {}
        self.ctrader_adapter: Any | None = None
        # environment = compte cTrader utilise ('demo' ou 'live') ; broker_ready =
        # connexion effective. Sans broker connecte, main() refuse de demarrer.
        self.environment = "demo"
        self.broker_ready = False

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

        # 2. Corpus (JSONL du moteur de recherche ou JSON historique)
        if CORPUS_PATH.exists():
            count = _count_corpus(CORPUS_PATH)
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

        # 4. Pipeline de features (schema MIDAS = noms du corpus)
        try:
            from einherjar.signals.feature_pipeline import FeaturePipeline

            fp = FeaturePipeline()
            self.results["FEATURE_ENGINE"] = {
                "status": "OK",
                "version": "pipeline-schema-midas",
                "lookback": fp.max_lookback,
            }
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
        """Verifie les credentials cTrader et tente la connexion (demo ou live)."""
        if not CREDENTIALS_PATH.exists():
            self.results["CTRADER"] = {
                "status": "ABSENT : renseigner config/credentials.json",
                "path": str(CREDENTIALS_PATH),
            }
            self.broker_ready = False
            return

        try:
            with open(CREDENTIALS_PATH, encoding="utf-8") as f:
                creds = json.load(f)
            if not isinstance(creds, dict):
                raise ValueError("credentials.json must contain a JSON object")
        except (json.JSONDecodeError, ValueError) as exc:
            self.results["CTRADER"] = {
                "status": f"INVALID CREDENTIALS: {exc}",
                "path": str(CREDENTIALS_PATH),
            }
            self.broker_ready = False
            return

        # ENVIRONNEMENT : 'demo' (compte demo cTrader) ou 'live'. Le host et le
        # compte diffèrent, le code et le point d'entree des donnees sont identiques.
        env = str(creds.get("environment") or "").strip().lower()
        if env not in ("demo", "live"):
            env = "live" if "live" in str(creds.get("host", "")).lower() else "demo"
        self.environment = env
        host = str(creds.get("host") or ("live.ctraderapi.com" if env == "live"
                                        else "demo.ctraderapi.com"))
        manquants = [k for k in ("client_id", "client_secret", "access_token") if not creds.get(k)]
        if manquants or not int(creds.get("account_id", 0)):
            self.results["CTRADER"] = {
                "status": f"INCOMPLET [{env}] : {', '.join(manquants)}"
                          + ("" if int(creds.get("account_id", 0)) else ", account_id"),
                "path": str(CREDENTIALS_PATH),
            }
            self.broker_ready = False
            return

        try:
            from einherjar.brokers import CTraderAdapter
            self.ctrader_adapter = CTraderAdapter(
                client_id=creds.get("client_id", ""),
                client_secret=creds.get("client_secret", ""),
                access_token=creds.get("access_token", ""),
                account_id=int(creds.get("account_id", 0)),
                host=host,
                port=int(creds.get("port", 5035)),
                broker_name=creds.get("broker_name", "ic_markets"),
            )
            connected = asyncio.run(self.ctrader_adapter.connect())
            if connected:
                acc = asyncio.run(self.ctrader_adapter.get_account())
                self.results["CTRADER"] = {
                    "status": f"OK [{env}] | Equity=${acc.equity:,.2f} | Leverage={acc.leverage}x",
                    "host": self.ctrader_adapter.host,
                }
                self.broker_ready = True
            else:
                self.results["CTRADER"] = {
                    "status": f"FAIL [{env}] (connexion refusee)",
                    "host": self.ctrader_adapter.host,
                }
                self.broker_ready = False
        except Exception as exc:
            self.results["CTRADER"] = {"status": f"ERROR [{env}]: {exc}"}
            self.broker_ready = False

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
        if not self.broker_ready:
            print(f"\n  [COMPTE {self.environment.upper()}] Aucun broker cTrader connecte.")
            print("  Renseignez config/credentials.json :")
            print('    environment   : "demo" (compte demo cTrader) ou "live"')
            print('    host          : demo.ctraderapi.com  |  live.ctraderapi.com')
            print("    account_id, client_id, client_secret, access_token")
            print("  Le systeme NE demarre PAS sans broker : il n'existe aucun mode de")
            print("  rejeu local en production (donnees et execution = cTrader uniquement).")
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


async def start_inference_loop(checker: StatusChecker) -> None:
    """Lance la boucle d'inference sur le compte cTrader connecte (demo ou live)."""
    from einherjar.scheduler.loop import InferenceLoop
    from einherjar.signals.feature_pipeline import FeaturePipeline
    from einherjar.signals.einher_engine import EinherEngine
    from einherjar.risk.manager import RiskManager
    from einherjar.data.live_store import LiveDataStore
    from einherjar.data.store import DataStore
    from einherjar.core.config import load_settings
    from einherjar.brokers.broker_utils import ASSET_CLASS_MAP

    system_config = load_settings(CONFIG_PATH)
    # Pipeline de features = schema MIDAS complet (noms `feature_ref` du corpus) :
    # les conditions du corpus ne sont evaluables qu'avec ces colonnes.
    feature_engine = FeaturePipeline(max_lookback=1500)
    einher_engine = EinherEngine()
    einher_engine.load_corpus(str(CORPUS_PATH))
    risk_manager = RiskManager(system_config)
    # La fenetre du store live doit couvrir le lookback du pipeline, sinon le
    # recalcul de chaque bougie se ferait sur un historique tronque.
    live_store = LiveDataStore(window_size=feature_engine.max_lookback)
    data_store = DataStore(db_path=DB_PATH)

    # Couples (asset, timeframe) A EVALUER : ceux couverts par le corpus.
    # Prendre le produit cartesien complet (29 actifs x 5 TF = 145 couples) ferait
    # tourner des couples sans aucun einher et ne tiendrait pas les fenetres
    # (le calcul de features coute ~90 s par couple).
    from einherjar.signals.corpus_bridge import universe_index

    timeframes = list(system_config.timeframes)
    assets_timeframes = [
        (asset, tf)
        for asset, tf in universe_index(CORPUS_PATH).keys()
        if tf in timeframes
    ]
    if not assets_timeframes:
        assets = list(ASSET_CLASS_MAP.keys())
        assets_timeframes = [(a, tf) for a in assets for tf in timeframes]
        logger.warning("Corpus sans couple exploitable : repli sur le produit complet")
    else:
        logger.info(
            "Couples (asset, tf) issus du corpus : %d (au lieu de %d possibles)",
            len(assets_timeframes),
            len(ASSET_CLASS_MAP) * len(timeframes),
        )

    # POINT D'ENTREE DE DONNEES UNIQUE : le compte cTrader (demo ou live selon
    # `environment`). Aucun mode "demo local" : sans broker connecte, le systeme
    # ne demarre pas (voir `main()`), au lieu de rejouer des donnees locales.
    if checker.ctrader_adapter is None or not checker.broker_ready:
        raise RuntimeError(
            "Broker cTrader non connecte : renseignez config/credentials.json "
            "(environment, host, account_id, client_id, client_secret, access_token)."
        )
    broker = checker.ctrader_adapter
    logger.info("InferenceLoop sur compte %s (%s)", checker.environment.upper(), broker.host)

    # Ne garder que les couples que le broker sert REELLEMENT : un actif absent
    # echouerait a chaque cycle (aucune feature, aucun signal) et polluerait les
    # erreurs. Detail verifiable hors ligne : scripts/ctrader_verifier_univers.py.
    try:
        from einherjar.brokers.broker_utils import normalize_symbol

        symboles_broker = await broker.get_symboles_disponibles()
        if symboles_broker:
            broker_name = str(getattr(broker, "broker_name", "ic_markets"))
            indisponibles = sorted(
                {
                    asset
                    for asset, _ in assets_timeframes
                    if normalize_symbol(asset, broker_name).upper() not in symboles_broker
                }
            )
            if indisponibles:
                avant = len(assets_timeframes)
                assets_timeframes = [(a, tf) for a, tf in assets_timeframes if a not in indisponibles]
                logger.warning(
                    "Actifs absents chez le broker : %s -> %d couple(s) ecarte(s) (%d restants)",
                    ", ".join(indisponibles),
                    avant - len(assets_timeframes),
                    len(assets_timeframes),
                )
            else:
                logger.info(
                    "Univers broker verifie : %d symboles, tous les actifs du corpus sont disponibles",
                    len(symboles_broker),
                )
        else:
            logger.warning("Liste de symboles broker vide : univers non filtre")
    except Exception as exc:  # noqa: BLE001 - le filtrage ne doit pas bloquer le demarrage
        logger.warning("Verification de l'univers broker impossible (%s)", exc)

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

    # POINT D'ENTREE UNIQUE (donnees + execution = le compte cTrader). Sans broker
    # connecte on s'arrete ici : aucun mode de rejeu local n'existe en production.
    if not checker.broker_ready:
        print(f"\n[ARRET] Compte cTrader {checker.environment.upper()} non connecte.")
        print("        Renseignez config/credentials.json puis relancez.")
        print("        (donnees ET execution viennent de cTrader : pas de mode local)\n")
        return 2

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
