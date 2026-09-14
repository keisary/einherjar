#!/usr/bin/env python
"""Test de la boucle d'inference avec la DOUBLURE de rejeu local (hors broker reel).

Ce script n'est PAS la demo : en production, donnees et execution passent
exclusivement par `CTraderAdapter` (compte demo ou live selon `environment` dans
config/credentials.json). Ici on remplace le broker par `LocalReplayBroker`
(rejeu des CSV locaux + compte papier) pour tester l'orchestration :
    amortage LiveDataStore -> bougie -> features -> EinherEngine -> confluence
    -> RiskManager -> ordre simule

Usage :
    python scripts/smoke_loop_replay.py --asset BTCUSD --timeframe 1h
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import tempfile
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import main as app  # noqa: E402  (chemins de reference : CONFIG_PATH, CORPUS_PATH)
from einherjar.brokers.local_replay import LocalReplayBroker  # noqa: E402
from einherjar.core.config import load_settings  # noqa: E402
from einherjar.core.enums import TimeFrame  # noqa: E402
from einherjar.data.live_store import LiveDataStore  # noqa: E402
from einherjar.data.store import DataStore  # noqa: E402
from einherjar.risk.manager import RiskManager  # noqa: E402
from einherjar.scheduler.loop import InferenceLoop  # noqa: E402
from einherjar.signals.einher_engine import EinherEngine  # noqa: E402
from einherjar.signals.feature_pipeline import FeaturePipeline  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")


async def run(asset: str, timeframe: str, lookback: int) -> int:
    """Execute l'amorcage puis un cycle d'inference complet sur un couple."""
    system_config = load_settings(app.CONFIG_PATH)

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        store = DataStore(db_path=Path(tmp) / "einherjar.db")
        live_store = LiveDataStore(base_dir=Path(tmp) / "live", window_size=lookback)
        loop = InferenceLoop(
            broker=LocalReplayBroker(),
            assets_timeframes=[(asset, timeframe)],
            feature_engine=FeaturePipeline(max_lookback=lookback),
            einher_engine=EinherEngine(),
            risk_manager=RiskManager(system_config),
            live_store=live_store,
            data_store=store,
            config=system_config,
        )
        loop.einher_engine.load_corpus(str(app.CORPUS_PATH))
        print(f"einhers charges : {len(loop.einher_engine.einhers)}")

        t0 = time.time()
        ajouts = await loop.bootstrap_history()
        print(f"amorcage : {ajouts} en {time.time()-t0:.1f}s")

        fenetre = live_store.get_window(asset, timeframe)
        print(f"fenetre live : {fenetre.height} bougies | colonnes {fenetre.columns}")
        assert fenetre.height >= 200, "amorcage insuffisant pour des features a fenetre"

        t0 = time.time()
        durees = await loop.warmup_features()
        print(f"echauffement JIT : {durees} en {time.time()-t0:.1f}s")

        bougie = await loop._fetch_last_candle(asset, timeframe)
        print(f"derniere bougie : {bougie}")
        # Le broker de demo renvoie la derniere bougie REELLE, deja presente dans le
        # store apres amorcage : on decale son horodatage d'un pas de temps pour
        # simuler la cloture suivante et declencher effectivement le cycle.
        pas = {
            "5m": timedelta(minutes=5), "15m": timedelta(minutes=15),
            "1h": timedelta(hours=1), "4h": timedelta(hours=4), "1d": timedelta(days=1),
        }[timeframe]
        bougie = {**bougie, "timestamp": bougie["timestamp"] + pas}

        async def _fetch_simulee(_asset: str, _tf: str, *_: Any, **__: Any) -> dict[str, Any]:
            return bougie

        loop._fetch_last_candle = _fetch_simulee  # type: ignore[method-assign]

        t0 = time.time()
        result = await loop._process_asset_tf(asset, timeframe)
        premier = time.time() - t0
        print(f"cycle 1 (a froid, JIT inclus) : {premier:.1f}s -> signals={result['signals_count']} "
              f"forming={result['forming_count']} error={result['error']}")

        # Second cycle : mesure le cout REEL par bougie une fois le JIT chaud
        # (c'est ce chiffre qui determine la tenue des fenetres 5m/15m/1h).
        bougie = {**bougie, "timestamp": bougie["timestamp"] + pas}
        t0 = time.time()
        result2 = await loop._process_asset_tf(asset, timeframe)
        chaud = time.time() - t0
        print(f"cycle 2 (JIT chaud) : {chaud:.1f}s -> signals={result2['signals_count']} "
              f"forming={result2['forming_count']} error={result2['error']}")
        print(f"etapes sautees (calcul cible) : {loop.feature_engine.last_report.get('etapes_sautees')}")
        for sig in result["signals"][:5]:
            print(f"  SIGNAL {sig.einher_name} [{sig.direction.value}] entry={sig.entry_price:.4f} "
                  f"tp={sig.tp_price:.4f} sl={sig.sl_price:.4f} conf={sig.confidence}")

        # Cycle complet (confluence + risk manager + ordre simule)
        await loop._run_cycle(__import__("datetime").datetime.now(__import__("datetime").UTC))
        print("cycle complet (confluence + risque + execution simulee) : OK")
        store.close()
        return 0


def main() -> int:
    """Point d'entree."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--asset", default="BTCUSD")
    ap.add_argument("--timeframe", default="1h")
    ap.add_argument("--lookback", type=int, default=1500)
    args = ap.parse_args()
    return asyncio.run(run(args.asset, args.timeframe, args.lookback))


if __name__ == "__main__":
    raise SystemExit(main())
