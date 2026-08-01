"""
Test ciblé : assets qui n'avaient pas d'einhers (AMZN, TSLA, JPM, XOM).

Avant le fix : le portfolio plantait sur "no execution results",
le summary n'était pas écrit.

Après le fix : le portfolio retourne des objets vides, le
PairExporter écrit au moins le summary même en cas d'échec.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-5s] %(name)s | %(message)s",
    )

    from config.config import Config
    from config.dataset import DatasetConfig
    from core.runner import DiscoveryOrchestrator, DiscoverySettings
    from core.types import DiscoveryTarget

    config = Config()
    config.dataset = DatasetConfig(
        midas_root=r"D:/midas_v2/midasV3/src/data/compiled",
        asset="",
        asset_class="",
        timeframe="15m",
    )

    # Test sur 3 assets connus pour ne produire aucun einher
    # (validation/execution trop stricts) : AMZN, TSLA, XOM
    assets = ["AMZN", "TSLA", "XOM"]
    target_pairs = tuple(
        DiscoveryTarget(asset=a, timeframe="15m") for a in assets
    )

    settings = DiscoverySettings(
        pairs=target_pairs,
        output_root=Path("outputs/_test_empty_portfolio"),
        export_pair_results=True,
        export_run_summary=True,
        continue_on_error=True,
    )

    print("=" * 70)
    print("Test empty portfolio -- assets sans einhers")
    print("=" * 70)
    print(f"Assets: {assets}")
    print()

    orchestrator = DiscoveryOrchestrator(config, settings=settings)
    t0 = time.time()
    result = orchestrator.run()
    elapsed = time.time() - t0

    print()
    print(f"Duree   : {elapsed:.1f}s")
    print(f"Pairs   : {result.pair_count}")
    print(f"Success : {result.success_count}")
    print(f"Failed  : {result.failure_count}")
    print(f"Errors  : {list(result.errors)}")
    print()

    for pr in result.pair_results:
        print(f"  [{'OK' if pr.success else 'KO'}] {pr.pair_key} "
              f"einhers={pr.einher_count} "
              f"paths={list(pr.export_paths.keys())}")

    # Vérification : chaque pair doit avoir au moins
    # un summary_json écrit, même si pipeline KO
    print()
    print("=" * 70)
    print("VERIFICATION FICHIERS")
    print("=" * 70)
    ok = True
    for pr in result.pair_results:
        if "summary_json" not in pr.export_paths:
            print(f"  [KO] {pr.pair_key} -- pas de summary_json !")
            ok = False
            continue
        path = Path(pr.export_paths["summary_json"])
        if not path.exists():
            print(f"  [KO] {pr.pair_key} -- summary introuvable : {path}")
            ok = False
            continue
        size = path.stat().st_size
        content = json.loads(path.read_text(encoding="utf-8"))
        success = content.get("success", False)
        einher_count = content.get("einher_count", 0)
        print(f"  [OK] {pr.pair_key} -- {size}B -- success={success} "
              f"einher_count={einher_count}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
