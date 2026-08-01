"""
Test à moyenne échelle : 8 assets mixtes.

Vérifie que chaque pair écrit au moins son summary,
que la pipeline tourne sans crash silencieux, et que
les 7 fichiers sont produits pour les pairs productifs.
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
        level=logging.WARNING,
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

    # Mix d'assets : certains produisent des einhers,
    # d'autres pas. L'important est que chacun écrive
    # au moins son summary.
    test_assets = [
        "XAUUSD",   # forex, doit produire
        "EURUSD",   # forex, doit produire
        "BTCUSD",   # crypto, doit produire
        "AMZN",     # stocks_tech, ne produit pas (0 einhers)
        "TSLA",     # stocks_tech, ne produit pas
        "JPM",      # stocks_value, ne produit pas
        "XOM",      # stocks_growth, ne produit pas
        "SP500",    # indices, ne produit pas
    ]

    target_pairs = tuple(
        DiscoveryTarget(asset=a, timeframe="15m") for a in test_assets
    )

    settings = DiscoverySettings(
        pairs=target_pairs,
        output_root=Path("outputs/_test_broad"),
        export_pair_results=True,
        export_run_summary=True,
        continue_on_error=True,
    )

    print("=" * 70)
    print(f"Test broad -- {len(test_assets)} assets")
    print("=" * 70)
    print(f"Assets: {test_assets}")
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

    # Vérification
    print()
    print("=" * 70)
    print("VERIFICATION FICHIERS")
    print("=" * 70)
    all_ok = True
    for pr in result.pair_results:
        if "summary_json" not in pr.export_paths:
            print(f"  [KO] {pr.pair_key} -- pas de summary !")
            all_ok = False
            continue
        path = Path(pr.export_paths["summary_json"])
        if not path.exists():
            print(f"  [KO] {pr.pair_key} -- fichier introuvable : {path}")
            all_ok = False
            continue
        size = path.stat().st_size
        content = json.loads(path.read_text(encoding="utf-8"))
        success = content.get("success", False)
        einher_count = content.get("einher_count", 0)
        # Pour les pairs productifs (XAUUSD, EURUSD, BTCUSD), on attend les 7 fichiers
        if pr.einher_count > 0 and len(pr.export_paths) < 7:
            print(f"  [KO] {pr.pair_key} -- seulement "
                  f"{len(pr.export_paths)}/7 fichiers pour {pr.einher_count} einhers")
            all_ok = False
            continue
        print(f"  [OK] {pr.pair_key} -- {size}B -- success={success} "
              f"einher_count={einher_count} "
              f"files={len(pr.export_paths)}/7")

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
