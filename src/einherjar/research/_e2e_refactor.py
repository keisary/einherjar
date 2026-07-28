"""
E2E test du refactoring ORCH-001 / CORE-001.

Lance un run réel sur XAUUSD / 15m via la nouvelle
architecture (core.Engine + core.runner).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-export", action="store_true",
        help="Désactive l'export (utile en cas de bug de "
             "sérialisation pré-existant).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-5s] %(name)s | %(message)s",
    )

    print("=" * 70)
    print("E2E -- Refactoring ORCH-001 / CORE-001 (XAUUSD / 15m)")
    print("=" * 70)

    from config.config import Config
    from config.dataset import DatasetConfig
    from core import Engine
    from core import DiscoveryTarget
    from core.exceptions import DatasetContractError

    config = Config()
    config.dataset = DatasetConfig(
        midas_root=r"D:/midas_v2/midasV3/src/data/compiled",
        asset="XAUUSD",
        asset_class="forex",
        timeframe="15m",
    )

    target = DiscoveryTarget(asset="XAUUSD", timeframe="15m")

    engine = Engine(
        config,
        run_id="refactor_e2e",
        output_root=Path("outputs/refactor_e2e"),
        continue_on_error=True,
        export_pair_results=not args.skip_export,
    )

    print()
    print(f"Engine       : {engine}")
    print(f"Target       : {target.key}")
    print(f"Output       : {engine.output_root}")
    print(f"Export pair  : {engine.export_pair_results}")
    print()

    t0 = time.time()
    result = engine.run_pair(target, index=0)
    elapsed = time.time() - t0

    print()
    print("=" * 70)
    print(f"Duree        : {elapsed:.1f}s")
    print(f"Result.success: {result.success}")
    print(f"Result.errors : {result.errors}")
    print()
    print("Phases (par-paire) :")
    for name, status in result.state.phases.items():
        marker = {
            "success": "OK",
            "failed": "KO",
            "skipped": "--",
            "running": ">>",
            "pending": "..",
        }.get(status.status, "??")
        err = f" -- {status.error}" if status.error else ""
        print(f"  [{marker}] {name:14s} {status.status:8s}{err}")
    print()
    print(f"  candidates       : {len(result.candidates)}")
    print(f"  validated        : {len(result.validated)}")
    print(f"  rejected         : {len(result.rejected)}")
    print(f"  execution_results: {len(result.execution_results)}")
    print(f"  einhers          : {len(result.einhers)}")
    print(f"  export_paths     : {list(result.export_paths.keys())}")
    print()

    failed = result.state.failed_phase
    if failed:
        print(f"PIPELINE A ECHOUE sur la phase '{failed}'.")
        return 3

    print("PIPELINE COMPLET -- toutes les phases obligatoires OK.")
    return 0 if result.success else 2


if __name__ == "__main__":
    raise SystemExit(main())
