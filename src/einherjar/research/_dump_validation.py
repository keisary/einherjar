"""
Dump des métriques de validation pour XAUUSD/15m.

Compare les métriques binary avant/après le fix du
signal_mask.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import core  # noqa: F401  - ncessaire pour viter le circular import

from config.config import Config
from config.dataset import DatasetConfig
from core import Engine
from core import DiscoveryTarget


def main() -> int:
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
        run_id="validation_dump",
        output_root=Path("outputs/validation_dump"),
        continue_on_error=True,
        export_pair_results=False,
    )

    t0 = time.time()
    result = engine.run_pair(target, index=0)
    elapsed = time.time() - t0

    print("=" * 70)
    print("Mtriques de validation post-fix (Chantier C)")
    print("=" * 70)
    print(f"Dure: {elapsed:.1f}s, success={result.success}")
    print()

    if not result.validated:
        print("Aucun validated candidate.")
        return 0

    print(f"Validated count: {len(result.validated)}")
    for idx, vc in enumerate(result.validated):
        if hasattr(vc, "metrics"):
            m = vc.metrics
            print(f"\n--- Validated Candidate #{idx} ---")
            # Assessment metrics
            if "validation" in m:
                vm = m["validation"]
                print(f"  score: {vm.get('score', 0.0):.4f}")
                print(f"  support: {vm.get('support', 0)}")
                print(f"  coverage: {vm.get('coverage', 0.0):.4f}")
                print(f"  baseline_mean: {vm.get('baseline_mean', 0.0):.6f}")
                print(f"  signal_mean: {vm.get('signal_mean', 0.0):.6f}")
                print(f"  lift: {vm.get('lift', 0.0):.6f}")
                print(f"  significance_score: {vm.get('significance_score', 0.0):.4f}")
                print(f"  robustness_score: {vm.get('robustness_score', 0.0):.4f}")
                print(f"  binary_precision: {vm.get('binary_precision', 0.0):.4f}")
                print(f"  binary_recall: {vm.get('binary_recall', 0.0):.4f}")
                print(f"  binary_f1: {vm.get('binary_f1', 0.0):.4f}")
                print(f"  directional_accuracy: {vm.get('directional_accuracy', 0.0):.4f}")
            if "assessment" in m:
                a = m["assessment"]
                print(f"  assessment.passed: {a.get('passed')}")
                print(f"  assessment.rejection_reasons: {a.get('rejection_reasons', [])}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
