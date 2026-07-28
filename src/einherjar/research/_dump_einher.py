"""
Dump d'un Einher produit par Engine.run_pair() — sans exporter.

Court-circuite l'export et imprime le contenu direct du
DiscoveryPairResult.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

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
        run_id="einher_dump",
        output_root=Path("outputs/einher_dump"),
        continue_on_error=True,
        export_pair_results=False,  # pas d'export
    )

    print(f"Lancement du pipeline sur {target.key} (sans export)...")
    t0 = time.time()
    result = engine.run_pair(target, index=0)
    elapsed = time.time() - t0
    print(f"Terminé en {elapsed:.1f}s, success={result.success}")
    print()

    # === Einher ===
    print("=" * 70)
    print("EINHERS PRODUITS")
    print("=" * 70)
    print(f"Nombre: {len(result.einhers)}")
    for idx, einher in enumerate(result.einhers):
        print(f"\n--- Einher #{idx} ---")
        print(f"  Repr: {einher!r}")
        print(f"  Fingerprint: {einher.fingerprint}")
        if hasattr(einher, "profile") and einher.profile:
            print(f"  Profile.name: {getattr(einher.profile, 'name', '?')}")
        if hasattr(einher, "candidate") and einher.candidate:
            cand = einher.candidate
            print(f"  Candidate.condition_count: {getattr(cand, 'condition_count', '?')}")
            if hasattr(cand, "hypothesis") and cand.hypothesis:
                hyp = cand.hypothesis
                print(f"  Hypothesis.conditions ({len(hyp.conditions)}):")
                for i, cond in enumerate(hyp.conditions):
                    left = getattr(cond, "left", None)
                    op = getattr(cond, "operator", "?")
                    right = getattr(cond, "right", "?")
                    left_name = getattr(left, "name", "?") if left else "?"
                    left_family = (
                        getattr(getattr(left, "economic_family", None), "value", "?")
                        if left else "?"
                    )
                    print(f"    [{i}] {left_name} ({left_family}) {op} {right!r}")
        if hasattr(einher, "report") and einher.report:
            rpt = einher.report
            print(f"  Report.metadata: {dict(getattr(rpt, 'metadata', {}))}")
        if hasattr(einher, "journal") and einher.journal:
            jrn = einher.journal
            n_trades = len(jrn) if hasattr(jrn, "__len__") else "?"
            print(f"  Journal trades: {n_trades}")
        if hasattr(einher, "execution_result") and einher.execution_result:
            er = einher.execution_result
            print(f"  ExecutionResult.success: {getattr(er, 'success', '?')}")
            if hasattr(er, "metrics"):
                print(f"  ExecutionResult.metrics: {getattr(er, 'metrics', '?')}")
        # to_dict complet
        try:
            d = einher.to_dict()
            print(f"\n  to_dict (résumé):")
            print(f"    profile: {d.get('profile', {})}")
        except Exception as exc:
            print(f"  to_dict ERROR: {exc!r}")

    # === Execution Results (détails des trades) ===
    print()
    print("=" * 70)
    print("EXECUTION RESULTS")
    print("=" * 70)
    for idx, er in enumerate(result.execution_results):
        print(f"\n--- ExecutionResult #{idx} ---")
        if hasattr(er, "subject_fingerprint"):
            print(f"  subject_fingerprint: {er.subject_fingerprint}")
        if hasattr(er, "success"):
            print(f"  success: {er.success}")
        if hasattr(er, "trades"):
            n = len(er.trades) if hasattr(er.trades, "__len__") else "?"
            print(f"  trades.count: {n}")
            for j, t in enumerate(er.trades[:3]):
                print(f"    trade[{j}]: {t!r}")
        if hasattr(er, "metrics") and er.metrics:
            print(f"  metrics: {er.metrics}")
        if hasattr(er, "diagnostics") and er.diagnostics:
            print(f"  diagnostics: {dict(er.diagnostics)}")

    # === Portfolio Report ===
    print()
    print("=" * 70)
    print("PORTFOLIO REPORT")
    print("=" * 70)
    pr = result.portfolio_report
    if pr is None:
        print("(vide)")
    else:
        print(f"  name: {getattr(pr, 'name', '?')}")
        if hasattr(pr, "entries"):
            print(f"  entries.count: {len(pr.entries) if hasattr(pr.entries, '__len__') else '?'}")
        for attr in ("rejected", "selected", "excluded", "metrics"):
            if hasattr(pr, attr):
                val = getattr(pr, attr)
                if val is None:
                    print(f"  {attr}: None")
                elif hasattr(val, "__len__"):
                    print(f"  {attr}.count: {len(val)}")
                else:
                    print(f"  {attr}: {val!r}")
        if hasattr(pr, "metadata"):
            print(f"  metadata: {dict(pr.metadata)}")

    # === Allocation ===
    print()
    print("=" * 70)
    print("ALLOCATION")
    print("=" * 70)
    alloc = result.allocation
    if alloc is None:
        print("(vide)")
    else:
        print(f"  type: {type(alloc).__name__}")
        if hasattr(alloc, "entries"):
            print(f"  entries.count: {len(alloc.entries) if hasattr(alloc.entries, '__len__') else '?'}")
            for j, entry in enumerate(alloc.entries[:3]):
                fp = getattr(entry, "subject_fingerprint", "?")
                w = getattr(entry, "weight", "?")
                cap = getattr(entry, "capital", "?")
                acc = getattr(entry, "accepted", "?")
                print(f"    entry[{j}]: fp={fp} weight={w} capital={cap} accepted={acc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
