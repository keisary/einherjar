"""
Smoke test du refactoring ORCH-001 / CORE-001.

Vérifie, sans lancer le pipeline complet :
- les imports
- les types
- l'instanciation de l'Engine
- la résolution des actifs
- le contrat de l'EngineState
- le strict typing du bootstrap
"""

from __future__ import annotations

import sys
import traceback
from datetime import datetime
from datetime import timezone
from pathlib import Path

# Le bootstrap attend ces paths
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def main() -> int:
    print("=" * 70)
    print("SMOKE TEST — Refactoring ORCH-001 / CORE-001")
    print("=" * 70)

    errors: list[str] = []

    # ---------------------------------------------------------
    # 1) Imports core
    # ---------------------------------------------------------
    print("\n[1] Imports core")
    try:
        from core import (
            ConfigurationError,
            DatasetContractError,
            DatasetError,
            DatasetValidationError,
            DiscoveryContractError,
            DiscoveryError,
            DiscoveryPairResult,
            DiscoveryTarget,
            Engine,
            EngineContext,
            EngineState,
            ExecutionContractError,
            ExecutionError,
            ExportContractError,
            KnowledgeContractError,
            KnowledgeError,
            MemoryContractError,
            MemoryError,
            PHASE_NAMES,
            PhaseContractError,
            PhaseStatus,
            PortfolioContractError,
            PortfolioError,
            SearchError,
            ValidationContractError,
            ValidationError,
            known_assets,
            reset_cache,
            resolve_asset_class,
            resolve_asset_meta,
        )
        print("    OK — tous les imports core passent")
    except Exception as exc:
        errors.append(f"core imports failed: {exc!r}")
        traceback.print_exc()
        return 1

    # ---------------------------------------------------------
    # 2) Types valeur
    # ---------------------------------------------------------
    print("\n[2] Types valeur")
    try:
        t1 = DiscoveryTarget(asset="XAUUSD", timeframe="15m")
        assert t1.key == "XAUUSD@15m", f"key mismatch: {t1.key}"
        assert t1.slug == "xauusd__15m", f"slug mismatch: {t1.slug}"
        t2 = DiscoveryTarget.from_dict({"asset": "BTCUSD", "timeframe": "1h"})
        assert t2.key == "BTCUSD@1h", f"from_dict key mismatch: {t2.key}"
        print("    OK — DiscoveryTarget construction et serialization")
    except Exception as exc:
        errors.append(f"DiscoveryTarget failed: {exc!r}")
        traceback.print_exc()

    # ---------------------------------------------------------
    # 3) Asset resolver
    # ---------------------------------------------------------
    print("\n[3] Asset resolver")
    try:
        ac = resolve_asset_class("XAUUSD")
        assert ac == "forex", f"XAUUSD -> {ac} (expected 'forex')"
        ac = resolve_asset_class("BTCUSD")
        assert ac == "crypto", f"BTCUSD -> {ac} (expected 'crypto')"
        all_assets = known_assets()
        assert "XAUUSD" in all_assets
        assert "BTCUSD" in all_assets
        print(f"    OK — {len(all_assets)} actifs résolus depuis assets_v1.json")
    except Exception as exc:
        errors.append(f"asset resolver failed: {exc!r}")
        traceback.print_exc()

    # ---------------------------------------------------------
    # 4) EngineState
    # ---------------------------------------------------------
    print("\n[4] EngineState per-pair")
    try:
        st = EngineState()
        assert st.current_phase == "initialization"
        assert not st.success
        st.start()
        assert st.started_at is not None
        st.begin_phase("discovery")
        assert st.current_phase == "discovery"
        assert st.get("discovery").status == "running"
        st.complete_phase("discovery", metadata={"candidates": 5})
        assert st.is_phase_success("discovery")
        st.fail_phase("validation", "boom")
        assert st.failed_phase == "validation"
        assert not st.success
        st.finish(success=False)
        assert st.current_phase == "terminal"
        print("    OK — EngineState suit correctement la séquence de phases")
    except Exception as exc:
        errors.append(f"EngineState failed: {exc!r}")
        traceback.print_exc()

    # ---------------------------------------------------------
    # 5) PhaseStatus
    # ---------------------------------------------------------
    print("\n[5] PhaseStatus")
    try:
        ps = PhaseStatus()
        ps.start()
        assert ps.status == "running"
        ps.succeed(metadata={"k": 1})
        assert ps.status == "success"
        assert ps.metadata["k"] == 1
        print("    OK — PhaseStatus transitions")
    except Exception as exc:
        errors.append(f"PhaseStatus failed: {exc!r}")
        traceback.print_exc()

    # ---------------------------------------------------------
    # 6) Engine instantiation (sans run complet)
    # ---------------------------------------------------------
    print("\n[6] Engine instantiation")
    try:
        from config.config import Config
        config = Config()
        engine = Engine(config, run_id="smoke")
        assert engine.run_id == "smoke"
        assert engine.continue_on_error is True
        assert engine.output_root == Path("outputs")
        print("    OK — Engine s'instancie avec la config globale")
    except Exception as exc:
        errors.append(f"Engine instantiation failed: {exc!r}")
        traceback.print_exc()

    # ---------------------------------------------------------
    # 7) Bootstrap (DiscoveryOrchestrator)
    # ---------------------------------------------------------
    print("\n[7] Bootstrap DiscoveryOrchestrator")
    try:
        from core.runner import (
            DiscoveryOrchestrator,
            DiscoveryRunResult,
            DiscoverySettings,
        )
        settings = DiscoverySettings(
            assets=("XAUUSD", "BTCUSD"),
            timeframes=("15m",),
            output_root=Path("outputs/smoke"),
            continue_on_error=True,
        )
        # resolve_targets is on the instance — build one
        orch = DiscoveryOrchestrator(config, settings=settings)
        targets = orch.resolve_targets()
        assert len(targets) == 2, f"expected 2 targets, got {len(targets)}"
        assert {t.asset for t in targets} == {"XAUUSD", "BTCUSD"}
        assert {t.timeframe for t in targets} == {"15m"}
        print(f"    OK — {len(targets)} cibles résolues par le bootstrap")
    except Exception as exc:
        errors.append(f"bootstrap failed: {exc!r}")
        traceback.print_exc()

    # ---------------------------------------------------------
    # 8) Dataset package
    # ---------------------------------------------------------
    print("\n[8] Dataset package")
    try:
        from dataset import (
            DatasetContract,
            DatasetInspector,
            DatasetLoader,
            DatasetSplit,
            DatasetStatistics,
            DatasetValidator,
            MidasArrays,
        )
        print("    OK — dataset package exports")
    except Exception as exc:
        errors.append(f"dataset package failed: {exc!r}")
        traceback.print_exc()

    # ---------------------------------------------------------
    # 9) DatasetContract verify
    # ---------------------------------------------------------
    print("\n[9] DatasetContract.verify_for_midas")
    try:
        # Contrat valide
        c1 = DatasetContract(
            feature_count=3,
            horizons=("h1", "h5"),
            feature_names=("a", "b", "c"),
            label_names=("y",),
            dtype="float64",
        )
        c1.verify_for_midas()  # doit passer

        # Contrat sans horizons : doit lever DatasetContractError
        try:
            c2 = DatasetContract(
                feature_count=3,
                horizons=(),
                feature_names=("a", "b", "c"),
                label_names=("y",),
                dtype="float64",
            )
            c2.verify_for_midas()
            errors.append("DatasetContract without horizons did not raise")
        except DatasetContractError:
            pass  # attendu

        # Contrat feature_count <= 0 : doit lever
        try:
            DatasetContract(
                feature_count=0,
                horizons=("h1",),
                feature_names=(),
                label_names=(),
                dtype="float64",
            )
            errors.append("DatasetContract with feature_count=0 did not raise")
        except DatasetContractError:
            pass  # attendu

        print("    OK — DatasetContract.verify_for_midas strict")
    except Exception as exc:
        errors.append(f"DatasetContract verify failed: {exc!r}")
        traceback.print_exc()

    # ---------------------------------------------------------
    # 10) Engine.run_pair() with invalid target — doit lever
    # ---------------------------------------------------------
    print("\n[10] Engine.run_pair() — contrat strict")
    try:
        from config.config import Config
        config = Config()
        engine = Engine(config, run_id="strict", continue_on_error=False)
        # target inexistant (asset inconnu)
        target = DiscoveryTarget(asset="UNKNOWN_ASSET_XYZ", timeframe="15m")
        try:
            engine.run_pair(target)
            errors.append("run_pair with unknown asset did not raise")
        except Exception as exc:
            # doit lever DatasetContractError (asset pas dans assets_v1.json)
            assert "UNKNOWN_ASSET_XYZ" in str(exc) or "not found" in str(exc).lower(), (
                f"unexpected error: {exc!r}"
            )
            print(f"    OK — contrat strict respecté ({type(exc).__name__})")
    except Exception as exc:
        errors.append(f"Engine strict test failed: {exc!r}")
        traceback.print_exc()

    # ---------------------------------------------------------
    # Conclusion
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    if errors:
        print(f"FAIL — {len(errors)} erreur(s) :")
        for e in errors:
            print(f"  - {e}")
        return 1
    else:
        print("OK — smoke test complet passé.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
