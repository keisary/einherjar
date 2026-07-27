import sys
sys.path.insert(0, r"D:/midas_v2/einherjar/src/einherjar/research")

import importlib.util
spec = importlib.util.spec_from_file_location("discovery_orchestrator", r"D:/midas_v2/einherjar/src/einherjar/research/discovery.py")
discovery_mod = importlib.util.module_from_spec(spec)
sys.modules["discovery_orchestrator"] = discovery_mod
spec.loader.exec_module(discovery_mod)

from config.config import Config
from config.dataset import DatasetConfig

config = Config()
config.dataset = DatasetConfig(
    midas_root=r"D:/midas_v2/midasV3/src/data/compiled",
    asset="XAUUSD",
    asset_class="forex",
    timeframe="15m",
)

orchestrator = discovery_mod.DiscoveryOrchestrator(config)
print("Orchestrator created, run_id:", orchestrator.run_id)

try:
    result = orchestrator.run(assets=["XAUUSD"], timeframes=["15m"])
    print("RUN finished")
    print("Pairs:", result.pair_count)
    print("Success:", result.success_count)
    print("Failures:", result.failure_count)
    for pr in result.pair_results:
        print(f"  {pr.pair_key}: success={pr.success}")
        if not pr.success:
            print(f"    errors: {pr.errors}")
except Exception as e:
    import traceback
    traceback.print_exc()
