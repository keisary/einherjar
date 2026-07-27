import sys
sys.path.insert(0, r"D:/midas_v2/einherjar/src/einherjar/research")

import importlib.util
spec = importlib.util.spec_from_file_location("discovery_run", r"D:/midas_v2/einherjar/src/einherjar/research/discovery.py")
discovery_mod = importlib.util.module_from_spec(spec)
sys.modules["discovery_run"] = discovery_mod
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

# Inspect DiscoveryComponents.create
components = discovery_mod.DiscoveryComponents.create(config)
print("=== COMPONENTS ===")
print("dataset_loader:", components.dataset_loader)
print("discovery_explorer:", components.discovery_explorer)
print("discovery_generator:", components.discovery_generator)
print("validation_engine:", components.validation_engine)
print("execution_engine:", components.execution_engine)

# Test _load_dataset
orch = discovery_mod.DiscoveryOrchestrator(config)
target = discovery_mod.DiscoveryTarget(asset="XAUUSD", timeframe="15m")
ctx = discovery_mod.DiscoveryContext(run_id="test", target=target, index=0)

# Simuler _load_dataset
try:
    dataset = orch._load_dataset(target, context=ctx)
    print("\n=== DATASET ===")
    print("dataset:", dataset)
    print("type:", type(dataset))
except Exception as e:
    print("_load_dataset ERROR:", type(e).__name__, e)

# Test _discover
try:
    discovery_output = orch._discover(dataset, target=target, context=ctx)
    print("\n=== DISCOVERY OUTPUT ===")
    print("output:", discovery_output)
    print("type:", type(discovery_output))
except Exception as e:
    print("_discover ERROR:", type(e).__name__, e)
