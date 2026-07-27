import sys
sys.path.insert(0, r"D:/midas_v2/einherjar/src/einherjar/research")

from config.config import Config
from config.dataset import DatasetConfig

config = Config()
config.dataset = DatasetConfig(
    midas_root=r"D:/midas_v2/midasV3/src/data/compiled",
    asset="XAUUSD",
    asset_class="forex",
    timeframe="15m",
)

# Test _instantiate directement
import importlib.util
spec = importlib.util.spec_from_file_location("discovery_run", r"D:/midas_v2/einherjar/src/einherjar/research/discovery.py")
discovery_mod = importlib.util.module_from_spec(spec)
sys.modules["discovery_run"] = discovery_mod
spec.loader.exec_module(discovery_mod)

_instantiate = discovery_mod._instantiate

loader = _instantiate(("dataset.loader",), ("DatasetLoader",), config=config)
print("loader:", loader, "type:", type(loader))

# Test DatasetLoader.from_config directement
from dataset.loader import DatasetLoader
try:
    dl = DatasetLoader.from_config(config)
    print("from_config OK:", dl)
except Exception as e:
    print("from_config ERROR:", type(e).__name__, e)

# Test Explorer
explorer = _instantiate(("discovery.explorer",), ("Explorer",), config=config)
print("explorer:", explorer, "type:", type(explorer))

# Test Generator
gen = _instantiate(("discovery.generator",), ("DiscoveryGenerator",), config=config)
print("generator:", gen, "type:", type(gen))
