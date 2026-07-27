import sys
sys.path.insert(0, r"D:/midas_v2/einherjar/src/einherjar/research")

from pathlib import Path
from models.feature_registry import FeatureRegistry
from discovery.generator import DiscoveryGenerator
from config.config import Config

config = Config()
meta_path = Path(r"D:/midas_v2/midasV3/src/data/compiled/forex/15m/metadata.json")
registry = FeatureRegistry(str(meta_path))
print("Registry features:", registry.feature_count)

gen = DiscoveryGenerator(config, registry)
print("Generator created:", gen)

result = gen.seed()
print("Seed result:", result)
