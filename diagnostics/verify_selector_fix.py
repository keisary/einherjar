import types, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from einherjar.research.config.loader import load_config
from einherjar.research.selection.selector import GeneratorSelector
from einherjar.research.generators.protocol import GenerationProtocol

class FakeEngine:
    pass

proto = GenerationProtocol(seed=42, data_version="test")
config = load_config(Path("src/einherjar/research/config"))
sel = types.SimpleNamespace(
    generator_class="TypedGPGenerator", generator_name="TypedGPGenerator",
    protocol=proto, rank=1, score=0.5,
)
g = GeneratorSelector.instantiate(sel, config=config, engine=FakeEngine())
print("instantiate TypedGP OK ->", type(g).__name__)
assert type(g).__name__ == "TypedGPGenerator"
print("VERIF_SELECTOR_FIX: OK")