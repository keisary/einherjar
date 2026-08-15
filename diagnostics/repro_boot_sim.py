"""bootstrap_sharpe sort-il des CI finis sur des rets simulés réalistes (SR annuel ~6.3) ?"""
import warnings
warnings.filterwarnings("ignore")
import sys
import numpy as np
sys.path.insert(0, "src")

from einherjar.research.engine.bootstrap import bootstrap_sharpe
from einherjar.research.config.loader import load_config

cfg = load_config("src/einherjar/research/config")
rng = np.random.default_rng(42)
# 377 trades, rendement net moyen 0.13% / trade, std 1.0% (SR par trade 0.13
# -> SR annuel ~0.13*sqrt(2336) ~ 6.3 avec 15 bougies tenues en moyenne)
rets = rng.normal(loc=0.0013, scale=0.010, size=377)
for ppy in (1.0, 2336.0):
    ci = bootstrap_sharpe(rets.astype(float), cfg, periods_per_year=ppy, rng_seed=42)
    print(f"ppy={ppy}: ci={ci}")