"""Rejouer bootstrap_sharpe sur les rets archives d'une hyp BOOTSTRAP_CI_FAIL (15m v7, SR 8.6)."""
import json
import numpy as np

from einherjar.research.engine.bootstrap import bootstrap_sharpe
from einherjar.research.config.loader import load_config

cfg = load_config("src/einherjar/research/config")

ARCH = r"D:\midas_v2\einherjar\outputs\archive\archive.jsonl"
target = None
for line in open(ARCH, encoding="utf-8"):
    line = line.strip()
    if not line:
        continue
    e = json.loads(line)
    if (str(e.get("date_rejet", "")).startswith("2026-08-10T19:40")
            and e.get("raison_rejet") == "BOOTSTRAP_CI_FAIL"
            and target is None):
        target = e

if target is None:
    print("cible introuvable")
    raise SystemExit(1)

rets = target.get("ret_series") or []
print("n_rets:", len(rets), "| sharpe_net:", (target.get("mesures_brutes_val") or {}).get("sharpe_net"))
f = float(np.mean(rets)) / float(np.std(rets)) if len(rets) > 1 and float(np.std(rets)) > 0 else float("nan")
print("sharpe par trade:", round(f, 4))
for ppy in (1.0, 35040.0):
    r = bootstrap_sharpe(rets, cfg, periods_per_year=ppy, rng_seed=42)
    print(f"ppy={ppy}: ci=({r.ci_low:.3f}, {r.ci_high:.3f}) observed={r.observed:.3f}")