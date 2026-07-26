import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backtest.calibrator import Calibrator

# Charger corpus v2
with open(ROOT / "config" / "corpus_brut_v2.json", "r", encoding="utf-8") as f:
    corpus = json.load(f)

# Prendre 20 einhers de differents domaines et les forcer sur BTCUSD 15m
domains_seen = set()
selected = []
for e in corpus["einhers"]:
    dom = e["domain"]
    if dom not in domains_seen and len(selected) < 20:
        e2 = dict(e)
        e2["assets"] = [{"asset": "BTCUSD", "class": "crypto"}]
        e2["timeframes"] = ["15m"]
        selected.append(e2)
        domains_seen.add(dom)
    if len(selected) >= 20:
        break

mini_path = ROOT / "data" / "mini_corpus_20.json"
with open(mini_path, "w", encoding="utf-8") as f:
    json.dump({"einhers": selected}, f, indent=2)

print(f"Mini-corpus: {len(selected)} Einhers")
for e in selected:
    print(f"  - {e['name']}: {e['trigger']}")

print("\n=== Lancement mini-backtest BTCUSD 15m ===")
cal = Calibrator(
    corpus_path=mini_path,
    native_exits_path=ROOT / "config" / "native_exits.json",
)
results = cal.run(output_path=ROOT / "data" / "mini_results_20.json", progress_every=1)

print(f"\n=== Resultats ===")
print(f"Total resultats: {len(results)}")
for r in results[:10]:
    print(f"  {r['einher_name']}: {r['n_trades']} trades, Sharpe={r['sharpe_ratio']}, WinRate={r['win_rate']}, Return={r['total_return']}")
