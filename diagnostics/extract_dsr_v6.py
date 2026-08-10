"""Verdicts par cluster temporel (run v6 apres fix DSR)."""
import json
from collections import Counter

ARCH = r"D:\midas_v2\einherjar\outputs\archive\archive.jsonl"
CLUSTERS = [("15m", "18:50"), ("1h", "18:55"), ("4h", "18:57"), ("?18:28", "18:28"), ("?19:05", "19:05")]

rows = []
for line in open(ARCH, encoding="utf-8"):
    line = line.strip()
    if not line:
        continue
    try:
        e = json.loads(line)
    except Exception:
        continue
    d = str(e.get("date_rejet", ""))
    if d >= "2026-08-10T18:00:00":
        rows.append((d, e))

for label, minute in CLUSTERS:
    es = [(d, e) for d, e in rows if d[11:16] == minute]
    if not es:
        print(f"== {label}: aucun")
        continue
    c = Counter(str(e.get("raison_rejet")) for _, e in es)
    print(f"== {label}: {len(es)} rejets | {dict(c)}")
    dsr = [(d, e) for d, e in es if str(e.get("raison_rejet")) == "DSR_FAIL"]
    dsr.sort(key=lambda x: (x[1].get("deflated_sharpe_ratio") or 0), reverse=True)
    for d, e in dsr[:4]:
        mb = e.get("mesures_brutes_val") or {}
        print(f"   p_dsr={e.get('deflated_sharpe_ratio'):.3f} sharpe_net={mb.get('sharpe_net')} "
              f"n_signals={mb.get('n_signals')} pbo={e.get('probability_of_backtest_overfitting')}")