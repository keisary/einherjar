"""Distribution des sharpe_net et p DSR par admission v7."""
import json
from collections import defaultdict

ARCH = r"D:\midas_v2\einherjar\outputs\archive\archive.jsonl"
START = "2026-08-10T19:12:00"

clusters = defaultdict(list)
for line in open(ARCH, encoding="utf-8"):
    line = line.strip()
    if not line:
        continue
    e = json.loads(line)
    d = str(e.get("date_rejet", ""))
    if d < START:
        continue
    m = d[11:16]
    mb = e.get("mesures_brutes_val") or {}
    sh = mb.get("sharpe_net")
    clusters[m].append((
        sh, e.get("deflated_sharpe_ratio"), e.get("raison_rejet"),
        mb.get("n_signals"),
    ))

for m in sorted(clusters):
    rows = clusters[m]
    shs = sorted((r[0] for r in rows if isinstance(r[0], (int, float))), reverse=True)
    print(f"--- {m} ({len(rows)} rejets) ---")
    for sh, p, rai, ns in rows[:5]:
        print(f"  sharpe={sh} p_dsr={p} n={ns} raison={rai}")
    if shs:
        print("  top5 sharpe:", [round(x, 2) for x in shs[:5]])