"""Verdicts du run v7 (post 19:12) : clusters par minute + stats."""
import json
from collections import Counter, defaultdict

ARCH = r"D:\midas_v2\einherjar\outputs\archive\archive.jsonl"
START = "2026-08-10T19:12:00"

by_min = defaultdict(lambda: Counter())
best = {}
for line in open(ARCH, encoding="utf-8"):
    line = line.strip()
    if not line:
        continue
    e = json.loads(line)
    d = str(e.get("date_rejet", ""))
    if d < START:
        continue
    m = d[11:16]
    by_min[m][str(e.get("raison_rejet"))] += 1
    sh = e.get("sharpe_net_val")
    try:
        sh = float(sh)
    except (TypeError, ValueError):
        sh = float("-inf")
    if sh > best.get(m, (-10**9,))[0]:
        best[m] = (sh, e.get("id"), e.get("deflated_sharpe_ratio"))

for m in sorted(by_min):
    c = dict(by_min[m])
    b = best.get(m)
    print(m, c, "| max sharpe:", round(b[0], 2) if b else None, "p_dsr:", b[2] if b else None)