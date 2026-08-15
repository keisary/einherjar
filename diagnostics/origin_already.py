"""Origine des 4 ALREADY_IN_ARCHIVE du 15m v7 : premiere apparition des fingerprints."""
import json

ARCH = r"D:\midas_v2\einherjar\outputs\archive\archive.jsonl"
seen = {}
order = []
for line in open(ARCH, encoding="utf-8"):
    line = line.strip()
    if not line:
        continue
    e = json.loads(line)
    fp = e.get("fingerprint_comportemental", "")
    d = str(e.get("date_rejet", ""))
    if fp and fp not in seen:
        seen[fp] = d
        order.append(fp)

# les ALREADY v7 15m (19:40-19:42)
for line in open(ARCH, encoding="utf-8"):
    line = line.strip()
    if not line:
        continue
    e = json.loads(line)
    if ("2026-08-10T19:40:00" <= str(e.get("date_rejet", "")) <= "2026-08-10T19:42:00"
            and e.get("raison_rejet") == "ALREADY_IN_ARCHIVE"):
        fp = e.get("fingerprint_comportemental", "")
        print("ALREADY v7:", fp[:16], "| premiere apparition:", seen.get(fp, "?"))