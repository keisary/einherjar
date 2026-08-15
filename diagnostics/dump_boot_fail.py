"""Dump complet d'une entree BOOTSTRAP_CI_FAIL du 15m v7."""
import json

ARCH = r"D:\midas_v2\einherjar\outputs\archive\archive.jsonl"
for line in open(ARCH, encoding="utf-8"):
    line = line.strip()
    if not line:
        continue
    e = json.loads(line)
    if str(e.get("date_rejet", "")) >= "2026-08-10T19:12:00" and e.get("raison_rejet") == "BOOTSTRAP_CI_FAIL":
        print(json.dumps(e, ensure_ascii=False, indent=1)[:2400])
        break