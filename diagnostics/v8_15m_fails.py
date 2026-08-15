"""Entrees v8 15m (cluster 21:10) : contenu des BOOTSTRAP_CI_FAIL et DIVERSITY_FAIL."""
import json

ARCH = r"D:\midas_v2\einherjar\outputs\archive\archive.jsonl"
for line in open(ARCH, encoding="utf-8"):
    line = line.strip()
    if not line:
        continue
    e = json.loads(line)
    if str(e.get("date_rejet", "")).startswith("2026-08-10T21:1"):
        r = e.get("raison_rejet")
        if r in ("BOOTSTRAP_CI_FAIL", "DIVERSITY_FAIL"):
            b = e.get("bootstrap_ci_val", {})
            print(r, "| p_dsr=", round(e.get("deflated_sharpe_ratio", float("nan")), 3),
                  "| pbo=", e.get("probability_of_backtest_overfitting"),
                  "| ci_low=", b.get("sharpe_ci_low") if isinstance(b, dict) else b)