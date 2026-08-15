"""Les BOOTSTRAP_CI_FAIL du 15m v7 (cluster 19:35-19:42) avec tous les champs."""
import json

ARCH = r"D:\midas_v2\einherjar\outputs\archive\archive.jsonl"
for line in open(ARCH, encoding="utf-8"):
    line = line.strip()
    if not line:
        continue
    e = json.loads(line)
    d = str(e.get("date_rejet", ""))
    if "2026-08-10T19:3" <= d <= "2026-08-10T19:4" and e.get("raison_rejet") == "BOOTSTRAP_CI_FAIL":
        print(json.dumps({
            "id": e.get("id"),
            "raison": e.get("raison_rejet"),
            "n_trades": e.get("n_trades"),
            "sharpe_net": e.get("sharpe_net_val"),
            "ci": e.get("bootstrap_ci_val"),
            "p_dsr": e.get("deflated_sharpe_ratio"),
            "pbo": e.get("probability_of_backtest_overfitting"),
        }, ensure_ascii=False))