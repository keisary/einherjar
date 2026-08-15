"""Repro : rejouer le chemin exact d'admission sur une VRAIE hyp BOOTSTRAP_CI_FAIL (15m v7)."""
import warnings
warnings.filterwarnings("ignore")
import sys
import json
sys.path.insert(0, "src")

from einherjar.research.discovery import _load_real_data
from einherjar.research.engine.evaluator import EvaluationEngine
from einherjar.research.utils.types import Hypothesis, Amplitude
from einherjar.research.config.loader import load_config

ARCH = r"D:/midas_v2/archive_backup_20260810.jsonl"
cfg = load_config("src/einherjar/research/config")

# 1) trouver la premiere hyp BOOTSTRAP_CI_FAIL du 15m v7 (19:40)
for line in open(ARCH, encoding="utf-8"):
    line = line.strip()
    if not line:
        continue
    e = json.loads(line)
    if ("2026-08-10T19:40:00" <= str(e.get("date_rejet", "")) <= "2026-08-10T19:42:00"
            and e.get("raison_rejet") == "BOOTSTRAP_CI_FAIL"):
        el = e["element"]
        amp = el.get("amplitude", 2.0)
        if isinstance(amp, dict):
            amp = Amplitude(valeur=amp.get("valeur", 2.0), unité=amp.get("unité", "ABSOLU"))
        hyp = Hypothesis(
            id=1,
            condition_tree=el["condition_tree"],
            direction=el.get("direction", "long"),
            universe=el.get("universe", "BTCUSD"),
            amplitude=amp,
            cooldown_k=el.get("cooldown_k", 1),
        )
        print("hyp replay:", e["id"], "| p_dsr=", round(e.get("deflated_sharpe_ratio", 0), 3))
        break

# 2) charger les donnees reelles 15m
train_ohlcv, train_features, val_ohlcv, val_features, _, _, _ = _load_real_data(
    config=cfg, data_root=r"D:\midas_v2\midasV3\src\data\compiled", asset="BTCUSD",
    asset_class="crypto", timeframe="15m",
)

# 3) calibration + test_on AVEC bootstrap (defaut True), chemin exact de l'admission
eng = EvaluationEngine(config=cfg, data_version="raw", seed=42)
cal = eng.train_calibrate(hyp, train_ohlcv, train_features)
m = eng.test_on(hyp, val_ohlcv, val_features, cal, "val")
print("sharpe_net:", round(m.sharpe_net, 3))
print("bootstrap_sharpe_ci_low:", m.bootstrap_sharpe_ci_low, "high:", m.bootstrap_sharpe_ci_high)
print("VERDICT_BOOTSTRAP:", "PASS" if m.bootstrap_sharpe_ci_low and m.bootstrap_sharpe_ci_low > 0 else "FAIL")