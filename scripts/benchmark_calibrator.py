import json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backtest.calibrator import Calibrator
from backtest.data_source import load_ohlcv

# Test de vitesse : 1 Einher sur BTCUSD 15m, 10 fois
print("=== Benchmark calibrator ===")
df = load_ohlcv("BTCUSD", "15m", "crypto")
print(f"Dataframe shape: {df.shape}")

einher = {
    "name": "E_PATTERN_HAMMER_5m_1",
    "domain": "Pattern pur",
    "direction": "long",
    "timeframes": ["15m"],
    "trigger": "pattern_hammer == 1",
    "filters": [],
    "assets": [{"asset": "BTCUSD", "class": "crypto"}],
    "tp_rule": {"type": "atr_multiple", "value": 2.5},
    "sl_rule": {"type": "atr_multiple", "value": 1.5},
    "max_holding": "1d",
    "cooldown": "4h"
}

cal = Calibrator(
    corpus_path=ROOT / "data" / "mini_corpus_20.json",
    native_exits_path=ROOT / "config" / "native_exits.json",
)

times = []
for i in range(5):
    t0 = time.time()
    res = cal._simulate_einher(einher, df, "BTCUSD", "15m", "crypto")
    t1 = time.time()
    times.append(t1 - t0)
    print(f"Run {i+1}: {t1-t0:.3f}s, trades={res.n_trades if res else 0}")

avg = sum(times) / len(times)
print(f"\nMoyenne par Einher: {avg:.3f}s")
print(f"Estimation 7535 Einhers x 28 actifs x 1 TF: {avg * 7535 * 28 / 3600:.1f} heures")
