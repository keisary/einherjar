import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backtest.analyzer import select_top, print_summary

results_path = ROOT / "data" / "mini_results_20.json"
corpus_brut_path = ROOT / "data" / "mini_corpus_20.json"
output_path = ROOT / "data" / "test_corpus_v2.json"

thresholds = {
    "min_sharpe": 0.0,
    "min_winrate": 0.0,
    "min_profit_factor": 0.0,
    "max_drawdown": -1.0,
    "min_trades": 1,
    "min_trades_per_month": 0.0,
}

r = select_top(
    results_path=results_path,
    thresholds=thresholds,
    top_n=None,
    corpus_brut_path=corpus_brut_path,
    output_path=output_path,
)
print_summary(r)
print(f"Selectionne: {len(r)}")
