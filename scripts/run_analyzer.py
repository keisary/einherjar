"""
Script standalone d'analyse des resultats de calibration.

Permet de relancer la selection avec des seuils differents
sans refaire le backtest complet.

Usage :
  python scripts/run_analyzer.py --min-sharpe 0.5 --min-winrate 0.50
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backtest.analyzer import select_top, print_summary

RESULTS_PATH = ROOT / "data" / "calibration_results.json"
CORPUS_BRUT = ROOT / "config" / "corpus_brut_v1.json"
CORPUS_V2 = ROOT / "config" / "corpus_v2.json"


def main():
    parser = argparse.ArgumentParser(description="Analyse des resultats EINHERJAR")
    parser.add_argument("--min-sharpe", type=float, default=2.3)
    parser.add_argument("--min-winrate", type=float, default=0.60)
    parser.add_argument("--min-pf", type=float, default=1.0)
    parser.add_argument("--max-dd", type=float, default=-0.20)
    parser.add_argument("--min-trades", type=int, default=12)
    parser.add_argument("--top-n", type=int, default=None)
    parser.add_argument("--output", type=str, default=str(CORPUS_V2))
    args = parser.parse_args()

    thresholds = {
        "min_sharpe": args.min_sharpe,
        "min_winrate": args.min_winrate,
        "min_profit_factor": args.min_pf,
        "max_drawdown": args.max_dd,
        "min_trades": args.min_trades,
        "min_trades_per_month": 0.3,
    }

    selected = select_top(
        results_path=RESULTS_PATH,
        thresholds=thresholds,
        top_n=args.top_n,
        corpus_brut_path=CORPUS_BRUT,
        output_path=Path(args.output),
    )
    print_summary(selected)


if __name__ == "__main__":
    main()
