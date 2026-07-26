"""
Script de calibration backtest EINHERJAR.

Usage :
  python scripts/run_calibration.py

Processus :
  1. Charge corpus_brut_v2.json
  2. Pour chaque (asset, tf), charge les donnees MIDAS
  3. Simule chaque Einher avec SL/TP et frais
  4. Sauvegarde les resultats dans data/calibration_results.json
  5. Lance l'analyseur pour generer corpus_v2.json

L'utilisateur peut ajuster les seuils et top_n dans config/calibration.json
et relancer l'analyseur seul sans refaire le backtest.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backtest.calibrator import Calibrator
from backtest.analyzer import select_top, print_summary

CORPUS_BRUT = ROOT / "config" / "corpus_brut_v2.json"
NATIVE_EXITS = ROOT / "config" / "native_exits.json"
CALIBRATION_CONFIG = ROOT / "config" / "calibration.json"
RESULTS_PATH = ROOT / "data" / "calibration_results.json"
CORPUS_V2 = ROOT / "config" / "corpus_v2.json"


def main():
    # Charger config calibration
    config = {}
    if CALIBRATION_CONFIG.exists():
        with open(CALIBRATION_CONFIG, "r", encoding="utf-8") as f:
            config = json.load(f)

    # Creer dossier data si besoin
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  EINHERJAR — Phase 3 : Calibration Backtest")
    print("=" * 60)

    cal = Calibrator(
        corpus_path=CORPUS_BRUT,
        native_exits_path=NATIVE_EXITS,
        config=config.get("backtest_params"),
    )
    cal.run(output_path=RESULTS_PATH, progress_every=5)

    print("\n--- Analyse des resultats ---")
    thresholds = config.get("thresholds", {})
    top_n = thresholds.get("top_n") if isinstance(thresholds, dict) else None
    selected = select_top(
        results_path=RESULTS_PATH,
        thresholds=thresholds,
        top_n=top_n,
        corpus_brut_path=CORPUS_BRUT,
        output_path=CORPUS_V2,
    )
    print_summary(selected)

    print(f"\nFichiers generes :")
    print(f"  - {RESULTS_PATH}")
    print(f"  - {CORPUS_V2}")


if __name__ == "__main__":
    main()
