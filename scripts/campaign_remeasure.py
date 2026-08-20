"""campaign_remeasure.py - Campagne de re-mesure.

Sprint 3.6.

Apres les corrections de bugs (Sprints 3.3+3.4) et des limitations (Sprint 3.5),
on relance les campagnes pour avoir les VRAIS chiffres.
"""
from __future__ import annotations

import json
import logging
import subprocess
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
logger = logging.getLogger("campaign")


# Vrais actifs (selection des plus liquides par classe)
ASSET_CLASSES = {
    "crypto": ["BTCUSD", "ETHUSD", "LTCUSD", "BCHUSD", "ADAUSD"],
    "forex": ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"],
    "commodities": ["WTIUSD", "BRENT", "NATGAS", "COPPER", "COCOA"],
    "indices": ["SP500", "NASDAQ100", "DOWJONES", "DAX40", "FTSE100"],
    "stocks_growth": ["AAPL", "MSFT", "AMZN", "GOOGL", "META"],
    "stocks_tech": ["NVDA", "AMD", "INTC", "CSCO", "ORCL"],
    "stocks_value": ["JPM", "BAC", "GS", "JNJ", "KO"],
}

# Horizons a tester pour BTC
HORIZONS = ["6h", "12h", "1d", "2d"]


def run_command(cmd: list[str], timeout: int = 1800) -> dict:
    """Execute une commande et retourne un dict avec stdout/stderr/returncode."""
    logger.info(f"CMD: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "cmd": " ".join(cmd),
            "returncode": result.returncode,
            "stdout_tail": "\n".join(result.stdout.splitlines()[-30:]),
            "stderr_tail": "\n".join(result.stderr.splitlines()[-10:]) if result.stderr else "",
        }
    except subprocess.TimeoutExpired:
        return {"cmd": " ".join(cmd), "returncode": -1, "error": "TIMEOUT"}


def run_pipeline(
    assets: list[str],
    asset_class: str,
    horizon: str,
    scope: str = "asset",
    output: str = None,
) -> dict:
    """Lance le pipeline xgb_einhers via CLI."""
    if output is None:
        tag = "_".join(assets[:3]) if len(assets) <= 3 else f"multi_{len(assets)}"
        output = f"outputs/campaign_{asset_class}_{tag}_{horizon}.jsonl"

    cmd = [
        "D:/midas_v2/midas/Scripts/python.exe",
        "-m", "einherjar.research.xgb_einhers.runner", "run",
        "--timeframe", "1h",
        "--horizon", horizon,
        "--n-estimators", "100",
        "--max-depth", "3",
        "--max-paths", "30",
        "--min-score", "0.0005",
        "--debug",
        "--regularized",
        "--apply-dedup",
        "--drop-sparse",
        "--min-holdout-trades", "5",
        "--output", output,
    ]
    if scope == "market" or len(assets) > 1:
        cmd.extend(["--scope", "market"])
        cmd.extend(["--asset-classes", asset_class])
        cmd.extend(["--max-assets", str(len(assets))])
    else:
        cmd.extend(["--asset", assets[0]])
        cmd.extend(["--asset-class", asset_class])
    return run_command(cmd)


def main():
    results = {
        "multi_asset_per_class_2d": {},
        "multi_horizon_btc": {},
    }
    t0 = time.time()

    # Etape 1 : Multi-actif par classe x 2d
    logger.info("=" * 70)
    logger.info("ETAPE 1 : Multi-actif par classe x 2d (7 classes x 1 horizon)")
    logger.info("=" * 70)
    for asset_class, assets in ASSET_CLASSES.items():
        logger.info(f"\nClasse {asset_class} : {assets}")
        result = run_pipeline(assets, asset_class, "2d", scope="market")
        results["multi_asset_per_class_2d"][asset_class] = {
            "assets": assets,
            "returncode": result["returncode"],
            "stdout_tail": result.get("stdout_tail", ""),
        }
        logger.info(f"  returncode={result['returncode']}")

    # Etape 2 : Multi-horizon BTC (4 horizons)
    logger.info("=" * 70)
    logger.info("ETAPE 2 : Multi-horizon BTC (4 horizons)")
    logger.info("=" * 70)
    for horizon in HORIZONS:
        logger.info(f"\nHorizon {horizon}")
        result = run_pipeline(["BTCUSD"], "crypto", horizon, scope="asset")
        results["multi_horizon_btc"][horizon] = {
            "returncode": result["returncode"],
            "stdout_tail": result.get("stdout_tail", ""),
        }
        logger.info(f"  returncode={result['returncode']}")

    # Sauver
    elapsed = time.time() - t0
    results["total_seconds"] = elapsed
    out = Path("D:/midas_v2/Einherjar/outputs/campaign_remeasure_2026-08-20.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"\nRapport sauvegarde dans {out}")
    logger.info(f"Temps total : {elapsed:.1f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()
