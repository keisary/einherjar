"""scripts/batch_search.py — Recherche STGP+admission sur toutes les paires d'actifs.

Lit assets_v1.json et lance discovery.admit (moteur trend signal, lexicase +
MAP-Elites) sur chaque actif, avec un timeframe par défaut selon la classe.
Résilient : chaque paire est isolée, les erreurs sont loggées et n'arrêtent
pas le batch. Les résultats (admis/rejetés + raisons) sont appendés dans
outputs/batch_results.jsonl au fil de l'eau (survit aux coupures machine).

Usage:
  export PYTHONPATH=src
  python scripts/batch_search.py                # toutes les paires (défaut TF)
  python scripts/batch_search.py --tf 15m       # force un timeframe unique
  python scripts/batch_search.py --only BTCUSD  # filtre par actif
  python scripts/batch_search.py --pop 10 --gen 4 --n-eval 12
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS_JSON = REPO_ROOT / "src/einherjar/research/config/assets_v1.json"
RESULTS_PATH = REPO_ROOT / "outputs/batch_results.jsonl"
PY = r"D:/midas_v2/midas/Scripts/python.exe"

# Timeframe par défaut par classe (choix guidé par le test de signal :
# 15m pour les actifs très liquides à forte fréquence, 1h pour forex,
# 4h pour indices/commodities).
DEFAULT_TF = {
    "crypto": "15m",
    "forex": "1h",
    "indices": "4h",
    "commodities": "4h",
    "stocks_tech": "15m",
    "stocks_growth": "15m",
    "stocks_value": "15m",
}


def load_assets() -> list[dict]:
    with open(ASSETS_JSON, "r", encoding="utf-8") as f:
        return json.load(f)["assets"]


def run_one(asset: str, asset_class: str, tf: str, args: argparse.Namespace) -> dict:
    """Lance discovery.admit sur une paire et retourne le résumé."""
    cmd = [
        PY, "-m", "einherjar.research.discovery", "admit",
        "--data-asset", asset,
        "--data-class", asset_class,
        "--data-timeframe", tf,
        "--pop-size", str(args.pop),
        "--n-gen", str(args.gen),
        "--n-eval", str(args.n_eval),
        "--horizon-index", "1",
        "--selection", "lexicase",
        "--log-level", "INFO",
    ]
    if args.use_sltp:
        cmd.append("--use-sltp")
    t0 = time.time()
    import os
    env = dict(os.environ)
    env["PYTHONPATH"] = "src" + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    proc = subprocess.run(
        cmd, cwd=str(REPO_ROOT), env=env,
        capture_output=True, text=True, timeout=args.timeout,
    )
    elapsed = round(time.time() - t0, 1)
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    # Les logs (logger Python) sortent sur STDERR ; on parse les deux flux.
    combined = stdout + "\n" + stderr
    n_admitted = n_rejected = 0
    for line in combined.splitlines():
        if "Admission :" in line:
            import re
            m = re.search(r"Admission : (\d+) admis, (\d+) rejetés", line)
            if m:
                n_admitted, n_rejected = int(m.group(1)), int(m.group(2))
    return {
        "asset": asset, "class": asset_class, "timeframe": tf,
        "n_admitted": n_admitted, "n_rejected": n_rejected,
        "elapsed_s": elapsed, "exit_code": proc.returncode,
        "stderr_tail": stderr[-400:] if proc.returncode != 0 else "",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Batch search STGP+admission")
    parser.add_argument("--tf", default=None, help="Timeframe unique (override la classe)")
    parser.add_argument("--only", default=None, help="Filtrer par actif (ex: BTCUSD)")
    parser.add_argument("--pop", type=int, default=40)
    parser.add_argument("--gen", type=int, default=10)
    parser.add_argument("--n-eval", type=int, default=100)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--use-sltp", action="store_true", default=False,
                        help="Utilise SL/TP calibré (défaut: mode hold).")
    args = parser.parse_args(argv)

    assets = load_assets()
    if args.only:
        assets = [a for a in assets if a["asset"] == args.only]
        if not assets:
            print(f"Actif inconnu : {args.only}")
            return 2

    print(f"Batch : {len(assets)} actifs | pop={args.pop} gen={args.gen} n-eval={args.n_eval}")
    print(f"Résultats appendés dans {RESULTS_PATH}")

    for a in assets:
        asset, cls = a["asset"], a["class"]
        tf = args.tf or DEFAULT_TF.get(cls, "1h")
        try:
            res = run_one(asset, cls, tf, args)
        except subprocess.TimeoutExpired:
            res = {"asset": asset, "class": cls, "timeframe": tf,
                   "n_admitted": 0, "n_rejected": 0, "elapsed_s": args.timeout,
                   "exit_code": -1, "stderr_tail": "TIMEOUT"}
        line = json.dumps(res, ensure_ascii=False)
        with open(RESULTS_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        status = f"admis={res['n_admitted']}" if res["n_admitted"] else f"rejetés={res['n_rejected']}"
        print(f"  [{asset:>10} {cls:<12} {tf:<4}] {status} ({res['elapsed_s']}s)")
        if res.get("stderr_tail"):
            print(f"       err: {res['stderr_tail'][-120:]}")

    print("Batch terminé.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())