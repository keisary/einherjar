"""Rapport de campagne BTCUSD 2026-08-10 : Einhers, rejets, moyennes.

Lit :
- outputs/corpus.jsonl            -> Einhers admis
- outputs/archive/archive.jsonl   -> rejets (append-only, filtre timestamp run)
- outputs/run_<ts>/run.log        -> verdicts admission par TF (X/Y admis,
                                     breakdown raisons)
Affiche un rapport par TF + global.
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(r"D:\midas_v2\einherjar")
CORPUS = ROOT / "outputs" / "corpus.jsonl"
ARCHIVE = ROOT / "outputs" / "archive" / "archive.jsonl"
RUN_DIRS = sorted(list((ROOT / "outputs").glob("run_20260810_16*")) + list((ROOT / "outputs").glob("run_20260810_17*")))

START = "2026-08-10T16:20:00"  # debut campagne v5


def ts_of(fname: str) -> str:
    m = re.search(r"run_(\d{8})_(\d{6})", fname)
    if not m:
        return ""
    d, t = m.groups()
    return f"{d[:4]}-{d[4:6]}-{d[6:]}T{t[:2]}:{t[2:4]}:{t[4:6]}"


def main() -> int:
    # 1) Corpus : Einhers admis (tout l'historique, puis filtre date)
    admis = []
    if CORPUS.exists():
        for line in CORPUS.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                admis.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    admis_run = [a for a in admis if str(a.get("date_admission", "")) >= START]

    # 2) Archive : rejets
    rejets = 0
    raisons = Counter()
    if ARCHIVE.exists():
        for line in ARCHIVE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(e.get("date_rejet", "")) >= START:
                rejets += 1
                raisons[str(e.get("raison_rejet", "?"))] += 1

    # 3) Verdicts admission par TF depuis les run.log
    per_tf = {}
    for rd in RUN_DIRS:
        log = rd / "run.log"
        if not log.exists():
            continue
        txt = log.read_text(encoding="utf-8", errors="replace")
        m_rejet = re.search(r"Admission terminée : (\d+)/(\d+) admis", txt)
        m_brk = re.search(r"Breakdown rejets : (\{.*\})", txt)
        m_tf = re.search(r"OHLCV chargé : BTCUSD × (\S+)", txt)
        if m_rejet and m_tf:
            tf = m_tf.group(1)
            brk = {}
            if m_brk:
                try:
                    brk = json.loads(m_brk.group(1).replace("'", '"'))
                except json.JSONDecodeError:
                    brk = {}
            per_tf[tf] = {
                "admis": int(m_rejet.group(1)),
                "evalues": int(m_rejet.group(2)),
                "breakdown": brk,
            }

    print("=" * 64)
    print("RAPPORT CAMPAIGNE BTCUSD 2026-08-10 (mono TypedGP, 4 TF)")
    print("=" * 64)
    print(f"Admissions par TF (lu dans les run.log) :")
    for tf in ("15m", "1h", "4h", "5m"):
        v = per_tf.get(tf)
        if v:
            print(f"  {tf:>4}: {v['admis']}/{v['evalues']} admis "
                  f"-> rejets: {v['breakdown']}")
        else:
            print(f"  {tf:>4}: (pas encore terminé)")
    print(f"\nEinhers admis au corpus (run): {len(admis_run)}")
    if admis_run:
        keys = [k for k in ("sharpe_net", "cagr", "n_trades_total")
                if k in admis_run[0]]
        if keys:
            print("Moyennes globales des admis (run) :")
            n = len(admis_run)
            for k in keys:
                vals = [float(a.get(k, 0) or 0) for a in admis_run]
                print(f"  {k}: moyenne={sum(vals)/n:.4f} "
                      f"min={min(vals):.4f} max={max(vals):.4f}")
        else:
            print("  (clés du corpus:", sorted(admis_run[0].keys())[:12], ")")
    print(f"\nHypothèses rejetées (archive, run): {rejets}")
    for raison, n in raisons.most_common():
        print(f"  {raison}: {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())