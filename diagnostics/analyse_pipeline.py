"""Analyse compacte d'un run pipeline multi-TF BTCUSD.

Lit : outputs/corpus.jsonl, outputs/admit_summary.json, outputs/selection.json,
les outputs/run_*/run.log, et les baselines JSON produits par TF.
"""
import json, sys
from pathlib import Path
from collections import Counter

ROOT = Path(r"D:\midas_v2\einherjar\outputs")

def load_json(p: Path):
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        return f"ERREUR: {e}"

print("=" * 70)
print("1. CORPUS FINAL (corpus.jsonl)")
corpus = ROOT / "corpus.jsonl"
if corpus.exists():
    lines = corpus.read_text(encoding="utf-8").strip().splitlines()
    print(f"  lignes corpus = {len(lines)}")
    for l in lines[:10]:
        print("   ", l[:180])
else:
    print("  corpus.jsonl absent")

print("=" * 70)
print("2. BILAN ADMISSION (admit_summary.json)")
adm = load_json(ROOT / "admit_summary.json")
if adm:
    for k, v in (adm.items() if isinstance(adm, dict) else []):
        if k in ("n_admitted", "n_total", "admission_rate", "n_rejected", "reasons"):
            print(f"  {k}: {str(v)[:300]}")
else:
    print("  pas de admit_summary.json")

print("=" * 70)
print("3. SELECTION GENERATEUR (selection.json)")
sel = load_json(ROOT / "selection.json")
if sel:
    s = json.dumps(sel, ensure_ascii=False)[:600]
    print("  ", s)
else:
    print("  pas de selection.json")

print("=" * 70)
print("4. RAPPORTS par run (run_*/run.log, resume des étapes)")
runs = sorted(ROOT.glob("run_*/run.log"), key=lambda p: p.stat().st_mtime, reverse=True)[:5]
for rp in runs:
    text = rp.read_text(encoding="utf-8")
    steps = [l.split("|")[-1].strip()[:70] for l in text.splitlines() if "[STEP" in l]
    wins = [l.strip()[:110] for l in text.splitlines() if "winner=" in l or "Comparaison termin" in l]
    admits = [l.strip()[:110] for l in text.splitlines() if "admit" in l.lower() and ("OK" in l or "admis" in l.lower())]
    print(f"\n--- {rp.parent.name} ---")
    print("  steps:", " | ".join(steps[:8]))
    if wins: print("  compare:", wins[-1])
    if admits: print("  admit:", admits[-2:])

print("=" * 70)
print("5. FICHIERS DE SYNTHESE (admit/holdout/refined)")
for f in ROOT.glob("*.json*"):
    if f.stat().st_size > 0:
        print(f"  {f.name}: {f.stat().st_size} octets")