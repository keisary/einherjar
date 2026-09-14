#!/usr/bin/env python
"""Compare les features einherjar (copies portees) aux enrichisseurs MIDAS ORIGINAUX.

Les deux chaines tournent sur EXACTEMENT le meme OHLCV (meme tranche) :
  - MIDAS   : scripts/midas_reference_features.py (module original data_enrichment)
  - EINHERJAR : einherjar.signals.feature_pipeline (copies portees + branchements)

Verdict par colonne : IDENTIQUE (ecart max < 1e-6), PROCHE (corr >= 0.9999),
DIVERGENT, MANQUANT_*. Rapport JSON dans outputs/manifests/midas_vs_einherjar_*.json.

Usage :
    python scripts/compare_midas_vs_einherjar.py --asset BTCUSD --timeframe 1h --rows 5500
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from einherjar.signals.corpus_bridge import corpus_feature_refs  # noqa: E402
from einherjar.signals.feature_pipeline import FeaturePipeline  # noqa: E402

CORPUS_PATH = PROJECT_ROOT / "outputs" / "corpus.jsonl"
MANIFEST_DIR = PROJECT_ROOT / "outputs" / "manifests"


def load_slice(asset: str, timeframe: str, asset_class: str, data_version: str, rows: int) -> pl.DataFrame:
    """Charge la tranche OHLCV commune aux deux chaines (derniere `rows` bougies)."""
    from einherjar.research.data.ohlcv import OhlcvProvider

    df = OhlcvProvider().load(asset, timeframe, data_version, asset_class=asset_class).df
    return df.tail(rows)


def run_reference(asset: str, timeframe: str, asset_class: str, data_version: str,
                  rows: int, out: Path) -> Path:
    """Execute la chaine MIDAS de reference dans un sous-processus."""
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "midas_reference_features.py"),
        "--asset", asset,
        "--timeframe", timeframe,
        "--asset-class", asset_class,
        "--data-version", data_version,
        "--rows", str(rows),
        "--out", str(out),
    ]
    print("→ reference MIDAS :", " ".join(cmd[-6:]))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    tail = "\n".join(proc.stdout.splitlines()[-8:])
    print(tail)
    if proc.returncode != 0:
        print(proc.stderr[-2000:])
        raise SystemExit(f"reference MIDAS en echec (code {proc.returncode})")
    return out


def compare_frames(live: pl.DataFrame, ref: pl.DataFrame, refs: list[str]) -> list[dict]:
    """Compare colonne a colonne les features live et la reference MIDAS."""
    common = [c for c in live.columns if c in ref.columns]
    rows: list[dict] = []
    for col in common:
        a = live[col].to_numpy()
        b = ref[col].to_numpy()
        if a.dtype.kind not in "fiu" or b.dtype.kind not in "fiu":
            continue
        a = a.astype(np.float64)
        b = b.astype(np.float64)
        mask = np.isfinite(a) & np.isfinite(b)
        row = {"colonne": col, "dans_corpus": col in refs, "n_valides": int(mask.sum())}
        if mask.sum() == 0:
            row["statut"] = "AUCUNE_VALEUR"
            rows.append(row)
            continue
        aa, bb = a[mask], b[mask]
        diff = np.abs(aa - bb)
        scale = np.maximum(np.abs(bb), 1e-12)
        row["ecart_max_rel"] = float(np.max(diff / scale))
        row["ecart_moyen"] = float(diff.mean())
        if aa.std() > 0 and bb.std() > 0:
            row["correlation"] = float(np.corrcoef(aa, bb)[0, 1])
        else:
            row["correlation"] = 1.0 if np.allclose(aa, bb) else 0.0
        if row["ecart_max_rel"] < 1e-6:
            row["statut"] = "IDENTIQUE"
        elif row["correlation"] >= 0.9999 or row["ecart_max_rel"] < 1e-3:
            row["statut"] = "PROCHE"
        elif row["correlation"] >= 0.99:
            row["statut"] = "ECART"
        else:
            row["statut"] = "DIVERGENT"
        rows.append(row)
    return rows


def main() -> int:
    """Point d'entree."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--asset", default="BTCUSD")
    ap.add_argument("--timeframe", default="1h")
    ap.add_argument("--asset-class", default="crypto")
    ap.add_argument("--data-version", default="v1")
    ap.add_argument("--rows", type=int, default=5500)
    ap.add_argument("--skip-ref", action="store_true", help="Reutiliser la reference deja calculee")
    args = ap.parse_args()

    refs = corpus_feature_refs(CORPUS_PATH)
    df = load_slice(args.asset, args.timeframe, args.asset_class, args.data_version, args.rows)
    print(f"tranche OHLCV : {df.height} bougies ({args.asset}/{args.timeframe})")

    ref_path = PROJECT_ROOT / "outputs" / f"_ref_midas_{args.asset}_{args.timeframe}.parquet"
    if not (args.skip_ref and ref_path.exists()):
        run_reference(args.asset, args.timeframe, args.asset_class, args.data_version,
                      args.rows, ref_path)
    reference = pl.read_parquet(ref_path)
    print(f"reference MIDAS : {reference.height} x {reference.width}")

    t0 = time.time()
    live = FeaturePipeline().compute(df, asset=args.asset, timeframe=args.timeframe)
    print(f"pipeline einherjar : {live.width} colonnes en {time.time()-t0:.1f}s")

    rows = compare_frames(live, reference, refs)
    statuts: dict[str, int] = {}
    for r in rows:
        statuts[r["statut"]] = statuts.get(r["statut"], 0) + 1
    print("\n=== Verdict (colonnes communes) ===")
    for k, v in sorted(statuts.items(), key=lambda kv: -kv[1]):
        print(f"  {k}: {v}")

    manquants = [c for c in refs if c not in live.columns]
    absents_ref = [c for c in refs if c not in reference.columns]
    print(f"\nfeature_ref du corpus absentes du live : {len(manquants)} {manquants[:12]}")
    print(f"feature_ref du corpus absentes de la reference MIDAS : {len(absents_ref)} "
          f"{absents_ref[:8]}")

    pbs = [r for r in rows if r["statut"] not in ("IDENTIQUE", "PROCHE")]
    if pbs:
        print("\n=== Colonnes a corriger ===")
        par_statut: dict[str, list[dict]] = {}
        for r in pbs:
            par_statut.setdefault(r["statut"], []).append(r)
        for statut, items in sorted(par_statut.items(), key=lambda kv: -len(kv[1])):
            print(f"  {statut} ({len(items)}) :")
            for r in items[:15]:
                print(f"    {r['colonne']:34s} corr={r.get('correlation'):.5f} "
                      f"ecart_rel={r.get('ecart_max_rel'):.2e} n={r['n_valides']}")

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    out = MANIFEST_DIR / f"midas_vs_einherjar_{args.asset}_{args.timeframe}.json"
    out.write_text(json.dumps({
        "asset": args.asset,
        "timeframe": args.timeframe,
        "lignes": int(df.height),
        "colonnes_live": int(live.width),
        "colonnes_reference": int(reference.width),
        "statuts": statuts,
        "feature_refs_absentes_live": manquants,
        "feature_refs_absentes_reference": absents_ref,
        "colonnes": rows,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nRapport : {out}")
    echecs = statuts.get("DIVERGENT", 0) + statuts.get("AUCUNE_VALEUR", 0) + len(manquants)
    return 0 if echecs == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
