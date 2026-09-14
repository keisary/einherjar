#!/usr/bin/env python
"""Verifie que le pipeline de features live reproduit les features de recherche.

Compare, pour un couple (asset, timeframe) :
  - les features calculees par `einherjar.signals.feature_pipeline.FeaturePipeline`
    sur l'OHLCV reelle (`technical_agent_dataset_brut`),
  - la matrice compilee MIDAS (`midasV3/src/data/compiled/<classe>/<tf>/<asset>_X.npy`,
    colonnes = `metadata.json:feature_names`),
sur les `feature_ref` du corpus de recherche (`outputs/corpus.jsonl`).

Sortie : tableau de statut par feature (OK / ECART / VIDE / ABSENT) + rapport JSON
dans `outputs/manifests/verify_live_features_<asset>_<tf>.json`.

Usage :
    python scripts/verify_live_features.py --asset BTCUSD --timeframe 1h \
        --asset-class crypto --tail 6000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from einherjar.signals.corpus_bridge import corpus_feature_refs  # noqa: E402
from einherjar.signals.feature_pipeline import FeaturePipeline  # noqa: E402

COMPILED_ROOT = Path("D:/midas_v2/midasV3/src/data/compiled")
CORPUS_PATH = PROJECT_ROOT / "outputs" / "corpus.jsonl"
MANIFEST_DIR = PROJECT_ROOT / "outputs" / "manifests"

# Seuils de verdict
CORR_OK = 0.99
CORR_ECART = 0.90


def load_ohlcv(asset: str, timeframe: str, asset_class: str, data_version: str, tail: int | None):
    """Charge l'OHLCV reel d'un actif via le provider du module research."""
    from einherjar.research.data.ohlcv import OhlcvProvider

    frame = OhlcvProvider().load(asset, timeframe, data_version, asset_class=asset_class)
    df = frame.df
    if tail and df.height > tail:
        df = df.tail(tail)
    return df


def load_compiled(asset: str, asset_class: str, timeframe: str):
    """Charge la matrice compilee MIDAS (X, timestamps, noms de colonnes)."""
    directory = COMPILED_ROOT / asset_class / timeframe
    meta_path = directory / "metadata.json"
    x_path = directory / f"{asset}_X.npy"
    ts_path = directory / f"{asset}_ts.npy"
    for p in (meta_path, x_path, ts_path):
        if not p.exists():
            raise FileNotFoundError(f"Artefact compile absent : {p}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    names = list(meta["feature_names"])
    x = np.load(x_path, mmap_mode="r")
    ts = np.load(ts_path, mmap_mode="r")
    return names, np.asarray(x), np.asarray(ts)


def _to_ns(values) -> np.ndarray:
    """Normalise des timestamps heterogenes en int64 nanosecondes."""
    arr = np.asarray(values)
    if arr.dtype.kind == "M":
        return arr.astype("datetime64[ns]").astype("int64")
    if arr.dtype.kind in "iu":
        scale = 1
        # Secondes / millisecondes / microsecondes / nanosecondes
        peak = int(arr.max()) if arr.size else 0
        if peak < 10**11:
            scale = 10**9
        elif peak < 10**14:
            scale = 10**6
        elif peak < 10**17:
            scale = 10**3
        return arr.astype("int64") * scale
    return arr.astype("datetime64[ns]").astype("int64")


def compare(live, live_ts, refs, names, x, ts):
    """Compare les colonnes live et compilees sur les timestamps communs.

    Returns:
        (lignes, index_live, index_compiled) ou lignes est une liste de dicts de statut.
    """
    live_ns = _to_ns(live_ts)
    comp_ns = _to_ns(ts)
    common = np.intersect1d(live_ns, comp_ns)
    if common.size == 0:
        raise RuntimeError("Aucun timestamp commun entre l'OHLCV live et la matrice compilee")
    live_pos = {v: i for i, v in enumerate(live_ns.tolist())}
    comp_pos = {v: i for i, v in enumerate(comp_ns.tolist())}

    rows = []
    for ref in refs:
        row = {"feature": ref}
        if ref not in live.columns:
            row["statut"] = "ABSENT_LIVE"
            rows.append(row)
            continue
        if ref not in names:
            row["statut"] = "ABSENT_COMPILE"
            rows.append(row)
            continue
        li = np.array([live_pos[t] for t in common.tolist()])
        ci = np.array([comp_pos[t] for t in common.tolist()])
        a = np.asarray(live[ref].to_numpy()[li], dtype=np.float64)
        b = np.asarray(x[ci, names.index(ref)], dtype=np.float64)
        # La matrice compilee est normalisee par compile_dataset.py : les colonnes
        # autres que OHLCV sont CLIPPEES a [-5, 5] (nan_to_num avant clip). On
        # reproduit exactement cette etape pour comparer des grandeurs identiques.
        a = np.clip(a, -5.0, 5.0)
        mask = np.isfinite(a) & np.isfinite(b)
        row["n_commun"] = int(common.size)
        row["n_valides"] = int(mask.sum())
        if mask.sum() < 10:
            row["statut"] = "VIDE"
            rows.append(row)
            continue
        aa, bb = a[mask], b[mask]
        if aa.std() == 0 or bb.std() == 0:
            corr = 1.0 if np.allclose(aa, bb) else 0.0
        else:
            corr = float(np.corrcoef(aa, bb)[0, 1])
        mad = float(np.mean(np.abs(aa - bb)))
        scale = float(np.mean(np.abs(bb))) or 1.0
        row.update(
            correlation=round(corr, 6),
            diff_moyenne=round(mad, 8),
            diff_relative=round(mad / scale, 6),
            live_moyenne=round(float(aa.mean()), 6),
            live_ecart=round(float(aa.std()), 6),
            compile_moyenne=round(float(bb.mean()), 6),
            compile_ecart=round(float(bb.std()), 6),
            part_saturee=round(float(np.mean(np.abs(bb) >= 4.9999)), 4),
        )
        if corr >= CORR_OK:
            row["statut"] = "OK"
        elif corr >= CORR_ECART:
            row["statut"] = "ECART"
        else:
            row["statut"] = "DIVERGENT"
        if row.get("part_saturee", 0.0) > 0.5 and "SATURATION_COMPILE" not in row["statut"]:
            row["statut"] = f"SATURE_COMPILE({row['statut']})"
        rows.append(row)
    return rows


def main() -> int:
    """Point d'entree CLI."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--asset", default="BTCUSD")
    ap.add_argument("--timeframe", default="1h")
    ap.add_argument("--asset-class", default="crypto")
    ap.add_argument("--data-version", default="v1")
    ap.add_argument("--tail", type=int, default=6000,
                    help="Bougies comparees, comptees depuis la fin de la fenetre compilee")
    ap.add_argument("--warmup", type=int, default=1500,
                    help="Bougies supplementaires avant la fenetre (fenetres roulantes)")
    ap.add_argument("--all-columns", action="store_true", help="Comparer les 246 colonnes compilees")
    args = ap.parse_args()

    refs = corpus_feature_refs(CORPUS_PATH)
    names_comp, x, ts = load_compiled(args.asset, args.asset_class, args.timeframe)
    if args.all_columns:
        refs = list(dict.fromkeys(refs + names_comp))
    print(f"feature_ref a comparer : {len(refs)}")

    df_full = load_ohlcv(args.asset, args.timeframe, args.asset_class, args.data_version, None)
    comp_ns = _to_ns(ts)
    live_ns_full = _to_ns(df_full["timestamp"])
    end = int(comp_ns.max())
    # Toutes les bougies live jusqu'a la fin de la matrice compilee, moins le warmup
    cutoff = int(comp_ns.min())
    upto = int(np.searchsorted(live_ns_full, end, side="right"))
    start = max(0, upto - args.tail - args.warmup)
    if upto <= start:
        raise RuntimeError("Aucune bougie live dans la fenetre compilee")
    df = df_full.slice(start, upto - start)
    print(f"OHLCV {args.asset}/{args.timeframe} : {df.height} bougies chargees "
          f"(compiled de {cutoff} a {end})")

    pipe = FeaturePipeline()
    t0 = time.time()
    enriched = pipe.compute(df, asset=args.asset, timeframe=args.timeframe)
    elapsed = time.time() - t0
    print(f"pipeline : {enriched.width} colonnes en {elapsed:.1f}s | {pipe.last_report}")

    print(f"compile : X={x.shape} colonnes={len(names_comp)}")

    dump = PROJECT_ROOT / "outputs" / f"_verify_live_{args.asset}_{args.timeframe}.parquet"
    enriched.write_parquet(dump)
    print(f"features live sauvegardees : {dump}")

    rows = compare(enriched, df["timestamp"], refs, names_comp, x, ts)
    statuts: dict[str, int] = {}
    for r in rows:
        statuts[r["statut"]] = statuts.get(r["statut"], 0) + 1
    print("\n=== Verdict ===")
    for k, v in sorted(statuts.items(), key=lambda kv: -kv[1]):
        print(f"  {k}: {v}")
    problematiques = [r for r in rows if r["statut"] not in ("OK",)]
    if problematiques:
        print("\n=== Features a corriger ===")
        for r in problematiques[:40]:
            extra = ""
            if "correlation" in r:
                extra = (f" corr={r['correlation']} diff_rel={r.get('diff_relative')}"
                         f" | live m={r.get('live_moyenne')} sd={r.get('live_ecart')}"
                         f" | compile m={r.get('compile_moyenne')} sd={r.get('compile_ecart')}")
            print(f"  {r['feature']:34s} {r['statut']:14s}{extra}")

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    out = MANIFEST_DIR / f"verify_live_features_{args.asset}_{args.timeframe}.json"
    out.write_text(
        json.dumps(
            {
                "asset": args.asset,
                "timeframe": args.timeframe,
                "asset_class": args.asset_class,
                "lignes_ohlcv": int(df.height),
                "colonnes_pipeline": int(enriched.width),
                "duree_s": round(elapsed, 2),
                "statuts": statuts,
                "features": rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nRapport : {out} ({datetime.now(timezone.utc).isoformat(timespec='seconds')})")
    return 0 if statuts.get("OK", 0) == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
