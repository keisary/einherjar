#!/usr/bin/env python
"""Smoke test d'inference de bout en bout (chemin live, hors broker).

Chaine testee :
    OHLCV reel -> FeaturePipeline (schema MIDAS) -> EinherEngine (corpus JSONL) -> signaux

Verifie aussi la COUVERTURE : pour chaque einher du couple (asset, timeframe), toutes
les `feature_ref` de sa condition doivent exister dans les colonnes produites par le
pipeline (sinon la condition serait evaluee a False en silence).

Usage :
    python scripts/smoke_inference.py --asset BTCUSD --timeframe 1h --asset-class crypto
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from einherjar.core.enums import TimeFrame  # noqa: E402
from einherjar.signals.corpus_bridge import feature_refs, load_entries  # noqa: E402
from einherjar.signals.einher_engine import EinherEngine  # noqa: E402
from einherjar.signals.feature_pipeline import FeaturePipeline  # noqa: E402

CORPUS_PATH = PROJECT_ROOT / "outputs" / "corpus.jsonl"
MANIFEST_DIR = PROJECT_ROOT / "outputs" / "manifests"


def main() -> int:
    """Point d'entree."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--asset", default="BTCUSD")
    ap.add_argument("--timeframe", default="1h")
    ap.add_argument("--asset-class", default="crypto")
    ap.add_argument("--data-version", default="v1")
    ap.add_argument("--rows", type=int, default=1600)
    args = ap.parse_args()

    from einherjar.research.data.ohlcv import OhlcvProvider

    df = OhlcvProvider().load(args.asset, args.timeframe, args.data_version,
                              asset_class=args.asset_class).df.tail(args.rows)
    print(f"OHLCV {args.asset}/{args.timeframe} : {df.height} bougies")

    t0 = time.time()
    pipe = FeaturePipeline(max_lookback=args.rows)
    enriched = pipe.compute(df, asset=args.asset, timeframe=args.timeframe)
    duree = time.time() - t0
    print(f"features : {enriched.width} colonnes en {duree:.1f}s")

    # --- Couverture des conditions du corpus pour ce couple (asset, timeframe)
    entries = [e for e in load_entries(CORPUS_PATH)
               if (e.get("universe") or {}).get("asset") == args.asset
               and (e.get("universe") or {}).get("timeframe") == args.timeframe]
    besoin: set[str] = set()
    for e in entries:
        besoin.update(feature_refs(e.get("condition_tree")))
    manquantes = sorted(r for r in besoin if r not in enriched.columns)
    print(f"einhers {args.asset}/{args.timeframe} : {len(entries)} | feature_ref requises : {len(besoin)}"
          f" | manquantes : {len(manquantes)} {manquantes}")

    # --- Evaluation (chemin live : EinherEngine sur la derniere bougie)
    engine = EinherEngine()
    engine.load_corpus(str(CORPUS_PATH))
    t0 = time.time()
    signals, forming = engine.evaluate(enriched, args.asset, TimeFrame(args.timeframe))
    print(f"evaluation : {len(signals)} signal(aux), {len(forming)} en formation "
          f"en {time.time()-t0:.1f}s")

    for sig in signals[:5]:
        print(f"  SIGNAL {sig.einher_name} [{sig.direction.value}] entry={sig.entry_price:.4f} "
              f"tp={sig.tp_price:.4f} sl={sig.sl_price:.4f} conf={sig.confidence}")

    # --- Non-regression : un trigger sans feature manquante doit pouvoir s'evaluer
    exemple = [e for e in entries if e.get("condition_tree")][:1]
    if exemple:
        from einherjar.signals.corpus_bridge import entry_to_einher

        ein = entry_to_einher(exemple[0])
        ok = engine._eval_condition(enriched, ein.trigger)
        print(f"evaluation directe du 1er einher ({ein.name}) : {ok}")

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    out = MANIFEST_DIR / f"smoke_inference_{args.asset}_{args.timeframe}.json"
    out.write_text(json.dumps({
        "asset": args.asset,
        "timeframe": args.timeframe,
        "bougies": int(df.height),
        "colonnes_features": int(enriched.width),
        "duree_pipeline_s": round(duree, 2),
        "einhers_couple": len(entries),
        "feature_refs_requises": sorted(besoin),
        "feature_refs_manquantes": manquantes,
        "signaux": len(signals),
        "en_formation": len(forming),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nRapport : {out}")
    return 0 if not manquantes else 1


if __name__ == "__main__":
    raise SystemExit(main())
