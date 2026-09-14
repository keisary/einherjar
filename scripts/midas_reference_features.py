#!/usr/bin/env python
"""Reference MIDAS : calcule les features avec les enrichisseurs MIDAS ORIGINAUX.

Sert de verite terrain pour verifier les copies portees dans einherjar
(le pipeline compiled/*.npy est clippe a [-5,5] et zero-fill : inutilisable
pour une comparaison valeur a valeur).

Sortie : parquet avec les colonnes OHLCV + technique + quant + pattern_* + Factor_*/signaux.

Usage :
    python scripts/midas_reference_features.py --asset BTCUSD --timeframe 1h \
        --asset-class crypto --start 2025-01-01 --out ref.parquet
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

MIDAS_DIR = Path("D:/midas_v2/midasV3/src/agents/technical/data_enrichment")
STRAT_DIR = Path("D:/midas_v2/midasV3/src/agents/technical/strategy_generator")

sys.path.insert(0, str(MIDAS_DIR))
sys.path.insert(0, str(STRAT_DIR))


def main() -> int:
    """Point d'entree."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--asset", default="BTCUSD")
    ap.add_argument("--timeframe", default="1h")
    ap.add_argument("--asset-class", default="crypto")
    ap.add_argument("--data-version", default="v1")
    ap.add_argument("--rows", type=int, default=5500)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sys.path.insert(0, "D:/midas_v2/einherjar/src")
    import numpy as np
    import polars as pl
    from einherjar.research.data.ohlcv import OhlcvProvider

    frame = OhlcvProvider().load(args.asset, args.timeframe, args.data_version,
                                 asset_class=args.asset_class)
    df = frame.df.tail(args.rows)
    pdf = df.to_pandas()
    pdf["asset"] = args.asset
    pdf["timeframe"] = args.timeframe
    print(f"OHLCV {args.asset}/{args.timeframe}: {pdf.shape}")

    from technical_indicators import TechnicalIndicatorsEnricher  # type: ignore
    from quantitative_features import OptimizedQuantitativeFeaturesEnricher  # type: ignore

    t0 = time.time()
    tech = TechnicalIndicatorsEnricher(mode="full", chunk_size=50000, max_memory_gb=8.0,
                                       n_jobs=1, use_dask=False, _is_worker=True)
    pdf = tech.enrich_dataframe(pdf)
    print(f"  technique MIDAS : {pdf.shape[1]} colonnes en {time.time()-t0:.1f}s")

    t0 = time.time()
    quant = OptimizedQuantitativeFeaturesEnricher(chunk_size=50000, max_memory_gb=4.0)
    pdf = quant.enrich_dataset(pdf)
    print(f"  quant MIDAS     : {pdf.shape[1]} colonnes en {time.time()-t0:.1f}s")

    t0 = time.time()
    from numba_pattern_detectors import (  # type: ignore
        PATTERN_THRESHOLDS,
        NumbaPatternDetectors,
        PatternMetadataManager,
    )

    manager = PatternMetadataManager(PATTERN_THRESHOLDS)
    detector = NumbaPatternDetectors(manager)
    patterns = manager.get_all_patterns()
    ohlcv = pl.from_pandas(pdf[["timestamp", "open", "high", "low", "close", "volume"]])
    detected = detector.detect(ohlcv, patterns)
    for name, arr in detected.items():
        if len(np.asarray(arr)) == len(pdf):
            pdf[f"pattern_{name}"] = np.asarray(arr, dtype=np.float32)
    print(f"  patterns MIDAS  : {len(detected)} en {time.time()-t0:.1f}s")

    t0 = time.time()
    from alpha_factors.calculator import AlphaFactorCalculator  # type: ignore

    calc = AlphaFactorCalculator()
    ldf = pl.from_pandas(pdf).lazy()
    out = calc.calculate_base_factors(ldf)
    try:
        out = calc.calculate_sophisticated_factors_lazy(out)
    except Exception as exc:
        print(f"  facteurs sophistiques indisponibles : {str(exc)[:120]}")
    ref = out.collect()
    print(f"  facteurs MIDAS  : {ref.width} colonnes en {time.time()-t0:.1f}s")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    ref.write_parquet(args.out)
    print(f"reference ecrite : {args.out} ({ref.height} x {ref.width})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
