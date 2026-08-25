# pyright: reportArgumentType=false, reportAssignmentType=false, reportCallIssue=false
"""runner.py — Pipeline de recherche : STGP + MAP-Elites → admission C1-C6 → corpus.

Orchestration de bout en bout (étape E du plan) :
  1. données + split 60/20/20 avec embargo (C1, conventions xgb_einhers)
  2. pool de seuils empiriques sur TRAIN uniquement (anti-lookahead)
  3. recherche MAP-Elites sur fenêtre de VALIDATION (fitness cheap : Sharpe
     net sur échantillon aléatoire contigu, sans bootstrap)
  4. backtests complets VAL + HOLD-OUT des candidats de l'archive
  5. admission C1-C6 (CI bootstrap par blocs, DSR, FDR Benjamini-Hochberg,
     dédup Jaccard/corrélation)
  6. corpus append-only + rapport JSON + résumé stdout

Usage :
  python -m einherjar.research.search_engine.runner \
      --asset BTCUSD --asset-class crypto --timeframe 1h --horizon 2d \
      --seed 42 --n-pop 30 --n-generations 8
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np

from einherjar.research.baselines.runner import parse_horizon
from einherjar.research.search_engine.admission import Candidate, admit_batch
from einherjar.research.search_engine.corpus import (
    append_candidate,
    append_einher,
    condition_from_dict,
    fingerprint_of,
    load_corpus,
)
from einherjar.research.search_engine.evaluator import collect_tree_features, eval_condition_ast
from einherjar.research.search_engine.map_elites import run_map_elites
from einherjar.research.search_engine.space import SpaceConfig
from einherjar.research.search_engine.threshold_pool import ThresholdPool
from einherjar.research.xgb_einhers.backtester import backtest_einher
from einherjar.research.xgb_einhers.data_loader import (
    align_xy_with_ohlcv,
    load_ohlcv,
    load_xy,
    temporal_split,
)
from einherjar.research.xgb_einhers.label_engineer import load_costs

_OUTPUTS = Path("outputs")
_TAXONOMY = "src/einherjar/research/feature_taxonomy_corrected.json"


def _load_taxonomy() -> dict[str, Any]:
    """_load_taxonomy."""
    with open(_TAXONOMY, encoding="utf-8") as fh:
        data = json.load(fh)
    return data.get("features", data)


def run(
    asset: str,
    asset_class: str,
    timeframe: str,
    horizon: str,
    seed: int,
    n_pop: int,
    n_generations: int,
    costs_pct: float | None,
    output_dir: Path = _OUTPUTS,
) -> dict[str, Any]:
    """Exécute une recherche complète et écrit rapport + corpus."""
    # 1. Données + split C1 (purgé/embargoé, conventions xgb_einhers)
    loaded = load_xy(asset, timeframe, asset_class)
    ohlcv = load_ohlcv(asset, timeframe, asset_class)
    X, ohlcv_al, _ts_al = align_xy_with_ohlcv(loaded, ohlcv)
    feature_names = list(loaded.feature_names)
    if horizon not in loaded.horizons:
        raise ValueError(f"Horizon {horizon} absent de {loaded.horizons}")
    horizon_idx = loaded.horizons.index(horizon)
    horizon_bars = parse_horizon(horizon)
    split = temporal_split(
        X, loaded.Y_ret[:, horizon_idx], embargo_bars=50, horizon_bars=horizon_bars,
    )
    val_idx, ho_idx = split.val_indices, split.holdout_indices
    X_val, X_ho = X[val_idx], X[ho_idx]
    ohlcv_val = ohlcv_al.slice(int(val_idx[0]), len(val_idx))
    ohlcv_ho = ohlcv_al.slice(int(ho_idx[0]), len(ho_idx))
    costs = costs_pct if costs_pct is not None else load_costs(asset)
    universe = {"asset": asset, "asset_class": asset_class, "timeframe": timeframe}
    data_version = f"{asset}_{timeframe}_{horizon}"

    # 2. Config + pool de seuils sur TRAIN uniquement (anti-lookahead)
    cfg = SpaceConfig(data_version=data_version, feature_names=tuple(feature_names))
    rng = np.random.default_rng(seed)
    pool = ThresholdPool.build(split.train_X, feature_names, cfg, rng)

    taxonomy = _load_taxonomy()

    # 3. Recherche MAP-Elites sur VAL (fitness cheap)
    archive = run_map_elites(
        rng,
        cfg,
        pool,
        taxonomy,
        {"ohlcv_df": ohlcv_val, "X": X_val, "feature_names": feature_names},
        costs_pct=costs,
        amplitude_bars=horizon_bars,
        universe=universe,
        n_pop=n_pop,
        n_generations=n_generations,
        sample_frac=0.5,
        data_version=data_version,
    )

    # 4. Backtests complets VAL + HOLD-OUT des candidats de l'archive
    candidates: list[Candidate] = []
    cells_order: list[tuple[str, ...]] = []
    ho_metrics: list[Any] = []
    for cell, entry in sorted(archive.cells.items()):
        cells_order.append(cell)
        einher = entry.einher
        val_res = backtest_einher(
            einher, ohlcv_val, X_val, feature_names, costs_pct=costs,
        )
        einher = dataclasses.replace(einher, metrics=val_res.metrics)
        ho_res = backtest_einher(
            einher, ohlcv_ho, X_ho, feature_names, costs_pct=costs,
        )
        ho_metrics.append(ho_res.metrics)
        candidates.append(
            Candidate(
                einher=einher,
                val_mask=eval_condition_ast(einher.condition_tree, X_val, feature_names),
                features=collect_tree_features(einher.condition_tree),
                fingerprint=fingerprint_of(einher.condition_tree),
                holdout_metrics=ho_res.metrics,
            )
        )

    # 5. Admission C1-C6 (batch : FDR sur toutes les p-values)
    #    Dédup GLOBAL : les admis historiques du corpus GP sont aussi
    #    comparés (un même signal trouvé par un autre seed est un doublon).
    corpus_file = output_dir / "corpus_GP.jsonl"
    accepted_history: list[Candidate] = []
    for entry in load_corpus(corpus_file):
        ast = condition_from_dict(entry["condition_tree"])
        accepted_history.append(
            Candidate(
                einher=None,
                val_mask=eval_condition_ast(ast, X_val, feature_names),
                features=collect_tree_features(ast),
                fingerprint=entry.get("fingerprint", ""),
                holdout_metrics=None,
            )
        )
    outcomes = admit_batch(candidates, seed=seed, initial_accepted=accepted_history)

    # 6. Rapport + corpus (corpus_GP = admis GP ; archive_GP = rejetés + raisons)
    output_dir.mkdir(exist_ok=True)
    archive_file = output_dir / "archive_GP.jsonl"
    report = _report(
        asset, timeframe, horizon, cfg, archive,
        candidates, outcomes, costs, n_pop, n_generations,
    )
    out = output_dir / f"search_engine_{asset}_{timeframe}_{horizon}_seed{seed}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"[runner] rapport écrit : {out}")

    for c, o, cell, ho_m in zip(candidates, outcomes, cells_order, ho_metrics):
        if o.admitted:
            append_einher(
                c.einher, o, fingerprint=c.fingerprint,
                holdout_metrics=ho_m, path=corpus_file,
            )
        else:
            append_candidate(
                c.einher, o, fingerprint=c.fingerprint,
                holdout_metrics=ho_m,
                extra={"cell_index": _report_index(report, c), "cell": list(cell)},
                path=archive_file,
            )
    n_admitted = sum(1 for o in outcomes if o.admitted)
    print(
        f"[runner] admis : {n_admitted}/{len(candidates)} → corpus_GP.jsonl ; "
        f"rejetés → archive_GP.jsonl ({archive_file})"
    )
    return report


def _report_index(report: dict[str, Any], candidate: Candidate) -> int:
    """Index du candidat dans le rapport (jointure par fingerprint)."""
    for item in report["admission"]["details"]:
        if item.get("fingerprint") == candidate.fingerprint:
            return item["cell_index"]
    return -1


def _report(
    asset: str,
    timeframe: str,
    horizon: str,
    cfg: SpaceConfig,
    archive: Any,
    candidates: list[Candidate],
    outcomes: list[Any],
    costs: float,
    n_pop: int,
    n_generations: int,
) -> dict[str, Any]:
    """Assemble le rapport JSON."""
    return {
        "meta": {
            "engine": "search_engine",
            "asset": asset,
            "timeframe": timeframe,
            "horizon": horizon,
            "costs_pct": costs,
            "n_pop": n_pop,
            "n_generations": n_generations,
            "max_depth": cfg.max_depth,
            "max_size": cfg.max_size,
            "n_features": len(cfg.feature_names),
            "space": cfg.to_dict(),
        },
        "archive": {
            "n_cells_occupied": len(archive.cells),
            "cells": [
                {
                    "cell": list(cell),
                    "fitness_cheap": entry.fitness,
                    "sharpe_val": entry.einher.metrics.sharpe_ratio,
                    "direction": entry.direction,
                }
                for cell, entry in sorted(
                    archive.cells.items(), key=lambda kv: -kv[1].fitness,
                )
            ],
        },
        "admission": {
            "n_candidates": len(candidates),
            "n_admitted": sum(1 for o in outcomes if o.admitted),
            "details": [
                {
                    "cell_index": i,
                    "fingerprint": c.fingerprint,
                    "sharpe_val": c.einher.metrics.sharpe_ratio,
                    "n_trades": int(c.einher.metrics.n_trades),
                    "p_value": c.einher.metrics.p_value,
                    "reasons": o.reasons,
                    "admitted": o.admitted,
                }
                for i, (c, o) in enumerate(zip(candidates, outcomes))
            ],
        },
    }


def main(argv: list[str] | None = None) -> None:
    """Main.

    Args:
        argv: TODO document.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", default="BTCUSD")
    parser.add_argument("--asset-class", default="crypto")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--horizon", default="2d")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-pop", type=int, default=30)
    parser.add_argument("--n-generations", type=int, default=8)
    parser.add_argument("--costs-pct", type=float, default=None)
    args = parser.parse_args(argv)

    run(
        args.asset, args.asset_class, args.timeframe, args.horizon,
        args.seed, args.n_pop, args.n_generations, args.costs_pct,
        output_dir=_OUTPUTS,
    )


if __name__ == "__main__":
    main()
