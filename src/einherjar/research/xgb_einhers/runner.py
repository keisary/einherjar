"""runner.py - CLI pour le pipeline xgb_einhers.

Sprint 2.3 : ajout des options anti-overfit.

Usage single-asset (legacy) :
    python -m einherjar.research.xgb_runner run \
        --asset BTCUSD --timeframe 1h --horizon 6h

Usage multi-actif (Sprint 2.3) :
    python -m einherjar.research.xgb_runner run \
        --assets BTCUSD,ETHUSD,LTCUSD --timeframe 1h --horizon 2d \
        --regularized --apply-dedup --drop-sparse

Options anti-overfit (Sprint 2.3) :
    --regularized      : GBDTConfig.regularized() (min_child_weight=50, etc.)
    --apply-dedup      : feature_dedup avant entrainement (drop |r|>0.85)
    --drop-sparse      : feature_filter avant entrainement (drop patterns < 0.5% True)
    --assets           : liste separee par virgules (overrides --asset)

Etapes :
    1. Charger X, Y (single ou multi-actif)
    2. Drop sparse patterns si demande
    3. Drop features correlees si demande
    4. Split temporel
    5. Entrainer XGBoost
    6. Extraire les chemins
    7. Pour chaque chemin : construire un Einher
    8. Backtest chaque Einher
    9. Appliquer l'admission
    10. Sauvegarder dans JSONL
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

from einherjar.research.xgb_einhers.admission import AdmissionConfig, check_admission
from einherjar.research.xgb_einhers.backtester import backtest_einher
from einherjar.research.xgb_einhers.condition_tree import path_to_ast
from einherjar.research.xgb_einhers.data_loader import (
    align_xy_with_ohlcv,
    load_ohlcv,
    load_xy,
    temporal_split,
)
from einherjar.research.xgb_einhers.einher_builder import (
    build_einher_from_path,
    set_einher_holdout_metrics,
    set_einher_metrics,
    set_einher_tp_sl,
)
from einherjar.research.xgb_einhers.einher_io import save_einher
from einherjar.research.xgb_einhers.feature_dedup import apply_dedup
from einherjar.research.xgb_einhers.feature_filter import filter_sparse_patterns
from einherjar.research.xgb_einhers.label_engineer import build_target, load_costs
from einherjar.research.xgb_einhers.model import (
    GBDTConfig,
    has_xgboost,
    predict_gbdt,
    train_gbdt,
    feature_importance,
)
from einherjar.research.xgb_einhers.multi_asset_loader import load_multi_asset
from einherjar.research.xgb_einhers.multiple_testing import apply_bh_to_einhers
from einherjar.research.xgb_einhers.path_extractor import extract_paths
from einherjar.research.xgb_einhers.types import Einher

logger = logging.getLogger(__name__)


def parse_horizon(horizon_str: str) -> int:
    """Convertit '6h', '12h', '1d' en nombre de bars (pour TF 1h, ratio 1:1)."""
    if horizon_str.endswith("h"):
        return int(horizon_str[:-1])
    if horizon_str.endswith("d"):
        return int(horizon_str[:-1]) * 24
    if horizon_str.endswith("m"):
        return max(1, int(horizon_str[:-1]) // 60)
    raise ValueError(f"Format d'horizon non supporté : {horizon_str}")


def run_pipeline(
    assets: list[str],
    timeframe: str,
    horizon_str: str,
    asset_class: str = "crypto",
    output_path: Path = Path("outputs/einhers.jsonl"),
    n_estimators: int = 100,
    max_depth: int = 4,
    min_score: float = 0.005,
    max_paths: int = 100,
    embargo_bars: int = 50,
    debug: bool = False,
    regularized: bool = False,
    apply_dedup_flag: bool = False,
    drop_sparse: bool = False,
    min_holdout_trades: int = 0,
    bagging_seeds: int = 1,
    walk_forward_folds: int = 1,
    scope: str = "asset",  # Sprint 3.2 P2 : asset | market | general
) -> dict[str, Any]:
    """Pipeline complet XGBoost -> Einher (single ou multi-actif).

    Args:
        assets : liste d'actifs (1 pour single, N pour multi)
        regularized : utiliser GBDTConfig.regularized() (Sprint 2.3)
        apply_dedup_flag : appliquer feature_dedup avant entrainement
        drop_sparse : drop les patterns sparses avant entrainement

    Returns:
        dict avec stats et chemins d'outputs.
    """
    primary_asset = assets[0]
    multi = len(assets) > 1
    logger.info("=" * 70)
    logger.info("PIPELINE xgb_einhers : %s x %s x horizon=%s (multi=%d)",
                "+".join(assets) if multi else primary_asset,
                timeframe, horizon_str, multi)
    logger.info("  regularized=%s, dedup=%s, drop_sparse=%s",
                regularized, apply_dedup_flag, drop_sparse)
    logger.info("=" * 70)

    # 1. Charger X, Y (single ou multi)
    logger.info("[1/10] Chargement X, Y ...")
    if multi:
        # Strategie : on charge d'abord le primary seul, on aligne avec OHLCV,
        # puis on charge le multi pour l'entrainement. Les memes filtres
        # (drop sparse + dedup) seront appliques aux deux.
        primary_loaded = load_xy(primary_asset, timeframe, asset_class)
        ohlcv_df = load_ohlcv(primary_asset, timeframe, asset_class)
        X_aligned_full, ohlcv_aligned, ts_aligned = align_xy_with_ohlcv(primary_loaded, ohlcv_df)
        # Charger le multi
        multi_data = load_multi_asset(assets, asset_class, timeframe)
        feature_names_full = list(multi_data.feature_names)  # noms avant filtre
        feature_names = feature_names_full
        horizons = multi_data.horizons
        horizon_idx = horizons.index(horizon_str)
        horizon_bars = parse_horizon(horizon_str)
        X_global = multi_data.X
        Y_dir_global = multi_data.Y_dir
        Y_ret_global = multi_data.Y_ret
        Y_hor_global = multi_data.Y_hor
        logger.info("  Multi-actif: %d actifs, %d samples global, primary aligned %d samples",
                    len(assets), X_global.shape[0], X_aligned_full.shape[0])
    else:
        loaded = load_xy(primary_asset, timeframe, asset_class)
        feature_names_full = list(loaded.feature_names)
        feature_names = feature_names_full
        horizons = loaded.horizons
        horizon_idx = horizons.index(horizon_str)
        horizon_bars = parse_horizon(horizon_str)
        X_global = loaded.X
        Y_dir_global = loaded.Y_dir
        Y_ret_global = loaded.Y_ret
        Y_hor_global = loaded.Y_hor
        ohlcv_df = load_ohlcv(primary_asset, timeframe, asset_class)
        X_aligned_full, ohlcv_aligned, ts_aligned = align_xy_with_ohlcv(loaded, ohlcv_df)
        logger.info("  N=%d, F=%d, H=%d, horizons=%s",
                    primary_loaded.n_samples if multi else loaded.n_samples,
                    primary_loaded.n_features if multi else loaded.n_features,
                    len(horizons), horizons)

    # 2. Drop sparse patterns (Sprint 2.3)
    n_features_before = X_global.shape[1]
    if drop_sparse:
        logger.info("[2/10] Drop sparse patterns (pct_True < 0.5%% ou > 99.5%%) ...")
        X_global, feature_names, dropped = filter_sparse_patterns(
            X_global, feature_names,
        )
        logger.info("  %d features dropped, %d restantes", len(dropped), len(feature_names))

    # 3. Feature dedup (Sprint 2.3)
    if apply_dedup_flag:
        logger.info("[3/10] Feature dedup (drop |r| > 0.85) ...")
        importances = {name: 1.0 for name in feature_names}
        X_global, feature_names, dropped_dedup = apply_dedup(
            X_global, feature_names, importances, corr_threshold=0.85,
        )
        logger.info("  %d features dedup-dropped, %d restantes", len(dropped_dedup), len(feature_names))

    # IMPORTANT : appliquer TOUS les filtres d'un coup a X_aligned_full
    # (sinon les indices ne correspondent plus apres 2 sous-selections)
    if drop_sparse or apply_dedup_flag:
        keep_set = set(feature_names)
        keep_idx = [i for i, n in enumerate(feature_names_full) if n in keep_set]
        X_aligned_full = X_aligned_full[:, keep_idx]
        logger.info("  X_aligned_full filtre : %d/%d colonnes", len(keep_idx), len(feature_names_full))

    # X_aligned est maintenant X_aligned_full (memes colonnes que X_global)
    X_aligned = X_aligned_full

    # 4. Target + valid mask
    logger.info("[4/10] Construction du target ...")
    # On doit reconstruire le valid_mask en fonction de l'horizon
    # Si multi-actif, on a Y_dir_global sinon loaded.Y_dir
    valid_mask = Y_dir_global[:, horizon_idx] != -100
    target = Y_ret_global[:, horizon_idx].copy()
    y_hor = Y_hor_global[:, horizon_idx].copy()
    logger.info("  %d/%d samples valides (%.1f%%)",
                valid_mask.sum(), len(valid_mask), 100 * valid_mask.sum() / len(valid_mask))

    # 5. Filtrer valides
    X_valid = X_global[valid_mask]
    y_valid = target[valid_mask].astype(np.float32)
    logger.info("  X_valid : %d lignes", X_valid.shape[0])

    # 6. Split temporel
    logger.info("[6/10] Split temporel 60/20/20 ...")
    split = temporal_split(X_valid, y_valid, embargo_bars=embargo_bars, horizon_bars=horizon_bars)
    logger.info("  train=%d, val=%d, holdout=%d",
                len(split.train_X), len(split.val_X), len(split.holdout_X))

    # 7. Entrainer GBDT
    logger.info("[7/10] Entrainement GBDT ...")
    if regularized:
        config = GBDTConfig.regularized()
        # Override n_estimators / max_depth si specifies explicitement
        if n_estimators != 100:
            config = GBDTConfig(**{**config.__dict__, "n_estimators": n_estimators})
        if max_depth != 4:
            config = GBDTConfig(**{**config.__dict__, "max_depth": max_depth})
    else:
        config = GBDTConfig(
            n_estimators=n_estimators,
            max_depth=max_depth,
            backend="auto",
        )
    model, backend = train_gbdt(
        split.train_X, split.train_y, split.val_X, split.val_y, config,
    )
    logger.info("  backend = %s, config = %s", backend, config)

    # 8. Extraire les chemins
    logger.info("[8/10] Extraction des chemins ...")
    paths = extract_paths(
        model, backend, feature_names,
        min_score=min_score, max_paths=max_paths,
    )
    logger.info("  %d chemins retenus", len(paths))

    # 9. Pour chaque chemin : construire un Einher et le backtest
    logger.info("[9/10] Generation des Einhers et backtest ...")
    # Sprint 3.0 FIX #3 : utiliser des couts realistes (taker 0.05% x 2 = 0.10%)
    # L'ancien load_costs(primary_asset) sous-estimait les frais crypto
    costs = max(load_costs(primary_asset), 0.0010)  # minimum 0.10% round-trip
    logger.info("  Cout round-trip : %.4f (Sprint 3.0 : minimum 0.10%%)", costs)
    n_admitted = 0
    n_rejected = 0
    n_generated = 0
    einhers_admitted = []
    admission_cfg = AdmissionConfig.debug() if debug else AdmissionConfig()
    # Sprint 2.4.1 : override du min_holdout_trades
    if min_holdout_trades > 0:
        admission_cfg = AdmissionConfig(
            **{**admission_cfg.__dict__, "min_holdout_trades": min_holdout_trades}
        )

    # Phase 1 : generer + backtester TOUS les Einhers (sans admission)
    logger.info("[9a/10] Generation + backtest de tous les Einhers (avant BH) ...")
    all_einhers: list[Einher] = []
    for path in paths:
        einher = build_einher_from_path(
            path=path,
            asset=primary_asset if not multi else "multi",
            asset_class=asset_class,
            timeframe=timeframe,
            horizon_str=horizon_str,
            horizon_bars=horizon_bars,
            min_abs_score=min_score,  # Sprint 2.3 : respecte --min-score
        )
        if einher is None:
            continue
        n_generated += 1
        # Sprint 2.5.1 : FIX bug "val=full"
        n_aligned = X_aligned.shape[0]
        if not multi and n_aligned > 0:
            val_start = int(n_aligned * 0.6)
            val_end = int(n_aligned * 0.8)
            val_result = backtest_einher(
                einher=einher,
                ohlcv_df=ohlcv_aligned[val_start:val_end],
                X=X_aligned[val_start:val_end],
                feature_names=feature_names,
                costs_pct=costs,
            )
            result = val_result
            # Holdout backtest (80-100%) - pour le filtre min_holdout_trades
            if admission_cfg.min_holdout_trades > 0:
                holdout_start = int(n_aligned * 0.8)
                if holdout_start < n_aligned:
                    holdout_result = backtest_einher(
                        einher=einher,
                        ohlcv_df=ohlcv_aligned[holdout_start:],
                        X=X_aligned[holdout_start:],
                        feature_names=feature_names,
                        costs_pct=costs,
                    )
                    einher = set_einher_holdout_metrics(einher, holdout_result.metrics)
        else:
            result = backtest_einher(
                einher=einher,
                ohlcv_df=ohlcv_aligned,
                X=X_aligned,
                feature_names=feature_names,
                costs_pct=costs,
            )
        einher = set_einher_metrics(einher, result.metrics)
        einher = set_einher_tp_sl(einher, result.effective_tp_pct, result.effective_sl_pct)
        all_einhers.append(einher)

    # Phase 2 : Sprint 3.1 P1 - Benjamini-Hochberg sur TOUS les Einhers
    bh_rejected_list: list[bool] = [True] * len(all_einhers)
    if admission_cfg.apply_bh and len(all_einhers) > 0:
        logger.info("[9b/10] Benjamini-Hochberg sur %d candidats (FDR=%.2f) ...",
                    len(all_einhers), admission_cfg.fdr)
        _, pvalues, bh_rejected = apply_bh_to_einhers(
            all_einhers, fdr=admission_cfg.fdr,
        )
        bh_rejected_list = bh_rejected
        n_bh_rejected = sum(1 for r in bh_rejected if not r)
        logger.info("  BH : %d/%d Einhers rejetes (FDR %.0f%%)",
                    n_bh_rejected, len(all_einhers), admission_cfg.fdr * 100)
    else:
        logger.info("  BH : desactive (tous les Einhers passent par defaut)")

    # Phase 3 : admission finale (avec check BH)
    logger.info("[9c/10] Admission finale ...")
    n_admitted = 0
    n_rejected = 0
    einhers_admitted = []
    for einher, bh in zip(all_einhers, bh_rejected_list):
        passed, reason = check_admission(einher, admission_cfg, bh_rejected=bh)
        if passed:
            n_admitted += 1
            einhers_admitted.append(einher)
            save_einher(einher, output_path)
        else:
            n_rejected += 1

    # 10. Resume
    summary = {
        "assets": assets,
        "multi_asset": multi,
        "timeframe": timeframe,
        "horizon": horizon_str,
        "horizon_bars": horizon_bars,
        "n_features_before": n_features_before,
        "n_features_after": len(feature_names),
        "regularized": regularized,
        "apply_dedup": apply_dedup_flag,
        "drop_sparse": drop_sparse,
        "n_samples_total": X_global.shape[0],
        "n_valid": X_valid.shape[0],
        "n_train": len(split.train_X),
        "n_val": len(split.val_X),
        "n_holdout": len(split.holdout_X),
        "n_paths_extracted": len(paths),
        "n_einhers_generated": n_generated,
        "n_admitted": n_admitted,
        "n_rejected": n_rejected,
        "output_path": str(output_path),
        "costs_pct": costs,
        "gbdt_config": {k: v for k, v in config.__dict__.items() if not k.startswith("_")},
    }
    logger.info("=" * 70)
    logger.info("RESUME : %d generes, %d admis, %d rejetes", n_generated, n_admitted, n_rejected)
    logger.info("Output : %s", output_path)
    logger.info("=" * 70)
    return summary


def cmd_run(args: argparse.Namespace) -> int:
    """Commande `run`."""
    output = Path(args.output)
    # Resolution des actifs : --assets prime sur --asset
    if args.assets:
        assets = [a.strip() for a in args.assets.split(",") if a.strip()]
    elif args.scope == "market" or args.scope == "general":
        # Sprint 3.2 P2 : resolution automatique des actifs selon le scope
        from einherjar.research.xgb_einhers.multi_asset_loader import list_available_assets
        asset_classes = [c.strip() for c in args.asset_classes.split(",") if c.strip()]
        all_assets = []
        for cls in asset_classes:
            # Verifier que les donnees OHLCV sont dispo (pour le backtest)
            cls_assets = list_available_assets(
                asset_class=cls, timeframe=args.timeframe, require_ohlcv=True,
            )
            all_assets.extend(cls_assets)
        # Dedupliquer et limiter
        assets = sorted(set(all_assets))[:args.max_assets]
        print(f"Scope {args.scope} : {len(assets)} actifs selectionnes sur {len(all_assets)} dispos")
    else:
        assets = [args.asset]
    summary = run_pipeline(
        assets=assets,
        timeframe=args.timeframe,
        horizon_str=args.horizon,
        asset_class=args.asset_class,
        output_path=output,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_score=args.min_score,
        max_paths=args.max_paths,
        embargo_bars=args.embargo,
        debug=args.debug,
        regularized=args.regularized,
        apply_dedup_flag=args.apply_dedup,
        drop_sparse=args.drop_sparse,
        min_holdout_trades=args.min_holdout_trades,
        bagging_seeds=args.bagging_seeds,
        walk_forward_folds=args.walk_forward_folds,
        scope=args.scope,
    )
    # Resume JSON
    asset_tag = "_".join(assets) if len(assets) <= 3 else f"multi_{len(assets)}"
    summary_path = output.parent / f"diagnostics_{asset_tag}_{args.timeframe}_{args.horizon}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="xgb_runner",
        description="Pipeline XGBoost -> Einhers",
    )
    sub = parser.add_subparsers(dest="command")

    p_run = sub.add_parser("run", help="Run le pipeline complet")
    p_run.add_argument("--asset", type=str, default="BTCUSD",
                       help="Actif unique (legacy)")
    p_run.add_argument("--assets", type=str, default=None,
                       help="Liste d'actifs separes par virgules (Sprint 2.3, multi-actif)")
    p_run.add_argument("--timeframe", type=str, default="1h")
    p_run.add_argument("--horizon", type=str, default="6h",
                       choices=["6h", "12h", "1d", "2d", "1h", "2h", "4h", "8h"])
    p_run.add_argument("--asset-class", type=str, default="crypto")
    p_run.add_argument("--output", type=Path, default=Path("outputs/einhers.jsonl"))
    p_run.add_argument("--n-estimators", type=int, default=100)
    p_run.add_argument("--max-depth", type=int, default=4)
    p_run.add_argument("--min-score", type=float, default=0.003)
    p_run.add_argument("--max-paths", type=int, default=100)
    p_run.add_argument("--embargo", type=int, default=50)
    p_run.add_argument("--debug", action="store_true", help="Seuils d'admission souples (debug)")
    p_run.add_argument("--backend", type=str, default="auto", choices=["auto", "xgboost", "sklearn"])
    # Sprint 2.3 : options anti-overfit
    p_run.add_argument("--regularized", action="store_true",
                       help="Utiliser GBDTConfig.regularized() (min_child_weight=50, etc.)")
    p_run.add_argument("--apply-dedup", action="store_true",
                       help="Appliquer feature_dedup avant entrainement (drop |r|>0.85)")
    p_run.add_argument("--drop-sparse", action="store_true",
                       help="Drop les patterns sparses (pct_True < 0.5%%)")
    p_run.add_argument("--min-holdout-trades", type=int, default=0,
                       help="Sprint 2.4.1 : min trades sur holdout pour admettre un Einher (0=desactive)")
    p_run.add_argument("--bagging-seeds", type=int, default=1,
                       help="Sprint 2.4.2 : nombre de seeds pour bagging (1=desactive)")
    p_run.add_argument("--walk-forward-folds", type=int, default=1,
                       help="Sprint 2.4.3 : nombre de folds walk-forward (1=desactive)")
    # Sprint 3.2 P2 : 3 niveaux de scope
    p_run.add_argument("--scope", type=str, default="asset",
                       choices=["asset", "market", "general"],
                       help="Sprint 3.2 P2 : asset=1 actif, market=1 classe, general=toutes classes")
    p_run.add_argument("--asset-classes", type=str, default="crypto",
                       help="Sprint 3.2 P2 : classes separees par virgules (ex: crypto,forex,indices)")
    p_run.add_argument("--max-assets", type=int, default=10,
                       help="Sprint 3.2 P2 : limite d'actifs par run (pour scaler)")
    p_run.set_defaults(func=cmd_run)

    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    sys.exit(main())
