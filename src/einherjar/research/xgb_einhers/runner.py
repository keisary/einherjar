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
from typing import Any, Optional

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
from einherjar.research.xgb_einhers.multi_asset_loader import (
    load_multi_asset,
    load_multi_asset_split,
)
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


def _quick_importances(
    X: np.ndarray,
    feature_names: list[str],
    Y_ret: np.ndarray,
    horizon_idx: int,
    valid_mask: np.ndarray,
    n_estimators: int = 30,
) -> dict[str, float]:
    """FIX BUG-07 : pre-train rapide pour avoir les vraies importances.

    Entraine un petit XGBoost (30 estimateurs) sur les donnees, retourne
    le dict {feature_name: gain}.

    Objectif : remplacer le vecteur uniforme {name: 1.0} qui causait
    une elimination arbitraire en cas d'egalite d'importance.
    """
    from einherjar.research.xgb_einhers.model import (
        GBDTConfig, train_gbdt, feature_importance,
    )
    try:
        target = Y_ret[valid_mask, horizon_idx].astype(np.float32)
        X_valid = X[valid_mask]
        # Split rapide 80/20
        n = X_valid.shape[0]
        split = int(n * 0.8)
        config = GBDTConfig(
            n_estimators=n_estimators,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
        )
        model, backend = train_gbdt(
            X_valid[:split], target[:split],
            X_valid[split:], target[split:],
            config,
        )
        return feature_importance(model, backend, feature_names)
    except Exception as e:
        logger.warning(f"_quick_importances failed: {e}, fallback uniforme")
        return {name: 1.0 for name in feature_names}


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
    scope: str = "asset",
    primary_class: str = None,  # Sprint 3.6 : pour scope=market multi-classes
    corpus_path: Optional[Path] = None,  # Sprint 3.6 : si fourni, ecrit les admis ici
    archive_path: Optional[Path] = None,  # Sprint 3.6 : si fourni, ecrit les rejetes ici
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
        # Sprint 3.4 + 3.6 FIX BUG-03 + scope multi-classes :
        # primary_class est passe en parametre par cmd_run
        _primary_class = primary_class if primary_class else asset_class
        primary_loaded = load_xy(primary_asset, timeframe, _primary_class)
        ohlcv_df = load_ohlcv(primary_asset, timeframe, _primary_class)
        X_aligned_full, ohlcv_aligned, ts_aligned = align_xy_with_ohlcv(primary_loaded, ohlcv_df)
        # Determiner horizon_idx
        # FIX Sprint 3.6 : utiliser _primary_class (la classe du primary)
        # au lieu de asset_class pour cohence
        multi_data = load_multi_asset(assets, _primary_class, timeframe)
        feature_names_full = list(multi_data.feature_names)  # noms avant filtre
        feature_names = feature_names_full
        horizons = multi_data.horizons
        horizon_idx = horizons.index(horizon_str)
        horizon_bars = parse_horizon(horizon_str)
        # FIX BUG-03 : utiliser load_multi_asset_split au lieu de load_multi_asset + temporal_split
        # Cela fait le split par actif PUIS concat (pas de leakage cross-actif)
        multi_split = load_multi_asset_split(
            assets, horizon_idx, _primary_class, timeframe,
            train_ratio=0.6, val_ratio=0.2, holdout_ratio=0.2,
            embargo_bars=embargo_bars,
        )
        # Stocker les donnees du split
        X_global_train = multi_split.train_X
        y_global_train = multi_split.train_y
        X_global_val = multi_split.val_X
        y_global_val = multi_split.val_y
        X_global_holdout = multi_split.holdout_X
        y_global_holdout = multi_split.holdout_y
        # FIX BUG-09 (Sprint 3.6) : X_global doit etre coherent avec Y_ret_global
        # pour que le dedup puisse fonctionner. En multi, on utilise multi_data.X
        # (full) et non pas la concatenation des splits (qui a deja filtre valid).
        X_global = multi_data.X
        Y_dir_global = multi_data.Y_dir
        Y_ret_global = multi_data.Y_ret
        Y_hor_global = multi_data.Y_hor
        # valid_mask sur l'horizon demande (coherent avec X_global = multi_data.X)
        valid_mask = Y_dir_global[:, horizon_idx] != -100
        logger.info("  Multi-actif (FIX BUG-03 + BUG-09): %d actifs, train=%d, val=%d, holdout=%d, full=%d",
                    len(assets), X_global_train.shape[0], X_global_val.shape[0],
                    X_global_holdout.shape[0], X_global.shape[0])
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

    # 2. Pre-compute valid_mask (Sprint 3.6 FIX BUG-08 : dedup en avait besoin)
    logger.info("[2a/10] Pre-compute valid_mask ...")
    if multi:
        # En multi, toutes les bougies chargees par load_multi_asset_split sont valides
        valid_mask = np.ones(X_global.shape[0], dtype=bool)
    else:
        valid_mask = Y_dir_global[:, horizon_idx] != -100
    logger.info("  valid_mask : %d/%d True (%.1f%%)",
                valid_mask.sum(), len(valid_mask),
                100 * valid_mask.sum() / max(1, len(valid_mask)))

    # 2. Drop sparse patterns (Sprint 2.3)
    n_features_before = X_global.shape[1]
    if drop_sparse:
        logger.info("[2/10] Drop sparse patterns (pct_True < 0.5%% ou > 99.5%%) ...")
        X_global, feature_names, dropped = filter_sparse_patterns(
            X_global, feature_names,
        )
        logger.info("  %d features dropped, %d restantes", len(dropped), len(feature_names))

    # 3. Feature dedup (Sprint 2.3 + FIX BUG-07)
    if apply_dedup_flag:
        logger.info("[3/10] Feature dedup (drop |r| > 0.85) ...")
        # FIX BUG-07 : pre-train rapide pour avoir les vraies importances
        # au lieu d'un vecteur uniforme (elimination arbitraire en cas d'egalite)
        if drop_sparse:
            # Si on a deja drop les patterns, on a des features propres
            importances = _quick_importances(
                X_global, feature_names, Y_ret_global, horizon_idx, valid_mask,
            )
        else:
            # Sinon, on utilise quand meme les importances rapides sur X_global
            # (avec potentiellement du bruit, mais c'est mieux que uniforme)
            importances = _quick_importances(
                X_global, feature_names, Y_ret_global, horizon_idx, valid_mask,
            )
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

    # 4. Target (valid_mask deja calcule plus haut - Sprint 3.6 FIX BUG-08)
    logger.info("[4/10] Construction du target ...")
    target = Y_ret_global[:, horizon_idx].copy()
    y_hor = Y_hor_global[:, horizon_idx].copy()
    logger.info("  %d/%d samples valides (%.1f%%)",
                valid_mask.sum(), len(valid_mask), 100 * valid_mask.sum() / len(valid_mask))

    # 5. Filtrer valides (single uniquement, multi deja filtre dans load_multi_asset_split)
    if multi:
        # Multi : X_global_train, X_global_val, X_global_holdout sont deja prets
        X_valid = X_global  # juste pour la compat (pas utilise en multi)
        y_valid = target[valid_mask].astype(np.float32) if not multi else y_global_train  # pas utilise
        split_train_X = X_global_train
        split_val_X = X_global_val
        split_holdout_X = X_global_holdout
        split_train_y = y_global_train
        split_val_y = y_global_val
        split_holdout_y = y_global_holdout
        logger.info("  Multi-actif : splits deja calcules par load_multi_asset_split")
    else:
        X_valid = X_global[valid_mask]
        y_valid = target[valid_mask].astype(np.float32)
        logger.info("  X_valid : %d lignes", X_valid.shape[0])

        # 6. Split temporel (single uniquement)
        logger.info("[6/10] Split temporel 60/20/20 ...")
        split = temporal_split(X_valid, y_valid, embargo_bars=embargo_bars, horizon_bars=horizon_bars)
        split_train_X = split.train_X
        split_val_X = split.val_X
        split_holdout_X = split.holdout_X
        split_train_y = split.train_y
        split_val_y = split.val_y
        split_holdout_y = split.holdout_y
        logger.info("  train=%d, val=%d, holdout=%d",
                    len(split.train_X), len(split.val_X), len(split.holdout_X))

    # 7. Entrainer GBDT
    logger.info("[7/10] Entrainement GBDT ...")
    if regularized:
        config = GBDTConfig.regularized()
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
        split_train_X, split_train_y, split_val_X, split_val_y, config,
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
        # Sprint 2.5.1 + 3.3 : FIX bug "val=full" + embargo en backtest
        n_aligned = X_aligned.shape[0]
        if n_aligned > 0:
            # Sprint 3.3 FIX BUG-05 : embargo proportionnel sur le backtest aussi
            # (meme logique que temporal_split)
            backtest_embargo = max(50, horizon_bars)
            val_start = int(n_aligned * 0.6) + backtest_embargo
            val_end = int(n_aligned * 0.8)
            holdout_start = int(n_aligned * 0.8) + backtest_embargo
            # Sprint 3.3 FIX BUG-04 : slicing val+holdout applique meme en multi
            if val_start < val_end:
                val_result = backtest_einher(
                    einher=einher,
                    ohlcv_df=ohlcv_aligned[val_start:val_end],
                    X=X_aligned[val_start:val_end],
                    feature_names=feature_names,
                    costs_pct=costs,
                )
                result = val_result
            else:
                # Pas assez de bougies pour le val avec embargo
                result = backtest_einher(
                    einher=einher,
                    ohlcv_df=ohlcv_aligned[:0],  # 0 bougies
                    X=X_aligned[:0],
                    feature_names=feature_names,
                    costs_pct=costs,
                )
            # Holdout backtest
            if admission_cfg.min_holdout_trades > 0 and holdout_start < n_aligned:
                holdout_result = backtest_einher(
                    einher=einher,
                    ohlcv_df=ohlcv_aligned[holdout_start:],
                    X=X_aligned[holdout_start:],
                    feature_names=feature_names,
                    costs_pct=costs,
                )
                einher = set_einher_holdout_metrics(einher, holdout_result.metrics)
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
    # Sprint 3.6 : on peut optionnellement ecrire dans corpus (admis) et
    # archive (rejetes avec raison) au lieu du --output legacy.
    logger.info("[9c/10] Admission finale ...")
    n_admitted = 0
    n_rejected = 0
    einhers_admitted = []
    # Stores optionnels (Sprint 3.6)
    corpus_store = None
    archive_store = None
    if corpus_path is not None:
        from einherjar.research.xgb_einhers.corpus import CorpusStore
        corpus_store = CorpusStore(corpus_path)
    if archive_path is not None:
        from einherjar.research.xgb_einhers.archive import ArchiveStore
        archive_store = ArchiveStore(archive_path)
    for einher, bh in zip(all_einhers, bh_rejected_list):
        passed, reason = check_admission(einher, admission_cfg, bh_rejected=bh)
        if passed:
            n_admitted += 1
            einhers_admitted.append(einher)
            if corpus_store is not None:
                corpus_store.add(einher)
            else:
                # Legacy : ecrit dans --output
                save_einher(einher, output_path)
        else:
            n_rejected += 1
            if archive_store is not None:
                # universe est un dict (types.Einher.universe: dict[str, Any])
                u = einher.universe
                archive_store.add(
                    einher,
                    rejection_reason=reason,
                    scope="market" if multi else "asset",
                    asset=u.get("asset", ""),
                    asset_class=u.get("asset_class", ""),
                    timeframe=u.get("timeframe", ""),
                    horizon=u.get("horizon", ""),
                )

    # 10. Resume
    # FIX BUG-10 (Sprint 3.6) : en multi, on utilise les splits du multi_split
    # pas `split` (qui n'existe qu'en single).
    if multi:
        _n_train = len(split_train_X)
        _n_val = len(split_val_X)
        _n_holdout = len(split_holdout_X)
    else:
        _n_train = len(split.train_X)
        _n_val = len(split.val_X)
        _n_holdout = len(split.holdout_X)
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
        "n_train": _n_train,
        "n_val": _n_val,
        "n_holdout": _n_holdout,
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
    asset_classes_list: list[str] = []
    if args.assets:
        assets = [a.strip() for a in args.assets.split(",") if a.strip()]
    elif args.scope == "market" or args.scope == "general":
        # Sprint 3.2 P2 : resolution automatique des actifs selon le scope
        from einherjar.research.xgb_einhers.multi_asset_loader import list_available_assets
        asset_classes_list = [c.strip() for c in args.asset_classes.split(",") if c.strip()]
        all_assets = []
        for cls in asset_classes_list:
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
    # Sprint 3.6 : --corpus et --archive optionnels (sinon legacy --output)
    corpus_path = Path(args.corpus) if args.corpus else None
    archive_path = Path(args.archive) if args.archive else None
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
        primary_class=asset_classes_list[0] if asset_classes_list else args.asset_class,
        corpus_path=corpus_path,
        archive_path=archive_path,
    )
    # Resume JSON
    asset_tag = "_".join(assets) if len(assets) <= 3 else f"multi_{len(assets)}"
    summary_path = output.parent / f"diagnostics_{asset_tag}_{args.timeframe}_{args.horizon}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


# ---------------------------------------------------------------------------- #
# Sprint 3.6 : cmd_discover
# Lance TOUS les modeles (asset/class/global) sur tous les triplets
# (tf x horizon) en parallele. Output : corpus + archive.
# ---------------------------------------------------------------------------- #

# Inventaire complet des actifs (Sprint 3.6 - aligne sur technical_agent_dataset_brut)
ALL_ASSET_CLASSES = [
    "crypto", "forex", "commodities", "indices",
    "stocks_growth", "stocks_tech", "stocks_value",
]
DEFAULT_TIMEFRAMES = ["1h"]
DEFAULT_HORIZONS = ["6h", "12h", "1d", "2d"]


def _discover_one_triplet(
    triplet: dict,
) -> dict:
    """Execute un seul triplet (asset, asset_class, scope, tf, horizon).

    Utilise par ProcessPoolExecutor. Doit etre picklable.
    """
    from einherjar.research.xgb_einhers.corpus import CorpusStore
    from einherjar.research.xgb_einhers.archive import ArchiveStore

    asset = triplet["asset"]
    asset_class = triplet["asset_class"]
    scope = triplet["scope"]
    tf = triplet["timeframe"]
    horizon = triplet["horizon"]
    corpus_path = triplet["corpus_path"]
    archive_path = triplet["archive_path"]
    debug = triplet.get("debug", False)
    n_estimators = triplet.get("n_estimators", 100)
    max_depth = triplet.get("max_depth", 3)
    max_paths = triplet.get("max_paths", 30)
    min_score = triplet.get("min_score", 0.0005)
    min_holdout_trades = triplet.get("min_holdout_trades", 5)
    max_assets = triplet.get("max_assets", 5)
    multi_assets = triplet.get("multi_assets", None)

    try:
        if multi_assets:
            assets = multi_assets
        elif scope == "asset":
            assets = [asset]
        else:
            from einherjar.research.xgb_einhers.multi_asset_loader import list_available_assets
            cls_assets = list_available_assets(
                asset_class=asset_class, timeframe=tf, require_ohlcv=True,
            )
            assets = sorted(cls_assets)[:max_assets]
            if not assets:
                return {"status": "skipped", "reason": f"no assets for {asset_class} tf={tf}"}

        summary = run_pipeline(
            assets=assets,
            timeframe=tf,
            horizon_str=horizon,
            asset_class=asset_class,
            output_path=Path("outputs/_unused.jsonl"),
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_score=min_score,
            max_paths=max_paths,
            debug=debug,
            regularized=True,
            apply_dedup_flag=True,
            drop_sparse=True,
            min_holdout_trades=min_holdout_trades,
            scope=scope,
            primary_class=asset_class,
            corpus_path=Path(corpus_path),
            archive_path=Path(archive_path),
        )
        return {"status": "ok", "summary": summary}
    except Exception as e:
        import traceback
        return {
            "status": "error",
            "asset": asset,
            "asset_class": asset_class,
            "scope": scope,
            "timeframe": tf,
            "horizon": horizon,
            "error": str(e),
            "traceback": traceback.format_exc(limit=5),
        }


def build_discovery_triplets(
    asset_classes: list[str] | None = None,
    timeframes: list[str] | None = None,
    horizons: list[str] | None = None,
    include_global: bool = True,
    include_per_asset: bool = True,
    include_per_class: bool = True,
    max_assets_per_class: int = 5,
) -> list[dict]:
    """Construit la liste de tous les triplets a executer.

    Par defaut : 3 scopes (asset, class, global) x N triplets.
    - asset : 1 modele par actif
    - class : 1 modele par classe (tous actifs de la classe)
    - global : 1 modele toutes classes (tous actifs)
    """
    from einherjar.research.xgb_einhers.multi_asset_loader import list_available_assets

    asset_classes = asset_classes or ALL_ASSET_CLASSES
    timeframes = timeframes or DEFAULT_TIMEFRAMES
    horizons = horizons or DEFAULT_HORIZONS

    # FIX BUG-13 (Sprint 3.6) : filtrer les horizons dont le nom finit par 'd'
    # ou 'h' valide. Defensive : si PowerShell a mange le 'd' de "1d" (le
    # transformant en chemin relatif "1"), on le detecte et on alerte.
    valid_horizons = []
    for h in horizons:
        if not h:
            continue
        if h.endswith("h") or h.endswith("d") or h.endswith("m"):
            try:
                # parse_horizon valide le format
                from einherjar.research.xgb_einhers.runner import parse_horizon
                parse_horizon(h)
                valid_horizons.append(h)
                continue
            except (ValueError, ImportError):
                pass
        logger.warning("Horizon '%s' ignore (format invalide ou corrompu par shell)", h)
    horizons = valid_horizons
    if not horizons:
        raise ValueError("Aucun horizon valide. Verifier le quoting (les horizons type '1d' doivent etre quotes en shell)")

    triplets: list[dict] = []

    # 1. Per-asset : 1 modele par (asset, tf, horizon)
    if include_per_asset:
        for cls in asset_classes:
            # On prend juste le 1er TF pour la liste (sera etendu par timeframes)
            tf0 = timeframes[0]
            try:
                # FIX Sprint 3.6 : appliquer max_assets_per_class aussi au per-asset
                # sinon 16 assets x 4 horizons = 64 triplets par classe, trop.
                assets = list_available_assets(
                    asset_class=cls, timeframe=tf0, require_ohlcv=True,
                )[:max_assets_per_class]
            except Exception:
                continue
            for asset in assets:
                for tf in timeframes:
                    for h in horizons:
                        triplets.append({
                            "asset": asset,
                            "asset_class": cls,
                            "scope": "asset",
                            "timeframe": tf,
                            "horizon": h,
                            "multi_assets": None,
                        })

    # 2. Per-class : 1 modele par (class, tf, horizon) - tous actifs de la classe
    if include_per_class:
        for cls in asset_classes:
            for tf in timeframes:
                for h in horizons:
                    try:
                        assets = list_available_assets(
                            asset_class=cls, timeframe=tf, require_ohlcv=True,
                        )[:max_assets_per_class]
                    except Exception:
                        continue
                    if not assets:
                        continue
                    triplets.append({
                        "asset": assets[0],  # primary pour le backtest
                        "asset_class": cls,
                        "scope": "market",
                        "timeframe": tf,
                        "horizon": h,
                        "multi_assets": assets,
                    })

    # 3. Global : 1 modele (toutes classes, tf, horizon)
    if include_global:
        for tf in timeframes:
            for h in horizons:
                all_assets: list[str] = []
                for cls in asset_classes:
                    try:
                        cls_assets = list_available_assets(
                            asset_class=cls, timeframe=tf, require_ohlcv=True,
                        )
                    except Exception:
                        continue
                    all_assets.extend(cls_assets)
                if not all_assets:
                    continue
                all_assets = sorted(set(all_assets))[:max_assets_per_class * 2]
                triplets.append({
                    "asset": all_assets[0],
                    "asset_class": "global",
                    "scope": "general",
                    "timeframe": tf,
                    "horizon": h,
                    "multi_assets": all_assets,
                })

    return triplets


def cmd_discover(args: argparse.Namespace) -> int:
    """Sprint 3.6 : discovery complet - tous modeles x tous triplets, en parallele.

    Strategie :
    - Construit la liste des triplets (asset/class/global) x tf x horizon
    - Lance N workers en parallele (ProcessPoolExecutor)
    - Chaque worker append dans corpus.jsonl (admis) et archive.jsonl (rejetes)
    - Affiche un rapport agrege a la fin
    """
    import time
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from einherjar.research.xgb_einhers.corpus import CorpusStore
    from einherjar.research.xgb_einhers.archive import ArchiveStore

    # 1. Construire les triplets
    asset_classes = [c.strip() for c in args.asset_classes.split(",") if c.strip()] \
        if args.asset_classes else ALL_ASSET_CLASSES
    timeframes = [t.strip() for t in args.timeframes.split(",") if t.strip()] \
        if args.timeframes else DEFAULT_TIMEFRAMES
    horizons = [h.strip() for h in args.horizons.split(",") if h.strip()] \
        if args.horizons else DEFAULT_HORIZONS

    logger.info("=" * 70)
    logger.info("DISCOVERY Sprint 3.6")
    logger.info("  Classes : %s", asset_classes)
    logger.info("  TF      : %s", timeframes)
    logger.info("  Horizons: %s", horizons)
    logger.info("  Per-asset  : %s", args.per_asset)
    logger.info("  Per-class  : %s", args.per_class)
    logger.info("  Global     : %s", args.global_scope)
    logger.info("  Workers    : %d", args.workers)
    logger.info("  Corpus     : %s", args.corpus)
    logger.info("  Archive    : %s", args.archive)
    logger.info("=" * 70)

    triplets = build_discovery_triplets(
        asset_classes=asset_classes,
        timeframes=timeframes,
        horizons=horizons,
        include_global=args.global_scope,
        include_per_asset=args.per_asset,
        include_per_class=args.per_class,
        max_assets_per_class=args.max_assets,
    )
    logger.info("Total triplets : %d", len(triplets))
    if args.limit and args.limit > 0:
        triplets = triplets[:args.limit]
        logger.info("Limite a %d triplets", len(triplets))

    # 2. Parametres communs injectes dans chaque triplet
    common = {
        "corpus_path": str(Path(args.corpus).resolve()),
        "archive_path": str(Path(args.archive).resolve()),
        "debug": args.debug,
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "max_paths": args.max_paths,
        "min_score": args.min_score,
        "min_holdout_trades": args.min_holdout_trades,
        "max_assets": args.max_assets,
    }
    jobs = [{**t, **common} for t in triplets]

    # 3. Lancer en parallele
    t0 = time.time()
    n_ok = 0
    n_err = 0
    n_skipped = 0
    n_admitted_total = 0
    n_rejected_total = 0
    errors: list[dict] = []

    if args.workers <= 1:
        # Mode sequentiel (debug)
        for job in jobs:
            res = _discover_one_triplet(job)
            if res["status"] == "ok":
                n_ok += 1
                s = res["summary"]
                n_admitted_total += s.get("n_admitted", 0)
                n_rejected_total += s.get("n_rejected", 0)
                logger.info("  OK  %s/%s/%s scope=%s : %d admis, %d rejetes",
                            job.get("asset_class"), job["timeframe"], job["horizon"],
                            job["scope"], s.get("n_admitted", 0), s.get("n_rejected", 0))
            elif res["status"] == "skipped":
                n_skipped += 1
            else:
                n_err += 1
                errors.append(res)
                logger.error("  ERR %s/%s/%s scope=%s : %s",
                             job.get("asset_class"), job["timeframe"], job["horizon"],
                             job["scope"], res.get("error", "?")[:200])
    else:
        # Mode parallele
        ctx_args = {} if sys.platform == "win32" else {"ctx": None}
        with ProcessPoolExecutor(max_workers=args.workers, **ctx_args) as ex:
            futures = {ex.submit(_discover_one_triplet, j): j for j in jobs}
            for fut in as_completed(futures):
                job = futures[fut]
                try:
                    res = fut.result()
                except Exception as e:
                    res = {"status": "error", "error": str(e)}
                if res["status"] == "ok":
                    n_ok += 1
                    s = res["summary"]
                    n_admitted_total += s.get("n_admitted", 0)
                    n_rejected_total += s.get("n_rejected", 0)
                    logger.info("  OK  %s/%s/%s scope=%s : %d admis, %d rejetes",
                                job.get("asset_class"), job["timeframe"], job["horizon"],
                                job["scope"], s.get("n_admitted", 0), s.get("n_rejected", 0))
                elif res["status"] == "skipped":
                    n_skipped += 1
                else:
                    n_err += 1
                    errors.append(res)
                    logger.error("  ERR %s/%s/%s scope=%s : %s",
                                 job.get("asset_class"), job["timeframe"], job["horizon"],
                                 job["scope"], res.get("error", "?")[:200])

    elapsed = time.time() - t0

    # 4. Rapport final
    try:
        corpus_n = CorpusStore(args.corpus).count()
    except Exception:
        corpus_n = 0
    try:
        archive_n = ArchiveStore(args.archive).count()
    except Exception:
        archive_n = 0

    rapport = {
        "triplets_total": len(jobs),
        "triplets_ok": n_ok,
        "triplets_err": n_err,
        "triplets_skipped": n_skipped,
        "einhers_admitted_total": n_admitted_total,
        "einhers_rejected_total": n_rejected_total,
        "corpus_count": corpus_n,
        "archive_count": archive_n,
        "elapsed_seconds": elapsed,
        "errors": errors[:10],  # top 10 erreurs
    }
    out = Path("outputs/discover_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rapport, f, indent=2, ensure_ascii=False)
    logger.info("=" * 70)
    logger.info("RAPPORT DISCOVER :")
    logger.info("  Triplets OK : %d / %d", n_ok, len(jobs))
    logger.info("  Triplets ERR: %d", n_err)
    logger.info("  Trielets SKIP: %d", n_skipped)
    logger.info("  Einhers admis total : %d", n_admitted_total)
    logger.info("  Einhers rejetes total : %d", n_rejected_total)
    logger.info("  Corpus final : %d Einhers (dans %s)", corpus_n, args.corpus)
    logger.info("  Archive     : %d Einhers (dans %s)", archive_n, args.archive)
    logger.info("  Temps       : %.1fs (%.1fmin)", elapsed, elapsed / 60)
    logger.info("  Rapport     : %s", out)
    logger.info("=" * 70)
    return 0 if n_err == 0 else 1


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
    p_run.add_argument("--min-score", type=float, default=0.0005,
                       help="Sprint 3.5 : default 0.0005 (au lieu de 0.003) pour eviter d'eliminer trop de chemins")
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
    # Sprint 3.6 : corpus + archive (optionnels, remplacent --output si fournis)
    p_run.add_argument("--corpus", type=str, default=None,
                       help="Sprint 3.6 : chemin du corpus (Einhers admis). Si fourni, "
                            "les admis vont ici au lieu de --output.")
    p_run.add_argument("--archive", type=str, default=None,
                       help="Sprint 3.6 : chemin de l'archive (Einhers rejetes avec raison).")
    p_run.set_defaults(func=cmd_run)

    # Sprint 3.6 : cmd_discover - discovery complet en parallele
    p_disc = sub.add_parser(
        "discover",
        help="Sprint 3.6 : lance TOUS les modeles (asset/class/global) x tf x horizon en parallele",
    )
    p_disc.add_argument("--asset-classes", type=str, default=None,
                        help=f"Classes a scanner (defaut: {','.join(ALL_ASSET_CLASSES)})")
    p_disc.add_argument("--timeframes", type=str, default=None,
                        help=f"Timeframes (defaut: {','.join(DEFAULT_TIMEFRAMES)})")
    p_disc.add_argument("--horizons", type=str, default=None,
                        help=f"Horizons (defaut: {','.join(DEFAULT_HORIZONS)})")
    p_disc.add_argument("--per-asset", action="store_true", default=True,
                        help="Inclure les modeles per-asset (defaut: True)")
    p_disc.add_argument("--no-per-asset", dest="per_asset", action="store_false")
    p_disc.add_argument("--per-class", action="store_true", default=True,
                        help="Inclure les modeles per-class (defaut: True)")
    p_disc.add_argument("--no-per-class", dest="per_class", action="store_false")
    p_disc.add_argument("--global-scope", action="store_true", default=True,
                        help="Inclure le modele global toutes classes (defaut: True)")
    p_disc.add_argument("--no-global", dest="global_scope", action="store_false")
    p_disc.add_argument("--max-assets", type=int, default=5,
                        help="Limite d'actifs par classe (pour scaler le per-class et le global)")
    p_disc.add_argument("--workers", type=int, default=4,
                        help="Nombre de workers en parallele (1=sequentiel)")
    p_disc.add_argument("--limit", type=int, default=0,
                        help="Limite le nombre de triplets (0=tous). Utile pour test rapide.")
    p_disc.add_argument("--debug", action="store_true", help="Seuils d'admission souples (debug)")
    p_disc.add_argument("--n-estimators", type=int, default=100)
    p_disc.add_argument("--max-depth", type=int, default=3)
    p_disc.add_argument("--max-paths", type=int, default=30)
    p_disc.add_argument("--min-score", type=float, default=0.0005)
    p_disc.add_argument("--min-holdout-trades", type=int, default=5)
    p_disc.add_argument("--corpus", type=str, default="outputs/corpus.jsonl",
                        help="Chemin du corpus (Einhers admis)")
    p_disc.add_argument("--archive", type=str, default="outputs/archive.jsonl",
                        help="Chemin de l'archive (Einhers rejetes avec raison)")
    p_disc.set_defaults(func=cmd_discover)

    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    sys.exit(main())
