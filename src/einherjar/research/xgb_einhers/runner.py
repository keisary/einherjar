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
from datetime import UTC
from pathlib import Path
from typing import Any

import numpy as np

from .admission import AdmissionConfig, check_admission
from .backtester import backtest_einher, backtest_einher_multi
from .data_loader import (
    align_xy_with_ohlcv,
    load_ohlcv,
    load_xy,
    temporal_split,
)
from .einher_builder import (
    build_einher_from_path,
    set_einher_holdout_metrics,
    set_einher_metrics,
    set_einher_tp_sl,
)
from .einher_io import save_einher
from .feature_dedup import apply_dedup
from .feature_filter import filter_sparse_patterns
from .label_engineer import load_costs
from .model import (
    GBDTConfig,
    feature_importance,
    train_gbdt,
    train_gbdt_grid,  # Problème 5 : grid search params
)
from .multi_asset_loader import (
    load_multi_asset_split,
)
from .multiple_testing import apply_bh_to_einhers
from .path_extractor import extract_paths
from .paths import (
    ARCHIVE_PATH,
    CORPUS_PATH,
    DISCOVER_STATE_PATH,
    OUTPUTS_DIR,
    resolve_output,
)
from .types import Einher

logger = logging.getLogger(__name__)


def make_triplet_id(asset: str, asset_class: str, scope: str, timeframe: str, horizon: str) -> str:
    """Identifiant STABLE d'un triplet de discovery (checkpoint/reprise).

    Contrairement a l'id d'Einher (qui contient un uuid aleatoire), cet id ne
    depend QUE des parametres du triplet : meme triplet relance -> meme id.
    """
    import hashlib

    raw = f"{asset}|{asset_class}|{scope}|{timeframe}|{horizon}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _load_done_triplets(state_path=None) -> set[str]:
    """Charge l'ensemble des triplet_id deja termines avec succes (checkpoint).

    Format du journal : une ligne JSON par job termine :
        {"triplet_id": "...", "status": "ok", "finished_at": "..."}
    """
    import json as _json

    p = state_path or DISCOVER_STATE_PATH
    done: set[str] = set()
    if not p.exists():
        return done
    try:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = _json.loads(line)
                    if d.get("status") == "ok" and d.get("triplet_id"):
                        done.add(d["triplet_id"])
                except Exception:
                    continue  # ligne corrompue : on ignore, on ne bloque pas la reprise
    except Exception as e:
        logger.warning("Checkpoint illisible (%s) : reprise sans skip - %s", p, e)
    return done


def _mark_triplet_done(triplet_id: str, status: str, extra: dict | None = None, state_path=None) -> None:
    """Append une entree au journal de progression (flush immediat sur disque)."""
    import json as _json
    from datetime import datetime

    p = state_path or DISCOVER_STATE_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "triplet_id": triplet_id,
        "status": status,
        "finished_at": datetime.now(UTC).isoformat(),
    }
    if extra:
        entry.update(extra)
    try:
        with open(p, "a", encoding="utf-8") as f:
            f.write(_json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning("Impossible d'ecrire le checkpoint : %s", e)


def _cuda_available() -> bool:
    """True si xgboost peut utiliser CUDA (cache par process)."""
    global _CUDA_CACHE
    if _CUDA_CACHE is None:
        try:
            import xgboost as _xgb

            # build_info peut etre un dict OU une fonction selon les versions.
            use_cuda = False
            bi = getattr(_xgb, "build_info", None)
            if isinstance(bi, dict):
                use_cuda = bool(bi.get("USE_CUDA"))
            elif callable(bi):
                try:
                    use_cuda = bool(bi().get("USE_CUDA"))
                except Exception:
                    use_cuda = False
            # xgboost >= 2.0 : device='cuda' dispo -> le probe fait foi.
            _CUDA_CACHE = use_cuda or (_probe_cuda())
        except Exception:
            _CUDA_CACHE = False
    return _CUDA_CACHE


def _probe_cuda() -> bool:
    """Probe leger : un petit fit device=cuda reussit-il ? (une fois par process)."""
    global _CUDA_PROBED
    if _CUDA_PROBED is not None:
        return _CUDA_PROBED
    try:
        import numpy as _np
        import xgboost as _xgb

        _m = _xgb.XGBRegressor(n_estimators=1, max_depth=1, device="cuda", tree_method="hist")
        _m.fit(_np.zeros((8, 2), dtype=np.float32), _np.zeros(8, dtype=np.float32))
        _CUDA_PROBED = True
    except Exception:
        _CUDA_PROBED = False
    return _CUDA_PROBED


_CUDA_CACHE: bool | None = None
_CUDA_PROBED: bool | None = None


# Durées des TF en minutes (pour convertir horizon en bars)
_TF_MINUTES = {"5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}


def _horizon_to_minutes(horizon_str: str) -> int:
    """Convertit un nom d'horizon ('6h', '1d', '30m', '2d') en minutes."""
    h = horizon_str.strip().lower()
    if h.endswith("m"):
        return int(h[:-1])
    if h.endswith("h"):
        return int(h[:-1]) * 60
    if h.endswith("d"):
        return int(h[:-1]) * 1440
    raise ValueError(f"Format d'horizon non supporté : {horizon_str}")


def parse_horizon(horizon_str: str, timeframe: str = "1h") -> int:
    """Convertit un horizon ('6h', '12h', '1d'...) en nombre de BARS du TF.

    Correctif (2026-08-21) : l'ancienne version supposait le ratio 1h = 1:1,
    ce qui donnait un nombre de bars FAUX dès qu'on change de TF (5m/15m/4h/1d).
    Exemple : horizon '6h' sur TF 5m  -> 6*60/5 = 72 bars.
            horizon '1d' sur TF 1h  -> 1440/60 = 24 bars.

    Args:
        horizon_str : nom d'horizon ('6h', '1d', '2h'...)
        timeframe   : '5m' | '15m' | '1h' | '4h' | '1d'

    Returns:
        Nombre de bars (>= 1) pour l'amplitude.
    """
    horizon_min = _horizon_to_minutes(horizon_str)
    tf_min = _TF_MINUTES.get(timeframe)
    if tf_min is None:
        raise ValueError(f"Timeframe non supporté pour parse_horizon : {timeframe}")
    bars = int(round(horizon_min / tf_min))
    return max(1, bars)


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
            X_valid[:split],
            target[:split],
            X_valid[split:],
            target[split:],
            config,
        )
        return feature_importance(model, backend, feature_names)
    except Exception as e:
        logger.warning("_quick_importances failed: %s, fallback uniforme", e)
        return {name: 1.0 for name in feature_names}


def run_pipeline(
    assets: list[str],
    timeframe: str,
    horizon_str: str,
    asset_class: str = "crypto",
    output_path: Path | None = None,  # None => OUTPUTS_DIR/einhers.jsonl (legacy)
    n_estimators: int = 100,
    max_depth: int = 4,
    min_score: float = 0.0,  # <=0 => auto (1e-9) pour ne pas exclure de feuilles
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
    primary_class: str | None = None,  # Sprint 3.6 : pour scope=market multi-classes
    corpus_path: Path | None = None,  # Sprint 3.6 : si fourni, ecrit les admis ici
    archive_path: Path | None = None,  # Sprint 3.6 : si fourni, ecrit les rejetes ici
    runner_scope: str | None = None,  # Sprint 3.7 : scope reel (asset/market/general)
    optimize_params: bool = True,  # Problème 5 : grid search léger TOUJOURS actif
) -> dict[str, Any]:
    """Pipeline complet XGBoost -> Einher (single ou multi-actif).

    Args:
        assets : liste d'actifs (1 pour single, N pour multi).
        timeframe : timeframe OHLCV (ex: '1h', '4h', '1d').
        horizon_str : horizon du target (ex: '6h', '12h', '1d', '2d').
        asset_class : classe d'actifs (ex: 'crypto', 'forex').
        output_path : chemin du fichier JSONL de sortie (legacy, si pas de corpus).
        n_estimators : nombre d'arbres XGBoost.
        max_depth : profondeur max des arbres.
        min_score : score minimum des chemins extraits (<=0 => auto 1e-9).
        max_paths : nombre max de chemins a extraire par arbre.
        embargo_bars : nb de bougies exclues aux frontieres (defaut 50).
        debug : utiliser AdmissionConfig.debug() (seuils souples).
        regularized : utiliser GBDTConfig.regularized() (Sprint 2.3).
        apply_dedup_flag : appliquer feature_dedup avant entrainement.
        drop_sparse : drop les patterns sparses avant entrainement.
        min_holdout_trades : nb min de trades sur holdout pour admettre.
        bagging_seeds : nombre de seeds pour bagging (1=desactive).
        walk_forward_folds : nombre de folds walk-forward (1=desactive).
        scope : scope du run ('asset' | 'market' | 'general').
        primary_class : classe de l'actif primary (pour scope=market multi-classes).
        corpus_path : chemin du corpus (Einhers admis) si fourni.
        archive_path : chemin de l'archive (Einhers rejetes) si fourni.
        runner_scope : scope reel pour propagation dans l'archive.
        optimize_params : grid search leger TOUJOURS actif (Sprint 3.7).

    Returns:
        dict avec stats et chemins d'outputs.
    """
    primary_asset = assets[0]
    multi = len(assets) > 1
    if output_path is None:
        output_path = OUTPUTS_DIR / "einhers.jsonl"
    logger.info("=" * 70)
    logger.info(
        "PIPELINE xgb_einhers : %s x %s x horizon=%s (multi=%d)",
        "+".join(assets) if multi else primary_asset,
        timeframe,
        horizon_str,
        multi,
    )
    logger.info(
        "  regularized=%s, dedup=%s, drop_sparse=%s", regularized, apply_dedup_flag, drop_sparse
    )
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
        # FIX MEM-01 (2026-08-24) : plus de load_multi_asset() ici (il chargeait
        # TOUS les arrays de tous les actifs juste pour lire feature_names et
        # horizons, qui sont identiques entre actifs). On lit metadata.json du
        # primary : RAM peak divisee par ~2 en scope market/general.
        from .data_loader import load_feature_meta

        feature_names_full, horizons = load_feature_meta(
            primary_asset, timeframe, _primary_class
        )
        feature_names = feature_names_full
        horizon_idx = horizons.index(horizon_str)
        horizon_bars = parse_horizon(horizon_str, timeframe)
        # FIX BUG-03 : split par actif PUIS concat (pas de leakage cross-actif)
        multi_split = load_multi_asset_split(
            assets,
            horizon_idx,
            _primary_class,
            timeframe,
            train_ratio=0.6,
            val_ratio=0.2,
            holdout_ratio=0.2,
            embargo_bars=embargo_bars,
        )
        # Stocker les donnees du split
        X_global_train = multi_split.train_X
        y_global_train = multi_split.train_y
        X_global_val = multi_split.val_X
        y_global_val = multi_split.val_y
        X_global_holdout = multi_split.holdout_X
        # FIX BUG-09 : pour le dedup en multi on garde une reference aux splits
        # (train+val+holdout concat), PAS une 2e copie full de tous les actifs.
        # Les filtres (sparse/dedup) sont appliques par nom de feature, donc
        # travailler sur la concat des splits suffit et evite le doublon RAM.
        X_global = np.concatenate(
            [multi_split.train_X, multi_split.val_X, multi_split.holdout_X], axis=0
        )
        Y_dir_global = None  # non utilise en multi (valid_mask deja applique au split)
        Y_ret_global = None
        valid_mask = np.ones(X_global.shape[0], dtype=bool)
        logger.info(
            "  Multi-actif (FIX BUG-03 + MEM-01): %d actifs, train=%d, val=%d, holdout=%d",
            len(assets),
            X_global_train.shape[0],
            X_global_val.shape[0],
            X_global_holdout.shape[0],
        )
    else:
        loaded = load_xy(primary_asset, timeframe, asset_class)
        feature_names_full = list(loaded.feature_names)
        feature_names = feature_names_full
        horizons = loaded.horizons
        horizon_idx = horizons.index(horizon_str)
        horizon_bars = parse_horizon(horizon_str, timeframe)
        X_global = loaded.X
        Y_dir_global = loaded.Y_dir
        Y_ret_global = loaded.Y_ret
        ohlcv_df = load_ohlcv(primary_asset, timeframe, asset_class)
        X_aligned_full, ohlcv_aligned, ts_aligned = align_xy_with_ohlcv(loaded, ohlcv_df)
        logger.info(
            "  N=%d, F=%d, H=%d, horizons=%s",
            primary_loaded.n_samples if multi else loaded.n_samples,
            primary_loaded.n_features if multi else loaded.n_features,
            len(horizons),
            horizons,
        )

    # 2. Pre-compute valid_mask (Sprint 3.6 FIX BUG-08 : dedup en avait besoin)
    logger.info("[2a/10] Pre-compute valid_mask ...")
    if not multi:
        valid_mask = Y_dir_global[:, horizon_idx] != -100
    # (multi : valid_mask deja tout-True, les invalides ont ete filtres par actif)
    logger.info(
        "  valid_mask : %d/%d True (%.1f%%)",
        valid_mask.sum(),
        len(valid_mask),
        100 * valid_mask.sum() / max(1, len(valid_mask)),
    )

    # 2. Drop sparse patterns (Sprint 2.3)
    n_features_before = X_global.shape[1]
    if drop_sparse:
        logger.info("[2/10] Drop sparse patterns (pct_True < 0.5%% ou > 99.5%%) ...")
        X_global, feature_names, dropped = filter_sparse_patterns(
            X_global,
            feature_names,
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
                X_global,
                feature_names,
                Y_ret_global,
                horizon_idx,
                valid_mask,
            )
        else:
            # Sinon, on utilise quand meme les importances rapides sur X_global
            # (avec potentiellement du bruit, mais c'est mieux que uniforme)
            importances = _quick_importances(
                X_global,
                feature_names,
                Y_ret_global,
                horizon_idx,
                valid_mask,
            )
        X_global, feature_names, dropped_dedup = apply_dedup(
            X_global,
            feature_names,
            importances,
            corr_threshold=0.95,  # FIX 2026-08-21 : 0.85 trop agressif
        )
        logger.info(
            "  %d features dedup-dropped, %d restantes", len(dropped_dedup), len(feature_names)
        )

    # IMPORTANT : appliquer TOUS les filtres d'un coup a X_aligned_full
    # (sinon les indices ne correspondent plus apres 2 sous-selections)
    if drop_sparse or apply_dedup_flag:
        keep_set = set(feature_names)
        keep_idx = [i for i, n in enumerate(feature_names_full) if n in keep_set]
        X_aligned_full = X_aligned_full[:, keep_idx]
        # FIX P0-3 (AI Review 2026-08-20) : slicer aussi les splits multi-actif
        # sinon XGBoost entraine sur 213 cols mais feature_names en a 120
        # → AST invalide, 0 trades en backtest
        if multi:
            X_global_train = X_global_train[:, keep_idx]
            X_global_val = X_global_val[:, keep_idx]
            X_global_holdout = X_global_holdout[:, keep_idx]
            logger.info(
                "  Multi-asset splits filtres : %d/%d colonnes",
                len(keep_idx),
                len(feature_names_full),
            )
        logger.info(
            "  X_aligned_full filtre : %d/%d colonnes", len(keep_idx), len(feature_names_full)
        )

    # X_aligned est maintenant X_aligned_full (memes colonnes que X_global)
    X_aligned = X_aligned_full

    # FIX BUG-1 (2026-08-21) : en mode multi (scope=market/general), on prepare
    # les donnees backtest de TOUS les actifs du scope (per_asset), pour que
    # l'evaluation porte sur tout l'univers, pas seulement l'actif primaire.
    multi_per_asset: list = []
    if multi:
        from .data_loader import align_xy_with_ohlcv as _align
        from .data_loader import load_ohlcv as _loh
        from .data_loader import load_xy as _lxy
        _keep_set = set(feature_names)
        _keep_idx = [i for i, n in enumerate(feature_names_full) if n in _keep_set]
        for _a in assets:
            try:
                _d = _lxy(_a, timeframe, _primary_class if multi else asset_class)
                _o = _loh(_a, timeframe, _primary_class if multi else asset_class)
                _Xa, _oa, _tsa = _align(_d, _o)
                _Xa = _Xa[:, _keep_idx] if _keep_idx else _Xa
                multi_per_asset.append((_oa, _Xa))
            except Exception as _e:
                logger.warning("  multi setup : actif %s backtest indispo (%s)", _a, _e)
        logger.info("  Multi backtest sur %d actifs (scope %s)", len(multi_per_asset), scope)


    # 4. Target (valid_mask deja calcule plus haut - Sprint 3.6 FIX BUG-08)
    logger.info("[4/10] Construction du target ...")
    if multi:
        # MEM-01 : Y_ret_global n'est plus charge en full (RAM) ; le target
        # multi est deja present dans les splits (train_y/val_y).
        target = None
        logger.info(
            "  %d/%d samples valides (%.1f%%)",
            valid_mask.sum(),
            len(valid_mask),
            100 * valid_mask.sum() / max(1, len(valid_mask)),
        )
    else:
        target = Y_ret_global[:, horizon_idx].copy()
        logger.info(
            "  %d/%d samples valides (%.1f%%)",
            valid_mask.sum(),
            len(valid_mask),
            100 * valid_mask.sum() / len(valid_mask),
        )

    # 5. Filtrer valides (single uniquement, multi deja filtre dans load_multi_asset_split)
    if multi:
        # Multi : X_global_train, X_global_val, X_global_holdout sont deja prets
        X_valid = X_global  # pour le resume (n_valid)
        y_valid = y_global_train  # pas utilise en multi
        split_train_X = X_global_train
        split_val_X = X_global_val
        split_holdout_X = X_global_holdout
        split_train_y = y_global_train
        split_val_y = y_global_val
        logger.info("  Multi-actif : splits deja calcules par load_multi_asset_split")
    else:
        X_valid = X_global[valid_mask]
        y_valid = target[valid_mask].astype(np.float32)
        logger.info("  X_valid : %d lignes", X_valid.shape[0])

        # 6. Split temporel (single uniquement)
        logger.info("[6/10] Split temporel 60/20/20 ...")
        split = temporal_split(
            X_valid, y_valid, embargo_bars=embargo_bars, horizon_bars=horizon_bars
        )
        split_train_X = split.train_X
        split_val_X = split.val_X
        split_holdout_X = split.holdout_X
        split_train_y = split.train_y
        split_val_y = split.val_y
        logger.info(
            "  train=%d, val=%d, holdout=%d",
            len(split.train_X),
            len(split.val_X),
            len(split.holdout_X),
        )

    # 7. Entrainer GBDT
    logger.info("[7/10] Entrainement GBDT ...")
    # Problème 2 : min_score adaptatif si non fourni (auto).
    # min_score <= 0  =>  seuil quasi-nul pour NE PAS exclure de feuilles.
    # Correction 2026-08-21 : un min_score=0.0005 FIXE éliminait 100% des chemins
    # sur forex/indices/commodities (vol ~10-50x plus faible que la crypto).
    # La limite sur le nombre de candidats est déjà donnée par max_paths, et le
    # tri par |score| + l'admission (backtest) font le filtrage réel.
    effective_min_score = min_score
    if effective_min_score <= 0:
        # Auto : on transmet 0 à extract_paths, qui calcule alors un seuil
        # RELATIF = 33e percentile des |scores| des feuilles (adapté à la
        # volatilité crypto vs forex). Un 1e-9 fixe gardait les feuilles
        # extrêmes/rares -> 0 trades.
        effective_min_score = 0.0
        logger.info("  min_score auto : active (seuil p33 des |scores| dans extract_paths)")

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
    # PERF-GPU (2026-08-24) : detection CUDA une fois par process. Sur les
    # machines avec GPU NVIDIA (GTX 1660 Ti ici), xgboost device=cuda accelere
    # fortement l'entrainement hist ; fallback CPU silencieux sinon.
    if _cuda_available():
        config = GBDTConfig(**{**config.__dict__, "device": "cuda", "tree_method": "hist"})
        logger.info("  GPU CUDA detecte : xgboost device=cuda")
    else:
        logger.info("  Pas de CUDA : entrainement CPU")
    # Problème 5 : grid search léger TOUJOURS actif (validé Jovanny).
    # Essaie max_depth x n_estimators, garde la config min val_rmse
    # (proxy qualité des chemins -> maximise la recherche d'einhers).
    # PERF-03 (2026-08-24) : grid search (8 entrainements) reserve aux trains
    # larges (market/general). En per-asset (~42k lignes), la config regularized
    # suffit et le grid multipliait le temps par ~8 pour un gain marginal.
    _grid_worth_it = optimize_params and (
        len(assets) > 1 or split_train_X.shape[0] > 80_000
    )
    if _grid_worth_it:
        model, backend, config, _best_rmse = train_gbdt_grid(
            split_train_X,
            split_train_y,
            split_val_X,
            split_val_y,
            config,
        )
    else:
        model, backend = train_gbdt(
            split_train_X,
            split_train_y,
            split_val_X,
            split_val_y,
            config,
        )
    logger.info("  backend = %s, config = %s", backend, config)

    # 8. Extraire les chemins
    logger.info("[8/10] Extraction des chemins ...")
    paths = extract_paths(
        model,
        backend,
        feature_names,
        min_score=effective_min_score,
        max_paths=max_paths,
        # LOGIC-01 (2026-08-24) : variantes OR/NOT/XOR DESACTIVEES par defaut.
        # Preuve disque : 34/68 candidats BTC/1h/6h avaient 0 trade a cause des
        # variantes NOT arbitraires. Reactivation explicite uniquement.
        enable_logical_variants=False,
    )
    logger.info("  %d chemins retenus", len(paths))

    # 9. Pour chaque chemin : construire un Einher et le backtest
    logger.info("[9/10] Generation des Einhers et backtest ...")
    # FIX P0-2 (AI Review 2026-08-20) : cost floor crypto-only.
    # Forex/indices ont des spreads ~1-3 bps, pas 10 bps. Imposer 0.10%
    # sur du forex/commodities = pénalité énorme sur chaque exit.
    raw_cost = load_costs(primary_asset)
    if asset_class == "crypto":
        costs = max(raw_cost, 0.0010)  # 10 bps floor crypto (taker fee)
    else:
        costs = max(raw_cost, 0.0001)  # 1 bp floor autres (spread+comm)
    logger.info("  Cout round-trip : %.4f (asset_class=%s, raw=%.4f)", costs, asset_class, raw_cost)
    # P2-3 (2026-08-24) : min_trades ADAPTATIF a la taille de la fenetre val.
    # Preuve disque : avec val=20% de N (~14k barres BTC/1h), un seuil fixe de 30
    # trades rejetait 27/68 candidats qui avaient 1-29 trades - dont certains
    # rentables. On garde un plancher bas (10) et on plafonne au nombre de
    # trades PHYSIQUEMENT possibles sur la fenetre (1 signal par amplitude).
    n_val_bars = max(0, X_aligned.shape[0] - int(X_aligned.shape[0] * 0.6) - max(50, horizon_bars))
    max_possible_trades = max(1, n_val_bars // max(1, horizon_bars))
    adaptive_min_trades = int(min(30, max(10, max_possible_trades // 10)))
    logger.info(
        "  Admission adaptative : fenetre val=%d barres, max possible=%d trades -> min_trades=%d",
        n_val_bars, max_possible_trades, adaptive_min_trades,
    )
    n_generated = 0
    admission_cfg = AdmissionConfig.debug() if debug else AdmissionConfig()
    if not debug:
        admission_cfg = AdmissionConfig(
            **{**admission_cfg.__dict__, "min_trades": adaptive_min_trades}
        )
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
        # FIX BUG-04 (2026-08-24) : result DOIT etre defini dans tous les cas
        # (avant : UnboundLocalError si n_aligned == 0 en single).
        backtest_embargo = max(50, horizon_bars)
        if multi and multi_per_asset and n_aligned > 0:
            # FIX BUG-1 : en multi, backtest sur TOUT l'univers (multi_per_asset).
            result = backtest_einher_multi(
                einher=einher,
                per_asset=multi_per_asset,
                feature_names=feature_names,
                costs_pct=costs,
                holdout_embargo=backtest_embargo,
            )
            # holdout : applique aussi en multi (union sur tous actifs)
            if admission_cfg.min_holdout_trades > 0:
                holdout_result = backtest_einher_multi(
                    einher=einher,
                    per_asset=[(o, X) for o, X in multi_per_asset],
                    feature_names=feature_names,
                    costs_pct=costs,
                    holdout_embargo=backtest_embargo,
                    phase="holdout",
                )
                einher = set_einher_holdout_metrics(einher, holdout_result.metrics)
        elif n_aligned > 0:
            train_end = int(n_aligned * 0.6)
            val_start = train_end + backtest_embargo
            val_end = min(n_aligned, val_start + int(n_aligned * 0.2))
            if val_start < val_end:
                result = backtest_einher(
                    einher=einher,
                    ohlcv_df=ohlcv_aligned[val_start:val_end],
                    X=X_aligned[val_start:val_end],
                    feature_names=feature_names,
                    costs_pct=costs,
                )
            else:
                # Pas assez de bougies pour le val avec embargo
                result = backtest_einher(
                    einher=einher,
                    ohlcv_df=ohlcv_aligned[:0],  # 0 bougies
                    X=X_aligned[:0],
                    feature_names=feature_names,
                    costs_pct=costs,
                )
        else:
            result = backtest_einher(
                einher=einher,
                ohlcv_df=ohlcv_aligned[:0],
                X=X_aligned[:0],
                feature_names=feature_names,
                costs_pct=costs,
            )
        einher = set_einher_metrics(einher, result.metrics)
        einher = set_einher_tp_sl(einher, result.effective_tp_pct, result.effective_sl_pct)
        all_einhers.append(einher)

    # Phase 2 : Sprint 3.1 P1 - Benjamini-Hochberg sur TOUS les Einhers
    bh_rejected_list: list[bool] = [True] * len(all_einhers)
    if admission_cfg.apply_bh and len(all_einhers) > 0:
        logger.info(
            "[9b/10] Benjamini-Hochberg sur %d candidats (FDR=%.2f) ...",
            len(all_einhers),
            admission_cfg.fdr,
        )
        _, pvalues, bh_rejected = apply_bh_to_einhers(
            all_einhers,
            fdr=admission_cfg.fdr,
        )
        bh_rejected_list = bh_rejected
        n_bh_rejected = sum(1 for r in bh_rejected if not r)
        logger.info(
            "  BH : %d/%d Einhers rejetes (FDR %.0f%%)",
            n_bh_rejected,
            len(all_einhers),
            admission_cfg.fdr * 100,
        )
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
        from .corpus import CorpusStore

        corpus_store = CorpusStore(corpus_path)
    if archive_path is not None:
        from .archive import ArchiveStore

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
                # FIX P2-6 (AI Review 2026-08-20) : propager le vrai scope
                # (asset/market/general) au lieu d'overwrite market/asset
                actual_scope = runner_scope or scope or ("market" if multi else "asset")
                archive_store.add(
                    einher,
                    rejection_reason=reason,
                    scope=actual_scope,
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
    # P1-MEM (2026-08-24) : liberation explicite des gros objets avant retour.
    # Le worker multiprocessing enchaine les triplets : sans ca, les arrays du
    # triplet N restent references jusqu'au rebinding au triplet N+1 et la
    # fragmentation s'accumule. gc.collect() rend la memoire a l'OS (Windows).
    for _name in (
        "X_global_train", "X_global_val", "X_global_holdout",
        "split_train_X", "split_val_X", "split_holdout_X",
        "X_global", "X_valid", "X_aligned", "ohlcv_df",
        "X_aligned_full", "ohlcv_aligned", "multi_per_asset",
        "model",
    ):
        # del securise : certaines variables n'existent que en single ou multi
        if _name in locals():
            del locals()[_name]
    import gc as _gc

    _gc.collect()
    return summary


def cmd_run(args: argparse.Namespace) -> int:
    """Commande `run`."""
    # FIX BUG-01 : chemins de sortie resolus contre OUTPUTS_DIR du repo,
    # plus jamais le cwd du lanceur.
    output = resolve_output(args.output, OUTPUTS_DIR / "einhers.jsonl")
    # Resolution des actifs : --assets prime sur --asset
    asset_classes_list: list[str] = []
    if args.assets:
        assets = [a.strip() for a in args.assets.split(",") if a.strip()]
    elif args.scope == "market" or args.scope == "general":
        # Sprint 3.2 P2 : resolution automatique des actifs selon le scope
        from .multi_asset_loader import list_available_assets

        asset_classes_list = [c.strip() for c in args.asset_classes.split(",") if c.strip()]
        all_assets = []
        for cls in asset_classes_list:
            # Verifier que les donnees OHLCV sont dispo (pour le backtest)
            cls_assets = list_available_assets(
                asset_class=cls,
                timeframe=args.timeframe,
                require_ohlcv=True,
            )
            all_assets.extend(cls_assets)
        # Dedupliquer et limiter
        assets = sorted(set(all_assets))[: args.max_assets]
        print(
            f"Scope {args.scope} : {len(assets)} actifs selectionnes sur {len(all_assets)} dispos"
        )
    else:
        assets = [args.asset]
    # Sprint 3.6 : --corpus et --archive optionnels (sinon legacy --output)
    corpus_path = resolve_output(args.corpus, CORPUS_PATH) if args.corpus else None
    archive_path = resolve_output(args.archive, ARCHIVE_PATH) if args.archive else None
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
        runner_scope=args.scope,
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
    "crypto",
    "forex",
    "commodities",
    "indices",
    "stocks_growth",
    "stocks_tech",
    "stocks_value",
]
# FIX 2026-08-21 (prob. 4) : TF tous dispo (via asset_selection.GLOBAL_TIMEFRAMES).
# Les horizons NE sont PAS hardcodes globalement — ils sont lus par TF depuis
# metadata.json (ex. 5m->15m/30m/1h/2h, 1h->6h/12h/1d/2d, 1d->5d/10d/20d/60d).
DEFAULT_TIMEFRAMES = ["5m", "15m", "1h", "4h", "1d"]
DEFAULT_HORIZONS = None  # None => lire depuis metadata par TF


def _discover_one_triplet(
    triplet: dict,
) -> dict:
    """Execute un seul triplet (asset, asset_class, scope, tf, horizon).

    Utilise par ProcessPoolExecutor. Doit etre picklable.
    """
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
    min_score = triplet.get("min_score", 0.0)  # <=0 => auto (aucune feuille exclue)
    min_holdout_trades = triplet.get("min_holdout_trades", 5)
    multi_assets = triplet.get("multi_assets", None)

    try:
        # FIX 2026-08-21 (prob. 1) : les actifs sont deja la bonne selection
        # (assets_v1.json) injectee par build_discovery_triplets. On ne re-resout
        # PLUS par ordre alphabetique ici.
        if multi_assets:
            assets = multi_assets
        elif scope == "asset":
            assets = [asset]
        else:
            # Defensif : si multi_assets absent, on garde l asset primaire.
            assets = [asset] if asset else []
            if not assets:
                _mark_triplet_done(
                    triplet.get("triplet_id", "?"), "skipped",
                    extra={"reason": f"no assets for scope={scope}"},
                )
                return {"status": "skipped", "reason": f"no assets for scope={scope}"}

        summary = run_pipeline(
            assets=assets,
            timeframe=tf,
            horizon_str=horizon,
            asset_class=asset_class,
            output_path=OUTPUTS_DIR / "_unused.jsonl",
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
            runner_scope=scope,  # FIX P2-6 : propager le vrai scope
        )
        # CKPT-01 : checkpoint immediat APRES ecriture corpus/archive du triplet.
        _mark_triplet_done(
            triplet["triplet_id"], "ok",
            extra={"n_admitted": summary.get("n_admitted", 0),
                   "n_rejected": summary.get("n_rejected", 0)},
        )
        return {"status": "ok", "summary": summary}
    except Exception as e:
        import traceback

        _mark_triplet_done(
            triplet.get("triplet_id", "?"), "error",
            extra={"error": str(e)[:200]},
        )
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

    PROBLEMES 1 + 4 (FIX 2026-08-21) :
    - Le run par defaut traite les 28 actifs EXACTS de assets_v1.json
      (asset_selection.load_asset_selection), plus jamais un tri alphabetique.
    - Pour chaque actif : TOUS ses TF disponibles (crypto n'a PAS 1d).
    - Pour chaque TF : SES horizons propres (lus depuis metadata.json, pas
      hardcodes). Ex : 5m -> 15m/30m/1h/2h ; 1h -> 6h/12h/1d/2d.

    Scopes :
    - asset   : 1 modele par (actif, tf, horizon)
    - market  : 1 modele par (classe, tf, horizon) - tous actifs de la classe
    - general : 1 modele (toutes classes, tf, horizon)
    """
    from .asset_selection import (
        GLOBAL_TIMEFRAMES,
        available_timeframes,
        horizons_for,
        load_asset_selection,
    )

    # --- Actifs <<assets_v1>> (28 exacts) si pas de filtre de classes ---
    if asset_classes:
        specs = [s for s in load_asset_selection() if s.asset_class in asset_classes]
    else:
        specs = load_asset_selection()

    # --- TF par defaut : tous ceux dispo globalement ---
    tf_ok = timeframes if timeframes else GLOBAL_TIMEFRAMES

    # --- Horizons explicites : valider le format ; sinon lus par TF ---
    if horizons:
        valid_h = []
        for h in horizons:
            if not h:
                continue
            try:
                _horizon_to_minutes(h)
                valid_h.append(h)
            except (ValueError, TypeError):
                logger.warning("Horizon '%s' ignore (format invalide)", h)
        horizons = valid_h or None

    triplets: list[dict] = []

    # 1. Per-asset : 1 modele par (actif, tf dispo, horizon du tf)
    if include_per_asset:
        for spec in specs:
            tfs = [tf for tf in tf_ok if tf in available_timeframes(spec.asset, spec.asset_class)]
            if not tfs:
                continue
            for tf in tfs:
                try:
                    hs = horizons if horizons else horizons_for(spec.asset, spec.asset_class, tf)
                except Exception as e:
                    logger.warning(
                        "horizons_for(%s, %s, %s) a echoue, triplet ignore : %s",
                        spec.asset,
                        spec.asset_class,
                        tf,
                        e,
                    )
                    continue
                for h in hs:
                    triplets.append(
                        {
                            "asset": spec.asset,
                            "asset_class": spec.asset_class,
                            "scope": "asset",
                            "timeframe": tf,
                            "horizon": h,
                            "multi_assets": None,
                            "triplet_id": make_triplet_id(spec.asset, spec.asset_class, "asset", tf, h),
                        }
                    )

    # 2. Per-class : 1 modele par (classe, tf, horizon) - tous actifs dispo de la classe
    if include_per_class:
        cls_map: dict[str, list] = {}
        for spec in specs:
            cls_map.setdefault(spec.asset_class, []).append(spec)
        for cls, cls_specs in cls_map.items():
            for tf in tf_ok:
                assets = [s.asset for s in cls_specs if tf in available_timeframes(s.asset, cls)]
                if not assets:
                    continue
                assets = assets[:max_assets_per_class]
                try:
                    hs = horizons if horizons else horizons_for(assets[0], cls, tf)
                except Exception as e:
                    logger.warning(
                        "horizons_for(%s, %s, %s) a echoue, triplet ignore : %s",
                        assets[0],
                        cls,
                        tf,
                        e,
                    )
                    continue
                for h in hs:
                    triplets.append(
                        {
                            "asset": assets[0],
                            "asset_class": cls,
                            "scope": "market",
                            "timeframe": tf,
                            "horizon": h,
                            "multi_assets": assets,
                            "triplet_id": make_triplet_id(assets[0], cls, "market", tf, h),
                        }
                    )

    # 3. Global : 1 modele (toutes classes, tf, horizon)
    if include_global:
        for tf in tf_ok:
            pairs = [
                (s.asset, s.asset_class)
                for s in specs
                if tf in available_timeframes(s.asset, s.asset_class)
            ]
            if not pairs:
                continue
            all_assets = sorted(set(a for a, _ in pairs))[: max_assets_per_class * 2]
            first_asset, first_cls = pairs[0]
            try:
                hs = horizons if horizons else horizons_for(first_asset, first_cls, tf)
            except Exception as e:
                logger.warning(
                    "horizons_for(%s, %s, %s) a echoue, triplet ignore : %s",
                    first_asset,
                    first_cls,
                    tf,
                    e,
                )
                continue
            for h in hs:
                triplets.append(
                    {
                        "asset": all_assets[0],
                        "asset_class": "global",
                        "scope": "general",
                        "timeframe": tf,
                        "horizon": h,
                        "multi_assets": all_assets,
                        "triplet_id": make_triplet_id(all_assets[0], "global", "general", tf, h),
                    }
                )

    logger.info(
        "build_discovery_triplets : %d triplets (actifs=%d, tf=%s)",
        len(triplets),
        len(specs),
        tf_ok,
    )
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

    from .archive import ArchiveStore
    from .corpus import CorpusStore

    # 1. Construire les triplets
    asset_classes = (
        [c.strip() for c in args.asset_classes.split(",") if c.strip()]
        if args.asset_classes
        else ALL_ASSET_CLASSES
    )
    timeframes = (
        [t.strip() for t in args.timeframes.split(",") if t.strip()]
        if args.timeframes
        else DEFAULT_TIMEFRAMES
    )
    horizons = (
        [h.strip() for h in args.horizons.split(",") if h.strip()]
        if args.horizons
        else DEFAULT_HORIZONS
    )

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
        triplets = triplets[: args.limit]
        logger.info("Limite a %d triplets", len(triplets))

    # FIX BUG-01 : corpus/archive resolus contre OUTPUTS_DIR du repo (jamais le cwd).
    corpus_file = resolve_output(args.corpus, CORPUS_PATH)
    archive_file = resolve_output(args.archive, ARCHIVE_PATH)

    # CKPT-01 : reprise - skip des triplets deja termines avec succes.
    done_ids = _load_done_triplets()
    if done_ids:
        logger.info("Checkpoint : %d triplets deja termines seront skips", len(done_ids))

    # 2. Parametres communs injectes dans chaque triplet
    common = {
        "corpus_path": str(corpus_file),
        "archive_path": str(archive_file),
        "debug": args.debug,
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "max_paths": args.max_paths,
        "min_score": args.min_score,
        "min_holdout_trades": args.min_holdout_trades,
        "max_assets": args.max_assets,
    }
    jobs = [{**t, **common} for t in triplets]
    if done_ids:
        n_before = len(jobs)
        jobs = [j for j in jobs if j.get("triplet_id") not in done_ids]
        n_skipped_ckpt = n_before - len(jobs)
        if n_skipped_ckpt:
            logger.info("Reprise : %d/%d triplets skips (checkpoint)", n_skipped_ckpt, n_before)
        if not jobs:
            logger.info("Tous les triplets sont deja termines (checkpoint a jour). Rien a faire.")
            return 0

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
        for i, job in enumerate(jobs, start=1):
            res = _discover_one_triplet(job)
            if res["status"] == "ok":
                n_ok += 1
                s = res["summary"]
                n_admitted_total += s.get("n_admitted", 0)
                n_rejected_total += s.get("n_rejected", 0)
                logger.info(
                    "[%d/%d] OK  %s/%s/%s scope=%s : %d admis, %d rejetes",
                    i,
                    len(jobs),
                    job.get("asset_class"),
                    job["timeframe"],
                    job["horizon"],
                    job["scope"],
                    s.get("n_admitted", 0),
                    s.get("n_rejected", 0),
                )
            elif res["status"] == "skipped":
                n_skipped += 1
                logger.warning(
                    "[%d/%d] SKIP %s/%s/%s scope=%s : %s",
                    i,
                    len(jobs),
                    job.get("asset_class"),
                    job["timeframe"],
                    job["horizon"],
                    job["scope"],
                    res.get("reason", "?"),
                )
            else:
                n_err += 1
                errors.append(res)
                logger.error(
                    "[%d/%d] ERR %s/%s/%s scope=%s : %s",
                    i,
                    len(jobs),
                    job.get("asset_class"),
                    job["timeframe"],
                    job["horizon"],
                    job["scope"],
                    res.get("error", "?")[:200],
                )
    else:
        # Mode parallele
        ctx_args = {} if sys.platform == "win32" else {"ctx": None}
        # MEM-02 : max_tasks_per_child recycle les workers regulierement -> purge
        # la fragmentation memoire numpy/xgboost/polars. Plus d'OOM apres N jobs.
        with ProcessPoolExecutor(
            max_workers=args.workers, max_tasks_per_child=4, **ctx_args
        ) as ex:
            futures = {ex.submit(_discover_one_triplet, j): j for j in jobs}
            n_done = 0
            for fut in as_completed(futures):
                job = futures[fut]
                n_done += 1
                try:
                    res = fut.result()
                except Exception as e:
                    res = {"status": "error", "error": str(e)}
                if res["status"] == "ok":
                    n_ok += 1
                    s = res["summary"]
                    n_admitted_total += s.get("n_admitted", 0)
                    n_rejected_total += s.get("n_rejected", 0)
                    logger.info(
                        "[%d/%d] OK  %s/%s/%s scope=%s : %d admis, %d rejetes",
                        n_done,
                        len(jobs),
                        job.get("asset_class"),
                        job["timeframe"],
                        job["horizon"],
                        job["scope"],
                        s.get("n_admitted", 0),
                        s.get("n_rejected", 0),
                    )
                elif res["status"] == "skipped":
                    n_skipped += 1
                    logger.warning(
                        "[%d/%d] SKIP %s/%s/%s scope=%s : %s",
                        n_done,
                        len(jobs),
                        job.get("asset_class"),
                        job["timeframe"],
                        job["horizon"],
                        job["scope"],
                        res.get("reason", "?"),
                    )
                else:
                    n_err += 1
                    errors.append(res)
                    logger.error(
                        "[%d/%d] ERR %s/%s/%s scope=%s : %s",
                        n_done,
                        len(jobs),
                        job.get("asset_class"),
                        job["timeframe"],
                        job["horizon"],
                        job["scope"],
                        res.get("error", "?")[:200],
                    )

    elapsed = time.time() - t0

    # 4. Rapport final
    try:
        corpus_n = CorpusStore(corpus_file).count()
    except Exception:
        corpus_n = 0
    try:
        archive_n = ArchiveStore(archive_file).count()
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
    logger.info("  Triplets SKIP: %d", n_skipped)
    logger.info("  Einhers admis total : %d", n_admitted_total)
    logger.info("  Einhers rejetes total : %d", n_rejected_total)
    logger.info("  Corpus final : %d Einhers (dans %s)", corpus_n, args.corpus)
    logger.info("  Archive     : %d Einhers (dans %s)", archive_n, args.archive)
    logger.info("  Temps       : %.1fs (%.1fmin)", elapsed, elapsed / 60)
    logger.info("  Rapport     : %s", out)
    logger.info("=" * 70)
    return 0 if n_err == 0 else 1


def main(argv: list[str] | None = None) -> int:
    """Point d'entree CLI : dispatch vers cmd_run ou cmd_discover.

    Si aucun subcommand n'est precise, lance automatiquement cmd_discover
    avec les flags top-level definis (Sprint 3.7).

    Args:
        argv : arguments CLI (None = sys.argv[1:]).
    """
    parser = argparse.ArgumentParser(
        prog="xgb_runner",
        description="Pipeline XGBoost -> Einhers (Sprint 3.7 : sans subcommand = discover auto)",
    )
    sub = parser.add_subparsers(dest="command")

    # Sprint 3.7 : flags top-level qui s'appliquent au discover par defaut.
    # Sans ca, on ne peut pas passer --workers 1 sans specifier `discover`.
    parser.add_argument(
        "--workers", type=int, default=6, help="Nombre de workers en parallele (defaut: 6)"
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default="outputs/corpus.jsonl",
        help="Chemin du corpus (Einhers admis)",
    )
    parser.add_argument(
        "--archive",
        type=str,
        default="outputs/archive.jsonl",
        help="Chemin de l'archive (Einhers rejetes)",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Limite le nombre de triplets (0=tous)"
    )
    parser.add_argument("--debug", action="store_true", help="Seuils d'admission souples")
    parser.add_argument(
        "--asset-classes", type=str, default=None, help="Classes a scanner (defaut: toutes)"
    )
    parser.add_argument("--timeframes", type=str, default=None, help="Timeframes (defaut: 1h)")
    parser.add_argument(
        "--horizons", type=str, default=None, help="Horizons (defaut: 6h,12h,1d,2d)"
    )
    parser.add_argument(
        "--max-assets", type=int, default=3, help="Limite actifs par classe (defaut: 3)"
    )
    parser.add_argument("--no-per-class", dest="per_class", action="store_false", default=True)
    parser.add_argument("--no-global", dest="global_scope", action="store_false", default=True)
    parser.add_argument("--no-per-asset", dest="per_asset", action="store_false", default=True)

    p_run = sub.add_parser("run", help="Run le pipeline complet")
    p_run.add_argument("--asset", type=str, default="BTCUSD", help="Actif unique (legacy)")
    p_run.add_argument(
        "--assets",
        type=str,
        default=None,
        help="Liste d'actifs separes par virgules (Sprint 2.3, multi-actif)",
    )
    p_run.add_argument("--timeframe", type=str, default="1h")
    p_run.add_argument(
        "--horizon",
        type=str,
        default="6h",
        choices=["6h", "12h", "1d", "2d", "1h", "2h", "4h", "8h"],
    )
    p_run.add_argument("--asset-class", type=str, default="crypto")
    p_run.add_argument("--output", type=Path, default=Path("outputs/einhers.jsonl"))
    p_run.add_argument("--n-estimators", type=int, default=100)
    p_run.add_argument("--max-depth", type=int, default=4)
    p_run.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Sprint 3.5 : 0.0005 (au lieu de 0.003) pour eviter trop d'elimination",
    )
    p_run.add_argument("--max-paths", type=int, default=100)
    p_run.add_argument("--embargo", type=int, default=50)
    p_run.add_argument("--debug", action="store_true", help="Seuils d'admission souples (debug)")
    p_run.add_argument(
        "--backend", type=str, default="auto", choices=["auto", "xgboost", "sklearn"]
    )
    # Sprint 2.3 : options anti-overfit
    p_run.add_argument(
        "--regularized",
        action="store_true",
        help="Utiliser GBDTConfig.regularized() (min_child_weight=50, etc.)",
    )
    p_run.add_argument(
        "--apply-dedup",
        action="store_true",
        help="Appliquer feature_dedup avant entrainement (drop |r|>0.85)",
    )
    p_run.add_argument(
        "--drop-sparse", action="store_true", help="Drop les patterns sparses (pct_True < 0.5%%)"
    )
    p_run.add_argument(
        "--min-holdout-trades",
        type=int,
        default=0,
        help="Sprint 2.4.1 : min trades sur holdout pour admettre un Einher (0=desactive)",
    )
    p_run.add_argument(
        "--bagging-seeds",
        type=int,
        default=1,
        help="Sprint 2.4.2 : nombre de seeds pour bagging (1=desactive)",
    )
    p_run.add_argument(
        "--walk-forward-folds",
        type=int,
        default=1,
        help="Sprint 2.4.3 : nombre de folds walk-forward (1=desactive)",
    )
    # Sprint 3.2 P2 : 3 niveaux de scope
    p_run.add_argument(
        "--scope",
        type=str,
        default="asset",
        choices=["asset", "market", "general"],
        help="Sprint 3.2 P2 : asset=1 actif, market=1 classe, general=toutes classes",
    )
    p_run.add_argument(
        "--asset-classes",
        type=str,
        default="crypto",
        help="Sprint 3.2 P2 : classes separees par virgules (ex: crypto,forex,indices)",
    )
    p_run.add_argument(
        "--max-assets",
        type=int,
        default=10,
        help="Sprint 3.2 P2 : limite d'actifs par run (pour scaler)",
    )
    # Sprint 3.6 : corpus + archive (optionnels, remplacent --output si fournis)
    p_run.add_argument(
        "--corpus",
        type=str,
        default=None,
        help="Sprint 3.6 : chemin du corpus (Einhers admis). Si fourni, "
        "les admis vont ici au lieu de --output.",
    )
    p_run.add_argument(
        "--archive",
        type=str,
        default=None,
        help="Sprint 3.6 : chemin de l'archive (Einhers rejetes avec raison).",
    )
    p_run.set_defaults(func=cmd_run)

    # Sprint 3.6 : cmd_discover - discovery complet en parallele
    p_disc = sub.add_parser(
        "discover",
        help="Sprint 3.6 : lance TOUS les modeles (asset/class/global) x tf x horizon en parallele",
    )
    p_disc.add_argument(
        "--asset-classes",
        type=str,
        default=None,
        help=f"Classes a scanner (defaut: {','.join(ALL_ASSET_CLASSES)})",
    )
    p_disc.add_argument(
        "--timeframes",
        type=str,
        default=None,
        help=f"Timeframes (defaut: {','.join(DEFAULT_TIMEFRAMES)})",
    )
    _horizons_default_help = (
        ",".join(DEFAULT_HORIZONS) if DEFAULT_HORIZONS else "lu depuis metadata par TF"
    )
    p_disc.add_argument(
        "--horizons",
        type=str,
        default=None,
        help=f"Horizons (defaut: {_horizons_default_help})",
    )
    p_disc.add_argument(
        "--per-asset",
        action="store_true",
        default=True,
        help="Inclure les modeles per-asset (defaut: True)",
    )
    p_disc.add_argument("--no-per-asset", dest="per_asset", action="store_false")
    p_disc.add_argument(
        "--per-class",
        action="store_true",
        default=True,
        help="Inclure les modeles per-class (defaut: True)",
    )
    p_disc.add_argument("--no-per-class", dest="per_class", action="store_false")
    p_disc.add_argument(
        "--global-scope",
        action="store_true",
        default=True,
        help="Inclure le modele global toutes classes (defaut: True)",
    )
    p_disc.add_argument("--no-global", dest="global_scope", action="store_false")
    p_disc.add_argument(
        "--max-assets",
        type=int,
        default=5,
        help="Limite d'actifs par classe (pour scaler le per-class et le global)",
    )
    p_disc.add_argument(
        "--workers", type=int, default=6, help="Nombre de workers en parallele (1=sequentiel)"
    )
    p_disc.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limite le nombre de triplets (0=tous). Utile pour test rapide.",
    )
    p_disc.add_argument("--debug", action="store_true", help="Seuils d'admission souples (debug)")
    p_disc.add_argument("--n-estimators", type=int, default=100)
    p_disc.add_argument("--max-depth", type=int, default=3)
    p_disc.add_argument("--max-paths", type=int, default=30)
    p_disc.add_argument("--min-score", type=float, default=0.0)
    p_disc.add_argument("--min-holdout-trades", type=int, default=5)
    p_disc.add_argument(
        "--corpus",
        type=str,
        default="outputs/corpus.jsonl",
        help="Chemin du corpus (Einhers admis)",
    )
    p_disc.add_argument(
        "--archive",
        type=str,
        default="outputs/archive.jsonl",
        help="Chemin de l'archive (Einhers rejetes avec raison)",
    )
    p_disc.set_defaults(func=cmd_discover)

    # FIX Sprint 3.7 (user request) : si pas de subcommand, lancer
    # discover automatiquement. On utilise parse_known_args pour
    # accepter les flags discover (--workers, --corpus, etc.) avant
    # la subcommand.
    args, remaining = parser.parse_known_args(argv)
    if args.command is None:
        # Re-parser avec les flags restants en tant qu'args de discover
        # D'abord prendre les defaults de discover
        defaults = vars(p_disc.parse_args([]))
        # Override avec les flags detectes au top level
        cli_overrides = vars(parser.parse_args(argv))
        for k in (
            "workers",
            "debug",
            "corpus",
            "archive",
            "max_assets",
            "n_estimators",
            "max_depth",
            "max_paths",
            "min_score",
            "min_holdout_trades",
            "asset_classes",
            "timeframes",
            "horizons",
            "per_asset",
            "per_class",
            "global_scope",
            "limit",
        ):
            v = cli_overrides.get(k)
            if v is not None and v != defaults.get(k):
                defaults[k] = v
        discover_args = argparse.Namespace(**defaults)
        return cmd_discover(discover_args)
    # Si subcommand explicite, parser normalement avec remaining
    if remaining:
        args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    )
    sys.exit(main())
