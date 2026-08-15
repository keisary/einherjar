"""discovery.py — Point d'entrée du moteur de découverte Einherjar.

Modes (Phase 1, simplifié) :
  engine  — Construit / vérifie le moteur d'évaluation
  admit   — STGP → génération → calibration → admission → corpus
  holdout — Évaluation finale unique sur le holdout (sacré)

Usage :
  python -m einherjar.research.discovery admit --data-asset BTCUSD \\
      --data-timeframe 15m --horizon-index 2 --pop-size 20 --n-gen 5

Pas de baselines, pas de comparateur, pas de sélection multi-générateurs.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

from einherjar.research.config.loader import EinherjarConfig, load_config
from einherjar.research.utils.logging import configure_logging

# Imports lazys pour éviter les circular imports au module level
# (import dans chaque handler)

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

MODES = ("engine", "admit", "holdout")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="discovery",
        description="Moteur de découverte Einherjar (pipeline simplifié Phase 1).",
    )
    parser.add_argument("mode", choices=MODES, help="Étape du pipeline.")
    parser.add_argument(
        "--config", type=Path,
        default=Path(__file__).resolve().parent / "config",
        help="Dossier de configuration (défaut: ./config).",
    )
    parser.add_argument("--data-version", type=str, default=None,
                        help="Override du data_version.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed RNG maître (défaut: 42).")
    parser.add_argument(
        "--data-root", type=str,
        default=r"D:\midas_v2\midasV3\src\data\compiled",
        help="Racine des datasets .npy MIDAS V3.",
    )
    parser.add_argument("--data-asset", type=str, default="BTCUSD",
                        help="Symbole de l'actif (défaut: BTCUSD).")
    parser.add_argument("--data-class", type=str, default="crypto",
                        choices=("crypto", "forex", "indices", "commodities",
                                 "stocks_growth", "stocks_tech", "stocks_value"),
                        help="Classe d'actifs (défaut: crypto).")
    parser.add_argument("--data-timeframe", type=str, default="15m",
                        help="Timeframe (défaut: 15m).")

    # Paramètres STGP
    parser.add_argument("--horizon-index", type=int, default=1, choices=[0, 1, 2, 3],
                        help="Indice d'horizon (0-3, défaut: 1). Correspond à la colonne "
                             "des matrices Y_* (pour traçage uniquement en Phase 1).")
    parser.add_argument("--max-depth", type=int, default=4,
                        help="Profondeur max des conditions (défaut: 4).")
    parser.add_argument("--pop-size", type=int, default=20,
                        help="Taille de la population STGP (défaut: 20).")
    parser.add_argument("--n-gen", type=int, default=5,
                        help="Nombre de générations (défaut: 5).")
    parser.add_argument("--n-eval", type=int, default=50,
                        help="Budget d'évaluations admission (défaut: 50).")

    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    parser.add_argument("--dry-run", action="store_true",
                        help="Affiche le plan sans exécuter.")
    parser.add_argument("--hypothesis-file", type=Path, default=None,
                        help="Fichier JSON Hypothesis (utilisé par holdout).")
    return parser


# --------------------------------------------------------------------------- #
# Helpers données
# --------------------------------------------------------------------------- #


def _load_real_data(
    config: EinherjarConfig, data_root: str,
    asset: str, asset_class: str, timeframe: str,
):
    """Charge OHLCV + features depuis les .npy et découpe en train/val/holdout

    Retourne (train_ohlcv, train_features, val_ohlcv, val_features,
              holdout_ohlcv, holdout_features, splits_key).
    """
    from einherjar.research.data.features import FeaturesFrame
    from einherjar.research.data.npy_real_loader import load_features_from_npy
    from einherjar.research.data.ohlcv import OhlcvProvider
    from einherjar.research.data.validation import validate_or_raise
    from einherjar.research.data.versioning import make_frame_data_version, make_splits_hash

    root = Path(data_root)
    full_ohlcv = OhlcvProvider().load(
        asset=asset, timeframe=timeframe, data_version="raw",
        asset_class=asset_class,
    )
    full_features = load_features_from_npy(
        asset=asset, asset_class=asset_class, timeframe=timeframe,
        config=config, data_root=root,
    )
    common_ts = full_ohlcv.df.select("timestamp").join(
        full_features.df.select("timestamp"), on="timestamp", how="inner",
    )
    if common_ts.is_empty():
        raise RuntimeError(
            f"No common timestamp between raw OHLCV and features for {asset} x {timeframe}"
        )
    full_ohlcv = OhlcvFrame(
        asset=asset, timeframe=timeframe,
        df=full_ohlcv.df.join(common_ts, on="timestamp", how="inner").sort("timestamp"),
        data_version="pending",
    )
    full_features = FeaturesFrame(
        asset=asset, timeframe=timeframe,
        df=full_features.df.join(common_ts, on="timestamp", how="inner").sort("timestamp"),
        feature_names=full_features.feature_names, data_version="pending",
    )
    data_version = make_frame_data_version(full_ohlcv, full_features, config).tag
    full_ohlcv = OhlcvFrame(asset, timeframe, full_ohlcv.df, data_version)
    full_features = FeaturesFrame(
        asset, timeframe, full_features.df, full_features.feature_names, data_version,
    )
    validate_or_raise(full_ohlcv, full_features)
    if full_ohlcv.n_bougies != full_features.n_bougies:
        raise RuntimeError(
            f"OHLCV/features désynchronisés : OHLCV={full_ohlcv.n_bougies} "
            f"!= features={full_features.n_bougies}"
        )

    n = full_ohlcv.n_bougies
    ratios = config.splits["ratios"]
    train_boundary = int(n * float(ratios["train"]))
    val_boundary = train_boundary + int(n * float(ratios["val"]))
    purge = int(config.evaluation["n_window"]["max_n"])
    embargo = int(config.splits.get("embargo", {}).get("bougies", 0))
    if not config.splits.get("purging", {}).get("enabled", True):
        purge = 0
    if not config.splits.get("embargo", {}).get("enabled", True):
        embargo = 0
    train_end = train_boundary - purge
    val_start = train_boundary + embargo
    val_end = val_boundary - purge
    holdout_start = val_boundary + embargo
    if train_end <= 0 or val_start >= val_end or holdout_start >= n:
        raise RuntimeError(
            f"Insufficient data after purge/embargo: n={n}, purge={purge}, embargo={embargo}"
        )

    def _slice_ohlcv(start: int, end: int) -> "OhlcvFrame":
        from einherjar.research.data.ohlcv import OhlcvFrame
        return OhlcvFrame(
            asset=full_ohlcv.asset, timeframe=full_ohlcv.timeframe,
            df=full_ohlcv.df.slice(start, end - start),
            data_version=full_ohlcv.data_version,
        )

    def _slice_features(start: int, end: int) -> "FeaturesFrame":
        from einherjar.research.data.features import FeaturesFrame
        return FeaturesFrame(
            asset=full_features.asset, timeframe=full_features.timeframe,
            df=full_features.df.slice(start, end - start),
            feature_names=full_features.feature_names,
            data_version=full_features.data_version,
        )

    splits_key = make_splits_hash(
        train_start=0, train_end=train_end,
        val_start=val_start, val_end=val_end,
        holdout_start=holdout_start, holdout_end=n,
        embargo_bougies=embargo, horizon_label=purge,
    )
    return (
        _slice_ohlcv(0, train_end), _slice_features(0, train_end),
        _slice_ohlcv(val_start, val_end), _slice_features(val_start, val_end),
        _slice_ohlcv(holdout_start, n), _slice_features(holdout_start, n),
        splits_key,
    )


def _persist_data_version(
    config: EinherjarConfig,
    train_ohlcv: Any, train_features: Any,
    *,
    store_path: Path = Path("outputs/data_versions.jsonl"),
) -> Any:
    """Persiste le DataVersion courant via DataVersionStore."""
    from einherjar.research.data.versioning import (
        DataVersionStore, make_frame_data_version, verify_data_version_locked,
    )
    dv = make_frame_data_version(train_ohlcv, train_features, config)
    store = DataVersionStore(store_path)
    locked = verify_data_version_locked(dv, store)
    logger.info("DataVersion verrouillé = %s (hash=%s)", locked.tag, locked.hash[:12])
    return locked


# --------------------------------------------------------------------------- #
# Handlers
# --------------------------------------------------------------------------- #


def handle_engine(args: argparse.Namespace) -> int:
    """Step 0 — Construit/vérifie le moteur d'évaluation."""
    logger.info("[ENGINE] Moteur d'évaluation")
    config = load_config(args.config)
    from einherjar.research.engine.evaluator import EvaluationEngine
    engine = EvaluationEngine(
        config=config, data_version=args.data_version or "v1", seed=args.seed,
    )
    logger.info("Moteur OK : seed=%d, ATR period=%d", engine.seed, engine._atr_estimator.period)
    return 0


def handle_admit(args: argparse.Namespace) -> int:
    """Step 5 (ex-step) — STGP → calibration → admission → corpus.

    Instancie TypedGPGenerator directement, génère, calibre, évalue,
    puis applique AdmissionDecider complet.
    """
    logger.info("[ADMIT] Admission au corpus")
    config = load_config(args.config)
    data_root = args.data_root

    # 1. Charge les données
    from einherjar.research.data.ohlcv import OhlcvFrame
    from einherjar.research.data.features import FeaturesFrame
    try:
        loaded = _load_real_data(config, data_root, args.data_asset, args.data_class, args.data_timeframe)
        train_ohlcv, train_features, val_ohlcv, val_features, _, _, splits_hash = loaded
    except Exception as exc:
        logger.error("Impossible de charger les données : %s", exc)
        return 2

    _persist_data_version(config, train_ohlcv, train_features)
    data_version = args.data_version or train_ohlcv.data_version
    logger.info(
        "Données : %s %s × %s, split train=%d val=%d",
        args.data_asset, args.data_class, args.data_timeframe,
        train_ohlcv.n_bougies, val_ohlcv.n_bougies,
    )

    # 2. Instancie le moteur + config STGP
    from einherjar.research.engine.evaluator import EvaluationEngine
    from einherjar.research.generators.config import TypedGPConfig
    from einherjar.research.generators.typedgp import TypedGPGenerator

    engine = EvaluationEngine(config=config, data_version=data_version, seed=args.seed)

    stgp_config = TypedGPConfig(
        seed=args.seed,
        einherjar_config=config,
        data_version=data_version,
        max_depth=args.max_depth,
        horizon_index=args.horizon_index,
        timeframe=args.data_timeframe,
        asset=args.data_asset,
        population_size=args.pop_size,
        n_generations=args.n_gen,
        n_eval_budget=args.n_eval,
    )
    logger.info("Config STGP : horizon_index=%d, pop=%d, gen=%d, max_depth=%d",
                stgp_config.horizon_index, stgp_config.population_size,
                stgp_config.n_generations, stgp_config.max_depth)

    # 3. Instancie le générateur et génère
    generator = TypedGPGenerator(
        config=stgp_config,
        engine=engine,
        population_size=stgp_config.population_size,
        n_generations=stgp_config.n_generations,
    )
    generator.bind_data(train_ohlcv, train_features, val_ohlcv, val_features)
    result = generator.generate()
    hyps = result.hypotheses
    logger.info("Génération : %d hypothèses (%s)", len(hyps), result.generation_time_s)

    if not hyps:
        logger.warning("Aucune hypothèse générée — fin.")
        return 0

    # 4. Calibration + évaluation validation
    eval_budget = min(args.n_eval, len(hyps))
    hyps_a_evaluer = hyps[:eval_budget]
    evaluated_candidates: dict[str, tuple[Any, Any]] = {}
    for hyp in hyps_a_evaluer:
        try:
            calibrated = engine.train_calibrate(hyp, train_ohlcv, train_features)
            measures = engine.test_on(hyp, val_ohlcv, val_features, calibrated, "val")
            evaluated_candidates[hyp.id] = (calibrated, measures)
        except Exception as exc:
            logger.debug("Candidat exclu : %s — %s", hyp.id, exc)

    if not evaluated_candidates:
        logger.warning("Aucun candidat évalué — fin.")
        return 0

    # 5. Admission
    from einherjar.research.admission.decision import AdmissionDecider
    from einherjar.research.corpus.store import CorpusEntry, CorpusStore
    from einherjar.research.utils.stats import periods_per_year_for_timeframe

    decider = AdmissionDecider(config=config, data_version=data_version, seed=args.seed)
    corpus = CorpusStore()
    corpus_entries = corpus.load()

    ppy = periods_per_year_for_timeframe(val_ohlcv.timeframe)
    n_val_years = max(len(val_ohlcv.df) / ppy, 0.05) if ppy and ppy > 0 else 1.0
    n_trials_admission = max(1, len(hyps_a_evaluer))

    # Matrice PBO
    pbo_candidate_paths = [
        tuple((t.entry_idx, t.exit_idx, t.ret_pct_net)
              for t in measures.trades)
        for _, measures in evaluated_candidates.values()
    ]

    n_admitted = 0
    n_rejected = 0
    reasons: dict[str, int] = {}

    for hyp in hyps_a_evaluer:
        evaluation = evaluated_candidates.get(hyp.id)
        if evaluation is None:
            continue
        calibrated, m_val = evaluation
        returns_val = [t.ret_pct_net for t in m_val.trades]

        from einherjar.research.admission.diversity import compute_corpus_fracs
        current_fracs = compute_corpus_fracs(corpus_entries, config)

        decision = decider.decide(
            hypothesis_id=hyp.id,
            condition_tree=hyp.condition_tree,
            direction=hyp.direction,
            universe=hyp.universe,
            amplitude=hyp.amplitude,
            calibrated=calibrated,
            mesures_val=m_val,
            returns_val=returns_val,
            signal_indices=tuple(t.entry_idx - 1 for t in m_val.trades),
            corpus_signal_dates=[tuple(e.meta.get("signal_indices", ())) for e in corpus_entries],
            corpus_ret_series=[e.ret_series for e in corpus_entries],
            pbo_candidate_paths=pbo_candidate_paths,
            current_corpus_fracs=current_fracs,
            cooldown_k=hyp.cooldown_k,
            n_indep_trials=n_trials_admission,
            n_val_years=n_val_years,
            corpus_fingerprints=[e.fingerprint for e in corpus_entries],
            horizon_index=stgp_config.horizon_index,
        )
        if decision.admitted:
            from einherjar.research.corpus.store import CorpusEntry
            entry = CorpusEntry(
                hypothesis_id=hyp.id,
                condition_tree=hyp.condition_tree,
                direction=hyp.direction,
                universe=hyp.universe,
                amplitude=hyp.amplitude,
                calibrated=calibrated,
                measures=m_val,
                ret_series=tuple(t.ret_pct_net for t in m_val.trades),
                meta={
                    "seed": args.seed,
                    "data_version": data_version,
                    "timeframe": args.data_timeframe,
                    "asset": args.data_asset,
                    "data_class": args.data_class,
                    "splits_key": splits_hash,
                    "horizon_index": stgp_config.horizon_index,
                    "max_depth": stgp_config.max_depth,
                    "signal_indices": [t.entry_idx - 1 for t in m_val.trades],
                    "decision": decision.to_dict() if hasattr(decision, "to_dict") else {},
                },
            )
            corpus.append(entry)
            corpus_entries.append(entry)
            n_admitted += 1
            logger.info("ADMIS : %s (sharpe=%.4f)", hyp.id, m_val.sharpe_net)
        else:
            n_rejected += 1
            reasons[decision.reason or "UNKNOWN"] = reasons.get(decision.reason or "UNKNOWN", 0) + 1

    logger.info(
        "Admission : %d admis, %d rejetés (%s)",
        n_admitted, n_rejected,
        dict(reasons) if reasons else "aucun motif enregistré",
    )
    return 0


def handle_holdout(args: argparse.Namespace) -> int:
    """Step 6 — Évaluation finale unique sur le holdout.
    
    À implémenter avec holdout/ledger.py et holdout/evaluator.py.
    """
    logger.info("[HOLDOUT] Évaluation sacrée — à implémenter")
    logger.warning("Holdout pas encore câblé dans cette version simplifiée.")
    return 0


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    configure_logging(getattr(args, "log_level", "INFO"))
    logger.info("=== Discovery Phase 1 — %s ===", datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("Mode=%s | %s %s %s%s",
                args.mode, args.data_asset, args.data_class, args.data_timeframe,
                f" seed={args.seed}" if args.seed else "")

    handlers = {
        "engine": handle_engine,
        "admit": handle_admit,
        "holdout": handle_holdout,
    }
    handler = handlers.get(args.mode)
    if handler is None:
        logger.error("Mode inconnu : %s", args.mode)
        return 1
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())