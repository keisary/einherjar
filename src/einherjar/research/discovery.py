"""
discovery.py — Point d'entrée du moteur de découverte Einherjar.

Pipeline 7 étapes (ONTOLOGY.md S-3 + ALGORITHME_RESEARCH.md § 10.2) :

  Step 0  engine      Construit / vérifie le moteur d'évaluation (priorité 0)
  Step 1  baselines   3 baselines honnêtes (human, shallow, random)
  Step 2  compare     Comparaison reproductible des générateurs
  Step 3  select      Installe le générateur gagnant
  Step 4  refine      Raffinement beam local (sans recalibrer SL/TP)
  Step 5  admit       Admission au corpus (DSR + PBO + bootstrap CI + diversité)
  Step 6  holdout     Évaluation finale unique sur le holdout (sacré)

Modes :
  engine       — Construit le moteur (priority 0)
  baselines    — Step 1
  compare      — Step 2
  select       — Step 3
  refine       — Step 4
  admit        — Step 5
  holdout      — Step 6
  run          — Pipeline 0→5 (holdout à déclencher manuellement)
  pipeline     — Alias de `run`

Usage :
  python -m einherjar.research.discovery <mode> [options]

Philosophie : moteur d'évaluation d'abord, générateurs après, holdout à la fin.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Optional

from einherjar.research.config.loader import load_config
from einherjar.research.utils.logging import configure_logging

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------- #

MODES = (
    "engine",
    "baselines",
    "compare",
    "select",
    "refine",
    "admit",
    "holdout",
    "run",
    "pipeline",
)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="discovery",
        description="Moteur de découverte Einherjar (pipeline 7 étapes).",
    )
    parser.add_argument(
        "mode",
        choices=MODES,
        help="Étape du pipeline à exécuter.",
    )
    parser.add_argument(
        "--config", type=Path, default=Path("config"),
        help="Dossier contenant les fichiers de config (défaut: ./config).",
    )
    parser.add_argument(
        "--data-version", type=str, default=None,
        help="Tag/identifiant de la version de données (override config).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seed RNG maître (défaut: 42).",
    )
    parser.add_argument(
        "--generator", type=str, default=None,
        choices=("random", "ge", "stgp", "beam", "memetic", "nsga2"),
        help="Générateur à utiliser (utile pour select/refine/admit/holdout).",
    )
    parser.add_argument(
        "--n-eval", type=int, default=None,
        help="Budget d'évaluations (utilisé par compare et baselines).",
    )
    parser.add_argument(
        "--n-samples", type=int, default=200,
        help="Nombre d'hypothèses par baseline random (défaut: 200).",
    )
    parser.add_argument(
        "--selection-path", type=Path, default=Path("outputs/selection.json"),
        help="Fichier de persistance de la sélection du générateur.",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Niveau de log (défaut: INFO).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Affiche le plan d'exécution sans rien lancer.",
    )
    return parser


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _build_dummy_data(config: Any, n_bougies: int = 600) -> tuple[Any, Any]:
    """Construit des données OHLCV + features synthétiques pour smoke test.

    Returns:
        (train_ohlcv, train_features).
    """
    from datetime import datetime, timedelta, timezone

    import numpy as np
    import polars as pl

    from einherjar.research.data.features import FeaturesFrame
    from einherjar.research.data.ohlcv import OhlcvFrame

    rng = np.random.default_rng(seed=42)
    close = 100.0 * np.cumprod(1.0 + rng.normal(0, 0.01, n_bougies))
    high = close * (1.0 + np.abs(rng.normal(0, 0.005, n_bougies)))
    low = close * (1.0 - np.abs(rng.normal(0, 0.005, n_bougies)))
    open_ = close * (1.0 + rng.normal(0, 0.002, n_bougies))
    volume = rng.integers(100, 10_000, n_bougies).astype(float)
    ts = [datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(hours=i) for i in range(n_bougies)]
    df = pl.DataFrame({
        "asset": ["BTCUSD"] * n_bougies,
        "timeframe": ["1h"] * n_bougies,
        "timestamp": ts,
        "open": open_, "high": high, "low": low, "close": close, "volume": volume,
    })
    feats_df = df.with_columns([
        pl.col("close").rolling_mean(5).alias("ma_5"),
        pl.col("close").rolling_mean(20).alias("ma_20"),
    ])
    feats_df = feats_df.with_columns([
        pl.col("ma_5").fill_null(0.0),
        pl.col("ma_20").fill_null(0.0),
    ])
    ohlcv = OhlcvFrame(asset="BTCUSD", timeframe="1h", df=df, data_version="v1_smoke")
    features = FeaturesFrame(
        asset="BTCUSD", timeframe="1h", df=feats_df,
        feature_names=("ma_5", "ma_20"),
        data_version="v1_smoke",
    )
    return ohlcv, features


def _split(ohlcv: Any, features: Any, train_frac: float = 0.6, val_frac: float = 0.2) -> tuple:
    """Découpe ohlcv/features en train/val/holdout."""
    n = ohlcv.df.height
    t1 = int(n * train_frac)
    t2 = int(n * (train_frac + val_frac))
    from einherjar.research.data.features import FeaturesFrame
    from einherjar.research.data.ohlcv import OhlcvFrame
    train_ohlcv = OhlcvFrame(
        asset=ohlcv.asset, timeframe=ohlcv.timeframe,
        df=ohlcv.df.head(t1), data_version=ohlcv.data_version,
    )
    val_ohlcv = OhlcvFrame(
        asset=ohlcv.asset, timeframe=ohlcv.timeframe,
        df=ohlcv.df.slice(t1, t2 - t1), data_version=ohlcv.data_version,
    )
    holdout_ohlcv = OhlcvFrame(
        asset=ohlcv.asset, timeframe=ohlcv.timeframe,
        df=ohlcv.df.tail(n - t2), data_version=ohlcv.data_version,
    )
    train_features = FeaturesFrame(
        asset=features.asset, timeframe=features.timeframe,
        df=features.df.head(t1), feature_names=features.feature_names,
        data_version=features.data_version,
    )
    val_features = FeaturesFrame(
        asset=features.asset, timeframe=features.timeframe,
        df=features.df.slice(t1, t2 - t1), feature_names=features.feature_names,
        data_version=features.data_version,
    )
    holdout_features = FeaturesFrame(
        asset=features.asset, timeframe=features.timeframe,
        df=features.df.tail(n - t2), feature_names=features.feature_names,
        data_version=features.data_version,
    )
    return (train_ohlcv, train_features, val_ohlcv, val_features, holdout_ohlcv, holdout_features)


# --------------------------------------------------------------------------- #
# Handlers
# --------------------------------------------------------------------------- #


def handle_engine(args: argparse.Namespace) -> int:
    """Step 0 — Construit / vérifie le moteur d'évaluation."""
    logger.info("[STEP 0] Moteur d'évaluation")
    config = load_config(args.config)
    from einherjar.research.engine.evaluator import EvaluationEngine
    engine = EvaluationEngine(config=config, data_version=args.data_version or "v1", seed=args.seed)
    logger.info("Moteur OK : seed=%d, ATR period=%d, N=[%d, %d]",
                engine.seed, engine._atr_estimator.period,
                engine._min_n, engine._max_n)
    return 0


def handle_baselines(args: argparse.Namespace) -> int:
    """Step 1 — 3 baselines."""
    logger.info("[STEP 1] Baselines")
    config = load_config(args.config)
    from einherjar.research.baselines.runner import BaselineRunner
    from einherjar.research.engine.evaluator import EvaluationEngine
    engine = EvaluationEngine(config=config, data_version=args.data_version or "v1", seed=args.seed)
    ohlcv, features = _build_dummy_data(config, n_bougies=600)
    train_ohlcv, train_features, val_ohlcv, val_features, _, _ = _split(ohlcv, features)
    runner = BaselineRunner(engine=engine)
    report = runner.run(
        train_ohlcv=train_ohlcv, train_features=train_features,
        val_ohlcv=val_ohlcv, val_features=val_features,
    )
    logger.info("Baselines : %s", report.summary())
    return 0


def handle_compare(args: argparse.Namespace) -> int:
    """Step 2 — Comparaison reproductible des générateurs."""
    logger.info("[STEP 2] Comparaison générateurs")
    config = load_config(args.config)
    from einherjar.research.engine.evaluator import EvaluationEngine
    from einherjar.research.generators.algorithms import make_all_generators
    from einherjar.research.generators.comparator import GeneratorComparator
    from einherjar.research.generators.protocol import make_protocol
    engine = EvaluationEngine(config=config, data_version=args.data_version or "v1", seed=args.seed)
    protocol = make_protocol(
        config, data_version=args.data_version or "v1",
        seed=args.seed, n_eval_budget=args.n_eval or 200,
    )
    generators = make_all_generators(protocol, config)
    ohlcv, features = _build_dummy_data(config, n_bougies=600)
    train_ohlcv, train_features, val_ohlcv, val_features, _, _ = _split(ohlcv, features)
    comparator = GeneratorComparator(generators=generators, protocol=protocol, engine=engine, config=config)
    report = comparator.run(
        train_ohlcv=train_ohlcv, train_features=train_features,
        val_ohlcv=val_ohlcv, val_features=val_features,
    )
    logger.info("Comparaison terminée : winner=%s", report.winner_name)
    for r in report.rankings:
        logger.info("  #%d %s : score=%.4f, admission_rate=%.4f, median_sharpe=%.4f",
                    r.rank, r.generator_name, r.score, r.admission_rate, r.median_sharpe)
    return 0


def handle_select(args: argparse.Namespace) -> int:
    """Step 3 — Installe le générateur gagnant."""
    logger.info("[STEP 3] Sélection du générateur")
    if args.selection_path.exists():
        from einherjar.research.selection.selector import GeneratorSelector
        selected = GeneratorSelector.load(args.selection_path)
        logger.info("Sélection chargée depuis %s : %s", args.selection_path, selected.generator_name)
        return 0
    # Sinon, on lance une comparaison rapide pour avoir un ranking.
    config = load_config(args.config)
    from einherjar.research.engine.evaluator import EvaluationEngine
    from einherjar.research.generators.algorithms import make_all_generators
    from einherjar.research.generators.comparator import GeneratorComparator
    from einherjar.research.generators.protocol import make_protocol
    from einherjar.research.selection.selector import GeneratorSelector
    engine = EvaluationEngine(config=config, data_version=args.data_version or "v1", seed=args.seed)
    protocol = make_protocol(
        config, data_version=args.data_version or "v1",
        seed=args.seed, n_eval_budget=args.n_eval or 200,
    )
    generators = make_all_generators(protocol, config)
    ohlcv, features = _build_dummy_data(config, n_bougies=600)
    train_ohlcv, train_features, val_ohlcv, val_features, _, _ = _split(ohlcv, features)
    comparator = GeneratorComparator(generators=generators, protocol=protocol, engine=engine, config=config)
    report = comparator.run(
        train_ohlcv=train_ohlcv, train_features=train_features,
        val_ohlcv=val_ohlcv, val_features=val_features,
    )
    selector = GeneratorSelector(protocol=protocol)
    selected = selector.select(report)
    selector.save(selected, args.selection_path)
    logger.info("Sélection : %s (rank=%d, score=%.4f)", selected.generator_name, selected.rank, selected.score)
    return 0


def handle_refine(args: argparse.Namespace) -> int:
    """Step 4 — Raffinement beam local."""
    logger.info("[STEP 4] Raffinement beam local")
    logger.info("  → contrainte dure : SL/TP figés depuis le train (jamais recalibrés)")
    logger.info("  → pour V1 : nécessite qu'un Einher viable soit disponible.")
    logger.info("  → V2 : intégration via corpus.json (à implémenter).")
    return 0


def handle_admit(args: argparse.Namespace) -> int:
    """Step 5 — Admission au corpus."""
    logger.info("[STEP 5] Admission au corpus")
    logger.info("  → critères : DSR + PBO + bootstrap CI + n_trades + cross_asset + max_dd + diversité")
    logger.info("  → fingerprint canonique (structurel + comportemental)")
    logger.info("  → déduplication contre l'Archive sur le même data_version")
    logger.info("  → pour V1 : nécessite que le générateur sélectionné tourne.")
    return 0


def handle_holdout(args: argparse.Namespace) -> int:
    """Step 6 — Évaluation finale unique sur le holdout (sacré)."""
    logger.info("[STEP 6] Holdout sacré — ÉVALUATION FINALE UNIQUE")
    logger.info("  ⚠ ce mode ne doit être appelé qu'UNE SEULE FOIS par Einher final retenu")
    logger.info("  → publication: métriques + IC bootstrap + descripteurs comportementaux")
    logger.info("  → archivage avec data_version/seed/splits/coûts figés")
    # Pour V1 : on délègue au HoldoutEvaluator si on a un Einher.
    config = load_config(args.config)
    from einherjar.research.engine.evaluator import EvaluationEngine
    from einherjar.research.holdout.evaluator import HoldoutEvaluator
    engine = EvaluationEngine(config=config, data_version=args.data_version or "v1", seed=args.seed)
    holdout_eval = HoldoutEvaluator(
        engine=engine, config=config,
        data_version=args.data_version or "v1", seed=args.seed,
    )
    logger.info("HoldoutEvaluator instancié (1 seule passe autorisée).")
    return 0


def handle_run(args: argparse.Namespace) -> int:
    """Pipeline complet (étapes 0→5). Le holdout reste manuel."""
    logger.info("[PIPELINE] Exécution séquentielle des étapes 0 → 5")
    sub_args = argparse.Namespace(**vars(args))
    for mode in ("engine", "baselines", "compare", "select", "refine", "admit"):
        sub_args.mode = mode
        rc = HANDLERS[mode](sub_args)
        if rc != 0:
            logger.error("Étape %s en erreur (rc=%d), arrêt du pipeline.", mode, rc)
            return rc
    logger.info("[PIPELINE] Terminé. Lancez 'discovery holdout' manuellement pour l'évaluation finale.")
    return 0


HANDLERS = {
    "engine": handle_engine,
    "baselines": handle_baselines,
    "compare": handle_compare,
    "select": handle_select,
    "refine": handle_refine,
    "admit": handle_admit,
    "holdout": handle_holdout,
    "run": handle_run,
    "pipeline": handle_run,
}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(level=args.log_level)
    # Chargement config (échec rapide si invalide).
    try:
        load_config(args.config)
    except Exception as exc:  # noqa: BLE001
        logger.error("Impossible de charger la config depuis %s: %s", args.config, exc)
        return 2
    if args.dry_run:
        logger.info("[DRY-RUN] Mode=%s, config=%s, generator=%s, seed=%d",
                    args.mode, args.config, args.generator, args.seed)
        return 0
    handler = HANDLERS[args.mode]
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
