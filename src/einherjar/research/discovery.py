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

from einherjar.research.config.loader import EinherjarConfig, load_config
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
        "--data-root", type=str,
        default=r"D:\midas_v2\midasV3\src\data\compiled",
        help="Racine des datasets .npy (défaut: MIDAS V3 compiled).",
    )
    parser.add_argument(
        "--data-asset", type=str, default="BTCUSD",
        help="Symbole de l'actif (défaut: BTCUSD).",
    )
    parser.add_argument(
        "--data-class", type=str, default="crypto",
        choices=("crypto", "forex", "indices", "commodities",
                 "stocks_growth", "stocks_tech", "stocks_value"),
        help="Classe d'actifs (défaut: crypto).",
    )
    parser.add_argument(
        "--data-timeframe", type=str, default="1h",
        help="Timeframe (défaut: 1h).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Affiche le plan d'exécution sans rien lancer.",
    )
    return parser


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _load_real_data(
    config: EinherjarConfig,
    data_root: str,
    asset: str,
    asset_class: str,
    timeframe: str,
) -> tuple[
    "OhlcvFrame", "FeaturesFrame",
    "OhlcvFrame", "FeaturesFrame",
    "OhlcvFrame", "FeaturesFrame",
]:
    """Charge OHLCV + features depuis le .npy et découpe en train/val/holdout.

    Le loader `load_ohlcv_from_npy` retourne DÉJÀ un (OhlcvFrame, validity_mask).
    On passe ce mask à `load_features_from_npy` pour aligner les features sur
    la même série sanitizée (évite la désynchro OHLCV=3260 vs features=69708).

    Le split 60/20/20 suit splits.yaml (mode temporel strict, pas de shuffle).

    Returns:
        (train_ohlcv, train_features, val_ohlcv, val_features, holdout_ohlcv, holdout_features).
        Lève NpyRealLoaderError si les données sont absentes / invalides.
    """
    from einherjar.research.data.features import FeaturesFrame
    from einherjar.research.data.npy_real_loader import (
        load_features_from_npy,
        load_ohlcv_from_npy,
    )
    from einherjar.research.data.ohlcv import OhlcvFrame

    root = Path(data_root)
    full_ohlcv, mask = load_ohlcv_from_npy(
        asset=asset, asset_class=asset_class, timeframe=timeframe, data_root=root,
    )
    full_features = load_features_from_npy(
        asset=asset, asset_class=asset_class, timeframe=timeframe,
        config=config, data_root=root, validity_mask=mask,
    )
    if full_ohlcv.n_bougies != full_features.n_bougies:
        raise RuntimeError(
            f"OHLCV/features desynchronisés : OHLCV={full_ohlcv.n_bougies} "
            f"!= features={full_features.n_bougies}"
        )

    n = full_ohlcv.n_bougies
    t1 = int(n * 0.60)
    t2 = int(n * 0.80)

    def _slice_ohlcv(start: int, end: int) -> OhlcvFrame:
        return OhlcvFrame(
            asset=full_ohlcv.asset, timeframe=full_ohlcv.timeframe,
            df=full_ohlcv.df.slice(start, end - start),
            data_version=full_ohlcv.data_version,
        )

    def _slice_features(start: int, end: int) -> FeaturesFrame:
        return FeaturesFrame(
            asset=full_features.asset, timeframe=full_features.timeframe,
            df=full_features.df.slice(start, end - start),
            feature_names=full_features.feature_names,
            data_version=full_features.data_version,
        )

    return (
        _slice_ohlcv(0, t1), _slice_features(0, t1),
        _slice_ohlcv(t1, t2), _slice_features(t1, t2),
        _slice_ohlcv(t2, n), _slice_features(t2, n),
    )


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
    try:
        train_ohlcv, train_features, val_ohlcv, val_features, _, _ = _load_real_data(
            config=config,
            data_root=args.data_root,
            asset=args.data_asset, asset_class=args.data_class, timeframe=args.data_timeframe,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Impossible de charger les données réelles : %s", exc)
        return 2
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
    try:
        train_ohlcv, train_features, val_ohlcv, val_features, _, _ = _load_real_data(
            config=config,
            data_root=args.data_root,
            asset=args.data_asset, asset_class=args.data_class, timeframe=args.data_timeframe,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Impossible de charger les données réelles : %s", exc)
        return 2
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
    try:
        train_ohlcv, train_features, val_ohlcv, val_features, _, _ = _load_real_data(
            config=config,
            data_root=args.data_root,
            asset=args.data_asset, asset_class=args.data_class, timeframe=args.data_timeframe,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Impossible de charger les données réelles : %s", exc)
        return 2
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
