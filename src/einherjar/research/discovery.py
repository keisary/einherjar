"""
discovery.py — Point d'entrée du moteur de découverte Einherjar.

Implémente les modes du pipeline 7 étapes défini dans ONTOLOGY.md (S-3)
et ALGORITHME_RESEARCH.md (§ 10.2) :

  Step 0  Moteur d'évaluation (priorité 0 — toujours exécuté en premier)
  Step 1  Baselines
  Step 2  Compétition reproductible des générateurs
  Step 3  Choix du générateur
  Step 4  Raffinement local
  Step 5  Admission au corpus
  Step 6  Holdout sacré (1 seule passe)

Usage :
  python -m einherjar.research.discovery <mode> [options]

Modes disponibles :
  engine       Construit / vérifie le moteur d'évaluation
  baselines    Lance les 3 baselines sur le val
  compare      Comparaison reproductible des générateurs (random/GE/GP/beam)
  select       Installe le générateur gagnant
  refine       Raffinement beam local
  admit        Admission au corpus (critères S-3.4 + diversité comportementale)
  holdout      Évaluation finale unique sur le holdout
  run          Pipeline complet (engine → baselines → compare → select → refine → admit → holdout)
  pipeline     Alias de `run` avec stages configurables

Philosophie : moteur d'évaluation d'abord, générateurs après, holdout à la fin.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from einherjar.research.config.loader import load_config
from einherjar.research.utils.logging import configure_logging

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Modes du pipeline
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


def build_parser() -> argparse.ArgumentParser:
    """Construit le parser CLI."""
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
        "--config",
        type=Path,
        default=Path("config"),
        help="Dossier contenant les fichiers de config (défaut: ./config).",
    )
    parser.add_argument(
        "--data-version",
        type=str,
        default=None,
        help="Tag/identifiant de la version de données à utiliser (override config).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed RNG maître (override config).",
    )
    parser.add_argument(
        "--generator",
        type=str,
        default=None,
        choices=("random", "ge", "stgp", "beam", "memetic", "nsga2"),
        help="Générateur à utiliser (utile pour 'select' / 'refine' / 'admit' / 'holdout').",
    )
    parser.add_argument(
        "--n-eval",
        type=int,
        default=None,
        help="Budget d'évaluations (utilisé par 'compare' et 'baselines').",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Niveau de log (défaut: INFO).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Affiche le plan d'exécution sans rien lancer.",
    )
    return parser


# --------------------------------------------------------------------------- #
# Handlers par mode
# --------------------------------------------------------------------------- #


def handle_engine(args: argparse.Namespace) -> int:
    """Step 0 — Construit / vérifie le moteur d'évaluation."""
    logger.info("[STEP 0] Moteur d'évaluation")
    logger.info("  → chargement config depuis %s", args.config)
    logger.info("  → vérification cohérence splits/costs/features_taxonomy")
    logger.info("  → construction simulateur intrabar TP/SL + block bootstrap CI")
    # Implémentation déléguée à engine/evaluator.py
    raise NotImplementedError("engine.evaluator pas encore implémenté — voir engine/")


def handle_baselines(args: argparse.Namespace) -> int:
    """Step 1 — Baselines honnêtes (human, shallow_enum, random_constrained)."""
    logger.info("[STEP 1] Baselines")
    logger.info("  → human_rules (1-3 conditions triviales)")
    logger.info("  → shallow_enum (énumération 1-2 conditions)")
    logger.info("  → random_constrained (tirage sous contraintes)")
    logger.info("  → budget: %s évaluations", args.n_eval or "config")
    # Implémentation déléguée à baselines/runner.py
    raise NotImplementedError("baselines.runner pas encore implémenté — voir baselines/")


def handle_compare(args: argparse.Namespace) -> int:
    """Step 2 — Comparaison reproductible des générateurs."""
    logger.info("[STEP 2] Compétition générateurs")
    logger.info("  → candidats: random, ge, stgp, beam (optionnel: memetic, nsga2)")
    logger.info("  → protocole: mêmes seeds/splits/budget/métriques/coûts")
    logger.info("  → critère de classement: taux d'admission × qualité médiane × diversité")
    # Implémentation déléguée à generators/comparator.py
    raise NotImplementedError("generators.comparator pas encore implémenté — voir generators/")


def handle_select(args: argparse.Namespace) -> int:
    """Step 3 — Installe le générateur gagnant du comparateur."""
    logger.info("[STEP 3] Sélection du générateur")
    if args.generator:
        logger.info("  → override CLI: %s", args.generator)
    else:
        logger.info("  → chargement du gagnant depuis le comparateur")
    # Implémentation déléguée à selection/selector.py
    raise NotImplementedError("selection.selector pas encore implémenté — voir selection/")


def handle_refine(args: argparse.Namespace) -> int:
    """Step 4 — Raffinement beam local (sans recalibrer SL/TP)."""
    logger.info("[STEP 4] Raffinement beam local")
    logger.info("  → beam search sur Einhers viables, 1-2 niveaux")
    logger.info("  → contrainte dure: SL/TP figés depuis le train (jamais recalibrés)")
    # Implémentation déléguée à refinement/beam.py
    raise NotImplementedError("refinement.beam pas encore implémenté — voir refinement/")


def handle_admit(args: argparse.Namespace) -> int:
    """Step 5 — Admission au corpus (critères S-3.4 + diversité comportementale)."""
    logger.info("[STEP 5] Admission au corpus")
    logger.info("  → critères: DSR, PBO, bootstrap CI, n_trades, cross_asset, max_dd, diversité")
    logger.info("  → fingerprint canonique (structurel + comportemental)")
    logger.info("  → déduplication contre l'Archive sur le même data_version")
    # Implémentation déléguée à admission/decision.py
    raise NotImplementedError("admission.decision pas encore implémenté — voir admission/")


def handle_holdout(args: argparse.Namespace) -> int:
    """Step 6 — Évaluation finale unique sur le holdout (sacré)."""
    logger.info("[STEP 6] Holdout sacré — ÉVALUATION FINALE UNIQUE")
    logger.info("  ⚠ ce mode ne doit être appelé qu'UNE SEULE FOIS par Einher final retenu")
    logger.info("  → publication: métriques + IC bootstrap + descripteurs comportementaux")
    logger.info("  → archivage avec data_version/seed/splits/coûts figés")
    # Implémentation déléguée à holdout/evaluator.py
    raise NotImplementedError("holdout.evaluator pas encore implémenté — voir holdout/")


def handle_run(args: argparse.Namespace) -> int:
    """Pipeline complet (7 étapes) — sauf holdout (à déclencher manuellement)."""
    logger.info("[PIPELINE] Exécution séquentielle des étapes 0 → 5")
    for mode in ("engine", "baselines", "compare", "select", "refine", "admit"):
        args.mode = mode
        handler = HANDLERS[mode]
        try:
            handler(args)
        except NotImplementedError as exc:
            logger.error("Étape %s non implémentée: %s", mode, exc)
            return 1
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
# Point d'entrée
# --------------------------------------------------------------------------- #


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    configure_logging(level=args.log_level)

    # Chargement config (échec rapide si config invalide)
    try:
        load_config(args.config)
    except Exception as exc:  # noqa: BLE001 — on veut le message exact
        logger.error("Impossible de charger la config depuis %s: %s", args.config, exc)
        return 2

    if args.dry_run:
        logger.info("[DRY-RUN] Mode=%s, config=%s, generator=%s", args.mode, args.config, args.generator)
        return 0

    handler = HANDLERS[args.mode]
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
