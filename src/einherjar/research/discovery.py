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
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from einherjar.research.config.loader import EinherjarConfig, load_config
from einherjar.research.utils.logging import configure_logging
from einherjar.research.data.features import FeaturesFrame
from einherjar.research.data.ohlcv import OhlcvFrame

logger = logging.getLogger(__name__)

# --- Competition "arbres d'abord" (decision utilisateur 2026-08-09) ---
# Les generateurs de recherche aleatoire (RandomSearch) et l'expansion par
# grammaire (GrammaticalEvolution) sortent de la competition par defaut :
# les random gagnent uniquement par volume de points, sans construire de
# structure exploitable ; GE score tres faible (0.0029) avec branches mortes.
_DEFAULT_EXCLUDED_GENERATORS = (
    "RandomSearchGenerator",
    "GrammaticalEvolutionGenerator",
)


def _filter_competition_generators(generators, args: argparse.Namespace):
    """Retire de la competition les generateurs ecartes par defaut (arbres
    d'abord), sauf si l'utilisateur les re-inclut explicitement via
    --generator (alias simple) ou --generators (liste, prioritaire)."""
    if getattr(args, "generators", None):
        wanted = {g.strip() for g in args.generators.split(",") if g.strip()}
        return [g for g in generators if g.__class__.__name__ in wanted]
    if getattr(args, "generator", None):
        # alias existant : laisse passer (l'utilisateur demande explicitement)
        return generators
    return [g for g in generators if g.__class__.__name__ not in _DEFAULT_EXCLUDED_GENERATORS]


def _compare_report_path(args: argparse.Namespace) -> Path:
    asset = getattr(args, "data_asset", None) or "BTCUSD"
    tf = getattr(args, "data_timeframe", None) or "1h"
    return Path("outputs") / f"compare_report_{asset}_{tf}.json"


def _load_compare_report(args: argparse.Namespace):
    """Retourne le dict du rapport de comparaison si celui-ci correspond au
    run courant (asset/TF/seed/n_eval/max_conditions), sinon None."""
    path = _compare_report_path(args)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Rapport de comparaison illisible (%s) : ignore.", exc)
        return None
    meta = payload.get("meta", {})
    want = {
        "asset": getattr(args, "data_asset", None) or "BTCUSD",
        "timeframe": getattr(args, "data_timeframe", None) or "1h",
        "seed": getattr(args, "seed", None),
        "n_eval": getattr(args, "n_eval", None),
        "max_conditions": getattr(args, "max_conditions", None),
    }
    if not all(meta.get(k) == want.get(k) for k in ("asset", "timeframe", "seed", "n_eval", "max_conditions")):
        logger.info("Rapport de comparaison present mais stale pour ce run : ignore.")
        return None
    return payload["report"]


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
        "--config", type=Path, default=Path(__file__).resolve().parent / "config",
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
        "--generators", type=str, default=None,
        help="Liste de générateurs pour la comparaison (noms de classes, séparés par "
             "des virgules, ex: TypedGPGenerator,NSGA2Generator). Par défaut, seuls "
             "les générateurs d'arbres participent (random et GE exclus).",
    )
    parser.add_argument(
        "--max-conditions", type=int, default=4,
        help="Profondeur max des conditions pour les générateurs d'arbres (défaut: 4).",
    )
    parser.add_argument(
        "--n-eval", type=int, default=None,
        help="Budget d'évaluations moteur — plafond de COÛT global reparti équitablement "
             "entre generateurs (utilisé par compare et baselines).",
    )
    parser.add_argument(
        "--n-candidates", type=int, default=None,
        help="Volume de GÉNÉRATION : nombre d'hypothèses produites par moteur "
             "(défaut 100k). Indépendant du budget d'évaluation : l'évaluation "
             "et l'admission resserrent l'étau ensuite.",
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
        "--data-assets", type=str, default=None,
        help="Liste d'actifs separes par des virgules; requise en mode multi-actifs.",
    )
    parser.add_argument(
        "--all-assets", action="store_true",
        help="Lance sur la selection complete de 28 actifs depuis config/assets_v1.json. "
             "Le pipeline est execute par classe (5 classes). Le corpus s'accumule.",
    )
    parser.add_argument(
        "--all-timeframes", action="store_true",
        help="Avec --all-assets : itere aussi sur tous les timeframes presents dans "
             "le data_root pour chaque classe (genere 7 classes x N TF runs). "
             "Sans ce flag : le timeframe est fixe par --data-timeframe (defaut 1h).",
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
    parser.add_argument(
        "--hypothesis-file", type=Path, default=None,
        help="Fichier JSON avec une Hypothesis sérialisée (utilisé par 'holdout').",
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
    OhlcvFrame, FeaturesFrame, OhlcvFrame, FeaturesFrame, OhlcvFrame, FeaturesFrame
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
    from einherjar.research.data.npy_real_loader import load_features_from_npy
    from einherjar.research.data.ohlcv import OhlcvProvider
    from einherjar.research.data.validation import validate_or_raise
    from einherjar.research.data.versioning import make_frame_data_version, make_splits_hash

    root = Path(data_root)
    # MIDAS X.npy is normalized feature data. It must never provide execution
    # prices for ATR, SL/TP or trade returns.
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
            f"OHLCV/features desynchronisés : OHLCV={full_ohlcv.n_bougies} "
            f"!= features={full_features.n_bougies}"
        )

    n = full_ohlcv.n_bougies
    ratios = config.splits["ratios"]
    train_boundary = int(n * float(ratios["train"]))
    val_boundary = train_boundary + int(n * float(ratios["val"]))
    # Fixed conservative purge: every hypothesis has N <= max_n, so no signal
    # can use a future bar across a split boundary.
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

    # (fix) splits_hash : hash des BORNES RÉELLES des splits (purge/embargo
    # inclus), pas de la data_version. Deux fenêtres différentes sur la même
    # data produisent deux splits_hash différents — c'est ce qui trace la
    # portion temporelle exacte utilisée pour chaque Einher du corpus.
    splits_key = make_splits_hash(
        train_start=0,
        train_end=train_end,
        val_start=val_start,
        val_end=val_end,
        holdout_start=holdout_start,
        holdout_end=n,
        embargo_bougies=embargo,
        horizon_label=purge,
    )
    return (
        _slice_ohlcv(0, train_end), _slice_features(0, train_end),
        _slice_ohlcv(val_start, val_end), _slice_features(val_start, val_end),
        _slice_ohlcv(holdout_start, n), _slice_features(holdout_start, n),
        splits_key,
    )


def _load_real_data_multi(
    config: EinherjarConfig,
    data_root: str,
    assets: tuple[str, ...],
    asset_class: str,
    timeframe: str,
) -> dict[str, tuple]:
    """Charge OHLCV + features pour PLUSIEURS actifs (P0-05).

    Pour chaque actif de la liste, appelle `_load_real_data` et stocke
    le resultat (train_ohlcv, train_features, val_ohlcv, val_features,
    holdout_ohlcv, holdout_features) dans un dict par nom d'actif.

    Si un actif echoue au chargement, on leve NpyRealLoaderError apres
    avoir tente tous les actifs (fail-fast explicite, pas de fallback
    silencieux — P0 #7).

    Args:
        config: Configuration chargee.
        data_root: racine des .npy MIDAS V3.
        assets: tuple de noms d'actifs (ex: ("BTCUSD", "ETHUSD")).
        asset_class: classe d'actifs (ex: "crypto").
        timeframe: granularite (ex: "1h").

    Returns:
        Dict {asset_name: (train_ohlcv, train_features, val_ohlcv,
        val_features, holdout_ohlcv, holdout_features)}.
    """
    if not assets:
        raise ValueError(
            "P0-05 : _load_real_data_multi requiert au moins 1 actif "
            "(utiliser _load_real_data pour le single-asset)"
        )
    results: dict[str, tuple] = {}
    errors: list[tuple[str, str]] = []
    for asset in assets:
        try:
            results[asset] = _load_real_data(
                config=config, data_root=data_root,
                asset=asset, asset_class=asset_class, timeframe=timeframe,
            )
        except Exception as exc:
            errors.append((asset, str(exc)))
    if errors:
        # Fail-fast : on leve une erreur explicite listant tous les echecs.
        details = "; ".join(f"{a}: {e}" for a, e in errors)
        raise RuntimeError(
            f"P0-05 : echec du chargement multi-actifs. {len(errors)}/{len(assets)} "
            f"actifs en erreur. Details: {details}"
        )
    return results


def _resolve_assets(args: argparse.Namespace) -> tuple[str, ...]:
    """Resoud la liste d'actifs a charger depuis les args CLI (P0-05).

    Priorite :
      1. --data-assets (liste separee par virgules) si fourni.
      2. --data-asset (single) sinon (en tuple d'1 element).

    Returns:
        Tuple de symboles d'actifs, jamais vide.
    """
    if args.data_assets:
        return tuple(a.strip() for a in args.data_assets.split(",") if a.strip())
    return (args.data_asset,)


def load_default_assets(
    config_dir: Optional[Path] = None,
) -> dict[str, list[str]]:
    """Charge la selection par defaut d'actifs depuis config/assets_v1.json.

    Recherche par ordre :
      1. config_dir/assets_v1.json (parametre explicite).
      2. <cwd>/config/assets_v1.json (racine du repo).
      3. <module_dir>/../../config/assets_v1.json (relatif au module).

    Returns:
        Dict {asset_class: [asset_symbols]}. 5 classes (crypto, forex,
        stocks_tech, stocks_value, stocks_growth, indices, commodities).
        28 actifs au total.
    """
    candidates: list[Path] = []
    if config_dir is not None:
        candidates.append(config_dir / "assets_v1.json")
    candidates.append(Path("config") / "assets_v1.json")
    candidates.append(
        Path(__file__).resolve().parent.parent.parent / "config" / "assets_v1.json"
    )
    assets_path: Path | None = None
    for c in candidates:
        if c.exists():
            assets_path = c
            break
    if assets_path is None:
        raise FileNotFoundError(
            f"Fichier de selection d'actifs introuvable. Chemins testes : "
            f"{[str(c) for c in candidates]}"
        )
    with assets_path.open("r", encoding="utf-8") as fp:
        data = json.loads(fp.read())
    result: dict[str, list[str]] = {}
    for entry in data.get("assets", []):
        cls = entry.get("class")
        asset = entry.get("asset")
        if cls and asset:
            result.setdefault(cls, []).append(asset)
    return result


def list_available_timeframes(
    data_root: Path | str,
    asset_class: str,
) -> tuple[str, ...]:
    """Liste les timeframes disponibles pour une classe dans le data_root MIDAS.

    Args:
        data_root: Racine des .npy MIDAS V3.
        asset_class: Classe d'actifs (ex: 'crypto').

    Returns:
        Tuple des timeframes (ex: ('5m', '15m', '1h', '4h', '1d')).
        Trie pour reproductibilite. Vide si pas de dossier.
    """
    root = Path(data_root)
    cls_dir = root / asset_class
    if not cls_dir.exists():
        return ()
    tfs: list[str] = []
    for entry in sorted(cls_dir.iterdir()):
        if entry.is_dir() and not entry.name.startswith("."):
            tfs.append(entry.name)
    return tuple(tfs)


def _load_for_handler(
    config: EinherjarConfig,
    args: argparse.Namespace,
) -> dict[str, tuple]:
    """Charge les donnees reelles selon --data-asset ou --data-assets (P0-05).

    Wrapper au-dessus de _load_real_data / _load_real_data_multi qui dispatch
    automatiquement sur le mode single ou multi.

    Returns:
        Dict {asset_name: (train_ohlcv, train_features, val_ohlcv, val_features,
        holdout_ohlcv, holdout_features)}.
    """
    assets = _resolve_assets(args)
    if len(assets) == 1:
        result = _load_real_data(
            config=config, data_root=args.data_root, asset=assets[0],
            asset_class=args.data_class, timeframe=args.data_timeframe,
        )
        return {assets[0]: result}
    return _load_real_data_multi(
        config=config, data_root=args.data_root, assets=assets,
        asset_class=args.data_class, timeframe=args.data_timeframe,
    )


def _primary_asset(loaded: dict[str, tuple]) -> str:
    """Retourne le nom de l'actif principal a utiliser pour l'engine.

    En mode multi-actifs, l'actif principal est le 1er charge (ordre stable).
    L'engine reste single-asset pour V1 ; les autres actifs sont utilises
    pour la diversite Jaccard vs corpus (P1-10) et la mediane cross-actifs
    (contrainte NSGA-II #4).
    """
    if not loaded:
        raise ValueError("Aucun actif charge dans le dict multi-actifs")
    return next(iter(loaded))


def _persist_data_version(
    config: EinherjarConfig,
    train_ohlcv: Any,
    train_features: Any,
    *,
    store_path: Path = Path("outputs/data_versions.jsonl"),
) -> Any:
    """Persiste le DataVersion courant via DataVersionStore (P0-03).

    Procedure :
      1. Reconstruit le DataVersion depuis les frames train (make_frame_data_version).
      2. Le persiste via verify_data_version_locked (append-only JSONL, fsync).
      3. Si le tag/hash existe deja : OK (lock).
      4. Sinon : append, retourne le DataVersion verrouille.

    Args:
        config: Config Einherjar.
        train_ohlcv/train_features: Frames train dont on veut figer la version.
        store_path: Chemin du store append-only (defaut: outputs/data_versions.jsonl).

    Returns:
        Le DataVersion verrouille (DataVersion).
    """
    from einherjar.research.data.versioning import (
        DataVersionStore,
        make_frame_data_version,
        verify_data_version_locked,
    )
    dv = make_frame_data_version(train_ohlcv, train_features, config)
    store = DataVersionStore(store_path)
    locked = verify_data_version_locked(dv, store)
    logger.info(
        "P0-03 : data_version verrouille = %s (hash=%s, store=%s)",
        locked.tag, locked.hash[:12], store_path,
    )
    return locked


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
    from einherjar.research.admission.baseline_gate import make_baseline_admission_fn
    from einherjar.research.baselines.runner import BaselineRunner
    from einherjar.research.engine.evaluator import EvaluationEngine
    try:
        loaded = _load_for_handler(config, args)
        primary = _primary_asset(loaded)
        if len(loaded) > 1:
            logger.info("P0-05 : multi-actifs charge (%d), actif principal = %s", len(loaded), primary)
        train_ohlcv, train_features, val_ohlcv, val_features, _, _, splits_hash = loaded[primary]
    except Exception as exc:  # noqa: BLE001
        logger.error("Impossible de charger les données réelles : %s", exc)
        return 2
    # P0-03 : persistance verrouillable du data_version (avant l'engine).
    _persist_data_version(config, train_ohlcv, train_features)
    engine = EvaluationEngine(
        config=config, data_version=args.data_version or train_ohlcv.data_version, seed=args.seed,
    )
    # Admission RÉELLE (7 critères S-3.4), pas un fallback "tout admis".
    admission_fn, counter = make_baseline_admission_fn(config)
    # Budget moteur : part = n_eval // n_baselines (P1-08, meme philosophie que le comparateur).
    budget = args.n_eval if getattr(args, "n_eval", None) else None
    runner = BaselineRunner(engine=engine, eval_budget=budget)
    report = runner.run(
        train_ohlcv=train_ohlcv, train_features=train_features,
        val_ohlcv=val_ohlcv, val_features=val_features,
        admission_fn=admission_fn,
    )
    logger.info(
        "Baselines : %d essais, %d admis (DSR corrige pour multiple-testing)",
        counter["n"], counter["n_admitted"],
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
    from einherjar.research.admission.baseline_gate import make_baseline_admission_fn
    engine = EvaluationEngine(config=config, data_version=args.data_version or "v1", seed=args.seed)
    # Charge les données AVANT les générateurs : les générateurs évolutionnaires
    # (NSGA-II, Memetic) ont besoin d'accéder aux données pour évaluer leur fitness.
    try:
        loaded = _load_for_handler(config, args)
        primary = _primary_asset(loaded)
        if len(loaded) > 1:
            logger.info(
                "P0-05 : mode multi-actifs (%d actifs charges), actif principal = %s. "
                "Les autres actifs serviront a la mediane cross-actifs (NSGA-II #4) "
                "et a la diversite Jaccard vs corpus (P1-10).",
                len(loaded), primary,
            )
        train_ohlcv, train_features, val_ohlcv, val_features, _, _, splits_hash = loaded[primary]
    except Exception as exc:  # noqa: BLE001
        logger.error("Impossible de charger les données réelles : %s", exc)
        return 2
    data_version = args.data_version or train_ohlcv.data_version
    engine = EvaluationEngine(config=config, data_version=data_version, seed=args.seed)
    # P0-03 : persistance verrouillable du data_version (anti-faux sentiment
    # de reproductibilite). Append-only JSONL avec fsync.
    _persist_data_version(config, train_ohlcv, train_features)
    protocol = make_protocol(
        config, data_version=data_version,
        seed=args.seed,
        n_eval_budget=args.n_eval if args.n_eval is not None else 2_000,
        n_candidates=args.n_candidates if args.n_candidates is not None else 100_000,
        # BUG #4 : les artefacts portaient toujours universe.timeframes=['1h']
        # (defaut make_protocol) quel que soit le TF du run — l'etiquette du
        # run reel est portee par les donnees chargees, pas par le defaut.
        assets=tuple(loaded.keys()),
        timeframes=(train_ohlcv.timeframe,),
        max_conditions=getattr(args, "max_conditions", 4) or 4,
    )
    # Philosophie arbres d'abord (decision utilisateur 2026-08-09).
    all_generators = make_all_generators(protocol, config, engine=engine)
    all_generators = _filter_competition_generators(all_generators, args)
    if not all_generators:
        logger.error("Aucun générateur actif apres filtrage competition.")
        return 2
    # Injecte les données dans les générateurs qui en ont besoin (NSGA-II, Memetic).
    for gen in all_generators:
        gen.bind_data(train_ohlcv, train_features, val_ohlcv, val_features)
    generators = all_generators
    comparator = GeneratorComparator(generators=generators, protocol=protocol, engine=engine, config=config)
    admission_fn, _ = make_baseline_admission_fn(config)
    # P1-10 : passe le dict multi-actifs au comparator (active l'evaluation
    # cross-actifs pour NSGA-II). Si single-asset, on passe None.
    multi_for_compare = loaded if len(loaded) > 1 else None
    report = comparator.run(
        train_ohlcv=train_ohlcv, train_features=train_features,
        val_ohlcv=val_ohlcv, val_features=val_features,
        admission_fn=admission_fn,
        multi_assets=multi_for_compare,
    )
    logger.info("Comparaison terminée : winner=%s", report.winner_name)
    for r in report.rankings:
        logger.info("  #%d %s : score=%.4f, admission_rate=%.4f, median_sharpe=%.4f",
                    r.rank, r.generator_name, r.score, r.admission_rate, r.median_sharpe)
    # P1 : persiste le rapport pour que 'select' l'utilise sans relancer la
    # comparaison (~540 evaluations economisees par TF). Indexe par asset/TF.
    report_path = _compare_report_path(args)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps({
            "meta": {
                "asset": args.data_asset,
                "timeframe": args.data_timeframe,
                "seed": args.seed,
                "n_eval": args.n_eval,
            },
            "report": report.to_dict(),
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Rapport de comparaison persisté : %s", report_path)
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
    # P1 : si 'compare' a déjà tourné pour ce run (même asset/TF/seed/n_eval),
    # on consomme son rapport au lieu de relancer la comparaison.
    from einherjar.research.generators.comparator import ComparisonReport
    cached = _load_compare_report(args)
    if cached is not None:
        config = load_config(args.config)
        try:
            loaded = _load_for_handler(config, args)
            primary = _primary_asset(loaded)
            train_ohlcv = loaded[primary][0]
        except Exception as exc:  # noqa: BLE001
            logger.error("Impossible de charger les données réelles : %s", exc)
            return 2
        data_version = args.data_version or train_ohlcv.data_version
        report = ComparisonReport.from_dict(cached)
        from einherjar.research.selection.selector import GeneratorSelector
        selector = GeneratorSelector(protocol=report.protocol)
        selected = selector.select(report)
        selector.save(selected, args.selection_path)
        logger.info("Sélection (sans re-comparer) : %s (rank=%d, score=%.4f)",
                    selected.generator_name, selected.rank, selected.score)
        return 0
    config = load_config(args.config)
    from einherjar.research.engine.evaluator import EvaluationEngine
    from einherjar.research.generators.algorithms import make_all_generators
    from einherjar.research.generators.comparator import GeneratorComparator
    from einherjar.research.generators.protocol import make_protocol
    from einherjar.research.selection.selector import GeneratorSelector
    from einherjar.research.admission.baseline_gate import make_baseline_admission_fn
    try:
        loaded = _load_for_handler(config, args)
        primary = _primary_asset(loaded)
        if len(loaded) > 1:
            logger.info("P0-05 : multi-actifs charge (%d), actif principal = %s", len(loaded), primary)
        train_ohlcv, train_features, val_ohlcv, val_features, _, _, splits_hash = loaded[primary]
    except Exception as exc:  # noqa: BLE001
        logger.error("Impossible de charger les données réelles : %s", exc)
        return 2
    # P0-03 : persistance verrouillable du data_version.
    _persist_data_version(config, train_ohlcv, train_features)
    data_version = args.data_version or train_ohlcv.data_version
    engine = EvaluationEngine(config=config, data_version=data_version, seed=args.seed)
    protocol = make_protocol(
        config, data_version=data_version, seed=args.seed,
        n_eval_budget=args.n_eval if args.n_eval is not None else 2_000,
        n_candidates=args.n_candidates if args.n_candidates is not None else 100_000,
        assets=tuple(loaded.keys()),
        timeframes=(train_ohlcv.timeframe,),
        max_conditions=getattr(args, "max_conditions", 4) or 4,
    )
    generators = _filter_competition_generators(
        make_all_generators(protocol, config, engine=engine), args)
    for generator in generators:
        generator.bind_data(train_ohlcv, train_features, val_ohlcv, val_features)
    comparator = GeneratorComparator(generators=generators, protocol=protocol, engine=engine, config=config)
    admission_fn, _ = make_baseline_admission_fn(config)
    report = comparator.run(
        train_ohlcv=train_ohlcv, train_features=train_features,
        val_ohlcv=val_ohlcv, val_features=val_features,
        admission_fn=admission_fn,
    )
    selector = GeneratorSelector(protocol=protocol)
    selected = selector.select(report)
    selector.save(selected, args.selection_path)
    logger.info("Sélection : %s (rank=%d, score=%.4f)", selected.generator_name, selected.rank, selected.score)
    return 0


def handle_refine(args: argparse.Namespace) -> int:
    """Step 4 — Raffinement beam local des hypothèses du générateur sélectionné.

    Pour V1 : consomme la sélection produite par 'select', génère N
    hypothèses via le générateur sélectionné, calibre+test_on(val) sur
    chaque, garde le top M (par Sharpe val), applique BeamRefiner sur
    chacun, persiste les meilleurs dans outputs/refined.json.

    Contrainte dure : SL/TP figés depuis le train (jamais recalibrés).
    """
    logger.info("[STEP 4] Raffinement beam local")
    if not args.selection_path.exists():
        logger.error("Selection absente : %s — lance d'abord 'discovery select'", args.selection_path)
        return 2
    config = load_config(args.config)
    from einherjar.research.engine.evaluator import EvaluationEngine
    from einherjar.research.refinement.beam import make_default_refiner
    from einherjar.research.selection.selector import GeneratorSelector
    selected = GeneratorSelector.load(args.selection_path)
    logger.info("Generateur selectionne : %s (rank=%d, score=%.4f)",
                selected.generator_name, selected.rank, selected.score)
    # Charge les données réelles.
    try:
        loaded = _load_for_handler(config, args)
        primary = _primary_asset(loaded)
        if len(loaded) > 1:
            logger.info("P0-05 : multi-actifs charge (%d), actif principal = %s", len(loaded), primary)
        train_ohlcv, train_features, val_ohlcv, val_features, _, _, splits_hash = loaded[primary]
    except Exception as exc:  # noqa: BLE001
        logger.error("Impossible de charger les données réelles : %s", exc)
        return 2
    # P0-03 : persistance verrouillable du data_version.
    _persist_data_version(config, train_ohlcv, train_features)
    # Instancie le générateur avec un budget limité.
    import dataclasses
    n_eval = args.n_eval or 20
    protocol = dataclasses.replace(
        selected.protocol, n_eval_budget=n_eval,
        assets=tuple(loaded.keys()),
        timeframes=(train_ohlcv.timeframe,),
        max_conditions=getattr(args, "max_conditions", 4) or 4,
    )
    engine = EvaluationEngine(
        config=config, data_version=args.data_version or train_ohlcv.data_version, seed=args.seed,
    )
    generator = GeneratorSelector.instantiate(selected, config, engine=engine)
    # Patche le protocol sur le générateur (instantiate a recréé avec l'ancien).
    generator.protocol = protocol
    generator.bind_data(train_ohlcv, train_features, val_ohlcv, val_features)
    result = generator.generate()
    logger.info("Génération OK : %d hypothèses en %.2fs", len(result.hypotheses), result.generation_time_s)
    # Plafond d'évaluation : on NE teste que `n_eval` hypothèses max (budget
    # moteur partagé, philosophie P1-08). Sans ce plafond, RandomSearch (n_candidates
    # = 100k par défaut) enverrait ~100k calibrations à la suite (~28 h pour le 1h).
    eval_budget = max(1, n_eval)
    hyps_a_evaluer = result.hypotheses[:eval_budget]
    if len(result.hypotheses) > eval_budget:
        logger.info(
            "Plafond raffinement : %d hypothèses générées -> %d évaluées (budget n_eval)",
            len(result.hypotheses), eval_budget,
        )
    # Calibre + test_on(val) sur chaque hypothèse du budget, garde le top M.
    n_top = min(5, len(hyps_a_evaluer))
    candidates: list[tuple[float, Any, Any, Any]] = []  # (sharpe_val, hyp, calibrated, m_val)
    for hyp in hyps_a_evaluer:
        try:
            calibrated = engine.train_calibrate(hyp, train_ohlcv, train_features)
            m_val = engine.test_on(hyp, val_ohlcv, val_features, calibrated, "val")
            sharpe = m_val.sharpe_net
            if sharpe == sharpe:  # not NaN
                candidates.append((sharpe, hyp, calibrated, m_val))
        except Exception as exc:  # noqa: BLE001
            logger.debug("Échec calibration/val pour %s : %s", hyp.id, exc)
    candidates.sort(key=lambda t: t[0], reverse=True)
    top = candidates[:n_top]
    logger.info("Top %d après val (Sharpe): %s",
                n_top, [f"{t[0]:.3f} ({t[1].id})" for t in top])
    if not top:
        logger.warning("Aucun candidat viable à raffiner.")
        return 0
    # BeamRefiner sur chaque top.
    refiner = make_default_refiner(config=config, engine=engine, seed=args.seed)
    refined_records: list[dict[str, Any]] = []
    for sharpe_orig, hyp, calibrated, m_val in top:
        rr = refiner.refine(
            hypothesis=hyp, calibrated=calibrated,
            train_ohlcv=train_ohlcv, train_features=train_features,
            val_ohlcv=val_ohlcv, val_features=val_features,
        )
        record = {
            "original_id": hyp.id,
            "original_sharpe_val": sharpe_orig,
            "improved": rr.improved,
            "best_sharpe_val": rr.best_sharpe_val,
            "n_evaluated": rr.n_evaluated,
            "n_iterations": rr.n_iterations,
            "best_hypothesis": rr.best_hypothesis.to_dict() if rr.best_hypothesis else None,
        }
        refined_records.append(record)
        logger.info("Refine %s : improved=%s, sharpe %.4f -> %.4f",
                    hyp.id, rr.improved, sharpe_orig, rr.best_sharpe_val)
    # Persiste dans outputs/refined.json.
    import json as _json
    out_path = Path("outputs") / "refined.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        _json.dumps({"refined": refined_records, "n_input": len(result.hypotheses),
                     "n_viable": len(candidates), "n_refined": len(refined_records)},
                    indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Raffinement terminé : %d/%d améliorés, persisté dans %s",
                sum(1 for r in refined_records if r["improved"]),
                len(refined_records), out_path)
    return 0


def handle_admit(args: argparse.Namespace) -> int:
    """Step 5 — Admission au corpus.

    Pour V1 : consomme la sélection produite par 'select', génère N
    hypothèses via le générateur sélectionné, calibre+test_on(val) sur
    chaque, applique AdmissionDecider (DSR + PBO + bootstrap CI + n_trades
    + cross_asset + max_dd + diversité + dédup + quota).

    Les rejets sont ajoutés aux archives et les admis sont persistés dans
    le corpus avec leur hypothèse, calibration, mesures et série de retours.

    Note : pour le pipeline bout-en-bout, l'admission est aussi appliquée
    dans 'baselines' via baseline_gate.make_baseline_admission_fn (volet
    critères uniquement, sans archive). Ce mode 'admit' applique
    AdmissionDecider COMPLET (avec archive).
    """
    logger.info("[STEP 5] Admission au corpus")
    if not args.selection_path.exists():
        logger.error("Selection absente : %s — lance d'abord 'discovery select'", args.selection_path)
        return 2
    config = load_config(args.config)
    from einherjar.research.admission.decision import AdmissionDecider
    from einherjar.research.engine.evaluator import EvaluationEngine
    from einherjar.research.selection.selector import GeneratorSelector
    selected = GeneratorSelector.load(args.selection_path)
    logger.info("Generateur selectionne : %s (rank=%d, score=%.4f)",
                selected.generator_name, selected.rank, selected.score)
    try:
        loaded = _load_for_handler(config, args)
        primary = _primary_asset(loaded)
        if len(loaded) > 1:
            logger.info("P0-05 : multi-actifs charge (%d), actif principal = %s", len(loaded), primary)
        train_ohlcv, train_features, val_ohlcv, val_features, _, _, splits_hash = loaded[primary]
    except Exception as exc:  # noqa: BLE001
        logger.error("Impossible de charger les données réelles : %s", exc)
        return 2
    # P0-03 : persistance verrouillable du data_version.
    _persist_data_version(config, train_ohlcv, train_features)
    import dataclasses
    n_eval = args.n_eval or 20
    protocol = dataclasses.replace(
        selected.protocol, n_eval_budget=n_eval,
        assets=tuple(loaded.keys()),
        timeframes=(train_ohlcv.timeframe,),
        max_conditions=getattr(args, "max_conditions", 4) or 4,
    )
    engine = EvaluationEngine(
        config=config, data_version=args.data_version or train_ohlcv.data_version, seed=args.seed,
    )
    generator = GeneratorSelector.instantiate(selected, config, engine=engine)
    generator.protocol = protocol
    generator.bind_data(train_ohlcv, train_features, val_ohlcv, val_features)
    result = generator.generate()
    logger.info("Génération OK : %d hypothèses en %.2fs", len(result.hypotheses), result.generation_time_s)
    # Plafond (P1-08, même philosophie que baselines/refine) : on ne construit
    # la matrice CPCV que sur `n_eval` hypothèses max. Sans cela, RandomSearch
    # (n_candidates=100k) enverrait 100k calibrations à ~2,7 s chacune (≈75 h).
    eval_budget = max(1, n_eval)
    hyps_a_evaluer = result.hypotheses[:eval_budget]
    if len(result.hypotheses) > eval_budget:
        logger.info(
            "Plafond admission : %d hypothèses générées -> %d évaluées (budget n_eval)",
            len(result.hypotheses), eval_budget,
        )
    data_version = args.data_version or train_ohlcv.data_version
    engine = EvaluationEngine(config=config, data_version=data_version, seed=args.seed)
    decider = AdmissionDecider(config=config, data_version=data_version, seed=args.seed)
    from einherjar.research.corpus.store import CorpusEntry, CorpusStore
    corpus = CorpusStore()
    corpus_entries = corpus.load()
    # CPCV/PBO requiert la matrice de tous les candidats testés sur la même
    # validation. On l'évalue avant toute admission, sans conditionner le holdout.
    evaluated_candidates: dict[str, tuple[Any, Any]] = {}
    for hyp in hyps_a_evaluer:
        try:
            calibrated = engine.train_calibrate(hyp, train_ohlcv, train_features)
            measures = engine.test_on(hyp, val_ohlcv, val_features, calibrated, "val")
            evaluated_candidates[hyp.id] = (calibrated, measures)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Candidat exclu de la matrice CPCV %s : %s", hyp.id, exc)
    pbo_candidate_paths = [
        tuple((trade.entry_idx, trade.exit_idx, trade.ret_pct_net) for trade in measures.trades)
        for _, measures in evaluated_candidates.values()
    ]
    n_admitted = 0
    n_rejected = 0
    reasons: dict[str, int] = {}
    for i, hyp in enumerate(hyps_a_evaluer, start=1):
        evaluation = evaluated_candidates.get(hyp.id)
        if evaluation is None:
            continue
        calibrated, m_val = evaluation
        returns_val = [t.ret_pct_net for t in m_val.trades]
        # (fix refactor) Quotas I-8 : les fractions du corpus sont calculées à
        # chaque itération (le corpus s'enrichit au fil des admissions). Avant
        # ce branchement, current_corpus_fracs n'était JAMAIS fourni → les
        # quotas structurels n'étaient jamais appliqués.
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
            n_indep_trials=i,
            corpus_fingerprints=[
                (e.fingerprint_structurel, e.fingerprint_comportemental)
                for e in corpus_entries
            ],
        )
        if decision.admitted:
            n_admitted += 1
            verdicts = {v.name: v.observed for v in decision.criteria_verdict.verdicts}
            corpus_entry = CorpusEntry(
                id=hyp.id,
                hypothesis=hyp.to_dict(),
                direction=hyp.direction.value,
                universe={"assets": hyp.universe.assets, "timeframes": hyp.universe.timeframes},
                amplitude=hyp.amplitude.to_dict(),
                sl_n_atr=calibrated.sl_n_atr, tp_n_atr=calibrated.tp_n_atr,
                sl_distance=calibrated.sl_distance, tp_distance=calibrated.tp_distance,
                n_window=calibrated.n_window,
                fingerprint_structurel=decision.meta["fp_struct"],
                fingerprint_comportemental=decision.meta["fp_comport"],
                metrics_val=m_val.to_dict(), sharpe_val=m_val.sharpe_net,
                bootstrap_sharpe_ci_low_val=m_val.bootstrap_sharpe_ci_low,
                bootstrap_sharpe_ci_high_val=m_val.bootstrap_sharpe_ci_high,
                deflated_sharpe_ratio=float(verdicts.get("DSR", float("nan"))),
                probability_of_backtest_overfitting=float(verdicts.get("PBO", float("nan"))),
                ret_series=tuple(returns_val),
                data_version=train_ohlcv.data_version, seed=args.seed,
                splits_hash=splits_hash,
                admission_timestamp=CorpusEntry.now_utc(),
                meta={"signal_indices": [t.entry_idx - 1 for t in m_val.trades]},
            )
            corpus.append(corpus_entry)
            corpus_entries.append(corpus_entry)
        else:
            n_rejected += 1
            reason = decision.primary_reason.value if decision.primary_reason else "OTHER"
            reasons[reason] = reasons.get(reason, 0) + 1
    logger.info("Admission terminée : %d/%d admis", n_admitted, n_admitted + n_rejected)
    if reasons:
        logger.info("Breakdown rejets : %s", reasons)
    # Persiste un résumé.
    import json as _json
    out_path = Path("outputs") / "admit_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        _json.dumps({
            "n_generated": len(result.hypotheses),
            "n_admitted": n_admitted,
            "n_rejected": n_rejected,
            "rejection_breakdown": reasons,
            "generator": selected.generator_name,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Résumé persisté dans %s", out_path)
    return 0


def handle_holdout(args: argparse.Namespace) -> int:
    """Step 6 — Évaluation finale unique sur le holdout (sacré).

    Le fichier fourni doit correspondre exactement à un Einher admis du
    corpus ; un JSON arbitraire ne peut pas ouvrir le holdout.

    Le holdout est consulté UNE SEULE FOIS par session : le 2e appel
    lèvera une erreur (cf. HoldoutEvaluator._holdout_used).
    """
    logger.info("[STEP 6] Holdout sacré — ÉVALUATION FINALE UNIQUE")
    if args.hypothesis_file is None:
        logger.error(
            "Aucun --hypothesis-file fourni. Pour V1, le holdout consomme un fichier JSON "
            "contenant une Hypothesis sérialisée (via Hypothesis.to_dict()). "
            "Le pipeline P0 #4 V2 (corpus persisté → holdout automatique) n'est pas encore implémenté."
        )
        return 2
    if not args.hypothesis_file.exists():
        logger.error("Fichier hypothesis introuvable : %s", args.hypothesis_file)
        return 2
    # Charge l'Hypothesis depuis le JSON.
    import json
    from einherjar.research.engine.evaluator import EvaluationEngine
    from einherjar.research.holdout.evaluator import HoldoutEvaluator
    from einherjar.research.utils.types import Hypothesis
    try:
        hyp_dict = json.loads(args.hypothesis_file.read_text(encoding="utf-8"))
        hypothesis = Hypothesis.from_dict(hyp_dict)
    except Exception as exc:  # noqa: BLE001
        logger.error("Hypothesis JSON invalide : %s", exc)
        return 2
    from einherjar.research.corpus.store import CorpusStore
    admitted = {entry.id: entry for entry in CorpusStore().load()}
    corpus_entry = admitted.get(hypothesis.id)
    if corpus_entry is None or corpus_entry.hypothesis != hypothesis.to_dict():
        logger.error(
            "Holdout refusÃ© : l'hypothÃ¨se doit Ãªtre un Einher admis, inchangÃ©, du corpus."
        )
        return 2
    # Charge les données + split holdout.
    config = load_config(args.config)
    try:
        loaded = _load_for_handler(config, args)
        primary = _primary_asset(loaded)
        if len(loaded) > 1:
            logger.info("P0-05 : multi-actifs charge (%d), actif principal = %s", len(loaded), primary)
        train_ohlcv, train_features, val_ohlcv, val_features, holdout_ohlcv, holdout_features, splits_hash = loaded[primary]
    except Exception as exc:  # noqa: BLE001
        logger.error("Impossible de charger les données réelles : %s", exc)
        return 2
    # Pipeline complet : train_calibrate + test_on(val) + HoldoutEvaluator.
    # P0-03 : persistance verrouillable du data_version.
    _persist_data_version(config, train_ohlcv, train_features)
    data_version = args.data_version or train_ohlcv.data_version
    engine = EvaluationEngine(config=config, data_version=data_version, seed=args.seed)
    try:
        calibrated = engine.train_calibrate(hypothesis, train_ohlcv, train_features)
        m_val = engine.test_on(hypothesis, val_ohlcv, val_features, calibrated, "val")
    except Exception as exc:  # noqa: BLE001
        logger.error("Échec train/val pour %s : %s", hypothesis.id, exc)
        return 2
    holdout_eval = HoldoutEvaluator(
        engine=engine, config=config,
        data_version=data_version, seed=args.seed,
    )
    val_sharpe = m_val.sharpe_net
    val_snapshot = m_val.to_dict()
    result = holdout_eval.evaluate(
        hypothesis=hypothesis, calibrated=calibrated,
        holdout_ohlcv=holdout_ohlcv, holdout_features=holdout_features,
        val_sharpe=val_sharpe, val_metrics_snapshot=val_snapshot,
    )
    # Persiste le résultat dans outputs/.
    import json as _json
    out_path = Path("outputs") / "holdout_result.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_json.dumps(result.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Holdout terminé : flag=%s, sharpe_holdout=%.4f, persisté dans %s",
                result.degradation_flag, result.metrics_holdout.sharpe_net, out_path)
    return 0


def handle_run(args: argparse.Namespace) -> int:
    """Pipeline complet (étapes 0→5). Le holdout reste manuel.

    Si --all-assets est active, itere sur les 5 classes de la selection
    par defaut (28 actifs) et accumule les Einhers admis dans le meme
    corpus (data/corpus.jsonl, append-only). Le bilan par classe est
    persiste dans outputs/discover_report.json.
    """
    if getattr(args, "all_assets", False):
        logger.info("[PIPELINE ALL] Mode full system sur 28 actifs (5 classes)")
        config = load_config(args.config)
        per_class = load_default_assets(args.config)
        report: dict[str, Any] = {
            "started_at": datetime.now().isoformat(),
            "n_classes": len(per_class),
            "n_assets_total": sum(len(v) for v in per_class.values()),
            "classes": {},
        }
        # Liste les timeframes si --all-timeframes.
        all_tfs = (
            list_available_timeframes(args.data_root, "__any__")
            if getattr(args, "all_timeframes", False)
            else ()
        )
        for cls, assets in sorted(per_class.items()):
            # Pour cette classe, liste les TF si all_timeframes.
            tfs_for_cls: tuple[str, ...] = (
                list_available_timeframes(args.data_root, cls)
                if getattr(args, "all_timeframes", False)
                else ()
            )
            if not tfs_for_cls:
                tfs_for_cls = (args.data_timeframe,)
            for tf in tfs_for_cls:
                logger.info(
                    "[PIPELINE ALL] === Classe %s / TF %s (%d actifs : %s) ===",
                    cls, tf, len(assets), ", ".join(assets),
                )
                sub_args = argparse.Namespace(**vars(args))
                sub_args.data_class = cls
                sub_args.data_timeframe = tf
                sub_args.data_assets = ",".join(assets)
                sub_args.data_asset = assets[0]  # actif principal = 1er
                sub_args.all_assets = False  # evite recursion infinie
                sub_args.all_timeframes = False
                class_tf_rc = 0
                class_tf_stats: dict[str, int] = {}
                for mode in ("engine", "baselines", "compare", "select", "refine", "admit"):
                    sub_args.mode = mode
                    rc = HANDLERS[mode](sub_args)
                    class_tf_stats[mode] = rc
                    if rc != 0:
                        logger.error(
                            "Classe %s / TF %s, etape %s en erreur (rc=%d)",
                            cls, tf, mode, rc,
                        )
                        class_tf_rc = rc
                        break
                # Bilan par (classe, tf) : on garde une entree par TF.
                report["classes"].setdefault(cls, {}).setdefault("timeframes", {})
                report["classes"][cls].setdefault("n_assets", len(assets))
                report["classes"][cls].setdefault("assets", list(assets))
                report["classes"][cls]["timeframes"][tf] = {
                    "rc": class_tf_rc,
                    "step_rcs": class_tf_stats,
                    "status": "ok" if class_tf_rc == 0 else "error",
                }
        report["finished_at"] = datetime.now().isoformat()
        # Bilan global : on compte chaque (classe, tf).
        n_ok = 0
        n_total = 0
        for cls_data in report["classes"].values():
            for tf_data in cls_data.get("timeframes", {}).values():
                n_total += 1
                if tf_data["rc"] == 0:
                    n_ok += 1
        report["summary"] = (
            f"{n_ok}/{n_total} runs (classe x TF) OK, "
            f"corpus = data/corpus.jsonl"
        )
        # Persistance du rapport.
        out_path = Path("outputs") / "discover_report.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info(
            "[PIPELINE ALL] Termine : %s. Rapport dans %s.",
            report["summary"], out_path,
        )
        return 0 if n_ok == n_total and n_total > 0 else 1
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
