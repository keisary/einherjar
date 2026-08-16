"""scripts/signal_baseline_test.py — Test du signal (buy-and-hold) — Phase 2.

DIAGNOSTIC : y a-t-il du signal exploitable dans les données actuelles, et le
moteur technique (SL/TP court) est-il capable de le capturer ?

Méthode :
  1. Buy-and-hold BRUT sur la fenêtre de validation (prix de clôture)
       -> CAGR annuel et Sharpe annuel du MARCHÉ (référence).
  2. Buy-and-hold sur la fenêtre d'entraînement (même calcul).
  3. "Toujours long" et "toujours short" évalués par le VRAI moteur
     (SL/TP calibré) -> est-ce que le moteur technique capture le trend ?

Interprétation :
  - Si brut>>0 mais moteur<<0  -> le signal existe, l'évaluation technique
    (horizon court + SL/TP + coûts) ne le capture pas. Problème d'évaluation.
  - Si brut<<0 ou ~0           -> pas de trend net sur cette fenêtre.
  - Le script est purement diagnostique ; il ne touche pas au chemin d'exécution.

Usage:
  export PYTHONPATH=src
  python scripts/signal_baseline_test.py --asset BTCUSD --class crypto --tf 15m
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s :: %(message)s")
logger = logging.getLogger("signal_baseline")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


def _annualized_stats(log_returns: list[float], periods_per_year: float) -> dict:
    """CAGR et Sharpe annuel à partir de log-rendements.

    CAGR = exp(mean * ppy) - 1 ; Sharpe = (mean / std) * sqrt(ppy).
    Retourne {} si pas assez de données (std=0).
    """
    n = len(log_returns)
    if n < 2:
        return {}
    mean = sum(log_returns) / n
    var = sum((r - mean) ** 2 for r in log_returns) / (n - 1)
    std = math.sqrt(var)
    cagr = math.expm1(mean * periods_per_year)
    sharpe = (mean / std) * math.sqrt(periods_per_year) if std > 0 else float("nan")
    return {"cagr": cagr, "sharpe": sharpe, "n_periods": n}


def _buy_and_hold(close: list[float]) -> dict:
    """Buy-and-hold brut : tous les log-rendements 1-période de la série close."""
    logs = [math.log(close[i] / close[i - 1]) for i in range(1, len(close))
            if close[i - 1] and close[i] > 0]
    return {"total_return": (close[-1] / close[0] - 1) if close[0] else 0.0,
            "log_returns": logs}


def _always_true_condition(feature_name: str) -> "ConditionNode":
    """Construit une condition qui retourne True sur toutes les lignes.

    OR(feature >= -1e18, feature <= 1e18) couvre tous les réels ET les NaN
    (car fill_null(False) sur chaque branche, mais l'OR des deux branches
    couvre le cas où l'un est NaN=False et l'autre est True).
    Si la feature elle-même est NaN, les deux branches sont False → on
    utilise un OR à trois branches avec un NOT pour garantir True même alors.
    """
    from einherjar.research.utils.types import (
        CompareOp, Condition, ConditionNode, LogicalOp,
    )
    # OR(GT(-1e18), LE(1e18)) : couvre tous les réels.
    left = Condition(feature_ref=feature_name, operator=CompareOp.GT,
                    value=-1e18, transformation=None)
    right = Condition(feature_ref=feature_name, operator=CompareOp.LE,
                      value=1e18, transformation=None)
    return ConditionNode(op=LogicalOp.OR, left=left, right=right)


def _make_always_hypothesis(
    direction: "Direction", asset: str, timeframe: str,
    feature_name: str, cooldown_k: int = 1,
) -> "Hypothesis":
    """Crée une Hypothèse 'toujours vrai' dans la direction donnée."""
    from einherjar.research.utils.types import (
        Amplitude, AmplitudeUnit, Direction, Hypothesis, Universe,
    )
    tree = _always_true_condition(feature_name)
    return Hypothesis(
        id=f"always_{'long' if direction == Direction.LONG else 'short'}",
        condition_tree=tree,
        amplitude=Amplitude(valeur=5.0, unité=AmplitudeUnit.MULTIPLE_ATR,
                            direction_implicite=direction),
        direction=direction,
        universe=Universe(assets=(asset,), timeframes=(timeframe,)),
        cooldown_k=cooldown_k,
    )


def _run_engine_test(
    config, engine, train_ohlcv, train_features, val_ohlcv, val_features,
    direction, feature_name,
):
    """Évalue une hypothèse 'toujours {direction}' via le vrai moteur."""
    from einherjar.research.utils.types import Direction
    dir_obj = Direction.LONG if direction == "long" else Direction.SHORT
    hyp = _make_always_hypothesis(
        dir_obj, train_ohlcv.asset, train_ohlcv.timeframe, feature_name,
    )
    calibrated = engine.train_calibrate(hyp, train_ohlcv, train_features)
    measures = engine.test_on(
        hyp, val_ohlcv, val_features, calibrated, "val",
        with_bootstrap=False,
    )
    n_trades = getattr(measures, "n_signals", 0) or 0
    avg_hold = getattr(measures, "avg_holding_period", 0.0) or 0.0
    tp = getattr(measures, "tp_hit_rate", 0.0) or 0.0
    sharpe = getattr(measures, "sharpe_annualized", float("nan"))
    if sharpe != sharpe:
        sharpe = getattr(measures, "sharpe", float("nan"))
    ret_mean = getattr(measures, "ret_mean_pct_net", float("nan"))
    logger.info(
        "  Moteur [%-5s] : %5d trades | TP=%.1f%% | hold=%.1f bougies | "
        "ret_mean_net=%+.4f%% | Sharpe=%+.2f",
        direction.upper(), n_trades, tp * 100, avg_hold, ret_mean, sharpe,
    )
    return measures


def periods_per_year_for(tf: str) -> float:
    """Nb de périodes par an pour un timeframe donné (approximation 365j)."""
    per_day = {"5m": 288, "15m": 96, "1h": 24, "4h": 6, "1d": 1}.get(tf, 1)
    return per_day * 365


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Test du signal (buy-and-hold)")
    parser.add_argument("--asset", default="BTCUSD")
    parser.add_argument("--class", dest="asset_class", default="crypto")
    parser.add_argument("--tf", default="15m")
    parser.add_argument("--config", default=str(REPO_ROOT / "src/einherjar/research/config"))
    parser.add_argument("--data-root", default=r"D:\midas_v2\midasV3\src\data\compiled")
    args = parser.parse_args(argv)

    from einherjar.research.config.loader import load_config
    from einherjar.research.discovery import _load_real_data, _persist_data_version

    config = load_config(args.config)
    loaded = _load_real_data(config, args.data_root, args.asset, args.asset_class, args.tf)
    train_ohlcv, _, val_ohlcv, _, holdout_ohlcv, _, _ = loaded
    _persist_data_version(config, train_ohlcv, val_ohlcv)
    ppy = periods_per_year_for(args.tf)

    logger.info(
        "=== Diagramme de référence — %s %s × %s ===", args.asset, args.asset_class, args.tf
    )

    for label, frame in (("TRAIN", train_ohlcv), ("VAL", val_ohlcv),
                         ("HOLDOUT", holdout_ohlcv)):
        close = frame.df["close"].to_list()
        bh = _buy_and_hold(close)
        stats = _annualized_stats(bh["log_returns"], ppy)
        ret = bh["total_return"]
        cagr = stats.get("cagr", float("nan"))
        sharpe = stats.get("sharpe", float("nan"))
        logger.info(
            "Buy-and-hold BRUT [%-7s] %6d bougies | rendement total=%+.2f%% | CAGR=%+.1f%% | Sharpe annuel=%+.2f",
            label, frame.n_bougies, ret * 100, cagr * 100, sharpe,
        )

    logger.info("")
    logger.info("Interprétation :")
    logger.info("  - Brut VAL > 0 significatif  => du trend/momentum existe dans les données.")
    logger.info("  - Si le moteur STGP sort tout négatif alors que le brut VAL est positif,")
    logger.info("    le signal existe mais l'évaluation technique (SL/TP court) ne le capture pas.")

    # --- Phase 2 : test via le VRAI moteur (SL/TP calibré) ---
    logger.info("")
    logger.info("=== Test moteur technique (SL/TP calibré) ===")
    logger.info("Hypothèses 'toujours long' et 'toujours short' (condition toujours vraie)")
    logger.info("→ Si le moteur détruit le signal long alors que le brut est positif,")
    logger.info("  la faute est dans l'évaluation (SL/TP court), pas dans la génération.")
    logger.info("")

    from einherjar.research.engine.evaluator import EvaluationEngine
    engine = EvaluationEngine(config=config, data_version=train_ohlcv.data_version, seed=42)

    # Trouve une feature continue pour la condition always-true.
    feature_name = "adx_14"  # feature continue standard, toujours présente
    if feature_name not in config.usable_feature_names:
        feature_name = config.usable_feature_names[0]
    logger.info("Feature utilisée pour la condition always-true : %s", feature_name)

    # Charge les features de val (nécessaires pour test_on).
    from einherjar.research.data.npy_real_loader import load_features_from_npy
    from pathlib import Path as _Path
    val_features_full = load_features_from_npy(
        asset=args.asset, asset_class=args.asset_class, timeframe=args.tf,
        config=config, data_root=_Path(args.data_root),
    )
    # Re-découpe val_features comme _load_real_data le fait.
    n_total = val_features_full.n_bougies
    ratios = config.splits["ratios"]
    train_boundary = int(n_total * float(ratios["train"]))
    val_boundary = train_boundary + int(n_total * float(ratios["val"]))
    purge = int(config.evaluation["n_window"]["max_n"])
    embargo = int(config.splits.get("embargo", {}).get("bougies", 0))
    if not config.splits.get("purging", {}).get("enabled", True):
        purge = 0
    if not config.splits.get("embargo", {}).get("enabled", True):
        embargo = 0
    val_start = train_boundary + embargo
    val_end = val_boundary - purge

    from einherjar.research.data.features import FeaturesFrame
    import polars as pl
    val_features = FeaturesFrame(
        asset=val_features_full.asset, timeframe=val_features_full.timeframe,
        df=val_features_full.df[val_start:val_end],
        feature_names=val_features_full.feature_names,
        data_version=val_features_full.data_version,
    )
    # Recharge aussi les train_features pour calibration.
    train_features = FeaturesFrame(
        asset=val_features_full.asset, timeframe=val_features_full.timeframe,
        df=val_features_full.df[:train_boundary - purge],
        feature_names=val_features_full.feature_names,
        data_version=val_features_full.data_version,
    )

    _run_engine_test(
        config, engine, train_ohlcv, train_features, val_ohlcv, val_features,
        "long", feature_name,
    )
    _run_engine_test(
        config, engine, train_ohlcv, train_features, val_ohlcv, val_features,
        "short", feature_name,
    )
    logger.info("")
    logger.info("=== Synthèse ===")
    logger.info("Si BRUT > 0 mais MOTEUR long < 0 : le signal existe, l'évaluation le tue.")
    logger.info("Si BRUT > 0 ET MOTEUR long > 0 : l'évaluation capture le signal (le STGP est en cause).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())