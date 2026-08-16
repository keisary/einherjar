"""scripts/signal_baseline_test.py — Test du signal SANS SL/TP — Phase 2.

HYPOTHÈSE : l'idée même de sortie SL/TP courte détruit le signal de trend.
Ici on évalue les mêmes stratégies (toujours long / toujours short) via le
VRAI moteur, mais avec SL/TP rendus inatteignables (sortie = timeout au close
après N bougies). On calcule quand même le CAGR, le Sharpe, le nb de trades,
le hold moyen, le ret moyen net — sans la sortie anticipée.

Objectif : confirmer que, dès qu'on enlève le SL/TP court, le signal long
devient positif (sur les actifs/marchés haussiers). On teste plusieurs
paires asset×timeframe pour élargir l'évidence.

Usage:
  export PYTHONPATH=src
  python scripts/signal_baseline_test.py
  python scripts/signal_baseline_test.py --pairs "BTCUSD,crypto,15m" "XAUUSD,forex,1h" ...
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

# Pairs par défaut : (asset, asset_class, timeframe)
DEFAULT_PAIRS = [
    ("BTCUSD", "crypto", "15m"),
    ("XAUUSD", "forex", "1h"),
    ("SP500", "indices", "4h"),
    ("GOOGL", "stock_tech", "15m"),
]


def periods_per_year_for(tf: str) -> float:
    per_day = {"5m": 288, "15m": 96, "1h": 24, "4h": 6, "1d": 1}.get(tf, 1)
    return per_day * 365


def _make_always_hypothesis(direction, asset, timeframe, feature_name):
    """Hypothèse 'toujours vrai' (condition toujours vraie) dans `direction`."""
    from einherjar.research.utils.types import (
        Amplitude, AmplitudeUnit, Direction, Hypothesis, Universe,
    )
    from einherjar.research.utils.types import CompareOp, Condition, ConditionNode, LogicalOp
    left = Condition(feature_ref=feature_name, operator=CompareOp.GT,
                     value=-1e18, transformation=None)
    right = Condition(feature_ref=feature_name, operator=CompareOp.LE,
                      value=1e18, transformation=None)
    tree = ConditionNode(op=LogicalOp.OR, left=left, right=right)
    return Hypothesis(
        id=f"always_{direction.value}",
        condition_tree=tree,
        amplitude=Amplitude(valeur=5.0, unité=AmplitudeUnit.MULTIPLE_ATR,
                            direction_implicite=direction),
        direction=direction,
        universe=Universe(assets=(asset,), timeframes=(timeframe,)),
        cooldown_k=5,
    )


def _make_no_sltp_calibrated(engine, hypothesis, train_ohlcv, train_features):
    """Calibre comme le moteur mais force SL/TP inatteignables (1e6 ATR).

    Le trade ne peut alors PAS être coupé par SL/TP : il termine toujours
    au timeout (sortie au close après N bougies). CAGR/Sharpe/n_trades sont
    calculés normalement, seule la sortie anticipée est supprimée.
    """
    from einherjar.research.engine.evaluator import CalibratedParams
    calibrated = engine.train_calibrate(hypothesis, train_ohlcv, train_features)
    # Force des distances ATR immenses → SL/TP inatteignables sur la fenêtre N.
    no_sltp = CalibratedParams(
        n_window=calibrated.n_window,
        sl_n_atr=1_000_000.0,
        tp_n_atr=1_000_000.0,
        sl_distance=1_000_000.0,
        tp_distance=1_000_000.0,
        atr_p50=calibrated.atr_p50,
        n_observations=calibrated.n_observations,
    )
    return no_sltp


def _run_pair(config, data_root, asset, asset_class, tf):
    """Charge la paire, calcule buy-and-hold + moteur SANS SL/TP long/short."""
    from einherjar.research.data.features import FeaturesFrame
    from einherjar.research.data.ohlcv import OhlcvFrame, OhlcvProvider
    from einherjar.research.data.npy_real_loader import load_features_from_npy
    from einherjar.research.engine.evaluator import EvaluationEngine

    # Mapping brut↔npy pour les classes d'actions (dossier CSV ≠ dossier .npy).
    _BLANKET_MAP = {
        "stock_tech": ("stocks", "stocks_tech"),
        "stocks_tech": ("stocks", "stocks_tech"),
        "stock_growth": ("stocks", "stocks_growth"),
        "stocks_growth": ("stocks", "stocks_growth"),
        "stock_value": ("stocks", "stocks_value"),
        "stocks_value": ("stocks", "stocks_value"),
    }
    _raw, _npy = _BLANKET_MAP.get(asset_class, (asset_class, asset_class))
    root = Path(data_root)

    # Charge OHLCV (brut) et features (.npy) séparément si les classes diffèrent.
    raw_ohlcv = OhlcvProvider().load(
        asset=asset, timeframe=tf, data_version="raw", asset_class=_raw,
    )
    full_features = load_features_from_npy(
        asset=asset, asset_class=_npy, timeframe=tf,
        config=config, data_root=root,
    )

    # Aligne OHLCV et features sur les timestamps communs.
    common_ts = raw_ohlcv.df.select("timestamp").join(
        full_features.df.select("timestamp"), on="timestamp", how="inner",
    )
    full_ohlcv = OhlcvFrame(
        asset=asset, timeframe=tf,
        df=raw_ohlcv.df.join(common_ts, on="timestamp", how="inner").sort("timestamp"),
        data_version="raw",
    )
    full_features = FeaturesFrame(
        asset=asset, timeframe=tf,
        df=full_features.df.join(common_ts, on="timestamp", how="inner").sort("timestamp"),
        feature_names=full_features.feature_names, data_version="npy",
    )

    # Split train/val/holdout (mêmes ratios que _load_real_data).
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

    def _slice(frame, start, end):
        return type(frame)(
            asset=frame.asset, timeframe=frame.timeframe,
            df=frame.df.slice(max(0, start), max(0, end - max(0, start))),
            **({"feature_names": frame.feature_names} if hasattr(frame, "feature_names") else {}),
            data_version=frame.data_version,
        )

    train_ohlcv = _slice(full_ohlcv, 0, train_boundary - purge)
    train_features = _slice(full_features, 0, train_boundary - purge)
    val_start, val_end = train_boundary + embargo, val_boundary - purge
    val_ohlcv = _slice(full_ohlcv, val_start, val_end)
    val_features = _slice(full_features, val_start, val_end)
    ppy = periods_per_year_for(tf)

    # Buy-and-hold brut (référence marché)
    close = val_ohlcv.df["close"].to_list()
    logs = [math.log(close[i] / close[i - 1]) for i in range(1, len(close))
            if close[i - 1] and close[i] > 0]
    mean = sum(logs) / len(logs)
    var = sum((r - mean) ** 2 for r in logs) / (len(logs) - 1)
    std = math.sqrt(var)
    bh_cagr = math.expm1(mean * ppy)
    bh_sharpe = (mean / std) * math.sqrt(ppy) if std > 0 else float("nan")

    # Moteur
    engine = EvaluationEngine(config=config, data_version=train_ohlcv.data_version, seed=42)
    from einherjar.research.utils.types import Direction

    logger.info("")
    logger.info("=== %s %s × %s ===", asset, asset_class, tf)
    logger.info("  Buy-and-hold BRUT [VAL %d bougies] : CAGR=%+.1f%% | Sharpe annuel=%+.2f",
                val_ohlcv.n_bougies, bh_cagr * 100, bh_sharpe)

    for direction in (Direction.LONG, Direction.SHORT):
        feature = "adx_14"
        if feature not in train_features.feature_names:
            feature = train_features.feature_names[0]
        hyp = _make_always_hypothesis(direction, asset, tf, feature)
        try:
            calibrated = _make_no_sltp_calibrated(engine, hyp, train_ohlcv, train_features)
            m = engine.test_on(hyp, val_ohlcv, val_features, calibrated, "val",
                               with_bootstrap=False)
            n = getattr(m, "n_signals", 0)
            hold = getattr(m, "avg_holding_period", 0.0)
            ret_mean = getattr(m, "ret_mean_pct_net", float("nan"))
            sharpe = getattr(m, "sharpe_net", float("nan"))
            # CAGR approché à partir du ret moyen par trade, annualisé.
            cagr_est = float("nan")
            if hold > 0 and ret_mean == ret_mean:
                n_per_year = ppy / max(hold, 1)
                cagr_est = math.expm1(n_per_year * math.log1p(max(ret_mean, -0.99))) \
                    if ret_mean > -1.0 else float("-inf")
            logger.info(
                "  Moteur SANS SL/TP [%-5s] : %5d trades | hold=%.1f bougies | "
                "ret_mean_net=%+.4f%% | CAGR_est=%+.1f%% | Sharpe=%+.2f",
                direction.value.upper(), n, hold, ret_mean, cagr_est * 100, sharpe,
            )
        except Exception as exc:
            logger.warning("  Moteur [%-5s] échoué : %s", direction.value.upper(), exc)

    logger.info("  → Lecture : si BRLUT>0 et Moteur SANS SL/TP long>0, la sortie SL/TP court était bien la cause.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Test du signal SANS SL/TP")
    parser.add_argument("--pairs", nargs="*",
                        help="Liste 'asset,class,tf' (défaut: 4 paires).")
    parser.add_argument("--config",
                        default=str(REPO_ROOT / "src/einherjar/research/config"))
    parser.add_argument("--data-root", default=r"D:\midas_v2\midasV3\src\data\compiled")
    args = parser.parse_args(argv)

    from einherjar.research.config.loader import load_config
    config = load_config(args.config)

    pairs_raw = args.pairs if args.pairs else DEFAULT_PAIRS
    pairs = [tuple(p.split(",")) if isinstance(p, str) else tuple(p) for p in pairs_raw]
    pairs = [(p[0], p[1], p[2]) for p in pairs]

    logger.info("=== Test du signal SANS SL/TP === %d paires", len(pairs))
    for asset, cls, tf in pairs:
        try:
            _run_pair(config, args.data_root, asset, cls, tf)
        except Exception as exc:
            logger.error("Paire %s/%s/%s échouée : %s", asset, cls, tf, exc)
    logger.info("=== Terminé ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())