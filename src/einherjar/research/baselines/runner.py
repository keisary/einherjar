"""runner.py — CLI des baselines anti-hasard (Option B du plan).

Baselines calculées sur les mêmes conventions que le pipeline xgb_einhers
(entrée à OPEN[t+1], TP/SL 2.5%/1.5% par défaut, une position à la fois,
coûts round-trip par actif depuis fees_ctrader.json) :

1. buy_hold       : rendement brut long-only sur la fenêtre (val + holdout)
                    + Sharpe annualisé sur rendements journaliers.
2. always_long/short : Einher trivial (condition toujours vraie), la même
                       mécanique de trades que les candidats réels.
3. random_search  : K Einhers aléatoires (AND 1..3 feuilles, seuils des
                    quantiles TRAIN), backtestés sur val (sélection) et
                    holdout (une passe, rapport seul).

Usage :
    python -m einherjar.research.baselines.runner --asset BTCUSD --timeframe 1h \\
        --horizon 2d --n-random 200 --seed 42
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from einherjar.research.baselines.random_gen import generate_random_einhers
from einherjar.research.baselines.vector_eval import eval_cond_ast
from einherjar.research.xgb_einhers.backtester import backtest_einher
from einherjar.research.xgb_einhers.data_loader import (
    align_xy_with_ohlcv,
    load_ohlcv,
    load_xy,
    temporal_split,
)
from einherjar.research.xgb_einhers.label_engineer import load_costs
from einherjar.research.xgb_einhers.types import Condition, ConditionNode, Einher, EinherMetrics

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path("D:/midas_v2/einherjar")
OUTPUTS_DIR = REPO_ROOT / "outputs"
AUDITS_DIR = REPO_ROOT / "audits"


def parse_horizon(horizon_str: str) -> int:
    """Convertit '6h', '12h', '1d' en nombre de bars (TF 1h, ratio 1:1)."""
    if horizon_str.endswith("h"):
        return int(horizon_str[:-1])
    if horizon_str.endswith("d"):
        return int(horizon_str[:-1]) * 24
    if horizon_str.endswith("m"):
        return max(1, int(horizon_str[:-1]) // 60)
    raise ValueError(f"Format d'horizon non supporté : {horizon_str}")


def condition_str(ast: Condition | ConditionNode) -> str:
    """Rendu lisible d'un AST de condition."""
    if isinstance(ast, Condition):
        return f"{ast.feature_ref} {ast.operator} {ast.value:.4g}"
    left = condition_str(ast.left)
    if ast.op == "NOT":
        return f"NOT ({left})"
    right = condition_str(ast.right) if ast.right is not None else "?"
    return f"({left} {ast.op} {right})"


def n_leaves(ast: Condition | ConditionNode) -> int:
    """Nombre de conditions atomiques dans l'AST."""
    if isinstance(ast, Condition):
        return 1
    return n_leaves(ast.left) + (n_leaves(ast.right) if ast.right is not None else 0)


def _trivial_einher(
    asset: str,
    asset_class: str,
    timeframe: str,
    horizon: str,
    horizon_bars: int,
    direction: str,
    feature_name: str,
    tag: str,
) -> Einher:
    """Einher dont la condition est toujours vraie (baseline mécanique)."""
    universe = {
        "asset": asset,
        "asset_class": asset_class,
        "timeframe": timeframe,
        "horizon": horizon,
        "horizon_bars": horizon_bars,
    }
    return Einher(
        id=f"bl_{asset}_{timeframe}_{horizon}_{direction.lower()}_always",
        condition_tree=Condition(feature_ref=feature_name, operator=">=", value=float("-inf")),
        direction=direction,
        amplitude_bars=horizon_bars,
        tp_pct=0.0,
        sl_pct=0.0,
        universe=universe,
        metrics=EinherMetrics(
            n_trades=0, n_tp=0, n_sl=0, n_timeout=0, win_rate=0.0,
            avg_net_return=0.0, total_return=0.0, sharpe_ratio=0.0,
            max_drawdown=0.0, profit_factor=0.0, avg_holding_bars=0.0,
            buy_hold_return=0.0, alpha=0.0,
        ),
        scope="asset",
        source={"kind": tag},
    )


def _buy_and_hold(ohlcv_slice: pl.DataFrame) -> dict[str, float]:
    """Rendement brut + Sharpe annualisé (rendements journaliers)."""
    closes = ohlcv_slice["close"].to_numpy().astype(np.float64)
    if len(closes) < 2 or closes[0] <= 0:
        return {"total_return": 0.0, "sharpe_daily": 0.0, "n_days": 0}
    total = float((closes[-1] - closes[0]) / closes[0])

    daily = (
        ohlcv_slice.select("timestamp", "close")
        .sort("timestamp")
        .group_by_dynamic("timestamp", every="1d")
        .agg(pl.col("close").last())["close"]
        .to_numpy()
        .astype(np.float64)
    )
    rets = np.diff(daily) / daily[:-1]
    n_days = len(rets)
    sharpe = 0.0
    if n_days >= 2 and np.std(rets) > 0:
        sharpe = float(np.mean(rets) / np.std(rets, ddof=1) * np.sqrt(365))
    return {"total_return": total, "sharpe_daily": sharpe, "n_days": n_days}


def _backtest_metrics(
    einher: Einher,
    ohlcv_slice: pl.DataFrame,
    X_slice: np.ndarray,
    feature_names: list[str],
    costs_pct: float,
) -> dict[str, Any]:
    """Backtest d'un Einher sur une fenêtre, retourne metrics.to_dict()."""
    result = backtest_einher(
        einher, ohlcv_slice, X_slice, feature_names, costs_pct=costs_pct,
    )
    return result.metrics.to_dict()


def _dist_stats(values: list[float]) -> dict[str, float]:
    """Statistiques de distribution (percentiles numpy)."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "p5": float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def run(
    asset: str,
    timeframe: str,
    asset_class: str,
    horizon: str,
    n_random: int,
    seed: int,
    costs_pct: float | None,
    output_path: Path,
) -> dict[str, Any]:
    """Exécute les baselines et écrit le rapport JSON."""
    # 1. Données
    loaded = load_xy(asset, timeframe, asset_class)
    ohlcv = load_ohlcv(asset, timeframe, asset_class)
    X, ohlcv_al, ts_al = align_xy_with_ohlcv(loaded, ohlcv)
    feature_names = list(loaded.feature_names)

    if horizon not in loaded.horizons:
        raise ValueError(f"Horizon {horizon} absent de {loaded.horizons}")
    horizon_idx = loaded.horizons.index(horizon)
    horizon_bars = parse_horizon(horizon)

    # 2. Split temporel 60/20/20 purgé/embargoed (C1, conventions xgb_einhers)
    split = temporal_split(
        X, loaded.Y_ret[:, horizon_idx], embargo_bars=50, horizon_bars=horizon_bars,
    )
    val_idx, ho_idx = split.val_indices, split.holdout_indices
    X_val, X_ho = X[val_idx], X[ho_idx]
    ohlcv_val = ohlcv_al.slice(int(val_idx[0]), len(val_idx))
    ohlcv_ho = ohlcv_al.slice(int(ho_idx[0]), len(ho_idx))

    # 3. Coûts
    if costs_pct is None:
        costs_pct = load_costs(asset)
        costs_source = "fees_ctrader.json (load_costs)"
    else:
        costs_source = "CLI --costs"

    logger.info(
        "Baselines %s %s %s : N=%d val=%d holdout=%d costs=%.4f%% (%.4f)",
        asset, timeframe, horizon, len(X), len(X_val), len(X_ho), costs_pct * 100, costs_pct,
    )

    # 4. Buy & hold + always long/short (val + holdout)
    bh_val = _buy_and_hold(ohlcv_val)
    bh_ho = _buy_and_hold(ohlcv_ho)

    def _always(direction: str, tag: str) -> dict[str, Any]:
        e = _trivial_einher(
            asset, asset_class, timeframe, horizon, horizon_bars,
            direction, feature_names[0], tag,
        )
        return {
            "val": _backtest_metrics(e, ohlcv_val, X_val, feature_names, costs_pct),
            "holdout": _backtest_metrics(e, ohlcv_ho, X_ho, feature_names, costs_pct),
        }

    always_long = _always("BUY", "baseline_always_long")
    always_short = _always("SELL", "baseline_always_short")

    # 5. Random search
    rng = np.random.default_rng(seed)
    train_X = split.train_X
    candidates = generate_random_einhers(
        rng, n_random, asset, asset_class, timeframe, horizon, horizon_bars,
        feature_names, train_X,
    )

    val_metrics, ho_metrics, trigger_rates = [], [], []
    for e in candidates:
        val_metrics.append(_backtest_metrics(e, ohlcv_val, X_val, feature_names, costs_pct))
        ho_metrics.append(_backtest_metrics(e, ohlcv_ho, X_ho, feature_names, costs_pct))
        trigger_rates.append(float(eval_cond_ast(e.condition_tree, X_val, feature_names).mean()))

    def _key(m: dict[str, Any]) -> float:
        return float(m["sharpe_ratio"])

    # 6. Statistiques
    val_sharpes = [_key(m) for m in val_metrics]
    ho_sharpes = [_key(m) for m in ho_metrics]
    win_rates = [float(m["win_rate"]) for m in val_metrics]
    n_trades = [int(m["n_trades"]) for m in val_metrics]
    total_rets = [float(m["total_return"]) for m in val_metrics]

    n_admissible = sum(
        1 for m in val_metrics
        if int(m["n_trades"]) >= 30 and float(m["sharpe_ratio"]) > 0
    )
    # passes_admission complet (critères par défaut, types.py:110)
    n_passes_admission = sum(
        1 for m in val_metrics if _passes_default(m)
    )
    share_beating_bh = 0.0
    if bh_val["total_return"] != 0.0:
        share_beating_bh = float(
            np.mean([r > bh_val["total_return"] for r in total_rets])
        )

    # Sélection sur val uniquement (holdout = rapport une passe)
    order = np.argsort(val_sharpes)[::-1]
    best_by_val = []
    for pos in order[:5]:
        e = candidates[int(pos)]
        best_by_val.append({
            "id": e.id,
            "direction": e.direction,
            "n_conditions": n_leaves(e.condition_tree),
            "condition": condition_str(e.condition_tree),
            "val": val_metrics[int(pos)],
            "holdout": ho_metrics[int(pos)],
        })

    # Corrélation val/holdout (candidats avec >= 10 trades des deux côtés)
    paired = [
        (vs, hs)
        for i, (vs, hs) in enumerate(zip(val_sharpes, ho_sharpes))
        if int(val_metrics[i]["n_trades"]) >= 10 and int(ho_metrics[i]["n_trades"]) >= 10
    ]
    holdout_corr = 0.0
    if len(paired) >= 5:
        vs_arr = np.array([p[0] for p in paired])
        hs_arr = np.array([p[1] for p in paired])
        if np.std(vs_arr) > 0 and np.std(hs_arr) > 0:
            holdout_corr = float(np.corrcoef(vs_arr, hs_arr)[0, 1])

    # Par direction (check biais long/short, skill quant-strategy-audit)
    by_dir: dict[str, dict[str, Any]] = {}
    for d in ("BUY", "SELL"):
        idx = [i for i, e in enumerate(candidates) if e.direction == d]
        if idx:
            by_dir[d] = _dist_stats([val_sharpes[i] for i in idx])

    report: dict[str, Any] = {
        "meta": {
            "asset": asset,
            "asset_class": asset_class,
            "timeframe": timeframe,
            "horizon": horizon,
            "horizon_bars": horizon_bars,
            "n_random": n_random,
            "seed": seed,
            "costs_pct": costs_pct,
            "costs_source": costs_source,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "n_rows": int(len(X)),
        },
        "windows": {
            "val": {
                "start": str(ts_al[int(val_idx[0])]),
                "end": str(ts_al[int(val_idx[-1])]),
                "n_bars": int(len(val_idx)),
            },
            "holdout": {
                "start": str(ts_al[int(ho_idx[0])]),
                "end": str(ts_al[int(ho_idx[-1])]),
                "n_bars": int(len(ho_idx)),
            },
        },
        "buy_hold": {"val": bh_val, "holdout": bh_ho},
        "always_long": always_long,
        "always_short": always_short,
        "random_search": {
            "val": {
                "sharpe": _dist_stats(val_sharpes),
                "win_rate": _dist_stats(win_rates),
                "n_trades": _dist_stats([float(x) for x in n_trades]),
                "total_return": _dist_stats(total_rets),
                "trigger_rate": _dist_stats(trigger_rates),
                "n_positive_sharpe": int(sum(1 for s in val_sharpes if s > 0)),
                "n_admissible_min": n_admissible,
                "n_passes_admission": n_passes_admission,
                "share_beating_buy_hold": share_beating_bh,
            },
            "by_direction": by_dir,
            "holdout_corr_sharpe": holdout_corr,
            "best_by_val_sharpe": best_by_val,
        },
    }
    report["verdict"] = _verdict(report)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Rapport écrit : %s", output_path)
    return report


def _passes_default(m: dict[str, Any]) -> bool:
    """Réplique passes_admission (types.py:110) sur un dict de métriques."""
    if int(m["n_trades"]) < 30:
        return False
    if float(m["sharpe_ratio"]) < 0.3:
        return False
    if float(m["win_rate"]) < 0.40:
        return False
    if float(m["profit_factor"]) < 1.0:
        return False
    if abs(float(m["max_drawdown"])) > 0.30:
        return False
    if float(m["total_return"]) <= 0:
        return False
    return True


def _verdict(report: dict[str, Any] | None) -> dict[str, Any]:
    """Verdict anti-hasard calculé depuis les statistiques."""
    if report is None:
        return {}
    rnd = report["random_search"]["val"]
    sharpe = rnd["sharpe"]
    bh = report["buy_hold"]["val"]["total_return"]
    n_pos = rnd["n_positive_sharpe"]
    n = sharpe["n"]
    pos_rate = n_pos / n if n else 0.0
    median = sharpe["median"]
    p95 = sharpe["p95"]
    median_wr = rnd["win_rate"]["median"]

    # Pipeline sain : win_rate aléatoire proche de la théorie TP/(TP+SL) ~37.5%
    # (backtester 2.5%/1.5%), médiane négative nette de coûts, % positifs < 50%.
    pipeline_sane = bool(
        0.25 <= median_wr <= 0.50
        and median < 0.05
        and pos_rate < 0.50
    )

    return {
        "n_positive_sharpe_rate": pos_rate,
        "random_median_sharpe": median,
        "random_p95_sharpe": p95,
        "median_win_rate": median_wr,
        "share_beating_buy_hold": float(rnd["share_beating_buy_hold"]),
        "pipeline_sane": pipeline_sane,
        "note": (
            "Pipeline sain = win_rate aléatoire proche de TP/(TP+SL) (~37.5%), "
            "médiane aléatoire négative nette de coûts, <50% de candidats positifs. "
            "Le STGP devra battre le quantile p95 des aléatoires (val) ET survivre "
            "à la validation lourde (holdout + DSR + FDR) pour prouver qu'il n'est "
            "pas du bruit."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Baselines anti-hasard Einherjar")
    parser.add_argument("--asset", default="BTCUSD")
    parser.add_argument("--asset-class", default="crypto")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--horizon", default="2d")
    parser.add_argument("--n-random", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--costs", type=float, default=None, help="Coût round-trip (décimal), défaut fees_ctrader.json")
    parser.add_argument("--output", default=None, help="Chemin du rapport JSON")
    args = parser.parse_args()

    output = Path(args.output) if args.output else OUTPUTS_DIR / f"baselines_{args.asset}_{args.timeframe}_{args.horizon}.json"
    run(
        asset=args.asset,
        timeframe=args.timeframe,
        asset_class=args.asset_class,
        horizon=args.horizon,
        n_random=args.n_random,
        seed=args.seed,
        costs_pct=args.costs,
        output_path=output,
    )


if __name__ == "__main__":
    main()