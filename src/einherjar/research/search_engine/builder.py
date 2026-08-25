"""builder.py — Assemble des Einhers depuis les expressions STGP.

Transforme une BoolExpr (arbre STGP) en Einher backtestable : le pont
evaluator.to_condition_tree produit un Condition/ConditionNode avec atomes
expr, que le backtester xgb évalue via le dispatch ajouté dans
condition_tree.py. Placeholder metrics rempli au backtest.
"""
from __future__ import annotations

from einherjar.research.search_engine.evaluator import to_condition_tree
from einherjar.research.xgb_einhers.types import Einher, EinherMetrics

DEFAULT_TP_PCT = 2.5
DEFAULT_SL_PCT = 1.5


def empty_metrics(buy_hold_return: float = 0.0) -> EinherMetrics:
    """Métriques placeholder (remplacées après backtest)."""
    return EinherMetrics(
        n_trades=0, n_tp=0, n_sl=0, n_timeout=0,
        win_rate=0.0, avg_net_return=0.0, total_return=0.0,
        sharpe_ratio=0.0, max_drawdown=0.0, profit_factor=0.0,
        avg_holding_bars=0.0, buy_hold_return=buy_hold_return,
        alpha=0.0, t_statistic=0.0, p_value=1.0, trade_returns=(),
    )


def build_einher(
    expr: object,
    direction: str,
    amplitude_bars: int,
    universe: dict,
    *,
    costs_pct: float,
    data_version: str = "",
) -> Einher:
    """Assemble un Einher backtestable depuis une BoolExpr STGP."""
    return Einher(
        id="",  # assigné à l'admission (corpus)
        condition_tree=to_condition_tree(expr),
        direction=direction,
        amplitude_bars=amplitude_bars,
        tp_pct=DEFAULT_TP_PCT,
        sl_pct=DEFAULT_SL_PCT,
        universe=universe,
        metrics=empty_metrics(),
        scope="general",
        source={"engine": "search_engine", "costs_pct": costs_pct},
        data_version=data_version,
    )
