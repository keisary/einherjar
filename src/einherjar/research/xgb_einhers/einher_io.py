"""einher_io.py - Sérialisation JSONL des Einhers.

Réponse Q20 : format JSON.
Un fichier par (asset, TF, horizon) : outputs/einhers_{asset}_{tf}_{horizon}.jsonl
Format JSONL : un Einher par ligne.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterator

from einherjar.research.xgb_einhers.types import Einher

logger = logging.getLogger(__name__)


def einher_to_json(einher: Einher) -> str:
    """Sérialise un Einher en ligne JSON (sans newline)."""
    return json.dumps(einher.to_dict(), ensure_ascii=False, default=str)


def save_einher(einher: Einher, path: Path, append: bool = True) -> None:
    """Append un Einher à un fichier JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with open(path, mode, encoding="utf-8") as f:
        f.write(einher_to_json(einher) + "\n")
    logger.debug("Saved Einher %s to %s", einher.id, path)


def load_einhers(path: Path) -> list[Einher]:
    """Charge tous les Einhers d'un fichier JSONL."""
    if not path.exists():
        return []
    einhers = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            einhers.append(_dict_to_einher(d))
    return einhers


def iter_einhers(path: Path) -> Iterator[Einher]:
    """Itère sur les Einhers d'un fichier JSONL."""
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            yield _dict_to_einher(d)


def _dict_to_einher(d: dict) -> Einher:
    """Reconstruit un Einher depuis son dict JSON."""
    from einherjar.research.xgb_einhers.types import (
        Condition, ConditionNode, EinherMetrics,
    )
    # condition_tree : récursif
    ct = d["condition_tree"]
    if "op" in ct and "left" in ct and "right" in ct:
        condition_tree = ConditionNode(
            op=ct["op"],
            left=_dict_to_ast(ct["left"]),
            right=_dict_to_ast(ct["right"]) if ct.get("right") is not None else None,
        )
    else:
        condition_tree = _dict_to_ast(ct)

    metrics_d = d["metrics"]
    metrics = EinherMetrics(
        n_trades=metrics_d["n_trades"],
        n_tp=metrics_d["n_tp"],
        n_sl=metrics_d["n_sl"],
        n_timeout=metrics_d["n_timeout"],
        win_rate=metrics_d["win_rate"],
        avg_net_return=metrics_d["avg_net_return"],
        total_return=metrics_d["total_return"],
        sharpe_ratio=metrics_d["sharpe_ratio"],
        max_drawdown=metrics_d["max_drawdown"],
        profit_factor=metrics_d["profit_factor"],
        avg_holding_bars=metrics_d["avg_holding_bars"],
        buy_hold_return=metrics_d["buy_hold_return"],
        alpha=metrics_d["alpha"],
    )
    # Sprint 2.4.1 : holdout_metrics (optionnel)
    holdout_metrics = None
    if "holdout_metrics" in d and d["holdout_metrics"] is not None:
        hm = d["holdout_metrics"]
        holdout_metrics = EinherMetrics(
            n_trades=hm["n_trades"],
            n_tp=hm["n_tp"],
            n_sl=hm["n_sl"],
            n_timeout=hm["n_timeout"],
            win_rate=hm["win_rate"],
            avg_net_return=hm["avg_net_return"],
            total_return=hm["total_return"],
            sharpe_ratio=hm["sharpe_ratio"],
            max_drawdown=hm["max_drawdown"],
            profit_factor=hm["profit_factor"],
            avg_holding_bars=hm["avg_holding_bars"],
            buy_hold_return=hm.get("buy_hold_return", 0.0),
            alpha=hm.get("alpha", 0.0),
        )

    return Einher(
        id=d["id"],
        condition_tree=condition_tree,
        direction=d["direction"],
        amplitude_bars=d["amplitude_bars"],
        tp_pct=d["tp_pct"],
        sl_pct=d["sl_pct"],
        universe=d["universe"],
        metrics=metrics,
        scope=d.get("scope", "asset"),
        cross_asset_test=d.get("cross_asset_test"),
        source=d.get("source", {}),
        created_at=d.get("created_at", ""),
        data_version=d.get("data_version", ""),
        holdout_metrics=holdout_metrics,  # Sprint 2.4.1
    )


def _dict_to_ast(d: dict):
    """Reconstruit récursivement un AST depuis son dict."""
    from einherjar.research.xgb_einhers.types import Condition, ConditionNode
    if "op" in d and "left" in d and "right" in d:
        return ConditionNode(
            op=d["op"],
            left=_dict_to_ast(d["left"]),
            right=_dict_to_ast(d["right"]) if d.get("right") is not None else None,
        )
    return Condition(
        feature_ref=d["feature_ref"],
        operator=d["operator"],
        value=d["value"],
        transformation=d.get("transformation"),
    )
