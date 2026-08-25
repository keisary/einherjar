"""einher_io.py - Sérialisation JSONL des Einhers.

Format JSONL : un Einher par ligne.

FIX P0-2 (2026-08-24) : round-trip FIDELE.
- Les noeuds ConditionNode unaires (NOT) sont serialises SANS cle "right" ;
  l'ancien _dict_to_ast exigeait la presence de "right" pour reconstruire un
  ConditionNode -> tout Einher contenant un NOT etait relu comme une Condition
  atomique (KeyError feature_ref). Desormais : presence de "op"+"left" suffit.
- t_statistic, p_value, tp_hit_rate sont restaures (avant : defauts silencieux
  0.0/1.0/0.0 -> toute re-analyse BH depuis le corpus etait fausse).
- trade_returns est restaure s'il est present (peut etre absent pour alléger).
"""
from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path

from .types import Einher

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
    with open(path, encoding="utf-8") as f:
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
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            yield _dict_to_einher(d)


def _metrics_from_dict(m: dict):
    """Reconstruit un EinherMetrics depuis son dict (round-trip fidele)."""
    from .types import EinherMetrics

    tr = m.get("trade_returns") or ()
    return EinherMetrics(
        n_trades=m["n_trades"],
        n_tp=m["n_tp"],
        n_sl=m["n_sl"],
        n_timeout=m["n_timeout"],
        win_rate=m["win_rate"],
        avg_net_return=m["avg_net_return"],
        total_return=m["total_return"],
        sharpe_ratio=m["sharpe_ratio"],
        max_drawdown=m["max_drawdown"],
        profit_factor=m["profit_factor"],
        avg_holding_bars=m["avg_holding_bars"],
        buy_hold_return=m.get("buy_hold_return", 0.0),
        alpha=m.get("alpha", 0.0),
        # FIX P0-2 : restaurer les champs perdus (BH re-analyse correcte)
        t_statistic=float(m.get("t_statistic", 0.0)),
        p_value=float(m.get("p_value", 1.0)),
        tp_hit_rate=float(m.get("tp_hit_rate", 0.0)),
        trade_returns=tuple(float(x) for x in tr),
    )


def _dict_to_einher(d: dict) -> Einher:
    """Reconstruit un Einher depuis son dict JSON."""
    ct = d["condition_tree"]
    condition_tree = _dict_to_ast(ct)

    metrics = _metrics_from_dict(d["metrics"])
    holdout_metrics = None
    hm_raw = d.get("holdout_metrics")
    if hm_raw is not None:
        holdout_metrics = _metrics_from_dict(hm_raw)

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
        holdout_metrics=holdout_metrics,
    )


def _is_node(d: dict) -> bool:
    """True si le dict represente un ConditionNode (binaire AND/OR/XOR ou unaire NOT).

    FIX P0-2 : un NOT unaire est serialise {op, left} sans right. L'ancien test
    exigeait 'right' -> le NOT etait mal reconstruit en Condition atomique.
    """
    return isinstance(d, dict) and "op" in d and "left" in d


def _dict_to_ast(d: dict):
    """Reconstruit récursivement un AST depuis son dict."""
    from .types import Condition, ConditionNode

    if _is_node(d):
        return ConditionNode(
            op=d["op"],
            left=_dict_to_ast(d["left"]),
            right=_dict_to_ast(d["right"]) if d.get("right") is not None else None,
        )
    expr = d.get("expr")
    return Condition(
        feature_ref=d["feature_ref"],
        operator=d["operator"],
        value=d["value"],
        transformation=d.get("transformation"),
        expr=expr,
    )
