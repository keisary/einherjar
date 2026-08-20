"""corpus.py — Corpus versionné des Einhers admis (append-only).

Le corpus est le SEUL artefact critique persisté (pratique du projet,
corpus.jsonl) : une ligne JSON par Einher admis + son dossier d'admission.
Append-only : on n'édite jamais les lignes existantes.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

OUTPUT_DIR = Path(r"D:/midas_v2/einherjar/outputs")
CORPUS_FILENAME = "corpus.jsonl"


def condition_from_dict(d: dict[str, Any]) -> Any:
    """Reconstruit la Condition/ConditionNode depuis sa forme sérialisée.

    Une condition sérialisée (JSON) est un dict : soit un atome
    {feature_ref, operator, value, expr}, soit un nœud {op, left, right}.
    L'expr numérique (kind: feature/const/binnum) est reconstruite en
    NumExpr STGP.
    """
    from einherjar.research.search_engine.expression import BinNum, Const, Feature
    from einherjar.research.xgb_einhers.types import Condition, ConditionNode

    def rebuild_expr(x: dict[str, Any]) -> Any:
        if x.get("kind") == "feature":
            return Feature(x["feature_ref"])
        if x.get("kind") == "const":
            return Const(x["value"])
        return BinNum(op=x["op"], left=rebuild_expr(x["left"]), right=rebuild_expr(x["right"]))

    left, right = d.get("left"), d.get("right")
    if left is not None or right is not None:
        return ConditionNode(
            op=d["op"],
            left=condition_from_dict(left) if left is not None else None,
            right=condition_from_dict(right) if right is not None else None,
        )
    return Condition(
        feature_ref=d.get("feature_ref", ""),
        operator=d.get("operator", ""),
        value=d.get("value", 0.0),
        expr=rebuild_expr(d["expr"]) if d.get("expr") else None,
    )


def fingerprint_of(ast: Any) -> str:
    """Empreinte exacte (sha256) de la condition, forme canonique JSON."""
    canon = json.dumps(ast.to_dict(), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def load_corpus(path: Path | None = None) -> list[dict[str, Any]]:
    path = path or OUTPUT_DIR / CORPUS_FILENAME
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _assign_gp_id(entry: dict[str, Any], fingerprint: str) -> dict[str, Any]:
    """Id GP (gp_<fingerprint[:12]>) si l'Einher n'en porte pas."""
    if not entry.get("id"):
        entry["id"] = f"gp_{fingerprint[:12]}"
    return entry


def entry_of(
    einher: Any,
    outcome: Any,
    *,
    fingerprint: str,
    holdout_metrics: Any | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Construit l'entrée JSON d'un Einher + son dossier d'admission."""
    entry = einher.to_dict()
    entry = _assign_gp_id(entry, fingerprint)
    entry["fingerprint"] = fingerprint
    entry["admission"] = {
        "admitted": outcome.admitted,
        "reasons": outcome.reasons,
    }
    if holdout_metrics is not None:
        entry["holdout_metrics"] = (
            holdout_metrics.to_dict() if hasattr(holdout_metrics, "to_dict") else holdout_metrics
        )
    if extra:
        entry.update(extra)
    return entry


def append_entry(entry: dict[str, Any], path: Path) -> None:
    """Écrit une ligne JSON append-only (crée le parent si besoin)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def append_einher(
    einher: Any,
    outcome: Any,
    *,
    fingerprint: str,
    holdout_metrics: Any | None = None,
    path: Path | None = None,
) -> dict[str, Any]:
    """Ajoute un Einher admis au corpus ; retourne l'entrée écrite."""
    path = path or OUTPUT_DIR / CORPUS_FILENAME
    entry = entry_of(
        einher, outcome,
        fingerprint=fingerprint, holdout_metrics=holdout_metrics,
    )
    append_entry(entry, path)
    return entry


def append_candidate(
    einher: Any,
    outcome: Any,
    *,
    fingerprint: str,
    holdout_metrics: Any | None = None,
    extra: dict[str, Any] | None = None,
    path: Path | None = None,
) -> dict[str, Any]:
    """Ajoute un candidat (admis OU rejeté) avec son dossier complet."""
    path = path or OUTPUT_DIR / CORPUS_FILENAME
    entry = entry_of(
        einher, outcome,
        fingerprint=fingerprint, holdout_metrics=holdout_metrics, extra=extra,
    )
    append_entry(entry, path)
    return entry