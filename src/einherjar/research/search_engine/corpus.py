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
    entry = einher.to_dict()
    entry["fingerprint"] = fingerprint
    entry["admission"] = {
        "admitted": outcome.admitted,
        "reasons": outcome.reasons,
    }
    if holdout_metrics is not None:
        entry["holdout_metrics"] = (
            holdout_metrics.to_dict() if hasattr(holdout_metrics, "to_dict") else holdout_metrics
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry