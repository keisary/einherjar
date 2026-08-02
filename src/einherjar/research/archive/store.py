"""
archive/store.py — I/O append-only pour l'Archive (ONTOLOGY.md § 8).

L'archive est append-only : on n'efface jamais, on n'écrase jamais.
La réévaluation d'une hypothèse sur un nouveau data_version est permise
(et n'est pas considérée comme un doublon).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterator, Optional

from einherjar.research.archive.schema import ArchiveEntry

logger = logging.getLogger(__name__)


ARCHIVE_FILENAME = "archive.jsonl"   # JSON Lines : un entry par ligne


def archive_path(base: Path = Path("outputs")) -> Path:
    """Chemin par défaut de l'archive."""
    return base / "archive" / ARCHIVE_FILENAME


def append_entry(entry: ArchiveEntry, path: Optional[Path] = None) -> None:
    """Ajoute une entrée à l'archive. Valide le schéma avant d'écrire.

    Raises:
        ValueError: si l'entrée ne respecte pas le schéma.
    """
    errors = entry.validate()
    if errors:
        raise ValueError(f"Entrée d'archive invalide: {errors}")

    p = path or archive_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
    logger.info(
        "Archive append: id=%s type=%s raison=%s",
        entry.id, entry.type_élément, entry.raison_rejet.value,
    )


def iter_entries(path: Optional[Path] = None) -> Iterator[ArchiveEntry]:
    """Itère sur toutes les entrées de l'archive (lecture seule)."""
    p = path or archive_path()
    if not p.exists():
        return
    with p.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Ligne d'archive invalide ignorée: %s", exc)
                continue
            yield _entry_from_dict(d)


def has_fingerprint(
    fingerprint_structurel: str,
    data_version: str,
    path: Optional[Path] = None,
) -> bool:
    """Vérifie si un fingerprint structurel existe déjà dans l'archive pour ce data_version.

    Utilisé par `admission/decision.py` pour la déduplication (invariant I-6).
    """
    for e in iter_entries(path):
        if e.fingerprint_structurel == fingerprint_structurel and e.data_version == data_version:
            return True
    return False


def has_comportemental_fingerprint(
    fingerprint_comportemental: str,
    data_version: str,
    path: Optional[Path] = None,
) -> bool:
    """Vérifie si un fingerprint comportemental existe déjà pour ce data_version."""
    for e in iter_entries(path):
        if e.fingerprint_comportemental == fingerprint_comportemental and e.data_version == data_version:
            return True
    return False


def _entry_from_dict(d: dict) -> ArchiveEntry:
    """Reconstruit une ArchiveEntry depuis son dict (best-effort)."""
    from einherjar.research.utils.types import MesuresBrutes, RejectionReason
    return ArchiveEntry(
        id=d.get("id", ""),
        type_élément=d.get("type_élément", "hypothesis"),
        raison_rejet=RejectionReason(d.get("raison_rejet", "OTHER")),
        date_rejet=d.get("date_rejet", ""),
        data_version=d.get("data_version", ""),
        seed=d.get("seed", 0),
        splits=d.get("splits", {}),
        costs_simulated=d.get("costs_simulated", {}),
        sl_tp_source=d.get("sl_tp_source", "from_train"),
        metriques_portefeuille_val=d.get("metriques_portefeuille_val", {}),
        bootstrap_ci_val=d.get("bootstrap_ci_val", {}),
        deflated_sharpe_ratio=d.get("deflated_sharpe_ratio"),
        probability_of_backtest_overfitting=d.get("probability_of_backtest_overfitting"),
        descriptors_comportementaux=d.get("descriptors_comportementaux", {}),
        fingerprint_structurel=d.get("fingerprint_structurel", ""),
        fingerprint_comportemental=d.get("fingerprint_comportemental", ""),
        fingerprint=d.get("fingerprint", ""),
        ret_series=tuple(d.get("ret_series", ())),
        element_ref_id=d.get("element_ref_id", ""),
        # mesures_brutes_train/val non reconstruits ici (best-effort, allourdit)
    )
