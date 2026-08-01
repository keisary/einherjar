"""data/corpus.py — Lecture/écriture du corpus actif d'Einhers.

Le corpus est la mémoire opérationnelle (distincte de l'archive qui
conserve les rejets). Format : JSON append-friendly, versionné.

Format de fichier :
  corpus_v1.json = {
    "version": "corpus_v1",
    "data_version": "v1_2026-08-01",
    "einhers": [ <Einher.to_dict()>, ... ],
    "stats": { "n_einhers": N, "...": "..." },
    "updated_at": "2026-08-01T..."
  }
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Optional  # noqa: F401

from einherjar.research.utils.types import Einher

logger = logging.getLogger(__name__)


CORPUS_FILENAME = "corpus.json"


@dataclass
class Corpus:
    """Mémoire opérationnelle des Einhers actifs."""

    version: str
    data_version: str
    einhers: list[Einher] = field(default_factory=list)
    updated_at: str = ""

    def n_einhers(self) -> int:  # noqa: D102
        return len(self.einhers)

    def to_dict(self) -> dict:  # noqa: D102
        return {
            "version": self.version,
            "data_version": self.data_version,
            "n_einhers": self.n_einhers(),
            "einhers": [e.to_dict() for e in self.einhers],
            "updated_at": self.updated_at or datetime.now(UTC).isoformat(),
        }


def load_corpus(corpus_path: Path) -> Corpus:
    """Charge un corpus depuis un fichier JSON. Retourne un Corpus vide si absent."""
    if not corpus_path.exists():
        logger.info("Corpus introuvable à %s, retour d'un corpus vide.", corpus_path)
        return Corpus(version="corpus_v1", data_version="")
    try:
        d = json.loads(corpus_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Corpus JSON invalide : {corpus_path} ({exc})") from exc
    return Corpus(
        version=d.get("version", "corpus_v1"),
        data_version=d.get("data_version", ""),
        einhers=[],  # reconstruction depuis dict sera ajoutée dans admission/
        updated_at=d.get("updated_at", ""),
    )


def save_corpus(corpus: Corpus, corpus_path: Path) -> None:
    """Sauvegarde le corpus en JSON, atomiquement (tmp + rename)."""
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = corpus_path.with_suffix(corpus_path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(corpus.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    tmp.replace(corpus_path)
    logger.info("Corpus sauvegardé : %d einhers dans %s", corpus.n_einhers(), corpus_path)


def default_corpus_path(base: Path = Path("outputs")) -> Path:
    """Chemin par défaut du corpus actif (dans outputs/)."""
    return base / "corpus" / CORPUS_FILENAME
