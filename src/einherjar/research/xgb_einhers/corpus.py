"""corpus.py - Store append-only pour les Einhers ADMIS.

Sprint 3.6 (P1 #7) : le corpus est la source de verite des Einhers
qui ont passe TOUTES les validations (val + holdout). Format JSONL,
un Einher par ligne, jamais ecrase, juste append.

Usage :
    corpus = CorpusStore("outputs/corpus.jsonl")
    corpus.add(einher)
    for e in corpus.iter(): ...
    n = corpus.count()
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Iterator, Optional

from .types import Einher

logger = logging.getLogger(__name__)


class CorpusStore:
    """Append-only JSONL store pour Einhers admis.

    Thread-safe via un lock (les workers en parallele peuvent append).
    """

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        if not self.path.exists():
            self.path.touch()

    def add(self, einher: Einher) -> None:
        """Append un Einher au corpus (thread-safe)."""
        d = einher.to_dict()
        line = json.dumps(d, ensure_ascii=False, default=str)
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def add_batch(self, einhers: list[Einher]) -> int:
        """Append N Einhers d'un coup, retourne le nombre ajoute."""
        if not einhers:
            return 0
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                for e in einhers:
                    f.write(json.dumps(e.to_dict(), ensure_ascii=False, default=str) + "\n")
        return len(einhers)

    def iter(self) -> Iterator[Einher]:
        """Itere sur tous les Einhers du corpus."""
        from .einher_io import _dict_to_einher
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield _dict_to_einher(json.loads(line))

    def count(self) -> int:
        """Compte les Einhers (approx rapide via wc ligne)."""
        if not self.path.exists():
            return 0
        n = 0
        with open(self.path, "r", encoding="utf-8") as f:
            for _ in f:
                n += 1
        return n

    def clear(self) -> None:
        """Vide le corpus (utilise avec precaution)."""
        with self._lock:
            self.path.write_text("", encoding="utf-8")
