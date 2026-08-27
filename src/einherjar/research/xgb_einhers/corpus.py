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
import os
import struct
from collections.abc import Iterator
from pathlib import Path

from .types import Einher

logger = logging.getLogger(__name__)


def _file_lock(path: Path) -> None:
    """Acquire an exclusive file lock (cross-process, Windows+Linux)."""
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    # Busy-wait with a lock file (simple, cross-platform)
    import time
    for _ in range(500):  # max 5 seconds
        try:
            # O_CREAT | O_EXCL : atomic create, fails if exists
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return
        except FileExistsError:
            time.sleep(0.01)
    # Fallback : proceed without lock (better than hanging)
    logger.warning("File lock timeout on %s, proceeding without lock", path)


def _file_unlock(path: Path) -> None:
    """Release the file lock."""
    lock_path = path.with_suffix(path.suffix + ".lock")
    try:
        os.remove(str(lock_path))
    except OSError:
        pass


class CorpusStore:
    """Append-only JSONL store pour Einhers admis.

    Cross-process safe via file lock (fonctionne avec multiprocessing).
    """

    def __init__(self, path: Path | str):
        """__init__.

        Args:
            path: TODO document.
        """
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()

    def add(self, einher: Einher) -> None:
        """Append un Einher au corpus (cross-process safe)."""
        d = einher.to_dict()
        line = json.dumps(d, ensure_ascii=False, default=str)
        _file_lock(self.path)
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
                os.fsync(f.fileno())
        finally:
            _file_unlock(self.path)

    def add_batch(self, einhers: list[Einher]) -> int:
        """Append N Einhers d'un coup, retourne le nombre ajoute."""
        if not einhers:
            return 0
        _file_lock(self.path)
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                for e in einhers:
                    f.write(json.dumps(e.to_dict(), ensure_ascii=False, default=str) + "\n")
                f.flush()
                os.fsync(f.fileno())
        finally:
            _file_unlock(self.path)
        return len(einhers)

    def iter(self) -> Iterator[Einher]:
        """Itere sur tous les Einhers du corpus."""
        from .einher_io import _dict_to_einher
        with open(self.path, encoding="utf-8") as f:
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
        with open(self.path, encoding="utf-8") as f:
            for _ in f:
                n += 1
        return n

    def clear(self) -> None:
        """Vide le corpus (utilise avec precaution)."""
        _file_lock(self.path)
        try:
            self.path.write_text("", encoding="utf-8")
        finally:
            _file_unlock(self.path)
