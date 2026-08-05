"""holdout/ledger.py — Ledger persistant des accès holdout (P1 #4).

Le holdout est SACRE : une seule évaluation par (strategy_id, data_version).
Pour garantir cette propriété même après redémarrage du processus, on
persiste un ledger append-only sur disque.

Garanties :
  - Append-only : on n'efface jamais, on n'écrase jamais.
  - Atomique : chaque écriture est flushée immédiatement (fsync).
  - Vérifiable : has_access() consulte le ledger avant toute évaluation.
  - Anti-réentrance : un 2e appel à evaluate() sur le même couple
    (strategy_id, data_version) lève une erreur BLOQUANTE.

Format du ledger : JSON Lines (un entry par ligne).
"""

from __future__ import annotations

import json
import logging
import hashlib
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone as dt_timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# Chemin par defaut ancre sur la racine du repo (pas le cwd).
# Sans cet ancrage, deux invocations depuis deux cwd differents obtiennent
# deux fichiers ledger distincts, ce qui defait silencieusement l'invariant
# "le holdout n'est consulte qu'une seule fois" (S-3.8) : chaque execution
# se croit unique.
# Path(__file__) = .../src/einherjar/research/holdout/ledger.py
# parents[3] = racine du repo.
DEFAULT_LEDGER_PATH: Path = (
    Path(__file__).resolve().parents[3] / "outputs" / "holdout_ledger.jsonl"
)


# --------------------------------------------------------------------------- #
# Entrée du ledger
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class HoldoutEntry:
    """Une entrée du ledger = un accès holdout effectué (irréversible)."""

    strategy_id: str
    data_version: str
    timestamp: str                          # ISO 8601 UTC
    n_trades: int = 0
    sharpe: float = 0.0
    degradation_flag: str = "OK"
    metrics_snapshot: dict[str, Any] = field(default_factory=dict)
    seed: int = 42
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "data_version": self.data_version,
            "timestamp": self.timestamp,
            "n_trades": self.n_trades,
            "sharpe": self.sharpe,
            "degradation_flag": self.degradation_flag,
            "metrics_snapshot": self.metrics_snapshot,
            "seed": self.seed,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "HoldoutEntry":
        return cls(
            strategy_id=d["strategy_id"],
            data_version=d["data_version"],
            timestamp=d["timestamp"],
            n_trades=int(d.get("n_trades", 0)),
            sharpe=float(d.get("sharpe", 0.0)),
            degradation_flag=d.get("degradation_flag", "OK"),
            metrics_snapshot=dict(d.get("metrics_snapshot", {})),
            seed=int(d.get("seed", 42)),
            meta=dict(d.get("meta", {})),
        )


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #


class HoldoutAlreadyUsedError(Exception):
    """Erreur BLOQUANTE : le holdout a déjà été consommé pour ce couple."""


# --------------------------------------------------------------------------- #
# Ledger
# --------------------------------------------------------------------------- #


class HoldoutLedger:
    """Ledger persistant des accès holdout (append-only, atomique).

    Usage :
        ledger = HoldoutLedger()
        if not ledger.has_access(strategy_id, data_version):
            # faire l'évaluation
            entry = HoldoutEntry(...)
            ledger.record(entry)  # atomique
        else:
            raise HoldoutAlreadyUsedError(...)
    """

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path: Path = path or DEFAULT_LEDGER_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            # Crée le fichier vide.
            self.path.touch()
        logger.info("HoldoutLedger initialisé : %s", self.path)

    def _reservation_path(self, strategy_id: str, data_version: str) -> Path:
        key = hashlib.sha256(f"{strategy_id}\0{data_version}".encode("utf-8")).hexdigest()
        return self.path.with_name(f"{self.path.name}.{key}.pending")

    def reserve(self, strategy_id: str, data_version: str) -> None:
        """Atomically consume access before holdout evaluation."""
        reservation = self._reservation_path(strategy_id, data_version)
        if self.has_access(strategy_id, data_version):
            raise HoldoutAlreadyUsedError(
                f"Holdout already reserved or consumed for {strategy_id}/{data_version}"
            )
        try:
            fd = os.open(str(reservation), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise HoldoutAlreadyUsedError(
                f"Holdout already reserved or consumed for {strategy_id}/{data_version}"
            ) from exc
        with os.fdopen(fd, "w", encoding="utf-8") as fp:
            fp.write(json.dumps({"strategy_id": strategy_id, "data_version": data_version}))
            fp.flush()
            os.fsync(fp.fileno())

    def finalize_reservation(self, strategy_id: str, data_version: str) -> None:
        reservation = self._reservation_path(strategy_id, data_version)
        if reservation.exists():
            reservation.unlink()

    def has_access(self, strategy_id: str, data_version: str) -> bool:
        """True si une entrée existe pour (strategy_id, data_version)."""
        if self._reservation_path(strategy_id, data_version).exists():
            return True
        for entry in self.iter_entries():
            if entry.strategy_id == strategy_id and entry.data_version == data_version:
                return True
        return False

    def get_entry(
        self, strategy_id: str, data_version: str,
    ) -> Optional[HoldoutEntry]:
        """Retourne l'entrée existante pour (strategy_id, data_version), ou None."""
        for entry in self.iter_entries():
            if entry.strategy_id == strategy_id and entry.data_version == data_version:
                return entry
        return None

    def record(self, entry: HoldoutEntry) -> None:
        """Append atomique d'une entrée. Lève si (strategy_id, data_version) déjà présent.

        Atomicité : on écrit la ligne + flush + fsync AVANT de retourner.
        Si crash après fsync, l'entrée est persistée.
        Si crash avant fsync, l'entrée n'est pas persistée (et on peut re-tenter).
        """
        if self.get_entry(entry.strategy_id, entry.data_version) is not None:
            raise HoldoutAlreadyUsedError(
                f"Holdout déjà consommé pour (strategy_id={entry.strategy_id}, "
                f"data_version={entry.data_version}). Voir {self.path}."
            )
        line = json.dumps(entry.to_dict(), ensure_ascii=False) + "\n"
        with self.path.open("a", encoding="utf-8") as fp:
            fp.write(line)
            fp.flush()
            try:
                import os
                os.fsync(fp.fileno())
            except OSError:
                # Windows ne supporte pas toujours fsync sur tous les FS.
                # On log un warning mais on continue (le fichier est déjà flushé).
                logger.debug("fsync non supporté sur ce FS, flush suffit")
        logger.info(
            "Holdout ledger append : strategy_id=%s, data_version=%s, n_trades=%d, sharpe=%.4f",
            entry.strategy_id, entry.data_version, entry.n_trades, entry.sharpe,
        )

    def iter_entries(self) -> list[HoldoutEntry]:
        """Itère sur toutes les entrées du ledger (lecture seule).

        Returns:
            Liste d'entrées. Volontairement une liste (pas un générateur) pour
            faciliter la réutilisation et le comptage.
        """
        if not self.path.exists():
            return []
        entries: list[HoldoutEntry] = []
        with self.path.open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    entries.append(HoldoutEntry.from_dict(d))
                except (json.JSONDecodeError, KeyError) as exc:
                    logger.warning("Ligne ledger invalide ignorée : %s", exc)
        return entries

    def count(self) -> int:
        """Nombre d'entrées dans le ledger."""
        return len(self.iter_entries())

    def clear_for_testing(self) -> None:
        """Vide le ledger. PROTEGE : interdit hors contexte de test.

        Le holdout est sacre (S-3.8). Permettre d'effacer le ledger depuis
        n'importe quel script de maintenance / notebook / debug rouvrirait
        silencieusement la porte a une double consultation. On verifie
        donc que l'appelant est bien un test (PYTEST_CURRENT_TEST dans
        l'env) OU que le chemin du ledger pointe sous le tempdir
        (cas typique : tests avec ledger isole).
        """
        import tempfile
        in_pytest = "PYTEST_CURRENT_TEST" in os.environ
        in_tempdir = str(self.path).startswith(tempfile.gettempdir())
        if not (in_pytest or in_tempdir):
            raise RuntimeError(
                "clear_for_testing() interdit hors contexte de test. "
                "Si c'est un script de maintenance legitime, utilise "
                "un chemin sous tempfile.gettempdir() ou lance pytest. "
                f"(path={self.path}, PYTEST_CURRENT_TEST={'PYTEST_CURRENT_TEST' in os.environ})"
            )
        self.path.write_text("", encoding="utf-8")
        logger.warning("Holdout ledger cleared (testing only) path=%s", self.path)
