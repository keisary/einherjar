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
from dataclasses import dataclass, field
from datetime import datetime, timezone as dt_timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


DEFAULT_LEDGER_PATH: Path = Path("outputs") / "holdout_ledger.jsonl"


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

    def has_access(self, strategy_id: str, data_version: str) -> bool:
        """True si une entrée existe pour (strategy_id, data_version)."""
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
        if self.has_access(entry.strategy_id, entry.data_version):
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
        """Vide le ledger (POUR LES TESTS UNIQUEMENT)."""
        self.path.write_text("", encoding="utf-8")
        logger.warning("Holdout ledger cleared (testing only)")
