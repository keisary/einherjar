"""archive.py - Store append-only pour les Einhers REJETES.

Sprint 3.6 (P1 #8) : les Einhers qui ont des metriques val (donc
backtest reussi) mais qui ratent l'admission finale (BH, min_trades,
sharpe trop bas, etc.) sont stockes ici pour inspection.

Chaque entree archivee contient :
- L'Einher complet (avec metriques val + holdout si dispo)
- La raison du rejet (passed=False reason)
- Le scope d'ou il vient (asset, market, general)
- Le timestamp

Format JSONL, un Einher par ligne.
"""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

from einherjar.research.xgb_einhers.types import Einher

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ArchiveEntry:
    """Une entree d'archive = Einher + raison du rejet + contexte."""
    einher: Einher
    rejection_reason: str
    scope: str                       # asset / market / general
    asset: str
    asset_class: str
    timeframe: str
    horizon: str
    rejected_at: str = ""

    def to_dict(self) -> dict:
        return {
            "einher": self.einher.to_dict(),
            "rejection_reason": self.rejection_reason,
            "scope": self.scope,
            "asset": self.asset,
            "asset_class": self.asset_class,
            "timeframe": self.timeframe,
            "horizon": self.horizon,
            "rejected_at": self.rejected_at,
        }


class ArchiveStore:
    """Append-only JSONL store pour les Einhers rejetes (avec raison)."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        if not self.path.exists():
            self.path.touch()

    def add(
        self,
        einher: Einher,
        rejection_reason: str,
        scope: str = "asset",
        asset: str = "",
        asset_class: str = "",
        timeframe: str = "",
        horizon: str = "",
    ) -> None:
        """Append une entree d'archive."""
        from datetime import datetime, timezone
        entry = ArchiveEntry(
            einher=einher,
            rejection_reason=rejection_reason,
            scope=scope,
            asset=asset,
            asset_class=asset_class,
            timeframe=timeframe,
            horizon=horizon,
            rejected_at=datetime.now(timezone.utc).isoformat(),
        )
        line = json.dumps(entry.to_dict(), ensure_ascii=False, default=str)
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def add_batch(
        self,
        einhers: list[Einher],
        rejection_reason: str,
        scope: str = "asset",
        asset: str = "",
        asset_class: str = "",
        timeframe: str = "",
        horizon: str = "",
    ) -> int:
        """Append N Einhers avec la meme raison (rapide)."""
        if not einhers:
            return 0
        from datetime import datetime, timezone
        ts = datetime.now(timezone.utc).isoformat()
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                for e in einhers:
                    entry = ArchiveEntry(
                        einher=e,
                        rejection_reason=rejection_reason,
                        scope=scope,
                        asset=asset,
                        asset_class=asset_class,
                        timeframe=timeframe,
                        horizon=horizon,
                        rejected_at=ts,
                    )
                    f.write(json.dumps(entry.to_dict(), ensure_ascii=False, default=str) + "\n")
        return len(einhers)

    def iter(self) -> Iterator[ArchiveEntry]:
        """Itere sur toutes les entrees archivees."""
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                from einherjar.research.xgb_einhers.einher_io import _dict_to_einher
                yield ArchiveEntry(
                    einher=_dict_to_einher(d["einher"]),
                    rejection_reason=d["rejection_reason"],
                    scope=d.get("scope", "asset"),
                    asset=d.get("asset", ""),
                    asset_class=d.get("asset_class", ""),
                    timeframe=d.get("timeframe", ""),
                    horizon=d.get("horizon", ""),
                    rejected_at=d.get("rejected_at", ""),
                )

    def count(self) -> int:
        if not self.path.exists():
            return 0
        n = 0
        with open(self.path, "r", encoding="utf-8") as f:
            for _ in f:
                n += 1
        return n

    def count_by_reason(self) -> dict[str, int]:
        """Compte par raison de rejet."""
        out: dict[str, int] = {}
        for e in self.iter():
            out[e.rejection_reason] = out.get(e.rejection_reason, 0) + 1
        return out

    def clear(self) -> None:
        with self._lock:
            self.path.write_text("", encoding="utf-8")
