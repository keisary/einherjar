"""
==========================================================
Core Types
==========================================================

Types valeur partagés entre le bootstrap (discovery.py) et
le moteur (core.Engine).

- DiscoveryTarget   : une paire asset / timeframe.
- DiscoveryPairResult : résultat complet d'un Engine.run_pair().
- DiscoveryRunResult  : résultat agrégé d'un run (construit
  par le bootstrap).
- DiscoverySettings   : paramètres d'un run (construit par
  le bootstrap).
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timezone
from typing import Any
from typing import Mapping


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_text(value: Any, default: str = "") -> str:
    text = str(value or "").strip()
    return text if text else default


def _slugify(value: Any) -> str:
    text = _normalize_text(value, "unknown").lower()
    out: list[str] = []
    for ch in text:
        if ch.isalnum():
            out.append(ch)
        elif ch in {" ", "-", "_", ".", "/", ":"}:
            out.append("_")
    slug = "".join(out)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_") or "unknown"


# ==========================================================
# DISCOVERY TARGET
# ==========================================================

@dataclass(slots=True, frozen=True)
class DiscoveryTarget:
    """
    Paire asset / timeframe à traiter par le moteur.
    """

    asset: str
    timeframe: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "asset", _normalize_text(self.asset, "unknown"))
        object.__setattr__(self, "timeframe", _normalize_text(self.timeframe, "unknown"))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def key(self) -> str:
        return f"{self.asset}@{self.timeframe}"

    @property
    def slug(self) -> str:
        return f"{_slugify(self.asset)}__{_slugify(self.timeframe)}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "asset": self.asset,
            "timeframe": self.timeframe,
            "metadata": dict(self.metadata),
            "key": self.key,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DiscoveryTarget":
        return cls(
            asset=str(data.get("asset", "unknown")),
            timeframe=str(data.get("timeframe", "unknown")),
            metadata=dict(data.get("metadata", {})),
        )


# ==========================================================
# DISCOVERY PAIR RESULT
# ==========================================================

@dataclass(slots=True)
class DiscoveryPairResult:
    """
    Résultat complet du pipeline pour une paire.

    Produit par Engine.run_pair() et consommé par le
    bootstrap pour construire le DiscoveryRunResult.
    """

    target: DiscoveryTarget
    index: int

    # EngineState par paire
    state: Any

    # Dataset / phases
    dataset: Any
    candidates: tuple[Any, ...]
    validated: tuple[Any, ...]
    rejected: tuple[Any, ...]
    execution_results: tuple[Any, ...]
    execution_report: Any

    # Einhers + portfolio
    einhers: tuple[Any, ...]
    selection: Any
    allocation: Any
    portfolio_report: Any

    # Memory / Knowledge / Export
    memory_snapshot: dict[str, Any]
    knowledge_snapshot: dict[str, Any]
    export_paths: dict[str, str]

    metadata: dict[str, Any]
    started_at: datetime
    finished_at: datetime
    success: bool
    errors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidates", tuple(self.candidates))
        object.__setattr__(self, "validated", tuple(self.validated))
        object.__setattr__(self, "rejected", tuple(self.rejected))
        object.__setattr__(self, "execution_results", tuple(self.execution_results))
        object.__setattr__(self, "einhers", tuple(self.einhers))
        object.__setattr__(self, "memory_snapshot", dict(self.memory_snapshot))
        object.__setattr__(self, "knowledge_snapshot", dict(self.knowledge_snapshot))
        object.__setattr__(self, "export_paths", dict(self.export_paths))
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "errors", tuple(str(e) for e in self.errors))

    # ==================================================
    # PROPERTIES
    # ==================================================

    @property
    def pair_key(self) -> str:
        return self.target.key

    @property
    def asset(self) -> str:
        return self.target.asset

    @property
    def timeframe(self) -> str:
        return self.target.timeframe

    @property
    def execution_count(self) -> int:
        return len(self.execution_results)

    @property
    def einher_count(self) -> int:
        return len(self.einhers)

    @property
    def validated_count(self) -> int:
        return len(self.validated)

    @property
    def rejected_count(self) -> int:
        return len(self.rejected)

    # ==================================================
    # SERIALIZATION
    # ==================================================

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target.to_dict(),
            "index": self.index,
            "success": self.success,
            "errors": list(self.errors),
            "execution_count": self.execution_count,
            "einher_count": self.einher_count,
            "validated_count": self.validated_count,
            "rejected_count": self.rejected_count,
            "state": self.state.to_dict() if hasattr(self.state, "to_dict") else {},
            "metadata": dict(self.metadata),
            "export_paths": dict(self.export_paths),
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat(),
        }

    # ==================================================
    # PYTHON PROTOCOL
    # ==================================================

    def __repr__(self) -> str:
        return (
            "DiscoveryPairResult("
            f"pair='{self.pair_key}', "
            f"success={self.success}, "
            f"einhers={self.einher_count}"
            ")"
        )
