# execution/knowledge.py
"""
==========================================================
Execution Knowledge
==========================================================

Mémoire sérialisable de ce que la phase Execution apprend
sur les stratégies.

Le module conserve des entrées compactes et réutilisables :
- fingerprints,
- métriques d'exécution,
- profil,
- diagnostics,
- MAE/MFE,
- métadonnées.

Il peut rester en RAM ou être exporté en dict / JSON.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timezone
from typing import Any
from typing import Iterable
from typing import Mapping
from typing import Sequence

from .execution_report import ExecutionResult
from .fingerprint import ExecutionFingerprint
from .mae_mfe import MAEMFESummary
from .profiler import ExecutionProfile
from .diagnostics import ExecutionDiagnostics

__all__ = [
    "ExecutionKnowledgeEntry",
    "ExecutionKnowledgeSummary",
    "ExecutionKnowledge",
]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _to_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}

    if hasattr(value, "to_dict") and callable(value.to_dict):
        value = value.to_dict()

    if isinstance(value, Mapping):
        return dict(value)

    return {}


def _coerce_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)


@dataclass(frozen=True, slots=True)
class ExecutionKnowledgeEntry:
    """
    Entrée compacte de mémoire d'exécution.
    """

    subject_fingerprint: str
    execution_fingerprint: str

    candidate_fingerprint: str | None = None
    hypothesis_fingerprint: str | None = None

    direction: str = "long"
    quantity: float = 1.0

    trade_count: int = 0
    total_pnl: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0

    healthy: bool = True
    issue_count: int = 0

    profile_name: str | None = None
    profile_description: str | None = None

    profile: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    mae_mfe: dict[str, Any] = field(default_factory=dict)

    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        object.__setattr__(self, "subject_fingerprint", str(self.subject_fingerprint))
        object.__setattr__(self, "execution_fingerprint", str(self.execution_fingerprint))
        object.__setattr__(self, "candidate_fingerprint", self.candidate_fingerprint)
        object.__setattr__(self, "hypothesis_fingerprint", self.hypothesis_fingerprint)
        object.__setattr__(self, "direction", str(self.direction).strip().lower() or "long")
        object.__setattr__(self, "quantity", max(0.0, float(self.quantity)))
        object.__setattr__(self, "trade_count", max(0, _coerce_int(self.trade_count, 0)))
        object.__setattr__(self, "total_pnl", float(self.total_pnl))
        object.__setattr__(self, "win_rate", min(1.0, max(0.0, float(self.win_rate))))
        object.__setattr__(self, "profit_factor", float(self.profit_factor))
        object.__setattr__(self, "expectancy", float(self.expectancy))
        object.__setattr__(self, "healthy", _coerce_bool(self.healthy, True))
        object.__setattr__(self, "issue_count", max(0, _coerce_int(self.issue_count, 0)))
        object.__setattr__(self, "profile_name", self.profile_name)
        object.__setattr__(self, "profile_description", self.profile_description)
        object.__setattr__(self, "profile", dict(self.profile))
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))
        object.__setattr__(self, "mae_mfe", dict(self.mae_mfe))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject_fingerprint": self.subject_fingerprint,
            "execution_fingerprint": self.execution_fingerprint,
            "candidate_fingerprint": self.candidate_fingerprint,
            "hypothesis_fingerprint": self.hypothesis_fingerprint,
            "direction": self.direction,
            "quantity": self.quantity,
            "trade_count": self.trade_count,
            "total_pnl": self.total_pnl,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "expectancy": self.expectancy,
            "healthy": self.healthy,
            "issue_count": self.issue_count,
            "profile_name": self.profile_name,
            "profile_description": self.profile_description,
            "profile": dict(self.profile),
            "diagnostics": dict(self.diagnostics),
            "mae_mfe": dict(self.mae_mfe),
            "metadata": dict(self.metadata),
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_result(cls, result: ExecutionResult) -> "ExecutionKnowledgeEntry":
        profile = result.profile.to_dict() if result.profile is not None else {}
        diagnostics = result.diagnostics.to_dict() if result.diagnostics is not None else {}
        mae_mfe = result.mae_mfe.to_dict() if result.mae_mfe is not None else {}

        return cls(
            subject_fingerprint=result.subject_fingerprint,
            execution_fingerprint=result.execution_fingerprint.digest,
            candidate_fingerprint=getattr(result.candidate, "fingerprint", None),
            hypothesis_fingerprint=getattr(result.hypothesis, "fingerprint", None),
            direction=result.replay.metrics.direction,
            quantity=result.replay.metrics.quantity,
            trade_count=result.trade_count,
            total_pnl=result.total_pnl,
            win_rate=result.win_rate,
            profit_factor=result.replay.metrics.profit_factor,
            expectancy=result.replay.metrics.expectancy,
            healthy=result.healthy,
            issue_count=result.issue_count,
            profile_name=(result.profile.name if result.profile is not None else None),
            profile_description=(result.profile.description if result.profile is not None else None),
            profile=profile,
            diagnostics=diagnostics,
            mae_mfe=mae_mfe,
            metadata=dict(result.metadata),
            created_at=result.created_at,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExecutionKnowledgeEntry":
        created_at = data.get("created_at")
        if isinstance(created_at, str) and created_at:
            created_at = datetime.fromisoformat(created_at)
        else:
            created_at = _utc_now()

        return cls(
            subject_fingerprint=data.get("subject_fingerprint", ""),
            execution_fingerprint=data.get("execution_fingerprint", ""),
            candidate_fingerprint=data.get("candidate_fingerprint"),
            hypothesis_fingerprint=data.get("hypothesis_fingerprint"),
            direction=data.get("direction", "long"),
            quantity=_coerce_float(data.get("quantity"), 1.0),
            trade_count=_coerce_int(data.get("trade_count"), 0),
            total_pnl=_coerce_float(data.get("total_pnl"), 0.0),
            win_rate=_coerce_float(data.get("win_rate"), 0.0),
            profit_factor=_coerce_float(data.get("profit_factor"), 0.0),
            expectancy=_coerce_float(data.get("expectancy"), 0.0),
            healthy=_coerce_bool(data.get("healthy"), True),
            issue_count=_coerce_int(data.get("issue_count"), 0),
            profile_name=data.get("profile_name"),
            profile_description=data.get("profile_description"),
            profile=_to_mapping(data.get("profile", {})),
            diagnostics=_to_mapping(data.get("diagnostics", {})),
            mae_mfe=_to_mapping(data.get("mae_mfe", {})),
            metadata=_to_mapping(data.get("metadata", {})),
            created_at=created_at,
        )


@dataclass(frozen=True, slots=True)
class ExecutionKnowledgeSummary:
    """
    Vue compacte de la mémoire d'exécution.
    """

    entry_count: int
    subject_count: int
    healthy_count: int
    unhealthy_count: int

    total_trade_count: int
    total_pnl: float
    average_pnl: float

    best_subject_fingerprint: str | None = None
    best_execution_fingerprint: str | None = None

    direction_counts: dict[str, int] = field(default_factory=dict)
    profile_counts: dict[str, int] = field(default_factory=dict)
    issue_counts: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "entry_count", max(0, _coerce_int(self.entry_count, 0)))
        object.__setattr__(self, "subject_count", max(0, _coerce_int(self.subject_count, 0)))
        object.__setattr__(self, "healthy_count", max(0, _coerce_int(self.healthy_count, 0)))
        object.__setattr__(self, "unhealthy_count", max(0, _coerce_int(self.unhealthy_count, 0)))
        object.__setattr__(self, "total_trade_count", max(0, _coerce_int(self.total_trade_count, 0)))
        object.__setattr__(self, "total_pnl", float(self.total_pnl))
        object.__setattr__(self, "average_pnl", float(self.average_pnl))
        object.__setattr__(self, "direction_counts", dict(self.direction_counts))
        object.__setattr__(self, "profile_counts", dict(self.profile_counts))
        object.__setattr__(self, "issue_counts", dict(self.issue_counts))

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_count": self.entry_count,
            "subject_count": self.subject_count,
            "healthy_count": self.healthy_count,
            "unhealthy_count": self.unhealthy_count,
            "total_trade_count": self.total_trade_count,
            "total_pnl": self.total_pnl,
            "average_pnl": self.average_pnl,
            "best_subject_fingerprint": self.best_subject_fingerprint,
            "best_execution_fingerprint": self.best_execution_fingerprint,
            "direction_counts": dict(self.direction_counts),
            "profile_counts": dict(self.profile_counts),
            "issue_counts": dict(self.issue_counts),
        }


@dataclass(slots=True)
class ExecutionKnowledge:
    """
    Mémoire d'exécution sérialisable.
    """

    name: str = "execution_knowledge"
    metadata: dict[str, Any] = field(default_factory=dict)

    entries: list[ExecutionKnowledgeEntry] = field(default_factory=list)
    subject_index: dict[str, int] = field(default_factory=dict)

    direction_counts: Counter[str] = field(default_factory=Counter)
    profile_counts: Counter[str] = field(default_factory=Counter)
    issue_counts: Counter[str] = field(default_factory=Counter)

    healthy_count: int = 0
    unhealthy_count: int = 0

    total_trade_count: int = 0
    total_pnl: float = 0.0

    best_subject_fingerprint: str | None = None
    best_execution_fingerprint: str | None = None
    best_total_pnl: float = float("-inf")
    last_seen_at: datetime | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", str(self.name).strip() or "execution_knowledge")
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "entries", list(self.entries))
        object.__setattr__(self, "subject_index", dict(self.subject_index))
        object.__setattr__(self, "direction_counts", Counter(self.direction_counts))
        object.__setattr__(self, "profile_counts", Counter(self.profile_counts))
        object.__setattr__(self, "issue_counts", Counter(self.issue_counts))

    @property
    def entry_count(self) -> int:
        return len(self.entries)

    @property
    def subject_count(self) -> int:
        return len(self.subject_index)

    @property
    def average_pnl(self) -> float:
        if self.entry_count == 0:
            return 0.0
        return self.total_pnl / self.entry_count

    @property
    def healthy_rate(self) -> float:
        if self.entry_count == 0:
            return 0.0
        return self.healthy_count / self.entry_count

    @property
    def summary(self) -> ExecutionKnowledgeSummary:
        return ExecutionKnowledgeSummary(
            entry_count=self.entry_count,
            subject_count=self.subject_count,
            healthy_count=self.healthy_count,
            unhealthy_count=self.unhealthy_count,
            total_trade_count=self.total_trade_count,
            total_pnl=self.total_pnl,
            average_pnl=self.average_pnl,
            best_subject_fingerprint=self.best_subject_fingerprint,
            best_execution_fingerprint=self.best_execution_fingerprint,
            direction_counts=dict(self.direction_counts),
            profile_counts=dict(self.profile_counts),
            issue_counts=dict(self.issue_counts),
        )

    def remember(self, result: ExecutionResult) -> ExecutionKnowledgeEntry:
        entry = ExecutionKnowledgeEntry.from_result(result)
        self.entries.append(entry)
        self.subject_index[entry.subject_fingerprint] = len(self.entries) - 1

        self.direction_counts[entry.direction] += 1
        if entry.profile_name:
            self.profile_counts[entry.profile_name] += 1

        if entry.healthy:
            self.healthy_count += 1
        else:
            self.unhealthy_count += 1
            if entry.diagnostics:
                for issue in entry.diagnostics.get("issues", []):
                    code = issue.get("code")
                    if code:
                        self.issue_counts[str(code)] += 1

        self.total_trade_count += entry.trade_count
        self.total_pnl += entry.total_pnl
        self.last_seen_at = entry.created_at

        if entry.total_pnl > self.best_total_pnl:
            self.best_total_pnl = entry.total_pnl
            self.best_subject_fingerprint = entry.subject_fingerprint
            self.best_execution_fingerprint = entry.execution_fingerprint

        return entry

    def remember_result(self, result: ExecutionResult) -> ExecutionKnowledgeEntry:
        return self.remember(result)

    def get(self, subject_fingerprint: str) -> ExecutionKnowledgeEntry | None:
        index = self.subject_index.get(subject_fingerprint)
        if index is None:
            return None
        if index < 0 or index >= len(self.entries):
            return None
        return self.entries[index]

    def latest(self, subject_fingerprint: str) -> ExecutionKnowledgeEntry | None:
        return self.get(subject_fingerprint)

    def best_entry(self) -> ExecutionKnowledgeEntry | None:
        if not self.entries:
            return None
        return max(self.entries, key=lambda entry: entry.total_pnl)

    def top(self, n: int = 10) -> tuple[ExecutionKnowledgeEntry, ...]:
        n = max(1, _coerce_int(n, 10))
        return tuple(sorted(self.entries, key=lambda entry: (-entry.total_pnl, -entry.trade_count, entry.subject_fingerprint))[:n])

    def merge(self, other: "ExecutionKnowledge") -> None:
        for entry in other.entries:
            self.entries.append(entry)
            self.subject_index[entry.subject_fingerprint] = len(self.entries) - 1
            self.direction_counts[entry.direction] += 1
            if entry.profile_name:
                self.profile_counts[entry.profile_name] += 1
            if entry.healthy:
                self.healthy_count += 1
            else:
                self.unhealthy_count += 1
            self.total_trade_count += entry.trade_count
            self.total_pnl += entry.total_pnl
            if entry.diagnostics:
                for issue in entry.diagnostics.get("issues", []):
                    code = issue.get("code")
                    if code:
                        self.issue_counts[str(code)] += 1
            if self.last_seen_at is None or entry.created_at > self.last_seen_at:
                self.last_seen_at = entry.created_at
            if entry.total_pnl > self.best_total_pnl:
                self.best_total_pnl = entry.total_pnl
                self.best_subject_fingerprint = entry.subject_fingerprint
                self.best_execution_fingerprint = entry.execution_fingerprint

    def clear(self) -> None:
        self.entries.clear()
        self.subject_index.clear()
        self.direction_counts.clear()
        self.profile_counts.clear()
        self.issue_counts.clear()
        self.healthy_count = 0
        self.unhealthy_count = 0
        self.total_trade_count = 0
        self.total_pnl = 0.0
        self.best_subject_fingerprint = None
        self.best_execution_fingerprint = None
        self.best_total_pnl = float("-inf")
        self.last_seen_at = None

    def to_dict(self, *, summary_only: bool = False) -> dict[str, Any]:
        payload = {
            "name": self.name,
            "metadata": dict(self.metadata),
            "entries": [] if summary_only else [entry.to_dict() for entry in self.entries],
            "subject_index": dict(self.subject_index),
            "direction_counts": dict(self.direction_counts),
            "profile_counts": dict(self.profile_counts),
            "issue_counts": dict(self.issue_counts),
            "healthy_count": self.healthy_count,
            "unhealthy_count": self.unhealthy_count,
            "total_trade_count": self.total_trade_count,
            "total_pnl": self.total_pnl,
            "best_subject_fingerprint": self.best_subject_fingerprint,
            "best_execution_fingerprint": self.best_execution_fingerprint,
            "best_total_pnl": None if self.best_total_pnl == float("-inf") else self.best_total_pnl,
            "last_seen_at": self.last_seen_at.isoformat() if self.last_seen_at else None,
            "summary": self.summary.to_dict(),
        }
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExecutionKnowledge":
        knowledge = cls(
            name=data.get("name", "execution_knowledge"),
            metadata=_to_mapping(data.get("metadata", {})),
            subject_index={str(k): int(v) for k, v in dict(data.get("subject_index", {})).items()},
            direction_counts=Counter(_to_mapping(data.get("direction_counts", {}))),
            profile_counts=Counter(_to_mapping(data.get("profile_counts", {}))),
            issue_counts=Counter(_to_mapping(data.get("issue_counts", {}))),
            healthy_count=_coerce_int(data.get("healthy_count"), 0),
            unhealthy_count=_coerce_int(data.get("unhealthy_count"), 0),
            total_trade_count=_coerce_int(data.get("total_trade_count"), 0),
            total_pnl=_coerce_float(data.get("total_pnl"), 0.0),
            best_subject_fingerprint=data.get("best_subject_fingerprint"),
            best_execution_fingerprint=data.get("best_execution_fingerprint"),
            best_total_pnl=_coerce_float(data.get("best_total_pnl"), float("-inf")),
            last_seen_at=(
                datetime.fromisoformat(data["last_seen_at"])
                if data.get("last_seen_at")
                else None
            ),
        )

        knowledge.entries = [
            ExecutionKnowledgeEntry.from_dict(item)
            for item in data.get("entries", [])
        ]

        if not knowledge.subject_index:
            knowledge.subject_index = {
                entry.subject_fingerprint: index
                for index, entry in enumerate(knowledge.entries)
            }

        return knowledge

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)

    def __repr__(self) -> str:
        return (
            "ExecutionKnowledge("
            f"entries={len(self.entries)}, "
            f"subjects={self.subject_count}, "
            f"pnl={self.total_pnl:.4f}"
            ")"
        )