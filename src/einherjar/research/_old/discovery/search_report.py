"""
==========================================================
Search Report
==========================================================

Journal statistique de la phase Discovery.

Ce module enregistre ce que le moteur de recherche produit
et consomme pendant l'exploration :
- hypothèses générées,
- actions appliquées,
- familles touchées,
- profondeur,
- budget consommé,
- rejets et doublons,
- meilleures découvertes.

Le SearchReport ne génère rien et ne valide rien.
Il observe et agrège.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from typing import Iterable
from typing import Mapping

from models.enums import EconomicFamily

from .generator import GenerationResult
from .search_budget import BudgetSnapshot


__all__ = [
    "SearchEvent",
    "SearchReport",
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


def _coerce_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default

    try:
        return int(value)
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


def _family_key(family: EconomicFamily | str | None) -> str:
    if family is None:
        return "unknown"

    if isinstance(family, EconomicFamily):
        return family.value

    return str(family).strip().lower() or "unknown"


def _family_from_key(value: str) -> EconomicFamily | str:
    try:
        return EconomicFamily(value)
    except ValueError:
        try:
            return EconomicFamily[value.upper()]
        except KeyError:
            return value


@dataclass(frozen=True, slots=True)
class SearchEvent:
    """
    Evénement de recherche individuel.

    Un SearchEvent conserve une vue compacte d'une
    transformation ou d'un résultat produit par Discovery.
    """

    index: int
    action: str
    fingerprint: str

    parent_fingerprint: str | None = None

    family: str = "unknown"
    depth: int = 0
    condition_count: int = 0

    score: float = 0.0
    accepted: bool = True

    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    created_at: datetime = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        object.__setattr__(self, "action", str(self.action).strip().lower())
        object.__setattr__(self, "family", str(self.family).strip().lower() or "unknown")
        object.__setattr__(self, "depth", max(0, _coerce_int(self.depth, 0)))
        object.__setattr__(self, "condition_count", max(0, _coerce_int(self.condition_count, 0)))
        object.__setattr__(self, "score", float(self.score))
        object.__setattr__(self, "accepted", _coerce_bool(self.accepted, True))
        object.__setattr__(self, "reason", str(self.reason))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "action": self.action,
            "fingerprint": self.fingerprint,
            "parent_fingerprint": self.parent_fingerprint,
            "family": self.family,
            "depth": self.depth,
            "condition_count": self.condition_count,
            "score": self.score,
            "accepted": self.accepted,
            "reason": self.reason,
            "metadata": dict(self.metadata),
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SearchEvent":
        created_at = data.get("created_at")
        return cls(
            index=_coerce_int(data.get("index"), 0),
            action=data.get("action", ""),
            fingerprint=data.get("fingerprint", ""),
            parent_fingerprint=data.get("parent_fingerprint"),
            family=data.get("family", "unknown"),
            depth=_coerce_int(data.get("depth"), 0),
            condition_count=_coerce_int(data.get("condition_count"), 0),
            score=_coerce_float(data.get("score"), 0.0),
            accepted=_coerce_bool(data.get("accepted"), True),
            reason=data.get("reason", ""),
            metadata=_to_mapping(data.get("metadata", {})),
            created_at=(
                datetime.fromisoformat(created_at)
                if isinstance(created_at, str) and created_at
                else _utc_now()
            ),
        )

    def __hash__(self) -> int:
        return hash(self.fingerprint)

    def __repr__(self) -> str:
        return (
            "SearchEvent("
            f"index={self.index}, "
            f"action='{self.action}', "
            f"family='{self.family}', "
            f"score={self.score:.4f}, "
            f"accepted={self.accepted}"
            ")"
        )


@dataclass(slots=True)
class SearchReport:
    """
    Rapport cumulatif de la phase Discovery.

    Le report agrège les événements produits pendant la
    recherche sans intervenir dans la logique de décision.
    """

    name: str = "discovery"

    metadata: dict[str, Any] = field(default_factory=dict)

    started_at: datetime | None = None
    finished_at: datetime | None = None

    events: list[SearchEvent] = field(default_factory=list)
    budget_snapshots: list[BudgetSnapshot] = field(default_factory=list)

    total_generated: int = 0
    total_seeded: int = 0
    total_expanded: int = 0
    total_mutated: int = 0
    total_replaced: int = 0
    total_pruned: int = 0

    total_accepted: int = 0
    total_rejected: int = 0
    total_duplicates: int = 0
    total_pruned_by_budget: int = 0

    family_counts: Counter[str] = field(default_factory=Counter)
    action_counts: Counter[str] = field(default_factory=Counter)
    depth_counts: Counter[int] = field(default_factory=Counter)

    feature_counts: Counter[int] = field(default_factory=Counter)

    best_fingerprint: str | None = None
    best_score: float = float("-inf")
    best_event_index: int | None = None

    last_reason: str | None = None
    exhausted: bool = False
    stopped_reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", str(self.name).strip() or "discovery")
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "events", list(self.events))
        object.__setattr__(self, "budget_snapshots", list(self.budget_snapshots))
        object.__setattr__(self, "family_counts", Counter(self.family_counts))
        object.__setattr__(self, "action_counts", Counter(self.action_counts))
        object.__setattr__(self, "depth_counts", Counter(self.depth_counts))
        object.__setattr__(self, "feature_counts", Counter(self.feature_counts))

    # ==================================================
    # LIFECYCLE
    # ==================================================

    def start(self) -> None:
        if self.started_at is None:
            self.started_at = _utc_now()
        self.finished_at = None
        self.exhausted = False
        self.stopped_reason = None

    def finish(self, reason: str | None = None) -> None:
        if self.started_at is None:
            self.start()

        self.finished_at = _utc_now()
        self.exhausted = True
        self.stopped_reason = reason or self.stopped_reason or "finished"

    def reset(self) -> None:
        self.started_at = None
        self.finished_at = None

        self.events.clear()
        self.budget_snapshots.clear()

        self.total_generated = 0
        self.total_seeded = 0
        self.total_expanded = 0
        self.total_mutated = 0
        self.total_replaced = 0
        self.total_pruned = 0

        self.total_accepted = 0
        self.total_rejected = 0
        self.total_duplicates = 0
        self.total_pruned_by_budget = 0

        self.family_counts.clear()
        self.action_counts.clear()
        self.depth_counts.clear()
        self.feature_counts.clear()

        self.best_fingerprint = None
        self.best_score = float("-inf")
        self.best_event_index = None

        self.last_reason = None
        self.exhausted = False
        self.stopped_reason = None

    # ==================================================
    # PROPERTIES
    # ==================================================

    @property
    def event_count(self) -> int:
        return len(self.events)

    @property
    def duration_seconds(self) -> float:
        if self.started_at is None:
            return 0.0

        end = self.finished_at or _utc_now()
        return max(0.0, (end - self.started_at).total_seconds())

    @property
    def is_running(self) -> bool:
        return self.started_at is not None and self.finished_at is None

    @property
    def success_rate(self) -> float:
        if self.total_generated == 0:
            return 0.0
        return self.total_accepted / self.total_generated

    @property
    def rejection_rate(self) -> float:
        if self.total_generated == 0:
            return 0.0
        return self.total_rejected / self.total_generated

    @property
    def duplicate_rate(self) -> float:
        if self.total_generated == 0:
            return 0.0
        return self.total_duplicates / self.total_generated

    @property
    def average_score(self) -> float:
        if not self.events:
            return 0.0
        return sum(event.score for event in self.events) / len(self.events)

    @property
    def top_family(self) -> str | None:
        if not self.family_counts:
            return None
        return max(self.family_counts.items(), key=lambda item: item[1])[0]

    @property
    def top_action(self) -> str | None:
        if not self.action_counts:
            return None
        return max(self.action_counts.items(), key=lambda item: item[1])[0]

    @property
    def depth_max(self) -> int:
        if not self.depth_counts:
            return 0
        return max(self.depth_counts)

    @property
    def summary(self) -> dict[str, Any]:
        return self.to_dict(summary_only=True)

    # ==================================================
    # RECORDING
    # ==================================================

    def record_generation(
        self,
        result: GenerationResult | Mapping[str, Any],
        *,
        depth: int = 0,
        accepted: bool = True,
        reason: str = "",
        duplicate: bool = False,
        budget_exhausted: bool = False,
        metadata: Mapping[str, Any] | None = None,
    ) -> SearchEvent:
        if isinstance(result, GenerationResult):
            data = result.to_dict()
        else:
            data = dict(result)

        hypothesis = data.get("hypothesis")
        if isinstance(hypothesis, Mapping):
            fingerprint = str(data.get("fingerprint") or hypothesis.get("fingerprint") or "")
        else:
            fingerprint = str(data.get("fingerprint") or "")

        if not fingerprint:
            fingerprint = str(data.get("fingerprint") or "")

        details = _to_mapping(data.get("details", {}))

        family = _family_key(
            data.get("family")
            or details.get("family")
            or details.get("target_family")
        )

        action = str(
            data.get("action")
            or details.get("action")
            or "unknown"
        ).strip().lower()

        score = _coerce_float(
            data.get("score")
            or details.get("score")
            or details.get("selection_score")
            or 0.0,
            0.0,
        )

        feature_index = data.get("feature_index", details.get("feature_index"))

        condition_count = _coerce_int(
            data.get("condition_count")
            or details.get("before_size")
            or 0,
            0,
        )
        if condition_count == 0 and isinstance(hypothesis, Mapping):
            conditions = hypothesis.get("conditions")
            if isinstance(conditions, list):
                condition_count = len(conditions)

        event = SearchEvent(
            index=self.event_count,
            action=action,
            fingerprint=fingerprint,
            parent_fingerprint=data.get("parent_fingerprint"),
            family=family,
            depth=depth,
            condition_count=condition_count,
            score=score,
            accepted=accepted,
            reason=reason,
            metadata={
                **_to_mapping(metadata),
                "duplicate": duplicate,
                "budget_exhausted": budget_exhausted,
                "source": "generation",
                "result": data,
            },
        )
        self._commit_event(event, feature_index=feature_index, duplicate=duplicate, budget_exhausted=budget_exhausted)
        return event

    def record_event(
        self,
        event: SearchEvent,
    ) -> None:
        if not isinstance(event, SearchEvent):
            raise TypeError("event must be a SearchEvent.")

        self._commit_event(event)

    def record_budget(
        self,
        snapshot: BudgetSnapshot | Mapping[str, Any],
    ) -> None:
        if isinstance(snapshot, BudgetSnapshot):
            self.budget_snapshots.append(snapshot)
            return

        mapping = dict(snapshot)
        self.budget_snapshots.append(
            BudgetSnapshot(
                configured_total=_coerce_int(mapping.get("configured_total"), 0),
                configured_frontier=_coerce_int(mapping.get("configured_frontier"), 0),
                configured_family=_coerce_int(mapping.get("configured_family"), 0),
                configured_depth=_coerce_int(mapping.get("configured_depth"), 0),
                configured_conditions=_coerce_int(mapping.get("configured_conditions"), 0),
                generated_total=_coerce_int(mapping.get("generated_total"), 0),
                active_total=_coerce_int(mapping.get("active_total"), 0),
                elapsed_seconds=_coerce_float(mapping.get("elapsed_seconds"), 0.0),
                remaining_total=mapping.get("remaining_total"),
                remaining_frontier=mapping.get("remaining_frontier"),
                exhausted=_coerce_bool(mapping.get("exhausted"), False),
                reason=mapping.get("reason"),
                family_generated=_to_mapping(mapping.get("family_generated", {})),
                depth_generated=_to_mapping(mapping.get("depth_generated", {})),
            )
        )

    def mark_accepted(self, fingerprint: str, score: float | None = None) -> None:
        self.total_accepted += 1
        if score is not None and score > self.best_score:
            self.best_score = float(score)
            self.best_fingerprint = fingerprint

    def mark_rejected(self, reason: str | None = None) -> None:
        self.total_rejected += 1
        if reason:
            self.last_reason = reason

    def mark_duplicate(self) -> None:
        self.total_duplicates += 1

    def mark_budget_pruned(self) -> None:
        self.total_pruned_by_budget += 1

    # ==================================================
    # INTERNAL COMMIT
    # ==================================================

    def _commit_event(
        self,
        event: SearchEvent,
        *,
        feature_index: int | None = None,
        duplicate: bool = False,
        budget_exhausted: bool = False,
    ) -> None:
        self.events.append(event)

        self.total_generated += 1
        self.action_counts[event.action] += 1
        self.family_counts[event.family] += 1
        self.depth_counts[event.depth] += 1

        if feature_index is not None:
            self.feature_counts[_coerce_int(feature_index, -1)] += 1

        if event.action == "seed":
            self.total_seeded += 1
        elif event.action == "expand":
            self.total_expanded += 1
        elif event.action == "mutate":
            self.total_mutated += 1
        elif event.action == "replace":
            self.total_replaced += 1
        elif event.action == "prune":
            self.total_pruned += 1

        if event.accepted:
            self.total_accepted += 1
        else:
            self.total_rejected += 1

        if duplicate:
            self.total_duplicates += 1

        if budget_exhausted:
            self.total_pruned_by_budget += 1
            self.exhausted = True
            self.stopped_reason = self.stopped_reason or "budget_exhausted"

        if event.score > self.best_score:
            self.best_score = float(event.score)
            self.best_fingerprint = event.fingerprint
            self.best_event_index = event.index

        if event.reason:
            self.last_reason = event.reason

    # ==================================================
    # SERIALIZATION
    # ==================================================

    def to_dict(self, *, summary_only: bool = False) -> dict[str, Any]:
        base = {
            "name": self.name,
            "metadata": dict(self.metadata),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "total_generated": self.total_generated,
            "total_seeded": self.total_seeded,
            "total_expanded": self.total_expanded,
            "total_mutated": self.total_mutated,
            "total_replaced": self.total_replaced,
            "total_pruned": self.total_pruned,
            "total_accepted": self.total_accepted,
            "total_rejected": self.total_rejected,
            "total_duplicates": self.total_duplicates,
            "total_pruned_by_budget": self.total_pruned_by_budget,
            "family_counts": dict(self.family_counts),
            "action_counts": dict(self.action_counts),
            "depth_counts": dict(self.depth_counts),
            "feature_counts": dict(self.feature_counts),
            "best_fingerprint": self.best_fingerprint,
            "best_score": None if self.best_score == float("-inf") else self.best_score,
            "best_event_index": self.best_event_index,
            "last_reason": self.last_reason,
            "exhausted": self.exhausted,
            "stopped_reason": self.stopped_reason,
            "duration_seconds": self.duration_seconds,
            "success_rate": self.success_rate,
            "rejection_rate": self.rejection_rate,
            "duplicate_rate": self.duplicate_rate,
            "average_score": self.average_score,
            "top_family": self.top_family,
            "top_action": self.top_action,
            "depth_max": self.depth_max,
            "budget_snapshots": [snapshot.to_dict() for snapshot in self.budget_snapshots],
        }

        if summary_only:
            return base

        base["events"] = [event.to_dict() for event in self.events]
        return base

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SearchReport":
        report = cls(
            name=data.get("name", "discovery"),
            metadata=_to_mapping(data.get("metadata", {})),
            started_at=(
                datetime.fromisoformat(data["started_at"])
                if data.get("started_at")
                else None
            ),
            finished_at=(
                datetime.fromisoformat(data["finished_at"])
                if data.get("finished_at")
                else None
            ),
            total_generated=_coerce_int(data.get("total_generated"), 0),
            total_seeded=_coerce_int(data.get("total_seeded"), 0),
            total_expanded=_coerce_int(data.get("total_expanded"), 0),
            total_mutated=_coerce_int(data.get("total_mutated"), 0),
            total_replaced=_coerce_int(data.get("total_replaced"), 0),
            total_pruned=_coerce_int(data.get("total_pruned"), 0),
            total_accepted=_coerce_int(data.get("total_accepted"), 0),
            total_rejected=_coerce_int(data.get("total_rejected"), 0),
            total_duplicates=_coerce_int(data.get("total_duplicates"), 0),
            total_pruned_by_budget=_coerce_int(data.get("total_pruned_by_budget"), 0),
            family_counts=Counter(_to_mapping(data.get("family_counts", {}))),
            action_counts=Counter(_to_mapping(data.get("action_counts", {}))),
            depth_counts=Counter({int(k): int(v) for k, v in _to_mapping(data.get("depth_counts", {})).items()}),
            feature_counts=Counter({int(k): int(v) for k, v in _to_mapping(data.get("feature_counts", {})).items()}),
            best_fingerprint=data.get("best_fingerprint"),
            best_score=_coerce_float(data.get("best_score"), float("-inf")),
            best_event_index=data.get("best_event_index"),
            last_reason=data.get("last_reason"),
            exhausted=_coerce_bool(data.get("exhausted"), False),
            stopped_reason=data.get("stopped_reason"),
        )

        report.budget_snapshots = [
            BudgetSnapshot.from_dict(snapshot) if hasattr(BudgetSnapshot, "from_dict") else BudgetSnapshot(
                configured_total=_coerce_int(snapshot.get("configured_total"), 0),
                configured_frontier=_coerce_int(snapshot.get("configured_frontier"), 0),
                configured_family=_coerce_int(snapshot.get("configured_family"), 0),
                configured_depth=_coerce_int(snapshot.get("configured_depth"), 0),
                configured_conditions=_coerce_int(snapshot.get("configured_conditions"), 0),
                generated_total=_coerce_int(snapshot.get("generated_total"), 0),
                active_total=_coerce_int(snapshot.get("active_total"), 0),
                elapsed_seconds=_coerce_float(snapshot.get("elapsed_seconds"), 0.0),
                remaining_total=snapshot.get("remaining_total"),
                remaining_frontier=snapshot.get("remaining_frontier"),
                exhausted=_coerce_bool(snapshot.get("exhausted"), False),
                reason=snapshot.get("reason"),
                family_generated=_to_mapping(snapshot.get("family_generated", {})),
                depth_generated=_to_mapping(snapshot.get("depth_generated", {})),
            )
            for snapshot in data.get("budget_snapshots", [])
        ]

        report.events = [SearchEvent.from_dict(event) for event in data.get("events", [])]
        return report

    # ==================================================
    # PYTHON PROTOCOL
    # ==================================================

    def __len__(self) -> int:
        return len(self.events)

    def __iter__(self):
        return iter(self.events)

    def __bool__(self) -> bool:
        return bool(self.events)

    def __repr__(self) -> str:
        return (
            "SearchReport("
            f"name='{self.name}', "
            f"events={len(self.events)}, "
            f"generated={self.total_generated}, "
            f"best_score={None if self.best_score == float('-inf') else round(self.best_score, 4)}"
            ")"
        )