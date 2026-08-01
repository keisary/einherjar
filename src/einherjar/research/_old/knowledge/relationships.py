# knowledge/relationships.py
"""
==========================================================
Knowledge Relationships
==========================================================

Relations explicites entre les objets du corpus.

Le module ne calcule pas la connaissance :
- il enregistre les liens,
- les regroupe,
- les normalise.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

from .similarity import SimilarityScore

__all__ = [
    "RelationshipKind",
    "Relationship",
    "RelationshipStore",
    "RelationshipBuilder",
]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


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


def _to_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "to_dict") and callable(value.to_dict):
        value = value.to_dict()
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _normalize_text(value: Any) -> str:
    return str(value or "").strip()


class RelationshipKind(str, Enum):
    RELATED = "related"
    SIMILAR = "similar"
    VERY_SIMILAR = "very_similar"
    DUPLICATE = "duplicate"
    PARENT = "parent"
    CHILD = "child"
    SIBLING = "sibling"
    SAME_FAMILY = "same_family"
    SAME_PROFILE = "same_profile"
    SAME_KIND = "same_kind"
    COMPLEMENTARY = "complementary"
    CONFLICT = "conflict"
    DERIVED = "derived"
    ANCESTOR = "ancestor"
    DESCENDANT = "descendant"


@dataclass(frozen=True, slots=True)
class Relationship:
    """
    Relation entre deux objets.
    """

    source: str
    target: str
    kind: RelationshipKind | str
    weight: float = 1.0
    score: float = 0.0
    directed: bool = False
    source_kind: str = ""
    target_kind: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", _normalize_text(self.source))
        object.__setattr__(self, "target", _normalize_text(self.target))
        object.__setattr__(self, "kind", RelationshipKind(str(self.kind)) if not isinstance(self.kind, RelationshipKind) and str(self.kind) in RelationshipKind._value2member_map_ else self.kind)
        object.__setattr__(self, "weight", max(0.0, float(self.weight)))
        object.__setattr__(self, "score", float(self.score))
        object.__setattr__(self, "directed", _coerce_bool(self.directed, False))
        object.__setattr__(self, "source_kind", _normalize_text(self.source_kind))
        object.__setattr__(self, "target_kind", _normalize_text(self.target_kind))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def is_strong(self) -> bool:
        return self.weight >= 0.75 or self.score >= 0.75

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "kind": self.kind.value if isinstance(self.kind, RelationshipKind) else str(self.kind),
            "weight": self.weight,
            "score": self.score,
            "directed": self.directed,
            "source_kind": self.source_kind,
            "target_kind": self.target_kind,
            "metadata": dict(self.metadata),
            "created_at": self.created_at.isoformat(),
        }


@dataclass(slots=True)
class RelationshipStore:
    """
    Stockage en mémoire des relations.
    """

    relationships: list[Relationship] = field(default_factory=list)
    _outgoing: dict[str, list[Relationship]] = field(default_factory=lambda: defaultdict(list))
    _incoming: dict[str, list[Relationship]] = field(default_factory=lambda: defaultdict(list))

    def add(self, relationship: Relationship) -> Relationship:
        self.relationships.append(relationship)
        self._outgoing[relationship.source].append(relationship)
        self._incoming[relationship.target].append(relationship)
        if not relationship.directed:
            mirror = Relationship(
                source=relationship.target,
                target=relationship.source,
                kind=relationship.kind,
                weight=relationship.weight,
                score=relationship.score,
                directed=False,
                source_kind=relationship.target_kind,
                target_kind=relationship.source_kind,
                metadata=dict(relationship.metadata),
            )
            self._outgoing[mirror.source].append(mirror)
            self._incoming[mirror.target].append(mirror)
        return relationship

    def link(
        self,
        source: str,
        target: str,
        kind: RelationshipKind | str,
        *,
        weight: float = 1.0,
        score: float = 0.0,
        directed: bool = False,
        source_kind: str = "",
        target_kind: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> Relationship:
        relationship = Relationship(
            source=source,
            target=target,
            kind=kind,
            weight=weight,
            score=score,
            directed=directed,
            source_kind=source_kind,
            target_kind=target_kind,
            metadata=dict(metadata or {}),
        )
        return self.add(relationship)

    def between(self, source: str, target: str) -> tuple[Relationship, ...]:
        source = _normalize_text(source)
        target = _normalize_text(target)
        return tuple(
            rel
            for rel in self.relationships
            if (rel.source == source and rel.target == target) or (not rel.directed and rel.source == target and rel.target == source)
        )

    def outgoing(self, source: str) -> tuple[Relationship, ...]:
        return tuple(self._outgoing.get(_normalize_text(source), ()))

    def incoming(self, target: str) -> tuple[Relationship, ...]:
        return tuple(self._incoming.get(_normalize_text(target), ()))

    def by_kind(self, kind: RelationshipKind | str) -> tuple[Relationship, ...]:
        kind_value = kind.value if isinstance(kind, RelationshipKind) else str(kind)
        return tuple(rel for rel in self.relationships if (rel.kind.value if isinstance(rel.kind, RelationshipKind) else str(rel.kind)) == kind_value)

    def counts(self) -> dict[str, int]:
        counter = Counter(
            rel.kind.value if isinstance(rel.kind, RelationshipKind) else str(rel.kind)
            for rel in self.relationships
        )
        return dict(counter)

    def to_dict(self) -> dict[str, Any]:
        return {
            "relationships": [rel.to_dict() for rel in self.relationships],
            "counts": self.counts(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RelationshipStore":
        store = cls()
        for item in data.get("relationships", []):
            created_at = item.get("created_at")
            if isinstance(created_at, str) and created_at:
                created_at = datetime.fromisoformat(created_at)
            else:
                created_at = _utc_now()
            store.relationships.append(
                Relationship(
                    source=item.get("source", ""),
                    target=item.get("target", ""),
                    kind=item.get("kind", RelationshipKind.RELATED.value),
                    weight=_coerce_float(item.get("weight"), 1.0),
                    score=_coerce_float(item.get("score"), 0.0),
                    directed=_coerce_bool(item.get("directed"), False),
                    source_kind=item.get("source_kind", ""),
                    target_kind=item.get("target_kind", ""),
                    metadata=_to_mapping(item.get("metadata", {})),
                    created_at=created_at,
                )
            )
        for rel in store.relationships:
            store._outgoing[rel.source].append(rel)
            store._incoming[rel.target].append(rel)
        return store

    def __len__(self) -> int:
        return len(self.relationships)

    def __iter__(self):
        return iter(self.relationships)


class RelationshipBuilder:
    """
    Construit des relations à partir de similarités ou de métadonnées.
    """

    @staticmethod
    def from_similarity(
        source: str,
        target: str,
        similarity: SimilarityScore,
        *,
        source_kind: str = "",
        target_kind: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> Relationship:
        if similarity.score >= 0.95:
            kind = RelationshipKind.DUPLICATE
        elif similarity.score >= 0.85:
            kind = RelationshipKind.VERY_SIMILAR
        elif similarity.score >= 0.65:
            kind = RelationshipKind.SIMILAR
        else:
            kind = RelationshipKind.RELATED

        return Relationship(
            source=source,
            target=target,
            kind=kind,
            weight=similarity.score,
            score=similarity.score,
            directed=False,
            source_kind=source_kind,
            target_kind=target_kind,
            metadata={
                **dict(metadata or {}),
                "similarity": similarity.to_dict(),
            },
        )

    @staticmethod
    def sibling(source: str, target: str, *, weight: float = 0.75, metadata: Mapping[str, Any] | None = None) -> Relationship:
        return Relationship(source=source, target=target, kind=RelationshipKind.SIBLING, weight=weight, score=weight, metadata=dict(metadata or {}))

    @staticmethod
    def parent(source: str, target: str, *, weight: float = 1.0, metadata: Mapping[str, Any] | None = None) -> Relationship:
        return Relationship(source=source, target=target, kind=RelationshipKind.PARENT, weight=weight, score=weight, directed=True, metadata=dict(metadata or {}))

    @staticmethod
    def child(source: str, target: str, *, weight: float = 1.0, metadata: Mapping[str, Any] | None = None) -> Relationship:
        return Relationship(source=source, target=target, kind=RelationshipKind.CHILD, weight=weight, score=weight, directed=True, metadata=dict(metadata or {}))