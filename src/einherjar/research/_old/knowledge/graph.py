# knowledge/graph.py
"""
==========================================================
Knowledge Graph
==========================================================

Structure centrale du module knowledge.

Le graphe relie les objets du corpus par des noeuds et des
relations explicites.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from .fingerprints import FingerprintRegistry
from .fingerprints import KnowledgeFingerprint
from .fingerprints import build_knowledge_fingerprint
from .relationships import Relationship
from .relationships import RelationshipBuilder
from .relationships import RelationshipKind
from .relationships import RelationshipStore
from .similarity import SimilarityEngine

__all__ = [
    "KnowledgeNode",
    "KnowledgeGraph",
    "KnowledgeGraphBuilder",
]


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


def _fingerprint_value(obj: Any) -> str:
    if isinstance(obj, KnowledgeFingerprint):
        return obj.digest
    if hasattr(obj, "digest"):
        value = getattr(obj, "digest", None)
        if value:
            return str(value)
    return build_knowledge_fingerprint(obj).digest


def _kind_value(obj: Any) -> str:
    for attr in ("kind", "source_kind", "source_stage", "object_kind"):
        value = getattr(obj, attr, None)
        if value:
            return str(value).strip().lower()
    return "knowledge"


@dataclass(frozen=True, slots=True)
class KnowledgeNode:
    """
    Noeud du graphe de connaissance.
    """

    fingerprint: KnowledgeFingerprint
    kind: str = "knowledge"
    label: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _normalize_text(self.kind) or "knowledge")
        object.__setattr__(self, "label", _normalize_text(self.label))
        object.__setattr__(self, "payload", dict(self.payload))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def digest(self) -> str:
        return self.fingerprint.digest

    def to_dict(self) -> dict[str, Any]:
        return {
            "fingerprint": self.fingerprint.to_dict(),
            "kind": self.kind,
            "label": self.label,
            "payload": dict(self.payload),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "KnowledgeNode":
        return cls(
            fingerprint=KnowledgeFingerprint.from_dict(data["fingerprint"]),
            kind=data.get("kind", "knowledge"),
            label=data.get("label", ""),
            payload=_to_mapping(data.get("payload", {})),
            metadata=_to_mapping(data.get("metadata", {})),
        )


@dataclass(slots=True)
class KnowledgeGraph:
    """
    Graphe relationnel des objets du corpus.
    """

    nodes: dict[str, KnowledgeNode] = field(default_factory=dict)
    relationships: RelationshipStore = field(default_factory=RelationshipStore)
    fingerprints: FingerprintRegistry = field(default_factory=FingerprintRegistry)

    similarity_engine: SimilarityEngine = field(default_factory=SimilarityEngine)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_node(
        self,
        obj: Any,
        *,
        kind: str | None = None,
        label: str = "",
        tags: Sequence[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> KnowledgeNode:
        fp = build_knowledge_fingerprint(
            obj,
            kind=kind or _kind_value(obj),
            label=label,
            tags=tags,
            metadata=metadata,
        )
        node = KnowledgeNode(
            fingerprint=fp,
            kind=fp.kind,
            label=fp.label,
            payload=_to_mapping(obj),
            metadata=dict(metadata or {}),
        )
        self.nodes[node.digest] = node
        self.fingerprints.add(fp)
        return node

    def get_node(self, digest: str) -> KnowledgeNode | None:
        return self.nodes.get(str(digest))

    def add_relationship(self, relationship: Relationship) -> Relationship:
        return self.relationships.add(relationship)

    def relate(
        self,
        left: Any,
        right: Any,
        *,
        kind: RelationshipKind | str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Relationship:
        left_node = self.add_node(left)
        right_node = self.add_node(right)

        if kind is None:
            similarity = self.similarity_engine.compare(left, right)
            relationship = RelationshipBuilder.from_similarity(
                left_node.digest,
                right_node.digest,
                similarity,
                source_kind=left_node.kind,
                target_kind=right_node.kind,
                metadata=metadata,
            )
        else:
            relationship = Relationship(
                source=left_node.digest,
                target=right_node.digest,
                kind=kind,
                weight=1.0,
                score=1.0,
                source_kind=left_node.kind,
                target_kind=right_node.kind,
                metadata=dict(metadata or {}),
            )

        return self.add_relationship(relationship)

    def connect(
        self,
        source: Any,
        target: Any,
        *,
        kind: RelationshipKind | str = RelationshipKind.RELATED,
        weight: float = 1.0,
        score: float = 0.0,
        directed: bool = False,
        metadata: Mapping[str, Any] | None = None,
    ) -> Relationship:
        left_node = self.add_node(source)
        right_node = self.add_node(target)
        relationship = Relationship(
            source=left_node.digest,
            target=right_node.digest,
            kind=kind,
            weight=weight,
            score=score,
            directed=directed,
            source_kind=left_node.kind,
            target_kind=right_node.kind,
            metadata=dict(metadata or {}),
        )
        return self.add_relationship(relationship)

    def build_from_objects(
        self,
        objects: Iterable[Any],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> "KnowledgeGraph":
        objects = tuple(objects)
        for obj in objects:
            self.add_node(obj, metadata=metadata)

        for i in range(len(objects)):
            for j in range(i + 1, len(objects)):
                self.relate(objects[i], objects[j], metadata=metadata)

        return self

    def neighbors(self, digest: str, *, kinds: Sequence[RelationshipKind | str] | None = None) -> tuple[KnowledgeNode, ...]:
        digest = str(digest)
        allowed = None
        if kinds is not None:
            allowed = {
                kind.value if isinstance(kind, RelationshipKind) else str(kind)
                for kind in kinds
            }

        related: list[KnowledgeNode] = []
        for rel in self.relationships.outgoing(digest):
            rel_kind = rel.kind.value if isinstance(rel.kind, RelationshipKind) else str(rel.kind)
            if allowed is not None and rel_kind not in allowed:
                continue
            node = self.nodes.get(rel.target)
            if node is not None:
                related.append(node)
        return tuple(related)

    def related_nodes(self, digest: str) -> tuple[KnowledgeNode, ...]:
        return self.neighbors(digest)

    def edge_count(self) -> int:
        return len(self.relationships)

    def node_count(self) -> int:
        return len(self.nodes)

    def summary(self) -> dict[str, Any]:
        kinds = Counter(node.kind for node in self.nodes.values())
        relationship_counts = self.relationships.counts()
        return {
            "node_count": self.node_count(),
            "edge_count": self.edge_count(),
            "kinds": dict(kinds),
            "relationships": relationship_counts,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "relationships": self.relationships.to_dict(),
            "fingerprints": self.fingerprints.to_dict(),
            "summary": self.summary(),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "KnowledgeGraph":
        graph = cls(metadata=_to_mapping(data.get("metadata", {})))
        for item in data.get("nodes", []):
            node = KnowledgeNode.from_dict(item)
            graph.nodes[node.digest] = node
            graph.fingerprints.add(node.fingerprint)
        graph.relationships = RelationshipStore.from_dict(data.get("relationships", {}))
        graph.fingerprints = FingerprintRegistry.from_dict(data.get("fingerprints", {}))
        return graph

    def __len__(self) -> int:
        return len(self.nodes)

    def __iter__(self):
        return iter(self.nodes.values())

    def __repr__(self) -> str:
        return f"KnowledgeGraph(nodes={self.node_count()}, edges={self.edge_count()})"


class KnowledgeGraphBuilder:
    """
    Construit un graphe de connaissance.
    """

    @staticmethod
    def build(
        objects: Iterable[Any],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> KnowledgeGraph:
        graph = KnowledgeGraph(metadata=dict(metadata or {}))
        return graph.build_from_objects(objects, metadata=metadata)

    @staticmethod
    def from_objects(objects: Iterable[Any], *, metadata: Mapping[str, Any] | None = None) -> KnowledgeGraph:
        return KnowledgeGraphBuilder.build(objects, metadata=metadata)