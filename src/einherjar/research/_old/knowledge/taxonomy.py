# knowledge/taxonomy.py
"""
==========================================================
Knowledge Taxonomy
==========================================================

Classe et organise les objets du corpus selon une taxonomie
simple et stable.

Ce module ne décide pas :
- il classe,
- il étiquette,
- il structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "TaxonomyClassification",
    "TaxonomyNode",
    "TaxonomyEngine",
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
    return str(value or "").strip().lower()


def _object_kind(obj: Any) -> str:
    for attr in ("source_kind", "kind", "object_kind", "stage"):
        value = getattr(obj, attr, None)
        if value:
            return _normalize_text(value)
    return "knowledge"


def _family(obj: Any) -> str:
    for attr in ("family", "target_family"):
        value = getattr(obj, attr, None)
        if value:
            return _normalize_text(value)
    metadata = _to_mapping(getattr(obj, "metadata", None))
    for key in ("family", "target_family", "portfolio_family"):
        if key in metadata and metadata[key] is not None:
            value = _normalize_text(metadata[key])
            if value:
                return value
    return "unknown"


def _profile(obj: Any) -> str:
    for attr in ("profile_name", "profile"):
        value = getattr(obj, attr, None)
        if value:
            return _normalize_text(value)
    metadata = _to_mapping(getattr(obj, "metadata", None))
    for key in ("profile_name", "strategy_name", "einher_name"):
        if key in metadata and metadata[key] is not None:
            value = _normalize_text(metadata[key])
            if value:
                return value
    return "unknown"


@dataclass(frozen=True, slots=True)
class TaxonomyClassification:
    """
    Classification taxonomique d'un objet.
    """

    kind: str
    family: str
    profile: str
    labels: tuple[str, ...] = ()
    path: tuple[str, ...] = ()
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _normalize_text(self.kind) or "knowledge")
        object.__setattr__(self, "family", _normalize_text(self.family) or "unknown")
        object.__setattr__(self, "profile", _normalize_text(self.profile) or "unknown")
        object.__setattr__(self, "labels", tuple(sorted({str(label).strip().lower() for label in self.labels if str(label).strip()})))
        if not self.path:
            object.__setattr__(self, "path", (self.kind, self.family, self.profile))
        else:
            object.__setattr__(self, "path", tuple(_normalize_text(part) for part in self.path if _normalize_text(part)))
        object.__setattr__(self, "confidence", max(0.0, min(1.0, float(self.confidence))))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def path_text(self) -> str:
        return "/".join(self.path)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "family": self.family,
            "profile": self.profile,
            "labels": list(self.labels),
            "path": list(self.path),
            "confidence": self.confidence,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TaxonomyClassification":
        return cls(
            kind=data.get("kind", "knowledge"),
            family=data.get("family", "unknown"),
            profile=data.get("profile", "unknown"),
            labels=tuple(data.get("labels", ())),
            path=tuple(data.get("path", ())),
            confidence=float(data.get("confidence", 1.0)),
            metadata=_to_mapping(data.get("metadata", {})),
        )


@dataclass(frozen=True, slots=True)
class TaxonomyNode:
    """
    Noeud taxonomique.
    """

    name: str
    parent: str | None = None
    description: str = ""
    aliases: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _normalize_text(self.name) or "unknown")
        object.__setattr__(self, "parent", _normalize_text(self.parent) or None)
        object.__setattr__(self, "description", str(self.description).strip())
        object.__setattr__(self, "aliases", tuple(sorted({str(alias).strip().lower() for alias in self.aliases if str(alias).strip()})))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "parent": self.parent,
            "description": self.description,
            "aliases": list(self.aliases),
            "metadata": dict(self.metadata),
        }


class TaxonomyEngine:
    """
    Classe les objets du corpus.
    """

    def __init__(self, taxonomy: Iterable[TaxonomyNode] | None = None) -> None:
        self._nodes: dict[str, TaxonomyNode] = {}
        self._aliases: dict[str, str] = {}
        if taxonomy is not None:
            for node in taxonomy:
                self.add_node(node)

    def add_node(self, node: TaxonomyNode) -> TaxonomyNode:
        self._nodes[node.name] = node
        for alias in node.aliases:
            self._aliases[alias] = node.name
        return node

    def get(self, name: str) -> TaxonomyNode | None:
        name = _normalize_text(name)
        resolved = self._aliases.get(name, name)
        return self._nodes.get(resolved)

    def resolve(self, name: str) -> str:
        name = _normalize_text(name)
        return self._aliases.get(name, name)

    def classify(self, obj: Any, *, labels: Sequence[str] | None = None) -> TaxonomyClassification:
        kind = _object_kind(obj)
        family = _family(obj)
        profile = _profile(obj)

        meta = _to_mapping(getattr(obj, "metadata", None))
        derived_labels = set(labels or ())
        derived_labels.update({kind, family, profile})
        if meta:
            for key in ("source_kind", "stage", "status", "category"):
                if key in meta and meta[key] is not None:
                    derived_labels.add(_normalize_text(meta[key]))

        confidence = 1.0
        if family == "unknown":
            confidence -= 0.2
        if profile == "unknown":
            confidence -= 0.1

        path = (kind, family, profile)
        return TaxonomyClassification(
            kind=kind,
            family=family,
            profile=profile,
            labels=tuple(sorted({label for label in derived_labels if label})),
            path=path,
            confidence=confidence,
            metadata=meta,
        )

    def classify_many(self, objects: Iterable[Any]) -> tuple[TaxonomyClassification, ...]:
        return tuple(self.classify(obj) for obj in objects)

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [node.to_dict() for node in self._nodes.values()],
            "aliases": dict(self._aliases),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TaxonomyEngine":
        engine = cls()
        for item in data.get("nodes", []):
            engine.add_node(TaxonomyNode(
                name=item.get("name", "unknown"),
                parent=item.get("parent"),
                description=item.get("description", ""),
                aliases=tuple(item.get("aliases", ())),
                metadata=_to_mapping(item.get("metadata", {})),
            ))
        return engine

    def __len__(self) -> int:
        return len(self._nodes)

    def __iter__(self):
        return iter(self._nodes.values())

    def __repr__(self) -> str:
        return f"TaxonomyEngine(nodes={len(self._nodes)})"