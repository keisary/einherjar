# knowledge/ontology.py
"""
==========================================================
Knowledge Ontology
==========================================================

Représente les concepts sémantiques du système.

L'ontologie ne remplace ni la taxonomie ni le graphe :
- elle donne un sens stable aux catégories,
- elle relie les concepts et leurs alias,
- elle fournit une lecture sémantique.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "OntologyConcept",
    "OntologyRelation",
    "OntologyMap",
    "OntologyEngine",
]


def _normalize_text(value: Any) -> str:
    return str(value or "").strip().lower()


def _to_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "to_dict") and callable(value.to_dict):
        value = value.to_dict()
    if isinstance(value, Mapping):
        return dict(value)
    return {}


@dataclass(frozen=True, slots=True)
class OntologyRelation:
    """
    Relation sémantique entre concepts.
    """

    source: str
    target: str
    relation: str = "related"
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", _normalize_text(self.source))
        object.__setattr__(self, "target", _normalize_text(self.target))
        object.__setattr__(self, "relation", _normalize_text(self.relation) or "related")
        object.__setattr__(self, "weight", max(0.0, float(self.weight)))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "relation": self.relation,
            "weight": self.weight,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class OntologyConcept:
    """
    Concept sémantique.
    """

    name: str
    description: str = ""
    aliases: tuple[str, ...] = ()
    parent: str | None = None
    properties: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _normalize_text(self.name) or "unknown")
        object.__setattr__(self, "description", str(self.description).strip())
        object.__setattr__(self, "aliases", tuple(sorted({str(alias).strip().lower() for alias in self.aliases if str(alias).strip()})))
        object.__setattr__(self, "parent", _normalize_text(self.parent) or None)
        object.__setattr__(self, "properties", dict(self.properties))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "aliases": list(self.aliases),
            "parent": self.parent,
            "properties": dict(self.properties),
        }


class OntologyMap:
    """
    Registre de concepts et de relations.
    """

    def __init__(self, concepts: Iterable[OntologyConcept] | None = None, relations: Iterable[OntologyRelation] | None = None) -> None:
        self._concepts: dict[str, OntologyConcept] = {}
        self._aliases: dict[str, str] = {}
        self._relations: list[OntologyRelation] = []

        if concepts is not None:
            for concept in concepts:
                self.add_concept(concept)
        if relations is not None:
            for relation in relations:
                self.add_relation(relation)

    def add_concept(self, concept: OntologyConcept) -> OntologyConcept:
        self._concepts[concept.name] = concept
        for alias in concept.aliases:
            self._aliases[alias] = concept.name
        return concept

    def add_relation(self, relation: OntologyRelation) -> OntologyRelation:
        self._relations.append(relation)
        return relation

    def register(
        self,
        name: str,
        *,
        description: str = "",
        aliases: Sequence[str] | None = None,
        parent: str | None = None,
        properties: Mapping[str, Any] | None = None,
    ) -> OntologyConcept:
        concept = OntologyConcept(
            name=name,
            description=description,
            aliases=tuple(aliases or ()),
            parent=parent,
            properties=_to_mapping(properties or {}),
        )
        return self.add_concept(concept)

    def resolve(self, name: str) -> str:
        name = _normalize_text(name)
        return self._aliases.get(name, name)

    def get(self, name: str) -> OntologyConcept | None:
        return self._concepts.get(self.resolve(name))

    def related(self, name: str) -> tuple[OntologyRelation, ...]:
        name = self.resolve(name)
        return tuple(rel for rel in self._relations if rel.source == name or rel.target == name)

    def concepts(self) -> tuple[OntologyConcept, ...]:
        return tuple(self._concepts.values())

    def relations(self) -> tuple[OntologyRelation, ...]:
        return tuple(self._relations)

    def to_dict(self) -> dict[str, Any]:
        return {
            "concepts": [concept.to_dict() for concept in self._concepts.values()],
            "relations": [relation.to_dict() for relation in self._relations],
            "aliases": dict(self._aliases),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "OntologyMap":
        ontology = cls()
        for item in data.get("concepts", []):
            ontology.add_concept(
                OntologyConcept(
                    name=item.get("name", "unknown"),
                    description=item.get("description", ""),
                    aliases=tuple(item.get("aliases", ())),
                    parent=item.get("parent"),
                    properties=_to_mapping(item.get("properties", {})),
                )
            )
        for item in data.get("relations", []):
            ontology.add_relation(
                OntologyRelation(
                    source=item.get("source", ""),
                    target=item.get("target", ""),
                    relation=item.get("relation", "related"),
                    weight=float(item.get("weight", 1.0)),
                    metadata=_to_mapping(item.get("metadata", {})),
                )
            )
        return ontology

    def __len__(self) -> int:
        return len(self._concepts)

    def __iter__(self):
        return iter(self._concepts.values())

    def __repr__(self) -> str:
        return f"OntologyMap(concepts={len(self._concepts)}, relations={len(self._relations)})"


class OntologyEngine:
    """
    Construit une ontologie de travail.
    """

    def __init__(self, ontology: OntologyMap | None = None) -> None:
        self._ontology = ontology or OntologyMap()
        self._bootstrap()

    @property
    def ontology(self) -> OntologyMap:
        return self._ontology

    def _bootstrap(self) -> None:
        if len(self._ontology) > 0:
            return

        self._ontology.register("einher", description="Strategie finale retenue")
        self._ontology.register("corpus", description="Corpus final d'Einhers")
        self._ontology.register("family", description="Famille de features ou de stratégies")
        self._ontology.register("profile", description="Profil d'exécution")
        self._ontology.register("feature", description="Feature du moteur de recherche")
        self._ontology.register("relationship", description="Lien explicite entre objets")
        self._ontology.register("cluster", description="Groupe d'objets proches")
        self._ontology.register("insight", description="Conclusion exploitable")
        self._ontology.add_relation(OntologyRelation("einher", "corpus", "part_of", 1.0))
        self._ontology.add_relation(OntologyRelation("feature", "family", "belongs_to", 1.0))
        self._ontology.add_relation(OntologyRelation("profile", "einher", "describes", 1.0))
        self._ontology.add_relation(OntologyRelation("cluster", "einher", "groups", 0.8))
        self._ontology.add_relation(OntologyRelation("insight", "corpus", "summarizes", 1.0))

    def classify(self, name: str) -> OntologyConcept | None:
        return self._ontology.get(name)

    def add_concept(self, concept: OntologyConcept) -> OntologyConcept:
        return self._ontology.add_concept(concept)

    def add_relation(self, relation: OntologyRelation) -> OntologyRelation:
        return self._ontology.add_relation(relation)

    def resolve(self, name: str) -> str:
        return self._ontology.resolve(name)

    def to_dict(self) -> dict[str, Any]:
        return self._ontology.to_dict()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "OntologyEngine":
        return cls(OntologyMap.from_dict(data))

    def __repr__(self) -> str:
        return f"OntologyEngine(concepts={len(self._ontology)})"