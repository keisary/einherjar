# knowledge/insights.py
"""
==========================================================
Knowledge Insights
==========================================================

Extrait des conclusions exploitables à partir du graphe,
des clusters, de la taxonomie et des relations.

Le module ne sélectionne rien :
- il observe,
- il résume,
- il propose des pistes.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .clustering import ClusterEngine
from .clustering import KnowledgeCluster
from .graph import KnowledgeGraph
from .taxonomy import TaxonomyClassification
from .taxonomy import TaxonomyEngine

__all__ = [
    "InsightSeverity",
    "Insight",
    "InsightReport",
    "InsightEngine",
]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


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


def _bounded_unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _to_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "to_dict") and callable(value.to_dict):
        value = value.to_dict()
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _family_key(obj: Any) -> str:
    for attr in ("family", "target_family"):
        value = getattr(obj, attr, None)
        if value:
            return str(value).strip().lower()
    metadata = _to_mapping(getattr(obj, "metadata", None))
    for key in ("family", "target_family", "portfolio_family"):
        if key in metadata and metadata[key] is not None:
            value = str(metadata[key]).strip().lower()
            if value:
                return value
    return "unknown"


def _profile_name(obj: Any) -> str:
    for attr in ("profile_name", "profile"):
        value = getattr(obj, attr, None)
        if value:
            return str(value).strip().lower()
    profile = getattr(obj, "profile", None)
    if profile is not None and getattr(profile, "name", None):
        value = str(profile.name).strip().lower()
        if value:
            return value
    metadata = _to_mapping(getattr(obj, "metadata", None))
    for key in ("profile_name", "strategy_name", "einher_name"):
        if key in metadata and metadata[key] is not None:
            value = str(metadata[key]).strip().lower()
            if value:
                return value
    return "unknown"


class InsightSeverity(str):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class Insight:
    """
    Conclusion exploitable.
    """

    code: str
    message: str
    severity: str = InsightSeverity.INFO
    confidence: float = 0.5
    evidence: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", str(self.code).strip().lower())
        object.__setattr__(self, "message", str(self.message).strip())
        object.__setattr__(self, "severity", str(self.severity).strip().lower())
        object.__setattr__(self, "confidence", _bounded_unit(self.confidence))
        object.__setattr__(self, "evidence", dict(self.evidence))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "confidence": self.confidence,
            "evidence": dict(self.evidence),
            "metadata": dict(self.metadata),
            "created_at": self.created_at.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class InsightReport:
    """
    Rapport d'insights.
    """

    insights: tuple[Insight, ...]
    summary: dict[str, Any] = field(default_factory=dict)
    recommendations: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        object.__setattr__(self, "insights", tuple(self.insights))
        object.__setattr__(self, "summary", dict(self.summary))
        object.__setattr__(self, "recommendations", tuple(str(x) for x in self.recommendations))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "insights": [insight.to_dict() for insight in self.insights],
            "summary": dict(self.summary),
            "recommendations": list(self.recommendations),
            "metadata": dict(self.metadata),
            "created_at": self.created_at.isoformat(),
        }


class InsightEngine:
    """
    Produit des insights à partir du knowledge.
    """

    def __init__(
        self,
        *,
        taxonomy_engine: TaxonomyEngine | None = None,
        cluster_engine: ClusterEngine | None = None,
    ) -> None:
        self._taxonomy = taxonomy_engine or TaxonomyEngine()
        self._cluster = cluster_engine or ClusterEngine()

    @property
    def taxonomy_engine(self) -> TaxonomyEngine:
        return self._taxonomy

    @property
    def cluster_engine(self) -> ClusterEngine:
        return self._cluster

    def analyze(
        self,
        objects: Iterable[Any] | None = None,
        *,
        graph: KnowledgeGraph | None = None,
        clusters: Sequence[KnowledgeCluster] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> InsightReport:
        objects = tuple(objects or ())
        taxonomy = tuple(self._taxonomy.classify_many(objects)) if objects else ()
        if clusters is None and objects:
            clusters = self._cluster.cluster(objects, metadata=metadata)
        clusters = tuple(clusters or ())

        insights: list[Insight] = []
        recommendations: list[str] = []

        if objects:
            family_counts = Counter(_family_key(obj) for obj in objects)
            profile_counts = Counter(_profile_name(obj) for obj in objects)
            total = len(objects)

            dominant_family, dominant_family_count = family_counts.most_common(1)[0]
            dominant_profile, dominant_profile_count = profile_counts.most_common(1)[0]

            family_share = dominant_family_count / total
            profile_share = dominant_profile_count / total

            if family_share > 0.5:
                insights.append(
                    Insight(
                        code="family_concentration",
                        message="Une famille domine largement le corpus.",
                        severity=InsightSeverity.WARNING,
                        confidence=min(1.0, family_share),
                        evidence={"family": dominant_family, "share": family_share, "counts": dict(family_counts)},
                    )
                )
                recommendations.append("augmenter la diversité des familles")

            if profile_share > 0.5:
                insights.append(
                    Insight(
                        code="profile_concentration",
                        message="Un profil domine largement le corpus.",
                        severity=InsightSeverity.WARNING,
                        confidence=min(1.0, profile_share),
                        evidence={"profile": dominant_profile, "share": profile_share, "counts": dict(profile_counts)},
                    )
                )
                recommendations.append("augmenter la diversité des profils")

            unique_families = len(family_counts)
            unique_profiles = len(profile_counts)
            if unique_families <= 1:
                insights.append(
                    Insight(
                        code="single_family",
                        message="Le corpus n'ouvre qu'une seule famille.",
                        severity=InsightSeverity.INFO,
                        confidence=1.0,
                        evidence={"families": dict(family_counts)},
                    )
                )
            if unique_profiles <= 1:
                insights.append(
                    Insight(
                        code="single_profile",
                        message="Le corpus n'ouvre qu'un seul profil.",
                        severity=InsightSeverity.INFO,
                        confidence=1.0,
                        evidence={"profiles": dict(profile_counts)},
                    )
                )

        if clusters:
            large_clusters = [cluster for cluster in clusters if cluster.size >= 3]
            if large_clusters:
                biggest = max(large_clusters, key=lambda c: c.size)
                insights.append(
                    Insight(
                        code="dense_cluster",
                        message="Un cluster dense mérite d'être examiné comme zone structurée.",
                        severity=InsightSeverity.INFO,
                        confidence=min(1.0, biggest.size / max(1, len(objects) if objects else biggest.size)),
                        evidence={"cluster": biggest.to_dict()},
                    )
                )
                recommendations.append("explorer les clusters denses")

            singleton_count = sum(1 for cluster in clusters if cluster.size == 1)
            if singleton_count > 0 and len(clusters) > 0:
                insights.append(
                    Insight(
                        code="singleton_presence",
                        message="Le graphe contient des éléments isolés.",
                        severity=InsightSeverity.INFO,
                        confidence=_bounded_unit(singleton_count / len(clusters)),
                        evidence={"singleton_count": singleton_count, "cluster_count": len(clusters)},
                    )
                )

        if graph is not None:
            summary = graph.summary()
            if summary.get("edge_count", 0) == 0 and summary.get("node_count", 0) > 1:
                insights.append(
                    Insight(
                        code="disconnected_graph",
                        message="Le graphe ne contient pas encore de relations.",
                        severity=InsightSeverity.INFO,
                        confidence=1.0,
                        evidence=summary,
                    )
                )
                recommendations.append("enrichir les relations")

            relationship_counts = _to_mapping(summary.get("relationships", {}))
            if relationship_counts:
                strongest = max(relationship_counts.items(), key=lambda item: item[1])
                insights.append(
                    Insight(
                        code="dominant_relationship",
                        message="Un type de relation domine la structure.",
                        severity=InsightSeverity.INFO,
                        confidence=1.0,
                        evidence={"relationship": strongest[0], "count": strongest[1], "summary": summary},
                    )
                )

        if taxonomy:
            unknown = sum(1 for item in taxonomy if item.family == "unknown" or item.profile == "unknown")
            if unknown > 0:
                insights.append(
                    Insight(
                        code="taxonomy_unknowns",
                        message="Certaines classifications restent incomplètes.",
                        severity=InsightSeverity.WARNING if unknown > len(taxonomy) * 0.3 else InsightSeverity.INFO,
                        confidence=_bounded_unit(unknown / max(1, len(taxonomy))),
                        evidence={"unknown": unknown, "taxonomy_size": len(taxonomy)},
                    )
                )
                recommendations.append("améliorer la classification taxonomique")

        if not insights:
            insights.append(
                Insight(
                    code="stable_knowledge",
                    message="Le corpus ne montre pas de faiblesse structurelle évidente.",
                    severity=InsightSeverity.INFO,
                    confidence=0.75,
                    evidence={"objects": len(objects), "clusters": len(clusters)},
                )
            )

        summary = {
            "object_count": len(objects),
            "cluster_count": len(clusters),
            "insight_count": len(insights),
            "taxonomy_count": len(taxonomy),
            "families": dict(Counter(_family_key(obj) for obj in objects)) if objects else {},
            "profiles": dict(Counter(_profile_name(obj) for obj in objects)) if objects else {},
        }

        return InsightReport(
            insights=tuple(insights),
            summary=summary,
            recommendations=tuple(dict.fromkeys(recommendations)),
            metadata=dict(metadata or {}),
        )

    def inspect(self, objects: Iterable[Any] | None = None, **kwargs: Any) -> InsightReport:
        return self.analyze(objects, **kwargs)

    def __repr__(self) -> str:
        return "InsightEngine()"