# knowledge/clustering.py
"""
==========================================================
Knowledge Clustering
==========================================================

Regroupe les objets du corpus en clusters de similarité.

Le clustering aide à :
- réduire la redondance,
- visualiser les familles proches,
- extraire des zones de connaissance denses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .similarity import SimilarityEngine
from .similarity import SimilarityMatrix
from .similarity import SimilarityScore

__all__ = [
    "ClusterSettings",
    "KnowledgeCluster",
    "ClusterSummary",
    "ClusterEngine",
]


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


def _fingerprint_value(obj: Any) -> str:
    if hasattr(obj, "digest"):
        value = getattr(obj, "digest", None)
        if value:
            return str(value)
    if hasattr(obj, "subject_fingerprint"):
        value = getattr(obj, "subject_fingerprint", None)
        if value:
            return str(value)
    if hasattr(obj, "execution_fingerprint"):
        fp = getattr(obj, "execution_fingerprint", None)
        if fp is not None:
            digest = getattr(fp, "digest", None)
            if digest:
                return str(digest)
    return repr(obj)


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


class _UnionFind:
    def __init__(self, items: Sequence[str]) -> None:
        self.parent = {item: item for item in items}
        self.rank = {item: 0 for item in items}

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        rank_left = self.rank[root_left]
        rank_right = self.rank[root_right]
        if rank_left < rank_right:
            self.parent[root_left] = root_right
        elif rank_left > rank_right:
            self.parent[root_right] = root_left
        else:
            self.parent[root_right] = root_left
            self.rank[root_left] += 1


@dataclass(frozen=True, slots=True)
class ClusterSettings:
    """
    Paramètres de clustering.
    """

    threshold: float = 0.75
    min_cluster_size: int = 2
    max_cluster_size: int = 64
    sample_points: int = 64

    def __post_init__(self) -> None:
        object.__setattr__(self, "threshold", _bounded_unit(_coerce_float(self.threshold, 0.75)))
        object.__setattr__(self, "min_cluster_size", max(1, _coerce_int(self.min_cluster_size, 2)))
        object.__setattr__(self, "max_cluster_size", max(self.min_cluster_size, _coerce_int(self.max_cluster_size, 64)))
        object.__setattr__(self, "sample_points", max(8, _coerce_int(self.sample_points, 64)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "min_cluster_size": self.min_cluster_size,
            "max_cluster_size": self.max_cluster_size,
            "sample_points": self.sample_points,
        }


@dataclass(frozen=True, slots=True)
class KnowledgeCluster:
    """
    Cluster d'objets proches.
    """

    id: str
    members: tuple[str, ...]
    size: int
    centroid: str = ""
    score: float = 0.0
    families: tuple[str, ...] = ()
    profiles: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", str(self.id))
        object.__setattr__(self, "members", tuple(self.members))
        object.__setattr__(self, "size", max(0, _coerce_int(self.size, 0)))
        object.__setattr__(self, "centroid", str(self.centroid))
        object.__setattr__(self, "score", _bounded_unit(self.score))
        object.__setattr__(self, "families", tuple(sorted({str(f).strip().lower() for f in self.families if str(f).strip()})))
        object.__setattr__(self, "profiles", tuple(sorted({str(p).strip().lower() for p in self.profiles if str(p).strip()})))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "members": list(self.members),
            "size": self.size,
            "centroid": self.centroid,
            "score": self.score,
            "families": list(self.families),
            "profiles": list(self.profiles),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class ClusterSummary:
    """
    Résumé du clustering.
    """

    cluster_count: int
    item_count: int
    average_cluster_size: float
    max_cluster_size: int
    singleton_count: int
    density: float
    threshold: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "cluster_count", max(0, _coerce_int(self.cluster_count, 0)))
        object.__setattr__(self, "item_count", max(0, _coerce_int(self.item_count, 0)))
        object.__setattr__(self, "average_cluster_size", max(0.0, float(self.average_cluster_size)))
        object.__setattr__(self, "max_cluster_size", max(0, _coerce_int(self.max_cluster_size, 0)))
        object.__setattr__(self, "singleton_count", max(0, _coerce_int(self.singleton_count, 0)))
        object.__setattr__(self, "density", _bounded_unit(self.density))
        object.__setattr__(self, "threshold", _bounded_unit(self.threshold))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_count": self.cluster_count,
            "item_count": self.item_count,
            "average_cluster_size": self.average_cluster_size,
            "max_cluster_size": self.max_cluster_size,
            "singleton_count": self.singleton_count,
            "density": self.density,
            "threshold": self.threshold,
            "metadata": dict(self.metadata),
        }


class ClusterEngine:
    """
    Calcule des clusters de similarité.
    """

    def __init__(self, settings: ClusterSettings | None = None, *, similarity_engine: SimilarityEngine | None = None) -> None:
        self._settings = settings or ClusterSettings()
        self._similarity_engine = similarity_engine or SimilarityEngine()

    @property
    def settings(self) -> ClusterSettings:
        return self._settings

    @property
    def similarity_engine(self) -> SimilarityEngine:
        return self._similarity_engine

    def cluster(self, objects: Iterable[Any], *, metadata: Mapping[str, Any] | None = None) -> tuple[KnowledgeCluster, ...]:
        objects = tuple(objects)
        if not objects:
            return ()

        labels = [_fingerprint_value(obj) for obj in objects]
        sim_matrix: SimilarityMatrix = self._similarity_engine.matrix(objects, metadata=metadata)

        uf = _UnionFind(labels)
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                if sim_matrix.matrix[i, j] >= self._settings.threshold:
                    uf.union(labels[i], labels[j])

        groups: dict[str, list[int]] = {}
        for index, label in enumerate(labels):
            root = uf.find(label)
            groups.setdefault(root, []).append(index)

        clusters: list[KnowledgeCluster] = []
        for cluster_index, (root, indices) in enumerate(groups.items()):
            if len(indices) < self._settings.min_cluster_size:
                continue

            members = tuple(labels[i] for i in indices)
            cluster_scores = []
            for i in indices:
                for j in indices:
                    if i < j:
                        cluster_scores.append(float(sim_matrix.matrix[i, j]))

            family_counts = {_family_key(objects[i]) for i in indices}
            profile_counts = {_profile_name(objects[i]) for i in indices}

            cluster = KnowledgeCluster(
                id=f"cluster-{cluster_index}",
                members=members,
                size=len(indices),
                centroid=members[0],
                score=float(np.mean(cluster_scores)) if cluster_scores else 1.0,
                families=tuple(sorted(family_counts)),
                profiles=tuple(sorted(profile_counts)),
                metadata=dict(metadata or {}),
            )
            clusters.append(cluster)

        return tuple(sorted(clusters, key=lambda c: (-c.size, c.id)))

    def build_summary(self, clusters: Sequence[KnowledgeCluster], *, item_count: int | None = None, metadata: Mapping[str, Any] | None = None) -> ClusterSummary:
        clusters = tuple(clusters)
        item_count = item_count if item_count is not None else sum(cluster.size for cluster in clusters)
        sizes = [cluster.size for cluster in clusters]
        singleton_count = sum(1 for size in sizes if size == 1)
        average_cluster_size = float(np.mean(sizes)) if sizes else 0.0
        max_cluster_size = max(sizes) if sizes else 0
        density = 0.0
        if item_count > 0:
            density = min(1.0, sum(size for size in sizes if size > 1) / item_count)

        return ClusterSummary(
            cluster_count=len(clusters),
            item_count=item_count,
            average_cluster_size=average_cluster_size,
            max_cluster_size=max_cluster_size,
            singleton_count=singleton_count,
            density=density,
            threshold=self._settings.threshold,
            metadata=dict(metadata or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"settings": self._settings.to_dict()}

    def __repr__(self) -> str:
        return f"ClusterEngine(threshold={self._settings.threshold})"