# memory/learning.py
"""
==========================================================
Learning Memory
==========================================================

Synthèse d'apprentissage à partir des différentes mémoires.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

from .corpus_history import CorpusHistory
from .explored_regions import ExploredRegions
from .failed_regions import FailedRegions
from .family_history import FamilyHistory
from .feature_history import FeatureHistory
from .search_history import SearchHistory
from .successful_regions import SuccessfulRegions

__all__ = [
    "LearningInsight",
    "LearningSummary",
    "LearningState",
    "LearningEngine",
]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_int(value: Any, default: int = 0) -> int:
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


def _normalize_text(value: Any, default: str = "") -> str:
    text = str(value or "").strip()
    return text if text else default


@dataclass(frozen=True, slots=True)
class LearningInsight:
    code: str
    message: str
    severity: str = "info"
    confidence: float = 0.5
    evidence: dict[str, Any] = field(default_factory=dict)
    recommendations: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", _normalize_text(self.code).lower())
        object.__setattr__(self, "message", _normalize_text(self.message))
        object.__setattr__(self, "severity", _normalize_text(self.severity, "info").lower())
        object.__setattr__(self, "confidence", _bounded_unit(self.confidence))
        object.__setattr__(self, "evidence", dict(self.evidence))
        object.__setattr__(self, "recommendations", tuple(dict.fromkeys(_normalize_text(item) for item in self.recommendations if _normalize_text(item))))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "confidence": self.confidence,
            "evidence": dict(self.evidence),
            "recommendations": list(self.recommendations),
            "metadata": dict(self.metadata),
            "created_at": self.created_at.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class LearningSummary:
    insight_count: int
    dominant_families: tuple[str, ...] = ()
    dominant_features: tuple[str, ...] = ()
    promising_regions: tuple[str, ...] = ()
    failed_regions: tuple[str, ...] = ()
    corpus_versions: int = 0
    search_count: int = 0
    explored_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "insight_count", max(0, _coerce_int(self.insight_count)))
        object.__setattr__(self, "dominant_families", tuple(self.dominant_families))
        object.__setattr__(self, "dominant_features", tuple(self.dominant_features))
        object.__setattr__(self, "promising_regions", tuple(self.promising_regions))
        object.__setattr__(self, "failed_regions", tuple(self.failed_regions))
        object.__setattr__(self, "corpus_versions", max(0, _coerce_int(self.corpus_versions)))
        object.__setattr__(self, "search_count", max(0, _coerce_int(self.search_count)))
        object.__setattr__(self, "explored_count", max(0, _coerce_int(self.explored_count)))
        object.__setattr__(self, "success_count", max(0, _coerce_int(self.success_count)))
        object.__setattr__(self, "failure_count", max(0, _coerce_int(self.failure_count)))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "insight_count": self.insight_count,
            "dominant_families": list(self.dominant_families),
            "dominant_features": list(self.dominant_features),
            "promising_regions": list(self.promising_regions),
            "failed_regions": list(self.failed_regions),
            "corpus_versions": self.corpus_versions,
            "search_count": self.search_count,
            "explored_count": self.explored_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class LearningState:
    insights: list[LearningInsight] = field(default_factory=list)
    summary: LearningSummary | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    updated_at: datetime = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "insights", list(self.insights))

    def add(self, insight: LearningInsight) -> LearningInsight:
        self.insights.append(insight)
        self.updated_at = _utc_now()
        self.summary = None
        return insight

    def extend(self, insights: Iterable[LearningInsight]) -> None:
        for insight in insights:
            self.add(insight)

    @property
    def compiled_summary(self) -> LearningSummary:
        if self.summary is not None:
            return self.summary
        if not self.insights:
            return LearningSummary(0, metadata=dict(self.metadata))
        return LearningSummary(
            insight_count=len(self.insights),
            dominant_families=tuple(),
            dominant_features=tuple(),
            promising_regions=tuple(),
            failed_regions=tuple(),
            metadata=dict(self.metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "insights": [insight.to_dict() for insight in self.insights],
            "summary": self.compiled_summary.to_dict(),
            "metadata": dict(self.metadata),
            "updated_at": self.updated_at.isoformat(),
        }


class LearningEngine:
    def __init__(self) -> None:
        self.state = LearningState()

    def learn(
        self,
        *,
        search_history: SearchHistory | None = None,
        explored_regions: ExploredRegions | None = None,
        successful_regions: SuccessfulRegions | None = None,
        failed_regions: FailedRegions | None = None,
        feature_history: FeatureHistory | None = None,
        family_history: FamilyHistory | None = None,
        corpus_history: CorpusHistory | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> LearningState:
        insights: list[LearningInsight] = []

        if search_history is not None and len(search_history) > 0:
            summary = search_history.summary
            if summary.success_count > 0:
                insights.append(
                    LearningInsight(
                        code="search_success_present",
                        message="Le moteur a déjà identifié des recherches fructueuses.",
                        severity="info",
                        confidence=min(1.0, summary.success_count / max(1, summary.entry_count)),
                        evidence=summary.to_dict(),
                        recommendations=("réutiliser les schémas de recherche positifs",),
                    )
                )

        if explored_regions is not None and len(explored_regions) > 0:
            summary = explored_regions.summary
            if summary.region_count > 0:
                insights.append(
                    LearningInsight(
                        code="exploration_memory",
                        message="La mémoire d'exploration contient des régions déjà visitées.",
                        severity="info",
                        confidence=1.0,
                        evidence=summary.to_dict(),
                        recommendations=("éviter les régions déjà saturées",),
                    )
                )

        if successful_regions is not None and len(successful_regions) > 0:
            summary = successful_regions.summary
            if summary.region_count > 0:
                promising = tuple(region.region_key for region in successful_regions.entries[:10])
                insights.append(
                    LearningInsight(
                        code="promising_regions",
                        message="Certaines régions ont produit des résultats utiles.",
                        severity="info",
                        confidence=min(1.0, summary.average_yield_rate),
                        evidence=summary.to_dict(),
                        recommendations=("prioriser les régions prometteuses",),
                    )
                )

        if failed_regions is not None and len(failed_regions) > 0:
            summary = failed_regions.summary
            if summary.region_count > 0:
                insights.append(
                    LearningInsight(
                        code="failed_regions",
                        message="Des régions montrent une faible valeur d'exploration.",
                        severity="warning",
                        confidence=min(1.0, summary.total_failure_count / max(1, summary.total_attempts)),
                        evidence=summary.to_dict(),
                        recommendations=("réduire la priorité de ces régions",),
                    )
                )

        if feature_history is not None and len(feature_history) > 0:
            summary = feature_history.summary
            if summary.top_features:
                insights.append(
                    LearningInsight(
                        code="feature_preference",
                        message="Certaines features reviennent comme plus utiles que d'autres.",
                        severity="info",
                        confidence=1.0,
                        evidence=summary.to_dict(),
                        recommendations=("favoriser les features les plus prometteuses",),
                    )
                )

        if family_history is not None and len(family_history) > 0:
            summary = family_history.summary
            if summary.dominant_families:
                insights.append(
                    LearningInsight(
                        code="family_memory",
                        message="Le moteur a une mémoire exploitable des familles.",
                        severity="info",
                        confidence=1.0,
                        evidence=summary.to_dict(),
                        recommendations=("équilibrer les familles dominantes et sous-explorées",),
                    )
                )

        if corpus_history is not None and len(corpus_history) > 0:
            summary = corpus_history.summary
            if summary.version_count > 0:
                insights.append(
                    LearningInsight(
                        code="corpus_evolution",
                        message="Le corpus a déjà une évolution historique exploitable.",
                        severity="info",
                        confidence=1.0,
                        evidence=summary.to_dict(),
                        recommendations=("suivre les évolutions du corpus",),
                    )
                )

        if not insights:
            insights.append(
                LearningInsight(
                    code="insufficient_memory",
                    message="La mémoire ne contient pas encore assez d'information pour apprendre.",
                    severity="info",
                    confidence=0.5,
                    evidence={},
                    recommendations=("enrichir les historiques",),
                )
            )

        dominant_families = tuple()
        dominant_features = tuple()
        promising_regions = tuple()
        failed_region_list = tuple()

        if family_history is not None and len(family_history) > 0:
            dominant_families = tuple(family.family_key for family in family_history.entries[:10])
        if feature_history is not None and len(feature_history) > 0:
            dominant_features = tuple(feature.feature_key for feature in feature_history.entries[:10])
        if successful_regions is not None and len(successful_regions) > 0:
            promising_regions = tuple(region.region_key for region in successful_regions.entries[:10])
        if failed_regions is not None and len(failed_regions) > 0:
            failed_region_list = tuple(region.region_key for region in failed_regions.entries[:10])

        summary = LearningSummary(
            insight_count=len(insights),
            dominant_families=dominant_families,
            dominant_features=dominant_features,
            promising_regions=promising_regions,
            failed_regions=failed_region_list,
            corpus_versions=len(corpus_history) if corpus_history is not None else 0,
            search_count=len(search_history) if search_history is not None else 0,
            explored_count=len(explored_regions) if explored_regions is not None else 0,
            success_count=len(successful_regions) if successful_regions is not None else 0,
            failure_count=len(failed_regions) if failed_regions is not None else 0,
            metadata=dict(metadata or {}),
        )

        self.state = LearningState(
            insights=insights,
            summary=summary,
            metadata=dict(metadata or {}),
        )
        return self.state

    def suggest_families(self, limit: int = 5) -> tuple[str, ...]:
        limit = max(1, _coerce_int(limit, 5))
        summary = self.state.compiled_summary
        return tuple(summary.dominant_families[:limit])

    def suggest_features(self, limit: int = 5) -> tuple[str, ...]:
        limit = max(1, _coerce_int(limit, 5))
        summary = self.state.compiled_summary
        return tuple(summary.dominant_features[:limit])

    def suggest_regions(self, limit: int = 5) -> tuple[str, ...]:
        limit = max(1, _coerce_int(limit, 5))
        summary = self.state.compiled_summary
        return tuple(summary.promising_regions[:limit])

    def to_dict(self) -> dict[str, Any]:
        return self.state.to_dict()

    def __repr__(self) -> str:
        return f"LearningEngine(insights={len(self.state.insights)})"