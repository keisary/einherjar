# knowledge/similarity.py
"""
==========================================================
Knowledge Similarity
==========================================================

Mesure la proximité entre objets du corpus, des Einhers et
des structures de knowledge.

Le module produit des scores, jamais des décisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .fingerprints import KnowledgeFingerprint
from .fingerprints import fingerprint_object

__all__ = [
    "SimilaritySettings",
    "SimilarityScore",
    "SimilarityMatrix",
    "SimilarityEngine",
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


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    value = _to_mapping(value)

    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            output.update(_flatten(item, path))
        return output

    if isinstance(value, (list, tuple, set)):
        output: dict[str, Any] = {}
        for index, item in enumerate(value):
            path = f"{prefix}[{index}]"
            output.update(_flatten(item, path))
        return output

    return {prefix: value}


def _extract_numeric_vector(obj: Any, keys: Sequence[str]) -> np.ndarray:
    mapping = _flatten(obj)
    values: list[float] = []
    for key in keys:
        if key in mapping:
            try:
                values.append(float(mapping[key]))
            except (TypeError, ValueError):
                continue
    return np.asarray(values, dtype=float)


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 or right.size == 0:
        return 0.0
    if left.size != right.size:
        size = max(left.size, right.size)
        left = np.resize(left, size)
        right = np.resize(right, size)
    lnorm = float(np.linalg.norm(left))
    rnorm = float(np.linalg.norm(right))
    if lnorm <= 1e-12 or rnorm <= 1e-12:
        return 0.0
    return float(np.dot(left, right) / (lnorm * rnorm))


def _label_tokens(obj: Any) -> set[str]:
    tokens: set[str] = set()

    candidates = [
        getattr(obj, "family", None),
        getattr(obj, "profile_name", None),
        getattr(obj, "name", None),
        getattr(obj, "label", None),
        getattr(obj, "kind", None),
        getattr(obj, "source_kind", None),
    ]
    metadata = _to_mapping(getattr(obj, "metadata", None))
    candidates.extend(metadata.get(key) for key in ("family", "profile_name", "source_kind", "kind", "label"))

    for value in candidates:
        text = str(value or "").strip().lower()
        if text:
            tokens.update(part for part in text.replace("-", "_").split("_") if part)
    return tokens


def _fingerprint_value(obj: Any) -> str:
    if isinstance(obj, KnowledgeFingerprint):
        return obj.digest
    if hasattr(obj, "digest"):
        value = getattr(obj, "digest", None)
        if value:
            return str(value)
    return fingerprint_object(obj, kind="knowledge")


@dataclass(frozen=True, slots=True)
class SimilaritySettings:
    """
    Pondérations de similarité.
    """

    structural_weight: float = 0.30
    numeric_weight: float = 0.40
    token_weight: float = 0.20
    fingerprint_weight: float = 0.10

    def __post_init__(self) -> None:
        weights = {
            "structural": max(0.0, _coerce_float(self.structural_weight, 0.30)),
            "numeric": max(0.0, _coerce_float(self.numeric_weight, 0.40)),
            "token": max(0.0, _coerce_float(self.token_weight, 0.20)),
            "fingerprint": max(0.0, _coerce_float(self.fingerprint_weight, 0.10)),
        }
        total = sum(weights.values())
        if total <= 0:
            weights = {"structural": 0.30, "numeric": 0.40, "token": 0.20, "fingerprint": 0.10}
        object.__setattr__(self, "structural_weight", weights["structural"])
        object.__setattr__(self, "numeric_weight", weights["numeric"])
        object.__setattr__(self, "token_weight", weights["token"])
        object.__setattr__(self, "fingerprint_weight", weights["fingerprint"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "structural_weight": self.structural_weight,
            "numeric_weight": self.numeric_weight,
            "token_weight": self.token_weight,
            "fingerprint_weight": self.fingerprint_weight,
        }


@dataclass(frozen=True, slots=True)
class SimilarityScore:
    """
    Score de similarité entre deux objets.
    """

    left: str
    right: str
    score: float
    structural: float
    numeric: float
    token: float
    fingerprint: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "left", str(self.left))
        object.__setattr__(self, "right", str(self.right))
        object.__setattr__(self, "score", _bounded_unit(self.score))
        object.__setattr__(self, "structural", _bounded_unit(self.structural))
        object.__setattr__(self, "numeric", _bounded_unit(self.numeric))
        object.__setattr__(self, "token", _bounded_unit(self.token))
        object.__setattr__(self, "fingerprint", _bounded_unit(self.fingerprint))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "left": self.left,
            "right": self.right,
            "score": self.score,
            "structural": self.structural,
            "numeric": self.numeric,
            "token": self.token,
            "fingerprint": self.fingerprint,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class SimilarityMatrix:
    """
    Matrice de similarité.
    """

    labels: tuple[str, ...]
    matrix: np.ndarray
    scores: tuple[SimilarityScore, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "labels", tuple(self.labels))
        object.__setattr__(self, "matrix", np.asarray(self.matrix, dtype=float))
        object.__setattr__(self, "scores", tuple(self.scores))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "labels": list(self.labels),
            "matrix": self.matrix.tolist(),
            "scores": [score.to_dict() for score in self.scores],
            "metadata": dict(self.metadata),
        }


class SimilarityEngine:
    """
    Calcule des similarités entre objets.
    """

    def __init__(self, settings: SimilaritySettings | None = None) -> None:
        self._settings = settings or SimilaritySettings()

    @property
    def settings(self) -> SimilaritySettings:
        return self._settings

    def compare(self, left: Any, right: Any, *, metadata: Mapping[str, Any] | None = None) -> SimilarityScore:
        left_map = _flatten(left)
        right_map = _flatten(right)

        left_tokens = _label_tokens(left)
        right_tokens = _label_tokens(right)

        structural = _jaccard(set(left_map.keys()), set(right_map.keys()))
        token = _jaccard(left_tokens, right_tokens)

        numeric_keys = [
            "score",
            "weight",
            "capital",
            "trade_count",
            "total_pnl",
            "win_rate",
            "profit_factor",
            "expectancy",
            "max_drawdown",
            "signal_coverage",
            "exposure_ratio",
            "average_mae",
            "average_mfe",
            "average_mfe_to_mae_ratio",
        ]
        left_vec = _extract_numeric_vector(left_map, numeric_keys)
        right_vec = _extract_numeric_vector(right_map, numeric_keys)
        numeric = (_cosine(left_vec, right_vec) + 1.0) / 2.0 if left_vec.size or right_vec.size else 0.0

        left_fp = _fingerprint_value(left)
        right_fp = _fingerprint_value(right)
        fingerprint = 1.0 if left_fp == right_fp else 0.0

        score = (
            self._settings.structural_weight * structural
            + self._settings.numeric_weight * numeric
            + self._settings.token_weight * token
            + self._settings.fingerprint_weight * fingerprint
        )

        return SimilarityScore(
            left=left_fp,
            right=right_fp,
            score=score,
            structural=structural,
            numeric=numeric,
            token=token,
            fingerprint=fingerprint,
            metadata=dict(metadata or {}),
        )

    def matrix(self, values: Sequence[Any], *, metadata: Mapping[str, Any] | None = None) -> SimilarityMatrix:
        values = tuple(values)
        labels = tuple(_fingerprint_value(value) for value in values)
        size = len(values)
        matrix = np.eye(size, dtype=float)
        scores: list[SimilarityScore] = []

        for i in range(size):
            for j in range(i + 1, size):
                score = self.compare(values[i], values[j], metadata=metadata)
                matrix[i, j] = matrix[j, i] = score.score
                scores.append(score)

        return SimilarityMatrix(labels=labels, matrix=matrix, scores=tuple(scores), metadata=dict(metadata or {}))

    def similarity_matrix(self, values: Sequence[Any], *, metadata: Mapping[str, Any] | None = None) -> SimilarityMatrix:
        return self.matrix(values, metadata=metadata)

    def compare_many(self, values: Sequence[Any], *, metadata: Mapping[str, Any] | None = None) -> tuple[SimilarityScore, ...]:
        return self.matrix(values, metadata=metadata).scores

    def __repr__(self) -> str:
        return f"SimilarityEngine(structural={self._settings.structural_weight})"