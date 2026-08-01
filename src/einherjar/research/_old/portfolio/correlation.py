# portfolio/correlation.py
"""
==========================================================
Portfolio Correlation
==========================================================

Mesure la corrélation entre les Einhers candidats au
portefeuille.

Le module transforme les résultats d'exécution en vecteurs
comparables puis construit une matrice de corrélation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from execution.execution_report import ExecutionResult

try:  # optional config module
    from config.portfolio import PortfolioConfig  # type: ignore
except Exception:  # pragma: no cover
    PortfolioConfig = Any  # type: ignore[misc,assignment]

__all__ = [
    "PortfolioCorrelationSettings",
    "PortfolioCorrelationPair",
    "PortfolioCorrelationMatrix",
    "PortfolioCorrelationAnalyzer",
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


def _bounded_unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _safe_mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=float)))


def _safe_max(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(np.max(np.asarray(values, dtype=float)))


def _safe_abs_max(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(np.max(np.abs(np.asarray(values, dtype=float))))


def _normalize_series(values: np.ndarray, *, size: int = 64) -> np.ndarray:
    if values.size == 0:
        return np.zeros(size, dtype=float)

    values = values.astype(float, copy=False).reshape(-1)
    if values.size == 1:
        series = np.full(size, float(values[0]), dtype=float)
    else:
        x_src = np.linspace(0.0, 1.0, values.size)
        x_dst = np.linspace(0.0, 1.0, size)
        series = np.interp(x_dst, x_src, values)

    mean = float(np.mean(series))
    std = float(np.std(series))
    if std <= 1e-12:
        return series - mean
    return (series - mean) / std


def _result_series(result: ExecutionResult, *, size: int) -> np.ndarray:
    records = getattr(result, "records", ())
    pnl = [float(getattr(record.trade, "pnl", 0.0)) for record in records]
    if pnl:
        equity = np.cumsum(np.asarray(pnl, dtype=float))
        return _normalize_series(equity, size=size)

    metrics = result.replay.metrics
    vector = np.asarray(
        [
            float(metrics.total_pnl),
            float(metrics.expectancy),
            float(metrics.win_rate),
            float(metrics.profit_factor if np.isfinite(metrics.profit_factor) else 0.0),
            float(-metrics.max_drawdown),
            float(metrics.trade_count),
            float(metrics.average_duration_bars),
            float(metrics.signal_coverage),
        ],
        dtype=float,
    )
    return _normalize_series(vector, size=size)


@dataclass(frozen=True, slots=True)
class PortfolioCorrelationSettings:
    """
    Paramètres de la corrélation portefeuille.
    """

    sample_points: int = 64
    min_overlap: int = 2
    abs_correlation_penalty: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "sample_points", max(8, _coerce_int(self.sample_points, 64)))
        object.__setattr__(self, "min_overlap", max(2, _coerce_int(self.min_overlap, 2)))
        object.__setattr__(self, "abs_correlation_penalty", max(0.0, _coerce_float(self.abs_correlation_penalty, 1.0)))

    @classmethod
    def from_config(cls, config: Any | None) -> "PortfolioCorrelationSettings":
        if config is None:
            return cls()

        root = _to_mapping(config)
        port = _to_mapping(root.get("portfolio", root.get("portfolio_config", {})))
        corr = _to_mapping(port.get("correlation", port.get("correlations", {})))

        return cls(
            sample_points=_coerce_int(corr.get("sample_points", root.get("sample_points", 64)), 64),
            min_overlap=_coerce_int(corr.get("min_overlap", root.get("min_overlap", 2)), 2),
            abs_correlation_penalty=_coerce_float(corr.get("abs_correlation_penalty", 1.0), 1.0),
        )


@dataclass(frozen=True, slots=True)
class PortfolioCorrelationPair:
    """
    Corrélation entre deux Einhers.
    """

    left: str
    right: str
    correlation: float
    absolute_correlation: float
    distance: float
    left_family: str = "unknown"
    right_family: str = "unknown"
    left_profile: str = "unknown"
    right_profile: str = "unknown"

    def __post_init__(self) -> None:
        object.__setattr__(self, "left", str(self.left))
        object.__setattr__(self, "right", str(self.right))
        object.__setattr__(self, "correlation", float(self.correlation))
        object.__setattr__(self, "absolute_correlation", _bounded_unit(self.absolute_correlation))
        object.__setattr__(self, "distance", _bounded_unit(self.distance))
        object.__setattr__(self, "left_family", str(self.left_family).strip().lower() or "unknown")
        object.__setattr__(self, "right_family", str(self.right_family).strip().lower() or "unknown")
        object.__setattr__(self, "left_profile", str(self.left_profile).strip().lower() or "unknown")
        object.__setattr__(self, "right_profile", str(self.right_profile).strip().lower() or "unknown")

    def to_dict(self) -> dict[str, Any]:
        return {
            "left": self.left,
            "right": self.right,
            "correlation": self.correlation,
            "absolute_correlation": self.absolute_correlation,
            "distance": self.distance,
            "left_family": self.left_family,
            "right_family": self.right_family,
            "left_profile": self.left_profile,
            "right_profile": self.right_profile,
        }


@dataclass(frozen=True, slots=True)
class PortfolioCorrelationMatrix:
    """
    Matrice de corrélation portefeuille.
    """

    labels: tuple[str, ...]
    matrix: np.ndarray

    vectors: dict[str, np.ndarray] = field(default_factory=dict)
    pairs: tuple[PortfolioCorrelationPair, ...] = ()

    average_correlation: float = 0.0
    average_absolute_correlation: float = 0.0
    max_absolute_correlation: float = 0.0
    min_correlation: float = 0.0
    max_correlation: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "labels", tuple(self.labels))
        object.__setattr__(self, "matrix", np.asarray(self.matrix, dtype=float))
        object.__setattr__(self, "vectors", {str(k): np.asarray(v, dtype=float) for k, v in dict(self.vectors).items()})
        object.__setattr__(self, "pairs", tuple(self.pairs))
        object.__setattr__(self, "average_correlation", float(self.average_correlation))
        object.__setattr__(self, "average_absolute_correlation", float(self.average_absolute_correlation))
        object.__setattr__(self, "max_absolute_correlation", float(self.max_absolute_correlation))
        object.__setattr__(self, "min_correlation", float(self.min_correlation))
        object.__setattr__(self, "max_correlation", float(self.max_correlation))

    @property
    def size(self) -> int:
        return len(self.labels)

    def pair(self, left: str, right: str) -> PortfolioCorrelationPair | None:
        if left == right:
            return PortfolioCorrelationPair(left, right, 1.0, 1.0, 0.0)
        for pair in self.pairs:
            if (pair.left == left and pair.right == right) or (pair.left == right and pair.right == left):
                return pair
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "labels": list(self.labels),
            "matrix": self.matrix.tolist(),
            "pairs": [pair.to_dict() for pair in self.pairs],
            "average_correlation": self.average_correlation,
            "average_absolute_correlation": self.average_absolute_correlation,
            "max_absolute_correlation": self.max_absolute_correlation,
            "min_correlation": self.min_correlation,
            "max_correlation": self.max_correlation,
        }


class PortfolioCorrelationAnalyzer:
    """
    Calcule les corrélations entre résultats d'exécution.
    """

    def __init__(
        self,
        settings: PortfolioCorrelationSettings | None = None,
        *,
        config: PortfolioConfig | Any | None = None,
    ) -> None:
        if settings is not None:
            self._settings = settings
        elif config is not None:
            self._settings = PortfolioCorrelationSettings.from_config(config)
        else:
            self._settings = PortfolioCorrelationSettings()
        self._settings = settings or PortfolioCorrelationSettings()

    @property
    def settings(self) -> PortfolioCorrelationSettings:
        return self._settings

    def build_vectors(self, results: Iterable[ExecutionResult]) -> dict[str, np.ndarray]:
        vectors: dict[str, np.ndarray] = {}
        for result in results:
            key = getattr(result, "subject_fingerprint", None) or getattr(result.execution_fingerprint, "digest", None)
            if not key:
                continue
            vectors[str(key)] = _result_series(result, size=self._settings.sample_points)
        return vectors

    def correlate(self, results: Iterable[ExecutionResult]) -> PortfolioCorrelationMatrix:
        results = tuple(results)
        labels = []
        vectors: list[np.ndarray] = []
        meta: dict[str, dict[str, str]] = {}

        for result in results:
            key = getattr(result, "subject_fingerprint", None) or getattr(result.execution_fingerprint, "digest", None)
            if not key:
                continue
            key = str(key)
            labels.append(key)
            vectors.append(_result_series(result, size=self._settings.sample_points))
            meta[key] = {
                "family": _family_key_from_result(result),
                "profile": _profile_name_from_result(result),
            }

        if not labels:
            empty = np.zeros((0, 0), dtype=float)
            return PortfolioCorrelationMatrix(
                labels=(),
                matrix=empty,
                vectors={},
                pairs=(),
                average_correlation=0.0,
                average_absolute_correlation=0.0,
                max_absolute_correlation=0.0,
                min_correlation=0.0,
                max_correlation=0.0,
            )

        matrix = np.vstack(vectors)
        corr = np.eye(len(labels), dtype=float)

        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                left = matrix[i]
                right = matrix[j]
                if np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
                    value = 0.0
                else:
                    value = float(np.corrcoef(left, right)[0, 1])
                    if not np.isfinite(value):
                        value = 0.0
                corr[i, j] = corr[j, i] = value

        pairs: list[PortfolioCorrelationPair] = []
        values: list[float] = []
        abs_values: list[float] = []

        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                value = float(corr[i, j])
                abs_value = abs(value)
                values.append(value)
                abs_values.append(abs_value)
                pairs.append(
                    PortfolioCorrelationPair(
                        left=labels[i],
                        right=labels[j],
                        correlation=value,
                        absolute_correlation=abs_value,
                        distance=1.0 - abs_value,
                        left_family=meta[labels[i]]["family"],
                        right_family=meta[labels[j]]["family"],
                        left_profile=meta[labels[i]]["profile"],
                        right_profile=meta[labels[j]]["profile"],
                    )
                )

        vectors_map = {label: vectors[idx] for idx, label in enumerate(labels)}

        return PortfolioCorrelationMatrix(
            labels=tuple(labels),
            matrix=corr,
            vectors=vectors_map,
            pairs=tuple(pairs),
            average_correlation=_safe_mean(values),
            average_absolute_correlation=_safe_mean(abs_values),
            max_absolute_correlation=_safe_max(abs_values),
            min_correlation=min(values) if values else 0.0,
            max_correlation=max(values) if values else 0.0,
        )

    def distance(self, results: Iterable[ExecutionResult]) -> PortfolioCorrelationMatrix:
        return self.correlate(results)

    def __repr__(self) -> str:
        return f"PortfolioCorrelationAnalyzer(sample_points={self._settings.sample_points})"


def _family_key_from_result(result: ExecutionResult) -> str:
    metadata = _to_mapping(result.metadata)
    for key in ("family", "target_family", "portfolio_family"):
        if key in metadata and metadata[key] is not None:
            value = str(metadata[key]).strip().lower()
            if value:
                return value
    candidate = getattr(result, "candidate", None)
    hypothesis = getattr(result, "hypothesis", None)
    for source in (candidate, hypothesis):
        if source is None:
            continue
        src_meta = _to_mapping(getattr(source, "metadata", None))
        for key in ("family", "target_family", "portfolio_family"):
            if key in src_meta and src_meta[key] is not None:
                value = str(src_meta[key]).strip().lower()
                if value:
                    return value
    try:
        conditions = getattr(hypothesis, "conditions", None)
        if conditions:
            fam = conditions[0].left.economic_family.value
            if fam:
                return str(fam).strip().lower()
    except Exception:
        pass
    return "unknown"


def _profile_name_from_result(result: ExecutionResult) -> str:
    profile = getattr(result, "profile", None)
    if profile is not None and getattr(profile, "name", None):
        name = str(profile.name).strip().lower()
        if name:
            return name
    metadata = _to_mapping(result.metadata)
    for key in ("profile_name", "strategy_name", "einher_name"):
        if key in metadata and metadata[key] is not None:
            value = str(metadata[key]).strip().lower()
            if value:
                return value
    return "unknown"