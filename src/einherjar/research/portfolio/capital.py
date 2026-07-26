# portfolio/capital.py
"""
==========================================================
Portfolio Capital
==========================================================

Définit l'enveloppe de capital et la transforme en budget
alloué aux Einhers sélectionnés.

Le module ne choisit pas les stratégies :
- il prend une liste d'Einhers,
- applique les contraintes de capital,
- sort un plan de capital exploitable.
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
    "CapitalSettings",
    "CapitalPlanEntry",
    "CapitalPlan",
    "CapitalManager",
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


def _result_key(result: ExecutionResult) -> str:
    value = getattr(result, "subject_fingerprint", None)
    if value:
        return str(value)
    fp = getattr(result, "execution_fingerprint", None)
    if fp is not None:
        digest = getattr(fp, "digest", None)
        if digest:
            return str(digest)
    return ""


def _profile_name(result: ExecutionResult) -> str:
    profile = getattr(result, "profile", None)
    if profile is not None and getattr(profile, "name", None):
        name = str(profile.name).strip().lower()
        if name:
            return name

    metadata = _to_mapping(result.metadata)
    for key in ("profile_name", "strategy_name", "einher_name"):
        if key in metadata and metadata[key] is not None:
            name = str(metadata[key]).strip().lower()
            if name:
                return name
    return "unknown"


def _family_key(result: ExecutionResult) -> str:
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


@dataclass(frozen=True, slots=True)
class CapitalSettings:
    """
    Paramètres de capital.
    """

    total_capital: float = 1.0
    reserve_ratio: float = 0.0

    min_position_ratio: float = 0.0
    max_position_ratio: float = 0.35

    min_position_value: float = 0.0
    max_position_value: float = float("inf")

    min_positions: int = 1
    max_positions: int = 12

    allow_residual: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "total_capital", max(0.0, _coerce_float(self.total_capital, 1.0)))
        object.__setattr__(self, "reserve_ratio", _bounded_unit(_coerce_float(self.reserve_ratio, 0.0)))
        object.__setattr__(self, "min_position_ratio", _bounded_unit(_coerce_float(self.min_position_ratio, 0.0)))
        object.__setattr__(self, "max_position_ratio", _bounded_unit(_coerce_float(self.max_position_ratio, 0.35)))
        object.__setattr__(self, "min_position_value", max(0.0, _coerce_float(self.min_position_value, 0.0)))
        object.__setattr__(self, "max_position_value", max(0.0, _coerce_float(self.max_position_value, float("inf"))))
        object.__setattr__(self, "min_positions", max(1, _coerce_int(self.min_positions, 1)))
        object.__setattr__(self, "max_positions", max(self.min_positions, _coerce_int(self.max_positions, 12)))
        object.__setattr__(self, "allow_residual", _coerce_bool(self.allow_residual, True))

    @classmethod
    def from_config(cls, config: Any | None) -> "CapitalSettings":
        if config is None:
            return cls()

        root = _to_mapping(config)
        port = _to_mapping(root.get("portfolio", root.get("portfolio_config", {})))
        capital = _to_mapping(port.get("capital", port.get("capital_allocation", {})))

        return cls(
            total_capital=_coerce_float(capital.get("total_capital", root.get("total_capital", 1.0)), 1.0),
            reserve_ratio=_coerce_float(capital.get("reserve_ratio", root.get("reserve_ratio", 0.0)), 0.0),
            min_position_ratio=_coerce_float(capital.get("min_position_ratio", root.get("min_position_ratio", 0.0)), 0.0),
            max_position_ratio=_coerce_float(capital.get("max_position_ratio", root.get("max_position_ratio", 0.35)), 0.35),
            min_position_value=_coerce_float(capital.get("min_position_value", root.get("min_position_value", 0.0)), 0.0),
            max_position_value=_coerce_float(capital.get("max_position_value", root.get("max_position_value", float("inf"))), float("inf")),
            min_positions=_coerce_int(capital.get("min_positions", root.get("min_positions", 1)), 1),
            max_positions=_coerce_int(capital.get("max_positions", root.get("max_positions", 12)), 12),
            allow_residual=_coerce_bool(capital.get("allow_residual", root.get("allow_residual", True)), True),
        )


@dataclass(frozen=True, slots=True)
class CapitalPlanEntry:
    """
    Allocation de capital pour un Einher.
    """

    result: ExecutionResult
    weight: float
    capital: float
    family: str = "unknown"
    profile_name: str = "unknown"
    score: float = 0.0
    accepted: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "weight", max(0.0, float(self.weight)))
        object.__setattr__(self, "capital", max(0.0, float(self.capital)))
        object.__setattr__(self, "family", str(self.family).strip().lower() or "unknown")
        object.__setattr__(self, "profile_name", str(self.profile_name).strip().lower() or "unknown")
        object.__setattr__(self, "score", float(self.score))
        object.__setattr__(self, "accepted", _coerce_bool(self.accepted, True))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def subject_fingerprint(self) -> str:
        return _result_key(self.result)

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject_fingerprint": self.subject_fingerprint,
            "weight": self.weight,
            "capital": self.capital,
            "family": self.family,
            "profile_name": self.profile_name,
            "score": self.score,
            "accepted": self.accepted,
            "result": self.result.to_dict(summary_only=True),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class CapitalPlan:
    """
    Plan de capital final.
    """

    total_capital: float
    reserve_capital: float
    investable_capital: float

    entries: tuple[CapitalPlanEntry, ...] = ()
    settings: CapitalSettings = field(default_factory=CapitalSettings)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "total_capital", max(0.0, float(self.total_capital)))
        object.__setattr__(self, "reserve_capital", max(0.0, float(self.reserve_capital)))
        object.__setattr__(self, "investable_capital", max(0.0, float(self.investable_capital)))
        object.__setattr__(self, "entries", tuple(self.entries))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def allocated_capital(self) -> float:
        return float(sum(entry.capital for entry in self.entries if entry.accepted))

    @property
    def residual_capital(self) -> float:
        return max(0.0, self.investable_capital - self.allocated_capital)

    @property
    def utilization_ratio(self) -> float:
        if self.investable_capital <= 1e-12:
            return 0.0
        return min(1.0, self.allocated_capital / self.investable_capital)

    @property
    def selected_results(self) -> tuple[ExecutionResult, ...]:
        return tuple(entry.result for entry in self.entries if entry.accepted and entry.capital > 0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_capital": self.total_capital,
            "reserve_capital": self.reserve_capital,
            "investable_capital": self.investable_capital,
            "allocated_capital": self.allocated_capital,
            "residual_capital": self.residual_capital,
            "utilization_ratio": self.utilization_ratio,
            "entries": [entry.to_dict() for entry in self.entries],
            "settings": {
                "total_capital": self.settings.total_capital,
                "reserve_ratio": self.settings.reserve_ratio,
                "min_position_ratio": self.settings.min_position_ratio,
                "max_position_ratio": self.settings.max_position_ratio,
                "min_position_value": self.settings.min_position_value,
                "max_position_value": self.settings.max_position_value,
                "min_positions": self.settings.min_positions,
                "max_positions": self.settings.max_positions,
                "allow_residual": self.settings.allow_residual,
            },
            "metadata": dict(self.metadata),
        }


class CapitalManager:
    """
    Convertit une sélection en budget de capital.
    """

    def __init__(
        self,
        settings: CapitalSettings | None = None,
        *,
        config: PortfolioConfig | Any | None = None,
    ) -> None:
        self._settings = settings or CapitalSettings.from_config(config)

    @property
    def settings(self) -> CapitalSettings:
        return self._settings

    def plan(
        self,
        results: Iterable[ExecutionResult],
        *,
        weights: Sequence[float] | Mapping[str, float] | None = None,
        total_capital: float | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> CapitalPlan:
        results = tuple(results)
        settings = self._settings

        if total_capital is None:
            total_capital = settings.total_capital
        total_capital = max(0.0, float(total_capital))

        reserve_capital = total_capital * settings.reserve_ratio
        investable_capital = max(0.0, total_capital - reserve_capital)

        if not results:
            return CapitalPlan(
                total_capital=total_capital,
                reserve_capital=reserve_capital,
                investable_capital=investable_capital,
                entries=(),
                settings=settings,
                metadata=dict(metadata or {}),
            )

        weights_arr = self._normalize_weights(results, weights)
        weights_arr = self._enforce_bounds(weights_arr, investable_capital)

        target_capitals = investable_capital * weights_arr
        target_capitals = self._enforce_minimums(target_capitals, investable_capital)
        target_capitals = self._enforce_maximums(target_capitals, investable_capital)

        total = float(target_capitals.sum())
        if total > 1e-12:
            if not settings.allow_residual:
                target_capitals = target_capitals / total * investable_capital
            else:
                scale = min(1.0, investable_capital / total)
                target_capitals = target_capitals * scale

        entries: list[CapitalPlanEntry] = []
        for result, weight, capital in zip(results, weights_arr, target_capitals):
            entries.append(
                CapitalPlanEntry(
                    result=result,
                    weight=float(weight),
                    capital=float(capital),
                    family=_family_key(result),
                    profile_name=_profile_name(result),
                    score=self._entry_score(result, weight),
                    accepted=capital > 0,
                    metadata={
                        "subject_fingerprint": _result_key(result),
                    },
                )
            )

        accepted_count = sum(1 for entry in entries if entry.accepted)
        if accepted_count < settings.min_positions and len(entries) >= settings.min_positions:
            # ré-allocation minimale sur les meilleures entrées
            top = sorted(entries, key=lambda e: (-e.score, -e.weight, e.subject_fingerprint))[: settings.min_positions]
            min_cap = investable_capital / max(1, settings.min_positions)
            entries = []
            for result, weight in zip(results, weights_arr):
                capital = min_cap if _result_key(result) in {entry.subject_fingerprint for entry in top} else 0.0
                entries.append(
                    CapitalPlanEntry(
                        result=result,
                        weight=float(weight),
                        capital=float(capital),
                        family=_family_key(result),
                        profile_name=_profile_name(result),
                        score=self._entry_score(result, weight),
                        accepted=capital > 0,
                        metadata={"subject_fingerprint": _result_key(result)},
                    )
                )

        return CapitalPlan(
            total_capital=total_capital,
            reserve_capital=reserve_capital,
            investable_capital=investable_capital,
            entries=tuple(entries),
            settings=settings,
            metadata=dict(metadata or {}),
        )

    def allocate(
        self,
        results: Iterable[ExecutionResult],
        *,
        weights: Sequence[float] | Mapping[str, float] | None = None,
        total_capital: float | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> CapitalPlan:
        return self.plan(results, weights=weights, total_capital=total_capital, metadata=metadata)

    def _normalize_weights(
        self,
        results: Sequence[ExecutionResult],
        weights: Sequence[float] | Mapping[str, float] | None,
    ) -> np.ndarray:
        if weights is None:
            raw = np.asarray([self._entry_score(result, 1.0) for result in results], dtype=float)
        elif isinstance(weights, Mapping):
            raw_values = []
            for result in results:
                key = _result_key(result)
                raw_values.append(float(weights.get(key, weights.get(_profile_name(result), 0.0))))
            raw = np.asarray(raw_values, dtype=float)
        else:
            raw = np.asarray(list(weights), dtype=float)

        if raw.size != len(results):
            raise ValueError("weights must match results length.")

        raw = np.maximum(raw, 0.0)
        if raw.sum() <= 1e-12:
            raw = np.full(len(results), 1.0 / len(results), dtype=float)
        else:
            raw = raw / raw.sum()
        return raw

    def _enforce_bounds(self, weights: np.ndarray, investable_capital: float) -> np.ndarray:
        settings = self._settings
        if weights.size == 0:
            return weights

        weights = np.maximum(weights, 0.0)
        if weights.sum() <= 1e-12:
            weights = np.full(weights.size, 1.0 / weights.size, dtype=float)

        max_weight = settings.max_position_ratio
        min_weight = settings.min_position_ratio

        weights = np.clip(weights, min_weight, max_weight)
        total = float(weights.sum())
        if total <= 1e-12:
            weights = np.full(weights.size, 1.0 / weights.size, dtype=float)
        else:
            weights = weights / total

        return weights

    def _enforce_minimums(self, capitals: np.ndarray, investable_capital: float) -> np.ndarray:
        settings = self._settings
        if capitals.size == 0:
            return capitals

        if settings.min_position_value <= 0:
            return capitals

        positive = capitals > 0
        capitals = capitals.copy()
        capitals[positive] = np.maximum(capitals[positive], settings.min_position_value)

        total = float(capitals.sum())
        if total > investable_capital and total > 1e-12:
            capitals = capitals / total * investable_capital

        return capitals

    def _enforce_maximums(self, capitals: np.ndarray, investable_capital: float) -> np.ndarray:
        settings = self._settings
        if capitals.size == 0:
            return capitals

        max_value = settings.max_position_value
        max_ratio_value = settings.max_position_ratio * investable_capital
        cap = min(max_value, max_ratio_value) if np.isfinite(max_value) else max_ratio_value

        if cap <= 0:
            return np.zeros_like(capitals)

        capitals = np.minimum(capitals, cap)
        total = float(capitals.sum())
        if total > investable_capital and total > 1e-12:
            capitals = capitals / total * investable_capital
        return capitals

    def _entry_score(self, result: ExecutionResult, weight_hint: float) -> float:
        metrics = result.replay.metrics
        profile = getattr(result, "profile", None)
        diagnostics = getattr(result, "diagnostics", None)

        base = (
            0.25 * _bounded_unit(0.5 * (1.0 + np.tanh(metrics.total_pnl)))
            + 0.20 * _bounded_unit(metrics.expectancy)
            + 0.20 * _bounded_unit(metrics.win_rate)
            + 0.15 * _bounded_unit(1.0 - np.exp(-max(0.0, metrics.profit_factor - 1.0))) if np.isfinite(metrics.profit_factor) else 0.0
        )
        if not np.isfinite(base):
            base = 0.0

        drawdown = 1.0
        if profile is not None:
            drawdown = 1.0 / (1.0 + max(0.0, float(getattr(profile, "max_drawdown", 0.0))))
        health = 1.0 if diagnostics is None or bool(getattr(diagnostics, "healthy", True)) else 0.75
        return float(max(0.0, min(1.0, 0.5 * base + 0.3 * drawdown + 0.2 * health + 0.1 * _bounded_unit(weight_hint))))

    def __repr__(self) -> str:
        return (
            "CapitalManager("
            f"total_capital={self._settings.total_capital}, "
            f"reserve_ratio={self._settings.reserve_ratio}"
            ")"
        )