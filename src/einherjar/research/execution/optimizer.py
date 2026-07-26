# execution/optimizer.py
"""
==========================================================
Execution Optimizer
==========================================================

Teste des variantes de paramètres d'exécution sur une
stratégie validée.

Ce module reste volontairement limité à l'exécution :
- fees,
- slippage,
- spread,
- quantity,
- direction,
- max_open_positions,
- éventuellement le prix de référence.

Il n'optimise pas la stratégie elle-même.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from itertools import product
from math import exp
from typing import Any
from typing import Callable
from typing import Iterable
from typing import Mapping
from typing import Sequence

import numpy as np

from config.execution import ExecutionConfig
from core.context import EngineContext
from models.candidate import Candidate
from models.hypothesis import Hypothesis
from models.validated_candidate import ValidatedCandidate

from .executor import ExecutionEngine
from .execution_report import ExecutionResult

__all__ = [
    "OptimizationSettings",
    "OptimizationTrial",
    "OptimizationResult",
    "ExecutionOptimizer",
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


def _normalize_direction(value: Any, default: str = "long") -> str:
    text = str(value or default).strip().lower()
    if text in {"long", "buy", "bull", "bullish"}:
        return "long"
    if text in {"short", "sell", "bear", "bearish"}:
        return "short"
    raise ValueError(f"Unknown direction: {value!r}")


def _as_tuple(values: Any) -> tuple[Any, ...]:
    if values is None:
        return ()
    if isinstance(values, tuple):
        return values
    if isinstance(values, list):
        return tuple(values)
    if isinstance(values, set):
        return tuple(values)
    return (values,)


def _subject_to_candidate(subject: Any) -> tuple[Any, Hypothesis]:
    if isinstance(subject, ValidatedCandidate):
        return subject.candidate, subject.hypothesis

    if isinstance(subject, Candidate):
        return subject, subject.hypothesis

    if isinstance(subject, Hypothesis):
        return Candidate(hypothesis=subject), subject

    if hasattr(subject, "candidate") and hasattr(subject.candidate, "hypothesis"):
        candidate = subject.candidate
        return candidate, candidate.hypothesis

    if hasattr(subject, "hypothesis"):
        hypothesis = subject.hypothesis
        if isinstance(hypothesis, Hypothesis):
            return subject, hypothesis

    raise TypeError("subject must be a ValidatedCandidate, a Candidate or a Hypothesis.")


def _bounded_profit_factor(value: float) -> float:
    if not np.isfinite(value):
        return 1.0 if value > 0 else 0.0
    if value <= 0:
        return 0.0
    return float(1.0 - np.exp(-max(0.0, value - 1.0)))


def _bounded_drawdown(value: float) -> float:
    value = max(0.0, float(value))
    return float(1.0 / (1.0 + value))


def _bounded_win_rate(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _bounded_health(value: bool) -> float:
    return 1.0 if value else 0.0


@dataclass(frozen=True, slots=True)
class OptimizationSettings:
    """
    Paramètres de l'optimiseur d'exécution.
    """

    max_trials: int = 24
    top_k: int = 1

    require_healthy: bool = False
    min_trades: int = 1

    fees_grid: tuple[float, ...] = ()
    slippage_grid: tuple[float, ...] = ()
    spread_grid: tuple[float, ...] = ()
    quantity_grid: tuple[float, ...] = ()
    direction_grid: tuple[str, ...] = ()
    max_open_positions_grid: tuple[int, ...] = ()

    allow_matrix_search: bool = False
    allow_dataset_search: bool = False

    weight_total_pnl: float = 0.35
    weight_expectancy: float = 0.20
    weight_win_rate: float = 0.20
    weight_profit_factor: float = 0.15
    weight_health: float = 0.10

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_trials", max(1, _coerce_int(self.max_trials, 24)))
        object.__setattr__(self, "top_k", max(1, _coerce_int(self.top_k, 1)))
        object.__setattr__(self, "require_healthy", _coerce_bool(self.require_healthy, False))
        object.__setattr__(self, "min_trades", max(1, _coerce_int(self.min_trades, 1)))

        object.__setattr__(self, "fees_grid", tuple(float(v) for v in self.fees_grid))
        object.__setattr__(self, "slippage_grid", tuple(float(v) for v in self.slippage_grid))
        object.__setattr__(self, "spread_grid", tuple(float(v) for v in self.spread_grid))
        object.__setattr__(self, "quantity_grid", tuple(float(v) for v in self.quantity_grid))
        object.__setattr__(self, "direction_grid", tuple(_normalize_direction(v) for v in self.direction_grid))
        object.__setattr__(self, "max_open_positions_grid", tuple(max(1, _coerce_int(v, 1)) for v in self.max_open_positions_grid))

        object.__setattr__(self, "allow_matrix_search", _coerce_bool(self.allow_matrix_search, False))
        object.__setattr__(self, "allow_dataset_search", _coerce_bool(self.allow_dataset_search, False))

        object.__setattr__(self, "weight_total_pnl", max(0.0, _coerce_float(self.weight_total_pnl, 0.35)))
        object.__setattr__(self, "weight_expectancy", max(0.0, _coerce_float(self.weight_expectancy, 0.20)))
        object.__setattr__(self, "weight_win_rate", max(0.0, _coerce_float(self.weight_win_rate, 0.20)))
        object.__setattr__(self, "weight_profit_factor", max(0.0, _coerce_float(self.weight_profit_factor, 0.15)))
        object.__setattr__(self, "weight_health", max(0.0, _coerce_float(self.weight_health, 0.10)))

        total = (
            self.weight_total_pnl
            + self.weight_expectancy
            + self.weight_win_rate
            + self.weight_profit_factor
            + self.weight_health
        )
        if total <= 0:
            object.__setattr__(self, "weight_total_pnl", 0.35)
            object.__setattr__(self, "weight_expectancy", 0.20)
            object.__setattr__(self, "weight_win_rate", 0.20)
            object.__setattr__(self, "weight_profit_factor", 0.15)
            object.__setattr__(self, "weight_health", 0.10)

    @classmethod
    def from_config(cls, config: Any | None) -> "OptimizationSettings":
        if config is None:
            return cls()

        optimizer = _to_mapping(
            _to_mapping(config).get("optimizer", _to_mapping(config).get("execution_optimizer", {}))
        )
        execution = _to_mapping(_to_mapping(config).get("execution", {}))

        def candidate(*keys: str, default: Any = None) -> Any:
            for key in keys:
                if key in optimizer:
                    return optimizer[key]
                if key in execution:
                    return execution[key]
                if key in _to_mapping(config):
                    root = _to_mapping(config)
                    if key in root:
                        return root[key]
            return default

        return cls(
            max_trials=_coerce_int(candidate("max_trials", "trials", default=24), 24),
            top_k=_coerce_int(candidate("top_k", "keep_top_k", default=1), 1),
            require_healthy=_coerce_bool(candidate("require_healthy", default=False), False),
            min_trades=_coerce_int(candidate("min_trades", default=1), 1),
            fees_grid=_as_tuple(candidate("fees_grid", "fees_candidates", default=())),
            slippage_grid=_as_tuple(candidate("slippage_grid", "slippage_candidates", default=())),
            spread_grid=_as_tuple(candidate("spread_grid", "spread_candidates", default=())),
            quantity_grid=_as_tuple(candidate("quantity_grid", "size_grid", default=())),
            direction_grid=_as_tuple(candidate("direction_grid", "directions", default=())),
            max_open_positions_grid=_as_tuple(candidate("max_open_positions_grid", "max_positions_grid", default=())),
            allow_matrix_search=_coerce_bool(candidate("allow_matrix_search", default=False), False),
            allow_dataset_search=_coerce_bool(candidate("allow_dataset_search", default=False), False),
            weight_total_pnl=_coerce_float(candidate("weight_total_pnl", default=0.35), 0.35),
            weight_expectancy=_coerce_float(candidate("weight_expectancy", default=0.20), 0.20),
            weight_win_rate=_coerce_float(candidate("weight_win_rate", default=0.20), 0.20),
            weight_profit_factor=_coerce_float(candidate("weight_profit_factor", default=0.15), 0.15),
            weight_health=_coerce_float(candidate("weight_health", default=0.10), 0.10),
        )


@dataclass(frozen=True, slots=True)
class OptimizationTrial:
    """
    Essai d'optimisation pour une configuration donnée.
    """

    config: ExecutionConfig
    quantity: float
    direction: str

    result: ExecutionResult
    score: float

    rank: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "quantity", max(0.0, float(self.quantity)))
        object.__setattr__(self, "direction", _normalize_direction(self.direction))
        object.__setattr__(self, "score", float(self.score))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": {
                "fees": self.config.fees,
                "slippage": self.config.slippage,
                "spread": self.config.spread,
                "allow_long": self.config.allow_long,
                "allow_short": self.config.allow_short,
                "max_open_positions": self.config.max_open_positions,
            },
            "quantity": self.quantity,
            "direction": self.direction,
            "result": self.result.to_dict(summary_only=True),
            "score": self.score,
            "rank": self.rank,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    """
    Résultat global d'une optimisation d'exécution.
    """

    best_trial: OptimizationTrial | None
    trials: tuple[OptimizationTrial, ...]
    baseline_config: ExecutionConfig

    best_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "trials", tuple(self.trials))
        object.__setattr__(self, "best_score", float(self.best_score))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def best_config(self) -> ExecutionConfig | None:
        return None if self.best_trial is None else self.best_trial.config

    @property
    def best_result(self) -> ExecutionResult | None:
        return None if self.best_trial is None else self.best_trial.result

    def to_dict(self) -> dict[str, Any]:
        return {
            "best_trial": None if self.best_trial is None else self.best_trial.to_dict(),
            "trials": [trial.to_dict() for trial in self.trials],
            "baseline_config": {
                "fees": self.baseline_config.fees,
                "slippage": self.baseline_config.slippage,
                "spread": self.baseline_config.spread,
                "allow_long": self.baseline_config.allow_long,
                "allow_short": self.baseline_config.allow_short,
                "max_open_positions": self.baseline_config.max_open_positions,
            },
            "best_score": self.best_score,
            "metadata": dict(self.metadata),
        }


class ExecutionOptimizer:
    """
    Explore des variantes de paramètres d'exécution et garde
    la meilleure selon un score composite.
    """

    def __init__(
        self,
        *,
        base_config: ExecutionConfig | None = None,
        settings: OptimizationSettings | None = None,
        engine_factory: Callable[[ExecutionConfig], ExecutionEngine] | None = None,
        knowledge: Any | None = None,
    ) -> None:
        self._base_config = base_config or ExecutionConfig()
        self._settings = settings or OptimizationSettings()
        self._engine_factory = engine_factory or (lambda cfg: ExecutionEngine(config={"execution": cfg}))
        self._knowledge = knowledge

    @property
    def base_config(self) -> ExecutionConfig:
        return self._base_config

    @property
    def settings(self) -> OptimizationSettings:
        return self._settings

    @property
    def knowledge(self) -> Any | None:
        return self._knowledge

    def optimize(
        self,
        subject: Any,
        *,
        dataset: Any | None = None,
        matrix: Any | None = None,
        prices: Any | None = None,
        timestamps: Sequence[Any] | None = None,
        price_column: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> OptimizationResult:
        candidate, hypothesis = _subject_to_candidate(subject)

        configs = self._candidate_variants()
        trials: list[OptimizationTrial] = []

        for index, (config, quantity, direction) in enumerate(configs):
            engine = self._engine_factory(config)

            result = engine.execute(
                subject,
                dataset=dataset,
                matrix=matrix,
                prices=prices,
                timestamps=timestamps,
                direction=direction,
                quantity=quantity,
                price_column=price_column,
                metadata=metadata,
            )

            score = self._score_result(result)
            trial = OptimizationTrial(
                config=config,
                quantity=quantity,
                direction=direction,
                result=result,
                score=score,
                rank=index,
                metadata={
                    "subject_fingerprint": result.subject_fingerprint,
                    "hypothesis_conditions": len(hypothesis.conditions),
                    "trade_count": result.trade_count,
                },
            )
            trials.append(trial)

            if self._knowledge is not None and hasattr(self._knowledge, "remember"):
                self._knowledge.remember(result)

            if len(trials) >= self._settings.max_trials:
                break

        ordered = tuple(sorted(trials, key=lambda item: (-item.score, -item.result.total_pnl, item.rank)))
        best_trial = ordered[0] if ordered else None
        best_score = best_trial.score if best_trial is not None else 0.0

        return OptimizationResult(
            best_trial=best_trial,
            trials=ordered[: self._settings.top_k],
            baseline_config=self._base_config,
            best_score=best_score,
            metadata=dict(metadata or {}),
        )

    def tune(self, subject: Any, **kwargs: Any) -> OptimizationResult:
        return self.optimize(subject, **kwargs)

    def search(self, subject: Any, **kwargs: Any) -> OptimizationResult:
        return self.optimize(subject, **kwargs)

    def _candidate_variants(self) -> tuple[tuple[ExecutionConfig, float, str], ...]:
        fees_values = self._resolve_grid(
            self._settings.fees_grid,
            base=self._base_config.fees,
            factors=(0.5, 1.0, 1.5),
            minimum=0.0,
        )
        slippage_values = self._resolve_grid(
            self._settings.slippage_grid,
            base=self._base_config.slippage,
            factors=(0.5, 1.0, 1.5),
            minimum=0.0,
        )
        spread_values = self._resolve_grid(
            self._settings.spread_grid,
            base=self._base_config.spread,
            factors=(0.5, 1.0, 1.5),
            minimum=0.0,
        )
        quantity_values = self._resolve_grid(
            self._settings.quantity_grid,
            base=1.0,
            factors=(0.5, 1.0, 2.0),
            minimum=0.0001,
        )
        direction_values = self._settings.direction_grid or ("long", "short")
        max_positions_values = self._settings.max_open_positions_grid or (self._base_config.max_open_positions,)

        if not self._base_config.allow_long:
            direction_values = tuple(v for v in direction_values if v != "long") or ("short",)
        if not self._base_config.allow_short:
            direction_values = tuple(v for v in direction_values if v != "short") or ("long",)

        output: list[tuple[ExecutionConfig, float, str]] = []
        for fees, slippage, spread, quantity, direction, max_positions in product(
            fees_values,
            slippage_values,
            spread_values,
            quantity_values,
            direction_values,
            max_positions_values,
        ):
            cfg = ExecutionConfig(
                fees=float(max(0.0, fees)),
                slippage=float(max(0.0, slippage)),
                spread=float(max(0.0, spread)),
                allow_long=self._base_config.allow_long,
                allow_short=self._base_config.allow_short,
                max_open_positions=max(1, _coerce_int(max_positions, self._base_config.max_open_positions)),
            )
            output.append((cfg, float(quantity), _normalize_direction(direction)))

            if len(output) >= self._settings.max_trials * 4:
                break

        if not output:
            output.append((self._base_config, 1.0, "long"))

        return tuple(output)

    def _resolve_grid(
        self,
        grid: Sequence[float] | tuple[float, ...],
        *,
        base: float,
        factors: Sequence[float],
        minimum: float = 0.0,
    ) -> tuple[float, ...]:
        if grid:
            values = tuple(max(minimum, float(v)) for v in grid)
            return tuple(dict.fromkeys(values))

        values = tuple(max(minimum, float(base) * float(factor)) for factor in factors)
        if not values:
            values = (max(minimum, float(base)),)
        return tuple(dict.fromkeys(values))

    def _score_result(self, result: ExecutionResult) -> float:
        metrics = result.replay.metrics
        profile = result.profile
        diagnostics = result.diagnostics

        total_pnl_component = 0.5 * (1.0 + np.tanh(metrics.total_pnl / max(1.0, abs(metrics.total_pnl) + metrics.trade_count)))
        expectancy_component = 0.5 * (1.0 + np.tanh(metrics.expectancy))
        win_rate_component = _bounded_win_rate(metrics.win_rate)
        profit_factor_component = _bounded_profit_factor(metrics.profit_factor)

        health_component = 1.0 if diagnostics is None else _bounded_health(diagnostics.healthy)
        if self._settings.require_healthy and not health_component:
            health_component = 0.0

        drawdown_component = 1.0
        if profile is not None:
            drawdown_component = _bounded_drawdown(profile.max_drawdown)

        if metrics.trade_count < self._settings.min_trades:
            health_component *= 0.5

        score = (
            self._settings.weight_total_pnl * total_pnl_component
            + self._settings.weight_expectancy * expectancy_component
            + self._settings.weight_win_rate * win_rate_component
            + self._settings.weight_profit_factor * profit_factor_component
            + self._settings.weight_health * health_component
        )

        score *= 0.85 + 0.15 * drawdown_component

        if metrics.trade_count == 0:
            score *= 0.0

        if self._settings.require_healthy and not (diagnostics.healthy if diagnostics is not None else True):
            score *= 0.5

        return float(max(0.0, min(1.0, score)))

    def __repr__(self) -> str:
        return (
            "ExecutionOptimizer("
            f"max_trials={self._settings.max_trials}, "
            f"top_k={self._settings.top_k}"
            ")"
        )