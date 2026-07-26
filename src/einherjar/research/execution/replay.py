# execution/replay.py
"""
==========================================================
Replay Engine
==========================================================

Simule l'exécution d'un ValidatedCandidate sur un jeu de
données.

Le replay :
- évalue l'hypothèse en masque booléen,
- transforme ce masque en trades,
- calcule les métriques d'exécution,
- produit un journal et une empreinte de replay.

Il ne valide rien et ne corrige rien.
"""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from typing import Any
from typing import Iterable
from typing import Mapping
from typing import Sequence

import numpy as np

from config.execution import ExecutionConfig
from models.candidate import Candidate
from models.condition import Condition
from models.feature import Feature
from models.fingerprint import Fingerprint
from models.fingerprint import fingerprint
from models.fingerprint import fingerprint_model
from models.hypothesis import Hypothesis
from models.journal import Journal
from models.trade import Trade
from models.validated_candidate import ValidatedCandidate

from .trade_builder import ExecutedTradeRecord
from .trade_builder import TradeBuilder

__all__ = [
    "ReplaySettings",
    "ReplayMetrics",
    "ReplayResult",
    "ReplayEngine",
]


# ==========================================================
# HELPERS
# ==========================================================

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


def _normalize_direction(value: Any, default: str = "long") -> str:
    text = str(value or default).strip().lower()
    if text in {"long", "buy", "bull", "bullish"}:
        return "long"
    if text in {"short", "sell", "bear", "bearish"}:
        return "short"
    raise ValueError(f"Unknown direction: {value!r}")


def _to_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}

    if hasattr(value, "to_dict") and callable(value.to_dict):
        value = value.to_dict()

    if isinstance(value, Mapping):
        return dict(value)

    return {}


def _as_float_array(values: Any) -> np.ndarray:
    if values is None:
        return np.asarray([], dtype=float)

    if isinstance(values, np.ndarray):
        if values.ndim == 0:
            return np.asarray([float(values)], dtype=float)
        return values.astype(float, copy=False).reshape(-1)

    if isinstance(values, (list, tuple, set)):
        return np.asarray(list(values), dtype=float).reshape(-1)

    try:
        return np.asarray([float(values)], dtype=float)
    except (TypeError, ValueError):
        return np.asarray([], dtype=float)


def _as_bool_array(values: Any) -> np.ndarray:
    if values is None:
        return np.asarray([], dtype=bool)

    if isinstance(values, np.ndarray):
        if values.ndim == 0:
            return np.asarray([bool(values)], dtype=bool)
        return values.astype(bool, copy=False).reshape(-1)

    if isinstance(values, (list, tuple, set)):
        return np.asarray(list(values), dtype=bool).reshape(-1)

    return np.asarray([bool(values)], dtype=bool)


def _ensure_2d_matrix(values: Any) -> np.ndarray:
    if values is None:
        return np.asarray([[]], dtype=float).reshape(0, 0)

    if isinstance(values, np.ndarray):
        if values.ndim == 1:
            return values.reshape(-1, 1).astype(float, copy=False)
        if values.ndim == 2:
            return values.astype(float, copy=False)
        raise ValueError("Feature matrix must be 1D or 2D.")

    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    if array.ndim == 2:
        return array
    raise ValueError("Feature matrix must be 1D or 2D.")


def _extract_field(source: Any, *names: str, default: Any = None) -> Any:
    if source is None:
        return default

    if isinstance(source, Mapping):
        for name in names:
            if name in source:
                return source[name]
        return default

    for name in names:
        if hasattr(source, name):
            return getattr(source, name)

    return default


def _extract_timestamps(source: Any, count: int) -> tuple[datetime, ...] | None:
    timestamps = _extract_field(source, "timestamps", "time", "index", default=None)
    if timestamps is None:
        return None

    timestamps = tuple(timestamps)
    if len(timestamps) != count:
        raise ValueError("timestamps length must match feature matrix length.")
    return tuple(
        _coerce_datetime(value, fallback_index=index)
        for index, value in enumerate(timestamps)
    )


def _coerce_datetime(value: Any, *, fallback_index: int = 0) -> datetime:
    if isinstance(value, datetime):
        return value

    if hasattr(value, "to_pydatetime") and callable(value.to_pydatetime):
        converted = value.to_pydatetime()
        if isinstance(converted, datetime):
            return converted

    if isinstance(value, np.datetime64):
        ts_ns = value.astype("datetime64[ns]").astype(np.int64)
        return datetime.utcfromtimestamp(ts_ns / 1_000_000_000)

    if isinstance(value, (int, float, np.integer, np.floating)):
        return datetime.utcfromtimestamp(float(value))

    return datetime.utcfromtimestamp(float(fallback_index))


def _extract_prices(source: Any, *, price_column: int = -1) -> np.ndarray | None:
    prices = _extract_field(source, "prices", "price", "close", "close_prices", default=None)
    if prices is not None:
        return _as_float_array(prices)

    matrix = _extract_field(source, "X", "x", "features", "matrix", "data", default=None)
    if matrix is None:
        return None

    matrix = _ensure_2d_matrix(matrix)
    if matrix.shape[1] == 0:
        return None

    col = price_column
    if col < 0:
        col = matrix.shape[1] + col
    col = max(0, min(matrix.shape[1] - 1, col))
    return matrix[:, col].astype(float, copy=False).reshape(-1)


def _extract_matrix(source: Any) -> np.ndarray | None:
    matrix = _extract_field(source, "X", "x", "features", "matrix", "data", default=None)
    if matrix is None:
        return None
    return _ensure_2d_matrix(matrix)


def _extract_subject_metadata(subject: Any) -> dict[str, Any]:
    if subject is None:
        return {}

    meta = _extract_field(subject, "metadata", default=None)
    return _to_mapping(meta)


def _condition_operator_value(condition: Condition) -> str:
    operator = getattr(condition, "operator", None)
    if hasattr(operator, "value"):
        return str(operator.value).strip().lower()
    return str(operator).strip().lower()


def _condition_signature(condition: Condition) -> tuple[Any, ...]:
    right = condition.right.column_index if isinstance(condition.right, Feature) else repr(condition.right)
    return (
        condition.left.column_index,
        _condition_operator_value(condition),
        "feature" if isinstance(condition.right, Feature) else "constant",
        right,
    )


def _safe_mean(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.mean(values))


def _safe_std(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.std(values))


def _max_drawdown(equity: np.ndarray) -> float:
    if equity.size == 0:
        return 0.0

    peak = np.maximum.accumulate(equity)
    drawdown = peak - equity
    return float(np.max(drawdown)) if drawdown.size else 0.0


def _profit_factor(pnls: np.ndarray) -> float:
    if pnls.size == 0:
        return 0.0

    gross_profit = float(np.sum(pnls[pnls > 0]))
    gross_loss = float(abs(np.sum(pnls[pnls < 0])))

    if gross_loss <= 1e-12:
        return float("inf") if gross_profit > 0 else 0.0

    return gross_profit / gross_loss


def _extract_candidate(subject: Any) -> tuple[Any, Hypothesis]:
    """
    Retourne l'objet candidat et son hypothèse sous-jacente.
    """
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

    raise TypeError("subject must be a Candidate, a ValidatedCandidate or a Hypothesis.")


def _evaluate_condition(
    condition: Condition,
    matrix: np.ndarray,
    *,
    previous_cache: dict[int, float],
) -> tuple[np.ndarray, dict[int, float]]:
    left_idx = condition.left.column_index
    left = matrix[:, left_idx]
    operator = _condition_operator_value(condition)
    updated_previous: dict[int, float] = {}

    if isinstance(condition.right, Feature):
        right = matrix[:, condition.right.column_index]
    else:
        right = condition.right

    if operator in {"cross_over", "crossover", "crossover"}:
        return _evaluate_crossover_condition(
            left_idx=left_idx,
            left=left,
            right=right,
            previous_cache=previous_cache,
            cross_under=False,
        )

    if operator in {"cross_under", "crossunder", "cross_under"}:
        return _evaluate_crossover_condition(
            left_idx=left_idx,
            left=left,
            right=right,
            previous_cache=previous_cache,
            cross_under=True,
        )

    if operator in {"between"}:
        if isinstance(right, (tuple, list, np.ndarray)) and len(right) == 2:
            low, high = right
            return (left >= low) & (left <= high), updated_previous
        return np.zeros(matrix.shape[0], dtype=bool), updated_previous

    if operator in {"is_true", "truthy"}:
        return left.astype(bool, copy=False), updated_previous

    if operator in {"is_false", "falsy"}:
        return ~left.astype(bool, copy=False), updated_previous

    if operator in {"gt", "greater_than", ">"}:
        return (left > right), updated_previous

    if operator in {"ge", "gte", "greater_equal", ">="}:
        return (left >= right), updated_previous

    if operator in {"lt", "less_than", "<"}:
        return (left < right), updated_previous

    if operator in {"le", "lte", "less_equal", "<="}:
        return (left <= right), updated_previous

    if operator in {"eq", "=="}:
        return np.isclose(left, right, equal_nan=False), updated_previous

    if operator in {"ne", "neq", "!="}:
        return ~np.isclose(left, right, equal_nan=False), updated_previous

    return np.zeros(matrix.shape[0], dtype=bool), updated_previous


def _evaluate_crossover_condition(
    *,
    left_idx: int,
    left: np.ndarray,
    right: Any,
    previous_cache: dict[int, float],
    cross_under: bool,
) -> tuple[np.ndarray, dict[int, float]]:
    if np.isscalar(right):
        right_values = np.full(left.shape[0], float(right), dtype=float)
    elif isinstance(right, np.ndarray):
        right_values = right.reshape(-1).astype(float, copy=False)
    elif isinstance(right, (list, tuple)):
        right_values = np.asarray(right, dtype=float).reshape(-1)
    else:
        try:
            right_values = np.asarray(right, dtype=float).reshape(-1)
        except Exception:
            right_values = np.full(left.shape[0], np.nan, dtype=float)

    if right_values.size not in {1, left.size}:
        if right_values.size > 0:
            right_values = np.resize(right_values, left.size)
        else:
            right_values = np.full(left.shape[0], np.nan, dtype=float)

    if right_values.size == 1:
        right_values = np.full(left.shape[0], float(right_values[0]), dtype=float)

    prev_left = np.empty_like(left, dtype=float)
    prev_right = np.empty_like(right_values, dtype=float)

    prev_left[0] = previous_cache.get(left_idx, left[0])
    prev_right[0] = right_values[0]

    if left.shape[0] > 1:
        prev_left[1:] = left[:-1]
        prev_right[1:] = right_values[:-1]

    if cross_under:
        mask = (prev_left >= prev_right) & (left < right_values)
    else:
        mask = (prev_left <= prev_right) & (left > right_values)

    previous_cache[left_idx] = float(left[-1])
    return mask, {left_idx: float(left[-1])}


def _evaluate_hypothesis_matrix(hypothesis: Hypothesis, matrix: np.ndarray) -> np.ndarray:
    if not hasattr(hypothesis, "conditions"):
        raise TypeError("hypothesis must expose conditions.")

    if matrix.ndim != 2:
        raise ValueError("Feature matrix must be 2-dimensional.")

    if len(hypothesis.conditions) == 0:
        return np.zeros(matrix.shape[0], dtype=bool)

    mask = np.ones(matrix.shape[0], dtype=bool)
    previous_cache: dict[int, float] = {}

    for condition in hypothesis.conditions:
        condition_mask, updated = _evaluate_condition(
            condition,
            matrix,
            previous_cache=previous_cache,
        )
        previous_cache.update(updated)
        mask &= condition_mask
        if not mask.any():
            break

    return mask


# ==========================================================
# SETTINGS
# ==========================================================

@dataclass(frozen=True, slots=True)
class ReplaySettings:
    """
    Paramètres du replay.
    """

    execution: ExecutionConfig = field(default_factory=ExecutionConfig)

    quantity: float = 1.0
    price_column: int = -1
    direction: str = "long"

    close_on_end: bool = True
    allow_synthetic_timestamps: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "quantity", max(0.0, _coerce_float(self.quantity, 1.0)))
        object.__setattr__(self, "price_column", _coerce_int(self.price_column, -1))
        object.__setattr__(self, "direction", _normalize_direction(self.direction))
        object.__setattr__(self, "close_on_end", _coerce_bool(self.close_on_end, True))
        object.__setattr__(self, "allow_synthetic_timestamps", _coerce_bool(self.allow_synthetic_timestamps, True))

    @classmethod
    def from_config(cls, config: Any | None) -> "ReplaySettings":
        if config is None:
            return cls()

        execution = _extract_field(config, "execution", "execution_config", default=None)
        if isinstance(execution, Mapping):
            execution = ExecutionConfig(**dict(execution))
        elif not isinstance(execution, ExecutionConfig):
            execution = ExecutionConfig()

        replay = _extract_field(config, "replay", "replay_config", default=config)

        return cls(
            execution=execution,
            quantity=_coerce_float(_extract_field(replay, "quantity", default=1.0), 1.0),
            price_column=_coerce_int(_extract_field(replay, "price_column", default=-1), -1),
            direction=_normalize_direction(_extract_field(replay, "direction", default="long")),
            close_on_end=_coerce_bool(_extract_field(replay, "close_on_end", default=True), True),
            allow_synthetic_timestamps=_coerce_bool(_extract_field(replay, "allow_synthetic_timestamps", default=True), True),
        )


# ==========================================================
# METRICS
# ==========================================================

@dataclass(frozen=True, slots=True)
class ReplayMetrics:
    """
    Synthèse d'un replay.
    """

    bar_count: int
    signal_true_count: int
    signal_coverage: float

    trade_count: int
    winning_trades: int
    losing_trades: int

    total_pnl: float
    gross_profit: float
    gross_loss: float
    profit_factor: float

    average_trade_pnl: float
    median_trade_pnl: float
    pnl_std: float

    average_duration_seconds: float
    average_duration_bars: float
    max_drawdown: float

    exposure_ratio: float
    win_rate: float
    expectancy: float

    direction: str
    quantity: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "bar_count": self.bar_count,
            "signal_true_count": self.signal_true_count,
            "signal_coverage": self.signal_coverage,
            "trade_count": self.trade_count,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "total_pnl": self.total_pnl,
            "gross_profit": self.gross_profit,
            "gross_loss": self.gross_loss,
            "profit_factor": self.profit_factor,
            "average_trade_pnl": self.average_trade_pnl,
            "median_trade_pnl": self.median_trade_pnl,
            "pnl_std": self.pnl_std,
            "average_duration_seconds": self.average_duration_seconds,
            "average_duration_bars": self.average_duration_bars,
            "max_drawdown": self.max_drawdown,
            "exposure_ratio": self.exposure_ratio,
            "win_rate": self.win_rate,
            "expectancy": self.expectancy,
            "direction": self.direction,
            "quantity": self.quantity,
        }


# ==========================================================
# RESULT
# ==========================================================

@dataclass(slots=True)
class ReplayResult:
    """
    Résultat complet d'un replay.
    """

    subject_fingerprint: str
    execution_fingerprint: Fingerprint

    validated_candidate: ValidatedCandidate | None
    candidate: Any
    hypothesis: Hypothesis

    journal: Journal
    trades: tuple[Trade, ...]
    records: tuple[ExecutedTradeRecord, ...]

    signal_mask: np.ndarray
    prices: np.ndarray
    timestamps: tuple[datetime, ...]

    metrics: ReplayMetrics
    diagnostics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "trades", tuple(self.trades))
        object.__setattr__(self, "records", tuple(self.records))
        object.__setattr__(self, "signal_mask", self.signal_mask.astype(bool, copy=False).reshape(-1))
        object.__setattr__(self, "prices", self.prices.astype(float, copy=False).reshape(-1))
        object.__setattr__(self, "timestamps", tuple(self.timestamps))
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def trade_count(self) -> int:
        return self.metrics.trade_count

    @property
    def total_pnl(self) -> float:
        return self.metrics.total_pnl

    @property
    def win_rate(self) -> float:
        return self.metrics.win_rate

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject_fingerprint": self.subject_fingerprint,
            "execution_fingerprint": self.execution_fingerprint.to_dict(),
            "validated_candidate": (
                None if self.validated_candidate is None else self.validated_candidate.to_dict()
            ),
            "candidate": self.candidate.to_dict() if hasattr(self.candidate, "to_dict") else repr(self.candidate),
            "hypothesis": self.hypothesis.to_dict() if hasattr(self.hypothesis, "to_dict") else repr(self.hypothesis),
            "journal": self.journal.to_dict(),
            "trades": [trade.to_dict() for trade in self.trades],
            "records": [record.to_dict() for record in self.records],
            "signal_mask": self.signal_mask.astype(bool).tolist(),
            "prices": self.prices.tolist(),
            "timestamps": [ts.isoformat() for ts in self.timestamps],
            "metrics": self.metrics.to_dict(),
            "diagnostics": dict(self.diagnostics),
            "metadata": dict(self.metadata),
        }

    def __repr__(self) -> str:
        return (
            "ReplayResult("
            f"trades={self.trade_count}, "
            f"pnl={self.total_pnl:.4f}, "
            f"win_rate={self.win_rate:.3f}"
            ")"
        )


# ==========================================================
# ENGINE
# ==========================================================

class ReplayEngine:
    """
    Simule l'exécution d'une hypothèse validée.

    Le moteur accepte un ValidatedCandidate, un Candidate ou
    directement une Hypothesis. Il extrait les matrices et les
    prix, évalue le signal, puis construit les trades.
    """

    def __init__(
        self,
        settings: ReplaySettings | None = None,
        *,
        config: Any | None = None,
        trade_builder: TradeBuilder | None = None,
    ) -> None:
        self._settings = settings or ReplaySettings.from_config(config)
        self._trade_builder = trade_builder or TradeBuilder(
            config=self._settings.execution,
            quantity=self._settings.quantity,
        )

    @classmethod
    def from_config(
        cls,
        config: Any,
        *,
        trade_builder: TradeBuilder | None = None,
    ) -> "ReplayEngine":
        return cls(
            settings=ReplaySettings.from_config(config),
            config=config,
            trade_builder=trade_builder,
        )

    @property
    def settings(self) -> ReplaySettings:
        return self._settings

    @property
    def trade_builder(self) -> TradeBuilder:
        return self._trade_builder

    # ==================================================
    # PUBLIC API
    # ==================================================

    def run(
        self,
        subject: Any,
        *,
        dataset: Any | None = None,
        matrix: Any | None = None,
        prices: Any | None = None,
        timestamps: Sequence[Any] | None = None,
        direction: str | None = None,
        quantity: float | None = None,
        price_column: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ReplayResult:
        candidate, hypothesis = _extract_candidate(subject)
        candidate_fingerprint = self._subject_fingerprint(candidate, hypothesis)

        matrix, prices, timestamps = self._resolve_inputs(
            dataset=dataset,
            matrix=matrix,
            prices=prices,
            timestamps=timestamps,
            price_column=price_column,
        )

        if matrix is None:
            raise ValueError("A feature matrix is required for replay.")
        if prices is None:
            raise ValueError("A price series is required for replay.")

        if matrix.shape[0] != prices.size:
            raise ValueError("Feature matrix and price series must have the same number of rows.")

        if timestamps is None:
            if not self._settings.allow_synthetic_timestamps:
                raise ValueError("Timestamps are required for replay.")
            timestamps = tuple(
                _coerce_datetime(i, fallback_index=i)
                for i in range(prices.size)
            )
        else:
            if len(timestamps) != prices.size:
                raise ValueError("Timestamps length must match price series length.")

        signal_mask = _evaluate_hypothesis_matrix(hypothesis, matrix)

        trade_direction = self._resolve_direction(candidate, hypothesis, direction)
        quantity = self._settings.quantity if quantity is None else max(0.0, _coerce_float(quantity, self._settings.quantity))

        records = self._trade_builder.build_from_signal_mask(
            signal_mask=signal_mask,
            prices=prices,
            timestamps=timestamps,
            direction=trade_direction,
            quantity=quantity,
            max_open_positions=self._settings.execution.max_open_positions,
            close_on_end=self._settings.close_on_end,
            metadata={
                **_extract_subject_metadata(subject),
                **dict(metadata or {}),
            },
        )

        journal = Journal(record.trade for record in records)
        metrics = self._build_metrics(
            signal_mask=signal_mask,
            prices=prices,
            records=records,
            direction=trade_direction,
            quantity=quantity,
        )
        exec_fp = self._build_execution_fingerprint(
            candidate_fingerprint=candidate_fingerprint,
            hypothesis=hypothesis,
            records=records,
            metrics=metrics,
            direction=trade_direction,
            quantity=quantity,
            price_count=prices.size,
        )

        diagnostics = self._build_diagnostics(
            hypothesis=hypothesis,
            records=records,
            signal_mask=signal_mask,
            prices=prices,
            metrics=metrics,
        )

        return ReplayResult(
            subject_fingerprint=candidate_fingerprint,
            execution_fingerprint=exec_fp,
            validated_candidate=subject if isinstance(subject, ValidatedCandidate) else None,
            candidate=candidate,
            hypothesis=hypothesis,
            journal=journal,
            trades=journal.trades,
            records=records,
            signal_mask=signal_mask,
            prices=prices,
            timestamps=timestamps,
            metrics=metrics,
            diagnostics=diagnostics,
            metadata={
                **_extract_subject_metadata(subject),
                **dict(metadata or {}),
            },
        )

    def replay(
        self,
        subject: Any,
        **kwargs: Any,
    ) -> ReplayResult:
        return self.run(subject, **kwargs)

    def simulate(
        self,
        subject: Any,
        **kwargs: Any,
    ) -> ReplayResult:
        return self.run(subject, **kwargs)

    # ==================================================
    # INPUT RESOLUTION
    # ==================================================

    def _resolve_inputs(
        self,
        *,
        dataset: Any | None,
        matrix: Any | None,
        prices: Any | None,
        timestamps: Sequence[Any] | None,
        price_column: int | None,
    ) -> tuple[np.ndarray | None, np.ndarray | None, tuple[datetime, ...] | None]:
        source = dataset if dataset is not None else {}

        if matrix is None:
            matrix = _extract_matrix(source)
        else:
            matrix = _ensure_2d_matrix(matrix)

        if prices is None:
            price_col = self._settings.price_column if price_column is None else _coerce_int(price_column, self._settings.price_column)
            prices = _extract_prices(source, price_column=price_col)
        else:
            prices = _as_float_array(prices)

        if timestamps is None:
            timestamps = _extract_timestamps(source, count=matrix.shape[0] if matrix is not None else (prices.size if prices is not None else 0))

        return matrix, prices, timestamps

    def _resolve_direction(
        self,
        candidate: Any,
        hypothesis: Hypothesis,
        explicit_direction: str | None,
    ) -> str:
        if explicit_direction is not None:
            return _normalize_direction(explicit_direction)

        for source in (candidate, hypothesis):
            meta = _extract_subject_metadata(source)
            for key in ("direction", "side", "trade_direction", "execution_direction"):
                if key in meta:
                    return _normalize_direction(meta[key])

        return self._settings.direction

    def _subject_fingerprint(self, candidate: Any, hypothesis: Hypothesis) -> str:
        try:
            return fingerprint_model(candidate)
        except Exception:
            pass

        try:
            return fingerprint_model(hypothesis)
        except Exception:
            pass

        return fingerprint(
            {
                "candidate": repr(candidate),
                "hypothesis": repr(hypothesis),
            }
        )

    # ==================================================
    # METRICS
    # ==================================================

    def _build_metrics(
        self,
        *,
        signal_mask: np.ndarray,
        prices: np.ndarray,
        records: tuple[ExecutedTradeRecord, ...],
        direction: str,
        quantity: float,
    ) -> ReplayMetrics:
        pnls = np.asarray([record.trade.pnl for record in records], dtype=float)
        durations_seconds = np.asarray([record.duration.total_seconds() for record in records], dtype=float)
        durations_bars = np.asarray([record.exit_index - record.entry_index + 1 for record in records], dtype=float)

        trade_count = int(records.__len__())
        winning_trades = int(sum(record.trade.is_profitable for record in records))
        losing_trades = trade_count - winning_trades

        total_pnl = float(np.sum(pnls)) if pnls.size > 0 else 0.0
        gross_profit = float(np.sum(pnls[pnls > 0])) if pnls.size > 0 else 0.0
        gross_loss = float(abs(np.sum(pnls[pnls < 0]))) if pnls.size > 0 else 0.0

        average_trade_pnl = float(np.mean(pnls)) if pnls.size > 0 else 0.0
        median_trade_pnl = float(np.median(pnls)) if pnls.size > 0 else 0.0
        pnl_std = float(np.std(pnls)) if pnls.size > 0 else 0.0

        average_duration_seconds = float(np.mean(durations_seconds)) if durations_seconds.size > 0 else 0.0
        average_duration_bars = float(np.mean(durations_bars)) if durations_bars.size > 0 else 0.0

        equity_curve = np.cumsum(pnls) if pnls.size > 0 else np.asarray([], dtype=float)
        max_drawdown = _max_drawdown(equity_curve)

        signal_true_count = int(np.sum(signal_mask))
        signal_coverage = signal_true_count / max(1, signal_mask.size)

        exposure_seconds = float(np.sum(durations_seconds)) if durations_seconds.size > 0 else 0.0
        total_span_seconds = 0.0
        if signal_mask.size > 1:
            total_span_seconds = float(signal_mask.size - 1)
        exposure_ratio = exposure_seconds / max(1e-12, total_span_seconds) if total_span_seconds > 0 else 0.0

        win_rate = winning_trades / max(1, trade_count)
        expectancy = total_pnl / max(1, trade_count)

        return ReplayMetrics(
            bar_count=int(prices.size),
            signal_true_count=signal_true_count,
            signal_coverage=signal_coverage,
            trade_count=trade_count,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            total_pnl=total_pnl,
            gross_profit=gross_profit,
            gross_loss=gross_loss,
            profit_factor=_profit_factor(pnls),
            average_trade_pnl=average_trade_pnl,
            median_trade_pnl=median_trade_pnl,
            pnl_std=pnl_std,
            average_duration_seconds=average_duration_seconds,
            average_duration_bars=average_duration_bars,
            max_drawdown=max_drawdown,
            exposure_ratio=exposure_ratio,
            win_rate=win_rate,
            expectancy=expectancy,
            direction=direction,
            quantity=quantity,
        )

    def _build_execution_fingerprint(
        self,
        *,
        candidate_fingerprint: str,
        hypothesis: Hypothesis,
        records: tuple[ExecutedTradeRecord, ...],
        metrics: ReplayMetrics,
        direction: str,
        quantity: float,
        price_count: int,
    ) -> Fingerprint:
        components = {
            "candidate_fingerprint": candidate_fingerprint,
            "hypothesis": hypothesis.to_dict() if hasattr(hypothesis, "to_dict") else repr(hypothesis),
            "direction": direction,
            "quantity": quantity,
            "price_count": price_count,
            "trade_count": len(records),
            "metrics": metrics.to_dict(),
            "execution": asdict(self._settings.execution),
        }

        return Fingerprint.from_components(
            components,
            kind="execution",
            version=1,
            parent_digest=candidate_fingerprint or None,
            metadata={
                "module": "execution.replay",
            },
        )

    def _build_diagnostics(
        self,
        *,
        hypothesis: Hypothesis,
        records: tuple[ExecutedTradeRecord, ...],
        signal_mask: np.ndarray,
        prices: np.ndarray,
        metrics: ReplayMetrics,
    ) -> dict[str, Any]:
        reasons: list[str] = []

        if len(hypothesis.conditions) == 0:
            reasons.append("empty_hypothesis")

        if metrics.trade_count == 0:
            reasons.append("no_trades_generated")

        if metrics.signal_true_count == 0:
            reasons.append("no_signal_activation")

        if metrics.signal_coverage < 0.01 and prices.size > 100:
            reasons.append("low_signal_coverage")

        if metrics.pnl_std == 0.0 and metrics.trade_count > 1:
            reasons.append("flat_trade_profile")

        if metrics.max_drawdown == 0.0 and metrics.total_pnl <= 0:
            reasons.append("no_resilience")

        if len({repr(_condition_signature(condition)) for condition in hypothesis.conditions}) != len(hypothesis.conditions):
            reasons.append("duplicate_conditions")

        return {
            "reasons": reasons,
            "trade_count": metrics.trade_count,
            "signal_true_count": metrics.signal_true_count,
            "signal_coverage": metrics.signal_coverage,
            "max_drawdown": metrics.max_drawdown,
            "total_pnl": metrics.total_pnl,
            "win_rate": metrics.win_rate,
            "bars": int(prices.size),
            "records": [record.to_dict() for record in records],
        }

    # ==================================================
    # PYTHON PROTOCOL
    # ==================================================

    def __repr__(self) -> str:
        return (
            "ReplayEngine("
            f"direction='{self._settings.direction}', "
            f"quantity={self._settings.quantity}, "
            f"price_column={self._settings.price_column}"
            ")"
        )