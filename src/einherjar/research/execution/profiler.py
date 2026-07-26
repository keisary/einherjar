# execution/profiler.py
"""
==========================================================
Execution Profiler
==========================================================

Résume le comportement d'un replay de stratégie.

Le profiler n'exécute rien :
- il lit un ReplayResult,
- il agrège les mesures de trades,
- il produit un profil exploitable pour diagnostics,
  connaissance ou export.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Mapping
from typing import Sequence

import numpy as np

from models.profile import Profile
from .mae_mfe import MAEMFESummary
from .replay import ReplayResult

__all__ = [
    "ExecutionProfileSettings",
    "ExecutionProfile",
    "ExecutionProfiler",
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


def _safe_mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else 0.0


def _safe_std(values: np.ndarray) -> float:
    return float(np.std(values)) if values.size else 0.0


def _safe_median(values: np.ndarray) -> float:
    return float(np.median(values)) if values.size else 0.0


def _safe_max(values: np.ndarray) -> float:
    return float(np.max(values)) if values.size else 0.0


def _safe_min(values: np.ndarray) -> float:
    return float(np.min(values)) if values.size else 0.0


def _longest_streak(values: np.ndarray, positive: bool) -> int:
    best = current = 0
    for value in values:
        hit = value > 0 if positive else value < 0
        if hit:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def _drawdown_from_series(series: np.ndarray) -> float:
    if series.size == 0:
        return 0.0
    peak = np.maximum.accumulate(series)
    dd = peak - series
    return float(np.max(dd)) if dd.size else 0.0


@dataclass(frozen=True, slots=True)
class ExecutionProfileSettings:
    """
    Paramètres du profiler.
    """

    min_trades: int = 1
    min_win_rate: float = 0.0
    min_profit_factor: float = 0.0
    min_expectancy: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "min_trades", max(1, _coerce_int(self.min_trades, 1)))
        object.__setattr__(self, "min_win_rate", min(1.0, max(0.0, _coerce_float(self.min_win_rate, 0.0))))
        object.__setattr__(self, "min_profit_factor", max(0.0, _coerce_float(self.min_profit_factor, 0.0)))
        object.__setattr__(self, "min_expectancy", _coerce_float(self.min_expectancy, 0.0))


@dataclass(frozen=True, slots=True)
class ExecutionProfile:
    """
    Profil d'exécution d'une stratégie.
    """

    name: str
    description: str

    trade_count: int
    win_rate: float
    total_pnl: float

    gross_profit: float
    gross_loss: float
    profit_factor: float

    expectancy: float
    average_trade_pnl: float
    median_trade_pnl: float
    pnl_std: float

    average_duration_seconds: float
    average_duration_bars: float
    max_drawdown: float
    recovery_factor: float
    exposure_ratio: float

    longest_win_streak: int
    longest_loss_streak: int

    average_mae: float = 0.0
    average_mfe: float = 0.0
    average_mfe_to_mae_ratio: float = 0.0
    favorable_trade_rate: float = 0.0

    label: Profile | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", str(self.name).strip() or "execution_profile")
        object.__setattr__(self, "description", str(self.description).strip())
        object.__setattr__(self, "trade_count", max(0, _coerce_int(self.trade_count, 0)))
        object.__setattr__(self, "win_rate", min(1.0, max(0.0, float(self.win_rate))))
        object.__setattr__(self, "total_pnl", float(self.total_pnl))
        object.__setattr__(self, "gross_profit", float(self.gross_profit))
        object.__setattr__(self, "gross_loss", float(self.gross_loss))
        object.__setattr__(self, "profit_factor", float(self.profit_factor))
        object.__setattr__(self, "expectancy", float(self.expectancy))
        object.__setattr__(self, "average_trade_pnl", float(self.average_trade_pnl))
        object.__setattr__(self, "median_trade_pnl", float(self.median_trade_pnl))
        object.__setattr__(self, "pnl_std", max(0.0, float(self.pnl_std)))
        object.__setattr__(self, "average_duration_seconds", max(0.0, float(self.average_duration_seconds)))
        object.__setattr__(self, "average_duration_bars", max(0.0, float(self.average_duration_bars)))
        object.__setattr__(self, "max_drawdown", max(0.0, float(self.max_drawdown)))
        object.__setattr__(self, "recovery_factor", float(self.recovery_factor))
        object.__setattr__(self, "exposure_ratio", min(1.0, max(0.0, float(self.exposure_ratio))))
        object.__setattr__(self, "longest_win_streak", max(0, _coerce_int(self.longest_win_streak, 0)))
        object.__setattr__(self, "longest_loss_streak", max(0, _coerce_int(self.longest_loss_streak, 0)))
        object.__setattr__(self, "average_mae", max(0.0, float(self.average_mae)))
        object.__setattr__(self, "average_mfe", max(0.0, float(self.average_mfe)))
        object.__setattr__(self, "average_mfe_to_mae_ratio", float(self.average_mfe_to_mae_ratio))
        object.__setattr__(self, "favorable_trade_rate", min(1.0, max(0.0, float(self.favorable_trade_rate))))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def is_profitable(self) -> bool:
        return self.total_pnl > 0

    @property
    def sharpness(self) -> float:
        if self.pnl_std <= 1e-12:
            return 0.0
        return self.expectancy / self.pnl_std

    def to_profile_model(self) -> Profile:
        return self.label or Profile(
            name=self.name,
            description=self.description,
            metadata={
                "trade_count": self.trade_count,
                "win_rate": self.win_rate,
                "total_pnl": self.total_pnl,
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "trade_count": self.trade_count,
            "win_rate": self.win_rate,
            "total_pnl": self.total_pnl,
            "gross_profit": self.gross_profit,
            "gross_loss": self.gross_loss,
            "profit_factor": self.profit_factor,
            "expectancy": self.expectancy,
            "average_trade_pnl": self.average_trade_pnl,
            "median_trade_pnl": self.median_trade_pnl,
            "pnl_std": self.pnl_std,
            "average_duration_seconds": self.average_duration_seconds,
            "average_duration_bars": self.average_duration_bars,
            "max_drawdown": self.max_drawdown,
            "recovery_factor": self.recovery_factor,
            "exposure_ratio": self.exposure_ratio,
            "longest_win_streak": self.longest_win_streak,
            "longest_loss_streak": self.longest_loss_streak,
            "average_mae": self.average_mae,
            "average_mfe": self.average_mfe,
            "average_mfe_to_mae_ratio": self.average_mfe_to_mae_ratio,
            "favorable_trade_rate": self.favorable_trade_rate,
            "label": None if self.label is None else self.label.to_dict(),
            "metadata": dict(self.metadata),
        }


class ExecutionProfiler:
    """
    Produit un profil d'exécution à partir d'un replay.
    """

    def __init__(
        self,
        settings: ExecutionProfileSettings | None = None,
    ) -> None:
        self._settings = settings or ExecutionProfileSettings()

    @property
    def settings(self) -> ExecutionProfileSettings:
        return self._settings

    def profile(
        self,
        replay: ReplayResult,
        *,
        mae_mfe: MAEMFESummary | None = None,
        name: str = "execution_profile",
        description: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ExecutionProfile:
        metrics = replay.metrics
        records = replay.records
        pnls = np.asarray([record.trade.pnl for record in records], dtype=float)
        durations_seconds = np.asarray([record.duration.total_seconds() for record in records], dtype=float)
        durations_bars = np.asarray([record.exit_index - record.entry_index + 1 for record in records], dtype=float)
        equity = np.cumsum(pnls) if pnls.size else np.asarray([], dtype=float)

        win_rate = metrics.win_rate
        total_pnl = metrics.total_pnl
        gross_profit = metrics.gross_profit
        gross_loss = metrics.gross_loss
        profit_factor = metrics.profit_factor
        expectancy = metrics.expectancy

        avg_trade_pnl = _safe_mean(pnls)
        median_trade_pnl = _safe_median(pnls)
        pnl_std = _safe_std(pnls)

        avg_duration_seconds = _safe_mean(durations_seconds)
        avg_duration_bars = _safe_mean(durations_bars)
        max_drawdown = _drawdown_from_series(equity)

        recovery_factor = total_pnl / max_drawdown if max_drawdown > 1e-12 else float("inf") if total_pnl > 0 else 0.0
        exposure_ratio = metrics.exposure_ratio

        longest_win_streak = _longest_streak(pnls, positive=True)
        longest_loss_streak = _longest_streak(pnls, positive=False)

        average_mae = 0.0
        average_mfe = 0.0
        average_mfe_to_mae_ratio = 0.0
        favorable_trade_rate = 0.0
        if mae_mfe is not None:
            average_mae = mae_mfe.avg_mae
            average_mfe = mae_mfe.avg_mfe
            average_mfe_to_mae_ratio = mae_mfe.avg_mfe_to_mae_ratio
            favorable_trade_rate = mae_mfe.favorable_trade_rate

        desc = description or (
            f"{metrics.direction} | trades={metrics.trade_count} | win_rate={win_rate:.2%} | pf={profit_factor:.2f}"
        )

        label = Profile(
            name=name,
            description=desc,
            metadata={
                "trade_count": metrics.trade_count,
                "win_rate": win_rate,
                "profit_factor": profit_factor,
                "expectancy": expectancy,
            },
        )

        return ExecutionProfile(
            name=name,
            description=desc,
            trade_count=metrics.trade_count,
            win_rate=win_rate,
            total_pnl=total_pnl,
            gross_profit=gross_profit,
            gross_loss=gross_loss,
            profit_factor=profit_factor,
            expectancy=expectancy,
            average_trade_pnl=avg_trade_pnl,
            median_trade_pnl=median_trade_pnl,
            pnl_std=pnl_std,
            average_duration_seconds=avg_duration_seconds,
            average_duration_bars=avg_duration_bars,
            max_drawdown=max_drawdown,
            recovery_factor=recovery_factor,
            exposure_ratio=exposure_ratio,
            longest_win_streak=longest_win_streak,
            longest_loss_streak=longest_loss_streak,
            average_mae=average_mae,
            average_mfe=average_mfe,
            average_mfe_to_mae_ratio=average_mfe_to_mae_ratio,
            favorable_trade_rate=favorable_trade_rate,
            label=label,
            metadata=dict(metadata or {}),
        )

    def summarize(
        self,
        replay: ReplayResult,
        *,
        mae_mfe: MAEMFESummary | None = None,
        name: str = "execution_profile",
        description: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ExecutionProfile:
        return self.profile(
            replay,
            mae_mfe=mae_mfe,
            name=name,
            description=description,
            metadata=metadata,
        )

    def __repr__(self) -> str:
        return "ExecutionProfiler()"