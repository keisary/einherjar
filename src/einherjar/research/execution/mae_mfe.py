# execution/mae_mfe.py
"""
==========================================================
MAE / MFE Analyzer
==========================================================

Calcule les excursions favorables et défavorables des trades
issus d'un replay.

Le module reste purement analytique :
- il lit les prix,
- il mesure les excursions,
- il produit un résumé exploitable par profiler.py et
  diagnostics.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from math import inf
from typing import Any, Iterable
from typing import Mapping
from typing import Sequence

import numpy as np

from models.trade import Trade
from models.fingerprint import Fingerprint
from .trade_builder import ExecutedTradeRecord
from .replay import ReplayResult

__all__ = [
    "MAEMFERecord",
    "MAEMFESummary",
    "MAEMFEAnalyzer",
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


def _safe_mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else 0.0


def _safe_std(values: np.ndarray) -> float:
    return float(np.std(values)) if values.size else 0.0


def _safe_median(values: np.ndarray) -> float:
    return float(np.median(values)) if values.size else 0.0


def _safe_percentile(values: np.ndarray, percentile: float) -> float:
    """
    Calcule un percentile de manière safe (retourne 0.0 si vide).
    percentile en [0, 100] (style numpy, pas [0, 1]).
    """
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, percentile))


def _safe_max(values: np.ndarray) -> float:
    return float(np.max(values)) if values.size else 0.0


def _safe_min(values: np.ndarray) -> float:
    return float(np.min(values)) if values.size else 0.0


def _normalize_direction(value: Any, default: str = "long") -> str:
    text = str(value or default).strip().lower()
    if text in {"long", "buy", "bull", "bullish"}:
        return "long"
    if text in {"short", "sell", "bear", "bearish"}:
        return "short"
    raise ValueError(f"Unknown direction: {value!r}")


def _extract_price_segment(
    prices: np.ndarray,
    start: int,
    end: int,
) -> np.ndarray:
    start = max(0, _coerce_int(start, 0))
    end = max(start, _coerce_int(end, start))
    if prices.size == 0:
        return np.asarray([], dtype=float)
    end = min(end, prices.size - 1)
    if start >= prices.size or end < start:
        return np.asarray([], dtype=float)
    return prices[start : end + 1].astype(float, copy=False).reshape(-1)


def _directional_delta(segment: np.ndarray, entry_price: float, direction: str) -> np.ndarray:
    if segment.size == 0:
        return np.asarray([], dtype=float)

    if direction == "long":
        return segment - entry_price
    return entry_price - segment


def _ratio(value: float, base: float) -> float:
    if abs(base) <= 1e-12:
        return 0.0
    return value / base


@dataclass(frozen=True, slots=True)
class MAEMFERecord:
    """
    Excursion MAE/MFE d'un trade individuel.
    """

    trade: ExecutedTradeRecord

    mae: float
    mfe: float

    mae_pct: float
    mfe_pct: float

    mfe_to_mae_ratio: float

    mae_index: int
    mfe_index: int

    segment_length: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "mae", max(0.0, float(self.mae)))
        object.__setattr__(self, "mfe", max(0.0, float(self.mfe)))
        object.__setattr__(self, "mae_pct", max(0.0, float(self.mae_pct)))
        object.__setattr__(self, "mfe_pct", max(0.0, float(self.mfe_pct)))
        object.__setattr__(self, "mfe_to_mae_ratio", float(self.mfe_to_mae_ratio))
        object.__setattr__(self, "mae_index", max(0, _coerce_int(self.mae_index, 0)))
        object.__setattr__(self, "mfe_index", max(0, _coerce_int(self.mfe_index, 0)))
        object.__setattr__(self, "segment_length", max(0, _coerce_int(self.segment_length, 0)))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def realized_pnl(self) -> float:
        return self.trade.net_pnl

    @property
    def direction(self) -> str:
        return self.trade.direction

    def to_dict(self) -> dict[str, Any]:
        return {
            "trade": self.trade.to_dict(),
            "mae": self.mae,
            "mfe": self.mfe,
            "mae_pct": self.mae_pct,
            "mfe_pct": self.mfe_pct,
            "mfe_to_mae_ratio": self.mfe_to_mae_ratio,
            "mae_index": self.mae_index,
            "mfe_index": self.mfe_index,
            "segment_length": self.segment_length,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MAEMFERecord":
        return cls(
            trade=ExecutedTradeRecord.from_dict(data["trade"]),
            mae=_coerce_float(data.get("mae"), 0.0),
            mfe=_coerce_float(data.get("mfe"), 0.0),
            mae_pct=_coerce_float(data.get("mae_pct"), 0.0),
            mfe_pct=_coerce_float(data.get("mfe_pct"), 0.0),
            mfe_to_mae_ratio=_coerce_float(data.get("mfe_to_mae_ratio"), 0.0),
            mae_index=_coerce_int(data.get("mae_index"), 0),
            mfe_index=_coerce_int(data.get("mfe_index"), 0),
            segment_length=_coerce_int(data.get("segment_length"), 0),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True, slots=True)
class MAEMFESummary:
    """
    Résumé des excursions de tous les trades.

    Les percentiles p75 et p90 sont calculés pour permettre
    la calibration dynamique des TP/SL (cf. PLAN_COMPLET_V2.md
    section 2.3 : tp_rule avec percentile=75, sl_rule avec
    percentile=90).
    """

    trade_count: int

    avg_mae: float
    avg_mfe: float

    median_mae: float
    median_mfe: float

    max_mae: float
    max_mfe: float

    avg_mae_pct: float
    avg_mfe_pct: float

    median_mae_pct: float
    median_mfe_pct: float

    avg_mfe_to_mae_ratio: float
    favorable_trade_rate: float

    # Percentiles optionnels (p75 / p90) pour calibration TP/SL
    p75_mae: float = 0.0
    p75_mfe: float = 0.0
    p90_mae: float = 0.0
    p90_mfe: float = 0.0
    p75_mae_pct: float = 0.0
    p75_mfe_pct: float = 0.0
    p90_mae_pct: float = 0.0
    p90_mfe_pct: float = 0.0

    records: tuple[MAEMFERecord, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "trade_count", max(0, _coerce_int(self.trade_count, 0)))
        object.__setattr__(self, "avg_mae", max(0.0, float(self.avg_mae)))
        object.__setattr__(self, "avg_mfe", max(0.0, float(self.avg_mfe)))
        object.__setattr__(self, "median_mae", max(0.0, float(self.median_mae)))
        object.__setattr__(self, "median_mfe", max(0.0, float(self.median_mfe)))
        object.__setattr__(self, "p75_mae", max(0.0, float(self.p75_mae)))
        object.__setattr__(self, "p75_mfe", max(0.0, float(self.p75_mfe)))
        object.__setattr__(self, "p90_mae", max(0.0, float(self.p90_mae)))
        object.__setattr__(self, "p90_mfe", max(0.0, float(self.p90_mfe)))
        object.__setattr__(self, "max_mae", max(0.0, float(self.max_mae)))
        object.__setattr__(self, "max_mfe", max(0.0, float(self.max_mfe)))
        object.__setattr__(self, "avg_mae_pct", max(0.0, float(self.avg_mae_pct)))
        object.__setattr__(self, "avg_mfe_pct", max(0.0, float(self.avg_mfe_pct)))
        object.__setattr__(self, "median_mae_pct", max(0.0, float(self.median_mae_pct)))
        object.__setattr__(self, "median_mfe_pct", max(0.0, float(self.median_mfe_pct)))
        object.__setattr__(self, "p75_mae_pct", max(0.0, float(self.p75_mae_pct)))
        object.__setattr__(self, "p75_mfe_pct", max(0.0, float(self.p75_mfe_pct)))
        object.__setattr__(self, "p90_mae_pct", max(0.0, float(self.p90_mae_pct)))
        object.__setattr__(self, "p90_mfe_pct", max(0.0, float(self.p90_mfe_pct)))
        object.__setattr__(self, "avg_mfe_to_mae_ratio", float(self.avg_mfe_to_mae_ratio))
        object.__setattr__(self, "favorable_trade_rate", min(1.0, max(0.0, float(self.favorable_trade_rate))))
        object.__setattr__(self, "records", tuple(self.records))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "trade_count": self.trade_count,
            "avg_mae": self.avg_mae,
            "avg_mfe": self.avg_mfe,
            "median_mae": self.median_mae,
            "median_mfe": self.median_mfe,
            "p75_mae": self.p75_mae,
            "p75_mfe": self.p75_mfe,
            "p90_mae": self.p90_mae,
            "p90_mfe": self.p90_mfe,
            "max_mae": self.max_mae,
            "max_mfe": self.max_mfe,
            "avg_mae_pct": self.avg_mae_pct,
            "avg_mfe_pct": self.avg_mfe_pct,
            "median_mae_pct": self.median_mae_pct,
            "median_mfe_pct": self.median_mfe_pct,
            "p75_mae_pct": self.p75_mae_pct,
            "p75_mfe_pct": self.p75_mfe_pct,
            "p90_mae_pct": self.p90_mae_pct,
            "p90_mfe_pct": self.p90_mfe_pct,
            "avg_mfe_to_mae_ratio": self.avg_mfe_to_mae_ratio,
            "favorable_trade_rate": self.favorable_trade_rate,
            "records": [record.to_dict() for record in self.records],
            "metadata": dict(self.metadata),
        }


class MAEMFEAnalyzer:
    """
    Calcule les excursions MAE/MFE à partir d'un replay.
    """

    def __init__(self) -> None:
        self._last_summary: MAEMFESummary | None = None

    @property
    def last_summary(self) -> MAEMFESummary | None:
        return self._last_summary

    def assess_trade(
        self,
        record: ExecutedTradeRecord,
        prices: Any,
    ) -> MAEMFERecord:
        price_array = _as_float_array(prices)
        if price_array.size == 0:
            raise ValueError("prices cannot be empty.")

        start = max(0, _coerce_int(record.entry_index, 0))
        end = max(start, _coerce_int(record.exit_index, start))
        segment = _extract_price_segment(price_array, start, end)
        if segment.size == 0:
            raise ValueError("Invalid trade segment.")

        direction = _normalize_direction(record.direction)
        entry_price = float(record.entry_raw_price)

        delta = _directional_delta(segment, entry_price, direction)
        if delta.size == 0:
            delta = np.asarray([0.0], dtype=float)

        mfe_index_local = int(np.argmax(delta))
        mae_index_local = int(np.argmin(delta))

        mfe = max(0.0, float(delta[mfe_index_local]))
        mae = max(0.0, float(-delta[mae_index_local]))

        entry_abs = abs(entry_price) if abs(entry_price) > 1e-12 else 1.0
        mfe_pct = mfe / entry_abs
        mae_pct = mae / entry_abs

        ratio = float("inf") if mae <= 1e-12 else mfe / mae

        return MAEMFERecord(
            trade=record,
            mae=mae,
            mfe=mfe,
            mae_pct=mae_pct,
            mfe_pct=mfe_pct,
            mfe_to_mae_ratio=ratio,
            mae_index=start + mae_index_local,
            mfe_index=start + mfe_index_local,
            segment_length=int(segment.size),
            metadata={
                "entry_price": entry_price,
                "exit_price": float(record.exit_raw_price),
            },
        )

    def assess_records(
        self,
        records: Iterable[ExecutedTradeRecord],
        prices: Any,
    ) -> MAEMFESummary:
        price_array = _as_float_array(prices)
        records = tuple(records)

        if not records:
            summary = MAEMFESummary(
                trade_count=0,
                avg_mae=0.0,
                avg_mfe=0.0,
                median_mae=0.0,
                median_mfe=0.0,
                max_mae=0.0,
                max_mfe=0.0,
                avg_mae_pct=0.0,
                avg_mfe_pct=0.0,
                median_mae_pct=0.0,
                median_mfe_pct=0.0,
                avg_mfe_to_mae_ratio=0.0,
                favorable_trade_rate=0.0,
                records=(),
                metadata={"reason": "no_records"},
            )
            self._last_summary = summary
            return summary

        recs: list[MAEMFERecord] = []
        maes: list[float] = []
        mfes: list[float] = []
        maes_pct: list[float] = []
        mfes_pct: list[float] = []
        ratios: list[float] = []
        favorable = 0

        for record in records:
            item = self.assess_trade(record, price_array)
            recs.append(item)
            maes.append(item.mae)
            mfes.append(item.mfe)
            maes_pct.append(item.mae_pct)
            mfes_pct.append(item.mfe_pct)
            if np.isfinite(item.mfe_to_mae_ratio):
                ratios.append(item.mfe_to_mae_ratio)
            if item.mfe > item.mae:
                favorable += 1

        maes_arr = np.asarray(maes, dtype=float)
        mfes_arr = np.asarray(mfes, dtype=float)
        maes_pct_arr = np.asarray(maes_pct, dtype=float)
        mfes_pct_arr = np.asarray(mfes_pct, dtype=float)

        summary = MAEMFESummary(
            trade_count=len(recs),
            avg_mae=_safe_mean(maes_arr),
            avg_mfe=_safe_mean(mfes_arr),
            median_mae=_safe_median(maes_arr),
            median_mfe=_safe_median(mfes_arr),
            p75_mae=_safe_percentile(maes_arr, 75.0),
            p75_mfe=_safe_percentile(mfes_arr, 75.0),
            p90_mae=_safe_percentile(maes_arr, 90.0),
            p90_mfe=_safe_percentile(mfes_arr, 90.0),
            max_mae=_safe_max(maes_arr),
            max_mfe=_safe_max(mfes_arr),
            avg_mae_pct=_safe_mean(maes_pct_arr),
            avg_mfe_pct=_safe_mean(mfes_pct_arr),
            median_mae_pct=_safe_median(maes_pct_arr),
            median_mfe_pct=_safe_median(mfes_pct_arr),
            p75_mae_pct=_safe_percentile(maes_pct_arr, 75.0),
            p75_mfe_pct=_safe_percentile(mfes_pct_arr, 75.0),
            p90_mae_pct=_safe_percentile(maes_pct_arr, 90.0),
            p90_mfe_pct=_safe_percentile(mfes_pct_arr, 90.0),
            avg_mfe_to_mae_ratio=_safe_mean(np.asarray(ratios, dtype=float)) if ratios else 0.0,
            favorable_trade_rate=favorable / max(1, len(recs)),
            records=tuple(recs),
            metadata={},
        )
        self._last_summary = summary
        return summary

    def assess_replay(self, replay: ReplayResult) -> MAEMFESummary:
        return self.assess_records(replay.records, replay.prices)

    def summarize(self, replay: ReplayResult) -> MAEMFESummary:
        return self.assess_replay(replay)

    def __repr__(self) -> str:
        return "MAEMFEAnalyzer()"