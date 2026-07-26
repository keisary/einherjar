# execution/trade_builder.py
"""
==========================================================
Trade Builder
==========================================================

Construit des objets Trade à partir d'une séquence de
signaux et d'une série de prix.

Le builder reste volontairement pur :
- il applique les coûts d'exécution,
- il transforme des segments de signal en trades,
- il conserve des métadonnées d'exécution.

Il ne simule rien à lui seul.
"""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timedelta
from typing import Any
from typing import Iterable
from typing import Mapping
from typing import Sequence

import numpy as np

from config.execution import ExecutionConfig
from models.trade import Trade
from models.journal import Journal

__all__ = [
    "ExecutedTradeRecord",
    "TradeBuilder",
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


def _ensure_datetime(value: Any, *, fallback_index: int | None = None) -> datetime:
    if isinstance(value, datetime):
        return value

    if hasattr(value, "to_pydatetime") and callable(value.to_pydatetime):
        converted = value.to_pydatetime()
        if isinstance(converted, datetime):
            return converted

    if isinstance(value, np.datetime64):
        # Conversion prudente via timestamp ns.
        ts_ns = value.astype("datetime64[ns]").astype(np.int64)
        return datetime.utcfromtimestamp(ts_ns / 1_000_000_000)

    if isinstance(value, (int, float, np.integer, np.floating)):
        return datetime.utcfromtimestamp(float(value))

    if fallback_index is None:
        fallback_index = 0

    return datetime.utcfromtimestamp(float(fallback_index))


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


def _coerce_timestamp_sequence(
    timestamps: Sequence[Any] | None,
    *,
    count: int,
) -> tuple[datetime, ...]:
    if timestamps is None:
        base = datetime.utcfromtimestamp(0)
        return tuple(base + timedelta(seconds=i) for i in range(count))

    timestamps = tuple(timestamps)
    if len(timestamps) != count:
        raise ValueError("timestamps length must match signal length.")

    return tuple(_ensure_datetime(item, fallback_index=index) for index, item in enumerate(timestamps))


def _coerce_price_sequence(prices: Any, *, count: int) -> np.ndarray:
    array = _as_float_array(prices)
    if array.size != count:
        raise ValueError("prices length must match signal length.")
    return array


# ==========================================================
# RESULTS
# ==========================================================

@dataclass(frozen=True, slots=True)
class ExecutedTradeRecord:
    """
    Trade exécuté avec métadonnées d'exécution.

    Le Trade du modèle métier reste la source principale de
    données, mais ce record conserve les détails utiles pour
    les diagnostics et l'analyse MAE/MFE.
    """

    trade: Trade

    direction: str

    entry_index: int
    exit_index: int

    entry_time: datetime
    exit_time: datetime

    entry_raw_price: float
    exit_raw_price: float

    entry_exec_price: float
    exit_exec_price: float

    gross_pnl: float
    fees_paid: float
    slippage_paid: float
    spread_paid: float

    quantity: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "direction", _normalize_direction(self.direction))
        object.__setattr__(self, "entry_index", max(0, _coerce_int(self.entry_index, 0)))
        object.__setattr__(self, "exit_index", max(0, _coerce_int(self.exit_index, 0)))
        object.__setattr__(self, "entry_raw_price", float(self.entry_raw_price))
        object.__setattr__(self, "exit_raw_price", float(self.exit_raw_price))
        object.__setattr__(self, "entry_exec_price", float(self.entry_exec_price))
        object.__setattr__(self, "exit_exec_price", float(self.exit_exec_price))
        object.__setattr__(self, "gross_pnl", float(self.gross_pnl))
        object.__setattr__(self, "fees_paid", max(0.0, float(self.fees_paid)))
        object.__setattr__(self, "slippage_paid", max(0.0, float(self.slippage_paid)))
        object.__setattr__(self, "spread_paid", max(0.0, float(self.spread_paid)))
        object.__setattr__(self, "quantity", float(self.quantity))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def net_pnl(self) -> float:
        return self.trade.pnl

    @property
    def duration(self):
        return self.trade.duration

    def to_dict(self) -> dict[str, Any]:
        return {
            "trade": self.trade.to_dict(),
            "direction": self.direction,
            "entry_index": self.entry_index,
            "exit_index": self.exit_index,
            "entry_time": self.entry_time.isoformat(),
            "exit_time": self.exit_time.isoformat(),
            "entry_raw_price": self.entry_raw_price,
            "exit_raw_price": self.exit_raw_price,
            "entry_exec_price": self.entry_exec_price,
            "exit_exec_price": self.exit_exec_price,
            "gross_pnl": self.gross_pnl,
            "fees_paid": self.fees_paid,
            "slippage_paid": self.slippage_paid,
            "spread_paid": self.spread_paid,
            "quantity": self.quantity,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExecutedTradeRecord":
        trade_data = _to_mapping(data.get("trade", {}))
        trade = Trade.from_dict(trade_data)

        return cls(
            trade=trade,
            direction=data.get("direction", "long"),
            entry_index=_coerce_int(data.get("entry_index"), 0),
            exit_index=_coerce_int(data.get("exit_index"), 0),
            entry_time=_ensure_datetime(data.get("entry_time"), fallback_index=0),
            exit_time=_ensure_datetime(data.get("exit_time"), fallback_index=0),
            entry_raw_price=_coerce_float(data.get("entry_raw_price"), trade.entry_price),
            exit_raw_price=_coerce_float(data.get("exit_raw_price"), trade.exit_price),
            entry_exec_price=_coerce_float(data.get("entry_exec_price"), trade.entry_price),
            exit_exec_price=_coerce_float(data.get("exit_exec_price"), trade.exit_price),
            gross_pnl=_coerce_float(data.get("gross_pnl"), trade.pnl),
            fees_paid=_coerce_float(data.get("fees_paid"), 0.0),
            slippage_paid=_coerce_float(data.get("slippage_paid"), 0.0),
            spread_paid=_coerce_float(data.get("spread_paid"), 0.0),
            quantity=_coerce_float(data.get("quantity"), trade.quantity),
            metadata=_to_mapping(data.get("metadata", {})),
        )

    def __repr__(self) -> str:
        return (
            "ExecutedTradeRecord("
            f"direction='{self.direction}', "
            f"entry={self.entry_index}, "
            f"exit={self.exit_index}, "
            f"pnl={self.trade.pnl:.4f}"
            ")"
        )


# ==========================================================
# BUILDER
# ==========================================================

class TradeBuilder:
    """
    Transforme une suite de signaux en liste de trades.

    Le builder reste indépendant de la logique de génération
    des signaux. Il reçoit simplement :
    - un masque booléen,
    - une série de prix,
    - une série de timestamps.

    Il applique ensuite les coûts d'exécution configurés.
    """

    def __init__(
        self,
        config: ExecutionConfig | None = None,
        *,
        quantity: float = 1.0,
    ) -> None:
        self._config = config or ExecutionConfig()
        self._quantity = max(0.0, _coerce_float(quantity, 1.0))

    @property
    def config(self) -> ExecutionConfig:
        return self._config

    @property
    def quantity(self) -> float:
        return self._quantity

    # ==================================================
    # PRICE ADJUSTMENT
    # ==================================================

    def effective_price(
        self,
        raw_price: float,
        *,
        direction: str = "long",
        side: str = "entry",
    ) -> float:
        """
        Retourne le prix d'exécution après spread/slippage.

        direction:
            long / short

        side:
            entry / exit
        """
        direction = _normalize_direction(direction)
        side = str(side).strip().lower()
        price = float(raw_price)

        slippage = max(0.0, float(self._config.slippage))
        spread = max(0.0, float(self._config.spread))

        if direction == "long":
            if side == "entry":
                return price * (1.0 + slippage) + spread
            if side == "exit":
                return price * (1.0 - slippage) - spread
        else:
            if side == "entry":
                return price * (1.0 - slippage) - spread
            if side == "exit":
                return price * (1.0 + slippage) + spread

        raise ValueError("side must be 'entry' or 'exit'.")

    def _compute_trade_pnl(
        self,
        *,
        direction: str,
        entry_exec_price: float,
        exit_exec_price: float,
        quantity: float,
    ) -> tuple[float, float, float, float]:
        direction = _normalize_direction(direction)

        notional_entry = abs(entry_exec_price) * quantity
        notional_exit = abs(exit_exec_price) * quantity
        fees_paid = (notional_entry + notional_exit) * max(0.0, float(self._config.fees))

        if direction == "long":
            gross_pnl = (exit_exec_price - entry_exec_price) * quantity
        else:
            gross_pnl = (entry_exec_price - exit_exec_price) * quantity

        spread_paid = 2.0 * max(0.0, float(self._config.spread)) * quantity
        slippage_paid = 2.0 * max(0.0, float(self._config.slippage)) * quantity

        net_pnl = gross_pnl - fees_paid
        return net_pnl, gross_pnl, fees_paid, slippage_paid + spread_paid

    # ==================================================
    # BUILDERS
    # ==================================================

    def build_trade(
        self,
        *,
        entry_index: int,
        exit_index: int,
        entry_time: datetime,
        exit_time: datetime,
        entry_price: float,
        exit_price: float,
        quantity: float | None = None,
        direction: str = "long",
        metadata: Mapping[str, Any] | None = None,
    ) -> ExecutedTradeRecord:
        """
        Construit un trade unique à partir d'un segment.
        """
        quantity = self._quantity if quantity is None else max(0.0, _coerce_float(quantity, self._quantity))
        direction = _normalize_direction(direction)

        entry_exec_price = self.effective_price(entry_price, direction=direction, side="entry")
        exit_exec_price = self.effective_price(exit_price, direction=direction, side="exit")

        net_pnl, gross_pnl, fees_paid, overhead_paid = self._compute_trade_pnl(
            direction=direction,
            entry_exec_price=entry_exec_price,
            exit_exec_price=exit_exec_price,
            quantity=quantity,
        )

        trade = Trade(
            entry_time=entry_time,
            exit_time=exit_time,
            entry_price=entry_exec_price,
            exit_price=exit_exec_price,
            quantity=quantity,
            pnl=net_pnl,
        )

        return ExecutedTradeRecord(
            trade=trade,
            direction=direction,
            entry_index=entry_index,
            exit_index=exit_index,
            entry_time=entry_time,
            exit_time=exit_time,
            entry_raw_price=float(entry_price),
            exit_raw_price=float(exit_price),
            entry_exec_price=entry_exec_price,
            exit_exec_price=exit_exec_price,
            gross_pnl=gross_pnl,
            fees_paid=fees_paid,
            slippage_paid=max(0.0, overhead_paid - 2.0 * max(0.0, float(self._config.spread)) * quantity),
            spread_paid=2.0 * max(0.0, float(self._config.spread)) * quantity,
            quantity=quantity,
            metadata=dict(metadata or {}),
        )

    def build_from_signal_mask(
        self,
        signal_mask: Any,
        prices: Any,
        timestamps: Sequence[Any] | None,
        *,
        direction: str = "long",
        quantity: float | None = None,
        max_open_positions: int | None = None,
        close_on_end: bool = True,
        metadata: Mapping[str, Any] | None = None,
    ) -> tuple[ExecutedTradeRecord, ...]:
        """
        Convertit un masque booléen en trades.

        Un trade est ouvert sur un front montant du signal
        et fermé sur un front descendant.
        """
        signal = _as_bool_array(signal_mask)
        price_array = _coerce_price_sequence(prices, count=signal.size)
        time_array = _coerce_timestamp_sequence(timestamps, count=signal.size)

        direction = _normalize_direction(direction)
        if direction == "long" and not self._config.allow_long:
            raise ValueError("Long direction is disabled by execution config.")
        if direction == "short" and not self._config.allow_short:
            raise ValueError("Short direction is disabled by execution config.")

        max_open_positions = self._config.max_open_positions if max_open_positions is None else max(1, _coerce_int(max_open_positions, 1))
        quantity = self._quantity if quantity is None else max(0.0, _coerce_float(quantity, self._quantity))

        if signal.size == 0:
            return ()

        records: list[ExecutedTradeRecord] = []
        open_entry_index: int | None = None

        for index, active in enumerate(signal):
            if active and open_entry_index is None:
                open_entry_index = index

            if (not active) and open_entry_index is not None:
                exit_index = max(open_entry_index, index - 1)
                records.append(
                    self.build_trade(
                        entry_index=open_entry_index,
                        exit_index=exit_index,
                        entry_time=time_array[open_entry_index],
                        exit_time=time_array[exit_index],
                        entry_price=price_array[open_entry_index],
                        exit_price=price_array[exit_index],
                        quantity=quantity,
                        direction=direction,
                        metadata={
                            **dict(metadata or {}),
                            "signal_start_index": open_entry_index,
                            "signal_end_index": exit_index,
                            "signal_direction": direction,
                        },
                    )
                )
                open_entry_index = None

        if open_entry_index is not None and close_on_end:
            exit_index = signal.size - 1
            records.append(
                self.build_trade(
                    entry_index=open_entry_index,
                    exit_index=exit_index,
                    entry_time=time_array[open_entry_index],
                    exit_time=time_array[exit_index],
                    entry_price=price_array[open_entry_index],
                    exit_price=price_array[exit_index],
                    quantity=quantity,
                    direction=direction,
                    metadata={
                        **dict(metadata or {}),
                        "signal_start_index": open_entry_index,
                        "signal_end_index": exit_index,
                        "signal_direction": direction,
                    },
                )
            )

        # max_open_positions est conservé pour compatibilité
        # future ; avec un signal booléen simple on garde une
        # seule position active par segment.
        return tuple(records)

    def build_journal(
        self,
        signal_mask: Any,
        prices: Any,
        timestamps: Sequence[Any] | None,
        *,
        direction: str = "long",
        quantity: float | None = None,
        max_open_positions: int | None = None,
        close_on_end: bool = True,
        metadata: Mapping[str, Any] | None = None,
    ) -> Journal:
        """
        Construit directement un Journal à partir du signal.
        """
        records = self.build_from_signal_mask(
            signal_mask=signal_mask,
            prices=prices,
            timestamps=timestamps,
            direction=direction,
            quantity=quantity,
            max_open_positions=max_open_positions,
            close_on_end=close_on_end,
            metadata=metadata,
        )
        return Journal(record.trade for record in records)

    def build_from_segments(
        self,
        segments: Iterable[tuple[int, int]],
        prices: Any,
        timestamps: Sequence[Any],
        *,
        direction: str = "long",
        quantity: float | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> tuple[ExecutedTradeRecord, ...]:
        """
        Construit des trades à partir de segments explicites.
        """
        price_array = _as_float_array(prices)
        if price_array.size == 0:
            return ()

        time_array = _coerce_timestamp_sequence(timestamps, count=price_array.size)
        direction = _normalize_direction(direction)
        quantity = self._quantity if quantity is None else max(0.0, _coerce_float(quantity, self._quantity))

        records: list[ExecutedTradeRecord] = []

        for entry_index, exit_index in segments:
            entry_index = max(0, _coerce_int(entry_index, 0))
            exit_index = max(entry_index, _coerce_int(exit_index, entry_index))
            if exit_index >= price_array.size:
                exit_index = price_array.size - 1

            records.append(
                self.build_trade(
                    entry_index=entry_index,
                    exit_index=exit_index,
                    entry_time=time_array[entry_index],
                    exit_time=time_array[exit_index],
                    entry_price=price_array[entry_index],
                    exit_price=price_array[exit_index],
                    quantity=quantity,
                    direction=direction,
                    metadata=metadata,
                )
            )

        return tuple(records)

    # ==================================================
    # SERIALIZATION
    # ==================================================

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": asdict(self._config),
            "quantity": self._quantity,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TradeBuilder":
        config = data.get("config")
        if isinstance(config, Mapping):
            config = ExecutionConfig(**dict(config))
        else:
            config = ExecutionConfig()

        return cls(
            config=config,
            quantity=_coerce_float(data.get("quantity"), 1.0),
        )

    # ==================================================
    # PYTHON PROTOCOL
    # ==================================================

    def __repr__(self) -> str:
        return (
            "TradeBuilder("
            f"fees={self._config.fees}, "
            f"slippage={self._config.slippage}, "
            f"spread={self._config.spread}, "
            f"quantity={self._quantity}"
            ")"
        )