"""Backtest portefeuille sans look-ahead.

Ce simulateur est volontairement separe de l'ancien calibrator. Il traite
des bougies reelles, ouvre a la prochaine ouverture et suit chaque position.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np


@dataclass(frozen=True)
class BacktestSignal:
    """Intention calculee a la cloture d'une bougie."""

    asset: str
    direction: str
    entry_index: int
    tp_rule: dict[str, float | str]
    sl_rule: dict[str, float | str]
    einher_name: str
    confidence: float = 1.0


@dataclass
class OpenPosition:
    """Position independante ouverte au prix d'ouverture suivant."""

    position_id: str
    asset: str
    direction: str
    quantity: float
    entry_price: float
    tp_price: float
    sl_price: float
    opened_index: int
    einher_name: str


@dataclass(frozen=True)
class ClosedTrade:
    """Position cloturee avec son resultat net."""

    position_id: str
    asset: str
    direction: str
    entry_price: float
    exit_price: float
    quantity: float
    opened_index: int
    closed_index: int
    reason: str
    net_return: float


@dataclass
class PortfolioResult:
    """Resultat complet et mesurable du portefeuille."""

    equity_curve: list[float]
    closed_trades: list[ClosedTrade]
    initial_capital: float

    @property
    def net_return(self) -> float:
        return self.equity_curve[-1] / self.initial_capital - 1.0

    @property
    def max_drawdown(self) -> float:
        equity = np.asarray(self.equity_curve, dtype=float)
        peak = np.maximum.accumulate(equity)
        return float(np.max((peak - equity) / np.maximum(peak, 1e-12)))

    @property
    def win_rate(self) -> float:
        if not self.closed_trades:
            return 0.0
        return sum(trade.net_return > 0 for trade in self.closed_trades) / len(self.closed_trades)

    @property
    def profit_factor(self) -> float:
        profits = sum(max(trade.net_return, 0.0) for trade in self.closed_trades)
        losses = -sum(min(trade.net_return, 0.0) for trade in self.closed_trades)
        return profits / losses if losses else float("inf") if profits else 0.0

    @property
    def sharpe_per_bar(self) -> float:
        returns = np.diff(np.asarray(self.equity_curve, dtype=float)) / np.asarray(self.equity_curve[:-1], dtype=float)
        if len(returns) < 2 or float(np.std(returns, ddof=1)) == 0:
            return 0.0
        return float(np.mean(returns) / np.std(returns, ddof=1))


class PortfolioSimulator:
    """Simulateur portefeuille pour une grille de bougies synchronisees.

    `bars` contient des lignes dictionnaire avec asset, open, high, low et
    close. Chaque ligne represente une bougie cloturee. `signal_factory`
    decide a la cloture et fournit des intentions pour l'ouverture suivante.
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        fee_rate: float = 0.001,
        slippage_rate: float = 0.0005,
        max_positions: int = 5,
    ) -> None:
        if initial_capital <= 0:
            raise ValueError("initial_capital doit etre positif")
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.fee_rate = fee_rate
        self.slippage_rate = slippage_rate
        self.max_positions = max_positions
        self.positions: list[OpenPosition] = []
        self.closed_trades: list[ClosedTrade] = []
        self.equity_curve = [initial_capital]
        self._pending: dict[int, list[BacktestSignal]] = {}
        self._next_position_id = 1

    def run(
        self,
        bars: Iterable[dict[str, float | int | str]],
        signal_factory: Callable[[int, dict[str, float | int | str]], Iterable[BacktestSignal]],
    ) -> PortfolioResult:
        """Execute le backtest sans trade au close de detection."""
        rows = list(bars)
        for index, bar in enumerate(rows):
            self._close_hit_positions(index, bar)
            self._open_pending(index, bar)
            if index + 1 < len(rows):
                for signal in signal_factory(index, bar):
                    self._validate_signal(signal, index, bar)
                    self._pending.setdefault(index + 1, []).append(signal)
            self.equity_curve.append(self._mark_to_market(bar))
        for position in list(self.positions):
            last_index = len(rows) - 1
            last_close = float(rows[-1]["close"])
            self._close_position(position, last_index, last_close, "END")
        if rows:
            self.equity_curve[-1] = self.capital
        return PortfolioResult(self.equity_curve, self.closed_trades, self.initial_capital)

    def _validate_signal(self, signal: BacktestSignal, index: int, bar: dict[str, float | int | str]) -> None:
        if signal.direction not in {"long", "short"}:
            raise ValueError("direction doit etre long ou short")
        if signal.entry_index != index:
            raise ValueError("entry_index doit correspondre a la bougie de detection")
        if signal.asset != bar["asset"]:
            raise ValueError("un signal doit appartenir a la bougie evaluee")

    def _open_pending(self, index: int, bar: dict[str, float | int | str]) -> None:
        for signal in self._pending.pop(index, []):
            if len(self.positions) >= self.max_positions or signal.asset != bar["asset"]:
                continue
            entry = float(bar["open"])
            entry *= 1 + self.slippage_rate if signal.direction == "long" else 1 - self.slippage_rate
            tp, sl = self._exit_prices(entry, signal.direction, signal.tp_rule, signal.sl_rule, bar)
            if (signal.direction == "long" and not sl < entry < tp) or (signal.direction == "short" and not tp < entry < sl):
                continue
            risk_per_unit = abs(entry - sl)
            quantity = min(self.capital * 0.01 / risk_per_unit, self.capital / entry)
            if quantity <= 0:
                continue
            position = OpenPosition(
                position_id=f"POS-{self._next_position_id}",
                asset=signal.asset,
                direction=signal.direction,
                quantity=quantity,
                entry_price=entry,
                tp_price=tp,
                sl_price=sl,
                opened_index=index,
                einher_name=signal.einher_name,
            )
            self._next_position_id += 1
            self.capital -= entry * quantity * (1 + self.fee_rate)
            self.positions.append(position)

    def _exit_prices(
        self,
        entry: float,
        direction: str,
        tp_rule: dict[str, float | str],
        sl_rule: dict[str, float | str],
        bar: dict[str, float | int | str],
    ) -> tuple[float, float]:
        high, low = float(bar["high"]), float(bar["low"])
        height = max(high - low, entry * 0.0001)
        tp_distance = self._rule_distance(entry, tp_rule, height, default=0.01)
        sl_distance = self._rule_distance(entry, sl_rule, height, default=0.005)
        if direction == "long":
            return entry + tp_distance, entry - sl_distance
        return entry - tp_distance, entry + sl_distance

    @staticmethod
    def _rule_distance(entry: float, rule: dict[str, float | str], height: float, default: float) -> float:
        kind = str(rule.get("type", ""))
        value = float(rule.get("value", default))
        if kind in {"mfe_calibrated", "mae_calibrated", "return_pct"}:
            return entry * max(value, 0.0)
        if kind == "pattern_height":
            return max(value or height, 0.0)
        if kind == "fibonacci_ratio":
            return height * max(value, 0.0)
        if kind == "atr_multiple":
            atr = float(rule.get("atr", height))
            return atr * max(value, 0.0)
        return entry * default

    def _close_hit_positions(self, index: int, bar: dict[str, float | int | str]) -> None:
        for position in list(self.positions):
            if position.asset != bar["asset"] or position.opened_index >= index:
                continue
            high, low = float(bar["high"]), float(bar["low"])
            if position.direction == "long":
                hit_sl, hit_tp = low <= position.sl_price, high >= position.tp_price
            else:
                hit_sl, hit_tp = high >= position.sl_price, low <= position.tp_price
            # Une bougie incapable d'ordonner ses extremes applique le stop.
            if hit_sl:
                self._close_position(position, index, position.sl_price, "SL")
            elif hit_tp:
                self._close_position(position, index, position.tp_price, "TP")

    def _close_position(self, position: OpenPosition, index: int, raw_price: float, reason: str) -> None:
        exit_price = raw_price * (1 - self.slippage_rate if position.direction == "long" else 1 + self.slippage_rate)
        sign = 1.0 if position.direction == "long" else -1.0
        gross = sign * (exit_price - position.entry_price) / position.entry_price
        net = gross - 2 * self.fee_rate
        if position.direction == "long":
            self.capital += exit_price * position.quantity * (1 - self.fee_rate)
        else:
            pnl = position.quantity * (position.entry_price - exit_price)
            self.capital += position.entry_price * position.quantity * (1 - self.fee_rate) + pnl
        self.closed_trades.append(ClosedTrade(
            position_id=position.position_id,
            asset=position.asset,
            direction=position.direction,
            entry_price=position.entry_price,
            exit_price=exit_price,
            quantity=position.quantity,
            opened_index=position.opened_index,
            closed_index=index,
            reason=reason,
            net_return=net,
        ))
        self.positions.remove(position)

    def _mark_to_market(self, bar: dict[str, float | int | str]) -> float:
        equity = self.capital
        close = float(bar["close"])
        for position in self.positions:
            if position.asset != bar["asset"]:
                continue
            if position.direction == "long":
                equity += position.quantity * close
            else:
                equity += position.quantity * position.entry_price + position.quantity * (position.entry_price - close)
        return equity


def walk_forward(
    bars: list[dict[str, float | int | str]],
    signal_factory: Callable[[int, dict[str, float | int | str]], Iterable[BacktestSignal]],
    split: float = 0.7,
) -> tuple[PortfolioResult, PortfolioResult]:
    """Execute train et validation sans reutiliser les donnees terminales."""
    if not 0 < split < 1:
        raise ValueError("split doit etre entre 0 et 1")
    boundary = max(2, int(len(bars) * split))
    train = PortfolioSimulator().run(bars[:boundary], signal_factory)
    validation = PortfolioSimulator().run(bars[boundary:], signal_factory)
    return train, validation
