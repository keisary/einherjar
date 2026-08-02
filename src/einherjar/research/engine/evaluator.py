"""Moteur d'évaluation hors-échantillon — PRIORITÉ 0.

C'est le CONTRAT d'évaluation du système de découverte. Tous les générateurs
(baselines, GE, GP, beam, etc.) DOIVENT consommer ce moteur pour être
comparables. Sans moteur d'évaluation audité, aucune comparaison de
générateurs n'a de sens.

Pipeline :
  1. `train_calibrate(hypothesis, train_ohlcv, train_features)`
     → CalibratedParams (N, SL, TP figés depuis le train uniquement)
  2. `test_on(hypothesis, ohlcv, features, calibrated, split_name)`
     → MesuresBrutes (calculées sur le split demandé, sans recalibrage)
  3. `evaluate(hypothesis, train, val, holdout)`
     → (MesuresBrutes_train, MesuresBrutes_val, CalibratedParams, MesuresBrutes_holdout)

Invariants durs :
  - Aucun paramètre (N, SL, TP) n'est calculé sur val/holdout (I-5).
  - Entrée à l'OPEN de t+1, simulation intrabar TP/SL sur high/low.
  - Convention : SL avant TP sur la même bougie (conservateur).
  - Coûts (spread, commission, slippage) appliqués en round-trip.
  - Block bootstrap CI sur Sharpe et ret total (gère l'autocorrélation).
  - Le holdout n'est consulté qu'une seule fois par Einher final.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
import polars as pl

from einherjar.research.config.loader import EinherjarConfig
from einherjar.research.data.features import FeaturesFrame
from einherjar.research.data.ohlcv import OhlcvFrame
from einherjar.research.engine.bootstrap import (
    bootstrap_ret_total,
    bootstrap_sharpe,
)
from einherjar.research.engine.simulator import simulate
from einherjar.research.utils.stats import atr_wilder, percentile, periods_per_year_for_timeframe
from einherjar.research.utils.types import (
    AmplitudeUnit,
    CompareOp,
    Condition,
    ConditionNode,
    Direction,
    ExitReason,
    Hypothesis,
    LogicalOp,
    MesuresBrutes,
    TradeMesure,
)

logger = logging.getLogger(__name__)

# Splits valides (utilisés pour tracer l'accès au holdout)
HOLDOUT_SPLIT_NAME: str = "holdout"

# Constantes de coût (par défaut, overridables par config)
DEFAULT_COSTS: dict[str, float] = {
    "spread_pct": 0.0002,
    "commission_pct": 0.0001,
    "slippage_pct": 0.0001,
}


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #


class EvaluationError(Exception):
    """Erreur générique du moteur d'évaluation."""


class CalibrationError(EvaluationError):
    """Erreur de calibration (train insuffisant, données manquantes)."""


class HoldoutAccessError(EvaluationError):
    """Accès invalide au holdout (deuxième tentative, etc.)."""


# --------------------------------------------------------------------------- #
# Calibration figée depuis le train
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CalibratedParams:
    """Paramètres de calibration calculés sur le train, gelés pour val/holdout.

    Les SL et TP sont stockés comme DISTANCES RELATIVES (en multiples d'ATR
    et en %), recalculés à chaque entrée à partir du prix d'entrée et de
    l'ATR local. Aucun prix absolu n'est stocké (anti-leak d'entrée
    arbitraire).

    Une fois créés, ces paramètres ne sont JAMAIS recalculés. Toute
    modification passe par la création d'une nouvelle instance.

    Attributs:
        n_window: Horizon d'observation (en bougies).
        sl_n_atr: Distance SL en multiple d'ATR (sl = entry - sl_n_atr * atr).
        tp_n_atr: Distance TP en multiple d'ATR (tp = entry + tp_n_atr * atr).
        sl_distance: Distance SL en % (sl = entry * (1 - sl_distance)).
        tp_distance: Distance TP en % (tp = entry * (1 + tp_distance)).
        atr_p50: ATR(14) médian sur le train (référence, traçabilité).
        n_observations: Nombre de bougies du train (traçabilité).
        mfe_p50: MFE médian sur le train (pour traçabilité).
        mae_p75: MAE p75 sur le train (pour traçabilité).
    """

    n_window: int
    sl_n_atr: float
    tp_n_atr: float
    sl_distance: float
    tp_distance: float
    atr_p50: float
    n_observations: int
    mfe_p50: float = 0.0
    mae_p75: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Sérialisation pour persistance (Archive, fingerprint, etc.)."""
        return {
            "n_window": self.n_window,
            "sl_n_atr": self.sl_n_atr,
            "tp_n_atr": self.tp_n_atr,
            "sl_distance": self.sl_distance,
            "tp_distance": self.tp_distance,
            "atr_p50": self.atr_p50,
            "n_observations": self.n_observations,
            "mfe_p50": self.mfe_p50,
            "mae_p75": self.mae_p75,
        }

    def compute_sl_tp_at_entry(
        self,
        entry_price: float,
        atr_at_entry: float,
        direction: Direction,
    ) -> tuple[float, float]:
        """Calcule les niveaux SL/TP en prix absolus à partir du prix d'entrée et de l'ATR local.

        Les distances sont en multiples d'ATR (sl_n_atr, tp_n_atr), recalculées
        à chaque trade. Pas de prix absolu figé.

        Args:
            entry_price: Prix d'entrée (OPEN de t+1).
            atr_at_entry: ATR local calculé sur la fenêtre se terminant à t+1.
            direction: Direction du trade.

        Returns:
            (sl_price, tp_price) en prix absolus.
        """
        if atr_at_entry <= 0:
            atr_at_entry = self.atr_p50
        sl_dist = self.sl_n_atr * atr_at_entry
        tp_dist = self.tp_n_atr * atr_at_entry
        if direction == Direction.LONG:
            return (entry_price - sl_dist, entry_price + tp_dist)
        return (entry_price + sl_dist, entry_price - tp_dist)


# --------------------------------------------------------------------------- #
# Coûts — value object
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TradingCosts:
    """Coûts de transaction par trade (round-trip)."""

    spread_pct: float
    commission_pct: float
    slippage_pct: float

    @property
    def total_round_trip_pct(self) -> float:
        """Coût total en %, à appliquer une fois par trade (entrée + sortie)."""
        # L'entrée paye spread/2 + commission + slippage, la sortie pareil.
        # On simplifie : on débourse le coût total une fois par trade.
        return (self.spread_pct + self.commission_pct + self.slippage_pct) * 2.0

    @classmethod
    def from_config(cls, config: EinherjarConfig, asset: str | None = None) -> TradingCosts:
        """Construit depuis la config (avec override par asset si présent)."""
        costs_cfg = config.costs
        default = dict(costs_cfg.get("default", DEFAULT_COSTS))
        overrides = dict(costs_cfg.get("overrides", {}))
        if asset and asset in overrides:
            default.update(overrides[asset])
        return cls(
            spread_pct=float(default.get("spread_pct", DEFAULT_COSTS["spread_pct"])),
            commission_pct=float(default.get("commission_pct", DEFAULT_COSTS["commission_pct"])),
            slippage_pct=float(default.get("slippage_pct", DEFAULT_COSTS["slippage_pct"])),
        )

    def to_dict(self) -> dict[str, float]:
        """Sérialisation pour persistance et audit (Archive)."""
        return {
            "spread_pct": self.spread_pct,
            "commission_pct": self.commission_pct,
            "slippage_pct": self.slippage_pct,
            "total_round_trip_pct": self.total_round_trip_pct,
        }


# --------------------------------------------------------------------------- #
# Helpers privés — chacun a UNE responsabilité
# --------------------------------------------------------------------------- #


class _ATREstimator:
    """Calcule ATR(14) sur une série OHLCV, expose la médiane (p50)."""

    def __init__(self, period: int = 14, percentile_p: float = 50.0) -> None:
        self.period = period
        self.percentile_p = percentile_p

    def p50(self, ohlcv: OhlcvFrame) -> float:
        """Calcule l'ATR(period) et retourne le percentile p (médiane par défaut)."""
        arr = ohlcv.to_arrays()
        atr_values = atr_wilder(
            highs=arr["high"],
            lows=arr["low"],
            closes=arr["close"],
            period=self.period,
        )
        # On ne garde que les valeurs finies (les premières bougies n'ont pas d'ATR).
        finite = [v for v in atr_values if not (math.isnan(v) or math.isinf(v))]
        if not finite:
            raise CalibrationError(
                f"ATR({self.period}) non calculable sur {ohlcv.asset}/{ohlcv.timeframe} "
                f"({ohlcv.n_bougies} bougies, {len(finite)} valeurs finies)"
            )
        return percentile(finite, self.percentile_p)


class _ConditionEvaluator:
    """Évalue un arbre de conditions sur un FeaturesFrame.

    Retourne un masque polars de booléens, aligné sur l'index de la frame.

    Invariant V1 : aucune transformation n'est appliquée silencieusement.
    Une `Condition` avec `transformation != None` lève EvaluationError —
    c'est au BNF (P0 #8/#9) de produire des Conditions valides pour V1
    (à savoir `transformation=None`).

    Tout opérateur non supporté lève une EvaluationError.
    """

    def evaluate(self, condition: Condition | ConditionNode, features: FeaturesFrame) -> pl.Series:
        """Évalue la condition sur la frame. Retourne une pl.Series[bool]."""
        if isinstance(condition, Condition):
            return self._eval_atomic(condition, features)
        return self._eval_compound(condition, features)

    def _eval_atomic(self, c: Condition, features: FeaturesFrame) -> pl.Series:
        # Pas de fallback silencieux : une transformation non supportée DOIT
        # être signalée explicitement (évite qu'une Condition "zscore(...)"
        # soit évaluée comme si la transformation n'existait pas, ce qui
        # ferait passer des règles sémantiquement invalides).
        if c.transformation not in (None, ""):
            raise EvaluationError(
                f"Transformation non supportée : {c.transformation!r} (feature={c.feature_ref}). "
                "Le moteur V1 n'implémente aucune transformation explicite — "
                "toute Condition avec transformation doit être levée par le BNF/P0 #9."
            )
        if not features.has(c.feature_ref):
            # Feature inconnue → False partout (NaN-propagation, règle dure S-1).
            return pl.Series("cond", [False] * features.n_bougies)
        col = features.column(c.feature_ref)
        return self._apply_op(col, c.operator, c.value)

    @staticmethod
    def _apply_op(col: pl.Series, op: CompareOp, value: Any) -> pl.Series:
        if op == CompareOp.LT:
            return (col < value).fill_null(False)
        if op == CompareOp.LE:
            return (col <= value).fill_null(False)
        if op == CompareOp.GT:
            return (col > value).fill_null(False)
        if op == CompareOp.GE:
            return (col >= value).fill_null(False)
        if op == CompareOp.EQ:
            # Comparaison flottante : isclose avec tolérance.
            return (col - value).abs() < 1e-9
        if op == CompareOp.NE:
            return (col - value).abs() >= 1e-9
        if op == CompareOp.IN:
            # value attendu itérable
            return col.is_in(value)
        raise EvaluationError(f"Opérateur non supporté : {op}")

    def _eval_compound(self, node: ConditionNode, features: FeaturesFrame) -> pl.Series:
        left = self.evaluate(node.left, features) if isinstance(
            node.left, (Condition, ConditionNode)
        ) else node.left
        if node.op == LogicalOp.NOT:
            return ~left
        right = self.evaluate(node.right, features) if isinstance(
            node.right, (Condition, ConditionNode)
        ) else None
        if right is None:
            raise EvaluationError(f"Opérateur {node.op} requiert un noeud right")
        if node.op == LogicalOp.AND:
            return left & right
        if node.op == LogicalOp.OR:
            return left | right
        if node.op == LogicalOp.XOR:
            return left ^ right
        raise EvaluationError(f"Opérateur logique non supporté : {node.op}")


class _SignalFilter:
    """Applique le cooldown K sur un masque de signaux.

    Si deux signaux sont à moins de K bougies, seul le premier est conservé.
    """

    def __init__(self, cooldown_k: int) -> None:
        if cooldown_k < 1:
            raise ValueError(f"cooldown_k doit être >= 1, got {cooldown_k}")
        self.cooldown_k = cooldown_k

    def filter(self, signal_mask: pl.Series) -> list[int]:
        """Retourne la liste des indices de signaux retenus (après cooldown)."""
        signals = signal_mask.to_numpy().astype(bool)
        kept: list[int] = []
        last_kept = -10**9
        for i, s in enumerate(signals):
            if not s:
                continue
            if i - last_kept >= self.cooldown_k:
                kept.append(i)
                last_kept = i
        return kept


class _ATRSeries:
    """Pré-calcule l'ATR(14) sur toute une série OHLCV, indexable par bougie.

    Permet de récupérer l'ATR local à l'index d'entrée d'un trade,
    pour recalculer les niveaux SL/TP en distances relatives.

    Attributes:
        atr: np.ndarray de même longueur que la série (NaN pour les
            premières bougies où l'ATR n'est pas calculable).
        period: Période de l'ATR (Wilder).
    """

    def __init__(self, ohlcv: OhlcvFrame, period: int = 14) -> None:
        arr = ohlcv.to_arrays()
        self.atr = atr_wilder(
            highs=arr["high"],
            lows=arr["low"],
            closes=arr["close"],
            period=period,
        )
        self.period = period

    def at_index(self, idx: int, fallback: float) -> float:
        """Retourne l'ATR à l'index `idx`. Si NaN, retourne `fallback` (typiquement atr_p50)."""
        v = float(self.atr[idx]) if 0 <= idx < len(self.atr) else float("nan")
        if math.isnan(v) or math.isinf(v) or v <= 0:
            return fallback
        return v


class _TradeRunner:
    """Exécute un trade : entrée à l'OPEN de t+1, simulation intrabar, application des coûts.

    Délègue la simulation intrabar à `engine.simulator.simulate`.
    """

    def __init__(self, direction: Direction, costs: TradingCosts) -> None:
        self.direction = direction
        self.costs = costs

    def run(
        self,
        entry_idx: int,
        ohlcv_arrays: dict[str, np.ndarray],
        calibrated: CalibratedParams,
        atr_at_entry: float,
    ) -> TradeMesure | None:
        """Simule un trade à partir de l'index de signal `entry_idx`.

        L'entrée est l'OPEN de la bougie t+1 (entry_idx + 1). La fenêtre
        d'observation va de t+1 à t+N. Si la fenêtre déborde, retourne None.

        SL et TP sont calculés à l'entrée (multiples d'ATR × ATR local),
        conformément à l'invariant I-5 (jamais de prix absolu figé).

        Args:
            entry_idx: Index du signal (t).
            ohlcv_arrays: Arrays numpy de la série OHLCV.
            calibrated: CalibratedParams figée depuis train.
            atr_at_entry: ATR local calculé sur la fenêtre se terminant à t+1.

        Returns:
            TradeMesure, ou None si la fenêtre déborde.
        """
        n_window = calibrated.n_window
        entry_pos = entry_idx + 1  # entrée à l'OPEN de t+1
        end_pos = entry_pos + n_window
        if end_pos > len(ohlcv_arrays["close"]):
            return None

        opens = ohlcv_arrays["open"][entry_pos:end_pos]
        highs = ohlcv_arrays["high"][entry_pos:end_pos]
        lows = ohlcv_arrays["low"][entry_pos:end_pos]
        closes = ohlcv_arrays["close"][entry_pos:end_pos]

        if len(opens) == 0:
            return None

        entry_price = float(opens[0])
        if entry_price <= 0:
            return None  # bougie invalide

        # SL/TP recalculés à l'entrée (anti prix absolu figé).
        sl_price, tp_price = calibrated.compute_sl_tp_at_entry(
            entry_price=entry_price,
            atr_at_entry=atr_at_entry,
            direction=self.direction,
        )

        # Délégation au simulateur intrabar.
        exit_price, exit_reason, mfe, mae, n_held = simulate(
            direction=self.direction,
            entry=entry_price,
            sl_price=sl_price,
            tp_price=tp_price,
            highs=highs.tolist(),
            lows=lows.tolist(),
            closes=closes.tolist(),
        )

        # Rendement brut (en %).
        if self.direction == Direction.LONG:
            ret_brut = (exit_price - entry_price) / entry_price
        else:
            ret_brut = (entry_price - exit_price) / entry_price

        # Application des coûts (round-trip).
        ret_net = ret_brut - self.costs.total_round_trip_pct

        return TradeMesure(
            entry_idx=entry_pos,
            exit_idx=entry_pos + n_held - 1,
            entry_price=entry_price,
            exit_price=exit_price,
            exit_reason=exit_reason,
            mfe_pct=mfe / entry_price,
            mae_pct=mae / entry_price,
            ret_pct_brut=ret_brut,
            ret_pct_net=ret_net,
            n_bougies_held=n_held,
        )


class _MesuresAggregator:
    """Agrège une liste de TradeMesure en MesuresBrutes + calcule le block bootstrap CI."""

    def __init__(
        self,
        config: EinherjarConfig,
        calibrated: CalibratedParams,
        costs: TradingCosts,
        timeframe: str = "1d",
    ) -> None:
        self._config = config
        self._calibrated = calibrated
        self._costs = costs
        # P2 #1 : periods_per_year dynamique selon le timeframe (plus de sqrt(365) hardcoded).
        self._periods_per_year = periods_per_year_for_timeframe(timeframe)

    def aggregate(
        self,
        trades: list[TradeMesure],
        per_asset_trades: dict[str, list[TradeMesure]],
    ) -> MesuresBrutes:
        """Construit MesuresBrutes à partir de la liste de trades."""
        n = len(trades)
        if n == 0:
            return self._empty_mesures()

        returns_net = [t.ret_pct_net for t in trades]
        mfe_list = [t.mfe_pct for t in trades]
        mae_list = [t.mae_pct for t in trades]
        n_tp = sum(1 for t in trades if t.exit_reason == ExitReason.TP)
        n_sl = sum(1 for t in trades if t.exit_reason == ExitReason.SL)
        n_to = sum(1 for t in trades if t.exit_reason == ExitReason.TIMEOUT)
        held = [t.n_bougies_held for t in trades]

        # Block bootstrap CI.
        bs_sharpe = bootstrap_sharpe(returns_net, self._config)
        bs_ret = bootstrap_ret_total(returns_net, self._config)

        # Sharpe annualisé (rendements par bougie → on suppose 1 bougie = 1 unité).
        ret_mean = float(np.mean(returns_net)) if n else float("nan")
        ret_std = float(np.std(returns_net, ddof=1)) if n > 1 else float("nan")
        if ret_std and not math.isnan(ret_std) and ret_std > 0:
            sharpe = (ret_mean / ret_std) * math.sqrt(self._periods_per_year)
        else:
            sharpe = float("nan")

        # Per-asset stats.
        per_asset_stats: dict[str, MesuresBrutes] = {}
        for asset, asset_trades in per_asset_trades.items():
            if not asset_trades:
                continue
            per_asset_stats[asset] = self._aggregate_subset(asset_trades)

        return MesuresBrutes(
            n_signals=n,
            n_tp_hit=n_tp,
            n_sl_hit=n_sl,
            n_timeout=n_to,
            mfe_mean_pct=float(np.mean(mfe_list)),
            mae_mean_pct=float(np.mean(mae_list)),
            mfe_p50=percentile(mfe_list, 50),
            mfe_p75=percentile(mfe_list, 75),
            mfe_p90=percentile(mfe_list, 90),
            mae_p50=percentile(mae_list, 50),
            mae_p75=percentile(mae_list, 75),
            mae_p90=percentile(mae_list, 90),
            ret_mean_pct_net=ret_mean,
            ret_std_pct=ret_std,
            sharpe_net=sharpe,
            tp_hit_rate=n_tp / n if n else 0.0,
            sl_hit_rate=n_sl / n if n else 0.0,
            timeout_rate=n_to / n if n else 0.0,
            avg_holding_period=float(np.mean(held)) if held else 0.0,
            avg_time_to_amplitude=(
                float(np.mean([t.n_bougies_held for t in trades if t.exit_reason == ExitReason.TP]))
                if n_tp else 0.0
            ),
            bootstrap_sharpe_ci_low=bs_sharpe.ci_low,
            bootstrap_sharpe_ci_high=bs_sharpe.ci_high,
            bootstrap_ret_ci_low=bs_ret.ci_low,
            bootstrap_ret_ci_high=bs_ret.ci_high,
            per_asset_stats=per_asset_stats,
            trades=tuple(trades),
            n_window=self._calibrated.n_window,
            sl_n_atr=self._calibrated.sl_n_atr, sl_distance=self._calibrated.sl_distance,
            tp_n_atr=self._calibrated.tp_n_atr, tp_distance=self._calibrated.tp_distance,
            costs_applied=self._costs.to_dict(),
        )

    def _aggregate_subset(self, trades: list[TradeMesure]) -> MesuresBrutes:
        """Agrège un sous-ensemble de trades (par-asset). Les IC bootstrap sont vides."""
        n = len(trades)
        if n == 0:
            return self._empty_mesures()
        returns = [t.ret_pct_net for t in trades]
        mfe = [t.mfe_pct for t in trades]
        mae = [t.mae_pct for t in trades]
        n_tp = sum(1 for t in trades if t.exit_reason == ExitReason.TP)
        n_sl = sum(1 for t in trades if t.exit_reason == ExitReason.SL)
        n_to = sum(1 for t in trades if t.exit_reason == ExitReason.TIMEOUT)
        ret_std = (
            float(np.std(returns, ddof=1)) if n > 1 else float("nan")
        )
        ret_mean = float(np.mean(returns)) if n else float("nan")
        sharpe = (ret_mean / ret_std) * math.sqrt(self._periods_per_year) if ret_std and ret_std > 0 else float("nan")
        return MesuresBrutes(
            n_signals=n, n_tp_hit=n_tp, n_sl_hit=n_sl, n_timeout=n_to,
            mfe_mean_pct=float(np.mean(mfe)), mae_mean_pct=float(np.mean(mae)),
            mfe_p50=percentile(mfe, 50), mfe_p75=percentile(mfe, 75), mfe_p90=percentile(mfe, 90),
            mae_p50=percentile(mae, 50), mae_p75=percentile(mae, 75), mae_p90=percentile(mae, 90),
            ret_mean_pct_net=ret_mean, ret_std_pct=ret_std, sharpe_net=sharpe,
            tp_hit_rate=n_tp / n if n else 0.0,
            sl_hit_rate=n_sl / n if n else 0.0,
            timeout_rate=n_to / n if n else 0.0,
            avg_holding_period=float(np.mean([t.n_bougies_held for t in trades])),
            avg_time_to_amplitude=0.0,
            bootstrap_sharpe_ci_low=float("nan"),
            bootstrap_sharpe_ci_high=float("nan"),
            bootstrap_ret_ci_low=float("nan"),
            bootstrap_ret_ci_high=float("nan"),
            per_asset_stats={},
            trades=tuple(trades),
            n_window=self._calibrated.n_window,
            sl_n_atr=self._calibrated.sl_n_atr, sl_distance=self._calibrated.sl_distance,
            tp_n_atr=self._calibrated.tp_n_atr, tp_distance=self._calibrated.tp_distance,
            costs_applied=self._costs.to_dict(),
        )

    def _empty_mesures(self) -> MesuresBrutes:
        return MesuresBrutes(
            n_signals=0, n_tp_hit=0, n_sl_hit=0, n_timeout=0,
            mfe_mean_pct=0.0, mae_mean_pct=0.0,
            mfe_p50=0.0, mfe_p75=0.0, mfe_p90=0.0,
            mae_p50=0.0, mae_p75=0.0, mae_p90=0.0,
            ret_mean_pct_net=0.0, ret_std_pct=0.0, sharpe_net=float("nan"),
            tp_hit_rate=0.0, sl_hit_rate=0.0, timeout_rate=0.0,
            avg_holding_period=0.0, avg_time_to_amplitude=0.0,
            bootstrap_sharpe_ci_low=0.0, bootstrap_sharpe_ci_high=0.0,
            bootstrap_ret_ci_low=0.0, bootstrap_ret_ci_high=0.0,
            per_asset_stats={}, trades=(),
            n_window=self._calibrated.n_window,
            sl_n_atr=self._calibrated.sl_n_atr, sl_distance=self._calibrated.sl_distance,
            tp_n_atr=self._calibrated.tp_n_atr, tp_distance=self._calibrated.tp_distance,
            costs_applied=self._costs.to_dict(),
        )


# --------------------------------------------------------------------------- #
# Moteur public
# --------------------------------------------------------------------------- #


class EvaluationEngine:
    """Moteur d'évaluation hors-échantillon (PRIORITÉ 0).

    Usage type :
        engine = EvaluationEngine(config, data_version="v1", seed=42)
        calibrated = engine.train_calibrate(hypothesis, train_ohlcv, train_features)
        m_val = engine.test_on(hypothesis, val_ohlcv, val_features, calibrated, "val")
        # ... éventuellement, plus tard, sur le holdout final :
        m_holdout = engine.test_on(hypothesis, holdout_ohlcv, holdout_features, calibrated, "holdout")

    Attributes:
        config: Configuration chargée.
        data_version: Identifiant de version de données.
        seed: Graine RNG maître.
    """

    def __init__(self, config: EinherjarConfig, data_version: str, seed: int = 42) -> None:
        """Initialise le moteur d'évaluation.

        Args:
            config: Configuration chargée (config/).
            data_version: Identifiant de version de données (pour traçabilité).
            seed: Graine RNG maître (défaut 42).
        """
        self.config = config
        self.data_version = data_version
        self.seed = seed
        # Compteur d'accès au holdout (1 maximum, pour traçabilité I-5).
        self._holdout_accessed: bool = False

        ev_cfg = config.evaluation
        self._atr_estimator = _ATREstimator(
            period=int(ev_cfg["atr"]["period"]),
            percentile_p=float(ev_cfg["atr"]["percentile"]),
        )
        self._min_n = int(ev_cfg["n_window"]["min_n"])
        self._max_n = int(ev_cfg["n_window"]["max_n"])
        self._condition_evaluator = _ConditionEvaluator()

        logger.info(
            "EvaluationEngine instancié : data_version=%s, seed=%d, ATR period=%d p=%.0f, N=[%d, %d]",
            data_version, seed, self._atr_estimator.period, self._atr_estimator.percentile_p,
            self._min_n, self._max_n,
        )

    # ------------------------------------------------------------------ #
    # API publique
    # ------------------------------------------------------------------ #

    def train_calibrate(
        self,
        hypothesis: Hypothesis,
        train_ohlcv: OhlcvFrame,
        train_features: FeaturesFrame,
    ) -> CalibratedParams:
        """Calibre N, sl_n_atr et tp_n_atr sur le train UNIQUEMENT.

        Algorithme :
          1. ATR_p50 sur le train.
          2. N = clamp(ceil(amplitude / atr_p50), min_N, max_N).
          3. Passe provisoire avec distances 1.5×ATR pour SL et TP.
             → Mesure MFE_p50 (en %) et MAE_p75 (en %) sur de vrais trades,
               avec prix et ATR locaux observés (PAS de prix neutre 1.0).
          4. Convertit MFE_p50 → tp_n_atr (= MFE_p50_atr / atr_p50) et
             MAE_p75 → sl_n_atr (= MAE_p75_atr / atr_p50).
          5. Stocke ces distances (multiples d'ATR) dans CalibratedParams.

        Aucun accès au val/holdout ici. Aucun prix absolu n'est figé.
        """
        logger.info(
            "Calibration train : %s × %s (N_window cible)",
            train_ohlcv.asset, train_ohlcv.timeframe,
        )

        # 1. ATR_p50.
        atr_p50 = self._atr_estimator.p50(train_ohlcv)
        if atr_p50 <= 0:
            raise CalibrationError(f"ATR_p50 non positif : {atr_p50}")

        # 2. N.
        amplitude = hypothesis.amplitude
        if amplitude.unité == AmplitudeUnit.PRICE_ABSOLU:
            n_window = self._compute_n_price_absolu(amplitude.valeur, atr_p50)
        elif amplitude.unité == AmplitudeUnit.MULTIPLE_ATR:
            n_window = self._compute_n_multiple_atr(amplitude.valeur, train_ohlcv.timeframe, atr_p50)
        else:
            raise CalibrationError(f"Unité d'amplitude non supportée : {amplitude.unité}")

        # 3. Passe provisoire avec distances 1.5×ATR (symétriques pour avoir
        # un premier signal mesurable). C'est une vraie simulation sur
        # de vrais trades avec prix et ATR observés.
        provisional_n_atr = 1.5
        provisional_calibrated = CalibratedParams(
            n_window=n_window,
            sl_n_atr=provisional_n_atr,
            tp_n_atr=provisional_n_atr,
            sl_distance=provisional_n_atr,  # approximation initiale (1.5 = 150%)
            tp_distance=provisional_n_atr,
            atr_p50=atr_p50,
            n_observations=train_ohlcv.n_bougies,
        )

        # Simule les trades sur le train pour mesurer MFE/MAE.
        train_trades = self._run_all_trades(
            hypothesis=hypothesis,
            ohlcv=train_ohlcv,
            features=train_features,
            calibrated=provisional_calibrated,
        )

        if not train_trades:
            raise CalibrationError(
                f"Aucun signal sur le train pour {hypothesis.id} — calibration impossible"
            )

        # 4. Calcule MFE_p50 et MAE_p75 (en %) sur le train.
        mfe_p50 = percentile([t.mfe_pct for t in train_trades], 50)
        mae_p75 = percentile([t.mae_pct for t in train_trades], 75)

        # 5. Convertit en distances ATR.
        # MFE_p50 est en % du prix. Pour convertir en multiple d'ATR :
        #   mfe_p50_atr = (mfe_p50 * entry_median) / atr_p50
        # On utilise l'entry médian pour la conversion.
        entry_median = float(np.median([t.entry_price for t in train_trades]))
        if atr_p50 > 0 and entry_median > 0:
            tp_n_atr = (mfe_p50 * entry_median) / atr_p50
            sl_n_atr = (mae_p75 * entry_median) / atr_p50
        else:
            tp_n_atr = provisional_n_atr
            sl_n_atr = provisional_n_atr

        # Bornes de sécurité : SL et TP doivent être positifs et bornés.
        sl_n_atr = max(0.1, min(20.0, sl_n_atr))
        tp_n_atr = max(0.1, min(50.0, tp_n_atr))

        sl_distance = (sl_n_atr * atr_p50) / entry_median if entry_median > 0 else 0.0
        tp_distance = (tp_n_atr * atr_p50) / entry_median if entry_median > 0 else 0.0

        calibrated = CalibratedParams(
            n_window=n_window,
            sl_n_atr=sl_n_atr,
            tp_n_atr=tp_n_atr,
            sl_distance=sl_distance,
            tp_distance=tp_distance,
            atr_p50=atr_p50,
            n_observations=train_ohlcv.n_bougies,
            mfe_p50=mfe_p50,
            mae_p75=mae_p75,
        )
        logger.info(
            "Calibration OK : N=%d, sl_n_atr=%.3f (dist=%.3f%%), "
            "tp_n_atr=%.3f (dist=%.3f%%), ATR_p50=%.4f, %d trades train",
            calibrated.n_window, calibrated.sl_n_atr, calibrated.sl_distance * 100,
            calibrated.tp_n_atr, calibrated.tp_distance * 100, calibrated.atr_p50,
            len(train_trades),
        )
        return calibrated

    def test_on(
        self,
        hypothesis: Hypothesis,
        ohlcv: OhlcvFrame,
        features: FeaturesFrame,
        calibrated: CalibratedParams,
        split_name: str,
    ) -> MesuresBrutes:
        """Évalue l'hypothèse sur un split, en utilisant la CalibratedParams figée.

        Le holdout n'est consultable qu'UNE SEULE FOIS par Einher final.
        Toute seconde tentative lève HoldoutAccessError.

        Args:
            hypothesis: Hypothèse à tester.
            ohlcv: Frame OHLCV du split.
            features: FeaturesFrame du split.
            calibrated: CalibratedParams figée (depuis train).
            split_name: 'train' | 'val' | 'holdout'.

        Returns:
            MesuresBrutes calculées sur ce split (sans recalibrage).
        """
        if split_name == HOLDOUT_SPLIT_NAME:
            if self._holdout_accessed:
                raise HoldoutAccessError(
                    f"Holdout déjà accédé une fois — accès interdit (I-5, S-3.8). "
                    f"Hypothèse {hypothesis.id} ne peut être évaluée qu'une seule fois sur le holdout."
                )
            self._holdout_accessed = True
            logger.warning(
                "⚠ ACCÈS HOLDOUT : %s × %s, hyp=%s, timestamp=%s",
                ohlcv.asset, ohlcv.timeframe, hypothesis.id, datetime.now(UTC).isoformat(),
            )

        logger.info(
            "test_on(%s) : %s × %s, %d bougies, %d features",
            split_name, ohlcv.asset, ohlcv.timeframe, ohlcv.n_bougies, features.n_features,
        )

        costs = TradingCosts.from_config(self.config, asset=ohlcv.asset)
        runner = _TradeRunner(direction=hypothesis.direction, costs=costs)

        # Évalue la condition → masque → indices de signaux (après cooldown).
        mask = self._condition_evaluator.evaluate(hypothesis.condition_tree, features)
        signal_filter = _SignalFilter(cooldown_k=hypothesis.cooldown_k)
        signal_indices = signal_filter.filter(mask)

        # Pré-calcule la série ATR pour récupérer l'ATR local à chaque entrée.
        atr_series = _ATRSeries(ohlcv, period=self._atr_estimator.period)
        atr_fallback = calibrated.atr_p50

        # Simule les trades.
        ohlcv_arrays = ohlcv.to_arrays()
        trades: list[TradeMesure] = []
        for idx in signal_indices:
            atr_at_entry = atr_series.at_index(idx + 1, atr_fallback)
            trade = runner.run(
                entry_idx=idx,
                ohlcv_arrays=ohlcv_arrays,
                calibrated=calibrated,
                atr_at_entry=atr_at_entry,
            )
            if trade is not None:
                trades.append(trade)

        # Agrège.
        aggregator = _MesuresAggregator(self.config, calibrated, costs, timeframe=ohlcv.timeframe)
        # Per-asset : ici un seul asset, mais on garde la structure.
        per_asset = {ohlcv.asset: trades} if trades else {}
        mesures = aggregator.aggregate(trades=trades, per_asset_trades=per_asset)

        logger.info(
            "test_on(%s) OK : %d signaux → %d trades (TP=%d, SL=%d, TO=%d), Sharpe=%.3f [%.3f, %.3f]",
            split_name, len(signal_indices), mesures.n_signals,
            mesures.n_tp_hit, mesures.n_sl_hit, mesures.n_timeout,
            mesures.sharpe_net, mesures.bootstrap_sharpe_ci_low, mesures.bootstrap_sharpe_ci_high,
        )
        return mesures

    def evaluate(
        self,
        hypothesis: Hypothesis,
        train_ohlcv: OhlcvFrame,
        train_features: FeaturesFrame,
        val_ohlcv: OhlcvFrame,
        val_features: FeaturesFrame,
        holdout_ohlcv: OhlcvFrame | None = None,
        holdout_features: FeaturesFrame | None = None,
    ) -> tuple[MesuresBrutes, MesuresBrutes, CalibratedParams, MesuresBrutes | None]:
        """Pipeline complet : calibration train + test val + test (unique) holdout optionnel.

        Returns:
            (m_train, m_val, calibrated, m_holdout) où m_holdout=None si pas de holdout fourni.
        """
        calibrated = self.train_calibrate(hypothesis, train_ohlcv, train_features)
        m_train = self.test_on(hypothesis, train_ohlcv, train_features, calibrated, "train")
        m_val = self.test_on(hypothesis, val_ohlcv, val_features, calibrated, "val")
        m_holdout: MesuresBrutes | None = None
        if holdout_ohlcv is not None and holdout_features is not None:
            m_holdout = self.test_on(hypothesis, holdout_ohlcv, holdout_features, calibrated, "holdout")
        return m_train, m_val, calibrated, m_holdout

    # ------------------------------------------------------------------ #
    # Helpers privés — calcul N, SL/TP
    # ------------------------------------------------------------------ #

    def _compute_n_price_absolu(self, amplitude_value: float, atr_p50: float) -> int:
        n = math.ceil(amplitude_value / atr_p50)
        return max(self._min_n, min(self._max_n, n))

    def _compute_n_multiple_atr(self, amplitude_value: float, timeframe: str, atr_p50: float) -> int:
        k_atr_map = self.config.evaluation["n_window"].get("k_atr_by_timeframe", {})
        k_atr = float(k_atr_map.get(timeframe, self.config.evaluation["n_window"].get("default_k_atr", 1.0)))
        # amplitude.valeur est en multiple d'ATR, donc N = amplitude × k_atr (arrondi).
        n = round(amplitude_value * k_atr)
        return max(self._min_n, min(self._max_n, n))

    def _run_all_trades(
        self,
        hypothesis: Hypothesis,
        ohlcv: OhlcvFrame,
        features: FeaturesFrame,
        calibrated: CalibratedParams,
    ) -> list[TradeMesure]:
        """Évalue la condition et simule tous les trades (helper interne)."""
        costs = TradingCosts.from_config(self.config, asset=ohlcv.asset)
        runner = _TradeRunner(direction=hypothesis.direction, costs=costs)
        mask = self._condition_evaluator.evaluate(hypothesis.condition_tree, features)
        signal_filter = _SignalFilter(cooldown_k=hypothesis.cooldown_k)
        indices = signal_filter.filter(mask)
        arrays = ohlcv.to_arrays()
        atr_series = _ATRSeries(ohlcv, period=self._atr_estimator.period)
        atr_fallback = calibrated.atr_p50
        trades: list[TradeMesure] = []
        for i in indices:
            atr_at_entry = atr_series.at_index(i + 1, atr_fallback)
            t = runner.run(
                entry_idx=i,
                ohlcv_arrays=arrays,
                calibrated=calibrated,
                atr_at_entry=atr_at_entry,
            )
            if t is not None:
                trades.append(t)
        return trades


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #


def make_default_engine(
    config: EinherjarConfig,
    data_version: str,
    seed: int = 42,
) -> EvaluationEngine:
    """Construit un EvaluationEngine avec la config chargée.

    Args:
        config: Configuration chargée (de `research.config.loader.load_config`).
        data_version: Identifiant de version de données (pour traçabilité).
        seed: Graine RNG maître (défaut 42).

    Returns:
        EvaluationEngine prêt à l'emploi.
    """
    return EvaluationEngine(config=config, data_version=data_version, seed=seed)
