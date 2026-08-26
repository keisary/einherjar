"""types.py - Types de données du pipeline xgb_einhers.

Tous les dataclasses sont frozen (immutables) pour reproductibilité.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np

# --------------------------------------------------------------------------- #
# Données brutes
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class LoadedData:
    """Données chargées depuis MIDAS V3 pour un (asset, TF)."""
    asset: str
    asset_class: str
    timeframe: str
    timestamps: np.ndarray         # (N,) int64 ms epoch
    X: np.ndarray                  # (N, F) float32, OHLCV déjà exclues
    Y_dir: np.ndarray              # (N, H) int8 {-100, 0=SELL, 1=HOLD, 2=BUY}
    Y_ret: np.ndarray              # (N, H) float32, signed
    Y_hor: np.ndarray              # (N, H) float32, bars
    feature_names: tuple[str, ...]   # noms des colonnes de X (OHLCV exclues)
    horizons: tuple[str, ...]        # noms ("6h", "12h", "1d", "2d")

    @property
    def n_samples(self) -> int:
        """n_samples."""
        return self.X.shape[0]

    @property
    def n_features(self) -> int:
        """n_features."""
        return self.X.shape[1]

    @property
    def n_horizons(self) -> int:
        """n_horizons."""
        return len(self.horizons)


@dataclass(frozen=True)
class TrainValHoldoutSplit:
    """Split temporel 60/20/20 avec embargo."""
    train_X: np.ndarray
    train_y: np.ndarray
    val_X: np.ndarray
    val_y: np.ndarray
    holdout_X: np.ndarray
    holdout_y: np.ndarray
    train_indices: np.ndarray
    val_indices: np.ndarray
    holdout_indices: np.ndarray
    embargo_bars: int


# --------------------------------------------------------------------------- #
# Trade (résultat d'un backtest)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TradeResult:
    """Résultat d'un trade simulé."""
    entry_idx: int
    exit_idx: int
    entry_price: float
    exit_price: float
    exit_reason: str          # 'tp' | 'sl' | 'timeout'
    gross_return: float
    net_return: float        # après coûts
    n_bars_held: int
    entry_timestamp_ms: int
    exit_timestamp_ms: int


# --------------------------------------------------------------------------- #
# Métriques
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class EinherMetrics:
    """Métriques d'un Einher calculées par le backtester."""
    n_trades: int
    n_tp: int
    n_sl: int
    n_timeout: int
    win_rate: float
    avg_net_return: float
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    profit_factor: float
    avg_holding_bars: float
    buy_hold_return: float
    alpha: float
    # Sprint 3.3 FIX BUG-02 : t-stat et p-value pour correction multi-tests (BH)
    t_statistic: float = 0.0  # t = mean(rets) / (std(rets) / sqrt(n))
    p_value: float = 1.0      # p-value bilaterale H0: mean(rets) = 0
    # Rendements par trade (utilises pour vrai bootstrap si on veut)
    trade_returns: tuple[float, ...] = field(default_factory=tuple)
    # FIX METRICS (2026-08-21) : taux de sorties par TP, distinct du win_rate.
    # win_rate = % de trades avec net_return > 0 (inclut les timeouts gagnants).
    # tp_hit_rate = % de trades sortis par take-profit uniquement.
    tp_hit_rate: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialise les metriques (SANS trade_returns - artefact de prod).

        FIX PROD (2026-08-26) : trade_returns exclu du JSON. Les Einhers sont
        des artefacts de production : seules les metriques synthetiques comptent.
        Mesure : trade_returns = 82% du poids des lignes JSONL du corpus, sans
        valeur d'usage en prod. Disponibles EN MEMOIRE pendant le run.
        """
        d = {k: v for k, v in asdict(self).items() if k != "trade_returns"}
        return d

    def passes_admission(
        self,
        min_trades: int = 30,
        min_sharpe: float = 0.3,
        min_win_rate: float = 0.40,
        min_profit_factor: float = 1.0,
        max_drawdown: float = 0.30,
    ) -> tuple[bool, str | None]:
        """Vérifie les critères d'admission minimaux.

        Returns:
            (passed, reason) : passed=True si tous critères OK, sinon reason = raison du rejet
        """
        if self.n_trades < min_trades:
            return False, f"n_trades={self.n_trades} < {min_trades}"
        if self.sharpe_ratio < min_sharpe:
            return False, f"sharpe={self.sharpe_ratio:.3f} < {min_sharpe}"
        if self.win_rate < min_win_rate:
            return False, f"win_rate={self.win_rate:.3f} < {min_win_rate}"
        if self.profit_factor < min_profit_factor:
            return False, f"profit_factor={self.profit_factor:.3f} < {min_profit_factor}"
        # Sprint 3.3 FIX BUG-01 : max_drawdown est stocke NEGATIF (dd = eq - peak <= 0)
        # On utilise abs() pour comparer au seuil positif
        if abs(self.max_drawdown) > max_drawdown:
            return False, f"max_drawdown={abs(self.max_drawdown):.3f} > {max_drawdown}"
        if self.total_return <= 0:
            return False, f"total_return={self.total_return:.4f} <= 0"
        return True, None


# --------------------------------------------------------------------------- #
# Einher
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Condition:
    """Une condition atomique : feature + op + value."""
    feature_ref: str
    operator: str              # '<' | '<=' | '>' | '>=' | '==' | '!='
    value: float | int
    transformation: str | None = None
    expr: object | None = None  # STGP: expression numerique (search_engine) sinon None

    def to_dict(self) -> dict[str, Any]:
        """to_dict."""
        d = {k: v for k, v in asdict(self).items() if v is not None}
        if self.expr is not None and hasattr(self.expr, "to_dict"):
            d["expr"] = self.expr.to_dict()  # pyright: ignore[reportAttributeAccessIssue]
        return d


@dataclass(frozen=True)
class ConditionNode:
    """Noeud d'un arbre de conditions : AND/OR/NOT/XOR + enfants."""
    op: str                    # 'AND' | 'OR' | 'NOT' | 'XOR'
    left: Condition | ConditionNode
    right: Condition | ConditionNode | None = None  # None pour NOT unaire

    def to_dict(self) -> dict[str, Any]:
        """to_dict."""
        d = {"op": self.op, "left": self.left.to_dict()}
        if self.right is not None:
            d["right"] = self.right.to_dict()
        return d


# Alias pour faciliter les annotations circulaires
Condition.__class_getitem__ = lambda cls, x: cls  # pyright: ignore[reportAttributeAccessIssue]
ConditionNode.__class_getitem__ = lambda cls, x: cls  # pyright: ignore[reportAttributeAccessIssue]


@dataclass(frozen=True)
class Einher:
    """Une stratégie de trading : condition + direction + SL/TP + univers + métriques."""
    id: str
    condition_tree: ConditionNode | Condition
    direction: str                    # 'BUY' | 'SELL'
    amplitude_bars: int               # horizon (en bars)
    tp_pct: float
    sl_pct: float
    universe: dict[str, Any]          # {asset, asset_class, timeframe, horizon, horizon_bars}
    metrics: EinherMetrics
    scope: str                        # 'asset' | 'general' | 'market'
    cross_asset_test: dict[str, Any] | None = None
    source: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    data_version: str = ""
    # Sprint 2.4.1 : holdout metrics pour filtrer les Einhers non significatifs
    holdout_metrics: EinherMetrics | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise l'Einher pour la PRODUCTION (JSONL compact et essentiel).

        FIX PROD (2026-08-26) : nettoyage du format JSONL.
        - cross_asset_test / data_version : systematiquement vides a ce stade
          (audit corpus 24/24) -> exclus ; inclus seulement s'ils sont remplis.
        - holdout_metrics : inclus seulement si present.
        - source.feature_names : redondant avec condition_tree -> exclu.
        Les champs utilises par le moteur de prod (condition, direction,
        amplitude, tp/sl, universe) et les metriques d'admission sont conserves.
        """
        out: dict[str, Any] = {
            "id": self.id,
            "condition_tree": self.condition_tree.to_dict(),
            "direction": self.direction,
            "amplitude_bars": self.amplitude_bars,
            "tp_pct": self.tp_pct,
            "sl_pct": self.sl_pct,
            "universe": self.universe,
            "metrics": self.metrics.to_dict(),
            "scope": self.scope,
            "source": {k: v for k, v in self.source.items() if k != "feature_names"},
            "created_at": self.created_at,
        }
        if self.cross_asset_test:
            out["cross_asset_test"] = self.cross_asset_test
        if self.data_version:
            out["data_version"] = self.data_version
        if self.holdout_metrics is not None:
            out["holdout_metrics"] = self.holdout_metrics.to_dict()
        return out

    @classmethod
    def to_jsonl_line(cls, einher: Einher) -> str:
        """Sérialise un Einher en une ligne JSON (JSONL)."""
        return json.dumps(einher.to_dict(), ensure_ascii=False) + "\n"
