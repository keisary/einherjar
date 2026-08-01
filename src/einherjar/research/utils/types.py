"""utils/types.py — Types de données partagés dans tout le moteur de découverte.

Ces types sont les "vrais noms" des concepts de l'ontologie, instanciés en
objets Python. Ils sont utilisés par tous les autres modules.

Philosophie : on garde des dataclasses frozen (immuables). Toute mutation
passe par la création d'un nouvel objet (hashing + reproductibilité faciles).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# --------------------------------------------------------------------------- #
# Énumérations
# --------------------------------------------------------------------------- #


class Direction(str, Enum):
    LONG = "long"
    SHORT = "short"


class AmplitudeUnit(str, Enum):
    PRICE_ABSOLU = "prix_absolu"
    MULTIPLE_ATR = "multiple_ATR"


class FeatureType(str, Enum):
    ATOMIC = "atomic"
    QUANTITATIVE = "quantitative"
    PATTERN = "pattern"
    SIGNAL = "signal"
    FACTOR = "factor"


class EconomicFamily(str, Enum):
    PRICE_ACTION = "price_action"
    VOLUME_FLOW = "volume_flow"
    MOMENTUM = "momentum"
    TREND = "trend"
    VOLATILITY = "volatility"
    MARKET_REGIME = "market_regime"
    STATISTICAL = "statistical"
    RISK = "risk"
    MICROSTRUCTURE = "microstructure"
    MARKET_STRUCTURE = "market_structure"
    CROSS_ASSET = "cross_asset"
    OTHER = "other"


class LogicalOp(str, Enum):
    AND = "AND"
    OR = "OR"
    NOT = "NOT"
    XOR = "XOR"


class CompareOp(str, Enum):
    LT = "<"
    LE = "<="
    GT = ">"
    GE = ">="
    EQ = "=="
    NE = "!="
    IN = "in"


class ExitReason(str, Enum):
    TP = "tp"
    SL = "sl"
    TIMEOUT = "timeout"


class RejectionReason(str, Enum):
    """Catalogue normalisé des raisons de rejet (S-3.6)."""
    DSR_FAIL = "DSR_FAIL"
    PBO_FAIL = "PBO_FAIL"
    BOOTSTRAP_CI_FAIL = "BOOTSTRAP_CI_FAIL"
    N_TRADES_FAIL = "N_TRADES_FAIL"
    CROSS_ASSET_FAIL = "CROSS_ASSET_FAIL"
    DD_FAIL = "DD_FAIL"
    DIVERSITY_FAIL = "DIVERSITY_FAIL"
    ALREADY_IN_ARCHIVE = "ALREADY_IN_ARCHIVE"
    SEMANTIC_CHANGED = "SEMANTIC_CHANGED"
    OTHER = "OTHER"
    # Raisons opérationnelles
    EVALUATION_ERROR = "EVALUATION_ERROR"
    TIMEOUT = "TIMEOUT"
    MEMORY_ERROR = "MEMORY_ERROR"


# --------------------------------------------------------------------------- #
# Conditions
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Condition:
    """Une condition atomique : feature + op + valeur (+ transformation opt)."""

    feature_ref: str
    operator: CompareOp
    value: float | int | str
    transformation: str | None = None     # ex: "percentile(20)", "crossover(sma_50)"

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_ref": self.feature_ref,
            "operator": self.operator.value,
            "value": self.value,
            "transformation": self.transformation,
        }


@dataclass(frozen=True)
class ConditionNode:
    """Noeud d'un arbre de conditions. Soit feuille (Condition), soit composé (gauche, droite, op)."""

    op: LogicalOp
    left: Condition | ConditionNode
    right: Condition | ConditionNode | None = None  # None pour NOT unaire

    def to_dict(self) -> dict[str, Any]:
        d = {
            "op": self.op.value,
            "left": self.left.to_dict() if isinstance(self.left, ConditionNode) else self.left.to_dict(),
        }
        if self.right is not None:
            d["right"] = self.right.to_dict() if isinstance(self.right, ConditionNode) else self.right.to_dict()
        return d


# --------------------------------------------------------------------------- #
# Amplitude
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Amplitude:
    valeur: float
    unité: AmplitudeUnit
    direction_implicite: Direction

    def to_dict(self) -> dict[str, Any]:
        return {
            "valeur": self.valeur,
            "unité": self.unité.value,
            "direction_implicite": self.direction_implicite.value,
        }


# --------------------------------------------------------------------------- #
# Universe (couple asset × timeframe)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Universe:
    assets: tuple[str, ...]              # ex: ("BTCUSD", "ETHUSD", "SOLUSD")
    timeframes: tuple[str, ...]          # ex: ("1h", "4h")
    # `*` = wildcards (gérés par le générateur, pas ici)

    def to_dict(self) -> dict[str, Any]:
        return {
            "assets": list(self.assets),
            "timeframes": list(self.timeframes),
        }


# --------------------------------------------------------------------------- #
# Hypothesis
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Hypothesis:
    """Unité de recherche. Candidat à la validation, pas encore un Einher."""

    id: str
    condition_tree: Condition | ConditionNode
    amplitude: Amplitude
    direction: Direction
    universe: Universe
    cooldown_k: int = 5                  # K bougies minimum entre 2 signaux
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "condition_tree": self.condition_tree.to_dict() if isinstance(self.condition_tree, ConditionNode) else self.condition_tree.to_dict(),
            "amplitude": self.amplitude.to_dict(),
            "direction": self.direction.value,
            "universe": self.universe.to_dict(),
            "cooldown_k": self.cooldown_k,
            "meta": self.meta,
        }


# --------------------------------------------------------------------------- #
# MesuresBrutes (sortie du moteur d'évaluation)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TradeMesure:
    """Mesures d'un trade individuel."""

    entry_idx: int
    exit_idx: int
    entry_price: float
    exit_price: float
    exit_reason: ExitReason
    mfe_pct: float                  # max favorable excursion, en %
    mae_pct: float                  # max adverse excursion, en %
    ret_pct_brut: float             # rendement brut (sans frais)
    ret_pct_net: float              # rendement net (après frais)
    n_bougies_held: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_idx": self.entry_idx,
            "exit_idx": self.exit_idx,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "exit_reason": self.exit_reason.value,
            "mfe_pct": self.mfe_pct,
            "mae_pct": self.mae_pct,
            "ret_pct_brut": self.ret_pct_brut,
            "ret_pct_net": self.ret_pct_net,
            "n_bougies_held": self.n_bougies_held,
        }


@dataclass(frozen=True)
class MesuresBrutes:
    """Sortie du moteur d'évaluation (S-2). Snapshot complet sur un jeu temporel."""

    n_signals: int
    n_tp_hit: int
    n_sl_hit: int
    n_timeout: int

    mfe_mean_pct: float
    mae_mean_pct: float
    mfe_p50: float
    mfe_p75: float
    mfe_p90: float
    mae_p50: float
    mae_p75: float
    mae_p90: float

    ret_mean_pct_net: float         # rendement moyen net
    ret_std_pct: float              # écart-type des rendements nets
    sharpe_net: float               # ret_mean / ret_std * sqrt(periods_per_year)

    tp_hit_rate: float
    sl_hit_rate: float
    timeout_rate: float

    avg_holding_period: float
    avg_time_to_amplitude: float    # bougies moyennes pour TP (sur les tp_hit)

    # Bootstrap CI (engine/bootstrap.py)
    bootstrap_sharpe_ci_low: float
    bootstrap_sharpe_ci_high: float
    bootstrap_ret_ci_low: float
    bootstrap_ret_ci_high: float

    # Per-asset stats
    per_asset_stats: dict[str, MesuresBrutes] = field(default_factory=dict)

    # Trades détaillés (optionnel, allourdit l'objet — désactivable en config)
    trades: tuple[TradeMesure, ...] = field(default_factory=tuple)

    # Contexte d'évaluation (pour traçabilité Archive)
    n_window: int = 0               # N figée depuis train
    sl_price: float = 0.0           # SL figé depuis train
    tp_price: float = 0.0           # TP figé depuis train
    costs_applied: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = {
            "n_signals": self.n_signals,
            "n_tp_hit": self.n_tp_hit,
            "n_sl_hit": self.n_sl_hit,
            "n_timeout": self.n_timeout,
            "mfe_mean_pct": self.mfe_mean_pct,
            "mae_mean_pct": self.mae_mean_pct,
            "mfe_p50": self.mfe_p50, "mfe_p75": self.mfe_p75, "mfe_p90": self.mfe_p90,
            "mae_p50": self.mae_p50, "mae_p75": self.mae_p75, "mae_p90": self.mae_p90,
            "ret_mean_pct_net": self.ret_mean_pct_net,
            "ret_std_pct": self.ret_std_pct,
            "sharpe_net": self.sharpe_net,
            "tp_hit_rate": self.tp_hit_rate,
            "sl_hit_rate": self.sl_hit_rate,
            "timeout_rate": self.timeout_rate,
            "avg_holding_period": self.avg_holding_period,
            "avg_time_to_amplitude": self.avg_time_to_amplitude,
            "bootstrap_sharpe_ci_low": self.bootstrap_sharpe_ci_low,
            "bootstrap_sharpe_ci_high": self.bootstrap_sharpe_ci_high,
            "bootstrap_ret_ci_low": self.bootstrap_ret_ci_low,
            "bootstrap_ret_ci_high": self.bootstrap_ret_ci_high,
            "per_asset_stats": {k: v.to_dict() for k, v in self.per_asset_stats.items()},
            "n_window": self.n_window,
            "sl_price": self.sl_price,
            "tp_price": self.tp_price,
            "costs_applied": self.costs_applied,
        }
        return d


# --------------------------------------------------------------------------- #
# Einher (Hypothesis validée et figée)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Einher:
    """Hypothèse validée, complète, exécutable. C'est l'unité de production de PnL."""

    id: str
    hypothesis_id: str              # référence à l'hypothèse d'origine
    condition_tree: Condition | ConditionNode
    direction: Direction
    universe: Universe
    amplitude: Amplitude
    sl_price: float                 # figé depuis train
    tp_price: float                 # figé depuis train
    n_window: int                   # N figée depuis train
    fingerprint_structurel: str
    fingerprint_comportemental: str

    # Métriques de validation (publiques, sur le val)
    metrics_val: MesuresBrutes
    sharpe_val: float
    bootstrap_sharpe_ci_low_val: float
    bootstrap_sharpe_ci_high_val: float

    # DSR, PBO
    deflated_sharpe_ratio: float
    probability_of_backtest_overfitting: float

    # Identifiants de version
    data_version: str
    seed: int
    splits_hash: str                # hash des bornes train/val/holdout
    admission_timestamp: str        # ISO 8601 UTC

    # Statut
    statut: str = "validé"          # validé | actif | dégradé | archivé

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "hypothesis_id": self.hypothesis_id,
            "condition_tree": self.condition_tree.to_dict() if isinstance(self.condition_tree, ConditionNode) else self.condition_tree.to_dict(),
            "direction": self.direction.value,
            "universe": self.universe.to_dict(),
            "amplitude": self.amplitude.to_dict(),
            "sl_price": self.sl_price,
            "tp_price": self.tp_price,
            "n_window": self.n_window,
            "fingerprint_structurel": self.fingerprint_structurel,
            "fingerprint_comportemental": self.fingerprint_comportemental,
            "metrics_val": self.metrics_val.to_dict(),
            "sharpe_val": self.sharpe_val,
            "bootstrap_sharpe_ci_low_val": self.bootstrap_sharpe_ci_low_val,
            "bootstrap_sharpe_ci_high_val": self.bootstrap_sharpe_ci_high_val,
            "deflated_sharpe_ratio": self.deflated_sharpe_ratio,
            "probability_of_backtest_overfitting": self.probability_of_backtest_overfitting,
            "data_version": self.data_version,
            "seed": self.seed,
            "splits_hash": self.splits_hash,
            "admission_timestamp": self.admission_timestamp,
            "statut": self.statut,
        }
