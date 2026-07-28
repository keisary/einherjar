# exporters/corpus.py
"""
==========================================================
Corpus Export
==========================================================

Représentation canonique du corpus final d'Einhers.

Ce module ne sélectionne rien :
- il reçoit un portefeuille final,
- il le normalise,
- il le prépare pour les exports.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from execution.execution_report import ExecutionResult
from portfolio.allocator import PortfolioAllocation
from portfolio.allocator import PortfolioAllocationEntry
from portfolio.portfolio_report import PortfolioReport
from portfolio.portfolio_report import PortfolioReportEntry
from portfolio.selector import PortfolioSelection
from portfolio.selector import PortfolioSelectionEntry

__all__ = [
    "CorpusEntry",
    "CorpusSummary",
    "Corpus",
    "CorpusBuilder",
]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


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


def _safe_max(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(np.max(np.asarray(values, dtype=float)))


def _safe_min(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(np.min(np.asarray(values, dtype=float)))


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


def _einher_fingerprint_from_result(result: ExecutionResult) -> str:
    """
    Calcule le fingerprint canonique d'un Einher partir d'un
    ExecutionResult. Utilise `fingerprint_model` si possible
    (le result contient un Einher), sinon fallback sur le
    subject_fingerprint.
    """
    try:
        from models.einher import Einher
        from models.fingerprint import fingerprint_model
        # Si l'ExecutionResult a un ExecutionResult dans
        # .execution_result (cas d'un Einher), utiliser
        # fingerprint_model pour avoir le hash canonique.
        if isinstance(result, Einher):
            return fingerprint_model(result)
    except Exception:
        pass
    return _result_key(result)


def _extract_conditions(result: ExecutionResult) -> tuple[dict[str, Any], ...]:
    """
    Extrait les conditions de l'hypothèse d'un
    ExecutionResult sous forme de tuples de dicts
    sérialisables.
    """
    conditions_list: list[dict[str, Any]] = []
    candidate = getattr(result, "candidate", None)
    hypothesis = getattr(candidate, "hypothesis", None) if candidate is not None else None
    if hypothesis is None:
        hypothesis = getattr(result, "hypothesis", None)

    if hypothesis is None or not hasattr(hypothesis, "conditions"):
        return ()

    for cond in hypothesis.conditions:
        try:
            left = getattr(cond, "left", None)
            op = getattr(cond, "operator", None)
            right = getattr(cond, "right", None)

            op_str = (
                str(op.value) if hasattr(op, "value") else str(op)
            )

            # Left side : Feature
            left_dict: dict[str, Any] = {}
            if left is not None and hasattr(left, "name"):
                left_dict = {
                    "name": str(getattr(left, "name", "")),
                    "column_index": int(
                        getattr(left, "column_index", -1)
                    ),
                    "family": str(
                        getattr(
                            getattr(left, "economic_family", None),
                            "value", "unknown",
                        )
                    ),
                }

            # Right side : constant or Feature
            right_dict: dict[str, Any] = {}
            if hasattr(right, "name"):
                # Feature right
                right_dict = {
                    "type": "feature",
                    "name": str(getattr(right, "name", "")),
                    "column_index": int(
                        getattr(right, "column_index", -1)
                    ),
                }
            else:
                # Constant right
                right_dict = {
                    "type": "constant",
                    "value": repr(right),
                }

            conditions_list.append({
                "left": left_dict,
                "operator": op_str,
                "right": right_dict,
            })
        except Exception:
            # En cas d'erreur, on ajoute un placeholder
            conditions_list.append({
                "left": {"name": "unknown", "column_index": -1, "family": "unknown"},
                "operator": "?",
                "right": {"type": "constant", "value": "?"},
            })

    return tuple(conditions_list)


def _build_edge_dict(result: ExecutionResult) -> dict[str, Any]:
    """
    Construit le dict d'edge metrics (PLAN 2.1) à partir d'un
    ExecutionResult. Contient les agrégats d'exécution (pas le
    détail des trades).
    """
    edge: dict[str, Any] = {}
    metrics_obj = None
    if hasattr(result, "replay") and result.replay is not None:
        metrics_obj = getattr(result.replay, "metrics", None)

    if metrics_obj is not None:
        edge["score"] = float(getattr(metrics_obj, "score", 0.0) or 0.0)
        edge["win_rate"] = float(getattr(metrics_obj, "win_rate", 0.0) or 0.0)
        edge["profit_factor"] = float(
            getattr(metrics_obj, "profit_factor", 0.0) or 0.0
        )
        edge["expectancy"] = float(
            getattr(metrics_obj, "expectancy", 0.0) or 0.0
        )
        edge["total_pnl"] = float(
            getattr(metrics_obj, "total_pnl", 0.0) or 0.0
        )
        edge["trade_count"] = int(
            getattr(metrics_obj, "trade_count", 0) or 0
        )
        # Sharpe per trade (approximation)
        pnl_std = float(
            getattr(metrics_obj, "pnl_std", 0.0) or 0.0
        )
        if pnl_std > 0:
            avg_pnl = edge["expectancy"]
            edge["sharpe_per_trade"] = avg_pnl / pnl_std
        else:
            edge["sharpe_per_trade"] = 0.0
        # p_value : non calculé ici (le validator le calcule).
        # On le récupère depuis les assessment metrics si dispo.
        try:
            vc = getattr(result, "validated_candidate", None)
            if vc is not None:
                vm = vc.metrics.get("validation", {})
                edge["p_value"] = float(vm.get("p_value", 0.0) or 0.0)
        except Exception:
            edge["p_value"] = 0.0

    return edge


def _build_calibration_dict(
    result: ExecutionResult,
    *,
    direction: str = "long",
) -> dict[str, Any]:
    """
    Construit le dict de calibration TP/SL (PLAN 2.3) à
    partir d'un ExecutionResult.

    Inclut :
    - mfe_p50/p75/p90 et mae_p50/p75/p90 (depuis MAEMFESummary)
    - best_horizon (1 par défaut, TODO: depuis discovery)
    - tp_rule et sl_rule calibrés sur les percentiles 75/90
      du MFE/MAE
    """
    calibration: dict[str, Any] = {
        "direction": direction,
        "best_horizon": 1,  # TODO : récupérer depuis discovery
    }

    mae_mfe = getattr(result, "mae_mfe", None)
    if mae_mfe is not None:
        # p50 / p75 / p90 du MFE/MAE
        calibration["mfe_p50"] = float(getattr(mae_mfe, "median_mfe", 0.0) or 0.0)
        calibration["mfe_p75"] = float(getattr(mae_mfe, "p75_mfe", 0.0) or 0.0)
        calibration["mfe_p90"] = float(getattr(mae_mfe, "p90_mfe", 0.0) or 0.0)
        calibration["mae_p50"] = float(getattr(mae_mfe, "median_mae", 0.0) or 0.0)
        calibration["mae_p75"] = float(getattr(mae_mfe, "p75_mae", 0.0) or 0.0)
        calibration["mae_p90"] = float(getattr(mae_mfe, "p90_mae", 0.0) or 0.0)

        # p75/p90 en % de l'entry price
        calibration["mfe_p75_pct"] = float(
            getattr(mae_mfe, "p75_mfe_pct", 0.0) or 0.0
        )
        calibration["mae_p90_pct"] = float(
            getattr(mae_mfe, "p90_mae_pct", 0.0) or 0.0
        )

        # Règles TP/SL calibrées (PLAN 2.3)
        calibration["tp_rule"] = {
            "type": "mfe_calibrated",
            "percentile": 75,
            "value": calibration["mfe_p75"],
        }
        calibration["sl_rule"] = {
            "type": "mae_calibrated",
            "percentile": 90,
            "value": calibration["mae_p90"],
        }

    return calibration


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


def _profile_name(result: ExecutionResult) -> str:
    profile = getattr(result, "profile", None)
    if profile is not None and getattr(profile, "name", None):
        value = str(profile.name).strip().lower()
        if value:
            return value

    metadata = _to_mapping(result.metadata)
    for key in ("profile_name", "strategy_name", "einher_name"):
        if key in metadata and metadata[key] is not None:
            value = str(metadata[key]).strip().lower()
            if value:
                return value

    return "unknown"


def _entry_from_any(entry: Any) -> tuple[ExecutionResult, float, float, float, str, str, int, dict[str, Any]]:
    """
    Normalise une entrée de portfolio en un tuple canonique.

    Retour :
    - result
    - weight
    - capital
    - score
    - family
    - profile_name
    - rank
    - metadata
    """
    if isinstance(entry, PortfolioReportEntry):
        return (
            entry.result,
            _coerce_float(entry.weight, 0.0),
            _coerce_float(entry.capital, 0.0),
            _coerce_float(entry.score, 0.0),
            str(entry.family),
            str(entry.profile_name),
            _coerce_int(entry.rank, 0),
            _to_mapping(entry.metadata),
        )

    if isinstance(entry, PortfolioAllocationEntry):
        return (
            entry.result,
            _coerce_float(entry.target_weight, 0.0),
            _coerce_float(entry.capital, 0.0),
            _coerce_float(entry.score, 0.0),
            str(entry.family),
            str(entry.profile_name),
            _coerce_int(entry.rank, 0),
            _to_mapping(entry.metadata),
        )

    if isinstance(entry, PortfolioSelectionEntry):
        result = entry.result
        return (
            result,
            _coerce_float(getattr(entry, "target_weight", getattr(entry, "score", 0.0)), 0.0),
            _coerce_float(getattr(entry, "capital", 0.0), 0.0),
            _coerce_float(entry.score, 0.0),
            str(entry.family),
            str(entry.profile_name),
            _coerce_int(entry.rank_hint, 0),
            _to_mapping(entry.metadata),
        )

    if isinstance(entry, ExecutionResult):
        return (
            entry,
            0.0,
            0.0,
            _coerce_float(entry.replay.metrics.total_pnl, 0.0),
            _family_key(entry),
            _profile_name(entry),
            0,
            {},
        )

    raise TypeError(f"Unsupported corpus entry type: {type(entry)!r}")


@dataclass(frozen=True, slots=True)
class CorpusEntry:
    """
    Entrée canonique du corpus d'Einhers.

    Conforme au PLAN_COMPLET_V2.md section 2.3 (découverte) et
    section 2.1 (métriques d'edge).

    Champs :

    Identité
    - subject_fingerprint
    - execution_fingerprint
    - einher_fingerprint (fingerprint canonique de l'Einher)

    Cible (provenance de l'edge)
    - asset
    - timeframe
    - direction (long / short / both)
    - calibrated_on (période de calibration)

    Profil (description)
    - profile: {name, description, family}

    Conditions (hypothèse)
    - conditions: list[{left, op, right}, ...]

    Edge (PLAN section 2.1)
    - edge: {score, win_rate, profit_factor, expectancy,
             total_pnl, trade_count, p_value, sharpe_per_trade}

    Calibration (PLAN section 2.3)
    - calibration: {mfe_p50/p75/p90, mae_p50/p75/p90,
                   best_horizon, tp_rule, sl_rule}

    Statut
    - selected, weight, capital, rank
    - rejection_reasons (tuple vide si sélectionné)

    Le détail des trades (records, journal) n'est PAS dans
    cette entrée. Il vit dans le corpus brut (parquet/csv).
    """

    # Identité
    subject_fingerprint: str
    execution_fingerprint: str
    einher_fingerprint: str = ""

    # Cible
    asset: str = "unknown"
    timeframe: str = "unknown"
    direction: str = "long"
    calibrated_on: str = ""

    # Profil
    profile: dict[str, Any] = field(default_factory=dict)

    # Conditions de l'hypothèse (sérialisées en dict)
    conditions: tuple[dict[str, Any], ...] = ()

    # Métriques d'edge (PLAN 2.1)
    edge: dict[str, Any] = field(default_factory=dict)

    # Calibration TP/SL (PLAN 2.3)
    calibration: dict[str, Any] = field(default_factory=dict)

    # Statut de sélection
    selected: bool = False
    weight: float = 0.0
    capital: float = 0.0
    rank: int = 0
    rejection_reasons: tuple[str, ...] = ()

    # Provenance
    source_kind: str = "portfolio_selection"
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        object.__setattr__(self, "subject_fingerprint", str(self.subject_fingerprint))
        object.__setattr__(self, "execution_fingerprint", str(self.execution_fingerprint))
        object.__setattr__(self, "einher_fingerprint", str(self.einher_fingerprint))
        object.__setattr__(self, "asset", str(self.asset).strip().lower() or "unknown")
        object.__setattr__(self, "timeframe", str(self.timeframe).strip().lower() or "unknown")
        object.__setattr__(self, "direction", str(self.direction).strip().lower() or "long")
        object.__setattr__(self, "calibrated_on", str(self.calibrated_on).strip())
        object.__setattr__(self, "profile", dict(self.profile))
        object.__setattr__(self, "conditions", tuple(dict(c) for c in self.conditions))
        object.__setattr__(self, "edge", dict(self.edge))
        object.__setattr__(self, "calibration", dict(self.calibration))
        object.__setattr__(self, "selected", _coerce_bool(self.selected, False))
        object.__setattr__(self, "weight", max(0.0, float(self.weight)))
        object.__setattr__(self, "capital", max(0.0, float(self.capital)))
        object.__setattr__(self, "rank", max(0, _coerce_int(self.rank, 0)))
        object.__setattr__(
            self, "rejection_reasons",
            tuple(str(r) for r in self.rejection_reasons),
        )
        object.__setattr__(self, "source_kind", str(self.source_kind).strip().lower() or "portfolio_selection")
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def is_final(self) -> bool:
        """Un Einher est final (= retenu par le portfolio) ssi
        selected=True ET capital > 0 ET weight > 0."""
        return self.selected and self.capital > 0 and self.weight > 0

    @property
    def is_rejected(self) -> bool:
        """Un Einher est rejeté si non final (rejeté par le
        selector ou par le risk model)."""
        return not self.is_final

    @property
    def short_fingerprint(self) -> str:
        return self.subject_fingerprint[:12]

    # ==================================================
    # BACKWARD-COMPAT PROPERTIES
    # ==================================================
    # L'ancienne structure exposait ces champs en top-level.
    # Le CorpusSummaryBuilder et les exporters les utilisent.
    # On les dérive depuis la nouvelle structure pour ne pas
    # avoir à réécrire tous les call-sites.

    @property
    def family(self) -> str:
        return str(self.profile.get("family", "unknown"))

    @property
    def profile_name(self) -> str:
        return str(self.profile.get("name", "unknown"))

    @property
    def score(self) -> float:
        return float(self.edge.get("score", 0.0))

    @property
    def win_rate(self) -> float:
        return float(self.edge.get("win_rate", 0.0))

    @property
    def profit_factor(self) -> float:
        return float(self.edge.get("profit_factor", 0.0))

    @property
    def expectancy(self) -> float:
        return float(self.edge.get("expectancy", 0.0))

    @property
    def total_pnl(self) -> float:
        return float(self.edge.get("total_pnl", 0.0))

    @property
    def trade_count(self) -> int:
        return int(self.edge.get("trade_count", 0))

    @property
    def max_drawdown(self) -> float:
        """Pas dans edge ; recalculé depuis la calibration
        MFE/MAE (max_drawdown = max mae - mfe en %)."""
        mae_p90 = float(self.calibration.get("mae_p90", 0.0))
        mfe_p90 = float(self.calibration.get("mfe_p90", 0.0))
        return max(0.0, mae_p90 - mfe_p90)

    @property
    def healthy(self) -> bool:
        """Un Einher est 'healthy' si son PF >= 1.0 (proxy
        simplifié). Le diagnostic complet reste dans
        ExecutionResult.diagnostics pour les consumers qui
        en ont besoin."""
        return self.profit_factor >= 1.0

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "subject_fingerprint": self.subject_fingerprint,
            "execution_fingerprint": self.execution_fingerprint,
            "einher_fingerprint": self.einher_fingerprint,
            "asset": self.asset,
            "timeframe": self.timeframe,
            "direction": self.direction,
            "calibrated_on": self.calibrated_on,
            "profile": dict(self.profile),
            "conditions": list(self.conditions),
            "edge": dict(self.edge),
            "calibration": dict(self.calibration),
            "selected": self.selected,
            "weight": self.weight,
            "capital": self.capital,
            "rank": self.rank,
            "rejection_reasons": list(self.rejection_reasons),
            "source_kind": self.source_kind,
            "metadata": dict(self.metadata),
            "created_at": self.created_at.isoformat(),
        }
        return payload

    @classmethod
    def from_result(
        cls,
        result: ExecutionResult,
        *,
        asset: str = "unknown",
        timeframe: str = "unknown",
        direction: str = "long",
        calibrated_on: str = "",
        selected: bool = False,
        weight: float = 0.0,
        capital: float = 0.0,
        rank: int = 0,
        rejection_reasons: tuple[str, ...] = (),
        source_kind: str = "portfolio_selection",
        metadata: Mapping[str, Any] | None = None,
    ) -> "CorpusEntry":
        """
        Construit un CorpusEntry à partir d'un ExecutionResult.

        La sortie est conforme au plan :
        - pas de mae_mfe.records (40 MB)
        - pas de journal.trades
        - pas de diagnostics verbose
        - seulement les agrégats d'edge et la calibration
        """

        execution_fp = getattr(result, "execution_fingerprint", None)
        execution_fingerprint = (
            getattr(execution_fp, "digest", None) if execution_fp is not None else None
        )
        if not execution_fingerprint:
            execution_fingerprint = _result_key(result)

        # Profile descriptif
        profile_obj = getattr(result, "profile", None)
        profile_dict: dict[str, Any] = {}
        if profile_obj is not None and hasattr(profile_obj, "name"):
            profile_dict = {
                "name": str(getattr(profile_obj, "name", "unknown")),
                "description": str(getattr(profile_obj, "description", "")),
                "family": _family_key(result),
            }

        # Conditions de l'hypothèse
        conditions_list = _extract_conditions(result)

        # Edge metrics (PLAN 2.1)
        edge = _build_edge_dict(result)

        # Calibration (PLAN 2.3) : TP/SL via MFE/MAE percentiles
        calibration = _build_calibration_dict(result, direction=direction)

        return cls(
            subject_fingerprint=_result_key(result),
            execution_fingerprint=str(execution_fingerprint),
            einher_fingerprint=_einher_fingerprint_from_result(result),
            asset=asset,
            timeframe=timeframe,
            direction=direction,
            calibrated_on=calibrated_on,
            profile=profile_dict,
            conditions=conditions_list,
            edge=edge,
            calibration=calibration,
            selected=selected,
            weight=weight,
            capital=capital,
            rank=rank,
            rejection_reasons=rejection_reasons,
            source_kind=source_kind,
            metadata=dict(metadata or {}),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CorpusEntry":
        created_at = data.get("created_at")
        if isinstance(created_at, str) and created_at:
            created_at = datetime.fromisoformat(created_at)
        else:
            created_at = _utc_now()

        return cls(
            subject_fingerprint=data.get("subject_fingerprint", ""),
            execution_fingerprint=data.get("execution_fingerprint", ""),
            family=data.get("family", "unknown"),
            profile_name=data.get("profile_name", "unknown"),
            source_kind=data.get("source_kind", "portfolio"),
            score=_coerce_float(data.get("score"), 0.0),
            weight=_coerce_float(data.get("weight"), 0.0),
            capital=_coerce_float(data.get("capital"), 0.0),
            trade_count=_coerce_int(data.get("trade_count"), 0),
            total_pnl=_coerce_float(data.get("total_pnl"), 0.0),
            win_rate=_coerce_float(data.get("win_rate"), 0.0),
            profit_factor=_coerce_float(data.get("profit_factor"), 0.0),
            expectancy=_coerce_float(data.get("expectancy"), 0.0),
            max_drawdown=_coerce_float(data.get("max_drawdown"), 0.0),
            exposure_ratio=_coerce_float(data.get("exposure_ratio"), 0.0),
            signal_coverage=_coerce_float(data.get("signal_coverage"), 0.0),
            healthy=_coerce_bool(data.get("healthy"), True),
            issue_count=_coerce_int(data.get("issue_count"), 0),
            mae_mfe=_to_mapping(data.get("mae_mfe", {})),
            profile=_to_mapping(data.get("profile", {})),
            diagnostics=_to_mapping(data.get("diagnostics", {})),
            risk=_to_mapping(data.get("risk", {})),
            diversification=_to_mapping(data.get("diversification", {})),
            selection=_to_mapping(data.get("selection", {})),
            allocation=_to_mapping(data.get("allocation", {})),
            metadata=_to_mapping(data.get("metadata", {})),
            rank=_coerce_int(data.get("rank"), 0),
            created_at=created_at,
        )

    def to_record(self) -> dict[str, Any]:
        record = self.to_dict()
        record["is_final"] = self.is_final
        record["short_fingerprint"] = self.short_fingerprint
        return record


@dataclass(frozen=True, slots=True)
class CorpusSummary:
    """
    Résumé global du corpus.
    """

    entry_count: int
    selected_count: int
    total_capital: float
    total_weight: float
    total_pnl: float

    average_score: float
    best_score: float
    best_subject_fingerprint: str | None = None

    average_win_rate: float = 0.0
    average_profit_factor: float = 0.0
    average_expectancy: float = 0.0
    average_drawdown: float = 0.0

    healthy_count: int = 0
    unhealthy_count: int = 0

    unique_family_count: int = 0
    unique_profile_count: int = 0

    family_counts: dict[str, int] = field(default_factory=dict)
    profile_counts: dict[str, int] = field(default_factory=dict)

    min_weight: float = 0.0
    max_weight: float = 0.0
    mean_weight: float = 0.0

    min_capital: float = 0.0
    max_capital: float = 0.0
    mean_capital: float = 0.0

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "entry_count", max(0, _coerce_int(self.entry_count, 0)))
        object.__setattr__(self, "selected_count", max(0, _coerce_int(self.selected_count, 0)))
        object.__setattr__(self, "total_capital", max(0.0, float(self.total_capital)))
        object.__setattr__(self, "total_weight", max(0.0, float(self.total_weight)))
        object.__setattr__(self, "total_pnl", float(self.total_pnl))
        object.__setattr__(self, "average_score", _bounded_unit(self.average_score))
        object.__setattr__(self, "best_score", _bounded_unit(self.best_score))
        object.__setattr__(self, "average_win_rate", _bounded_unit(self.average_win_rate))
        object.__setattr__(self, "average_profit_factor", float(self.average_profit_factor))
        object.__setattr__(self, "average_expectancy", float(self.average_expectancy))
        object.__setattr__(self, "average_drawdown", max(0.0, float(self.average_drawdown)))
        object.__setattr__(self, "healthy_count", max(0, _coerce_int(self.healthy_count, 0)))
        object.__setattr__(self, "unhealthy_count", max(0, _coerce_int(self.unhealthy_count, 0)))
        object.__setattr__(self, "unique_family_count", max(0, _coerce_int(self.unique_family_count, 0)))
        object.__setattr__(self, "unique_profile_count", max(0, _coerce_int(self.unique_profile_count, 0)))
        object.__setattr__(self, "family_counts", dict(self.family_counts))
        object.__setattr__(self, "profile_counts", dict(self.profile_counts))
        object.__setattr__(self, "min_weight", max(0.0, float(self.min_weight)))
        object.__setattr__(self, "max_weight", max(0.0, float(self.max_weight)))
        object.__setattr__(self, "mean_weight", max(0.0, float(self.mean_weight)))
        object.__setattr__(self, "min_capital", max(0.0, float(self.min_capital)))
        object.__setattr__(self, "max_capital", max(0.0, float(self.max_capital)))
        object.__setattr__(self, "mean_capital", max(0.0, float(self.mean_capital)))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_count": self.entry_count,
            "selected_count": self.selected_count,
            "total_capital": self.total_capital,
            "total_weight": self.total_weight,
            "total_pnl": self.total_pnl,
            "average_score": self.average_score,
            "best_score": self.best_score,
            "best_subject_fingerprint": self.best_subject_fingerprint,
            "average_win_rate": self.average_win_rate,
            "average_profit_factor": self.average_profit_factor,
            "average_expectancy": self.average_expectancy,
            "average_drawdown": self.average_drawdown,
            "healthy_count": self.healthy_count,
            "unhealthy_count": self.unhealthy_count,
            "unique_family_count": self.unique_family_count,
            "unique_profile_count": self.unique_profile_count,
            "family_counts": dict(self.family_counts),
            "profile_counts": dict(self.profile_counts),
            "min_weight": self.min_weight,
            "max_weight": self.max_weight,
            "mean_weight": self.mean_weight,
            "min_capital": self.min_capital,
            "max_capital": self.max_capital,
            "mean_capital": self.mean_capital,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class Corpus:
    """
    Corpus final exportable.
    """

    name: str = "corpus"
    entries: list[CorpusEntry] = field(default_factory=list)
    summary: CorpusSummary | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", str(self.name).strip() or "corpus")
        object.__setattr__(self, "entries", list(self.entries))
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "rejected", list(self.rejected))

        if self.summary is None:
            object.__setattr__(self, "summary", CorpusSummaryBuilder.build(self.entries, metadata=self.metadata))

    @property
    def selected_count(self) -> int:
        return sum(1 for entry in self.entries if entry.is_final)

    @property
    def final_entries(self) -> tuple[CorpusEntry, ...]:
        return tuple(entry for entry in self.entries if entry.is_final)

    def to_dict(self, *, summary_only: bool = False) -> dict[str, Any]:
        payload = {
            "name": self.name,
            "created_at": self.created_at.isoformat(),
            "entries": [] if summary_only else [entry.to_dict() for entry in self.entries],
            "rejected": [] if summary_only else [dict(item) for item in self.rejected],
            "summary": self.summary.to_dict() if self.summary is not None else None,
            "metadata": dict(self.metadata),
        }
        return payload

    def to_records(self) -> list[dict[str, Any]]:
        return [entry.to_record() for entry in self.entries]

    def to_rows(self) -> list[dict[str, Any]]:
        return self.to_records()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Corpus":
        summary_data = data.get("summary")
        summary = CorpusSummary.from_dict(summary_data) if isinstance(summary_data, Mapping) else None
        return cls(
            name=data.get("name", "corpus"),
            entries=[CorpusEntry.from_dict(item) for item in data.get("entries", [])],
            summary=summary,
            metadata=_to_mapping(data.get("metadata", {})),
            rejected=[dict(item) for item in data.get("rejected", [])],
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else _utc_now(),
        )

    def __iter__(self):
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    def __repr__(self) -> str:
        return (
            "Corpus("
            f"name='{self.name}', "
            f"entries={len(self.entries)}, "
            f"selected={self.selected_count}"
            ")"
        )


class CorpusSummaryBuilder:
    """
    Construit un résumé de corpus.
    """

    @staticmethod
    def build(entries: Sequence[CorpusEntry], *, metadata: Mapping[str, Any] | None = None) -> CorpusSummary:
        entries = tuple(entries)
        if not entries:
            return CorpusSummary(
                entry_count=0,
                selected_count=0,
                total_capital=0.0,
                total_weight=0.0,
                total_pnl=0.0,
                average_score=0.0,
                best_score=0.0,
                best_subject_fingerprint=None,
                average_win_rate=0.0,
                average_profit_factor=0.0,
                average_expectancy=0.0,
                average_drawdown=0.0,
                healthy_count=0,
                unhealthy_count=0,
                unique_family_count=0,
                unique_profile_count=0,
                family_counts={},
                profile_counts={},
                min_weight=0.0,
                max_weight=0.0,
                mean_weight=0.0,
                min_capital=0.0,
                max_capital=0.0,
                mean_capital=0.0,
                metadata=dict(metadata or {}),
            )

        family_counts = Counter(entry.family for entry in entries)
        profile_counts = Counter(entry.profile_name for entry in entries)

        weights = [entry.weight for entry in entries]
        capitals = [entry.capital for entry in entries]
        scores = [entry.score for entry in entries]
        win_rates = [entry.win_rate for entry in entries]
        pfs = [entry.profit_factor for entry in entries]
        expectancies = [entry.expectancy for entry in entries]
        drawdowns = [entry.max_drawdown for entry in entries]

        selected_entries = [entry for entry in entries if entry.is_final]

        best_entry = max(entries, key=lambda entry: (entry.score, entry.capital, entry.weight, entry.total_pnl))

        return CorpusSummary(
            entry_count=len(entries),
            selected_count=len(selected_entries),
            total_capital=float(sum(capitals)),
            total_weight=float(sum(weights)),
            total_pnl=float(sum(entry.total_pnl for entry in entries)),
            average_score=_safe_mean(scores),
            best_score=best_entry.score,
            best_subject_fingerprint=best_entry.subject_fingerprint,
            average_win_rate=_safe_mean(win_rates),
            average_profit_factor=_safe_mean(pfs),
            average_expectancy=_safe_mean(expectancies),
            average_drawdown=_safe_mean(drawdowns),
            healthy_count=sum(1 for entry in entries if entry.healthy),
            unhealthy_count=sum(1 for entry in entries if not entry.healthy),
            unique_family_count=len(family_counts),
            unique_profile_count=len(profile_counts),
            family_counts=dict(family_counts),
            profile_counts=dict(profile_counts),
            min_weight=_safe_min(weights),
            max_weight=_safe_max(weights),
            mean_weight=_safe_mean(weights),
            min_capital=_safe_min(capitals),
            max_capital=_safe_max(capitals),
            mean_capital=_safe_mean(capitals),
            metadata=dict(metadata or {}),
        )


class CorpusBuilder:
    """
    Construit le corpus final à partir d'un portefeuille.
    """

    @staticmethod
    def from_portfolio_report(
        report: PortfolioReport,
        *,
        include_rejected: bool = True,
        source_kind: str = "portfolio_report",
        metadata: Mapping[str, Any] | None = None,
    ) -> Corpus:
        entries: list[CorpusEntry] = []
        for idx, entry in enumerate(report.entries):
            entries.append(
                CorpusEntry.from_result(
                    entry.result,
                    weight=_coerce_float(entry.weight, 0.0),
                    capital=_coerce_float(entry.capital, 0.0),
                    score=_coerce_float(entry.score, 0.0),
                    source_kind=source_kind,
                    rank=_coerce_int(entry.rank, idx),
                    metadata={
                        **_to_mapping(report.metadata),
                        **_to_mapping(entry.metadata),
                        "selection": entry.to_dict(),
                    },
                )
            )

        summary = CorpusSummaryBuilder.build(entries, metadata={**_to_mapping(report.metadata), **_to_mapping(metadata)})
        rejected = list(report.rejected) if include_rejected else []

        return Corpus(
            name=report.name if getattr(report, "name", None) else "corpus",
            entries=entries,
            summary=summary,
            metadata={**_to_mapping(report.metadata), **_to_mapping(metadata)},
            rejected=rejected,
            created_at=getattr(report, "created_at", _utc_now()),
        )

    @staticmethod
    def from_portfolio_allocation(
        allocation: PortfolioAllocation,
        *,
        include_rejected: bool = True,
        metadata: Mapping[str, Any] | None = None,
    ) -> Corpus:
        entries: list[CorpusEntry] = []
        rejected: list[dict[str, Any]] = []

        for idx, entry in enumerate(allocation.entries):
            if not entry.accepted or entry.capital <= 0:
                rejected.append(
                    {
                        "subject_fingerprint": entry.subject_fingerprint,
                        "family": entry.family,
                        "profile_name": entry.profile_name,
                        "reason": "allocation_not_accepted",
                        "score": entry.score,
                        "weight": entry.target_weight,
                        "capital": entry.capital,
                    }
                )
                continue

            entries.append(
                CorpusEntry.from_result(
                    entry.result,
                    weight=_coerce_float(entry.target_weight, 0.0),
                    capital=_coerce_float(entry.capital, 0.0),
                    score=_coerce_float(entry.score, 0.0),
                    source_kind="portfolio_allocation",
                    rank=idx,
                    metadata={
                        **_to_mapping(entry.metadata),
                        "allocation": entry.to_dict(),
                    },
                )
            )

        summary = CorpusSummaryBuilder.build(entries, metadata={**_to_mapping(metadata)})
        if not include_rejected:
            rejected = []

        return Corpus(
            name="corpus",
            entries=entries,
            summary=summary,
            metadata={**_to_mapping(metadata)},
            rejected=rejected,
        )

    @staticmethod
    def from_portfolio_selection(
        selection: PortfolioSelection,
        *,
        allocation: PortfolioAllocation | None = None,
        include_rejected: bool = True,
        asset: str = "unknown",
        timeframe: str = "unknown",
        calibrated_on: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> Corpus:
        """
        Construit un corpus à partir d'une PortfolioSelection.

        Chaque PortfolioSelectionEntry (selected ou rejected)
        devient un CorpusEntry conforme au PLAN_COMPLET_V2.md
        section 2.3 :
        - selected : capital/weight de l'allocation, is_final=True
        - rejected : capital=0, weight=0, is_final=True
          (= le Einher existe, il n'est juste pas dans le
          portefeuille), rejection_reasons = liste des raisons
        """

        entries: list[CorpusEntry] = []
        rejected_meta: list[dict[str, Any]] = []

        # Index allocation par fingerprint pour récupérer
        # le capital/weight effectif des Einhers retenus.
        alloc_by_fp: dict[str, PortfolioAllocationEntry] = {}
        if allocation is not None:
            for alloc_entry in allocation.entries:
                fp = getattr(alloc_entry, "subject_fingerprint", "")
                if fp:
                    alloc_by_fp[fp] = alloc_entry

        # 1) Einhers retenus par le selector
        for idx, entry in enumerate(selection.selected):
            fp = entry.subject_fingerprint
            alloc_entry = alloc_by_fp.get(fp)
            weight = (
                _coerce_float(alloc_entry.target_weight, 0.0)
                if alloc_entry is not None
                else 0.0
            )
            capital = (
                _coerce_float(alloc_entry.capital, 0.0)
                if alloc_entry is not None
                else 0.0
            )
            entries.append(
                CorpusEntry.from_result(
                    entry.result,
                    asset=asset,
                    timeframe=timeframe,
                    calibrated_on=calibrated_on,
                    selected=True,
                    weight=weight,
                    capital=capital,
                    rank=idx,
                    source_kind="portfolio_selection",
                    metadata={
                        **_to_mapping(metadata or {}),
                        "asset": asset,
                        "timeframe": timeframe,
                    },
                )
            )

        # 2) Einhers rejetés par le selector
        if include_rejected:
            for rejected_entry in selection.rejected:
                entries.append(
                    CorpusEntry.from_result(
                        rejected_entry.result,
                        asset=asset,
                        timeframe=timeframe,
                        calibrated_on=calibrated_on,
                        selected=False,
                        weight=0.0,
                        capital=0.0,
                        rank=0,
                        source_kind="portfolio_selection_rejected",
                        rejection_reasons=tuple(rejected_entry.reasons),
                        metadata={
                            **_to_mapping(metadata or {}),
                            "asset": asset,
                            "timeframe": timeframe,
                        },
                    )
                )
                rejected_meta.append(
                    {
                        "subject_fingerprint": rejected_entry.subject_fingerprint,
                        "family": rejected_entry.family,
                        "profile_name": rejected_entry.profile_name,
                        "reasons": list(rejected_entry.reasons),
                        "score": rejected_entry.score,
                        "accepted": rejected_entry.accepted,
                    }
                )

        summary = CorpusSummaryBuilder.build(entries, metadata={**_to_mapping(metadata)})
        return Corpus(
            name=f"{asset}__{timeframe}" if asset != "unknown" else "corpus",
            entries=entries,
            summary=summary,
            metadata={**_to_mapping(metadata)},
            rejected=rejected_meta,
        )

    @staticmethod
    def from_results(
        results: Iterable[ExecutionResult],
        *,
        include_rejected: bool = False,
        metadata: Mapping[str, Any] | None = None,
    ) -> Corpus:
        entries = [
            CorpusEntry.from_result(
                result,
                weight=0.0,
                capital=0.0,
                score=float(result.replay.metrics.total_pnl),
                source_kind="execution_result",
                rank=index,
                metadata={"result": result.to_dict(summary_only=True)},
            )
            for index, result in enumerate(results)
        ]
        summary = CorpusSummaryBuilder.build(entries, metadata={**_to_mapping(metadata)})
        return Corpus(
            name="corpus",
            entries=entries,
            summary=summary,
            metadata={**_to_mapping(metadata)},
            rejected=[] if not include_rejected else [],
        )