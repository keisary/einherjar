# portfolio/selector.py
"""
==========================================================
Portfolio Selector
==========================================================

Sélectionne les Einhers candidats à intégrer au portefeuille
final à partir des résultats d'exécution.

Le selector ne calcule pas les poids :
- il filtre,
- il classe,
- il élimine les doublons,
- il prépare l'entrée du portfolio.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from math import tanh
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from execution.execution_report import ExecutionResult
from execution.mae_mfe import MAEMFESummary

try:  # optional config module
    from config.portfolio import PortfolioConfig  # type: ignore
except Exception:  # pragma: no cover
    PortfolioConfig = Any  # type: ignore[misc,assignment]

__all__ = [
    "PortfolioSelectorSettings",
    "PortfolioSelectionEntry",
    "PortfolioSelection",
    "PortfolioSelector",
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


def _safe_mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=float)))


def _safe_std(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(np.std(np.asarray(values, dtype=float)))


def _bounded_unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _bounded_profit_factor(value: float) -> float:
    if not np.isfinite(value):
        return 1.0 if value > 0 else 0.0
    if value <= 0:
        return 0.0
    return float(1.0 - np.exp(-max(0.0, value - 1.0)))


def _normalize_result(obj: Any) -> Any:
    """
    Si l'objet est un Einher avec un execution_result, retourne
    l'execution_result. Sinon retourne l'objet tel quel.
    """
    execution_result = getattr(obj, "execution_result", None)
    if execution_result is not None:
        return execution_result
    return obj


def _family_key(result: Any) -> str:
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

    conditions = getattr(hypothesis, "conditions", None)
    if conditions:
        try:
            family = conditions[0].left.economic_family.value
            if family:
                return str(family).strip().lower()
        except Exception:
            pass

    return "unknown"


def _profile_name(result: Any) -> str:
    profile = getattr(result, "profile", None)
    if profile is not None and getattr(profile, "name", None):
        name = str(profile.name).strip().lower()
        if name:
            return name

    metadata = _to_mapping(result.metadata)
    for key in ("profile_name", "strategy_name", "einher_name"):
        if key in metadata and metadata[key] is not None:
            name = str(metadata[key]).strip().lower()
            if name:
                return name

    return "unknown"


def _subject_fingerprint(result: Any) -> str:
    value = getattr(result, "subject_fingerprint", None)
    if value:
        return str(value)
    # Fallback for Einher
    candidate = getattr(result, "candidate", None)
    if candidate is not None:
        fp = getattr(candidate, "fingerprint", None)
        if fp:
            return str(fp)
    execution_fp = getattr(result, "execution_fingerprint", None)
    if execution_fp is not None:
        digest = getattr(execution_fp, "digest", None)
        if digest:
            return str(digest)
    return ""


def _execution_fingerprint(result: Any) -> str:
    fp = getattr(result, "execution_fingerprint", None)
    if fp is None:
        return _subject_fingerprint(result)
    digest = getattr(fp, "digest", None)
    return str(digest or fp)


def _default_series(result: Any) -> np.ndarray:
    records = getattr(result, "records", ())
    if not records:
        # Fallback for Einher: build from journal trades
        journal = getattr(result, "journal", None)
        if journal is not None:
            trades = getattr(journal, "trades", ())
            values = [float(getattr(trade, "pnl", 0.0)) for trade in trades]
            if values:
                return np.asarray(values, dtype=float)
    values = [float(getattr(record.trade, "pnl", 0.0)) for record in records]
    if not values:
        replay = getattr(result, "replay", None)
        if replay is not None:
            metrics = replay.metrics
            values = [
                float(metrics.total_pnl),
                float(metrics.expectancy),
                float(metrics.win_rate),
                float(metrics.profit_factor if np.isfinite(metrics.profit_factor) else 0.0),
                float(-metrics.max_drawdown),
                float(metrics.trade_count),
                float(metrics.average_duration_bars),
                float(metrics.signal_coverage),
            ]
    return np.asarray(values, dtype=float)


def _diagnostic_penalty(result: Any) -> float:
    diagnostics = getattr(result, "diagnostics", None)
    if diagnostics is None:
        return 0.0

    issues = getattr(diagnostics, "issues", ())
    if not issues:
        return 0.0

    penalty_map = {
        "info": 0.01,
        "warning": 0.05,
        "error": 0.15,
    }
    penalty = 0.0
    for issue in issues:
        severity = str(getattr(issue, "severity", "warning")).strip().lower()
        penalty += penalty_map.get(severity, 0.05)
    return min(1.0, penalty)


def _normalized_series(values: np.ndarray, *, size: int = 64) -> np.ndarray:
    if values.size == 0:
        return np.zeros(size, dtype=float)

    values = values.astype(float, copy=False).reshape(-1)
    if values.size == 1:
        series = np.full(size, float(values[0]), dtype=float)
    else:
        x_src = np.linspace(0.0, 1.0, values.size)
        x_dst = np.linspace(0.0, 1.0, size)
        series = np.interp(x_dst, x_src, values)

    mean = float(np.mean(series))
    std = float(np.std(series))
    if std <= 1e-12:
        return series - mean
    return (series - mean) / std


@dataclass(frozen=True, slots=True)
class PortfolioSelectorSettings:
    """
    Paramètres du selector.
    """

    max_selected: int = 12
    min_trade_count: int = 3

    min_win_rate: float = 0.40
    min_profit_factor: float = 1.0
    min_expectancy: float = 0.0
    min_total_pnl: float = 0.0

    require_healthy: bool = False
    exclude_duplicates: bool = True
    keep_best_per_family: int = 3

    max_same_profile: int = 2
    max_same_family: int = 4

    weight_total_pnl: float = 0.28
    weight_expectancy: float = 0.18
    weight_win_rate: float = 0.18
    weight_profit_factor: float = 0.14
    weight_health: float = 0.10
    weight_drawdown: float = 0.08
    weight_mae_mfe: float = 0.04
    weight_coverage: float = 0.04

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_selected", max(1, _coerce_int(self.max_selected, 12)))
        object.__setattr__(self, "min_trade_count", max(1, _coerce_int(self.min_trade_count, 3)))
        object.__setattr__(self, "min_win_rate", _bounded_unit(self.min_win_rate))
        object.__setattr__(self, "min_profit_factor", max(0.0, _coerce_float(self.min_profit_factor, 1.0)))
        object.__setattr__(self, "min_expectancy", _coerce_float(self.min_expectancy, 0.0))
        object.__setattr__(self, "min_total_pnl", _coerce_float(self.min_total_pnl, 0.0))
        object.__setattr__(self, "require_healthy", _coerce_bool(self.require_healthy, False))
        object.__setattr__(self, "exclude_duplicates", _coerce_bool(self.exclude_duplicates, True))
        object.__setattr__(self, "keep_best_per_family", max(1, _coerce_int(self.keep_best_per_family, 3)))
        object.__setattr__(self, "max_same_profile", max(1, _coerce_int(self.max_same_profile, 2)))
        object.__setattr__(self, "max_same_family", max(1, _coerce_int(self.max_same_family, 4)))

        weights = {
            "pnl": max(0.0, _coerce_float(self.weight_total_pnl, 0.28)),
            "expectancy": max(0.0, _coerce_float(self.weight_expectancy, 0.18)),
            "win_rate": max(0.0, _coerce_float(self.weight_win_rate, 0.18)),
            "profit_factor": max(0.0, _coerce_float(self.weight_profit_factor, 0.14)),
            "health": max(0.0, _coerce_float(self.weight_health, 0.10)),
            "drawdown": max(0.0, _coerce_float(self.weight_drawdown, 0.08)),
            "mae_mfe": max(0.0, _coerce_float(self.weight_mae_mfe, 0.04)),
            "coverage": max(0.0, _coerce_float(self.weight_coverage, 0.04)),
        }
        total = sum(weights.values())
        if total <= 0:
            weights = {
                "pnl": 0.28,
                "expectancy": 0.18,
                "win_rate": 0.18,
                "profit_factor": 0.14,
                "health": 0.10,
                "drawdown": 0.08,
                "mae_mfe": 0.04,
                "coverage": 0.04,
            }

        object.__setattr__(self, "weight_total_pnl", weights["pnl"])
        object.__setattr__(self, "weight_expectancy", weights["expectancy"])
        object.__setattr__(self, "weight_win_rate", weights["win_rate"])
        object.__setattr__(self, "weight_profit_factor", weights["profit_factor"])
        object.__setattr__(self, "weight_health", weights["health"])
        object.__setattr__(self, "weight_drawdown", weights["drawdown"])
        object.__setattr__(self, "weight_mae_mfe", weights["mae_mfe"])
        object.__setattr__(self, "weight_coverage", weights["coverage"])

    @classmethod
    def from_config(cls, config: Any | None) -> "PortfolioSelectorSettings":
        if config is None:
            return cls()

        root = _to_mapping(config)
        port = _to_mapping(root.get("portfolio", root.get("portfolio_config", {})))

        return cls(
            max_selected=_coerce_int(port.get("max_selected", root.get("max_selected", 12)), 12),
            min_trade_count=_coerce_int(port.get("min_trade_count", root.get("min_trade_count", 3)), 3),
            min_win_rate=_coerce_float(port.get("min_win_rate", root.get("min_win_rate", 0.40)), 0.40),
            min_profit_factor=_coerce_float(port.get("min_profit_factor", root.get("min_profit_factor", 1.0)), 1.0),
            min_expectancy=_coerce_float(port.get("min_expectancy", root.get("min_expectancy", 0.0)), 0.0),
            min_total_pnl=_coerce_float(port.get("min_total_pnl", root.get("min_total_pnl", 0.0)), 0.0),
            require_healthy=_coerce_bool(port.get("require_healthy", root.get("require_healthy", False)), False),
            exclude_duplicates=_coerce_bool(port.get("exclude_duplicates", root.get("exclude_duplicates", True)), True),
            keep_best_per_family=_coerce_int(port.get("keep_best_per_family", root.get("keep_best_per_family", 3)), 3),
            max_same_profile=_coerce_int(port.get("max_same_profile", root.get("max_same_profile", 2)), 2),
            max_same_family=_coerce_int(port.get("max_same_family", root.get("max_same_family", 4)), 4),
            weight_total_pnl=_coerce_float(port.get("weight_total_pnl", 0.28), 0.28),
            weight_expectancy=_coerce_float(port.get("weight_expectancy", 0.18), 0.18),
            weight_win_rate=_coerce_float(port.get("weight_win_rate", 0.18), 0.18),
            weight_profit_factor=_coerce_float(port.get("weight_profit_factor", 0.14), 0.14),
            weight_health=_coerce_float(port.get("weight_health", 0.10), 0.10),
            weight_drawdown=_coerce_float(port.get("weight_drawdown", 0.08), 0.08),
            weight_mae_mfe=_coerce_float(port.get("weight_mae_mfe", 0.04), 0.04),
            weight_coverage=_coerce_float(port.get("weight_coverage", 0.04), 0.04),
        )


@dataclass(frozen=True, slots=True)
class PortfolioSelectionEntry:
    """
    Entrée de sélection de portefeuille.
    """

    result: Any
    score: float
    accepted: bool = True
    reasons: tuple[str, ...] = ()
    family: str = "unknown"
    profile_name: str = "unknown"
    rank_hint: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "score", float(self.score))
        object.__setattr__(self, "accepted", _coerce_bool(self.accepted, True))
        object.__setattr__(self, "reasons", tuple(str(item) for item in self.reasons))
        object.__setattr__(self, "family", str(self.family).strip().lower() or "unknown")
        object.__setattr__(self, "profile_name", str(self.profile_name).strip().lower() or "unknown")
        object.__setattr__(self, "rank_hint", max(0, _coerce_int(self.rank_hint, 0)))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def subject_fingerprint(self) -> str:
        return _subject_fingerprint(self.result)

    @property
    def execution_fingerprint(self) -> str:
        return _execution_fingerprint(self.result)

    @property
    def trade_count(self) -> int:
        return int(self.result.replay.metrics.trade_count)

    @property
    def total_pnl(self) -> float:
        return float(self.result.replay.metrics.total_pnl)

    @property
    def win_rate(self) -> float:
        return float(self.result.replay.metrics.win_rate)

    @property
    def profit_factor(self) -> float:
        return float(self.result.replay.metrics.profit_factor)

    def to_dict(self, *, summary_only: bool = False) -> dict[str, Any]:
        """
        Sérialise une PortfolioSelectionEntry.

        En mode summary_only=True, le champ `metadata` (qui
        peut contenir le mae_mfe complet, ~40 MB par entry)
        est omis. Le `result` est sérialisé en summary_only.
        """
        payload = {
            "subject_fingerprint": self.subject_fingerprint,
            "execution_fingerprint": self.execution_fingerprint,
            "score": self.score,
            "accepted": self.accepted,
            "reasons": list(self.reasons),
            "family": self.family,
            "profile_name": self.profile_name,
            "rank_hint": self.rank_hint,
            "result": self.result.to_dict(summary_only=True),
        }
        if not summary_only:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True, slots=True)
class PortfolioSelection:
    """
    Résultat de la sélection portefeuille.
    """

    selected: tuple[PortfolioSelectionEntry, ...]
    rejected: tuple[PortfolioSelectionEntry, ...]
    settings: PortfolioSelectorSettings
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "selected", tuple(self.selected))
        object.__setattr__(self, "rejected", tuple(self.rejected))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def results(self) -> tuple[Any, ...]:
        return tuple(entry.result for entry in self.selected)

    @property
    def selected_count(self) -> int:
        return len(self.selected)

    @property
    def rejected_count(self) -> int:
        return len(self.rejected)

    @property
    def families(self) -> tuple[str, ...]:
        return tuple(sorted({entry.family for entry in self.selected}))

    @property
    def profile_names(self) -> tuple[str, ...]:
        return tuple(sorted({entry.profile_name for entry in self.selected}))

    def to_dict(self, *, summary_only: bool = False) -> dict[str, Any]:
        """
        Sérialise la sélection.

        En mode summary_only=True, chaque entry (selected et
        rejected) est sérialisée en mode summary pour éviter
        le dump du mae_mfe complet.
        """
        return {
            "selected": [
                entry.to_dict(summary_only=summary_only)
                for entry in self.selected
            ],
            "rejected": [
                entry.to_dict(summary_only=summary_only)
                for entry in self.rejected
            ],
            "settings": {
                "max_selected": self.settings.max_selected,
                "min_trade_count": self.settings.min_trade_count,
                "min_win_rate": self.settings.min_win_rate,
                "min_profit_factor": self.settings.min_profit_factor,
                "min_expectancy": self.settings.min_expectancy,
                "min_total_pnl": self.settings.min_total_pnl,
                "require_healthy": self.settings.require_healthy,
                "exclude_duplicates": self.settings.exclude_duplicates,
                "keep_best_per_family": self.settings.keep_best_per_family,
                "max_same_profile": self.settings.max_same_profile,
                "max_same_family": self.settings.max_same_family,
            },
            "metadata": dict(self.metadata),
        }

    def __iter__(self):
        return iter(self.selected)

    def __len__(self) -> int:
        return len(self.selected)


class PortfolioSelector:
    """
    Filtre et classe les Einhers candidats au portefeuille.
    """

    def __init__(
        self,
        settings: PortfolioSelectorSettings | None = None,
        *,
        config: PortfolioConfig | Any | None = None,
        knowledge: Any | None = None,
    ) -> None:
        self._settings = settings or PortfolioSelectorSettings.from_config(config)
        self._knowledge = knowledge
        self._history: list[PortfolioSelectionEntry] = []

    @property
    def settings(self) -> PortfolioSelectorSettings:
        return self._settings

    @property
    def knowledge(self) -> Any | None:
        return self._knowledge

    @property
    def history(self) -> tuple[PortfolioSelectionEntry, ...]:
        return tuple(self._history)

    def select(
        self,
        results: Iterable[Any],
        *,
        limit: int | None = None,
        families: Iterable[str] | None = None,
        profiles: Iterable[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> PortfolioSelection:
        settings = self._settings
        limit = settings.max_selected if limit is None else max(1, _coerce_int(limit, settings.max_selected))
        allowed_families = {str(item).strip().lower() for item in families} if families is not None else None
        allowed_profiles = {str(item).strip().lower() for item in profiles} if profiles is not None else None

        candidates: list[PortfolioSelectionEntry] = []
        rejected: list[PortfolioSelectionEntry] = []
        seen_subjects: set[str] = set()
        seen_executions: set[str] = set()

        per_family_counts: Counter[str] = Counter()
        per_profile_counts: Counter[str] = Counter()

        for rank, result in enumerate(results):
            # PORT-005 : normaliser si on reçoit un Einher
            result = _normalize_result(result)

            family = _family_key(result)
            profile_name = _profile_name(result)

            score, reasons = self._score_result(result)

            if allowed_families is not None and family not in allowed_families:
                reasons = reasons + ("family_not_allowed",)
            if allowed_profiles is not None and profile_name not in allowed_profiles:
                reasons = reasons + ("profile_not_allowed",)

            subject_fp = _subject_fingerprint(result)
            execution_fp = _execution_fingerprint(result)

            if settings.exclude_duplicates:
                if subject_fp and subject_fp in seen_subjects:
                    reasons = reasons + ("duplicate_subject",)
                if execution_fp and execution_fp in seen_executions:
                    reasons = reasons + ("duplicate_execution",)

            if result.replay.metrics.trade_count < settings.min_trade_count:
                reasons = reasons + ("too_few_trades",)

            if result.replay.metrics.win_rate < settings.min_win_rate:
                reasons = reasons + ("win_rate_too_low",)

            if np.isfinite(result.replay.metrics.profit_factor) and result.replay.metrics.profit_factor < settings.min_profit_factor:
                reasons = reasons + ("profit_factor_too_low",)

            if result.replay.metrics.expectancy < settings.min_expectancy:
                reasons = reasons + ("expectancy_too_low",)

            if result.replay.metrics.total_pnl < settings.min_total_pnl:
                reasons = reasons + ("total_pnl_too_low",)

            diagnostics = getattr(result, "diagnostics", None)
            if settings.require_healthy and diagnostics is not None and not bool(getattr(diagnostics, "healthy", True)):
                reasons = reasons + ("unhealthy_execution",)

            if per_family_counts[family] >= settings.keep_best_per_family:
                reasons = reasons + ("family_cap_reached",)

            if per_profile_counts[profile_name] >= settings.max_same_profile:
                reasons = reasons + ("profile_cap_reached",)

            accepted = len(reasons) == 0

            entry = PortfolioSelectionEntry(
                result=result,
                score=score,
                accepted=accepted,
                reasons=tuple(dict.fromkeys(reasons)),
                family=family,
                profile_name=profile_name,
                rank_hint=rank,
                metadata={
                    "subject_fingerprint": subject_fp,
                    "execution_fingerprint": execution_fp,
                    "diagnostic_penalty": _diagnostic_penalty(result),
                    "mae_mfe": None if result.mae_mfe is None else result.mae_mfe.to_dict(),
                    "profile": None if result.profile is None else result.profile.to_dict(),
                },
            )

            if accepted:
                if subject_fp:
                    seen_subjects.add(subject_fp)
                if execution_fp:
                    seen_executions.add(execution_fp)
                per_family_counts[family] += 1
                per_profile_counts[profile_name] += 1
                candidates.append(entry)
            else:
                rejected.append(entry)

            self._history.append(entry)

        selected = tuple(
            sorted(
                candidates,
                key=lambda item: (
                    -item.score,
                    -item.total_pnl,
                    -item.win_rate,
                    item.rank_hint,
                    item.subject_fingerprint,
                ),
            )[:limit]
        )

        rejected.extend(
            PortfolioSelectionEntry(
                result=entry.result,
                score=entry.score,
                accepted=False,
                reasons=entry.reasons + ("not_in_top_selection",),
                family=entry.family,
                profile_name=entry.profile_name,
                rank_hint=entry.rank_hint,
                metadata=entry.metadata,
            )
            for entry in candidates
            if entry not in selected
        )

        selection = PortfolioSelection(
            selected=selected,
            rejected=tuple(rejected),
            settings=settings,
            metadata=dict(metadata or {}),
        )
        return selection

    def score(self, result: Any) -> float:
        score, _ = self._score_result(result)
        return score

    def _score_result(self, result: Any) -> tuple[float, tuple[str, ...]]:
        metrics = result.replay.metrics
        profile = getattr(result, "profile", None)
        mae_mfe = getattr(result, "mae_mfe", None)
        diagnostics = getattr(result, "diagnostics", None)

        reasons: list[str] = []

        total_pnl_component = 0.5 * (1.0 + np.tanh(metrics.total_pnl / max(1.0, abs(metrics.total_pnl) + metrics.trade_count)))
        expectancy_component = 0.5 * (1.0 + np.tanh(metrics.expectancy))
        win_rate_component = _bounded_unit(metrics.win_rate)
        profit_factor_component = _bounded_profit_factor(metrics.profit_factor)
        health_component = 1.0 if diagnostics is None or bool(getattr(diagnostics, "healthy", True)) else 0.0
        drawdown_component = 1.0
        coverage_component = _bounded_unit(metrics.signal_coverage)

        if profile is not None:
            drawdown_component = float(1.0 / (1.0 + max(0.0, float(getattr(profile, "max_drawdown", 0.0)))))

        mae_mfe_component = 0.0
        if mae_mfe is not None:
            ratio = float(getattr(mae_mfe, "avg_mfe_to_mae_ratio", 0.0))
            mae_mfe_component = _bounded_unit(1.0 - np.exp(-max(0.0, ratio)))

        if metrics.trade_count < self._settings.min_trade_count:
            reasons.append("too_few_trades")
        if metrics.win_rate < self._settings.min_win_rate:
            reasons.append("win_rate_too_low")
        if np.isfinite(metrics.profit_factor) and metrics.profit_factor < self._settings.min_profit_factor:
            reasons.append("profit_factor_too_low")
        if metrics.expectancy < self._settings.min_expectancy:
            reasons.append("expectancy_too_low")
        if metrics.total_pnl < self._settings.min_total_pnl:
            reasons.append("total_pnl_too_low")
        if self._settings.require_healthy and diagnostics is not None and not bool(getattr(diagnostics, "healthy", True)):
            reasons.append("unhealthy_execution")

        score = (
            self._settings.weight_total_pnl * total_pnl_component
            + self._settings.weight_expectancy * expectancy_component
            + self._settings.weight_win_rate * win_rate_component
            + self._settings.weight_profit_factor * profit_factor_component
            + self._settings.weight_health * health_component
            + self._settings.weight_drawdown * drawdown_component
            + self._settings.weight_mae_mfe * mae_mfe_component
            + self._settings.weight_coverage * coverage_component
        )

        if diagnostics is not None:
            score *= 1.0 - min(0.40, _diagnostic_penalty(result))

        if profile is not None and getattr(profile, "recovery_factor", 0.0) not in (None, 0):
            recovery = float(getattr(profile, "recovery_factor", 0.0))
            if np.isfinite(recovery):
                score *= 0.85 + 0.15 * _bounded_unit(np.tanh(recovery))

        if self._knowledge is not None:
            known = getattr(self._knowledge, "get", None)
            if callable(known):
                existing = known(_subject_fingerprint(result))
                if existing is not None:
                    score *= 0.98

        return max(0.0, min(1.0, float(score))), tuple(dict.fromkeys(reasons))

    def __repr__(self) -> str:
        return (
            "PortfolioSelector("
            f"max_selected={self._settings.max_selected}, "
            f"min_trade_count={self._settings.min_trade_count}"
            ")"
        )
