# execution/diagnostics.py
"""
==========================================================
Execution Diagnostics
==========================================================

Détecte les défauts d'un replay et produit un diagnostic
structuré.

Le module ne modifie rien :
- il lit un ReplayResult,
- il compare les métriques à des seuils,
- il agrège les problèmes observés.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Iterable
from typing import Mapping
from typing import Sequence

import numpy as np

from .mae_mfe import MAEMFESummary
from .profiler import ExecutionProfile
from .replay import ReplayResult

__all__ = [
    "DiagnosticSettings",
    "DiagnosticIssue",
    "ExecutionDiagnostics",
    "ExecutionDiagnoser",
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


def _safe_div(numerator: float, denominator: float) -> float:
    if abs(denominator) <= 1e-12:
        return 0.0
    return float(numerator / denominator)


def _normalize_severity(value: Any) -> str:
    text = str(value or "warning").strip().lower()
    if text in {"info", "warning", "error"}:
        return text
    return "warning"


def _weighted_score(issues: Sequence["DiagnosticIssue"]) -> float:
    if not issues:
        return 1.0

    weight_map = {"info": 0.02, "warning": 0.08, "error": 0.20}
    penalty = sum(weight_map.get(issue.severity, 0.08) for issue in issues)
    return max(0.0, min(1.0, 1.0 - penalty))


@dataclass(frozen=True, slots=True)
class DiagnosticSettings:
    """
    Paramètres des diagnostics.
    """

    min_trades: int = 3
    min_win_rate: float = 0.40
    min_profit_factor: float = 1.0
    min_expectancy: float = 0.0

    max_drawdown_ratio: float = 0.50
    max_fee_pressure: float = 0.50
    min_mfe_to_mae_ratio: float = 1.0

    min_signal_coverage: float = 0.01
    min_trade_var: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "min_trades", max(1, _coerce_int(self.min_trades, 3)))
        object.__setattr__(self, "min_win_rate", min(1.0, max(0.0, _coerce_float(self.min_win_rate, 0.40))))
        object.__setattr__(self, "min_profit_factor", max(0.0, _coerce_float(self.min_profit_factor, 1.0)))
        object.__setattr__(self, "min_expectancy", _coerce_float(self.min_expectancy, 0.0))
        object.__setattr__(self, "max_drawdown_ratio", min(1.0, max(0.0, _coerce_float(self.max_drawdown_ratio, 0.50))))
        object.__setattr__(self, "max_fee_pressure", min(10.0, max(0.0, _coerce_float(self.max_fee_pressure, 0.50))))
        object.__setattr__(self, "min_mfe_to_mae_ratio", max(0.0, _coerce_float(self.min_mfe_to_mae_ratio, 1.0)))
        object.__setattr__(self, "min_signal_coverage", min(1.0, max(0.0, _coerce_float(self.min_signal_coverage, 0.01))))
        object.__setattr__(self, "min_trade_var", max(0.0, _coerce_float(self.min_trade_var, 0.0)))


@dataclass(frozen=True, slots=True)
class DiagnosticIssue:
    """
    Problème détecté pendant l'exécution.
    """

    code: str
    message: str
    severity: str = "warning"
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", str(self.code).strip().lower())
        object.__setattr__(self, "message", str(self.message).strip())
        object.__setattr__(self, "severity", _normalize_severity(self.severity))
        object.__setattr__(self, "details", dict(self.details))

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "details": dict(self.details),
        }


@dataclass(frozen=True, slots=True)
class ExecutionDiagnostics:
    """
    Résultat d'un diagnostic d'exécution.
    """

    healthy: bool
    score: float
    issues: tuple[DiagnosticIssue, ...]

    summary: dict[str, Any] = field(default_factory=dict)
    profile: ExecutionProfile | None = None
    mae_mfe: MAEMFESummary | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "healthy", _coerce_bool(self.healthy, False))
        object.__setattr__(self, "score", min(1.0, max(0.0, float(self.score))))
        object.__setattr__(self, "issues", tuple(self.issues))
        object.__setattr__(self, "summary", dict(self.summary))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def issue_count(self) -> int:
        return len(self.issues)

    @property
    def has_errors(self) -> bool:
        return any(issue.severity == "error" for issue in self.issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "healthy": self.healthy,
            "score": self.score,
            "issues": [issue.to_dict() for issue in self.issues],
            "summary": dict(self.summary),
            "profile": None if self.profile is None else self.profile.to_dict(),
            "mae_mfe": None if self.mae_mfe is None else self.mae_mfe.to_dict(),
            "metadata": dict(self.metadata),
        }


class ExecutionDiagnoser:
    """
    Construit un diagnostic d'exécution à partir d'un replay.
    """

    def __init__(self, settings: DiagnosticSettings | None = None) -> None:
        self._settings = settings or DiagnosticSettings()

    @property
    def settings(self) -> DiagnosticSettings:
        return self._settings

    def diagnose(
        self,
        replay: ReplayResult,
        *,
        profile: ExecutionProfile | None = None,
        mae_mfe: MAEMFESummary | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ExecutionDiagnostics:
        metrics = replay.metrics
        records = replay.records
        settings = self._settings
        issues: list[DiagnosticIssue] = []

        pnls = np.asarray([record.trade.pnl for record in records], dtype=float)
        fees = float(sum(record.fees_paid for record in records))
        slippage = float(sum(record.slippage_paid for record in records))
        spread = float(sum(record.spread_paid for record in records))
        overhead = fees + slippage + spread

        if metrics.trade_count < settings.min_trades:
            issues.append(
                DiagnosticIssue(
                    code="too_few_trades",
                    message="Le replay génère trop peu de trades.",
                    severity="error",
                    details={"trade_count": metrics.trade_count, "min_trades": settings.min_trades},
                )
            )

        if metrics.win_rate < settings.min_win_rate and metrics.trade_count > 0:
            issues.append(
                DiagnosticIssue(
                    code="weak_win_rate",
                    message="Le taux de trades gagnants est trop faible.",
                    severity="warning",
                    details={"win_rate": metrics.win_rate, "min_win_rate": settings.min_win_rate},
                )
            )

        if np.isfinite(metrics.profit_factor) and metrics.profit_factor < settings.min_profit_factor:
            issues.append(
                DiagnosticIssue(
                    code="weak_profit_factor",
                    message="Le profit factor est insuffisant.",
                    severity="warning",
                    details={"profit_factor": metrics.profit_factor, "min_profit_factor": settings.min_profit_factor},
                )
            )

        if metrics.expectancy < settings.min_expectancy:
            issues.append(
                DiagnosticIssue(
                    code="negative_expectancy",
                    message="L'expectancy est négative ou trop faible.",
                    severity="error" if metrics.expectancy < 0 else "warning",
                    details={"expectancy": metrics.expectancy, "min_expectancy": settings.min_expectancy},
                )
            )

        if metrics.max_drawdown > 0 and metrics.total_pnl > 0:
            drawdown_ratio = metrics.max_drawdown / max(abs(metrics.total_pnl), 1e-12)
            if drawdown_ratio > settings.max_drawdown_ratio:
                issues.append(
                    DiagnosticIssue(
                        code="excessive_drawdown",
                        message="Le drawdown est trop élevé par rapport au résultat total.",
                        severity="warning",
                        details={
                            "drawdown_ratio": drawdown_ratio,
                            "max_drawdown_ratio": settings.max_drawdown_ratio,
                            "max_drawdown": metrics.max_drawdown,
                            "total_pnl": metrics.total_pnl,
                        },
                    )
                )

        if metrics.signal_coverage < settings.min_signal_coverage:
            issues.append(
                DiagnosticIssue(
                    code="low_signal_coverage",
                    message="Le signal couvre trop peu de barres.",
                    severity="warning",
                    details={
                        "signal_coverage": metrics.signal_coverage,
                        "min_signal_coverage": settings.min_signal_coverage,
                    },
                )
            )

        if metrics.trade_count > 0:
            fee_pressure = _safe_div(overhead, abs(metrics.gross_profit) if abs(metrics.gross_profit) > 1e-12 else abs(metrics.total_pnl) + 1e-12)
            if fee_pressure > settings.max_fee_pressure:
                issues.append(
                    DiagnosticIssue(
                        code="high_fee_pressure",
                        message="Les coûts d'exécution absorbent trop de performance.",
                        severity="warning",
                        details={
                            "fee_pressure": fee_pressure,
                            "max_fee_pressure": settings.max_fee_pressure,
                            "fees": fees,
                            "slippage": slippage,
                            "spread": spread,
                        },
                    )
                )

        if pnls.size > 1 and float(np.var(pnls)) <= settings.min_trade_var:
            issues.append(
                DiagnosticIssue(
                    code="flat_trade_variance",
                    message="La variance des trades est trop faible.",
                    severity="info",
                    details={"trade_variance": float(np.var(pnls)), "min_trade_var": settings.min_trade_var},
                )
            )

        if mae_mfe is not None:
            if mae_mfe.avg_mfe_to_mae_ratio < settings.min_mfe_to_mae_ratio:
                issues.append(
                    DiagnosticIssue(
                        code="weak_excursion_profile",
                        message="Le profil MAE/MFE montre un avantage trop faible.",
                        severity="warning",
                        details={
                            "avg_mfe_to_mae_ratio": mae_mfe.avg_mfe_to_mae_ratio,
                            "min_mfe_to_mae_ratio": settings.min_mfe_to_mae_ratio,
                        },
                    )
                )

        if profile is not None and profile.recovery_factor < 1.0 and profile.total_pnl <= 0:
            issues.append(
                DiagnosticIssue(
                    code="no_recovery",
                    message="Le profil n'affiche pas de capacité de récupération.",
                    severity="warning",
                    details={"recovery_factor": profile.recovery_factor, "total_pnl": profile.total_pnl},
                )
            )

        if metrics.trade_count > 0 and metrics.win_rate >= 0.5 and metrics.total_pnl <= 0:
            issues.append(
                DiagnosticIssue(
                    code="win_rate_pnl_mismatch",
                    message="Le taux de réussite ne se traduit pas en performance positive.",
                    severity="warning",
                    details={"win_rate": metrics.win_rate, "total_pnl": metrics.total_pnl},
                )
            )

        healthy = not any(issue.severity == "error" for issue in issues)
        score = _weighted_score(issues)

        summary = {
            "trade_count": metrics.trade_count,
            "win_rate": metrics.win_rate,
            "profit_factor": metrics.profit_factor,
            "expectancy": metrics.expectancy,
            "total_pnl": metrics.total_pnl,
            "max_drawdown": metrics.max_drawdown,
            "signal_coverage": metrics.signal_coverage,
            "fee_pressure": _safe_div(overhead, abs(metrics.gross_profit) if abs(metrics.gross_profit) > 1e-12 else abs(metrics.total_pnl) + 1e-12),
        }

        return ExecutionDiagnostics(
            healthy=healthy,
            score=score,
            issues=tuple(issues),
            summary=summary,
            profile=profile,
            mae_mfe=mae_mfe,
            metadata=dict(metadata or {}),
        )

    def diagnose_replay(self, replay: ReplayResult, **kwargs: Any) -> ExecutionDiagnostics:
        return self.diagnose(replay, **kwargs)

    def __repr__(self) -> str:
        return (
            "ExecutionDiagnoser("
            f"min_trades={self._settings.min_trades}, "
            f"min_win_rate={self._settings.min_win_rate}"
            ")"
        )