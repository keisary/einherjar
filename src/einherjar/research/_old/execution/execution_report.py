# execution/execution_report.py
"""
==========================================================
Execution Report
==========================================================

Rapport cumulatif de la phase Execution.

Ce module agrège :
- les ReplayResult,
- les excursions MAE/MFE,
- les profils d'exécution,
- les diagnostics,
- les empreintes d'exécution.

Il ne simule rien et ne diagnostique rien à lui seul.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timezone
from typing import Any
from typing import Iterable
from typing import Mapping

from models.journal import Journal
from models.trade import Trade
from models.validated_candidate import ValidatedCandidate

from .diagnostics import ExecutionDiagnostics
from .fingerprint import ExecutionFingerprint
from .mae_mfe import MAEMFESummary
from .profiler import ExecutionProfile
from .replay import ReplayResult
from .trade_builder import ExecutedTradeRecord

__all__ = [
    "ExecutionResult",
    "ExecutionReport",
]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _to_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}

    if hasattr(value, "to_dict") and callable(value.to_dict):
        value = value.to_dict()

    if isinstance(value, Mapping):
        return dict(value)

    return {}


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


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    """
    Résultat complet d'une exécution.
    """

    subject_fingerprint: str
    execution_fingerprint: ExecutionFingerprint

    validated_candidate: ValidatedCandidate | None
    candidate: Any
    hypothesis: Any

    replay: ReplayResult
    journal: Journal
    trades: tuple[Trade, ...]
    records: tuple[ExecutedTradeRecord, ...]

    mae_mfe: MAEMFESummary | None = None
    profile: ExecutionProfile | None = None
    diagnostics: ExecutionDiagnostics | None = None

    success: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        object.__setattr__(self, "subject_fingerprint", str(self.subject_fingerprint))
        object.__setattr__(self, "trades", tuple(self.trades))
        object.__setattr__(self, "records", tuple(self.records))
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "success", _coerce_bool(self.success, True))

    @property
    def trade_count(self) -> int:
        return self.replay.metrics.trade_count

    @property
    def total_pnl(self) -> float:
        return self.replay.metrics.total_pnl

    @property
    def win_rate(self) -> float:
        return self.replay.metrics.win_rate

    @property
    def healthy(self) -> bool:
        if self.diagnostics is not None:
            return self.diagnostics.healthy
        return self.success

    @property
    def issue_count(self) -> int:
        if self.diagnostics is None:
            return 0
        return self.diagnostics.issue_count

    def to_dict(self, *, summary_only: bool = False) -> dict[str, Any]:
        """
        Sérialise un ExecutionResult.

        En mode summary_only=True (défaut pour les reports),
        on garde uniquement les métriques agrégées et les
        compteurs. On drop :
        - replay (volumineux, détaillé)
        - journal (liste de trades)
        - trades (liste de Trade détaillés)
        - records (43K+ ExecutedTradeRecord)
        - mae_mfe (détails des excursions)
        - profile (sub-objets détaillés)
        - diagnostics (verbose)
        - validated_candidate / candidate / hypothesis
          (les métriques de l'Einher sont déjà dans le corpus)

        Le détail reste accessible via les attributs directs
        (self.replay, self.records, etc.) et via le corpus
        compressé (parquet/csv).
        """
        payload = {
            "subject_fingerprint": self.subject_fingerprint,
            "success": self.success,
            "healthy": self.healthy,
            "issue_count": self.issue_count,
            "trade_count": self.trade_count,
            "total_pnl": self.total_pnl,
            "win_rate": self.win_rate,
            "created_at": self.created_at.isoformat(),
        }

        # En mode summary_only, on ne garde que le digest du
        # fingerprint (le components/metadata peut peser plusieurs
        # centaines de MB si l'embedder/contexte est inclus).
        if self.execution_fingerprint is not None:
            digest = (
                self.execution_fingerprint.fingerprint.digest
                if hasattr(self.execution_fingerprint, "fingerprint")
                and self.execution_fingerprint.fingerprint is not None
                else None
            )
            payload["execution_fingerprint_digest"] = digest
            payload["execution_fingerprint_version"] = (
                self.execution_fingerprint.version
            )
            payload["execution_kind"] = (
                self.execution_fingerprint.execution_kind
            )
        else:
            payload["execution_fingerprint_digest"] = None

        # Métriques complètes issues du replay (si disponible)
        if self.replay is not None:
            metrics = self.replay.metrics.to_dict()
            payload["metrics"] = metrics
            payload["profit_factor"] = float(metrics.get("profit_factor", 0.0))
            payload["expectancy"] = float(metrics.get("expectancy", 0.0))
            payload["max_drawdown"] = float(metrics.get("max_drawdown", 0.0))
        else:
            payload["metrics"] = {}
            payload["profit_factor"] = 0.0
            payload["expectancy"] = 0.0
            payload["max_drawdown"] = 0.0

        if summary_only:
            return payload

        payload.update({
            "execution_fingerprint": (
                self.execution_fingerprint.to_dict()
                if self.execution_fingerprint is not None
                else None
            ),
            "validated_candidate": (
                None if self.validated_candidate is None
                else self.validated_candidate.to_dict()
            ),
            "candidate": (
                self.candidate.to_dict()
                if hasattr(self.candidate, "to_dict")
                else repr(self.candidate)
            ),
            "hypothesis": (
                self.hypothesis.to_dict()
                if hasattr(self.hypothesis, "to_dict")
                else repr(self.hypothesis)
            ),
            "replay": self.replay.to_dict(summary_only=False),
            "journal": self.journal.to_dict(),
            "trades": [trade.to_dict() for trade in self.trades],
            "records": [record.to_dict() for record in self.records],
            "mae_mfe": None if self.mae_mfe is None else self.mae_mfe.to_dict(),
            "profile": None if self.profile is None else self.profile.to_dict(),
            "diagnostics": None if self.diagnostics is None else self.diagnostics.to_dict(),
            "metadata": dict(self.metadata),
        })
        return payload

    def __repr__(self) -> str:
        return (
            "ExecutionResult("
            f"trades={self.trade_count}, "
            f"pnl={self.total_pnl:.4f}, "
            f"healthy={self.healthy}"
            ")"
        )


@dataclass(slots=True)
class ExecutionReport:
    """
    Rapport cumulatif de la phase Execution.
    """

    name: str = "execution"
    metadata: dict[str, Any] = field(default_factory=dict)

    started_at: datetime | None = None
    finished_at: datetime | None = None

    results: list[ExecutionResult] = field(default_factory=list)

    total_executions: int = 0
    total_trades: int = 0
    total_pnl: float = 0.0

    healthy_count: int = 0
    unhealthy_count: int = 0

    diagnostic_issue_counts: Counter[str] = field(default_factory=Counter)
    direction_counts: Counter[str] = field(default_factory=Counter)

    best_total_pnl: float = float("-inf")
    best_subject_fingerprint: str | None = None
    best_execution_fingerprint: str | None = None

    last_reason: str | None = None
    stopped_reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", str(self.name).strip() or "execution")
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "results", list(self.results))
        object.__setattr__(self, "diagnostic_issue_counts", Counter(self.diagnostic_issue_counts))
        object.__setattr__(self, "direction_counts", Counter(self.direction_counts))

    # ==================================================
    # LIFECYCLE
    # ==================================================

    def start(self) -> None:
        if self.started_at is None:
            self.started_at = _utc_now()
        self.finished_at = None
        self.stopped_reason = None

    def finish(self, reason: str | None = None) -> None:
        if self.started_at is None:
            self.start()
        self.finished_at = _utc_now()
        self.stopped_reason = reason or self.stopped_reason or "finished"

    def reset(self) -> None:
        self.started_at = None
        self.finished_at = None
        self.results.clear()
        self.total_executions = 0
        self.total_trades = 0
        self.total_pnl = 0.0
        self.healthy_count = 0
        self.unhealthy_count = 0
        self.diagnostic_issue_counts.clear()
        self.direction_counts.clear()
        self.best_total_pnl = float("-inf")
        self.best_subject_fingerprint = None
        self.best_execution_fingerprint = None
        self.last_reason = None
        self.stopped_reason = None

    # ==================================================
    # PROPERTIES
    # ==================================================

    @property
    def duration_seconds(self) -> float:
        if self.started_at is None:
            return 0.0
        end = self.finished_at or _utc_now()
        return max(0.0, (end - self.started_at).total_seconds())

    @property
    def average_pnl_per_execution(self) -> float:
        if self.total_executions == 0:
            return 0.0
        return self.total_pnl / self.total_executions

    @property
    def average_trades_per_execution(self) -> float:
        if self.total_executions == 0:
            return 0.0
        return self.total_trades / self.total_executions

    @property
    def healthy_rate(self) -> float:
        if self.total_executions == 0:
            return 0.0
        return self.healthy_count / self.total_executions

    @property
    def unhealthy_rate(self) -> float:
        if self.total_executions == 0:
            return 0.0
        return self.unhealthy_count / self.total_executions

    @property
    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "total_executions": self.total_executions,
            "total_trades": self.total_trades,
            "total_pnl": self.total_pnl,
            "average_pnl_per_execution": self.average_pnl_per_execution,
            "average_trades_per_execution": self.average_trades_per_execution,
            "healthy_count": self.healthy_count,
            "unhealthy_count": self.unhealthy_count,
            "healthy_rate": self.healthy_rate,
            "duration_seconds": self.duration_seconds,
            "best_total_pnl": None if self.best_total_pnl == float("-inf") else self.best_total_pnl,
            "best_subject_fingerprint": self.best_subject_fingerprint,
            "best_execution_fingerprint": self.best_execution_fingerprint,
            "diagnostic_issue_counts": dict(self.diagnostic_issue_counts),
            "direction_counts": dict(self.direction_counts),
        }

    @property
    def best_result(self) -> ExecutionResult | None:
        if not self.results:
            return None
        return max(self.results, key=lambda item: item.total_pnl)

    # ==================================================
    # RECORDING
    # ==================================================

    def record_result(self, result: ExecutionResult) -> ExecutionResult:
        self.results.append(result)

        self.total_executions += 1
        self.total_trades += result.trade_count
        self.total_pnl += result.total_pnl

        if result.healthy:
            self.healthy_count += 1
        else:
            self.unhealthy_count += 1

        direction = getattr(result.replay.metrics, "direction", "unknown")
        self.direction_counts[str(direction).strip().lower()] += 1

        if result.diagnostics is not None:
            for issue in result.diagnostics.issues:
                self.diagnostic_issue_counts[issue.code] += 1
                self.last_reason = issue.code

        if result.total_pnl > self.best_total_pnl:
            self.best_total_pnl = result.total_pnl
            self.best_subject_fingerprint = result.subject_fingerprint
            self.best_execution_fingerprint = result.execution_fingerprint.digest

        return result

    def record(self, result: ExecutionResult) -> ExecutionResult:
        return self.record_result(result)

    # ==================================================
    # SERIALIZATION
    # ==================================================

    def to_dict(self, *, summary_only: bool = False) -> dict[str, Any]:
        payload = {
            "name": self.name,
            "metadata": dict(self.metadata),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "results": [
                result.to_dict(summary_only=summary_only)
                for result in self.results
            ],
            "total_executions": self.total_executions,
            "total_trades": self.total_trades,
            "total_pnl": self.total_pnl,
            "healthy_count": self.healthy_count,
            "unhealthy_count": self.unhealthy_count,
            "diagnostic_issue_counts": dict(self.diagnostic_issue_counts),
            "direction_counts": dict(self.direction_counts),
            "best_total_pnl": None if self.best_total_pnl == float("-inf") else self.best_total_pnl,
            "best_subject_fingerprint": self.best_subject_fingerprint,
            "best_execution_fingerprint": self.best_execution_fingerprint,
            "last_reason": self.last_reason,
            "stopped_reason": self.stopped_reason,
            "duration_seconds": self.duration_seconds,
            "healthy_rate": self.healthy_rate,
            "summary": self.summary,
        }
        return payload

    def __len__(self) -> int:
        return len(self.results)

    def __iter__(self):
        return iter(self.results)

    def __repr__(self) -> str:
        return (
            "ExecutionReport("
            f"name='{self.name}', "
            f"executions={self.total_executions}, "
            f"pnl={self.total_pnl:.4f}"
            ")"
        )