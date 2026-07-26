# execution/executor.py
"""
==========================================================
Execution Executor
==========================================================

Orchestre la phase Execution.

Le moteur :
- reçoit un ValidatedCandidate, un Candidate ou une Hypothesis,
- lance le replay,
- calcule MAE/MFE,
- produit le profil,
- génère les diagnostics,
- enregistre le résultat dans le report.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Iterable
from typing import Mapping
from typing import Sequence

import numpy as np

from config.execution import ExecutionConfig
from core.context import EngineContext
from models.candidate import Candidate
from models.hypothesis import Hypothesis
from models.validated_candidate import ValidatedCandidate

from .diagnostics import ExecutionDiagnoser
from .execution_report import ExecutionReport
from .execution_report import ExecutionResult
from .fingerprint import ExecutionFingerprint
from .fingerprint import build_execution_fingerprint
from .mae_mfe import MAEMFEAnalyzer
from .profiler import ExecutionProfiler
from .replay import ReplayEngine
from .replay import ReplaySettings
from .trade_builder import TradeBuilder

__all__ = [
    "ExecutionEngine",
]


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


def _extract_config_part(config: Any | None, *names: str, default: Any = None) -> Any:
    if config is None:
        return default

    return _extract_field(config, *names, default=default)


def _extract_candidate_subject(subject: Any) -> tuple[Any, Hypothesis]:
    if isinstance(subject, ValidatedCandidate):
        return subject.candidate, subject.hypothesis

    if isinstance(subject, Candidate):
        return subject, subject.hypothesis

    if isinstance(subject, Hypothesis):
        return Candidate(hypothesis=subject), subject

    if hasattr(subject, "candidate") and hasattr(subject.candidate, "hypothesis"):
        candidate = subject.candidate
        return candidate, candidate.hypothesis

    if hasattr(subject, "hypothesis"):
        hypothesis = subject.hypothesis
        if isinstance(hypothesis, Hypothesis):
            return subject, hypothesis

    raise TypeError("subject must be a ValidatedCandidate, a Candidate or a Hypothesis.")


@dataclass(slots=True)
class ExecutionEngine:
    """
    Orchestrateur principal de l'exécution.
    """

    config: Any | None = None

    replay_engine: ReplayEngine = field(default_factory=ReplayEngine)
    trade_builder: TradeBuilder = field(default_factory=TradeBuilder)
    mae_mfe: MAEMFEAnalyzer = field(default_factory=MAEMFEAnalyzer)
    profiler: ExecutionProfiler = field(default_factory=ExecutionProfiler)
    diagnoser: ExecutionDiagnoser = field(default_factory=ExecutionDiagnoser)
    report: ExecutionReport = field(default_factory=ExecutionReport)

    def __post_init__(self) -> None:
        if self.config is not None:
            self._configure_from_config(self.config)

    @classmethod
    def from_config(
        cls,
        config: Any,
    ) -> "ExecutionEngine":
        return cls(config=config)

    @property
    def settings(self) -> ExecutionConfig:
        return self.replay_engine.settings.execution

    def _configure_from_config(self, config: Any) -> None:
        execution_cfg = _extract_config_part(config, "execution", "execution_config", default=None)
        replay_cfg = _extract_config_part(config, "replay", "replay_config", default=None)
        diagnostics_cfg = _extract_config_part(config, "diagnostics", "diagnostic", default=None)
        profiler_cfg = _extract_config_part(config, "profile", "profiler", "execution_profile", default=None)
        report_cfg = _extract_config_part(config, "execution_report", "report", default=None)

        if isinstance(execution_cfg, Mapping):
            execution_cfg = ExecutionConfig(**dict(execution_cfg))
        elif not isinstance(execution_cfg, ExecutionConfig):
            execution_cfg = ExecutionConfig()

        if replay_cfg is None:
            replay_cfg = {}
        if isinstance(replay_cfg, Mapping):
            replay_settings = ReplaySettings.from_config(
                {
                    "execution": execution_cfg,
                    "replay": replay_cfg,
                }
            )
            self.replay_engine = ReplayEngine(settings=replay_settings)
        else:
            self.replay_engine = ReplayEngine(settings=ReplaySettings.from_config(config))

        self.trade_builder = TradeBuilder(config=execution_cfg)

        if isinstance(diagnostics_cfg, Mapping):
            self.diagnoser = ExecutionDiagnoser()
        if isinstance(profiler_cfg, Mapping):
            self.profiler = ExecutionProfiler()
        if isinstance(report_cfg, Mapping):
            self.report = ExecutionReport(
                name=str(report_cfg.get("name", "execution")).strip() or "execution",
                metadata=dict(report_cfg.get("metadata", {})),
            )

    def execute(
        self,
        subject: Any,
        *,
        dataset: Any | None = None,
        matrix: Any | None = None,
        prices: Any | None = None,
        timestamps: Sequence[Any] | None = None,
        direction: str | None = None,
        quantity: float | None = None,
        price_column: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ExecutionResult:
        candidate, hypothesis = _extract_candidate_subject(subject)

        self.report.start()

        replay = self.replay_engine.run(
            subject,
            dataset=dataset,
            matrix=matrix,
            prices=prices,
            timestamps=timestamps,
            direction=direction,
            quantity=quantity,
            price_column=price_column,
            metadata=metadata,
        )

        mae_mfe = self.mae_mfe.assess_replay(replay)
        profile = self.profiler.profile(
            replay,
            mae_mfe=mae_mfe,
            name=_extract_field(metadata, "profile_name", default="execution_profile") if metadata else "execution_profile",
            description=_extract_field(metadata, "profile_description", default=None) if metadata else None,
            metadata=metadata,
        )
        diagnostics = self.diagnoser.diagnose(
            replay,
            profile=profile,
            mae_mfe=mae_mfe,
            metadata=metadata,
        )

        execution_fp = self._build_execution_fingerprint(
            replay=replay,
            candidate=candidate,
            hypothesis=hypothesis,
            mae_mfe=mae_mfe,
            profile=profile,
            diagnostics=diagnostics,
            direction=direction,
            quantity=quantity,
            metadata=metadata,
        )

        result = ExecutionResult(
            subject_fingerprint=replay.subject_fingerprint,
            execution_fingerprint=execution_fp,
            validated_candidate=subject if isinstance(subject, ValidatedCandidate) else None,
            candidate=candidate,
            hypothesis=hypothesis,
            replay=replay,
            journal=replay.journal,
            trades=replay.trades,
            records=replay.records,
            mae_mfe=mae_mfe,
            profile=profile,
            diagnostics=diagnostics,
            success=diagnostics.healthy,
            metadata=dict(metadata or {}),
        )

        self.report.record_result(result)
        return result

    def run(self, subject: Any, **kwargs: Any) -> ExecutionResult:
        return self.execute(subject, **kwargs)

    def replay(self, subject: Any, **kwargs: Any) -> ExecutionResult:
        return self.execute(subject, **kwargs)

    def execute_candidate(self, candidate: Candidate, **kwargs: Any) -> ExecutionResult:
        return self.execute(candidate, **kwargs)

    def execute_validated_candidate(self, candidate: ValidatedCandidate, **kwargs: Any) -> ExecutionResult:
        return self.execute(candidate, **kwargs)

    def execute_hypothesis(self, hypothesis: Hypothesis, **kwargs: Any) -> ExecutionResult:
        return self.execute(hypothesis, **kwargs)

    def execute_batch(
        self,
        subjects: Iterable[Any],
        **kwargs: Any,
    ) -> tuple[ExecutionResult, ...]:
        return tuple(self.execute(subject, **kwargs) for subject in subjects)

    def _build_execution_fingerprint(
        self,
        *,
        replay: Any,
        candidate: Any,
        hypothesis: Hypothesis,
        mae_mfe: Any | None,
        profile: Any | None,
        diagnostics: Any | None,
        direction: str | None,
        quantity: float | None,
        metadata: Mapping[str, Any] | None,
    ) -> ExecutionFingerprint:
        replay_metrics = replay.metrics
        execution_cfg = self.settings

        components = {
            "subject_fingerprint": replay.subject_fingerprint,
            "candidate_fingerprint": getattr(candidate, "fingerprint", None) or getattr(candidate, "digest", None) or None,
            "hypothesis_fingerprint": getattr(hypothesis, "fingerprint", None) or None,
            "direction": direction or replay_metrics.direction,
            "quantity": quantity if quantity is not None else replay_metrics.quantity,
            "trade_count": replay_metrics.trade_count,
            "total_pnl": replay_metrics.total_pnl,
            "win_rate": replay_metrics.win_rate,
            "profit_factor": replay_metrics.profit_factor,
            "expectancy": replay_metrics.expectancy,
            "signal_coverage": replay_metrics.signal_coverage,
            "max_drawdown": replay_metrics.max_drawdown,
            "execution_config": {
                "fees": execution_cfg.fees,
                "slippage": execution_cfg.slippage,
                "spread": execution_cfg.spread,
                "allow_long": execution_cfg.allow_long,
                "allow_short": execution_cfg.allow_short,
                "max_open_positions": execution_cfg.max_open_positions,
            },
            "mae_mfe": None if mae_mfe is None else mae_mfe.to_dict(),
            "profile": None if profile is None else profile.to_dict(),
            "diagnostics": None if diagnostics is None else diagnostics.to_dict(),
            "metadata": dict(metadata or {}),
        }

        return build_execution_fingerprint(
            subject_fingerprint=replay.subject_fingerprint,
            components=components,
            execution_kind="execution",
            version=1,
            parent_digest=replay.execution_fingerprint.digest,
            metadata={
                "module": "execution.executor",
            },
            candidate_fingerprint=replay.subject_fingerprint,
            hypothesis_fingerprint=getattr(hypothesis, "fingerprint", None),
            trade_count=replay.metrics.trade_count,
            total_pnl=replay.metrics.total_pnl,
            win_rate=replay.metrics.win_rate,
            profit_factor=replay.metrics.profit_factor,
            expectancy=replay.metrics.expectancy,
            direction=replay.metrics.direction,
            quantity=replay.metrics.quantity,
        )

    def reset(self) -> None:
        self.report.reset()

    def finish(self, reason: str | None = None) -> None:
        self.report.finish(reason)

    def __repr__(self) -> str:
        return (
            "ExecutionEngine("
            f"trades={len(self.report.results)}, "
            f"pnl={self.report.total_pnl:.4f}"
            ")"
        )