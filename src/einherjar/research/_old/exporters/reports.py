# exporters/reports.py
"""
==========================================================
Reports Export
==========================================================

Export des rapports du pipeline.

Ce module ne construit pas les rapports métiers :
- il les rassemble,
- les normalise,
- les prépare pour sérialisation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping

from execution.execution_report import ExecutionReport
from portfolio.portfolio_report import PortfolioReport
from validation.validation_report import ValidationReport

__all__ = [
    "ReportBundle",
    "ReportBundleBuilder",
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


@dataclass(slots=True)
class ReportBundle:
    """
    Ensemble de rapports exportables.
    """

    validation: ValidationReport | None = None
    execution: ExecutionReport | None = None
    portfolio: PortfolioReport | None = None

    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self, *, summary_only: bool = False) -> dict[str, Any]:
        payload = {
            "created_at": self.created_at.isoformat(),
            "validation": None if self.validation is None else self.validation.to_dict(summary_only=summary_only),
            "execution": None if self.execution is None else self.execution.to_dict(summary_only=summary_only),
            "portfolio": None if self.portfolio is None else self.portfolio.to_dict(summary_only=summary_only),
            "metadata": dict(self.metadata),
        }
        return payload

    def to_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []

        if self.validation is not None:
            records.append(
                {
                    "report_type": "validation",
                    "name": getattr(self.validation, "name", "validation"),
                    "summary": _to_mapping(getattr(self.validation, "summary", {})),
                    "total_items": len(self.validation),
                }
            )

        if self.execution is not None:
            records.append(
                {
                    "report_type": "execution",
                    "name": getattr(self.execution, "name", "execution"),
                    "summary": _to_mapping(getattr(self.execution, "summary", {})),
                    "total_items": len(self.execution),
                }
            )

        if self.portfolio is not None:
            records.append(
                {
                    "report_type": "portfolio",
                    "name": getattr(self.portfolio, "name", "portfolio"),
                    "summary": _to_mapping(getattr(self.portfolio, "summary", {})),
                    "total_items": len(self.portfolio),
                }
            )

        return records

    def __repr__(self) -> str:
        return (
            "ReportBundle("
            f"validation={self.validation is not None}, "
            f"execution={self.execution is not None}, "
            f"portfolio={self.portfolio is not None}"
            ")"
        )


class ReportBundleBuilder:
    """
    Construit un bundle de rapports.
    """

    @staticmethod
    def build(
        *,
        validation: ValidationReport | None = None,
        execution: ExecutionReport | None = None,
        portfolio: PortfolioReport | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ReportBundle:
        return ReportBundle(
            validation=validation,
            execution=execution,
            portfolio=portfolio,
            metadata=dict(metadata or {}),
        )

    @staticmethod
    def from_reports(**kwargs: Any) -> ReportBundle:
        return ReportBundleBuilder.build(**kwargs)