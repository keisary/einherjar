"""
==========================================================
Global Configuration
==========================================================

Point d'entrée unique de toute la configuration.
"""

from dataclasses import dataclass
from typing import Any

from .dataset import DatasetConfig
from .execution import ExecutionConfig
from .export import ExportConfig
from .logging import LoggingConfig
from .parallel import ParallelConfig
from .portfolio import PortfolioConfig
from .scoring import ScoringConfig
from .search import SearchConfig
from .validation import ValidationConfig


@dataclass(slots=True)
class Config:
    """
    Global configuration object.

    Toutes les phases du moteur utilisent cette
    instance unique.
    """

    dataset: DatasetConfig = DatasetConfig()

    search: SearchConfig = SearchConfig()

    validation: ValidationConfig = ValidationConfig()

    execution: ExecutionConfig = ExecutionConfig()

    portfolio: PortfolioConfig = PortfolioConfig()

    scoring: ScoringConfig = ScoringConfig()

    parallel: ParallelConfig = ParallelConfig()

    export: ExportConfig = ExportConfig()
    logging: LoggingConfig = LoggingConfig()

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset.to_dict() if hasattr(self.dataset, "to_dict") else {},
            "search": self.search.to_dict() if hasattr(self.search, "to_dict") else {},
            "validation": self.validation.to_dict() if hasattr(self.validation, "to_dict") else {},
            "execution": self.execution.to_dict() if hasattr(self.execution, "to_dict") else {},
            "portfolio": self.portfolio.to_dict() if hasattr(self.portfolio, "to_dict") else {},
            "scoring": self.scoring.to_dict() if hasattr(self.scoring, "to_dict") else {},
            "parallel": self.parallel.to_dict() if hasattr(self.parallel, "to_dict") else {},
            "export": self.export.to_dict() if hasattr(self.export, "to_dict") else {},
            "logging": self.logging.to_dict() if hasattr(self.logging, "to_dict") else {},
        }
