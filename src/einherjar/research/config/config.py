"""
==========================================================
Global Configuration
==========================================================

Point d'entrée unique de toute la configuration.
"""

from dataclasses import dataclass

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