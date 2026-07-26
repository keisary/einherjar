"""
==========================================================
Engine Context
==========================================================

Conteneur des ressources partagées par l'ensemble du moteur.

Toutes les phases reçoivent une unique instance de
EngineContext afin d'accéder aux ressources communes.
"""

from __future__ import annotations

from dataclasses import dataclass

from config.config import Config

from dataset.inspector import DatasetInspector
from dataset.loader import DatasetLoader
from dataset.statistics import DatasetStatistics
from dataset.validator import DatasetValidator

from models.feature_registry import FeatureRegistry

from .state import EngineState


@dataclass(frozen=True, slots=True)
class EngineContext:
    """
    Contexte global du moteur.
    """

    config: Config

    state: EngineState

    feature_registry: FeatureRegistry

    dataset_loader: DatasetLoader

    dataset_validator: DatasetValidator

    dataset_statistics: DatasetStatistics

    dataset_inspector: DatasetInspector

    # ==================================================
    # SHORTCUTS
    # ==================================================

    @property
    def dataset(self) -> DatasetLoader:
        return self.dataset_loader

    @property
    def features(self) -> FeatureRegistry:
        return self.feature_registry

    @property
    def validator(self) -> DatasetValidator:
        return self.dataset_validator

    @property
    def statistics(self) -> DatasetStatistics:
        return self.dataset_statistics

    @property
    def inspector(self) -> DatasetInspector:
        return self.dataset_inspector

    # ==================================================
    # PYTHON PROTOCOL
    # ==================================================

    def __repr__(self) -> str:

        return (
            "EngineContext("
            f"features={self.feature_registry.feature_count}, "
            f"splits={self.dataset_loader.splits}"
            ")"
        )