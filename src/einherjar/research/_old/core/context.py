"""
==========================================================
Engine Context
==========================================================

Conteneur des ressources nécessaires à l'exécution du
pipeline pour UNE PAIRE asset / timeframe.

L'Engine crée une instance d'EngineContext pour chaque
paire traitée. L'instance est détruite à la fin du
traitement de la paire. Aucun EngineContext n'est jamais
partagé entre paires.

Champs :

- config             : configuration globale, read-only.
- state              : état d'avancement du pipeline pour
                       cette paire (EngineState).
- target             : paire asset / timeframe traitée.
- feature_registry   : registre des features de la paire.
- dataset_loader     : loader MIDAS de la paire.
- dataset_validator  : validateur du dataset de la paire.
- dataset_statistics : statistiques du dataset de la paire.
- dataset_inspector  : inspector du dataset de la paire.
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
    Contexte d'exécution pour une paire asset / timeframe.
    """

    config: Config

    state: EngineState

    target: Any  # DiscoveryTarget (évite un import cyclique)

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

    @property
    def asset(self) -> str:
        return self.target.asset

    @property
    def timeframe(self) -> str:
        return self.target.timeframe

    @property
    def pair_key(self) -> str:
        return self.target.key

    # ==================================================
    # PYTHON PROTOCOL
    # ==================================================

    def __repr__(self) -> str:

        return (
            "EngineContext("
            f"pair='{self.target.key}', "
            f"features={self.feature_registry.feature_count}, "
            f"sample_count={self.dataset.midas.sample_count if self.dataset.is_midas_mode else 'n/a'}"
            ")"
        )
