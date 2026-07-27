"""
==========================================================
Discovery Engine
==========================================================

Point d'entrée principal du moteur.

L'Engine construit toutes les ressources partagées,
initialise le contexte puis orchestre les différentes
phases de découverte.

Aucune logique métier n'est implémentée ici.
"""

from __future__ import annotations

from config.config import Config

from dataset.inspector import DatasetInspector
from dataset.loader import DatasetLoader
from dataset.statistics import DatasetStatistics
from dataset.validator import DatasetValidator

from models.feature_registry import FeatureRegistry

from .context import EngineContext
from .state import EngineState


class Engine:
    """
    Orchestrateur principal.
    """

    # ==================================================
    # INITIALIZATION
    # ==================================================

    def __init__(
        self,
        config: Config,
    ) -> None:

        self._config = config

        self._registry = FeatureRegistry(
            config.dataset.metadata_path,
        )

        self._dataset = DatasetLoader(
            config.dataset,
        )

        self._validator = DatasetValidator()

        self._statistics = DatasetStatistics(
            self._dataset,
        )

        self._inspector = DatasetInspector(
            self._dataset,
        )

        self._state = EngineState()

        self._context = EngineContext(
            config=self._config,
            state=self._state,
            feature_registry=self._registry,
            dataset_loader=self._dataset,
            dataset_validator=self._validator,
            dataset_statistics=self._statistics,
            dataset_inspector=self._inspector,
        )

    # ==================================================
    # PROPERTIES
    # ==================================================

    @property
    def context(self) -> EngineContext:
        return self._context

    @property
    def state(self) -> EngineState:
        return self._state

    # ==================================================
    # LIFECYCLE
    # ==================================================

    def initialize(self) -> None:

        self._validator.validate(
            self._dataset,
        )

    def run(
        self,
        *,
        pairs: Any | None = None,
        assets: Any | None = None,
        timeframes: Any | None = None,
    ) -> Any:
        """
        Point d'entrée du pipeline.

        Phase A : Data Contract (initialize)
        Phase B-E : discovery -> validation -> execution -> portfolio
        """
        self.initialize()
        from discovery import DiscoveryOrchestrator
        orchestrator = DiscoveryOrchestrator(config=self._config)
        return orchestrator.run(
            pairs=pairs,
            assets=assets,
            timeframes=timeframes,
        )
        """
        Point d'entrée du pipeline.

        Les différentes phases seront ajoutées
        progressivement.
        """

        self.initialize()

    # ==================================================
    # PYTHON PROTOCOL
    # ==================================================

    def __repr__(self) -> str:

        return (
            "Engine("
            f"dataset={self._dataset}, "
            f"features={self._registry.feature_count}"
            ")"
        )