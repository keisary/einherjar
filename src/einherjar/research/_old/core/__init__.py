"""
==========================================================
Core Package
==========================================================

Le package core contient les fondations partagées du
système :

- Le contexte d'exécution par paire (``EngineContext``).
- L'état d'avancement par paire (``EngineState``,
  ``PhaseStatus``).
- Les exceptions métier (``DiscoveryError``,
  ``*ContractError``).
- Les types valeur partagés (``DiscoveryTarget``,
  ``DiscoveryPairResult``).
- Les helpers transverses (résolution d'asset).
- Le moteur per-pair (``Engine``) — importable
  paresseusement depuis ``core.engine``.
- Le bootstrap run-level (``DiscoveryOrchestrator``,
  ``DiscoveryRunResult``, ``DiscoverySettings``) —
  importable paresseusement depuis ``core.runner``.
- L'exporter run-level (``PairExporter``) —
  importable paresseusement depuis ``core.exporter``.

AUCUNE logique métier d'algorithme de recherche n'est
implémentée ici.

NOTE SUR LES IMPORTS
--------------------

``core/__init__.py`` n'importe PAS ``Engine``,
``DiscoveryOrchestrator`` ni ``PairExporter`` de manière
eager. Ces modules tirent des dépendances lourdes
(``discovery.explorer`` -> ``core.context``) et leur
import depuis ``core/__init__`` créerait un cycle :

    core.__init__ -> core.engine
        -> discovery.explorer
            -> core.context
                -> core.__init__  (déjà en cours de
                                    chargement)

Casser ce cycle impose d'importer ``Engine`` directement
depuis ``core.engine``, ``DiscoveryOrchestrator``
depuis ``core.runner``, et ``PairExporter`` depuis
``core.exporter``. Les ``__init__.py`` des couches
haut-niveau (``research/__init__.py``, etc.) doivent
respecter ce découpage.
"""

from .assets import known_assets
from .assets import reset_cache
from .assets import resolve_asset_class
from .assets import resolve_asset_meta
from .context import EngineContext
from .exceptions import ConfigurationError
from .exceptions import DatasetContractError
from .exceptions import DatasetError
from .exceptions import DatasetValidationError
from .exceptions import DiscoveryContractError
from .exceptions import DiscoveryError
from .exceptions import ExecutionContractError
from .exceptions import ExecutionError
from .exceptions import ExportContractError
from .exceptions import KnowledgeContractError
from .exceptions import KnowledgeError
from .exceptions import MemoryContractError
from .exceptions import MemoryError
from .exceptions import PhaseContractError
from .exceptions import PortfolioContractError
from .exceptions import PortfolioError
from .exceptions import SearchError
from .exceptions import ValidationContractError
from .exceptions import ValidationError
from .state import EngineState
from .state import PHASE_NAMES
from .state import PhaseStatus
from .types import DiscoveryPairResult
from .types import DiscoveryTarget


def __getattr__(name):
    """
    Lazy access pour Engine / PairExporter /
    DiscoveryOrchestrator / DiscoveryRunResult /
    DiscoverySettings.

    Permet ``from core import Engine`` etc. sans
    déclencher l'import circulaire au chargement
    initial de ``core``.
    """

    if name == "Engine":
        from .engine import Engine
        return Engine
    if name == "PairExporter":
        from .exporter import PairExporter
        return PairExporter
    if name in {
        "DiscoveryOrchestrator",
        "DiscoveryRunResult",
        "DiscoverySettings",
    }:
        from . import runner as _runner
        return getattr(_runner, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ConfigurationError",
    "DatasetContractError",
    "DatasetError",
    "DatasetValidationError",
    "DiscoveryContractError",
    "DiscoveryError",
    "DiscoveryOrchestrator",
    "DiscoveryPairResult",
    "DiscoveryRunResult",
    "DiscoverySettings",
    "DiscoveryTarget",
    "Engine",
    "EngineContext",
    "EngineState",
    "ExecutionContractError",
    "ExecutionError",
    "ExportContractError",
    "KnowledgeContractError",
    "KnowledgeError",
    "MemoryContractError",
    "MemoryError",
    "PHASE_NAMES",
    "PairExporter",
    "PhaseContractError",
    "PhaseStatus",
    "PortfolioContractError",
    "PortfolioError",
    "SearchError",
    "ValidationContractError",
    "ValidationError",
    "known_assets",
    "reset_cache",
    "resolve_asset_class",
    "resolve_asset_meta",
]
