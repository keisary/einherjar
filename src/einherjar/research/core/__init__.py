"""
==========================================================
Core Package
==========================================================

Le package core contient :

- Le contexte d'exécution par paire (EngineContext).
- L'état d'avancement par paire (EngineState, PhaseStatus).
- L'orchestrateur par paire (Engine).
- Les exceptions métier (DiscoveryError, *ContractError).
- Les types valeur partagés (DiscoveryTarget, DiscoveryPairResult).
- Les helpers transverses (asset resolution).

Aucune logique métier d'algorithme de recherche n'est
implémentée ici.
"""

from .assets import known_assets
from .assets import reset_cache
from .assets import resolve_asset_class
from .assets import resolve_asset_meta
from .context import EngineContext
from .engine import Engine
from .runner import DiscoveryOrchestrator
from .runner import DiscoveryRunResult
from .runner import DiscoverySettings
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
