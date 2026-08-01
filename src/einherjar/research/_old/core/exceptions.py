"""
==========================================================
Engine Exceptions
==========================================================

Exceptions métier du moteur de découverte.

Hiérarchie :

    DiscoveryError
    ├── ConfigurationError
    ├── DatasetError
    │   ├── DatasetContractError    (contrat de données)
    │   └── DatasetValidationError  (données chargées)
    ├── PhaseContractError          (contrat entre phases)
    │   ├── DiscoveryContractError
    │   ├── ValidationContractError
    │   ├── ExecutionContractError
    │   ├── PortfolioContractError
    │   ├── MemoryContractError
    │   ├── KnowledgeContractError
    │   └── ExportContractError
    ├── SearchError
    ├── ValidationError
    ├── ExecutionError
    ├── PortfolioError
    ├── KnowledgeError
    └── MemoryError

Toute violation de contrat interrompt immédiatement
l'exécution. Aucun fallback silencieux n'est appliqué.
"""

from __future__ import annotations


class DiscoveryError(Exception):
    """
    Classe mère de toutes les exceptions métier du moteur.
    """


# ==========================================================
# CONFIGURATION
# ==========================================================

class ConfigurationError(DiscoveryError):
    """
    Erreur de configuration (Config, settings, paths).
    """


# ==========================================================
# DATASET
# ==========================================================

class DatasetError(DiscoveryError):
    """
    Erreur générique liée au dataset.
    """


class DatasetContractError(DatasetError):
    """
    Contrat de données invalide ou incomplet.

    Levée lorsque les métadonnées du dataset (feature_count,
    horizons, dtype, ...) ne satisfont pas les exigences
    structurelles du moteur.
    """


class DatasetValidationError(DatasetError):
    """
    Données chargées invalides.

    Levée lorsque la géométrie, le dtype ou la cohérence
    d'un tableau MIDAS / split ne satisfait pas le contrat.
    """


# ==========================================================
# PHASE CONTRACTS
# ==========================================================

class PhaseContractError(DiscoveryError):
    """
    Classe mère des violations de contrat entre phases.

    Levée dès qu'une phase ne respecte pas le contrat
    d'entrée ou de sortie attendu par la phase suivante.
    """


class DiscoveryContractError(PhaseContractError):
    """
    Contrat de la phase Discovery violé.

    Exemples typiques :
    - aucun candidat produit
    - sortie du moteur incompatible
    """


class ValidationContractError(PhaseContractError):
    """
    Contrat de la phase Validation violé.

    Exemples typiques :
    - sortie vide alors que des candidats existent
    - métadonnées de validation manquantes
    """


class ExecutionContractError(PhaseContractError):
    """
    Contrat de la phase Execution violé.

    Exemples typiques :
    - moteur d'exécution indisponible
    - matrices MIDAS absentes
    """


class PortfolioContractError(PhaseContractError):
    """
    Contrat de la phase Portfolio violé.

    Exemples typiques :
    - aucun résultat exécuté
    - allocation invalide
    """


class MemoryContractError(PhaseContractError):
    """
    Contrat de la phase Memory violé.
    """


class KnowledgeContractError(PhaseContractError):
    """
    Contrat de la phase Knowledge violé.
    """


class ExportContractError(PhaseContractError):
    """
    Contrat de la phase Export violé.
    """


# ==========================================================
# PHASE ERRORS
# ==========================================================

class SearchError(DiscoveryError):
    """
    Erreur durant la phase Discovery (recherche).
    """


class ValidationError(DiscoveryError):
    """
    Erreur durant la phase Validation.
    """


class ExecutionError(DiscoveryError):
    """
    Erreur durant la phase Execution.
    """


class PortfolioError(DiscoveryError):
    """
    Erreur portefeuille.
    """


class KnowledgeError(DiscoveryError):
    """
    Erreur base de connaissance.
    """


class MemoryError(DiscoveryError):
    """
    Erreur mémoire du moteur.
    """
