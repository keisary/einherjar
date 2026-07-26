"""
==========================================================
Engine Exceptions
==========================================================

Exceptions métier du moteur de découverte.
"""

from __future__ import annotations


class DiscoveryError(Exception):
    """
    Classe mère de toutes les exceptions métier.
    """


# ==========================================================
# CONFIGURATION
# ==========================================================

class ConfigurationError(DiscoveryError):
    """
    Erreur de configuration.
    """


# ==========================================================
# DATASET
# ==========================================================

class DatasetError(DiscoveryError):
    """
    Erreur liée au dataset.
    """


class DatasetValidationError(DatasetError):
    """
    Dataset invalide.
    """


# ==========================================================
# DISCOVERY
# ==========================================================

class SearchError(DiscoveryError):
    """
    Erreur durant la découverte.
    """


# ==========================================================
# VALIDATION
# ==========================================================

class ValidationError(DiscoveryError):
    """
    Erreur durant la validation.
    """


# ==========================================================
# EXECUTION
# ==========================================================

class ExecutionError(DiscoveryError):
    """
    Erreur durant l'exécution.
    """


# ==========================================================
# PORTFOLIO
# ==========================================================

class PortfolioError(DiscoveryError):
    """
    Erreur portefeuille.
    """


# ==========================================================
# KNOWLEDGE
# ==========================================================

class KnowledgeError(DiscoveryError):
    """
    Erreur base de connaissance.
    """


# ==========================================================
# MEMORY
# ==========================================================

class MemoryError(DiscoveryError):
    """
    Erreur mémoire du moteur.
    """