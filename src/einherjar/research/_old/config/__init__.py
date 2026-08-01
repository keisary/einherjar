"""
==========================================================
EINHERJAR Research Configuration Package
==========================================================

Centralise toute la configuration du moteur de recherche.

Ce package ne contient aucune logique métier.
Il expose uniquement les différents modules de configuration.
"""

from .config import Config

__all__ = ["Config"]