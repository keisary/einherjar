"""
==========================================================
EINHERJAR Domain Model
==========================================================

Toutes les entités métier manipulées par le moteur de
recherche.

Le Domain Model constitue le langage commun utilisé par
l'ensemble des phases du pipeline.

Phase A
↓

Phase B
↓

Phase C
↓

Phase D
↓

Phase E

Aucune structure métier importante ne doit être représentée
par un dictionnaire anonyme.

Toutes les entités doivent être définies ici sous forme de
dataclasses fortement typées.
"""

from .enums import *

__all__ = [
    "FeatureType",
    "EconomicFamily",
    "ConditionOperator",
    "HypothesisState",
    "CandidateState",
    "ValidationStatus",
    "TradeDirection",
    "TradeExitReason",
    "EinherState",
    "PortfolioState",
    "KnowledgeRelation",
]