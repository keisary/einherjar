"""
==========================================================
EINHERJAR - Domain Enumerations
==========================================================

Toutes les énumérations officielles du Domain Model.

Aucun autre module ne doit définir d'Enum.

Ce fichier constitue la référence unique pour tous les états,
catégories et comportements manipulés par le moteur.
"""

from enum import Enum


# ==========================================================
# FEATURE TAXONOMY
# ==========================================================

class FeatureType(str, Enum):
    """
    Nature intrinsèque d'une feature.
    """

    ATOMIC = "atomic"
    QUANTITATIVE = "quantitative"
    PATTERN = "pattern"
    COMPOSITE = "composite"


class EconomicFamily(str, Enum):
    """
    Phénomène économique principalement mesuré.
    """

    MOMENTUM = "momentum"
    TREND = "trend"
    VOLATILITY = "volatility"
    VOLUME_FLOW = "volume_flow"
    MARKET_STRUCTURE = "market_structure"
    PRICE_ACTION = "price_action"
    MARKET_REGIME = "market_regime"
    RISK = "risk"
    STATISTICAL = "statistical"
    MICROSTRUCTURE = "microstructure"
    SENTIMENT = "sentiment"
    CROSS_ASSET = "cross_asset"
    OTHER = "other"


# ==========================================================
# FEATURE DATA TYPE
# ==========================================================

class FeatureValueType(str, Enum):
    """
    Nature des valeurs manipulées.
    """

    FLOAT = "float"

    INTEGER = "integer"

    BOOLEAN = "boolean"

    CATEGORICAL = "categorical"

    ORDINAL = "ordinal"


# ==========================================================
# CONDITIONS
# ==========================================================

class ConditionOperator(str, Enum):
    """
    Opérateurs supportés par le moteur.
    """

    # -----------------------------
    # Comparaison
    # -----------------------------

    LT = "<"

    LE = "<="

    GT = ">"

    GE = ">="

    EQ = "=="

    NE = "!="

    # -----------------------------
    # Intervalle
    # -----------------------------

    BETWEEN = "between"

    NOT_BETWEEN = "not_between"

    # -----------------------------
    # Catégoriel
    # -----------------------------

    IN = "in"

    NOT_IN = "not_in"

    # -----------------------------
    # Booléen
    # -----------------------------

    IS_TRUE = "is_true"

    IS_FALSE = "is_false"

    # -----------------------------
    # Changement d'état
    # -----------------------------

    CHANGED_TO = "changed_to"

    CHANGED_FROM = "changed_from"

    # -----------------------------
    # Cross
    # -----------------------------

    CROSS_OVER = "cross_over"

    CROSS_UNDER = "cross_under"

    # -----------------------------
    # Pente
    # -----------------------------

    RISING = "rising"

    FALLING = "falling"

    ACCELERATING = "accelerating"

    DECELERATING = "decelerating"


# ==========================================================
# DISCOVERY
# ==========================================================

class SearchStrategy(str, Enum):
    """
    Type d'exploration.
    """

    RANDOM = "random"

    GREEDY = "greedy"

    BEAM = "beam"

    NOVELTY = "novelty"

    EXPLORATION = "exploration"

    EXPLOITATION = "exploitation"

    FAMILY_BALANCED = "family_balanced"


class SearchRegionState(str, Enum):
    """
    Etat d'une région de recherche.
    """

    UNKNOWN = "unknown"

    PROMISING = "promising"

    EXPLORED = "explored"

    SATURATED = "saturated"

    DEAD = "dead"

    ARCHIVED = "archived"


class DiscoveryOutcome(str, Enum):
    """
    Résultat d'une exploration.
    """

    NEW_EDGE = "new_edge"

    EXPANDED = "expanded"

    DUPLICATE = "duplicate"

    DOMINATED = "dominated"

    SATURATED = "saturated"

    DEAD_END = "dead_end"

    PRUNED = "pruned"


# ==========================================================
# HYPOTHESIS
# ==========================================================

class HypothesisState(str, Enum):

    NEW = "new"

    GENERATED = "generated"

    EXPLORED = "explored"

    EXPANDED = "expanded"

    CANDIDATE = "candidate"

    REJECTED = "rejected"

    ARCHIVED = "archived"


# ==========================================================
# CANDIDATE
# ==========================================================

class CandidateState(str, Enum):

    CREATED = "created"

    VALIDATING = "validating"

    VALIDATED = "validated"

    REJECTED = "rejected"


# ==========================================================
# VALIDATION
# ==========================================================

class ValidationStatus(str, Enum):

    PENDING = "pending"

    PASSED = "passed"

    FAILED = "failed"

    WARNING = "warning"


# ==========================================================
# EXECUTION
# ==========================================================

class ExecutionStatus(str, Enum):

    PENDING = "pending"

    RUNNING = "running"

    COMPLETED = "completed"

    FAILED = "failed"


# ==========================================================
# TRADING
# ==========================================================

class TradeDirection(str, Enum):

    LONG = "long"

    SHORT = "short"


class TradeExitReason(str, Enum):

    TAKE_PROFIT = "take_profit"

    STOP_LOSS = "stop_loss"

    TIME_EXIT = "time_exit"

    SIGNAL_EXIT = "signal_exit"

    MANUAL_EXIT = "manual_exit"

    END_OF_DATASET = "end_of_dataset"

    UNKNOWN = "unknown"


# ==========================================================
# EINHER
# ==========================================================

class EinherState(str, Enum):

    VALIDATED = "validated"

    PROFILED = "profiled"

    READY = "ready"

    ACTIVE = "active"

    DEGRADED = "degraded"

    RETIRED = "retired"

    ARCHIVED = "archived"


# ==========================================================
# PORTFOLIO
# ==========================================================

class PortfolioState(str, Enum):

    BUILDING = "building"

    READY = "ready"

    ACTIVE = "active"

    REBALANCING = "rebalancing"

    ARCHIVED = "archived"


# ==========================================================
# KNOWLEDGE GRAPH
# ==========================================================

class KnowledgeRelation(str, Enum):

    SIMILAR_TO = "similar_to"

    PARENT_OF = "parent_of"

    CHILD_OF = "child_of"

    SPECIALIZES = "specializes"

    GENERALIZES = "generalizes"

    COMPLEMENTS = "complements"

    CONTRADICTS = "contradicts"

    CORRELATED = "correlated"

    UNCORRELATED = "uncorrelated"

    SUBSTITUTES = "substitutes"


# ==========================================================
# MEMORY
# ==========================================================

class MemoryRegionState(str, Enum):

    UNKNOWN = "unknown"

    PROMISING = "promising"

    SATURATED = "saturated"

    DEAD = "dead"


class MemoryEvent(str, Enum):
    """
    Type d'information enregistrée dans la mémoire.
    """

    DISCOVERY = "discovery"

    VALIDATION = "validation"

    EXECUTION = "execution"

    REJECTION = "rejection"

    EXPANSION = "expansion"

    PORTFOLIO = "portfolio"


# ==========================================================
# EXPORT
# ==========================================================

class ExportFormat(str, Enum):

    JSON = "json"

    PARQUET = "parquet"

    CSV = "csv"


# ==========================================================
# RUNTIME
# ==========================================================

class RunStatus(str, Enum):

    CREATED = "created"

    RUNNING = "running"

    PAUSED = "paused"

    COMPLETED = "completed"

    FAILED = "failed"

    CANCELLED = "cancelled"