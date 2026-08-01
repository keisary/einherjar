"""
==========================================================
Feature Taxonomy
==========================================================

Définit la taxonomie officielle des features utilisées par
le moteur EINHERJAR.

Chaque feature appartient obligatoirement à :

1. Un FeatureType
   -> Nature de la feature

2. Une EconomicFamily
   -> Phénomène de marché mesuré

Cette double classification est utilisée par le Discovery
Engine pour équilibrer intelligemment la recherche.
"""

from enum import Enum


# ==========================================================
# FEATURE TYPE
# ==========================================================

class FeatureType(str, Enum):
    """
    Nature intrinsèque de la feature.
    """

    ATOMIC = "atomic"
    QUANTITATIVE = "quantitative"
    PATTERN = "pattern"
    COMPOSITE = "composite"


# ==========================================================
# ECONOMIC FAMILY
# ==========================================================

class EconomicFamily(str, Enum):
    """
    Phénomène de marché principalement mesuré.
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
# FEATURE GROUPS
# ==========================================================

FEATURE_TYPE_GROUPS = {
    FeatureType.ATOMIC: [],
    FeatureType.QUANTITATIVE: [],
    FeatureType.PATTERN: [],
    FeatureType.COMPOSITE: [],
}


ECONOMIC_FAMILY_GROUPS = {
    EconomicFamily.MOMENTUM: [],
    EconomicFamily.TREND: [],
    EconomicFamily.VOLATILITY: [],
    EconomicFamily.VOLUME_FLOW: [],
    EconomicFamily.MARKET_STRUCTURE: [],
    EconomicFamily.PRICE_ACTION: [],
    EconomicFamily.MARKET_REGIME: [],
    EconomicFamily.RISK: [],
    EconomicFamily.STATISTICAL: [],
    EconomicFamily.MICROSTRUCTURE: [],
    EconomicFamily.SENTIMENT: [],
    EconomicFamily.CROSS_ASSET: [],
    EconomicFamily.OTHER: [],
}