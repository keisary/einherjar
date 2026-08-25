"""Enumeration centralisee pour tous les etats, types et directions du systeme."""

from enum import StrEnum


class EinherState(StrEnum):
    """Etats possibles d'un Einher en temps reel."""

    IDLE = "idle"
    FORMING = "forming"
    TRIGGERED = "triggered"
    IN_POSITION = "in_position"
    COOLDOWN = "cooldown"
    PROBATION = "probation"
    DISABLED = "disabled"


class Direction(StrEnum):
    """Direction d'un signal ou d'une position."""

    LONG = "long"
    SHORT = "short"
    BOTH = "both"
    HOLD = "hold"


class OrderType(StrEnum):
    """Types d'ordre supportes par le systeme."""

    MARKET = "market"
    LIMIT = "limit"
    STOP_MARKET = "stop_market"
    BRACKET_OCO = "bracket_oco"


class TimeFrame(StrEnum):
    """Timeframes actifs du systeme."""

    M5 = "5m"
    M15 = "15m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"


class AssetClass(StrEnum):
    """Classes d'actifs couvertes par EINHERJAR."""

    CRYPTO = "crypto"
    STOCK_US = "stock_us"
    STOCK_EU = "stock_eu"
    STOCK_ASIA = "stock_asia"
    FOREX = "forex"
    METAL = "metal"
    INDEX = "index"
    COMMODITY = "commodity"


class RejectionReason(StrEnum):
    """Raisons de rejet d'une intention de trade par le Risk Manager."""

    EXPOSURE_TOTAL = "exposure_total"
    EXPOSURE_ASSET = "exposure_asset"
    EXPOSURE_CLASS = "exposure_class"
    MAX_POSITIONS = "max_positions"
    CORRELATION = "correlation"
    DAILY_LOSS = "daily_loss"
    DRAWDOWN = "drawdown"
    WEEKLY_LOSS = "weekly_loss"
    MIN_SIZE = "min_size"
    CONFIDENCE_TOO_LOW = "confidence_too_low"
    MARGIN = "margin"
