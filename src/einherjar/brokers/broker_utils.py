"""Utilitaires communs pour l'adaptateur cTrader.

Helpers partages : conversion de symboles MIDAS vers cTrader,
retry avec backoff, formatage des DataFrames polars,
gestion d'erreurs, mapping des timeframes.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

import polars as pl

T = TypeVar("T")

FEES_DIR = Path(__file__).resolve().parents[2] / "config"

# ---------------------------------------------------------------------------
# Mapping symboles MIDAS -> cTrader (conventions IC Markets cTrader par defaut)
# ---------------------------------------------------------------------------
MIDAS_TO_CTRADER_DEFAULT: dict[str, str] = {
    # Forex (8)
    "EURUSD": "EURUSD",
    "GBPUSD": "GBPUSD",
    "USDJPY": "USDJPY",
    "AUDUSD": "AUDUSD",
    "USDCAD": "USDCAD",
    "USDCHF": "USDCHF",
    "EURGBP": "EURGBP",
    "NZDUSD": "NZDUSD",
    # Crypto (5)
    "BTCUSD": "BTCUSD",
    "ETHUSD": "ETHUSD",
    "ADAUSD": "ADAUSD",
    "BCHUSD": "BCHUSD",
    "LTCUSD": "LTCUSD",
    # Actions US (8)
    "AAPL": "AAPL",
    "MSFT": "MSFT",
    "NVDA": "NVDA",
    "AMZN": "AMZN",
    "GOOGL": "GOOGL",
    "TSLA": "TSLA",
    "JPM": "JPM",
    "XOM": "XOM",
    # Indices (4)
    "SP500": "US500",
    "NASDAQ100": "US100",
    "DOWJONES": "US30",
    "DAX40": "DE40",
    # Commodities + Metals (4)
    "XAUUSD": "XAUUSD",
    "WTIUSD": "USOUSD",
    "BRENT": "UKOUSD",
    "COPPER": "XCUUSD",
}

# Overrides par broker si le broker utilise un nommage different
MIDAS_TO_CTRADER_IC_MARKETS: dict[str, str] = {
    # Identique au default pour IC Markets cTrader
}

MIDAS_TO_CTRADER_PEPPERSTONE: dict[str, str] = {
    # Pepperstone cTrader utilise generalement les memes noms
    # mais certaines actions peuvent avoir un prefixe "US."
    "AAPL": "US.AAPL",
    "MSFT": "US.MSFT",
    "NVDA": "US.NVDA",
    "AMZN": "US.AMZN",
    "GOOGL": "US.GOOGL",
    "TSLA": "US.TSLA",
    "JPM": "US.JPM",
    "XOM": "US.XOM",
}

# Compte demo Spotware (constate le 2026-09-19, `ProtoOATrader.brokerName = "Spotware"`,
# 830 symboles) : les actions ET les indices sont servis sous leur NOM COMPLET, pas
# sous le ticker. Un ticker envoye tel quel est absent du broker -> aucun prix.
# Mesure : 10/28 actifs du corpus avec le mapping par defaut, 24/28 avec celui-ci.
MIDAS_TO_CTRADER_SPOTWARE: dict[str, str] = {
    # Actions (nom complet)
    "AAPL": "APPLE",
    "MSFT": "MICROSOFT",
    "NVDA": "NVIDIA",
    "AMZN": "AMAZON",
    "GOOGL": "ALPHABET",
    "TSLA": "TESLA MOTORS",
    "JPM": "JPMORGAN",
    "XOM": "EXXON",
    # Indices (noms cTrader de ce fournisseur)
    "SP500": "US 500",
    "NASDAQ100": "US TECH 100",
    "DOWJONES": "US 30",
    "DAX40": "GERMANY 40",
    # Energies (WTI = XTIUSD, Brent = XBRUSD ; USOUSD/UKOUSD n'existent pas ici)
    "WTIUSD": "XTIUSD",
    "BRENT": "XBRUSD",
}

# Mapping par broker. La cle est comparee en minuscules : "ic_markets", "pepperstone",
# "spotware". Un broker inconnu retombe sur le mapping par defaut.
MAPS_PAR_BROKER: dict[str, dict[str, str]] = {
    "ic_markets": MIDAS_TO_CTRADER_IC_MARKETS,
    "pepperstone": MIDAS_TO_CTRADER_PEPPERSTONE,
    "spotware": MIDAS_TO_CTRADER_SPOTWARE,
}


def _mapping_pour_broker(broker: str) -> dict[str, str]:
    """Retourne le mapping MIDAS -> cTrader d'un broker (defaut inclus).

    Args:
        broker: Nom du broker ("ic_markets", "pepperstone", "spotware", ...).

    Returns:
        Le mapping complet (defaut + overrides du broker).
    """
    overrides = MAPS_PAR_BROKER.get(str(broker or "").strip().lower(), {})
    return {**MIDAS_TO_CTRADER_DEFAULT, **overrides}

# ---------------------------------------------------------------------------
# Asset class mapping (centralise ici pour eviter la duplication)
# ---------------------------------------------------------------------------
from einherjar.core.enums import AssetClass  # noqa: E402  (apres le header docstring/bloc)

ASSET_CLASS_MAP: dict[str, AssetClass] = {
    "BTCUSD": AssetClass.CRYPTO,
    "ETHUSD": AssetClass.CRYPTO,
    "ADAUSD": AssetClass.CRYPTO,
    "BCHUSD": AssetClass.CRYPTO,
    "LTCUSD": AssetClass.CRYPTO,
    "EURUSD": AssetClass.FOREX,
    "GBPUSD": AssetClass.FOREX,
    "USDJPY": AssetClass.FOREX,
    "AUDUSD": AssetClass.FOREX,
    "USDCAD": AssetClass.FOREX,
    "USDCHF": AssetClass.FOREX,
    "EURGBP": AssetClass.FOREX,
    "NZDUSD": AssetClass.FOREX,
    "XAUUSD": AssetClass.METAL,
    "AAPL": AssetClass.STOCK_US,
    "MSFT": AssetClass.STOCK_US,
    "NVDA": AssetClass.STOCK_US,
    "AMZN": AssetClass.STOCK_US,
    "GOOGL": AssetClass.STOCK_US,
    "TSLA": AssetClass.STOCK_US,
    "JPM": AssetClass.STOCK_US,
    "XOM": AssetClass.STOCK_US,
    "SP500": AssetClass.INDEX,
    "NASDAQ100": AssetClass.INDEX,
    "DOWJONES": AssetClass.INDEX,
    "DAX40": AssetClass.INDEX,
    "WTIUSD": AssetClass.COMMODITY,
    "BRENT": AssetClass.COMMODITY,
    "COPPER": AssetClass.COMMODITY,
}

# ---------------------------------------------------------------------------
# Helpers de conversion
# ---------------------------------------------------------------------------

def normalize_symbol(asset: str, broker: str = "ic_markets") -> str:
    """Convertit un symbole MIDAS en symbole cTrader.

    Args:
        asset: Symbole MIDAS (ex: "BTCUSD").
        broker: Nom du broker ("ic_markets", "pepperstone").

    Returns:
        Symbole normalise pour cTrader.
    """
    return _mapping_pour_broker(broker).get(asset, asset)


def denormalize_symbol(broker_symbol: str, broker: str = "ic_markets") -> str:
    """Convertit un symbole cTrader en symbole MIDAS.

    Args:
        broker_symbol: Symbole cTrader.
        broker: Nom du broker.

    Returns:
        Symbole MIDAS.
    """
    mapping = _mapping_pour_broker(broker)
    reverse = {v: k for k, v in mapping.items()}
    return reverse.get(broker_symbol, broker_symbol)


# `ProtoOATrendbarPeriod` (paquet ctrader-open-api) n'est PAS une duree en minutes :
# c'est une enumeration ordinale ou M5=5, M10=6, M15=7, M30=8, H1=9, H4=10, H12=11,
# D1=12. Envoyer des minutes ne tombe juste que pour 5m (M5=5) et leve
# `ValueError: invalid enumerator` pour 15m (15), 1h (60), 4h (240) et 1d (1440).
# Verifie le 2026-09-19 contre le paquet installe : les 5 valeurs ci-dessous sont
# acceptees et `ProtoOAGetTrendbarsRes.period` renvoie exactement la valeur demandee.
TF_TO_CTRADER_PERIOD: dict[str, int] = {
    "5m": 5,  # ProtoOATrendbarPeriod.M5
    "15m": 7,  # ProtoOATrendbarPeriod.M15
    "1h": 9,  # ProtoOATrendbarPeriod.H1
    "4h": 10,  # ProtoOATrendbarPeriod.H4
    "1d": 12,  # ProtoOATrendbarPeriod.D1
}


def timeframe_to_ctrader_period(tf: str) -> int:
    """Convertit un timeframe EINHERJAR en `ProtoOATrendbarPeriod` cTrader.

    Args:
        tf: Timeframe ("5m", "15m", "1h", "4h", "1d").

    Returns:
        Valeur de l'enumeration `ProtoOATrendbarPeriod` (5, 7, 9, 10, 12) —
        PAS des minutes.

    Raises:
        ValueError: Si le timeframe n'est pas supporte.
    """
    if tf not in TF_TO_CTRADER_PERIOD:
        raise ValueError(f"Timeframe cTrader non supporte: {tf}")
    return TF_TO_CTRADER_PERIOD[tf]


def ctrader_period_to_timeframe(period: int) -> str:
    """Convertit une `ProtoOATrendbarPeriod` en timeframe EINHERJAR.

    Args:
        period: Valeur de l'enumeration `ProtoOATrendbarPeriod`.

    Returns:
        Timeframe EINHERJAR, ou `period_<n>` si la valeur est hors du corpus.
    """
    inverse = {v: k for k, v in TF_TO_CTRADER_PERIOD.items()}
    return inverse.get(int(period), f"period_{int(period)}")


# ---------------------------------------------------------------------------
# Helpers generiques
# ---------------------------------------------------------------------------

def ohlcv_to_polars(data: list[list[float | int]]) -> pl.DataFrame:
    """Convertit un tableau OHLCV brut en DataFrame polars.

    Args:
        data: Liste de [timestamp, open, high, low, close, volume].

    Returns:
        DataFrame polars avec colonnes nommees.
    """
    if not data:
        return pl.DataFrame(
            schema={
                "timestamp": pl.Int64,
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Float64,
            }
        )
    return pl.DataFrame(
        {
            "timestamp": [int(row[0]) for row in data],
            "open": [float(row[1]) for row in data],
            "high": [float(row[2]) for row in data],
            "low": [float(row[3]) for row in data],
            "close": [float(row[4]) for row in data],
            "volume": [float(row[5]) for row in data],
        }
    )


def load_fees(broker_name: str = "ctrader") -> dict[str, Any]:
    """Charge la configuration de frais depuis le fichier JSON.

    Args:
        broker_name: Nom du broker (defaut "ctrader").

    Returns:
        Dict des frais avec cle 'default' et optionnellement 'per_symbol'.
    """
    path = FEES_DIR / f"fees_{broker_name}.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {"default": {"spread_pct": 0.0, "commission_per_lot": 0.0, "swap_long": 0.0, "swap_short": 0.0}}


async def retry_with_backoff(
    func: Callable[[], T],
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> T:
    """Execute une fonction avec retry et backoff exponentiel.

    Args:
        func: Fonction a executer (peut etre async ou sync).
        max_retries: Nombre maximum de tentatives.
        base_delay: Delai de base en secondes.
        max_delay: Delai maximum en secondes.
        exceptions: Tuple d'exceptions a capturer.

    Returns:
        Resultat de la fonction.

    Raises:
        La derniere exception si toutes les tentatives echouent.
    """
    for attempt in range(max_retries):
        try:
            if asyncio.iscoroutinefunction(func):
                return await func()
            return func()  # pyright: ignore[reportReturnType]
        except exceptions:
            if attempt == max_retries - 1:
                raise
            delay = min(base_delay * (2 ** attempt), max_delay)
            await asyncio.sleep(delay)
    raise RuntimeError("retry_with_backoff: unreachable")


def now_utc_ms() -> int:
    """Retourne le timestamp UTC actuel en millisecondes."""
    return int(datetime.now(UTC).timestamp() * 1000)
