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
    mapping: dict[str, str]
    if broker == "pepperstone":
        mapping = {**MIDAS_TO_CTRADER_DEFAULT, **MIDAS_TO_CTRADER_PEPPERSTONE}
    elif broker == "ic_markets":
        mapping = {**MIDAS_TO_CTRADER_DEFAULT, **MIDAS_TO_CTRADER_IC_MARKETS}
    else:
        mapping = MIDAS_TO_CTRADER_DEFAULT
    return mapping.get(asset, asset)


def denormalize_symbol(broker_symbol: str, broker: str = "ic_markets") -> str:
    """Convertit un symbole cTrader en symbole MIDAS.

    Args:
        broker_symbol: Symbole cTrader.
        broker: Nom du broker.

    Returns:
        Symbole MIDAS.
    """
    mapping: dict[str, str]
    if broker == "pepperstone":
        mapping = {**MIDAS_TO_CTRADER_DEFAULT, **MIDAS_TO_CTRADER_PEPPERSTONE}
    elif broker == "ic_markets":
        mapping = {**MIDAS_TO_CTRADER_DEFAULT, **MIDAS_TO_CTRADER_IC_MARKETS}
    else:
        mapping = MIDAS_TO_CTRADER_DEFAULT
    reverse = {v: k for k, v in mapping.items()}
    return reverse.get(broker_symbol, broker_symbol)


def timeframe_to_ctrader_period(tf: str) -> int:
    """Convertit un timeframe EINHERJAR en period cTrader (minutes).

    Args:
        tf: Timeframe ("5m", "15m", "1h", "4h", "1d").

    Returns:
        Periode cTrader en minutes (5, 15, 60, 240, 1440).

    Raises:
        ValueError: Si le timeframe n'est pas supporte.
    """
    mapping = {
        "5m": 5,
        "15m": 15,
        "1h": 60,
        "4h": 240,
        "1d": 1440,
    }
    if tf not in mapping:
        raise ValueError(f"Timeframe cTrader non supporte: {tf}")
    return mapping[tf]


def ctrader_period_to_timeframe(period: int) -> str:
    """Convertit une periode cTrader en timeframe EINHERJAR.

    Args:
        period: Periode en minutes.

    Returns:
        Timeframe EINHERJAR.
    """
    mapping = {
        5: "5m",
        15: "15m",
        60: "1h",
        240: "4h",
        1440: "1d",
    }
    return mapping.get(period, f"{period}m")


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
            return func()
        except exceptions:
            if attempt == max_retries - 1:
                raise
            delay = min(base_delay * (2 ** attempt), max_delay)
            await asyncio.sleep(delay)
    raise RuntimeError("retry_with_backoff: unreachable")


def now_utc_ms() -> int:
    """Retourne le timestamp UTC actuel en millisecondes."""
    return int(datetime.now(UTC).timestamp() * 1000)
