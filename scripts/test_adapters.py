"""Test rapide des adaptateurs live sans appels API reels.

Valide les imports, conversions de symboles, et structure des adaptateurs.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from einherjar.brokers.broker_utils import (
    denormalize_symbol,
    load_fees,
    normalize_symbol,
    now_utc_ms,
    ohlcv_to_polars,
    timeframe_to_ccxt,
    timeframe_to_oanda_granularity,
)

print("=== Test broker_utils ===")

# Normalisation symboles
assert normalize_symbol("BTCUSD", "binance") == "BTC/USDT"
assert normalize_symbol("AAPL", "alpaca") == "AAPL"
assert normalize_symbol("EURUSD", "oanda") == "EUR/USD"
assert normalize_symbol("SP500", "oanda") == "SPX500_USD"
print("  normalize_symbol: OK")

# Denormalisation
assert denormalize_symbol("BTC/USDT", "binance") == "BTCUSD"
assert denormalize_symbol("EUR/USD", "oanda") == "EURUSD"
print("  denormalize_symbol: OK")

# Timeframes
assert timeframe_to_ccxt("15m") == "15m"
assert timeframe_to_oanda_granularity("1h") == "H1"
assert timeframe_to_oanda_granularity("1d") == "D"
print("  timeframe conversions: OK")

# OHLCV to polars
import polars as pl

ohlcv_raw = [
    [1000000, 100.0, 101.0, 99.0, 100.5, 1000.0],
    [1000060, 100.5, 102.0, 100.0, 101.5, 1500.0],
]
df = ohlcv_to_polars(ohlcv_raw)
assert df.shape == (2, 6)
assert df.columns == ["timestamp", "open", "high", "low", "close", "volume"]
print("  ohlcv_to_polars: OK")

# Empty OHLCV
df_empty = ohlcv_to_polars([])
assert df_empty.shape == (0, 6)
print("  ohlcv_to_polars (empty): OK")

# Load fees
fees = load_fees("binance")
assert "default" in fees
print("  load_fees: OK")

# now_utc_ms
ts = now_utc_ms()
assert isinstance(ts, int) and ts > 0
print("  now_utc_ms: OK")

# Test instanciation adaptateurs (sans cles API)
print("\n=== Test instanciation adaptateurs ===")

from einherjar.brokers.binance_adapter import BinanceAdapter

ba = BinanceAdapter(testnet=True)
assert ba.name == "binance"
assert ba.testnet is True
print("  BinanceAdapter: OK")

from einherjar.brokers.alpaca_adapter import AlpacaAdapter

try:
    aa = AlpacaAdapter("dummy_key", "dummy_secret", paper=True)
    assert aa.name == "alpaca"
    print("  AlpacaAdapter: OK")
except ImportError:
    print("  AlpacaAdapter: SKIP (alpaca-trade-api non installe)")

from einherjar.brokers.oanda_adapter import OandaAdapter

try:
    oa = OandaAdapter("dummy_account", "dummy_token", practice=True)
    assert oa.name == "oanda"
    print("  OandaAdapter: OK")
except ImportError:
    print("  OandaAdapter: SKIP (oandapyV20 non installe)")

print("\n=== Tous les tests passes ===")
