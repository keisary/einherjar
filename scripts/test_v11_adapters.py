import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from einherjar.brokers.binance_futures_adapter import BinanceFuturesAdapter
from einherjar.brokers.cfd_adapter import CfdAdapter, SUPPORTED_CFD_BROKERS

print("BinanceFuturesAdapter: OK")
print("CfdAdapter: OK")
print(f"Brokers CFD supportes: {list(SUPPORTED_CFD_BROKERS.keys())}")
