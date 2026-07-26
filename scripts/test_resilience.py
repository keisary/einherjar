"""Test du module de resilience sans appels API reels.

Valide le circuit breaker, le rate limiter, et le wrapper ResilientBroker.
"""

import sys
import asyncio
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from einherjar.brokers.resilience import (
    CircuitBreaker,
    CircuitBreakerConfig,
    RateLimiter,
    RateLimitConfig,
    ResilientBroker,
)

print("=== Test CircuitBreaker ===")

cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=3, recovery_timeout=1.0))
assert cb.state == "CLOSED"
assert cb.can_execute() is True

# Simuler des echecs
for _ in range(3):
    cb.record_failure()

assert cb.state == "OPEN"
assert cb.can_execute() is False
print("  Circuit breaker ouvert apres 3 failures: OK")

# Attendre recovery
import time
time.sleep(1.1)
assert cb.can_execute() is True
assert cb.state == "HALF_OPEN"
print("  Circuit breaker half-open apres timeout: OK")

# Succes en half-open
cb.record_success()
cb.record_success()
cb.record_success()
assert cb.state == "CLOSED"
print("  Circuit breaker ferme apres recovery: OK")

print("\n=== Test RateLimiter ===")

rl = RateLimiter(RateLimitConfig(max_calls_per_second=5, max_calls_per_minute=100))

async def test_rate():
    for i in range(10):
        await rl.acquire()
    print("  10 appels limites: OK")

asyncio.run(test_rate())

print("\n=== Test ResilientBroker ===")

class DummyAdapter:
    name = "dummy"
    def __init__(self, max_failures=2):
        self.call_count = 0
        self.max_failures = max_failures
    async def get_ohlcv(self, asset, timeframe, since=None, limit=500):
        self.call_count += 1
        if self.call_count <= self.max_failures:
            raise RuntimeError("Simulated failure")
        return {"asset": asset, "timeframe": timeframe}
    def get_fees(self, asset):
        return {"maker": 0.001}

async def test_resilient():
    dummy = DummyAdapter(max_failures=2)
    rb = ResilientBroker(
        dummy,
        circuit_config=CircuitBreakerConfig(failure_threshold=2, recovery_timeout=0.5),
    )

    # 2 echecs -> circuit ouvert
    for i in range(2):
        try:
            await rb.get_ohlcv("BTCUSD", "15m")
        except RuntimeError:
            pass

    status = rb.get_status()
    assert status["circuit_state"] == "OPEN"
    print("  ResilientBroker circuit ouvert apres 2 failures: OK")

    # Attendre recovery (half-open puis succes)
    await asyncio.sleep(0.6)
    # 3 succes en half-open pour fermer le circuit
    for _ in range(3):
        result = await rb.get_ohlcv("BTCUSD", "15m")
        assert result == {"asset": "BTCUSD", "timeframe": "15m"}
    print("  ResilientBroker recovery et succes: OK")

    # Verifier que le circuit est ferme
    status = rb.get_status()
    assert status["circuit_state"] == "CLOSED"
    print("  ResilientBroker circuit ferme apres succes: OK")
    await asyncio.sleep(0.6)
    result = await rb.get_ohlcv("BTCUSD", "15m")
    assert result == {"asset": "BTCUSD", "timeframe": "15m"}
    print("  ResilientBroker recovery et succes: OK")

    # Verifier que le circuit est ferme
    status = rb.get_status()
    assert status["circuit_state"] == "CLOSED"
    print("  ResilientBroker circuit ferme apres succes: OK")

asyncio.run(test_resilient())

print("\n=== Tous les tests resilience passes ===")
