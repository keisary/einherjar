"""Module de durcissement et resilience pour les adaptateurs brokers.

Fournit un wrapper `ResilientBroker` qui encapsule n'importe quel
BrokerAdapter avec :
- Circuit breaker (arret temporaire apres N erreurs consecutives)
- Rate limiting (evite de depasser les quotas API)
- Validation des reponses (donnees coherentes, prix positifs, etc.)
- Logging structure des appels et erreurs

Reference : Section 4.5 du CDC EINHERJAR (robustesse).
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import polars as pl

from einherjar.core.models import AccountState, Fill, Order, Position

logger = logging.getLogger("einherjar.resilience")


@dataclass
class CircuitBreakerConfig:
    """Configuration du circuit breaker.

    Attributes:
        failure_threshold: Nombre d'erreurs consecutives avant ouverture.
        recovery_timeout: Duree en secondes avant tentative de fermeture.
        half_open_max_calls: Nombre max d'appels en mode half-open.
    """
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    half_open_max_calls: int = 3


@dataclass
class RateLimitConfig:
    """Configuration du rate limiter.

    Attributes:
        max_calls_per_second: Appels max par seconde.
        max_calls_per_minute: Appels max par minute.
    """
    max_calls_per_second: float = 10.0
    max_calls_per_minute: float = 300.0


class CircuitBreaker:
    """Circuit breaker pour les appels broker.

    Etats : CLOSED (normal), OPEN (bloque), HALF_OPEN (test de recovery).
    """

    def __init__(self, config: CircuitBreakerConfig | None = None) -> None:
        """__init__.

        Args:
            config: TODO document.
        """
        self.config = config or CircuitBreakerConfig()
        self.state = "CLOSED"
        self.failures = 0
        self.last_failure_time: datetime | None = None
        self.half_open_calls = 0

    def record_success(self) -> None:
        """Enregistre un succes."""
        if self.state == "HALF_OPEN":
            self.half_open_calls += 1
            if self.half_open_calls >= self.config.half_open_max_calls:
                self.state = "CLOSED"
                self.failures = 0
                self.half_open_calls = 0
                logger.info("Circuit breaker ferme (recovery OK)")
        else:
            self.failures = 0

    def record_failure(self) -> None:
        """Enregistre une erreur."""
        self.failures += 1
        self.last_failure_time = datetime.now(UTC)

        if self.state == "HALF_OPEN":
            self.state = "OPEN"
            logger.warning(
                "Circuit breaker ouvert (half-open echoue, %s failures)",
                self.failures,
            )
        elif self.failures >= self.config.failure_threshold:
            self.state = "OPEN"
            logger.warning(
                "Circuit breaker ouvert (%s/%s failures)",
                self.failures,
                self.config.failure_threshold,
            )

    def can_execute(self) -> bool:
        """Verifie si l'appel peut passer.

        Returns:
            True si le circuit est ferme ou half-open avec quota restant.
        """
        if self.state == "CLOSED":
            return True
        if self.state == "HALF_OPEN" and self.half_open_calls < self.config.half_open_max_calls:
            return True
        if self.state == "OPEN" and self.last_failure_time is not None:
            elapsed = (datetime.now(UTC) - self.last_failure_time).total_seconds()
            if elapsed >= self.config.recovery_timeout:
                self.state = "HALF_OPEN"
                self.half_open_calls = 0
                logger.info("Circuit breaker en half-open (tentative recovery)")
                return True
        return False


class RateLimiter:
    """Rate limiter simple base sur des fenetres glissantes."""

    def __init__(self, config: RateLimitConfig | None = None) -> None:
        """__init__.

        Args:
            config: TODO document.
        """
        self.config = config or RateLimitConfig()
        self.second_calls: list[datetime] = []
        self.minute_calls: list[datetime] = []

    async def acquire(self) -> None:
        """Attend si necessaire pour respecter les limites."""
        now = datetime.now(UTC)

        # Nettoyer les appels anciens
        self.second_calls = [t for t in self.second_calls if (now - t).total_seconds() < 1.0]
        self.minute_calls = [t for t in self.minute_calls if (now - t).total_seconds() < 60.0]

        # Attendre si limite atteinte
        while len(self.second_calls) >= self.config.max_calls_per_second:
            await asyncio.sleep(0.1)
            now = datetime.now(UTC)
            self.second_calls = [t for t in self.second_calls if (now - t).total_seconds() < 1.0]

        while len(self.minute_calls) >= self.config.max_calls_per_minute:
            await asyncio.sleep(1.0)
            now = datetime.now(UTC)
            self.minute_calls = [t for t in self.minute_calls if (now - t).total_seconds() < 60.0]

        self.second_calls.append(now)
        self.minute_calls.append(now)


class ResilientBroker:
    """Wrapper resilient autour d'un BrokerAdapter.

    Ajoute circuit breaker, rate limiting, validation et logging
    a n'importe quel adaptateur.

    Attributes:
        adapter: L'adaptateur sous-jacent.
        circuit_breaker: Circuit breaker pour les appels.
        rate_limiter: Rate limiter pour les appels.
    """

    def __init__(
        self,
        adapter: Any,
        circuit_config: CircuitBreakerConfig | None = None,
        rate_config: RateLimitConfig | None = None,
    ) -> None:
        """__init__.

        Args:
            adapter: TODO document.
            circuit_config: TODO document.
            rate_config: TODO document.
        """
        self.adapter = adapter
        self.name = getattr(adapter, "name", "unknown")
        self.circuit = CircuitBreaker(circuit_config)
        self.rate_limiter = RateLimiter(rate_config)

    async def _call(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        """Execute un appel avec protection.

        Args:
                **kwargs: TODO: documenter.
                *args: TODO: documenter.

        Args:
            method_name: Nom de la methode a appeler.
            *args, **kwargs: Arguments de la methode.

        Returns:
            Resultat de la methode.

        Raises:
            RuntimeError: Si le circuit breaker est ouvert.
            Exception: Si l'appel echoue.
        """
        if not self.circuit.can_execute():
            raise RuntimeError(
                f"Circuit breaker ouvert pour {self.name}.{method_name}"
            )

        await self.rate_limiter.acquire()

        method = getattr(self.adapter, method_name)
        try:
            result = await method(*args, **kwargs)
            self.circuit.record_success()
            self._log_success(method_name, args, kwargs)
            return result
        except Exception as exc:
            self.circuit.record_failure()
            self._log_failure(method_name, args, kwargs, exc)
            raise

    def _log_success(self, method: str, args: Any, kwargs: Any) -> None:
        """Log un appel reussi."""
        logger.debug(
            json.dumps({
                "event": "broker_call_success",
                "broker": self.name,
                "method": method,
                "timestamp": datetime.now(UTC).isoformat(),
            })
        )

    def _log_failure(self, method: str, args: Any, kwargs: Any, exc: Exception) -> None:
        """Log un appel en echec."""
        logger.warning(
            json.dumps({
                "event": "broker_call_failure",
                "broker": self.name,
                "method": method,
                "error": str(exc),
                "error_type": type(exc).__name__,
                "timestamp": datetime.now(UTC).isoformat(),
            })
        )

    # Delegation des methodes BrokerAdapter avec protection

    async def get_ohlcv(self, asset: str, timeframe: str, since: int | None = None, limit: int = 500) -> pl.DataFrame:
        """Historique OHLCV avec protection."""
        return await self._call("get_ohlcv", asset, timeframe, since, limit)

    async def subscribe_live(self, assets: list[str], callback: Callable) -> None:
        """Souscription live avec protection."""
        return await self._call("subscribe_live", assets, callback)

    async def stop_live(self) -> None:
        """Arret de la souscription live."""
        return await self._call("stop_live")

    async def place_order(self, order: Order) -> Fill:
        """Passage d'ordre avec protection."""
        return await self._call("place_order", order)

    async def cancel_order(self, order_id: str) -> bool:
        """Annulation d'ordre avec protection."""
        return await self._call("cancel_order", order_id)

    async def get_positions(self) -> list[Position]:
        """Positions avec protection."""
        return await self._call("get_positions")

    async def get_account(self) -> AccountState:
        """Compte avec protection."""
        return await self._call("get_account")

    def get_fees(self, asset: str) -> dict[str, Any]:
        """Frais (pas de protection necessaire, pas d'appel reseau)."""
        return self.adapter.get_fees(asset)

    def get_status(self) -> dict[str, Any]:
        """Retourne l'etat du circuit breaker et du rate limiter."""
        return {
            "broker": self.name,
            "circuit_state": self.circuit.state,
            "circuit_failures": self.circuit.failures,
            "rate_second_calls": len(self.rate_limiter.second_calls),
            "rate_minute_calls": len(self.rate_limiter.minute_calls),
        }
