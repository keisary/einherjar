"""Aggregation deterministe des signaux compatibles.

Le module ne choisit jamais entre un long et un short. Les deux
intentions restent distinctes afin que le RiskManager applique ses limites.
"""

from __future__ import annotations

from collections import defaultdict
from statistics import fmean

from einherjar.core.enums import Direction, TimeFrame
from einherjar.core.models import ConfluenceCluster, Signal


class ConfluenceEngine:
    """Transforme des signaux bruts en intentions par actif et direction."""

    def __init__(self, minimum_confidence: float = 0.0) -> None:
        self.minimum_confidence = minimum_confidence

    def aggregate(self, signals: list[Signal]) -> list[ConfluenceCluster]:
        """Agrege les signaux du cycle courant.

        Les prix sont ponderes par la confiance. Le score privilegie
        l'accord de plusieurs domaines, jamais les doublons identiques.
        """
        grouped: dict[tuple[str, Direction], list[Signal]] = defaultdict(list)
        for signal in signals:
            if signal.direction not in (Direction.LONG, Direction.SHORT):
                continue
            if signal.confidence >= self.minimum_confidence:
                grouped[(signal.asset, signal.direction)].append(signal)

        clusters: list[ConfluenceCluster] = []
        for (asset, direction), members in grouped.items():
            weights = [max(signal.confidence, 0.01) for signal in members]
            weight_sum = sum(weights)
            weighted = lambda values: sum(value * weight for value, weight in zip(values, weights)) / weight_sum
            domains = {self._domain(signal) for signal in members}
            raw_score = weighted([signal.confidence for signal in members])
            diversity = min(len(domains) / 3.0, 1.0)
            agreement = min(len(members) / 3.0, 1.0)
            score = min(1.0, raw_score * (0.70 + 0.15 * diversity + 0.15 * agreement))
            timeframe = self._dominant_timeframe(members)
            clusters.append(
                ConfluenceCluster(
                    asset=asset,
                    direction=direction,
                    timeframe=timeframe,
                    entry_price=weighted([signal.entry_price for signal in members]),
                    tp_price=weighted([signal.tp_price for signal in members]),
                    sl_price=weighted([signal.sl_price for signal in members]),
                    confidence=score,
                    contributing_einhers=[signal.einher_name for signal in members],
                    score=score,
                    context={
                        "signal_count": len(members),
                        "domains": sorted(domains),
                        "raw_confidence": fmean(signal.confidence for signal in members),
                    },
                )
            )
        return sorted(clusters, key=lambda cluster: cluster.score, reverse=True)

    @staticmethod
    def _domain(signal: Signal) -> str:
        """Extrait un domaine sans imposer une convention de nommage."""
        explicit = signal.context.get("domain")
        if explicit:
            return str(explicit)
        parts = signal.einher_name.split("_")
        return parts[1] if len(parts) > 1 else "unknown"

    @staticmethod
    def _dominant_timeframe(signals: list[Signal]) -> TimeFrame:
        """Choisit le timeframe le plus represente, puis le plus court."""
        counts: dict[TimeFrame, int] = defaultdict(int)
        for signal in signals:
            counts[signal.timeframe] += 1
        return max(counts, key=lambda timeframe: (counts[timeframe], -len(timeframe.value)))
