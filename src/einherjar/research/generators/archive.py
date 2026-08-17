"""generators/archive.py — Archive MAP-Elites (Quality-Diversity) pour le STGP.

Phase 2, Étape 3 : archive à niches de qualité-diversité (Mouret & Clune 2015).

Idée :
  - L'espace comportemental est découpé en NICHES le long d'axes descripteurs.
  - Chaque niche héberge le MEILLEUR individu (standpoint fitness) qui tombe
    dedans.
  - À la fin de l'évolution, l'archive contient une population DIVERSIFIÉE :
    un bon représentant par style de stratégie, au lieu de N jumeaux.

Axes (default) :
  - direction            : Long / Short            → 2
  - fréquence de trades  : rare / moyen / fréquent → 3 (log buckets)
  - win rate             : faible / moyen / élevé  → 3
  Total : 2 × 3 × 3 = 18 niches.

Chaque occupant est (hypothesis, fitness).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

from einherjar.research.utils.types import Direction, Hypothesis

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Descripteurs comportementaux (extraits des MesuresBrutes)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Behavior:
    """Descripteur comportemental d'un individu, utilisé pour le classement en niche."""

    direction: str
    trades_per_year_bucket: int   # 0=rare, 1=moyen, 2=fréquent
    win_rate_bucket: int          # 0=faible, 1=moyen, 2=élevé

    def key(self) -> tuple[str, int, int]:
        return (self.direction, self.trades_per_year_bucket, self.win_rate_bucket)


def _log_bucket(value: float, edges: list[float]) -> int:
    """Range une valeur positive en bucket logarithmique sur `edges`."""
    if value <= 0:
        return 0
    log_v = math.log10(value)
    for i, e in enumerate(edges):
        if log_v < e:
            return i
    return len(edges)


def behavior_from_measures(measures: Any, direction: Direction) -> Behavior:
    """Calcule le descripteur depuis MesuresBrutes + direction.

    Axes comportementaux adaptés au mode HOLD (pas de SL/TP) :
      - direction           : Long / Short                     -> 2
      - sharpe_bucket       : négatif / faible / bon           -> 3
      - freq_bucket         : peu de trades / moyen / fréquent -> 3
    Total : 2 x 3 x 3 = 18 niches.

    En mode hold, tp_hit_rate est toujours 0 (tous TIMEOUT) donc inutilisable
    comme axe de diversité. On utilise Sharpe et n_signals à la place.

    Args:
        measures: MesuresBrutes (ou None si individu invalide).
        direction: Direction du trade.

    Returns:
        Behavior avec les buckets calculés.
    """
    n = getattr(measures, "n_signals", 0) or 0
    sharpe = getattr(measures, "sharpe_net", float("nan"))

    # Bucket Sharpe : négatif (<0), faible (0-1), bon (>1)
    if not math.isfinite(sharpe):
        sharpe_bucket = 0
    elif sharpe <= 0:
        sharpe_bucket = 0
    elif sharpe <= 1.0:
        sharpe_bucket = 1
    else:
        sharpe_bucket = 2

    # Bucket fréquence : peu (0-50), moyen (50-500), fréquent (>500)
    if n <= 0:
        freq_bucket = 0
    elif n <= 50:
        freq_bucket = 0
    elif n <= 500:
        freq_bucket = 1
    else:
        freq_bucket = 2

    return Behavior(
        direction=direction.value,
        trades_per_year_bucket=freq_bucket,
        win_rate_bucket=sharpe_bucket,
    )


# --------------------------------------------------------------------------- #
# Archive MAP-Elites
# --------------------------------------------------------------------------- #


@dataclass
class MAPElitesArchive:
    """Archive à niches : garde le meilleur individu par case comportementale.

    Attributes:
        _cells: dict[key, Occupant] où Occupant = (hypothesis, fitness).
    """

    _cells: dict[tuple[str, int, int], tuple[Hypothesis, float]] = field(default_factory=dict)

    @property
    def size(self) -> int:
        """Nombre de niches occupées."""
        return len(self._cells)

    @property
    def max_capacity(self) -> int:
        """Capacité théorique = nombre total de combinaisons de niche."""
        # direction (2) × 3 buckets trades × 3 buckets winrate
        return 2 * 3 * 3

    def update(self, hypothesis: Hypothesis, fitness: float, measures: Any) -> bool:
        """Ajoute/remplace l'individu dans sa niche si sa fitness est meilleure.

        Returns:
            True si l'archive a été modifiée (nouveau ou meilleur occupant).
        """
        if not math.isfinite(fitness):
            return False
        key = behavior_from_measures(measures, hypothesis.direction).key()
        current = self._cells.get(key)
        if current is None or fitness > current[1]:
            self._cells[key] = (hypothesis, fitness)
            return True
        return False

    def individuals(self) -> list[Hypothesis]:
        """Retourne les hypothèses des niches occupées (population diversifiée)."""
        # Tri par fitness décroissante pour un ordre stable / reproductible.
        return [h for h, _ in sorted(self._cells.values(), key=lambda x: x[1], reverse=True)]

    def best(self) -> tuple[Hypothesis, float] | None:
        """Retourne le meilleur (hypothesis, fitness) global de l'archive."""
        if not self._cells:
            return None
        best_h, best_f = max(self._cells.values(), key=lambda x: x[1])
        return best_h, best_f

    def stats(self) -> dict[str, Any]:
        """Résumé des niches occupées (pour logs)."""
        long_n = sum(1 for k in self._cells if k[0] == Direction.LONG.value)
        short_n = sum(1 for k in self._cells if k[0] == Direction.SHORT.value)
        return {
            "n_occupied": self.size,
            "n_long": long_n,
            "n_short": short_n,
            "occupancy_rate": (self.size / self.max_capacity) if self.max_capacity else 0.0,
        }