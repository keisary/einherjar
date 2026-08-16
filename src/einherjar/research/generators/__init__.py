"""generators — Seul TypedGP est actif (Phase 1) + archive MAP-Elites (Phase 2).

Expose le moteur de recherche et l'archive qualité-diversité qui l'augmente.
"""

from einherjar.research.generators.archive import MAPElitesArchive, Behavior
from einherjar.research.generators.typedgp import (
    GeneratorResult,
    TypedGPGenerator,
)

__all__ = [
    "TypedGPGenerator",
    "GeneratorResult",
    "MAPElitesArchive",
    "Behavior",
]