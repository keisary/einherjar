"""generators/config.py — Config de recherche TypedGP (Phase 1).

Remplace l'ancien GenerationProtocol (multi-générateurs, comparateur) par une
config simple, mono-générateur, qui porte explicitement le mapping
timeframe → horizon.

Contenu :
  - TypedGPConfig : dataclass figée avec seed, max_depth, horizon_index,
    taillea (population / générations / GA params), et délégation à la config
    Einherjar pour la taxonomie de features et les seuils.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TypedGPConfig:
    """Configuration de recherche pour le STGP (Phase 1).

    Porte explicitement :
      - seed  : graine RNG maître
      - max_depth : profondeur max des conditions (arbres)
      - horizon_index : indice d'horizon (0-3) explicite pour ce run —
        cf. § Horizon (point central Phase 1). Non connecté aux Y_* pour
        l'instant, mais tracé dans tous les logs et transmis à l'évaluation.
      - timeframe / asset : univers ciblé
      - taste_samples : sous-échantillonnage bougies pour l'évolution (0 = complet)
      - paramètres génétiques (population, générations, probs, élitisme)

    Note : les attributs de taxonomie (usable_feature_names, thresholds,
    features_taxonomy) sont délégués à la config Einherjar sous-jacente via
    `einherjar_config`.
    """

    seed: int
    einherjar_config: Any
    data_version: str = "v1"
    max_depth: int = 4
    horizon_index: int = 1
    timeframe: str = "15m"
    asset: str = "BTCUSD"
    amplitude_value: float = 5.0
    cooldown_k: int = 5
    taste_samples: int = 0
    population_size: int = 50
    n_generations: int = 10
    crossover_prob: float = 0.8
    mutation_prob: float = 0.2
    tournament_size: int = 3
    elitism: int = 2
    n_eval_budget: int = 200

    # Délégation à la config Einherjar (taxonomie / seuils)
    @property
    def features_taxonomy(self):
        return self.einherjar_config.features_taxonomy

    @property
    def usable_feature_names(self):
        return self.einherjar_config.usable_feature_names

    @property
    def thresholds(self):
        return self.einherjar_config.thresholds

    @property
    def splits(self):
        return self.einherjar_config.splits

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "data_version": self.data_version,
            "max_depth": self.max_depth,
            "horizon_index": self.horizon_index,
            "timeframe": self.timeframe,
            "asset": self.asset,
            "amplitude_value": self.amplitude_value,
            "cooldown_k": self.cooldown_k,
            "taste_samples": self.taste_samples,
            "population_size": self.population_size,
            "n_generations": self.n_generations,
            "crossover_prob": self.crossover_prob,
            "mutation_prob": self.mutation_prob,
            "tournament_size": self.tournament_size,
            "elitism": self.elitism,
            "n_eval_budget": self.n_eval_budget,
        }