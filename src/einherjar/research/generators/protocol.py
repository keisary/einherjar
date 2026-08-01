"""generators/protocol.py — Protocole reproductible de comparaison des générateurs.

Tout générateur qui veut être comparé à un autre doit être évalué avec :
  - MÊME seed maître
  - MÊMES splits (train/val/holdout)
  - MÊME budget d'évaluations
  - MÊMES métriques
  - MÊMES coûts

Ce fichier définit le dataclass qui encapsule ces contraintes, et la
fonction qui l'instancie depuis la config.

Conforme à ALGORITHME_RESEARCH.md § 10.2 étape 2.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from einherjar.research.config.loader import EinherjarConfig

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Protocole
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GenerationProtocol:
    """Protocole reproductible de comparaison des générateurs (Step 2).

    Tout générateur qui tourne sous ce protocole utilise exactement les
    mêmes contraintes, ce qui rend les comparaisons valides.

    Attributes:
        seed: Graine RNG maître (propagation déterministe).
        data_version: Identifiant de version de données.
        splits: Bornes explicites (ou ratios si mode='ratio').
        n_eval_budget: Nombre max d'évaluations autorisées (mur d'arrêt).
        max_conditions: Profondeur max des conditions générées.
        p_compound: Probabilité de générer une condition composée.
        assets: Assets du universe (tuple).
        timeframes: Timeframes du universe (tuple).
        amplitude_value: Valeur d'amplitude (prix absolu par défaut).
        cooldown_k: Cooldown d'observation (K bougies).
    """

    seed: int
    data_version: str
    splits: dict[str, Any] = field(default_factory=dict)
    n_eval_budget: int = 10_000
    max_conditions: int = 3
    p_compound: float = 0.3
    assets: tuple[str, ...] = ("BTCUSD",)
    timeframes: tuple[str, ...] = ("1h",)
    amplitude_value: float = 50.0
    cooldown_k: int = 5

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "data_version": self.data_version,
            "splits": self.splits,
            "n_eval_budget": self.n_eval_budget,
            "max_conditions": self.max_conditions,
            "p_compound": self.p_compound,
            "assets": list(self.assets),
            "timeframes": list(self.timeframes),
            "amplitude_value": self.amplitude_value,
            "cooldown_k": self.cooldown_k,
        }


def make_protocol(
    config: EinherjarConfig,
    data_version: str,
    seed: int = 42,
    n_eval_budget: int = 10_000,
    assets: tuple[str, ...] = ("BTCUSD",),
    timeframes: tuple[str, ...] = ("1h",),
) -> GenerationProtocol:
    """Construit un GenerationProtocol depuis la config + data_version + seed.

    Args:
        config: Configuration chargée.
        data_version: Identifiant de version de données.
        seed: Graine RNG maître.
        n_eval_budget: Budget max d'évaluations (mur d'arrêt commun à tous).
        assets: Assets cibles.
        timeframes: Timeframes cibles.

    Returns:
        GenerationProtocol figé.
    """
    splits_cfg = config.splits
    return GenerationProtocol(
        seed=seed,
        data_version=data_version,
        splits={
            "mode": splits_cfg.get("mode", "ratio"),
            "train_ratio": splits_cfg["ratios"]["train"],
            "val_ratio": splits_cfg["ratios"]["val"],
            "holdout_ratio": splits_cfg["ratios"]["holdout"],
            "purge_window": splits_cfg.get("purging", {}).get("enabled", True),
            "embargo_bougies": splits_cfg.get("embargo", {}).get("bougies", 1),
        },
        n_eval_budget=n_eval_budget,
        assets=assets,
        timeframes=timeframes,
    )
