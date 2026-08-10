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

    NOTE (refactor budget/candidats, décision utilisateur) :
      - `n_candidates` = nombre d'HYPOTHÈSES GÉNÉRÉES par un générateur.
        Les moteurs produisent un MAXIMUM de candidats « au minimum pas
        absurdes » (critères larges) ; c'est l'évaluation/admission qui
        resserrent l'étau ensuite. Par défaut 100k.
      - `n_eval_budget` = le mur de COÛT : nombre maximal d'évaluations
        moteur (train_calibrate + test_on) que le comparator peut
        consommer au total pour comparer équitablement les générateurs.
        Les générateurs NON évolutionnaires (random, typedGP pur, BNF)
        génèrent sans appeler le moteur ; les évolutionnaires (beam,
        NSGA-II, memetic) consomment des évaluations pendant leur
        recherche. Les deux notions sont INDÉPENDANTES.

    Attributes:
        seed: Graine RNG maître (propagation déterministe).
        data_version: Identifiant de version de données.
        splits: Bornes explicites (ou ratios si mode='ratio').
        n_candidates: Nombre d'hypothèses à générer par moteur
            (volume de génération, défaut 100k).
        n_eval_budget: Nombre MAX d'appels moteur autorisés (mur d'arrêt de
            COÛT, PAS un plafond de génération).
        max_conditions: Profondeur max des conditions générées.
        p_compound: Probabilité de générer une condition composée.
        assets: Assets du universe (tuple).
        timeframes: Timeframes du universe (tuple).
        amplitude_value: Valeur d'amplitude. L'unite est fixee a
            AmplitudeUnit.MULTIPLE_ATR dans BaseGenerator._make_amplitude
            (cf. algorithms.py). Avec la valeur par defaut 5.0, cela
            represente 5x ATR (multiple d'ATR), PAS 5 unites de prix absolu.
            Si tu veux exprimer un mouvement en prix absolu, passe par
            AmplitudeUnit.PRICE_ABSOLU directement (non utilise par les
            generateurs actuels).
        cooldown_k: Cooldown d'observation (K bougies).
    """

    seed: int
    data_version: str
    splits: dict[str, Any] = field(default_factory=dict)
    n_candidates: int = 100_000
    n_eval_budget: int = 2_000
    max_conditions: int = 3
    p_compound: float = 0.3
    assets: tuple[str, ...] = ("BTCUSD",)
    timeframes: tuple[str, ...] = ("1h",)
    amplitude_value: float = 5.0
    cooldown_k: int = 5
    # Tasting : nb de bougies échantillonnées par test_on pendant l'ÉVOLUTION
    # (0 = fenêtre complète). Décision 2026-08-10 : l'échantillon seedé de
    # blocs contigus est identique pour toute la population → évolution
    # ~10-50× plus rapide ; l'admission reste sur le val complet.
    n_samples: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "data_version": self.data_version,
            "splits": self.splits,
            "n_candidates": self.n_candidates,
            "n_eval_budget": self.n_eval_budget,
            "max_conditions": self.max_conditions,
            "p_compound": self.p_compound,
            "assets": list(self.assets),
            "timeframes": list(self.timeframes),
            "amplitude_value": self.amplitude_value,
            "cooldown_k": self.cooldown_k,
            "n_samples": self.n_samples,
        }


def make_protocol(
    config: EinherjarConfig,
    data_version: str,
    seed: int = 42,
    n_eval_budget: int = 2_000,
    n_candidates: int = 100_000,
    assets: tuple[str, ...] = ("BTCUSD",),
    timeframes: tuple[str, ...] = ("1h",),
    max_conditions: int = 4,
    n_samples: int = 0,
) -> GenerationProtocol:
    """Construit un GenerationProtocol depuis la config + data_version + seed.

    Args:
        config: Configuration chargée.
        data_version: Identifiant de version de données.
        seed: Graine RNG maître.
        n_eval_budget: Mur de COÛT — nb max d'évaluations moteur (commun).
        n_candidates: Volume de GÉNÉRATION — nb d'hypothèses par moteur.
        assets: Assets cibles.
        timeframes: Timeframes cibles.
        max_conditions: Profondeur max des conditions générées (arbres).

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
            # The exact per-hypothesis horizon is applied by the data splitter;
            # record the conservative maximum here, never a boolean flag.
            "purge_window": (
                int(config.evaluation["n_window"]["max_n"])
                if splits_cfg.get("purging", {}).get("enabled", True)
                else 0
            ),
            "embargo_bougies": splits_cfg.get("embargo", {}).get("bougies", 1),
        },
        n_eval_budget=n_eval_budget,
        n_candidates=n_candidates,
        assets=assets,
        timeframes=timeframes,
        max_conditions=max_conditions,
        n_samples=n_samples,
    )
