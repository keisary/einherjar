"""admission.py - Critères d'admission pour les Einhers.

Réponse Q15 : tous ceux qui passent l'admission.

Critères minimaux (V1) :
- n_trades >= 30
- sharpe_ratio >= 0.3
- win_rate >= 0.40
- profit_factor >= 1.0
- max_drawdown < 0.30 (DD < 30%)
- total_return > 0

Critères ajoutés Sprint 2.2 :
- >= 2 familles économiques différentes dans les conditions (diversité)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from einherjar.research.xgb_einhers.types import Einher, EinherMetrics

logger = logging.getLogger(__name__)


# Sprint 2.2.2 : mapping feature_name -> economic_family
TAXONOMY_PATH = Path(
    "D:/midas_v2/Einherjar/src/einherjar/research/config/features_taxonomy.json"
)
_FAMILY_CACHE: Optional[dict[str, str]] = None


def load_feature_family_map() -> dict[str, str]:
    """Charge le mapping feature_name -> economic_family depuis la taxonomie.

    Returns:
        dict {feature_name: family_name}
    """
    global _FAMILY_CACHE
    if _FAMILY_CACHE is None:
        with open(TAXONOMY_PATH) as f:
            tax = json.load(f)
        _FAMILY_CACHE = {
            name: meta.get("economic_family", "unknown")
            for name, meta in tax["features"].items()
        }
    return _FAMILY_CACHE


def get_einher_families(einher: Einher) -> set[str]:
    """Retourne l'ensemble des familles économiques utilisées par l'Einher.

    Parcourt récursivement l'AST pour extraire les feature_refs uniques.
    """
    family_map = load_feature_family_map()
    features = set()

    def _walk(node):
        if isinstance(node, dict):
            if "feature_ref" in node:
                features.add(node["feature_ref"])
            for v in node.values():
                if isinstance(v, (dict, list)):
                    _walk(v)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    # condition_tree est un ConditionNode (objet) ou un dict selon l'origine
    ct = einher.condition_tree
    if hasattr(ct, "to_dict"):
        _walk(ct.to_dict())
    elif isinstance(ct, dict):
        _walk(ct)
    return {family_map.get(f, "unknown") for f in features}


@dataclass(frozen=True)
class AdmissionConfig:
    """Seuils d'admission."""
    min_trades: int = 30
    min_sharpe: float = 0.3
    min_win_rate: float = 0.40
    min_profit_factor: float = 1.0
    max_drawdown: float = 0.30  # On rejette si max_dd > 0.30 (donc DD < 30%)
    min_families: int = 2  # Sprint 2.2.2 : >= 2 familles différentes
    min_holdout_trades: int = 100  # Sprint 3.1 P1 : Gemini recommande 100+ pour significativite
    fdr: float = 0.05  # Sprint 3.1 P1 : False Discovery Rate pour Benjamini-Hochberg
    apply_bh: bool = True  # Sprint 3.1 P1 : activer la correction multi-tests

    @classmethod
    def debug(cls) -> "AdmissionConfig":
        """Seuils très souples pour tester le pipeline (debug uniquement)."""
        return cls(
            min_trades=5,
            min_sharpe=-1.0,
            min_win_rate=0.30,
            min_profit_factor=0.5,
            max_drawdown=0.99,
            min_families=1,  # debug : pas de quota famille
            min_holdout_trades=0,  # debug : pas de check holdout
            fdr=1.0,  # debug : pas de correction BH
            apply_bh=False,  # debug : pas de BH
        )


def check_admission(
    einher: Einher,
    config: AdmissionConfig = AdmissionConfig(),
    bh_rejected: Optional[bool] = None,
) -> tuple[bool, Optional[str]]:
    """Vérifie si un Einher passe les critères d'admission.

    Args:
        einher : l'Einher à tester
        config : configuration d'admission
        bh_rejected : Sprint 3.1 P1. Si fourni et False, l'Einher est rejeté
                      (résultat de Benjamini-Hochberg depuis le caller).

    Returns:
        (passed, reason) : passed=True si OK, reason = raison du rejet sinon.
    """
    # Sprint 3.1 P1 : check BH (Benjamini-Hochberg)
    if bh_rejected is False:
        return False, "BH REJECTED : non significatif apres correction multi-tests"

    # Sprint 2.2.2 : check diversité inter-familles
    if config.min_families >= 2:
        families = get_einher_families(einher)
        if len(families) < config.min_families:
            return False, (
                f"Diversity REJECTED : {len(families)} families ({families}) "
                f"< min_families={config.min_families}"
            )

    # Sprint 2.4.1 : check holdout (filtre les Einhers non significatifs)
    if config.min_holdout_trades > 0 and einher.holdout_metrics is not None:
        if einher.holdout_metrics.n_trades < config.min_holdout_trades:
            return False, (
                f"Holdout REJECTED : {einher.holdout_metrics.n_trades} trades "
                f"< min_holdout_trades={config.min_holdout_trades}"
            )

    passed, reason = einher.metrics.passes_admission(
        min_trades=config.min_trades,
        min_sharpe=config.min_sharpe,
        min_win_rate=config.min_win_rate,
        min_profit_factor=config.min_profit_factor,
        max_drawdown=config.max_drawdown,
    )
    if not passed:
        logger.info("Admission REJECTED : %s (%s)", einher.id, reason)
    else:
        logger.info("Admission ACCEPTED : %s (sharpe=%.3f, wr=%.2f, pf=%.2f, dd=%.3f, fam=%d)",
                    einher.id, einher.metrics.sharpe_ratio,
                    einher.metrics.win_rate, einher.metrics.profit_factor,
                    einher.metrics.max_drawdown, len(get_einher_families(einher)))
    return passed, reason
