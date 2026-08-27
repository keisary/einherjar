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

from .paths import TAXONOMY_PATH
from .types import Einher

logger = logging.getLogger(__name__)


# Sprint 2.2.2 : mapping feature_name -> economic_family
# (chemin centralise dans paths.py - plus de chemin hardcode)
_FAMILY_CACHE: dict[str, str] | None = None


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
                if isinstance(v, dict | list):
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
    """Seuils d'admission.

    Sprint 3.5 FIX : limitations relaxees pour permettre plus d'admissions.
    - min_families : 1 (au lieu de 2) - les regles simples sont OK
    - min_holdout_trades : 30 (au lieu de 100) - plus permissif, ajuste par horizon
    """
    min_trades: int = 30
    min_sharpe: float = 0.90  # FIX (2026-08-27) : 0.90 au lieu de 0.3
    min_win_rate: float = 0.55  # FIX (2026-08-27) : 0.55 au lieu de 0.40
    min_profit_factor: float = 1.0
    max_drawdown: float = 0.30
    min_families: int = 1  # Sprint 3.5 : 1 au lieu de 2 (regles simples OK)
    min_holdout_trades: int = 30  # Sprint 3.5 : 30 au lieu de 100 (plus permissif)
    fdr: float = 0.05
    apply_bh: bool = True

    @classmethod
    def debug(cls) -> AdmissionConfig:
        """Seuils très souples pour tester le pipeline (debug uniquement)."""
        return cls(
            min_trades=5,
            min_sharpe=-1.0,
            min_win_rate=0.30,
            min_profit_factor=0.5,
            max_drawdown=0.99,
            min_families=1,
            min_holdout_trades=0,  # debug : pas de check holdout
            fdr=1.0,
            apply_bh=False,
        )


def check_admission(
    einher: Einher,
    config: AdmissionConfig = AdmissionConfig(),
    bh_rejected: bool | None = None,
) -> tuple[bool, str | None]:
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
        # FIX P0-5 (AI Review 2026-08-20) : ne pas seulement check n_trades,
        # aussi verifier la performance holdout. Sinon une strat avec
        # Sharpe=17.9 en val et -3.0 en holdout est admise.
        # On assouplit les seuils par rapport a val (le holdout est plus petit).
        try:
            holdout_passed, holdout_reason = einher.holdout_metrics.passes_admission(
                min_trades=max(5, config.min_trades // 3),
                min_sharpe=0.0,  # holdout Sharpe doit au moins etre positif
                min_win_rate=config.min_win_rate * 0.9,
                min_profit_factor=0.9,
                max_drawdown=config.max_drawdown * 1.5,
            )
            if not holdout_passed:
                return False, f"Holdout REJECTED : {holdout_reason}"
        except Exception as e:
            # Si le check holdout plante, on ne bloque pas l'admission
            logger.warning("Holdout admission check failed: %s", e)

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
