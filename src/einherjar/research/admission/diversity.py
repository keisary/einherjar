"""admission/diversity.py — Descripteurs comportementaux + quotas structurels.

L'invariant I-8 (diversité du corpus) impose des quotas structurels
(famille, type, direction), mais deux Einhers structurellement différents
peuvent produire les mêmes dates de signal. La diversité de portefeuille
n'est garantie que par des descripteurs comportementaux.

Ce module :
  1. Calcule les descripteurs comportementaux d'un Einher (S-3.7).
  2. Calcule le fingerprint comportemental (à partir des descripteurs).
  3. Évalue la diversité d'un Einher par rapport au corpus courant.
  4. Applique les quotas structurels (famille, type, direction).

Conforme à ONTOLOGY.md S-3.7, S-3.4 et ALGORITHME_RESEARCH.md § 10.2 étape 5.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

from einherjar.research.config.loader import EinherjarConfig
from einherjar.research.utils.fingerprint import fingerprint_comportemental
from einherjar.research.utils.types import (
    Condition,
    ConditionNode,
    EconomicFamily,
    FeatureType,
    MesuresBrutes,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Descripteurs comportementaux (S-3.7)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BehavioralDescriptors:
    """Descripteurs comportementaux exportés par un Einher (S-3.7).

    Ils caractérisent le COMPORTEMENT économique d'une règle, pas sa
    structure syntaxique. Deux Einhers structurellement différents
    mais avec des descripteurs proches sont économiquement similaires.
    """

    signal_dates: tuple[int, ...] = ()
    signal_overlap_vs_corpus: float = 0.0
    ret_corr_vs_corpus: float = 0.0
    distribution_by_regime: dict[str, dict[str, float]] = field(default_factory=dict)
    distribution_by_horizon: dict[str, dict[str, float]] = field(default_factory=dict)
    holding_period_hist: tuple[int, ...] = ()
    exit_reason_breakdown: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Sérialisation pour logs / persistance."""
        return {
            "n_signals": len(self.signal_dates),
            "signal_overlap_vs_corpus": self.signal_overlap_vs_corpus,
            "ret_corr_vs_corpus": self.ret_corr_vs_corpus,
            "distribution_by_regime": dict(self.distribution_by_regime),
            "distribution_by_horizon": dict(self.distribution_by_horizon),
            "holding_period_hist": list(self.holding_period_hist),
            "exit_reason_breakdown": dict(self.exit_reason_breakdown),
        }

    def fingerprint(self, rounding_decimals: int = 3) -> str:
        """Calcule le fingerprint comportemental (stable, arrondi à une grille)."""
        return fingerprint_comportemental(self.to_dict(), rounding_decimals=rounding_decimals)


def _build_descriptors(
    mesures: MesuresBrutes,
    signal_indices: tuple[int, ...],
    corpus_signal_dates: list[tuple[int, ...]] | None = None,
    corpus_ret_series: list[tuple[float, ...]] | None = None,
    this_ret_series: tuple[float, ...] = (),
) -> BehavioralDescriptors:
    """Construit les descripteurs comportementaux depuis MesuresBrutes + signaux.

    Args:
        mesures: MesuresBrutes (sortie du moteur).
        signal_indices: Indices des bougies qui ont déclenché un signal.
        corpus_signal_dates: Liste des signal_dates de chaque Einher du corpus.
        corpus_ret_series: Liste des ret_series (rendements par trade) de chaque Einher du corpus.
        this_ret_series: ret_series de l'Einher candidat.

    Returns:
        BehavioralDescriptors.
    """
    # Exit breakdown.
    n = max(mesures.n_signals, 1)
    exit_breakdown = {
        "tp": mesures.n_tp_hit / n,
        "sl": mesures.n_sl_hit / n,
        "timeout": mesures.n_timeout / n,
    }
    # Distribution par "régime" : V1 simplifié, on note juste la tendance globale.
    distribution_by_regime = {
        "global": {
            "winrate": exit_breakdown["tp"],
            "avg_ret": mesures.ret_mean_pct_net,
            "sharpe": mesures.sharpe_net if not math.isnan(mesures.sharpe_net) else 0.0,
        },
    }
    distribution_by_horizon = {
        "tp": {"n": float(mesures.n_tp_hit), "avg_ret": 0.0},
        "sl": {"n": float(mesures.n_sl_hit), "avg_ret": 0.0},
        "timeout": {"n": float(mesures.n_timeout), "avg_ret": 0.0},
    }
    # Overlap signaux vs corpus (Jaccard max).
    overlap_max = _max_jaccard(set(signal_indices), corpus_signal_dates or [])
    # Corrélation des ret_series vs corpus.
    corr_max = _max_pearson(
        this_ret_series, signal_indices, corpus_ret_series or [], corpus_signal_dates or [],
    )
    # Holding period histogram (binned).
    hist = _holding_period_hist(mesures, n_bins=20)
    return BehavioralDescriptors(
        signal_dates=signal_indices,
        signal_overlap_vs_corpus=overlap_max,
        ret_corr_vs_corpus=corr_max,
        distribution_by_regime=distribution_by_regime,
        distribution_by_horizon=distribution_by_horizon,
        holding_period_hist=hist,
        exit_reason_breakdown=exit_breakdown,
    )


def _max_jaccard(
    this_set: set[int],
    corpus_sets: list[tuple[int, ...]],
) -> float:
    """Jaccard max entre un set de dates et une liste de sets."""
    if not corpus_sets or not this_set:
        return 0.0
    jmax = 0.0
    for s in corpus_sets:
        other = set(s)
        if not other:
            continue
        inter = len(this_set & other)
        union = len(this_set | other)
        if union == 0:
            continue
        jmax = max(jmax, inter / union)
    return jmax


def _max_pearson(
    this_series: tuple[float, ...],
    this_dates: tuple[int, ...] | list[tuple[float, ...]],
    corpus_series: list[tuple[float, ...]] | None = None,
    corpus_dates: list[tuple[int, ...]] | None = None,
) -> float:
    """Pearson max (en valeur absolue) entre une série et une liste (P1 #5).

    Les retours sont alignés par date de signal. Comparer leur rang dans
    deux listes de trades crée une corrélation fictive quand les entrées
    ne sont pas synchrones.
    """
    # Compatibilité de l'ancien helper de tests : _max_pearson(series, corpus).
    # Le chemin applicatif fournit toujours dates + corpus_series + corpus_dates.
    if corpus_series is None:
        corpus_series = this_dates  # type: ignore[assignment]
        this_dates = tuple(range(len(this_series)))
        corpus_dates = [tuple(range(len(series))) for series in corpus_series]
    if not corpus_series or not this_series or len(this_series) < 2:
        return 0.0
    this_by_date = dict(zip(this_dates, this_series))
    cmax = 0.0
    for dates, series in zip(corpus_dates or [], corpus_series):
        other_by_date = dict(zip(dates, series))
        common_dates = sorted(set(this_by_date) & set(other_by_date))
        if len(common_dates) < 2:
            continue
        a = [this_by_date[date] for date in common_dates]
        b = [other_by_date[date] for date in common_dates]
        n = len(common_dates)
        try:
            mean_a = sum(a) / n
            mean_b = sum(b) / n
            num = sum((a[i] - mean_a) * (b[i] - mean_b) for i in range(n))
            den_a = math.sqrt(sum((x - mean_a) ** 2 for x in a))
            den_b = math.sqrt(sum((x - mean_b) ** 2 for x in b))
            if den_a == 0 or den_b == 0:
                continue
            cmax = max(cmax, abs(num / (den_a * den_b)))
        except (ZeroDivisionError, ValueError):
            continue
    return cmax


def _holding_period_hist(mesures: MesuresBrutes, n_bins: int = 20) -> tuple[int, ...]:
    """Histogramme grossier de la durée des trades (binned).

    V1 : on retourne un tuple de zéros de longueur n_bins, le calcul exact
    se fera quand on aura accès à la liste de trades dans MesuresBrutes.
    """
    return (0,) * n_bins


# --------------------------------------------------------------------------- #
# Quotas structurels (I-8)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class QuotaReport:
    """Résultat de l'application des quotas structurels."""

    family_ok: bool
    type_ok: bool
    direction_ok: bool
    current_family_fracs: dict[str, float]
    current_type_fracs: dict[str, float]
    current_direction_fracs: dict[str, float]
    new_family: str
    new_type: str
    new_direction: str

    @property
    def passed(self) -> bool:
        """True si tous les quotas (family + type + direction) passent."""
        return self.family_ok and self.type_ok and self.direction_ok

    def to_dict(self) -> dict[str, Any]:
        """Sérialisation pour logs / persistance."""
        return {
            "passed": self.passed,
            "family_ok": self.family_ok,
            "type_ok": self.type_ok,
            "direction_ok": self.direction_ok,
            "current_family_fracs": self.current_family_fracs,
            "current_type_fracs": self.current_type_fracs,
            "current_direction_fracs": self.current_direction_fracs,
            "new_family": self.new_family,
            "new_type": self.new_type,
            "new_direction": self.new_direction,
        }


def evaluate_quotas(
    new_family: str,
    new_type: str,
    new_direction: str,
    current_family_fracs: dict[str, float],
    current_type_fracs: dict[str, float],
    current_direction_fracs: dict[str, float],
    config: EinherjarConfig,
) -> QuotaReport:
    """Vérifie que l'ajout d'un nouvel Einher respecte les quotas.

    Args:
        new_family: Famille économique de l'Einher candidat.
        new_type: Type de feature dominant (atomic/quantitative/...).
        new_direction: 'long' ou 'short'.
        current_family_fracs: Fractions actuelles par famille dans le corpus.
        current_type_fracs: Fractions actuelles par type.
        current_direction_fracs: Fractions actuelles par direction.
        config: Config (pour les seuils de quotas).

    Returns:
        QuotaReport.
    """
    q = config.thresholds["diversity"]["quotas"]
    family_max = float(q["family_max_frac"])
    type_max = float(q["type_max_frac"])
    direction_min = float(q["direction_min_frac"])

    # Simule l'ajout (incrémente la fraction de la nouvelle famille/type/direction).
    future_family_fracs = _increment(current_family_fracs, new_family)
    future_type_fracs = _increment(current_type_fracs, new_type)
    future_dir_fracs = _increment(current_direction_fracs, new_direction)

    family_ok = all(frac <= family_max for frac in future_family_fracs.values())
    type_ok = all(frac <= type_max for frac in future_type_fracs.values())
    direction_ok = all(frac >= direction_min for frac in future_dir_fracs.values())

    return QuotaReport(
        family_ok=family_ok,
        type_ok=type_ok,
        direction_ok=direction_ok,
        current_family_fracs=current_family_fracs,
        current_type_fracs=current_type_fracs,
        current_direction_fracs=current_direction_fracs,
        new_family=new_family,
        new_type=new_type,
        new_direction=new_direction,
    )


def _increment(fracs: dict[str, float], key: str) -> dict[str, float]:
    """Incrémente la fraction d'une clé en renormalisant (V1 simplifié)."""
    new_fracs = dict(fracs)
    new_fracs[key] = new_fracs.get(key, 0.0) + 1.0
    total = sum(new_fracs.values())
    if total > 0:
        new_fracs = {k: v / total for k, v in new_fracs.items()}
    return new_fracs


# --------------------------------------------------------------------------- #
# Helpers pour extraire la famille/type dominant d'un Einher
# --------------------------------------------------------------------------- #


def extract_dominant_family(
    condition_tree: Condition | ConditionNode,
    config: EinherjarConfig,
) -> str:
    """Extrait la famille économique dominante d'un arbre de conditions.

    V1 : on prend la famille de la première feature rencontrée (DFS).
    """
    feat = _first_feature(condition_tree)
    if feat is None:
        return EconomicFamily.OTHER.value
    info = config.features_taxonomy.get("features", {}).get(feat, {})
    return info.get("economic_family", EconomicFamily.OTHER.value)


def extract_dominant_type(
    condition_tree: Condition | ConditionNode,
    config: EinherjarConfig,
) -> str:
    """Extrait le type de feature dominant d'un arbre de conditions."""
    feat = _first_feature(condition_tree)
    if feat is None:
        return FeatureType.ATOMIC.value
    info = config.features_taxonomy.get("features", {}).get(feat, {})
    return info.get("feature_type", FeatureType.ATOMIC.value)


def _first_feature(node: Condition | ConditionNode) -> str | None:
    """Retourne la première feature rencontrée (DFS)."""
    if isinstance(node, Condition):
        return node.feature_ref
    left = _first_feature(node.left)
    if left is not None:
        return left
    if node.right is not None:
        return _first_feature(node.right)
    return None
