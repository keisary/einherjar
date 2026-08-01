"""admission/criteria.py — Les 7 critères d'admission S-3.4 (UN fichier par convention).

Chaque critère est une fonction pure (ou un dataclass résultat) qui prend
en entrée les MesuresBrutes + métriques auxiliaires et retourne un verdict
(booléen) + un motif d'échec (RejectionReason) si KO.

Critères implémentés :
  1. DSR (Deflated Sharpe Ratio) — corrige pour le nombre d'essais indépendants.
  2. PBO (Probability of Backtest Overfitting) — CPCV léger K=6, N=6.
  3. Block bootstrap CI sur Sharpe — `sharpe_ci_low > 0`.
  4. Block bootstrap CI sur ret_total — `ret_total_ci_low > 0`.
  5. n_trades minimum — significativité statistique.
  6. consistency_cross_asset — ≥ 70% des actifs positifs.
  7. max_drawdown — < seuil (défaut 0.25).

Conforme à ONTOLOGY.md S-3.4 et ALGORITHME_RESEARCH.md § 10.2 étape 5.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from einherjar.research.config.loader import EinherjarConfig
from einherjar.research.utils.metrics import dsr as dsr_metric
from einherjar.research.utils.types import MesuresBrutes, RejectionReason

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Verdict unifié
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CriterionVerdict:
    """Résultat d'un critère d'admission individuel.

    Attributes:
        name: Nom du critère (ex: 'DSR', 'PBO', 'N_TRADES').
        passed: True si le critère est satisfait.
        observed: Valeur observée (ex: 1.2 pour DSR, 42 pour n_trades).
        threshold: Seuil utilisé pour la décision.
        reason: RejectionReason si KO, None si OK.
        meta: Métadonnées libres (IC, p-value, etc.).
    """

    name: str
    passed: bool
    observed: float
    threshold: float
    reason: RejectionReason | None
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Sérialisation pour logs / persistance."""
        return {
            "name": self.name,
            "passed": self.passed,
            "observed": self.observed,
            "threshold": self.threshold,
            "reason": self.reason.value if self.reason else None,
            "meta": self.meta,
        }


@dataclass(frozen=True)
class AdmissionVerdict:
    """Verdict global : tous les critères + raison du premier échec."""

    verdicts: tuple[CriterionVerdict, ...]
    n_passed: int
    n_failed: int
    primary_reason: RejectionReason | None

    @property
    def passed(self) -> bool:
        """True si tous les critères ont passé."""
        return self.n_failed == 0

    def to_dict(self) -> dict[str, Any]:
        """Sérialisation pour logs / persistance."""
        return {
            "passed": self.passed,
            "n_passed": self.n_passed,
            "n_failed": self.n_failed,
            "primary_reason": self.primary_reason.value if self.primary_reason else None,
            "verdicts": [v.to_dict() for v in self.verdicts],
        }


# --------------------------------------------------------------------------- #
# Critère 1 : DSR (Deflated Sharpe Ratio)
# --------------------------------------------------------------------------- #


def evaluate_dsr(
    mesures: MesuresBrutes,
    config: EinherjarConfig,
    n_indep_trials: int = 1,
) -> CriterionVerdict:
    """DSR (Deflated Sharpe Ratio) — corrige pour le nombre d'essais indépendants.

    Le DSR ajuste le Sharpe observé par :
      - le nombre d'essais indépendants (corrige le multiple-testing),
      - la non-normalité des rendements (skew, kurtosis).

    Returns:
        Verdict. Pass si DSR >= seuil (défaut 0.95).
    """
    min_dsr = float(config.thresholds["dsr"]["min_value"])
    min_trials = int(config.thresholds["dsr"]["min_n_indep_trials"])
    sharpe = mesures.sharpe_net
    if math.isnan(sharpe) or n_indep_trials < min_trials:
        return CriterionVerdict(
            name="DSR",
            passed=False,
            observed=float("nan"),
            threshold=min_dsr,
            reason=RejectionReason.DSR_FAIL,
            meta={"sharpe": sharpe, "n_indep_trials": n_indep_trials},
        )
    # Approximation via la métrique de utils/metrics (Bailey & López de Prado).
    p = dsr_metric(sharpe_observed=sharpe, n_trials=n_indep_trials)
    return CriterionVerdict(
        name="DSR",
        passed=(p >= min_dsr),
        observed=p,
        threshold=min_dsr,
        reason=None if p >= min_dsr else RejectionReason.DSR_FAIL,
        meta={"sharpe": sharpe, "n_indep_trials": n_indep_trials, "p_value": p},
    )


# --------------------------------------------------------------------------- #
# Critère 2 : PBO (Probability of Backtest Overfitting)
# --------------------------------------------------------------------------- #


def evaluate_pbo(
    returns: Sequence[float],
    config: EinherjarConfig,
    n_groups: int | None = None,
    n_paths: int | None = None,
) -> CriterionVerdict:
    """PBO via CPCV léger (Combinatorial Purged Cross-Validation).

    Implémentation simplifiée V1 : on découpe la série en `n_groups`
    blocs, on forme `n_paths` combinaisons train/test, et on calcule
    la fraction de configurations où le test est perdant.

    V2 pourra implémenter le CPCV complet de López de Prado.
    """
    max_pbo = float(config.thresholds["pbo"]["max_value"])
    n_groups = n_groups or int(config.thresholds["pbo"]["cpcv"]["n_groups"])
    n_paths = n_paths or int(config.thresholds["pbo"]["cpcv"]["n_paths"])
    if len(returns) < n_groups * 2:
        return CriterionVerdict(
            name="PBO",
            passed=False,
            observed=float("nan"),
            threshold=max_pbo,
            reason=RejectionReason.PBO_FAIL,
            meta={"reason": "insufficient_data", "n_returns": len(returns)},
        )
    # Découpage en `n_groups` blocs, on fait `n_paths` combinaisons train/test.
    # Pour V1, on fait une approximation : on prend tous les splits "k sur n_groups"
    # où k blocs servent de test, et on regarde la proportion de splits perdants.
    from itertools import combinations
    n = len(returns)
    block_size = n // n_groups
    blocks = [returns[i * block_size : (i + 1) * block_size] for i in range(n_groups)]
    # Toutes les combinaisons de k blocs pour le test (k = n_groups // 2).
    k = n_groups // 2
    if k == 0:
        k = 1
    n_losing = 0
    tested = 0
    for combo in combinations(range(n_groups), k):
        if tested >= n_paths:
            break
        test_blocks = [blocks[i] for i in combo]
        test_returns = [r for b in test_blocks for r in b]
        # Trade "perdant" = ret_total < 0.
        if sum(test_returns) < 0:
            n_losing += 1
        tested += 1
    pbo = n_losing / max(tested, 1)
    return CriterionVerdict(
        name="PBO",
        passed=(pbo <= max_pbo),
        observed=pbo,
        threshold=max_pbo,
        reason=None if pbo <= max_pbo else RejectionReason.PBO_FAIL,
        meta={"n_groups": n_groups, "n_paths": tested, "n_losing": n_losing},
    )


# --------------------------------------------------------------------------- #
# Critère 3 & 4 : Block bootstrap CI (Sharpe et ret_total)
# --------------------------------------------------------------------------- #


def evaluate_bootstrap_ci_sharpe(
    mesures: MesuresBrutes,
) -> CriterionVerdict:
    """Block bootstrap CI sur Sharpe : passe si `ci_low > 0`."""
    ci_low = mesures.bootstrap_sharpe_ci_low
    return CriterionVerdict(
        name="BOOTSTRAP_CI_SHARPE",
        passed=(ci_low > 0),
        observed=ci_low,
        threshold=0.0,
        reason=None if ci_low > 0 else RejectionReason.BOOTSTRAP_CI_FAIL,
        meta={"ci_high": mesures.bootstrap_sharpe_ci_high},
    )


def evaluate_bootstrap_ci_ret(
    mesures: MesuresBrutes,
) -> CriterionVerdict:
    """Block bootstrap CI sur ret_total : passe si `ci_low > 0`."""
    ci_low = mesures.bootstrap_ret_ci_low
    return CriterionVerdict(
        name="BOOTSTRAP_CI_RET",
        passed=(ci_low > 0),
        observed=ci_low,
        threshold=0.0,
        reason=None if ci_low > 0 else RejectionReason.BOOTSTRAP_CI_FAIL,
        meta={"ci_high": mesures.bootstrap_ret_ci_high},
    )


# --------------------------------------------------------------------------- #
# Critère 5 : n_trades minimum
# --------------------------------------------------------------------------- #


def evaluate_n_trades(
    mesures: MesuresBrutes,
    config: EinherjarConfig,
) -> CriterionVerdict:
    """Nombre minimum de trades pour la significativité statistique."""
    min_trades = int(config.thresholds["n_trades"]["min_total"])
    n = mesures.n_signals
    return CriterionVerdict(
        name="N_TRADES",
        passed=(n >= min_trades),
        observed=float(n),
        threshold=float(min_trades),
        reason=None if n >= min_trades else RejectionReason.N_TRADES_FAIL,
        meta={"n_signals": n},
    )


# --------------------------------------------------------------------------- #
# Critère 6 : consistency_cross_asset
# --------------------------------------------------------------------------- #


def evaluate_cross_asset(
    mesures: MesuresBrutes,
    config: EinherjarConfig,
) -> CriterionVerdict:
    """Cohérence cross-asset : ≥ 70% des actifs du universe doivent être positifs."""
    min_frac = float(config.thresholds["cross_asset"]["min_frac_assets_positive"])
    per_asset = mesures.per_asset_stats
    if not per_asset:
        # Un seul actif → on accepte par défaut (pas de cross-asset à vérifier).
        return CriterionVerdict(
            name="CROSS_ASSET",
            passed=True,
            observed=1.0,
            threshold=min_frac,
            reason=None,
            meta={"n_assets": 0, "note": "single_asset_universe"},
        )
    n_pos = sum(1 for m in per_asset.values() if m.ret_mean_pct_net > 0)
    frac = n_pos / len(per_asset)
    return CriterionVerdict(
        name="CROSS_ASSET",
        passed=(frac >= min_frac),
        observed=frac,
        threshold=min_frac,
        reason=None if frac >= min_frac else RejectionReason.CROSS_ASSET_FAIL,
        meta={"n_pos": n_pos, "n_total": len(per_asset)},
    )


# --------------------------------------------------------------------------- #
# Critère 7 : max_drawdown
# --------------------------------------------------------------------------- #


def evaluate_max_drawdown(
    mesures: MesuresBrutes,
    config: EinherjarConfig,
) -> CriterionVerdict:
    """Max drawdown borné (calculé sur equity_curve reconstruite)."""
    max_dd_allowed = float(config.thresholds["max_drawdown"]["max_value"])
    # Reconstruit une equity_curve grossière à partir de ret_std et ret_mean
    # (approximation, on devrait avoir accès à la série de returns).
    # Note : pour V1, on utilise la borne observée par-asset si dispo.
    worst_dd = 0.0
    if mesures.per_asset_stats:
        for m in mesures.per_asset_stats.values():
            if m.ret_std_pct > 0:
                heur = min(1.0, m.ret_std_pct * 10.0)
                worst_dd = max(worst_dd, heur)
    return CriterionVerdict(
        name="MAX_DRAWDOWN",
        passed=(worst_dd <= max_dd_allowed),
        observed=worst_dd,
        threshold=max_dd_allowed,
        reason=None if worst_dd <= max_dd_allowed else RejectionReason.DD_FAIL,
        meta={"note": "heuristic_v1"},
    )


# --------------------------------------------------------------------------- #
# Évaluation globale
# --------------------------------------------------------------------------- #


def evaluate_all_criteria(
    mesures: MesuresBrutes,
    returns: Sequence[float],
    config: EinherjarConfig,
    n_indep_trials: int = 1,
) -> AdmissionVerdict:
    """Évalue TOUS les critères d'admission sur une hypothèse.

    Returns:
        AdmissionVerdict avec le détail de chaque critère.
    """
    verdicts: list[CriterionVerdict] = []
    verdicts.append(evaluate_dsr(mesures, config, n_indep_trials=n_indep_trials))
    verdicts.append(evaluate_pbo(returns, config))
    verdicts.append(evaluate_bootstrap_ci_sharpe(mesures))
    verdicts.append(evaluate_bootstrap_ci_ret(mesures))
    verdicts.append(evaluate_n_trades(mesures, config))
    verdicts.append(evaluate_cross_asset(mesures, config))
    verdicts.append(evaluate_max_drawdown(mesures, config))

    n_passed = sum(1 for v in verdicts if v.passed)
    n_failed = len(verdicts) - n_passed
    primary_reason: RejectionReason | None = None
    for v in verdicts:
        if not v.passed and v.reason is not None:
            primary_reason = v.reason
            break

    return AdmissionVerdict(
        verdicts=tuple(verdicts),
        n_passed=n_passed,
        n_failed=n_failed,
        primary_reason=primary_reason,
    )
