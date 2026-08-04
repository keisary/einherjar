"""admission/criteria.py — Les 7 critères d'admission S-3.4 (UN fichier par convention).

Chaque critère est une fonction pure (ou un dataclass résultat) qui prend
en entrée les MesuresBrutes + métriques auxiliaires et retourne un verdict
(booléen) + un motif d'échec (RejectionReason) si KO.

Critères implémentés :
  1. DSR (Deflated Sharpe Ratio) — corrige pour le nombre d'essais indépendants.
  2. PBO (Probability of Backtest Overfitting) — CPCV K=6, embargo configurable.
  3. Block bootstrap CI sur Sharpe — `sharpe_ci_low > 0`.
  4. Block bootstrap CI sur ret_total — `ret_total_ci_low > 0`.
  5. n_trades minimum — significativité statistique.
  6. consistency_cross_asset — ≥ 70% des actifs positifs (≥ 2 actifs).
  7. max_drawdown — sur equity_curve par trade (vrai calcul, pas heuristique).

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
from einherjar.research.utils.stats import max_drawdown_from_returns
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
    candidate_paths: Sequence[Sequence[tuple[int, int, float]]] | None = None,
) -> CriterionVerdict:
    """PBO via CPCV (Combinatorial Purged Cross-Validation) — López de Prado §12.

    `candidate_paths` contient une ligne par candidat et, pour chaque trade,
    `(entry_idx, exit_idx, ret_pct_net)`. Le CPCV sélectionne le meilleur
    candidat in-sample pour chaque combinaison de groupes, puis mesure son
    rang out-of-sample. Sans cette matrice candidats x temps, aucun PBO
    statistiquement défendable ne peut être produit : le verdict échoue.

    Returns:
        Verdict. Pass si PBO <= seuil.
    """
    max_pbo = float(config.thresholds["pbo"]["max_value"])
    n_groups = n_groups or int(config.thresholds["pbo"]["cpcv"]["n_groups"])
    n_paths = n_paths or int(config.thresholds["pbo"]["cpcv"]["n_paths"])
    embargo_pct = float(config.thresholds["pbo"]["cpcv"].get("embargo_pct", 0.01))
    if not candidate_paths or len(candidate_paths) < 2:
        return CriterionVerdict(
            name="PBO",
            passed=False,
            observed=float("nan"),
            threshold=max_pbo,
            reason=RejectionReason.PBO_FAIL,
            meta={"reason": "candidate_matrix_required", "n_candidates": len(candidate_paths or ())},
        )
    max_exit = max((exit_idx for path in candidate_paths for _, exit_idx, _ in path), default=-1)
    if max_exit < n_groups * 2:
        return CriterionVerdict(
            name="PBO", passed=False, observed=float("nan"), threshold=max_pbo,
            reason=RejectionReason.PBO_FAIL,
            meta={"reason": "insufficient_temporal_span", "max_exit": max_exit, "n_groups": n_groups},
        )
    from itertools import combinations
    block_size = math.ceil((max_exit + 1) / n_groups)
    k_test = max(1, n_groups // 2)
    embargo_n = max(1, math.ceil(block_size * embargo_pct))
    n_overfit = 0
    tested = 0
    for combo in combinations(range(n_groups), k_test):
        if tested >= n_paths:
            break
        test_groups = set(combo)
        train_groups = set(range(n_groups)) - test_groups
        train_scores: list[float] = []
        test_scores: list[float] = []
        for path in candidate_paths:
            train_returns: list[float] = []
            test_returns: list[float] = []
            for entry_idx, exit_idx, ret in path:
                entry_group = min(n_groups - 1, entry_idx // block_size)
                exit_group = min(n_groups - 1, exit_idx // block_size)
                # Purge : un trade traversant une frontiere de groupes est exclu.
                if entry_group != exit_group:
                    continue
                if entry_group in test_groups:
                    test_returns.append(ret)
                elif entry_group in train_groups:
                    # Embargo : pas de trade train juste apres une zone test.
                    if any(0 <= entry_idx - ((group + 1) * block_size) < embargo_n for group in test_groups):
                        continue
                    train_returns.append(ret)
            train_scores.append(_sharpe_or_neg_inf(train_returns))
            test_scores.append(_sharpe_or_neg_inf(test_returns))
        eligible = [i for i in range(len(candidate_paths)) if math.isfinite(train_scores[i]) and math.isfinite(test_scores[i])]
        if len(eligible) < 2:
            continue
        selected = max(eligible, key=lambda index: train_scores[index])
        oos_rank = 1 + sum(test_scores[index] < test_scores[selected] for index in eligible)
        percentile = oos_rank / (len(eligible) + 1)
        if percentile < 0.5:
            n_overfit += 1
        tested += 1
    if tested == 0:
        return CriterionVerdict(
            name="PBO",
            passed=False,
            observed=float("nan"),
            threshold=max_pbo,
            reason=RejectionReason.PBO_FAIL,
            meta={"reason": "no_valid_cpcv_splits", "n_groups": n_groups},
        )
    pbo = n_overfit / tested
    return CriterionVerdict(
        name="PBO",
        passed=(pbo <= max_pbo),
        observed=pbo,
        threshold=max_pbo,
        reason=None if pbo <= max_pbo else RejectionReason.PBO_FAIL,
        meta={
            "n_groups": n_groups,
            "n_paths": tested,
            "n_overfit": n_overfit,
            "k_test": k_test,
            "embargo_pct": embargo_pct,
            "embargo_n": embargo_n,
            "block_size": block_size,
            "n_candidates": len(candidate_paths),
            "method": "cpcv_rank_oos",
        },
    )


def _sharpe_or_neg_inf(returns: Sequence[float]) -> float:
    if len(returns) < 2:
        return float("-inf")
    mean = sum(returns) / len(returns)
    variance = sum((value - mean) ** 2 for value in returns) / (len(returns) - 1)
    if variance <= 0:
        return float("-inf")
    return mean / math.sqrt(variance)


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
    """Cohérence cross-asset : ≥ `min_frac_assets_positive` des actifs positifs
    ET au moins `min_n_assets` actifs testés.

    Si `min_n_assets` n'est pas atteint :
      - Si `allow_single_asset=true` (opt-in) → passe par défaut (DÉCONSEILLÉ).
      - Sinon → FAIL (exige un vrai test multi-actif).
    """
    min_frac = float(config.thresholds["cross_asset"]["min_frac_assets_positive"])
    min_n_assets = int(config.thresholds["cross_asset"].get("min_n_assets", 2))
    allow_single = bool(config.thresholds["cross_asset"].get("allow_single_asset", False))
    per_asset = mesures.per_asset_stats
    n_total = len(per_asset)
    # Pas assez d'actifs testés → FAIL sauf opt-in explicite.
    if n_total < min_n_assets:
        return CriterionVerdict(
            name="CROSS_ASSET",
            passed=allow_single,
            observed=0.0,
            threshold=min_frac,
            reason=None if allow_single else RejectionReason.CROSS_ASSET_FAIL,
            meta={
                "n_assets": n_total,
                "min_n_assets": min_n_assets,
                "note": "single_asset_universe" if allow_single else "insufficient_assets",
            },
        )
    n_pos = sum(1 for m in per_asset.values() if m.ret_mean_pct_net > 0)
    frac = n_pos / n_total
    return CriterionVerdict(
        name="CROSS_ASSET",
        passed=(frac >= min_frac),
        observed=frac,
        threshold=min_frac,
        reason=None if frac >= min_frac else RejectionReason.CROSS_ASSET_FAIL,
        meta={"n_pos": n_pos, "n_total": n_total},
    )


# --------------------------------------------------------------------------- #
# Critère 7 : max_drawdown
# --------------------------------------------------------------------------- #


def evaluate_max_drawdown(
    mesures: MesuresBrutes,
    config: EinherjarConfig,
) -> CriterionVerdict:
    """Max drawdown borné — calculé sur la courbe d'equity réelle (par trade).

    Le moteur d'évaluation expose `mesures.trades` (tuple de TradeMesure) avec
    `ret_pct_net` pour chaque trade. On reconstruit l'equity_curve
    (capital = 1.0 initial) et on mesure la chute max depuis un pic.

    Pour le cas multi-asset (per_asset_stats), on calcule aussi le DD par
    asset et on conserve le pire (le plus pénalisant pour l'admission).

    Returns:
        Verdict. Pass si max_dd <= max_dd_allowed (défaut 0.25 = -25%).
    """
    max_dd_allowed = float(config.thresholds["max_drawdown"]["max_value"])
    # DD global : sur l'equity_curve reconstruite à partir de tous les trades.
    global_returns = [t.ret_pct_net for t in mesures.trades]
    worst_dd = max_drawdown_from_returns(global_returns)
    # DD par-asset (le pire).
    per_asset_dd: dict[str, float] = {}
    for asset, sub in mesures.per_asset_stats.items():
        per_asset_dd[asset] = max_drawdown_from_returns([t.ret_pct_net for t in sub.trades])
        if per_asset_dd[asset] > worst_dd:
            worst_dd = per_asset_dd[asset]
    return CriterionVerdict(
        name="MAX_DRAWDOWN",
        passed=(worst_dd <= max_dd_allowed),
        observed=worst_dd,
        threshold=max_dd_allowed,
        reason=None if worst_dd <= max_dd_allowed else RejectionReason.DD_FAIL,
        meta={
            "method": "equity_curve_from_trades",
            "n_trades": len(mesures.trades),
            "per_asset_max_dd": per_asset_dd,
        },
    )


# --------------------------------------------------------------------------- #
# Évaluation globale
# --------------------------------------------------------------------------- #


def evaluate_all_criteria(
    mesures: MesuresBrutes,
    returns: Sequence[float],
    config: EinherjarConfig,
    n_indep_trials: int = 1,
    pbo_candidate_paths: Sequence[Sequence[tuple[int, int, float]]] | None = None,
    include_pbo: bool = True,
) -> AdmissionVerdict:
    """Évalue TOUS les critères d'admission sur une hypothèse.

    Returns:
        AdmissionVerdict avec le détail de chaque critère.
    """
    verdicts: list[CriterionVerdict] = []
    verdicts.append(evaluate_dsr(mesures, config, n_indep_trials=n_indep_trials))
    if include_pbo:
        verdicts.append(evaluate_pbo(returns, config, candidate_paths=pbo_candidate_paths))
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
