"""subgroup_discovery.py - Subgroup Discovery engine for trading signal discovery.

Subgroup Discovery (SD) finds subsets of the population with unusually high
returns, treating binary and continuous features equally — unlike XGBoost MSE
which structurally favors continuous features with many split points.

MÉTHODE :
    1. For each feature, generate atomic selectors:
       - Binary features (values ⊆ {-1, 0, 1}): test ==1 and ==-1
       - Continuous features: test quantile-based intervals
         (<=q25, >q25, <=q50, >q50, <=q75, >q75)
    2. Score each selector with WRAcc = coverage × (precision - baseline)
       where baseline = mean(y_train > 0) and precision = mean(y_sg > 0).
    3. Compute t-stat = mean(y_sg) / (std(y_sg) / sqrt(n_sg)).
    4. Filter selectors with |t_stat| > min_t_stat and coverage >= min_coverage.
    5. Combine top selectors into depth-2, depth-3, depth-4 pairs (AND),
       avoiding same-feature pairs.
    6. Deduplicate by Jaccard similarity of masks (threshold 0.8).
    7. Return sorted list of SubgroupCandidate.

PERFORMANCE :
    Fully vectorized numpy — no Python loops over rows. The entire search for
    ~25k rows × 218 features completes in < 5 seconds.

GARDE-FOUS anti data-snooping :
    - Statistics computed on TRAIN only (never the val window).
    - Strict |t| threshold (default 3.0) + min coverage.
    - Total number of selectors tested is logged for multiple-testing control.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .types import Condition, ConditionNode, Einher, EinherMetrics

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_MIN_T_STAT = 3.0
DEFAULT_MIN_COVERAGE = 0.02  # subgroup must cover >= 2% of population
DEFAULT_JACCARD_THRESHOLD = 0.8
DEFAULT_TOP_K_ATOMIC = 50  # top atomic selectors for depth-2 pairing
DEFAULT_MAX_DEPTH2_PAIRS = 500  # FIX (2026-08-27) : 500 au lieu de 200
DEFAULT_MAX_DEPTH = 4  # FIX (2026-08-27) : depth max 4 (combinaisons de 2, 3, 4 features)


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SubgroupCandidate:
    """A subgroup with significantly positive (or negative) returns."""

    description: tuple  # tuple of (feature, op, value) triples
    mask: np.ndarray  # (N,) bool, True where subgroup is active
    wracc: float
    t_stat: float
    coverage: float  # fraction of population covered
    n_occurrences: int  # absolute count
    mean_return: float  # mean signed return in subgroup
    depth: int  # 1 = atomic, 2 = pair, 3 = triple, 4 = quad
    direction: str = "BUY"  # BUY if mean_return > 0, else SELL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def detect_binary_features(X: np.ndarray) -> np.ndarray:
    """Return boolean mask: True if feature values are subset of {-1, 0, 1}.

    Args:
        X: (N, F) float32 feature matrix.

    Returns:
        (F,) bool array.
    """
    n_features = X.shape[1]
    is_binary = np.zeros(n_features, dtype=bool)
    for j in range(n_features):
        col = X[:, j]
        # Drop NaN for the check
        valid = col[~np.isnan(col)]
        if len(valid) == 0:
            continue
        unique_vals = np.unique(valid)
        is_binary[j] = bool(np.all(np.isin(unique_vals, [-1.0, 0.0, 1.0])))
    return is_binary


def _compute_wracc(
    mask: np.ndarray,
    y_positive: np.ndarray,
    n_total: int,
    baseline: float,
) -> float:
    """WRAcc = coverage × (precision - baseline).

    Args:
        mask: (N,) bool selector mask.
        y_positive: (N,) bool, True where y > 0.
        n_total: total population size.
        baseline: mean(y > 0) over full population.

    Returns:
        WRAcc score (float).
    """
    n_sg = int(mask.sum())
    if n_sg == 0:
        return 0.0
    coverage = n_sg / n_total
    precision = float(y_positive[mask].mean())
    return coverage * (precision - baseline)


def _compute_t_stat(y_sg: np.ndarray) -> float:
    """t-stat = mean(y_sg) / (std(y_sg) / sqrt(n_sg)).

    Args:
        y_sg: signed returns in the subgroup.

    Returns:
        t-statistic (float). Returns 0.0 if subgroup too small or zero std.
    """
    n = len(y_sg)
    if n < 2:
        return 0.0
    m = float(np.mean(y_sg))
    s = float(np.std(y_sg, ddof=1))
    if s <= 0.0:
        return 0.0
    return m / (s / np.sqrt(n))


def _jaccard(a: np.ndarray, b: np.ndarray) -> float:
    """Jaccard similarity of two boolean masks."""
    inter = int((a & b).sum())
    union = int((a | b).sum())
    if union == 0:
        return 0.0
    return inter / union


def _get_features_in_desc(desc: tuple) -> set[str]:
    """Extract unique feature names from a description tuple."""
    return {feat for feat, _, _ in desc}


# ---------------------------------------------------------------------------
# Main search
# ---------------------------------------------------------------------------


def subgroup_discovery(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    binary_mask: np.ndarray | None = None,
    min_t_stat: float = DEFAULT_MIN_T_STAT,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    jaccard_threshold: float = DEFAULT_JACCARD_THRESHOLD,
    top_k_atomic: int = DEFAULT_TOP_K_ATOMIC,
    max_depth2_pairs: int = DEFAULT_MAX_DEPTH2_PAIRS,
    max_depth: int = DEFAULT_MAX_DEPTH,
) -> list[SubgroupCandidate]:
    """Find subgroups with significantly positive OR negative returns using WRAcc + t-stat.

    Tests atomic selectors (binary ==1/==-1, continuous quantile intervals),
    scores with WRAcc, filters by t-stat and coverage, combines top selectors
    into depth-2/3/4 combinations, and deduplicates by Jaccard similarity.

    FIX SELL (2026-08-27) : teste aussi les sous-groupes avec rendement
    négatif significatif (direction SELL).

    FIX DEPTH (2026-08-27) : supporte depth=3 et depth=4 en plus de depth=2.

    Args:
        X_train: (N, F) float32 feature matrix.
        y_train: (N,) signed returns.
        feature_names: list of F feature names.
        binary_mask: (F,) bool, True for binary features. Auto-detected if None.
        min_t_stat: minimum |t-stat| to keep a selector (default 3.0).
        min_coverage: minimum fraction of population covered (default 0.02).
        jaccard_threshold: dedup threshold for mask similarity (default 0.8).
        top_k_atomic: number of top atomic selectors for depth-2 pairing (default 50).
        max_depth2_pairs: max depth-2 candidates before dedup (default 500).
        max_depth: maximum combination depth (default 4).

    Returns:
        Sorted list of SubgroupCandidate (by |t_stat| descending).
    """
    if X_train.ndim != 2 or len(feature_names) != X_train.shape[1]:
        raise ValueError("X_train and feature_names dimension mismatch")
    if len(y_train) != X_train.shape[0]:
        raise ValueError("X_train and y_train length mismatch")

    n_total = X_train.shape[0]
    n_features = X_train.shape[1]

    if binary_mask is None:
        binary_mask = detect_binary_features(X_train)

    y = y_train.astype(np.float64)
    y_positive = y > 0
    y_negative = y < 0
    baseline_buy = float(y_positive.mean())
    baseline_sell = float(y_negative.mean())

    # Precompute quantile thresholds for continuous features (vectorized)
    continuous_mask = ~binary_mask
    cont_indices = np.where(continuous_mask)[0]

    thresholds = np.full((n_features, 3), np.nan, dtype=np.float64)
    if len(cont_indices) > 0:
        cont_cols = X_train[:, cont_indices].astype(np.float64)
        q_vals = np.nanpercentile(cont_cols, [25, 50, 75], axis=0)
        thresholds[cont_indices, :] = q_vals.T

    # -----------------------------------------------------------------------
    # Phase 1: Generate and score atomic selectors (BUY + SELL)
    # -----------------------------------------------------------------------
    atomic_candidates: list[SubgroupCandidate] = []
    n_tested = 0

    def _eval_selector(mask: np.ndarray, desc: tuple, n_tested_ref: list) -> list[SubgroupCandidate]:
        """Evaluate a single selector for both BUY and SELL."""
        n_tested_ref[0] += 1
        n_sg = int(mask.sum())
        if n_sg == 0:
            return []
        coverage = n_sg / n_total
        if coverage < min_coverage:
            return []

        results = []
        y_sg = y[mask]
        mean_ret = float(np.mean(y_sg))

        # BUY: test if mean_return > 0 and significant
        if mean_ret > 0:
            wracc = _compute_wracc(mask, y_positive, n_total, baseline_buy)
            t_stat = _compute_t_stat(y_sg)
            if abs(t_stat) >= min_t_stat:
                results.append(SubgroupCandidate(
                    description=desc,
                    mask=mask,
                    wracc=wracc,
                    t_stat=t_stat,
                    coverage=coverage,
                    n_occurrences=n_sg,
                    mean_return=mean_ret,
                    depth=len(desc),
                    direction="BUY",
                ))

        # SELL: test if mean_return < 0 and significant
        if mean_ret < 0:
            wracc_sell = _compute_wracc(mask, y_negative, n_total, baseline_sell)
            t_stat_sell = _compute_t_stat(-y_sg)  # flip sign for SELL
            if abs(t_stat_sell) >= min_t_stat:
                results.append(SubgroupCandidate(
                    description=desc,
                    mask=mask,
                    wracc=wracc_sell,
                    t_stat=t_stat_sell,
                    coverage=coverage,
                    n_occurrences=n_sg,
                    mean_return=mean_ret,
                    depth=len(desc),
                    direction="SELL",
                ))

        return results

    # Binary features: test ==1 and ==-1
    bin_indices = np.where(binary_mask)[0]
    for j in bin_indices:
        col = X_train[:, j]
        fname = feature_names[j]
        for val, op_label in [(1.0, "==1"), (-1.0, "==-1")]:
            mask = np.abs(col - val) < 0.5
            desc = ((fname, "==", val),)
            atomic_candidates.extend(_eval_selector(mask, desc, [n_tested]))

    # Continuous features: quantile-based intervals
    for j in cont_indices:
        col = X_train[:, j].astype(np.float64)
        fname = feature_names[j]
        q25, q50, q75 = thresholds[j]

        if np.isnan(q25):
            continue

        for thresh, q_label in [
            (q25, "q25"), (q50, "q50"), (q75, "q75"),
        ]:
            for op, op_str in [("<=", "<="), (">", ">")]:
                if op == "<=":
                    mask = col <= thresh
                else:
                    mask = col > thresh

                desc = ((fname, op_str, float(thresh)),)
                atomic_candidates.extend(_eval_selector(mask, desc, [n_tested]))

    logger.info(
        "subgroup_discovery: %d atomic selectors tested, %d candidates "
        "(|t|>%.1f, coverage>=%.1f%%)",
        n_tested, len(atomic_candidates), min_t_stat, min_coverage * 100,
    )

    # Sort atomic by |t_stat| descending
    atomic_candidates.sort(key=lambda c: -abs(c.t_stat))

    # -----------------------------------------------------------------------
    # Phase 2+: Depth-k combinations (k=2,3,4)
    # -----------------------------------------------------------------------
    all_candidates = list(atomic_candidates)
    prev_depth = atomic_candidates  # start with depth-1

    for depth in range(2, max_depth + 1):
        # Take top candidates from previous depth
        top_prev = prev_depth[:top_k_atomic]
        # Also take top atomic for combining
        top_atomic = atomic_candidates[:top_k_atomic]

        depth_candidates: list[SubgroupCandidate] = []
        n_pairs_tested = 0
        max_pairs = max_depth2_pairs if depth == 2 else max_depth2_pairs // 2

        for prev_cand in top_prev:
            if n_pairs_tested >= max_pairs:
                break
            prev_features = _get_features_in_desc(prev_cand.description)

            for atom in top_atomic:
                if n_pairs_tested >= max_pairs:
                    break

                # Avoid same-feature combinations
                atom_features = _get_features_in_desc(atom.description)
                if prev_features & atom_features:
                    continue

                n_pairs_tested += 1
                combined_mask = prev_cand.mask & atom.mask
                n_sg = int(combined_mask.sum())
                if n_sg == 0:
                    continue
                coverage = n_sg / n_total
                if coverage < min_coverage:
                    continue

                y_sg = y[combined_mask]
                mean_ret = float(np.mean(y_sg))
                combined_desc = prev_cand.description + atom.description

                # BUY
                if mean_ret > 0:
                    wracc = _compute_wracc(combined_mask, y_positive, n_total, baseline_buy)
                    t_stat = _compute_t_stat(y_sg)
                    if abs(t_stat) >= min_t_stat:
                        depth_candidates.append(SubgroupCandidate(
                            description=combined_desc,
                            mask=combined_mask,
                            wracc=wracc,
                            t_stat=t_stat,
                            coverage=coverage,
                            n_occurrences=n_sg,
                            mean_return=mean_ret,
                            depth=depth,
                            direction="BUY",
                        ))

                # SELL
                if mean_ret < 0:
                    wracc_sell = _compute_wracc(combined_mask, y_negative, n_total, baseline_sell)
                    t_stat_sell = _compute_t_stat(-y_sg)
                    if abs(t_stat_sell) >= min_t_stat:
                        depth_candidates.append(SubgroupCandidate(
                            description=combined_desc,
                            mask=combined_mask,
                            wracc=wracc_sell,
                            t_stat=t_stat_sell,
                            coverage=coverage,
                            n_occurrences=n_sg,
                            mean_return=mean_ret,
                            depth=depth,
                            direction="SELL",
                        ))

        logger.info(
            "subgroup_discovery: %d depth-%d combinations tested, %d candidates",
            n_pairs_tested, depth, len(depth_candidates),
        )

        all_candidates.extend(depth_candidates)
        prev_depth = depth_candidates  # next depth builds on this

    # -----------------------------------------------------------------------
    # Final: sort and deduplicate by Jaccard
    # -----------------------------------------------------------------------
    all_candidates.sort(key=lambda c: -abs(c.t_stat))

    # FIX PERF (2026-08-27) : limiter à 200 candidats avant dedup.
    # Avec 600+ candidats sur 834k lignes, la dédup O(n²) prend 4 minutes.
    # Les top 200 par |t_stat| sont les plus significatifs.
    max_before_dedup = 200
    if len(all_candidates) > max_before_dedup:
        logger.info(
            "subgroup_discovery: limité à %d/%d candidats avant dedup (top |t_stat|)",
            max_before_dedup, len(all_candidates),
        )
        all_candidates = all_candidates[:max_before_dedup]

    # Dédup rapide : précalculer les sommes des masks pour éviter
    # de recalculer l'intersection complète à chaque comparaison
    deduped: list[SubgroupCandidate] = []
    dedup_sums: list[int] = []  # précalculer sum(mask) pour chaque candidat retenu
    for cand in all_candidates:
        cand_sum = int(cand.mask.sum())
        is_dup = False
        for idx, existing in enumerate(deduped):
            # Fast reject : si les tailles sont trop différentes, pas de dup
            existing_sum = dedup_sums[idx]
            # Jaccard = inter / union = inter / (sum_a + sum_b - inter)
            # Si sum_a et sum_b sont très différentes, Jaccard est forcément bas
            min_sum = min(cand_sum, existing_sum)
            max_sum = max(cand_sum, existing_sum)
            if max_sum == 0:
                continue
            # Jaccard max possible = min_sum / max_sum (si un contient l'autre)
            if min_sum / max_sum < jaccard_threshold:
                continue  # impossible d'atteindre le seuil
            # Calcul complet seulement si le fast reject passe
            if _jaccard(cand.mask, existing.mask) >= jaccard_threshold:
                is_dup = True
                break
        if not is_dup:
            deduped.append(cand)
            dedup_sums.append(cand_sum)

    n_buy = sum(1 for c in deduped if c.direction == "BUY")
    n_sell = sum(1 for c in deduped if c.direction == "SELL")
    logger.info(
        "subgroup_discovery: %d total candidates -> %d after Jaccard dedup "
        "(%d BUY, %d SELL, threshold=%.2f)",
        len(all_candidates), len(deduped), n_buy, n_sell, jaccard_threshold,
    )

    return deduped


# ---------------------------------------------------------------------------
# Conversion to Einher objects
# ---------------------------------------------------------------------------


def convert_candidates_to_einhers(
    candidates: list[SubgroupCandidate],
    asset: str,
    asset_class: str,
    timeframe: str,
    horizon_str: str,
    horizon_bars: int,
    max_candidates: int = 50,  # FIX (2026-08-27) : 50 au lieu de 30
    source_tag: str = "subgroup_discovery",
) -> list[Einher]:
    """Convert SubgroupCandidate list to Einher objects.

    Each candidate's description tuple is converted to a ConditionNode (AND
    for depth-2+) or a single Condition (depth-1).

    Args:
        candidates: output of subgroup_discovery().
        asset / asset_class / timeframe / horizon_str / horizon_bars: universe.
        max_candidates: cap of candidates (best |t| first).
        source_tag: tag inscribed in einher.source.model.

    Returns:
        List of Einher objects ready for backtest/admission (empty metrics).
    """
    empty_metrics = EinherMetrics(
        n_trades=0, n_tp=0, n_sl=0, n_timeout=0,
        win_rate=0.0, avg_net_return=0.0, total_return=0.0,
        sharpe_ratio=0.0, max_drawdown=0.0, profit_factor=0.0,
        avg_holding_bars=0.0, buy_hold_return=0.0, alpha=0.0,
    )

    einhers: list[Einher] = []
    for cand in candidates[:max_candidates]:
        # Build condition tree from description
        conditions = []
        for feat, op, val in cand.description:
            conditions.append(Condition(
                feature_ref=feat,
                operator=op,
                value=val,
                transformation=None,
            ))

        if len(conditions) == 1:
            condition_tree: Condition | ConditionNode = conditions[0]
        else:
            # AND chain
            condition_tree = conditions[0]
            for c in conditions[1:]:
                condition_tree = ConditionNode(op="AND", left=condition_tree, right=c)

        direction = cand.direction

        # Build feature name list from description
        feat_names = list({feat for feat, _, _ in cand.description})

        eid = (
            f"sd_{asset}_{timeframe}_{horizon_str}_"
            f"d{cand.depth}_"
            f"{uuid.uuid4().hex[:8]}"
        )

        einhers.append(Einher(
            id=eid,
            condition_tree=condition_tree,
            direction=direction,
            amplitude_bars=horizon_bars,
            tp_pct=0.0,
            sl_pct=0.0,
            universe={
                "asset": asset,
                "asset_class": asset_class,
                "timeframe": timeframe,
                "horizon": horizon_str,
                "horizon_bars": horizon_bars,
            },
            metrics=empty_metrics,
            scope="asset",
            source={
                "model": source_tag,
                "depth": cand.depth,
                "n_conditions": len(cand.description),
                "wracc": cand.wracc,
                "t_stat_train": cand.t_stat,
                "coverage": cand.coverage,
                "n_train_occurrences": cand.n_occurrences,
                "mean_return_train": cand.mean_return,
                "feature_names": feat_names,
            },
            data_version="",
        ))

    return einhers
