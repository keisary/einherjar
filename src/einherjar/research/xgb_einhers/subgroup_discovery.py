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
    5. Combine top selectors into depth-2 pairs (AND), avoiding same-feature pairs.
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
DEFAULT_MAX_DEPTH2_PAIRS = 200  # max depth-2 candidates before dedup


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
    depth: int  # 1 = atomic, 2 = pair
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
) -> list[SubgroupCandidate]:
    """Find subgroups with significantly positive returns using WRAcc + t-stat.

    Tests atomic selectors (binary ==1/==-1, continuous quantile intervals),
    scores with WRAcc, filters by t-stat and coverage, combines top selectors
    into depth-2 pairs, and deduplicates by Jaccard similarity.

    Args:
        X_train: (N, F) float32 feature matrix.
        y_train: (N,) signed returns.
        feature_names: list of F feature names.
        binary_mask: (F,) bool, True for binary features. Auto-detected if None.
        min_t_stat: minimum |t-stat| to keep a selector (default 3.0).
        min_coverage: minimum fraction of population covered (default 0.02).
        jaccard_threshold: dedup threshold for mask similarity (default 0.8).
        top_k_atomic: number of top atomic selectors for depth-2 pairing (default 50).
        max_depth2_pairs: max depth-2 candidates before dedup (default 200).

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
    baseline = float(y_positive.mean())

    # Precompute quantile thresholds for continuous features (vectorized)
    # Shape: (n_features, 3) for q25, q50, q75
    quantile_levels = np.array([0.25, 0.50, 0.75])
    # Only compute for non-binary features
    continuous_mask = ~binary_mask
    cont_indices = np.where(continuous_mask)[0]

    thresholds = np.full((n_features, 3), np.nan, dtype=np.float64)
    if len(cont_indices) > 0:
        cont_cols = X_train[:, cont_indices].astype(np.float64)
        # nanpercentile along axis=0 gives (3, n_cont_features)
        q_vals = np.nanpercentile(cont_cols, [25, 50, 75], axis=0)
        thresholds[cont_indices, :] = q_vals.T  # (n_cont, 3)

    # -----------------------------------------------------------------------
    # Phase 1: Generate and score atomic selectors
    # -----------------------------------------------------------------------
    atomic_candidates: list[SubgroupCandidate] = []
    n_tested = 0

    # Binary features: test ==1 and ==-1
    bin_indices = np.where(binary_mask)[0]
    for j in bin_indices:
        col = X_train[:, j]
        fname = feature_names[j]
        for val, op_label in [(1.0, "==1"), (-1.0, "==-1")]:
            # Vectorized mask: |col - val| < 0.5 (handles float noise)
            mask = np.abs(col - val) < 0.5
            n_sg = int(mask.sum())
            n_tested += 1
            if n_sg == 0:
                continue
            coverage = n_sg / n_total
            if coverage < min_coverage:
                continue

            y_sg = y[mask]
            wracc = _compute_wracc(mask, y_positive, n_total, baseline)
            t_stat = _compute_t_stat(y_sg)

            if abs(t_stat) < min_t_stat:
                continue

            direction = "BUY" if float(np.mean(y_sg)) > 0 else "SELL"
            atomic_candidates.append(SubgroupCandidate(
                description=((fname, "==", val),),
                mask=mask,
                wracc=wracc,
                t_stat=t_stat,
                coverage=coverage,
                n_occurrences=n_sg,
                mean_return=float(np.mean(y_sg)),
                depth=1,
                direction=direction,
            ))

    # Continuous features: quantile-based intervals
    # 6 selectors per feature: <=q25, >q25, <=q50, >q50, <=q75, >q75
    for j in cont_indices:
        col = X_train[:, j].astype(np.float64)
        fname = feature_names[j]
        q25, q50, q75 = thresholds[j]

        # Skip if all NaN
        if np.isnan(q25):
            continue

        for thresh, q_label in [
            (q25, "q25"), (q50, "q50"), (q75, "q75"),
        ]:
            for op, op_str in [("<=", "<="), (">", ">")]:
                # Vectorized mask
                if op == "<=":
                    mask = col <= thresh
                else:
                    mask = col > thresh

                n_sg = int(mask.sum())
                n_tested += 1
                if n_sg == 0:
                    continue
                coverage = n_sg / n_total
                if coverage < min_coverage:
                    continue

                y_sg = y[mask]
                wracc = _compute_wracc(mask, y_positive, n_total, baseline)
                t_stat = _compute_t_stat(y_sg)

                if abs(t_stat) < min_t_stat:
                    continue

                direction = "BUY" if float(np.mean(y_sg)) > 0 else "SELL"
                atomic_candidates.append(SubgroupCandidate(
                    description=((fname, op_str, float(thresh)),),
                    mask=mask,
                    wracc=wracc,
                    t_stat=t_stat,
                    coverage=coverage,
                    n_occurrences=n_sg,
                    mean_return=float(np.mean(y_sg)),
                    depth=1,
                    direction=direction,
                ))

    logger.info(
        "subgroup_discovery: %d atomic selectors tested, %d candidates "
        "(|t|>%.1f, coverage>=%.1f%%)",
        n_tested, len(atomic_candidates), min_t_stat, min_coverage * 100,
    )

    # Sort atomic by |t_stat| descending
    atomic_candidates.sort(key=lambda c: -abs(c.t_stat))

    # -----------------------------------------------------------------------
    # Phase 2: Depth-2 pairs (AND of top atomic selectors)
    # -----------------------------------------------------------------------
    top_atomic = atomic_candidates[:top_k_atomic]
    depth2_candidates: list[SubgroupCandidate] = []
    n_pairs_tested = 0

    # Build feature index for each atomic candidate to avoid same-feature pairs
    def _get_feature(desc: tuple) -> str:
        return desc[0][0]

    for i in range(len(top_atomic)):
        if n_pairs_tested >= max_depth2_pairs:
            break
        for j in range(i + 1, len(top_atomic)):
            if n_pairs_tested >= max_depth2_pairs:
                break
            a = top_atomic[i]
            b = top_atomic[j]
            # Avoid same-feature pairs
            if _get_feature(a.description) == _get_feature(b.description):
                continue

            n_pairs_tested += 1
            combined_mask = a.mask & b.mask
            n_sg = int(combined_mask.sum())
            if n_sg == 0:
                continue
            coverage = n_sg / n_total
            if coverage < min_coverage:
                continue

            y_sg = y[combined_mask]
            wracc = _compute_wracc(combined_mask, y_positive, n_total, baseline)
            t_stat = _compute_t_stat(y_sg)

            if abs(t_stat) < min_t_stat:
                continue

            direction = "BUY" if float(np.mean(y_sg)) > 0 else "SELL"
            combined_desc = a.description + b.description
            depth2_candidates.append(SubgroupCandidate(
                description=combined_desc,
                mask=combined_mask,
                wracc=wracc,
                t_stat=t_stat,
                coverage=coverage,
                n_occurrences=n_sg,
                mean_return=float(np.mean(y_sg)),
                depth=2,
                direction=direction,
            ))

    logger.info(
        "subgroup_discovery: %d depth-2 pairs tested, %d candidates",
        n_pairs_tested, len(depth2_candidates),
    )

    # -----------------------------------------------------------------------
    # Phase 3: Merge, deduplicate by Jaccard, sort
    # -----------------------------------------------------------------------
    all_candidates = atomic_candidates + depth2_candidates
    all_candidates.sort(key=lambda c: -abs(c.t_stat))

    # Deduplicate by Jaccard similarity of masks
    deduped: list[SubgroupCandidate] = []
    for cand in all_candidates:
        is_dup = False
        for existing in deduped:
            if _jaccard(cand.mask, existing.mask) >= jaccard_threshold:
                # Keep the one with higher |t_stat| (already sorted, so existing wins)
                is_dup = True
                break
        if not is_dup:
            deduped.append(cand)

    logger.info(
        "subgroup_discovery: %d total candidates -> %d after Jaccard dedup (threshold=%.2f)",
        len(all_candidates), len(deduped), jaccard_threshold,
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
    max_candidates: int = 30,
    source_tag: str = "subgroup_discovery",
) -> list[Einher]:
    """Convert SubgroupCandidate list to Einher objects.

    Each candidate's description tuple is converted to a ConditionNode (AND
    for depth-2) or a single Condition (depth-1).

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
