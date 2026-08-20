"""admission.py — Porte d'admission C1-C6 (plan section C).

Un candidat entre au corpus seulement s'il passe TOUS les contrôles :
- C1 : split temporel purgé + embargo (fait EN AMONT par temporal_split, xgb)
- C6 : CI bootstrap par blocs sur val : borne basse > 0 (test primaire)
- C2 : DSR > 0.95 (Prob(Sharpe vrai > benchmark), correction sélection)
- C5 : FDR Benjamini-Hochberg α=0.05 sur les p-values val du batch
- Dédup : fingerprint exact + Jaccard des features < 0.30 et corrélation des
  signaux < 0.50 contre les candidats déjà admis (ordre : fitness val décroissante)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from einherjar.research.search_engine.bootstrap import block_bootstrap_ci
from einherjar.research.search_engine.dsr import dsr_probability

DEFAULT_MIN_TRADES = 30
DEFAULT_DSR_THRESHOLD = 0.95
DEFAULT_FDR_ALPHA = 0.05
DEFAULT_DUP_JACCARD = 0.30
DEFAULT_DUP_CORR = 0.50


@dataclass
class Candidate:
    einher: Any
    val_mask: np.ndarray
    features: set[str]
    fingerprint: str
    holdout_metrics: Any = None


@dataclass
class AdmissionOutcome:
    admitted: bool
    reasons: dict[str, Any] = field(default_factory=dict)


def benjamini_hochberg(pvalues: list[float], alpha: float = DEFAULT_FDR_ALPHA) -> set[int]:
    """Indices des p-values rejetées (significatives) par la procédure BH."""
    n = len(pvalues)
    if n == 0:
        return set()
    order = sorted(range(n), key=lambda i: pvalues[i])
    k_max = 0
    for k in range(1, n + 1):
        if pvalues[order[k - 1]] <= (k / n) * alpha:
            k_max = k
    return {order[i] for i in range(k_max)}


def jaccard(a: set[str], b: set[str]) -> float:
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _mask_corr(m_a: np.ndarray, m_b: np.ndarray) -> float:
    a, b = m_a.astype(np.float64), m_b.astype(np.float64)
    sa, sb = a.std(), b.std()
    if sa == 0.0 or sb == 0.0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def admit_batch(
    candidates: list[Candidate],
    *,
    initial_accepted: list[Candidate] | None = None,
    min_trades: int = DEFAULT_MIN_TRADES,
    dsr_threshold: float = DEFAULT_DSR_THRESHOLD,
    fdr_alpha: float = DEFAULT_FDR_ALPHA,
    dup_jaccard: float = DEFAULT_DUP_JACCARD,
    dup_corr: float = DEFAULT_DUP_CORR,
    require_beat_holdout_bh: bool = True,
    seed: int = 0,
) -> list[AdmissionOutcome]:
    """Évalue le batch complet ; l'ordre des candidats détermine le dédup."""
    outcomes: list[AdmissionOutcome] = []
    if not candidates:
        return outcomes

    # Ordre de traitement : fitness val décroissante (les meilleurs passent
    # d'abord → le dédup garde les plus forts)
    order = sorted(range(len(candidates)), key=lambda i: -candidates[i].einher.metrics.sharpe_ratio)

    pvalues = [float(c.einher.metrics.p_value) for c in candidates]
    fdr_pass = benjamini_hochberg(pvalues, alpha=fdr_alpha)

    accepted: list[Candidate] = list(initial_accepted or [])
    per_candidate: dict[int, AdmissionOutcome] = {}

    for i in order:
        c = candidates[i]
        m = c.einher.metrics
        n_trades = int(m.n_trades)
        reasons: dict[str, Any] = {
            "c1_split_embargo": True,  # garanti par temporal_split (amont)
        }

        # C6 : CI bootstrap par blocs (borne basse > 0)
        lo, hi, mean = block_bootstrap_ci(m.trade_returns, seed=seed)
        reasons["c6_bootstrap"] = {"lo": lo, "hi": hi, "mean": mean, "pass": bool(np.isfinite(lo) and lo > 0)}

        # C2 : DSR
        dsr = dsr_probability(m.trade_returns, min_trades=min_trades)
        reasons["c2_dsr"] = {"value": dsr, "pass": dsr >= dsr_threshold}

        # C5 : FDR (appartenance au set BH du batch)
        reasons["c5_fdr"] = {"pass": i in fdr_pass}

        # Dédup : vs candidats déjà admis
        dup_against: str | None = None
        for acc in accepted:
            j = jaccard(c.features, acc.features)
            corr = _mask_corr(c.val_mask, acc.val_mask)
            if j >= dup_jaccard or corr >= dup_corr:
                dup_against = acc.fingerprint
                break
        reasons["dedup"] = {"duplicate_of": dup_against, "pass": dup_against is None}

        # C7 : battre le buy-and-hold sur le HOLD-OUT (test hors-échantillon).
        # La val peut être un bull run (beta pur) ; le holdout révèle l'alpha
        # réel. Si les métriques holdout ne sont pas fournies, porte désactivée.
        c7: bool = True
        c7_info = None
        if require_beat_holdout_bh and c.holdout_metrics is not None:
            ho = c.holdout_metrics
            bh = float(getattr(ho, "buy_hold_return", 0.0) or 0.0)
            total = float(getattr(ho, "total_return", 0.0) or 0.0)
            c7 = total >= bh
            c7_info = {"total_return": total, "buy_hold_return": bh, "pass": c7}
        reasons["c7_beat_bh_holdout"] = c7_info

        ok = (
            n_trades >= min_trades
            and reasons["c6_bootstrap"]["pass"]
            and reasons["c2_dsr"]["pass"]
            and reasons["c5_fdr"]["pass"]
            and reasons["dedup"]["pass"]
            and c7
        )
        reasons["min_trades"] = n_trades >= min_trades
        per_candidate[i] = AdmissionOutcome(admitted=ok, reasons=reasons)
        if ok:
            accepted.append(c)

    return [per_candidate[i] for i in range(len(candidates))]