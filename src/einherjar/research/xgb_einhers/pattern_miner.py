"""pattern_miner.py - Générateur event-study pour les features binaires.

P3-1 (Phase 3, 2026-08-25) - Pluralisme des générateurs.

PROBLÈME RÉSOLU :
XGBoost avec objectif MSE + min_child_weight élevé est structurellement
aveugle aux événements rares conditionnels (patterns binaires actifs sur
60-300 bougies). L'event-study direct (t-stat sur le rendement conditionnel)
détecte ce signal sans entraînement de modèle.

Preuves (rapport ML 2026-08-24) :
- Event-study BTC/1h/6h val : 6/36 patterns significatifs (|t|>1.96),
  ~1 attendu par hasard ; rendements conditionnels jusqu'à +/-24bp/trade.
- Corpus xgb admis : 0 condition price_action/market_structure.

MÉTHODE :
Pour chaque feature binaire (value_type=boolean dans la taxonomie) :
    ret_cond = mean(Y_ret[train] | pattern actif)
    t = ret_cond / (std(Y_ret[actif]) / sqrt(n))
    si |t| > min_t_stat ET n >= min_occurrences :
        -> candidat Einher (condition atomique pattern==1 ou ==-1,
           direction = signe du rendement conditionnel, amplitude = horizon)

Ces candidats passent ensuite dans le MÊME circuit que ceux de XGBoost :
backtest val -> holdout -> admission BH -> corpus/archive.
La sélection finale reste méritocratique ; ce module élargit uniquement
qui a le droit d'être candidat.

GARDE-FOUS anti data-snooping :
- Statistiques calculées SUR LE TRAIN UNIQUEMENT (jamais la fenêtre val).
- Seuil |t| strict (défaut 3.0, soit p < ~0.003 unilatéral) + min n=60.
- Le nombre total de patterns testés est loggé pour un éventuel contrôle
  multiple-testing global ultérieur.
"""
from __future__ import annotations

import logging
import math
import re
import uuid
from dataclasses import dataclass

import numpy as np

from .types import Condition, Einher, EinherMetrics

logger = logging.getLogger(__name__)

# Garde-fous par défaut (alignés sur le rapport ML Phase 3)
DEFAULT_MIN_T_STAT = 3.0
DEFAULT_MIN_OCCURRENCES = 60


@dataclass(frozen=True)
class PatternCandidate:
    """Un pattern binaire retenu par l'event-study train."""

    feature: str
    active_value: float          # 1.0 (pattern présent) ou -1.0 (sens inverse)
    n_occurrences: int
    conditional_return: float    # moyenne Y_ret quand actif (train)
    t_stat: float


def mine_pattern_candidates(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    binary_mask: np.ndarray | None = None,
    min_t_stat: float = DEFAULT_MIN_T_STAT,
    min_occurrences: int = DEFAULT_MIN_OCCURRENCES,
) -> list[PatternCandidate]:
    """Détecte les patterns binaires au signal conditionnel significatif.

    Args:
        X_train : (N_train, F) features du split TRAIN uniquement.
        y_train : (N_train,) target signé (Y_ret horizon courant), train.
        feature_names : noms des colonnes de X_train.
        binary_mask : optionnel, mask (F,) bool des colonnes binaires.
            Si None : détection automatique (valeurs subset de {0,1,-1}).
        min_t_stat : seuil de significativité (défaut 3.0).
        min_occurrences : occurrences minimales du pattern actif (défaut 60).

    Returns:
        Liste de candidats triés par |t| décroissant. Vide si aucun signal.
    """
    if X_train.ndim != 2 or len(feature_names) != X_train.shape[1]:
        raise ValueError("X_train et feature_names incohérents")
    n_samples = X_train.shape[0]
    if n_samples < min_occurrences * 2:
        logger.info("pattern_miner : train trop petit (%d lignes), skip", n_samples)
        return []

    y = y_train.astype(np.float64)
    baseline_std = float(np.std(y, ddof=1)) if n_samples > 1 else 0.0
    if baseline_std <= 0.0:
        return []

    candidates: list[PatternCandidate] = []
    n_tested = 0
    for j, name in enumerate(feature_names):
        col = X_train[:, j]
        # Détection binaire (subset de {0, 1, -1})
        uniq = np.unique(col)
        is_binary = bool(np.all(np.isin(uniq, [0.0, 1.0, -1.0])))
        if binary_mask is not None and not binary_mask[j]:
            continue
        if not is_binary and binary_mask is None:
            continue
        if len(uniq) < 2 or np.all(uniq == 0.0):
            continue

        for active_value in (1.0, -1.0):
            active = col > 0.5 if active_value > 0 else col < -0.5
            k = int(active.sum())
            n_tested += 1
            if k < min_occurrences:
                continue
            cond_mean = float(np.mean(y[active]))
            cond_std = float(np.std(y[active], ddof=1)) if k > 1 else baseline_std
            se = cond_std / math.sqrt(k) if k > 0 else 0.0
            if se <= 0.0:
                continue
            t_stat = cond_mean / se
            if abs(t_stat) > min_t_stat:
                candidates.append(PatternCandidate(
                    feature=name,
                    active_value=active_value,
                    n_occurrences=k,
                    conditional_return=cond_mean,
                    t_stat=float(t_stat),
                ))

    candidates.sort(key=lambda c: -abs(c.t_stat))
    logger.info(
        "pattern_miner : %d tests binaires, %d candidats retenus (|t|>%.1f, n>=%d)",
        n_tested, len(candidates), min_t_stat, min_occurrences,
    )
    return candidates


def build_einhers_from_patterns(
    candidates: list[PatternCandidate],
    asset: str,
    asset_class: str,
    timeframe: str,
    horizon_str: str,
    horizon_bars: int,
    max_candidates: int = 15,
    source_tag: str = "event_study",
) -> list[Einher]:
    """Construit des Einhers depuis les candidats event-study.

    Args:
        candidates : sortie de mine_pattern_candidates.
        asset: TODO: documenter.
        asset_class: TODO: documenter.
        horizon_bars: TODO: documenter.
        horizon_str: TODO: documenter.
        timeframe: TODO: documenter.
        asset / asset_class / timeframe / horizon_str / horizon_bars : univers.
        max_candidates : cap de candidats par triplet (les meilleurs |t| d'abord).
        source_tag : tag inscrit dans einher.source.model.
            asset: TODO: documenter.
            asset_class: TODO: documenter.
            horizon_bars: TODO: documenter.
            horizon_str: TODO: documenter.
            timeframe: TODO: documenter.

    Returns:
        Einhers prêts pour backtest/admission (métriques vides à ce stade).
    """
    empty_metrics = EinherMetrics(
        n_trades=0, n_tp=0, n_sl=0, n_timeout=0,
        win_rate=0.0, avg_net_return=0.0, total_return=0.0,
        sharpe_ratio=0.0, max_drawdown=0.0, profit_factor=0.0,
        avg_holding_bars=0.0, buy_hold_return=0.0, alpha=0.0,
    )
    einhers: list[Einher] = []
    for cand in candidates[:max_candidates]:
        direction = "BUY" if cand.conditional_return > 0 else "SELL"
        ast = Condition(
            feature_ref=cand.feature,
            operator=">" if cand.active_value > 0 else "<",
            value=0.5 if cand.active_value > 0 else -0.5,
            transformation=None,
        )
        eid = (
            f"pat_{asset}_{timeframe}_{horizon_str}_"
            f"{re.sub(r'[^A-Za-z0-9_]', '_', cand.feature)[:40]}_"
            f"{uuid.uuid4().hex[:6]}"
        )
        einhers.append(Einher(
            id=eid,
            condition_tree=ast,
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
                "feature": cand.feature,
                "active_value": cand.active_value,
                "n_train_occurrences": cand.n_occurrences,
                "conditional_return_train": cand.conditional_return,
                "t_stat_train": cand.t_stat,
                "n_conditions": 1,
                "feature_names": [cand.feature],
            },
            data_version="",
        ))
    return einhers
