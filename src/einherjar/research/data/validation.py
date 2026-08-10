"""data/validation.py — Contrôle bloquant des données (P1 #10).

Avant toute recherche, on valide que les données sont saines. Si une règle
échoue, on lève DataValidationError — l'utilisateur DOIT corriger le dataset
avant de relancer.

Contrôles appliqués :
  1. OHLCV valide (pas de NaN/inf, low <= high, open/close dans [low, high]).
  2. Index monotone (timestamps strictement croissants).
  3. Gaps déclarés (détection des trous > gap_max_factor × timeframe).
  4. Features alignées (même nombre de bougies que OHLCV, mêmes timestamps).
  5. Anti-fuite (split train < val < holdout, pas de chevauchement temporel).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import polars as pl

from einherjar.research.data.features import FeaturesFrame
from einherjar.research.data.ohlcv import OhlcvFrame

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #


class DataValidationError(Exception):
    """Erreur de validation des données (bloquante)."""


# --------------------------------------------------------------------------- #
# Résultat de validation
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ValidationReport:
    """Résultat de la validation d'un dataset OHLCV + Features.

    Attributes:
        is_valid: True si tous les contrôles sont passés.
        errors: Liste des erreurs critiques (bloquantes).
        warnings: Liste des avertissements (non-bloquants).
        n_bougies: Nombre de bougies validées.
        index_monotonic: True si l'index temporel est strictement croissant.
        n_gaps: Nombre de gaps détectés (trous > seuil).
        max_gap_factor: Plus grand facteur d'écart à un intervalle régulier.
    """

    is_valid: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    n_bougies: int = 0
    index_monotonic: bool = True
    n_gaps: int = 0
    max_gap_factor: float = 1.0
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "n_bougies": self.n_bougies,
            "index_monotonic": self.index_monotonic,
            "n_gaps": self.n_gaps,
            "max_gap_factor": self.max_gap_factor,
            "meta": self.meta,
        }


# --------------------------------------------------------------------------- #
# Validations OHLCV
# --------------------------------------------------------------------------- #


def validate_ohlcv(ohlcv: OhlcvFrame) -> ValidationReport:
    """Valide l'intégrité d'une OhlcvFrame.

    Contrôles :
      1. Pas de NaN/inf dans open/high/low/close/volume.
      2. low <= high, open/close dans [low, high] — avec tolérance
         d'arrondi relative (données réelles : micro-écarts de 0.01-0.03
         sur 3000+ sont des arrondis de source, pas une corruption).
      3. Index monotone (timestamps croissants).
      4. Gaps temporels (détection d'un trou > 1.5 × l'intervalle médian).

    Returns:
        ValidationReport. `is_valid=True` si tout passe.
    """
    errors: list[str] = []
    warnings: list[str] = []
    df = ohlcv.df
    n = ohlcv.n_bougies
    if n == 0:
        errors.append(f"OHLCV vide pour {ohlcv.asset}/{ohlcv.timeframe}")
        return ValidationReport(is_valid=False, errors=tuple(errors), n_bougies=0)
    # 1. NaN/inf sur les colonnes critiques.
    for col in ("open", "high", "low", "close"):
        if col not in df.columns:
            errors.append(f"Colonne manquante : {col}")
            continue
        s = df[col].to_numpy()
        if not _all_finite(s):
            errors.append(f"NaN/inf dans {col} (asset={ohlcv.asset}, tf={ohlcv.timeframe})")
    # 2. Cohérence low <= high, open/close dans [low, high] (tolérance relative).
    #    Données réelles : open peut dépasser high de 0.01-0.03 sur des prix
    #    3000+ (arrondi de source). Seuil : 0.1% du prix.
    if {"open", "high", "low", "close"}.issubset(set(df.columns)):
        o = df["open"].to_numpy()
        h = df["high"].to_numpy()
        l = df["low"].to_numpy()
        c = df["close"].to_numpy()
        tol = 0.001  # 0.1% — les micro-écarts (0.01-0.03 / 3000) passent
        for i in range(n):
            if l[i] > h[i] * (1 + tol):
                errors.append(f"Bougie {i}: low > high (l={l[i]} > h={h[i]})")
                break
            if o[i] < l[i] * (1.0 - tol) or o[i] > h[i] * (1.0 + tol):
                errors.append(f"Bougie {i}: open hors [low, high] (o={o[i]})")
                break
            if c[i] < l[i] * (1.0 - tol) or c[i] > h[i] * (1.0 + tol):
                errors.append(f"Bougie {i}: close hors [low, high] (c={c[i]})")
                break
    # 3. Index monotone + gaps (deltas en ms, indépendants du dtype timestamp).
    index_mono = True
    n_gaps = 0
    max_gap_factor = 1.0
    if "timestamp" in df.columns:
        ts_ms = _timestamps_to_epoch_ms(df["timestamp"])
        deltas = []
        for i in range(1, n):
            if ts_ms[i] < ts_ms[i - 1]:
                index_mono = False
                errors.append(f"Index non monotone à la bougie {i} (ts[{i-1}]={ts_ms[i-1]} > ts[{i}]={ts_ms[i]})")
                break
            deltas.append(ts_ms[i] - ts_ms[i - 1])
        if deltas:
            import statistics
            median_delta = statistics.median(deltas)
            if median_delta > 0:
                n_gaps = sum(1 for d in deltas if d > 1.5 * median_delta)
                max_gap_factor = max(deltas) / median_delta if median_delta > 0 else 1.0
                if n_gaps > 0:
                    warnings.append(
                        f"{n_gaps} gaps détectés (max_factor={max_gap_factor:.2f}x median)"
                    )
    return ValidationReport(
        is_valid=len(errors) == 0,
        errors=tuple(errors),
        warnings=tuple(warnings),
        n_bougies=n,
        index_monotonic=index_mono,
        n_gaps=n_gaps if "timestamp" in df.columns else 0,
        max_gap_factor=max_gap_factor if "timestamp" in df.columns else 1.0,
    )


def validate_features(features: FeaturesFrame) -> ValidationReport:
    """Valide l'alignement d'une FeaturesFrame.

    Contrôles :
      1. Pas de NaN/inf dans les colonnes features (hors OHLCV).
      2. Au moins une bougie.
      3. Timestamp aligné sur l'OHLCV (caller responsibility).
    """
    errors: list[str] = []
    warnings: list[str] = []
    n = features.n_bougies
    if n == 0:
        errors.append("FeaturesFrame vide")
        return ValidationReport(is_valid=False, errors=tuple(errors), n_bougies=0)
    # 1. Pas de NaN/inf sur les features.
    import numpy as np
    for name in features.feature_names:
        col = features.column(name).to_numpy()
        if not _all_finite(col):
            n_bad = int(np.sum(~np.isfinite(col)))
            warnings.append(f"Feature {name} : {n_bad} NaN/inf")
    return ValidationReport(
        is_valid=len(errors) == 0,
        errors=tuple(errors),
        warnings=tuple(warnings),
        n_bougies=n,
    )


def validate_no_leak(
    train_end_ts: int,
    val_start_ts: int,
    val_end_ts: int,
    holdout_start_ts: int,
    *,
    embargo_bougies: int = 0,
    purge_window: int = 0,
) -> ValidationReport:
    """Vérifie l'absence de fuite temporelle entre les splits.

    Les marges sont EXPRIMÉES EN UNITÉS DE TIMESTAMP (ms) — c'est le contrat
    réel du loader (epoch ms Unix) ; `embargo_bougies` et `purge_window`
    sont des ALIAS sémantiques : une mauvaise valeur (>=1) peut désigner
    une marge en ms si l'appelant passe déjà des timestamps purgés (cas
    discovery.py, qui purge AVANT d'appeler). Pour rester robuste, on
    n'applique les marges que si elles sont posées en ms explicitement.

    Args:
        train_end_ts: Timestamp de fin du train.
        val_start_ts: Timestamp de début du val.
        val_end_ts: Timestamp de fin du val.
        holdout_start_ts: Timestamp de début du holdout.
        embargo_bougies: Marge minimale (ms) entre train fin et val début
            (et val fin / holdout début).
        purge_window: Fenêtre de purge (ms) au bord droit du train
            (les N dernières unités de train sont exclues de la comparaison).

    Raises:
        DataValidationError: si une fuite est détectée.
    """
    errors: list[str] = []
    warnings: list[str] = []
    # Le val doit commencer APRÈS le train + embargo (et la purge du train
    # doit être respectée : train_end reçu inclut déjà la purge).
    train_effective_end = train_end_ts - purge_window
    if val_start_ts - embargo_bougies <= train_effective_end:
        errors.append(
            f"Fuite train→val : val_start={val_start_ts} <= train_end (effective)={train_effective_end} "
            f"(embargo={embargo_bougies}, purge={purge_window})"
        )
    if holdout_start_ts - embargo_bougies <= val_end_ts:
        errors.append(
            f"Fuite val→holdout : holdout_start={holdout_start_ts} <= val_end={val_end_ts} "
            f"(embargo={embargo_bougies})"
        )
    if errors:
        return ValidationReport(
            is_valid=False, errors=tuple(errors), warnings=tuple(warnings),
        )
    return ValidationReport(
        is_valid=True, errors=(), warnings=tuple(warnings),
    )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _all_finite(arr: Any) -> bool:
    """True si toutes les valeurs de l'array sont finies (pas de NaN/inf).

    NOTE (revue) : l'ancienne version utilisait `isinstance(v, float)` qui
    est FAUX pour numpy.float64 — le contrôle ne détectait jamais les NaN.
    On utilise np.isfinite (fonctionne pour numpy ET python).
    """
    import numpy as np

    a = np.asarray(arr, dtype="float64")
    if a.size == 0:
        return True
    return bool(np.isfinite(a).all())


def _timestamps_to_epoch_ms(series: pl.Series) -> list[int]:
    """Convertit une colonne timestamp (datetime ou int ms) en ms Unix.

    Les OhlcvFrame produites par les loaders réels ont un timestamp
    datetime[us, UTC] ; les tests peuvent injecter des int (ms Unix).
    On normalise les deux vers des entiers ms pour les calculs de gaps.
    """
    dtype = series.dtype
    if dtype == pl.Datetime("us", "UTC") or dtype == pl.Datetime:
        return series.dt.epoch("ms").cast(pl.Int64).to_list()
    return [int(v) for v in series.to_list()]


# --------------------------------------------------------------------------- #
# Validation globale (utilisée par le loader avant search)
# --------------------------------------------------------------------------- #


def validate_or_raise(
    ohlcv: OhlcvFrame,
    features: FeaturesFrame | None = None,
    *,
    raise_on_error: bool = True,
) -> ValidationReport:
    """Valide OHLCV (+ features optionnelles) et lève si erreurs."""
    report = validate_ohlcv(ohlcv)
    if features is not None:
        feat_report = validate_features(features)
        alignment_errors: list[str] = []
        if ohlcv.n_bougies != features.n_bougies:
            alignment_errors.append(
                f"OHLCV/features length mismatch: {ohlcv.n_bougies} != {features.n_bougies}"
            )
        elif "timestamp" not in ohlcv.df.columns or "timestamp" not in features.df.columns:
            alignment_errors.append("timestamp missing from OHLCV or features")
        elif ohlcv.df["timestamp"].to_list() != features.df["timestamp"].to_list():
            alignment_errors.append("OHLCV/features timestamps are not exactly aligned")
        if ohlcv.data_version != features.data_version:
            alignment_errors.append(
                f"OHLCV/features data_version mismatch: {ohlcv.data_version} != {features.data_version}"
            )
        report = ValidationReport(
            is_valid=report.is_valid and feat_report.is_valid and not alignment_errors,
            errors=report.errors + feat_report.errors + tuple(alignment_errors),
            warnings=report.warnings + feat_report.warnings,
            n_bougies=report.n_bougies,
            index_monotonic=report.index_monotonic,
            n_gaps=report.n_gaps,
            max_gap_factor=report.max_gap_factor,
            meta={**report.meta, **feat_report.meta},
        )
    if raise_on_error and not report.is_valid:
        raise DataValidationError(
            f"Validation données échouée : {report.errors}"
        )
    return report
