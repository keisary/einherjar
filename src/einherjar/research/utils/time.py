"""
utils/time.py — Découpage des séries temporelles en splits train/val/holdout.

Implémente S-3.1 (ONTOLOGY.md) :
  - split temporel strict (pas de shuffle, pas de shuffle avant)
  - purging : exclusion des bougies dont le label d'amplitude déborde
  - embargo : exclusion de N bougies supplémentaires après chaque frontière

Toutes les fonctions opèrent sur des indices entiers (0..N-1) plutôt que
des timestamps, pour rester agnostiques du calendrier exact. Le mapping
timestamp ↔ indice est délégué à `data/ohlcv.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Split:
    """Bornes d'un jeu temporel, en indices inclusifs/exclusifs."""

    name: str                   # 'train' | 'val' | 'holdout'
    start: int                  # index de la première bougie du jeu
    end: int                    # index de la dernière bougie du jeu + 1 (semi-ouvert)
    purge_start: int = 0        # début effectif après purging
    purge_end: int = 0          # fin effective après purging
    embargo_applied: int = 0    # nb bougies d'embargo appliquées à ce split

    @property
    def length(self) -> int:
        """Longueur effective (après purging) en bougies."""
        return max(0, self.purge_end - self.purge_start)


@dataclass(frozen=True)
class SplitBundle:
    """Ensemble des 3 splits + paramètres de génération."""

    train: Split
    val: Split
    holdout: Split
    n_total: int
    train_ratio: float
    val_ratio: float
    holdout_ratio: float
    embargo_bougies: int
    horizon_label: int          # N bougies purgées en bordure de label

    def is_split_indices(self) -> tuple[set[int], set[int], set[int]]:
        """Retourne (train_idx, val_idx, holdout_idx) après purging/embargo."""
        return (
            set(range(self.train.purge_start, self.train.purge_end)),
            set(range(self.val.purge_start, self.val.purge_end)),
            set(range(self.holdout.purge_start, self.holdout.purge_end)),
        )


def make_splits_ratio(
    n_total: int,
    train_ratio: float = 0.60,
    val_ratio: float = 0.20,
    holdout_ratio: float = 0.20,
    horizon_label: int = 1,
    embargo_bougies: int = 1,
) -> SplitBundle:
    """Construit les splits par ratios, avec purging/embargo.

    Le purging exclut `horizon_label` bougies à chaque frontière de split
    côté jeu suivant (pour éviter qu'un trade ouvert en fin de train ne
    soit résolu avec une bougie du val). L'embargo ajoute `embargo_bougies`
    bougies de marge supplémentaire.

    Convention semi-ouverte : [start, end).
    """
    if not (0.0 < train_ratio < 1.0):
        raise ValueError(f"train_ratio hors borne: {train_ratio}")
    if not (0.0 < val_ratio < 1.0):
        raise ValueError(f"val_ratio hors borne: {val_ratio}")
    if not (0.0 < holdout_ratio < 1.0):
        raise ValueError(f"holdout_ratio hors borne: {holdout_ratio}")
    s = train_ratio + val_ratio + holdout_ratio
    if abs(s - 1.0) > 1e-9:
        raise ValueError(f"Les ratios doivent sommer à 1.0, got {s}")

    if n_total <= 0:
        raise ValueError(f"n_total doit être > 0, got {n_total}")

    t1 = int(n_total * train_ratio)
    t2 = int(n_total * (train_ratio + val_ratio))
    # t2 = n_total, on n'a pas besoin de t3

    purge = max(0, horizon_label)
    emb = max(0, embargo_bougies)

    # Train : [0, t1) — purge du bord droit.
    # (fix fuite train→val) Un calibrage basé sur la bougie t1-1 utilise des
    # trades dont le label déborde dans le val (fenêtre [t1, t1+horizon)) :
    # le label « vu » par le train est informé par des bougies du val, ce qui
    # est une fuite. On purge `purge` bougies à droite du train exactement
    # comme on purge la gauche du val. L'embargo, lui, ne s'applique qu'au
    # début du jeu suivant (val), pas à la fin du train.
    train = Split(
        name="train",
        start=0,
        end=t1,
        purge_start=0,
        purge_end=t1 - purge,
        embargo_applied=0,
    )

    # Val : [t1, t2) — purge de `horizon_label` bougies à gauche, embargo en plus
    val_purge_start = t1 + purge + emb
    val = Split(
        name="val",
        start=t1,
        end=t2,
        purge_start=val_purge_start,
        purge_end=t2,
        embargo_applied=emb,
    )

    # Holdout : [t2, n_total) — purge + embargo
    holdout_purge_start = t2 + purge + emb
    holdout = Split(
        name="holdout",
        start=t2,
        end=n_total,
        purge_start=holdout_purge_start,
        purge_end=n_total,
        embargo_applied=emb,
    )

    return SplitBundle(
        train=train,
        val=val,
        holdout=holdout,
        n_total=n_total,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        holdout_ratio=holdout_ratio,
        embargo_bougies=emb,
        horizon_label=purge,
    )


def horizon_for_amplitude(
    amplitude_value: float,
    atr_p50: float,
    min_n: int = 3,
    max_n: int = 50,
) -> int:
    """Calcule la fenêtre d'observation N pour une amplitude (S-2.1).

    N = clamp(ceil(amplitude / atr_p50), min_n, max_n)

    Note : cette fonction suppose amplitude.unité == prix_absolu.
    Pour multiple_ATR, voir `engine/evaluator.py`.
    """
    if atr_p50 <= 0:
        raise ValueError(f"atr_p50 doit être > 0, got {atr_p50}")
    if amplitude_value <= 0:
        raise ValueError(f"amplitude_value doit être > 0, got {amplitude_value}")
    import math
    n = math.ceil(amplitude_value / atr_p50)
    return max(min_n, min(max_n, n))
