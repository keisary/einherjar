"""Tests legers pour smart_sampling (dollar bars + volume filter) et feature_reduction (IC).

Ne teste PAS le pipeline complet (lourd) — seulement les primitives, afin de
verouiller le comportement avant/après branchement dans runner.py.
"""
import numpy as np
import pytest

from einherjar.research.xgb_einhers.smart_sampling import (
    combine_indices,
    dollar_bars_indices,
    filter_volume_indices,
)
from einherjar.research.xgb_einhers.feature_reduction import select_features_by_ic


# --------------------------------------------------------------------------- #
# dollar_bars_indices
# --------------------------------------------------------------------------- #

def test_dollar_bars_small_input_identity():
    """n < 100 : retourne tous les indices (pas de resampling)."""
    n = 50
    prices = np.ones(n)
    volumes = np.ones(n)
    ts = np.arange(n)
    out = dollar_bars_indices(ts, prices, volumes)
    assert np.array_equal(out, np.arange(n))


def test_dollar_bars_reduces_and_is_sorted():
    """Sur un gros input, dollar bars <= input, ordonnance croissante."""
    rng = np.random.default_rng(0)
    n = 20_000
    prices = 100 + rng.random(n) * 10
    volumes = rng.random(n) * 100 + 1
    ts = np.arange(n)
    out = dollar_bars_indices(ts, prices, volumes, threshold_mult=20.0)
    assert len(out) < n
    assert np.all(np.diff(out) > 0)  # strictement croissant
    # chaque dollar bar = fin de barre : dernier index inclus au max
    assert out[-1] < n


def test_dollar_bars_respects_threshold():
    """Le cumul de $ entre deux barres de sortie >= seuil (hors derniere)."""
    rng = np.random.default_rng(1)
    n = 5_000
    prices = 50 + rng.random(n)
    volumes = 10 + rng.random(n) * 10
    dollar_vol = prices * volumes
    threshold = 500.0
    out = dollar_bars_indices(np.arange(n), prices, volumes, dollar_threshold=threshold)
    # cumul entre barres consecutives
    prev = -1
    for idx in out:
        seg = dollar_vol[prev + 1 : idx + 1].sum()
        # la derniere barre peut etre < seuil (residu)
        if idx != out[-1]:
            assert seg >= threshold
        prev = idx


def test_dollar_bars_too_high_threshold_single_bar():
    """Seuil gigantesque -> au moins la derniere barre."""
    n = 1000
    prices = np.ones(n)
    volumes = np.ones(n)
    out = dollar_bars_indices(np.arange(n), prices, volumes, dollar_threshold=1e12)
    assert len(out) >= 1
    assert out[-1] == n - 1


# --------------------------------------------------------------------------- #
# filter_volume_indices
# --------------------------------------------------------------------------- #

def test_filter_volume_small_input_identity():
    n = 40
    vols = np.ones(n)
    out = filter_volume_indices(vols)
    assert np.array_equal(out, np.arange(n))


def test_filter_volume_selects_high_volume():
    """Les barres a volume 10x la moyenne doivent etre conservees, les barres
    a volume ~0 rejetees."""
    rng = np.random.default_rng(2)
    n = 10_000
    vols = rng.random(n) + 0.1  # baseline ~0.6
    # injecter un gros spike
    vols[5000] = 100.0
    out = filter_volume_indices(vols, volume_mult=1.5, window=20)
    assert 5000 in out
    # une barre de volume quasi nul ne doit pas etre gardee (hors bord gauche)
    assert 0 not in out


# --------------------------------------------------------------------------- #
# combine_indices
# --------------------------------------------------------------------------- #

def test_combine_indices_intersection():
    a = np.array([1, 2, 5, 9])
    b = np.array([2, 9, 11])
    assert np.array_equal(combine_indices(a, b), np.array([2, 9]))


def test_combine_indices_empty():
    assert combine_indices(np.array([1]), np.array([], dtype=np.int64)).size == 0


# --------------------------------------------------------------------------- #
# feature_reduction
# --------------------------------------------------------------------------- #

def test_select_features_by_ic_reduces():
    """Sur des features correlees + bruit, la reduction doit capturer le signal
    et retourner <= max_features features."""
    rng = np.random.default_rng(3)
    n = 2000
    f = 60
    X = rng.standard_normal((n, f))
    # feature 0 et 1 portent le signal (correlees entre elles)
    signal = np.linspace(-1, 1, n)
    X[:, 0] = signal + 0.1 * rng.standard_normal(n)
    X[:, 1] = signal + 0.1 * rng.standard_normal(n)
    y = signal + 0.2 * rng.standard_normal(n)
    names = [f"f{j}" for j in range(f)]
    kept, idx = select_features_by_ic(X, y, names, max_features=40)
    assert len(kept) == len(idx)
    assert len(kept) < f  # a reduit
    assert len(kept) <= 40
    # la feature signal doit etre dans les conservees
    assert "f0" in kept


def test_select_features_small_n_identity():
    """n < 50 : retourne toutes les features sans reduction."""
    X = np.random.default_rng(4).standard_normal((30, 10))
    y = np.random.default_rng(5).standard_normal(30)
    names = [f"g{j}" for j in range(10)]
    kept, idx = select_features_by_ic(X, y, names)
    assert len(kept) == 10
    assert np.array_equal(idx, np.arange(10))