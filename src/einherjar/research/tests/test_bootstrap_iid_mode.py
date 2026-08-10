"""Mode iid du bootstrap CI : plus serre que le block sur rets a drift sequentiel, reproductible."""
import random

from einherjar.research.engine.bootstrap import iid_bootstrap_ci
from einherjar.research.utils.stats import block_bootstrap_ci


def _mean(v):
    return sum(v) / len(v)


def test_iid_ci_reproductible():
    random.seed(0)
    rets = [random.gauss(0.1, 1.0) for _ in range(300)]
    a = iid_bootstrap_ci(rets, _mean, n_resamples=2000, ci_level=0.95, rng_seed=7)
    b = iid_bootstrap_ci(rets, _mean, n_resamples=2000, ci_level=0.95, rng_seed=7)
    assert a == b  # meme graine -> CI identiques


def test_iid_ci_contient_observed():
    random.seed(1)
    rets = [random.gauss(0.1, 1.0) for _ in range(300)]
    ci_low, ci_high, obs = iid_bootstrap_ci(rets, _mean, n_resamples=2000, ci_level=0.95, rng_seed=7)
    assert ci_low <= obs <= ci_high
    assert ci_low > -0.2  # signaux souvent > 0 avec ce niveau de bruit


def test_iid_plus_serre_que_block_sur_drift_sequentiel():
    # drift sequentiel : le signal porte sur l'ORDRE des rets (momentum-like)
    rets = [0.4] * 150 + [-0.02] * 150
    obs_mean = _mean(rets)
    assert obs_mean > 0.0
    block_low, _, _ = block_bootstrap_ci(
        rets, _mean, n_resamples=2000, block_length=15, ci_level=0.95, rng_seed=7,
    )
    iid_low, _, _ = iid_bootstrap_ci(rets, _mean, n_resamples=2000, ci_level=0.95, rng_seed=7)
    # le block (rearrange l'ordre) detruit le signal -> borne basse plus basse
    assert iid_low > block_low


def test_iid_vide_nan():
    assert iid_bootstrap_ci([], _mean)[0] != iid_bootstrap_ci([], _mean)[0]  # NaN