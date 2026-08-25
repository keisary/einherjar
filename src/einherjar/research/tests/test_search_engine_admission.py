"""Tests de la validation C1-C6 (bootstrap, DSR, FDR, dédup, corpus)."""
from __future__ import annotations

import dataclasses

import numpy as np

from einherjar.research.search_engine.admission import (
    Candidate,
    admit_batch,
    benjamini_hochberg,
)
from einherjar.research.search_engine.bootstrap import block_bootstrap_ci
from einherjar.research.search_engine.builder import build_einher
from einherjar.research.search_engine.corpus import append_einher, fingerprint_of, load_corpus
from einherjar.research.search_engine.dsr import dsr_probability
from einherjar.research.search_engine.expression import Cmp, Feature
from einherjar.research.xgb_einhers.types import EinherMetrics


def _metrics(
    *,
    n_trades: int = 60,
    sharpe: float = 1.5,
    p_value: float = 0.01,
    seed: int = 0,
) -> EinherMetrics:
    rng = np.random.default_rng(seed)
    rets = rng.normal(loc=0.004, scale=0.02, size=n_trades)
    return EinherMetrics(
        n_trades=n_trades, n_tp=int(n_trades * 0.5), n_sl=int(n_trades * 0.3),
        n_timeout=n_trades - int(n_trades * 0.8), win_rate=0.5,
        avg_net_return=float(rets.mean()), total_return=float(rets.sum()),
        sharpe_ratio=sharpe, max_drawdown=0.3, profit_factor=1.4,
        avg_holding_bars=24.0, buy_hold_return=0.1, alpha=0.2,
        t_statistic=2.5, p_value=p_value, trade_returns=tuple(rets.tolist()),
    )


def _candidate(
    i: int,
    *,
    metrics: EinherMetrics | None = None,
    mask: np.ndarray | None = None,
    features: set[str] | None = None,
) -> Candidate:
    einher = build_einher(
        Cmp(expr=Feature(f"f{i}"), operator=">", value=0.0),
        "BUY", 24, {"asset": "T", "asset_class": "crypto", "timeframe": "1h", "horizon": "1d", "horizon_bars": 24},
        costs_pct=0.0014,
    )
    if metrics is not None:
        einher = dataclasses.replace(einher, metrics=metrics)
    return Candidate(
        einher=einher,
        val_mask=(mask if mask is not None else np.arange(500) % 3 == 0)[:500],
        features=features or {f"f{i}", f"f{i + 1}"},
        fingerprint=f"fp_{i}",
    )


class TestBH:
    def test_exact_small_case(self) -> None:
        p = [0.001, 0.01, 0.02, 0.5]
        assert benjamini_hochberg(p, alpha=0.05) == {0, 1, 2}

    def test_no_significant(self) -> None:
        assert benjamini_hochberg([0.4, 0.3, 0.2, 0.1], alpha=0.05) == set()


class TestBootstrap:
    def test_ci_contains_mean(self) -> None:
        rng = np.random.default_rng(3)
        rets = rng.normal(loc=0.005, scale=0.02, size=200)
        lo, hi, mean = block_bootstrap_ci(rets, n_boot=300, seed=1)
        assert lo <= mean <= hi
        assert lo > 0  # le signal est assez fort

    def test_insufficient_trades(self) -> None:
        lo, hi, mean = block_bootstrap_ci([0.01] * 10, n_boot=10)
        assert np.isnan(lo)


class TestDSR:
    def test_strong_signal_high_prob(self) -> None:
        rng = np.random.default_rng(4)
        rets = rng.normal(loc=0.01, scale=0.02, size=200)
        p = dsr_probability(rets)
        assert p > 0.95

    def test_degenerate_zero(self) -> None:
        assert dsr_probability([0.01] * 50) == 0.0

    def test_noise_low_prob(self) -> None:
        rng = np.random.default_rng(5)
        rets = rng.normal(loc=0.0, scale=0.02, size=200)
        p = dsr_probability(rets)
        assert p < 0.7


class TestAdmission:
    def test_strong_admitted_weak_rejected(self) -> None:
        strong = _candidate(0, metrics=_metrics(n_trades=80, sharpe=2.0, p_value=0.001))
        weak = _candidate(1, metrics=_metrics(n_trades=10, sharpe=0.1, p_value=0.9))
        outs = admit_batch([strong, weak], seed=7)
        assert outs[0].admitted is True
        assert outs[1].admitted is False
        assert outs[1].reasons["min_trades"] is False

    def test_duplicate_rejected(self) -> None:
        m1 = _metrics(n_trades=80, sharpe=2.0, p_value=0.001)
        m2 = _metrics(n_trades=80, sharpe=1.9, p_value=0.002)  # presque identique
        a = _candidate(0, metrics=m1, features={"f0", "f1"})
        b = _candidate(1, metrics=m2, features={"f0", "f1"})
        outs = admit_batch([a, b], seed=7)
        assert outs[0].admitted is True
        assert outs[1].admitted is False  # doublon
        assert outs[1].reasons["dedup"]["pass"] is False

    def test_fdr_rejects_many_noise(self) -> None:
        """Beaucoup de bruit → très peu de signal passe le FDR."""
        cands = []
        for i in range(40):
            p = 0.5 if i % 5 else 0.004
            m = _metrics(n_trades=60, sharpe=1.0 if p < 0.01 else 0.0, p_value=p)
            cands.append(_candidate(i, metrics=m, features={f"g{i}"}))
        outs = admit_batch(cands, seed=3)
        assert sum(o.admitted for o in outs) <= 8  # seul ~1/5 du bruit passe


class TestCorpus:
    def test_append_and_load_and_fingerprint(self, tmp_path) -> None:
        from einherjar.research.search_engine.admission import AdmissionOutcome

        p = tmp_path / "corpus.jsonl"
        c = _candidate(0, metrics=_metrics(n_trades=70, sharpe=2.0, p_value=0.001))
        fp = fingerprint_of(c.einher.condition_tree)
        entry = append_einher(c.einher, AdmissionOutcome(admitted=True, reasons={"ok": True}),
                              fingerprint=fp, path=p)
        assert entry["fingerprint"] == fp
        rows = load_corpus(p)
        assert len(rows) == 1
        assert rows[0]["admission"]["admitted"] is True
        # idempotence de l'empreinte
        assert fingerprint_of(c.einher.condition_tree) == fp

def _mk_strong(sharpe: float, n: int = 120, features=None, mask_slice=None) -> Candidate:
    """Candidate admissible : rendements positifs bruités."""
    rng = np.random.default_rng(3)
    rets = rng.normal(0.01, 0.03, n)  # t-stat élevé -> C6/DSR passent en test
    m = EinherMetrics(
        n_trades=n, n_tp=int(n*0.6), n_sl=int(n*0.4), n_timeout=0,
        win_rate=0.6, avg_net_return=float(rets.mean()), total_return=float(rets.sum()),
        sharpe_ratio=sharpe, max_drawdown=-0.1, profit_factor=1.5,
        avg_holding_bars=10, buy_hold_return=0.2, alpha=0.1,
        t_statistic=4.0, p_value=0.0001, trade_returns=list(rets),
    )
    e = build_einher(Cmp(expr=Feature("mom"), operator=">", value=0.0), "BUY", 10,
                     {"asset": "BTCUSD"}, costs_pct=0.0014)
    e = dataclasses.replace(e, metrics=m)
    mask = np.zeros(500, dtype=bool)
    mask[slice(0, 200) if mask_slice is None else mask_slice] = True
    return Candidate(
        einher=e, val_mask=mask,
        features={"mom", "vol"} if features is None else features,
        fingerprint=f"fp_{sharpe}",
    )


def test_dedup_cross_batch_rejects_duplicate() -> None:
    """Un doublon d'un admis HISTORIQUE (autre batch/seed) est rejeté."""
    a = _mk_strong(2.0)
    b = _mk_strong(1.9)  # quasi-identique (mêmes features, même masque)
    outcomes = admit_batch([b], initial_accepted=[a], dup_jaccard=0.30, dup_corr=0.50)
    assert not outcomes[0].admitted
    assert outcomes[0].reasons["dedup"]["duplicate_of"] == "fp_2.0"


def test_dedup_cross_batch_allows_distinct() -> None:
    """Un candidat nouveau (features disjointes, masque décorrélé) passe malgré.

    un historique non vide.
    """
    # features disjointes + masque décorrélé vs l'historique
    b = _mk_strong(2.5, features={"rsi", "kurt"}, mask_slice=slice(200, 500))
    outcomes = admit_batch([b], initial_accepted=[_mk_strong(2.0)], dup_jaccard=0.30, dup_corr=0.50)
    assert outcomes[0].admitted
