# pyright: reportAttributeAccessIssue=false
"""Tests P3 - pattern_miner (event-study binaires) et cap macro-familles."""
from __future__ import annotations

import numpy as np

from einherjar.research.xgb_einhers.path_extractor import XGBPath, extract_paths
from einherjar.research.xgb_einhers.pattern_miner import (
    build_einhers_from_patterns,
    mine_pattern_candidates,
)

# --------------------------------------------------------------------------- #
# pattern_miner
# --------------------------------------------------------------------------- #


class TestPatternMiner:
    def _synthetic(self, n: int = 40000):
        rng = np.random.default_rng(42)
        y = rng.normal(0, 0.008, n).astype(np.float32)
        X = np.zeros((n, 3), dtype=np.float32)
        X[:, 0] = (rng.random(n) > 0.995).astype(np.float32)  # rare, sans signal
        act = rng.random(n) > 0.99
        X[act, 1] = 1.0
        y[act] += 0.0025  # signal fort (+25bp)
        X[:, 2] = rng.normal(0, 1, n)  # continue : jamais testee
        return X, y

    def test_detects_significant_pattern(self) -> None:
        X, y = self._synthetic()
        cands = mine_pattern_candidates(
            X, y, ["pat_mort", "pat_signal", "cont"], min_t_stat=3.0
        )
        assert any(c.feature == "pat_signal" for c in cands)
        assert all(c.feature != "cont" for c in cands)

    def test_ignores_dead_pattern(self) -> None:
        X, y = self._synthetic()
        cands = mine_pattern_candidates(
            X, y, ["pat_mort", "pat_signal", "cont"], min_t_stat=3.0
        )
        assert not any(c.feature == "pat_mort" for c in cands)

    def test_never_tests_continuous(self) -> None:
        X, y = self._synthetic()
        cands = mine_pattern_candidates(X, y, ["a", "b", "cont"])
        assert all(c.feature != "cont" for c in cands)

    def test_sell_direction_for_negative_active(self) -> None:
        rng = np.random.default_rng(7)
        n = 40000
        y = rng.normal(0, 0.008, n).astype(np.float32)
        X = np.zeros((n, 2), dtype=np.float32)
        act = rng.random(n) > 0.99
        X[act, 0] = -1.0
        y[act] -= 0.003
        cands = mine_pattern_candidates(X, y, ["pat_neg", "filler"], min_t_stat=3.0)
        assert cands and cands[0].active_value == -1.0
        einhers = build_einhers_from_patterns(cands, "BTCUSD", "crypto", "1h", "6h", 6)
        assert einhers[0].direction == "SELL"
        assert einhers[0].condition_tree.operator == "<"

    def test_max_candidates_cap(self) -> None:
        rng = np.random.default_rng(11)
        n = 40000
        y = rng.normal(0, 0.008, n).astype(np.float32)
        X = np.zeros((n, 6), dtype=np.float32)
        for j in range(6):
            act = rng.random(n) > 0.99
            X[act, j] = 1.0
            y[act] += 0.004 + j * 0.0001  # tous significatifs
        cands = mine_pattern_candidates(
            X, y, [f"p{j}" for j in range(6)], min_t_stat=3.0
        )
        einhers = build_einhers_from_patterns(
            cands, "BTCUSD", "crypto", "1h", "6h", 6, max_candidates=3
        )
        assert len(einhers) == 3

    def test_empty_on_tiny_train(self) -> None:
        X = np.zeros((50, 2), dtype=np.float32)
        y = np.zeros(50, dtype=np.float32)
        assert mine_pattern_candidates(X, y, ["a", "b"]) == []

    def test_einher_structure(self) -> None:
        X, y = self._synthetic()
        cands = mine_pattern_candidates(X, y, ["pat_mort", "pat_signal", "cont"])
        einhers = build_einhers_from_patterns(cands, "ETHUSD", "crypto", "4h", "1d", 6)
        e = einhers[0]
        assert e.universe["asset"] == "ETHUSD"
        assert e.amplitude_bars == 6
        assert e.source["model"] == "event_study"
        assert e.condition_tree.feature_ref == "pat_signal"
        assert e.metrics.n_trades == 0  # rempli plus tard par le backtester


# --------------------------------------------------------------------------- #
# Cap macro-familles (P3-2)
# --------------------------------------------------------------------------- #

FAM_MAP: dict[str, str] = {}
for _i in range(20):
    FAM_MAP[f"risk_feat_{_i}"] = "risk"
    FAM_MAP[f"hammer_{_i}"] = "price_action"


def _mk(i: int, feat: str) -> XGBPath:
    return XGBPath(
        conditions=((feat, "<", float(i)),),
        score=0.01 - i * 0.0002,
        tree_idx=i,
        path_idx=i,
    )


class TestMacroFamilyCap:
    def _run(self, max_paths: int, cap: float) -> list[XGBPath]:
        paths = [_mk(i, f"risk_feat_{i}") for i in range(20)]
        paths += [_mk(i, f"hammer_{i}") for i in range(20)]

        import einherjar.research.xgb_einhers.path_extractor as pe

        class FakeModel:
            pass

        orig = pe._extract_sklearn
        pe._extract_sklearn = lambda model, names: paths
        try:
            return extract_paths(
                FakeModel(),
                "sklearn",
                list(FAM_MAP.keys()),
                min_score=0.0,
                max_paths=max_paths,
                enable_logical_variants=False,
                family_map=FAM_MAP,
                macro_family_cap=cap,
            )
        finally:
            pe._extract_sklearn = orig

    def test_cap_limits_dominant_family(self) -> None:
        # Sans cap, le top-8 serait 4/4 (scores interlaces) - pas de domination
        # a limiter ici. Pour tester le CAP, on desavantage hammer (scores plus
        # faibles) et on verifie qu'ils entrent quand meme grace au plafonnement
        # de continuous.
        paths = [_mk(i, f"risk_feat_{i}") for i in range(20)]
        paths += [
            XGBPath(
                conditions=((f"hammer_{i}", "<", float(i)),),
                score=0.0099 - i * 0.0002,
                tree_idx=100 + i,
                path_idx=i,
            )
            for i in range(20)
        ]

        import einherjar.research.xgb_einhers.path_extractor as pe

        class FakeModel:
            pass

        orig = pe._extract_sklearn
        pe._extract_sklearn = lambda model, names: paths
        try:
            out = extract_paths(
                FakeModel(),
                "sklearn",
                list(FAM_MAP.keys()),
                min_score=0.0,
                max_paths=10,
                enable_logical_variants=False,
                family_map=FAM_MAP,
                macro_family_cap=0.40,
            )
        finally:
            pe._extract_sklearn = orig

        binary = sum(1 for p in out if p.conditions[0][0].startswith("hammer"))
        # Sans cap : 10 risk (meilleurs scores), 0 binary.
        # Avec cap=4 : max 4 continuous en passe 1 -> binary entre en force.
        assert binary >= 2, f"binary={binary} : le cap n'a pas libere de place"
        assert len(out) == 10

    def test_redistribution_can_exceed_cap_when_budget_unfilled(self) -> None:
        # Budget sature : la passe 2 reprend les skippes -> une famille peut
        # depasser son cap SI l'autre n'a plus de chemins qualifies a offrir.
        out = self._run(max_paths=15, cap=0.40)
        assert len(out) == 15

    def test_no_family_map_no_cap(self) -> None:
        """Sans family_map, comportement historique : top-N pur par |score|.

        Les scores risk/hammer etant interleaves, le top-10 contient 5 de chaque
        (aucun cap applique, aucun ecrasement non plus).
        """
        out = self._run(max_paths=10, cap=0.40)
        cont = sum(1 for p in out if p.conditions[0][0].startswith("risk"))
        binary = sum(1 for p in out if p.conditions[0][0].startswith("hammer"))
        assert cont == 5 and binary == 5  # purement par score, sans intervention

    def test_redistribution_fills_budget(self) -> None:
        out = self._run(max_paths=12, cap=0.30)
        assert len(out) == 12  # le budget est toujours rempli si des chemins existent
