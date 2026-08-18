"""test_admission_diversity.py - Test Sprint 2.2.2.

Quotas inter-familles : un Einher doit utiliser >= 2 familles
economiques differentes dans ses conditions.

Sinon, l'Einher risque d'etre trop mono-dimensionnel (ex: que du volatility).
"""
from __future__ import annotations

import unittest

from einherjar.research.xgb_einhers.admission import (
    AdmissionConfig,
    check_admission,
    get_einher_families,
    load_feature_family_map,
)
from einherjar.research.xgb_einhers.einher_io import iter_einhers
from einherjar.research.xgb_einhers.types import (
    Condition,
    ConditionNode,
    Einher,
    EinherMetrics,
)


def _make_einher(features: list[str], direction: str = "BUY") -> Einher:
    """Construit un Einher minimal avec une liste de features en AND."""
    if not features:
        raise ValueError("Au moins une feature")
    if len(features) == 1:
        tree = Condition(feature_ref=features[0], operator="<", value=0.0)
    else:
        # AND chain
        tree = Condition(feature_ref=features[0], operator="<", value=0.0)
        for f in features[1:]:
            tree = ConditionNode(
                op="AND",
                left=tree,
                right=Condition(feature_ref=f, operator="<", value=0.0),
            )
    metrics = EinherMetrics(
        n_trades=100, n_tp=60, n_sl=30, n_timeout=10,
        win_rate=0.6, avg_net_return=0.01, total_return=0.5,
        sharpe_ratio=1.5, max_drawdown=-0.1, profit_factor=2.0,
        avg_holding_bars=5.0, buy_hold_return=0.3, alpha=0.2,
    )
    return Einher(
        id="test_einher",
        condition_tree=tree,
        direction=direction,
        amplitude_bars=48,
        tp_pct=0.025,
        sl_pct=0.015,
        universe={"asset": "BTCUSD", "asset_class": "crypto", "timeframe": "1h", "horizon": "2d"},
        metrics=metrics,
        scope="asset",
    )


class TestFamilyMap(unittest.TestCase):
    """Test que le mapping feature->famille est chargeable."""

    def test_load_feature_family_map(self):
        """Doit retourner un dict non-vide."""
        m = load_feature_family_map()
        self.assertIsInstance(m, dict)
        self.assertGreater(len(m), 0)
        # Doit contenir des familles connues
        for fam in ("volatility", "momentum", "trend", "statistical"):
            self.assertIn(fam, set(m.values()),
                          f"Famille {fam} absente du mapping")

    def test_real_features_have_family(self):
        """Les features reelles doivent avoir une famille (pas 'unknown')."""
        m = load_feature_family_map()
        # Au moins 80% des features doivent avoir une famille reelle
        n_unknown = sum(1 for v in m.values() if v == "unknown")
        self.assertLess(n_unknown / len(m), 0.20,
                        f"{n_unknown}/{len(m)} features avec famille 'unknown'")


class TestEinherFamilies(unittest.TestCase):
    """Test extraction des familles depuis l'AST d'un Einher."""

    def test_single_feature_einher(self):
        """Un Einher avec 1 feature => 1 famille."""
        e = _make_einher(["rsi_14"])
        fams = get_einher_families(e)
        self.assertEqual(len(fams), 1)

    def test_two_families_einher(self):
        """Un Einher avec 2 features de 2 familles differentes => 2 familles."""
        e = _make_einher(["rsi_14", "atr_14"])
        fams = get_einher_families(e)
        # rsi_14 est momentum, atr_14 est volatility (a verifier dans la taxo)
        self.assertEqual(len(fams), 2)

    def test_same_family_einher(self):
        """Un Einher avec 2 features de la meme famille => 1 famille."""
        e = _make_einher(["rsi_14", "momentum_10"])
        fams = get_einher_families(e)
        # Les deux sont dans momentum -> 1 seule famille
        self.assertEqual(len(fams), 1)


class TestAdmissionDiversity(unittest.TestCase):
    """Test du quota >= 2 familles dans l'admission."""

    def test_single_family_rejected(self):
        """Un Einher avec 1 seule famille doit etre REJETE (quota=2)."""
        e = _make_einher(["rsi_14"])
        # Verifier d'abord que rsi_14 est bien 1 famille
        fams = get_einher_families(e)
        if len(fams) >= 2:
            self.skipTest("rsi_14 a plus d'1 famille dans la taxonomie, test non applicable")
        passed, reason = check_admission(e, AdmissionConfig())
        self.assertFalse(passed, f"Einher 1-famille devrait etre REJETE : {reason}")
        self.assertIn("famil", reason.lower() if reason else "")

    def test_two_families_accepted(self):
        """Un Einher avec 2+ familles doit etre ACCEPTE (si autres criteres OK)."""
        e = _make_einher(["rsi_14", "atr_14", "obv"])
        passed, reason = check_admission(e, AdmissionConfig())
        self.assertTrue(passed, f"Einher 3-familles devrait etre ACCEPTE : {reason}")

    def test_debug_config_allows_single_family(self):
        """En mode debug (min_families=1), un Einher 1-famille est accepte."""
        e = _make_einher(["rsi_14"])
        fams = get_einher_families(e)
        if len(fams) >= 2:
            self.skipTest("rsi_14 a plus d'1 famille, test non applicable")
        passed, reason = check_admission(e, AdmissionConfig.debug())
        self.assertTrue(passed, f"Mode debug devrait accepter 1-famille : {reason}")


class TestDiversityOnRealEinhers(unittest.TestCase):
    """Verifie la diversite des 7 Einhers debug (sanity check)."""

    def test_print_diversity(self):
        """Affiche la diversite des Einhers reels (pour visibilite)."""
        from pathlib import Path
        jsonl = Path("D:/midas_v2/Einherjar/outputs/einhers_btcusd_1h_2d_debug.jsonl")
        if not jsonl.exists():
            self.skipTest("Pas de fichier d'Einhers")
        einhers = list(iter_einhers(jsonl))
        if not einhers:
            self.skipTest("Aucun Einher dans le JSONL")
        # Compter les familles par Einher
        n_per_fam_count = {}
        for e in einhers:
            fams = get_einher_families(e)
            n = len(fams)
            n_per_fam_count[n] = n_per_fam_count.get(n, 0) + 1
        print(f"\n[DIVERSITY] Repartition des 7 Einhers par nb de familles : {n_per_fam_count}")
        # Sanity check : on doit avoir au moins 1 Einher avec >= 2 familles
        self.assertGreater(n_per_fam_count.get(2, 0) + n_per_fam_count.get(3, 0) +
                           n_per_fam_count.get(4, 0) + n_per_fam_count.get(5, 0), 0,
                           "Aucun Einher n'a 2+ familles - probleme de taxonomie")


if __name__ == "__main__":
    unittest.main()
