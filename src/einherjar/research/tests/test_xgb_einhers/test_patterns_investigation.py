"""test_patterns_investigation.py - Sprint 2.2.4.

Investigue pourquoi 0 pattern et 0 market_structure features sont
utilisees par les modeles XGBoost.

Hypotheses a tester :
- H1 : Les patterns sont des bool (0/1) avec tres peu de 1 (sparse)
- H2 : Les market_structure sont des features calculees sur des fenetres
        longues, donc trop smooth/correllees
- H3 : XGBoost privilegie les features continues (statistical, risk)
        par defaut (split threshold plus informatif)

Methode : charger BTCUSD 1h 2d, entrainer XGBoost, mesurer :
- % de 1 dans chaque pattern (sparsity)
- importance moyenne par famille
- nombre de features a importance > 0 par famille
"""
from __future__ import annotations

import json
import unittest
from collections import defaultdict
from pathlib import Path

import numpy as np

from einherjar.research.xgb_einhers.data_loader import load_xy
from einherjar.research.xgb_einhers.label_engineer import build_target
from einherjar.research.xgb_einhers.model import (
    GBDTConfig,
    feature_importance,
    train_gbdt,
)

TAXONOMY_PATH = Path(
    "D:/midas_v2/Einherjar/src/einherjar/research/config/features_taxonomy.json"
)
HORIZON_IDX = 3  # 2d


def _load_family_map() -> dict[str, str]:
    with open(TAXONOMY_PATH) as f:
        tax = json.load(f)
    return {
        name: meta.get("economic_family", "unknown")
        for name, meta in tax["features"].items()
    }


class TestPatternsInvestigation(unittest.TestCase):
    """Diagnostic complet patterns + market_structure."""

    @classmethod
    def setUpClass(cls):
        cls.loaded = load_xy("BTCUSD", "1h", "crypto")
        target, valid_mask, _ = build_target(cls.loaded, HORIZON_IDX)
        X = cls.loaded.X[valid_mask]
        y = target[valid_mask].astype(np.float32)
        cls.X = X
        cls.y = y
        cls.feature_names = list(cls.loaded.feature_names)
        cls.family_map = _load_family_map()

        # Train
        from einherjar.research.xgb_einhers.data_loader import temporal_split
        cls.split = temporal_split(X, y)
        model, backend = train_gbdt(
            cls.split.train_X, cls.split.train_y,
            cls.split.val_X, cls.split.val_y,
            config=GBDTConfig(n_estimators=50, max_depth=4, learning_rate=0.05),
        )
        cls.model = model
        cls.backend = backend
        cls.importances = feature_importance(model, backend, cls.feature_names)

    def test_h1_pattern_sparsity(self):
        """H1 : les patterns sont-ils tres sparse (peu de 1) ?

        Si > 90% des patterns ont < 5% de True, XGBoost ne peut pas
        les utiliser comme split majeur.
        """
        pattern_features = [
            (i, name) for i, name in enumerate(self.feature_names)
            if self.family_map.get(name) == "price_action"
            and name.startswith("pattern_")
        ]
        if not pattern_features:
            self.skipTest("Aucun pattern dans le dataset")
        sparsity = []
        for idx, name in pattern_features[:30]:  # top 30 pour vitesse
            col = self.X[:, idx]
            if col.dtype.kind in "fc":  # float ou compatible
                pct_true = float(np.mean(col > 0.5))
            else:
                pct_true = float(np.mean(col == 1))
            sparsity.append((name, pct_true))
        sparsity.sort(key=lambda x: x[1])
        print("\n[H1 SPARSITY] Top 10 patterns les plus rares :")
        for name, pct in sparsity[:10]:
            print(f"  {name:40s} : {pct*100:.2f}% True")
        print(f"[H1 SPARSITY] Mediane pct_True = {np.median([s[1] for s in sparsity])*100:.2f}%")

    def test_h2_market_structure_smoothness(self):
        """H2 : les market_structure sont-elles smooth (peu de variance) ?

        Si std/mean est tres bas, la feature n'apporte pas d'info
        discriminante.
        """
        ms_features = [
            (i, name) for i, name in enumerate(self.feature_names)
            if self.family_map.get(name) == "market_structure"
        ]
        if not ms_features:
            self.skipTest("Aucune market_structure dans le dataset")
        cv = []  # coefficient of variation
        for idx, name in ms_features[:30]:
            col = self.X[:, idx]
            mean = float(np.mean(col))
            std = float(np.std(col))
            cv_val = abs(std / mean) if abs(mean) > 1e-9 else 0.0
            cv.append((name, mean, std, cv_val))
        cv.sort(key=lambda x: x[3])
        print("\n[H2 SMOOTHNESS] Top 10 market_structure les moins variables :")
        for name, mean, std, cvv in cv[:10]:
            print(f"  {name:40s} : mean={mean:+.4f}, std={std:.4f}, CV={cvv:.3f}")
        print(f"[H2 SMOOTHNESS] CV median = {np.median([c[3] for c in cv]):.3f}")

    def test_h3_feature_usage_by_family(self):
        """H3 : XGBoost utilise-t-il certaines familles plus que d'autres ?

        On groupe par famille et on calcule :
        - nb total features dans la famille
        - nb features avec importance > 0
        - somme des importances
        """
        by_family: dict[str, list[float]] = defaultdict(list)
        for name, imp in self.importances.items():
            fam = self.family_map.get(name, "unknown")
            by_family[fam].append(float(imp))

        stats = []
        for fam, imps in by_family.items():
            total = len(imps)
            used = sum(1 for v in imps if v > 0)
            sum_imp = sum(imps)
            stats.append((fam, total, used, sum_imp))
        stats.sort(key=lambda x: -x[3])  # par importance totale
        print("\n[H3 USAGE] Importance par famille :")
        print(f"  {'Famille':<25} {'Total':>6} {'Used':>6} {'%used':>6} {'SumImp':>10}")
        for fam, total, used, sum_imp in stats:
            pct = used / total * 100 if total else 0
            print(f"  {fam:<25} {total:>6} {used:>6} {pct:>5.1f}% {sum_imp:>10.4f}")

        # Sanity : le test doit juste printer, pas fail
        # On documente la famille la plus utilisee
        most_used = stats[0]
        self.assertGreater(most_used[2], 0, "Aucune famille n'a de feature utilisee")

    def test_write_investigation_report(self):
        """Ecrit un rapport JSON pour analyse."""
        # Par famille
        by_family_total: dict[str, int] = defaultdict(int)
        by_family_used: dict[str, int] = defaultdict(int)
        by_family_imp: dict[str, float] = defaultdict(float)
        for name in self.feature_names:
            fam = self.family_map.get(name, "unknown")
            by_family_total[fam] += 1
        for name, imp in self.importances.items():
            fam = self.family_map.get(name, "unknown")
            if imp > 0:
                by_family_used[fam] += 1
            by_family_imp[fam] += float(imp)

        report = {
            "asset": "BTCUSD",
            "timeframe": "1h",
            "horizon": "2d",
            "n_features_total": len(self.feature_names),
            "n_features_with_importance": sum(1 for v in self.importances.values() if v > 0),
            "by_family": {},
        }
        for fam in sorted(by_family_total.keys()):
            total = by_family_total[fam]
            used = by_family_used.get(fam, 0)
            pct = used / total * 100 if total else 0
            report["by_family"][fam] = {
                "total": total,
                "used": used,
                "pct_used": round(pct, 2),
                "sum_importance": round(by_family_imp.get(fam, 0.0), 4),
            }
        out = Path("D:/midas_v2/Einherjar/outputs/investigation_patterns_2_2_4.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n[REPORT] Written to {out}")
        # Documented but never failing
        self.assertTrue(out.exists())


if __name__ == "__main__":
    unittest.main()
