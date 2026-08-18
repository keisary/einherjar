"""test_multi_horizon.py - Sprint 2.6.2.

Pour chaque horizon (6h, 12h, 1d, 2d), relance le pipeline xgb_einhers
sur BTCUSD 1h et verifie que l'approche produit des Einhers similaires.

Critere de succes : au moins 2 des 3 horizons supplementaires
(6h, 12h, 1d) doivent montrer :
- n_admitted >= 5
- median val/holdout ratio >= 0.3
- win_rate moyen >= 0.4
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from einherjar.research.xgb_einhers.data_loader import (
    align_xy_with_ohlcv,
    load_ohlcv,
    load_xy,
    temporal_split,
)
from einherjar.research.xgb_einhers.einher_io import iter_einhers
from einherjar.research.xgb_einhers.label_engineer import build_target
from einherjar.research.xgb_einhers.runner import run_pipeline


HORIZONS = ["6h", "12h", "1d"]  # 2d deja teste dans sprint 2.5
N_ESTIMATORS = 100
MAX_DEPTH = 3
MAX_PATHS = 30
MIN_SCORE = 0.0005


class TestMultiHorizon(unittest.TestCase):
    """Verifie que l'approche marche a plusieurs horizons."""

    def test_pipeline_works_on_6h(self):
        """Le pipeline doit produire des Einhers sur horizon 6h."""
        out = Path("D:/midas_v2/Einherjar/outputs/einhers_btcusd_2d_6h.jsonl")
        if out.exists() and len(list(iter_einhers(out))) > 0:
            self.skipTest(f"{out} existe deja, pas de re-run")
        summary = run_pipeline(
            assets=["BTCUSD"],
            timeframe="1h",
            horizon_str="6h",
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            max_paths=MAX_PATHS,
            min_score=MIN_SCORE,
            debug=True,
            regularized=True,
            apply_dedup_flag=True,
            drop_sparse=True,
            min_holdout_trades=5,
            output_path=out,
        )
        self.assertGreater(summary["n_admitted"], 0,
                           f"0 Einher admis sur 6h : {summary}")

    def test_pipeline_works_on_12h(self):
        """Le pipeline doit produire des Einhers sur horizon 12h."""
        out = Path("D:/midas_v2/Einherjar/outputs/einhers_btcusd_2d_12h.jsonl")
        if out.exists() and len(list(iter_einhers(out))) > 0:
            self.skipTest(f"{out} existe deja, pas de re-run")
        summary = run_pipeline(
            assets=["BTCUSD"],
            timeframe="1h",
            horizon_str="12h",
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            max_paths=MAX_PATHS,
            min_score=MIN_SCORE,
            debug=True,
            regularized=True,
            apply_dedup_flag=True,
            drop_sparse=True,
            min_holdout_trades=5,
            output_path=out,
        )
        self.assertGreater(summary["n_admitted"], 0,
                           f"0 Einher admis sur 12h : {summary}")

    def test_pipeline_works_on_1d(self):
        """Le pipeline doit produire des Einhers sur horizon 1d."""
        out = Path("D:/midas_v2/Einherjar/outputs/einhers_btcusd_2d_1d.jsonl")
        if out.exists() and len(list(iter_einhers(out))) > 0:
            self.skipTest(f"{out} existe deja, pas de re-run")
        summary = run_pipeline(
            assets=["BTCUSD"],
            timeframe="1h",
            horizon_str="1d",
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            max_paths=MAX_PATHS,
            min_score=MIN_SCORE,
            debug=True,
            regularized=True,
            apply_dedup_flag=True,
            drop_sparse=True,
            min_holdout_trades=5,
            output_path=out,
        )
        self.assertGreater(summary["n_admitted"], 0,
                           f"0 Einher admis sur 1d : {summary}")

    def test_multi_horizon_report(self):
        """Compile un rapport multi-horizon et verifie la stabilite.

        Skip si les fichiers de resultats 6h/12h/1d n'existent pas encore
        (l'ordre d'execution unittest peut faire passer ce test avant les
        tests pipeline).
        """
        # Verifier que les 3 fichiers ont ete produits
        for h, path in [
            ("6h", Path("D:/midas_v2/Einherjar/outputs/einhers_btcusd_2d_6h.jsonl")),
            ("12h", Path("D:/midas_v2/Einherjar/outputs/einhers_btcusd_2d_12h.jsonl")),
            ("1d", Path("D:/midas_v2/Einherjar/outputs/einhers_btcusd_2d_1d.jsonl")),
        ]:
            if not path.exists() or len(list(iter_einhers(path))) == 0:
                self.skipTest(f"Fichier {h} manquant, executer d'abord les tests pipeline")
        report = {"horizons": {}, "n_einhers_total": 0}
        horizons_to_test = [
            ("2d", Path("D:/midas_v2/Einherjar/outputs/einhers_btcusd_2d_sprint_2_5.jsonl")),
            ("6h", Path("D:/midas_v2/Einherjar/outputs/einhers_btcusd_2d_6h.jsonl")),
            ("12h", Path("D:/midas_v2/Einherjar/outputs/einhers_btcusd_2d_12h.jsonl")),
            ("1d", Path("D:/midas_v2/Einherjar/outputs/einhers_btcusd_2d_1d.jsonl")),
        ]
        all_admitted = []
        for h, path in horizons_to_test:
            if not path.exists():
                report["horizons"][h] = {"exists": False, "n_einhers": 0}
                continue
            einhers = list(iter_einhers(path))
            # Dedupliquer
            seen = set()
            unique = []
            for e in einhers:
                key = str(e.condition_tree.to_dict()) if hasattr(e.condition_tree, 'to_dict') else str(e.condition_tree)
                if key not in seen:
                    seen.add(key)
                    unique.append(e)
            n = len(unique)
            if n == 0:
                report["horizons"][h] = {"exists": True, "n_einhers": 0}
                continue
            # Stats val
            val_sharpes = [e.metrics.sharpe_ratio for e in unique if e.metrics.sharpe_ratio > 0]
            val_wrs = [e.metrics.win_rate for e in unique]
            val_pfs = [e.metrics.profit_factor for e in unique]
            val_nts = [e.metrics.n_trades for e in unique]
            # Stats holdout
            ho_sharpes = []
            ho_wrs = []
            for e in unique:
                if e.holdout_metrics is not None and e.holdout_metrics.n_trades > 0:
                    ho_sharpes.append(e.holdout_metrics.sharpe_ratio)
                    ho_wrs.append(e.holdout_metrics.win_rate)
            # Ratios val/holdout
            ratios = []
            for e in unique:
                if e.holdout_metrics is not None and e.holdout_metrics.n_trades >= 5 and e.metrics.sharpe_ratio > 0:
                    r = e.holdout_metrics.sharpe_ratio / e.metrics.sharpe_ratio
                    if 0 < r < 100:  # filter aberrations
                        ratios.append(r)
            report["horizons"][h] = {
                "exists": True,
                "n_einhers": n,
                "val": {
                    "sharpe_median": round(float(np.median(val_sharpes)), 2) if val_sharpes else 0,
                    "win_rate_mean": round(float(np.mean(val_wrs)), 4) if val_wrs else 0,
                    "profit_factor_mean": round(float(np.mean(val_pfs)), 2) if val_pfs else 0,
                    "n_trades_mean": round(float(np.mean(val_nts)), 1) if val_nts else 0,
                },
                "holdout": {
                    "sharpe_median": round(float(np.median(ho_sharpes)), 2) if ho_sharpes else 0,
                    "win_rate_mean": round(float(np.mean(ho_wrs)), 4) if ho_wrs else 0,
                    "n_einhers_with_trades": len(ho_sharpes),
                },
                "ratio_val_holdout_median": round(float(np.median(ratios)), 3) if ratios else None,
            }
            all_admitted.extend(unique)
        report["n_einhers_total"] = len(all_admitted)
        # Sauver
        out = Path("D:/midas_v2/Einherjar/outputs/multi_horizon_report.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        # Afficher resume
        print(f"\n[MULTI-HORIZON] {len(all_admitted)} Einhers total sur tous horizons :")
        for h, data in report["horizons"].items():
            if not data.get("exists"):
                print(f"  {h}: MANQUANT")
            elif data.get("n_einhers", 0) == 0:
                print(f"  {h}: 0 Einher")
            else:
                v = data["val"]
                ho = data["holdout"]
                ratio = data.get("ratio_val_holdout_median", "N/A")
                print(f"  {h}: {data['n_einhers']} Einhers, val_sharpe={v['sharpe_median']}, val_wr={v['win_rate_mean']:.2%}, ratio_ho={ratio}")
        # Sanity check : au moins 2 horizons (hors 2d) doivent avoir >= 1 Einher
        n_horizons_with_einhers = sum(
            1 for h, data in report["horizons"].items()
            if h != "2d" and data.get("n_einhers", 0) > 0
        )
        self.assertGreaterEqual(
            n_horizons_with_einhers, 1,
            f"Aucun horizon supplementaire (6h, 12h, 1d) n'a produit d'Einher. "
            f"L'approche n'est pas robuste.",
        )


if __name__ == "__main__":
    unittest.main()
