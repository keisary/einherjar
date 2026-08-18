"""test_cross_asset.py - Sprint 2.6.1.

Teste les 14 Einhers BTC sur d'autres actifs (ETH, LTC, ADA, BCH)
pour mesurer la generalisation cross-asset.

Si un Einher BTC performe aussi sur ETH/LTC/ADA/BCH, c'est un signal
cross-actif (generalisable). Sinon c'est un artefact BTC-specifique.

Criteres de generalisation :
- win_rate >= 0.4 (au-dessus du hasard)
- n_trades >= 5 (significativite statistique)
- sharpe >= 0.0 (pas de perte)
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
)
from einherjar.research.xgb_einhers.einher_io import iter_einhers
from einherjar.research.xgb_einhers.backtester import backtest_einher


BTC_EINHERS_JSONL = Path("D:/midas_v2/Einherjar/outputs/einhers_btcusd_2d_sprint_2_5.jsonl")
CROSS_ASSETS = ["ETHUSD", "LTCUSD", "ADAUSD", "BCHUSD"]  # actifs a tester
MIN_TRADES = 5
MIN_WIN_RATE = 0.40


class TestCrossAssetGeneralization(unittest.TestCase):
    """Verifie que les Einhers BTC se generalisent aux autres actifs."""

    @classmethod
    def setUpClass(cls):
        # Charger les Einhers BTC
        if not BTC_EINHERS_JSONL.exists():
            raise FileNotFoundError(f"Pas de fichier d'Einhers : {BTC_EINHERS_JSONL}")
        cls.btc_einhers = list(iter_einhers(BTC_EINHERS_JSONL))
        if not cls.btc_einhers:
            raise RuntimeError("Aucun Einher dans le JSONL")
        # Dedupliquer les Einhers (le runner a pu append plusieurs fois)
        seen = set()
        unique = []
        for e in cls.btc_einhers:
            # Identifie par les conditions
            key = str(e.condition_tree.to_dict()) if hasattr(e.condition_tree, 'to_dict') else str(e.condition_tree)
            if key not in seen:
                seen.add(key)
                unique.append(e)
        cls.btc_einhers = unique
        # Charger les features BTC (reference pour les noms de colonnes)
        cls.btc_loaded = load_xy("BTCUSD", "1h", "crypto")

    def _backtest_on_asset(self, einher, asset: str):
        """Backtest un Einher sur un autre actif."""
        try:
            loaded = load_xy(asset, "1h", "crypto")
        except FileNotFoundError:
            return None, f"asset {asset} not found"
        try:
            ohlcv = load_ohlcv(asset, "1h", "crypto")
        except FileNotFoundError:
            return None, f"ohlcv {asset} not found"
        try:
            X_aligned, ohlcv_aligned, _ = align_xy_with_ohlcv(loaded, ohlcv)
        except Exception as e:
            return None, f"align error: {e}"
        result = backtest_einher(
            einher=einher,
            ohlcv_df=ohlcv_aligned,
            X=X_aligned,
            feature_names=list(loaded.feature_names),
            costs_pct=0.0008,
        )
        return result, None

    def test_cross_asset_report(self):
        """Genere un rapport JSON de cross-asset validation."""
        report = {
            "btc_einhers": len(self.btc_einhers),
            "assets_tested": CROSS_ASSETS,
            "results": {},
        }
        # Pour chaque actif
        for asset in CROSS_ASSETS:
            asset_results = []
            n_triggered = 0  # n_trades >= MIN_TRADES
            n_profitable = 0  # sharpe > 0
            n_passing = 0  # win_rate >= MIN_WIN_RATE
            for einher in self.btc_einhers:
                result, err = self._backtest_on_asset(einher, asset)
                if result is None:
                    asset_results.append({
                        "id": einher.id,
                        "error": err,
                    })
                    continue
                m = result.metrics
                triggered = m.n_trades >= MIN_TRADES
                passing = m.n_trades >= MIN_TRADES and m.win_rate >= MIN_WIN_RATE
                if triggered:
                    n_triggered += 1
                if m.sharpe_ratio > 0:
                    n_profitable += 1
                if passing:
                    n_passing += 1
                asset_results.append({
                    "id": einher.id,
                    "n_trades": m.n_trades,
                    "win_rate": round(m.win_rate, 4),
                    "sharpe": round(m.sharpe_ratio, 4),
                    "profit_factor": round(m.profit_factor, 4),
                    "triggered": triggered,
                    "passing": passing,
                })
            # Stats agregees
            n_total = len(self.btc_einhers)
            report["results"][asset] = {
                "n_einhers": n_total,
                "n_triggered": n_triggered,
                "n_profitable": n_profitable,
                "n_passing": n_passing,
                "pct_triggered": round(100 * n_triggered / n_total, 1) if n_total else 0,
                "pct_profitable": round(100 * n_profitable / n_total, 1) if n_total else 0,
                "pct_passing": round(100 * n_passing / n_total, 1) if n_total else 0,
                "details": asset_results,
            }
        # Sauver le rapport
        out = Path("D:/midas_v2/Einherjar/outputs/cross_asset_report_BTC_Einhers.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        # Afficher un resume
        print(f"\n[CROSS-ASSET] {len(self.btc_einhers)} Einhers BTC testes sur {len(CROSS_ASSETS)} actifs :")
        print(f"  {'Asset':<10} {'%declenche':>12} {'%profitable':>12} {'%passing':>10}")
        for asset in CROSS_ASSETS:
            r = report["results"][asset]
            print(f"  {asset:<10} {r['pct_triggered']:>11.1f}% {r['pct_profitable']:>11.1f}% {r['pct_passing']:>9.1f}%")

    def test_at_least_one_asset_generalizes(self):
        """Au moins 1 actif doit avoir un passing rate > 30% (generalisation)."""
        results = {}
        for asset in CROSS_ASSETS:
            n_passing = 0
            n_total = 0
            for einher in self.btc_einhers:
                result, err = self._backtest_on_asset(einher, asset)
                if result is None:
                    continue
                n_total += 1
                if result.metrics.n_trades >= MIN_TRADES and result.metrics.win_rate >= MIN_WIN_RATE:
                    n_passing += 1
            if n_total > 0:
                results[asset] = n_passing / n_total
        # Le test passe si au moins 1 actif a un passing rate >= 30%
        max_rate = max(results.values()) if results else 0
        self.assertGreaterEqual(
            max_rate, 0.30,
            f"Aucun actif n'a un passing rate >= 30% (max = {max_rate:.2%}). "
            f"Resultats : {results}",
        )

    def test_write_summary(self):
        """Ecrit un summary markdown pour lecture humaine."""
        report_path = Path("D:/midas_v2/Einherjar/outputs/cross_asset_report_BTC_Einhers.json")
        if not report_path.exists():
            self.skipTest("Rapport cross-asset pas encore genere")
        with open(report_path) as f:
            report = json.load(f)
        # Generer un summary markdown
        lines = [
            "# Cross-Asset Validation Report",
            "",
            f"**Date**: 2026-08-17",
            f"**Einhers BTC testes**: {report['btc_einhers']}",
            f"**Actifs cibles**: {', '.join(report['assets_tested'])}",
            "",
            "## Resume",
            "",
            "| Asset | % declenche (>5 trades) | % profitable (sharpe>0) | % passing (wr>40%) |",
            "|---|---|---|---|",
        ]
        for asset in report["assets_tested"]:
            r = report["results"][asset]
            lines.append(
                f"| {asset} | {r['pct_triggered']:.1f}% | {r['pct_profitable']:.1f}% | {r['pct_passing']:.1f}% |"
            )
        out = Path("D:/midas_v2/Einherjar/outputs/cross_asset_summary.md")
        with open(out, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"\n[CROSS-ASSET] Summary ecrit dans {out}")


if __name__ == "__main__":
    unittest.main()
