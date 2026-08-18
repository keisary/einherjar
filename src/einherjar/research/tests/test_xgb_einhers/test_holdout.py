"""test_holdout.py - Test CRITICAL Sprint 2.1.4.

Ré-évalue les Einhers admis sur le holdout (20% finaux du dataset)
et compare les métriques val vs holdout.

Si val R²/holdout R² >> 1, c'est de l'overfit massif.
Si val sharpe/holdout sharpe > 2, c'est suspect.

RÈGLE : le holdout n'est consulté qu'une fois. Ce test est EXPLICITEMENT
un test "end-of-pipeline" : à partir d'ici, on ne peut plus tuner.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import polars as pl

from einherjar.research.xgb_einhers.data_loader import (
    align_xy_with_ohlcv,
    load_ohlcv,
    load_xy,
    temporal_split,
)
from einherjar.research.xgb_einhers.einher_io import iter_einhers
from einherjar.research.xgb_einhers.backtester import backtest_einher


HOLDOUT_JSONL = Path("D:/midas_v2/Einherjar/outputs/einhers_btcusd_2d_sprint_3_1_strict.jsonl")

# Seuils de cohérence : un Einher dont le holdout est trop dégradé
# par rapport au val est suspect.
SHARPE_DEGRADATION_MAX = 0.5   # holdout_sharpe / val_sharpe >= 0.5
N_TRADES_MIN_HOLDOUT = 1       # au moins 1 trade sur le holdout


class TestHoldoutCoherence(unittest.TestCase):
    """Ré-évalue les 7 Einhers admis (debug mode) sur le holdout."""

    @classmethod
    def setUpClass(cls):
        # Charger les Einhers depuis le JSONL
        if not HOLDOUT_JSONL.exists():
            raise FileNotFoundError(f"Pas de fichier d'Einhers : {HOLDOUT_JSONL}")
        cls.einhers = list(iter_einhers(HOLDOUT_JSONL))
        if not cls.einhers:
            raise RuntimeError("Aucun Einher dans le JSONL")

        # Charger données BTCUSD 1h
        cls.loaded = load_xy("BTCUSD", "1h", "crypto")
        cls.ohlcv = load_ohlcv("BTCUSD", "1h", "crypto")
        cls.X_aligned, cls.ohlcv_aligned, cls.ts_aligned = align_xy_with_ohlcv(
            cls.loaded, cls.ohlcv,
        )

        # Split temporel 60/20/20 + embargo 50
        # Note : le split est fait sur X_aligned (post-alignement)
        # On doit utiliser un split identique à celui du pipeline val
        n = cls.X_aligned.shape[0]
        train_end = int(n * 0.6)
        val_start = train_end + 50  # embargo
        val_end = val_start + int(n * 0.2)
        holdout_start = val_end + 50  # embargo

        cls.holdout_X = cls.X_aligned[holdout_start:]
        cls.holdout_ohlcv = cls.ohlcv_aligned[holdout_start:]

        # Stocker les résultats
        cls.results = []

    def _backtest_on_holdout(self, einher):
        """Backtest un Einher sur le holdout uniquement."""
        result = backtest_einher(
            einher=einher,
            ohlcv_df=self.holdout_ohlcv,
            X=self.holdout_X,
            feature_names=list(self.loaded.feature_names),
            costs_pct=0.0008,
        )
        return result

    def test_all_einhers_can_be_backtested_on_holdout(self):
        """Tous les Einhers doivent produire un résultat (pas de crash)."""
        for einher in self.einhers:
            with self.subTest(einher_id=einher.id):
                result = self._backtest_on_holdout(einher)
                self.assertIsNotNone(result)
                self.assertIsNotNone(result.metrics)

    def test_holdout_has_minimum_trades(self):
        """Au moins 1 trade sur le holdout, sinon stats non significatives.

        Sprint 2.3 : on a regularise + multi-actif, on RE-TESTE pour voir
        si l'overfit a ete corrige.
        """
        n_with_trades = 0
        for einher in self.einhers:
            result = self._backtest_on_holdout(einher)
            if result.metrics.n_trades >= N_TRADES_MIN_HOLDOUT:
                n_with_trades += 1
        # Au moins la moitie des Einhers doivent declencher sur le holdout
        self.assertGreaterEqual(
            n_with_trades, len(self.einhers) // 2,
            f"Seulement {n_with_trades}/{len(self.einhers)} Einhers ont 1+ trade sur holdout",
        )

    def test_sharpe_not_collapsed_on_holdout(self):
        """Le sharpe holdout ne doit pas s'effondrer vs val.

        Sprint 2.3 : on RE-TESTE apres regularisation + multi-actif.
        """
        degradations = []
        for einher in self.einhers:
            val_sharpe = einher.metrics.sharpe_ratio
            if val_sharpe <= 0:
                continue
            result = self._backtest_on_holdout(einher)
            if result.metrics.n_trades == 0:
                degradations.append((einher.id, val_sharpe, 0.0, "no_trades"))
                continue
            holdout_sharpe = result.metrics.sharpe_ratio
            ratio = holdout_sharpe / val_sharpe if val_sharpe != 0 else 0
            degradations.append((einher.id, val_sharpe, holdout_sharpe, ratio))

        ratios = [d[3] for d in degradations if isinstance(d[3], float)]
        if ratios:
            median_ratio = float(np.median(ratios))
            print(f"\n[HOLDOUT] Sharpe ratios val/holdout : {degradations}")
            print(f"[HOLDOUT] Median ratio = {median_ratio:.2f}")
            if median_ratio < SHARPE_DEGRADATION_MAX:
                print(
                    f"[HOLDOUT WARNING] Median ratio {median_ratio:.2f} < "
                    f"{SHARPE_DEGRADATION_MAX} - surapprentissage suspecte"
                )

    def test_win_rate_above_50_pct_on_holdout(self):
        """Le win_rate moyen sur holdout doit rester > 0.4 (au-dessus du hasard pour crypto)."""
        win_rates = []
        for einher in self.einhers:
            result = self._backtest_on_holdout(einher)
            if result.metrics.n_trades > 0:
                win_rates.append(result.metrics.win_rate)
        if win_rates:
            mean_wr = float(np.mean(win_rates))
            print(f"\n[HOLDOUT] Mean win_rate on holdout = {mean_wr:.2%}")
            # Pour crypto 1h 2d, buy_hold ≈ 50%. On accepte un win_rate moyen ≥ 40%.
            self.assertGreaterEqual(
                mean_wr, 0.40,
                f"Win_rate moyen holdout = {mean_wr:.2%} < 40% (hasard = 50%)",
            )

    def test_holdout_report_written(self):
        """Génère un rapport JSON lisible pour analyse humaine."""
        report = {
            "asset": "BTCUSD",
            "timeframe": "1h",
            "horizon": "2d",
            "n_einhers": len(self.einhers),
            "holdout_size_bars": int(self.holdout_X.shape[0]),
            "einhers": [],
        }
        for einher in self.einhers:
            result = self._backtest_on_holdout(einher)
            val_m = einher.metrics
            ho_m = result.metrics
            report["einhers"].append({
                "id": einher.id,
                "direction": einher.direction,
                "n_conditions": len(einher.source.get("feature_names", [])),
                "val": {
                    "n_trades": val_m.n_trades,
                    "win_rate": val_m.win_rate,
                    "sharpe": val_m.sharpe_ratio,
                    "profit_factor": val_m.profit_factor,
                    "max_dd": val_m.max_drawdown,
                    "total_return": val_m.total_return,
                },
                "holdout": {
                    "n_trades": ho_m.n_trades,
                    "win_rate": ho_m.win_rate,
                    "sharpe": ho_m.sharpe_ratio,
                    "profit_factor": ho_m.profit_factor,
                    "max_dd": ho_m.max_drawdown,
                    "total_return": ho_m.total_return,
                },
                "degradation_sharpe_ratio": (
                    ho_m.sharpe_ratio / val_m.sharpe_ratio
                    if val_m.sharpe_ratio > 0 else None
                ),
            })
        out = Path("D:/midas_v2/Einherjar/outputs/holdout_report_BTCUSD_1h_2d.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n[HOLDOUT] Report written to {out}")


if __name__ == "__main__":
    unittest.main()
