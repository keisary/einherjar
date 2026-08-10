"""Tests de la correction DSR 2026-08-10 (unité annualisée + horizon en années).

Contexte : le mode historique (Sharpe par trade avec T=n_trades) appliquait
sqrt(T-1) sur des observations de durées hétérogènes et rendait le seuil 0.95
quasi infranchissable pour des stratégies BTCUSD réalistes (0/116 admis sur
4 TF). Le mode annualisé (n_val_years fourni) utilise sharpe_net (annualisé
par test_on) et le t-stat approx. SR*sqrt(Y).

Valeurs de référence (Bailey & LP, 2014) :
  z = SR*sqrt(Y)/SE - sqrt(2*ln(n_trials)) ; p = Phi(z)
  - SR=3.0, Y=1, n_trials=1  -> z=3.00  -> p~0.9987 >= 0.95 (passe)
  - SR=3.0, Y=1, n_trials=550 -> e_max~3.55 -> z=-0.55 -> p~0.29 (échoue)
  - SR=6.0, Y=1, n_trials=33 -> e_max~2.64 -> z=3.36 -> p~0.9996 (passe)
"""

from __future__ import annotations

import unittest

from einherjar.research.admission.criteria import evaluate_dsr
from einherjar.research.config.loader import load_config
from einherjar.research.utils.types import MesuresBrutes

CONFIG_PATH = "src/einherjar/research/config"


def _m(
    sharpe_net: float,
    n_signals: int = 200,
    ret_mean: float = 0.001,
    ret_std: float = 0.01,
) -> MesuresBrutes:
    return MesuresBrutes(
        n_signals=n_signals,
        n_tp_hit=int(n_signals * 0.4),
        n_sl_hit=int(n_signals * 0.2),
        n_timeout=int(n_signals * 0.4),
        mfe_mean_pct=1.0, mae_mean_pct=0.5,
        mfe_p50=0.8, mfe_p75=1.2, mfe_p90=2.0,
        mae_p50=0.3, mae_p75=0.6, mae_p90=1.0,
        ret_mean_pct_net=ret_mean, ret_std_pct=ret_std, sharpe_net=sharpe_net,
        tp_hit_rate=0.4, sl_hit_rate=0.2, timeout_rate=0.4,
        avg_holding_period=10.0, avg_time_to_amplitude=8.0,
        bootstrap_sharpe_ci_low=0.0, bootstrap_sharpe_ci_high=0.0,
        bootstrap_ret_ci_low=0.0, bootstrap_ret_ci_high=0.0,
    )


class TestDsrCorrectionAnnualisee(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.config = load_config(str(CONFIG_PATH))

    def test_sharpe_annuel_3_sur_1_an_passe_essai_unique(self):
        v = evaluate_dsr(_m(sharpe_net=3.0), self.config, n_indep_trials=1, n_val_years=1.0)
        self.assertTrue(v.passed, f"DSR={v.observed}")

    def test_meme_sharpe_550_essais_echoue_deflation(self):
        v = evaluate_dsr(_m(sharpe_net=3.0), self.config, n_indep_trials=550, n_val_years=1.0)
        self.assertFalse(v.passed, f"DSR={v.observed}")

    def test_sharpe_annuel_6_sur_1_an_passe_meme_33_essais(self):
        v = evaluate_dsr(_m(sharpe_net=6.0), self.config, n_indep_trials=33, n_val_years=1.0)
        self.assertTrue(v.passed, f"DSR={v.observed}")

    def test_mode_historique_conserve_comportement(self):
        # Mode historique : sharpe = ret_mean/ret_std = 0.1, T=200 -> p~0.92 < 0.95.
        v = evaluate_dsr(
            _m(sharpe_net=99.0, ret_mean=0.05, ret_std=0.5),
            self.config, n_indep_trials=1, n_val_years=None,
        )
        self.assertEqual(v.name, "DSR")
        self.assertFalse(v.passed, f"DSR={v.observed}")

    def test_sharpe_nan_rejete(self):
        v = evaluate_dsr(_m(sharpe_net=float("nan")), self.config, n_indep_trials=1, n_val_years=1.0)
        self.assertFalse(v.passed)

    def test_horizon_zero_retombe_sur_mode_annuel_borne(self):
        v = evaluate_dsr(_m(sharpe_net=3.0), self.config, n_indep_trials=1, n_val_years=0.0)
        # n_val_years=0 -> mode historique (sharpe par trade ~0.1) -> échoue.
        self.assertFalse(v.passed)


if __name__ == "__main__":
    unittest.main()