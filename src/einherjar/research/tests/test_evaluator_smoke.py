"""
tests/test_evaluator_smoke.py — Smoke test du moteur d'évaluation.

Vérifie que le pipeline complet fonctionne avec des données synthétiques :
  - OhlcvProvider (backend in-memory) charge 1000 bougies
  - FeaturesProvider calcule des features (engine identité pour le test)
  - EvaluationEngine calibre, évalue sur val, refuse un 2e accès holdout

Ce test ne valide PAS la qualité des métriques (c'est un smoke test, pas
un test fonctionnel). Il valide :
  - imports,
  - API publique,
  - non-régression sur le pattern d'usage principal.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl

from einherjar.research.config.loader import load_config
from einherjar.research.data.features import FeaturesProvider
from einherjar.research.data.ohlcv import OhlcvProvider, _InMemoryBackend
from einherjar.research.engine.evaluator import (
    EvaluationEngine,
    HoldoutAccessError,
    TradingCosts,
    make_default_engine,
)
from einherjar.research.utils.types import (
    Amplitude,
    AmplitudeUnit,
    Condition,
    CompareOp,
    Direction,
    Hypothesis,
    Universe,
)


def _make_synthetic_ohlcv(n: int = 1000, start: str = "2024-01-01") -> pl.DataFrame:
    """Génère une série OHLCV synthétique (random walk)."""
    rng = np.random.default_rng(seed=42)
    close = 100.0 * np.cumprod(1.0 + rng.normal(0, 0.01, n))
    high = close * (1.0 + np.abs(rng.normal(0, 0.005, n)))
    low = close * (1.0 - np.abs(rng.normal(0, 0.005, n)))
    open_ = close * (1.0 + rng.normal(0, 0.002, n))
    volume = rng.integers(100, 10_000, n).astype(float)
    ts = [datetime.fromisoformat(start) + timedelta(hours=i) for i in range(n)]
    return pl.DataFrame({
        "asset": ["BTCUSD"] * n,
        "timeframe": ["1h"] * n,
        "timestamp": ts,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


def _make_synthetic_features(ohlcv_df: pl.DataFrame) -> pl.DataFrame:
    """Génère une frame de features synthétiques (moyennes mobiles sur close)."""
    close = ohlcv_df["close"].to_numpy()
    feats = {
        "timestamp": ohlcv_df["timestamp"],
        "open": ohlcv_df["open"],
        "high": ohlcv_df["high"],
        "low": ohlcv_df["low"],
        "close": ohlcv_df["close"],
        "volume": ohlcv_df["volume"],
    }
    # On génère 5 features fictives pour tester le provider (largement < 218).
    for i, window in enumerate([5, 10, 20, 50, 100]):
        ma = np.full_like(close, np.nan, dtype=float)
        for k in range(window - 1, len(close)):
            ma[k] = float(np.mean(close[k - window + 1 : k + 1]))
        feats[f"ma_{window}"] = ma
    return pl.DataFrame(feats)


class TestOhlcvProvider(unittest.TestCase):
    """Smoke test : chargement OHLCV via backend in-memory."""

    def test_load_returns_validated_frame(self):
        df = _make_synthetic_ohlcv(n=500)
        provider = OhlcvProvider(backend=_InMemoryBackend())
        # On enregistre manuellement.
        provider._backend.register("BTCUSD", "1h", df)  # noqa: SLF001 (introspection pour test)

        frame = provider.load("BTCUSD", "1h", data_version="v1_test")
        self.assertEqual(frame.n_bougies, 500)
        self.assertEqual(frame.asset, "BTCUSD")
        self.assertEqual(frame.timeframe, "1h")
        self.assertIn("close", frame.df.columns)

    def test_cache_hit(self):
        df = _make_synthetic_ohlcv(n=200)
        provider = OhlcvProvider(backend=_InMemoryBackend())
        provider._backend.register("BTCUSD", "1h", df)  # noqa: SLF001
        f1 = provider.load("BTCUSD", "1h", data_version="v1")
        f2 = provider.load("BTCUSD", "1h", data_version="v1", use_cache=True)
        # Cache hit : même objet retourné.
        self.assertIs(f1, f2)

    def test_invalidate_clears_cache(self):
        df = _make_synthetic_ohlcv(n=100)
        provider = OhlcvProvider(backend=_InMemoryBackend())
        provider._backend.register("BTCUSD", "1h", df)  # noqa: SLF001
        provider.load("BTCUSD", "1h", data_version="v1")
        n = provider.invalidate()
        self.assertEqual(n, 1)
        self.assertEqual(provider.list_assets(), [])


class TestFeaturesProvider(unittest.TestCase):
    """Smoke test : calcul de features avec engine identité (test)."""

    def test_compute_returns_frame(self):
        # Config réelle (taxonomie chargée).
        cfg = load_config("src/einherjar/research/config")
        # Le test engine identité retourne le DataFrame tel quel.
        from einherjar.research.data.features import make_test_provider
        provider = make_test_provider(cfg)
        df = _make_synthetic_features(_make_synthetic_ohlcv(n=200))
        from einherjar.research.data.ohlcv import OhlcvFrame
        ohlcv = OhlcvFrame(asset="BTCUSD", timeframe="1h", df=df, data_version="v1")
        frame = provider.compute(ohlcv)
        # L'engine identité retourne l'OHLCV tel quel : 0 feature exploitable
        # (les ma_* sont fictifs et ne sont pas dans la taxonomie 218).
        # On vérifie juste que le frame est bien construit.
        self.assertEqual(frame.n_bougies, 200)
        self.assertEqual(frame.asset, "BTCUSD")
        self.assertEqual(frame.timeframe, "1h")


def _make_identity_engine():
    """Petit stub d'engine identité pour les tests (utile ailleurs)."""
    from einherjar.research.data.features import _IdentityFeatureEngine
    return _IdentityFeatureEngine()


class TestEvaluationEngine(unittest.TestCase):
    """Smoke test : pipeline complet (calibrate + test_on + holdout)."""

    def setUp(self):
        # Config réelle (taxonomie chargée).
        self.config = load_config("src/einherjar/research/config")
        self.data_version = "smoke_v1"
        self.engine = make_default_engine(self.config, self.data_version, seed=42)

    def _make_data(self, n: int = 1000):
        df = _make_synthetic_ohlcv(n=n)
        feats = _make_synthetic_features(df)
        from einherjar.research.data.ohlcv import OhlcvFrame
        from einherjar.research.data.features import FeaturesFrame
        ohlcv = OhlcvFrame(asset="BTCUSD", timeframe="1h", df=df, data_version=self.data_version)
        features = FeaturesFrame(
            asset="BTCUSD", timeframe="1h", df=feats,
            feature_names=tuple(c for c in feats.columns if c.startswith("ma_")),
            data_version=self.data_version,
        )
        return ohlcv, features

    def test_calibrate_returns_frozen_params(self):
        ohlcv, features = self._make_data(n=600)
        h = Hypothesis(
            id="hyp_smoke",
            condition_tree=Condition(feature_ref="ma_20", operator=CompareOp.LT, value=105.0),
            amplitude=Amplitude(valeur=10.0, unité=AmplitudeUnit.PRICE_ABSOLU, direction_implicite=Direction.LONG),
            direction=Direction.LONG,
            universe=Universe(assets=("BTCUSD",), timeframes=("1h",)),
        )
        calibrated = self.engine.train_calibrate(h, ohlcv, features)
        self.assertGreater(calibrated.n_window, 0)
        # SL et TP sont maintenant des distances relatives (multiples d'ATR).
        # Les deux doivent être > 0, mais il n'y a aucune garantie que TP > SL
        # (ça dépend de la dynamique de marché : si MAE_p75 > MFE_p50, SL > TP en ATR).
        self.assertGreater(calibrated.sl_n_atr, 0.0)
        self.assertGreater(calibrated.tp_n_atr, 0.0)
        self.assertGreater(calibrated.atr_p50, 0.0)
        # Frozen
        import dataclasses
        with self.assertRaises(dataclasses.FrozenInstanceError):
            calibrated.n_window = 999  # type: ignore[misc]

    def test_full_pipeline_train_val(self):
        ohlcv, features = self._make_data(n=1000)
        # Split arbitraire 60/40.
        from einherjar.research.data.ohlcv import OhlcvFrame
        from einherjar.research.data.features import FeaturesFrame
        train_df = ohlcv.df.head(600)
        val_df = ohlcv.df.tail(400)
        train_feats = features.df.head(600)
        val_feats = features.df.tail(400)

        train_ohlcv = OhlcvFrame(asset="BTCUSD", timeframe="1h", df=train_df, data_version="v1")
        val_ohlcv = OhlcvFrame(asset="BTCUSD", timeframe="1h", df=val_df, data_version="v1")
        train_features = FeaturesFrame(
            asset="BTCUSD", timeframe="1h", df=train_feats,
            feature_names=features.feature_names, data_version="v1",
        )
        val_features = FeaturesFrame(
            asset="BTCUSD", timeframe="1h", df=val_feats,
            feature_names=features.feature_names, data_version="v1",
        )
        h = Hypothesis(
            id="hyp_pipe",
            condition_tree=Condition(feature_ref="ma_20", operator=CompareOp.LT, value=105.0),
            amplitude=Amplitude(valeur=10.0, unité=AmplitudeUnit.PRICE_ABSOLU, direction_implicite=Direction.LONG),
            direction=Direction.LONG,
            universe=Universe(assets=("BTCUSD",), timeframes=("1h",)),
        )
        m_train, m_val, calibrated, m_holdout = self.engine.evaluate(
            h, train_ohlcv, train_features, val_ohlcv, val_features,
        )
        self.assertIsNone(m_holdout)  # pas de holdout fourni
        self.assertGreater(m_train.n_signals + m_val.n_signals, 0)
        self.assertEqual(calibrated.n_window, m_train.n_window)

    def test_holdout_access_only_once(self):
        ohlcv, features = self._make_data(n=800)
        h = Hypothesis(
            id="hyp_holdout",
            condition_tree=Condition(feature_ref="ma_20", operator=CompareOp.LT, value=110.0),
            amplitude=Amplitude(valeur=10.0, unité=AmplitudeUnit.PRICE_ABSOLU, direction_implicite=Direction.LONG),
            direction=Direction.LONG,
            universe=Universe(assets=("BTCUSD",), timeframes=("1h",)),
        )
        calibrated = self.engine.train_calibrate(h, ohlcv, features)
        # 1er accès OK.
        self.engine.test_on(h, ohlcv, features, calibrated, "holdout")
        # 2e accès : HoldoutAccessError.
        with self.assertRaises(HoldoutAccessError):
            self.engine.test_on(h, ohlcv, features, calibrated, "holdout")


if __name__ == "__main__":
    unittest.main()
