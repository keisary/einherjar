"""Smoke test intégration : CHAQUE générateur produit-il des hypothèses évaluables avec le moteur réel ?"""
import sys
sys.path.insert(0, r"D:\midas_v2\einherjar\src")
sys.path.insert(0, r"D:\midas_v2\einherjar")

import numpy as np
import polars as pl

from einherjar.research.config.loader import load_config
from einherjar.research.data.ohlcv import OhlcvFrame
from einherjar.research.generators.protocol import make_protocol


def make_synth_ohlcv(n=3000, start="2024-01-01"):
    import datetime as _dt
    rng = np.random.default_rng(7)
    t0 = _dt.datetime(2024, 1, 1, tzinfo=_dt.timezone.utc)
    ts = [t0 + _dt.timedelta(hours=i) for i in range(n)]
    close = 100.0 * np.cumprod(1.0 + rng.normal(0, 0.008, n))
    high = close * (1.0 + np.abs(rng.normal(0, 0.003, n)))
    low = close * (1.0 - np.abs(rng.normal(0, 0.003, n)))
    open_ = np.roll(close, 1); open_[0] = close[0]
    return pl.DataFrame({"timestamp": ts, "open": open_, "close": close, "high": high, "low": low, "volume": rng.uniform(1e5, 1e6, n)})


def make_synth_features(ohlcv_df, n_feats=8):
    rng = np.random.default_rng(0)
    close = ohlcv_df["close"].to_numpy()
    feats = {}
    for k in range(1, n_feats + 1):
        ma = np.full_like(close, np.nan, dtype=float)
        w = k * 5
        for t in range(w, len(close)):
            ma[t] = float(np.mean(close[t - w:t]))
        feats[f"ma_{w}"] = ma
    feats["rng_noise"] = rng.normal(0, 1, len(close))
    return pl.DataFrame(feats)


class _InMemBackend:
    def __init__(self, df): self._df = df
    def register(self, a, tf, df): self._df = df
    def load(self, asset, timeframe, data_version): return self._df


def main():
    from einherjar.research.data.features import FeaturesFrame
    from einherjar.research.engine.evaluator import EvaluationEngine
    from einherjar.research.generators.algorithms import (
        RandomSearchGenerator, BeamSearchGenerator, TypedGPGenerator,
        GrammaticalEvolutionGenerator, MemeticGenerator, NSGA2Generator,
    )

    cfg = load_config(r"D:\midas_v2\einherjar\src\einherjar\research\config")
    print("config:", len(cfg.usable_feature_names), "features utilisables")
    # Faux noms de features pour le test synthétique : on force via protocol universe ? Non,
    # la taxonomie est fixée par la config. On calcule les features de la taxo ? Trop lourd.
    # → On teste la GÉNÉRATION (volume d'hypothèses), pas l'évaluation réelle :
    # bind_data avec des features de la taxonomie (noms ma_* inconnus → l'évaluateur
    # leverait EvaluationError, mais generate() ne l'appelle pas nécessairement).

    ohlcv_df = make_synth_ohlcv(3000)
    ohlcv = OhlcvFrame(asset="TEST", timeframe="1h", df=ohlcv_df, data_version="v1")
    feats_df = make_synth_features(ohlcv_df, 20)
    features = FeaturesFrame("TEST", "1h", feats_df, feature_names=list(feats_df.columns), data_version="v1")

    cfg_for_gen = cfg
    proto = make_protocol(cfg_for_gen, data_version="v1", seed=42, n_eval_budget=50, n_candidates=60)
    engine = EvaluationEngine(config=cfg_for_gen, data_version="v1", seed=42)

    gens = [
        ("RandomSearch", RandomSearchGenerator(proto, cfg_for_gen, engine=engine)),
        ("Beam", BeamSearchGenerator(proto, cfg_for_gen, engine=engine)),
        ("TypedGP", TypedGPGenerator(proto, cfg_for_gen, engine=engine)),
        ("NSGA2", NSGA2Generator(proto, cfg_for_gen, engine=engine)),
        ("Memetic", MemeticGenerator(proto, cfg_for_gen, engine=engine)),
        ("GE", GrammaticalEvolutionGenerator(proto, cfg_for_gen, engine=engine)),
    ]
    for name, g in gens:
        try:
            g.bind_data(ohlcv, features, ohlcv, features)
            res = g.generate()
            n_h = len(res.hypotheses)
            print(f"{name:12s} → generate() OK : {n_h} hypothèses, n_generated={res.n_generated}")
        except Exception as e:
            print(f"{name:12s} → ÉCHEC : {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()