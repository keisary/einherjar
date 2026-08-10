"""Diagnostic : distribution TP/SL vs plancher economique (NASDAQ100 15m reel).

Reutilise le loader exact du pipeline (discovery._load_real_data) : memes
donnees npy MIDAS + same splits. Calibre 30 hypotheses et compare les
distances calibrees au plancher economique (couts round-trip x0.08%).
"""
import sys
import numpy as np

sys.path.insert(0, "src")
from einherjar.research.config.loader import load_config
from einherjar.research.discovery import _load_real_data
from einherjar.research.engine.evaluator import EvaluationEngine, CalibrationError
from einherjar.research.utils.types import (
    Hypothesis, Condition, Amplitude, CompareOp, AmplitudeUnit, Direction, Universe,
)

config = load_config("src/einherjar/research/config")
train_ohlcv, train_feats, val_ohlcv, val_feats, hold_ohlcv, hold_feats, splits_key = _load_real_data(
    config,
    r"D:/midas_v2/midasV3/src/data/compiled",
    "NASDAQ100",
    "indices",
    "15m",
)
print(f"train bougies={train_ohlcv.n_bougies}, val={val_ohlcv.n_bougies}, hold={hold_ohlcv.n_bougies}")
print(f"features={len(train_feats.feature_names)}")

engine = EvaluationEngine(config=config, data_version="v1", seed=42)
close = np.asarray(train_ohlcv.df["close"].to_numpy(), dtype=float)
med_close = float(np.median(close))

rng = np.random.default_rng(7)
tp_dists, sl_dists, n_ok = [], [], 0
errs = {}
feat_pool = ["rsi_14", "sma_20", "ema_12"]
for i in range(30):
    ref = feat_pool[i % len(feat_pool)]
    op = CompareOp.GT if rng.random() < 0.5 else CompareOp.LT
    direc = Direction.LONG if rng.random() < 0.5 else Direction.SHORT
    h = Hypothesis(
        id=f"diag_{i}",
        condition_tree=Condition(feature_ref=ref, operator=op, value=float(med_close * rng.uniform(0.97, 1.03))),
        amplitude=Amplitude(valeur=float(rng.uniform(200.0, 2000.0)), unité=AmplitudeUnit.PRICE_ABSOLU, direction_implicite=direc),
        direction=direc,
        universe=Universe(assets=("NASDAQ100",), timeframes=("15m",)),
    )
    try:
        c = engine.train_calibrate(h, train_ohlcv, train_feats)
        tp_dists.append(c.tp_distance)
        sl_dists.append(c.sl_distance)
        n_ok += 1
    except CalibrationError as e:
        msg = str(e)[:100]
        errs[msg] = errs.get(msg, 0) + 1
    except Exception as e:  # noqa
        msg = f"{type(e).__name__}: {str(e)[:80]}"
        errs[msg] = errs.get(msg, 0) + 1

tp_dists = np.array(tp_dists)
sl_dists = np.array(sl_dists)
print(f"OK calibres={n_ok}/30")
for msg, c in sorted(errs.items(), key=lambda kv: -kv[1])[:6]:
    print(f"  x{c}: {msg}")
if n_ok:
    print(
        f"TP % : min={tp_dists.min()*100:.3f} med={np.median(tp_dists)*100:.3f} "
        f"p90={np.percentile(tp_dists, 90)*100:.3f} max={tp_dists.max()*100:.3f}"
    )
    print(
        f"SL % : min={sl_dists.min()*100:.3f} med={np.median(sl_dists)*100:.3f} "
        f"p90={np.percentile(sl_dists, 90)*100:.3f} max={sl_dists.max()*100:.3f}"
    )
    floor_tp = 0.0008 * 3.0
    floor_sl = 0.0008 * 2.0
    print(f"Plancher TP (couts x3) = {floor_tp*100:.3f}% : {int((tp_dists >= floor_tp).sum())}/{n_ok} au-dessus")
    print(f"Plancher SL (couts x2) = {floor_sl*100:.3f}% : {int((sl_dists >= floor_sl).sum())}/{n_ok} au-dessus")