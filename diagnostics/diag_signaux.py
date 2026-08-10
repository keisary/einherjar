"""Diagnostic : combien d'hypothèses RandomSearch ont réellement des signaux sur train/val (NASDAQ100 15m réel)."""
import sys, time, json
sys.path.insert(0, r"D:\midas_v2\einherjar\src")

from einherjar.research.config.loader import load_config
from einherjar.research.data.ohlcv import OhlcvProvider, OhlcvFrame
from einherjar.research.data.features import FeaturesFrame
from einherjar.research.data.npy_real_loader import load_features_from_npy
from einherjar.research.utils.time import make_splits_ratio


def slice_frame(fr, start, end):
    return type(fr)(
        asset=fr.asset, timeframe=fr.timeframe, df=fr.df.slice(start, end - start),
        **(dict(feature_names=fr.feature_names) if hasattr(fr, "feature_names") else {}),
        data_version=fr.data_version,
    )


def main():
    config = load_config(r"D:\midas_v2\einherjar\src\einherjar\research\config")
    print("config OK, features utilisables:", len(config.usable_feature_names))

    full_ohlcv = OhlcvProvider().load(asset="NASDAQ100", timeframe="15m", data_version="raw", asset_class="indices")
    full_features = load_features_from_npy(asset="NASDAQ100", asset_class="indices", timeframe="15m", config=config)
    common_ts = full_ohlcv.df.select("timestamp").join(full_features.df.select("timestamp"), on="timestamp", how="inner")
    full_ohlcv = OhlcvFrame("NASDAQ100", "15m", full_ohlcv.df.join(common_ts, on="timestamp", how="inner").sort("timestamp"), "pending")
    full_features = FeaturesFrame("NASDAQ100", "15m", full_features.df.join(common_ts, on="timestamp", how="inner").sort("timestamp"), full_features.feature_names, "pending")
    print("OHLCV bougies:", full_ohlcv.n_bougies, "| features bougies:", full_features.n_bougies)

    n = full_ohlcv.n_bougies
    splits = make_splits_ratio(n_total=n, horizon_label=50, embargo_bougies=0)
    train_end = splits.train.start + splits.train.length
    val_start = splits.val.start
    val_end = val_start + splits.val.length
    train_ohlcv = slice_frame(full_ohlcv, 0, train_end)
    train_features = slice_frame(full_features, 0, train_end)
    val_ohlcv = slice_frame(full_ohlcv, val_start, val_end)
    val_features = slice_frame(full_features, val_start, val_end)
    print("train:", train_ohlcv.n_bougies, "| val:", val_ohlcv.n_bougies)

    from einherjar.research.generators.algorithms import RandomSearchGenerator
    from einherjar.research.generators.protocol import make_protocol
    from einherjar.research.engine.evaluator import EvaluationEngine

    proto = make_protocol(config, data_version="test", seed=42, n_eval_budget=200, n_candidates=200)
    engine = EvaluationEngine(config=config, data_version="test", seed=42)
    gen = RandomSearchGenerator(proto, config, engine=engine)
    gen.bind_data(train_ohlcv, train_features, val_ohlcv, val_features)
    res = gen.generate()
    print("hypothèses générées:", res.n_generated)

    n_ok = n_neg = n_nan = 0
    sharpe_list = []
    n_signals_list = []
    t0 = time.time()
    for hyp in res.hypotheses[:50]:
        try:
            cal = engine.train_calibrate(hyp, train_ohlcv, train_features)
            m = engine.test_on(hyp, val_ohlcv, val_features, cal, "val")
            if m.n_signals >= 30:
                n_ok += 1
                sharpe_list.append(round(m.sharpe_net, 3))
                n_signals_list.append(m.n_signals)
            else:
                n_neg += 1
        except Exception:
            n_nan += 1
    dur = time.time() - t0
    print(f"50 hyp : {n_ok} >=30 trades val, {n_neg} <30 trades, {n_nan} echec calib, {dur:.0f}s")
    print(f"sharpe des OK: {sharpe_list[:10]}")
    print(f"nb trades des OK: {n_signals_list[:10]}")
    result = {"n_ok": n_ok, "n_neg": n_neg, "n_nan": n_nan, "duration_s": round(dur, 1), "sharpe_ok": sharpe_list[:10], "n_signals_ok": n_signals_list[:10]}
    with open(r"D:\midas_v2\einherjar\outputs\diag_signaux.json", "w", encoding="utf-8") as fp:
        json.dump(result, fp, indent=2)
    print("DIAG DONE")


if __name__ == "__main__":
    main()