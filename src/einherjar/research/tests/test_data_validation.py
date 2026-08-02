"""Tests pour P1 #10 : data/validation.py."""
import math
import unittest
from datetime import datetime, timezone, timedelta

import numpy as np
import polars as pl

from einherjar.research.data.features import FeaturesFrame
from einherjar.research.data.ohlcv import OhlcvFrame
from einherjar.research.data.validation import (
    DataValidationError,
    validate_features,
    validate_no_leak,
    validate_ohlcv,
    validate_or_raise,
)


def _make_ohlcv(n=100, base_ts=None, with_nan=False, low_gt_high=False, non_monotonic=False):
    """Helper pour créer une OhlcvFrame."""
    base = base_ts or int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    rows = []
    for i in range(n):
        ts = base + i * 3600_000  # 1h
        if non_monotonic and i == n // 2:
            ts = ts - 7200_000  # recul de 2h
        o = 100.0 + i * 0.01
        h = o + 1.0
        l = o - 1.0
        if low_gt_high and i == n // 2:
            h, l = l, h  # low > high à mi-chemin
        c = o + 0.5
        v = 1000.0 + i
        if with_nan and i == 0:
            o = float("nan")
        rows.append((ts, o, h, l, c, v))
    df = pl.DataFrame({
        "timestamp": [r[0] for r in rows],
        "open": [r[1] for r in rows],
        "high": [r[2] for r in rows],
        "low": [r[3] for r in rows],
        "close": [r[4] for r in rows],
        "volume": [r[5] for r in rows],
    })
    return OhlcvFrame(asset="BTCUSD", timeframe="1h", df=df, data_version="v1")


class TestValidateOHLCV(unittest.TestCase):

    def test_clean_ohlcv_is_valid(self):
        ohlcv = _make_ohlcv(n=100)
        r = validate_ohlcv(ohlcv)
        self.assertTrue(r.is_valid)
        self.assertEqual(r.n_bougies, 100)
        self.assertTrue(r.index_monotonic)

    def test_nan_detected(self):
        ohlcv = _make_ohlcv(n=100, with_nan=True)
        r = validate_ohlcv(ohlcv)
        self.assertFalse(r.is_valid)
        self.assertTrue(any("NaN/inf" in e for e in r.errors))

    def test_low_gt_high_detected(self):
        ohlcv = _make_ohlcv(n=100, low_gt_high=True)
        r = validate_ohlcv(ohlcv)
        self.assertFalse(r.is_valid)
        self.assertTrue(any("low > high" in e for e in r.errors))

    def test_non_monotonic_detected(self):
        ohlcv = _make_ohlcv(n=100, non_monotonic=True)
        r = validate_ohlcv(ohlcv)
        self.assertFalse(r.is_valid)
        self.assertTrue(any("non monotone" in e for e in r.errors))


class TestValidateNoLeak(unittest.TestCase):

    def test_no_leak_valid(self):
        # train [0, 1000], val [2000, 3000], holdout [4000, 5000]
        r = validate_no_leak(
            train_end_ts=1000, val_start_ts=2000, val_end_ts=3000, holdout_start_ts=4000,
        )
        self.assertTrue(r.is_valid)

    def test_leak_train_val_detected(self):
        r = validate_no_leak(
            train_end_ts=1000, val_start_ts=500, val_end_ts=3000, holdout_start_ts=4000,
        )
        self.assertFalse(r.is_valid)
        self.assertTrue(any("train→val" in e for e in r.errors))

    def test_leak_val_holdout_detected(self):
        r = validate_no_leak(
            train_end_ts=1000, val_start_ts=2000, val_end_ts=3000, holdout_start_ts=2500,
        )
        self.assertFalse(r.is_valid)
        self.assertTrue(any("val→holdout" in e for e in r.errors))


class TestValidateOrRaise(unittest.TestCase):

    def test_raise_on_invalid(self):
        ohlcv = _make_ohlcv(n=10, with_nan=True)
        with self.assertRaises(DataValidationError):
            validate_or_raise(ohlcv, raise_on_error=True)

    def test_no_raise_on_valid(self):
        ohlcv = _make_ohlcv(n=100)
        # Should not raise.
        r = validate_or_raise(ohlcv, raise_on_error=True)
        self.assertTrue(r.is_valid)


if __name__ == "__main__":
    unittest.main()
