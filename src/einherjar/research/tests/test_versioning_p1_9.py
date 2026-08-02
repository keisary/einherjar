"""Tests pour P1 #9 : data/versioning.py enrichi."""
import unittest
import tempfile
from datetime import datetime, timezone as dt_timezone
from pathlib import Path

import numpy as np

from einherjar.research.config.loader import load_config
from einherjar.research.data.versioning import (
    DEFAULT_CLEANING_RULES,
    DEFAULT_TIMEZONE,
    make_data_version,
    make_splits_hash,
    _inspect_npy_file,
)


class TestInspectNpy(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.path = Path(self.tmpdir.name) / "test.npy"
        arr = np.array([[1.0, 2.0, 3.0]] * 10, dtype="float64")
        np.save(self.path, arr)
        self.ts_path = Path(self.tmpdir.name) / "test_ts.npy"
        ts = np.array([1000, 2000, 3000], dtype="int64")
        np.save(self.ts_path, ts)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_inspect_npy_data(self):
        info = _inspect_npy_file(self.path)
        self.assertEqual(info["format"], "npy")
        self.assertEqual(info["dtype"], "float64")
        self.assertEqual(info["n_bougies"], 10)
        self.assertEqual(info["shape"], [10, 3])
        self.assertIn("content_sha256", info)
        self.assertEqual(len(info["content_sha256"]), 64)

    def test_inspect_npy_timestamps(self):
        info = _inspect_npy_file(self.ts_path)
        self.assertEqual(info["start_ts_ms"], 1000)
        self.assertEqual(info["end_ts_ms"], 3000)
        self.assertEqual(info["n_bougies"], 3)


class TestMakeDataVersion(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.x_path = Path(self.tmpdir.name) / "BTCUSD_X.npy"
        self.ts_path = Path(self.tmpdir.name) / "BTCUSD_ts.npy"
        np.save(self.x_path, np.array([[1.0] * 246] * 50, dtype="float32"))
        np.save(self.ts_path, np.arange(1_700_000_000_000, 1_700_000_000_000 + 50 * 3_600_000, 3_600_000, dtype="int64"))

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_data_version_contains_schema(self):
        config = load_config("src/einherjar/research/config")
        dv = make_data_version(
            ohlcv_paths={"BTCUSD_1h": self.x_path, "BTCUSD_1h_ts": self.ts_path},
            features_paths={},
            config=config,
            tag="v1_test",
        )
        m = dv.manifest
        self.assertIn("schema", m)
        self.assertIn("timezone", m)
        self.assertEqual(m["timezone"], DEFAULT_TIMEZONE)
        self.assertIn("cleaning_rules", m)
        self.assertEqual(m["cleaning_rules"], DEFAULT_CLEANING_RULES)
        self.assertIn("content_sha256", m["schema"]["ohlcv"]["BTCUSD_1h"])
        self.assertIn("content_sha256", m["schema"]["ohlcv"]["BTCUSD_1h_ts"])
        ts_info = m["schema"]["ohlcv"]["BTCUSD_1h_ts"]
        self.assertIn("start_ts_ms", ts_info)
        self.assertIn("end_ts_ms", ts_info)
        self.assertIn("costs", m)
        self.assertIn("thresholds_hash", m)
        self.assertIn("evaluation_hash", m)
        self.assertEqual(dv.tag, "v1_test")
        self.assertEqual(len(dv.hash), 64)


class TestMakeSplitsHash(unittest.TestCase):

    def test_splits_hash_is_stable(self):
        h1 = make_splits_hash(0, 100, 200, 300, 400, 500, 1, 0)
        h2 = make_splits_hash(0, 100, 200, 300, 400, 500, 1, 0)
        self.assertEqual(h1, h2)
        h3 = make_splits_hash(0, 100, 200, 300, 400, 500, 1, 1)
        self.assertNotEqual(h1, h3)


if __name__ == "__main__":
    unittest.main()
