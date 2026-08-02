"""Tests pour P1 #7 (corpus persistant) + P1 #8 (archive complete)."""
import unittest
import tempfile
from pathlib import Path

from einherjar.research.archive.schema import ArchiveEntry, RejectionReason
from einherjar.research.archive.store import _entry_from_dict
from einherjar.research.corpus.store import CorpusEntry, CorpusStore


class TestCorpusStore(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.path = Path(self.tmpdir.name) / "corpus.jsonl"

    def tearDown(self):
        self.tmpdir.cleanup()

    def _make_entry(self, id="einh_001", direction="long", sharpe=1.5):
        return CorpusEntry(
            id=id,
            hypothesis={"id": f"hyp_{id}", "condition_tree": {"feature_ref": "rsi_14", "operator": ">", "value": 70.0}},
            direction=direction,
            universe={"assets": ("BTCUSD",), "timeframes": ("1h",)},
            amplitude={"valeur": 0.02, "unite": "prix_absolu", "direction_implicite": direction},
            sl_n_atr=1.5, tp_n_atr=2.0, sl_distance=0.02, tp_distance=0.03, n_window=20,
            fingerprint_structurel=f"struct_{id}",
            fingerprint_comportemental=f"comp_{id}",
            sharpe_val=sharpe,
            ret_series=(0.01, 0.02, -0.005, 0.015),
            data_version="v1", seed=42, splits_hash="h123",
        )

    def test_append_and_load(self):
        store = CorpusStore(path=self.path)
        store.append(self._make_entry())
        store.append(self._make_entry(id="einh_002", sharpe=2.0))
        loaded = store.load()
        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[0].id, "einh_001")
        self.assertEqual(loaded[1].id, "einh_002")
        self.assertEqual(loaded[0].ret_series, (0.01, 0.02, -0.005, 0.015))

    def test_summary(self):
        store = CorpusStore(path=self.path)
        store.append(self._make_entry(id="e1", direction="long"))
        store.append(self._make_entry(id="e2", direction="short"))
        store.append(self._make_entry(id="e3", direction="long"))
        s = store.summary()
        self.assertEqual(s["n_total"], 3)
        self.assertEqual(s["n_long"], 2)
        self.assertEqual(s["n_short"], 1)
        self.assertAlmostEqual(s["n_long_frac"], 2 / 3)


class TestArchiveEntryRetSeries(unittest.TestCase):

    def test_ret_series_in_serialization(self):
        entry = ArchiveEntry(
            id="rej_001",
            type_élément="hypothesis",
            raison_rejet=RejectionReason.DSR_FAIL,
            date_rejet="2026-08-02T00:00:00+00:00",
            data_version="v1", seed=42, splits={}, costs_simulated={},
            sl_tp_source="from_train",
            fingerprint_structurel="f1",
            ret_series=(0.01, 0.02, -0.005),
        )
        d = entry.to_dict()
        self.assertEqual(d["ret_series"], [0.01, 0.02, -0.005])
        # Round-trip.
        e2 = _entry_from_dict(d)
        self.assertEqual(e2.ret_series, (0.01, 0.02, -0.005))


if __name__ == "__main__":
    unittest.main()
