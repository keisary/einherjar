"""test_corpus_archive.py - Tests pour corpus.py et archive.py (Sprint 3.6)."""
import tempfile
import unittest
from pathlib import Path

from einherjar.research.xgb_einhers.archive import ArchiveEntry, ArchiveStore
from einherjar.research.xgb_einhers.corpus import CorpusStore
from einherjar.research.xgb_einhers.types import (
    Condition,
    Einher,
    EinherMetrics,
)


def make_einher(name: str = "test") -> Einher:
    """Helper pour creer un Einher minimal."""
    cond = Condition(feature_ref="x", operator="<", value=0.0)
    return Einher(
        id=f"einher_{name}",
        condition_tree=cond,
        direction="BUY",
        amplitude_bars=48,
        tp_pct=0.02,
        sl_pct=0.01,
        universe={"asset": "BTCUSD", "asset_class": "crypto", "timeframe": "1h", "horizon": "2d", "horizon_bars": 48},
        metrics=EinherMetrics(
            n_trades=10, n_tp=7, n_sl=3, n_timeout=0,
            win_rate=0.7, avg_net_return=0.01, total_return=0.1,
            sharpe_ratio=2.5, max_drawdown=-0.05, profit_factor=2.3,
            avg_holding_bars=10.0, buy_hold_return=0.0, alpha=0.1,
        ),
        scope="asset",
    )


class TestCorpusStore(unittest.TestCase):
    def test_add_and_count(self):
        with tempfile.TemporaryDirectory() as d:
            store = CorpusStore(Path(d) / "corpus.jsonl")
            self.assertEqual(store.count(), 0)
            store.add(make_einher("a"))
            store.add(make_einher("b"))
            self.assertEqual(store.count(), 2)

    def test_iter(self):
        with tempfile.TemporaryDirectory() as d:
            store = CorpusStore(Path(d) / "corpus.jsonl")
            e1 = make_einher("a")
            e2 = make_einher("b")
            store.add(e1)
            store.add(e2)
            ids = [e.id for e in store.iter()]
            self.assertEqual(ids, ["einher_a", "einher_b"])

    def test_thread_safe_concurrent(self):
        import threading
        with tempfile.TemporaryDirectory() as d:
            store = CorpusStore(Path(d) / "corpus.jsonl")
            def add_batch(prefix):
                for i in range(50):
                    store.add(make_einher(f"{prefix}_{i}"))
            t1 = threading.Thread(target=add_batch, args=("a",))
            t2 = threading.Thread(target=add_batch, args=("b",))
            t1.start()
            t2.start()
            t1.join()
            t2.join()
            self.assertEqual(store.count(), 100)

    def test_clear(self):
        with tempfile.TemporaryDirectory() as d:
            store = CorpusStore(Path(d) / "corpus.jsonl")
            store.add(make_einher("a"))
            store.clear()
            self.assertEqual(store.count(), 0)


class TestArchiveStore(unittest.TestCase):
    def test_add_and_count(self):
        with tempfile.TemporaryDirectory() as d:
            store = ArchiveStore(Path(d) / "archive.jsonl")
            self.assertEqual(store.count(), 0)
            e = make_einher("rej")
            store.add(e, rejection_reason="sharpe < 0", scope="asset", asset="BTCUSD",
                      asset_class="crypto", timeframe="1h", horizon="2d")
            self.assertEqual(store.count(), 1)

    def test_count_by_reason(self):
        with tempfile.TemporaryDirectory() as d:
            store = ArchiveStore(Path(d) / "archive.jsonl")
            e = make_einher("rej")
            store.add(e, "sharpe < 0", scope="asset")
            store.add(e, "BH rejected", scope="asset")
            store.add(e, "BH rejected", scope="asset")
            stats = store.count_by_reason()
            self.assertEqual(stats.get("sharpe < 0"), 1)
            self.assertEqual(stats.get("BH rejected"), 2)

    def test_iter_returns_entry(self):
        with tempfile.TemporaryDirectory() as d:
            store = ArchiveStore(Path(d) / "archive.jsonl")
            e = make_einher("rej")
            store.add(e, "min_trades", scope="market", asset="ETHUSD",
                      asset_class="crypto", timeframe="1h", horizon="2d")
            entries = list(store.iter())
            self.assertEqual(len(entries), 1)
            entry: ArchiveEntry = entries[0]
            self.assertEqual(entry.einher.id, "einher_rej")
            self.assertEqual(entry.rejection_reason, "min_trades")
            self.assertEqual(entry.scope, "market")
            self.assertEqual(entry.asset, "ETHUSD")
            self.assertNotEqual(entry.rejected_at, "")

    def test_add_batch(self):
        with tempfile.TemporaryDirectory() as d:
            store = ArchiveStore(Path(d) / "archive.jsonl")
            einhers = [make_einher(f"b_{i}") for i in range(5)]
            n = store.add_batch(einhers, rejection_reason="BH", scope="asset")
            self.assertEqual(n, 5)
            self.assertEqual(store.count(), 5)


if __name__ == "__main__":
    unittest.main()
