"""Tests des ecritures disque du LiveDataStore (throttle + flush).

polars ne sait pas appender un Parquet : chaque sauvegarde reecrit le fichier.
Les ecritures sont donc throttlees, avec `flush()` pour vider le tampon a
l'arret. Ces tests verrouillent ce compromis.
"""

from __future__ import annotations

import polars as pl

from einherjar.data.live_store import LiveDataStore


def _candle(i: int, prix: float = 100.0) -> dict[str, float]:
    """Bougie OHLCV + une colonne inconnue (doit etre ecartee par le store)."""
    return {
        "timestamp": 1_700_000_000_000 + i * 3_600_000,
        "open": prix + i,
        "high": prix + i + 1,
        "low": prix + i - 1,
        "close": prix + i + 0.5,
        "volume": 10.0 + i,
        "feature_inconnue": 42.0,
    }


def test_ecritures_throttlees_puis_flush(tmp_path):
    """La 1re bougie est ecrite tout de suite, les suivantes sont differees."""
    store = LiveDataStore(base_dir=tmp_path, window_size=100, write_throttle_s=999.0)
    for i in range(5):
        store.append("BTCUSD", "1h", _candle(i))

    fichiers = list(tmp_path.glob("*.parquet"))
    assert len(fichiers) == 1, "le couple doit exister des la premiere bougie"
    assert pl.read_parquet(fichiers[0]).height == 1, "les 4 bougies suivantes sont differees"
    assert store.get_window("BTCUSD", "1h").height == 5, "la memoire fait foi"

    ecrits = store.flush()
    assert ecrits == 1, "flush doit ecrire le couple en attente"
    sur_disque = pl.read_parquet(fichiers[0])
    assert sur_disque.height == 5
    assert set(sur_disque.columns) == {"timestamp", "open", "high", "low", "close", "volume"}


def test_flush_ne_reecrit_pas_un_couple_deja_ecrit(tmp_path):
    """Un deuxieme flush sans nouvelle bougie n'ecrit rien (pas de churn)."""
    store = LiveDataStore(base_dir=tmp_path, window_size=100, write_throttle_s=999.0)
    store.append("BTCUSD", "1h", _candle(0))
    assert store.flush() == 0, "la 1re ecriture est deja faite"

    store.append("BTCUSD", "1h", _candle(1))
    assert store.flush() == 1
    assert pl.read_parquet(next(iter(tmp_path.glob("*.parquet")))).height == 2


def test_throttle_zero_ecrit_immediatement(tmp_path):
    """write_throttle_s=0 : comportement historique (ecriture a chaque bougie)."""
    store = LiveDataStore(base_dir=tmp_path, window_size=100, write_throttle_s=0.0)
    store.append("BTCUSD", "1h", _candle(0))
    fichiers = list(tmp_path.glob("*.parquet"))
    assert len(fichiers) == 1
    assert pl.read_parquet(fichiers[0]).height == 1


def test_bulk_append_ecrit_le_bootstrap(tmp_path):
    """L'amorcage historique est ecrit tout de suite (pas de perte au demarrage)."""
    store = LiveDataStore(base_dir=tmp_path, window_size=100, write_throttle_s=999.0)
    store.bulk_append("BTCUSD", "1h", [_candle(i) for i in range(10)])
    fichiers = list(tmp_path.glob("*.parquet"))
    assert len(fichiers) == 1
    assert pl.read_parquet(fichiers[0]).height == 10


def test_bulk_append_accepte_un_dataframe(tmp_path):
    """Une DataFrame polars est acceptee comme une liste de dicts."""
    store = LiveDataStore(base_dir=tmp_path, window_size=100, write_throttle_s=999.0)
    df = pl.DataFrame([_candle(i) for i in range(4)])
    store.bulk_append("BTCUSD", "1h", df)
    assert store.get_window("BTCUSD", "1h").height == 4
    assert pl.read_parquet(next(iter(tmp_path.glob("*.parquet")))).height == 4


def test_fenetre_glissante_respectee(tmp_path):
    """Le store ne garde que `window_size` bougies."""
    store = LiveDataStore(base_dir=tmp_path, window_size=3, write_throttle_s=999.0)
    for i in range(6):
        store.append("BTCUSD", "1h", _candle(i))
    df = store.get_window("BTCUSD", "1h")
    assert df.height == 3
    assert df["timestamp"].to_list() == [1_700_000_000_000 + i * 3_600_000 for i in (3, 4, 5)]
