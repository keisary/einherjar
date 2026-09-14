"""Tests unitaires du pipeline de features (sans calcul lourd).

Le calcul complet (enrichisseurs MIDAS + patterns + facteurs) est couvert par
`scripts/compare_midas_vs_einherjar.py` et `scripts/smoke_inference.py` ; ici on
teste la logique d'orchestration, de troncature et de couverture.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from einherjar.signals.feature_pipeline import (  # noqa: E402
    PATTERN_COLUMN_PREFIX,
    FeaturePipeline,
    FeaturePipelineError,
    _pandas_to_polars,
)


def _ohlcv(n: int = 30) -> pl.DataFrame:
    """Construit un OHLCV minimal (n bougies) pour les tests d'orchestration."""
    close = np.linspace(100.0, 110.0, n)
    return pl.DataFrame({
        "timestamp": pl.datetime_range(
            __import__("datetime").datetime(2025, 1, 1),
            __import__("datetime").datetime(2025, 12, 31),
            interval="1h", eager=True,
        ).head(n),
        "open": close,
        "high": close + 0.5,
        "low": close - 0.5,
        "close": close,
        "volume": np.full(n, 1000.0),
    })


def test_ohlcv_incomplet_leve_une_erreur():
    """Un OHLCV sans colonne requise est refuse explicitement (pas de NaN silencieux)."""
    df = _ohlcv(10).drop("volume")
    with pytest.raises(FeaturePipelineError):
        FeaturePipeline().compute(df)


def test_compute_incremental_tronque_et_forwarde(monkeypatch):
    """compute_incremental ajoute la bougie, tronque a max_lookback et transmet asset/tf."""
    pipe = FeaturePipeline(max_lookback=20)
    captured: dict[str, object] = {}

    def fake_compute(df, *, asset="ASSET", timeframe="1h", with_factors=True):
        captured["height"] = df.height
        captured["last_close"] = float(df["close"][-1])
        captured["asset"] = asset
        captured["timeframe"] = timeframe
        return df

    monkeypatch.setattr(pipe, "compute", fake_compute)
    historique = _ohlcv(30)
    import datetime as dt

    candle = {
        "timestamp": historique["timestamp"][-1] + dt.timedelta(hours=1),
        "open": 200.0, "high": 201.0, "low": 199.0, "close": 200.5, "volume": 10.0,
    }
    out = pipe.compute_incremental(historique, candle, asset="BTCUSD", timeframe="15m")

    assert captured["height"] == 20, "la fenetre doit etre tronquee a max_lookback"
    assert captured["last_close"] == pytest.approx(200.5), "la nouvelle bougie doit etre en dernier"
    assert captured["asset"] == "BTCUSD"
    assert captured["timeframe"] == "15m"
    assert out.height == 20


def test_get_required_lookback_utilise_le_registre():
    """Les fenetres par feature proviennent de LOOKBACK_WINDOWS, motif `pattern_` gere."""
    pipe = FeaturePipeline(max_lookback=999)
    assert pipe.get_required_lookback("rsi_14") == 14
    assert pipe.get_required_lookback(f"{PATTERN_COLUMN_PREFIX}double_top") == 100
    assert pipe.get_required_lookback("feature_inconnue") == 999


def test_coverage_detecte_absentes_vides_et_valides():
    """coverage distingue colonne absente, colonne vide (NaN) et colonne exploitable."""
    n = 10
    df = pl.DataFrame({
        "rsi_14": np.linspace(20.0, 80.0, n),
        "atr_21": np.full(n, np.nan),
    })
    report = FeaturePipeline.coverage(df, ["rsi_14", "atr_21", "pattern_bull_flag"])
    assert report["presentes"] == ["rsi_14", "atr_21"]
    assert report["absentes"] == ["pattern_bull_flag"]
    assert report["colonnes_vides"] == ["atr_21"]
    assert report["ratio_valides"]["rsi_14"] == 1.0
    assert report["total"] == 3


def test_missing_refs():
    """missing_refs liste les features du corpus absentes du DataFrame enrichi."""
    df = pl.DataFrame({"rsi_14": [1.0, 2.0]})
    assert FeaturePipeline.missing_refs(df, ["rsi_14", "atr_21"]) == ["atr_21"]


def test_pandas_to_polars_dedoublonne_les_colonnes():
    """pl.from_pandas suffixe les colonnes dupliquees : on ne garde que la premiere."""
    import pandas as pd

    pdf = pd.DataFrame(np.ones((3, 3)), columns=["a", "b", "b"])
    out = _pandas_to_polars(pdf)
    assert out.columns == ["a", "b"]


def test_pandas_to_polars_accepte_deja_du_polars():
    """Un DataFrame polars est retourne tel quel (pas de conversion inutile)."""
    df = pl.DataFrame({"a": [1.0]})
    assert _pandas_to_polars(df) is df


def test_load_corpus_detecte_jsonl_et_json(tmp_path):
    """EinherEngine charge un corpus JSONL de recherche comme un JSON historique."""
    from einherjar.signals.einher_engine import EinherEngine

    jsonl = tmp_path / "corpus.jsonl"
    jsonl.write_text(json.dumps({
        "id": "e1", "condition_tree": {"feature_ref": "rsi_14", "operator": "<", "value": 30.0},
        "direction": "BUY", "amplitude_bars": 6, "tp_pct": 0.02, "sl_pct": 0.01,
        "universe": {"asset": "BTCUSD", "timeframe": "1h", "horizon": "6h"},
        "metrics": {"sharpe_ratio": 2.1, "win_rate": 0.7, "n_trades": 50},
        "source": {"model": "XGBRegressor"},
    }), encoding="utf-8")

    engine = EinherEngine()
    engine.load_corpus(str(jsonl))
    assert len(engine.einhers) == 1
    einher = engine.einhers[0]
    assert einher.trigger == "(rsi_14 < 30.0)"
    assert einher.direction == "long"
    assert einher.assets == ["BTCUSD"]

    legacy = tmp_path / "corpus_v2.json"
    legacy.write_text(json.dumps({"einhers": [{
        "name": "legacy", "domain": "pattern", "direction": "short",
        "timeframes": ["1h"], "trigger": "pattern_double_top == 1",
    }]}), encoding="utf-8")
    engine.load_corpus(str(legacy))
    assert [e.name for e in engine.einhers] == ["legacy"]
