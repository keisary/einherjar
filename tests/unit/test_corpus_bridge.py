"""Tests unitaires du pont corpus de recherche -> systeme vivant."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from einherjar.core.models import Einher  # noqa: E402
from einherjar.signals.corpus_bridge import (  # noqa: E402
    CorpusBridgeError,
    corpus_feature_refs,
    entry_to_einher,
    load_entries,
    load_einhers,
    tree_to_expr,
    universe_index,
)

ENTRY = {
    "id": "or_NVDA_1d_60d_954d6c",
    "condition_tree": {
        "op": "OR",
        "left": {"feature_ref": "parabolic_sar", "operator": ">=", "value": 4.26380014},
        "right": {"feature_ref": "atr_21", "operator": "<", "value": 0.0211679507},
    },
    "direction": "BUY",
    "amplitude_bars": 60,
    "tp_pct": 0.2635616359616991,
    "sl_pct": 0.1581369815770195,
    "universe": {
        "asset": "NVDA",
        "asset_class": "stocks_tech",
        "timeframe": "1d",
        "horizon": "60d",
        "horizon_bars": 60,
    },
    "metrics": {"n_trades": 15, "win_rate": 0.733, "sharpe_ratio": 2.329, "total_return": 4.94},
    "scope": "asset",
    "source": {"model": "or_regimes"},
    "created_at": "2026-09-13T00:50:00",
}


def test_tree_to_expr_or():
    """Un arbre OR imbrique se convertit en expression parenthesee evaluable."""
    expr = tree_to_expr(ENTRY["condition_tree"])
    assert expr == "((parabolic_sar >= 4.26380014) or (atr_21 < 0.0211679507))"


def test_tree_to_expr_and_imbrique():
    """Un arbre AND/OR profond conserve la structure et les seuils litteraux."""
    tree = {
        "op": "AND",
        "left": {"feature_ref": "rsi_14", "operator": "<", "value": 30},
        "right": {
            "op": "OR",
            "left": {"feature_ref": "pattern_bull_flag", "operator": "==", "value": 1},
            "right": {"feature_ref": "obv_ema", "operator": ">", "value": 0.5},
        },
    }
    expr = tree_to_expr(tree)
    assert expr == "((rsi_14 < 30.0) and ((pattern_bull_flag == 1.0) or (obv_ema > 0.5)))"
    assert eval(expr, {"__builtins__": {}}, {"rsi_14": 20.0, "pattern_bull_flag": 0.0, "obv_ema": 0.9})


def test_tree_to_expr_operateur_inconnu():
    """Un operateur logique hors vocabulaire leve une erreur explicite."""
    with pytest.raises(CorpusBridgeError):
        tree_to_expr({"op": "XOR", "left": {"feature_ref": "a", "operator": ">", "value": 1}})


def test_entry_to_einher_mapping():
    """L'entree de recherche est convertie avec les champs attendus par EinherEngine."""
    einher = entry_to_einher(ENTRY)
    assert isinstance(einher, Einher)
    assert einher.name == "or_NVDA_1d_60d_954d6c"
    assert einher.direction == "long"
    assert einher.timeframes == ["1d"]
    assert einher.assets == ["NVDA"]
    assert einher.domain == "or_regimes"
    assert einher.max_holding == "60d"
    assert einher.tp_rule == {"type": "mfe_calibrated", "value": ENTRY["tp_pct"]}
    assert einher.sl_rule == {"type": "mae_calibrated", "value": ENTRY["sl_pct"]}
    assert einher.sharpe == pytest.approx(2.329)
    assert einher.trade_count == 15
    assert einher.profit_horizon == "60d"


def test_entry_to_einher_direction_short():
    """SELL devient la direction short du systeme vivant."""
    entry = {**ENTRY, "direction": "SELL"}
    assert entry_to_einher(entry).direction == "short"


def test_entry_to_einher_timeframe_hors_systeme():
    """Un timeframe hors systeme vivant (ex 2h) est refuse, pas converti en silence."""
    entry = {**ENTRY, "universe": {**ENTRY["universe"], "timeframe": "2h"}}
    with pytest.raises(CorpusBridgeError):
        entry_to_einher(entry)


def test_max_holding_conversion_minutes():
    """amplitude_bars est converti en duree reelle selon le timeframe."""
    entry = {**ENTRY, "amplitude_bars": 8, "universe": {**ENTRY["universe"], "timeframe": "15m"}}
    # 8 bougies de 15 min = 120 min = 2h
    assert entry_to_einher(entry).max_holding == "2h"


def test_load_entries_jsonl_tolerant(tmp_path):
    """Les lignes vides et les lignes JSON cassees sont ignorees, le reste est charge."""
    path = tmp_path / "corpus.jsonl"
    path.write_text(
        "\n".join([json.dumps(ENTRY), "", "r{corrompu", json.dumps({**ENTRY, "id": "x"})]),
        encoding="utf-8",
    )
    entries = load_entries(path)
    assert len(entries) == 2
    assert load_einhers(path)[0].name == ENTRY["id"]


def test_load_einhers_rejette_entrees_invalides(tmp_path):
    """Une entree invalide est ecartee sans casser le chargement des autres."""
    bad = {**ENTRY, "id": "bad", "universe": {**ENTRY["universe"], "timeframe": "3h"}}
    path = tmp_path / "corpus.jsonl"
    path.write_text("\n".join([json.dumps(ENTRY), json.dumps(bad)]), encoding="utf-8")
    einhers = load_einhers(path)
    assert [e.name for e in einhers] == [ENTRY["id"]]


def test_corpus_feature_refs_et_index(tmp_path):
    """Les feature_ref sont dedupliquees et l'index (asset, tf) est construit."""
    path = tmp_path / "corpus.jsonl"
    path.write_text(json.dumps(ENTRY), encoding="utf-8")
    assert corpus_feature_refs(path) == ["atr_21", "parabolic_sar"]
    assert universe_index(path) == {("NVDA", "1d"): [ENTRY["id"]]}


def test_corpus_reel_du_repo():
    """Le corpus live du repo se charge integralement (test d'integration leger)."""
    corpus = PROJECT_ROOT / "outputs" / "corpus.jsonl"
    if not corpus.exists():
        pytest.skip("corpus live absent")
    einhers = load_einhers(corpus)
    assert len(einhers) == len(load_entries(corpus))
    assert len(einhers) > 0
    assert {e.direction for e in einhers} <= {"long", "short"}
    assert len(corpus_feature_refs(corpus)) >= 70
