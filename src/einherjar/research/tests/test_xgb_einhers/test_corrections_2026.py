"""Tests des corrections 2026-08-21 (problèmes 1 à 5).

Couverture :
- asset_selection : 28 actifs depuis assets_v1.json, TF dispo (crypto sans 1d),
  horizons propres à chaque TF (lus depuis metadata).
- runner.parse_horizon : conversion horizon -> bars selon le TF réel.
- path_extractor : variantes logiques OR/NOT + extraction auto-min_score.
- condition_tree : path_to_ast construit les bons AST AND/OR/NOT/XOR.
"""
from __future__ import annotations

import sys

import pytest

sys.path.insert(0, "src")

from einherjar.research.xgb_einhers.asset_selection import (  # noqa: F401
    available_timeframes,
    horizons_for,
    load_asset_selection,
    resolve_compiled_class,
)
from einherjar.research.xgb_einhers.condition_tree import path_to_ast
from einherjar.research.xgb_einhers.path_extractor import (
    XGBPath,
    build_logical_variants,
)
from einherjar.research.xgb_einhers.runner import parse_horizon

# --------------------------------------------------------------------------- #
# Problème 1 : assets_v1.json, 28 actifs exacts
# --------------------------------------------------------------------------- #


def test_load_asset_selection_28():
    specs = load_asset_selection()
    assert len(specs) == 28
    # les 8 stocks + 5 crypto + 8 forex + 4 indices + 3 commodities
    names = {s.asset for s in specs}
    for expected in ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "TSLA", "JPM", "XOM"]:
        assert expected in names
    assert "BTCUSD" in names and "EURUSD" in names and "SP500" in names


def test_resolve_compiled_class():
    assert resolve_compiled_class("stocks_tech", "AAPL") == "stocks_tech"
    assert resolve_compiled_class("stocks_value", "JPM") == "stocks_value"
    assert resolve_compiled_class("stocks_growth", "XOM") == "stocks_growth"
    assert resolve_compiled_class("forex", "EURUSD") == "forex"
    assert resolve_compiled_class("crypto", "BTCUSD") == "crypto"


def test_crypto_has_no_1d():
    # Les cryptos n'ont pas de TF 1d dans les données compilées
    tfs = available_timeframes("BTCUSD", "crypto")
    assert "1d" not in tfs
    assert "5m" in tfs and "1h" in tfs


# --------------------------------------------------------------------------- #
# Problème 4 : horizons propres à chaque TF (lus depuis metadata, pas hardcode)
# --------------------------------------------------------------------------- #


def test_horizons_per_timeframe():
    assert horizons_for("BTCUSD", "crypto", "5m") == ["15m", "30m", "1h", "2h"]
    assert horizons_for("BTCUSD", "crypto", "1h") == ["6h", "12h", "1d", "2d"]
    assert horizons_for("AAPL", "stocks_tech", "1d") == ["5d", "10d", "20d", "60d"]


def test_parse_horizon_across_tf():
    # 6h sur 1h -> 6 bars ; 6h sur 5m -> 72 bars (correctif problème 4b)
    assert parse_horizon("6h", "1h") == 6
    assert parse_horizon("6h", "5m") == 72
    assert parse_horizon("1d", "4h") == 6
    assert parse_horizon("2d", "1h") == 48
    assert parse_horizon("30m", "5m") == 6
    assert parse_horizon("5d", "1d") == 5


# --------------------------------------------------------------------------- #
# Problème 3 : variantes logiques OR/NOT/XOR
# --------------------------------------------------------------------------- #


def test_build_logical_variants_or_not():
    base = [
        XGBPath(conditions=(("a", "<", 0.5), ("b", ">", 1.2)), score=0.01, tree_idx=0, path_idx=0),
        XGBPath(conditions=(("c", "<=", 2.0),), score=-0.02, tree_idx=1, path_idx=1),
    ]
    variants = build_logical_variants(base, top_n=2)
    ops = {v.logical_op for v in variants}
    assert "OR" in ops
    assert "NOT" in ops
    # Chaque variante doit produire un AST valide (pas d'exception)
    for v in variants:
        path_to_ast(v)


def test_path_to_ast_logical():
    # NOT sur 2 conditions -> AND(NOT(c1), c2) sans double négation
    p = XGBPath(
        conditions=(("a", "<", 0.5), ("b", ">", 1.2)),
        score=0.01, tree_idx=0, path_idx=0, logical_op="NOT",
    )
    ast = path_to_ast(p)
    d = ast.to_dict()
    assert d["op"] == "AND"
    assert d["left"]["op"] == "NOT"
    assert d["left"]["left"]["feature_ref"] == "a"


def test_parse_horizon_invalid_tf_raises():
    with pytest.raises(ValueError):
        parse_horizon("6h", "99z")
    with pytest.raises(ValueError):
        parse_horizon("zz", "1h")
