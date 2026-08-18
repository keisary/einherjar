"""tests/test_xgb_einhers/ - Tests unitaires du pipeline xgb_einhers.

Hiérarchie (P0 = critique) :
- P0 : data_loader (shape, exclusion OHLCV, alignement)
- P0 : backtester (no_lookahead, deterministic, known_signal)
- P1 : label_engineer, path_extractor, condition_tree, admission
- P2 : model XGBoost, einher_io, runner

Usage:
    cd D:/midas_v2/Einherjar
    $env:PYTHONPATH='src'
    python -m unittest discover -s src/einherjar/research/tests/test_xgb_einhers -p 'test_*.py'
"""
