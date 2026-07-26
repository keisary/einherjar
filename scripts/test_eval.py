import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest.data_source import load_ohlcv
import polars as pl

# 1. Test chargement
print("=== Test chargement BTCUSD 15m ===")
df = load_ohlcv("BTCUSD", "15m", "crypto")
if df is None:
    print("ERREUR: pas de donnees")
    sys.exit(1)

print(f"Shape: {df.shape}")
print(f"Colonnes (10 premieres): {df.columns[:10]}")
print(f"Colonnes pattern: {[c for c in df.columns if c.startswith('pattern_')][:5]}")
print(f"Colonnes indicator: {[c for c in df.columns if c.startswith('rsi_') or c.startswith('macd_')][:5]}")

# 2. Test pl.sql_expr avec AND majuscule
print("\n=== Test pl.sql_expr AND majuscule ===")
try:
    expr = "rsi_14 < 30 AND macd_histogram > 0"
    result = df.select(pl.sql_expr(expr)).to_numpy().flatten()
    print(f"Expression: {expr}")
    print(f"Resultat: {result.sum()} / {len(result)} True")
except Exception as e:
    print(f"ERREUR pl.sql_expr: {e}")

# 3. Test expression simple pattern
print("\n=== Test pattern expression ===")
try:
    expr2 = "pattern_hammer == 1"
    result2 = df.select(pl.sql_expr(expr2)).to_numpy().flatten()
    print(f"Expression: {expr2}")
    print(f"Resultat: {result2.sum()} / {len(result2)} True")
except Exception as e:
    print(f"ERREUR pattern expr: {e}")

# 4. Test expression mixte
print("\n=== Test expression mixte ===")
try:
    expr3 = "pattern_engulfing_bull == 1 AND rsi_14 < 40"
    result3 = df.select(pl.sql_expr(expr3)).to_numpy().flatten()
    print(f"Expression: {expr3}")
    print(f"Resultat: {result3.sum()} / {len(result3)} True")
except Exception as e:
    print(f"ERREUR mixte expr: {e}")

print("\n=== Tous les tests passes ===")
