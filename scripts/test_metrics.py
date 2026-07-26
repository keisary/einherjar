import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backtest.metrics import compute_all
import numpy as np

# Test avec donnees simulees
returns = np.array([0.01, -0.005, 0.02, -0.01, 0.015, -0.003, 0.008])
equity = [1.0]
for r in returns:
    equity.append(equity[-1] * (1 + r))

ts = [i * 86400000 for i in range(len(returns))]  # 1 trade par jour

m = compute_all(returns, equity, ts)
print("=== Test metriques corrigees ===")
for k, v in m.items():
    print(f"  {k}: {v}")

# Verification manuelle
print("\n=== Verifications ===")
print(f"Total return compose attendu: {np.prod(1+returns)-1:.6f}")
print(f"Total return somme (ancien bug): {np.sum(returns):.6f}")
print(f"Sharpe raw (non annualise): {np.mean(returns)/np.std(returns, ddof=1):.4f}")
print(f"Sharpe annualise (252 trades/an): {np.mean(returns)/np.std(returns, ddof=1)*np.sqrt(252):.4f}")
print(f"Trades par mois: {len(returns) / (30.44):.2f}")
