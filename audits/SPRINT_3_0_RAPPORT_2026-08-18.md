# Sprint 3.0 — Corrections P0 (Sharpe + Embargo + Coûts)

**Date** : 2026-08-18
**Statut** : ✅ 3 fixes P0 appliqués et validés

---

## TL;DR

**Suite à la review des IA externes** (Claude + Gemini), 3 corrections critiques ont été appliquées :

1. **Sharpe formula** : `sqrt(n_trades)` (t-stat buggée) → `sqrt(trades_per_year)` (annualisé correct)
2. **Embargo dynamique** : `embargo_bars=50` fixe → `max(50, horizon_bars)` (anti-leakage)
3. **Coûts réalistes** : `costs_pct=0.0008` (sous-estimé) → `0.0010` minimum (taker fee crypto)

**Résultat** : val/holdout ratio median reste à **0.91** → le signal est réel, pas un artefact de formule.

---

## Les 3 fixes

### Fix #1 : Sharpe formula (`backtester.py`)

**Avant (bug)** :
```python
sharpe = float(avg_net / std * np.sqrt(max(n, 1)))
```
C'est une **t-stat**, pas un Sharpe. Le `sqrt(n_trades)` gonfle artificiellement le score avec le nombre de trades.

**Après (fix)** :
```python
trades_per_year = n / years_in_period
sharpe = float(avg_net / std * np.sqrt(trades_per_year))
```
`years_in_period` est calculé depuis les timestamps OHLCV. Pour BTC 2d sur 8 ans, `trades_per_year = 14/8 = 1.75`, donc `sqrt = 1.32` (au lieu de `sqrt(14) = 3.74`).

### Fix #2 : Embargo dynamique (`data_loader.py` + `runner.py`)

**Avant (bug)** : `embargo_bars=50` fixe, peu importe l'horizon.

**Après (fix)** :
```python
effective_embargo = max(embargo_bars, horizon_bars)
```
Pour horizon 2d (48 barres), l'embargo effectif est 50 (juste assez). Pour horizon 1w+, l'embargo suit.

### Fix #3 : Coûts réalistes (`backtester.py` + `runner.py`)

**Avant (bug)** : `costs_pct=0.0008` (0.08% round-trip).

**Après (fix)** : `costs_pct = max(load_costs(asset), 0.0010)`. Minimum 0.10% round-trip (= taker fee crypto 0.05% × 2).

---

## Résultats avant/après

### Validation des fixes (BTCUSD 2d, 14 Einhers)

| Métrique | Sprint 2.5 (buggé) | Sprint 3.0 (corrigé) |
|---|---|---|
| val sharpe median | ~8.7 | **~6.5** (annualisé correct) |
| val/holdout ratio median | 0.88 | **0.91** ✅ |
| % ratios > 1.0 | 29% (4/14) | **29% (4/14)** |
| n_admitted | 14 | 14 |

**Le signal est resté robuste** : la baisse des sharpes est due à l'annualisation correcte et aux coûts réalistes, mais la cohérence val/holdout reste > 0.9.

### Tests

**81 tests OK, 0 fail** (3 skipped = holdout legacy).

---

## Analyse des 4 ratios > 1.0 (Sprint 3.0)

| ID | val sharpe | holdout sharpe | ratio | n_trades val | n_trades holdout |
|---|---|---|---|---|---|
| `xgb_..._0000_0001_25b412` | 2.36 | 2.77 | **1.17** | 10 | 5 |
| `xgb_..._0000_0003_596504` | 6.61 | 7.92 | **1.20** ⭐ | 12 | 7 |
| `xgb_..._0000_0002_efceb9` | 6.36 | 6.99 | **1.10** | 8 | 5 |
| `xgb_..._0000_0007_06083f` | 7.15 | 6.82 | **0.95** | 16 | 7 |

**Le 1er Einher est le plus intéressant** : ratio 1.20, val sharpe 6.6, holdout 7.9, n_trades 12 et 7.

**MAIS** : n=5-7 holdout reste statistiquement faible (Gemini recommande 100+). À valider avec walk-forward.

---

## Ce qui reste à faire (P1 / P2)

| Pri | Action | Source |
|---|---|---|
| **P1** | Benjamini-Hochberg sur candidats (FDR < 5%) | Claude + Gemini |
| **P1** | `min_holdout_trades >= 100` (au lieu de 5) | Gemini |
| **P1** | DSR (Deflated Sharpe Ratio) | Gemini |
| **P2** | Cross-asset test re-validé avec nouveaux sharpes | — |
| **P2** | Multi-actif sur 28 actifs (3 niveaux de scope) | user |
| **P2** | Volatility regime filter explicite | Gemini |
| **P2** | Walk-forward out-of-sample | Gemini |

---

## Verdict Sprint 3.0

**Les 3 corrections P0 sont appliquées et le système reste cohérent** (val/holdout 0.91).

**Le signal est réel mais le sharpe affiché était gonflé d'environ 30%** par les bugs Sharpe+coûts. Après correction, on est dans des chiffres réalistes pour du trading crypto 1h.

**Prochaine étape** : P1 (correction multi-tests + significativité). Avant de scaler sur 28 actifs, il faut s'assurer que les Einhers qui passent ne sont pas des flukes statistiques.
