# Sprint 3.3 — Correction des bugs critiques

**Date** : 2026-08-18
**Statut** : ✅ 5/7 bugs corrigés + 1 à faire en Sprint 3.4 (BUG-03)
**Suite** : 81/81 tests verts après corrections

---

## TL;DR

L'analyse externe a identifié **7 bugs** (5 critiques + 2 majeurs). Après vérification :
- **5 bugs corrigés** dans ce sprint
- **1 bug à corriger** en Sprint 3.4 (BUG-03 multi-actif leakage, gros refactoring)
- **1 bug non retenu** comme bloquant (BUG-07 dedup, maintenant corrigé)

Tous les tests passent. Le système est plus robuste mais **le mode multi-actif reste à valider** (Sprint 3.4).

---

## Vérification des 7 bugs

| ID | Gravité | Vérifié | Statut |
|---|---|---|---|
| **BUG-01** | CRITIQUE | ✅ max_dd=-0.95 → passes_admission=True (confirmé) | ✅ FIXÉ |
| **BUG-02** | CRITIQUE | ✅ p_value(Sharpe=1.0)=0.317, BH rejette 30/30 (confirmé) | ✅ FIXÉ |
| **BUG-03** | CRITIQUE | ✅ np.concatenate empile actifs puis split → leakage | ⚠️ À FAIRE Sprint 3.4 |
| **BUG-04** | CRITIQUE | ✅ Branche `else` (multi=full) dans runner.py | ✅ FIXÉ |
| **BUG-05** | MAJEURE | ✅ val_start/val_end sans embargo dans backtest | ✅ FIXÉ |
| **BUG-06** | MAJEURE | ✅ Pas de check in_position, stacking possible | ✅ FIXÉ |
| **BUG-07** | MAJEURE | ✅ Dedup avec importances uniformes = arbitraire | ✅ FIXÉ |

---

## Détail des corrections

### ✅ BUG-01 : max_drawdown signe

**Problème** : `max_dd = min(eq - peak) ≤ 0`, mais `passes_admission` testait `if self.max_drawdown > 0.30` → toujours False (négatif < positif).

**Fix** (`types.py:125-126`) :
```python
# AVANT
if self.max_drawdown > max_drawdown:  # -0.95 > 0.30 = False (bug)
# APRES
if abs(self.max_drawdown) > max_drawdown:  # abs(-0.95) = 0.95 > 0.30 = True
```

**Vérification** :
```
DD=-0.95 (=95%) : passed=False [OK]  ← avant : passed=True
DD=-0.50 (=50%) : passed=False [OK]
DD=-0.30 (=30%) : passed=True [OK]  (limite)
DD=-0.10 (=10%) : passed=True [OK]
```

### ✅ BUG-02 : BH fallback erroné

**Problème** : `apply_bh_to_einhers` utilisait `pvalue = 2*(1-Phi(|sharpe|))`. Pour sharpe=1.0, pvalue=0.317 → BH rejette 100% des candidats. Fallback catastrophique.

**Fix** : ajout de `t_statistic` et `p_value` dans `EinherMetrics`, calculés dans `compute_metrics` (vraie t-stat sur rendements de trades).

**Fix 1** (`types.py`) : ajout champs
```python
@dataclass(frozen=True)
class EinherMetrics:
    # ... existing fields ...
    t_statistic: float = 0.0  # t = mean(rets) / (std(rets) / sqrt(n))
    p_value: float = 1.0      # p-value bilaterale H0: mean(rets) = 0
    trade_returns: tuple[float, ...] = field(default_factory=tuple)
```

**Fix 2** (`backtester.py`) : calcul t-stat + p-value
```python
if n > 1 and std > 0:
    t_stat = float(avg_net / (std / np.sqrt(n)))
    from math import erf, sqrt
    p_val = 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(t_stat) / sqrt(2.0))))
    p_val = max(p_val, 1e-10)
```

**Fix 3** (`multiple_testing.py`) : utilise la vraie p-value
```python
p_val = getattr(e.metrics, "p_value", None)
if p_val is None or p_val == 0:
    # Fallback de securite (devrait plus jamais arriver)
    ...
else:
    pvalue = p_val
```

**Vérification** :
```
30 Einhers p_value=0.003 (t=3.0, n=100) : rejected=0/30 [OK]
```

### ⚠️ BUG-03 : Multi-actif leakage (NON FIXÉ)

**Problème** : `load_multi_asset` empile les actifs `[X_A1, X_A2, X_A3]`, puis `temporal_split` coupe par index. Résultat : le train peut contenir le futur d'un actif, et le val peut contenir le passé d'un autre actif.

**Exemple chiffré** :
- 3 actifs × 100k bougies concaténés = 300k samples
- Split 60/20/20 : train=[0, 180k), val=[180k, 240k), holdout=[240k, 300k)
- Asset 1 (0-100k) est en totalité dans le train
- Asset 2 (100k-200k) est entre train (100k) et val (180k) et val (200k)
- Asset 3 (200k-300k) est en val (220k) et holdout (300k)
- Le modèle entraîné voit 100% de l'Asset 1 (incluant 2024) et la moitié de l'Asset 2

**Status** : non corrigé dans Sprint 3.3 (trop invasif, nécessite refonte de `load_multi_asset` + `temporal_split` + `runner`).

**Fix prévu Sprint 3.4** :
1. `load_multi_asset` : retourner des splits par actif (train_dict, val_dict, holdout_dict)
2. `runner` : concat les splits séparément (X_train_global = concat(X_train_i for i in assets))
3. Vérifier l'invariant : `max(timestamp(X_train_global)) < min(timestamp(X_val_global))`

### ✅ BUG-04 : val=full en multi-actif

**Problème** : branche `else` dans `runner.py:305-312` faisait `backtest_einher(ohlcv_aligned, X_aligned)` = 100% de l'historique en multi.

**Fix** (`runner.py:280-313`) : slicing uniforme val + holdout appliqué à tous les modes.
```python
# Branche else SUPPRIMEE
n_aligned = X_aligned.shape[0]
if n_aligned > 0:
    backtest_embargo = max(50, horizon_bars)  # FIX BUG-05
    val_start = int(n_aligned * 0.6) + backtest_embargo
    val_end = int(n_aligned * 0.8)
    holdout_start = int(n_aligned * 0.8) + backtest_embargo
    # Slicing val + holdout applique a TOUS les modes
    val_result = backtest_einher(ohlcv_aligned[val_start:val_end], ...)
    if holdout_start < n_aligned:
        holdout_result = backtest_einher(ohlcv_aligned[holdout_start:], ...)
```

### ✅ BUG-05 : no embargo en backtest

**Problème** : `temporal_split` applique `embargo = max(50, horizon_bars)` mais le backtest dans le runner ne le fait pas.

**Fix** (dans BUG-04 ci-dessus) : `backtest_embargo = max(50, horizon_bars)` ajouté.

### ✅ BUG-06 : Trade stacking

**Problème** : si un signal reste vrai pendant 10 bougies consécutives, 10 positions sont ouvertes en parallèle → levier implicite.

**Fix** (`backtester.py:286-291`) :
```python
# Sprint 3.3 FIX BUG-06 : tracker in_position
in_position_until_idx = -1
for sig_idx in signal_indices:
    entry_idx = sig_idx + 1
    if entry_idx >= n:
        break
    if entry_idx <= in_position_until_idx:
        continue  # Skip si deja en position
    ...
    in_position_until_idx = exit_idx
```

### ✅ BUG-07 : Dedup aveugle

**Problème** : `runner.py` passait `importances = {name: 1.0 for name in feature_names}` au dedup. En cas d'égalité (1.0 == 1.0), drop arbitraire.

**Fix** (`runner.py:_quick_importances()`) : pré-train XGBoost (30 estimators) avant le dedup pour avoir les vraies importances.
```python
def _quick_importances(X, feature_names, Y_ret, horizon_idx, valid_mask, n_estimators=30):
    """Pre-train rapide pour vraies importances."""
    target = Y_ret[valid_mask, horizon_idx].astype(np.float32)
    X_valid = X[valid_mask]
    n = X_valid.shape[0]
    split = int(n * 0.8)
    config = GBDTConfig(n_estimators=n_estimators, max_depth=4, learning_rate=0.1, random_state=42)
    model, backend = train_gbdt(X_valid[:split], target[:split], X_valid[split:], target[split:], config)
    return feature_importance(model, backend, feature_names)
```

---

## Tests après corrections

**81/81 tests OK, 0 fail, 3 skipped** (3 skipped = holdout legacy + scope market qui skippe quand les fichiers existent).

```
Ran 81 tests in 402.936s
OK (skipped=3)
```

---

## Bilan global

| Catégorie | Avant Sprint 3.3 | Après Sprint 3.3 |
|---|---|---|
| Bugs critiques identifiés | 5 | 0 corrigés (4 sur 5) |
| Bugs majeurs identifiés | 3 | 0 corrigés (2 sur 3) |
| Bug BUG-03 (multi-actif) | Open | **Sprint 3.4** |
| Tests verts | 81/81 | 81/81 |

**Statut** : 5/7 bugs corrigés. Le système est plus robuste mais le mode multi-actif reste invalidé jusqu'au Sprint 3.4 (BUG-03).

**Recommandation** : ne PAS utiliser le mode multi-actif (`--assets`, `--scope market`, `--scope general`) en production tant que BUG-03 n'est pas corrigé. Utiliser le mode single-asset uniquement.

**Sprint 3.4 planifié** : refactoring `load_multi_asset` pour split par actif + concat des splits séparés + invariant temporel strict.
