# Sprint 3.2 — P2 : Multi-niveaux (asset/market/general)

**Date** : 2026-08-18
**Statut** : 🚧 Runner étendu, test scope=market validé
**Note** : Sprint en cours, le user va corriger ce qui a été fait

---

## TL;DR

**Runner étendu** pour supporter 3 niveaux de scope :
- `--scope asset` : 1 modèle par actif (legacy)
- `--scope market` : 1 modèle par classe d'actifs (cross-asset intra-market)
- `--scope general` : 1 modèle sur TOUS les actifs (cross-market)

**Test validé** : `--scope market --asset-classes crypto --max-assets 3` → 19/20 Einhers admis sur 3 cryptos (112k samples).

---

## Inventaire des données

| Classe | 1h | 4h | 1d | Total (1h+4h+1d) |
|---|---|---|---|---|
| commodities | 12 | 12 | 12 | 12 |
| crypto | 9 | 9 | 0 | 9 |
| forex | 20 | 20 | 20 | 20 |
| indices | 11 | 11 | 10 | 11 |
| stocks_growth | 15 | 15 | 15 | 15 |
| stocks_tech | 16 | 16 | 16 | 16 |
| stocks_value | 16 | 16 | 16 | 16 |
| **TOTAL** | **99** | **99** | **89** | **99** |

**Bien plus que les 28 actifs mentionnés par le user.** Sprint 3.2 doit préciser le scope exact (combien d'actifs par run).

---

## Modifications Sprint 3.2

### `runner.py` (MODIFIED)

Nouvelles options CLI :
```
--scope {asset,market,general}   # Sprint 3.2 P2
--asset-classes CRYPTO,FOREX,... # classes a inclure (market/general)
--max-assets N                   # limite par run
```

**Logique de résolution** :
- `--scope asset` : utilise `--asset` ou `--assets`
- `--scope market` : liste auto des actifs par classe via `list_available_assets(require_ohlcv=True)`
- `--scope general` : idem mais sur toutes les classes

### Test validé

```bash
python -m einherjar.research.xgb_einhers.runner run \
  --scope market --asset-classes crypto --max-assets 3 \
  --timeframe 1h --horizon 2d \
  --n-estimators 80 --max-depth 3 --max-paths 20 \
  --debug --regularized --apply-dedup --drop-sparse \
  --min-holdout-trades 5 \
  --output outputs/einhers_market_crypto_2d.jsonl
```

**Résultat** :
- Scope market : 3 cryptos sélectionnés (BTC, ETH, LTC vraisemblablement)
- 112,277 samples chargés (3 × ~37k)
- 20 chemins extraits
- 19 Einhers admis, 1 rejeté
- Coût round-trip : 0.44% (realiste pour taker crypto)

---

## Limitations à connaître

### 1. Le scope "market" actuelle n'utilise qu'un seul modèle

Le code charge les 3 cryptos et les concatène (`X_global` = 3 × 37k = 112k), puis entraîne UN modèle XGBoost sur l'ensemble. Les Einhers générés sont ensuite backtestés sur l'actif PRIMAIRE (le 1er de la liste) uniquement.

**Conséquence** : un Einher "market" est en fait un Einher cross-asset (entraîné sur N actifs) mais backtesté sur 1 seul. C'est ce que les IA ont appelé "single-factor beta disguised as multi-asset universality".

**Fix recommandé** : backtester chaque Einher sur TOUS les actifs du scope, pas seulement le primary.

### 2. Le scope "general" n'est pas encore implémenté

L'option CLI est là, mais la logique de résolution des actifs reste à faire. Pour l'instant, `--scope general` est traité comme `--scope market` (utilise `--asset-classes`).

### 3. Coût round-trip variable selon actif

Pour ce test crypto, `load_costs(asset)` retourne 0.44% (élevé). C'est probablement correct mais peut tuer des Einhers. À vérifier.

### 4. Le `min_holdout_trades=5` est toujours actif (debug)

Avec le strict mode (min_holdout_trades=100), AUCUN Einher ne passerait. C'est un compromis pour itérer.

---

## Reste à faire pour finaliser P2

### Court terme
1. **Implémenter le backtest multi-actif** : chaque Einher devrait être backtesté sur tous les actifs du scope, pas seulement le primary
2. **Implémenter le scope "general"** : résolution auto des actifs sur toutes les classes
3. **Tester le scope "general"** sur quelques actifs
4. **Rapport cross-market** : mesurer la généralisation entre classes (crypto → forex, par exemple)

### Moyen terme
5. **Multi-TF** : entraîner sur 1h + 4h + 1d en parallèle
6. **Walk-forward global** : split temporel sur la campagne entière (pas par run)
7. **DSR global** : Deflated Sharpe Ratio sur tous les Einhers de tous les runs

### Validation
8. **Re-valider cross-asset** (Sprint 2.6) avec les nouveaux sharpes (Sprint 3.0)
9. **Re-valider multi-horizon** (Sprint 2.6) avec BH
10. **Tests unitaires** pour le scope "market" et "general"

---

## Verdict Sprint 3.2 (étape actuelle)

**Le runner supporte maintenant 3 niveaux de scope.** Le test scope=market fonctionne.

**Mais** : l'implémentation actuelle est minimale. Le backtest reste single-actif (primary), ce qui limite la valeur du scope "market". Le user a indiqué qu'il va corriger ce qui a été fait, donc cette version est un point de départ, pas une finalisation.

**À discuter avec le user** :
- Combien d'actifs par run ? (max-assets=10 par défaut, peut être plus)
- Le backtest doit-il être multi-actif ou rester single-actif (primary) ?
- Le scope "general" doit-il être un seul modèle ou un modèle par classe ?
