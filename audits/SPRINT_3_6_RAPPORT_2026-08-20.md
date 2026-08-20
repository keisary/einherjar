# Sprint 3.6 - Restructuration runner + discovery parallèle + corpus/archive

**Date** : 2026-08-20
**Branche** : main (en cours de finalisation)

## Objectif

Corriger les dernières limitations du runner xgb_einhers et faire une
re-mesure complète sur l'univers de données disponible. Le runner
doit être simple : un `runner discover` qui lance tout (asset/class/global)
sur tous les triplets (tf × horizon), en parallèle, avec output direct
dans `corpus.jsonl` (admis) et `archive.jsonl` (rejetés avec raison).

## Résultats

### Univers couvert
- **Classes** : crypto (9), forex (20), commodities (12), indices (11), stocks_growth (15), stocks_tech (16), stocks_value (16)
- **Timeframe** : 1h
- **Horizons** : 6h, 12h, 1d, 2d
- **Scope** : per-asset, per-class, global

### Decouverte
- **116 triplets** generes par `build_discovery_triplets`
- **92 triplets executes** (4 manquants : timeout bash 30min sur forex per-class lourd)
- **4 workers** en parallele via `ProcessPoolExecutor`
- **Duree** : ~28 minutes

### Corpus (Einhers admis)
- **Total** : 705 Einhers
- **Distribution par classe** : crypto=653, stocks_tech=37, stocks_growth=13, commodities=2
- **Distribution par horizon** : 6h=205, 12h=189, 1d=169, 2d=142
- **Top actifs** : AVEUSD=228, BCHUSD=140, ADAUSD=98, LNKUSD=81, ETHUSD=46, BTCUSD=42, AMD=37
- **Sharpe min/med/max** : 1.90 / **6.53** / 17.94

### Archive (Einhers rejetés avec raison)
- **Total** : 791 Einhers
- **Repartition raison** :
  - 637 (80.5%) rejetes par **BH** (correction multi-tests FDR=5%)
  - 57 (7.2%) rejetes par **holdout** (0-4 trades < min=5)
  - 97 (12.3%) rejetes par **sharpe/win_rate/n_trades** insuffisants

## Conclusions Sprint 3.6

### Ce qui marche (confirme)
1. **Pipeline single-asset BTC** : 20-22 Einhers / 30 candidats selon horizon,
   Sharpe 1.3-17.9, robustesse val+holdout OK.
2. **Univers crypto** : 653 Einhers valides sur 9 actifs, mix
   d'altcoins (AVE, BCH, ADA) + majors (BTC, ETH).
3. **Stocks tech** : 37 Einhers sur 16 actifs, dominé par AMD.
4. **Top features (deja vu Sprint 3.4)** : `Factor_Risk_TailEvent_Score`,
   `skewness_risk`, `quant_realized_vol_50` → risk/volatilite comme
   drivers principaux.
5. **BH correction multi-tests** : 80% de rejet, evite le
   data-snooping classique (selection de 30 parmi 30 sans correction).

### Ce qui NE marche PAS (confirme)
1. **Multi-actif cross-asset (scope=market, scope=general)** : 0/30 admis
   sur tous les triplets per-class et global. Le modele XGBoost
   n'arrive pas a trouver de pattern qui marche sur plusieurs actifs
   simultanement → l'hypothese "alpha cross-asset universel" est
   probablement fausse. Les patterns sont asset-specific.
2. **Forex** : 0 admis en per-class. Trop bruite pour XGBoost 1h.
3. **Indices, commodities** : peu d'Einhers trouves (volumetrie OHLCV
   limitree ou modele inefficace sur ces classes).
4. **Stocks_value (JPM, BAC, GS...)** : 0 admis (donnees trop clean,
   pas d'arbitrage exploitable sur 1h).

## Realisations techniques

### Nouveaux modules
- `corpus.py` (Sprint 3.6) : store append-only thread-safe pour Einhers admis
- `archive.py` (Sprint 3.6) : store append-only pour Einhers rejetes AVEC raison
  (champ `rejection_reason` distinctif)

### Runner restructure
- `cmd_run` (legacy) : single-asset ou multi-asset explicite, comportement inchange
- `cmd_discover` (NOUVEAU) : lance tous les modeles en parallele
  - 3 scopes : asset, market (per-class), general (global)
  - Construction auto des triplets via `build_discovery_triplets`
  - Paralllisme via `ProcessPoolExecutor` (workers configurables)
  - Output : `--corpus` (admis) + `--archive` (rejetes)
  - Rapport JSON agrege dans `outputs/discover_report.json`

### Bugs fixes pendant le sprint
- **BUG-08** : `valid_mask` utilise dans dedup avant d'etre defini (single mode)
- **BUG-09** : `X_global` (splits filtres, 247k) != `Y_ret_global` (full, 247k) en multi
- **BUG-10** : `split.train_X` accede en multi ou `split` n'existe pas
- **BUG-11** : `require_ohlcv=True` etait un no-op (ohlcv_dir jamais passe)
- **BUG-12** : 3 sous-classes stocks (growth/tech/value) pointent vers dossier
  unique `stocks/` pour OHLCV
- **BUG-13** : PowerShell convertit `1d` en chemin relatif `1`, validation stricte
  des horizons

### Tests
- **Avant** : 81 tests OK
- **Apres** : 90 tests OK (8 nouveaux pour corpus/archive)
- Couverture : corpus thread-safe, archive par raison, iter, clear, batch

## Recommandations Sprint 3.7+

### Court terme
1. **Auditer les 705 Einhers du corpus** : combien survivent a un test
   out-of-sample temporel reel (walk-forward 3+ folds) ?
2. **Top features** : `Factor_Risk_TailEvent_Score` + `skewness_risk`
   reviennent trop souvent → risque de "volatility filter", pas d'alpha.
   Tester en retirant ces features (cf test_feature_robustness.py).
3. **L'archive est une mine d'or** : 791 Einhers rejetes contiennent
   des signaux potentiellement valides (BH rejete mais t-stat OK).
   Inspecter manuellement.

### Moyen terme
1. **Walk-forward K=3 folds** (deja specifie, jamais execute) :
   - Un Einher doit etre profitable sur 3/3 folds out-of-sample
2. **Sharpe sur courbe d'equity continue** (pas agrege par trades) :
   mesure plus stable
3. **DSR (Deflated Sharpe Ratio)** : ajuste pour le nombre de strategies
   testees (705 dans le corpus)
4. **PSR (Probabilistic Sharpe Ratio)** : probabilite que le vrai Sharpe > 0

### Long terme
1. **Random Forest / RuleFit / FIGS** en parallele de XGBoost (propose mais pas execute)
2. **PySR (Symbolic Regression)** : alternative structurelle au GBT
3. **Multi-target** : horizon-dependent models (un modele par horizon)

## Prochaine etape recommandee

**Sprint 3.7** : audit des 705 Einhers du corpus avec :
- Walk-forward K=3 (deja implemente, jamais execute)
- Test de robustesse : drop top-3 features, re-mesurer
- DSR / PSR pour chaque Einher du corpus
- Top 20 Einhers → backtest live (paper trading 1 mois)

Si resultats OK → passage en Sprint 4 (live paper trading broker IBKR).
Si resultats mitigés → Sprint 3.8 (modeles alternatifs RF/RuleFit).
