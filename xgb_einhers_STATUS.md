# xgb_einhers — STATUS au 2026-08-17

## État

Le pipeline xgb_einhers est **fonctionnel end-to-end** mais ne produit **pas encore d'Einhers admis** sur les horizons testés (6h, 1d) pour BTCUSD × 1h.

## Ce qui a été fait

### Code créé (12 fichiers)
- `src/einherjar/research/xgb_einhers/__init__.py`
- `src/einherjar/research/xgb_einhers/types.py` (dataclasses)
- `src/einherjar/research/xgb_einhers/data_loader.py` (charge X, Y, OHLCV, aligne)
- `src/einherjar/research/xgb_einhers/label_engineer.py` (target supervisé)
- `src/einherjar/research/xgb_einhers/model.py` (double backend xgboost/sklearn avec fallback)
- `src/einherjar/research/xgb_einhers/path_extractor.py` (extraction arbres xgboost + sklearn)
- `src/einherjar/research/xgb_einhers/condition_tree.py` (AST de conditions, AND-only)
- `src/einjar/research/xgb_einhers/einher_builder.py` (construit Einher depuis XGBPath)
- `src/einherjar/research/xgb_einhers/backtester.py` (NOUVEAU moteur, simule trades, SL-first, P0)
- `src/einherjar/research/xgb_einhers/admission.py` (critères)
- `src/einherjar/research/xgb_einhers/einher_io.py` (JSONL)
- `src/einherjar/research/xgb_einhers/runner.py` (CLI)

### Tests créés (36 tests, 100% passants)
- `tests/test_xgb_einhers/test_data_loader.py` (19 tests, P0)
- `tests/test_xgb_einhers/test_backtester.py` (12 tests, P0 critiques)
- `tests/test_xgb_einhers/test_model.py` (5 tests)

## Découvertes clés (sanity check du 2026-08-17)

Données BTCUSD × 1h (69708 bougies, 2017-2025) :

| Horizon | mean Y_ret | std Y_ret | net_profitable (après 0.08% coûts) |
|---|---|---|---|
| 6h  | +0.0001 | 0.0145 | 32.2% |
| 12h | +0.0003 | 0.0170 | 43.3% |
| 1d  | +0.0004 | 0.0183 | 48.9% |
| 2d  | +0.0004 | 0.0189 | 51.0% |

**Conclusion** : à 6h et 1d, le marché BTC a un edge quasi nul (Y_ret moyen ~ 0).
Seuls 12h et 2d montrent un edge > 40%.

**Y_hor est variable par ligne** (1-10+ bars), pas constant par horizon. Découverte majeure qui invalide partiellement le plan initial ("amplitude FIXE par horizon XGBoost").

## Premier run : ça marche, mais 0 admission

Backend : **sklearn** (xgboost pas encore installé, en cours par l'utilisateur)
Hyperparamètres : 50 arbres, max_depth=4, learning_rate=0.05
Résultat R² val : **0.10** (modèle apprend quelque chose mais pas de signal fort)

| Métrique | BTC × 1h × 6h | BTC × 1h × 1d |
|---|---|---|
| N samples | 69702 | 69684 |
| R² val | 0.1023 | 0.0504 |
| Paths extraits | 50 | 30 |
| Trades par Einher | 1-6 (max 6) | 1-9 (max 9) |
| Admis | **0** | **0** |

**Tous les Einhers sont rejetés sur n_trades < 30**.

## Cause racine

Les arbres GBDT (depth 4, 50 arbres) produisent des **règles très spécifiques** qui se déclenchent rarement. Le marché BTC à ces horizons a un signal trop faible pour générer des règles fréquentes.

## Solutions à explorer (par ordre d'impact)

### 1. Multi-actif (priorité haute)
Entraîner sur les 28 actifs simultanément. Avec ~1.4M samples (au lieu de 70k), le modèle capte mieux les invariants cross-actifs.

**Gain attendu** : ×5-10 sur R² et nombre de trades par règle.

### 2. Tester sur 2d (où le signal est meilleur : 51% net_profitable)
**Gain attendu** : +50% d'admissions.

### 3. Hyperparamètres plus exploratoires
- max_depth=2 (règles plus générales)
- n_estimators=200 (plus de diversité)
- min_samples_leaf=50 (anti-surapprentissage)

**Gain attendu** : règles plus fréquentes mais moins précises.

### 4. Seuils d'admission plus souples temporairement (debug)
- min_trades=10 au lieu de 30
- min_sharpe=0.1 au lieu de 0.3

**But** : voir si le pipeline produit quelque chose, même imparfait.

### 5. Multi-TF dans le même modèle
Inclure 1h ET 15m ET 4h dans le même entraînement. Plus de signal de différents régimes.

**Gain attendu** : ×3-5 sur la quantité de données.

## Décisions à prendre par l'utilisateur

### A. Installer xgboost
```bash
pip install xgboost
```
xgboost est ~10× plus rapide que sklearn et a un API plus propre pour l'extraction d'arbres.

### B. Stratégie pour le premier run productif

Option 1 : **Multi-actif × 1h × 1d** (priorité signal)
- 28 actifs × 70k bougies = ~2M samples
- Horizon 1d où le signal est meilleur
- Risque : Einhers pas spécifiques à un actif

Option 2 : **Single-actif × 1h × 2d** (priorité précision)
- BTCUSD seulement
- Horizon 2d (51% net_profitable)
- Risque : peu de données

Option 3 : **5m timeframe** (priorité volume)
- Plus de bougies (~700k pour 1h → ~2.8M pour 5m)
- Risque : noise plus importante

### C. Backend pour le développement immédiat
- Tant que xgboost n'est pas installé, on peut utiliser sklearn (déjà testé)
- xgboost sera utilisé dès qu'installé (fallback automatique)

## Suite immédiate recommandée

1. Installer xgboost (5 min)
2. Réduire les seuils d'admission temporairement (debug) → debug.py
3. Tester multi-actif sur 1d (gain attendu le plus élevé)
4. Si toujours 0 admission : augmenter max_depth à 2 et n_estimators à 200
5. Documenter les findings au fur et à mesure
