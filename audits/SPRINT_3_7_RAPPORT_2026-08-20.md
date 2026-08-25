# Sprint 3.7 — Application des fix de la revue AI

**Date** : 2026-08-20
**Branche** : main
**Source** : audits/AI_REVIEW_2026-08-20.md

## Résumé exécutif

L'AI review a identifié 4 bugs P0 + 2 P1 + 2 P2. Tous ont été fixés
et testés. Le re-discover (run v4) sur 116 triplets a montré que :

- **Le bug P0-3 (multi-asset feature dim desync) était bien le smoking gun** :
  crypto market est passé de 0/30 à 11 admis (1d) et quelques admits sur 12h/2d.
- **Le bug P0-5 (admission check holdout) est plus strict** : corpus final
  252 (vs 705 avant) car les Einhers qui overfittent val sont rejetés.
- **Bug résiduel non détecté par l'AI** : `global` scope utilise
  `asset_class="global"` mais le dossier `compiled/global/1h` n'existe pas
  → 4 triplets en erreur. Bug à fixer dans Sprint 3.8.

## Fixes appliqués (P0 → P2)

| # | Sévérité | Fichier | Fix |
|---|---|---|---|
| P0-1 | Critique | `backtester.py` | TP/SL dynamique ATR-based au lieu de 2.5%/1.5% hardcodé |
| P0-2 | Critique | `runner.py` | Cost floor 0.10% crypto-only, 0.01% autres |
| P0-3 | Critique | `runner.py` | Slicer split_train_X/val_X/holdout_X avec keep_idx en multi |
| P0-4 | Critique | `runner.py` | valid_mask multi = Y_dir != -100 (au lieu de ones) |
| P0-5 | Critique | `admission.py` | Check holdout_metrics performance (Sharpe>0, wr>=0.36) |
| P1-1 | Haute | `backtester.py` | p-value one-sided upper tail (t<=epsilon → p=1.0) |
| P1-2 | Haute | `runner.py` | Aligner val/holdout slicing avec temporal_split |
| P2-1 | Design | `condition_tree.py` | `merge_paths_or()` pour DNF (OR) |
| P2-6 | Mineur | `runner.py` | Propager le vrai scope dans l'archive |

**Bonus user request** : `runner` sans subcommand = discover auto + workers=6.

## Résultats re-mesure (1h, 4 horizons, 116 triplets)

| Métrique | v1 (avant fix) | v4 (après fix) |
|---|---|---|
| Tests | 90/90 OK | **103/103 OK** |
| Triplets OK | 92/116 | **112/116** (4 erreurs global scope) |
| Corpus (admis) | 705 | **252** (plus strict, holdout check) |
| Archive (rejetés) | 791 | **714** |
| Sharpe min/med/max | 1.90 / 6.53 / **17.94** | **2.14 / 5.87 / 15.08** |
| Multi-actif (market) | **0/30 partout** | crypto 1d : **11 admis** (smoking gun confirmé) |
| Per-asset distrib | crypto 653, autres 0 | crypto 224, stocks_tech 18, stocks_growth 10 |

### Détail par classe (v4 corpus)

| Classe | Admis | Avant v4 |
|---|---|---|
| crypto | 224 | 653 (réduit à cause de holdout check) |
| stocks_tech | 18 | 37 |
| stocks_growth | 10 | 13 |
| stocks_value | 0 | 0 |
| forex | 0 | 0 |
| commodities | 0 | 2 |
| indices | 0 | 0 |

### Détail par raison d'archive (v4)

- **503** (70%) rejetés par **BH** (correction multi-tests)
- **140** (20%) rejetés par **Holdout** (performance dégradée)
- **71** (10%) rejetés par **sharpe/win_rate/n_trades**

## Améliorations confirmées

1. **P0-3 fix marche** : `crypto/1h/1d market` est passé de 0/30 à 11 admis.
   Le bug était bien que `split_train_X` (213 cols) n'était pas slicé
   alors que `feature_names` (120) l'était → XGBoost split sur les
   mauvais indices.
2. **P0-5 fix marche** : les Einhers avec Sharpe 5.0 val et -1.5 holdout
   sont maintenant rejetés.
3. **P1-1 fix marche** : p-value=1.0 pour t<=0 (one-sided upper tail).
4. **Sharpe plus raisonnable** : max 15.08 (vs 17.94), distribution
   resserrée.
5. **`runner` simple** : `python -m einherjar.research.xgb_einhers.runner`
   sans subcommand lance discover auto avec workers=6.

## Limites restantes (à fixer Sprint 3.8)

### Bug 1 — Global scope path
`build_discovery_triplets` met `asset_class="global"` pour le scope
general, mais `load_xy` cherche `D:/midas_v2/midasV3/src/data/compiled/global/1h`
qui n'existe pas. **4 triplets en erreur**.

**Fix** : pour le global scope, itérer sur toutes les classes réelles
et concat les assets. Ou faire un dossier `compiled/global/` qui
contient juste les paths (sans data). À voir.

### Bug 2 — OHLCV stocks pour les tests
Tu m'as corrigé : les OHLCV sont bien sous `stocks/` (corrigé dans
Sprint 3.6 via `resolve_ohlcv_class`). MAIS j'aurais dû vérifier
explicitement le chemin des données COMPILED pour les stocks
(`D:/midas_v2/midasV3/src/data/compiled/`) avant de lancer — il y a
peut-être un dossier `stocks_*/` qui n'a pas de X.npy mais existe
quand même. Je n'ai pas testé. **À vérifier Sprint 3.8**.

### Bug 3 — Forex/Indices/Commodities toujours 0
Avec le fix P0-1 (TP/SL dynamique ATR), on s'attendait à voir des
admis sur forex/indices. Mais c'est toujours 0. Hypothèses :

1. **Volumes trop faibles** : forex/indices 1h ont très peu de variation
   intra-bar → ATR très petit → TP/SL microscopiques → 0 trade généré.
2. **Modèle XGBoost inadapté** : sur 99 actifs différents, le modèle
   overfit les classes dominantes (crypto) et ne trouve rien sur les
   autres.
3. **Multi-actif P0-3 pas suffisant** : il reste peut-être un bug
   spécifique à ces classes (volatilité faible, features non-stables).

**À investiguer Sprint 3.8** avec des runs dédiés par classe
single-asset pour voir si le problème est data ou model.

## Tests

**103/103 OK, 3 skipped** (vs 90/90 avant).

- 13 nouveaux tests pour les fix AI (one-sided p, holdout check,
  merge_paths_or, ATR TP/SL, p-value floor)
- 8 tests corpus/archive (Sprint 3.6)
- 82 tests legacy (Sprints 2.x → 3.5)

## Fichiers modifiés

```
M  src/einherjar/research/xgb_einhers/backtester.py          (P0-1, P1-1)
M  src/einherjar/research/xgb_einhers/runner.py              (P0-2, P0-3, P0-4, P1-2, P2-6, auto-discover)
M  src/einherjar/research/xgb_einhers/admission.py           (P0-5)
M  src/einherjar/research/xgb_einhers/condition_tree.py     (P2-1)
M  src/einherjar/research/xgb_einhers/multi_asset_loader.py  (BUG-12 stocks alias)
M  src/einherjar/research/xgb_einhers/data_loader.py        (BUG-12 stocks alias)
A  audits/AI_REVIEW_2026-08-20.md                            (sauvegarde revue)
A  src/einherjar/research/tests/test_xgb_einhers/test_ai_review_fixes.py  (13 tests)
A  src/einherjar/research/xgb_einhers/corpus.py             (Sprint 3.6)
A  src/einherjar/research/xgb_einhers/archive.py             (Sprint 3.6)
A  src/einherjar/research/tests/test_xgb_einhers/test_corpus_archive.py
```

## Recommandations Sprint 3.8

1. **Fixer global scope path** : utiliser une vraie classe pour le path
2. **Vérifier explicitement le chemin compiled des stocks** (commande
   `ls` PowerShell avant de lancer)
3. **Runs single-asset forex/indices/commodities** pour voir si
   l'absence de signaux est data ou model
4. **Walk-forward K=3** sur les 252 Einhers du corpus v4
5. **DSR/PSR** pour ajuster les seuils d'admission

## Prochaine étape recommandée

Sprint 3.8 :
- [ ] Fix global scope path (15 min)
- [ ] Vérif manuelle des paths compiled pour stocks (5 min)
- [ ] Runs single-asset forex EURUSD, indices SPY, commodities WTI (1h)
- [ ] Walk-forward K=3 sur top 50 Einhers du corpus v4

Si walk-forward confirme, **passage en paper trading 1 mois** sur le
top 10 des Einhers.
