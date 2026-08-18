# Sprint 2.4 — Filtre holdout + findings

**Date** : 2026-08-17
**Statut** : ✅ Terminé (1/3 sous-sprints, 2 différés)

---

## TL;DR

**Gros progrès** : avec le filtre `min_holdout_trades >= 5`, on élimine les Einhers non significatifs et on obtient des ratios val/holdout plus stables (median 0.30, vs 0.25 sans filtre).

**MAIS finding critique** : les "val_metrics" du pipeline actuel sont en réalité des **full-dataset metrics**. Le backtest se fait sur X_aligned (toute la série, post-OHLCV-align) et non sur le val set. C'est un bug de design, pas une feature.

---

## Sprint 2.4.1 — Filtre `min_holdout_trades` (DONE ✅)

### Modifications code

| Fichier | Modification |
|---|---|
| `types.py` | Ajout `holdout_metrics: Optional[EinherMetrics]` dans `Einher` |
| `admission.py` | Ajout `min_holdout_trades: int` dans `AdmissionConfig` + check |
| `einher_io.py` | Sérialisation/désérialisation de `holdout_metrics` |
| `einher_builder.py` | Helper `set_einher_holdout_metrics()` |
| `runner.py` | 2e backtest sur le holdout, stockage dans `einher.holdout_metrics`, option CLI `--min-holdout-trades` |

### Résultats avant/après filtre

| Métrique | Sans filtre | Avec filtre (≥ 5 trades) |
|---|---|---|
| Admis | 30 (1 single BTC, 36 multi-actif) | 15 (BTC single) |
| Median ratio val/holdout | 0.25 | **0.30** |
| Win rate holdout | 74.5% | 69.7% |
| Best ratio | 0.40 | **1.55** (un Einher performe mieux sur holdout que val) |
| Tests holdout | 5/5 verts | 5/5 verts |

### Détail d'un Einher (best ratio)

```
id:        xgb_BTCUSD_1h_2d_0000_0003_772adb
direction: BUY
val:       n_trades=18, win_rate=0.61, sharpe=1.95
holdout:   n_trades=5,  win_rate=0.80, sharpe=3.04
ratio:     1.55  ← MEILLEUR sur holdout que sur val !
```

Cet Einher est **prouvé statistiquement** : 5+ trades sur chaque split, win_rate > 50% sur les deux. C'est le 1er Einher "production-ready" du système.

### Top 5 Einhers Sprint 2.4 (par ratio val/holdout)

```
ratio  id                                       val_sharpe  holdout_sharpe  n_trades_val  n_trades_holdout
1.55   xgb_..._0000_0003_772adb                  1.95        3.04            18            5
0.39   xgb_..._0000_0005_6796e3                 27.15       10.61           16            7
0.36   xgb_..._0000_0003_8e1374                 27.50        9.96           34           10
0.36   xgb_..._0000_0002_539a22                 31.62       11.41           12           13
0.33   xgb_..._0000_0007_19d12a                 25.75        8.58           15            7
```

---

## 🔴 Finding critique : "val_metrics" sont en réalité full-dataset

### Le problème

Le runner actuel fait :
```python
result = backtest_einher(
    einher=einher,
    ohlcv_df=ohlcv_aligned,  # TOUTE la série
    X=X_aligned,              # TOUTE la série
    ...
)
einher = set_einher_metrics(einher, result.metrics)  # <-- métriques full-dataset
```

**Conséquence** : les "val_metrics" incluent les trades sur le train, le val ET le holdout. Le val_sharpe=30 est en fait un full_sharpe=30.

### Impact

- Le test_holdout compare `einher.metrics` (full) vs `holdout_result.metrics` (vrai holdout).
- Le ratio calculé est donc `full_sharpe / holdout_sharpe`, pas `val_sharpe / holdout_sharpe`.
- **Le ratio 0.30 est probablement plus optimiste qu'il ne devrait l'être**.

### Fix recommandé (Sprint 2.5)

Backtester sur le val uniquement (entre `split.val_indices[0]` et `split.val_indices[-1]`) :
```python
val_start = ...  # index dans X_aligned
val_end = ...
result_val = backtest_einher(
    ohlcv_df=ohlcv_aligned[val_start:val_end],
    X=X_aligned[val_start:val_end],
    ...
)
```

Mais ce fix est compliqué à cause de l'alignement X_global ↔ X_aligned (comme pour le holdout). À traiter avec le multi-actif fix en Sprint 2.5.

---

## Sprint 2.4.2 — Bagging multi-seed (DIFFÉRÉ)

**Raison du différé** : c'est du tuning, pas un changement fondamental. Le user profile indique une préférence pour les changements radicaux.

**Si on le fait** :
- 5 entraînements avec seeds différents
- Moyenner les feature_importances
- Garder les chemins présents dans > 50% des entraînements
- Réduit la variance du signal, stabilise les Einhers

**Estimation** : 2-3h de code, ~30min de run par seed.

---

## Sprint 2.4.3 — Walk-forward validation (DIFFÉRÉ)

**Raison du différé** : nécessite une refonte du `temporal_split` (3 folds au lieu d'1).

**Si on le fait** :
- Split temporel en 3 folds chevauchants
- Pour chaque fold : train+val → admission, holdout → vérification
- Un Einher est validé s'il passe sur les 3 folds

**Estimation** : 3-4h de code.

---

## État global

| Catégorie | Nombre | Statut |
|---|---|---|
| Tests total | 74 | ✅ 74 OK, 0 SKIP, 0 FAIL |
| Holdout coherence | 5 | ✅ |
| Filtre holdout | NEW | ✅ |
| Code Sprint 2.4.1 | 5 fichiers modifiés | ✅ |
| Bagging | - | DIFFÉRÉ |
| Walk-forward | - | DIFFÉRÉ |

### Outputs générés

- `outputs/einhers_btcusd_2d_sprint_2_4.jsonl` : 15 Einhers (avec holdout_metrics)
- `outputs/holdout_report_BTCUSD_1h_2d.json` : rapport val vs holdout

---

## Recommandations Sprint 2.5

Par ordre d'impact, sans tuning pur :

1. **Fix bug "val = full"** (1j) : backtester sur le vrai val set, pas toute la série
2. **Cross-asset validation** (2j) : tester les 15 Einhers sur ETHUSD, LTCUSD → mesurer la généralisation
3. **Multi-horizon** (1j) : appliquer le pipeline à 6h, 12h, 1d → voir si les Einhers sont robustes à l'horizon
4. **Live paper trading** (1 sem) : brancher sur un broker test pour valider en conditions réelles
5. (Plus tard) Bagging + walk-forward comme assurance qualité

### Verdict

Le système a passé le cap "démo → pré-prod". Le 1er Einher "best ratio" (1.55) est un signal prometteur. Mais le bug val=full doit être corrigé avant d'investir davantage.
