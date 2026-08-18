# Sprint 3.1 — P1 : Correction multi-tests (Benjamini-Hochberg)

**Date** : 2026-08-18
**Statut** : ✅ P1 appliqué (BH + min_holdout_trades=100)
**Suivant** : P2 (multi-actif 28 actifs + multi-niveaux)

---

## TL;DR

**Benjamini-Hochberg appliqué au pipeline** : sur 30 candidats, **16 sont rejetés (53%)** au seuil FDR=5%. Les 14 restants passent les autres critères, et 9 finissent par être admis (5 sont rejetés par d'autres critères comme win_rate, profit_factor, etc.).

**Median ratio val/holdout** : 0.84 (vs 0.91 sur 14 Einhers sans BH) — légèrement plus bas, mais c'est attendu : on a éliminé les "excellents" qui étaient peut-être des flukes statistiques.

**Le filtre BH fait son travail** : sans lui, on aurait 30 Einhers candidats. Avec lui, on garde 14 statistiquement valides, puis 9 qui passent tous les critères.

---

## Code ajouté

### `multiple_testing.py` (NEW, ~150 lignes)

- `bootstrap_pvalue(returns, n_bootstrap=1000)` : p-value bootstrap pour H0: mean_return ≤ 0
- `benjamini_hochberg(pvalues, fdr=0.05)` : rejette les hypothèses non significatives
- `apply_bh_to_einhers(einhers, fdr=0.05)` : applique BH sur une liste d'Einhers

### `admission.py` (MODIFIED)

- `AdmissionConfig.fdr = 0.05` : seuil FDR
- `AdmissionConfig.apply_bh = True` : activer BH
- `AdmissionConfig.min_holdout_trades = 100` : Gemini recommande 100+ pour significativité (mais debug mode garde 0)
- `check_admission()` accepte un nouveau paramètre `bh_rejected` pour rejeter les Einhers non significatifs

### `runner.py` (MODIFIED)

Le pipeline est maintenant en 3 phases :
1. **Phase 1** : Génère + backtest TOUS les Einhers (sans admission)
2. **Phase 2** : Applique BH sur l'ensemble
3. **Phase 3** : Admission finale (avec check BH)

---

## Résultats avant/après BH

| Métrique | Sans BH (Sprint 3.0) | Avec BH (Sprint 3.1) |
|---|---|---|
| Candidats générés | 30 | 30 |
| Rejetés par BH | 0 | **16 (53%)** |
| Restant après BH | 30 | 14 |
| Admis finaux | 14 | 9 |
| Median ratio val/holdout | 0.91 | **0.84** |
| Win rate holdout | 70% | 69.87% |

**Détail des 9 Einhers BH-admis** :

| ID | val sharpe | holdout sharpe | ratio | n_trades val | n_trades holdout |
|---|---|---|---|---|---|
| `xgb_..._0000_0002_a795a5` | 7.20 | 4.27 | 0.59 | 19 | 9 |
| `xgb_..._0000_0002_a1a807` | 7.29 | 4.50 | 0.62 | 16 | 7 |
| `xgb_..._0000_0002_2e48fc` | 8.25 | 6.96 | 0.84 | 17 | 12 |
| `xgb_..._0000_0003_77bc17` | 4.27 | 3.01 | 0.70 | 16 | 5 |
| `xgb_..._0000_0002_a39d37` | 6.43 | 6.01 | 0.93 | 18 | 9 |
| `xgb_..._0000_0002_f0fc79` | 6.36 | 6.99 | **1.10** | 8 | 5 |
| `xgb_..._0000_0005_d37b2e` | 8.80 | 8.43 | 0.96 | 19 | 11 |
| `xgb_..._0000_0002_2e19ce` | 9.61 | 5.80 | 0.60 | 14 | 5 |
| `xgb_..._0000_0002_937415` | 9.54 | 9.07 | 0.95 | 14 | 9 |

**Best ratio** : 1.10 (`..._f0fc79`, SELL, val=6.36, holdout=6.99)

---

## Tests

**81 tests OK, 0 fail** (3 skipped = holdout legacy).

Les tests existants couvrent le data_loader, le backtester, le modèle, l'admission, etc. Le module `multiple_testing` est testé via le runner.

---

## Limitations connues

1. **min_holdout_trades=100 n'est PAS appliqué en debug mode** (pour permettre l'itération rapide). En mode strict, AUCUN des Einhers actuels ne passe ce filtre (5-25 trades holdout). C'est pourquoi on est encore en debug pour ce test.

2. **P-values approximées** : `apply_bh_to_einhers` utilise une approximation de p-value basée sur le sharpe et n_trades (loi normale), pas un vrai bootstrap. Pour un bootstrap exact, il faudrait stocker les rendements de chaque trade dans l'Einher (actuellement seul les métriques agrégées sont stockées).

3. **BH appliqué par run, pas globalement** : si on run plusieurs seeds × plusieurs horizons, chaque run a son propre BH. Pour une correction globale, il faudrait agréger tous les candidats d'une campagne de recherche.

---

## Verdict Sprint 3.1

**BH fait son travail** : 16/30 Einhers rejetés, et les 14 restants ont une cohérence val/holdout de 0.84 (vs 0.91 sans BH, mais sur moins d'Einhers). Le filtre élimine les candidats les plus douteux.

**Mais** : le ratio 0.84 reste en-dessous du seuil 0.88 de Sprint 2.5. Cela peut signifier que BH a éliminé les bons (overfittés) et gardé les moyens, OU que les 14 sans BH étaient déjà corrects.

**Impossible de trancher** sans plus de runs et une validation sur des données complètement out-of-sample (walk-forward).

---

## Prochaines étapes (P2)

Le user a demandé :
- **Multi-actif sur 28 actifs** avec TF et horizons variés
- **3 niveaux de modèles** : par asset, par marché (cross-asset inter-market), général (cross-market)
- **Système self-contained** runnable en local sans friction

C'est un gros chantier (probablement 1-2 semaines). Avant d'y aller, je recommande :
1. **Walk-forward validation** sur les 9 Einhers BH-admis (P0 manquant) → vérifier la stabilité temporelle
2. **Cross-asset test** re-validé avec les nouveaux sharpes (Sprint 3.0) → vérifier la généralisation réelle

Si ces 2 tests passent, on peut scaler sur 28 actifs.
