# BASELINES ANTI-HASARD — RAPPORT

Date : 2026-08-20
Module : `src/einherjar/research/baselines/` (nouveau)
Référence : plan `docs/PLAN_TECHNIQUES_RECHERCHE_FEATURES.md` — étape A6 (random baseline obligatoire, lignes 253-254) et couche 1 (lignes 491-493)
Exécution : `python -m einherjar.research.baselines.runner --asset {BTCUSD|ETHUSD} --timeframe 1h --horizon 2d --n-random 200 --seed 42`
Rapports JSON : `outputs/baselines_BTCUSD_1h_2d.json`, `outputs/baselines_ETHUSD_1h_2d.json`

---

## 1. OBJECTIF (Option B)

Établir la PREUVE ANTI-HASARD avant toute recherche GP :
- calibrer le pipeline d'évaluation (est-il sain ?),
- établir la distribution de performance d'Einhers ALEATOIRES (référence que le STGP devra battre),
- réutiliser exactement les conventions du pipeline xgb_einhers existant (mêmes frais, mêmes TP/SL, même mécanique de trades).

## 2. MÉTHODE — conventions identiques au pipeline xgb_einhers

| Paramètre | Valeur | Source |
|---|---|---|
| Split temporel | 60/20/20 + embargo max(50, horizon_bars) | `data_loader.temporal_split` (convention xgb_einhers) |
| Coûts round-trip | BTCUSD 0.14%, ETHUSD 0.20% | `config/fees_ctrader.json` via `label_engineer.load_costs` |
| TP/SL | 2.5% / 1.5% (défauts si tp_pct=sl_pct=0) | `backtester.backtest_einher` (backtester.py:301-308) |
| Entrée | à OPEN[t+1] | `backtester.py:322` |
| Positions | 1 seule à la fois (pas de stacking, FIX BUG-06) | `backtester.py` |
| Univers | 1 asset × 1 timeframe × 1 horizon | BTCUSD 1h 2d ; ETHUSD 1h 2d |
| Random search | 200 Einhers, seed 42, conditions AND 1-3 feuilles, opérateurs </>, seuils = quantiles de la fenêtre TRAIN uniquement (aucun lookahead), taux de déclenchement contraint [2%, 70%] par feuille | `baselines/random_gen.py` |

Baselines calculées :
1. **buy_hold** : rendement brut long-only + Sharpe annualisé sur rendements journaliers.
2. **always_long / always_short** : Einher à condition toujours vraie (même mécanique de trades que les candidats).
3. **random_search** : les 200 aléatoires, backtestés sur **val** (sélection) et **holdout** (une passe, rapport seul — aucune décision prise sur le holdout).

## 3. FIX BASELINE-01 (bug latente du backtester révélée par les baselines)

**Symptôme** : des candidats à 2-3 trades (tous TP) produisaient des Sharpe de l'ordre de 10^15.
**Cause** : des trades quasi identiques donnent un std numérique ~1e-16 (non nul au dernier ulp) ; le garde `std > 0` le laissait passer → division par ~0.
**Fix** : garde anti-dégénérescence dans `compute_metrics` — si `std <= 1e-12 * max(1e-12, |mean|)`, Sharpe et t-stat sont traités comme non définis (0.0 / 0.0 / p=1.0).
**Preuve** : test unitaire ajouté `test_identical_returns_sharpe_zero`.

## 4. RÉSULTATS

### 4.1 Calibration du pipeline — est-il sain ?

| Vérification | Attendue | BTCUSD 1h 2d | ETHUSD 1h 2d | Verdict |
|---|---|---|---|---|
| win_rate always_long | ~TP/(TP+SL) = 37.5% (théorie, drift nul) | 34.9% | 36.7% | OK (bull market → léger déficit) |
| win_rate médian aléatoire | ~30-40% | 30.4% | 32.4% | OK |
| médiane aléatoire nette de coûts | ≤ 0 (pas de free lunch) | -0.55 | -1.49 | OK |
| % candidats positifs | < 50% | 21% | 6% | OK |
| always_short < always_long (marché haussier) | cohérent avec le régime | -2.85 < -0.17 | -3.50 < -1.24 | OK |

→ **Pipeline sain** (verdict `pipeline_sane=True` dans les deux rapports). Le marché ne donne rien gratuitement : la médiane aléatoire est négative nette de coûts.

### 4.2 Référence anti-hasard pour le STGP

| Statistique (val) | BTCUSD 1h 2d | ETHUSD 1h 2d |
|---|---|---|
| Médiane Sharpe aléatoire | **-0.55** | **-1.49** |
| p95 Sharpe aléatoire | **0.70** | **0.03** |
| Médiane total return | -22% | -44% |
| % Sharpe > 0 | 21% | 6% |
| Passent l'admission complète | 4/200 | 3/200 |
| Corrélation Sharpe val/holdout | 0.38 | 0.34 |

**Lecture pour la suite** : le STGP devra battre **le quantile p95 aléatoire sur val (0.70 BTC) ET survivre à la validation lourde** (holdout + DSR + FDR). La corrélation val→holdout (~0.35) est faible : sélectionner sur val seul ne transfère PAS — c'est exactement pourquoi le plan impose la stack de validation de la section C.

### 4.3 Candidats notables (rappel : ce sont des ALÉATOIRES, pas des stratégies)

| Candidat | Condition | val Sharpe | val ret | holdout Sharpe | holdout ret |
|---|---|---|---|---|---|
| bl_BTCUSD_0069 | (Factor_Volatility_Score < 0.3397 AND skewness_risk < 0.3547) | 1.96 | +80.7% (326 trades) | 1.29 | +60.4% (401 trades) |
| bl_BTCUSD_0130 | ((aroon_down < 5 AND adx_strength_signal > 0.2826) AND quant_rolling_skewness > 0.4109) | 1.38 | +63.5% | 0.24 | +7.1% |
| bl_ETHUSD_0120 | quant_rolling_skewness > 0.9064 | 0.99 | +55.0% (550 trades) | 2.78 | +168.5% (627 trades) |
| bl_ETHUSD_0052 | ((macd_line > -0.01443 AND quant_dynamic_cvar > 0.01306) AND quant_max_drawdown > 0.04704) | 1.80 | +28.6% (42 trades) | -0.15 | -4.0% |

Interprétation honnête : avec 200 tirages × 2 actifs, QUELQUES candidats qui survivent au holdout sont statistiquement attendus (FDR). Le BTC-0069 (326/401 trades, positif des deux côtés) et l'ETH-0120 (+168% holdout en bear market ETH) sont des indices que l'espace CONTIENT du signal — mais leur admission réelle exigera la stack C complète (bootstrap CI, DSR, FDR) avant toute persistance. C'est la matière première du STGP : la preuve qu'il existe des combinaisons non triviales à découvrir systématiquement.

## 5. VERDICT GLOBAL

1. **Pipeline d'évaluation CALIBRÉ** : win rates, médianes négatives nettes de coûts, biais directionnel cohérent avec le régime — rien d'exotique.
2. **Référence anti-hasard ÉTABLIE** : médianes aléatoires -0.55 / -1.49 (Sharpe annuel), p95 0.70 / 0.03.
3. **La sélection sur val ne transfère pas sans validation lourde** (corr ~0.35) → le STGP sera jugé sur : battre p95 aléatoire en val + survie holdout + tests multiples.
4. **Le garde anti-dégénérescence (BASELINE-01) couvre un vrai bug latent du backtester** que le pipeline xgb n'aurait jamais déclenché (l'admission exige ≥30 trades).

## 6. FICHIERS

| Fichier | Rôle |
|---|---|
| `src/einherjar/research/baselines/__init__.py` | package |
| `src/einherjar/research/baselines/vector_eval.py` | évaluation vectorisée des AST (O(N) numpy au lieu d'O(N·F) python) |
| `src/einherjar/research/baselines/random_gen.py` | génération d'Einhers aléatoires (seuils train-only, taux bornés) |
| `src/einherjar/research/baselines/runner.py` | CLI + rapport JSON + verdict |
| `src/einherjar/research/tests/test_baselines.py` | 11 tests unitaires |
| `src/einherjar/research/xgb_einhers/backtester.py` | FIX BASELINE-01 (garde dégénérée) |
| `src/einherjar/research/tests/test_xgb_einhers/test_backtester.py` | test du cas dégénéré ajouté |
| `outputs/baselines_BTCUSD_1h_2d.json` / `baselines_ETHUSD_1h_2d.json` | rapports complets |

## 7. PROCHAINES ÉTAPES (avant construction A→Z)

1. **Verrouiller les 5 paramètres du plan** (langage STGP, fitness cheap, descripteurs MAP-Elites, seuils de validation, découpage temporel) + trancher le coût (plan 0.08% vs fees_ctrader 0.14-0.20% réel).
2. Construire le système couche 1 : random baseline (existant) + STGP + MAP-Elites, avec admission C1/C6 (bootstrap CI borne basse > 0) dès la première version.
3. Mettre à jour le corpus.jsonl uniquement via la procédure d'admission existante.