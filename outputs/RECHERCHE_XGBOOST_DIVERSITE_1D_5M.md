# RAPPORT DE RECHERCHE — XGBOOST EINHERJAR

# Diversité, 1D, 5M

**Date:** 2026-08-28
**Auteur:** Hermes Agent (recherche multi-agents)
**Base:** Corpus 1080 Einhers, 20167 rejetés, 218 features, 28 actifs, 5 TF

---

# 1. DIVERSITÉ ET NOMBRE D'EINHERS

## 1.1 Problématique

Comment faire en sorte que XGBoost explore une partie beaucoup plus large de
l'espace des signaux et produise davantage d'Einhers réellement diversifiés ?

## 1.2 État actuel du système

**Concentration extrême :**

- 3 features → 45%+ du corpus (chaikin_oscillator 45.6%, Factor_Momentum_Score 39.6%, skewness_risk 34.7%)
- 109/213 features utilisées (51%) → 104 features jamais exploitées
- 94 Einhers avec sharpe > 15 (valeurs absurdes, surtout en multi-asset)

**Sources d'Einhers :**

- XGBoost : 341 (31.6%)
- XGBoost + veto : 247 (22.9%)
- Subgroup Discovery : 245 (22.7%)
- SD + veto : 243 (22.5%)
- or_regimes/event_study : 4 (0.4%)

**Distribution par TF :**

- 15m : 532 (49.3%)
- 1h : 480 (44.4%)
- 4h : 64 (5.9%)
- 1d : 4 (0.4%)

## 1.3 Causes principales

### Cause 1 : Biais de gain XGBoost

XGBoost MSE sélectionne greedily les features avec le plus de gain. Les features
continues à forte variance (Factor_Momentum_Score, skewness_risk) dominent
car elles offrent le meilleur split à chaque nœud. Les features binaires
(pattern_*, signal_*) ont un gain quasi nul car :

- min_child_weight=50 empêche les feuilles sur les classes rares (5% True)
- MSE ne valorise pas les splits sur des classes déséquilibrées

### Cause 2 : Absorption par les Factor_* composites

Factor_Momentum_Score = RSI + Stochastic + Williams %R + ROC + momentum.
Un seul split sur ce composite capture l'information de 5 indicateurs.
Résultat : les indicateurs individuels (RSI_14, stoch_k, etc.) sont rendus
redondants par le composite.

### Cause 3 : Pas de pression de diversité

Le pipeline actuel : XGBoost → extract_paths → Top-K → backtest → admission.
Le Top-K sélectionne par qualité pure → converge sur les mêmes features.
Aucun mécanisme ne pénalise la similarité avec les Einhers déjà sélectionnés.

### Cause 4 : Prolifération de jumeaux

Pas de détection des Einhers quasi-identiques (mêmes features, seuils proches).
Les slots d'admission sont gaspillés sur des doublons.

## 1.4 Approches étudiées

### Approche A : Modèles XGBoost par famille (RECOMMANDÉ — Phase 1)

**Concept :** Entraîner 11 modèles XGBoost séparés, un par famille de features.
Chaque modèle ne voit que les features de sa famille.

**Pourquoi ça marche :**

- Élimine la concentration par construction
- Les familles binaires (price_action, market_structure) obtiennent leur propre
  modèle avec min_child_weight adapté (5 au lieu de 50)
- Chaque famille contribue proportionnellement

**Littérature :**

- Council of Alphas (HackEurope 2026) : 4 spécialistes locked par famille,
  Consensus Gate = 1.76 Sharpe annualisé sur SOL/USD
- AlphaMix (KDD 2023) : MoE à 3 stages, experts par style, meilleur performance
- MIGA (arXiv:2410.02241) : experts groupés par style, +24% rendement excédentaire

**Implémentation :**

```python
families = {
    'trend': trend_features,           # 26 features
    'momentum': momentum_features,     # 15
    'volatility': volatility_features, # 14
    'statistical': statistical_features, # 15
    'volume_flow': volume_flow_features, # 10
    'risk': risk_features,             # 8
    'market_regime': regime_features,  # 6
    'microstructure': micro_features,  # 2
    'price_action': price_action_features,  # 62 (55 bool + 7 float)
    'market_structure': market_structure_features, # 52 (49 bool + 3 float)
    'factors': factor_features,        # 21
}

for family_name, features in families.items():
    is_binary = family_name in ('price_action', 'market_structure')
    model = xgb.XGBRegressor(
        colsample_bytree=0.8,
        min_child_weight=5 if is_binary else 50,
        max_depth=4,
        n_estimators=100,
    )
    model.fit(X[features], y)
    # Extract paths from this model
```

**Impact attendu :** Features utilisées : 109/213 → 180+/213. Concentration :
45%+ → <15% par feature.

### Approche B : Décomposition Factor_* (RECOMMANDÉ — Phase 1)

**Concept :** Entraîner un modèle SANS les Factor_* composites pour forcer
la découverte d'indicateurs individuels.

**Pourquoi ça marche :**

- Factor_Momentum_Score absorbe RSI, Stoch, Williams, ROC
- Sans le composite, XGBoost doit découvrir quels sous-indicateurs sont prédictifs
- Différentes stratégies peuvent utiliser différents sous-indicateurs

**Implémentation :**

```python
non_factor_features = [f for f in all_features if not f.startswith('Factor_')]
model_without_factors = xgb.XGBRegressor(...)
model_without_factors.fit(X[non_factor_features], y)
```

**Impact attendu :** Débloque ~50 features actuellement ombragées par les composites.

### Approche C : colsample_bytree=0.5 (RECOMMANDÉ — Phase 1)

**Concept :** Forcer chaque arbre à utiliser seulement 50% des features.

**Littérature :**

- Ho (1998) : Random Subspace Method — decorréler les membres de l'ensemble
- EMA-FS (arXiv:2606.26337) : subsampling guidé par EMA des gains
- arXiv:2601.08121 : éviter colsample_bynode (casse les interactions ratio)

**Impact attendu :** Modéré — réduit la concentration mais ne garantit pas la
couverture de toutes les familles. À combiner avec les modèles par famille.

### Approche D : MMR — Maximal Marginal Relevance (Phase 2)

**Concept :** Remplacer le Top-K par un ranking qui pénalise la similarité
avec les Einhers déjà sélectionnés.

**Littérature :**

- Carbonell & Goldstein (1998) : MMR original
- α-NDCG : pénalise la redondance dans les listes classées
- DPP (Determinantal Point Process) : modèle probabiliste de diversité

**Formule :**

```
score(d) = λ · quality(d) - (1-λ) · max_sim(d, selected)
```

**Impact attendu :** Élevé — adresse directement la prolifération de jumeaux
et la concentration de features.

### Approche E : Détection et fusion de jumeaux (Phase 2)

**Concept :** Détecter les Einhers qui partagent les mêmes features avec des
seuils proches, les fusionner en versions généralisées.

**Critères de jumeaux :**

- Overlap de features > 80% (Jaccard)
- Seuils distants de < 10% de la valeur

**Impact attendu :** Libère des slots d'admission pour des stratégies réellement
différentes.

### Approche F : MAP-Elites Archive (Phase 3)

**Concept :** Remplacer la sélection flat par une archive MAP-Elites indexée
par des descripteurs comportementaux.

**Littérature :**

- Mouret & Clune (2015) : MAP-Elites original
- QuantEvolve (arXiv:2510.18569) : MAP-Elites pour trading, 16 bins = optimal
- QD Portfolio (arXiv:2402.16118) : diversité et qualité ne s'opposent pas

**Descripteurs pour Einherjar :**

1. Famille de features dominante (11 catégories)
2. Profil de risque (Sharpe bucket : low/med/high)
3. Timeframe (5m/15m/1h/4h/1d)

**Impact attendu :** Très élevé — garantit un champion par niche comportementale.

## 1.5 Comparaison

| Approche               | Effort      | Diversité   | Risque qualité | Complexité  |
| ---------------------- | ----------- | ----------- | -------------- | ----------- |
| Modèles par famille    | Faible      | Élevée      | Faible         | Faible      |
| Décomposition Factor_* | Faible      | Élevée      | Faible         | Faible      |
| colsample_bytree=0.5   | Très faible | Moyenne     | Très faible    | Très faible |
| MMR                    | Moyen       | Élevée      | Très faible    | Moyen       |
| Fusion de jumeaux      | Moyen       | Moyenne     | Très faible    | Moyen       |
| MAP-Elites             | Élevé       | Très élevée | Faible         | Élevé       |
| MoE routing            | Élevé       | Très élevée | Moyen          | Élevé       |

## 1.6 Contre-analyse

**Critique 1 :** Les modèles par famille produisent des stratégies de moindre
qualité car chaque modèle a accès à moins de features.
**Réponse :** Vrai, mais le but est la qualité au niveau PORTFOLIO, pas individuel.
Un portfolio diversifié de 7/10 surpasse un portfolio concentré de 10/10
(MAPLE, QuantEvolve, AlphaMix).

**Critique 2 :** Peut-être que chaikin_oscillator est réellement la meilleure
feature et forcer la diversité ajoute du bruit.
**Réponse :** Dépend de l'objectif. Si le but est la robustesse portfolio,
la diversité est essentielle car les portfolios concentrés échouent lors
des changements de régime. Renaissance Technologies : "l'avenir est dans
les ensembles de many marginal edges."

**Critique 3 :** colsample_bytree=0.5 dilue les SHAP values.
**Réponse :** Problème d'interprétation, pas d'entraînement. Pour l'analyse,
entraîner un modèle séparé sans subsampling.

## 1.7 Recommandation finale

**Phase 1 (immédiate) :**

1. Modèles XGBoost par famille (11 modèles)
2. Décomposer Factor_* (modèle sans composites)
3. colsample_bytree=0.5
4. min_child_weight=5 pour les familles binaires

**Phase 2 (2-4 semaines) :**
5. MMR pour la sélection
6. Détection/fusion de jumeaux
7. Métrique de diversité comportementale

**Phase 3 (1-2 mois) :**
8. Archive MAP-Elites
9. MoE routing

---

# 2. EXPLOITATION DU 1D

## 2.1 Problématique

Comment adapter le système XGBoost aux données daily/1D afin d'exploiter
leur signal propre sans les traiter comme une simple version plus lente ?

## 2.2 État actuel

**Données 1D :**

- 89 actifs, rows=1179-5203 (médiane ~2500)
- 213 features, ratio observations/features = ~12:1 (dangereusement bas)
- Horizons : 5d, 10d, 20d, 60d
- Résultat actuel : 4 Einhers admis sur 120 triplets (99.7% de rejet)

**Problème principal :** Le ratio observations/features est trop bas pour
XGBoost. Avec 2500 lignes et 213 features, le modèle a trop de degrés
de liberté par rapport aux données disponibles.

## 2.3 Causes principales

### Cause 1 : Trop de features pour peu de données

213 features × 2500 lignes = ratio 12:1. En ML, le ratio minimum recommandé
est ~50:1 pour éviter l'overfitting. Avec 213 features, il faudrait ~10,000
lignes minimum.

### Cause 2 : BH trop strict même avec FDR adaptatif

Avec peu de données, les t-stats sont naturellement plus faibles. Le BH
rejette 78% des candidats. Avec FDR=0.12-0.15, c'est encore trop strict
pour le 1D.

### Cause 3 : Split temporel dévore les données

60% train = 1500 lignes, 20% val = 500, 20% holdout = 500.
Après embargo (50+ barres), la fenêtre val a encore moins de données.
Peu de trades possibles → t-stats faibles → BH rejette.

### Cause 4 : Non-stationnarité

Les données daily couvrent 4-20 ans. Les régimes changent (bull/bear/sideways).
Un modèle entraîné sur 2018-2022 peut ne pas généraliser à 2023-2025.

## 2.4 Approches étudiées

### Approche A : Réduction drastique des features (RECOMMANDÉ)

**Concept :** Pour le 1D, utiliser seulement 20-40 features au lieu de 213.

**Sélection :**

- Top 10 par famille de features (par importance dans un modèle préliminaire)
- Ou features avec la plus faible autocorrélation (plus d'information indépendante)
- Ou features sélectionnées par Lasso/ElasticNet

**Littérature :**

- Guyon & Elisseeff (2003) : "An Introduction to Variable and Feature Selection"
- Avec n=2500, p=20-40 donne un ratio 60-125:1 → suffisant pour XGBoost

**Impact attendu :** Ratio observations/features passe de 12:1 à 60-125:1.
Overfitting réduit drastiquement.

### Approche B : Walk-Forward Validation (RECOMMANDÉ)

**Concept :** Remplacer le split fixe 60/20/20 par un walk-forward avec
fenêtres glissantes.

**Pourquoi c'est critique pour le 1D :**

- Le 1D couvre 4-20 ans → les régimes changent
- Un split fixe peut tester sur un régime totalement différent de l'entraînement
- Walk-forward simule la réalité : entraîner sur le passé, tester sur le futur

**Implémentation :**

```python
# Walk-forward avec 5 folds
n_folds = 5
fold_size = len(X) // (n_folds + 1)

for i in range(n_folds):
    train_start = 0
    train_end = fold_size * (i + 1)
    val_start = train_end + embargo
    val_end = val_start + fold_size
    
    X_train = X[train_start:train_end]
    y_train = y[train_start:train_end]
    X_val = X[val_start:val_end]
    y_val = y[val_start:val_end]
    
    model.fit(X_train, y_train)
    # Backtest on X_val
```

**Impact attendu :** Validation plus réaliste, détection d'overfitting,
meilleure généralisation.

### Approche C : Features spécifiques au 1D

**Concept :** Ajouter des features adaptées aux données daily.

**Features à ajouter :**

- Rendements sur 5/10/20/60 jours (lag features)
- Volatilité réalisée sur 20/60 jours
- Drawdown depuis le plus haut
- Régime de marché (bull/bear/sideway) via HMM ou z-score
- Saisonnalité (mois de l'année, jour de la semaine)
- Cross-sectional rank (position relative aux autres actifs)

**Littérature :**

- Gu, Kelly & Xiu (2020) : "Empirical Asset Pricing via Machine Learning"
  → les lag features et cross-sectional ranks sont les plus importants pour
  les données daily

### Approche D : XGBoost paramètres adaptés au 1D

**Paramètres recommandés :**

```python
xgb.XGBRegressor(
    n_estimators=50,        # Moins d'arbres (moins de données)
    max_depth=3,            # Plus shallow (moins d'overfitting)
    learning_rate=0.01,     # Plus lent (plus de régularisation)
    subsample=0.7,          # Plus de subsampling
    colsample_bytree=0.5,   # Plus de subsampling features
    min_child_weight=20,    # Plus élevé (feuilles plus grandes)
    reg_alpha=2.0,          # Plus de régularisation L1
    reg_lambda=10.0,        # Plus de régularisation L2
    early_stopping_rounds=10,
)
```

### Approche E : Modèles par régime

**Concept :** Détecter les régimes de marché et entraîner un modèle par régime.

**Pourquoi c'est critique pour le 1D :**

- Le 1D couvre 4-20 ans → multiple régimes
- Un modèle unique mélange les régimes → signal dilué
- Des modèles séparés capturent les spécificités de chaque régime

**Implémentation :**

```python
# Détecter les régimes via HMM ou seuils simples
regime = detect_regime(ohlcv)  # 'bull', 'bear', 'sideways'

# Entraîner un modèle par régime
for r in ['bull', 'bear', 'sideways']:
    mask = regime == r
    if mask.sum() > 100:  # Assez de données
        model_r = xgb.XGBRegressor(...)
        model_r.fit(X[mask], y[mask])
```

### Approche F : Cross-sectional modeling

**Concept :** Au lieu de prédire le rendement absolu, prédire le rang
relatif des actifs.

**Pourquoi c'est pertinent pour le 1D :**

- Les rendements daily sont très bruités
- Le rang relatif est plus stable (moins de variance)
- Permet de comparer les actifs entre eux

**Littérature :**

- Gu, Kelly & Xiu (2020) : cross-sectional models surperforment les
  time-series models pour les données daily

## 2.5 Comparaison

| Approche                | Impact qualité | Impact diversité | Effort      | Risque |
| ----------------------- | -------------- | ---------------- | ----------- | ------ |
| Réduction features      | Très élevé     | Moyen            | Faible      | Faible |
| Walk-forward            | Élevé          | Faible           | Moyen       | Faible |
| Features spécifiques 1D | Élevé          | Élevé            | Moyen       | Faible |
| Paramètres adaptés      | Moyen          | Faible           | Très faible | Faible |
| Modèles par régime      | Élevé          | Élevé            | Élevé       | Moyen  |
| Cross-sectional         | Élevé          | Élevé            | Élevé       | Moyen  |

## 2.6 Contre-analyse

**Critique 1 :** Réduire à 20-40 features risque de perdre des signaux importants.
**Réponse :** Avec 2500 lignes, 213 features causent de l'overfitting. Mieux vaut
20 features bien choisies que 213 features avec overfitting. La sélection peut
être walk-forward (sélectionner sur train, valider sur val).

**Critique 2 :** Walk-forward avec 5 folds sur 2500 lignes = 500 lignes par fold.
C'est trop peu pour un backtest significatif.
**Réponse :** Utiliser des folds plus grands (3 folds de 800 lignes) ou un
expanding window (train croissant, val fixe).

**Critique 3 :** Les modèles par régime nécessitent de détecter les régimes
correctement, ce qui est un problème en soi.
**Réponse :** Commencer par des seuils simples (rendement 200j > 10% = bull,
< -10% = bear, sinon sideways). Affiner avec HMM si nécessaire.

## 2.7 Recommandation finale

**Immédiat :**

1. Réduire les features à 30-40 pour le 1D (sélection par importance)
2. Paramètres XGBoost adaptés (depth=3, n_estimators=50, reg_lambda=10)
3. FDR encore plus permissif (0.20) pour le 1D

**Court terme :**
4. Walk-forward validation (3-5 folds)
5. Features spécifiques 1D (lag, régime, cross-sectional)

**Moyen terme :**
6. Modèles par régime
7. Cross-sectional modeling (pooling 28 actifs = 56k lignes au lieu de 2k)

**Insight clé de l'agent 1D :** Le pooling cross-sectional est la méthode
utilisée par TOUS les grands quant firms pour les données daily. Au lieu de
28 modèles séparés avec 2k lignes chacun, un seul modèle avec 56k lignes.
Le ratio observations/features passe de 9:1 à 1,120:1.

**Insight critique :** Le taux de rejet de 99.7% sur le 1D est peut-être
CORRECT — la plupart des "signaux" avec 218 features et 2000 lignes sont
des artefacts de multiple testing. L'objectif n'est pas d'augmenter le taux
d'admission mais la QUALITÉ des stratégies admises.

---

# 3. EXPLOITATION DU 5M

## 3.1 Problématique

Comment exploiter le 5M avec XGBoost tout en conservant le signal utile,
en contrôlant le bruit et en maintenant un coût computationnel raisonnable ?

## 3.2 État actuel

**Données 5M :**

- 99 actifs, rows=37,430-1,169,617 (BTCUSD = 834k)
- 213 features, ratio observations/features = ~4000:1 (excellent)
- Horizons : 15m, 30m, 1h, 2h
- Résultat actuel : 0 Einhers admis sur les tests 5M

**Problème principal :** Volume énorme mais signal faible. Le 5M est dominé
par le bruit de microstructure. Le coût computationnel est prohibitif
(834k lignes × 213 features = ~700 MB par array).

## 3.3 Causes principales

### Cause 1 : Bruit de microstructure

Le 5M contient du bruit de bid-ask bounce, du bruit d'exécution, des
micro-évènements. Le ratio signal/bruit est très bas comparé au 1h ou 1d.

### Cause 2 : Coût computationnel

834k lignes × 213 features = ~700 MB par array. Avec 3 workers = 2.1 GB
juste pour les données, plus XGBoost = 6-12 GB total. Swap → deadlock.

### Cause 3 : Autocorrélation intraday

Les prix 5M sont très autocorrélés (lag-1 correlation > 0.99). Cela signifie
que les observations ne sont pas indépendantes → le nombre effectif
d'observations est beaucoup plus faible que 834k.

### Cause 4 : Coûts de transaction

Sur le 5M, les coûts de transaction (spread + commission) représentent une
part plus importante des gains. Un signal qui fonctionne en théorie peut
être éliminé par les coûts.

### Cause 5 : Le 5M n'est PAS un 1h plus rapide

Le 5M a une structure de bruit fondamentalement différente :

- Bid-ask bounce dominant
- Microstructure (order flow, queue position)
- Non-IID returns (autocorrélation, hétéroscédasticité, queues grasses)
- Signal/bruit ∝ √T → un barre daily a √288× plus de signal qu'une barre 5M

## 3.4 Approches étudiées

### Approche A : Dollar Bars (RECOMMANDÉ — #1 priorité)

**Concept :** Convertir les barres temporelles 5M en "dollar bars" — échantillonner
tous les $N dollars tradés au lieu de toutes les 5 minutes.

**Pourquoi c'est le changement le plus important :**

- Les barres temporelles sont la PIRE façon d'échantillonner un marché
- L'information n'arrive pas à taux constant
- Les dollar bars normalisent le flux d'information
- Les rendements des dollar bars sont plus proches de IID (critique pour ML)
- Réduction de données de 5-20× tout en préservant le signal

**Littérature :**

- López de Prado (2018) : "Advances in Financial Machine Learning", Chapter 2
  → dollar bars surperforment les time bars dans tous les cas testés
- Bieganowski & Ślepaczuk (2026) : arXiv:2602.00776
  → patterns stables cross-asset dans les features de microstructure

**Impact attendu :** Réduction 5-20× du dataset, returns plus IID, signal/bruit amélioré.

### Approche B : Sampling intelligent (RECOMMANDÉ)

**Concept :** Réduire le dataset en échantillonnant intelligemment au lieu
de prendre toutes les barres 5M.

**Méthodes :**

#### A1. Volume-based sampling

Ne garder que les barres où le volume est significatif (> moyenne × 1.5).
**Pourquoi :** Les barres à faible volume sont du bruit. Les signaux
significatifs apparaissent sur les barres à volume élevé.

#### A2. Event-based sampling

Ne garder que les barres où quelque chose d'important se passe :

- Variation de prix > 2σ
- Volume > 3σ
- Changement de tendance (cross EMA)
**Pourquoi :** Réduit le dataset de 80-90% tout en gardant les événements
significatifs.

#### A3. Temporal aggregation

Agréger les barres 5M en barres 15M ou 30M pour les features, mais
garder la granularité 5M pour le trading.
**Pourquoi :** Les features sur 15M sont moins bruitées, mais le signal
de trading peut être à 5M.

**Littérature :**

- López de Prado (2018) : "Advances in Financial Machine Learning"
  → bar sampling, volume sampling, tick sampling
- Easley, López de Prado, O'Hara (2012) : "Flow Toxicity and Liquidity
  in a High Frequency World" → le volume est le meilleur filtre

**Impact attendu :** Réduction du dataset de 80-90% (834k → 80-170k lignes).
Signal/bruit amélioré. Coût computationnel divisé par 5-10.

### Approche B : Features spécifiques au 5M

**Concept :** Utiliser des features adaptées à la microstructure intraday.

**Features à ajouter :**

- Heure de la jour (0-23) → saisonnalité intraday
- Jour de la semaine (0-4) → effet day-of-week
- Volume relatif (vs moyenne 20 barres)
- Spread bid-ask (si disponible)
- Imbalance acheteur/vendeur
- Distance au VWAP
- Volatilité réalisée intraday (sur 1h, 4h)
- Régime de volatilité (VIX intraday proxy)

**Littérature :**

- Cont, Kukanov & Stoikov (2014) : "The Price Impact of Order Book Events"
  → les features de microstructure sont les plus prédictives à haute fréquence

### Approche C : XGBoost GPU (RECOMMANDÉ pour le volume)

**Concept :** Utiliser l'accélération GPU pour traiter les gros datasets 5M.

**Configuration actuelle :**

- tree_method='hist' (CPU)
- device='cpu'

**Configuration recommandée :**

```python
xgb.XGBRegressor(
    tree_method='hist',
    device='cuda',  # GPU
    max_bin=128,    # Réduit de 256 (défaut) pour accélérer
    n_estimators=100,
    max_depth=4,
)
```

**Contraintes :**

- GPU GTX 1660 Ti = 6 GB VRAM
- 834k × 213 × 4 bytes = ~700 MB → OK pour le GPU
- Mais : ne fonctionne qu'avec 1 worker (pas de partage GPU)
- Avec 1 worker GPU + 2 workers CPU = bon compromis

**Littérature :**

- XGBoost docs : GPU hist est 5-10x plus rapide que CPU hist pour les
  gros datasets

### Approche D : Multi-résolution

**Concept :** Combiner des features à différentes résolutions temporelles.

**Implémentation :**

```python
# Features 5M (bruitées mais réactives)
features_5m = compute_features(ohlcv_5m)

# Features 15M (moins bruitées)
features_15m = resample_and_compute(ohlcv_15m)

# Features 1h (signal plus fort)
features_1h = resample_and_compute(ohlcv_1h)

# Combiner
X = np.concatenate([features_5m, features_15m, features_1h], axis=1)
```

**Pourquoi c'est pertinent pour le 5M :**

- Les features 5M sont bruitées
- Les features 1h capturent les tendances
- La combinaison donne le meilleur des deux mondes

**Littérature :**

- Zhang et al. (2019) : "Multi-scale Feature Engineering for Stock Prediction"
  → les features multi-échelles surperforment les features mono-échelle

### Approche E : QuantileDMatrix + External Memory (RECOMMANDÉ)

**Concept :** Utiliser QuantileDMatrix pour la compression mémoire (4-8×)
et ExtMemQuantileDMatrix pour l'entraînement out-of-core.

**QuantileDMatrix :**

```python
# Au lieu de DMatrix classique (float64) :
dmatrix = xgboost.QuantileDMatrix(X, y)  # Stockage quantizé
# Économie mémoire : 4-8×
```

**External Memory (out-of-core) :**

```python
class DataIterator(xgboost.DataIter):
    def next(self, input_data):
        chunk = load_chunk(self.file_paths[self._it])
        input_data(data=chunk['X'], label=chunk['y'])
        self._it += 1
        return 1

it = DataIterator(chunks)
dmatrix = xgboost.ExtMemQuantileDMatrix(it)
```

**Impact :** Permet d'entraîner sur des datasets de n'importe quelle taille.
GPU external memory cache en RAM ou disque (XGBoost 3.0+).

### Approche F : Features microstructure spécifiques

**Concept :** Utiliser des features adaptées à la microstructure intraday.

**Features à ajouter :**

- Order flow imbalance : (buy_vol - sell_vol) / total_vol
- Trade intensity : trades/bar / avg trades/bar
- Spread bps : (ask-bid) / mid × 10000
- Garman-Klass vol : estimateur de volatilité OHLC
- Return autocorrelation (lag 1, 5)
- Heure/jour encodé en sin/cos (pas raw)
- Session features (Asian/EU/US)
- Multi-resolution (features 15m/1h agrégées depuis 5M)

**Features à SUPPRIMER pour le 5M :**

- Moyennes mobiles longues (50, 100, 200 SMA)
- Momentum hebdo/mensuel
- Volatilité longue (30d, 60d)

### Approche G : Gestion mémoire optimisée

**Concept :** Minimiser la mémoire pour permettre le parallélisme.

**Optimisations :**

1. **mmap_mode='r'** : déjà implémenté, charge les NPY en mémoire virtuelle
2. **float32** : déjà utilisé (pas de changement nécessaire)
3. **Streaming XGBoost** : entraîner par batches sur le dataset
4. **Subsampling avant entraînement** : échantillonner 100k lignes au lieu de 834k

**XGBoost streaming :**

```python
# XGBoost supporte l'entraînement incrémental
model = xgb.XGBRegressor(...)
for batch in chunks(X, size=100_000):
    model.fit(batch.X, batch.y, xgb_model=model.get_booster())
```

### Approche F : Stratégies courtes

**Concept :** Le 5M est naturellement adapté aux stratégies courtes
(horizon 15m-2h). Ne pas essayer de trouver des signaux long terme.

**Implications :**

- Horizons 15m et 30m seulement (pas 1h ou 2h)
- Coûts de transaction plus importants → seuils de profit plus élevés
- Plus de trades → t-stats plus élevés → BH moins strict

## 3.5 Comparaison

| Approche                | Impact signal | Impact RAM | Impact CPU | Effort |
| ----------------------- | ------------- | ---------- | ---------- | ------ |
| Volume sampling         | Élevé         | Très élevé | Très élevé | Faible |
| Event sampling          | Élevé         | Très élevé | Très élevé | Moyen  |
| Temporal aggregation    | Moyen         | Élevé      | Élevé      | Faible |
| Features microstructure | Élevé         | Faible     | Faible     | Moyen  |
| XGBoost GPU             | Faible        | Moyen      | Très élevé | Faible |
| Multi-résolution        | Élevé         | Moyen      | Moyen      | Élevé  |
| Streaming XGBoost       | Faible        | Élevé      | Moyen      | Moyen  |
| Stratégies courtes      | Moyen         | Faible     | Faible     | Faible |

## 3.6 Contre-analyse

**Critique 1 :** Le sampling perd de l'information.
**Réponse :** L'information perdue est du bruit. Le volume sampling et
l'event sampling sont des techniques standard en microstructure. López de Prado
les recommande explicitement.

**Critique 2 :** Le GPU ne fonctionne qu'avec 1 worker.
**Réponse :** Vrai, mais 1 worker GPU est plus rapide que 3 workers CPU
pour les gros datasets. Le compromis : 1 worker GPU pour le 5M, 3 workers
CPU pour les autres TF.

**Critique 3 :** Les features multi-résolution augmentent la dimensionnalité.
**Réponse :** Vrai, mais le ratio observations/features reste excellent
(834k / 400 features = 2000:1). Le gain en signal compense l'augmentation
de dimensionnalité.

**Critique 4 :** Le 5M est trop bruité pour XGBoost.
**Réponse :** Le sampling intelligent résout ce problème. En gardant
seulement les barres significatives (volume ou événement), le signal/bruit
s'améliore considérablement.

## 3.7 Recommandation finale

**Immédiat :**

1. Volume-based sampling (garder barres volume > 1.5× moyenne)
2. XGBoost GPU pour le 5M (1 worker)
3. Horizons courts seulement (15m, 30m)

**Court terme :**
4. Event-based sampling (variation > 2σ)
5. Features microstructure (heure, volume relatif, VWAP)
6. FDR très strict (0.02) car beaucoup de données

**Moyen terme :**
7. Multi-résolution (5M + 15M + 1h features)
8. Streaming XGBoost pour les très gros datasets

**Insight clé de l'agent 5M :** Les dollar bars sont le changement unique
le plus impactant. Ils résolvent le bruit (returns plus IID), le volume
(réduction 5-20×) et la qualité du signal (chaque barre = même quantité
d'information) en une seule étape.

**Insight critique :** Si les dollar bars + bon modeling des coûts + features
microstructure ne produisent toujours pas d'Einhers, la conclusion honnête
est que l'alpha 5M n'existe pas après coûts pour ces actifs et horizons.
Le signal/bruit à 5M est extrêmement difficile (signal/bruit ∝ √T).

---

# SYNTHÈSE GLOBALE

## Conclusions des trois axes

### Diversité

Le problème principal est STRUCTUREL : XGBoost MSE converge toujours vers
les mêmes features à fort gain. La solution est de forcer la diversité par
construction (modèles par famille) et par sélection (MMR).

### 1D

Le problème principal est le RATIO observations/features : 12:1 est trop
bas. La solution est de réduire les features à 30-40 et d'adapter les
paramètres XGBoost.

### 5M

Le problème principal est le BRUIT et le COÛT computationnel. La solution
est le sampling intelligent (volume/event) et l'accélération GPU.

## Interactions

- Les modèles par famille (Diversité) aident aussi le 1D : moins de features
  par modèle = meilleur ratio observations/features
- Le sampling (5M) peut aussi être appliqué au 15M pour accélérer
- Le walk-forward (1D) devrait être appliqué à tous les TF
- La réduction Factor_* (Diversité) aide tous les TF

## Priorités

### Priorité 1 — Impact immédiat (cette semaine)

1. Modèles XGBoost par famille (11 modèles)
2. Décomposer Factor_* (modèle sans composites)
3. colsample_bytree=0.5
4. min_child_weight=5 pour les familles binaires
5. Réduire features à 30-40 pour le 1D
6. Volume sampling pour le 5M

### Priorité 2 — Impact élevé (2-4 semaines)

7. MMR pour la sélection
2. Walk-forward validation
3. XGBoost GPU pour le 5M
4. Features spécifiques par TF

### Priorité 3 — Architecture (1-2 mois)
 1. MAP-Elites archive
 2. Multi-résolution pour le 5M
 3. Cross-sectional modeling pour le 1D

## Architecture globale recommandée

```
                    ┌─────────────────────────────────────┐
                    │         EINHERJAR DISCOVERY          │
                    └─────────────────────────────────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    │                 │                  │
              ┌─────▼─────┐    ┌─────▼─────┐    ┌─────▼─────┐
              │    5M      │    │  15M/1H    │    │    1D     │
              │  Sampling  │    │  Standard  │    │  Reduced  │
              │  + GPU     │    │  Pipeline  │    │  Features │
              └─────┬─────┘    └─────┬─────┘    └─────┬─────┘
                    │                 │                  │
              ┌─────▼─────┐    ┌─────▼─────┐    ┌─────▼─────┐
              │  Per-Family│    │  Per-Family│    │  Per-Family│
              │  XGBoost   │    │  XGBoost   │    │  XGBoost   │
              │  (11 models)│   │  (11 models)│   │  (11 models)│
              └─────┬─────┘    └─────┬─────┘    └─────┬─────┘
                    │                 │                  │
              ┌─────▼─────┐    ┌─────▼─────┐    ┌─────▼─────┐
              │  SD +      │    │  SD +      │    │  SD +      │
              │  Pattern   │    │  Pattern   │    │  Pattern   │
              │  Miner     │    │  Miner     │    │  Miner     │
              └─────┬─────┘    └─────┬─────┘    └─────┬─────┘
                    │                 │                  │
                    └─────────────────┼─────────────────┘
                                      │
                              ┌───────▼───────┐
                              │  MMR Selection │
                              │  + Twin Merge  │
                              └───────┬───────┘
                                      │
                              ┌───────▼───────┐
                              │  BH Correction │
                              │  (Adaptive FDR)│
                              └───────┬───────┘
                                      │
                              ┌───────▼───────┐
                              │  Walk-Forward  │
                              │  Validation    │
                              └───────┬───────┘
                                      │
                              ┌───────▼───────┐
                              │  MAP-Elites    │
                              │  Archive       │
                              └───────┬───────┘
                                      │
                              ┌───────▼───────┐
                              │  CORPUS        │
                              └───────────────┘
```

## Prochaines expériences

1. **Test per-family XGBoost** sur 10 triplets (1h) : mesurer la diversité
   des features et le nombre d'Einhers admis
2. **Test Factor_* decomposition** sur 10 triplets : comparer avec et sans
   composites
3. **Test volume sampling** sur BTCUSD/5M : mesurer la réduction du dataset
   et la qualité du signal
4. **Test walk-forward** sur 10 triplets (1D) : comparer avec le split fixe
5. **Test MMR** sur le corpus existant : mesurer la réduction de jumeaux

---

# SOURCES

## Scientifique

1. Yun et al. (2025) : "QuantEvolve: Automating Quantitative Strategy Discovery
   through Multi-Agent Evolutionary Framework" — arXiv:2510.18569
   → MAP-Elites pour trading, 16 bins = optimal

2. Den et al. (2025) : "MAPLE: Efficient and Diverse Multi-Alpha Generation
   for Portfolio Construction" — arXiv:2607.24131
   → Diversity regularizer, 10-23% Sharpe improvement

3. Gu, Kelly & Xiu (2020) : "Empirical Asset Pricing via Machine Learning"
   → Cross-sectional models surperforment pour les données daily

4. López de Prado (2018) : "Advances in Financial Machine Learning"
   → Bar sampling, volume sampling, triple-barrier method

5. Ho (1998) : "The Random Subspace Method for Constructing Decision Forests"
   — IEEE TPAMI
   → colsample_bytree théorie fondamentale

6. Mouret & Clune (2015) : "Illuminating Search Spaces by Mapping Elites"
   → MAP-Elites original

7. Yuan & Lin (2006) : "Group Lasso"
   → Sélection de groupes de features

8. Carbonell & Goldstein (1998) : "The Use of MMR, Diversity-Based Reranking
   for Reordering Documents and Producing Summaries"
   → MMR original

## Technique

1. AlphaMix (KDD 2023) : MoE pour trading, 3 stages
2. TradingMoE (arXiv:2608.11785) : Query-Key router pour trading
3. MIGA (arXiv:2410.02241) : MoE avec Group Aggregation, +24% rendement
4. Council of Alphas (HackEurope 2026) : 4 spécialistes par famille
5. EMA-FS (arXiv:2606.26337) : Feature subsampling guidé
6. arXiv:2601.08121 : Warning sur colsample_bynode

## Type de source

- **Preuve scientifique** : QuantEvolve, MAPLE, Gu/Kelly/Xiu, Ho, Yuan/Lin
- **Observation expérimentale** : AlphaMix, MIGA, TradingMoE, Council of Alphas
- **Expérience communautaire** : López de Prado (volume sampling)
- **Inférence** : Recommandations spécifiques à Einherjar basées sur la littérature
- **Hypothèse** : Impact quantitatif des changements proposés (nécessite validation)
