# Einherjar — TypeGP : Troisième avis (analyse indépendante)

**Date** : 2026-08-14
**Auteur** : analyse forensique menée par un troisième agent, sur la base d'une
inspection directe des fichiers et des artefacts présents sur le disque, sans
modification d'aucun code.

> Ce document est le **troisième avis**. Deux audits parallèles existent déjà :
> `audits/AUDIT_FORENSIQUE_TYPEGP_2026-08-14.md` (audit A) et un second audit
> (audit B) fourni en session. Le présent document ajoute un point de vue
> indépendant, distinct à la fois dans sa thèse centrale et dans ses
> recommandations prioritaires.

---

## 1. Executive Summary

Les deux audits existants partagent une hypothèse commune que je considère comme
**non démontrée** : que le problème central est *soit* un problème de pipeline et
de traçabilité (audit B), *soit* un problème de choix de modèle — pire, un
signal faible qui justifierait de migrer vers XGBoost (audit A).

Mon constat, fondé sur inspection directe des artefacts, est plus en amont :

**La question prioritaire n'est ni « réparer TypeGP » ni « lancer XGBoost ».
C'est : y a-t-il un signal exploitable dans ces données avec ces labels, et où
se perd-il exactement ?**

Raisons matérielles, vérifiées sur le disque :

1. Le `selection.json` (12/08 00:34) montre que TypedGP a été sélectionné avec
   un score de **0.5**, un `admission_rate` de **0.0**, un `median_sharpe` de
   **0.0** et un `median_sharpe_all` de **-142.07**. Tous les subscores normés
   valent **0.5** (symptôme du comparateur min-max à générateur unique).
2. L'`admit_summary.json` montre **35 générés, 0 admis, 35 rejetés, tous par
   DSR_FAIL** (100 %).
3. Le `corpus.jsonl` (73 599 octets pour 3 Einhers) a été modifié le
   **11/08 22:28**, c'est-à-dire **avant** la sélection TypeGP (12/08 00:34).
   Le corpus actuel n'est donc **pas** issu du run qui a sélectionné TypeGP.
4. Les données `.npy` réelles existent et sont massives (ex. un seul `_X.npy`
   de 140 Mo ; plusieurs classes × 5 timeframes × ≥ 9 actifs). L'univers réel
   est important.
5. L'environnement Python du projet **n'est pas configuré** dans cette session :
   pas de `numpy`, pas de `polars` dans l'interpréteur shell. Aucune campagne
   réelle ne peut être relancée à l'identique aujourd'hui.

Conséquence stratégique : **tout résultat présent est soit non reproductible,
soit issu d'un run antérieur à la décision qui prétend le justifier.** Aucun
élément actuel ne permet d'affirmer que le moteur produit (ou non) des Einhers
fiables, ni que les données contiennent (ou non) un signal. Il faut d'abord
reconstruire une base de mesure fiable.

**Position de ce troisième avis** :
- Pas de pivot XGBoost maintenant (audit A) : ce serait optimiser du bruit plus
  vite sur un contrat non fiabilisé.
- Pas seulement « réparer le contrat » (audit B) : réparer sans d'abord avoir
  une mesure de prédictibilité de base revient à calibrer un instrument dont on
  ne sait pas s'il peut mesurer quelque chose.
- **Priorité 1 : construire le socle de mesure** (reproductibilité + floor
  trivial + test de signal) **avant** toute optimisation du moteur ou tout
  changement de modèle.

---

## 2. Faits vérifiés par inspection directe (matériel, pas théorique)

### 2.1 La sélection TypeGP est un artefact sans signification statistique

`outputs/selection.json` (12/08 00:34) :

```json
{
  "generator_name": "TypedGPGenerator",
  "selection_timestamp": "2026-08-12T00:34:38+00:00",
  "protocol": { "seed": 42, "splits": { "train_ratio": 0.6, "val_ratio": 0.2,
               "holdout_ratio": 0.2, "purge_window": 50, "embargo_bougies": 1 },
               "n_candidates": 100000, "n_eval_budget": 1200,
               "max_conditions": 4, "p_compound": 0.3,
               "assets": ["BTCUSD"], "timeframes": ["5m"] },
  "ranking_snapshot": {
    "generator_name": "TypedGPGenerator", "rank": 1, "score": 0.5,
    "n_generated": 50, "n_evaluated": 34, "n_passed_admission": 0,
    "admission_rate": 0.0, "median_sharpe": 0.0,
    "median_sharpe_all": -142.0688498159889, "n_distinct_features": 23,
    "semantic_coherence": 0.0,
    "subscores": { "sharpe": 0.0, "admission_rate": 0.0, "diversity": 23.0,
                   "coherence": 0.0, "norm_sharpe": 0.5,
                   "norm_admission_rate": 0.5, "norm_diversity": 0.5,
                   "norm_coherence": 0.5, "composite": 0.5 }
  }
}
```

Lecture critique :
- Le score composite de **0.5** provient de subscores tous normalisés à 0.5.
  C'est le comportement attendu d'une normalisation **min-max à un seul
  générateur** : avec un seul moteur dans le comparateur, tout point est à la
  fois le min et le max → 0.5 partout. **Le score 0.5 n'apporte aucune
  information.**
- `admission_rate = 0.0` et `median_sharpe = 0.0` : au moment de la sélection,
  le générateur n'avait **pas admis une seule hypothèse** et avait un Sharpe
  médian nul.
- `median_sharpe_all = -142.07` : les hypothèses évaluées ont une fitness
  médiane très négative. C'est un signe fort que la fitness évolutive était
  dominée par le bruit ou une anomalie (cf. section tasting), pas par du signal.
- `n_generated=50` mais `n_evaluated=34` : **34 seulement ont survécu assez
  longtemps pour être évaluées**. 16 disparaissent avant l'évaluation (contraintes
  de validité ? min_trades ? erreurs ?). Information aujourd'hui non persistée.
- Protocole : **mono-actif (BTCUSD), mono-timeframe (5m)**. La sélection ne
  couvre qu'un seul couple (actif, TF).

### 2.2 L'admission rejette 100 % par DSR

`outputs/admit_summary.json` :

```json
{ "n_generated": 35, "n_admitted": 0, "n_rejected": 35,
  "rejection_breakdown": { "DSR_FAIL": 35 }, "generator": "TypedGPGenerator" }
```

- 35 générés (cette fois), 35 rejetés, tous par **DSR_FAIL**.
- Le DSR (Deflated Sharpe Ratio, Bailey & López de Prado) pénalise la
  multiplicité des essais. 100 % de rejets DSR est un signal important : il
  signifie que, après correction du multi-tri, **aucune hypothèse ne se
  distingue significativement du hasard** sur la validation.
- Deux lectures possibles :
  - (a) Les hypothèses sont réellement du bruit → cohérent avec
    `median_sharpe_all=-142` et l'absence d'admission à la sélection.
  - (b) Il existe un bug de plongée (pricing du DSR, calcul du nombre d'essais,
    variance trop large, trades fantômes dans le tasting) qui *rend* le rejet
    systématique.
- Dans les deux cas, la donnée est cohérente : **rien n'admet**. Le goulot est
  réel et il est à l'admission, mais la cause profonde reste indéterminée faute
  d'instrumentation intermédiaire.

### 2.3 Le corpus actuel n'est PAS le produit du run TypeGP sélectionné

- `corpus.jsonl` modifié le **11/08 22:28**, 73 599 octets, **3 Einhers**.
- `selection.json` (TypedGP) du **12/08 00:34**.
- L'`admit_summary` du run actuel dit **0 admis**.

Donc : **les trois Einhers du corpus datent d'avant la sélection TypeGP.**
Ils proviennent de campagnes antérieures (traces `campaign_btc_20260810*.log`,
`run_20260810*`...). Leur existence **ne démontre rien** sur le comportement du
moteur actuel. C'est un artefact historique, pas une preuve de fonctionnement.
C'est la confirmation matérielle de la « rupture de traçabilité » évoquée par
l'audit B, mais avec une conséquence plus forte : on ne peut même pas
reproduire les 3 Einhers, ni savoir avec quels paramètres ils ont été produits.

### 2.4 Les Einhers sont massifs

- 73 599 octets / 3 Einhers ≈ 24,5 Ko par Einher en moyenne.
- L'audit B a mesuré des profondeurs 4, 7 et 8 sur les artefacts, avec 26, 56
  et 61 nœuds, et des tailles JSON de 1 490 à 3 427 caractères. Coherent avec
  un bloat.
- La sérialisation « enrichie » (ret_series, fingerprint, etc.) gonfle
  l'empreinte au-delà de la seule condition.

### 2.5 L'univers de données réel est grand

- Classes présentes dans `D:/midas_v2/midasV3/src/data/compiled/` : `crypto`,
  `forex`, `indices`, `stocks_growth`, `stocks_tech`, `stocks_value`,
  `commodities` — chacune avec 5 timeframes (5m, 15m, 1h, 4h, 1d).
- Exemple : `crypto/15m/ADAUSD_X.npy` = 140 Mo ; présents `_X`, `_ts`,
  `_Y_dir`, `_Y_hor`, `_Y_ret`. Crypto/15m contient au moins 9 actifs.
- Les labels (`Y_dir`, `Y_ret`, `Y_hor`) sont **pré-calculés offline par
  midasV3** et chargés tels quels par `npy_real_loader`. ⚠ À vérifier :
  l'horizon et les coûts qui ont généré ces labels correspondent-ils à ceux
  que réutilise l'évaluation TypeGP ? Une désynchronisation possible est un
  point de vigilance non tranché.

### 2.6 L'environnement n'est pas reproductible en l'état

- L'interpréteur shell (`python`) est celui de l'agent (3.11.15), **sans**
  `numpy`, `polars` ni venv projet local.
- Le README indique que le moteur requiert `duckdb` et `polars`.
- Conclusion : **aucune campagne réelle ne tourne aujourd'hui dans un
  environnement documenté et reproductible.** C'est un blocage opérationnel
  préalable à toute mesure fiable.

---

## 3. Pipeline réellement exécutée

### 3.1 Chemin de commande effectif

Le point d'entrée unique documenté est :
`python -m einherjar.research.discovery` (module `discovery.py`, 1 340 lignes).
Modes observés dans le README et confirmés par la structure :

```
discovery run|pipeline   → enchaîne engine → baselines → compare → select → refine → admit
discovery compare        → 6 générateurs partagent un budget d'éval et produisent un ranking
discovery select         → installe le « gagnant » dans selection.json
discovery refine         → raffine les top-N (déprécié, BeamRefiner)
discovery admit          → DSR/PBO/CI/n_trades/cross-asset/diversité → corpus ou archive
discovery holdout        → évaluation sacrée une seule fois
```

### 3.2 Le flux de données réel d'un run `admit`

1. Charger un seul `Universe` (actif + timeframe), OHLCV/features depuis les
   `.npy` MIDAS V3 via `npy_real_loader` + `duckdb`.
2. Découper en splits (train 60 % / val 20 % / holdout 20 %) avec purge et
   embargo.
3. Instancier le générateur (ici TypedGPGenerator) **une nouvelle fois**,
   distincte de celle qui a été comparée.
4. Générer une population, l'évaluer (tasting sur train, fitness), puis
   calibration SL/TP + évaluation sur validation.
5. Admission multi-critères sur les « finalistes » régénérés.
6. Admis → corpus ; rejets → archive.

### 3.3 Point de rupture confirmé : la sélection et l'admission n'opèrent pas sur le même jeu

Ce point est DIRECTEMENT confirmé par les artefacts :
- Le `ranking_snapshot` dit `n_generated=50, n_evaluated=34`, avec `score=0.5`
  et `admission_rate=0.0`.
- L'`admit_summary` dit `n_generated=35, n_rejected=35`.

**34 évalués à la sélection ≠ 35 générés à l'admission.** Deux nombres
différents, deux populations différentes, probablement deux instanciations et
deux seeds/protocoles (régénération). `selection.json` contient bien un
"protocol" (n_candidates=100000, max_conditions=4, cooldown_k=5...), mais rien
ne garantit que l'instanciation d'admission réutilise exactement ce protocole.

Conséquence : **le PBO/DSR de l'admission porte sur des finalistes régénérés,
pas sur l'ensemble réellement exposé à la recherche.** Si la recherche a
sélectionné des lignées sur la base d'une fitness (éventuellement corrompue),
puis que l'admission évalue *d'autres* candidats, la preuve statistique ne
s'applique pas aux individus qui ont réellement piloté la recherche.

### 3.4 Ce qui manque pour reconstruire la chaîne quantitative demandée

La chaîne voulue est :

```
N générés → N valides → N évalués → N survivants → N distincts
→ N validés → N candidats admission → N Einhers
```

Aujourd'hui, seuls quatre chiffres éparpillés existent :
- `n_generated` (sélection : 50 ; admission : 35)
- `n_evaluated` (34)
- `n_rejected` (35, tous DSR)
- `n_admitted` (0 / 3 historiques)

**Il manque partout** : le nombre de valides, les fitness de chaque individu,
les parents, les opérateurs appliqués, le compteur Long/Short, les features
utilisées par individu, les candidats écartés par chaque porte d'admission, et
les raisons d'arrêt avant évaluation. Aucun de ces nombres n'est persisté dans
le flux de sortie actuel (ni selection, ni admit_summary, ni corpus). Sans cette
instrumentation, **le goulot ne peut pas être localisé quantitativement** : on
constate seulement que rien n'aboutit, pas *où* cela se perd.

### 3.5 Le sens de lecture que je propose

Plutôt que de supposer un goulot précis, on peut classer les hypothèses de
perte par ordre de vérifabilité :

1. **Pertes avant évaluation** (16/50 = 32 % à la sélection) : contraintes de
   validité, min_trades, erreurs silencieuses. Vérifiable seulement en
   instrumentant.
2. **Fitness évolutive dominée par le bruit** (`median_sharpe_all=-142`).
   Si la fitness du tasting est corrompue (voir § tasting), l'évolution finance
   des mauvaises lignées dès le début. C'est une perte d'opportunités, pas une
   perte d'individus.
3. **Admission DSR 100 %** : soit bruit réel, soit bug de plongée. À
   discriminer par le test de signal (§ 5).
4. **Pas de lignée Long survivante** : hypothèse non vérifiée par manque de
   compteurs. Ne pas conclure à une propriété de marché (cf. § 6).

### 3.6 Conclusion de section

Le pipeline s'exécute du point de vue de la mécanique (fichiers produits,
aucun crash apparent), mais **sa sortie n'est ni reproductible, ni traçable,
ni représentative de la recherche qui l'a précédée.** La phrase « le pipeline
s'exécute sans erreur ≠ le moteur fonctionne » est ici exactement vérifiée : le
pipeline a bien produit des fichiers, mais ils ne permettent ni d'expliquer les
3 Einhers, ni de justifier la sélection TypeGP.

---

## 4. Analyse TypeGP (ce que le moteur fait réellement)

Sur la base de l'architecture documentée (README, STATUS.md) et de la structure
du générateur (`generators/algorithms.py`, `TypedGPGenerator`) :

### 4.1 Ce qui est réellement implémenté

- **Initialisation** : mélange de `grow` et `full` (Koza 1992 / Montana 1995),
  direction Long/Short tirée uniformément à l'état initial.
- **Sélection** : tournoi de taille k=3.
- **Crossover** : sous-arbre, « type-preserving ».
- **Mutation** : sous-arbre.
- **Élitisme** : les meilleurs rescapent.
- **Calibration / fitness** : calibration SL/TP sur 80 % du train, fitness sur
  les 20 % restants.

### 4.2 Limites structurelles (au-delà du « typage booléen »)

Deux limites importantes ressortent, au-delà de ce que mentionnent les audits A
et B :

1. **Ce n'est pas un STGP complet.** Le « typage » se réduit à `atomic` vs
   `compound`. Tout arbre retourne une condition booléenne. Il n'y a pas de
   composition numérique typée multiple. Conséquence : l'espace des programmes
   réellement exprimables est une *famille restreinte* de règles de décision, ce
   qui borne d'emblée la richesse des stratégies que la recherche *peut* trouver
   — mais ce n'est pas en soi un bug, c'est un choix de représentation
   (conditions combinables, cohérent avec `max_conditions=4`).

2. **Le crossover sans budget de profondeur post-greffe** : un sous-arbre profond
   peut être greffé à un point peu profond → bloat. Les profondeurs 7-8 observées
   sur les artefacts, avec 56-61 nœuds, sont cohérentes avec ce mécanisme. Le
   bloat, à lui seul, n'est pas la cause de l'admission nulle, mais il rend le
   résultat inexploitable et peut corréler avec une exploration déséquilibrée
   (les arbres dérivants s'empilent sur des structures massives qui « écrasent »
   le signal).

### 4.3 Question centrale : le moteur explore-t-il vraiment l'espace ?

**Probablement pas aussi largement qu'annoncé, mais c'est un problème de
mesure, pas une certitude.**

- Aucun compteur de diversité de population, de distribution des features, de
  profondeur, d'opérateurs, de Long/Short n'est persisté au fil de
  l'évolution.
- La déduplication n'est appliquée qu'à la fin de la génération (et/ou à
  l'admission), pas comme pression continue de diversité.
- La fitness unique (Sharpe de tasting) crée une pression sélective forte et
  unique ; toutes les lignées qui ne grimpent pas sur cette seule métrique (même
  bruitée) sont éliminées.

Donc, plutôt que d'affirmer que « TypeGP explore mal », je formule ainsi :
**l'architecture ne fournit aucune preuve de la richesse de l'exploration, et
la pression sélective repose sur une seule métrique possiblement bruitée.** La
faible diversité est *expliquée de façon plausible* par ce design, mais n'est
*prouvée* par aucun artefact.

---

## 5. Le point que les deux autres audits n'ont pas traité : et s'il n'y a pas de signal ?

C'est ma contribution principale. Les audits A et B débattent de *comment*
extraire un signal (GP vs XGBoost, réparer vs pivoter). **Aucun des deux ne pose
la question préalable : comment sait-on qu'il existe un signal exploitable dans
ces features + ces labels + ce contexte de coûts ?**

### 5.1 Les chiffres sont compatibles avec « pas de signal »

- `median_sharpe_all = -142.07` sur 34 hypothèses : une fitness évolutive
  médiane extrêmement négative est ce qu'on attend **si la plupart des règles
  aléatoires génèrent des signaux qui ressemblent à du bruit ou à des
  configurations perdantes systématiques** (ex. Short permanent contre-tendance,
  coûts non couverts, prix mal alignés).
- `DSR_FAIL` sur 35/35 : exactement ce qu'on attend si aucune stratégie ne
  surpasse significativement le hasard après correction multi-tri.
- Ces deux phénomènes sont **réunis par une seule explication simple** :
  l'optimisation travaille sur du bruit.

### 5.2 Pourquoi c'est important pour la décision XGBoost

Si le problème est « pas de signal dans les *labels actuels* », alors :
- **Réparer TypeGP ne donnera rien** de plus qu'un autre corpus vide.
- **Lancer XGBoost sur les mêmes labels produira pareil — plus vite.**
  XGBoost optimisera le bruit au moins aussi efficacement que TypeGP, et
  générera des milliers de règles surajustées. C'est exactement le risque que
  l'audit B souligne, et je le renforce : XGBoost n'est **pas** une solution
  au problème du signal ; c'est une façon de *fabriquer plus vite* des candidats,
  qu'ils soient bruités ou non.

L'argument « 28 actifs × 15 ans = plus de données = plus de signal » (audit A)
est **trop optimiste** : plus de lignes ne crée pas du signal, surtout si la
variable cible (`Y_dir` pré-calculée, éventuellement avec un horizon/coût
incompatible) est elle-même bruitée ou désynchronisée.

### 5.3 Ce qu'il faut mesurer avant tout : un floor de prédictibilité

Je propose un test de signal minimal, exécutable rapidement et sans toucher au
moteur de recherche :

1. **Baseline triviale** : buy-and-hold (entrer en Long en début de val, sortir à
   la fin) et un « toujours Short ». Donne le Sharpe de référence du marché sur
   la période — indispensable pour interpréter tout Sharpe ultérieur.
2. **Modèle linéaire / arbres de décision** (logistic, RF peu profond) entraînés
   sur les features → `Y_dir` sur un split train, score sur l'autre. Si l'AUC /
   la précision est proche du hasard (≈ 0.5), **il n'y a probablement pas de
   signal prédictif dans ces features**. C'est une mesure directe de la qualité
   du problème, avant toute GP.
3. **Permutation test** : mélanger les labels, relancer l'évaluation d'un même
   type de règle, comparer. Calibre le DSR et PBO contre la distribution nulle
   réelle du problème (pas une hypothèse théorique).
4. **Alignement label vs évaluation** : rouvrir `npy_real_loader` + `Y_*` et
   vérifier manuellement (sur quelques indices) que l'horizon, le prix d'entrée,
   les coûts et l'index temporel correspondent à ce que le moteur suppose. Un
   décalage d'index (look-ahead/leakage) ou d'horizon invaliderait *tout* le
   reste — c'est le test le plus rentable, car il peut à lui seul expliquer
   `median_sharpe_all=-142` et DSR 100 %.

Ces quatre mesures ne modifient **aucun** code de recherche ; ce sont des
diagnostics autonomes. Elles donneront la réponse à la question que les deux
audits éludent : **le moteur échoue-t-il parce qu'il est cassé, ou parce qu'il
n'y a rien de fiable à trouver ?**

### 5.4 Verdict intermédiaire

- Si le test de signal est nettement positif → le problème est *technique*
  (pipeline/tasting/admission) ; la voie B (« réparer ») devient la bonne et
  XGBoost n'est même pas nécessaire.
- Si le test est au niveau du hasard → le problème est le *signal* ; alors la
  priorité n'est ni GP ni XGBoost, mais redéfinir les labels / coûts / horizon /
  features, avant de reconsidérer la génération.
- Si la désynchronisation label/éval est confirmée → c'est un bug critique
  quasi-certain (le nº 1 du tableau), à corriger avant toute autre chose.

Dans les trois cas, **la prochaine action n'est pas d'ajouter un modèle**.