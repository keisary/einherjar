# Planification — Techniques pour résoudre la problématique d'Einherjar

> **Date** : 2026-08-17
> **Statut** : recherche / planification (aucune implémentation)
> **Objet** : recenser, sourcer et expliquer les techniques / algorithmes / systèmes
> permettant de trouver des **combinaisons de features rentables** dans un espace
> de combinaisons limité, dans notre contexte financier.

---

## Rappel du problème

**Problématique générale d'Einherjar :**
> Comment trouver, dans un espace de combinaisons de features **borné**, les
> combinaisons **rentables** — lesquelles parmi toutes les combinaisons possibles
> rapportent de l'argent.

**Produit visé :** un **corpus d'Einhers** (stratégies autonomes), chacune
caractérisée par : (1) condition de déclenchement, (2) direction, (3) amplitude,
(4) univers, (5) métriques (win rate, sharpe, CAGR, maxDD…).

**Contexte :** système de trading autonome ; moteur de recherche de stratégies ;
données OHLCV + ~218 features, multi-actifs / multi-timeframes (15m, 1h, 4h…) ;
validation hors-échantillon indispensable.

La problématique se décompose en **3 sous-problèmes**, qui demandent des
techniques différentes :

1. **Explorer** l'espace de combinaisons (chercher intelligemment).
2. **Borner** cet espace (sélectionner les features pertinentes).
3. **Valider sans se faire berner** (anti-overfitting) **et diversifier** (trouver
   plusieurs combinaisons rentables indépendantes).

---

## A. Explorer l'espace de combinaisons de features

### A1. Programmation Génétique (GP) / STGP
- **Ce que c'est** : fait évoluer des expressions (arbres) via sélection, crossover,
  mutation. **STGP** (Strongly-Typed GP) contraint le typage des nœuds (features
  continues vs booléennes, opérateurs compatibles).
- **Lien avec notre problème** : cadre naturel pour chercher des combinaisons de
  features — une "combinaison rentable" = un arbre de conditions sur des features.
  Explore directement l'espace des expressions, pas juste des paramètres.
- **Source** : Koza, J., *Genetic Programming: On the Programming of Computers by
  Means of Natural Selection*, MIT Press, 1992. — Montana, D., "Strongly Typed
  Genetic Programming", *Evolutionary Computation* 3(2), 1995.
- **Adéquation** : très élevée (c'était l'approche déjà utilisée). Limites connues :
  bloat (arbres qui gonflent), convergence prématurée vers des jumeaux.

### A2. Évolution Grammaticale (Grammatical Evolution, GE)
- **Ce que c'est** : encode un génome (séquence) traduit en programme via une
  grammaire **BNF** (Backus-Naur). Découple génotype (facile à faire évoluer) du
  phénotype (la stratégie).
- **Lien** : permet de contraindre toute la syntaxe d'une combinaison de features
  (plus expressive que de simples seuils : transitions, fenêtres, croisements de
  features) tout en gardant une évolution simple.
- **Source** : O'Neill, M. & Ryan, C., *Grammatical Evolution*, Springer, 2003.
- **Adéquation** : élevée — particulièrement si le STGP devient limité en
  expressivité. Candidat sérieux pour enrichir l'espace de combinaisons.

### A3. Algorithmes évolutionnaires multi-objectifs — NSGA-II
- **Ce que c'est** : optimise plusieurs objectifs à la fois (rendement, drawdown,
  nombre de trades…) et retourne un **front de Pareto** de solutions, pas une seule.
- **Lien** : une combinaison rentable n'est pas définie par un seul critère ; on veut
  les combos qui dominent sur plusieurs axes. NSGA-II gère la diversité via le tri
  non-dominé + crowding distance.
- **Source** : Deb, K., Pratap, A., Agarwal, S., Meyarivan, T., "A Fast and Elitist
  Multiobjective Genetic Algorithm: NSGA-II", *IEEE Trans. Evolutionary Computation*
  6(2), 2002.
- **Adéquation** : élevée — aligné sur "trouver plusieurs combinaisons différentes
  qui rapportent", avec une vraie gestion de la diversité.

### A4. Stratégies d'évolution — CMA-ES
- **Ce que c'est** : optimisation continue par adaptation de la covariance
  (population de vecteurs de paramètres réels, pas d'arbres).
- **Lien** : utile pour **affiner** les paramètres d'une combinaison (seuils, poids,
  fenêtres) une fois la structure trouvée.
- **Source** : Hansen, N. & Ostermeier, A., "Completely Derandomized Self-Adaptation
  in Evolution Strategies", *Evolutionary Computation* 9(2), 2001.
- **Adéquation** : moyenne — complémentaire (fine-tuning), pas le moteur principal.

### A5. Optimisation Bayésienne (Bayesian Optimization, BO)
- **Ce que c'est** : modèle de substitution (souvent un GP) + fonction d'acquisition
  (Expected Improvement, UCB) pour décider où évaluer ensuite, en minimisant le
  nombre d'évaluations coûteuses.
- **Lien** : précieuse quand chaque évaluation d'une combinaison coûte cher (un
  backtest). Très économe en évaluations.
- **Source** : Snoek, J., Larochelle, H., Adams, R.P., "Practical Bayesian
  Optimization of Machine Learning Algorithms", NeurIPS, 2012. — Shahriari, B. et
  al., "Taking the Human Out of the Loop", *Proc. IEEE* 104(1), 2016.
- **Adéquation** : élevée pour les combos coûteuses à évaluer, mais moins adaptée
  aux espaces discrets/énormes que le GP ; bon complément.

### A6. Recherche aléatoire / quasi-aléatoire (baseline)
- **Ce que c'est** : échantillonner l'espace (idéalement quasi-Monte Carlo, suite de
  Sobol) plutôt qu'une grille exhaustive.
- **Lien** : référence à TOUJOURS battre pour prouver qu'un algorithme fait mieux
  que le hasard. Sur certains espaces de combinaisons, le random search est
  étonnamment bon.
- **Source** : Bergstra, J. & Bengio, Y., "Random Search for Hyper-Parameter
  Optimization", *JMLR* 13, 2012.
- **Adéquation** : obligatoire comme baseline de validation.

---

## B. Borner / réduire l'espace des features

### B1. Sélection de features par filtres (corrélation, information mutuelle)
- **Ce que c'est** : éliminer les features redondantes ou non informatives AVANT la
  recherche, pour réduire l'explosion combinatoire.
- **Lien** : moins de features = un espace "limit" mais utile ; on ne cherche plus
  dans le bruit.
- **Source** : Peng, H., Long, F., Ding, C., "Feature Selection Based on Mutual
  Information: Max-Dependency, Max-Relevance, Min-Redundancy", *IEEE TPAMI* 27(8),
  2005.
- **Adéquation** : élevée, étape de pré-traitement peu coûteuse.

### B2. mRMR / méthodes enveloppes et embedded (LASSO, Boruta)
- **Ce que c'est** : mRMR maximise la pertinence et minimise la redondance ; LASSO
  pénalise pour pousser à zéro les features inutiles ; Boruta compare aux "shadow
  features" aléatoires.
- **Lien** : compléments pour borner l'espace avant le GP.
- **Source** : Kursa, M. & Rudnicki, W., "Feature Selection with the Boruta Package",
  *Journal of Statistical Software* 36(11), 2010. — Tibshirani, R., "Regression
  Shrinkage and Selection via the Lasso", *JRSS-B* 58, 1996.
- **Adéquation** : élevée MAIS piège : les combos rentables sont souvent
  non-linéaires/contextuelles, la sélection linéaire peut les rater. À utiliser en
  prune, pas comme vérité absolue.

---

## C. Valider sans se faire berner (le cœur du problème)

> Sans une validation rigoureuse, "une combinaison que rapporte en backtest" n'est
> qu'une illusion. C'est LE sous-problème décisif.

### C1. Purging, embargo et split hors-échantillon (walk-forward)
- **Ce que c'est** : découper l'historique en train/val/holdout ordonnés dans le
  temps ; **purger** les labels qui débordent sur le jeu suivant ; **embargo** après
  chaque frontière (évite les fuites de features lissées).
- **Lien** : garantit qu'on juge une combinaison sur des données qu'elle n'a jamais
  vues. Le socle : sans ça, tout le reste est faux.
- **Source** : López de Prado, M., *Advances in Financial Machine Learning*, Wiley,
  2018 — ch. 7 (purging/embargo) et ch. 12 (CPCV).
- **Adéquation** : déjà la base du système (splits 60/20/20 + purge + embargo).

### C2. Deflated Sharpe Ratio (DSR)
- **Ce que c'est** : corrige le Sharpe observé pour (a) le nombre de combos testées
  (multiple testing) et (b) la non-normalité (skew/kurtosis). Probabilité que le vrai
  Sharpe soit > 0.
- **Lien** : répond au risque "j'ai testé 1000 combos, certaines semblent rentables
  par hasard".
- **Source** : Bailey, D. & López de Prado, M., "The Deflated Sharpe Ratio:
  Correcting for Selection Bias, Backtest Overfitting and Non-Normality", *Journal of
  Portfolio Management*, 2014.
- **Adéquation** : directement nôtre (l'un des critères d'admission).

### C3. PBO / CPCV (Probability of Backtest Overfitting)
- **Ce que c'est** : via une validation croisée combinatoire purgée (CPCV), mesure la
  probabilité que la combinaison choisie in-sample soit dépassée out-of-sample.
- **Lien** : meilleure mesure actuelle du risque d'overfitting d'une recherche de
  stratégies.
- **Source** : Bailey, D., Borwein, J., López de Prado, M., Zhu, Q., "The Probability
  of Backtest Overfitting", *Journal of Computational Finance*, 2017. — López de
  Prado (AFML), 2018, ch. 12.
- **Adéquation** : directement nôtre (critère d'admission).

### C4. Tests de data snooping (White Reality Check, Hansen SPA)
- **Ce que c'est** : tests statistiques formels vérifiant que la meilleure stratégie
  bat réellement un benchmark, en tenant compte de la recherche dans un grand nombre
  de stratégies.
- **Lien** : p-value honnête de l'avantage d'une combo face au multiple-testing.
- **Source** : White, H., "A Reality Check for Data Snooping", *Econometrica* 68(5),
  2000. — Hansen, P.R., "A Test for Superior Predictive Ability", *JBES*, 2005.
- **Adéquation** : élevée, complément rigoureux au DSR/PBO.

### C5. Contrôle du FDR (Benjamini–Hochberg) / Bonferroni
- **Ce que c'est** : corrections du multiple testing quand on retient PLUSIEURS combos
  à la fois (notre but = un corpus).
- **Lien** : contrôle le taux de fausses découvertes parmi les combos retenues.
- **Source** : Benjamini, Y. & Hochberg, Y., "Controlling the False Discovery Rate: A
  Practical and Powerful Approach to Multiple Testing", *JRSS-B* 57(1), 1995.
- **Adéquation** : élevée — plus adapté qu'une correction stricte quand on veut garder
  beaucoup de combos rentables.

### C6. Block bootstrap pour intervalles de confiance
- **Ce que c'est** : ré-échantillonner les trades/blocs pour estimer l'incertitude du
  Sharpe et du rendement total, en respectant l'autocorrélation (blocs contigus, pas
  i.i.d.).
- **Lien** : IC de la rentabilité d'une combo ; on n'admet que si la borne basse est
  positive.
- **Source** : Politis, D. & Romano, J., "The Stationary Bootstrap", *JASA*, 1994. —
  López de Prado (AFML), 2018, ch. 13.
- **Adéquation** : directement nôtre (critères bootstrap CI).

---

## D. Diversifier — trouver PLUSIEURS combinaisons rentables

> L'objectif n'est pas UNE combinaison, c'est un **corpus** de combos rentables et
> indépendantes.

### D1. MAP-Elites (Quality-Diversity)
- **Ce que c'est** : découpe l'espace comportemental en niches (direction × fréquence
  × style…) et garde le meilleur individu par niche. Population finale diversifiée,
  un bon individu par style.
- **Lien** : idéal pour produire un corpus : force à trouver une combo rentable par
  "case comportementale" au lieu de laisser tout converger vers un seul champion.
- **Source** : Mouret, J.-B. & Clune, J., "Illuminating Search Spaces by Mapping
  Elites", arXiv:1504.04909, 2015.
- **Adéquation** : très élevée — déjà dans le pipeline ; la technique-clé du côté
  "corpus".

### D2. Novelty Search
- **Ce que c'est** : récompense la nouveauté du comportement (pas seulement la
  fitness).
- **Lien** : en complément de MAP-Elites, évite de re-trouver les mêmes combos.
- **Source** : Lehman, J. & Stanley, K., "Exploiting Open-Endedness to Solve Problems
  Through the Search for Novelty", ALIFE, 2008.
- **Adéquation** : moyenne, en complément.

---

## E. Système complet (architecture globale)

### E1. Pipeline d'alpha discovery (recherche → validation → corpus → réutilisation)
- **Ce que c'est** : enchaîner : génération de combos → évaluation purgée/embargoed →
  validation anti-overfitting (DSR/PBO/bootstrap) → dédup comportementale → admission
  au corpus → une seule passe sur le "holdout sacré" → mise en production des
  survivants.
- **Lien** : c'est exactement le pipeline visé. La "recette" contre les fausses
  découvertes est une méthodologie complète, pas un seul outil.
- **Source** : López de Prado, M., *Advances in Financial Machine Learning*, Wiley,
  2018 — "le backtest ne suffit pas, il faut prouver la non-fragilité". — Harvey, C.
  & Liu, Y., "Backtesting", *Journal of Portfolio Management*, 2015.
- **Adéquation** : le squelette du système.

---

## Étape A — Décision d'exploration (recommandation argumentée)

Contexte : espace de combinaisons de features borné, objectif = **corpus** (plusieurs
combos rentables et diverses), validation coûteuse en aval, délai serré (~3 semaines).

### Recommandé (noyau)
1. **STGP — générateur cœur (A1)**
   - Notre unité de recherche est un arbre de conditions sur des features : l'espace
     d'intérêt EST un espace d'arbres symboliques. STGP l'explore directement et
     **invente la structure** (quelle combinaison), ce que CMA-ES/BO ne savent pas faire.
   - Typage fort → évite les combinaisons absurdes, espace propre et interprétable.
   - Déjà utilisé par le projet → risque d'implémentation faible (important pour le délai).
2. **Random search — baseline obligatoire (A6)**
   - Preuve que le STGP bat le hasard. Coût quasi nul, référence honnête.
3. **MAP-Elites — couche de diversité (D1)**
   - Range chaque bon individu dans une niche comportementale et en garde un par niche
     → produit structurellement un **corpus diversifié** (exactement notre sortie voulue).
   - Sépare la diversité **comportementale** (ce qui compte pour un corpus), que
     NSGA-II (diversité en espace d'objectifs) ne gère pas nativement.

### Optionnel (affinage)
4. **Optimisation Bayésienne — affinage des finalistes (A5)**
   - Après que STGP+MAP-Elites ont trouvé les structures, ajuster leurs seuils avec très
     peu d'évaluations (frugal, donne une incertitude). Juste avant la validation coûteuse.
   - Ne peut PAS inventer la structure → uniquement en aval, pas comme moteur principal.

### Différés / écartés (justification)
- **GE (A2) — différé** : plus expressif mais plus cher à implémenter/régler ; notre
  espace borné est couvert par STGP. Réintroduire seulement si on élargit le langage
  des combinaisons.
- **NSGA-II (A3) — différé** : pertinent uniquement si compromis multi-objectif explicite
  (ex. CAGR/drawdown). MAP-Elites suffit pour la diversité. Phase 2 éventuelle.
- **CMA-ES (A4) — écarté** : ne peut pas inventer la structure ; pour l'ajustement continu
  BO est plus adapté (frugalité + incertitude).

### Combinaison (chaîne)
```
Random search ─► baseline (preuve anti-hasard)
STGP (arbres) ─► population de combos ; fitness de recherche CHEAP (surrogate)
MAP-Elites ─► 1 bon individu par niche → pool diversifié de structures
[OPTION] BO : affinage des seuils des finalistes
validation COÛTEUSE (DSR / PBO / bootstrap) → ADMISSION → CORPUS
```
**Point de conception clé** : deux niveaux d'évaluation — une fitness bon marché pendant
l'exploration (STGP/MAP-Elites), et la validation rigoureuse coûteuse (DSR/PBO/bootstrap)
uniquement sur les finalistes. C'est ce qui permet de faire une recherche large + une
validation stricte sans brûler le budget sur tout l'espace.

---

## Étape B — Décision de bornage de l'espace de features

Contexte : 218 features, objectif = combos **non-linéaires** (arbres de conditions),
donc un choix délicat entre « réduire » et « ne pas casser les combos ».

### Point conceptuel
L'espace de combinaisons est déjà borné par la STRUCTURE de la recherche (profondeur
d'arbre, opérateurs, pool de seuils, universe). Réduire les features sert à (a) retirer
le bruit/redondance pour une convergence plus rapide et propre, (b) rendre l'espace
« limité » explicite et défendable. Mais **mal réduit, on détruit des combos qui
n'existent que par interaction** — c'est le piège de l'étape B.

### Recommandé
1. **Présélection « safe » (sans perte de signal) — à faire**
   - Élaguer les features invalides/inutilisables : variance quasi nulle, taux de valeurs
     manquantes trop élevé, features qui ne déclenchent jamais, NaN systématiques.
   - Dédoublonner par corrélation : regrouper en clusters, garder un représentant par
     cluster (ex. variance max). Supprime la REDONDANCE sans perdre d'information.
     (La taxonomie avait déjà exclu 28 features pour collinéarité.)
   - C'est le seul « borner » 100% sûr : réduit l'explosion combinatoire sans risque.
2. **Classement par information mutuelle / mRMR comme GUIDE, pas couperet — recommandé**
   - Calculer l'IM (ou mRMR) de chaque feature avec la cible (mouvement futur / amplitude)
     pour **prioriser**. Garder un **gros pool** (ex. top ~80%) ; la recherche garde accès
     au reste.
   - Pourquoi : l'IM est non-linéaire (détecte des relations que la corrélation linéaire
     rate) → meilleur guide ; mais on ne fait JAMAIS de coupe dure agressive (une feature
     isolée faible peut être décisive en combinaison).
3. **Borner aussi le LANGAGE — recommandé**
   - L'espace est borné par les opérateurs, les transformations autorisées, max_depth et
     le pool de seuils. Fixer ces bornes = rendre « l'espace limité » concret, versionné,
     défendable. C'est un artefact de config à part entière.

### Écarté / à ne pas utiliser comme couperet
- **LASSO (B2)** : modèle linéaire → une feature inutile seule mais puissante en
  combinaison seuillée est pénalisée à zéro. Déciderait à tort de l'espace. → écarté comme
  règle de décision.
- **Boruta (B2)** : non-linéaire (random forest), acceptable comme **crible grossier
  optionnel**, mais juge encore la pertinence isolée, pas la valeur en combinaison. Pas une
  autorité.

**Règle de conduite** : on élimine ce qui est invalid/redondant, on priorise pour guider,
on ne supprime pas ce qui pourrait être utile en combinaison.

### Combinaison avec A
B définit le **pool de features** que le STGP (A1) peut tirer ; résultat = un ensemble
versionné de features + bornes de langage = « l'espace limité » exact que le moteur
explore. Réduit le bruit/redondance pour des combos plus propres et des niches MAP-Elites
plus interprétables, sans casser les combos d'interaction.

---

## Étape C — Décision de validation (anti-fausses-découvertes)

Contexte : on teste beaucoup de combos (STGP), on veut sortir un **corpus** de plusieurs
combos rentables, validation coûteuse, délai ~3 semaines.

### Recommandé (noyau)
1. **C1 — Socle temporel purgé/embargoed + holdout sacré (fondamental)**
   - train / val / holdout ordonnés, **purgé** (labels qui débordent) et **embargoed**
     (fuites de features lissées). Le val sert à sélectionner/admettre ; le **holdout
     sacré** sert UNE seule fois, à la fin, comme vraie preuve.
   - Sans structure temporelle honnête, toutes les autres métriques sont fausses. Socle.
   - CPCV / walk-forward multiple : option pour plus de robustesse (plus coûteux).
2. **C6 — Block bootstrap CI (Sharpe + ret) comme test primaire**
   - On juge l'IC de chaque combo : **admis seulement si borne basse du CI > 0**. Le block
     bootstrap respecte l'autocorrélation (blocs contigus).
   - Garde-fou direct contre « la combo rentable par hasard », donne une incertitude honnête.
3. **C2 — DSR (Deflated Sharpe Ratio)**
   - Corrige le Sharpe pour le **nombre d'essais testés** (multiple testing) et la
     **non-normalité** (skew/kurtosis). Répond au risque « chez 500 combos, certaines
     semblent bonnes par chance ».
   - Complémentaire du bootstrap : CI juge UNE combo, DSR la juge en sachant qu'elle vient
     d'un pool testé en masse.
4. **C5 — Contrôle FDR (Benjamini–Hochberg) au niveau du CORPUS**
   - On garde plusieurs combos à la fois → contrôle du **taux de fausses découvertes
     global** (pas combo par combo). Adapté pour garder beaucoup de résultats (contrairement
     à Bonferroni, trop strict).

### Plus lourd / différé
5. **C3 — PBO/CPCV : phase 2 ou audit**
   - La mesure la plus forte contre l'overfitting **du processus de sélection**. Mais
     nécessite de conserver la **matrice candidats × temps** de tous les essais → coût
     d'infrastructure réel. Pas dans le MVP ; à activer quand la recherche est stabilisée.
6. **C4 — White Reality Check / Hansen SPA : différé**
   - Tests de data snooping formels ; rigoureux mais redondants avec DSR+PBO à notre
     échelle et plus lourds. Si on veut durcir la preuve plus tard.

### Points de conception non négociables
- **Deux niveaux d'évaluation** : fitness cheap pendant la recherche (retour/Sharpe sur
  données échantillonnées, sans bootstrap) ; validation complète (bootstrap + DSR)
  seulement sur les **finalistes**. Rend validation rigoureuse + budget court compatibles.
- **Métriques NET de coûts** : coût par trade (round-trip, ~0.08% défaut) appliqué avant
  d'admettre, sinon on admet des combos qui meurent dès qu'il y a des frais.
- **Holdout sacré** : 1 seule passe, aucun reverse-engineering dessus (sinon invalide).
- **Audit anti-piège d'annualisation intraday** : sur 15m/1h/4h le Sharpe annualisé gonfle
  par sqrt(nb périodes/an) → vérifier qu'on ne compare pas des artefacts d'unité ; le
  Sharpe per-trade est plus fiable pour comparer.

---

## Étape D — Décision de diversification (corpus de combos indépendantes)

Contexte : objectif = **corpus** de combos rentables et **indépendantes**. MAP-Elites est
déjà retenu côté exploration (A) ; cette étape approfondit le « comment bien diversifier »
et distingue diversité **côté recherche** (D) vs **dédup côté admission** (C).

### Recommandé (cœur)
1. **MAP-Elites avec descripteurs BIEN choisis**
   - Découper l'espace comportemental en niches, garder **le meilleur par niche** → une
     population finale = un bon représentant de CHAQUE style = un corpus par construction.
   - **L'écueil du code précédent : le choix des descripteurs fait tout.** Les niches ne
     doivent PAS recopier la fitness (un bucket de Sharpe pur ne crée pas de diversité).
     Elles doivent décrire **ce qui rend une combo économiquement différente** :
     - direction (long/short),
     - fréquence / durée de tenue (sépare scalping vs trend-following),
     - qualité (Sharpe/rentabilité) : bucket large pour trier DANS une niche, pas comme axe,
     - **famille de features dominante** (momentum, volatilité, volume, market regime…),
     - éventuellement régime de marché privilégié.
   - Qualités d'un bon descripteur : calculable **à la volée sur la fitness cheap**,
     **signifiant économiquement**, **bonne couverture** (peu de niches mortes), **stable**.
   - Pourquoi MAP-Elites : produit directement une sortie « un bon par style » = la forme
     d'un corpus, sans décider à l'avance combien de combos garder.
2. **Novelty Search — COMPLÉMENT OPTIONNEL (phase 2)**
   - Récompense la **nouveauté comportementale** en plus de la performance.
   - La nouveauté SEULE ignore la rentabilité → remplirait le corpus de combos originales
     mais non rentables. Couplé à MAP-Elites, évite la stagnation quand les niches se
     saturent. Réglage supplémentaire → pas dans le MVP.

### Contrepoint indispensable : indépendance côté ADMISSION (guard de C)
La diversité « recherche » ne suffit pas : une génération diverse peut produire des combos
qui **se comportent pareil**. À l'admission :
- **Dédup par fingerprint** (structurel + comportemental) : rejeter les combos déjà vues.
- **Seuils de corrélation/chevauchement** : signal_overlap (Jaccard des dates de signal vs
  corpus) et ret_corr (corrélation des rendements vs corpus) sous seuils (ex. 0.30 / 0.50).
- Pourquoi : l'objectif est un corpus de combos **indépendantes** ; deux combos qui font les
  mêmes trades/rendements n'en valent pas deux.

**Duo** : diversifier à la génération (D) + vérifier l'indépendance à l'admission (guard C).

---

## Étape E — Architecture du système complet (intégration A→B→C→D)

Contexte : relier tous les choix en un pipeline cohérent, dans le cadre d'un **système de
trading autonome** : le moteur de recherche ne produit pas juste des stratégies, il
alimente un système qui les exécute et les surveille.

### Architecture (de bout en bout)
**0. Données & features (socle)**
- Ingestion OHLCV multi-actifs / multi-timeframes (15m, 1h, 4h…).
- Calcul des features ; réduction par l'étape B (features valides + non redondantes, guide
  IM/mRMR, bornes de langage) → l'« espace de combinaisons limité » explicite.
- **Versioning + verrouillage** du jeu de données et des bornes (data_version) :
  reproductibilité, rejets traçables sur une version précise.
- Découpage temporel train/val/holdout purgés + embargoed (C1). Holdout sacré.

**1. Recherche (A + D)** — sur le pool de features borné
- **Random search** = baseline de référence (preuve anti-hasard).
- **STGP** = génération des combos (arbres), fitness **cheap** (données échantillonnées,
  sans bootstrap).
- **MAP-Elites** = diversité : un bon individu par niche comportementale → le candidate pool
  diversifié.

**2. Affinage optionnel (A5)** — ajuster les seuils des finalistes (Optimisation Bayésienne,
frugal) avant la validation lourde.

**3. Validation (C)**
- Test primaire : block bootstrap CI (Sharpe + ret), borne basse > 0.
- DSR : correction du multiple-testing.
- FDR au niveau du corpus : contrôle global des fausses découvertes.
- Dédup + indépendance : fingerprint + seuils de corrélation/chevauchement vs corpus.
- Tout en **métriques nettes de coûts**.

**4. Admission → CORPUS (produit)**
- Combos admises → corpus **versionné**, avec leurs métriques (5e attribut de l'einher) +
  les 4 attributs d'identification (condition, direction, amplitude, univers).
- Rejets → **Archive** = base de connaissances négative (réévaluable sur nouveau
  data_version, jamais réécrite).

**5. Holdout sacré** — 1 seule passe finale avant production.

**6. Exécution (système autonome)**
- Les einhers du corpus tournent sur données live.
- Risk manager + exécution broker.
- **Monitoring** : statuts (actif → dégradé si dérive persistante → archivé ; reconvergence
  → réactif). Le corpus est vivant, pas un dump figé.

**7. Boucle de rétroaction**
- L'Archive nourrit la recherche (ce qui a été testé/échoué) et les quotas de diversité.
- Les métriques live revalident / retirent les einhers dégradés.

### Principes transverses (pourquoi c'est robuste)
- **Deux niveaux d'évaluation** : fitness cheap en recherche ; validation lourde
  (bootstrap+DSR) sur les seuls finalistes → rigueur compatible budget/délai courts.
- **Séparation stricte train/val/holdout/live** : jamais de fuite, holdout sacré.
- **Versioning partout** (données, bornes d'espace, corpus) : reproductibilité + réévaluation
  honnête.
- **Modularité** : chaque étape correspond à une décision A/B/C/D déjà posée → remplaçable
  indépendamment.

En résumé : **0 Données → 1 Recherche (random+STGP+MAP-Elites) → 2 affinage BO → 3 Validation
(CI+DSR+FDR+indépendance) → 4 Corpus → 5 Holdout → 6 Exécution autonome → 7 rétroaction
archive**. Cible d'architecture ; planification uniquement, rien d'implémenté.

---

## F. Vue WorldQuant — minage d'alpha formulaïque (GP / Symbolic Regression)

> Recherche arXiv effectuée le 2026-08-18 (skill arxiv). Cette approche est le cadre de
> référence **industriel** du problème exact d'Einherjar : générer automatiquement des
> formules symboliques (combo de features) prédictives et **interprétables** du rendement
> futur, puis les combiner en portefeuille. C'est la lignée « WorldQuant formulaic alpha ».
> Papiers principaux identifiés et lus (résumés) :

### F1. PySR — Symbolic Regression PRÊTE à l'emploi (outil)
- **Référence** : Cranmer, M., *Interpretable Machine Learning for Science with PySR and
  SymbolicRegression.jl*, arXiv:2305.01582, 2023.
- **Ce que c'est** : bibliothèque open-source de regression symbolique. Cherche des
  formules closes interprétables. Algorithme = **algorithme évolutionnaire multi-population**
  avec boucle unique **evolve–simplify–optimize** (optimise les constantes scalaires des
  expressions découvertes). Backend Julia hautement optimisé (parallélisable, AD, fusion SIMD).
- **Lien avec notre problème** : PySR **incarne concrètement** l'approche GP/symbolique pour
  découvrir des formules de features. Contrairement à un GP « maison », il est éprouvé,
  parallélisable, et gère nativement l'optimisation des constantes.
- **Adéquation / mise en garde** : excellent pour **générer** des expressions candidates,
  mais il trouve des **formules continues** (regression), pas des **conditions discrètes à
  déclenchement** (le cerveau d'un einher). → complément possible du STGP, PAS un
  remplacement direct ; il faut garder notre couche de validation (DSR/PBO/bootstrap).

### F2. QuantFactor REINFORCE — minage d'alpha par RL
- **Référence** : Zhao, J., Zhang, C., Qin, M., Yang, P., *QuantFactor REINFORCE: Mining
  Steady Formulaic Alpha Factors with Variance-bounded REINFORCE*, arXiv:2409.05144, 2024.
  (24 citations.)
- **Ce que c'est** : remplace PPO (RL pour générer les alphas formulaïques) par REINFORCE
  **variance-borné**, avec une baseline dédiée pour réduire la variance, et un **information
  ratio** comme reward shaping → produit des alphas « stables » (adaptés à la volatilité).
- **Lien** : montre une **alternative au GP** pour générer des combos de features : le GP
  comme moteur génératif concurrent du RL.
- **Adéquation** : dans notre contexte (STGP déjà retenu), le RL est plus complexe à calibrer
  ; le GP reste la voie principale. À surveiller comme évolution possible, pas pour le MVP.

### F3. AlphaForge — génération + COMBINAISON d'alphas
- **Référence** : Shi, H., Song, W., Zhang, X., et al., *AlphaForge: A Framework to Mine and
  Dynamically Combine Formulaic Alpha Factors*, arXiv:2406.18394, 2024.
- **Ce que c'est** : framework 2 étapes : (1) un **réseau génératif-prédictif** génère les
  alphas (exploration spatiale profonde + diversité préservée) ; (2) un module de **combinaison
  qui pondère dynamiquement** chaque alpha selon sa performance temporelle (au lieu de poids
  fixes).
- **Lien avec notre étape E (corpus)** : le point fort est la **combinaison dynamique** +
  la **diversité préservée à la génération** — exactement ce qu'on veut pour un corpus dont la
  combinaison robuste fait croître le capital. Valide notre choix de ne pas figer les poids et
  de chercher plusieurs alphas indépendants.

### F4. Alpha Jungle (LLM + MCTS) — le plus récent
- **Référence** : Shi, Y., Duan, Y., Li, J., *Navigating the Alpha Jungle: An LLM-Powered MCTS
  Framework for Formulaic Factor Mining*, arXiv:2505.11122, 2025.
- **Ce que c'est** : utilise un **LLM pour générer/affiner itérativement** les formules
  d'alpha, piloté par un **Monte Carlo Tree Search (MCTS)**, avec un guidage par retour
  quantitatif du backtest de chaque candidat, et un **mécanisme d'évitement de sous-arbres
  fréquents** pour diversifier (anti-homogénéisation).
- **Lien avec notre étape D (diversité)** : le mécanisme anti-homogénéisation des formules =
  le pendant de notre MAP-Elites + dédup comportementale. Le cadre GP reste notre base ; le
  LLM+MCTS est une voie de recherche émergente, lourde (LLM en boucle), à garder en veille.

### Synthèse de cette vue WorldQuant et impact sur nos décisions
Notre pile (A–E) est **cohérente** avec l'état de l'art : GP/STGP pour explorer les combos,
diversité (MAP-Elites / anti-homogénéisation), et **une validation rigoureuse en aval** — ce
que les papiers ci-dessus traitent moins. Les apports concrets :
- **Outils prêts à l'emploi** : PySR (F1) peut être ajouté comme **générateur de formules
  complémentaire** du STGP, pour enrichir l'espace d'expression (ex. formules continues côté
  amplitude/score) avant notre validation.
- **Diversité + combinaison dynamique** (F3) : confirme l'étape D et suggère que la mie en
  corpus gagne à combiner les alphas avec des poids adaptatifs (pas fixes).
- **Anti-homogénéisation** (F4) : renforce notre principe MAP-Elites + fingerprint/dédup.
- **RL (F2/F4) et LLM (F4)** : voies concurrentes plus lourdes, à garder en veille, pas dans
  le MVP (délai 3 semaines).

---

## Synthèse / priorités pour notre contexte

Notre problème = espace borné de combinaisons de features. La réponse n'est pas UN
algorithme mais une **chaîne** :

- **Explorer** : GP/STGP (ou GE si plus d'expression) + diversité MAP-Elites / NSGA-II,
  avec **random search en baseline** (A1, A3, D1, A6).
- **Borner** : prune de features pertinentes (B1/B2) pour rendre l'espace manageable.
- **Valider** : le trio anti-fausses-découvertes = **DSR + PBO/CPCV + bootstrap CI**,
  sur des splits **purgés/embargoed**, avec **holdout sacré** (C1–C6).
- **Diversifier en sortie** : un corpus de combos indépendantes (D1 + dédup
  comportementale).

**Les 3 techniques à mettre en avant pour notre exact contexte :**
1. **GP/STGP (A1)** — l'exploration des combinaisons.
2. **DSR + PBO (C2–C3)** — la validation anti-fausses-découvertes.
3. **MAP-Elites (D1)** — le corpus diversifié.

Plus les **splits purgés/embargoed (C1)** qui sont le socle de tout.

---

## Prochaines étapes possibles (à valider)
- [ ] Comparer en profondeur STGP vs GE vs NSGA-II pour notre espace de features.
- [ ] Détailler la formule mathématique exacte de DSR et PBO.
- [ ] Définir le périmètre exact de l'espace de combinaisons (features autorisées,
      opérateurs, amplitude, univers).
- [ ] Choisir le découpage de validation (walk-forward / CPCV / holdout sacré).
