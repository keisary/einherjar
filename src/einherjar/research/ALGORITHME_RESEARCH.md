# ALGORITHME_RESEARCH — Recherche d'architectures algorithmiques pour le moteur de découverte d'Einherjar

> Document de recherche — Étude comparative des familles d'algorithmes candidates à la phase de découverte du moteur Einherjar.
> Source de vérité pour la conception algorithmique, à utiliser conjointement avec `ONTOLOGY.md` (contrat conceptuel) et `feature_taxonomy_corrected.json` (vocabulaire des features).
> État : **Non figé — V1.1 en cours**. Une critique d'IA tierce (2026-08-01) a invalidé plusieurs décisions prématurées (gel de GE comme moteur principal, gel de la métrique composite, gel du walk-forward 70/30). Le document est en cours de refonte ; voir § 10.2 et § 13 pour la nouvelle approche.

---

## 0. Méthodologie et portée

Cette étude couvre **huit familles algorithmiques** ayant un historique dans la découverte automatique de règles de trading, plus une analyse des **méthodes hybrides** et des **frontières LLM-augmentées** qui dominent l'état de l'art 2024-2026. Chaque famille est traitée avec un protocole homogène en 11 sections. Une synthèse finale propose une architecture algorithmique modulaire compatible avec le contrat conceptuel d'Einherjar (9 concepts, 10 invariants, 3 sections de sémantique).

**Sources principales** : articles scientifiques 2023-2026, prépublications arXiv, surveys récents (en particulier la survey RL en finance de 2025 [1], la survey deep learning/LLM en alpha strategies de 2025 [2], les papiers SOTA de 2025-2026 sur FactorEngine [3], AlphaAgent [4], CogAlpha [5], Hubble [6], QuantaAlpha [7], QuantEvolve [8], et les papiers de référence de Lopez de Prado sur le backtest overfitting [9][10]).

**Périmètre** : algorithmes de **génération et validation** de stratégies. Hors périmètre : exécution, risk management, portfolio allocation — qui sont des couches séparées dans l'architecture d'Einherjar.

**Note méthodologique importante (V1.1, 2026-08-01)** : ce document décrit des **candidats algorithmiques**. Aucun n'est définitivement retenu. Le choix de l'algorithme principal (random / GE / GP typé / beam / memetic / NSGA-II) est **une décision empirique** à prendre **après** la construction du moteur d'évaluation décrit dans `ONTOLOGY.md` (S-2, S-3). Voir § 10.2 pour le nouveau protocole de comparaison.

---

## 1. Le contrat conceptuel à respecter

Avant toute analyse algorithmique, les contraintes suivantes (définies dans `ONTOLOGY.md`) doivent être respectées :

- L'unité d'évaluation n'est pas le **signal** mais le **trade complet** (entrée + sortie avec SL/TP). L'algorithme de recherche produit des **hypothèses**, qui deviennent des **Einher** si elles passent la validation.
- L'**amplitude** est un concept de premier ordre : l'Einher cherche à capturer un mouvement d'une taille spécifiée, en unités de prix ou en multiple d'ATR.
- La validation est **strictement temporelle** : split **train/val/holdout** (60/20/20) avec **purging** (exclusion des bougies dont le label déborde) et **embargo** (exclusion de N bougies supplémentaires après chaque split). Walk-forward seul est insuffisant.
- L'admission exige au minimum : **DSR** (Deflated Sharpe Ratio), **PBO** (Probability of Backtest Overfitting) via CPCV léger, **block bootstrap CI** sur Sharpe et ret total, **cohérence cross-asset** (≥ 70% des actifs positifs), `n_trades` minimal, `max_drawdown` borné, **diversité comportementale** vs corpus.
- Le moteur d'évaluation (entrée OPEN t+1, simulation intrabar TP/SL, block bootstrap CI) est construit **avant** le générateur. C'est l'inversion de priorité par rapport à une architecture naïve "génération d'abord".
- **SL/TP figés depuis le train** : `sl_price` et `tp_price` sont calculés une fois sur le train, jamais recalibrés sur val/holdout/live.
- **Holdout sacré** : consulté une seule fois, à la toute fin, sur l'Einher final retenu.
- Le scoring de recherche (rapide, optimiste) et la validation finale (lente, stricte) sont **deux étapes distinctes**. Le moteur ne s'auto-valide pas.
- Les Einhers rejetés sont **archivés** (jamais supprimés) avec `data_version`, `seed`, `splits`, snapshot complet de métriques, fingerprint canonique (structurel + comportemental), et **réévaluables** sur un nouveau `data_version`.
- La diversité du corpus (invariant I-8) est mesurée par des **descripteurs comportementaux** (overlap signaux, corrélation rendements conditionnels, distribution par régime/horizon) en plus des quotas structurels. Deux Einhers structurellement différents peuvent être économiquement équivalents.

Ces contraintes pèsent sur tous les choix algorithmiques. Une méthode incompatible avec l'un de ces points est écartée d'office, même si elle est populaire dans la littérature.

---

## 2. Chapitre 1 — Recherche évolutionnaire

### 1.1 Présentation générale

La recherche évolutionnaire est la famille dominante dans la littérature sur la découverte automatique de règles de trading. Elle regroupe plusieurs sous-familles (programmation génétique, évolution grammaticale, algorithmes génétiques, stratégies d'évolution) qui partagent le paradigme d'une population d'individus évoluant par sélection, croisement et mutation.

### 1.2 Fonctionnement

Une population de candidats (règles de trading, représentées comme arbres, séquences, ou vecteurs) est initialisée aléatoirement ou par une heuristique. À chaque génération, chaque individu est évalué sur des données historiques (fitness), les meilleurs sont sélectionnés, recombinés (croisement) et mutés. Le processus itère jusqu'à convergence ou budget-temps dépassé.

### 1.3 Variantes importantes

| Variante | Représentation | Avantage | Limite |
|---|---|---|---|
| **Programmation génétique (GP)** | Arbre syntaxique | Expressivité maximale, flexible | Convergence lente, bloat (arbres qui gonflent) |
| **Grammatical Evolution (GE)** | Séquences d'entiers + grammaire BNF | Contraintes syntaxiques, contrôle de la complexité | Plus lent à cause du mapping, grammaires à définir |
| **Cartesian GP (CGP)** | Graphe acyclique | Compact, pas de bloat | Moins expressif |
| **Linear GP (LGP)** | Séquences d'instructions | Exécution rapide, type-safe | Moins flexible |
| **Gene Expression Programming (GEP)** | Expression linéaire → arbre | Combine avantages GP et GA | Mapping complexe |
| **Stratégies d'évolution (ES)** | Vecteurs réels | Très efficace en optimisation continue | Pas adapté aux représentations symboliques |
| **NeuroEvolution** | Réseaux de neurones | Pour des modèles hybrides | Boîte noire, peu interprétable |

Pour Einherjar, qui manipule des **règles symboliques** (features, opérateurs, comparaisons), **GP et GE sont les candidats principaux**. CGP/LGP/GEP sont des alternatives viables mais n'apportent pas d'avantage décisif sur le problème posé.

### 1.4 État de l'art

**GP pour le trading** : la lignée de recherche Kampouridis & Long (Essex) est l'état de l'art, avec des résultats récents (2024-2025) sur 110 à 220 datasets internationaux montrant que la GP multi-objectif bat systématiquement la GP mono-objectif [11][12]. La méthode `α-dominance` permet de relaxer le critère de Pareto pour améliorer la convergence [13].

**Strongly-Typed GP (STGP)** : permet de séparer les types (technique vs sentiment), avec des résultats améliorés pour les fusions multi-sources [14][15].

**Vectorial GP** : extension permettant des opérations vectorielles, utile pour les transformations de features [16].

### 1.5 Forces

- **Expressivité** : GP/GE peuvent représenter n'importe quelle règle dans l'espace de recherche.
- **Maturité** : 30 ans de recherche, nombreuses bibliothèques (DEAP, ECJ, etc.).
- **Parallélisation triviale** : évaluation indépendante de chaque individu.
- **Interprétabilité préservée** : les arbres GP sont lisibles (contrairement aux réseaux).
- **Adapté à notre grammaire BNF** : GE s'intègre naturellement avec la grammaire des conditions.

### 1.6 Faiblesses

- **Bloat** : les arbres GP ont tendance à croître sans gain de fitness, ce qui ralentit l'évaluation.
- **Convergence prématurée** : risque de converger vers des optima locaux (dérive génétique).
- **Coût de calcul** : évaluer des milliers d'individus × de multiples bougies est cher.
- **Sensibilité aux hyperparamètres** : taille de population, taux de mutation/croisement, sélection.

### 1.7 Complexité

- **Temporelle** : O(générations × population × temps_d'évaluation). En pratique, avec une population de 1000 et 20 générations, on évalue 20 000 candidats — ce qui est le coût d'une seule itération.
- **Mémoire** : O(population × taille_individu). En général négligeable.
- **Parallélisation** : triviale (chaque individu indépendant).

### 1.8 Cas d'utilisation

GP est utilisé en production ou quasi-production par plusieurs acteurs :
- Qraft Technologies a publié QuantEvolve [8], un framework multi-agents évolutionnaire déployé industriellement.
- Man Numeric (filiale de Man Group, $500B hedge fund) a déployé un système de découverte de signaux basé sur des agents autonomes (équivalent à de la GP orchestrée) en 2025.
- La recherche académique de l'Université d'Essex a publié des dizaines de papiers sur GP pour le trading depuis 2010.

### 1.9 Compatibilité avec Einherjar

**Forte compatibilité.** GP/GE s'aligne avec :
- L'ontologie : produit naturellement des **conditions** (arbres) composables en **hypothèses**.
- La sémantique : l'évaluation d'un arbre = `eval(condition, context) → bool` est directe.
- Les invariants : I-1 (Einher complet) ✓, I-2 (reproductible) ✓, I-3 (généralisable) ✓, I-5 (pas de fuite) ✓ si on intègre walk-forward.
- Les exclusions : GP ne touche pas au sizing, cooldown, capital allocation.

**Incompatibilités** :
- L'invariant I-6 (pas de doublons) demande que GP intègre la notion de **relations entre features** (RSI ~ RSI_norm). C'est faisable via des pénalités dans la fitness.
- L'invariant I-10 (deux étapes de validation) demande que GP ne soit utilisé qu'au **scoring rapide**, pas à la validation finale.

### 1.10 Conclusion

**À retenir.** GP/GE est l'algorithme de référence pour la génération d'hypothèses dans Einherjar. Il est expressif, parallélisable, et aligné avec le contrat conceptuel. Il est l'algorithme principal du **scoring rapide** (génération de milliers d'hypothèses candidates).

### 1.11 Décision provisoire (révisée 2026-08-01)

**GE est un candidat fort, pas un choix définitif.** L'ancien texte affirmait "GE comme moteur principal, BNF déjà définie". C'était prématuré. La grammaire BNF complète n'est pas encore écrite (ONTOLOGY.md contient la table de typage et les transformations autorisées, mais pas la BNF couvrant amplitude, univers, direction, constantes, canonisation, limites de profondeur). Et le choix entre GE, GP typé, random search et beam search doit être fait **empiriquement** après la construction du moteur d'évaluation (cf. § 10.2 et `ONTOLOGY.md` S-2, S-3).

GE reste néanmoins un **candidat de tête** à comparer en pratique, pour les raisons suivantes :
- Intégration naturelle avec une future grammaire BNF (contrainte syntaxique, contrôle de complexité).
- Bonne compatibilité avec l'ontologie (les arbres se mappent directement sur des `condition_tree`).
- Parallélisation triviale.

Mais cette décision sera **tranchée par l'expérience**, pas par un choix a priori.

---

## 3. Chapitre 2 — Recherche probabiliste

### 2.1 Présentation générale

La recherche probabiliste explore l'espace par échantillonnage aléatoire. Elle inclut le Monte Carlo pur, le Monte Carlo Tree Search (MCTS), le Sequential Monte Carlo (SMC), la recherche aléatoire pure, et les algorithmes à distribution d'estimation (EDA).

### 2.2 Fonctionnement

- **Random search pur** : tirer N candidats au hasard, garder les meilleurs. Pas d'état entre itérations.
- **MCTS** : construire un arbre de recherche par simulations successives (sélection, expansion, simulation, backpropagation). Utilise UCB1 ou PUCT pour équilibrer exploration/exploitation.
- **SMC (particle filters)** : maintenir une population pondérée de candidats, mettre à jour les poids selon la fitness.
- **EDA** : apprendre une distribution probabiliste sur les bonnes solutions, échantillonner depuis cette distribution.

### 2.3 Variantes importantes

| Variante | Domaine | Pertinence pour Einherjar |
|---|---|---|
| Random search | Tous | **Forte** (baseline honnête) |
| MCTS | Espaces arborescents | Moyenne (peut s'appliquer au choix des conditions) |
| SMC | Espaces à poids | Faible (notre problème est binaire/discret) |
| EDA | Optimisation continue | Faible (espace symbolique) |

### 2.4 État de l'art

**MCTS en finance** : utilisé principalement pour l'allocation de portefeuille et la découverte d'expressions (Cazenave 2024 [17] utilise MCTS pour découvrir des termes d'exploration pour MCTS lui-même). Récemment appliqué à la génération d'alphas par Alpha2 [18] et Navigating the Alpha Jungle [19] (combiné avec LLM).

**Random search** : reste la **baseline** dans la plupart des papiers GP/GE. La comparaison "GP vs random search" montre que GP gagne souvent de 10-30% en performance, mais pas toujours (le random est étonnamment fort en haute dimension).

### 2.5 Forces

- **Simplicité** : random search tient en 5 lignes de code.
- **Parallélisation maximale** : aucune dépendance entre échantillons.
- **Pas de paramètres** : random search n'a aucun hyperparamètre à régler.
- **Honnêteté** : sert de baseline non-biaisée pour évaluer les autres méthodes.
- **MCTS** : très efficace sur les espaces où la fitness est chère et où on peut guider la sélection.

### 2.6 Faiblesses

- **Inefficace en haute dimension** : random search devient prohibitif au-delà de quelques millions de candidats.
- **Aucune réutilisation** : ne tire pas parti des informations des essais précédents (sauf MCTS et EDA).
- **MCTS** : sensible au design de l'arbre et à la fonction de simulation.

### 2.7 Complexité

- **Random search** : O(N × coût_d'évaluation). Trivial.
- **MCTS** : O(itérations × coût_simulation). Modéré.

### 2.8 Cas d'utilisation

Random search est la **baseline universelle**. MCTS est utilisé en RL et dans certains frameworks d'alphas (Alpha2, Navigating the Alpha Jungle).

### 2.9 Compatibilité avec Einherjar

**Forte compatibilité pour random search.** Il s'intègre naturellement comme :
- **Baseline de référence** pour comparer les méthodes plus sophistiquées.
- **Étape de bootstrap** dans une stratégie hybride (random search d'abord, puis raffinement).
- **Diversification** : tirer des Einhers diversifiés sous contrainte de quota du corpus.

**MCTS : compatibilité moyenne.** Pourrait servir au **scoring rapide** si on structure la recherche en arbre (par exemple : choix d'abord de la famille, puis de la feature, puis de l'opérateur, puis du seuil). Mais complexifie l'implémentation sans gain démontré pour notre problème.

### 2.10 Conclusion

**Random search est essentiel comme baseline et comme source de diversité.** MCTS est une option secondaire pour des raffinements futurs.

### 2.11 Décision provisoire

**À retenir : random search.** À tester en parallèle de GP pour vérifier l'overhead. MCTS : **à approfondir** si la GP stagne en performance.

---

## 4. Chapitre 3 — Optimisation locale

### 3.1 Présentation générale

Les algorithmes d'optimisation locale partent d'une solution et l'améliorent par perturbations locales. Ils incluent le recuit simulé (SA), la recherche tabou (TS), la recherche par faisceau (Beam Search), le hill climbing, la recherche à voisinage variable (VNS) et la recherche locale itérée (ILS).

### 3.2 Fonctionnement

- **Hill climbing** : à chaque itération, on évalue les voisins de la solution courante et on prend le meilleur. Stagne dans les optima locaux.
- **Recuit simulé (SA)** : comme hill climbing, mais on accepte parfois des solutions moins bonnes selon une **température** qui décroît. Permet d'échapper aux optima locaux.
- **Recherche tabou (TS)** : on maintient une **liste taboue** des solutions récemment visitées pour éviter les cycles.
- **Beam search** : on garde les K meilleurs candidats à chaque étape (compromis entre exhaustivité et hill climbing).
- **VNS/ILS** : on change la structure du voisinage pour échapper aux optima locaux.

### 3.3 Variantes importantes

| Variante | Force | Pour Einherjar |
|---|---|---|
| Hill climbing | Simplicité | Baseline de raffinement local |
| Recuit simulé | Échappement d'optima locaux | **Utile pour raffiner** |
| Recherche tabou | Mémoire explicite | **Utile pour éviter la redondance** |
| Beam search | Compromis exploration/exploitation | **Très pertinent** |
| VNS/ILS | Adaptabilité du voisinage | Option avancée |

### 3.4 État de l'art

**SA et TS pour le trading** : utilisés principalement pour l'optimisation de portefeuille (Schaerf 2023 [20] : SA trouve des solutions quasi-optimales en 5 secondes, TS surpasse SA pour le Sharpe). Pour la découverte de stratégies, application limitée mais documentée (Chen 2023 [21] : algorithme mémétique = GA + SA pour optimiser les paramètres d'indicateurs techniques).

**Beam search en régression symbolique** : gagne du terrain depuis 2024. PIGP (NeurIPS 2024 [22]) utilise un transformer pour générer des candidats, puis beam search pour les raffiner. "Scaling Up Unbiased Search-based Symbolic Regression" (IJCAI 2024 [23]) montre que la recherche systématique (dont beam) bat les GP/RL state-of-the-art en SR.

**Beam search self-supervised pour SR** [24] : combine beam search et heuristique apprise, résultats compétitifs avec SRBench.

### 3.5 Forces

- **Exploitation efficace** : raffinement rapide autour d'un bon candidat.
- **Beam search** : bon compromis exploration/exploitation pour des espaces structurés.
- **Tabu search** : particulièrement adapté pour éviter la **re-soumission** (notre invariant I-6 sur les doublons).

### 3.6 Faiblesses

- **Optima locaux** : sans mécanisme d'évasion, le hill climbing stagne. SA/TS/VNS atténuent mais ne résolvent pas.
- **Pas de diversité** : par construction, l'optimisation locale ne produit pas de solutions diverses.
- **Coût par itération** : SA et TS nécessitent une bonne fonction de voisinage, qui est non triviale dans notre espace symbolique.

### 3.7 Complexité

- **Hill climbing** : O(itérations × |voisinage| × coût_évaluation).
- **SA / TS** : O(itérations × |voisinage| × coût_évaluation), avec mémoire constante pour TS.
- **Beam search** : O(K × profondeur × |branching| × coût_évaluation).

### 3.8 Cas d'utilisation

- **SA** : optimisation de portefeuille (reparameterisation continue).
- **TS** : problèmes d'allocation discrète, séquencement.
- **Beam search** : SR, NLP, parcours d'arbres.

### 3.9 Compatibilité avec Einherjar

**Beam search et TS : forte compatibilité pour la phase de raffinement.**

- **Beam search** peut être utilisé pour explorer l'arbre des Einhers candidats à profondeur fixe, en gardant les K meilleurs à chaque niveau. Compatible avec notre grammaire BNF.
- **TS** peut empêcher la re-soumission de candidats déjà rejetés (intégration avec l'Archive, invariant I-6).
- **SA** est moins pertinent (notre espace est principalement symbolique/discret).

### 3.10 Conclusion

Beam search est un **candidat sérieux pour le raffinement** d'Einhers candidats. TS peut servir à la **déduplication** au niveau de l'Archive.

### 3.11 Décision provisoire

**À retenir : beam search** comme méthode de raffinement dans une architecture hybride. **À approfondir : TS** pour la déduplication. **À écarter pour V1 : SA, VNS, ILS** (trop complexes pour le gain attendu).

---

## 5. Chapitre 4 — Métaheuristiques de population

### 4.1 Présentation générale

Les métaheuristiques de population font évoluer un ensemble de solutions, contrairement à l'optimisation locale qui n'en suit qu'une. Cette catégorie inclut les algorithmes génétiques (GA), l'évolution différentielle (DE), l'optimisation par essaim de particules (PSO), l'optimisation par colonies de fourmis (ACO), l'algorithme des abeilles (ABC) et la recherche d'harmonie (HS).

### 4.2 Fonctionnement

- **GA** : sélection, croisement, mutation sur une population de chromosomes.
- **DE** : différentiation entre individus pour générer de nouveaux vecteurs.
- **PSO** : particules qui se déplacent dans l'espace, influencées par leur meilleure position et celle de l'essaim.
- **ACO** : fourmis qui déposent des phéromones sur les bonnes solutions.
- **ABC** : abeilles qui explorent, exploitent et recrutent.
- **HS** : vecteurs dans un espace de paramètres musicaux, combinés harmoniquement.

### 4.3 Variantes importantes

| Variante | Domaine d'excellence | Pour Einherjar |
|---|---|---|
| **GA** | Discret + continu | Pertinent (paramètres) |
| **DE** | Continu | Pertinent (seuils continus) |
| **PSO** | Continu | Pertinent (calibrage) |
| **ACO** | Combinatoire | Faiblement pertinent |
| **ABC** | Continu | Peu de littérature en trading |
| **HS** | Continu | Marginal |

### 4.4 État de l'art

**GA pour le trading** : classique (Alves 2024 [25], studies fondatrices de 1990s). Sert souvent de baseline dans les papiers GP/GE (qui sont en fait des extensions de GA avec représentation en arbre).

**DE pour le trading** : utilisé pour l'optimisation de portefeuille (Han 2024 [26] : Memetic DE pour optimiser Mean-CVaR, résultats prometteurs). Un framework DE haute performance pour fintech [27] montre que DE peut traiter des centaines de milliers de stratégies par seconde.

**PSO** : peu utilisé pour la découverte directe de stratégies, plus pour l'optimisation de paramètres.

**Memetic DE** : combinaison DE (global) + recherche locale (exploitation) — la famille mémétique est très efficace.

### 4.5 Forces

- **Maturité et bases théoriques** : 50 ans de littérature.
- **Parallélisation** : triviale.
- **Robustesse** : moins sensibles aux optima locaux que l'optimisation locale pure.
- **DE** : très peu de paramètres, robuste.

### 4.6 Faiblesses

- **GA** : représentation chromosome peu adaptée aux arbres/grammaires.
- **PSO, ABC, HS** : pour espace continu surtout ; notre problème est mixte (symbolique + continu).
- **Convergence prématurée** : la population peut stagner.

### 4.7 Complexité

- **GA** : O(générations × population × coût_évaluation).
- **DE** : O(générations × population × dimension × coût_évaluation).
- **PSO** : similaire à GA.

### 4.8 Cas d'utilisation

- **GA** : optimisation de portefeuilles, sélection de features.
- **DE** : optimisation de paramètres d'indicateurs, optimisation de portefeuille.

### 4.9 Compatibilité avec Einherjar

**Compatibilité moyenne.** Les métaheuristiques de population pure ne sont pas idéales pour notre problème parce que :
- L'espace de recherche est symbolique (conditions sur features), pas continu.
- La représentation chromosome ne capture pas naturellement la structure d'un Einher.

**Mais** : on peut les utiliser pour le **calibrage des seuils continus** (un sous-problème de l'optimisation Einher) ou en **hybridation avec GP** (le memetic algorithm).

### 4.10 Conclusion

Pas l'algorithme principal. Utile en sous-traitance pour des problèmes de calibration, ou en composante d'une architecture hybride (mémétique).

### 4.11 Décision provisoire

**À écarter comme moteur principal.** **À garder en réserve** pour des sous-problèmes de calibration. Memetic = hybridation GA/DE + local search, à surveiller.

---

## 6. Chapitre 5 — Recherche guidée et logique

### 5.1 Présentation générale

Cette catégorie regroupe les algorithmes qui exploitent la structure logique du problème : A*, recherche best-first, branch and bound, programmation par contraintes, SAT/SMT, rule learning, et programmation logique inductive (ILP).

### 5.2 Fonctionnement

- **A* / best-first** : exploration d'un graphe d'états avec une heuristique.
- **Branch and bound** : exploration systématique avec élagage basé sur des bornes.
- **Programmation par contraintes** : déclaratif, le solveur explore l'espace.
- **SAT/SMT** : réduction à un problème de satisfiabilité booléenne.
- **Rule learning** : induction de règles if-then à partir d'exemples.
- **ILP** : induction de programmes logiques (clauses Prolog) à partir d'exemples positifs/négatifs et de connaissances de base.

### 5.3 Variantes importantes

| Variante | Force | Pour Einherjar |
|---|---|---|
| A* / best-first | Optimalité (avec bonne heuristique) | Marginal (pas de graphe naturel) |
| B&B | Élagage | Marginal |
| CP / SAT / SMT | Exploitation des contraintes | **Pertinent** (notre grammaire est une contrainte) |
| Rule learning | Lisible | **Pertinent** |
| ILP | Expressivité logique | **Pertinent historiquement** |

### 5.4 État de l'art

**ILP pour le trading** : Badea 2000 [28] a montré qu'on peut induire des règles de trading par ILP en labellisant les "occasions idéales" et en donnant ces exemples à un apprenant ILP. Plus récemment, Murray 2023 [29] utilise ILP pour découvrir des motifs symboliques fréquents utilisés comme features pour du meta-RL.

**Rule learning (CN2, RIPPER)** : utilisé pour l'équity trading (Keegstra [30] compare CART, CN2, RIPPER, RUG ; CN2 surpasse RIPPER en précision). Adapté aux données catégorielles, mais nécessite une discrétisation des features continues.

**SAT/SMT** : pas d'application directe en découverte de stratégies, mais utilisé pour vérifier la cohérence des grilles de contraintes.

### 5.5 Forces

- **ILP** : règles très expressives, connaissances de base intégrables.
- **Rule learning** : règles lisibles, classiques en data mining.
- **CP** : peut gérer des contraintes complexes sur les Einhers.

### 5.6 Faiblesses

- **ILP** : coûteux en calcul, sensible au bruit (notre problème est TRÈS bruité), supposé supervisé.
- **Rule learning** : discrétisation des features = perte d'information.
- **CP/SAT** : sur des problèmes continus, le passage au discret est délicat.

### 5.7 Complexité

- **ILP** : exponentielle dans le pire cas, mais traitables si la théorie cible est petite.
- **CP** : NP-difficile en général.

### 5.8 Cas d'utilisation

- **ILP** : domaines avec connaissances expertes riches (médecine, biologie).
- **Rule learning** : data mining classique.

### 5.9 Compatibilité avec Einherjar

**Compatibilité faible à moyenne.** Notre problème est :
- **Non supervisé** : on n'a pas d'exemples positifs/négatifs pré-établis. Les seules "occasions idéales" viendraient du futur (ce qu'on ne peut pas utiliser sans fuite).
- **Bruyant** : ILP et rule learning sont conçus pour des données propres.
- **À features continues** : la discrétisation est destructrice.

**Possible** : utiliser ILP pour valider la cohérence logique des Einhers découverts (vérifier qu'un Einher généré n'est pas trivialement équivalent à un autre).

### 5.10 Conclusion

Pas un algorithme principal pour Einherjar. Marginal pour des usages annexes.

### 5.11 Décision provisoire

**À écarter pour V1.** Pourrait être revisité si on découvre que le bruit du marché est plus faible que prévu ou si on a un bon générateur d'exemples synthétiques.

---

## 7. Chapitre 6 — Découverte de règles

### 6.1 Présentation générale

Cette catégorie regroupe les techniques spécifiquement conçues pour extraire des règles à partir de données : Association Rule Mining (ARM), Symbolic Regression (SR), RuleFit, et les algorithmes classiques de rule learning (RIPPER, CN2).

### 6.2 Fonctionnement

- **ARM** : extraire des règles du type "si A et B alors C" avec des seuils de support et de confiance. Nécessite des données **catégorielles**.
- **SR** : découvrir une expression mathématique f(X) = y qui explique les données. Très utilisée en finance pour les alphas.
- **RuleFit** : combiner des règles interprétables avec des méthodes ensemblistes.
- **RIPPER/CN2** : induction de règles if-then par séparation-et-conquête.

### 6.3 Variantes importantes

| Variante | Sortie | Pour Einherjar |
|---|---|---|
| **ARM** | Règles catégorielles | Faible (besoin de discrétisation) |
| **SR** | Expressions mathématiques | **Forte** (alphas = expressions) |
| **RuleFit** | Ensemble de règles | Pertinent en post-traitement |
| **RIPPER/CN2** | Règles if-then | Pertinent en post-traitement |

### 6.4 État de l'art

**SR pour les alphas** : c'est le **domaine qui explose** depuis 2023-2025.

- **AlphaFormer** (2024 [35]) : Transformer encodeur-décodeur pré-entraîné sur données synthétiques, génère des alphas de bout en bout.
- **AlphaForge** (2024 [36]) : framework à deux composants (génération + timing), utilise le RL pour la combinaison.
- **Alpha-GFN** : GFlowNet pour générer des alphas (approche générative probabiliste).
- **FactorEngine** (2025-2026 [3]) : découverte de facteurs au niveau programme (Turing-complet), guidé par LLM avec recherche bayésienne d'hyperparamètres. État de l'art 2026.
- **AlphaAgent** (2025 [4]) : LLM avec trois mécanismes (originalité, alignement, complexité) pour éviter l'alpha decay.
- **CogAlpha** (2025-2026 [5]) : évolution de code alpha par LLM avec multi-stage prompts.
- **Hubble** (2026 [6]) : LLM + AST sandbox + evolutionary feedback.
- **QuantaAlpha** (2026 [7]) : évolution au niveau des trajectoires, pas des alphas individuels.
- **QuantEvolve** (2025 [8]) : framework multi-agents avec feature map pour préserver la diversité.

**ARM pour le trading** : applicable mais limité (nécessite discrétisation). Papiers typiques : Prathibha 2013 [37], Horzyk 2024 [38].

**Sequential pattern mining** : applicable pour découvrir des séquences d'événements (e.g., "prix monte, volume spike, vol baisse → signal"). Moins direct pour notre problème.

### 6.5 Forces

- **SR** : expressivité mathématique, interprétabilité, alignement naturel avec les alphas factoriels.
- **LLM-augmented SR** : la tendance 2024-2026, résultats state-of-the-art, déployée en production (Man Group).
- **ARM** : règles explicites, comptage de fréquence.

### 6.6 Faiblesses

- **ARM** : discrétisation destructrice pour les features continues.
- **SR classique (GP)** : bloat, convergence lente.
- **LLM-augmented** : coût computationnel élevé, dépendance à un service externe, risque d'instabilité.

### 6.7 Complexité

- **SR (GP)** : voir Chapitre 1.
- **LLM-augmented** : chaque appel LLM coûte $0.001-0.01 et quelques secondes. Pour 10 000 hypothèses, c'est $10-100 et plusieurs heures. Faisable mais cher.

### 6.8 Cas d'utilisation

- **SR classique** : benchmark de référence (AlphaFormer, FactorEngine).
- **LLM-augmented** : déployé en production par Man Group, Qraft, plusieurs hedge funds.

### 6.9 Compatibilité avec Einherjar

**Forte compatibilité pour SR.** La Symbolic Regression produit naturellement des **expressions** qui peuvent être ré-interprétées comme des **conditions Einher** (par exemple, en seuillant l'expression).

**LLM-augmented : forte compatibilité mais complexité opérationnelle.** L'intégration d'un LLM dans le moteur Einher ajoute :
- Une dépendance externe (coût, latence, disponibilité).
- Une non-déterminisme (même prompt ≠ même output).
- Un risque que le LLM produise des conditions syntaxiquement invalides.

**Pour V1, je recommande de NE PAS utiliser de LLM** — trop de complexité pour un gain non démontré. À reconsidérer pour V2 quand le moteur de base fonctionne.

### 6.10 Conclusion

**SR (GP/GE) est dans le pipeline.** LLM-augmented est dans la roadmap V2.

### 6.11 Décision provisoire

**À retenir : SR classique** (aligné avec GP du Chapitre 1). **À écarter pour V1 : LLM**. **À approfondir en V2 : LLM-augmented** quand le corpus de base est mature et qu'on a les moyens d'intégrer un service externe.

---

## 8. Chapitre 7 — Recherche multi-objectifs

### 7.1 Présentation générale

Les algorithmes d'optimisation multi-objectifs (MOO) trouvent un ensemble de solutions Pareto-optimales plutôt qu'une seule solution. Ils incluent NSGA-II, NSGA-III, SPEA2, MOEA/D, et leurs variantes.

### 7.2 Fonctionnement

- **NSGA-II** : trie la population par fronts de non-domination, utilise la crowding distance pour préserver la diversité au sein de chaque front.
- **NSGA-III** : extension pour les problèmes à plus de 3 objectifs, utilise des directions de référence.
- **SPEA2** : archive externe des solutions non-dominées, force de fitness basée sur la dominance et la densité.
- **MOEA/D** : décompose le problème multi-objectif en sous-problèmes scalaires.

### 7.3 Variantes importantes

| Variante | Nb d'objectifs | Pour Einherjar |
|---|---|---|
| **NSGA-II** | 2-3 | **Idéal** (3 objectifs : retour, risque, drawdown) |
| **NSGA-III** | 4+ | Réserve (si on ajoute un 4e objectif) |
| **SPEA2** | 2-3 | Concurrent de NSGA-II |
| **MOEA/D** | 2-3 | Concurrent de NSGA-II |

### 7.4 État de l'art

**NSGA-II pour le trading** : établi depuis 2002, massivement utilisé.
- Prasad 2021 [39] : NSGA-II pour optimiser Sharpe et Max Drawdown, validation walk-forward rolling.
- Long 2025 [11] : NSGA-II combiné à GP, résultats SOTA sur 110 datasets.
- Wu 2024 [40] : NSGA-II pour la sélection de stratégies (PASS), réduction de 73.9% du risque drawdown.
- Alonso 2023 [41] : collaboration de MOEAs pour la construction du front de Pareto.

**Théorie récente** : Zheng & Doerr 2024 [42] ont montré que NSGA-II avec truthful crowding distance a des garanties runtime pour les problèmes à 2 objectifs. Pour ≥3 objectifs, NSGA-II peut être sub-optimal.

### 7.5 Forces

- **Fronte de Pareto** : on obtient directement un ensemble de solutions diversifiées.
- **Pas de pondération a priori** : pas besoin de définir des poids sur les objectifs.
- **Diversité intégrée** : la crowding distance force la diversité.
- **Production-ready** : NSGA-II est un standard industriel.

### 7.6 Faiblesses

- **Coût de calcul** : O(M × N²) pour M objectifs et N individus.
- **Scalabilité** : NSGA-II classique a du mal au-delà de 3 objectifs.
- **Interprétabilité du front** : le front de Pareto peut être grand, dur à exploiter sans critère de sélection final.

### 7.7 Complexité

- **NSGA-II** : O(M × N² × coût_évaluation) par génération. Acceptable.
- **NSGA-III** : O(M × N × log N) par génération. Meilleure scalabilité.

### 7.8 Cas d'utilisation

- **NSGA-II** : trading [39][11][40], ingénierie, optimisation combinatoire.
- **NSGA-III** : problèmes à 4+ objectifs (rare en trading).

### 7.9 Compatibilité avec Einherjar

**Excellente compatibilité.** Einherjar doit optimiser plusieurs objectifs :
- Retour (max)
- Risque / Volatilité (min)
- Max Drawdown (min)
- Potentiellement : stabilité cross-asset, fit temporel.

**NSGA-II est directement applicable** comme algorithme de scoring multi-objectif. Le front de Pareto final peut être filtré pour ne garder que les solutions qui satisfont nos critères d'admission (I-3 : ≥ 3 actifs, etc.).

**Important** : un seul Einher n'est pas un front de Pareto. Le moteur doit soit :
- Choisir une solution de compromis sur le front (selon une métrique composite).
- Garder tout le front comme **variantes d'un même Einher** (à rejeter — trop complexe).
- **Mieux** : faire de la MOO **au niveau de la fitness** mais ne garder qu'une solution (celle qui maximise une métrique composite, par exemple un Sharpe modifié qui pénalise le drawdown).

### 7.10 Conclusion

**NSGA-II est la référence** pour la MOO dans Einherjar. Utilisable directement, avec une **métrique composite** pour la sélection finale.

### 7.11 Décision provisoire (révisée 2026-08-01)

**NSGA-II reste un candidat de tête pour la couche de scoring multi-objectif**, mais la **métrique composite pondérée n'est plus figée**. L'ancien texte proposait `0.5*retour - 0.3*volatilité - 0.2*drawdown + 0.1*novelty_bonus` avec des poids affirmés a priori. C'était prématuré.

Nouvelle position :
- **Pas de pondération figée** sans protocole de calibration empirique.
- Si on garde une métrique composite, ses poids seront **recalibrés sur les baselines** (étape 1 du nouveau pipeline), pas affirmés par raisonnement.
- NSGA-II reste la **méthode de référence** si on observe que la fitness composite masque des trade-offs (front de Pareto réellement conflictuel).
- Les deux options (composite vs NSGA-II) sont **mises en compétition** dans l'étape 2 du pipeline (cf. § 10.2).

---

## 9. Chapitre 8 — Méthodes hybrides et frontière LLM-augmentée

### 8.1 Présentation générale

Les méthodes hybrides combinent plusieurs algorithmes pour exploiter leurs forces complémentaires. Elles dominent la recherche récente (2023-2026), notamment via les **architectures mémétiques** (EA + local search) et les **frontières LLM-augmentées** (LLM + evolutionary search).

### 8.2 Fonctionnement

- **Mémétique** : algorithme évolutionnaire (global) + recherche locale (exploitation). L'archétype : GA + SA ou GP + hill climbing.
- **Hyper-heuristiques** : algorithme qui choisit parmi plusieurs heuristiques de bas niveau.
- **LLM-augmented** : un LLM (typiquement GPT-4) génère des candidats, guidé par du feedback quantitatif (backtest).
- **MCTS + LLM** : LLM comme policy, MCTS comme mécanisme d'exploration.
- **Quality-Diversity** : algorithmes qui préservent explicitement la diversité (MAP-Elites, novelty search).

### 8.3 Variantes importantes

| Hybride | Combinaison | Pour Einherjar |
|---|---|---|
| **Mémétique** | EA + local search | **Très pertinent** (raffinement) |
| **Hyper-heuristique** | Sélection d'heuristiques | Pertinent (choisir entre GP, random, beam) |
| **LLM-augmented SR** | LLM + GP | Pertinent pour V2 |
| **MCTS + LLM** | LLM policy + MCTS exploration | Pertinent pour V2 |
| **Quality-Diversity** | MAP-Elites | Pertinent pour la diversité du corpus |

### 8.4 État de l'art

**Mémétique pour le trading** : établi.
- Chen 2023 [21] : GA + SA pour optimiser les paramètres d'indicateurs, 26-33% de retour sur 3 datasets.
- Han 2024 [26] : Memetic DE pour portfolio optimization avec Mean-CVaR.
- Chen 2024 [43] : Memetic Algorithm pour portfolio de stratégies diverses.

**LLM-augmented alpha mining** : c'est **LE sujet brûlant** 2024-2026.
- AlphaAgent [4] : LLM + 3 mécanismes de régularisation (originalité AST, alignement sémantique, complexité).
- Alpha-GPT (2025) [44] : framework interactif humain-LLM pour alpha mining.
- CogAlpha [5] : évolution de code par LLM avec multi-stage prompts et feedback financier.
- FactorEngine [3] : programme Turing-complet + LLM-guided search + Bayesian optimization.
- Hubble [6] : LLM dans AST sandbox avec validation statistique rigoureuse.
- QuantaAlpha [7] : évolution au niveau trajectoire (chaque run est une trajectoire).
- Navigating the Alpha Jungle [19] : LLM + MCTS avec feedback de backtesting.
- Automate Strategy Finding with LLM [45] : 53.17% de retour cumulé sur SSE 50 (2023-2024) en production.

**Production deployment** :
- Man Numeric (Man Group) a déployé un système agentique LLM pour la découverte de signaux en 2025 [46]. "Plusieurs douzaines d'investment signals approved for live trading" — c'est un déploiement institutionnel à $500B d'AUM.
- Qraft Technologies utilise QuantEvolve [8] en production.
- Alpha-GPT est utilisé en pratique dans plusieurs boutiques quant.

**Quality-Diversity** : QuantEvolve [8] utilise explicitement un "feature map" pour préserver la diversité. C'est exactement l'invariant I-8 d'Einherjar (diversité structurelle du corpus).

### 8.5 Forces

- **Mémétique** : combine exploration globale et exploitation locale. Souvent le meilleur des deux mondes.
- **LLM-augmented** : état de l'art 2025-2026, déployé en production.
- **Hyper-heuristique** : permet d'adapter la stratégie au problème.

### 8.6 Faiblesses

- **Mémétique** : plus complexe à paramétrer.
- **LLM-augmented** : coût, latence, dépendance externe, non-déterminisme, instabilité.
- **Quality-Diversity** : complexe à mettre en œuvre.

### 8.7 Complexité

- **Mémétique** : O(EA × LS × coût_évaluation).
- **LLM-augmented** : O(N_requêtes × coût_requête + N_backtests × coût_backtest). En 2026, quelques secondes par requête LLM.

### 8.8 Cas d'utilisation

- **Mémétique** : trading [21][26], optimisation combinatoire, design engineering.
- **LLM-augmented** : déployé en production par Man Group, Qraft, plusieurs boutiques quant.

### 8.9 Compatibilité avec Einherjar

**Mémétique : excellente compatibilité.** Le pattern "EA pour explorer + LS pour raffiner" est exactement ce dont Einherjar a besoin :
- Étape 1 : GP ou random search pour générer des Einhers candidats.
- Étape 2 : SA ou beam search pour raffiner les seuils et la structure.
- Étape 3 : NSGA-II pour optimiser multi-objectif.

**LLM-augmented : compatibilité complexe.** Pour V1, c'est trop tôt (manque de maturité du système). Pour V2, c'est la **promesse principale** du domaine.

**Quality-Diversity : excellente compatibilité** avec l'invariant I-8 (diversité du corpus).

### 8.10 Conclusion

Les méthodes hybrides sont **le futur proche** d'Einherjar. La mémétique devrait être dans la V1. Le LLM-augmented dans la V2.

### 8.11 Décision provisoire (révisée 2026-08-01)

L'ancien texte affirmait "architecture mémétique (GP + beam search + NSGA-II composite) pour V1". C'était prématuré pour les mêmes raisons que § 1.11 et § 7.11.

Nouvelle position :
- **Mémétique (EA + local search) reste un candidat**, mais n'est plus le choix par défaut.
- **Quality-Diversity (MAP-Elites) reste un candidat** pour la diversité du corpus, mais les quotas + descripteurs comportementaux suffisent pour V1.
- **LLM-augmented reste en V2**, à cause de la complexité opérationnelle.
- Tous ces candidats seront **comparés empiriquement** dans l'étape 2 du pipeline (cf. § 10.2), avec mêmes seeds, mêmes splits, même budget, mêmes métriques, mêmes coûts.

---

## 10. Synthèse — Architecture algorithmique recommandée pour Einherjar

### 10.1 Principe directeur

L'architecture recommandée **n'est PAS un seul algorithme** mais un **pipeline à plusieurs étages** où chaque algorithme fait ce qu'il sait faire le mieux. Elle respecte le principe de simplicité (« ce qui peut être fait simplement, doit être fait simplement ») tout en intégrant les leçons de l'état de l'art.

**Inversion de priorité par rapport à l'architecture V1 précédente** : le **moteur d'évaluation** (entrée OPEN t+1, simulation intrabar TP/SL, block bootstrap CI, splits train/val/holdout) est construit **AVANT** tout générateur. Sans moteur d'évaluation verrouillé et audité, aucune comparaison de générateurs n'a de sens.

**Décisions empiriques** : le choix du générateur (random / GE / GP typé / beam / memetic / NSGA-II) n'est pas figé. Il sera tranché par une étude comparative reproductible (mêmes seeds, mêmes splits, même budget, mêmes métriques, mêmes coûts). Voir § 10.2.

### 10.2 Le pipeline corrigé (V1.1)

```
[ETAPE 0] Moteur d'évaluation (priorité 0, AVANT tout générateur)
    ↓
[ETAPE 1] Baselines (règle humaine, énumération peu profonde, random contraint)
    ↓
[ETAPE 2] Comparaison générateurs (random / GE / GP typé / beam) — protocole reproductible
    ↓
[ETAPE 3] Choix du générateur V1 sur résultats
    ↓
[ETAPE 4] Raffinement local (beam search autour des Einhers viables)
    ↓
[ETAPE 5] Admission (DSR, PBO, bootstrap CI, cross-asset, diversité structurelle + comportementale)
    ↓
[ETAPE 6] Évaluation finale unique sur le holdout
```

#### Étape 0 — Moteur d'évaluation (priorité 0, AVANT tout générateur)

**Objectif** : disposer d'un évaluateur hors-échantillon **exécutable, versionné, seedé, auditable**, qui sera utilisé par tous les générateurs et toutes les étapes suivantes.

**Spécification** : conforme à `ONTOLOGY.md` S-2 et S-3. Points durs :
- Entrée à l'OPEN de la bougie t+1, pas au close.
- Simulation intrabar TP/SL sur high/low de la fenêtre [t+1, t+N].
- Convention de priorité : SL touché avant TP sur la même bougie (conservateur).
- Coûts simulés : spread, commission, slippage (tirés de la config broker, pas inventés).
- Block bootstrap CI sur Sharpe et ret total (longueur de bloc = `1.5 × N_max` par défaut).
- Splits train/val/holdout (60/20/20) avec purging et embargo, figés par `data_version`.

**Livrable** : module Python testable, avec un test de non-régression (mêmes inputs → mêmes outputs) et un test de non-fuite (aucun paramètre du val/holdout n'est utilisé sur le train).

**Pourquoi c'est l'étape 0** : sans moteur d'évaluation commun, toute comparaison de générateurs est biaisée (chacun calcule son propre PnL, ses propres seuils, sa propre convention de sortie). L'évaluation doit être un **contrat**, pas une propriété émergente d'un générateur.

#### Étape 1 — Baselines honnêtes

**Objectif** : mesurer le **plancher de performance** avant de chercher à faire mieux.

**Baselines** :
- **Règle humaine minimale** : 1-3 conditions triviales choisies à la main (ex : "RSI < 30 sur 1h, position long, sortie à TP=2×ATR ou SL=1×ATR"). Sert de sanity check.
- **Énumération peu profonde** : tous les Einhers à 1-2 conditions sur un sous-espace restreint de features. Borne inférieure de ce qui est atteignable par force brute.
- **Random search contraint** : tirage aléatoire de conditions sous contraintes (typage, profondeur, ratios), évaluation par le moteur de l'étape 0. Mesure la valeur ajoutée des méthodes plus sophistiquées.

**Livrable** : courbe de distribution de Sharpe et de PnL pour chaque baseline, sur le val (jamais sur le holdout). Permet de calibrer l'espérance : un générateur qui ne bat pas ses baselines est cassé.

#### Étape 2 — Comparaison reproductible des générateurs

**Objectif** : trancher empiriquement entre random / GE / GP typé / beam / memetic / NSGA-II.

**Protocole** (identique pour tous les candidats) :
- **Mêmes seeds** : un seed maître, propagation déterministe à chaque candidat.
- **Mêmes splits** : train/val/holdout identiques, issus de l'étape 0.
- **Même budget de calcul** : nombre d'évaluations égal entre candidats (ex : 100 000 évaluations chacun).
- **Mêmes métriques** : Sharpe, PnL net, MDD, DSR, PBO, n_trades, descripteurs comportementaux.
- **Mêmes coûts simulés** : spread/commission/slippage figés.
- **Critère de comparaison principal** : nombre d'Einhers valides (passant S-3.4) produits par unité de budget, et qualité médiane de ces Einhers.

**Candidats à comparer** :
- **Random search** (baseline algorithmique de référence).
- **GE contrainte par grammaire BNF** (si la grammaire est écrite à ce stade).
- **GP typé (Strongly-Typed GP)** (contrôle de types, pas besoin de BNF).
- **Beam search** (profondeur fixe, K=64 ou 128).
- **Mémétique (GE + hill climbing)** (optionnel, si la GE pure stagne).
- **NSGA-II sur fitness composite** (optionnel, si on a une métrique composite stable).

**Livrable** : classement des candidats par (a) taux d'admission, (b) qualité médiane des Einhers admis, (c) diversité comportementale produite. Choix du générateur V1 sur résultats.

**Pourquoi ne pas figer GE ici** : aucun benchmark publié n'a comparé GE à random search et beam search sur exactement notre problème (règles de trading avec contrainte d'amplitude, frais réalistes, validation DSR/PBO, holdout sacré). Affirmer que GE bat le random est un a priori, pas un fait.

#### Étape 3 — Choix du générateur V1

**Objectif** : installer le générateur retenu dans le pipeline de production.

**Critère** : le générateur qui maximise le **taux d'admission** (proportion d'Einhers passant S-3.4 par unité de budget) ET la **diversité comportementale** (le corpus doit être diversifié, pas dominé par un seul profil de trade).

**Si aucun candidat ne se distingue** : fallback sur random search (baseline), avec raffinement beam en aval. On n'est pas obligé d'avoir une méthode sophistiquée.

#### Étape 4 — Raffinement local

**Objectif** : pour les Einhers viables mais sous-optimaux, raffiner les seuils et la structure.

**Algorithme** : **Beam search** autour de l'Einher viable, en variant un paramètre à la fois (seuil d'une condition, opérateur, constante).

- Profondeur de beam : 1-2 niveaux.
- Critère d'arrêt : 100 itérations sans amélioration, ou budget épuisé.

**Règle dure** : le raffinement ne recalibre **jamais** SL/TP (figés depuis train). Il ne peut que modifier les conditions ou les constantes internes.

#### Étape 5 — Admission au corpus

**Objectif** : ne garder que des Einhers qui satisfont l'invariant I-8 (diversité du corpus) ET les critères S-3.4.

**Algorithme** : **quotas structurels + descripteurs comportementaux + déduplication par fingerprint canonique**.

1. Calculer le **fingerprint canonique** = hash de `condition_tree + direction + universe + amplitude_cible + sl + tp` (structurel) + hash des descripteurs comportementaux arrondis à une grille stable (comportemental).
2. Vérifier la déduplication contre l'Archive (invariants I-6 et I-7) sur le **même `data_version`**.
3. Vérifier les quotas structurels (max 40% d'une famille, max 60% d'un type, équilibre long/short).
4. Vérifier la diversité comportementale vs corpus (`signal_overlap` < seuil, `ret_corr` < seuil, cf. S-3.4).
5. Si tout passe → admission au corpus.
6. Sinon → archivage (concept #8) avec raison normalisée (catalogue S-3.6).

**Pourquoi pas MAP-Elites par défaut** : MAP-Elites est un excellent algorithme de quality-diversity, mais son gain sur des quotas + descripteurs comportementaux n'a pas été mesuré sur notre problème. À tester en V2 si les quotas se révèlent trop rigides. Pour V1, on garde une approche plus simple et mieux comprise.

#### Étape 6 — Évaluation finale unique sur le holdout

**Objectif** : publier la performance réelle de l'Einher final retenu, sans aucun recalibrage possible.

**Algorithme** : cf. `ONTOLOGY.md` S-3.8. Une seule passe, sur le holdout jamais consulté pendant le développement. SL/TP et N figés depuis S-3.2. Métriques publiées avec leur IC bootstrap.

### 10.3 Algorithmes rejetés pour V1 (inchangé)

| Algorithme | Raison du rejet pour V1 |
|---|---|
| LLM-augmented | Complexité opérationnelle, coût, non-déterminisme, instabilité des résultats |
| MCTS | Pas d'arbre de décision naturel dans notre problème |
| ILP | Bruyant, supervisé, coûteux |
| A*, B&B, CP, SAT | Pas de graphe d'état naturel |
| ACO, ABC, HS | Peu adaptés au symbolique |
| NS, ES, PSO purs | Pour optimisation continue, pas symbolique |
| SA, TS purs | Trop simples, beam search suffit |
| Mémétique complète (par défaut) | Beam search suffit pour le raffinement en V1 |
| Quality-Diversity (MAP-Elites, par défaut) | Quotas + descripteurs comportementaux suffisent pour V1 |
| Métrique composite figée (avant recalibrage) | Aucune pondération n'est validée sans protocole de calibration empirique |

### 10.4 Algorithmes retenus pour V1 (révisé)

| Étape | Algorithme | Rôle |
|---|---|---|
| 0 | **Moteur d'évaluation** (figé, audité) | Contrat d'évaluation hors-échantillon |
| 1 | **Règle humaine + énumération peu profonde + random search** | Baselines honnêtes |
| 2 | **Random / GE / GP typé / beam / mémétique / NSGA-II** (à comparer) | Comparaison reproductible |
| 3 | **Générateur retenu** (choix empirique) | Génération de production |
| 4 | **Beam search local** | Raffinement (sans recalibrage SL/TP) |
| 5 | **Quotas + descripteurs comportementaux + fingerprint canonique** | Admission au corpus |
| 6 | **Holdout sacré** (une seule passe) | Évaluation finale |

**Note 2026-08-03** : GE dispose du décodage BNF et produit des candidats. Sa boucle de sélection, croisement et mutation doit être finalisée avant d'être qualifiée de moteur évolutionnaire. Les six générateurs sont exposés via `make_all_generators`.
  1. `RandomSearchGenerator` (random search, pas d'engine)
  2. `BeamSearchGenerator` (vraie expansion par niveaux, requiert engine)
  3. `TypedGPGenerator` (STGP Koza+Montana, requiert engine)
  4. `GrammaticalEvolutionGenerator` (BNF 218 features + bloc relations OHLCV, chromosome 8 bits × 12 gènes, décodeur Ryan 1998, requiert engine)
  5. `MemeticGenerator` (EA TypedGP + LSO hill climbing, requiert engine)
  6. `NSGA2Generator` (Deb 2002 multi-objectif, requiert engine)

Le **comparateur multi-objectif** (Phase 4) classe les générateurs via un score composite :
  `0.40·norm_sharpe + 0.30·norm_admission + 0.15·norm_diversity + 0.15·norm_coherence`
  avec normalisation min-max entre moteurs, redistribution des poids si coherence=0.
  Le **pilotage** (`pilotage.py`) produit un rapport structuré par moteur (volume, perf, diversité, admissions, rejets).

### 10.5 Algorithmes à explorer en V2

| Algorithme | Bénéfice attendu | Trigger de passage en V2 |
|---|---|---|
| LLM-augmented SR | État de l'art 2025-2026, déployé industriellement | Si le générateur retenu plafonne et qu'on a l'infra |
| NSGA-II complet | Si la métrique composite (recalibrée) montre des trade-offs cachés | Si l'analyse Pareto révèle des solutions dominées |
| CPCV (Lopez de Prado) | Validation plus robuste | Déjà en V1 (CPCV léger pour PBO), full CPCV en V2 |
| Memetic complet | Si le générateur retenu stagne | Si taux d'admission < 5% |
| Quality-Diversity (MAP-Elites) | Si les quotas simples sont trop rigides | Si la diversité comportementale mesurée est insuffisante |
| Multi-objectif GP | Si on veut explorer plus de types d'Einhers simultanément | Si l'ontologie s'enrichit de nouveaux concepts |

### 10.6 Pourquoi cette architecture est cohérente avec le contrat conceptuel

| Invariant | Comment l'architecture le respecte |
|---|---|
| I-1 (Einher complet) | Tous les algorithmes opèrent sur des Einhers complets (trigger + SL/TP) |
| I-2 (Einher reproductible) | Moteur d'évaluation seedé ; générateurs déterministes (random avec seed) |
| I-3 (généralisable) | Validation train/val/holdout + DSR + PBO + cohérence cross-asset |
| I-4 (edge net de frais) | Coûts simulés (spread, commission, slippage) dans le moteur d'évaluation |
| I-5 (pas de fuite) | Splits disjoints + purging + embargo + SL/TP figés depuis train + holdout sacré |
| I-6 (pas de doublons) | Fingerprint canonique (structurel + comportemental) + Archive par `data_version` |
| I-7 (types non confondus) | GP typé / GE contrainte par BNF qui encode la compatibilité type/transformation |
| I-8 (diversité du corpus) | Quotas structurels + descripteurs comportementaux (overlap signaux, corrélation rendements) |
| I-9 (métriques objectives) | Toutes les métriques sont quantitatives, calculées sur des données hors-échantillon, accompagnées d'IC bootstrap |
| I-10 (deux étapes de validation) | Étape 1 baselines (rapide) et Étape 5 admission (stricte) sont strictement séparées |

### 10.7 Pourquoi cette architecture est simple

- **7 étapes** clairement définies, une responsabilité par étape.
- **Moteur d'évaluation = contrat unique** : tout le monde parle le même langage de Sharpe/PnL/MDD.
- **Décisions empiriques, pas idéologiques** : le générateur est choisi sur résultats, pas sur popularité.
- **Pas de dépendance externe** pour V1 (pas de LLM, pas d'API cloud).
- **Pas de mémétique complexe** : beam search suffit pour le raffinement.
- **Quotas + descripteurs** au lieu de MAP-Elites (plus simple, plus auditable).
- **Métrique composite mise en réserve** : si on en garde une, elle sera recalibrée empiriquement, pas affirmée a priori.

Le pipeline peut être implémenté en quelques milliers de lignes de Python, avec une base solide pour évoluer.

---

## 11. Recommandations finales et décisions provisoires (V1.1)

### 11.1 Décisions fermes (V1)

1. **Moteur d'évaluation construit en premier** (priorité 0). Sans lui, aucune comparaison n'a de sens.
2. **Splits train/val/holdout (60/20/20) avec purging/embargo, figés par `data_version`**. Holdout consulté une seule fois.
3. **SL/TP figés depuis le train**, jamais recalibrés.
4. **Critères d'admission révisés** : DSR + PBO (CPCV léger) + block bootstrap CI + cross-asset + n_trades + max_dd + diversité comportementale.
5. **Diversité comportementale obligatoire** pour l'admission (descripteurs exportés, seuils appliqués).
6. **Archive enrichie** : `data_version`, `seed`, `splits`, `costs_simulated`, snapshot métriques complet, fingerprint canonique, **réévaluable** sur nouveau `data_version`.
7. **Baselines d'abord** (règle humaine + énumération peu profonde + random) avant tout générateur sophistiqué.
8. **LLM exclu V1**, maintenu en V2.

### 11.2 Décisions à trancher empiriquement (V1)

1. **Générateur principal** : random / GE / GP typé / beam / mémétique / NSGA-II — choisi après la comparaison reproductible de l'étape 2.
2. **Pondération de la métrique composite** (si on en garde une) — recalibrée sur les baselines, pas affirmée a priori.
3. **Valeur exacte des seuils S-3.4** (DSR cible, PBO max, n_trades_min, dd_max, overlap_max, corr_max) — recalibrés après 50+ Einhers testés.
4. **Longueur de bloc du block bootstrap** — défaut `1.5 × N_max`, à valider empiriquement.
5. **Paramètres du CPCV léger** — K=6, N=6 par défaut, à valider empiriquement.

### 11.3 Algorithmes écartés pour V1 (inchangé)

1. **LLM-augmented SR** — trop complexe pour V1
2. **MCTS** — pas d'arbre naturel dans notre problème
3. **ILP, Rule learning, ARM, A*, CP, SAT** — incompatibles avec notre problème (bruyant, non supervisé, continu)
4. **ACO, ABC, HS, PSO, ES purs** — pour optimisation continue, pas symbolique
5. **SA, TS purs** — beam search suffit
6. **NeuroEvolution** — boîte noire, incompatible avec l'invariant d'interprétabilité

### 11.4 Risques identifiés

- **Coût de calcul** : 100 000 Einhers × 1-10 secondes d'évaluation = 1-10 jours CPU. Faisable mais pas gratuit. À paralléliser.
- **Surapprentissage** : malgré le train/val/holdout + DSR + PBO, le risque persiste. Les seuils S-3.4 sont **empiriques** (à recalibrer après 50+ Einhers testés). Le holdout n'a pas le droit de "réparer" une admission contestable.
- **Stabilité du LLM (V2)** : les résultats LLM-augmented sont très sensibles au prompt, à la température, au modèle. Nécessite une infrastructure de versioning et de rejouabilité.
- **Sur-confiance dans le DSR/PBO** : DSR et PBO sont des outils statistiques, pas des oracles. Avec 10 000-100 000 candidats, la barre DSR > 0.95 peut être trop laxiste ou trop stricte selon le régime. À calibrer empiriquement.
- **Look-ahead résiduel** : malgré le split train/val/holdout + purging + embargo, des fuites subtiles peuvent subsister (ex : features à fenêtre glissante qui dépassent la borne, par effet de rolling). Audit manuel requis sur les premiers runs.

### 11.5 Prochaines étapes concrètes (révisées 2026-08-03)

1. **Implémenter le moteur d'évaluation** (priorité 0), conforme à `ONTOLOGY.md` S-2 et S-3.1-S-3.3. Tests de non-régression et de non-fuite. ✅ FAIT
2. **Lancer les baselines** : règle humaine, énumération peu profonde, random search. Mesurer la distribution de Sharpe et PnL sur le val. ✅ FAIT
3. **Lancer la comparaison reproductible des générateurs** (random / GE / GP typé / beam), mêmes seeds, splits, budget, métriques, coûts. Choisir sur résultats. ✅ FAIT (6 générateurs comparables)
4. **Écrire la grammaire BNF** (si GE est retenu) couvrant : amplitude, univers, direction, constantes, canonisation, limites de profondeur. Sinon, formaliser la représentation interne de GP typé / beam. ✅ FAIT (BNF Phase 1 : 218/218 terminaux + bloc relations OHLCV, Lots 0-4e)
5. **Implémenter les descripteurs comportementaux** et le calcul de fingerprint canonique (structurel + comportemental). ✅ FAIT (P1 #7, P1 #8)
6. **Implémenter l'Archive enrichie** (nouveau schéma de stockage). ✅ FAIT (P1 #7, P1 #8)
7. **Lancer un premier run de bout en bout** sur 1 actif × 1 timeframe, en respectant scrupuleusement la discipline train/val/holdout. ⏳ Reste à faire
8. **Calibrer** les seuils S-3.4 et la métrique composite (si retenue) sur les résultats des baselines et des premiers runs. ⏳ Reste à faire
9. **Étendre** progressivement à plus d'actifs et de timeframes. ⏳ Reste à faire

**Chantier BNF (récap)** :
  - **Phase 1** (terminaux 218 features) : ✅ FAIT
  - **Phase 2** (anti-tautologies `compute_ohlcv_range_quantiles`) : ⏸ DIFFÉRÉ à la demande user
  - **Phase 3** (orientation sémantique 108 patterns) : ✅ FAIT
  - **Phase 4** (parser BNF + intégration GE) : ✅ FAIT

**Chantiers complémentaires livrés** :
  - **Comparateur multi-objectif** : ✅ FAIT (score composite 4 axes, branche `comparator-multiobj`)
  - **Pilotage** : ✅ FAIT (rapport structuré par moteur, branche `pilotage-report`)

---

## 12. Conclusion

L'état de l'art 2024-2026 de la découverte automatique de stratégies de trading est dominé par **deux tendances** : la **génétique programming (GP/GE)** comme méthode éprouvée et mature, et le **LLM-augmented symbolic regression** comme frontière de recherche récente, déjà déployée en production par des acteurs institutionnels (Man Group, Qraft, plusieurs boutiques quant).

Pour Einherjar, l'architecture révisée (V1.1) est un **pipeline à 7 étapes** qui inverse la priorité par rapport à l'architecture V1 précédente :
- **Le moteur d'évaluation est construit EN PREMIER** (étape 0), comme contrat d'évaluation hors-échantillon auditable.
- **Les baselines honnêtes passent AVANT** les générateurs sophistiqués (étape 1).
- **Le choix du générateur est EMPIRIQUE** (étape 2-3), pas idéologique : random / GE / GP typé / beam sont comparés sur mêmes seeds, mêmes splits, même budget, mêmes métriques, mêmes coûts.
- **L'admission exige des preuves statistiques de non-surapprentissage** : DSR + PBO (CPCV léger) + block bootstrap CI + cross-asset + n_trades + max_dd + diversité comportementale.
- **Le holdout est sacré** : consulté une seule fois, à la toute fin.
- **SL/TP figés depuis le train**, jamais recalibrés.
- **La diversité est comportementale**, pas seulement structurelle.

Cette architecture est **simple, modulaire, reproductible, et évolutive**. Elle respecte tous les invariants du contrat conceptuel. Elle peut être implémentée en quelques milliers de lignes de Python. Elle laisse la porte ouverte aux avancées futures (LLM-augmented, NSGA-II, full CPCV, MAP-Elites) sans les rendre bloquantes pour V1.

**La règle d'or** : ne pas chercher l'algorithme parfait, mais construire un pipeline qui **filet correctement** les quelques Einhers utiles dans le bruit. L'overfitting est le risque n°1, pas l'algorithme sous-optimal. La meilleure protection contre l'overfitting est **l'ordre dans lequel on construit les choses** : moteur d'évaluation d'abord, baselines ensuite, générateur sophistiqué en dernier, et holdout consulté une seule fois à la fin.

---

## 13. Critique IA tierce et révision V1.1

### 13.1 Contexte

Le 2026-08-01, le document V1 et l'ontologie V2 ont été soumis à une **revue par une IA tierce indépendante** (Claude Opus, dans le cadre d'une session dédiée de critique d'architecture). La critique a porté sur les deux livrables conjoints.

### 13.2 Points acceptés (8 sur 9)

| # | Critique | Action |
|---|---|---|
| 1 | **SL/TP calculés après le test** : look-ahead bias si « historique » inclut le test. SL/TP doivent être estimés exclusivement sur train, gelés, puis évalués sur val/holdout. | ✅ Corrigé : S-2 + S-3.2 imposent `sl_price`/`tp_price`/`N` calculés une fois sur le train, jamais recalibrés. |
| 2 | **L'évaluation brute n'évalue pas réellement TP/SL** : `hit_tp` défini comme `ret > 0`, `hit_sl_first` en placeholder, entrée au `close[idx+1]` au lieu de l'OPEN. | ✅ Corrigé : S-2 spécifie entrée à l'OPEN t+1, simulation intrabar TP/SL sur high/low, convention SL avant TP, distinction explicite des `exit_reason`. |
| 3 | **Horizon N fuyant et non stationnaire** : ATR moyen global inadapté aux régimes distincts et à plusieurs actifs. | ✅ Corrigé : N calculé une fois sur le train, `atr_p50` (médiane, plus robuste que la moyenne), `clamp` avec garde-fous `min_N`/`max_N`. |
| 4 | **Seuils statistiques insuffisants** : n≥30, Sharpe>0.5, cooldown 5 — ne suffisent pas avec autocorrélation, fenêtres qui se chevauchent, sélection massive. | ✅ Corrigé : S-3.4 substitue DSR + PBO (CPCV léger) + block bootstrap CI + cross-asset + n_trades + max_dd + diversité comportementale. |
| 5 | **Diversité structurelle ≠ indépendance économique** : deux règles structurellement différentes peuvent produire les mêmes signaux. | ✅ Corrigé : descripteurs comportementaux obligatoires (S-3.7), `signal_overlap` et `ret_corr` croisés avec le corpus (S-3.4). |
| 6 | **Grammaire BNF pas encore définie** : l'ontologie contient la table de typage, pas la BNF complète. Dire "déjà définie" masque un travail à faire. | ✅ Corrigé : ALGORITHME_RESEARCH.md § 1.11, § 7.11, § 8.11 et § 10 retirent "BNF déjà définie" et "grammaire déjà définie". La BNF est listée comme **à écrire** dans les prochaines étapes (§ 11.5). |
| 7 | **GE figée comme moteur principal sans comparaison** : affirmation "GE comme moteur principal" non étayée. | ✅ Corrigé : GE est désormais un **candidat à comparer** avec random / GP typé / beam sur mêmes seeds, splits, budget, métriques, coûts (§ 10.2 étape 2). |
| 8 | **Métrique composite pondérée sans protocole de calibration** : poids affirmés a priori. | ✅ Corrigé : aucune pondération n'est figée. Si on garde une métrique composite, ses poids sont **recalibrés empiriquement** sur les baselines, pas affirmés. NSGA-II reste en réserve. |

### 13.3 Points à approfondir (V2+)

| # | Critique | Action |
|---|---|---|
| 9 | **MAP-Elites écarté sans comparaison** : écarter QD "par principe" plutôt que sur résultats. | 🟡 Accepté partiellement : on garde les quotas + descripteurs comportementaux pour V1 (plus simple, plus auditable), et MAP-Elites est listé comme **candidat V2 avec trigger explicite** (§ 10.5) : "si la diversité comportementale mesurée est insuffisante". Ce n'est plus un rejet de principe. |

### 13.4 Points NON retenus (1 sur 9)

Aucun point de la critique n'a été intégralement rejeté. Les neuf points ont été soit acceptés, soit acceptés avec nuance. La critique est jugée **techniquement solide et alignée avec l'état de l'art**.

### 13.5 Leçons méthodologiques

Cette critique a fait apparaître trois biais récurrents dans la V1 :
1. **Biais de figeage** : transformer des "candidats" en "décisions" sans validation empirique.
2. **Biais de précision** : donner des chiffres et des pondérations sans protocole de calibration (ex : `0.5*retour - 0.3*volatilité`).
3. **Biais d'affirmation d'existence** : dire qu'un artefact (BNF complète, comparaisons entre générateurs) existe alors qu'il n'existe pas encore.

**Règle V1.1** : aucun artefact n'est affirmé comme "défini" ou "existant" tant qu'il n'a pas été produit. Les "candidats" restent candidats jusqu'à décision empirique. Les "seuils" restent "à recalibrer" jusqu'à données réelles. La grammaire BNF est listée comme **à écrire**, pas comme existante.

### 13.6 Documents révisés

| Document | V1 | V1.1 (2026-08-01) |
|---|---|---|
| `ONTOLOGY.md` | S-2 naïf, S-3 walk-forward 70/30 + 9 critères, Archive minimale | S-2 avec OPEN t+1 + intrabar + block bootstrap, S-3 train/val/holdout + DSR/PBO/bootstrap CI + SL/TP figés, Archive enrichie (data_version, seed, splits, fingerprint canonique) |
| `ALGORITHME_RESEARCH.md` | Pipeline 5 étages, GE figée, métrique composite pondérée figée, walk-forward 70/30, "BNF déjà définie" | Pipeline 7 étapes (étape 0 = moteur d'évaluation), générateur choisi empiriquement, splits train/val/holdout, BNF listée comme à écrire, métrique composite mise en réserve |

---

## Références

[1] Pippas, Ludvig, Turkay. "The Evolution of Reinforcement Learning in Quantitative Finance: A Survey." 2025. https://arxiv.org/pdf/2408.10932.pdf

[2] "From Deep Learning to LLMs: A survey of AI in Quantitative Finance." 2025. https://arxiv.org/html/2503.21422v1

[3] Lin et al. "FactorEngine: A Program-level Knowledge-Infused Factor Mining Framework for Quantitative Investment." 2026. https://arxiv.org/pdf/2603.16365v1.pdf

[4] Tang et al. "AlphaAgent: LLM-Driven Alpha Mining with Regularized Exploration to Counteract Alpha Decay." 2025. https://arxiv.org/abs/2502.16789

[5] Liu et al. "Cognitive Alpha Mining via LLM-Driven Code-Based Evolution." 2026. https://arxiv.org/html/2511.18850v3

[6] Shi et al. "Hubble: An LLM-Driven Agentic Framework for Safe and Automated Alpha Factor Discovery." 2026. https://arxiv.org/html/2604.09601v1

[7] Han et al. "QuantaAlpha: An Evolutionary Framework for LLM-Driven Alpha Mining." 2026. https://arxiv.org/html/2602.07085v2

[8] Yun, Lee, Jeon. "QuantEvolve: Automating Quantitative Strategy Discovery through Multi-Agent Evolutionary Framework." 2025. https://arxiv.org/html/2510.18569v1

[9] "Backtest overfitting in the machine learning era." 2024. https://www.sciencedirect.com/science/article/abs/pii/S0950705124011110

[10] "Combinatorial Purged Cross-Validation Insights." 2018 (foundational), 2024 update. https://www.scribd.com/document/725401650/SSRN-id4778909

[11] Long, Kampouridis. "Multi-objective genetic programming-based algorithmic trading, using directional changes and a modified sharpe ratio score for identifying optimal trading strategies." AIR 2025. https://repository.essex.ac.uk/41752/1/s10462-025-11390-9.pdf

[12] Long, Kampouridis, Kanellopoulos. "An In-Depth Investigation of Genetic Programming Under Physical Time and Directional Change Frameworks for Algorithmic Trading." IEEE Access 2025. https://repository.essex.ac.uk/41437/

[13] Long, Kampouridis. "α-dominance two-objective Optimization Genetic Programming for algorithmic trading under a directional changes environment." CIFEr 2024. https://repository.essex.ac.uk/38741/1/CIFEr_2024_paper_56.pdf

[14] "A Novel Strongly-Typed Genetic Programming Algorithm for Combining Sentiment and Technical Analysis for Algorithmic Trading." SSRN 2024. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4995172

[15] "Enhanced Strongly typed Genetic Programming for Algorithmic Trading." GECCO 2023. https://dl.acm.org/doi/pdf/10.1145/3583131.3590359

[16] "Evolving Financial Trading Strategies with Vectorial Genetic Programming." 2025. https://arxiv.org/html/2504.05418v1

[17] Cazenave. "Monte Carlo Search Algorithms Discovering Monte Carlo Tree Search Exploration Terms." 2024. https://arxiv.org/abs/2404.09304

[18] "ALPHA2: LLM-Driven Alpha Mining." 2024. http://arxiv.org/pdf/2406.16505.pdf

[19] "Navigating the Alpha Jungle: An LLM-Powered MCTS Framework for Formulaic Factor Mining." 2025. https://ui.adsabs.harvard.edu/abs/2025arXiv250511122S/abstract

[20] "Time-limited Metaheuristics for Cardinality-constrained Portfolio Optimisation." 2023. https://arxiv.org/abs/2307.04045

[21] Chen et al. "A memetic-based technical indicator portfolio and parameters optimization approach for finding trading signals to construct transaction robot in smart city era." 2023. https://journals.sagepub.com/doi/10.3233/IDA-220755

[22] "Evolutionary and Transformer based methods for Symbolic Regression." NeurIPS ML4PS 2024. https://ml4physicalsciences.github.io/2024/files/NeurIPS_ML4PS_2024_115.pdf

[23] "Scaling Up Unbiased Search-based Symbolic Regression." IJCAI 2024. https://www.ijcai.org/proceedings/2024/0471.pdf

[24] "Symbolic Regression with Self-Supervised Heuristic Beam Search." https://openreview.net/forum?id=3fsrvwLRr0

[25] Alves. "A Comparative Study of Technical Trading Strategies Using a Genetic Algorithm." 2024. https://dl.acm.org/doi/abs/10.1007/s10614-022-10348-1

[26] Han, Chen, Ye. "Stock portfolio optimization based on factor analysis and second-order memetic differential evolution algorithm." Memetic Comp. 2024. https://link.springer.com/article/10.1007/s12293-024-00405-7

[27] "High-Performance Machine Learning for FinTech." 2024. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5030509

[28] Badea. "Learning Trading Rules with Inductive Logic Programming." ECML 2000. https://colab.ws/articles/10.1007%2F3-540-45164-1_5

[29] "Neuro-symbolic Meta Reinforcement Learning for Trading." 2023. https://arxiv.org/pdf/2302.08996.pdf

[30] Keegstra. "Equity Trading by means of Interpretable Machine Learning." 2022. https://thesis.eur.nl/pub/59452/

[31] "A Comparative Analysis of RIPPER and CN2 Algorithms." 2023. https://www.diva-portal.org/smash/get/diva2:1882627/FULLTEXT01.pdf

[32] "Concise rule induction algorithm based on one-sided heuristics." 2024. https://www.sciencedirect.com/science/article/abs/pii/S0957417423018675

[33] "Unlocking Market Alpha: A Strategic Guide to Sequential Pattern Mining in Finance with Python." 2024. https://www.quantlabsnet.com/post/unlocking-market-alpha-a-strategic-guide-to-sequential-pattern-mining-in-finance-with-python

[34] "Alpha Discovery via Grammar-Guided Learning and Search." 2026. https://www.arxiv.org/pdf/2601.22119.pdf

[35] "End-to-End Symbolic Regression of Alpha Factors with Transformers." 2024. https://openreview.net/pdf/3e2990e745a7ffb867a5a7df9fa26b724a680e2b.pdf

[36] "AlphaForge: A Framework to Mine and Dynamically Combine Formulaic Alpha Factors." 2024. https://arxiv.org/html/2406.18394v3

[37] "Association Mining on Stock Index Indicators." 2013. https://www.ijcce.org/vol4/380-C032.pdf

[38] "Advanced Stock Market Forecasting Using Synergic of Sentiment Analysis and Association Rule Mining." 2024. https://home.agh.edu.pl/~horzyk/presentation/Horzyk_Adrian_6162_Advanced_Stock_Market_Forecasting_Using_Synergic_of_Sentiment_Analysis_and_Association_Rule_Mining.pdf

[39] Prasad et al. "Optimal Technical Indicator-based Trading Strategies Using NSGA-II." 2021. https://arxiv.org/pdf/2111.13364

[40] Wu et al. "PASS: Portfolio Analysis of Selecting Strategies on quantitative trading via NSGA-II." 2024. https://www.impactio.com/publication-attachments/2077/2909851977.pdf

[41] Alonso et al. "Collaborative Multiobjective Evolutionary Algorithms in the Search of Better Pareto Fronts: An Application to Trading Systems." 2023. https://www.mdpi.com/2076-3417/13/22/12485

[42] Zheng, Doerr. "Mathematical runtime analysis for NSGA-II." AIJ 2024. https://arxiv.org/pdf/2407.17687.pdf

[43] Chen, Hsu, Hong. "An Optimization Approach for Finding Diverse Trading Strategy Portfolio Using the Memetic Algorithm." ACIIDS 2024. https://dl.acm.org/doi/abs/10.1007/978-981-97-4982-9_25

[44] "Alpha-GPT: Human-AI Interactive Alpha Mining for Quantitative Investment." EMNLP Demos 2025. https://aclanthology.org/2025.emnlp-demos.14.pdf

[45] Kou et al. "Automate Strategy Finding with LLM in Quant Investment." EMNLP Findings 2025. https://aclanthology.org/2025.findings-emnlp.1005.pdf

[46] "Man Group's AI Quant System Replacing Junior Traders." 2025. https://www.youtube.com/watch?v=vs0WXLyhses

[47] "Grammar-based genetic programming: a survey." https://scispace.com/pdf/grammar-based-genetic-programming-a-survey-50iq33za3h.pdf

---

**État du document** : **Non figé — V1.1 en cours** (2026-08-01). Voir § 13 pour la liste exhaustive des révisions suite à la critique IA tierce.

**Prochaines actions** (par ordre de priorité, cf. § 11.5) :
1. Implémenter le **moteur d'évaluation** (priorité 0) — entrée OPEN t+1, simulation intrabar TP/SL, block bootstrap CI, splits train/val/holdout avec purging/embargo.
2. Lancer les **baselines** (règle humaine, énumération peu profonde, random).
3. Lancer la **comparaison reproductible des générateurs** (random / GE / GP typé / beam).
4. **Écrire la grammaire BNF** complète si GE est retenue.
5. Implémenter les **descripteurs comportementaux** et le fingerprint canonique.
6. Implémenter l'**Archive enrichie** (nouveau schéma).
7. Lancer un **premier run de bout en bout** sur 1 actif × 1 timeframe.
8. **Calibrer** les seuils S-3.4 et la métrique composite (si retenue) sur résultats.
9. Étendre à plus d'actifs et de timeframes.
