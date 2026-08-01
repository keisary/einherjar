# EINHERJAR RESEARCH ARCHITECTURE
### Scientific Alpha Discovery Engine

Version : 2.0 (Research Rewrite)

Status : Draft

Author : OpenAI & Project EINHERJAR

---

# 1. Introduction

## 1.1 Objectif du document

Ce document constitue la référence technique officielle du moteur de recherche d'EINHERJAR.

Il décrit les principes scientifiques, l'architecture logicielle, les algorithmes, les structures de données et les décisions d'ingénierie qui régissent le fonctionnement du système.

Aucune décision d'implémentation ne doit contredire ce document.

En cas de divergence entre le code et cette documentation, la documentation fait foi.

---

# 1.2 Pourquoi une réécriture ?

La première génération du moteur Discovery avait un objectif simple :

Découvrir automatiquement des règles de trading rentables.

Cette approche a permis de produire plusieurs milliers de stratégies mais a également mis en évidence plusieurs limites fondamentales.

Parmi les principales :

- domination excessive de certaines familles de features ;
- recherche guidée principalement par un score local ;
- confusion entre découverte, validation et optimisation ;
- simulation différente du contrat de données utilisé pendant l'apprentissage ;
- absence de mémoire du moteur ;
- absence de compréhension des causes de réussite ou d'échec ;
- architecture monolithique difficile à maintenir.

Ces limitations rendent difficile l'évolution du moteur et augmentent le risque de surapprentissage.

Une simple amélioration incrémentale ne permettrait pas de résoudre ces problèmes.

La version 2 constitue donc une réécriture complète du moteur de recherche.

---

# 1.3 Vision

EINHERJAR n'est pas un générateur de stratégies.

EINHERJAR n'est pas un optimiseur de paramètres.

EINHERJAR est un moteur de découverte scientifique appliqué aux marchés financiers.

Sa mission consiste à identifier des comportements récurrents du marché, à démontrer statistiquement leur existence puis à les transformer en connaissances exploitables.

Le moteur ne cherche donc pas directement une stratégie.

Il cherche à découvrir un phénomène de marché.

Une stratégie devient simplement une représentation exécutable de ce phénomène.

---

# 1.4 Philosophie

La philosophie générale du projet repose sur cinq principes.

## Principe 1

Une hypothèse est plus importante qu'un score.

Le moteur ne cherche pas la règle possédant le meilleur Sharpe.

Il cherche la meilleure hypothèse de marché.

---

## Principe 2

Toute hypothèse doit pouvoir être expliquée.

Une stratégie rentable mais incompréhensible possède peu de valeur scientifique.

Le moteur doit être capable d'expliquer :

- pourquoi une hypothèse fonctionne ;
- dans quelles conditions elle fonctionne ;
- dans quelles conditions elle échoue.

---

## Principe 3

Le moteur doit apprendre.

Chaque recherche enrichit les connaissances du système.

Une région de recherche déjà explorée ne doit pas être oubliée.

Les hypothèses rejetées sont également des connaissances.

---

## Principe 4

Le moteur privilégie la robustesse à la performance.

Une stratégie légèrement moins rentable mais robuste sera toujours préférée à une stratégie spectaculaire mais fragile.

---

## Principe 5

La découverte et l'optimisation sont deux problèmes différents.

Découvrir un edge de marché consiste à démontrer qu'un comportement existe.

Optimiser une stratégie consiste ensuite à exploiter ce comportement.

Ces deux étapes ne doivent jamais être mélangées.

---

# 2. Mission d'EINHERJAR

Le moteur poursuit un objectif unique.

Découvrir automatiquement des anomalies statistiques persistantes dans les marchés financiers capables de produire une croissance durable du capital sur des données futures.

Pour atteindre cet objectif, chaque hypothèse devra satisfaire simultanément plusieurs critères indépendants :

- validité statistique ;
- cohérence économique ;
- robustesse temporelle ;
- robustesse structurelle ;
- diversité comportementale ;
- potentiel de croissance.

Aucun critère unique ne peut à lui seul justifier la conservation d'une hypothèse.

---

# 3. Définition d'un Einher

Un Einher est une connaissance de marché validée.

Il ne s'agit pas simplement d'une règle logique.

Un Einher possède :

- une hypothèse de marché ;
- une représentation logique ;
- une validation scientifique ;
- un historique d'exécution ;
- un profil comportemental ;
- une empreinte (Fingerprint) ;
- un journal de trades ;
- une relation avec les autres Einhers.

Un Einher représente donc une unité de connaissance.

Le corpus final est une base de connaissances et non une simple liste de stratégies.

---

# 4. Cycle de vie d'un Einher

Chaque Einher suit un cycle de vie parfaitement défini.

Hypothesis

↓

Candidate

↓

Validated Candidate

↓

Execution Analysis

↓

Einher

↓

Portfolio Candidate

↓

Production

↓

Monitoring

↓

Archive

Aucun objet ne saute une étape.

Chaque transition ajoute des connaissances sans modifier les connaissances précédentes.

Le moteur ne détruit jamais une hypothèse.

Il change uniquement son état.

---

# 5. Architecture Générale

Le moteur est organisé en cinq grandes phases indépendantes.

Phase A

Contrat des données

Garantit que les observations utilisées représentent fidèlement le marché.

---

Phase B

Discovery Engine

Explore intelligemment l'espace des hypothèses.

---

Phase C

Scientific Validation Engine

Mesure la crédibilité scientifique des hypothèses.

---

Phase D

Execution & Knowledge Engine

Transforme une hypothèse validée en connaissance exploitable.

---

Phase E

Portfolio Intelligence Engine

Construit un portefeuille optimal d'Einhers complémentaires.

Chaque phase possède une responsabilité unique.

Aucune phase ne peut modifier les responsabilités d'une autre.

# 6. Architecture Générale du Système

# 6.1 Vision Globale

L'architecture d'EINHERJAR V2 repose sur une idée simple :

Le moteur ne recherche plus directement des stratégies.

Il construit progressivement une connaissance du marché.

Chaque étape ajoute une nouvelle couche d'information.

Aucune étape ne détruit les informations produites par les précédentes.

Le moteur devient ainsi un système cumulatif de découverte scientifique.

L'objectif n'est plus uniquement de produire un corpus performant.

L'objectif est de construire une base de connaissances capable de s'enrichir continuellement.

---

# 6.2 Vue d'ensemble

L'architecture complète est organisée autour d'un pipeline orienté connaissances.




                     DATASET MIDAS
                           │
                           │
                           ▼
                ┌────────────────────┐
                │   Phase A          │
                │ Data Contract      │
                └────────────────────┘
                           │
                           ▼
                Dataset Certifié
                           │
                           ▼
                ┌────────────────────┐
                │   Phase B          │
                │ Discovery Engine   │
                └────────────────────┘
                           │
                           ▼
                  Hypothèses Candidates
                           │
                           ▼
                ┌────────────────────┐
                │   Phase C          │
                │ Validation Engine  │
                └────────────────────┘
                           │
                           ▼
                  Hypothèses Validées
                           │
                           ▼
                ┌────────────────────┐
                │   Phase D          │
                │ Knowledge Engine   │
                └────────────────────┘
                           │
                           ▼
                       Einhers
                           │
                           ▼
                ┌────────────────────┐
                │   Phase E          │
                │ Portfolio Engine   │
                └────────────────────┘
                           │
                           ▼
                  Portefeuilles Optimisés


# 6.3 Les quatre flux du moteur

Le moteur ne transporte pas uniquement des données.

Quatre flux circulent simultanément.

## Flux 1

Flux des données

Dataset

↓

Hypothèses

↓

Einhers


C'est le flux principal.

---

## Flux 2

Flux de connaissances

Recherche

↓

Connaissances

↓

Mémoire

↓

Réutilisation

Chaque exécution enrichit la mémoire.

Le moteur devient progressivement meilleur.


## Flux 3

Flux d'apprentissage

Succès

+

Echecs

↓

Analyse

↓

Nouvelles hypothèses


Les échecs sont aussi importants que les réussites.

---

## Flux 4

Flux de décision


Configuration

↓

Budget

↓

Recherche

↓

Validation

↓

Export


Tous les paramètres sont pilotés depuis la configuration.

Le code ne contient jamais de constantes métier.

---

# 6.4 Les cinq moteurs

EINHERJAR est constitué de cinq moteurs indépendants.

Ils communiquent mais ne connaissent jamais l'implémentation interne des autres.

---

## A — Data Contract Engine

Mission

Garantir que toutes les données respectent un contrat unique.

Responsabilités

• Charger MIDAS

• Vérifier la cohérence

• Mapper les features

• Mapper les horizons

• Produire un dataset certifié

Interdictions

• Ne valide aucune stratégie

• Ne calcule aucun score

---

## B — Discovery Engine

Mission

Explorer l'espace des hypothèses.

Entrée

Dataset certifié

Sortie

Liste de candidats

Responsabilités

• Génération

• Expansion

• Exploration

• Gestion du budget

• Diversité

• Mémoire de recherche

Le Discovery ne juge jamais une hypothèse.

Il propose uniquement des candidats.

---

## C — Scientific Validation Engine

Mission

Déterminer si une hypothèse mérite d'être conservée.

Responsabilités

Validation économique

Validation statistique

Validation temporelle

Validation structurelle

Validation comportementale

Validation de robustesse

Le moteur ne produit jamais un TP.

Le moteur ne modifie jamais une règle.

Il juge uniquement.

---

## D — Execution & Knowledge Engine

Mission

Transformer une hypothèse en connaissance.

Responsabilités

Replay exact

Journal de trades

Profil

Fingerprint

MAE

MFE

Diagnostics

Rapports

Apprentissage

Le moteur ne cherche plus.

Il comprend.

---

## E — Portfolio Intelligence Engine

Mission

Construire un portefeuille optimal.

Responsabilités

Sélection

Allocation

Corrélations

Diversification

Gestion du risque

Optimisation du capital

Le portefeuille travaille sur des Einhers.

Jamais sur des règles.

---

# 6.5 Principe fondamental

Chaque moteur ne possède qu'une seule responsabilité.


Discovery

↓

Découvrir



Validation

↓

Valider



Execution

↓

Comprendre



Portfolio

↓

Combiner


Si un moteur réalise deux responsabilités, l'architecture est incorrecte.

---

# 6.6 Architecture interne

Chaque moteur est lui-même composé de modules indépendants.

Exemple


Discovery Engine

│

├── Generator

├── Explorer

├── Family Manager

├── Novelty

├── Expansion

├── Scheduler

├── Search Budget

└── Report


Chaque module possède une API publique.

Aucun module ne modifie directement les données d'un autre.

---

# 6.7 Communication entre modules

Les modules communiquent uniquement grâce aux objets du domaine.

Jamais par des dictionnaires anonymes.

Exemple

Correct


Hypothesis

↓

Candidate

↓

ValidatedCandidate

↓

Einher


Incorrect


dict

↓

dict

↓

dict


Toutes les structures importantes possèdent leur propre classe.

---

# 6.8 Les objets du domaine

Le moteur manipule uniquement des objets métier.

Les principaux sont :

Feature

Condition

Hypothesis

Candidate

ValidatedCandidate

Trade

TradeJournal

Fingerprint

Profile

Einher

Portfolio

RunReport

KnowledgeNode

Chaque objet possède :

• un identifiant unique

• un cycle de vie

• une responsabilité

• une méthode de sérialisation

• un historique

---

# 6.9 Le Registry

Tous les moteurs utilisent un registre central.

Le Registry est la mémoire instantanée du système.

Il contient :

Toutes les Features

Toutes les familles

Toutes les hypothèses

Tous les candidats

Tous les Einhers

Toutes les statistiques

Tous les rapports

Toutes les références

Le Registry remplace les variables globales.

---

# 6.10 La mémoire

Le moteur possède une mémoire persistante.

Elle est indépendante du corpus.

Elle stocke notamment :

Zones déjà explorées

Zones prometteuses

Zones mortes

Historique des recherches

Historique des familles

Historique des performances

Historique des budgets

Historique des validations

Ainsi deux exécutions successives ne repartent jamais de zéro.

---

# 6.11 Le graphe de connaissances

Les Einhers ne sont jamais isolés.

Ils sont reliés dans un graphe.

Exemple


Einher A

↓

complète

↓

Einher B

↓

similaire

↓

Einher C

↓

inverse

↓

Einher D


Ce graphe permettra plus tard :

• recommandations

• clustering

• sélection de portefeuille

• détection des doublons

• recherche intelligente

---

# 6.12 Pipeline d'une hypothèse

Une hypothèse suit toujours le même parcours.


Feature

↓

Condition

↓

Hypothesis

↓

Candidate

↓

Validated Candidate

↓

Execution Analysis

↓

Einher

↓

Portfolio Candidate

↓

Production

↓

Monitoring

↓

Archive


Aucune étape ne peut être sautée.

Chaque étape ajoute des connaissances.

Aucune étape ne supprime les connaissances précédentes.

---

# 6.13 Les états

Chaque objet possède un état.

Exemple

Hypothesis


NEW

↓

EXPLORED

↓

EXPANDED

↓

REJECTED

↓

ARCHIVED


Einher


VALIDATED

↓

PROFILED

↓

OPTIMIZED

↓

READY

↓

ACTIVE

↓

DEGRADED

↓

ARCHIVED


Ainsi le moteur connaît en permanence l'état exact de chaque connaissance.

---

# 6.14 Les sorties

Une exécution produit plusieurs catégories de résultats.

Corpus

Einhers validés

Rejets

Hypothèses rejetées

Journaux

Trades

Rapports

Statistiques

Analytics

Visualisations

Mémoire

Historique

Archives

Toutes ces sorties sont indépendantes.

Aucune n'est supprimée automatiquement.

---

# 6.15 Architecture orientée connaissance

Le changement majeur introduit par EINHERJAR V2 est le suivant.

Le moteur ne manipule plus des stratégies.

Il manipule de la connaissance.

Une stratégie devient simplement une représentation opérationnelle d'une connaissance de marché.

Cette distinction est fondamentale.

Elle permet au moteur :

• d'apprendre ;

• de mémoriser ;

• d'expliquer ;

• de généraliser ;

• d'évoluer sans perdre les découvertes précédentes.

L'objectif final n'est donc plus de construire un catalogue de règles.

L'objectif est de construire une base de connaissances scientifiques sur le comportement des marchés financiers.

Ce principe constitue le fondement de toute l'architecture présentée dans ce document.

# 7. Domain Model

---

# 7.1 Philosophie

L'ensemble d'EINHERJAR repose sur un principe unique :

**Tout est un objet métier.**

Le moteur ne manipule jamais directement des dictionnaires anonymes, des listes de valeurs ou des structures non documentées.

Chaque concept du domaine possède son propre objet.

Cette approche présente plusieurs avantages :

- typage fort ;
- meilleure lisibilité ;
- sérialisation uniforme ;
- évolution facilitée ;
- réduction des erreurs ;
- documentation implicite du système.

Le Domain Model constitue le cœur de l'architecture.

Toutes les phases (A, B, C, D et E) manipulent exclusivement ces objets.

---

# 7.2 Cycle de vie global

Un objet ne change jamais de nature.

Il évolue.


Feature

↓

Condition

↓

Hypothesis

↓

Candidate

↓

ValidatedCandidate

↓

Einher

↓

PortfolioMember


Chaque transition ajoute des informations.

Aucune transition ne supprime les informations existantes.

---

# 7.3 Objet Feature

## Mission

Une Feature représente une mesure élémentaire calculée par MIDAS.

Elle constitue la plus petite unité d'information manipulée par le moteur.

Une Feature ne possède aucune logique métier.

Elle décrit uniquement une variable disponible dans le dataset.

---

## Exemple


RSI_14

ATR_PERCENTILE

ADX

EMA_DISTANCE

CMF

VWAP_DISTANCE


---

## Attributs


id

name

family

description

dtype

normalization

source

timeframe

metadata


---

## Responsabilités

Une Feature :

- connaît sa famille ;
- connaît son origine ;
- connaît son type ;
- peut être sérialisée.

Elle ne possède aucune connaissance statistique.

---

# 7.4 Objet Family

## Mission

Une Family regroupe plusieurs Features décrivant un même phénomène.

Exemple


Momentum

Trend

Volatility

Volume

Structure

Microstructure

Price Action


---

## Pourquoi ?

Le moteur ne recherche pas directement parmi les Features.

Il répartit son budget par familles.

Cette approche empêche une domination excessive d'un petit groupe d'indicateurs.

---

## Attributs


id

name

description

priority

search_budget

features

statistics


---

# 7.5 Objet Condition

## Mission

Une Condition représente une affirmation logique.

Exemple


RSI < 20

ATR > 95%

ADX >= 30

Volume ZScore > 2


Une Condition n'est pas une stratégie.

Elle est un fait.

---

## Attributs


feature

operator

threshold

confidence

support

metadata


---

## Responsabilités

Une Condition :

- connaît la Feature utilisée ;
- connaît son seuil ;
- peut être évaluée sur une observation.

Elle ne connaît jamais les autres Conditions.

---

# 7.6 Objet Hypothesis

## Mission

Une Hypothesis représente une idée de marché.

Elle est composée d'une ou plusieurs Conditions.

Une Hypothesis n'a encore jamais été validée.

Elle représente uniquement une possibilité.

---

## Exemple


RSI < 20

AND

ATR Percentile > 80%


---

## Attributs


id

conditions

families

complexity

generation

parent

children

origin

creation_time

status

metadata


---

## Etats


NEW

EXPLORED

EXPANDED

REJECTED

ARCHIVED


---

## Responsabilités

Une Hypothesis :

- peut être développée ;
- peut générer des enfants ;
- connaît son historique.

Elle ne possède encore aucune statistique.

---

# 7.7 Objet Candidate

## Mission

Une Candidate est une Hypothesis ayant terminé la Phase B.

Elle possède désormais des résultats de recherche.

---

## Attributs supplémentaires


search_score

novelty_score

family_score

coverage

beam_history

visited_nodes

expansion_depth

discovery_report


---

## Responsabilités

Une Candidate est prête à être validée.

Elle ne connaît toujours pas sa rentabilité réelle.

---

# 7.8 Objet ValidationReport

Mission

Décrire complètement les résultats de la Phase C.

Une Candidate possède un ValidationReport.

Jamais l'inverse.

---

## Contenu


expectancy

profit_factor

psr

dsr

p_value

walk_forward

robustness

persistence

diversity

economic_score

structural_score

status


---

# 7.9 Objet ValidatedCandidate

Mission

Une Candidate ayant satisfait tous les critères scientifiques.

Elle devient éligible à la Phase D.

---

## Attributs supplémentaires


validation_report

validation_date

validation_version

scientific_confidence

accepted_tests

failed_tests


---

## Responsabilités

Elle ne possède encore aucun Trade Journal.

---

# 7.10 Objet Trade

Mission

Représenter un trade unique.

Le Trade devient l'unité fondamentale de connaissance de la Phase D.

---

## Attributs


id

asset

timeframe

entry_time

exit_time

entry_price

exit_price

duration

return

fees

slippage

mae

mfe

drawdown

reason_exit

metadata


---

## Responsabilités

Un Trade est immuable.

Il ne change jamais après sa création.

---

# 7.11 Objet TradeJournal

Mission

Stocker l'ensemble des Trades d'un Einher.

---

## Contenu


Trades

↓

Statistics

↓

Diagnostics

↓

Insights


---

## Statistiques

Le journal calcule notamment :

- nombre de trades ;
- expectancy ;
- durée moyenne ;
- MAE moyen ;
- MFE moyen ;
- Profit Factor ;
- Drawdown ;
- distribution temporelle.

---

# 7.12 Objet Fingerprint

Mission

Décrire l'identité comportementale d'un Einher.

Le Fingerprint permet de comparer deux Einhers sans regarder leurs Conditions.

---

## Exemple


Momentum

92%

Trend

18%

Volatility

65%

Mean Reversion

88%

Intraday

40%

Swing

75%


---

## Utilisations

- clustering ;
- recherche ;
- portfolio ;
- détection des doublons.

---

# 7.13 Objet Profile

Mission

Décrire le comportement global d'un Einher.

Le Profile est une synthèse de haut niveau.

---

## Contenu


Market Regime

Preferred Volatility

Holding Style

Risk Level

Expected Growth

Typical Drawdown

Market Preference

Timeframe Preference

Confidence


---

# 7.14 Objet Einher

Mission

L'Einher représente une connaissance scientifique validée.

Il constitue l'objet le plus important du moteur.

---

## Composition

Un Einher contient :


Hypothesis

+

ValidationReport

+

TradeJournal

+

Fingerprint

+

Profile

+

Knowledge

+

History


---

## Etats


VALIDATED

PROFILED

OPTIMIZED

READY

ACTIVE

DEGRADED

ARCHIVED


---

## Responsabilités

Un Einher :

- possède une identité unique ;
- possède une histoire ;
- possède une confiance scientifique ;
- possède un comportement documenté ;
- peut être utilisé dans un portefeuille.

---

# 7.15 Objet Portfolio

Mission

Représenter un ensemble cohérent d'Einhers.

Le Portfolio ne contient jamais de règles.

Il contient uniquement des Einhers.

---

## Contenu


Members

Weights

Capital Allocation

Risk Allocation

Correlation Matrix

Performance

Diagnostics


---

# 7.16 Objet RunReport

Mission

Décrire entièrement une exécution du moteur.

---

## Contenu


Execution Time

CPU

Memory

Hypotheses Generated

Candidates

Validated

Rejected

New Einhers

Portfolio

Errors

Warnings


---

# 7.17 Objet KnowledgeNode

Mission

Construire le graphe de connaissances.

Chaque KnowledgeNode représente un Einher.

---

## Relations


SIMILAR_TO

PARENT_OF

CHILD_OF

COMPLEMENTS

CONTRADICTS

CORRELATED

SPECIALIZES

GENERALIZES


Le moteur pourra ainsi naviguer dans son corpus de connaissances.

---

# 7.18 Règles d'évolution des objets

Tous les objets suivent les règles suivantes :

## 1

Un objet ne perd jamais d'information.

---

## 2

Un objet ne change jamais de type.

Une Hypothesis ne devient pas un Einher.

Elle est encapsulée dans un nouvel objet.

---

## 3

Les objets sont immuables dès qu'ils changent de phase.

Les modifications produisent une nouvelle version.

---

## 4

Chaque objet possède un identifiant universel (UUID).

Les identifiants ne sont jamais recyclés.

---

## 5

Tous les objets sont sérialisables.

Ils doivent pouvoir être exportés en :

- JSON
- Parquet
- DataFrame

sans perte d'information.

---

# 7.19 Diagramme des relations


Feature
   │
   ▼
Condition
   │
   ▼
Hypothesis
   │
   ▼
Candidate
   │
   ▼
ValidatedCandidate
   │
   ├──────────────┐
   ▼              ▼
Trade         ValidationReport
   │              │
   └──────┬───────┘
          ▼
     TradeJournal
          │
          ▼
      Fingerprint
          │
          ▼
        Profile
          │
          ▼
         Einher
          │
          ▼
       Portfolio


---

# 7.20 Principe fondamental

Le Domain Model constitue le langage commun d'EINHERJAR.

Toutes les phases du moteur manipulent exactement les mêmes objets.

Cette homogénéité garantit :

- une architecture stable ;
- une maintenance simplifiée ;
- une forte extensibilité ;
- une documentation implicite du code ;
- une séparation claire entre la logique métier et les algorithmes.

Aucun module ne doit manipuler directement des structures anonymes lorsqu'un objet métier existe déjà pour représenter cette information.

# ==========================================================
# CHAPITRE 8 — CONTRAT D'ARCHITECTURE DU DOMAIN MODEL
# ==========================================================

## Objectif

Après plusieurs phases de réflexion, l'architecture du moteur est considérée comme figée.

Les développements futurs ne doivent plus remettre en cause les concepts fondamentaux du Domain Model sauf découverte d'un bug majeur.

La priorité est désormais la production de code robuste et cohérent.

---

# Philosophie générale

EINHERJAR n'est pas un moteur de recherche statistique.

EINHERJAR est un Discovery Engine chargé de découvrir des comportements de marché réellement exploitables.

L'objectif n'est donc PAS :

- maximiser une p-value,
- maximiser un Sharpe,
- découvrir le plus grand nombre de règles.

L'objectif est :

Découvrir des Einhers capables de faire croître durablement un capital réel tout en restant robustes hors échantillon.

Toutes les décisions d'architecture doivent servir cet objectif.

---

# Définition officielle d'un Einher

Un Einher est une fonction qui transforme un état du marché en une décision.

Etat du marché

↓

Hypothèse

↓

Validation

↓

Décision LONG ou SHORT

↓

Simulation

↓

Einher

Le moteur d'exécution reste responsable :

- du Take Profit
- du Stop Loss
- du Time Exit
- des règles de sortie

Les Einhers V2 ne découvrent que les conditions d'entrée.

---

# Contrat officiel des données MIDAS

Le dataset MIDAS constitue la source de vérité.

Les matrices X_*.npy représentent les données officielles utilisées par le Discovery Engine.

Une Feature est identifiée principalement par :

- son index de colonne

et secondairement par :

- son nom.

Le nom est uniquement un alias lisible.

Le moteur doit toujours privilégier les index afin de garantir la synchronisation parfaite avec les matrices numpy.

Le fichier metadata.json constitue le mapping officiel entre :

index
↓

feature

Aucun code ne doit dépendre de l'ordre alphabétique des features.

Les horizons suivent exactement le même principe.

---

# Architecture officielle du Domain Model

Le Domain Model suit obligatoirement cette hiérarchie :

Feature

↓

Expression

↓

Condition

↓

Hypothesis

↓

Candidate

↓

ValidatedCandidate

↓

Einher

Chaque niveau possède une responsabilité unique.

Aucun niveau ne doit fusionner plusieurs responsabilités.

---

# Feature

Une Feature décrit une variable disponible dans MIDAS.

Une Feature ne contient jamais les données.

Elle décrit uniquement :

- son index
- son nom
- sa famille économique
- son type
- son type de valeur
- ses opérateurs autorisés
- sa politique de génération des seuils
- son coût de découverte
- ses métadonnées

Une Feature constitue une description statique.

Les valeurs sont contenues uniquement dans les matrices X.

---

# Feature Registry

Le FeatureRegistry constitue la source officielle des Features.

Il est chargé :

- du chargement depuis metadata.json
- du mapping index → Feature
- du mapping nom → Feature
- des recherches par famille
- des recherches par type
- des statistiques
- de la validation des index

Le Discovery Engine ne manipule jamais directement des dictionnaires de Features.

Toutes les requêtes passent par le registre.

---

# Expressions

Le moteur manipule des Expressions.

Jamais directement des Features.

Une Expression représente une valeur pouvant être évaluée.

Pour la V2, seules trois implémentations sont prévues :

- FeatureExpression
- ConstantExpression
- FunctionExpression

Les BinaryExpression ainsi que les arbres mathématiques complexes sont volontairement reportés à une version ultérieure.

---

# Conditions

Une Condition compare toujours deux Expressions.

Structure officielle :

LEFT_EXPRESSION

↓

OPERATOR

↓

RIGHT_EXPRESSION

Exemples :

RSI > 50

Pattern_Double_Top == True

ATR > EMA(ATR,20)

VWAP > Close

Une Condition ne connaît aucune logique de validation statistique.

Elle représente uniquement une relation logique.

---

# Hypothèses

Une Hypothesis est une collection ordonnée de Conditions.

Toutes les Conditions sont reliées par un ET implicite.

La V2 ne supporte volontairement pas :

- OR
- NOT
- arbres logiques complexes

Cette évolution reste prévue pour une version future.

---

# Discovery Engine

Le Discovery Engine constitue le cœur du système.

Il ne repose pas sur un simple Beam Search.

Il combine plusieurs stratégies :

- exploration
- exploitation
- diversification
- mémoire
- régions mortes
- novelty search
- budgets dynamiques
- pénalisation des doublons

Le Beam Search n'est qu'une heuristique parmi d'autres.

---

# Fingerprints

Toutes les entités importantes doivent posséder un fingerprint déterministe.

Au minimum :

- Expression
- Condition
- Hypothesis
- Candidate
- Einher

Les fingerprints servent :

- à détecter les doublons
- aux caches
- à la mémoire
- au graphe de connaissances

---

# Immutabilité

Les objets métier sont considérés comme immuables.

Ils doivent être implémentés avec :

@dataclass(
    frozen=True,
    slots=True
)

lorsque cela est possible.

Les objets représentant un processus en cours peuvent rester mutables.

---

# Séparation des responsabilités

Le Domain Model ne contient aucune logique métier complexe.

Les modèles peuvent uniquement fournir :

- validation interne
- fingerprint
- clone
- sérialisation
- helpers

Ils ne doivent jamais :

- explorer
- valider
- optimiser
- simuler
- rechercher

Ces responsabilités appartiennent exclusivement au dossier core/.

---

# Compatibilité

Toutes les évolutions futures doivent conserver la compatibilité avec :

- metadata.json
- X_*.npy
- les index des colonnes
- les index des horizons

Toute modification rompant cette compatibilité est considérée comme une rupture d'architecture.

---

# Principe de développement

La phase de conception est désormais considérée comme terminée.

Les prochaines étapes consistent uniquement à produire les fichiers du projet.

Les futures conversations doivent privilégier :

- du code complet
- du code directement copiable
- des fichiers terminés
- une implémentation fidèle à cette architecture

Les futures réponses ne doivent plus remettre en question les fondations de l'architecture sauf découverte d'un bug bloquant démontré.
Règle absolue : toute proposition d'évolution architecturale doit démontrer qu'elle résout un problème concret identifié dans le moteur. Les améliorations purement théoriques ou spéculatives sont proscrites pendant la phase d'implémentation. L'objectif est désormais de transformer cette architecture en un logiciel fonctionnel, maintenable et performant.