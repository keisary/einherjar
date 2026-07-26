Phase A — Les données

Questions :

les labels sont-ils cohérents ?
les horizons représentent-ils réellement ce qu'on pense ?
Y_ret est-il une bonne cible ?
Y_hor est-il utilisé correctement ?
TP/SL et horizon racontent-ils la même histoire ?

C'est ici que je pense qu'il y a déjà un problème.

Par exemple :

si

Y_ret

est calculé sur

32 bougies

mais que

max_hold = 24

alors le moteur optimise quelque chose qu'il ne sera jamais autorisé à exécuter.

C'est un problème de cohérence.

Phase B — La recherche

Aujourd'hui tu fais :

Feature

↓

Beam Search

↓

Condition

↓

Condition

Je veux répondre à la question :

est-ce réellement la meilleure manière de découvrir des règles ?

Je vais comparer :

Beam Search
Best First Search
Monte Carlo Tree Search
Branch & Bound
Evolutionary Search
Genetic Programming
Novelty Search

Je ne veux pas choisir le plus rapide.

Je veux choisir celui qui découvre le plus d'edges.

Phase C — Les métriques

Aujourd'hui :

expectancy

t-test

BH

Sharpe

WinRate

Mais les sociétés quantitatives utilisent maintenant des métriques beaucoup plus adaptées lorsque des milliers de stratégies sont testées.

Par exemple :

Deflated Sharpe Ratio
Probability of Backtest Overfitting (PBO)
Probabilistic Sharpe Ratio
White Reality Check
Hansen SPA Test

Le point commun de ces méthodes est qu'elles corrigent explicitement le biais créé par le fait d'essayer énormément de stratégies. Elles sont justement conçues pour les moteurs comme EINHERJAR, qui explorent un très grand espace de recherche.

Phase D — La simulation

Je pense que c'est probablement le plus gros chantier.

Aujourd'hui :

validation

↓

simulation

Je voudrais presque inverser les deux.

Autrement dit :

la simulation devrait produire directement les retours qui serviront aux statistiques.

Comme ça :

condition

↓

entrée

↓

TP

↓

SL

↓

sortie

↓

retour

↓

statistiques

Tout serait enfin cohérent.

Aujourd'hui tu mesures :

retour fixe

pour ensuite valider :

sortie TP/SL

Ce sont deux expériences différentes.

Phase E — Le score

Là aussi je pense qu'il faudra repartir de zéro.

Tu m'as donné un objectif très clair.

faire grandir un petit capital le plus vite possible

Ce n'est pas du tout la même chose que

Sharpe maximal

Le Sharpe préfère parfois une stratégie qui gagne peu mais très régulièrement.

Toi tu cherches plutôt :

Croissance composée

Donc je vais regarder :

Kelly
Log Utility
Expected Log Growth
Risk of Ruin
CAGR
Calmar
MAR
Omega

Je pense que le score final devra être construit autour de la croissance géométrique, pas seulement du rendement moyen.

Phase F — La diversité

C'est probablement le point qui m'intéresse le plus.

Aujourd'hui le moteur cherche :

la meilleure feature

Moi je voudrais qu'il cherche :

la meilleure nouvelle information

C'est complètement différent.

Par exemple :

Stoch K

et

Stoch D

peuvent tous les deux être excellents.

Mais s'ils racontent la même chose,

alors garder les deux n'apporte presque rien.

À l'inverse,

une feature moins performante mais totalement différente peut enrichir énormément le corpus.

Je pense qu'il faudra intégrer une notion de diversité directement dans la recherche.


**Résultat de l'audit de la Phase A**

Je vais raisonner comme si je devais publier un article scientifique sur le dataset.

A1 — Les features (X)

Je suis plutôt rassuré.

D'après les fichiers que tu m'as fournis précédemment :

les features sont calculées barre par barre ;
elles utilisent uniquement l'historique disponible ;
elles sont normalisées de manière cohérente ;
elles sont ensuite compilées dans un dataset unifié.

Je n'ai pas vu de fuite temporelle évidente.

✅ Pour moi cette partie est valide.

A2 — Les labels d'action

Là aussi je trouve la logique saine.

Les différents patchs montrent qu'il y a eu énormément d'efforts pour garantir que :

chaque timeframe est isolé ;
les lignes des autres TF deviennent NULL (sentinelle -100) ;
BUY = 1, SELL = -1, HOLD = 0 ;
les horizons ne sont calculés que lorsque l'action est valide.

C'est exactement ce que j'aurais fait.

Je ne vois rien qui me choque.

A3 — Les profits

C'est ici que j'ai commencé à réfléchir beaucoup.

Le script de calcul des horizons montre que :

label_profit

est utilisé comme prix cible.

Ensuite le patch cherche :

combien de bougies faut-il pour atteindre ce prix cible ?

Ce n'est donc pas un simple retour fixe.

Le code fait essentiellement :

entry

↓

target = entry × (1+profit)

↓

scan forward

↓

première bougie qui atteint target

puis

time_horizon = j

Et ça, je trouve ça très intelligent.

Ce qui est intéressant

Cela signifie que :

Y_ret et Y_hor ne sont pas indépendants.

Ils décrivent le même évènement.

Autrement dit :

Y_ret

↓

objectif

↓

Y_hor

↓

temps nécessaire pour atteindre cet objectif

Jusqu'ici tout est cohérent.

A4 — Les horizons

Là je pense avoir trouvé quelque chose de très intéressant.

Le patch calcule :

max_bars = parse_horizon_to_bars(tf,horizon)

Par exemple

15m

↓

8h

↓

32 bougies

Puis

si objectif atteint

↓

temps réel

sinon

↓

32

Autrement dit

Y_hor est un horizon observé.

Pas un horizon arbitraire.

C'est très bien.

MAIS...

Maintenant regardons Discovery.

Je vois :

max_holding_bars

15m = 24

Alors que tes horizons sont :

15m

4

8

16

32

Donc

very_long

↓

32

mais

simulation

↓

24

Là je ne parle plus d'un bug.

Je parle d'une incohérence mathématique.

Le dataset dit :

"on observe pendant 32 bougies"

La simulation dit :

"on arrête à 24"

Ces deux expériences sont différentes.

Je pense que c'est le premier vrai problème de Phase A.

A5 — La cohérence globale

C'est ici que je pense qu'il faut être très exigeant.

Aujourd'hui j'identifie quatre objets différents.

Features

↓

Action

↓

Profit cible

↓

Temps nécessaire

Le Discovery, lui, ajoute :

TP

SL

Cooldown

Max Holding

Frais

Simulation

Or ces derniers n'existent pas dans les labels.

Autrement dit,

le dataset ne décrit pas directement la stratégie qui sera exécutée.

Il décrit une cible.

Puis Discovery construit une autre stratégie.

C'est là que les deux mondes commencent à diverger.

Ce que je considère comme le vrai problème de la Phase A

Je vais formuler ça comme un théorème.

Le dataset apprend :

"Quel est le profit théorique atteignable sur cet horizon ?"

Le moteur essaie ensuite d'exécuter :

"Une stratégie TP/SL dynamique avec sortie anticipée."

Ce ne sont pas les mêmes expériences.

Donc :

les données sont cohérentes entre elles,

mais elles ne sont plus cohérentes avec la stratégie réellement simulée.

Ce qui me rassure énormément

Je craignais trouver :

des erreurs de mapping ;
des fuites temporelles ;
des horizons calculés bizarrement ;
des colonnes mélangées entre TF.

Franchement...

je ne les vois pas.

Au contraire,

les patchs montrent beaucoup de soin.

Ma conclusion sur la Phase A

Je lui donnerais :

Qualité du dataset

9.5 / 10

Très propre.

Cohérence des labels

9 / 10

Bonne.

Cohérence dataset ↔ Discovery

5 / 10

C'est ici que réside, selon moi, le vrai problème.

Le Discovery transforme les labels en une expérience différente de celle qui a servi à les créer.

Ce que je propose avant de passer à la Phase B

Je pense qu'il manque une dernière pièce pour clore définitivement la Phase A.

Je voudrais reconstruire le contrat de données de MIDAS.

Pas le code.

Le contrat.

Quelque chose comme :

Chaque ligne i représente :

Information connue à t

↓

Feature vector X(i)

↓

Action optimale

↓

Profit cible

↓

Temps nécessaire

↓

Toutes ces valeurs sont calculées uniquement à partir des données futures
et ne doivent jamais être réinterprétées différemment par Discovery.

Ce document deviendrait la "constitution" du projet. Ensuite, en Phase B, chaque étape du Discovery devra respecter ce contrat. Si une étape le viole (comme le fait probablement aujourd'hui la simulation TP/SL), on saura immédiatement que ce n'est pas le dataset qu'il faut modifier, mais l'algorithme qui l'utilise.

Je pense que cette séparation entre contrat des données et algorithme de découverte sera très utile pour la suite, car elle évitera de mélanger des problèmes de génération des labels avec des problèmes de conception du moteur de recherche.

**Phase B**

Le problème que l'on cherche réellement à résoudre

Mathématiquement, ton problème est le suivant :

Explorer un espace de plusieurs milliards de règles afin d'identifier quelques dizaines d'hypothèses qui correspondent à un phénomène réel et reproductible du marché.

Ce problème existe déjà dans d'autres domaines.

Par exemple :

découverte de molécules en chimie ;
recherche de gènes en bio-informatique ;
découverte de lois physiques ;
recherche d'architectures de réseaux neuronaux (Neural Architecture Search) ;
optimisation de programmes.

Dans tous ces domaines, le défi n'est pas seulement de trouver "le meilleur score", mais d'explorer efficacement un espace immense sans rester bloqué dans une petite région.

Les technologies dont je m'inspirerais

Je ne copierais aucun algorithme existant. Je construirais un hybride inspiré des meilleures idées.

Domaine	Apport à EINHERJAR
Beam Search	Exploration rapide des combinaisons
Monte Carlo Tree Search (MCTS)	Équilibre exploration / exploitation
Novelty Search	Découverte de comportements différents
mRMR (Minimum Redundancy Maximum Relevance)	Éviter les features redondantes
Sequential Feature Selection	Construction progressive des règles
Purged Walk Forward Validation	Validation robuste
Deflated Sharpe / PBO	Réduction du surapprentissage
Multi-objective Optimization (NSGA-II, Pareto)	Optimiser plusieurs critères simultanément

Aucun de ces algorithmes, pris isolément, ne répond à ton besoin. Ensemble, ils forment une base très solide.

Le principe fondamental du nouvel algorithme

Le moteur ne chercherait plus :

"Quelle est la meilleure règle ?"

Il chercherait :

"Quelle est la prochaine hypothèse de marché la plus prometteuse à explorer ?"

C'est complètement différent.

Architecture proposée

Je découperais le moteur en 8 modules indépendants.

Module 1 — Génération des hypothèses

Entrée :

Toutes les features

Le moteur ne génère pas immédiatement des stratégies.

Il génère des hypothèses élémentaires.

Exemple :

RSI < 20

ADX > 30

ATR percentile > 90%

Volume ZScore > 2

Chaque hypothèse reçoit un identifiant unique.

Aucune élimination ici.

Objectif :

Créer un univers complet.

Module 2 — Regroupement sémantique

C'est une étape qui n'existe pas aujourd'hui.

Le moteur classe automatiquement les features.

Par exemple :

Momentum

RSI

CCI

Williams

Stochastic
Trend

ADX

DI

EMA Distance
Volatility

ATR

Parkinson

Realized Vol

Entropy
Volume

OBV

CMF

Chaikin

MFI

Pourquoi ?

Parce qu'on ne veut plus que quatre oscillateurs monopolisent la recherche.

Chaque famille aura son propre budget d'exploration.

Module 3 — Exploration intelligente

C'est ici que Beam Search disparaît.

À la place :

Chaque famille reçoit un quota.

Par exemple :

Momentum

↓

20 candidats
Volatility

↓

20 candidats
Volume

↓

20 candidats
Trend

↓

20 candidats

Puis les meilleurs de chaque famille sont combinés.

On obtient naturellement une très forte diversité.

Module 4 — Construction progressive

Une règle devient un objet.

Au lieu de :

Feature

↓

Feature + Feature

Le moteur construit progressivement :

Hypothèse

↓

Hypothèse filtrée

↓

Hypothèse spécialisée

↓

Stratégie

À chaque ajout de condition, il répond à une seule question :

Cette condition apporte-t-elle une information nouvelle ?

Si non,

elle est rejetée.

Module 5 — Exploration adaptative

C'est la partie inspirée du Monte Carlo Tree Search.

Chaque branche possède désormais un historique.

Par exemple :

RSI

↓

beaucoup de règles robustes

↓

explorer davantage

Inversement :

Chaikin

↓

100 essais

↓

aucun résultat

↓

explorer moins

Mais jamais zéro.

Une partie du budget reste réservée à l'exploration pure.

C'est ce qui évite de passer à côté d'un edge rare.

Module 6 — Validation progressive

Aujourd'hui :

Recherche

↓

Validation

Je propose :

Recherche

↓

Validation légère

↓

Recherche

↓

Validation moyenne

↓

Recherche

↓

Validation finale

Les tests coûteux ne sont exécutés que sur les candidats prometteurs.

On économise énormément de temps.

Module 7 — Diversité

C'est probablement la plus grosse nouveauté.

Le moteur maintient un corpus.

Avant d'accepter une nouvelle règle,

il vérifie :

Est-elle vraiment différente ?

Différente :

par ses features ;
par les marchés concernés ;
par les timeframes ;
par les périodes gagnantes ;
par ses trades.

Deux règles gagnant exactement sur les mêmes trades n'apportent presque rien au corpus.

Une seule est conservée.

Module 8 — Corpus final

Les Einhers ne sont plus classés uniquement par score.

Chaque Einher possède maintenant un profil.

Exemple :

Type

Momentum Reversal

Confiance

94%

Diversité

élevée

Marchés

BTC

ETH

EURUSD

Timeframes

15m

1h

Robustesse

très élevée

Croissance

élevée

Le corpus devient une bibliothèque de comportements de marché, et non plus une simple liste de règles.

La philosophie de l'algorithme

Aujourd'hui :

Feature

↓

Score

↓

Conserver

Demain :

Hypothèse

↓

Exploration

↓

Validation

↓

Diversité

↓

Robustesse

↓

Croissance

↓

Corpus

Le score n'est plus utilisé pour piloter l'exploration.

Il sert uniquement à classer les candidats une fois qu'ils ont prouvé leur intérêt.

Pourquoi je pense que cette architecture est meilleure

Elle corrige pratiquement tous les problèmes que nous avons identifiés :

plus de domination des Stochastic ou d'une seule famille d'indicateurs ;
possibilité de découvrir des synergies entre features qui seraient éliminées par un Beam Search classique ;
séparation claire entre exploration et évaluation, ce qui réduit le risque d'abandonner trop tôt des pistes prometteuses ;
meilleure couverture de l'espace de recherche grâce à une exploration guidée par la diversité ;
corpus final plus riche, composé de comportements de marché réellement distincts.
Une évolution que je proposerais

Il y a un dernier changement que je considère comme très important.

Aujourd'hui, une règle est essentiellement définie par ses conditions.

Je pense qu'à terme, l'entité fondamentale du moteur devrait être une hypothèse de marché.

Par exemple :

"Le marché est en compression de volatilité avant une expansion."

Cette hypothèse pourrait être représentée par plusieurs combinaisons de features différentes selon l'actif ou le timeframe.

Ainsi, le moteur découvrirait des phénomènes plutôt que des combinaisons arbitraires d'indicateurs. Les conditions deviendraient une représentation d'une hypothèse, et non l'objectif final de la recherche.


**Phase C**

Je pense que l'erreur actuelle est fondamentale

Aujourd'hui, la logique est approximativement :

Rule

↓

Expectancy

↓

t-test

↓

BH

↓

Sharpe

↓

Score

Autrement dit, on cherche à répondre à une seule question :

"Cette règle est-elle rentable ?"

Or ce n'est pas la bonne question.

La vraie question est :

"Quelle est la probabilité que cette règle continue de gagner dans un futur inconnu ?"

C'est une différence énorme.

Ce que doit mesurer la Phase C

Après avoir étudié les méthodes utilisées en recherche quantitative, je pense qu'une bonne stratégie doit être évaluée sur six dimensions indépendantes.

Pas une seule.

Axe 1 — Qualité économique

Première question :

Si je trade réellement cette règle, vais-je gagner de l'argent ?

On retrouve ici :

Expectancy
Profit Factor
CAGR
Return / Drawdown
Growth Rate

Ces métriques répondent uniquement à :

Est-ce rentable ?

Rien d'autre.

Axe 2 — Robustesse statistique

Deuxième question :

Est-ce que ce résultat pourrait être dû au hasard ?

Aujourd'hui tu fais déjà :

t-test
Benjamini-Hochberg

C'est bien.

Mais ce n'est plus suffisant.

Je voudrais remplacer ou compléter avec :

Probabilistic Sharpe Ratio (PSR)
Deflated Sharpe Ratio (DSR)
White Reality Check
Hansen SPA Test

Pourquoi ?

Parce que tu testes des milliers de règles.

Les t-tests classiques deviennent optimistes.

Axe 3 — Robustesse temporelle

Question :

Cette règle fonctionne-t-elle uniquement en 2024 ?

ou

fonctionne-t-elle aussi ailleurs ?

Ici on regarde :

Walk Forward
stabilité inter-blocs
variance des performances

Une stratégie qui gagne énormément sur un bloc et perd partout ailleurs devrait être éliminée.

Axe 4 — Robustesse structurelle

Question beaucoup plus rarement utilisée.

Je pense pourtant qu'elle est essentielle.

On modifie légèrement la règle.

Exemple.

Aujourd'hui :

RSI < 20

On teste :

RSI < 19

RSI <21

RSI <22

Si la stratégie disparaît immédiatement,

c'est probablement du surajustement.

Une vraie anomalie de marché devrait survivre à de petites perturbations.

Je trouve que cette métrique manque aujourd'hui.

Axe 5 — Diversité

Celui-ci n'existe pas actuellement.

Question :

Cette règle apporte-t-elle quelque chose au corpus ?

Imaginons :

Rule A

+

Rule B

Les deux gagnent exactement sur les mêmes trades.

Même Sharpe.

Même Equity.

Même périodes.

En réalité,

la deuxième n'apporte presque rien.

Aujourd'hui tu la conserves quand même.

Moi je voudrais mesurer :

Distance comportementale

Pas seulement

Distance entre features

Deux stratégies utilisant des features différentes peuvent produire exactement les mêmes décisions.

Elles sont donc redondantes.

Axe 6 — Croissance réelle

C'est probablement le plus important compte tenu de ton objectif.

Tu m'as dit :

"Faire grandir rapidement un petit capital."

Alors pourquoi optimiser le Sharpe ?

Le Sharpe récompense souvent des stratégies très prudentes.

Or ton objectif est différent.

Je pense qu'il faut optimiser quelque chose proche de :

Expected Log Growth
Kelly Growth
Risk of Ruin
CAGR

Autrement dit :

Croissance géométrique

pas

rendement moyen
Une nouvelle architecture de scoring

Je pense qu'il faut abandonner l'idée d'un score unique calculé directement.

Je proposerais plutôt une évaluation en cascade.

Niveau 1 — Élimination rapide

Très rapide.

Questions :

Trades suffisants ?
Expectancy positive ?
Profit Factor minimal ?
Drawdown acceptable ?

90 % des règles meurent ici.

Niveau 2 — Validation statistique

Pour les survivantes.

Questions :

PSR
DSR
Walk Forward
BH
Persistance

Encore énormément d'éliminations.

Niveau 3 — Validation structurelle

Questions :

La règle survit-elle si :

on change légèrement un seuil ?
on décale le TP ?
on décale le SL ?

Une règle fragile est rejetée.

Niveau 4 — Validation comportementale

Questions :

Les trades ressemblent-ils trop à une règle déjà présente ?

Si oui,

elle n'est probablement pas utile.

Niveau 5 — Score final

Seulement maintenant.

Le score n'est plus utilisé pour éliminer.

Il sert uniquement à classer les survivantes.

Je trouve cette séparation très importante.

Je pense également qu'il faut abandonner la notion de "score pondéré"

Aujourd'hui tu fais probablement quelque chose du style :

0.30 Sharpe

0.20 Win Rate

0.20 Profit Factor

...

Je ne suis plus convaincu que ce soit une bonne idée.

Pourquoi ?

Parce que changer les poids change complètement le classement.

Je préférerais utiliser un système inspiré de l'optimisation multi-objectifs.

Par exemple :

Une stratégie A :

Sharpe = 1.4

CAGR = 38%

DD = 7%

Une stratégie B :

Sharpe = 1.8

CAGR = 15%

DD = 2%

Les deux sont intéressantes.

Aucune ne domine complètement l'autre.

En optimisation, on appelle cela un front de Pareto.

Le moteur pourrait conserver les stratégies non dominées, puis seulement les ordonner selon ton objectif principal (croissance du capital).

Une idée qui m'est venue

Je pense qu'un Einher ne devrait plus recevoir une simple note.

Il devrait recevoir un profil de confiance.

Par exemple :

Identifiant

E_004531
Confiance statistique

97 %
Croissance

A+
Robustesse

A
Diversité

B+
Risque

B
Généralisation

A

Ainsi, tu ne choisis plus uniquement "le meilleur score". Tu peux aussi sélectionner les Einhers les plus diversifiés, les plus robustes ou les plus agressifs selon le contexte.

Une réflexion plus profonde

En étudiant tout le pipeline, je pense que le projet évolue progressivement.

Au départ, EINHERJAR ressemblait à un générateur de règles.

Avec les modifications que nous avons définies :

Phase A : contrat des données.
Phase B : moteur d'exploration intelligent.
Phase C : validation scientifique multi-dimensionnelle.

Il devient progressivement un Scientific Alpha Discovery Engine.

Et je pense que c'est exactement cette direction qu'il faut suivre.

Le moteur ne cherchera plus simplement des règles qui fonctionnent sur un backtest. Il cherchera des hypothèses de marché qui résistent à plusieurs formes de validation indépendantes : économique, statistique, temporelle, structurelle, comportementale et orientée croissance.

Avant la Phase D, je voudrais ajouter une dernière recommandation

Je pense qu'il faut introduire une distinction qui n'existe pas aujourd'hui :

Les critères de rejet : ils répondent à la question "cette hypothèse est-elle suffisamment solide pour continuer ?". Ils doivent être utilisés très tôt et rester relativement simples.
Les critères de classement : ils répondent à la question "parmi toutes les hypothèses déjà validées, lesquelles sont les plus intéressantes selon notre objectif ?". Ils ne doivent intervenir qu'à la toute fin.

Aujourd'hui, ces deux rôles sont en partie mélangés. Les séparer rendra le pipeline plus lisible, plus facile à faire évoluer, et surtout beaucoup plus cohérent avec l'objectif que tu as fixé : découvrir des stratégies réellement exploitables pour faire croître un capital sur des données futures.


**Phse D**
