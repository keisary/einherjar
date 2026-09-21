# Audit de fin de phase — corpus Einherjar, 1D/5M et noyau SST

**Date du constat :** 11 septembre 2026.  
**Périmètre :** lecture du corpus de référence, des archives, des sorties 1D du 11 septembre, des métadonnées compilées et du chemin d'exécution `xgb_einhers`. Aucune source du moteur n'a été modifiée.

## Résumé décisionnel

**Décision : HYBRIDE — réparer et revalider deux contrats de preuve, puis figer le noyau de recherche.**

Le corpus est assez large pour justifier le passage prochain à l'intégration produit, mais il n'est pas encore assez fiable pour être figé comme portefeuille de référence : les métriques contiennent des valeurs manifestement invraisemblables et le chemin récent 1D/5M casse l'alignement entre colonnes entraînées et colonnes backtestées. Une nouvelle exploration exhaustive n'est pas justifiée. Les seules expériences qui méritent d'être financées sont des expériences courtes servant à réparer/mesurer ces contrats.

Deux ensembles ne doivent pas être confondus :

- Les références demandées, `outputs/corpus.jsonl` et `outputs/archive.jsonl`, ont été modifiées pour la dernière fois le 30 août. Elles ne contiennent aucun candidat 5M.
- La campagne 1D isolée du 11 septembre écrit `corpus_1d_full.jsonl` et `archive_1d_full.jsonl`. Son état annonce 115 triplets OK, un échec de verrouillage, six admissions comptées et 8 418 rejets. Les fichiers contiennent 11 admissions et 8 625 rejets, donc ils ne sont pas une photographie proprement liée au seul rapport agrégé.

Le diagnostic « le 1D n'a pas de signal » est donc **insuffisamment démontré**. Le bon diagnostic est : *la campagne 1D n'est pas encore une mesure valide du signal, car son backtest n'évalue pas systématiquement les mêmes colonnes que le modèle*.

# 1. État du corpus

## Référence du 30 août

| Mesure | Constat |
| --- | ---: |
| Lignes JSON valides | 1 092 |
| Ligne malformée | 1 (`r{...}` en ligne 983) |
| Stratégies exactes (AST + seuils) | 1 031 |
| Doublons exacts | 61 (5,6 %) |
| Structures distinctes (features/opérateurs, sans seuils) | 850 |
| Répétitions structurelles | 242 (22,2 %) |
| Actifs/univers | 29, dont l'univers `multi` |
| Directions | 641 BUY / 451 SELL |
| Timeframes | 15m: 532 ; 1h: 480 ; 4h: 64 ; 1d: 16 ; **5m: 0** |
| Conditions par stratégie | médiane 4, moyenne 3,78, maximum 7 |

La couverture 15m/1h est substantielle, mais le corpus n'est pas équitablement réparti : 92,7 % des entrées sont concentrées sur ces deux TF. Les 16 entrées 1D ne constituent pas une preuve suffisante de l'absence ou de la présence d'un edge 1D.

Les 109 features vues dans le corpus ne représentent qu'environ 45 % des 241 colonnes utilisables du schéma actuel (246 colonnes compilées moins les cinq OHLCV retirées par le chargeur). Les plus fréquentes sont `chaikin_oscillator` (497), `Factor_Momentum_Score` (428), `skewness_risk` (378), `quant_rolling_skewness` (213), `Factor_Risk_TailEvent_Score` (190) et `bb_percent` (154). La diversité existe, mais est fortement concentrée : les composites Factor et quelques variables de risque/volume dominent.

Les métriques sont un avertissement plutôt qu'une validation : médiane Sharpe 5,88, mais moyenne 23,25, p95 176,78 et maximum 510,26 ; médiane rendement total 1,91, mais maximum 135 572,6. Quatre-vingt-quatorze entrées ont Sharpe > 15 et 33 ont un rendement total > 100. Ces extrêmes rendent les moyennes inutilisables comme preuve de qualité économique sans réévaluation traçable.

## Corpus 1D isolé du 11 septembre

Il contient 11 entrées BUY uniquement, huit stratégies exactes, médiane de trois conditions et trois doublons exacts. La médiane n'est que de 14 trades et le Sharpe médian est 2,22. Huit des onze entrées viennent des sources `cross` ou `family`. Or ces sources sont atteintes par le défaut d'alignement décrit en section 6. Leur admission ne prouve donc rien sur le 1D.

# 2. Analyse des archives

L'archive de référence contient 26 462 rejets, sans ligne malformée. Elle est plus redondante que le corpus : 4 366 doublons exacts (16,5 %) et 10 259 répétitions de structure (38,8 %). 1 748 candidats ont exactement une structure déjà présente dans le corpus et 3 022 (11,4 %) ont une similarité de features d'au moins 0,8 avec une entrée du corpus. Elle documente donc beaucoup de répétitions non éliminées assez tôt.

| Catégorie de rejet normalisée | Nombre | Part |
| --- | ---: | ---: |
| BH / non-significatif après correction multi-tests | 21 907 | 82,8 % |
| Holdout | 3 002 | 11,3 % |
| Win rate de validation | 819 | 3,1 % |
| Veto remplaçant une variante | 493 | 1,9 % |
| Sharpe de validation | 176 | 0,7 % |
| Trades de validation | 64 | 0,2 % |

Le BH fait ce qu'il doit faire : il écarte la grande majorité de candidats. Cela ne justifie pas d'abaisser encore les seuils 1D de façon générale. Les rejetés à Sharpe apparent très élevé ne sont pas des « faux négatifs » démontrés : l'archive contient aussi des outliers extrêmes (Sharpe jusqu'à 437,8, rendement total jusqu'à 276 750), précisément le genre de résultat que le holdout/BH doit empêcher de devenir une décision produit.

La campagne 1D isolée a 8 625 rejets, dont 8 363 (97,0 %) par BH. Son archive contient 1 333 doublons exacts et 2 548 répétitions structurelles. Avant toute interprétation statistique de ce taux, le défaut de correspondance des colonnes doit être supprimé et la campagne relancée dans un nouveau couple corpus/archive immuable.

# 3. Diversité et qualité des Einhers

## Diversité structurelle

**Conclusion solide :** le corpus n'est pas une collection de clones exacts, mais il n'est pas assez diversifié pour être considéré comme un portefeuille indépendant. La diversité exacte est 1 031/1 092, alors que seulement 850 formes de stratégies subsistent une fois les seuils retirés. La médiane de quatre conditions est raisonnable ; le maximum sept indique toutefois que la limite de complexité n'est pas uniforme dans les artefacts historiques.

Les opérateurs sont essentiellement des comparaisons seuilées (`>=` 1 345, `>` 1 073, `<` 804, `<=` 775). Cette diversité est principalement paramétrique, non comportementale. Le corpus ne persiste ni masque de signaux, ni fréquence comparée sur période commune, ni corrélation des PnL, ni exposition régimes/cross-asset : la diversité comportementale demandée est donc **non vérifiable** à partir des JSONL actuels.

## Diversité temporelle et cross-asset

29 univers sont représentés et les deux directions existent. C'est une couverture de recherche, pas une démonstration de robustesse cross-asset : les candidats `multi` sont 180 et les métriques ne contiennent pas de résultat par actif, de dispersion par actif ou de corrélation inter-Einher. Il n'est pas possible d'affirmer qu'un candidat fonctionne transversalement au lieu d'être dominé par un actif.

## Qualité

**Conclusion solide :** les statistiques agrégées sont trop élevées et trop asymétriques pour servir de base à un portefeuille final.  
**Conclusion plausible :** une part importante des extrêmes vient de la faible fréquence de trades, de la capitalisation/annualisation, ou du backtest ; il faut le mesurer par réévaluation, pas le supposer.  
**Hypothèse à tester :** après déduplication comportementale et réévaluation holdout, le corpus utile sera significativement inférieur à 1 092 mais restera suffisant pour l'intégration.

# 4. Analyse du 1D

## Réalité des données et du mapping

Le schéma compilé 1D courant a quatre horizons distincts : 5d, 10d, 20d et 60d. Exemple vérifié : `stocks_tech/1d/metadata.json` contient 246 features et des séquences de 2 029 à 3 815 lignes selon l'actif. Le code monofichier lit l'index de l'horizon depuis ces métadonnées puis prend la colonne correspondante de `Y_ret`; c'est correct.

L'affirmation du document de recherche « environ 2 500 lignes pour 213 features » est directionnellement pertinente (petit échantillon, forte dimension), mais ses chiffres sont périmés pour le schéma présent : 246 colonnes brutes / 241 utilisables, et la longueur varie fortement selon l'actif.

## Ce qui est cohérent dans la stratégie proposée

- Réduire les features seulement sur train, régulariser davantage et garder un holdout temporel : **conclusion solide** comme principe.
- Les paramètres 1D dédiés sont bien présents : depth <= 3, `learning_rate=0.02`, `min_child_weight=30`, `colsample_bytree=0.4`, `subsample=0.7`, L1/L2 renforcés.
- La réduction IC Spearman sur train, déduplication de corrélation et cap 40 est **implémentée**. Elle n'est toutefois pas encore prouvée par une campagne valide.

## Ce qui ne démontre rien pour l'instant

Le rapport préconise le walk-forward comme remplacement du 60/20/20. Le module existe, mais il ne réentraîne pas le modèle à chaque fold : il backteste un Einher déjà construit. De plus, `walk_forward_folds` vaut 1 par défaut et n'est pas propagé par la commande `discover`. La campagne 1D complète ne peut donc pas avoir exécuté ce filtre depuis cette voie. C'est **partiellement implémenté**, non pas une validation walk-forward de l'entraînement.

Le pooling cross-sectional est une piste **plausible**, pas une solution démontrée : le moteur crée des splits par actif puis concatène, ce qui est mieux qu'un split naïf ; mais il impose dans ce chemin un embargo fixe de 48 barres, indépendant de l'horizon réel. Pour 1D/60d, le purge est insuffisant (48 < 60) ; pour 5M/15m il est excessif (48 > 3). La fonction doit recevoir `horizon_bars` réel.

## Réponse 1D

**Le système actuel n'est pas démontré incapable d'exploiter le 1D.** Il combine bien un problème statistique réel (peu d'observations, régime changeant, faible nombre de trades) et un problème d'implémentation qui invalide la mesure récente. La priorité n'est donc ni d'assouplir le FDR à l'aveugle ni de lancer tous les actifs ; c'est de rétablir l'alignement entraînement/backtest, puis de faire une campagne courte de comparaison.

# 5. Analyse du 5M

## Réalité de l'évidence

Les métadonnées 5M existent : par exemple crypto/BTCUSD a 834 767 lignes et les horizons 15m, 30m, 1h, 2h. Mais les deux fichiers de référence ne contiennent **aucun** 5M, et le `discover_state` du 11 septembre décrit une campagne 1D. Il n'existe donc pas, dans les artefacts demandés, de résultat 5M sur lequel estimer un gain ou un taux d'admission.

## Évaluation de la stratégie proposée

Le filtre volume, le sélecteur de fins de dollar bars, la configuration 5M (`subsample=.7`, `colsample=.5`, `min_child_weight=20`) et le choix GPU conditionnel sont présents dans le code. Le GPU n'est actif que si CUDA est disponible, `workers=1` et train >= 100 000 lignes ; c'est une bonne protection contre la contention, mais pas une preuve de performance dans l'environnement actuel.

Le document présente les dollar bars comme une réduction de données. L'implémentation actuelle ne reconstruit pas des dollar bars, leurs features ou leurs horizons : elle sous-échantillonne les lignes 5M existantes. Puis le backtester entre à la ligne sélectionnée suivante et conserve une durée `amplitude_bars` comptée en **lignes échantillonnées**, alors que l'horizon a été calculé en barres 5M brutes. Un trade 15m peut donc durer bien plus que 15m et traverser des trous de marché ignorés. L'embargo est également compté en indices échantillonnés. Ce n'est pas une approximation acceptable pour décider de la qualité 5M.

Les méthodes QuantileDMatrix/external memory, les features microstructure dédiées et la multi-résolution ne sont pas implémentées dans ce module. Elles sont donc des pistes, non des capacités présentes.

## Réponse 5M

**Le 5M ne mérite pas une exploration dédiée exhaustive maintenant.** Son coût serait élevé (données massives, contraintes de backtest/costs, GPU séquentiel), et le mécanisme de sampling actuel rend les métriques non interprétables. Le gain potentiel est non quantifié, car il n'y a aucun corpus 5M validé. Il mérite un seul prototype contrôlé après correction du contrat temporel, pas une campagne multi-actifs.

# 6. Audit de l'implémentation

| Recommandation du document | État | Vérification comportementale |
| --- | --- | --- |
| Modèle global par famille | Implémenté, mais métriques invalides | Les modèles entraînent des sous-matrices, puis le backtest reçoit la matrice complète avec les noms du sous-ensemble. Les index ne correspondent plus. |
| Modèle sans `Factor_*` | Implémenté, mais métriques invalides | Même désalignement de colonnes pour `cross_idx`. |
| Réduction 1D à max 40 IC | Implémentée, mais métriques invalides | `feature_names` devient la sélection IC ; `X_aligned` utilisé par backtest reste complet. Les signaux ne sont pas évalués sur les features entraînées. |
| `colsample_bytree=.5` | Partiel | .6 global, .4 1D et .5 5M : l'intention de sous-échantillonnage est présente, la valeur proposée n'est pas générale. |
| Fusion de jumeaux | Partielle | Le clustering Jaccard + proximité de seuils existe et génère une version généralisée, mais le corpus historique conserve encore 22,2 % de répétitions structurelles. Il ne remplace pas une déduplication finale du corpus. |
| MMR de sélection | Absent | Aucun classement qualité moins similarité trouvé. |
| MAP-Elites | Absent | Aucun archiveur par niches comportementales trouvé. |
| Walk-forward 1D | Partiel/inactif via discovery | Module présent, post-hoc et sans réentraînement ; default désactivé et flag non transmis par `discover`. |
| Paramètres 1D | Implémenté | Configuration dédiée réellement construite avant entraînement. |
| Pooling cross-sectional | Partiel | Split par actif avant concat, mais embargo fixe 48 barres, non relié à l'horizon. |
| Volume/dollar sampling 5M | Implémenté de façon incorrecte pour l'évaluation | Sous-échantillonnage des lignes, sans mapping backtest vers barres brutes ni recomputation des horizons. |
| GPU 5M | Implémenté conditionnellement | CUDA seulement séquentiel et gros train ; non vérifié en exécution ici. |
| Quantile/external memory | Absent | Aucune utilisation trouvée. |
| Features 5M microstructure/multi-résolution | Absentes | Les features restent le schéma général 246 colonnes. |

Deux points déjà résolus méritent d'être conservés : les horizons sont lus des métadonnées par TF dans la découverte, et le chemin single-asset passe le vrai `horizon_bars` au split temporel. Les problèmes ne sont donc pas « mapping absent », mais des ruptures spécifiques dans le chemin multi/sous-échantillonné et dans le backtest de colonnes réduites.

# 7. État de maturité du noyau SST

| Dimension | État | Ce qui manque pour maturité suffisante |
| --- | --- | --- |
| Intégrité | Insuffisante pour figer | Corriger la ligne JSON malformée par régénération contrôlée, et séparer les générations par run immutable. |
| Stabilité | Moyenne | Une campagne récente a échoué par verrou de fichier ; les sorties et `runner.py` sont actuellement modifiés. |
| Données | Bonne couverture, preuves incomplètes | Schéma/horizons réellement disponibles ; manque de contrôle de cohérence des artefacts de run. |
| Génération | Avancée | Global, cross, familles, subgroup, OR, veto existent. Il faut valider que chaque chemin évalue les mêmes colonnes qu'il entraîne. |
| Diversité | Moyenne | Déduplication finale comportementale et par structure ; persistance de masques/signaux/corrélations. |
| Qualité des Einhers | Non certifiée | Recalculer les métriques avec un backtest valide, garder holdout séparé. |
| Robustesse | Partielle | Purge correcte en single, incorrecte en multi ; walk-forward non opérationnel dans la voie de campagne. |
| Performance | Partielle | Limitation workers/GPU et sous-échantillonnage présents ; coût réel 5M non mesuré. |
| Reproductibilité | Faible à moyenne | Versionner config, seed, commit, données, candidats et résultat pour chaque run ; ne pas mélanger JSONL historiques. |
| Maintenabilité | Moyenne | Plusieurs sources de candidats et branches spécialisées ont besoin de contrats de données explicites et de tests d'intégration. |

Le noyau est **avancé fonctionnellement**, mais pas encore mature au sens où ses sorties seraient suffisamment fiables pour être gelées et promues dans un produit.

# 8. Option A — continuer la recherche

Avantage : le 1D peut encore produire un signal après correction, et le 5M pourrait être intéressant pour des stratégies courtes. Coût : il faut d'abord réparer le contrat de colonnes, le contrat de temps 5M, l'embargo multi et le suivi de run. Sans cela, une recherche plus grande augmente seulement le volume d'artefacts non comparables.

Un programme complet 1D + 5M maintenant aurait un faible rapport information/coût. Le 5M ajouterait de grosses charges de données, une exécution GPU à un seul worker et une révision nécessaire du backtester. Aucun chiffre actuel ne permet de promettre un rendement de cet investissement.

# 9. Option B — figer le corpus et terminer le projet

Avantage : le corpus historique apporte déjà suffisamment de matière 15m/1h pour construire les interfaces, l'export, les logs, le monitoring et le pipeline d'exécution en **mode non productif**. Les composants produit ne dépendent pas de 5M ni d'un corpus 1D final.

Risque : figer immédiatement ce corpus comme corpus de production cristalliserait les outliers, les doublons structurels et les candidats des branches aux colonnes mal alignées. Le nettoyage ne peut pas être une simple suppression de lignes : il doit être fondé sur une réévaluation cohérente.

# 10. Comparaison des deux options

| Critère | Continuer exhaustivement | Figer immédiatement | Hybride ciblé |
| --- | --- | --- | --- |
| Information nouvelle fiable | Faible tant que les contrats sont cassés | Nulle | Élevée : répond aux deux incertitudes réelles |
| Coût calcul | Élevé, surtout 5M | Faible | Borné et faible à moyen |
| Risque de faux progrès | Très élevé | Élevé pour le corpus final | Réduit par critères d'arrêt |
| Avancement produit | Retardé | Rapide mais fondation risquée | Commence après une courte validation |
| Recommandation | Non | Non | **Oui** |

# 11. Recommandation finale

**HYBRIDE.** Ne pas poursuivre l'exploration de masse, ne pas non plus figer le corpus actuel comme final.

1. Corriger les deux contrats de données et écrire des tests d'intégration qui échouent si une matrice et `feature_names` ne décrivent pas les mêmes colonnes, ou si une durée/embargo 5M est mesurée après sous-échantillonnage.
2. Produire un corpus de validation nouveau, versionné et séparé, sur un petit périmètre 1D. Sa seule mission est de comparer le pipeline global sans réduction à la version réduction IC, avec mêmes actifs, horizons, coûts et holdout.
3. Lancer un prototype 5M sur un actif représentatif seulement après avoir backtesté sur les barres brutes : les indices échantillonnés doivent servir de *points de signal*, pas d'horloge de trading. Ne poursuivre que si le prototype a une performance holdout stable et un coût mesuré acceptable.
4. Si les critères échouent ou restent ambigus, geler un corpus 15m/1h réévalué/dédoublonné et basculer vers Execution, Dashboard, Logs, Monitoring et Export.

# 12. Plan d'action

## Expérience 1 — validité 1D (obligatoire, courte)

1. Test d'intégration : pour chaque mode global/cross/family/IC, vérifier explicitement que les colonnes de train, val, holdout et backtest portent les mêmes noms dans le même ordre.
2. Propager `horizon_bars` réel dans `load_multi_asset_split`; propager le paramètre de walk-forward dans discovery, ou retirer le terme « walk-forward » de la promesse jusqu'à une vraie validation réentraînée.
3. Sur 6 actifs représentatifs et horizons 5d/20d, comparer deux branches : global régularisé et IC-40. Garder holdout intact ; persister candidat, seed, config, version de données, métriques train/val/holdout et signaux.
4. Décision : continuer 1D seulement si IC-40 améliore la médiane holdout et le nombre de candidats stables, sans explosion d'outliers et avec au moins trois actifs ayant un résultat non négatif après coûts. Sinon, classer 1D « non prioritaire ».

## Expérience 2 — faisabilité 5M (un seul actif)

1. Choisir BTCUSD 5M, horizons 15m/30m, et mesurer le coût mémoire/temps avec `workers=1`.
2. Définir les indices volume/dollar comme dates de décision, mais faire entrée, SL/TP, timeout et embargo dans l'OHLCV 5M brut. Ne jamais interpréter `amplitude_bars` comme un nombre de lignes filtrées.
3. Comparer aucune sélection, volume, dollar : mêmes dates de train/val/holdout et mêmes coûts.
4. Arrêt : abandonner immédiatement 5M si le backtest brut ne montre pas de stabilité holdout ou si le coût extrapolé est disproportionné face au gain 15m/1h. Aucun passage multi-actifs avant ce critère.

## Après ces deux expériences

- Réévaluer le corpus 15m/1h avec les mêmes contrats, retirer JSON malformé et doublons exacts, puis sélectionner par structure, actif/TF/direction et comportement mesuré.
- Geler un manifest de corpus (hashes, données, commit, config, candidats et décisions), plutôt que modifier `corpus.jsonl` et `archive.jsonl` en place.
- Commencer ensuite les composants produit, avec un statut explicite « recherche validée sur périmètre X » et sans présenter le 1D/5M comme validés s'ils ne franchissent pas les seuils.

## Double check contradictoire

Cette recommandation ne suppose ni que le 1D cache forcément un edge, ni que le 5M est inutile. Elle résiste aux deux objections :

- Si le 1D est réellement trop difficile, l'expérience courte échouera proprement et évitera de gaspiller une campagne complète.
- Si le 5M est sous-estimé, le prototype brut fournira enfin une mesure de signal/coût comparable, au lieu d'un résultat fondé sur une horloge de barres filtrées.
- Si le corpus paraît excellent grâce à ses métriques, les 94 Sharpe > 15, les rendements extrêmes et les doublons montrent qu'il ne faut pas l'utiliser comme seule preuve.
- Si la réparation ne fournit aucun gain, la décision devient nettement « terminer le noyau » : 15m/1h restent assez couverts pour achever le produit.

# Tableau final des problèmes

| ID | Domaine | Problème | Cause | Impact | État actuel | Gravité | Action recommandée |
| -- | ------- | -------- | ----- | ------ | ----------- | ------- | ------------------ |
| P1 | Backtest | Colonnes entraînées ≠ colonnes backtestées | Sous-ensembles cross/family/IC sans slicing de `X_aligned`/multi backtest | Métriques et admissions invalides | Présent | Critique | Un contrat unique `(X, feature_names)` et un test intégration par branche |
| P2 | 5M | Durée/entrée/embargo mesurés sur lignes filtrées | Sampling transforme l'index, pas le backtester | Résultats 5M non interprétables | Présent | Critique | Mapper les signaux vers OHLCV brut ou reconstruire des barres/labels cohérents |
| P3 | Multi-actif | Embargo fixe 48 | `load_multi_asset_split` ne reçoit pas horizon/timeframe | Leakage 1D long, gaspillage 5M court | Présent | Élevée | Passer `horizon_bars` réel |
| P4 | 1D | Walk-forward promis mais inactif dans discovery | Default 1, flag non propagé ; pas de réentraînement par fold | Robustesse 1D non démontrée | Partiel | Élevée | Rendre le mode explicite et mesurer correctement |
| P5 | Corpus | Ligne JSON corrompue | Écriture/concaténation historique | Lecture non déterministe selon outil | Présent | Moyenne | Régénérer un corpus versionné ; ne pas éditer à la main |
| P6 | Corpus | Outliers et doublons structurels | Réévaluation/final dedup insuffisants | Portefeuille final biaisé | Présent | Élevée | Réévaluer et dédupliquer par comportement/structure |
| P7 | Archive | 82,8 % rejet BH, forte redondance | Génération répétitive et tests multiples | Coût de calcul sans information proportionnelle | Présent | Moyenne | Dédupliquer avant BH et persister les clusters |
| P8 | 5M | Aucun résultat référence | Absence de campagne 5M dans les JSONL de référence | Gain potentiel inconnu | Présent | Moyenne | Prototype unique avec critères d'arrêt |
| P9 | Reproductibilité | Artefacts de générations mélangées | JSONL append + sortie de campagne distincte | Comparaison impossible | Présent | Élevée | Manifest immuable par run |
| P10 | Validation | Suite de tests pertinente non exécutable ici | `pytest` absent du runtime disponible | Contrats nouveaux non vérifiés dynamiquement | Bloqué environnement | Moyenne | Utiliser l'environnement projet avec pytest avant validation finale |

# Tableau des décisions

| Sujet | État actuel | Décision | Justification | Expérience nécessaire |
| --------------- | ----------- | -------- | ------------- | --------------------- |
| Corpus actuel | Large mais métriques/outliers/doublons non finalisés | Ne pas figer immédiatement | 1 092 lignes ne valent pas 1 092 edges indépendants | Réévaluation + déduplication après P1 |
| 1D | Mesure récente invalide | Corriger puis tester court | Le faible taux d'admission mélange difficulté statistique et défaut de colonne | 6 actifs, 5d/20d, IC-40 vs global |
| 5M | Données présentes, zéro preuve corpus | Reporter l'exploration massive | Sampling/backtest non cohérents et coût non mesuré | BTCUSD brut, 15m/30m, trois samplings |
| Diversité | Modérée, concentration forte | Mesurer et filtrer | 22,2 % de répétitions structurelles | Masques/corrélations + quotas par niche |
| Qualité | Non certifiée | Réévaluer | Outliers incompatibles avec une confiance directe | Holdout séparé versionné |
| Noyau SST | Avancé mais non mature | Réparer contrats ciblés | Génération riche, preuves insuffisantes | Tests intégration P1/P2/P3 |
| Suite du projet | Peut démarrer après court verrouillage | Passer ensuite au produit | 15m/1h donnent déjà une base utile | Terminer les deux expériences avec critères d'arrêt |
