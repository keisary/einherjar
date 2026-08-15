# Audit systemique — TypeGP -> hypothese -> evaluation -> validation -> admission

Date : 2026-08-13. Perimetre : pipeline actif `src/einherjar/research/`; le dossier
`research/_old/` est exclu. Audit de code et des tests locaux, sans modification du moteur.

## Verdict

**Partiellement fonctionnel, mais fortement compromis comme chaine de recherche
statistique reproductible.** Le moteur construit et evalue des arbres booleens,
simule les trades sans look-ahead evident et dispose de vraies portes d'admission.
Cependant, TypeGP elimine tres tot des regions de recherche, ne respecte pas sa
contrainte globale de profondeur apres les operateurs evolutifs, et le pipeline
selectionne puis re-genere une population differente. Les preuves de validation
statistique ne couvrent donc pas de facon fiable l'exploration ayant produit le
candidat.

## Flux reel observe

```text
taxonomie de features
  -> TypedGPGenerator._atom() (feuille Condition booleenne)
  -> _grow/_full() (ConditionNode AND/OR/XOR/NOT)
  -> Hypothesis (direction, amplitude ATR, universe, cooldown)
  -> train_calibrate sur les 80 % internes du train
  -> test_on sur les 20 % internes du train (fitness croissance)
  -> tournoi + crossover/mutation + elitisme
  -> population finale dedupliquee (seuls survivants)
  -> comparator: train_calibrate + test_on sur val externe pour classer un generateur
  -> select: choix du nom du generateur
  -> admit: regeneration, calibration train + evaluation val externe des seuls finalistes
  -> DSR/PBO/IC/trades/croissance/cross-asset/DD + diversite/dedup/quota
  -> CorpusEntry JSONL si admis; Archive JSONL si rejete
  -> holdout unique, posteriorieur a l'admission
```

L'objet conserve a l'admission est l'hypothese serialisee, les parametres calibres,
les metriques val et les fingerprints. La trajectoire evolutive, les parents,
la fitness interne, le budget reel consomme et les candidats elimines ne sont pas
conserves.

## TypeGP : constat

L'initialisation grow/full, le tournoi, le crossover de sous-arbres et la mutation
sont effectivement implementes. Le hasard est local a `random.Random(seed)` et
la fitness interne utilise le meme echantillon seed pour tous les individus.

Ce n'est toutefois pas un STGP general : l'arbre entier est de type booleen; le
« typage » du crossover est seulement la categorie `atomic`/`compound`. Les
features float et booleennes sont traitees differemment dans les feuilles, mais
il n'existe ni type de retour exprime, ni fonctions numeriques composees, ni
verification de type sur les sous-arbres au sens GP type. C'est une variante
booleenne documentee de facon plus ambitieuse que son implementation.

## Problemes confirmes (double passe)

| ID | Gravite | Phase | Fichier/fonction | Cause racine | Consequence | Correction |
|---|---|---|---|---|---|---|
| TGP-01 | CRITIQUE | Selection/admission | `discovery.py:handle_admit` | selection d'un generateur, puis regeneration avec un protocole different (`n_eval`, `n_samples=400`) | les candidats admis ne sont pas ceux qui ont gagne la comparaison; la validation du comparateur ne justifie pas l'admission | persister les candidats classes et leurs preuves; admettre ces memes IDs, sans regeneration |
| TGP-02 | MAJEUR | Evolution | `generators/algorithms.py:TypedGPGenerator._subtree_crossover/_subtree_mutation` | le sous-arbre est greffe sans budget de profondeur relatif a son point d'insertion | profondeur/taille reelle peut depasser `max_conditions`; bloat et exploration non bornee | mesurer profondeur/taille apres chaque operateur, rejeter ou re-generer seulement un sous-arbre restant dans le budget |
| TGP-03 | MAJEUR | Discovery | `TypedGPGenerator._evaluate_population`, `_growth_fitness` | porte dure `n_signals < min_total -> -inf`, suivie d'un elitisme top-N | les hypotheses peu frequentes disparaissent avant evaluation/validation complete; conflit avec la regle Discovery ouverte | utiliser une fitness douce/novelty pour la survie; laisser les seuils d'admission exclusivement a Admission |
| TGP-04 | MAJEUR | Evidence/validation | `TypedGPGenerator.generate`, `GeneratorResult`, `handle_admit` | ne retourne que la population finale dedupliquee et annonce `n_evaluated=len(unique)` | les evaluations internes et hypotheses eliminees deviennent invisibles; budget, biais et correction de multiple testing sont invérifiables | journal append-only de chaque candidat/evaluation/parent/operateur; compter les vrais appels moteur |
| TGP-05 | MAJEUR | Reproductibilite | `utils/stats.py:block_bootstrap_ci` (test `test_engine_no_leak.py`) | le test exige que deux seeds donnent des IC differents, mais les sorties sont identiques pour l'echantillon de test | le contrat de test de variabilite normale echoue; impossible d'etablir le comportement attendu des seeds | definir le contrat (determinisme meme seed; resultat seed-dependent non garanti) et corriger le test ou l'implementation selon le choix |
| TGP-06 | MODERE | Admission/archivage | `admission/decision.py`, `archive/store.py` | chaque rejet est archive puis tout fingerprint d'archive bloque les essais suivants au meme `data_version` | un echec DSR/PBO ponctuel devient une interdiction permanente, y compris apres correction de protocole/configuration non encodee dans `data_version` | distinguer doublon structurel de rejet historique; versionner aussi config/splits/couts, ou ne bloquer que les doublons admis |
| TGP-07 | MODERE | Comparaison | `generators/comparator.py` | le score min-max est relatif au groupe de generateurs execute et les genererateurs par defaut excluent random/GE | le gagnant peut changer avec la composition de la competition, sans changement de donnees; ce n'est pas une preuve absolue de superiorite | figer la liste de comparateurs dans l'artefact et comparer aussi a des baselines fixes, sans confondre classement relatif et validation |
| TGP-08 | MODERE | Semantique TypeGP | `TypedGPGenerator` | direction LONG/SHORT est tiree independamment des patterns et le type est limite a float/boolean | hypotheses semantiquement contradictoires et espace de recherche biaise/non explique | encoder les orientations connues comme contraintes ou etiquettes; conserver une voie explicitement contrarienne, evaluee separement |
| TGP-09 | MINEUR | Auditabilite admission | `handle_admit` | les exceptions d'evaluation sont seulement loggees, mais absentes du resume (`n_generated`, `n_admitted`, `n_rejected`) | les candidats ni admis ni rejetes peuvent disparaitre des comptes | ajouter `n_attempted`, `n_eval_failed`, IDs et causes dans le resume et l'archive technique |

## Evaluation et validation

Points positifs observes : signal sur t, entree OPEN t+1, ATR local a l'entree,
SL/TP calibres uniquement depuis train, et garde d'acces holdout unique. Cela
constitue une separation utile train/val/holdout.

Limite majeure de preuve : la fitness TypeGP est faite sur un split interne du
train; le comparateur utilise ensuite le val externe pour choisir un generateur;
puis `admit` reexecute ce generateur et teste de nouveaux finalistes sur ce meme
val. Cette reutilisation de la validation pour choisir le moteur et valider les
candidats doit etre tracee et integree a la correction de selection. Le PBO
recoit seulement les trajectoires des finalistes de l'admission, pas l'ensemble
des hypotheses ayant influence le choix du generateur.

## Trace d'une hypothese reelle (structurelle)

Pour un individu `TypedGPGenerator_000000` : une feature float ou pattern devient
une `Condition`; grow/full l'insere dans un arbre logique; une direction aleatoire
et une amplitude 5 ATR sont attachees a `Hypothesis`. Le moteur calibre N/SL/TP
sur le sous-train interne, evalue le sous-val interne, puis l'individu peut etre
perdu au tournoi, a l'elitisme ou a la deduplication. S'il survit, il est evalue
par le comparateur sur val externe. Mais son ID concret n'est pas transporte au
mode `admit`: un nouveau run genere une nouvelle population avant l'evaluation
finale. Aucune trace sur donnees reelles n'a ete produite, car l'environnement
Python disponible n'a ni `polars` ni `PyYAML`.

## Chaine causale principale

```text
Choix relatif d'un generateur sur val externe
  -> regeneration avec autres parametres/budget
  -> candidats differents de ceux compares
  -> PBO et DSR calcules sur un sous-ensemble final seulement
  -> admission apparemment rigoureuse mais preuve non rattachee a la recherche
  -> corpus difficile a reproduire ou a interpreter
```

## Plan de correction, dans cet ordre

1. Corriger TGP-01 et TGP-04 : artefact immutable « candidate ledger » reliant
   candidat, run, seed, parents, parametres, evaluations et statut.
2. Corriger TGP-02 : invariants de profondeur/taille apres crossover et mutation,
   avec tests de propriete seedes.
3. Corriger TGP-03 : separer fitness de survie/exploration et gates d'admission;
   mesurer le taux de survie par region de l'espace.
4. Decider et formaliser la correction de selection statistique (TGP-01/TGP-07),
   puis adapter DSR/PBO a la population reellement exposee au val.
5. Revoir la politique Archive/dedup (TGP-06), sans supprimer l'historique.
6. Ajouter les compteurs et echantillons reproductibles; ensuite seulement
   executer une campagne sur donnees reelles et produire une trace complete.

## Verification effectuee et limites

- `compileall` sur `src/einherjar/research` : succes.
- `unittest` cible `test_engine_no_leak.py` : 13 tests, 1 echec confirme sur
  `test_different_seeds_different_result` (IC bootstrap identiques pour seeds 42 et 43).
- Suite complete non executable dans le runtime disponible : dependances manquantes
  `polars` et `PyYAML`. Ce n'est pas une validation fonctionnelle du pipeline.

