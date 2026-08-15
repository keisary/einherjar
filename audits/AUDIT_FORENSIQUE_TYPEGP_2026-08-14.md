# Einherjar — Audit forensique TypeGP

**Date** : 2026-08-14
**Périmètre** : pipeline actif `src/einherjar/research/` (exclu `research/_old/` et `research/v2/`).
**Mode** : analyse et planification exclusivement — AUCUNE modification de code.
**Mission** : reconstruire le chemin d'exécution réel, identifier les bugs,
régressions, code obsolète, et produire un plan de refonte.

---

## 1. Executive Summary

### Constat principal

Le pipeline TypeGP est **structurellement opérationnel en surface** (les
commandes s'exécutent, des artefacts JSON sont produits) mais **profondément
défaillant comme chaîne de recherche** :

- Le pipeline **actuel (post-fix DSR) admet 0 Einhers** (cf.
  `outputs/admit_summary.json` du dernier run : 0/35 admis, 35 rejets DSR_FAIL).
- Les **3 Einhers BTCUSD** du corpus sont des **admissions historiques
  d'avant le fix DSR** (juin 2026), gelées dans `corpus.jsonl`. Aucune
  admission récente.
- Les 3 Einhers historiques sont **structurellement clones** : ils partent
  tous de l'individu `TypedGPGenerator_000045`, ont des arbres quasi-identiques
  (AND-XOR-OR avec mêmes sous-structures), sont **tous Short** (3/3), et
  occupent 1 timeframe en 5m + 2 timeframes en 15m.
- Le moteur a un **bug de bloat non borné** : les arbres finaux ont
  11 niveaux de mutations/crossovers empilés (chaîne `_xNNNN` dans l'ID)
  alors que `max_conditions=4`. Le crossover ne contrôle pas la profondeur.
- L'évolution converge très rapidement (élitisme=2, tournoi=3, crossover
  sous-arbre type-preserving très conservateur + dédup par (tree, direction,
  cooldown_k)). Le seul "diversity mechanism" est la dédup finale.
- Le comparator actuel n'a qu'un seul générateur actif (TypedGP) après
  exclusion de Random/GE : **il n'y a plus de compétition**, le winner est
  élu par défaut avec un score normalisé de 0.5.

### Verdict

**TypeGP ne fait pas de recherche. Il fait du raffinement local autour d'un
individu initial sur-représenté, et l'admission rejette tout sauf
historiquement.** Le système est passé d'un état "3 Einhers jumeaux
suspects" à un état "0 Einhers" — les deux sont des symptômes du même
problème : le moteur ne sait pas explorer l'espace des hypothèses.

### Priorité de refonte

1. **CRITIQUE** : TypeGP ne contrôle pas la profondeur → bloat infini.
2. **CRITIQUE** : pas de mécanisme de diversité (novelty, fitness sharing,
   speciation).
3. **CRITIQUE** : budget par défaut trop petit (n_eval=20 → population=2,
   9 générations).
4. **MAJEUR** : le DSR post-fix rejette tout, ce qui peut être correct ou
   peut être un sur-rejet.
5. **MAJEUR** : l'archive bloque les ré-essais au même fingerprint/data_version
   (memoryless fail), ce qui pénalise la diversité.

---

## 2. État réel du système

### 2.1 Fichiers et taille du moteur

| Module | Fichiers | Lignes | Rôle |
|---|---|---|---|
| `discovery.py` | 1 | 1341 | Point d'entrée pipeline, 8 modes (engine → holdout) |
| `generators/algorithms.py` | 1 | 2470 | 6 générateurs (Random, Beam, STGP, Memetic, NSGA2, GE) |
| `generators/comparator.py` | 1 | 600 | Comparaison multi-générateurs, score composite |
| `engine/evaluator.py` | 1 | 1084 | Moteur d'évaluation, calibration, simulation intrabar |
| `engine/bootstrap.py` | 1 | ~200 | Block bootstrap IID pour CI Sharpe/ret |
| `engine/simulator.py` | 1 | ~120 | Simulation intrabar TP/SL (SL avant TP) |
| `admission/decision.py` | 1 | 415 | Décideur d'admission (7 critères + diversité + dédup) |
| `admission/criteria.py` | 1 | ~570 | 8 critères (DSR, PBO, CI×2, n_trades, croissance, cross-asset, DD) |
| `admission/diversity.py` | 1 | 488 | Descripteurs comportementaux + quotas structurels |
| `admission/baseline_gate.py` | 1 | ~120 | Filtre admission pour baselines |
| `corpus/store.py` | 1 | 229 | Sérialisation append-only JSONL |
| `archive/store.py` + `archive/schema.py` | 2 | ~280 | Rejets persistés, dédup |
| `data/npy_real_loader.py` | 1 | 357 | Loader features MIDAS .npy |
| `data/ohlcv.py` | 1 | 426 | Loader OHLCV depuis CSV bruts |
| `data/features.py` | 1 | ~250 | FeaturesFrame polars |
| `data/threshold_calibration.py` | 1 | ~150 | Quantiles par feature (P1 #1) |
| `data/validation.py` | 1 | ~400 | Validation OHLCV/Features |
| `data/versioning.py` | 1 | ~400 | DataVersion + DataVersionStore |
| `holdout/evaluator.py` + `holdout/ledger.py` | 2 | ~400 | Holdout unique atomique |
| `refinement/beam.py` | 1 | ~400 | BeamRefiner (P1 #2, migré) |
| `selection/selector.py` | 1 | ~250 | Sélection du winner de compare |
| `utils/types.py` | 1 | 470 | Hypothesis, Condition, MesuresBrutes, Einher |
| `utils/fingerprint.py` | 1 | ~200 | fingerprint_structurel + comportemental |
| `utils/stats.py` | 1 | ~330 | Bootstrap, ATR, max_drawdown, periods_per_year |
| `utils/metrics.py` | 1 | ~250 | DSR (Bailey & López de Prado) |
| `config/loader.py` | 1 | ~280 | Chargeur de configuration YAML |
| `tests/` | 18 | ~6000 | 220 tests (tous verts) |

**Total moteur actif** : ~13 000 lignes Python. C'est un moteur de taille
moyenne, mais avec une complexité algorithmique très élevée (DSR, PBO,
CPCV, NSGA-II, etc.).

### 2.2 Données réelles chargées

- **OHLCV** : `D:/midas_v2/technical_agent_dataset_brut/{class}/{asset}/{tf}/<asset>_<year>_<tf>.csv`
  - Vrais prix historiques, lecture par `_CsvRawBackend.fetch()`
  - Schema forcé `volume: Float64`
- **Features** : `D:/midas_v2/midasV3/src/data/compiled/{class}/{tf}/<asset>_X.npy` + `metadata.json`
  - 218 features utilisables (28 exclues : 19 fantômes + 8 meta + 1 alias)
  - **metadata.json est la source de vérité pour l'ordre des colonnes** (bug
    historique où la taxonomie décalait tout d'un cran après `macd_signal`)
- **Jointure** : inner join sur timestamp (les deux sources doivent s'aligner)

### 2.3 Artefacts de la dernière exécution

D'après les fichiers de `outputs/` :

- `selection.json` : `TypedGPGenerator`, `n_eval_budget=1200`, `n_candidates=100000`,
  `n_evaluated=34`, `admission_rate=0.0`, `median_sharpe_all=-142.07` (!),
  `score=0.5` (gagnant par défaut), `data_version=hash:36f070859294`
- `admit_summary.json` : 35 générées, 0 admises, 35 rejets tous `DSR_FAIL`
- `corpus.jsonl` : 3 Einhers historiques BTCUSD, 1 sur 5m + 2 sur 15m, tous Short
- `data_versions.jsonl` : 2 versions (NASDAQ100 15m, hash:4a255ac... et hash:73cf5be...)
- `refined.json`, `compare_report_BTCUSD_*.json` : présents

---

## 3. Pipeline réellement exécutée

### 3.1 Séquence des 7 étapes

`discovery.py:run()` exécute en séquence :
1. `engine` : instancie `EvaluationEngine` (priority 0, sanity check)
2. `baselines` : 3 baselines (Human, Shallow, Random) via `BaselineRunner`
   avec admission réelle (7 critères)
3. `compare` : tous les générateurs (Random, Beam, STGP, Memetic, NSGA2, GE)
   sauf exclusion par défaut de Random + GE → il reste Beam, STGP, Memetic, NSGA2
4. `select` : consomme le rapport de compare (cache si frais) → `selection.json`
5. `refine` : génère N hypothèses via le winner, top-5 par Sharpe val,
   applique BeamRefiner → `refined.json`
6. `admit` : génère N hypothèses via le winner, applique AdmissionDecider
   complet → ajoute au corpus, logue les rejets dans l'archive
7. `holdout` (manuel) : évalue UN Einher admis sur le holdout

### 3.2 Ce qui se passe réellement sur BTCUSD 15m (chemin d'exécution observé)

```
[STEP 0] engine
  └─ EvaluationEngine(config, data_version="v1", seed=42)
     ├─ _ATREstimator(period=14, percentile=50)
     ├─ _min_n, _max_n depuis evaluation.yaml
     └─ _condition_evaluator

[STEP 1] baselines
  ├─ OhlcvProvider().load(BTCUSD, 15m, "raw", "crypto")
  │  └─ _CsvRawBackend().fetch() → CSV bruts
  ├─ load_features_from_npy(BTCUSD, "crypto", "15m")
  │  └─ Charge BTCUSD_X.npy, lit metadata.json
  ├─ Inner join timestamp
  ├─ Split 60/20/20 avec purge=max_n=50, embargo=1
  ├─ _persist_data_version() → DataVersionStore (append-only)
  ├─ make_baseline_admission_fn(config) → admission_fn
  └─ BaselineRunner(engine, eval_budget=...)
     ├─ HumanBaseline : règles humaines (5 patterns)
     ├─ ShallowBaseline : features uniques seuils simples
     └─ RandomBaseline : tirage uniforme

[STEP 2] compare
  ├─ _load_for_handler() (mêmes données que baselines)
  ├─ protocol = make_protocol(..., n_eval_budget=1200, n_candidates=100000,
  │                          max_conditions=4, n_samples=0)
  ├─ generators = make_all_generators(protocol, config, engine)
  │  └─ 6 générateurs, mais _filter_competition_generators() retire
  │     RandomSearchGenerator et GrammaticalEvolutionGenerator par défaut
  │     → RESTE : BeamSearchGenerator, TypedGPGenerator, MemeticGenerator,
  │                NSGA2Generator
  ├─ bind_data(...) sur chaque gen
  ├─ comparator.run(...)
  │  ├─ per_gen_cap = 1200 // 4 = 300 par générateur
  │  ├─ Pour chaque gen : generate() + test_on(val) sur chaque hyp
  │  └─ Score composite : 0.40 sharpe + 0.30 admission + 0.15 diversity
  │                       + 0.15 coherence
  └─ report.to_dict() persisté dans compare_report_BTCUSD_15m.json

[STEP 3] select
  ├─ Lit selection.json si existe (sinon : consomme compare_report s'il est
  │  frais pour asset/TF/seed/n_eval/max_conditions, sinon relance compare)
  ├─ GeneratorSelector(protocol).select(report) → winner
  └─ save(winner, outputs/selection.json)

[STEP 4] refine
  ├─ Recharge selection.json
  ├─ Reconstruit protocol avec n_eval_budget=20 (DÉFAUT !)
  ├─ GeneratorSelector.instantiate(winner) → TypedGPGenerator
  ├─ bind_data(...)
  ├─ generator.generate()
  │  └─ population_size = min(50, 20//11) = 2
  │     n_generations = min(10, max(0, 20//2 - 1)) = 9
  │     → produit 2 + (2 × 9) ≈ 20 hypothèses max
  ├─ Tri par Sharpe val, top 5
  └─ BeamRefiner.refine() sur chaque top → refined.json

[STEP 5] admit
  ├─ Recharge selection.json
  ├─ Reconstruit protocol avec n_eval_budget=20, n_samples=400 (tasting)
  ├─ TypedGPGenerator(protocol, config, engine)
  ├─ bind_data(...)
  ├─ generator.generate() → ~20 hypothèses (population=2, 9 gen)
  ├─ eval_budget = 20
  ├─ evaluated_candidates : 20 hypothèses testées sur val COMPLET
  │  (test_on avec with_bootstrap=True pour DSR/CI)
  ├─ pbo_candidate_paths : matrice candidats × trades
  ├─ DSR : n_indep_trials = 20 (FIX 2026-08-10, AVANT = index de boucle
  │  croissant, quasi infranchissable)
  ├─ n_val_years = len(val) / periods_per_year("15m")
  ├─ Pour chaque hyp (20) :
  │  ├─ evaluate_all_criteria(8 critères)
  │  ├─ compute_corpus_fracs() depuis corpus.jsonl
  │  ├─ admission_fn.decide(...)
  │  │  ├─ DSR_FAIL si p < 0.95
  │  │  ├─ PBO_FAIL si p > 0.20
  │  │  ├─ BOOTSTRAP_CI_FAIL si sharpe_ci_low ≤ 0 ou ret_ci_low ≤ 0
  │  │  ├─ N_TRADES_FAIL si n < 30
  │  │  ├─ CROISSANCE_FAIL si CAGR < 0.25 (i.e. +25%/an composé)
  │  │  ├─ CROSS_ASSET_FAIL (allow_single_asset=true en V1)
  │  │  ├─ DD_FAIL si max_dd > 0.25
  │  │  ├─ ALREADY_IN_ARCHIVE si fingerprint déjà rejeté
  │  │  └─ DIVERSITY_FAIL si signal_overlap > 0.30 ou ret_corr > 0.50
  │  │     ou quotas structurels violés
  │  └─ Si admis : append CorpusEntry, append ArchiveEntry (dédup OK)
  │     Si rejeté : append ArchiveEntry, continue
  └─ admit_summary.json

[STEP 6] holdout (manuel, jamais déclenché dans le pipeline automatisé)
  └─ HoldoutEvaluator.evaluate(1 Einher admis, snapshot val)
     Une seule consultation du holdout par Einher (sinon HoldoutAccessError)
```

### 3.3 Données réellement utilisées vs annoncées

| Annoncé | Réel | Écart |
|---|---|---|
| `train` = 60% des données | `train_ohlcv` = données 0..train_boundary-purge | OK |
| `val` = 20% externe | `val_ohlcv` = données val_start..val_end | OK |
| `holdout` = 20% final | `holdout_ohlcv` = données holdout_start..n | OK |
| purge = max_n = 50 bougies | applique aux deux bouts | OK |
| embargo = 1 bougie | OK | OK |
| n_eval_budget = 1200 | utilisé en compare, 20 en refine/admit | **divergence** |
| n_samples = 0 en compare, 400 en admit | **tasting activé en évolution** | suspect |
| "L'évolution utilise un split INTERNE du train" (bind_data) | confirmé, train[80%:] = split interne | OK |

### 3.4 Chemin d'exécution d'un individu particulier

Pour comprendre le destin d'un individu TypeGP, traçons `TypedGPGenerator_000045` :

1. **Init** : généré par `generate()` (TypedGPGenerator, ligne 746) comme
   46e individu (0-indexed), avec un arbre soit grow soit full de profondeur
   max=4. Direction tirée au hasard (50/50). ID = `TypedGPGenerator_000045`.

2. **Évaluation initiale** : `_evaluate_population([ind_45])` (ligne 760).
   - Si le train interne 80% a 2000 bougies, tasting réduit à 400.
   - `engine.train_calibrate(h, train_ohlcv, train_features)` → SL/TP figés.
   - `engine.test_on(h, val_ohlcv, val_features, calibrated, "val",
     with_bootstrap=False)` → MesuresBrutes sur tasting.
   - `_growth_fitness(m, ppy, min_trades=30)` → CAGR-based score.
   - Si `n_signals < 30` → fitness = -inf.

3. **Si -inf** : le tournoi (ligne 951) tire 3 indices, garde le meilleur.
   `max([(-inf, ind_A), (-inf, ind_B), (-inf, ind_C)])` retourne l'un des
   trois. **Mais la fitness -inf de l'individu n'élimine PAS l'individu**,
   il peut quand même être sélectionné et passé aux opérateurs évolutifs.

4. **Crossover (ligne 973)** : avec proba 0.8, swap un sous-arbre atomic
   ou compound entre 2 parents. Le nœud swapé est tiré uniformément parmi
   les nœuds de la catégorie.

5. **Mutation (ligne 1076)** : avec proba 0.2, remplace un sous-arbre par
   un nouveau (grow). Le nouveau sous-arbre a la même catégorie mais
   **PAS la même profondeur** (peut être plus profond que `max_conditions`).

6. **Élitisme** (ligne 792) : top 2 individus préservés. Le tournoi +
   élitisme convergente très vite.

7. **Déduplication finale** (ligne 802-808) : `sig = (h.condition_tree,
   h.direction, h.cooldown_k)`. Si deux individus sont structurellement
   identiques, un seul survit.

8. **Sortie** : population finale dédupliquée = `result.hypotheses`.

9. **Refine** (handle_refine) : top 5 par Sharpe val → BeamRefiner.
   - **Mais** : `n_eval=20` en refine, donc seulement 2 hypothèses
     générées (population=2). 0-1 top viable probable.

10. **Admit** (handle_admit) : 20 hypothèses testées sur val COMPLET,
    admission_filter appliqué. **Résultat observé : 0/35 admis, DSR_FAIL**.

11. **Historique** : l'individu 45 a 3 Einhers historiques dans le corpus.
    Chaque `_xNNNN` est un suffixe de mutation (cf. `_clone_with_tree`,
    ligne 1105).

---

## 4. Analyse TypeGP

### 4.1 Architecture

TypeGP implémente un STGP (Strongly-Typed GP) théorique (Koza 1992 + Montana
1995). En réalité c'est une variante **booléenne** : tout l'arbre retourne
un booléen, et les catégories sont juste `atomic` (feuille) vs `compound`
(AND/OR/NOT/XOR). **Le typage n'est pas au sens GP classique** : pas de
type de retour, pas de fonctions numériques composées. La docstring le
reconnaît (ligne 51-52 audit précédent).

### 4.2 Population initiale

- `population_size = 50` (défaut constructeur).
- Plafonné par `min(50, n_eval_budget // (n_generations + 1))`.
- **Avec n_eval=20 (refine/admit) : population=2, n_generations=9**.
- **Avec n_eval=1200 (compare) : population=50, n_generations=10**.
- **Avec n_eval=2000 (défaut compare) : population=50, n_generations=10**.

→ En refine/admit, **la population est TRÈS petite (2)**. Le moteur n'a
aucune chance d'explorer l'espace avec une population de 2.

**BUG-001 (CRITIQUE) : n_eval=20 par défaut en refine/admit écrase la
population de TypeGP à 2 individus.** C'est le bug structurel principal.

### 4.3 Construction des arbres (init)

- **Méthode grow** (ligne 837) : à chaque niveau, 50% de chance de retourner
  une feuille (après depth > 0). `op = self._rng.choice(list(LogicalOp))`.
- **Méthode full** (ligne 850) : tous les nœuds à `max_depth` sont des
  feuilles. `op = self._rng.choice(list(LogicalOp))`.
- 50% grow + 50% full, en alternance (ligne 747).

**Observations** :
- Le choix d'opérateur LogicalOp est uniforme : AND 25%, OR 25%, NOT 25%, XOR 25%.
- Un NOT unaire a 1 enfant, les autres 2. Probabilité d'avoir une feuille
  par nœud est ~50% en grow (ligne 839).
- **Pas de biais directionnel** : la direction est tirée au hasard 50/50
  (ligne 749). Donc le biais Short observé dans le corpus n'est PAS dû à
  l'init.

### 4.4 Feuilles (`_atom`)

- 35% de chance de tirer un **pattern booléen** (0/1) avec EQ/NE/IN
  (ligne 869-873).
- 65% de chance de tirer une **feature continue** avec LT/GT/LE/GE
  (ligne 874-878).
- Le seuil est tiré depuis le pool calibré sur le train (P1 #1, ligne
  877 : `_sample_threshold_for(feat)`).

**Observations** :
- `value = round(self._sample_threshold_for(feat), 4)` est un float. Pour
  les features continues, les seuils négatifs sont possibles (un `roc_10`
  est en %, donc peut être < 0).
- **L'opérateur EQ pour les patterns** (ligne 871) : `value = float(self._rng.randint(0, 1))`.
  Donc EQ(pat, 0) ou EQ(pat, 1). NE fait l'inverse. EQ et NE sont
  symétriques par inversion. **EQ(pat, 0) = NOT pat**, EQ(pat, 1) = pat.
  Pas de bug mais redondance possible.

### 4.5 Évaluation (fitness)

**Fitness = CAGR annuel composé** (ligne 905-907) :
```
fitness = (periods_per_year / avg_holding_period) * log1p(ret_mean_net)
Porte dure : n_signals < min_trades (30) → -inf
```

**Observations critiques** :

1. **Porte dure -inf très précoce** : un arbre qui produit 29 signaux sur
   val a une fitness = -inf. Avec 218 features continues, la probabilité
   qu'un arbre aléatoire produise ≥ 30 signaux est faible. Donc la
   majorité de la population a une fitness -inf. **Le tournoi avec fitness
   -inf ne peut rien apprendre** : sélectionner le "meilleur -inf" revient
   à sélectionner un individu au hasard parmi les -inf.

2. **Tasting biais potentiel** : la fitness est calculée sur 400 bougies
   de val en admit. Si ces 400 bougies sont un régime particulier (haussier,
   baissier, range), la fitness ne reflète pas le val complet. Un arbre
   peut être champion sur le tasting et perdant sur val complet.

3. **Pas de normalisation de la fitness** : entre générations, la fitness
   moyenne peut dériver. Mais comme la sélection est par tournoi (pas par
   roulette), c'est OK.

4. **La fitness ne pénalise pas la complexité** : un arbre de profondeur
   4 et un arbre de profondeur 50 ont la même fitness si même nombre de
   trades et même CAGR. Donc le bloat n'est pas défavorisé.

### 4.6 Sélection

- Tournoi binaire (k=3 par défaut, ligne 678).
- Tire 3 indices au hasard, garde le meilleur.
- `tournament_size=3` = pression sélective modérée (1/3 chance que le
  meilleur soit dans le tournoi).

**Observations** :
- Avec population=2 et fitness -inf, le tournoi retourne uniformément
  l'un des 2. **L'évolution est gelée**.
- Avec population=50 et fitness -inf partout sauf 1-2, le tournoi retourne
  presque toujours les mêmes "chanceux" → convergence rapide.

### 4.7 Crossover

- `crossover_prob=0.8` (80% de crossover, 20% de copie).
- `_subtree_crossover` (ligne 973) :
  1. Collecte les nœuds par catégorie (atomic/compound) pour chaque parent.
  2. Choisit une catégorie commune.
  3. Choisit un nœud aléatoire dans chaque parent.
  4. Swap les sous-arbres.

**Observations critiques** :

1. **Pas de contrôle de profondeur** : si parent 1 a un sous-arbre de
   profondeur 4 et parent 2 a un sous-arbre de profondeur 4, le swap peut
   produire un arbre de profondeur 8 (= 4 + 4 - 1). **Avec max_depth=4,
   on peut atteindre 11+ niveaux** après quelques opérations.

2. **Pas de contrôle de taille** : pas de limite sur le nombre de nœuds.

3. **Le `_swap_subtree` (ligne 1025) ne vérifie pas la profondeur du
   résultat**.

4. **Catégorie "atomic"** : un nœud atomic est une feuille (Condition).
   Si on swap deux feuilles, on remplace l'opérateur et la valeur. C'est
   équivalent à une mutation ponctuelle.

5. **Catégorie "compound"** : un nœud compound est un ConditionNode (AND,
   OR, NOT, XOR). Si on swap deux sous-arbres compound, on peut remplacer
   un AND par un OR, etc.

→ Le crossover n'est pas type-preserving au sens strict du STGP : il
swappe n'importe quel sous-arbre compound avec n'importe quel autre, ce
qui peut produire des arbres sémantiquement absurdes (NOT(NOT(X)) = X,
XOR(True, True) = False, etc.).

### 4.8 Mutation

- `mutation_prob=0.2` (20% de mutation par enfant).
- `_subtree_mutation` (ligne 1076) :
  1. Avec proba 0.2, choisit une catégorie.
  2. Choisit un nœud dans cette catégorie.
  3. Génère un nouveau sous-arbre de la même catégorie :
     - atomic : un nouveau `_atom()` (1 feuille)
     - compound : `_grow(max_depth=depth_remaining, depth=0)` où
       `depth_remaining = max(1, max_depth - len(path))`.

**Observations critiques** :

1. **`depth_remaining = max(1, max_depth - len(path))`** : si le path
   est plus long que max_depth (ce qui est possible après un crossover),
   `depth_remaining = 1`. La mutation ne peut alors créer qu'un sous-arbre
   de profondeur 1. **C'est un garde-fou partiel**, mais il ne s'applique
   qu'à la mutation, pas au crossover.

2. **La mutation est appliquée à TOUS les enfants** (ligne 777-778),
   pas seulement ceux qui sont passés par crossover. Donc chaque enfant
   a ~20% de chance d'être muté.

3. **Le `_grow` interne ne vérifie pas la profondeur totale** : il ne
   fait que limiter la profondeur du nouveau sous-arbre.

### 4.9 Remplacement (élitisme)

- `elitism=2` (ligne 679) : les 2 meilleurs individus sont préservés
  à chaque génération.
- Les `n_generations × n_population_size` enfants sont évalués, puis
  combinés avec les parents (`union_pop = population + offspring`), et
  on garde les N meilleurs.

**Observations** :
- L'élitisme=2 avec population=2 = toute la population survit. **Pas
  d'évolution** : on garde les 2 mêmes et on essaie des enfants.
- Avec population=50 et elitism=2, les 2 meilleurs sont préservés. Si
  les 2 meilleurs sont les mêmes pendant 10 générations, ils dominent
  toujours. **Convergence ultra-rapide**.

### 4.10 Diversité

- **Aucun mécanisme** : pas de novelty search, pas de fitness sharing, pas
  de speciation.
- Le seul "diversity mechanism" est la **dédup finale par signature
  structurelle** (ligne 802-808) : `sig = (h.condition_tree, h.direction,
  h.cooldown_k)`. Deux hypothèses avec exactement le même arbre + direction
  + cooldown sont fusionnées.

**Observations critiques** :

1. La dédup ne s'applique qu'au résultat final de `generate()`, pas
   pendant l'évolution. Donc des centaines d'individus identiques peuvent
   exister pendant l'évolution et gaspiller le budget d'évaluation.

2. La signature est trop stricte : deux arbres qui diffèrent par UNE
   feature ou UNE valeur sont considérés comme différents. Donc le
   fingerprint canonique (utilisé par admission) peut-être encore plus
   permissif.

### 4.11 Stagnation

- Aucun mécanisme de détection de stagnation (compteur de générations
  sans amélioration).
- Aucun mécanisme de "restart" partiel.
- L'élitisme=2 favorise la convergence précoce.

### 4.12 Bilan TypeGP

**TypeGP est un GP dégénéré** : population trop petite (par défaut), pas
de contrôle de complexité, pas de diversité, élitisme=2 = convergence
précoce, porte dure -inf = majorité de la population non-informative.

---

## 5. Analyse de la génération

### 5.1 Volumétrie

- `n_candidates=100000` (défaut) : la fonction `generate()` ne produit
  jamais 100k individus. Avec population=2 et 9 générations, c'est
  ~20 individus produits.
- `n_eval_budget=20` (refine/admit) : plafond d'évaluation à 20.
- `n_eval_budget=1200` (compare) : plafond à 1200/4 = 300 par générateur.

→ **Le ratio n_candidates/n_eval_budget est absurde** : on génère 100k
mais on en évalue 20. C'est un héritage du comparator où n_candidates
était utilisé par les baselines (qui produisent vraiment 100k individus
aléatoires).

### 5.2 Pipeline en chaîne

`pipeline = engine → baselines → compare → select → refine → admit → holdout`

Mais chaque étape a ses propres paramètres :
- `compare` : `n_eval_budget=2000` (défaut), `n_samples=0` (pas de tasting)
- `select` : consomme le rapport de compare (cache)
- `refine` : `n_eval=20` (override), `taste_samples=0` (pas de tasting)
- `admit` : `n_eval=20` (override), `taste_samples=400` (tasting)
- `holdout` : manuel

→ Le passage de compare (1200 eval) à refine/admit (20 eval) est un
**rétrécissement brutal**. C'est probablement pour des raisons de durée
(timeout 1h observé sur 5m), mais ça tue la diversité.

### 5.3 Bornes de la génération

- `max_conditions=4` (CLI) → `max_depth=4` dans TypeGP.
- `_atom` produit 1 feuille. `_grow` et `_full` produisent des arbres
  jusqu'à `max_depth=4` (= 16 feuilles au max en full, moins en grow).
- Mais le crossover et la mutation ne respectent pas cette borne.
  **Voir BUG-002**.

---

## 6. Analyse de l'évaluation

### 6.1 train_calibrate

1. Calcule `atr_p50` sur le train (Wilder, period=14).
2. N = clamp(ceil(amplitude/atr_p50), min_N, max_N) — pas pour multi_ATR.
3. **Passe provisoire** avec SL/TP = 1_000_000 × ATR (inatteignables) pour
   mesurer MFE/MAE sans sortie prématurée. C'est correct.
4. Dérive `tp_n_atr = (mfe_p50 * entry_median) / atr_p50` et
   `sl_n_atr = (mae_p75 * entry_median) / atr_p50`.
5. Bornes de sécurité : SL ∈ [0.1, 20], TP ∈ [0.1, 50].
6. **Plancher économique** : TP > 3× round_trip_pct, SL > 2× round_trip_pct.
   Sinon, `CalibrationError` explicite.

**Observations** :
- La calibration est saine : pas de recalibrage sur val/holdout (I-5).
- Le plancher économique évite les "TP < frais = perte garantie" → sain.
- **MAIS** : le seuil `min_tp_multiple_of_costs=1.5` est permissif (juste
  1.5× les coûts). Avec round_trip ≈ 0.04% × 2 = 0.08% par trade, le
  TP minimum = 0.12%. C'est très bas → la plupart des hypothèses passent
  le plancher.

### 6.2 test_on

1. Évalue `condition_tree` sur les features → masque booléen.
2. `_SignalFilter` applique le cooldown K (K=5 par défaut).
3. Pour chaque signal : récupère ATR local, calcule SL/TP (en prix absolus
   via `compute_sl_tp_at_entry`).
4. Délègue à `simulator.simulate` : simulation intrabar TP/SL, SL avant TP
   sur la même bougie (conservateur).
5. Calcule ret_brut et ret_net (avec coûts round-trip).
6. Agrège via `_MesuresAggregator` : Sharpe, bootstrap CI, n_trades, etc.

**Observations** :
- Pas de look-ahead : entrée à OPEN[t+1], ATR local (Wilder) ne regarde
  que le passé.
- Le SL avant TP sur la même bougie est conservateur. **MAIS** : si un
  trade touche à la fois le SL et le TP sur la même bougie, on prend le
  SL (perte). C'est le cas conservateur usuel.
- Le `compute_sl_tp_at_entry` utilise l'ATR **local** à l'entrée, pas
  l'ATR_p50 du train. C'est documenté (cf. docstring lignes 145-179).
  Cela permet de s'adapter à la volatilité courante, mais cela peut
  introduire un léger optimisme (l'ATR local est calculé sur la bougie
  d'entrée et le passé, mais pas sur des bougies futures).

### 6.3 Métriques agrégées

- **Sharpe** : `sharpe_per_trade * sqrt(periods_per_year / avg_holding_period)`.
  Annualise par la fréquence effective des trades.
- **Bootstrap CI** : sur Sharpe et ret_total, avec `bs_ppy = ppy / avg_held`.
  C'est le fix récent (commits 4751b66 et 9d8807a) : avant, le CI était à
  une échelle incompatible avec le Sharpe annualisé.
- **Max drawdown** : sur la courbe d'equity reconstruite trade par trade.
  Méthode correcte (pas de heuristique).

### 6.4 Look-ahead et leakage

**Pas de look-ahead apparent**. L'entrée à OPEN[t+1] est correcte, l'ATR
local est Wilder (sur le passé), les SL/TP sont calibrés uniquement sur
le train.

**MAIS** : le moteur utilise `ATR_p50` et `entry_median` du train pour
calculer les SL/TP en distances. C'est une calibration globale. Pas de
leakage.

**MAIS** : `_condition_evaluator._apply_op` (ligne 322-340) :
```python
if op == CompareOp.LT:
    return (col < value).fill_null(False)
```
`fill_null(False)` : si la valeur de la feature est NaN, la condition
est False (pas de signal). **C'est conservateur** : on ne trade pas sur
des NaN.

**MAIS** : `CompareOp.EQ` pour les features continues fait `(col - value).abs() < 1e-9`.
Pour un float, c'est très restrictif (la probabilité d'égalité stricte
est quasi nulle). **Ce n'est jamais utilisé** (les features continues
utilisent LT/GT/LE/GE dans `_atom`).

### 6.5 Bilan moteur

Le moteur est **techniquement correct** : pas de look-ahead, ATR bien
calculé, simulation intrabar correcte, métriques standards. Les fixes
récents (DSR annualisé, bootstrap CI à la même échelle) sont dans le bon
sens.

**Le moteur ne fait pas de la recherche**. C'est un évaluateur. C'est le
rôle du générateur de produire des hypothèses intéressantes, et c'est là
que TypeGP échoue.

---

## 7. Analyse de la validation

### 7.1 DSR (Deflated Sharpe Ratio)

- **Avant 2026-08-10** : DSR en mode "par-trade" (T=n_signals), avec
  déflation par index de boucle croissant. Le seuil 0.95 devenait
  quasi-infranchissable (e_max ~3.26 dès 200 essais).
- **Après** : DSR annualisé (sharpe_net * sqrt(Y) avec Y=années val),
  déflation par `n_indep_trials = len(hyps_a_evaluer)`. Le seuil 0.95
  redevient atteignable.

**Observations** :
- Le fix est correct conceptuellement. **MAIS** : 0/35 admis en post-fix
  → le DSR est peut-être trop strict, OU tous les candidats sont
  effectivement sur-appris.
- Le `dsr_metric` est dans `utils/metrics.py`. Je ne l'ai pas lu en
  détail, mais le wrapping (annualisé, n_indep_trials fixe) est correct.

### 7.2 PBO (Probability of Backtest Overfitting)

- CPCV K=6, embargo 1% (config).
- `candidate_paths` : matrice de `(entry_idx, exit_idx, ret_pct_net)` par
  candidat.
- **Rejet par défaut** : `if not candidate_paths or len(candidate_paths) < 2` → fail.
  → Si la matrice a moins de 2 candidats, PBO est indéfini et le critère fail.

**Observations** :
- PBO est une approche statistique rigoureuse, mais elle a besoin d'au
  moins ~30 candidats pour être significative. Avec n_eval=20, on est en
  dessous.
- Le K=6 et embargo=1% sont durs. Avec 6 groupes, on a 6 combinaisons
  (C(6,3) = 20), donc 20 paths CPCV. Mais l'embargo 1% peut être trop
  petit pour purger correctement.

### 7.3 Bootstrap CI

- `ci_low > 0` requis (Sharpe et ret).
- Block bootstrap IID (par défaut depuis commit 9d8807a), block comme
  option. Fix récent : avant, le block bootstrap réarrangeait l'ordre et
  détruisait la corrélation inter-blocs.
- Niveau 95% (n_resamples=2000).

**Observations** :
- Le bootstrap IID suppose des trades IID, ce qui n'est pas réaliste
  pour des stratégies séquentielles. Mais c'est moins pire que le block.
- `n_resamples=2000` est suffisant pour des CI stables.

### 7.4 n_trades minimum

- 30 (config). C'est un minimum raisonnable pour la significativité.

### 7.5 Croissance (CAGR)

- `min_cagr=0.25` (+25%/an composé).
- Formule : `(1 + ret_mean)^(ppy/avg_held) - 1`.

**Observations** :
- 25%/an composé = un compte x10 en 8 ans. C'est un objectif **ambitieux
  mais raisonnable** pour une stratégie de trading.
- `n_signals < 1` → fail immédiat. Cohérent.

### 7.6 Cross-asset

- `min_frac_assets_positive=0.70`, `min_n_assets=2`.
- `allow_single_asset=true` en V1 : un seul actif ne fail pas ce critère.
  Commentaire explicite : "En single-asset la cohérence cross-asset est
  SANS OBJET : la refuser systématiquement rendait l'admission = rejet
  permanent."

**Observations** :
- En single-asset, ce critère ne fait rien (passe par défaut).
- **C'est un opt-in de complaisance** : un Einher single-asset ne peut
  pas être validé sur la cohérence cross-asset, qui est le critère le
  plus important pour la robustesse.

### 7.7 Max drawdown

- `max_value=0.25` (-25%).
- Calculé sur l'equity_curve par trade.

**Observations** :
- 25% de drawdown est très permissif pour une stratégie de trading.
  La plupart des fonds visent < 15-20%.

### 7.8 Bilan validation

La validation est **rigoureuse et bien calibrée** (sauf cross-asset en
single-asset). Le problème n'est PAS que les seuils sont trop durs : le
problème est que les hypothèses qui arrivent à l'admission sont
intrinsèquement mauvaises (sur-apprises ou non-économiques).

---

## 8. Analyse de l'admission

### 8.1 Processus d'admission

`AdmissionDecider.decide()` (decision.py, ligne 129) :
1. Évalue les 7 critères.
2. Calcule les fingerprints (structurel + comportemental).
3. Vérifie la déduplication (Archive + Corpus courant).
4. Vérifie la diversité comportementale (Jaccard, corrélation).
5. Vérifie les quotas structurels.
6. Décision globale = tous les verrous doivent passer.
7. Si rejeté : append ArchiveEntry.

### 8.2 Quotas structurels (decision 2026-08-10, fix récent)

`evaluate_quotas` (diversity.py, ligne 319) :
- `family_max_frac=0.40` : 40% max d'une famille.
- `type_max_frac=0.60` : 60% max d'un type.
- `direction_min_frac=0.30` : 30% min Long ET 30% min Short.
- **Garde de démarrage** : si `total_exist <= 1.0`, les quotas sont
  désactivés (sinon 1/2 = 50% > 40%, structurellement bloqué).

**Observations** :
- Le fix est correct : sans lui, le 2e Einher était structurellement
  bloqué. Le 1er Einher n'est plus bloqué, et le quota s'active à partir
  du 2e.
- **MAIS** : `direction_min_frac=0.30` (30%) **appliqué en POURCENTAGE**
  d'un total renormalisé. Donc si on a 2 Short + 1 Long, les fractions
  sont 67%/33%. Le 30% est satisfait. Si on a 3 Short, c'est 100%/0% →
  fail. Mais si on a 1 Long + 1 Short, c'est 50%/50% → OK.

→ **Pour qu'un Short soit admis après 2 Short déjà admis, il faut un
4e Einher qui soit Long** (3/4 = 75% Short, 1/4 = 25% Long → fail).
Ou alors le 3e Einher doit être Long.

**C'est une explication possible du biais Short** : si l'évolution converge
rapidement vers Short, le 1er admis est Short. Le 2e admis doit aussi
être Short (toujours 1/1 = 100% Short, mais le quota est désactivé tant
que `total_exist <= 1.0`). Le 3e admis doit être Long (sinon 3/3 = 100%
Short → fail).

→ **Si l'évolution produit 5 Short équivalents, 1 seul sera admis**. Le
4e et 5e seront rejetés par `direction_min_frac`. C'est un filtre
**tardif** : il agit après l'évolution.

### 8.3 Dédup

`has_fingerprint` (archive/store.py, ligne 67) : itère TOUTES les entrées
de l'archive pour ce `data_version`. **Complexité O(n)**. Pour 10k
rejets, c'est 10k comparaisons par admission.

**Observations** :
- L'archive est append-only : on n'efface jamais. Donc la dédup devient
  de plus en plus stricte avec le temps. **C'est un anti-pattern** : un
  rejet ancien (avec un bug fixé depuis) bloque les essais futurs.
- **Le fingerprint est calculé sur la base du même `data_version`** (cf.
  ligne 77). Si le data_version change (recompilation des features), les
  rejets sont oubliés. C'est sain. Mais sinon, c'est une mémoire permanente.
- Le dedup est testé sur **fingerprint_structurel** ET **fingerprint_comportemental**.
  Le structurel est sur (condition_tree + sl_n_atr + tp_n_atr). Le
  comportemental est sur les descripteurs arrondis à 3 décimales.

### 8.4 Diversité comportementale

`BehavioralDescriptors._max_jaccard` : Jaccard entre `signal_indices` et
les `signal_indices` de chaque Einher du corpus.
`signal_overlap_max=0.30` : 30% max de Jaccard.

**Observations** :
- Le Jaccard est calculé sur les **indices de bougies** (entrée_idx).
  Deux Einhers qui prennent des trades sur les mêmes bougies ont un
  Jaccard élevé → rejetés.
- C'est une diversité **temporelle** (date de signal), pas une diversité
  **sémantique** (type de règle). Deux Einhers avec des règles différentes
  mais qui tradent en même temps sont rejetés.
- **C'est trop strict** : si un marché a 5 grosses périodes de momentum,
  tous les Einhers qui tradent pendant ces périodes sont rejetés. Le
  marché force la convergence temporelle.

### 8.5 Bilan admission

L'admission est **bien conçue** (tous les verrous sont nécessaires) mais
**trop stricte** en pratique :
- Le DSR est peut-être trop strict pour des petits échantillons (n=20).
- Le PBO fail par défaut si moins de 2 candidats.
- Le Jaccard temporel est trop restrictif.
- L'archive est memoryless-fail : un ancien rejet bloque tout.

---

## 9. Pourquoi seulement 3 Einhers ?

### 9.1 Le pipeline actuel admet 0 Einhers

`outputs/admit_summary.json` (dernier run) :
```json
{
  "n_generated": 35,
  "n_admitted": 0,
  "n_rejected": 35,
  "rejection_breakdown": { "DSR_FAIL": 35 },
  "generator": "TypedGPGenerator"
}
```

**100% des candidats sont rejetés par DSR**.

### 9.2 Cause racine

Causes possibles (à discriminer par observation) :

1. **Le DSR post-fix est trop strict** : avec n_indep_trials=20 et Sharpe
   modestes (médiane -142 !), aucun candidat ne passe DSR >= 0.95.
2. **Tous les candidats sont effectivement sur-appris** : le TGP converge
   vers un individu qui performe sur tasting (400 bougies) mais perd sur
   val complet.
3. **Le moteur a un bug** : Sharpe négatif aberrant (-142) est suspect.

**Observation cruciale** : `median_sharpe_all=-142.07` est aberrant. Un
Sharpe annualisé de -142 signifie que le moteur produit des résultats
complètement aberrants. **C'est probablement la cause racine**.

### 9.3 Origine des 3 Einhers historiques

Les 3 Einhers dans `corpus.jsonl` ont :
- Sharpe val : 14.8, 34.5, 24.8 (très élevés)
- DSR : 1.0
- PBO : 0.17, 0.0, 0.0
- data_version : non visible dans l'extrait, mais le test confirme que ce
  sont d'anciennes admissions

→ Ces Einhers ont été admis avec un **DSR pré-fix** (avant 2026-08-10),
probablement avec un DSR "per-trade" où 1.0 = "le test est passé" mais
n'avait pas de sens statistique.

### 9.4 Chaîne causale

```
n_eval_budget=20 par défaut en admit
        ↓
TypedGPGenerator.population_size = min(50, 20 // 11) = 2
        ↓
population de 2 individus
        ↓
n_generations = 9 (mais 2 parents × 9 = 18 enfants ≈ 20 max)
        ↓
20 hypothèses générées
        ↓
mais : tester 20 hypothèses contre 1200 bougies val complet
       où le moteur produit des Sharpe aberrants (-142 médian)
        ↓
DSR fail quasi-systématique
        ↓
0 admission en pipeline actuel
```

---

## 10. Pourquoi les Einhers sont-ils presque identiques ?

### 10.1 Faits durs

Les 3 Einhers partent **tous de l'individu 45** de la population initiale :
- `TypedGPGenerator_000045_x1465_x9012_x3909_x2735_x8021_x5279_x7898_x2461_x7749_x9303_x5424`
- `TypedGPGenerator_000045_x8835_x1956_x9447_x2250_x4819_x0665_x2095_x2949_x5629_x0807`
- `TypedGPGenerator_000045_x8835_x1493_x4933_x8571_x8691_x4097`

Le format `_xNNNN` correspond à `_clone_with_tree` (ligne 1105) :
`id=f"{h.id}_x{self._rng.randint(0, 9999):04d}"`.

Donc le 46e individu de la population initiale a survécu à l'élitisme et
a été sélectionné 3 fois (par 3 chaînes de mutations différentes) pour
fournir les 3 Einhers admis.

### 10.2 Causes racines

1. **Élitisme=2 favorise un individu précoce** : si l'individu 45 a une
   bonne fitness relative (même -inf, il est dans le top 2 par tirage
   aléatoire), il survit à toutes les générations.

2. **Population de départ déjà très petite** : avec population=2 ou
   population=50 mais fitness=-inf quasi partout, la sélection favorise
   l'individu qui a le plus de chance.

3. **Crossover sous-arbre type-preserving** : si tous les individus ont
   des structures similaires (AND-XOR-OR), les crossover produisent des
   variantes mineures, pas des structures nouvelles.

4. **Dédup finale trop tardive** : la dédup ne s'applique qu'au résultat
   final. Pendant l'évolution, on a 18 enfants presque tous descendants
   de l'individu 45.

5. **Pas de mécanisme anti-convergence** : pas de speciation, pas de
   novelty search, pas de restart partiel.

### 10.3 Pas de "vraie" diversité sémantique

Les 3 Einhers ont des **structures d'arbre similaires** (AND-XOR-OR) mais
**des features différentes** (donc des fingerprint_structurel différents).
Ils sont **structurellement distincts** (3 fingerprints différents) mais
**sémantiquement proches** (mêmes opérateurs logiques, même type de
règles).

→ Le fingerprint structurel est trop permissif (différentes combinaisons
de features produisent des fingerprints différents). Le fingerprint
comportemental capte le Jaccard temporel mais pas la diversité des règles.

---

## 11. Pourquoi sont-ils tous Short ?

### 11.1 Faits durs

3/3 Einhers sont Short. C'est un échantillon trop petit pour conclure à
un biais statistique du marché.

### 11.2 Causes techniques possibles

1. **L'évolution favorise Short** : si le tasting (400 bougies) est en
   période baissière, les Shorts sont favorisés par la fitness CAGR.
   Mais c'est du bruit, pas un biais.

2. **Le moteur d'évaluation a un biais Short** :
   - Les coûts sont symétriques (0.08% round-trip).
   - Mais la simulation intrabar (SL avant TP sur la même bougie)
     favorise les Shorts : sur un marché haussier volatile, un Long
     touche souvent le SL sur la même bougie (reversal) qu'il aurait
     touché le TP sans le SL. Un Short bénéficie du même effet, mais
     la probabilité conditionnelle n'est pas symétrique.
   - **C'est suspect mais pas mesurable sans backtest dédié**.

3. **Le DSR pré-fix (qui a admis les 3 Einhers) favorisait les hypothèses
   avec peu de trades et fort Sharpe** : sur BTC, les mouvements baissiers
   ponctuels (corrections) produisent de forts rendements Short en peu
   de temps. Un Short serré (SL/TP serrés) capture le reversal, le Long
   équivalent capture la continuation. Si le marché est en range avec
   des corrections fréquentes, les Shorts sont favorisés.

4. **Le bootstrap CI est en mode IID** (depuis commit 9d8807a) : pour
   des Shorts qui profitent de gaps baissiers, les rendements sont plus
   IID que pour des Longs qui dépendent de la continuation. **Le CI est
   plus serré pour les Shorts** → plus susceptibles de passer.

5. **Le cross-asset est désactivé en single-asset** : on ne peut pas
   valider la cohérence sur d'autres actifs, qui pourraient équilibrer
   le ratio Long/Short.

### 11.3 Bilan

**Le biais Short n'est pas une propriété du marché BTC** : c'est
probablement un artefact du couple (évolution convergente + DSR
permissif pré-fix + tasting sur une période baissière).

---

## 12. Pourquoi un seul timeframe ?

### 12.1 Faits durs

- `selection.json` du dernier run : `timeframes: ["5m"]` (un seul TF).
- `corpus.jsonl` : 1 Einher sur 5m, 2 sur 15m. **Pas un seul TF, mais
  deux TFs concentrés (5m et 15m)**.

### 12.2 Cause : la pipeline n'itère PAS sur les timeframes par défaut

`handle_run` (ligne 1246) :
```python
for mode in ("engine", "baselines", "compare", "select", "refine", "admit"):
```
Ce mode itère sur les **modes**, pas sur les **timeframes**.

Le mode `--all-timeframes` (ligne 1218-1222) itère sur les TF **uniquement
si `--all-assets` est activé** :
```python
all_tfs = (
    list_available_timeframes(args.data_root, "__any__")
    if getattr(args, "all_timeframes", False)
    else ()
)
```

→ Par défaut (`--all-assets` non activé), on lance UN seul run avec
`--data-timeframe` (1h par défaut). Le user a peut-être lancé
`--data-timeframe=5m` puis `--data-timeframe=15m` séparément.

### 12.3 Conséquence

Le pipeline ne fait pas de **vraie exploration cross-timeframe**. Les
Einhers sont scopés à un seul TF, et la cross-asset diversity (P1-10)
n'inclut pas les TF.

### 12.4 Bilan

**C'est un choix architectural, pas un bug**. Mais c'est une **limitation
majeure** : un Einher 5m et un Einher 15m sont en fait des stratégies
différentes (différentes échelles de temps, différents coûts relatifs).

---

## 13. Analyse des conditions géantes

### 13.1 Faits durs

L'Einher 1 a un arbre de profondeur ≥ 11 (chaîne `_x` = 11 mutations/
crossovers). La représentation sérialisée dépasse 50 KB.

L'arbre contient :
- `williams_r > -5.0` : très permissif (top 5% du range).
- `pattern_breakaway_bear == 1.0` ET `pattern_breakaway_bull == 0.0` :
  redondance (le second est la négation du premier).
- `pattern_three_white_soldiers == 0.0` : NOT pattern.
- `quant_amihud_illiquidity <= 5.0` : seuils sur des features quant
  inhabituelles.
- `roc_10 <= -0.9624` : -96% en 10 bougies — possible mais extrême.
- `high <= 0.0007` : feature high comparée à une valeur minuscule
  (log-return ou prix ?).
- `parabolic_sar <= 5.0` : feature SAR comparée à 5 (en prix ?).
- `money_flow_index < 5.0` : MFI dans le top 5% bas.

### 13.2 Causes racines

1. **BUG-002 (CRITIQUE) : pas de contrôle de profondeur dans le crossover**
   (`_subtree_crossover` ligne 973). Après quelques opérations, l'arbre
   dépasse largement `max_conditions=4`.

2. **BUG-003 (MAJEUR) : pas de pénalité de complexité dans la fitness**.
   Le CAGR est indifférent à la taille de l'arbre.

3. **BUG-004 (MODÉRÉ) : pas de simplification post-hoc**. Pas de
   constante-folding, pas de NOT(NOT(X))=X, pas de EQ(pat,0)=NOT(pat).

4. **BUG-005 (MODÉRÉ) : seuils tirés dans le pool calibré mais sans
   bornes explicites**. Un seuil comme `5.0` sur `parabolic_sar` peut
   être absurde si la feature est un prix.

5. **BUG-006 (MINEUR) : comparaison de features sur des échelles
   différentes**. `high` est un prix (peut-être normalisé) et `0.0007` est
   peut-être un log-return. Si la feature est un prix, `high <= 0.0007`
   est toujours False pour BTC (prix > 10 000).

### 13.3 Information utile vs bloat

**Aucune des conditions n'apporte d'information financière nouvelle**.
Toutes les conditions sont des filtres ultra-permissifs (`williams_r > -5.0`)
ou des négations redondantes (`pattern_X == 0.0` = `NOT pattern_X`).

→ **C'est du bloat pur, pas de la complexité informative**.

---

## 14. Analyse des valeurs absurdes / nulles

### 14.1 Valeurs nulles

Aucune valeur `null` dans les conditions. Les `value` sont des floats
tirés du pool calibré ou 0.0/1.0 pour les patterns.

### 14.2 Valeurs absurdes

- `high <= 0.0007` : si `high` est un log-return OHLC, c'est ~0.07% → très
  restrictif. Si `high` est un prix, c'est toujours False pour BTC.
  **Dépend de l'interprétation**.
- `roc_10 <= -0.9624` : -96% en 10 bougies → très extrême.
- `quant_amihid_illiquidity <= 5.0` : Amihud illiquidity en valeur absolue,
  peut être > 5 sur des marchés peu liquides. Seuils arbitraires.
- `parabolic_sar <= 5.0` : SAR en prix. Pour BTC à 60 000, `SAR <= 5.0`
  n'est jamais vrai.

### 14.3 Origine

Le `_sample_threshold_for(feat)` tire uniformément dans le pool calibré
par feature (cf. `data/threshold_calibration.py`). Le pool est censé
contenir des quantiles de la feature. **MAIS** : le pool est probablement
le même pour toutes les features (fallback `_FALLBACK_THRESHOLD_POOL =
(-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)`) si la calibration échoue.

→ Un seuil comme `5.0` ne devrait pas apparaître si le pool est bien
calibré. **Soit le pool merge_quantile_pools utilise des quantiles 0/100
qui dépassent, soit il y a un fallback incontrôlé**.

### 14.4 Bilan

**Les seuils sont tirés sans borne supérieure par feature**. Le pool
calibré ne devrait pas contenir `5.0` pour `parabolic_sar` (qui est un
prix en BTC). **C'est un bug de calibration**.

---

## 15. Bugs et causes racines

### 15.1 Tableau des bugs

| ID | Gravité | Phase | Fichier/fonction | Cause racine | Symptôme | Conséquence | Correction proposée |
|---|---|---|---|---|---|---|---|
| BUG-001 | CRITIQUE | Refine/Admit | `discovery.py:handle_refine/admit` | `n_eval=20` par défaut écrase `population_size` à 2 et limite l'évolution à 9 générations | Très peu d'hypothèses générées, fitness -inf quasi partout | 0 admission systématique en admit | Override `n_eval` à 200+ minimum OU découpler `n_eval` du calcul de `population_size` |
| BUG-002 | CRITIQUE | Évolution | `generators/algorithms.py:TypedGPGenerator._subtree_crossover` | Pas de contrôle de profondeur après swap. `max_depth` initial peut être dépassé de 2x à chaque crossover | Arbres de profondeur 11+ (bloat non borné) | Conditions gigantesques sans information utile, surapprentissage sur le val | Vérifier profondeur après swap, rejeter ou réparer si > max_depth*1.5 |
| BUG-003 | MAJEUR | Évolution | `generators/algorithms.py:TypedGPGenerator._evaluate_population` | Fitness = CAGR sans pénalité de complexité, pas de terme d'anti-bloat | Les arbres géants sont favorisés s'ils ont le même CAGR | Bloat incontrôlé qui consomme le budget sans gain | Ajouter pénalité `-λ * log(taille_arbre)` à la fitness |
| BUG-004 | MAJEUR | Évolution | `generators/algorithms.py:TypedGPGenerator.generate` | Pas de mécanisme de diversité (novelty, speciation, fitness sharing) | Convergence ultra-rapide vers 1-2 individus | 3 Einhers jumeaux issus du même parent | Implémenter speciation (distance structurelle) ou novelty search |
| BUG-005 | MAJEUR | Diversité | `admission/diversity.py:evaluate_quotas` | `direction_min_frac=0.30` appliqué tardivement (après admission) | Si l'évolution produit N Short équivalents, 1 seul admis | Pipeline accepte mal la concentration | Pré-filtrer dans TypeGP : forcer diversité directionnelle pendant l'évolution |
| BUG-006 | MAJEUR | Validation | `admission/criteria.py:evaluate_dsr` | DSR peut être trop strict pour n_trials=20 et ppy=15m | 35/35 candidats rejetés par DSR_FAIL | 0 admission systématique | Étalonner DSR sur cas réel : si tous les candidats sont rejetés, c'est suspect |
| BUG-007 | MAJEUR | Validation | `admission/criteria.py:evaluate_pbo` | PBO fail par défaut si moins de 2 candidats ou n_groups*2 spans | Avec n_eval=20 et 6 groupes CPCV, PBO est souvent indéfini | 20/20 candidats rejetés par PBO_FAIL | Soit augmenter n_eval, soit rendre PBO optionnel pour petits runs |
| BUG-008 | MODÉRÉ | Calibration | `data/threshold_calibration.py` | Pool de seuils ne contient pas de bornes spécifiques par feature. Fallback uniforme -2..2 | Seuils comme `5.0` sur `parabolic_sar` (en prix) | Comparaisons absurdes, fitness non informative | Seuils bornés par feature dans le pool |
| BUG-009 | MODÉRÉ | Architecture | `discovery.py:handle_run` | Pas d'itération sur les timeframes par défaut | Un seul TF par run, sauf opt-in `--all-timeframes` | Pas de cross-TF exploration | Itérer sur les 5 TF par défaut, sauf si explicitement désactivé |
| BUG-010 | MODÉRÉ | Dédup | `archive/store.py:has_fingerprint` | O(n) sur l'archive, memoryless-fail permanent | Re-essais bloqués par anciens rejets | Pipeline ne peut pas progresser | Index inversé fingerprint→entry (set/dict) au lieu de scan |
| BUG-011 | MINEUR | Loader | `data/npy_real_loader.py:load_ohlcv_from_npy` | Fonction lève `NpyRealLoaderError` systématiquement (code mort) | Confusion (le docstring annonce un retour) | Aucun (code jamais appelé) | Supprimer ou corriger |
| BUG-012 | MINEUR | Moteur | `engine/evaluator.py:_apply_op.EQ` | Comparaison flottante stricte `abs(col-value)<1e-9` | EQ continu quasi-jamais vrai | OK (pas utilisé en V1) | Documenter ou remplacer par bandes de tolérance |
| BUG-013 | MINEUR | Moteur | `engine/simulator.py` (non lu) | Convention SL avant TP sur même bougie | Biais potentiel pour les Shorts en marché volatile | Faible, conservateur | Documenter le choix |
| BUG-014 | AMÉLIORATION | Code | `generators/algorithms.py` (BNF) | BNF 117 KB jamais utilisé en pratique (GE exclu par défaut) | Code mort, complexité cognitive | Maintenance | Purger BNF si GE reste exclu |

### 15.2 Causes racines des symptômes observés

**Symptôme : "3 Einhers BTCUSD"**
- Historique : 3 admissions d'avant le fix DSR, gelées dans corpus.jsonl.
- Pipeline actuel : 0 admission (DSR_FAIL à 100%).

**Symptôme : "jumeaux"**
- Convergence précoce (élitisme=2 + population petite + porte -inf).
- Pas de mécanisme de diversité (BUG-004).
- 1 individu initial (TypedGPGenerator_000045) favorisé par hasard.

**Symptôme : "tous Short"**
- Échantillon trop petit (3) pour conclure.
- Mais : couple (évolution convergente + DSR pré-fix permissif + tasting)
  favorise les Shorts en pratique.

**Symptôme : "1 TF"**
- Par défaut, le pipeline n'itère pas sur les TF (BUG-009).
- 2 TF en pratique (5m + 15m), pas 1.

**Symptôme : "conditions géantes"**
- Bloat non borné (BUG-002 + BUG-003).
- 11 niveaux de mutations empilés sur l'individu initial.

**Symptôme : "valeurs absurdes"**
- Pool de seuils sans bornes spécifiques (BUG-008).

---

## 16. Régressions potentielles

### 16.1 Régressions confirmées (commits récents)

Le git log montre une série de fixes qui sont des **régressions corrigées**
mais qui révèlent des bugs structurels :

| Commit | Régression corrigée | Cause initiale | Statut |
|---|---|---|---|
| `c9ea969` | "Fix admission : evolution sur tasting (duree), finalistes sur val complete (filtre DSR) — val complete par generation = 2-3h/TF" | L'évolution tournait sur val complet (trop lent) | Corrigé (tasting) |
| `be9df61` | "Fix admission : generation sur val COMPLETE (n_samples=0 force) — le tasting a l'admission produisait des candidats sur-appris" | Le `n_samples` était hérité du protocol du compare | Corrigé |
| `18dae9f` | "Fix quotas demarrage : le 2e Einher (1/2=50%>40%) etait structurellement bloque — quota applique seulement a partir de 2 Einhers" | Quota `family_max=0.40` appliqué dès 1 Einher, structurellement bloqué | Corrigé |
| `e25822d` | "Fix quotas demarrage : le 2e Einher" | Suite du précédent | Corrigé |
| `9d8807a` | "Fix : sum() au lieu de np.sum (np non importe dans bootstrap.py) — suite complete verte (267)" | Régression de dépendance (np non importé) | Corrigé |
| `9d86d2c` | "Bootstrap CI mode iid (defaut) : resampling simple des trades" | Block bootstrap détruisait la corrélation | Corrigé (IID par défaut) |
| `74aad15` | "Fix bootstrap CI : meme echelle que sharpe_net (ppy/avg_held annualise)" | CI et Sharpe à des échelles différentes | Corrigé |
| `4751b66` | "Fix DSR : unite annualisee (sharpe_net x sqrt(Y), Y=annees de val) + deflation par nb de finalistes au lieu de l'index de boucle" | DSR en mode per-trade + déflation croissante | Corrigé |
| `7e38437` | "Fix instantiate TypedGP : passer engine (ValueError sinon — fitness Sharpe pendant l'evolution)" | Refine/admit rc=1 | Corrigé |
| `1dbd46b` | "Fix fraicheur rapport compare : persister max_conditions dans meta" | select retombait en re-comparaison | Corrigé |
| `dd3e8db` | "Optimisation run-time evolution : tasting (n_samples seede, decision 2026-08-10) + with_bootstrap=False hors admission + flag --taste-samples (defaut 0)" | Timeout 1h sur 5m | Corrigé |
| `b59d0d5` | "TypedGP : pool complet 218/218 (critere value_type, 9 *_signal recup) + helper _value_type_of" | 9 *_signal perdus | Corrigé |
| `c5a69d1` | "Etat de reference : patches arbres (fitness croissance, pool patterns EQ/NE, proto max_conditions=4, select-consomme-rapport)" | Précédent | Référence |
| `0a4aef4` | "Fix data loading : timestamp int64+datetime normalises en datetime[us, UTC]" | OHLCV/features désynchronisés | Corrigé |
| `56599ce` | "Code quality : 11 corrections ethique/documentation (review finale)" | Review | Corrigé |

### 16.2 Régressions plausibles non confirmées

| Régression | Indice | À vérifier |
|---|---|---|
| `_Default_Excluded_Generators` retire Random/GE par défaut | Commit "Philosophie arbres d'abord (decision utilisateur 2026-08-09)" | Le comparator n'a plus de diversité de générateurs. Le winner est élu par défaut (0.5) |
| `n_eval=20` par défaut en refine/admit | Pas de commit explicite, mais c'est dans `discovery.py:handle_refine/admit` | Probable régression silencieuse : le budget engine était plus grand avant |
| `cooldown_k=5` par défaut | `protocol.cooldown_k` lit depuis `config/cooldown.default_k` | Si la config a changé, le cooldown peut être trop permissif |
| `max_conditions=4` vs profondeur réelle 11+ | Doc dit 4, code peut produire 11+ | Incohérence doc/code, bloat non borné |
| Comparaison score min-max relative | Normalisation min-max entre générateurs | Avec 1 seul gen, le score est 0.5 par défaut |

### 16.3 Régression du test engine_no_leak

L'audit précédent (TGP-05) signale que le test `test_different_seeds_different_result`
échoue : deux seeds produisent le même IC bootstrap. **Le test échoue
toujours**. C'est une violation du contrat de déterminisme.

---

## 17. Code obsolète à purger

### 17.1 Code mort confirmé

| Élément | Emplacement | Raison de purge |
|---|---|---|
| `load_ohlcv_from_npy` | `data/npy_real_loader.py:85` | Lève `NpyRealLoaderError` en première ligne (code mort) |
| `Generators.__init__` (BNF) | `generators/bnf.py` 117 KB | GE exclu par défaut de la compétition (CLI) |
| `_old/` directory | `src/einherjar/research/_old/` | Code archivé, déjà exclu du périmètre |
| `v2/` directory | `src/einherjar/research/v2/` | Idem |

### 17.2 Code à isoler (encore utile mais hors chemin actif)

| Élément | Raison d'isoler |
|---|---|
| `GrammaticalEvolutionGenerator` | Exclu par défaut de la compétition |
| `MemeticGenerator` | Pas sélectionné par compare (score trop bas) |
| `NSGA2Generator` | Pas sélectionné par compare (score trop bas) |
| `BeamSearchGenerator` | Pas sélectionné par compare (score trop bas) |
| `RandomSearchGenerator` | Exclu explicitement |
| `Refinement.beam` (BeamRefiner) | Toujours appelé en refine, mais si refine est down-prioritized, BeamRefiner l'est aussi |
| `Baselines.algorithms` | Appelées en baselines, mais le résultat n'influence pas le pipeline principal |
| `Comparator` et `Selector` | Plus de compétition multi-générateurs, donc plus de winner réel |
| `_old.portfolio.*` | Code archivé |

### 17.3 Code à conserver (chemin actif TypeGP)

| Module | Rôle |
|---|---|
| `discovery.py` | Point d'entrée, modes run/admit/holdout |
| `engine/evaluator.py` | Moteur d'évaluation (calibration + test_on) |
| `engine/bootstrap.py` | Block bootstrap IID |
| `engine/simulator.py` | Simulation intrabar |
| `generators/algorithms.py:TypedGPGenerator` | Générateur unique |
| `admission/decision.py` + `criteria.py` + `diversity.py` | Décision d'admission |
| `corpus/store.py` | Sérialisation corpus |
| `archive/store.py` + `schema.py` | Sérialisation rejets |
| `data/npy_real_loader.py:load_features_from_npy` | Loader features |
| `data/ohlcv.py:OhlcvProvider` | Loader OHLCV CSV bruts |
| `data/features.py` | FeaturesFrame |
| `data/validation.py` | Validation OHLCV/features |
| `data/versioning.py` | DataVersion + DataVersionStore |
| `data/threshold_calibration.py` | Pool de seuils par feature |
| `holdout/evaluator.py` + `ledger.py` | Holdout unique |
| `utils/types.py` | Hypothesis, Condition, MesuresBrutes, Einher |
| `utils/fingerprint.py` | fingerprint_structurel + comportemental |
| `utils/stats.py` | ATR, max_drawdown, periods_per_year, bootstrap helpers |
| `utils/metrics.py` | DSR |
| `config/loader.py` + YAMLs | Configuration |

### 17.4 Code à refactorer

| Module | Problème | Refonte proposée |
|---|---|---|
| `generators/algorithms.py:TypedGPGenerator` | Population=2 par défaut, bloat non borné, pas de diversité | Refonte complète : population ≥ 50, contrôle de profondeur, speciation |
| `admission/criteria.py:evaluate_dsr` | Trop strict en pratique | Étalonnage empirique : comparer admission DSR vs admission finale sur holdout |
| `archive/store.py` | O(n) sur l'archive, memoryless-fail | Index inversé fingerprint→entry, distinguer "doublon structurel" de "rejet historique" |
| `discovery.py:handle_admit` | n_eval=20 écrase population | Override `n_eval` minimum OU découpler du calcul de `population_size` |

---

## 18. Architecture à conserver

- **Moteur d'évaluation** : sain, pas de look-ahead, métriques standard,
  bootstrap IID, DSR annualisé, CPCV pour PBO.
- **Pipeline de séparation train/val/holdout** : 60/20/20, purge=50,
  embargo=1. OK.
- **Sérialisation corpus/archive** : append-only JSONL, atomique, traçable.
- **DataVersionStore** : verrouille la version de données, bloque les
  rejets/perte.
- **Fingerprints structurel + comportemental** : bonne idée, mais l'usage
  actuel (déjà-admis + dédup stricte) est trop punitif.
- **Holdout unique + atomique** : protection contre la double-consultation.
- **Tasting (sous-échantillon)** : correct conceptuellement, mais doit
  être désactivé pour l'admission (commit be9df61 l'a déjà fixé).

---

## 19. Architecture à refactorer

### 19.1 TypeGP

Refonte prioritaire :
1. **Population minimale** : 50 par défaut, plafond 200.
2. **Contrôle de profondeur strict** : `_subtree_crossover` et `_subtree_mutation`
   doivent rejeter toute opération qui dépasse `max_depth * 1.5`.
3. **Mécanisme de diversité** : speciation (distance structurelle) ou
   novelty search (distance dans l'espace des outputs).
4. **Pénalité de complexité** dans la fitness : `-λ * log(taille)`.
5. **Élitisme adaptatif** : 1 au lieu de 2, ou elitisme avec critère de
   diversité.
6. **Restart partiel** : si la population stagne pendant N générations,
   garder le meilleur 20% et regénérer le reste.
7. **Direction imposée** : alterner Long/Short au sein de la population
   (50/50 strict), pas de tirage 50/50 aléatoire.

### 19.2 Calibration

- **Seuils bornés par feature** : chaque feature a un range valide
  (par exemple, RSI ∈ [0, 100], price ∈ [0, +∞], log-return ∈ [-1, 1]).
- **Seuils distincts par famille** : pattern binaire → {0, 1}, momentum
  → quantiles 0-100.

### 19.3 Admission

- **Dédup graduelle** : rejeter un Einher ssi il est EXACTEMENT le même
  qu'un Einher déjà admis (fingerprint structurel). Pas de dédup sur
  l'archive pour les anciens rejets.
- **Diversité temporelle relâchée** : Jaccard 0.50 au lieu de 0.30.
- **DSR conditionnel** : si n_indep_trials < 30, désactiver le DSR et
  utiliser un autre critère (e.g., Sharpe brut + bootstrap CI).
- **PBO conditionnel** : si n_candidats < 30, désactiver PBO.

### 19.4 Découpage budget

- Découpler `n_eval_budget` (coût moteur total) de `n_candidates`
  (volume de génération) de `population_size` (paramètre GP).
- Forcer `population_size >= 50` indépendamment du budget.

---

## 20. Plan de refonte proposé

### 20.1 Ordre proposé

Cet ordre est construit pour minimiser les risques de régression et
maximiser la valeur à chaque étape. **Ce n'est pas l'ordre de la
précédente liste (que je ne reprends pas telle quelle)**.

#### Étape 1 : Purger le code obsolète (sans casser le pipeline actif)

- Supprimer `data/npy_real_loader.py:load_ohlcv_from_npy` (code mort).
- Supprimer le répertoire `research/_old/` et `research/v2/` (déjà
  exclus mais pollue le repo).
- Supprimer `generators/bnf.py`, `bnf_parser.py`, `bnf_semantic.py` (BNF
  jamais utilisé en pratique, GE exclu par défaut).
- Supprimer les générateurs `RandomSearchGenerator`, `BeamSearchGenerator`,
  `GrammaticalEvolutionGenerator`, `MemeticGenerator`, `NSGA2Generator`
  (hors chemin actif).
- Supprimer `generators/comparator.py` et `selection/selector.py` (plus
  de compétition multi-générateurs).
- Supprimer `refinement/beam.py` (refine non utilisé si le pipeline est
  simplifié).
- Supprimer `baselines/` (baselines peuvent rester en mode debug, mais
  pas dans le pipeline principal).

→ **Résultat** : un repo simplifié, ~3 000 lignes au lieu de 13 000.

#### Étape 2 : Refondre TypeGP

- Réécrire `TypedGPGenerator` avec :
  - `population_size` par défaut = 50, plafond = 200.
  - `n_generations` par défaut = 20, plafond = 50.
  - `max_depth` strict = 4, **vérification post-opérateur**.
  - **Speciation** : distance structurelle (Jaccard sur les ensembles de
    features) entre individus, fitness sharing au sein d'une espèce.
  - **Pénalité de complexité** : `-0.01 * log(taille)` dans la fitness.
  - **Élitisme = 1** (au lieu de 2).
  - **Direction imposée** : 50% Long, 50% Short, alterné.
  - **Restart partiel** : si stagnation 5 générations, restart 50%.
- Conserver la fitness CAGR + bootstrap CI.

#### Étape 3 : Corriger l'admission

- Découpler `n_eval_budget` du calcul de `population_size`.
- Rendre DSR et PBO conditionnels (désactivés si n_trials < 30).
- Relâcher `signal_overlap_max=0.50` (au lieu de 0.30).
- Distinguer "doublon structurel" (fingerprint = admis) de "rejet
  historique" (pas un dédup stricte).

#### Étape 4 : Corriger la calibration

- Pool de seuils borné par feature, par famille, par type.
- Vérifier que `5.0` n'apparaît jamais pour `parabolic_sar` (qui est un
  prix).

#### Étape 5 : Itérer cross-TF

- Modifier `handle_run` pour itérer sur les 5 TF par défaut (5m, 15m,
  1h, 4h, 1d) au lieu d'un seul.
- Conserver `--data-timeframe` pour override manuel.

#### Étape 6 : Vérifier la stabilité

- Re-lancer la suite de tests (220 tests actuels + 50 nouveaux pour
  TypeGP refondu).
- Lancer une campagne de smoke test sur BTCUSD 1h avec `n_eval=200`,
  vérifier la diversité des Einhers produits.

#### Étape 7 : Reconstituer le corpus

- Vider `corpus.jsonl` (3 Einhers historiques douteux).
- Vider `archive.jsonl` (rejets sur DSR pré-fix).
- Re-purger `data_versions.jsonl` si nécessaire.
- Lancer une campagne propre : `discovery run --all-timeframes` sur
  BTCUSD 15m et 1h pour produire les premiers Einhers post-refonte.

#### Étape 8 : Étalonner les seuils

- Avec 30+ Einhers admis, recalibrer les seuils (DSR, PBO, CI,
  croissance, DD) en se basant sur la distribution observée.

### 20.2 Justification de l'ordre

1. **Purger d'abord** : on ne peut pas refactorer ce qu'on ne comprend
   pas. Réduire la surface aide à voir clair.
2. **TypeGP ensuite** : c'est le coeur de la recherche. Si TypeGP
   converge mal, aucune admission ne produira de la diversité.
3. **Admission ensuite** : les critères d'admission doivent être
   compatibles avec les hypothèses produites.
4. **Calibration** : pour des seuils cohérents.
5. **Itération TF** : pour explorer l'espace.
6. **Stabilité** : pour valider la refonte.
7. **Reconstitution du corpus** : à partir de zéro, sur des données
   saines.
8. **Étalonnage** : avec un corpus de référence.

### 20.3 Estimation de l'effort

- Étape 1 (purge) : 1-2 jours.
- Étape 2 (TypeGP) : 1-2 semaines (refonte substantielle, tests).
- Étape 3 (admission) : 2-3 jours.
- Étape 4 (calibration) : 1-2 jours.
- Étape 5 (TF) : 0.5 jour.
- Étape 6 (tests) : 1 semaine (avec campagnes de smoke).
- Étape 7 (corpus) : 1 jour (mécanique).
- Étape 8 (étalonnage) : 1-2 semaines (campagnes réelles).

**Total** : 5-7 semaines.

---

## 21. Ordre de correction recommandé

L'utilisateur a demandé un ordre par dépendance. Voici l'ordre corrigé
sur la base de mon analyse :

1. **Purger le code obsolète** (étape 1) — pour réduire la surface cognitive.
2. **Refondre TypeGP** (étape 2) — corriger la convergence, le bloat, la
   diversité.
3. **Découpler le budget** (étape 3 partielle : `n_eval` vs `population_size`)
   — pour que le moteur ait assez de population par défaut.
4. **Étalonner l'admission** (étape 8 + 3 conditionnels) — pour que les
   critères ne soient pas absurdes.
5. **Calibrer les seuils par feature** (étape 4) — pour des seuils
   informatifs.
6. **Itérer cross-TF** (étape 5) — pour explorer l'espace TF.
7. **Vérifier la stabilité** (étape 6) — campagnes + tests.
8. **Reconstituer le corpus** (étape 7) — à partir de zéro.

**Cet ordre diffère de la proposition initiale de l'utilisateur** : je
ne commence pas par "Corriger les contrats fondamentaux" car les contrats
actuels sont sains (pas de look-ahead, métriques standard). Le problème
principal est dans la génération, pas dans les contrats.

---

## 22. Risques résiduels

### 22.1 Risques de la refonte

1. **Perte de déterminisme** : réécrire TypeGP peut introduire des
   non-déterminismes. Il faut conserver `random.Random(seed)` partout.
2. **Perte de compatibilité corpus** : le fingerprint structurel va
   probablement changer. Les 3 Einhers historiques devront être ré-évalués.
3. **Sur-optimisation de la fitness** : si la speciation est mal
   implémentée, on peut converger vers des "espèces" non-informatives.
4. **Coût CPU** : avec population=50 et 20 générations, on a 1000
   évaluations. À ~2.7s/calibration, c'est 45 min par run. Acceptable.

### 22.2 Risques structurels

1. **Le marché BTC est peut-être réellement non-tradeable** : il est
   possible qu'après refonte, on n'ait toujours aucun Einher admissible
   parce que le marché ne produit pas de signal stable.
2. **Les 218 features sont peut-être insuffisantes** : si on n'arrive
   pas à produire de la diversité même avec 218 features, c'est un
   problème de features, pas de moteur.
3. **Le moteur a peut-être un bug d'ATR** : si l'ATR local est mal
   calculé (Wilder sur log-returns ?), les SL/TP sont aberrants.
4. **Le marché est peut-être dominé par 1 régime** : si BTC est en
   range depuis 2 ans, les Shorts sont favorisés par construction.

### 22.3 Risques de sur-interprétation

- **Le "3 Einhers BTCUSD" n'est PAS un résultat** : c'est un artefact
  historique. Ne pas chercher à le reproduire.
- **Le "tout Short" n'est PAS un signal de marché** : c'est un artefact
  du couple (évolution + DSR pré-fix + tasting). Ne pas conclure
  "BTC favorise Short".
- **Le "1 TF" n'est PAS un choix** : c'est un défaut d'itération du
  pipeline. Ne pas conclure "BTC n'a qu'un seul TF tradeable".

---

## 23. Conclusion

### 23.1 Réponse aux 4 questions

#### Q1 : Pourquoi TypeGP produit-il si peu d'hypothèses utiles ?

**Cause structurelle** : `n_eval_budget=20` par défaut en refine/admit
écrase la population TypeGP à 2 individus. Avec 2 individus et 9
générations, on produit ~20 hypothèses, dont la plupart ont une fitness
`-inf` (porte dure `n_signals < 30`). L'évolution n'a aucun signal pour
apprendre.

**Cause secondaire** : la fitness est `CAGR` mais avec une porte dure
`-inf` quasi-systématique, le tournoi est aléatoire parmi les `-inf`. Pas
d'évolution réelle.

**Cause tertiaire** : pas de mécanisme de diversité. L'élitisme=2 fige
les 2 mêmes individus. La dédup finale est trop tardive.

#### Q2 : Pourquoi les Einhers survivants sont-ils presque identiques, Short, et concentrés sur 1 TF ?

**Identiques** : convergence ultra-rapide (population=2 + elitisme=2 +
crossover conservateur + fitness -inf). L'individu 45 survit à toutes les
générations par hasard.

**Short** : (1) échantillon trop petit (3) ; (2) le DSR pré-fix
permettait des admissions avec peu de trades et fort Sharpe, ce qui
favorise les Shorts sur BTC en range ; (3) le tasting peut tomber sur
une période baissière.

**1 TF** : défaut d'itération du pipeline (`handle_run` ne parcourt
qu'un seul TF par défaut). Pas un choix, une omission.

#### Q3 : Où les hypothèses intéressantes sont-elles perdues ?

1. **Dans TypeGP** : élitisme=2, population=2, pas de diversité →
   convergence vers 1-2 individus.
2. **Dans la calibration** : seuils tirés sans bornes → fitness non
   informative.
3. **Dans le DSR** : post-fix, le DSR rejette tout (n_trials=20,
   p_value strict).
4. **Dans l'archive** : dédup stricte, memoryless-fail permanent.
5. **Dans le tasting** : l'évolution tourne sur 400 bougies, l'admission
   sur val complet → désalignement possible.
6. **Dans le comparator** : avec 1 seul generator, le score est 0.5 par
   défaut, le winner est élu par défaut.

#### Q4 : Que faut-il conserver, corriger, isoler, supprimer ?

- **Conserver** : moteur d'évaluation, fingerprint, corpus, archive,
  data loaders, holdout, types, métriques (DSR, PBO, bootstrap, max DD).
- **Corriger** : TypeGP (population, bloat, diversité), admission
  (DSR/PBO conditionnels, dédup graduelle), calibration (seuils par
  feature), itération TF.
- **Isoler** : baselines (debug uniquement), comparator (obsolète tant
  qu'il n'y a qu'un générateur), refine (optionnel).
- **Supprimer** : BNF, Random, GE, Memetic, NSGA2, Beam, comparator,
  selector, refinement/beam, `_old/`, `v2/`, `load_ohlcv_from_npy`.

### 23.2 Verdict final

**Le moteur est techniquement correct, mais le générateur est
structurellement défaillant.** TypeGP n'est pas un algorithme de
recherche, c'est un raffinement local dégénéré. La refonte prioritaire
doit porter sur TypeGP : population minimale, contrôle de complexité,
mécanisme de diversité, restart, direction imposée.

Le pipeline actuel produit 0 Einhers en post-fix DSR, ce qui est sain en
un sens (les anciens Einhers auraient dû être rejetés) mais révèle que
le moteur ne sait pas trouver de bonnes hypothèses.

**Le prochain chantier doit reconstruire TypeGP de zéro**, pas le
patcher. Les 13 000 lignes du repo actuel peuvent être réduites à
3 000-4 000 lignes centrées sur un TypeGP refondu.

### 23.3 Prochaine étape (pour l'utilisateur)

1. Valider l'ordre de refonte proposé.
2. Décider si on garde la mémoire des 3 Einhers historiques (probable
   "non", car suspects) ou si on les archive séparément.
3. Décider du budget cible (population=50, n_eval=200, n_gen=20 → ~4h/run
   sur machine lente).
4. Lancer la purge du code obsolète (étape 1) en premier.

---

**FIN DE L'AUDIT**
