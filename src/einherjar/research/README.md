# Einherjar — Moteur de découverte (research/)

Ce sous-package implémente le **moteur de découverte d'Einhers** : un système qui explore automatiquement des règles de trading, les valide, et les archive dans un corpus exploitable.

## Point d'entrée

```bash
python -m einherjar.research.discovery <mode> [options]
```

Modes disponibles (voir `discovery.py` pour le détail) :

| Mode       | Étape | Description                                                |
|------------|-------|------------------------------------------------------------|
| `engine`   | 0     | Construit / vérifie le moteur d'évaluation (priorité 0)    |
| `baselines`| 1     | 3 baselines honnêtes (human + shallow + random)            |
| `compare`  | 2     | Comparaison reproductible des générateurs (random/GE/GP/beam) |
| `select`   | 3     | Sélection du générateur gagnant                            |
| `refine`   | 4     | Raffinement beam local (sans recalibrer SL/TP)              |
| `admit`    | 5     | Admission au corpus (DSR + PBO + bootstrap CI + diversité) |
| `holdout`  | 6     | Évaluation finale unique sur le holdout (sacré)             |
| `run`      | 0→5   | Pipeline complet (sans le holdout)                          |
| `pipeline` | 0→5   | Alias de `run`                                              |

Options communes :
- `--config <path>` : dossier contenant les fichiers de config (défaut: `config`)
- `--data-version <tag>` : identifiant de version de données (défaut: `v1`)
- `--seed <int>` : seed RNG maître (défaut: 42)
- `--n-eval <int>` : budget d'évaluations (pour compare)
- `--dry-run` : affiche le plan sans rien lancer

## Architecture

Voir `ONTOLOGY.md` (contrat conceptuel) et `ALGORITHME_RESEARCH.md` (étude comparative).

### Pipeline 7 étapes

```
[Step 0] engine     [Step 1] baselines   [Step 2] compare    [Step 3] select
                                                                    
[Step 4] refine     [Step 5] admit       [Step 6] holdout (sacré)
```

### Arborescence

```
research/
├── discovery.py              # CLI entry point
├── README.md                 # ce fichier
│
├── config/                   # Configs (thresholds, splits, costs, features)
│   ├── features_taxonomy.json    # 218 features utilisables (28 exclues)
│   ├── thresholds.yaml           # S-3.4 : DSR, PBO, bootstrap CI, n_trades, dd, diversité
│   ├── splits.yaml               # train/val/holdout + purging + embargo
│   ├── costs.yaml                # spread, commission, slippage
│   ├── evaluation.yaml           # ATR, N, simulation intrabar, block bootstrap
│   └── loader.py                 # Chargement + validation
│
├── data/                     # Interfaces données
│   ├── ohlcv.py                  # Loader OHLCV (cache, validation, backend injectable)
│   ├── features.py               # Calcul features (FeatureEngine, filtre 218)
│   ├── versioning.py             # data_version (hash reproductible)
│   └── corpus.py                 # Lecture/écriture corpus actif
│
├── utils/                    # Utilitaires transverses
│   ├── logging.py                # Logging structuré (1 fichier/run)
│   ├── time.py                   # Splits + purging + embargo
│   ├── stats.py                  # Block bootstrap, percentiles, ATR(14)
│   ├── fingerprint.py            # Fingerprint canonique (structurel + comportemental)
│   ├── metrics.py                # Sharpe, Sortino, MAR, MDD, CAGR, DSR
│   └── types.py                  # Hypothesis, Einher, MesuresBrutes, TradeMesure
│
├── engine/                   # Step 0 — MOTEUR D'ÉVALUATION (priorité 0)
│   ├── evaluator.py              # train_calibrate + test_on + evaluate
│   ├── simulator.py              # Simulation intrabar TP/SL
│   └── bootstrap.py              # Block bootstrap CI
│
├── baselines/                # Step 1 — Baselines (3 dans 1 fichier)
│   ├── algorithms.py             # HumanRules + ShallowEnumeration + RandomConstrained
│   └── runner.py                 # Lance les 3, produit distribution Sharpe
│
├── generators/               # Step 2 — Compétition reproductible
│   ├── algorithms.py             # 5 candidats : Random, Beam, TypedGP, GE, Memetic, NSGA2
│   ├── protocol.py               # Protocole figé (seed, budget, splits, max_conditions)
│   └── comparator.py             # Compare, classe, retourne le gagnant
│
├── selection/                # Step 3 — Sélection du générateur
│   └── selector.py               # Installe le gagnant, persiste pour runs suivants
│
├── refinement/               # Step 4 — Raffinement beam
│   └── beam.py                   # Beam search local (sans recalibrer SL/TP)
│
├── admission/                # Step 5 — Admission au corpus
│   ├── criteria.py               # 7 critères S-3.4 (UN fichier)
│   ├── diversity.py              # Descripteurs comportementaux + quotas structurels
│   └── decision.py               # Combine critères + diversité + dédup fingerprint
│
├── holdout/                  # Step 6 — Holdout sacré
│   └── evaluator.py              # Évaluation finale unique (1 seule passe)
│
├── archive/                  # Mémoire scientifique (rejets)
│   ├── reasons.py                # Catalogue normalisé (13 raisons)
│   ├── schema.py                 # ArchiveEntry + validation
│   └── store.py                  # I/O append-only + dédup fingerprint
│
├── v2/                       # Candidats V2 (PAS actifs en V1, mais testables)
│   └── (LLM, NSGA2 complet, memetic complet, MAP-Elites, full CPCV)
│
└── tests/                    # Tests
    ├── test_engine_no_leak.py    # Non-régression + anti-leak
    ├── test_evaluator_smoke.py   # Smoke test du moteur
    └── test_baselines_admission_generators.py
```

## Philosophie

1. **Moteur d'évaluation d'abord** (Step 0). Sans lui, aucune comparaison n'a de sens.
2. **Baselines avant générateurs** (Step 1). Pas de sophistication sans plancher de performance.
3. **Choix empirique** (Step 2-3). Pas de "GE figé comme moteur principal" sans benchmark reproductible.
4. **SL/TP figés depuis le train** (jamais recalibrés sur val/holdout/live).
5. **Holdout sacré** (Step 6) — consulté une seule fois à la fin.
6. **Diversité comportementale**, pas seulement structurelle (I-8).
7. **Archive append-only**, réévaluable sur nouveau `data_version`.
8. **Décisions empiriques, pas idéologiques** (cf. critique IA tierce, ALGORITHME_RESEARCH.md § 13).

## Données

- **218 features utilisables** (28 exclues : 19 fantômes, 8 meta-factors, 1 alias) — voir `config/features_taxonomy.json`.
- **Splits train/val/holdout 60/20/20** + purging + embargo.
- **Coûts simulés** (spread, commission, slippage) — voir `config/costs.yaml`.
- **Seuils d'admission** — voir `config/thresholds.yaml`.

## Tests

```bash
# Depuis la racine du projet
cd D:\midas_v2\Einherjar
$env:PYTHONPATH = 'src'
python -m unittest discover -s src/einherjar/research/tests -p 'test_*.py'
```

Tests critiques :
- `test_engine_no_leak.py` : déterminisme, anti-leak, splits disjoints, fingerprint stable.
- `test_evaluator_smoke.py` : pipeline complet (calibrate + test_on + holdout une seule fois).
- `test_baselines_admission_generators.py` : 3 modules, 18 tests.

## Exemples d'usage

```bash
# Vérifier que la config charge
python -m einherjar.research.discovery engine --dry-run

# Lancer un pipeline complet (étapes 0 à 5)
python -m einherjar.research.discovery run --data-version v1

# Lancer juste les baselines
python -m einherjar.research.discovery baselines --n-samples 500

# Comparer les générateurs
python -m einherjar.research.discovery compare --n-eval 1000

# Sélectionner le gagnant
python -m einherjar.research.discovery select --selection-path outputs/selection.json

# Holdout (à déclencher manuellement UNE SEULE FOIS par Einher final)
python -m einherjar.research.discovery holdout --data-version v1
```

## Statut

| Module | Statut | Notes |
|---|---|---|
| `engine/` | ✅ Codé | Calibration + test_on + holdout unique |
| `data/ohlcv.py` | ✅ Codé | Provider avec cache, backend injectable |
| `data/features.py` | ✅ Codé | Provider avec filtre 28 exclues, cache |
| `baselines/` | ✅ Codé | 3 baselines, distribution Sharpe |
| `generators/` | ✅ Codé (5/6) | GE en placeholder (BNF à écrire) |
| `admission/` | ✅ Codé | 7 critères + diversité + decision |
| `refinement/` | ✅ Codé | Beam search local sans recalibrage SL/TP |
| `selection/` | ✅ Codé | Lecture/écriture de la sélection |
| `holdout/` | ✅ Codé | Évaluation unique + flag dégradation |
| `archive/` | ✅ Codé | Schéma + catalogue + I/O append-only |
| `v2/` | 🟡 Skeleton | Candidats V2 à brancher |
| `discovery.py` | ✅ Codé | 9 modes wired |

## TODO restants (V1 → V2)

1. **Écrire la grammaire BNF** complète (cf. ALGORITHME_RESEARCH.md § 11.5)
2. **Brancher `refinement` sur le corpus** (V1 : nécessite un Einher viable en input)
3. **Brancher `admit` end-to-end** (V1 : stub, à connecter au runner)
4. **Corriger les warnings ruff D102/D107/E501** (mineurs, non bloquants)
5. **Remplacer les stubs** (NSGA2, Memetic complet) par implémentations complètes
6. **MAP-Elites** si diversité comportementale insuffisante en V1
7. **CPCV complet** (Lopez de Prado) au lieu du CPCV léger actuel
