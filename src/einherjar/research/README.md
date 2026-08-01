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

## Architecture

Voir `ONTOLOGY.md` (contrat conceptuel) et `ALGORITHME_RESEARCH.md` (étude comparative).

Arborescence :

```
research/
├── discovery.py              # CLI entry point
├── config/                   # Configs (thresholds, splits, costs, features)
├── data/                     # Interfaces OHLCV + features + corpus
├── utils/                    # Logging, fingerprint, types, stats
├── engine/                   # Step 0 — moteur d'évaluation
├── baselines/                # Step 1 — baselines (1 fichier pour les 3)
├── generators/               # Step 2 — compétition (1 fichier pour tous)
├── selection/                # Step 3 — sélection du générateur
├── refinement/               # Step 4 — raffinement beam
├── admission/                # Step 5 — admission (1 fichier pour les 7 critères)
├── holdout/                  # Step 6 — holdout sacré
├── archive/                  # Mémoire scientifique (rejets)
├── v2/                       # Candidats V2 (NSGA-II, memetic, MAP-Elites, full CPCV + LLM stub)
└── tests/                    # Tests unitaires (anti-leak, DSR/PBO, bootstrap, fingerprint)
```

## Philosophie

1. **Moteur d'évaluation d'abord** (Step 0). Sans lui, aucune comparaison n'a de sens.
2. **Baselines avant générateurs** (Step 1). Pas de sophistication sans plancher de performance.
3. **Choix empirique** (Step 2-3). Pas de "GE figé comme moteur principal" sans benchmark.
4. **SL/TP figés depuis le train** (jamais recalibrés sur val/holdout/live).
5. **Holdout sacré** (Step 6) — consulté une seule fois à la fin.
6. **Diversité comportementale**, pas seulement structurelle.
7. **Archive append-only**, réévaluable sur nouveau `data_version`.

## Données

- 218 features utilisables (28 exclues) — voir `config/features_taxonomy.json`.
- Splits train/val/holdout 60/20/20 + purging + embargo.
- Coûts simulés (spread, commission, slippage) — voir `config/costs.yaml`.
- Seuils d'admission — voir `config/thresholds.yaml`.

## Tests

```bash
pytest src/einherjar/research/tests/
```

Tests critiques :
- `test_engine_no_leak.py` — vérifie qu'aucun paramètre du val/holdout n'est utilisé sur le train.
- `test_dsr_pbo.py` — vérifie les calculs DSR/PBO.
- `test_bootstrap.py` — vérifie le block bootstrap CI.
- `test_fingerprint.py` — vérifie la stabilité et l'anti-collision du fingerprint canonique.

## Statut

- **Moteur d'évaluation** : pas encore implémenté (priorité 0).
- **Baselines** : pas encore implémentés.
- **Générateurs** : pas encore implémentés.
- **Admission** : pas encore implémentée.
- **Holdout** : pas encore implémenté.

Tous les handlers de `discovery.py` lèvent `NotImplementedError` tant que les modules ne sont pas remplis.
