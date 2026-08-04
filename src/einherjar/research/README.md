# Discovery — moteur de recherche d'Einherjar

CLI unique pour piloter le pipeline 7 étapes du moteur de découverte. Un seul
point d'entrée : `python -m einherjar.research.discovery`.

---

## Prérequis

- Python 3.11+
- venv activé : `$env:PYTHONPATH='src'; $env:PYTHONIOENCODING='utf-8'`
- Données MIDAS V3 compilées à `D:\midas_v2\midasV3\src\data\compiled\`
  (override via `--data-root`)
- Le module `duckdb` doit être installé dans le venv (utilisé par le moteur)

---

## Modes (étapes du pipeline)

| Mode | Étape | Rôle |
|---|---|---|
| `engine` | 0 | Construit/vérifie le moteur d'évaluation. Smoke test : à lancer en 1er pour valider l'environnement. |
| `baselines` | 1 | Évalue 3 baselines (Human, Shallow, Random) avec admission réelle (7 critères S-3.4). Sert à calibrer les seuils d'admission. |
| `compare` | 2 | Compare les 6 générateurs (Random, Beam, TypedGP, GE, Memetic, NSGA-II) sur le même train/val et retourne un ranking multi-objectif. |
| `select` | 3 | Installe le générateur gagnant dans `outputs/selection.json`. Si la sélection existe déjà, la recharge. |
| `refine` | 4 | Raffine les top-N hypothèses du générateur sélectionné via BeamRefiner. |
| `admit` | 5 | Applique `AdmissionDecider` (DSR + PBO + bootstrap CI + n_trades + cross_asset + max_dd + diversité + dédup + quota). Admis → corpus. Rejets → archive. |
| `holdout` | 6 | Évaluation finale unique sur le holdout sacré. **1 seule fois par session.** |
| `run` | 0→5 | Enchaîne engine → baselines → compare → select → refine → admit. Le holdout reste manuel. |
| `pipeline` | 0→5 | Alias de `run`. |

---

## Cas d'usage concrets

### 1. Smoke test (vérifier que le moteur tourne)

```powershell
python -m einherjar.research.discovery engine --data-asset BTCUSD --data-timeframe 1h
```

Aucune hypothèse générée. Valide juste que :
- le moteur d'évaluation s'instancie
- les données OHLCV chargent
- le data_version est calculé

### 2. Single asset / single timeframe

```powershell
python -m einherjar.research.discovery run `
  --data-asset BTCUSD `
  --data-timeframe 1h `
  --n-eval 200
```

Charge **un seul actif** (`BTCUSD`) sur **un seul timeframe** (`1h`).
Effectue le pipeline complet sauf holdout. Budget de 200 évaluations.

Variante (étape par étape) :

```powershell
python -m einherjar.research.discovery baselines --data-asset BTCUSD --data-timeframe 1h
python -m einherjar.research.discovery compare --data-asset BTCUSD --data-timeframe 1h
python -m einherjar.research.discovery select  --data-asset BTCUSD --data-timeframe 1h
python -m einherjar.research.discovery refine  --data-asset BTCUSD --data-timeframe 1h --n-eval 50
python -m einherjar.research.discovery admit   --data-asset BTCUSD --data-timeframe 1h --n-eval 50
```

### 3. Multi-actifs (--data-assets)

```powershell
python -m einherjar.research.discovery run `
  --data-assets BTCUSD,ETHUSD,SOLUSD `
  --data-timeframe 1h `
  --n-eval 200
```

Charge **plusieurs actifs** via `--data-assets` (séparés par virgules).
L'actif principal est le 1er chargé (utilisé par les générateurs non-NSGA-II).
NSGA-II exploite tous les actifs via `_evaluate_multi_asset` :
médiane des Sharpe par actif, contrainte #4 multi-actifs effective.

Si tu passes à la fois `--data-asset` et `--data-assets`, **c'est `--data-assets` qui gagne**.

### 4. Choisir un moteur sans passer par la comparaison

Si tu sais déjà quel générateur tu veux utiliser, tu peux court-circuiter
`compare` et `select` en passant `--generator=...` directement à `admit`
ou `holdout`. Le générateur est instancié depuis son nom (alias ou nom de classe).

```powershell
# Admettre des hypothèses générées par NSGA-II directement
python -m einherjar.research.discovery admit `
  --generator nsga2 `
  --data-asset BTCUSD `
  --data-timeframe 1h `
  --n-eval 100
```

Générateurs disponibles (alias) :
| Alias | Classe |
|---|---|
| `random` | `RandomSearchGenerator` |
| `beam` | `BeamSearchGenerator` |
| `stgp` | `TypedGPGenerator` |
| `ge` | `GrammaticalEvolutionGenerator` |
| `memetic` | `MemeticGenerator` |
| `nsga2` | `NSGA2Generator` |

> **Note** : sans `compare` préalable, le `select` ne peut pas recharger
> une sélection existante. Le générateur est instancié à la volée par
> `GeneratorSelector.instantiate`. Utile pour itérer sur un générateur
> spécifique sans repasser par la comparaison complète.

### 5. Run multi-actifs avec NSGA-II seulement

```powershell
python -m einherjar.research.discovery run `
  --data-assets BTCUSD,ETHUSD,SOLUSD `
  --data-timeframe 1h `
  --generator nsga2 `
  --n-eval 200
```

`--generator` filtre l'instance : seul NSGA-II est créé, ce qui économise
du temps si tu veux comparer NSGA-II à un autre setup sans relancer tous
les générateurs.

### 6. Holdout sacré (à déclencher manuellement)

```powershell
# D'abord, il faut un Einher admis (via admit).
# Son JSON sérialisé sert d'input au holdout.
python -m einherjar.research.discovery holdout `
  --hypothesis-file outputs/some_admitted_hyp.json
```

Le holdout est évalué **une seule fois par session**. Une 2e tentative
lèvera une erreur (`HoldoutEvaluator._holdout_used`).

---

## Arguments de la CLI

| Argument | Défaut | Effet |
|---|---|---|
| `--config` | `./config` | Dossier de configuration (thresholds, splits, costs, taxonomy). |
| `--data-version` | None | Tag du data_version à utiliser (override config). |
| `--seed` | 42 | Seed RNG maître (reproductibilité). |
| `--generator` | None | Filtre le générateur à utiliser (`random`, `beam`, `stgp`, `ge`, `memetic`, `nsga2`). |
| `--n-eval` | 200 | Budget d'évaluations (compare, baselines, refine, admit). |
| `--n-samples` | 200 | Nombre d'hypothèses par baseline random. |
| `--selection-path` | `outputs/selection.json` | Fichier de persistance de la sélection. |
| `--log-level` | INFO | `DEBUG`, `INFO`, `WARNING`, `ERROR`. |
| `--data-root` | `D:\midas_v2\midasV3\src\data\compiled` | Racine des `.npy` MIDAS V3. |
| `--data-asset` | `BTCUSD` | Actif unique (ex: `BTCUSD`, `ETHUSD`). |
| `--data-assets` | None | Liste d'actifs séparés par virgules (ex: `BTCUSD,ETHUSD,SOLUSD`). Priorité sur `--data-asset`. |
| `--data-class` | `crypto` | `crypto`, `forex`, `indices`, `commodities`, `stocks_growth`, `stocks_tech`, `stocks_value`. |
| `--data-timeframe` | `1h` | `1m`, `5m`, `15m`, `1h`, `4h`, `1d`, etc. |
| `--dry-run` | False | Affiche le plan d'exécution sans rien lancer. |
| `--hypothesis-file` | None | Fichier JSON d'Hypothesis (utilisé par `holdout`). |

---

## Comportements automatiques (à connaître)

- **P0-05** : `--data-assets` active le mode multi-actifs. NSGA-II utilise
  la médiane des Sharpe par actif (contrainte #4). Les autres générateurs
  restent sur l'actif principal (1er de la liste).

- **P1-10** : NSGA-II charge automatiquement le corpus (`CorpusStore`) et
  injecte les `feature_set` dans `_corpus_feature_sets` pour le calcul
  Jaccard vs corpus (objectif #3 diversité).

- **P0-03** : `DataVersionStore` (append-only JSONL) verrouille le
  `data_version` au début de chaque handler. Si un run utilise un
  data_version non encore persisté, il est appendé avec `fsync`. Sinon,
  le tag existant est réutilisé (lock).

- **P1-08** : `n_eval_budget` est partagé entre tous les générateurs.
  Quand le compteur global atteint le budget, les générateurs suivants
  sont skippés (avec warning).

- **Holdout** : sacré, 1 seule évaluation par session, atomique
  (`HoldoutLedger` avec `fsync`).

- **Échecs** : pas de fallback silencieux (P0 #7). Toute erreur
  d'évaluation lève explicitement (ValueError, NpyRealLoaderError,
  AdmissionError, etc.).

---

## Tests

```powershell
cd D:\midas_v2\Einherjar
$env:PYTHONPATH='src'
$env:PYTHONIOENCODING='utf-8'
python -m unittest discover -s src/einherjar/research/tests -p 'test_*.py'
```

225 tests verts, 1 skip légitime (test qui requiert les `.npy` MIDAS réels).

---

## Fichiers de sortie

- `outputs/selection.json` : générateur sélectionné par `select`.
- `outputs/refined.json` : top-N hypothèses raffinées par `refine`.
- `outputs/admit_summary.json` : synthèse d'admission (admis/rejets).
- `outputs/holdout_result.json` : résultat du holdout sacré.
- `outputs/data_versions.jsonl` : append-only des `DataVersion` (P0-03).
- `outputs/archive/archive.jsonl` : append-only des Einhers rejetés.
- `data/corpus.jsonl` : append-only des Einhers admis (corpus).

---

## Pour aller plus loin

- `STATUS.md` à la racine : état détaillé du moteur (socle + BNF + review critique).
- `src/einherjar/research/ALGORITHME_RESEARCH.md` : algorithmes des générateurs.
- `src/einherjar/research/ONTOLOGY.md` : concepts (Einhers, MesuresBrutes, etc.).
- `src/einherjar/research/config/` : configuration (thresholds, taxonomy, splits).
