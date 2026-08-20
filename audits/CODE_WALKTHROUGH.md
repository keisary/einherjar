# Einherjar xgb_einhers — Code Walkthrough Complet

**Date** : 2026-08-18
**Périmètre** : les 15 modules du système (16 fichiers, ~3000 lignes)
**Objectif** : expliquer le code dans les moindres détails, module par module, fonction par fonction

---

## Vue d'ensemble

Le système prend des données OHLCV + 213 features techniques MIDAS V3, entraîne un modèle XGBoost supervisé sur les rendements forward, extrait les chemins d'arbres, les convertit en règles de trading (Einhers), les backteste sur OHLCV brut avec SL/TP dynamiques, et les admet dans un corpus JSONL selon des critères statistiques stricts.

### Architecture des 16 modules

```
src/einherjar/research/xgb_einhers/
├── __init__.py              # Exports publics (vide actuellement)
├── types.py                 # 6 dataclasses frozen (LoadedData, Einher, Condition, etc.)
├── data_loader.py           # Charge X.npy, Y_*.npy, OHLCV.csv + aligne par timestamp
├── label_engineer.py        # Construit target Y_ret (signed return) + filtre bougies invalides
├── model.py                 # Double backend GBDT (xgboost + sklearn fallback) + GBDTConfig
├── path_extractor.py        # Parse dump XGBoost texte ou sklearn tree → XGBPath
├── condition_tree.py        # Convertit XGBPath → AST de conditions (AND-only)
├── einher_builder.py        # Construit Einher depuis XGBPath + helpers imutabilité
├── backtester.py            # NOUVEAU moteur backtest (intrabar, SL-first, P0-tested)
├── admission.py             # Critères admission + mapping feature→famille + BH check
├── einher_io.py             # Sérialisation JSONL (save/load Einhers)
├── runner.py                # CLI : orchestre tous les modules
├── feature_dedup.py         # Anti-duplication par matrice de corrélation
├── feature_filter.py        # Drop patterns sparses (pct_True < 0.5%)
├── multiple_testing.py      # Benjamini-Hochberg (correction multi-tests)
└── multi_asset_loader.py    # Concat N actifs (cross-asset)
```

---

## 1. `types.py` — Les dataclasses du système (174 lignes)

**Rôle** : définit tous les types de données échangés entre les modules. Toutes les dataclasses sont `@dataclass(frozen=True)` (immutables) pour garantir la reproductibilité.

### Les 6 dataclasses

#### `LoadedData` (lignes 18-42)
```python
@dataclass(frozen=True)
class LoadedData:
    """Données chargées depuis MIDAS V3 pour un (asset, TF)."""
    asset: str                           # ex 'BTCUSD'
    asset_class: str                     # ex 'crypto'
    timeframe: str                       # ex '1h'
    timestamps: np.ndarray               # (N,) int64 ms epoch
    X: np.ndarray                        # (N, 213) float32, OHLCV déjà exclues
    Y_dir: np.ndarray                    # (N, H) int8 {-100, 0, 1, 2}
    Y_ret: np.ndarray                    # (N, H) float32, signed
    Y_hor: np.ndarray                    # (N, H) float32, bars
    feature_names: tuple[str, ...]       # 213 noms
    horizons: tuple[str, ...]            # ex ('6h', '12h', '1d', '2d')
```

**Propriétés** : `n_samples`, `n_features`, `n_horizons`

**Pourquoi frozen** : empêche la mutation accidentelle entre modules. Si on doit "modifier", on recrée avec `dataclasses.replace()`.

#### `TrainValHoldoutSplit` (lignes 45-57)
Split temporel 60/20/20 avec embargo. Chaque split contient les indices originaux pour pouvoir remonter à l'alignement OHLCV.

#### `TradeResult` (lignes 65-77)
Résultat d'un trade simulé. Contient prix d'entrée/sortie, raison de sortie (`tp|sl|timeout`), rendement brut/net, nb de bougies détenues.

#### `EinherMetrics` (lignes 85-130)
Les 13 métriques d'un Einher + une méthode `passes_admission()` qui check les critères minimaux :

```python
def passes_admission(
    self,
    min_trades: int = 30,
    min_sharpe: float = 0.3,
    min_win_rate: float = 0.40,
    min_profit_factor: float = 1.0,
    max_drawdown: float = 0.30,
) -> tuple[bool, str | None]:
    """Vérifie les critères d'admission minimaux."""
    if self.n_trades < min_trades:
        return False, f"n_trades={self.n_trades} < {min_trades}"
    if self.sharpe_ratio < min_sharpe:
        return False, f"sharpe={self.sharpe_ratio:.3f} < {min_sharpe}"
    # ... 4 autres critères
    return True, None
```

**C'est la première couche de filtres** (métriques pures). La couche admission (Sprint 2.2+ et 3.1) ajoute diversité familles + holdout + BH.

#### `Condition` (lignes 138-147)
Une condition atomique : `feature < threshold`. Le `feature_ref` est le nom (pas l'index), pour la sérialisation JSON.

#### `ConditionNode` (lignes 150-161)
Un nœud d'AST : `op ∈ {AND, OR, NOT, XOR}` + left/right. AND-only actuellement (cf. `condition_tree.py`).

#### `Einher` (lignes 169-208)
L'objet central. Contient :
- `id` : identifiant unique (`xgb_BTCUSD_1h_2d_0000_0004_54a5b0`)
- `condition_tree` : AST des conditions
- `direction` : `BUY` ou `SELL`
- `amplitude_bars` : horizon (ex 48 pour 2d en 1h)
- `tp_pct`, `sl_pct` : en décimal (0.025 = 2.5%)
- `universe` : dict `{asset, asset_class, timeframe, horizon, horizon_bars}`
- `metrics` : `EinherMetrics`
- `scope` : `asset | general | market`
- `holdout_metrics` : `Optional[EinherMetrics]` (ajouté Sprint 2.4.1)

**Méthode `to_dict()`** : sérialise en dict JSON-compatible pour JSONL.

---

## 2. `data_loader.py` — Chargement et alignement (263 lignes)

**Rôle** : charge les données MIDAS V3 (.npy) + OHLCV bruts (.csv) + aligne par timestamp.

### Constantes (lignes 32-40)
```python
COMPILED_DIR = Path("D:/midas_v2/midasV3/src/data/compiled")
OHLCV_DIR = Path("D:/midas_v2/technical_agent_dataset_brut")
TAXONOMY_PATH = Path("D:/midas_v2/Einherjar/src/einherjar/research/config/features_taxonomy.json")
OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")
```

### `load_usable_feature_names()` (lignes 48-56)
Charge la taxonomie et retourne les features **non exclues** (set). Appelé par `load_xy()`.

### `load_xy(asset, tf, asset_class)` (lignes 64-136) — FONCTION CLÉ

Charge 5 fichiers .npy :
- `{asset}_ts.npy` : timestamps ms
- `{asset}_X.npy` : features brutes (246 colonnes = 5 OHLCV + 241 features)
- `{asset}_Y_dir.npy`, `_Y_ret.npy`, `_Y_hor.npy` : labels pour H horizons

Puis **filtre** :
1. **Exclut les 5 OHLCV** de X (réponse Q6) → 241 features
2. **Exclut les features marquées `excluded=True` dans la taxonomie** → 213 features
3. **Vérifie qu'il n'y a pas de NaN/Inf** (assert)

**Retourne** : `LoadedData` avec 213 features.

```python
# Code critique : exclusion OHLCV + taxonomie
ohlcv_idx = [i for i, n in enumerate(all_feature_names) if n in OHLCV_COLUMNS]
keep_idx = [i for i in range(len(all_feature_names)) if i not in ohlcv_idx]
X = X_raw[:, keep_idx]
feature_names = tuple(n for i, n in enumerate(all_feature_names) if i in keep_idx)
usable = load_usable_feature_names()
final_idx = [i for i, n in enumerate(feature_names) if n in usable]
X = X[:, final_idx]
```

### `load_ohlcv(asset, tf, asset_class)` (lignes 144-191)
Charge les CSV annuels OHLCV depuis `technical_agent_dataset_brut`. Concatène avec polars, trie par timestamp, force le fuseau UTC.

### `align_xy_with_ohlcv(loaded, ohlcv_df)` (lignes 194-234) — FONCTION CLÉ

**Inner join** sur timestamp entre X/Y et OHLCV. C'est CRITIQUE pour le backtest car X (features) et OHLCV (prix) doivent être alignés à la même bougie.

```python
# Code critique : inner join + réindexation
ts_dt = pl.from_numpy(loaded.timestamps.astype(np.int64), schema=["ts_ms"])
ts_dt = ts_dt.with_columns(
    pl.from_epoch(pl.col("ts_ms"), time_unit="ms")
    .dt.replace_time_zone("UTC").dt.cast_time_unit("us").alias("timestamp")
)
ohlcv_with_idx = ohlcv_df.with_row_index(name="ohlcv_idx")
joined = ohlcv_with_idx.join(ts_dt, on="timestamp", how="inner").sort("timestamp")
orig_idx = joined["orig_idx"].to_numpy()
X_aligned = loaded.X[orig_idx]
```

**Retourne** : `(X_aligned, ohlcv_aligned, ts_aligned)` de même longueur.

### `get_target_for_horizon(loaded, horizon_idx)` (lignes 242-257)
Extrait la colonne d'un horizon spécifique : `Y_ret[:, h]`, `Y_dir[:, h]`, `Y_hor[:, h]`.

### `temporal_split(X, y, ratios, embargo, horizon_bars)` (lignes 265-319) — FONCTION CLÉ

Split temporel 60/20/20 avec embargo. **Sprint 3.0 fix** : l'embargo est `max(embargo_bars, horizon_bars)` pour éviter le leakage du target.

```python
# Code critique : embargo proportionnel
effective_embargo = max(embargo_bars, horizon_bars)
train_end = int(n * train_ratio)
val_start = train_end + effective_embargo
val_end = val_start + int(n * val_ratio)
holdout_start = val_end + effective_embargo
```

**Pourquoi cet embargo** : le label `Y_ret[t]` utilise la bougie `t+horizon_bars`. Si on split à `t=val_end`, le label de la dernière bougie de train utilise la bougie `t+horizon` qui tombe dans val. Embargo = on enlève ces bougies "contaminées".

---

## 3. `model.py` — Entraînement GBDT (177 lignes)

**Rôle** : entraîner un GBDT (xgboost primaire, sklearn fallback) sur les features et target.

### `GBDTConfig` (lignes 48-87)
Dataclass frozen des hyperparamètres. **Deux configs** :
- **Par défaut** : `n_estimators=100, max_depth=4, min_child_weight=10` (peu régularisé)
- **`regularized()`** : `n_estimators=200, max_depth=3, min_child_weight=50, reg_alpha=1, reg_lambda=5` (anti-overfit)

```python
@classmethod
def regularized(cls) -> "GBDTConfig":
    return cls(
        n_estimators=200,
        max_depth=3,           # arbres moins profonds
        learning_rate=0.05,
        subsample=0.7,         # bagging agressif
        colsample_bytree=0.6,  # sous-ensemble de features
        min_child_weight=50,   # feuilles plus grosses
        reg_alpha=1.0,         # L1 fort
        reg_lambda=5.0,        # L2 fort
        early_stopping_rounds=20,
    )
```

### `train_gbdt(X_train, y_train, X_val, y_val, config)` (lignes 95-111)
Fonction unifiée. Retourne `(model, backend_name)`.

### `_train_xgb()` (lignes 123-156)
```python
params = {
    "n_estimators": config.n_estimators,
    "max_depth": config.max_depth,
    "learning_rate": config.learning_rate,
    "subsample": config.subsample,
    "colsample_bytree": config.colsample_bytree,
    "min_child_weight": config.min_child_weight,
    "reg_alpha": config.reg_alpha,
    "reg_lambda": config.reg_lambda,
    "tree_method": "hist",
    "eval_metric": "rmse",
    "objective": "reg:squarederror",
}
model = _xgb.XGBRegressor(**params)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
```

### `feature_importance(model, backend, feature_names)` (lignes 198-214)
Retourne un dict `{feature_name: gain}` trié décroissant.

---

## 4. `path_extractor.py` — Extraction des arbres (232 lignes)

**Rôle** : extraire tous les chemins feuilles d'un modèle GBDT sous forme de `XGBPath`.

### `XGBPath` (lignes 21-27)
```python
@dataclass(frozen=True)
class XGBPath:
    conditions: tuple[tuple[str, str, float], ...]  # ((feature, op, threshold), ...)
    score: float                                    # valeur de la feuille
    tree_idx: int
    path_idx: int
```

### `parse_xgb_dump(dump_str)` (lignes 35-72)
Parse le dump texte XGBoost (format `0:[f5<70] yes=1,no=2,missing=1`).

**Regex** :
```python
node_re = re.compile(
    r"^(\d+):\[(.+?)\s*([<>=!]+)\s*([\-\d\.eE+]+)\]\s*yes=(\d+),no=(\d+),missing=(\d+)"
)
leaf_re = re.compile(r"^(\d+):leaf=([\-\d\.eE+]+)")
```

**Algorithme** :
1. Parse tous les nœuds dans un dict `{node_id: {type, feature, op, threshold, yes, no}}`
2. Trouve la racine (nœud qui n'est la cible d'aucun autre)
3. `_walk_xgb()` récursif : pour chaque nœud, ajoute la condition et descend à gauche/droite

### `_walk_xgb()` (lignes 75-111)
Récursion sur l'arbre. Pour chaque nœud interne :
- Branche "yes" : ajoute `(feat, op, threshold)`
- Branche "no" : ajoute la condition INVERSE (`<` → `>=`, `==` → `!=`)
- Récursion sur les deux branches
- Quand on arrive à une feuille : crée un `XGBPath`

```python
# Code critique : inversion des conditions sur la branche "no"
if op == "<":
    yes_cond = (feat, "<", threshold)
    no_cond = (feat, ">=", threshold)
elif op == ">=":
    yes_cond = (feat, ">=", threshold)
    no_cond = (feat, "<", threshold)
elif op == "==":
    yes_cond = (feat, "==", threshold)
    no_cond = (feat, "!=", threshold)
# ...
```

### `parse_sklearn_tree()` (lignes 119-148)
Équivalent pour sklearn. Accède aux attributs internes du `DecisionTreeRegressor.tree_` (structure C-level).

### `extract_paths(model, backend, feature_names, ...)` (lignes 193-224) — FONCTION CLÉ

**Pipeline complet** :
1. Selon le backend, appelle `_extract_xgb()` ou `_extract_sklearn()`
2. **Filtre** : `min_path_length <= len(conditions) <= max_path_length` et `min_score <= |score| <= max_score`
3. **Trie** par `|score|` décroissant
4. **Garde** les `max_paths` meilleurs

```python
# Code critique : filtrage
filtered = [
    p for p in all_paths
    if min_path_length <= len(p.conditions) <= max_path_length
    and min_score <= abs(p.score) <= max_score
]
filtered.sort(key=lambda p: abs(p.score), reverse=True)
result = filtered[:max_paths]
```

### `_name_features_in_dump()` (lignes 258-264)
Remplace les `f0`, `f1`, etc. par les vrais noms de features dans le dump XGBoost.

---

## 5. `condition_tree.py` — AST de conditions (123 lignes)

**Rôle** : convertir `XGBPath` en AST de conditions évaluables, et évaluer l'AST sur une matrice X.

### `path_to_ast(path)` (lignes 32-63) — FONCTION CLÉ
```python
def path_to_ast(path: XGBPath) -> Condition | ConditionNode:
    if len(path.conditions) == 0:
        raise ValueError("Chemin vide")
    conditions = [
        Condition(feature_ref=feat, operator=op, value=value)
        for feat, op, value in path.conditions
    ]
    if len(conditions) == 1:
        return conditions[0]
    # AND récursif (left-associative)
    result = conditions[0]
    for c in conditions[1:]:
        result = ConditionNode(op="AND", left=result, right=c)
    return result
```

**Pourquoi left-associative** : `(a AND b) AND c` plutôt que `a AND (b AND c)`. C'est équivalent logiquement mais plus naturel à lire.

### `evaluate_condition_on_value(ast, features_at_t)` (lignes 71-100)
Évalue un AST sur un dict de features au temps t. Supporte AND, OR, NOT, XOR (mais AND-only actuellement).

### `_eval_atomic(c, features_at_t)` (lignes 103-128)
Évalue une condition atomique. **Important** : retourne `False` si la feature est `None` ou `NaN` (conservateur).

```python
# Code critique : NaN check
if v != v:  # NaN check (NaN != NaN)
    return False
```

### `evaluate_ast_on_array(ast, X, feature_names)` (lignes 131-149) — FONCTION CLÉ
Évalue l'AST sur toute une matrice X. Retourne un mask `(N,)` bool.

**⚠️ Point critique de performance** : c'est une boucle Python `for i in range(n)`. Pour 70k bougies, c'est lent. À vectoriser en V2 (numpy.where sur les conditions atomiques).

---

## 6. `einher_builder.py` — Construction Einher (118 lignes)

**Rôle** : transformer un `XGBPath` en `Einher` (objet métier).

### `MIN_ABS_SCORE_FOR_DIRECTION = 0.003` (ligne 31)
Seuil : `|score| > 0.3%` pour qu'un chemin devienne BUY ou SELL. Sinon = None (skip).

### `build_einher_from_path(...)` (lignes 34-117) — FONCTION CLÉ

```python
# Direction depuis le signe du score
if path.score > min_abs_score:
    direction = "BUY"
elif path.score < -min_abs_score:
    direction = "SELL"
else:
    return None  # Signal trop faible

# AST de la condition
ast = path_to_ast(path)

# ID unique : xgb_BTCUSD_1h_2d_0000_0004_54a5b0
einher_id = f"xgb_{asset}_{timeframe}_{horizon_str}_{path.tree_idx:04d}_{path.path_idx:04d}_{uuid.uuid4().hex[:6]}"
```

**Note anti-tautologie** : `tp_pct=0, sl_pct=0` au moment de la création. Les vrais SL/TP (défault 2.5%/1.5%) sont calculés par le backtester et stockés via `set_einher_tp_sl()` (Sprint 2.1.1 fix).

### `set_einher_metrics()`, `set_einher_tp_sl()`, `set_einher_holdout_metrics()`
Helpers immutables (`dataclasses.replace()`).

---

## 7. `backtester.py` — Moteur de backtest (311 lignes)

**Rôle** : simuler des trades intrabar et calculer les métriques. C'est le **NOUVEAU** moteur qui remplace le buggy.

### `BacktestResult` (lignes 47-53)
```python
@dataclass
class BacktestResult:
    trades: list[TradeResult]
    metrics: EinherMetrics
    equity_curve: np.ndarray  # cumsum des net_returns
    effective_tp_pct: float = 0.0
    effective_sl_pct: float = 0.0
```

### `compute_atr(high, low, close, period=14)` (lignes 56-79)
ATR Wilder (Relative Moving Average) :
```python
# Code critique : Wilder smoothing (RMA)
atr[period - 1] = np.mean(tr[:period])  # SMA initiale
for i in range(period, n):
    atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period  # RMA
```

### `simulate_trade(...)` (lignes 95-155) — FONCTION CRITIQUE

**Convention** : entrée à `OPEN[t+1]` (bougie suivante), sortie dans `[t+1, t+amplitude]`.

**SL-first sur bougie ambiguë** (conservateur) :
```python
# Code critique : convention SL-first
if sl_hit and tp_hit:
    return sl_price, "sl", offset + 1
if sl_hit:
    return sl_price, "sl", offset + 1
if tp_hit:
    return tp_price, "tp", offset + 1
```

**Timeout** : exit à `OPEN[t+amplitude]` (bougie d'après la dernière).

### `compute_metrics(trades, buy_hold_return, years_in_period)` (lignes 158-233) — FONCTION CLÉ

Calcule les 13 métriques. **Sprint 3.0 fix** : le Sharpe est annualisé correctement.

```python
# Sprint 3.0 FIX #1 : Sharpe annualisé CORRECT
# AVANT (bug) : sharpe = avg_net / std * sqrt(n_trades)  # t-stat, pas annualisé
# APRES (fix) : sharpe = avg_net / std * sqrt(trades_per_year)
std = float(np.std(rets, ddof=1)) if n > 1 else 0.0
if std > 0 and years_in_period > 0:
    trades_per_year = n / years_in_period
    sharpe = float(avg_net / std * np.sqrt(trades_per_year))
```

**Max drawdown** :
```python
eq = np.cumsum(rets)
peak = np.maximum.accumulate(eq)
dd = eq - peak
max_dd = float(np.min(dd))
```

### `backtest_einher(...)` (lignes 236-362) — FONCTION PRINCIPALE

**Pipeline** :
1. Vérifie que `len(X) == len(ohlcv_df)` (cohérence)
2. **Évalue les conditions** : `signal_mask = evaluate_signals(einher, X, feature_names)`
3. **Filtre les signaux** : exclut ceux dont `signal_idx + amplitude_bars >= n` (débordement)
4. **SL/TP par défaut** : si `einher.tp_pct == 0`, utilise 2.5% / 1.5% (ATR)
5. **Simule chaque trade** : entrée `OPEN[t+1]`, sortie `simulate_trade()`
6. **Calcule buy_hold** : `(closes[-1] - closes[0]) / closes[0]`
7. **Calcule years_in_period** : depuis timestamps
8. **Calcule métriques** : `compute_metrics()`
9. **Construit equity curve**

```python
# Code critique : simulation d'un trade
entry_idx = sig_idx + 1  # entrée à OPEN[t+1] (anti-lookahead)
entry_price = float(opens[entry_idx])
exit_price, exit_reason, n_bars = simulate_trade(
    entry_idx, einher.amplitude_bars, einher.direction,
    entry_price, tp_pct, sl_pct, highs, lows, opens
)
gross = (exit_price - entry_price) / entry_price  # BUY
net = gross - costs_pct
```

---

## 8. `admission.py` — Critères d'admission (135 lignes)

**Rôle** : filtrer les Einhers selon des critères statistiques + diversité + BH.

### Mapping feature → famille (lignes 29-78)

`load_feature_family_map()` charge la taxonomie et retourne `{feature: family}`. Cache global.

`get_einher_families(einher)` parcourt récursivement l'AST pour extraire les `feature_ref` puis mappe vers les familles.

```python
# Code critique : walk récursif de l'AST
def _walk(node):
    if isinstance(node, dict):
        if "feature_ref" in node:
            features.add(node["feature_ref"])
        for v in node.values():
            if isinstance(v, (dict, list)):
                _walk(v)
```

### `AdmissionConfig` (lignes 81-107)

```python
@dataclass(frozen=True)
class AdmissionConfig:
    min_trades: int = 30
    min_sharpe: float = 0.3
    min_win_rate: float = 0.40
    min_profit_factor: float = 1.0
    max_drawdown: float = 0.30
    min_families: int = 2  # Sprint 2.2.2
    min_holdout_trades: int = 100  # Sprint 3.1 P1
    fdr: float = 0.05  # Sprint 3.1 P1 (Benjamini-Hochberg)
    apply_bh: bool = True  # Sprint 3.1 P1
```

**Méthode `debug()`** : seuils très souples (min_trades=5, fdr=1.0) pour itérer vite.

### `check_admission(einher, config, bh_rejected=None)` (lignes 110-161) — FONCTION CLÉ

**Pipeline de filtres dans l'ordre** :
1. **Sprint 3.1** : si `bh_rejected == False` → REJET (correction multi-tests)
2. **Sprint 2.2.2** : si `len(families) < min_families` → REJET (diversité)
3. **Sprint 2.4.1** : si `holdout_metrics.n_trades < min_holdout_trades` → REJET (significativité)
4. **Critères métriques** : via `einher.metrics.passes_admission()`

```python
# Code critique : ordre des filtres
if bh_rejected is False:
    return False, "BH REJECTED ..."
if config.min_families >= 2:
    families = get_einher_families(einher)
    if len(families) < config.min_families:
        return False, "Diversity REJECTED ..."
if config.min_holdout_trades > 0 and einher.holdout_metrics is not None:
    if einher.holdout_metrics.n_trades < config.min_holdout_trades:
        return False, "Holdout REJECTED ..."
passed, reason = einher.metrics.passes_admission(...)
```

---

## 9. `multiple_testing.py` — Benjamini-Hochberg (134 lignes)

**Rôle** : corriger le biais de tests multiples (on teste 30+ hypothèses par run).

### `bootstrap_pvalue(returns, n_bootstrap=1000)` (lignes 21-52)
Calcule la p-value bootstrap pour H0: `mean(returns) <= 0`. Resample n fois, calcule la stat, compte la fraction qui dépasse l'observée.

### `benjamini_hochberg(pvalues, fdr=0.05)` (lignes 65-108) — FONCTION CLÉ

Procédure BH pour contrôler le False Discovery Rate.

**Algorithme** :
1. Trier les p-values
2. Pour chaque rang i, seuil = `i/n * fdr`
3. Trouver le plus grand i où `p_i <= seuil`
4. Rejeter toutes les hypothèses de rang <= i

```python
# Code critique : algorithme BH
sorted_idx = np.argsort(pvals)
sorted_pvals = pvals[sorted_idx]
thresholds = np.arange(1, n + 1) / n * fdr
significant_sorted = sorted_pvals <= thresholds
if not significant_sorted.any():
    return [False] * n
max_significant_idx = np.where(significant_sorted)[0].max()
rejected_sorted = np.zeros(n, dtype=bool)
rejected_sorted[:max_significant_idx + 1] = True
rejected = np.zeros(n, dtype=bool)
rejected[sorted_idx] = rejected_sorted
```

### `apply_bh_to_einhers(einhers, fdr=0.05)` (lignes 111-157)
Calcule la p-value pour chaque Einher (approximation depuis sharpe + n si pas de rendements stockés) puis applique BH.

---

## 10. `multi_asset_loader.py` — Chargement multi-actifs (127 lignes)

**Rôle** : concaténer N actifs du même asset_class et timeframe pour augmenter la taille d'entraînement.

### `list_available_assets(asset_class, timeframe, require_ohlcv=False)` (lignes 51-79)
Liste les actifs qui ont des fichiers `_X.npy`, `_Y_dir.npy`, `_Y_ret.npy`. Optionnellement filtre sur la présence d'OHLCV.

### `load_multi_asset(assets, ...)` (lignes 82-134) — FONCTION CLÉ

```python
# Pipeline
loaded_list = [load_xy(asset, ...) for asset in assets]
# Verifier coherence des noms de features et horizons
ref_names = loaded_list[0].feature_names
for d in loaded_list[1:]:
    assert d.feature_names == ref_names
# Concatenation
X = np.concatenate([d.X for d in loaded_list], axis=0)
Y_dir = np.concatenate([d.Y_dir for d in loaded_list], axis=0)
asset_idx = np.concatenate([
    np.full(d.X.shape[0], i, dtype=np.int32)
    for i, d in enumerate(loaded_list)
])
```

**Hypothèse** : tous les actifs ont les mêmes horizons et feature_names (vrai pour MIDAS V3 compilé).

---

## 11. `feature_dedup.py` — Anti-duplication (133 lignes)

**Rôle** : drop les features trop corrélées (|r| > 0.85) pour éviter la redondance.

### `compute_corr_matrix(X)` (lignes 28-42)
Matrice de corrélation Pearson avec `np.corrcoef(rowvar=False)`. NaN remplacés par 0 (features constantes).

### `select_features_to_drop(X, feature_names, importances, threshold=0.85)` (lignes 66-139) — FONCTION CLÉ

**Algorithme glouton** :
1. Tant qu'il existe une paire (i, j) avec |r| > threshold :
   - Drop la feature la MOINS importante des deux
2. Retourne la liste des features à dropper

```python
# Code critique : stratégie gloutonne
while True:
    if current_X.shape[1] < 2:
        break
    corr = compute_corr_matrix(current_X)
    # Trouver la paire la plus correlee
    max_r, max_i, max_j = 0, -1, -1
    for i in range(F_curr):
        for j in range(i + 1, F_curr):
            if corr[i, j] > max_r:
                max_r, max_i, max_j = corr[i, j], i, j
    if max_r <= corr_threshold:
        break
    # Drop la moins importante
    if current_imp[max_i] < current_imp[max_j]:
        drop_idx = max_i
    else:
        drop_idx = max_j
    # Reconstruire keep_mask
    ...
```

### `apply_dedup(X, feature_names, importances, threshold=0.85)` (lignes 142-155)
Pipeline complet : retourne `(X_dedup, kept_names, dropped_names)`.

---

## 12. `feature_filter.py` — Drop patterns sparses (59 lignes)

**Rôle** : drop les features binaires trop rares (pct_True < 0.5%) ou trop saturées (pct_True > 99.5%).

### `is_binary_feature(col)` (lignes 26-32)
Détecte si une colonne est binaire (0/1 ou -1/0/1).

### `filter_sparse_patterns(X, feature_names)` (lignes 40-72) — FONCTION CLÉ

```python
# Code critique : filtrage binaire
for i in range(n_features):
    col = X[:, i]
    if not is_binary_feature(col):
        continue
    pct = compute_sparsity(col)
    if pct < min_pct or pct > max_pct:
        keep_mask[i] = False
```

**Pourquoi ce filtre** : un pattern avec `pct_True = 0.1%` (= 1 bougie sur 1000) ne peut pas aider XGBoost. Un split sur `== 1` n'isole qu'1 observation.

---

## 13. `einher_io.py` — Sérialisation JSONL (127 lignes)

**Rôle** : save/load les Einhers en JSON Lines.

### `save_einher(einher, path, append=True)` (lignes 24-30)
```python
def save_einher(einher: Einher, path: Path, append: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with open(path, mode, encoding="utf-8") as f:
        f.write(einher_to_json(einher) + "\n")
```

### `load_einhers(path)` / `iter_einhers(path)` (lignes 33-58)
Charge tous les Einhers (liste) ou itère (générateur).

### `_dict_to_einher(d)` (lignes 61-128) — FONCTION CRITIQUE

Reconstruit un Einher depuis son dict JSON. **Cas particulier** : le `condition_tree` peut être :
- `Condition` (1 seule condition) : juste `feature_ref, operator, value`
- `ConditionNode` (AND récursif) : `op, left, right`

```python
# Code critique : dispatch Condition vs ConditionNode
ct = d["condition_tree"]
if "op" in ct and "left" in ct and "right" in ct:
    condition_tree = ConditionNode(op=ct["op"], left=_dict_to_ast(ct["left"]), right=...)
else:
    condition_tree = _dict_to_ast(ct)
```

---

## 14. `label_engineer.py` — Target supervisé (96 lignes)

**Rôle** : construire le target (Y_ret signed return) + charger les coûts par actif.

### `build_target(loaded, horizon_idx)` (lignes 20-39) — FONCTION CLÉ
```python
def build_target(loaded: LoadedData, horizon_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid_mask = loaded.Y_dir[:, horizon_idx] != -100
    target = loaded.Y_ret[:, horizon_idx].copy()
    y_hor = loaded.Y_hor[:, horizon_idx].copy()
    return target, valid_mask, y_hor
```

### `load_costs(asset)` (lignes 82-118)
Charge les frais depuis `config/fees_ctrader.json`. **Bug** : la commission `$/lot` n'est pas convertie en % (TODO ligne 106).

---

## 15. `runner.py` — Orchestration (449 lignes)

**Rôle** : CLI qui orchestre tous les modules. C'est le **cœur** du système.

### `parse_horizon(horizon_str)` (lignes 77-85)
Convertit `'6h'` → 6, `'1d'` → 24, etc.

### `run_pipeline(...)` (lignes 88-285) — FONCTION PRINCIPALE

**Pipeline en 10 étapes** :

#### Étape 1 : Charger X, Y
- Single : `load_xy()` + `load_ohlcv()` + `align_xy_with_ohlcv()`
- Multi : `load_multi_asset()` (pour l'entraînement) + alignement sur le primary (pour le backtest)

#### Étape 2 : Drop sparse patterns (si `--drop-sparse`)
```python
X_global, feature_names, dropped = filter_sparse_patterns(X_global, feature_names)
```

#### Étape 3 : Feature dedup (si `--apply-dedup`)
```python
importances = {name: 1.0 for name in feature_names}  # uniforme
X_global, feature_names, dropped_dedup = apply_dedup(X_global, feature_names, importances)
```

#### Étape 4 : Appliquer les filtres à X_aligned_full
**CRITIQUE** : on doit appliquer les filtres d'un coup à X_aligned pour garder les mêmes indices de colonnes.

#### Étape 5 : Target + valid_mask
```python
valid_mask = Y_dir_global[:, horizon_idx] != -100
target = Y_ret_global[:, horizon_idx].copy()
X_valid = X_global[valid_mask]
y_valid = target[valid_mask].astype(np.float32)
```

#### Étape 6 : Split temporel (avec embargo proportionnel)
```python
split = temporal_split(X_valid, y_valid, embargo_bars=embargo_bars, horizon_bars=horizon_bars)
```

#### Étape 7 : Entraîner XGBoost
```python
config = GBDTConfig.regularized() if regularized else GBDTConfig(n_estimators=..., max_depth=...)
model, backend = train_gbdt(split.train_X, split.train_y, split.val_X, split.val_y, config)
```

#### Étape 8 : Extraire les chemins
```python
paths = extract_paths(model, backend, feature_names, min_score=min_score, max_paths=max_paths)
```

#### Étape 9 : Générer + backtester tous les Einhers (PHASE 1)
Pour chaque chemin, construire l'Einher et faire 2 backtests (val + holdout) :

```python
# Code critique : val sur [60%, 80%], holdout sur [80%, 100%]
val_start = int(n_aligned * 0.6)
val_end = int(n_aligned * 0.8)
val_result = backtest_einher(einher, ohlcv_aligned[val_start:val_end], X_aligned[val_start:val_end], ...)
if admission_cfg.min_holdout_trades > 0:
    holdout_start = int(n_aligned * 0.8)
    holdout_result = backtest_einher(einher, ohlcv_aligned[holdout_start:], X_aligned[holdout_start:], ...)
    einher = set_einher_holdout_metrics(einher, holdout_result.metrics)
```

#### Étape 9b : Benjamini-Hochberg (PHASE 2)
```python
if admission_cfg.apply_bh:
    _, _, bh_rejected = apply_bh_to_einhers(all_einhers, fdr=admission_cfg.fdr)
```

#### Étape 9c : Admission finale (PHASE 3)
```python
for einher, bh in zip(all_einhers, bh_rejected):
    passed, reason = check_admission(einher, admission_cfg, bh_rejected=bh)
    if passed:
        save_einher(einher, output_path)
```

### `cmd_run(args)` (lignes 290-360)
Dispatcher argparse. Gère les options `--scope`, `--asset-classes`, `--max-assets`, etc.

---

## Le pipeline complet — Vue d'ensemble

```
DONNÉES BRUTES
   │
   ├─► data_loader.load_xy()  ──────► LoadedData
   │   (X.npy, Y_*.npy)              (N, 213) features
   │                                  (N, H) labels
   │
   ├─► data_loader.load_ohlcv()  ──► pl.DataFrame OHLCV
   │   (CSV bruts)
   │
   └─► data_loader.align_xy_with_ohlcv()  ──► X_aligned + ohlcv_aligned
       (inner join timestamp)

NETTOYAGE FEATURES
   │
   ├─► feature_filter.filter_sparse_patterns()  ──► drop patterns < 0.5%
   │
   └─► feature_dedup.apply_dedup()  ──► drop |r| > 0.85

TARGET
   │
   └─► label_engineer.build_target()  ──► (target, valid_mask)
       Y_ret[:, horizon_idx]

SPLIT TEMPOREL
   │
   └─► data_loader.temporal_split()  ──► TrainValHoldoutSplit
       (60/20/20 + embargo)

ENTRAÎNEMENT XGBOOST
   │
   └─► model.train_gbdt()  ──► model XGBoost
       (n_estimators=200, max_depth=3, etc.)

EXTRACTION CHEMINS
   │
   └─► path_extractor.extract_paths()  ──► list[XGBPath]
       (filtre score + length, top 30)

CONSTRUCTION EINHER
   │
   └─► einher_builder.build_einher_from_path()  ──► Einher
       (direction, AST, ID unique)

BACKTEST (val + holdout)
   │
   └─► backtester.backtest_einher()  ──► BacktestResult
       (trades, metrics, equity_curve)

MULTI-TESTS
   │
   └─► multiple_testing.apply_bh_to_einhers()  ──► rejected[]
       (Benjamini-Hochberg FDR 5%)

ADMISSION FINALE
   │
   └─► admission.check_admission()  ──► (passed, reason)
       (BH + familles + holdout + métriques)

SAUVEGARDE
   │
   └─► einher_io.save_einher()  ──► JSONL append
       outputs/einhers_*.jsonl
```

---

## Les invariants et points critiques

### À NE PAS casser

1. **Dataclasses frozen** : utiliser `dataclasses.replace()` pour "modifier", pas d'assignment direct
2. **Anti-lookahead** : entrée à `OPEN[t+1]`, jamais `OPEN[t]` (ligne 295 backtester.py)
3. **SL-first sur bougie ambiguë** : conservatif (ligne 143-144 backtester.py)
4. **Embargo proportionnel** : `max(50, horizon_bars)` (ligne 294 data_loader.py)
5. **Sharpe annualisé correct** : `sqrt(trades_per_year)` (ligne 202 backtester.py)
6. **Filtre holdout >= 100** : significativité statistique (Sprint 3.1)
7. **BH activé par défaut** : éviter le p-hacking (Sprint 3.1)

### Connus bugs/limitations

1. **Sharpe sur trade returns** : la formule actuelle est correcte (annualisée) mais utilise la distribution des trades, pas la courbe d'equity continue. Gemini recommande d'aller plus loin avec la courbe d'equity horaire.
2. **Extraction de chemin XGBoost** : un chemin d'arbre = delta résiduel, pas E[Y|conditions]. C'est une approximation.
3. **Backtest single-actif** : le backtest reste sur l'actif primary, pas cross-asset.
4. **`commission_pct` non converti** : `load_costs()` lit `commission_per_lot` (en $) mais l'utilise comme si c'était % (label_engineer.py ligne 106).

### Améliorations prioritaires (Sprint 3.3+)

1. **Random Forest en parallèle** (validant) : si RF produit des Einhers similaires à XGBoost → signal réel
2. **Backtest multi-actif** : chaque Einher "market" testé sur tous les actifs du scope
3. **DSR (Deflated Sharpe Ratio)** : ajustement plus strict que BH
4. **Walk-forward** : split temporel sur 3 folds

---

## Comment lancer le système

```bash
# Single-asset, regularized, avec filtres
$env:PYTHONPATH='src'
& "D:/midas_v2/midas/Scripts/python.exe" -m einherjar.research.xgb_einhers.runner run `
  --asset BTCUSD --timeframe 1h --horizon 2d `
  --regularized --apply-dedup --drop-sparse `
  --min-holdout-trades 5 `
  --output outputs/einhers_btcusd.jsonl

# Multi-actif (scope=market)
& "D:/midas_v2/midas/Scripts/python.exe" -m einherjar.research.xgb_einhers.runner run `
  --scope market --asset-classes crypto --max-assets 5 `
  --timeframe 1h --horizon 2d `
  --regularized --apply-dedup --drop-sparse `
  --output outputs/einhers_market_crypto.jsonl

# Tests
$env:PYTHONPATH='src'; $env:PYTHONIOENCODING='utf-8'
& "D:/midas_v2/midas/Scripts/python.exe" -m unittest discover `
  -s src/einherjar/research/tests/test_xgb_einhers -p "test_*.py"
```

---

## Résumé

Le système est composé de **15 modules spécialisés** (~3000 lignes) qui forment un pipeline linéaire :
1. **Chargement** (data_loader, multi_asset_loader)
2. **Nettoyage** (feature_filter, feature_dedup)
3. **Target** (label_engineer)
4. **Split** (data_loader.temporal_split)
5. **Modèle** (model)
6. **Extraction** (path_extractor)
7. **AST** (condition_tree)
8. **Construction** (einher_builder)
9. **Backtest** (backtester)
10. **Multi-tests** (multiple_testing)
11. **Admission** (admission)
12. **I/O** (einher_io)
13. **Orchestration** (runner)

Les points sensibles sont : le **look-ahead bias** (entrée à `OPEN[t+1]`), la **convention SL-first**, l'**embargo proportionnel**, et le **Sharpe annualisé correct** (Sprint 3.0 fix). Les **Sprints 2.6 et 3.0** ont validé la cohérence val/holdout (ratio 0.91) et la généralisation cross-asset (100% sur 4 cryptos).
