# Plan A-Z : XGBoost → Einhers
**Date** : 2026-08-17
**Périmètre** : nouveau pipeline `xgb_einher` (à créer from scratch)
**Statut** : plan validé sur les bases — aucun code écrit tant que le user n'a pas confirmé
**Vision** : XGBoost supervisé multi-horizon → extraction d'arbres → conversion en Einhers → backtest → corpus

---

## 0. Résumé exécutif

**Objectif** : remplacer le moteur GP buggé par un pipeline XGBoost supervisé qui :
1. Charge X (218 features utilisables, hors OHLCV), Y_dir/Y_ret/Y_hor (4 horizons), OHLCV (CSV bruts)
2. Entraîne un `XGBRegressor` par (asset, timeframe, horizon) → prédit Y_ret
3. Extrait les arbres → chaque chemin feuille = un Einher candidat
4. Backtest chaque Einher sur OHLCV brut (nouveau moteur, pas le buggé)
5. Admet ceux qui passent les critères dans un fichier JSON

**Première itération** : BTCUSD × 1h × horizon 6h. 1 jour pour avoir un prototype fonctionnel.

**Différenciation Einhers** : on distingue *général* (entraîné sur tous les actifs), *marché* (entraîné sur un sous-ensemble de régime), *actif* (entraîné sur 1 seul actif).

---

## 1. Vue d'ensemble

```
[X.npy 218 features]  +  [Y_dir, Y_ret, Y_hor]  +  [OHLCV CSV bruts]
                              ↓
                [data_loader.py]
                              ↓
         [Nettoyage : drop OHLCV, drop Y_dir=-100, fillna]
                              ↓
                [label_engineer.py]
                              ↓
         [Y_target = Y_ret (signed return)]
                              ↓
            [model.py : XGBRegressor par (asset, TF, horizon)]
                              ↓
        [path_extractor.py : tree → list of (conditions, score)]
                              ↓
   [condition_tree.py : conditions → ConditionNode AST]
                              ↓
       [einher_builder.py : AST + direction + amplitude → Einher]
                              ↓
   [backtester.py : OHLCV → simulate trades → metrics]   ← NOUVEAU MOTEUR
                              ↓
              [admission.py : critères → admit / reject]
                              ↓
              [einher_io.py : JSON serialization]
                              ↓
         outputs/einhers_btcusd_1h_6h.jsonl
```

**Deux outputs principaux** :
- Un fichier JSON d'Einhers par (asset, TF, horizon) : `outputs/einhers_{asset}_{tf}_{horizon}.jsonl`
- Un rapport de diagnostic : `outputs/diagnostics_{asset}_{tf}_{horizon}.json`

---

## 2. Données : ce qu'on charge, ce qu'on exclut

### 2.1 Données d'entrée

**Source 1 : Features et labels** (`D:/midas_v2/midasV3/src/data/compiled/{class}/{tf}/`)

Pour chaque (asset, TF) :
| Fichier | Shape | Contenu |
|---|---|---|
| `{asset}_ts.npy` | (N,) int64 | Timestamps en ms epoch |
| `{asset}_X.npy` | (N, 246) float32 | 5 OHLCV (log-returns + log1p vol) + 241 features techniques |
| `{asset}_Y_dir.npy` | (N, 4) int8 | Direction : -100=invalide, 0=SELL, 1=HOLD, 2=BUY |
| `{asset}_Y_ret.npy` | (N, 4) float32 | Retour signé sur chaque horizon, clipé à [-0.15, 0.15] |
| `{asset}_Y_hor.npy` | (N, 4) float32 | Horizon en bars pour chaque colonne |
| `metadata.json` | dict | Ordre exact des colonnes X, horizons, sequence_lengths |

**Source 2 : OHLCV bruts** (`D:/midas_v2/technical_agent_dataset_brut/{class}/{asset}/{tf}/`)

CSV annuels : `{asset}_{year}_{tf}.csv` avec colonnes `timestamp, asset, timeframe, open, high, low, close, volume`.
- Prix réels (pas log-returns)
- Volume brut

### 2.2 Données EXCLUES (réponse Q6)

On **exclut** les 5 premières colonnes de `X.npy` (open, high, low, close, volume) car :
- Elles sont déjà des log-returns (donc "regardent" le passé proche)
- L'utilisateur veut éviter toute fuite de prix présente
- On a les vrais prix via les CSV bruts pour la simulation

On garde donc **213 features utilisables** (218 - 5 OHLCV = 213).

### 2.3 Mapping des features

Le mapping `feature_name → column_index` est dans `metadata.json` du dossier compiled.
Le mapping `feature_name → (type, value_type, family)` est dans `src/einherjar/research/config/features_taxonomy.json`.

**Filtre appliqué** :
- Exclure les 5 OHLCV de `metadata.json.feature_names`
- Exclure les features marquées `excluded=True` dans `features_taxonomy.json`
- Garder uniquement celles avec `excluded=False` : **213 features**

### 2.4 Nettoyage des labels

Pour chaque ligne i :
- Si `Y_dir[i, h] == -100` (invalide) → drop la ligne
- Sinon, on garde la ligne et le label est `Y_ret[i, h]`

**Note importante** : `Y_ret` est **signé**. Pour un BUY (Y_dir==2), Y_ret > 0 = gain. Pour un SELL (Y_dir==0), Y_ret < 0 = gain.

---

## 3. Taxonomie des features (213 utilisables)

**Source** : `src/einherjar/research/config/features_taxonomy.json`

### 3.1 Répartition par type

| Type | Nombre | Value type | Exemples |
|---|---|---|---|
| atomic | 58 | float | `rsi_14`, `macd_line`, `atr_14`, `bb_upper` |
| quantitative | 31 | float | `quant_realized_vol_20`, `quant_hurst_exponent`, `quant_garch_volatility` |
| factor | 13 | float (clip [0,1]) | `Factor_Momentum_Score`, `Factor_Trend_Score`, `Factor_Volatility_Score` |
| composite_derived | 9 | bool/float | `macd_trend_signal`, `aroon_trend_signal`, `supertrend_signal` |
| pattern | 107 | bool (0/1) | `pattern_hammer`, `pattern_doji`, `pattern_breakaway_bull` |
| **TOTAL** | **218** | | |

**Moins les 5 OHLCV de base = 213 features effectivement utilisées**.

### 3.2 Répartition par famille

| Famille | Nombre | Sens économique |
|---|---|---|
| price_action | 62 | Patterns de chandelles, structures de prix |
| market_structure | 52 | Supports, résistances, triangles, harmonic patterns |
| trend | 29 | EMA, SMA, ADX, Aroon, supertrend |
| momentum | 15 | RSI, MACD, momentum, ROC, Williams %R |
| statistical | 15 | Quant features (Hurst, entropy, fractal dim) |
| volatility | 14 | ATR, BB width, realized vol, GARCH |
| volume_flow | 11 | OBV, VWAP, MFI, volume ratios |
| market_regime | 9 | Trending/ranging detection, Hurst regime |
| risk | 8 | Kurtosis, skewness, CVaR |
| microstructure | 2 | VPIN, trade intensity |
| other | 1 | Divers |

### 3.3 Répartition par value_type (pour le mapping vers `Condition`)

- **float (106)** : `Condition(operator in {LT, GT, LE, GE, EQ, NE}, value=float)`
- **boolean (112)** : `Condition(operator in {EQ, NE}, value in {0, 1})`

---

## 4. Label engineering (XGBoost target)

**Réponse Q8** : on utilise une **régression sur Y_ret** directement. Avantage : on garde l'amplitude.

### 4.1 Construction du label

```python
# Pour chaque (asset, TF, horizon_h)
# horizon_h est l'index de colonne 0-3 dans Y_ret

target = Y_ret[:, horizon_h].astype(np.float32)
# target ∈ [-0.15, 0.15] (clipé)
# target > 0 = le prix a monté de X% sur l'horizon
# target < 0 = le prix a baissé de X% sur l'horizon
```

### 4.2 Masque de validité

```python
valid_mask = (Y_dir[:, horizon_h] != -100)  # Exclure les bougies invalides
# On garde seulement les lignes où valid_mask == True
```

### 4.3 Coûts de transaction

**Réponse Q9** : coûts variables par actif, depuis `config/fees_ctrader.json`.

```python
# Structure par symbole :
# EURUSD: spread_pct=0.000008, commission_per_lot=3.5
# BTCUSD: à vérifier dans fees_ctrader.json
# Format: spread_pct + commission + slippage

# On applique :
# round_trip_cost = (spread_pct + slippage_pct) * 2 + commission_per_trade_normalized
# Pour crypto, commission est souvent en % du trade, pas en $/lot
```

**Pour la Phase 1**, on simplifie : on utilise un coût forfaitaire par actif depuis `fees_ctrader.json`, converti en `pct` par trade.

### 4.4 Filtre de rentabilité

**Réponse Q11** : on exclut les bougies invalides, mais on garde toutes les autres (mêmes perdantes) pour que XGBoost apprenne la distribution complète.

Pas de filtre "que les gagnantes" — ce serait du biais de sélection.

---

## 5. Modèle XGBoost

### 5.1 Architecture

**Réponse Q10** : 1 modèle XGBoost par (asset, TF, horizon) = **4 modèles par (asset, TF)** car 4 horizons.

Pour 1h : horizons = [6h, 12h, 1d, 2d] → 4 modèles
Pour 15m : horizons = [1h, 2h, 4h, 8h] → 4 modèles

### 5.2 Hyperparamètres

```python
XGBRegressor(
    n_estimators=100,         # 100 arbres par modèle (ajustable)
    max_depth=4,              # Profondeur max 4 (contrôle la complexité)
    learning_rate=0.05,       # LR modéré
    subsample=0.8,            # 80% des samples par arbre (régularisation)
    colsample_bytree=0.8,     # 80% des features par arbre
    min_child_weight=10,      # Min 10 samples par feuille (anti-surapprentissage)
    reg_alpha=0.1,            # L1 regularization
    reg_lambda=1.0,           # L2 regularization
    random_state=42,          # Reproductibilité
    n_jobs=-1,                # Tous les CPU
    tree_method='hist',       # Histogram-based (rapide)
    # device='cuda'            # GPU si dispo (Phase 1 : CPU first, switch GPU Phase 2)
)
```

### 5.3 Split temporel

**60/20/20** avec embargo entre train/val/holdout.

```python
# Pour 70 000 bougies (BTCUSD 1h) :
# Train: bougie 0 à 42 000
# Embargo: 50 bougies (config)
# Val: bougie 42 050 à 56 000
# Embargo: 50 bougies
# Holdout: bougie 56 050 à 70 000
```

**Règle stricte** : le holdout est consulté **une seule fois** à la fin, jamais pendant l'entraînement ou la sélection.

### 5.4 Early stopping

```python
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=10,  # Arrêt si pas d'amélioration sur val en 10 itérations
    verbose=False
)
```

### 5.5 Validation croisée

**Pas de k-fold temporel** pour l'instant : le split 60/20/20 est suffisant en V1.
Le cross-asset (tester un Einher sur d'autres actifs) sera notre validation de généralisation (cf. § 11).

---

## 6. Extraction des chemins d'arbres

### 6.1 Principe

Chaque arbre XGBoost a une structure : nœuds internes (split) → feuilles (valeur).

Un **chemin** = suite de splits de la racine à une feuille.

**Exemple** :
```
Root: rsi_14 < 70
├── Left: macd_line > 0.5
│   ├── Left: pattern_hammer == 1
│   │   └── Leaf: 0.012 (predicted return)
│   └── Right: bb_width < 0.03
│       └── Leaf: -0.005
└── Right: atr_14 > 100
    └── Leaf: 0.003
```

**Chemin 1** : `rsi_14 < 70 AND macd_line > 0.5 AND pattern_hammer == 1` → leaf value 0.012
**Chemin 2** : `rsi_14 < 70 AND macd_line > 0.5 AND bb_width < 0.03` → leaf value -0.005
**Chemin 3** : `rsi_14 >= 70 AND atr_14 > 100` → leaf value 0.003

### 6.2 Algorithme d'extraction (pseudo-code)

```
def extract_paths(model, feature_names):
    paths = []
    for tree_idx in range(model.n_estimators):
        tree = model.get_booster().get_dump()[tree_idx]
        # Parser le texte de l'arbre (format XGBoost natif)
        # OU utiliser tree.predict_path() si on convertit en sklearn
        walk_tree(tree.root, conditions=[], leaf_value=None, paths)
    return paths

def walk_tree(node, conditions, leaf_value, paths):
    if node is leaf:
        paths.append({
            'conditions': conditions,  # liste de (feature, op, threshold)
            'score': leaf_value,        # valeur prédite par XGBoost
            'tree_idx': tree_idx
        })
        return
    # Nœud interne
    feature = feature_names[node.feature_idx]
    threshold = node.threshold
    # Branche gauche : feature < threshold
    walk_tree(node.left, conditions + [(feature, '<', threshold)], ...)
    # Branche droite : feature >= threshold
    walk_tree(node.right, conditions + [(feature, '>=', threshold)], ...)
```

### 6.3 Filtrage des chemins

Tous les chemins ne sont pas intéressants. On filtre :

```python
# Critères de filtrage
min_score = 0.005  # Prédiction absolue > 0.5%
max_score = 0.10   # Éviter les outliers
min_path_length = 1
max_path_length = 4  # Au-delà, c'est du surapprentissage probable

# Dédupliquer les chemins identiques (peu probable mais sûr)
seen = set()
unique_paths = []
for p in paths:
    key = tuple(p['conditions'])
    if key not in seen:
        seen.add(key)
        unique_paths.append(p)

# Trier par |score| décroissant
unique_paths.sort(key=lambda p: abs(p['score']), reverse=True)

# Top N
top_paths = unique_paths[:max_paths]  # 100 par défaut
```

### 6.4 Output

Une liste de dictionnaires :
```python
{
    'conditions': [
        ('rsi_14', '<', 70.0),
        ('macd_line', '>=', 0.5),
        ('pattern_hammer', '==', 1.0)
    ],
    'score': 0.012,            # Y_ret prédit (signed)
    'tree_idx': 42,
    'horizon': 6                # 6 bars (1h × 6 = 6h)
}
```

---

## 7. Construction des Einhers depuis les chemins

### 7.1 Représentation d'un Einher (la vision du user)

**Un Einher = une stratégie de trading** caractérisée par :
- un `id` unique
- ses **conditions de déclenchement** (trigger) : `condition_tree` (AST)
- un **TP** et un **SL** (en prix absolu ou en multiple d'ATR)
- son **univers** : actif(s), classe(s), timeframe(s), horizon
- ses **métriques** : win rate, max DD, Sharpe, CAGR, etc.

### 7.2 Conversion chemin → condition_tree (AST)

**Réponse Q12** : on commence par **AND-only** (XGBoost est naturellement AND), puis on ajoute OR/NOT/XOR en V2 si besoin.

```python
def path_to_condition_tree(path):
    """Convertit un chemin XGBoost en AST de conditions."""
    if len(path['conditions']) == 1:
        feat, op, val = path['conditions'][0]
        return Condition(
            feature_ref=feat,
            operator=map_op(op),    # '<' → CompareOp.LT, '>=' → CompareOp.GE
            value=float(val),
            transformation=None
        )
    # AND récursif
    left = path_to_condition_tree({'conditions': [path['conditions'][0]]})
    right = path_to_condition_tree({'conditions': path['conditions'][1:]})
    return ConditionNode(op=LogicalOp.AND, left=left, right=right)

def map_op(xgb_op):
    return {
        '<': CompareOp.LT,
        '<=': CompareOp.LE,
        '>': CompareOp.GT,
        '>=': CompareOp.GE,
        '==': CompareOp.EQ,
        '!=': CompareOp.NE
    }[xgb_op]
```

### 7.3 Direction de l'Einher (BUY ou SELL)

**Réponse Q8** : Y_dir peut rester comme confirmation, mais l'amplitude vient de Y_ret.

Logique :
- Si `Y_ret prédit > 0` → **BUY** (le prix va monter)
- Si `Y_ret prédit < 0` → **SELL** (le prix va baisser)
- Si `Y_ret prédit ≈ 0` → on ignore le chemin (pas de signal)

```python
if path['score'] > min_score:  # Y_ret prédit > seuil
    direction = 'BUY'
elif path['score'] < -min_score:
    direction = 'SELL'
else:
    skip  # Pas de signal
```

### 7.4 Amplitude de l'Einher (horizon du trade)

**Réponse Q13** : amplitude FIXE par horizon XGBoost.

- Pour 1h × 6h : amplitude = 6 bars
- Pour 1h × 12h : amplitude = 12 bars
- Pour 1h × 1d : amplitude = 24 bars
- Pour 1h × 2d : amplitude = 48 bars

L'Einher **entre à OPEN[t+1]** et **sort au plus tard à OPEN[t+amplitude]** (ou avant si TP/SL touché).

### 7.5 SL/TP de l'Einher (réponse Q14)

**RÈGLE GÉNÉRALE** : un système robuste utilise un **SL/TP dérivé des données**, pas de la prédiction.

**Pour la Phase 1, on utilise 2 stratégies en parallèle** et on compare :

**Stratégie A : TP/SL fixes en multiple d'ATR**
```python
tp_atr_mult = 2.5   # TP = 2.5 × ATR
sl_atr_mult = 1.5   # SL = 1.5 × ATR
# ATR calculé Wilder(14) sur les N bougies AVANT l'entrée
# (N = amplitude, ex: 6 pour 1h × 6h)
```

**Stratégie B : TP/SL dérivés de la prédiction Y_ret**
```python
# Le modèle a prédit Y_ret = 0.012 (1.2%)
# On prend TP à 1.0 × Y_ret, SL à 0.5 × Y_ret
tp_pct = abs(predicted_yret) * 1.0
sl_pct = abs(predicted_yret) * 0.5
# → Asymétrique : on prend moins de gain que ce qu'on prédit
```

**Pour V1, on retient la stratégie A** (ATR-based) car plus standard, plus robuste, et la stratégie B peut mener à du surapprentissage.

### 7.6 Structure de l'Einher final

```python
@dataclass
class Einher:
    id: str                              # 'xgb_btcusd_1h_6h_0001'
    condition_tree: ConditionNode         # AST des conditions
    direction: str                       # 'BUY' ou 'SELL'
    amplitude_bars: int                  # 6, 12, 24, 48
    sl_pct: float                        # ex: 0.015 (1.5%)
    tp_pct: float                        # ex: 0.025 (2.5%)
    universe: dict                       # {asset: 'BTCUSD', tf: '1h', horizon: '6h'}
    metrics: dict                       # win_rate, sharpe, max_dd, cagr, n_trades, ...
    scope: str                           # 'asset' (par défaut), 'general', 'market'
    created_at: str                      # ISO timestamp
    source: dict                         # {model: 'XGBRegressor', score: 0.012, ...}
```

---

## 8. Backtester (NOUVEAU, remplacement du moteur buggé)

**RÉPONSE Q14** : le moteur actuel est buggé. On en construit un nouveau, simple, vérifié.

### 8.1 Principe

Pour chaque bougie `t` où la `condition_tree` est vraie :
1. **Entrée** à `OPEN[t+1]`
2. Pendant les `amplitude` bougies suivantes (`[t+1, t+amplitude]`) :
   - Tracker le `high` max et le `low` min
   - Si `high >= entry_price * (1 + tp_pct)` ET `low <= entry_price * (1 - sl_pct)` → conflit, convention SL d'abord
   - Si TP touché avant SL → exit au TP, **win**
   - Si SL touché avant TP → exit au SL, **loss**
3. Si ni TP ni SL touché → exit à `CLOSE[t+amplitude]`, **timeout**

### 8.2 Algorithme (pseudo-code)

```python
def backtest_einher(einher, ohlcv_df, costs_pct):
    """
    Args:
        einher: Einher avec condition_tree, direction, amplitude_bars, sl_pct, tp_pct
        ohlcv_df: DataFrame polars avec colonnes [timestamp, open, high, low, close, volume]
        costs_pct: Coût round-trip (decimal, ex: 0.0008)
    
    Returns:
        trades: list of TradeResult
        metrics: dict
    """
    # 1. Évaluer la condition_tree sur tout l'historique → mask bool
    signal_mask = evaluate_condition(einher.condition_tree, ohlcv_df)
    
    # 2. Appliquer un cooldown (optionnel, défaut 0 pour V1)
    signal_indices = np.where(signal_mask)[0]
    
    # 3. Pour chaque signal
    trades = []
    for t in signal_indices:
        entry_idx = t + 1
        if entry_idx + einher.amplitude_bars > len(ohlcv_df):
            break  # Pas assez d'historique
        
        entry_price = ohlcv_df['open'][entry_idx]
        
        # Calculer TP/SL absolus
        if einher.direction == 'BUY':
            tp_price = entry_price * (1 + einher.tp_pct)
            sl_price = entry_price * (1 - einher.sl_pct)
        else:  # SELL
            tp_price = entry_price * (1 - einher.tp_pct)
            sl_price = entry_price * (1 + einher.sl_pct)
        
        # Simuler l'intrabar sur la fenêtre
        exit_price, exit_reason, n_bars_held = simulate_intrabar(
            entry_idx=entry_idx,
            amplitude=einher.amplitude_bars,
            direction=einher.direction,
            tp_price=tp_price,
            sl_price=sl_price,
            ohlcv_df=ohlcv_df
        )
        
        # Calculer le PnL net
        if einher.direction == 'BUY':
            gross_return = (exit_price - entry_price) / entry_price
        else:
            gross_return = (entry_price - exit_price) / entry_price
        
        net_return = gross_return - costs_pct
        
        trades.append(TradeResult(
            entry_idx=entry_idx,
            exit_idx=entry_idx + n_bars_held - 1,
            entry_price=entry_price,
            exit_price=exit_price,
            exit_reason=exit_reason,  # 'tp' | 'sl' | 'timeout'
            gross_return=gross_return,
            net_return=net_return
        ))
    
    # 4. Agréger les métriques
    metrics = compute_metrics(trades, einher, ohlcv_df)
    return trades, metrics
```

### 8.3 Simulation intrabar

Pour chaque bougie de la fenêtre `[entry_idx, entry_idx + amplitude]` :
- Comparer `high` avec TP et SL
- Comparer `low` avec TP et SL
- Si TP et SL touchés sur la même bougie : convention **SL d'abord** (conservateur)
- Retourner le prix de sortie et la raison

### 8.4 Métriques calculées

| Métrique | Formule |
|---|---|
| n_trades | len(trades) |
| win_rate | (trades où exit_reason='tp') / n_trades |
| avg_net_return | mean(trade.net_return) |
| total_return | sum(trade.net_return) |
| sharpe_ratio | mean(net_return) / std(net_return) * sqrt(trades_per_year) |
| max_drawdown | max_drawdown_from_equity_curve(net_returns) |
| profit_factor | sum(gains) / abs(sum(losses)) |
| avg_holding_bars | mean(trade.n_bars_held) |
| buy_hold_return | (ohlcv.close[-1] - ohlcv.close[0]) / ohlcv.close[0] |
| alpha | total_return - buy_hold_return |

### 8.5 Tests obligatoires pour le backtester

| Test | Description |
|---|---|
| test_no_lookahead | Les trades utilisent UNIQUEMENT des données jusqu'à t+amplitude |
| test_deterministic | Même input → même output (mêmes trades, mêmes métriques) |
| test_tp_before_sl | Convention SL-first respectée |
| test_costs_applied | Coûts déduits correctement du PnL |
| test_empty_universe | 0 trades → métriques nulles, pas de crash |
| test_single_trade | 1 trade → win_rate = 0 ou 1, sharpe = 0 |
| test_known_dataset | Un dataset de référence avec un signal connu → les métriques sont prévisibles |

---

## 9. Critères d'admission

**Réponse Q15** : tous ceux qui passent l'admission.

### 9.1 Critères minimaux (V1)

D'après `config/calibration.json` et `config/settings.json` :

| Critère | Seuil | Source |
|---|---|---|
| `n_trades` | ≥ 30 | calibration.json |
| `sharpe_ratio` | ≥ 0.3 | calibration.json |
| `win_rate` | ≥ 0.40 | calibration.json |
| `profit_factor` | ≥ 1.0 | calibration.json |
| `max_drawdown` | > -0.30 (donc DD < 30%) | calibration.json |
| `total_return` | > 0 | (positif) |
| `n_trades_per_month` | ≥ 0.3 | calibration.json |

### 9.2 Distinction scope Einher (réponse Q16)

Chaque Einher admis reçoit un attribut `scope` :
- **`asset`** : entraîné et testé sur un seul actif (par défaut)
- **`general`** : entraîné sur plusieurs actifs, testé sur d'autres (cross-asset, robustesse)
- **`market`** : entraîné sur un sous-ensemble de régime (bull, bear, range), testé ailleurs

Le `scope` est déterminé par les tests cross-asset (cf. § 11).

### 9.3 Structure du JSON de sortie

```json
{
  "id": "xgb_btcusd_1h_6h_0001",
  "condition_tree": {
    "op": "AND",
    "left": {
      "feature_ref": "rsi_14",
      "operator": "<",
      "value": 70.0,
      "transformation": null
    },
    "right": {
      "op": "AND",
      "left": {
        "feature_ref": "macd_line",
        "operator": ">",
        "value": 0.5,
        "transformation": null
      },
      "right": {
        "feature_ref": "pattern_hammer",
        "operator": "==",
        "value": 1.0,
        "transformation": null
      }
    }
  },
  "direction": "BUY",
  "amplitude_bars": 6,
  "tp_pct": 0.025,
  "sl_pct": 0.015,
  "universe": {
    "asset": "BTCUSD",
    "asset_class": "crypto",
    "timeframe": "1h",
    "horizon": "6h",
    "horizon_bars": 6
  },
  "metrics": {
    "n_trades": 47,
    "win_rate": 0.532,
    "sharpe_ratio": 0.85,
    "max_drawdown": -0.124,
    "profit_factor": 1.42,
    "total_return": 0.087,
    "avg_holding_bars": 4.2,
    "buy_hold_return": 0.045,
    "alpha": 0.042
  },
  "scope": "asset",
  "cross_asset_test": null,
  "source": {
    "model": "XGBRegressor",
    "tree_idx": 42,
    "path_score": 0.012,
    "feature_names": ["rsi_14", "macd_line", "pattern_hammer"]
  },
  "created_at": "2026-08-17T14:32:11Z",
  "data_version": "hash:..."
}
```

---

## 10. Sérialisation JSON

### 10.1 Format de sortie

**Réponse Q20** : JSON.

Un fichier par (asset, TF, horizon) :
```
outputs/einhers_{asset}_{tf}_{horizon}.jsonl
```

Format JSONL (une ligne par Einher), append-only.

### 10.2 Schéma JSON

Voir § 9.3. Chaque Einher est un objet JSON sérialisable directement.

### 10.3 Validation

Avant écriture :
- Tous les champs requis sont présents
- Types corrects (int, float, str, dict)
- `condition_tree` est un AST valide (récursivement)
- `metrics` contient au moins `n_trades`, `sharpe_ratio`, `win_rate`, `max_drawdown`

---

## 11. Cross-asset et scope des Einhers

### 11.1 Détermination du scope (réponse Q16)

**Phase 1** : on génère des Einhers "asset" par défaut.

**Phase 2** (optionnelle) : on évalue chaque Einher sur d'autres actifs :

```python
def determine_scope(einher, train_asset, all_assets_ohlcv):
    """
    Teste l'Einher sur tous les autres actifs disponibles.
    
    Returns:
        scope: 'asset' | 'general' | 'market'
        cross_asset_metrics: dict par asset testé
    """
    cross_results = {}
    for asset, ohlcv in all_assets_ohlcv.items():
        if asset == train_asset:
            continue
        try:
            trades, metrics = backtest_einher(einher, ohlcv, costs_pct=0.0008)
            cross_results[asset] = metrics
        except Exception:
            continue
    
    # Critères de scope
    n_tested = len(cross_results)
    n_profitable = sum(1 for m in cross_results.values() if m['total_return'] > 0)
    
    if n_tested == 0:
        scope = 'asset'  # Pas de test possible
    elif n_profitable / n_tested >= 0.7:  # 70% des autres actifs positifs
        scope = 'general'
    else:
        scope = 'asset'
    
    return scope, cross_results
```

### 11.2 Einhers par marché (régime)

**Phase 3** (optionnelle) : on identifie le régime (bull/bear/range) en utilisant une feature dédiée (`quant_regime_detection`) et on groupe les Einhers par régime.

**Pour V1**, on ignore le scope "market" et on se concentre sur "asset" et "general".

---

## 12. Architecture des fichiers

**Réponse implicite** : nouveau dossier `src/einherjar/research/xgb_einher/`.

```
D:/midas_v2/Einherjar/
├── src/
│   └── einherjar/
│       └── research/
│           ├── config/
│           │   ├── features_taxonomy.json        (existe déjà, 218 features)
│           │   ├── xgb_einher_config.yaml        (NOUVEAU - hyperparamètres)
│           │   └── fees_ctrader.json              (existe déjà)
│           ├── xgb_einher/                        (NOUVEAU MODULE)
│           │   ├── __init__.py
│           │   ├── data_loader.py                  # Charge X, Y, OHLCV
│           │   ├── label_engineer.py               # Y_ret → target supervisé
│           │   ├── model.py                        # XGBoost training/prediction
│           │   ├── path_extractor.py               # Arbres → chemins
│           │   ├── condition_tree.py               # Chemins → AST
│           │   ├── einher_builder.py               # AST → Einher
│           │   ├── backtester.py                   # NOUVEAU backtester
│           │   ├── admission.py                    # Critères
│           │   ├── scope_determiner.py             # Cross-asset
│           │   ├── einher_io.py                    # JSON serialization
│           │   ├── runner.py                       # CLI
│           │   └── types.py                        # Dataclasses Einher, Trade, etc.
│           ├── xgb_runner.py                       (NOUVEAU - entry CLI)
│           └── tests/
│               └── test_xgb_einher/                 (NOUVEAU)
│                   ├── test_data_loader.py
│                   ├── test_label_engineer.py
│                   ├── test_path_extractor.py
│                   ├── test_condition_tree.py
│                   ├── test_backtester.py
│                   ├── test_admission.py
│                   └── test_runner.py
├── outputs/
│   ├── einhers_{asset}_{tf}_{horizon}.jsonl       (JSONL par run)
│   ├── diagnostics_{asset}_{tf}_{horizon}.json    (rapport de diagnostic)
│   └── corpus.jsonl                                (consolidé, tous les Einhers)
└── audits/
    └── PLAN_XGBOOST_EINHER_AZ.md                  (CE DOCUMENT)
```

### 12.1 `xgb_einher/types.py` (NOUVEAU)

```python
@dataclass
class TradeResult:
    entry_idx: int
    exit_idx: int
    entry_price: float
    exit_price: float
    exit_reason: str  # 'tp' | 'sl' | 'timeout'
    gross_return: float
    net_return: float
    n_bars_held: int
    entry_timestamp: int
    exit_timestamp: int

@dataclass
class EinherMetrics:
    n_trades: int
    win_rate: float
    sharpe_ratio: float
    max_drawdown: float
    profit_factor: float
    total_return: float
    avg_holding_bars: float
    avg_net_return: float
    buy_hold_return: float
    alpha: float
    
    # Métriques par horizon
    n_tp: int
    n_sl: int
    n_timeout: int

@dataclass
class Einher:
    id: str
    condition_tree: dict  # AST sérialisable
    direction: str  # 'BUY' | 'SELL'
    amplitude_bars: int
    tp_pct: float
    sl_pct: float
    universe: dict
    metrics: EinherMetrics
    scope: str  # 'asset' | 'general' | 'market'
    cross_asset_test: Optional[dict]
    source: dict
    created_at: str
    data_version: str
```

### 12.2 `xgb_einher_config.yaml` (NOUVEAU)

```yaml
# Hyperparamètres XGBoost
xgb:
  n_estimators: 100
  max_depth: 4
  learning_rate: 0.05
  subsample: 0.8
  colsample_bytree: 0.8
  min_child_weight: 10
  reg_alpha: 0.1
  reg_lambda: 1.0
  random_state: 42
  tree_method: 'hist'
  device: 'cpu'  # ou 'cuda' si GPU dispo
  early_stopping_rounds: 10

# Splits temporels
splits:
  train_ratio: 0.6
  val_ratio: 0.2
  holdout_ratio: 0.2
  embargo_bars: 50  # Bougies d'embargo entre splits

# Extraction de chemins
paths:
  min_score: 0.005       # |Y_ret prédit| > 0.5%
  max_score: 0.10        # Évite outliers
  min_path_length: 1
  max_path_length: 4
  max_paths_per_model: 100

# SL/TP
risk:
  sl_atr_mult: 1.5
  tp_atr_mult: 2.5
  atr_period: 14

# Coûts (override de fees_ctrader.json)
costs:
  default_spread_pct: 0.0001
  default_commission_pct: 0.0001
  default_slippage_pct: 0.0001

# Critères d'admission
admission:
  min_trades: 30
  min_sharpe: 0.3
  min_win_rate: 0.40
  min_profit_factor: 1.0
  max_drawdown: 0.30  # DD > -0.30 (donc DD < 30%)

# Cross-asset (Phase 2)
cross_asset:
  enabled: false       # true pour Phase 2
  min_assets_profitable_pct: 0.7
  max_assets_to_test: 28
```

---

## 13. Tests

**Réponse Q19** : hiérarchie des tests, lancer les critiques en priorité.

### 13.1 Hiérarchie (du plus critique au moins critique)

| Priorité | Test | Pourquoi critique |
|---|---|---|
| **P0** | `test_backtester_no_lookahead` | Le moteur actuel a des fuites. Si le nouveau en a aussi, tout est faux. |
| **P0** | `test_backtester_deterministic` | Reproductibilité. Si non, débogage impossible. |
| **P0** | `test_data_loader_alignment` | X/Y/OHLCV alignés. Sinon, on apprend sur le futur. |
| **P1** | `test_label_engineer_no_future` | Y_ret[i, h] calculé uniquement avec données futures à t. |
| **P1** | `test_path_extractor_valid_paths` | Tous les chemins sont des AST valides. |
| **P1** | `test_admission_filters_correctly` | Les Einhers qui passent sont réellement ceux qui passent les seuils. |
| **P2** | `test_xgb_fit_predict_consistent` | Le modèle peut fit et predict sans erreur. |
| **P2** | `test_einher_io_roundtrip` | Sérialisation / désérialisation sans perte. |
| **P3** | `test_runner_cli_basic` | CLI fonctionne end-to-end sur un cas simple. |

### 13.2 Tests de non-régression (backtester)

**Le test le plus important** : `test_backtester_known_signal`.

```python
def test_backtester_known_signal():
    """Sur un dataset synthétique avec un signal connu, le backtester
    doit retourner un win_rate > 0.5 et un sharpe > 0."""
    
    # Créer un dataset synthétique :
    # - Bougies avec un signal 'pattern_hammer == 1' qui précède un mouvement haussier
    ohlcv = create_synthetic_with_known_pattern(
        n_bars=1000,
        pattern='hammer',
        direction='BUY',
        move_pct=0.02,
        move_within_bars=6
    )
    
    einher = Einher(
        condition_tree=Condition(feature='pattern_hammer', operator=EQ, value=1),
        direction='BUY',
        amplitude_bars=6,
        tp_pct=0.02,
        sl_pct=0.01
    )
    
    trades, metrics = backtest_einher(einher, ohlcv, costs_pct=0.0)
    
    assert metrics['n_trades'] > 30
    assert metrics['win_rate'] > 0.7  # Le pattern est quasi-déterministe
    assert metrics['sharpe_ratio'] > 1.0
```

Si ce test passe, on sait que le backtester fonctionne. S'il échoue, on a un bug fondamental.

---

## 14. CLI

**Réponse Q21** : CLI simple, pas de dashboard.

### 14.1 `xgb_runner.py` (NOUVEAU)

```bash
# Sanity check : charger les données, vérifier la distribution
python -m einherjar.research.xgb_runner sanity-check \
    --asset BTCUSD --timeframe 1h --horizon 6h

# Run complet : train XGBoost, extraire les chemins, backtest, admettre
python -m einherjar.research.xgb_runner run \
    --asset BTCUSD --timeframe 1h --horizon 6h \
    --n-estimators 100 --max-depth 4 \
    --max-paths 100 \
    --output outputs/einhers_btcusd_1h_6h.jsonl

# Mode diagnostic : juste entraîner XGBoost, voir les feature importances
python -m einherjar.research.xgb_runner train \
    --asset BTCUSD --timeframe 1h --horizon 6h \
    --show-feature-importance

# Cross-asset : tester un Einher sur d'autres actifs
python -m einherjar.research.xgb_runner cross-asset-test \
    --einhers outputs/einhers_btcusd_1h_6h.jsonl \
    --test-assets BTCUSD,ETHUSD,EURUSD,AAPL
```

### 14.2 Outputs

```
outputs/
├── einhers_btcusd_1h_6h.jsonl          # Einhers admis
├── diagnostics_btcusd_1h_6h.json      # Rapport de diagnostic
├── xgb_feature_importance_btcusd_1h_6h.json
└── corpus.jsonl                        # Consolidé (tous les Einhers)
```

---

## 15. Roadmap A-Z (étapes d'implémentation)

**Réponse Q17** : système complet. **Réponse Q22** : pas de deadline, on itère.

### Étape 0 : Sanity check (½ journée)

**Objectif** : comprendre les données réelles avant tout code.

- Charger `X.npy`, `Y_dir.npy`, `Y_ret.npy`, `Y_hor.npy` pour BTCUSD × 1h
- Vérifier shapes, distributions, valeurs manquantes
- Calculer la matrice `Y_dir == -100` (combien d'invalides ?)
- Distribution de Y_ret par horizon
- Vérifier la cohérence entre X.npy et metadata.json

**Livrable** : `audits/DATA_SANITY_BTCUSD_1H.md` avec stats et conclusions.

### Étape 1 : Squelette du module (1 jour)

**Objectif** : avoir la structure de fichiers vide + un test qui échoue proprement.

- Créer `xgb_einher/` avec tous les fichiers `__init__.py` + stubs
- Créer `xgb_einher_config.yaml`
- Créer `tests/test_xgb_einher/` avec tests squelettes
- Créer `xgb_runner.py` avec parser argparse et commandes stub

**Livrable** : `python -m einherjar.research.xgb_runner --help` fonctionne.

### Étape 2 : Data loader (1 jour)

**Objectif** : charger X, Y, OHLCV alignés.

- Implémenter `data_loader.py` :
  - `load_xy(asset, tf) -> (X, Y_dir, Y_ret, Y_hor, feature_names, valid_horizons)`
  - `load_ohlcv(asset, tf) -> pl.DataFrame`
  - `align_xy_with_ohlcv(X, ohlcv) -> (X_aligned, ohlcv_aligned)`
- Tests :
  - `test_data_loader_shapes` (shapes correctes)
  - `test_data_loader_alignment` (X et OHLCV alignés sur timestamp)
  - `test_data_loader_excludes_ohlcv_columns` (les 5 OHLCV sont exclues)
  - `test_data_loader_filters_invalid_y` (les Y_dir=-100 sont filtrés)

**Livrable** : `data_loader.py` + 4 tests passants.

### Étape 3 : Label engineer (½ journée)

**Objectif** : construire le target supervisé.

- Implémenter `label_engineer.py` :
  - `build_target(Y_ret, Y_dir, horizon_idx) -> (target, valid_mask)`
- Tests :
  - `test_label_shape`
  - `test_label_no_nan_inf`
  - `test_label_mask_filters_invalid`

**Livrable** : `label_engineer.py` + 3 tests.

### Étape 4 : Modèle XGBoost (1 jour)

**Objectif** : entraîner et prédire.

- Implémenter `model.py` :
  - `train_xgb(X_train, y_train, X_val, y_val, config) -> XGBRegressor`
  - `predict(model, X) -> np.ndarray`
- Tests :
  - `test_xgb_fit_predict_consistent`
  - `test_xgb_early_stopping`
  - `test_xgb_feature_importance` (bonus)

**Livrable** : `model.py` + 3 tests.

### Étape 5 : Path extractor (1 jour)

**Objectif** : extraire les arbres et les filtrer.

- Implémenter `path_extractor.py` :
  - `extract_paths(model, feature_names) -> list[Path]`
  - `filter_paths(paths, config) -> list[Path]`
- Tests :
  - `test_extract_paths_count` (100 arbres × ~5 feuilles = ~500 chemins)
  - `test_filter_by_score` (seuls les |score| > min_score)
  - `test_deduplicate` (chemins identiques fusionnés)

**Livrable** : `path_extractor.py` + 3 tests.

### Étape 6 : Condition tree (½ journée)

**Objectif** : convertir les chemins en AST.

- Implémenter `condition_tree.py` :
  - `path_to_ast(path, feature_names) -> dict`  (AST sérialisable)
- Tests :
  - `test_single_condition_to_ast`
  - `test_multiple_conditions_and`
  - `test_boolean_feature_value` (== 0 ou == 1)

**Livrable** : `condition_tree.py` + 3 tests.

### Étape 7 : Einher builder (½ journée)

**Objectif** : construire l'objet Einher final.

- Implémenter `einher_builder.py` :
  - `build_einher(ast, direction, amplitude, score, ...) -> Einher`
- Tests :
  - `test_build_einher_complete`
  - `test_einher_serialization_roundtrip`

**Livrable** : `einher_builder.py` + 2 tests.

### Étape 8 : Backtester (2 jours, **PRIORITÉ**)

**Objectif** : un backtester correct, vérifié, robuste.

- Implémenter `backtester.py` :
  - `evaluate_condition(ast, ohlcv) -> np.ndarray` (bool mask)
  - `simulate_intrabar(entry_idx, amplitude, direction, tp_price, sl_price, ohlcv) -> (exit_price, exit_reason, n_bars)`
  - `backtest_einher(einher, ohlcv, costs_pct) -> (trades, metrics)`
  - `compute_metrics(trades, einher, ohlcv) -> EinherMetrics`
- Tests (PRIORITÉ) :
  - **P0** : `test_backtester_no_lookahead`
  - **P0** : `test_backtester_deterministic`
  - **P0** : `test_backtester_known_signal` (sur dataset synthétique)
  - P1 : `test_backtester_tp_sl_priority` (SL-first sur bougie ambiguë)
  - P1 : `test_backtester_costs_applied`
  - P1 : `test_backtester_empty_universe`
  - P2 : `test_backtester_metrics_correctness` (sur dataset connu)

**Livrable** : `backtester.py` + 7 tests, **tous passants**.

### Étape 9 : Admission (1 jour)

**Objectif** : appliquer les critères.

- Implémenter `admission.py` :
  - `check_admission(einher, metrics, config) -> (admitted: bool, reason: str | None)`
- Tests :
  - `test_admit_passes_thresholds`
  - `test_reject_low_sharpe`
  - `test_reject_low_trades`
  - `test_reject_high_drawdown`

**Livrable** : `admission.py` + 4 tests.

### Étape 10 : Einher I/O (½ journée)

**Objectif** : sérialisation JSON.

- Implémenter `einher_io.py` :
  - `save_einher(einher, path)` → append au JSONL
  - `load_einhers(path) -> list[Einher]`
  - `save_diagnostics(report, path)`
- Tests :
  - `test_einher_io_roundtrip`
  - `test_jsonl_append`

**Livrable** : `einher_io.py` + 2 tests.

### Étape 11 : Runner end-to-end (1 jour)

**Objectif** : assembler le tout.

- Implémenter `xgb_runner.py` :
  - `cmd_sanity_check(args)` : charge les données, affiche les stats
  - `cmd_run(args)` : pipeline complet
  - `cmd_train(args)` : juste l'entraînement
  - `cmd_cross_asset_test(args)` : test cross-asset (Phase 2)
- Test E2E :
  - `test_runner_end_to_end` (run sur BTCUSD × 1h × 6h, vérifier le JSONL)

**Livrable** : `xgb_runner.py` + 1 test E2E.

### Étape 12 : Premier run réel (1 jour)

**Objectif** : produire les premiers Einhers réels.

```bash
python -m einherjar.research.xgb_runner run \
    --asset BTCUSD --timeframe 1h --horizon 6h \
    --n-estimators 100 --max-depth 4 \
    --output outputs/einhers_btcusd_1h_6h.jsonl
```

**Critères de succès** :
- [ ] Le pipeline tourne sans erreur
- [ ] On obtient ≥ 1 Einher admis (sinon c'est un signal faible)
- [ ] Les Einhers ont des métriques cohérentes (win_rate > 0, sharpe défini)
- [ ] Le JSONL est valide (lisible par `json.loads`)

**Livrable** : `outputs/einhers_btcusd_1h_6h.jsonl` avec au moins 1 Einher.

### Étape 13 : Expansion multi-horizon, multi-TF, multi-actif (à itérer)

- BTCUSD × 1h × [6h, 12h, 1d, 2d] (4 modèles)
- BTCUSD × 15m × [1h, 2h, 4h, 8h] (4 modèles)
- BTCUSD × 4h, 1d
- ETHUSD, EURUSD, AAPL, etc.

### Étape 14 : Cross-asset et scope (Phase 2, optionnelle)

- Implémenter `scope_determiner.py`
- Activer le cross-asset test
- Classifier les Einhers en "asset" vs "general"

### Étape 15 : Consolidation du corpus (Phase 2)

- Agréger tous les JSONL en un `corpus.jsonl` global
- Déduplication par fingerprint
- Rapport global

---

## 16. Critères de succès globaux

| # | Critère | Mesure | Priorité |
|---|---|---|---|
| 1 | Le pipeline tourne end-to-end | `python -m einherjar.research.xgb_runner run` retourne 0 | P0 |
| 2 | Au moins 1 Einher admis sur BTCUSD 1h 6h | `len(einhers) > 0` | P0 |
| 3 | Le backtester est correct | `test_backtester_known_signal` passe | P0 |
| 4 | Pas de look-ahead | `test_backtester_no_lookahead` passe | P0 |
| 5 | Reproductibilité | Même seed → même output | P0 |
| 6 | Les Einhers ont des métriques cohérentes | Win rate entre 0.3 et 0.7, Sharpe entre 0.3 et 3.0 | P1 |
| 7 | Multi-horizon fonctionne | Au moins 1 Einher par horizon | P1 |
| 8 | Multi-actif fonctionne | Au moins 1 Einher par actif | P1 |
| 9 | Cross-asset scope fonctionne | Au moins 1 Einher "general" détecté | P2 |
| 10 | Le corpus consolidé est utilisable | `corpus.jsonl` valide et lisible | P2 |

**Si on atteint 1-5, le système marche.** 6-10 sont des bonus.

---

## 17. Risques et mitigations

| # | Risque | Impact | Mitigation |
|---|---|---|---|
| 1 | Le backtester a un bug de look-ahead | Critique | `test_backtester_known_signal` + `test_backtester_no_lookahead` obligatoires |
| 2 | XGBoost sur-apprend et produit des Einhers bidons | Élevé | max_depth=4, min_child_weight=10, early_stopping |
| 3 | Les conditions XGBoost ne sont pas traduisibles en AST | Moyen | Test de sérialisation sur un échantillon |
| 4 | Le split temporel fuit (e.g., features avec rolling) | Élevé | Vérifier que les features à t utilisent uniquement les bougies < t |
| 5 | Le coût de round-trip est sous-estimé | Moyen | Comparer le win_rate avec/sans coûts |
| 6 | Aucun Einher n'est admis (trop strict) | Élevé | Baisser `min_sharpe` à 0.1 pour debug |
| 7 | Trop d'Einhers admis (trop laxiste) | Élevé | Ajouter `min_n_trades_per_month` |
| 8 | L'alignement timestamp échoue sur certains actifs | Moyen | `assert` sur la proportion d'overlap (> 99%) |
| 9 | Le GPU ne fonctionne pas avec XGBoost | Faible | CPU fallback (`tree_method='hist'`) |
| 10 | Le test `test_backtester_known_signal` échoue | Critique | Le backtester est buggé, on n'avance pas |

---

## 18. Glossaire

| Terme | Définition |
|---|---|
| **Einher** | Une stratégie de trading : condition_tree + direction + TP/SL + univers + métriques |
| **Path** | Un chemin dans un arbre XGBoost (racine → feuille) |
| **Condition tree (AST)** | Représentation arborescente des conditions : nœuds AND/OR + feuilles (feature, op, value) |
| **X.npy** | Features (246 colonnes, 5 OHLCV + 241 techniques) |
| **Y_dir** | Direction du mouvement : SELL=0, HOLD=1, BUY=2, invalide=-100 |
| **Y_ret** | Retour signé sur l'horizon, clipé à [-0.15, 0.15] |
| **Y_hor** | Horizon en bars (ex: 6 pour 1h × 6h) |
| **ATR** | Average True Range, indicateur de volatilité |
| **Sharpe ratio** | (return moyen / std) annualisé |
| **Max drawdown** | Perte max depuis un pic |
| **Profit factor** | somme(gains) / |somme(pertes)| |
| **Scope** | asset | general | market |
| **PCV** | (non utilisé ici) Combinatorial Purged Cross-Validation |
| **DSR** | (non utilisé ici) Deflated Sharpe Ratio |

---

## 19. Première itération : checklist

Quand tu commenceras l'étape 0, voici les commandes à lancer dans l'ordre :

```bash
# 1. Sanity check des données
python -c "
import numpy as np
X = np.load('D:/midas_v2/midasV3/src/data/compiled/crypto/1h/BTCUSD_X.npy')
print('X shape:', X.shape, 'dtype:', X.dtype)
Y_dir = np.load('D:/midas_v2/midasV3/src/data/compiled/crypto/1h/BTCUSD_Y_dir.npy')
print('Y_dir shape:', Y_dir.shape, 'unique:', np.unique(Y_dir, return_counts=True))
Y_ret = np.load('D:/midas_v2/midasV3/src/data/compiled/crypto/1h/BTCUSD_Y_ret.npy')
print('Y_ret shape:', Y_ret.shape, 'min:', Y_ret.min(), 'max:', Y_ret.max(), 'mean:', Y_ret.mean())
Y_hor = np.load('D:/midas_v2/midasV3/src/data/compiled/crypto/1h/BTCUSD_Y_hor.npy')
print('Y_hor shape:', Y_hor.shape, 'unique:', np.unique(Y_hor))
import json
with open('D:/midas_v2/midasV3/src/data/compiled/crypto/1h/metadata.json') as f:
    m = json.load(f)
print('Horizons:', m['horizons'])
print('Features:', m['features_count'])
print('First 10 features:', m['feature_names'][:10])
"

# 2. Vérifier qu'on a xgboost installé
python -c "import xgboost; print('xgboost version:', xgboost.__version__)"

# 3. Vérifier qu'on a les CSV OHLCV
ls "D:/midas_v2/technical_agent_dataset_brut/crypto/BTCUSD/1h/" | head -3
```

Si ces 3 commandes passent, on peut commencer l'étape 0 formellement.

---

## 20. Résumé des fichiers à créer

| Fichier | Lignes estimées | Priorité |
|---|---|---|
| `xgb_einher/__init__.py` | 10 | P2 |
| `xgb_einher/types.py` | 80 | P0 |
| `xgb_einher/data_loader.py` | 150 | P0 |
| `xgb_einher/label_engineer.py` | 60 | P0 |
| `xgb_einher/model.py` | 100 | P0 |
| `xgb_einher/path_extractor.py` | 120 | P0 |
| `xgb_einher/condition_tree.py` | 80 | P0 |
| `xgb_einher/einher_builder.py` | 100 | P0 |
| `xgb_einher/backtester.py` | 300 | **P0** |
| `xgb_einher/admission.py` | 100 | P0 |
| `xgb_einher/scope_determiner.py` | 80 | P1 (Phase 2) |
| `xgb_einher/einher_io.py` | 80 | P0 |
| `xgb_einher/runner.py` | 150 | P0 |
| `xgb_runner.py` | 50 | P0 |
| `config/xgb_einher_config.yaml` | 60 | P0 |
| **TOTAL code** | **~1520 lignes** | |
| Tests (8 fichiers) | ~600 lignes | P0 |

**Total estimé** : ~2100 lignes. Faisable en 1-2 semaines pour un développeur seul.

---

**FIN DU PLAN**

Quand tu valides, on commence par l'**étape 0** (sanity check des données) qui prend 1-2 heures et donne un go/no-go définitif.
