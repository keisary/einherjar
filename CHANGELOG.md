# Changelog EINHERJAR

> Tout changement notable du projet est documente ici.
> Format : date + description + fichier(s) touche(s) + auteur.

## [Unreleased]

### 2026-08-10
- **Optimisation run-time de l'évolution (objectif verdicts ~60-90 min, décision 2026-08-10)** :
  - `protocol.py` : champ `n_samples` (tasting) + option `make_protocol(n_samples=...)`. Valeur 0 = fenêtre complète (comportement historique préservé).
  - `algorithms.py` : helper `_taste_frames` — échantillonne le val en blocs contigus seedés (même fenêtre pour toute la population, cache par (val, n_samples)) ; branchement dans `_evaluate_population` (TypedGP). L'admission finale évalue TOUJOURS le val complet : l'honnêteté est préservée.
  - `evaluator.py` : `test_on(..., with_bootstrap=False)` — le block bootstrap (~90 % du temps de test_on) n'est lu QUE par l'admission ; pendant l'évolution/la comparaison, CI = NaN. Admission inchangée (bootstrap conservé).
  - `comparator.py` : competition en `with_bootstrap=False`.
  - `discovery.py` : flag `--taste-samples` (défaut 0) transmis aux 4 pipelines (compare/select/refine/admit).
  - Suppression de `_tasted_eval` (mort, logique doublonnée par le branchement direct).
  - Mesures : 15m bootstrap OFF → ~0,11 s/test_on au lieu de ~0,64 s ; run BTCUSD 3 TF estimé ~10 min au lieu de ~10 h.
- **Fichiers** : `src/einherjar/research/generators/protocol.py`, `algorithms.py`, `engine/evaluator.py`, `generators/comparator.py`, `research/discovery.py`

### 2026-08-10 (fix DSR — unité annualisée + déflation par finalistes)

**Problème** : 0/116 Einher admis sur BTCUSD 4 TF (run réel). Diagnostic :
le DSR recevait le Sharpe **par trade** (`ret_mean_pct_net/ret_std_pct`)
avec `n_observations = n_trades` : appliquer `sqrt(T-1)` sur des trades de
durées hétérogènes est incorrect, et la déflation `n_indep_trials=i`
(compteur de boucle d'admission) croissait linéairement — même la meilleure
hypothèse (Sharpe annuel loggé ~4-6) ne pouvait pas passer.

**Correction** :
- `criteria.evaluate_dsr(mesures, config, n_indep_trials, n_val_years=None)` :
  mode annualisé (par défaut en admission) — Sharpe = `mesures.sharpe_net`
  (annualisé par durée de détention, déjà loggé par test_on), z =
  `SR·sqrt(Y)` avec Y = années de la fenêtre val ; SE=1 (hypothèse normale,
  les moments par barre ne sont pas disponibles). Mode historique (par
  trade) conservé quand `n_val_years=None` (compat tests/appelants).
- `utils/metrics.dsr(...)` : nouveaux paramètres optionnels `sqrt_factor`
  (remplace `sqrt(n-1)`) et `correct_non_normality=False` (SE=1).
- `discovery.handle_admit` : `n_indep_trials = len(hyps_a_evaluer)` (nombre
  réel de finalistes testés sur CE val) au lieu de l'index de boucle ;
  `n_val_years = len(val)/periods_per_year(timeframe)` transmis au décideur.
- Tests : `tests/test_dsr_correction.py` (6 cas : passe/échoue par unité,
  sensibilité à la déflation, NaN, mode historique).

### 2026-08-10 (run réel BTCUSD 4 TF — campagne `/d/midas_v2/campaign_btc_20260810.sh`)
- **Bugs découverts en conditions réelles (fixés et poussés)** :
  - `discovery.py` (`1dbd46b`) : `handle_compare` ne persistait pas `max_conditions` dans la meta du rapport → `_load_compare_report` le déclarait toujours stale → select re-comparait à chaque fois (~540 évals perdues/TF). Fix : persister `max_conditions`.
  - `selection/selector.py` (`7e38437`) : `instantiate` créait `TypedGPGenerator(protocol, config)` SANS `engine` → ValueError systématique à refine/admit. Fix : `engine=engine`.
- **Résultats campagne** (mono TypedGP, n_eval=1200, --taste-samples 400, 4 TF, durée ~1h10) :
  - 15m : 0/33 admis (30 DSR_FAIL, 3 DIVERSITY_FAIL) ; 1h : 0/28 ; 4h : 0/16 ; 5m : 0/39 (tous DSR_FAIL).
  - Total : **0 Einher admis / 116 hypothèses évaluées en admission** (114 DSR_FAIL, 3 DIVERSITY_FAIL; 1 ligne d'écart archive vs run.log — double comptage d'une hyp réévaluée).
  - Lecture : aucune stratégie BTCUSD ne passe la porte statistique (DSR) après coûts avec les features actuelles — verdict honnête, cohérent avec 2026-08-09.
  - Pipeline validé de bout en bout : ~17 min/TF (contre ~10 h avant optimisation), tasting + bootstrap OFF effectifs, sélection sans re-comparaison, admission bootstrap ON.

### 2026-07-20
- **Cahier des charges** : mise a jour complete v0.3
  - Limites globales relachees (expo 60%, 15 positions, etc.)
  - Theme dashboard nordique "Einherjar" ajoute (section 6.1)
  - Philosophie de dev ajoutee : JSON exclusif, naming conventions, docstrings
  - Short crypto tranche : MT5 CFDs en v1.1, long-only spot en v1
  - Estimation charge machine supprimee, remplacee par contraintes mesurees
  - Phase 3 : objectif min 30-60 Einhers, potentiel reel 300+
- **Fondation** : creation de l'arborescence et modules de base
  - `pyproject.toml` : deps polars, numba, duckdb, ccxt, fastapi, etc.
  - `core/enums.py` : EinherState, Direction, OrderType, TimeFrame, AssetClass, RejectionReason
  - `core/config.py` : RiskLimits, ValidationConfig, SystemConfig, loader JSON
  - `core/models.py` : Einher, Signal, Order, Fill, Position, AccountState, Rejection (dataclasses + docstrings)
  - `brokers/adapter.py` : Protocol BrokerAdapter (get_ohlcv, place_order, get_positions, etc.)
  - `data/store.py` : DataStore DuckDB, 8 tables (ohlcv, signals, orders, fills, positions, equity_curve, rejections, einher_stats)
  - `data/ohlcv_manager.py` : fetch incremental, detection fraicheur, fallback broker
  - Configs JSON templates : settings, fees_binance, fees_alpaca, fees_oanda, native_exits
- **Phase 1 — Inventaire MIDAS V3** : exploration complete de `midasV3/src/data/compiled`
  - 7 classes, 99 actifs uniques, 5 timeframes (5m, 15m, 1h, 4h, 1d)
  - Structure confirmee : `*_ts.npy` (timestamps), `*_X.npy` (246 features float32), `*_Y_dir/hor/ret.npy` (labels)
  - Selection top 28 actifs : choix manuel par expertise marche, tous verifies dans MIDAS
  - Resultat : 5 crypto, 8 actions US, 8 forex, 4 indices, 3 commodities
  - Fichiers : `docs/inventory_npy.json`, `config/assets_v1.json`
- **Phase 1 — PaperBroker + adaptateurs stubs** : simulation et architecture broker
  - `brokers/paper_broker.py` : simulation avec slippage pessimiste, frais par broker, latence 200ms, gestion positions
  - `brokers/binance_adapter.py` : stub CCXT pour crypto spot
  - `brokers/alpaca_adapter.py` : stub pour actions US fractionnaires
  - `brokers/oanda_adapter.py` : stub pour forex/CFD/metaux
- **Phase 2 — Signal Engine (COMPLETE)** : portage TOTAL des modules MIDAS
  - `signals/midas_bridge.py` : PatternBridge, IndicatorBridge, QuantBridge. Isolation des fonctions numba pures du batching Dask/multiprocessing interne de MIDAS
  - `signals/feature_engine.py` : reecriture complete avec les **183 features totales** : 107 patterns + 52 indicateurs + 24 features quantitatives. LOOKBACK_WINDOWS exhaustif pour chaque feature, compute batch + compute incremental
  - `signals/einher_engine.py` : evaluation polars, etats forming/triggered, calcul TP/SL natif + fallback ATR
  - `config/native_exits.json` : regles de sortie natives pour 35+ familles de patterns (TP/SL natifs)
  - `config/corpus_v1_demo.json` : 6 Einhers de demo couvrant 5 domaines
- **Phase 2 — Scheduler live** : orchestrateur asyncio du cycle d'inference
  - `scheduler/loop.py` : InferenceLoop, calcul des clotures alignees, marge 10s, parallélisation par actif, cycle fetch -> store -> features -> einher -> risk -> execution -> journalisation
  - `data/live_store.py` : LiveDataStore Parquet incremental par (asset, tf), fenetre glissante 500 bougies
- **Phase 2 — Risk Manager** : dimensionnement et limites globales
  - `risk/manager.py` : RiskManager avec sizing par risque fixe, plafond confiance, circuit breakers (daily loss, drawdown, weekly loss), limites exposition/positions/correlation, journalisation rejets
  - Mapping correlation simplifie, calendrier marche simplifie (crypto 24/7, forex week-end, actions US heures)
- **Phase 2 — Corpus Brut (GENERE)** : generation automatique de 1496 Einhers couvrant l'ensemble des 183 features
  - 7 domaines : Pattern pur (208), Pattern + confluence (1040), Indicateur classique (132), Breakout/volatilite (28), Quantitatif (52), Multi-TF (16), Cross-asset (20)
  - Contrainte respectee : max 3 conditions (trigger + 2 filtres) par Einher
  - Directions : long (652), short (616), both (228)
  - Fichier : `config/corpus_brut_v1.json`
- **Phase 3 — Calibration Backtest (PRET)** : moteur de backtest complet sur les 1496 Einhers
  - `backtest/calibrator.py` : simulation trade par trade avec SL/TP natif ou fallback ATR, frais par broker, slippage, cooldown, max_holding. Groupement par (asset, tf) pour optimiser la RAM. Reconstruction des prix absolus depuis X.npy normalise
  - `backtest/metrics.py` : Sharpe, Sortino, win rate, profit factor, max drawdown, expectancy, trades/mois
  - `backtest/analyzer.py` : selection par seuils configurables (min_sharpe, min_winrate, etc.) ou top N par score composite. Enrichissement avec les definitions du corpus brut
  - `backtest/data_source.py` : chargement MIDAS, mapping noms corpus -> MIDAS, reconstruction OHLCV absolus
  - `scripts/run_calibration.py` : entry point backtest complet
  - `scripts/run_analyzer.py` : entry point analyse standalone (relancable sans refaire le backtest)
  - `config/calibration.json` : parametres de backtest et seuils de selection configurables
- **Fichiers** : `backtest/*.py`, `scripts/*.py`, `config/calibration.json`

- **Phase 3 — Corpus Brut v2 (GENERE)** : regeneration complete avec vrais noms MIDAS et strategies diversifiees
  - 7535 Einhers brut (non calibres) couvrant les 183 features totales
  - 7 domaines : Pattern+confluence (3780), Indicateur+confluence (1600), Quant+confluence (1040), Indicateur pur (500), Pattern pur (305), Quantitatif (220), Multi-features (90)
  - Noms de features alignes sur MIDAS : `pattern_hammer`, `rsi_14`, `quant_hurst_exponent`, etc. Zero residu `col_`
  - Triggers et filtres utilisent les vrais noms de colonnes du DataFrame MIDAS
  - Fichier : `config/corpus_brut_v2.json`

- **Phase 3 — Calibration Backtest (CORRIGE + VALIDE)** : moteur adapte au corpus v2
  - `backtest/data_source.py` : mapping `map_feature_name` rendu defensif (ne mapper que si `col_` present). Support natif des noms MIDAS
  - `backtest/calibrator.py` : `_eval_expression` valide avec `pl.sql_expr` et operateur `AND` majuscule
  - `scripts/run_calibration.py` : pointe vers `corpus_brut_v2.json`, lit `top_n` depuis `calibration.json`
  - `config/calibration.json` : ajout `top_n` configurable (null = tous les seuils passes)
  - `backtest/metrics.py` : **CORRECTION MATHEMATIQUE MAJEURE**
    - Sharpe ratio : annualise par `sqrt(trades_par_an)` calcule depuis la frequence reelle. Avant : non annualise, incomparable
    - Sortino ratio : idem annualise. Downside deviation par rapport a MAR=0
    - Total return : `prod(1+r) - 1` (compose) au lieu de `sum(r)` (somme fausse)
    - Profit factor, win rate, expectancy, max drawdown, trades/mois : formules verifiees et correctes
  - Mini-backtest valide : 7 Einhers sur BTCUSD 15m, 4 resultats produits, metriques corrigees
  - Fichiers : `backtest/*.py`, `scripts/run_calibration.py`, `config/calibration.json`

- **Phase 4 — Risk Manager + PaperBroker + Portfolio (COMPLETE)** : pipeline temps reel operationnel
  - `brokers/paper_broker.py` : matching complet avec SL/TP logiciel, gestion positions ouvertes, P&L latent, fermeture sur reversal. Simulation realiste avec slippage pessimiste et frais par broker
  - `risk/manager.py` : sizing par risque fixe, plafonds exposition/positions/correlation, circuit breakers journaliers/hebdomadaires/drawdown, journalisation des rejets
  - `signals/einher_engine.py` : evaluation reelle des conditions polars, detection etat `forming` (conditions partiellement remplies), calcul TP/SL natif + fallback ATR, score de confiance base sur les filtres passes
  - `data/store.py` : methodes append_order, append_fill, update_position, remove_position, get_positions. Journalisation complete des cycles
  - `scripts/test_pipeline_phase4.py` : test integre validant le cycle complet fetch -> features -> einher -> risk -> paper broker -> journal. 200 bougies simulees, signaux triggered + forming detectes, ordres executes, base DuckDB peuplee
- **Fichiers** : `src/einherjar/brokers/paper_broker.py`, `src/einherjar/risk/manager.py`, `src/einherjar/signals/einher_engine.py`, `src/einherjar/data/store.py`, `scripts/test_pipeline_phase4.py`

- **Phase 5 — Adaptateurs Live (COMPLETE)** : implementation complete des 3 brokers v1
  - `brokers/broker_utils.py` : helpers communs (normalisation symboles, conversion OHLCV, retry backoff, load fees, timeframes)
  - `brokers/binance_adapter.py` : adaptateur CCXT complet pour crypto spot (testnet/live). get_ohlcv, subscribe_live (polling), place_order, cancel_order, get_positions (balances), get_account. Import ccxt differe
  - `brokers/alpaca_adapter.py` : adaptateur complet pour actions US fractionnaire (paper/live). get_ohlcv, subscribe_live (polling), place_order, cancel_order, get_positions, get_account. Import alpaca-trade-api differe
  - `brokers/oanda_adapter.py` : adaptateur complet pour forex/metaux/indices CFD (practice/live). get_ohlcv, subscribe_live (polling), place_order (avec TP/SL natifs OANDA), cancel_order, get_positions (long+short natifs), get_account. Import oandapyV20 differe
  - Mapping symboles MIDAS <-> broker pour chaque venue : `BTCUSD` -> `BTC/USDT` (Binance), `EURUSD` -> `EUR/USD` (OANDA), etc.
  - Tests de structure valides : imports, normalisation, OHLCV polars, instanciation sans cles API
- **Fichiers** : `src/einherjar/brokers/broker_utils.py`, `src/einherjar/brokers/binance_adapter.py`, `src/einherjar/brokers/alpaca_adapter.py`, `src/einherjar/brokers/oanda_adapter.py`, `scripts/test_adapters.py`

- **Phase 5 — Durcissement et Resilience (COMPLETE)** : protection contre les pannes reseau et API
  - `brokers/resilience.py` : wrapper `ResilientBroker` qui encapsule tout adaptateur avec :
    - Circuit breaker (CLOSED/OPEN/HALF_OPEN) avec seuil configurable et recovery automatique
    - Rate limiter (appels par seconde et par minute) avec attente non bloquante
    - Logging structure JSON de tous les appels (succes et echecs)
    - Delegation protegee de toutes les methodes BrokerAdapter
  - Tests valides : circuit breaker (3 etats), rate limiter (10 appels), ResilientBroker (recovery apres panne simulee)
- **Fichiers** : `src/einherjar/brokers/resilience.py`, `scripts/test_resilience.py`

- **Phase 5b/v1.1 — Short complet (COMPLETE)** : activation du short sur toutes les classes d'actifs
  - `brokers/binance_futures_adapter.py` : **NOUVEAU** adaptateur Binance Futures perpetuels USDT-M. Supporte long + short natif avec levier configurable (1-125x). get_ohlcv, subscribe_live, place_order, get_positions (long+short), get_account (marge, equity). Import ccxt differe
  - `brokers/cfd_adapter.py` : **NOUVEAU** adaptateur generique CFD via CCXT. Supporte Pepperstone, FP Markets, IC Markets, AvaTrade. Long + short natif sur crypto, forex, indices, commodites. get_ohlcv, subscribe_live, place_order, get_positions (long+short), get_account. Import ccxt differe
  - `brokers/alpaca_adapter.py` : **short active** — le code supportait deja le short, activation confirmee pour la v1.1. Positions negatives (sell) et get_positions avec direction SHORT
  - Couverture short par classe :
    - Crypto : Binance Futures (levier) + CfdAdapter (Pepperstone/FP Markets/IC Markets/AvaTrade)
    - Actions US : Alpaca (short natif)
    - Forex/indices/metaux : OANDA (short natif, deja en v1)
  - Tests : imports OK pour BinanceFuturesAdapter et CfdAdapter, 4 brokers CFD listes
- **Fichiers** : `src/einherjar/brokers/binance_futures_adapter.py`, `src/einherjar/brokers/cfd_adapter.py`, `scripts/test_v11_adapters.py`

---

---

## Format des entrees

```
### YYYY-MM-DD
- **Categorie** : description du changement
  - Details supplementaires
- **Fichier(s)** : `chemin/vers/fichier.py`
```

Categories : Added, Changed, Deprecated, Removed, Fixed, Security.
