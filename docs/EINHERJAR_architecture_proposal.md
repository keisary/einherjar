# EINHERJAR — Proposition d'architecture

> Version 0.1 — Document de travail. Base de discussion avant le cahier des charges.
> Système de trading algorithmique léger, détaché de MIDAS, à règles déterministes.

---

## 1. Principes directeurs

1. **Légèreté absolue** : tourne sur 20 GB RAM / 12 cœurs, aucune infra lourde (pas de Dask, pas de GPU, pas de deep learning).
2. **Règles explicites** : chaque signal est une condition polars/numba lisible, avec TP/SL dérivés des règles du pattern ou de la stratégie.
3. **Réutilisation maximale** : détecteurs numba (105 patterns), indicateurs numba (~25), templates de stratégies (299), structures de validation.
4. **Capital-agnostique** : sizing en % du portefeuille. Même code pour 10 USD et 100 000 USD.
5. **Paper = Live** : un seul switch de configuration change la destination des ordres.
6. **Préfiguration MIDAS** : le moteur d'exécution, le flux temps réel et le risk manager d'EINHERJAR deviendront les briques manquantes de MIDAS.

---

## 2. Vue d'ensemble

```
                    ┌──────────────────────────────────────────┐
                    │              EINHERJAR CORE              │
                    │                                          │
  Sources marché    │  ┌─────────┐   ┌───────────────────┐     │
  (REST + WS)  ────►│  │ MARKET  │──►│  SIGNAL ENGINE    │     │
                    │  │  DATA   │   │  - indicateurs    │     │
                    │  │  layer  │   │  - patterns (105) │     │
                    │  └────┬────┘   │  - stratégies     │     │
                    │       │        └─────────┬─────────┘     │
                    │       │                  │ signaux       │
                    │       │        ┌─────────▼─────────┐     │
                    │       │        │  CONFLUENCE /     │     │
                    │       │        │  ARBITRAGE        │     │
                    │       │        └─────────┬─────────┘     │
                    │       │                  │ intentions    │
                    │       │        ┌─────────▼─────────┐     │
                    │       │        │  RISK MANAGER     │     │
                    │       │        │  (allégé)         │     │
                    │       │        └─────────┬─────────┘     │
                    │       │                  │ ordres        │
                    │       │        ┌─────────▼─────────┐     │
                    │       └───────►│  EXECUTION        │────►│──► Broker / Paper
                    │                │  ENGINE           │     │
                    │                └─────────┬─────────┘     │
                    │                          │ fills          │
                    │                ┌─────────▼─────────┐     │
                    │                │  PORTFOLIO +      │     │
                    │                │  JOURNAL (DuckDB) │     │
                    │                └─────────┬─────────┘     │
                    │                          │               │
                    │                ┌─────────▼─────────┐     │
                    │                │  DASHBOARD (web)  │     │
                    │                └───────────────────┘     │
                    └──────────────────────────────────────────┘
```

---

## 3. Modules

### 3.1 Market Data Layer

- **Source unique par classe d'actifs**, double accès :
  - Crypto : Binance via CCXT (REST historique + WebSocket live).
  - Actions/indices/forex/commodités : à décider selon broker retenu (Alpaca, IBKR, ou yfinance en dégradé pour le paper).
- **Store local** : DuckDB + fichiers Parquet par actif/timeframe. Mise à jour incrémentale, jamais de re-téléchargement complet.
- **Règle d'or** : indicateurs et patterns calculés uniquement sur **bougies clôturées**. Le flux live sert à l'exécution, au suivi des stops/TP et au monitoring.
- **Multi-TF natif** : 5m, 15m, 1h, 4h, 1d. Resampling 1m → TF supérieurs si la source le permet, sinon téléchargement par TF.
- **Univers** : ~25 actifs sélectionnés depuis les 99 de `unique_assets.txt` par liquidité + volatilité + frais. Sélection paramétrable, revue périodique.

### 3.2 Signal Engine

- **Réutilisation directe** :
  - `numba_pattern_detectors.py` (105 détecteurs, entrées numpy OHLCV).
  - `technical_indicators.py` (~25 indicateurs numba).
- **Format d'une stratégie EINHERJAR** (adapté de `ValidatedStrategy`) :

```python
@dataclass
class EinherjarStrategy:
    name: str
    category: str
    direction: str            # 'long' | 'short' | 'both'
    timeframes: list[str]     # TF d'évaluation, ex: ['1h', '4h']
    condition: str            # expression polars, comme dans config.py
    tp_rule: dict             # ex: {'type': 'pattern_height'} ou {'type': 'atr', 'mult': 2.5}
    sl_rule: dict             # ex: {'type': 'pattern_neckline'} ou {'type': 'atr', 'mult': 1.5}
    max_holding: str | None   # durée max en position
    min_confidence: float     # seuil d'émission
    # Métriques de validation (remplies par le screening)
    sharpe: float = 0.0
    win_rate: float = 0.0
    profit_horizon: str = ''
```

- **TP/SL natifs** : chaque famille de pattern emporte ses règles classiques :
  - Double top : SL au-dessus du sommet, TP = hauteur du pattern projetée sous la ligne de cou.
  - Cassure Ichimoku : SL de l'autre côté du nuage, TP par trailing ou ATR.
  - Harmoniques : entrée PRZ, SL sous X, TP sur ratios Fibonacci.
  - Fallback générique : SL = k1 × ATR, TP = k2 × ATR.
- **Progression partielle** : pour chaque stratégie multi-conditions, exposition de l'état (2/3 conditions remplies, en attente de confirmation). Base du dashboard.

### 3.3 Confluence / Arbitrage

Résolution des signaux concurrents, dans cet ordre :

1. **Partition par horizon** : deux signaux opposés sur des horizons différents (TP +5% court vs +40% long) ne sont pas en conflit. Chacun produit une intention distincte, traitée par le Risk Manager avec des poches de capital séparées.
2. **Agrégation même direction, même horizon** : les signaux concordants renforcent la confiance (score = fonction du nombre et de la qualité historique des stratégies alignées).
3. **Conflit même direction opposée, même horizon** (rare) : arbitrage par performance historique (sharpe, win rate, score_de_valeur). À défaut d'historique, non-trade.

### 3.4 Risk Manager (allégé)

Version simplifiée de l'Agent Risque de MIDAS. Entrées : intentions + état du portefeuille. Sorties : ordres dimensionnés.

- **Sizing** : risque fixe par trade, ex 1% du capital (paramétrable). Taille = (capital × risque%) / distance SL. Kelly optionnel plus tard.
- **Limites globales** :
  - Exposition max totale (ex 30% du capital).
  - Exposition max par actif et par classe d'actifs (évite 5 positions corrélées crypto).
  - Nombre max de positions simultanées.
  - Perte journalière max → circuit breaker, arrêt des nouvelles entrées jusqu'au lendemain.
  - Drawdown max depuis le plus haut → réduction de taille ou pause.
- **Gestion des positions** : suivi live des SL/TP, trailing optionnel, sortie sur `max_holding`.

### 3.5 Execution Engine

- **Interface unique `BrokerAdapter`** avec deux implémentations :
  - `PaperBroker` : matching simulé sur prix live, slippage et frais paramétrables.
  - `LiveBroker` : CCXT (crypto) puis broker actions selon choix.
- Ordres supportés v1 : market, limit, stop. OCO (bracket TP/SL) si le broker le permet, sinon gestion logicielle.
- Journalisation systématique : signal → intention → ordre → fill, avec latence et slippage.

### 3.6 Portfolio + Journal

- DuckDB local : positions, ordres, fills, equity curve, signaux émis (même non exécutés).
- Le journal des signaux non exécutés est précieux : il mesure ce que le Risk Manager filtre, et alimentera le feedback loop de MIDAS.

### 3.7 Dashboard (phase 2)

- Web léger (backend FastAPI + frontend simple).
- Vues : positions ouvertes, equity curve, signaux en formation (progression partielle des conditions), performance par stratégie, état du circuit breaker.

---

## 4. Boucle de fonctionnement

1. **Scheduler** : à chaque clôture de bougie (5m/15m/1h/4h/1d, +10s de marge), mise à jour incrémentale des données.
2. **Calcul** : indicateurs + patterns sur la fenêtre récente (numba, millisecondes par actif).
3. **Évaluation** : chaque stratégie active évalue ses conditions polars.
4. **Confluence** : partition par horizon, agrégation, arbitrage.
5. **Risque** : sizing, vérification des limites, émission ou rejet.
6. **Exécution** : envoi au broker (paper ou live), pose des SL/TP.
7. **Surveillance continue** (flux live) : suivi des stops, trailing, circuit breaker.
8. **Journal** : tout est écrit en base.

Estimation de charge : 25 actifs × 5 TF × ~50 stratégies actives, conditions vectorisées polars + numba → largement sous la seconde par cycle sur ta machine. Marge énorme.

---

## 5. Plan de screening des stratégies

Objectif : passer de 299 templates (~826 expansions) à un corps de 30 à 60 stratégies actives, robustes et complémentaires.

### Étape 1 — Filtrage structurel (sans calcul)

- Écarter les templates trop dépendants d'une paire spécifique ou d'un contexte macro non mesurable en live (saisonnalité météo, "China stimulus", événements earnings).
- Écarter les stratégies nécessitant des données qu'EINHERJAR n'aura pas (news, cotations obligataires).
- Regrouper les doublons sémantiques.
- Résultat attendu : ~100 à 150 candidates.

### Étape 2 — Backtest de validation

- Données : historique OHLCV des ~25 actifs retenus, 3 à 5 ans, multi-TF.
- Moteur : backtest vectorisé simple (polars), TP/SL natifs de chaque stratégie, frais et slippage réalistes.
- Métriques par stratégie × actif × TF : sharpe, win rate, avg profit, max DD, trade count, profit horizon empirique.
- Seuils (reprenant ton `ValidationConfig`) : sharpe > 1.0, win rate > 45%, min 30 trades, max DD < 25%.

### Étape 3 — Sélection de portefeuille

- Garder les stratégies validées sur plusieurs actifs (généralisabilité), pas un seul coup de chance.
- Dédupliquer par corrélation des signaux : deux stratégies qui entrent au même moment n'apportent qu'une fois l'information.
- Assurer la diversité : trend, reversal, breakout, patterns, quant, cross-asset, plusieurs horizons.
- Résultat : corps de 30 à 60 stratégies actives, chacune avec ses métriques embarquées (utilisées par l'arbitrage et la confiance).

### Étape 4 — Incrémental en live

- Les métriques live remplacent progressivement les métriques de backtest.
- Désactivation automatique d'une stratégie sous son seuil sur N trades réels. Réactivation possible après revue.

---

## 6. Phasage proposé

| Phase | Contenu | Critère de sortie |
|-------|---------|-------------------|
| 0 | Cahier des charges détaillé | Document validé |
| 1 | Market Data Layer + store local + univers 25 actifs | Données multi-TF fraîches en continu |
| 2 | Signal Engine (portage patterns + indicateurs + format stratégie) | Signaux émis sur historique récent |
| 3 | Screening (backtest de validation) | Corps de stratégies actives validé |
| 4 | Risk Manager + PaperBroker + Portfolio/Journal | Paper trading en continu 2-4 semaines |
| 5 | LiveBroker + circuit breakers + durcissement | Premier trade réel, petit capital |
| 6 | Dashboard + observabilité | Suivi temps réel complet |

Les phases 1-2 sont indépendantes et peuvent tourner en parallèle. La phase 3 dépend des deux.

---

## 7. Questions ouvertes à trancher

1. **Premier marché live** : je recommande crypto (Binance, CCXT, pas de friction d'horaires, petits montants possibles dès 10 USD). Les actions viendraient en phase 5b. Accord ?
2. **Données historiques** : as-tu déjà les historiques multi-TF des 99 actifs en local (ceux de la pipeline d'enrichissement) ? Chemins ?
3. **Stratégies short** : les templates incluent des signaux Bear. En crypto spot pas de short ; en actions non plus facilement. On reste **long-only v1** avec signaux Bear = sortie/évitement, ou on prévoit dérivés (futures, CFD) ?
4. **Frais/slippage de référence** pour le screening : je propose 0.1% par côté + slippage 0.05% en crypto. À ajuster selon broker.
5. **Nom du corpus final** : les stratégies retenues pourraient s'appeler les **Einhers** (un guerrier = une stratégie). Cosmétique, mais utile pour le code et les logs.

---

## 8. Stack technique proposée

- Python 3.11+, polars, numpy, numba (réutilisés de MIDAS).
- DuckDB + Parquet (store local).
- CCXT (crypto), interface broker abstraite.
- APScheduler ou boucle asyncio maison pour le scheduler.
- FastAPI + htmx ou React léger pour le dashboard (phase 6).
- pytest pour les tests, ruff pour le lint.
- Aucune dépendance à MIDAS en runtime : le code repris est **copié** dans `einherjar/`, pas importé, pour garantir l'indépendance.
