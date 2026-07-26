# EINHERJAR — Cahier des charges

> Version 0.2 — Document vivant, rempli au fur et à mesure des décisions.
> Dernière mise à jour : section 1 (inférence et brokers).

**Décisions actées**
- Nom de code : EINHERJAR. Une stratégie retenue = un **Einher**.
- Marchés : tous ceux de l'univers MIDAS (crypto, actions, forex, indices, commodités, métaux), dès la conception.
- Frais : modélisation des frais réels par exchange et par classe d'actifs.
- Capital-agnostique : sizing en % du portefeuille.
- Paper trading et live trading sur le même code, switch de configuration.
- Données historiques : fichiers `.npy` optimisés training dans `midasV3/src/data/compiled`.
- Base de features : 107 patterns, 51 indicateurs, 35 features quantitatives.
- Découverte des stratégies : méthodologie définie ici (section 2), stratégies concrètes définies pendant le développement, pas avant.

**Points en attente**
- [x] Short : long-only spot crypto v1 ; short natif OANDA (forex/CFD) ; short crypto via MT5 CFDs (Pepperstone, FP Markets, IC Markets, AvaTrade) en v1.1. MetaTrader 5 est le canal privilegie pour le short crypto.
- [x] Structure exacte des `.npy` de `midasV3/src/data/compiled` : inventaire complete. 7 classes, 99 actifs uniques, 5 TF (5m, 15m, 1h, 4h, 1d). Fichiers par actif : `*_ts.npy` (timestamps int64), `*_X.npy` (features float32, 246 colonnes), `*_Y_dir.npy` (direction int8), `*_Y_hor.npy` (horizon), `*_Y_ret.npy` (returns).
- [ ] Capital de départ et comptes broker ouverts par l'utilisateur.

---

## Section 1 — Inférence et brokers

### 1.1 Principe d'inférence

EINHERJAR n'utilise aucun modèle appris. L'"inférence" est une **évaluation déterministe de règles** sur bougies clôturées. Un cycle d'inférence complet :

1. **Clôture de bougie détectée** (par actif, par timeframe, marge de 10 s après clôture exchange).
2. **Mise à jour incrémentale des données** : append de la bougie au store local.
3. **Recalcul ciblé** : indicateurs et patterns sur la fenêtre nécessaire (numba, numpy en entrée). Aucun recalcul global.
4. **Évaluation des stratégies actives** : chaque Einher évalue sa condition polars sur le dataframe enrichi. Résultat binaire + contexte (niveaux de prix du pattern, ATR, etc.).
5. **Formation des signaux** : signal = actif, direction, timeframe, Einher émetteur, niveaux TP/SL calculés par les règles natives, score de confiance.
6. **Confluence et arbitrage** (voir architecture) : partition par horizon, agrégation, arbitrage par performance.
7. **Risk Manager** : sizing et vérification des limites. Émission d'une intention d'ordre ou rejet journalisé.
8. **Exécution** : envoi au broker actif (paper ou live) via le `BrokerAdapter`.
9. **Journalisation** : tout est écrit en DuckDB, y compris les rejets et les non-trades.

**Contraintes de performance**
- Cycle complet par actif/TF < 1 s sur la machine cible (20 GB RAM, 12 cœurs).
- Parallélisation par actif (processus ou threads selon le GIL et numba).
- Aucune dépendance réseau pendant le calcul : le réseau ne sert qu'à fetch les bougies et envoyer les ordres.

**États d'un Einher en temps réel**
- `idle` : condition non remplie.
- `forming` : conditions partielles remplies (ex 2/3), en attente de confirmation. Exposé au monitoring.
- `triggered` : condition complète, signal émis.
- `in_position` : position ouverte suite à ce signal.
- `cooldown` : période de non-réémission après clôture (paramétrable, évite les signaux en rafale).

### 1.2 Timeframes et séquencement

- Timeframes actifs : 5m, 15m, 1h, 4h, 1d.
- Ordonnanceur : boucle asyncio, réveil aligné sur les clôtures + marge.
- Les actifs actions/indices ne produisent des bougies qu'en heures de marché : l'ordonnanceur respecte les calendriers de chaque place (fournis par le broker ou une table de calendriers interne).
- Crypto/forex : continu (forex fermé le week-end).

### 1.3 Brokers et flux de données par classe d'actifs

Principe : **une interface `BrokerAdapter` unique**, une implémentation par broker. Chaque implémentation fournit : historique OHLCV, flux live, passage d'ordres, état du compte, frais.

| Classe | Broker recommandé | Pourquoi | Données | Paper trading |
|---|---|---|---|---|
| Crypto | **Binance** (CCXT) | Liquidité, API mature, testnet | REST + WebSocket natifs | Testnet Binance |
| Actions US | **Alpaca** | API simple, compte paper natif gratuit, fractionnaire | REST + WebSocket, historique gratuit | Natif |
| Forex + métaux + indices CFD | **OANDA** | API REST/stream propre, petits montants, réputation | REST + streaming prix | Compte practice natif |
| Actions Europe/Asie, commodités | **Interactive Brokers** (phase 5b) | Couverture mondiale unique | API TWS/IB Gateway | Paper account |

**Décision proposée** : v1 = Binance + Alpaca + OANDA (3 adaptateurs, tous avec paper natif). IBKR ajouté en phase 5b pour couvrir le reste de l'univers sans changer l'architecture.

**Alternative étudiée** : IBKR seul pour tout. Rejetée pour la v1 : API lourde (TWS Gateway), pas de crypto spot complet, courbe d'apprentissage élevée. Mais l'interface `BrokerAdapter` est conçue pour l'accueillir ensuite, et c'est aussi l'adaptateur que MIDAS réutilisera.

### 1.4 Modélisation des frais (réels, par venue)

Chaque adaptateur expose une table de frais utilisée partout (screening, paper, sizing) :

- **Crypto Binance spot** : 0.1% maker/taker par défaut (0.075% avec BNB), slippage modélisé 0.05% par défaut, ajustable par paire selon liquidité.
- **Actions US Alpaca** : commission 0 (compte standard), slippage + spread modélisés (0.02-0.05% sur large caps).
- **Forex/CFD OANDA** : pas de commission séparée, coût = spread. Table de spreads moyens par paire (ex EURUSD ~1.2 pips, XAUUSD ~30 cents), révisable.
- **Financement overnight (CFD/forex)** : taux de carry journalier par instrument, appliqué aux positions > 1 jour. Important pour les stratégies à horizon 4h/1d.

Format : fichier `fees.yaml` par broker, chargé au démarrage, surchargeable par actif.

### 1.5 Correspondance univers → brokers

Les 99 actifs de MIDAS sont routés automatiquement :

- `*USD` crypto (BTCUSD, ETHUSD...) → Binance (conversion paire spot, ex BTCUSD → BTC/USDT).
- Actions US (AAPL, NVDA...) → Alpaca.
- Paires forex (EURUSD, GBPJPY...), XAUUSD, XAGUSD → OANDA.
- Indices (SP500, DAX40...), commodités (WTIUSD, BRENT, COCOA...) → OANDA si disponible en CFD, sinon IBKR (phase 5b).
- Actions Europe/Asie (AIR.PA, 7203.T...) → IBKR (phase 5b).

La sélection du top 25 (liquidité × volatilité) se fera en privilégiant les actifs couverts par les 3 brokers v1.

### 1.6 Données historiques

**Structure des `.npy` MIDAS V3** (`midasV3/src/data/compiled/`)
- Organisation : `classe/timeframe/ACTIF_{ts,X,Y_dir,Y_hor,Y_ret}.npy`
- Classes : commodities (12), crypto (9), forex (20), indices (11), stocks_growth (15), stocks_tech (16), stocks_value (16) = **99 actifs uniques**
- Timeframes : 5m, 15m, 1h, 4h, 1d (toutes classes sauf crypto sans 1d)
- `*_ts.npy` : timestamps Unix ms, int64
- `*_X.npy` : features, float32, **246 colonnes** (supérieur aux 193 attendus : patterns 107 + indicateurs 51 + quant 35 = 193 ; les 53 supplémentaires sont des variantes multi-périodes)
- `*_Y_dir.npy` : direction labels, int8, valeurs [-100, 0, 1, 2], shape (n, 4)
- `*_Y_hor.npy` : horizon labels
- `*_Y_ret.npy` : returns, shape (n, 4)

**Top 28 actifs selectionnes** (choix manuel par expertise marche, tous verifies dans les donnees MIDAS V3)

| # | Actif | Classe | Broker v1 | Nom complet |
|---|-------|--------|-----------|-------------|
| 1 | BTCUSD | crypto | Binance | Bitcoin |
| 2 | ETHUSD | crypto | Binance | Ethereum |
| 3 | ADAUSD | crypto | Binance | Cardano |
| 4 | BCHUSD | crypto | Binance | Bitcoin Cash |
| 5 | LTCUSD | crypto | Binance | Litecoin |
| 6 | EURUSD | forex | OANDA | Euro / Dollar US |
| 7 | GBPUSD | forex | OANDA | Livre Sterling / Dollar US |
| 8 | USDJPY | forex | OANDA | Dollar US / Yen |
| 9 | AUDUSD | forex | OANDA | Dollar Australien / Dollar US |
| 10 | USDCAD | forex | OANDA | Dollar US / Dollar Canadien |
| 11 | USDCHF | forex | OANDA | Dollar US / Franc Suisse |
| 12 | EURGBP | forex | OANDA | Euro / Livre Sterling |
| 13 | XAUUSD | forex | OANDA | Or / Dollar US |
| 14 | AAPL | stocks_tech | Alpaca | Apple Inc. |
| 15 | MSFT | stocks_tech | Alpaca | Microsoft Corp. |
| 16 | NVDA | stocks_tech | Alpaca | NVIDIA Corp. |
| 17 | AMZN | stocks_tech | Alpaca | Amazon.com Inc. |
| 18 | GOOGL | stocks_tech | Alpaca | Alphabet Inc. |
| 19 | TSLA | stocks_tech | Alpaca | Tesla Inc. |
| 20 | JPM | stocks_value | Alpaca | JPMorgan Chase |
| 21 | XOM | stocks_growth | Alpaca | Exxon Mobil Corp. |
| 22 | SP500 | indices | OANDA | S&P 500 |
| 23 | NASDAQ100 | indices | OANDA | NASDAQ 100 |
| 24 | DOWJONES | indices | OANDA | Dow Jones Industrial |
| 25 | DAX40 | indices | OANDA | DAX 40 |
| 26 | WTIUSD | commodities | OANDA | WTI Crude Oil |
| 27 | BRENT | commodities | OANDA | Brent Crude Oil |
| 28 | COPPER | commodities | OANDA | Copper |

Repartition : 5 crypto, 8 forex, 8 actions US, 4 indices, 3 commodites. Tous couverts par les 3 brokers v1.
Fichier de reference : `config/assets_v1.json`.

### 1.7 Limites connues de la v1

- Pas de carnet d'ordres (order book) : les stratégies Supply/Demand de MIDAS restent hors scope.
- Pas de short crypto spot en v1 : signaux Bear = sortie de position ou abstention. Short crypto via MT5 CFDs en v1.1.
- Fractionnaire : crypto et actions US OK ; forex en unités OK chez OANDA.
- Slippage modélisé, pas mesuré : ajusté après les premières semaines de paper.

---

## Section 2 — Signaux et stratégies

### 2.1 Principe

Les stratégies concrètes ne sont PAS définies dans ce document. On définit ici :

1. Le **format** d'un Einher.
2. Les **domaines** de stratégies (quelles combinaisons de features sont légitimes).
3. La **méthode de découverte** (comment on trouve et valide les stratégies pendant le développement).

### 2.2 Base de features disponibles

| Famille | Nombre | Source | Nature |
|---|---|---|---|
| Patterns (candlestick + chartistes + harmoniques) | 107 | `numba_pattern_detectors.py` | Signaux exploitables directement, TP/SL souvent natifs |
| Indicateurs techniques | 51 | `technical_indicators.py` | Features continues ou croisements |
| Features quantitatives | 35 | `quantitative_features.py` | Hurst, entropie, vol, autocorr, etc. Certaines directement exploitables |

Deux rôles pour une feature dans un Einher :
- **Trigger** : déclenche le signal (ex : double top détecté, cassure SMA50).
- **Filtre** : conditionne la validité (ex : ADX > 25, régime de vol bas, Hurst > 0.5).

### 2.3 Format d'un Einher

```python
@dataclass
class Einher:
    name: str                    # ex: "E_DoubleTop_ADXfilter_4h"
    domain: str                  # voir 2.4
    direction: str               # 'long' | 'short' | 'both'
    timeframes: list[str]        # TF d'évaluation
    trigger: str                 # condition polars du déclencheur
    filters: list[str]           # conditions polars complémentaires (AND)
    assets: str                  # 'all' | classe d'actifs | liste explicite
    # Sortie
    tp_rule: dict                # natif ou calibré, voir 2.5
    sl_rule: dict
    max_holding: str | None      # durée max en position
    cooldown: str                # délai avant réémission sur le même actif
    # Métriques (remplies par la calibration, section 5)
    sharpe: float = 0.0
    win_rate: float = 0.0
    avg_tp_pct: float = 0.0      # TP moyen observé
    avg_sl_pct: float = 0.0
    trade_count: int = 0
    profit_horizon: str = ''     # horizon empirique dominant
    calibrated_on: str = ''      # période de calibration
```

Tout Einher est sérialisable en JSON/YAML : le corpus vit dans un fichier de config, pas dans le code.

### 2.4 Domaines de stratégies

Un Einher appartient à un domaine, ce qui structure la recherche et garantit la diversité du corpus :

| Domaine | Trigger typique | Filtres typiques | TP/SL |
|---|---|---|---|
| **Pattern pur** | un des 107 patterns | aucun ou régime de vol | natifs du pattern |
| **Pattern + confluence** | pattern | indicateur(s) confirmant (RSI, volume, ADX) | natifs du pattern |
| **Indicateur classique** | croisement/seuil indicateur (SMA, MACD, Ichimoku...) | régime (Hurst, vol, trend) | calibrés empiriquement |
| **Quantitatif** | feature quant en zone extrême (entropie basse, Hurst shift...) | direction donnée par momentum/trend | calibrés empiriquement |
| **Breakout / volatilité** | squeeze BB, Donchian, compression de vol | volume, trend supérieur | mixte |
| **Multi-TF** | setup TF inférieur | direction imposée par TF supérieur | calibrés ou ATR |
| **Cross-asset** | mouvement d'un leader (BTC, SP500, DXY...) | confirmation sur l'actif tradé | calibrés empiriquement |

Règles de construction :
- 1 trigger obligatoire, 0 à 3 filtres maximum (au-delà, surapprentissage quasi certain).
- Un Einher doit avoir une **justification économique ou technique** énonçable en une phrase. Pas de combinaison gratuite.
- Les domaines "Pattern pur" et "Pattern + confluence" viennent en premier dans la recherche : TP/SL natifs, interprétabilité totale.

### 2.5 Détermination des TP/SL

**Cas A — TP/SL natifs** (patterns et setups documentés). Exemples :
- Double top : SL au-dessus du plus haut des deux sommets (+ marge ATR), TP = hauteur du pattern projetée depuis la ligne de cou.
- Épaule-tête-épaule : même logique, hauteur tête/ligne de cou.
- Harmoniques : entrée PRZ, SL au-delà du point X, TP sur ratios 0.382 / 0.618 de AD.
- Cassure de triangle/rectangle : TP = hauteur de la figure projetée, SL de l'autre côté de la borne cassée.
- Chaque famille aura sa règle codée dans une table `native_exits.yaml`, rédigée pendant le développement à partir des règles classiques de l'analyse technique.

**Cas B — TP/SL calibrés empiriquement** (indicateurs, quant, multi-TF, cross-asset) :
1. Le trigger est lancé sur l'historique (données `.npy` de `midasV3/src/data/compiled`).
2. Pour chaque occurrence : récolte du **MFE** (excursion favorable max avant sortie) et du **MAE** (excursion défavorable max), sur une fenêtre de N bougies après le signal.
3. Le couple TP/SL est choisi dans la **zone stable** de la surface de performance (plateau, pas pic isolé). Un couple dont la performance s'effondre pour une variation de ±10% est rejeté.
4. Validation **walk-forward** : calibration sur 70% de l'historique, vérification sur les 30% terminaux. Écart de performance > 50% entre les deux = rejet.
5. Stockage : TP/SL retenus + distribution complète (TP moyen, médian, taux de toucher du TP avant SL, horizon empirique dominant). La distribution alimentera le dashboard.

**Fallback** : si la calibration ne dégage aucune zone stable, l'Einher utilise SL = 1.5 × ATR(14), TP = 2.5 × ATR(14), ou est rejeté.

### 2.6 Méthode de découverte des stratégies

Processus itératif pendant le développement, par domaine :

1. **Deep research** : revue de la littérature trading (règles classiques des patterns, études quantitatives, comportements documentés par classe d'actifs) pour produire une liste d'hypothèses par domaine, chacune avec sa justification en une phrase.
2. **Traduction** : chaque hypothèse devient un Einher candidat (trigger + filtres + règle de sortie Cas A ou B).
3. **Calibration et validation** : pipeline de la section 5 (backtest, walk-forward, seuils de `ValidationConfig` : sharpe > 1.0, win rate > 45%, 30+ trades, max DD < 25%).
4. **Déduplication** : corrélation des signaux entre Einhers validés ; deux Einhers entrant aux mêmes moments sur les mêmes actifs n'en font qu'un.
5. **Admission au corpus** : objectif 30 à 60 Einhers couvrant les 7 domaines et plusieurs horizons.

Le corpus est versionné (`corpus_v1.yaml`, `corpus_v2.yaml`...) : chaque ajout/retrait d'Einher est traçable.

### 2.7 Signaux Bear en v1 (rappel, en attente de validation)

- Crypto spot (Binance) : long-only. Signal Bear = fermeture de la position longue eventuelle + blocage de nouvelles entrees longues sur l'actif pendant le cooldown.
- OANDA (forex, métaux, indices CFD) : short natif autorisé, traité symétriquement au long.
- Crypto short : via MT5 CFD brokers (Pepperstone, FP Markets, IC Markets, AvaTrade) en v1.1. Ces brokers offrent des CFDs crypto avec levier, long et short, sur MT4/MT5.

## Section 3 — Risk Manager

### 3.1 Rôle

Transformer les intentions de trade (signal + TP/SL + confiance) en ordres dimensionnés, ou les rejeter. Version allégée de l'Agent Risque MIDAS : règles fixes, pas de modèle appris.

### 3.2 Sizing

- **Risque fixe par trade** : `risque_trade = capital × risk_per_trade` (défaut 1%, paramétrable 0.25% - 2%).
- **Taille de position** : `quantité = risque_trade / distance_SL` où `distance_SL = |prix_entrée − prix_SL|`.
- Plafond additionnel par confiance : taille × score_confiance (0.5 à 1.0). Un signal faible prend une demi-position.
- Arrondi aux lots de la venue (fractionnaire crypto/Alpaca, unités OANDA). Si la taille minimale de la venue > risque calculé, le trade est rejeté (protection petit capital).

### 3.3 Limites globales

| Limite | Défaut | Effet en cas d'atteinte |
|---|---|---|
| Exposition totale | 60% du capital | Blocage nouvelles entrées |
| Exposition par actif | 20% du capital | Blocage sur cet actif |
| Exposition par classe (crypto, forex...) | 35% du capital | Blocage sur la classe |
| Positions simultanées | 15 | Blocage nouvelles entrées |
| Corrélation | max 3 positions sur actifs corrélés > 0.8 (30j) | Rejet de la nouvelle entrée |
| Perte journalière | 5% du capital | **Circuit breaker** : stop entrées jusqu'au lendemain |
| Drawdown depuis plus haut | 15% : tailles ÷ 2 ; 25% : pause complète | Réduction / arrêt |
| Perte hebdomadaire | 10% | Revue manuelle requise |

### 3.4 Gestion des positions ouvertes

- SL/TP posés dès l'entrée (ordre bracket/OCO si la venue le permet, sinon surveillance logicielle sur flux live).
- **Trailing** optionnel par Einher : activation quand le prix atteint 50% du TP, trailing = 1 × ATR.
- Sortie sur `max_holding` atteint (fermeture au marché).
- Signal Bear opposé sur l'actif (long-only) : fermeture immédiate de la position longue.
- Circuit breaker : les positions ouvertes gardent leurs SL/TP (jamais de position nue sans protection).

### 3.5 Journalisation des rejets

Chaque intention rejetée est écrite avec la règle ayant bloqué. Permet de mesurer si les limites sont trop strictes ou trop lâches, et servira à calibrer l'Agent Risque de MIDAS.

## Section 4 — Exécution et portfolio

### 4.1 BrokerAdapter (interface)

```python
class BrokerAdapter(Protocol):
    def get_ohlcv(asset, timeframe, since) -> pl.DataFrame
    def subscribe_live(assets, callback)            # prix temps réel
    def place_order(order: Order) -> Fill
    def cancel_order(order_id)
    def get_positions() -> list[Position]
    def get_account() -> AccountState               # cash, equity, marge
    def get_fees(asset) -> FeeSchedule
```

Implémentations : `PaperBroker` (simulation sur prix live, slippage/frais de `fees.yaml`), `BinanceAdapter`, `AlpacaAdapter`, `OandaAdapter`. IBKR en phase 5b.

### 4.2 Ordres supportés v1

- Market, limit, stop-market.
- Bracket OCO (TP + SL attachés) si la venue le supporte nativement (Binance OCO, Alpaca bracket, OANDA OCO) ; sinon gestion logicielle avec surveillance live < 1 s.
- Pas d'ordres iceberg/TWAP en v1 : tailles trop petites pour en avoir besoin.

### 4.3 PaperBroker

- Matching : market = dernier prix + slippage ; limit/stop = toucher du prix sur flux live.
- Frais appliqués depuis `fees.yaml` (identiques au live visé).
- Latence simulée configurable (défaut 200 ms).
- Objectif : le paper doit être **pessimiste** (slippage légèrement surestimé) pour que le live surprenne positivement.

### 4.4 Portfolio et journal (DuckDB)

Tables : `signals` (tous, exécutés ou non), `orders`, `fills`, `positions`, `equity_curve` (snapshot 5m), `rejections`, `einher_stats` (performance glissante par Einher).

Réconciliation : au démarrage et toutes les heures, comparaison positions locales vs broker. Écart = alerte + mode dégradé (pas de nouvelle entrée jusqu'à résolution).

### 4.5 Robustesse

- Reprise sur crash : état reconstruit depuis DuckDB + broker au redémarrage.
- Perte du flux live : fallback polling REST, alerte si > 60 s sans données.
- Horodatage UTC partout, stockage des prix en float64.

## Section 5 — Screening et validation

### 5.1 Données

- Source primaire : `.npy` de `midasV3/src/data/compiled` (structure à inventorier : colonnes, TF, profondeur).
- Complément : API brokers si actifs ou TF manquants.
- Split : 70% calibration / 30% validation (walk-forward), jamais de fuite.

### 5.2 Pipeline de backtest

- Moteur vectorisé polars : un Einher est évalué sur tout l'historique en une passe.
- Simulation réaliste : entrée à la bougie suivant le signal (prix d'ouverture), frais `fees.yaml` + slippage, gestion SL/TP/trailing/max_holding bougie par bougie.
- Sortie par Einher × actif × TF : trade count, win rate, sharpe, avg profit, max DD, MFE/MAE, horizon empirique.

### 5.3 Seuils d'admission

Reprise de `ValidationConfig` : sharpe > 1.0, win rate > 45%, ≥ 30 trades, max DD < 25%, avg profit > 0.05%. Plus : stabilité walk-forward (écart perf calibration/validation < 50%) et généralisabilité (valide sur ≥ 3 actifs de sa classe, sauf domaine cross-asset).

### 5.4 Déduplication et composition du corpus

- Matrice de corrélation des séries de signaux entre Einhers validés. Corrélation > 0.7 = on garde le meilleur sharpe.
- Quotas souples par domaine pour garantir la diversité : aucun domaine > 40% du corpus.
- Objectif : 30 à 60 Einhers, 7 domaines couverts, horizons variés.
- Corpus versionné YAML + rapport de screening (métriques, distributions MFE/MAE, matrices).

### 5.5 Maintenance en live

- Fenêtre glissante 50 trades : Einher sous ses seuils → `probation` → désactivation si pas de redressement sur 20 trades.
- Revue mensuelle : recalibration possible, rapport de dérive backtest vs live.

## Section 6 — Dashboard et observabilité

### 6.1 Stack

FastAPI (backend, sert l'API JSON) + frontend React. Lecture seule sur DuckDB. Aucun ordre passe depuis le dashboard en v1 (lecture + bouton kill switch uniquement).

**Design UI/UX -- Theme Nordique "Einherjar"**
- Palette : noir profond (#0A0A0A), blanc casse (#F5F5F0), gris pale (#B0B0B0), accents gris froid (#6B7280).
- Typographie : police runique/inspiree nordique pour les titres (ex : Cinzel, Norse, ou Runic via Google Fonts), monospace clair pour les donnees (JetBrains Mono ou IBM Plex Mono).
- Ambiance : minimaliste, froid, epuree. Fond noir, cards gris tres sombres, bordures fines gris froid.
- Animations : transitions fluides CSS (fade, slide, scale 0.3s ease-out), jauges et graphiques avec animation de remplissage progressive, skeleton loaders.
- Iconographie : runes simplifiees pour les etats (idle, forming, triggered, in_position), haches stylisees comme motif decoratif discret.
- Responsive : grille adaptable, priorite aux metriques temps reel sur mobile.

### 6.2 Vues

1. **Vue d'ensemble** : equity curve, P&L jour/semaine, exposition par classe, état circuit breaker.
2. **Positions ouvertes** : actif, Einher, entrée, SL/TP, P&L latent, jauge de progression vers TP/SL.
3. **Signaux en formation** : Einhers en état `forming` avec conditions remplies (2/3) et ce qui manque. C'est la vue demandée pour anticiper les déclenchements.
4. **Performance par Einher** : table triable (trades, win rate, sharpe live vs backtest, état actif/probation).
5. **Journal** : signaux, ordres, rejets avec raisons.
6. **Santé système** : fraîcheur des données par actif/TF, latence cycles, erreurs broker.

### 6.3 Alertes

Logs structurés (JSON) + alertes sur : circuit breaker, réconciliation en écart, flux données mort, Einher désactivé. Canal : fichier + console en v1, notification externe (Telegram/email) en option.

## Section 6.5 -- Philosophie de developpement et conventions de code

**Format de configuration privilegie : JSON**
Tout fichier de configuration (corpus, fees, native_exits, settings) est au format JSON.
Structure : objet racine, cles en snake_case, pas de commentaires inline (utiliser un champ `_comment` si necessaire).

**Documentation**
Chaque module Python commence par une docstring multiligne decrivant :
- Son role dans l'architecture.
- Ses entrees et sorties principales.
- Ses dependances.
Chaque classe et fonction publique a une docstring (Google style ou numpy style).

**Nommage des variables et fonctions**
Convention snake_case pour tout le code Python.
Les noms doivent etre explicites et suivre une logique de domaine :
- `einher_*` : objets lies aux strategies (ex: `einher_name`, `einher_trigger`).
- `risk_*` : objets lies au risk manager (ex: `risk_per_trade`, `risk_limit_daily`).
- `broker_*` : objets lies aux adaptateurs broker (ex: `broker_adapter`, `broker_fee_taker`).
- `market_*` : objets lies aux donnees de marche (ex: `market_ohlcv`, `market_atr_14`).
- Les indicateurs techniques portent leur nom canonique (ex: `sma_50`, `rsi_14`, `macd_line`).
- Les constantes globales sont en UPPER_SNAKE_CASE dans un module `config.py` dedie.

**Structure des fichiers JSON**
- Corpus : `corpus_v{N}.json` -- liste d'objets Einher avec tous les champs definis en section 2.3.
- Fees : `fees_{broker}.json` -- objet avec cles par asset_class, puis par symbole.
- Native exits : `native_exits.json` -- objet avec cles pattern_family, valeurs = regle de sortie.

---

## Section 7 — Plan de développement

| Phase | Livrable | Critère d'acceptation |
|---|---|---|
| 0 | Ce cahier des charges complet | Validé par l'utilisateur |
| 1 | Market Data Layer : adaptateurs données (Binance, Alpaca, OANDA), store Parquet/DuckDB, inventaire `.npy`, sélection top 25 | Historiques + live OK pour les 25 actifs, 5 TF |
| 2 | Signal Engine : portage patterns/indicateurs, format Einher, native_exits.yaml (domaines Pattern) | Signaux rejoués sur historique récent, états forming exposés |
| 3 | Pipeline de calibration/backtest + deep research par domaine + corpus v1 | Corpus v1 validé walk-forward, rapport de screening. Objectif minimum : 30-60 Einhers. Potentiel réel : 300+ compte tenu des 107 patterns et de leurs combinaisons. Le nombre exact sera mesuré, pas inventé. |
| 4 | Risk Manager + PaperBroker + Portfolio/Journal | Paper trading 2-4 semaines sans incident, rejets journalisés |
| 5 | Adaptateurs live (Binance puis Alpaca/OANDA) + durcissement | Premiers trades réels petit capital, réconciliation OK |
| 5b | IBKR (actions EU/Asie, commodités restantes) | Couverture complète de l'univers |
| 6 | Dashboard + alertes | Suivi temps réel complet, vue forming |
| 7 | Rétroportage MIDAS : adaptateurs, risk manager, journal | MIDAS réutilise les briques |

Ordre : phases 1 et 2 en parallèle ; 3 dépend de 1+2 ; 4-6 séquentielles.

**Contraintes de ressources (objectifs, pas estimations inventées)**
- RAM en exploitation : objectif < 1 GB en cycle live. Légèreté = priorité architecturale.
- CPU : cycle d'inférence complet par actif/TF mesuré empiriquement, pas prédit. Cible de confort = < 1 s, mais sera ajustée sur données réelles.
- Stockage : Parquet par actif/TF, métadonnées DuckDB. Pas de cache RAM agressif.
- Aucun chiffre de charge machine n'est affiché tant qu'il n'est pas mesuré sur la cible réelle.

**Stack** : Python 3.11+, polars, numpy, numba, DuckDB, CCXT, APScheduler/asyncio, FastAPI, pytest, ruff. Code repris de MIDAS copié dans `einherjar/`, aucune dépendance runtime à MIDAS.
