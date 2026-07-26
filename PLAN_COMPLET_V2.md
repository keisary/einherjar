# PLAN DE TRAVAIL COMPLET -- EINHERJAR v2.0

> Date : 2026-07-22
> Objectif : Transformer EINHERJAR d'un prototype architectural en systeme de trading operationnel et rentable
> Philosophie : Rentabilite avant tout. Pas de "joli sur GitHub". Un seul but : gagner de l'argent.

---

## SOMMAIRE DES PROBLEMES IDENTIFIES

### P1 -- CRITIQUE : Le backtest est faux
- Simulation dupliquee dans calibrator.py (trades doubles)
- Entree au close au lieu de l'open (look-ahead bias)
- Cooldown arbitraire empeche toute diversification temporelle
- max_holding en nombre de barres = absurde (ignore la volatilite)
- Direction "both" simule long ET short sur la meme bougie (incoherent)
- Pas de gestion de positions multiples sur le meme actif

### P2 -- CRITIQUE : Les TP/SL natifs n'existent pas
- native_exits.json contient des descriptions textuelles
- Le code utilise TOUJOURS des multiples ATR
- PatternBridge ne retourne pas les niveaux de structure (neckline, PRZ, etc.)
- Aucun calcul de pattern_height, beyond_structure, fibonacci_ratio

### P3 -- CRITIQUE : 7535 Einhers = data mining massif
- _gen_corpus_brut_v2.py genere des combinaisons mecaniques
- Aucune validation statistique (p-value) sur les edges
- Le calibrator teste des strategies sans verifier si l'edge est significatif
- Surapprentissage garanti

### P4 -- CRITIQUE : Le confluence n'existe pas
- Aucun module d'aggregation de signaux
- Le Risk Manager recoit des signaux individuels et en rejette 9 sur 10
- Au lieu de renforcer la confiance par agrégation, on détruit l'information

### P5 -- BLOQUANT : cTrader ne fonctionne pas
- credentials.json est vide
- Resolution symbolique par hash (jamais en production)
- Aucun compte demo configure

### P6 -- BLOQUANT : Le dashboard est 100% mock
- Toutes les données affichees sont inventees (_mock_* dans server.py)
- Aucune connexion au DataStore reel
- L'utilisateur ne peut pas voir l'etat reel du systeme

### P7 -- BUG : Code duplique / incoherent
- _check_ctrader dupliquee dans main.py
- AccountState champs dupliques dans models.py
- _load_credentials dupliquee dans server.py

---

## PHASE 1 : FONDATIONS -- Nettoyage et réparation (2-3 jours)

### 1.1 Corriger les bugs de code
**Fichiers** : main.py, core/models.py, api/server.py
- [ ] Supprimer la methode _check_ctrader dupliquee (main.py)
- [ ] Corriger les champs dupliques dans AccountState (models.py)
- [ ] Nettoyer _load_credentials dupliquee (server.py)
- [ ] Lancer ruff/mypy sur tout le codebase et corriger

**Critere** : `python main.py` s'execute sans erreur, tous les composants passent le status check.

### 1.2 Configurer cTrader
**Fichiers** : config/credentials.json (manuel par utilisateur), docs/GUIDE_CTRADER.md
- [ ] Ouvrir compte demo IC Markets cTrader
- [ ] Creer application Open API (client_id, client_secret)
- [ ] Generer access_token
- [ ] Remplir credentials.json
- [ ] Tester avec scripts/test_ctrader.py
- [ ] Si ctrader-open-api indisponible, implementer fallback REST polling

**Critere** : Le script test_ctrader.py affiche balance/equity/marge reelles du compte demo.

### 1.3 Implementer la resolution symbolique cTrader
**Fichiers** : src/einherjar/brokers/ctrader_adapter.py
- [ ] Charger la liste complete des symboles via ProtoOAGetSymbolsReq
- [ ] Construire un cache symbol_name -> symbol_id (int) reel
- [ ] Mapper les symboles MIDAS vers les symboles cTrader par broker
- [ ] Tester la resolution sur EURUSD, BTCUSD, AAPL

**Critere** : `normalize_symbol("BTCUSD", "ic_markets")` retourne un symbol_id valide confirmé par l'API.

---

## PHASE 2 : DISCOVERY ENGINE -- Trouver les edges reels (5-7 jours)

C'est la piece maitresse. On remplace la generation brute force par une decouverte statistique guidee.

### 2.1 Creer le module DiscoveryEngine
**Fichier** : src/einherjar/research/discovery_engine.py

**Entree** : Les .npy MIDAS (X, Y_dir, Y_hor, Y_ret) pour chaque actif/TF
**Sortie** : Une table "feature_edges.json" contenant pour chaque feature :
- edge_score : le return moyen quand la feature est active
- win_rate : % de fois ou la direction predite est correcte
- p_value : significativite statistique (t-test)
- sharpe_per_trade : ratio mean/std des returns
- mfe_50, mfe_75, mfe_90 : percentiles MFE (pour calibrer TP)
- mae_50, mae_75, mae_90 : percentiles MAE (pour calibrer SL)
- n_occurrences : nombre de fois ou la feature s'est declenchee
- best_horizon : horizon (1,2,3,4) ou l'edge est maximal

**Algorithme** :
```
Pour chaque actif, chaque TF :
    Charger X.npy, Y_dir.npy, Y_ret.npy
    Pour chaque feature f dans X :
        Determiner les indices ou f est active
            (pattern == 1, indicateur > seuil, quant > seuil...)
        Pour chaque horizon h dans [1,2,3,4] :
            Collecter Y_ret[idx, h] pour tous les idx actifs
            Calculer edge_score = mean(Y_ret)
            Calculer win_rate = mean(Y_dir == direction)
            Calculer p_value (t-test vs 0)
            Collecter MFE/MAE sur fenetre de h bougies
    Sauvegarder resultats
```

**Seuils de feature** (a calibrer) :
- Patterns : == 1 (detecte)
- RSI : < 30 (oversold long) ou > 70 (overbought short)
- MACD : histogram > 0 ou croisement
- Hurst : > 0.6 (trending) ou < 0.4 (mean-reverting)
- ADX : > 25 (trend fort)
- Bollinger %b : < 0.05 (bas de bande) ou > 0.95 (haut)
- Entropie : < seuil (predictible) ou > seuil (chaotique)

### 2.2 Interaction Mining -- trouver les combinaisons valides
**Fichier** : src/einherjar/research/interaction_miner.py

Pour chaque feature trigger t dont p_value < 0.05 :
    Pour chaque feature filtre f :
        Condition A : t active ET f active -> edge_A
        Condition B : t active ET f inactive -> edge_B
        Si edge_A > edge_B * 1.2 ET p_value_A < 0.05 :
            Interaction valide : (t, f) retient
        Sinon : t seul suffit

**Sortie** : "interactions.json" avec les paires (trigger, filtres) validees

### 2.3 Construire les Einhers a partir des edges
**Fichier** : src/einherjar/research/einher_builder.py

Pour chaque edge valide (trigger + filtres optionnels) :
    Construire un Einher avec :
        trigger : condition polars du trigger
        filters : conditions des filtres valides (max 2)
        tp_rule : {type: "mfe_calibrated", percentile: 75}
            (TP = entry + mfe_75 pour long, entry - mfe_75 pour short)
        sl_rule : {type: "mae_calibrated", percentile: 90}
            (SL = entry - mae_90 pour long, entry + mae_90 pour short)
        direction : deduite de l'edge (edge positive = long, negative = short)
        timeframes : [le TF teste]
        assets : [l'actif teste]  (pas "all" -- on evite la generalisation forcee)
        sharpe, win_rate : extraits de l'edge
        profit_horizon : l'horizon h optimal
        trade_count : n_occurrences
        calibrated_on : periode des donnees MIDAS

**Sortie** : corpus_discovery.json (~50-200 Einhers, pas 7535)

### 2.4 Diversification par actif/TF
**Regle** : Au moins 3 Einhers par actif, sur des TF differents. Pas de concentration sur un seul actif.

**Critere de succes Phase 2** :
- Le DiscoveryEngine produit un corpus de 50-200 Einhers
- Chaque Einher a un p-value < 0.05 sur son actif/TF
- Les TP/SL sont calibres sur MFE/MAE reels, pas des multiples ATR arbitraires
- Le corpus couvre au moins 15 actifs sur 3 classes differentes

---

## PHASE 3 : PORTFOLIO CALIBRATOR -- Backtest realiste (4-5 jours)

### 3.1 Refondre le calibrator
**Fichier** : src/einherjar/backtest/portfolio_calibrator.py (nouveau, remplace calibrator.py)

**Principes** :
- Portfolio-centric : simule le portefeuille entier, pas des trades isolés
- Entree au prix d'ouverture de la barre suivante (pas au close)
- Chaque position a son propre ID, son propre SL/TP, sa propre gestion
- Pas de cooldown entre trades independants sur le meme actif
- max_holding = consequence du TP/SL, pas une contrainte externe
- Gestion des positions ouvertes : a chaque bougie, verifier SL/TP pour TOUTES les positions

**Architecture du simulateur** :
```python
class PortfolioSimulator:
    def __init__(self, einhers, initial_capital, fees):
        self.capital = initial_capital
        self.equity = [initial_capital]
        self.positions = []  # list[OpenPosition]
        self.closed_trades = []
        
    def step(self, df_row):
        # 1. Evaluer les Einhers sur la bougie courante
        signals = self.evaluate_einhers(df_row)
        
        # 2. Mettre a jour les positions ouvertes (SL/TP)
        for pos in self.positions:
            if pos.sl_hit(df_row): self.close_position(pos, "SL")
            elif pos.tp_hit(df_row): self.close_position(pos, "TP")
            elif pos.max_holding_expired(df_row): self.close_position(pos, "TIME")
        
        # 3. Risk Manager : filtrer les signaux
        valid_signals = self.risk_manager.filter(signals, self.positions, self.capital)
        
        # 4. Ouvrir les nouvelles positions
        for sig in valid_signals:
            self.open_position(sig)
        
        # 5. Snapshot equity
        self.equity.append(self.calculate_equity())
```

### 3.2 Walk-forward validation
**Regle** : 70% des donnees pour calibrer les edges (Phase 2), 30% terminaux pour valider le portfolio.

**Si le Sharpe du portfolio sur les 30% est < 50% du Sharpe sur les 70%** : le systeme est overfit. Retour à Phase 2 avec des seuils plus stricts.

### 3.3 Métriques portfolio
- Sharpe annualise (daily equity curve)
- Max drawdown
- Win rate
- Profit factor
- Calmar ratio (return / max DD)
- Nombre de trades par mois
- Correlation entre actifs dans le portfolio

**Critere de succes Phase 3** :
- Sharpe annualise > 1.0 sur les 30% validation
- Max drawdown < 20%
- Au moins 10 trades par mois en moyenne
- Le Calmar ratio > 1.5

---

## PHASE 4 : CONFLUENCE ENGINE -- Agréger les signaux (3-4 jours)

### 4.1 Creer le module ConfluenceEngine
**Fichier** : src/einherjar/core/confluence.py

**Role** : Transformer N signaux bruts en M intentions agrégées (M <= N, typiquement M = nombre d'actifs actifs).

**Algorithme** :
```python
class ConfluenceEngine:
    def aggregate(self, signals: list[Signal]) -> list[ConfluenceCluster]:
        # 1. Grouper par (actif, direction)
        groups = defaultdict(list)
        for sig in signals:
            groups[(sig.asset, sig.direction)].append(sig)
        
        # 2. Pour chaque groupe, calculer un score composite
        clusters = []
        for (asset, direction), sigs in groups.items():
            score = self._composite_score(sigs)
            # Diversite des domaines : bonus si plusieurs types d'Einher s'alignent
            domains = set(s.einher_domain for s in sigs)
            diversity_bonus = min(len(domains) / 3.0, 1.0)
            
            # Horizon dominant
            horizons = [s.profit_horizon for s in sigs]
            dominant_horizon = max(set(horizons), key=horizons.count)
            
            cluster = ConfluenceCluster(
                asset=asset,
                direction=direction,
                score=score * (0.8 + 0.2 * diversity_bonus),
                tp_price=self._weighted_tp(sigs, score),
                sl_price=self._weighted_sl(sigs, score),
                confidence=min(score, 1.0),
                contributing_einhers=[s.einher_name for s in sigs],
            )
            clusters.append(cluster)
        
        return clusters
```

### 4.2 Rebrancher le pipeline
**Fichiers** : scheduler/loop.py, risk/manager.py

**Nouveau flux** :
```
InferenceLoop -> EinherEngine.evaluate() -> liste de Signaux
    -> ConfluenceEngine.aggregate() -> liste de ConfluenceClusters
    -> RiskManager.evaluate(cluster) -> Order ou Rejection
    -> BrokerAdapter.place_order() -> Fill
```

**Critere de succes Phase 4** :
- Quand 3+ Einhers s'alignent sur un actif, le score de confiance > 0.8
- Quand 1 seul Einher se declenche, le score < 0.6 (faible confiance)
- Le Risk Manager recoit des clusters, pas des signaux individuels

---

## PHASE 5 : LIVE CTRADER -- Connexion et execution (3-4 jours)

### 5.1 Finaliser l'adapter cTrader
**Fichier** : src/einherjar/brokers/ctrader_adapter.py

- [ ] Resolution symbolique reelle (pas de hash)
- [ ] Gestion des erreurs gRPC (reconnexion automatique)
- [ ] Ordres avec SL/TP inline (ProtoOANewOrderReq avec stopLoss/takeProfit)
- [ ] Volume en unites de base (pas de lots) -- adapter au broker
- [ ] Récupération des fills et mise à jour des positions locales

### 5.2 Paper trading loop
**Fichier** : main.py (mode demo avec _MockBrokerAdapter ameliore)

- [ ] Le mock broker doit simuler des bougies réelles (chargées depuis les .npy ou via API)
- [ ] Les positions ouvertes doivent evoluer avec les prix reels
- [ ] Le P&L latent doit etre calcule a chaque bougie
- [ ] Le DataStore doit etre peuple avec de vraies donnees

### 5.3 Test end-to-end
**Script** : scripts/test_pipeline_live.py

```
1. Lancer main.py en mode demo
2. Laisser tourner 24h (ou simuler 24h sur historique rapide)
3. Verifier que :
   - Les signaux sont émis
   - Les ordres sont passés (mock)
   - Les positions sont trackées
   - Le journal DuckDB est coherent
   - Le dashboard affiche des donnees reelles
```

**Critere de succes Phase 5** :
- 24h de paper trading sans crash
- Le dashboard affiche les positions reelles (meme si mock)
- Le journal contient des signaux, ordres, fills coherents

---

## PHASE 6 : DASHBOARD REEL -- Connecter les donnees (2-3 jours)

### 6.1 Connecter l'API au DataStore
**Fichier** : src/einherjar/api/server.py

Remplacer TOUTES les fonctions _mock_* par des requetes DuckDB :

- `/api/overview` -> equity_curve depuis DataStore, positions reelles
- `/api/positions` -> positions depuis broker + DataStore
- `/api/forming` -> Einhers en etat FORMING depuis EinherEngine
- `/api/performance` -> einher_stats depuis DataStore
- `/api/journal` -> signals + orders + rejections depuis DataStore
- `/api/health` -> etat reel des composants

### 6.2 Ajouter le polling WebSocket ou SSE
**Fichier** : dashboard/einherjar-ui/src/hooks/useData.ts

- Remplacer les fetch simples par du polling toutes les 5s
- Ajouter un etat "live" vs "stale" si pas de donnees depuis 30s

### 6.3 Ajouter le kill switch
**Fichier** : api/server.py + dashboard

- Endpoint POST /api/kill_switch qui met le systeme en pause
- Bouton rouge dans le header du dashboard
- En pause : pas de nouveaux ordres, positions gardent leurs SL/TP

**Critere de succes Phase 6** :
- Le dashboard affiche les donnees reelles du systeme
- Les metriques s'actualisent toutes les 5 secondes
- Le kill switch fonctionne

---

## PHASE 7 : OPTIMISATION ET ROBUSTESSE (3-4 jours)

### 7.1 Performance
- [ ] Profiler le cycle d'inference (< 1s par actif/TF)
- [ ] Optimiser le FeatureEngine (cache numpy, eviter les copies)
- [ ] Paralleliser l'evaluation par actif (asyncio.gather)

### 7.2 Robustesse
- [ ] Reprise sur crash : reconstruire l'etat depuis DuckDB
- [ ] Fallback si cTrader deconnecte : passer en mode "watch only"
- [ ] Alertes Telegram/Discord sur circuit breaker, erreurs, deconnexion

### 7.3 Logging
- [ ] Tous les evenements en JSON structure
- [ ] Rotation des logs (pas de fichier de 10GB)
- [ ] Niveaux : DEBUG (feature calc), INFO (signals), WARNING (rejets), ERROR (crash)

---

## PHASE 8 : GO LIVE -- Premier capital reel (1-2 jours)

### 8.1 Checklist pre-live
- [ ] Compte cTrader live ouvert et finance (petit capital : 500-1000 USD)
- [ ] credentials.json pointe vers live.ctraderapi.com
- [ ] Hedging active sur le compte
- [ ] Levier connu et configure dans RiskLimits
- [ ] Paper trading 2 semaines consecutives sans incident
- [ ] Sharpe paper > 0.5 (realiste, pas optimiste)

### 8.2 Parametres conservateurs
- risk_per_trade = 0.5% (pas 1%)
- max_positions = 5 (pas 15)
- exposure_total_pct = 0.30 (pas 0.60)
- daily_loss_pct = 0.02 (pas 0.05)

### 8.3 Monitoring intense
- Observer les premiers trades en temps reel
- Comparer fills paper vs fills live (slippage reel)
- Ajustement des frais si necessaire

---

## ORDRE D'EXECUTION RECOMMANDE

```
Semaine 1 :
  J1-2 : Phase 1 (bugs + cTrader config + resolution symboles)
  J3-5 : Phase 2 (DiscoveryEngine + edges + corpus_discovery.json)

Semaine 2 :
  J1-2 : Phase 3 (Portfolio Calibrator + walk-forward)
  J3-4 : Phase 4 (ConfluenceEngine + rebranchage)
  J5   : Phase 5 (cTrader live + paper trading 24h)

Semaine 3 :
  J1-2 : Phase 6 (Dashboard reel)
  J3-4 : Phase 7 (Optimisation + robustesse)
  J5   : Phase 8 (Go live avec capital minimal)
```

---

## ESTIMATION DES RISQUES

| Risque | Probabilite | Impact | Mitigation |
|--------|-------------|--------|------------|
| Les edges decouverts ne sont pas significatifs | Moyenne | ELEVE | Tester sur plusieurs actifs, ajuster les seuils |
| cTrader API instable en live | Moyenne | MOYEN | Circuit breaker + mode degrade |
| Overfitting du portfolio | ELEVEE | ELEVE | Walk-forward strict, pas d'optimisation des parametres |
| Slippage reel > slippage paper | ELEVEE | MOYEN | Commencer avec des ordres limit, pas market |
| Bug en production | Moyenne | ELEVE | Kill switch, capital minimal, hedging |

---

## METRIQUES DE SUCCES FINALES

| Metrique | Cible EINHERJAR | Cible MIDAS V3 |
|----------|-----------------|----------------|
| Sharpe Ratio | > 1.0 | > 1.5 |
| Win Rate | > 55% | > 60% |
| Max Drawdown | < 20% | < 15% |
| ROI Annuel | > 15% | > 20% |
| Trades/mois | > 10 | -- |
| Uptime | > 99% | -- |

> Note : Les cibles EINHERJAR sont plus conservatrices car c'est un systeme de regles deterministes, pas un modele ML. Un Sharpe > 1.0 sur un systeme a regles est deja exceptionnel.

---

**Document valide. Pret a executer.**
