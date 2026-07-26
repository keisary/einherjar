# Plan de Refonte — Broker Unique cTrader Open API (V3)

> Decision : abandon de MT5 et de tous les adapters multiples. Unique interface cTrader Open API (Spotware) via le cloud.
>
> Objectif : architecture cloud-native, mono-broker, gestion du volume dynamique (levier de compte fixe), resilience integree.

---

## 1. Fichiers a SUPPRIMER

| Fichier | Raison |
|---------|--------|
| `src/einherjar/brokers/binance_adapter.py` | Remplace par cTrader |
| `src/einherjar/brokers/binance_futures_adapter.py` | Remplace par cTrader |
| `src/einherjar/brokers/oanda_adapter.py` | Remplace par cTrader |
| `src/einherjar/brokers/alpaca_adapter.py` | Remplace par cTrader |
| `src/einherjar/brokers/paper_broker.py` | Remplace par le mode demo du dashboard |
| `src/einherjar/brokers/cfd_adapter.py` | Remplace par cTrader |
| `config/fees_binance.json` | Obsolete |
| `config/fees_alpaca.json` | Obsolete |
| `config/fees_oanda.json` | Obsolete |
| `docs/GUIDE_BROKERS.md` | A recrire pour cTrader |
| `docs/PLAN_REFONTE_MT5.md` | Obsolete |

> **Note** : `resilience.py` est conserve. Ses classes `CircuitBreaker` et `RateLimiter` sont composees dans `CTraderAdapter`.

---

## 2. Fichiers a CREER

| Fichier | Role |
|---------|------|
| `src/einherjar/brokers/ctrader_adapter.py` | Adapter unique cTrader — gRPC/protobuf, auth OAuth2, OHLCV, ordres, positions, compte, resilience integree |
| `config/fees_ctrader.json` | Spread + commission + swap overnight par symbole cTrader |
| `docs/GUIDE_CTRADER.md` | Guide unique : ouvrir compte cTrader (IC Markets ou Pepperstone), obtenir les cles API, connexion |
| `docs/PLAN_REFONTE_CTRADER.md` | Ce fichier |
| `scripts/test_ctrader.py` | Script de test de connexion cTrader |

---

## 3. Fichiers a REFONDRE

### 3.1 `src/einherjar/brokers/broker_utils.py`

**Nouveau** :
- Mapping MIDAS → symboles cTrader **par broker** (ic_markets, pepperstone)
  - Les brokers cTrader nomment differemment : `AAPL` peut etre `AAPL.NAS`, `US.AAPL`, etc.
  - Mapping par defaut + override par broker
- Conversion timeframe EINHERJAR → cTrader period (`M5`=5, `M15`=15, `H1`=60, `H4`=240, `D1`=1440)
- Helper `retry_with_backoff` conserve
- Supprimer `MIDAS_TO_BINANCE`, `MIDAS_TO_OANDA`, `MIDAS_TO_ALPACA`
- Ajouter `MIDAS_TO_CTRADER_DEFAULT`, `MIDAS_TO_CTRADER_IC_MARKETS`, `MIDAS_TO_CTRADER_PEPPERSTONE`

### 3.2 `src/einherjar/brokers/ctrader_adapter.py`

**Spec** :
```python
class CTraderAdapter:
    def __init__(self, client_id: str, client_secret: str, access_token: str,
                 account_id: int, host: str = "demo.ctraderapi.com", port: int = 5035,
                 broker_name: str = "ic_markets")
    
    # Connexion / Resilience
    async def connect() -> bool                    # gRPC channel + auth app + auth account
    async def disconnect() -> None                 # Fermeture channel
    async def ensure_connected() -> bool           # Reconnexion auto si deconnecte
    
    # Marche
    async def get_ohlcv(asset, tf, limit=500) -> pl.DataFrame   # ProtoOATrendbarReq
    async def get_tick(asset) -> dict              # ProtoOASymbolByIdReq + prix bid/ask
    
    # Ordres
    async def place_order(order: Order) -> Fill    # ProtoOANewOrderReq avec SL/TP inline, retry x3
    async def close_position(position_id: int) -> bool  # ProtoOAClosePositionReq
    
    # Compte
    async def get_positions() -> list[Position]    # ProtoOAGetPositionListReq
    async def get_account() -> AccountState        # ProtoOAGetAccountListReq
    
    # Frais
    def get_fees(asset: str) -> dict               # Depuis fees_ctrader.json
    
    # Resilience integree
    def get_status() -> dict                       # Circuit state, rate limit, connection, latency
```

**Remarques cTrader** :
- Auth en 2 etapes : `ProtoOAApplicationAuthReq` → `ProtoOAAccountAuthReq`
- Symboles identifies par `symbolId` (int). Cache `symbol_name -> symbolId` apres resolution.
- `get_ohlcv` utilise `ProtoOATrendbarReq` (Trendbars = candles). Retourne OHLCV en protobuf.
- Ordres : `ProtoOANewOrderReq` avec `stopLoss` et `takeProfit` inline (en prix, pas en pips).
- Volume cTrader : en unites de base (pas de lots). 1.0 = 1 unité. Le broker impose des min/max par symbole.
- Levier de compte : lu via API, **non modifiable par code**.
- Connexion gRPC : utiliser `grpc.aio` ou wrapper Twisted. L'adapter utilise un thread interne pour Twisted si la lib `ctrader-open-api` est utilisee, sinon `grpcio` pur avec les stubs protobuf.

### 3.3 `src/einherjar/risk/manager.py`

**Nouvelles regles avec volume dynamique (levier de compte fixe)** :

```python
class RiskManager:
    def _calculate_volume(self, signal: Signal, account: AccountState) -> float:
        """Calcule le volume en unites de base.
        
        Formule : volume = (equity * risk_per_trade) / distance_SL
        Puis plafonne par confiance.
        """
        
    def _check_margin(self, volume: float, entry_price: float, account_leverage: float,
                      margin_available: float) -> bool:
        """Verifie que la marge necessaire tient dans la marge disponible.
        
        marge = (volume * entry_price) / account_leverage
        marge * (1 + margin_buffer_pct) <= margin_available
        """
```

**Supprimer** :
- `MIN_SIZE_MAP` (obsolete, min size varie par symbole cTrader — sera géré par l'adapter)
- `ASSET_CLASS_MAP` deplace dans `broker_utils.py` ou `core/models.py` si partage

**Conserver** :
- `CORRELATION_GROUPS` (toujours utile)
- Circuit breakers journalier/hebdomadaire
- Limites d'exposition

**Ajouter dans `RiskLimits`** (`core/config.py`) :
```python
base_leverage: int = 10          # Valeur lue depuis cTrader, ici valeur par defaut
margin_buffer_pct: float = 0.10  # Marge de securite 10%
```

### 3.4 `src/einherjar/scheduler/loop.py`

**Changements** :
- Supprimer `adapters: dict[str, BrokerAdapter]`, `asset_broker_map: dict[str, str]`
- Remplacer par `broker: CTraderAdapter`
- `_get_adapter(asset)` → `return self.broker`
- `_get_account_for_asset()` → `await self.broker.get_account()`
- `_get_positions_for_asset()` → `await self.broker.get_positions()`
- Ajouter `await self.broker.ensure_connected()` au debut de chaque cycle

### 3.5 `src/einherjar/api/server.py`

**Nouveaux endpoints** :
```
GET /api/account   -> { balance, equity, margin, marginFree, leverage, connected, accountId }
GET /api/health    -> { ctrader: { connected, host, latency, circuitState }, database, config, corpus }
GET /api/overview  -> equity curve + metrics reelles (depuis DataStore si dispo, sinon mock en demo)
GET /api/positions -> positions reelles depuis cTrader
```

**Mode demo** : si cTrader non connecte et mode=demo, utiliser les comptes demo du SettingsContext.

### 3.6 `src/einherjar/core/config.py`

**Ajouter dans `RiskLimits`** :
```python
base_leverage: int = 10
margin_buffer_pct: float = 0.10
```

### 3.7 `main.py`

**Nouveau** :
- Charger credentials cTrader depuis `config/credentials.json` (champs : client_id, client_secret, access_token, account_id, host, broker_name)
- Instancier `CTraderAdapter` avec `connect()` au demarrage
- Si echec connexion → mode demo automatique + warning
- Banniere : `cTrader [ACCOUNT_ID] @ [HOST] | Leverage: X | Equity: $Y`

### 3.8 Dashboard React

| Page | Changement |
|------|-----------|
| `Header.tsx` | Afficher `cTrader — Equity $X / Margin $Y / Levier Zx` |
| `HealthPage.tsx` | Statut cTrader (connecte/deconnecte), circuit breaker, latency, margin info |
| `OverviewPage.tsx` | Levier courant (lu depuis cTrader), marge libre, equity curve reelle |
| `SettingsPage.tsx` | Champs cTrader : client_id, client_secret, access_token, account_id, host (demo/live), broker (dropdown IC Markets / Pepperstone), mode hedging |

---

## 4. Resilience integree dans CTraderAdapter

Composition des classes de `resilience.py` :

```python
class CTraderAdapter:
    def __init__(...):
        self.circuit = CircuitBreaker()
        self.rate_limiter = RateLimiter()
        self._connected = False
        self._last_ping = 0
        
    async def _safe_call(self, method_name: str, *args, **kwargs):
        """Wrapper avec circuit breaker + rate limiter + reconnexion auto."""
        if not self.circuit.can_execute():
            raise RuntimeError("Circuit breaker ouvert")
        await self.rate_limiter.acquire()
        
        if not self._connected or (time.time() - self._last_ping) > 30:
            await self.ensure_connected()
        
        try:
            method = getattr(self, f"_{method_name}")
            result = await method(*args, **kwargs)
            self.circuit.record_success()
            return result
        except Exception as exc:
            self.circuit.record_failure()
            raise
```

---

## 5. Arborescence finale brokers

```
src/einherjar/brokers/
  __init__.py           # Expose CTraderAdapter
  adapter.py            # Interface BrokerAdapter (Protocol)
  broker_utils.py       # Mapping MIDAS→cTrader par broker, helpers
  ctrader_adapter.py    # Adapter unique + resilience integree
  resilience.py         # Classes CircuitBreaker, RateLimiter
```

---

## 6. Sequence de mise en oeuvre

| Etape | Action | Fichiers |
|-------|--------|----------|
| 1 | Supprimer adapters obsoletes | 10 fichiers |
| 2 | Refondre `broker_utils.py` | Mapping cTrader |
| 3 | Creer `ctrader_adapter.py` | Adapter + resilience |
| 4 | Creer `fees_ctrader.json` | Frais |
| 5 | Refondre `risk/manager.py` | Volume dynamique, margin check |
| 6 | Refondre `core/config.py` | base_leverage, margin_buffer_pct |
| 7 | Refondre `scheduler/loop.py` | Mono-adapter cTrader |
| 8 | Refondre `api/server.py` | Endpoints cTrader |
| 9 | Refondre `main.py` | Connexion cTrader au demarrage |
| 10 | Corriger dashboard | Header, Health, Settings |
| 11 | Creer `scripts/test_ctrader.py` | Test connexion |
| 12 | Ecrire `docs/GUIDE_CTRADER.md` | Guide utilisateur |

---

## 7. Questions — Reponses

| Question | Reponse |
|----------|---------|
| Hedging cTrader ? | Oui, supporte. A activer a l'ouverture du compte. |
| Compte demo ? | Oui, obligatoire pour valider avant live. Gratuit chez IC Markets et Pepperstone. |
| Levier modifiable par code ? | **Non**. Levier fixe par le broker (1:30 a 1:500). Lu via API. Le risk manager calcule le volume, pas le levier. |
| Cloud vs local ? | Cloud-native. Connexion gRPC directe a `demo.ctraderapi.com:5035` ou live. Pas de terminal local. |
| Tous les actifs disponibles ? | Oui, via CFD. |
| Timeframes disponibles ? | Oui, 5m/15m/1h/4h/1d tous supportes (period 5, 15, 60, 240, 1440). |

---

**Plan valide. Pret a coder sur ton GO.**
