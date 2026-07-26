# Guide cTrader Open API — EINHERJAR

> Ce guide explique comment obtenir les credentials necessaires pour connecter
> EINHERJAR a un compte cTrader (IC Markets ou Pepperstone) via l'Open API.

---

## 1. Choisir un broker cTrader

EINHERJAR supporte les brokers proposant des comptes **cTrader** :

| Broker | Site | Compte demo | Hedging |
|--------|------|-------------|---------|
| **IC Markets** | https://icmarkets.com | Gratuit | Oui (a activer) |
| **Pepperstone** | https://pepperstone.com | Gratuit | Oui (a activer) |

> **Recommandation** : commencez obligatoirement par un **compte demo** pour
> valider la connexion et l'execution avant tout passage en live.

---

## 2. Creer un compte cTrader

1. Rendez-vous sur le site du broker choisi.
2. Ouvrez un compte de trading (demo ou live).
3. Dans le portail client, accedez a la section **cTrader**.
4. Notez votre **ctidTraderAccountId** (ID du compte de trading, ex: `1234567`).

---

## 3. Enregistrer une application Open API

1. Allez sur le **cTrader Open API Playground** :
   - Demo : https://openapi.ctrader.com/apps/demo
   - Live : https://openapi.ctrader.com/apps/live
2. Connectez-vous avec votre compte cTrader.
3. Cliquez sur **Create App**.
4. Remplissez les informations (nom, description, redirect URI : `http://localhost`).
5. Une fois creee, vous obtenez :
   - **Client ID** (54 caracteres)
   - **Client Secret** (50 caracteres)

> Conservez ces deux cles precieusement. Elles identifient votre application
> aupres des serveurs Spotware.

---

## 4. Generer un Access Token

1. Dans la page de votre application (Playground), allez dans l'onglet
   **Tokens** ou **OAuth**.
2. Generez un **Access Token** pour votre compte de trading.
3. Le token est une chaine de 43 caracteres.

> Le token expire apres un certain temps. Si la connexion echoue avec une
> erreur d'authentification, regenerez le token.

---

## 5. Configurer EINHERJAR

Creez le fichier `config/credentials.json` a la racine du projet :

```json
{
  "client_id": "VOTRE_CLIENT_ID_54_CARACTERES",
  "client_secret": "VOTRE_CLIENT_SECRET_50_CARACTERES",
  "access_token": "VOTRE_ACCESS_TOKEN_43_CARACTERES",
  "account_id": 1234567,
  "host": "demo.ctraderapi.com",
  "port": 5035,
  "broker_name": "ic_markets"
}
```

| Champ | Description |
|-------|-------------|
| `client_id` | Client ID de l'application (54 chars) |
| `client_secret` | Client Secret de l'application (50 chars) |
| `access_token` | Access token OAuth2 (43 chars) |
| `account_id` | ctidTraderAccountId (entier) |
| `host` | `demo.ctraderapi.com` ou `live.ctraderapi.com` |
| `port` | `5035` (defaut gRPC) |
| `broker_name` | `ic_markets` ou `pepperstone` (mapping symboles) |

---

## 6. Tester la connexion

Installez la librairie Python :

```bash
pip install ctrader-open-api
```

Puis lancez le script de test :

```bash
python scripts/test_ctrader.py
```

Si tout est configure correctement, vous verrez :
- Confirmation de la connexion
- Affichage du solde/equity/marge
- Liste des positions ouvertes
- Quelques bougies OHLCV

---

## 7. Hedging

Les deux brokers supportent le **mode hedging** sur cTrader. Il doit etre
active explicitement lors de la creation du compte de trading.

> Sans hedging, cTrader inverse automatiquement une position opposee au lieu
d'en ouvrir une nouvelle. Verifiez ce parametre dans votre portail client.

---

## 8. Levier

Le levier du compte est fixe par le broker (ex: 1:500). Il est lu
automatiquement via l'API au demarrage. **Vous ne pouvez pas le modifier
par code.**

Le Risk Manager EINHERJAR calcule dynamiquement le **volume** de chaque
position en fonction de l'equity et du drawdown, en respectant la marge
disponible.

---

## 9. Depannage

| Symptome | Cause probable | Solution |
|----------|---------------|----------|
| `Timeout connexion` | Credentials invalides | Verifiez client_id, secret, token |
| `Compte introuvable` | Mauvais account_id | Verifiez le ctidTraderAccountId |
| `ModuleNotFoundError` | `ctrader-open-api` non installe | `pip install ctrader-open-api` |
| `Circuit breaker ouvert` | Trop d'erreurs consecutives | Attendre 60s ou redemarrer |
| `Symbole inconnu` | Mapping symbole incorrect | Verifier `broker_name` dans credentials |

---

## 10. Ressources

- Documentation cTrader Open API : https://spotware.github.io/OpenApiPy/
- GitHub Spotware (OpenApiPy) : https://github.com/spotware/OpenApiPy
- Endpoints Spotware : https://openapi.ctrader.com/docs/

---

**Guide valide pour EINHERJAR v1.1 (cTrader Cloud).**
