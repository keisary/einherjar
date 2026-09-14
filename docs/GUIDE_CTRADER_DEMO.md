# Guide — compte démo cTrader et démarrage d'EINHERJAR

Objectif : obtenir les 5 valeurs de `config/credentials.json` et vérifier que **tous
les actifs entraînés** sont négociables, avant de lancer le système.

Chaîne complète : **broker (IC Markets) → cTID → compte démo cTrader → application
Open API → token → `account_id` → credentials.json → vérification → `main.py`**.

---

## 1. Ouvrir le compte chez le broker (IC Markets)

C'est le broker qui héberge le compte, pas cTrader. Étapes IC Markets
(source : `ic.com`, page cTrader Web) :

1. **Créer le compte client** : https://secure.ic.com/en/Account/Register
   (pour ne créer qu'une démo côté broker : ajouter `?type=demo`).
2. **Terminer l'inscription et la vérification** (KYC : pièce d'identité,
   justificatif de domicile). Rien ne fonctionne avant validation.
3. **Recevoir l'email cTrader ID (cTID)** et **définir le mot de passe cTID**.
4. Se connecter à cTrader (Web, desktop ou mobile) avec ce cTID : **le compte de
   trading apparaît automatiquement** une fois lié au cTID.

> Le cTID est l'identité unique cTrader : c'est lui qui porte l'autorisation Open
> API, et il peut contenir plusieurs comptes (démo et live, chez un ou plusieurs
> brokers).

## 2. Créer le compte de trading DÉMO

(source : `help.ctrader.com/ctrader/trading-accounts/`)

1. Dans cTrader, cliquer sur la **barre de compte** (en haut à droite) →
   **Create new trading account**.
2. Formulaire **Demo account** : montant du dépôt, devise du compte, **levier**, et
   le **type de compte** :
   - **Hedging** ← à choisir pour EINHERJAR (plusieurs positions sur le même
     symbole, sens opposés autorisés : nos einhers sont indépendants) ;
   - Netting (une seule position par symbole, les ordres opposés se compensent).
3. **Create account**.

Deux points qui comptent pour « trader tous les actifs » :

- **Éviter un compte AMF / « French-risk »** : certains symboles peuvent être
  désactivés par le broker sur ce type de compte, et le stop loss garanti y est
  obligatoire (`help.ctrader.com`). Ce type ne se crée d'ailleurs pas depuis
  cTrader, il faut le demander au broker.
- Utiliser **l'application cTrader IC Markets** (ou l'app cross-broker en
  sélectionnant IC Markets) : l'Open API se connecte au serveur **du broker**. Un
  compte démo créé chez un autre broker ne sera pas servi par le même endpoint.

## 3. Enregistrer l'application Open API (client_id / client_secret)

(source : `help.ctrader.com/open-api/api-application/`)

1. Aller sur https://openapi.ctrader.com/ et **se connecter avec le cTID**.
2. Ouvrir la page **Applications** : https://openapi.ctrader.com/apps
3. **Add new app** → remplir le formulaire (une description détaillée accélère
   l'approbation) → **Save**.
4. Statut initial `submitted` : **Spotware examine la demande et répond par email**.
5. Après approbation : dans la colonne **Credentials**, bouton **View** →
   copier **`client_id`** (~54 caractères) et **`client_secret`** (~50).

> Le premier *redirect URI* fourni par défaut est celui du **Playground** : il sert
> uniquement aux tests (étape 4) et ne doit pas être utilisé en production.

## 4. Obtenir l'`access_token` (le plus simple : le Playground)

(source : `help.ctrader.com/open-api/account-authentication/`)

**Option A — Playground (recommandé pour ton propre cTID)**
1. Page **Applications** → bouton **Playground**.
2. Choisir le **scope** : `trading` (accès complet, opérations de trading
   autorisées) — `accounts` ne donnerait qu'un accès en lecture.
3. **Get token** → copier `accessToken` (et `refreshToken`).

**Option B — flux OAuth complet (pour une app distribuée)**
```
https://id.ctrader.com/my/settings/openapi/grantingaccess/?client_id={clientId}&redirect_uri={redirectURI}&scope=trading&product=web
```
L'utilisateur autorise l'app, cTrader redirige vers `redirect_uri?code=...`, puis :
```
GET https://openapi.ctrader.com/apps/token?grant_type=authorization_code&code={code}&redirect_uri={redirectURI}&client_id={clientId}&client_secret={clientSecret}
```

**Durées de vie** : le code d'autorisation expire en **1 minute** ; l'`access_token`
en **2 628 000 s (~30 jours)** ; le **refresh token n'expire pas**. Pour rafraîchir :
```
curl -X POST 'https://openapi.ctrader.com/apps/token?grant_type=refresh_token&refresh_token={refreshToken}&client_id={clientId}&client_secret={clientSecret}'
```

## 5. Trouver l'`account_id` (ctidTraderAccountId)

⚠️ **Le `ctidTraderAccountId` de l'Open API ne correspond PAS au numéro de compte
affiché par le broker ou reçu par email** (constaté aussi sur le forum cTrader).
Il s'obtient du protocole : `ProtoOAGetAccountListByAccessTokenRes` renvoie la
liste des comptes autorisés.

Le script le fait pour toi :
```bash
cd /d/midas_v2/einherjar
python scripts/ctrader_comptes.py
```
Il affiche :
```
  {'account_id': 17537049,  'demo',  login 5xxxxxxx  <-- a utiliser
   ...}
-> Recopier ceci dans credentials.json :
   "account_id": 17537049,
```
Il existe **un `ctidTraderAccountId` par compte** (démo et live ont des ids
différents) — et `isLive` indique lequel est lequel.

## 6. Remplir `config/credentials.json`

Le fichier existe déjà (modèle versionné : `config/credentials.example.json`,
ignoré par git) :
```json
{
  "environment": "demo",          creer le compte demo cTrader
  "host": "demo.ctraderapi.com",  port 5035 (les deux endpoints)
  "port": 5035,
  "account_id": 0,                <- etape 5
  "client_id": "",                <- etape 3
  "client_secret": "",            <- etape 3
  "access_token": "",             <- etape 4
  "broker_name": "ic_markets"     "ic_markets" ou "pepperstone"
}
```
Endpoints officiels (constantes de la librairie `ctrader_open_api.EndPoints`) :
`demo.ctraderapi.com` / `live.ctraderapi.com`, **port 5035** pour les deux.
`broker_name` pilote la conversion des symboles MIDAS → cTrader.

## 7. Vérifier que TOUS les actifs entraînés sont négociables

```bash
python scripts/ctrader_verifier_univers.py            # liste des symboles
python scripts/ctrader_verifier_univers.py --probe     # + 5 bougies par actif
```
Le script compare les **28 actifs du corpus** (837 einhers) à la liste de symboles
réellement servie par le compte, puis affiche par actif : symbole cTrader, nombre
d'einhers concernés, et `OK` ou `ABSENT CHEZ LE BROKER`.

Mapping déjà en place dans `src/einherjar/brokers/broker_utils.py` (couverture
**28/28 vérifiée hors ligne**) :

| Corpus | Symbole cTrader (IC Markets) | | Corpus | Symbole cTrader |
|---|---|---|---|---|
| NASDAQ100 | US100 | | SP500 | US500 |
| DOWJONES | US30 | | DAX40 | DE40 |
| WTIUSD | USOUSD | | BRENT | UKOUSD |
| COPPER | XCUUSD | | XAUUSD | XAUUSD |
| BTC/ETH/ADA/BCH/LTC USD | identiques | | EURUSD, GBPUSD, … | identiques |
| AAPL, MSFT, NVDA, GOOGL, AMZN, TSLA, JPM, XOM | identiques | | | |

**Ce que le broker peut refuser** (à confirmer par le script, pas par supposition) :
les **CFD actions** et les **crypto** sont soumis au profil/entité du compte et
peuvent être absents d'un compte démo. Si un actif manque, **`main.py` l'écarte
automatiquement** (log `Actifs absents chez le broker : … -> N couple(s) ecarte(s)`),
donc le système ne boucle jamais sur un couple inexistant.

## 8. Lancer

```bash
cd /d/midas_v2/einherjar
python main.py
```
Séquence attendue : 837 einhers chargés → vérification de l'univers broker →
amorçage historique (~16 s) → échauffement JIT (~100 s) → cycles ~0,5 s.

Dashboard (le serveur sert l'API **et** l'interface construite) :
```bash
PYTHONPATH=src python -m uvicorn einherjar.api.server:app --port 8000
# http://localhost:8000
```

## 9. Dépannage

| Symptôme | Cause / action |
|---|---|
| `[ARRET] Compte cTrader DEMO non connecte` | champs vides dans `credentials.json` (le message les liste) |
| `ACCESS_DENIED` au token | `client_id`/`client_secret`/`redirect_uri` doivent venir de **la même** application |
| Aucun compte dans `ctrader_comptes.py` | le compte n'est pas autorisé : refaire le flux d'autorisation (étape 4) ; **les comptes créés après l'autorisation doivent être ré-autorisés** |
| Token expiré (~30 j) | `grant_type=refresh_token` (le refresh token n'expire pas) |
| `Symbole indisponible chez le broker` | actif non servi par le compte → voir étape 7 (le couple est écarté) |
| Erreur `ctrader-open-api manquant` | `pip install ctrader-open-api` puis **vérifier que protobuf reste en 7.x** (le paquet épingle 3.20.1, incompatible avec onnx/streamlit du venv) |

## 10. À valider lors de la première session (non testable sans compte)

1. Aller-retour d'authentification (`ProtoOAApplicationAuthReq` → `ProtoOAAccountAuthReq`).
2. `digits` / `minVolume` / `stepVolume` réels du broker (le contrôle de volume
   refuse un ordre sous le minimum, `ctrader_adapter.place_order_sync`).
3. Réception des fills (`ProtoOAExecutionEvent`) et état des positions
   (`ProtoOAReconcileReq`) : c'est ce qui alimente le journal et la page Positions.
4. Comportement de sortie : les TP/SL partent au broker, la durée de tenue est
   appliquée par EINHERJAR (`risk/exits.py`).

Pitfalls techniques vérifiés sur ce projet : `references/ctrader-open-api.md`
(skill `einherjar-research`).
