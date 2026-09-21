# Déploiement

Trois chemins, du plus adapté au plus contraint. Les chiffres cités sont **mesurés**
sur le compte de démonstration, pas estimés.

## Ce dont le système a besoin

| Ressource | Pourquoi | Minimum réel constaté |
|---|---|---|
| Processus **toujours actif** | la boucle d'inférence calcule un cycle à chaque clôture de bougie ; endormie, elle ne produit aucun signal (les TP/SL restent chez le broker) | jamais de mise en veille |
| **~1 vCPU** | échauffement JIT Numba + recalcul des features de 54 couples | 0,1 CPU = démarrage 30-50 min |
| **≈1 Go de RAM** | polars + pandas + numba + dask chargés | 250 Mo observés au repos |
| **Disque persistant** | base DuckDB (journal, fills, equity), parquets OHLCV, cache Numba | sinon tout est recompilé/réamorcé à chaque redémarrage |
| **Sortie TCP vers `demo.ctraderapi.com:5035`** | API Open API du broker | autorisée sur Render, vérifiée en production |
| **HTTPS public** | dashboard + page de connexion | — |

Mesures comparatives, mêmes données : l'amorçage historique coûte **~0,3 s par couple**
sur un poste multicœur et **14 s par couple** sur une instance Render gratuite (0,1 CPU) ;
l'échauffement Numba prend **93 s** sur un poste multicœur.

## Chemin A — VM (recommandé : Oracle Cloud Always Free, Hetzner ou Azure Students)

Une VM réelle apporte le CPU, la persistance et l'absence de mise en veille. Oracle
**exige une carte bancaire** à l'inscription (vérification d'identité ; les cartes
virtuelles et prépayées sont refusées). Azure for Students ne demande **aucune carte**,
avec un e-mail universitaire vérifié.

### A.1 Avec Docker (le plus court)

```bash
# Sur la VM (Ubuntu 24.04) : Docker puis le projet
curl -fsSL https://get.docker.com | sudo sh
git clone https://github.com/keisary/einherjar.git && cd einherjar
cp .env.example .env && nano .env          # identifiants cTrader + mot de passe d'accès
docker compose up -d --build
docker compose logs -f                     # amorçage, échauffement, cycles
```

### A.2 Sans Docker (systemd)

```bash
sudo apt install -y python3.11 python3.11-venv git
sudo useradd --system --create-home einherjar
sudo git clone https://github.com/keisary/einherjar.git /opt/einherjar
cd /opt/einherjar
sudo python3.11 -m venv .venv
sudo .venv/bin/pip install -r requirements.txt
sudo .venv/bin/pip install --no-deps ctrader-open-api==0.9.2
sudo .venv/bin/pip install --no-deps -e .
sudo cp .env.example .env && sudo nano .env
sudo chown -R einherjar:einherjar /opt/einherjar
sudo cp deploy/einherjar.service /etc/systemd/system/
sudo systemctl enable --now einherjar
journalctl -u einherjar -f
```

### A.3 HTTPS sans configuration

Caddy obtient le certificat automatiquement ; le domaine doit pointer vers la VM et les
ports 80/443 être ouverts dans le pare-feu du fournisseur (chez Oracle : *Security List*
du VCN **et** `ufw`/`iptables` de l'instance).

```bash
sudo apt install -y caddy
sudo cp deploy/Caddyfile /etc/caddy/Caddyfile   # y mettre son domaine
sudo systemctl reload caddy
```

> [!NOTE]
> Sur Oracle, choisir la région **Frankfurt** ou **Paris** (latence vers l'API cTrader)
> et une instance **Ampere A1 (Arm)**. Les roues utilisées (`numpy`, `numba`, `polars`,
> `pyarrow`, `duckdb`) publient des versions `aarch64` ; si une installation échoue,
> `pip install -r requirements.txt -v` indique la roue manquante.

## Chemin B — Render (déploiement actuel)

- **Build Command** : `pip install -r requirements.txt && pip install --no-deps -e . && pip install --no-deps ctrader-open-api==0.9.2`
- **Start Command** : `python main.py` (le port vient de `$PORT`)
- **Health Check Path** : `/healthz`
- **Variables** : `PYTHON_VERSION=3.11.9` + les `EINHERJAR_*` (voir `.env.example`)
- **Limites du plan gratuit** : 0,1 CPU (démarrage 30-50 min), mise en veille après
  15 min sans requête HTTP — la sonde `.github/workflows/keepalive.yml` interroge
  `/healthz` toutes les 10 min pour l'éviter — et aucun disque persistant (le journal
  DuckDB est perdu à chaque redéploiement ; l'historique se réamorce depuis le broker).

## Chemin C — n'importe quelle plateforme de conteneurs

L'image est autonome (le dashboard construit est versionné, aucune compilation Node au
déploiement) :

```bash
docker build -t einherjar .
docker run -d --name einherjar --restart unless-stopped \
  --env-file .env -p 8000:8000 -v "$PWD/data:/app/data" einherjar
```

Sur une plateforme qui impose son port (Fly, Koyeb, Hugging Face) : `PORT` suffit.
Un volume est indispensable pour `data/`, sinon chaque redémarrage recompile Numba.

## Vérifier un déploiement

```bash
# 1) sonde publique
curl -s -o /dev/null -w "%{http_code}\n" https://<domaine>/healthz    # -> 200

# 2) tout le reste est protégé (401 sans session, redirection pour le navigateur)
curl -s -o /dev/null -w "%{http_code}\n" https://<domaine>/api/account # -> 401

# 3) contrôle complet : authentification, compte réel, broker, boucle, couverture
python scripts/verifier_deploiement_render.py https://<domaine>
```

Dans les journaux, la séquence saine est : bannière des composants (`[ OK ] CTRADER`)
→ `Amorcage historique : N couples` → `Echauffement features (JIT)` → `Couverture des
features : N couples` → `Cycle ... | assets= | signals= | orders= | errors=0`.

## Sécurité

- Page de connexion obligatoire (cookie signé HMAC, `HttpOnly`, `Secure` en HTTPS,
  comparaison à temps constant, 5 tentatives par IP). `/healthz` seul reste public.
- Aucun secret dans le dépôt : `.env` et `config/credentials.json` sont gitignorés ;
  l'environnement est prioritaire sur le fichier.
- Le **token Open API expire en ~30 jours** : prévoir le `refresh_token` pour un
  fonctionnement continu, sinon la boucle s'arrête à l'expiration.
- `POST /api/kill_switch` coupe les nouvelles exécutions sans fermer les positions.
