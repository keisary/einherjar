"""Pousse les variables d'environnement du deploiement Render EINHERJAR.

Les valeurs sensibles (identifiants cTrader, mot de passe de la page de connexion)
sont lues sur le disque local et envoyees directement a l'API Render : elles ne sont
JAMAIS affichees, ni en sortie, ni dans un journal.

Usage :
    python deploy_render_env.py srv-xxxxx
"""

from __future__ import annotations

import json
import os
import secrets
import sys
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CREDENTIALS = PROJECT_ROOT / "config" / "credentials.json"
FICHIER_ACCES = PROJECT_ROOT / "RENDER_ACCES.local.txt"
HERMES_ENV = Path(os.environ.get("LOCALAPPDATA", "")) / "hermes" / ".env"


def cle_api_render() -> str:
    """Recupere la cle API Render depuis l'environnement Hermes (jamais affichee)."""
    for source in (os.environ.get("MCP_RENDER_API_KEY"), None):
        if source:
            return source
    if HERMES_ENV.exists():
        for ligne in HERMES_ENV.read_text(encoding="utf-8", errors="replace").splitlines():
            ligne = ligne.strip()
            if ligne.startswith("MCP_RENDER_API_KEY=") and not ligne.startswith("#"):
                return ligne.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("MCP_RENDER_API_KEY introuvable")


def requete(methode: str, url: str, corps: dict | None, cle: str) -> dict:
    """Appel HTTP JSON a l'API Render."""
    donnees = json.dumps(corps).encode() if corps is not None else None
    requete_http = urllib.request.Request(url, data=donnees, method=methode)
    requete_http.add_header("Authorization", f"Bearer {cle}")
    requete_http.add_header("Content-Type", "application/json")
    requete_http.add_header("Accept", "application/json")
    try:
        with urllib.request.urlopen(requete_http, timeout=60) as reponse:
            corps_texte = reponse.read().decode()
    except urllib.error.HTTPError as exc:  # message lisible sans exposer de secret
        raise SystemExit(f"HTTP {exc.code} sur {methode} {url}: {exc.read().decode()[:400]}")
    return json.loads(corps_texte) if corps_texte.strip() else {}


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit("usage: python deploy_render_env.py <serviceId>")
    service_id = sys.argv[1]
    cle = cle_api_render()

    if not CREDENTIALS.exists():
        raise SystemExit(f"{CREDENTIALS} absent : impossible de recuperer les identifiants")
    creds = json.loads(CREDENTIALS.read_text(encoding="utf-8"))

    # Mot de passe de la page de connexion : genere ici, ecrit uniquement dans un
    # fichier local non versionne, puis envoye a Render.
    mot_de_passe = secrets.token_urlsafe(24)
    secret_session = secrets.token_urlsafe(48)

    variables = {
        "PYTHON_VERSION": "3.11.9",
        "EINHERJAR_ENVIRONMENT": str(creds.get("environment", "demo")),
        "EINHERJAR_HOST": str(creds.get("host", "demo.ctraderapi.com")),
        "EINHERJAR_CLIENT_ID": str(creds.get("client_id", "")),
        "EINHERJAR_CLIENT_SECRET": str(creds.get("client_secret", "")),
        "EINHERJAR_ACCESS_TOKEN": str(creds.get("access_token", "")),
        "EINHERJAR_ACCOUNT_ID": str(creds.get("account_id", "")),
        "EINHERJAR_BROKER_NAME": str(creds.get("broker_name", "")),
        "EINHERJAR_AUTH_USERNAME": "jovanny",
        "EINHERJAR_AUTH_PASSWORD": mot_de_passe,
        "EINHERJAR_AUTH_SECRET": secret_session,
        "PYTHONUNBUFFERED": "1",
    }
    manquantes = [c for c in ("client_id", "client_secret", "access_token", "account_id") if not creds.get(c)]
    if manquantes:
        raise SystemExit(f"Identifiants incomplets dans credentials.json : {manquantes}")

    reponse = requete(
        "PUT",
        f"https://api.render.com/v1/services/{service_id}/env-vars",
        [{"key": k, "value": v} for k, v in variables.items()],
        cle,
    )
    noms = sorted(v["key"] for v in variables.items().__iter__()) if False else sorted(variables)
    print(f"variables envoyees ({len(noms)}) : {', '.join(noms)}")

    deploiement = requete(
        "POST",
        f"https://api.render.com/v1/services/{service_id}/deploys",
        {"clearCache": "do_not_clear"},
        cle,
    )
    identifiant_deploiement = (
        deploiement.get("id") or (deploiement.get("deploy") or {}).get("id") or "?"
    )
    print(f"deploiement declenche : {identifiant_deploiement}")

    FICHIER_ACCES.write_text(
        "Acces a l'application EINHERJAR deployee sur Render\n"
        "===================================================\n\n"
        "URL      : https://einherjar.onrender.com\n"
        "Utilisateur : jovanny\n"
        f"Mot de passe : {mot_de_passe}\n\n"
        "Ce fichier n'est pas versionne (gitignore) : il ne quitte pas cette machine.\n"
        "Pour changer le mot de passe : modifier EINHERJAR_AUTH_PASSWORD dans le\n"
        "tableau de bord Render (Environment), puis redeployer.\n",
        encoding="utf-8",
    )
    print(f"acces ecrit dans : {FICHIER_ACCES.name} (non versionne)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
