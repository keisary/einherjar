"""Chargement des identifiants cTrader (fichier local OU variables d'environnement).

Un deploiement heberge ne peut pas embarquer `config/credentials.json` (gitignore, et
un secret ne se met pas dans un depot). Ce module est donc la source unique :

1. `config/credentials.json` s'il existe (poste de travail, usage local) ;
2. complete (et remplace) par les variables d'environnement `EINHERJAR_*` (Render,
   Docker, CI) — l'environnement gagne toujours, pour ne jamais deployer un fichier
   oublie sur le disque par erreur.

Usage :
    creds = charger_credentials()
    if creds is None: ...
    adaptateur = CTraderAdapter(client_id=creds["client_id"], ...)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

#: Racine du projet (deux niveaux au-dessus de `src/einherjar/config/`).
PROJECT_ROOT = Path(__file__).resolve().parents[3]
CREDENTIALS_PATH = PROJECT_ROOT / "config" / "credentials.json"

#: Correspondance cle du fichier -> variable d'environnement.
CREDENTIALS_ENV = {
    "client_id": "EINHERJAR_CLIENT_ID",
    "client_secret": "EINHERJAR_CLIENT_SECRET",
    "access_token": "EINHERJAR_ACCESS_TOKEN",
    "refresh_token": "EINHERJAR_REFRESH_TOKEN",
    "account_id": "EINHERJAR_ACCOUNT_ID",
    "environment": "EINHERJAR_ENVIRONMENT",
    "host": "EINHERJAR_HOST",
    "port": "EINHERJAR_PORT",
    "broker_name": "EINHERJAR_BROKER_NAME",
    "redirect_uri": "EINHERJAR_REDIRECT_URI",
}

#: Cles sans lesquelles le systeme ne peut pas se connecter.
CLES_OBLIGATOIRES = ("client_id", "client_secret", "access_token", "account_id")

#: Valeurs par defaut du compte de demonstration cTrader.
DEFAUTS_DEMO = {
    "environment": "demo",
    "host": "demo.ctraderapi.com",
    "port": 5035,
}


def charger_credentials(chemin: Path | None = None) -> dict[str, Any] | None:
    """Charge les identifiants cTrader depuis le fichier puis l'environnement.

    Args:
        chemin: Chemin du fichier JSON (defaut : `config/credentials.json`).

    Returns:
        Dict des identifiants, ou None si aucune source n'est exploitable.
    """
    fichier = Path(chemin) if chemin is not None else CREDENTIALS_PATH
    donnees: dict[str, Any] = {}

    if fichier.exists():
        try:
            brut = json.loads(fichier.read_text(encoding="utf-8"))
            if isinstance(brut, dict):
                donnees.update(brut)
        except (OSError, json.JSONDecodeError):
            donnees = {}

    for cle, variable in CREDENTIALS_ENV.items():
        valeur = os.environ.get(variable)
        if valeur not in (None, ""):
            donnees[cle] = valeur

    if not donnees:
        return None

    if "account_id" in donnees:
        try:
            donnees["account_id"] = int(str(donnees["account_id"]).strip())
        except (TypeError, ValueError):
            pass
    if "port" in donnees:
        try:
            donnees["port"] = int(str(donnees["port"]).strip())
        except (TypeError, ValueError):
            donnees.pop("port")

    environnement = str(donnees.get("environment") or "").strip().lower()
    if environnement not in ("demo", "live"):
        environnement = "live" if "live" in str(donnees.get("host", "")).lower() else "demo"
    donnees["environment"] = environnement
    for cle, valeur in DEFAUTS_DEMO.items():
        if cle == "environment":
            continue
        if not donnees.get(cle):
            donnees[cle] = valeur if environnement == "demo" else ("live.ctraderapi.com" if cle == "host" else valeur)

    return donnees


def credentials_complets(creds: dict[str, Any] | None) -> tuple[bool, list[str]]:
    """Indique si les identifiants sont suffisants pour se connecter.

    Args:
        creds: Dict retourne par :func:`charger_credentials`.

    Returns:
        Tuple (complets, cles_manquantes).
    """
    if not creds:
        return False, list(CLES_OBLIGATOIRES)
    manquants = [cle for cle in CLES_OBLIGATOIRES if not creds.get(cle)]
    return (not manquants), manquants
