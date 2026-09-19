"""Liste les comptes cTrader autorises par le token d'acces.

Sert a renseigner `account_id` dans config/credentials.json : le
`ctidTraderAccountId` de l'Open API ne correspond PAS au numero de compte
affiche par le broker (ni a celui recu par email).

Usage :
    python scripts/ctrader_comptes.py

Sortie : un tableau account_id / demo|live / login, et la valeur a recopier
selon `environment` du fichier de credentials. Exit 1 si rien n'est lisible.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

CREDENTIALS = PROJECT_ROOT / "config" / "credentials.json"


def _charger_credentials() -> dict:
    if not CREDENTIALS.exists():
        print(f"[ERREUR] {CREDENTIALS} absent (copier config/credentials.example.json).")
        raise SystemExit(1)
    data = json.loads(CREDENTIALS.read_text(encoding="utf-8"))
    requis = ["client_id", "client_secret", "access_token"]
    vides = [c for c in requis if not str(data.get(c, "")).strip()]
    if vides:
        print(f"[ERREUR] Champs vides dans {CREDENTIALS.name} : {', '.join(vides)}")
        raise SystemExit(1)
    return data


async def _main() -> int:
    from einherjar.brokers.ctrader_adapter import CTRADER_AVAILABLE, CTraderAdapter

    if not CTRADER_AVAILABLE:
        print("[ERREUR] ctrader-open-api indisponible : pip install ctrader-open-api")
        return 1

    credentials = _charger_credentials()
    environnement = str(credentials.get("environment", "demo")).lower()
    adapter = CTraderAdapter(
        client_id=str(credentials["client_id"]),
        client_secret=str(credentials["client_secret"]),
        access_token=str(credentials["access_token"]),
        account_id=int(credentials.get("account_id", 0) or 0),
        host=str(credentials.get("host", "demo.ctraderapi.com")),
        port=int(credentials.get("port", 5035)),
        broker_name=str(credentials.get("broker_name", "ic_markets")),
    )

    print(f"Connexion a {credentials.get('host')} (environment={environnement})...")
    if not await adapter.connect():
        print("[ERREUR] Connexion refusee : verifier client_id/client_secret/access_token.")
        return 1

    try:
        comptes = await adapter.get_comptes_autorises()
    finally:
        await adapter.disconnect()

    if not comptes:
        print("[ERREUR] Aucun compte autorise pour ce token.")
        print("        -> Le compte doit etre autorise depuis le lien d'autorisation OAuth")
        print("           (ou regenere dans le Playground) puis relance ce script.")
        return 1

    print(f"\n{len(comptes)} compte(s) autorise(s) :\n")
    print(f"  {'account_id':>14s}  {'type':6s}  {'login':>12s}")
    print(f"  {'-' * 14}  {'-' * 6}  {'-' * 12}")
    actuel = int(credentials.get("account_id", 0) or 0)
    for compte in comptes:
        marque = " <-- account_id actuel" if compte["account_id"] == actuel else ""
        print(
            f"  {compte['account_id']:>14d}  {'live' if compte['is_live'] else 'demo':6s}"
            f"  {compte['login']:>12d}{marque}"
        )

    # Piege verifie : mettre le numero de compte du broker (`traderLogin`) a la place
    # du `ctidTraderAccountId` ne fait PAS echouer la connexion, mais l'API refuse
    # ensuite TOUTE requete ("Trading account is not authorized"). On nomme le compte
    # a utiliser, jamais "le premier de la liste".
    par_login = {c["login"]: c for c in comptes}
    if actuel and actuel not in {c["account_id"] for c in comptes}:
        if actuel in par_login:
            vrai = par_login[actuel]
            print(
                f"\n[ATTENTION] {actuel} est un `traderLogin` (numero de compte broker), "
                "pas un `ctidTraderAccountId`."
            )
            print(f"            -> ce compte s'ecrit account_id: {vrai['account_id']}")
        else:
            print(f"\n[ATTENTION] account_id {actuel} n'est dans aucun compte autorise.")

    attendu = [c for c in comptes if (not c["is_live"]) == (environnement == "demo")]
    if not attendu:
        print(f"\n[ATTENTION] Aucun compte {environnement} dans la liste :")
        print("            creer le compte demo chez le broker puis relancer l'autorisation.")
        return 1

    choisi = next(
        (c for c in attendu if c["account_id"] == actuel), None
    ) or (par_login.get(actuel) if actuel in par_login else None) or attendu[0]
    print(f"\n-> Recopier ceci dans {CREDENTIALS.name} :")
    print(f'   "account_id": {choisi["account_id"]},')
    print(f'   "environment": "{environnement}",')
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
