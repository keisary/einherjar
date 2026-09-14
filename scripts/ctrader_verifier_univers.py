"""Verifie que TOUS les actifs entraines sont negotiables chez le broker.

Compare les actifs du corpus (outputs/corpus.jsonl) a la liste de symboles
reellement servie par cTrader, puis (option --probe) demande 5 bougies par actif
pour prouver que les donnees circulent.

Usage :
    python scripts/ctrader_verifier_univers.py            # liste des symboles
    python scripts/ctrader_verifier_univers.py --probe     # + bougies par actif

Exit 0 si tous les actifs sont disponibles, 1 sinon (liste des manquants et
nombre d'einhers concernes affiches).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

CREDENTIALS = PROJECT_ROOT / "config" / "credentials.json"
CORPUS = PROJECT_ROOT / "outputs" / "corpus.jsonl"


def _universe_corpus() -> dict[str, int]:
    """Actif -> nombre d'einhers du corpus qui le surveillent."""
    compteur: Counter[str] = Counter()
    for ligne in CORPUS.read_text(encoding="utf-8").splitlines():
        if not ligne.strip():
            continue
        univers = (json.loads(ligne).get("universe") or {})
        actif = univers.get("asset")
        if actif:
            compteur[str(actif)] += 1
    return dict(compteur)


def _charger_credentials() -> dict:
    if not CREDENTIALS.exists():
        print(f"[ERREUR] {CREDENTIALS} absent.")
        raise SystemExit(1)
    data = json.loads(CREDENTIALS.read_text(encoding="utf-8"))
    for champ in ("client_id", "client_secret", "access_token", "account_id"):
        if not str(data.get(champ, "")).strip() or str(data.get(champ)) == "0":
            print(f"[ERREUR] Champ '{champ}' vide dans {CREDENTIALS.name}.")
            print("        (account_id : lancer d'abord scripts/ctrader_comptes.py)")
            raise SystemExit(1)
    return data


async def _main() -> int:
    parser = argparse.ArgumentParser(description="Verifie la couverture des actifs du corpus")
    parser.add_argument("--probe", action="store_true", help="demande aussi 5 bougies par actif")
    parser.add_argument("--timeframe", default="1h", help="timeframe du probe (defaut 1h)")
    args = parser.parse_args()

    from einherjar.brokers.broker_utils import normalize_symbol
    from einherjar.brokers.ctrader_adapter import CTRADER_AVAILABLE, CTraderAdapter

    if not CTRADER_AVAILABLE:
        print("[ERREUR] ctrader-open-api indisponible : pip install ctrader-open-api")
        return 1

    credentials = _charger_credentials()
    actifs = _universe_corpus()
    adapter = CTraderAdapter(
        client_id=str(credentials["client_id"]),
        client_secret=str(credentials["client_secret"]),
        access_token=str(credentials["access_token"]),
        account_id=int(credentials["account_id"]),
        host=str(credentials.get("host", "demo.ctraderapi.com")),
        port=int(credentials.get("port", 5035)),
        broker_name=str(credentials.get("broker_name", "ic_markets")),
    )

    print(f"Connexion a {credentials.get('host')} (environment={credentials.get('environment')})...")
    if not await adapter.connect():
        print("[ERREUR] Connexion refusee.")
        return 1

    manquants: list[str] = []
    try:
        symboles = await adapter.get_symboles_disponibles()
        print(f"Symboles servis par le broker : {len(symboles)}\n")
        if not symboles:
            print("[ERREUR] Aucun symbole recu : verifier que le compte est bien autorise.")
            return 1

        print(f"  {'actif':10s} {'symbole':10s} {'einhers':>7s}  etat")
        print(f"  {'-' * 10} {'-' * 10} {'-' * 7}  {'-' * 24}")
        for actif in sorted(actifs, key=lambda a: (-actifs[a], a)):
            symbole = normalize_symbol(actif, str(credentials.get("broker_name", "ic_markets")))
            present = symbole.upper() in symboles
            etat = "OK" if present else "ABSENT CHEZ LE BROKER"
            if not present:
                manquants.append(actif)
            print(f"  {actif:10s} {symbole:10s} {actifs[actif]:>7d}  {etat}")

        if args.probe and not manquants:
            print(f"\nProbe OHLCV ({args.timeframe}) :")
            for actif in sorted(actifs, key=lambda a: (-actifs[a], a)):
                try:
                    df = await adapter.get_ohlcv(actif, args.timeframe, limit=5)
                    if df.height == 0:
                        print(f"  {actif:10s} AUCUNE BOUGIE")
                    else:
                        derniere = df.tail(1).row(0, named=True)
                        print(f"  {actif:10s} {df.height} bougies | dernier close={derniere['close']}")
                except Exception as exc:  # noqa: BLE001 - on veut le detail par actif
                    print(f"  {actif:10s} ERREUR: {exc}")
    finally:
        await adapter.disconnect()

    total_einhers = sum(actifs.values())
    concernes = sum(actifs[a] for a in manquants)
    print(
        f"\n{len(actifs) - len(manquants)}/{len(actifs)} actifs disponibles "
        f"({total_einhers - concernes}/{total_einhers} einhers negociables)"
    )
    if manquants:
        print(f"Actifs ABSENTS : {', '.join(manquants)}")
        print("-> Ces couples doivent etre retires de l'univers live avant de trader.")
        return 1
    print("Tous les actifs entraines sont negociables chez ce broker.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
