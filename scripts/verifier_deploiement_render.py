"""Verification de bout en bout de l'application deployee sur Render.

Le mot de passe de connexion est lu dans `RENDER_ACCES.local.txt` (fichier local non
versionne) : il n'est jamais affiche, seulement utilise pour ouvrir une session.

Usage :
    python scripts/verifier_deploiement_render.py [url]
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from http.cookiejar import CookieJar
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FICHIER_ACCES = PROJECT_ROOT / "RENDER_ACCES.local.txt"


def identifiants() -> tuple[str, str]:
    """Lit l'URL, l'utilisateur et le mot de passe depuis le fichier d'acces local."""
    if not FICHIER_ACCES.exists():
        raise SystemExit(f"{FICHIER_ACCES.name} absent : lancer scripts/deploy_render_env.py")
    url, utilisateur, mot_de_passe = "", "", ""
    for ligne in FICHIER_ACCES.read_text(encoding="utf-8").splitlines():
        if ligne.startswith("URL"):
            url = ligne.split(":", 1)[1].strip()
        elif ligne.startswith("Utilisateur"):
            utilisateur = ligne.split(":", 1)[1].strip()
        elif ligne.startswith("Mot de passe"):
            mot_de_passe = ligne.split(":", 1)[1].strip()
    if not (url and utilisateur and mot_de_passe):
        raise SystemExit("Fichier d'acces incomplet")
    return url, utilisateur, mot_de_passe


def main() -> int:
    url, utilisateur, mot_de_passe = identifiants()
    if len(sys.argv) > 1:
        url = sys.argv[1].rstrip("/")
    base = url.rstrip("/")
    ouvrir = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(CookieJar()),
        urllib.request.HTTPRedirectHandler(),
    )

    def appeler(chemin: str, corps: dict | None = None) -> tuple[int, str]:
        donnees = urllib.parse.urlencode(corps).encode() if corps else None
        requete = urllib.request.Request(base + chemin, data=donnees, method="POST" if corps else "GET")
        if corps:
            requete.add_header("Content-Type", "application/x-www-form-urlencoded")
        try:
            with ouvrir.open(requete, timeout=90) as reponse:
                return reponse.status, reponse.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read().decode("utf-8", "replace")

    resultats: list[tuple[str, bool, str]] = []

    # 1) Sans session : l'API doit refuser, la page doit rediriger vers /login.
    code, _ = appeler("/api/account")
    resultats.append(("API /api/account sans session -> 401", code == 401, str(code)))
    code, corps = appeler("/api/health")
    resultats.append(("API /api/health sans session -> 401", code == 401, str(code)))
    code, corps = appeler("/login")
    resultats.append(("Page /login servie (200 + formulaire)", code == 200 and "mot_de_passe" in corps, str(code)))
    code, _ = appeler("/")
    resultats.append(("Racine sans session -> redirection /login", code in (200, 303), str(code)))

    # 2) Mauvais mot de passe refuse.
    code, _ = appeler("/login", {"utilisateur": utilisateur, "mot_de_passe": "mauvais-mot-de-passe", "suite": "/"})
    resultats.append(("Mauvais mot de passe -> 401", code == 401, str(code)))

    # 3) Connexion reelle.
    code, _ = appeler("/login", {"utilisateur": utilisateur, "mot_de_passe": mot_de_passe, "suite": "/"})
    resultats.append(("Connexion -> redirection (303)", code in (200, 303), str(code)))

    # 4) Donnees reelles du compte.
    code, corps = appeler("/api/account")
    try:
        compte = json.loads(corps)
    except json.JSONDecodeError:
        compte = {}
    ok_compte = code == 200 and compte.get("connected") is True and compte.get("accountId")
    resultats.append(
        (
            "Compte reel servi par l'API (/api/account)",
            bool(ok_compte),
            f"HTTP {code} accountId={compte.get('accountId')} balance={compte.get('balance')} connected={compte.get('connected')}",
        )
    )

    # 5) Sante : broker connecte, boucle, couverture.
    code, corps = appeler("/api/health")
    try:
        sante = json.loads(corps)
    except json.JSONDecodeError:
        sante = {}
    composants = sante.get("components", {}) if isinstance(sante, dict) else {}
    broker = composants.get("ctrader", {}) if isinstance(composants, dict) else {}
    resultats.append(
        (
            "Broker connecte dans /api/health",
            bool(broker.get("connected")),
            f"connected={broker.get('connected')} circuit={broker.get('circuitState')} lastError={broker.get('lastError')}",
        )
    )
    boucle = composants.get("loop") or {}
    resultats.append(
        (
            "Boucle d'inference en marche (etat persiste)",
            bool(isinstance(boucle, dict) and boucle.get("running")),
            json.dumps(boucle, ensure_ascii=False)[:200] if boucle else "aucun cycle encore",
        )
    )
    couverture = composants.get("couverture") or {}
    resultats.append(
        (
            "Couverture des features verifiee",
            bool(isinstance(couverture, dict) and couverture.get("couples_verifies")),
            json.dumps(couverture, ensure_ascii=False)[:200] if couverture else "non publiee",
        )
    )
    resultats.append(("Einhers charges", (composants.get("corpusEinhers") or 0) > 0, str(composants.get("corpusEinhers"))))

    # 6) Dashboard servi.
    code, corps = appeler("/")
    resultats.append(("Dashboard servi (HTML du build)", code == 200 and "<div id=\"root\"" in corps, str(code)))

    # 7) Deconnexion.
    code, _ = appeler("/logout")
    code2, _ = appeler("/api/account")
    resultats.append(("Deconnexion invalide la session", code2 == 401, f"logout={code} puis account={code2}"))

    print("=" * 78)
    print(f"VERIFICATION DU DEPLOIEMENT : {base}")
    print("=" * 78)
    echecs = 0
    for nom, ok, detail in resultats:
        print(f"  [{'OK ' if ok else 'ECHEC'}] {nom}")
        if detail:
            print(f"          {detail}")
        echecs += 0 if ok else 1
    print("=" * 78)
    print(f"{len(resultats) - echecs}/{len(resultats)} verifications reussies")
    return 1 if echecs else 0


if __name__ == "__main__":
    raise SystemExit(main())
