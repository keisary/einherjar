"""Authentification simple de l'application (page de connexion + cookie signe).

Objectif : une fois le systeme deploye sur une URL publique, personne ne doit
pouvoir lire le compte, les positions ou lancer une action sans s'etre connecte.
Le mecanisme est volontairement minimal et sans dependance externe :

- identifiants lus dans l'environnement (`EINHERJAR_AUTH_USERNAME`,
  `EINHERJAR_AUTH_PASSWORD`) ; si le mot de passe n'est pas fourni, un mot de passe
  aleatoire est genere et affiche UNE fois dans les logs (l'application n'est donc
  jamais ouverte par defaut) ;
- session = jeton `HMAC-SHA256(secret, "utilisateur:expiration")` signe, pose en
  cookie `HttpOnly` + `SameSite=Lax` (+ `Secure` des que la requete est en HTTPS) ;
- comparaison a temps constant (`hmac.compare_digest`) et limitation des tentatives
  par adresse IP pour rendre le bourrage d'identifiants inoperant.

Le module ne connait ni FastAPI ni le dashboard : il expose des fonctions pures,
testables sans serveur (`verifier_identifiants`, `creer_jeton`, `lire_jeton`).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import os
import secrets
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("einherjar.auth")

COOKIE_NAME = "einherjar_session"
DUREE_SESSION_S = 12 * 3600  # 12 h : une journee de trading, sans reconnexion
MAX_TENTATIVES = 5
FENETRE_TENTATIVES_S = 300  # 5 minutes

# Chemins accessibles sans authentification (page de connexion + sonde de sante
# minimale pour la plateforme d'hebergement : aucune donnee de compte n'y figure).
CHEMINS_PUBLICS = frozenset({"/login", "/logout", "/favicon.ico", "/healthz"})


@dataclass(frozen=True)
class Identifiants:
    """Identifiants attendus par l'application.

    Attributes:
        utilisateur: Nom de connexion.
        mot_de_passe: Mot de passe (jamais journalise).
        genere: True si le mot de passe a ete genere (absent de l'environnement).
    """

    utilisateur: str
    mot_de_passe: str
    genere: bool = False


def charger_identifiants(environ: dict[str, str] | None = None) -> Identifiants:
    """Lit les identifiants dans l'environnement, ou en genere un mot de passe.

    Args:
        environ: Environnement a lire (defaut : `os.environ`), utile aux tests.

    Returns:
        Les identifiants a utiliser. `genere=True` signale un mot de passe tire au
        hasard : l'appelant doit l'afficher une fois pour que l'operateur puisse se
        connecter (aucun mot de passe par defaut n'est code en dur).
    """
    env = os.environ if environ is None else environ
    utilisateur = (env.get("EINHERJAR_AUTH_USERNAME") or "").strip()
    mot_de_passe = env.get("EINHERJAR_AUTH_PASSWORD") or ""

    if utilisateur and mot_de_passe:
        return Identifiants(utilisateur=utilisateur, mot_de_passe=mot_de_passe)

    utilisateur = utilisateur or "einherjar"
    mot_de_passe = mot_de_passe or secrets.token_urlsafe(12)
    logger.warning(
        "AUTHENTIFICATION : EINHERJAR_AUTH_USERNAME/EINHERJAR_AUTH_PASSWORD ne sont "
        "pas definis. Identifiants temporaires -> utilisateur=%s mot de passe=%s "
        "(a definir dans l'environnement pour un acces stable).",
        utilisateur,
        mot_de_passe,
    )
    return Identifiants(utilisateur=utilisateur, mot_de_passe=mot_de_passe, genere=True)


def _secret(identifiants: Identifiants, environ: dict[str, str] | None = None) -> bytes:
    """Cle de signature des jetons de session.

    Elle derive du mot de passe (les sessions survivent donc a un redemarrage) et,
    si une variable dediee est fournie, elle a la priorite : changer le mot de passe
    invalide naturellement toutes les sessions ouvertes.
    """
    env = os.environ if environ is None else environ
    base = env.get("EINHERJAR_AUTH_SECRET") or identifiants.mot_de_passe
    return hashlib.sha256(f"einherjar|{identifiants.utilisateur}|{base}".encode()).digest()


def verifier_identifiants(
    utilisateur_fourni: str, mot_de_passe_fourni: str, identifiants: Identifiants
) -> bool:
    """Verifie un couple utilisateur/mot de passe a temps constant.

    Args:
        utilisateur_fourni: Valeur saisie dans le formulaire.
        mot_de_passe_fourni: Valeur saisie dans le formulaire.
        identifiants: Identifiants attendus.

    Returns:
        True si le couple est correct.
    """
    bon_utilisateur = hmac.compare_digest(
        str(utilisateur_fourni or ""), identifiants.utilisateur
    )
    bon_mot_de_passe = hmac.compare_digest(
        str(mot_de_passe_fourni or ""), identifiants.mot_de_passe
    )
    return bool(bon_utilisateur and bon_mot_de_passe)


def _signer(charge: str, cle: bytes) -> str:
    return base64.urlsafe_b64encode(
        hmac.new(cle, charge.encode(), hashlib.sha256).digest()
    ).decode().rstrip("=")


def creer_jeton(
    identifiants: Identifiants,
    duree_s: int = DUREE_SESSION_S,
    maintenant: float | None = None,
    environ: dict[str, str] | None = None,
) -> str:
    """Cree un jeton de session signe et date.

    Args:
        identifiants: Identifiants de l'application.
        duree_s: Duree de validite en secondes.
        maintenant: Instant de reference (defaut : maintenant).
        environ: Environnement a lire pour la cle.

    Returns:
        Jeton `utilisateur.expiration.signature` (cookie).
    """
    expiration = int((maintenant if maintenant is not None else time.time()) + duree_s)
    charge = f"{identifiants.utilisateur}.{expiration}"
    return f"{charge}.{_signer(charge, _secret(identifiants, environ))}"


def lire_jeton(
    jeton: str | None,
    identifiants: Identifiants,
    maintenant: float | None = None,
    environ: dict[str, str] | None = None,
) -> str | None:
    """Valide un jeton de session.

    Args:
        jeton: Valeur du cookie (peut etre absente).
        identifiants: Identifiants de l'application.
        maintenant: Instant de reference (defaut : maintenant).
        environ: Environnement a lire pour la cle.

    Returns:
        Le nom d'utilisateur si le jeton est valide et non expire, sinon None.
    """
    if not jeton or jeton.count(".") != 2:
        return None
    utilisateur, expiration_txt, signature = jeton.split(".")
    charge = f"{utilisateur}.{expiration_txt}"
    attendue = _signer(charge, _secret(identifiants, environ))
    if not hmac.compare_digest(signature, attendue):
        return None
    try:
        expiration = int(expiration_txt)
    except ValueError:
        return None
    if expiration < int(maintenant if maintenant is not None else time.time()):
        return None
    return utilisateur or None


class LimiteurTentatives:
    """Limiteur de tentatives de connexion par adresse IP (en memoire).

    Suffisant pour une application mono-process : il rend le bourrage
    d'identifiants lent sans ajouter de dependance ni de stockage partage.
    """

    def __init__(self, maximum: int = MAX_TENTATIVES, fenetre_s: int = FENETRE_TENTATIVES_S):
        """Initialise le limiteur.

        Args:
            maximum: Nombre de tentatives avant blocage temporaire.
            fenetre_s: Duree de memoire des echecs, en secondes.
        """
        self.maximum = maximum
        self.fenetre_s = fenetre_s
        self._echecs: dict[str, list[float]] = {}

    def _recent(self, cle: str, maintenant: float) -> list[float]:
        horodatages = [
            t for t in self._echecs.get(cle, []) if maintenant - t < self.fenetre_s
        ]
        if horodatages:
            self._echecs[cle] = horodatages
        else:
            self._echecs.pop(cle, None)
        return horodatages

    def autorise(self, cle: str, maintenant: float | None = None) -> bool:
        """Indique si une nouvelle tentative est permise.

        Args:
            cle: Identifiant du client (adresse IP).
            maintenant: Instant de reference.

        Returns:
            False si trop d'echecs recents.
        """
        instant = maintenant if maintenant is not None else time.time()
        return len(self._recent(cle, instant)) < self.maximum

    def enregistrer_echec(self, cle: str, maintenant: float | None = None) -> None:
        """Enregistre un echec de connexion."""
        instant = maintenant if maintenant is not None else time.time()
        self._echecs.setdefault(cle, []).append(instant)

    def reinitialiser(self, cle: str) -> None:
        """Efface l'historique d'echecs d'un client (connexion reussie)."""
        self._echecs.pop(cle, None)


def page_connexion(erreur: str | None = None, cible: str = "/") -> str:
    """Retourne la page HTML de connexion (autonome, sans ressource externe).

    Args:
        erreur: Message d'erreur a afficher, s'il y en a un.
        cible: URL vers laquelle rediriger apres connexion.

    Returns:
        Document HTML complet.
    """
    message = ""
    if erreur:
        message = f'<p class="erreur">{erreur}</p>'
    return f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<meta name="robots" content="noindex, nofollow" />
<title>EINHERJAR — connexion</title>
<style>
  :root {{
    --fond: #050505; --surface: #0a0a0a; --bord: #1c1c1c; --bord-haut: #2a2a2a;
    --texte: #e8e8e8; --texte-doux: #8a8a8a; --accent: #b0c4de; --erreur: #ff6b6b;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; min-height: 100vh; background: var(--fond); color: var(--texte);
    font-family: ui-sans-serif, -apple-system, "Segoe UI", Inter, sans-serif;
    display: grid; place-items: center; padding: 24px;
    background-image: radial-gradient(120% 80% at 50% -10%, #10151c 0%, transparent 60%);
  }}
  .carte {{
    width: 100%; max-width: 380px; background: var(--surface);
    border: 1px solid var(--bord); border-radius: 14px; padding: 32px 28px 26px;
    box-shadow: 0 24px 60px rgba(0,0,0,.55);
  }}
  .rune {{ font-size: 22px; letter-spacing: .35em; color: var(--accent); opacity: .85; }}
  h1 {{ font-size: 17px; font-weight: 600; letter-spacing: .22em; margin: 14px 0 4px; }}
  .sous {{ color: var(--texte-doux); font-size: 12px; letter-spacing: .04em; margin: 0 0 22px; }}
  label {{ display: block; font-size: 11px; letter-spacing: .12em; color: var(--texte-doux);
    text-transform: uppercase; margin: 0 0 6px; }}
  input {{
    width: 100%; padding: 11px 12px; margin-bottom: 16px; color: var(--texte);
    background: #060606; border: 1px solid var(--bord); border-radius: 9px;
    font-size: 14px; outline: none; transition: border-color .15s ease;
  }}
  input:focus {{ border-color: var(--bord-haut); }}
  button {{
    width: 100%; padding: 12px; border-radius: 9px; cursor: pointer;
    border: 1px solid var(--bord-haut); background: #111; color: var(--texte);
    font-size: 13px; letter-spacing: .16em; text-transform: uppercase;
    transition: background .15s ease, border-color .15s ease;
  }}
  button:hover {{ background: #161616; border-color: var(--accent); }}
  .erreur {{
    color: var(--erreur); font-size: 12.5px; margin: 0 0 16px;
    border: 1px solid rgba(255,107,107,.28); background: rgba(255,107,107,.06);
    padding: 9px 11px; border-radius: 8px;
  }}
  footer {{ margin-top: 20px; color: #4a4a4a; font-size: 11px; text-align: center; }}
</style>
</head>
<body>
  <main class="carte">
    <div class="rune">ᛖᛁᚾᚺᛖᚱᛃᚨᚱ</div>
    <h1>EINHERJAR</h1>
    <p class="sous">Acces reserve — identification requise</p>
    {message}
    <form method="post" action="/login">
      <input type="hidden" name="suite" value="{cible}" />
      <label for="utilisateur">Identifiant</label>
      <input id="utilisateur" name="utilisateur" autocomplete="username"
             autocapitalize="off" spellcheck="false" required autofocus />
      <label for="mot_de_passe">Mot de passe</label>
      <input id="mot_de_passe" name="mot_de_passe" type="password"
             autocomplete="current-password" required />
      <button type="submit">Entrer</button>
    </form>
    <footer>Session signee, valable 12 h</footer>
  </main>
</body>
</html>
"""


def extraire_cookie(entete_cookie: str | None, nom: str = COOKIE_NAME) -> str | None:
    """Lit une valeur de cookie dans l'en-tete `Cookie` brut.

    Args:
        entete_cookie: Contenu de l'en-tete `Cookie`.
        nom: Nom du cookie recherche.

    Returns:
        La valeur, ou None.
    """
    if not entete_cookie:
        return None
    for morceau in entete_cookie.split(";"):
        cle, _, valeur = morceau.strip().partition("=")
        if cle == nom:
            return valeur or None
    return None


def chemin_public(chemin: str) -> bool:
    """Indique si un chemin est accessible sans authentification.

    Args:
        chemin: Chemin demande (sans query string).

    Returns:
        True si le chemin est public.
    """
    return chemin in CHEMINS_PUBLICS or chemin.startswith(("/static/", "/assets/"))


def souvenir_sure(requete: Any) -> bool:
    """Indique si la requete est en HTTPS (cookie a marquer `Secure`).

    Sur un hebergeur, le TLS est termine par le proxy : l'en-tete
    `X-Forwarded-Proto` fait foi quand il est present.

    Args:
        requete: Requete (objet avec `url` et `headers`).

    Returns:
        True si la connexion est chiffree.
    """
    entetes = getattr(requete, "headers", {}) or {}
    transmis = (entetes.get("x-forwarded-proto") or "").split(",")[0].strip().lower()
    if transmis:
        return transmis == "https"
    return str(getattr(getattr(requete, "url", None), "scheme", "")).lower() == "https"
