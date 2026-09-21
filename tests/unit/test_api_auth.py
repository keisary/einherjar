"""Tests de l'authentification de l'application (auth.py).

Verrouille ce qui protege reellement l'acces a une API qui trade un compte reel :
signature du cookie, expiration, comparaison a temps constant, limitation des
tentatives, et non-fuite du mot de passe dans les messages d'erreur.

Aucun serveur n'est lance : le module est concu comme une brique pure, testable
sans FastAPI ni navigateur.
"""

from __future__ import annotations

import pytest

from einherjar.api import auth


ENV = {"EINHERJAR_AUTH_USERNAME": "jovanny", "EINHERJAR_AUTH_PASSWORD": "S3cret-Demo!"}


def test_identifiants_lus_dans_l_environnement():
    identifiants = auth.charger_identifiants(ENV)
    assert identifiants.utilisateur == "jovanny"
    assert identifiants.mot_de_passe == "S3cret-Demo!"
    assert identifiants.genere is False


def test_mot_de_passe_genere_si_absent_et_application_fermee_par_defaut():
    """Sans configuration, l'application n'est JAMAIS ouverte : mot de passe aleatoire."""
    identifiants = auth.charger_identifiants({})
    assert identifiants.genere is True
    assert len(identifiants.mot_de_passe) >= 12
    assert identifiants.mot_de_passe != auth.charger_identifiants({}).mot_de_passe


def test_utilisateur_par_defaut_quand_seul_le_mot_de_passe_est_fourni():
    identifiants = auth.charger_identifiants({"EINHERJAR_AUTH_PASSWORD": "abc12345"})
    assert identifiants.utilisateur == "einherjar"
    assert identifiants.mot_de_passe == "abc12345"


@pytest.mark.parametrize(
    ("utilisateur", "mot_de_passe", "attendu"),
    [
        ("jovanny", "S3cret-Demo!", True),
        ("jovanny", "mauvais", False),
        ("autre", "S3cret-Demo!", False),
        ("", "", False),
        ("JOVANNY", "S3cret-Demo!", False),
    ],
)
def test_verification_des_identifiants(utilisateur: str, mot_de_passe: str, attendu: bool):
    identifiants = auth.charger_identifiants(ENV)
    assert auth.verifier_identifiants(utilisateur, mot_de_passe, identifiants) is attendu


def test_jeton_valide_puis_expire():
    identifiants = auth.charger_identifiants(ENV)
    jeton = auth.creer_jeton(identifiants, duree_s=3600, maintenant=1_000_000, environ=ENV)
    assert auth.lire_jeton(jeton, identifiants, maintenant=1_000_000 + 10, environ=ENV) == "jovanny"
    assert auth.lire_jeton(jeton, identifiants, maintenant=1_000_000 + 3601, environ=ENV) is None


def test_jeton_falsifie_refuse():
    identifiants = auth.charger_identifiants(ENV)
    jeton = auth.creer_jeton(identifiants, maintenant=1_000_000, environ=ENV)
    utilisateur, expiration, signature = jeton.split(".")
    # Signature remplacee, expiration repoussee, utilisateur change : tous refuses.
    assert auth.lire_jeton(f"{utilisateur}.{expiration}.{'A' * len(signature)}", identifiants, 1_000_001, ENV) is None
    assert auth.lire_jeton(f"{utilisateur}.{int(expiration) + 99999}.{signature}", identifiants, 1_000_001, ENV) is None
    assert auth.lire_jeton(f"intrus.{expiration}.{signature}", identifiants, 1_000_001, ENV) is None


def test_jeton_invalide_ne_leve_pas():
    identifiants = auth.charger_identifiants(ENV)
    for jeton in (None, "", ".", "a.b", "a.b.c.d", "abc", "a.b.c"):
        assert auth.lire_jeton(jeton, identifiants, environ=ENV) is None


def test_changer_le_mot_de_passe_invalide_les_sessions():
    ancien = auth.charger_identifiants(ENV)
    jeton = auth.creer_jeton(ancien, maintenant=1_000_000, environ=ENV)
    nouveau = auth.charger_identifiants({**ENV, "EINHERJAR_AUTH_PASSWORD": "NouveauMotDePasse"})
    assert auth.lire_jeton(jeton, nouveau, maintenant=1_000_001, environ={**ENV, "EINHERJAR_AUTH_PASSWORD": "NouveauMotDePasse"}) is None


def test_cle_dediee_prioritaire_sur_le_mot_de_passe():
    identifiants = auth.charger_identifiants(ENV)
    env_avec_cle = {**ENV, "EINHERJAR_AUTH_SECRET": "cle-de-signature-independante"}
    jeton = auth.creer_jeton(identifiants, maintenant=1_000_000, environ=env_avec_cle)
    assert auth.lire_jeton(jeton, identifiants, maintenant=1_000_001, environ=env_avec_cle) == "jovanny"
    assert auth.lire_jeton(jeton, identifiants, maintenant=1_000_001, environ=ENV) is None


def test_limiteur_de_tentatives():
    limiteur = auth.LimiteurTentatives(maximum=3, fenetre_s=300)
    for i in range(3):
        assert limiteur.autorise("1.2.3.4", maintenant=1000 + i) is True
        limiteur.enregistrer_echec("1.2.3.4", maintenant=1000 + i)
    assert limiteur.autorise("1.2.3.4", maintenant=1003) is False
    # Une autre adresse n'est pas penalisee
    assert limiteur.autorise("5.6.7.8", maintenant=1003) is True
    # Apres la fenetre, l'adresse est de nouveau autorisee
    assert limiteur.autorise("1.2.3.4", maintenant=1000 + 301) is True
    # Une connexion reussie efface l'historique
    limiteur.reinitialiser("1.2.3.4")
    assert limiteur.autorise("1.2.3.4", maintenant=1000 + 302) is True


def test_chemins_publics():
    assert auth.chemin_public("/login") is True
    assert auth.chemin_public("/healthz") is True
    assert auth.chemin_public("/assets/index-abc.js") is True
    assert auth.chemin_public("/api/account") is False
    assert auth.chemin_public("/api/health") is False
    assert auth.chemin_public("/") is False


def test_extraction_du_cookie():
    entete = "autre=1; einherjar_session=abc.def.ghi; theme=dark"
    assert auth.extraire_cookie(entete) == "abc.def.ghi"
    assert auth.extraire_cookie("autre=1") is None
    assert auth.extraire_cookie(None) is None
    assert auth.extraire_cookie("einherjar_session=") is None


def test_page_de_connexion_ne_fuite_rien():
    page = auth.page_connexion("Identifiant ou mot de passe incorrect.")
    assert "EINHERJAR" in page
    assert 'type="password"' in page
    assert 'method="post"' in page
    assert 'action="/login"' in page
    assert "noindex" in page
    # Aucun identifiant ni mot de passe n'est ecrit dans la page
    assert "S3cret" not in page
    assert 'value="jovanny"' not in page


def test_cible_de_redirection_limitee_au_site():
    """Une redirection ouverte (URL externe) doit etre impossible."""
    from einherjar.api.server import _cible_sure

    assert _cible_sure("/positions") == "/positions"
    assert _cible_sure("//evil.example.com") == "/"
    assert _cible_sure("https://evil.example.com") == "/"
    assert _cible_sure(None) == "/"
    assert _cible_sure("") == "/"


class _FausseRequete:
    """Requete minimale pour tester la detection HTTPS."""

    def __init__(self, scheme: str, entetes: dict[str, str]) -> None:
        self.url = type("Url", (), {"scheme": scheme})()
        self.headers = entetes


def test_cookie_secure_seulement_en_https():
    assert auth.souvenir_sure(_FausseRequete("http", {})) is False
    assert auth.souvenir_sure(_FausseRequete("https", {})) is True
    # Derriere le proxy de l'hebergeur, l'en-tete fait foi
    assert auth.souvenir_sure(_FausseRequete("http", {"x-forwarded-proto": "https"})) is True
    assert auth.souvenir_sure(_FausseRequete("https", {"x-forwarded-proto": "http"})) is False
