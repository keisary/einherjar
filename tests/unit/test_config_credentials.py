"""Tests du chargement des identifiants cTrader (fichier local et environnement).

Un deploiement heberge (Render) n'embarque pas `config/credentials.json` : les
variables `EINHERJAR_*` doivent suffire. Ces tests verrouillent les deux chemins et
la priorite donnee a l'environnement.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from einherjar.config.credentials import (
    charger_credentials,
    credentials_complets,
)

VARIABLES = (
    "EINHERJAR_CLIENT_ID",
    "EINHERJAR_CLIENT_SECRET",
    "EINHERJAR_ACCESS_TOKEN",
    "EINHERJAR_ACCOUNT_ID",
    "EINHERJAR_ENVIRONMENT",
    "EINHERJAR_HOST",
    "EINHERJAR_PORT",
    "EINHERJAR_BROKER_NAME",
)


@pytest.fixture(autouse=True)
def _environnement_propre(monkeypatch: pytest.MonkeyPatch) -> None:
    """Aucune variable EINHERJAR_* heritee du shell de test."""
    for variable in VARIABLES:
        monkeypatch.delenv(variable, raising=False)


def test_fichier_absent_et_environnement_absent_retourne_none(tmp_path: Path) -> None:
    assert charger_credentials(tmp_path / "absent.json") is None


def test_environnement_seul_suffit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EINHERJAR_CLIENT_ID", "12345")
    monkeypatch.setenv("EINHERJAR_CLIENT_SECRET", "secret")
    monkeypatch.setenv("EINHERJAR_ACCESS_TOKEN", "jeton")
    monkeypatch.setenv("EINHERJAR_ACCOUNT_ID", "48706210")

    creds = charger_credentials(tmp_path / "absent.json")

    assert creds is not None
    assert creds["client_id"] == "12345"
    assert creds["account_id"] == 48706210  # converti en entier
    assert creds["environment"] == "demo"  # deduit de l'absence de host live
    assert creds["host"] == "demo.ctraderapi.com"
    assert creds["port"] == 5035
    assert credentials_complets(creds) == (True, [])


def test_environnement_prime_sur_le_fichier(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fichier = tmp_path / "credentials.json"
    fichier.write_text(
        json.dumps(
            {
                "client_id": "du-fichier",
                "client_secret": "du-fichier",
                "access_token": "du-fichier",
                "account_id": 111,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("EINHERJAR_CLIENT_ID", "de-l-environnement")

    creds = charger_credentials(fichier)

    assert creds is not None
    assert creds["client_id"] == "de-l-environnement"
    assert creds["account_id"] == 111  # conserve depuis le fichier


def test_environnement_live_choisit_le_bon_host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EINHERJAR_CLIENT_ID", "1")
    monkeypatch.setenv("EINHERJAR_CLIENT_SECRET", "2")
    monkeypatch.setenv("EINHERJAR_ACCESS_TOKEN", "3")
    monkeypatch.setenv("EINHERJAR_ACCOUNT_ID", "4")
    monkeypatch.setenv("EINHERJAR_ENVIRONMENT", "live")

    creds = charger_credentials(tmp_path / "absent.json")

    assert creds is not None
    assert creds["environment"] == "live"
    assert creds["host"] == "live.ctraderapi.com"


def test_identifiants_incomplets_sont_signales(tmp_path: Path) -> None:
    fichier = tmp_path / "credentials.json"
    fichier.write_text(json.dumps({"client_id": "1"}), encoding="utf-8")

    creds = charger_credentials(fichier)

    assert creds is not None
    complets, manquants = credentials_complets(creds)
    assert complets is False
    assert "access_token" in manquants
    assert "account_id" in manquants


def test_fichier_json_invalide_ne_plante_pas(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fichier = tmp_path / "credentials.json"
    fichier.write_text("{pas du json", encoding="utf-8")
    monkeypatch.setenv("EINHERJAR_CLIENT_ID", "1")
    monkeypatch.setenv("EINHERJAR_CLIENT_SECRET", "2")
    monkeypatch.setenv("EINHERJAR_ACCESS_TOKEN", "3")
    monkeypatch.setenv("EINHERJAR_ACCOUNT_ID", "99")

    creds = charger_credentials(fichier)

    assert creds is not None
    assert creds["account_id"] == 99
