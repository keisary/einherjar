"""
utils/fingerprint.py — Fingerprint canonique des Einhers (S-3.7 + Archive).

Deux empreintes :
  - structurel : hash de condition_tree + direction + universe + amplitude + sl + tp
                  (anti-doublon exact, déterministe)
  - comportemental : hash des descripteurs comportementaux arrondis à une grille stable
                     (détection d'équivalence économique)

La concaténation des deux constitue le `fingerprint` global d'un Einher.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Optional

from einherjar.research.utils.types import Einher, Hypothesis


# --------------------------------------------------------------------------- #
# Sérialisation canonique
# --------------------------------------------------------------------------- #


def _canonical_json(obj: Any) -> str:
    """Sérialise un objet en JSON canonique (clés triées, pas d'espaces).

    Garantit que deux objets sémantiquement égaux produisent le même hash,
    indépendamment de l'ordre des clés.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _hash_canonical(obj: Any, algo: str = "sha256") -> str:
    """Hash canonique d'un objet JSON-sérialisable."""
    payload = _canonical_json(obj).encode("utf-8")
    h = hashlib.new(algo)
    h.update(payload)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Fingerprints
# --------------------------------------------------------------------------- #


def fingerprint_structurel(
    hypothesis: Hypothesis,
    sl_price: float,
    tp_price: float,
    algo: str = "sha256",
) -> str:
    """Empreinte structurelle : condition + direction + universe + amplitude + SL + TP.

    Anti-doublon exact. Si deux Einhers ont la même empreinte structurelle,
    ils sont structurellement identiques (sur la même version de sérialisation).
    """
    payload = {
        "condition_tree": hypothesis.condition_tree.to_dict() if hasattr(hypothesis.condition_tree, "to_dict") else str(hypothesis.condition_tree),
        "direction": hypothesis.direction.value,
        "universe": hypothesis.universe.to_dict(),
        "amplitude": hypothesis.amplitude.to_dict(),
        "sl_price": round(float(sl_price), 9),
        "tp_price": round(float(tp_price), 9),
    }
    return _hash_canonical(payload, algo=algo)


def fingerprint_comportemental(
    descriptors: dict[str, Any],
    rounding_decimals: int = 3,
    algo: str = "sha256",
) -> str:
    """Empreinte comportementale : descripteurs arrondis à une grille stable.

    Permet de détecter des Einhers structurellement différents mais
    économiquement équivalents (sur la même époque de données).

    Les descripteurs DOIVENT être des nombres (float) ou des tuples/strings
    sérialisables. Les arrondis se font sur les floats uniquement.
    """
    rounded = _round_floats(descriptors, decimals=rounding_decimals)
    return _hash_canonical(rounded, algo=algo)


def fingerprint_global(
    structurel: str,
    comportemental: str,
) -> str:
    """Concaténation canonique des deux empreintes (séparateur `:`)."""
    return f"{structurel}:{comportemental}"


def fingerprint_einher(einher: Einher) -> str:
    """Calcule le fingerprint global d'un Einher à partir de ses deux empreintes."""
    return fingerprint_global(
        einher.fingerprint_structurel,
        einher.fingerprint_comportemental,
    )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _round_floats(obj: Any, decimals: int) -> Any:
    """Récursivement arrondit tous les floats d'une structure."""
    if isinstance(obj, float):
        return round(obj, decimals)
    if isinstance(obj, (int, str, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {k: _round_floats(v, decimals) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round_floats(v, decimals) for v in obj]
    # Type non supporté : on laisse tel quel
    return obj
