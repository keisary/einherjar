"""Regles de sortie de position.

Repartition des responsabilites :
- le BROKER gere les sorties de PRIX (TP/SL sont transmis a l'ouverture, cf.
  `CTraderAdapter.place_order`) ;
- EINHERJAR gere la sortie de DUREE (`Einher.max_holding`, issue de
  `amplitude_bars` du corpus). Sans cette regle, une position qui n'atteint ni TP
  ni SL resterait ouverte indefiniment — c'est la regle de tenue du CDC.

Ce module est volontairement pur (aucun broker, aucune I/O) pour rester testable.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from typing import Any

# Minutes par timeframe : ce module doit rester utilisable hors du broker.
TF_MINUTES: dict[str, int] = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "4h": 240,
    "1d": 1440,
    "1w": 10080,
}

_DUREE_RE = re.compile(r"^(\d+)\s*(d|h|m)$", re.IGNORECASE)
_BARS_RE = re.compile(r"^(\d+)\s*bars?$", re.IGNORECASE)


def parse_duree(texte: str | None, timeframes: list[str] | None = None) -> timedelta | None:
    """Convertit une duree texte en `timedelta`.

    Formats produits par le corpus (`corpus_bridge._holding_str`) : `45m`, `6h`,
    `3d`, ou `12 bars` quand le timeframe n'est pas connu a la construction.

    Args:
        texte: Duree textuelle (`6h`, `12 bars`, ...).
        timeframes: Timeframes de l'einher, utilises pour convertir `N bars`.

    Returns:
        La duree, ou None si elle n'est pas interpretable (jamais une valeur
        inventee : mieux vaut ne pas fermer que fermer au hasard).
    """
    if not texte or not isinstance(texte, str):
        return None
    valeur = texte.strip()

    correspondance = _DUREE_RE.match(valeur)
    if correspondance:
        nombre = int(correspondance.group(1))
        if nombre <= 0:
            return None
        unite = correspondance.group(2).lower()
        return {
            "d": timedelta(days=nombre),
            "h": timedelta(hours=nombre),
            "m": timedelta(minutes=nombre),
        }[unite]

    correspondance = _BARS_RE.match(valeur)
    if correspondance:
        nombre = int(correspondance.group(1))
        if nombre <= 0 or not timeframes:
            return None
        minutes = TF_MINUTES.get(str(timeframes[0]).lower())
        if minutes is None:
            return None
        return timedelta(minutes=nombre * minutes)

    return None


def doit_fermer(
    position: Any,
    einher: Any,
    maintenant: datetime | None = None,
) -> str | None:
    """Motif de fermeture d'une position, ou None si elle reste ouverte.

    Args:
        position: Position du broker (doit porter `opened_at`).
        einher: Einher emetteur (porte `max_holding`).
        maintenant: Instant de reference (defaut : maintenant UTC).

    Returns:
        Le motif de fermeture, ou None.
    """
    if position is None or einher is None:
        return None

    limite = parse_duree(
        getattr(einher, "max_holding", None),
        getattr(einher, "timeframes", None),
    )
    if limite is None:
        return None

    ouvert = getattr(position, "opened_at", None)
    if not isinstance(ouvert, datetime):
        return None
    if ouvert.tzinfo is None:
        ouvert = ouvert.replace(tzinfo=UTC)

    instant = maintenant or datetime.now(UTC)
    if instant.tzinfo is None:
        instant = instant.replace(tzinfo=UTC)

    return "duree_max_depassee" if instant - ouvert >= limite else None
