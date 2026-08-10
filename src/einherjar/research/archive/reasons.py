"""
archive/reasons.py — Catalogue normalisé des raisons de rejet (S-3.6).

Toutes les raisons de rejet DOIVENT appartenir à ce catalogue. Aucune
raison libre n'est acceptée. Cela garantit la comparabilité et l'auditabilité.
"""

from __future__ import annotations

from einherjar.research.utils.types import RejectionReason


# Catalogue officiel (cf. ONTOLOGY.md S-3.6)
OFFICIAL_REASONS: tuple[RejectionReason, ...] = (
    RejectionReason.DSR_FAIL,
    RejectionReason.PBO_FAIL,
    RejectionReason.BOOTSTRAP_CI_FAIL,
    RejectionReason.N_TRADES_FAIL,
    RejectionReason.CROISSANCE_FAIL,
    RejectionReason.CROSS_ASSET_FAIL,
    RejectionReason.DD_FAIL,
    RejectionReason.DIVERSITY_FAIL,
    RejectionReason.ALREADY_IN_ARCHIVE,
    RejectionReason.SEMANTIC_CHANGED,
    RejectionReason.OTHER,
    RejectionReason.EVALUATION_ERROR,
    RejectionReason.TIMEOUT,
    RejectionReason.MEMORY_ERROR,
)


def is_valid_reason(reason: str | RejectionReason) -> bool:
    """Vérifie qu'une raison (string ou enum) est dans le catalogue officiel."""
    if isinstance(reason, RejectionReason):
        return reason in OFFICIAL_REASONS
    try:
        return RejectionReason(reason) in OFFICIAL_REASONS
    except ValueError:
        return False


def normalize_reason(reason: str | RejectionReason) -> RejectionReason:
    """Normalise une raison (string ou enum) en `RejectionReason`.

    Lève `ValueError` si la raison n'est pas dans le catalogue.
    """
    if isinstance(reason, RejectionReason):
        if reason not in OFFICIAL_REASONS:
            raise ValueError(f"Raison hors catalogue: {reason}")
        return reason
    try:
        r = RejectionReason(reason)
    except ValueError as exc:
        raise ValueError(
            f"Raison de rejet inconnue: {reason!r}. "
            f"Catalogue officiel: {[r.value for r in OFFICIAL_REASONS]}"
        ) from exc
    if r not in OFFICIAL_REASONS:
        raise ValueError(f"Raison hors catalogue: {r}")
    return r
