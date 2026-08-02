"""
archive/schema.py — Schéma d'une entrée d'archive (ONTOLOGY.md § 8).

Toute entrée d'archive doit respecter ce schéma (validation à l'écriture).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from einherjar.research.archive.reasons import normalize_reason
from einherjar.research.utils.types import MesuresBrutes, RejectionReason


@dataclass
class ArchiveEntry:
    """Une entrée d'archive = un rejet (hypothèse ou einher rejeté).

    P1 #8 : le schéma capture TOUT ce qui est nécessaire pour audit
    et reproductibilité :
      - Règle BNF canonique (via fingerprint_structurel)
      - Paramètres figés (via mesures_brutes_val + seed + data_version)
      - Version données (data_version, seed, splits)
      - Métriques (mesures_brutes, metriques_portefeuille, bootstrap_ci, DSR, PBO)
      - Décision (raison_rejet, date_rejet)
      - Séries de retours (ret_series, ajoutées en P1 #8 pour la diversité)
      - Descripteurs comportementaux
    """

    id: str
    type_élément: str                       # 'hypothesis' | 'einher'
    raison_rejet: RejectionReason
    date_rejet: str                         # ISO 8601 UTC

    # Contexte de reproductibilité
    data_version: str
    seed: int
    splits: dict[str, Any]                  # {train: [...], val: [...], holdout: [...], purge_window, embargo_bougies}
    costs_simulated: dict[str, float]       # {spread_pct, commission_pct, slippage_pct}
    sl_tp_source: str                       # 'from_train' (seul autorisé)

    # Snapshot métriques complet
    mesures_brutes_train: Optional[MesuresBrutes] = None
    mesures_brutes_val: Optional[MesuresBrutes] = None
    metriques_portefeuille_val: dict[str, float] = field(default_factory=dict)
    bootstrap_ci_val: dict[str, float] = field(default_factory=dict)
    deflated_sharpe_ratio: Optional[float] = None
    probability_of_backtest_overfitting: Optional[float] = None
    descriptors_comportementaux: dict[str, Any] = field(default_factory=dict)

    # Fingerprints
    fingerprint_structurel: str = ""
    fingerprint_comportemental: str = ""
    fingerprint: str = ""

    # P1 #8 : série de rendements nets par trade (pour corrélation diversité).
    ret_series: tuple[float, ...] = ()

    # Référence à l'élément rejeté (juste l'id, pas l'objet complet — déjà dans l'Archive)
    element_ref_id: str = ""

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "type_élément": self.type_élément,
            "raison_rejet": self.raison_rejet.value,
            "date_rejet": self.date_rejet,
            "data_version": self.data_version,
            "seed": self.seed,
            "splits": self.splits,
            "costs_simulated": self.costs_simulated,
            "sl_tp_source": self.sl_tp_source,
            "metriques_portefeuille_val": self.metriques_portefeuille_val,
            "bootstrap_ci_val": self.bootstrap_ci_val,
            "deflated_sharpe_ratio": self.deflated_sharpe_ratio,
            "probability_of_backtest_overfitting": self.probability_of_backtest_overfitting,
            "descriptors_comportementaux": self.descriptors_comportementaux,
            "fingerprint_structurel": self.fingerprint_structurel,
            "fingerprint_comportemental": self.fingerprint_comportemental,
            "fingerprint": self.fingerprint,
            "ret_series": list(self.ret_series),
            "element_ref_id": self.element_ref_id,
        }
        if self.mesures_brutes_train is not None:
            d["mesures_brutes_train"] = self.mesures_brutes_train.to_dict()
        if self.mesures_brutes_val is not None:
            d["mesures_brutes_val"] = self.mesures_brutes_val.to_dict()
        return d

    def validate(self) -> list[str]:
        """Vérifie la conformité de l'entrée au schéma. Retourne la liste des erreurs (vide = OK)."""
        errors: list[str] = []
        if not self.id:
            errors.append("id manquant")
        if self.type_élément not in ("hypothesis", "einher"):
            errors.append(f"type_élément invalide: {self.type_élément!r}")
        try:
            normalize_reason(self.raison_rejet)
        except ValueError as exc:
            errors.append(f"raison_rejet invalide: {exc}")
        if not self.data_version:
            errors.append("data_version manquant")
        if not isinstance(self.seed, int):
            errors.append(f"seed doit être un int, got {type(self.seed).__name__}")
        if self.sl_tp_source != "from_train":
            errors.append(f"sl_tp_source doit être 'from_train', got {self.sl_tp_source!r}")
        if not self.fingerprint_structurel:
            errors.append("fingerprint_structurel manquant")
        return errors

    @staticmethod
    def now_utc() -> str:
        return datetime.now(timezone.utc).isoformat()
