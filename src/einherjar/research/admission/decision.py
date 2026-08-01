"""admission/decision.py — Décision d'admission globale.

Combine :
  1. Les 7 critères d'admission S-3.4 (admission/criteria.py)
  2. La diversité (admission/diversity.py) :
     - Quotas structurels (famille, type, direction)
     - Descripteurs comportementaux vs corpus (signal_overlap, ret_corr)
  3. La déduplication par fingerprint canonique contre l'Archive (I-6)

Returns:
  - Verdict global (admis / rejeté avec raison)
  - Effet de bord : ajout à l'archive si rejeté, ajout au corpus si admis

Conforme à ONTOLOGY.md S-3.4, S-3.5, § 8 (Archive) et ALGORITHME_RESEARCH.md § 10.2 étape 5.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from einherjar.research.admission.criteria import (
    AdmissionVerdict,
    evaluate_all_criteria,
)
from einherjar.research.admission.diversity import (
    BehavioralDescriptors,
    QuotaReport,
    evaluate_quotas,
    extract_dominant_family,
    extract_dominant_type,
)
from einherjar.research.archive.schema import ArchiveEntry
from einherjar.research.archive.store import (
    append_entry,
    has_comportemental_fingerprint,
    has_fingerprint,
)
from einherjar.research.config.loader import EinherjarConfig
from einherjar.research.engine.evaluator import CalibratedParams
from einherjar.research.utils.fingerprint import fingerprint_structurel
from einherjar.research.utils.types import (
    Direction,
    MesuresBrutes,
    RejectionReason,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Décision finale
# --------------------------------------------------------------------------- #


@dataclass
class AdmissionDecision:
    """Décision finale pour une hypothèse candidate.

    Attributes:
        admitted: True si admis au corpus, False si rejeté.
        primary_reason: Raison principale du rejet (None si admis).
        criteria_verdict: Verdict des 7 critères S-3.4.
        quota_report: Rapport des quotas structurels.
        diversity_passed: True si la diversité (comportementale + structurelle) passe.
        dedup_passed: True si pas de doublon (fingerprint structurel + comportemental) dans l'Archive.
        archive_entry: Entrée créée (si rejeté, pour traçabilité).
        meta: Métadonnées libres.
    """

    admitted: bool
    primary_reason: RejectionReason | None
    criteria_verdict: AdmissionVerdict
    quota_report: QuotaReport
    diversity_passed: bool
    dedup_passed: bool
    archive_entry: ArchiveEntry | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Sérialisation pour logs / persistance."""
        return {
            "admitted": self.admitted,
            "primary_reason": self.primary_reason.value if self.primary_reason else None,
            "criteria_verdict": self.criteria_verdict.to_dict(),
            "quota_report": self.quota_report.to_dict(),
            "diversity_passed": self.diversity_passed,
            "dedup_passed": self.dedup_passed,
            "meta": self.meta,
        }


# --------------------------------------------------------------------------- #
# Décideur
# --------------------------------------------------------------------------- #


class AdmissionDecider:
    """Décideur d'admission global.

    Combine critères S-3.4 + diversité + déduplication.
    Effet de bord : append à l'Archive si rejeté, prêt à ajouter au corpus si admis.

    Attributes:
        config: Configuration chargée.
        data_version: Version de données (pour fingerprint + Archive).
        seed: Graine RNG maître.
    """

    def __init__(
        self,
        config: EinherjarConfig,
        data_version: str,
        seed: int = 42,
    ) -> None:
        """Initialise le décideur d'admission.

        Args:
            config: Configuration chargée.
            data_version: Version de données (pour fingerprint + Archive).
            seed: Graine RNG maître.
        """
        self.config = config
        self.data_version = data_version
        self.seed = seed
        logger.info("AdmissionDecider instancié (data_version=%s)", data_version)

    def decide(
        self,
        hypothesis_id: str,
        condition_tree: Any,
        direction: Direction,
        universe: Any,
        amplitude: Any,
        calibrated: CalibratedParams,
        mesures_val: MesuresBrutes,
        returns_val: list[float],
        *,
        current_corpus_fracs: dict[str, dict[str, float]] | None = None,
        corpus_signal_dates: list[tuple[int, ...]] | None = None,
        corpus_ret_series: list[tuple[float, ...]] | None = None,
        signal_indices: tuple[int, ...] = (),
        n_indep_trials: int = 1,
    ) -> AdmissionDecision:
        """Prend la décision d'admission complète.

        Args:
            hypothesis_id: ID de l'hypothèse candidate.
            condition_tree: Arbre de conditions.
            direction: Direction de l'Einher.
            universe: Universe (assets, timeframes).
            amplitude: Amplitude cible.
            calibrated: CalibratedParams (N, SL, TP figés).
            mesures_val: MesuresBrutes sur le val.
            returns_val: Liste des rendements nets sur val.
            current_corpus_fracs: Fractions actuelles du corpus (par family/type/direction).
            corpus_signal_dates: Liste des signal_dates de chaque Einher du corpus.
            corpus_ret_series: Liste des ret_series de chaque Einher du corpus.
            signal_indices: Indices des signaux retenus sur le val.
            n_indep_trials: Nombre d'essais indépendants pour le DSR.

        Returns:
            AdmissionDecision.
        """
        # 1. Critères S-3.4.
        criteria_verdict = evaluate_all_criteria(
            mesures=mesures_val,
            returns=returns_val,
            config=self.config,
            n_indep_trials=n_indep_trials,
        )

        # 2. Fingerprints (structurel + comportemental).
        fp_struct = fingerprint_structurel(
            hypothesis=_build_hypothesis_for_fp(
                hypothesis_id, condition_tree, direction, universe, amplitude,
            ),
            sl_n_atr=calibrated.sl_n_atr,
            tp_n_atr=calibrated.tp_n_atr,
        )
        descriptors = _build_descriptors(
            mesures=mesures_val,
            signal_indices=signal_indices,
            corpus_signal_dates=corpus_signal_dates or [],
            corpus_ret_series=corpus_ret_series or [],
        )
        fp_comport = descriptors.fingerprint(
            rounding_decimals=int(self.config.evaluation["fingerprint"]["behavioral"]["rounding_decimals"]),
        )

        # 3. Déduplication contre l'Archive.
        dedup_struct = has_fingerprint(fp_struct, self.data_version)
        dedup_comport = has_comportemental_fingerprint(fp_comport, self.data_version)
        dedup_passed = (not dedup_struct) and (not dedup_comport)

        # 4. Diversité comportementale.
        div_overlap_ok = descriptors.signal_overlap_vs_corpus <= float(
            self.config.thresholds["diversity"]["signal_overlap_max"],
        )
        div_corr_ok = descriptors.ret_corr_vs_corpus <= float(
            self.config.thresholds["diversity"]["ret_corr_max"],
        )
        diversity_behavioral_ok = div_overlap_ok and div_corr_ok

        # 5. Quotas structurels.
        current = current_corpus_fracs or {"family": {}, "type": {}, "direction": {}}
        new_family = extract_dominant_family(condition_tree, self.config)
        new_type = extract_dominant_type(condition_tree, self.config)
        new_dir = direction.value
        quota_report = evaluate_quotas(
            new_family=new_family,
            new_type=new_type,
            new_direction=new_dir,
            current_family_fracs=current.get("family", {}),
            current_type_fracs=current.get("type", {}),
            current_direction_fracs=current.get("direction", {}),
            config=self.config,
        )

        # 6. Décision globale : tous les verrous doivent passer.
        all_ok = (
            criteria_verdict.passed
            and dedup_passed
            and diversity_behavioral_ok
            and quota_report.passed
        )

        # 7. Construction de la raison principale.
        primary_reason: RejectionReason | None = None
        if not all_ok:
            primary_reason = _first_failure_reason(
                criteria_verdict=criteria_verdict,
                dedup_struct=dedup_struct,
                dedup_comport=dedup_comport,
                diversity_overlap_ok=div_overlap_ok,
                diversity_corr_ok=div_corr_ok,
                quota_report=quota_report,
            )

        # 8. Si rejeté, on crée une entrée d'archive (effet de bord).
        archive_entry: ArchiveEntry | None = None
        if not all_ok:
            archive_entry = _make_archive_entry(
                hypothesis_id=hypothesis_id,
                primary_reason=primary_reason,
                calibrated=calibrated,
                mesures_val=mesures_val,
                fp_struct=fp_struct,
                fp_comport=fp_comport,
                data_version=self.data_version,
                seed=self.seed,
            )
            try:
                append_entry(archive_entry)
                logger.info("Rejet archivé : %s (raison=%s)", hypothesis_id, primary_reason)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Échec append archive pour %s : %s", hypothesis_id, exc)

        return AdmissionDecision(
            admitted=all_ok,
            primary_reason=primary_reason,
            criteria_verdict=criteria_verdict,
            quota_report=quota_report,
            diversity_passed=diversity_behavioral_ok,
            dedup_passed=dedup_passed,
            archive_entry=archive_entry,
            meta={
                "fp_struct": fp_struct,
                "fp_comport": fp_comport,
                "dedup_struct": dedup_struct,
                "dedup_comport": dedup_comport,
                "div_overlap": descriptors.signal_overlap_vs_corpus,
                "div_corr": descriptors.ret_corr_vs_corpus,
            },
        )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _build_hypothesis_for_fp(
    hypothesis_id: str,
    condition_tree: Any,
    direction: Direction,
    universe: Any,
    amplitude: Any,
) -> Any:
    """Construit un Hypothesis minimal pour le calcul de fingerprint (pas un vrai Hypothesis, juste ce qu'il faut)."""
    from einherjar.research.utils.types import Hypothesis
    return Hypothesis(
        id=hypothesis_id,
        condition_tree=condition_tree,
        amplitude=amplitude,
        direction=direction,
        universe=universe,
        cooldown_k=5,
    )


def _build_descriptors(
    mesures: MesuresBrutes,
    signal_indices: tuple[int, ...],
    corpus_signal_dates: list[tuple[int, ...]],
    corpus_ret_series: list[tuple[float, ...]],
) -> BehavioralDescriptors:
    """Construit un BehavioralDescriptors (delegation à diversity.py)."""
    from einherjar.research.admission.diversity import _build_descriptors as _bd
    return _bd(
        mesures=mesures,
        signal_indices=signal_indices,
        corpus_signal_dates=corpus_signal_dates,
        corpus_ret_series=corpus_ret_series,
    )


def _first_failure_reason(
    *,
    criteria_verdict: AdmissionVerdict,
    dedup_struct: bool,
    dedup_comport: bool,
    diversity_overlap_ok: bool,
    diversity_corr_ok: bool,
    quota_report: QuotaReport,
) -> RejectionReason:
    """Détermine la raison principale (premier verrou qui échoue)."""
    if not criteria_verdict.passed and criteria_verdict.primary_reason is not None:
        return criteria_verdict.primary_reason
    if dedup_struct or dedup_comport:
        return RejectionReason.ALREADY_IN_ARCHIVE
    if not diversity_overlap_ok or not diversity_corr_ok:
        return RejectionReason.DIVERSITY_FAIL
    if not quota_report.passed:
        return RejectionReason.DIVERSITY_FAIL
    return RejectionReason.OTHER


def _make_archive_entry(
    *,
    hypothesis_id: str,
    primary_reason: RejectionReason,
    calibrated: CalibratedParams,
    mesures_val: MesuresBrutes,
    fp_struct: str,
    fp_comport: str,
    data_version: str,
    seed: int,
) -> ArchiveEntry:
    """Construit une ArchiveEntry (pas encore append — c'est l'appelant qui le fait)."""
    return ArchiveEntry(
        id=f"rej_{hypothesis_id}",
        type_élément="hypothesis",
        raison_rejet=primary_reason,
        date_rejet=ArchiveEntry.now_utc(),
        data_version=data_version,
        seed=seed,
        splits={},  # À remplir par le caller si dispo
        costs_simulated=mesures_val.costs_applied,
        sl_tp_source="from_train",
        mesures_brutes_val=mesures_val,
        fingerprint_structurel=fp_struct,
        fingerprint_comportemental=fp_comport,
        fingerprint=f"{fp_struct}:{fp_comport}",
        element_ref_id=hypothesis_id,
    )
