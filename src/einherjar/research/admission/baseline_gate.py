"""admission/baseline_gate.py — Admission légère pour les baselines.

Contrairement à `AdmissionDecider` (qui append chaque rejet à l'Archive et
évalue diversité/dédup/quota), la baseline_admission_fn est volontairement
simplifiée : elle applique UNIQUEMENT les 7 critères S-3.4 (criteria.py)
et incrémente un compteur d'essais pour le DSR.

Philosophie : un baseline runner peut rejeter des milliers d'hypothèses
en quelques minutes. L'archive doit rester l'historique des rejets
PONDÉRÉS (un Einher candidat à l'admission finale), pas un dump de
toutes les hypothèses testées par Random.

Usage :
    fn, counter = make_baseline_admission_fn(config)
    runner = BaselineRunner(engine=engine, admission_fn=fn)
    report = runner.run(...)
    print(f"{counter['n']} essais, {report.n_passed_admission} admis")

Conforme à ALGORITHME_RESEARCH.md § 10.2 étape 5 (volet critères uniquement).
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from einherjar.research.admission.criteria import evaluate_all_criteria
from einherjar.research.config.loader import EinherjarConfig

logger = logging.getLogger(__name__)


# Type du callable attendu par BaselineRunner.run(admission_fn=...).
BaselineAdmissionFn = Callable[[Any, Any, Any], bool]


def make_baseline_admission_fn(
    config: EinherjarConfig,
) -> tuple[BaselineAdmissionFn, dict[str, int]]:
    """Construit l'admission_fn consommée par le baseline runner.

    Args:
        config: Configuration chargée (pour les seuils des 7 critères).

    Returns:
        Tuple (admission_fn, counter) où :
          - admission_fn(hypothesis, calibrated, mesures_val) -> bool
            applique les 7 critères S-3.4 et incrémente le compteur d'essais
            (utilisé comme `n_indep_trials` pour le DSR, afin de pénaliser
            correctement le multiple-testing sur les N hypothèses testées).
          - counter est un dict {"n": int, "n_admitted": int} qui expose
            le nb d'essais et le nb d'admis observés (utile pour les logs).
    """
    counter: dict[str, int] = {"n": 0, "n_admitted": 0}

    def admission_fn(
        hypothesis: Any,
        calibrated: Any,
        mesures_val: Any,
    ) -> bool:
        counter["n"] += 1
        # Séquence de rendements nets sur val (un point par trade).
        returns_val = [t.ret_pct_net for t in mesures_val.trades]
        verdict = evaluate_all_criteria(
            mesures=mesures_val,
            returns=returns_val,
            config=config,
            n_indep_trials=counter["n"],
        )
        if verdict.passed:
            counter["n_admitted"] += 1
        else:
            logger.debug(
                "Baseline rejet : %s — raison=%s",
                hypothesis.id,
                verdict.primary_reason.value if verdict.primary_reason else "OTHER",
            )
        return verdict.passed

    return admission_fn, counter
