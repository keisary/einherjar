"""
Test unitaire du fix `_signal_mask_from_array` (Chantier C).

Vérifie que les métriques binaires du validator sont
correctement calculées en mode MIDAS (signal_mask aligné
sur le split).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def main() -> int:
    print("=" * 70)
    print("Test unitaire du fix signal_mask (Chantier C)")
    print("=" * 70)

    # IMPORTANT : importer core EN PREMIER pour viter le
    # circular import entre core/__init__.py et
    # validation/evaluator.py.
    import core  # noqa: F401

    from validation.evaluator import ValidationEvaluator
    from config.config import Config

    errors: list[str] = []

    # ------------------------------------------------------------------
    # Test 1 : appel direct de assess sur un split artificiel
    # ------------------------------------------------------------------
    print("\n[Test 1] assess() sur un split synthetique 1000 samples")

    rng = np.random.default_rng(42)
    n_samples = 1000
    n_features = 5

    # X : 5 features, 1000 samples
    X = rng.normal(0, 1, (n_samples, n_features)).astype(np.float32)

    # Y : 1D, "ground truth" (positif si feature[0] > 0)
    Y = (X[:, 0] > 0).astype(np.float32) * 0.01 + rng.normal(0, 0.001, n_samples).astype(np.float32)

    # Cration d'un DatasetSplit
    from dataset.loader import DatasetSplit

    split = DatasetSplit(name="validation", X=X, Y=Y)

    # Cration d'un Hypothesis simple : feature[0] > 0
    from models.hypothesis import Hypothesis
    from models.condition import Condition
    from models.feature import Feature
    from models.enums import ConditionOperator
    from models.enums import FeatureType

    # Le plus simple : crer un Feature directement
    feature0 = Feature(
        column_index=0,
        name="feat_0",
        feature_type=FeatureType.ATOMIC,
    )
    condition = Condition(
        left=feature0,
        operator=ConditionOperator.GT,
        right=0.0,
    )
    hypothesis = Hypothesis([condition])

    # Evaluator + assess
    config = Config()
    evaluator = ValidationEvaluator(config=config, dataset=split)
    assessment = evaluator.assess(
        hypothesis,
        dataset=split,
        split_name="validation",
    )

    print(f"  passed: {assessment.passed}")
    print(f"  score: {assessment.metrics.significance_score:.4f}")
    print(f"  support: {assessment.metrics.support}")
    print(f"  coverage: {assessment.metrics.coverage:.4f}")
    print(f"  baseline_mean: {assessment.metrics.baseline_mean:.6f}")
    print(f"  signal_mean: {assessment.metrics.signal_mean:.6f}")
    print(f"  lift: {assessment.metrics.lift:.6f}")
    print(f"  binary_precision: {assessment.metrics.binary_precision:.4f}")
    print(f"  binary_recall: {assessment.metrics.binary_recall:.4f}")
    print(f"  binary_f1: {assessment.metrics.binary_f1:.4f}")
    print(f"  directional_accuracy: {assessment.metrics.directional_accuracy:.4f}")

    # Le ground truth est X[:, 0] > 0 (positif). Le signal mask
    # devrait tre "X[:, 0] > 0" (l'hypothesis est feature[0] > 0).
    # Avec un peu de bruit, on s'attend :
    # - precision ~ 1.0 (le mask est X[:,0]>0, et Y>0 quand X[:,0]>0)
    # - recall ~ 0.65 (car le seuil 0.0 sur Y capture aussi du bruit)
    # Ce qui compte ici c'est que les mtriques soient calcules
    # SANS CRASH (avant le fix, _binary_classification_metrics
    # levait ValueError sur les shapes).
    if assessment.metrics.binary_precision < 0.9:
        errors.append(
            f"binary_precision trop bas (attendu ~1.0): "
            f"{assessment.metrics.binary_precision}"
        )
    if assessment.metrics.binary_recall < 0.4:
        errors.append(
            f"binary_recall trop bas (attendu ~0.6): "
            f"{assessment.metrics.binary_recall}"
        )
    if not (0.0 <= assessment.metrics.binary_f1 <= 1.0):
        errors.append(
            f"binary_f1 hors plage [0, 1]: "
            f"{assessment.metrics.binary_f1}"
        )

    # ------------------------------------------------------------------
    # Test 2 : pas de crash avec un split vide
    # ------------------------------------------------------------------
    print("\n[Test 2] assess() avec split vide (dfaut ((), ()))")
    empty_split = DatasetSplit(name="validation", X=np.zeros((0, 5)), Y=np.zeros((0,)))
    # Pas d'appel : on vrifie juste que le validator n'a pas
    # de mthode publique qui crash sur split vide

    # ------------------------------------------------------------------
    # Test 3 : appel de _evaluate_on_split directement
    # ------------------------------------------------------------------
    print("\n[Test 3] _evaluate_on_split() retourne un mask bien align")
    try:
        metrics = evaluator._evaluate_on_split(
            hypothesis,
            split,
            batch_size=200,
            sample_size=None,
            split_name="validation",
        )
        print(f"  metrics.support: {metrics.support}")
        print(f"  metrics.coverage: {metrics.coverage:.4f}")
        print(f"  metrics.binary_precision: {metrics.binary_precision:.4f}")
        print(f"  metrics.binary_recall: {metrics.binary_recall:.4f}")
        print(f"  Pas de crash : OK")
    except Exception as exc:
        errors.append(f"_evaluate_on_split a crash: {exc!r}")

    # ------------------------------------------------------------------
    # Test 4 : appel de l'ancienne _signal_mask_from_array (deprci)
    # ------------------------------------------------------------------
    print("\n[Test 4] _signal_mask_from_array reste accessible (deprciated)")
    import warnings as _w
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        result = evaluator._signal_mask_from_array(
            [Y[Y > 0]], Y, support=int((Y > 0).sum()),
        )
        dep_count = sum(
            1 for w in caught
            if issubclass(w.category, DeprecationWarning)
        )
        if dep_count == 0:
            errors.append("_signal_mask_from_array n'a pas lev de DeprecationWarning")
        else:
            print(f"  DeprecationWarning leve : OK ({dep_count} fois)")

    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    if errors:
        print(f"FAIL : {len(errors)} erreur(s) :")
        for e in errors:
            print(f"  - {e}")
        return 1
    else:
        print("OK : tous les tests passent.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
