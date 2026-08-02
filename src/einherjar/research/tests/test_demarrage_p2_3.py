"""Test de demarrage reel (P2 #3).

Verifie que la config se charge, que le moteur s'instancie,
et que les imports essentiels marchent. C'est le test qu'on lance
apres un 'pip install -e .' pour verifier que tout est OK.
"""
import unittest


class TestDemarrageReel(unittest.TestCase):

    def test_config_chargeable(self):
        """La config par defaut doit se charger sans erreur."""
        from einherjar.research.config.loader import load_config
        config = load_config("src/einherjar/research/config")
        # Invariants P0 : 218 features utilisables, 28 exclues.
        self.assertEqual(len(config.usable_feature_names), 218)
        self.assertEqual(len(config.excluded_feature_names), 28)

    def test_moteur_instanciable(self):
        """Le moteur d'evaluation doit s'instancier sans erreur."""
        from einherjar.research.config.loader import load_config
        from einherjar.research.engine.evaluator import EvaluationEngine
        config = load_config("src/einherjar/research/config")
        engine = EvaluationEngine(config=config, data_version="v1", seed=42)
        self.assertEqual(engine.data_version, "v1")
        self.assertEqual(engine.seed, 42)

    def test_imports_essentiels(self):
        """Les imports des modules cles doivent marcher."""
        # Moteur.
        from einherjar.research.engine.evaluator import EvaluationEngine, CalibratedParams
        from einherjar.research.engine.bootstrap import bootstrap_sharpe, bootstrap_ret_total
        from einherjar.research.engine.simulator import simulate
        # Donnees.
        from einherjar.research.data.npy_real_loader import load_ohlcv_from_npy, load_features_from_npy
        from einherjar.research.data.threshold_calibration import compute_feature_quantiles
        from einherjar.research.data.validation import validate_ohlcv, validate_no_leak
        from einherjar.research.data.versioning import make_data_version
        # Generateurs.
        from einherjar.research.generators.algorithms import (
            RandomSearchGenerator, BeamSearchGenerator, TypedGPGenerator,
            MemeticGenerator, NSGA2Generator, make_all_generators,
        )
        # Comparateur / admission.
        from einherjar.research.generators.comparator import GeneratorComparator
        from einherjar.research.admission.criteria import evaluate_all_criteria
        from einherjar.research.admission.decision import AdmissionDecider
        from einherjar.research.admission.baseline_gate import make_baseline_admission_fn
        # Persistance.
        from einherjar.research.archive.store import append_entry, iter_entries
        from einherjar.research.corpus.store import CorpusStore
        from einherjar.research.holdout.ledger import HoldoutLedger
        from einherjar.research.holdout.evaluator import HoldoutEvaluator
        # CLI.
        from einherjar.research.discovery import build_parser
        # Verifie juste que les imports n'ont pas leve d'exception.
        self.assertTrue(callable(build_parser))


if __name__ == "__main__":
    unittest.main()
