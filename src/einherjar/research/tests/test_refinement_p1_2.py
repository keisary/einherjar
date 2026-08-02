"""Tests pour P1 #2 : raffinement déprécié (BeamRefiner)."""
import unittest
import warnings

from einherjar.research.refinement.beam import BeamRefiner


class TestBeamRefinerDeprecated(unittest.TestCase):

    def test_construct_emits_deprecation_warning(self):
        # On n'a pas besoin d'un vrai engine/config : juste vérifier le warning.
        from unittest.mock import MagicMock
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            BeamRefiner(MagicMock(), MagicMock(), seed=42)
            deprecation_warnings = [
                warning for warning in w
                if issubclass(warning.category, DeprecationWarning)
            ]
            # Au moins un DeprecationWarning (classe ou sous-classe).
            self.assertGreaterEqual(len(deprecation_warnings), 1)
            # Le message mentionne P1 #2 et la migration recommandee.
            msg = str(deprecation_warnings[0].message)
            self.assertIn("P1 #2", msg)
            self.assertIn("migrez", msg.lower())

    def test_warning_flag_set_after_first_construct(self):
        """Apres construction, le flag _p1_2_warned est True sur la classe."""
        from unittest.mock import MagicMock
        # Reset le flag au cas ou un autre test l'aurait deja mis.
        BeamRefiner._p1_2_warned = False
        BeamRefiner(MagicMock(), MagicMock(), seed=42)
        self.assertTrue(BeamRefiner._p1_2_warned)


if __name__ == "__main__":
    unittest.main()
