"""
generators/ — Step 2 : Compétition reproductible des générateurs.

Tous les algorithmes candidats sont dans `algorithms.py` :
  - random_search
  - grammatical_evolution (si grammaire BNF écrite)
  - strongly_typed_gp
  - beam_search
  - memetic (EA + local search)
  - nsga2 (si métrique composite stable)

`protocol.py` définit le protocole reproductible (mêmes seeds, splits, budget, métriques, coûts).
`comparator.py` compare et classe les candidats, retourne le gagnant pour Step 3.

Conforme à ALGORITHME_RESEARCH.md § 10.2 étape 2.
"""
