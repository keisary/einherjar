"""
v2/ — Candidats V2 (PAS actifs en V1, mais testables).

Tous les algorithmes candidats V2 sont dans `algorithms.py` :
  - nsga2_complet    : NSGA-II avec Pareto front si métrique composite stable
  - memetic_complet  : EA + local search complet
  - map_elites       : Quality-Diversity (déclenché si diversité insuffisante)
  - cpcv_full        : Combinatorial Purged CV complet (K=N=10+)
  - llm_stub         : LLM (TROP LOURD pour V1, interface stub dans llm_stub.py)

Conforme à ALGORITHME_RESEARCH.md § 10.5.
"""
