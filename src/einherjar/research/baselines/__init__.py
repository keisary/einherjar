"""Einhers baselines : preuve anti-hasard (Option B du plan).

Référence A6 du plan (random search) : établir la distribution de
performance d'Einhers ALEATOIRES sur le même pipeline d'évaluation que les
candidats réels, pour prouver plus tard que le STGP bat le hasard.

Composants :
- vector_eval : évaluation vectorisée des AST (le evaluate_ast_on_array
  existant est O(N*F) en Python pur, trop lent pour des centaines de runs).
- random_gen  : génération d'Einhers aléatoires (conditions AND 1..3, seuils
  des quantiles de la fenêtre TRAIN uniquement — pas de lookahead).
- runner      : CLI produisant le rapport baselines JSON + audit MD.
"""
