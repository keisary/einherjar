"""
refinement/ — Step 4 : Raffinement beam local.

Pour les Einhers viables mais sous-optimaux, beam search local variant
un paramètre à la fois (seuil, opérateur, constante interne).

Règle dure : ne recalibre JAMAIS SL/TP (figés depuis train).

Conforme à ALGORITHME_RESEARCH.md § 10.2 étape 4.
"""
