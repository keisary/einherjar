"""search_engine — Moteur de recherche d'Einhers (STGP + MAP-Elites + admission).

Implémentation du plan docs/PLAN_TECHNIQUES_RECHERCHE_FEATURES.md, étape E :
0 données → 1 recherche (random + STGP + MAP-Elites) → 2 affinage BO (option)
→ 3 validation (CI bootstrap + DSR + FDR + indépendance) → 4 corpus.

Ce package est l'implémentation « couche 1 » : tout ce qui concerne la
recherche de combinaisons de features rentables vit ICI (décision Jovanny,
2026-08-20). Le pipeline d'évaluation (backtester, split purgé/embargoed,
types Einher) est réutilisé depuis xgb_einhers.
"""
