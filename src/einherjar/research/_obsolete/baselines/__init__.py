"""baselines/ — Step 1 : Baselines honnêtes.

Tous les algorithmes en compétition sont dans `algorithms.py` :
  - human_rules        : règles manuelles 1-3 conditions
  - shallow_enum       : énumération peu profonde (1-2 conditions)
  - random_constrained : random search sous contraintes (typage/profondeur)

Le runner (`runner.py`) lance les 3, agrège les MesuresBrutes sur le val,
et produit une distribution de Sharpe/PnL pour calibrer les attentes.

Conforme à ALGORITHME_RESEARCH.md § 10.2 étape 1.
"""
