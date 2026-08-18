"""xgb_einhers - Pipeline XGBoost supervisé pour générer des Einhers.

Module From-scratch (plan A-Z : audits/PLAN_XGBOOST_EINHER_AZ.md).

Pipeline :
    X.npy + Y_dir/Y_ret/Y_hor + OHLCV CSV
        -> data_loader (charge, aligne, nettoie)
        -> label_engineer (Y_ret -> target supervisé)
        -> model (XGBoostRegressor par (asset, TF, horizon))
        -> path_extractor (arbres -> chemins)
        -> condition_tree (chemin -> AST)
        -> einher_builder (AST -> Einher)
        -> backtester (NOUVEAU moteur, simule trades)
        -> admission (critères)
        -> einher_io (JSON)

Réponse Q4 user : OHLCV vient des CSV bruts (technical_agent_dataset_brut).
Réponse Q6 : les 5 colonnes OHLCV de X.npy sont EXCLUES pour éviter la fuite.
Réponse Q10 : 1 modèle XGBoost par (asset, TF, horizon).
"""

__version__ = "0.1.0"
__all__ = [
    "types",
    "data_loader",
    "label_engineer",
    "model",
    "path_extractor",
    "condition_tree",
    "einher_builder",
    "backtester",
    "admission",
    "scope_determiner",
    "einher_io",
    "runner",
]
