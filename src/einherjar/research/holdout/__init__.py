"""
holdout/ — Step 6 : Évaluation finale unique sur le holdout (SACRÉ).

Le holdout n'est consulté qu'UNE SEULE FOIS dans la vie d'un Einher,
à la toute fin, après que tous les hyperparamètres du générateur, les
poids de fitness, les seuils d'admission et les splits train/val sont
gelés.

Règle dure : aucun recalibrage n'est possible après le holdout. Le
résultat est publié tel quel, accompagné de son IC bootstrap.

Conforme à ONTOLOGY.md S-3.8 et ALGORITHME_RESEARCH.md § 10.2 étape 6.
"""
