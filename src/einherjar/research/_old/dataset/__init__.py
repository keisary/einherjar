"""
==========================================================
Dataset Package
==========================================================

Fournit le chargement, la validation et l'inspection des
datasets utilisés par le moteur de découverte.

Deux modes sont supportés :

- Mode "splits" : trois DatasetSplit explicites (train,
  validation, test) avec un fichier metadata.json.

- Mode "MIDAS"   : tableaux bruts X / Y_ret / ts pour un
  couple (asset, timeframe) à partir d'un répertoire de
  compilation MIDAS.

Aucun composant du package n'orchestre le pipeline.
"""

from .contract import DatasetContract
from .inspector import DatasetInspector
from .loader import DatasetLoader
from .loader import DatasetSplit
from .loader import MidasArrays
from .statistics import DatasetStatistics
from .validator import DatasetValidator

__all__ = [
    "DatasetContract",
    "DatasetInspector",
    "DatasetLoader",
    "DatasetSplit",
    "DatasetStatistics",
    "DatasetValidator",
    "MidasArrays",
]
