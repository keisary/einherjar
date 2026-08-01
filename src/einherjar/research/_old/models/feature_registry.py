"""
==========================================================
Feature Registry
==========================================================

Le FeatureRegistry est l'unique source de vérité concernant
les Features disponibles dans le moteur.

Il charge le metadata.json une seule fois, construit les
objets Feature puis fournit des accès rapides aux autres
composants.

Le registre ne contient aucune logique métier.
Il ne fait qu'indexer les Features.

Le System est responsable de créer UNE SEULE instance du
registre puis de la partager avec tous les modules.
"""

from __future__ import annotations

import json
import logging

from pathlib import Path
from typing import Any

from .enums import EconomicFamily
from .enums import FeatureType
from .enums import FeatureValueType
from .feature import Feature

logger = logging.getLogger("einherjar.registry")


def _load_taxonomy(taxonomy_path: str | Path | None = None) -> dict[str, dict[str, str]]:
    """Charge la taxonomie manuelle des features depuis le JSON."""
    if taxonomy_path is None:
        # Chemin relatif au package research
        taxonomy_path = Path(__file__).resolve().parent.parent / "config" / "feature_taxonomy.json"

    taxonomy_path = Path(taxonomy_path)
    if not taxonomy_path.exists():
        logger.warning("Fichier de taxonomie introuvable : %s", taxonomy_path)
        return {}

    try:
        with taxonomy_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("features", {})
    except Exception as exc:
        logger.warning("Impossible de charger la taxonomie (%s) : %s", taxonomy_path, exc)
        return {}


class FeatureRegistry:
    """
    Index immutable des Feature du dataset.
    """

    # ==================================================
    # INITIALISATION
    # ==================================================

    def __init__(
        self,
        metadata_path: str | Path,
        taxonomy_path: str | Path | None = None,
    ) -> None:

        self._metadata_path = Path(metadata_path)

        if not self._metadata_path.exists():
            raise FileNotFoundError(
                self._metadata_path
            )

        self._taxonomy = _load_taxonomy(taxonomy_path)

        self._features: list[Feature] = []

        self._by_index: dict[int, Feature] = {}

        self._by_name: dict[str, Feature] = {}

        self._horizons: tuple[str, ...] = ()

        self._load()

    # ==================================================
    # LOADING
    # ==================================================

    def _load(self) -> None:

        with self._metadata_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            metadata = json.load(file)

        self._horizons = tuple(
            metadata.get("horizons", ())
        )

        feature_names = metadata.get(
            "feature_names",
            (),
        )

        for column_index, name in enumerate(
            feature_names
        ):
            taxonomy = self._taxonomy.get(name, {})

            feature = Feature(
                column_index=column_index,
                name=name,
                feature_type=self._resolve_feature_type(taxonomy.get("feature_type")),
                economic_family=self._resolve_economic_family(taxonomy.get("economic_family")),
                value_type=self._resolve_value_type(taxonomy.get("value_type")),
            )

            self._features.append(feature)
            self._by_index[column_index] = feature
            self._by_name[name] = feature

        self._features = tuple(self._features)

        n_taxonomized = sum(1 for f in self._features if f.economic_family != EconomicFamily.OTHER)
        logger.info(
            "FeatureRegistry : %d features, %d avec taxonomie explicite",
            len(self._features),
            n_taxonomized,
        )

        self._validate()

    # ==================================================
    # TAXONOMY RESOLUTION
    # ==================================================

    @staticmethod
    def _resolve_feature_type(value: str | None) -> FeatureType:
        if not value:
            return FeatureType.ATOMIC
        try:
            return FeatureType(value)
        except ValueError:
            return FeatureType.ATOMIC

    @staticmethod
    def _resolve_economic_family(value: str | None) -> EconomicFamily:
        if not value:
            return EconomicFamily.OTHER
        try:
            return EconomicFamily(value)
        except ValueError:
            return EconomicFamily.OTHER

    @staticmethod
    def _resolve_value_type(value: str | None) -> FeatureValueType:
        if not value:
            return FeatureValueType.FLOAT
        try:
            return FeatureValueType(value)
        except ValueError:
            return FeatureValueType.FLOAT

    # ==================================================
    # VALIDATION
    # ==================================================

    def _validate(self) -> None:

        if len(self._features) != len(self._by_index):
            raise RuntimeError(
                "Feature registry is inconsistent."
            )

        if len(self._features) != len(self._by_name):
            raise RuntimeError(
                "Feature registry is inconsistent."
            )

        for expected_index, feature in enumerate(
            self._features
        ):

            if feature.column_index != expected_index:
                raise RuntimeError(
                    "Invalid feature ordering."
                )

    # ==================================================
    # ACCESSORS
    # ==================================================

    @property
    def horizons(self) -> tuple[str, ...]:
        return self._horizons

    @property
    def features(self) -> tuple[Feature, ...]:
        return self._features

    @property
    def feature_count(self) -> int:
        return len(self._features)

    def get(
        self,
        key: int | str,
    ) -> Feature:

        if isinstance(key, int):
            return self._by_index[key]

        if isinstance(key, str):
            return self._by_name[key]

        raise TypeError(
            "Feature key must be int or str."
        )

    def by_index(
        self,
        column_index: int,
    ) -> Feature:
        return self._by_index[column_index]

    def by_name(
        self,
        name: str,
    ) -> Feature:
        return self._by_name[name]

    def by_family(
        self,
        family: EconomicFamily | str,
    ) -> tuple[Feature, ...]:
        """Retourne toutes les features d'une famille économique donnée."""
        target = family if isinstance(family, EconomicFamily) else EconomicFamily(family)
        return tuple(f for f in self._features if f.economic_family == target)

    # ==================================================
    # PYTHON PROTOCOL
    # ==================================================

    def __contains__(
        self,
        key: int | str,
    ) -> bool:

        if isinstance(key, int):
            return key in self._by_index

        if isinstance(key, str):
            return key in self._by_name

        return False

    def __getitem__(
        self,
        key: int | str,
    ) -> Feature:
        return self.get(key)

    def __iter__(self):
        return iter(self._features)

    def __len__(self) -> int:
        return len(self._features)

    def __repr__(self) -> str:
        return (
            "FeatureRegistry("
            f"feature_count={len(self)}, "
            f"horizons={self._horizons}"
            ")"
        )
