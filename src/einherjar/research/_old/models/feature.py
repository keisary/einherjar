"""
==========================================================
Feature Model
==========================================================

Décrit une feature disponible dans le dataset MIDAS.

Une Feature est une description IMMUTABLE d'une colonne du
dataset.

Elle ne contient jamais de données de marché.

Le contrat principal est son column_index, qui correspond
exactement à la colonne des matrices X_*.npy.

Le nom n'est qu'un alias lisible destiné au debug et aux
rapports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .enums import EconomicFamily
from .enums import FeatureType
from .enums import FeatureValueType


@dataclass(frozen=True, slots=True)
class Feature:
    """
    Description statique d'une feature du dataset.

    Une Feature représente une colonne du dataset et aucune
    autre information.

    Deux Feature ayant le même column_index représentent
    obligatoirement la même colonne.
    """

    # ==================================================
    # IDENTIFICATION (CONTRACTUELLE)
    # ==================================================

    column_index: int

    name: str

    # ==================================================
    # DESCRIPTION
    # ==================================================

    description: str = ""

    # ==================================================
    # TAXONOMIE
    # ==================================================

    feature_type: FeatureType = FeatureType.ATOMIC

    economic_family: EconomicFamily = EconomicFamily.OTHER

    value_type: FeatureValueType = FeatureValueType.FLOAT

    # ==================================================
    # DATASET
    # ==================================================

    enabled: bool = True

    # ==================================================
    # METADATA
    # ==================================================

    tags: tuple[str, ...] = ()

    metadata: dict[str, Any] | None = None

    # ==================================================
    # VALIDATION
    # ==================================================

    def __post_init__(self) -> None:
        if self.column_index < 0:
            raise ValueError("column_index must be >= 0.")

        if not self.name:
            raise ValueError("Feature name cannot be empty.")

        object.__setattr__(
            self,
            "metadata",
            {} if self.metadata is None else dict(self.metadata),
        )

    # ==================================================
    # PROPERTIES
    # ==================================================

    @property
    def is_atomic(self) -> bool:
        return self.feature_type is FeatureType.ATOMIC

    @property
    def is_pattern(self) -> bool:
        return self.feature_type is FeatureType.PATTERN

    @property
    def is_quantitative(self) -> bool:
        return self.feature_type is FeatureType.QUANTITATIVE

    @property
    def is_composite(self) -> bool:
        return self.feature_type is FeatureType.COMPOSITE

    @property
    def is_boolean(self) -> bool:
        return self.value_type is FeatureValueType.BOOLEAN

    @property
    def is_numeric(self) -> bool:
        return self.value_type in (
            FeatureValueType.FLOAT,
            FeatureValueType.INTEGER,
        )

    # ==================================================
    # EXPORT
    # ==================================================

    def to_dict(self) -> dict[str, Any]:
        return {
            "column_index": self.column_index,
            "name": self.name,
            "description": self.description,
            "feature_type": self.feature_type.value,
            "economic_family": self.economic_family.value,
            "value_type": self.value_type.value,
            "enabled": self.enabled,
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Feature":
        return cls(
            column_index=data["column_index"],
            name=data["name"],
            description=data.get("description", ""),
            feature_type=FeatureType(data["feature_type"]),
            economic_family=EconomicFamily(data["economic_family"]),
            value_type=FeatureValueType(data["value_type"]),
            enabled=data.get("enabled", True),
            tags=tuple(data.get("tags", ())),
            metadata=data.get("metadata", {}),
        )

    # ==================================================
    # OBJECT PROTOCOL
    # ==================================================

    def __hash__(self) -> int:
        """
        L'identité d'une Feature est sa colonne dans le dataset.
        """
        return hash(self.column_index)

    def __repr__(self) -> str:
        return (
            "Feature("
            f"column_index={self.column_index}, "
            f"name='{self.name}', "
            f"type='{self.feature_type.value}', "
            f"family='{self.economic_family.value}'"
            ")"
        )
