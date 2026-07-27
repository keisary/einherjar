"""
==========================================================
Dataset Contract
==========================================================

Décrit le contrat structurel d'un dataset.

Il ne charge aucun fichier et ne contient aucune logique
métier.

Le Validator utilise ce contrat pour vérifier que les
données chargées sont conformes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class DatasetContract:
    """
    Contrat structurel d'un dataset.
    """

    feature_count: int

    horizons: tuple[str, ...]

    feature_names: tuple[str, ...]

    label_names: tuple[str, ...]

    dtype: str

    version: str = ""

    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:

        if self.feature_count <= 0:
            raise ValueError(
                "feature_count must be > 0."
            )

        if len(self.feature_names) != self.feature_count:
            raise ValueError(
                "feature_names size mismatch."
            )

        object.__setattr__(
            self,
            "metadata",
            {} if self.metadata is None else dict(self.metadata),
        )

    @property
    def label_count(self) -> int:
        return len(self.label_names)

    def to_dict(self) -> dict[str, Any]:

        return {
            "feature_count": self.feature_count,
            "feature_names": list(self.feature_names),
            "label_names": list(self.label_names),
            "horizons": list(self.horizons),
            "dtype": self.dtype,
            "version": self.version,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
    ) -> "DatasetContract":
        # Compatibilité MIDAS : features_count vs feature_count
        feature_count = data.get("feature_count") or data.get("features_count", 0)
        return cls(
            feature_count=feature_count,
            feature_names=tuple(data.get("feature_names", ())),
            label_names=tuple(data.get("label_names", ())),
            horizons=tuple(data.get("horizons", ())),
            dtype=data.get("dtype", "float64"),
            version=data.get("version", ""),
            metadata=data.get("metadata", {}),
        )