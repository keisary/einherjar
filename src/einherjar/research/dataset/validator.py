"""
==========================================================
Dataset Validator
==========================================================

Valide l'intégrité structurelle du dataset.

Le Validator ne charge jamais complètement les données
en mémoire et ne modifie jamais le contenu des fichiers.
"""

from __future__ import annotations

import numpy as np

from .loader import DatasetLoader
from .loader import DatasetSplit


class DatasetValidator:
    """
    Valide la cohérence d'un DatasetLoader.
    """

    # ==================================================
    # PUBLIC
    # ==================================================

    def validate(
        self,
        loader: DatasetLoader,
    ) -> None:

        contract = loader.contract

        for split in loader:

            self._validate_split(
                split,
                contract.feature_count,
            )

    # ==================================================
    # PRIVATE
    # ==================================================

    @staticmethod
    def _validate_split(
        split: DatasetSplit,
        expected_feature_count: int,
    ) -> None:

        DatasetValidator._validate_dimensions(split)

        DatasetValidator._validate_dtype(split)

        DatasetValidator._validate_feature_count(
            split,
            expected_feature_count,
        )

        DatasetValidator._validate_sample_count(split)

    @staticmethod
    def _validate_dimensions(
        split: DatasetSplit,
    ) -> None:

        if split.X.ndim != 2:
            raise ValueError(
                f"{split.name}: X must be 2-dimensional."
            )

        if split.Y.ndim not in (1, 2):
            raise ValueError(
                f"{split.name}: Y must be 1D or 2D."
            )

    @staticmethod
    def _validate_dtype(
        split: DatasetSplit,
    ) -> None:

        if not np.issubdtype(
            split.X.dtype,
            np.number,
        ):
            raise TypeError(
                f"{split.name}: X must contain numeric values."
            )

        if not np.issubdtype(
            split.Y.dtype,
            np.number,
        ):
            raise TypeError(
                f"{split.name}: Y must contain numeric values."
            )

    @staticmethod
    def _validate_feature_count(
        split: DatasetSplit,
        expected: int,
    ) -> None:

        if split.feature_count != expected:
            raise ValueError(
                f"{split.name}: expected {expected} features "
                f"but got {split.feature_count}."
            )

    @staticmethod
    def _validate_sample_count(
        split: DatasetSplit,
    ) -> None:

        if split.X.shape[0] != split.Y.shape[0]:
            raise ValueError(
                f"{split.name}: X and Y have different "
                "numbers of samples."
            )

        if split.sample_count == 0:
            raise ValueError(
                f"{split.name}: dataset is empty."
            )