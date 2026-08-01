"""
==========================================================
Dataset Validator
==========================================================

Valide l'intégrité structurelle du dataset.

Le Validator ne charge jamais complètement les données
en mémoire et ne modifie jamais le contenu des fichiers.

Deux modes de validation :

- validate_splits() : pour le mode splits (train/val/test).
- validate_midas()  : pour le mode MIDAS (X / Y_ret / ts).

Tout échec de validation lève une exception explicite ;
aucun fallback silencieux n'est appliqué.
"""

from __future__ import annotations

import numpy as np

from core.exceptions import DatasetContractError
from core.exceptions import DatasetValidationError

from .loader import DatasetLoader
from .loader import DatasetSplit
from .loader import MidasArrays


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
        """
        Valide le dataset en fonction de son mode.

        Lève DatasetContractError ou DatasetValidationError
        au premier écart détecté.
        """

        # Le contrat doit toujours être présent, quel que soit le mode.
        contract = loader.contract
        contract.verify()

        if loader.is_midas_mode:
            self.validate_midas(loader)
            contract.verify_for_midas()
            return

        if loader.is_splits_mode:
            self.validate_splits(loader)
            contract.verify_for_splits()
            return

        raise DatasetValidationError(
            "DatasetLoader : ni en mode splits ni en mode MIDAS."
        )

    def validate_splits(
        self,
        loader: DatasetLoader,
    ) -> None:

        contract = loader.contract

        for split in loader:

            self._validate_split(
                split,
                contract.feature_count,
            )

    def validate_midas(
        self,
        loader: DatasetLoader,
    ) -> None:

        if not loader.is_midas_mode:
            raise DatasetValidationError(
                "DatasetLoader is not in MIDAS mode."
            )

        contract = loader.contract
        midas = loader.midas

        self._validate_midas_arrays(midas, contract.feature_count)

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
    def _validate_midas_arrays(
        midas: MidasArrays,
        expected_feature_count: int,
    ) -> None:

        # --- X ---------------------------------------------------------
        if midas.X.ndim != 2:
            raise DatasetValidationError(
                f"MIDAS X must be 2-dimensional "
                f"(got ndim={midas.X.ndim})."
            )

        if not np.issubdtype(midas.X.dtype, np.number):
            raise DatasetValidationError(
                "MIDAS X must contain numeric values."
            )

        if midas.feature_count != expected_feature_count:
            raise DatasetValidationError(
                f"MIDAS X feature count mismatch: "
                f"expected {expected_feature_count}, "
                f"got {midas.feature_count}."
            )

        # --- Y_ret -----------------------------------------------------
        if midas.Y_ret.ndim not in (1, 2):
            raise DatasetValidationError(
                f"MIDAS Y_ret must be 1D or 2D "
                f"(got ndim={midas.Y_ret.ndim})."
            )

        if not np.issubdtype(midas.Y_ret.dtype, np.number):
            raise DatasetValidationError(
                "MIDAS Y_ret must contain numeric values."
            )

        # --- ts --------------------------------------------------------
        if midas.ts.ndim != 1:
            raise DatasetValidationError(
                f"MIDAS ts must be 1-dimensional "
                f"(got ndim={midas.ts.ndim})."
            )

        if midas.ts.shape[0] != midas.X.shape[0]:
            raise DatasetValidationError(
                f"MIDAS ts length mismatch: "
                f"ts.shape[0]={midas.ts.shape[0]} "
                f"vs X.shape[0]={midas.X.shape[0]}."
            )

        if midas.Y_ret.shape[0] != midas.X.shape[0]:
            raise DatasetValidationError(
                f"MIDAS Y_ret length mismatch: "
                f"Y_ret.shape[0]={midas.Y_ret.shape[0]} "
                f"vs X.shape[0]={midas.X.shape[0]}."
            )

        # --- Échantillons ---------------------------------------------
        if midas.sample_count == 0:
            raise DatasetValidationError(
                "MIDAS dataset is empty."
            )

    @staticmethod
    def _validate_dimensions(
        split: DatasetSplit,
    ) -> None:

        if split.X.ndim != 2:
            raise DatasetValidationError(
                f"{split.name}: X must be 2-dimensional."
            )

        if split.Y.ndim not in (1, 2):
            raise DatasetValidationError(
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
            raise DatasetValidationError(
                f"{split.name}: X must contain numeric values."
            )

        if not np.issubdtype(
            split.Y.dtype,
            np.number,
        ):
            raise DatasetValidationError(
                f"{split.name}: Y must contain numeric values."
            )

    @staticmethod
    def _validate_feature_count(
        split: DatasetSplit,
        expected: int,
    ) -> None:

        if split.feature_count != expected:
            raise DatasetValidationError(
                f"{split.name}: expected {expected} features "
                f"but got {split.feature_count}."
            )

    @staticmethod
    def _validate_sample_count(
        split: DatasetSplit,
    ) -> None:

        if split.X.shape[0] != split.Y.shape[0]:
            raise DatasetValidationError(
                f"{split.name}: X and Y have different "
                "numbers of samples."
            )

        if split.sample_count == 0:
            raise DatasetValidationError(
                f"{split.name}: dataset is empty."
            )
