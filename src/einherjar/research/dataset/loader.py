"""
==========================================================
Dataset Loader
==========================================================

Point d'entrée unique vers les données du projet.

Le DatasetLoader ne charge jamais entièrement les tableaux
en mémoire. Tous les fichiers .npy sont ouverts en mode
memory-mapped afin de limiter la consommation de RAM.

Le System crée une seule instance du DatasetLoader et la
partage avec l'ensemble du moteur.
"""

from __future__ import annotations

import json

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from config.dataset_config import DatasetConfig

from .contract import DatasetContract


@dataclass(frozen=True, slots=True)
class DatasetSplit:
    """
    Représente un split du dataset.
    """

    name: str

    X: NDArray

    Y: NDArray

    @property
    def sample_count(self) -> int:
        return self.X.shape[0]

    @property
    def feature_count(self) -> int:
        return self.X.shape[1]

    @property
    def label_count(self) -> int:

        if self.Y.ndim == 1:
            return 1

        return self.Y.shape[1]

    @property
    def shape(self) -> tuple[int, int]:
        return self.X.shape


class DatasetLoader:
    """
    Charge et expose le dataset.

    Les tableaux sont ouverts en lecture seule via
    numpy.memmap.
    """

    # ==================================================
    # INITIALIZATION
    # ==================================================

    def __init__(
        self,
        config: DatasetConfig,
    ) -> None:

        self._config = config

        self._contract = self._load_contract()

        self._splits: dict[str, DatasetSplit] = {}

        self._load()

    # ==================================================
    # PRIVATE
    # ==================================================

    def _load(self) -> None:

        self._splits["train"] = DatasetSplit(
            name="train",
            X=self._load_array(
                self._config.x_train_path
            ),
            Y=self._load_array(
                self._config.y_train_path
            ),
        )

        self._splits["validation"] = DatasetSplit(
            name="validation",
            X=self._load_array(
                self._config.x_validation_path
            ),
            Y=self._load_array(
                self._config.y_validation_path
            ),
        )

        self._splits["test"] = DatasetSplit(
            name="test",
            X=self._load_array(
                self._config.x_test_path
            ),
            Y=self._load_array(
                self._config.y_test_path
            ),
        )

    def _load_contract(
        self,
    ) -> DatasetContract:

        with self._config.metadata_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            metadata = json.load(file)

        return DatasetContract.from_dict(
            metadata
        )

    @staticmethod
    def _load_array(
        path: str | Path,
    ) -> NDArray:

        return np.load(
            Path(path),
            mmap_mode="r",
            allow_pickle=False,
        )

    # ==================================================
    # PROPERTIES
    # ==================================================

    @property
    def contract(self) -> DatasetContract:
        return self._contract

    @property
    def splits(self) -> tuple[str, ...]:
        return tuple(self._splits.keys())

    # ==================================================
    # ACCESSORS
    # ==================================================

    def get(
        self,
        name: str,
    ) -> DatasetSplit:

        try:
            return self._splits[name]

        except KeyError as exc:
            raise KeyError(
                f"Unknown dataset split '{name}'."
            ) from exc

    def train(self) -> DatasetSplit:
        return self.get("train")

    def validation(self) -> DatasetSplit:
        return self.get("validation")

    def test(self) -> DatasetSplit:
        return self.get("test")

    # ==================================================
    # PYTHON PROTOCOL
    # ==================================================

    def __contains__(
        self,
        name: str,
    ) -> bool:
        return name in self._splits

    def __getitem__(
        self,
        name: str,
    ) -> DatasetSplit:
        return self.get(name)

    def __iter__(self):
        return iter(self._splits.values())

    def __len__(self) -> int:
        return len(self._splits)

    def __repr__(self) -> str:

        splits = ", ".join(self.splits)

        return (
            "DatasetLoader("
            f"splits=[{splits}]"
            ")"
        )