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
import logging

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from config.dataset import DatasetConfig

from .contract import DatasetContract

logger = logging.getLogger("einherjar.dataset")


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


@dataclass(frozen=True, slots=True)
class MidasArrays:
    """
    Arrays bruts MIDAS pour un actif donné.
    """

    X: NDArray
    Y_ret: NDArray
    ts: NDArray


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

        self._splits: dict[str, DatasetSplit] = {}
        self._midas: MidasArrays | None = None
        self._contract: DatasetContract | None = None

        self._load()
    
    @classmethod
    def from_config(cls, config: Any | None) -> "DatasetLoader":
        if isinstance(config, DatasetConfig):
            return cls(config)
        if hasattr(config, "dataset") and isinstance(config.dataset, DatasetConfig):
            return cls(config.dataset)
        raise TypeError("config must be a DatasetConfig or have a .dataset attribute")

    # ==================================================
    # PRIVATE
    # ==================================================

    def _load(self) -> None:
        # Mode MIDAS (asset / timeframe)
        if self._config.midas_root and self._config.asset and self._config.asset_class and self._config.timeframe:
            self._load_midas()
            return

        # Mode "splits" classique
        if self._config.metadata_path:
            self._contract = self._load_contract()
            self._load_splits()
            return

        logger.warning(
            "DatasetLoader : aucune configuration de chargement valide fournie. "
            "Fournissez soit (midas_root + asset + asset_class + timeframe), "
            "soit (metadata_path + x_train_path + ...)."
        )

    def _load_midas(self) -> None:
        """Charge les arrays MIDAS pour un actif donné."""
        midas_root = Path(self._config.midas_root)
        asset = self._config.asset
        asset_class = self._config.asset_class
        timeframe = self._config.timeframe

        base = midas_root / asset_class / timeframe
        paths = {
            "X": base / f"{asset}_X.npy",
            "Y_ret": base / f"{asset}_Y_ret.npy",
            "ts": base / f"{asset}_ts.npy",
        }

        arrays = {}
        for key, p in paths.items():
            if not p.exists():
                raise FileNotFoundError(f"Fichier MIDAS manquant : {p}")
            arrays[key] = np.load(p, mmap_mode="r", allow_pickle=False)

        self._midas = MidasArrays(
            X=arrays["X"],
            Y_ret=arrays["Y_ret"],
            ts=arrays["ts"],
        )

        meta_path = base / "metadata.json"
        if meta_path.exists():
            with meta_path.open("r", encoding="utf-8") as f:
                metadata = json.load(f)
            self._contract = DatasetContract.from_dict(metadata)

    def _load_splits(self) -> None:

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

        with Path(self._config.metadata_path).open(
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
    def contract(self) -> DatasetContract | None:
        return self._contract

    @property
    def splits(self) -> tuple[str, ...]:
        return tuple(self._splits.keys())

    @property
    def midas(self) -> MidasArrays | None:
        return self._midas

    @property
    def is_midas_mode(self) -> bool:
        return self._midas is not None

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
        mode = "midas" if self.is_midas_mode else "splits"

        return (
            "DatasetLoader("
            f"mode={mode}, "
            f"splits=[{splits}]"
            ")"
        )
