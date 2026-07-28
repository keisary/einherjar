"""
==========================================================
Dataset Loader
==========================================================

Point d'entrée unique vers les données du projet.

Le DatasetLoader ne charge jamais entièrement les tableaux
en mémoire. Tous les fichiers .npy sont ouverts en mode
memory-mapped afin de limiter la consommation de RAM.

Le moteur de découverte crée une instance de DatasetLoader
PAR PAIRE asset/timeframe traitée, dans Engine.run_pair().
Une instance n'est jamais partagée entre paires.
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

    Champs :
    - X       : matrice des features (n_samples, n_features)
    - Y_ret   : vecteur des rendements futurs (n_samples,) ou
                matrice multi-horizons (n_samples, n_horizons)
    - ts      : vecteur des timestamps (n_samples,)
    """

    X: NDArray
    Y_ret: NDArray
    ts: NDArray

    @property
    def sample_count(self) -> int:
        return int(self.X.shape[0])

    @property
    def feature_count(self) -> int:
        return int(self.X.shape[1])

    @property
    def horizon_count(self) -> int:
        if self.Y_ret.ndim == 1:
            return 1
        return int(self.Y_ret.shape[1])


class DatasetLoader:
    """
    Charge et expose le dataset.

    Les tableaux sont ouverts en lecture seule via
    numpy.memmap.

    Deux modes de chargement :

    - Mode "splits" : trois DatasetSplit explicites
      (train, validation, test) + un metadata.json.

    - Mode "MIDAS"  : tableaux bruts X / Y_ret / ts pour
      un couple (asset, timeframe).
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

    @classmethod
    def for_pair(
        cls,
        *,
        midas_root: str | Path,
        asset: str,
        asset_class: str,
        timeframe: str,
    ) -> "DatasetLoader":
        """
        Construit un DatasetLoader pour une paire (asset, timeframe).

        Le loader est strictement lié à cette paire.
        """

        from config.dataset import DatasetConfig

        dataset_cfg = DatasetConfig(
            midas_root=str(midas_root),
            asset=asset,
            asset_class=asset_class,
            timeframe=timeframe,
        )
        return cls(dataset_cfg)

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

        raise RuntimeError(
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

        if self._contract is None:
            raise RuntimeError(
                f"metadata.json introuvable pour {asset}/{timeframe} "
                f"(cherché : {base / 'metadata.json'}). "
                f"Le contrat de données est obligatoire."
            )

        # Vérification immédiate : on refuse un dataset sans contrat exploitable
        self._contract.verify_for_midas()

        if self._midas.feature_count != self._contract.feature_count:
            raise RuntimeError(
                f"MIDAS dataset shape / contract mismatch pour "
                f"{asset}/{timeframe} : "
                f"X.shape[1]={self._midas.feature_count} "
                f"vs contract.feature_count={self._contract.feature_count}."
            )

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
    def contract(self) -> DatasetContract:
        """
        Contrat du dataset.

        Lève RuntimeError si le contrat n'a pas pu être
        chargé — il est obligatoire dans les deux modes.
        """

        if self._contract is None:
            raise RuntimeError(
                "DatasetLoader : aucun contrat de données n'a été chargé."
            )
        return self._contract

    @property
    def splits(self) -> tuple[str, ...]:
        return tuple(self._splits.keys())

    @property
    def midas(self) -> MidasArrays:
        """
        Arrays MIDAS du dataset.

        Lève RuntimeError si le loader n'est pas en mode MIDAS.
        """

        if self._midas is None:
            raise RuntimeError(
                "DatasetLoader : pas de tableaux MIDAS (mode splits)."
            )
        return self._midas

    @property
    def is_midas_mode(self) -> bool:
        return self._midas is not None

    @property
    def is_splits_mode(self) -> bool:
        return bool(self._splits) and self._midas is None

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
        mode = "midas" if self.is_midas_mode else ("splits" if self.is_splits_mode else "empty")

        return (
            "DatasetLoader("
            f"mode={mode}, "
            f"splits=[{splits}]"
            ")"
        )
