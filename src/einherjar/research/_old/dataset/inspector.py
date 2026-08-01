"""
==========================================================
Dataset Inspector
==========================================================

Outils d'inspection d'un DatasetLoader.

L'Inspector est destiné au développement, au debug et à
l'analyse interactive. Il ne modifie jamais les données et
n'effectue jamais de copie complète des tableaux.

Toutes les opérations sont réalisées directement sur les
numpy.memmap.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .loader import DatasetLoader
from .loader import DatasetSplit


class DatasetInspector:
    """
    API d'inspection d'un DatasetLoader.
    """

    # ==================================================
    # INITIALIZATION
    # ==================================================

    def __init__(
        self,
        loader: DatasetLoader,
    ) -> None:

        self._loader = loader

    # ==================================================
    # SPLITS
    # ==================================================

    def splits(self) -> tuple[str, ...]:
        return self._loader.splits

    def split(
        self,
        name: str,
    ) -> DatasetSplit:
        return self._loader[name]

    # ==================================================
    # GENERAL
    # ==================================================

    def shape(
        self,
        split: str,
    ) -> tuple[int, int]:

        return self._loader[split].shape

    def dtype(
        self,
        split: str,
    ) -> str:

        return str(
            self._loader[split].X.dtype
        )

    def memory_usage(
        self,
        split: str,
    ) -> dict[str, float]:

        dataset = self._loader[split]

        x_bytes = dataset.X.nbytes
        y_bytes = dataset.Y.nbytes

        return {
            "X_bytes": x_bytes,
            "Y_bytes": y_bytes,
            "total_bytes": x_bytes + y_bytes,
            "X_mb": x_bytes / (1024 ** 2),
            "Y_mb": y_bytes / (1024 ** 2),
            "total_mb": (x_bytes + y_bytes) / (1024 ** 2),
            "X_gb": x_bytes / (1024 ** 3),
            "Y_gb": y_bytes / (1024 ** 3),
            "total_gb": (x_bytes + y_bytes) / (1024 ** 3),
        }

    # ==================================================
    # SAMPLES
    # ==================================================

    def head(
        self,
        split: str,
        rows: int = 5,
    ) -> np.ndarray:

        return self._loader[split].X[:rows]

    def tail(
        self,
        split: str,
        rows: int = 5,
    ) -> np.ndarray:

        return self._loader[split].X[-rows:]

    def sample(
        self,
        split: str,
        size: int = 100,
        seed: int | None = None,
    ) -> np.ndarray:

        dataset = self._loader[split]

        rng = np.random.default_rng(seed)

        indices = rng.choice(
            dataset.sample_count,
            size=min(size, dataset.sample_count),
            replace=False,
        )

        return dataset.X[indices]

    # ==================================================
    # FEATURES
    # ==================================================

    def feature(
        self,
        split: str,
        column_index: int,
    ) -> np.ndarray:

        dataset = self._loader[split]

        return dataset.X[:, column_index]

    def feature_preview(
        self,
        split: str,
        column_index: int,
        rows: int = 10,
    ) -> np.ndarray:

        dataset = self._loader[split]

        return dataset.X[
            :rows,
            column_index,
        ]

    # ==================================================
    # LABELS
    # ==================================================

    def labels(
        self,
        split: str,
    ) -> np.ndarray:

        return self._loader[split].Y

    # ==================================================
    # SEARCH
    # ==================================================

    def row(
        self,
        split: str,
        index: int,
    ) -> np.ndarray:

        return self._loader[split].X[index]

    def rows(
        self,
        split: str,
        start: int,
        stop: int,
    ) -> np.ndarray:

        return self._loader[split].X[start:stop]

    # ==================================================
    # SUMMARY
    # ==================================================

    def summary(self) -> dict[str, Any]:

        output = {}

        for split in self._loader:

            output[split.name] = {
                "shape": split.shape,
                "dtype": str(split.X.dtype),
                "samples": split.sample_count,
                "features": split.feature_count,
                "labels": split.label_count,
            }

        return output

    # ==================================================
    # PYTHON PROTOCOL
    # ==================================================

    def __repr__(self) -> str:

        return (
            "DatasetInspector("
            f"splits={self._loader.splits}"
            ")"
        )