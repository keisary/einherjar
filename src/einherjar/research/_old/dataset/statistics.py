"""
==========================================================
Dataset Statistics
==========================================================

Calcule des statistiques descriptives sur les datasets.

Toutes les opérations sont effectuées par blocs afin de
préserver la mémoire même avec des datasets de plusieurs
dizaines ou centaines de gigaoctets.
"""

from __future__ import annotations

from math import inf

import numpy as np

from .loader import DatasetLoader
from .loader import DatasetSplit


class DatasetStatistics:
    """
    Statistiques descriptives du dataset.
    """

    DEFAULT_BATCH_SIZE = 100_000

    # ==================================================
    # INITIALIZATION
    # ==================================================

    def __init__(
        self,
        loader: DatasetLoader,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:

        self._loader = loader
        self._batch_size = batch_size

    # ==================================================
    # PUBLIC
    # ==================================================

    def describe(
        self,
        split_name: str,
    ) -> dict:

        return self._describe(
            self._loader[split_name],
        )

    def describe_all(self) -> dict:

        return {
            split.name: self._describe(split)
            for split in self._loader
        }

    # ==================================================
    # PRIVATE
    # ==================================================

    def _describe(
        self,
        split: DatasetSplit,
    ) -> dict:

        x = split.X

        rows = x.shape[0]

        minimum = inf
        maximum = -inf

        total = 0.0
        total_sq = 0.0

        nan_count = 0

        count = 0

        for start in range(
            0,
            rows,
            self._batch_size,
        ):

            end = min(
                start + self._batch_size,
                rows,
            )

            batch = x[start:end]

            valid = batch[~np.isnan(batch)]

            if valid.size == 0:
                continue

            minimum = min(
                minimum,
                float(valid.min()),
            )

            maximum = max(
                maximum,
                float(valid.max()),
            )

            total += float(valid.sum())

            total_sq += float(
                np.square(valid).sum()
            )

            nan_count += int(
                np.isnan(batch).sum()
            )

            count += valid.size

        mean = total / count

        variance = (
            total_sq / count
        ) - (mean * mean)

        std = variance**0.5

        return {
            "shape": x.shape,
            "dtype": str(x.dtype),
            "feature_count": split.feature_count,
            "sample_count": split.sample_count,
            "label_count": split.label_count,
            "min": minimum,
            "max": maximum,
            "mean": mean,
            "std": std,
            "nan": nan_count,
        }