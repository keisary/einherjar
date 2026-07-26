"""
Dataset configuration.
"""

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class DatasetConfig:
    """MIDAS dataset configuration."""

    dataset_version: str = "MIDAS_V2"

    cache_dataset: bool = True

    validate_contract: bool = True

    normalize_features: bool = False

    shuffle: bool = False
