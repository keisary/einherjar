"""
Dataset configuration.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True, frozen=True)
class DatasetConfig:
    """MIDAS dataset configuration."""

    dataset_version: str = "MIDAS_V2"

    cache_dataset: bool = True

    validate_contract: bool = True

    normalize_features: bool = False

    shuffle: bool = False

    # --- Champs pour le mode "splits" (train/validation/test) ---
    metadata_path: str | Path = ""
    x_train_path: str | Path = ""
    y_train_path: str | Path = ""
    x_validation_path: str | Path = ""
    y_validation_path: str | Path = ""
    x_test_path: str | Path = ""
    y_test_path: str | Path = ""

    # --- Champs pour le mode MIDAS (asset/timeframe) ---
    midas_root: str | Path = ""
    asset: str = ""
    asset_class: str = ""
    timeframe: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_version": self.dataset_version,
            "cache_dataset": self.cache_dataset,
            "validate_contract": self.validate_contract,
            "normalize_features": self.normalize_features,
            "shuffle": self.shuffle,
            "metadata_path": str(self.metadata_path) if self.metadata_path else "",
            "x_train_path": str(self.x_train_path) if self.x_train_path else "",
            "y_train_path": str(self.y_train_path) if self.y_train_path else "",
            "x_validation_path": str(self.x_validation_path) if self.x_validation_path else "",
            "y_validation_path": str(self.y_validation_path) if self.y_validation_path else "",
            "x_test_path": str(self.x_test_path) if self.x_test_path else "",
            "y_test_path": str(self.y_test_path) if self.y_test_path else "",
            "midas_root": str(self.midas_root) if self.midas_root else "",
            "asset": self.asset,
            "asset_class": self.asset_class,
            "timeframe": self.timeframe,
        }
