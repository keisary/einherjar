"""
Export configuration.
"""

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True, frozen=True)
class ExportConfig:

    export_json: bool = True

    export_parquet: bool = True

    export_csv: bool = False

    compress: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "export_json": self.export_json,
            "export_parquet": self.export_parquet,
            "export_csv": self.export_csv,
            "compress": self.compress,
        }
