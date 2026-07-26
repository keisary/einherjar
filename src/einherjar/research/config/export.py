"""
Export configuration.
"""

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class ExportConfig:

    export_json: bool = True

    export_parquet: bool = True

    export_csv: bool = False

    compress: bool = True