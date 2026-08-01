# exporters/parquet.py
"""
==========================================================
Parquet Exporter
==========================================================

Export Parquet des corpus, rejets et rapports.

Ce module est volontairement optionnel au niveau des dépendances :
- pandas/pyarrow peuvent être présents,
- sinon l'export lève une erreur claire.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from .corpus import Corpus
from .rejected import RejectedCorpus
from .reports import ReportBundle

__all__ = [
    "ParquetExporterSettings",
    "ParquetExporter",
]


try:  # optional dependency
    import pandas as _pd  # type: ignore
except Exception:  # pragma: no cover
    _pd = None


def _ensure_path(path: str | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)


def _default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    return value


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    value = _default(value)

    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            new_key = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flatten(item, new_key))
        return result

    if isinstance(value, (list, tuple, set)):
        return {prefix: list(value)}

    return {prefix: value}


class ParquetExporterSettings:
    """
    Paramètres d'export Parquet.
    """

    def __init__(self, *, compression: str = "snappy") -> None:
        self.compression = compression

    def to_dict(self) -> dict[str, Any]:
        return {"compression": self.compression}


class ParquetExporter:
    """
    Sérialise les objets du pipeline en Parquet.
    """

    def __init__(self, settings: ParquetExporterSettings | None = None) -> None:
        self._settings = settings or ParquetExporterSettings()

    @property
    def settings(self) -> ParquetExporterSettings:
        return self._settings

    def _ensure_dataframe(self, rows: list[dict[str, Any]]):
        if _pd is None:
            raise ImportError(
                "pandas is required for Parquet export. "
                "Install pandas and a parquet engine such as pyarrow."
            )
        return _pd.DataFrame(rows)

    def write_rows(self, rows: list[dict[str, Any]], path: str | Path) -> Path:
        path = _ensure_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        df = self._ensure_dataframe(rows)
        df.to_parquet(path, index=False, compression=self._settings.compression)
        return path

    def export_corpus(self, corpus: Corpus, path: str | Path) -> Path:
        # On aplatit les rows pour gérer les champs dict/list
        # imbriqués. Sinon pyarrow lève "Cannot write struct type
        # 'metadata' with no child field" quand un row a un dict vide
        # ou un sous-struct non-uniforme.
        rows = [_flatten(record) for record in corpus.to_records()]
        return self.write_rows(rows, path)

    def export_rejected(self, rejected: RejectedCorpus, path: str | Path) -> Path:
        rows = [_flatten(record) for record in rejected.to_records()]
        return self.write_rows(rows, path)

    def export_reports(self, reports: ReportBundle, path: str | Path) -> Path:
        rows = [_flatten(record) for record in reports.to_records()]
        return self.write_rows(rows, path)

    def export(self, obj: Any, path: str | Path) -> Path:
        if hasattr(obj, "to_records"):
            return self.write_rows(list(obj.to_records()), path)
        if isinstance(obj, list):
            rows = [_flatten(item) for item in obj]
            return self.write_rows(rows, path)
        return self.write_rows([_flatten(_default(obj))], path)

    def __repr__(self) -> str:
        return f"ParquetExporter(compression={self._settings.compression!r})"