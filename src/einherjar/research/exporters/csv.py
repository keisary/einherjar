# exporters/csv.py
"""
==========================================================
CSV Exporter
==========================================================

Export tabulaire des corpus, rejets et rapports.

Le module aplatie les structures pour les rendre lisibles
dans un tableur ou un outil d'analyse.
"""

from __future__ import annotations

import csv as _csv
import json as _json
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from .corpus import Corpus
from .rejected import RejectedCorpus
from .reports import ReportBundle

__all__ = [
    "CSVExporterSettings",
    "CSVExporter",
]


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
        return {prefix: _json.dumps(list(value), ensure_ascii=False)}

    return {prefix: value}


class CSVExporterSettings:
    """
    Paramètres d'export CSV.
    """

    def __init__(
        self,
        *,
        delimiter: str = ",",
        encoding: str = "utf-8",
        newline: str = "",
        quotechar: str = '"',
    ) -> None:
        self.delimiter = delimiter
        self.encoding = encoding
        self.newline = newline
        self.quotechar = quotechar

    def to_dict(self) -> dict[str, Any]:
        return {
            "delimiter": self.delimiter,
            "encoding": self.encoding,
            "newline": self.newline,
            "quotechar": self.quotechar,
        }


class CSVExporter:
    """
    Sérialise les objets du pipeline en CSV.
    """

    def __init__(self, settings: CSVExporterSettings | None = None) -> None:
        self._settings = settings or CSVExporterSettings()

    @property
    def settings(self) -> CSVExporterSettings:
        return self._settings

    def write_rows(self, rows: list[dict[str, Any]], path: str | Path) -> Path:
        path = _ensure_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)

        with path.open("w", newline=self._settings.newline, encoding=self._settings.encoding) as fh:
            writer = _csv.DictWriter(
                fh,
                fieldnames=fieldnames,
                delimiter=self._settings.delimiter,
                quotechar=self._settings.quotechar,
            )
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in fieldnames})

        return path

    def export_corpus(self, corpus: Corpus, path: str | Path) -> Path:
        return self.write_rows(corpus.to_records(), path)

    def export_rejected(self, rejected: RejectedCorpus, path: str | Path) -> Path:
        return self.write_rows(rejected.to_records(), path)

    def export_reports(self, reports: ReportBundle, path: str | Path) -> Path:
        rows = []
        for record in reports.to_records():
            rows.append(_flatten(record))
        return self.write_rows(rows, path)

    def export(self, obj: Any, path: str | Path) -> Path:
        if hasattr(obj, "to_records"):
            return self.write_rows(list(obj.to_records()), path)
        if isinstance(obj, list) and obj and isinstance(obj[0], Mapping):
            return self.write_rows([_flatten(item) for item in obj], path)
        return self.write_rows([_flatten(_default(obj))], path)

    def __repr__(self) -> str:
        return f"CSVExporter(delimiter={self._settings.delimiter!r})"