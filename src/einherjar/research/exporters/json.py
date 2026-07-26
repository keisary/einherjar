# exporters/json.py
"""
==========================================================
JSON Exporter
==========================================================

Export JSON pour corpus, rejets et rapports.

Le module se contente de sérialiser des objets déjà
normalisés.
"""

from __future__ import annotations

import json as _json
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from .corpus import Corpus
from .rejected import RejectedCorpus
from .reports import ReportBundle

__all__ = [
    "JSONExporterSettings",
    "JSONExporter",
]


def _default(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _ensure_path(path: str | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)


class JSONExporterSettings:
    """
    Paramètres d'export JSON.
    """

    def __init__(
        self,
        *,
        indent: int = 2,
        sort_keys: bool = False,
        ensure_ascii: bool = False,
        separators: tuple[str, str] | None = None,
    ) -> None:
        self.indent = indent
        self.sort_keys = sort_keys
        self.ensure_ascii = ensure_ascii
        self.separators = separators

    def to_dict(self) -> dict[str, Any]:
        return {
            "indent": self.indent,
            "sort_keys": self.sort_keys,
            "ensure_ascii": self.ensure_ascii,
            "separators": self.separators,
        }


class JSONExporter:
    """
    Sérialise les objets du pipeline en JSON.
    """

    def __init__(self, settings: JSONExporterSettings | None = None) -> None:
        self._settings = settings or JSONExporterSettings()

    @property
    def settings(self) -> JSONExporterSettings:
        return self._settings

    def dumps(self, obj: Any) -> str:
        return _json.dumps(
            obj,
            default=_default,
            indent=self._settings.indent,
            sort_keys=self._settings.sort_keys,
            ensure_ascii=self._settings.ensure_ascii,
            separators=self._settings.separators,
        )

    def dump(self, obj: Any, path: str | Path) -> Path:
        path = _ensure_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.dumps(obj), encoding="utf-8")
        return path

    def export_corpus(self, corpus: Corpus, path: str | Path) -> Path:
        return self.dump(corpus.to_dict(), path)

    def export_rejected(self, rejected: RejectedCorpus, path: str | Path) -> Path:
        return self.dump(rejected.to_dict(), path)

    def export_reports(self, reports: ReportBundle, path: str | Path) -> Path:
        return self.dump(reports.to_dict(), path)

    def export(self, obj: Any, path: str | Path) -> Path:
        return self.dump(obj, path)

    def loads(self, text: str) -> Any:
        return _json.loads(text)

    def load(self, path: str | Path) -> Any:
        path = _ensure_path(path)
        return self.loads(path.read_text(encoding="utf-8"))

    def __repr__(self) -> str:
        return f"JSONExporter(indent={self._settings.indent})"