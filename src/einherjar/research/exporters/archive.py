# exporters/archive.py
"""
==========================================================
Archive Exporter
==========================================================

Regroupe les exports dans une archive finale.

L'archive n'invente pas de contenu :
- elle assemble les exporteurs existants,
- écrit un manifest,
- produit un paquet transportable.

Toute erreur d'export est loguée explicitement ; les échecs
Parquet ne sont plus silencieux.
"""

from __future__ import annotations

import json as _json
import logging
import tempfile
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .corpus import Corpus
from .csv import CSVExporter
from .json import JSONExporter
from .parquet import ParquetExporter
from .rejected import RejectedCorpus
from .reports import ReportBundle

__all__ = [
    "ArchiveSettings",
    "ArchiveManifest",
    "ArchiveExporter",
]

logger = logging.getLogger("einherjar.archive")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _to_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "to_dict") and callable(value.to_dict):
        value = value.to_dict()
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _ensure_path(path: str | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)


@dataclass(frozen=True, slots=True)
class ArchiveSettings:
    """
    Paramètres de l'archive finale.
    """

    include_json: bool = True
    include_csv: bool = True
    include_parquet: bool = True

    compression: int = zipfile.ZIP_DEFLATED
    manifest_name: str = "manifest.json"

    def __post_init__(self) -> None:
        object.__setattr__(self, "include_json", bool(self.include_json))
        object.__setattr__(self, "include_csv", bool(self.include_csv))
        object.__setattr__(self, "include_parquet", bool(self.include_parquet))
        object.__setattr__(self, "manifest_name", str(self.manifest_name).strip() or "manifest.json")

    def to_dict(self) -> dict[str, Any]:
        return {
            "include_json": self.include_json,
            "include_csv": self.include_csv,
            "include_parquet": self.include_parquet,
            "compression": self.compression,
            "manifest_name": self.manifest_name,
        }


@dataclass(frozen=True, slots=True)
class ArchiveManifest:
    """
    Manifest de l'archive.
    """

    created_at: datetime
    files: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    errors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "files", tuple(self.files))
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "errors", tuple(str(e) for e in self.errors))

    def to_dict(self) -> dict[str, Any]:
        return {
            "created_at": self.created_at.isoformat(),
            "files": list(self.files),
            "metadata": dict(self.metadata),
            "errors": list(self.errors),
        }


class ArchiveExporter:
    """
    Produit une archive finale des exports.
    """

    def __init__(
        self,
        settings: ArchiveSettings | None = None,
        *,
        json_exporter: JSONExporter | None = None,
        csv_exporter: CSVExporter | None = None,
        parquet_exporter: ParquetExporter | None = None,
    ) -> None:
        self._settings = settings or ArchiveSettings()
        self._json = json_exporter or JSONExporter()
        self._csv = csv_exporter or CSVExporter()
        self._parquet = parquet_exporter or ParquetExporter()

    @property
    def settings(self) -> ArchiveSettings:
        return self._settings

    def build(
        self,
        *,
        corpus: Corpus | None = None,
        rejected: RejectedCorpus | None = None,
        reports: ReportBundle | None = None,
        path: str | Path,
        stem: str = "export",
        metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        path = _ensure_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        manifest_files: list[str] = []
        errors: list[str] = []

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            seen_names: set[str] = set()

            def _add_to_zip(zf: zipfile.ZipFile, file_path: Path) -> None:
                name = file_path.name
                if name in seen_names:
                    return
                seen_names.add(name)
                zf.write(file_path, arcname=name)
                manifest_files.append(name)

            with zipfile.ZipFile(path, "w", compression=self._settings.compression) as zf:
                if self._settings.include_json:
                    if corpus is not None:
                        try:
                            file_path = tmpdir_path / f"{stem}_corpus.json"
                            self._json.export_corpus(corpus, file_path)
                            _add_to_zip(zf, file_path)
                        except Exception as exc:
                            errors.append(f"json_corpus: {exc}")
                            logger.warning("Export JSON corpus échoué : %s", exc)

                    if rejected is not None:
                        try:
                            file_path = tmpdir_path / f"{stem}_rejected.json"
                            self._json.export_rejected(rejected, file_path)
                            _add_to_zip(zf, file_path)
                        except Exception as exc:
                            errors.append(f"json_rejected: {exc}")
                            logger.warning("Export JSON rejected échoué : %s", exc)

                    if reports is not None:
                        try:
                            file_path = tmpdir_path / f"{stem}_reports.json"
                            # On respecte le mode summary_only par
                            # défaut de JSONExporter (True) pour
                            # éviter le dump de plusieurs GB.
                            self._json.export_reports(
                                reports, file_path, summary_only=True,
                            )
                            _add_to_zip(zf, file_path)
                        except Exception as exc:
                            errors.append(f"json_reports: {exc}")
                            logger.warning("Export JSON reports échoué : %s", exc)

                if self._settings.include_csv:
                    if corpus is not None:
                        try:
                            file_path = tmpdir_path / f"{stem}_corpus.csv"
                            self._csv.export_corpus(corpus, file_path)
                            _add_to_zip(zf, file_path)
                        except Exception as exc:
                            errors.append(f"csv_corpus: {exc}")
                            logger.warning("Export CSV corpus échoué : %s", exc)

                    if rejected is not None:
                        try:
                            file_path = tmpdir_path / f"{stem}_rejected.csv"
                            self._csv.export_rejected(rejected, file_path)
                            _add_to_zip(zf, file_path)
                        except Exception as exc:
                            errors.append(f"csv_rejected: {exc}")
                            logger.warning("Export CSV rejected échoué : %s", exc)

                    if reports is not None:
                        try:
                            file_path = tmpdir_path / f"{stem}_reports.csv"
                            self._csv.export_reports(reports, file_path)
                            _add_to_zip(zf, file_path)
                        except Exception as exc:
                            errors.append(f"csv_reports: {exc}")
                            logger.warning("Export CSV reports échoué : %s", exc)

                if self._settings.include_parquet:
                    if corpus is not None:
                        try:
                            file_path = tmpdir_path / f"{stem}_corpus.parquet"
                            self._parquet.export_corpus(corpus, file_path)
                            _add_to_zip(zf, file_path)
                        except Exception as exc:
                            errors.append(f"parquet_corpus: {exc}")
                            logger.warning("Export Parquet corpus échoué : %s", exc)

                    if rejected is not None:
                        try:
                            file_path = tmpdir_path / f"{stem}_rejected.parquet"
                            self._parquet.export_rejected(rejected, file_path)
                            _add_to_zip(zf, file_path)
                        except Exception as exc:
                            errors.append(f"parquet_rejected: {exc}")
                            logger.warning("Export Parquet rejected échoué : %s", exc)

                    if reports is not None:
                        try:
                            file_path = tmpdir_path / f"{stem}_reports.parquet"
                            self._parquet.export_reports(reports, file_path)
                            _add_to_zip(zf, file_path)
                        except Exception as exc:
                            errors.append(f"parquet_reports: {exc}")
                            logger.warning("Export Parquet reports échoué : %s", exc)

                manifest = ArchiveManifest(
                    created_at=_utc_now(),
                    files=tuple(sorted(manifest_files)),
                    metadata={**_to_mapping(metadata)},
                    errors=tuple(errors),
                )
                manifest_path = tmpdir_path / self._settings.manifest_name
                manifest_path.write_text(
                    _json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                _add_to_zip(zf, manifest_path)

        return path

    def export(self, **kwargs: Any) -> Path:
        return self.build(**kwargs)

    def __repr__(self) -> str:
        return f"ArchiveExporter(include_parquet={self._settings.include_parquet})"
