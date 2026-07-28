"""
==========================================================
Pair Exporter
==========================================================

Le PairExporter est la **seule** couche autorisée à écrire
sur le disque pour un run Discovery. Il est strictement
isolé du pipeline de découverte : il reçoit un
DiscoveryPairResult (objet valeur) et le sérialise.

Responsabilités :

- construire le répertoire de sortie
  ``<output_root>/<run_id>/<pair_slug>/`` ;
- sérialiser les 7 livrables par paire (summary JSON,
  corpus JSON / CSV / Parquet, rejected JSON, reports
  JSON, archive ZIP) ;
- respecter la politique ``summary_only`` des reports ;
- retourner un dict ``{format: path}``.

Le PairExporter ne calcule RIEN. Il ne touche pas au
pipeline de découverte. Il ne connaît pas la liste des
paires. C'est un objet *run-level* (il porte ``run_id``
et ``output_root``), pas un objet *per-pair* (il ne porte
pas le contexte d'exécution).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from exporters.archive import ArchiveExporter
from exporters.corpus import CorpusBuilder
from exporters.csv import CSVExporter
from exporters.json import JSONExporter
from exporters.parquet import ParquetExporter
from exporters.rejected import RejectedBuilder
from exporters.reports import ReportBundleBuilder

from .exceptions import ExportContractError
from .types import DiscoveryPairResult

logger = logging.getLogger("einherjar.exporter")


@dataclass(slots=True)
class PairExporter:
    """
    Exporter run-level pour les résultats d'une paire.

    Responsabilités :

    - connaît ``output_root`` et ``run_id`` (run-level) ;
    - expose ``export_pair(pair_result)`` qui écrit les
      7 fichiers et retourne leurs chemins ;
    - respecte ``export_full_reports`` (False par défaut,
      pour éviter le dump de 1.4 GB).

    Le PairExporter est *idempotent* dans le sens où
    ré-exporter le même ``pair_result`` écrase les
    fichiers existants (mkdir avec ``exist_ok=True``).
    """

    output_root: Path
    run_id: str
    export_full_reports: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_root", Path(self.output_root))
        object.__setattr__(self, "run_id", str(self.run_id).strip())

    def export_pair(
        self, pair_result: DiscoveryPairResult,
    ) -> dict[str, str]:
        """
        Exporte les livrables d'une paire sur disque.

        Contrat : renvoie un dict ``{format: path}`` avec
        au minimum ``summary_json`` même en cas d'échec
        du pipeline. Lève ``ExportContractError`` si la
        sérialisation du summary échoue (erreur
        considérée comme bloquante car c'est le fichier
        minimum de traçabilité).
        """

        target = pair_result.target
        pair_dir = self.output_root / self.run_id / target.slug
        pair_dir.mkdir(parents=True, exist_ok=True)
        stem = target.slug

        run_meta = {
            "asset": target.asset,
            "timeframe": target.timeframe,
            "run_id": self.run_id,
        }

        # 1) Summary — toujours écrit, même si pipeline KO.
        summary = self._build_summary(pair_result, run_meta)
        json_exporter = JSONExporter()
        paths: dict[str, str] = {}

        try:
            paths["summary_json"] = str(
                json_exporter.export(summary, pair_dir / f"{stem}_summary.json")
            )
        except Exception as exc:
            raise ExportContractError(
                f"JSON summary export failed for {target.key}: {exc!r}"
            ) from exc

        # Si la pipeline a échoué, on s'arrête là : pas la
        # peine d'essayer de sérialiser des objets None.
        if not pair_result.success or pair_result.einhers is None:
            logger.info(
                "[%s] pipeline KO — seul le summary est écrit "
                "(errors=%s)",
                target.key,
                pair_result.errors,
            )
            return paths

        # 2) Corpus / rejected / reports via builders
        # NOTE : on utilise from_portfolio_selection (pas
        # from_portfolio_report) pour que les Einhers rejetés
        # par le selector (ex: PF < 1) soient inclus dans
        # corpus.entries avec is_final=False, plutôt que perdus.
        try:
            corpus = CorpusBuilder.from_portfolio_selection(
                pair_result.selection,
                allocation=pair_result.allocation,
                include_rejected=True,
                asset=target.asset,
                timeframe=target.timeframe,
                calibrated_on=(
                    target.metadata.get("calibrated_on", "")
                    if hasattr(target, "metadata") else ""
                ),
                metadata={"run_id": self.run_id},
            )
        except Exception as exc:
            logger.warning(
                "[%s] CorpusBuilder failed: %r — corpus export skipped",
                target.key, exc,
            )
            return paths

        try:
            rejected = RejectedBuilder.from_report(
                pair_result.portfolio_report,
                metadata={"run_id": self.run_id},
            )
        except Exception as exc:
            logger.warning(
                "[%s] RejectedBuilder failed: %r — rejected export skipped",
                target.key, exc,
            )
            rejected = None

        try:
            bundle = ReportBundleBuilder.build(
                validation=None,
                execution=pair_result.execution_report,
                portfolio=pair_result.portfolio_report,
                metadata={"run_id": self.run_id},
            )
        except Exception as exc:
            logger.warning(
                "[%s] ReportBundleBuilder failed: %r — reports export skipped",
                target.key, exc,
            )
            bundle = None

        # 3) Sérialisation des 6 fichiers restants
        if corpus is not None:
            self._safe_export(
                "JSON corpus", target.key, paths,
                lambda: json_exporter.export_corpus(
                    corpus, pair_dir / f"{stem}_corpus.json",
                ),
                key="corpus_json",
            )
            self._safe_export(
                "CSV corpus", target.key, paths,
                lambda: CSVExporter().export_corpus(
                    corpus, pair_dir / f"{stem}_corpus.csv",
                ),
                key="corpus_csv",
            )
            self._safe_export(
                "Parquet corpus", target.key, paths,
                lambda: ParquetExporter().export_corpus(
                    corpus, pair_dir / f"{stem}_corpus.parquet",
                ),
                key="corpus_parquet",
            )

        if rejected is not None:
            self._safe_export(
                "JSON rejected", target.key, paths,
                lambda: json_exporter.export_rejected(
                    rejected, pair_dir / f"{stem}_rejected.json",
                ),
                key="rejected_json",
            )

        if bundle is not None:
            self._safe_export(
                "JSON reports", target.key, paths,
                lambda: json_exporter.export_reports(
                    bundle,
                    pair_dir / f"{stem}_reports.json",
                    summary_only=not self.export_full_reports,
                ),
                key="reports_json",
            )

        self._safe_export(
            "Archive", target.key, paths,
            lambda: ArchiveExporter().build(
                corpus=corpus,
                rejected=rejected,
                reports=bundle,
                path=pair_dir / f"{stem}.zip",
                stem=stem,
                metadata=run_meta,
            ) if corpus is not None else None,
            key="archive",
        )

        return paths

    def _build_summary(
        self,
        pair_result: DiscoveryPairResult,
        run_meta: dict[str, Any],
    ) -> dict[str, Any]:
        state = pair_result.state
        return {
            "target": pair_result.target.to_dict(),
            "index": pair_result.index,
            "state": state.to_dict() if state is not None else {},
            "success": bool(pair_result.success),
            "errors": list(pair_result.errors or ()),
            "einher_count": len(pair_result.einhers or ()),
            "execution_count": len(pair_result.execution_results or ()),
            "validated_count": len(pair_result.validated or ()),
            "rejected_count": len(pair_result.rejected or ()),
            "started_at": (
                pair_result.started_at.isoformat()
                if pair_result.started_at else None
            ),
            "finished_at": (
                pair_result.finished_at.isoformat()
                if pair_result.finished_at else None
            ),
            "metadata": dict(run_meta),
        }

    def _safe_export(
        self,
        label: str,
        target_key: str,
        paths: dict[str, str],
        exporter_callable,
        *,
        key: str,
    ) -> None:
        """
        Exécute un export en mode best-effort.

        En cas d'échec, on log un warning et on continue.
        Les exports best-effort ne bloquent jamais la
        sérialisation des autres fichiers — seul le
        summary JSON est obligatoire.
        """

        try:
            result = exporter_callable()
        except Exception as exc:
            logger.warning(
                "[%s] %s export failed: %r",
                target_key, label, exc,
            )
            return
        if result is None:
            return
        paths[key] = str(result)


__all__ = ["PairExporter"]
