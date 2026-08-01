"""
==========================================================
Core Runner
==========================================================

Le runner est le **point d'entrée Discovery**. Il est le
SEUL à connaître la liste des paires, le run_id, le
chemin de sortie, et la politique de poursuite sur
erreur. L'Engine (per-pair) ne sait rien de tout ça.

Responsabilités :

1. Construire la liste résolue de cibles
   (asset / timeframe).
2. Générer le ``run_id`` (timestamp UTC).
3. Instancier un ``core.Engine`` (pure per-pair).
4. Instancier un ``core.exporter.PairExporter``
   (run-level, gestion de l'écriture disque).
5. Pour chaque cible :
   - appeler ``engine.run_pair(target)`` ;
   - si l'export est activé, appeler
     ``exporter.export_pair(result)`` ;
   - respecter la politique ``continue_on_error``.
6. Agréger les ``DiscoveryPairResult`` dans un
   ``DiscoveryRunResult``.
7. Écrire le résumé global du run (run_summary.json).

Le runner ne contient AUCUNE logique métier d'algo. Il
délègue tout : Engine pour la pipeline per-pair,
PairExporter pour la persistance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any
from typing import Mapping
from typing import Sequence

from .engine import Engine
from .exceptions import DiscoveryError
from .exporter import PairExporter
from .types import DiscoveryPairResult
from .types import DiscoveryTarget

logger = logging.getLogger("einherjar.runner")


# ==========================================================
# HELPERS
# ==========================================================

def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _to_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            value = value.to_dict()
        except Exception:
            value = None
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)


def _normalize_text(value: Any, default: str = "") -> str:
    text = str(value or "").strip()
    return text if text else default


def _resolve_target_from_any(value: Any) -> DiscoveryTarget:
    if isinstance(value, DiscoveryTarget):
        return value
    if isinstance(value, str):
        text = value.strip()
        if "@" in text:
            asset, timeframe = text.split("@", 1)
            return DiscoveryTarget(
                asset=asset.strip(), timeframe=timeframe.strip(),
            )
        return DiscoveryTarget(asset=text or "unknown", timeframe="unknown")
    if isinstance(value, Mapping):
        return DiscoveryTarget(
            asset=_normalize_text(
                value.get("asset") or value.get("symbol") or "unknown",
                "unknown",
            ),
            timeframe=_normalize_text(
                value.get("timeframe") or value.get("tf") or "unknown",
                "unknown",
            ),
            metadata=_to_mapping(value.get("metadata", {})),
        )
    raise TypeError(
        f"Unsupported target type: {type(value).__name__}."
    )


# ==========================================================
# DISCOVERY SETTINGS
# ==========================================================

@dataclass(slots=True)
class DiscoverySettings:
    """
    Paramètres d'un run Discovery.

    Consommés par DiscoveryOrchestrator pour piloter
    l'itération sur les paires.
    """

    pairs: tuple[DiscoveryTarget, ...] = ()
    assets: tuple[str, ...] = ()
    timeframes: tuple[str, ...] = ()

    output_root: Path = field(default_factory=lambda: Path("outputs"))
    run_name: str = ""
    max_pairs: int = 0

    export_pair_results: bool = True
    export_run_summary: bool = True
    export_full_reports: bool = False
    continue_on_error: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "assets",
            tuple(
                _normalize_text(x) for x in self.assets
                if _normalize_text(x)
            ),
        )
        object.__setattr__(
            self, "timeframes",
            tuple(
                _normalize_text(x) for x in self.timeframes
                if _normalize_text(x)
            ),
        )
        object.__setattr__(
            self, "pairs",
            tuple(_resolve_target_from_any(x) for x in self.pairs),
        )
        object.__setattr__(self, "output_root", Path(self.output_root))
        object.__setattr__(self, "run_name", _normalize_text(self.run_name))
        object.__setattr__(
            self, "max_pairs", max(0, _coerce_int(self.max_pairs, 0)),
        )
        object.__setattr__(
            self, "export_pair_results",
            _coerce_bool(self.export_pair_results, True),
        )
        object.__setattr__(
            self, "export_run_summary",
            _coerce_bool(self.export_run_summary, True),
        )
        object.__setattr__(
            self, "export_full_reports",
            _coerce_bool(self.export_full_reports, False),
        )
        object.__setattr__(
            self, "continue_on_error",
            _coerce_bool(self.continue_on_error, True),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "pairs": [p.to_dict() for p in self.pairs],
            "assets": list(self.assets),
            "timeframes": list(self.timeframes),
            "output_root": str(self.output_root),
            "run_name": self.run_name,
            "max_pairs": self.max_pairs,
            "export_pair_results": self.export_pair_results,
            "export_run_summary": self.export_run_summary,
            "export_full_reports": self.export_full_reports,
            "continue_on_error": self.continue_on_error,
        }


# ==========================================================
# DISCOVERY RUN RESULT
# ==========================================================

@dataclass(slots=True)
class DiscoveryRunResult:
    """
    Résultat global d'un run Discovery.

    Agrège les DiscoveryPairResult de chaque paire.
    """

    run_id: str
    settings: DiscoverySettings
    pair_results: list[DiscoveryPairResult] = field(default_factory=list)
    errors: tuple[str, ...] = ()
    started_at: datetime = field(default_factory=_utc_now)
    finished_at: datetime | None = None
    export_paths: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "pair_results", list(self.pair_results))
        object.__setattr__(self, "errors", tuple(str(e) for e in self.errors))
        object.__setattr__(self, "export_paths", dict(self.export_paths))
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "run_id", _normalize_text(self.run_id))

    @property
    def pair_count(self) -> int:
        return len(self.pair_results)

    @property
    def success_count(self) -> int:
        return sum(1 for r in self.pair_results if r.success)

    @property
    def failure_count(self) -> int:
        return sum(1 for r in self.pair_results if not r.success)

    @property
    def total_einher_count(self) -> int:
        return sum(r.einher_count for r in self.pair_results)

    def summary(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "pair_count": self.pair_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "total_einher_count": self.total_einher_count,
            "errors": list(self.errors),
            "started_at": self.started_at.isoformat(),
            "finished_at": (
                self.finished_at.isoformat() if self.finished_at else None
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "settings": self.settings.to_dict(),
            "pair_results": [r.to_dict() for r in self.pair_results],
            "errors": list(self.errors),
            "started_at": self.started_at.isoformat(),
            "finished_at": (
                self.finished_at.isoformat() if self.finished_at else None
            ),
            "export_paths": dict(self.export_paths),
            "metadata": dict(self.metadata),
            "summary": self.summary(),
        }

    def __repr__(self) -> str:
        return (
            "DiscoveryRunResult("
            f"run_id='{self.run_id}', "
            f"pairs={self.pair_count}, "
            f"success={self.success_count}, "
            f"failures={self.failure_count}"
            ")"
        )


# ==========================================================
# DISCOVERY ORCHESTRATOR (BOOTSTRAP)
# ==========================================================

class DiscoveryOrchestrator:
    """
    Bootstrap du run Discovery.

    Responsabilités :
    - construire la liste de cibles,
    - instancier un Engine,
    - itérer sur les cibles via Engine.run_pair(),
    - agréger les résultats,
    - exporter le résumé global.

    AUCUNE logique métier n'est implémentée ici.
    """

    # ==================================================
    # INITIALIZATION
    # ==================================================

    def __init__(
        self,
        config: Any,
        *,
        settings: DiscoverySettings | None = None,
    ) -> None:

        self._config = config
        self._settings = settings or DiscoverySettings()
        self._run_id: str = ""

    # ==================================================
    # PROPERTIES
    # ==================================================

    @property
    def config(self) -> Any:
        return self._config

    @property
    def settings(self) -> DiscoverySettings:
        return self._settings

    @property
    def run_id(self) -> str:
        return self._run_id

    # ==================================================
    # ENTRY POINT
    # ==================================================

    def run(
        self,
        *,
        pairs: Sequence[Any] | None = None,
        assets: Sequence[str] | None = None,
        timeframes: Sequence[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> DiscoveryRunResult:
        """
        Lance le run sur la liste résolue de cibles.

        Pipeline :
        1. Résout les cibles.
        2. Crée le run_id (timestamp UTC).
        3. Instancie un Engine et un PairExporter.
        4. Pour chaque cible : engine.run_pair puis
           exporter.export_pair.
        5. Agrège et exporte le résumé global.
        """

        targets = self.resolve_targets(
            pairs=pairs, assets=assets, timeframes=timeframes,
        )
        if not targets:
            raise ValueError(
                "No asset/timeframe pair provided to "
                "DiscoveryOrchestrator.run()."
            )

        self._run_id = _utc_now().strftime("run_%Y%m%d_%H%M%S")

        engine = Engine(self._config)
        exporter = PairExporter(
            output_root=self._settings.output_root,
            run_id=self._run_id,
            export_full_reports=self._settings.export_full_reports,
        )

        started_at = _utc_now()
        pair_results: list[DiscoveryPairResult] = []
        errors: list[str] = []

        for index, target in enumerate(targets):
            if self._settings.max_pairs and index >= self._settings.max_pairs:
                break

            # L'Engine reçoit le run_id via target.metadata
            # pour les champs informatifs (memory, knowledge,
            # portfolio metadata) ; mais le run_id n'est PAS
            # utilisé pour décider quoi que ce soit côté Engine.
            target_with_meta = DiscoveryTarget(
                asset=target.asset,
                timeframe=target.timeframe,
                metadata={
                    **dict(getattr(target, "metadata", {}) or {}),
                    "run_id": self._run_id,
                },
            )

            try:
                result = engine.run_pair(
                    target_with_meta, index=index, metadata=metadata,
                )
            except DiscoveryError as exc:
                if not self._settings.continue_on_error:
                    raise
                errors.append(f"{target.key}: {exc!r}")
                logger.error("Pair %s failed: %r", target.key, exc)
                # On construit un failure result minimal pour
                # que l'exporter puisse quand même écrire un
                # summary de l'échec.
                result = self._make_minimal_failure_result(
                    target_with_meta, index, exc, metadata,
                )

            # Export best-effort : l'exporter écrit au moins
            # un summary même en cas d'échec de pipeline.
            if self._settings.export_pair_results:
                try:
                    export_paths = exporter.export_pair(result)
                    result.export_paths.update(export_paths)
                except Exception as exc:
                    logger.error(
                        "[%s] export failed: %r", target.key, exc,
                    )
                    errors.append(f"{target.key} export: {exc!r}")

            pair_results.append(result)

        run_result = DiscoveryRunResult(
            run_id=self._run_id,
            settings=self._settings,
            pair_results=pair_results,
            errors=tuple(errors),
            started_at=started_at,
            finished_at=_utc_now(),
            metadata=dict(metadata or {}),
        )

        if self._settings.export_run_summary:
            export_paths = self._export_run_summary(run_result)
            run_result.export_paths.update(export_paths)

        return run_result

    def _make_minimal_failure_result(
        self,
        target: DiscoveryTarget,
        index: int,
        exc: BaseException,
        metadata: Mapping[str, Any] | None,
    ) -> DiscoveryPairResult:
        """
        Construit un DiscoveryPairResult minimal pour qu'un
        échec d'engine.run_pair soit quand même exportable
        (summary traçant l'erreur).
        """

        from .state import EngineState

        state = EngineState()
        state.start()
        state.fail(repr(exc))
        state.finish(success=False)

        return DiscoveryPairResult(
            target=target,
            index=index,
            state=state,
            dataset=None,
            candidates=(),
            validated=(),
            rejected=(),
            execution_results=(),
            execution_report=None,
            einhers=(),
            selection=None,
            allocation=None,
            portfolio_report=None,
            memory_snapshot={},
            knowledge_snapshot={},
            export_paths={},
            metadata=dict(metadata or {}),
            started_at=state.started_at or _utc_now(),
            finished_at=state.finished_at or _utc_now(),
            success=False,
            errors=(repr(exc),),
        )

    # ==================================================
    # TARGET RESOLUTION
    # ==================================================

    def resolve_targets(
        self,
        *,
        pairs: Sequence[Any] | None = None,
        assets: Sequence[str] | None = None,
        timeframes: Sequence[str] | None = None,
    ) -> tuple[DiscoveryTarget, ...]:
        if pairs is not None and len(pairs) > 0:
            return tuple(_resolve_target_from_any(p) for p in pairs if p is not None)

        resolved_assets = tuple(
            _normalize_text(a) for a in (
                assets if assets is not None else self._settings.assets
            ) if _normalize_text(a)
        )
        resolved_timeframes = tuple(
            _normalize_text(tf) for tf in (
                timeframes if timeframes is not None else self._settings.timeframes
            ) if _normalize_text(tf)
        )

        if self._settings.pairs:
            return tuple(self._settings.pairs)

        if resolved_assets and resolved_timeframes:
            return tuple(
                DiscoveryTarget(asset=a, timeframe=tf)
                for a in resolved_assets
                for tf in resolved_timeframes
            )

        if resolved_assets:
            return tuple(
                DiscoveryTarget(asset=a, timeframe="unknown")
                for a in resolved_assets
            )

        if resolved_timeframes:
            return tuple(
                DiscoveryTarget(asset="unknown", timeframe=tf)
                for tf in resolved_timeframes
            )

        return ()

    # ==================================================
    # RUN-LEVEL EXPORT
    # ==================================================

    def _export_run_summary(
        self, run_result: DiscoveryRunResult,
    ) -> dict[str, str]:
        from exporters.json import JSONExporter

        run_dir = self._settings.output_root / self._run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        json_exporter = JSONExporter()
        paths: dict[str, str] = {}
        try:
            paths["run_summary_json"] = str(
                json_exporter.export(
                    run_result.to_dict(),
                    run_dir / "run_summary.json",
                )
            )
        except Exception as exc:
            logger.warning("Run summary export failed: %r", exc)
        return paths

    # ==================================================
    # PYTHON PROTOCOL
    # ==================================================

    def __repr__(self) -> str:
        return (
            "DiscoveryOrchestrator("
            f"run_id='{getattr(self, '_run_id', '<not-started>')}', "
            f"pairs={len(self.resolve_targets())}"
            ")"
        )


# ==========================================================
# ENTRY POINTS
# ==========================================================

def main(
    config: Any | None = None,
    *,
    pairs: Sequence[Any] | None = None,
    assets: Sequence[str] | None = None,
    timeframes: Sequence[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> DiscoveryRunResult:
    """
    Point d'entrée fonctionnel.
    """

    orchestrator = DiscoveryOrchestrator(config=config)
    return orchestrator.run(
        pairs=pairs,
        assets=assets,
        timeframes=timeframes,
        metadata=metadata,
    )
