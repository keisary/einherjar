"""
==========================================================
Discovery Entry Point
==========================================================

Orchestre le pipeline complet du système sur une grille
asset / timeframe.

Rôle :
- charger la configuration,
- itérer sur les paires asset / timeframe,
- instancier les modules,
- chaîner discovery -> validation -> execution -> portfolio,
- produire le corpus final,
- exporter les résultats,
- alimenter la mémoire et la connaissance.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# ==========================================================
# PATH / IMPORT SAFETY
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ==========================================================
# CONFIG FALLBACKS
# ==========================================================

try:
    from config.search import SearchConfig  # type: ignore
except Exception:  # pragma: no cover
    @dataclass(slots=True, frozen=True)
    class SearchConfig:  # type: ignore[no-redef]
        max_conditions: int = 3
        beam_width: int = 200
        max_depth: int = 3
        max_candidates_per_family: int = 100
        exploration_ratio: float = 0.25
        exploitation_ratio: float = 0.75
        novelty_weight: float = 0.30
        diversity_weight: float = 0.25
        family_balance_weight: float = 0.20
        random_seed: int = 42


try:
    from config.execution import ExecutionConfig  # type: ignore
except Exception:  # pragma: no cover
    @dataclass(slots=True, frozen=True)
    class ExecutionConfig:  # type: ignore[no-redef]
        fees: float = 0.0006
        slippage: float = 0.0002
        spread: float = 0.0001
        allow_long: bool = True
        allow_short: bool = True
        max_open_positions: int = 1


# ==========================================================
# HELPERS
# ==========================================================


def _resolve_asset_class(asset: str) -> str:
    """Résout la classe d'actif depuis assets_v1.json."""
    assets_path = Path(r"D:/midas_v2/einherjar/config/assets_v1.json")
    if assets_path.exists():
        try:
            data = json.loads(assets_path.read_text(encoding="utf-8"))
            for entry in data.get("assets", []):
                if entry.get("asset") == asset:
                    return entry.get("class", "unknown")
        except Exception:
            pass
    return "unknown"

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


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
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


def _slugify(value: Any) -> str:
    text = _normalize_text(value, "unknown").lower()
    out = []
    for ch in text:
        if ch.isalnum():
            out.append(ch)
        elif ch in {" ", "-", "_", ".", "/", ":"}:
            out.append("_")
    slug = "".join(out)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_") or "unknown"


def _as_sequence(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    if isinstance(value, set):
        return tuple(value)
    if isinstance(value, (str, bytes)):
        return (value,)
    return (value,)


def _first_non_none(*values: Any, default: Any = None) -> Any:
    for value in values:
        if value is not None:
            return value
    return default


def _extract_attr(obj: Any, *names: str, default: Any = None) -> Any:
    if obj is None:
        return default
    for name in names:
        if isinstance(obj, Mapping) and name in obj:
            value = obj.get(name)
            if value is not None:
                return value
        if hasattr(obj, name):
            value = getattr(obj, name)
            if value is not None:
                return value
    return default


def _try_call(obj: Any, method_names: Sequence[str], *args: Any, **kwargs: Any) -> Any:
    if obj is None:
        return None

    if callable(obj) and not inspect.ismodule(obj):
        try:
            return obj(*args, **kwargs)
        except TypeError:
            try:
                return obj(**kwargs)
            except TypeError:
                try:
                    return obj(*args)
                except TypeError:
                    pass

    for name in method_names:
        if not hasattr(obj, name):
            continue
        fn = getattr(obj, name)
        if not callable(fn):
            continue
        try:
            return fn(*args, **kwargs)
        except TypeError:
            try:
                return fn(**kwargs)
            except TypeError:
                try:
                    return fn(*args)
                except TypeError:
                    continue
    return None


def _load_module(module_name: str) -> Any | None:
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


def _load_symbol(module_names: Sequence[str], symbol_names: Sequence[str]) -> Any | None:
    for module_name in module_names:
        module = _load_module(module_name)
        if module is None:
            continue
        for symbol_name in symbol_names:
            symbol = getattr(module, symbol_name, None)
            if symbol is not None:
                return symbol
    return None


def _instantiate(
    module_names: Sequence[str],
    class_names: Sequence[str],
    *,
    config: Any | None = None,
    allow_module_fallback: bool = True,
) -> Any | None:
    for module_name in module_names:
        module = _load_module(module_name)
        if module is None:
            continue

        for class_name in class_names:
            cls = getattr(module, class_name, None)
            if cls is None:
                continue

            if inspect.isclass(cls) or callable(cls):
                if config is not None:
                    for kwargs in (
                        {"config": config},
                        {"settings": config},
                        {},
                    ):
                        try:
                            return cls(**kwargs)  # type: ignore[misc]
                        except Exception:
                            continue
                    # Essayer from_config / from_settings
                    for factory in ("from_config", "from_settings"):
                        if hasattr(cls, factory) and callable(getattr(cls, factory)):
                            try:
                                return getattr(cls, factory)(config)  # type: ignore[misc]
                            except Exception:
                                continue
                else:
                    try:
                        return cls()  # type: ignore[misc]
                    except Exception:
                        continue

        if allow_module_fallback:
            return module

    return None


def _extract_items(obj: Any, *names: str) -> list[Any]:
    if obj is None:
        return []

    if isinstance(obj, (list, tuple, set)):
        return list(obj)

    for name in names:
        value = _extract_attr(obj, name)
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            return list(value)
        if hasattr(value, "__iter__") and not isinstance(value, (str, bytes, Mapping)):
            try:
                return list(value)
            except TypeError:
                pass
        return [value]

    return [obj]


def _extract_feature_key(value: Any) -> str:
    if value is None:
        return "unknown"

    meta = _to_mapping(_extract_attr(value, "metadata", default=None))
    for key in ("feature", "feature_key", "target_feature", "seed_feature"):
        if key in meta and meta[key] is not None:
            text = _normalize_text(meta[key], "unknown").lower()
            if text:
                return text

    for name in ("feature", "feature_key", "target_feature", "seed_feature"):
        attr = _extract_attr(value, name, default=None)
        if attr is not None:
            text = _normalize_text(attr, "unknown").lower()
            if text:
                return text

    return "unknown"


def _extract_family_key(value: Any) -> str:
    if value is None:
        return "unknown"

    for name in ("family", "target_family", "portfolio_family"):
        attr = _extract_attr(value, name, default=None)
        if attr is not None:
            text = _normalize_text(attr, "unknown").lower()
            if text:
                return text

    meta = _to_mapping(_extract_attr(value, "metadata", default=None))
    for key in ("family", "target_family", "portfolio_family"):
        if key in meta and meta[key] is not None:
            text = _normalize_text(meta[key], "unknown").lower()
            if text:
                return text

    return "unknown"


def _extract_profile_key(value: Any) -> str:
    if value is None:
        return "unknown"

    profile = _extract_attr(value, "profile", default=None)
    if profile is not None and _extract_attr(profile, "name", default=None):
        text = _normalize_text(_extract_attr(profile, "name"), "unknown").lower()
        if text:
            return text

    for name in ("profile_name", "strategy_name", "einher_name"):
        attr = _extract_attr(value, name, default=None)
        if attr is not None:
            text = _normalize_text(attr, "unknown").lower()
            if text:
                return text

    meta = _to_mapping(_extract_attr(value, "metadata", default=None))
    for key in ("profile_name", "strategy_name", "einher_name"):
        if key in meta and meta[key] is not None:
            text = _normalize_text(meta[key], "unknown").lower()
            if text:
                return text

    return "unknown"


def _resolve_pair_any(value: Any) -> "DiscoveryTarget":
    if isinstance(value, DiscoveryTarget):
        return value

    if isinstance(value, str):
        text = value.strip()
        if "@" in text:
            asset, timeframe = text.split("@", 1)
            return DiscoveryTarget(asset=asset.strip(), timeframe=timeframe.strip())

        return DiscoveryTarget(asset=text or "unknown", timeframe="unknown")

    if isinstance(value, Mapping):
        return DiscoveryTarget(
            asset=_normalize_text(value.get("asset") or value.get("symbol") or value.get("instrument"), "unknown"),
            timeframe=_normalize_text(value.get("timeframe") or value.get("tf"), "unknown"),
            metadata=_to_mapping(value.get("metadata", {})),
        )

    asset = _extract_attr(value, "asset", "symbol", "instrument", default="unknown")
    timeframe = _extract_attr(value, "timeframe", "tf", default="unknown")
    metadata = _to_mapping(_extract_attr(value, "metadata", default={}))
    return DiscoveryTarget(asset=_normalize_text(asset, "unknown"), timeframe=_normalize_text(timeframe, "unknown"), metadata=metadata)


def _build_search_config(config: Any | None) -> SearchConfig:
    root = _to_mapping(config)
    disc = _to_mapping(root.get("discovery", root.get("search", root.get("search_config", {}))))
    dataset = _to_mapping(root.get("dataset", {}))

    def pick(*keys: str, default: Any = None) -> Any:
        for key in keys:
            if key in disc and disc[key] is not None:
                return disc[key]
            if key in root and root[key] is not None:
                return root[key]
            if key in dataset and dataset[key] is not None:
                return dataset[key]
        return default

    return SearchConfig(
        max_conditions=_coerce_int(pick("max_conditions", default=3), 3),
        beam_width=_coerce_int(pick("beam_width", default=200), 200),
        max_depth=_coerce_int(pick("max_depth", default=3), 3),
        max_candidates_per_family=_coerce_int(pick("max_candidates_per_family", default=100), 100),
        exploration_ratio=_coerce_float(pick("exploration_ratio", default=0.25), 0.25),
        exploitation_ratio=_coerce_float(pick("exploitation_ratio", default=0.75), 0.75),
        novelty_weight=_coerce_float(pick("novelty_weight", default=0.30), 0.30),
        diversity_weight=_coerce_float(pick("diversity_weight", default=0.25), 0.25),
        family_balance_weight=_coerce_float(pick("family_balance_weight", default=0.20), 0.20),
        random_seed=_coerce_int(pick("random_seed", default=42), 42),
    )


def _build_execution_config(config: Any | None) -> ExecutionConfig:
    root = _to_mapping(config)
    exec_cfg = _to_mapping(root.get("execution", root.get("execution_config", {})))

    def pick(*keys: str, default: Any = None) -> Any:
        for key in keys:
            if key in exec_cfg and exec_cfg[key] is not None:
                return exec_cfg[key]
            if key in root and root[key] is not None:
                return root[key]
        return default

    return ExecutionConfig(
        fees=_coerce_float(pick("fees", default=0.0006), 0.0006),
        slippage=_coerce_float(pick("slippage", default=0.0002), 0.0002),
        spread=_coerce_float(pick("spread", default=0.0001), 0.0001),
        allow_long=_coerce_bool(pick("allow_long", default=True), True),
        allow_short=_coerce_bool(pick("allow_short", default=True), True),
        max_open_positions=_coerce_int(pick("max_open_positions", default=1), 1),
    )


def _summarize_counts(items: Iterable[str]) -> dict[str, int]:
    return dict(Counter(item for item in items if item))


# ==========================================================
# SETTINGS / TARGETS / CONTEXT
# ==========================================================

@dataclass(slots=True, frozen=True)
class DiscoverySettings:
    """
    Paramètres d'orchestration du run Discovery.
    """

    assets: tuple[str, ...] = ()
    timeframes: tuple[str, ...] = ()
    pairs: tuple["DiscoveryTarget", ...] = ()

    output_root: Path = field(default_factory=lambda: Path("outputs"))
    run_name: str = ""
    max_pairs: int = 0

    export_formats: tuple[str, ...] = ("json", "csv", "archive")
    export_pair_results: bool = True
    export_run_summary: bool = True

    continue_on_error: bool = True
    build_knowledge: bool = True
    build_memory: bool = True

    search_config: SearchConfig = field(default_factory=SearchConfig)
    execution_config: ExecutionConfig = field(default_factory=ExecutionConfig)

    def __post_init__(self) -> None:
        object.__setattr__(self, "assets", tuple(_normalize_text(x) for x in _as_sequence(self.assets) if _normalize_text(x)))
        object.__setattr__(self, "timeframes", tuple(_normalize_text(x) for x in _as_sequence(self.timeframes) if _normalize_text(x)))
        object.__setattr__(self, "pairs", tuple(_resolve_pair_any(x) for x in _as_sequence(self.pairs) if x is not None))
        object.__setattr__(self, "output_root", Path(self.output_root))
        object.__setattr__(self, "run_name", _normalize_text(self.run_name))
        object.__setattr__(self, "max_pairs", max(0, _coerce_int(self.max_pairs, 0)))
        object.__setattr__(self, "export_formats", tuple(_normalize_text(x).lower() for x in _as_sequence(self.export_formats) if _normalize_text(x)))
        object.__setattr__(self, "export_pair_results", _coerce_bool(self.export_pair_results, True))
        object.__setattr__(self, "export_run_summary", _coerce_bool(self.export_run_summary, True))
        object.__setattr__(self, "continue_on_error", _coerce_bool(self.continue_on_error, True))
        object.__setattr__(self, "build_knowledge", _coerce_bool(self.build_knowledge, True))
        object.__setattr__(self, "build_memory", _coerce_bool(self.build_memory, True))

    @classmethod
    def from_config(cls, config: Any | None) -> "DiscoverySettings":
        root = _to_mapping(config)
        disc = _to_mapping(root.get("discovery", root.get("discovery_config", {})))
        dataset = _to_mapping(root.get("dataset", {}))
        output = _to_mapping(root.get("output", root.get("paths", {})))

        def pick(*keys: str, default: Any = None) -> Any:
            for key in keys:
                if key in disc and disc[key] is not None:
                    return disc[key]
                if key in dataset and dataset[key] is not None:
                    return dataset[key]
                if key in root and root[key] is not None:
                    return root[key]
                if key in output and output[key] is not None:
                    return output[key]
            return default

        assets = _as_sequence(pick("assets", default=()))
        timeframes = _as_sequence(pick("timeframes", default=()))
        pairs = _as_sequence(pick("pairs", default=()))

        output_root = pick("output_root", "outputs_root", "root", default="outputs")
        run_name = pick("run_name", "name", default="")

        return cls(
            assets=tuple(str(x) for x in assets if _normalize_text(x)),
            timeframes=tuple(str(x) for x in timeframes if _normalize_text(x)),
            pairs=tuple(_resolve_pair_any(x) for x in pairs if x is not None),
            output_root=Path(output_root),
            run_name=_normalize_text(run_name),
            max_pairs=_coerce_int(pick("max_pairs", default=0), 0),
            export_formats=tuple(_as_sequence(pick("export_formats", default=("json", "csv", "archive")))),
            export_pair_results=_coerce_bool(pick("export_pair_results", default=True), True),
            export_run_summary=_coerce_bool(pick("export_run_summary", default=True), True),
            continue_on_error=_coerce_bool(pick("continue_on_error", default=True), True),
            build_knowledge=_coerce_bool(pick("build_knowledge", default=True), True),
            build_memory=_coerce_bool(pick("build_memory", default=True), True),
            search_config=_build_search_config(config),
            execution_config=_build_execution_config(config),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "assets": list(self.assets),
            "timeframes": list(self.timeframes),
            "pairs": [pair.to_dict() for pair in self.pairs],
            "output_root": str(self.output_root),
            "run_name": self.run_name,
            "max_pairs": self.max_pairs,
            "export_formats": list(self.export_formats),
            "export_pair_results": self.export_pair_results,
            "export_run_summary": self.export_run_summary,
            "continue_on_error": self.continue_on_error,
            "build_knowledge": self.build_knowledge,
            "build_memory": self.build_memory,
            "search_config": self.search_config.__dict__,
            "execution_config": self.execution_config.__dict__,
        }


@dataclass(slots=True, frozen=True)
class DiscoveryTarget:
    """
    Une paire asset / timeframe.
    """

    asset: str
    timeframe: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "asset", _normalize_text(self.asset, "unknown"))
        object.__setattr__(self, "timeframe", _normalize_text(self.timeframe, "unknown"))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def key(self) -> str:
        return f"{self.asset}@{self.timeframe}"

    @property
    def slug(self) -> str:
        return f"{_slugify(self.asset)}__{_slugify(self.timeframe)}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "asset": self.asset,
            "timeframe": self.timeframe,
            "metadata": dict(self.metadata),
            "key": self.key,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DiscoveryTarget":
        return cls(
            asset=data.get("asset", "unknown"),
            timeframe=data.get("timeframe", "unknown"),
            metadata=_to_mapping(data.get("metadata", {})),
        )


@dataclass(slots=True)
class DiscoveryContext:
    """
    Contexte d'un run sur une paire donnée.
    """

    run_id: str
    target: DiscoveryTarget
    index: int
    metadata: dict[str, Any] = field(default_factory=dict)
    started_at: datetime = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _normalize_text(self.run_id))
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "index", max(0, _coerce_int(self.index, 0)))

    @property
    def asset(self) -> str:
        return self.target.asset

    @property
    def timeframe(self) -> str:
        return self.target.timeframe

    @property
    def key(self) -> str:
        return self.target.key

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "target": self.target.to_dict(),
            "index": self.index,
            "metadata": dict(self.metadata),
            "started_at": self.started_at.isoformat(),
        }


# ==========================================================
# COMPONENTS
# ==========================================================

@dataclass(slots=True)
class DiscoveryComponents:
    """
    Registre des composants utilisés par l'orchestrateur.
    """

    dataset_loader: Any | None = None
    dataset_validator: Any | None = None

    discovery_generator: Any | None = None
    discovery_explorer: Any | None = None
    validation_engine: Any | None = None

    execution_engine: Any | None = None

    portfolio_selector: Any | None = None
    portfolio_correlation: Any | None = None
    portfolio_diversification: Any | None = None
    portfolio_risk: Any | None = None
    portfolio_capital: Any | None = None
    portfolio_allocator: Any | None = None
    portfolio_reporter: Any | None = None
    portfolio_optimizer: Any | None = None

    corpus_cls: Any | None = None
    rejected_cls: Any | None = None
    corpus_builder: Any | None = None
    rejected_builder: Any | None = None
    report_bundle_builder: Any | None = None

    json_exporter: Any | None = None
    csv_exporter: Any | None = None
    parquet_exporter: Any | None = None
    archive_exporter: Any | None = None

    search_history: Any | None = None
    explored_regions: Any | None = None
    successful_regions: Any | None = None
    failed_regions: Any | None = None
    feature_history: Any | None = None
    family_history: Any | None = None
    corpus_history: Any | None = None
    learning_engine: Any | None = None

    fingerprint_registry: Any | None = None
    knowledge_graph: Any | None = None
    taxonomy_engine: Any | None = None
    ontology_engine: Any | None = None
    cluster_engine: Any | None = None
    insight_engine: Any | None = None

    @classmethod
    def create(cls, config: Any | None) -> "DiscoveryComponents":
        # Les composants stateful globaux (memory, knowledge, exporters)
        # sont instanciés une fois. Les composants par-paire (loader,
        # generator, validator, execution, portfolio) sont construits
        # à la volée dans run_pair() quand les dépendances sont connues.
        return cls(
            dataset_loader=None,
            dataset_validator=None,
            discovery_generator=None,
            discovery_explorer=None,
            validation_engine=None,
            execution_engine=None,
            portfolio_selector=None,
            portfolio_correlation=None,
            portfolio_diversification=None,
            portfolio_risk=None,
            portfolio_capital=None,
            portfolio_allocator=None,
            portfolio_reporter=None,
            portfolio_optimizer=None,
            corpus_cls=_load_symbol(("exporters.corpus",), ("Corpus",)),
            rejected_cls=_load_symbol(("exporters.rejected",), ("RejectedCorpus",)),
            corpus_builder=_load_symbol(("exporters.corpus",), ("CorpusBuilder",)),
            rejected_builder=_load_symbol(("exporters.rejected",), ("RejectedBuilder",)),
            report_bundle_builder=_load_symbol(("exporters.reports",), ("ReportBundleBuilder",)),
            json_exporter=_instantiate(
                ("exporters.json",),
                ("JSONExporter",),
                config=None,
            ),
            csv_exporter=_instantiate(
                ("exporters.csv",),
                ("CSVExporter",),
                config=None,
            ),
            parquet_exporter=_instantiate(
                ("exporters.parquet",),
                ("ParquetExporter",),
                config=None,
            ),
            archive_exporter=_instantiate(
                ("exporters.archive",),
                ("ArchiveExporter",),
                config=None,
            ),
            search_history=_instantiate(("memory.search_history",), ("SearchHistory",), config=None),
            explored_regions=_instantiate(("memory.explored_regions",), ("ExploredRegions",), config=None),
            successful_regions=_instantiate(("memory.successful_regions",), ("SuccessfulRegions",), config=None),
            failed_regions=_instantiate(("memory.failed_regions",), ("FailedRegions",), config=None),
            feature_history=_instantiate(("memory.feature_history",), ("FeatureHistory",), config=None),
            family_history=_instantiate(("memory.family_history",), ("FamilyHistory",), config=None),
            corpus_history=_instantiate(("memory.corpus_history",), ("CorpusHistory",), config=None),
            learning_engine=_instantiate(("memory.learning",), ("LearningEngine",), config=None),
            fingerprint_registry=_instantiate(("knowledge.fingerprints",), ("FingerprintRegistry",), config=None),
            knowledge_graph=_instantiate(("knowledge.graph",), ("KnowledgeGraph",), config=None),
            taxonomy_engine=_instantiate(("knowledge.taxonomy",), ("TaxonomyEngine",), config=None),
            ontology_engine=_instantiate(("knowledge.ontology",), ("OntologyEngine",), config=None),
            cluster_engine=_instantiate(("knowledge.clustering",), ("ClusterEngine",), config=None),
            insight_engine=_instantiate(("knowledge.insights",), ("InsightEngine",), config=None),
        )


# ==========================================================
# RUN RESULT OBJECTS
# ==========================================================

@dataclass(slots=True)
class DiscoveryPairResult:
    """
    Résultat complet pour une paire asset / timeframe.
    """

    context: DiscoveryContext
    dataset: Any = None

    discovery_output: Any = None
    validation_output: Any = None

    execution_results: tuple[Any, ...] = ()
    execution_report: Any = None

    portfolio_selection: Any = None
    portfolio_allocation: Any = None
    portfolio_report: Any = None

    corpus: Any = None
    rejected: Any = None
    report_bundle: Any = None

    export_paths: dict[str, str] = field(default_factory=dict)
    memory_snapshot: dict[str, Any] = field(default_factory=dict)
    knowledge_snapshot: dict[str, Any] = field(default_factory=dict)

    success: bool = True
    errors: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    started_at: datetime = field(default_factory=_utc_now)
    finished_at: datetime | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "execution_results", tuple(self.execution_results))
        object.__setattr__(self, "export_paths", dict(self.export_paths))
        object.__setattr__(self, "memory_snapshot", dict(self.memory_snapshot))
        object.__setattr__(self, "knowledge_snapshot", dict(self.knowledge_snapshot))
        object.__setattr__(self, "errors", tuple(str(e) for e in self.errors))
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "success", _coerce_bool(self.success, True))

    @property
    def pair_key(self) -> str:
        return self.context.key

    @property
    def asset(self) -> str:
        return self.context.asset

    @property
    def timeframe(self) -> str:
        return self.context.timeframe

    @property
    def execution_count(self) -> int:
        return len(self.execution_results)

    @property
    def selected_count(self) -> int:
        if self.corpus is None:
            return 0
        return _coerce_int(_extract_attr(self.corpus, "selected_count", default=0), 0)

    @property
    def corpus_size(self) -> int:
        if self.corpus is None:
            return 0
        return len(_extract_attr(self.corpus, "entries", default=())) if hasattr(self.corpus, "entries") else 0

    @property
    def best_subject_fingerprint(self) -> str | None:
        summary = _extract_attr(self.corpus, "summary", default=None)
        if summary is not None:
            value = _extract_attr(summary, "best_subject_fingerprint", default=None)
            if value:
                return str(value)
        return None

    def to_dict(self, *, summary_only: bool = False) -> dict[str, Any]:
        return {
            "context": self.context.to_dict(),
            "success": self.success,
            "errors": list(self.errors),
            "execution_count": self.execution_count,
            "selected_count": self.selected_count,
            "corpus_size": self.corpus_size,
            "best_subject_fingerprint": self.best_subject_fingerprint,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "metadata": dict(self.metadata),
            "export_paths": dict(self.export_paths),
            "memory_snapshot": dict(self.memory_snapshot),
            "knowledge_snapshot": dict(self.knowledge_snapshot),
            "discovery_output": None if summary_only else _safe_to_dict(self.discovery_output),
            "validation_output": None if summary_only else _safe_to_dict(self.validation_output),
            "execution_report": None if summary_only else _safe_to_dict(self.execution_report),
            "portfolio_selection": None if summary_only else _safe_to_dict(self.portfolio_selection),
            "portfolio_allocation": None if summary_only else _safe_to_dict(self.portfolio_allocation),
            "portfolio_report": None if summary_only else _safe_to_dict(self.portfolio_report),
            "corpus": None if summary_only else _safe_to_dict(self.corpus),
            "rejected": None if summary_only else _safe_to_dict(self.rejected),
            "report_bundle": None if summary_only else _safe_to_dict(self.report_bundle),
        }

    def __len__(self) -> int:
        return self.execution_count

    def __repr__(self) -> str:
        return (
            "DiscoveryPairResult("
            f"pair='{self.pair_key}', "
            f"success={self.success}, "
            f"executions={self.execution_count}, "
            f"selected={self.selected_count}"
            ")"
        )


@dataclass(slots=True)
class DiscoveryRunResult:
    """
    Résultat global du run Discovery.
    """

    run_id: str
    settings: DiscoverySettings
    pair_results: list[DiscoveryPairResult] = field(default_factory=list)

    corpus: Any = None
    rejected: Any = None

    report_bundles: list[Any] = field(default_factory=list)

    export_paths: dict[str, str] = field(default_factory=dict)
    memory_snapshot: dict[str, Any] = field(default_factory=dict)
    knowledge_snapshot: dict[str, Any] = field(default_factory=dict)

    errors: tuple[str, ...] = ()
    started_at: datetime = field(default_factory=_utc_now)
    finished_at: datetime | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "pair_results", list(self.pair_results))
        object.__setattr__(self, "report_bundles", list(self.report_bundles))
        object.__setattr__(self, "export_paths", dict(self.export_paths))
        object.__setattr__(self, "memory_snapshot", dict(self.memory_snapshot))
        object.__setattr__(self, "knowledge_snapshot", dict(self.knowledge_snapshot))
        object.__setattr__(self, "errors", tuple(str(e) for e in self.errors))
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "run_id", _normalize_text(self.run_id))
        object.__setattr__(self, "settings", self.settings)

    @property
    def pair_count(self) -> int:
        return len(self.pair_results)

    @property
    def success_count(self) -> int:
        return sum(1 for result in self.pair_results if result.success)

    @property
    def failure_count(self) -> int:
        return sum(1 for result in self.pair_results if not result.success)

    @property
    def total_corpus_entries(self) -> int:
        if self.corpus is None:
            return 0
        return len(_extract_attr(self.corpus, "entries", default=())) if hasattr(self.corpus, "entries") else 0

    @property
    def total_rejected_entries(self) -> int:
        if self.rejected is None:
            return 0
        return len(_extract_attr(self.rejected, "entries", default=())) if hasattr(self.rejected, "entries") else 0

    @property
    def summary(self) -> dict[str, Any]:
        corpus_summary = _safe_to_dict(_extract_attr(self.corpus, "summary", default=None))
        return {
            "run_id": self.run_id,
            "pair_count": self.pair_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "total_corpus_entries": self.total_corpus_entries,
            "total_rejected_entries": self.total_rejected_entries,
            "corpus_summary": corpus_summary,
            "errors": list(self.errors),
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
        }

    def to_dict(self, *, summary_only: bool = False) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "settings": self.settings.to_dict(),
            "pair_results": [] if summary_only else [result.to_dict(summary_only=summary_only) for result in self.pair_results],
            "corpus": None if summary_only else _safe_to_dict(self.corpus),
            "rejected": None if summary_only else _safe_to_dict(self.rejected),
            "report_bundles": [] if summary_only else [_safe_to_dict(item) for item in self.report_bundles],
            "export_paths": dict(self.export_paths),
            "memory_snapshot": dict(self.memory_snapshot),
            "knowledge_snapshot": dict(self.knowledge_snapshot),
            "errors": list(self.errors),
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "metadata": dict(self.metadata),
            "summary": self.summary,
        }

    def __len__(self) -> int:
        return len(self.pair_results)

    def __iter__(self):
        return iter(self.pair_results)

    def __repr__(self) -> str:
        return (
            "DiscoveryRunResult("
            f"run_id='{self.run_id}', "
            f"pairs={self.pair_count}, "
            f"success={self.success_count}, "
            f"failures={self.failure_count}"
            ")"
        )


def _safe_to_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            return value.to_dict()
        except Exception:
            return {"repr": repr(value)}
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, (list, tuple, set)):
        return {"items": [ _safe_to_dict(v) if hasattr(v, "to_dict") else v for v in value ]}
    return {"repr": repr(value)}

def _extract_candidates(output: Any) -> list[Any]:
    """
    Extrait une liste de candidats depuis la sortie du discovery engine.

    Supporte :
    - liste / tuple / set
    - objet avec attributs candidates, hypotheses, results, items, selected
    - mapping avec les mêmes clés
    - itérable générique
    - objet unique en dernier recours
    """
    if output is None:
        return []

    if isinstance(output, (list, tuple, set)):
        return list(output)

    if isinstance(output, Mapping):
        for key in (
            "candidates",
            "hypotheses",
            "results",
            "items",
            "selected",
            "discovered",
            "generated",
        ):
            value = output.get(key)
            if value is not None:
                if isinstance(value, (list, tuple, set)):
                    return list(value)
                return [value]
        return []

    for attr_name in (
        "candidates",
        "hypotheses",
        "results",
        "items",
        "selected",
        "discovered",
        "generated",
    ):
        value = getattr(output, attr_name, None)
        if value is not None:
            if isinstance(value, (list, tuple, set)):
                return list(value)
            if hasattr(value, "__iter__") and not isinstance(value, (str, bytes, Mapping)):
                try:
                    return list(value)
                except TypeError:
                    pass
            return [value]

    if hasattr(output, "__iter__") and not isinstance(output, (str, bytes, Mapping)):
        try:
            return list(output)
        except TypeError:
            pass

    return [output]

# ==========================================================
# ORCHESTRATOR
# ==========================================================

class DiscoveryOrchestrator:
    """
    Orchestrateur principal du pipeline Discovery.
    """

    def __init__(self, config: Any | None = None, *, components: DiscoveryComponents | None = None) -> None:
        self.config = config
        self.settings = DiscoverySettings.from_config(config) if config is not None else DiscoverySettings()
        self.components = components or DiscoveryComponents.create(config)

        run_name = self.settings.run_name or _utc_now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"discovery_{_slugify(run_name)}_{_utc_now().strftime('%H%M%S')}"
        self.run_root = self.settings.output_root / "discovery" / self.run_id

    # ==================================================
    # PUBLIC API
    # ==================================================

    def run(
        self,
        *,
        pairs: Sequence[Any] | None = None,
        assets: Sequence[str] | None = None,
        timeframes: Sequence[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> DiscoveryRunResult:
        targets = self.resolve_targets(pairs=pairs, assets=assets, timeframes=timeframes)
        if not targets:
            raise ValueError("No asset/timeframe pair provided to DiscoveryOrchestrator.run().")

        started_at = _utc_now()
        pair_results: list[DiscoveryPairResult] = []
        errors: list[str] = []
        report_bundles: list[Any] = []

        for index, target in enumerate(targets):
            if self.settings.max_pairs and index >= self.settings.max_pairs:
                break

            try:
                result = self.run_pair(target, index=index)
                pair_results.append(result)
                if result.report_bundle is not None:
                    report_bundles.append(result.report_bundle)
            except Exception as exc:
                if not self.settings.continue_on_error:
                    raise
                errors.append(f"{target.key}: {exc!r}")
                pair_results.append(
                    DiscoveryPairResult(
                        context=DiscoveryContext(
                            run_id=self.run_id,
                            target=target,
                            index=index,
                            metadata={"error": str(exc)},
                        ),
                        success=False,
                        errors=(repr(exc),),
                        metadata={"error": str(exc)},
                        finished_at=_utc_now(),
                    )
                )

        corpus, rejected = self._build_global_outputs(pair_results, metadata=metadata)
        memory_snapshot = self._build_memory_snapshot(pair_results, corpus=corpus, rejected=rejected, metadata=metadata)
        knowledge_snapshot = self._build_knowledge_snapshot(pair_results, corpus=corpus, rejected=rejected, metadata=metadata)

        result = DiscoveryRunResult(
            run_id=self.run_id,
            settings=self.settings,
            pair_results=pair_results,
            corpus=corpus,
            rejected=rejected,
            report_bundles=report_bundles,
            export_paths={},
            memory_snapshot=memory_snapshot,
            knowledge_snapshot=knowledge_snapshot,
            errors=tuple(errors),
            started_at=started_at,
            finished_at=_utc_now(),
            metadata=dict(metadata or {}),
        )

        if self.settings.export_run_summary:
            export_paths = self._export_run(result)
            result.export_paths.update(export_paths)

        return result

    def run_pair(
        self,
        target: Any,
        *,
        index: int = 0,
        dataset: Any | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> DiscoveryPairResult:
        target = _resolve_pair_any(target)
        context = DiscoveryContext(
            run_id=self.run_id,
            target=target,
            index=index,
            metadata=dict(metadata or {}),
        )

        started_at = _utc_now()
        errors: list[str] = []

        try:
            dataset = dataset if dataset is not None else self._load_dataset(target, context=context)
            discovery_output = self._discover(dataset, target=target, context=context)
            candidates = _extract_candidates(discovery_output)

            validation_output, validated_candidates, rejected_candidates = self._validate(
                candidates,
                dataset=dataset,
                target=target,
                context=context,
            )

            execution_results, execution_report = self._execute(
                validated_candidates,
                dataset=dataset,
                target=target,
                context=context,
            )

            portfolio_selection, portfolio_allocation, portfolio_report = self._build_portfolio(
                execution_results,
                target=target,
                context=context,
            )

            corpus = self._build_corpus(portfolio_report)
            rejected = self._build_rejected(portfolio_report, rejected_candidates=rejected_candidates)
            report_bundle = self._build_report_bundle(validation_output, execution_report, portfolio_report)

            memory_snapshot = self._update_memory(
                target=target,
                context=context,
                discovery_output=discovery_output,
                validation_output=validation_output,
                execution_report=execution_report,
                portfolio_report=portfolio_report,
                corpus=corpus,
                rejected=rejected,
            )
            knowledge_snapshot = self._update_knowledge(
                target=target,
                context=context,
                corpus=corpus,
                rejected=rejected,
            )

            result = DiscoveryPairResult(
                context=context,
                dataset=dataset,
                discovery_output=discovery_output,
                validation_output=validation_output,
                execution_results=tuple(execution_results),
                execution_report=execution_report,
                portfolio_selection=portfolio_selection,
                portfolio_allocation=portfolio_allocation,
                portfolio_report=portfolio_report,
                corpus=corpus,
                rejected=rejected,
                report_bundle=report_bundle,
                memory_snapshot=memory_snapshot,
                knowledge_snapshot=knowledge_snapshot,
                success=True,
                errors=tuple(errors),
                metadata=dict(metadata or {}),
                started_at=started_at,
                finished_at=_utc_now(),
            )

            if self.settings.export_pair_results:
                result.export_paths.update(
                    self._export_pair(result)
                )

            return result

        except Exception as exc:
            errors.append(repr(exc))
            if not self.settings.continue_on_error:
                raise

            return DiscoveryPairResult(
                context=context,
                dataset=dataset,
                success=False,
                errors=tuple(errors),
                metadata={"error": str(exc), **dict(metadata or {})},
                started_at=started_at,
                finished_at=_utc_now(),
            )

    # ==================================================
    # RESOLUTION
    # ==================================================

    def resolve_targets(
        self,
        *,
        pairs: Sequence[Any] | None = None,
        assets: Sequence[str] | None = None,
        timeframes: Sequence[str] | None = None,
    ) -> tuple[DiscoveryTarget, ...]:
        if pairs is not None and len(pairs) > 0:
            return tuple(_resolve_pair_any(pair) for pair in pairs if pair is not None)

        resolved_assets = tuple(
            _normalize_text(asset)
            for asset in _as_sequence(assets if assets is not None else self.settings.assets)
            if _normalize_text(asset)
        )
        resolved_timeframes = tuple(
            _normalize_text(tf)
            for tf in _as_sequence(timeframes if timeframes is not None else self.settings.timeframes)
            if _normalize_text(tf)
        )

        if self.settings.pairs:
            return tuple(self.settings.pairs)

        if resolved_assets and resolved_timeframes:
            return tuple(
                DiscoveryTarget(asset=asset, timeframe=timeframe)
                for asset in resolved_assets
                for timeframe in resolved_timeframes
            )

        if resolved_assets:
            return tuple(DiscoveryTarget(asset=asset, timeframe="unknown") for asset in resolved_assets)

        if resolved_timeframes:
            return tuple(DiscoveryTarget(asset="unknown", timeframe=timeframe) for timeframe in resolved_timeframes)

        return ()

    # ==================================================
    # STAGES
    # ==================================================

    def _load_dataset(self, target: DiscoveryTarget, *, context: DiscoveryContext) -> Any:
        from config.dataset import DatasetConfig
        from dataset.loader import DatasetLoader

        dataset_cfg = DatasetConfig(
            midas_root=r"D:/midas_v2/midasV3/src/data/compiled",
            asset=target.asset,
            asset_class=_resolve_asset_class(target.asset),
            timeframe=target.timeframe,
        )
        return DatasetLoader(dataset_cfg)

    def _discover(self, dataset: Any, *, target: DiscoveryTarget, context: DiscoveryContext) -> Any:
        from pathlib import Path
        from models.feature_registry import FeatureRegistry
        from discovery.family_manager import FamilyManager
        from discovery.generator import DiscoveryGenerator
        from discovery.explorer import Explorer

        # Charger le metadata.json de l'actif courant
        meta_path = Path(r"D:/midas_v2/midasV3/src/data/compiled")
        meta_path = meta_path / _resolve_asset_class(target.asset) / target.timeframe / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"metadata.json introuvable : {meta_path}")

        registry = FeatureRegistry(str(meta_path))
        family_manager = FamilyManager.from_config(self.config, registry)

        # Construire le generator
        generator = DiscoveryGenerator(self.config, registry)

        # Construire l'explorer avec le generator
        explorer = Explorer(
            registry=registry,
            config=self.config,
            generator=generator,
            search_config=self.settings.search_config,
        )

        # Lancer la recherche avec un seed_size raisonnable
        result = explorer.run(
            seed_size=5,
            max_iterations=getattr(self.settings.search_config, "max_depth", 3),
        )
        return result

    def _validate(
        self,
        candidates: Sequence[Any],
        *,
        dataset: Any,
        target: DiscoveryTarget,
        context: DiscoveryContext,
    ) -> tuple[Any, tuple[Any, ...], tuple[Any, ...]]:
        from validation.evaluator import ValidationEvaluator

        validator = ValidationEvaluator(
            config=self.config,
            dataset=dataset,
        )

        validation_output = _try_call(
            validator,
            ("validate", "run", "process", "score", "evaluate"),
            candidates=candidates,
            items=candidates,
            hypotheses=candidates,
            dataset=dataset,
            data=dataset,
            asset=target.asset,
            timeframe=target.timeframe,
            target=target,
            context=context,
            config=self.config,
            metadata=target.metadata,
        )
        if validation_output is None:
            validation_output = _try_call(
                validator,
                ("validate", "run", "process", "score", "evaluate"),
                candidates,
                dataset=dataset,
                target=target,
                context=context,
                config=self.config,
                metadata=target.metadata,
            )

        validated = _extract_items(
            validation_output,
            "validated_candidates",
            "accepted_candidates",
            "accepted",
            "selected",
            "candidates",
            "items",
            "results",
        )
        rejected = _extract_items(
            validation_output,
            "rejected_candidates",
            "rejected",
            "discarded",
            "failed",
        )

        if not validated:
            validated = list(candidates)

        return validation_output, tuple(validated), tuple(rejected)

    def _execute(
        self,
        validated_candidates: Sequence[Any],
        *,
        dataset: Any,
        target: DiscoveryTarget,
        context: DiscoveryContext,
    ) -> tuple[tuple[Any, ...], Any]:
        from execution.executor import ExecutionEngine

        engine = ExecutionEngine(config=self.config)
        if engine is None:
            raise RuntimeError("Execution engine is unavailable.")

        if hasattr(engine, "report") and hasattr(engine.report, "reset"):
            try:
                engine.report.reset()
            except Exception:
                pass

        if hasattr(engine, "execute_batch"):
            execution_results = _try_call(
                engine,
                ("execute_batch", "run_batch"),
                validated_candidates,
                dataset=dataset,
                data=dataset,
                target=target,
                context=context,
                config=self.settings.execution_config,
                metadata=target.metadata,
            )
            if execution_results is None:
                execution_results = _try_call(
                    engine,
                    ("execute_batch", "run_batch"),
                    validated_candidates,
                )
            execution_results = _extract_items(execution_results, "results", "executions")
        else:
            execution_results = []
            for candidate in validated_candidates:
                result = _try_call(
                    engine,
                    ("execute", "run", "replay"),
                    candidate,
                    dataset=dataset,
                    data=dataset,
                    target=target,
                    context=context,
                    config=self.settings.execution_config,
                    metadata=target.metadata,
                )
                if result is None:
                    result = _try_call(
                        engine,
                        ("execute", "run", "replay"),
                        candidate,
                    )
                if result is not None:
                    execution_results.append(result)

        execution_report = _extract_attr(engine, "report", default=None)
        if execution_report is None:
            execution_report = _try_call(engine, ("report",), default=None)

        return tuple(execution_results), execution_report

    def _build_portfolio(
        self,
        execution_results: Sequence[Any],
        *,
        target: DiscoveryTarget,
        context: DiscoveryContext,
    ) -> tuple[Any, Any, Any]:
        from portfolio.selector import PortfolioSelector
        from portfolio.correlation import PortfolioCorrelationAnalyzer
        from portfolio.diversification import DiversificationEngine
        from portfolio.risk import PortfolioRiskModel
        from portfolio.capital import CapitalManager
        from portfolio.allocator import PortfolioAllocator
        from portfolio.portfolio_report import PortfolioReporter
        from portfolio.optimizer import PortfolioOptimizer

        selector = PortfolioSelector(config=self.config)
        correlation = PortfolioCorrelationAnalyzer(config=self.config)
        diversification = DiversificationEngine(config=self.config)
        risk_model = PortfolioRiskModel(config=self.config)
        capital_manager = CapitalManager(config=self.config)
        allocator = PortfolioAllocator(config=self.config)
        reporter = PortfolioReporter(config=self.config)
        optimizer = PortfolioOptimizer(config=self.config)

        if selector is None or allocator is None or reporter is None:
            raise RuntimeError("Portfolio stack is unavailable.")

        selection = _try_call(
            selector,
            ("select",),
            execution_results,
            limit=None,
            metadata={"asset": target.asset, "timeframe": target.timeframe, "run_id": self.run_id},
        )
        if selection is None:
            selection = _try_call(
                selector,
                ("select",),
                execution_results,
            )

        selected_results = _extract_items(selection, "results", "selected")
        if not selected_results:
            selected_results = list(execution_results)

        corr_matrix = None
        if correlation is not None:
            corr_matrix = _try_call(correlation, ("correlate", "matrix", "similarity_matrix"), selected_results, metadata={"asset": target.asset, "timeframe": target.timeframe})
            if corr_matrix is None:
                corr_matrix = _try_call(correlation, ("correlate", "matrix", "similarity_matrix"), selected_results)

        diversification_assessment = None
        if diversification is not None:
            diversification_assessment = _try_call(
                diversification,
                ("assess", "score"),
                selected_results,
                correlation=corr_matrix,
                metadata={"asset": target.asset, "timeframe": target.timeframe},
            )
            if diversification_assessment is None:
                diversification_assessment = _try_call(
                    diversification,
                    ("assess", "score"),
                    selected_results,
                    correlation=corr_matrix,
                )

        risk_assessment = None
        if risk_model is not None:
            risk_assessment = _try_call(
                risk_model,
                ("assess", "score"),
                selected_results,
                correlation=corr_matrix,
                diversification=diversification_assessment,
                metadata={"asset": target.asset, "timeframe": target.timeframe},
            )
            if risk_assessment is None:
                risk_assessment = _try_call(
                    risk_model,
                    ("assess", "score"),
                    selected_results,
                    correlation=corr_matrix,
                    diversification=diversification_assessment,
                )

        if optimizer is not None:
            optimization = _try_call(
                optimizer,
                ("optimize", "search", "tune"),
                selected_results,
                correlation=corr_matrix,
                diversification=diversification_assessment,
                risk=risk_assessment,
                total_capital=getattr(self.settings.execution_config, "max_open_positions", 1),
                metadata={"asset": target.asset, "timeframe": target.timeframe},
            )
            best_allocation = _extract_attr(optimization, "best_allocation", default=None)
            if best_allocation is not None:
                allocation = best_allocation
            else:
                allocation = None
        else:
            allocation = None

        if allocation is None:
            allocation = _try_call(
                allocator,
                ("allocate", "plan"),
                selected_results,
                weights=None,
                total_capital=getattr(self.settings.execution_config, "max_open_positions", 1),
                risk=risk_assessment,
                diversification=diversification_assessment,
                correlation=corr_matrix,
                metadata={"asset": target.asset, "timeframe": target.timeframe},
            )
            if allocation is None:
                allocation = _try_call(
                    allocator,
                    ("allocate", "plan"),
                    selected_results,
                )

        portfolio_report = _try_call(
            reporter,
            ("build", "summarize"),
            allocation=allocation,
            selection=selection,
            risk=risk_assessment,
            diversification=diversification_assessment,
            correlation=corr_matrix,
            capital_plan=_extract_attr(allocation, "capital_plan", default=None),
            rejected=_extract_attr(selection, "rejected", default=()),
            name=f"{target.slug}",
            metadata={"asset": target.asset, "timeframe": target.timeframe, "run_id": self.run_id},
        )
        if portfolio_report is None:
            portfolio_report = _try_call(
                reporter,
                ("build", "summarize"),
                allocation=allocation,
                selection=selection,
            )

        return selection, allocation, portfolio_report

    # ==================================================
    # BUILDERS
    # ==================================================

    def _build_corpus(self, portfolio_report: Any) -> Any:
        if self.components.corpus_builder is not None:
            corpus = _try_call(
                self.components.corpus_builder,
                ("from_portfolio_report",),
                portfolio_report,
                include_rejected=True,
                source_kind="discovery_portfolio",
                metadata={"run_id": self.run_id},
            )
            if corpus is not None:
                return corpus

        corpus_cls = self.components.corpus_cls
        if corpus_cls is None:
            return None

        entries = []
        for entry in _extract_items(portfolio_report, "entries"):
            result = _extract_attr(entry, "result", default=None)
            if result is None:
                continue
            corpus_entry = _try_call(
                corpus_cls,
                ("from_result",),
                result,
                weight=_coerce_float(_extract_attr(entry, "weight", "target_weight", default=0.0), 0.0),
                capital=_coerce_float(_extract_attr(entry, "capital", default=0.0), 0.0),
                score=_coerce_float(_extract_attr(entry, "score", default=0.0), 0.0),
                source_kind="discovery_portfolio",
                rank=_coerce_int(_extract_attr(entry, "rank", default=0), 0),
                metadata={"run_id": self.run_id},
            )
            if corpus_entry is not None:
                entries.append(corpus_entry)

        return corpus_cls(entries=entries, metadata={"run_id": self.run_id}, rejected=[])

    def _build_rejected(self, portfolio_report: Any, *, rejected_candidates: Sequence[Any]) -> Any:
        if self.components.rejected_builder is not None:
            rejected = _try_call(
                self.components.rejected_builder,
                ("from_report",),
                portfolio_report,
                metadata={"run_id": self.run_id},
            )
            if rejected is not None:
                return rejected

        rejected_cls = self.components.rejected_cls
        if rejected_cls is None:
            return None

        return rejected_cls(entries=[], metadata={"run_id": self.run_id})

    def _build_report_bundle(self, validation_output: Any, execution_report: Any, portfolio_report: Any) -> Any:
        if self.components.report_bundle_builder is None:
            return None

        return _try_call(
            self.components.report_bundle_builder,
            ("build", "from_reports"),
            validation=validation_output,
            execution=execution_report,
            portfolio=portfolio_report,
            metadata={"run_id": self.run_id},
        )

    # ==================================================
    # MEMORY / KNOWLEDGE
    # ==================================================

    def _update_memory(
        self,
        *,
        target: DiscoveryTarget,
        context: DiscoveryContext,
        discovery_output: Any,
        validation_output: Any,
        execution_report: Any,
        portfolio_report: Any,
        corpus: Any,
        rejected: Any,
    ) -> dict[str, Any]:
        snapshot: dict[str, Any] = {}

        if not self.settings.build_memory:
            return snapshot

        # Search history
        if self.components.search_history is not None:
            try:
                entry = _try_call(
                    self.components.search_history,
                    ("record", "add", "append"),
                    query=target.key,
                    phase="discovery",
                    objective="build_corpus",
                    seed=target.asset,
                    features=(),
                    families=(target.asset,),
                    regions=(target.timeframe,),
                    parameters={"run_id": self.run_id},
                    result_count=_coerce_int(_extract_attr(corpus, "selected_count", default=0), 0),
                    accepted_count=_coerce_int(_extract_attr(corpus, "selected_count", default=0), 0),
                    rejected_count=_coerce_int(_extract_attr(rejected, "summary.entry_count", default=0), 0),
                    useful=bool(_extract_attr(corpus, "selected_count", default=0)),
                    success=bool(_extract_attr(corpus, "selected_count", default=0)),
                    score=_coerce_float(_extract_attr(portfolio_report, "average_score", default=0.0), 0.0),
                    reason="pair_completed",
                    notes=(target.key,),
                    metadata={"asset": target.asset, "timeframe": target.timeframe, "run_id": self.run_id},
                )
                snapshot["search_history"] = _safe_to_dict(entry)
            except Exception:
                snapshot["search_history"] = _safe_to_dict(self.components.search_history)

        # Explored regions
        if self.components.explored_regions is not None:
            try:
                region = _try_call(
                    self.components.explored_regions,
                    ("register", "add"),
                    region_key=target.key,
                    phase="discovery",
                    family=target.asset,
                    feature=target.timeframe,
                    depth=0,
                    size=_coerce_int(_extract_attr(corpus, "selected_count", default=0), 0),
                    attempts=1,
                    score=_coerce_float(_extract_attr(portfolio_report, "average_score", default=0.0), 0.0),
                    metadata={"run_id": self.run_id},
                )
                snapshot["explored_regions"] = _safe_to_dict(region)
            except Exception:
                snapshot["explored_regions"] = _safe_to_dict(self.components.explored_regions)

        # Successful / failed regions
        if self.components.successful_regions is not None:
            try:
                region = _try_call(
                    self.components.successful_regions,
                    ("register", "add"),
                    region_key=target.key,
                    phase="discovery",
                    family=target.asset,
                    feature=target.timeframe,
                    hits=1,
                    success_count=1 if _coerce_int(_extract_attr(corpus, "selected_count", default=0), 0) > 0 else 0,
                    score=_coerce_float(_extract_attr(portfolio_report, "average_score", default=0.0), 0.0),
                    yield_rate=1.0 if _coerce_int(_extract_attr(corpus, "selected_count", default=0), 0) > 0 else 0.0,
                    metadata={"run_id": self.run_id},
                )
                snapshot["successful_regions"] = _safe_to_dict(region)
            except Exception:
                snapshot["successful_regions"] = _safe_to_dict(self.components.successful_regions)

        if self.components.failed_regions is not None:
            try:
                region = _try_call(
                    self.components.failed_regions,
                    ("register", "add"),
                    region_key=target.key,
                    phase="discovery",
                    family=target.asset,
                    feature=target.timeframe,
                    attempts=1,
                    failure_count=0 if _coerce_int(_extract_attr(corpus, "selected_count", default=0), 0) > 0 else 1,
                    score=_coerce_float(_extract_attr(portfolio_report, "average_score", default=0.0), 0.0),
                    reason="no_result" if _coerce_int(_extract_attr(corpus, "selected_count", default=0), 0) == 0 else "",
                    metadata={"run_id": self.run_id},
                )
                snapshot["failed_regions"] = _safe_to_dict(region)
            except Exception:
                snapshot["failed_regions"] = _safe_to_dict(self.components.failed_regions)

        # Feature / family histories
        if self.components.feature_history is not None:
            try:
                touched = 0
                for entry in _extract_items(corpus, "entries"):
                    result = _extract_attr(entry, "result", default=None)
                    if result is None:
                        continue
                    feature_key = _extract_feature_key(result)
                    self.components.feature_history.register(
                        feature_key,
                        family=_extract_family_key(result),
                        phase="execution",
                        success=bool(_extract_attr(entry, "capital", default=0.0)),
                        score=_coerce_float(_extract_attr(entry, "score", default=0.0), 0.0),
                        metadata={"run_id": self.run_id},
                    )
                    touched += 1
                snapshot["feature_history"] = {"touched": touched, "summary": _safe_to_dict(self.components.feature_history.summary)}
            except Exception:
                snapshot["feature_history"] = _safe_to_dict(self.components.feature_history)

        if self.components.family_history is not None:
            try:
                touched = 0
                for entry in _extract_items(corpus, "entries"):
                    result = _extract_attr(entry, "result", default=None)
                    if result is None:
                        continue
                    self.components.family_history.register(
                        _extract_family_key(result),
                        success=bool(_extract_attr(entry, "capital", default=0.0)),
                        score=_coerce_float(_extract_attr(entry, "score", default=0.0), 0.0),
                        metadata={"run_id": self.run_id},
                    )
                    touched += 1
                snapshot["family_history"] = {"touched": touched, "summary": _safe_to_dict(self.components.family_history.summary)}
            except Exception:
                snapshot["family_history"] = _safe_to_dict(self.components.family_history)

        # Corpus history
        if self.components.corpus_history is not None:
            try:
                version = _coerce_int(len(self.components.corpus_history) + 1, 1) if hasattr(self.components.corpus_history, "__len__") else 1
                version_obj = self.components.corpus_history.register(
                    corpus_key=target.key,
                    version=version,
                    entry_count=_coerce_int(_extract_attr(corpus, "selected_count", default=0), 0),
                    selected_count=_coerce_int(_extract_attr(corpus, "selected_count", default=0), 0),
                    total_capital=_coerce_float(_extract_attr(corpus, "total_capital", default=0.0), 0.0),
                    total_weight=_coerce_float(_extract_attr(corpus, "total_weight", default=0.0), 0.0),
                    total_pnl=_coerce_float(_extract_attr(_extract_attr(corpus, "summary", default=None), "total_pnl", default=0.0), 0.0),
                    fingerprints=tuple(
                        _extract_attr(entry, "subject_fingerprint", default="")
                        for entry in _extract_items(corpus, "entries")
                    ),
                    summary=_safe_to_dict(_extract_attr(corpus, "summary", default=None)),
                    metadata={"run_id": self.run_id},
                )
                snapshot["corpus_history"] = _safe_to_dict(version_obj)
            except Exception:
                snapshot["corpus_history"] = _safe_to_dict(self.components.corpus_history)

        # Learning
        if self.components.learning_engine is not None:
            try:
                state = self.components.learning_engine.learn(
                    search_history=self.components.search_history,
                    explored_regions=self.components.explored_regions,
                    successful_regions=self.components.successful_regions,
                    failed_regions=self.components.failed_regions,
                    feature_history=self.components.feature_history,
                    family_history=self.components.family_history,
                    corpus_history=self.components.corpus_history,
                    metadata={"run_id": self.run_id, "asset": target.asset, "timeframe": target.timeframe},
                )
                snapshot["learning"] = _safe_to_dict(state)
            except Exception:
                snapshot["learning"] = _safe_to_dict(self.components.learning_engine)

        return snapshot

    def _update_knowledge(
        self,
        *,
        target: DiscoveryTarget,
        context: DiscoveryContext,
        corpus: Any,
        rejected: Any,
    ) -> dict[str, Any]:
        snapshot: dict[str, Any] = {}

        if not self.settings.build_knowledge:
            return snapshot

        entries = list(_extract_items(corpus, "entries"))
        if not entries:
            return snapshot

        # Taxonomy / ontology / fingerprints / graph / clusters / insights
        if self.components.taxonomy_engine is not None:
            try:
                taxonomy = self.components.taxonomy_engine.classify_many(entries)
                snapshot["taxonomy"] = [item.to_dict() for item in taxonomy]
            except Exception:
                snapshot["taxonomy"] = _safe_to_dict(self.components.taxonomy_engine)

        if self.components.fingerprint_registry is not None:
            try:
                fingerprints = []
                for entry in entries:
                    fp = self.components.fingerprint_registry.build(
                        entry,
                        kind="corpus_entry",
                        label=_extract_profile_key(entry),
                        metadata={"asset": target.asset, "timeframe": target.timeframe, "run_id": self.run_id},
                    )
                    fingerprints.append(fp.to_dict())
                snapshot["fingerprints"] = fingerprints
            except Exception:
                snapshot["fingerprints"] = _safe_to_dict(self.components.fingerprint_registry)

        if self.components.knowledge_graph is not None:
            try:
                graph = self.components.knowledge_graph.build_from_objects(
                    entries,
                    metadata={"asset": target.asset, "timeframe": target.timeframe, "run_id": self.run_id},
                )
                snapshot["graph"] = graph.to_dict()
            except Exception:
                snapshot["graph"] = _safe_to_dict(self.components.knowledge_graph)

        if self.components.cluster_engine is not None:
            try:
                clusters = self.components.cluster_engine.cluster(
                    entries,
                    metadata={"asset": target.asset, "timeframe": target.timeframe, "run_id": self.run_id},
                )
                snapshot["clusters"] = [cluster.to_dict() for cluster in clusters]
            except Exception:
                snapshot["clusters"] = _safe_to_dict(self.components.cluster_engine)

        if self.components.insight_engine is not None:
            try:
                graph_obj = self.components.knowledge_graph if self.components.knowledge_graph is not None else None
                clusters_obj = None
                if "clusters" in snapshot:
                    clusters_obj = tuple(snapshot["clusters"])
                insights = self.components.insight_engine.analyze(
                    entries,
                    graph=graph_obj,
                    clusters=clusters_obj,
                    metadata={"asset": target.asset, "timeframe": target.timeframe, "run_id": self.run_id},
                )
                snapshot["insights"] = insights.to_dict()
            except Exception:
                snapshot["insights"] = _safe_to_dict(self.components.insight_engine)

        if self.components.ontology_engine is not None:
            try:
                snapshot["ontology"] = self.components.ontology_engine.to_dict()
            except Exception:
                snapshot["ontology"] = _safe_to_dict(self.components.ontology_engine)

        return snapshot

    # ==================================================
    # EXPORTS
    # ==================================================

    def _build_memory_snapshot(
        self,
        pair_results: list[DiscoveryPairResult],
        *,
        corpus: Any,
        rejected: Any,
        metadata: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        snapshot: dict[str, Any] = {"pairs": []}
        for pr in pair_results:
            ms = pr.memory_snapshot
            if ms:
                snapshot["pairs"].append({"pair": pr.pair_key, **ms})
        return snapshot

    def _build_knowledge_snapshot(
        self,
        pair_results: list[DiscoveryPairResult],
        *,
        corpus: Any,
        rejected: Any,
        metadata: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        snapshot: dict[str, Any] = {"pairs": []}
        for pr in pair_results:
            ks = pr.knowledge_snapshot
            if ks:
                snapshot["pairs"].append({"pair": pr.pair_key, **ks})
        return snapshot
    
    def _export_pair(self, result: DiscoveryPairResult) -> dict[str, str]:
        export_paths: dict[str, str] = {}
        pair_dir = self.run_root / result.context.target.slug
        pair_dir.mkdir(parents=True, exist_ok=True)
        stem = result.context.target.slug

        summary_path = pair_dir / f"{stem}_summary.json"
        if self.components.json_exporter is not None:
            try:
                export_paths["summary_json"] = str(
                    self.components.json_exporter.export(result.to_dict(summary_only=True), summary_path)
                )
            except Exception:
                pass

        corpus = result.corpus
        rejected = result.rejected
        report_bundle = result.report_bundle

        if self.components.json_exporter is not None:
            try:
                if corpus is not None:
                    path = pair_dir / f"{stem}_corpus.json"
                    export_paths["corpus_json"] = str(self.components.json_exporter.export_corpus(corpus, path))
                if rejected is not None:
                    path = pair_dir / f"{stem}_rejected.json"
                    export_paths["rejected_json"] = str(self.components.json_exporter.export_rejected(rejected, path))
                if report_bundle is not None:
                    path = pair_dir / f"{stem}_reports.json"
                    export_paths["reports_json"] = str(self.components.json_exporter.export_reports(report_bundle, path))
            except Exception:
                pass

        if self.components.csv_exporter is not None:
            try:
                if corpus is not None:
                    path = pair_dir / f"{stem}_corpus.csv"
                    export_paths["corpus_csv"] = str(self.components.csv_exporter.export_corpus(corpus, path))
                if rejected is not None:
                    path = pair_dir / f"{stem}_rejected.csv"
                    export_paths["rejected_csv"] = str(self.components.csv_exporter.export_rejected(rejected, path))
                if report_bundle is not None:
                    path = pair_dir / f"{stem}_reports.csv"
                    export_paths["reports_csv"] = str(self.components.csv_exporter.export_reports(report_bundle, path))
            except Exception:
                pass

        if self.components.parquet_exporter is not None:
            try:
                if corpus is not None:
                    path = pair_dir / f"{stem}_corpus.parquet"
                    export_paths["corpus_parquet"] = str(self.components.parquet_exporter.export_corpus(corpus, path))
                if rejected is not None:
                    path = pair_dir / f"{stem}_rejected.parquet"
                    export_paths["rejected_parquet"] = str(self.components.parquet_exporter.export_rejected(rejected, path))
                if report_bundle is not None:
                    path = pair_dir / f"{stem}_reports.parquet"
                    export_paths["reports_parquet"] = str(self.components.parquet_exporter.export_reports(report_bundle, path))
            except Exception:
                pass

        if self.components.archive_exporter is not None:
            try:
                archive_path = pair_dir / f"{stem}.zip"
                export_paths["archive"] = str(
                    self.components.archive_exporter.export(
                        corpus=corpus,
                        rejected=rejected,
                        reports=report_bundle,
                        path=archive_path,
                        stem=stem,
                        metadata={"asset": result.asset, "timeframe": result.timeframe, "run_id": self.run_id},
                    )
                )
            except Exception:
                pass

        return export_paths

    def _export_run(self, result: DiscoveryRunResult) -> dict[str, str]:
        export_paths: dict[str, str] = {}
        run_dir = self.run_root
        run_dir.mkdir(parents=True, exist_ok=True)

        summary_path = run_dir / "run_summary.json"
        if self.components.json_exporter is not None:
            try:
                export_paths["run_summary_json"] = str(
                    self.components.json_exporter.export(result.to_dict(summary_only=True), summary_path)
                )
            except Exception:
                pass

        if result.corpus is not None and self.components.json_exporter is not None:
            try:
                export_paths["run_corpus_json"] = str(
                    self.components.json_exporter.export_corpus(result.corpus, run_dir / "run_corpus.json")
                )
            except Exception:
                pass

        if result.rejected is not None and self.components.json_exporter is not None:
            try:
                export_paths["run_rejected_json"] = str(
                    self.components.json_exporter.export_rejected(result.rejected, run_dir / "run_rejected.json")
                )
            except Exception:
                pass

        if result.corpus is not None and self.components.csv_exporter is not None:
            try:
                export_paths["run_corpus_csv"] = str(
                    self.components.csv_exporter.export_corpus(result.corpus, run_dir / "run_corpus.csv")
                )
            except Exception:
                pass

        if result.rejected is not None and self.components.csv_exporter is not None:
            try:
                export_paths["run_rejected_csv"] = str(
                    self.components.csv_exporter.export_rejected(result.rejected, run_dir / "run_rejected.csv")
                )
            except Exception:
                pass

        if result.corpus is not None and self.components.archive_exporter is not None:
            try:
                export_paths["run_archive"] = str(
                    self.components.archive_exporter.export(
                        corpus=result.corpus,
                        rejected=result.rejected,
                        reports=None,
                        path=run_dir / "run_archive.zip",
                        stem="run",
                        metadata={"run_id": self.run_id},
                    )
                )
            except Exception:
                pass

        return export_paths

    # ==================================================
    # AGGREGATION
    # ==================================================

    def _build_global_outputs(
        self,
        pair_results: Sequence[DiscoveryPairResult],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> tuple[Any, Any]:
        corpus_cls = self.components.corpus_cls
        rejected_cls = self.components.rejected_cls

        entries = []
        rejected_entries = []

        for pair in pair_results:
            if pair.corpus is not None and hasattr(pair.corpus, "entries"):
                entries.extend(list(pair.corpus.entries))
            if pair.rejected is not None and hasattr(pair.rejected, "entries"):
                rejected_entries.extend(list(pair.rejected.entries))

        if corpus_cls is not None:
            try:
                corpus = corpus_cls(
                    entries=entries,
                    metadata={"run_id": self.run_id, **dict(metadata or {})},
                    rejected=[item.to_dict() if hasattr(item, "to_dict") else _safe_to_dict(item) for item in rejected_entries],
                )
            except Exception:
                corpus = None
        else:
            corpus = None

        if rejected_cls is not None:
            try:
                rejected = rejected_cls(
                    entries=rejected_entries,
                    metadata={"run_id": self.run_id, **dict(metadata or {})},
                )
            except Exception:
                rejected = None
        else:
            rejected = None

        return corpus, rejected

    # ==================================================
    # SPAWN HELPERS
    # ==================================================

    def _spawn_pair_component(self, component: Any, config: Any | None) -> Any | None:
        if component is None:
            return None

        if inspect.isclass(component):
            return component

        if inspect.ismodule(component):
            return component

        cls = type(component)
        if cls.__name__ == "module":
            return component

        for kwargs in ({"config": config}, {"settings": config}, {}):
            try:
                return cls(**kwargs)
            except Exception:
                continue

        return component


# ==========================================================
# CONVENIENCE API
# ==========================================================

def build_orchestrator(config: Any | None = None) -> DiscoveryOrchestrator:
    return DiscoveryOrchestrator(config=config)


def main(
    config: Any | None = None,
    *,
    pairs: Sequence[Any] | None = None,
    assets: Sequence[str] | None = None,
    timeframes: Sequence[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> DiscoveryRunResult:
    orchestrator = DiscoveryOrchestrator(config=config)
    return orchestrator.run(
        pairs=pairs,
        assets=assets,
        timeframes=timeframes,
        metadata=metadata,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EINHERJAR Discovery — recherche d'Einhers sur l'univers MIDAS")
    parser.add_argument("--asset", type=str, default="", help="Actif unique (ex: XAUUSD)")
    parser.add_argument("--asset-class", type=str, default="", help="Classe d'actif (ex: forex, crypto, commodities)")
    parser.add_argument("--timeframe", type=str, default="15m", help="Timeframe (ex: 5m, 15m, 1h, 4h, 1d)")
    parser.add_argument("--debug", action="store_true", help="Logs détaillés")
    args = parser.parse_args()

    import logging
    if args.debug:
        logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)-5s] %(message)s")
    else:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)-5s] %(message)s")

    from config.config import Config
    from config.dataset import DatasetConfig

    config = Config()
    config.dataset = DatasetConfig(
        midas_root=r"D:/midas_v2/midasV3/src/data/compiled",
        asset=args.asset,
        asset_class=args.asset_class,
        timeframe=args.timeframe,
    )

    assets_v1_path = Path(r"D:/midas_v2/einherjar/config/assets_v1.json")
    assets_cfg = json.loads(assets_v1_path.read_text(encoding="utf-8"))
    assets_list = assets_cfg.get("assets", [])

    if args.asset:
        # Trouver la classe d'actif si non fournie
        asset_class = args.asset_class
        if not asset_class:
            for entry in assets_list:
                if entry["asset"] == args.asset:
                    asset_class = entry["class"]
                    break
        if not asset_class:
            raise ValueError(f"Actif {args.asset} non trouvé dans assets_v1.json — fournissez --asset-class")
        config.dataset = DatasetConfig(
                    midas_root=r"D:/midas_v2/midasV3/src/data/compiled",
                    asset=args.asset,
                    asset_class=asset_class,
                    timeframe=args.timeframe,
                )
        result = main(config, assets=[args.asset], timeframes=[args.timeframe])
    else:
        # Tout l'univers
        all_assets = [entry["asset"] for entry in assets_list]
        all_classes = {entry["asset"]: entry["class"] for entry in assets_list}
        # On lance asset par asset avec la bonne classe
        result = main(config, assets=all_assets, timeframes=[args.timeframe])

    print("=" * 60)
    print(f"Run ID : {result.run_id}")
    print(f"Pairs  : {result.pair_count}")
    print(f"Success: {result.success_count}")
    print(f"Failures: {result.failure_count}")
    for pr in result.pair_results:
        status = "OK" if pr.success else "FAIL"
        print(f"  [{status}] {pr.pair_key}")
