"""
==========================================================
Engine
==========================================================

Orchestrateur **per-pair** du pipeline de découverte.

L'Engine est responsable de la séquence des phases pour
**une** paire asset / timeframe reçue en argument :

    1. charger le dataset
    2. vérifier le contrat de données
    3. lancer Discovery
    4. lancer Validation
    5. lancer Execution
    6. construire les Einhers
    7. construire le Portfolio
    8. mettre à jour Memory (par-paire)
    9. mettre à jour Knowledge (par-paire)
   10. produire le DiscoveryPairResult

L'Engine NE CONNAÎT PAS :

- le ``run_id`` (responsabilité du runner) ;
- le ``output_root`` (responsabilité de l'exporter) ;
- la liste des autres paires (responsabilité du runner).

L'Engine agit comme une boîte noire qui reçoit
``(target, dataset)`` et renvoie un ``DiscoveryPairResult``.
C'est le point d'entrée Discovery (typiquement
``core.runner.DiscoveryOrchestrator``) qui :

- résout la liste des paires ;
- crée un Engine par run ;
- appelle ``engine.run_pair(target)`` pour chaque paire ;
- délègue l'export à ``core.exporter.PairExporter`` ;
- remonte les erreurs graves.

Chaque étape est strictement obligatoire. La violation d'un
contrat entre phases lève une exception explicite ; aucun
fallback silencieux n'est appliqué.

Un EngineContext est créé pour CHAQUE paire, puis détruit
à la fin. Aucun contexte n'est partagé entre paires.
"""

from __future__ import annotations

import logging
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any
from typing import Mapping
from typing import Sequence

import numpy as np

from config.config import Config

from dataset.inspector import DatasetInspector
from dataset.loader import DatasetLoader
from dataset.loader import DatasetSplit
from dataset.statistics import DatasetStatistics
from dataset.validator import DatasetValidator

from discovery.explorer import Explorer
from discovery.generator import DiscoveryGenerator
from execution.execution_report import ExecutionReport
from execution.execution_report import ExecutionResult
from execution.executor import ExecutionEngine

from models.einher import Einher
from models.feature_registry import FeatureRegistry
from models.hypothesis import Hypothesis
from models.validated_candidate import ValidatedCandidate

from portfolio.allocator import PortfolioAllocation
from portfolio.allocator import PortfolioAllocator
from portfolio.correlation import PortfolioCorrelationAnalyzer
from portfolio.diversification import DiversificationEngine
from portfolio.optimizer import PortfolioOptimizer
from portfolio.portfolio_report import PortfolioReport
from portfolio.portfolio_report import PortfolioReporter
from portfolio.risk import PortfolioRiskModel
from portfolio.selector import PortfolioSelection
from portfolio.selector import PortfolioSelector

from validation.evaluator import ValidationEvaluator

from .assets import resolve_asset_class
from .context import EngineContext
from .exceptions import DatasetContractError
from .exceptions import DiscoveryContractError
from .exceptions import ExecutionContractError
from .exceptions import PhaseContractError
from .state import EngineState
from .types import DiscoveryPairResult
from .types import DiscoveryTarget

logger = logging.getLogger("einherjar.engine")


# ==========================================================
# HELPERS
# ==========================================================

def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _read_run_id(context: EngineContext) -> str:
    """
    Lit le run_id depuis les métadonnées du target.

    L'Engine ne possède pas de run_id. Si le caller en
    a placé un dans ``target.metadata["run_id"]``, on le
    récupère ; sinon on renvoie "" — c'est juste
    informationnel (les fichiers sont gérés par le
    PairExporter, run-level).
    """

    target = getattr(context, "target", None)
    if target is None:
        return ""
    md = getattr(target, "metadata", None)
    if not isinstance(md, dict):
        return ""
    return str(md.get("run_id", "") or "").strip()


# ==========================================================
# ENGINE
# ==========================================================

class Engine:
    """
    Orchestrateur du pipeline de découverte.

    Une instance d'Engine :

    - détient la configuration globale du run,
    - est responsable de la résolution des paires à traiter,
    - délègue l'orchestration intra-paire à run_pair(),
    - ne contient aucune logique métier.
    """

    # ==================================================
    # INITIALIZATION
    # ==================================================

    def __init__(self, config: Config) -> None:
        self._config = config

    # ==================================================
    # PROPERTIES
    # ==================================================

    @property
    def config(self) -> Config:
        return self._config

    # ==================================================
    # PUBLIC ENTRY POINTS
    # ==================================================

    def run_pair(
        self,
        target: Any,
        *,
        index: int = 0,
        dataset: DatasetLoader | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> DiscoveryPairResult:
        """
        Exécute le pipeline per-pair pour une cible.

        Cette méthode est le SEUL point d'entrée de
        l'orchestration per-pair. Elle :

        1. construit un EngineContext par paire,
        2. exécute chaque phase en séquence (dataset,
           discovery, validation, execution, einhers,
           portfolio, memory, knowledge),
        3. met à jour l'EngineState à chaque transition,
        4. produit un DiscoveryPairResult.

        L'Engine NE FAIT PAS D'EXPORT. C'est le
        ``PairExporter`` (run-level) qui écrit les
        fichiers à partir du ``DiscoveryPairResult``
        retourné.

        Toute violation de contrat interrompt l'exécution.
        """

        pair_target = self._resolve_target(target)

        state = EngineState()
        state.start()
        state.begin_phase("initialization")
        state.complete_phase("initialization")

        # 1) Construction du contexte par paire
        try:
            context = self._build_pair_context(
                pair_target, state=state, dataset=dataset,
            )
        except Exception as exc:
            state.fail(repr(exc))
            logger.error(
                "[%s] context build failed: %r", pair_target.key, exc,
            )
            return self._make_failure_result(
                pair_target, index, state, metadata, exc,
            )

        # 2) Validation du contrat dataset
        state.begin_phase("dataset")
        try:
            self._verify_dataset(context)
        except Exception as exc:
            state.fail_phase("dataset", repr(exc))
            logger.error("[%s] dataset phase failed: %r", pair_target.key, exc)
            return self._make_failure_result(
                pair_target, index, state, metadata, exc,
            )
        state.complete_phase("dataset")

        state.begin_phase("contract")
        state.complete_phase("contract")

        # 3) Discovery
        state.begin_phase("discovery")
        try:
            candidates = self._discover(context)
        except Exception as exc:
            state.fail_phase("discovery", repr(exc))
            logger.error("[%s] discovery phase failed: %r", pair_target.key, exc)
            return self._make_failure_result(
                pair_target, index, state, metadata, exc,
            )
        state.complete_phase("discovery", metadata={"candidates": len(candidates)})

        # 4) Validation
        state.begin_phase("validation")
        try:
            validated, rejected = self._validate(candidates, context)
        except Exception as exc:
            state.fail_phase("validation", repr(exc))
            logger.error("[%s] validation phase failed: %r", pair_target.key, exc)
            return self._make_failure_result(
                pair_target, index, state, metadata, exc,
            )
        state.complete_phase(
            "validation",
            metadata={"validated": len(validated), "rejected": len(rejected)},
        )

        # 5) Execution
        state.begin_phase("execution")
        try:
            execution_results, execution_report = self._execute(validated, context)
        except Exception as exc:
            state.fail_phase("execution", repr(exc))
            logger.error("[%s] execution phase failed: %r", pair_target.key, exc)
            return self._make_failure_result(
                pair_target, index, state, metadata, exc,
            )
        state.complete_phase(
            "execution",
            metadata={"results": len(execution_results)},
        )

        # 6) Einhers + Portfolio
        # NOTE : _build_portfolio retourne des objets VIDES
        # (pas une exception) si pas d'execution_results.
        # On continue toujours le pipeline pour qu'un
        # DiscoveryPairResult soit produit et que l'export
        # (par le PairExporter) puisse écrire au moins
        # un summary traçant l'absence d'einhers.
        state.begin_phase("portfolio")
        try:
            einhers = self._build_einhers(execution_results)
            selection, allocation, portfolio_report = self._build_portfolio(
                execution_results, context,
            )
        except Exception as exc:
            state.fail_phase("portfolio", repr(exc))
            logger.error("[%s] portfolio phase failed: %r", pair_target.key, exc)
            return self._make_failure_result(
                pair_target, index, state, metadata, exc,
            )
        state.complete_phase(
            "portfolio",
            metadata={
                "einhers": len(einhers),
                "selected": (
                    selection.selected_count
                    if selection is not None else 0
                ),
            },
        )

        # 7) Memory
        state.begin_phase("memory")
        try:
            memory_snapshot = self._update_memory(
                context, execution_results, einhers, allocation, portfolio_report,
            )
        except Exception as exc:
            state.fail_phase("memory", repr(exc))
            logger.error("[%s] memory phase failed: %r", pair_target.key, exc)
            return self._make_failure_result(
                pair_target, index, state, metadata, exc,
            )
        state.complete_phase("memory")

        # 8) Knowledge
        state.begin_phase("knowledge")
        try:
            knowledge_snapshot = self._update_knowledge(
                context, einhers, allocation, portfolio_report,
            )
        except Exception as exc:
            state.fail_phase("knowledge", repr(exc))
            logger.error("[%s] knowledge phase failed: %r", pair_target.key, exc)
            return self._make_failure_result(
                pair_target, index, state, metadata, exc,
            )
        state.complete_phase("knowledge")

        # 9) Résultat final
        # NOTE : ``state.success`` reflète ici la capacité du
        # pipeline à s'exécuter jusqu'au bout, PAS la présence
        # d'einhers. Un pipeline qui produit 0 einhers (par
        # exemple parce que la validation rejette tout) est
        # un succès : la pipeline a tourné sans erreur. Le
        # nombre d'einhers est dans ``state.einher_count`` /
        # ``result.einher_count``.
        state.success = True
        state.finish(success=True)

        return DiscoveryPairResult(
            target=pair_target,
            index=index,
            state=state,
            dataset=context.dataset,
            candidates=tuple(candidates),
            validated=tuple(validated),
            rejected=tuple(rejected),
            execution_results=tuple(execution_results),
            execution_report=execution_report,
            einhers=tuple(einhers),
            selection=selection,
            allocation=allocation,
            portfolio_report=portfolio_report,
            memory_snapshot=dict(memory_snapshot),
            knowledge_snapshot=dict(knowledge_snapshot),
            export_paths={},
            metadata=dict(metadata or {}),
            started_at=state.started_at or _utc_now(),
            finished_at=state.finished_at or _utc_now(),
            success=state.success,
        )

    # ==================================================
    # PAIR CONTEXT BUILDER
    # ==================================================

    def _build_pair_context(
        self,
        target: DiscoveryTarget,
        *,
        state: EngineState,
        dataset: DatasetLoader | None = None,
    ) -> EngineContext:
        """
        Construit un EngineContext strictement lié à la paire.

        Chaque ressource (loader, registry, validator, ...) est
        créée ici, à partir des seules données de la paire.
        """

        loader = dataset if dataset is not None else self._load_dataset(target)

        if not loader.is_midas_mode:
            raise DatasetContractError(
                f"Engine requires MIDAS mode for pair {target.key}."
            )

        contract = loader.contract
        if not contract.horizons:
            raise DatasetContractError(
                f"Dataset contract for {target.key} has no horizons."
            )

        # FeatureRegistry par paire
        meta_path = (
            Path(self._config.dataset.midas_root)
            / loader._config.asset_class
            / loader._config.timeframe
            / "metadata.json"
        )
        if not meta_path.exists():
            raise DatasetContractError(
                f"metadata.json introuvable pour {target.key} : {meta_path}."
            )
        registry = FeatureRegistry(str(meta_path))

        validator = DatasetValidator()
        statistics = DatasetStatistics(loader)
        inspector = DatasetInspector(loader)

        return EngineContext(
            config=self._config,
            state=state,
            target=target,
            feature_registry=registry,
            dataset_loader=loader,
            dataset_validator=validator,
            dataset_statistics=statistics,
            dataset_inspector=inspector,
        )

    def _load_dataset(self, target: DiscoveryTarget) -> DatasetLoader:
        """
        Construit un DatasetLoader strictement pour la paire.
        """

        try:
            asset_class = resolve_asset_class(target.asset)
        except KeyError as exc:
            raise DatasetContractError(str(exc)) from exc

        return DatasetLoader.for_pair(
            midas_root=self._config.dataset.midas_root,
            asset=target.asset,
            asset_class=asset_class,
            timeframe=target.timeframe,
        )

    # ==================================================
    # PHASE 1 : DATASET CONTRACT
    # ==================================================

    def _verify_dataset(self, context: EngineContext) -> None:
        """
        Vérifie que le dataset est conforme au contrat.

        Toute violation lève une DatasetContractError ou
        DatasetValidationError.
        """

        loader = context.dataset_loader
        contract = loader.contract
        contract.verify_for_midas()
        context.dataset_validator.validate_midas(loader)

    # ==================================================
    # PHASE 2 : DISCOVERY
    # ==================================================

    def _discover(
        self,
        context: EngineContext,
    ) -> tuple[Any, ...]:
        """
        Lance la phase Discovery.

        Contrat de sortie : tuple de candidats (Hypothesis, ou
        tout objet exposant un attribut .hypothesis de type
        Hypothesis). tuple vide possible. Toute autre forme
        de sortie est une violation du contrat.
        """

        loader = context.dataset_loader
        registry = context.feature_registry
        midas = loader.midas

        feature_statistics = self._compute_feature_statistics(registry, midas.X)

        generator = DiscoveryGenerator(
            self._config,
            registry,
            feature_statistics=feature_statistics,
        )

        explorer = Explorer(
            registry=registry,
            config=self._config,
            generator=generator,
            search_config=self._config.search,
        )

        raw_output = explorer.run(
            seed_size=5,
            max_iterations=int(getattr(self._config.search, "max_depth", 3)),
        )

        return self._coerce_candidates(raw_output)

    @staticmethod
    def _compute_feature_statistics(
        registry: FeatureRegistry,
        X: np.ndarray,
    ) -> dict[Any, dict[str, Any]]:
        stats: dict[Any, dict[str, Any]] = {}
        for feature in registry.features:
            if not feature.enabled:
                continue
            col = feature.column_index
            if col < 0 or col >= X.shape[1]:
                continue
            column = X[:, col]
            finite = column[np.isfinite(column)]
            if finite.size == 0:
                continue
            entry = {
                "min": float(np.min(finite)),
                "max": float(np.max(finite)),
                "mean": float(np.mean(finite)),
                "std": float(np.std(finite)),
                "quantiles": {
                    0.25: float(np.quantile(finite, 0.25)),
                    0.50: float(np.quantile(finite, 0.50)),
                    0.75: float(np.quantile(finite, 0.75)),
                },
            }
            stats[col] = entry
            stats[str(col)] = entry
            stats[feature.name] = entry
            stats[feature.name.lower()] = entry
        return stats

    @staticmethod
    def _coerce_candidates(raw_output: Any) -> tuple[Any, ...]:
        """
        Normalise la sortie de Discovery en tuple de candidats.

        Contrat strict : la sortie doit être

        - None                       -> ()
        - DiscoveryResult            -> tuple des .hypothesis
                                        depuis frontier + history
        - DiscoveryNode unique       -> (node.hypothesis,)
        - iterable de Hypothesis      -> tuple
        - iterable de DiscoveryNode   -> tuple des .hypothesis
        - iterable de Candidate      -> tuple

        Tout autre type est une violation du contrat.
        """

        if raw_output is None:
            return ()

        # 1) DiscoveryResult
        if Engine._is_discovery_result(raw_output):
            nodes = list(raw_output.frontier) + list(raw_output.history)
            return tuple(Engine._node_to_hypothesis(n) for n in nodes)

        # 2) DiscoveryNode unique
        if Engine._is_discovery_node(raw_output):
            return (raw_output.hypothesis,)

        # 3) Iterable de candidats
        if hasattr(raw_output, "__iter__") and not isinstance(
            raw_output, (str, bytes, Mapping)
        ):
            items = list(raw_output)
            coerced: list[Any] = []
            for item in items:
                if isinstance(item, Hypothesis):
                    coerced.append(item)
                elif Engine._is_discovery_node(item):
                    coerced.append(item.hypothesis)
                elif Engine._is_candidate_like(item):
                    coerced.append(item)
                else:
                    raise DiscoveryContractError(
                        f"Discovery output contains an item of type "
                        f"{type(item).__name__} which is not a "
                        f"Hypothesis / DiscoveryNode / Candidate."
                    )
            return tuple(coerced)

        # 4) Candidate unique
        if Engine._is_candidate_like(raw_output):
            return (raw_output,)

        raise DiscoveryContractError(
            f"Discovery output type {type(raw_output).__name__} "
            f"violates the discovery contract."
        )

    @staticmethod
    def _is_discovery_result(value: Any) -> bool:
        cls = type(value)
        return (
            cls.__name__ == "DiscoveryResult"
            and hasattr(value, "frontier")
            and hasattr(value, "history")
        )

    @staticmethod
    def _is_discovery_node(value: Any) -> bool:
        return type(value).__name__ == "DiscoveryNode" and hasattr(
            value, "hypothesis"
        )

    @staticmethod
    def _node_to_hypothesis(node: Any) -> Hypothesis:
        hyp = getattr(node, "hypothesis", None)
        if not isinstance(hyp, Hypothesis):
            raise DiscoveryContractError(
                f"DiscoveryNode has invalid hypothesis: "
                f"{type(hyp).__name__}."
            )
        return hyp

    @staticmethod
    def _is_candidate_like(value: Any) -> bool:
        if isinstance(value, Hypothesis):
            return True
        if hasattr(value, "hypothesis") and isinstance(
            getattr(value, "hypothesis"), Hypothesis
        ):
            return True
        return False

    # ==================================================
    # PHASE 3 : VALIDATION
    # ==================================================

    def _validate(
        self,
        candidates: Sequence[Any],
        context: EngineContext,
    ) -> tuple[tuple[ValidatedCandidate, ...], tuple[Any, ...]]:
        """
        Lance la phase Validation.

        Contrat de sortie :
        - tuple de ValidatedCandidate
        - tuple de candidats rejetés

        Si la liste de candidats est vide, le résultat est
        ((), ()) — pas une erreur.

        En mode MIDAS, le validator reçoit un DatasetSplit
        synthétique construit à partir des arrays MIDAS.
        """

        if not candidates:
            return (), ()

        loader = context.dataset_loader
        split = self._resolve_validation_split(loader)

        evaluator = ValidationEvaluator(
            config=self._config,
            dataset=split,
        )

        validated: list[ValidatedCandidate] = []
        rejected: list[Any] = []
        for item in candidates:
            assessment = evaluator.assess(
                item,
                dataset=split,
                split_name=split.name,
            )
            if assessment.passed and assessment.validated_candidate is not None:
                validated.append(assessment.validated_candidate)
            else:
                rejected.append(item)

        return tuple(validated), tuple(rejected)

    @staticmethod
    def _resolve_validation_split(loader: DatasetLoader):
        """
        Retourne un DatasetSplit à partir du loader.

        - Mode splits : renvoie le split "validation".
        - Mode MIDAS  : construit un split synthétique
          (X, Y_ret) à partir des arrays MIDAS.
        """

        if not loader.is_midas_mode:
            return loader.validation()

        midas = loader.midas
        return DatasetSplit(
            name="validation",
            X=midas.X,
            Y=midas.Y_ret,
        )

    # ==================================================
    # PHASE 4 : EXECUTION
    # ==================================================

    def _execute(
        self,
        validated_candidates: Sequence[ValidatedCandidate],
        context: EngineContext,
    ) -> tuple[tuple[ExecutionResult, ...], ExecutionReport]:
        """
        Lance la phase Execution.

        Contrat de sortie :
        - tuple de ExecutionResult
        - ExecutionReport agrégé

        Si la liste de candidats est vide, renvoie ((), rapport vide).
        """

        if not validated_candidates:
            return (), ExecutionReport()

        engine = ExecutionEngine(config=self._config)
        if engine is None:
            raise ExecutionContractError("Execution engine is unavailable.")

        engine.reset()

        midas = context.dataset_loader.midas

        # Le replay attend une price series 1D alignée sur matrix.
        # En mode MIDAS, Y_ret est (n_samples, n_horizons). On prend
        # la première colonne (horizon court) comme proxy de la
        # price series. Le contract d'entrée du replay est strict.
        prices = self._resolve_prices_series(midas)

        # Le replay attend des timestamps en SECONDES depuis epoch.
        # Les timestamps MIDAS sont en MILLISECONDES. Conversion
        # obligatoire pour rester dans la plage valide de
        # datetime.utcfromtimestamp.
        timestamps = self._resolve_timestamps(midas)

        results = engine.execute_batch(
            tuple(validated_candidates),
            dataset=context.dataset_loader,
            matrix=midas.X,
            prices=prices,
            timestamps=timestamps,
        )

        for r in results:
            if not isinstance(r, ExecutionResult):
                raise ExecutionContractError(
                    f"Execution engine returned item of type "
                    f"{type(r).__name__}, expected ExecutionResult."
                )

        return tuple(results), engine.report

    @staticmethod
    def _resolve_prices_series(midas: Any) -> np.ndarray:
        """
        Adapte les arrays MIDAS en price series 1D compatible
        avec le ReplayEngine.

        - Y_ret (n, k) → on prend la première colonne.
        - Y_ret (n,)   → renvoyé tel quel.
        """

        y = midas.Y_ret
        if y.ndim == 1:
            return y
        if y.ndim == 2 and y.shape[1] >= 1:
            return y[:, 0]
        raise ExecutionContractError(
            f"Cannot derive 1D price series from MIDAS Y_ret "
            f"with shape {y.shape}."
        )

    @staticmethod
    def _resolve_timestamps(midas: Any) -> np.ndarray:
        """
        Convertit les timestamps MIDAS (ms) en secondes depuis epoch.

        Le ReplayEngine attend des timestamps en secondes.
        """

        ts = midas.ts
        # Heuristique : si l'ordre de grandeur est > 1e10, on est
        # en millisecondes ; sinon en secondes. Année 2286 en
        # secondes = 1e10 ; année 2286 en ms = 1e13.
        sample = float(ts[0]) if ts.size > 0 else 0.0
        if sample > 1e10:
            return (ts.astype(np.int64) // 1000).astype(np.int64)
        return ts.astype(np.int64)

    # ==================================================
    # PHASE 5 : EINHERS
    # ==================================================

    def _build_einhers(
        self,
        execution_results: Sequence[ExecutionResult],
    ) -> tuple[Einher, ...]:
        """
        Construit les objets Einher à partir des ExecutionResult.

        Contrat de sortie : tuple de Einher.
        """

        einhers: list[Einher] = []
        for result in execution_results:
            if not isinstance(result, ExecutionResult):
                raise ExecutionContractError(
                    f"Cannot build Einher from non-ExecutionResult "
                    f"of type {type(result).__name__}."
                )
            einher = Einher.from_execution_result(result)
            if not isinstance(einher, Einher):
                raise ExecutionContractError(
                    f"Einher.from_execution_result returned "
                    f"{type(einher).__name__} instead of Einher."
                )
            einhers.append(einher)
        return tuple(einhers)

    # ==================================================
    # PHASE 6 : PORTFOLIO
    # ==================================================

    def _build_portfolio(
        self,
        execution_results: Sequence[ExecutionResult],
        context: EngineContext,
    ) -> tuple[PortfolioSelection, PortfolioAllocation, PortfolioReport]:
        """
        Lance la phase Portfolio.

        Contrat de sortie :
        - ``(PortfolioSelection, PortfolioAllocation, PortfolioReport)``
        - si ``execution_results`` est vide, renvoie des
          objets **vides** (pas une exception) ; le pipeline
          continue et un DiscoveryPairResult avec
          ``einhers=()`` est produit. C'est le PairExporter
          qui écrira au moins un summary traçant l'absence
          d'einhers.
        - lève ``PhaseContractError`` uniquement si une étape
          du portfolio retourne un objet du mauvais type.
        """

        if not execution_results:
            return self._empty_portfolio(context)

        selector = PortfolioSelector(config=self._config)
        selection = selector.select(
            tuple(execution_results),
            metadata={
                "asset": context.target.asset,
                "timeframe": context.target.timeframe,
                "run_id": _read_run_id(context),
            },
        )
        if not isinstance(selection, PortfolioSelection):
            raise PhaseContractError(
                f"PortfolioSelector returned {type(selection).__name__} "
                f"instead of PortfolioSelection."
            )

        selected_results = list(selection.results)

        correlation = PortfolioCorrelationAnalyzer(config=self._config)
        corr_matrix = correlation.correlate(tuple(selected_results))

        diversification = DiversificationEngine(config=self._config)
        div_assessment = diversification.assess(
            tuple(selected_results),
            correlation=corr_matrix,
            metadata={
                "asset": context.target.asset,
                "timeframe": context.target.timeframe,
            },
        )

        risk = PortfolioRiskModel(config=self._config)
        risk_assessment = risk.assess(
            tuple(selected_results),
            correlation=corr_matrix,
            diversification=div_assessment,
            metadata={
                "asset": context.target.asset,
                "timeframe": context.target.timeframe,
            },
        )

        optimizer = PortfolioOptimizer(config=self._config)
        optimization = optimizer.optimize(
            tuple(selected_results),
            correlation=corr_matrix,
            diversification=div_assessment,
            risk=risk_assessment,
            total_capital=int(
                getattr(self._config.execution, "max_open_positions", 1)
            ),
            metadata={
                "asset": context.target.asset,
                "timeframe": context.target.timeframe,
            },
        )
        allocation = getattr(optimization, "best_allocation", None)
        if allocation is None:
            allocator = PortfolioAllocator(config=self._config)
            allocation = allocator.allocate(
                tuple(selected_results),
                weights=None,
                total_capital=int(
                    getattr(self._config.execution, "max_open_positions", 1)
                ),
                risk=risk_assessment,
                diversification=div_assessment,
                correlation=corr_matrix,
                metadata={
                    "asset": context.target.asset,
                    "timeframe": context.target.timeframe,
                },
            )

        if not isinstance(allocation, PortfolioAllocation):
            raise PhaseContractError(
                f"Allocator returned {type(allocation).__name__} "
                f"instead of PortfolioAllocation."
            )

        reporter = PortfolioReporter(config=self._config)
        portfolio_report = reporter.build(
            allocation=allocation,
            selection=selection,
            risk=risk_assessment,
            diversification=div_assessment,
            correlation=corr_matrix,
            name=context.target.slug,
            metadata={
                "asset": context.target.asset,
                "timeframe": context.target.timeframe,
                "run_id": _read_run_id(context),
            },
        )
        if not isinstance(portfolio_report, PortfolioReport):
            raise PhaseContractError(
                f"PortfolioReporter returned "
                f"{type(portfolio_report).__name__} instead of "
                f"PortfolioReport."
            )

        return selection, allocation, portfolio_report

    @staticmethod
    def _empty_portfolio(
        context: EngineContext,
    ) -> tuple[PortfolioSelection, PortfolioAllocation, PortfolioReport]:
        """
        Construit un portfolio strictement vide (utilisé
        quand ``execution_results`` est vide).

        Renvoie trois objets qui seérialisent en dict
        minimal : pas de crash, pas d'exception, le
        pipeline continue.
        """

        from portfolio.allocator import PortfolioAllocation
        from portfolio.allocator import PortfolioAllocatorSettings
        from portfolio.capital import CapitalPlan
        from portfolio.capital import CapitalSettings
        from portfolio.portfolio_report import PortfolioReport
        from portfolio.selector import PortfolioSelection
        from portfolio.selector import PortfolioSelectorSettings

        run_id = _read_run_id(context)
        slug = context.target.slug

        empty_selection = PortfolioSelection(
            selected=(),
            rejected=(),
            settings=PortfolioSelectorSettings(),
            metadata={
                "asset": context.target.asset,
                "timeframe": context.target.timeframe,
                "run_id": run_id,
                "empty": True,
            },
        )

        capital_settings = CapitalSettings(total_capital=0.0)
        empty_capital_plan = CapitalPlan(
            total_capital=0.0,
            reserve_capital=0.0,
            investable_capital=0.0,
            entries=(),
            settings=capital_settings,
            metadata={"empty": True},
        )
        empty_allocation = PortfolioAllocation(
            entries=(),
            capital_plan=empty_capital_plan,
            risk=None,
            diversification=None,
            correlation=None,
            score=0.0,
            metadata={
                "asset": context.target.asset,
                "timeframe": context.target.timeframe,
                "run_id": run_id,
                "empty": True,
            },
        )

        empty_report = PortfolioReport(
            name=slug,
            entries=[],
            rejected=[],
            allocation=empty_allocation,
            selection=empty_selection,
            risk=None,
            diversification=None,
            correlation=None,
            capital_plan=empty_capital_plan,
            metadata={
                "asset": context.target.asset,
                "timeframe": context.target.timeframe,
                "run_id": run_id,
                "empty": True,
            },
            selected_count=0,
            rejected_count=0,
        )
        return empty_selection, empty_allocation, empty_report

    # ==================================================
    # PHASE 7 : MEMORY
    # ==================================================

    def _update_memory(
        self,
        context: EngineContext,
        execution_results: Sequence[ExecutionResult],
        einhers: Sequence[Einher],
        allocation: PortfolioAllocation,
        portfolio_report: PortfolioReport,
    ) -> dict[str, Any]:
        """
        Met à jour la mémoire du moteur (per-pair).

        Toute erreur de la phase est propagée. L'Engine
        ne porte pas de run_id — il lit celui du target
        metadata si le caller l'y a placé.
        """

        from memory.corpus_history import CorpusHistory
        from memory.explored_regions import ExploredRegions
        from memory.failed_regions import FailedRegions
        from memory.family_history import FamilyHistory
        from memory.feature_history import FeatureHistory
        from memory.learning import LearningEngine
        from memory.search_history import SearchHistory
        from memory.successful_regions import SuccessfulRegions

        target = context.target
        run_id = _read_run_id(context)
        run_meta = {
            "asset": target.asset,
            "timeframe": target.timeframe,
            "run_id": run_id,
        }

        # 1) search_history
        search_history = SearchHistory()
        sh_entry = search_history.record(
            query=target.key,
            phase="discovery",
            objective="build_corpus",
            seed=target.asset,
            features=(),
            families=(target.asset,),
            regions=(target.timeframe,),
            parameters={"run_id": run_id},
            result_count=len(einhers),
            accepted_count=len(einhers),
            rejected_count=0,
            useful=bool(einhers),
            success=bool(einhers),
            score=float(getattr(portfolio_report, "average_score", 0.0) or 0.0),
            reason="pair_completed",
            notes=(target.key,),
            metadata=run_meta,
        )

        # 2) explored_regions
        explored = ExploredRegions()
        ex_entry = explored.register(
            region_key=target.key,
            phase="discovery",
            family=target.asset,
            feature=target.timeframe,
            depth=0,
            size=len(einhers),
            attempts=1,
            score=float(getattr(portfolio_report, "average_score", 0.0) or 0.0),
            metadata={"run_id": run_id},
        )

        # 3) successful_regions
        successful = SuccessfulRegions()
        su_entry = successful.register(
            region_key=target.key,
            phase="discovery",
            family=target.asset,
            feature=target.timeframe,
            hits=1,
            success_count=1 if einhers else 0,
            score=float(getattr(portfolio_report, "average_score", 0.0) or 0.0),
            yield_rate=1.0 if einhers else 0.0,
            metadata={"run_id": run_id},
        )

        # 4) failed_regions
        failed = FailedRegions()
        fa_entry = failed.register(
            region_key=target.key,
            phase="discovery",
            family=target.asset,
            feature=target.timeframe,
            attempts=1,
            failure_count=0 if einhers else 1,
            score=float(getattr(portfolio_report, "average_score", 0.0) or 0.0),
            reason="" if einhers else "no_result",
            metadata={"run_id": run_id},
        )

        # 5) feature_history
        feature_history = FeatureHistory()
        for einher in einhers:
            feature_history.register(
                self._feature_key(einher),
                family=self._family_key(einher),
                phase="execution",
                success=True,
                score=float(self._score(einher)),
                metadata={"run_id": run_id},
            )

        # 6) family_history
        family_history = FamilyHistory()
        for einher in einhers:
            family_history.register(
                self._family_key(einher),
                success=True,
                score=float(self._score(einher)),
                metadata={"run_id": run_id},
            )

        # 7) corpus_history
        corpus_history = CorpusHistory()
        corpus_history.register(
            corpus_key=target.key,
            version=1,
            entry_count=len(einhers),
            selected_count=len(einhers),
            total_capital=0.0,
            total_weight=0.0,
            total_pnl=0.0,
            fingerprints=tuple(e.fingerprint for e in einhers),
            summary={},
            metadata={"run_id": run_id},
        )

        # 8) learning — synthèse à partir de TOUTES les mémoires
        learning = LearningEngine().learn(
            search_history=search_history,
            explored_regions=explored,
            successful_regions=successful,
            failed_regions=failed,
            feature_history=feature_history,
            family_history=family_history,
            corpus_history=corpus_history,
            metadata=run_meta,
        )

        # Snapshot final
        snapshot: dict[str, Any] = {
            "search_history": sh_entry.to_dict(),
            "explored_regions": ex_entry.to_dict(),
            "successful_regions": su_entry.to_dict(),
            "failed_regions": fa_entry.to_dict(),
            "feature_history": {
                "touched": len(einhers),
                "summary": (
                    feature_history.summary.to_dict()
                    if hasattr(feature_history, "summary")
                    else {}
                ),
            },
            "family_history": {
                "touched": len(einhers),
                "summary": (
                    family_history.summary.to_dict()
                    if hasattr(family_history, "summary")
                    else {}
                ),
            },
            "corpus_history": {
                "version": 1,
                "entry_count": len(einhers),
            },
            "learning": learning.to_dict(),
        }

        return snapshot

    # ==================================================
    # PHASE 8 : KNOWLEDGE
    # ==================================================

    def _update_knowledge(
        self,
        context: EngineContext,
        einhers: Sequence[Einher],
        allocation: PortfolioAllocation,
        portfolio_report: PortfolioReport,
    ) -> dict[str, Any]:
        """
        Met à jour la base de connaissance (per-pair).

        Si aucun Einher, renvoie un snapshot vide.
        """

        from knowledge.clustering import ClusterEngine
        from knowledge.fingerprints import FingerprintRegistry
        from knowledge.graph import KnowledgeGraph
        from knowledge.insights import InsightEngine
        from knowledge.ontology import OntologyEngine
        from knowledge.taxonomy import TaxonomyEngine

        target = context.target
        run_id = _read_run_id(context)
        run_meta = {
            "asset": target.asset,
            "timeframe": target.timeframe,
            "run_id": run_id,
        }
        snapshot: dict[str, Any] = {}

        if not einhers:
            return snapshot

        snapshot["taxonomy"] = [
            item.to_dict()
            for item in TaxonomyEngine().classify_many(list(einhers))
        ]

        registry = FingerprintRegistry()
        fingerprints = []
        for einher in einhers:
            fp = registry.build(
                einher,
                kind="einher",
                label=einher.profile.name,
                metadata=run_meta,
            )
            fingerprints.append(fp.to_dict())
        snapshot["fingerprints"] = fingerprints

        graph = KnowledgeGraph().build_from_objects(
            list(einhers), metadata=run_meta,
        )
        snapshot["graph"] = graph.to_dict()

        clusters = ClusterEngine().cluster(list(einhers), metadata=run_meta)
        snapshot["clusters"] = [c.to_dict() for c in clusters]

        insights = InsightEngine().analyze(
            list(einhers),
            graph=graph,
            clusters=tuple(clusters),
            metadata=run_meta,
        )
        snapshot["insights"] = insights.to_dict()

        snapshot["ontology"] = OntologyEngine().to_dict()

        return snapshot

    # ==================================================
    # HELPERS
    # ==================================================

    @staticmethod
    def _resolve_target(target: Any) -> DiscoveryTarget:
        if isinstance(target, DiscoveryTarget):
            return target
        if isinstance(target, str):
            text = target.strip()
            if "@" in text:
                asset, timeframe = text.split("@", 1)
                return DiscoveryTarget(
                    asset=asset.strip(), timeframe=timeframe.strip(),
                )
            return DiscoveryTarget(asset=text or "unknown", timeframe="unknown")
        if isinstance(target, Mapping):
            return DiscoveryTarget(
                asset=str(target.get("asset", "unknown")),
                timeframe=str(target.get("timeframe", "unknown")),
                metadata=dict(target.get("metadata", {})),
            )
        raise TypeError(
            f"Unsupported target type: {type(target).__name__}."
        )

    @staticmethod
    def _feature_key(einher: Einher) -> str:
        try:
            return str(
                einher.candidate.hypothesis.conditions[0].left.name
            ).lower()
        except Exception:
            return "unknown"

    @staticmethod
    def _family_key(einher: Einher) -> str:
        try:
            return str(
                einher.candidate.hypothesis.conditions[0]
                .left.economic_family.value
            ).lower()
        except Exception:
            return "unknown"

    @staticmethod
    def _score(einher: Einher) -> float:
        try:
            return float(einher.report.metadata.get("score", 0.0) or 0.0)
        except Exception:
            return 0.0

    # ==================================================
    # FAILURE RESULT
    # ==================================================

    def _make_failure_result(
        self,
        target: DiscoveryTarget,
        index: int,
        state: EngineState,
        metadata: Mapping[str, Any] | None,
        exc: BaseException,
    ) -> DiscoveryPairResult:

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
    # PYTHON PROTOCOL
    # ==================================================

    def __repr__(self) -> str:
        return "Engine()"
