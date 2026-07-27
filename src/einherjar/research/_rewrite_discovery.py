path = r"D:/midas_v2/einherjar/src/einherjar/research/discovery.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# ============================================
# 1. Réécrire DiscoveryComponents.create()
# ============================================
old_create = '''    @classmethod
    def create(cls, config: Any | None) -> "DiscoveryComponents":
        root_config = config
        search_config = _build_search_config(config)
        execution_config = _build_execution_config(config)

        return cls(
            dataset_loader=_instantiate(
                ("dataset.loader",),
                ("DatasetLoader", "Loader", "DataLoader"),
                config=root_config,
            ),
            dataset_validator=_instantiate(
                ("dataset.validator",),
                ("DatasetValidator", "Validator", "DataValidator"),
                config=root_config,
            ),
            discovery_generator=_instantiate(
                ("discovery.generator",),
                ("DiscoveryGenerator", "Generator", "DiscoveryEngine", "SearchEngine"),
                config=search_config,
            ),
            discovery_explorer=_instantiate(
                ("discovery.explorer",),
                ("DiscoveryExplorer", "Explorer", "ExplorerEngine"),
                config=search_config,
            ),
            validation_engine=_instantiate(
                ("validation.engine", "validation.validator", "validation.validation_report"),
                ("ValidationEngine", "Validator", "ValidationPipeline", "Evaluator"),
                config=root_config,
            ),
            execution_engine=_instantiate(
                ("execution.executor",),
                ("ExecutionEngine",),
                config=root_config,
            ),
            portfolio_selector=_instantiate(
                ("portfolio.selector",),
                ("PortfolioSelector",),
                config=root_config,
            ),
            portfolio_correlation=_instantiate(
                ("portfolio.correlation",),
                ("PortfolioCorrelationAnalyzer",),
                config=root_config,
            ),
            portfolio_diversification=_instantiate(
                ("portfolio.diversification",),
                ("DiversificationEngine",),
                config=root_config,
            ),
            portfolio_risk=_instantiate(
                ("portfolio.risk",),
                ("PortfolioRiskModel",),
                config=root_config,
            ),
            portfolio_capital=_instantiate(
                ("portfolio.capital",),
                ("CapitalManager",),
                config=root_config,
            ),
            portfolio_allocator=_instantiate(
                ("portfolio.allocator",),
                ("PortfolioAllocator",),
                config=root_config,
            ),
            portfolio_reporter=_instantiate(
                ("portfolio.portfolio_report",),
                ("PortfolioReporter",),
                config=root_config,
            ),
            portfolio_optimizer=_instantiate(
                ("portfolio.optimizer",),
                ("PortfolioOptimizer",),
                config=root_config,
            ),'''

new_create = '''    @classmethod
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
            portfolio_optimizer=None,'''

if old_create in content:
    content = content.replace(old_create, new_create)
    print("1. DiscoveryComponents.create() réécrit")
else:
    print("ERREUR: DiscoveryComponents.create() non trouvé")

# ============================================
# 2. Réécrire _load_dataset()
# ============================================
old_load_dataset = '''    def _load_dataset(self, target: DiscoveryTarget, *, context: DiscoveryContext) -> Any:
        loader = self.components.dataset_loader
        if loader is None:
            raise RuntimeError("Dataset loader is unavailable.")

        # Si le loader est une instance DatasetLoader, le retourner directement
        if hasattr(loader, "_load") or hasattr(loader, "midas") or hasattr(loader, "splits"):
            return loader

        return _try_call(
            loader,
            ("load_pair", "load", "load_dataset", "get_dataset"),
            asset=target.asset,
            timeframe=target.timeframe,
            target=target,
            context=context,
            config=self.config,
            search_config=self.settings.search_config,
            execution_config=self.settings.execution_config,
            metadata=target.metadata,
        )'''

new_load_dataset = '''    def _load_dataset(self, target: DiscoveryTarget, *, context: DiscoveryContext) -> Any:
        from config.dataset import DatasetConfig
        from dataset.loader import DatasetLoader

        dataset_cfg = DatasetConfig(
            midas_root=r"D:/midas_v2/midasV3/src/data/compiled",
            asset=target.asset,
            asset_class=_resolve_asset_class(target.asset),
            timeframe=target.timeframe,
        )
        return DatasetLoader(dataset_cfg)'''

if old_load_dataset in content:
    content = content.replace(old_load_dataset, new_load_dataset)
    print("2. _load_dataset() réécrit")
else:
    print("ERREUR: _load_dataset() non trouvé")

# ============================================
# 3. Réécrire _discover()
# ============================================
old_discover = '''    def _discover(self, dataset: Any, *, target: DiscoveryTarget, context: DiscoveryContext) -> Any:
        stage = self.components.discovery_explorer or self.components.discovery_generator
        if stage is None:
            raise RuntimeError("Discovery engine is unavailable.")

        # Si c'est un Explorer avec une méthode run(), l'appeler directement
        if hasattr(stage, "run") and callable(getattr(stage, "run")):
            try:
                return stage.run(
                    seed_size=getattr(self.settings.search_config, "seed_population_size", None),
                    max_iterations=getattr(self.settings.search_config, "max_depth", 3),
                )
            except Exception:
                pass

        output = _try_call(
            stage,
            ("run", "discover", "generate", "search", "explore"),
            dataset=dataset,
            data=dataset,
            asset=target.asset,
            timeframe=target.timeframe,
            target=target,
            context=context,
            config=self.settings.search_config,
            metadata=target.metadata,
        )

        if output is None:
            output = _try_call(
                stage,
                ("run", "discover", "generate", "search", "explore"),
                dataset,
                target=target,
                context=context,
                config=self.settings.search_config,
                metadata=target.metadata,
            )

        return output'''

new_discover = '''    def _discover(self, dataset: Any, *, target: DiscoveryTarget, context: DiscoveryContext) -> Any:
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

        # Lancer la recherche
        result = explorer.run(
            seed_size=getattr(self.settings.search_config, "seed_population_size", None),
            max_iterations=getattr(self.settings.search_config, "max_depth", 3),
        )
        return result'''

if old_discover in content:
    content = content.replace(old_discover, new_discover)
    print("3. _discover() réécrit")
else:
    print("ERREUR: _discover() non trouvé")

# ============================================
# 4. Réécrire _validate()
# ============================================
old_validate = '''    def _validate(
        self,
        candidates: Sequence[Any],
        *,
        dataset: Any,
        target: DiscoveryTarget,
        context: DiscoveryContext,
    ) -> tuple[Any, tuple[Any, ...], tuple[Any, ...]]:
        validator = self.components.validation_engine
        if validator is None:
            return None, tuple(candidates), ()'''

new_validate = '''    def _validate(
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
        )'''

if old_validate in content:
    content = content.replace(old_validate, new_validate)
    print("4. _validate() réécrit (début)")
else:
    print("ERREUR: _validate() non trouvé")

# ============================================
# 5. Réécrire _execute()
# ============================================
old_execute_start = '''    def _execute(
        self,
        validated_candidates: Sequence[Any],
        *,
        dataset: Any,
        target: DiscoveryTarget,
        context: DiscoveryContext,
    ) -> tuple[tuple[Any, ...], Any]:
        engine = self._spawn_pair_component(self.components.execution_engine, self.settings.execution_config)
        if engine is None:
            raise RuntimeError("Execution engine is unavailable.")'''

new_execute_start = '''    def _execute(
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
            raise RuntimeError("Execution engine is unavailable.")'''

if old_execute_start in content:
    content = content.replace(old_execute_start, new_execute_start)
    print("5. _execute() réécrit (début)")
else:
    print("ERREUR: _execute() non trouvé")

# ============================================
# 6. Réécrire _build_portfolio()
# ============================================
old_portfolio = '''    def _build_portfolio(
        self,
        execution_results: Sequence[Any],
        *,
        target: DiscoveryTarget,
        context: DiscoveryContext,
    ) -> tuple[Any, Any, Any]:
        selector = self._spawn_pair_component(self.components.portfolio_selector, self.config)
        correlation = self._spawn_pair_component(self.components.portfolio_correlation, self.config)
        diversification = self._spawn_pair_component(self.components.portfolio_diversification, self.config)
        risk_model = self._spawn_pair_component(self.components.portfolio_risk, self.config)
        capital_manager = self._spawn_pair_component(self.components.portfolio_capital, self.config)
        allocator = self._spawn_pair_component(self.components.portfolio_allocator, self.config)
        reporter = self._spawn_pair_component(self.components.portfolio_reporter, self.config)
        optimizer = self._spawn_pair_component(self.components.portfolio_optimizer, self.config)'''

new_portfolio = '''    def _build_portfolio(
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
        optimizer = PortfolioOptimizer(config=self.config)'''

if old_portfolio in content:
    content = content.replace(old_portfolio, new_portfolio)
    print("6. _build_portfolio() réécrit")
else:
    print("ERREUR: _build_portfolio() non trouvé")

# ============================================
# 7. Ajouter helper _resolve_asset_class
# ============================================
if "def _resolve_asset_class" not in content:
    # Insérer avant _utc_now ou au début des helpers
    helper = '''
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

'''
    content = content.replace("def _utc_now() -> datetime:", helper + "def _utc_now() -> datetime:")
    print("7. _resolve_asset_class() ajouté")
else:
    print("7. _resolve_asset_class() déjà présent")

# Sauvegarder
with open(path, "w", encoding="utf-8") as f:
    f.write(content)
print("\ndiscovery.py sauvegardé")
