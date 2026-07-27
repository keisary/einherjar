path = r"D:/midas_v2/einherjar/src/einherjar/research/discovery.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Corriger _instantiate pour essayer from_config / from_settings
old_instantiate = '''def _instantiate(
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
                    for kwargs in ({"config": config}, {"settings": config}, {}):
                        try:
                            return cls(**kwargs)  # type: ignore[misc]
                        except Exception:
                            continue
                else:
                    try:
                        return cls()  # type: ignore[misc]
                    except Exception:
                        continue

        if allow_module_fallback:
            return module

    return None'''

new_instantiate = '''def _instantiate(
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

    return None'''

if old_instantiate in content:
    content = content.replace(old_instantiate, new_instantiate)
    print("_instantiate corrigé (from_config ajouté)")
else:
    print("ERREUR: _instantiate non trouvé")

# 2. Corriger _load_dataset pour retourner le loader directement
old_load = '''    def _load_dataset(self, target: DiscoveryTarget, *, context: DiscoveryContext) -> Any:
        loader = self.components.dataset_loader
        if loader is None:
            raise RuntimeError("Dataset loader is unavailable.")

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

new_load = '''    def _load_dataset(self, target: DiscoveryTarget, *, context: DiscoveryContext) -> Any:
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

if old_load in content:
    content = content.replace(old_load, new_load)
    print("_load_dataset corrigé")
else:
    print("ERREUR: _load_dataset non trouvé")

# 3. Corriger _discover pour passer le loader correctement
old_discover = '''    def _discover(self, dataset: Any, *, target: DiscoveryTarget, context: DiscoveryContext) -> Any:
        stage = self.components.discovery_explorer or self.components.discovery_generator
        if stage is None:
            raise RuntimeError("Discovery engine is unavailable.")

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

if old_discover in content:
    content = content.replace(old_discover, new_discover)
    print("_discover corrigé")
else:
    print("ERREUR: _discover non trouvé")

with open(path, "w", encoding="utf-8") as f:
    f.write(content)
print("discovery.py sauvegardé")
