path = r"D:/midas_v2/einherjar/src/einherjar/research/discovery.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

old = '''        # Lancer la recherche
        result = explorer.run(
            seed_size=getattr(self.settings.search_config, "seed_population_size", None),
            max_iterations=getattr(self.settings.search_config, "max_depth", 3),
        )'''

new = '''        # Lancer la recherche avec un seed_size raisonnable
        result = explorer.run(
            seed_size=5,
            max_iterations=getattr(self.settings.search_config, "max_depth", 3),
        )'''

if old in content:
    content = content.replace(old, new)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("seed_size corrigé -> 5")
else:
    print("ERREUR: bloc non trouvé")
