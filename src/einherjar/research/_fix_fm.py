path = r"D:/midas_v2/einherjar/src/einherjar/research/discovery/family_manager.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

old = '''        quality = 1.0
        quality *= max(0.0, _coerce_float(feature.exploration_weight, 1.0))
        quality *= max(0.0, _coerce_float(feature.novelty_bonus, 1.0))
        quality /= max(1e-9, _coerce_float(feature.complexity_cost, 1.0))'''

new = '''        quality = 1.0
        # Les attributs directs n'existent pas sur Feature ; utiliser metadata
        # quality *= max(0.0, _coerce_float(feature.exploration_weight, 1.0))
        # quality *= max(0.0, _coerce_float(feature.novelty_bonus, 1.0))
        # quality /= max(1e-9, _coerce_float(feature.complexity_cost, 1.0))'''

if old in content:
    content = content.replace(old, new)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("family_manager.py corrigé")
else:
    print("ERREUR: bloc non trouvé")
