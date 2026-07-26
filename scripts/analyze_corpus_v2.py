import json
from collections import Counter

with open("D:/midas_v2/einherjar/config/corpus_brut_v2.json", "r", encoding="utf-8") as f:
    data = json.load(f)

domains = Counter(e["domain"] for e in data["einhers"])
print("Repartition domaines:")
for k, v in domains.most_common():
    print(f"  {k}: {v}")
print(f"Total: {len(data['einhers'])}")

# Verifier diversite des features utilisees
features = set()
for e in data["einhers"][:500]:
    trig = e["trigger"]
    for word in trig.replace("(", " ").replace(")", " ").replace("==", " ").replace("<", " ").replace(">", " ").replace("<=", " ").replace(">=", " ").split():
        if word in ("AND", "OR", "1", "0", "30", "40", "50", "60", "70", "80"):
            continue
        try:
            float(word)
        except ValueError:
            features.add(word)

print(f"\nFeatures uniques dans les 500 premiers Einhers: {len(features)}")
print("Exemples:", sorted(features)[:20])

# Verifier que pas de col_ residuel
col_residuel = [e["name"] for e in data["einhers"] if "col_" in e["trigger"]]
print(f"\nEinhers avec col_ residuel: {len(col_residuel)}")
