path = r"D:/midas_v2/einherjar/src/einherjar/research/discovery/explorer.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

content = content.replace(
    "self._heuristics.available_actions(",
    "self._generator.available_actions("
)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)
print("explorer.py corrigé (available_actions -> generator)")
