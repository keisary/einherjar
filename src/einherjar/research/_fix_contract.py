path = r"D:/midas_v2/einherjar/src/einherjar/research/dataset/contract.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

old = '''    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
    ) -> "DatasetContract":

        return cls(
            feature_count=data["feature_count"],
            feature_names=tuple(data["feature_names"]),
            label_names=tuple(data.get("label_names", ())),
            horizons=tuple(data.get("horizons", ())),
            dtype=data["dtype"],
            version=data.get("version", ""),
            metadata=data.get("metadata", {}),
        )'''

new = '''    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
    ) -> "DatasetContract":
        # Compatibilité MIDAS : features_count vs feature_count
        feature_count = data.get("feature_count") or data.get("features_count", 0)
        return cls(
            feature_count=feature_count,
            feature_names=tuple(data.get("feature_names", ())),
            label_names=tuple(data.get("label_names", ())),
            horizons=tuple(data.get("horizons", ())),
            dtype=data.get("dtype", "float64"),
            version=data.get("version", ""),
            metadata=data.get("metadata", {}),
        )'''

if old in content:
    content = content.replace(old, new)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("contract.py corrigé")
else:
    print("ERREUR: from_dict non trouvé")
