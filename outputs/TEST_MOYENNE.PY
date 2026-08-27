import pandas as pd

fichier = r"D:\midas_v2\einherjar\outputs\corpus_30v2.jsonl"

df = pd.read_json(fichier, lines=True)

# Aplatit tous les objets imbriqués
data = pd.json_normalize(df["metrics"])

#print(data.columns.tolist())
print("Moyenne n_trades :", data["n_trades"].mean())
print("Moyenne win_rate :", data["win_rate"].mean())
print("Moyenne total_return :", data["total_return"].mean())
print("Moyenne sharpe_ratio :", data["sharpe_ratio"].mean())
print("Moyenne profit_factor :", data["profit_factor"].mean())
