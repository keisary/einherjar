import pandas as pd

fichier = r"D:\midas_v2\einherjar\outputs\corpus.jsonl"

df = pd.read_json(fichier, lines=True)

# Aplatit tous les objets imbriqués
data = pd.json_normalize(df["metrics"])

#print(data.columns.tolist())
print("Moyenne n_trades :", data["n_trades"].mean())
print("Moyenne win_rate :", data["win_rate"].mean())
print("Moyenne total_return :", data["total_return"].mean())
print("Moyenne sharpe_ratio :", data["sharpe_ratio"].mean())
print("Moyenne profit_factor :", data["profit_factor"].mean())

print("max n_trades :", data["n_trades"].max())
print("max win_rate :", data["win_rate"].max())
print("max total_return :", data["total_return"].max())
print("max sharpe_ratio :", data["sharpe_ratio"].max())
print("max profit_factor :", data["profit_factor"].max())

print("min n_trades :", data["n_trades"].min())
print("min", data["win_rate"].min())
print("min total_return :", data["total_return"].min())
print("min sharpe_ratio :", data["sharpe_ratio"].min())
print("min profit_factor :", data["profit_factor"].min())