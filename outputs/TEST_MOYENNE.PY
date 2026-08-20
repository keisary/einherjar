import pandas as pd

fichier = r"D:\midas_v2\einherjar\outputs\archive.jsonl"

df = pd.read_json(fichier, lines=True)

metrics = pd.json_normalize(df["metrics"])

print(metrics.columns)
print("Moyenne n_trades :", metrics["n_trades"].mean())
print("Moyenne win_rate :", metrics["win_rate"].mean())
print("Moyenne total_return :", metrics["total_return"].mean())
print("Moyenne sharpe_ratio :", metrics["sharpe_ratio"].mean())
print("Moyenne profit_factor :", metrics["profit_factor"].mean())