import pandas as pd
fichier = r"D:\midas_v2\einherjar\outputs\corpus.jsonl"
df = pd.read_json(fichier, lines=True)

moyenne = df["win_rate"].mean()

print(f"Moyenne : {moyenne}")