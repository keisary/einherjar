"""
Analyseur de resultats de calibration — selection du corpus final.

Filtre les Einhers par seuils configurables de metriques.
Exporte le corpus calibre (corpus_v2.json).
"""

import json
from pathlib import Path
from typing import Optional

DEFAULT_THRESHOLDS = {
    "min_sharpe": 0.5,
    "min_winrate": 0.45,
    "min_profit_factor": 1.1,
    "max_drawdown": -0.25,
    "min_trades": 10,
    "min_trades_per_month": 0.5,
    "score_weights": {
        "sharpe_ratio": 0.30,
        "win_rate": 0.25,
        "profit_factor": 0.20,
        "expectancy": 0.15,
        "max_drawdown": 0.10,
    },
}


def compute_score(row: dict, weights: dict) -> float:
    """Calcule un score composite pondere."""
    score = 0.0
    for key, w in weights.items():
        val = row.get(key, 0.0)
        if key == "max_drawdown":
            # Drawdown est negatif, on veut le moins negatif possible
            val = max(0.0, 1.0 + val)  # ex: -0.15 -> 0.85
        score += val * w
    return round(score, 4)


def select_top(
    results_path: Path,
    thresholds: Optional[dict] = None,
    top_n: Optional[int] = None,
    corpus_brut_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> list[dict]:
    """
    Filtre les resultats de calibration selon des seuils.

    Args:
        results_path: chemin vers calibration_results.json
        thresholds: dict avec min_sharpe, min_winrate, etc.
        top_n: si precise, garde uniquement les top_n meilleurs scores
        corpus_brut_path: chemin vers corpus_brut_v1.json pour enrichir les metadonnees
        output_path: chemin de sortie pour le corpus calibre

    Returns:
        Liste des Einhers selectionnes avec leurs metriques.
    """
    cfg = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    weights = cfg.get("score_weights", DEFAULT_THRESHOLDS["score_weights"])

    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    # Chargement corpus brut pour recuperer les definitions completes
    corpus_map = {}
    if corpus_brut_path and corpus_brut_path.exists():
        with open(corpus_brut_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
            corpus_list = raw.get("einhers", raw)
            for e in corpus_list:
                corpus_map[e.get("name", e.get("einher_id", ""))] = e

    filtered = []
    for row in results:
        if row.get("n_trades", 0) < cfg["min_trades"]:
            continue
        if row.get("sharpe_ratio", 0.0) < cfg["min_sharpe"]:
            continue
        if row.get("win_rate", 0.0) < cfg["min_winrate"]:
            continue
        if row.get("profit_factor", 0.0) < cfg["min_profit_factor"]:
            continue
        if row.get("max_drawdown", 0.0) < cfg["max_drawdown"]:
            continue
        if row.get("trades_per_month", 0.0) < cfg["min_trades_per_month"]:
            continue

        row["score"] = compute_score(row, weights)
        # Enrichir avec la definition complete du corpus brut
        einher_name = row.get("einher_name", "")
        if einher_name in corpus_map:
            row["definition"] = corpus_map[einher_name]
        filtered.append(row)

    # Tri par score decroissant
    filtered.sort(key=lambda x: x["score"], reverse=True)

    if top_n is not None:
        filtered = filtered[:top_n]

    if output_path:
        output = {
            "_comment": "Corpus calibre EINHERJAR — selection post-backtest",
            "meta": {
                "total_tested": len(results),
                "total_selected": len(filtered),
                "thresholds": cfg,
            },
            "einhers": filtered,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"[Analyzer] {len(filtered)} Einhers selectionnes -> {output_path}")

    return filtered


def print_summary(results: list):
    """Affiche un resume rapide des resultats selectionnes."""
    if not results:
        print("Aucun Einher ne passe les filtres.")
        return
    print(f"\n=== Resume selection ({len(results)} Einhers) ===")
    print(f"Sharpe moyen : {sum(r['sharpe_ratio'] for r in results)/len(results):.3f}")
    print(f"Win rate moyen : {sum(r['win_rate'] for r in results)/len(results):.3f}")
    print(f"Profit factor moyen : {sum(r['profit_factor'] for r in results)/len(results):.3f}")
    print(f"Drawdown max moyen : {sum(r['max_drawdown'] for r in results)/len(results):.3f}")
    print(f"\nTop 5 par score :")
    for r in results[:5]:
        print(f"  {r['einher_name']:50s} | Sharpe {r['sharpe_ratio']:.2f} | WR {r['win_rate']:.2f} | PF {r['profit_factor']:.2f} | Score {r.get('score', 0):.3f}")
